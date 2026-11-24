"""A tiny synthetic ERNIE 5.5 checkpoint converted end to end: plan, expert repack, metadata, drafter, pack."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_sections as S  # noqa: E402
import checkpoint_ernie55 as E  # noqa: E402
from model_file import ModelFile  # noqa: E402

H, LAT, FF, NE, NL = 256, 128, 128, 4, 3   # hidden, latent, expert ff, experts, layers (0 dense, 1 HCA, 2 MLA)
VOCAB = 512
E4M3_CODES = {0.0: 0x00, 1.0: 0x38, -1.0: 0xB8, 2.0: 0x40, -2.0: 0xC0, 0.5: 0x30, -0.5: 0xB0}


def bf16_bytes(x: np.ndarray) -> bytes:
    u = (x.astype(np.float32).view(np.uint32) + 0x8000) >> 16
    return u.astype("<u2").tobytes()


def f32_bytes(x: np.ndarray) -> bytes:
    return x.astype("<f4").tobytes()


def fp8_tensor(rng, rows, cols):
    """An FP8 E4M3 matrix (codes + one F32 scale per 128x128 block) from exact E4M3 values."""
    vals = rng.choice(list(E4M3_CODES), size=(rows, cols)).astype(np.float32)
    codes = np.zeros((rows, cols), np.uint8)
    for v, c in E4M3_CODES.items():
        codes[vals == v] = c
    scales = np.ones((rows // 128, cols // 128), np.float32)
    return codes.tobytes(), scales.tobytes(), vals


def mxfp4_tensor(rng, rows, cols):
    """An MXFP4 matrix as the checkpoint stores it: packed int8 pairs + UE8M0 scales per 32."""
    codes = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    scales = rng.integers(120, 130, size=(rows, cols // 32), dtype=np.uint8)
    return codes.tobytes(), scales.tobytes(), codes, scales


class TinyCheckpoint:
    """Writes a tiny ERNIE 5.5 checkpoint (config, shards, tokenizer files) into a directory."""

    def __init__(self, d: Path):
        rng = np.random.default_rng(7)
        self.tensors = {}
        self.expect_q8 = {}

        def add(name, data, dtype, shape):
            self.tensors[name] = (data, dtype, shape)

        def add_fp8(name, rows, cols):
            codes, scales, vals = fp8_tensor(rng, rows, cols)
            add(name, codes, "F8_E4M3", [rows, cols])
            base = name[:-len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else name + "_scale_inv"
            add(base, scales, "F32", [rows // 128, cols // 128])
            self.expect_q8[name] = vals

        def add_bf16(name, shape):
            x = rng.standard_normal(shape).astype(np.float32)
            add(name, bf16_bytes(x), "BF16", list(shape))
            return x

        def add_f32(name, shape):
            x = rng.standard_normal(shape).astype(np.float32)
            add(name, f32_bytes(x), "F32", list(shape))
            return x

        ratios = [128, 128, -2]
        for l in range(NL):
            pre = f"model.layers.{l}."
            add_bf16(pre + "input_layernorm.weight", (H,))
            add_bf16(pre + "post_attention_layernorm.weight", (H,))
            add_fp8(pre + "self_attn.gate_proj.weight", 128, 128)
            for kind in ("self_attention", "mlp"):
                add_f32(pre + f"{kind}_hyper_connection.mapping_proj.weight", (4 * H, 24))
                add_f32(pre + f"{kind}_hyper_connection.bias", (24,))
                for a in ("alpha_pre", "alpha_post", "alpha_res"):
                    add_f32(pre + f"{kind}_hyper_connection.{a}", (1,))
            add_bf16(pre + "self_attn.vha_postmix_U", (128, 16))
            add_bf16(pre + "self_attn.vha_postmix_V", (128, 16))
            if ratios[l] == 128:   # HCA
                add_fp8(pre + "self_attn.linear_q_down_proj.weight", 128, H)
                add_fp8(pre + "self_attn.linear_q_up_proj.weight", 256, 128)
                add_fp8(pre + "self_attn.linear_kv_proj.weight", 128, H)
                add_fp8(pre + "self_attn.linear_o_group_proj", 128, H)
                add_fp8(pre + "self_attn.o_proj.weight", H, 128)
                add_bf16(pre + "self_attn.q_layernorm.weight", (128,))
                add_bf16(pre + "self_attn.kv_layernorm.weight", (128,))
                add_f32(pre + "self_attn.core_attention.attn_sink", (64,))
                add_f32(pre + "self_attn.core_attention.compressor.ape", (128, 64))
                add_bf16(pre + "self_attn.core_attention.compressor.linear_wkv.weight", (128, H))
                add_bf16(pre + "self_attn.core_attention.compressor.linear_wgate.weight", (128, H))
                add_bf16(pre + "self_attn.core_attention.compressor.norm.weight", (128,))
            else:                  # MLA (DSA)
                add_fp8(pre + "self_attn.q_a_proj.weight", 128, H)
                add_fp8(pre + "self_attn.q_b_proj.weight", 256, 128)
                add_fp8(pre + "self_attn.kv_a_proj_with_mqa.weight", 128, H)
                add_fp8(pre + "self_attn.kv_b_proj.weight", 256, 128)
                add_fp8(pre + "self_attn.o_proj.weight", H, 256)
                add_bf16(pre + "self_attn.q_a_layernorm.weight", (128,))
                add_bf16(pre + "self_attn.kv_a_layernorm.weight", (128,))
                add_bf16(pre + "self_attn.core_attention.softmax_offset", (64,))
                add_fp8(pre + "self_attn.core_attention.indexer.wq_b.weight", 128, 128)
                add_bf16(pre + "self_attn.core_attention.indexer.wk.weight", (128, H))
                add_bf16(pre + "self_attn.core_attention.indexer.weights_proj.weight", (64, H))
                add_bf16(pre + "self_attn.core_attention.indexer.k_norm.weight", (128,))
                add_bf16(pre + "self_attn.core_attention.indexer.k_norm.bias", (128,))
            if l == 0:             # the dense layer
                add_fp8(pre + "mlp.gate_proj.weight", 128, H)
                add_fp8(pre + "mlp.up_proj.weight", 128, H)
                add_fp8(pre + "mlp.down_proj.weight", H, 128)
                continue
            add_bf16(pre + "block_sparse_moe.gate.weight", (NE, H))
            add_bf16(pre + "block_sparse_moe.gate.weight_1", (NE, H))
            add_f32(pre + "block_sparse_moe.e_score_correction_bias", (NE,))
            add_f32(pre + "block_sparse_moe.gate.routed_scaling_factor_param", (NE,))
            add_bf16(pre + "block_sparse_moe.latent_norm.weight", (LAT,))
            add_fp8(pre + "block_sparse_moe.fc1_latent_proj.weight", LAT, H)
            add_fp8(pre + "block_sparse_moe.fc2_latent_proj.weight", H, LAT)
            add_fp8(pre + "block_sparse_moe.shared_experts.w1.weight", FF, LAT)
            add_fp8(pre + "block_sparse_moe.shared_experts.w3.weight", FF, LAT)
            add_fp8(pre + "block_sparse_moe.shared_experts.w2.weight", LAT, FF)
            for e in range(NE):
                for r, (rows, cols) in (("w1", (FF, LAT)), ("w3", (FF, LAT)), ("w2", (LAT, FF))):
                    codes, scales, _, _ = mxfp4_tensor(rng, rows, cols)
                    p = pre + f"block_sparse_moe.experts.{e}.{r}."
                    add(p + "weight", codes, "I8", [rows, cols // 2])
                    add(p + "weight_scale_inv", scales, "F8_E8M0", [rows, cols // 32])
        add_bf16("model.embed_tokens.weight", (VOCAB, H))
        add_bf16("lm_head.weight", (VOCAB, H))
        add_bf16("model.norm.weight", (H,))
        add_f32("model.mhc_contract.hc_head_base", (4,))
        add_f32("model.mhc_contract.hc_head_scale", (1,))
        add_f32("model.mhc_contract.hc_head_fn", (4 * H, 4))
        add_bf16("model.lm_head.multimax_ranges", (4,))
        add_bf16("model.lm_head.multimax_ts", (4,))
        # the drafter: one MTP stage with two experts
        for e in range(2):
            codes, scales, _, _ = mxfp4_tensor(rng, FF, LAT)
            add(f"model.mtp.0.block_sparse_moe.experts.{e}.w1.weight", codes, "I8", [FF, LAT // 2])
            add(f"model.mtp.0.block_sparse_moe.experts.{e}.w1.weight_scale_inv", scales, "F8_E8M0", [FF, LAT // 32])
        add_bf16("model.mtp.markov_head.markov_w1.weight", (64, 64))

        # shards: everything in one, except the embed in a second
        names = sorted(self.tensors)
        half = [n for n in names if n != "model.embed_tokens.weight"]
        self._write_shard(d / "model-00001-of-00002.safetensors", half)
        self._write_shard(d / "model-00002-of-00002.safetensors", ["model.embed_tokens.weight"])
        (d / "config.json").write_text(json.dumps({
            "model_type": "ernie5_5_moe", "num_hidden_layers": NL, "hidden_size": H, "vocab_size": VOCAB,
            "max_position_embeddings": 4096, "num_attention_heads": 64, "num_key_value_heads": 1,
            "head_dim": 512, "v_head_dim": 512, "hybrid_mla_qk_nope_head_dim": 192,
            "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256, "q_lora_rank": 1024,
            "kv_lora_rank": 512, "sliding_window": 128, "o_groups": 8, "o_lora_rank": 1024,
            "rms_norm_eps": 1e-6, "compress_ratios": ratios, "compress_rope_theta": 160000.0,
            "index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048, "qk_rope_head_dim": 64,
            "rope_theta": 8000000, "rope_scaling": {"type": "yarn", "factor": 16,
                                                    "original_max_position_embeddings": 65536,
                                                    "beta_fast": 32, "beta_slow": 1},
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "first_k_dense_replace": 1,
            "n_routed_experts": NE, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "moe_intermediate_size": FF, "moe_latent_size": LAT, "norm_topk_prob": True,
            "routed_scaling_factor": 3.0, "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
            "dspark_block_size": 7, "dspark_markov_rank": 256, "dspark_target_layer_ids": [40, 41, 42],
            "dspark_noise_token_id": 1024, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        }), encoding="utf-8")
        (d / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "byte_fallback": True, "vocab": {"a": 3, "b": 4, "c": 5, "d": 6, "e": 7, "ab": 8},
                      "merges": ["a b"]},
            "added_tokens": [{"id": 0, "content": "<|ERNIE_PAD|>", "special": True},
                             {"id": 1, "content": "<|ERNIE_BEGIN|>", "special": True},
                             {"id": 2, "content": "<|ERNIE_END|>", "special": True}],
        }), encoding="utf-8")
        (d / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": True}), encoding="utf-8")

    def _write_shard(self, path: Path, names):
        header = {}
        off = 0
        for n in names:
            data, dtype, shape = self.tensors[n]
            header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
            off += len(data)
        h = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for n in names:
                f.write(self.tensors[n][0])


def q8_0_values(raw: bytes) -> np.ndarray:
    return S.q8_0_values(np.frombuffer(raw, dtype=np.uint8))


class Ernie55Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        TinyCheckpoint(self.d)
        import checkpoint as C
        self.ck = C.Checkpoint(str(self.d))
        self.addCleanup(self.tmp.cleanup)

    def test_expert_repack_is_bit_exact(self):
        experts = E.ErnieExperts(self.ck)
        self.assertEqual(experts.moe_layers, [1, 2])
        blob = 2 * FF * E.mxfp4_row_bytes(LAT) + LAT * E.mxfp4_row_bytes(FF) + 16
        self.assertEqual(experts.blob, blob)
        out = E._expert_job((experts.entries[0][:1], 1))
        wp, wo, _, wshape, _ = self.ck.t["model.layers.1.block_sparse_moe.experts.0.w1.weight"]
        sp, so, _, sshape, _ = self.ck.t["model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv"]
        codes = np.fromfile(wp, dtype=np.uint8, offset=wo, count=np.prod(wshape)).reshape(wshape)
        scales = np.fromfile(sp, dtype=np.uint8, offset=so, count=np.prod(sshape)).reshape(sshape)
        want = E.mxfp4_matrix(codes, scales)
        got = np.frombuffer(out, dtype=np.uint8, count=len(want))
        np.testing.assert_array_equal(got, want)
        tail = np.frombuffer(out, dtype="<f4", count=4, offset=len(out) - 16)
        np.testing.assert_array_equal(tail, [1.0, 1.0, 1.0, 0.0])

    def test_mxfp4_layout(self):
        codes = np.full((2, 16), 0x21, np.uint8)               # 2 rows of 32 values, v[even] = 1, v[odd] = 2
        scales = np.array([[127], [130]], np.uint8)
        rows = E.mxfp4_matrix(codes, scales)
        self.assertEqual(rows.shape, (2 * 17,))
        self.assertEqual(rows[0], 127)
        self.assertEqual(rows[17], 130)
        # a ggml MXFP4 block packs value j in byte j's low nibble and value j + 16 in its high nibble
        want = [(1 | (1 << 4)) if j % 2 == 0 else (2 | (2 << 4)) for j in range(16)]
        self.assertEqual(list(rows[1:17]), want)

    def test_plan_coverage_and_forms(self):
        plan = E.Plan(self.ck)
        by_name = plan.tensors
        self.assertIn("token_embd.weight", by_name)
        self.assertIn("output.weight", by_name)
        self.assertIn("blk.0.ffn_gate.weight", by_name)
        self.assertIn("blk.1.attn_q_a.weight", by_name)
        self.assertIn("blk.2.attn_kv_a_mqa.weight", by_name)
        self.assertIn("blk.1.attn_vha.weight", by_name)
        self.assertIn("blk.1.hc_attn_scale.weight", by_name)
        self.assertIn("blk.1.ffn_gate_inp.weight", by_name)
        self.assertEqual(by_name["blk.1.ffn_gate_inp.weight"].ne, [H, 2 * NE])
        self.assertEqual(by_name["blk.1.hc_attn_scale.weight"].ne, [3])
        self.assertEqual(by_name["blk.1.hc_attn_fn.weight"].ne, [4 * H, 24])
        self.assertEqual(by_name["blk.2.attn_kv_b.weight"].type_name, "Q8_0")
        # no tensor is left unmapped
        covered = set()
        for t in by_name.values():
            covered.add(t.src)
        rest = set(self.ck.order) - covered
        rest = {k for k in rest if not (E.MTP_RE.match(k) or E.EXPERT_RE.match(k) or k in E.MULTIMAX
                                        or k.endswith("vha_postmix_U") or k.endswith("vha_postmix_V")
                                        or k.endswith("weight_scale_inv") or k.endswith("_scale_inv")
                                        or "alpha_" in k or "mapping_proj.weight" in k or "hyper_connection.bias" in k
                                        or k.endswith("gate.weight") or k.endswith("gate.weight_1"))}
        self.assertEqual(rest, set())

    def test_orphan_companion_rejected(self):
        self.ck.order.append("model.layers.1.unknown.weight_scale_inv")
        with self.assertRaisesRegex(S.ConvertError, "unknown.weight_scale_inv"):
            E.Plan(self.ck)

    def test_vha_fold_and_router_stack(self):
        plan = E.Plan(self.ck)
        u = self.ck.values("model.layers.1.self_attn.vha_postmix_U")
        v = self.ck.values("model.layers.1.self_attn.vha_postmix_V")
        want = np.eye(128, dtype=np.float32) + v @ u.T
        got = plan._values("blk.1.attn_vha.weight")
        np.testing.assert_allclose(got, want, rtol=1e-6)
        w0 = self.ck.values("model.layers.1.block_sparse_moe.gate.weight")
        w1 = self.ck.values("model.layers.1.block_sparse_moe.gate.weight_1")
        got = plan._values("blk.1.ffn_gate_inp.weight")
        np.testing.assert_array_equal(got, np.concatenate([w0, w1], axis=0))

    def test_fp8_q8_0_exact(self):
        plan = E.Plan(self.ck)
        import multiprocessing
        with multiprocessing.Pool(2) as pool:
            raw = b"".join(plan.chunks("blk.0.ffn_gate.weight", pool))
        got = q8_0_values(raw).reshape(128, H)
        # Q8_0 of exact E4M3 values times unit scales is near-exact
        np.testing.assert_allclose(got, self._fp8_want("model.layers.0.mlp.gate_proj.weight", (128, H)),
                                   atol=2.0 / 256, rtol=0.02)

    def _fp8_want(self, name, shape):
        wp, wo, _, _, _ = self.ck.t[name]
        codes = np.fromfile(wp, dtype=np.uint8, offset=wo, count=shape[0] * shape[1]).reshape(shape)
        return E.E4M3[codes]

    def test_pretokenizer_regressions(self):
        import tokenizer
        t = tokenizer.Tokenizer([], [], pre="ernie5")
        self.assertEqual(t._split("１２３４５６７"), ["１", "２３４", "５６７"])
        self.assertEqual(t._split("abc1234567def"), ["abc", "1", "234", "567", "def"])
        self.assertEqual(t._split("12345"), ["12", "345"])
        for n in (510, 511, 1020, 1021):
            pieces = t._split("1" * n)
            self.assertEqual("".join(pieces), "1" * n)
            self.assertLessEqual(max(map(len, pieces)), 3)

    def test_metadata(self):
        kv = E.model_kv(self.ck, "Tiny-ERNIE")
        A = "ernie55."
        self.assertEqual(kv["general.architecture"], "ernie55")
        self.assertEqual(kv[A + "block_count"], NL)
        self.assertEqual(kv[A + "attention.compress_ratios"], [128, 128, 0])
        self.assertEqual(kv[A + "expert_count"], NE)
        self.assertEqual(kv[A + "expert_latent_length"], LAT)
        self.assertEqual(kv[A + "dspark.block_size"], 7)
        self.assertEqual(len(kv[A + "logit_multimax"]), 8)
        tk = E.tokenizer_kv(self.ck)
        self.assertEqual(tk["tokenizer.ggml.pre"], "ernie5")
        self.assertEqual(tk["tokenizer.ggml.bos_token_id"], 1)
        self.assertEqual(len(tk["tokenizer.ggml.tokens"]), VOCAB)

    def test_build_pack(self):
        import convert
        import multiprocessing

        class A:
            checkpoint = str(self.d)
            out = str(self.d / "tiny.basalt")
            sections_from = None
            name = "Tiny-ERNIE"
            force = True
            workers = 2

        # Exercise the actual CLI dispatch, not just the architecture helper.
        import subprocess
        subprocess.run([sys.executable, str(Path(convert.__file__)), "--checkpoint", A.checkpoint,
                        "--out", A.out, "--workers", "2", "--name", A.name], check=True, capture_output=True)
        mf = ModelFile(str(self.d / "tiny.basalt"))
        names = set(mf.names())
        for want in ("gguf", "experts", "experts.layout", "dspark/raw", "dspark/manifest.json", "meta",
                     "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer.json"):
            self.assertIn(want, names)
        meta = json.loads(mf.read("meta"))
        self.assertEqual(meta["arch"], "ernie55")
        self.assertEqual(meta["gguf_kv"]["general.architecture"], "ernie55")
        layout = mf.read("experts.layout").decode()
        self.assertIn("n_expert %d" % NE, layout)
        lines = [x for x in layout.splitlines() if x and not x.startswith("#")]
        self.assertEqual([int(x.split()[0]) for x in lines], [1, 2])
        manifest = json.loads(mf.read("dspark/manifest.json"))
        self.assertIn("model.mtp.markov_head.markov_w1.weight", manifest)
        raw = mf.read("dspark/raw")
        self.assertEqual(len(raw), sum(v["bytes"] for v in manifest.values()))
        cfg = json.loads(mf.read("tokenizer/tokenizer.json"))
        self.assertEqual(cfg["pre"], "ernie5")
        self.assertEqual(len(cfg["pre_patterns"]), 5)


if __name__ == "__main__":
    unittest.main()
