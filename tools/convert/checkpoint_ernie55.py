"""An ERNIE 5.5 Flash Preview pack from Baidu's safetensors checkpoint (baidu/ERNIE-5.5-Flash-Preview), for
tools/convert/convert.py --checkpoint DIR --out MODEL.basalt (no --sections-from; everything comes from DIR).

  experts / experts.layout  each routed expert's MXFP4 w1 (gate) | w3 (up) | w2 (down) as ggml MXFP4 rows (E2M1 codes
                            and UE8M0 scales per 32 values, unchanged from the checkpoint's packed-int8 pairs and
                            E8M0 bytes) with the FP32 tail {1, 1, 1, 0} (the checkpoint has no per-matrix scales)
  gguf                      every other language-model tensor under Basalt's tensor names: the FP8 E4M3 projections
                            (F32 scales per 128 x 128 block) as Q8_0 of their exact values, the token embedding and
                            output head Q8_0 from BF16; norms, routers, hyper-connections, sinks and the folded VHA
                            mix widened to F32; the compressor and indexer projections kept BF16
  dspark/raw + dspark/manifest.json
                            the Preview's DSpark drafter (model.mtp.*, the markov and confidence heads, the feature
                            projection) as raw safetensors bytes with a JSON manifest; the drafter's own converter
                            stage folds it into the form the engine reads
  gguf_kv, tokenizer        the model metadata and tokenizer from config.json, generation_config.json, README.md and
                            the tokenizer files

The tensor math (the VHA fold m = I + V U^T, the stacked routers, the hyper-connection transforms, the absorbed MLA
dims) follows the checkpoint's config.json and llama.cpp's draft ERNIE 5.5 conversion (PR #31925, unmerged); the
Preview's three-stage MTP block is kept raw because that draft only covers the Base model's single NextN layer.

The checkpoint's shards are read in file order. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pathlib
import re
import tempfile
import time

import numpy as np

import model_sections as S
from checkpoint import Checkpoint, bounded_imap, front_matter
from checkpoint_deepseek41 import E4M3
from model_file import TENSOR_ALIGN, ModelWriter, Stream

ARCH = "ernie55"
MXFP4 = 39
TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30}
UNUSED, NORMAL, CONTROL, USER_DEFINED = 5, 1, 3, 4
ROLES = ("w1", "w3", "w2")           # gate, up, down
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(weight|weight_scale_inv)$")
MTP_RE = re.compile(r"^model\.mtp\.")
# Tensor names consumed as companions of a table entry (FP8 block scales, hyper-connection parts, the second router)
_COMPANION_RE = re.compile(
    r"(?:_scale_inv$|_hyper_connection\.(?:alpha_pre|alpha_post|alpha_res|bias|mapping_proj\.weight)$"
    r"|^block_sparse_moe\.gate\.weight(?:_1)?$)")
VHA_PAIR = ("self_attn.vha_postmix_U", "self_attn.vha_postmix_V")


# ------------------------------------------------------------------ routed experts
def mxfp4_matrix(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """A checkpoint MXFP4 matrix (codes [rows, cols/2]: value 2i in the low nibble of byte i, 2i + 1 in the high one;
    UE8M0 scales [rows, cols/32]) as ggml MXFP4 rows: per 32 values the scale byte, then 16 bytes whose low nibble is
    value j and high nibble value j + 16."""
    rows, cols = codes.shape[0], codes.shape[1] * 2
    v = np.empty((rows, cols), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, cols // 32, 2, 16)
    out = np.empty((rows, cols // 32, 17), np.uint8)
    out[..., 0] = scales
    out[..., 1:] = v[:, :, 0, :] | (v[:, :, 1, :] << 4)
    return out.reshape(-1)


def mxfp4_row_bytes(n: int) -> int:
    """Bytes of one MXFP4 row of n values (blocks of 32 in 17 bytes)."""
    if n % 32:
        raise S.ConvertError(f"an MXFP4 row of {n} values is not whole blocks")
    return n // 32 * 17


def _expert_job(args):
    """Blobs of experts [s, s + n) of one layer: [gate | up | down] MXFP4 rows and the FP32 tail {1, 1, 1, 0}."""
    entries, n = args
    out = []
    for e in range(n):
        parts = []
        for (wp, wo, wshape), (sp, so, sshape) in entries[e]:
            w = np.fromfile(wp, dtype=np.uint8, count=wshape[0] * wshape[1], offset=wo).reshape(wshape)
            sc = np.fromfile(sp, dtype=np.uint8, count=sshape[0] * sshape[1], offset=so).reshape(sshape)
            if (sc == 0xFF).any():
                return f"a NaN UE8M0 scale at {wp.name}+{wo}"
            parts.append(mxfp4_matrix(w, sc))
        parts.append(np.array([1.0, 1.0, 1.0, 0.0], dtype="<f4").tobytes())
        out.append(b"".join(parts))
    return b"".join(out)


class ErnieExperts:
    """The `experts` section from the checkpoint's per-expert MXFP4 tensors (layers 1..n-1; layer 0 is dense)."""

    def __init__(self, ck: Checkpoint):
        tc = ck.config
        self.ck = ck
        self.n_layers = int(tc["num_hidden_layers"])
        self.first_dense = int(tc.get("first_k_dense_replace", 1))
        self.n_expert = int(tc["n_routed_experts"])
        lat, ff = int(tc["moe_latent_size"]), int(tc["moe_intermediate_size"])
        self.shapes = {"w1": (ff, lat), "w3": (ff, lat), "w2": (lat, ff)}
        self.blob = 2 * ff * mxfp4_row_bytes(lat) + lat * mxfp4_row_bytes(ff) + 16
        self.moe_layers = list(range(self.first_dense, self.n_layers))
        known = {f"{r}.{k}" for r in ROLES for k in ("weight", "weight_scale_inv")}
        for k in ck.t:
            m = EXPERT_RE.match(k)
            if m and (int(m.group(2)) >= self.n_expert or f"{m.group(3)}.{m.group(4)}" not in known
                      or int(m.group(1)) not in self.moe_layers):
                raise S.ConvertError(f"unexpected expert tensor {k}")
        self.entries = []
        for l in self.moe_layers:
            layer = []
            for e in range(self.n_expert):
                ex = []
                for r in ROLES:
                    p = f"model.layers.{l}.block_sparse_moe.experts.{e}.{r}."
                    rows, cols = self.shapes[r]
                    ck.need(p + "weight", "I8", (rows, cols // 2))
                    ck.need(p + "weight_scale_inv", "F8_E8M0", (rows, cols // 32))
                    w, s = ck.t[p + "weight"], ck.t[p + "weight_scale_inv"]
                    ex.append(((w[0], w[1], w[3]), (s[0], s[1], s[3])))
                layer.append(ex)
            self.entries.append(layer)
        self.layout = [(l, MXFP4, MXFP4, i * self.blob * self.n_expert, self.blob)
                       for i, l in enumerate(self.moe_layers)]
        self.total = len(self.moe_layers) * self.n_expert * self.blob

    def layout_text(self) -> bytes:
        head = ("# experts v6: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); "
                "an MXFP4 blob is gate | up | down ggml MXFP4 rows ending in FP32 {1, 1, 1, 0}\n"
                % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % x for x in self.layout)).encode("utf-8")

    def jobs(self, group: int):
        for i in range(len(self.moe_layers)):
            for s in range(0, self.n_expert, group):
                n = min(group, self.n_expert - s)
                yield self.entries[i][s:s + n], n

    def chunks(self, pool, group: int = 32, depth: int = 32):
        for b in bounded_imap(pool, _expert_job, self.jobs(group), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


# ------------------------------------------------------------------ every other language-model tensor
# checkpoint tensor (without `model.layers.N.`) -> (pack name without `blk.N.`, form):
#   Q8_0  FP8/BF16 projection quantized once (ggml reference rounding)
#   F32   widened exactly       BF16  as stored
HCA = {
    "self_attn.linear_q_down_proj.weight": ("attn_q_a.weight", "Q8_0"),
    "self_attn.linear_q_up_proj.weight": ("attn_q_b.weight", "Q8_0"),
    "self_attn.linear_kv_proj.weight": ("attn_kv.weight", "Q8_0"),
    "self_attn.linear_o_group_proj": ("attn_output_a.weight", "Q8_0"),
    "self_attn.o_proj.weight": ("attn_output_b.weight", "Q8_0"),
    "self_attn.core_attention.attn_sink": ("attn_sinks.weight", "F32"),
    "self_attn.core_attention.compressor.ape": ("attn_compressor_ape.weight", "F32"),
    "self_attn.core_attention.compressor.linear_wkv.weight": ("attn_compressor_kv.weight", "BF16"),
    "self_attn.core_attention.compressor.linear_wgate.weight": ("attn_compressor_gate.weight", "BF16"),
}
MLA = {
    "self_attn.q_a_proj.weight": ("attn_q_a.weight", "Q8_0"),
    "self_attn.q_b_proj.weight": ("attn_q_b.weight", "Q8_0"),
    "self_attn.kv_a_proj_with_mqa.weight": ("attn_kv_a_mqa.weight", "Q8_0"),
    "self_attn.kv_b_proj.weight": ("attn_kv_b.weight", "Q8_0"),
    "self_attn.o_proj.weight": ("attn_output.weight", "Q8_0"),
    "self_attn.core_attention.softmax_offset": ("attn_sinks.weight", "F32"),
    "self_attn.core_attention.indexer.wq_b.weight": ("indexer.attn_q_b.weight", "Q8_0"),
    "self_attn.core_attention.indexer.wk.weight": ("indexer.attn_k.weight", "BF16"),
    "self_attn.core_attention.indexer.weights_proj.weight": ("indexer.proj.weight", "BF16"),
}
BOTH = {
    "self_attn.gate_proj.weight": ("attn_gate.weight", "Q8_0"),
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("ffn_norm.weight", "F32"),
}
F32_FLOATS = {
    "self_attn.q_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.kv_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.core_attention.compressor.norm.weight": "attn_compressor_norm.weight",
    "self_attn.core_attention.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attn.core_attention.indexer.k_norm.bias": "indexer.k_norm.bias",
    "block_sparse_moe.e_score_correction_bias": "exp_probs_b.bias",
    "block_sparse_moe.gate.routed_scaling_factor_param": "exp_scale.weight",
    "block_sparse_moe.latent_norm.weight": "ffn_latent_norm.weight",
}
MOE_Q8 = {
    "block_sparse_moe.fc1_latent_proj.weight": "ffn_latent_down.weight",
    "block_sparse_moe.fc2_latent_proj.weight": "ffn_latent_up.weight",
    "block_sparse_moe.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "block_sparse_moe.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "block_sparse_moe.shared_experts.w2.weight": "ffn_down_shexp.weight",
}
DENSE_Q8 = {
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
GLOBAL_Q8 = {
    "model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
}
GLOBAL_F32 = {
    "model.norm.weight": "output_norm.weight",
    "model.mhc_contract.hc_head_base": "output_hc_base.weight",
    "model.mhc_contract.hc_head_scale": "output_hc_scale.weight",
}
MULTIMAX = ("model.lm_head.multimax_ranges", "model.lm_head.multimax_ts")
METHODS = {
    "fp8_q8_0": "E4M3 code x F32 scale per 128 x 128 block (exact in f32), then Q8_0 (ggml reference rounding)",
    "bf16_q8_0": "BF16 to Q8_0 (ggml reference rounding)",
    "bf16": "as stored", "bf16_f32": "BF16 widened to F32 (exact)", "f32": "as stored",
    "vha_fold": "m = I + V U^T in f32 (the VHA head mix, folded)",
    "hc_fn": "mapping_proj.weight transposed to [out, in], F32",
    "hc_scale": "alpha_pre, alpha_post, alpha_res stacked, F32",
    "router_stack": "gate.weight and gate.weight_1 stacked [2E, H], F32",
    "hc_head_fn": "hc_head_fn transposed to [out, in], F32",
}


def _scale_name(src: str) -> str:
    """The FP8 block scales of an FP8 tensor: <tensor>.weight_scale_inv beside <tensor>.weight (and the one
    `linear_o_group_proj` -> `linear_o_group_proj_scale_inv`)."""
    if src.endswith(".weight"):
        return src[: -len(".weight")] + ".weight_scale_inv"
    return src + "_scale_inv"


class Tensor:
    """One pack tensor made from a checkpoint tensor: `ne` is the pack shape (fastest dimension first)."""

    def __init__(self, name, type_name, src, how, ne):
        self.name, self.type_name, self.src, self.how, self.ne = name, type_name, src, how, ne
        self.type_id = TYPE_ID[type_name]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne)) if self.ne else 1
        return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34}[self.type_name]


class Plan:
    """Every language-model tensor but the routed experts and the drafter, with `chunks(name, pool)` the streamed
    bytes of a quantized tensor and `float_bytes(name)` the stored bytes of a float one."""

    def __init__(self, ck: Checkpoint):
        tc = ck.config
        self.ck = ck
        self.n_layers = int(tc["num_hidden_layers"])
        self.first_dense = int(tc.get("first_k_dense_replace", 1))
        ratios = [max(int(r), 0) for r in tc["compress_ratios"]]
        if len(ratios) != self.n_layers:
            raise S.ConvertError(f"config.json: compress_ratios has {len(ratios)} entries for {self.n_layers} layers")
        self.is_hca = [r != 0 for r in ratios]
        self.tensors: dict[str, Tensor] = {}
        unknown = []
        for k in ck.order:
            if MTP_RE.match(k) or EXPERT_RE.match(k) or k in MULTIMAX:
                continue   # the drafter, the experts section, or read for the metadata
            m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", k)
            if m:
                l, key = int(m.group(1)), m.group(2)
                if l >= self.n_layers:
                    raise S.ConvertError(f"{k}: the config has {self.n_layers} layers")
                if key in VHA_PAIR or _COMPANION_RE.search(key):
                    continue   # folded into blk.l.attn_vha.weight or consumed with a table entry
                if self._layer_tensor(l, key, k) is None:
                    unknown.append(k)
                continue
            if k in GLOBAL_Q8:
                self._add_q8(GLOBAL_Q8[k], k, how="bf16_q8_0")
            elif k in GLOBAL_F32:
                self._add_f32(GLOBAL_F32[k], k)
            elif k == "model.mhc_contract.hc_head_fn":
                _, _, dt, shape, _ = self.ck.t[k]
                if dt != "F32":
                    raise S.ConvertError(f"{k} is {dt}; expected F32")
                self._add("output_hc_fn.weight", "F32", k, "hc_head_fn", list(shape))
            else:
                unknown.append(k)
        for l in range(self.n_layers):
            pre = f"model.layers.{l}."
            if pre + VHA_PAIR[0] in ck.t:
                self._add_vha(l)
            for kind, short in (("self_attention", "attn"), ("mlp", "ffn")):
                base = f"{pre}{kind}_hyper_connection."
                self._add(f"blk.{l}.hc_{short}_fn.weight", "F32", base + "mapping_proj.weight", "hc_fn",
                          list(ck.t[base + "mapping_proj.weight"][3]))
                self._add(f"blk.{l}.hc_{short}_base.weight", "F32", base + "bias", "f32", None)
                self._add(f"blk.{l}.hc_{short}_scale.weight", "F32", base, "hc_scale", [3])
            if l >= self.first_dense:
                self._add_router(l)
        # Every skipped companion must belong to a tensor actually emitted. Do not silently drop a new scale
        # or hyper-connection leaf just because its suffix looks familiar.
        consumed = set()
        for t in self.tensors.values():
            if t.how == "fp8_q8_0":
                consumed.add(_scale_name(t.src))
            elif t.how == "hc_scale":
                consumed.update(t.src + "alpha_" + q for q in ("pre", "post", "res"))
            elif t.how == "router_stack":
                consumed.update((t.src + "weight", t.src + "weight_1"))
            else:
                consumed.add(t.src)
        for k in ck.order:
            m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", k)
            if m and not EXPERT_RE.match(k) and _COMPANION_RE.search(m.group(2)) and k not in consumed:
                unknown.append(k)
        if unknown:
            raise S.ConvertError("checkpoint tensors with no pack counterpart: " + ", ".join(unknown[:8])
                                 + (f" ... and {len(unknown) - 8} more" if len(unknown) > 8 else ""))

    # -- table lookups
    def _layer_tensor(self, l: int, key: str, src: str):
        if key in BOTH:
            dst, form = BOTH[key]
        elif key in F32_FLOATS:
            dst, form = F32_FLOATS[key], "F32"
        elif key in (HCA if self.is_hca[l] else MLA):
            dst, form = (HCA if self.is_hca[l] else MLA)[key]
        elif l >= self.first_dense and key in MOE_Q8:
            dst, form = MOE_Q8[key], "Q8_0"
        elif l < self.first_dense and key in DENSE_Q8:
            dst, form = DENSE_Q8[key], "Q8_0"
        else:
            return None
        if form == "Q8_0":
            self._add_q8(f"blk.{l}.{dst}", src)
        elif form == "F32":
            self._add_f32(f"blk.{l}.{dst}", src)
        else:
            self._add_bf16(f"blk.{l}.{dst}", src)
        return dst

    # -- builders
    def _add(self, name, ty, src, how, ne):
        if name in self.tensors:
            raise S.ConvertError(f"two checkpoint tensors map to {name}")
        if ne is None:
            _, _, dt, shape, _ = self.ck.t[src]
            ne = list(reversed(shape)) if shape else [1]
        self.tensors[name] = Tensor(name, ty, src, how, ne)

    def _add_q8(self, name, src, how="fp8_q8_0"):
        _, _, dt, shape, _ = self.ck.t[src]
        if len(shape) != 2 or shape[1] % 32:
            raise S.ConvertError(f"{src}: rows of {shape[1] if len(shape) == 2 else '?'} values cannot be Q8_0")
        if how == "fp8_q8_0":
            if dt != "F8_E4M3":
                raise S.ConvertError(f"{src} is {dt}; expected F8_E4M3")
            sc = _scale_name(src)
            if sc not in self.ck.t:
                raise S.ConvertError(f"the checkpoint has no scale tensor for {src}")
            _, _, sdt, sshape, _ = self.ck.t[sc]
            if sdt != "F32" or len(sshape) != 2 or [s * 128 for s in sshape] != list(shape):
                raise S.ConvertError(f"{sc} is {sdt} {sshape}; expected F32 "
                                     f"{[s // 128 for s in shape]} blocks of 128 x 128")
        elif dt != "BF16":
            raise S.ConvertError(f"{src} is {dt}; expected BF16")
        self._add(name, "Q8_0", src, how, [shape[1], shape[0]])

    def _add_f32(self, name, src):
        _, _, dt, shape, _ = self.ck.t[src]
        if dt not in ("F32", "BF16"):
            raise S.ConvertError(f"{src} is {dt}; expected BF16 or F32")
        self._add(name, "F32", src, "f32" if dt == "F32" else "bf16_f32",
                  list(reversed(shape)) if shape else [1])

    def _add_bf16(self, name, src):
        _, _, dt, shape, _ = self.ck.t[src]
        if dt != "BF16":
            raise S.ConvertError(f"{src} is {dt}; expected BF16")
        self._add(name, "BF16", src, "bf16", list(reversed(shape)))

    def _add_vha(self, l: int):
        pre = f"model.layers.{l}."
        u, v = self.ck.t[pre + VHA_PAIR[0]], self.ck.t[pre + VHA_PAIR[1]]
        if u[2] != "BF16" or v[2] != "BF16" or u[3][0] != v[3][0] or u[3][1] != v[3][1]:
            raise S.ConvertError(f"{pre}vha_postmix_{{U,V}}: {u[2]} {u[3]} / {v[2]} {v[3]}")
        self.tensors[f"blk.{l}.attn_vha.weight"] = Tensor(f"blk.{l}.attn_vha.weight", "F32", pre, "vha_fold",
                                                          [u[3][0], u[3][0]])

    def _add_router(self, l: int):
        pre = f"model.layers.{l}.block_sparse_moe.gate."
        w0, w1 = self.ck.t[pre + "weight"], self.ck.t[pre + "weight_1"]
        if w0[2] != "BF16" or w1[2] != "BF16" or w0[3] != w1[3]:
            raise S.ConvertError(f"{pre}weight{{,_1}}: {w0[2]} {w0[3]} / {w1[2]} {w1[3]}")
        self.tensors[f"blk.{l}.ffn_gate_inp.weight"] = Tensor(f"blk.{l}.ffn_gate_inp.weight", "F32", pre,
                                                              "router_stack", [w0[3][1], 2 * w0[3][0]])

    # -- values
    def _values(self, name) -> np.ndarray:
        """A float tensor's values in the pack order, as float32."""
        t = self.tensors[name]
        if t.how == "vha_fold":
            u = self.ck.values(t.src + VHA_PAIR[0])
            v = self.ck.values(t.src + VHA_PAIR[1])
            n = u.shape[0]
            return (np.eye(n, dtype=np.float32) + v @ u.T).astype(np.float32)
        if t.how == "hc_scale":
            parts = [self.ck.values(t.src + f"alpha_{s}") for s in ("pre", "post", "res")]
            return np.concatenate([p.reshape(1) for p in parts]).astype(np.float32)
        if t.how == "router_stack":
            return np.concatenate([self.ck.values(t.src + "weight"), self.ck.values(t.src + "weight_1")],
                                  axis=0).astype(np.float32)
        x = self.ck.values(t.src)
        if t.how in ("hc_fn", "hc_head_fn"):
            x = x.T
        return np.ascontiguousarray(x, dtype=np.float32)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self._values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    def jobs(self, name):
        """Row ranges of a Q8_0 tensor's source for _quant_job, whole 128-row blocks for FP8."""
        t = self.tensors[name]
        p, off, dt, shape, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        scale = None
        if t.how == "fp8_q8_0":
            scale = self.ck.t[_scale_name(t.src)]
        step = max(128, ((1 << 22) // ne0) & ~127)
        for r in range(0, ne1, step):
            yield (str(p), off, scale, ne0, ne1, r, min(step, ne1 - r), t.how)

    def chunks(self, name, pool, depth: int = 64):
        for b in bounded_imap(pool, _quant_job, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _quant_job(args):
    """Rows [row0, row0 + rows) of a projection (FP8 with 128 x 128 F32 block scales, or BF16) as Q8_0."""
    path, off, scale, ne0, ne1, row0, rows, how = args
    if how == "bf16_q8_0":
        u = np.memmap(path, dtype="<u2", mode="r", offset=off + 2 * (row0 * ne0), shape=(rows, ne0))
        x = S._bf16_to_f32(np.ascontiguousarray(u)).reshape(-1)
    else:
        codes = np.memmap(path, dtype=np.uint8, mode="r", offset=off + row0 * ne0, shape=(rows, ne0))
        sp, soff, _, _, _ = scale
        nb0 = ne0 // 128
        br0, br1 = row0 // 128, (row0 + rows + 127) // 128
        sc = np.memmap(sp, dtype="<f4", mode="r", offset=soff + 4 * (br0 * nb0), shape=(br1 - br0, nb0))
        if np.isnan(sc).any():
            return "a NaN FP8 block scale"
        v = E4M3[codes]
        if np.isnan(v).any():
            return "a NaN E4M3 weight"
        s = np.repeat(np.repeat(sc, 128, 0), 128, 1)
        skip = row0 - br0 * 128
        x = (v * s[skip:skip + rows]).astype(np.float32).reshape(-1)
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q8_0(x)


# ------------------------------------------------------------------ the drafter, kept raw
def _drafter_job(args):
    p, off, n = args
    return np.fromfile(p, dtype=np.uint8, count=n, offset=off).tobytes()


class Drafter:
    """The Preview's DSpark drafter (model.mtp.*): raw safetensors bytes in the checkpoint's file order, with a
    JSON manifest (name -> dtype, shape, offset, bytes). Folded by the drafter's own converter stage, not here."""

    def __init__(self, ck: Checkpoint):
        self.names = [k for k in ck.order if MTP_RE.match(k)]
        if not self.names:
            raise S.ConvertError("the checkpoint has no model.mtp.* tensors (the DSpark drafter)")
        self.ck = ck
        off, self.manifest = 0, {}
        for k in self.names:
            _, _, dt, shape, n = ck.t[k]
            self.manifest[k] = {"dtype": dt, "shape": shape, "offset": off, "bytes": n}
            off += n
        self.total = off

    def chunks(self, pool, depth: int = 16):
        jobs = ((str(self.ck.t[k][0]), self.ck.t[k][1], self.ck.t[k][4]) for k in self.names)
        yield from bounded_imap(pool, _drafter_job, jobs, depth)


# ------------------------------------------------------------------ metadata and tokenizer
def tokenizer_kv(ck: Checkpoint) -> dict:
    """The tokenizer as GGUF metadata: ERNIE's byte-level BPE with the `ernie5` pre-tokenizer sequence
    (tools/tokenizer.py's PRE_PATTERNS)."""
    d = ck.dir
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8")) \
        if (d / "tokenizer_config.json").is_file() else {}
    model = tj.get("model", {})
    if model.get("type") != "BPE" or not model.get("byte_fallback"):
        raise S.ConvertError("tokenizer.json: expected a byte-level BPE model with byte_fallback")
    n_vocab = int(ck.config["vocab_size"])
    rev = {int(i): t for t, i in model["vocab"].items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    for i, a in added.items():
        rev[i] = a["content"]
    if max(rev) >= n_vocab:
        raise S.ConvertError(f"tokenizer.json: id {max(rev)} is beyond the vocabulary of {n_vocab}")
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(UNUSED)
            continue
        t = rev[i]
        tokens.append(t)
        if i in added:
            special = added[i].get("special") or (t.startswith("<|") and t.endswith("|>"))
            types.append(CONTROL if special else USER_DEFINED)
        else:
            types.append(NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "ernie5", "tokenizer.ggml.tokens": tokens,
          "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges}
    for key, cfg_key in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("padding", "pad_token_id")):
        v = ck.config.get(cfg_key, tcfg.get(cfg_key.removesuffix("_id")))
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str):
            v = model["vocab"].get(v)
        if v is not None:
            kv[f"tokenizer.ggml.{key}_token_id"] = int(v)
    kv["tokenizer.ggml.add_bos_token"] = bool(tcfg.get("add_bos_token", False))
    template = tcfg.get("chat_template")
    if template:
        kv["tokenizer.chat_template"] = template
    return kv


def model_kv(ck: Checkpoint, name: str | None) -> dict:
    """The GGUF metadata the engine reads, from config.json, generation_config.json and the model card."""
    c = ck.config
    if c.get("model_type") != "ernie5_5_moe":
        raise S.ConvertError(f"config.json: model_type {c.get('model_type')}, expected ernie5_5_moe")
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) \
        if (d / "generation_config.json").is_file() else {}
    card = front_matter(d / "README.md")
    A = ARCH + "."
    rope = c.get("rope_scaling", {})
    kv = {"general.architecture": ARCH, "general.type": "model"}
    if name:
        kv["general.name"] = name
    for src, dst in (("temperature", "general.sampling.temp"), ("top_p", "general.sampling.top_p"),
                     ("top_k", "general.sampling.top_k"), ("repetition_penalty", "general.sampling.penalty_repeat")):
        if src in gen and gen[src] is not None:
            kv[dst] = float(np.float32(gen[src])) if isinstance(gen[src], float) else int(gen[src])
    if "license" in card:
        kv["general.license"] = card["license"]
    if card.get("base_model"):
        bases = card["base_model"]
        bases = [bases] if isinstance(bases, str) else bases
        kv["general.base_model.count"] = len(bases)
        for i, b in enumerate(bases):
            kv[f"general.base_model.{i}.repo_url"] = f"https://huggingface.co/{b}"
    multimax = (ck.values("model.lm_head.multimax_ranges").tolist() + ck.values("model.lm_head.multimax_ts").tolist())
    kv.update({
        A + "block_count": int(c["num_hidden_layers"]),
        A + "context_length": int(c["max_position_embeddings"]),
        A + "embedding_length": int(c["hidden_size"]),
        A + "attention.head_count": int(c["num_attention_heads"]),
        A + "attention.head_count_kv": int(c["num_key_value_heads"]),
        A + "attention.key_length": int(c["head_dim"]),
        A + "attention.value_length": int(c["v_head_dim"]),
        A + "attention.key_length_mla": int(c["hybrid_mla_qk_nope_head_dim"]) + int(c["hybrid_mla_qk_rope_head_dim"]),
        A + "attention.value_length_mla": int(c["hybrid_mla_v_head_dim"]),
        A + "attention.q_lora_rank": int(c["q_lora_rank"]),
        A + "attention.kv_lora_rank": int(c["kv_lora_rank"]),
        A + "attention.sliding_window": int(c["sliding_window"]),
        A + "attention.output_group_count": int(c["o_groups"]),
        A + "attention.output_lora_rank": int(c["o_lora_rank"]),
        A + "attention.layer_norm_rms_epsilon": float(c["rms_norm_eps"]),
        A + "attention.compress_ratios": [max(int(r), 0) for r in c["compress_ratios"]],
        A + "attention.compress_rope_freq_base": float(c["compress_rope_theta"]),
        A + "attention.indexer.head_count": int(c["index_n_heads"]),
        A + "attention.indexer.key_length": int(c["index_head_dim"]),
        A + "attention.indexer.top_k": int(c["index_topk"]),
        A + "rope.dimension_count": int(c["qk_rope_head_dim"]),
        A + "rope.freq_base": float(c["rope_theta"]),
        A + "rope.scaling.type": str(rope.get("type", "yarn")),
        A + "rope.scaling.factor": float(rope.get("factor", 1)),
        A + "rope.scaling.original_context_length": int(rope.get("original_max_position_embeddings",
                                                                 c["max_position_embeddings"])),
        A + "rope.scaling.beta_fast": float(rope.get("beta_fast", 32)),
        A + "rope.scaling.beta_slow": float(rope.get("beta_slow", 1)),
        A + "hyper_connection.count": int(c["hc_mult"]),
        A + "hyper_connection.sinkhorn_iterations": int(c["hc_sinkhorn_iters"]),
        A + "hyper_connection.epsilon": float(c["hc_eps"]),
        A + "leading_dense_block_count": int(c.get("first_k_dense_replace", 1)),
        A + "expert_count": int(c["n_routed_experts"]),
        A + "expert_used_count": int(c["num_experts_per_tok"]),
        A + "expert_shared_count": int(c["n_shared_experts"]),
        A + "expert_feed_forward_length": int(c["moe_intermediate_size"]),
        A + "expert_latent_length": int(c["moe_latent_size"]),
        A + "expert_weights_norm": bool(c["norm_topk_prob"]),
        A + "expert_weights_scale": float(c["routed_scaling_factor"]),
        A + "activation.situ_beta": float(c["activation_situ_beta"]),
        A + "activation.situ_linear_beta": float(c["activation_situ_linear_beta"]),
        A + "logit_multimax": [float(np.float32(v)) for v in multimax],
        A + "dspark.block_size": int(c["dspark_block_size"]),
        A + "dspark.markov_rank": int(c["dspark_markov_rank"]),
        A + "dspark.target_layers": [int(x) for x in c["dspark_target_layer_ids"]],
        A + "dspark.noise_token_id": int(c["dspark_noise_token_id"]),
        A + "vocab_size": int(c["vocab_size"]),
    })
    return kv


def build(a, Job, fingerprint, die) -> None:
    """convert.py --checkpoint DIR --out MODEL.basalt for an ERNIE 5.5 Flash Preview checkpoint."""
    def source_entry(p):
        return {"path": p.name, "size": p.stat().st_size,
                "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}

    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = Checkpoint(a.checkpoint)
        plan = Plan(ck)
        experts = ErnieExperts(ck)
        drafter = Drafter(ck)
        name = a.name or "ERNIE-5.5-Flash-Preview MXFP4"
        kv = model_kv(ck, name)
        kv.update(tokenizer_kv(ck))
    except S.ConvertError as e:
        die(str(e))
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp, multiprocessing.Pool(a.workers) as pool:
            import tokenizer
            tokenizer.extract_metadata(kv, pathlib.Path(tmp))
            print("writing sections:", flush=True)
            for p in sorted(q for q in (pathlib.Path(tmp) / "tokenizer").iterdir() if q.is_file()):
                job.file(f"tokenizer/{p.name}", p)
            t = time.time()
            parts, names = [], []
            for n, info in plan.tensors.items():
                if info.type_name == "Q8_0":
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            job.chunks("experts", experts.chunks(pool))
            job.bytes("experts.layout", experts.layout_text())
            job.chunks("dspark/raw", drafter.chunks(pool))
            job.bytes("dspark/manifest.json", json.dumps(drafter.manifest, separators=(",", ":")).encode("utf-8"))
            tens = {n: {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.ne],
                        "bytes": t_.expected_bytes(), "offset": off}
                    for n, t_, off in ((n, plan.tensors[n], o) for n, o in zip(names, offs))}
            used = sorted({t_.how for t_ in plan.tensors.values()})
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [source_entry(p) for p in ck.shards],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": "apache-2.0", "text": "Apache License 2.0 (the checkpoint's LICENSE file)"},
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "experts": {"format": "mxfp4",
                                "source": "weight (E2M1 pairs packed in int8), weight_scale_inv (UE8M0 per 32)",
                                "repack": "codes and scales unchanged, nibbles moved into 32-value ggml MXFP4 "
                                          "blocks", "tail": "fp32 {1, 1, 1, 0} per expert"},
                    "drafter": "model.mtp.* kept raw (dspark/raw + dspark/manifest.json)",
                    "transforms": {k: METHODS[k] for k in used}},
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        die(str(e))
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s"
          f" -> {out}")
