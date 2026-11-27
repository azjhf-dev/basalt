#!/usr/bin/env python3
"""A numpy oracle for the ERNIE 5.5 CPU-reference math (src/models/ernie55/ref.cpp).

An independent second implementation of one HCA layer and one MLA-DSA layer, written straight from the
documented semantics (ref.hpp's header comment) against the same converted pack. It reads the pack's gguf
section with the repo's own GGUF reader, dequantizes with the repo's ggufq, runs both layers in numpy, and
writes a binary oracle file (inputs + expected outputs) that ernie55_ref_test compares against:

    i32 Thca, f32 xh[Thca*H], f32 hca[Thca*H], i32 Tmla, f32 xm[Tmla*H], f32 mla[Tmla*H], f32 idx[Tmla*Tmla]

Usage: ernie55_ref_oracle.py PACK OUT_BIN
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ggufq  # noqa: E402
from model_file import ModelFile  # noqa: E402


class Pack:
    def __init__(self, path: str):
        mf = ModelFile(path)
        self.meta = json.loads(mf.read("meta"))["gguf_kv"]
        self.tensors = json.loads(mf.read("meta"))["tensors"]
        start, size = mf.section("gguf")
        with open(path, "rb") as f:
            f.seek(start)
            self.blob = f.read(size)

    def deq(self, name: str) -> np.ndarray:
        """Values as float32 in checkpoint layout: 2D -> [out, in], 1D -> [n]."""
        t = self.tensors[name]
        raw = np.frombuffer(self.blob[t["offset"]:t["offset"] + t["bytes"]], np.uint8)
        v = ggufq.dequantize(raw, ggufq.GGMLQuantizationType[t["type"]]).astype(np.float32)
        return v.reshape(tuple(reversed(t["shape"])))


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * w


def yarn_inv_freq(rot, base, freq_scale, n_ctx_orig, beta_fast, beta_slow):
    i = np.arange(rot // 2, dtype=np.float64)
    theta = base ** (-2.0 * i / rot)
    wavelen = 2 * np.pi / theta
    low, high = n_ctx_orig / beta_fast, n_ctx_orig / beta_slow
    out = theta.copy()
    mid = (wavelen >= low) & (wavelen <= high)
    smooth = (n_ctx_orig / wavelen[mid] - beta_slow) / (beta_fast - beta_slow)
    out[mid] = theta[mid] * (1 - smooth) + theta[mid] * freq_scale * smooth
    out[wavelen > high] = theta[wavelen > high] * freq_scale
    return out


def rope(x, off, pos, inv_freq):
    """Interleaved rope on x[..., off:off+2*len(inv_freq)], in place on the last axis."""
    ang = pos * inv_freq
    c, s = np.cos(ang), np.sin(ang)
    seg = x[..., off:off + 2 * len(inv_freq)].reshape(x.shape[:-1] + (-1, 2)).copy()
    out = np.empty_like(seg)
    out[..., 0] = seg[..., 0] * c - seg[..., 1] * s
    out[..., 1] = seg[..., 0] * s + seg[..., 1] * c
    x[..., off:off + 2 * len(inv_freq)] = out.reshape(x.shape[:-1] + (2 * len(inv_freq),))
    return x


def rope_back(x, off, pos, inv_freq):
    ang = pos * inv_freq
    c, s = np.cos(ang), np.sin(ang)
    seg = x[..., off:off + 2 * len(inv_freq)].reshape(x.shape[:-1] + (-1, 2)).copy()
    out = np.empty_like(seg)
    out[..., 0] = seg[..., 0] * c + seg[..., 1] * s
    out[..., 1] = -seg[..., 0] * s + seg[..., 1] * c
    x[..., off:off + 2 * len(inv_freq)] = out.reshape(x.shape[:-1] + (2 * len(inv_freq),))
    return x


def rope_neox(x, pos, inv_freq):
    rot = 2 * len(inv_freq)
    ang = pos * inv_freq
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., :rot // 2].copy(), x[..., rot // 2:rot].copy()
    x[..., :rot // 2] = a * c - b * s
    x[..., rot // 2:rot] = a * s + b * c
    return x


def softmax_sink(scores, sink):
    """scores [...], scalar sink -> weights over the last axis; the sink only pads the denominator."""
    mx = np.maximum(scores.max(axis=-1, keepdims=True), sink)
    e = np.exp(scores - mx)
    den = e.sum(axis=-1, keepdims=True) + np.exp(sink - mx)
    return e / den


def hca_forward(pk, g, x):
    nh, hd, H = g["nh"], g["hd"], g["H"]
    ql, eps, win, ratio = g["q_lora"], g["eps"], g["window"], g["ratio"]
    og, ol, rot = g["o_groups"], g["o_lora"], g["rot"]
    pre = "blk.0."
    q_a, q_b = pk.deq(pre + "attn_q_a.weight"), pk.deq(pre + "attn_q_b.weight")
    kvw, o_a, o_b = pk.deq(pre + "attn_kv.weight"), pk.deq(pre + "attn_output_a.weight"), pk.deq(
        pre + "attn_output_b.weight")
    gatew = pk.deq(pre + "attn_gate.weight")
    ckv, cgate, ape = pk.deq(pre + "attn_compressor_kv.weight"), pk.deq(
        pre + "attn_compressor_gate.weight"), pk.deq(pre + "attn_compressor_ape.weight")
    qan, kvn, cn = pk.deq(pre + "attn_q_a_norm.weight"), pk.deq(pre + "attn_kv_a_norm.weight"), pk.deq(
        pre + "attn_compressor_norm.weight")
    sinks, vha = pk.deq(pre + "attn_sinks.weight"), pk.deq(pre + "attn_vha.weight")
    inv = yarn_inv_freq(rot, g["comp_base"], 1 / g["yarn_factor"], g["yarn_ctx"], g["beta_fast"], g["beta_slow"])
    off = hd - rot
    T = len(x)
    qr = rmsnorm(x @ q_a.T, qan, eps)
    q = (qr @ q_b.T).reshape(T, nh, hd)
    q = q / np.sqrt(np.mean(q**2, axis=-1, keepdims=True) + eps)
    for t in range(T):
        rope(q[t], off, t, inv)
    kv = rmsnorm(x @ kvw.T, kvn, eps)
    for t in range(T):
        rope(kv[t:t + 1], off, t, inv)
    skv = x @ ckv.T
    ssc = x @ cgate.T + ape[np.arange(T) % ratio]
    nb = T // ratio
    comp = np.zeros((nb, hd), np.float64)
    for b in range(nb):
        w = np.exp(ssc[b * ratio:(b + 1) * ratio] - ssc[b * ratio:(b + 1) * ratio].max(axis=0, keepdims=True))
        w /= w.sum(axis=0, keepdims=True)
        comp[b] = (w * skv[b * ratio:(b + 1) * ratio]).sum(axis=0)
    comp = rmsnorm(comp, cn, eps)
    for b in range(nb):
        rope(comp[b:b + 1], off, (b + 1) * ratio - 1, inv)
    out = np.zeros((T, H), np.float64)
    scale = 1 / np.sqrt(hd)
    for t in range(T):
        j0 = max(0, t - win + 1)
        keys = np.concatenate([kv[j0:t + 1], comp[:t // ratio]])
        s = (q[t] @ keys.T) * scale
        p = np.stack([softmax_sink(s[h], sinks[h]) for h in range(nh)])
        heads = p @ keys
        for h in range(nh):
            rope_back(heads[h:h + 1], off, t, inv)
        heads = np.einsum("hg,gd->hd", vha, heads)
        gate = 1 / (1 + np.exp(-(qr[t] @ gatew.T)))
        oa = np.zeros(og * ol)
        for gg in range(og):
            sl = heads[gg * (nh // og):(gg + 1) * (nh // og)].reshape(-1)
            oa[gg * ol:(gg + 1) * ol] = (o_a[gg * ol:(gg + 1) * ol] @ sl) * gate[gg * ol:(gg + 1) * ol]
        out[t] = o_b @ oa
    return out


def mla_forward(pk, g, x):
    nh, H, ql = g["nh"], g["H"], g["q_lora"]
    nope, rot, kl, v, eps = g["nope"], g["rot"], g["kv_lora"], g["mla_v"], g["eps"]
    idh, idd = g["idx_heads"], g["idx_dim"]
    pre = "blk.2."
    q_a, q_b = pk.deq(pre + "attn_q_a.weight"), pk.deq(pre + "attn_q_b.weight")
    kv_a, kv_b, ow = pk.deq(pre + "attn_kv_a_mqa.weight"), pk.deq(pre + "attn_kv_b.weight"), pk.deq(
        pre + "attn_output.weight")
    gatew = pk.deq(pre + "attn_gate.weight")
    qan, kvan = pk.deq(pre + "attn_q_a_norm.weight"), pk.deq(pre + "attn_kv_a_norm.weight")
    sinks, vha = pk.deq(pre + "attn_sinks.weight"), pk.deq(pre + "attn_vha.weight")
    iw, iqb, iproj = pk.deq(pre + "indexer.attn_k.weight"), pk.deq(pre + "indexer.attn_q_b.weight"), pk.deq(
        pre + "indexer.proj.weight")
    knw, knb = pk.deq(pre + "indexer.k_norm.weight"), pk.deq(pre + "indexer.k_norm.bias")
    inv = g["rope_base"] ** (-2.0 * np.arange(rot // 2) / rot)
    T = len(x)
    q_c = rmsnorm(x @ q_a.T, qan, eps)
    gate = 1 / (1 + np.exp(-(q_c @ gatew.T)))
    q = (q_c @ q_b.T).reshape(T, nh, nope + rot)
    kvf = x @ kv_a.T
    kv_c = rmsnorm(kvf[:, :kl], kvan, eps)
    # q_pe / k_pe: plain interleaved rope at rope_theta, no YaRN
    q_pe = q[:, :, nope:].copy()
    for t in range(T):
        rope(q_pe[t], 0, t, inv)
    k_pe = kvf[:, kl:].copy()
    for t in range(T):
        rope(k_pe[t:t + 1], 0, t, inv)
    kb = kv_b.reshape(nh, nope + v, kl)
    q_abs = np.einsum("thn,hnc->thc", q[:, :, :nope], kb[:, :nope])
    Q = np.concatenate([q_abs, q_pe], axis=-1)
    K = np.concatenate([kv_c, k_pe], axis=-1)
    scale = 1 / np.sqrt(nope + rot)
    out = np.zeros((T, H), np.float64)
    for t in range(T):
        s = (Q[t] @ K[:t + 1].T) * scale
        p = np.stack([softmax_sink(s[h], sinks[h]) for h in range(nh)])
        a = p @ kv_c[:t + 1]
        heads = np.einsum("hvc,hc->hv", kb[:, nope:], a)
        heads = np.einsum("hg,gd->hd", vha, heads)
        out[t] = ow @ (heads.reshape(-1) * gate[t])
    # indexer
    ik = x @ iw.T
    mean = ik.mean(axis=-1, keepdims=True)
    var = ik.var(axis=-1, keepdims=True)
    ik = (ik - mean) / np.sqrt(var + 1e-6) * knw + knb
    iq = (q_c @ iqb.T).reshape(T, idh, idd)
    for t in range(T):
        rope_neox(ik[t:t + 1], t, inv)
        rope_neox(iq[t], t, inv)
    wproj = (x @ iproj.T) / np.sqrt(idd * idh)
    scores = np.full((T, T), -np.inf)
    for t in range(T):
        rel = np.maximum(iq[t] @ ik[:t + 1].T, 0.0)
        scores[t, :t + 1] = (rel * wproj[t, :, None]).sum(axis=0)
    return out, scores


def main(pack_path: str, out_path: str):
    pk = Pack(pack_path)
    A = "ernie55."
    m = pk.meta
    g = dict(H=int(m[A + "embedding_length"]), nh=int(m[A + "attention.head_count"]),
             hd=int(m[A + "attention.key_length"]), q_lora=int(m[A + "attention.q_lora_rank"]),
             kv_lora=int(m[A + "attention.kv_lora_rank"]), eps=float(m[A + "attention.layer_norm_rms_epsilon"]),
             window=int(m[A + "attention.sliding_window"]), ratio=128,
             o_groups=int(m[A + "attention.output_group_count"]),
             o_lora=int(m[A + "attention.output_lora_rank"]), rot=int(m[A + "rope.dimension_count"]),
             comp_base=float(m[A + "attention.compress_rope_freq_base"]),
             yarn_factor=float(m[A + "rope.scaling.factor"]),
             beta_fast=float(m[A + "rope.scaling.beta_fast"]),
             beta_slow=float(m[A + "rope.scaling.beta_slow"]),
             yarn_ctx=int(m[A + "rope.scaling.original_context_length"]),
             rope_base=float(m[A + "rope.freq_base"]),
             nope=int(m[A + "attention.key_length_mla"]) - int(m[A + "rope.dimension_count"]),
             mla_v=int(m[A + "attention.value_length_mla"]),
             idx_heads=int(m[A + "attention.indexer.head_count"]),
             idx_dim=int(m[A + "attention.indexer.key_length"]))
    rng = np.random.default_rng(42)
    Th, Tm = 200, 64
    xh = rng.standard_normal((Th, g["H"]))
    xm = rng.standard_normal((Tm, g["H"]))
    hca = hca_forward(pk, g, xh)
    mla, idx = mla_forward(pk, g, xm)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", Th))
        f.write(xh.astype("<f4").tobytes())
        f.write(hca.astype("<f4").tobytes())
        f.write(struct.pack("<i", Tm))
        f.write(xm.astype("<f4").tobytes())
        f.write(mla.astype("<f4").tobytes())
        f.write(idx.astype("<f4").tobytes())
    print(f"oracle written: hca {hca.shape}, mla {mla.shape}, idx {idx.shape} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
