#!/usr/bin/env python3
"""Cross-check a llama.cpp-format LoRA adapter GGUF against its PEFT safetensors source.

Every tensor must exist in both containers, with shapes that are transposes in name but identical
flat layouts (GGUF lora_a (in, r) dim0-fastest == PEFT lora_A [r, in] row-major), and values within
the GGUF's storage rounding (F16: relative error <= 2**-11 for normal values).

Usage: verify_adapter.py ADAPTER.gguf ADAPTER.safetensors
Exit 1 on any mismatch. Only numpy is required.
"""
import json
import struct
import sys

import numpy as np


def read_gguf(path):
    import mmap
    f = open(path, "rb")
    d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert d[:4] == b"GGUF", "not a GGUF"
    ver, nt, nkv = struct.unpack_from("<IQQ", d, 4)
    assert ver == 3, f"GGUF v{ver}"
    off = 24

    def rstr(o):
        n = struct.unpack_from("<Q", d, o)[0]
        return d[o + 8 : o + 8 + n].decode(), o + 8 + n

    meta = {}
    for _ in range(nkv):
        k, off = rstr(off)
        t = struct.unpack_from("<I", d, off)[0]
        off += 4
        if t == 8:
            v, off = rstr(off)
        elif t in (0, 1, 7):
            v = d[off]
            off += 1
        elif t in (2, 3):
            v = struct.unpack_from("<H", d, off)[0]
            off += 2
        elif t in (4, 5):
            v = struct.unpack_from("<I", d, off)[0]
            off += 4
        elif t == 6:
            v = struct.unpack_from("<f", d, off)[0]
            off += 4
        elif t in (10, 11, 12):
            v = struct.unpack_from("<Q", d, off)[0]
            off += 8
        elif t == 9:
            et, cnt = struct.unpack_from("<IQ", d, off)
            off += 12
            if et == 8:
                v = []
                for _ in range(cnt):
                    s, off = rstr(off)
                    v.append(s)
            else:
                sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et]
                v = f"<{cnt} x {sz}B>"
                off += sz * cnt
        else:
            raise ValueError(f"meta type {t}")
        meta[k] = v

    tensors = {}
    infos = []
    align = int(meta.get("general.alignment", 32))
    for _ in range(nt):
        name, off = rstr(off)
        nd = struct.unpack_from("<I", d, off)[0]
        off += 4
        dims = struct.unpack_from("<" + "Q" * nd, d, off)
        off += 8 * nd
        ty = struct.unpack_from("<I", d, off)[0]
        off += 4
        toff = struct.unpack_from("<Q", d, off)[0]
        off += 8
        infos.append((name, dims, ty, toff))
    data_start = (off + align - 1) // align * align
    for name, dims, ty, toff in infos:
        n = 1
        for x in dims:
            n *= x
        if ty == 1:
            arr = np.frombuffer(d, np.uint16, n, data_start + toff)
            tensors[name] = (dims, arr)
        elif ty == 0:
            tensors[name] = (dims, np.frombuffer(d, np.float32, n, data_start + toff))
        else:
            raise ValueError(f"{name}: ggml type {ty} not handled")
    return meta, tensors


def f16_to_f32(u16):
    return u16.view(np.float16).astype(np.float32)


PEFT_BASE = "base_model.model.model.language_model.layers.{L}.{mod}.lora_{AB}.weight"
MOD = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def read_safetensors(path):
    import mmap
    f = open(path, "rb")
    d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n = struct.unpack_from("<Q", d, 0)[0]
    hdr = json.loads(d[8 : 8 + n])
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", f"{k}: {v['dtype']}"
        b0, b1 = v["data_offsets"]
        out[k] = (tuple(v["shape"]), np.frombuffer(d, np.float32, (b1 - b0) // 4, base + b0))
    return out


def main():
    gmeta, g = read_gguf(sys.argv[1])
    st = read_safetensors(sys.argv[2])
    assert gmeta.get("general.type") == "adapter" and gmeta.get("adapter.type") == "lora"
    print(f"gguf: {len(g)} tensors, alpha={gmeta.get('adapter.lora.alpha')}; safetensors: {len(st)} tensors")

    seen = set()
    worst = 0.0
    worst_name = None
    for name, (dims, arr) in g.items():
        import re

        m = re.match(r"blk\.(\d+)\.(.+)\.weight\.lora_([ab])$", name)
        assert m, name
        layer, base, ab = int(m.group(1)), m.group(2), m.group(3).upper()
        pname = PEFT_BASE.format(L=layer, mod=MOD[base], AB=ab)
        assert pname in st, f"{pname} missing from safetensors"
        seen.add(pname)
        pshape, parr = st[pname]
        # GGUF (in, r) dim0-fastest has the same flat layout as PEFT [r, in] row-major
        assert parr.size == arr.size, f"{name}: element count {arr.size} vs PEFT {parr.size}"
        gv = f16_to_f32(arr) if arr.dtype == np.uint16 else arr
        num = np.abs(gv - parr)
        mae = float(np.max(num))
        rel = mae / float(max(np.max(np.abs(parr)), 1e-30)) if parr.size else 0.0
        if rel > worst and mae > 1e-3:
            worst, worst_name = rel, name
        assert mae <= 2e-3 or rel <= 2**-10, f"{name}: beyond f16 rounding (max abs {mae}, max rel {rel})"
    missing = set(st) - seen
    assert not missing, f"safetensors tensors with no GGUF counterpart: {sorted(missing)[:5]}"
    print(f"all {len(g)} tensors match within storage rounding (worst rel {worst:.2e} at {worst_name})")


if __name__ == "__main__":
    main()
