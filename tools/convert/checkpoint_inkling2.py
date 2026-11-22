"""Inkling 2 Mini pack from the ggml-org GGUF (arch inkling2_loop), for tools/convert/convert.py.

STATUS: skeleton - inventories the GGUF and writes nothing yet. Run:
    python3 checkpoint_inkling2.py INKLEI2.gguf --inventory

llama.cpp GGUF tensor names (from azjhf-dev/Avalon branch inkling2-mini, gguf-py tensor_mapping):
  blk.N.attn_qkv.weight / .bias?   fused QKV projection (+ wqkv_gate per the model card)
  blk.N.attn_rel_bias.weight       per-layer relative bias table  <- THE table the Basalt FA kernel gathers
  blk.N.attn_sink_proj.weight      learned sinks projected from the token embedding
  blk.N.attn_post_conv.weight      short conv on attention output (+ conv state)
  blk.N.ffn_gate_inp.weight        router (sigmoid, 128 experts top-8, expert_weights_scale)
  blk.N.ffn_gate_exps.weight / ffn_up_exps / ffn_down_exps (+ _s scales: MXFP8/NVFP4 block scales)
  blk.N.ffn_post_conv.weight
  loop.N.*                         per-physical-loop-layer pass-gate tensors (N = layer - loop.start)
  token_embd.weight, output_norm.weight, output_res_query.weight, output.weight

TODO(iteration 3):
  - map each tensor to a .basalt section via model_file.ModelWriter (see checkpoint_qwen35.py:
    NVFP4 as planes type 45 + <name>.scale F32[1], MXFP8 as F8_E4M3 type 44 + scale)
  - write the inkling2_loop.* metadata keys meta.hpp::validate_meta requires
  - tokenizer/* + meta from the HF repo (config.json, generation_config, model card chat form)
  - decide the pack layout for loop pass-gate tensors (indexed by pass vs by physical layer)
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gguf_reader import GGUFReader  # noqa: E402  (name check: adjust to the actual class in tools/gguf_reader.py)


def inventory(path: str) -> None:
    r = GGUFReader(path)
    for name, shape, dtype in r.tensors():
        print(f"{name}\t{dtype}\t{shape}")
    for k, v in r.metadata():
        print(f"@{k} = {v}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--inventory":
        inventory(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(1)
