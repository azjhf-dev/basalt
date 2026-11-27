#!/usr/bin/env python3
"""Delta parity harness: basalt --lora vs llama.cpp --lora (docs/lora.md, iteration 3's run).

Runs both engines on a fixed prompt set, base and adapted, temp 0, and compares:
  1. greedy decode agreement of the ADAPTED models (exact-match prefix per prompt),
  2. base-vs-base drift per engine pair (context for how far the NVFP4/FP8 base sits from Q8_0),
  3. the delta structure: where base outputs differ, whether both adapters move their engine's
     output the same way (same first-divergence direction is a soft signal; the strict signal is 1).

llama.cpp serves the adapter card's recipe: Q8_0 base GGUF + the F16 adapter GGUF. basalt serves its
dense NVFP4/FP8 pack + --lora the same adapter GGUF.

Requires a GPU box (Ash's 5090). Basalt must be built from this branch; llama.cpp is cloned and
built at --llama-ref (pin the commit here after the first run: LLAMA_REF_DEFAULT).

TODO (iteration 4): per-position logit deltas. llama.cpp can dump logits (llama-cli --logits-file
or the server's completion probabilities); basalt needs a small --dump-logits flag on generate.
Until then this is token-level.

Usage:
  parity.py --basalt /path/to/basalt --basalt-pack /path/to/qwen3.8-27B.pack \
            --llama-base /path/to/Qwen3.8-27B-Q8_0.gguf --adapter /path/to/super-adapter-lora-F16.gguf \
            [--llama-ref COMMIT] [--tokens 256] [--workdir /tmp/lora-parity]
"""
import argparse
import json
import os
import subprocess
import sys

LLAMA_REF_DEFAULT = "f89a243d53c741cecc2b3630e491de61f459294c"  # ggml-org/llama.cpp master, 2026-11-27
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"

PROMPTS = [
    # code
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Implement a thread-safe LRU cache in C++ with std::mutex and std::list.",
    "Explain what this does and then rewrite it without recursion:\ndef f(x): return 1 if x < 2 else x * f(x - 1)",
    # prose
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Write a short product description for a mechanical keyboard aimed at programmers.",
    "Explain the difference between TCP and UDP to a high-school student.",
]


def run(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def build_llama(ref, workdir):
    src = os.path.join(workdir, "llama.cpp")
    if not os.path.isdir(src):
        run(["git", "clone", LLAMA_REPO, src])
    run(["git", "-C", src, "fetch", "origin"])
    run(["git", "-C", src, "checkout", ref])
    b = os.path.join(src, "build")
    if not os.path.exists(os.path.join(b, "bin", "llama-cli")):
        run(["cmake", "-B", b, "-DGGML_CUDA=ON"], cwd=src)
        run(["cmake", "--build", b, "--config", "Release", "-j", "--target", "llama-cli"], cwd=src)
    return os.path.join(b, "bin", "llama-cli")


def llama_tokens(cli, model, prompt, n, adapter=None, lora_scale=None):
    cmd = [cli, "-m", model, "-p", prompt, "-n", str(n), "--temp", "0", "-ngl", "99", "-c", "8192"]
    if adapter:
        cmd += ["--lora", adapter]
        if lora_scale is not None:
            cmd += ["--lora-scale", str(lora_scale)]
    out = run(cmd).stdout
    # llama-cli echoes the prompt then the generation; split on the prompt tail
    gen = out[out.rfind(prompt) + len(prompt):] if prompt in out else out
    return gen.strip()


def basalt_tokens(basalt, pack, prompt, n, adapter=None, lora_scale=None):
    # TODO: confirm this branch's CLI spelling on the built binary (basalt generate --help):
    # the iteration-2 flags are --lora / --lora-scale on the qwen35 model's generate command.
    cmd = [basalt, "generate", pack, "--prompt", prompt, "-n", str(n), "--temp", "0"]
    if adapter:
        cmd += ["--lora", adapter]
        if lora_scale is not None:
            cmd += ["--lora-scale", str(lora_scale)]
    return run(cmd).stdout.strip()


def common_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basalt", required=True)
    ap.add_argument("--basalt-pack", required=True)
    ap.add_argument("--llama-base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--llama-ref", default=LLAMA_REF_DEFAULT)
    ap.add_argument("--lora-scale", type=float, default=None)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--workdir", default="/tmp/lora-parity")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    cli = build_llama(a.llama_ref, a.workdir)

    report = []
    for i, p in enumerate(PROMPTS):
        lb = llama_tokens(cli, a.llama_base, p, a.tokens)
        la = llama_tokens(cli, a.llama_base, p, a.tokens, a.adapter, a.lora_scale)
        bb = basalt_tokens(a.basalt, a.basalt_pack, p, a.tokens)
        ba = basalt_tokens(a.basalt, a.basalt_pack, p, a.tokens, a.adapter, a.lora_scale)
        row = {
            "prompt": i,
            "adapted_common_chars": common_prefix(la, ba),
            "adapted_len_min": min(len(la), len(ba)),
            "base_drift_chars": common_prefix(lb, bb),
            "llama_adapter_changed_output": la != lb,
            "basalt_adapter_changed_output": ba != bb,
        }
        report.append(row)
        print(json.dumps(row))
    out = os.path.join(a.workdir, "parity-report.json")
    with open(out, "w") as f:
        json.dump({"llama_ref": a.llama_ref, "tokens": a.tokens, "lora_scale": a.lora_scale, "rows": report}, f, indent=2)
    ok = all(r["adapted_common_chars"] == r["adapted_len_min"] for r in report)
    changed = all(r["llama_adapter_changed_output"] and r["basalt_adapter_changed_output"] for r in report)
    print(f"report: {out}")
    print(f"greedy exact-match on all {len(report)} prompts: {ok}; adapter moved both engines on all: {changed}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
