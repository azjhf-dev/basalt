# Inkling 2 Mini: fork plan (azjhf-dev/basalt)

2026-11-23. This file replaces the earlier design doc and scaffolding (branch history
`c54910ab..edb501ba`, kept in git). Upstream shipped full Inkling 2 Mini support in
[v0.5.5](https://github.com/jesdga/basalt/releases/tag/v0.5.5) (`ff0b978d`), so this branch is
reset to that tag and the fork's job changes from "build the support" to "add what upstream
does not have".

All upstream performance numbers quoted below are developer-reported in the v0.5.5 release
notes and `docs/models/inkling2.md`. Nothing in this fork has been run on hardware yet.

## What upstream v0.5.5 already covers

Against the original asks (the rel-bias / flash-attention / quantized-KV problems):

- **Relative bias inside the attention kernel - done.** Decode
  (`src/models/inkling2/kernels/attn.cu`): each query head's scores take a per-query relative
  bias over the last `kExtent` positions and a per-query sink logit sits in the softmax
  denominator. Prompt (`kernels/attn_prompt.cu`): flash-style tensor-core path where the bias
  is one more product, R (rows' relative states) times the bias-table band the tile spans,
  staged through shared memory. No materialized n_kv x n_tokens bias buffer - the llama.cpp
  prefill memory blowup does not exist here.
- **Learned sinks - done**, in the same kernels (merge kernel folds the sink into the
  denominator).
- **Quantized KV - partially.** Cache is fp16 or E4M3 fp8 (`--kv fp8`), per `caches.hpp` and
  the docs matrix. No int8/q4 options; Qwen4exp in the same engine has `--kv int8/q4_0/k8v4`,
  so the patterns exist in-tree but are not wired for inkling2.
- **Loop / layer visits - done.** `meta.hpp` reads `inkling2.loop.start/end/count`;
  `session.cpp` gives each of the 48 visits its own cache and conv state.
- **MoE at speed - done.** NVFP4 experts bit-for-bit on the FP4 block-scaled tensor cores in
  prefill; CUDA-graph decode with programmatic dependent launch; least-routed experts paged to
  RAM when context needs the room (`spill.cpp`, shipped profile `data/profiles/inkling2-24x128.prof`).
- **Vision, DSpark drafting, 1M context, tools, thinking effort - done** (see
  `docs/models/inkling2.md`).

Upstream's own stated limits for inkling2 (`docs/models/inkling2.md`, `docs/models/index.md`):

- One card: **no second card, no layer split** (Qwen4exp in the same engine has both:
  up to three expert-tier cards, and matched-card layer split).
- **Parallel slots "not measured"** for inkling2 (`--parallel N` exists engine-wide).
- KV cache: fp16/fp8 only.
- NVIDIA sm_120 only, CUDA 13+. No HIP/RDNA3 anywhere in the tree.

## Fork differentiators (in priority order)

### 1. Multi-GPU for inkling2: expert-tier second card, then layer split

Gap is explicit in upstream's matrix. The engine already has the machinery for other models:
`include/basalt/core/peer_experts.hpp`, `expert_source.hpp` (Qwen4exp/Hy4-flash run expert
tiers on extra cards, with optional sm_89 peer kernels via `PEER_ADA`), and Qwen3.8-27B has a
layer split with pipelined prompt chunks. The inkling2 model dir has no peer usage at all.

Plan:
1. Read how qwen4exp wires `peer_experts` (dispatch, the host-written join word, sizing after
   the drafter) and what a model must provide.
2. Inkling2 expert-tier card: route the least-routed experts (profile already shipped) to a
   second card before falling back to RAM. Smallest useful step; helps sub-32GB main cards.
3. Layer split for inkling2 (matched cards), modeled on qwen38-27b's split with pipelined
   chunks. Bigger job; the loop (visits 2 and 3 re-enter the core) makes the split boundary
   and hand-off state more involved than a linear stack.

### 2. Parallel-slot measurement and tuning

`--parallel N` is unmeasured for inkling2. Once the model runs on Ash's 5090: bench N=1..4,
find where the caches and drafter scratch stop fitting, document real numbers. Code work only
if the measurements show a fixable bottleneck.

### 3. KV int8/q4 options

Port the qwen4exp `--kv int8/q4_0/k8v4` cache formats to inkling2's `VisitCache` and the
decode/prompt kernels' unpack paths (fp16/E4M3 today, `attn.cu` `unpack16*`).
Moderate, self-contained, and directly extends what Ash asked for originally (quantized KV).
Note: fp8 E4M3 already halves the cache; int8/q4 is for tighter budgets, not accuracy.

### 4. RDNA3 (HIP) - not near-term, notes only

An honest feasibility read of the v0.5.5 sources:

- The inkling2 kernels alone are ~2,900 lines of CUDA with heavy inline PTX:
  `mma.sync.m16n8k16 f16` (prompt attention), `ldmatrix`, `cp.async` (16 sites),
  thread-block clusters (`barrier.cluster.*`), and programmatic dependent launch
  (~20 sites across decode). The shared engine's FP4 prefill (`prefill::fp4`,
  `include/basalt/kernels/nvfp4_planes.hpp`) targets sm_120 block-scaled tensor cores.
- None of these have HIP equivalents on RDNA3: no clusters, no PDL, no `cp.async`,
  no FP4 hardware; `mma.sync` would become rocWMMA with different fragment shapes.
  This is a rewrite of the attention and expert kernels, not a port. Estimate: weeks,
  and it would track upstream badly (every upstream kernel change needs re-porting).
- The engine also hard-gates `CMAKE_CUDA_ARCHITECTURES` to 120 and needs CUDA 13.

Recommendation: the near-term RDNA3 answer for Inkling 2 Mini stays the Avalon fork
(ggml HIP backend shares the ggml-cuda sources, so the inkling2_loop arch work there is
arch-neutral). Revisit Basalt-on-RDNA3 only if Avalon's HIP path proves too slow to matter.

## Near-term queue

1. Get Ash's 5090 time: build v0.5.5 as-is, pack Inkling-2-Mini-NVFP4-q8
   (`tools/convert/convert.py --checkpoint DIR` against the official release), reproduce the
   upstream bench numbers before any fork change lands. Everything above depends on a working
   baseline.
2. Expert-tier second card for inkling2 (differentiator 1, step 1-2).
3. `--parallel` measurement (differentiator 2) once the baseline runs.
4. KV int8/q4 (differentiator 3) - code-only, can start before hardware.
