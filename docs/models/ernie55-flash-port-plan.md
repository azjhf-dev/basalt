# ERNIE 5.5 Flash Preview on Basalt: port plan

Iteration 1 (2026-11-24): feasibility and memory plan. Nothing GPU-tested yet; the agent's sandbox has
no GPU or CUDA toolkit, so every build here is a compile check and every runtime number needs Ash's
5090 + 5060 Ti box. Fork-only work (azjhf-dev/basalt); no upstream issues or PRs, per Ash 2026-11-23.

## Target

`baidu/ERNIE-5.5-Flash-Preview` (weights released 2026-11-23, Apache 2.0). Three repos: default
(~166 GB, blockwise FP8 with MXFP4 routed experts and an FP4 DSA indexer), FP8 (~302 GB), BF16
(~600 GB). The port targets the **default 166 GB build**: the only one small enough to consider on a
48 GB VRAM box, and the build Baidu's own serving commands point at.

Architecture (from the repo config.json and llama.cpp draft PR #31925, unmerged since 2026-11-09,
which this port uses as the math reference):

- 280B total / 14B active, 43 layers (1 dense + 42 MoE), hidden 4096, vocab 201,216, 1M context.
- **37 HCA layers** (compress ratio 128 + 128-token window): 64 heads, head dim 512, single KV head,
  q lora rank 1024, qk nope 448 + rope 64, v 512, sigmoid output gate, grouped low-rank output
  (8 groups, o lora rank 1024), VHA head mix (I + V U^T, folded at conversion), attention sink bias,
  YaRN (compress rope theta 160000).
- **6 MLA-DSA layers** (compress ratio -2): absorbed latent [kv 512 | rope 64] with the indexer key
  (128) in one cache row; 64 heads, qk nope 192, v 256; lightning indexer 64 heads x 128 dim,
  top-k 2048.
- **Latent MoE**: down 4096 -> 2048, experts gate/up/down in the 2048 latent, norm + up back;
  512 routed top-10 + 1 shared; split-feature routing (two sigmoid routers summed, learnable
  routed scaling 3.0, renormalized); SiTU activation (beta 4.0, linear beta 25.0).
- **mHC**: 4 hyper-connection streams, 20 Sinkhorn iterations, per-layer pre-norm 1/(sqrt(mean(x^2))+eps).
- **MultiMax** output head (learned ranges + ts).
- **DSpark drafter** (Preview only): block 7 (anchor + 6), Markov rank 256, target layers 40-42,
  noise token 1024; MTP block is one more MLA layer on [enorm(e) | hnorm(h)] with its own mHC head.
- Precision: MXFP4 routed experts (E2M1, UE8M0 scales, W4A8), blockwise FP8 E4M3 (128x128) elsewhere,
  FP4 DSA indexer.

## Why this is feasible on Basalt

ERNIE 5.5 is a DeepSeek-V4 hybrid (PR #31925's own words), and Basalt's `deepseek41` model is a
DeepSeek-V4.1-Flash engine that already has most of the machinery:

- DSV4-style hybrid raw cache: SWA window layers + compressed rows + a lightning indexer
  (`kv-cache-dsv4` in llama.cpp; `tier.cpp`, `attn_batch.cpp`, `context_policy.hpp` here).
- DSpark drafter with kept-experts ranking, block-parallel draft and confidence calibration
  (`dspark.cpp`/`dspark_kernels.cu`). ERNIE's DSpark differs in block size (7 vs 5), Markov rank,
  target layers and stage internals.
- MXFP4 handling (Engram table rows, NVFP4 test pack's expert path), q8_1 activation quantization,
  GPU + CPU expert compute with a RAM tier and a second-card split (`tier.cpp`) — the exact offload
  story a 166 GB model needs on 48 GB of VRAM.
- Hyper-connections in f32 (`Norms, router, hyper-connections: F32`), though ERNIE's mHC runs 4
  streams with 20 Sinkhorn iterations and its own pre-norm factor.

What is genuinely new (the work, in risk order):

1. **Converter** (`tools/convert/arch/ernie55.py`): read Baidu's safetensors (FP8 blockwise + MXFP4
   experts + FP4 indexer), fold the VHA head mix, emit a `.basalt` pack (dense/gguf/experts/dspark
   sections). PR #31925's `conversion/ernie5.py` is the reference for tensor names and the fold.
   Decision: keep experts MXFP4 natively (matches the pack, best size) rather than requantizing.
2. **HCA attention path**: gated grouped-low-rank output, single 512-dim KV head per layer in the SWA
   half of the cache, sink bias, YaRN. Needs new CUDA decode/prefill kernels + parity tests.
3. **MLA-DSA path**: absorbed latent + indexer key in one row; indexer top-k 2048 (DeepSeek's is a
   two-level top-512) — indexer memory and gather kernel changes.
4. **Latent MoE**: expert GEMMs in the 2048 latent with SiTU; the expert tier's per-expert size,
   CPU dequant path and routing-profile format all adjust (512 experts of 3 x 2048 x 2048).
5. **mHC (4 streams, Sinkhorn-20)** and **MultiMax** head: small but load-bearing for correctness.
6. **DSpark block-7 drafter** from the checkpoint's MTP block.

## Memory plan for the 5090 (32 GB) + 5060 Ti (16 GB) box

Weight budget (default pack, ~166 GB):

- Routed experts: 512 x 42 layers x 3 matrices x 2048 x 2048 params = ~270.6B params; MXFP4
  (~4.25 bits/param with scales) = **~144 GB**. This is the offload problem.
- Dense (attention, embeddings, lm_head, norms, indexer) in FP8: **~10-12 GB** on the main card.
- DSpark drafter: a few GB on the main card (DeepSeek's takes 2.41 GiB with 32 kept experts/stage).

KV cache per token (fp8, as Basalt stores DeepSeek's): HCA 37 x 512 B + MLA 6 x (512+64+128) B
= ~23.2 KB/token. 8K context ~0.2 GB; 32K ~0.75 GB; 128K ~3 GB; 1M ~24 GB (not realistic beside the
experts; plan `--max-context` 32K-128K on this box).

Placement plan (mirrors deepseek41's tier): main card holds dense + KV + drafter, fills the rest with
expert homes/copies (~15-18 GB); second card holds its expert share (~13 GB); RAM holds the remainder.

- **RAM >= 128 GB**: all remaining experts resident; decode limited by compute, the happy path.
- **RAM 96-128 GB**: coldest experts stream from SSD; expect a visible tok/s drop (DeepSeek measured
  2.8 tok/s with 172.8 GiB of experts on SSD; ERNIE's per-token expert read is ~10 x 12.6M x 0.53 B
  = ~67 MB worst case, so partial SSD residency is less fatal but still slow).
- **RAM 64 GB**: not recommended; most experts on SSD.

**Open question for Ash: how much RAM does the 5090 box have?** (The 126 GiB figure in the community
table is someone else's row.) The plan above sets the pack strategy.

## Iteration roadmap

1. **This doc**: feasibility + memory plan. (done)
2. **Converter**: `tools/convert/arch/ernie55.py` + pack writer; verify against the HF safetensors
   index (shapes, quant formats) and PR #31925's tensor map. CPU-only, testable here.
3. **Model skeleton**: `src/models/ernie55/` adapter + hparams + layer build, reusing the deepseek41
   session/tier; compile check only.
4. **Kernels + parity**: HCA attention, MLA-DSA top-2048 indexer, latent-MoE SiTU GEMMs (GPU mmvq +
   CPU path), mHC-4, MultiMax. Microbenches + CPU-sim parity, as with the qwen4exp MMQ fix.
5. **DSpark block-7 drafter** + kept-expert profile.
6. **Bench + tune on Ash's box** (5090 + 5060 Ti split, RAM tier sizing, `--max-context` guidance),
   honest numbers reported per iteration.

## Risks

- llama.cpp PR #31925 is an unmerged draft; its math is unverified by us and may lag the Preview
  checkpoint (it targets Flash Base; DSpark details come from the Preview config).
- MXFP4 W4A8 needs an activation-quant path Basalt's NVFP4 expert kernels only partially cover.
- HCA's VHA fold and sink bias are easy to get subtly wrong; parity against a HF reference forward
  pass (CPU, small context) is the gate before any bench.
- The 5060 Ti's PCIe 4.0 x4 link prices its expert share down (Basalt already prices by link speed).
