# Inkling 2 Mini: 5060 Ti A/B

Experimental source bundle, prepared 2026-11-24. No GPU result is claimed.

## What this branch contains

Runtime/source snapshot of `xyehya/basalt` branch `inkling2-5080-setup` at
`3fdb44e9314abd4007fa0112ddd46e7fe2b40040` (see `SOURCE.json` for the verified full SHA).
It already integrates PRs #606, #609, #636, #649 and #650, their prerequisite work
(including #552, #525 and N11b), and upstream commits after v0.5.7. No integration
conflicts were resolved locally. This is not five independent cherry-picks and is
not an isolated test of any one PR.

GitHub Actions workflows are deliberately preserved from Ash's existing fork main.
The repo-only token cannot update workflows. No workflow is added or enabled for
this experiment. Source changes are a snapshot commit on a new branch, not a merge
into the working QFN branch. Upstream attribution and MIT license are retained.
`fix-qwen4exp-mmq-preload` stays at `94d06f7a0bba070f9ca00530fd4751a2dff0aaed`.

## Gate before running

Use Linux, CUDA 13+, CMake 3.24+, GCC 13+ and sufficient system RAM. The full pack is
about 19.6 GB; a 16 GB card spills experts to CPU/RAM. CPU type, worker count, RAM
speed, free VRAM and PCIe transfer speed all matter. Your PCIe 4.0 x4 link is not
the guide author's PCIe 5.0 x16 setup. Decode can compute RAM experts on the CPU;
not every spill is streamed across PCIe for every token. Record both CPU and link
conditions rather than predicting performance from PCIe alone.

Do not stop or reconfigure QFN automatically. Check whether its current process
uses the 5060 Ti. If QFN owns both cards, arrange downtime first; two processes must
not compete for this card during the A/B. QFN may continue on the 5090 only if that
is already its configuration, but keep its load constant or idle because it can
still contend for CPU/RAM. Do not overwrite its checkout, config or binaries.

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,power.limit --format=csv
nvidia-smi
lscpu
free -h
nvcc --version
cmake --version
g++ --version
```

Identify the 5060 Ti by its UUID, not an assumed GPU index. In the following shell:

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-5060-TI-UUID
# Inside this restricted process, the 5060 Ti is logical GPU 0.
```

## Separate builds, pinned revisions

From an existing clone of **azjhf-dev/basalt**, not from the QFN worktree:

```bash
git fetch origin experiment/inkling2-5060ti-5080-bundle
git fetch https://github.com/jesdga95/basalt.git tag v0.5.7
git worktree add ../inkling-ab-bundle origin/experiment/inkling2-5060ti-5080-bundle
git worktree add --detach ../inkling-ab-v057 cea3989c11e7314acc8d0c1c4393c0a7b47dabe6
for tree in ../inkling-ab-v057 ../inkling-ab-bundle; do
  cmake -S "$tree" -B "$tree/build" -DCMAKE_BUILD_TYPE=Release \
    -DBASALT_MODELS=inkling2 -DBUILD_TESTING=ON
  cmake --build "$tree/build" -j 4
  ctest --test-dir "$tree/build" -L cpu --output-on-failure
 done
```

Use the same compiler/CUDA version and build options on both. Reduce `-j 4` if RAM
is tight. Build failure is a gate, not a reason to substitute a release binary.
On the bundle, also run GPU parity before timing:

```bash
ctest --test-dir ../inkling-ab-bundle/build -L gpu \
  -R 'inkling2_(kernels|kv_int8|dspark|prompt)_parity|gqa_window_parity' \
  --output-on-failure
```

Check that named tests ran, not just that ctest returned success with no matches.
Some parity targets can be slow. Save errors and stop on CUDA faults, OOM, memory
shortage placement warnings or incoherent output. A memory-shortage placement
warning invalidates a controlled timing comparison even if the engine recovers.

## Model and settings

Reuse the same existing `Inkling-2-Mini-NVFP4-MX8.basalt` file for both. Do not
redownload weights or execute a remote install script for this experiment.

```bash
export MODEL=/absolute/path/to/Inkling-2-Mini-NVFP4-MX8.basalt
sha256sum "$MODEL"
export A=$(realpath ../inkling-ab-v057)
export B=$(realpath ../inkling-ab-bundle)
export RESULTS=$(pwd)/inkling-ab-results
mkdir -p "$RESULTS"
```

Documented pack SHA-256:
`28ec70723feaa037942a02f55b9d5e898276f3e62a3b1fa083745f932a041b99`.
If your existing pack differs, record its exact hash and name. Do not compare
results from different pack formats as if they were engine-only changes.

Primary comparison: `--gpu 0 --max-context 262144 --kv fp8 --spec 8`, single session,
no images. Start at 32K prompt depth before 131K. FP8 is supported by both builds;
v0.5.7 does not have this branch's INT8-KV addition. Test INT8 against the bundle's
FP8 separately, with fresh runs and a quality gate. It is not a like-for-like
v0.5.7 INT8 comparison. Keep the auto-selected pool-worker count the same, or pin
`--pool-workers N` identically after inspecting the startup logs.

Optional HTTP configs are in `experiments/inkling2-5060ti/config-*.json`. Replace
`REPLACE_WITH_ABSOLUTE_MODEL_PATH`, keep logical GPU 0 and bind to loopback port
8081. Use one server at a time with its own checkout's `bin/basalt serve --config`.
The benchmark below bypasses HTTP and uses the engine protocol, so it does not
read those JSON configs.

## Timing: warmup, then ABBA

Run from the bundle checkout so both engines use exactly the same Inkling fixtures.
One discarded warmup per build; then A, B, B, A. Each command starts a fresh engine.
Keep power limits, other jobs, room conditions and CPU affinity unchanged. Do not
change power limits without choosing to do so separately.

```bash
cd "$B"
for label in A B; do
  tree=$A; [ "$label" = B ] && tree=$B
  "$tree/bin/basalt-bench" --engine "$tree/bin/basalt" --model "$MODEL" \
    --fixtures "$B/data/bench/inkling2" --gpu 0 --max-context 262144 --kv fp8 --spec 8 \
    -t conv,struct,prose -p 16384 --seed 42 -r 1 -o json \
    --logs "$RESULTS/warmup-$label" > "$RESULTS/warmup-$label.json"
done
n=0
for label in A B B A; do
  n=$((n+1)); tree=$A; [ "$label" = B ] && tree=$B
  "$tree/bin/basalt-bench" --engine "$tree/bin/basalt" --model "$MODEL" \
    --fixtures "$B/data/bench/inkling2" --gpu 0 --max-context 262144 --kv fp8 --spec 8 \
    -t conv,struct,prose -p 16384,32768,131072 --seed 42 -r 1 -o json \
    --logs "$RESULTS/$n-$label" > "$RESULTS/$n-$label.json" || break
done
```

The `-t` decode rows are short-context conv/struct/prose, not decode after each
`-p` depth. Do not label them as deep-context results. For actual deep decode:

```bash
n=0
for label in A B B A; do
  n=$((n+1)); tree=$A; [ "$label" = B ] && tree=$B
  for depth in 32768 131072; do
    python3 tools/inkling2_deep_ab.py --engine "$tree/bin/basalt" --model "$MODEL" \
      --fixtures "$B/data/bench/inkling2" --depth "$depth" --max-context 262144 \
      --kv fp8 --spec 8 --tokens 512 \
      --out "$RESULTS/deep-$n-$label-$depth.json" || exit 1
  done
done
```

The helper uses deterministic repeated filler, greedy sampling and a fresh engine;
it records prompt hash, exact output token IDs, raw DONE statistics, TTFT, hit rate,
acceptance and stderr. It has been checked against a fake protocol engine only,
not against a GPU here. Early EOS can shorten output, so compare actual generated
tokens and lengths, not the requested 512 alone. Long-context behavior on real
code or conversation still needs separate tests.

For repeatability, rerun B at each depth and compare output token hashes and startup
placement logs. Cross-build token differences can reflect changed expert placement
or arithmetic, not necessarily a bug. Do not declare quality parity from throughput
or token identity on this filler. Run the existing coding/tool/long-context evals on
both with fixed requests and check truncation, malformed tool calls and regressions.
If you have a validated KLD reference for this same pack, also run the documented
`basalt-bench --quality --kld-base FILE` gate; do not invent a reference or threshold.

After the FP8 comparison, repeat bundle-only commands with `--kv int8`. Save to
new filenames. This measures the INT8 memory/performance tradeoff separately.
Start with 32K; only go to 131K after parity and quality checks pass. A 239K prompt
is optional after the smaller tests fit. Do not assume a 5080's 451 structured
tok/s is sustained chat speed or a target for this 5060 Ti.

## Send back

- Both exact commit IDs, model SHA, CUDA/driver/compiler versions, CPU and RAM speed.
- UUID, actual negotiated PCIe link and any measured host-to-device rate; free VRAM,
  peak VRAM/RSS, power cap, temperature and whether the 5090/QFN was active.
- All ABBA JSON and engine logs, not just the fastest row. Report per-build median
  and range for prefill at each depth, short conv/struct/prose decode, and deep
  32K/131K decode. Percent gain is `(B/A - 1) * 100` using matched metrics.
- TTFT, generated lengths, draft acceptance, GPU expert hit rate, CPU expert share,
  placement/copy-set messages, OOM/CUDA errors, and real eval pass counts.
- Identify FP8 bundle-v-release separately from INT8-v-FP8 on the bundle.

## v0.5.7 config break and rollback

`gpu: auto` is removed. Multiple cards require both an explicit list, such as
`"gpu": "0,1"`, and an explicit `"gpu-mode": "peer"` or `"split"`. This experiment
uses one visible GPU and needs neither mode. Do not copy the new multi-GPU syntax
into the old working QFN config or migrate QFN as part of this test.

Rollback: stop only the experiment process; leave the QFN process/config/worktree
alone. A failed A/B does not change the working branch. No upstream issue/PR was
created. No server on Ash's hardware was started by this preparation.

## Local validation

See `VALIDATION.md` and `SOURCE.json`. Runtime source matches the pinned community
tree; only the preserved workflows and this experiment's added files differ.
Eight focused C++ host tests compile and pass. Full engine/GPU build is unverified:
installed CMake is 3.22.1 (minimum 3.24), GCC is 11.4 (documented requirement 13+),
and nvcc/GPU are absent. These are preparation checks, not performance evidence.
