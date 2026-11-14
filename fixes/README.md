# Fixes branch: `build/basalt-e9db1ae-fixes`

A Basalt companion branch in the style Robert Fairburn uses for his Strata-rdna3
fork: a named branch per stack of changes, each documented with its provenance.

Base: `jesdga95/basalt` @ `e9db1aea` (2026-11-14). Everything here is additive:
one header, one test, this directory. No engine source is modified.

## What was ported, and from where

The source work is rfairburn's stack on top of the Strata-rdna3 fork
([rfairburn/Strata-rdna3](https://github.com/rfairburn/Strata-rdna3), MIT, itself
on [Niko1221/Strata](https://github.com/Niko1221/Strata), MIT, which Basalt also
forks):

| Fix | Source branch | Source commits |
| --- | --- | --- |
| Four-GPU pipeline + queue wake fixes | `build/rdna3-f557b54-p4-fixes` | `b7c5a264` |
| DP4 / pipelined-MTP + frontend integrations | `integration/rdna3-dp4-pmtp-frontend` | `58926bc1` |
| Owner-local mmap4K recipe + allocator workaround | `experiment/owner-local-mmap4k-primary` | `2c57b12b`, `30ed20d4` |

## Fix 1 - better multi-GPU support: PARTIALLY PORTED

**What rfairburn built.** A data-parallel prefill planner
(`include/strata/prefill/dp_plan.hpp`): one long prompt is split into 256-aligned
chunks dealt round-robin across the GPU paths, every card running all layers for
its own chunks, with the last chunk kept above a minimum so a short tail does not
pay for its own pass. Plus the pipeline wiring in `src/prefill/prefill.cpp` and
`src/program/generate.cpp`, and ROCm KFD queue-wake fixes in `src/core/kfdq.cpp`.

**What Basalt already has.** Up to four cards (a main card plus up to three
expert tiers; a fifth idles), and long prompts read as a *layer* pipeline over
matched cards: each card reads a block of layers and hands the chunk on
([docs/configuration/gpus.md](../docs/configuration/gpus.md)). That is a
different shape from rfairburn's data-parallel chunk fan-out.

**Ported here.** The planner itself, as
[`include/basalt/prefill/dp_plan.hpp`](../include/basalt/prefill/dp_plan.hpp)
(namespace changed, logic identical) with its property test
[`tests/prefill/dp_plan_test.cpp`](../tests/prefill/dp_plan_test.cpp),
registered as a `cpu`-labelled ctest (`dp_plan_test`). The test passes on this
branch.

**Not ported, deliberately.** Wiring the planner into qwen4exp's prompt read
(`src/models/qwen4exp/serve/prompt_split.cpp` and friends) is an architectural
change: expert ownership per card, KV handoff, and GDN state across chunk
boundaries all differ from Strata's layout, and nothing can be validated without
Blackwell hardware. The planner lands as a tested building block for that future
work, not as a live path. The KFD queue-wake fixes are ROCm-only (the KFD is
AMD's kernel driver); NVIDIA has no equivalent subsystem, so there is nothing to
carry.

## Fix 2 - MTP on pipeline parallelism: ALREADY IN BASALT

**What rfairburn carried.** Mainline Strata's pipelined MTP: with `--batch`
slots pipelined through the stages of a layer split, per-slot drafting is
normally dropped; `STRATA_PIPED_MTP` (with the resolver in
`include/strata/program/batch_groups.hpp`) keeps the drafts, one slot per group,
each stage committing a group's window after the last stage judges its
proposals.

**Basalt's equivalent.** On two alike cards, Basalt's layer split already
overlaps consecutive windows *with* the drafter: the first card starts window
K+1 from the guess that window K is accepted whole while the second card still
verifies K, and a failed guess restores state and re-runs, so output is unchanged
([docs/configuration/gpus.md](../docs/configuration/gpus.md), "Layer split of
alike cards, overlapped windows"; the log line is `pipelined windows: N
launched, M held`). In a plain layer split the drafter stays on the first card
and reads the last card's rows of each window. Drafting is on by default
(`--spec`, [docs/models/qwen38-flash-next.md](../docs/models/qwen38-flash-next.md)).

Nothing to port: the capability the Strata-rdna3 fork was missing exists here by
design.

## Fix 3 - mmap with improved prefill speed: ADAPTED

**What rfairburn built.** An opt-in recipe (not an engine change): a wrapper
that maps the expert arena from a file (`STRATA_ARENA_MMAP=1`), pins prefill to
4K chunks with DP prefill and borrowing off, suppresses native profiling hooks,
and sets `MALLOC_MMAP_THRESHOLD_=131072` so glibc returns large freed allocations
to the OS instead of retaining them in arenas. His README is explicit that this
is a narrowly tested workaround on one 4x RX 7900 XTX (ROCm) host, not a general
speedup claim.

**What Basalt already has.** Native `--mmap-experts` (a file mapping instead of
the resident RAM copy, with `--resident-budget-gib` to keep only the hottest
experts in RAM) and `--shared-expert-arena` for one arena file across processes
([docs/configuration/index.md](../docs/configuration/index.md)). The arena is
already an anonymous mapping whose pages are committed as the load writes them
(`src/core/expert_source.cpp`).

**Ported here.** The one engine-agnostic piece: the glibc allocator workaround
and wrapper discipline, as [`fixes/basalt-mmap-launcher.sh`](basalt-mmap-launcher.sh),
rewritten for Basalt's CLI. Basalt reads no profiling environment variables
(checked at the base commit), so unlike the original wrapper there is nothing to
suppress beyond `PWD`. Combine with `--mmap-experts --resident-budget-gib N` when
RAM is tight; the launcher is orthogonal to those flags.

**Not claimed.** Any prefill speedup on Blackwell. rfairburn's own numbers are
ROCm-only and predate the allocator change; this branch adds no measurements.

## Verification

- `dp_plan_test` compiles clean (`-Wall -Wextra -Werror -std=c++17`) and passes
  all checks on this branch. Run via `cmake -S . -B build -DBUILD_TESTING=ON &&
  cmake --build build --target dp_plan_test && ctest --test-dir build -R
  dp_plan_test`, or standalone: `g++ -std=c++17 -Iinclude tests/prefill/dp_plan_test.cpp
  -o /tmp/dp_plan_test && /tmp/dp_plan_test`.
- Everything else is documentation and a shell launcher. **No Blackwell GPU was
  available to this work; nothing here has run on hardware.** Treat the launcher
  as experimental and keep your normal launch path as the rollback, exactly as
  the source recipe instructs for its own output.

## Licence

Basalt, Strata and rfairburn's fork are all MIT. The ported files keep their
provenance in header comments; original copyrights apply (see
[LICENSE](../LICENSE) and [THIRD_PARTY_LICENSES](../THIRD_PARTY_LICENSES)).
