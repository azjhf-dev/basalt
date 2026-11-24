# Preparation validation, 2026-11-24

## Passed

Eight focused C++20 tests compiled with the available GCC 11.4 and ran successfully:

- inkling2_spill: spill plan, staging and memory-reclaim calculations
- inkling2_copy_heat: conversation-based expert ranking and decay
- inkling2_loan_layout: prompt-memory loan packing
- inkling2_expert_map: expert mapping
- tier_policy: expert placement/admission policy
- chunk_plan: prompt-chunk planning
- multi_gpu: mode validation over fake devices
- gpu_check: GPU checks over fake devices

Commands used `g++ -std=c++20 -O2 -Iinclude -Isrc` with each test's source and its
CPU implementation source, as listed in CMakeLists.txt. This is a focused manual
host build, not a successful full CMake or CUDA build.

`python3 -m unittest tools.test_inkling2_deep_ab serve.tests.test_config`:
57 discovered, 56 passed, 1 skipped. Includes three tests for the new helper,
including a complete fake-engine protocol exchange. Both supplied configs also
parsed successfully through the repository's actual config loader.

The model-layering check passed. Python byte-compilation of the new helper passed.
A separate small bench/launcher unittest run discovered 19, passed 2 and skipped
17 because no native binaries exist here. Those skips are not test coverage.

## Not verified

- Full CMake configure stopped: available 3.22.1 is below required 3.24.
- nvcc and GPU are absent. Documented GCC requirement is 13+, while available
  compiler is 11.4. CUDA compilation, device parity, real protocol execution,
  model loading, quality and all performance claims remain unverified.
- A broad server unittest discovery was interrupted at its 120-second limit.
  It has no completion result and is not counted as passed.
- The imported source has an existing trailing-blank-line warning at
  `tools/convert/test_checkpoint.py:721`. It was not edited to keep runtime and
  upstream source fidelity. Added experiment files pass whitespace checking.

## Integration provenance

Runtime/source is the community bundle at the SHA recorded in SOURCE.json.
No local code conflict resolution was needed. Existing fork-main GitHub Actions
workflow files were retained, because the saved repo-scoped token cannot update
workflows. Therefore this is a source snapshot commit, not a merge preserving
all the community branch ancestry. Neither main nor the working QFN fix branch
is an update target. No upstream issue or pull request is created.

For the next gate, run the two CUDA builds and the parity commands in README.md
on the 5060 Ti, then return ABBA logs and real eval results. Do not call this
GPU-tested or assume the 5080 author's throughput transfers to this hardware.
