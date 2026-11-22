# Inkling 2 Mini (work in progress)

Branch `inkling2-mini`. Status: **scaffolding, not yet registered or compiled.** Nothing here runs.

- Design: [INKLING2.md](../../INKLING2.md) - relative bias gathered inside the flash-attention
  kernel (no materialized per-head mask, no compute-buffer blowup), quantized KV from day one
  (Basalt's int8 KV kernels), looped core as new kernel work.
- Files so far: `src/models/inkling2/{meta.hpp,adapter.hpp,adapter.cpp,inkling2.cmake}`,
  `tools/convert/checkpoint_inkling2.py` (inventory only).
- First host build (once registered, on the RTX 5090 box, CUDA 13):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2
cmake --build build -j
```

- Registration checklist: `inkling2` in `BASALT_BUILTIN_MODELS` (cmake/models.cmake),
  `if(BASALT_MODEL_INKLING2)` + `include(src/models/inkling2/inkling2.cmake)` in CMakeLists.txt,
  `make_adapter()` implemented, converter section in tools/convert/convert.py,
  dialect `serve/models/inkling2.py` (chat form from the model card - GAP: not yet extracted).
