# QNN backend (Qualcomm Hexagon NPU, Windows on Snapdragon)

An experimental ggml backend that runs matrix multiplications on the Hexagon NPU (HTP)
through QNN (Qualcomm AI Engine Direct). Developed and tested on a Snapdragon X Elite
(X1E80100, HTP v73) laptop running Windows 11 ARM64.

**This is a research backend, not a production accelerator.** Read the status section
before using it. Note that upstream llama.cpp ships an official, Qualcomm-maintained
Hexagon backend (`ggml/src/ggml-hexagon`, built on the Hexagon SDK with custom HVX
kernels). This backend is an independent experiment that uses the QNN runtime instead:
no test-signing, no custom DSP kernels, only the QAIRT community SDK headers at build
time and `QnnHtp.dll` at run time.

## Status

| What | State |
|---|---|
| `test-backend-ops` MUL_MAT (F32/F16 weights) | 45/45 pass |
| Single-matmul kernel throughput (burst clocks + static weights) | 6-11 TFLOP/s fp16, measured |
| Real-model inference | completes, no hangs; unsupported shapes fall back to the CPU automatically |
| End-to-end speed vs the Adreno GPU (OpenCL) or a KleidiAI CPU build | currently loses on dense 9-14B models |
| Decode (single-token) offload | intentionally not claimed (bandwidth-bound, the CPU is better placed) |

The honest summary: the kernels are fast, the eager per-op execution model is robust,
but per-op scheduling and IO copies eat the advantage on real models. The practical value
today is (a) the robustness machinery (load-time shape prevalidation, persistent
failed-shape denylist, watchdog with clean CPU fallback), (b) the measured evidence that
compile-once/load-fast AOT context binaries are the right next step (a serialized context
reloads ~8x faster than a fresh finalize: ~5 ms vs ~41 ms measured), and (c) a bounded static-weight path that bakes
quantized weights to fp16 on the NPU within a memory budget.

## Requirements

- Windows 11 ARM64 on a Snapdragon with an HTP (tested: X Elite / HTP v73).
- Qualcomm AI Runtime (QAIRT) community SDK - headers only at build time.
- `QnnHtp.dll` and its dependencies available at run time via `PATH` or `QNN_SDK_ROOT`
  (the SDK's `lib/aarch64-windows-msvc` directory works).
- Newer QAIRT runtimes (2.45 tested) also need `ADSP_LIBRARY_PATH` set to the SDK's
  `lib/hexagon-v73/unsigned` directory, or the first graph execute dies silently.
  QAIRT 2.34 did not need this.

## Build

```
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release -DGGML_QNN=ON -DQNN_SDK_ROOT=<path-to-qairt>/<version>
cmake --build build
```

The backend composes with the other backends; a combined CPU (KleidiAI) + Adreno GPU
(OpenCL) + NPU (QNN) binary works. With full GPU offload (`-ngl 99`) the QNN backend is
idle by design; with partial offload it takes a bounded slice of the CPU-resident matmuls.

## How it behaves

- `supports_op` builds, finalizes and test-executes a QNN graph for every new shape
  before claiming it, so inference only ever runs graphs proven to execute. Shapes the
  HTP rejects or that wedge it are denylisted and permanently routed to the CPU.
- Model weights are baked into their graphs once (dequantized to fp16 when quantized) in
  HTP-native layout, within a memory budget (default 2048 MB). Weights past the budget
  stay on the CPU. The first time a shape is seen it pays graph build + finalize + bake
  (tens of ms to seconds per shape), so the first prompt is noticeably slower than
  steady state - expected, not a hang.
- Static-weight graphs pad the batch dimension to a bucket (default 512) so one graph
  and one baked weight serve every prompt length.
- Calls that can hang the HTP run under a watchdog; a timeout degrades the whole backend
  to a safe idle state and the model keeps running on the CPU/GPU.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_QNN_DISABLE` | unset | disable the backend entirely |
| `GGML_QNN_MIN_DIM` | 32 | minimum matmul dimension to claim (smaller goes to the CPU) |
| `GGML_QNN_STATIC_BUDGET_MB` | 2048 | cap on baked static-weight bytes, 0 = unlimited |
| `GGML_QNN_NPAD` | 512 | batch-dim bucket floor for static graphs. This is the load-bearing knob for large models: graph execute hangs when a padded IO buffer crosses a runtime-dependent threshold (~1-1.5 MB measured), so keep `max(K, M) * NPAD * 4` bytes under ~1 MB - e.g. `GGML_QNN_NPAD=64` (with a matching `-ub 64`) for model-scale weights |
| `GGML_QNN_NO_OPT` | unset | drop the finalize-optimization flag (matmul graphs; debugging lever) |
| `GGML_QNN_NO_STATIC_WEIGHTS` | unset | disable static weight baking |
| `GGML_QNN_DENYLIST` | unset | file that persists failed shapes across runs |
| `GGML_QNN_NO_PREVALIDATE` | unset | skip the test-execute during shape validation |
| `GGML_QNN_TIMEOUT_MS` | 15000 | watchdog timeout for finalize/execute |
| `GGML_QNN_NO_BURST` | unset | do not lock the HTP to TURBO clocks |
| `GGML_QNN_SHARED_MEM` | unset | use registered fastrpc buffers for graph IO |
| `GGML_QNN_QUANTIZED` | unset | experimental per-execute dequant path (off: quantized weights require static baking) |
| `GGML_QNN_ELEMENTWISE` | unset | experimental ADD/MUL offload (a known HTP broadcast bug makes some shapes wrong, keep off) |
| `GGML_QNN_MIN_ELEMENTS` | 1M | elementwise offload threshold |
| `GGML_QNN_AOT_TEST` | unset | one-shot AOT context-binary round-trip measurement |
| `GGML_QNN_DEBUG` | unset | verbose QNN logging |

## Benchmarking notes

- On Snapdragon X Elite the all-physical-cores thread default costs 2-5x on token
  generation and makes results noisy (44% vs 7% relative stddev, measured on AC power) -
  decode's per-token threadpool barriers pay for oversubscription, prefill shows no
  comparable collapse. Use about half the cores (`-t 6` on the 12-core X1E80100) for any
  decode measurement. Battery behavior has not been cleanly measured and may be worse.
- The NPU is a single-client device: never run two NPU-using processes at once, the HTP
  can wedge and need a device reset.
- Memory is unified (CPU, GPU and NPU share the same LPDDR5x pool and bandwidth); budget
  accordingly.

## Known limitations

- Windows ARM64 only in practice (the dlopen paths exist for Linux but are untested).
- fp16 math internally: F32 elementwise cannot meet strict 1e-7 tolerances (matmul
  tolerances pass).
- A rare crash inside `QnnHtp.dll` at process exit can occur after the results are
  produced (driver teardown behavior).
- The eager per-op model is the wrong long-term architecture; the roadmap is AOT context
  binaries (compile at load, cache to disk, execute only known-good graphs).
