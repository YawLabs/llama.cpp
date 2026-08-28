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
| End-to-end speed vs the Adreno GPU (OpenCL) or a KleidiAI CPU build | loses to both, measured on a 4B; also loses on dense 9-14B models (see Measured comparison) |
| Decode (single-token) offload | intentionally not claimed; it runs on the CPU, though the Adreno is 25% faster at it (see Measured comparison) |

The honest summary: the kernels are fast, the eager per-op execution model is robust,
but per-op scheduling and IO copies eat the advantage on real models. The practical value
today is (a) the robustness machinery (load-time shape prevalidation, persistent
failed-shape denylist, watchdog with clean CPU fallback), (b) the measured evidence that
compile-once/load-fast AOT context binaries are the right next step (a serialized context
reloads ~8x faster than a fresh finalize: ~5 ms vs ~41 ms measured), and (c) a bounded static-weight path that bakes
quantized weights to fp16 on the NPU within a memory budget.

## Measured comparison

Qwen3-4B-Q4_K_M (2.32 GiB) on a Snapdragon X Elite (X1E80100), Windows 11 ARM64,
`llama-bench -t 6 -p 512 -n 128 -r 5`, one build per backend from `b840f5720`. Six
counterbalanced legs (CPU, GPU, NPU, NPU, GPU, CPU), 120 s cooldowns, each backend's pair
averaged. Measured on AC with a settled pack - 100% charge drawing 4.6 W - which matters
more than it sounds; see the power note under "Benchmarking notes". The NPU leg runs
`-ub 64` with `GGML_QNN_NPAD=64`.

| Backend | pp512 t/s | tg128 t/s |
|---|---:|---:|
| Adreno X1-85 (OpenCL) | 228.1 | 20.0 |
| CPU (KleidiAI) | 132.2 | not reported |
| Hexagon NPU (QNN) | 116.9 | 19.8 |

The GPU takes prefill by 1.73x over the CPU and 1.95x over the NPU. The NPU loses prefill to
the CPU by 12%. NPU decode tracks the CPU because the backend does not claim decode - that
leg is the CPU path with the QNN backend registered.

**CPU decode is deliberately not reported.** Its two counterbalanced legs came back 24.76 and
20.01 t/s - a 21% pair spread, against 0.2% for the GPU and 0.4% for the NPU on the same run -
and a separate pair of probe runs at 13-20% charge gave 12.66 and 22.91. So it is not an
artifact of one power state: the metric is unstable settled and unstable depleted. The
instability is reproducible and specific to CPU decode; no single figure would be honest, and
averaging the pair would hide that rather than express it.

The other five pairs agree within 2% (GPU 0.6% / 0.2%, NPU 1.8% / 0.4%, CPU prefill 1.6%),
which is the counterbalanced design reporting its own cleanliness. NPU prefill is the noisiest
surviving number at 5.5-7.4% relative stddev between repetitions against 0.2-1.3% for the GPU.
Per-op scheduling and first-hit bake sit on the critical path, which is the case for AOT
context binaries.

These supersede an earlier table that had the CPU at 116.4 / 16.0 and the NPU at 101.5 / 15.8,
and that claimed the GPU took decode by 25%. Two things changed. The GPU reproduced to +0.3%,
but both CPU-thread-bound backends came back higher on prefill - CPU by 13.5%, NPU by 15.2% -
tracking a CPU clock that averaged 85.6% of base on this run against 74.9% on the earlier one; the earlier run's
pack state was never recorded, so the cause cannot now be established - only that the clocks
were lower. And the GPU's 25% decode lead rested on that CPU figure of 16.0: CPU decode does
not reproduce there, and even the low leg of its unstable pair lands at 20.01, level with the
GPU's 20.03. On this run the three engines converge on decode (GPU 20.03, NPU 19.75, CPU
20.01-24.76) while prefill separates them 228 / 132 / 117 - the shape of bandwidth-bound
decode against compute-bound prefill. The earlier "the engine still matters for decode"
reading is withdrawn. The NPU is not claimed for decode and so is not measured for it - that
column says nothing about what the HTP would do.

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

This fork carries a build fix the CPU backend needs on Windows ARM64. KleidiAI's assembly
selects armasm syntax on `_MSC_VER` and GNU syntax otherwise, but clang targeting
`*-windows-msvc` defines `_MSC_VER` while assembling GNU syntax into COFF, so it matches
neither branch. Upstream #26077 started compiling those files, which broke the build
outright. `ggml/src/ggml-cpu/kleidiai/kleidiai-patch-coff-asm.cmake` adds the missing
branch to the fetched sources at configure time. It is idempotent and becomes a no-op once
upstream fixes it. Build with `-DGGML_CPU_KLEIDIAI=OFF` to skip the whole question.

The backend composes with the other backends; a combined CPU (KleidiAI) + Adreno GPU
(OpenCL) + NPU (QNN) binary works. With full GPU offload (`-ngl 99`) the QNN backend is
idle by design; with partial offload it takes a bounded slice of the CPU-resident matmuls.

## How it behaves

- `supports_op` builds, finalizes and test-executes a QNN graph for every new shape
  before claiming it, so inference only ever runs graphs proven to execute. Shapes the
  HTP rejects or that wedge it are denylisted and permanently routed to the CPU.
  Weights probed before their data is resident (llama's model-load pass) are answered
  from the type and shape policy alone - a trial build needs the real bytes - and the
  HTP trial happens on the first scheduled node instead.
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
  decode measurement. That fixes the oversubscription collapse but does not make CPU decode
  stable: at `-t 6` on a settled pack it still ran 13.8-16.7% relative stddev within a leg
  and 21% between counterbalanced legs. Treat the 7% above as the best case, not the
  expectation, and see "Measured comparison" for why no CPU tg128 figure is quoted.
- The NPU is a single-client device: never run two NPU-using processes at once, the HTP
  can wedge and need a device reset.
- Memory is unified (CPU, GPU and NPU share the same LPDDR5x pool and bandwidth); budget
  accordingly.
- A back-to-back sweep cannot rank backends on this machine. Repeating the first leg at the
  end of a four-leg sweep came back 26% low on AC and 43% low on battery. On the AC run the
  clock slid from 87% to 69% of base across the four legs; on battery, 43% to 33%.
  Counterbalance the order (CPU, GPU, NPU, NPU, GPU, CPU) and cool down 120 s between legs,
  then average each backend's pair. That reproduces prefill for every backend within 2% and
  decode for the GPU and NPU within 0.4%. CPU decode does not converge even counterbalanced -
  a 21% pair spread - so do not quote a single CPU tg128 figure from one sweep.
- Battery is not just slower, it is differently shaped. A position-matched pass on DC
  measured the GPU 1.6x down but the NPU 3.5x down (pp512 27.7 vs 95.8 t/s).
- AC alone is not enough: the pack must also be SETTLED. A deeply discharged pack on AC runs
  CPU prefill at about half speed - 58.1 and 57.9 t/s measured at 13-20% charge against 132.2
  settled. Whether the GPU escapes this is UNMEASURED - the low-charge probe ran CPU and NPU
  legs only, so treat the requirement below as applying to every backend until someone runs a
  GPU leg on a depleted pack. Recovery is largely done by 33-42%, but not complete: single
  legs there returned 124.5 t/s at 33% and 112.6 at 41.6%, still 6% and 15% under the settled
  132.2, and non-monotonic between the two. Require >40% charge and a charge draw under 5 W before
  measuring, and record charge percent and draw per sample so a suspect run stays legible
  afterwards; "plugged in" on its own will silently halve CPU-bound and NPU-bound prefill
  while looking correct. Charge draw also collapses mid-leg under load (32 W to 1.1 W inside
  a single CPU leg), so a reading taken before the leg does not describe the leg.
- Throughput cannot tell you which engine ran. Verify GPU placement with PID-filtered
  `\GPU Engine(*)\Utilization Percentage` (about 95% under load against a ~2% idle
  baseline); `--device`, `--list-devices` and reported free memory have all failed to catch
  a silent CPU fallback.

## Known limitations

- Windows ARM64 only in practice (the dlopen paths exist for Linux but are untested).
- fp16 math internally: F32 elementwise cannot meet strict 1e-7 tolerances (matmul
  tolerances pass).
- A rare crash inside `QnnHtp.dll` at process exit can occur after the results are
  produced (driver teardown behavior).
- The eager per-op model is the wrong long-term architecture; the roadmap is AOT context
  binaries (compile at load, cache to disk, execute only known-good graphs).
