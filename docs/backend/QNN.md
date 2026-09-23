# QNN backend (Qualcomm Hexagon NPU, Windows on Snapdragon)

An experimental ggml backend that runs matrix multiplications on the Hexagon NPU (HTP)
through QNN (Qualcomm AI Engine Direct). Developed and tested on a Snapdragon X Elite
(X1E80100, HTP v73) laptop running Windows 11 ARM64.

**This is a research backend, not a production accelerator.** Read the status section
before using it. Note that upstream llama.cpp ships an official, Qualcomm-maintained
Hexagon backend (`ggml/src/ggml-hexagon`, built on the Hexagon SDK with custom HVX
kernels). This backend is an independent experiment that uses the QNN runtime instead:
no test-signing, no custom DSP kernels, only the QAIRT community SDK headers at build
time and the QAIRT runtime DLLs (`QnnHtp.dll` and its dependencies) at run time, plus
`ADSP_LIBRARY_PATH` on QAIRT 2.45 and newer - see Requirements.

## Status

| What | State |
|---|---|
| `test-backend-ops` MUL_MAT (F32/F16 weights) | 47/47 pass via `ctest -R test-backend-ops-qnn`; one 46/47 flake seen in five runs, undiagnosed |
| Single-matmul kernel throughput (burst clocks + static weights) | no figure claimed: the August 2026 numbers were taken without recording box load and have not been re-taken |
| Real-model inference | completes, no hangs; unsupported shapes fall back to the CPU automatically |
| End-to-end speed vs the Adreno GPU (OpenCL) or a KleidiAI CPU build | no valid NPU figure yet: the one 4B sweep's NPU leg never executed a matmul on the HTP (see the retraction under Measured comparison); unmeasured on 9-14B |
| Decode (single-token) offload | intentionally not claimed below 32-token ubatches (`GGML_QNN_MIN_DIM`), so it runs on the CPU; on the settled-pack run the two measured engines (GPU, CPU) and the CPU-run leg labelled NPU converge on decode (see Measured comparison) |

The honest summary: the kernels are fast, the eager per-op execution model is robust,
and per-op scheduling and IO copies are expected to eat the advantage on real models -
expected, not yet measured end to end; see the retraction below. The practical value
today is (a) the robustness machinery (load-time shape prevalidation, persistent
failed-shape denylist, watchdog with clean CPU fallback), (b) a working reload primitive
showing that compile-once/load-fast AOT context binaries are the right next step (a
serialized context is retrieved with `graphRetrieve` instead of being finalized again; how
much time that saves is unmeasured), and (c) a bounded static-weight path that bakes
quantized weights to fp16 on the NPU within a memory budget.

Kernel throughput is not claimed. The single-matmul harness figures this section used to
quote were taken in August 2026 without recording what else was running on the machine -
the same flaw that invalidated the model-run timings retracted below - so they are
withdrawn rather than repeated here. The two mechanisms behind them are still in the code:
a DCVS TURBO power config, and baking a weight once into the HTP-native layout instead of
re-tiling it on every execute. What either is worth is an open question, and the
single-matmul harness can answer it on a machine with nothing else running.

## Measured comparison

Qwen3-4B-Q4_K_M (2.32 GiB) on a Snapdragon X Elite (X1E80100), Windows 11 ARM64, QAIRT
2.45.0.260326, `llama-bench -t 6 -p 512 -n 128 -r 5`, one build per backend from
`b840f5720`. The CPU legs and the "NPU" leg ran the KleidiAI+REPACK build; the GPU leg's
`-ngl` was not recorded. Six counterbalanced legs (CPU, GPU, NPU, NPU, GPU, CPU), 120 s
cooldowns, each backend's pair averaged. Measured on AC with a settled pack - 100% charge
drawing 4.6 W - which matters more than it sounds; see the power note under "Benchmarking
notes". The NPU leg runs `-ub 64` with `GGML_QNN_NPAD=64`.

| Backend | pp512 t/s | tg128 t/s |
|---|---:|---:|
| Adreno X1-85 (OpenCL) | 228.1 | 20.0 |
| CPU (KleidiAI) | 132.2 | not reported |
| CPU (NPU leg never executed a matmul, see retraction) | 116.9 | 19.8 |

**Retraction (2026-09-16): the NPU row is a CPU number.** A re-run of the NPU-leg
configuration (`llama-bench -t 6 -ub 64`, `GGML_QNN_NPAD=64`, `GGML_SCHED_DEBUG=2`, the
KleidiAI+REPACK build, QAIRT 2.45.0.260326) showed what that leg actually did.
`load_tensors` put the whole Qwen3-4B-Q4_K_M model in `CPU_REPACK` (2375 MiB); the QNN
backend's `supports_buft` refuses that buffer type, so no scheduler split can land on the
NPU. The scheduler's upgrade pass still trial-built `attn_q` (`MUL_MAT_q4_K_f32_2560x4096`
at N=64, output exactly 1 MiB) from the repacked bytes: finalize passed in 29 ms, the
validation execute timed out at the 15 s watchdog, the session degraded, and 22 of 22
splits ran on the CPU. Every NPU figure in this table is therefore a CPU number taken
after a 15 s stall in warmup. The runtime setup is not the cause: the lifecycle modelscale
shapes (largest output 655 KB) executed correctly one minute later in the same
environment, so the hang is the IO-size law described under `GGML_QNN_NPAD`.
Two gates now keep it from recurring: `supports_op` refuses any source that sits in a non-host buffer (`CPU_REPACK` is one) before it builds anything, so the upgrade pass no longer trial-builds from repacked bytes, and `GGML_QNN_IO_MAX_KB` refuses a padded IO of 1 MiB before a graph is created.
The numbers stay because they are what was measured; they do not measure the NPU.

To put K-quant weights on the NPU, run with `-dev QNN`: llama-model-loader then probes
the QNN device first and the weight resolves to a plain CPU buffer, bypassing repack.
`llama-bench` has no `--no-repack`, so without `-dev QNN` a K-quant model on a
KleidiAI+REPACK build cannot reach the NPU at all. F16/F32 weights stay in the plain CPU
buffer and do reach the NPU. See "Routing K-quant weights to the NPU" for the caveats (`GGML_QNN_NPAD` must be set as well).

Read against that: the GPU takes prefill by 1.73x over the CPU leg and 1.95x over the
"NPU" leg, and the "NPU" leg is 12% under the CPU leg - consistent with `-ub 64` on a CPU
run, not an NPU result. Its decode tracks the CPU because decode is not claimed anyway:
the backend does not claim matmuls below 32-token ubatches (`GGML_QNN_MIN_DIM`), which
covers single-stream decode, but a server decoding 32 or more slots per step is claimed
and executes the padded graph.

**CPU decode is deliberately not reported.** Its two counterbalanced legs came back 24.76 and
20.01 t/s - a 21% pair spread, against 0.2% for the GPU and 0.4% for the "NPU" leg on the same run -
and a separate pair of probe runs at 13-20% charge gave 12.66 and 22.91. So it is not an
artifact of one power state: the metric is unstable settled and unstable depleted. The
instability is reproducible and specific to CPU decode; no single figure would be honest, and
averaging the pair would hide that rather than express it.

The other five pairs agree within 2% (GPU 0.6% / 0.2%, "NPU" leg 1.8% / 0.4%, CPU prefill 1.6%),
which is the counterbalanced design reporting its own cleanliness. The "NPU" prefill is the
noisiest surviving number at 5.5-7.4% relative stddev between repetitions against 0.2-1.3% for
the GPU - and per the retraction that leg was a CPU run at `-ub 64`, so its noise says nothing
about per-op scheduling or first-hit bake on the HTP.

**Do not read those pair spreads as the uncertainty on the table.** A pair spread measures
whether the two legs of ONE run agree with each other. It says nothing about whether a second
run reproduces the number, and on this machine the two differ by an order of magnitude. The
same command on a second settled-pack run (CPU-only, not interleaved with the GPU and NPU
legs) returned CPU pp512 121.88 against 132.15 here - 8.1% apart - and CPU tg128 16.78 against
the 20.01-24.76 pair here - 16-32% under either leg - while both runs agreed with themselves
to under 2.2%. That comparison bounds the figure rather than measuring it cleanly, because
the two runs differed in workload context as well as in being separate runs.

So quotability is per METRIC, not per run. Prefill at ~8% is defensible from one controlled
run. Decode is not: quote a range or re-measure across three or more runs.

**The GPU number reproduces; there is no NPU number to reproduce.** Three sweeps have now measured
GPU pp512 - the superseded table at 227.4, this one at 228.1, and a third at 223.3 - agreeing
within 2.1%, and the GPU is also the tightest leg within a run (+-0.49 on 226.9, 0.2%). The
third sweep picked up foreign CPU load partway through and is not clean, which makes the GPU
agreement more informative rather than less: it held across a run whose CPU-bound legs were
visibly disturbed, consistent with the GPU not leaning on the CPU threads.

That same third sweep also shows why the NPU legs never agreed among themselves. Its pp512
came back 39% below this table, but two of the six legs ran at 62-64% total CPU where a
`-t 6` workload predicts ~50%, and the "NPU" leg runs six CPU threads - so the drop is what
contamination predicts and cannot be separated from genuine between-run variation. The
superseded table's 101.5 is no help
either, since that run's pack state was never recorded. And per the retraction, the table's
NPU leg never reached the HTP, and the earlier NPU legs were not verified to reach it either -
throughput cannot tell which engine ran - so "does the NPU number reproduce" is the wrong
question until a `-dev QNN` sweep produces one.

These supersede an earlier table that had the CPU at 116.4 / 16.0 and the NPU at 101.5 / 15.8,
and that claimed the GPU took decode by 25%. Two things changed. The GPU reproduced to +0.3%,
but both CPU-thread-bound backends came back higher on prefill - CPU by 13.5%, NPU by 15.2% -
tracking a CPU clock that averaged 85.6% of base on this run against 74.9% on the earlier one; the earlier run's
pack state was never recorded, so the cause cannot now be established - only that the clocks
were lower. And the GPU's 25% decode lead rested on that CPU figure of 16.0: CPU decode does
not reproduce there, and even the low leg of its unstable pair lands at 20.01, level with the
GPU's 20.03. On this run the two measured engines and the CPU-run leg labelled NPU converge on decode (GPU 20.03, "NPU" leg 19.75, CPU
20.01-24.76) while prefill separates them 228 / 132 / 117 - the 117 being the CPU at `-ub 64`,
per the retraction - the shape of bandwidth-bound decode against compute-bound prefill. The
earlier "the engine still matters for decode" reading is withdrawn. The NPU is not claimed
for decode and so is not measured for it - that column says nothing about what the HTP would
do, and per the retraction above neither does its prefill entry.

## Requirements

- Windows 11 ARM64 on a Snapdragon with an HTP (tested: X Elite / HTP v73).
- Qualcomm AI Runtime (QAIRT) community SDK - headers only at build time.
- `QnnHtp.dll` and its dependencies available at run time via `PATH`, or via `QNN_SDK_ROOT`,
  under which the backend tries the SDK's `lib/aarch64-windows-msvc` and then
  `lib/arm64x-windows-msvc` directory. `QNN_SDK_ROOT` is also read from the environment at
  configure time when `-DQNN_SDK_ROOT` is not given.
  A non-Windows aarch64 build loads `libQnnHtp.so` the same way and tries `lib/aarch64-android`, `lib/aarch64-ubuntu-gcc9.4`, `lib/aarch64-oe-linux-gcc11.2` and `lib/aarch64-oe-linux-gcc9.3` (untested, see Known limitations).
- Newer QAIRT runtimes (2.45 tested) also need `ADSP_LIBRARY_PATH` set to the SDK's
  `lib/hexagon-v73/unsigned` directory, or the first graph execute dies silently.
  QAIRT 2.34 did not need this.
  On Windows the backend logs a WARN that names the path when the variable is unset.

## Build

```
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release -DGGML_QNN=ON -DQNN_SDK_ROOT=<path-to-qairt>/<version>
cmake --build build
```

On Windows, enable long paths for the checkout before building: `git clone -c
core.longpaths=true ...`, or `git config core.longpaths true` in an existing clone. Some
`tools/ui` and `tools/server/webui` paths exceed MAX_PATH, and without it git reports
`Filename too long` and leaves those files out of the working tree.

This fork carries a build fix the CPU backend needs on Windows ARM64. KleidiAI's assembly
selects armasm syntax on `_MSC_VER` and GNU syntax otherwise, but clang targeting
`*-windows-msvc` defines `_MSC_VER` while assembling GNU syntax into COFF, so it matches
neither branch. Upstream #26077 started compiling those files, which broke the build
outright. `ggml/src/ggml-cpu/kleidiai/kleidiai-patch-coff-asm.cmake` adds the missing
branch to the fetched sources at configure time. It is re-runnable (guarded by the
`elif defined(_WIN32)` sentinel), runs on every Windows toolchain, and rewrites the fetched
KleidiAI tree in place - including a `FETCHCONTENT_SOURCE_DIR_KLEIDIAI` checkout. The added
branch is inert under armasm64 (cl.exe, clang-cl) and active for clang's GNU driver and
llvm-mingw. Build with `-DGGML_CPU_KLEIDIAI=OFF` to skip the whole question.

The backend composes with the other backends; a combined CPU (KleidiAI) + Adreno GPU
(OpenCL) + NPU (QNN) binary works. With full GPU offload (`-ngl 99`) the QNN backend is
idle by design; with partial offload it takes a bounded slice of the CPU-resident matmuls.

## How it behaves

- `supports_op` builds, finalizes and test-executes a QNN graph for every new shape
  before claiming it, so inference only ever runs graphs proven to execute. Shapes the
  HTP rejects or that wedge it are denylisted and routed to the CPU for the rest of the process; what is also written to the `GGML_QNN_DENYLIST` file is set out in the denylist bullet below.
  Weights probed before their data is resident (llama's model-load pass) are answered
  from the type and shape policy alone - a trial build needs the real bytes - and the
  HTP trial happens on the first scheduled node instead. A quantized weight is the
  exception: its only path is the static bake, which needs the bytes in a buffer the
  caller tags WEIGHTS, so it is claimed unallocated only when the probe carries the
  destination buffer (llama hangs a dummy one on the weight for exactly this). A
  quantized tensor probed with no buffer at all - a graph tensor, as in
  `test-backend-ops` - is refused, because nothing will tag it and the bake cannot happen.
- Before any of that, `supports_op` does the checks that are arithmetic on the shape and need no session. A matmul whose padded IO would reach `GGML_QNN_IO_MAX_KB`, or overflow the 32-bit sizes QNN takes, is reported as not supported: it runs on the CPU and is never a failed op (a claimed node that the graph policy then refuses does not fall back, it fails the graph).
  A weight probed for placement at load has no real batch yet, so the smallest bucket (the `GGML_QNN_NPAD` floor) is used: a weight capped even there is not placed on the NPU, so on a KleidiAI build it keeps `CPU_REPACK` instead of landing in a plain CPU buffer it would never leave. At the default `GGML_QNN_NPAD=512` that is every weight 512 or more wide, i.e. all of a model-scale model.
  The first IO-cap refusal that a setting can fix is printed once per process, straight to stderr, and names the `GGML_QNN_NPAD` and `-ub` that fit that shape. A shape that fits only below `GGML_QNN_MIN_DIM` (a 151936-wide output layer, a 9728-wide FFN) can never run here, so it stays at DEBUG and does not use up that one notice; later refusals are DEBUG.
  A source in a non-host buffer is refused as well, see "Routing K-quant weights to the NPU". An exception inside the trial build (host out of memory) is an ERROR and the op is not claimed.
- Model weights are baked into their graphs once (dequantized to fp16 when quantized) in
  HTP-native layout, within a memory budget (default 1024 MB, a conservative choice: in one run on a
  machine loaded with other work the HTP stopped mapping baked weights at about 1170 MiB
  committed, which has not been reproduced on an idle machine). Weights past the budget
  stay on the CPU. If a static graph still fails to build on a shape that already built and
  validated in the session, that is NPU weight memory running out, not a verdict on the
  shape: the budget is clamped to what is committed, later weights stay on the CPU, nothing
  is denylisted and the session keeps the graphs it has (one WARN, `budget_clamped` in
  `GGML_QNN_STATS`). Static graph keys carry a fingerprint of the weight contents, so a
  different weight landing at a reused address does not hit a stale graph. Each static
  bake pays graph build + finalize + bake (tens of ms to seconds per shape), and that is
  paid at context creation: the scheduler's reserve pass at `n_ubatch` trial-builds every
  resident weight, so `llama_init_from_model` is slower than steady state - expected, not
  a hang. It is re-paid per context, since the session and its baked weights are freed
  with the last backend instance. Only dynamic graphs (LoRA, pooling) are built on the
  first prompt that needs them.
  A trainable weight (`GGML_TENSOR_FLAG_PARAM`, which `llama_opt_init` sets for finetuning) is never baked: the optimizer updates it in place, so it takes the dynamic path and is copied on every execute.
  If the host cannot allocate the staging copy of a weight, the shape is a policy reject (ERROR `out of host memory staging the weight of ...`): no QNN graph is created and nothing is denylisted.
  The fingerprint (the `_h` suffix of a graph key in the log) is not stable across builds; shape keys and the denylist file format are.
- Static-weight graphs pad the batch dimension to a bucket (default 512, `GGML_QNN_NPAD`),
  so one graph and one baked weight serve every N up to NPAD. Larger ubatches get one
  graph and one bake per power-of-two bucket, each charged to the budget, and a bucket
  whose padded IO would reach `GGML_QNN_IO_MAX_KB` is refused. Dynamic graphs are
  bucket-padded the same way.
- Calls that can hang the HTP run under a watchdog; a timeout degrades the whole backend
  to a safe idle state and the model keeps running on the CPU/GPU. There are two limits:
  `GGML_QNN_BUILD_TIMEOUT_MS` (120 s) bounds graph finalize only, which is a compile, and
  `GGML_QNN_TIMEOUT_MS` (15 s) bounds every execute, the validation execute included. The
  validation execute exists to predict compute-time behavior, so a graph that cannot
  validate inside the compute limit is not claimed; otherwise the first `llama_decode`
  fails instead of falling back. Every timeout log line names the phase, the limit and the
  variable that raises it.
  On a device that is only slow, raising `GGML_QNN_TIMEOUT_MS` is not enough: the validation then completes, takes longer than `GGML_QNN_SLOW_EXEC_MS` and is refused as too slow, so that variable must be raised as well or set to 0. The validation-timeout ERROR says so.
  If the watchdog thread itself cannot be started, the session degrades with that reason (`cannot start the watchdog thread`, or `out of host memory starting the watchdog thread`), the shape is refused and stays off the denylist.
- An execute that completes but takes `GGML_QNN_SLOW_EXEC_MS` (2 s) or longer
  marks the NPU as running far below normal speed: the session stops claiming ops, nothing
  is denylisted (it is a device state, not a verdict on the shape) and the model runs on
  the CPU. A healthy HTP executes any shape under the IO cap in milliseconds.
  The check runs on the validation execute and on every compute-time execute; the second catches a device that slows down after its graphs validated (a cached graph is never re-validated). At compute time the output is valid, so the node succeeds, and the WARN names the op.
  This degrade is slow-only: nodes the scheduler already placed on graphs that built keep executing on the NPU instead of failing their batch, until the scheduler next splits a graph. Nothing new is built, so a placed node whose graph does not exist yet fails, and any later hard failure (timeout, execute error) ends the exemption.
  All of this needs the execute to finish inside `GGML_QNN_TIMEOUT_MS`: past it a slow device looks the same as a wedge.
- The denylist has two levels. Every failed shape is remembered for the process. Only an HTP verdict on the shape, which is what a rerun must skip, is also written to the `GGML_QNN_DENYLIST` file:
  a finalize error; a finalize timeout; a compute-time execute timeout; and a validation-execute timeout, but only when the session has already seen a validation execute complete under `GGML_QNN_SLOW_EXEC_MS` (under 2 s when that check is set to 0).
  That last condition exists because a timeout cannot tell a wedge from a device that is merely slow or a machine that is busy, and either used to ban healthy shapes for every later run.
  No timeout is written once the static budget is clamped: a failed graph then sits in the context and can hang the next execute, whatever its shape.
  A graph create, tensor or node failure and an execute error return (validation or compute time) can be a transient driver state and are remembered for the process only.
  Policy rejects (IO cap, budget, a proven shape failing at the budget clamp, host out of memory, no watchdog thread) are never denylisted.
  Nothing removes an entry from the file. Run `test-qnn-health` before a run that has `GGML_QNN_DENYLIST` set, and if a run on a slow or busy machine wrote to the file, delete the file (or the entry) once `test-qnn-health` passes with nothing else running.

## Routing K-quant weights to the NPU

On a KleidiAI build with `GGML_CPU_REPACK` (the default), `load_tensors` places K-quant
weights in the `CPU_REPACK` buffer type. The QNN backend's `supports_buft` refuses that
type, so those weights never reach the NPU. The scheduler's upgrade pass used to trial-build a graph from the repacked bytes all the same, which is how the retracted sweep stalled; `supports_op` now refuses any source in a non-host buffer (`CPU_REPACK` has no `is_host`) before it builds anything, so neither the trial build nor the stall recurs.
Pass `-dev QNN`: llama-model-loader then probes the QNN device first, the weight resolves
to a plain CPU buffer, bypassing repack, and the static bake sees the real bytes.
`llama-bench` has no `--no-repack`, so `-dev QNN` is the only route there. F16/F32 weights
stay in the plain CPU buffer regardless and reach the NPU without it.

`-dev QNN` is not enough at the default `GGML_QNN_NPAD=512`. The placement probe refuses every weight whose padded IO reaches `GGML_QNN_IO_MAX_KB` at the smallest bucket, which at 512 is every weight 512 or more wide, so a model-scale model stays in `CPU_REPACK` and nothing runs on the NPU. The log then carries one WARN that names the `GGML_QNN_NPAD` and `-ub` that fit the first refused shape. Set both by weight width, see the `GGML_QNN_NPAD` row.

Measured on 2026-09-16 (Qwen3-4B-Q4_K_M, `-t 6 -p 32 -ub 32 -dev QNN`, `GGML_QNN_NPAD=32`,
QAIRT 2.45.0.260326). Placement works: with the watchdog limits raised by hand (to repeat this on a device that slow, raise `GGML_QNN_TIMEOUT_MS` past the slowest execute and set `GGML_QNN_SLOW_EXEC_MS=0` as well; with the timeout alone the completed validation is refused as too slow), 81 static graphs
built and validated inside the 1024 MB budget and the scheduler put 81 matmuls on the NPU
in 61 splits (all four attention projections of about 20 layers). At `-ub 32` with
`NPAD=32` the attention projections (up to 4096 wide, 512 KiB padded IO) fit under
`GGML_QNN_IO_MAX_KB`, while the FFN projections (9728 wide, 1216 KiB) do not and stay on
the CPU. In one run with the budget at its old 2048 MB default, on the same loaded
machine, the HTP could no longer map weights at about 1170 MiB committed (finalize 6020,
then an execute 6002, on shapes that had built dozens of times). That has not been
reproduced on an idle machine, and memory pressure from the other work may have caused it;
the default is 1024 MB as a conservative margin, and such a failure clamps the budget
instead of degrading the session either way.

Throughput was not measured. Every timing taken from 2026-09-16 to 2026-09-18 ran while
other heavy work (builds, test suites, benchmarks from other sessions) was loading the
machine, so none of it is a valid measurement of the NPU, and this document draws no
conclusion from it about the NPU's speed or about any cause. Under that load, graphs with
FP16 weights (K-quant weights are baked to fp16) took seconds per execute while graphs with
F32 weights took milliseconds; whether that holds with nothing else running is untested.
End to end, a six-graph run under the same conditions completed with correct results.
Before any speed claim, run `test-qnn-health` with nothing else running on the machine.
That refusal needs the execute to finish inside `GGML_QNN_TIMEOUT_MS` (15 s). In this state the 335 M-MAC projection does not, so it takes the timeout path and reads as a wedge. A validation timeout goes to the `GGML_QNN_DENYLIST` file only when the session has already seen one validation at normal speed, and a compute-time timeout always does, so a device that is slow from the start writes nothing, but one that slows down mid-run can. Once `test-qnn-health` passes again, delete the file or the entry.

Caveat: `-dev QNN` also makes `llama_context` create two QNN backend instances on one
session - one from the device loop and one from the ACCEL loop. This works, but it was not
anticipated.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_QNN_DISABLE` | unset | disable the backend entirely |
| `GGML_QNN_MIN_DIM` | 32 | minimum matmul dimension to claim (smaller goes to the CPU) |
| `GGML_QNN_STATIC_BUDGET_MB` | 1024 | cap on baked static-weight bytes, 0 = unlimited. 1024 as a conservative margin: weight mapping failed at about 1170 MiB committed in one run on a machine loaded with other work, not reproduced on an idle one; a build failure on a shape that already built in the session clamps the budget to the committed bytes for the rest of the session |
| `GGML_QNN_NPAD` | 512 | batch-dim bucket floor for static and dynamic graphs. This is the load-bearing knob for large models: graph execute hangs when a padded IO buffer crosses a runtime-dependent threshold (~1-1.5 MB measured; exactly 1 MiB hung on QAIRT 2.45). `GGML_QNN_IO_MAX_KB` now rejects such a shape to the CPU instead of hanging, so a too-large NPAD costs offload rather than a watchdog stall. `NPAD * max(K, M) * 4` bytes must stay UNDER the cap (equal is refused), with a matching `-ub`: 64 with `-ub 64` for weights up to 2560 wide; 32 with `-ub 32` for 4096-wide projections such as Qwen3-4B attention, where 64 lands exactly on 1 MiB. Some projections fit no usable bucket: the 9728-wide Qwen3-4B FFN needs 26 or less (16 as a power of two), which is below the default `GGML_QNN_MIN_DIM`, so it stays on the CPU. At the default 512 every weight 512 or more wide is refused, at placement too; the first refusal of a process is a WARN that names the `GGML_QNN_NPAD` and `-ub` that fit that shape |
| `GGML_QNN_IO_MAX_KB` | 1024 | policy-reject a matmul whose padded input or output would reach this many KB (strict less-than passes); the shape runs on the CPU instead of hanging the HTP, and as a policy reject it is not denylisted. `supports_op` applies it before it claims, so such a matmul is "not supported", never a failed op, and a weight capped at the smallest bucket is not placed on the NPU at load |
| `GGML_QNN_NO_OPT` | unset | drop the finalize-optimization flag (matmul graphs; debugging lever); also bypasses denylist entries loaded from the file |
| `GGML_QNN_NO_STATIC_WEIGHTS` | unset | disable static weight baking |
| `GGML_QNN_DENYLIST` | unset | file that persists HTP verdicts on shapes across runs: finalize errors and watchdog timeouts, with the exceptions in the denylist bullet under "How it behaves". Every other failure is remembered for the process only. It starts with the header line `# ggml-qnn denylist v1`. Nothing removes an entry: delete the file, or the entry, once `test-qnn-health` passes with nothing else running, if a run on a slow or busy machine wrote to it |
| `GGML_QNN_NO_PREVALIDATE` | unset | skip the test-execute during shape validation |
| `GGML_QNN_BUILD_TIMEOUT_MS` | 120000 | watchdog timeout for graph finalize only, which is a compile and may legitimately take long. It does not bound the validation execute |
| `GGML_QNN_TIMEOUT_MS` | 15000 | watchdog timeout for every graph execute, the validation execute included (a healthy execute is ms). On a device that is only slow, raising it is not enough: `GGML_QNN_SLOW_EXEC_MS` must be raised as well or set to 0, or the completed validation is refused as too slow |
| `GGML_QNN_SLOW_EXEC_MS` | 2000 | an execute at or over this many ms, the validation execute or a compute-time one, marks the NPU as too slow to use: the session claims nothing more and falls back to the CPU, nothing is denylisted. At compute time the node still succeeds, and nodes already placed on built graphs keep running (a slow-only degrade) until a hard failure. 0 disables (the lifecycle tests do, `test-qnn-health` covers speed) |
| `GGML_QNN_NO_BURST` | unset | do not lock the HTP to TURBO clocks |
| `GGML_QNN_SHARED_MEM` | unset | use registered fastrpc buffers for graph IO; host buffers are used when the init-time self-test fails (`shm_selftest` in `GGML_QNN_STATS` tells why) |
| `GGML_QNN_QUANTIZED` | unset | experimental per-execute dequant path (off: quantized weights require static baking) |
| `GGML_QNN_ELEMENTWISE` | unset | experimental ADD/MUL offload (a known HTP broadcast bug makes some shapes wrong, keep off) |
| `GGML_QNN_MIN_ELEMENTS` | 1M | elementwise offload threshold |
| `GGML_QNN_AOT_TEST` | unset | one-shot AOT context-binary round-trip measurement |
| `GGML_QNN_DEBUG` | unset | verbose QNN logging; pass `-v`/`--verbose` on the host tool or nothing prints |
| `GGML_QNN_STATS` | unset | file that receives fourteen `name value` lines when the session is freed or degrades: the eight original counters plus `burst_applied`, `budget_clamped` (0/1), `exec_count`, `exec_slow` (compute-time executes of 1 s or more), `exec_max_ms` and `shm_selftest`, the init-time shared-memory self-test of the last session (0 = the fastrpc library is absent, 1 = present but the self-test failed, 2 = ok, 3 = not requested, `GGML_QNN_SHARED_MEM` unset) |
| `GGML_QNN_FAIL_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: the VALIDATION execute of every graph whose key contains it fails before QNN is called. Compute-time executes are never touched, and it has no effect under `GGML_QNN_NO_PREVALIDATE`. `GGML_QNN_FAIL_EXECUTE_SKIP=<n>` skips the first n matches, which validate normally. Never set either outside the test suite |
| `GGML_QNN_DELAY_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: one execute of a matching graph, validation or compute-time alike, sleeps `GGML_QNN_DELAY_EXECUTE_MS` inside the timed call right before `graphExecute`, so the delay counts toward `GGML_QNN_TIMEOUT_MS` and `GGML_QNN_SLOW_EXEC_MS` and reaches the slow and timeout verdicts on a healthy device. Unlike a real wedge the delayed call then completes. Never set outside the test suite |
| `GGML_QNN_DELAY_EXECUTE_MS` | 0 | test-only: the delay in ms for `GGML_QNN_DELAY_EXECUTE`, capped at a day; 0 leaves the hook off |
| `GGML_QNN_DELAY_EXECUTE_SKIP` | 0 | test-only: the first n matching executes run undelayed and only the next one is delayed (one-shot); later executes of the graph run at normal speed |
| `GGML_QNN_FAIL_FINALIZE` | unset | test-only fault hook. The value is a graph-key substring: the finalize of every matching graph returns `QNN_GRAPH_ERROR_GENERAL` without calling `graphFinalize`, after the graph was created and its node added, so the real finalize-error path runs: a static shape that already built and validated in the session clamps the static budget, any other shape is refused and written to the `GGML_QNN_DENYLIST` file. Never set outside the test suite |
| `GGML_QNN_FAIL_FINALIZE_SKIP` | 0 | test-only: the first n matching graphs finalize normally; every match after them fails |

A value that fails to parse falls back to the default with a WARN in the log. On Windows an
unset `ADSP_LIBRARY_PATH` is reported at WARN, which `llama-cli` and `llama-server` show at the default verbosity, since on QAIRT 2.45 and newer the first
execute dies silently without it.

Why the backend is unavailable is reported at INFO: `QnnHtp.dll could not be loaded, backend unavailable (tried ...)` lists every path tried with the loader's reason for each, and a failed `QnnDevice_create`, a build that is not aarch64 and `GGML_QNN_DISABLE` have an INFO line each. `llama-cli` and `llama-server` drop INFO at the default verbosity, so `-dev QNN` is then rejected with `invalid device: QNN` and nothing else. Rerun with `-lv 4` or `-v` placed before `-dev` (the device list is resolved while the arguments are parsed) to see the reason. `llama-bench` prints it without a flag, because it loads the backends before it installs its log filter.

## Testing

Configure with `-DLLAMA_BUILD_TESTS=ON` and run, from the build directory:

```
ctest -R 'test-qnn|test-list-devices|test-kleidiai' --output-on-failure
```

The lifecycle binary (`tests/test-qnn-lifecycle.cpp`) registers 32 ctest entries, one mode
or mode variant each; the header comment of that file describes every mode:

- core behavior: `test-qnn-basic`, `test-qnn-budget`, `test-qnn-denylist`,
  `test-qnn-denylist-noopt`, `test-qnn-watchdog`, `test-qnn-fault`, `test-qnn-clamp`,
  `test-qnn-clamp-unlimited`, `test-qnn-health`
- model-scale bakes and their env variants: `test-qnn-modelscale`,
  `test-qnn-modelscale-npad0`, `test-qnn-noopt`, `test-qnn-modelscale-nostatic`,
  `test-qnn-modelscale-noburst`, `test-qnn-modelscale-shm`
- configuration surface: `test-qnn-disable`, `test-qnn-mindim`, `test-qnn-rebake`,
  `test-qnn-elementwise`, `test-qnn-elementwise-on`, `test-qnn-loadprobe`, `test-qnn-envparse`
- sessions and weight paths: `test-qnn-reuse`, `test-qnn-dyncache`, `test-qnn-quantized`
- the test-only hooks `GGML_QNN_DELAY_EXECUTE` and `GGML_QNN_FAIL_FINALIZE`, on small F32
  graphs that execute in milliseconds even where the fp16 path is slow:
  `test-qnn-slow-validate`, `test-qnn-slow-compute`, `test-qnn-validate-timeout`,
  `test-qnn-validate-timeout-cold`, `test-qnn-compute-timeout`, `test-qnn-finalize-error`,
  `test-qnn-denylist-append`

All but `test-qnn-health` are functional and do not depend on device speed (every mode
runs with `GGML_QNN_SLOW_EXEC_MS=0` unless the mode sets it itself; only health keeps a
value the caller exported). `test-qnn-health` is the one that does: it fails when a
model-scale F16-weight matmul takes a second or more to execute, and its verdict only
means something with nothing else running on the machine, because other load slows the
execute too.
`test-qnn-watchdog`, `test-qnn-slow-validate`, `test-qnn-slow-compute`,
`test-qnn-validate-timeout`, `test-qnn-validate-timeout-cold` and `test-qnn-compute-timeout`
keep a degraded session on purpose and end the process without DLL detach; ctest judges
them by their `<TAG>-CHECKS-PASSED` marker line, not by the exit code (`test-qnn-fault`
ends the same way and is judged by its exit code).
The registered entries run serially (`RUN_SERIAL`) because the HTP is a single-client
device. Each has a ctest `TIMEOUT` so a wedged HTP cannot hold the serial queue: 180 s for
the watchdog, fault, hook, reuse, dyncache, quantized and envparse entries, 600 s for the
rest. Exit code 77 means no HTP or an HTP init failure; ctest reports it as
NOT RUN, not FAILED, so check the `ggml-qnn` INFO line in the output to tell the two apart.
On a box known to have an HTP, configure with `-DLLAMA_QNN_TEST_REQUIRE_HTP=ON` (default
OFF), which sets `GGML_QNN_TEST_REQUIRE_HTP=1` on every `test-qnn-*` entry, or export that
variable for a single run: the skip becomes a failure (exit 1), so a device or context that failed to create cannot turn the whole suite into skips while ctest still exits 0.
`test-qnn-modelscale-shm` also exits 77 when the fastrpc library is absent (`shm_selftest 0` in its stats file); a library that is present with a failed self-test (`shm_selftest 1`) fails the entry.
`test-backend-ops -b QNN -o MUL_MAT` is the figure in the Status table, but ONLY with
`GGML_QNN_NPAD=32` and `GGML_QNN_MIN_DIM=1` set. At the default pad bucket of 512 every one of
the ~1680 test shapes is lifted over `GGML_QNN_IO_MAX_KB` and declined by `supports_op`, and the
binary then prints `Backend QNN: OK` having compared nothing. The ctest entry
`test-backend-ops-qnn` pins those settings and fails if the executed count is zero, so a green
run means something; a bare manual invocation at the defaults does not.

`test-kleidiai-coff-patch` needs no HTP and no compiler: it runs
`kleidiai-patch-coff-asm.cmake` (see Build) with `cmake -P` over generated `.S` fixtures
shaped like the KleidiAI v1.24.0 tarball and like trees an earlier version of the script
left behind, and checks the result. It is registered in every build where the patch script
exists, with or without `GGML_QNN`, with a 60 s `TIMEOUT`.

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
  then average each backend's pair. Within one run that brings prefill for every backend
  inside 2% and decode for the GPU and the "NPU"-labelled leg (a CPU run, see the retraction) inside 0.4%. CPU decode does not converge even counterbalanced
  - a 21% pair spread - so never quote a single CPU tg128 figure from one sweep. Note what
  this does and does not buy: pair agreement is WITHIN a run and is not reproduction. A
  second settled-pack run moved CPU prefill 8.1% and CPU decode 16-32% (against either leg
  of the unaveraged pair) while each run agreed with itself to under 2.2%, so treat a pair
  spread as a contamination check, never as the uncertainty on a published number.
- Battery is not just slower, it is differently shaped. A position-matched pass on DC
  measured the GPU 1.6x down but the NPU 3.5x down (pp512 27.7 vs 95.8 t/s). The 95.8 AC
  reference comes from a separate, un-settled run (pack state not recorded) and is not
  comparable to the table above; and throughput alone did not verify which engine that leg
  ran on (see the retraction), so the 3.5x describes that configuration, not necessarily
  the HTP.
- AC alone is not enough: the pack must also be SETTLED. A deeply discharged pack on AC runs
  CPU prefill at about half speed - 58.1 and 57.9 t/s measured at 13-20% charge against 132.2
  settled. Whether the GPU escapes this is UNMEASURED - the low-charge probe ran CPU and "NPU"-labelled
  legs only, so treat the requirement below as applying to every backend until someone runs a
  GPU leg on a depleted pack. Recovery is largely done by 33-42%, but not complete: single
  legs there returned 124.5 t/s at 33% and 112.6 at 41.6%, still 6% and 15% under the settled
  132.2, and non-monotonic between the two. Require a charge draw under 5 W before measuring
  - that is the precondition the settled run met (4.6 W at 100%) - and record charge percent
  and draw per sample so a suspect run stays legible afterwards; "plugged in" on its own will
  silently halve CPU-bound prefill - the "NPU"-labelled legs included, none of which was verified to reach the HTP - while looking correct. A >40% charge
  threshold is not supported by the two probes so far (33% and 41.6% were both still under
  the settled figure, and non-monotonic), so treat charge percent as a recorded covariate,
  not a gate. Charge draw also collapses mid-leg under load (32 W to 1.1 W inside a single
  CPU leg), so a reading taken before the leg does not describe the leg.
- Throughput cannot tell you which engine ran. Verify GPU placement with PID-filtered
  `\GPU Engine(*)\Utilization Percentage` (about 95% under load against a ~2% idle
  baseline); `--device`, `--list-devices` and reported free memory have all failed to catch
  a silent CPU fallback.
- The same holds for the NPU, and `llama-bench` hides the evidence: without `-v` it installs a null log callback, so every backend WARN and ERROR is dropped and a degraded NPU leg prints CPU numbers under the `QNN` backend name. The first degrade of a process is therefore also written straight to stderr: `ggml-qnn: NPU degraded (<reason>), claiming no ops for the rest of this process: everything runs on the CPU from here`.
  A leg that never claimed anything degrades nothing and prints nothing: weights left in `CPU_REPACK`, or the IO cap at a too-large `GGML_QNN_NPAD`. The first fixable IO-cap refusal of a process goes straight to stderr and names the `GGML_QNN_NPAD` and `-ub` that fit, so `llama-bench` shows it without `-v` as well (verified 2026-09-17: `-dev QNN` at the default `GGML_QNN_NPAD=512` prints it and leaves the whole model in `CPU_REPACK`).
  The proof that an NPU leg ran is `GGML_QNN_STATS`: `graphs_created` and `exec_count` must be non-zero, and `exec_max_ms` in the millisecond range.

## Known limitations

- Windows ARM64 only in practice (the dlopen paths exist for Linux but are untested).
- fp16 math internally: F32 elementwise cannot meet strict 1e-7 tolerances (matmul
  tolerances pass).
- 512-class static bakes (a 512-wide weight at the default `GGML_QNN_NPAD=512`) have been
  seen hanging at the validation execute on QAIRT 2.45, first on battery and then on AC; it
  is the IO-size law under `GGML_QNN_NPAD`, and `GGML_QNN_IO_MAX_KB` now refuses the shape
  instead. Pick `GGML_QNN_NPAD`, with a matching `-ub`, by weight width so that `NPAD * max(K, M) * 4` stays under the cap: 64 for weights up to 2560 wide, 32 for 4096-wide projections; see the `GGML_QNN_NPAD` row.
- The shared-memory IO path (`GGML_QNN_SHARED_MEM`, `test-qnn-modelscale-shm`) failed
  intermittently on 2026-08-26 and has passed every run since; the failing output was
  overwritten, so that failure mode is uncharacterized.
- A crash inside `QnnHtp.dll` at process exit remains possible in principle: the session
  is freed with the last backend instance, and a session still live or degraded at exit is
  torn down by the OS. It has not been observed lately - exits with a live or degraded HTP
  session were clean on 2026-08-28 (ctest log) and 2026-09-16 - and upstream ggml-hexagon
  never frees its registration-time sessions either.
- The eager per-op model is the wrong long-term architecture; the roadmap is AOT context
  binaries (compile at load, cache to disk, execute only known-good graphs).

## Rebase notes

Upstream status at 2026-09-16 (ggml-org master `4bc272fd7`, 366 commits past the fork base):

- There is no QNN backend upstream; `ggml/src/ggml-qnn/` and its wiring carry over as-is.
- The `--list-devices` `fflush` fix and the KleidiAI COFF patch are both still needed.
- The fork's OpenCL commit (`70a556600`, recoverable staging-buffer allocation) conflicts
  with upstream #27630 in 14 hunks, and its CONV_2D contiguity gate is superseded by
  upstream #28503 - drop the gate when rebasing.
