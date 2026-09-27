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
| `test-backend-ops` MUL_MAT (F32/F16 weights) | 47/47 pass via `ctest -R test-backend-ops-qnn`, 40 of 40 clean on an idle machine; one 46/47 was seen once under heavy machine load and has not reproduced on either IO path |
| Single-matmul kernel throughput (burst clocks + static weights) | no figure claimed: the August 2026 numbers were taken without recording box load and have not been re-taken |
| Real-model inference | completes, no hangs; unsupported shapes fall back to the CPU automatically; since 2026-09-27 K-quant weights also run at `-ub 512`, the fp16-IO graphs no longer being held under the 1 MiB IO cap |
| Real-model accuracy vs the CPU | measured 2026-09-27 on Qwen3-4B-Q4_K_M (wikitext-2, 8 chunks of 512): the NPU path diverges from a CPU reference, mean KL divergence 0.047 +- 0.025, maximum 33.7, same top token 96.8-97.1%, where a CPU control without repack reads 0.000000. The cause is not identified, so the NPU path costs accuracy as well as speed (see the 2026-09-27 subsection under Measured comparison) |
| End-to-end speed vs the Adreno GPU (OpenCL) or a KleidiAI CPU build | measured on 2026-09-26 with placement proven by the counters, Qwen3-4B-Q4_K_M prefill: the NPU configuration (`-dev QNN -ub 32`, `GGML_QNN_NPAD=32`) is a net loss, 44-53 t/s against the same binary's CPU at 101-114 with its own `-ub 512` (~2.2x) and 74-80 at the NPU's forced `-ub 32` (~1.6x), with the fp16-IO fix in place and whether or not the weight budget is lifted. That is a verdict on the configuration, not on the HTP alone: 41 (default budget) or 68 (budget lifted) of the 252 eligible projections ran on the NPU per context, the rest on the CPU without repack. The GPU (228.1 on the settled-pack run) was not re-measured; unmeasured on 9-14B (see the 2026-09-26 subsection under Measured comparison; the earlier sweep's NPU leg never reached the HTP and that retraction stands) |
| Decode (single-token) offload | intentionally not claimed below 32-token ubatches (`GGML_QNN_MIN_DIM`), so it runs on the CPU in every configuration - the 2026-09-26 counters show no decode execute on the HTP; every decode figure in this document is CPU decode, which is bimodal on this box (about 24.7 or about 15.9 t/s at `-t 6`) and is not quoted as a point; on the settled-pack run the two measured engines (GPU, CPU) and the CPU-run leg labelled NPU converge on decode (see Measured comparison) |

The honest summary: the kernels are fast, the eager per-op execution model is robust,
and per-op scheduling and IO copies were expected from the start to eat the advantage on
real models. What 2026-09-26 measured end to end, on one dense quantized 4B model with
placement proven by the counters, is the configuration: `-dev QNN` at `-ub 32` is a net loss
there, and that run cannot say how much of the loss is the NPU slice and how much is the CPU
running most of the projections without repack (see Measured comparison; the earlier
sweep's retraction stays). 2026-09-27 added that the NPU path costs accuracy too: on the same
model its output distribution diverges from the CPU's (mean KL divergence 0.047, the top
token different at about 3% of positions) for a cause not yet identified. The practical value
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
at N=64, output exactly 1 MiB with the fp32 graph IO of the time) from the repacked bytes: finalize passed in 29 ms, the
validation execute timed out at the 15 s watchdog, the session degraded, and 22 of 22
splits ran on the CPU. Every NPU figure in this table is therefore a CPU number taken
after a 15 s stall in warmup. The runtime setup is not the cause: the lifecycle modelscale
shapes (largest output 655 KB) executed correctly one minute later in the same
environment. The stall was read at the time as the IO-size law described under
`GGML_QNN_NPAD`, but it ran on the mixed-dtype path that `e3ba3f695` later replaced (an fp16
weight against fp32 activations, which drops to a reference kernel on the HTP), and that path
is slow enough to explain it alone: this shape is 671 M multiply-accumulates (2560 x 4096 x 64),
and the same path took 13.1 s for the 84 M of `test-qnn-health` (512 x 2560 x 64), about 105 s
at that rate against the 15 s watchdog. Which of the two caused the stall is not settled.
Two gates were put in to keep it from recurring: `supports_op` refuses any source that sits in a non-host buffer (`CPU_REPACK` is one) before it builds anything, so the upgrade pass no longer trial-builds from repacked bytes, and `GGML_QNN_IO_MAX_KB` refuses a padded IO of 1 MiB before a graph is created. The first gate is the one that covers this shape today: its graph IO is fp16 now, and since 2026-09-27 the cap covers fp16 graph IO only when the variable is set (see the `GGML_QNN_IO_MAX_KB` row), because fp16 IO ran far past 1 MiB without a hang.
The numbers stay because they are what was measured; they do not measure the NPU.

To put K-quant weights on the NPU, run with `-dev QNN`: llama-model-loader then probes
the QNN device first and the weight resolves to a plain CPU buffer, bypassing repack.
When this table was taken `llama-bench` had no way to turn repack off, so without `-dev QNN`
a K-quant model on a KleidiAI+REPACK build could not reach the NPU at all. Since the
2026-09-27 upstream merge it has `--repack 0|1` (upstream `965f89794`), like `--no-repack` in
`llama-cli` and `llama-server`; that route has not been run with the QNN backend. F16/F32
weights stay in the plain CPU buffer and do reach the NPU. See "Routing K-quant weights to the NPU" for the caveats (`GGML_QNN_NPAD` must be set as well).

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

**The GPU number reproduces; as of this table there was no NPU number to reproduce** (the 2026-09-26 subsection below has the first, with placement proven). Three sweeps have now measured
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
question until a `-dev QNN` sweep produces one. The 2026-09-26 sweep below is the first that
did: its two lifted-budget legs agreed to 0.5%, its two default-budget legs, in separate runs, to 18%.

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

### 2026-09-26: the first NPU run with proven placement

Qwen3-4B-Q4_K_M again, on the `build-npu` tree (QNN + KleidiAI with `GGML_CPU_REPACK`, no
OpenCL), QAIRT 2.45.0.260326, `llama-bench -t 6 -p 512 -n 64 -r 5` (`-n 64`, so tg64 here
against tg128 above). `ggml-qnn.dll` and the other DLLs were built on 2026-09-23 at 07:43,
after `e3ba3f695` - the fp16-IO fix, now the default - landed at 06:54; `llama-bench` itself
reports `build_commit f0891308a` because the bench binary is from 06:48 that day and predates
the commit. The `exec_max_ms 1` below, on fp16-weight graphs, is that fix at work: the
mixed-dtype path it replaced took about 13 s per execute (see `test-qnn-health-control` under
Testing). Four configurations, every leg the same binary and the same model:

- A: NPU. `-dev QNN -ub 32`, `GGML_QNN_NPAD=32`, the default 1024 MB static-weight budget. The budget refused its first weight at 1012.5 MiB committed; 41 weights (1022.5 MiB) were on the NPU in each of llama-bench's two contexts (see the placement paragraph below).
- D: A with `GGML_QNN_STATIC_BUDGET_MB=0`. Lifting the budget did not put every weight on the NPU: the backend clamped it at 1830.0 MiB committed (`budget_clamped 1`, the build failure on an already-built shape that the budget bullet under How it behaves describes). Per context 68 weights were on the NPU, and the 69th bake, a 47.5 MiB FFN projection that would have brought the total to 1877.5 MiB, is the one that failed (`weights_baked` and `graphs_created` count attempts, before finalize, and the raw 138 is 2 x 69). That ceiling, reached on a near-idle machine, is higher than the ~1170 MiB seen on 2026-09-16 under load (see the noise paragraph below). The phase and error code of the failure are not on record: the clamp WARN that names them went through the log callback, which `llama-bench` without `-v` drops. The backend now writes the first clamp of a process straight to stderr instead of the log.
- B: CPU. `GGML_QNN_DISABLE=1 -ub 32`, the ubatch the NPU configuration forces.
- C: CPU. `GGML_QNN_DISABLE=1 -ub 512`, its own ubatch.

The NPU-eligible weights at `GGML_QNN_NPAD=32` are all 252 projections of the 36 layers
(`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_up`, `ffn_gate` and `ffn_down`, all Q4_K or
Q6_K). The IO cap counts 2 bytes per element for them, since their graph IO is fp16, so the
9728-wide FFN is 9728 x 32 x 2 = 608 KiB, under the 1024 KiB cap; only the 151936-wide tied
output layer is over `GGML_QNN_IO_MAX_KB` and never leaves the CPU (the cap then covered fp16
graph IO by default; since 2026-09-27 it does so only when set). Which of them ran on the
NPU follows from the bytes. Baked to fp16, `attn_q` and `attn_output` are 20 MiB each, `attn_k`
and `attn_v` 5 MiB and each FFN projection 47.5 MiB, 192.5 MiB per layer, taken in graph order.
A's first refusal at 1012.5 MiB is 5 x 192.5 + 50: layers 0-4 whole and layer 5's attention,
then layer 5's first FFN projection refused and two 5 MiB projections of layer 6 still fitting,
41 weights and 1022.5 MiB in all. D's clamp at 1830.0 is 9 x 192.5 + 50 + 47.5: layers 0-8,
layer 9's attention and its first FFN projection, 68 weights. Attention weights alone reach
neither figure, since they come in multiples of 5 MiB and total 1800 MiB. The counters agree
once they are read per context: `llama-bench` builds one context per test (pp512, then tg64),
each bakes its weights at reserve, and the process-wide counters add both, so `weights_baked`
82 is 2 x 41 and 138 is 2 x 69 (68 plus the failed bake), while `exec_count` 3936 is 41 x 96
and 6528 is 68 x 96, 96 being the pp512 test's 16 ubatches of 32 tokens over its warmup and
five repetitions. The QnnHtp runtime's graph-prepare lines in the raw output split the same
way: 41 before the pp512 result and 41 after it in both A legs, 69 and 69 in both D legs. So
per context A ran 41 of the 252 on the NPU and D 68, and the other 211 (A) and 184 (D), 93 and
80 of them FFN projections, ran on the CPU.

Counterbalanced order A B C D D C B A, 120 s cooldowns, on AC, the pack at 100% for run 1;
CPU load and charge recorded per leg; the driver is described under Benchmarking notes. Run 1
lost AC during leg 7 (charge 100 -> 96% inside the leg), so leg 7 is excluded and leg 8 did
not run; the missing B and A legs were re-taken in a second run four minutes later with the
pack at 94-96% and charging, which is not a settled pack. Disk Cleanup (`cleanmgr`) was
running in the background during run 1. `+-` is llama-bench's stddev over the five
repetitions.

| Leg | Config | pp512 t/s | tg64 t/s (CPU decode in every leg) | CPU during leg | Charge | `GGML_QNN_STATS` |
|---|---|---:|---:|---:|---|---|
| 1 | A: NPU, 1024 MB budget | 44.08 +- 6.39 | 18.61 +- 5.57 | 46.8% | 100 -> 100% | weights_baked 82, exec_count 3936, exec_slow 0, exec_max_ms 1 |
| 2 | B: CPU `-ub 32` | 79.75 +- 21.83 | 24.79 +- 6.87 | 49.7% | 100 -> 100% | backend disabled |
| 3 | C: CPU `-ub 512` | 100.97 +- 24.84 | 24.67 +- 7.03 | 47.8% | 100 -> 100% | backend disabled |
| 4 | D: NPU, budget lifted | 49.09 +- 11.58 | 19.84 +- 6.68 | 42.1% | 100 -> 100% | weights_baked 138, exec_count 6528, exec_slow 0, exec_max_ms 1 |
| 5 | D: NPU, budget lifted | 49.34 +- 10.50 | 19.95 +- 6.43 | 43.7% | 100 -> 100% | weights_baked 138, exec_count 6528, exec_slow 0, exec_max_ms 1 |
| 6 | C: CPU `-ub 512` | 113.52 +- 22.06 | 15.80 +- 0.17 | 53.4% | 100 -> 100% | backend disabled |
| 7 | B: CPU `-ub 32` | 91.52 +- 16.52, excluded | 15.88 +- 0.04, excluded | 54.1% | 100 -> 96%, AC lost | backend disabled |
| run 2, 1 | B: CPU `-ub 32` | 73.72 +- 20.04 | 15.93 +- 0.38 | 59.1% | 95 -> 95%, charging | backend disabled |
| run 2, 2 | A: NPU, 1024 MB budget | 52.74 +- 10.32 | 21.03 +- 6.28 | 50.7% | 96 -> 96%, charging | weights_baked 82, exec_count 3936, exec_slow 0, exec_max_ms 1 |

Total CPU during each leg was 42-59%, where a `-t 6` workload on 12 cores predicts about 50%;
idle CPU at each leg's gate was 3-11%.

**1. The NPU executed on the HTP this time, and every execute was fast.** The proof the
Benchmarking notes ask for is on record for every NPU leg: `weights_baked` 82 (A) and 138 (D)
over both contexts, `exec_count` 3936 and 6528, `exec_slow` 0, `exec_max_ms` 1. The counters, not the throughput,
say which engine ran. This is the figure this document has said did not exist.

**2. Prefill: the NPU configuration is a net loss on a dense quantized 4B model.** NPU 44-53 t/s
(A pair 44.1 / 52.7, D pair 49.1 / 49.3) against CPU 101-114 at its own `-ub 512` and 74-80 at
the NPU's forced `-ub 32`: ~2.2x slower than the CPU at the CPU's own ubatch (the C pair's mean
107.2 against the four NPU legs' mean 48.8), ~1.6x slower even with the CPU handicapped to
`-ub 32` (76.7 against 48.8). Lifting the weight budget - 68 weights on the NPU per context
instead of 41 - changes nothing (49.2 against 48.4), and the fp16-IO fix was in place, so the
fix did not change the sign. What this compares is configurations, not engines in isolation.
Under `-dev QNN` every probe-accepted weight resolves to a plain CPU buffer (see Routing
K-quant weights to the NPU; the placement probe does not consult the budget), so the 211 (A)
and 184 (D) projections the budget left on the CPU - about 85% and 74% of the projection
weights by size - ran there without `CPU_REPACK`, while the CPU legs ran with every weight in
`CPU_REPACK`. That follows from the placement rule and was not read from a load log. D moved 27
weights, 13 of them FFN, from the unrepacked CPU to the NPU and gained 0.8 t/s, inside the A
pair's 18% spread, so the run cannot separate the NPU slice's own cost from the cost of losing
repack, and the Status section's expected cause - per-op scheduling and IO copies - is not
established by it. The verdict is on the configuration, and on this 4B model: `-dev QNN` with
the eager NPU slice, run as here, is slower than the CPU alone. The control that
would separate the two causes, a leg with the same placement and no weight on the NPU, has not
been run: `-dev QNN -ub 32` with `GGML_QNN_NPAD=32` and `GGML_QNN_STATIC_BUDGET_MB=1` keeps the
placement and lets the budget refuse every bake (`GGML_QNN_NO_STATIC_WEIGHTS=1` would not: it
makes quantized weights unclaimable, so they go back to `CPU_REPACK`), and since the upstream
merge `llama-bench --repack 0` with `GGML_QNN_DISABLE=1` is a second option. It would have to be
interleaved with the NPU legs, because this box's between-run spread (8.1% on CPU prefill, see
above) would swamp a comparison across runs.

**3. Decode is not an NPU measurement.** Every configuration ran decode on the CPU: the backend
claims nothing below 32-token ubatches (`GGML_QNN_MIN_DIM`), and the counters agree.
`exec_count` is a prefill-ubatch count with no decode term: 3936 and 6528 are the full 96
warmup-plus-repetition ubatches of the pp512 test on each of 41 and 68 weights (see the
placement paragraph above). The tg64 context bakes the same weights at reserve, then decodes at
N=1, below `GGML_QNN_MIN_DIM`; a decode step placed on the NPU would have added one execute per
resident weight per token - 41 x 321 = 13161 in A over the tg64 test's warmup and five
repetitions - and none appear. CPU
decode is bimodal on this box, as recorded above: 24.7-24.8 in legs 2-3, 15.8-15.9 in legs 6-7
and run 2, and the switch happened inside legs 1-5 and run 2's A leg, whose first repetitions
ran at 24-30 t/s and last at 13-16 (leg 2: 30.2, 30.1, 28.9, 18.7, 16.0), while legs 6-7 and
run 2's CPU leg sat at 15.6-16.4 for all five. The NPU-configured legs' decode (18.6-21.0,
+-5.6-6.7) straddles both modes. **CPU decode is deliberately not reported**, as above; the
tg64 column is there so the modes can be seen, not as a figure.

**4. Noise, stated plainly.** CPU prefill within-leg stddev was 20-25 t/s today, 19-27%
relative, and the C pair spread 11.7%, against a CPU prefill pair spread of 1.6% and a noisiest
leg at 5.5-7.4% relative on the 2026-08-27 settled run. So the CPU prefill is a 74-114 range
here, not a point, and today's C legs (101.0, 113.5) sit 14-24% under that run's 132.2. The NPU
D pair agreed to 0.5%, the A pair (in separate runs) to 18%, and the NPU legs ran 15-24%
relative stddev within a leg. In every prefill leg the repetitions fall from the first to the
last (leg 6: 135.5, 128.9, 122.0, 97.7, 83.5; leg 4: 69.3, 48.7, then a flat 42.5), so the
stddev is mostly a drift within the leg, not scatter around a level; the clock was not sampled today, so the drift is
recorded, not attributed. The first repetitions of the C legs (130.8, 135.5) sit at the settled
run's 132.2; the leg means do not. The verdict survives the worst pairing - the CPU's slowest
clean own-ubatch leg, 101.0, against the NPU's fastest, 52.7, is still 1.9x - and holds at both
ends of the drift: first repetitions 130.8-135.5 against 54.9-69.8, last repetitions 77.4-83.5
against 42.0-45.0. Run 2 was on a charging pack at 94-96%, not a settled one. Disk Cleanup ran
in the background during run 1.

**5. The GPU row was not re-measured.** The Adreno figures in the table above (228.1 / 20.0) are
from the 2026-08-27 settled-pack run and stand as measured then; `build-npu` has no OpenCL, and
no GPU leg ran today.

### 2026-09-27: fp16 graph IO past the cap, and what the NPU path costs in accuracy

Qwen3-4B-Q4_K_M on the `build-npu` tree again (`llama-bench` reports `build_commit 820d1835b`),
`-t 6` throughout. The raw logs and `GGML_QNN_STATS` files are kept outside the repository, in
the bench records' `2026-09-27/npu-experiments` folder.

**The IO cap is not needed on the fp16-IO path.** `test-qnn-lifecycle bigstatic`, one case per
process, ran its five K=512 static bakes (M = 512, 640, 768, 1024 and 2560) at `GGML_QNN_NPAD`
64, 256 and 512 with `GGML_QNN_IO_MAX_KB=1048576`: all 15 were claimed, computed and matched
the CPU, finalize 16.1-32.2 ms, validation execute 2.6-3.2 ms. The largest, 512 x 2560 at the
512 bucket, writes 2621440 bytes (2.5 MiB) by the graph's DDR summary, past the ~1 MiB that
hung on the mixed-dtype path, and built in 25.7 ms with a 3.2 ms validation execute. A model
run went further: `llama-bench -dev QNN -ub 512`, pp512 only and two repetitions, with
`GGML_QNN_NPAD=512`, the cap lifted and `GGML_QNN_STATIC_BUDGET_MB=0` reported `weights_baked` 58, `exec_slow` 0 and
`exec_max_ms` 18. That is 57 weights on the NPU - layers 0-7 whole, their 9728-wide FFN
projections with 9.5 MiB of padded IO each included, and layer 8's `attn_q`, 1560.0 MiB by the
byte arithmetic of the 2026-09-26 subsection - plus a 58th bake, a 2560 x 1024 projection
whose finalize failed with 6020 and clamped the budget at 1560.0 MiB committed; `exec_count`
171 is 57 x 3, one 512-token ubatch over the warmup and two repetitions. pp512 came back 58.2
+- 11.8 t/s (66.5, 49.9); that is one run of two repetitions with no interleaved CPU leg, so
it is not compared with the tables above. `llama-perplexity` at `-ub 512` with the same
settings showed the same: `weights_baked` 58, the clamp at 1560.0 MiB, `exec_slow` 0,
`exec_max_ms` 46. So on the fp16-IO path the cap only forced small ubatches, and it now covers
fp16 graph IO only when `GGML_QNN_IO_MAX_KB` is set (see its row). The `-ub 32` perplexity run
below clamped at 1762.5 MiB; why the 512 bucket clamps 202.5 MiB lower is not established.

**Accuracy: the NPU path diverges from the CPU.** `llama-perplexity` on the wikitext-2 test set,
`-c 512`, 8 chunks, KL divergence against the saved logits of a CPU run at `-ub 32`:

| Run | Weights on the NPU | PPL | PPL(Q)/PPL(base) | Mean KLD | 99.9% KLD | Max KLD | Same top token |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU reference | none | 15.1947 +- 1.1852 | - | - | - | - | - |
| CPU `--no-repack`, control | none | 15.1947 +- 1.1852 | 1.016 +- 0.008 | 0.000000 | 0.000051 | 0.000064 | 100.0% |
| NPU `-dev QNN -ub 32`, `GGML_QNN_NPAD=32`, budget 0 | 66 (`weights_baked` 67, `exec_count` 8448 = 66 x 128, `exec_slow` 0, `exec_max_ms` 4) | 15.0685 +- 1.1754 | 1.008 +- 0.015 | 0.047 +- 0.025 | 17.8 | 33.7 | 96.8 +- 0.4% |
| NPU `-dev QNN -ub 512`, `GGML_QNN_NPAD=512`, cap lifted, budget 0 | 57 (`weights_baked` 58, `exec_count` 456 = 57 x 8, `exec_slow` 0, `exec_max_ms` 46) | 15.0964 +- 1.1797 | 1.010 +- 0.015 | 0.047 +- 0.025 | 17.9 | 34.5 | 97.1 +- 0.4% |

`weights_baked` counts bake attempts, so each NPU row's count includes the one bake whose
finalize failed and clamped the budget (at 1762.5 and 1560.0 MiB committed); the executes
confirm the placement, 128 ubatches of 32 tokens and 8 of 512 over the 8 chunks.

- The control reproduces the reference (mean KLD 0.000000, maximum 0.000064, the same top token
  everywhere), so the CPU running the unbaked weights without repack, as it does in every
  `-dev QNN` run, is not the source: the divergence is the NPU path's.
- It is concentrated in a few positions: the median KLD is 0.0018-0.0019 and the 99th percentile
  0.051-0.056, while the 99.9th is 17.8-17.9 and the maximum 33.7-34.5, and the top token
  differs at about 3% of positions. The 32 and 512 buckets agree within noise, so the bucket
  size does not change it.
- The PPL ratio does not show it. The control reads 1.016 +- 0.008 against the same reference
  with zero divergence - an offset from how `llama-perplexity` stores the reference, its
  log-probabilities quantized to 16 bits per token (`tools/perplexity/perplexity.cpp`) - and the
  NPU runs' own PPL (15.07, 15.10) sits well inside the CPU's 15.19 +- 1.19.
- The cause is not identified. Candidates, none verified: the fp16 range on activations or
  outputs (graph IO is fp16, so a value past 65504 cannot be represented), and fp16
  accumulation inside the HTP matmul. The op-level MUL_MAT pass in the Status table compares
  single matmuls at a tolerance, not a model's activations, and does not settle it either way.

So the NPU path costs accuracy as well as speed: on this model it is slower than the CPU
(2026-09-26) and does not reproduce the CPU's output distribution.

## Requirements

- Windows 11 ARM64 on a Snapdragon with an HTP (tested: X Elite / HTP v73).
- Qualcomm AI Runtime (QAIRT) community SDK - headers only at build time.
- `QnnHtp.dll` and its dependencies available at run time via `PATH`, or via `QNN_SDK_ROOT`,
  under which the backend tries the SDK's `lib/aarch64-windows-msvc` and then
  `lib/arm64x-windows-msvc` directory. `QNN_SDK_ROOT` is also read from the environment at
  configure time when `-DQNN_SDK_ROOT` is not given; the value is then cached, so the
  environment is read only while that cache entry is empty, and an existing build directory is
  re-pointed with `-DQNN_SDK_ROOT=<path>`.
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
- Before any of that, `supports_op` does the checks that are arithmetic on the shape and need no session. A matmul whose padded IO would reach the IO cap, or overflow the 32-bit sizes QNN takes, is reported as not supported: it runs on the CPU and is never a failed op (a claimed node that the graph policy then refuses does not fall back, it fails the graph). The cap is `GGML_QNN_IO_MAX_KB` for every matmul graph when that is set; unset, its default 1024 covers only fp32 graph IO (an F32 weight, or any weight under `GGML_QNN_NO_F16_IO`), and an fp16-IO graph is limited by the 32-bit sizes alone.
  A weight probed for placement at load has no real batch yet, so the smallest bucket (the `GGML_QNN_NPAD` floor) is used: a weight capped even there is not placed on the NPU, so on a KleidiAI build it keeps `CPU_REPACK` instead of landing in a plain CPU buffer it would never leave. At the default `GGML_QNN_NPAD=512` with the cap unset that is every F32 weight 512 or more wide; F16 and quantized weights, whose graph IO is fp16, are placed at any width. With `GGML_QNN_IO_MAX_KB=1024` set it is also every F16 or quantized weight 1024 or more wide, i.e. all of a model-scale model, which is how every run before 2026-09-27 behaved.
  The first IO-cap refusal that a setting can fix is printed once per process, straight to stderr; it names the cap that refused it (a set `GGML_QNN_IO_MAX_KB`, or the default for fp32 graph IO) and the `GGML_QNN_NPAD` and `-ub` that fit that shape. A shape that fits only below `GGML_QNN_MIN_DIM` at a 1 MiB cap (a 151936-wide output layer; a 9728-wide FFN only with an F32 weight or under `GGML_QNN_NO_F16_IO`) can never run here at that cap, so it stays at DEBUG and does not use up that one notice; later refusals are DEBUG.
  A source in a non-host buffer is refused as well, see "Routing K-quant weights to the NPU". An exception inside the trial build (host out of memory) is an ERROR and the op is not claimed.
- Model weights are baked into their graphs once (dequantized to fp16 when quantized) in
  HTP-native layout, within a memory budget (default 1024 MB, a conservative choice: in one run on a
  machine loaded with other work the HTP stopped mapping baked weights at about 1170 MiB
  committed; on 2026-09-26, on a near-idle machine with the budget lifted, it clamped at 1830.0
  MiB, where the 47.5 MiB bake that failed would have brought it to 1877.5, and the default
  stays below both). Weights past the budget
  stay on the CPU. If a static graph still fails to build on a shape that already built and
  validated in the session, that is NPU weight memory running out, not a verdict on the
  shape: the budget is clamped to what is committed, later weights stay on the CPU, nothing
  is denylisted and the session keeps the graphs it has (one WARN with the phase and error
  code, which for the first clamp of a process goes straight to stderr instead of the log, so
  that `llama-bench` without `-v` shows it; `budget_clamped` in `GGML_QNN_STATS`). Once the budget is clamped, the
  one budget-full notice on stderr names the clamp and its error code instead of suggesting a
  larger `GGML_QNN_STATIC_BUDGET_MB`. Static graph keys carry a fingerprint of the weight
  contents (its size, its first and last 512 bytes and 64 strided 8-byte samples), so a
  different weight landing at a reused address does not hit a stale graph. Each static
  bake pays graph build + finalize + bake (tens of ms to seconds per shape), and that is
  paid at context creation: the scheduler's reserve pass at `n_ubatch` trial-builds every
  resident weight, so `llama_init_from_model` is slower than steady state - expected, not
  a hang. The reserve graph has as many output rows as tokens (`n_outputs = min(n_ubatch,
  n_outputs_max)`), so it also bakes the weights that at compute time run only on the output
  rows - the last layer's FFN after llama's output-row gather, and the output layer, which at
  fp16 graph IO the default cap no longer refuses at any vocabulary (with a 1 MiB cap set, at
  the 32 bucket, only a vocabulary under 16384 passes) - and
  charges them against `GGML_QNN_STATIC_BUDGET_MB`. A normal prefill has one output row, below
  `GGML_QNN_MIN_DIM`, so those run on the CPU and their bakes only use up budget; that matters
  only when the budget reaches that far (a small model, or the budget lifted). The bake cost is
  re-paid per context, since the session and its baked weights are freed with the last
  backend instance. Only dynamic graphs (LoRA) are built on the first prompt
  that needs them. A matmul whose first operand is computed in the graph rather than a weight
  or a view of one (mean pooling's `cont(transpose(inp))`, whose K follows the ubatch) is not
  claimed: every new K would build a graph that is never freed.
  A trainable weight (`GGML_TENSOR_FLAG_PARAM`, which `llama_opt_init` sets for finetuning) is never baked: the optimizer updates it in place, so it takes the dynamic path and is copied on every execute.
  If the host cannot allocate the staging copy of a weight, the shape is a policy reject (ERROR `out of host memory staging the weight of ...`): no QNN graph is created and nothing is denylisted.
  The fingerprint (the `_h` suffix of a graph key in the log) is not stable across builds; shape keys and the denylist file format are, except that the shape key of an fp16-IO graph (an F16 or quantized weight, unless `GGML_QNN_NO_F16_IO` is set) now carries an `_io16` tag in front of its `_s` or `_dyn` variant tag. A denylist entry written by an earlier build, or under `GGML_QNN_NO_F16_IO`, therefore no longer bans the fp16-IO graph of that shape.
- Static-weight graphs pad the batch dimension to a bucket (default 512, `GGML_QNN_NPAD`),
  so one graph and one baked weight serve every N up to NPAD. Larger ubatches get one
  graph and one bake per power-of-two bucket, each charged to the budget, and a bucket
  whose padded IO would reach the IO cap is refused (fp32 graph IO only, unless
  `GGML_QNN_IO_MAX_KB` is set). A ubatch smaller than the floor still executes the whole
  padded bucket. Dynamic graphs are bucket-padded the same way.
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
  the CPU. A healthy HTP executes a matmul in milliseconds: Qwen3-4B's projections at the 512 bucket, the 9728-wide FFN included, peaked at 18 and 46 ms in two runs on 2026-09-27.
  The check runs on the validation execute and on every compute-time execute; the second catches a device that slows down after its graphs validated (a cached graph is never re-validated). At compute time the output is valid, so the node succeeds, and the WARN names the op.
  This degrade is slow-only: nodes the scheduler already placed on graphs that built keep executing on the NPU instead of failing their batch, until the scheduler next splits a graph. Nothing new is built, so a placed node whose graph does not exist yet fails, and any later hard failure (timeout, execute error) ends the exemption.
  All of this needs the execute to finish inside `GGML_QNN_TIMEOUT_MS`: past it a slow device looks the same as a wedge.
- The session is created on first use and freed with its last backend instance, so a tool that creates and frees a context per run (`llama-bench` per test, `--fit` per probe) creates it again each time. If that fails after a session has worked in the process (another process holds the HTP, or it is wedged), the failure is latched for the process: nothing retries it, since a retry against a wedged HTP can hang in `QnnDevice_create` with no watchdog, and the backend still hands out instances that claim nothing and fail any matmul handed to them, so the context runs on the CPU instead of failing with `failed to initialize QNN backend`. The first such failure writes `ggml-qnn: the NPU session could not be re-created, claiming no ops for the rest of this process: everything runs on the CPU from here` to stderr.
- The denylist has two levels. Every failed shape is remembered for the process. Only an HTP verdict on the shape, which is what a rerun must skip, is also written to the `GGML_QNN_DENYLIST` file:
  a finalize error, always; a finalize timeout, unless the static budget has been clamped; and either kind of execute timeout - the validation execute or a compute-time execute - but only when the session has already seen a validation execute complete under `GGML_QNN_SLOW_EXEC_MS` (under 2 s when that check is set to 0) and the budget is not clamped.
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
`-dev QNN` is the route every measurement here used. `--no-repack` (`llama-cli`,
`llama-server`) and, since the 2026-09-27 upstream merge, `llama-bench --repack 0` also keep
K-quant weights out of `CPU_REPACK`; neither has been run with the QNN backend. F16/F32 weights
stay in the plain CPU buffer regardless and reach the NPU without it.

Since 2026-09-27 `-dev QNN` is enough at the default `GGML_QNN_NPAD=512` for F16 and quantized weights: their graph IO is fp16, which the IO cap covers only when `GGML_QNN_IO_MAX_KB` is set, so the placement probe accepts them and llama's default `-ub 512` runs every baked weight at the 512 bucket (the 2026-09-27 subsection under Measured comparison ran that, with the cap lifted by hand before the default changed and the weight budget lifted). Keep `GGML_QNN_NPAD` at or under `-ub`: a smaller ubatch still executes the whole padded bucket. One side effect: the 151936-wide output layer is accepted at placement as well now, so under `-dev QNN` it too leaves `CPU_REPACK` for a plain CPU buffer, although it runs on the CPU whenever fewer than `GGML_QNN_MIN_DIM` rows are output (every single-sequence decode step and a normal prefill) and is baked only if the budget reaches it; what that costs the CPU is unmeasured. Before that change the probe refused every weight whose padded IO reached the 1 MiB cap at the smallest bucket, which at 512 is every F16 or quantized weight 1024 or more wide, so a model-scale model stayed in `CPU_REPACK` and nothing ran on the NPU; that is why every earlier run used `-ub 32` with `GGML_QNN_NPAD=32`, and it is still what happens with `GGML_QNN_IO_MAX_KB=1024` set. An F32 weight 512 or more wide, or any weight under `GGML_QNN_NO_F16_IO`, is refused the same way at the default. That first refusal is written straight to stderr rather than through the log callback - so it survives `llama-bench` without `-v`, but does not appear in a redirected log - and names the cap and the `GGML_QNN_NPAD` and `-ub` that fit the shape. For those, set both by weight width, see the `GGML_QNN_NPAD` row.

Measured on 2026-09-16 (Qwen3-4B-Q4_K_M, `-t 6 -p 32 -ub 32 -dev QNN`, `GGML_QNN_NPAD=32`,
QAIRT 2.45.0.260326). Placement works: with the watchdog limits raised by hand (to repeat this on a device that slow, raise `GGML_QNN_TIMEOUT_MS` past the slowest execute and set `GGML_QNN_SLOW_EXEC_MS=0` as well; with the timeout alone the completed validation is refused as too slow), 81 static graphs
built and validated inside the 1024 MB budget and the scheduler put 81 matmuls on the NPU
in 61 splits (all four attention projections of about 20 layers). At `-ub 32` with
`NPAD=32` the attention projections (up to 4096 wide, 512 KiB padded IO) fit under
`GGML_QNN_IO_MAX_KB`, while the FFN projections (9728 wide, 1216 KiB) did not and stayed on
the CPU. Those are fp32 IO figures, from before `e3ba3f695`: graph IO is fp16 for these
weights now, which halves both, so at `NPAD=32` the FFN is 608 KiB and is admitted (the
2026-09-26 run put FFN projections on the NPU). In one run with the budget at its old 2048 MB
default, on the same loaded machine, the HTP could no longer map weights at about 1170 MiB
committed (finalize 6020, then an execute 6002, on shapes that had built dozens of times).
On 2026-09-26, on a near-idle machine with the budget lifted, the clamp came at 1830.0 MiB
committed instead, and by the byte arithmetic the 47.5 MiB FFN bake that failed would have
brought it to 1877.5, so memory pressure from the other work may explain the lower figure.
The default is 1024 MB, below both, and such a failure clamps the budget instead of degrading
the session either way.

Throughput was not measured. Every timing taken from 2026-09-16 to 2026-09-18 ran while
other heavy work (builds, test suites, benchmarks from other sessions) was loading the
machine, so none of it is a valid measurement of the NPU, and this document draws no
conclusion from it about the NPU's speed. Under that load, graphs with FP16 weights (K-quant
weights are baked to fp16) took seconds per execute while graphs with F32 weights took
milliseconds. That was not the load: it was the mixed-dtype matmul (an fp16 weight against
fp32 activations drops to a reference kernel), reproduced on an idle machine on 2026-09-23
and fixed in `e3ba3f695`; `GGML_QNN_NO_F16_IO` brings it back, and `test-qnn-health-control`
guards against it.
End to end, a six-graph run under the same conditions completed with correct results.
Before any speed claim, run `test-qnn-health` with nothing else running on the machine.
Both kinds of execute timeout - the validation execute and a compute-time execute - reach the `GGML_QNN_DENYLIST` file only when the session has already seen one validation complete at normal speed, and neither does once the static budget has been clamped. So a device that is slow from its first graph writes nothing, while one that slows down mid-run can. Once `test-qnn-health` passes again, delete the file or the entry.

Caveat: `-dev QNN` also makes `llama_context` create two QNN backend instances on one
session - one from the device loop and one from the ACCEL loop. This works, but it was not
anticipated.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_QNN_DISABLE` | unset | disable the backend entirely |
| `GGML_QNN_MIN_DIM` | 32 | minimum matmul dimension to claim (smaller goes to the CPU) |
| `GGML_QNN_STATIC_BUDGET_MB` | 1024 | cap on baked static-weight bytes (a quantized weight is charged at its fp16 on-device size), 0 = unlimited. 1024 as a conservative margin: weight mapping failed at about 1170 MiB committed in one run on a machine loaded with other work, and at 1830.0 MiB on a near-idle one with the budget lifted (2026-09-26); a build failure on a shape that already built in the session clamps the budget to the committed bytes for the rest of the session, and the first clamp of a process is written straight to stderr with its phase and error code. Weights that run only on the output rows are baked and charged at reserve as well, see the budget bullet under How it behaves |
| `GGML_QNN_NPAD` | 512 | batch-dim bucket floor for static and dynamic graphs: one graph serves every N up to it, and a ubatch below it still executes the whole padded bucket, so keep it at or under `-ub` (the defaults, 512 and 512, match). The padded IO is `NPAD * max(K, M) * e` bytes, where e = 2 for F16 and quantized weights (their graph IO is fp16) and 4 for F32 weights or under `GGML_QNN_NO_F16_IO`. Where the IO cap applies (see `GGML_QNN_IO_MAX_KB`: e = 4 by default, every graph when set) the padded IO must stay UNDER it (equal is refused), with a matching `-ub`, so a too-large NPAD costs offload rather than a watchdog stall. At a 1 MiB cap and e = 4: 64 for a 2560-wide weight, 32 for 4096-wide projections such as Qwen3-4B attention, 16 for the 9728-wide Qwen3-4B FFN and 1 for the 151936-wide output layer, the last two below the default `GGML_QNN_MIN_DIM`, so those stay on the CPU; at the default 512 every weight 512 or more wide is refused, at placement too. With `GGML_QNN_IO_MAX_KB=1024` set, e = 2 is capped too, at 128 for a 2560-wide weight (anything under 4096 wide), 64 for 4096-wide projections, where 128 lands exactly on 1 MiB, 32 for the FFN and 3 for the output layer, and at 512 every weight 1024 or more wide is refused. The first refusal of a process goes straight to stderr, not through the log callback, and names the cap and the `GGML_QNN_NPAD` and `-ub` that fit that shape. Where the cap comes from: graph execute hung when a padded IO buffer crossed a runtime-dependent threshold (~1-1.5 MB measured; exactly 1 MiB hung on QAIRT 2.45), and every one of those measurements ran on the mixed-dtype path that `e3ba3f695` replaced. fp16 graph IO does not follow it: on 2026-09-27, with the cap lifted, K=512 static bakes up to 2.5 MiB of padded output passed at NPAD 64, 256 and 512, and Qwen3-4B at `-ub 512` ran 57 weights on the NPU, the 9728-wide FFN with 9.5 MiB of padded IO among them, with no slow execute (see the 2026-09-27 subsection under Measured comparison). No measurement here covers an F32 weight past 1 MiB, so fp32 graph IO keeps the cap |
| `GGML_QNN_IO_MAX_KB` | unset (1024 for fp32 graph IO) | policy-reject a matmul whose padded input or output would reach this many KB (strict less-than passes); the shape runs on the CPU instead of hanging the HTP, and as a policy reject it is not denylisted. Set, it caps every matmul graph. Unset (or not a valid number, which WARNs), the default of 1024 caps only graphs whose IO is 4 bytes per element, an F32 weight or any weight under `GGML_QNN_NO_F16_IO`, the mixed-dtype path where the hang was measured; an F16 or quantized weight's fp16 graph IO is then limited only by the 32-bit sizes QNN takes, because it ran far past 1 MiB without a hang (2026-09-27, see `GGML_QNN_NPAD`) and the cap there only forced small ubatches. Before 2026-09-27 the default capped fp16 graph IO too; `GGML_QNN_IO_MAX_KB=1024` restores that. `supports_op` applies the cap before it claims, so such a matmul is "not supported", never a failed op, and a weight capped at the smallest bucket is not placed on the NPU at load. The padded IO counts 2 bytes per element for F16 and quantized weights and 4 for F32 weights or under `GGML_QNN_NO_F16_IO`, see `GGML_QNN_NPAD` |
| `GGML_QNN_NO_OPT` | unset | drop the finalize-optimization flag (matmul graphs; debugging lever); also bypasses denylist entries loaded from the file |
| `GGML_QNN_NO_STATIC_WEIGHTS` | unset | disable static weight baking |
| `GGML_QNN_NO_F16_IO` | unset | restore the old mixed-dtype matmul graphs (fp16 weight, fp32 activation and output), which drop to a reference kernel on the HTP: 418x slower on one shape measured on 2026-09-23 (208.9 ms against 0.5 ms), and 13099 ms instead of 0 ms for `test-qnn-health`. Only for reproducing that; it also makes the IO cap count 4 bytes per element for every weight, which puts every weight under the default cap (see `GGML_QNN_IO_MAX_KB`), and drops the `_io16` tag from shape keys. `test-qnn-health-control` sets it |
| `GGML_QNN_HTP_ARCH` | queried | the HTP arch the device is created for (68, 69, 73, 75, 79, 81, 85, 89), overriding the one queried from the device; 0 creates the device with no arch config, as builds before `e3ba3f695` did. The INFO line `ggml-qnn: HTP arch v73, SoC model 0` names what was used, and an arch that is still unknown gets a WARN. Setting the arch had no measured effect on speed |
| `GGML_QNN_SOC_MODEL` | queried | the SoC model added to the device config (60 = X Elite, the value Genie uses). The X Elite here reports the UNKNOWN sentinel, which is not forwarded, so without this variable its device is created with the arch alone (`SoC model 0` in the INFO line) |
| `GGML_QNN_DENYLIST` | unset | file that persists HTP verdicts on shapes across runs: finalize errors and watchdog timeouts, with the exceptions in the denylist bullet under "How it behaves". Every other failure is remembered for the process only. It starts with the header line `# ggml-qnn denylist v1`. Nothing removes an entry: delete the file, or the entry, once `test-qnn-health` passes with nothing else running, if a run on a slow or busy machine wrote to it |
| `GGML_QNN_NO_PREVALIDATE` | unset | skip the test-execute during shape validation |
| `GGML_QNN_BUILD_TIMEOUT_MS` | 120000 | watchdog timeout for graph finalize only, which is a compile and may legitimately take long. It does not bound the validation execute |
| `GGML_QNN_TIMEOUT_MS` | 15000 | watchdog timeout for every graph execute, the validation execute included (a healthy execute is ms). On a device that is only slow, raising it is not enough: `GGML_QNN_SLOW_EXEC_MS` must be raised as well or set to 0, or the completed validation is refused as too slow |
| `GGML_QNN_SLOW_EXEC_MS` | 2000 | an execute at or over this many ms, the validation execute or a compute-time one, marks the NPU as too slow to use: the session claims nothing more and falls back to the CPU, nothing is denylisted. At compute time the node still succeeds, and nodes already placed on built graphs keep running (a slow-only degrade) until a hard failure. 0 disables (the lifecycle tests do, `test-qnn-health` covers speed) |
| `GGML_QNN_NO_BURST` | unset | do not lock the HTP to TURBO clocks |
| `GGML_QNN_SHARED_MEM` | unset | use registered fastrpc buffers for graph IO; host buffers are used when the init-time self-test fails (`shm_selftest` in `GGML_QNN_STATS` tells why) |
| `GGML_QNN_QUANTIZED` | unset | experimental per-execute dequant path (off: quantized weights require static baking) |
| `GGML_QNN_ELEMENTWISE` | unset | experimental ADD/MUL offload (a known HTP broadcast bug makes some shapes wrong, keep off) |
| `GGML_QNN_MIN_ELEMENTS` | 1M | elementwise offload threshold, read only with `GGML_QNN_ELEMENTWISE` set |
| `GGML_QNN_AOT_TEST` | unset | one-shot AOT context-binary round-trip measurement |
| `GGML_QNN_DEBUG` | unset | verbose QNN logging; pass `-v`/`--verbose` on the host tool or nothing prints |
| `GGML_QNN_STATS` | unset | file that receives fourteen `name value` lines when the session is freed or degrades: the eight original counters plus `burst_applied`, `budget_clamped` (0/1), `exec_count`, `exec_slow` (compute-time executes of 1 s or more), `exec_max_ms` and `shm_selftest`, the init-time shared-memory self-test of the last session (0 = the fastrpc library is absent, 1 = present but the self-test failed, 2 = ok, 3 = not requested, `GGML_QNN_SHARED_MEM` unset). The counters are process-wide and only go up, so they add over every session of the process: a session is freed with its last backend instance, and `llama-bench` builds one context per test, each baking its weights at reserve, so a `-p 512 -n 64` run reports each resident weight's bake twice. `exec_count` counts compute-time executes only, not validation executes |
| `GGML_QNN_FAIL_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: the VALIDATION execute of every graph whose key contains it fails before QNN is called, and compute-time executes are not touched. Under `GGML_QNN_NO_PREVALIDATE` there is no validation execute, and the hook fails the compute-time executes of a matching graph instead, which reaches the execute-error branch a real mid-decode failure takes (`test-qnn-compute-error`). `GGML_QNN_FAIL_EXECUTE_SKIP=<n>` lets the first n matching executes run normally. Never set either outside the test suite |
| `GGML_QNN_DELAY_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: one execute of a matching graph, validation or compute-time alike, sleeps `GGML_QNN_DELAY_EXECUTE_MS` inside the timed call right before `graphExecute`, so the delay counts toward `GGML_QNN_TIMEOUT_MS` and `GGML_QNN_SLOW_EXEC_MS` and reaches the slow and timeout verdicts on a healthy device. Unlike a real wedge the delayed call then completes. Never set outside the test suite |
| `GGML_QNN_DELAY_EXECUTE_MS` | 0 | test-only: the delay in ms for `GGML_QNN_DELAY_EXECUTE`, capped at a day; 0 leaves the hook off |
| `GGML_QNN_DELAY_EXECUTE_SKIP` | 0 | test-only: the first n matching executes run undelayed and only the next one is delayed (one-shot); later executes of the graph run at normal speed |
| `GGML_QNN_FAIL_FINALIZE` | unset | test-only fault hook. The value is a graph-key substring: the finalize of every matching graph returns `QNN_GRAPH_ERROR_GENERAL` without calling `graphFinalize`, after the graph was created and its node added, so the real finalize-error path runs: a static shape that already built and validated in the session clamps the static budget, any other shape is refused and written to the `GGML_QNN_DENYLIST` file. Never set outside the test suite |
| `GGML_QNN_FAIL_FINALIZE_SKIP` | 0 | test-only: the first n matching graphs finalize normally; every match after them fails |
| `GGML_QNN_FAIL_INIT` | unset | test-only fault hook: with `GGML_QNN_FAIL_INIT=<n>` the first n session inits of the process run and the next one fails before `QnnHtp` is loaded (one-shot), so a re-init that fails after a working session can be reached on a healthy device (`test-qnn-reinit-fail`). Never set outside the test suite |

A value that fails to parse falls back to the default with a WARN in the log. On Windows an
unset `ADSP_LIBRARY_PATH` is reported at WARN, which `llama-cli` and `llama-server` show at the default verbosity, since on QAIRT 2.45 and newer the first
execute dies silently without it.

Why the backend is unavailable is reported at INFO: `QnnHtp.dll could not be loaded, backend unavailable (tried ...)` lists every path tried with the loader's reason for each, and a failed `QnnDevice_create`, a build that is not aarch64 and `GGML_QNN_DISABLE` have an INFO line each. `llama-cli` and `llama-server` drop INFO at the default verbosity, so `-dev QNN` is then rejected with `invalid device: QNN` and nothing else. Rerun with `-lv 4` or `-v` placed before `-dev` (the device list is resolved while the arguments are parsed) to see the reason. `llama-bench` prints it without a flag, because it loads the backends before it installs its log filter.

## Testing

Configure with `-DLLAMA_BUILD_TESTS=ON` and run, from the build directory:

```
ctest -R 'test-qnn|test-list-devices|test-kleidiai' --output-on-failure
```

The lifecycle binary (`tests/test-qnn-lifecycle.cpp`) registers 36 ctest entries, one mode
or mode variant each; the header comment of that file describes every mode:

- core behavior: `test-qnn-basic`, `test-qnn-budget`, `test-qnn-denylist`,
  `test-qnn-denylist-noopt`, `test-qnn-denylist-probe`, `test-qnn-watchdog`, `test-qnn-fault`,
  `test-qnn-clamp`, `test-qnn-clamp-unlimited`, `test-qnn-health`, `test-qnn-health-control`
- op-level correctness against the CPU: `test-backend-ops-qnn`, which runs `test-backend-ops -b QNN
  -o MUL_MAT` with `GGML_QNN_NPAD=32` and `GGML_QNN_MIN_DIM=1` and fails if it executed zero cases
- model-scale bakes and their env variants: `test-qnn-modelscale`,
  `test-qnn-modelscale-npad0`, `test-qnn-noopt`, `test-qnn-modelscale-nostatic`,
  `test-qnn-modelscale-noburst`, `test-qnn-modelscale-shm`
- configuration surface: `test-qnn-disable`, `test-qnn-mindim`, `test-qnn-rebake`,
  `test-qnn-elementwise`, `test-qnn-elementwise-on`, `test-qnn-loadprobe`, `test-qnn-envparse`
- sessions and weight paths: `test-qnn-reuse`, `test-qnn-dyncache`, `test-qnn-quantized`
- the test-only hooks `GGML_QNN_DELAY_EXECUTE`, `GGML_QNN_FAIL_FINALIZE`, `GGML_QNN_FAIL_EXECUTE`
  (at compute time, under `GGML_QNN_NO_PREVALIDATE`) and `GGML_QNN_FAIL_INIT`, on small F32
  graphs that execute in milliseconds, so the injected delays and failures are the only slow part:
  `test-qnn-slow-validate`, `test-qnn-slow-compute`, `test-qnn-validate-timeout`,
  `test-qnn-validate-timeout-cold`, `test-qnn-compute-timeout`, `test-qnn-compute-error`,
  `test-qnn-finalize-error`, `test-qnn-denylist-append`, `test-qnn-reinit-fail`.
  `test-qnn-compute-error` drives the execute-error branch a real mid-decode failure takes: the
  node fails, the session hard-degrades, nothing is written to the denylist file, and a re-init
  keeps the degraded session. `test-qnn-reinit-fail` fails the session re-init after a working
  session was freed and checks that init still returns a backend, that the backend claims
  nothing and fails a compute handed to it, and that the failure stays latched

All but `test-qnn-health` are functional and do not depend on device speed (every mode
runs with `GGML_QNN_SLOW_EXEC_MS=0` unless the mode sets it itself; only health keeps a
value the caller exported). `test-qnn-health` is the one that does: it fails when a
model-scale F16-weight matmul takes a second or more to execute, and its verdict only
means something with nothing else running on the machine, because other load slows the
execute too. `test-qnn-health-control` guards it: it reruns health with
`GGML_QNN_NO_F16_IO=1`, which restores the old mixed-dtype matmul that took about 13 s, and
passes only if health fails for that reason - on its own slow-execute check, not on a crash or
a missing DLL. A health test that went blind would turn the control red.
`test-qnn-watchdog`, `test-qnn-slow-validate`, `test-qnn-slow-compute`,
`test-qnn-validate-timeout`, `test-qnn-validate-timeout-cold`, `test-qnn-compute-timeout` and
`test-qnn-compute-error` keep a degraded session on purpose and end the process without DLL detach; ctest judges
them by their `<TAG>-CHECKS-PASSED` marker line, not by the exit code (`test-qnn-fault`
ends the same way and is judged by its exit code).
The registered entries run serially (`RUN_SERIAL`) because the HTP is a single-client
device. In a `GGML_QNN` build the upstream tests that can open the QNN device are marked
`RUN_SERIAL` too; the list is in `tests/CMakeLists.txt`, and `test-arg-parser` and
`test-model-resolution` are on it because argument parsing enumerates the backend registry
through `llama_supports_rpc`. Each lifecycle entry has a ctest `TIMEOUT` so a wedged HTP cannot hold the serial queue: 180 s for
the watchdog, fault, hook, reuse, dyncache, quantized and envparse entries, 600 s for the
rest. Exit code 77 means no HTP or an HTP init failure; ctest reports it as
NOT RUN, not FAILED, so check the `ggml-qnn` INFO line in the output to tell the two apart.
On a box known to have an HTP, configure with `-DLLAMA_QNN_TEST_REQUIRE_HTP=ON` (default
OFF), which sets `GGML_QNN_TEST_REQUIRE_HTP=1` on every `test-qnn-*` entry, or export that
variable for a single run: the skip becomes a failure (exit 1), so a device or context that failed to create cannot turn the whole suite into skips while ctest still exits 0.
`test-qnn-modelscale-shm` also exits 77 when the fastrpc library is absent (`shm_selftest 0` in its stats file); a library that is present with a failed self-test (`shm_selftest 1`) fails the entry.
`test-backend-ops -b QNN -o MUL_MAT` is the figure in the Status table, but ONLY with
`GGML_QNN_NPAD=32` and `GGML_QNN_MIN_DIM=1` set. At the defaults `supports_op` declined every
one of the ~1680 test shapes - measured 2026-09-22: zero executed at the defaults, 2 with
`GGML_QNN_NPAD=32` alone and 47 with `GGML_QNN_MIN_DIM=1` as well, so `GGML_QNN_MIN_DIM`
declines most of them and the IO cap at the 512 bucket (which then covered fp16 graph IO too)
the rest - and the binary then prints `Backend QNN: OK` having compared nothing. The ctest entry
`test-backend-ops-qnn` pins those settings and fails if the executed count is zero, so a green
run means something; a bare manual invocation at the defaults does not.

`test-kleidiai-coff-patch` needs no HTP and no compiler: it runs
`kleidiai-patch-coff-asm.cmake` (see Build) with `cmake -P` over generated `.S` fixtures
shaped like the KleidiAI v1.24.0 tarball and like trees an earlier version of the script
left behind, and checks the result. It is registered in every build where the patch script
exists, with or without `GGML_QNN`, with a 60 s `TIMEOUT`.

`scripts/fork-ci.sh` is this fork's CI: the fork runs no GitHub Actions (the script's header
says why), and the script builds and tests on this machine. `npu` (the default) builds
`build-npu` and runs the ctest entries above except `test-qnn-health`, which needs an idle
machine and is run by hand; `gpu` builds `build-gpu` and runs `test-backend-ops -b GPUOpenCL`
over every op, then upstream's unit tests; `cpu` builds `build-cpu` and runs upstream's unit
tests, then `test-backend-ops -b CPU`; `all` runs the three in turn. Upstream's unit tests are
`ctest -L main`, plus `-L python` and `-L model` when their inputs are present. Every mode
first runs `editorconfig-checker` over the files the fork changed since its upstream merge base,
when the checker is installed. A failed test step does not stop the others, and the script
exits 1 at the end if any failed; `BUILD_ONLY=1` skips the tests.

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
- The same holds for the NPU, and `llama-bench` hides the evidence: without `-v` it installs a null log callback, so every backend WARN and ERROR is dropped and a degraded NPU leg prints CPU numbers under the `QNN` backend name. The first degrade of a process is therefore also written straight to stderr: `ggml-qnn: NPU degraded (<reason>), claiming no ops for the rest of this process: everything runs on the CPU from here`. So are a session that cannot be re-created (`ggml-qnn: the NPU session could not be re-created, ...`), the first static-budget clamp (`ggml-qnn: <phase> of <key> failed (<code>) on a shape that built before: NPU weight memory is full at <X> MiB committed, ...`) and the one budget-full notice, which after a clamp reads `ggml-qnn: NPU weight memory clamped at <X> MiB after a <phase> failure (code <N>), ...`.
  A leg that never claimed anything degrades nothing and prints nothing: weights left in `CPU_REPACK`, or the IO cap at a too-large `GGML_QNN_NPAD`. The first fixable IO-cap refusal of a process goes straight to stderr and names the `GGML_QNN_NPAD` and `-ub` that fit, so `llama-bench` shows it without `-v` as well (verified 2026-09-17, when the default cap still covered fp16 graph IO: `-dev QNN` at the default `GGML_QNN_NPAD=512` printed it and left the whole model in `CPU_REPACK`; with the cap unset a K-quant model is now placed on the NPU there).
  The proof that an NPU leg ran is `GGML_QNN_STATS`: `graphs_created` and `exec_count` must be non-zero, and `exec_max_ms` in the millisecond range. The counters add up over every session in the process, and `llama-bench` builds one context per test, each baking its weights at reserve, so a `-p`/`-n` run counts each resident weight's bake once per test: the 2026-09-26 A legs report `weights_baked` 82 for 41 weights on the NPU. `exec_count` counts compute-time executes only, so dividing it by the prefill ubatches (warmup plus repetitions) gives the weights executed on the NPU per prefill ubatch.
- The 2026-09-26 driver runs (the dated subsection under "Measured comparison") are the first
  here with that proof on record, and they are repeatable from this description. One binary and
  one model for every leg; `llama-bench -t 6 -p 512 -n 64 -r 5 -o jsonl`, only the result lines
  parsed (the QnnHtp runtime prints its graph-prepare stages to stdout between them);
  `QNN_SDK_ROOT`, `ADSP_LIBRARY_PATH` and the QAIRT `lib\aarch64-windows-msvc` directory on
  `PATH` set once for all legs; per leg only `GGML_QNN_DISABLE`, `GGML_QNN_NPAD`,
  `GGML_QNN_STATIC_BUDGET_MB` and `GGML_QNN_STATS`, cleared between legs. Counterbalanced order
  A B C D D C B A with a 120 s cooldown between legs, so both NPU configurations are paired. A
  gate before every leg (and at batch start and end) requires AC power, charge at or above 40%,
  no other llama, genie or qnn process, and a 10 s `\Processor(_Total)\% Processor Time` window
  averaging under 15%, retried up to six times, and logs the charge and idle CPU it saw. Through
  each leg a 2 s sampler of the same counter runs and is averaged per leg; charge percent and AC
  state are read again after the leg. Every NPU leg has `GGML_QNN_STATS` pointed at its own
  file, so `weights_baked`, `exec_count`, `exec_slow` and `exec_max_ms` sit next to its timing
  (each file covers both of that leg's contexts, pp512 and tg64), and stderr is kept for the
  budget line. The driver script and the raw per-leg output (jsonl,
  stats, stderr, driver log) are kept outside the repository. Two things this design did not
  catch, both recorded after the fact: a mid-leg AC loss (leg 7, charge 100 -> 96% inside the
  leg; the gate runs before a leg, not during, and the run ended before leg 8), and background
  work that is not a llama, genie or qnn process (Disk Cleanup during run 1). Sample the clock
  per tick as well; today's within-leg drift went unattributed for want of it.

## Known limitations

- Windows ARM64 only in practice (the dlopen paths exist for Linux but are untested).
- fp16 math internally: F32 elementwise cannot meet strict 1e-7 tolerances (matmul
  tolerances pass).
- 512-class static bakes (a 512-wide weight at the default `GGML_QNN_NPAD=512`, 1 MiB of
  padded IO at fp32) were seen hanging at the validation execute on QAIRT 2.45, first on
  battery and then on AC. That, and the rest of the IO-size law under `GGML_QNN_NPAD`, was
  measured only on the mixed-dtype path that `e3ba3f695` replaced, whose reference kernel was
  slow in its own right. On fp16 graph IO the law does not hold: the same bakes, and larger
  ones up to 2.5 MiB of padded output, passed on 2026-09-27 with the cap lifted, and so did
  Qwen3-4B at `-ub 512`. So the default cap now covers fp32 graph IO only (an F32 weight, or
  `GGML_QNN_NO_F16_IO`), where no measurement has shown it can go; for those, pick `GGML_QNN_NPAD`,
  with a matching `-ub`, by weight width so that `NPAD * max(K, M) * 4` stays under 1 MiB: 32
  for 4096-wide projections, 16 (below `GGML_QNN_MIN_DIM`, so the CPU) for a 9728-wide FFN;
  see the `GGML_QNN_NPAD` row.
- The NPU path does not reproduce the CPU's output. On Qwen3-4B-Q4_K_M (2026-09-27, wikitext-2,
  8 chunks) its mean KL divergence from a CPU reference is 0.047 with a maximum of 33.7-34.5,
  and the top token differs at about 3% of positions, at the 32 and at the 512 bucket alike,
  where a CPU control without repack reads 0.000000. The cause is not identified; fp16 range on
  activations or outputs and fp16 accumulation are candidates, neither verified. See the
  2026-09-27 subsection under Measured comparison.
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

The fork is kept current by merging, not rebasing: master last merged ggml-org master at
`9adc7f420` on 2026-09-27 (merge `9d4aecc0c`, 102 upstream commits after the 2026-09-22 merge
at `e6ab7c1a4`).

- There is no QNN backend upstream; `ggml/src/ggml-qnn/` and its wiring carried over as-is.
  Upstream changed none of the QNN files and none of the ggml-backend interface files
  (`ggml-backend-impl.h`, `ggml-backend.h`, `ggml-backend.cpp`, `ggml-backend-reg.cpp`) in that
  range, so the backend's positional interface initializers still line up.
- The only conflicts were seven workflow files that upstream edited and the fork had deleted.
  They stay deleted: the fork runs no GitHub Actions, and `scripts/fork-ci.sh` (see Testing) is
  its CI.
- Still carried, with no upstream equivalent: the `--list-devices` `fflush` fix, the matching
  stdout flush at the end of `test-backend-ops` and, like it, at the end of
  `test-backend-sampler` (redirected stdout is buffered, and a backend DLL can end the process
  during teardown before the CRT flushes it), and the KleidiAI COFF patch, which is still
  needed (upstream still fetches KleidiAI v1.24.0, whose `.S` files lack the branch the patch
  adds).
- The fork's changes to upstream's OpenCL backend: the recoverable staging- and large-buffer
  allocation (`70a556600`, `cc8b36895`), a superset of upstream #27630 that the 2026-09-22
  merge kept; the Q5_K readback fix (`820d1835b`: upstream `a25c9865f` transposes a Q5_K
  tensor's scales in `set_tensor` only for the X2 binary kernels, but `get_tensor` always read
  the transposed copy, so reading a Q5_K weight back returned garbage on the X1-85 and
  `test-backend-ops` failed 8 of its MUL_MAT q5_K cases); an f16 ADD computed in f32 and
  rounded once, as the CPU does; the MoE router reorder reading the routing ids with their
  real row stride, which is not the expert count when the ids are compact; MUL_MAT_ID with
  `GGML_PREC_F32` on its activations (set when they can exceed the f16 range) kept off the
  Adreno MoE GEMMs that run for more than one token and narrow them to f16 or q8_1; the Q8_0
  and Q4_0 SoA lookup keyed on the tensor's SoA extra instead of its address, so a struct copy
  of a weight (`gguf_add_tensor` keeps one, and writing a GGUF reads the weight back through
  it) is restored from SoA rather than read as a plain buffer, which `test-gguf` failed on; the
  ops `test-backend-sampler`'s sampler chains needed and the backend lacked - ARGMAX, SUM, LOG
  (f32 and f16), CPY between f32 and i32, GET_ROWS of i32 rows, ARGSORT past one workgroup's
  row width (multi-pass, now also taking strided input, which used to assert) and TOP_K, which
  now runs on the GPU instead of falling back to the CPU and is slower than the CPU on
  vocabulary-wide rows - with ARGSORT and TOP_K declining a shape whose sort scratch would
  exceed the device's maximum allocation or 256 MiB; and f16 SGN,
  STEP, CEIL and FLOOR deciding from the half's bit pattern rather than from float arithmetic,
  because the X1-85 flushes f16 subnormals to zero (its `CL_DEVICE_HALF_FP_CONFIG` has no
  `CL_FP_DENORM`), which made `test-backend-ops` fail SGN now and then on a subnormal input.
  The fork's CONV_2D contiguity gate is gone: upstream #28503 (`7d701b592`) passes the strides
  to the kernel.
- One upstream change in the merge bears on this document: `965f89794` gave `llama-bench`
  `--repack 0|1` (see Measured comparison).
