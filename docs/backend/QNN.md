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
| `test-backend-ops` MUL_MAT (F32/F16 weights) | every count on record was taken on 47 cases, under the one 1 MiB IO cap that covered every graph until 2026-09-27: 47/47 pass via `ctest -R test-backend-ops-qnn`, 40 of 40 runs clean on an idle machine, and one 46/47 seen once under heavy machine load, which has not reproduced on either IO path. With the default cap for fp16 graph IO at 10 MiB the entry executes 49 cases by the arithmetic of the case list (the two F16-weight cases with 2.0 MiB of padded input join the 47, see Testing); no 49-case run is on record yet |
| Single-matmul kernel throughput (burst clocks + static weights) | no figure claimed: the August 2026 numbers were taken without recording box load and have not been re-taken |
| Real-model inference | completes, no hangs; unsupported shapes fall back to the CPU automatically; since 2026-09-27 K-quant weights also run at `-ub 512`, the default cap on fp16 graph IO being 10 MiB where it was 1 MiB. The same default makes a QNN build offload model-scale weights without `-dev QNN` as well, wherever a weight sits in a plain host buffer; `GGML_QNN_DISABLE=1` or `GGML_QNN_IO_MAX_KB=1024` is the way back (see Routing K-quant weights to the NPU) |
| Real-model output vs the CPU | measured 2026-09-27 on Qwen3-4B-Q4_K_M (wikitext-2, 8 chunks of 512, KL divergence against a CPU reference). The first reading attributed the divergence to the NPU path against a control that also differed in flash attention, which llama disabled under `-dev QNN`: with nothing on the NPU that configuration read mean 0.030, maximum 34.3, same top token 96.9%, to every digit what the CPU alone reads with `-fa off`. That was a bug and is fixed (see Rebase notes), and the same run now reproduces the reference (mean 0.000000, same top token 100%), so it explains the divergence when nothing is on the NPU. With weights on the NPU the output still differs from the CPU's (mean 0.015-0.047, maximum 20.3-34.9, same top token 96.5-97.3% over five placements). The typical position differs by about what the CPU's own two attention paths differ by (median 0.0016-0.0018 against 0.0016). The mean and the maximum are set by a few positions that flip completely and do not rank configurations, but the rows split by placement size: those with 43 or more weights on the NPU have three or more such positions of 2040 where the others have at most two, and that is not explained. So these runs do not show that the NPU path costs accuracy, and do not show that it equals the CPU (see the 2026-09-27 subsection on the output under Measured comparison) |
| End-to-end speed vs the Adreno GPU (OpenCL) or a KleidiAI CPU build | measured on 2026-09-27 by the protocol (AC, a settled pack, a quiet machine, counterbalanced pairs of five repetitions, placement proven by the counters), twice: Qwen3-4B-Q4_K_M prefill at `-ub 512`, flash attention at `auto`, which resolves to on in this tree since `7e90a52f3` (read from the code; `llama-bench` does not log the resolved value). The first run, on the build that placed every weight the probe accepted for the NPU whatever the weight budget (`d80e59573`), read `-dev QNN` as a net loss against the same binary's CPU alone: 70.1 t/s at the default budget (41 weights on the NPU per context) and 73.6 with the budget lifted (57) against 123.0, 1.76x and 1.67x slower. The loss was the placement's, not the NPU's: `-dev QNN` with nothing on the NPU read 61.5, rep for rep what the CPU alone reads with `--repack 0` (62.1), so placing the weights for the NPU cost the repack layout and halved the CPU's prefill, and with 41 weights on the NPU the run was 14% faster than that same placement run on the CPU, with 57 it was 20% faster; how much faster the NPU runs its own share is not measured. The placement probe now charges each weight to `GGML_QNN_STATIC_BUDGET_MB` and refuses what does not fit, so a refused weight keeps `CPU_REPACK` (see the budget bullet under How it behaves), and the second run, same protocol, legs and model on the build with that change, reads `-dev QNN` at the default budget as a net win: 135.2 and 139.2 t/s (pair mean 137.2) against 128.4 for the CPU alone in the same run (+7%; the run's other CPU leg was confounded by something else using the GPU and the CPU, and is excluded) and 123.0 in the first run (+12%), rep for rep above the CPU alone. `-dev QNN` with nothing on the NPU (`GGML_QNN_STATIC_BUDGET_MB=1`) now equals the CPU alone (127.9 and 129.6), and `GGML_QNN_STATIC_BUDGET_MB=0` is unchanged at 73.4 and 75.1, 1.8x slower than the default: 0 lifts the placement limit as well as the bake limit, so all 252 weights leave `CPU_REPACK`, the device takes 57 before it clamps and the other 195 run on the CPU without repack (to put more on the NPU than the default, set a number, not 0). That is a verdict on the configuration and on this 4B model, not on the HTP alone: 41 of the 252 eligible projections ran on the NPU per context, the other 211 on the CPU with repack; the share's own speed, and whether more weights than the default beat it on the gated build, are not measured, and the second run's CPU-alone figure rests on one clean leg plus the two pairs of the first. The 2026-09-26 verdict (44-53 t/s against 101-114, ~2.2x, and ~1.6x against the CPU at `-ub 32`) is superseded: its NPU legs ran with flash attention disabled (llama disabled it under `-dev QNN`, since fixed) and at the `-ub 32` that the 1 MiB cap of the time forced, its CPU legs with flash attention on, and it had no control that separated the NPU slice from the placement. The GPU (228.1 on the settled-pack run of 2026-08-27) was not re-measured; unmeasured on 9-14B (see the dated subsections under Measured comparison; the earlier sweep's NPU leg never reached the HTP and that retraction stands) |
| Decode (single-token) offload | intentionally not claimed below 32-token ubatches (`GGML_QNN_MIN_DIM`), so it runs on the CPU in every configuration - the counters of 2026-09-26 and of 2026-09-27 show no decode execute on the HTP; every decode figure in this document is CPU decode, which is bimodal on this box (about 24.7 or about 15.9 t/s at `-t 6` on 2026-09-26) and is not quoted as a point. In both 2026-09-27 protocol runs the `-dev QNN` legs started decode at 30-34 t/s and ended at 17-27 where the CPU-alone legs read 15-17 throughout; that is not an NPU effect (the leg with nothing on the NPU reads the same), and its likely cause, the CPU clock recovering while the NPU session starts, is not proven. On the settled-pack run the two measured engines (GPU, CPU) and the CPU-run leg labelled NPU converge on decode (see Measured comparison) |

The honest summary: the kernels are fast, the eager per-op execution model is robust,
and per-op scheduling and IO copies were expected from the start to eat the advantage on
real models. What 2026-09-27 measured end to end, twice, on one dense quantized 4B model
with placement proven by the counters, every leg at `-ub 512` and flash attention at
`auto` (on in this tree, read from the code), is where the loss came from and what taking
it out is worth. On the build that placed every probe-accepted weight for the NPU whatever
the weight budget, `-dev QNN` was 1.67-1.76x slower on prefill than the CPU alone, and the
placement was what cost it: a weight placed for the NPU loses the CPU's repack layout,
which halved the CPU's prefill (123.0 t/s to 61.5, what the CPU alone reads with
`--repack 0`), and only 41-57 of the 252 projections then ran on the NPU. With 41 weights
on the NPU the run was 14% faster than that same placement run on the CPU, with 57 it was
20% faster, which did not win back what the placement lost. The placement probe now
charges each weight to the static-weight budget and refuses what does not fit, so a
refused weight keeps `CPU_REPACK`, and the same protocol on the build with that change reads
`-dev QNN` at the default budget as a net win: 137.2 t/s (the pair mean) against 128.4 for
the CPU alone in the same run and 123.0 in the first, +7% and +12%, with 41 weights on the
NPU and 211 in `CPU_REPACK`; how much faster the NPU runs its share is not measured.
`-dev QNN` with nothing on the NPU now equals the CPU alone, and
`GGML_QNN_STATIC_BUDGET_MB=0` is 1.8x slower than the default, because 0 lifts the placement
limit too and every weight leaves `CPU_REPACK` whether or not the device can hold it. The
2026-09-26 verdict, a net loss of ~2.2x, is superseded: its NPU legs ran with flash
attention disabled and at `-ub 32`, its
CPU legs with flash attention on, and it had no control to separate the NPU slice from the
rest (see Measured comparison; the earlier sweep's retraction stays). 2026-09-27
also measured the output against the CPU's. The first reading of those runs compared against
a control that differed in flash attention as well, which llama disabled under `-dev QNN`,
and is withdrawn; with that fixed, `-dev QNN` with nothing on the NPU reproduces the CPU, and
the output with weights on the NPU still differs from the CPU's, at the typical position by
about what the CPU's own two attention paths differ by. The placements with 43 or more
weights on the NPU have three or more positions of 2040 that flip completely where the
others have at most two, which is not explained. That neither shows a cost in accuracy nor
shows the two equal. The practical value
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
Two gates were put in to keep it from recurring: `supports_op` refuses any source that sits in a non-host buffer (`CPU_REPACK` is one) before it builds anything, so the upgrade pass no longer trial-builds from repacked bytes, and `GGML_QNN_IO_MAX_KB` refuses a padded IO of 1 MiB before a graph is created. The first gate is the one that covers this shape today: its graph IO is fp16 now, 512 KiB of padded output at N=64, and since 2026-09-27 the default cap for fp16 graph IO is 10 MiB (see the `GGML_QNN_IO_MAX_KB` row), because fp16 IO ran up to 9.5 MiB without a hang.
The numbers stay because they are what was measured; they do not measure the NPU.

To put K-quant weights on the NPU, run with `-dev QNN`: llama-model-loader then probes
the QNN device first and a weight the probe accepts, and the static-weight budget has room
for, resolves to a plain CPU buffer, bypassing repack (the budget check at placement is
the change that followed the first 2026-09-27 protocol run; before it every accepted weight
left `CPU_REPACK`, see Routing K-quant weights to the NPU).
When this table was taken `llama-bench` had no way to turn repack off, so without `-dev QNN`
a K-quant model on a KleidiAI+REPACK build could not reach the NPU at all. Since the
2026-09-27 upstream merge it has `--repack 0|1` (upstream `965f89794`), like `--no-repack` in
`llama-cli` and `llama-server`; that route has not been run with the QNN backend (the
first 2026-09-27 protocol run used `--repack 0` for a CPU control, with `GGML_QNN_DISABLE=1`).
F16/F32 weights stay in the plain CPU buffer and do reach the NPU. See "Routing K-quant weights to the NPU" for the caveats (since 2026-09-27 no `GGML_QNN_NPAD` is needed for an F16 or quantized weight under 10240 wide at the default `-ub 512`; a wider one, an F32 weight, any weight under `GGML_QNN_NO_F16_IO`, or any weight with `GGML_QNN_IO_MAX_KB` set still needs a `GGML_QNN_NPAD` and matching `-ub` that fit its width; when this table was taken every weight needed one).

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
In the first 2026-09-27 protocol run every pair agreed within 3%; in the second every
`-dev QNN` pair did and one of the two CPU legs was confounded (see that subsection).

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

**Superseded as a verdict by the 2026-09-27 protocol runs below.** The NPU legs here ran with
flash attention disabled and at `-ub 32`, the CPU legs with flash attention on (point 6), and
no leg separated the NPU slice from the placement. The figures stay because they are what was
measured; they describe a configuration that today's code does not run.

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
output layer is over `GGML_QNN_IO_MAX_KB` and never leaves the CPU (one default of 1024 KB
then capped every graph; since 2026-09-27 the default for fp16 graph IO is 10240 KB, which
that layer's 9.27 MiB at the 32 bucket passes). Which of them ran on the
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
fix did not change the sign. What this compares is configurations, not engines in isolation,
and they differ in more than where the matmuls ran: in repack, in the ubatch, and in flash
attention, which llama disabled in the NPU legs and not in the CPU legs (point 6).
Under `-dev QNN` every probe-accepted weight resolved to a plain CPU buffer (see Routing
K-quant weights to the NPU; on the build of that day the placement probe did not consult the
budget, which it does since the placement-budget change described under How it behaves), so
the 211 (A) and 184 (D) projections the budget left on the CPU - about 85% and 74% of the
projection weights by size - ran there without `CPU_REPACK`, while the CPU legs ran with
every weight in `CPU_REPACK`. That follows from the placement rule of that build and was not
read from a load log. D moved 27
weights, 13 of them FFN, from the unrepacked CPU to the NPU and gained 0.8 t/s, inside the A
pair's 18% spread, so the run cannot separate the NPU slice's own cost from the cost of losing
repack or of losing flash attention, and the Status section's expected cause - per-op
scheduling and IO copies - is not
established by it. The verdict is on the configuration, and on this 4B model: `-dev QNN` with
the eager NPU slice, run as here, is slower than the CPU alone. The control that
would separate the NPU slice from the rest, a leg with the same placement and no weight on
the NPU, was not run that day. On the build of that day `-dev QNN -ub 32` with
`GGML_QNN_NPAD=32` and `GGML_QNN_STATIC_BUDGET_MB=1` kept the placement, at the cap setting
of the NPU legs it is interleaved with, and let the budget refuse every bake; since the
placement-budget change `GGML_QNN_STATIC_BUDGET_MB=1` refuses every weight at placement, so
they keep `CPU_REPACK` and the leg is the CPU-alone configuration under `-dev QNN`, and the
control for what the repack layout is worth is `llama-bench --repack 0` with
`GGML_QNN_DISABLE=1`, which the upstream merge added (every weight unrepacked, where A's
placement had 211 unrepacked and 41 on the NPU). Either has to be interleaved with the NPU
legs, because this box's between-run spread (8.1% on CPU prefill, see above) would swamp a
comparison across runs. The first 2026-09-27 protocol run below ran both controls, at
`-ub 512` with flash attention at `auto` (on, read from the code); neither has been run at
`-ub 32`.

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

**6. Flash attention was not the same in every leg (found 2026-09-27).** Every leg asked for
`llama-bench`'s default `-fa auto`: `flash_attn` is -1 in each leg's result lines. On the CPU
legs that resolves to on. Under `-dev QNN` it resolved to off: layer 0 is assigned to the QNN
device, the flash attention op runs on the CPU, and llama counted that as a device mismatch
and disabled flash attention for the whole model. That is read from the code path
(`resolve_fused_ops` in `src/llama-context.cpp`, present in the build of that day), not from
a log of that day: `llama-bench` without `-v` drops the WARN that says so, which the
2026-09-27 `llama-perplexity` logs of the same configuration carry. So the NPU legs ran
without flash attention and the CPU legs with it, and the NPU legs ran at the `-ub 32` that
the 1 MiB cap of the time forced, against the CPU's own `-ub 512` in C. How much of the 2.2x
either accounts for is not known, and neither is how much of the 1.6x against B flash
attention accounts for; B ran at the same `-ub 32`, so the ubatch is no part of that ratio.
The fork no longer disables flash attention there (see Rebase notes) and K-quant weights now
run at `-ub 512`, so the verdict stands for the configuration that was measured and does not
describe today's default one. The re-measurement at `-ub 512` with flash attention at `auto`
(on in this tree, read from the code) is the pair of 2026-09-27 protocol runs below, and
their verdict supersedes this one. The other `-ub 512`
figure on record, 58.2 t/s on 2026-09-27, is two repetitions with no interleaved CPU leg and
with flash attention disabled, and is not part of that run.

### 2026-09-27: fp16 graph IO past 1 MiB, and the output against the CPU's

Qwen3-4B-Q4_K_M on the `build-npu` tree again, `-t 6` throughout. `llama-bench` reports
`build_commit 820d1835b` for its run; the `llama-perplexity` logs carry no build id, and the
records' `README.txt` gives `5e4825195` as the tree of the runs it calls "before". These are
correctness runs, not timings: other work was running on the
machine during some of them. The raw logs, the `GGML_QNN_STATS` files, the saved reference
logits, the two driver scripts (`npu-experiments.sh` and `kld-bisect.sh`) and a `README.txt`
that lists every run are kept outside the repository, in the bench records'
`2026-09-27/npu-experiments` folder.

**fp16 graph IO ran without a hang up to 9.5 MiB of padded IO; larger is unmeasured.**
`npu-experiments.sh io` ran `test-qnn-lifecycle bigstatic`, one case per process: its five
K=512 static bakes (M = 512, 640, 768, 1024 and 2560) at `GGML_QNN_NPAD` 64, 256 and 512 with
`GGML_QNN_IO_MAX_KB=1048576`. All 15 were claimed, computed and matched the CPU, finalize
16.1-32.2 ms, validation execute 2.6-3.2 ms. The largest, 512 x 2560 at the 512 bucket,
writes 2621440 bytes (2.5 MiB) by the graph's DDR summary, past the ~1 MiB that hung on the
mixed-dtype path, and built in 25.7 ms with a 3.2 ms validation execute. A model run went
further: `llama-bench -dev QNN -ub 512 -p 512 -n 0 -r 2` with `GGML_QNN_NPAD=512`, the cap
lifted and `GGML_QNN_STATIC_BUDGET_MB=0` reported `weights_baked` 58, `exec_slow` 0 and
`exec_max_ms` 18. That is 57 weights on the NPU - layers 0-7 whole, their 9728-wide FFN
projections with 9.5 MiB of padded IO each included (9728 x 512 x 2), and layer 8's `attn_q`,
1560.0 MiB by the byte arithmetic of the 2026-09-26 subsection - plus a 58th bake, a 2560 x
1024 projection whose finalize failed with 6020 and clamped the budget at 1560.0 MiB
committed; `exec_count` 171 is 57 x 3, one 512-token ubatch over the warmup and two
repetitions. pp512 came back 58.2 +- 11.8 t/s (66.5, 49.9); that is one run of two
repetitions with no interleaved CPU leg, and with flash attention disabled as in every
`-dev QNN` run before the fix described below, so it is not compared with the tables above.
`llama-perplexity` at `-ub 512` with the same settings showed the same: `weights_baked` 58,
the clamp at 1560.0 MiB, `exec_slow` 0, `exec_max_ms` 46. So on the fp16-IO path the 1 MiB cap
only forced small ubatches, and it no longer applies there: unset, `GGML_QNN_IO_MAX_KB` now
caps fp16 graph IO at 10240 KB (see its row). That is a bound on what these runs measured,
not a hang threshold. The 9.5 MiB of the 9728-wide FFN is the largest fp16 padded IO that has
run, and it passes; the next size this model has, the 151936-wide output layer at the 512
bucket (148.4 MiB), has never run and is refused. With the budget lifted the two `-ub 32`
perplexity runs below clamped at 1762.5 and 1782.5 MiB, and the three `-ub 512` runs, this
one and two perplexity runs, at 1560.0; why the 512 bucket clamps about 200 MiB lower is not
established.

**The output against the CPU's: the first reading of these runs is withdrawn.**
`llama-perplexity` on the wikitext-2 test set, `-c 512 -b 512 --chunks 8 -t 6`, KL divergence
over the 2040 scored positions against the saved logits of one CPU reference run
(`kld-base.bin`, log `kld-cpu.log`): `GGML_QNN_DISABLE=1`, `-ub 32`, default repack, and
flash attention at its default `auto`, which resolves to on in a CPU run; PPL 15.1947 +-
1.1852. The reference and the two "before, budget 0" rows are `npu-experiments.sh kld`. The
rows with a tag are `kld-bisect.sh <tag> <ub> [GGML_QNN_*=value ...]`, which runs `-dev QNN`
with `GGML_QNN_NPAD` set to `<ub>` and writes `bisect-<tag>.log` and
`bisect-<tag>.stats.txt`. The two CPU rows were run by hand: no script holds their command
lines, `README.txt` names the setting each differs from the reference by, and neither log
has a `ggml-qnn` line or a QnnHtp graph-prepare line.

"Before" is a build without the fork's change to `src/llama-context.cpp`, "after" is one with
it (see Rebase notes). Every "before" `-dev QNN` log carries the two lines
`resolve_fused_ops: layer 0 is assigned to device QNN but Flash Attention is assigned to device CPU (usually due to missing support)`
and `resolve_fused_ops: Flash Attention not supported, set to disabled`; no "after" log does,
and the reference and both CPU logs do not either.

| Run (log or tag) | Flash attention | `weights_baked` | Mean KLD | Median | 99% | 99.9% | Maximum | Same top token |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CPU `--no-repack`, control (`kld-cpu-norepack.log`) | on | - | 0.000000 | 0.000000 | 0.000038 | 0.000051 | 0.000064 | 100.000% |
| CPU `-fa off` (`kld-cpu-fa-off.log`) | off | - | 0.030161 | 0.001591 | 0.045214 | 0.288603 | 34.2501 | 96.912% |
| before, budget 1 MB (`nobake`) | off | 0 | 0.030161 | 0.001591 | 0.045214 | 0.288603 | 34.2501 | 96.912% |
| before, budget 193, `-ub 512` (`layer0`) | off | 7 | 0.030627 | 0.001742 | 0.063262 | 0.276353 | 34.6796 | 97.304% |
| before, attention projections only, `-ub 32` (`attn-only`) | off | 17 | 0.030063 | 0.001585 | 0.044969 | 0.231028 | 34.2560 | 97.108% |
| before, budget 0, `-ub 32` (`kld-npu.log`) | off | 67 | 0.046799 | 0.001801 | 0.050593 | 17.807894 | 33.6920 | 96.814% |
| before, budget 0, `-ub 512`, cap lifted (`kld-npu512.log`) | off | 58 | 0.047382 | 0.001945 | 0.056312 | 17.861994 | 34.4967 | 97.108% |
| after, budget 1 MB (`fa-nobake`) | on | 0 | 0.000000 | 0.000000 | 0.000038 | 0.000051 | 0.000064 | 100.000% |
| after, budget 193, `-ub 512` (`fa-layers1`) | on | 7 | 0.030757 | 0.001638 | 0.049950 | 0.415830 | 34.8832 | 97.010% |
| after, budget 580, `-ub 512` (`fa-layers3`) | on | 21 | 0.015188 | 0.001749 | 0.052855 | 0.437309 | 20.2611 | 97.108% |
| after, budget 1160, `-ub 512` (`fa-layers6`) | on | 43 | 0.046687 | 0.001708 | 0.047936 | 18.014572 | 33.2914 | 96.520% |
| after, budget 0, `-ub 512` (`fa-npu512`) | on | 58 | 0.045236 | 0.001849 | 0.062865 | 17.917280 | 33.9646 | 97.010% |
| after, budget 0, `-ub 32` (`fa-npu32`) | on | 68 | 0.046422 | 0.001849 | 0.071601 | 17.564983 | 33.4883 | 97.304% |

The budget is `GGML_QNN_STATIC_BUDGET_MB`. `weights_baked` counts bake attempts, so each of the
four budget-0 rows includes the one bake that failed and clamped the budget, and 66, 57, 57
and 67 weights ran on the NPU: `exec_count` is 8448 = 66 x 128, 456 = 57 x 8, 456 and 8576 =
67 x 128, 128 ubatches of 32 tokens or 8 of 512 over the 8 chunks. The budgeted rows did not
clamp (`budget_clamped` 0) and every baked weight executed: 56 = 7 x 8, 168 = 21 x 8, 344 =
43 x 8, and 2176 = 17 x 128 for the attention projections. At 192.5 MiB a layer (the byte
arithmetic of the 2026-09-26 subsection) the 7, 21 and 43 weights are layer 0, layers 0-2,
and layers 0-5 with one 5 MiB projection that still fit. Both budget-1 rows report
`weights_baked` 0 and `exec_count` 0. `exec_slow` is 0 in every row, and `exec_max_ms` 4-46
where anything executed.

- The first finding does not stand as read: its control differed in flash attention as well.
  `-dev QNN` with nothing baked reads the same as the CPU alone with `-fa off`, to every
  digit of every column: llama had disabled flash attention under `-dev QNN`, and the
  reference ran with it on. The first reading of these runs set the two "before, budget 0"
  rows against the `--no-repack` control, which ran with flash attention on, and called the
  difference the NPU path's. It is withdrawn. With the change to `src/llama-context.cpp`,
  `-dev QNN` with nothing baked reproduces the reference: the `fa-nobake` row equals the
  control. So the disabled flash attention was a real bug, and it explains the divergence
  when nothing is on the NPU. It does not explain the rows with weights there. The two
  budget-0 placements read about the same with flash attention on (mean 0.045-0.046, maximum
  33.5-34.0) as they did with it off (0.047, 33.7-34.5), so what is withdrawn is the
  comparison, not the figures.
- The control still rules out repack. The CPU running weights without repack, as it does
  under `-dev QNN` for every weight that is not baked, reproduces the reference (mean KLD
  0.000000, maximum 0.000064, the same top token everywhere).
- With weights on the NPU the output differs from the CPU's. The typical position differs
  little, and by about what the CPU's own two attention paths differ by: in the five "after"
  rows with weights on the NPU the median is 0.0016-0.0018, the 99th percentile 0.048-0.072
  and the same top token 96.5-97.3%, against 0.0016, 0.045 and 96.9% for the CPU with
  `-fa off`.
- The mean and the maximum do not rank configurations here. They are set by a few positions
  that flip completely (KLD 18-35). The maximum does not grow with the number of layers on
  the NPU: one layer reads 34.9, three layers 20.3, six layers 33.3, the budget-0 placements
  33.5-34.0, and the CPU alone with `-fa off` has such a position too (34.3). The mean is not
  monotonic in it (0.031, 0.015, 0.047, 0.045-0.046), and `llama-perplexity` prints an
  uncertainty of +-0.010 to +-0.025 on each mean, against differences of 0.016-0.032 between
  rows.
- The rows do split by placement size. The five with 43 or more weights on the NPU, flash
  attention on or off, read a mean of 0.045-0.047 and a 99.9th percentile of 17.6-18.0, and
  every row with 21 or fewer weights that differs from the reference at all reads 0.015-0.031
  and 0.23-0.44. Over 2040 positions `llama-perplexity` interpolates the 99.9th percentile
  between the fourth and the third largest value, so the first group has three or more
  positions that flip completely and the second at most two. The per-chunk column places the
  difference in the third chunk, which adds 31-35 nat in the first group and under 2 in the
  second. One or two positions of 2040 is not a ranking, and why they flip from 43 weights up
  is not explained. This text has positions where the model's output is unstable under any
  small numeric change, flash attention on against off on the CPU included.
- The PPL ratio measures none of this. The control reads 1.016 +- 0.008 against the
  reference with zero divergence. That offset comes from how `llama-perplexity` stores the
  reference (`log_softmax` in `tools/perplexity/perplexity.cpp`): before it quantizes a
  position's log-probabilities to 16 bits it floors them 16 nats below the top token's, so a
  correct token less likely than that reads back with its NLL capped. That lowers PPL(base):
  14.95 read back from the file, where the reference run itself reported 15.19, which is
  33.3 nat over the 2040 positions. The 16-bit rounding cannot do it: one step is at most
  16/65535 nat, so rounding moves that sum by at most 0.25 nat. The PPL of every row
  (15.01-15.34) sits well inside the reference's 15.19 +- 1.19.

So these runs do not show that the NPU path costs accuracy, and they do not show that it
equals the CPU. Not measured: a text without such positions, another model, any task-level
quality. The op-level MUL_MAT pass in the Status table compares single matmuls at a
tolerance, not a model's activations, and does not settle it either way.

### 2026-09-27: the protocol runs at `-ub 512`, flash attention at auto

The re-measurement the 2026-09-26 subsection called for, run twice on the same day with
the same protocol, legs, model and model file. The first run is on the build at
`d80e59573`, which placed every weight the probe accepted for the NPU whatever the weight
budget; its figures and reading stay below as the record of that build. The second run,
after it, is on the same tree plus the placement-budget change (the placement probe charges
each weight to `GGML_QNN_STATIC_BUDGET_MB` and refuses what does not fit, so a refused
weight keeps `CPU_REPACK`; see the budget bullet under How it behaves), and is the current
verdict: `-dev QNN` at the default budget is a net win over the CPU alone.

**The first run, on the build that placed every probe-accepted weight.**
Qwen3-4B-Q4_K_M on the `build-npu` tree, QAIRT 2.45.0.260326, one binary and one model for
every leg: `llama-bench -t 6 -ub 512 -p 512 -n 64 -r 5 -o jsonl`, with `GGML_QNN_NPAD` and
`GGML_QNN_IO_MAX_KB` unset (the 512 bucket, fp16 graph IO capped at 10 MiB). The driver
logged the tree at `d80e59573`; `llama-bench` reports `build_commit 5e4825195` in its result
lines, and the records do not hold the build time of the binaries. Five configurations:

- PC: the CPU alone. `GGML_QNN_DISABLE=1`, default repack.
- PR: the CPU alone without the repack layout. `GGML_QNN_DISABLE=1` and `--repack 0`.
- PN: `-dev QNN` with nothing on the NPU. On this build `GGML_QNN_STATIC_BUDGET_MB=1` kept the placement of the NPU legs, every probe-accepted weight in a plain CPU buffer without repack, and let the budget refuse every bake: `weights_baked` 0, `exec_count` 0. (Since the placement-budget change the same setting refuses every weight at placement, so they keep `CPU_REPACK` and the leg is the CPU-alone configuration under `-dev QNN`; that is what it measures in the second run.)
- PA: `-dev QNN`, the default 1024 MB budget. 41 weights on the NPU in each of `llama-bench`'s two contexts: `weights_baked` 82 is 2 x 41, and `exec_count` 246 is 41 x 6, one 512-token ubatch over the warmup and five repetitions of the pp512 test. The budget was full at 1012.5 MiB committed, as on 2026-09-26.
- PD: PA with `GGML_QNN_STATIC_BUDGET_MB=0`. 57 weights on the NPU per context: `weights_baked` 116 is 2 x 58, the 58th bake being a 2560 x 1024 projection whose finalize failed with 6020 and clamped the budget at 1560.0 MiB committed (`budget_clamped` 1), and `exec_count` 342 is 57 x 6.

Flash attention was asked for as `auto` in every leg (`flash_attn` -1 in the result lines).
On this tree `auto` resolves to on under `-dev QNN`, as it does on the CPU, since
`7e90a52f3`. That is read from the code and from the "after" `llama-perplexity` logs of the
subsection above, not from a log of these legs: `llama-bench` without `-v` drops the line
that says so (see Benchmarking notes). Three things support it without proving it for the
binary: PN equals PR within 0.3% on four of five repetitions, and PR, a CPU leg, has flash
attention on with certainty; the leg stderr shows the 10 MiB fp16 default, which landed
after the fix, so the binary was built from a working tree that held both; and
`build-npu/qnn-fused-ops.log` (15:28:57) shows `Flash Attention enabled` under `-dev QNN`
on a rebuilt binary, which proves the tree and not the run's binary.

Conditions, from the driver's gates and samples: on AC with the pack at 100% and settled, a
charge draw of 0 mW at every gate; no other llama, genie or qnn process; idle CPU 1.9-8.9% in
the 10 s window before every leg, against a gate limit of 15% that no window exceeded; 120 s
cooldowns; counterbalanced order PA PC PD PN PN PD PC PA. The two PR legs are a second batch
that started 97 s after the first ended, under the same gates, with a 90 s lead cooldown and
120 s between its legs, so PR is paired with itself and not interleaved with the others. CPU
%, clock % of base and AC state were sampled every 2 s through each leg, and no sample was
off AC. A first attempt stopped at the gate before its second leg, on CPU windows of
15.6-47.9% from other work on the machine; its one leg is not used. `+-` is llama-bench's
stddev over the five repetitions.

| Leg | Config | pp512 t/s | Its five repetitions | tg64 t/s (CPU decode in every leg) | Its five repetitions | CPU during leg | Clock, % of base: mean (lowest - highest sample) | `GGML_QNN_STATS` |
|---|---|---:|---|---:|---|---:|---|---|
| 1 | PA: `-dev QNN`, 1024 MB budget | 69.59 +- 18.19 | 85.33 81.28 70.11 39.06 72.18 | 27.53 +- 7.56 | 33.58 33.26 32.06 21.09 17.66 | 40.8% | 86.7 (63.5 - 110.7) | weights_baked 82, budget_clamped 0, exec_count 246, exec_slow 0, exec_max_ms 14 |
| 2 | PC: CPU alone | 121.66 +- 15.68 | 136.29 130.60 126.04 119.54 95.85 | 17.10 +- 0.10 | 17.04 17.05 17.08 17.06 17.28 | 51.2% | 79.2 (64.0 - 105.2) | backend disabled |
| 3 | PD: `-dev QNN`, budget lifted | 74.73 +- 19.91 | 90.62 86.89 78.82 40.61 76.72 | 27.34 +- 7.67 | 33.50 33.20 31.90 20.59 17.51 | 37.1% | 89.0 (62.7 - 111.7) | weights_baked 116, budget_clamped 1, exec_count 342, exec_slow 0, exec_max_ms 14 |
| 4 | PN: `-dev QNN`, nothing baked | 61.45 +- 12.38 | 76.86 72.59 55.99 50.90 50.93 | 28.18 +- 6.13 | 31.88 31.81 31.07 28.69 17.45 | 50.3% | 78.9 (64.0 - 100.8) | weights_baked 0, budget_clamped 0, exec_count 0 |
| 5 | PN: `-dev QNN`, nothing baked | 61.60 +- 12.32 | 76.82 72.72 56.61 50.95 50.92 | 26.48 +- 6.70 | 31.77 31.71 30.23 21.32 17.35 | 50.7% | 78.6 (64.0 - 100.3) | weights_baked 0, budget_clamped 0, exec_count 0 |
| 6 | PD: `-dev QNN`, budget lifted | 72.56 +- 20.26 | 90.78 86.66 78.96 40.55 65.82 | 27.77 +- 6.62 | 31.90 33.24 31.57 24.55 17.60 | 37.7% | 90.1 (64.0 - 111.2) | weights_baked 116, budget_clamped 1, exec_count 342, exec_slow 0, exec_max_ms 13 |
| 7 | PC: CPU alone | 124.37 +- 14.01 | 138.03 131.90 127.14 123.49 101.27 | 16.82 +- 0.03 | 16.79 16.81 16.80 16.85 16.85 | 50.9% | 79.4 (64.0 - 105.1) | backend disabled |
| 8 | PA: `-dev QNN`, 1024 MB budget | 70.53 +- 18.03 | 85.90 81.92 72.83 40.08 71.92 | 27.34 +- 7.71 | 33.53 33.19 31.98 20.55 17.46 | 40.5% | 87.8 (64.1 - 111.2) | weights_baked 82, budget_clamped 0, exec_count 246, exec_slow 0, exec_max_ms 13 |
| batch 2, 1 | PR: CPU alone, `--repack 0` | 62.39 +- 12.14 | 77.03 72.65 60.58 50.85 50.85 | 17.24 +- 0.13 | 17.02 17.35 17.28 17.30 17.23 | 53.7% | 76.9 (64.0 - 100.1) | backend disabled |
| batch 2, 2 | PR: CPU alone, `--repack 0` | 61.72 +- 12.11 | 76.49 72.55 57.86 50.84 50.86 | 16.90 +- 0.36 | 16.29 16.94 17.17 16.93 17.20 | 54.2% | 75.8 (64.0 - 100.1) | backend disabled |

| Config | Weights on the NPU per context | pp512 t/s, mean of the pair | Against PC | Against PN |
|---|---:|---:|---:|---:|
| PC: CPU alone | - | 123.0 | - | - |
| PR: CPU alone, `--repack 0` | - | 62.1 | 1.98x slower | - |
| PN: `-dev QNN`, nothing baked | 0 | 61.5 | 2.00x slower | - |
| PA: `-dev QNN`, 1024 MB budget | 41 | 70.1 | 1.76x slower | +14% |
| PD: `-dev QNN`, budget lifted | 57 | 73.6 | 1.67x slower | +20% |

**1. Placement is proven, and every execute was fast.** `weights_baked` 82 (PA) and 116 (PD)
over both contexts, `exec_count` 246 and 342, `exec_slow` 0 and `exec_max_ms` 13-14, at the
512 bucket, where the 9728-wide FFN projections have 9.5 MiB of padded IO. PN's counters are
zero: nothing in those legs ran on the NPU.

**2. `-dev QNN` costs the repack layout, and that halves the CPU's prefill.** PN equals PR
rep for rep (first legs: 76.86, 72.59, 55.99, 50.90, 50.93 against 77.03, 72.65, 60.58,
50.85, 50.85), so what `-dev QNN` costs with nothing on the NPU is exactly the repack layout
the weights lose when they are placed for the NPU. Against the CPU alone with repack that is
123.0 t/s down to 61.5-62.1.

**3. The NPU speeds its own share up.** With weights on the NPU the run is faster than that
same placement on the CPU: 70.1 with 41 weights (+14%) and 73.6 with 57 (+20%) against PN's
61.5. By how much the NPU speeds its own share up is not measured: the gain is the whole
run's, and the legs with weights on the NPU also ran at a higher CPU clock (86.7-90.1% of
base against 78.6-78.9%).

**4. Against the CPU alone the configuration was still a net loss, 1.67-1.76x.** 123.01 /
70.06 is 1.76 and 123.01 / 73.64 is 1.67 (pair means before rounding to the one decimal the
table shows): 41-57 of the 252 projections ran on the NPU, and the other ~200 ran on the CPU
without repack. The verdict is on the configuration and on this 4B model: `-dev QNN` with
the eager NPU slice, at `-ub 512` with flash attention at `auto`, was slower than the CPU
alone on this build, and what made it slower was where the weights were placed, not what the
NPU did with the ones it got. That is what the placement-budget change removes, and what the
second run below measures.

**5. Noise.** The pairs agree within 3% on prefill. Within a leg the repetitions fall: the
CPU clock steps down under sustained load, from 100.1-111.7% of base at a leg's highest
sample to 62.7-64.1% at its lowest, so each mean carries a stddev of 12-20 t/s. Compare legs
by their pair means and rep by rep, not by one repetition. Every leg with weights on the NPU
has one slow repetition, the fourth (39-41 t/s). That is not explained.

**6. Decode is CPU work in every leg.** A 1-token batch is under `GGML_QNN_MIN_DIM`, and
`exec_count` has no decode term (246 is 41 x 6 exactly). So the faster start of decode under
`-dev QNN` is not an NPU effect: those legs start at 32-34 t/s and end at 17.5, PN with
nothing on the NPU included, and the CPU-alone legs read 17 throughout, with or without
repack. PN and PR run the same prefill repetitions and take the same wall time (68.8-69.8 s)
although PN's five decode repetitions take about 6 s less (11.98 and 12.84 s against 18.57
and 18.94 s), so the `-dev QNN` legs spend 5-6 s more outside the measured repetitions
(13.2-13.9 s against 8.1-8.2 s, the warmup included in both), in two NPU session starts. The
likely reading is that the CPU clock recovers in that pause and decode starts in its fast
mode. That is not proven: the clock was sampled every 2 s, not per repetition. **CPU decode
is deliberately not reported**, as above.

**7. Not measured.** The GPU, which `build-npu` does not have; any other model; these five
configurations at `-ub 32`; a QNN build run without `-dev QNN` and without
`GGML_QNN_DISABLE`.

The driver (`bench-picks.ps1 -Batch npu`), its log and, per leg, the raw `llama-bench`
output, stderr, `GGML_QNN_STATS` file and 2 s samples are kept outside the repository, in the
bench records' `2026-09-27/npu-speed` folder, with a `README.txt` that carries these figures
and this reading.

**The second run, on the build with the placement budget.** The same protocol, legs, model
and binary tree as the first run, plus the then-uncommitted placement-budget change in
`ggml/src/ggml-qnn`: the placement probe charges each weight to `GGML_QNN_STATIC_BUDGET_MB`
and refuses what does not fit beside the weights already placed, so a refused weight keeps
`CPU_REPACK` (see the budget bullet under How it behaves). The change landed after this run;
`llama-bench` reports `build_commit d80e59573` in its result lines and the driver logged the
same tree. `bench-picks.ps1 -Batch npu -Reps 5 -Cooldown 120 -LeadCooldown 120 -LegTimeout
480 -GateWindows 30`, one batch of eight legs (`run-npu-20260927-162809`) in the order PA PC
PD PN PN PD PC PA, 120 s of lead cooldown and 120 s between legs; PR was not re-run. The
gates and samples are the first run's: AC with the pack at 100% and settled (a charge draw
of 0 mW at every gate), no other llama, genie or qnn process, idle CPU 2.0-7.6% in the 10 s
window before every leg against the 15% limit, and no sample off AC. Flash attention at
`auto` again (`flash_attn` -1 in every result line), which resolves to on in this tree
(read from the code, as above). The leg commands are the first run's; what the settings mean
under the placement budget:

- PC: the CPU alone. `GGML_QNN_DISABLE=1`, default repack.
- PN: `-dev QNN`, `GGML_QNN_STATIC_BUDGET_MB=1`. Every weight is refused at placement and keeps `CPU_REPACK`: `weights_placed` 0, `placement_budget_refused` 252, `weights_baked` 0, `exec_count` 0. The leg is the CPU-alone configuration under `-dev QNN`, with an NPU session started and nothing on it.
- PA: `-dev QNN`, the default 1024 MB budget. 41 weights placed and 211 refused at placement, which keep `CPU_REPACK`: `weights_placed` 41, `placement_budget_refused` 211, `weights_baked` 82 (2 x 41, one bake per context), `exec_count` 246 (41 x 6), `exec_slow` 0, `exec_max_ms` 11-12. By the byte arithmetic of the 2026-09-26 subsection they are the 41 the first run baked: layers 0-4 whole, layer 5's attention, and layer 6's `attn_k` and `attn_v`, which still fit after layer 5's first FFN projection was refused at 1012.5 MiB placed in 39 weights.
- PD: `-dev QNN`, `GGML_QNN_STATIC_BUDGET_MB=0`. 0 lifts the placement limit as well as the bake limit, so all 252 weights leave `CPU_REPACK` (`weights_placed` 252, `placement_budget_refused` 0), the device takes 57 before it clamps (`weights_baked` 116 = 2 x 58, the 58th bake the 2560 x 1024 projection whose finalize failed with 6020 at 1560.0 MiB committed, `budget_clamped` 1, `exec_count` 342 = 57 x 6, `exec_max_ms` 14-23) and the other 195 run on the CPU without repack, as every unbaked weight did in the first run.

| Leg | Config | pp512 t/s | Its five repetitions | tg64 t/s (CPU decode in every leg) | Its five repetitions | CPU during leg | Clock, % of base: mean (lowest - highest sample) | Wall | `GGML_QNN_STATS` |
|---|---|---:|---|---:|---|---:|---|---:|---|
| 1 | PA: `-dev QNN`, 1024 MB budget | 135.21 +- 4.58 | 141.12 138.15 135.07 132.03 129.70 | 30.02 +- 1.97 | 30.33 31.70 31.31 30.04 26.71 | 38.0% | 96.7 (80.0 - 110.8) | 53.7 s | weights_placed 41, placement_budget_refused 211, weights_baked 82, budget_clamped 0, exec_count 246, exec_slow 0, exec_max_ms 12 |
| 2 | PC: CPU alone, confounded (point 5) | 105.78 +- 21.69 | 128.94 121.56 111.30 89.85 77.27 | 15.73 +- 0.58 | 15.04 15.36 15.89 15.77 16.58 | 61.4% | 73.4 (56.1 - 103.9) | 52.7 s | backend disabled |
| 3 | PD: `-dev QNN`, budget 0 | 73.43 +- 18.74 | 87.88 86.01 77.64 41.50 74.13 | 28.45 +- 7.00 | 33.53 33.37 32.54 25.31 17.52 | 39.1% | 89.2 (63.2 - 109.4) | 84.7 s | weights_placed 252, placement_budget_refused 0, weights_baked 116, budget_clamped 1, exec_count 342, exec_slow 0, exec_max_ms 23 |
| 4 | PN: `-dev QNN`, budget 1 | 127.88 +- 8.65 | 138.30 132.70 128.08 125.08 115.25 | 28.85 +- 2.71 | 30.36 30.47 30.28 29.02 24.11 | 47.7% | 88.4 (73.6 - 106.8) | 45.3 s | weights_placed 0, placement_budget_refused 252, weights_baked 0, budget_clamped 0, exec_count 0 |
| 5 | PN: `-dev QNN`, budget 1 | 129.61 +- 7.46 | 139.17 134.06 129.44 125.47 119.89 | 26.19 +- 5.43 | 30.46 29.56 29.01 24.52 17.38 | 48.0% | 87.7 (73.6 - 103.7) | 45.1 s | weights_placed 0, placement_budget_refused 252, weights_baked 0, budget_clamped 0, exec_count 0 |
| 6 | PD: `-dev QNN`, budget 0 | 75.07 +- 19.74 | 91.10 87.89 84.72 68.09 43.52 | 28.92 +- 6.75 | 33.43 33.26 32.18 28.30 17.44 | 38.0% | 92.3 (63.7 - 110.8) | 80.6 s | weights_placed 252, placement_budget_refused 0, weights_baked 116, budget_clamped 1, exec_count 342, exec_slow 0, exec_max_ms 14 |
| 7 | PC: CPU alone | 128.39 +- 8.67 | 138.97 133.44 128.65 124.73 116.16 | 17.05 +- 0.17 | 17.10 16.99 16.89 16.93 17.33 | 51.0% | 82.2 (64.0 - 104.8) | 45.1 s | backend disabled |
| 8 | PA: `-dev QNN`, 1024 MB budget | 139.24 +- 6.11 | 147.67 143.03 137.67 135.46 132.39 | 29.75 +- 4.01 | 31.42 32.47 31.79 30.37 22.71 | 37.8% | 95.0 (74.8 - 110.8) | 53.0 s | weights_placed 41, placement_budget_refused 211, weights_baked 82, budget_clamped 0, exec_count 246, exec_slow 0, exec_max_ms 11 |

| Config | Weights on the NPU per context | Weights in `CPU_REPACK` | pp512 t/s, mean of the pair | Against PC in this run (128.4) | Against PC in the first run (123.0) |
|---|---:|---:|---:|---:|---:|
| PC: CPU alone | - | 252 | 128.4 (leg 7 alone) | - | +4%, inside the between-run spread |
| PN: `-dev QNN`, budget 1 | 0 | 252 | 128.7 | equal (+0.3%) | +5%, inside the between-run spread |
| PA: `-dev QNN`, 1024 MB budget | 41 | 211 | 137.2 | +7% | +12% |
| PD: `-dev QNN`, budget 0 | 57 | 0 | 74.3 | 1.73x slower | 1.66x slower |

**1. Placement is proven, and the ledger's counters read as the code says.** Every PA leg
reports `weights_placed` 41 and `placement_budget_refused` 211, 252 in all, and its
`weights_baked` 82 is twice `weights_placed`: the two placement counters count each distinct
weight once per ledger lifetime where `weights_baked` counts a bake per context (measured
here, over `llama-bench`'s two contexts). PN reports 0 placed, 252 refused, nothing baked
and nothing executed; PD 252 placed, 0 refused, 57 baked and executed and a 58th bake that
failed. Every execute was fast: `exec_slow` 0, `exec_max_ms` 11-23.

**2. `-dev QNN` with nothing on the NPU now costs nothing.** PN reads 127.88 and 129.61
against 128.39 for the CPU alone in this run and 121.7-124.4 in the first. In the first run
PN read 61.5, rep for rep what `--repack 0` reads, because that build sent every
probe-accepted weight to a plain CPU buffer whether or not the NPU would bake it. Now a
weight the budget will not take stays in `CPU_REPACK`.

**3. At the default budget the NPU is a net win, the first on this box.** PA reads 135.21
and 139.24 (pair mean 137.2) against 128.39 in the same run (+7%) and 123.0 in the first run
(+12%). Rep for rep PA is above PC leg 7 (leg 1: +2, +4, +5, +6, +12%; leg 8: +6, +7, +7,
+9, +14%) and above both PC legs of the first run on every repetition. 41 of the 252
projections run on the NPU and the other 211 on the CPU with repack. The PA legs run the CPU
at 38% against 51% for the CPU alone, and their clock averages 95-97% of base against 82%.
The gain is the whole run's: how much faster the NPU runs its own share is not measured, and
this run's CPU-alone figure is one clean leg.

**4. `GGML_QNN_STATIC_BUDGET_MB=0` is unchanged, and now 1.8x slower than the default.** PD
reads 73.43 and 75.07 against 74.73 and 72.56 in the first run. 0 lifts the placement limit
as well as the bake limit, so all 252 weights leave `CPU_REPACK`, the device takes 57 before
it clamps and the other 195 run on the CPU without repack: the first run's configuration,
which the placement budget does not touch. To put more on the NPU than the default 1024 MB,
set a number, not 0. Whether more weights on the NPU than the default beat it is not
measured on the gated build (in the first run 57 weights were 5% faster than 41, both on the
ungated placement).

**5. PC leg 2 is confounded and excluded.** Its samples show GPU 3D at 16-35% for 8 s and
21% once more where every other leg reads 0.1-0.3%, the clock fell to 56% of base for 12 s,
and the CPU read 57-78% against 51-55% in the clean CPU leg: something else used the GPU and
the CPU during it (not identified; the gate 10 s before it read idle 7.6%, the highest of
the run). Its 105.78 is not used. This run's CPU-alone figure is leg 7, 128.39, in line with
the 121.66 and 124.37 of the first run; the +7% rests on that one leg and the +12% on the
first run's pair.

**6. Noise.** Within a leg the repetitions fall as the CPU clock steps down under sustained
load, as before, so each mean carries a stddev; compare legs by pair means and rep by rep,
not by one repetition. The PA legs are the tightest `-dev QNN` legs on record (stddev
4.6-6.1 against 18.0-18.2 in the first run): the slow fourth repetition every NPU leg had in
the first run is gone from PA, and PD still has one repetition at 41.5 and 43.5.

**7. Decode: the same reading as the first run.** The `-dev QNN` legs (PA, PD and PN alike)
read 26-30 t/s mean against 15.7-17.1 for the CPU alone, starting at 30-34 and ending at
17-27. Decode runs on the CPU in every leg and PN has the same weight layout as PC, so this
is not an NPU effect. PN and PC leg 7 take the same wall time (45.3 and 45.1 s against
45.1 s) although PN's five decode repetitions are about 7 s shorter by the mean rates (320
tokens at 28.9 and 26.2 t/s against 17.05), so a `-dev QNN` process spends that time outside
the measured repetitions (two NPU session starts), and the likely reading, as before, is
that the CPU clock recovers in that pause and decode starts in its fast mode. Not proven:
the clock is sampled every 2 s, not per repetition. **CPU decode is deliberately not
reported**, as above.

**8. Not measured.** PR on this build (PN is no longer its twin, and the exact control for
the default placement, 41 weights on the CPU without repack and 211 with it, has no
supported setting; a test-only fault hook approximates it and is not a recipe); more
weights on the NPU than the default on the gated build; the NPU's own share; the GPU; any
other model; these configurations at `-ub 32`; a QNN build run without `-dev QNN` and
without `GGML_QNN_DISABLE`. The accuracy caveat of the subsection above (a few positions
flip completely with 43 or more weights on the NPU) is not affected by this run and was not
re-measured.

The stderr of every PA leg carries the first placement refusal of the process, in the
wording of the build measured: `ggml-qnn: the 1024 MiB static-weight budget has no room for
blk.5.ffn_gate.weight (47.5 MiB) beside the 1012.5 MiB placed at model load in 39 weights:
it and every later weight that does not fit stay with the CPU; GGML_QNN_STATIC_BUDGET_MB
raises the budget, 0 lifts it. later placement refusals are logged at DEBUG` (the PN legs
name `blk.0.attn_q.weight (20.0 MiB) beside the 0.0 MiB placed at model load in 0 weights`;
the PD legs refuse nothing and carry the clamp lines instead). The landed build words the
tail differently, see the budget bullet under How it behaves. Every `-dev QNN` leg also
carries the IO-cap notice about the output layer, with this build's tail `later IO-cap
refusals at model load are logged at DEBUG`.

The driver, its log and, per leg, the raw `llama-bench` output, stderr, `GGML_QNN_STATS`
file and 2 s samples are kept outside the repository, in the bench records'
`2026-09-27/npu-placement` folder, with a `README.txt` that carries these figures and this
reading.

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
It does the same in a run with no GPU and no `-dev QNN`, for every weight that sits in a
plain host buffer, see "Routing K-quant weights to the NPU".

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
- Before any of that, `supports_op` does the checks that are arithmetic on the shape and need no session. A matmul whose padded IO would reach the IO cap, or overflow the 32-bit sizes QNN takes, is reported as not supported: it runs on the CPU and is never a failed op (a claimed node that the graph policy then refuses does not fall back, it fails the graph). The cap is `GGML_QNN_IO_MAX_KB` for every matmul graph when that is set; unset, it is 1024 KB for fp32 graph IO (an F32 weight, or any weight under `GGML_QNN_NO_F16_IO`) and 10240 KB for the fp16 graph IO of an F16 or quantized weight.
  A weight probed for placement at load has no real batch yet, so the smallest bucket (the `GGML_QNN_NPAD` floor) is used: a weight capped even there is not placed on the NPU, so on a KleidiAI build it keeps `CPU_REPACK` instead of landing in a plain CPU buffer it would never leave. At the default `GGML_QNN_NPAD=512` with the cap unset that is every F32 weight 512 or more wide and every F16 or quantized weight 10240 or more wide, the 151936-wide output layer among them; F16 and quantized weights under 10240 wide are placed. With `GGML_QNN_IO_MAX_KB=1024` set it is every F16 or quantized weight 1024 or more wide, i.e. all of a model-scale model, which is how every run before 2026-09-27 behaved.
  The first IO-cap refusal that a setting can fix is printed once per process at model load and once more at schedule time, straight to stderr; it names the cap that refused it (a set `GGML_QNN_IO_MAX_KB`, the default for fp16 graph IO, or the default for fp32 graph IO, with `GGML_QNN_NO_F16_IO` named when that is what selected it) and the `GGML_QNN_NPAD` and `-ub` that fit that shape. For the output layer of Qwen3-4B at the defaults it reads `ggml-qnn: a q6_K 2560x151936 matmul stays on the CPU: its padded IO is 148.4 MiB at N=512, at or over the 10.0 MiB cap (the GGML_QNN_IO_MAX_KB default for fp16 graph IO); GGML_QNN_NPAD=32 with -ub 32 (or lower) fits it. later IO-cap refusals at model load are logged at DEBUG` (the schedule-time one ends `of a batch are logged at DEBUG`; every `-dev QNN` leg of the two 2026-09-27 protocol runs has that line on stderr, the first run's with the tail of that build, `later IO-cap refusals are logged at DEBUG`). Since llama keeps the output layer in `CPU_REPACK` under `-dev QNN` (see "Routing K-quant weights to the NPU") that weight is never offered to the QNN device, so at the defaults this model loads with no IO-cap notice at all (verified on the landed build; the 2026-09-27 protocol runs predate the routing and carry it), and the one `ggml-qnn:` notice of a default run is the placement refusal described in the budget bullet below. The notice reports what fits the cap and is not advice to change the bucket, see "Routing K-quant weights to the NPU". A shape that fits only below `GGML_QNN_MIN_DIM` (at a 1 MiB cap: a 151936-wide output layer; a 9728-wide FFN only with an F32 weight or under `GGML_QNN_NO_F16_IO`) can never run here at that cap, so it stays at DEBUG and does not use up either notice; later refusals of the same kind are DEBUG.
  A source in a non-host buffer is refused as well, see "Routing K-quant weights to the NPU". An exception inside the trial build (host out of memory) is an ERROR and the op is not claimed.
- Model weights are baked into their graphs once (dequantized to fp16 when quantized) in
  HTP-native layout, within a memory budget (default 1024 MB, a conservative choice: in one run on a
  machine loaded with other work the HTP stopped mapping baked weights at about 1170 MiB
  committed; on 2026-09-26, on a near-idle machine with the budget lifted, it clamped at 1830.0
  MiB, where the 47.5 MiB bake that failed would have brought it to 1877.5, and the default
  stays below both). Weights past the budget
  stay on the CPU, and since the change that followed the first 2026-09-27 protocol run the
  budget is applied at placement as well: llama's weight-placement probe (`supports_op` on a
  weight with no data yet) charges each weight it would accept to a placement ledger, at the
  bytes its bake will take (the fp16 on-device size; four times the element count for an F32
  weight), and refuses the weight when the bytes placed so far plus its own exceed
  `GGML_QNN_STATIC_BUDGET_MB` (an exact fit is accepted): first fit in llama's load order, so
  a refused weight keeps `CPU_REPACK` on a KleidiAI build and every later weight that fits is
  still placed. The ledger is asked last, so a weight the minimum dimension, the IO cap, the
  quantized gate, a degraded or lost session or the denylist refuses is never charged. A
  weight is keyed by name, type and its two leading dimensions and charged once per ledger
  lifetime, so a `no_alloc` graph re-probe, or a second probe of a weight the ledger already
  holds, does not charge it again; the budget is read per call, 0 never refuses (every weight
  is placed), and `GGML_QNN_NO_STATIC_WEIGHTS` keeps no ledger, since quantized weights are
  then unclaimable and stay in `CPU_REPACK`. The ledger is cleared when the last QNN backend
  of the process is freed, and the session with it (the QNN device's buffer type is the CPU
  one, so the backend never sees a weight buffer freed): models alive at the same time share
  it, and a model loaded after that reset starts fresh. A `-fit` dry run is such a reset:
  `llama-completion`'s default `-fit on` loads the model once with a context for the dry run
  and frees it before the real load, so the real load places the same weights again (the
  placement is the same either way, and the counters count both loads, see `GGML_QNN_STATS`).
  The trade-off that leaves: two models loaded with no
  live context between them are each placed up to the budget, and at schedule time the second
  one's bakes are refused and its placed weights run on the CPU without repack. The charge is
  the one a bake makes and the test the one the bake policy makes, against the budget a
  session starts with, so the placed weights fit together whatever order they bake in, as
  long as each bakes once and nothing else is charged. Three things break that and leave a
  placed weight unbaked, on the CPU without repack: a second pad bucket (`-ub` over
  `GGML_QNN_NPAD`: each bucket is its own bake and its own charge), a host weight the probe
  never saw (a layer llama assigned to the CPU under a partial `-ngl` is still offered by the
  scheduler, first in graph order), and a device that clamps the session budget below this
  one (with 0 nothing is refused at placement at all: on 2026-09-27
  `GGML_QNN_STATIC_BUDGET_MB=0` placed all 252 projections of Qwen3-4B, the device took 57
  and clamped, and the other 195 ran on the CPU without repack, 1.8x slower than the default;
  to put more on the NPU, set a number). The first placement refusal of a process is written
  straight to stderr, so `llama-bench` without `-v` shows it, and on the landed build it is
  the only `ggml-qnn:` notice a default run of Qwen3-4B-Q4_K_M prints (the INFO lines about
  the budget, the HTP arch, the session and the TURBO clocks aside): `ggml-qnn: the 1024 MiB
  static-weight budget has no room for blk.5.ffn_gate.weight (47.5 MiB) beside the 1012.5 MiB
  placed at model load in 39 weights: it and every later weight that does not fit stay with
  the CPU, repacked; GGML_QNN_STATIC_BUDGET_MB=<MiB> raises the budget (0 places every
  weight, and those the device then cannot map run on the CPU without repack, about 1.8x
  slower than the default). later placement refusals are logged at DEBUG` (the build the
  second 2026-09-27 run measured ended it `GGML_QNN_STATIC_BUDGET_MB raises the budget, 0
  lifts it. later placement refusals are logged at DEBUG`); later refusals get one DEBUG line
  per distinct weight, `ggml-qnn: <name> <type> <ne0>x<ne1>: static weight needs X MiB, Y of
  Z MiB placed, staying with the CPU`, and `weights_placed` and `placement_budget_refused` in
  `GGML_QNN_STATS` count both outcomes.
  If a static graph still fails to build on a shape that already built and
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
  rows - the last layer's FFN after llama's output-row gather, and the output layer where it
  sits in a plain host buffer and the IO cap passes it (at the fp16 default of 10 MiB a
  vocabulary under 10240 at the 512 bucket and under 163840 at the 32 bucket, so Qwen3's
  151936 passes at 32 and not at 512; with a 1 MiB cap set, at the 32 bucket, only a
  vocabulary under 16384) - and charges them against `GGML_QNN_STATIC_BUDGET_MB`. A normal
  prefill has one output row, below `GGML_QNN_MIN_DIM`, so those run on the CPU and their
  bakes only use up budget. Under `-dev QNN` the output layer no longer gets that far: llama
  now keeps it in `CPU_REPACK` (see "Routing K-quant weights to the NPU"), so it is never
  offered to the QNN device, never charged at placement and never baked, at any
  `GGML_QNN_NPAD`: on the landed build the placement notice names the same
  `blk.5.ffn_gate.weight` refused at 1012.5 MiB placed in 39 weights at `GGML_QNN_NPAD=32` as
  at the default, so the ledger holds projections only at every bucket (verified on the
  HTP). Without that routing the placement ledger would have charged it first at
  `GGML_QNN_NPAD=32`, since llama probes the output layer before the layers: 741.9 MiB of the
  default 1024, the output layer plus 15 projections (1019.4 MiB) where 41 projections were
  baked at the 512 bucket (arithmetic on the code and the tensor list of Qwen3-4B-Q4_K_M, not
  a run). No output-layer bake has run on the device: in every run on record the budget was
  full or clamped before it, or the cap refused it. At 151936 x 2560 it is a 741.9 MiB bake,
  against 47.5 MiB for the largest that has run. Where it can still reach a bake (an output
  layer in a plain host buffer, as in a run without repack), a finalize error on that
  unproven shape denylists it, and a finalize or validation timeout (120 s and 15 s) or a
  validation execute error degrades the session, after which the placed weights run on the
  CPU without repack; `GGML_QNN_IO_MAX_KB=1024` refuses it by arithmetic at the 32 bucket and
  every larger one.
  The bake cost is
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
  whose padded IO would reach the IO cap is refused (the default of the graph's IO element
  size, or `GGML_QNN_IO_MAX_KB` when it is set). A ubatch smaller than the floor still
  executes the whole padded bucket. Dynamic graphs are bucket-padded the same way.
  Placement is judged once, at the `GGML_QNN_NPAD` floor, and every ubatch at its own bucket, so a weight can be placed for the NPU and still be refused at a larger bucket. A ubatch of more than 512 tokens pads to 1024, where the fp16 default admits only weights under 5120 wide (under 2560 at the 2048 bucket). So with `-ub 1024` and a prompt over 512 tokens the 9728-wide FFN weights of Qwen3-4B are placed (a plain CPU buffer, no repack) but their matmuls run on the CPU for those ubatches; a ubatch of 512 tokens or fewer pads to the 512 bucket, which the cap admits, and runs on the NPU where the budget holds that bucket's bake too. The first such refusal of a process is written straight to stderr and names `GGML_QNN_NPAD=512` with `-ub 512`: placement and schedule time have one notice each, so a notice spent at model load (on the builds of the 2026-09-27 runs, the output layer's) does not silence it. Later ones are logged at DEBUG. The way out is to keep `-ub 512`, or to set `GGML_QNN_IO_MAX_KB` knowing that fp16 IO over 9.5 MiB has never run on the device. This is arithmetic on the code, not a measurement: no run at `-ub 1024` is on record. To see it, give `llama-bench -ub 1024` a prompt over 512 tokens (`-p 1024`); with the default `-p 512` no ubatch reaches the 1024 bucket.
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
  the CPU. A healthy HTP executes a matmul in milliseconds: Qwen3-4B's projections at the 512 bucket, the 9728-wide FFN included, peaked at 18 and 46 ms in two runs on 2026-09-27, at 13-14 ms in the four NPU legs of the first protocol run of that day and at 11-23 ms in the second's.
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
Pass `-dev QNN`: llama-model-loader then probes the QNN device first, a weight the probe
accepts and the static-weight budget has room for resolves to a plain CPU buffer, bypassing
repack, and the static bake sees the real bytes; a weight the budget has no room for beside
those already placed is refused at placement and keeps `CPU_REPACK` (since the change that
followed the first 2026-09-27 protocol run; the build of that run placed every accepted
weight, see the budget bullet under How it behaves).
`-dev QNN` is the route every measurement here used. `--no-repack` (`llama-cli`,
`llama-server`) and, since the 2026-09-27 upstream merge, `llama-bench --repack 0` also keep
K-quant weights out of `CPU_REPACK`; neither has been run with the QNN backend. F16/F32 weights
stay in the plain CPU buffer regardless and reach the NPU without it.

`-dev QNN` had a price on the CPU side that the placement budget removed. In the first
2026-09-27 protocol run, on the build that placed every probe-accepted weight for the NPU
whatever the budget, every placed weight lost the repack layout whether or not it was then
baked, and on Qwen3-4B-Q4_K_M that halved the CPU's prefill: 123.0 t/s for the CPU alone,
61.5 under `-dev QNN` with nothing on the NPU, 70.1-73.6 with 41-57 weights there. With the
budget applied at placement the second run read `-dev QNN` with nothing on the NPU at
127.9-129.6 against 128.4 for the CPU alone, and the default at 135.2-139.2, +7% (see
Measured comparison). The price stays for `GGML_QNN_STATIC_BUDGET_MB=0`, which refuses
nothing at placement: 73.4-75.1, 1.8x slower than the default.

Since 2026-09-27 `-dev QNN` is enough at the default `GGML_QNN_NPAD=512` for F16 and quantized weights under 10240 wide: their graph IO is fp16, which the default cap admits up to 10 MiB, so the placement probe accepts them and llama's default `-ub 512` runs every baked weight at the 512 bucket (the 2026-09-27 subsections under Measured comparison ran that, first with the cap lifted by hand and the weight budget lifted, then at the defaults in the protocol runs). Keep `GGML_QNN_NPAD` at or under `-ub`: a smaller ubatch still executes the whole padded bucket. And keep `-ub` at or under the largest bucket the widest placed weight fits: placement is judged once at the `GGML_QNN_NPAD` floor and every ubatch at its own bucket, so at `-ub 1024` a ubatch of more than 512 tokens pads to 1024, where the fp16 default admits only weights under 5120 wide, and the 9728-wide FFN weights of Qwen3-4B, placed in a plain CPU buffer without repack, run their matmuls on the CPU for those ubatches, with one stderr notice for the first such refusal of the process and DEBUG lines after it (arithmetic on the code, not a measurement; see the bucket bullet under How it behaves). Before that change one default of 1 MiB capped every graph and the probe refused every weight whose padded IO reached it at the smallest bucket, which at 512 is every F16 or quantized weight 1024 or more wide, so a model-scale model stayed in `CPU_REPACK` and nothing ran on the NPU; that is why every earlier run used `-ub 32` with `GGML_QNN_NPAD=32`, and it is still what happens with `GGML_QNN_IO_MAX_KB=1024` set. At the default an F32 weight 512 or more wide, any weight that wide under `GGML_QNN_NO_F16_IO`, and an F16 or quantized weight 10240 or more wide (a 14336-wide FFN is one) are refused the same way. The first such refusal is written straight to stderr rather than through the log callback - so it survives `llama-bench` without `-v`, but does not appear in a redirected log - and names the cap and the `GGML_QNN_NPAD` and `-ub` that fit the shape. For those, set both by weight width, see the `GGML_QNN_NPAD` row.

The output layer is the weight the 10 MiB default refuses on Qwen3-4B: 151936 wide, 148.4 MiB of padded output at the 512 bucket. On the builds of the 2026-09-27 protocol runs it kept `CPU_REPACK` at the defaults, as it did before 2026-09-27, and the one stderr notice of a default run was about it and named `GGML_QNN_NPAD=32` with `-ub 32` (the `-dev QNN` legs of both runs carry it). That line reports what fits the cap; it is not advice to change the bucket. At `GGML_QNN_NPAD=32` the layer's padded output is 151936 x 32 x 2 = 9.27 MiB, which the default passes, so on those builds it left `CPU_REPACK` for a plain CPU buffer under `-dev QNN`, ran on the CPU without repack whenever fewer than `GGML_QNN_MIN_DIM` rows were output (every single-sequence decode step and a normal prefill), and was baked only if the budget reached it. Since the placement budget that changed: llama probes the output layer before the layers, so at `GGML_QNN_NPAD=32` the ledger would have charged it first, 741.9 MiB of the default 1024, and placed it plus 15 projections (1019.4 MiB) where 41 projections were baked at the 512 bucket (arithmetic on the code and the tensor list, not a run). So the fork's `src/llama-model.cpp` now selects the output layer's buffer type from the CPU buffer-type list when the output device's own buffer type is host memory, which the QNN device's is (see Rebase notes): under `-dev QNN` the output weight keeps `CPU_REPACK` at every `GGML_QNN_NPAD` and never reaches the QNN placement probe, so it is not charged, its matmul runs at `n_outputs` rows (one per sequence in generation; `llama-bench`'s prompt test asks for the last token's logits only), under `GGML_QNN_MIN_DIM`, and a default run prints no IO-cap notice at model load (verified on the landed build; the protocol runs predate the change and carry it), so the one `ggml-qnn:` notice of a default run is now the placement refusal. What that gives up is the NPU path for all-logits workloads (`llama-perplexity`), which never ran for the output layer: its bake, 741.9 MiB at fp16, has never run on the device (see the budget bullet under How it behaves). Where the output layer does sit in a plain host buffer (an F16 model, a build or run without repack) the cap arithmetic still decides: no single cap value admits the 9728-wide FFN at the 512 bucket (9.5 MiB) and refuses the output layer at the 32 bucket (9.27 MiB), so the default does not try, and at `GGML_QNN_NPAD=32` `GGML_QNN_IO_MAX_KB=1024` keeps it off the NPU path. That is the configuration of every run before 2026-09-27; it refuses the output layer and admits the FFN (608 KiB at 32).

A QNN build offloads without `-dev QNN` too. llama creates the QNN backend in every context (it adds every ACCEL device, whatever `-dev` says), and the scheduler offers it every matmul whose weight sits in a plain host buffer, before the CPU: F16 and F32 weights on any build, and quantized weights wherever repack does not hold them - a build without `GGML_CPU_REPACK`, `--no-repack`, `llama-bench --repack 0`. `-dev QNN` only adds the quantized weights that repack would otherwise take. Until 2026-09-27 the 1 MiB default refused every F16 or quantized weight 1024 or more wide at the default 512 bucket, so at the defaults such a run put nothing of model scale on the NPU. With the default for fp16 graph IO at 10 MiB it does: F16 and quantized weights under 10240 wide are baked, up to the weight budget, and their matmuls run on the NPU at ubatches of `GGML_QNN_MIN_DIM` or more. The placement budget does not enter, since no weight is placed for the QNN device without `-dev QNN`; the bake budget alone bounds it, so a weight past the budget stays a plain host weight on the CPU, without repack. This follows from the code; no run without `-dev QNN` has been measured since the default changed. The way back is `GGML_QNN_DISABLE=1`, which turns the backend off, or `GGML_QNN_IO_MAX_KB=1024`, which restores the refusal of before 2026-09-27.

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
| `GGML_QNN_DISABLE` | unset | disable the backend entirely. It is how a QNN build stays on the CPU: the backend is otherwise created in every context and claims matmuls with or without `-dev QNN` (see Routing K-quant weights to the NPU) |
| `GGML_QNN_MIN_DIM` | 32 | minimum matmul dimension to claim (smaller goes to the CPU) |
| `GGML_QNN_STATIC_BUDGET_MB` | 1024 | cap on static-weight bytes (a quantized weight is charged at its fp16 on-device size), applied twice: at placement, where llama's load-time probe charges each weight it would accept to a ledger and refuses one that does not fit beside those already placed (first fit in load order, charged once per distinct weight, the ledger cleared when the last QNN backend of the process is freed; a refused weight keeps `CPU_REPACK`, and the first refusal of a process is written straight to stderr), and at bake time. 0 = unlimited at both, which places every weight: those the device then cannot map run on the CPU without repack, 1.8x slower than the default on Qwen3-4B (2026-09-27), so to put more on the NPU set a number, not 0. 1024 as a conservative margin: weight mapping failed at about 1170 MiB committed in one run on a machine loaded with other work, and at 1830.0 MiB on a near-idle one with the budget lifted (2026-09-26); a build failure on a shape that already built in the session clamps the budget to the committed bytes for the rest of the session, and the first clamp of a process is written straight to stderr with its phase and error code. Weights that run only on the output rows are baked and charged at reserve as well, see the budget bullet under How it behaves, which also lists the three cases where a placed weight is not baked |
| `GGML_QNN_NPAD` | 512 | batch-dim bucket floor for static and dynamic graphs: one graph serves every N up to it, and a ubatch below it still executes the whole padded bucket, so keep it at or under `-ub`, and keep `-ub` at or under the largest bucket the widest placed weight fits (the defaults, 512 and 512, do both for a weight under 10240 wide). Placement is judged once at this floor and every ubatch at its own bucket: at `-ub 1024` a ubatch of more than 512 tokens pads to 1024, which a weight 5120 or more wide does not fit at the fp16 default, so its matmul runs on the CPU for that ubatch although the weight was placed for the NPU (see the bucket bullet under How it behaves). The padded IO is `NPAD * max(K, M) * e` bytes, where e = 2 for F16 and quantized weights (their graph IO is fp16) and 4 for F32 weights or under `GGML_QNN_NO_F16_IO`. It must stay UNDER the IO cap (equal is refused; see `GGML_QNN_IO_MAX_KB`: unset, 1024 KB at e = 4 and 10240 KB at e = 2; set, that value for every graph), with a matching `-ub`, so a too-large NPAD costs offload rather than a watchdog stall. With the cap unset and e = 2 the default 512 fits any weight under 10240 wide (the 9728-wide Qwen3-4B FFN is 9.5 MiB there, and 512 is the largest power of two that fits a weight 5120 to 10239 wide); a weight under 5120 wide also fits 1024 (a 2560-wide one is 5.0 MiB there, a 4096-wide one 8.0 MiB), a bucket no model run on record has used. The largest power of two that fits is 256 for a 14336-wide FFN and 32 for the 151936-wide output layer, which at 512 is 148.4 MiB and refused, at placement too (under `-dev QNN` llama now keeps that layer in `CPU_REPACK` at every bucket, see "Routing K-quant weights to the NPU"). With the cap unset and e = 4 it is 64 for a 2560-wide weight, 32 for 4096-wide projections such as Qwen3-4B attention, 16 for the 9728-wide Qwen3-4B FFN and 1 for the 151936-wide output layer, the last two below the default `GGML_QNN_MIN_DIM`, so those stay on the CPU; at the default NPAD of 512 every such weight 512 or more wide is refused, at placement too. With `GGML_QNN_IO_MAX_KB=1024` set, e = 2 is capped at 1 MiB too, at 128 for a 2560-wide weight (anything under 4096 wide), 64 for 4096-wide projections, where 128 lands exactly on 1 MiB, 32 for the FFN and 3 for the output layer, and at 512 every weight 1024 or more wide is refused. The first fixable IO-cap refusal at model load and the first at schedule time go straight to stderr, not through the log callback, and name the cap and the `GGML_QNN_NPAD` and `-ub` that fit that shape. Where the cap comes from: graph execute hung when a padded IO buffer crossed a runtime-dependent threshold (~1-1.5 MB measured; exactly 1 MiB hung on QAIRT 2.45), and every one of those measurements ran on the mixed-dtype path that `e3ba3f695` replaced. fp16 graph IO ran past that size: on 2026-09-27, with the cap lifted, K=512 static bakes up to 2.5 MiB of padded output passed at NPAD 64, 256 and 512, and Qwen3-4B at `-ub 512` ran 57 weights on the NPU, the 9728-wide FFN with 9.5 MiB of padded IO among them, with no slow execute (see the 2026-09-27 subsection on fp16 graph IO under Measured comparison). 9.5 MiB is the largest fp16 padded IO that has run, which is why the fp16 default is 10 MiB and no higher. No measurement here covers an F32 weight past 1 MiB, so fp32 graph IO keeps the 1 MiB default |
| `GGML_QNN_IO_MAX_KB` | unset (1024 for fp32 graph IO, 10240 for fp16 graph IO) | policy-reject a matmul whose padded input or output would reach this many KB (strict less-than passes); the shape runs on the CPU instead of hanging the HTP, and as a policy reject it is not denylisted. Set, it caps every matmul graph at that value. Unset, there is one default per IO element size. 1024 for the graphs whose IO is 4 bytes per element, an F32 weight or any weight under `GGML_QNN_NO_F16_IO`: the mixed-dtype path where the hang was measured, and the F32 weight that nothing has measured past 1 MiB. 10240 (10 MiB) for the fp16 graph IO of an F16 or quantized weight: a bound on what was measured, not a hang threshold. The largest fp16 padded IO that has run is 9.5 MiB (9728 x 512 x 2, the FFN of Qwen3-4B at the 512 bucket, 2026-09-27, see `GGML_QNN_NPAD`) and passes; nothing larger has run, and the 151936-wide output layer at the 512 bucket, 148.4 MiB, is refused. At the 32 bucket that layer is 9.27 MiB and passes, see "Routing K-quant weights to the NPU". Before 2026-09-27 one default of 1024 capped every graph, which `GGML_QNN_IO_MAX_KB=1024` restores; a value over 10240 admits fp16 IO of a size that has never run on the device. A value that is not a whole number of 1 or more counts as unset, and because unset is two numbers and not the one a set value is, the backend says so once on stderr instead of in the log: `ggml-qnn: GGML_QNN_IO_MAX_KB="12k" is not a whole number of KB, 1 or more: it counts as unset, so the defaults apply (fp32 graph IO capped at 1024 KB, fp16 graph IO at 10240 KB)`. Whitespace after the number is not part of the value (`set GGML_QNN_IO_MAX_KB=1024 && prog` in cmd.exe leaves a space there), and a value over 4194304 (4 GiB) acts as 4194304, since the 32-bit sizes QNN takes refuse IO of 4 GiB and more before the cap is asked. `supports_op` applies the cap before it claims, so such a matmul is "not supported", never a failed op, and a weight capped at the smallest bucket is not placed on the NPU at load. The padded IO counts 2 bytes per element for F16 and quantized weights and 4 for F32 weights or under `GGML_QNN_NO_F16_IO`, see `GGML_QNN_NPAD` |
| `GGML_QNN_NO_OPT` | unset | drop the finalize-optimization flag (matmul graphs; debugging lever); also bypasses denylist entries loaded from the file |
| `GGML_QNN_NO_STATIC_WEIGHTS` | unset | disable static weight baking: quantized weights are then unclaimable and keep `CPU_REPACK`, and no placement ledger is kept |
| `GGML_QNN_NO_F16_IO` | unset | restore the old mixed-dtype matmul graphs (fp16 weight, fp32 activation and output), which drop to a reference kernel on the HTP: 418x slower on one shape measured on 2026-09-23 (208.9 ms against 0.5 ms), and 13099 ms instead of 0 ms for `test-qnn-health`. Only for reproducing that; it also makes the IO cap count 4 bytes per element for every weight, which gives every weight the 1024 KB default of fp32 graph IO (see `GGML_QNN_IO_MAX_KB`), and drops the `_io16` tag from shape keys. `test-qnn-health-control` sets it |
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
| `GGML_QNN_STATS` | unset | file that receives sixteen `name value` lines when the session is freed or degrades: the eight original counters plus `burst_applied`, `budget_clamped` (0/1), `exec_count`, `exec_slow` (compute-time executes of 1 s or more), `exec_max_ms`, `shm_selftest`, the init-time shared-memory self-test of the last session (0 = the fastrpc library is absent, 1 = present but the self-test failed, 2 = ok, 3 = not requested, `GGML_QNN_SHARED_MEM` unset), `weights_placed`, the distinct weights the placement probe accepted at model load, and `placement_budget_refused`, the distinct weights it refused for `GGML_QNN_STATIC_BUDGET_MB`. The counters are process-wide and only go up. The others add over every session of the process: a session is freed with its last backend instance, and `llama-bench` builds one context per test, each baking its weights at reserve, so a `-p 512 -n 64` run reports each resident weight's bake twice. The two placement counters count each distinct weight (name, type, shape) once per ledger lifetime, the ledger being cleared with the last backend of the process, so over `llama-bench`'s two contexts `weights_baked` reads twice `weights_placed` (82 against 41 at the defaults in the second 2026-09-27 protocol run), and a load after that reset adds to both: `llama-completion`'s default `-fit on` loads the model twice, the dry run's context freed before the real load, so a default `llama-completion` run of Qwen3-4B-Q4_K_M reads `weights_placed` 82, `placement_budget_refused` 422 and `weights_baked` 41, where `-fit off` reads 41, 211 and 41, and `llama-bench`, which does no dry run, 41, 211 and 82; the placement is the same in all three (41 weights baked). `exec_count` counts compute-time executes only, not validation executes |
| `GGML_QNN_FAIL_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: the VALIDATION execute of every graph whose key contains it fails before QNN is called, and compute-time executes are not touched. Under `GGML_QNN_NO_PREVALIDATE` there is no validation execute, and the hook fails the compute-time executes of a matching graph instead, which reaches the execute-error branch a real mid-decode failure takes (`test-qnn-compute-error`). `GGML_QNN_FAIL_EXECUTE_SKIP=<n>` lets the first n matching executes run normally. Never set either outside the test suite |
| `GGML_QNN_DELAY_EXECUTE` | unset | test-only fault hook. The value is a graph-key substring: one execute of a matching graph, validation or compute-time alike, sleeps `GGML_QNN_DELAY_EXECUTE_MS` inside the timed call right before `graphExecute`, so the delay counts toward `GGML_QNN_TIMEOUT_MS` and `GGML_QNN_SLOW_EXEC_MS` and reaches the slow and timeout verdicts on a healthy device. Unlike a real wedge the delayed call then completes. Never set outside the test suite |
| `GGML_QNN_DELAY_EXECUTE_MS` | 0 | test-only: the delay in ms for `GGML_QNN_DELAY_EXECUTE`, capped at a day; 0 leaves the hook off |
| `GGML_QNN_DELAY_EXECUTE_SKIP` | 0 | test-only: the first n matching executes run undelayed and only the next one is delayed (one-shot); later executes of the graph run at normal speed |
| `GGML_QNN_FAIL_FINALIZE` | unset | test-only fault hook. The value is a graph-key substring: the finalize of every matching graph returns `QNN_GRAPH_ERROR_GENERAL` without calling `graphFinalize`, after the graph was created and its node added, so the real finalize-error path runs: a static shape that already built and validated in the session clamps the static budget, any other shape is refused and written to the `GGML_QNN_DENYLIST` file. Never set outside the test suite |
| `GGML_QNN_FAIL_FINALIZE_SKIP` | 0 | test-only: the first n matching graphs finalize normally; every match after them fails |
| `GGML_QNN_FAIL_INIT` | unset | test-only fault hook: with `GGML_QNN_FAIL_INIT=<n>` the first n session inits of the process run and the next one fails before `QnnHtp` is loaded (one-shot), so a re-init that fails after a working session can be reached on a healthy device (`test-qnn-reinit-fail`). Never set outside the test suite |

A value that is not a number, or is below the variable's minimum, falls back to the default
with a WARN in the log that names the default:
`ggml-qnn: <VAR>="<value>" is not a number, using the default <N>` or
`ggml-qnn: <VAR>=<value> is below the minimum <M>, using the default <N>`.
Whitespace after the number is not part of the value. `GGML_QNN_IO_MAX_KB` reports an unusable
value on stderr instead, because its default is two numbers (see its row). On Windows an
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
  -o MUL_MAT` with `GGML_QNN_NPAD=32`, `GGML_QNN_MIN_DIM=1` and `GGML_QNN_IO_MAX_KB` unset, and
  fails if it executed zero cases
- model-scale bakes and their env variants: `test-qnn-modelscale`,
  `test-qnn-modelscale-npad0`, `test-qnn-noopt`, `test-qnn-modelscale-nostatic`,
  `test-qnn-modelscale-noburst`, `test-qnn-modelscale-shm`
- configuration surface: `test-qnn-disable`, `test-qnn-mindim`, `test-qnn-rebake`,
  `test-qnn-elementwise`, `test-qnn-elementwise-on`, `test-qnn-loadprobe`, `test-qnn-envparse`.
  `test-qnn-mindim` pins the two IO cap defaults, the fp16 one by placement probes that bake
  nothing: an F16 weight 10239 wide is placed and one 10240 wide refused, the 9728-wide Q4_K
  FFN is placed and the 151936-wide Q6_K output layer refused
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
OFF), which sets `GGML_QNN_TEST_REQUIRE_HTP=1` on every `test-qnn-*` entry and on
`test-backend-ops-qnn` (the option's help text: `llama: fail the test-qnn-* entries and test-backend-ops-qnn when no HTP is found`), or export that variable for a single run: the skip becomes a failure (exit 1), so a device or context that failed to create cannot turn the whole suite into skips while ctest still exits 0. An existing build directory has to be configured again with the option for the entries to get the variable. Without it `test-backend-ops-qnn` on a box with no usable HTP prints `no usable HTP, skipping` and ctest reports it Passed, not NOT RUN.
`test-qnn-modelscale-shm` also exits 77 when the fastrpc library is absent (`shm_selftest 0` in its stats file); a library that is present with a failed self-test (`shm_selftest 1`) fails the entry.
`test-backend-ops -b QNN -o MUL_MAT` is the figure in the Status table, but ONLY with
`GGML_QNN_NPAD=32` and `GGML_QNN_MIN_DIM=1` set. At the defaults `supports_op` declined every
one of the ~1680 test shapes - measured 2026-09-22, when one 1 MiB cap covered every graph:
zero executed at the defaults, 2 with `GGML_QNN_NPAD=32` alone and 47 with
`GGML_QNN_MIN_DIM=1` as well, so `GGML_QNN_MIN_DIM` declined most of them - and the binary
then prints `Backend QNN: OK` having compared nothing. The defaults have not been measured
again since the cap default changed. The ctest entry `test-backend-ops-qnn` pins those two
settings, unsets `GGML_QNN_IO_MAX_KB` (a value exported by the caller's shell caps every graph
and changes the count) and fails if the executed count is zero, so a green run means
something; a bare manual invocation at the defaults does not.
With the cap unset the pinned run executes 49 cases, not 47, by the arithmetic of the case
list: the two F16-weight `m = 1` cases of 512 and 509 columns (`k` 2048 and 2051) pad to the
512 bucket, 2.0 MiB of padded input, which the 1 MiB cap refused and the 10 MiB default
passes. No 49-case run is on record yet; every count in the Status table is on 47 cases. The
driver does not pin everything the count follows: by the same arithmetic
`GGML_QNN_NO_F16_IO` in the caller's shell gives 47 again, and `GGML_QNN_QUANTIZED`,
`GGML_QNN_DENYLIST` and `GGML_QNN_DISABLE` change it as well.

`test-kleidiai-coff-patch` needs no HTP and no compiler: it runs
`kleidiai-patch-coff-asm.cmake` (see Build) with `cmake -P` over generated `.S` fixtures
shaped like the KleidiAI v1.24.0 tarball and like trees an earlier version of the script
left behind, and checks the result. It is registered in every build where the patch script
exists, with or without `GGML_QNN`, with a 60 s `TIMEOUT`.

`scripts/fork-ci.sh` is this fork's CI: the fork runs no GitHub Actions (the script's header
says why), and the script builds and tests on this machine. `npu` (the default) builds
`build-npu` and runs the ctest entries above except `test-qnn-health`, which needs an idle
machine and is run by hand, then the step `run_qnn_fused_ops` described below; `gpu` builds
`build-gpu` and runs `test-backend-ops -b GPUOpenCL`
over every op, then upstream's unit tests; `cpu` builds `build-cpu` and runs upstream's unit
tests, then `test-backend-ops -b CPU`; `all` runs the three in turn. Upstream's unit tests are
`ctest -L main`, plus `-L python` and `-L model` when their inputs are present. Every mode
first runs `editorconfig-checker` over the files the fork changed since its upstream merge base,
when the checker is installed. A failed test step does not stop the others, and the script
exits 1 at the end if any failed; `BUILD_ONLY=1` skips the tests.

`run_qnn_fused_ops` checks that `-dev QNN` leaves flash attention on, which is the fork's
change to `src/llama-context.cpp` (see Rebase notes). It is a step of the script, not a
ctest entry. It runs
`build-npu/bin/llama-completion -m $TEST_MODEL -dev QNN -no-cnv -n 1 -p Hello -lv 4` with
`GGML_QNN_STATIC_BUDGET_MB=1`, under a 300 s timeout, and keeps the output in
`build-npu/qnn-fused-ops.log`. It fails when the binary exits non-zero, when the log has
`Flash Attention not supported, set to disabled`, or when it has no `Flash Attention enabled`
line; the last check makes a reworded upstream message fail the step instead of passing it.
Three of its settings matter:

- `GGML_QNN_STATIC_BUDGET_MB=1` bakes nothing. llama resolves the fused ops at context creation from the device each layer is assigned to, not from what runs there, so nothing has to run on the NPU and one token is enough.
- `-lv 4`, because llama logs the `enabled` line at INFO, which the default verbosity of 3 drops (`-v` shows it too).
- no `-fa`. Only `auto` is resolved, so an explicit `-fa on` would pass with the fix removed.

`TEST_MODEL` defaults to the Qwen3-4B-Q4_K_M file of the measurements here. When the file
does not exist the step is skipped and listed under `skipped` at the end of the run. The
step opens the HTP, so nothing else that uses the NPU may run beside it.

The measurements of 2026-09-27 are kept outside the repository, in three folders of the
bench records. `2026-09-27/npu-speed` holds the first protocol run: the driver
`bench-picks.ps1`, a `README.txt` with the figures and the reading, and one folder per run
with the driver log, the results and, per leg, the raw `llama-bench` output, stderr,
`GGML_QNN_STATS` file and 2 s samples. `2026-09-27/npu-placement` holds the second protocol
run, on the build with the placement budget, laid out the same way.
`2026-09-27/npu-experiments` holds the fp16 IO and KL divergence runs: the
scripts `npu-experiments.sh` and `kld-bisect.sh`, a `README.txt` that lists every run, the
raw logs, the `GGML_QNN_STATS` files and the saved reference logits.

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
- The same holds for the NPU, and `llama-bench` hides the evidence: without `-v` it installs a null log callback, so every backend WARN and ERROR is dropped and a degraded NPU leg prints CPU numbers under the `QNN` backend name. The first degrade of a process is therefore also written straight to stderr: `ggml-qnn: NPU degraded (<reason>), claiming no ops for the rest of this process: everything runs on the CPU from here`. So are a session that cannot be re-created (`ggml-qnn: the NPU session could not be re-created, ...`), the first static-budget clamp (`ggml-qnn: <phase> of <key> failed (<code>) on a shape that built before: NPU weight memory is full at <X> MiB committed, ...`) and the one budget-full notice, which after a clamp reads `ggml-qnn: NPU weight memory clamped at <X> MiB after a <phase> failure (code <N>), ...`, and the first placement refusal of a process (`ggml-qnn: the <N> MiB static-weight budget has no room for <weight> (<X> MiB) beside the <Y> MiB placed at model load in <n> weights: ...`, see the budget bullet under How it behaves), which every `-dev QNN` leg with a finite budget prints at model load once the budget is full.
  A leg that never claimed anything degrades nothing and prints nothing: weights left in `CPU_REPACK`, or the IO cap at a too-large `GGML_QNN_NPAD`. The first fixable IO-cap refusal at model load and the first at schedule time go straight to stderr and name the `GGML_QNN_NPAD` and `-ub` that fit, so `llama-bench` shows them without `-v` as well (verified 2026-09-17, when one 1 MiB default capped every graph: `-dev QNN` at the default `GGML_QNN_NPAD=512` printed it and left the whole model in `CPU_REPACK`; with the fp16 default at 10 MiB a K-quant model is now placed on the NPU there, and the notice was then about the output layer alone, as in every `-dev QNN` leg of the two 2026-09-27 protocol runs, so it no longer means the leg ran on the CPU; since llama keeps the output layer in `CPU_REPACK` under `-dev QNN` a default run of this model prints no IO-cap notice at all, verified on the landed build, and its one notice is the placement refusal).
  The proof that an NPU leg ran is `GGML_QNN_STATS`: `graphs_created` and `exec_count` must be non-zero, and `exec_max_ms` in the millisecond range. The counters add up over every session in the process, and `llama-bench` builds one context per test, each baking its weights at reserve, so a `-p`/`-n` run counts each resident weight's bake once per test: the 2026-09-26 A legs report `weights_baked` 82 for 41 weights on the NPU. `weights_placed` and `placement_budget_refused` count each distinct weight once per ledger lifetime instead, so over the two contexts `weights_baked` reads twice `weights_placed` (82 against 41 in the second 2026-09-27 run) and their sum is the count of weights the probe would otherwise have accepted (252 for Qwen3-4B's projections); a `-fit` dry run is a ledger lifetime of its own and is counted too (see the `GGML_QNN_STATS` row). `exec_count` counts compute-time executes only, so dividing it by the prefill ubatches (warmup plus repetitions) gives the weights executed on the NPU per prefill ubatch.
  The controls changed with the placement budget. `-dev QNN` with `GGML_QNN_STATIC_BUDGET_MB=1` is now the CPU alone under `-dev QNN` (every weight refused at placement and kept in `CPU_REPACK`; on the build before the change it was the NPU legs' placement with nothing baked), so it no longer measures what a placement costs; the control for what the repack layout is worth is `--repack 0` with `GGML_QNN_DISABLE=1`, and the exact control for the default placement with nothing on the NPU (41 weights on the CPU without repack, 211 with it) has no supported setting; a test-only fault hook approximates it and is not a recipe. One model per process is not required: the ledger is cleared with the last backend of the process, so a model loaded after that starts fresh (`llama-bench`'s second context places nothing again: placement happens at model load, once); two models alive at the same time share the budget.
- Placement is not the only thing to prove: check flash attention too. `llama-bench` records the flash attention that was asked for (`flash_attn` -1 is `auto`), not what llama resolved it to, and without `-v` it drops the `resolve_fused_ops` line that says which. Until 2026-09-27 `auto` resolved to off under `-dev QNN` and to on in a CPU leg, so the 2026-09-26 legs differ in it (see point 6 of that subsection). Run the legs with an explicit `-fa on`, so the result lines record `flash_attn 1` (a driver should; only the CI check under Testing has to leave `auto` to be resolved), or keep the `-v` log. Neither 2026-09-27 protocol run did: their legs asked for `auto`, and that they ran with flash attention on is read from the code, not from their logs.
- The 2026-09-26 driver runs (the dated subsection under "Measured comparison") are the first
  here with that proof on record, and on the build of that day they were repeatable from this
  description. One binary and one model for every leg;
  `llama-bench -t 6 -p 512 -n 64 -r 5 -o jsonl`, only the result lines
  parsed (the QnnHtp runtime prints its graph-prepare stages to stdout between them);
  `QNN_SDK_ROOT`, `ADSP_LIBRARY_PATH` and the QAIRT `lib\aarch64-windows-msvc` directory on
  `PATH` set once for all legs; per leg only `GGML_QNN_DISABLE`, `GGML_QNN_NPAD`,
  `GGML_QNN_STATIC_BUDGET_MB` and `GGML_QNN_STATS`, cleared between legs. That is what the
  driver set on 2026-09-26. On the tree the 2026-09-27 runs used (`d80e59573`) the same four
  variables gave a different configuration in the NPU legs: `-fa auto` resolves to on under
  `-dev QNN` there, and with `GGML_QNN_IO_MAX_KB` unset the output layer passed the placement
  probe at `GGML_QNN_NPAD=32` (9.27 MiB, under the 10 MiB fp16 default) and left
  `CPU_REPACK`. To repeat the measured configuration there, run legs A and D with
  `GGML_QNN_IO_MAX_KB=1024` and `-fa off`, and legs B and C with `-fa on`, and add
  `GGML_QNN_IO_MAX_KB` to the variables cleared between legs. Without those the legs are a
  new configuration and do not compare with the 44-53 t/s table. On the current tree the
  placement budget changes leg A again (41 weights placed and 211 kept in `CPU_REPACK`, where
  that day all 252 were placed), so leg A's configuration cannot be repeated on it; leg D's
  can, since a budget of 0 refuses nothing at placement. This follows from the code; no such
  re-run is on record. The
  order was counterbalanced, A B C D D C B A with a 120 s cooldown between legs, so both NPU
  configurations are paired. A gate before every leg (and at batch start and end) requires
  AC power, charge at or above 40%, no other llama, genie or qnn process, and a 10 s
  `\Processor(_Total)\% Processor Time` window averaging under 15%, retried up to six times,
  and logs the charge and idle CPU it saw. Through
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
- The two 2026-09-27 protocol runs (their dated subsection under "Measured comparison") used
  a second driver, `bench-picks.ps1 -Batch npu`, which records more (the second run passed
  `-Reps 5 -Cooldown 120 -LeadCooldown 120 -LegTimeout 480 -GateWindows 30`). Its gate
  requires AC power, a settled pack (a charge draw of 5 W or less), no other llama, genie or
  qnn process and the same 10 s CPU window under 15%, and its 2 s sampler records the clock
  (`\Processor Information(_Total)\% Processor Performance`), the AC state and the charge
  draw beside the CPU load, so a leg that loses AC shows it in its samples and the drift
  within a leg can be set against the clock. Every leg is
  `llama-bench -t 6 -ub 512 -p 512 -n 64 -r 5 -o jsonl` from `build-npu`; a leg adds only
  `-dev QNN` or `--repack 0` and sets only `GGML_QNN_DISABLE` or `GGML_QNN_STATIC_BUDGET_MB`,
  with `GGML_QNN_STATS` pointed at its own file, and the driver clears `GGML_QNN_NPAD`,
  `GGML_QNN_STATS`, `GGML_QNN_DISABLE`, `GGML_QNN_STATIC_BUDGET_MB`, `GGML_QNN_IO_MAX_KB`,
  `GGML_QNN_MIN_DIM` and `GGML_QNN_NO_F16_IO` before every leg. Two things it left open in
  both runs: it passed no `-fa` and kept no `-v` log, so the flash attention of its legs is
  read from the code, and it logged the build times of `build-gpu` and `build-cpu` but not of
  `build-npu`. Its legs are the same commands on either build, but the PN leg
  (`GGML_QNN_STATIC_BUDGET_MB=1`) measures a different configuration on the tree with the
  placement budget: the CPU alone under `-dev QNN`, not the NPU legs' placement with nothing
  baked. The driver and the raw output of every leg are in the bench records'
  `2026-09-27/npu-speed` (first run) and `2026-09-27/npu-placement` (second run) folders (see
  Testing).

## Known limitations

- Windows ARM64 only in practice (the dlopen paths exist for Linux but are untested).
- fp16 math internally: F32 elementwise cannot meet strict 1e-7 tolerances (matmul
  tolerances pass).
- 512-class static bakes (a 512-wide weight at the default `GGML_QNN_NPAD=512`, 1 MiB of
  padded IO at fp32) were seen hanging at the validation execute on QAIRT 2.45, first on
  battery and then on AC. That, and the rest of the IO-size law under `GGML_QNN_NPAD`, was
  measured only on the mixed-dtype path that `e3ba3f695` replaced, whose reference kernel was
  slow in its own right. fp16 graph IO ran past that size: the same bakes, and larger ones up
  to 2.5 MiB of padded output, passed on 2026-09-27 with the cap lifted, and so did Qwen3-4B
  at `-ub 512`, up to 9.5 MiB. Nothing larger has run, so the default cap for fp16 graph IO
  is 10 MiB, and whether fp16 graph IO hangs at some larger size is not known. fp32 graph IO
  (an F32 weight, or `GGML_QNN_NO_F16_IO`) keeps the 1 MiB default, where no measurement has
  shown it can go further; for those, pick `GGML_QNN_NPAD`, with a matching `-ub`, by weight
  width so that `NPAD * max(K, M) * 4` stays under 1 MiB: 32 for 4096-wide projections, 16
  (below `GGML_QNN_MIN_DIM`, so the CPU) for a 9728-wide FFN; see the `GGML_QNN_NPAD` row.
- The output layer's bake has never run on the device. Under `-dev QNN` llama now keeps the
  output layer in `CPU_REPACK`, so it is never offered to the QNN device or charged at
  placement (see "Routing K-quant weights to the NPU"). Where it does sit in a plain host
  buffer (an F16 output layer, or a build or run without repack) the default cap admits a
  151936-wide layer at `GGML_QNN_NPAD=32` (9.27 MiB of padded output), and a budget that
  reaches it at reserve bakes 741.9 MiB, against 47.5 MiB for the largest bake on record;
  `GGML_QNN_IO_MAX_KB=1024` refuses it there.
- With weights on the NPU the output is not the CPU's, and what that is worth is not
  measured. On Qwen3-4B-Q4_K_M (2026-09-27, wikitext-2, 8 chunks) the KL divergence from a CPU
  reference has a median of 0.0016-0.0018 and the top token differs at about 3% of positions,
  which is about what the CPU's own two attention paths, flash attention on and off, differ
  by (0.0016, 3.1%). The mean (0.015-0.047) and the maximum (20.3-34.9) are set by a few
  positions that flip completely and do not rank configurations. The placements with 43 or
  more weights on the NPU have three or more such positions of 2040 where the others have at
  most two, and that is not explained. The reading this list first gave, that the divergence
  is the NPU path's, compared against a control that also differed in flash attention, which
  llama disabled under `-dev QNN`; that is fixed, and the reading is withdrawn.
  Not measured: a text without such positions, another model, any task-level quality. See the
  2026-09-27 subsection on the output under Measured comparison.
- `-dev QNN` at the default budget is faster than the CPU alone on the one model measured,
  and by little: +7% on Qwen3-4B-Q4_K_M prefill against the CPU alone in the same run and
  +12% against the CPU legs of the run before it (2026-09-27, `-ub 512`, flash attention at
  `auto`, which resolves to on in this tree by the code; one model, one box, five repetitions
  per leg, one clean CPU leg in that run plus the two CPU pairs of the run before). 41 of the
  252 projections run on the NPU and the other 211 on the CPU with repack; how much faster
  the NPU runs its own share is not measured, and neither is more weights than the default on
  this build. The gain exists because the weight budget is now applied at placement: on the
  build before that, which placed every probe-accepted weight for the NPU whatever the budget,
  the same protocol read a loss of 1.67-1.76x, since a weight placed for the NPU loses the
  CPU's repack layout, which halved the CPU's prefill, and 41-57 weights on the NPU made the
  run only 14-20% faster than that same placement on the CPU. `GGML_QNN_STATIC_BUDGET_MB=0`
  still places every weight, and reads 1.8x slower than the default (57 weights on the NPU,
  195 on the CPU without repack); set a number instead. `-dev QNN` with nothing on the NPU
  now reads what the CPU alone reads. Decode is CPU work in every configuration, so no decode
  figure is an NPU result, and the accuracy caveat above (a few positions flip with 43 or more
  weights on the NPU) is unchanged and was not re-measured on this build. See the protocol
  runs under Measured comparison.
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
- The fork's change to upstream's `src/llama-context.cpp`: `resolve_fused_ops` does not count
  a device mismatch when the fused op runs on a CPU device and the layer's device keeps its
  tensors in host memory (`ggml_backend_buft_is_host` of its buffer type; the QNN device's
  buffer type is the CPU's). So flash attention, and the other fused ops that function
  resolves (Gated Delta Net, Lightning Indexer, DeepSeek V4 HC), stay enabled when they run on
  the CPU under `-dev QNN`. Upstream's check disabled flash attention for the whole model as
  soon as a layer was assigned to QNN, which is what the 2026-09-26 NPU legs and the first
  2026-09-27 KL divergence runs ran with (see Measured comparison). A device that keeps its
  tensors in its own memory is judged as upstream judges it. `scripts/fork-ci.sh` checks the
  change in its `npu` mode (`run_qnn_fused_ops`, see Testing).
- The fork's change to upstream's `src/llama-model.cpp`: right after `dev_output` is
  assigned, its buffer-type list is swapped for the CPU one when the output device's own
  buffer type is host memory (`ggml_backend_buft_is_host` of `ggml_backend_dev_buffer_type`
  of that device; the QNN device's is). The device itself stays the QNN device, so the
  sampler and output buffers are unchanged; `output_norm` and the classifier heads follow the
  same list, and a tied `token_embd`/`output` (`TENSOR_DUPLICATED`) is covered. So under
  `-dev QNN` the output weight keeps `CPU_REPACK` and never reaches the QNN placement probe.
  Its matmul runs at `n_outputs` rows, one per
  sequence in generation, under `GGML_QNN_MIN_DIM`, so what this gives up is the NPU path for
  all-logits workloads (`llama-perplexity`), which never ran for it; what it prevents is the
  placement ledger charging 741.9 MiB to that layer at `GGML_QNN_NPAD=32` (see "Routing
  K-quant weights to the NPU"). Landed with the placement budget, after the second 2026-09-27
  protocol run.
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
