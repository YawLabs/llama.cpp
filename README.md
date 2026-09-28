# llama.cpp

[![Follow @YawLabs on X](https://img.shields.io/badge/follow-%40YawLabs-000000?logo=x&logoColor=white)](https://x.com/YawLabs)

> [!NOTE]
> **This is the YawLabs fork of [llama.cpp](https://github.com/ggml-org/llama.cpp).** It adds an
> experimental **QNN backend** for the Qualcomm Hexagon NPU on Windows on Snapdragon
> (`-DGGML_QNN=ON`) - see [docs/backend/QNN.md](docs/backend/QNN.md) for status, build
> instructions and measured results. The backend is a research vehicle, not a production
> accelerator; everything else in this repository is upstream llama.cpp, apart from the fixes
> and CI changes the docs' Rebase notes list. Not affiliated with or endorsed by the upstream
> project.

## What this fork adds

An experimental ggml backend (`ggml/src/ggml-qnn/`) that runs matmuls on the Hexagon NPU
through QNN (Qualcomm AI Engine Direct) on Windows ARM64 - no test-signing, no custom DSP
kernels, only the QAIRT community SDK headers at build time and the QAIRT runtime DLLs at
run time (plus `ADSP_LIBRARY_PATH` on QAIRT 2.45 and newer; see the docs' Requirements).
Developed and measured on a Snapdragon X Elite (X1E80100, HTP v73).

On that machine:

- **No single-matmul kernel figure is claimed.** The kernel numbers this list used to
  lead with were taken in August 2026 without recording what else was running on the
  machine - the same flaw that later invalidated a round of model-run timings (see the
  retraction in the docs) - so they are withdrawn rather than repeated. The two mechanisms
  they were meant to show are real and still in the code: a DCVS TURBO power config, and
  baking a weight once into the HTP-native layout instead of re-tiling it on every
  execute. What either is worth on an idle machine is an open question. The one throughput
  figure that is claimed is the end-to-end model run at the foot of this list: with the
  weight budget applied at placement, `-dev QNN` at the default budget is 7% faster on
  prefill than the CPU alone in the same run and 12% faster than the CPU legs of the run
  before it (2026-09-27, `-dev QNN -ub 512`, flash attention at `auto`, which resolves to on
  in this tree, read from the code). On the build before that change, which placed every
  probe-accepted weight for the NPU whatever the budget, the same protocol read a loss of
  1.67-1.76x; both runs supersede the 2026-09-26 one (`-dev QNN -ub 32`, flash attention
  disabled in the NPU legs and on in the CPU legs)
- **47/47** `test-backend-ops` MUL_MAT correctness (F32/F16) against the CPU, run by ctest as
  `test-backend-ops-qnn`: 40 of 40 clean on an idle machine. A single 46/47 was seen once, in
  a run taken while other heavy work was loading the machine, and has not reproduced since on
  either the current fp16 path or the old one - so it is recorded, not claimed as a known
  flake. All of those counts were taken on 47 cases, under the one 1 MiB IO cap that covered
  every graph until 2026-09-27; with the default for fp16 graph IO now at 10 MiB the entry
  executes 49 by the arithmetic of the case list (two F16-weight cases with 2.0 MiB of padded
  input join the 47), and a 49-case run is not on record yet. With a clean process exit -
  observed clean on the recorded runs (2026-08-28 ctest log, 2026-09-16), including with a
  live or degraded HTP session, though a teardown crash inside `QnnHtp.dll` stays listed as
  a known limitation in the docs - plus a dedicated lifecycle suite
  (`tests/test-qnn-lifecycle.cpp`, 36 registered ctest entries) covering static-weight bakes
  vs the CPU reference, the memory budget, the failed-shape denylist, watchdog degradation,
  the shared-memory IO path, and injected execute failures (at validation and at compute
  time), finalize errors, execute delays and session re-init failures
- **No hangs on real models**: the load-time probe answers from type, shape and buffer
  policy, and a resident weight is trial-built, finalized and test-executed at schedule
  time before the backend claims it; shapes the HTP rejects or that wedge it land in an
  in-process denylist (finalize errors and watchdog timeouts are also persisted across runs when `GGML_QNN_DENYLIST` points at a file; the docs give the exact rule) and
  run on the CPU instead, and a matmul whose padded IO would reach the IO cap (by default
  1024 KB for fp32 graph IO and 10240 KB for fp16 graph IO, `GGML_QNN_IO_MAX_KB` for every
  graph when it is set) is refused up front. A wedge costs one watchdog timeout (15 s for any execute, the validation execute included, and 120 s for a graph finalize)
  and a CPU fallback, not a hung process; configurations that previously
  wedged the machine now complete, and a session that cannot be re-created mid-process
  (`llama-bench` and `--fit` re-create it per context) falls back to the CPU instead of failing
  context creation
- Quantized weights (Q4/Q5/...) enter the NPU path by being dequantized to fp16 once at bake
  time, within a memory budget (default 1024 MB) that is applied at placement as well: a
  weight that does not fit beside those already placed keeps `CPU_REPACK`, and 0 lifts both
  limits and places every weight, which on Qwen3-4B ran 1.8x slower than the default, so to
  put more on the NPU set a number; one padded graph per weight serves every
  N up to `GGML_QNN_NPAD`, larger ubatches get one graph and bake per power-of-two bucket,
  each charged to the budget, and buckets over the IO cap are refused, per ubatch: a weight
  is placed by the smallest bucket, so at `-ub 1024` a weight 5120 or more wide (the
  9728-wide FFN of Qwen3-4B) is placed for the NPU and runs on the CPU for every ubatch of
  more than 512 tokens (arithmetic on the code, not a measurement). On a KleidiAI+REPACK
  build, K-quant weights reach the NPU with `-dev QNN`, the route every measurement here used
  (see the docs): since 2026-09-27 at the default `GGML_QNN_NPAD=512` and `-ub 512` for
  weights under 10240 wide, before that only with a `GGML_QNN_NPAD` small enough for the
  weight width; turning repack off (`--no-repack` in `llama-cli` and `llama-server`,
  `--repack 0` in `llama-bench` since the 2026-09-27 upstream merge) also keeps them out of
  `CPU_REPACK`, but has not been run with the QNN backend
- **A QNN build offloads without `-dev QNN` too.** llama creates the backend in every
  context, and the scheduler offers it every matmul whose weight sits in a plain host buffer:
  F16 and F32 weights on any build, quantized ones wherever repack does not hold them. Until
  2026-09-27 the 1 MiB default cap kept model-scale weights off the NPU at the default
  bucket. The 10 MiB default for fp16 graph IO does not, so at the defaults such a run now
  bakes weights under 10240 wide and runs their matmuls on the NPU (this follows from the
  code and has not been measured). `GGML_QNN_DISABLE=1` turns the backend off, and
  `GGML_QNN_IO_MAX_KB=1024` restores the earlier refusal
- **An IO-size cap that goes as far as the measurements and no further**: graph execute was
  seen hanging when a padded IO buffer crossed a runtime-dependent size threshold (~1-1.5 MB
  measured), only on the old mixed-dtype path. Today's fp16 graph IO ran past that size: on
  2026-09-27, with the cap lifted, static bakes up to 2.5 MiB of padded output passed at
  `GGML_QNN_NPAD` 64-512, and Qwen3-4B-Q4_K_M ran at `-dev QNN -ub 512` with 57 weights on
  the NPU, the 9728-wide FFN's 9.5 MiB of padded IO among them, no slow execute and a slowest
  execute of 18 ms (pp512 58.2 t/s over two repetitions with flash attention disabled, not a
  controlled comparison). Nothing larger than 9.5 MiB has run. So with `GGML_QNN_IO_MAX_KB`
  unset the cap is 1024 KB for graphs with fp32 IO (an F32 weight, or any weight under
  `GGML_QNN_NO_F16_IO`) and 10240 KB for fp16 graph IO, which is a bound on what was
  measured, not a hang threshold: it passes the FFN and refuses the 151936-wide output layer
  at the 512 bucket (148.4 MiB). Set, the variable caps every graph at its value. The padded
  IO is `GGML_QNN_NPAD` x the wider weight dimension x 4 bytes for fp32 IO and 2 for F16 and
  quantized weights, so under a 1 MiB cap a 4096-wide F32 projection fits `GGML_QNN_NPAD=32`.
  See the docs for the tuning guidance
- Composes with the other backends in one binary: KleidiAI CPU + Adreno GPU (OpenCL) + NPU

What it does not do (yet): beat a full-GPU setup end to end (the Adreno's 228.1 t/s stands
unchallenged), and against a KleidiAI-CPU setup the gain is 7-12% on one model - measured,
not expected. The current verdict is the second 2026-09-27 protocol run (on AC with a
settled pack and a quiet machine, counterbalanced pairs of five repetitions, the
`GGML_QNN_STATS` counters proving placement; Qwen3-4B-Q4_K_M at `-ub 512`, flash attention
at `auto`, which resolves to on in this tree, read from the code), on the build where the
weight budget is applied at placement. It put prefill at 135.2 and 139.2 t/s with `-dev QNN`
at the default weight budget (41 weights on the NPU in each of llama-bench's two contexts,
the other 211 in `CPU_REPACK`) against 128.4 for the same binary's CPU path alone in the same
run, +7% (that run's other CPU leg was confounded by something else using the GPU and the
CPU, and is excluded), and against 123.0 for the CPU legs of the first run that day, +12%,
above the CPU alone rep for rep. `-dev QNN` with nothing on the NPU
(`GGML_QNN_STATIC_BUDGET_MB=1`) reads what the CPU alone reads (127.9 and 129.6), and
`GGML_QNN_STATIC_BUDGET_MB=0` reads 73.4 and 75.1, 1.8x slower than the default, because 0
lifts the placement limit too: every weight leaves `CPU_REPACK`, the device takes 57 before
it clamps and the other 195 run on the CPU without repack; to put more on the NPU, set a
number. How much faster the NPU runs its own share is not measured, and neither is more
weights than the default on this build. It is a verdict on the configuration and on this 4B
model, not on the HTP alone. The first run that day, on the build that placed every
probe-accepted weight for the NPU whatever the budget, read the same protocol as a loss:
70.1 t/s at the default budget and 73.6 with the budget lifted against 123.0, 1.76x and
1.67x slower, because placing a weight for the NPU cost it the `CPU_REPACK` (ggml's
interleaved K-quant) layout whether or not the NPU then took it, which halved the CPU's
prefill (`-dev QNN` with nothing on the NPU read 61.5, rep for rep what the CPU alone reads
with `--repack 0`, 62.1). With 41 weights on the NPU the run was then 14% faster than that
same placement run on the CPU, with 57 it was 20% faster, and with 41-57 of the 252
projections on the NPU that did not make up for the ~200 the CPU ran without repack; the
docs keep that run's table as the record of that build. The first NPU model run with proven
placement (2026-09-26: 44-53 t/s against 101-114, about 2.2x, and about 1.6x against the CPU
at the NPU's `-ub 32`) is superseded, because its NPU legs ran with flash attention disabled
(llama disabled it under `-dev QNN`, which is fixed) and at the `-ub 32` that the 1 MiB cap
of the time forced, while its CPU legs ran with flash attention on; the docs keep its table
and give every run's caveats. The earlier 4B sweep's "NPU" leg never executed a matmul on
the HTP, because the K-quant weights were repacked out of the NPU's reach and the one trial
build stalled at the watchdog on the old mixed-dtype path (the docs carry the retraction,
which stays), and 9-14B models are unmeasured. Decode is not claimed for the NPU below
32-token ubatches, so it runs on the CPU in every configuration - the counters of every run
show no decode execute on the HTP - and on the settled-pack run the two measured engines
(GPU, CPU) and the CPU-run leg labelled NPU converge on decode; CPU decode on this box is
bimodal, so no decode figure is quoted. In both 2026-09-27 protocol runs the `-dev QNN` legs
started decode at 30-34 t/s and ended at 17-27 where the CPU-alone legs read 15-17
throughout; that is not an NPU effect (the leg with nothing on the NPU reads the same), and
its likely cause, the CPU clock recovering while the NPU session starts, is not proven.
Its output against the CPU's was measured on 2026-09-27 (`llama-perplexity`, wikitext-2, 8
chunks of 512, the same model, KL divergence from a CPU reference). The first reading
attributed the divergence to the NPU path against a control that also differed in flash
attention, which llama disabled under `-dev QNN`: with nothing on the NPU that
configuration read 0.030 on average with a maximum of 34.3 and the same top token at 96.9%,
to every digit what the CPU alone reads with `-fa off`. That was a bug and is fixed, and the
same run now reproduces the reference (0.000000, the same top token everywhere), so it
explains the divergence when nothing is on the NPU. With weights on the NPU the output still
differs from the CPU's (0.015-0.047 on average, a maximum of 20.3-34.9, the same top token
at 96.5-97.3% over five placements; the budget-0 placements read 0.045-0.046 and 33.5-34.0
with flash attention on, where the first reading had 0.047 and 33.7-34.5 with it disabled).
The typical position differs by about what the CPU's own two attention paths differ by (a
median of 0.0016-0.0018 against 0.0016).
The mean and the maximum are set by a few positions that flip completely and do not rank
configurations, but the placements with 43 or more weights on the NPU have three or more
such positions of 2040 where the others have at most two, and that is not explained. So
the runs do not show that the NPU path costs accuracy, and do not show that it equals the
CPU; the docs have the table. The placement change does not touch what the NPU computes,
and the output was not re-measured after it.
The case for fixing prefill with ahead-of-time compiled context binaries (a serialized
context is reloaded instead of finalized again; the timings that sized that win have not
been re-taken) is in [docs/backend/QNN.md](docs/backend/QNN.md).

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [QNN](docs/backend/QNN.md) | Hexagon NPU (Windows on ARM) |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
