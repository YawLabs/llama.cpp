# typed-local on a QNN build: the typed CLI's free tier against this fork's llama-server

The typed CLI ships a free tier, `typed-local`, whose inference runs against a
llama-server you start yourself. Its own guide (typed/docs/LOCAL_MODEL.md in the typed
repo) documents that end to end, but only for upstream ggml-org llama.cpp builds: winget,
the releases page, or `typed local install`. This page is the missing bridge: how a
Snapdragon X machine runs the same free tier against a llama-server built from THIS fork
with the experimental QNN backend (`-DGGML_QNN=ON`), so part of the matmuls execute on
the Hexagon NPU.

Every build, flag and environment claim about the NPU side comes from
[docs/backend/QNN.md](backend/QNN.md); everything about the typed side comes from typed's
`docs/LOCAL_MODEL.md`. Where the two disagree, this page says so rather than picking a
winner (see "Where the two documents disagree" at the end).

## 1. Build (see QNN.md, not here)

The full requirements and the cmake line live in QNN.md's Requirements and Build
sections. In short:

- Windows 11 ARM64 on a Snapdragon with an HTP (tested: X Elite / HTP v73).
- The Qualcomm AI Runtime (QAIRT) community SDK: headers at build time, `QnnHtp.dll` and
  its dependencies on `PATH` or under `QNN_SDK_ROOT` at run time, plus
  `ADSP_LIBRARY_PATH` pointing at the SDK's `lib/hexagon-v73/unsigned` on QAIRT 2.45 and
  newer (QAIRT 2.34 did not need it; without it the first graph execute dies silently and
  only a WARN says the variable is unset).
- Enable long paths in the clone first (`git config core.longpaths true`) or the checkout
  will be missing files under `tools/ui` and `tools/server/webui`.

On this box the fork's own scripts build it: `scripts/fork-ci.sh npu` configures
`build-npu` with `-DGGML_QNN=ON` (add `BUILD_ONLY=1` to skip the tests) and discovers a
QAIRT SDK under `~/yaw/sdk/qairt/`, or takes `QNN_SDK_ROOT` from the environment; it
fails with instructions when no SDK is present rather than letting cmake report a missing
header. The binaries land in `build-npu/bin`, including `llama-server.exe`.

Two build facts that matter for typed's command:

- **The fork's build scripts set `-DLLAMA_OPENSSL=OFF` ("no HTTPS (no -hf downloads")**,
  so a build made by `fork-ci.sh` cannot fetch the model with typed's `-hf
  unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_S` argument. Download the GGUF separately and pass
  `-m <path>` instead. A plain cmake build per QNN.md's Build section follows the tree's
  own defaults; this page does not claim what those are for HTTPS.
- The QAIRT SDK is absent as of 2026-10-15 on this box (see fork-ci.sh's message); you
  re-acquire it from the Qualcomm Software Center and point `QNN_SDK_ROOT` at it.

## 2. Start the server

This fork has no dedicated server run script -- `scripts/fork-build.sh` and
`scripts/fork-ci.sh` only build. So: run `typed local` in the typed CLI to get the
launch command sized for your machine (window, thread counts, slot count), then make
exactly these substitutions.

**a) `llama-server` -> the full path to this build's binary** (e.g.
`build-npu/bin/llama-server.exe`). typed only prints the command; it never launches the
server, so nothing else in typed needs to change. Note `typed local install` fetches an
upstream CPU build into `~/.local/share/typed/llama.cpp` -- that path is the wrong binary
for this page; keep the fork build.

**b) `--device none` -> `-dev QNN`.** This is the load-bearing change, and the honest
explanation is:

- typed's command passes `--device none` because that is the configuration typed
  measured -- a CPU run, deliberately kept off the (Vulkan) winget package. Its GPU
  offload section says the same: drop `--device none` and add `-ngl` when you want the
  GPU.
- On a QNN build, the flag that routes work to the NPU is `-dev QNN`, and QNN.md calls
  it "the route every measurement here used": the model loader then probes the QNN
  device first, and a weight the probe accepts and the static-weight budget has room
  for lands in a plain host buffer where the backend can bake it to fp16 in HTP-native
  layout.
- It is not a hard requirement to get ANY NPU work: since 2026-09-27 the default fp16
  IO cap lets a QNN build offload model-scale weights even without `-dev QNN`, wherever
  a weight sits in a plain host buffer (`GGML_QNN_DISABLE=1` or
  `GGML_QNN_IO_MAX_KB=1024` is the way back to CPU-only). typed's `--no-repack` keeps
  weights in plain host buffers too. But `-dev QNN` is the configuration every QNN.md
  measurement used, so this is the flag to change.
- If the backend is unavailable (wrong arch, missing `QnnHtp.dll`, no
  `ADSP_LIBRARY_PATH`), `llama-server` at default verbosity rejects `-dev QNN` with only
  `invalid device: QNN`; put `-lv 4` before `-dev` to see why (QNN.md's Environment
  variables section). If you just want a CPU run, keep typed's `--device none` instead.

**c) Keep `--port 9931`.** typed dials 9931 (its default endpoint), and this fork's
llama-server still defaults to 8080 -- it prints a notice that the default will change to
9931 in a future release, but today it is 8080.

**d) Keep everything else as typed printed it** -- `--jinja` (required for tool calls),
`-np 1` (or the multi-slot flags `typed local --slots N` printed), `--ctx-size`,
`--cache-ram 0`, `--no-repack`, `--flash-attn on`, the q8_0 KV cache types,
`--spec-type ngram-cache`, `--min-p 0.0 --presence-penalty <mode value>`,
`--batch-size 2048 --ubatch-size 1024`. All of these are accepted by this build
(checked against `llama-server --help` from `build-npu/bin`).

One nuance on the ubatch: typed prints `--ubatch-size 1024`, while every QNN.md
measurement ran at `-ub 512`. QNN.md's bucket rule is arithmetic, not a measurement: a
ubatch above 512 tokens pads to the 1024 bucket, where the default 10 MiB fp16 IO cap
admits only weights under 5120 wide, so wider placed weights compute on the CPU for
those batches. If you want the measured NPU configuration rather than typed's measured
CPU one, `-ub 512` is the setting QNN.md used (`GGML_QNN_NPAD` defaults to 512 and must
stay at or under `-ub`).

**Runtime environment.** Before starting, make sure `QNN_SDK_ROOT` (or the SDK's
`lib/aarch64-windows-msvc` on `PATH`) and `ADSP_LIBRARY_PATH=<sdk>/lib/hexagon-v73/unsigned`
are set -- QNN.md's Requirements, repeated here because the failure mode is a silent one:
on QAIRT 2.45+ the first graph execute dies without it.

**Model.** typed's plan is pinned to one preset,
`unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_S`,
and the model must have a tool-call-capable chat template or typed's sessions run
chat-only. Every QNN.md measurement is on Qwen3-4B-Q4_K_M; no NPU measurement exists for
typed's preset (see Status).

## 3. Point typed at it

- If the server is on `127.0.0.1:9931` and nothing else answers there, nothing to do:
  typed finds it at launch.
- Otherwise: `TYPED_LOCAL_ENDPOINT=http://<origin>` -- origin only, no `/v1`, the CLI
  appends `/v1/messages` itself -- and, to pin the engine,
  `TYPED_LOCAL_ENGINE=llama-server` or `/local llama-server` inside a session (valid
  values: `llama-server`, `snapdragon`). A QNN llama-server IS the `llama-server`
  engine from typed's point of view; typed probes capability, never identity, so it
  does not know or care that the NPU is behind it. (The `snapdragon` engine is a
  different project entirely: the GenieX Server on port 8123, typed's other Snapdragon
  path.)
- Then: `typed --model typed-local`, or plain `typed` with no paid plan;
  `typed use typed-local` pins the tier.

The typed-side requirements that do not change on this build: 32 GB of RAM floor, the
window sizing, the `typed local` flags -- all as documented in LOCAL_MODEL.md. The NPU
does not reduce the host-side RAM plan: weights are still resident in host memory, and
the backend adds its own budget on top (default 1024 MB of static-weight bakes,
`GGML_QNN_STATIC_BUDGET_MB`).

## 4. What typed checks at startup

1. **`GET /props` within 500 ms** (`TYPED_CLI_LOCAL_PROBE_TIMEOUT_MS`, 500-60000 ms,
   raises it for a busy or remote box). Missing it reads the server as down. From
   `/props` typed takes `n_ctx` (the window it plans against), the model name it
   reports in `tier typed-local -- the server at http://127.0.0.1:9931 is serving
   <model>`, and the sampler settings it compares against the session's mode.
2. **The tool probe:** one streamed `POST /v1/messages` carrying a single tool,
   `max_tokens: 1`; typed reads only the status line and lets go. 2xx -> tool calls
   enabled (you see `(local tool calls: enabled`); anything else -> chat-only for the
   session, with a `!` note quoting the server's answer. `TYPED_CLI_NO_TOOL_PROBE`
   skips it, which also means no tools. llama-server b9426 and later send the streamed
   headers before reading the prompt, so the verdict arrives in tens of milliseconds
   whatever the prefill speed; this fork's tree is well past that floor (build 11280 on
   this box), so the probe does not wait on an NPU prefill.
3. **The turn itself:** `POST /v1/messages`, streamed, through the whole session.

Quick manual checks, in PowerShell as `curl.exe` (a bare `curl` is an alias for
`Invoke-WebRequest` in PowerShell 5.1):

```
curl.exe http://127.0.0.1:9931/health
curl.exe http://127.0.0.1:9931/props
```

`/health` answers once the model has loaded. Context creation on a QNN build also pays
the static-weight bakes (tens of ms to seconds per shape) at reserve, so model load and
first context are slower than steady state -- QNN.md says explicitly that this is
expected, not a hang.

To see what actually reached the NPU, set `GGML_QNN_STATS=<file>` in the server's
environment: sixteen counters written when the session is freed, including
`weights_placed`, `weights_baked` and `exec_count`. Throughput cannot tell you which
engine ran; the counters can (QNN.md's own standard of proof).

## 5. Where the two documents disagree

- **Install path.** LOCAL_MODEL.md points at winget, the upstream releases page, and
  `typed local install` -- all upstream CPU/Vulkan builds. None of them contains the QNN
  backend. This fork's build is a fourth path the typed guide does not mention, and it
  needs the QAIRT SDK to produce.
- **Default port.** LOCAL_MODEL.md says the CLI defaults to 9931 because that is
  "llama-server's default bind on current builds"; this fork's llama-server still binds
  8080 by default and prints a notice that 9931 is coming. Keeping `--port 9931` in the
  command makes the disagreement moot -- but a server started without it will not be
  found.
- **`min_p` / `presence_penalty`.** LOCAL_MODEL.md documents, as measured on b10333,
  that a stock `/v1/messages` drops both request fields, so the launch flags carry them
  and typed reads them back from `/props`. This fork carries a patch (2026-09-29,
  "server : pass min_p and presence_penalty through /v1/messages") that forwards both
  from the request. The launch flags still work and `/props` still reports them, so
  typed's check is unaffected; just know that the fork's behavior differs from what
  typed's doc describes as universal.
- **Build floor.** LOCAL_MODEL.md requires b10333 or newer (its flags are rejected
  outright by older builds). The fork tracks upstream by merges, most recently
  2026-09-27, and its binary reports its own build count (11280 on this box) rather
  than an upstream b-number -- so "is it new enough" cannot be read off the b-number;
  run `llama-server --help` and check the flags are accepted (they are, on this box).
- **GPU vs NPU wording.** LOCAL_MODEL.md's offload section says the Adreno path is
  "drop `--device none` and add `-ngl`". The NPU path is `-dev QNN` instead, and QNN.md
  adds that with full GPU offload (`-ngl 99`) the QNN backend is idle by design; with
  partial offload it takes a bounded slice of the CPU-resident matmuls. The two
  accelerators compose in one binary; they are not alternative spellings of the same
  flag.

## 6. Status and limitations (read this before relying on it)

This is a research backend -- QNN.md's first line of the status section says so, and the
summary below is only that: the full caveats, with the measurements behind them, are in
QNN.md's Status, Measured comparison and Known limitations sections.

- **What is measured:** one model (Qwen3-4B-Q4_K_M), one machine (Snapdragon X Elite,
  Windows 11 ARM64, QAIRT 2.45). At the default weight budget, `-dev QNN` prefill is
  +7% against the CPU alone in the same run (+12% against the previous run's CPU
  legs), with 41 of 252 projections on the NPU and 211 on the CPU with repack. Earlier
  builds without the placement budget read 1.67-1.76x SLOWER; `GGML_QNN_STATIC_BUDGET_MB=0`
  still reads 1.8x slower. The verdict is on that configuration and that model.
- **Decode never runs on the NPU in typed's shape.** The backend claims no matmul below
  32-token ubatches (`GGML_QNN_MIN_DIM`), which covers single-stream decode, so every
  decode figure anywhere in QNN.md is CPU decode -- and is deliberately not quoted as a
  point because it is unstable on that box. typed's measured `--spec-type ngram-cache`
  decode win was measured on a CPU build and has not been re-measured against a QNN
  build.
- **Output differs slightly from the CPU's** with weights on the NPU: median KL
  divergence 0.0016-0.0018, top token matching 96.5-97.3%, about what the CPU's own two
  attention paths differ by -- except placements with 43+ weights on the NPU show a few
  positions (three of 2040) that flip completely, which is not explained. QNN.md's
  verdict stands: these runs do not show that the NPU costs accuracy, and do not show
  it equals the CPU.
- **Everything else falls back to the CPU automatically**: unsupported shapes, IO over
  the cap, watchdog timeouts (15 s execute / 120 s finalize), a slow execute (>= 2 s),
  denylisted shapes. None of it fails a request; the model keeps running.
- **Windows ARM64 only in practice** -- the Linux dlopen paths exist but are untested.
- **typed's preset on the NPU is unmeasured.** Qwen3.6-35B-A3B at UD-Q4_K_S, a 65536
  window, `--ubatch-size 1024`, multi-turn agentic prefills: none of that is in QNN.md's
  record. Run `GGML_QNN_STATS` yourself and read the counters -- that is the documented
  way to know what the NPU did.

Nothing about typed's behavior changes with this server behind it: same port contract,
same `/props` and tool-probe checks, same guardrails, same 413/503 honesty when the
window or the server runs out. If the server dies, typed reports the tier as down with
no failover -- that is the free tier's shape on every backend.