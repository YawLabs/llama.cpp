#!/usr/bin/env bash
#
# Local CI for this fork, replacing the 58 GitHub Actions workflows that were removed.
#
# Why they went: every one of them is upstream's, and none ever passed here. A push to this
# fork fired the lot and each returned startup_failure - they want runners this account does
# not have (self-hosted CUDA / Metal / Vulkan / WebGPU boxes) and secrets it does not hold
# (QDC_API_KEY, HF_TOKEN_CI, DEPLOY_KEY_RELEASE, PYPI_API_TOKEN, WINGET_GITHUB_TOKEN). The
# signal-to-noise was zero: dozens of red runs per push, none of which said anything about
# whether the NPU backend works. Actions is also disabled at the repo level, so re-adding a
# workflow file by merging upstream will not start anything.
#
# What actually guards this fork is here instead, and it runs on the one machine that has the
# hardware: a Snapdragon X Elite with a Hexagon NPU and an Adreno X1-85.
#
# The parts of upstream's CI that apply to this box are carried over:
#   - its unit tests, as ci/run.sh and build-cpu.yml ran them: ctest -L main, plus -L python
#     and -L model when their inputs are here (cpu and gpu)
#   - every op against the CPU reference, as ci/run.sh ran test-backend-ops for the
#     KleidiAI job (-b CPU, cpu) and the GPU jobs (here -b GPUOpenCL, gpu)
#   - editorconfig.yml, on the files this fork changed (every mode)
# The rest of them target other hardware, other OSes, or publishing.
#
# Usage:
#   scripts/fork-ci.sh            # build the QNN config and run the fork's tests
#   scripts/fork-ci.sh npu        # same, explicitly
#   scripts/fork-ci.sh gpu        # the OpenCL config
#   scripts/fork-ci.sh cpu        # the plain CPU config
#   scripts/fork-ci.sh all        # all three
#   CONFIGURE=1 scripts/fork-ci.sh npu     # force a fresh cmake configure
#   JOBS=4 scripts/fork-ci.sh cpu          # build parallelism (default: every core)
#   BUILD_ONLY=1 scripts/fork-ci.sh gpu    # skip the tests (scripts/fork-build.sh sets this)
#   TEST_MODEL=m.gguf scripts/fork-ci.sh cpu   # model for the -L model tests (default below)
#
# A failed test step does not stop the others (a failed build does); the script exits 1 at
# the end if any step failed.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLANG_DIR="${CLANG_DIR:-C:/Users/jeff/scoop/apps/llvm-arm64/current/bin}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-C:/Users/jeff/yaw/genie-npu/qairt/2.45.0.260326}"
OPENCL_SDK="${OPENCL_SDK:-C:/Users/jeff/yaw/sdk}"
# upstream CI gives its -L model tests Qwen3-0.6B. Not the stories15M test model: with it,
# test-backend-sampler's logit-bias check fails (+10 on "World" does not win after "Hello")
TEST_MODEL="${TEST_MODEL:-C:/Users/jeff/yaw/genie-npu/gguf/Qwen3-4B-Q4_K_M.gguf}"

# Shared base. GGML_CPU_ARM_ARCH must stay quoted: PowerShell splits an unquoted -D value at
# the dot and silently degrades the build to a bare armv8 baseline, which only shows up as
# slower inference. Git Bash is unaffected, which is why this script is bash.
BASE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=${CLANG_DIR}/clang.exe"
    "-DCMAKE_CXX_COMPILER=${CLANG_DIR}/clang++.exe"
    "-DCMAKE_CXX_FLAGS=-DNOMINMAX"
    -DGGML_NATIVE=OFF
    "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm"
    -DGGML_CPU_KLEIDIAI=ON
    -DGGML_CPU_REPACK=ON
    -DGGML_LLAMAFILE=ON
    -DGGML_OPENMP=OFF
    # no HTTPS (no -hf downloads): there is no OpenSSL on this box, and LLAMA_CURL is gone
    # upstream. -DLLAMA_BUILD_BORINGSSL=ON instead would fetch and build BoringSSL for it.
    -DLLAMA_OPENSSL=OFF
    # build the server UI from this tree with npm, and never fall back to a prebuilt upstream
    # UI: that one is keyed to the commit count, which in a fork names a different upstream
    # build, and the fallback would be silent if npm failed
    -DLLAMA_BUILD_UI=ON
    -DLLAMA_USE_PREBUILT_UI=OFF
)

say() { printf '\n=== %s ===\n' "$1"; }

# BUILD_ONLY follows CONFIGURE's convention: unset, empty or 0 means off
build_only() { case "${BUILD_ONLY:-0}" in 0|"") return 1 ;; *) return 0 ;; esac; }

build_one() {
    local target="$1" dir="$2"; shift 2

    # never build while a binary from this dir is running: relinking an in-use DLL or EXE
    # fails with a permission error that looks like a compiler problem. Opening for append
    # without writing leaves the mtime alone.
    local f busy=()
    for f in "$dir"/bin/*.dll "$dir"/bin/*.exe; do
        [ -e "$f" ] || continue
        { : >> "$f"; } 2>/dev/null || busy+=("${f##*/}")
    done
    if [ ${#busy[@]} -gt 0 ]; then
        echo "error: in use from $dir/bin: ${busy[*]} - stop those processes first" >&2
        return 1
    fi

    # always configure (~2 s on a warm tree): re-applies BASE_FLAGS and the extras to an
    # existing cache and retries a configure that failed. CONFIGURE=1 adds --fresh, which
    # also drops cached values this script no longer passes.
    local fresh=()
    case "${CONFIGURE:-0}" in 0|"") ;; *) fresh=(--fresh) ;; esac
    say "configure $target -> $dir"
    cmake -B "$dir" "${fresh[@]}" "${BASE_FLAGS[@]}" "$@"
    say "build $target"
    cmake --build "$dir" -j "${JOBS:-$(nproc)}"
}

run_fork_tests() {
    local dir="$1"
    export QNN_SDK_ROOT
    export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"
    # this is the box with the HTP: a missing or unusable NPU must fail the suite, not skip it
    export GGML_QNN_TEST_REQUIRE_HTP=1

    say "fork test suite ($dir)"
    # test-qnn-health is excluded on purpose: its pass criterion is wall-clock, so it only
    # means something with NOTHING else running on the machine. Run it by hand on an idle box:
    #   ctest --test-dir BUILD -R '^test-qnn-health$' --output-on-failure
    ctest --test-dir "$dir" \
        -R '^(test-qnn-|test-list-devices|test-kleidiai|test-backend-ops)' \
        -E '^test-qnn-health$' \
        --output-on-failure --no-tests=error
}

# -dev QNN must leave flash attention on: llama used to disable it for the whole model because
# the op runs on the CPU while the layer's device is QNN (fixed in src/llama-context.cpp).
# Nothing is baked (budget 1 MB): the decision depends on the device, not on what runs there.
# No -fa: only "auto" is resolved, an explicit "on" would pass with the fix removed
run_qnn_fused_ops() {
    local dir="$1" log="$1/qnn-fused-ops.log" rc=0
    say "flash attention under -dev QNN ($dir)"
    if [ ! -f "$TEST_MODEL" ]; then
        echo "note: skipped, TEST_MODEL does not exist: $TEST_MODEL"
        skipped+=("flash attention under -dev QNN ($dir): no $TEST_MODEL")
        return 0
    fi
    export QNN_SDK_ROOT
    export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"
    GGML_QNN_STATIC_BUDGET_MB=1 timeout 300 "$dir/bin/llama-completion" -m "$TEST_MODEL" -dev QNN \
        -no-cnv -n 1 -p Hello -lv 4 > "$log" 2>&1 < /dev/null || rc=$?
    if [ "$rc" -ne 0 ]; then
        tail -n 20 "$log"
        echo "error: llama-completion -dev QNN exited $rc, full log in $log" >&2
        return "$rc"
    fi
    if grep -q 'Flash Attention not supported, set to disabled' "$log"; then
        echo "error: -dev QNN disabled flash attention, see $log" >&2
        return 1
    fi
    # the positive line too, so a reworded upstream message fails here instead of passing
    grep 'Flash Attention enabled' "$log" || { echo "error: no 'Flash Attention enabled' line in $log" >&2; return 1; }
}

# every op of one device against the CPU reference implementation, as upstream's ci/run.sh ran
# test-backend-ops (its KleidiAI job with -b CPU): ~20k cases on the CPU (7 min), ~10k on the
# GPU (6 min). The GPU gets one worker: every OpenCL backend instance shares one context (its
# queue and kernels), so two workers race and fail tests that pass alone
run_op_tests() {
    local dir="$1" dev="$2" jobs="$3" log="$1/test-backend-ops-$2.log" rc=0
    say "$dev op tests ($dir)"
    timeout 3600 "$dir/bin/test-backend-ops" -b "$dev" -j "$jobs" > "$log" 2>&1 || rc=$?
    grep -E 'tests passed|backends passed' "$log" || true
    if [ "$rc" -ne 0 ]; then
        if grep -q '^Failing tests:' "$log"; then
            sed -n '/^Failing tests:/,/^ *Backend /p' "$log" | head -n 60
        else
            tail -n 20 "$log"
        fi
        echo "error: test-backend-ops -b $dev exited $rc, full log in $log" >&2
        return "$rc"
    fi
    # -b is an exact device-name match; with no such device every backend is "Skipping" and
    # the binary still exits 0, so also require a non-zero pass count
    grep -Eq '^ +[1-9][0-9]*/[0-9]+ tests passed' "$log" || { echo "error: no $dev op test ran, see $log" >&2; return 1; }
}

# upstream's own unit tests. GGML_SCHED_DEBUG_REALLOC=1 is from build-cpu.yml: it aborts on a
# graph reallocation the scheduler did not expect. No test is excluded: each one can run here
run_upstream_tests() {
    local dir="$1" labels=main rc=0
    say "upstream unit tests ($dir)"
    # label python is test-jinja-py: it renders with python's jinja2, and on 3.0 it fails for
    # want of the items filter. upstream CI pins 3.1.6
    if python -c 'import sys, jinja2; sys.exit(tuple(map(int, jinja2.__version__.split(".")[:2])) < (3, 1))' 2>/dev/null; then
        labels='main|python'
    else
        echo "note: test-jinja-py skipped, it needs python with jinja2 >= 3.1 (pip install jinja2==3.1.6)"
        skipped+=("test-jinja-py ($dir): jinja2 < 3.1")
    fi
    GGML_SCHED_DEBUG_REALLOC=1 ctest --test-dir "$dir" -L "$labels" --output-on-failure --no-tests=error --timeout 900 || rc=$?

    say "upstream model tests ($dir)"
    if [ -f "$TEST_MODEL" ]; then
        LLAMACPP_TEST_MODELFILE="$TEST_MODEL" GGML_SCHED_DEBUG_REALLOC=1 \
            ctest --test-dir "$dir" -L model --output-on-failure --no-tests=error --timeout 900 || rc=$?
    else
        echo "note: skipped, TEST_MODEL does not exist: $TEST_MODEL"
        skipped+=("-L model ($dir): no $TEST_MODEL")
    fi
    return "$rc"
}

# upstream's editorconfig.yml, on the files this fork changed since the last upstream merge.
# The check runs on a copy with CR stripped: core.autocrlf makes CRLF work trees of LF blobs
run_editorconfig() {
    local base tmp f rc=0
    local -a files
    say "editorconfig"
    if ! command -v editorconfig-checker >/dev/null; then
        echo "note: skipped, editorconfig-checker is not on PATH"
        skipped+=("editorconfig: checker not on PATH")
        return 0
    fi
    if ! base="$(git merge-base HEAD upstream/master 2>/dev/null)"; then
        echo "note: skipped, no upstream/master to find the fork's files against"
        skipped+=("editorconfig: no upstream/master")
        return 0
    fi
    mapfile -t files < <({ git diff --name-only --diff-filter=d "$base" -- . ':!.github'; git ls-files --others --exclude-standard; } | sort -u)
    tmp="$(mktemp -d)"
    cp .editorconfig .ecrc "$tmp/"
    for f in "${files[@]}"; do
        mkdir -p "$tmp/$(dirname "$f")"
        tr -d '\r' < "$f" > "$tmp/$f"
    done
    (cd "$tmp" && editorconfig-checker -no-color "${files[@]}") || rc=$?
    rm -rf "$tmp"
    [ "$rc" -eq 0 ] && echo "${#files[@]} files OK"
    return "$rc"
}

# run a test step and carry on if it fails, so that one red step does not hide the others
failed=()
skipped=()
check() { "$@" || failed+=("$*"); }

finish() {
    if [ ${#skipped[@]} -gt 0 ]; then
        say "skipped"
        printf '  %s\n' "${skipped[@]}"
    fi
    if [ ${#failed[@]} -gt 0 ]; then
        say "FAILED"
        printf '  %s\n' "${failed[@]}"
        exit 1
    fi
    say "done"
    exit 0
}

mode="${1:-npu}"
case "$mode" in
    npu|gpu|cpu|all) ;;
    *)
        echo "usage: $0 [npu|gpu|cpu|all]" >&2
        exit 1
        ;;
esac

build_only || check run_editorconfig

case "$mode" in
    npu|all)
        build_one "QNN / Hexagon NPU" build-npu \
            -DGGML_QNN=ON "-DQNN_SDK_ROOT=${QNN_SDK_ROOT}"
        if ! build_only; then
            check run_fork_tests build-npu
            check run_qnn_fused_ops build-npu
        fi
        [ "$mode" = "all" ] || finish
        ;&
    gpu)
        # do NOT add GGML_OPENCL_USE_ADRENO_BIN_KERNELS: that prebuilt lib targets X2 GPUs and
        # this box is an X1-85
        build_one "OpenCL / Adreno X1-85" build-gpu \
            -DGGML_OPENCL=ON \
            -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
            -DGGML_OPENCL_EMBED_KERNELS=ON \
            "-DOpenCL_INCLUDE_DIR=${OPENCL_SDK}/OpenCL-Headers" \
            "-DOpenCL_LIBRARY=${OPENCL_SDK}/OpenCL.lib"
        if ! build_only; then
            check run_op_tests build-gpu GPUOpenCL 1
            check run_upstream_tests build-gpu
        fi
        [ "$mode" = "all" ] || finish
        ;&
    cpu)
        build_one "CPU / KleidiAI" build-cpu
        if ! build_only; then
            check run_upstream_tests build-cpu
            check run_op_tests build-cpu CPU 2
        fi
        ;;
esac

finish
