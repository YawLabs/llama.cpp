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
# Usage:
#   scripts/fork-ci.sh            # build the QNN config and run the fork's tests
#   scripts/fork-ci.sh npu        # same, explicitly
#   scripts/fork-ci.sh gpu        # the OpenCL config
#   scripts/fork-ci.sh cpu        # the plain CPU config
#   scripts/fork-ci.sh all        # all three
#   CONFIGURE=1 scripts/fork-ci.sh npu     # force a fresh cmake configure
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLANG_DIR="${CLANG_DIR:-C:/Users/jeff/scoop/apps/llvm-arm64/current/bin}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-C:/Users/jeff/yaw/genie-npu/qairt/2.45.0.260326}"
OPENCL_SDK="${OPENCL_SDK:-C:/Users/jeff/yaw/sdk}"

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
    -DLLAMA_CURL=OFF
)

say() { printf '\n=== %s ===\n' "$1"; }

build_one() {
    local target="$1" dir="$2"; shift 2
    local extra=("$@")

    if [ ! -f "$dir/CMakeCache.txt" ] || [ -n "${CONFIGURE:-}" ]; then
        say "configure $target -> $dir"
        cmake -B "$dir" "${BASE_FLAGS[@]}" "${extra[@]}"
    fi
    say "build $target"
    # never build while a binary from this dir is running: relinking ggml.dll then fails with
    # a permission error that looks like a compiler problem
    cmake --build "$dir" -- -j "${JOBS:-6}"
}

run_fork_tests() {
    local dir="$1"
    export QNN_SDK_ROOT
    export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"

    say "fork test suite ($dir)"
    # test-qnn-health is excluded on purpose: its pass criterion is wall-clock, so it only
    # means something with NOTHING else running on the machine. Run it by hand on an idle box:
    #   ctest --test-dir BUILD -R '^test-qnn-health$' --output-on-failure
    ctest --test-dir "$dir" \
        -R '^(test-qnn-|test-list-devices|test-kleidiai|test-backend-ops)' \
        -E '^test-qnn-health$' \
        --output-on-failure
}

mode="${1:-npu}"

case "$mode" in
    npu|all)
        build_one "QNN / Hexagon NPU" build-npu \
            -DGGML_QNN=ON "-DQNN_SDK_ROOT=${QNN_SDK_ROOT}"
        run_fork_tests build-npu
        [ "$mode" = "all" ] || exit 0
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
        [ "$mode" = "all" ] || exit 0
        ;&
    cpu)
        build_one "CPU / KleidiAI" build-cpu
        ;;
    *)
        echo "usage: $0 [npu|gpu|cpu|all]" >&2
        exit 1
        ;;
esac

say "done"
