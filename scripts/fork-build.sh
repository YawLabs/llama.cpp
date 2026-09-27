#!/usr/bin/env bash
#
# Build the llama.cpp binaries for this box: the CPU version (KleidiAI) and the GPU version
# (OpenCL on the Adreno X1-85). Build only; no tests. The binaries land in build-cpu/bin and
# build-gpu/bin (llama-cli, llama-server, llama-bench, ...).
#
# A thin wrapper: scripts/fork-ci.sh holds the one copy of the build flags, so the two cannot
# drift. The GPU build also contains the CPU backend, so `-ngl 0` runs it on the CPU alone.
#
# Usage:
#   scripts/fork-build.sh           # both
#   scripts/fork-build.sh cpu       # CPU only
#   scripts/fork-build.sh gpu       # GPU only
#   CONFIGURE=1 scripts/fork-build.sh      # force a fresh cmake configure first
#   JOBS=4 scripts/fork-build.sh           # fewer parallel jobs (default: every core)
#
# Stop any llama-server / llama-cli running from build-cpu or build-gpu first: the build refuses
# to start while a binary in that bin/ is in use, since relinking it would fail.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# build only: fork-ci.sh skips its tests when this is set
export BUILD_ONLY=1

# run through bash: fork-ci.sh is stored without the executable bit. In "both", cpu goes first:
# it needs no SDK, so an OpenCL setup problem cannot cost you the CPU build.
case "${1:-both}" in
    cpu)  bash "$here/fork-ci.sh" cpu ;;
    gpu)  bash "$here/fork-ci.sh" gpu ;;
    both) bash "$here/fork-ci.sh" cpu && bash "$here/fork-ci.sh" gpu ;;
    *)
        echo "usage: $0 [cpu|gpu|both]" >&2
        exit 1
        ;;
esac
