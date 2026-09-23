#!/usr/bin/env bash
# Regression test: --list-devices must produce its listing on stdout in common-arg tools.
# The output was silently lost when a backend DLL crashed during process teardown before
# the CRT flushed the buffered stdout (fixed by flushing in the handler). Exit code alone
# cannot catch that class - the process exited 0 with zero bytes - so this asserts bytes.
# stdout and stderr are captured separately: backend init logs go to stderr, and merging
# them would let a stray "Available devices:" on stderr hide a lost stdout.
#
# Usage: test-list-devices-output.sh <path-to-llama-cli>

set -u

BIN="${1:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "usage: $0 <path-to-llama-cli>" >&2
    exit 1
fi

TMP="$(mktemp -d)" || { echo "FAIL: mktemp -d failed" >&2; exit 1; }
trap 'rm -rf "$TMP"' EXIT

"$BIN" --list-devices >"$TMP/out" 2>"$TMP/err"
RC=$?

dump() {
    echo "--- stdout ($(wc -c <"$TMP/out") bytes) ---" >&2
    cat "$TMP/out" >&2
    echo "--- stderr ($(wc -c <"$TMP/err") bytes) ---" >&2
    cat "$TMP/err" >&2
}

if [ $RC -ne 0 ]; then
    echo "FAIL: --list-devices exited $RC" >&2
    dump
    exit 1
fi

# strip CR so a Windows binary's CRLF output matches the anchored patterns below
tr -d '\r' <"$TMP/out" >"$TMP/out.lf"

if ! grep -q '^Available devices:$' "$TMP/out.lf"; then
    echo "FAIL: --list-devices printed no device listing on stdout" >&2
    dump
    exit 1
fi

# common_print_available_devices always prints at least one indented row after the header:
# "  (none)" in a CPU-only build, "  <name>: <description> (...)" per non-CPU device otherwise
if ! sed -n '/^Available devices:$/,$p' "$TMP/out.lf" | grep -q '^  [^ ]'; then
    echo "FAIL: --list-devices printed the header but no device row" >&2
    dump
    exit 1
fi

echo "OK: --list-devices produced a device listing on stdout ($(wc -c <"$TMP/out") bytes)"
exit 0
