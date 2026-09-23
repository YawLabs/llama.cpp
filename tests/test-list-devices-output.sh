#!/usr/bin/env bash
# Regression test: --list-devices must produce non-empty output in common-arg tools.
# The output was silently lost when a backend DLL crashed during process teardown before
# the CRT flushed the buffered stdout (fixed by flushing in the handler). Exit code alone
# cannot catch that class - the process exited 0 with zero bytes - so this asserts bytes.
#
# Usage: test-list-devices-output.sh <path-to-llama-cli>

set -u

BIN="${1:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "usage: $0 <path-to-llama-cli>" >&2
    exit 1
fi

OUT="$("$BIN" --list-devices 2>&1)"
RC=$?

if [ $RC -ne 0 ]; then
    echo "FAIL: --list-devices exited $RC" >&2
    printf '%s\n' "$OUT" >&2
    exit 1
fi

if ! printf '%s' "$OUT" | grep -q "Available devices:"; then
    echo "FAIL: --list-devices printed no device listing (bytes: ${#OUT})" >&2
    printf '%s\n' "$OUT" >&2
    exit 1
fi

echo "OK: --list-devices produced a device listing (${#OUT} bytes)"
exit 0
