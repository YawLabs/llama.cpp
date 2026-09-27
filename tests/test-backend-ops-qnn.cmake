# Runs test-backend-ops against the QNN backend and fails if it compared nothing.
#
# Two reasons this needs a driver instead of a plain add_test:
#
# 1. Upstream b78a39a2f made test-backend-ops a build-only target run from ci/run.sh,
#    so a QNN build gets no op-level correctness entry unless the fork adds one.
#
# 2. test-backend-ops reports "Backend QNN: OK" when every case was REFUSED by
#    supports_op - a pass that compared zero numbers. Measured 2026-09-22, when one
#    1 MiB IO cap covered every graph: default env 0 executed, NPAD=32 alone 2,
#    NPAD=32 + MIN_DIM=1 47. The default GGML_QNN_MIN_DIM of 32 declines most shapes
#    (m is 1 or 16, or n is under 32). The default env was not measured again after
#    the cap default changed (2026-09-27: unset, fp16 graph IO is capped at 10 MiB,
#    fp32 at 1 MiB), so the driver does not rely on it: it pins the settings that
#    make the shapes claimable AND asserts that a non-zero number of cases actually ran.
#
# GGML_QNN_IO_MAX_KB is pinned too, to unset: a value exported by the caller's shell
# caps every graph at that size and changes how many cases run (a set 1024 refuses
# the two F16 cases whose padded input is 2 MiB).
#
# Parsing note: the QNN SDK writes progress bars with bare CR and its own log lines
# interleave with the per-case results, so counting "): OK" lines undercounts wildly.
# The binary's own "N/M tests passed" summary is the reliable signal - assert on that.

if (NOT DEFINED BIN)
    message(FATAL_ERROR "BIN not set: pass -DBIN=<path to test-backend-ops>")
endif()

set(ENV{GGML_QNN_NPAD}    "32")
set(ENV{GGML_QNN_MIN_DIM} "1")
unset(ENV{GGML_QNN_IO_MAX_KB})

execute_process(
    COMMAND "${BIN}" -b QNN -o MUL_MAT
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err
    RESULT_VARIABLE _rc
)
set(_all "${_out}${_err}")

string(ASCII 27 _esc)
string(REGEX REPLACE "${_esc}\\[[0-9;]*m" "" _all "${_all}")
string(REGEX REPLACE "\r" "\n" _all "${_all}")

# no usable HTP: skip rather than fail, matching the lifecycle suite's SKIP_RETURN_CODE 77
if (_all MATCHES "QnnHtp.dll could not be loaded|backend unavailable|no QNN device")
    # like the lifecycle suite: GGML_QNN_TEST_REQUIRE_HTP (non-empty, not "0") turns the skip into a failure
    if (NOT "$ENV{GGML_QNN_TEST_REQUIRE_HTP}" STREQUAL "" AND NOT "$ENV{GGML_QNN_TEST_REQUIRE_HTP}" STREQUAL "0")
        message(STATUS "${_all}")
        message(FATAL_ERROR "no usable HTP, and GGML_QNN_TEST_REQUIRE_HTP is set")
    endif()
    message(STATUS "no usable HTP, skipping")
    return()
endif()

if (NOT _all MATCHES "Backend[ 0-9/]*: *QNN")
    message(STATUS "${_all}")
    message(FATAL_ERROR "test-backend-ops never reached the QNN backend")
endif()

if (NOT _all MATCHES "([0-9]+)/([0-9]+) tests passed")
    message(STATUS "${_all}")
    message(FATAL_ERROR "no 'N/M tests passed' summary in the output")
endif()
set(_passed ${CMAKE_MATCH_1})
set(_total  ${CMAKE_MATCH_2})

message(STATUS "QNN MUL_MAT: ${_passed}/${_total} executed cases matched the CPU")

if (_total EQUAL 0)
    message(STATUS "${_all}")
    message(FATAL_ERROR
        "test-backend-ops executed ZERO QNN MUL_MAT cases, so a green result proves nothing "
        "about the backend. Every shape was declined by supports_op - check GGML_QNN_NPAD, "
        "GGML_QNN_MIN_DIM and GGML_QNN_IO_MAX_KB.")
endif()

if (NOT _passed EQUAL _total OR NOT _rc EQUAL 0)
    message(STATUS "${_all}")
    message(FATAL_ERROR "QNN MUL_MAT correctness: ${_passed}/${_total} passed, exit ${_rc}")
endif()
