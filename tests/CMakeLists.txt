llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common) # TODO: [TAG_TESTS_LLAMA_LINK]
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-save-load-state.cpp)

if (GGML_QNN)
    # QNN backend lifecycle tests: need real HTP hardware, skip cleanly elsewhere.
    # Serial: the HTP is single-client. The watchdog, fault, slow-* and *-timeout modes leave
    # a degraded session behind on purpose and end the process without running DLL detach.
    llama_build(test-qnn-lifecycle.cpp)

    llama_test(test-qnn-lifecycle NAME test-qnn-basic    ARGS basic)
    llama_test(test-qnn-lifecycle NAME test-qnn-budget   ARGS budget)
    llama_test(test-qnn-lifecycle NAME test-qnn-denylist ARGS denylist)
    llama_test(test-qnn-lifecycle NAME test-qnn-watchdog ARGS watchdog)
    llama_test(test-qnn-lifecycle NAME test-qnn-fault    ARGS fault)
    llama_test(test-qnn-lifecycle NAME test-qnn-clamp    ARGS clamp)
    # device health, the one entry that depends on how fast the HTP executes: it fails on an
    # NPU that works but runs slowly, which every functional entry tolerates. only meaningful
    # with nothing else running on the machine
    llama_test(test-qnn-lifecycle NAME test-qnn-health   ARGS health)

    # the clamp at GGML_QNN_STATIC_BUDGET_MB=0 (it must still bite), and the denylist under
    # GGML_QNN_NO_OPT, where file entries are advisory. each mode sets its variable itself
    llama_test(test-qnn-lifecycle NAME test-qnn-clamp-unlimited ARGS clamp unlimited)
    llama_test(test-qnn-lifecycle NAME test-qnn-denylist-noopt  ARGS denylist noopt)

    # green guard for the working side of the padded-IO-size law, plus bucket-boundary
    # rounding, the exact-pow2 bucket branch, and the NO_OPT lever on a passing graph.
    # the third ARG names the env var the runner must have set: the -noopt, -shm, -nostatic
    # and -noburst variants are otherwise byte-identical invocations of test-qnn-modelscale
    llama_test(test-qnn-lifecycle NAME test-qnn-modelscale          ARGS modelscale)
    llama_test(test-qnn-lifecycle NAME test-qnn-modelscale-npad0    ARGS modelscale 0)
    llama_test(test-qnn-lifecycle NAME test-qnn-noopt               ARGS modelscale 64 GGML_QNN_NO_OPT)
    llama_test(test-qnn-lifecycle NAME test-qnn-modelscale-nostatic ARGS modelscale 64 GGML_QNN_NO_STATIC_WEIGHTS)
    llama_test(test-qnn-lifecycle NAME test-qnn-modelscale-noburst  ARGS modelscale 64 GGML_QNN_NO_BURST)

    # config-surface guards: the A/B kill-switch, the default min-dim gate, single-bake
    # across batch sizes, and the rpcmem shared-memory IO path
    llama_test(test-qnn-lifecycle NAME test-qnn-disable        ARGS disable)
    llama_test(test-qnn-lifecycle NAME test-qnn-mindim         ARGS mindim)
    llama_test(test-qnn-lifecycle NAME test-qnn-rebake         ARGS rebake)
    llama_test(test-qnn-lifecycle NAME test-qnn-modelscale-shm ARGS modelscale 64 GGML_QNN_SHARED_MEM)

    # default-env correctness gates: the elementwise opt-in must stay off (the HTP has a
    # known Add broadcast bug), and supports_op must not change its verdict between the
    # unallocated probe llama does at model load and the resident one at schedule time.
    # -on: with the opt-in set, the min-elements gate is the one left in front of the HTP
    llama_test(test-qnn-lifecycle NAME test-qnn-elementwise    ARGS elementwise)
    llama_test(test-qnn-lifecycle NAME test-qnn-elementwise-on ARGS elementwise-on)
    llama_test(test-qnn-lifecycle NAME test-qnn-loadprobe      ARGS loadprobe)

    # small F32 shapes, fast on any device: two backends on one session, the dynamic-weight
    # copy cache, malformed env values falling back to their defaults. quantized builds the
    # one fp16 graph (GGML_QNN_QUANTIZED), small and under no timing check
    llama_test(test-qnn-lifecycle NAME test-qnn-reuse     ARGS reuse)
    llama_test(test-qnn-lifecycle NAME test-qnn-dyncache  ARGS dyncache)
    llama_test(test-qnn-lifecycle NAME test-qnn-quantized ARGS quantized)
    llama_test(test-qnn-lifecycle NAME test-qnn-envparse  ARGS envparse)

    # the test-only hooks GGML_QNN_DELAY_EXECUTE and GGML_QNN_FAIL_FINALIZE reach the slow,
    # timeout and finalize-error verdicts on a healthy device, with small F32 shapes. the
    # slow-* and *-timeout modes keep a degraded session and end through hard_exit, so the
    # runner keys off their marker line (below), like watchdog
    llama_test(test-qnn-lifecycle NAME test-qnn-slow-validate         ARGS slow-validate)
    llama_test(test-qnn-lifecycle NAME test-qnn-slow-compute          ARGS slow-compute)
    llama_test(test-qnn-lifecycle NAME test-qnn-validate-timeout      ARGS validate-timeout)
    llama_test(test-qnn-lifecycle NAME test-qnn-validate-timeout-cold ARGS validate-timeout cold)
    llama_test(test-qnn-lifecycle NAME test-qnn-compute-timeout       ARGS compute-timeout)
    llama_test(test-qnn-lifecycle NAME test-qnn-finalize-error        ARGS finalize-error)
    llama_test(test-qnn-lifecycle NAME test-qnn-denylist-append       ARGS denylist-append)

    set_tests_properties(test-qnn-noopt               PROPERTIES ENVIRONMENT "GGML_QNN_NO_OPT=1")
    set_tests_properties(test-qnn-modelscale-shm      PROPERTIES ENVIRONMENT "GGML_QNN_SHARED_MEM=1")
    set_tests_properties(test-qnn-modelscale-nostatic PROPERTIES ENVIRONMENT "GGML_QNN_NO_STATIC_WEIGHTS=1")
    set_tests_properties(test-qnn-modelscale-noburst  PROPERTIES ENVIRONMENT "GGML_QNN_NO_BURST=1")
    set_tests_properties(test-qnn-elementwise-on      PROPERTIES ENVIRONMENT "GGML_QNN_ELEMENTWISE=1")

    # every test-qnn-* entry registered above this line gets the suite-wide properties below;
    # register new entries (and their per-entry ENVIRONMENT) above it
    get_property(_qnn_tests DIRECTORY PROPERTY TESTS)
    list(FILTER _qnn_tests INCLUDE REGEX "^test-qnn-")

    # SKIP_RETURN_CODE: the test exits 77 when no HTP is present (-shm also when fastrpc is
    # absent). Without it every entry returned 0 and the whole suite reported green on
    # hardware that never ran QNN code.
    # TIMEOUT: a wedged HTP must not hold the serial queue for the default 1500 s. the slowest
    # entry took about 66 s on the dev box while it was loaded by other work
    set_tests_properties(${_qnn_tests} PROPERTIES RUN_SERIAL TRUE SKIP_RETURN_CODE 77 TIMEOUT 600)

    # test-qnn-watchdog carries both properties and skips with 77 like the others:
    # SKIP_RETURN_CODE decides first, so on a machine with no HTP it reports NOT RUN, measured
    # on a run where QnnHtp.dll was not loadable. PASS_REGULAR_EXPRESSION judges only the runs
    # that reached the scenario, and a missing marker fails only those.
    set_tests_properties(test-qnn-watchdog PROPERTIES
        PASS_REGULAR_EXPRESSION "WATCHDOG-CHECKS-PASSED")
    # the hook modes that end through hard_exit the same way, each with its own marker
    set_tests_properties(test-qnn-slow-validate PROPERTIES
        PASS_REGULAR_EXPRESSION "SLOW-VALIDATE-CHECKS-PASSED")
    set_tests_properties(test-qnn-slow-compute PROPERTIES
        PASS_REGULAR_EXPRESSION "SLOW-COMPUTE-CHECKS-PASSED")
    set_tests_properties(test-qnn-validate-timeout test-qnn-validate-timeout-cold PROPERTIES
        PASS_REGULAR_EXPRESSION "VALIDATE-TIMEOUT-CHECKS-PASSED")
    set_tests_properties(test-qnn-compute-timeout PROPERTIES
        PASS_REGULAR_EXPRESSION "COMPUTE-TIMEOUT-CHECKS-PASSED")
    # these build a few small graphs and finish in seconds, the fp16 ones of fault and
    # quantized included even at the slow fp16 speed; 180 s is above the 120 s build watchdog
    set_tests_properties(
        test-qnn-watchdog test-qnn-fault
        test-qnn-slow-validate test-qnn-slow-compute
        test-qnn-validate-timeout test-qnn-validate-timeout-cold test-qnn-compute-timeout
        test-qnn-finalize-error test-qnn-denylist-append
        test-qnn-reuse test-qnn-dyncache test-qnn-quantized test-qnn-envparse
        PROPERTIES TIMEOUT 180)

    # a machine without an HTP must still pass the suite (every entry reports NOT RUN), so
    # the default is OFF. turn it ON on a box that has one: a missing QNN device then fails
    # every entry instead of skipping it, so a build where the backend never registers cannot
    # report green. APPEND keeps the per-entry ENVIRONMENT values set above
    option(LLAMA_QNN_TEST_REQUIRE_HTP "llama: fail the test-qnn-* entries when no HTP is found" OFF)
    if (LLAMA_QNN_TEST_REQUIRE_HTP)
        foreach (_test IN LISTS _qnn_tests)
            set_property(TEST ${_test} APPEND PROPERTY ENVIRONMENT "GGML_QNN_TEST_REQUIRE_HTP=1")
        endforeach()
    endif()
endif()

# --list-devices must produce output: a backend-DLL teardown crash used to destroy
# the buffered stdout before the CRT flushed it, exiting 0 with zero bytes.
# tools/ is added after tests/, so llama-cli is not a target yet: test the options that create it
if (TARGET llama-cli OR (LLAMA_BUILD_TOOLS AND LLAMA_BUILD_SERVER AND NOT EMSCRIPTEN))
    find_program(BASH_EXECUTABLE bash HINTS "$ENV{ProgramFiles}/Git/usr/bin" "$ENV{ProgramFiles}/Git/bin")
    set(_list_devices_bash "${BASH_EXECUTABLE}")
    set(_list_devices_why  "bash not found")
    if (WIN32 AND BASH_EXECUTABLE AND DEFINED ENV{SystemRoot})
        # a bash under %SystemRoot% is the WSL launcher (System32/bash.exe), found through PATH
        # when Git is installed per-user: it cannot open a C:/ script path, so the test would
        # fail in a plain CPU build. treat it as not found
        file(TO_CMAKE_PATH "$ENV{SystemRoot}" _list_devices_sysroot)
        string(TOLOWER "${_list_devices_sysroot}/" _list_devices_sysroot)
        string(TOLOWER "${BASH_EXECUTABLE}"        _list_devices_bash_lc)
        string(FIND "${_list_devices_bash_lc}" "${_list_devices_sysroot}" _list_devices_pos)
        if (_list_devices_pos EQUAL 0)
            set(_list_devices_bash "")
            set(_list_devices_why  "${BASH_EXECUTABLE} is the WSL launcher, which cannot open a Windows path (install Git for Windows or pass -DBASH_EXECUTABLE=<path to a MSYS or Git bash>)")
        endif()
    endif()
    # without a usable bash the entry is still registered, DISABLED, so ctest reports it as
    # not run instead of silently running one test fewer
    if (_list_devices_bash)
        set(_list_devices_cmd "${_list_devices_bash}")
    else()
        set(_list_devices_cmd bash)
    endif()
    llama_test_cmd(
        ${_list_devices_cmd}
        NAME test-list-devices-output
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-list-devices-output.sh $<TARGET_FILE:llama-cli>
    )
    # enumeration opens every backend's device, and a wedged one must not hold the queue
    set_tests_properties(test-list-devices-output PROPERTIES TIMEOUT 120)
    if (NOT _list_devices_bash)
        set_tests_properties(test-list-devices-output PROPERTIES DISABLED TRUE)
        message(STATUS "${_list_devices_why}, test-list-devices-output is disabled")
    endif()
endif()

# the KleidiAI COFF patch script rewrites fetched assembly for clang on Windows; this drives
# it over generated fixtures with cmake -P, so it needs no compiler, KleidiAI or hardware
set(_kleidiai_patch_script ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu/kleidiai/kleidiai-patch-coff-asm.cmake)
if (EXISTS ${_kleidiai_patch_script})
    llama_test_cmd(
        ${CMAKE_COMMAND}
        NAME test-kleidiai-coff-patch
        ARGS -DPATCH_SCRIPT=${_kleidiai_patch_script}
             -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-kleidiai-coff-patch
             -P ${CMAKE_CURRENT_SOURCE_DIR}/test-kleidiai-coff-patch.cmake
    )
    set_tests_properties(test-kleidiai-coff-patch PROPERTIES TIMEOUT 60)
endif()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-kimi-k3
        LABEL main
        ARGS -m "${MODEL_DIR}/kimi-k3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-kimi-k3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
llama_build(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

if (GGML_QNN)
    # the HTP is single-client: the tests that can open the QNN device (any model load or
    # backend enumeration) must not overlap EACH OTHER under ctest -j. the test-qnn-* entries
    # need nothing from this block, they are RUN_SERIAL already and so exclude every other test
    get_property(_qnn_all_tests DIRECTORY PROPERTY TESTS)
    set(_qnn_serial_tests
        test-backend-ops test-thread-safety test-model-load-cancel test-autorelease
        test-save-load-state test-state-restore-fragmented test-backend-sampler
        test-opt test-gguf
        test-list-devices-output test-llama-archs test-generate-models)
    foreach (_test IN LISTS _qnn_all_tests)
        if (_test MATCHES "^test-tokenizer-[01]-" OR _test MATCHES "^test-recurrent-state-rollback")
            list(APPEND _qnn_serial_tests ${_test})
        endif()
    endforeach()
    foreach (_test IN LISTS _qnn_serial_tests)
        if (TEST ${_test})
            set_tests_properties(${_test} PROPERTIES RUN_SERIAL TRUE)
        endif()
    endforeach()
endif()
