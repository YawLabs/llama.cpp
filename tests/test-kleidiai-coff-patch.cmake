# Driver for ggml/src/ggml-cpu/kleidiai/kleidiai-patch-coff-asm.cmake, which adds the
# GNU-syntax/COFF branch to KleidiAI's .S macro blocks for clang targeting Windows.
# Writes small .S fixtures shaped like the v1.24.0 tarball (and like trees an earlier
# version of the script left behind) under WORK_DIR, runs the script on them, and checks
# the result. No compiler and no hardware involved.
#
# Usage: cmake -DPATCH_SCRIPT=<kleidiai-patch-coff-asm.cmake> -DWORK_DIR=<scratch dir> -P test-kleidiai-coff-patch.cmake

cmake_minimum_required(VERSION 3.14)

if (NOT PATCH_SCRIPT OR NOT EXISTS "${PATCH_SCRIPT}")
    message(FATAL_ERROR "PATCH_SCRIPT not set or missing: '${PATCH_SCRIPT}'")
endif()
if (NOT WORK_DIR)
    message(FATAL_ERROR "WORK_DIR not set")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# FAILURES counts the failed checks; functions that call fail() pass it up to their caller
set(FAILURES "")
function(fail msg)
    message(STATUS "FAIL: ${msg}")
    list(APPEND FAILURES "x")
    set(FAILURES "${FAILURES}" PARENT_SCOPE)
endfunction()

# fixture pieces. bracket arguments keep them literal; the leading newline after [=[ is dropped.
# template A mirrors kai/kai_common_sme_asm.S (uses KAI_ASM_FUNCTION_USE), template B a
# ukernel such as kai_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla_asm.S (does not)
set(HEAD [=[
//
// SPDX-License-Identifier: Apache-2.0
//

]=])
set(GUARD_PRISTINE [=[
#if defined(_MSC_VER)
]=])
set(GUARD_PATCHED [=[
#if defined(_MSC_VER) && !defined(__clang__)
]=])
set(A_ARMASM [=[
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP
    #define KAI_ASM_FUNCTION_USE(name) name

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
        #define KAI_ASM_FUNCTION_USE(name) _##name
]=])
set(B_ARMASM [=[
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
]=])
# the branch as the earlier script wrote it, and the define the current one adds to it
set(WIN_BRANCH [=[
    #elif defined(_WIN32)
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name)
]=])
set(WIN_USE [=[
        #define KAI_ASM_FUNCTION_USE(name) name
]=])
set(A_ELSE [=[
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
        #define KAI_ASM_FUNCTION_USE(name) name
    #endif
]=])
set(B_ELSE [=[
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif
]=])
set(TAIL_COMMON [=[

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

]=])
set(A_BODY [=[
    KAI_ASM_CODE(kai_common)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_commit_za)
KAI_ASM_FUNCTION_TYPE(kai_commit_za)
KAI_ASM_FUNCTION_LABEL(kai_commit_za)
#if defined(__ARM_FEATURE_SME)
    bl KAI_ASM_FUNCTION_USE(__arm_tpidr2_save)
#endif
    ret
    KAI_ASM_FUNCTION_END(kai_commit_za)
    KAI_ASM_END
]=])
set(B_BODY [=[
    KAI_ASM_CODE(kernel)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel)
KAI_ASM_FUNCTION_TYPE(kai_kernel)
KAI_ASM_FUNCTION_LABEL(kai_kernel)
    KAI_ASM_INST(0x8540d6a1)  // ld1rw { z1.s }, p5/Z, [x21]
    ret
    KAI_ASM_FUNCTION_END(kai_kernel)
    KAI_ASM_END
]=])

foreach (t A B)
    # pristine: the tarball shape. old: what the earlier script left (no USE define).
    # current: what the script must produce from either
    set(${t}_PRISTINE "${HEAD}${GUARD_PRISTINE}${${t}_ARMASM}${${t}_ELSE}${TAIL_COMMON}${${t}_BODY}")
    set(${t}_OLD      "${HEAD}${GUARD_PATCHED}${${t}_ARMASM}${WIN_BRANCH}${${t}_ELSE}${TAIL_COMMON}${${t}_BODY}")
    set(${t}_CURRENT  "${HEAD}${GUARD_PATCHED}${${t}_ARMASM}${WIN_BRANCH}${WIN_USE}${${t}_ELSE}${TAIL_COMMON}${${t}_BODY}")
endforeach()

# upstream v1.24.0 already ships a clang-on-Windows branch in one neon file: left alone
set(UPSTREAM_CLANG [=[
#if defined(_MSC_VER) && !defined(__clang__)
#define KAI_ASM_HEADER AREA |.text|, CODE, READONLY, ALIGN=4
#define KAI_ASM_FUNCTION(label) |label|
#elif defined(_MSC_VER) && defined(__clang__)
#define KAI_ASM_HEADER        .text
#define KAI_ASM_FUNCTION(label) label:
#else  // _MSC_VER
#define KAI_ASM_HEADER .text
#define KAI_ASM_FUNCTION(label) label:
#endif  // _MSC_VER
    KAI_ASM_HEADER
]=])
set(NO_MSVC [=[
    .text
    .global kai_plain
kai_plain:
    ret
]=])
# switches on _MSC_VER in a shape neither edit recognizes
set(UNRECOGNIZED [=[
#ifdef _MSC_VER
    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
#else
    #define KAI_ASM_CODE(name) .text
#endif
    KAI_ASM_CODE(odd)
]=])

# write content with exact line endings: file(WRITE) translates \n to \r\n on Windows,
# configure_file's NEWLINE_STYLE does not depend on the host
function(write_fixture path content style)
    get_filename_component(dir "${path}" DIRECTORY)
    file(MAKE_DIRECTORY "${dir}")
    file(WRITE "${WORK_DIR}/fixture.in" "${content}")
    configure_file("${WORK_DIR}/fixture.in" "${path}" @ONLY NEWLINE_STYLE ${style})
endfunction()

function(run_patch tree rc_var out_var)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DKLEIDIAI_SRC=${tree}" -P "${PATCH_SCRIPT}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    set(${rc_var}  "${rc}"         PARENT_SCOPE)
    set(${out_var} "${out}${err}"  PARENT_SCOPE)
endfunction()

# content with CRLF folded to LF, after checking the file uses one line ending throughout:
# a CRLF file with LF lines spliced in, or a CRLF written through a translating stream
# (\r\r\n), is a corrupted rewrite even if the assembler would accept it
function(read_normalized path out_var)
    # the line-ending checks read bytes: plain file(READ) drops every \r on Windows
    file(READ "${path}" hex HEX)
    string(REGEX REPLACE "(..)" "\\1 " bytes "${hex}")
    string(FIND "${bytes}" "0d 0d " rr)
    if (NOT rr EQUAL -1)
        fail("${path}: contains \\r\\r (CRLF written through a translating stream)")
    endif()
    # every token is "hh ", so these patterns only match on byte boundaries
    string(REGEX MATCHALL "0a " lf_all "${bytes}")
    string(REGEX MATCHALL "0d 0a " crlf_all "${bytes}")
    list(LENGTH lf_all n_lf)
    list(LENGTH crlf_all n_crlf)
    if (NOT n_crlf EQUAL 0 AND NOT n_crlf EQUAL n_lf)
        fail("${path}: mixed line endings (${n_crlf} CRLF of ${n_lf} lines)")
    endif()
    file(READ "${path}" raw)
    string(REPLACE "\r\n" "\n" norm "${raw}")
    set(${out_var} "${norm}" PARENT_SCOPE)
    set(FAILURES "${FAILURES}" PARENT_SCOPE)
endfunction()

function(expect_content path expected)
    read_normalized("${path}" got)
    if (NOT got STREQUAL expected)
        fail("${path}: content after patching differs from the expected shape:\n--- got ---\n${got}--- expected ---\n${expected}")
    endif()
    set(FAILURES "${FAILURES}" PARENT_SCOPE)
endfunction()

# --- tree 1: every recognized shape, both line endings ---

set(T1 "${WORK_DIR}/ok")
# a CRLF tree is what the script's own file(WRITE) leaves on a Windows host, and on Windows
# file(READ) folds it back to LF. a non-Windows host never produces one (the tarball is LF)
# and would read the \r literally, so there the CRLF rows run as LF
set(CRLF CRLF)
if (NOT CMAKE_HOST_WIN32)
    set(CRLF UNIX)
endif()
# name -> content, line ending, expected content after patching
set(T1_FILES
    kai/a_pristine_lf.S              A_PRISTINE     UNIX A_CURRENT
    kai/a_pristine_crlf.S            A_PRISTINE     ${CRLF} A_CURRENT
    kai/ukernels/b_pristine_lf.S     B_PRISTINE     UNIX B_CURRENT
    kai/a_old_lf.S                   A_OLD          UNIX A_CURRENT
    kai/a_old_crlf.S                 A_OLD          ${CRLF} A_CURRENT
    kai/ukernels/b_old_crlf.S        B_OLD          ${CRLF} B_CURRENT
    kai/a_current_lf.S               A_CURRENT      UNIX A_CURRENT
    kai/ukernels/b_current_crlf.S    B_CURRENT      ${CRLF} B_CURRENT
    kai/ukernels/upstream_clang.S    UPSTREAM_CLANG ${CRLF} UPSTREAM_CLANG
    kai/no_msvc.S                    NO_MSVC        UNIX NO_MSVC)
# 6: the pristine and old fixtures change; the current, upstream-clang and no-MSVC ones
# are already in their final shape and must not be rewritten
set(T1_EXPECT_PATCHED 6)
set(T1_EXPECT_TOTAL   10)

set(T1_UNCHANGED "")
set(i 0)
list(LENGTH T1_FILES n)
while (i LESS n)
    math(EXPR i1 "${i} + 1")
    math(EXPR i2 "${i} + 2")
    math(EXPR i3 "${i} + 3")
    list(GET T1_FILES ${i}  name)
    list(GET T1_FILES ${i1} content)
    list(GET T1_FILES ${i2} style)
    list(GET T1_FILES ${i3} expected)
    write_fixture("${T1}/${name}" "${${content}}" ${style})
    if (content STREQUAL expected)
        file(SHA256 "${T1}/${name}" h)
        list(APPEND T1_UNCHANGED "${name}" "${h}")
    endif()
    math(EXPR i "${i} + 4")
endwhile()

run_patch("${T1}" rc out)
if (NOT rc EQUAL 0)
    fail("run 1 exited ${rc}:\n${out}")
endif()
if (NOT out MATCHES "in ${T1_EXPECT_PATCHED} of ${T1_EXPECT_TOTAL} \\.S file")
    fail("run 1 should report ${T1_EXPECT_PATCHED} of ${T1_EXPECT_TOTAL} patched:\n${out}")
endif()
if (out MATCHES "CMake Warning")
    fail("run 1 warned on a recognized shape:\n${out}")
endif()

set(i 0)
while (i LESS n)
    math(EXPR i3 "${i} + 3")
    list(GET T1_FILES ${i}  name)
    list(GET T1_FILES ${i3} expected)
    expect_content("${T1}/${name}" "${${expected}}")
    math(EXPR i "${i} + 4")
endwhile()

# already-final files are not rewritten at all, byte for byte
list(LENGTH T1_UNCHANGED nu)
set(i 0)
while (i LESS nu)
    math(EXPR i1 "${i} + 1")
    list(GET T1_UNCHANGED ${i}  name)
    list(GET T1_UNCHANGED ${i1} h0)
    file(SHA256 "${T1}/${name}" h)
    if (NOT h STREQUAL h0)
        fail("${name}: already in its final shape but was rewritten")
    endif()
    math(EXPR i "${i} + 2")
endwhile()

# second run: nothing left to do, nothing rewritten
file(GLOB_RECURSE t1_all "${T1}/*.S")
set(t1_hashes "")
foreach (f ${t1_all})
    file(SHA256 "${f}" h)
    list(APPEND t1_hashes "${h}")
endforeach()
run_patch("${T1}" rc out)
if (NOT rc EQUAL 0)
    fail("run 2 exited ${rc}:\n${out}")
endif()
if (NOT out MATCHES "in 0 of ${T1_EXPECT_TOTAL} \\.S file")
    fail("run 2 should report 0 of ${T1_EXPECT_TOTAL} patched:\n${out}")
endif()
set(k 0)
foreach (f ${t1_all})
    list(GET t1_hashes ${k} h0)
    file(SHA256 "${f}" h)
    if (NOT h STREQUAL h0)
        fail("${f}: changed on the second run")
    endif()
    math(EXPR k "${k} + 1")
endforeach()

# --- tree 2: unrecognized _MSC_VER shape in library code is fatal ---

set(T2 "${WORK_DIR}/fatal")
write_fixture("${T2}/kai/ukernels/odd.S" "${UNRECOGNIZED}" UNIX)
run_patch("${T2}" rc out)
if (rc EQUAL 0)
    fail("an unrecognized _MSC_VER shape under kai/ must fail the script, it exited 0:\n${out}")
elseif (NOT out MATCHES "kai/ukernels/odd\\.S switches on _MSC_VER")
    fail("the failure does not name kai/ukernels/odd.S:\n${out}")
endif()

# --- tree 3: the same shape outside kai/ (tests, benchmarks) only warns ---

set(T3 "${WORK_DIR}/warn")
write_fixture("${T3}/test/common/odd.S" "${UNRECOGNIZED}" UNIX)
file(SHA256 "${T3}/test/common/odd.S" h0)
run_patch("${T3}" rc out)
if (NOT rc EQUAL 0)
    fail("an unrecognized _MSC_VER shape outside kai/ must only warn, it exited ${rc}:\n${out}")
endif()
if (NOT out MATCHES "CMake Warning" OR NOT out MATCHES "test/common/odd\\.S switches on _MSC_VER")
    fail("no warning naming test/common/odd.S:\n${out}")
endif()
if (NOT out MATCHES "in 0 of 1 \\.S file")
    fail("the warned file should not count as patched:\n${out}")
endif()
file(SHA256 "${T3}/test/common/odd.S" h)
if (NOT h STREQUAL h0)
    fail("test/common/odd.S was rewritten")
endif()

list(LENGTH FAILURES nf)
if (nf GREATER 0)
    message(FATAL_ERROR "test-kleidiai-coff-patch: ${nf} check(s) failed")
endif()
message(STATUS "test-kleidiai-coff-patch: all checks passed")
