# Add the missing GNU-syntax/COFF branch to KleidiAI's assembly macro blocks.
#
# Each .S file picks armasm syntax on _MSC_VER and GNU syntax otherwise. clang targeting
# *-windows-msvc defines _MSC_VER but assembles GNU syntax into COFF, so it fits neither:
# the armasm branch fails on AREA/PROC/GLOBAL, the GNU branch fails on the ELF-only
# .type/.size. The __APPLE__ branch is close but prefixes symbols with an underscore,
# which aarch64 COFF does not use. Add the combination that is missing.
#
# Three edits per file, each keyed on its own sentinel: the armasm guard gains
# "&& !defined(__clang__)" unless it already carries it, the GNU side gains an
# "elif defined(_WIN32)" branch unless one is already there, and a branch written by an
# earlier version of this script, which lacked KAI_ASM_FUNCTION_USE, gains that define.
# The third edit is what upgrades an existing _deps/kleidiai-src: the second one skips it
# because the branch is already there. A file is written back only when its content
# changed, so the script is re-runnable over a patched tree and a re-configure rewrites
# nothing. file(WRITE) emits CRLF on Windows, so a patched file differs from the tarball
# on every line; the assembler does not care.
#
# It runs for every Windows toolchain: inert under armasm64 (cl.exe, clang-cl), active for
# clang's GNU driver and llvm-mingw. It rewrites the fetched tree in place, which includes
# a FETCHCONTENT_SOURCE_DIR_KLEIDIAI override.
#
# Upstream v1.24.0 already ships "#elif defined(_MSC_VER) && defined(__clang__)" in one
# neon file, with its guard already qualified; that file is left alone.

if (NOT KLEIDIAI_SRC)
    message(FATAL_ERROR "KLEIDIAI_SRC not set")
endif()

file(GLOB_RECURSE KAI_ASM_FILES "${KLEIDIAI_SRC}/*.S")
set(KAI_PATCHED 0)

# the injected branch in pieces: KAI_ELSE is the anchor it goes in front of, and
# KAI_BRANCH directly followed by KAI_ELSE is the branch as the earlier version wrote it
set(KAI_BRANCH
"    #elif defined(_WIN32)
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name)
")
set(KAI_BRANCH_USE
"        #define KAI_ASM_FUNCTION_USE(name) name
")
set(KAI_ELSE
"    #else
        #define KAI_ASM_GLOBAL(name) .global name")

foreach (f ${KAI_ASM_FILES})
    file(READ "${f}" orig)
    set(src "${orig}")

    string(FIND "${src}" "#if defined(_MSC_VER) && !defined(__clang__)" have_guard)
    if (have_guard EQUAL -1)
        string(REPLACE
            "#if defined(_MSC_VER)"
            "#if defined(_MSC_VER) && !defined(__clang__)"
            src "${src}")
    endif()

    string(FIND "${src}" "elif defined(_WIN32)" have_branch)
    if (have_branch EQUAL -1)
        string(REPLACE "${KAI_ELSE}" "${KAI_BRANCH}${KAI_BRANCH_USE}${KAI_ELSE}" src "${src}")
    endif()

    # its own sentinel is the search text: the branch with nothing between its
    # KAI_ASM_FUNCTION_END and the #else, which the edit above never produces
    string(REPLACE "${KAI_BRANCH}${KAI_ELSE}" "${KAI_BRANCH}${KAI_BRANCH_USE}${KAI_ELSE}" src "${src}")

    # A file that switches on _MSC_VER in a shape neither edit recognizes will still
    # assemble the wrong syntax under clang. Fatal for library code, a warning for the
    # tests and benchmarks this build does not compile.
    string(FIND "${src}" "_MSC_VER" have_msvc)
    string(FIND "${src}" "#if defined(_MSC_VER) && !defined(__clang__)" have_guard)
    string(FIND "${src}" "elif defined(_WIN32)" have_branch)
    if (NOT have_msvc EQUAL -1 AND have_guard EQUAL -1 AND have_branch EQUAL -1)
        file(RELATIVE_PATH rel "${KLEIDIAI_SRC}" "${f}")
        if (rel MATCHES "^kai/")
            message(FATAL_ERROR "kleidiai: ${rel} switches on _MSC_VER in a shape this script does not recognize")
        else()
            message(WARNING "kleidiai: ${rel} switches on _MSC_VER in a shape this script does not recognize")
        endif()
    endif()

    if (NOT src STREQUAL orig)
        file(WRITE "${f}" "${src}")
        math(EXPR KAI_PATCHED "${KAI_PATCHED} + 1")
    endif()
endforeach()

list(LENGTH KAI_ASM_FILES KAI_TOTAL)
message(STATUS "kleidiai: COFF assembly branch added or completed in ${KAI_PATCHED} of ${KAI_TOTAL} .S file(s)")
