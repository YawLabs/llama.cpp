# Add the missing GNU-syntax/COFF branch to KleidiAI's assembly macro blocks.
#
# Each .S file picks armasm syntax on _MSC_VER and GNU syntax otherwise. clang targeting
# *-windows-msvc defines _MSC_VER but assembles GNU syntax into COFF, so it fits neither:
# the armasm branch fails on AREA/PROC/GLOBAL, the GNU branch fails on the ELF-only
# .type/.size. The __APPLE__ branch is close but prefixes symbols with an underscore,
# which aarch64 COFF does not use. Add the combination that is missing.
#
# Idempotent: a file that already carries the branch is left alone.

if (NOT KLEIDIAI_SRC)
    message(FATAL_ERROR "KLEIDIAI_SRC not set")
endif()

file(GLOB_RECURSE KAI_ASM_FILES "${KLEIDIAI_SRC}/*.S")
set(KAI_PATCHED 0)

foreach (f ${KAI_ASM_FILES})
    file(READ "${f}" src)
    string(FIND "${src}" "elif defined(_WIN32)" have_branch)
    if (have_branch EQUAL -1)
        string(REPLACE
            "#if defined(_MSC_VER)"
            "#if defined(_MSC_VER) && !defined(__clang__)"
            src "${src}")
        string(REPLACE
"    #else
        #define KAI_ASM_GLOBAL(name) .global name"
"    #elif defined(_WIN32)
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name"
            src "${src}")
        file(WRITE "${f}" "${src}")
        math(EXPR KAI_PATCHED "${KAI_PATCHED} + 1")
    endif()
endforeach()

list(LENGTH KAI_ASM_FILES KAI_TOTAL)
message(STATUS "kleidiai: COFF assembly branch added to ${KAI_PATCHED} of ${KAI_TOTAL} .S file(s)")
