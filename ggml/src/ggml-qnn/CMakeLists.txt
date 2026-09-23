# Qualcomm AI Engine Direct (QNN) backend, targets the Hexagon NPU (HTP)
# on Windows on Snapdragon and other Qualcomm platforms.
#
# Building needs only the QNN SDK headers, the QNN libraries are loaded at
# runtime. Set QNN_SDK_ROOT to a Qualcomm AI Runtime (QAIRT) SDK installation.

if (NOT QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
    set(QNN_SDK_ROOT_FROM_ENV ON)
else()
    set(QNN_SDK_ROOT_FROM_ENV OFF)
endif()
file(TO_CMAKE_PATH "${QNN_SDK_ROOT}" QNN_SDK_ROOT)

# A plain variable, not find_path: a cached result would outlive a re-pointed
# QNN_SDK_ROOT in an existing build directory. Drop the entry older configures left behind.
unset(QNN_SDK_INCLUDE_DIR CACHE)
set(QNN_SDK_INCLUDE_DIR "${QNN_SDK_ROOT}/include/QNN")

if (NOT EXISTS "${QNN_SDK_INCLUDE_DIR}/QnnInterface.h")
    message(FATAL_ERROR "ggml-qnn: QNN SDK headers not found: expected include/QNN/QnnInterface.h under QNN_SDK_ROOT=\"${QNN_SDK_ROOT}\". Set QNN_SDK_ROOT to a Qualcomm AI Runtime (QAIRT) SDK installation: -DQNN_SDK_ROOT=<path>, or the environment variable, which is read only while the cache entry is empty.")
endif()

# The root itself is cached when it came from the environment, so a regenerate from a shell
# without the variable still finds the SDK. Only after the check above: a wrong value must
# not stick. From then on the entry wins over the environment and -DQNN_SDK_ROOT=<path>
# re-points it. FORCE replaces nothing but an empty entry, see the condition at the top.
if (QNN_SDK_ROOT_FROM_ENV)
    set(QNN_SDK_ROOT "${QNN_SDK_ROOT}" CACHE PATH "Qualcomm AI Runtime (QAIRT) SDK root" FORCE)
endif()

message(STATUS "ggml-qnn: using QNN SDK headers at ${QNN_SDK_INCLUDE_DIR}")

set(TARGET_NAME ggml-qnn)

ggml_add_backend_library(${TARGET_NAME}
    ggml-qnn.cpp
    qnn-lib.cpp
    qnn-lib.h
    qnn-mem.cpp
    qnn-mem.h
    qnn-dl.h
    ../../include/ggml-qnn.h)

# SYSTEM: the QAIRT headers are not warning-clean (QnnTypes.h gives an enum a 0xFFFFFFFF
# enumerator, which clang reports as -Wmicrosoft-enum-value), and without it a -Werror build
# fails in every TU that includes them rather than on anything this backend wrote
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${QNN_SDK_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
