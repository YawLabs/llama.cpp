# Qualcomm AI Engine Direct (QNN) backend, targets the Hexagon NPU (HTP)
# on Windows on Snapdragon and other Qualcomm platforms.
#
# Building needs only the QNN SDK headers, the QNN libraries are loaded at
# runtime. Set QNN_SDK_ROOT to a Qualcomm AI Runtime (QAIRT) SDK installation.

if (NOT QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
endif()
file(TO_CMAKE_PATH "${QNN_SDK_ROOT}" QNN_SDK_ROOT)

find_path(QNN_SDK_INCLUDE_DIR NAMES QnnInterface.h HINTS "${QNN_SDK_ROOT}/include/QNN")

if (NOT QNN_SDK_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-qnn: QNN SDK headers not found. Set QNN_SDK_ROOT to a Qualcomm AI Runtime (QAIRT) SDK installation.")
endif()

message(STATUS "ggml-qnn: using QNN SDK headers at ${QNN_SDK_INCLUDE_DIR}")

set(TARGET_NAME ggml-qnn)

ggml_add_backend_library(${TARGET_NAME}
    ggml-qnn.cpp
    qnn-lib.cpp
    qnn-lib.h
    qnn-mem.cpp
    qnn-mem.h
    ../../include/ggml-qnn.h)

target_include_directories(${TARGET_NAME} PRIVATE ${QNN_SDK_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
