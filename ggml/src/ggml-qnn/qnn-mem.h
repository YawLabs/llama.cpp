#pragma once

// fastrpc (libcdsprpc) shared memory registered with a QNN context, so graph IO tensors
// avoid the per-execute re-map/convert cost of unregistered RAW client buffers

#include <QnnInterface.h>

#include <cstddef>

struct ggml_qnn_mem_buffer {
    void *          data   = nullptr;
    int             fd     = -1;
    size_t          size   = 0;
    Qnn_MemHandle_t handle = nullptr;
};

// load libcdsprpc and resolve the rpcmem entry points once, returns false if unavailable
bool ggml_qnn_mem_available(void);

// allocate an rpcmem buffer and register it with the QNN context
bool ggml_qnn_mem_alloc(const QNN_INTERFACE_VER_TYPE * iface, Qnn_ContextHandle_t context,
                        size_t size, ggml_qnn_mem_buffer * out);

void ggml_qnn_mem_free(const QNN_INTERFACE_VER_TYPE * iface, ggml_qnn_mem_buffer * buf);

// alloc + register + deregister + free a small buffer, returns true if the whole path works
// used to probe fastrpc shared memory on the current device
bool ggml_qnn_mem_self_test(const QNN_INTERFACE_VER_TYPE * iface, Qnn_ContextHandle_t context);
