#pragma once

// internal interface between the ggml-backend glue (ggml-qnn.cpp) and the
// QNN (Qualcomm AI Engine Direct) runtime, loaded dynamically from QnnHtp

#include "ggml.h"
#include "ggml-backend.h"

#include "qnn-mem.h"

#include <QnnInterface.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// a finalized single-op QNN graph, cached by op signature
// handle == nullptr marks a signature that failed to build (negative cache)
struct ggml_qnn_graph {
    Qnn_GraphHandle_t handle = nullptr;
    // tensors bound to ggml data on every execute
    std::vector<Qnn_Tensor_t> inputs;
    Qnn_Tensor_t output = {};
    // weights baked into the graph at finalize, see ggml_qnn_weights_static
    Qnn_Tensor_t weights = {};
    bool weights_static = false;
    // quantized weights are dequantized to fp16 on the host into these scratch buffers,
    // then bound to the fp16 weight input on every execute
    bool weight_quantized = false;
    std::vector<float>       dequant_f32;
    std::vector<ggml_fp16_t> dequant_f16;
    // registered fastrpc shared memory for zero-remap graph IO (GGML_QNN_SHARED_MEM):
    // one buffer per input and one for the output, bound as MEMHANDLE instead of RAW clientBuf
    bool shared_mem = false;
    std::vector<ggml_qnn_mem_buffer> mem_inputs;
    ggml_qnn_mem_buffer              mem_output;
    // a constant weight is copied into its shared buffer once, this is the src it holds
    const void * weight_cached_ptr = nullptr;
    // backing storage for the dimension arrays referenced by the tensors
    std::vector<std::vector<uint32_t>> dims;
    bool warned = false;
};

struct ggml_qnn_session {
    void * lib = nullptr;
    QNN_INTERFACE_VER_TYPE iface = {};

    Qnn_LogHandle_t     log_handle     = nullptr;
    Qnn_BackendHandle_t backend_handle = nullptr;
    Qnn_DeviceHandle_t  device_handle  = nullptr;
    Qnn_ContextHandle_t context_handle = nullptr;

    // HTP burst-clock power config, held for the session lifetime
    uint32_t power_config_id  = 0;
    bool     has_power_config = false;

    // guards the graph cache and the bind+execute sequence, backend instances share one session
    std::mutex mutex;

    // a failed execute can poison the shared HTP context, so once it happens the whole session
    // stops claiming ops and everything falls back to the CPU
    bool degraded = false;

    // static-weight memory budget (bytes): only pin weights on the NPU up to this, the rest stay
    // on the CPU. 0 means unlimited. GGML_QNN_STATIC_BUDGET_MB sets it.
    size_t static_budget = 0;
    size_t static_bytes  = 0;

    std::unordered_map<std::string, ggml_qnn_graph> graphs;
};

// load QnnHtp, resolve the interface and create backend/device/context
// returns nullptr if the library or an HTP device is not available
ggml_qnn_session * ggml_qnn_session_init(void);

void ggml_qnn_session_free(ggml_qnn_session * sess);

// execute one ggml node (MUL_MAT, ADD, MUL) on the NPU
enum ggml_status ggml_qnn_compute_node(ggml_qnn_session * sess, struct ggml_tensor * node);

// whether a QNN graph builds for this node, the HTP can reject shapes at finalize time
// builds and caches the graph on first call
bool ggml_qnn_supports_node(ggml_qnn_session * sess, const struct ggml_tensor * node);
