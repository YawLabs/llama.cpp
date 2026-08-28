#pragma once

// internal interface between the ggml-backend glue (ggml-qnn.cpp) and the
// QNN (Qualcomm AI Engine Direct) runtime, loaded dynamically from QnnHtp

#include "ggml.h"
#include "ggml-backend.h"

#include "qnn-mem.h"

#include <QnnInterface.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_qnn_call_worker;

// a finalized single-op QNN graph, cached by op signature
// handle == nullptr marks a signature that failed to build (negative cache)
struct ggml_qnn_graph {
    Qnn_GraphHandle_t handle = nullptr;
    // graph IO tensors, bound once to the graph-owned IO buffers below
    std::vector<Qnn_Tensor_t> inputs;
    Qnn_Tensor_t output = {};
    // weights baked into the graph at finalize, see ggml_qnn_weights_static
    Qnn_Tensor_t weights = {};
    bool weights_static = false;
    // static-weight matmul graphs are built with the batch dim padded up to a
    // bucket so one graph (and one baked weight) serves every batch size, the
    // unused rows are never copied back. 0 means the exact ggml shape
    uint32_t n_pad = 0;
    // on-device bytes to charge against the static budget once finalize succeeds
    size_t pending_static_bytes = 0;
    // build declined by policy (budget full, gated path), not a shape the HTP rejected
    bool policy_reject = false;
    // graph-owned staging for the static bake (weight bytes, or fp16 when the source is
    // quantized), so a wedged finalize never points into model memory. freed after finalize
    std::vector<uint8_t> bake;
    bool weight_quantized = false;
    // graph-owned IO buffers: every execute copies in/out of these instead of
    // binding ggml buffers directly, so an abandoned (timed out) execute can
    // never touch memory the caller has freed. registered fastrpc buffers when
    // GGML_QNN_SHARED_MEM is set and available, plain host memory otherwise
    bool shared_mem = false;
    std::vector<ggml_qnn_mem_buffer> mem_inputs;
    ggml_qnn_mem_buffer              mem_output;
    // page-aligned (hygiene only - the size-threshold hang was traced to IO transfer size,
    // not alignment)
    std::vector<void *> host_inputs;
    void *              host_output = nullptr;
    // a constant weight is copied into its IO buffer once, this is the src it holds
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

    // HTP burst-clock power config, applied lazily on first graph use
    uint32_t power_config_id  = 0;
    bool     has_power_config = false;
    bool     burst_tried      = false;

    // runs QNN calls that can hang (finalize/execute) with a timeout,
    // abandoned if a call never returns. created on first use, leaked on abandon
    ggml_qnn_call_worker * worker = nullptr;

    // guards the graph cache and the bind+execute sequence, backend instances share one session
    std::mutex mutex;

    // a failed execute can poison the shared HTP context, so once it happens the whole session
    // stops claiming ops and everything falls back to the CPU
    bool degraded = false;

    // static-weight memory budget (bytes): only pin weights on the NPU up to this, the rest stay
    // on the CPU. 0 means unlimited. GGML_QNN_STATIC_BUDGET_MB sets it, default 2048
    size_t static_budget = 0;
    size_t static_bytes  = 0;

    std::unordered_map<std::string, ggml_qnn_graph> graphs;
};

// load QnnHtp, resolve the interface and create backend/device/context
// returns nullptr if the library or an HTP device is not available
ggml_qnn_session * ggml_qnn_session_init(void);

void ggml_qnn_session_free(ggml_qnn_session * sess);

// flush the GGML_QNN_STATS counters. Called from session free, and separately from the
// degraded-session path, which deliberately never frees - without that second call a run
// that degraded wrote no counters at all, exactly when they are most worth having
void ggml_qnn_stats_write(void);

// execute one ggml node (MUL_MAT, ADD, MUL) on the NPU
enum ggml_status ggml_qnn_compute_node(ggml_qnn_session * sess, struct ggml_tensor * node);

// whether a QNN graph builds AND test-executes for this node, the HTP can reject shapes at
// finalize time and can wedge at execute time. builds, validates and caches the graph on
// first call, so real inference only ever runs graphs already proven to execute
bool ggml_qnn_supports_node(ggml_qnn_session * sess, const struct ggml_tensor * node);
