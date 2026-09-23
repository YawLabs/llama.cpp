#pragma once

// internal interface between the ggml-backend glue (ggml-qnn.cpp) and the
// QNN (Qualcomm AI Engine Direct) runtime, loaded dynamically from QnnHtp

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include "qnn-mem.h"

#include <QnnInterface.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    // matmul graphs are built with the batch dim N padded up to a power-of-two bucket
    // (GGML_QNN_NPAD is the floor, default 512), for both the static-baked and the dynamic
    // variant: one graph serves every N in the bucket, so a weight bakes once and a LoRA or
    // pooling matmul gets one graph per bucket instead of one per exact N. the padded tail
    // rows are never copied back. 0 means the exact ggml shape (elementwise graphs).
    //
    // the padded-IO size law: a static-bake graph whose padded input (K * n_pad * 4) or
    // output (M * n_pad * 4) reaches about 1 MiB hangs at execute on QAIRT 2.45 - measured
    // 2026-09-16, a 4096 x 64 fp32 output of exactly 1 MiB timed out while 655 KB executed
    // fine. ggml_qnn_mul_mat_policy rejects max(in, out) >= GGML_QNN_IO_MAX_KB (default 1024,
    // strict) before graphCreate, as a policy verdict, never a denylist entry
    uint32_t n_pad = 0;
    // on-device bytes to charge against the static budget once finalize succeeds
    size_t pending_static_bytes = 0;
    // build declined by policy (budget full or clamped, gated path, IO cap), not a shape the
    // HTP rejected
    bool policy_reject = false;
    // build skipped because the shape was denylisted (this run, or from the file)
    bool denylisted = false;
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
    // a constant weight is copied into its IO buffer once: the src address it holds, and the
    // content fingerprint taken then, so a reused address after a model unload is not mistaken
    // for the same weight
    const void * weight_cached_ptr  = nullptr;
    uint64_t     weight_cached_hash = 0;
    // backing storage for the dimension arrays referenced by the tensors
    std::vector<std::vector<uint32_t>> dims;
    bool warned = false;
};

// locking: backend instances share one session. ggml_qnn_session_mutex in ggml-qnn.cpp
// guards the session's existence and refcount; every other field below is guarded by
// `mutex`, taken by ggml_qnn_supports_node and ggml_qnn_compute_node, except `degraded`,
// which the release path reads lock-free at refs == 0 (no other user of the session exists
// then) and is atomic for that reason
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

    // runs QNN calls that can hang (finalize/execute) with a timeout: GGML_QNN_BUILD_TIMEOUT_MS
    // for finalize, GGML_QNN_TIMEOUT_MS for every execute, the validation one included.
    // created on first use, joined on clean session free, detached and abandoned after a timeout
    ggml_qnn_call_worker * worker = nullptr;

    // guards the graph cache and the bind+execute sequence, backend instances share one session
    std::mutex mutex;

    // a failed execute can poison the shared HTP context, so once it happens the whole session
    // stops claiming ops and everything falls back to the CPU. set only via ggml_qnn_degrade
    std::atomic<bool> degraded{false};
    // the degrade was "NPU too slow" and nothing worse since: the device still executes
    // correctly, so supports_op claims nothing more but nodes already placed on graphs that
    // built keep running instead of failing their batch. guarded by `mutex`
    bool slow_only = false;
    // a validation execute completed under GGML_QNN_SLOW_EXEC_MS in this session, i.e. the
    // device was seen at normal speed. only then is a later validation-execute timeout a
    // verdict on the shape that may go to the denylist file, see ggml_qnn_denylist_add
    bool healthy_seen = false;

    // static-weight memory budget (bytes): only pin weights on the NPU up to this, the rest stay
    // on the CPU. 0 means unlimited. GGML_QNN_STATIC_BUDGET_MB sets it, default 1024 as a
    // margin (weight mapping failed at about 1170 MiB once, on a loaded machine)
    size_t static_budget = 0;
    size_t static_bytes  = 0;

    // shape keys whose static graph finalized and validated in this session. a later build
    // failure on one of them is device weight memory running out, not a verdict on the shape:
    // the budget is clamped to what is committed (budget_clamped) so no further weight is
    // baked, the shape stays off the denylist and the session keeps its working graphs
    std::unordered_set<std::string> proven_static;
    bool budget_clamped = false;

    // fastrpc shared memory: the init-time self-test verdict gates every graph's IO setup,
    // and the host-buffer fallback is reported once per session
    bool shared_mem_ok       = false;
    bool shm_fallback_warned = false;

    // the GGML_QNN_AOT_TEST round-trip runs on the first finalized graph of the session
    bool aot_tested = false;

    std::unordered_map<std::string, ggml_qnn_graph> graphs;
};

// load QnnHtp, resolve the interface and create backend/device/context
// returns nullptr if the library or an HTP device is not available
ggml_qnn_session * ggml_qnn_session_init(void);

void ggml_qnn_session_free(ggml_qnn_session * sess);

// flush the GGML_QNN_STATS counters. Called from session free, from every degrade, and from
// the degraded-session release path, which deliberately never frees - without those extra
// calls a run that degraded wrote no counters at all, exactly when they are most worth having
void ggml_qnn_stats_write(void);

// execute one ggml node (MUL_MAT, ADD, MUL) on the NPU
enum ggml_status ggml_qnn_compute_node(ggml_qnn_session * sess, struct ggml_tensor * node);

// whether a QNN graph builds AND test-executes for this node, the HTP can reject shapes at
// finalize time and can wedge at execute time. builds, validates and caches the graph on
// first call, so real inference only ever runs graphs already proven to execute
bool ggml_qnn_supports_node(ggml_qnn_session * sess, const struct ggml_tensor * node);

// whether the node's shape is on the failed-shape denylist (this process, or the
// GGML_QNN_DENYLIST file, loaded on first call). shape only, no session needed: for a
// MUL_MAT both the static and the dynamic variant are consulted, because an unallocated
// probe cannot yet tell which one the node will become
bool ggml_qnn_shape_denylisted(const struct ggml_tensor * node);

// the MUL_MAT weight-type policy, shared by supports_op and the graph policy so the two never
// disagree: F32 and F16 always; a quantized type with a dequantizer when it can be baked
// statically (the default, GGML_QNN_NO_STATIC_WEIGHTS off) or the experimental per-execute
// dequant path is enabled (GGML_QNN_QUANTIZED). both env vars are read once
bool ggml_qnn_mul_mat_type_claimable(enum ggml_type type);

// the MUL_MAT rejections that are pure arithmetic on the shape, no session and no data needed:
// the uint32_t size guard and the GGML_QNN_IO_MAX_KB cap, both at the padded batch
// ggml_qnn_pad_n(N). supports_op calls it before it claims anything and ggml_qnn_mul_mat_policy
// calls the same function, so a claimed matmul is never refused at compute for one of them (a
// claimed node that policy refuses FAILS the graph, it does not fall back).
// N is src1->ne[1], except for llama's weight-placement probe (placement_probe: src0->data is
// NULL and src0->buffer is the zero-size dummy), whose N is fictitious: there the smallest
// bucket ggml_qnn_pad_n(1) is used, because a weight capped even at that batch can never run
// on the NPU and must be refused so that it lands in CPU_REPACK instead of a plain CPU buffer.
// logs like the policy does: DEBUG once per shape, WARN for the first refusal of the process
bool ggml_qnn_mul_mat_arith_reject(const struct ggml_tensor * op, bool placement_probe);

// GGML_QNN_QUANTIZED: the experimental per-execute dequant path for a quantized weight that
// is not baked statically. read once
bool ggml_qnn_quantized_dynamic_ok(void);

// integer env var with a floor: unset or empty returns def; a value that is not entirely a
// number, or below min, warns naming the variable and returns def
static inline long long ggml_qnn_env_ll(const char * name, long long def, long long min) {
    const char * val = getenv(name);
    if (!val || !*val) {
        return def;
    }
    char * end = nullptr;
    errno = 0;
    const long long v = strtoll(val, &end, 10);
    if (end == val || *end != '\0' || errno == ERANGE) {
        GGML_LOG_WARN("ggml-qnn: %s=\"%s\" is not a number, using %lld\n", name, val, def);
        return def;
    }
    if (v < min) {
        GGML_LOG_WARN("ggml-qnn: %s=%lld is below the minimum %lld, using %lld\n", name, v, min, def);
        return def;
    }
    return v;
}

// test-only fault hook: GGML_QNN_FAIL_EXECUTE=<substring> makes the validation execute of
// every graph whose key contains the substring return an error before QNN is called, so the
// degrade + denylist path can be exercised without wedging real hardware.
// GGML_QNN_FAIL_EXECUTE_SKIP=<n> lets the first n matching graphs execute normally, which
// reaches the proven-shape path (budget clamp, no degrade) the same way
//
// test-only fault hook: GGML_QNN_DELAY_EXECUTE=<substring> with GGML_QNN_DELAY_EXECUTE_MS=<ms>
// sleeps that long inside the timed call, right before graphExecute, for an execute of a graph
// whose key contains the substring, validation and compute-time executes alike. the delay
// counts toward GGML_QNN_TIMEOUT_MS and GGML_QNN_SLOW_EXEC_MS, so both verdicts can be reached
// on a healthy device; unlike a real wedge the delayed call then completes, so an abandoned
// worker finishes and frees itself. GGML_QNN_DELAY_EXECUTE_SKIP=<n> runs the first n matching
// executes undelayed; only the one after them is delayed (one-shot), so later executes of the
// same graph run at normal speed
//
// test-only fault hook: GGML_QNN_FAIL_FINALIZE=<substring> makes the finalize of every graph
// whose key contains the substring return QNN_GRAPH_ERROR_GENERAL without calling graphFinalize,
// after the graph was created and its node added: the real finalize-error path runs (the
// unfinalized graph and its STATIC tensor stay in the context, persisted to the denylist for an
// unproven shape, a budget clamp for a proven one). GGML_QNN_FAIL_FINALIZE_SKIP=<n> lets the
// first n matching graphs finalize normally
