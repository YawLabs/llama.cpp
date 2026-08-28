#include "qnn-lib.h"
#include "qnn-mem.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <QnnTypes.h>
#include <QnnCommon.h>
#include <QnnLog.h>
#include <QnnGraph.h>
#include <QnnTensor.h>
#include <QnnOpDef.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    include <malloc.h>
#else
#    include <dlfcn.h>
#endif

//
// counters
//

// GGML_QNN_STATS names a file that receives these counters when the session is freed.
// They exist because several ctest variants differ from their parent only by an env var
// and could not observe whether that env var changed anything: test-qnn-rebake asserted a
// weight was "claimed without a re-bake" while a double bake passed, and -shm and -noopt
// could not tell an engaged feature from a silent fallback. A file keeps the QNN internals
// out of the public header and matches how GGML_QNN_DENYLIST is already wired.
static std::atomic<uint64_t> ggml_qnn_stat_graphs_created{0};
static std::atomic<uint64_t> ggml_qnn_stat_graph_cache_hits{0};
static std::atomic<uint64_t> ggml_qnn_stat_weights_baked{0};
static std::atomic<uint64_t> ggml_qnn_stat_io_shared{0};
static std::atomic<uint64_t> ggml_qnn_stat_io_host{0};
static std::atomic<uint64_t> ggml_qnn_stat_io_shm_fallback{0};
static std::atomic<uint64_t> ggml_qnn_stat_graphs_noopt{0};
static std::atomic<uint64_t> ggml_qnn_stat_pad_n_last{0};

// written on session teardown; counters are process-global and monotonic, so with a
// refcounted session the last write is the cumulative total for the run
void ggml_qnn_stats_write(void) {
    const char * path = getenv("GGML_QNN_STATS");
    if (!path || !*path) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        GGML_LOG_ERROR("ggml-qnn: cannot write stats to %s\n", path);
        return;
    }
    fprintf(f, "graphs_created %" PRIu64 "\n", ggml_qnn_stat_graphs_created.load());
    fprintf(f, "graph_cache_hits %" PRIu64 "\n", ggml_qnn_stat_graph_cache_hits.load());
    fprintf(f, "weights_baked %" PRIu64 "\n", ggml_qnn_stat_weights_baked.load());
    fprintf(f, "io_shared %" PRIu64 "\n", ggml_qnn_stat_io_shared.load());
    fprintf(f, "io_host %" PRIu64 "\n", ggml_qnn_stat_io_host.load());
    fprintf(f, "io_shm_fallback %" PRIu64 "\n", ggml_qnn_stat_io_shm_fallback.load());
    fprintf(f, "graphs_noopt %" PRIu64 "\n", ggml_qnn_stat_graphs_noopt.load());
    fprintf(f, "pad_n_last %" PRIu64 "\n", ggml_qnn_stat_pad_n_last.load());
    fclose(f);
}

// page-aligned host buffers for graph IO. kept as allocation hygiene: alignment was ruled
// out as the cause of the size-threshold execute hang (that follows padded IO transfer size
// alone - keep max(input, output) under ~1 MB via the pad bucket, see GGML_QNN_NPAD)
static void * ggml_qnn_host_alloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 4096);
#else
    void * p = nullptr;
    return posix_memalign(&p, 4096, size) == 0 ? p : nullptr;
#endif
}

static void ggml_qnn_host_free(void * p) {
    if (!p) {
        return;
    }
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

//
// dynamic loading
//

static void * ggml_qnn_dl_open(const char * path) {
#ifdef _WIN32
    if (strchr(path, '\\') || strchr(path, '/')) {
        return (void *) LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void * ggml_qnn_dl_sym(void * lib, const char * name) {
#ifdef _WIN32
    return (void *) GetProcAddress((HMODULE) lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void ggml_qnn_dl_close(void * lib) {
#ifdef _WIN32
    FreeLibrary((HMODULE) lib);
#else
    dlclose(lib);
#endif
}

static void * ggml_qnn_load_htp_lib(void) {
#ifdef _WIN32
    const char * lib_name = "QnnHtp.dll";
#else
    const char * lib_name = "libQnnHtp.so";
#endif
    void * lib = ggml_qnn_dl_open(lib_name);
    if (lib) {
        return lib;
    }
#ifdef _WIN32
    // fall back to the SDK installation, dependent DLLs resolve from the same directory
    const char * sdk_root = getenv("QNN_SDK_ROOT");
    if (sdk_root) {
        const char * lib_dirs[] = { "aarch64-windows-msvc", "arm64x-windows-msvc" };
        for (const char * dir : lib_dirs) {
            char path[1024];
            snprintf(path, sizeof(path), "%s\\lib\\%s\\QnnHtp.dll", sdk_root, dir);
            lib = ggml_qnn_dl_open(path);
            if (lib) {
                return lib;
            }
        }
    }
#endif
    return nullptr;
}

//
// logging
//

static void ggml_qnn_log_callback(const char * fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, args);
    if (level == QNN_LOG_LEVEL_ERROR) {
        GGML_LOG_ERROR("ggml-qnn: %s\n", buf);
    } else {
        GGML_LOG_DEBUG("ggml-qnn: %s\n", buf);
    }
    GGML_UNUSED(timestamp);
}

// device memory committed to baked static weights, process-global: a kept degraded session
// still has its weights mapped on the NPU, so a successor session must not re-spend that budget
static std::atomic<size_t> ggml_qnn_static_committed{0};

//
// timed calls
//

// graphFinalize / graphExecute can hang the HTP so the call never returns. they run on one
// persistent worker thread with a timeout; on timeout the worker is abandoned (it stays
// stuck in the driver) and the session degrades to the CPU. the session is then deliberately
// leaked, so the stuck call can only ever touch memory that is still alive. the worker
// deletes its own state when it exits (quit or found itself abandoned)
struct ggml_qnn_call_worker {
    std::mutex              m;
    std::condition_variable cv;
    std::function<Qnn_ErrorHandle_t()> job;
    bool has_job  = false;
    bool job_done = false;
    bool lost     = false;
    bool quit     = false;
    Qnn_ErrorHandle_t result = QNN_SUCCESS;
};

static void ggml_qnn_worker_loop(ggml_qnn_call_worker * w) {
    std::unique_lock<std::mutex> lock(w->m);
    for (;;) {
        w->cv.wait(lock, [w] { return w->has_job || w->quit; });
        if (w->quit) {
            break;
        }
        std::function<Qnn_ErrorHandle_t()> job = std::move(w->job);
        w->has_job = false;
        lock.unlock();
        Qnn_ErrorHandle_t r = job();
        lock.lock();
        w->result   = r;
        w->job_done = true;
        w->cv.notify_all();
        if (w->lost) {
            break;
        }
    }
    lock.unlock();
    delete w;
}

// returns true if the call completed, writing its result to *out. false means timeout
static bool ggml_qnn_call_timed(ggml_qnn_session * sess, std::function<Qnn_ErrorHandle_t()> call, Qnn_ErrorHandle_t * out) {
    static const int timeout_ms = getenv("GGML_QNN_TIMEOUT_MS") ? atoi(getenv("GGML_QNN_TIMEOUT_MS")) : 15000;

    if (!sess->worker) {
        sess->worker = new ggml_qnn_call_worker();
        std::thread(ggml_qnn_worker_loop, sess->worker).detach();
    }
    ggml_qnn_call_worker * w = sess->worker;

    std::unique_lock<std::mutex> lock(w->m);
    w->job      = std::move(call);
    w->has_job  = true;
    w->job_done = false;
    w->cv.notify_all();
    if (!w->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [w] { return w->job_done; })) {
        w->lost      = true;
        sess->worker = nullptr; // abandoned, leaked on purpose
        return false;
    }
    *out = w->result;
    return true;
}

//
// power
//

// lock the HTP to TURBO clocks with DCVS disabled, otherwise it idles at low clocks and the
// matmul runs an order of magnitude slower than the hardware is capable of
static void ggml_qnn_set_burst_mode(ggml_qnn_session * sess) {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (sess->iface.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
        GGML_LOG_DEBUG("ggml-qnn: could not get device infrastructure, HTP stays at default clocks\n");
        return;
    }

    QnnHtpDevice_Infrastructure_t * htp = (QnnHtpDevice_Infrastructure_t *) infra;
    QnnHtpDevice_PerfInfrastructure_t & perf = htp->perfInfra;
    if (!perf.createPowerConfigId || !perf.setPowerConfig) {
        return;
    }

    uint32_t power_config_id = 0;
    if (perf.createPowerConfigId(/*deviceId=*/0, /*coreId=*/0, &power_config_id) != QNN_SUCCESS) {
        GGML_LOG_DEBUG("ggml-qnn: createPowerConfigId failed\n");
        return;
    }
    sess->power_config_id = power_config_id;
    sess->has_power_config = true;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = {};
    dcvs.option                              = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvs.dcvsV3Config.contextId              = power_config_id;
    dcvs.dcvsV3Config.setDcvsEnable          = 1;
    dcvs.dcvsV3Config.dcvsEnable             = 0; // no dynamic scaling, hold the target corner
    dcvs.dcvsV3Config.powerMode              = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepLatency        = 1;
    dcvs.dcvsV3Config.sleepLatency           = 40;
    dcvs.dcvsV3Config.setBusParams           = 1;
    dcvs.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.setCoreParams          = 1;
    dcvs.dcvsV3Config.coreVoltageCornerMin   = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.coreVoltageCornerTarget= DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.coreVoltageCornerMax   = DCVS_VOLTAGE_VCORNER_TURBO;

    const QnnHtpPerfInfrastructure_PowerConfig_t * cfgs[] = { &dcvs, nullptr };
    if (perf.setPowerConfig(power_config_id, cfgs) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: setPowerConfig(TURBO) failed\n");
        return;
    }
    GGML_LOG_INFO("ggml-qnn: HTP locked to TURBO clocks (burst)\n");
}

//
// session
//

typedef Qnn_ErrorHandle_t (*ggml_qnn_get_providers_fn_t)(const QnnInterface_t *** providers, uint32_t * n_providers);

ggml_qnn_session * ggml_qnn_session_init(void) {
    if (getenv("GGML_QNN_DISABLE")) {
        GGML_LOG_INFO("ggml-qnn: disabled by GGML_QNN_DISABLE\n");
        return nullptr;
    }
    void * lib = ggml_qnn_load_htp_lib();
    if (!lib) {
        GGML_LOG_DEBUG("ggml-qnn: QnnHtp library not found, backend unavailable\n");
        return nullptr;
    }

    ggml_qnn_get_providers_fn_t get_providers =
        (ggml_qnn_get_providers_fn_t) ggml_qnn_dl_sym(lib, "QnnInterface_getProviders");
    if (!get_providers) {
        GGML_LOG_ERROR("ggml-qnn: QnnInterface_getProviders not found in QnnHtp library\n");
        ggml_qnn_dl_close(lib);
        return nullptr;
    }

    const QnnInterface_t ** providers = nullptr;
    uint32_t n_providers = 0;
    if (get_providers(&providers, &n_providers) != QNN_SUCCESS || !providers || n_providers == 0) {
        GGML_LOG_ERROR("ggml-qnn: failed to query QNN interface providers\n");
        ggml_qnn_dl_close(lib);
        return nullptr;
    }

    const QNN_INTERFACE_VER_TYPE * iface = nullptr;
    for (uint32_t i = 0; i < n_providers; i++) {
        const Qnn_Version_t & v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
            iface = &providers[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!iface) {
        GGML_LOG_ERROR("ggml-qnn: no QNN interface provider with API version %d found\n", QNN_API_VERSION_MAJOR);
        ggml_qnn_dl_close(lib);
        return nullptr;
    }

    ggml_qnn_session * sess = new ggml_qnn_session();
    sess->lib   = lib;
    sess->iface = *iface;

    QnnLog_Level_t log_level = getenv("GGML_QNN_DEBUG") ? QNN_LOG_LEVEL_DEBUG : QNN_LOG_LEVEL_ERROR;
    if (sess->iface.logCreate(ggml_qnn_log_callback, log_level, &sess->log_handle) != QNN_SUCCESS) {
        GGML_LOG_DEBUG("ggml-qnn: QnnLog_create failed, continuing without logging\n");
        sess->log_handle = nullptr;
    }

    if (sess->iface.backendCreate(sess->log_handle, nullptr, &sess->backend_handle) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: QnnBackend_create failed\n");
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    if (sess->iface.deviceCreate(sess->log_handle, nullptr, &sess->device_handle) != QNN_SUCCESS) {
        GGML_LOG_INFO("ggml-qnn: no HTP device available\n");
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    if (sess->iface.contextCreate(sess->backend_handle, sess->device_handle, nullptr, &sess->context_handle) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: QnnContext_create failed\n");
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    GGML_LOG_INFO("ggml-qnn: initialized Hexagon NPU (HTP)\n");

    // unlimited static pinning exhausts NPU mapped memory on full models, which can poison
    // the context, so cap it by default
    const char * mb = getenv("GGML_QNN_STATIC_BUDGET_MB");
    sess->static_budget = (mb ? (size_t) atoll(mb) : 2048) * 1024 * 1024;
    if (sess->static_budget) {
        GGML_LOG_INFO("ggml-qnn: static-weight budget %zu MB\n", sess->static_budget / (1024 * 1024));
    }

    // probe fastrpc shared memory, used for registered graph IO buffers
    if (getenv("GGML_QNN_SHARED_MEM")) {
        if (!ggml_qnn_mem_self_test(&sess->iface, sess->context_handle)) {
            GGML_LOG_INFO("ggml-qnn: fastrpc shared memory not available on this device\n");
        }
    }

    return sess;
}

void ggml_qnn_session_free(ggml_qnn_session * sess) {
    ggml_qnn_stats_write();
    if (!sess) {
        return;
    }
    if (sess->worker) {
        // tell the idle worker to exit, it deletes its own state
        std::lock_guard<std::mutex> lock(sess->worker->m);
        sess->worker->quit = true;
        sess->worker->cv.notify_all();
        sess->worker = nullptr;
    }
    // deregister + free shared buffers while the context is still alive
    for (auto & kv : sess->graphs) {
        ggml_qnn_graph & g = kv.second;
        for (auto & b : g.mem_inputs) {
            ggml_qnn_mem_free(&sess->iface, &b);
        }
        ggml_qnn_mem_free(&sess->iface, &g.mem_output);
        for (void * p : g.host_inputs) {
            ggml_qnn_host_free(p);
        }
        g.host_inputs.clear();
        ggml_qnn_host_free(g.host_output);
        g.host_output = nullptr;
    }
    if (sess->context_handle) {
        sess->iface.contextFree(sess->context_handle, nullptr);
        // the freed context returns its baked-weight device memory
        ggml_qnn_static_committed -= sess->static_bytes;
    }
    if (sess->has_power_config) {
        QnnDevice_Infrastructure_t infra = nullptr;
        if (sess->iface.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
            QnnHtpDevice_Infrastructure_t * htp = (QnnHtpDevice_Infrastructure_t *) infra;
            if (htp->perfInfra.destroyPowerConfigId) {
                htp->perfInfra.destroyPowerConfigId(sess->power_config_id);
            }
        }
    }
    if (sess->device_handle) {
        sess->iface.deviceFree(sess->device_handle);
    }
    if (sess->backend_handle) {
        sess->iface.backendFree(sess->backend_handle);
    }
    if (sess->log_handle) {
        sess->iface.logFree(sess->log_handle);
    }
    if (sess->lib) {
        ggml_qnn_dl_close(sess->lib);
    }
    delete sess;
}

//
// graph building
//

// the on-device weight dtype: F32 stays F32, F16 and every quantized type land as F16
static Qnn_DataType_t ggml_qnn_weight_dtype(enum ggml_type type) {
    return type == GGML_TYPE_F32 ? QNN_DATATYPE_FLOAT_32 : QNN_DATATYPE_FLOAT_16;
}

// weights in a model buffer are baked into the graph as a static tensor: QNN converts them
// to the HTP-native layout once at finalize instead of on every execute (up to 64x faster),
// and a quantized source is dequantized to fp16 once at bake. on by default, subject to the
// static budget; GGML_QNN_NO_STATIC_WEIGHTS disables
static bool ggml_qnn_weights_static(const ggml_tensor * node) {
    static const bool disabled = getenv("GGML_QNN_NO_STATIC_WEIGHTS") != nullptr;
    return !disabled && node->op == GGML_OP_MUL_MAT && node->src[0]->data &&
           node->src[0]->buffer && node->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           (!ggml_is_quantized(node->src[0]->type) || ggml_get_type_traits(node->src[0]->type)->to_float != NULL);
}

// on-device bytes of the baked static weight (always fp16 unless the source is fp32)
static size_t ggml_qnn_static_bytes(const ggml_tensor * w) {
    const size_t elem = w->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    return (size_t) ggml_nelements(w) * elem;
}

// static-weight matmul graphs are built with the batch dim padded up to a bucket, so llama
// probing N=512 and then decoding N=60 reuses one graph and bakes each weight once
static uint32_t ggml_qnn_pad_n(uint32_t n) {
    static const uint32_t floor_n = getenv("GGML_QNN_NPAD") ? (uint32_t) atoi(getenv("GGML_QNN_NPAD")) : 512;
    uint32_t p = floor_n ? floor_n : 1;
    while (p < n) {
        p <<= 1;
    }
    ggml_qnn_stat_pad_n_last.store(p);
    return p;
}

// dequantize a contiguous 2D weight to fp16 row by row, so no full fp32 copy is ever held
static void ggml_qnn_dequant_f16(const ggml_tensor * w, ggml_fp16_t * dst) {
    const auto * traits = ggml_get_type_traits(w->type);
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    std::vector<float> row((size_t) K);
    for (int64_t r = 0; r < M; r++) {
        traits->to_float((const char *) w->data + r * w->nb[1], row.data(), K);
        ggml_fp32_to_fp16_row(row.data(), dst + (size_t) r * K, K);
    }
}

static bool ggml_qnn_tensor_init(ggml_qnn_session * sess, ggml_qnn_graph & g, Qnn_Tensor_t & t,
                                 const char * name, Qnn_TensorType_t type, std::vector<uint32_t> dims,
                                 Qnn_DataType_t data_type, void * data = nullptr, uint32_t data_size = 0) {
    g.dims.push_back(std::move(dims));

    t = {};
    t.version         = QNN_TENSOR_VERSION_1;
    t.v1.id           = 0;
    t.v1.name         = name;
    t.v1.type         = type;
    t.v1.dataFormat   = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType     = data_type;
    t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v1.rank         = (uint32_t) g.dims.back().size();
    t.v1.dimensions   = g.dims.back().data();
    t.v1.memType      = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf    = { data, data_size };

    if (sess->iface.tensorCreateGraphTensor(g.handle, &t) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to create graph tensor %s\n", name);
        return false;
    }
    return true;
}

static bool ggml_qnn_add_node(ggml_qnn_session * sess, ggml_qnn_graph & g,
                              const char * type_name, Qnn_Param_t * params, uint32_t n_params,
                              Qnn_Tensor_t * op_inputs, uint32_t n_op_inputs) {
    Qnn_OpConfig_t op = {};
    op.version           = QNN_OPCONFIG_VERSION_1;
    op.v1.name           = "node";
    op.v1.packageName    = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName       = type_name;
    op.v1.numOfParams    = n_params;
    op.v1.params         = params;
    op.v1.numOfInputs    = n_op_inputs;
    op.v1.inputTensors   = op_inputs;
    op.v1.numOfOutputs   = 1;
    op.v1.outputTensors  = &g.output;

    if (sess->iface.graphAddNode(g.handle, op) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to add %s node\n", type_name);
        return false;
    }
    return true;
}

// dimensions of a ggml tensor in row-major order, as QNN expects them
static std::vector<uint32_t> ggml_qnn_dims(const ggml_tensor * t) {
    const int rank = ggml_n_dims(t);
    std::vector<uint32_t> dims(rank);
    for (int i = 0; i < rank; i++) {
        dims[i] = (uint32_t) t->ne[rank - 1 - i];
    }
    return dims;
}

// policy checks that need no QNN graph. they must run BEFORE graphCreate: a rejected shape
// must not leave an unfinalized graph in the shared context, because its deferred prepare
// hangs the next graph's execute. fills the g fields build_mul_mat relies on
static bool ggml_qnn_mul_mat_policy(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t M = (uint32_t) src0->ne[1];
    const uint32_t N = (uint32_t) src1->ne[1];

    g.weights_static   = ggml_qnn_weights_static(node);
    g.weight_quantized = ggml_is_quantized(src0->type);
    g.n_pad            = g.weights_static ? ggml_qnn_pad_n(N) : N;

    const uint32_t Nb = g.n_pad;
    if ((uint64_t) K * Nb * sizeof(float) > UINT32_MAX || (uint64_t) M * Nb * sizeof(float) > UINT32_MAX) {
        g.policy_reject = true; // local arithmetic, not an HTP verdict - keep it off the denylist
        return false;
    }

    // static weights fit within the NPU memory budget, past it a weight stays on the CPU.
    // the budget is charged only after finalize succeeds
    if (g.weights_static && sess->static_budget) {
        const size_t need = ggml_qnn_static_bytes(src0);
        if (ggml_qnn_static_committed.load() + need > sess->static_budget) {
            g.policy_reject = true;
            return false;
        }
        g.pending_static_bytes = need;
    }
    // a quantized weight that is not baked statically has no correct NPU path (the per-execute
    // dequant path is experimental and gated), so keep it on the CPU
    if (g.weight_quantized && !g.weights_static && !getenv("GGML_QNN_QUANTIZED")) {
        g.policy_reject = true;
        return false;
    }
    return true;
}

static bool ggml_qnn_build_mul_mat(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t M = (uint32_t) src0->ne[1];
    const uint32_t Nb = g.n_pad; // filled by ggml_qnn_mul_mat_policy before graphCreate

    // ggml: dst(NxM row-major) = src1(NxK) * src0(MxK)^T
    g.inputs.resize(g.weights_static ? 1 : 2);
    if (!ggml_qnn_tensor_init(sess, g, g.inputs[0], "in0", QNN_TENSOR_TYPE_APP_WRITE, {Nb, K}, QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    Qnn_Tensor_t & w = g.weights_static ? g.weights : g.inputs[1];
    if (g.weights_static) {
        // bake the weight once from graph-owned staging, dequantizing a quantized source to
        // fp16 first. QNN owns the converted copy from finalize on and the staging is freed;
        // a wedged finalize leaks the staging instead of pointing into model memory
        if (g.weight_quantized) {
            g.bake.resize((size_t) ggml_nelements(src0) * sizeof(ggml_fp16_t));
            ggml_qnn_dequant_f16(src0, (ggml_fp16_t *) g.bake.data());
        } else {
            g.bake.assign((const uint8_t *) src0->data, (const uint8_t *) src0->data + ggml_nbytes(src0));
        }
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_STATIC, {M, K}, ggml_qnn_weight_dtype(src0->type),
                                  g.bake.data(), (uint32_t) g.bake.size())) {
            return false;
        }
        ggml_qnn_stat_weights_baked.fetch_add(1);
    } else {
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_APP_WRITE, {M, K}, ggml_qnn_weight_dtype(src0->type))) {
            return false;
        }
    }

    if (!ggml_qnn_tensor_init(sess, g, g.output, "out", QNN_TENSOR_TYPE_APP_READ, {Nb, M}, QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    Qnn_Param_t params[2] = {};
    params[0].paramType             = QNN_PARAMTYPE_SCALAR;
    params[0].name                  = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    params[0].scalarParam.dataType  = QNN_DATATYPE_BOOL_8;
    params[0].scalarParam.bool8Value = 0;
    params[1].paramType             = QNN_PARAMTYPE_SCALAR;
    params[1].name                  = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    params[1].scalarParam.dataType  = QNN_DATATYPE_BOOL_8;
    params[1].scalarParam.bool8Value = 1;

    Qnn_Tensor_t op_inputs[2] = { g.inputs[0], w };
    return ggml_qnn_add_node(sess, g, QNN_OP_MAT_MUL, params, 2, op_inputs, 2);
}

static bool ggml_qnn_build_binary(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    g.n_pad = 0;
    g.inputs.resize(2);
    if (!ggml_qnn_tensor_init(sess, g, g.inputs[0], "in0", QNN_TENSOR_TYPE_APP_WRITE, ggml_qnn_dims(src0), QNN_DATATYPE_FLOAT_32) ||
        !ggml_qnn_tensor_init(sess, g, g.inputs[1], "in1", QNN_TENSOR_TYPE_APP_WRITE, ggml_qnn_dims(src1), QNN_DATATYPE_FLOAT_32) ||
        !ggml_qnn_tensor_init(sess, g, g.output,    "out", QNN_TENSOR_TYPE_APP_READ,  ggml_qnn_dims(node), QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    const char * type_name = node->op == GGML_OP_ADD ? QNN_OP_ELEMENT_WISE_ADD : QNN_OP_ELEMENT_WISE_MULTIPLY;
    return ggml_qnn_add_node(sess, g, type_name, nullptr, 0, g.inputs.data(), (uint32_t) g.inputs.size());
}

// shape-only key: stable across runs, used for the failed-shape denylist
static std::string ggml_qnn_shape_key(const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    // static matmul graphs use the padded batch dim, so every N in a bucket shares one graph.
    // the variant tag matters: static-baked and dynamic-weight graphs are different HTP
    // programs with different finalize outcomes, a denylist entry must only ban the one that failed
    const bool is_static = ggml_qnn_weights_static(node);
    const int64_t ne11 = is_static ? (int64_t) ggml_qnn_pad_n((uint32_t) src1->ne[1]) : src1->ne[1];

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%s_%s_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "%s",
             ggml_op_name(node->op), ggml_type_name(src0->type), ggml_type_name(src1->type),
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
             src1->ne[0], ne11, src1->ne[2], src1->ne[3],
             is_static ? "_s" : "_dyn");
    return buf;
}

static std::string ggml_qnn_graph_key(const ggml_tensor * node) {
    std::string key = ggml_qnn_shape_key(node);
    if (ggml_qnn_weights_static(node)) {
        // one baked-weight graph per weight tensor
        char suffix[32];
        snprintf(suffix, sizeof(suffix), "_w%p", node->src[0]->data);
        key += suffix;
    }
    return key;
}

//
// failed-shape denylist
//

// shapes that failed or wedged the HTP, kept process-global so the knowledge survives the
// session teardown between llama's probe and context phases. GGML_QNN_DENYLIST names a file
// that persists it across runs, so a rerun after a wedge skips the bad shape entirely
static std::mutex                       ggml_qnn_denylist_mutex;
static std::unordered_map<std::string, bool> ggml_qnn_denylist; // value unused

static const char * ggml_qnn_denylist_path(void) {
    static const char * path = getenv("GGML_QNN_DENYLIST");
    return path;
}

static void ggml_qnn_denylist_load_once(void) {
    static bool loaded = false;
    if (loaded) {
        return;
    }
    loaded = true;
    const char * path = ggml_qnn_denylist_path();
    if (!path) {
        return;
    }
    FILE * f = fopen(path, "r");
    if (!f) {
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0]) {
            ggml_qnn_denylist.emplace(line, true);
        }
    }
    fclose(f);
    GGML_LOG_INFO("ggml-qnn: loaded %zu denylisted shapes from %s\n", ggml_qnn_denylist.size(), path);
}

static bool ggml_qnn_denylisted(const std::string & shape_key) {
    std::lock_guard<std::mutex> lock(ggml_qnn_denylist_mutex);
    ggml_qnn_denylist_load_once();
    return ggml_qnn_denylist.count(shape_key) != 0;
}

static void ggml_qnn_denylist_add(const std::string & shape_key) {
    std::lock_guard<std::mutex> lock(ggml_qnn_denylist_mutex);
    if (!ggml_qnn_denylist.emplace(shape_key, true).second) {
        return;
    }
    const char * path = ggml_qnn_denylist_path();
    if (path) {
        FILE * f = fopen(path, "a");
        if (f) {
            fprintf(f, "%s\n", shape_key.c_str());
            fclose(f);
        }
    }
}

// serialize the finalized context to a binary, reload it into a fresh context, and time both,
// to see if a compiled-once/load-fast NPU path is worth building (route 2)
static void ggml_qnn_aot_roundtrip(ggml_qnn_session * sess, const std::string & graph_key, double finalize_ms) {
    Qnn_ContextBinarySize_t size = 0;
    if (sess->iface.contextGetBinarySize(sess->context_handle, &size) != QNN_SUCCESS || size == 0) {
        GGML_LOG_ERROR("ggml-qnn: AOT contextGetBinarySize failed\n");
        return;
    }

    std::vector<uint8_t> buf(size);
    Qnn_ContextBinarySize_t written = 0;
    auto t0 = std::chrono::steady_clock::now();
    if (sess->iface.contextGetBinary(sess->context_handle, buf.data(), size, &written) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: AOT contextGetBinary failed\n");
        return;
    }
    const double serialize_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    auto t1 = std::chrono::steady_clock::now();
    Qnn_ContextHandle_t ctx2 = nullptr;
    if (sess->iface.contextCreateFromBinary(sess->backend_handle, sess->device_handle, nullptr,
                                            buf.data(), written, &ctx2, nullptr) != QNN_SUCCESS || !ctx2) {
        GGML_LOG_ERROR("ggml-qnn: AOT contextCreateFromBinary failed\n");
        return;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    Qnn_GraphHandle_t g2 = nullptr;
    const bool retrieved = sess->iface.graphRetrieve(ctx2, graph_key.c_str(), &g2) == QNN_SUCCESS && g2 != nullptr;

    sess->iface.contextFree(ctx2, nullptr);

    GGML_LOG_INFO("ggml-qnn: AOT round-trip [%s]: finalize=%.1fms serialize=%.1fms binary=%zu bytes reload=%.1fms retrieve=%s\n",
                  graph_key.c_str(), finalize_ms, serialize_ms, (size_t) written, load_ms, retrieved ? "ok" : "FAIL");
}

//
// graph IO buffers
//

// byte size of graph input slot i, matching the on-device tensor dims and dtype
static size_t ggml_qnn_input_size(const ggml_qnn_graph & g, const ggml_tensor * node, size_t i) {
    if (node->op == GGML_OP_MUL_MAT) {
        if (i == 0) {
            return (size_t) node->src[1]->ne[0] * g.n_pad * sizeof(float);
        }
        return g.weight_quantized ? (size_t) ggml_nelements(node->src[0]) * sizeof(ggml_fp16_t)
                                  : ggml_nbytes(node->src[0]);
    }
    return ggml_nbytes(node->src[i]);
}

static size_t ggml_qnn_output_size(const ggml_qnn_graph & g, const ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT) {
        return (size_t) node->src[0]->ne[1] * g.n_pad * sizeof(float);
    }
    return ggml_nbytes(node);
}

static void * ggml_qnn_input_ptr(ggml_qnn_graph * g, size_t i) {
    return g->shared_mem ? g->mem_inputs[i].data : g->host_inputs[i];
}

static void * ggml_qnn_output_ptr(ggml_qnn_graph * g) {
    return g->shared_mem ? g->mem_output.data : g->host_output;
}

// allocate the graph-owned IO buffers and bind them into the tensors once. registered
// fastrpc memory when GGML_QNN_SHARED_MEM is set and works, plain host memory otherwise
static bool ggml_qnn_graph_setup_io(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const size_t n_in = g.inputs.size();

    if (getenv("GGML_QNN_SHARED_MEM") && ggml_qnn_mem_available()) {
        bool ok = true;
        g.mem_inputs.resize(n_in);
        for (size_t i = 0; i < n_in && ok; i++) {
            ok = ggml_qnn_mem_alloc(&sess->iface, sess->context_handle, ggml_qnn_input_size(g, node, i), &g.mem_inputs[i]);
        }
        if (ok) {
            ok = ggml_qnn_mem_alloc(&sess->iface, sess->context_handle, ggml_qnn_output_size(g, node), &g.mem_output);
        }
        if (ok) {
            for (size_t i = 0; i < n_in; i++) {
                memset(g.mem_inputs[i].data, 0, g.mem_inputs[i].size);
                g.inputs[i].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
                g.inputs[i].v1.memHandle = g.mem_inputs[i].handle;
            }
            memset(g.mem_output.data, 0, g.mem_output.size);
            g.output.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
            g.output.v1.memHandle = g.mem_output.handle;
            g.shared_mem = true;
            ggml_qnn_stat_io_shared.fetch_add(1);
            return true;
        }
        ggml_qnn_stat_io_shm_fallback.fetch_add(1);
        for (auto & b : g.mem_inputs) {
            ggml_qnn_mem_free(&sess->iface, &b);
        }
        ggml_qnn_mem_free(&sess->iface, &g.mem_output);
        g.mem_inputs.clear();
        GGML_LOG_DEBUG("ggml-qnn: shared memory setup failed, using host buffers for this graph\n");
    }

    g.host_inputs.assign(n_in, nullptr);
    for (size_t i = 0; i < n_in; i++) {
        const size_t sz = ggml_qnn_input_size(g, node, i);
        g.host_inputs[i] = ggml_qnn_host_alloc(sz);
        if (!g.host_inputs[i]) {
            return false;
        }
        memset(g.host_inputs[i], 0, sz);
        g.inputs[i].v1.clientBuf.data     = g.host_inputs[i];
        g.inputs[i].v1.clientBuf.dataSize = (uint32_t) sz;
    }
    const size_t out_sz = ggml_qnn_output_size(g, node);
    g.host_output = ggml_qnn_host_alloc(out_sz);
    if (!g.host_output) {
        return false;
    }
    memset(g.host_output, 0, out_sz);
    g.output.v1.clientBuf.data     = g.host_output;
    g.output.v1.clientBuf.dataSize = (uint32_t) out_sz;
    ggml_qnn_stat_io_host.fetch_add(1);
    return true;
}

static void ggml_qnn_graph_release_buffers(ggml_qnn_session * sess, ggml_qnn_graph & g) {
    for (auto & b : g.mem_inputs) {
        ggml_qnn_mem_free(&sess->iface, &b);
    }
    ggml_qnn_mem_free(&sess->iface, &g.mem_output);
    g.mem_inputs.clear();
    for (void * p : g.host_inputs) {
        ggml_qnn_host_free(p);
    }
    g.host_inputs.clear();
    ggml_qnn_host_free(g.host_output);
    g.host_output = nullptr;
    std::vector<uint8_t>().swap(g.bake);
}

//
// graph cache
//

static ggml_qnn_graph * ggml_qnn_get_graph(ggml_qnn_session * sess, const ggml_tensor * node) {
    const std::string key = ggml_qnn_graph_key(node);

    auto it = sess->graphs.find(key);
    if (it != sess->graphs.end()) {
        ggml_qnn_stat_graph_cache_hits.fetch_add(1);
        return &it->second;
    }

    // the graph lives in the cache from the start: map nodes are address-stable, so every
    // pointer a timed call captures into it stays valid even after a timeout abandons the call
    ggml_qnn_graph & g = sess->graphs.emplace(key, ggml_qnn_graph()).first->second;

    // a shape that failed before (this run or, with GGML_QNN_DENYLIST, an earlier one) is
    // never built again: no doomed 1-13s finalize per start, no second wedge on a bad shape
    const std::string shape_key = ggml_qnn_shape_key(node);
    if (ggml_qnn_denylisted(shape_key)) {
        return &g;
    }

    // policy rejections must not create a QNN graph at all
    if (node->op == GGML_OP_MUL_MAT && !ggml_qnn_mul_mat_policy(sess, g, node)) {
        return &g;
    }

    // burst clocks are applied on first real use, so merely enumerating the device
    // does not pin the NPU at TURBO
    if (!sess->burst_tried) {
        sess->burst_tried = true;
        if (!getenv("GGML_QNN_NO_BURST")) {
            ggml_qnn_set_burst_mode(sess);
        }
    }

    // HTP has no native FP32 matmul, request FP16 precision so those graphs convert internally,
    // elementwise ops stay FP32 to keep full precision
    QnnHtpGraph_CustomConfig_t precision = {};
    precision.option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    precision.precision = QNN_PRECISION_FLOAT16;

    // finalize at the highest optimization level so the HTP produces fast kernels instead of the
    // low-effort default, costs more finalize time but graphs are cached and reused
    QnnHtpGraph_CustomConfig_t optimize = {};
    optimize.option                    = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    optimize.optimizationOption.type   = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    optimize.optimizationOption.floatValue = 3.0f;

    QnnGraph_Config_t cfg_prec = {};
    cfg_prec.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    cfg_prec.customConfig = &precision;

    QnnGraph_Config_t cfg_opt = {};
    cfg_opt.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    cfg_opt.customConfig = &optimize;

    // GGML_QNN_NO_OPT drops the finalize-optimization flag, a debugging lever for
    // execute-time hangs whose graphs finalize fine. matmul-only: the elementwise path is
    // env-gated off everywhere, so a no-opt variant for it would ship unexercised
    static const bool no_opt = getenv("GGML_QNN_NO_OPT") != nullptr;
    const QnnGraph_Config_t * cfgs_mm[]       = { &cfg_prec, &cfg_opt, nullptr };
    const QnnGraph_Config_t * cfgs_mm_noopt[] = { &cfg_prec, nullptr };
    const QnnGraph_Config_t * cfgs_ew[]       = { &cfg_opt, nullptr };
    const QnnGraph_Config_t ** graph_cfgs = node->op == GGML_OP_MUL_MAT
        ? (no_opt ? cfgs_mm_noopt : cfgs_mm)
        : cfgs_ew;

    bool ok        = sess->iface.graphCreate(sess->context_handle, key.c_str(), graph_cfgs, &g.handle) == QNN_SUCCESS;
    if (ok) {
        ggml_qnn_stat_graphs_created.fetch_add(1);
        if (no_opt && node->op == GGML_OP_MUL_MAT) {
            ggml_qnn_stat_graphs_noopt.fetch_add(1);
        }
    }
    bool timed_out = false;
    if (!ok) {
        GGML_LOG_ERROR("ggml-qnn: failed to create graph %s\n", key.c_str());
        g.handle = nullptr;
    }

    if (ok) {
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ok = ggml_qnn_build_mul_mat(sess, g, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                ok = ggml_qnn_build_binary(sess, g, node);
                break;
            default:
                GGML_ABORT("ggml-qnn: unsupported op %s\n", ggml_op_desc(node));
        }
    }

    // experimental: measure whether an AOT context binary reloads faster than a fresh finalize
    static bool aot_tested = false;
    const bool aot_test = ok && !aot_tested && getenv("GGML_QNN_AOT_TEST");

    auto t_fin = std::chrono::steady_clock::now();
    if (ok) {
        // a shape that wedges the HTP hangs in finalize; the timeout makes supports_op
        // return false so the op is placed on the CPU before compute
        Qnn_GraphHandle_t        h   = g.handle;
        QNN_INTERFACE_VER_TYPE * ifp = &sess->iface;
        Qnn_ErrorHandle_t        fin = QNN_SUCCESS;
        const bool completed = ggml_qnn_call_timed(sess, [ifp, h]() { return ifp->graphFinalize(h, nullptr, nullptr); }, &fin);
        if (!completed) {
            GGML_LOG_ERROR("ggml-qnn: graph finalize timed out for %s, HTP wedged - degrading to CPU\n", key.c_str());
            sess->degraded = true;
            timed_out = true;
            ok = false;
        } else if (fin != QNN_SUCCESS) {
            GGML_LOG_ERROR("ggml-qnn: failed to finalize graph %s\n", key.c_str());
            ok = false;
        }
    }
    const double finalize_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fin).count();

    if (ok && g.weights_static) {
        sess->static_bytes += g.pending_static_bytes;
        ggml_qnn_static_committed += g.pending_static_bytes;
        // QNN owns the HTP-layout copy from finalize on, drop the host staging
        std::vector<uint8_t>().swap(g.bake);
    }

    if (aot_test && ok) {
        aot_tested = true;
        ggml_qnn_aot_roundtrip(sess, key, finalize_ms);
    }

    if (ok) {
        ok = ggml_qnn_graph_setup_io(sess, g, node);
        if (!ok) {
            // host allocation failure is an environment verdict, not an HTP one: negative-cache
            // for this session but never denylist the shape
            g.policy_reject = true;
        }
    }

    // test-execute once on the zeroed IO buffers: a shape that finalizes but wedges or fails
    // at execute is rejected here, at supports_op time, instead of aborting a llama_decode
    // batch later. an execute failure can poison the shared context, so it degrades the session
    if (ok && !getenv("GGML_QNN_NO_PREVALIDATE")) {
        Qnn_GraphHandle_t        h     = g.handle;
        Qnn_Tensor_t *           ins   = g.inputs.data();
        uint32_t                 n_ins = (uint32_t) g.inputs.size();
        Qnn_Tensor_t *           out   = &g.output;
        QNN_INTERFACE_VER_TYPE * ifp   = &sess->iface;
        Qnn_ErrorHandle_t        err   = QNN_SUCCESS;
        const bool completed = ggml_qnn_call_timed(sess,
            [ifp, h, ins, n_ins, out]() { return ifp->graphExecute(h, ins, n_ins, out, 1, nullptr, nullptr); }, &err);
        if (!completed) {
            GGML_LOG_ERROR("ggml-qnn: validation execute timed out for %s, HTP wedged - degrading to CPU\n", key.c_str());
            sess->degraded = true;
            timed_out = true;
            ok = false;
        } else if (err != QNN_SUCCESS) {
            GGML_LOG_ERROR("ggml-qnn: validation execute failed for %s: %" PRIu64 " - degrading to CPU\n", key.c_str(), (uint64_t) err);
            sess->degraded = true;
            ok = false;
        }
    }

    if (ok) {
        GGML_LOG_DEBUG("ggml-qnn: built graph %s%s%s\n", key.c_str(),
                       g.shared_mem ? " (shared mem)" : "", g.weights_static ? " (static weights)" : "");
    } else {
        // negative cache: remember the failure, QNN graphs live until the context is freed.
        // after a timeout the abandoned driver call may still touch the buffers, so leak them
        if (!g.policy_reject) {
            ggml_qnn_denylist_add(shape_key);
        }
        if (!timed_out) {
            ggml_qnn_graph_release_buffers(sess, g);
        }
        g.handle = nullptr;
    }

    return &g;
}

//
// execution
//

bool ggml_qnn_supports_node(ggml_qnn_session * sess, const struct ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(sess->mutex);

    if (sess->degraded) {
        return false;
    }

    ggml_qnn_graph * g = ggml_qnn_get_graph(sess, node);
    return g && g->handle != nullptr;
}

enum ggml_status ggml_qnn_compute_node(ggml_qnn_session * sess, struct ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(sess->mutex);

    // once the session degraded (a call wedged the HTP), fail fast so we do not re-hang on every
    // remaining op already placed on this backend in the current batch
    if (sess->degraded) {
        return GGML_STATUS_FAILED;
    }

    ggml_qnn_graph * g = ggml_qnn_get_graph(sess, node);
    if (!g || !g->handle) {
        if (g && !g->warned) {
            GGML_LOG_WARN("ggml-qnn: failing %s, the graph for this shape did not build\n", ggml_op_desc(node));
            g->warned = true;
        }
        return GGML_STATUS_FAILED;
    }

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            // MatMul in0 is the activations (src1), in1 is the weights (src0). only the real
            // N rows are copied, the padded tail rows produce output rows nobody reads
            memcpy(ggml_qnn_input_ptr(g, 0), node->src[1]->data, ggml_nbytes(node->src[1]));
            if (g->weights_static) {
                break;
            }
            const ggml_tensor * w = node->src[0];
            // a model weight is immutable, so it is copied into its IO buffer once
            const bool is_const = w->buffer && w->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            if (is_const && g->weight_cached_ptr == w->data) {
                break;
            }
            if (g->weight_quantized) {
                ggml_qnn_dequant_f16(w, (ggml_fp16_t *) ggml_qnn_input_ptr(g, 1));
            } else {
                memcpy(ggml_qnn_input_ptr(g, 1), w->data, ggml_nbytes(w));
            }
            g->weight_cached_ptr = is_const ? w->data : nullptr;
            break;
        }
        default:
            memcpy(ggml_qnn_input_ptr(g, 0), node->src[0]->data, ggml_nbytes(node->src[0]));
            memcpy(ggml_qnn_input_ptr(g, 1), node->src[1]->data, ggml_nbytes(node->src[1]));
            break;
    }

    Qnn_GraphHandle_t        h     = g->handle;
    Qnn_Tensor_t *           ins   = g->inputs.data();
    uint32_t                 n_ins = (uint32_t) g->inputs.size();
    Qnn_Tensor_t *           out   = &g->output;
    QNN_INTERFACE_VER_TYPE * ifp   = &sess->iface;
    Qnn_ErrorHandle_t        err   = QNN_SUCCESS;
    const bool completed = ggml_qnn_call_timed(sess,
        [ifp, h, ins, n_ins, out]() { return ifp->graphExecute(h, ins, n_ins, out, 1, nullptr, nullptr); }, &err);
    if (!completed) {
        GGML_LOG_ERROR("ggml-qnn: graph execute timed out for %s, HTP wedged - degrading to CPU\n", ggml_op_desc(node));
        ggml_qnn_denylist_add(ggml_qnn_shape_key(node));
        g->handle      = nullptr;
        sess->degraded = true;
        return GGML_STATUS_FAILED;
    }
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: graph execute failed for %s: %" PRIu64 "\n", ggml_op_desc(node), (uint64_t) err);
        // a failed execute (deferred prepare, device memory) can corrupt the shared HTP context,
        // so demote this graph and degrade the whole session to the CPU for what follows
        ggml_qnn_denylist_add(ggml_qnn_shape_key(node));
        g->handle    = nullptr;
        sess->degraded = true;
        return GGML_STATUS_FAILED;
    }

    memcpy(node->data, ggml_qnn_output_ptr(g), ggml_nbytes(node));

    return GGML_STATUS_SUCCESS;
}
