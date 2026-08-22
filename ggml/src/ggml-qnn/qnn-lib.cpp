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

#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

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

//
// session
//

typedef Qnn_ErrorHandle_t (*ggml_qnn_get_providers_fn_t)(const QnnInterface_t *** providers, uint32_t * n_providers);

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

ggml_qnn_session * ggml_qnn_session_init(void) {
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

    // lock the NPU to high clocks unless explicitly disabled
    if (!getenv("GGML_QNN_NO_BURST")) {
        ggml_qnn_set_burst_mode(sess);
    }

    // probe fastrpc shared memory, the foundation for registered zero-remap graph IO
    if (getenv("GGML_QNN_SHARED_MEM")) {
        if (!ggml_qnn_mem_self_test(&sess->iface, sess->context_handle)) {
            GGML_LOG_INFO("ggml-qnn: fastrpc shared memory not available on this device\n");
        }
    }

    return sess;
}

void ggml_qnn_session_free(ggml_qnn_session * sess) {
    if (!sess) {
        return;
    }
    // deregister + free shared buffers while the context is still alive
    for (auto & kv : sess->graphs) {
        ggml_qnn_graph & g = kv.second;
        for (auto & b : g.mem_inputs) {
            ggml_qnn_mem_free(&sess->iface, &b);
        }
        ggml_qnn_mem_free(&sess->iface, &g.mem_output);
    }
    if (sess->context_handle) {
        sess->iface.contextFree(sess->context_handle, nullptr);
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

static Qnn_DataType_t ggml_qnn_dtype(enum ggml_type type) {
    return type == GGML_TYPE_F16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
}

// the on-device weight dtype: F32 stays F32, F16 and every quantized type land as F16
static Qnn_DataType_t ggml_qnn_weight_dtype(enum ggml_type type) {
    return type == GGML_TYPE_F32 ? QNN_DATATYPE_FLOAT_32 : QNN_DATATYPE_FLOAT_16;
}

// weights in a model buffer can be baked into the graph as a static tensor, QNN then converts
// them to the HTP layout once at finalize instead of on every execute
// experimental: costs an internal weight copy and the graph cache is keyed by the data address,
// so a freed model can leave a stale graph behind
static bool ggml_qnn_weights_static(const ggml_tensor * node) {
    static const bool enabled = getenv("GGML_QNN_STATIC_WEIGHTS") != nullptr;
    return enabled && node->op == GGML_OP_MUL_MAT && node->src[0]->data &&
           !ggml_is_quantized(node->src[0]->type) &&
           node->src[0]->buffer && node->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
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

static bool ggml_qnn_build_mul_mat(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t M = (uint32_t) src0->ne[1];
    const uint32_t N = (uint32_t) src1->ne[1];

    g.weights_static   = ggml_qnn_weights_static(node);
    g.weight_quantized = ggml_is_quantized(src0->type);
    if (g.weight_quantized) {
        g.dequant_f32.resize((size_t) M * K);
        g.dequant_f16.resize((size_t) M * K);
    }

    // ggml: dst(NxM row-major) = src1(NxK) * src0(MxK)^T
    g.inputs.resize(g.weights_static ? 1 : 2);
    if (!ggml_qnn_tensor_init(sess, g, g.inputs[0], "in0", QNN_TENSOR_TYPE_APP_WRITE, {N, K}, QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    Qnn_Tensor_t & w = g.weights_static ? g.weights : g.inputs[1];
    if (g.weights_static) {
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_STATIC, {M, K}, ggml_qnn_weight_dtype(src0->type),
                                  src0->data, (uint32_t) ggml_nbytes(src0))) {
            return false;
        }
    } else {
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_APP_WRITE, {M, K}, ggml_qnn_weight_dtype(src0->type))) {
            return false;
        }
    }

    if (!ggml_qnn_tensor_init(sess, g, g.output, "out", QNN_TENSOR_TYPE_APP_READ, {N, M}, QNN_DATATYPE_FLOAT_32)) {
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

    g.inputs.resize(2);
    if (!ggml_qnn_tensor_init(sess, g, g.inputs[0], "in0", QNN_TENSOR_TYPE_APP_WRITE, ggml_qnn_dims(src0), QNN_DATATYPE_FLOAT_32) ||
        !ggml_qnn_tensor_init(sess, g, g.inputs[1], "in1", QNN_TENSOR_TYPE_APP_WRITE, ggml_qnn_dims(src1), QNN_DATATYPE_FLOAT_32) ||
        !ggml_qnn_tensor_init(sess, g, g.output,    "out", QNN_TENSOR_TYPE_APP_READ,  ggml_qnn_dims(node), QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    const char * type_name = node->op == GGML_OP_ADD ? QNN_OP_ELEMENT_WISE_ADD : QNN_OP_ELEMENT_WISE_MULTIPLY;
    return ggml_qnn_add_node(sess, g, type_name, nullptr, 0, g.inputs.data(), (uint32_t) g.inputs.size());
}

static std::string ggml_qnn_graph_key(const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    char buf[256];
    int n = snprintf(buf, sizeof(buf), "%s_%s_%s_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
             ggml_op_name(node->op), ggml_type_name(src0->type), ggml_type_name(src1->type),
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
             src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    if (ggml_qnn_weights_static(node)) {
        snprintf(buf + n, sizeof(buf) - n, "_w%p", src0->data);
    }
    return buf;
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

static void ggml_qnn_graph_setup_shared_mem(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node);

static ggml_qnn_graph * ggml_qnn_get_graph(ggml_qnn_session * sess, const ggml_tensor * node) {
    const std::string key = ggml_qnn_graph_key(node);

    auto it = sess->graphs.find(key);
    if (it != sess->graphs.end()) {
        return &it->second;
    }

    ggml_qnn_graph g;

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

    // matmul runs in fp16 for speed, elementwise stays fp32 for precision, both optimized
    const QnnGraph_Config_t * cfgs_mm[]  = { &cfg_prec, &cfg_opt, nullptr };
    const QnnGraph_Config_t * cfgs_ew[]  = { &cfg_opt, nullptr };
    const QnnGraph_Config_t ** graph_cfgs = node->op == GGML_OP_MUL_MAT ? cfgs_mm : cfgs_ew;

    bool ok = sess->iface.graphCreate(sess->context_handle, key.c_str(), graph_cfgs, &g.handle) == QNN_SUCCESS;
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

    // experimental: measure whether an AOT context binary reloads faster than a fresh finalize,
    // this is the decisive question for a compile-once/load-fast NPU path (route 2)
    static bool aot_tested = false;
    const bool aot_test = ok && !aot_tested && getenv("GGML_QNN_AOT_TEST");

    auto t_fin = std::chrono::steady_clock::now();
    if (ok && sess->iface.graphFinalize(g.handle, nullptr, nullptr) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to finalize graph %s\n", key.c_str());
        ok = false;
    }
    const double finalize_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fin).count();

    if (aot_test && ok) {
        aot_tested = true;
        ggml_qnn_aot_roundtrip(sess, key, finalize_ms);
    }

    if (ok) {
        ggml_qnn_graph_setup_shared_mem(sess, g, node);
        GGML_LOG_DEBUG("ggml-qnn: built graph %s%s\n", key.c_str(), g.shared_mem ? " (shared mem)" : "");
    }

    if (!ok) {
        // negative cache: remember the failure, QNN graphs live until the context is freed
        g.handle = nullptr;
    }

    auto res = sess->graphs.emplace(std::move(key), std::move(g));
    return &res.first->second;
}

//
// execution
//

static void ggml_qnn_bind_tensor(Qnn_Tensor_t & t, const ggml_tensor * src) {
    t.v1.clientBuf.data     = src->data;
    t.v1.clientBuf.dataSize = (uint32_t) ggml_nbytes(src);
}

// bind host data into input slot i: copy into the registered shared buffer and point the tensor
// at its memHandle (no per-execute re-map), or fall back to a RAW client buffer
static void ggml_qnn_bind_input(ggml_qnn_graph * g, size_t i, const void * data, uint32_t size) {
    if (g->shared_mem) {
        memcpy(g->mem_inputs[i].data, data, size);
        g->inputs[i].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        g->inputs[i].v1.memHandle = g->mem_inputs[i].handle;
    } else {
        g->inputs[i].v1.clientBuf.data     = (void *) data;
        g->inputs[i].v1.clientBuf.dataSize = size;
    }
}

// byte size the registered buffer for input slot i must hold (matches the on-device tensor dtype)
static uint32_t ggml_qnn_input_size(const ggml_tensor * node, bool weight_quantized, size_t i) {
    if (node->op == GGML_OP_MUL_MAT) {
        if (i == 0) {
            return (uint32_t) ggml_nbytes(node->src[1]);
        }
        return weight_quantized ? (uint32_t) (ggml_nelements(node->src[0]) * sizeof(ggml_fp16_t))
                                : (uint32_t) ggml_nbytes(node->src[0]);
    }
    return (uint32_t) ggml_nbytes(node->src[i]);
}

// allocate + register a shared buffer for each graph input and the output, called after finalize
static void ggml_qnn_graph_setup_shared_mem(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    if (!getenv("GGML_QNN_SHARED_MEM") || !ggml_qnn_mem_available()) {
        return;
    }

    bool ok = true;
    g.mem_inputs.resize(g.inputs.size());
    for (size_t i = 0; i < g.inputs.size() && ok; i++) {
        ok = ggml_qnn_mem_alloc(&sess->iface, sess->context_handle,
                                ggml_qnn_input_size(node, g.weight_quantized, i), &g.mem_inputs[i]);
    }
    if (ok) {
        ok = ggml_qnn_mem_alloc(&sess->iface, sess->context_handle, (size_t) ggml_nbytes(node), &g.mem_output);
    }
    if (!ok) {
        for (auto & b : g.mem_inputs) {
            ggml_qnn_mem_free(&sess->iface, &b);
        }
        ggml_qnn_mem_free(&sess->iface, &g.mem_output);
        g.mem_inputs.clear();
        GGML_LOG_DEBUG("ggml-qnn: shared memory setup failed, using RAW buffers for this graph\n");
        return;
    }
    g.shared_mem = true;
}

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
            // MatMul in0 is the activations (src1), in1 is the weights (src0)
            ggml_qnn_bind_input(g, 0, node->src[1]->data, (uint32_t) ggml_nbytes(node->src[1]));
            if (g->weights_static) {
                break;
            }
            const ggml_tensor * w = node->src[0];
            // a constant weight already sits in its shared buffer, skip the dequant and the copy
            if (g->shared_mem && g->weight_cached_ptr == w->data) {
                break;
            }
            if (g->weight_quantized) {
                // dequantize the weights to fp16 on the host, HTP has no quantized matmul path here
                const int64_t n = ggml_nelements(w);
                ggml_get_type_traits(w->type)->to_float(w->data, g->dequant_f32.data(), n);
                ggml_fp32_to_fp16_row(g->dequant_f32.data(), g->dequant_f16.data(), n);
                ggml_qnn_bind_input(g, 1, g->dequant_f16.data(), (uint32_t) (n * sizeof(ggml_fp16_t)));
            } else {
                ggml_qnn_bind_input(g, 1, w->data, (uint32_t) ggml_nbytes(w));
            }
            g->weight_cached_ptr = w->data;
            break;
        }
        default:
            ggml_qnn_bind_input(g, 0, node->src[0]->data, (uint32_t) ggml_nbytes(node->src[0]));
            ggml_qnn_bind_input(g, 1, node->src[1]->data, (uint32_t) ggml_nbytes(node->src[1]));
            break;
    }

    if (g->shared_mem) {
        g->output.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        g->output.v1.memHandle = g->mem_output.handle;
    } else {
        ggml_qnn_bind_tensor(g->output, node);
    }

    Qnn_ErrorHandle_t err = sess->iface.graphExecute(g->handle,
        g->inputs.data(), (uint32_t) g->inputs.size(), &g->output, 1, nullptr, nullptr);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: graph execute failed for %s: %" PRIu64 "\n", ggml_op_desc(node), (uint64_t) err);
        // a failed execute (deferred prepare, device memory) can corrupt the shared HTP context,
        // so demote this graph and degrade the whole session to the CPU for what follows
        g->handle    = nullptr;
        sess->degraded = true;
        return GGML_STATUS_FAILED;
    }

    if (g->shared_mem) {
        memcpy(node->data, g->mem_output.data, ggml_nbytes(node));
    }

    return GGML_STATUS_SUCCESS;
}
