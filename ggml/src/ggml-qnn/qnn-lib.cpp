#include "qnn-lib.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <QnnTypes.h>
#include <QnnCommon.h>
#include <QnnLog.h>
#include <QnnGraph.h>
#include <QnnTensor.h>
#include <QnnOpDef.h>
#include <HTP/QnnHtpGraph.h>

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
    return sess;
}

void ggml_qnn_session_free(ggml_qnn_session * sess) {
    if (!sess) {
        return;
    }
    if (sess->context_handle) {
        sess->iface.contextFree(sess->context_handle, nullptr);
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

// weights in a model buffer can be baked into the graph as a static tensor, QNN then converts
// them to the HTP layout once at finalize instead of on every execute
// experimental: costs an internal weight copy and the graph cache is keyed by the data address,
// so a freed model can leave a stale graph behind
static bool ggml_qnn_weights_static(const ggml_tensor * node) {
    static const bool enabled = getenv("GGML_QNN_STATIC_WEIGHTS") != nullptr;
    return enabled && node->op == GGML_OP_MUL_MAT && node->src[0]->data &&
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

    g.weights_static = ggml_qnn_weights_static(node);

    // ggml: dst(NxM row-major) = src1(NxK) * src0(MxK)^T
    g.inputs.resize(g.weights_static ? 1 : 2);
    if (!ggml_qnn_tensor_init(sess, g, g.inputs[0], "in0", QNN_TENSOR_TYPE_APP_WRITE, {N, K}, QNN_DATATYPE_FLOAT_32)) {
        return false;
    }

    Qnn_Tensor_t & w = g.weights_static ? g.weights : g.inputs[1];
    if (g.weights_static) {
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_STATIC, {M, K}, ggml_qnn_dtype(src0->type),
                                  src0->data, (uint32_t) ggml_nbytes(src0))) {
            return false;
        }
    } else {
        if (!ggml_qnn_tensor_init(sess, g, w, "in1", QNN_TENSOR_TYPE_APP_WRITE, {M, K}, ggml_qnn_dtype(src0->type))) {
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

    QnnGraph_Config_t cfg = {};
    cfg.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    cfg.customConfig = &precision;

    const QnnGraph_Config_t * cfgs[] = { &cfg, nullptr };
    const QnnGraph_Config_t ** graph_cfgs = node->op == GGML_OP_MUL_MAT ? cfgs : nullptr;

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

    if (ok && sess->iface.graphFinalize(g.handle, nullptr, nullptr) != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to finalize graph %s\n", key.c_str());
        ok = false;
    }

    if (ok) {
        GGML_LOG_DEBUG("ggml-qnn: built graph %s\n", key.c_str());
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

bool ggml_qnn_supports_node(ggml_qnn_session * sess, const struct ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(sess->mutex);

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
        case GGML_OP_MUL_MAT:
            // MatMul in0 is the activations (src1), in1 is the weights (src0)
            ggml_qnn_bind_tensor(g->inputs[0], node->src[1]);
            if (!g->weights_static) {
                ggml_qnn_bind_tensor(g->inputs[1], node->src[0]);
            }
            break;
        default:
            ggml_qnn_bind_tensor(g->inputs[0], node->src[0]);
            ggml_qnn_bind_tensor(g->inputs[1], node->src[1]);
            break;
    }
    ggml_qnn_bind_tensor(g->output, node);

    Qnn_ErrorHandle_t err = sess->iface.graphExecute(g->handle,
        g->inputs.data(), (uint32_t) g->inputs.size(), &g->output, 1, nullptr, nullptr);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: graph execute failed for %s: %" PRIu64 "\n", ggml_op_desc(node), (uint64_t) err);
        // the HTP can also fail at first execute (deferred prepare, device memory), demote the
        // shape so supports_op sends it to the CPU from the next graph on
        g->handle = nullptr;
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}
