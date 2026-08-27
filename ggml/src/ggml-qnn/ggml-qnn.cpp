#include "ggml-qnn.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "qnn-lib.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

// backend instances share one refcounted session, freed when the last instance is freed:
// a session still alive at process exit crashes inside QnnHtp (ExitProcess kills its worker
// threads before DLL detach; static destructors, atexit, DLL detach - all tried), so only
// mid-process free is safe. a degraded session is kept instead of freed: an abandoned
// watchdog call may still touch it from inside the driver, re-initializing against a wedged
// HTP could hang unwatchdogged in deviceCreate, and it keeps init_backend succeeding (it
// claims no ops, so everything falls back to the CPU) instead of failing llama_new_context
static std::mutex          ggml_qnn_session_mutex;
static ggml_qnn_session *  ggml_qnn_session_ptr    = nullptr;
static int                 ggml_qnn_session_refs   = 0;
static bool                ggml_qnn_session_worked = false;
static bool                ggml_qnn_session_failed = false;

static ggml_qnn_session * ggml_backend_qnn_session_acquire(bool add_ref) {
    std::lock_guard<std::mutex> lock(ggml_qnn_session_mutex);
    if (!ggml_qnn_session_ptr && !ggml_qnn_session_failed) {
        ggml_qnn_session_ptr = ggml_qnn_session_init();
        if (ggml_qnn_session_ptr) {
            ggml_qnn_session_worked = true;
        } else if (!ggml_qnn_session_worked) {
            // no HTP on this machine, stop probing. a failure after a successful run is
            // not latched, so a transient error can be retried on the next acquire
            ggml_qnn_session_failed = true;
        }
    }
    if (ggml_qnn_session_ptr && add_ref) {
        ggml_qnn_session_refs++;
    }
    return ggml_qnn_session_ptr;
}

static void ggml_backend_qnn_session_release(void) {
    std::lock_guard<std::mutex> lock(ggml_qnn_session_mutex);
    GGML_ASSERT(ggml_qnn_session_refs > 0);
    if (--ggml_qnn_session_refs == 0) {
        if (ggml_qnn_session_ptr->degraded) {
            GGML_LOG_WARN("ggml-qnn: keeping degraded session, NPU disabled for this process\n");
            return;
        }
        ggml_qnn_session_free(ggml_qnn_session_ptr);
        ggml_qnn_session_ptr = nullptr;
    }
}

// backend interface

static const char * ggml_backend_qnn_get_name(ggml_backend_t backend) {
    return "QNN";

    GGML_UNUSED(backend);
}

static void ggml_backend_qnn_free(ggml_backend_t backend) {
    ggml_backend_qnn_session_release();
    delete backend;
}

static enum ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_qnn_session * sess = (ggml_qnn_session *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                if (ggml_qnn_compute_node(sess, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_qnn_i = {
    /* .get_name                = */ ggml_backend_qnn_get_name,
    /* .free                    = */ ggml_backend_qnn_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_qnn_guid(void) {
    static ggml_guid guid = { 0x7c, 0x3d, 0x91, 0x5e, 0xa2, 0x08, 0x4b, 0xf1, 0xb6, 0x59, 0xdd, 0x27, 0x40, 0x8a, 0x1e, 0xc3 };
    return &guid;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

// device interface

static const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    return "QNN";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    return "Qualcomm Hexagon NPU (HTP)";

    GGML_UNUSED(dev);
}

static void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // the NPU works from system memory, no dedicated memory to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type        = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_qnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_qnn_session * sess = ggml_backend_qnn_session_acquire(/*add_ref=*/true);
    if (!sess) {
        return NULL;
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_qnn_guid(),
        /* .iface   = */ ggml_backend_qnn_i,
        /* .device  = */ dev,
        /* .context = */ sess,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // 2D weights (F32, F16, or any quantized type with a dequantizer) times F32
            // activations, the HTP runs the math in FP16
            // a quantized weight is claimable only when it can be baked statically (dequant once,
            // correct) or the experimental per-execute dequant path is enabled
            static const bool quant_ok  = getenv("GGML_QNN_QUANTIZED") != nullptr;
            static const bool static_on = getenv("GGML_QNN_NO_STATIC_WEIGHTS") == nullptr;
            const bool src0_ok = src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                                 ((quant_ok || static_on) && ggml_is_quantized(src0->type) &&
                                  ggml_get_type_traits(src0->type)->to_float != NULL);
            if (!src0_ok || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_is_matrix(src0) || !ggml_is_matrix(src1)) {
                return false;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            // the NPU pays off only for larger matrices, GGML_QNN_MIN_DIM=1 covers small shapes in tests
            static const int64_t min_dim = getenv("GGML_QNN_MIN_DIM") ? atoll(getenv("GGML_QNN_MIN_DIM")) : 32;

            const int64_t ne10 = src1->ne[0];
            const int64_t ne0  = op->ne[0];
            const int64_t ne1  = op->ne[1];

            if (ne0 < min_dim || ne1 < min_dim || ne10 < min_dim ||
                ggml_nbytes(src0) > UINT32_MAX ||
                ggml_nbytes(src1) > UINT32_MAX ||
                ggml_nbytes(op)   > UINT32_MAX ||
                src0->ne[0] * src0->ne[1] * 2 > UINT32_MAX) { // fp16 weight buffer on device
                return false;
            }
            break;
        }

        case GGML_OP_ADD:
        case GGML_OP_MUL:
        {
            // experimental: HTP runs FP32 as FP16 internally, and its Add broadcast
            // returns wrong results for some shapes, so elementwise offload is opt-in
            static const bool elementwise = getenv("GGML_QNN_ELEMENTWISE") != nullptr;
            if (!elementwise) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }
            // same shape, or a bias-style broadcast of a single row
            if (!ggml_are_same_shape(src0, src1) &&
                !(ggml_n_dims(src1) == 1 && src1->ne[0] == src0->ne[0])) {
                return false;
            }

            // elementwise ops are memory bound, offload only large tensors
            static const int64_t min_elements =
                getenv("GGML_QNN_MIN_ELEMENTS") ? atoll(getenv("GGML_QNN_MIN_ELEMENTS")) : 1 << 20;
            if (ggml_nelements(src0) < min_elements || ggml_nbytes(src0) > UINT32_MAX) {
                return false;
            }
            break;
        }

        default:
            return false;
    }

    // llama probes every weight at model load, before the data is resident: src0->data is
    // NULL and the buffer is not tagged WEIGHTS yet. A trial build needs the real bytes, so
    // answer from the type and shape policy above and leave the HTP trial to the first
    // scheduled node. Building here asked a different question than schedule time - a
    // quantized weight was refused at load and claimed at run - and cached a finalized
    // dynamic-variant graph for a shape inference never executes
    if (op->op == GGML_OP_MUL_MAT && src0->data == nullptr) {
        return true;
    }

    // the HTP can reject shapes at graph-finalize time (e.g. TCM tiling limits) and can wedge
    // at execute time, so claim an op only after its graph builds AND test-executes, the
    // result is cached with the graph. hold a ref so a concurrent backend free cannot tear
    // the session down mid-query
    ggml_qnn_session * sess = ggml_backend_qnn_session_acquire(/*add_ref=*/true);
    if (!sess) {
        return false;
    }
    const bool ok = ggml_qnn_supports_node(sess, op);
    ggml_backend_qnn_session_release();
    return ok;

    GGML_UNUSED(dev);
}

static bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_qnn_device_i = {
    /* .get_name             = */ ggml_backend_qnn_device_get_name,
    /* .get_description      = */ ggml_backend_qnn_device_get_description,
    /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
    /* .get_type             = */ ggml_backend_qnn_device_get_type,
    /* .get_props            = */ ggml_backend_qnn_device_get_props,
    /* .init_backend         = */ ggml_backend_qnn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_qnn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    return "QNN";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_qnn_session_acquire(/*add_ref=*/false) ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_qnn_device = {
        /* .iface   = */ ggml_backend_qnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_qnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_qnn_reg_i = {
    /* .get_name         = */ ggml_backend_qnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_qnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_qnn_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_qnn_reg(void) {
    static struct ggml_backend_reg ggml_backend_qnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_qnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_qnn_reg;
}

ggml_backend_t ggml_backend_qnn_init(void) {
    ggml_backend_reg_t reg = ggml_backend_qnn_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), NULL);
}

GGML_BACKEND_DL_IMPL(ggml_backend_qnn_reg)
