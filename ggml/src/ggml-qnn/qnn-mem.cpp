#include "qnn-mem.h"

#include "ggml-impl.h"

#include <QnnMem.h>

#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

// fastrpc constants, normally from the Hexagon SDK rpcmem.h which the QNN SDK does not ship
#define GGML_QNN_RPCMEM_HEAP_ID_SYSTEM 25
#define GGML_QNN_RPCMEM_DEFAULT_FLAGS  1

typedef void * (*rpcmem_alloc2_fn_t)(int heapid, uint32_t flags, size_t size);
typedef void   (*rpcmem_free_fn_t)  (void * po);
typedef int    (*rpcmem_to_fd_fn_t) (void * po);

static rpcmem_alloc2_fn_t rpcmem_alloc2 = nullptr;
static rpcmem_free_fn_t   rpcmem_free   = nullptr;
static rpcmem_to_fd_fn_t  rpcmem_to_fd  = nullptr;

#ifdef _WIN32
// the fastrpc driver lives in the DriverStore, its path is the qcnspmcdm service binary dir
static std::string ggml_qnn_fastrpc_dir(void) {
    std::string result;

    SC_HANDLE scm = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
    if (!scm) {
        return result;
    }
    SC_HANDLE svc = OpenServiceW(scm, L"qcnspmcdm", SERVICE_QUERY_CONFIG);
    if (!svc) {
        CloseServiceHandle(scm);
        return result;
    }

    DWORD size = 0;
    QueryServiceConfigW(svc, NULL, 0, &size);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<uint8_t> buf(size);
        auto * cfg = reinterpret_cast<LPQUERY_SERVICE_CONFIGW>(buf.data());
        if (QueryServiceConfigW(svc, cfg, size, &size)) {
            std::wstring path = cfg->lpBinaryPathName;
            path = path.substr(0, path.find_last_of(L'\\'));

            // the service path starts with the \SystemRoot placeholder, resolve it via windir
            const std::wstring placeholder = L"\\SystemRoot";
            if (path.compare(0, placeholder.size(), placeholder) == 0) {
                wchar_t windir[MAX_PATH];
                DWORD n = GetEnvironmentVariableW(L"windir", windir, MAX_PATH);
                if (n > 0 && n < MAX_PATH) {
                    path.replace(0, placeholder.size(), windir);
                }
            }
            int len = WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, NULL, 0, NULL, NULL);
            if (len > 0) {
                result.resize(len - 1);
                WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, result.data(), len, NULL, NULL);
            }
        }
    }

    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    return result;
}
#endif

static void * ggml_qnn_load_fastrpc(void) {
#ifdef _WIN32
    void * lib = (void *) LoadLibraryA("libcdsprpc.dll");
    if (!lib) {
        std::string dir = ggml_qnn_fastrpc_dir();
        if (!dir.empty()) {
            std::string path = dir + "\\libcdsprpc.dll";
            lib = (void *) LoadLibraryExA(path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    return lib;
#else
    return dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

static void * ggml_qnn_dl_sym(void * lib, const char * name) {
#ifdef _WIN32
    return (void *) GetProcAddress((HMODULE) lib, name);
#else
    return dlsym(lib, name);
#endif
}

bool ggml_qnn_mem_available(void) {
    static bool tried = false;
    static bool ok    = false;
    if (tried) {
        return ok;
    }
    tried = true;

    void * lib = ggml_qnn_load_fastrpc();
    if (!lib) {
        GGML_LOG_DEBUG("ggml-qnn: fastrpc (libcdsprpc) not found, shared memory unavailable\n");
        return false;
    }

    rpcmem_alloc2 = (rpcmem_alloc2_fn_t) ggml_qnn_dl_sym(lib, "rpcmem_alloc2");
    rpcmem_free   = (rpcmem_free_fn_t)   ggml_qnn_dl_sym(lib, "rpcmem_free");
    rpcmem_to_fd  = (rpcmem_to_fd_fn_t)  ggml_qnn_dl_sym(lib, "rpcmem_to_fd");

    ok = rpcmem_alloc2 && rpcmem_free && rpcmem_to_fd;
    if (!ok) {
        GGML_LOG_ERROR("ggml-qnn: fastrpc loaded but rpcmem symbols missing, shared memory unavailable\n");
    }
    return ok;
}

bool ggml_qnn_mem_alloc(const QNN_INTERFACE_VER_TYPE * iface, Qnn_ContextHandle_t context,
                        size_t size, ggml_qnn_mem_buffer * out) {
    if (!ggml_qnn_mem_available()) {
        return false;
    }

    void * data = rpcmem_alloc2(GGML_QNN_RPCMEM_HEAP_ID_SYSTEM, GGML_QNN_RPCMEM_DEFAULT_FLAGS, size);
    if (!data) {
        GGML_LOG_ERROR("ggml-qnn: rpcmem_alloc2 failed for %zu bytes\n", size);
        return false;
    }
    int fd = rpcmem_to_fd(data);
    if (fd < 0) {
        GGML_LOG_ERROR("ggml-qnn: rpcmem_to_fd failed\n");
        rpcmem_free(data);
        return false;
    }

    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    uint32_t dims[1]         = { (uint32_t) size };
    desc.memShape.numDim     = 1;
    desc.memShape.dimSize    = dims;
    desc.dataType            = QNN_DATATYPE_UFIXED_POINT_8;
    desc.memType             = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd          = fd;

    Qnn_MemHandle_t handle = nullptr;
    if (iface->memRegister(context, &desc, 1, &handle) != QNN_SUCCESS || !handle) {
        GGML_LOG_ERROR("ggml-qnn: QnnMem_register failed\n");
        rpcmem_free(data);
        return false;
    }

    out->data   = data;
    out->fd     = fd;
    out->size   = size;
    out->handle = handle;
    return true;
}

void ggml_qnn_mem_free(const QNN_INTERFACE_VER_TYPE * iface, ggml_qnn_mem_buffer * buf) {
    if (!buf || !buf->data) {
        return;
    }
    if (buf->handle) {
        iface->memDeRegister(&buf->handle, 1);
    }
    rpcmem_free(buf->data);
    *buf = {};
}

bool ggml_qnn_mem_self_test(const QNN_INTERFACE_VER_TYPE * iface, Qnn_ContextHandle_t context) {
    ggml_qnn_mem_buffer buf;
    if (!ggml_qnn_mem_alloc(iface, context, 4096, &buf)) {
        return false;
    }
    memset(buf.data, 0, buf.size);
    ggml_qnn_mem_free(iface, &buf);
    GGML_LOG_INFO("ggml-qnn: fastrpc shared memory available (alloc + register + free ok)\n");
    return true;
}
