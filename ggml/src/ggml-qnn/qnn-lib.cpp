#include "qnn-lib.h"
#include "qnn-mem.h"
#include "qnn-dl.h"

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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <system_error>
#include <thread>
#include <unordered_set>

#ifdef _WIN32
#    include <malloc.h>
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
static std::atomic<uint64_t> ggml_qnn_stat_burst_applied{0};
static std::atomic<uint64_t> ggml_qnn_stat_budget_clamped{0}; // 0/1, see ggml_qnn_budget_clamp
// compute-time executes: how many, how many took over a second, and the slowest (ms)
static std::atomic<uint64_t> ggml_qnn_stat_exec_count{0};
static std::atomic<uint64_t> ggml_qnn_stat_exec_slow{0};
static std::atomic<uint64_t> ggml_qnn_stat_exec_max_ms{0};
// the init-time shared-memory self-test of the last session: 0 = the fastrpc library is absent,
// 1 = it is present but the self-test failed, 2 = ok, 3 = not requested (GGML_QNN_SHARED_MEM
// unset). io_shared == 0 alone cannot tell a machine without fastrpc from a broken rpcmem or
// QnnMem_register path
static std::atomic<uint64_t> ggml_qnn_stat_shm_selftest{3};

// written on session teardown and on every degrade; counters are process-global and
// monotonic, so with a refcounted session the last write is the cumulative total for the run
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
    fprintf(f, "burst_applied %" PRIu64 "\n", ggml_qnn_stat_burst_applied.load());
    fprintf(f, "budget_clamped %" PRIu64 "\n", ggml_qnn_stat_budget_clamped.load());
    fprintf(f, "exec_count %" PRIu64 "\n", ggml_qnn_stat_exec_count.load());
    fprintf(f, "exec_slow %" PRIu64 "\n", ggml_qnn_stat_exec_slow.load());
    fprintf(f, "exec_max_ms %" PRIu64 "\n", ggml_qnn_stat_exec_max_ms.load());
    fprintf(f, "shm_selftest %" PRIu64 "\n", ggml_qnn_stat_shm_selftest.load());
    fclose(f);
}

// every degrade goes through here so the counters are flushed while the process is still
// alive: a server killed after a wedge would otherwise lose them.
// the first degrade of the process is written straight to stderr instead of the logger:
// llama-bench without -v installs a null log callback and then reports CPU throughput under
// the QNN backend name with nothing shown. it skips the WARN so it is reported once
static void ggml_qnn_degrade(ggml_qnn_session * sess, const char * why, bool slow_only = false) {
    static std::atomic<bool> reported{false};
    sess->degraded.store(true);
    sess->slow_only = slow_only; // any later hard degrade ends it, see ggml_qnn_session
    if (!reported.exchange(true)) {
        // a slow_only degrade keeps the graphs it already built running on the NPU, so the
        // blanket "everything runs on the CPU" tail would be false for exactly that case
        fprintf(stderr, "ggml-qnn: NPU degraded (%s), claiming no ops for the rest of this process%s\n", why,
                slow_only ? ": graphs already built keep running on the NPU, every new shape goes to the CPU"
                          : ": everything runs on the CPU from here");
        fflush(stderr);
    } else {
        GGML_LOG_WARN("ggml-qnn: NPU degraded (%s), claiming no ops for the rest of this process\n", why);
    }
    ggml_qnn_stats_write();
}

// page-aligned host buffers for graph IO. kept as allocation hygiene: alignment was ruled
// out as the cause of the size-threshold execute hang (that follows padded IO transfer size
// alone - keep max(input, output) under ~1 MB via the pad bucket, see GGML_QNN_NPAD and the
// GGML_QNN_IO_MAX_KB cap)
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

static void * ggml_qnn_load_htp_lib(void) {
#ifdef _WIN32
    const char * lib_name = "QnnHtp.dll";
    const char * lib_dirs[] = { "aarch64-windows-msvc", "arm64x-windows-msvc" };
    const char   sep = '\\';
#else
    const char * lib_name = "libQnnHtp.so";
    // aarch64 only: the device count is 0 on any other build, so this never runs there
    const char * lib_dirs[] = { "aarch64-android", "aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc11.2", "aarch64-oe-linux-gcc9.3" };
    const char   sep = '/';
#endif
    void * lib = ggml_qnn_dl_open(lib_name);
    if (lib) {
        return lib;
    }
    // fall back to the SDK installation, dependent libraries resolve from the same directory.
    // every path keeps its own failure reason: a library that exists but cannot load (missing
    // dependency, wrong architecture) must not read as "not found"
    std::string tried = "the default search path (" + ggml_qnn_dl_error() + ")";
    const char * sdk_root = getenv("QNN_SDK_ROOT");
    if (sdk_root && *sdk_root) {
        for (const char * dir : lib_dirs) {
            char path[1024];
            snprintf(path, sizeof(path), "%s%clib%c%s%c%s", sdk_root, sep, sep, dir, sep, lib_name);
            lib = ggml_qnn_dl_open(path);
            if (lib) {
                return lib;
            }
            const std::string why = ggml_qnn_dl_error();
            tried += ", ";
            tried += path;
            tried += " (" + why + ")";
        }
    } else {
        tried += "; QNN_SDK_ROOT is unset";
    }
    GGML_LOG_INFO("ggml-qnn: %s could not be loaded, backend unavailable (tried %s)\n", lib_name, tried.c_str());
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
// persistent worker thread with a timeout; on timeout the thread is detached and the worker
// abandoned (it stays stuck in the driver), and the session degrades to the CPU. the session
// is then deliberately leaked, so the stuck call can only ever touch memory that is still
// alive, and the abandoned worker deletes its own state if it ever returns. on a clean
// session free the idle worker is told to quit and joined before the context goes away
//
// there are two limits. finalize is a compile and may legitimately take long, so it runs
// under GGML_QNN_BUILD_TIMEOUT_MS (default 120000). every execute runs under
// GGML_QNN_TIMEOUT_MS (default 15000), the validation execute included: it exists to predict
// compute-time behavior, so a graph that cannot validate within the compute limit must not
// be claimed, or the first llama_decode fails instead of falling back. there is no first-
// execute premium: measured 2026-09-16, every execute of a graph took the same time. both
// limits are read once
struct ggml_qnn_call_worker {
    std::mutex              m;
    std::condition_variable cv;
    std::function<Qnn_ErrorHandle_t()> job;
    bool has_job  = false;
    bool job_done = false;
    bool lost     = false;
    bool quit     = false;
    Qnn_ErrorHandle_t result = QNN_SUCCESS;
    std::thread thread;
};

static void ggml_qnn_worker_loop(ggml_qnn_call_worker * w) {
    std::unique_lock<std::mutex> lock(w->m);
    for (;;) {
        w->cv.wait(lock, [w] { return w->has_job || w->quit; });
        if (w->quit) {
            // clean quit: session free joins this thread and deletes the state
            return;
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
            // abandoned after a timeout: the thread is already detached, nobody joins it
            lock.unlock();
            delete w;
            return;
        }
    }
}

// capped at a day so the wait deadline cannot overflow the clock
static long long ggml_qnn_build_timeout_ms(void) {
    static const long long ms = std::min<long long>(ggml_qnn_env_ll("GGML_QNN_BUILD_TIMEOUT_MS", 120000, 1), 86400000);
    return ms;
}

// a validation execute at or over this many ms marks the NPU as too slow to use, 0 disables
static long long ggml_qnn_slow_exec_ms(void) {
    static const long long ms = ggml_qnn_env_ll("GGML_QNN_SLOW_EXEC_MS", 2000, 0);
    return ms;
}

static long long ggml_qnn_compute_timeout_ms(void) {
    static const long long ms = std::min<long long>(ggml_qnn_env_ll("GGML_QNN_TIMEOUT_MS", 15000, 1), 86400000);
    return ms;
}

// test-only fault hook GGML_QNN_DELAY_EXECUTE, see qnn-lib.h: the delay in ms for the next
// execute of the graph with this key, 0 for none. one-shot, the match count is guarded by the
// session mutex like GGML_QNN_FAIL_EXECUTE's. key_of builds the key only when the hook is set
static long long ggml_qnn_test_delay_ms(const std::function<std::string()> & key_of) {
    static const char *    sub  = getenv("GGML_QNN_DELAY_EXECUTE");
    static const long long ms   = std::min<long long>(ggml_qnn_env_ll("GGML_QNN_DELAY_EXECUTE_MS", 0, 0), 86400000);
    static const long long skip = ggml_qnn_env_ll("GGML_QNN_DELAY_EXECUTE_SKIP", 0, 0);
    static long long       seen = 0;
    if (!sub || !*sub || ms <= 0) {
        return 0;
    }
    const std::string key = key_of();
    if (key.find(sub) == std::string::npos || seen++ != skip) {
        return 0;
    }
    GGML_LOG_WARN("ggml-qnn: GGML_QNN_DELAY_EXECUTE matches %s, delaying its execute by %lld ms\n", key.c_str(), ms);
    return ms;
}

// returns true if the call completed, writing its result to *out. false means the call did
// not complete: a timeout, or no worker thread could be started. either way the session is
// degraded when this returns false (the timeout callers degrade it themselves, with the phase
// in the reason), and *out holds an error in the second case
static bool ggml_qnn_call_timed(ggml_qnn_session * sess, long long timeout_ms, std::function<Qnn_ErrorHandle_t()> call, Qnn_ErrorHandle_t * out) {
    if (!sess->worker) {
        // the session sees the worker only once its thread runs: a worker without a thread
        // would take a job nobody executes, wait out the full timeout and then detach a
        // non-joinable thread
        const char * why = nullptr;
        try {
            std::unique_ptr<ggml_qnn_call_worker> w(new ggml_qnn_call_worker());
            w->thread = std::thread(ggml_qnn_worker_loop, w.get());
            sess->worker = w.release();
        } catch (const std::system_error &) {
            why = "cannot start the watchdog thread";
        } catch (const std::bad_alloc &) {
            why = "out of host memory starting the watchdog thread";
        }
        if (why) {
            *out = QNN_COMMON_ERROR_SYSTEM;
            ggml_qnn_degrade(sess, why);
            return false;
        }
    }
    ggml_qnn_call_worker * w = sess->worker;

    std::unique_lock<std::mutex> lock(w->m);
    w->job      = std::move(call);
    w->has_job  = true;
    w->job_done = false;
    w->cv.notify_all();
    if (!w->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [w] { return w->job_done; })) {
        // detached under the lock, so the worker cannot delete its state before this returns
        w->lost = true;
        w->thread.detach();
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
    Qnn_ErrorHandle_t err = sess->iface.deviceGetInfrastructure(&infra);
    if (err != QNN_SUCCESS || !infra) {
        GGML_LOG_WARN("ggml-qnn: deviceGetInfrastructure failed: %" PRIu64 ", HTP stays at default clocks\n", (uint64_t) err);
        return;
    }

    QnnHtpDevice_Infrastructure_t * htp = (QnnHtpDevice_Infrastructure_t *) infra;
    if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        GGML_LOG_WARN("ggml-qnn: device infrastructure type %d is not PERF, HTP stays at default clocks\n", (int) htp->infraType);
        return;
    }
    QnnHtpDevice_PerfInfrastructure_t & perf = htp->perfInfra;
    if (!perf.createPowerConfigId || !perf.setPowerConfig) {
        GGML_LOG_WARN("ggml-qnn: perf infrastructure has no power config entry points, HTP stays at default clocks\n");
        return;
    }

    uint32_t power_config_id = 0;
    err = perf.createPowerConfigId(/*deviceId=*/0, /*coreId=*/0, &power_config_id);
    if (err != QNN_SUCCESS) {
        GGML_LOG_WARN("ggml-qnn: createPowerConfigId failed: %" PRIu64 ", HTP stays at default clocks\n", (uint64_t) err);
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
    // microseconds the HTP may take to wake from sleep: a small value keeps it responsive
    // between the short bursts of single-op graphs, at a small idle power cost
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
    err = perf.setPowerConfig(power_config_id, cfgs);
    if (err != QNN_SUCCESS) {
        GGML_LOG_WARN("ggml-qnn: setPowerConfig(TURBO) failed: %" PRIu64 ", HTP stays at default clocks\n", (uint64_t) err);
        return;
    }
    ggml_qnn_stat_burst_applied.fetch_add(1);
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
        return nullptr;
    }

#ifdef _WIN32
    // QAIRT 2.45 resolves the Hexagon-side skel from ADSP_LIBRARY_PATH; without it the first
    // execute dies silently instead of failing. WARN, because llama-cli and llama-server drop
    // GGML INFO at the default verbosity and every runtime this build loads is new enough to
    // need it
    if (!getenv("ADSP_LIBRARY_PATH")) {
        const char * sdk_root = getenv("QNN_SDK_ROOT");
        GGML_LOG_WARN("ggml-qnn: ADSP_LIBRARY_PATH is unset, QAIRT 2.45 needs it set to %s\\lib\\hexagon-v73\\unsigned\n",
                      sdk_root && *sdk_root ? sdk_root : "<QNN_SDK_ROOT>");
    }
#endif

    ggml_qnn_get_providers_fn_t get_providers =
        (ggml_qnn_get_providers_fn_t) ggml_qnn_dl_sym(lib, "QnnInterface_getProviders");
    if (!get_providers) {
        GGML_LOG_ERROR("ggml-qnn: QnnInterface_getProviders not found in QnnHtp library\n");
        ggml_qnn_dl_close(lib);
        return nullptr;
    }

    const QnnInterface_t ** providers = nullptr;
    uint32_t n_providers = 0;
    const Qnn_ErrorHandle_t perr = get_providers(&providers, &n_providers);
    if (perr != QNN_SUCCESS || !providers || n_providers == 0) {
        GGML_LOG_ERROR("ggml-qnn: failed to query QNN interface providers: %" PRIu64 " (%u providers)\n", (uint64_t) perr, n_providers);
        ggml_qnn_dl_close(lib);
        return nullptr;
    }

    const QNN_INTERFACE_VER_TYPE * iface = nullptr;
    std::string found;
    for (uint32_t i = 0; i < n_providers; i++) {
        const Qnn_Version_t & v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
            iface = &providers[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%s%u.%u", found.empty() ? "" : ", ", (unsigned) v.major, (unsigned) v.minor);
        found += buf;
    }
    if (!iface) {
        GGML_LOG_ERROR("ggml-qnn: no QNN interface provider with API %d.%d or newer, found %s\n",
                       QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR, found.c_str());
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

    Qnn_ErrorHandle_t err = sess->iface.backendCreate(sess->log_handle, nullptr, &sess->backend_handle);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: QnnBackend_create failed: %" PRIu64 "\n", (uint64_t) err);
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    err = sess->iface.deviceCreate(sess->log_handle, nullptr, &sess->device_handle);
    if (err != QNN_SUCCESS) {
        GGML_LOG_INFO("ggml-qnn: no HTP device available, QnnDevice_create failed: %" PRIu64 "\n", (uint64_t) err);
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    err = sess->iface.contextCreate(sess->backend_handle, sess->device_handle, nullptr, &sess->context_handle);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: QnnContext_create failed: %" PRIu64 "\n", (uint64_t) err);
        ggml_qnn_session_free(sess);
        return nullptr;
    }

    GGML_LOG_INFO("ggml-qnn: initialized Hexagon NPU (HTP)\n");

    // unlimited static pinning exhausts NPU mapped memory on full models, which can poison
    // the context, so cap it by default. 1024 as a margin: the HTP stopped mapping baked
    // weights at about 1170 MiB committed in one run on a machine loaded with other work
    // (QAIRT 2.45, X Elite, 2026-09-16), not reproduced on an idle one
    sess->static_budget = (size_t) ggml_qnn_env_ll("GGML_QNN_STATIC_BUDGET_MB", 1024, 0) * 1024 * 1024;
    if (sess->static_budget) {
        GGML_LOG_INFO("ggml-qnn: static-weight budget %zu MB\n", sess->static_budget / (1024 * 1024));
    }

    // probe fastrpc shared memory, used for registered graph IO buffers. the verdict gates
    // every graph's IO setup for the life of the session
    if (getenv("GGML_QNN_SHARED_MEM")) {
        sess->shared_mem_ok = ggml_qnn_mem_self_test(&sess->iface, sess->context_handle);
        ggml_qnn_stat_shm_selftest.store(sess->shared_mem_ok ? 2 : ggml_qnn_mem_lib_present() ? 1 : 0);
        if (!sess->shared_mem_ok) {
            GGML_LOG_INFO("ggml-qnn: fastrpc shared memory not available on this device\n");
        }
    }

    return sess;
}

static void ggml_qnn_graph_release_buffers(ggml_qnn_session * sess, ggml_qnn_graph & g, bool keep_bake);

void ggml_qnn_session_free(ggml_qnn_session * sess) {
    if (!sess) {
        return;
    }
    // a session that never got a context ran nothing, and all-zero counters would only
    // shadow the init failure that is the actual news
    if (sess->context_handle) {
        ggml_qnn_stats_write();
    }
    if (sess->worker) {
        // the session is freed only at refs == 0, so the worker is idle: tell it to quit and
        // join it before the context it would call into is freed
        ggml_qnn_call_worker * w = sess->worker;
        {
            std::lock_guard<std::mutex> lock(w->m);
            w->quit = true;
            w->cv.notify_all();
        }
        w->thread.join();
        delete w;
        sess->worker = nullptr;
    }
    // deregister + free the IO buffers while the context is still alive. a bake that an
    // unfinalized graph still references (finalize error) is kept until the context is freed
    // and dies with the session below
    for (auto & kv : sess->graphs) {
        ggml_qnn_graph_release_buffers(sess, kv.second, /*keep_bake=*/true);
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
            if (htp->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF && htp->perfInfra.destroyPowerConfigId) {
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

static bool ggml_qnn_static_weights_on(void) {
    static const bool on = getenv("GGML_QNN_NO_STATIC_WEIGHTS") == nullptr;
    return on;
}

bool ggml_qnn_quantized_dynamic_ok(void) {
    static const bool on = getenv("GGML_QNN_QUANTIZED") != nullptr;
    return on;
}

bool ggml_qnn_mul_mat_type_claimable(enum ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) {
        return true;
    }
    return (ggml_qnn_quantized_dynamic_ok() || ggml_qnn_static_weights_on()) &&
           ggml_is_quantized(type) && ggml_get_type_traits(type)->to_float != NULL;
}

// weights in a model buffer are baked into the graph as a static tensor: QNN converts them
// to the HTP-native layout once at finalize instead of on every execute, and a quantized
// source is dequantized to fp16 once at bake. the TURBO power config is applied for the
// same reason: both move work off the per-execute path. no figure is quoted here - the
// ones that were came from the single-matmul harness, not a model run, and a comment is
// the wrong place to carry a number nobody re-measures; the README and
// docs/backend/QNN.md hold the measurements and the retraction that scopes them.
// on by default, subject to the static budget; GGML_QNN_NO_STATIC_WEIGHTS disables

// a tensor in a WEIGHTS buffer is immutable unless it is a training parameter: the optimizer
// updates a GGML_TENSOR_FLAG_PARAM weight in place, so it must be copied on every execute. baked,
// it would get a new fingerprint, a new static graph and a new budget charge per optimizer step
static bool ggml_qnn_weight_is_const(const ggml_tensor * w) {
    // resolved through view_src like the scheduler and supports_op do: a view has no buffer of
    // its own until the graph is allocated, and the answer must not change between the two
    const ggml_backend_buffer_t buf = w->view_src ? w->view_src->buffer : w->buffer;
    const int32_t flags = w->view_src ? (w->flags | w->view_src->flags) : w->flags;
    return buf && buf->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

static bool ggml_qnn_weights_static(const ggml_tensor * node) {
    return ggml_qnn_static_weights_on() && node->op == GGML_OP_MUL_MAT && node->src[0]->data &&
           ggml_qnn_weight_is_const(node->src[0]) &&
           (!ggml_is_quantized(node->src[0]->type) || ggml_get_type_traits(node->src[0]->type)->to_float != NULL);
}

// on-device bytes of the baked static weight (always fp16 unless the source is fp32)
static size_t ggml_qnn_static_bytes(const ggml_tensor * w) {
    const size_t elem = w->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    return (size_t) ggml_nelements(w) * elem;
}

// matmul graphs are built with the batch dim padded up to a bucket, so llama probing N=512
// and then decoding N=60 reuses one graph and bakes each weight once, and a dynamic-weight
// matmul gets one graph per bucket instead of one per exact N
static uint32_t ggml_qnn_pad_n(uint32_t n) {
    static const uint32_t floor_n = (uint32_t) ggml_qnn_env_ll("GGML_QNN_NPAD", 512, 0);
    uint32_t p = floor_n ? floor_n : 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// 64-bit FNV-1a over the byte count and the first and last 512 bytes of a resident weight:
// content identity beside the address, so an address reused after a model unload does not
// serve the previous model's bake. the old bake stays charged to the static budget, because
// QNN keeps its graph until contextFree. it runs on every graph lookup, before the cache is
// consulted, so it mixes a 64-bit word per step and only the tail bytes singly
static uint64_t ggml_qnn_weight_fingerprint(const ggml_tensor * w) {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&h](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        size_t i = 0;
        for (; i + sizeof(uint64_t) <= n; i += sizeof(uint64_t)) {
            uint64_t v;
            memcpy(&v, b + i, sizeof(v));
            h ^= v;
            h *= 0x100000001b3ull;
            h ^= h >> 29; // a multiply only carries upward, fold the high bits back down
        }
        for (; i < n; i++) {
            h ^= b[i];
            h *= 0x100000001b3ull;
        }
    };
    const size_t nbytes = ggml_nbytes(w);
    mix(&nbytes, sizeof(nbytes));
    const size_t span = std::min<size_t>(512, nbytes);
    const uint8_t * data = (const uint8_t *) w->data;
    mix(data, span);
    mix(data + nbytes - span, span);
    return h;
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

    const Qnn_ErrorHandle_t err = sess->iface.tensorCreateGraphTensor(g.handle, &t);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to create graph tensor %s: %" PRIu64 "\n", name, (uint64_t) err);
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

    const Qnn_ErrorHandle_t err = sess->iface.graphAddNode(g.handle, op);
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to add %s node: %" PRIu64 "\n", type_name, (uint64_t) err);
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

// shape-only key: stable across runs, used for the failed-shape denylist. is_static picks the
// variant tag, which matters: static-baked and dynamic-weight graphs are different HTP
// programs with different finalize outcomes, a denylist entry must only ban the one that failed
// n_bucket overrides the batch bucket the key is built in, 0 means the node's own. the
// placement probe needs the override: llama hands weight_buft_supported a fictitious N, so a
// key built from it names a graph nothing ever runs, while the entries already in the denylist
// file were written in whatever bucket the real ubatch produced
static std::string ggml_qnn_shape_key(const ggml_tensor * node, bool is_static, uint32_t n_bucket = 0) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    // matmul graphs use the padded batch dim, so every N in a bucket shares one graph
    const int64_t ne11 = node->op == GGML_OP_MUL_MAT
        ? (int64_t) (n_bucket ? n_bucket : ggml_qnn_pad_n((uint32_t) src1->ne[1]))
        : src1->ne[1];

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%s_%s_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "_%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "%s",
             ggml_op_name(node->op), ggml_type_name(src0->type), ggml_type_name(src1->type),
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
             src1->ne[0], ne11, src1->ne[2], src1->ne[3],
             is_static ? "_s" : "_dyn");
    return buf;
}

static std::string ggml_qnn_shape_key(const ggml_tensor * node) {
    return ggml_qnn_shape_key(node, ggml_qnn_weights_static(node));
}

static std::string ggml_qnn_graph_key(const ggml_tensor * node) {
    std::string key = ggml_qnn_shape_key(node);
    if (ggml_qnn_weights_static(node)) {
        // one baked-weight graph per weight tensor: address plus content fingerprint
        char suffix[64];
        snprintf(suffix, sizeof(suffix), "_w%p_h%016llx", node->src[0]->data,
                 (unsigned long long) ggml_qnn_weight_fingerprint(node->src[0]));
        key += suffix;
    }
    return key;
}

// the IO-size cap, see ggml_qnn_graph::n_pad. logged once per shape key: the same shape
// comes back once per weight and per session, and DEBUG once is enough to explain a CPU
// placement
//
// the first refusal of the process is a WARN that names the way out: with -dev QNN at the
// default GGML_QNN_NPAD every weight of a 4B model is refused here, and at DEBUG alone the
// user who asked for the NPU got a CPU run with no line saying why
static bool ggml_qnn_io_capped(const ggml_tensor * node, uint32_t pad) {
    static const uint64_t cap = (uint64_t) ggml_qnn_env_ll("GGML_QNN_IO_MAX_KB", 1024, 1) * 1024;
    const uint64_t in_bytes  = (uint64_t) node->src[1]->ne[0] * pad * sizeof(float);
    const uint64_t out_bytes = (uint64_t) node->src[0]->ne[1] * pad * sizeof(float);
    if (std::max(in_bytes, out_bytes) < cap) {
        return false;
    }
    static std::mutex                       logged_mutex;
    static std::unordered_set<std::string>  logged;
    static bool                             warned = false;
    // the placement probe evaluates a pad that is not the node's own, so the pad is part of
    // the once-per-shape key
    const std::string shape_key = ggml_qnn_shape_key(node);
    std::lock_guard<std::mutex> lock(logged_mutex);
    if (logged.insert(shape_key + "@" + std::to_string(pad)).second) {
        GGML_LOG_DEBUG("ggml-qnn: %s: padded IO %" PRIu64 " in / %" PRIu64 " out bytes at N=%u reaches GGML_QNN_IO_MAX_KB, staying on the CPU\n",
                       shape_key.c_str(), in_bytes, out_bytes, pad);
    }
    if (!warned) {
        // the largest power-of-two batch whose padded IO still fits under the cap
        const uint64_t row_bytes = std::max<uint64_t>(node->src[1]->ne[0], node->src[0]->ne[1]) * sizeof(float);
        uint32_t fit = 0;
        for (uint32_t p = 1; (uint64_t) p * row_bytes < cap; p <<= 1) {
            fit = p;
        }
        // warn only when a setting can fix it. supports_op refuses batches under
        // GGML_QNN_MIN_DIM, so a shape that fits only below that (a 151936-wide output layer,
        // a 9728-wide FFN) never runs here at any setting: it stays at DEBUG and does not
        // spend the one WARN a fixable refusal needs
        static const uint32_t min_dim = (uint32_t) ggml_qnn_env_ll("GGML_QNN_MIN_DIM", 32, 1);
        if (fit >= min_dim) {
            warned = true;
            // straight to stderr, like the first degrade: llama-bench without -v installs a null
            // log callback and would report a CPU run under the QNN backend name with no hint
            fprintf(stderr, "ggml-qnn: a %s %" PRId64 "x%" PRId64 " matmul stays on the CPU: its padded IO is %.1f MiB at N=%u, at or over the %.1f MiB cap "
                            "(GGML_QNN_IO_MAX_KB); GGML_QNN_NPAD=%u with -ub %u (or lower) fits it. later IO-cap refusals are logged at DEBUG\n",
                    ggml_type_name(node->src[0]->type), node->src[0]->ne[0], node->src[0]->ne[1],
                    std::max(in_bytes, out_bytes) / (1024.0 * 1024.0), pad, cap / (1024.0 * 1024.0), fit, fit);
            fflush(stderr);
        }
    }
    return true;
}

bool ggml_qnn_mul_mat_arith_reject(const ggml_tensor * op, bool placement_probe) {
    const uint64_t K  = (uint64_t) op->src[0]->ne[0];
    const uint64_t M  = (uint64_t) op->src[0]->ne[1];
    const uint32_t Nb = ggml_qnn_pad_n(placement_probe ? 1 : (uint32_t) op->src[1]->ne[1]);

    // the padded IO buffer sizes must fit the uint32_t QNN takes
    if (K * Nb * sizeof(float) > UINT32_MAX || M * Nb * sizeof(float) > UINT32_MAX) {
        return true;
    }
    // a padded IO buffer at or past the cap hangs the execute (the padded-IO size law)
    return ggml_qnn_io_capped(op, Nb);
}

// policy checks that need no QNN graph. they must run BEFORE graphCreate: a rejected shape
// must not leave an unfinalized graph in the shared context, because its deferred prepare
// hangs the next graph's execute. fills the g fields build_mul_mat relies on
static bool ggml_qnn_mul_mat_policy(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    g.weights_static   = ggml_qnn_weights_static(node);
    g.weight_quantized = ggml_is_quantized(src0->type);
    g.n_pad            = ggml_qnn_pad_n((uint32_t) src1->ne[1]);

    // local arithmetic, not an HTP verdict: a policy reject, never a denylist entry. supports_op
    // asks the same function first, so a shape refused here was never claimed
    if (ggml_qnn_mul_mat_arith_reject(node, /*placement_probe=*/false)) {
        g.policy_reject = true;
        return false;
    }

    // static weights fit within the NPU memory budget, past it a weight stays on the CPU.
    // the budget is charged only after finalize succeeds
    if (g.weights_static) {
        // tracked in unlimited mode too, so a clamp has a meaningful committed value
        const size_t need = ggml_qnn_static_bytes(src0);
        if (sess->static_budget && ggml_qnn_static_committed.load() + need > sess->static_budget) {
            // the only env-fixable refusal in this file that used to emit nothing at any level,
            // and the default was halved to 1024 MB, so it is twice as easy to reach. same shape
            // of notice the IO cap gets: one DEBUG per shape, one stderr line per process,
            // because llama-bench without -v installs a null log callback and would otherwise
            // report a CPU run under the QNN backend name with no hint
            const size_t committed = ggml_qnn_static_committed.load();
            GGML_LOG_DEBUG("ggml-qnn: %s: static weight needs %.1f MiB, %.1f of %.1f MiB committed, staying on the CPU\n",
                           ggml_qnn_shape_key(node).c_str(), need / (1024.0 * 1024.0),
                           committed / (1024.0 * 1024.0), sess->static_budget / (1024.0 * 1024.0));
            static std::atomic<bool> budget_reported{false};
            if (!budget_reported.exchange(true)) {
                fprintf(stderr, "ggml-qnn: the %.0f MiB static-weight budget is full (%.1f MiB committed), so further weights stay on "
                                "the CPU; GGML_QNN_STATIC_BUDGET_MB raises it, 0 lifts it. later refusals are logged at DEBUG\n",
                        sess->static_budget / (1024.0 * 1024.0), committed / (1024.0 * 1024.0));
                fflush(stderr);
            }
            g.policy_reject = true;
            return false;
        }
        g.pending_static_bytes = need;
    }
    // a quantized weight that is not baked statically has no correct NPU path (the per-execute
    // dequant path is experimental and gated), so keep it on the CPU
    if (g.weight_quantized && !g.weights_static && !ggml_qnn_quantized_dynamic_ok()) {
        g.policy_reject = true;
        return false;
    }
    return true;
}

// stage the static weight in graph-owned memory, dequantizing a quantized source to fp16. QNN
// owns the converted copy from finalize on and the staging is freed; a wedged finalize leaks
// the staging instead of pointing into model memory. this is the one allocation of a build that
// realistically fails, so it runs BEFORE graphCreate: a host OOM then creates no QNN graph.
// returns the host-side cost in ms, throws std::bad_alloc
static double ggml_qnn_stage_bake(ggml_qnn_graph & g, const ggml_tensor * src0) {
    const auto t_bake = std::chrono::steady_clock::now();
    if (g.weight_quantized) {
        g.bake.resize((size_t) ggml_nelements(src0) * sizeof(ggml_fp16_t));
        ggml_qnn_dequant_f16(src0, (ggml_fp16_t *) g.bake.data());
    } else {
        g.bake.assign((const uint8_t *) src0->data, (const uint8_t *) src0->data + ggml_nbytes(src0));
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_bake).count();
}

// a static graph expects g.bake staged by ggml_qnn_stage_bake
static bool ggml_qnn_build_mul_mat(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];

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
        // from here on the unfinalized graph references g.bake through the STATIC tensor's
        // clientBuf, so the staging must outlive every failure path that keeps the handle
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

//
// failed-shape denylist
//

// shapes that failed or wedged the HTP, kept process-global so the knowledge survives the
// session teardown between llama's probe and context phases. GGML_QNN_DENYLIST names a file
// that persists it across runs, so a rerun after a wedge skips the bad shape entirely. the
// value records whether the entry came from the file (GGML_QNN_NO_OPT ignores those, so the
// lever can reach the shape it exists to debug) or from a failure in this process
static std::mutex                            ggml_qnn_denylist_mutex;
static std::unordered_map<std::string, bool> ggml_qnn_denylist; // value: from_file

// the configured file, copied once: an empty value is unset
static const std::string & ggml_qnn_denylist_path(void) {
    static const std::string path = [] {
        const char * v = getenv("GGML_QNN_DENYLIST");
        return std::string(v ? v : "");
    }();
    return path;
}

static void ggml_qnn_denylist_load_once(void) {
    static bool loaded = false;
    if (loaded) {
        return;
    }
    loaded = true;
    const std::string & path = ggml_qnn_denylist_path();
    if (path.empty()) {
        return;
    }
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        // a file that does not exist yet is the normal first run, anything else is a
        // configuration error worth one line
        if (errno != ENOENT) {
            GGML_LOG_ERROR("ggml-qnn: cannot read denylist %s: %s\n", path.c_str(), strerror(errno));
        }
        return;
    }
    char line[256];
    bool first = true;
    while (fgets(line, sizeof(line), f)) {
        char * s = line;
        if (first && strncmp(s, "\xEF\xBB\xBF", 3) == 0) {
            s += 3; // UTF-8 BOM
        }
        first = false;
        while (*s == ' ' || *s == '\t') {
            s++;
        }
        char * e = s + strlen(s);
        while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) {
            e--;
        }
        *e = 0;
        if (*s && *s != '#') {
            ggml_qnn_denylist.emplace(s, true);
        }
    }
    fclose(f);
    GGML_LOG_INFO("ggml-qnn: loaded %zu denylisted shapes from %s\n", ggml_qnn_denylist.size(), path.c_str());
}

static bool ggml_qnn_denylisted(const std::string & shape_key) {
    static const bool no_opt = getenv("GGML_QNN_NO_OPT") != nullptr;
    std::lock_guard<std::mutex> lock(ggml_qnn_denylist_mutex);
    ggml_qnn_denylist_load_once();
    auto it = ggml_qnn_denylist.find(shape_key);
    if (it == ggml_qnn_denylist.end()) {
        return false;
    }
    // under GGML_QNN_NO_OPT a file entry is advisory: the lever exists to rebuild the shape
    // that failed, entries from this process still apply
    return !(no_opt && it->second);
}

static void ggml_qnn_denylist_append(const std::string & path, const std::string & shape_key) {
    FILE * f = fopen(path.c_str(), "a+");
    if (!f) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_ERROR("ggml-qnn: cannot append to denylist %s: %s\n", path.c_str(), strerror(errno));
        }
        return;
    }
    bool ok = fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? ftell(f) : -1;
    ok = ok && size >= 0;
    if (ok && size == 0) {
        ok = fprintf(f, "# ggml-qnn denylist v1\n") > 0;
    } else if (ok) {
        // an earlier writer may have left the file without a trailing newline
        ok = fseek(f, -1, SEEK_END) == 0;
        const int last = ok ? fgetc(f) : '\n';
        ok = ok && fseek(f, 0, SEEK_END) == 0;
        if (ok && last != '\n') {
            ok = fputc('\n', f) != EOF;
        }
    }
    if (ok) {
        ok = fprintf(f, "%s\n", shape_key.c_str()) > 0;
    }
    if (fclose(f) != 0) {
        ok = false;
    }
    if (!ok) {
        GGML_LOG_ERROR("ggml-qnn: failed to write %s to denylist %s: %s\n", shape_key.c_str(), path.c_str(), strerror(errno));
    }
}

// persist only what a rerun must skip, i.e. an HTP verdict on the shape. the rule:
//  - a finalize error, and a finalize timeout, go to the file
//  - a validation-execute timeout goes to the file only if this session has already seen a
//    validation execute complete under GGML_QNN_SLOW_EXEC_MS (ggml_qnn_session::healthy_seen).
//    a timeout cannot tell a wedge from a device that is merely slow or a machine that is
//    busy, and either used to ban healthy shapes for every later run
//  - a compute-time execute timeout follows the same healthy_seen rule: with
//    GGML_QNN_SLOW_EXEC_MS=0 or GGML_QNN_NO_PREVALIDATE a graph can reach compute on a slow device
//  - no timeout goes to the file once the budget is clamped: a failed graph then sits in the
//    context and its deferred prepare can hang the next execute, whatever the shape
//  - a create/tensor/node failure or an execute error return can be a transient driver state
//    and is remembered for this process only
// everything not persisted is still kept in-process. a static shape that already built in
// this session is never added, see ggml_qnn_budget_clamp
static void ggml_qnn_denylist_add(const std::string & shape_key, bool persist) {
    std::lock_guard<std::mutex> lock(ggml_qnn_denylist_mutex);
    ggml_qnn_denylist_load_once();
    auto it = ggml_qnn_denylist.find(shape_key);
    if (it != ggml_qnn_denylist.end()) {
        // already in the file; a fresh failure makes it a this-process entry, which the
        // GGML_QNN_NO_OPT gate does not ignore
        it->second = false;
        return;
    }
    ggml_qnn_denylist.emplace(shape_key, false);
    const std::string & path = ggml_qnn_denylist_path();
    if (persist && !path.empty()) {
        ggml_qnn_denylist_append(path, shape_key);
    }
}

// placement_probe: the node carries llama's fictitious weight-probe batch (512), not a batch
// any graph runs in, so a key built from it matches nothing that was ever persisted - the
// entries on disk were written in the bucket the real ubatch produced. At GGML_QNN_NPAD=32
// with -ub 32, the documented route for K-quants, that is the smallest bucket, so the probe
// asks about it too. Both buckets are checked rather than just the small one: dropping the
// node's own would move the blind spot to a -ub 512 run, whose graphs are in the 512 bucket
bool ggml_qnn_shape_denylisted(const ggml_tensor * node, bool placement_probe) {
    if (ggml_qnn_denylisted(ggml_qnn_shape_key(node, false))) {
        return true;
    }
    if (node->op == GGML_OP_MUL_MAT && ggml_qnn_denylisted(ggml_qnn_shape_key(node, true))) {
        return true;
    }
    if (!placement_probe || node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const uint32_t small = ggml_qnn_pad_n(1);
    if (small == ggml_qnn_pad_n((uint32_t) node->src[1]->ne[1])) {
        return false; // the probe already landed in that bucket, asked above
    }
    return ggml_qnn_denylisted(ggml_qnn_shape_key(node, false, small)) ||
           ggml_qnn_denylisted(ggml_qnn_shape_key(node, true, small));
}

// serialize the finalized context to a binary, reload it into a fresh context, and time both,
// to see if a compiled-once/load-fast NPU path is worth building (the AOT context-binary roadmap)
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
// fastrpc memory when the session's init-time self-test passed, plain host memory otherwise
static bool ggml_qnn_graph_setup_io(ggml_qnn_session * sess, ggml_qnn_graph & g, const ggml_tensor * node) {
    const size_t n_in = g.inputs.size();

    if (sess->shared_mem_ok) {
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
        if (!sess->shm_fallback_warned) {
            sess->shm_fallback_warned = true;
            GGML_LOG_WARN("ggml-qnn: shared memory setup failed, using host buffers for this graph (reported once per session)\n");
        }
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

// keep_bake leaves the static staging alone: an unfinalized graph in the live context still
// references it through the STATIC tensor's clientBuf
static void ggml_qnn_graph_release_buffers(ggml_qnn_session * sess, ggml_qnn_graph & g, bool keep_bake) {
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
    if (!keep_bake) {
        std::vector<uint8_t>().swap(g.bake);
    }
}

//
// graph cache
//

// a static graph failed to build on a shape that already finalized and validated in this
// session, so the shape is fine and the device ran out of mappable weight memory. the codes
// seen there are not memory-specific (6020 at finalize, 6002 at the first execute, QAIRT
// 2.45), which is why the shape history decides and not the error. clamp the budget to what
// is committed so ggml_qnn_mul_mat_policy rejects every further static bake; 0 would mean
// unlimited, hence the floor of 1. every later bake is then refused by policy, so this runs
// at most once per session
static void ggml_qnn_budget_clamp(ggml_qnn_session * sess, const std::string & key, const char * phase, Qnn_ErrorHandle_t err) {
    const size_t committed = ggml_qnn_static_committed.load();
    const size_t clamp     = std::max<size_t>(committed, 1);
    if (!sess->static_budget || sess->static_budget > clamp) {
        sess->static_budget = clamp;
    }
    ggml_qnn_stat_budget_clamped.store(1);
    if (!sess->budget_clamped) {
        sess->budget_clamped = true;
        GGML_LOG_WARN("ggml-qnn: %s of %s failed (%" PRIu64 ") on a shape that built before: NPU weight memory is full at %.1f MiB committed, "
                      "further weights stay on the CPU (a lower GGML_QNN_STATIC_BUDGET_MB avoids the failed attempt)\n",
                      phase, key.c_str(), (uint64_t) err, committed / (1024.0 * 1024.0));
    }
}

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
        g.denylisted = true;
        return &g;
    }

    // policy rejections must not create a QNN graph at all
    if (node->op == GGML_OP_MUL_MAT && !ggml_qnn_mul_mat_policy(sess, g, node)) {
        return &g;
    }

    // nor must a host OOM on the bake staging, see ggml_qnn_stage_bake
    double bake_ms = 0.0;
    if (g.weights_static) {
        try {
            bake_ms = ggml_qnn_stage_bake(g, node->src[0]);
        } catch (const std::bad_alloc &) {
            GGML_LOG_ERROR("ggml-qnn: out of host memory staging the weight of %s\n", key.c_str());
            std::vector<uint8_t>().swap(g.bake);
            g.policy_reject = true; // an environment verdict, not an HTP one
            return &g;
        }
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

    const Qnn_ErrorHandle_t cerr = sess->iface.graphCreate(sess->context_handle, key.c_str(), graph_cfgs, &g.handle);
    if (cerr != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: failed to create graph %s: %" PRIu64 "\n", key.c_str(), (uint64_t) cerr);
        g.handle = nullptr;
        std::vector<uint8_t>().swap(g.bake); // no graph exists, nothing references the staging
        ggml_qnn_denylist_add(shape_key, /*persist=*/false);
        return &g;
    }
    ggml_qnn_stat_graphs_created.fetch_add(1);
    if (graph_cfgs == cfgs_mm_noopt) {
        ggml_qnn_stat_graphs_noopt.fetch_add(1);
    }
    if (node->op == GGML_OP_MUL_MAT) {
        ggml_qnn_stat_pad_n_last.store(g.n_pad);
    }

    bool ok        = true;
    bool timed_out = false;
    bool persist   = false; // denylist entry goes to the file, see ggml_qnn_denylist_add
    bool validated = false;
    bool charged   = false; // pending_static_bytes are on the budget
    bool executed  = false; // an execute reached the HTP, so a charged weight may be mapped

    // see ggml_qnn_budget_clamp. GGML_QNN_NO_PREVALIDATE never proves a shape, so it keeps
    // the plain failure handling
    const bool proven = g.weights_static && sess->proven_static.count(shape_key) != 0;

    double finalize_ms = 0.0;
    double validate_ms = 0.0;

    auto uncharge = [&]() {
        if (charged) {
            sess->static_bytes        -= g.pending_static_bytes;
            ggml_qnn_static_committed -= g.pending_static_bytes;
            charged = false;
        }
    };

    // the IO buffers (and small bookkeeping) can still fail to allocate; a throw here would
    // leave a handle-bearing, IO-less cache entry that later executes on nothing. the graph
    // created above stays in the context either way
    try {
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ok = ggml_qnn_build_mul_mat(sess, g, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                ok = ggml_qnn_build_binary(sess, g, node);
                break;
            default:
                GGML_ABORT("ggml-qnn: %s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }

        // experimental: measure whether an AOT context binary reloads faster than a fresh finalize
        const bool aot_test = ok && !sess->aot_tested && getenv("GGML_QNN_AOT_TEST");

        auto t_fin = std::chrono::steady_clock::now();
        if (ok) {
            // a shape that wedges the HTP hangs in finalize; the timeout makes supports_op
            // return false so the op is placed on the CPU before compute
            Qnn_GraphHandle_t        h   = g.handle;
            QNN_INTERFACE_VER_TYPE * ifp = &sess->iface;
            Qnn_ErrorHandle_t        fin = QNN_SUCCESS;
            // test-only fault hook, see qnn-lib.h: a finalize error without calling
            // graphFinalize, so the unfinalized graph stays in the context like a real one
            static const char *    ffail_sub  = getenv("GGML_QNN_FAIL_FINALIZE");
            static const long long ffail_skip = ggml_qnn_env_ll("GGML_QNN_FAIL_FINALIZE_SKIP", 0, 0);
            static long long       ffail_seen = 0;
            const bool finject = ffail_sub && *ffail_sub && key.find(ffail_sub) != std::string::npos && ffail_seen++ >= ffail_skip;
            bool completed = true;
            if (finject) {
                GGML_LOG_WARN("ggml-qnn: GGML_QNN_FAIL_FINALIZE matches %s, injecting a finalize failure\n", key.c_str());
                fin = QNN_GRAPH_ERROR_GENERAL;
            } else {
                completed = ggml_qnn_call_timed(sess, ggml_qnn_build_timeout_ms(),
                    [ifp, h]() { return ifp->graphFinalize(h, nullptr, nullptr); }, &fin);
            }
            if (!completed && fin != QNN_SUCCESS) {
                // no watchdog thread, so finalize never ran and the session is already
                // degraded: an environment verdict, the shape stays off the denylist
                g.policy_reject = true;
                ok              = false;
            } else if (!completed) {
                GGML_LOG_ERROR("ggml-qnn: build: graph finalize for %s timed out at %lld ms (GGML_QNN_BUILD_TIMEOUT_MS raises the limit), treating the HTP as wedged\n",
                               key.c_str(), ggml_qnn_build_timeout_ms());
                ggml_qnn_degrade(sess, "finalize timeout");
                timed_out = true;
                persist   = !sess->budget_clamped; // see ggml_qnn_denylist_add
                ok        = false;
            } else if (fin != QNN_SUCCESS) {
                // the unfinalized graph stays in the live context and still references the
                // STATIC tensor's clientBuf: the tail keeps the staging
                ok = false;
                if (proven) {
                    ggml_qnn_budget_clamp(sess, key, "finalize", fin);
                    g.policy_reject = true; // keeps the shape off the denylist, in-process and file
                } else {
                    GGML_LOG_ERROR("ggml-qnn: failed to finalize graph %s: %" PRIu64 "\n", key.c_str(), (uint64_t) fin);
                    persist = true;
                }
            }
        }
        finalize_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fin).count();

        if (ok && g.weights_static) {
            sess->static_bytes += g.pending_static_bytes;
            ggml_qnn_static_committed += g.pending_static_bytes;
            charged = true;
            // QNN owns the HTP-layout copy from finalize on, drop the host staging
            std::vector<uint8_t>().swap(g.bake);
        }

        if (aot_test && ok) {
            sess->aot_tested = true;
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
        // batch later. an execute failure can poison the shared context, so it degrades the
        // session, unless the shape is proven and the failure is weight memory running out
        if (ok && !getenv("GGML_QNN_NO_PREVALIDATE")) {
            Qnn_GraphHandle_t        h     = g.handle;
            Qnn_Tensor_t *           ins   = g.inputs.data();
            uint32_t                 n_ins = (uint32_t) g.inputs.size();
            Qnn_Tensor_t *           out   = &g.output;
            QNN_INTERFACE_VER_TYPE * ifp   = &sess->iface;
            Qnn_ErrorHandle_t        err   = QNN_SUCCESS;
            bool                     completed = true;
            // test-only fault hook, see qnn-lib.h: fail before QNN is called. the match
            // count is guarded by the session mutex like the rest of this function
            static const char *    fail_sub  = getenv("GGML_QNN_FAIL_EXECUTE");
            static const long long fail_skip = ggml_qnn_env_ll("GGML_QNN_FAIL_EXECUTE_SKIP", 0, 0);
            static long long       fail_seen = 0;
            const bool inject = fail_sub && *fail_sub && key.find(fail_sub) != std::string::npos && fail_seen++ >= fail_skip;
            const auto t_val = std::chrono::steady_clock::now();
            if (inject) {
                GGML_LOG_WARN("ggml-qnn: GGML_QNN_FAIL_EXECUTE matches %s, injecting an execute failure\n", key.c_str());
                err = QNN_COMMON_ERROR_SYSTEM;
            } else {
                // test-only, see qnn-lib.h: the delay runs inside the timed call, and the
                // lambda captures values only, it may outlive an abandoned call
                const long long delay_ms = ggml_qnn_test_delay_ms([&key]() { return key; });
                // same limit as a compute-time execute, see ggml_qnn_call_worker
                completed = ggml_qnn_call_timed(sess, ggml_qnn_compute_timeout_ms(),
                    [ifp, h, ins, n_ins, out, delay_ms]() {
                        if (delay_ms > 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                        }
                        return ifp->graphExecute(h, ins, n_ins, out, 1, nullptr, nullptr);
                    }, &err);
            }
            validate_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_val).count();
            // false only when no watchdog thread could be started, see ggml_qnn_call_timed
            executed = completed || err == QNN_SUCCESS;
            if (!executed) {
                // the session is already degraded: an environment verdict, no denylist entry
                g.policy_reject = true;
                ok              = false;
            } else if (!completed) {
                GGML_LOG_ERROR("ggml-qnn: build: validation execute for %s timed out at %lld ms, treating the HTP as wedged. if the device is only slow, "
                               "GGML_QNN_TIMEOUT_MS raises this limit, and GGML_QNN_SLOW_EXEC_MS must then be raised as well or set to 0, "
                               "or the completed validation is refused as too slow\n",
                               key.c_str(), ggml_qnn_compute_timeout_ms());
                ggml_qnn_degrade(sess, "validation execute timeout");
                timed_out = true;
                // only a session that has seen the device execute at normal speed can call
                // this a verdict on the shape, see ggml_qnn_denylist_add
                persist   = sess->healthy_seen && !sess->budget_clamped;
                ok        = false;
            } else if (err != QNN_SUCCESS && proven) {
                // the weight did not fit. the session keeps the graphs that did, they are
                // worth more than the one that did not, so no degrade. the weight maps on the
                // first execute, so it never reached the device: uncharge it, the clamp is
                // then the bytes that work
                uncharge();
                ggml_qnn_budget_clamp(sess, key, "validation execute", err);
                g.policy_reject = true;
                ok              = false;
            } else if (err != QNN_SUCCESS) {
                GGML_LOG_ERROR("ggml-qnn: validation execute failed for %s: %" PRIu64 "\n", key.c_str(), (uint64_t) err);
                ggml_qnn_degrade(sess, "validation execute failed");
                ok = false;
            } else if (!inject && ggml_qnn_slow_exec_ms() > 0 && validate_ms >= (double) ggml_qnn_slow_exec_ms()) {
                // a healthy HTP executes any shape under the IO cap in ms. seconds means the
                // device is running far below normal speed or the machine is busy. either is
                // a condition, not a verdict on the shape: no denylist, and the whole session
                // falls back to the CPU
                GGML_LOG_WARN("ggml-qnn: validation execute for %s took %.0f ms, the NPU is running far below normal speed or the machine is busy; staying on the CPU (GGML_QNN_SLOW_EXEC_MS=0 disables this check)\n",
                              key.c_str(), validate_ms);
                // the graphs claimed so far still execute correctly, so nodes already placed
                // on them keep running instead of failing the batch, see slow_only
                ggml_qnn_degrade(sess, "NPU too slow", /*slow_only=*/true);
                g.policy_reject = true;
                ok              = false;
            } else {
                validated = true;
                // with the slow check disabled the default limit still decides "normal speed"
                const long long slow_ms = ggml_qnn_slow_exec_ms() > 0 ? ggml_qnn_slow_exec_ms() : 2000;
                if (!inject && validate_ms < (double) slow_ms) {
                    sess->healthy_seen = true;
                }
            }
        }
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR("ggml-qnn: out of host memory building graph %s\n", key.c_str());
        g.policy_reject = true;
        ok              = false;
        timed_out       = false;
    }

    // a weight charged after finalize maps on the first execute: when the build failed before
    // any execute reached the HTP (IO setup, host OOM) the budget must not keep counting it
    if (!ok && !executed) {
        uncharge();
    }

    if (ok) {
        if (g.weights_static && validated) {
            sess->proven_static.insert(shape_key);
        }
        GGML_LOG_DEBUG("ggml-qnn: built graph %s%s%s: bake %.1f ms, finalize %.1f ms, validation execute %.1f ms\n", key.c_str(),
                       g.shared_mem ? " (shared mem)" : "", g.weights_static ? " (static weights)" : "",
                       bake_ms, finalize_ms, validate_ms);
    } else {
        // negative cache: remember the failure, QNN graphs live until the context is freed.
        // after a timeout the abandoned driver call may still touch the buffers, so leak them
        if (!g.policy_reject) {
            ggml_qnn_denylist_add(shape_key, persist);
        }
        if (!timed_out) {
            // the QNN graph outlives this failure in the context. if it is unfinalized and its
            // STATIC tensor was created against the staging, it still references it through
            // clientBuf, so a staging that exists beside a handle is kept until the session
            // goes away: finalize error, a tensor or node failure after the STATIC tensor, a
            // host OOM. after a successful finalize the staging is already gone
            const bool keep_bake = g.handle != nullptr && !g.bake.empty();
            ggml_qnn_graph_release_buffers(sess, g, keep_bake);
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

    if (sess->degraded.load()) {
        return false;
    }

    ggml_qnn_graph * g = ggml_qnn_get_graph(sess, node);
    return g && g->handle != nullptr;
}

enum ggml_status ggml_qnn_compute_node(ggml_qnn_session * sess, struct ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(sess->mutex);

    // once the session degraded (a call wedged the HTP), fail fast so we do not re-hang on every
    // remaining op already placed on this backend in the current batch. a session degraded
    // only for being slow still executes correctly: supports_op claims nothing more, and the
    // nodes placed before the verdict run instead of failing the batch
    if (sess->degraded.load() && !sess->slow_only) {
        return GGML_STATUS_FAILED;
    }
    // a slow session runs what it already built and builds nothing new
    if (sess->slow_only && sess->graphs.find(ggml_qnn_graph_key(node)) == sess->graphs.end()) {
        return GGML_STATUS_FAILED;
    }

    ggml_qnn_graph * g = ggml_qnn_get_graph(sess, node);
    if (!g || !g->handle) {
        if (g && !g->warned) {
            const char * why = g->denylisted    ? "the shape is denylisted"
                             : g->policy_reject ? "policy declined the graph"
                                                : "the graph for this shape did not build";
            GGML_LOG_WARN("ggml-qnn: failing %s, %s\n", ggml_op_desc(node), why);
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
            // a model weight is immutable, so it is copied once per graph, not per weight
            // execute: address and content fingerprint identify the copy it holds. a training
            // parameter is not immutable and is copied every time, see ggml_qnn_weight_is_const
            const bool is_const = ggml_qnn_weight_is_const(w);
            const uint64_t fp = is_const ? ggml_qnn_weight_fingerprint(w) : 0;
            if (is_const && g->weight_cached_ptr == w->data && g->weight_cached_hash == fp) {
                break;
            }
            if (g->weight_quantized) {
                ggml_qnn_dequant_f16(w, (ggml_fp16_t *) ggml_qnn_input_ptr(g, 1));
            } else {
                memcpy(ggml_qnn_input_ptr(g, 1), w->data, ggml_nbytes(w));
            }
            g->weight_cached_ptr  = is_const ? w->data : nullptr;
            g->weight_cached_hash = fp;
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
    const long long limit_ms = ggml_qnn_compute_timeout_ms();
    // test-only, see qnn-lib.h and the validation execute in ggml_qnn_get_graph
    const long long delay_ms = ggml_qnn_test_delay_ms([node]() { return ggml_qnn_graph_key(node); });
    const auto t_exec = std::chrono::steady_clock::now();
    const bool completed = ggml_qnn_call_timed(sess, limit_ms,
        [ifp, h, ins, n_ins, out, delay_ms]() {
            if (delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }
            return ifp->graphExecute(h, ins, n_ins, out, 1, nullptr, nullptr);
        }, &err);
    const uint64_t exec_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_exec).count();
    ggml_qnn_stat_exec_count.fetch_add(1);
    if (exec_ms > ggml_qnn_stat_exec_max_ms.load()) {
        ggml_qnn_stat_exec_max_ms.store(exec_ms); // under sess->mutex, no race
    }
    if (exec_ms >= 1000) {
        ggml_qnn_stat_exec_slow.fetch_add(1);
        // the shape key, not the graph key: that one re-hashes the weight just to be logged
        GGML_LOG_DEBUG("ggml-qnn: slow execute, %s took %" PRIu64 " ms\n", ggml_qnn_shape_key(node, g->weights_static).c_str(), exec_ms);
    }
    if (!completed && err != QNN_SUCCESS) {
        // no watchdog thread could be started, the execute never ran and the session is
        // already degraded, see ggml_qnn_call_timed. not a verdict on the shape
        return GGML_STATUS_FAILED;
    }
    if (!completed) {
        GGML_LOG_ERROR("ggml-qnn: compute: graph execute for %s timed out at %lld ms (GGML_QNN_TIMEOUT_MS raises the limit), treating the HTP as wedged\n",
                       ggml_op_desc(node), limit_ms);
        // after a budget clamp a failed graph sits in the context, so a timeout is not a
        // verdict on this shape: keep it out of the denylist file
        ggml_qnn_denylist_add(ggml_qnn_shape_key(node), /*persist=*/sess->healthy_seen && !sess->budget_clamped);
        g->handle = nullptr;
        ggml_qnn_degrade(sess, "execute timeout");
        return GGML_STATUS_FAILED;
    }
    if (err != QNN_SUCCESS) {
        GGML_LOG_ERROR("ggml-qnn: graph execute failed for %s: %" PRIu64 "\n", ggml_op_desc(node), (uint64_t) err);
        // a failed execute (deferred prepare, device memory) can corrupt the shared HTP context,
        // so demote this graph and degrade the whole session to the CPU for what follows. an
        // error return is not an HTP verdict on the shape, so it stays in-process
        ggml_qnn_denylist_add(ggml_qnn_shape_key(node), /*persist=*/false);
        g->handle = nullptr;
        ggml_qnn_degrade(sess, "execute failed");
        return GGML_STATUS_FAILED;
    }

    memcpy(node->data, ggml_qnn_output_ptr(g), ggml_nbytes(node));

    // the slow-device check of the validation execute, for a device that slows down after its
    // graphs validated (a cache hit never re-validates). the result above is valid, so this
    // node succeeds; the session stops claiming ops and what is already placed still runs
    if (ggml_qnn_slow_exec_ms() > 0 && exec_ms >= (uint64_t) ggml_qnn_slow_exec_ms() && !sess->degraded.load()) {
        GGML_LOG_WARN("ggml-qnn: execute for %s took %" PRIu64 " ms, the NPU is running far below normal speed or the machine is busy; staying on the CPU (GGML_QNN_SLOW_EXEC_MS=0 disables this check)\n",
                      ggml_op_desc(node), exec_ms);
        ggml_qnn_degrade(sess, "NPU too slow", /*slow_only=*/true);
    }

    return GGML_STATUS_SUCCESS;
}
