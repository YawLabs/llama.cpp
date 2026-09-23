// Lifecycle and correctness tests for the QNN (Hexagon NPU) backend, driven through the
// public ggml-backend API. Needs real HTP hardware (Windows on Snapdragon) and QnnHtp.dll
// resolvable at run time; exits 77 (the ctest SKIP_RETURN_CODE) with a skip message anywhere else.
// GGML_QNN_TEST_REQUIRE_HTP=1 turns that skip into a failure (exit 1): set it on a box that is
// known to have an HTP, where a device or context that failed to create would otherwise turn
// the whole suite into skips and ctest would still exit 0.
//
// The backend latches its env vars at first use, so each scenario must be its own process
// (the mode argument) and sets its env itself before touching the backend registry. main first
// unsets every GGML_QNN_* variable the backend reads, so a shell that exports GGML_QNN_NPAD,
// GGML_QNN_DISABLE or a denylist does not change a verdict. what survives: the variable a ctest
// ENVIRONMENT property sets for the variant, GGML_QNN_DEBUG (log verbosity only), and for the
// bigstatic and health modes the caller's GGML_QNN_TIMEOUT_MS, GGML_QNN_BUILD_TIMEOUT_MS and
// GGML_QNN_SLOW_EXEC_MS (bigstatic also keeps GGML_QNN_NPAD and GGML_QNN_IO_MAX_KB):
//
//   test-qnn-lifecycle basic      correctness of the static-weight path vs the CPU backend
//                                 (f32/f16/quantized weights, N below/at/above the pad
//                                 bucket), probe/free/reacquire cycle, clean exit
//   test-qnn-lifecycle budget     a tiny budget refuses a weight as a policy reject and does
//                                 NOT denylist it; committed bytes return on session free;
//                                 a quantized weight is charged at its fp16 on-device size
//   test-qnn-lifecycle denylist [noopt]  a seeded static-variant entry blocks that shape; a
//                                 dynamic-variant entry does not block the static path; the
//                                 unallocated (load-time) probe is refused for both variants.
//                                 "noopt" sets GGML_QNN_NO_OPT, under which file entries are
//                                 advisory: the seeded shape is claimed and so are those probes
//   test-qnn-lifecycle watchdog   a 1 ms finalize timeout (GGML_QNN_BUILD_TIMEOUT_MS bounds graph
//                                 finalize only; every execute, the validation one included,
//                                 runs under GGML_QNN_TIMEOUT_MS) degrades the
//                                 session; claims stop, backend init keeps succeeding,
//                                 the timed-out shape round-trips
//                                 through the denylist file. prints WATCHDOG-CHECKS-PASSED and
//                                 ends the process without running DLL detach, because the
//                                 leaked degraded session can crash there - the runner keys
//                                 off the marker
//   test-qnn-lifecycle fault      GGML_QNN_FAIL_EXECUTE injects a validation-execute failure:
//                                 the shape is refused, the session degrades and refuses the
//                                 next shape without a trial build, the verdict is remembered
//                                 in-process only (never persisted), init still succeeds
//   test-qnn-lifecycle clamp [unlimited]  the same injected failure on a shape that ALREADY
//                                 built in the session (GGML_QNN_FAIL_EXECUTE_SKIP=1) is NPU weight
//                                 memory running out, not a shape verdict: the static budget
//                                 clamps, further bakes are refused by policy, nothing is
//                                 denylisted and the session does NOT degrade; a fresh session
//                                 afterwards gets its whole budget back. "unlimited" runs it at
//                                 GGML_QNN_STATIC_BUDGET_MB=0, where the clamp must still bite
//   test-qnn-lifecycle bigstatic [i]   diagnostic, NOT a gate (no ctest entry): 512-class
//                                 static bakes that discriminate the padded-IO-size hang
//                                 thresholds (GGML_QNN_IO_MAX_KB is lifted so they can reach
//                                 them). optional case index i runs a single case, for
//                                 order/isolation permutations
//   test-qnn-lifecycle modelscale [npad] [env]   green guard for the WORKING side of the
//                                 IO-size law: model-scale static bake at a small pad bucket
//                                 plus bucket-boundary correctness; npad defaults to 64, "0"
//                                 pins the exact-pow2 bucket branch. the optional env names
//                                 the variable a ctest variant sets: GGML_QNN_SHARED_MEM (the
//                                 rpcmem IO path), GGML_QNN_NO_OPT, GGML_QNN_NO_STATIC_WEIGHTS
//                                 (dynamic-weight graphs, bucket-keyed), GGML_QNN_NO_BURST (no
//                                 TURBO power vote: burst_applied must stay 0)
//   test-qnn-lifecycle disable    GGML_QNN_DISABLE=1 must remove the device entirely (the
//                                 A/B kill-switch; vacuous pass on machines with no HTP, except
//                                 under GGML_QNN_TEST_REQUIRE_HTP=1, which first loads QnnHtp
//                                 directly and fails if the runtime itself is not loadable)
//   test-qnn-lifecycle mindim     at DEFAULT env the min-dim gate refuses small matmuls
//                                 while claiming normal ones; the default of 32 is pinned
//   test-qnn-lifecycle elementwise  at DEFAULT env ADD/MUL must be refused (the HTP has a
//                                 known broadcast bug), while mul_mat is still claimed
//   test-qnn-lifecycle elementwise-on  with GGML_QNN_ELEMENTWISE=1 the GGML_QNN_MIN_ELEMENTS gate
//                                 still refuses an ADD below it, and one at the threshold
//                                 reaches the HTP trial without degrading the session; if the
//                                 trial claims it, its computed result must match the CPU
//   test-qnn-lifecycle loadprobe  supports_op must give the same verdict for a weight probed
//                                 unallocated (data == NULL) as for the resident weight it
//                                 becomes: the loader's dummy-buffer probe against the
//                                 WEIGHTS-tagged weight, and a buffer-less graph tensor
//                                 against the same weight resident in an untagged buffer.
//                                 the padded-IO cap is part of that answer: a capped
//                                 buffer-less matmul and a weight capped at the smallest
//                                 bucket are refused, never claimed and failed at compute.
//                                 also the refusals decided by form alone (3D, strided, BF16
//                                 weight, F16 activation) and by buffer (a weight in CPU_REPACK,
//                                 directly and through a reshape; supports_buft)
//   test-qnn-lifecycle health     device health, not backend logic: one model-scale matmul must
//                                 execute in under a second. the only entry that depends on
//                                 device speed. every mode, this one included, runs with
//                                 GGML_QNN_SLOW_EXEC_MS=0 unless the caller set it: health
//                                 measures the execute instead of letting the backend refuse it
//   test-qnn-lifecycle rebake     one weight probed at two N in the same pad bucket bakes
//                                 once: the budget is sized so a per-N re-bake regression
//                                 (the original NPU-memory-exhaustion failure) fails the test
//
// The modes below run small F32 shapes at GGML_QNN_NPAD=64 (fast on any device) unless noted:
//
//   test-qnn-lifecycle reuse      two backends share one session (the second computes on the
//                                 first one's cached graph after the first is freed), a fresh
//                                 session after the last free, a weight in a buffer_from_host_ptr
//                                 buffer bakes, and new content at the same weight address bakes
//                                 again instead of serving the stale graph
//   test-qnn-lifecycle dyncache   GGML_QNN_NO_STATIC_WEIGHTS: the dynamic path copies a weight
//                                 once and skips the copy while address and fingerprint hold,
//                                 and re-copies new content at the same address
//   test-qnn-lifecycle quantized  GGML_QNN_QUANTIZED: an untagged Q4_0 weight (fp16 on the
//                                 device, one small graph) is claimed on the per-execute dequant
//                                 path and matches the CPU
//   test-qnn-lifecycle envparse   malformed GGML_QNN_NPAD, GGML_QNN_MIN_DIM and
//                                 GGML_QNN_IO_MAX_KB values fall back to their defaults
//
// The modes below drive the test-only hooks GGML_QNN_DELAY_EXECUTE and GGML_QNN_FAIL_FINALIZE
// (qnn-lib.h) on small F32 shapes at GGML_QNN_NPAD=64, so their thresholds do not depend on
// device speed. Those marked [marker] keep a degraded session, print <TAG>-CHECKS-PASSED and
// end through hard_exit like watchdog:
//
//   test-qnn-lifecycle slow-validate  [marker SLOW-VALIDATE] a validation execute delayed past
//                                 GGML_QNN_SLOW_EXEC_MS completes: the shape is not claimed and
//                                 not denylisted, the session degrades slow_only, a new shape is
//                                 refused, the graph built before still computes
//   test-qnn-lifecycle slow-compute  [marker SLOW-COMPUTE] with GGML_QNN_NO_PREVALIDATE a first
//                                 compute delayed past GGML_QNN_SLOW_EXEC_MS still succeeds and
//                                 degrades slow_only; the cached graph keeps computing, a new
//                                 shape fails, and the kept-degraded release flushes exec_count
//   test-qnn-lifecycle validate-timeout [cold]  [marker VALIDATE-TIMEOUT] a validation execute
//                                 delayed past GGML_QNN_TIMEOUT_MS degrades the session; the shape
//                                 reaches the denylist file only after a fast validation earlier
//                                 in the session, never in the cold leg
//   test-qnn-lifecycle compute-timeout  [marker COMPUTE-TIMEOUT] with GGML_QNN_NO_PREVALIDATE a
//                                 compute delayed past GGML_QNN_TIMEOUT_MS fails the node and
//                                 hard-degrades the session; not persisted
//   test-qnn-lifecycle finalize-error  a finalize error on an unproven shape is persisted and
//                                 refused; on a proven shape it clamps the budget instead
//   test-qnn-lifecycle denylist-append  the backend appends to a non-empty denylist file that
//                                 lacks a trailing newline: newline repaired, no header
//
// Checks that exercise the backend are counted (check); checks that only prove the test's
// own wiring (the CPU backend, an env var the runner must set, a path main configured) are
// GGML_ASSERTs, so the counted total is the number of backend assertions.
//
// NOTE: run modes one at a time, and never concurrently with another NPU-using process (the
// HTP is single-client and can wedge).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

static int g_checks = 0;
static int g_failures = 0;

// set by modes whose ctest variant differs ONLY by an ENVIRONMENT property
static const char * g_require_env = nullptr;

// GGML_QNN_STATS counters, written by the backend when the session is freed and on every
// degrade. Several ctest variants differ from their parent only by an env var, so without
// these they could not observe whether that env var changed anything and could not fail
// when it silently did not.
static const char * g_stats_path = "test-qnn-lifecycle-stats.tmp";
static bool         g_stats_on   = false;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// _putenv_s with an empty value removes the variable, which is what getenv-presence switches need
static void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static void enable_stats(void) {
    remove(g_stats_path);
    set_env("GGML_QNN_STATS", g_stats_path);
    g_stats_on = true;
}

static void check(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        g_failures++;
    }
    printf("  %s  %s\n", ok ? "OK  " : "FAIL", what);
    fflush(stdout);
}

// end the process without running static destructors or DLL detach: a session the backend
// deliberately kept (degraded) is still alive, and QnnHtp has been seen crashing in its
// detach with such a session - a ctest "Exception" that PASS_REGULAR_EXPRESSION cannot
// rescue and that would fail a run whose checks all passed. stdout is flushed first because
// nothing after this point runs
static void hard_exit(int rc) {
    fflush(stdout);
    fflush(stderr);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), (UINT) rc);
#else
    _exit(rc);
#endif
}

static void fill_uniform(std::vector<float> & v, unsigned seed) {
    // deterministic LCG, values in [-1, 1]
    unsigned s = seed * 2654435761u + 12345u;
    for (auto & x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float) ((double) (s >> 8) / (double) (1u << 24)) * 2.0f - 1.0f;
    }
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = a[i] - b[i];
        err += d * d;
        ref += (double) a[i] * a[i];
    }
    return ref > 0.0 ? err / ref : err;
}

struct mul_mat_case {
    ggml_type wtype;
    int64_t   K, M, N;
};

// the backend's batch-dim bucket rule (ggml_qnn_pad_n): the GGML_QNN_NPAD floor doubled until
// it covers N; a floor of 0 means the exact power of two
static int64_t pad_n_for(int64_t npad, int64_t n) {
    int64_t p = npad ? npad : 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// the backend's shape-only key (ggml_qnn_shape_key) for a 2D mul_mat with f32 activations:
// what the denylist file stores, and what GGML_QNN_FAIL_EXECUTE matches against
static std::string shape_key(const mul_mat_case & c, int64_t npad, bool is_static) {
    char buf[160];
    snprintf(buf, sizeof(buf), "MUL_MAT_%s_f32_%" PRId64 "x%" PRId64 "x1x1_%" PRId64 "x%" PRId64 "x1x1_%s",
             ggml_type_name(c.wtype), c.K, c.M, c.K, pad_n_for(npad, c.N), is_static ? "s" : "dyn");
    return buf;
}

// dst(N,M) = src1(N,K) x src0(M,K)^T computed on one backend, weights in a buffer tagged
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS so the QNN static-bake path triggers. tag_weights = false
// leaves that buffer untagged, which is where a graph tensor lands and where no bake can happen.
// run_mul_mat builds, computes and frees one; held_mul_mat keeps one alive between calls, so
// its weight keeps its address and content and a static graph built for it is found again in
// the cache by a later compute
struct held_mul_mat {
    mul_mat_case          c     = {};
    ggml_context *        ctx_w = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_tensor *         dst   = nullptr;
    ggml_cgraph *         gf    = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
};

static void held_init(held_mul_mat & h, ggml_backend_t backend, const mul_mat_case & c, bool tag_weights = true) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };

    h.c = c;
    ggml_context * ctx_w = h.ctx_w = ggml_init(wp);
    ggml_context * ctx   = h.ctx   = ggml_init(gp);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, c.wtype, c.K, c.M);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.N);
    ggml_tensor * dst = h.dst = ggml_mul_mat(ctx, w, x);
    ggml_cgraph  * gf = h.gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf_w = h.buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (tag_weights) {
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    h.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // weight data: f32 source, quantized in one chunk when the type needs it
    std::vector<float> wf((size_t) c.K * c.M);
    fill_uniform(wf, (unsigned) (c.K + c.M));
    if (c.wtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, wf.data(), 0, wf.size() * sizeof(float));
    } else if (c.wtype == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> wh(wf.size());
        ggml_fp32_to_fp16_row(wf.data(), wh.data(), (int64_t) wf.size());
        ggml_backend_tensor_set(w, wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(c.wtype, wf.data(), q.data(), 0, c.M, c.K, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }

    std::vector<float> xf((size_t) c.K * c.N);
    fill_uniform(xf, (unsigned) (c.K * 31 + c.N));
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
}

static bool held_claimed(ggml_backend_t backend, const held_mul_mat & h) {
    return ggml_backend_dev_supports_op(ggml_backend_get_device(backend), h.dst);
}

// computes whether or not the backend claimed the op: callers that need the claim ask first
static bool held_compute(ggml_backend_t backend, held_mul_mat & h, std::vector<float> & out) {
    const bool ok = ggml_backend_graph_compute(backend, h.gf) == GGML_STATUS_SUCCESS;
    out.resize((size_t) h.c.N * h.c.M);
    if (ok) {
        ggml_backend_tensor_get(h.dst, out.data(), 0, out.size() * sizeof(float));
    }
    return ok;
}

static void held_free(held_mul_mat & h) {
    ggml_backend_buffer_free(h.buf);
    ggml_backend_buffer_free(h.buf_w);
    ggml_free(h.ctx);
    ggml_free(h.ctx_w);
    h = held_mul_mat();
}

// exact F32 reference, double accumulation: dst(N,M) = x(N,K) x w(M,K)^T
static std::vector<float> ref_mul_mat(const std::vector<float> & w, const std::vector<float> & x, int64_t K, int64_t M, int64_t N) {
    std::vector<float> out((size_t) N * M);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; k++) {
                acc += (double) w[(size_t) (m * K + k)] * x[(size_t) (n * K + k)];
            }
            out[(size_t) (n * M + m)] = (float) acc;
        }
    }
    return out;
}

// the F32 weight a held matmul holds right now
static std::vector<float> held_weight(const held_mul_mat & h) {
    GGML_ASSERT(h.c.wtype == GGML_TYPE_F32);
    std::vector<float> w((size_t) h.c.K * h.c.M);
    ggml_backend_tensor_get(h.dst->src[0], w.data(), 0, w.size() * sizeof(float));
    return w;
}

// overwrite the held F32 weight in place: same tensor, same address, new bytes
static void held_set_weight(held_mul_mat & h, const std::vector<float> & w) {
    GGML_ASSERT(h.c.wtype == GGML_TYPE_F32 && w.size() == (size_t) h.c.K * h.c.M);
    ggml_backend_tensor_set(h.dst->src[0], w.data(), 0, w.size() * sizeof(float));
}

// the exact result for what the held F32 matmul holds right now
static std::vector<float> held_ref(const held_mul_mat & h) {
    std::vector<float> x((size_t) h.c.K * h.c.N);
    ggml_backend_tensor_get(h.dst->src[1], x.data(), 0, x.size() * sizeof(float));
    return ref_mul_mat(held_weight(h), x, h.c.K, h.c.M, h.c.N);
}

// with claimed set, the op is computed only when supports_op claimed it: computing a refused op
// on the QNN backend fails the node and, if the graph exists anyway, executes it on the HTP,
// where a failure degrades the session and fails the NEXT check for the wrong reason
static bool run_mul_mat(ggml_backend_t backend, const mul_mat_case & c, std::vector<float> & out,
                        bool * claimed = nullptr, bool tag_weights = true) {
    held_mul_mat h;
    held_init(h, backend, c, tag_weights);

    bool compute = true;
    if (claimed) {
        *claimed = held_claimed(backend, h);
        compute  = *claimed;
    }

    bool ok = false;
    if (compute) {
        ok = held_compute(backend, h, out);
    } else {
        out.resize((size_t) c.N * c.M);
    }

    held_free(h);
    return ok;
}

static ggml_backend_t qnn_backend_init(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("QNN");
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// main already proved the QNN device exists, so a null here is the test's own wiring
static ggml_backend_t qnn_backend_init_checked(void) {
    ggml_backend_t qnn = qnn_backend_init();
    GGML_ASSERT(qnn != nullptr);
    return qnn;
}

static ggml_backend_t cpu_backend_init(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    GGML_ASSERT(cpu != nullptr);
    return cpu;
}

// probe supports_op for a static-weight mul_mat without computing, weight data resident
static bool probe_claim(ggml_backend_t backend, const mul_mat_case & c, bool tag_weights = true) {
    std::vector<float> dummy;
    bool claimed = false;
    run_mul_mat(backend, c, dummy, &claimed, tag_weights);
    return claimed;
}

// supports_op for a mul_mat whose tensors are unallocated (data == NULL). with_dummy hangs the
// loader's zero-size dummy buffer of the device buffer type on the weight (a placement probe);
// without it the weight is a graph tensor with no buffer at all, see scenario_loadprobe
static bool probe_unallocated(ggml_backend_dev_t dev, const mul_mat_case & c, bool with_dummy) {
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, c.wtype, c.K, c.M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.N);
    ggml_tensor * d = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t dummy = with_dummy ? ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 0) : nullptr;
    w->buffer = dummy;
    const bool claimed = ggml_backend_dev_supports_op(dev, d);
    w->buffer = nullptr;
    if (dummy) {
        ggml_backend_buffer_free(dummy);
    }
    ggml_free(ctx);
    return claimed;
}

// probe k DISTINCT 256x128 weights of type wt (kept alive together so their addresses stay
// distinct) and return how many the backend claims - the static budget caps the count. on
// the device every type lands as fp16 (64 KiB here), which is what the budget must charge
static int count_claims(ggml_backend_t backend, int k, ggml_type wt) {
    const int64_t K = 256, M = 128, N = 64;

    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (k + 2), nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * (size_t) (2 * k + 4) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);

    std::vector<ggml_tensor *> ws(k), dsts(k);
    for (int i = 0; i < k; i++) {
        ws[i] = ggml_new_tensor_2d(ctx_w, wt, K, M);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    for (int i = 0; i < k; i++) {
        dsts[i] = ggml_mul_mat(ctx, ws[i], x);
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> wf((size_t) K * M);
    for (int i = 0; i < k; i++) {
        fill_uniform(wf, (unsigned) i + 1);
        if (wt == GGML_TYPE_F32) {
            ggml_backend_tensor_set(ws[i], wf.data(), 0, wf.size() * sizeof(float));
        } else if (wt == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> wh(wf.size());
            ggml_fp32_to_fp16_row(wf.data(), wh.data(), (int64_t) wf.size());
            ggml_backend_tensor_set(ws[i], wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(ws[i]));
            ggml_quantize_chunk(wt, wf.data(), q.data(), 0, M, K, nullptr);
            ggml_backend_tensor_set(ws[i], q.data(), 0, q.size());
        }
    }
    std::vector<float> xf((size_t) K * N);
    fill_uniform(xf, 99);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    int claims = 0;
    for (int i = 0; i < k; i++) {
        if (ggml_backend_dev_supports_op(dev, dsts[i])) {
            claims++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return claims;
}

static long long read_stat(const char * key) {
    FILE * f = fopen(g_stats_path, "r");
    if (!f) {
        return -1;
    }
    char               name[64];
    unsigned long long val   = 0;
    long long          found = -1;
    while (fscanf(f, "%63s %llu", name, &val) == 2) {
        if (strcmp(name, key) == 0) {
            found = (long long) val;
            break;
        }
    }
    fclose(f);
    return found;
}

static void check_stat(const char * key, long long want, const char * why) {
    const long long got = read_stat(key);
    char m[224];
    snprintf(m, sizeof(m), "%s == %lld (got %lld): %s", key, want, got, why);
    check(got == want, m);
}

static long file_size(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fclose(f);
    return n;
}

// whole file as text (CRLF folded to LF on Windows), empty when absent
static std::string read_file(const char * path) {
    std::string s;
    FILE * f = fopen(path, "r");
    if (!f) {
        return s;
    }
    char buf[256];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    fclose(f);
    return s;
}

static int scenario_basic(void) {
    printf("scenario: basic\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    // batch sizes below / inside / at / above the default 512 pad bucket: a weight must bake
    // and stay correct at every N (padded IO copy paths). padded IO is in = K * pad * 4 and
    // out = M * pad * 4 bytes, and GGML_QNN_IO_MAX_KB (default 1024, strict) refuses 1 MiB:
    //   K=256 M=128 pad  512: in 512 KiB, out 256 KiB -> claimed
    //   K=128 M=128 pad 1024: in 512 KiB, out 512 KiB -> claimed, keeps the 1024-bucket copy path under test
    //   K=256 M=128 pad 1024: in exactly 1 MiB        -> policy reject, asserted after the loop
    struct shape { int64_t K, M, N; };
    const shape shapes[] = {
        { 256, 128,   1 },
        { 256, 128,   7 },
        { 256, 128,  60 },
        { 256, 128, 512 },
        { 128, 128, 513 },
    };
    const ggml_type wtypes[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K };

    bool prior_failure = false;
    for (ggml_type wt : wtypes) {
        for (const shape & s : shapes) {
            if (s.K % ggml_blck_size(wt) != 0) {
                continue; // K=128 is not a whole Q4_K block (256), the other three types cover that shape
            }
            const mul_mat_case c = { wt, s.K, s.M, s.N };
            bool claimed = false;
            std::vector<float> got, ref;
            const bool ok_q = run_mul_mat(qnn, c, got, &claimed);
            const bool ok_c = run_mul_mat(cpu, c, ref);

            char label[128];
            snprintf(label, sizeof(label), "mul_mat %s K=%" PRId64 " M=%" PRId64 " N=%" PRId64,
                     ggml_type_name(wt), c.K, c.M, c.N);
            const char * caveat = prior_failure ? " [NOT independent: session may be degraded by the earlier failure]" : "";
            char msg[224];
            if (!claimed) {
                // the HTP may reject a shape at trial time; that is a legal outcome, but for
                // these mainstream shapes it would be a regression worth failing on
                snprintf(msg, sizeof(msg), "%s claimed by QNN%s", label, caveat);
                check(false, msg);
                prior_failure = true;
                continue;
            }
            snprintf(msg, sizeof(msg), "%s computes%s", label, caveat);
            check(ok_q && ok_c, msg);
            if (!ok_q) {
                prior_failure = true;
            }
            if (ok_q && ok_c) {
                // fp16 math on the HTP: same tolerance test-backend-ops uses for MUL_MAT
                const double e = nmse(ref, got);
                snprintf(msg, sizeof(msg), "%s matches CPU (nmse %.2e)%s", label, e, caveat);
                check(e < 5e-4, msg);
            }
        }
    }

    // N=513 pads to 1024, so at K=256 in0 is exactly 1 MiB: the cap is strict, so this is a
    // policy refusal taken before graphCreate. it must not degrade the session, which the
    // compute after the reacquire below would show
    const mul_mat_case capped = { GGML_TYPE_F16, 256, 128, 513 };
    check(!probe_claim(qnn, capped), "256x128 N=513 refused: padded input reaches GGML_QNN_IO_MAX_KB");

    // probe/free/reacquire: llama frees all backends between model probe and context creation;
    // the backend must survive the cycle and still compute
    ggml_backend_free(qnn);
    qnn = qnn_backend_init();
    check(qnn != nullptr, "backend re-initializes after free");
    if (qnn) {
        const mul_mat_case c = { GGML_TYPE_F16, 256, 128, 60 };
        std::vector<float> got, ref;
        const bool ok = run_mul_mat(qnn, c, got) && run_mul_mat(cpu, c, ref);
        check(ok && nmse(ref, got) < 5e-4, "compute correct after free/reacquire cycle");
        ggml_backend_free(qnn);
    }
    ggml_backend_free(cpu);
    return g_failures ? 1 : 0;
}

static int scenario_budget(void) {
    printf("scenario: budget\n");
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    // f16 weight of 4 MiB against a 1 MiB budget: refused as policy, never denylisted. main
    // sets GGML_QNN_NPAD=64 so the padded IO (in 512 KiB, out 256 KiB) stays under
    // GGML_QNN_IO_MAX_KB: the refusal is then the budget's alone, and a deleted budget gate
    // shows up as a clean claim here rather than as an IO-size wedge at the validation
    // execute, which sits out the execute watchdog (GGML_QNN_TIMEOUT_MS, 15 s by default)
    const mul_mat_case big = { GGML_TYPE_F16, 2048, 1024, 64 };
    check(!probe_claim(qnn, big), "over-budget weight is refused");

    // 20 distinct 64 KiB weights against the 1 MiB budget: exactly 16 fit. shapes stay in
    // the 256-class deliberately - 512x512-and-up static bakes were seen hanging at
    // validation execute on the QAIRT 2.45 runtime on battery (worked on 2.34/AC), a
    // separate open issue; this scenario tests budget accounting, not HTP shape appetite
    const int claims1 = count_claims(qnn, 20, GGML_TYPE_F16);
    char msg[128];
    snprintf(msg, sizeof(msg), "budget caps claims at 16/20 (got %d)", claims1);
    check(claims1 == 16, msg);

    // freeing the last backend frees the session and returns the committed bytes: the full
    // 16 claims only fit again if the previous session's 1 MiB actually came back
    ggml_backend_free(qnn);
    qnn = qnn_backend_init();
    check(qnn != nullptr, "backend re-initializes after free");
    if (qnn) {
        const int claims2 = count_claims(qnn, 20, GGML_TYPE_F16);
        snprintf(msg, sizeof(msg), "budget returned after session free (16/20, got %d)", claims2);
        check(claims2 == 16, msg);
        ggml_backend_free(qnn);
    }
    // the over-budget probes never reach the bake: 16 per pass, and the counters are
    // process-global, so the file written by the second session free holds both passes
    check_stat("weights_baked", 32, "16 bakes per pass over two sessions, none for the refused weights");

    // a quantized weight is baked as fp16, so the budget must charge its on-device size
    // (64 KiB, 16 fit) and not its Q4_0 nbytes (18 KiB, all 20 would fit)
    qnn = qnn_backend_init_checked();
    const int claims3 = count_claims(qnn, 20, GGML_TYPE_Q4_0);
    snprintf(msg, sizeof(msg), "Q4_0 budget charges the fp16 on-device size: 16/20 (got %d; 20 would mean nbytes accounting)", claims3);
    check(claims3 == 16, msg);
    ggml_backend_free(qnn);
    check_stat("weights_baked", 48, "the Q4_0 pass baked its 16 claims on top of the two f16 passes");

    // file_size returns -1 absent and 0 empty; assert the file was never CREATED by any of
    // the policy rejects above, so that deleting denylist-writing entirely cannot make this
    // check greener instead of redder
    check(file_size(dl) < 0, "policy rejects did not create the denylist file");

    // a configured budget filling up is not the clamp: that one is reserved for a build
    // failure on a proven shape, see scenario_clamp
    check_stat("budget_clamped", 0, "an ordinary over-budget refusal does not clamp the budget");
    return g_failures ? 1 : 0;
}

// the placement probe carries llama's FICTITIOUS weight-probe batch, not a batch any graph
// runs in, so a denylist entry written by a real run sits in a different bucket. scenario_denylist
// cannot catch a mismatch: it pins GGML_QNN_NPAD=512 and probes at N=60, so seed and probe land
// in the same 512 bucket. At the documented K-quant route (GGML_QNN_NPAD=32 with -ub 32) every
// persisted entry is in the 32 bucket while the probe still arrives carrying 512, and the file
// entry matched nothing at model load. main() seeded the key a -ub 32 run would have written
static int scenario_denylist_probe(void) {
    printf("scenario: denylist-probe\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);

    // N=512 is what llama-model-loader hands weight_buft_supported (llama-model-loader.cpp,
    // weight_buft_supported builds src1 with ne[1]=512 regardless of the real ubatch)
    const mul_mat_case seeded = { GGML_TYPE_F16, 256, 128, 512 };
    check(!probe_unallocated(dev, seeded, /*with_dummy=*/true),
          "placement probe at the loader's fictitious N=512 matches the entry persisted in the 32 bucket");

    // the same shape WITHOUT the loader's dummy buffer is a plain graph tensor, so its N is
    // real: it would build in the 512 bucket, which was never denylisted. it must stay claimed,
    // which is also what keeps the check above from passing on a backend that refuses everything
    check(probe_unallocated(dev, seeded, /*with_dummy=*/false),
          "buffer-less graph tensor at a real N=512 is claimed: its own bucket is not denylisted");

    // control: a weight width that was never seeded is claimed at the same probe
    const mul_mat_case clean = { GGML_TYPE_F16, 256, 64, 512 };
    check(probe_unallocated(dev, clean, /*with_dummy=*/true),
          "an unseeded weight is still claimed at the same placement probe");

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

static int scenario_denylist(bool noopt) {
    printf("scenario: denylist%s\n", noopt ? " (GGML_QNN_NO_OPT=1)" : "");
    // main() seeded the file with:
    //   MUL_MAT_f16_f32_256x128x1x1_256x512x1x1_s    (static variant of case A)
    //   MUL_MAT_f16_f32_256x64x1x1_256x512x1x1_dyn   (dynamic variant of case B)
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);

    // case A: N=60 pads to the 512 bucket, so its static shape key matches the seeded entry
    const mul_mat_case a = { GGML_TYPE_F16, 256, 128, 60 };
    if (!noopt) {
        check(!probe_claim(qnn, a), "seeded static entry blocks the static shape");
    } else {
        // GGML_QNN_NO_OPT exists to rebuild the shape that failed, so an entry from the FILE is
        // advisory under it (only one from a failure in this process still applies). claim only:
        // the claim is what is under test, and an fp16 compute at the 512 bucket costs seconds
        // on a slow device
        held_mul_mat h;
        held_init(h, qnn, a);
        check(held_claimed(qnn, h), "GGML_QNN_NO_OPT: the file entry is advisory, the seeded static shape is claimed");
        held_free(h);
    }

    // case B: only the *dynamic* variant is seeded; the static path must not be blocked
    const mul_mat_case b = { GGML_TYPE_F16, 256, 64, 60 };
    check(probe_claim(qnn, b), "dynamic-variant entry does not block the static path");

    // the load-time probe has no data yet, so it cannot tell which variant the weight becomes
    // and consults both (ggml_qnn_shape_denylisted): each seeded entry refuses its unallocated
    // weight, with the loader's dummy buffer and as a buffer-less graph tensor. under NO_OPT
    // both entries came from the file, so the same probes are claimed
    const mul_mat_case seeded[] = { a, b };
    for (const mul_mat_case & c : seeded) {
        for (bool with_dummy : { true, false }) {
            const bool claimed = probe_unallocated(dev, c, with_dummy);
            char msg[224];
            snprintf(msg, sizeof(msg), "unallocated 256x%" PRId64 " probe (%s), %s entry seeded: %s", c.M,
                     with_dummy ? "placement dummy buffer" : "no buffer", c.M == a.M ? "static" : "dynamic",
                     noopt ? "claimed, the file entry is advisory under GGML_QNN_NO_OPT" : "refused");
            check(noopt ? claimed : !claimed, msg);
        }
    }

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// the padded-IO size law: static-bake graphs hang at execute when a padded IO buffer crosses
// a runtime-dependent threshold (~1.5MB out on QAIRT 2.34, lower on 2.45), and work at any
// weight size below it. this mode guards the WORKING side and is safe to gate on:
// model-scale bake at a small pad bucket + correctness across bucket boundaries.
// the optional npad argument (main) feeds GGML_QNN_NPAD; "0" pins the exact-pow2 branch
static int scenario_modelscale(void) {
    const char * npad_env = getenv("GGML_QNN_NPAD");
    const int64_t npad    = npad_env ? atoll(npad_env) : 512;
    printf("scenario: modelscale (GGML_QNN_NPAD=%s)\n", npad_env ? npad_env : "(unset)");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    // guard: a model-scale weight bakes and computes below the IO-size threshold; boundary
    // cases: N at, just above, and far below the 64 bucket exercise the rounding paths.
    // the last case sits exactly on its bucket (N == bucket: 64 at NPAD=64, 4 at NPAD=0) and
    // has a shape no earlier case used, so it is a fresh build and not a cache hit:
    // pad_n_last is recorded only when a graph is created, and a '<' vs '<=' slip in the
    // doubling rule moves exactly this sample to the next bucket (N=3 rounds to 4 either way)
    const int64_t n_last = npad ? npad : 4;
    const mul_mat_case cases[] = {
        { GGML_TYPE_F16, 512, 2560, 64 },     // model-scale guard (out 640KB at bucket 64)
        { GGML_TYPE_F16, 256,  256, 64 },     // N == bucket
        { GGML_TYPE_F16, 256,  256, 65 },     // N one past the bucket -> next pow2 bucket
        { GGML_TYPE_F16, 256,  256,  3 },     // tiny N; with NPAD=0 pins the exact-pow2 branch
        { GGML_TYPE_F16, 256,  512, n_last }, // N == bucket, distinct shape, sampled by pad_n_last
    };
    bool prior_failure = false;
    for (const auto & c : cases) {
        bool claimed = false;
        std::vector<float> got, ref;
        const bool ok_q = run_mul_mat(qnn, c, got, &claimed);
        const bool ok_c = run_mul_mat(cpu, c, ref);
        char msg[192];
        const char * caveat = prior_failure ? " [NOT independent: session may be degraded by the earlier failure]" : "";
        snprintf(msg, sizeof(msg), "static %" PRId64 "x%" PRId64 " N=%" PRId64 " claimed, computes, matches CPU%s",
                 c.K, c.M, c.N, caveat);
        const bool ok = claimed && ok_q && ok_c && nmse(ref, got) < 5e-4;
        check(ok, msg);
        if (!ok) {
            prior_failure = true;
        }
    }

    ggml_backend_free(qnn); // session teardown writes the counters
    ggml_backend_free(cpu);

    char m[224];
    if (prior_failure) {
        // pad_n_last is sampled at the last graph CREATED, and after a failure that is not
        // necessarily the last case, so the number would be read against the wrong shape
        printf("  skip  pad_n_last check: an earlier case failed, the sampled bucket is not the last case's\n");
    } else {
        // the pad bucket the last case actually landed on. this is the only thing that
        // distinguishes the -npad0 variant from its parent, which were otherwise identical runs
        const long long pad      = read_stat("pad_n_last");
        const long long want_pad = pad_n_for(npad, n_last);
        snprintf(m, sizeof(m), "NPAD=%s put the last case (N=%" PRId64 ") on bucket %lld (pad_n_last=%lld)",
                 npad_env ? npad_env : "(unset)", n_last, want_pad, pad);
        check(pad == want_pad, m);
    }

    // TURBO clocks are applied on the first real graph use, once per session; a silently
    // failed setPowerConfig would otherwise leave every check green, which is why the
    // counter is asserted and not the speed - what the vote is worth is unmeasured. the
    // -noburst variant is the A/B lever for that vote and must skip it entirely
    const bool no_burst = g_require_env && strcmp(g_require_env, "GGML_QNN_NO_BURST") == 0;
    check_stat("burst_applied", no_burst ? 0 : 1,
               no_burst ? "GGML_QNN_NO_BURST: no HTP burst power config applied" : "HTP burst power config applied once for the session");

    int rc = 0;
    if (g_require_env && strcmp(g_require_env, "GGML_QNN_SHARED_MEM") == 0) {
        const long long shared   = read_stat("io_shared");
        const long long fallback = read_stat("io_shm_fallback");
        const long long host     = read_stat("io_host");
        // the init-time self-test verdict: 0 = libcdsprpc absent, 1 = present but the self-test
        // failed, 2 = ok. the IO counters read the same (both 0) for 0 and 1, and only an absent
        // library means there is no rpcmem path on this device to hold to account
        const long long selftest = read_stat("shm_selftest");
        if (selftest == 0) {
            // a skip, not a pass with zero rpcmem assertions
            printf("fastrpc (libcdsprpc) absent - rpcmem IO path not exercised on this device, skipping\n");
            rc = 77;
        } else {
            // present but failed (rpcmem symbols, rpcmem_alloc2, rpcmem_to_fd, QnnMem_register)
            // is a broken shared-memory path, and a missing line is a backend without the counter
            snprintf(m, sizeof(m), "fastrpc is present and the init-time shared-memory self-test passed (shm_selftest=%lld, want 2)", selftest);
            check(selftest == 2, m);
            snprintf(m, sizeof(m), "rpcmem IO used for every graph, no silent host fallback "
                     "(io_shared=%lld io_shm_fallback=%lld io_host=%lld)", shared, fallback, host);
            check(shared > 0 && fallback == 0 && host == 0, m);
        }
    } else if (g_require_env && strcmp(g_require_env, "GGML_QNN_NO_OPT") == 0) {
        const long long noopt = read_stat("graphs_noopt");
        snprintf(m, sizeof(m), "no-opt graph config reached graphCreate (graphs_noopt=%lld)", noopt);
        check(noopt > 0, m);
    } else if (g_require_env && strcmp(g_require_env, "GGML_QNN_NO_STATIC_WEIGHTS") == 0) {
        // dynamic-weight graphs: nothing bakes, and the graph key is the padded shape alone
        // (no weight pointer), so the case list above builds one graph per distinct
        // (shape, bucket) pair. at NPAD=64: 512x2560@64, 256x256@64, 256x256@128 (N=65),
        // 256x256@64 again for N=3 (cache hit), 256x512@64 -> 4 graphs. at NPAD=0 the N=3
        // case lands on its own bucket 4 -> 5 graphs
        std::vector<std::string> keys;
        for (const auto & c : cases) {
            const std::string k = shape_key(c, npad, /*is_static=*/false);
            bool seen = false;
            for (const auto & o : keys) {
                seen = seen || o == k;
            }
            if (!seen) {
                keys.push_back(k);
            }
        }
        check_stat("weights_baked", 0, "no static bake with GGML_QNN_NO_STATIC_WEIGHTS");
        check_stat("graphs_created", (long long) keys.size(), "one dynamic graph per distinct (shape, bucket) pair in the case list");
    }
    return g_failures ? 1 : rc;
}

// 512-class static bakes were seen hanging at validation execute on the QAIRT 2.45 runtime
// on battery while the 256-class worked. this mode is the discriminator, but "run it on AC"
// is NOT a sufficient precondition: a deeply discharged pack on AC runs CPU-bound work at
// roughly half speed, so an AC hang can be power and be misread as implicating the runtime.
// run it on AC with a SETTLED pack - drawing under 5 W, with the charge percent recorded as a
// covariate rather than used as a gate - and only then
// does a clean pass isolate power limiting and a hang implicate the runtime. see the power
// note under "Benchmarking notes" in docs/backend/QNN.md.
// a failed case degrades the session, so later cases in the same process are not
// independent results; case_idx >= 0 runs one case alone for order/isolation permutations.
// the hang is at the validation execute, which runs under GGML_QNN_TIMEOUT_MS (15 s by default)
// like every execute; GGML_QNN_BUILD_TIMEOUT_MS (120 s) bounds graph finalize only. a hanging
// case therefore sits out 15 s, and so does an execute that is merely slow (other load on the
// machine slows it too): raise GGML_QNN_TIMEOUT_MS before reading a timeout as the hang this mode is
// looking for. on a slow device GGML_QNN_SLOW_EXEC_MS matters too: main leaves it at 0 (off)
// unless the caller sets it, and a value at or under the execute time refuses the completed
// validation as too slow. main keeps the caller's value of all three for this mode
static int scenario_bigstatic(int case_idx) {
    printf("scenario: bigstatic%s\n", case_idx >= 0 ? " (single case)" : "");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    const mul_mat_case cases[] = {
        { GGML_TYPE_F16, 512,  512, 64 },
        { GGML_TYPE_F16, 512,  768, 64 }, // non-pow2 M, hangs on 2.34 and 2.45
        { GGML_TYPE_F16, 512, 1024, 64 }, // pow2 M larger than the failing 768
        { GGML_TYPE_F16, 512,  640, 64 }, // non-pow2 M smaller than the failing 768
        { GGML_TYPE_F16, 512, 2560, 64 }, // model-scale M; passes at small pad buckets if the
                                          // hang follows padded IO size (run with GGML_QNN_NPAD=64)
    };
    const int n_cases = (int) (sizeof(cases) / sizeof(cases[0]));
    if (case_idx >= n_cases) {
        fprintf(stderr, "bigstatic: case index %d out of range (0..%d)\n", case_idx, n_cases - 1);
        return 1;
    }

    bool prior_failure = false;
    for (int i = 0; i < n_cases; i++) {
        if (case_idx >= 0 && i != case_idx) {
            continue;
        }
        const mul_mat_case & c = cases[i];
        bool claimed = false;
        std::vector<float> got, ref;
        const bool ok_q = run_mul_mat(qnn, c, got, &claimed);
        const bool ok_c = run_mul_mat(cpu, c, ref);
        char msg[192];
        const char * caveat = prior_failure ? " [NOT independent: session may be degraded by the earlier failure]" : "";
        snprintf(msg, sizeof(msg), "static bake %" PRId64 "x%" PRId64 " claimed%s", c.K, c.M, caveat);
        check(claimed, msg);
        if (claimed) {
            snprintf(msg, sizeof(msg), "static bake %" PRId64 "x%" PRId64 " computes and matches CPU%s", c.K, c.M, caveat);
            check(ok_q && ok_c && nmse(ref, got) < 5e-4, msg);
        }
        if (!claimed || !ok_q) {
            prior_failure = true;
        }
    }

    ggml_backend_free(qnn);
    ggml_backend_free(cpu);
    return g_failures ? 1 : 0;
}

static void * dl_open(const char * path) {
#ifdef _WIN32
    // like the backend (qnn-dl.h): a full path resolves its dependent DLLs from its own directory
    if (strchr(path, '\\') || strchr(path, '/')) {
        return (void *) LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void dl_close(void * lib) {
#ifdef _WIN32
    FreeLibrary((HMODULE) lib);
#else
    dlclose(lib);
#endif
}

// load the HTP runtime the way ggml_qnn_load_htp_lib does (default search path, then the
// QNN_SDK_ROOT lib dirs) and unload it again. no QNN call is made, so the NPU is not touched.
// returns the path that loaded, empty when none did
static std::string htp_runtime_loadable(void) {
#ifdef _WIN32
    const char * lib_name   = "QnnHtp.dll";
    const char * lib_dirs[] = { "aarch64-windows-msvc", "arm64x-windows-msvc" };
    const char   sep        = '\\';
#else
    const char * lib_name   = "libQnnHtp.so";
    const char * lib_dirs[] = { "aarch64-android", "aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc11.2", "aarch64-oe-linux-gcc9.3" };
    const char   sep        = '/';
#endif
    std::vector<std::string> paths = { lib_name };
    const char * sdk_root = getenv("QNN_SDK_ROOT");
    if (sdk_root && *sdk_root) {
        for (const char * dir : lib_dirs) {
            paths.push_back(std::string(sdk_root) + sep + "lib" + sep + dir + sep + lib_name);
        }
    }
    for (const std::string & p : paths) {
        void * lib = dl_open(p.c_str());
        if (lib) {
            dl_close(lib);
            return p;
        }
    }
    return std::string();
}

// the A/B kill-switch: GGML_QNN_DISABLE (set by main before any registry use) must remove
// the device entirely. on a machine with no HTP this passes vacuously - acceptable, since
// the assertion is absence. GGML_QNN_TEST_REQUIRE_HTP=1 names a box that has a working HTP
// runtime, so there the absence must be the kill-switch's doing: a runtime that cannot even
// load would remove the device too, and the check below would pass for the wrong reason
static int scenario_disable(void) {
    printf("scenario: disable\n");
    const char * req = getenv("GGML_QNN_TEST_REQUIRE_HTP");
    if (req && *req && strcmp(req, "0") != 0) {
        const std::string loaded = htp_runtime_loadable();
        char msg[512];
        snprintf(msg, sizeof(msg), "GGML_QNN_TEST_REQUIRE_HTP: the HTP runtime is loadable, so an absent device is the kill-switch (%s)",
                 loaded.empty() ? "NOT loadable from the default search path or QNN_SDK_ROOT" : loaded.c_str());
        check(!loaded.empty(), msg);
    }
    check(ggml_backend_dev_by_name("QNN") == nullptr, "GGML_QNN_DISABLE removes the QNN device");
    return g_failures ? 1 : 0;
}

// default-env gate: with GGML_QNN_MIN_DIM unset (default 32), small matmuls must be refused
// (they are memory-bound and belong on the CPU) while normal shapes are still claimed
static int scenario_mindim(void) {
    printf("scenario: mindim (default env)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();

    const mul_mat_case small = { GGML_TYPE_F16, 16, 16, 8 };
    check(!probe_claim(qnn, small), "16x16 matmul refused at the default min-dim gate");

    const mul_mat_case normal = { GGML_TYPE_F16, 256, 128, 64 };
    check(probe_claim(qnn, normal), "256x128 matmul still claimed at default env");

    // the documented default is 32, pinned from both sides on the batch dim: one below is a
    // policy refusal that never touches the HTP, exactly 32 goes through the trial build
    const mul_mat_case n31 = { GGML_TYPE_F16, 256, 256, 31 };
    check(!probe_claim(qnn, n31), "256x256 N=31 refused: one below the default GGML_QNN_MIN_DIM of 32");
    const mul_mat_case n32 = { GGML_TYPE_F16, 256, 256, 32 };
    check(probe_claim(qnn, n32), "256x256 N=32 claimed: exactly the default GGML_QNN_MIN_DIM");

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// ADD/MUL must stay refused at default env. The HTP returns wrong results for some Add
// broadcast shapes, so the GGML_QNN_ELEMENTWISE opt-in is the only thing keeping them off
// real models: a regression that dropped the gate would corrupt output silently while every
// other test stayed green
// a same-shape ADD or MUL: returns whether the device claims it. with out set it is also
// computed when claimed, *computed telling whether that succeeded
static bool probe_binary_claim(ggml_backend_t backend, bool use_add, int64_t K, int64_t N,
                               std::vector<float> * out = nullptr, bool * computed = nullptr) {
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * d = use_add ? ggml_add(ctx, a, b) : ggml_mul(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> f((size_t) K * N);
    fill_uniform(f, 3);
    ggml_backend_tensor_set(a, f.data(), 0, f.size() * sizeof(float));
    fill_uniform(f, 5);
    ggml_backend_tensor_set(b, f.data(), 0, f.size() * sizeof(float));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool claimed = ggml_backend_dev_supports_op(dev, d);

    if (out) {
        bool ok = false;
        out->assign((size_t) K * N, 0.0f);
        if (claimed) {
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(d, out->data(), 0, out->size() * sizeof(float));
            }
        }
        if (computed) {
            *computed = ok;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return claimed;
}

static int scenario_elementwise(void) {
    printf("scenario: elementwise (default env)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();

    // 2M elements clears the default GGML_QNN_MIN_ELEMENTS (1M), so a refusal here is the
    // ELEMENTWISE gate itself and not the size threshold
    const int64_t K = 2048, N = 1024;
    check(!probe_binary_claim(qnn, true,  K, N), "ADD refused with GGML_QNN_ELEMENTWISE unset");
    check(!probe_binary_claim(qnn, false, K, N), "MUL refused with GGML_QNN_ELEMENTWISE unset");

    // the matmul path must be unaffected by the elementwise gate
    const mul_mat_case normal = { GGML_TYPE_F16, 256, 128, 64 };
    check(probe_claim(qnn, normal), "mul_mat still claimed at default env");

    ggml_backend_free(qnn);
    // the gate refuses before any session work, so the matmul is the only graph of the run
    check_stat("graphs_created", 1, "only the matmul reached graphCreate, the elementwise probes were refused by the gate");
    return g_failures ? 1 : 0;
}

// with the opt-in on, the GGML_QNN_MIN_ELEMENTS gate is what stands between a small ADD and
// the HTP; it was untested because the ELEMENTWISE gate returned first at default env.
// main sets the threshold to 65536: the elementwise path has no IO-size cap, and at the
// default 1M the trial would move 4 MiB buffers, past where matmul graphs were seen hanging
static int scenario_elementwise_on(void) {
    printf("scenario: elementwise-on (GGML_QNN_ELEMENTWISE=1)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    // one row short of the threshold (256 * 255 = 65280): a policy refusal that never touches the HTP
    check(!probe_binary_claim(qnn, true, 256, 255), "ADD below GGML_QNN_MIN_ELEMENTS refused with the opt-in on");

    // exactly the threshold (256 * 256, 256 KiB per buffer) passes the gate and reaches the
    // HTP trial build. claimed or refused by that trial are both legal verdicts on this
    // runtime; what is not legal is a policy refusal (the graph never created) or a trial
    // that degrades the session. a claimed ADD is then computed: the validation execute ran on
    // zeroed buffers, so only a compute proves the copy-in, the execute and the copy-back.
    // same shape only - the HTP Add broadcast is documented wrong for some shapes
    std::vector<float> got, ref;
    bool ok_q = false, ok_c = false;
    const bool add_claimed = probe_binary_claim(qnn, true, 256, 256, &got, &ok_q);
    printf("  info  ADD at GGML_QNN_MIN_ELEMENTS %s by the HTP trial\n", add_claimed ? "claimed" : "refused");
    if (add_claimed) {
        GGML_ASSERT(probe_binary_claim(cpu, true, 256, 256, &ref, &ok_c) && ok_c);
        // the HTP runs the add in fp16 internally: inputs in [-1, 1] round to about 5e-4
        // relative, nmse about 1e-7, so the matmul tolerance leaves a wide margin
        const double e = ok_q ? nmse(ref, got) : -1.0;
        char msg[160];
        snprintf(msg, sizeof(msg), "the claimed same-shape ADD computes on the HTP and matches the CPU (nmse %.2e)", e);
        check(ok_q && e < 5e-4, msg);
    } else {
        printf("  skip  ADD compute: the HTP trial refused the shape\n");
    }

    const mul_mat_case normal = { GGML_TYPE_F16, 256, 128, 64 };
    check(probe_claim(qnn, normal), "mul_mat still claimed: the ADD trial did not degrade the session");

    ggml_backend_free(qnn);
    ggml_backend_free(cpu);
    // the ADD graph plus the matmul graph: a count of 1 means the ADD was refused by policy
    // instead of by the HTP. the ADD compute is a cache hit (an elementwise key is the shape)
    check_stat("graphs_created", 2, "the at-threshold ADD reached graphCreate, plus the matmul");
    return g_failures ? 1 : 0;
}

// llama's model loader probes supports_op for every weight BEFORE the data is resident: the
// weight sits in a no_alloc context, so data is NULL, and the loader hangs a zero-size dummy
// buffer of the candidate buffer type on it so the backend can see where the weight will land
// (llama-model-loader.cpp, weight_buft_supported). That verdict must match the one taken at
// schedule time on the same resident, WEIGHTS-tagged weight - otherwise the backend either
// claims a shape it will refuse later, or finalizes and permanently caches a dynamic-variant
// graph for a shape real inference never executes.
//
// A graph tensor is the other unallocated shape and asks a different question: it has no buffer
// at all, and whoever allocates it (test-backend-ops, ggml-alloc behind a scheduler split)
// leaves it untagged, so a quantized weight can never be baked from it. Its verdict must match
// the resident UNTAGGED one, the path the node really takes at compute; claiming it there fails
// the op outright where a refusal would have left it on the CPU
//
// supports_op refusals decided by the node's form alone, before any trial build. every tensor
// is resident, the weights in a WEIGHTS-tagged buffer like a model's, so a gate that regressed
// would send the node on to the static bake and the caller's graphs_created would show it.
// the control of the same size and buffers is claimed, which builds one static graph
static void check_form_refusals(ggml_backend_t qnn) {
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);
    const int64_t K = 256, M = 128, N = 64;

    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * 24, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);

    ggml_tensor * w     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32,  K, M);
    ggml_tensor * w3    = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32,  K, M, 2);
    ggml_tensor * wwide = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32,  2 * K, M);
    ggml_tensor * wbf16 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, K, M);

    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * x3    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, 2);
    ggml_tensor * xwide = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * K, N);
    ggml_tensor * xh    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);

    // ggml_mul_mat cannot pair a 3D weight with a 2D activation (the activation must repeat
    // the weight's ne2), so the 3D weight is swapped in after the node is built: that isolates
    // the src0 leg of the 2D gate, and d_x3 covers the src1 leg
    ggml_tensor * d_w3 = ggml_mul_mat(ctx, w, x);
    d_w3->src[0] = w3;
    ggml_tensor * d_x3 = ggml_mul_mat(ctx, w, x3);
    // row stride 2K, so neither view is contiguous. ggml_mul_mat asserts a weight is not
    // transposed, so a strided view is the non-contiguous weight a graph can carry
    ggml_tensor * d_wv   = ggml_mul_mat(ctx, ggml_view_2d(ctx, wwide, K, M, wwide->nb[1], 0), x);
    ggml_tensor * d_xv   = ggml_mul_mat(ctx, w, ggml_view_2d(ctx, xwide, K, N, xwide->nb[1], 0));
    ggml_tensor * d_bf16 = ggml_mul_mat(ctx, wbf16, x);
    ggml_tensor * d_xh   = ggml_mul_mat(ctx, w, xh);
    ggml_tensor * d_ctl  = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, qnn);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, qnn);
    // zeros are a valid value of every type here; only the control ever reads them
    ggml_backend_buffer_clear(buf_w, 0);
    ggml_backend_buffer_clear(buf, 0);

    check(!ggml_backend_dev_supports_op(dev, d_w3),   "3D weight (ne2=2) refused: the graph is built from ne[0] and ne[1] alone");
    check(!ggml_backend_dev_supports_op(dev, d_x3),   "3D activation (ne2=2) refused");
    check(!ggml_backend_dev_supports_op(dev, d_wv),   "non-contiguous weight (strided view) refused");
    check(!ggml_backend_dev_supports_op(dev, d_xv),   "non-contiguous activation (strided view) refused: its nbytes would overrun the padded input");
    check(!ggml_backend_dev_supports_op(dev, d_bf16), "BF16 weight refused: a raw copy into an fp16 tensor would reinterpret its bits");
    check(!ggml_backend_dev_supports_op(dev, d_xh),   "F16 activation refused: the input tensor is declared F32");
    check(ggml_backend_dev_supports_op(dev, d_ctl),   "control: the same F32 2D contiguous matmul is claimed");

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
}

// a weight that lives in the CPU backend's CPU_REPACK buffer holds interleaved bytes and its
// buffer type is not host: the static bake would read the repacked layout (a Q4_K weight did,
// 2026-09-16, and its validation execute wedged the session). supports_op must refuse it,
// resolving a reshape's buffer through view_src like the scheduler does, and supports_buft must
// keep llama from placing a weight there for this device at all
static void check_repack_refusal(ggml_backend_t qnn) {
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    check(ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(cpu_dev)), "supports_buft accepts the plain CPU buffer type");

    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    for (ggml_backend_buffer_type_t * p = get_extra ? get_extra(cpu_dev) : nullptr; p && *p; p++) {
        if (strstr(ggml_backend_buft_name(*p), "REPACK")) {
            repack = *p;
            break;
        }
    }
    if (!repack) {
        printf("  note  no CPU_REPACK buffer type in this build, skipping the repack refusal checks\n");
        return;
    }
    check(!ggml_backend_dev_supports_buft(dev, repack), "supports_buft refuses CPU_REPACK: it is not a host buffer type");

    const int64_t K = 256, M = 128, N = 64;
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);
    ggml_context * ctx_v = ggml_init(gp);

    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_0, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * d = ggml_mul_mat(ctx, w, x);

    // never written: set_tensor would repack, and only the placement is under test
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, repack);
    GGML_ASSERT(buf_w != nullptr);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, qnn);
    ggml_backend_buffer_clear(buf, 0);

    // made after the allocation and never allocated itself: data points into w, buffer is NULL,
    // the state a reshape of a resident weight has at split time
    ggml_tensor * wr = ggml_reshape_2d(ctx_v, w, K, M);
    ggml_tensor * dr = ggml_mul_mat(ctx_v, wr, x);
    GGML_ASSERT(wr->buffer == nullptr && wr->data == w->data);

    check(!ggml_backend_dev_supports_op(dev, d),  "Q4_0 weight resident in CPU_REPACK refused: its bytes are the repacked layout");
    check(!ggml_backend_dev_supports_op(dev, dr), "and a reshape of it refused too: the buffer is resolved through view_src");

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_v);
    ggml_free(ctx);
    ggml_free(ctx_w);
}

static int scenario_loadprobe(void) {
    printf("scenario: loadprobe\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);

    check_form_refusals(qnn);
    check_repack_refusal(qnn);

    const int64_t K = 256, M = 128, N = 64;
    const ggml_type wtypes[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0 };

    for (ggml_type wt : wtypes) {
        const mul_mat_case c = { wt, K, M, N };

        // unallocated shapes: data == NULL for both, the buffer is the whole difference.
        // graph tensor: no buffer at all, so nothing will tag it WEIGHTS
        const bool claim_graph = probe_unallocated(dev, c, /*with_dummy=*/false);

        // load-time weight: the loader's zero-size dummy buffer of the destination type
        const bool claim_load = probe_unallocated(dev, c, /*with_dummy=*/true);

        // schedule-time shape: same dims, resident data, buffer tagged WEIGHTS
        const bool claim_sched = probe_claim(qnn, c);

        // the same resident weight in an untagged buffer: no bake, so no quantized path
        const bool claim_untagged = probe_claim(qnn, c, /*tag_weights=*/false);

        char msg[176];
        snprintf(msg, sizeof(msg), "%s supports_op agrees load-time vs schedule-time (load=%d sched=%d)",
                 ggml_type_name(wt), (int) claim_load, (int) claim_sched);
        check(claim_load == claim_sched, msg);

        snprintf(msg, sizeof(msg), "%s supports_op agrees graph-tensor vs untagged resident (graph=%d untagged=%d)",
                 ggml_type_name(wt), (int) claim_graph, (int) claim_untagged);
        check(claim_graph == claim_untagged, msg);
    }

    // the arithmetic rejects (padded IO cap, uint32 sizes) are part of the unallocated answer:
    // a claimed node that ggml_qnn_mul_mat_policy refuses at compute does not fall back to the
    // CPU, it fails the graph (mean pooling over 32 sequences returned -3 from llama_encode).
    // a buffer-less graph matmul carries its real N. main pins GGML_QNN_NPAD=512, so N=32 pads
    // to 512 and the padded IO is K * 512 * 4 in, M * 512 * 4 out against the strict 1 MiB
    // default of GGML_QNN_IO_MAX_KB:
    //   F32 K=1024 M=1024 N=32: 2 MiB in and out -> refused
    //   F32 K=80   M=64   N=32: 160 KiB in, 128 KiB out -> claimed
    // and each verdict must match the same matmul resident in an untagged buffer, which is
    // the node compute really gets
    const mul_mat_case capped = { GGML_TYPE_F32, 1024, 1024, 32 };
    check(!probe_unallocated(dev, capped, /*with_dummy=*/false),
          "buffer-less F32 1024x1024 N=32 refused: 2 MiB of padded IO at the 512 bucket reaches GGML_QNN_IO_MAX_KB");
    check(!probe_claim(qnn, capped, /*tag_weights=*/false), "and the same matmul resident in an untagged buffer is refused too");

    const mul_mat_case small = { GGML_TYPE_F32, 80, 64, 32 };
    check(probe_unallocated(dev, small, /*with_dummy=*/false), "buffer-less F32 80x64 N=32 still claimed: 160 KiB of padded IO");
    check(probe_claim(qnn, small, /*tag_weights=*/false), "and the same matmul resident in an untagged buffer is claimed too");

    // a weight probed for PLACEMENT carries a fictitious N (llama probes with its own batch),
    // so the cap is evaluated at the smallest bucket, ggml_qnn_pad_n(1) = 512 here: K=1024 is
    // 2 MiB in at any N the weight will ever run at, and claiming it would cost the weight its
    // CPU_REPACK placement for a node that never reaches the NPU. the probe's own N must not
    // decide: 256x128 at N=513 would pad to 1024 (exactly 1 MiB in) and is still placed
    const mul_mat_case placed_capped = { GGML_TYPE_F16, 1024, 128, 64 };
    check(!probe_unallocated(dev, placed_capped, /*with_dummy=*/true),
          "F16 1024x128 weight refused at placement: capped even at the smallest bucket");
    const mul_mat_case placed_n513 = { GGML_TYPE_F16, 256, 128, 513 };
    check(probe_unallocated(dev, placed_n513, /*with_dummy=*/true),
          "F16 256x128 weight placed whatever the probe's N: the cap is taken at the smallest bucket");

    ggml_backend_free(qnn);
    // the unallocated probes must answer from policy alone and build nothing. what builds:
    // the form-refusal control (static F32), the two schedule-time static probes (f16 and
    // Q4_0), one bake each, and the untagged f16 probe and the untagged small F32 probe, which
    // are dynamic graphs with no bake. the untagged Q4_0 probe and the untagged capped F32
    // probe are policy rejects, and the form and repack refusals are taken before any session
    // work, so none of them creates a graph. so 5 graphs, 3 bakes - anything more means an
    // unallocated probe, a refused form, a repacked weight or a capped shape reached the HTP
    check_stat("graphs_created", 5, "the F32 control, two static schedule-time builds, the untagged f16 and small F32 dynamic graphs");
    check_stat("weights_baked", 3, "one bake each for the F32 control and the static schedule-time probes (f16, Q4_0)");
    return g_failures ? 1 : 0;
}

// device health, not backend logic: one model-scale matmul (84 M multiply-accumulates) must
// execute in well under a second. the functional entries stay green on a device far too slow
// to be useful, so this one times it. its verdict only means something with nothing else
// running on the machine: other load slows the execute too. main disables
// GGML_QNN_SLOW_EXEC_MS for every mode, this one included, so the functional entries do not
// depend on device speed and this one can time the execute instead of having the backend
// refuse the shape. the slow-execute fallback itself is covered by slow-validate and
// slow-compute, which inject the slowness with GGML_QNN_DELAY_EXECUTE instead of relying on it
static int scenario_health(void) {
    printf("scenario: health\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    const mul_mat_case c = { GGML_TYPE_F16, 512, 2560, 64 };
    std::vector<float> ref, got;
    bool claimed = false;
    const bool ok_c = run_mul_mat(cpu, c, ref);
    const bool ok_q = run_mul_mat(qnn, c, got, &claimed);
    check(claimed && ok_q && ok_c && nmse(ref, got) < 5e-4, "static 512x2560 N=64 claimed, computes, matches CPU");

    ggml_backend_free(qnn); // session teardown writes the counters
    ggml_backend_free(cpu);

    const long long max_ms = read_stat("exec_max_ms");
    const long long slow   = read_stat("exec_slow");
    char m[256];
    snprintf(m, sizeof(m), "slowest compute-time execute under 1000 ms (exec_max_ms=%lld, exec_slow=%lld): a healthy HTP takes a few ms; rerun with nothing else on the machine before concluding anything", max_ms, slow);
    check(max_ms >= 0 && max_ms < 1000, m);
    // exec_max_ms reads 0 when nothing executed, which the check above passes: run_mul_mat
    // computed the one claimed node once, and that compute-time execute is the one timed
    check_stat("exec_count", 1, "the one compute reached a timed execute");
    check_stat("burst_applied", 1, "HTP burst power config applied once for the session");
    return g_failures ? 1 : 0;
}

// one weight probed at two batch sizes inside the same pad bucket must bake ONCE: the graph
// key is (padded shape, weight address), so both probes share a graph. the budget (2 MB) fits
// exactly one bake of the 1.125 MB weight - a per-N re-bake regression (the failure that
// originally exhausted NPU memory) makes the second probe over-budget and fails the test.
// the weight stays alive across both probes so its address cannot be reused
static int scenario_rebake(void) {
    printf("scenario: rebake\n");
    ggml_backend_t qnn = qnn_backend_init_checked();

    const int64_t K = 768, M = 768;
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);

    ggml_tensor * w  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, K, M);
    ggml_tensor * x1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 8);  // pad bucket 64
    ggml_tensor * x2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 50); // same bucket
    ggml_tensor * d1 = ggml_mul_mat(ctx, w, x1);
    ggml_tensor * d2 = ggml_mul_mat(ctx, w, x2);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, qnn);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, qnn);

    std::vector<float> wf((size_t) K * M);
    fill_uniform(wf, 7);
    std::vector<ggml_fp16_t> wh(wf.size());
    ggml_fp32_to_fp16_row(wf.data(), wh.data(), (int64_t) wf.size());
    ggml_backend_tensor_set(w, wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));

    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);
    check(ggml_backend_dev_supports_op(dev, d1), "first N in the bucket claimed (bakes the weight)");
    check(ggml_backend_dev_supports_op(dev, d2), "second N in the same bucket claimed without a re-bake");

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    ggml_backend_free(qnn); // session teardown writes the counters

    // asserted directly rather than inferred from the budget: a per-N re-bake used to be
    // caught only because it exhausted GGML_QNN_STATIC_BUDGET_MB, so a budget change would
    // have silently retired the check this scenario exists for
    const long long baked = read_stat("weights_baked");
    const long long hits  = read_stat("graph_cache_hits");
    char m[160];
    snprintf(m, sizeof(m), "weight baked exactly once (weights_baked=%lld)", baked);
    check(baked == 1, m);
    snprintf(m, sizeof(m), "second N in the bucket reused the cached graph (graph_cache_hits=%lld)", hits);
    check(hits >= 1, m);
    return g_failures ? 1 : 0;
}

// main removed the file, so the backend created it: the "# ggml-qnn denylist v1" header line,
// then one shape key per line
static void check_denylist_holds(const char * dl, const std::string & key) {
    const std::string content = read_file(dl);
    const std::string want    = "# ggml-qnn denylist v1\n" + key + "\n";
    char m[256];
    snprintf(m, sizeof(m), "denylist file holds exactly the header and %s", key.c_str());
    check(content == want, m);
    if (content != want) {
        printf("        file content: \"%s\"\n", content.c_str());
    }
}

static int scenario_watchdog(void) {
    printf("scenario: watchdog\n");
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    // a real finalize takes ~40 ms, so a 1 ms finalize watchdog must fire on the first trial
    // build and degrade the session. main sets GGML_QNN_BUILD_TIMEOUT_MS only: the execute
    // limit (GGML_QNN_TIMEOUT_MS) never bounds a finalize, and the build never gets as far as
    // its validation execute
    const mul_mat_case c1 = { GGML_TYPE_F16, 256, 128, 64 };
    check(!probe_claim(qnn, c1), "first probe times out and is not claimed");

    // degraded session: every further shape is refused, fast
    const mul_mat_case c2 = { GGML_TYPE_F16, 128, 64, 64 };
    check(!probe_claim(qnn, c2), "degraded session refuses further shapes");

    // release the last reference BEFORE re-initializing: that is the branch that keeps a
    // degraded session instead of freeing it (an abandoned watchdog call may still touch it),
    // and it must leave the backend initializable (llama_new_context must not fail hard)
    ggml_backend_free(qnn);
    ggml_backend_t qnn2 = qnn_backend_init();
    check(qnn2 != nullptr, "backend init still succeeds after the degraded session was released");
    if (qnn2) {
        check(!probe_claim(qnn2, c2), "re-initialized backend still refuses shapes (degraded session kept)");
        ggml_backend_free(qnn2);
    }

    // the counters are written by the degrade and by each keep-degraded release: the timed-out
    // graph was created and its weight baked before finalize hung, and nothing built after
    check_stat("graphs_created", 1, "only the timed-out graph was created");
    check_stat("weights_baked", 1, "its weight was baked before finalize hung");

    // a finalize timeout is a genuine failure and must be PERSISTED so a rerun after a wedge
    // skips it: a real round-trip of the file format, header line included. (a validation-execute
    // timeout reaches the file only after a fast validation earlier in the session, a finalize
    // timeout needs none.) c1 is static (tagged WEIGHTS) at the 512 bucket main pins
    check_denylist_holds(dl, shape_key(c1, 512, /*is_static=*/true));

    // main ends the process right after the summary, see hard_exit; the runner keys off
    // this marker, not the exit code
    printf("WATCHDOG-CHECKS-%s\n", g_failures ? "FAILED" : "PASSED");
    fflush(stdout);
    return g_failures ? 1 : 0;
}

// the shape GGML_QNN_FAIL_EXECUTE targets: main derives the substring from it before the
// registry is touched, and the scenario probes it first
static const mul_mat_case g_fault_c1 = { GGML_TYPE_F16, 256, 128, 64 };

static int scenario_fault(void) {
    printf("scenario: fault (GGML_QNN_FAIL_EXECUTE=%s)\n", getenv("GGML_QNN_FAIL_EXECUTE"));
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    // the graph builds, bakes and finalizes, then the injected validation-execute failure
    // refuses the shape and degrades the session
    check(!probe_claim(qnn, g_fault_c1), "shape with the injected execute failure is not claimed");

    // degraded session: the next shape is refused before any trial build, so it takes
    // milliseconds, not the ~40 ms finalize and never a watchdog wait. the faulted shape is
    // the first of the session, so it is unproven and the failure degrades (scenario_clamp
    // covers the proven side)
    const mul_mat_case c2 = { GGML_TYPE_F16, 128, 64, 64 };
    const auto t0 = std::chrono::steady_clock::now();
    const bool claimed2 = probe_claim(qnn, c2);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    check(!claimed2, "degraded session refuses the next shape");
    char m[160];
    snprintf(m, sizeof(m), "refusal was fast, no trial build (%.0f ms)", ms);
    check(ms < 5000.0, m);

    // the load-time probe of an unallocated weight answers without a session call, from the
    // lock-free degraded flag: a wedged NPU must stop claiming weights at model load too, or
    // they are placed on it only to fall back at schedule time. c2 is F16 and under the IO
    // cap, so a healthy session would claim both probes (loadprobe pins that side)
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);
    check(!probe_unallocated(dev, c2, /*with_dummy=*/true),  "degraded session refuses an unallocated weight at placement (dummy buffer)");
    check(!probe_unallocated(dev, c2, /*with_dummy=*/false), "degraded session refuses a buffer-less graph matmul");

    // persistence rule (qnn-lib.cpp, ggml_qnn_denylist_add): a finalize timeout, a finalize
    // error and an execute timeout on a session that has executed at normal speed are HTP
    // verdicts on the shape and go to the file; an execute ERROR RETURN can be a transient
    // driver state and is remembered for this process only. so the file must not even exist
    check(file_size(dl) < 0, "execute-error verdict stays in-process: denylist file not created");

    // the backend must keep initializing after the degraded session is released
    ggml_backend_free(qnn);
    ggml_backend_t qnn2 = qnn_backend_init();
    check(qnn2 != nullptr, "backend init still succeeds after the degraded session was released");
    if (qnn2) {
        check(!probe_claim(qnn2, c2), "re-initialized backend still refuses shapes (degraded session kept)");
        ggml_backend_free(qnn2);
    }

    check_stat("graphs_created", 1, "only the faulted graph was created");
    check_stat("weights_baked", 1, "its weight was baked before the injected failure");
    return g_failures ? 1 : 0;
}

// the proven-shape rule: once a static shape has finalized and validated in the session, a
// build failure on another weight of that shape is the HTP running out of mappable weight
// memory (seen once at about 1170 MiB committed on QAIRT 2.45, on a loaded machine, with
// error codes that are not memory-specific), not a verdict on the shape. main sets GGML_QNN_FAIL_EXECUTE to the shape
// and GGML_QNN_FAIL_EXECUTE_SKIP=1, so the first weight builds for real and the second one
// takes the injected validation-execute failure. unlimited: main sets GGML_QNN_STATIC_BUDGET_MB=0,
// and the clamp must still replace that 0 (which means unlimited) with the committed bytes
static int scenario_clamp(bool unlimited) {
    printf("scenario: clamp (GGML_QNN_FAIL_EXECUTE=%s, SKIP=1%s)\n", getenv("GGML_QNN_FAIL_EXECUTE"),
           unlimited ? ", GGML_QNN_STATIC_BUDGET_MB=0" : "");
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    // three distinct weights of the faulted shape (count_claims uses g_fault_c1's 256x128 at
    // N=64): the first proves the shape, the second fails on the proven shape and clamps the
    // budget, the third is refused by the clamped budget before graphCreate
    const int claims = count_claims(qnn, 3, GGML_TYPE_F16);
    char msg[160];
    snprintf(msg, sizeof(msg), "only the weight built before the failure is claimed (1/3, got %d)", claims);
    check(claims == 1, msg);

    // a static weight of another shape is refused too: the clamp is on the budget, not the shape
    const mul_mat_case other = { GGML_TYPE_F16, 256, 64, 64 };
    check(!probe_claim(qnn, other), "clamped budget refuses a static weight of a different shape");

    // the session must NOT be degraded: a dynamic-weight graph (untagged buffer, nothing to
    // bake, so the budget does not apply) still builds, computes and matches the CPU
    // (its key ends in _dyn, so the fault hook's static shape key does not match it)
    const mul_mat_case dyn = g_fault_c1;
    bool claimed = false;
    std::vector<float> got, ref;
    const bool ok_q = run_mul_mat(qnn, dyn, got, &claimed, /*tag_weights=*/false);
    const bool ok_c = run_mul_mat(cpu, dyn, ref, nullptr, /*tag_weights=*/false);
    check(claimed, "session not degraded: a dynamic-weight matmul is still claimed");
    check(claimed && ok_q && ok_c && nmse(ref, got) < 5e-4, "and it computes and matches the CPU");

    // not a shape verdict, so nothing is persisted
    check(file_size(dl) < 0, "memory exhaustion on a proven shape is not denylisted: file not created");

    ggml_backend_free(qnn); // a clean session free, which writes the counters
    ggml_backend_free(cpu);

    check_stat("budget_clamped", 1, "the failure on the proven shape clamped the static budget");
    check_stat("graphs_created", 3, "two static graphs of the faulted shape plus the dynamic one; both clamped refusals came before graphCreate");
    check_stat("weights_baked", 2, "the proving weight and the one that failed, none after the clamp");

    // the failed weight was uncharged and the free subtracted the session's bytes from the
    // process-global committed count: the two must balance, or the count underflows and every
    // later static bake is refused. the clamp is per session, so a fresh one gets its whole
    // budget back. F32: its keys lack the FAIL_EXECUTE substring (an f16 key), whose match
    // count is process-global and would otherwise fire again
    qnn = qnn_backend_init_checked();
    const int claims2 = count_claims(qnn, 3, GGML_TYPE_F32);
    snprintf(msg, sizeof(msg), "a fresh session after the clamp claims all three F32 weights (3/3, got %d)", claims2);
    check(claims2 == 3, msg);
    ggml_backend_free(qnn);
    check_stat("graphs_created", 6, "the three above plus one static graph per F32 weight");
    check_stat("weights_baked", 5, "the two above plus the three F32 weights");
    return g_failures ? 1 : 0;
}

// the fault-hook modes below (GGML_QNN_DELAY_EXECUTE, GGML_QNN_FAIL_FINALIZE) run F32 weights
// at GGML_QNN_NPAD=64 only: F32-weight executes of these shapes took a few ms on the dev box
// even under load, while fp16-weight ones took seconds there. the limits also cover
// the first executes of each process, whose cost nothing measures, so they sit at 500 ms and
// 1 s, and every injected delay clears its limit by 3x or more
// A, B and C are distinct shapes, so a hook keyed on one never matches another
static const mul_mat_case g_hook_a = { GGML_TYPE_F32, 256, 128, 64 };
static const mul_mat_case g_hook_b = { GGML_TYPE_F32, 256,  64, 64 };
static const mul_mat_case g_hook_c = { GGML_TYPE_F32, 128,  64, 64 };
static const int          g_hook_npad = 64;

// slow-*: GGML_QNN_SLOW_EXEC_MS, and the delay that crosses it. slow-compute's delay also
// crosses the backend's fixed 1000 ms exec_slow threshold, which slow-validate's must not need
static const int g_slow_ms               = 500;
static const int g_slow_validate_delay   = 2000;
static const int g_slow_compute_delay    = 2500;
// *-timeout: GGML_QNN_TIMEOUT_MS, and the delay that outlives it. the process ends while the
// abandoned worker is still sleeping in the hook, before graphExecute, so no HTP call is in
// flight at exit
static const int g_timeout_ms            = 1000;
static const int g_timeout_delay         = 3000;

// the denylist-append seed: a key for a shape no mode here builds, deliberately without a
// trailing newline
static const char * g_append_seed = "MUL_MAT_f32_f32_512x512x1x1_512x64x1x1_s";

// these modes keep a degraded session (and some an abandoned worker): main ends them through
// hard_exit, and the runner keys off this marker, like watchdog
static void print_marker(const char * tag) {
    printf("%s-CHECKS-%s\n", tag, g_failures ? "FAILED" : "PASSED");
    fflush(stdout);
}

// validation-time slow degrade: main sets GGML_QNN_SLOW_EXEC_MS to g_slow_ms and delays B's
// validation execute by g_slow_validate_delay. the completed-but-slow validation is a verdict on the device, not the
// shape: B is not claimed and never denylisted, the session degrades slow_only, so a new shape
// is refused while A's graph, built before the verdict, still computes
static int scenario_slow_validate(void) {
    printf("scenario: slow-validate (GGML_QNN_SLOW_EXEC_MS=%d, B's validation delayed %d ms)\n", g_slow_ms, g_slow_validate_delay);
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    std::vector<float> ref, got;
    GGML_ASSERT(run_mul_mat(cpu, g_hook_a, ref));

    held_mul_mat a;
    held_init(a, qnn, g_hook_a);
    const bool claimed_a = held_claimed(qnn, a);
    check(claimed_a, "A claimed: its undelayed validation execute stays under GGML_QNN_SLOW_EXEC_MS");
    check(claimed_a && held_compute(qnn, a, got) && nmse(ref, got) < 5e-4, "A computes and matches the CPU before the degrade");

    check(!probe_claim(qnn, g_hook_b), "B not claimed: its delayed validation completed, but slower than GGML_QNN_SLOW_EXEC_MS");
    check(file_size(dl) < 0, "a slow validation is a device state, not a shape verdict: denylist file not created");
    check(!probe_claim(qnn, g_hook_c), "slow_only session refuses a new shape");

    // slow_only, not a hard degrade: the graph built before the verdict keeps running
    got.clear();
    const bool ok2 = claimed_a && held_compute(qnn, a, got);
    check(ok2 && nmse(ref, got) < 5e-4, "A's cached graph still computes after the slow_only degrade and matches the CPU");

    held_free(a);
    ggml_backend_free(qnn); // degraded, so kept: the release flushes the counters
    ggml_backend_free(cpu);

    // A and B were created (B's delayed validation came after its finalize); C was refused by
    // the degraded session before any build. both are static, so each baked once
    check_stat("graphs_created", 2, "A and B; C refused before any build");
    check_stat("weights_baked", 2, "one bake each for A and B");
    // exec_count counts compute-time executes only: A's two computes. the degrade wrote 1, so
    // 2 also shows the release flush saw the compute that ran after it
    check_stat("exec_count", 2, "A's compute before and after the degrade; validation executes are not counted");
    check_stat("exec_slow", 0, "no compute-time execute reached 1000 ms");
    print_marker("SLOW-VALIDATE");
    return g_failures ? 1 : 0;
}

// compute-time slow degrade: main sets GGML_QNN_NO_PREVALIDATE=1, GGML_QNN_SLOW_EXEC_MS to
// g_slow_ms and delays A's first execute (one-shot) by g_slow_compute_delay. A is claimed without a validation execute, so
// the slow verdict lands at compute: that node still succeeds with the right result, the
// session degrades slow_only, A keeps computing, and a new shape is refused
static int scenario_slow_compute(void) {
    printf("scenario: slow-compute (GGML_QNN_NO_PREVALIDATE=1, GGML_QNN_SLOW_EXEC_MS=%d, A's first compute delayed %d ms)\n", g_slow_ms, g_slow_compute_delay);
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    std::vector<float> ref, got;
    GGML_ASSERT(run_mul_mat(cpu, g_hook_a, ref));

    held_mul_mat a;
    held_init(a, qnn, g_hook_a);
    const bool claimed_a = held_claimed(qnn, a);
    check(claimed_a, "A claimed without a validation execute");

    const bool ok1 = claimed_a && held_compute(qnn, a, got);
    check(ok1 && nmse(ref, got) < 5e-4, "A's delayed first compute still succeeds and matches the CPU");

    got.clear();
    const bool ok2 = claimed_a && held_compute(qnn, a, got);
    check(ok2 && nmse(ref, got) < 5e-4, "A's second compute runs its cached graph under slow_only and matches the CPU");

    check(!probe_claim(qnn, g_hook_c), "slow_only session refuses a new shape");

    // a node the scheduler placed before the verdict but whose graph was never built fails at
    // compute instead of building now (compute_node's slow_only guard)
    held_mul_mat c;
    held_init(c, qnn, g_hook_c);
    std::vector<float> got_c;
    check(!held_compute(qnn, c, got_c), "a compute of a shape with no cached graph fails under slow_only, no build");
    held_free(c);

    held_free(a);
    ggml_backend_free(qnn); // degraded, so kept: the release flushes the counters
    ggml_backend_free(cpu);

    check_stat("graphs_created", 1, "A only: the new shape was refused by supports_op and by compute before any build");
    // the degrade wrote exec_count 1 (the delayed compute); 2 is only in the file if the
    // kept-degraded release flushed the compute that ran after the degrade
    check_stat("exec_count", 2, "both computes of A; the refused compute of C never executed");
    check_stat("exec_slow", 1, "only the delayed compute reached 1000 ms");
    const long long max_ms = read_stat("exec_max_ms");
    char m[160];
    snprintf(m, sizeof(m), "exec_max_ms covers the injected delay (%lld >= %d)", max_ms, g_slow_compute_delay);
    check(max_ms >= g_slow_compute_delay, m);
    print_marker("SLOW-COMPUTE");
    return g_failures ? 1 : 0;
}

// validation-execute timeout: main sets GGML_QNN_TIMEOUT_MS to g_timeout_ms and delays B's
// validation by g_timeout_delay. the timeout degrades the session and abandons the worker; the shape reaches the
// denylist FILE only if a validation already completed at normal speed in this session
// (healthy_seen). the default leg validates A fast first, so B's key is persisted; the cold leg
// times B out first, so nothing is written
static int scenario_validate_timeout(bool cold) {
    printf("scenario: validate-timeout%s (GGML_QNN_TIMEOUT_MS=%d, B's validation delayed %d ms)\n", cold ? " cold" : "", g_timeout_ms, g_timeout_delay);
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    if (!cold) {
        check(probe_claim(qnn, g_hook_a), "A claimed: a validation at normal speed, the session has seen a healthy execute");
    }
    check(!probe_claim(qnn, g_hook_b), "B not claimed: its delayed validation outlived GGML_QNN_TIMEOUT_MS");
    check(!probe_claim(qnn, g_hook_c), "degraded session refuses a new shape");

    if (cold) {
        check(file_size(dl) < 0, "no healthy validation seen first: the timeout stays in-process, denylist file not created");
    } else {
        check_denylist_holds(dl, shape_key(g_hook_b, g_hook_npad, /*is_static=*/true));
    }

    ggml_backend_free(qnn); // degraded, so kept: the release flushes the counters

    // C was refused by the degraded session before any build
    check_stat("graphs_created", cold ? 1 : 2, cold ? "B only" : "A and B");
    // probe_claim computes what it claimed: A once in the default leg. validation executes
    // are not counted, and nothing else was claimed
    check_stat("exec_count", cold ? 0 : 1, cold ? "no compute ran" : "A's one compute");
    print_marker("VALIDATE-TIMEOUT");
    return g_failures ? 1 : 0;
}

// compute-time execute timeout: main sets GGML_QNN_NO_PREVALIDATE=1, GGML_QNN_TIMEOUT_MS to
// g_timeout_ms and delays A's first compute by g_timeout_delay. the node fails, the graph is demoted and the session
// hard-degrades: the same and a new shape are refused, a second compute fails before any
// execute, and with no fast validation ever seen the shape is not persisted
static int scenario_compute_timeout(void) {
    printf("scenario: compute-timeout (GGML_QNN_NO_PREVALIDATE=1, GGML_QNN_TIMEOUT_MS=%d, A's first compute delayed %d ms)\n", g_timeout_ms, g_timeout_delay);
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    held_mul_mat a;
    held_init(a, qnn, g_hook_a);
    const bool claimed_a = held_claimed(qnn, a);
    check(claimed_a, "A claimed without a validation execute");

    std::vector<float> got;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok1 = claimed_a && held_compute(qnn, a, got);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    check(claimed_a && !ok1, "A's delayed first compute timed out and failed the node");
    // about g_timeout_ms when the watchdog cuts the call off, at least g_timeout_delay if it
    // waited out the delay; the midpoint separates the two
    const double bound = (g_timeout_ms + g_timeout_delay) / 2.0;
    char m[192];
    snprintf(m, sizeof(m), "the watchdog returned at GGML_QNN_TIMEOUT_MS, not after the delay (%.0f ms < %.0f)", ms, bound);
    check(ms < bound, m);

    check(!held_claimed(qnn, a), "hard-degraded: the timed-out shape is refused too");
    check(!probe_claim(qnn, g_hook_c), "hard-degraded session refuses a new shape");
    check(!held_compute(qnn, a, got), "a second compute of A fails: a hard degrade, not slow_only");
    check(file_size(dl) < 0, "no healthy validation seen: the compute timeout is not persisted, denylist file not created");

    held_free(a);
    ggml_backend_free(qnn); // degraded, so kept: the release flushes the counters

    check_stat("graphs_created", 1, "A only");
    // compute_node counts an execute that timed out; the second compute failed on the
    // degraded session before reaching one
    check_stat("exec_count", 1, "the timed-out compute; the second failed before any execute");
    print_marker("COMPUTE-TIMEOUT");
    return g_failures ? 1 : 0;
}

// finalize error, both sides of the proven split: main sets GGML_QNN_FAIL_FINALIZE=_s_w (every
// static graph key, never a _dyn one) with GGML_QNN_FAIL_FINALIZE_SKIP=1. in order:
//   A1 (seed K+M)  match 1, skipped: finalizes and validates for real, A's shape is now proven
//   B1             match 2, unproven: persisted to the file, not claimed
//   A2 (seed 1)    match 3, proven: the budget clamps to what is committed (A1), not denylisted
//   A3 (seed 2)    refused by the clamped budget before graphCreate, no match
// the injected failures leave unfinalized graphs in the context, and one of those can hang
// the next execute through its deferred prepare, so nothing executes after B1. the clean
// session free at the end then releases the context with both unfinalized graphs and their
// kept staging (keep_bake) still referenced
static int scenario_finalize_error(void) {
    printf("scenario: finalize-error (GGML_QNN_FAIL_FINALIZE=%s, SKIP=1)\n", getenv("GGML_QNN_FAIL_FINALIZE"));
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    check(probe_claim(qnn, g_hook_a), "A1 claimed: the first match finalizes for real and proves the shape");
    check(!probe_claim(qnn, g_hook_b), "B1, unproven, not claimed after its finalize error");

    // count_claims probes g_hook_a's shape (256x128 at N=64) with fresh content, so both are
    // new graph keys even if an address is reused
    const int claims = count_claims(qnn, 2, GGML_TYPE_F32);
    char m[192];
    snprintf(m, sizeof(m), "two more weights of the proven shape: the first fails finalize and clamps, the second is refused (0/2, got %d)", claims);
    check(claims == 0, m);

    check(!probe_claim(qnn, g_hook_b), "B is refused again without a build: denylisted in-process");
    check(!probe_claim(qnn, g_hook_c), "a static weight of a new shape is refused by the clamped budget");

    // B's key alone: the proven failure stays off the file
    check_denylist_holds(dl, shape_key(g_hook_b, g_hook_npad, /*is_static=*/true));

    ggml_backend_free(qnn); // not degraded: a clean session free, which writes the counters

    check_stat("budget_clamped", 1, "the finalize error on the proven shape clamped the budget");
    // A1, B1 and A2 reached graphCreate and created their STATIC tensor (the bake counter)
    // before finalize; A3 and C were refused by policy, B's second probe by the denylist, or by
    // B1's cached failed graph when the freed buffer address is reused (same key)
    check_stat("graphs_created", 3, "A1, B1, A2");
    check_stat("weights_baked", 3, "A1, B1, A2: the weight is baked before finalize");
    // probe_claim computes what it claimed, A1 only, and that before any injected failure
    check_stat("exec_count", 1, "A1's compute, the only one");
    return g_failures ? 1 : 0;
}

// denylist append to a NON-EMPTY file: main seeds it (binary) with one unrelated key and no
// trailing newline, and GGML_QNN_FAIL_FINALIZE fails B's finalize (unproven: persisted). the
// backend must repair the missing newline and must not write a header into a non-empty file
static int scenario_denylist_append(void) {
    printf("scenario: denylist-append (GGML_QNN_FAIL_FINALIZE=%s)\n", getenv("GGML_QNN_FAIL_FINALIZE"));
    const char * dl = getenv("GGML_QNN_DENYLIST");
    GGML_ASSERT(dl != nullptr);
    ggml_backend_t qnn = qnn_backend_init_checked();

    check(!probe_claim(qnn, g_hook_b), "B not claimed after its finalize error");

    const std::string content = read_file(dl);
    const std::string key     = shape_key(g_hook_b, g_hook_npad, /*is_static=*/true);
    const std::string want    = std::string(g_append_seed) + "\n" + key + "\n";
    check(content == want, "denylist file holds the seed, the repaired newline and B's key, no header");
    if (content != want) {
        printf("        file content: \"%s\"\n", content.c_str());
    }

    ggml_backend_free(qnn); // not degraded: a clean session free, which writes the counters
    check_stat("graphs_created", 1, "B only");
    return g_failures ? 1 : 0;
}

// claim and compute a held F32 matmul, checking the result against the exact reference for
// what it holds right now. returns whether it was claimed and matched
static bool held_check(ggml_backend_t backend, held_mul_mat & h, const char * what) {
    const std::vector<float> ref = held_ref(h);
    const bool claimed = held_claimed(backend, h);
    std::vector<float> got;
    const bool ok = claimed && held_compute(backend, h, got);
    const double e = ok ? nmse(ref, got) : -1.0;
    char msg[224];
    snprintf(msg, sizeof(msg), "%s: claimed %d, computes %d, matches the reference (nmse %.2e)", what, (int) claimed, (int) ok, e);
    check(ok && e < 5e-4, msg);
    return ok && e < 5e-4;
}

// session sharing and weight identity, all F32 at NPAD 64 on the default 1 GiB budget:
//   1. two backends on one session: A computes, A is freed, B computes the same held matmul
//      on A's cached graph (a refcount slip would free the session under B)
//   2. a third backend after the last free starts a fresh session that works
//   3. a WEIGHTS-tagged weight in a buffer_from_host_ptr buffer (how llama maps a model file)
//      bakes like any model weight
//   4. new content written at the SAME weight address bakes a new graph: the graph key carries
//      a content fingerprint beside the address, so the stale bake is not served
static int scenario_reuse(void) {
    printf("scenario: reuse\n");
    char m[224];

    // 1
    ggml_backend_t qa = qnn_backend_init_checked();
    ggml_backend_t qb = qnn_backend_init_checked();
    held_mul_mat h;
    held_init(h, qa, g_hook_a);
    held_check(qa, h, "A on the first backend");
    ggml_backend_free(qa); // refs 2 -> 1: the session and its graph cache must stay
    std::vector<float> got;
    const std::vector<float> ref_a = held_ref(h);
    const bool ok_b = held_compute(qb, h, got);
    snprintf(m, sizeof(m), "the second backend computes A after the first was freed (nmse %.2e)", ok_b ? nmse(ref_a, got) : -1.0);
    check(ok_b && nmse(ref_a, got) < 5e-4, m);
    held_free(h);
    ggml_backend_free(qb); // refs 1 -> 0: the session is freed and writes the counters

    // the claim built A's graph (a miss) and each compute looked it up (a hit). had the first
    // free torn the session down, B's compute would have rebuilt the graph or crashed
    check_stat("graphs_created", 1, "A's graph only: the second backend's compute reused it");
    check_stat("graph_cache_hits", 2, "one lookup per compute, one on each backend");
    check_stat("exec_count", 2, "one compute on each backend");

    // 2
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "a backend initializes a fresh session after the last free");
    if (!qnn) {
        return 1;
    }
    held_init(h, qnn, g_hook_a);
    held_check(qnn, h, "A on the fresh session");
    held_free(h);

    // 3: B's shape and content (the fill held_init uses), in a page-aligned host allocation
    {
        const mul_mat_case & c = g_hook_b;
        const size_t nbytes = (size_t) c.K * c.M * sizeof(float);
        std::vector<uint8_t> host(nbytes + 4096);
        void * base = (void *) (((uintptr_t) host.data() + 4095) & ~(uintptr_t) 4095);

        ggml_backend_dev_t dev = ggml_backend_get_device(qnn);
        ggml_backend_buffer_t buf_w = ggml_backend_dev_buffer_from_host_ptr(dev, base, nbytes, nbytes);
        check(buf_w != nullptr, "buffer_from_host_ptr wraps a host allocation");
        if (buf_w) {
            ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
            ggml_init_params gp = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx_w = ggml_init(wp);
            ggml_context * ctx   = ggml_init(gp);
            ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.K, c.M);
            GGML_ASSERT(ggml_backend_tensor_alloc(buf_w, w, base) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.N);
            ggml_tensor * d = ggml_mul_mat(ctx, w, x);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, d);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, qnn);

            std::vector<float> wf((size_t) c.K * c.M), xf((size_t) c.K * c.N);
            fill_uniform(wf, (unsigned) (c.K + c.M));
            fill_uniform(xf, (unsigned) (c.K * 31 + c.N));
            ggml_backend_tensor_set(w, wf.data(), 0, nbytes);
            ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
            const std::vector<float> ref = ref_mul_mat(wf, xf, c.K, c.M, c.N);

            const bool claimed = ggml_backend_dev_supports_op(dev, d);
            check(claimed, "a WEIGHTS-tagged weight in a host-pointer buffer is claimed");
            std::vector<float> out((size_t) c.N * c.M);
            const bool ok = claimed && ggml_backend_graph_compute(qnn, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(d, out.data(), 0, out.size() * sizeof(float));
            }
            snprintf(m, sizeof(m), "and it computes, matching the reference (nmse %.2e)", ok ? nmse(ref, out) : -1.0);
            check(ok && nmse(ref, out) < 5e-4, m);

            ggml_backend_buffer_free(buf);
            ggml_backend_buffer_free(buf_w); // does not own the host memory
            ggml_free(ctx);
            ggml_free(ctx_w);
        }
    }

    // 4
    held_init(h, qnn, g_hook_c);
    held_check(qnn, h, "C with its first content");
    const std::vector<float> ref_old = held_ref(h);
    const std::vector<float> w_old   = held_weight(h);
    std::vector<float> w_new(w_old.size());
    fill_uniform(w_new, 4242);
    // test wiring: the fingerprint samples the first and last 512 bytes, both must change
    GGML_ASSERT(memcmp(w_old.data(), w_new.data(), 512) != 0);
    GGML_ASSERT(memcmp((const uint8_t *) w_old.data() + w_old.size() * sizeof(float) - 512,
                       (const uint8_t *) w_new.data() + w_new.size() * sizeof(float) - 512, 512) != 0);
    held_set_weight(h, w_new);
    // and the stale bake must give a result the check below can tell apart
    GGML_ASSERT(nmse(held_ref(h), ref_old) > 0.1);
    held_check(qnn, h, "C with new content at the same address: a new bake, not the stale graph");
    held_free(h);

    ggml_backend_free(qnn); // a clean session free, which writes the counters

    // process-global totals across both sessions. every claim above built a static graph and
    // baked its weight: A (session 1), A (session 2), the host-pointer weight, C's two
    // contents. every compute is a cache hit of the graph its claim built, and one execute:
    // 2 in session 1, then A, the host-pointer weight and C twice
    check_stat("graphs_created", 5, "A twice (one per session), the host-pointer weight, C's two contents");
    check_stat("weights_baked", 5, "each of those graphs baked its weight, the host-pointer one included");
    check_stat("graph_cache_hits", 6, "one hit per compute");
    check_stat("exec_count", 6, "one execute per compute");
    return g_failures ? 1 : 0;
}

// GGML_QNN_NO_STATIC_WEIGHTS (set by main): a WEIGHTS-tagged weight goes the dynamic path,
// where it is copied into the graph's IO buffer once and the copy is skipped while address and
// content fingerprint still match. one graph (the key is the padded shape), four computes
static int scenario_dyncache(void) {
    printf("scenario: dyncache (GGML_QNN_NO_STATIC_WEIGHTS=1)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();

    held_mul_mat h;
    held_init(h, qnn, g_hook_a);
    const std::vector<float> ref_orig = held_ref(h);
    const std::vector<float> w_orig   = held_weight(h);

    // the first compute copies the weight, the second finds it in place
    held_check(qnn, h, "first compute (copies the weight)");
    std::vector<float> got;
    bool ok = held_compute(qnn, h, got);
    char m[224];
    snprintf(m, sizeof(m), "second compute of the unchanged weight matches (nmse %.2e)", ok ? nmse(ref_orig, got) : -1.0);
    check(ok && nmse(ref_orig, got) < 5e-4, m);

    // the skip is invisible while the content really is unchanged. a WEIGHTS buffer is
    // immutable by contract, and the fingerprint samples only nbytes and the first and last
    // 512 bytes, so rewriting the bytes between them breaks the contract without moving the
    // fingerprint: a compute that skips the copy still returns the ORIGINAL result, one that
    // re-copied every time returns the new one. if the fingerprint ever covers the whole
    // tensor, this check must flip to the rewritten content
    const size_t n_edge = 512 / sizeof(float);
    std::vector<float> w_mid = w_orig;
    std::vector<float> fresh(w_mid.size());
    fill_uniform(fresh, 777);
    for (size_t i = n_edge; i + n_edge < w_mid.size(); i++) {
        w_mid[i] = fresh[i];
    }
    held_set_weight(h, w_mid);
    GGML_ASSERT(nmse(held_ref(h), ref_orig) > 0.1); // test wiring: the two results differ
    ok = held_compute(qnn, h, got);
    snprintf(m, sizeof(m), "the copy was skipped: edges and address unchanged, the cached copy served (nmse %.2e vs the original)",
             ok ? nmse(ref_orig, got) : -1.0);
    check(ok && nmse(ref_orig, got) < 5e-4, m);

    // a real content change at the same address moves the fingerprint: the weight is copied again
    std::vector<float> w_new(w_orig.size());
    fill_uniform(w_new, 4243);
    GGML_ASSERT(memcmp(w_new.data(), w_mid.data(), 512) != 0);
    GGML_ASSERT(memcmp(w_new.data() + w_new.size() - n_edge, w_mid.data() + w_mid.size() - n_edge, 512) != 0);
    held_set_weight(h, w_new);
    const std::vector<float> ref_new = held_ref(h);
    ok = held_compute(qnn, h, got);
    snprintf(m, sizeof(m), "new content at the same address is copied again and matches it (nmse %.2e)", ok ? nmse(ref_new, got) : -1.0);
    check(ok && nmse(ref_new, got) < 5e-4, m);

    held_free(h);
    ggml_backend_free(qnn);

    check_stat("weights_baked", 0, "no static bake with GGML_QNN_NO_STATIC_WEIGHTS");
    check_stat("graphs_created", 1, "one dynamic graph: its key is the padded shape, no weight identity");
    check_stat("graph_cache_hits", 4, "the claim created the graph, each of the four computes found it");
    check_stat("exec_count", 4, "one execute per compute");
    return g_failures ? 1 : 0;
}

// GGML_QNN_QUANTIZED (set by main): an UNTAGGED Q4_0 weight cannot be baked, and without the
// opt-in it is a policy reject (loadprobe); with it the weight is dequantized to fp16 into the
// graph's input on every execute. one small graph: fp16 weights have been the slow case on the dev box
static int scenario_quantized(void) {
    printf("scenario: quantized (GGML_QNN_QUANTIZED=1)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();
    ggml_backend_t cpu = cpu_backend_init();

    const mul_mat_case c = { GGML_TYPE_Q4_0, 256, 128, 64 };
    std::vector<float> ref, got;
    GGML_ASSERT(run_mul_mat(cpu, c, ref, nullptr, /*tag_weights=*/false));

    held_mul_mat h;
    held_init(h, qnn, c, /*tag_weights=*/false);
    const bool claimed = held_claimed(qnn, h);
    check(claimed, "untagged Q4_0 weight claimed on the per-execute dequant path");
    const bool ok = claimed && held_compute(qnn, h, got);
    // the same row dequant to fp16 as the static bake, which basic holds to 5e-4 for Q4_0; the
    // CPU reference quantizes the activations to Q8_0, about 1e-5 nmse, well inside that
    const double e = ok ? nmse(ref, got) : -1.0;
    char m[160];
    snprintf(m, sizeof(m), "and it computes, matching the CPU (nmse %.2e)", e);
    check(ok && e < 5e-4, m);
    held_free(h);

    ggml_backend_free(qnn);
    ggml_backend_free(cpu);
    check_stat("graphs_created", 1, "one dynamic graph");
    check_stat("weights_baked", 0, "an untagged weight never bakes");
    check_stat("exec_count", 1, "the one compute");
    return g_failures ? 1 : 0;
}

// main sets malformed values: GGML_QNN_NPAD=abc, GGML_QNN_MIN_DIM=-5 (below its floor of 1),
// GGML_QNN_IO_MAX_KB=12k (a partial number). each must fall back to its default, with a warning,
// rather than take a partial parse: 12k read as 12 would refuse every matmul, abc read as 0
// would move every graph to the exact-pow2 bucket
static int scenario_envparse(void) {
    printf("scenario: envparse (GGML_QNN_NPAD=abc, GGML_QNN_MIN_DIM=-5, GGML_QNN_IO_MAX_KB=12k)\n");
    ggml_backend_t qnn = qnn_backend_init_checked();

    const mul_mat_case rows16 = { GGML_TYPE_F32, 256, 128, 16 };
    check(!probe_claim(qnn, rows16), "N=16 refused: the default GGML_QNN_MIN_DIM of 32 applies");
    // padded to the default 512 bucket: 512 KiB in, 256 KiB out, under the default 1 MiB cap
    const mul_mat_case under = { GGML_TYPE_F32, 256, 128, 64 };
    check(probe_claim(qnn, under), "256x128 N=64 claimed: 512 KiB of padded IO is under the default GGML_QNN_IO_MAX_KB");

    ggml_backend_free(qnn);
    check_stat("graphs_created", 1, "the claimed case only, the N=16 one was refused before any build");
    check_stat("pad_n_last", 512, "the claimed case landed on the default bucket 512");
    return g_failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "basic";

    // the backend latches env at first use, so all setup happens before the registry is touched
    const char * dl_path = "test-qnn-lifecycle-denylist.tmp";
    const bool uses_dl = mode == "budget" || mode == "denylist" || mode == "denylist-probe" || mode == "watchdog" ||
                         mode == "fault" || mode == "clamp" ||
                         mode == "slow-validate" || mode == "validate-timeout" || mode == "compute-timeout" ||
                         mode == "finalize-error" || mode == "denylist-append";
    // these leave a degraded session behind on purpose (the timeout ones also an abandoned
    // worker), so they end through hard_exit and never reach DLL detach
    const bool keeps_degraded = mode == "watchdog" || mode == "fault" || mode == "slow-validate" || mode == "slow-compute" ||
                                mode == "validate-timeout" || mode == "compute-timeout";

    // every mode starts from the backend's defaults, whatever the caller's shell exports: an
    // inherited GGML_QNN_NPAD=64 moved the hard-coded 512-bucket shape keys, GGML_QNN_DISABLE
    // skipped everything, a denylist file blocked shapes. the list is every variable the
    // backend reads (getenv and ggml_qnn_env_ll in ggml/src/ggml-qnn), except GGML_QNN_DEBUG,
    // which only raises the QNN log level. kept: the variable the ctest variant exists for
    // (its ENVIRONMENT property, named by argv or fixed by the mode), and for the two modes a
    // caller tunes on purpose their limits (bigstatic also its bucket and IO cap)
    const char * keep_env = mode == "modelscale" && argc > 3 ? argv[3]
                          : mode == "elementwise-on"         ? "GGML_QNN_ELEMENTWISE"
                                                             : nullptr;
    const bool tuned = mode == "bigstatic" || mode == "health";
    static const char * const backend_env[] = {
        "GGML_QNN_AOT_TEST",          "GGML_QNN_BUILD_TIMEOUT_MS",  "GGML_QNN_DENYLIST",
        "GGML_QNN_DELAY_EXECUTE",     "GGML_QNN_DELAY_EXECUTE_MS",  "GGML_QNN_DELAY_EXECUTE_SKIP",
        "GGML_QNN_FAIL_FINALIZE",     "GGML_QNN_FAIL_FINALIZE_SKIP",
        "GGML_QNN_DISABLE",           "GGML_QNN_ELEMENTWISE",       "GGML_QNN_FAIL_EXECUTE",
        "GGML_QNN_FAIL_EXECUTE_SKIP", "GGML_QNN_IO_MAX_KB",         "GGML_QNN_MIN_DIM",
        "GGML_QNN_MIN_ELEMENTS",      "GGML_QNN_NO_BURST",          "GGML_QNN_NO_OPT",
        "GGML_QNN_NO_PREVALIDATE",    "GGML_QNN_NO_STATIC_WEIGHTS", "GGML_QNN_NPAD",
        "GGML_QNN_QUANTIZED",         "GGML_QNN_SHARED_MEM",        "GGML_QNN_SLOW_EXEC_MS",
        "GGML_QNN_STATIC_BUDGET_MB",  "GGML_QNN_STATS",             "GGML_QNN_TIMEOUT_MS",
    };
    for (const char * name : backend_env) {
        if (keep_env && strcmp(name, keep_env) == 0) {
            continue;
        }
        if (tuned && (strcmp(name, "GGML_QNN_SLOW_EXEC_MS") == 0 || strcmp(name, "GGML_QNN_TIMEOUT_MS") == 0 ||
                      strcmp(name, "GGML_QNN_BUILD_TIMEOUT_MS") == 0)) {
            continue;
        }
        if (mode == "bigstatic" && (strcmp(name, "GGML_QNN_NPAD") == 0 || strcmp(name, "GGML_QNN_IO_MAX_KB") == 0)) {
            continue;
        }
        unset_env(name);
    }

    // the functional scenarios must not depend on how fast the device is: a slow HTP would
    // otherwise refuse every shape and fail them all for one reason. test-qnn-health covers speed
    if (!getenv("GGML_QNN_SLOW_EXEC_MS")) {
        set_env("GGML_QNN_SLOW_EXEC_MS", "0");
    }
    if (mode == "basic") {
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", "512"); // the scenario's shapes are laid out around this bucket
    } else if (mode == "bigstatic") {
        set_env("GGML_QNN_MIN_DIM", "1");
        // the diagnostic exists to reach the padded-IO hang thresholds that GGML_QNN_IO_MAX_KB
        // now refuses ahead of time (every 512-class case is exactly 1 MiB in at the default
        // bucket); lift the cap unless the caller set one
        if (!getenv("GGML_QNN_IO_MAX_KB")) {
            set_env("GGML_QNN_IO_MAX_KB", "1048576");
        }
    } else if (mode == "modelscale") {
        enable_stats();
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", argc > 2 ? argv[2] : "64");
        // the -noopt, -shm and -nostatic variants pass the same args and differ only by a
        // ctest ENVIRONMENT property; naming it here makes broken wiring fail the test
        if (argc > 3) {
            g_require_env = argv[3];
        }
    } else if (mode == "budget") {
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_STATIC_BUDGET_MB", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", "64"); // keeps the over-budget probe under the IO cap, see the scenario
    } else if (mode == "denylist-probe") {
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
        // the bucket a -ub 32 run builds in, and the one ggml_qnn_pad_n(1) resolves to
        set_env("GGML_QNN_NPAD", "32");
        FILE * f = fopen(dl_path, "w");
        if (!f) {
            fprintf(stderr, "cannot create %s\n", dl_path);
            return 1;
        }
        // byte-for-byte what a real run at this bucket persists - N is 32, NOT the 512 the
        // placement probe arrives with
        fprintf(f, "MUL_MAT_f16_f32_256x128x1x1_256x32x1x1_s\n");
        fclose(f);
    } else if (mode == "denylist") {
        // argv[2] "noopt": file entries are advisory under GGML_QNN_NO_OPT
        if (argc > 2 && strcmp(argv[2], "noopt") != 0) {
            fprintf(stderr, "denylist: unknown variant %s (noopt)\n", argv[2]);
            return 1;
        }
        if (argc > 2) {
            set_env("GGML_QNN_NO_OPT", "1");
        }
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", "512"); // the seeded keys below carry this bucket
        FILE * f = fopen(dl_path, "w");
        if (!f) {
            fprintf(stderr, "cannot create %s\n", dl_path);
            return 1;
        }
        // byte-for-byte the backend's shape-key format (ggml_qnn_shape_key): both variants
        // carry the PADDED N (60 -> 512 at the pinned bucket); the static graph key's weight
        // suffix is not part of the shape key
        fprintf(f, "MUL_MAT_f16_f32_256x128x1x1_256x512x1x1_s\n");
        fprintf(f, "MUL_MAT_f16_f32_256x64x1x1_256x512x1x1_dyn\n");
        fclose(f);
    } else if (mode == "watchdog") {
        remove(dl_path);
        enable_stats();
        // the build limit bounds graph finalize only, the first timed call of a trial build;
        // GGML_QNN_TIMEOUT_MS bounds every execute, the validation one included, and is left
        // at its default: the build never gets that far
        set_env("GGML_QNN_BUILD_TIMEOUT_MS", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", "512"); // the scenario expects c1's key at this bucket
    } else if (mode == "fault") {
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", "512"); // the bucket in the shape key below
        // the hook matches a substring of the graph key, which starts with the shape key
        set_env("GGML_QNN_FAIL_EXECUTE", shape_key(g_fault_c1, 512, /*is_static=*/true).c_str());
    } else if (mode == "clamp") {
        // argv[2] "unlimited": GGML_QNN_STATIC_BUDGET_MB=0, the clamp must still bite
        if (argc > 2 && strcmp(argv[2], "unlimited") != 0) {
            fprintf(stderr, "clamp: unknown variant %s (unlimited)\n", argv[2]);
            return 1;
        }
        if (argc > 2) {
            set_env("GGML_QNN_STATIC_BUDGET_MB", "0");
        }
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", "512"); // the bucket in the shape key below
        // count_claims probes g_fault_c1's shape; the first match builds, the second fails
        set_env("GGML_QNN_FAIL_EXECUTE", shape_key(g_fault_c1, 512, /*is_static=*/true).c_str());
        set_env("GGML_QNN_FAIL_EXECUTE_SKIP", "1");
    } else if (mode == "disable") {
        set_env("GGML_QNN_DISABLE", "1");
    } else if (mode == "mindim") {
        // deliberately no env: the point is behaviour at the DEFAULT configuration, which the
        // unset loop above guarantees
    } else if (mode == "elementwise") {
        // no backend env either; the counters only observe
        enable_stats();
    } else if (mode == "loadprobe") {
        enable_stats();
        // the default, pinned because the scenario's IO-cap cases are computed for this bucket
        set_env("GGML_QNN_NPAD", "512");
    } else if (mode == "elementwise-on") {
        enable_stats();
        set_env("GGML_QNN_MIN_ELEMENTS", "65536"); // keeps the ADD trial at 256 KiB IO, see the scenario
        g_require_env = "GGML_QNN_ELEMENTWISE";
    } else if (mode == "health") {
        enable_stats();
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", "64");
    } else if (mode == "rebake") {
        enable_stats();
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", "64");
        set_env("GGML_QNN_STATIC_BUDGET_MB", "2");
    } else if (mode == "slow-validate") {
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_SLOW_EXEC_MS", std::to_string(g_slow_ms).c_str());
        // the delayed validation must complete, not time out: pinned well above the delay
        set_env("GGML_QNN_TIMEOUT_MS", "10000");
        set_env("GGML_QNN_DELAY_EXECUTE", shape_key(g_hook_b, g_hook_npad, /*is_static=*/true).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE_MS", std::to_string(g_slow_validate_delay).c_str());
    } else if (mode == "slow-compute") {
        enable_stats();
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_NO_PREVALIDATE", "1");
        set_env("GGML_QNN_SLOW_EXEC_MS", std::to_string(g_slow_ms).c_str());
        set_env("GGML_QNN_TIMEOUT_MS", "10000"); // the delayed compute must complete
        set_env("GGML_QNN_DELAY_EXECUTE", shape_key(g_hook_a, g_hook_npad, /*is_static=*/true).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE_MS", std::to_string(g_slow_compute_delay).c_str());
    } else if (mode == "validate-timeout") {
        // argv[2] "cold": no fast validation before the timeout
        if (argc > 2 && strcmp(argv[2], "cold") != 0) {
            fprintf(stderr, "validate-timeout: unknown leg %s (cold)\n", argv[2]);
            return 1;
        }
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_SLOW_EXEC_MS", "0"); // A's validation counts as healthy under 2000 ms
        set_env("GGML_QNN_TIMEOUT_MS", std::to_string(g_timeout_ms).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE", shape_key(g_hook_b, g_hook_npad, /*is_static=*/true).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE_MS", std::to_string(g_timeout_delay).c_str());
    } else if (mode == "compute-timeout") {
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_NO_PREVALIDATE", "1");
        set_env("GGML_QNN_TIMEOUT_MS", std::to_string(g_timeout_ms).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE", shape_key(g_hook_a, g_hook_npad, /*is_static=*/true).c_str());
        set_env("GGML_QNN_DELAY_EXECUTE_MS", std::to_string(g_timeout_delay).c_str());
    } else if (mode == "finalize-error") {
        remove(dl_path);
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        // every static graph key carries "_s_w" (shape key, then the weight suffix), no
        // dynamic one does; the scenario lays out which match fails
        set_env("GGML_QNN_FAIL_FINALIZE", "_s_w");
        set_env("GGML_QNN_FAIL_FINALIZE_SKIP", "1");
    } else if (mode == "denylist-append") {
        enable_stats();
        set_env("GGML_QNN_DENYLIST", dl_path);
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_FAIL_FINALIZE", shape_key(g_hook_b, g_hook_npad, /*is_static=*/true).c_str());
        // binary, so the file ends in exactly the key's last byte
        FILE * f = fopen(dl_path, "wb");
        if (!f) {
            fprintf(stderr, "cannot create %s\n", dl_path);
            return 1;
        }
        fputs(g_append_seed, f);
        fclose(f);
    } else if (mode == "reuse") {
        enable_stats();
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
    } else if (mode == "dyncache") {
        enable_stats();
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_NO_STATIC_WEIGHTS", "1");
    } else if (mode == "quantized") {
        enable_stats();
        set_env("GGML_QNN_NPAD", std::to_string(g_hook_npad).c_str());
        set_env("GGML_QNN_QUANTIZED", "1");
    } else if (mode == "envparse") {
        enable_stats();
        // each malformed the way a user might: not a number, below the floor, a unit suffix
        set_env("GGML_QNN_NPAD", "abc");
        set_env("GGML_QNN_MIN_DIM", "-5");
        set_env("GGML_QNN_IO_MAX_KB", "12k");
    } else {
        fprintf(stderr, "unknown mode %s (basic|budget|denylist [noopt]|watchdog|fault|clamp [unlimited]|bigstatic|modelscale|health|"
                        "disable|mindim|rebake|elementwise|elementwise-on|loadprobe|slow-validate|slow-compute|validate-timeout [cold]|"
                        "compute-timeout|finalize-error|denylist-append|denylist-probe|reuse|dyncache|quantized|envparse)\n", mode.c_str());
        return 1;
    }

    // a GGML_BACKEND_DL build registers nothing until the backend DLLs are loaded, and every
    // entry would then skip as "no HTP". after the env setup above: loading registers the
    // backend, which latches its env
    ggml_backend_load_all();

    // the disable scenario asserts ABSENCE, so it must not be skipped by the availability check
    if (mode != "disable" && !ggml_backend_dev_by_name("QNN")) {
        // on a box known to have an HTP a missing device is a failure, not a skip: a device
        // or context that fails to create would otherwise skip the whole suite, ctest exit 0
        const char * req     = getenv("GGML_QNN_TEST_REQUIRE_HTP");
        const bool   require = req && *req && strcmp(req, "0") != 0;
        printf("QNN device not available: QnnHtp.dll not loadable, or the HTP device/context failed "
               "to create (check the ggml-qnn INFO line above) - %s\n",
               require ? "FAILING, GGML_QNN_TEST_REQUIRE_HTP is set" : "skipping");
        fflush(stdout);
        // no scenario ran, so the cleanup below is never reached: the denylist mode already
        // wrote its seed file
        if (uses_dl) {
            remove(dl_path);
        }
        // 77 is the ctest SKIP_RETURN_CODE for these entries. Returning 0 here made the
        // whole suite report green on any machine with no HTP, zero assertions executed
        return require ? 1 : 77;
    }

    // fail loudly if a variant that exists only for its ctest ENVIRONMENT lost that wiring
    if (g_require_env) {
        const char * v = getenv(g_require_env);
        if (!v || !*v) {
            fprintf(stderr, "required env %s is not set by the test runner\n", g_require_env);
        }
        GGML_ASSERT(v && *v);
    }

    int rc = 1;
    if (mode == "basic") {
        rc = scenario_basic();
    } else if (mode == "budget") {
        rc = scenario_budget();
    } else if (mode == "denylist-probe") {
        rc = scenario_denylist_probe();
    } else if (mode == "denylist") {
        rc = scenario_denylist(argc > 2);
    } else if (mode == "watchdog") {
        rc = scenario_watchdog();
    } else if (mode == "fault") {
        rc = scenario_fault();
    } else if (mode == "clamp") {
        rc = scenario_clamp(argc > 2);
    } else if (mode == "bigstatic") {
        rc = scenario_bigstatic(argc > 2 ? atoi(argv[2]) : -1);
    } else if (mode == "modelscale") {
        rc = scenario_modelscale();
    } else if (mode == "disable") {
        rc = scenario_disable();
    } else if (mode == "mindim") {
        rc = scenario_mindim();
    } else if (mode == "health") {
        rc = scenario_health();
    } else if (mode == "rebake") {
        rc = scenario_rebake();
    } else if (mode == "elementwise") {
        rc = scenario_elementwise();
    } else if (mode == "elementwise-on") {
        rc = scenario_elementwise_on();
    } else if (mode == "loadprobe") {
        rc = scenario_loadprobe();
    } else if (mode == "slow-validate") {
        rc = scenario_slow_validate();
    } else if (mode == "slow-compute") {
        rc = scenario_slow_compute();
    } else if (mode == "validate-timeout") {
        rc = scenario_validate_timeout(argc > 2);
    } else if (mode == "compute-timeout") {
        rc = scenario_compute_timeout();
    } else if (mode == "finalize-error") {
        rc = scenario_finalize_error();
    } else if (mode == "denylist-append") {
        rc = scenario_denylist_append();
    } else if (mode == "reuse") {
        rc = scenario_reuse();
    } else if (mode == "dyncache") {
        rc = scenario_dyncache();
    } else if (mode == "quantized") {
        rc = scenario_quantized();
    } else if (mode == "envparse") {
        rc = scenario_envparse();
    }

    if (uses_dl) {
        remove(dl_path);
    }
    if (g_stats_on) {
        if (rc == 0) {
            remove(g_stats_path);
        } else {
            // a FAILING (or skipped) run is exactly when these matter: io_shm_fallback is the
            // first thing to read if the intermittent -shm failure recurs. removing the file
            // here would destroy the evidence the run was instrumented to capture
            printf("counters kept at %s for diagnosis\n", g_stats_path);
        }
    }

    printf("%s: %d checks, %d failures\n", mode.c_str(), g_checks, g_failures);
    fflush(stdout);

    if (keeps_degraded) {
        hard_exit(rc);
    }
    return rc;
}
