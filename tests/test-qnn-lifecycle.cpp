// Lifecycle and correctness tests for the QNN (Hexagon NPU) backend, driven through the
// public ggml-backend API. Needs real HTP hardware (Windows on Snapdragon) and QnnHtp.dll
// resolvable at run time; exits 0 with a skip message anywhere else.
//
// The backend latches its env vars at first use, so each scenario must be its own process
// (the mode argument) and sets its env itself before touching the backend registry:
//
//   test-qnn-lifecycle basic      correctness of the static-weight path vs the CPU backend
//                                 (f32/f16/quantized weights, N below/at/above the pad
//                                 bucket), probe/free/reacquire cycle, clean exit
//   test-qnn-lifecycle budget     a tiny budget refuses a weight as a policy reject and does
//                                 NOT denylist it; committed bytes return on session free
//   test-qnn-lifecycle denylist   a seeded static-variant entry blocks that shape; a
//                                 dynamic-variant entry does not block the static path
//   test-qnn-lifecycle watchdog   a 1 ms timeout degrades the session; claims stop, backend
//                                 init keeps succeeding. prints WATCHDOG-CHECKS-PASSED before
//                                 returning because the leaked degraded session can crash the
//                                 process during exit (known teardown behavior) - key off the
//                                 marker, not the exit code
//   test-qnn-lifecycle bigstatic [i]   diagnostic, NOT a gate (no ctest entry): 512-class
//                                 static bakes that discriminate the padded-IO-size hang
//                                 thresholds. optional case index i runs a single case, for
//                                 order/isolation permutations
//   test-qnn-lifecycle modelscale [npad]   green guard for the WORKING side of the IO-size
//                                 law: model-scale static bake at a small pad bucket plus
//                                 bucket-boundary correctness; npad defaults to 64, "0" pins
//                                 the exact-pow2 bucket branch. run under GGML_QNN_SHARED_MEM=1
//                                 (the -shm ctest variant) it covers the rpcmem IO path too
//   test-qnn-lifecycle disable    GGML_QNN_DISABLE=1 must remove the device entirely (the
//                                 A/B kill-switch; vacuous pass on machines with no HTP)
//   test-qnn-lifecycle mindim     at DEFAULT env the min-dim gate refuses small matmuls
//                                 while claiming normal ones
//   test-qnn-lifecycle elementwise  at DEFAULT env ADD/MUL must be refused (the HTP has a
//                                 known broadcast bug), while mul_mat is still claimed
//   test-qnn-lifecycle loadprobe  supports_op must give the same verdict for a weight
//                                 probed unallocated (data == NULL, no WEIGHTS usage, as
//                                 llama does at model load) as for the resident, tagged one
//   test-qnn-lifecycle rebake     one weight probed at two N in the same pad bucket bakes
//                                 once: the budget is sized so a per-N re-bake regression
//                                 (the original NPU-memory-exhaustion failure) fails the test
//
// NOTE: run modes one at a time, and never concurrently with another NPU-using process (the
// HTP is single-client and can wedge).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

// set by modes whose ctest variant differs ONLY by an ENVIRONMENT property
static const char * g_require_env = nullptr;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void check(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        g_failures++;
    }
    printf("  %s  %s\n", ok ? "OK  " : "FAIL", what);
    fflush(stdout);
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

// dst(N,M) = src1(N,K) x src0(M,K)^T computed on one backend, weights in a buffer tagged
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS so the QNN static-bake path triggers
static bool run_mul_mat(ggml_backend_t backend, const mul_mat_case & c, std::vector<float> & out,
                        bool * claimed = nullptr) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };

    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, c.wtype, c.K, c.M);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.N);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, x);
    ggml_cgraph  * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

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

    if (claimed) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        *claimed = ggml_backend_dev_supports_op(dev, dst);
    }

    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;

    out.resize((size_t) c.N * c.M);
    if (ok) {
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return ok;
}

static ggml_backend_t qnn_backend_init(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("QNN");
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

static ggml_backend_t cpu_backend_init(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

// probe supports_op for a static-weight mul_mat without computing, weight data resident
static bool probe_claim(ggml_backend_t backend, const mul_mat_case & c) {
    std::vector<float> dummy;
    bool claimed = false;
    run_mul_mat(backend, c, dummy, &claimed);
    return claimed;
}

// probe k DISTINCT 64 KiB f16 weights (256x128, kept alive together so their addresses stay
// distinct) and return how many the backend claims - the static budget caps the count
static int count_claims(ggml_backend_t backend, int k) {
    const int64_t K = 256, M = 128, N = 64;

    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (k + 2), nullptr, true };
    ggml_init_params gp = { ggml_tensor_overhead() * (size_t) (2 * k + 4) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_context * ctx   = ggml_init(gp);

    std::vector<ggml_tensor *> ws(k), dsts(k);
    for (int i = 0; i < k; i++) {
        ws[i] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, K, M);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    for (int i = 0; i < k; i++) {
        dsts[i] = ggml_mul_mat(ctx, ws[i], x);
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> wf((size_t) K * M);
    std::vector<ggml_fp16_t> wh(wf.size());
    for (int i = 0; i < k; i++) {
        fill_uniform(wf, (unsigned) i + 1);
        ggml_fp32_to_fp16_row(wf.data(), wh.data(), (int64_t) wf.size());
        ggml_backend_tensor_set(ws[i], wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
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

static int scenario_basic(void) {
    printf("scenario: basic\n");
    ggml_backend_t qnn = qnn_backend_init();
    ggml_backend_t cpu = cpu_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    check(cpu != nullptr, "CPU backend initializes");
    if (!qnn || !cpu) {
        return 1;
    }

    // one weight shape, batch sizes below / inside / at / above the default 512 pad bucket:
    // the same weight must bake once and stay correct at every N (padded IO copy paths)
    const int64_t K = 256, M = 128;
    const int64_t Ns[] = { 1, 7, 60, 512, 513 };
    const ggml_type wtypes[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K };

    for (ggml_type wt : wtypes) {
        for (int64_t N : Ns) {
            const mul_mat_case c = { wt, K, M, N };
            bool claimed = false;
            std::vector<float> got, ref;
            const bool ok_q = run_mul_mat(qnn, c, got, &claimed);
            const bool ok_c = run_mul_mat(cpu, c, ref);

            char label[128];
            snprintf(label, sizeof(label), "mul_mat %s K=%" PRId64 " M=%" PRId64 " N=%" PRId64,
                     ggml_type_name(wt), K, M, N);
            if (!claimed) {
                // the HTP may reject a shape at trial time; that is a legal outcome, but for
                // these mainstream shapes it would be a regression worth failing on
                char msg[160];
                snprintf(msg, sizeof(msg), "%s claimed by QNN", label);
                check(false, msg);
                continue;
            }
            char msg[160];
            snprintf(msg, sizeof(msg), "%s computes", label);
            check(ok_q && ok_c, msg);
            if (ok_q && ok_c) {
                // fp16 math on the HTP: same tolerance test-backend-ops uses for MUL_MAT
                const double e = nmse(ref, got);
                snprintf(msg, sizeof(msg), "%s matches CPU (nmse %.2e)", label, e);
                check(e < 5e-4, msg);
            }
        }
    }

    // probe/free/reacquire: llama frees all backends between model probe and context creation;
    // the backend must survive the cycle and still compute
    ggml_backend_free(qnn);
    qnn = qnn_backend_init();
    check(qnn != nullptr, "backend re-initializes after free");
    if (qnn) {
        const mul_mat_case c = { GGML_TYPE_F16, K, M, 60 };
        std::vector<float> got, ref;
        const bool ok = run_mul_mat(qnn, c, got) && run_mul_mat(cpu, c, ref);
        check(ok && nmse(ref, got) < 5e-4, "compute correct after free/reacquire cycle");
        ggml_backend_free(qnn);
    }
    ggml_backend_free(cpu);
    return g_failures ? 1 : 0;
}

// GGML_QNN_STATS counters, written by the backend when the session is freed. Several ctest
// variants differ from their parent only by an env var, so without these they could not
// observe whether that env var changed anything and could not fail when it silently did not.
static const char * g_stats_path = "test-qnn-lifecycle-stats.tmp";

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

static int scenario_budget(void) {
    printf("scenario: budget\n");
    const char * dl = getenv("GGML_QNN_DENYLIST");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

    // f16 weight of 4 MiB against a 1 MiB budget: refused as policy, never denylisted
    const mul_mat_case big = { GGML_TYPE_F16, 2048, 1024, 64 };
    check(!probe_claim(qnn, big), "over-budget weight is refused");
    check(dl != nullptr, "denylist path is configured for this scenario");
    // file_size returns -1 absent and 0 empty; assert the file was never CREATED, so that
    // deleting denylist-writing entirely cannot make this check greener instead of redder
    const long dl_size = dl ? file_size(dl) : 0;
    check(dl_size < 0, "policy reject did not create the denylist file");

    // 20 distinct 64 KiB weights against the 1 MiB budget: exactly 16 fit. shapes stay in
    // the 256-class deliberately - 512x512-and-up static bakes were seen hanging at
    // validation execute on the QAIRT 2.45 runtime on battery (worked on 2.34/AC), a
    // separate open issue; this scenario tests budget accounting, not HTP shape appetite
    const int claims1 = count_claims(qnn, 20);
    char msg[96];
    snprintf(msg, sizeof(msg), "budget caps claims at 16/20 (got %d)", claims1);
    check(claims1 == 16, msg);

    // freeing the last backend frees the session and returns the committed bytes: the full
    // 16 claims only fit again if the previous session's 1 MiB actually came back
    ggml_backend_free(qnn);
    qnn = qnn_backend_init();
    check(qnn != nullptr, "backend re-initializes after free");
    if (qnn) {
        const int claims2 = count_claims(qnn, 20);
        snprintf(msg, sizeof(msg), "budget returned after session free (16/20, got %d)", claims2);
        check(claims2 == 16, msg);
        ggml_backend_free(qnn);
    }
    return g_failures ? 1 : 0;
}

static int scenario_denylist(void) {
    printf("scenario: denylist\n");
    // main() seeded the file with:
    //   MUL_MAT_f16_f32_256x128x1x1_256x512x1x1_s    (static variant of case A)
    //   MUL_MAT_f16_f32_256x64x1x1_256x512x1x1_dyn   (dynamic variant of case B)
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

    // case A: N=60 pads to the 512 bucket, so its static shape key matches the seeded entry
    const mul_mat_case a = { GGML_TYPE_F16, 256, 128, 60 };
    check(!probe_claim(qnn, a), "seeded static entry blocks the static shape");

    // case B: only the *dynamic* variant is seeded; the static path must not be blocked
    const mul_mat_case b = { GGML_TYPE_F16, 256, 64, 60 };
    check(probe_claim(qnn, b), "dynamic-variant entry does not block the static path");

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// the padded-IO size law: static-bake graphs hang at execute when a padded IO buffer crosses
// a runtime-dependent threshold (~1.5MB out on QAIRT 2.34, lower on 2.45), and work at any
// weight size below it. this mode guards the WORKING side and is safe to gate on:
// model-scale bake at a small pad bucket + correctness across bucket boundaries.
// the optional npad argument (main) feeds GGML_QNN_NPAD; "0" pins the exact-pow2 branch
static int scenario_modelscale(void) {
    printf("scenario: modelscale (GGML_QNN_NPAD=%s)\n", getenv("GGML_QNN_NPAD"));
    ggml_backend_t qnn = qnn_backend_init();
    ggml_backend_t cpu = cpu_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    check(cpu != nullptr, "CPU backend initializes");
    if (!qnn || !cpu) {
        return 1;
    }

    // guard: a model-scale weight bakes and computes below the IO-size threshold; boundary
    // cases: N at, just above, and far below the 64 bucket exercise the rounding paths
    const mul_mat_case cases[] = {
        { GGML_TYPE_F16, 512, 2560, 64 }, // model-scale guard (out 640KB at bucket 64)
        { GGML_TYPE_F16, 256,  256, 64 }, // N == bucket
        { GGML_TYPE_F16, 256,  256, 65 }, // N one past the bucket -> next pow2 bucket
        { GGML_TYPE_F16, 256,  256,  3 }, // tiny N; with NPAD=0 pins the exact-pow2 branch
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

    // the pad bucket the last case (N=3) actually landed on. this is the only thing that
    // distinguishes the -npad0 variant from its parent, which were otherwise identical runs
    const char * npad = getenv("GGML_QNN_NPAD");
    const long long pad = read_stat("pad_n_last");
    char m[224];
    const long long want_pad = (npad && strcmp(npad, "0") == 0) ? 4 : 64;
    snprintf(m, sizeof(m), "NPAD=%s put the last case on bucket %lld (pad_n_last=%lld)",
             npad ? npad : "(unset)", want_pad, pad);
    check(pad == want_pad, m);

    if (g_require_env && strcmp(g_require_env, "GGML_QNN_SHARED_MEM") == 0) {
        const long long shared   = read_stat("io_shared");
        const long long fallback = read_stat("io_shm_fallback");
        if (shared == 0 && fallback == 0) {
            // neither counter moved: fastrpc is absent, so the block was never entered and
            // there is no rpcmem path on this device to hold to account
            printf("fastrpc unavailable - rpcmem IO path not exercised on this device\n");
        } else {
            snprintf(m, sizeof(m), "rpcmem IO used for every graph, no silent host fallback "
                     "(io_shared=%lld io_shm_fallback=%lld)", shared, fallback);
            check(shared > 0 && fallback == 0, m);
        }
    } else if (g_require_env && strcmp(g_require_env, "GGML_QNN_NO_OPT") == 0) {
        const long long noopt = read_stat("graphs_noopt");
        snprintf(m, sizeof(m), "no-opt graph config reached graphCreate (graphs_noopt=%lld)", noopt);
        check(noopt > 0, m);
    }
    return g_failures ? 1 : 0;
}

// 512-class static bakes were seen hanging at validation execute on the QAIRT 2.45 runtime
// on battery while the 256-class worked. this mode is the discriminator: run it on AC - a
// clean pass isolates battery power limiting as the cause, a hang implicates the runtime.
// a failed case degrades the session, so later cases in the same process are not
// independent results; case_idx >= 0 runs one case alone for order/isolation permutations
static int scenario_bigstatic(int case_idx) {
    printf("scenario: bigstatic%s\n", case_idx >= 0 ? " (single case)" : "");
    ggml_backend_t qnn = qnn_backend_init();
    ggml_backend_t cpu = cpu_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    check(cpu != nullptr, "CPU backend initializes");
    if (!qnn || !cpu) {
        return 1;
    }

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

// the A/B kill-switch: GGML_QNN_DISABLE (set by main before any registry use) must remove
// the device entirely. on a machine with no HTP this passes vacuously - acceptable, since
// the assertion is absence
static int scenario_disable(void) {
    printf("scenario: disable\n");
    check(ggml_backend_dev_by_name("QNN") == nullptr, "GGML_QNN_DISABLE removes the QNN device");
    return g_failures ? 1 : 0;
}

// default-env gate: with GGML_QNN_MIN_DIM unset (default 32), small matmuls must be refused
// (they are memory-bound and belong on the CPU) while normal shapes are still claimed
static int scenario_mindim(void) {
    printf("scenario: mindim (default env)\n");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

    const mul_mat_case small = { GGML_TYPE_F16, 16, 16, 8 };
    check(!probe_claim(qnn, small), "16x16 matmul refused at the default min-dim gate");

    const mul_mat_case normal = { GGML_TYPE_F16, 256, 128, 64 };
    check(probe_claim(qnn, normal), "256x128 matmul still claimed at default env");

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// ADD/MUL must stay refused at default env. The HTP returns wrong results for some Add
// broadcast shapes, so the GGML_QNN_ELEMENTWISE opt-in is the only thing keeping them off
// real models: a regression that dropped the gate would corrupt output silently while every
// other test stayed green
static bool probe_binary_claim(ggml_backend_t backend, bool use_add, int64_t K, int64_t N) {
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * d = use_add ? ggml_add(ctx, a, b) : ggml_mul(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> f((size_t) K * N);
    fill_uniform(f, 3);
    ggml_backend_tensor_set(a, f.data(), 0, f.size() * sizeof(float));
    fill_uniform(f, 5);
    ggml_backend_tensor_set(b, f.data(), 0, f.size() * sizeof(float));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool claimed = ggml_backend_dev_supports_op(dev, d);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return claimed;
}

static int scenario_elementwise(void) {
    printf("scenario: elementwise (default env)\n");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

    // 2M elements clears the default GGML_QNN_MIN_ELEMENTS (1M), so a refusal here is the
    // ELEMENTWISE gate itself and not the size threshold
    const int64_t K = 2048, N = 1024;
    check(!probe_binary_claim(qnn, true,  K, N), "ADD refused with GGML_QNN_ELEMENTWISE unset");
    check(!probe_binary_claim(qnn, false, K, N), "MUL refused with GGML_QNN_ELEMENTWISE unset");

    // the matmul path must be unaffected by the elementwise gate
    const mul_mat_case normal = { GGML_TYPE_F16, 256, 128, 64 };
    check(probe_claim(qnn, normal), "mul_mat still claimed at default env");

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// llama's model loader probes supports_op for every weight BEFORE the data is resident: the
// weight sits in a no_alloc context, so data is NULL and its buffer is not tagged WEIGHTS.
// That verdict must match the one taken at schedule time on the same resident, tagged weight
// - otherwise the backend either claims a shape it will refuse later, or finalizes and
// permanently caches a dynamic-variant graph for a shape real inference never executes
static int scenario_loadprobe(void) {
    printf("scenario: loadprobe\n");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(qnn);

    const int64_t K = 256, M = 128, N = 64;
    const ggml_type wtypes[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0 };

    for (ggml_type wt : wtypes) {
        // load-time shape: weight never allocated, so data == NULL and no WEIGHTS usage tag
        ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, wt, K, M);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor * d = ggml_mul_mat(ctx, w, x);
        const bool claim_load = ggml_backend_dev_supports_op(dev, d);
        ggml_free(ctx);

        // schedule-time shape: same dims, resident data, buffer tagged WEIGHTS
        const mul_mat_case c = { wt, K, M, N };
        const bool claim_sched = probe_claim(qnn, c);

        char msg[176];
        snprintf(msg, sizeof(msg), "%s supports_op agrees load-time vs schedule-time (load=%d sched=%d)",
                 ggml_type_name(wt), (int) claim_load, (int) claim_sched);
        check(claim_load == claim_sched, msg);
    }

    ggml_backend_free(qnn);
    return g_failures ? 1 : 0;
}

// one weight probed at two batch sizes inside the same pad bucket must bake ONCE: the graph
// key is (padded shape, weight address), so both probes share a graph. the budget (2 MB) fits
// exactly one bake of the 1.125 MB weight - a per-N re-bake regression (the failure that
// originally exhausted NPU memory) makes the second probe over-budget and fails the test.
// the weight stays alive across both probes so its address cannot be reused
static int scenario_rebake(void) {
    printf("scenario: rebake\n");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

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

static int scenario_watchdog(void) {
    printf("scenario: watchdog\n");
    ggml_backend_t qnn = qnn_backend_init();
    check(qnn != nullptr, "QNN backend initializes");
    if (!qnn) {
        return 1;
    }

    // a real finalize takes ~40 ms, so a 1 ms watchdog must fire on the first trial build
    // and degrade the session
    const mul_mat_case c1 = { GGML_TYPE_F16, 256, 128, 64 };
    check(!probe_claim(qnn, c1), "first probe times out and is not claimed");

    // degraded session: every further shape is refused, fast
    const mul_mat_case c2 = { GGML_TYPE_F16, 128, 64, 64 };
    check(!probe_claim(qnn, c2), "degraded session refuses further shapes");

    // a timed-out shape is a genuine failure and must be PERSISTED so a rerun after a wedge
    // skips it. this is the only check that exercises the file-append branch
    const char * dl = getenv("GGML_QNN_DENYLIST");
    check(dl != nullptr, "denylist path is configured for this scenario");
    check(dl && file_size(dl) > 0, "timed-out shape was appended to the denylist file");

    // the backend must keep initializing (llama_new_context must not fail hard)
    ggml_backend_t qnn2 = qnn_backend_init();
    check(qnn2 != nullptr, "backend init still succeeds after degrade");
    if (qnn2) {
        ggml_backend_free(qnn2);
    }
    ggml_backend_free(qnn);

    // the leaked degraded session can crash the process during exit (known teardown
    // behavior); the runner keys off this marker, not the exit code
    printf("WATCHDOG-CHECKS-%s\n", g_failures ? "FAILED" : "PASSED");
    fflush(stdout);
    return g_failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "basic";

    // the backend latches env at first use, so all setup happens before the registry is touched
    const char * dl_path = "test-qnn-lifecycle-denylist.tmp";
    if (mode == "basic" || mode == "bigstatic") {
        set_env("GGML_QNN_MIN_DIM", "1");
    } else if (mode == "modelscale") {
        remove(g_stats_path);
        set_env("GGML_QNN_STATS", g_stats_path);
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", argc > 2 ? argv[2] : "64");
        // the -noopt and -shm variants pass the same args and differ only by a ctest
        // ENVIRONMENT property; naming it here makes broken wiring fail the test
        if (argc > 3) {
            g_require_env = argv[3];
        }
    } else if (mode == "budget") {
        remove(dl_path);
        set_env("GGML_QNN_STATIC_BUDGET_MB", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
    } else if (mode == "denylist") {
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
        FILE * f = fopen(dl_path, "w");
        if (!f) {
            fprintf(stderr, "cannot create %s\n", dl_path);
            return 1;
        }
        fprintf(f, "MUL_MAT_f16_f32_256x128x1x1_256x512x1x1_s\n");
        fprintf(f, "MUL_MAT_f16_f32_256x64x1x1_256x512x1x1_dyn\n");
        fclose(f);
    } else if (mode == "watchdog") {
        remove(dl_path);
        set_env("GGML_QNN_TIMEOUT_MS", "1");
        set_env("GGML_QNN_DENYLIST", dl_path);
    } else if (mode == "disable") {
        set_env("GGML_QNN_DISABLE", "1");
    } else if (mode == "mindim" || mode == "elementwise" || mode == "loadprobe") {
        // deliberately no env: the point is behaviour at the DEFAULT configuration
    } else if (mode == "rebake") {
        remove(g_stats_path);
        set_env("GGML_QNN_STATS", g_stats_path);
        set_env("GGML_QNN_MIN_DIM", "1");
        set_env("GGML_QNN_NPAD", "64");
        set_env("GGML_QNN_STATIC_BUDGET_MB", "2");
    } else {
        fprintf(stderr, "unknown mode %s (basic|budget|denylist|watchdog|bigstatic|modelscale|disable|mindim|rebake|elementwise|loadprobe)\n", mode.c_str());
        return 1;
    }

    // the disable scenario asserts ABSENCE, so it must not be skipped by the availability check
    if (mode != "disable" && !ggml_backend_dev_by_name("QNN")) {
        printf("QNN device not available (no HTP or QnnHtp.dll not found) - skipping\n");
        fflush(stdout);
        // 77 is the ctest SKIP_RETURN_CODE for these entries. Returning 0 here made the
        // whole suite report green on any machine with no HTP, zero assertions executed
        return 77;
    }

    // fail loudly if a variant that exists only for its ctest ENVIRONMENT lost that wiring
    if (g_require_env) {
        const char * v = getenv(g_require_env);
        char msg[128];
        snprintf(msg, sizeof(msg), "required env %s is set by the test runner", g_require_env);
        check(v && *v, msg);
    }

    int rc = 1;
    if (mode == "basic") {
        rc = scenario_basic();
    } else if (mode == "budget") {
        rc = scenario_budget();
    } else if (mode == "denylist") {
        rc = scenario_denylist();
    } else if (mode == "watchdog") {
        rc = scenario_watchdog();
    } else if (mode == "bigstatic") {
        rc = scenario_bigstatic(argc > 2 ? atoi(argv[2]) : -1);
    } else if (mode == "modelscale") {
        rc = scenario_modelscale();
    } else if (mode == "disable") {
        rc = scenario_disable();
    } else if (mode == "mindim") {
        rc = scenario_mindim();
    } else if (mode == "rebake") {
        rc = scenario_rebake();
    } else if (mode == "elementwise") {
        rc = scenario_elementwise();
    } else if (mode == "loadprobe") {
        rc = scenario_loadprobe();
    }

    if (mode == "budget" || mode == "denylist" || mode == "watchdog") {
        remove(dl_path);
    }
    if (mode == "modelscale" || mode == "rebake") {
        remove(g_stats_path);
    }

    printf("%s: %d checks, %d failures\n", mode.c_str(), g_checks, g_failures);
    fflush(stdout);
    return rc;
}
