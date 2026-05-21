/*
 * bench_dispatch.cu — Regime dispatch validation for cross-attention
 *
 * Tests dispatch.h cross_attn_pick() across three problem sizes that cover
 * both branches and the threshold boundary.  At each size:
 *   1. Load BOTH cubin variants (baseline + v2_pad).
 *   2. Run each against the single-head CPU reference.
 *   3. Confirm the dispatched variant's smem and symbol match what
 *      cross_attn_pick() returns.
 *
 * Test matrix:
 *   CLIP-77:    seq_q=256,  seq_kv=77   → product= 19 712 → baseline
 *   threshold:  seq_q=512,  seq_kv=384  → product=196 608 → baseline (just below)
 *   typical SD: seq_q=1024, seq_kv=256  → product=262 144 → v2_pad   (well above)
 *
 * Build (from repo root):
 *   nvcc --cubin -arch=sm_86 -O2 -o kernels/attention/cross_attention/cross_attn.sm_86.cubin \
 *        kernels/attention/cross_attention/cross_attn.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o kernels/attention/cross_attention/cross_attn_v2_pad.sm_86.cubin \
 *        kernels/attention/cross_attention/cross_attn_v2_pad.cu
 *   make kernels/attention/cross_attention/bench_dispatch
 *   cd kernels/attention/cross_attention && ./bench_dispatch
 *
 * Or simply:
 *   make attention
 *   cd kernels/attention/cross_attention && ./bench_dispatch
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"
#include "dispatch.h"

/* -------------------------------------------------------------------------
 * CPU cross-attention reference (FP32, row-wise online softmax).
 * Same algorithm as bench_v2.cu — single-head, one Q row at a time.
 * -------------------------------------------------------------------------*/
static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq_q, int seq_kv, int d_head, float scale
) {
    for (int q = 0; q < seq_q; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_kv; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d_head; i++)
                dot += Q[q * d_head + i] * K[k * d_head + i];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        for (int i = 0; i < d_head; i++) O[q * d_head + i] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq_kv; k++) {
            float w = score_buf[k] * rcp;
            for (int i = 0; i < d_head; i++)
                O[q * d_head + i] += w * V[k * d_head + i];
        }
    }
}

static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

/* -------------------------------------------------------------------------
 * run_variant — load cubin, configure smem, launch, copy back, check.
 * Returns true if all elements pass the tolerance check.
 * -------------------------------------------------------------------------*/
static bool run_variant(
    BenchDriver   &driver,
    const char    *cubin_path,
    const char    *symbol,
    size_t         smem_bytes,
    __half        *dQ_ptr,
    __half        *dK_ptr,
    __half        *dV_ptr,
    float         *dO_ptr,
    float         *hOut_ptr,
    const float   *hRef_ptr,
    int            seq_q,
    int            seq_kv,
    int            num_heads,
    float          scale,
    size_t         qe
) {
    CUfunction fn = driver.load_kernel(cubin_path, symbol);
    CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                (int)smem_bytes));

    /* Zero O before launch — v2_pad has row-validity guards that skip
     * out-of-bounds Q rows; stale data from a prior launch must not persist. */
    CHECK_CU(cuMemsetD32((CUdeviceptr)dO_ptr, 0, qe));

    int grid_x = cross_attn_grid_x(seq_q);
    void *args[] = { &dQ_ptr, &dK_ptr, &dV_ptr, &dO_ptr,
                     &seq_q, &seq_kv, &num_heads, &scale };
    CHECK_CU(cuLaunchKernel(fn, grid_x, num_heads, /*batch=*/1,
                            CROSS_ATTN_BLOCK_THREADS, 1, 1,
                            (unsigned)smem_bytes, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(hOut_ptr, (CUdeviceptr)dO_ptr, qe * sizeof(float)));

    CheckResult cr = check_fp32(hOut_ptr, hRef_ptr, (int)qe, 1e-2f, 1.0f, /*print=*/true);
    return (cr.num_errors == 0);
}

/* -------------------------------------------------------------------------
 * Test configuration
 * -------------------------------------------------------------------------*/
struct TestCase {
    int         seq_q;
    int         seq_kv;
    const char *label;
    /* Expected dispatch: "baseline" or "v2_pad" */
    const char *expected_variant;
};

int main(void) {
    static const TestCase tests[] = {
        {  256,  77, "CLIP-77  (256×77  = 19712)", "baseline" },
        {  512, 384, "mid-range(512×384 =196608)", "baseline" }, /* just below 200 000 */
        { 1024, 256, "typical  (1024×256=262144)", "v2_pad"   },
    };
    static const int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    const int   d_head    = CROSS_ATTN_D_HEAD;
    const int   num_heads = 1;  /* single-head correctness check */
    const float scale     = 1.0f / sqrtf((float)d_head);

    /* Cubin info for the two variants — loaded per-test so smem attr is
     * set fresh for each configuration. */
    struct VariantInfo {
        const char *cubin_path;
        const char *symbol;
        const char *label;
    };
    static const VariantInfo variants[2] = {
        { "cross_attn.sm_86.cubin",        "cross_attn_br16",   "baseline" },
        { "cross_attn_v2_pad.sm_86.cubin", "cross_attn_v2_pad", "v2_pad"   },
    };

    printf("=== Cross-Attention Dispatch Validation ===\n");
    printf("Threshold: seq_q × seq_kv >= %zu  →  v2_pad; else baseline\n\n",
           CROSS_ATTN_DISPATCH_THRESHOLD);

    BenchDriver driver;
    driver.init_context();

    int all_pass = 1;

    for (int ti = 0; ti < num_tests; ti++) {
        const TestCase &tc = tests[ti];

        /* Pad seq_q to Br_BLOCK multiple (same as other benches). */
        int sq = tc.seq_q;
        if (sq % CROSS_ATTN_BR_BLOCK != 0)
            sq = ((sq + CROSS_ATTN_BR_BLOCK - 1) / CROSS_ATTN_BR_BLOCK)
                 * CROSS_ATTN_BR_BLOCK;

        size_t qe  = (size_t)sq       * d_head;
        size_t kve = (size_t)tc.seq_kv * d_head;

        /* Allocate + fill host buffers (FP32 for CPU reference). */
        auto hQf  = driver.host_alloc<float>(qe);
        auto hKf  = driver.host_alloc<float>(kve);
        auto hVf  = driver.host_alloc<float>(kve);
        auto hRef = driver.host_alloc<float>(qe);
        auto hOut = driver.host_alloc<float>(qe);
        auto sBuf = driver.host_alloc<float>((size_t)tc.seq_kv);

        fill_random(hQf.get(), (int)qe,  (unsigned)(10 + ti));
        fill_random(hKf.get(), (int)kve, (unsigned)(11 + ti));
        fill_random(hVf.get(), (int)kve, (unsigned)(12 + ti));

        cpu_cross_attn(hQf.get(), hKf.get(), hVf.get(), hRef.get(),
                       sBuf.get(), sq, tc.seq_kv, d_head, scale);

        /* Convert to FP16 for GPU. */
        auto hQh = driver.host_alloc<__half>(qe);
        auto hKh = driver.host_alloc<__half>(kve);
        auto hVh = driver.host_alloc<__half>(kve);
        fp32_to_fp16(hQf.get(), hQh.get(), qe);
        fp32_to_fp16(hKf.get(), hKh.get(), kve);
        fp32_to_fp16(hVf.get(), hVh.get(), kve);

        /* Device buffers. */
        auto dQ = driver.device_alloc<__half>(qe);
        auto dK = driver.device_alloc<__half>(kve);
        auto dV = driver.device_alloc<__half>(kve);
        auto dO = driver.device_alloc<float>(qe);

        driver.copy_h2d(dQ, hQh, qe * sizeof(__half));
        driver.copy_h2d(dK, hKh, kve * sizeof(__half));
        driver.copy_h2d(dV, hVh, kve * sizeof(__half));

        /* Dispatch query. */
        CrossAttnVariant picked = cross_attn_pick(sq, tc.seq_kv);
        size_t           product = (size_t)sq * (size_t)tc.seq_kv;
        bool             pick_is_v2_pad = (product >= CROSS_ATTN_DISPATCH_THRESHOLD);
        const char      *pick_label = pick_is_v2_pad ? "v2_pad" : "baseline";

        printf("--- %s ---\n", tc.label);
        printf("  product=%zu  dispatch→ %s  (expected: %s)  %s\n",
               product, pick_label, tc.expected_variant,
               (strcmp(pick_label, tc.expected_variant) == 0) ? "✓" : "MISMATCH");

        if (strcmp(pick_label, tc.expected_variant) != 0) all_pass = 0;

        /* Run BOTH variants and check each against CPU reference. */
        for (int vi = 0; vi < 2; vi++) {
            const VariantInfo &var = variants[vi];

            /* Determine the smem for this variant using dispatch.h constants. */
            size_t smem_bytes = (vi == 0)
                ? CROSS_ATTN_SMEM_BASELINE
                : CROSS_ATTN_SMEM_V2_PAD;

            bool ok = run_variant(
                driver,
                var.cubin_path, var.symbol, smem_bytes,
                dQ.ptr, dK.ptr, dV.ptr, dO.ptr,
                hOut.ptr, hRef.ptr,
                sq, tc.seq_kv, num_heads, scale, qe);

            printf("  %s vs CPU: %s\n", var.label, ok ? "✓ PASS" : "✗ FAIL");
            if (!ok) all_pass = 0;
        }

        /* Confirm dispatch picks the right smem and symbol. */
        bool smem_ok     = (pick_is_v2_pad)
            ? (picked.smem_bytes == CROSS_ATTN_SMEM_V2_PAD)
            : (picked.smem_bytes == CROSS_ATTN_SMEM_BASELINE);
        bool symbol_ok   = (pick_is_v2_pad)
            ? (strcmp(picked.symbol, "cross_attn_v2_pad") == 0)
            : (strcmp(picked.symbol, "cross_attn_br16") == 0);
        printf("  dispatch smem=%zu (%s)  symbol=%s (%s)\n",
               picked.smem_bytes,
               smem_ok   ? "✓" : "MISMATCH",
               picked.symbol,
               symbol_ok ? "✓" : "MISMATCH");
        if (!smem_ok || !symbol_ok) all_pass = 0;

        printf("\n");
    }

    printf("=== Summary: %s ===\n", all_pass ? "ALL PASS ✓" : "FAILURES DETECTED ✗");
    return all_pass ? 0 : 1;
}
