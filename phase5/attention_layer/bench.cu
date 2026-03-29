/*
 * bench.cu — End-to-end Transformer self-attention layer pipeline
 *
 * Chains bare-metal kernels into a complete forward pass:
 *
 *   X [B*S × D] FP32 (input)
 *     1. LayerNorm(X)           → X_norm  [B*S × D]   FP32
 *     2. fp32_to_fp16           → X_fp16  [B*S × D]   FP16
 *     3. Q = X_fp16 @ W_q      → Q_flat  [B*S × D]   FP32  (HGEMM)
 *     4. K = X_fp16 @ W_k      → K_flat  [B*S × D]   FP32  (HGEMM)
 *     5. V = X_fp16 @ W_v      → V_flat  [B*S × D]   FP32  (HGEMM)
 *     6. fp32_to_fp16 + transpose [B,S,H,D]→[B,H,S,D] for Q,K,V
 *     7. O = FlashAttention(Q,K,V) → [B,H,S,D] FP32
 *     8. transpose_bhsd [B,H,S,D]→[B,S,H,D] = [B*S × D] FP32
 *     9. fp32_to_fp16           → O_fp16  [B*S × D]   FP16
 *    10. Out = O_fp16 @ W_out   → Out     [B*S × D]   FP32  (HGEMM)
 *    11. Result = Out + X       → Result  [B*S × D]   FP32  (residual add)
 *
 * Cubins loaded:
 *   ../../phase2/layernorm/layernorm.sm_86.cubin  (layernorm_block)
 *   ../../phase2/hgemm/hgemm.sm_86.cubin          (hgemm_wmma)
 *   ../../phase3/flash_attention/flash_br16.sm_86.cubin (flash_attn_br16)
 *   utils.sm_86.cubin                              (fp32_to_fp16, fp16_to_fp32,
 *                                                   transpose_bshd, transpose_bhsd,
 *                                                   residual_add)
 *
 * Constraints:
 *   D_HEAD = 64 (flash_attn_br16 requirement)
 *   d_model = heads × D_HEAD (e.g., 8 × 64 = 512)
 *   seq_len must be multiple of 64 (flash Br_BLOCK)
 *   d_model must be multiple of 16 (WMMA requirement)
 *
 * Build:
 *   # Build all prerequisite cubins first (from their directories):
 *   #   phase2/layernorm: nvcc --cubin -arch=sm_86 -O2 -o layernorm.sm_86.cubin layernorm.cu
 *   #   phase2/hgemm:     nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu
 *   #   phase3/flash_attention: nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o utils.sm_86.cubin utils.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench                        # batch=1, seq=256, heads=8, d_model=512
 *   ./bench 8 1024                 # batch=8, seq=1024
 *   ./bench 1 256 8 512            # batch=1, seq=256, heads=8, d_model=512
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference: single-head attention (FP32, naive)
// For correctness validation at small sizes only.
// -----------------------------------------------------------------------
static void cpu_attention_layer(
    const float *X,           // [B*S × D]
    const float *gamma,       // [D]
    const float *beta,        // [D]
    const float *W_q,         // [D × D]
    const float *W_k,
    const float *W_v,
    const float *W_out,
    float       *result,      // [B*S × D]
    int batch, int seq, int heads, int d_head, float epsilon
) {
    int d_model = heads * d_head;
    int BS = batch * seq;
    float scale = 1.0f / sqrtf((float)d_head);

    float *x_norm = (float*)malloc(BS * d_model * sizeof(float));
    float *Q      = (float*)malloc(BS * d_model * sizeof(float));
    float *K_mat  = (float*)malloc(BS * d_model * sizeof(float));
    float *V_mat  = (float*)malloc(BS * d_model * sizeof(float));
    float *O      = (float*)malloc(BS * d_model * sizeof(float));
    float *proj   = (float*)malloc(BS * d_model * sizeof(float));
    float *scores = (float*)malloc(seq * seq * sizeof(float));

    // 1. LayerNorm
    for (int row = 0; row < BS; row++) {
        float mean = 0.0f;
        for (int d = 0; d < d_model; d++) mean += X[row * d_model + d];
        mean /= d_model;
        float var = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float diff = X[row * d_model + d] - mean;
            var += diff * diff;
        }
        var /= d_model;
        float rsqrt_var = 1.0f / sqrtf(var + epsilon);
        for (int d = 0; d < d_model; d++) {
            float norm = (X[row * d_model + d] - mean) * rsqrt_var;
            x_norm[row * d_model + d] = gamma[d] * norm + beta[d];
        }
    }

    // 2. QKV projections (simple GEMM)
    cpu_sgemm(BS, d_model, d_model, 1.0f, x_norm, d_model, W_q, d_model, 0.0f, Q, d_model);
    cpu_sgemm(BS, d_model, d_model, 1.0f, x_norm, d_model, W_k, d_model, 0.0f, K_mat, d_model);
    cpu_sgemm(BS, d_model, d_model, 1.0f, x_norm, d_model, W_v, d_model, 0.0f, V_mat, d_model);

    // 3. Attention per head (naive two-pass softmax)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            for (int qi = 0; qi < seq; qi++) {
                // Score row
                float row_max = -3.402823466e+38f;
                for (int ki = 0; ki < seq; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_head; d++) {
                        int q_idx = (b * seq + qi) * d_model + h * d_head + d;
                        int k_idx = (b * seq + ki) * d_model + h * d_head + d;
                        dot += Q[q_idx] * K_mat[k_idx];
                    }
                    scores[qi * seq + ki] = dot * scale;
                    row_max = fmaxf(row_max, scores[qi * seq + ki]);
                }
                float sum = 0.0f;
                for (int ki = 0; ki < seq; ki++) {
                    scores[qi * seq + ki] = expf(scores[qi * seq + ki] - row_max);
                    sum += scores[qi * seq + ki];
                }
                float rcp = 1.0f / sum;
                // Weighted V sum
                for (int d = 0; d < d_head; d++) {
                    float acc = 0.0f;
                    for (int ki = 0; ki < seq; ki++) {
                        int v_idx = (b * seq + ki) * d_model + h * d_head + d;
                        acc += scores[qi * seq + ki] * rcp * V_mat[v_idx];
                    }
                    int o_idx = (b * seq + qi) * d_model + h * d_head + d;
                    O[o_idx] = acc;
                }
            }
        }
    }

    // 4. Output projection
    cpu_sgemm(BS, d_model, d_model, 1.0f, O, d_model, W_out, d_model, 0.0f, proj, d_model);

    // 5. Residual add
    for (int i = 0; i < BS * d_model; i++) result[i] = proj[i] + X[i];

    free(x_norm); free(Q); free(K_mat); free(V_mat); free(O); free(proj); free(scores);
}

// -----------------------------------------------------------------------
// FP32 → FP16 host conversion
// -----------------------------------------------------------------------
static void fp32_to_fp16_host(const float *src, unsigned short *dst, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int bits;
        memcpy(&bits, &src[i], 4);
        unsigned short sign = (bits >> 31) & 0x1;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        unsigned int mant = (bits >> 13) & 0x3FF;
        if (exp <= 0) { dst[i] = sign << 15; continue; }
        if (exp >= 31) exp = 31;
        dst[i] = (unsigned short)((sign << 15) | (exp << 10) | mant);
    }
}

// -----------------------------------------------------------------------
// Helper: compute grid size for utils kernels
// -----------------------------------------------------------------------
static int utils_grid(int n) { return (n + 256 * 4 - 1) / (256 * 4); }

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int batch   = (argc > 1) ? atoi(argv[1]) : 1;
    int seq     = (argc > 2) ? atoi(argv[2]) : 256;
    int heads   = (argc > 3) ? atoi(argv[3]) : 8;
    int d_model = (argc > 4) ? atoi(argv[4]) : 512;

    int d_head = 64;   // flash_attn_br16 requirement
    if (d_model != heads * d_head) {
        fprintf(stderr, "d_model (%d) must equal heads (%d) × D_HEAD (%d) = %d\n",
                d_model, heads, d_head, heads * d_head);
        return 1;
    }
    if (seq % 64 != 0) {
        fprintf(stderr, "seq (%d) must be multiple of 64 (flash Br_BLOCK)\n", seq);
        return 1;
    }

    int BS = batch * seq;
    int D  = d_model;
    float epsilon = 1e-5f;
    float attn_scale = 1.0f / sqrtf((float)d_head);

    printf("=== Transformer Self-Attention Layer — End-to-End Pipeline ===\n");
    printf("batch=%d  seq=%d  heads=%d  d_model=%d  D_HEAD=%d\n", batch, seq, heads, D, d_head);
    printf("Tokens: %d  Parameters: %d (4 weight matrices)\n\n", BS, 4 * D * D);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);
    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // ---- Load cubins ----
    CUmodule mod_ln, mod_hgemm, mod_flash, mod_utils;
    CUfunction fn_layernorm, fn_hgemm, fn_flash, fn_f2h, fn_h2f,
               fn_tr_bshd, fn_tr_bhsd, fn_residual;

    auto load_or_die = [](CUmodule *mod, const char *path) {
        if (cuModuleLoad(mod, path) != CUDA_SUCCESS) {
            fprintf(stderr, "Cannot load %s\n", path);
            exit(1);
        }
    };

    load_or_die(&mod_ln,    "../../phase2/layernorm/layernorm.sm_86.cubin");
    load_or_die(&mod_hgemm, "../../phase2/hgemm/hgemm.sm_86.cubin");
    load_or_die(&mod_flash, "../../phase3/flash_attention/flash_br16.sm_86.cubin");
    load_or_die(&mod_utils, "utils.sm_86.cubin");

    CHECK_CU(cuModuleGetFunction(&fn_layernorm, mod_ln,    "layernorm_block"));
    CHECK_CU(cuModuleGetFunction(&fn_hgemm,     mod_hgemm, "hgemm_wmma"));
    CHECK_CU(cuModuleGetFunction(&fn_flash,     mod_flash, "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_f2h,       mod_utils, "fp32_to_fp16"));
    CHECK_CU(cuModuleGetFunction(&fn_h2f,       mod_utils, "fp16_to_fp32"));
    CHECK_CU(cuModuleGetFunction(&fn_tr_bshd,   mod_utils, "transpose_bshd"));
    CHECK_CU(cuModuleGetFunction(&fn_tr_bhsd,   mod_utils, "transpose_bhsd"));
    CHECK_CU(cuModuleGetFunction(&fn_residual,  mod_utils, "residual_add"));

    // Flash attention needs 48 KB smem
    size_t flash_smem = 2 * 64 * 64 * sizeof(short)      // K+V tiles FP16
                      + 64 * 64 * sizeof(float)           // smem_work FP32
                      + 64 * 64 * sizeof(float);          // smem_pv FP32
    CHECK_CU(cuFuncSetAttribute(fn_flash,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)flash_smem));

    printf("All cubins loaded. Flash smem: %zu bytes.\n\n", flash_smem);

    // ---- Allocate host memory ----
    size_t bsd   = (size_t)BS * D;
    size_t dd    = (size_t)D * D;

    float *h_X     = (float*)malloc(bsd * sizeof(float));
    float *h_gamma = (float*)malloc(D * sizeof(float));
    float *h_beta  = (float*)malloc(D * sizeof(float));
    float *h_Wq    = (float*)malloc(dd * sizeof(float));
    float *h_Wk    = (float*)malloc(dd * sizeof(float));
    float *h_Wv    = (float*)malloc(dd * sizeof(float));
    float *h_Wout  = (float*)malloc(dd * sizeof(float));
    float *h_result = (float*)malloc(bsd * sizeof(float));
    float *h_ref    = (float*)malloc(bsd * sizeof(float));

    fill_random(h_X,    bsd, 42);
    fill_random(h_Wq,   dd,  50);
    fill_random(h_Wk,   dd,  51);
    fill_random(h_Wv,   dd,  52);
    fill_random(h_Wout, dd,  53);
    // gamma=1, beta=0 (identity affine — simplifies correctness comparison)
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    // Convert weight matrices to FP16 on host
    unsigned short *h_Wq_fp16  = (unsigned short*)malloc(dd * sizeof(short));
    unsigned short *h_Wk_fp16  = (unsigned short*)malloc(dd * sizeof(short));
    unsigned short *h_Wv_fp16  = (unsigned short*)malloc(dd * sizeof(short));
    unsigned short *h_Wout_fp16 = (unsigned short*)malloc(dd * sizeof(short));
    fp32_to_fp16_host(h_Wq,  h_Wq_fp16,  dd);
    fp32_to_fp16_host(h_Wk,  h_Wk_fp16,  dd);
    fp32_to_fp16_host(h_Wv,  h_Wv_fp16,  dd);
    fp32_to_fp16_host(h_Wout, h_Wout_fp16, dd);

    // ---- CPU reference (small configs only) ----
    bool run_ref = (BS <= 512 && seq <= 256);
    if (run_ref) {
        printf("Computing CPU reference...\n");
        cpu_attention_layer(h_X, h_gamma, h_beta, h_Wq, h_Wk, h_Wv, h_Wout,
                            h_ref, batch, seq, heads, d_head, epsilon);
        printf("Done.\n\n");
    } else {
        printf("CPU reference skipped (too large — use batch=1 seq=256 for correctness).\n\n");
    }

    // ---- Allocate device memory ----
    // Persistent: X, weights, gamma, beta, final result
    CUdeviceptr d_X, d_gamma, d_beta;
    CUdeviceptr d_Wq, d_Wk, d_Wv, d_Wout;
    CUdeviceptr d_result;

    CHECK_CU(cuMemAlloc(&d_X,      bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_gamma,  D * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_beta,   D * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_Wq,     dd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_Wk,     dd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_Wv,     dd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_Wout,   dd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_result, bsd * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(d_X,     h_X,     bsd * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_gamma, h_gamma, D * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_beta,  h_beta,  D * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_Wq,    h_Wq_fp16,  dd * sizeof(short)));
    CHECK_CU(cuMemcpyHtoD(d_Wk,    h_Wk_fp16,  dd * sizeof(short)));
    CHECK_CU(cuMemcpyHtoD(d_Wv,    h_Wv_fp16,  dd * sizeof(short)));
    CHECK_CU(cuMemcpyHtoD(d_Wout,  h_Wout_fp16, dd * sizeof(short)));

    // Intermediate buffers
    CUdeviceptr d_xnorm;    // [BS × D] FP32  (layernorm output)
    CUdeviceptr d_xnorm_h;  // [BS × D] FP16  (converted)
    CUdeviceptr d_qflat, d_kflat, d_vflat;  // [BS × D] FP32 (HGEMM output)
    CUdeviceptr d_q_h, d_k_h, d_v_h;        // [B,H,S,D] FP16 (transposed)
    CUdeviceptr d_o_bhsd;   // [B,H,S,D] FP32  (flash output)
    CUdeviceptr d_o_flat;   // [BS × D] FP32   (transposed back)
    CUdeviceptr d_o_flat_h; // [BS × D] FP16   (converted for output proj)
    CUdeviceptr d_out;      // [BS × D] FP32   (output projection result)

    CHECK_CU(cuMemAlloc(&d_xnorm,   bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_xnorm_h, bsd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_qflat,   bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_kflat,   bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_vflat,   bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_q_h,     bsd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_k_h,     bsd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_v_h,     bsd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_o_bhsd,  bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_o_flat,  bsd * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_o_flat_h,bsd * sizeof(short)));
    CHECK_CU(cuMemAlloc(&d_out,     bsd * sizeof(float)));

    // ---- VRAM usage ----
    size_t vram_weights = 4 * dd * 2;  // 4 weight matrices FP16
    size_t vram_buffers = (5 * bsd * 4) + (4 * bsd * 2) + (3 * bsd * 4);
    printf("VRAM: weights=%.1f MB  buffers=%.1f MB  total=%.1f MB\n\n",
           vram_weights / 1e6, vram_buffers / 1e6, (vram_weights + vram_buffers) / 1e6);

    // ---- Helper: launch a utils kernel ----
    int n_bsd = (int)bsd;
    int ug = utils_grid(n_bsd);

    // HGEMM grid
    int hgemm_gx = (D + 31) / 32;
    int hgemm_gy = (BS + 31) / 32;

    // Flash grid
    int flash_gx = seq / 64;

    // ====================================================================
    // Run the pipeline
    // ====================================================================
    auto run_pipeline = [&]() {
        // 1. LayerNorm: X → x_norm (FP32)
        {
            int num_rows = BS;
            int row_width = D;
            void *args[] = { &d_X, &d_gamma, &d_beta, &d_xnorm, &num_rows, &row_width, &epsilon };
            CHECK_CU(cuLaunchKernel(fn_layernorm,
                BS, 1, 1,   128, 1, 1,   0, NULL, args, NULL));
        }

        // 2. fp32→fp16: x_norm → xnorm_h
        {
            void *args[] = { &d_xnorm, &d_xnorm_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
        }

        // 3-5. Q/K/V projections: xnorm_h @ W → flat FP32
        {
            void *args_q[] = { &d_xnorm_h, &d_Wq, &d_qflat, &BS, &D, &D };
            CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, args_q, NULL));
            void *args_k[] = { &d_xnorm_h, &d_Wk, &d_kflat, &BS, &D, &D };
            CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, args_k, NULL));
            void *args_v[] = { &d_xnorm_h, &d_Wv, &d_vflat, &BS, &D, &D };
            CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, args_v, NULL));
        }

        // 6. fp32→fp16 + transpose [B,S,H,D]→[B,H,S,D] for Q,K,V
        //    HGEMM output is [BS × D] FP32. Viewed as [B, S, H, D_HEAD].
        //    Flash attention needs [B, H, S, D_HEAD] FP16.
        //    Step a: fp32→fp16 (elementwise, layout unchanged)
        //    Step b: transpose_bshd (FP16)
        {
            // Temporary FP16 buffer (reuse d_o_flat_h as scratch)
            CUdeviceptr d_tmp_h = d_o_flat_h;  // safe to reuse, not needed until step 9

            // Q
            void *f2h_q[] = { &d_qflat, &d_tmp_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, f2h_q, NULL));
            int tr_grid = (n_bsd + 255) / 256;
            void *tr_q[] = { &d_tmp_h, &d_q_h, &batch, &seq, &heads, &d_head };
            CHECK_CU(cuLaunchKernel(fn_tr_bshd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, tr_q, NULL));

            // K
            void *f2h_k[] = { &d_kflat, &d_tmp_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, f2h_k, NULL));
            void *tr_k[] = { &d_tmp_h, &d_k_h, &batch, &seq, &heads, &d_head };
            CHECK_CU(cuLaunchKernel(fn_tr_bshd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, tr_k, NULL));

            // V
            void *f2h_v[] = { &d_vflat, &d_tmp_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, f2h_v, NULL));
            void *tr_v[] = { &d_tmp_h, &d_v_h, &batch, &seq, &heads, &d_head };
            CHECK_CU(cuLaunchKernel(fn_tr_bshd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, tr_v, NULL));
        }

        // 7. Flash Attention: Q,K,V [B,H,S,D] FP16 → O [B,H,S,D] FP32
        {
            void *args[] = { &d_q_h, &d_k_h, &d_v_h, &d_o_bhsd, &seq, &heads, &attn_scale };
            CHECK_CU(cuLaunchKernel(fn_flash,
                flash_gx, heads, batch,   128, 1, 1,
                (unsigned)flash_smem, NULL, args, NULL));
        }

        // 8. transpose_bhsd: O [B,H,S,D] FP32 → [B,S,H,D] = [BS × D] FP32
        {
            int tr_grid = (n_bsd + 255) / 256;
            void *args[] = { &d_o_bhsd, &d_o_flat, &batch, &seq, &heads, &d_head };
            CHECK_CU(cuLaunchKernel(fn_tr_bhsd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
        }

        // 9. fp32→fp16: O_flat → O_flat_h
        {
            void *args[] = { &d_o_flat, &d_o_flat_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
        }

        // 10. Output projection: O_flat_h @ W_out → Out [BS × D] FP32
        {
            void *args[] = { &d_o_flat_h, &d_Wout, &d_out, &BS, &D, &D };
            CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, args, NULL));
        }

        // 11. Residual add: result = Out + X
        {
            void *args[] = { &d_out, &d_X, &d_result, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_residual, ug, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
        }
    };

    // ---- Correctness check ----
    if (run_ref) {
        run_pipeline();
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(h_result, d_result, bsd * sizeof(float)));

        printf("Correctness (GPU pipeline vs CPU reference):\n");
        // Use loose tolerance: FP16 quantization in QKV projections + flash attention
        auto r = check_fp32(h_result, h_ref, bsd, 0.5f, 0.5f);
        print_check_result("attention_layer (end-to-end)", r);
        printf("  Note: FP16 projections + online softmax accumulation → loose tolerance\n\n");
    }

    // ---- Performance benchmark ----
    int warmup = 5, bench_n = 20;
    printf("Performance (avg of %d runs, %d warmup):\n\n", bench_n, warmup);

    // Warmup
    for (int i = 0; i < warmup; i++) run_pipeline();
    CHECK_CU(cuCtxSynchronize());

    // Total pipeline timing
    float total_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_n; i++) run_pipeline();
        total_ms = timer.stop_ms() / bench_n;
    }

    // Per-stage timing
    auto time_stage = [&](const char *label, auto stage_fn) {
        for (int i = 0; i < warmup; i++) stage_fn();
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer t;
            t.start();
            for (int i = 0; i < bench_n; i++) stage_fn();
            ms = t.stop_ms() / bench_n;
        }
        printf("  %-30s %7.3f ms  (%4.1f%%)\n", label, ms, 100.0 * ms / total_ms);
        return ms;
    };

    printf("  %-30s %7.3f ms  (100%%)\n\n", "TOTAL PIPELINE", total_ms);

    time_stage("LayerNorm", [&]() {
        int num_rows = BS, row_width = D;
        void *a[] = { &d_X, &d_gamma, &d_beta, &d_xnorm, &num_rows, &row_width, &epsilon };
        CHECK_CU(cuLaunchKernel(fn_layernorm, BS, 1, 1, 128, 1, 1, 0, NULL, a, NULL));
    });

    time_stage("fp32→fp16 (X_norm)", [&]() {
        void *a[] = { &d_xnorm, &d_xnorm_h, &n_bsd };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
    });

    time_stage("3× HGEMM (Q,K,V proj)", [&]() {
        void *aq[] = { &d_xnorm_h, &d_Wq, &d_qflat, &BS, &D, &D };
        void *ak[] = { &d_xnorm_h, &d_Wk, &d_kflat, &BS, &D, &D };
        void *av[] = { &d_xnorm_h, &d_Wv, &d_vflat, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, aq, NULL));
        CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, ak, NULL));
        CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, av, NULL));
    });

    time_stage("fp32→fp16 + transpose Q,K,V", [&]() {
        CUdeviceptr d_tmp_h = d_o_flat_h;
        int tr_grid = (n_bsd + 255) / 256;
        for (int i = 0; i < 3; i++) {
            CUdeviceptr src = (i == 0) ? d_qflat : (i == 1) ? d_kflat : d_vflat;
            CUdeviceptr dst = (i == 0) ? d_q_h   : (i == 1) ? d_k_h   : d_v_h;
            void *f[] = { &src, &d_tmp_h, &n_bsd };
            CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, f, NULL));
            void *t[] = { &d_tmp_h, &dst, &batch, &seq, &heads, &d_head };
            CHECK_CU(cuLaunchKernel(fn_tr_bshd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, t, NULL));
        }
    });

    time_stage("Flash Attention", [&]() {
        void *a[] = { &d_q_h, &d_k_h, &d_v_h, &d_o_bhsd, &seq, &heads, &attn_scale };
        CHECK_CU(cuLaunchKernel(fn_flash, flash_gx, heads, batch, 128, 1, 1,
                 (unsigned)flash_smem, NULL, a, NULL));
    });

    time_stage("transpose O + fp32→fp16", [&]() {
        int tr_grid = (n_bsd + 255) / 256;
        void *t[] = { &d_o_bhsd, &d_o_flat, &batch, &seq, &heads, &d_head };
        CHECK_CU(cuLaunchKernel(fn_tr_bhsd, tr_grid, 1, 1, 256, 1, 1, 0, NULL, t, NULL));
        void *f[] = { &d_o_flat, &d_o_flat_h, &n_bsd };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, NULL, f, NULL));
    });

    time_stage("HGEMM (output proj)", [&]() {
        void *a[] = { &d_o_flat_h, &d_Wout, &d_out, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hgemm, hgemm_gx, hgemm_gy, 1, 64, 2, 1, 0, NULL, a, NULL));
    });

    time_stage("Residual add", [&]() {
        void *a[] = { &d_out, &d_X, &d_result, &n_bsd };
        CHECK_CU(cuLaunchKernel(fn_residual, ug, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
    });

    printf("\n  Sum of stages may exceed total (per-stage warmup re-caches data).\n");
    printf("  Total pipeline time is the authoritative measurement.\n");

    // ---- Cleanup ----
    cuMemFree(d_X); cuMemFree(d_gamma); cuMemFree(d_beta);
    cuMemFree(d_Wq); cuMemFree(d_Wk); cuMemFree(d_Wv); cuMemFree(d_Wout);
    cuMemFree(d_result);
    cuMemFree(d_xnorm); cuMemFree(d_xnorm_h);
    cuMemFree(d_qflat); cuMemFree(d_kflat); cuMemFree(d_vflat);
    cuMemFree(d_q_h); cuMemFree(d_k_h); cuMemFree(d_v_h);
    cuMemFree(d_o_bhsd); cuMemFree(d_o_flat); cuMemFree(d_o_flat_h);
    cuMemFree(d_out);

    cuModuleUnload(mod_ln); cuModuleUnload(mod_hgemm);
    cuModuleUnload(mod_flash); cuModuleUnload(mod_utils);
    cuCtxDestroy(ctx);

    free(h_X); free(h_gamma); free(h_beta);
    free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wout);
    free(h_result); free(h_ref);
    free(h_Wq_fp16); free(h_Wk_fp16); free(h_Wv_fp16); free(h_Wout_fp16);

    return 0;
}
