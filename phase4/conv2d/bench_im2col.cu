/*
 * bench_im2col.cu — im2col + WMMA Conv2d vs direct conv2d
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_im2col bench_im2col.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_im2col                         # default: N=1, H=64, W=64, Cin=320, Cout=320
 *   ./bench_im2col 1 32 32 320 320         # SD UNet mid spatial resolution
 *   ./bench_im2col 1 64 64 64 64           # small channels (compute overhead visible)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// ---- WMMA GEMM tile constants (must match conv2d_im2col.cu) ----
#define GEMM_BLOCK_M   64
#define GEMM_BLOCK_N   64
#define GEMM_BLOCK_K   16
#define SMEM_A_ROW_STRIDE  (GEMM_BLOCK_K + 8)
#define SMEM_B_ROW_STRIDE  (GEMM_BLOCK_N + 8)
#define SMEM_A_ELEMENTS    (GEMM_BLOCK_M * SMEM_A_ROW_STRIDE)
#define SMEM_B_ELEMENTS    (GEMM_BLOCK_K * SMEM_B_ROW_STRIDE)
#define WMMA_M  16
#define WMMA_N  16


// -----------------------------------------------------------------------
// CPU reference: 3×3 NHWC convolution (same as bench.cu)
// -----------------------------------------------------------------------
static void cpu_conv2d_nhwc_ref(
    const float *X, const float *W, float *Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    memset(Y, 0, (size_t)N * H * W_dim * Cout * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W_dim; w++)
    for (int cout_c = 0; cout_c < Cout; cout_c++) {
        double acc = 0.0;
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++) {
            int h_in = h + kh - 1, w_in = w + kw - 1;
            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W_dim) continue;
            for (int cin = 0; cin < Cin; cin++) {
                size_t x_idx = ((size_t)n * H * W_dim + h_in * W_dim + w_in) * Cin + cin;
                size_t w_idx = (size_t)cout_c * 9 * Cin + (kh * 3 + kw) * Cin + cin;
                acc += (double)X[x_idx] * W[w_idx];
            }
        }
        size_t y_idx = ((size_t)n * H * W_dim + h * W_dim + w) * Cout + cout_c;
        Y[y_idx] = (float)acc;
    }
}


// -----------------------------------------------------------------------
// FP32 → FP16 host helper (cuda_fp16.h provides __float2half on host)
// -----------------------------------------------------------------------
static unsigned short fp32_to_fp16_bits(float val) {
    __half h = __float2half(val);
    unsigned short bits;
    memcpy(&bits, &h, 2);
    return bits;
}


// -----------------------------------------------------------------------
// Reshape weights on host:
//   W_direct [Cout, kH, kW, Cin] FP32  →  W_t [Cin*kH*kW, Cout] FP16
//   W_t[cin*kH*kW + kh*kW + kw, cout]  =  W_direct[cout, kh, kw, cin]
// -----------------------------------------------------------------------
static void reshape_weights_to_col(
    const float *W_direct,
    unsigned short *W_t,      // [K × Cout] FP16 stored as uint16
    int Cout, int Cin, int kH, int kW
) {
    int K_dim = Cin * kH * kW;
    for (int k = 0; k < K_dim; k++) {
        int cin    = k / (kH * kW);
        int k_pos  = k % (kH * kW);
        int kh_idx = k_pos / kW;
        int kw_idx = k_pos % kW;
        for (int cout_c = 0; cout_c < Cout; cout_c++) {
            float val = W_direct[(size_t)cout_c * kH * kW * Cin
                                  + (kh_idx * kW + kw_idx) * Cin + cin];
            W_t[(size_t)k * Cout + cout_c] = fp32_to_fp16_bits(val);
        }
    }
}


// -----------------------------------------------------------------------
// Fill float array with uniform random values in [-scale, +scale]
// -----------------------------------------------------------------------
static void fill_rand(float *arr, size_t n, float scale = 0.1f) {
    for (size_t i = 0; i < n; i++)
        arr[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}


// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    // ---- Parse args ----
    int N     = (argc > 1) ? atoi(argv[1]) : 1;
    int H     = (argc > 2) ? atoi(argv[2]) : 64;
    int W_dim = (argc > 3) ? atoi(argv[3]) : 64;
    int Cin   = (argc > 4) ? atoi(argv[4]) : 320;
    int Cout  = (argc > 5) ? atoi(argv[5]) : 320;
    int kH = 3, kW = 3, pad = 1;
    int out_H = H, out_W = W_dim;   // same padding, stride=1

    int M_dim = N * out_H * out_W;
    int K_dim = Cin * kH * kW;

    printf("=== im2col + WMMA Conv2d Benchmark ===\n");
    printf("N=%d H=%d W=%d Cin=%d Cout=%d  (3x3 same padding)\n",
           N, H, W_dim, Cin, Cout);
    printf("M = N*H*W = %d   K = Cin*9 = %d   N_out = Cout = %d\n",
           M_dim, K_dim, Cout);

    // Check WMMA alignment requirements
    if (M_dim % WMMA_M != 0 || K_dim % 16 != 0 || Cout % WMMA_N != 0) {
        printf("WARN: M, K, or Cout not divisible by 16 — WMMA requires alignment.\n");
        printf("      Pad M to %d, K to %d, Cout to %d\n",
               ((M_dim + 15) / 16) * 16,
               ((K_dim + 15) / 16) * 16,
               ((Cout + 15) / 16) * 16);
    }

    float col_gb    = (float)M_dim * K_dim * 2 / 1e9f;
    float flops_gfl = 2.0f * (float)M_dim * K_dim * Cout / 1e9f;
    printf("col buffer:   %.2f MB\n", col_gb * 1000.0f);
    printf("Total GFLOPs: %.3f\n\n", flops_gfl);

    // ---- Allocate host memory ----
    size_t X_elems     = (size_t)N * H * W_dim * Cin;
    size_t W_elems     = (size_t)Cout * kH * kW * Cin;
    size_t Y_elems     = (size_t)M_dim * Cout;
    size_t col_elems   = (size_t)M_dim * K_dim;
    size_t Wt_elems    = (size_t)K_dim * Cout;

    float          *host_X      = new float[X_elems];
    float          *host_W      = new float[W_elems];
    float          *host_Y_ref  = new float[Y_elems];
    float          *host_Y_gpu  = new float[Y_elems];
    unsigned short *host_Wt     = new unsigned short[Wt_elems];

    srand(42);
    fill_rand(host_X, X_elems, 0.1f);
    fill_rand(host_W, W_elems, 0.1f / sqrtf((float)(Cin * kH * kW)));

    // ---- CPU reference ----
    printf("Running CPU reference (may take a moment for large configs)...\n");
    cpu_conv2d_nhwc_ref(host_X, host_W, host_Y_ref, N, H, W_dim, Cin, Cout);

    // ---- Reshape weights on host ----
    reshape_weights_to_col(host_W, host_Wt, Cout, Cin, kH, kW);

    // ---- CUDA setup ----
    CHECK_CU(cuInit(0));
    CUdevice  cu_dev;  CHECK_CU(cuDeviceGet(&cu_dev, 0));
    CUcontext cu_ctx;  CHECK_CU(cuCtxCreate(&cu_ctx, 0, cu_dev));

    // Load cubins
    CUmodule mod_im2col, mod_gemm;
    CHECK_CU(cuModuleLoad(&mod_im2col, "conv2d_im2col.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_gemm,   "conv2d_im2col.sm_86.cubin"));   // both in same cubin

    CUfunction fn_im2col, fn_gemm;
    CHECK_CU(cuModuleGetFunction(&fn_im2col, mod_im2col, "im2col_nhwc_fp16"));
    CHECK_CU(cuModuleGetFunction(&fn_gemm,   mod_gemm,   "wmma_gemm_conv"));

    // ---- Allocate device memory ----
    CUdeviceptr d_X, d_col, d_Wt, d_Y;
    CHECK_CU(cuMemAlloc(&d_X,   X_elems   * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_col, col_elems * sizeof(unsigned short)));
    CHECK_CU(cuMemAlloc(&d_Wt,  Wt_elems  * sizeof(unsigned short)));
    CHECK_CU(cuMemAlloc(&d_Y,   Y_elems   * sizeof(float)));

    // Copy inputs
    CHECK_CU(cuMemcpyHtoD(d_X,  host_X,  X_elems  * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_Wt, host_Wt, Wt_elems * sizeof(unsigned short)));

    // ---- Set wmma_gemm_conv shared memory (smem_A + smem_B, with padding) ----
    int gemm_smem_bytes = (SMEM_A_ELEMENTS + SMEM_B_ELEMENTS) * (int)sizeof(unsigned short);
    CHECK_CU(cuFuncSetAttribute(fn_gemm,
                                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                gemm_smem_bytes));

    // ---- Correctness check: im2col → wmma_gemm_conv → compare to CPU ----
    {
        // Step 1: im2col
        int im2col_threads = 256;
        size_t total_col_elems = col_elems;
        int im2col_blocks = (int)((total_col_elems + im2col_threads - 1) / im2col_threads);
        if (im2col_blocks > 65535) im2col_blocks = 65535;

        void *im2col_args[] = {
            &d_X, &d_col,
            &N, &H, &W_dim, &Cin,
            &kH, &kW, &pad,
            &out_H, &out_W
        };
        CHECK_CU(cuLaunchKernel(fn_im2col,
                                im2col_blocks, 1, 1,
                                im2col_threads, 1, 1,
                                0, 0, im2col_args, 0));

        // Step 2: WMMA GEMM
        int grid_m = (M_dim + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
        int grid_n = (Cout  + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;

        void *gemm_args[] = {
            &d_col, &d_Wt, &d_Y,
            &M_dim, &K_dim, &Cout
        };
        CHECK_CU(cuLaunchKernel(fn_gemm,
                                grid_m, grid_n, 1,
                                128, 1, 1,
                                gemm_smem_bytes, 0, gemm_args, 0));

        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_Y_gpu, d_Y, Y_elems * sizeof(float)));

        // im2col uses FP16 intermediates → expect ~1e-2 absolute error
        CheckResult cr = check_fp32(host_Y_gpu, host_Y_ref, (int)Y_elems, 1e-1f, 1e-1f);
        bool pass = (cr.num_errors == 0);
        printf("Correctness (vs CPU ref): %s  max_abs=%.3e  max_rel=%.3e\n",
               pass ? "PASS" : "FAIL",
               (double)cr.max_abs_error, (double)cr.max_rel_error);
    }

    // ---- Benchmark: im2col phase alone ----
    printf("\n--- im2col kernel ---\n");
    {
        int im2col_threads = 256;
        int im2col_blocks  = (int)((col_elems + im2col_threads - 1) / im2col_threads);
        if (im2col_blocks > 65535) im2col_blocks = 65535;

        void *im2col_args[] = {
            &d_X, &d_col,
            &N, &H, &W_dim, &Cin,
            &kH, &kW, &pad,
            &out_H, &out_W
        };

        // Warmup
        for (int trial = 0; trial < 20; trial++)
            CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
        CHECK_CU(cuCtxSynchronize());

        BenchTimer im2col_timer;
        im2col_timer.start();
        for (int trial = 0; trial < 100; trial++)
            CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
        float im2col_ms = im2col_timer.stop_ms() / 100.0f;
        // BW: reads X_elems*4 bytes, writes col_elems*2 bytes
        float bw_gb_s = ((float)X_elems * 4 + (float)col_elems * 2) / 1e9f / (im2col_ms * 1e-3f);
        printf("  Time: %.3f ms   BW: %.1f GB/s\n", im2col_ms, bw_gb_s);
    }

    // ---- Benchmark: WMMA GEMM phase alone ----
    printf("--- wmma_gemm_conv (Tensor Core GEMM) ---\n");
    {
        int grid_m = (M_dim + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
        int grid_n = (Cout  + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;

        void *gemm_args[] = { &d_col, &d_Wt, &d_Y, &M_dim, &K_dim, &Cout };

        // Warmup
        for (int trial = 0; trial < 20; trial++)
            CHECK_CU(cuLaunchKernel(fn_gemm, grid_m, grid_n,1, 128,1,1, gemm_smem_bytes,0, gemm_args,0));
        CHECK_CU(cuCtxSynchronize());

        BenchTimer gemm_timer;
        gemm_timer.start();
        for (int trial = 0; trial < 100; trial++)
            CHECK_CU(cuLaunchKernel(fn_gemm, grid_m, grid_n,1, 128,1,1, gemm_smem_bytes,0, gemm_args,0));
        float gemm_ms = gemm_timer.stop_ms() / 100.0f;
        float gflops  = flops_gfl / (gemm_ms * 1e-3f);
        printf("  Time: %.3f ms   GFLOPS: %.0f\n", gemm_ms, gflops);
        printf("  Tensor Core utilization: %.1f%% of 174 TFLOPS FP16 peak\n",
               gflops / 174000.0f * 100.0f);
    }

    // ---- Benchmark: combined im2col + GEMM ----
    printf("--- im2col + GEMM combined ---\n");
    {
        int im2col_threads = 256;
        int im2col_blocks  = (int)((col_elems + im2col_threads - 1) / im2col_threads);
        if (im2col_blocks > 65535) im2col_blocks = 65535;
        int grid_m = (M_dim + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
        int grid_n = (Cout  + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;

        void *im2col_args[] = { &d_X, &d_col, &N, &H, &W_dim, &Cin, &kH, &kW, &pad, &out_H, &out_W };
        void *gemm_args[]   = { &d_col, &d_Wt, &d_Y, &M_dim, &K_dim, &Cout };

        for (int trial = 0; trial < 20; trial++) {
            CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
            CHECK_CU(cuLaunchKernel(fn_gemm,   grid_m, grid_n,1, 128,1,1, gemm_smem_bytes,0, gemm_args,0));
        }
        CHECK_CU(cuCtxSynchronize());

        BenchTimer combined_timer;
        combined_timer.start();
        for (int trial = 0; trial < 100; trial++) {
            CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
            CHECK_CU(cuLaunchKernel(fn_gemm,   grid_m, grid_n,1, 128,1,1, gemm_smem_bytes,0, gemm_args,0));
        }
        float combined_ms = combined_timer.stop_ms() / 100.0f;
        float gflops_eff  = flops_gfl / (combined_ms * 1e-3f);
        printf("  Time: %.3f ms   Effective GFLOPS: %.0f\n", combined_ms, gflops_eff);
        printf("  Speedup vs direct conv2d estimate (~%.0f GFLOPS at these params): check bench\n",
               flops_gfl / (25e-3f));   // ~25 ms estimated for direct at large config
    }

    // Cleanup
    cuMemFree(d_X); cuMemFree(d_col); cuMemFree(d_Wt); cuMemFree(d_Y);
    cuModuleUnload(mod_im2col);
    cuCtxDestroy(cu_ctx);

    delete[] host_X; delete[] host_W; delete[] host_Y_ref;
    delete[] host_Y_gpu; delete[] host_Wt;

    return 0;
}
