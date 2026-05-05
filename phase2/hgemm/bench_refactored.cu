/*
 * bench_refactored.cu — HGEMM benchmark using BenchDriver
 *
 * Demonstrates the shared bench_driver.h API.
 * Before: ~350 lines (context init, alloc, fill, ref, launch, check, print)
 * After:  ~90 lines (business logic only)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_refactored bench_refactored.cu -lcuda -I../common
 */

#include <cuda.h>
#include <cstdint>
#include "../common/bench_driver.h"

// Minimal FP32→FP16 host conversion
static unsigned short fp32_to_fp16(float f) {
    unsigned int b; memcpy(&b, &f, 4);
    unsigned short s = ((b >> 31) & 1) << 15;
    int e = ((b >> 23) & 0xFF) - 127 + 15;
    if (e > 0 && e < 31) s |= (e << 10) | ((b >> 13) & 0x3FF);
    return s;
}

static void convert_fp32_to_fp16(const float* src, unsigned short* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp32_to_fp16(src[i]);
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    M = (M + 15) / 16 * 16; N = (N + 15) / 16 * 16; K = (K + 15) / 16 * 16;

    printf("=== HGEMM Benchmark (BenchDriver refactor) ===\n");
    printf("C[%dx%d] = A[%dx%d] * B[%dx%d]\n\n", M, N, M, K, K, N);

    BenchDriver driver;
    driver.init_context();

    // Allocate
    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;
    auto d_A = driver.device_alloc<unsigned short>(a_elems);
    auto d_B = driver.device_alloc<unsigned short>(b_elems);
    auto d_C = driver.device_alloc<float>(c_elems);

    auto h_A = driver.host_alloc<float>(a_elems);
    auto h_B = driver.host_alloc<float>(b_elems);
    auto h_C = driver.host_alloc<float>(c_elems);
    auto h_ref = driver.host_alloc<float>(c_elems);

    fill_random(h_A.get(), a_elems, 42);
    fill_random(h_B.get(), b_elems, 99);

    auto h_Ah = driver.host_alloc<unsigned short>(a_elems);
    auto h_Bh = driver.host_alloc<unsigned short>(b_elems);
    convert_fp32_to_fp16(h_A.get(), h_Ah.get(), a_elems);
    convert_fp32_to_fp16(h_B.get(), h_Bh.get(), b_elems);

    driver.copy_h2d(d_A, h_Ah, a_elems * sizeof(unsigned short));
    driver.copy_h2d(d_B, h_Bh, b_elems * sizeof(unsigned short));

    // CPU reference (small sizes only)
    bool have_ref = (M <= 512);
    if (have_ref) {
        printf("CPU reference...\n");
        cpu_sgemm(M, N, K, 1.0f, h_A.get(), K, h_B.get(), N, 0.0f, h_ref.get(), N);
    } else {
        printf("CPU reference skipped (M>512)\n");
    }

    // Variants
    struct V { const char *name, *cubin, *sym; dim3 g, b; unsigned smem; };
    std::vector<V> variants = {
        {"hgemm_16warp", "hgemm_16warp.sm_86.cubin", "hgemm_16warp",
         dim3((N+127)/128,(M+127)/128,1), dim3(256,1,1), 48*1024},
        {"hgemm_tiled",  "hgemm_tiled.sm_86.cubin",  "hgemm_tiled",
         dim3((N+127)/128,(M+127)/128,1), dim3(256,1,1), 0},
    };

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.sym, false);
        if (!fn) { printf("  %-20s not found\n", v.name); continue; }

        void *args[] = { &d_A, &d_B, &d_C, &M, &N, &K };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, v.smem, args);
        double gflops = compute_gflops_gemm(M, N, K, ms);

        if (have_ref) {
            driver.copy_d2h(h_C, d_C, c_elems * sizeof(float));
            driver.check(h_C.get(), h_ref.get(), (int)c_elems, 1e-1f, 1e-1f, v.name);
        }
        driver.print_result(v.name, M, N, K, ms, gflops);
    }
    return 0;
}
