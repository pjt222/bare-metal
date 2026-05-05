/*
 * test_inplace_race.cu — Minimal reproducer for cross-warp smem race (Issue #15)
 *
 * Root cause: In-place FP16->INT8 quantization creates a Write-After-Read hazard
 * across warps. Thread T writes INT8 at byte offset `idx`, which corrupts the
 * FP16 element at `idx/2` (bytes idx&~1 to idx|1). Within a warp, SIMT lockstep
 * guarantees read-before-write. Across warps, no such guarantee exists.
 *
 * Example:
 *   Warp 1 thread 32: writes INT8 at byte 32 -> corrupts FP16[16] (bytes 32-33)
 *   Warp 0 thread 16: reads FP16[16] -> may get corrupted data
 *
 * Fix: Two-phase quantize — read ALL FP16 to registers, __syncthreads(), write ALL INT8.
 *
 * Build:  nvcc -arch=sm_86 -O2 -o test_inplace_race test_inplace_race.cu
 * Run:    ./test_inplace_race
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Same tile parameters as igemm_online_quant.cu
#define BM 128
#define BK 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * 32)               // 256 threads
#define TILE_ELEMS (BM * BK)                      // 4096 elements per tile
#define ELEMS_PER_THREAD (TILE_ELEMS / BLOCK_SIZE) // 16

// ---- Kernel 1: BUGGY — no sync between read and write phases ----
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void inplace_buggy(
    const __half * __restrict__ input,
    signed char  * __restrict__ output,
    int tile_elems
) {
    __shared__ __half smem[TILE_ELEMS];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * tile_elems;

    // Load FP16 from global to smem
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        smem[idx] = input[block_offset + idx];
    }
    __syncthreads();

    // In-place FP16->INT8: read __half then write signed char to same buffer.
    // NO barrier between reads and writes — cross-warp WAR hazard!
    signed char *smem_i8 = (signed char *)smem;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        float v = __half2float(smem[idx]);         // read FP16 at bytes 2*idx..2*idx+1
        int q = __float2int_rn(v);
        q = max(-128, min(127, q));
        smem_i8[idx] = (signed char)q;             // write INT8 at byte idx — corrupts FP16[idx/2]!
    }
    __syncthreads();

    // Copy results to global
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        output[block_offset + idx] = smem_i8[idx];
    }
}

// ---- Kernel 2: FIXED — __syncthreads() between read and write phases ----
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void inplace_fixed(
    const __half * __restrict__ input,
    signed char  * __restrict__ output,
    int tile_elems
) {
    __shared__ __half smem[TILE_ELEMS];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * tile_elems;

    // Load FP16
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        smem[idx] = input[block_offset + idx];
    }
    __syncthreads();

    // Phase 1: Read ALL FP16 values to registers
    signed char reg_i8[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        float v = __half2float(smem[idx]);
        int q = __float2int_rn(v);
        q = max(-128, min(127, q));
        reg_i8[i] = (signed char)q;
    }

    __syncthreads();  // Barrier: ALL warps finish reading before ANY warp writes

    // Phase 2: Write ALL INT8 to in-place buffer
    signed char *smem_i8 = (signed char *)smem;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        smem_i8[idx] = reg_i8[i];
    }
    __syncthreads();

    // Copy results
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        output[block_offset + idx] = smem_i8[idx];
    }
}

// ---- Host-side FP32->FP16 conversion (no device code needed) ----
static unsigned short fp32_to_fp16_bits(float v) {
    unsigned int f = *(unsigned int *)&v;
    unsigned int sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    unsigned int mant = (f >> 13) & 0x03FF;
    if (exp <= 0) return (unsigned short)sign;
    if (exp >= 31) return (unsigned short)(sign | 0x7C00);
    return (unsigned short)(sign | (exp << 10) | mant);
}

int main() {
    printf("=== In-place FP16->INT8 cross-warp race reproducer (Issue #15) ===\n\n");
    printf("Tile: %d x %d = %d elements, %d warps, %d threads\n",
           BM, BK, TILE_ELEMS, NUM_WARPS, BLOCK_SIZE);

    int num_blocks = 48;  // Saturate SMs to increase scheduling variance
    int total_elems = num_blocks * TILE_ELEMS;

    // Host FP16 input: small integers [-64, 63] so INT8 quantization is exact
    unsigned short *host_input_bits = (unsigned short *)malloc(total_elems * sizeof(unsigned short));
    signed char *expected = (signed char *)malloc(total_elems);
    for (int i = 0; i < total_elems; i++) {
        float v = (float)((i % 128) - 64);
        host_input_bits[i] = fp32_to_fp16_bits(v);
        expected[i] = (signed char)((i % 128) - 64);
    }

    __half *dev_input;
    signed char *dev_output;
    cudaMalloc(&dev_input, total_elems * sizeof(__half));
    cudaMalloc(&dev_output, total_elems);
    cudaMemcpy(dev_input, host_input_bits, total_elems * sizeof(unsigned short),
               cudaMemcpyHostToDevice);

    signed char *host_output = (signed char *)malloc(total_elems);

    int num_trials = 1000;
    int buggy_fail_trials = 0;
    int buggy_total_wrong = 0;
    int fixed_fail_trials = 0;

    for (int trial = 0; trial < num_trials; trial++) {
        // Test buggy kernel
        cudaMemset(dev_output, 0, total_elems);
        inplace_buggy<<<num_blocks, BLOCK_SIZE>>>(dev_input, dev_output, TILE_ELEMS);
        cudaDeviceSynchronize();
        cudaMemcpy(host_output, dev_output, total_elems, cudaMemcpyDeviceToHost);

        int wrong = 0;
        for (int i = 0; i < total_elems; i++) {
            if (host_output[i] != expected[i]) wrong++;
        }
        if (wrong > 0) {
            buggy_fail_trials++;
            buggy_total_wrong += wrong;
        }

        // Test fixed kernel
        cudaMemset(dev_output, 0, total_elems);
        inplace_fixed<<<num_blocks, BLOCK_SIZE>>>(dev_input, dev_output, TILE_ELEMS);
        cudaDeviceSynchronize();
        cudaMemcpy(host_output, dev_output, total_elems, cudaMemcpyDeviceToHost);

        wrong = 0;
        for (int i = 0; i < total_elems; i++) {
            if (host_output[i] != expected[i]) wrong++;
        }
        if (wrong > 0) fixed_fail_trials++;
    }

    printf("\nResults (%d trials, %d blocks/trial):\n", num_trials, num_blocks);
    printf("  Buggy (no sync):   %d/%d trials failed", buggy_fail_trials, num_trials);
    if (buggy_fail_trials > 0)
        printf("  (avg %d wrong elements/failure)", buggy_total_wrong / buggy_fail_trials);
    printf("\n");
    printf("  Fixed (with sync): %d/%d trials failed\n", fixed_fail_trials, num_trials);
    printf("\n");

    if (buggy_fail_trials > 0 && fixed_fail_trials == 0) {
        printf("ROOT CAUSE CONFIRMED: Cross-warp WAR hazard in shared memory.\n");
        printf("FIX VALIDATED: Two-phase quantize (read-all -> sync -> write-all) eliminates the race.\n");
    } else if (buggy_fail_trials == 0) {
        printf("WARNING: Race not triggered in %d trials.\n", num_trials);
        printf("The race is timing-dependent. Try increasing num_trials or num_blocks.\n");
    } else {
        printf("ERROR: Fixed kernel also fails — investigate further.\n");
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
    free(host_input_bits);
    free(expected);
    free(host_output);
    return 0;
}
