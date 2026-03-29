/*
 * softmax.cu — Row-wise softmax using MUFU.EX2 + SHFL.BFLY reductions
 *
 * Key SASS instructions to observe:
 *   SHFL.BFLY    — butterfly shuffle for warp-level parallel reduction
 *   MUFU.EX2     — fast hardware 2^x unit (used for exp via exp(x) = 2^(x*log2e))
 *   MUFU.RCP     — fast hardware reciprocal (1/sum normalization)
 *   FMAX         — float max (used in max-reduction step)
 *
 * Numerically stable softmax:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Subtracting max before exp prevents overflow (exp of large positives → Inf).
 * Since we divide by the sum, the constant shift cancels out exactly.
 *
 * MUFU.EX2 computes 2^x in hardware. To use it for natural exp:
 *   exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408...)
 *
 * Two kernels:
 *   softmax_warp  — one warp per row, row_width ≤ 32.
 *                   Pure SHFL.BFLY without shared memory — best for SASS study.
 *   softmax_block — one block per row, row_width ≤ BLOCK_SIZE (up to 1024).
 *                   Uses shared memory for inter-warp reduction.
 *                   Production kernel for typical softmax sizes.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o softmax.sm_86.cubin softmax.cu
 *   cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'
 */

#define WARP_SIZE 32

// log2(e) = 1/ln(2)
#define LOG2E 1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel 1: softmax_warp — one warp per row, row_width ≤ 32
// Demonstrates pure SHFL.BFLY reduction with no shared memory.
//
// Grid:  (num_rows, 1, 1)
// Block: (32, 1, 1)  — exactly one warp
// -----------------------------------------------------------------------
extern "C" __global__ void softmax_warp(
    const float * __restrict__ input,    // [num_rows × row_width], row-major
    float       * __restrict__ output,   // [num_rows × row_width], row-major
    int num_rows,
    int row_width
) {
    int row_index = blockIdx.x;
    int lane      = threadIdx.x;   // 0..31

    if (row_index >= num_rows) return;

    const float *row_in  = input  + (size_t)row_index * row_width;
    float       *row_out = output + (size_t)row_index * row_width;

    // --- Step 1: Load element (use -inf for out-of-bounds lanes) ---
    float element_val = (lane < row_width) ? row_in[lane] : -3.402823466e+38f;

    // --- Step 2: Warp-level max reduction via SHFL.BFLY ---
    // Each SHFL.BFLY exchanges with lane ^ offset across the warp.
    // After 5 rounds (offsets 16,8,4,2,1), every lane holds the warp maximum.
    // In SASS: SHFL.BFLY PT, Rdst, Rsrc, offset, 0x1f
    float row_max = element_val;
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        float neighbor_val = __shfl_xor_sync(0xFFFFFFFF, row_max, reduction_offset);
        row_max = fmaxf(row_max, neighbor_val);
    }
    // row_max is now identical across all lanes

    // --- Step 3: Compute exp(x - max) using MUFU.EX2 ---
    // Subtract max first (numerical stability), then scale for base-2 exp.
    // exp(x - max) = 2^((x - max) * log2(e))
    // In SASS: FMUL + MUFU.EX2
    float shifted_val = element_val - row_max;
    float exp_val = exp2f(shifted_val * LOG2E);
    if (lane >= row_width) exp_val = 0.0f;  // zero out OOB lanes for sum

    // --- Step 4: Warp-level sum reduction via SHFL.BFLY ---
    // Same butterfly pattern as max reduction, now with addition.
    float exp_sum = exp_val;
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        exp_sum += __shfl_xor_sync(0xFFFFFFFF, exp_sum, reduction_offset);
    }
    // exp_sum is now identical across all lanes

    // --- Step 5: Normalize using MUFU.RCP ---
    // Multiply by reciprocal instead of divide — maps to MUFU.RCP in SASS.
    // __frcp_rn is the C intrinsic that guarantees MUFU.RCP emission.
    float rcp_sum = __frcp_rn(exp_sum);
    float softmax_val = exp_val * rcp_sum;

    // --- Step 6: Store result ---
    if (lane < row_width) {
        row_out[lane] = softmax_val;
    }
}

// -----------------------------------------------------------------------
// Kernel 2: softmax_block — one block per row, row_width ≤ BLOCK_SIZE
// Handles wider rows by assigning multiple elements per thread.
// Uses shared memory for inter-warp reduction.
//
// Grid:  (num_rows, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)  — must be a power of 2, typically 128 or 256
//
// BLOCK_SIZE must be defined at compile time via -DBLOCK_SIZE=128 etc.
// Default: 128 threads (handles rows up to 128 * ELEMENTS_PER_THREAD wide)
// -----------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// Number of elements each thread processes — increase for very wide rows
#define ELEMENTS_PER_THREAD 4

// Shared memory for inter-warp reduction (holds one value per warp)
#define NUM_WARPS_IN_BLOCK (BLOCK_SIZE / WARP_SIZE)

extern "C" __global__ void softmax_block(
    const float * __restrict__ input,    // [num_rows × row_width], row-major
    float       * __restrict__ output,   // [num_rows × row_width], row-major
    int num_rows,
    int row_width
) {
    __shared__ float warp_scratch[NUM_WARPS_IN_BLOCK];

    int row_index   = blockIdx.x;
    int thread_id   = threadIdx.x;
    int warp_id     = thread_id / WARP_SIZE;
    int lane        = thread_id % WARP_SIZE;

    if (row_index >= num_rows) return;

    const float *row_in  = input  + (size_t)row_index * row_width;
    float       *row_out = output + (size_t)row_index * row_width;

    // --- Step 1: Load ELEMENTS_PER_THREAD elements per thread ---
    // Each thread covers indices: thread_id, thread_id + BLOCK_SIZE, ...
    float thread_vals[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int global_col = thread_id + element_offset * BLOCK_SIZE;
        thread_vals[element_offset] = (global_col < row_width)
            ? row_in[global_col]
            : -3.402823466e+38f;   // -FLT_MAX for out-of-bounds
    }

    // --- Step 2: Thread-local max across the ELEMENTS_PER_THREAD values ---
    float thread_max = thread_vals[0];
    #pragma unroll
    for (int element_offset = 1; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        thread_max = fmaxf(thread_max, thread_vals[element_offset]);
    }

    // --- Step 3: Warp-level max reduction (SHFL.BFLY) ---
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, reduction_offset));
    }
    // thread_max is now the max across all elements in this warp

    // --- Step 4: Write warp max to shared memory, then block-level max ---
    if (lane == 0) warp_scratch[warp_id] = thread_max;
    __syncthreads();

    // First warp reduces across all warp maxima
    float block_max = -3.402823466e+38f;
    if (warp_id == 0) {
        block_max = (lane < NUM_WARPS_IN_BLOCK) ? warp_scratch[lane] : -3.402823466e+38f;
        #pragma unroll
        for (int reduction_offset = NUM_WARPS_IN_BLOCK / 2; reduction_offset > 0; reduction_offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xFFFFFFFF, block_max, reduction_offset));
        }
        if (lane == 0) warp_scratch[0] = block_max;  // broadcast to all warps
    }
    __syncthreads();
    block_max = warp_scratch[0];  // all threads read the block-wide max

    // --- Step 5: Compute exp(x - max) with MUFU.EX2 ---
    float thread_exp_vals[ELEMENTS_PER_THREAD];
    float thread_exp_sum = 0.0f;
    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int global_col = thread_id + element_offset * BLOCK_SIZE;
        float shifted = thread_vals[element_offset] - block_max;
        float exp_result = exp2f(shifted * LOG2E);
        thread_exp_vals[element_offset] = (global_col < row_width) ? exp_result : 0.0f;
        thread_exp_sum += thread_exp_vals[element_offset];
    }

    // --- Step 6: Warp-level sum reduction (SHFL.BFLY) ---
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        thread_exp_sum += __shfl_xor_sync(0xFFFFFFFF, thread_exp_sum, reduction_offset);
    }

    // --- Step 7: Block-level sum reduction via shared memory ---
    if (lane == 0) warp_scratch[warp_id] = thread_exp_sum;
    __syncthreads();

    float block_exp_sum = 0.0f;
    if (warp_id == 0) {
        block_exp_sum = (lane < NUM_WARPS_IN_BLOCK) ? warp_scratch[lane] : 0.0f;
        #pragma unroll
        for (int reduction_offset = NUM_WARPS_IN_BLOCK / 2; reduction_offset > 0; reduction_offset >>= 1) {
            block_exp_sum += __shfl_xor_sync(0xFFFFFFFF, block_exp_sum, reduction_offset);
        }
        if (lane == 0) warp_scratch[0] = block_exp_sum;
    }
    __syncthreads();
    block_exp_sum = warp_scratch[0];

    // --- Step 8: Normalize with MUFU.RCP ---
    float rcp_sum = __frcp_rn(block_exp_sum);

    // --- Step 9: Write results ---
    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int global_col = thread_id + element_offset * BLOCK_SIZE;
        if (global_col < row_width) {
            row_out[global_col] = thread_exp_vals[element_offset] * rcp_sum;
        }
    }
}
