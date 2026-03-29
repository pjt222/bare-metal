/*
 * flash_attn_wmma.cu — Flash Attention: 4 warps sharing KV tiles
 *
 * Key improvement over flash_attn_1warp (one warp = one query row):
 *   flash_attn_1warp:  each block = 1 query row, 1 warp loads K/V independently.
 *   flash_attn_4warp:  each block = 4 query rows (4 warps), all SHARE one K/V tile load.
 *                      4× fewer K/V global memory reads per KV position.
 *
 * Also doubles BLOCK_KV from 32 → 64, halving KV tile iterations.
 *
 * SASS instructions to observe:
 *   SHFL.BFLY   — warp dot product for Q·K scores (same as scalar version)
 *   MUFU.EX2    — attention weights, online softmax
 *   MUFU.RCP    — final normalization
 *   HMMA.16816  — NOT present in this kernel (WMMA PV requires Br=16 per warp;
 *                 left for Phase 3c where each warp handles 16 Q rows)
 *
 * Design:
 * -------
 *   D_HEAD    = 64   (head dimension)
 *   BLOCK_KV  = 64   (KV tile; 2× scalar version)
 *   NUM_WARPS = 4    (warps per block = query rows per block)
 *
 *   Grid:  (ceil(seq_len / NUM_WARPS), num_heads, batch_size)
 *   Block: (WARP_SIZE × NUM_WARPS, 1, 1) = (128, 1, 1)
 *
 * Shared memory per block:
 *   K_tile: [BLOCK_KV × D_HEAD] FP16 = 64×64×2 = 8 KB
 *   V_tile: [BLOCK_KV × D_HEAD] FP16 = 64×64×2 = 8 KB
 *   Total:  16 KB
 *
 * Bandwidth improvement (seq_len=1024, d=64):
 *   Scalar (1 warp/block):
 *     Q: read once per warp. K/V: read once per KV tile per warp.
 *     seq_len/BLOCK_KV = 16 iterations. 1 block per query row.
 *   4-warp shared tile:
 *     K/V tile load: amortized over 4 query rows → 4× fewer K/V loads.
 *     Expected: ~4× throughput improvement in K/V bandwidth limited regimes.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_wmma.sm_86.cubin flash_attn_wmma.cu
 *   nvcc -arch=sm_86 -O2 -o bench_wmma bench_wmma.cu -lcuda -I../../phase2/common
 *   cuobjdump -sass flash_wmma.sm_86.cubin | grep -E 'SHFL|MUFU|LDS|STS'
 */

#include <cuda_fp16.h>

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define BLOCK_KV    64

// log2(e) — for exp2f-based exp
#define LOG2E  1.4426950408889634f

// -----------------------------------------------------------------------
// Warp-level sum via SHFL.BFLY butterfly (5 instructions in SASS)
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float partial_val) {
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        partial_val += __shfl_xor_sync(0xFFFFFFFF, partial_val, reduction_offset);
    }
    return partial_val;
}

// -----------------------------------------------------------------------
// Kernel: flash_attn_4warp
//
// 4 warps per block; all 4 share one K/V tile loaded collaboratively.
// Online softmax identical to scalar version (per-warp, per-query-row).
// PV accumulation: scalar FFMA per warp (same SASS as scalar version).
//
// Inputs:
//   Q, K, V, O: [batch × heads × seq_len × D_HEAD] FP32, row-major
//               (converted FP16 on load for K/V tile; Q stays FP32 for scores)
//   seq_len, num_heads, scale: as expected
//
// Launch constraints:
//   seq_len must be multiple of NUM_WARPS (pad caller-side if needed)
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_4warp(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float       * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    // Shared memory: K_tile then V_tile, each [BLOCK_KV × D_HEAD] FP16
    __shared__ __half K_tile[BLOCK_KV * D_HEAD];
    __shared__ __half V_tile[BLOCK_KV * D_HEAD];

    int global_thread = threadIdx.x;              // 0..127
    int warp_id       = global_thread / WARP_SIZE; // 0..3
    int lane          = global_thread % WARP_SIZE; // 0..31

    // Each warp processes one query row
    int q_idx     = blockIdx.x * NUM_WARPS + warp_id;
    int head_idx  = blockIdx.y;
    int batch_idx = blockIdx.z;

    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;
    size_t base_offset  = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    const float *Q_head = Q + base_offset;
    const float *K_head = K + base_offset;
    const float *V_head = V + base_offset;
    float       *O_head = O + base_offset;

    bool valid_query = (q_idx < seq_len);

    // ---- Load Q row into registers (FP32, 2 elements per lane) ----
    float q_lo = 0.0f, q_hi = 0.0f;
    if (valid_query) {
        q_lo = Q_head[(size_t)q_idx * D_HEAD + lane];
        q_hi = Q_head[(size_t)q_idx * D_HEAD + lane + WARP_SIZE];
    }

    // ---- Initialize online softmax state ----
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float output_lo   = 0.0f;  // accumulated O[q_idx][lane]
    float output_hi   = 0.0f;  // accumulated O[q_idx][lane + WARP_SIZE]

    // ---- Main KV tile loop ----
    for (int kv_base = 0; kv_base < seq_len; kv_base += BLOCK_KV) {

        // ==============================================================
        // Phase A: All 128 threads collaboratively load K_tile + V_tile
        //
        // BLOCK_KV × D_HEAD = 64 × 64 = 4096 FP16 elements per tile.
        // 128 threads → each loads 4096/128 = 32 elements.
        // Strided pattern: thread t loads indices t, t+128, t+256, ...
        // This is coalesced: consecutive threads → consecutive columns.
        // ==============================================================
        #pragma unroll
        for (int load_idx = global_thread; load_idx < BLOCK_KV * D_HEAD;
             load_idx += NUM_WARPS * WARP_SIZE) {
            int kv_row    = load_idx / D_HEAD;
            int d_col     = load_idx % D_HEAD;
            int kv_global = kv_base + kv_row;

            K_tile[load_idx] = __float2half(
                (kv_global < seq_len) ? K_head[(size_t)kv_global * D_HEAD + d_col] : 0.0f);
            V_tile[load_idx] = __float2half(
                (kv_global < seq_len) ? V_head[(size_t)kv_global * D_HEAD + d_col] : 0.0f);
        }

        __syncthreads();  // all warps see K_tile and V_tile

        // ==============================================================
        // Phase B: Compute scores for this KV tile (per warp independently)
        //
        // score[kv] = scale * Q[q_idx] · K[kv_base + kv]
        // Lane holds q_lo = Q[lane], q_hi = Q[lane+32].
        // K_tile[kv][lane] and K_tile[kv][lane+32] from shared memory.
        // SHFL.BFLY sums partial dots across 32 lanes.
        // ==============================================================
        float tile_scores[BLOCK_KV];
        float tile_max = -3.402823466e+38f;

        if (valid_query) {
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) {
                float partial = q_lo * __half2float(K_tile[kv * D_HEAD + lane])
                              + q_hi * __half2float(K_tile[kv * D_HEAD + lane + WARP_SIZE]);
                float full_dot = warp_reduce_sum(partial);
                tile_scores[kv] = full_dot * scale;
                tile_max = fmaxf(tile_max, tile_scores[kv]);
            }
        }

        // ==============================================================
        // Phase C: Online softmax update and output accumulation (per warp)
        // ==============================================================
        float new_max        = fmaxf(running_max, tile_max);
        float rescale_factor = exp2f((running_max - new_max) * LOG2E);

        running_sum *= rescale_factor;
        output_lo   *= rescale_factor;
        output_hi   *= rescale_factor;

        if (valid_query) {
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) {
                float attention_weight = exp2f((tile_scores[kv] - new_max) * LOG2E);
                running_sum += attention_weight;
                output_lo   += attention_weight * __half2float(V_tile[kv * D_HEAD + lane]);
                output_hi   += attention_weight * __half2float(V_tile[kv * D_HEAD + lane + WARP_SIZE]);
            }
        }

        running_max = new_max;

        __syncthreads();  // all warps finished with K/V tile before next load
    }

    // ---- Final normalization and store ----
    if (valid_query) {
        float rcp_sum = __frcp_rn(running_sum);
        O_head[(size_t)q_idx * D_HEAD + lane]              = output_lo * rcp_sum;
        O_head[(size_t)q_idx * D_HEAD + lane + WARP_SIZE]  = output_hi * rcp_sum;
    }
}
