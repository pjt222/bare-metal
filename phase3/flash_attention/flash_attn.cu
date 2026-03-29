/*
 * flash_attn.cu — Flash Attention (online softmax) without materializing N×N scores
 *
 * Standard scaled dot-product attention:
 *   O = softmax(Q K^T / sqrt(d)) V
 *
 * Naive implementation: computes the full [seq_len × seq_len] score matrix → O(N²) memory.
 * Flash Attention v1: tiles the KV sequence, maintains running max/sum statistics,
 *                     and never writes the full score matrix → O(N) memory.
 *
 * Memory comparison (seq_len=1024, d_head=64, FP32):
 *   Naive:  Q+K+V = 768 KB  +  score matrix = 4 MB  +  output = 256 KB  ≈ 5 MB
 *   Flash:  Q+K+V = 768 KB  +  output = 256 KB                          ≈ 1 MB
 *   Ratio:  5× less memory → fits L2 cache for longer → faster
 *
 * Key SASS instructions emitted:
 *   SHFL.BFLY   — warp-level reduction of partial dot products (Q · K scores)
 *   MUFU.EX2    — fast hardware exp2 for attention weights (online softmax)
 *   MUFU.RCP    — fast reciprocal for final normalization (1/sum)
 *   FMAX        — running maximum update across KV tiles
 *   FFMA        — fused multiply-add for output accumulation (weight * V)
 *
 * Online softmax recurrence (numerically stable, no stored N×N matrix):
 * -----------------------------------------------------------------------
 *   State per query row: running_max m, running_sum l, output o[0..d-1]
 *
 *   For each KV tile:
 *     1. Compute scores s[k] = scale * Q[q] · K[k]     (dot product, warp reduction)
 *     2. tile_max = max(s[k])
 *     3. new_max  = max(m, tile_max)
 *     4. Rescale:  o  *= exp(m - new_max)               (MUFU.EX2, old output shrinks)
 *                  l  *= exp(m - new_max)               (rescale normalization constant)
 *     5. Accumulate:
 *          For each k:
 *            w = exp(s[k] - new_max)                    (MUFU.EX2)
 *            l += w
 *            o += w * V[k]                              (FFMA × d/warp_size elements)
 *     6. m = new_max
 *
 *   Final:  O[q] = o / l                               (MUFU.RCP)
 *
 * This is mathematically equivalent to:
 *   O[q] = sum_k softmax(s)_k * V[k]
 * where softmax is computed over all KV positions simultaneously.
 * The rescaling in steps 4–5 ensures no numerical overflow.
 *
 * Kernels:
 *   flash_attn_1warp   — one warp (32 threads) per query position, FP32
 *                        Grid: (seq_len, 1, 1)
 *                        Block: (WARP_SIZE=32, 1, 1)
 *
 *   flash_attn_multihead — batched multi-head version
 *                          Grid: (seq_len, num_heads, batch_size)
 *                          Block: (WARP_SIZE=32, 1, 1)
 *
 * Constraints (compile-time):
 *   D_HEAD    = 64   (head dimension; WARP_SIZE must divide D_HEAD)
 *   BLOCK_KV  = 32   (KV tile size = one warp = WARP_SIZE for clean alignment)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_attn.sm_86.cubin flash_attn.cu
 *   cuobjdump -sass flash_attn.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'
 */

#define WARP_SIZE         32
#define D_HEAD            64
#define BLOCK_KV          32   // KV positions per tile; must be == WARP_SIZE for clean load pattern
#define ELEMS_PER_THREAD  (D_HEAD / WARP_SIZE)   // = 2 for D_HEAD=64

// log2(e) for exp2f-based exp: exp(x) = exp2(x * log2e)
#define LOG2E  1.4426950408889634f

// -----------------------------------------------------------------------
// Helper: warp-level SHFL.BFLY butterfly reduce — sums partial_val across all lanes.
// After this, all 32 lanes hold the same total.
// This is the pattern that emits 5 back-to-back SHFL.BFLY instructions in SASS.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float partial_val) {
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        partial_val += __shfl_xor_sync(0xFFFFFFFF, partial_val, reduction_offset);
    }
    return partial_val;
}

// -----------------------------------------------------------------------
// Kernel 1: flash_attn_1warp
//
// Single-head, FP32. One warp per query position.
// Demonstrates the online softmax recurrence clearly.
//
// Q, K, V: [seq_len × D_HEAD] row-major, FP32
// O:       [seq_len × D_HEAD] output
// scale:   1.0f / sqrtf(D_HEAD) — applied to scores
// -----------------------------------------------------------------------
extern "C" __global__ void flash_attn_1warp(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float       * __restrict__ O,
    int seq_len,
    float scale
) {
    // Shared memory: one KV tile at a time (K block + V block)
    __shared__ float K_tile[BLOCK_KV][D_HEAD];
    __shared__ float V_tile[BLOCK_KV][D_HEAD];

    int q_idx = blockIdx.x;    // which query row this warp processes
    int lane  = threadIdx.x;   // 0..31 within the warp

    if (q_idx >= seq_len) return;

    // ---- Load Q row into registers ----
    // Thread lane owns elements [lane] and [lane + WARP_SIZE] of the D_HEAD=64 vector.
    // This splits one row of Q evenly across all 32 lanes (2 floats each).
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        q_reg[elem_idx] = Q[(size_t)q_idx * D_HEAD + lane + elem_idx * WARP_SIZE];
    }

    // ---- Initialize online softmax state ----
    float running_max = -3.402823466e+38f;  // -FLT_MAX
    float running_sum = 0.0f;
    float output_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        output_reg[elem_idx] = 0.0f;
    }

    // ---- Main loop: iterate over KV tiles ----
    for (int kv_base = 0; kv_base < seq_len; kv_base += BLOCK_KV) {

        // ---- Load K tile (coalesced) ----
        // Strided loading: thread lane loads K_tile[row][col] at positions
        //   idx = lane, lane+32, lane+64, ..., lane+2016
        // This is coalesced: at each step, all 32 lanes access consecutive memory.
        #pragma unroll
        for (int load_idx = lane; load_idx < BLOCK_KV * D_HEAD; load_idx += WARP_SIZE) {
            int kv_row  = load_idx / D_HEAD;
            int d_col   = load_idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            K_tile[kv_row][d_col] = (kv_global < seq_len)
                ? K[(size_t)kv_global * D_HEAD + d_col]
                : 0.0f;
        }

        // ---- Load V tile (coalesced, same pattern) ----
        #pragma unroll
        for (int load_idx = lane; load_idx < BLOCK_KV * D_HEAD; load_idx += WARP_SIZE) {
            int kv_row  = load_idx / D_HEAD;
            int d_col   = load_idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            V_tile[kv_row][d_col] = (kv_global < seq_len)
                ? V[(size_t)kv_global * D_HEAD + d_col]
                : 0.0f;
        }
        __syncwarp();

        // ---- Compute scores for this KV tile ----
        // For each KV row kv in [0..BLOCK_KV-1], score = scale * Q[q_idx] · K[kv_base+kv]
        // score is a scalar shared by all 32 lanes after warp reduction.
        float tile_scores[BLOCK_KV];
        float tile_max = -3.402823466e+38f;

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            // Each lane computes ELEMS_PER_THREAD = 2 terms of the dot product
            float partial_dot = 0.0f;
            #pragma unroll
            for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
                partial_dot += q_reg[elem_idx] * K_tile[kv][lane + elem_idx * WARP_SIZE];
            }

            // SHFL.BFLY sum: sum partial_dot across all 32 lanes → full dot product
            // Emits 5 SHFL.BFLY instructions in SASS (offsets 16,8,4,2,1)
            float full_dot = warp_reduce_sum(partial_dot);

            tile_scores[kv] = full_dot * scale;
            tile_max = fmaxf(tile_max, tile_scores[kv]);
        }

        // ---- Online softmax update ----

        // New global maximum
        float new_max = fmaxf(running_max, tile_max);

        // Rescaling factor for previous accumulators: exp(old_max - new_max)
        // Uses MUFU.EX2 via exp2f: exp(x) = 2^(x * log2e)
        float rescale_factor = exp2f((running_max - new_max) * LOG2E);

        // Rescale running statistics
        running_sum *= rescale_factor;
        #pragma unroll
        for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
            output_reg[elem_idx] *= rescale_factor;
        }

        // Accumulate contributions from this KV tile
        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            // Attention weight: exp(score - new_max)
            // MUFU.EX2 in SASS
            float attention_weight = exp2f((tile_scores[kv] - new_max) * LOG2E);
            running_sum += attention_weight;

            // Weighted V accumulation: output += weight * V[kv]
            // FFMA × ELEMS_PER_THREAD in SASS
            #pragma unroll
            for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
                output_reg[elem_idx] += attention_weight * V_tile[kv][lane + elem_idx * WARP_SIZE];
            }
        }

        running_max = new_max;
        __syncwarp();
    }

    // ---- Final normalization: divide by running_sum ----
    // MUFU.RCP in SASS: 1/running_sum
    float rcp_sum = __frcp_rn(running_sum);

    // ---- Store output row ----
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        O[(size_t)q_idx * D_HEAD + lane + elem_idx * WARP_SIZE] = output_reg[elem_idx] * rcp_sum;
    }
}

// -----------------------------------------------------------------------
// Kernel 2: flash_attn_multihead
//
// Batched multi-head version. Identical algorithm to flash_attn_1warp
// but parameterized by (batch, head, query) via 3D grid.
//
// Q, K, V, O: [batch_size × num_heads × seq_len × D_HEAD] row-major
// Grid:  (seq_len, num_heads, batch_size)
// Block: (WARP_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void flash_attn_multihead(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float       * __restrict__ O,
    int seq_len,
    int num_heads,
    float scale
) {
    __shared__ float K_tile[BLOCK_KV][D_HEAD];
    __shared__ float V_tile[BLOCK_KV][D_HEAD];

    int q_idx    = blockIdx.x;   // query position (0..seq_len-1)
    int head_idx = blockIdx.y;   // attention head (0..num_heads-1)
    int batch_idx = blockIdx.z;  // batch item (0..batch_size-1)
    int lane     = threadIdx.x;  // 0..31

    // Head stride and batch stride
    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;

    // Base pointers for this (batch, head) pair
    size_t base_offset = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    const float *Q_head = Q + base_offset;
    const float *K_head = K + base_offset;
    const float *V_head = V + base_offset;
    float       *O_head = O + base_offset;

    // ---- Load Q row ----
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        q_reg[elem_idx] = Q_head[(size_t)q_idx * D_HEAD + lane + elem_idx * WARP_SIZE];
    }

    // ---- Initialize online softmax state ----
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float output_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        output_reg[elem_idx] = 0.0f;
    }

    // ---- KV tile loop ----
    for (int kv_base = 0; kv_base < seq_len; kv_base += BLOCK_KV) {
        #pragma unroll
        for (int load_idx = lane; load_idx < BLOCK_KV * D_HEAD; load_idx += WARP_SIZE) {
            int kv_row    = load_idx / D_HEAD;
            int d_col     = load_idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            K_tile[kv_row][d_col] = (kv_global < seq_len)
                ? K_head[(size_t)kv_global * D_HEAD + d_col] : 0.0f;
            V_tile[kv_row][d_col] = (kv_global < seq_len)
                ? V_head[(size_t)kv_global * D_HEAD + d_col] : 0.0f;
        }
        __syncwarp();

        float tile_scores[BLOCK_KV];
        float tile_max = -3.402823466e+38f;

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            float partial_dot = 0.0f;
            #pragma unroll
            for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
                partial_dot += q_reg[elem_idx] * K_tile[kv][lane + elem_idx * WARP_SIZE];
            }
            float full_dot = warp_reduce_sum(partial_dot);
            tile_scores[kv] = full_dot * scale;
            tile_max = fmaxf(tile_max, tile_scores[kv]);
        }

        float new_max        = fmaxf(running_max, tile_max);
        float rescale_factor = exp2f((running_max - new_max) * LOG2E);

        running_sum *= rescale_factor;
        #pragma unroll
        for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
            output_reg[elem_idx] *= rescale_factor;
        }

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            float attention_weight = exp2f((tile_scores[kv] - new_max) * LOG2E);
            running_sum += attention_weight;
            #pragma unroll
            for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
                output_reg[elem_idx] += attention_weight * V_tile[kv][lane + elem_idx * WARP_SIZE];
            }
        }

        running_max = new_max;
        __syncwarp();
    }

    // ---- Normalize and store ----
    float rcp_sum = __frcp_rn(running_sum);
    #pragma unroll
    for (int elem_idx = 0; elem_idx < ELEMS_PER_THREAD; elem_idx++) {
        O_head[(size_t)q_idx * D_HEAD + lane + elem_idx * WARP_SIZE] =
            output_reg[elem_idx] * rcp_sum;
    }
}
