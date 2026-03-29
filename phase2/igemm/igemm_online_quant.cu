/*
 * igemm_online_quant.cu — Online FP16→INT8 quantized GEMM
 *
 * Reads FP16 from DRAM → quantizes to INT8 per-tile in smem → IMMA → FP32 output.
 * INT8 data never touches DRAM. If this beats HGEMM (7,853 GFLOPS), INT8 Tensor
 * Cores become a transparent accelerator for FP16 matrix multiply.
 *
 * Per-tile quantization:
 *   For each K-tile: warp-reduce max_abs across all FP16 elements in the tile,
 *   compute scale = max_abs / 127, quantize FP16 → INT8. Since scales differ
 *   per K-tile, we maintain FP32 running accumulators and dequantize after each tile.
 *
 * Pipeline per K-tile:
 *   1. cp.async FP16 tile N+1 → next_buf (async)
 *   2. IMMA on INT8 tile N (2 K-steps within BK=32)
 *   3. Dequant: running_fp32 += INT32_acc * scale_a_N * scale_b_N; reset INT32_acc
 *   4. Wait for cp.async, sync
 *   5. Quantize FP16 tile N+1 → INT8 (max_abs reduce + convert)
 *   6. sync
 *
 * Tile: 128×128, 8 warps (4×2 grid), 1 block/SM
 * Smem: FP16 double-buffer (32 KB) + INT8 working (8 KB) + epilogue (8 KB) = 48 KB
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_online_quant.sm_86.cubin igemm_online_quant.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

#define NUM_WARPS   8
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 256 threads

#define WARPS_Y 4
#define WARPS_X 2

// Each warp covers 32×64 = 2×4 WMMA tiles
#define WARP_TILES_M 2   // 32 / 16
#define WARP_TILES_N 4   // 64 / 16

// cp.async: 4 bytes per call, FP16 = 2 bytes/elem → 2 elems per call
// A FP16 tile: BM*BK*2 bytes = 128*32*2 = 8192 bytes
// B FP16 tile: BK*BN*2 bytes = 32*128*2 = 8192 bytes
#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8192/256/4 = 8
#define CP_ELEMS_B  (BK * BN * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8192/256/4 = 8

// INT8 element counts per thread for quantization
#define QUANT_ELEMS_A  (BM * BK / BLOCK_SIZE)   // 4096/256 = 16
#define QUANT_ELEMS_B  (BK * BN / BLOCK_SIZE)   // 4096/256 = 16

// Macro: block-wide max_abs reduction + FP16→INT8 conversion
// Reads from smem_a_fp16[BUF]/smem_b_fp16[BUF], writes smem_a_int8/smem_b_int8
// Sets tile_scale = scale_a * scale_b
// Requires: tid, warp_id, lane_id, reduce_max[] in scope
#define QUANTIZE_TILE(BUF)                                                     \
do {                                                                           \
    float _max_a = 0.0f;                                                       \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_a_fp16[BUF][_idx]);                       \
        _max_a = fmaxf(_max_a, fabsf(_v));                                     \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_a = fmaxf(_max_a, __shfl_xor_sync(0xFFFFFFFF, _max_a, _off));    \
                                                                               \
    float _max_b = 0.0f;                                                       \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_b_fp16[BUF][_idx]);                       \
        _max_b = fmaxf(_max_b, fabsf(_v));                                     \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_b = fmaxf(_max_b, __shfl_xor_sync(0xFFFFFFFF, _max_b, _off));    \
                                                                               \
    if (lane_id == 0) reduce_max[warp_id] = _max_a;                           \
    __syncthreads();                                                           \
    float _bma = reduce_max[0];                                                \
    for (int _w = 1; _w < NUM_WARPS; _w++)                                     \
        _bma = fmaxf(_bma, reduce_max[_w]);                                    \
                                                                               \
    if (lane_id == 0) reduce_max[warp_id] = _max_b;                           \
    __syncthreads();                                                           \
    float _bmb = reduce_max[0];                                                \
    for (int _w = 1; _w < NUM_WARPS; _w++)                                     \
        _bmb = fmaxf(_bmb, reduce_max[_w]);                                    \
                                                                               \
    float _sa = (_bma > 0.0f) ? (_bma / 127.0f) : 1.0f;                       \
    float _sb = (_bmb > 0.0f) ? (_bmb / 127.0f) : 1.0f;                       \
    float _inv_sa = 1.0f / _sa;                                                \
    float _inv_sb = 1.0f / _sb;                                                \
                                                                               \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_a_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * _inv_sa);                                 \
        _q = max(-128, min(127, _q));                                          \
        smem_a_int8[_idx] = (signed char)_q;                                   \
    }                                                                          \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_b_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * _inv_sb);                                 \
        _q = max(-128, min(127, _q));                                          \
        smem_b_int8[_idx] = (signed char)_q;                                   \
    }                                                                          \
    __syncthreads();                                                           \
    tile_scale = _sa * _sb;                                                    \
} while(0)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_online_quant(
    const __half * __restrict__ matrix_a,   // M×K row-major (FP16)
    const __half * __restrict__ matrix_b,   // K×N row-major (FP16)
    float        * __restrict__ matrix_c,   // M×N row-major (FP32)
    int M, int N, int K
) {
    // FP16 double-buffer for cp.async
    __shared__ __half smem_a_fp16[2][BM * BK];    // 2 × 128×32 = 2 × 4096 halfs = 16 KB
    __shared__ __half smem_b_fp16[2][BK * BN];    // 2 × 32×128 = 2 × 4096 halfs = 16 KB

    // INT8 working buffers (quantized from current FP16 tile)
    __shared__ signed char smem_a_int8[BM * BK];  // 128×32 = 4 KB
    __shared__ signed char smem_b_int8[BK * BN];  // 32×128 = 4 KB

    // Epilogue: store FP32 accumulator for coalesced writes
    __shared__ float epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 × 256 × 4 = 8 KB

    // Cross-warp reduction scratch — alias into epilogue_tile (unused until final store)
    float *reduce_max = epilogue_tile[0];  // 8 floats, reuses first row

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..3
    int wx      = warp_id % WARPS_X;   // 0..1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int num_tiles = (K + BK - 1) / BK;

    // INT32 IMMA accumulators (reset per K-tile)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        acc[WARP_TILES_M][WARP_TILES_N];

    // FP32 running accumulators (persist across K-tiles)
    float running[WARP_TILES_M][WARP_TILES_N][8];  // 8 elements per fragment
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++)
                running[i][j][e] = 0.0f;

    // tile_scale is set by the QUANTIZE_TILE macro
    float tile_scale = 0.0f;

    // ====================================================================
    // Prologue: cp.async FP16 tile 0 → buf 0, wait, quantize
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_A; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int elem_idx = byte_idx / 2;   // FP16: 2 bytes per element
        int row      = elem_idx / BK;
        int col      = elem_idx % BK;
        int g_row    = block_row + row;
        int g_col    = col;

        if (g_row < M && (g_col + 1) < K) {
            __pipeline_memcpy_async(
                (char*)smem_a_fp16[0] + byte_idx,
                (const char*)(matrix_a + g_row * K + g_col),
                CP_ASYNC_SIZE);
        } else {
            // Boundary: element-wise with zero fill
            __half *dst = smem_a_fp16[0] + elem_idx;
            for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                int gc = g_col + b;
                dst[b] = (g_row < M && gc < K) ? matrix_a[g_row * K + gc] : __float2half(0.0f);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_B; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int elem_idx = byte_idx / 2;
        int row      = elem_idx / BN;
        int col      = elem_idx % BN;
        int g_row    = row;
        int g_col    = block_col + col;

        if (g_row < K && (g_col + 1) < N) {
            __pipeline_memcpy_async(
                (char*)smem_b_fp16[0] + byte_idx,
                (const char*)(matrix_b + g_row * N + g_col),
                CP_ASYNC_SIZE);
        } else {
            __half *dst = smem_b_fp16[0] + elem_idx;
            for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                int gc = g_col + b;
                dst[b] = (g_row < K && gc < N) ? matrix_b[g_row * N + gc] : __float2half(0.0f);
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Quantize tile 0
    QUANTIZE_TILE(0);

    // ====================================================================
    // Main loop: load FP16(N+1), IMMA on INT8(N), dequant, quantize(N+1)
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        // --- Zero INT32 accumulators for this K-tile ---
        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::fill_fragment(acc[i][j], 0);

        // --- Phase 1: Issue cp.async for NEXT FP16 tile ---
        #pragma unroll
        for (int i = 0; i < CP_ELEMS_A; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
            int elem_idx = byte_idx / 2;
            int row      = elem_idx / BK;
            int col      = elem_idx % BK;
            int g_row    = block_row + row;
            int g_col    = next_k_base + col;

            if (g_row < M && (g_col + 1) < K) {
                __pipeline_memcpy_async(
                    (char*)smem_a_fp16[next_buf] + byte_idx,
                    (const char*)(matrix_a + g_row * K + g_col),
                    CP_ASYNC_SIZE);
            } else {
                __half *dst = smem_a_fp16[next_buf] + elem_idx;
                for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                    int gc = g_col + b;
                    dst[b] = (g_row < M && gc < K) ? matrix_a[g_row * K + gc] : __float2half(0.0f);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < CP_ELEMS_B; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
            int elem_idx = byte_idx / 2;
            int row      = elem_idx / BN;
            int col      = elem_idx % BN;
            int g_row    = next_k_base + row;
            int g_col    = block_col + col;

            if (g_row < K && (g_col + 1) < N) {
                __pipeline_memcpy_async(
                    (char*)smem_b_fp16[next_buf] + byte_idx,
                    (const char*)(matrix_b + g_row * N + g_col),
                    CP_ASYNC_SIZE);
            } else {
                __half *dst = smem_b_fp16[next_buf] + elem_idx;
                for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                    int gc = g_col + b;
                    dst[b] = (g_row < K && gc < N) ? matrix_b[g_row * N + gc] : __float2half(0.0f);
                }
            }
        }
        __pipeline_commit();

        // --- Phase 2: IMMA on current INT8 tile ---
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a_int8 + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b_int8 + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        // --- Phase 3: Dequantize INT32 → FP32 running accumulator ---
        #pragma unroll
        for (int wi = 0; wi < WARP_TILES_M; wi++)
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    running[wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;

        // --- Phase 4: Wait for cp.async, quantize next tile ---
        __pipeline_wait_prior(0);
        __syncthreads();

        QUANTIZE_TILE(next_buf);
    }

    // ====================================================================
    // Last K-tile: IMMA + final dequant
    // ====================================================================
    {
        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::fill_fragment(acc[i][j], 0);

        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a_int8 + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b_int8 + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        // Final dequant
        #pragma unroll
        for (int wi = 0; wi < WARP_TILES_M; wi++)
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    running[wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;
    }

    // ====================================================================
    // Store FP32 running accumulators via smem epilogue
    // ====================================================================
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            // Load running values into a float accumulator fragment
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
            #pragma unroll
            for (int e = 0; e < 8; e++)
                out_frag.x[e] = running[wi][wj][e];

            // Store to smem for coalesced write
            wmma::store_matrix_sync(
                epilogue_tile[warp_id],
                out_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();

            for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += WARP_SIZE) {
                int local_row = elem / WMMA_N;
                int local_col = elem % WMMA_N;
                matrix_c[(c_row + local_row) * N + (c_col + local_col)] =
                    epilogue_tile[warp_id][elem];
            }
            __syncwarp();
        }
    }
}
