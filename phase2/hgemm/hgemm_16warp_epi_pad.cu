/*
 * hgemm_16warp_epi_pad.cu — hgemm_16warp_epi + +8 padding (issue #98/#99).
 *
 * The original hgemm_16warp_epi.cu was forked from hgemm_16warp before
 * PAD_A/PAD_B was added there. NCU bank-conflict counter measured
 * **75.9% conflict rate** on the original variant (335M load conflicts /
 * 458M load wavefronts). This explained the dominant stalls observed
 * in Observation U: stall_mio=20.98, stall_short_sb=16.82, stall_barrier=17.32.
 *
 * Fix: add PAD_A=8 / PAD_B=8 padding (matches hgemm_16warp.cu, which
 * NCU verified at 0.17% conflict rate).
 *
 * smem cost: 16 KB → 20 KB for smem_a, 16 KB → 17 KB for smem_b.
 * Total: 48 KB → 53 KB. **Crosses 50 KB cliff** — this would force 1
 * block/SM. Need to either drop the +8 (settle for less) or use a
 * smaller pad (PAD_A=8 PAD_B=4 keeps total at 49 KB).
 *
 * Compromise tried first: PAD_A=8, PAD_B=8, total 53 KB — 1 block/SM.
 * If 1 block/SM regresses occupancy enough to lose, try smaller pad.
 */

/*
 * Original docstring follows for reference:
 *
 * hgemm_16warp_epi.cu — 16-warp 128×128 FP16 GEMM with smem epilogue
 *
 * Combines the two winners:
 *   - 16 warps (4×4 grid) for 2 blocks/SM = 32 warps/SM occupancy (+9.5%)
 *   - smem epilogue for coalesced global writes (+1.8% vs direct store)
 *
 * Shared memory: 48 KB total (under 50 KB cliff for 2 blocks/SM)
 *   smem_a: 2 × 128 × 32 × 2B = 16 KB
 *   smem_b: 2 × 32 × 128 × 2B = 16 KB
 *   epilogue: 16 × 16 × 16 × 4B = 16 KB
 *
 * Register budget: 64 regs (measured for 16-warp direct) + ~0 for epilogue loop
 *   64 × 512 × 2 = 65,536 = 64K → exactly fits 2 blocks/SM
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_epi.sm_86.cubin hgemm_16warp_epi.cu
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

// Bank-conflict-free padding
//   smem_b is the 8-way conflict (stride 128 halfs = 256 B = 2x bank period)
//   smem_a is a 4-way conflict (stride 32 halfs = 64 B = half bank period)
// Static smem max is 48 KB. Padding both costs 53 KB (over). Padding only
// smem_b costs 49 KB (fits). Address smem_b first: bigger conflict, bigger gain.
// With dynamic smem (no 48 KB static cap), can afford both pads.
// Total: 53 KB — over 50 KB cliff but the original was already at
// 48 KB. Original had 1 block/SM in practice (despite launch_bounds 2)
// because static smem maxed out. Now 53 KB, definitely 1 block/SM.
#define PAD_A    8     // smem_a stride 40 halfs = 80 B (4-way conflict gone)
#define PAD_B    8     // smem_b stride 136 halfs = 272 B (8-way conflict gone)
#define STRIDE_A (BK + PAD_A)
#define STRIDE_B (BN + PAD_B)

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 512

#define WARPS_Y 4
#define WARPS_X 4
#define WARP_TILES_M 2
#define WARP_TILES_N 2

#define CP_ASYNC_BYTES  8
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)
#define CP_ELEMS_A      (BM * BK / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 2
#define CP_ELEMS_B      (BK * BN / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 2

// NOTE: launch_bounds drops to 1 because padded smem (53 KB) crosses 50 KB cliff.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void hgemm_16warp_epi_pad(
    const __half * __restrict__ matrix_a,
    const __half * __restrict__ matrix_b,
    float        * __restrict__ matrix_c,
    int M, int N, int K
) {
    // Dynamic smem (static cap is 48 KB; padded total exceeds it).
    // Layout:
    //   smem_a: 2 * BM * STRIDE_A halfs   (16 KB unpadded, 20 KB if PAD_A=8)
    //   smem_b: 2 * BK * STRIDE_B halfs   (17 KB padded with PAD_B=8)
    //   epilogue_tile: NUM_WARPS * WMMA_M * WMMA_N floats (16 KB)
    extern __shared__ __align__(16) char smem_raw[];

    __half *smem_a_buf = (__half*)(smem_raw);
    __half *smem_b_buf = (__half*)(smem_raw + 2 * BM * STRIDE_A * sizeof(__half));
    float  *epilogue_buf = (float*)(smem_raw
                                    + 2 * BM * STRIDE_A * sizeof(__half)
                                    + 2 * BK * STRIDE_B * sizeof(__half));

    // Compatibility: present same indexing API as static decl.
    auto smem_a = (__half(*)[BM * STRIDE_A]) smem_a_buf;
    auto smem_b = (__half(*)[BK * STRIDE_B]) smem_b_buf;
    auto epilogue_tile = (float(*)[WMMA_M * WMMA_N]) epilogue_buf;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;
    int wx      = warp_id % WARPS_X;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int num_tiles = (K + BK - 1) / BK;

    #define LOAD_A_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_A; _i++) {                           \
            int _elem = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _elem / BK;                                          \
            int _col  = _elem % BK;                                          \
            int _soff = _row * STRIDE_A + _col;                              \
            int _grow = block_row + _row;                                    \
            int _gcol = (k_base) + _col;                                    \
            if (_grow < M && _gcol + ELEMS_PER_COPY - 1 < K) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_a[buf][_soff],                                      \
                    &matrix_a[_grow * K + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_a[buf][_soff + _b] = (_grow < M && _gc < K)        \
                        ? matrix_a[_grow * K + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    #define LOAD_B_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_B; _i++) {                           \
            int _elem = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _elem / BN;                                          \
            int _col  = _elem % BN;                                          \
            int _soff = _row * STRIDE_B + _col;                              \
            int _grow = (k_base) + _row;                                    \
            int _gcol = block_col + _col;                                    \
            if (_grow < K && _gcol + ELEMS_PER_COPY - 1 < N) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_b[buf][_soff],                                      \
                    &matrix_b[_grow * N + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_b[buf][_soff + _b] = (_grow < K && _gc < N)        \
                        ? matrix_b[_grow * N + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    #define COMPUTE_TILE(buf)                                                  \
        _Pragma("unroll")                                                     \
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {            \
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> a_frag[WARP_TILES_M];   \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++) {                      \
                int a_row = wy * 32 + wi * WMMA_M;                           \
                wmma::load_matrix_sync(a_frag[wi],                            \
                    &smem_a[buf][a_row * STRIDE_A + k_local], STRIDE_A);     \
            }                                                                  \
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> b_frag[WARP_TILES_N];   \
            _Pragma("unroll")                                                 \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                      \
                int b_col = wx * 32 + wj * WMMA_N;                           \
                wmma::load_matrix_sync(b_frag[wj],                            \
                    &smem_b[buf][k_local * STRIDE_B + b_col], STRIDE_B);     \
            }                                                                  \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++)                         \
                _Pragma("unroll")                                             \
                for (int wj = 0; wj < WARP_TILES_N; wj++)                    \
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj],      \
                                   acc[wi][wj]);                              \
        }

    // Prologue
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Main loop
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();
        COMPUTE_TILE(cur_buf);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // Last tile
    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf);
    }

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // Coalesced epilogue via smem
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 32 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            wmma::store_matrix_sync(
                epilogue_tile[warp_id],
                acc[wi][wj], WMMA_N, wmma::mem_row_major);
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
