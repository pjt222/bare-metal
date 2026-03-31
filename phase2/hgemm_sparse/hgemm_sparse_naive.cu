/*
 * hgemm_sparse_naive.cu — FP16 Sparse GEMM using 2:4 Structured Sparsity (mma.sp)
 *
 * First-ever use of Ampere's sparse Tensor Cores in this project.
 * Uses PTX mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col for HMMA.SP.
 *
 * 2:4 structured sparsity: exactly 2 of every 4 elements in A along K are zero.
 * The hardware compresses A to half the size and uses 2-bit metadata per group
 * to reconstruct the logical positions during the multiply-accumulate.
 *
 * For this POC, A uses a fixed pattern: positions 0,1 in every group of 4 are
 * nonzero; positions 2,3 are zero. Metadata = 0x44444444 for all threads.
 *   Nibble 0x4 = (1 << 2) | 0 = indices {0, 1} per group.
 *
 * Fragment loading: ldmatrix from shared memory handles the hardware fragment
 * layout automatically. No manual element-to-register mapping needed.
 *
 * Fragment layout (verified empirically on sm_86 via verify_wmma_ab_layout.cu):
 *   gid = lane >> 2 (0..7), tid = lane & 3 (0..3)
 *
 *   A (sparse, row-major [16 × K/2], 2 regs):
 *     a0 = {A_comp[gid][tid*2],   A_comp[gid][tid*2+1]}     row gid
 *     a1 = {A_comp[gid+8][tid*2], A_comp[gid+8][tid*2+1]}   row gid+8
 *
 *   B (dense, row-major [K × N] in smem, col-major register packing, 2 regs per sub-tile):
 *     b0 = {B[tid*2][n],   B[tid*2+1][n]}     K rows tid*2..tid*2+1
 *     b1 = {B[tid*2+8][n], B[tid*2+9][n]}     K rows tid*2+8..tid*2+9
 *     n = gid for left sub-tile (N=0..7), gid+8 for right (N=8..15)
 *
 *   Accumulator (4 regs):
 *     d0 = C[gid][tid*2],   d1 = C[gid][tid*2+1]
 *     d2 = C[gid+8][tid*2], d3 = C[gid+8][tid*2+1]
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_naive.sm_86.cubin hgemm_sparse_naive.cu
 *   cuobjdump -sass hgemm_sparse_naive.sm_86.cubin | grep -E 'HMMA|LDGSTS|LDMATRIX'
 */

#include <cuda_fp16.h>
#include <cstdint>

#define WARP_SIZE 32

// -----------------------------------------------------------------------
// Kernel: hgemm_sparse_naive
//
// C[M×N] = A_compressed[M×K/2] × B[K×N]  (with 2:4 sparse A, FP16→FP32)
//
// One warp per 16×16 output tile. No shared-memory tiling optimization.
// K must be a multiple of 16. M, N must be multiples of 16.
//
// Grid:  (N/16, M/16, 1)
// Block: (32, 1, 1) = 1 warp
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(WARP_SIZE)
void hgemm_sparse_naive(
    const __half   * __restrict__ A_compressed,  // [M × K/2] FP16 row-major
    const __half   * __restrict__ B,             // [K × N]   FP16 row-major
    float          * __restrict__ C,             // [M × N]   FP32 row-major
    int M, int N, int K
) {
    int lane = threadIdx.x;  // 0..31

    int tile_row = blockIdx.y * 16;  // M offset
    int tile_col = blockIdx.x * 16;  // N offset

    // ---- Shared memory ----
    // A_compressed tile: [16 × 8] FP16 = 256 bytes (row-major, stride 8)
    // B tile:            [16 × 16] FP16 = 512 bytes (row-major, stride 16)
    __shared__ __align__(16) __half smem_a[16 * 8];
    __shared__ __align__(16) __half smem_b[16 * 16];

    // ---- Accumulators (2 × m16n8 sub-tiles) ----
    float d0_left = 0, d1_left = 0, d2_left = 0, d3_left = 0;   // n=0..7
    float d0_right = 0, d1_right = 0, d2_right = 0, d3_right = 0; // n=8..15

    // Fixed 2:4 metadata: positions {0,1} per group of 4
    // Nibble = (1 << 2) | 0 = 0x4. All 8 nibbles = 0x44444444.
    uint32_t meta = 0x44444444;

    int K_stored = K / 2;  // compressed K dimension

    // ---- Main K loop (16 logical elements per iteration) ----
    for (int k_base = 0; k_base < K; k_base += 16) {

        // Load compressed A tile [16 × 8] to smem (128 elements, 4 per thread)
        for (int i = lane; i < 16 * 8; i += WARP_SIZE) {
            int row = i / 8;
            int col = i % 8;
            int grow = tile_row + row;
            int gcol = (k_base / 2) + col;
            smem_a[i] = (grow < M && gcol < K_stored)
                ? A_compressed[(size_t)grow * K_stored + gcol]
                : __float2half(0.0f);
        }

        // Load dense B tile [16 × 16] to smem (256 elements, 8 per thread)
        for (int i = lane; i < 16 * 16; i += WARP_SIZE) {
            int row = i / 16;
            int col = i % 16;
            int gk = k_base + row;
            int gn = tile_col + col;
            smem_b[i] = (gk < K && gn < N)
                ? B[(size_t)gk * N + gn]
                : __float2half(0.0f);
        }
        __syncwarp();

        // ---- Fragment construction using empirically verified sm_86 layout ----
        // Verified via verify_wmma_ab_layout.cu and test_dense_manual.cu bridging test.
        //
        // Key discovery: sm_86 uses (gid, gid+8) row mapping, NOT the PTX ISA's
        // documented (2*gid, 2*gid+1). B uses col-major register packing: 2 K-rows
        // per register for a single N column (gid→N, tid→K), not the row-major
        // packing the PTX docs suggest.
        int gid = lane >> 2;
        int tid = lane & 3;

        // A fragment (sparse, row-major [16 × 8], 2 regs)
        //   a0 = {A_comp[gid][tid*2], A_comp[gid][tid*2+1]}
        //   a1 = {A_comp[gid+8][tid*2], A_comp[gid+8][tid*2+1]}
        int a_row_lo = gid;
        int a_row_hi = gid + 8;

        __half2 a0_h2 = __halves2half2(smem_a[a_row_lo * 8 + tid * 2],
                                        smem_a[a_row_lo * 8 + tid * 2 + 1]);
        __half2 a1_h2 = __halves2half2(smem_a[a_row_hi * 8 + tid * 2],
                                        smem_a[a_row_hi * 8 + tid * 2 + 1]);
        uint32_t fa0 = *(uint32_t*)&a0_h2;
        uint32_t fa1 = *(uint32_t*)&a1_h2;

        // B fragment (dense, row-major [16 × 16] in smem, col-major reg packing, 2 regs per sub-tile)
        //   b0 = {B[tid*2][n], B[tid*2+1][n]}       K rows tid*2..tid*2+1
        //   b1 = {B[tid*2+8][n], B[tid*2+9][n]}     K rows tid*2+8..tid*2+9
        //   n = gid for left sub-tile (N=0..7), gid+8 for right (N=8..15)
        int b_k0 = tid * 2;
        int b_k1 = tid * 2 + 1;
        int b_k0_hi = tid * 2 + 8;
        int b_k1_hi = tid * 2 + 9;

        // Left sub-tile (N col = gid)
        __half2 b_left0_h2 = __halves2half2(smem_b[b_k0    * 16 + gid],
                                             smem_b[b_k1    * 16 + gid]);
        __half2 b_left1_h2 = __halves2half2(smem_b[b_k0_hi * 16 + gid],
                                             smem_b[b_k1_hi * 16 + gid]);
        uint32_t fb_left0 = *(uint32_t*)&b_left0_h2;
        uint32_t fb_left1 = *(uint32_t*)&b_left1_h2;

        // Right sub-tile (N col = gid + 8)
        __half2 b_right0_h2 = __halves2half2(smem_b[b_k0    * 16 + gid + 8],
                                              smem_b[b_k1    * 16 + gid + 8]);
        __half2 b_right1_h2 = __halves2half2(smem_b[b_k0_hi * 16 + gid + 8],
                                              smem_b[b_k1_hi * 16 + gid + 8]);
        uint32_t fb_right0 = *(uint32_t*)&b_right0_h2;
        uint32_t fb_right1 = *(uint32_t*)&b_right1_h2;

        // ---- Sparse HMMA: C_left += A_sparse × B_left ----
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
            : "=f"(d0_left), "=f"(d1_left), "=f"(d2_left), "=f"(d3_left)
            : "r"(fa0), "r"(fa1),
              "r"(fb_left0), "r"(fb_left1),
              "f"(d0_left), "f"(d1_left), "f"(d2_left), "f"(d3_left),
              "r"(meta));

        // ---- Sparse HMMA: C_right += A_sparse × B_right ----
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
            : "=f"(d0_right), "=f"(d1_right), "=f"(d2_right), "=f"(d3_right)
            : "r"(fa0), "r"(fa1),
              "r"(fb_right0), "r"(fb_right1),
              "f"(d0_right), "f"(d1_right), "f"(d2_right), "f"(d3_right),
              "r"(meta));

        __syncwarp();
    }

    // ---- Store output using verified sm_86 accumulator layout ----
    // gid = lane >> 2 (0..7), tid = lane & 3 (0..3)
    // d0 = C[gid][tid*2], d1 = C[gid][tid*2+1]
    // d2 = C[gid+8][tid*2], d3 = C[gid+8][tid*2+1]
    int groupID  = lane >> 2;
    int row_lo   = groupID;
    int row_hi   = groupID + 8;
    int col0     = (lane & 3) * 2;
    int col1     = col0 + 1;

    int gr_lo = tile_row + row_lo;
    int gr_hi = tile_row + row_hi;

    // Left sub-tile (n=0..7)
    if (gr_lo < M) {
        if (tile_col + col0 < N) C[(size_t)gr_lo * N + tile_col + col0] = d0_left;
        if (tile_col + col1 < N) C[(size_t)gr_lo * N + tile_col + col1] = d1_left;
    }
    if (gr_hi < M) {
        if (tile_col + col0 < N) C[(size_t)gr_hi * N + tile_col + col0] = d2_left;
        if (tile_col + col1 < N) C[(size_t)gr_hi * N + tile_col + col1] = d3_left;
    }

    // Right sub-tile (n=8..15)
    if (gr_lo < M) {
        if (tile_col + 8 + col0 < N) C[(size_t)gr_lo * N + tile_col + 8 + col0] = d0_right;
        if (tile_col + 8 + col1 < N) C[(size_t)gr_lo * N + tile_col + 8 + col1] = d1_right;
    }
    if (gr_hi < M) {
        if (tile_col + 8 + col0 < N) C[(size_t)gr_hi * N + tile_col + 8 + col0] = d2_right;
        if (tile_col + 8 + col1 < N) C[(size_t)gr_hi * N + tile_col + 8 + col1] = d3_right;
    }
}
