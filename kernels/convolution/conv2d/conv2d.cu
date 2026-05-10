/*
 * conv2d.cu — 3×3 Direct Convolution for Diffusion Model UNet Blocks
 *
 * Implements NHWC-layout direct convolution — the workhorse of every ResNet
 * and UNet block in Stable Diffusion.
 *
 * Formula:
 *   Y[n, h, w, c_out] = bias[c_out] +
 *       sum_{kh=0}^{2} sum_{kw=0}^{2} sum_{c_in=0}^{Cin-1}
 *           W[c_out, kh, kw, c_in] * X[n, h+kh-1, w+kw-1, c_in]
 *
 *   Padding: 1 (same), Stride: 1. Out-of-bounds → zero (zero-padding).
 *
 * Key SASS instructions:
 *   FFMA       — fused multiply-add for the inner loop accumulation
 *   LDG.E.128  — 128-bit vectorized global loads (4 floats at once)
 *   LDSL       — load from shared memory (weight tile)
 *
 * Two kernels:
 *   conv2d_nhwc      — direct 3×3 conv, each thread one (n, h, w, c_out)
 *   conv2d_1x1_nhwc  — 1×1 conv (no spatial loop; maps directly to GEMM row)
 *
 * Kernel design (conv2d_nhwc):
 *   Grid:  (N, ceil(H*W / TILE_HW), ceil(Cout / TILE_C))
 *   Block: (TILE_HW, TILE_C, 1)  — total 256 threads
 *
 *   Each thread computes one output element by iterating over:
 *   - 3×3 = 9 spatial offsets  (compile-time unrolled)
 *   - Cin input channels       (loop over tiles of CIN_TILE=16)
 *   Weights cached in shared memory per CIN_TILE block.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o conv2d.sm_86.cubin conv2d.cu
 *   cuobjdump -sass conv2d.sm_86.cubin | grep -E 'FFMA|LDG' | head -20
 *   → dense FFMA sequence (9 × CIN_TILE per output element)
 */

#define TILE_HW    16   // output spatial elements per block (threadIdx.x)
#define TILE_C      8   // output channels per block        (threadIdx.y)
#define CIN_TILE   16   // input channel tile (shared memory weight cache)
#define BLOCK_THREADS (TILE_HW * TILE_C)   // 128 threads per block

// -----------------------------------------------------------------------
// Kernel: conv2d_nhwc
//
// NHWC direct convolution — stride=1, padding=1 (same), kernel 3×3.
//
// Parameters:
//   X:     [N, H, W, Cin]  FP32 input
//   W:     [Cout, 3, 3, Cin]  FP32 weights (OIHW stored as [Cout, kh, kw, Cin])
//   bias:  [Cout] FP32 bias (optional; pass NULL for no bias)
//   Y:     [N, H, W, Cout] FP32 output
//   N, H, W, Cin, Cout: tensor dimensions
//
// Grid:  (N, ceil(H*W / TILE_HW), ceil(Cout / TILE_C))
// Block: (TILE_HW, TILE_C, 1)
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(BLOCK_THREADS)
void conv2d_nhwc(
    const float * __restrict__ X,
    const float * __restrict__ W,
    const float * __restrict__ bias,
    float       * __restrict__ Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    // Shared memory: cache [9, CIN_TILE, TILE_C] weight block
    // = 9 * 16 * 8 = 1152 floats = 4.5 KB
    __shared__ float smem_W[9][CIN_TILE][TILE_C];

    int sample_n   = blockIdx.x;
    int hw_base    = blockIdx.y * TILE_HW;
    int cout_base  = blockIdx.z * TILE_C;

    int hw_local   = threadIdx.x;   // local spatial index within tile
    int cout_local = threadIdx.y;   // local output channel within tile

    int hw_idx   = hw_base + hw_local;      // global (h*W + w) index
    int c_out    = cout_base + cout_local;   // global output channel

    int h_out = hw_idx / W_dim;
    int w_out = hw_idx % W_dim;

    float accumulator = 0.0f;
    if (bias != NULL && c_out < Cout) {
        accumulator = bias[c_out];
    }

    // ---- Loop over input channel tiles ----
    for (int cin_tile_start = 0; cin_tile_start < Cin; cin_tile_start += CIN_TILE) {
        int cin_tile_end = min(cin_tile_start + CIN_TILE, Cin);
        int cin_tile_len = cin_tile_end - cin_tile_start;

        // ---- Cooperative weight load into shared memory ----
        // All BLOCK_THREADS = TILE_HW * TILE_C threads load weights collaboratively.
        // Weight layout: W[c_out, kh, kw, c_in] = W[c_out * 9 * Cin + kh*3*Cin + kw*Cin + c_in]
        // We load smem_W[kh*3+kw][cin_local][cout_local] for this tile.
        int linear_thread = threadIdx.y * TILE_HW + threadIdx.x;
        int total_w_elems = 9 * CIN_TILE * TILE_C;

        for (int load_idx = linear_thread; load_idx < total_w_elems; load_idx += BLOCK_THREADS) {
            int kernel_pos = load_idx / (CIN_TILE * TILE_C);    // 0..8 (kh*3+kw)
            int rem        = load_idx % (CIN_TILE * TILE_C);
            int cin_local  = rem / TILE_C;
            int cout_loc   = rem % TILE_C;

            int global_cin  = cin_tile_start + cin_local;
            int global_cout = cout_base + cout_loc;

            float weight_val = 0.0f;
            if (global_cin < Cin && global_cout < Cout) {
                int kh = kernel_pos / 3;
                int kw = kernel_pos % 3;
                // W[global_cout, kh, kw, global_cin] in OIHW-like: [Cout][3][3][Cin]
                weight_val = W[(size_t)global_cout * 9 * Cin + (kh * 3 + kw) * Cin + global_cin];
            }
            smem_W[kernel_pos][cin_local][cout_loc] = weight_val;
        }
        __syncthreads();

        // ---- Compute: iterate over 3×3 kernel positions ----
        if (c_out < Cout && hw_idx < H * W_dim) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    int h_in = h_out + kh - 1;  // padding=1
                    int w_in = w_out + kw - 1;

                    // Zero-pad for out-of-bounds
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_dim) {
                        size_t x_base = (size_t)sample_n * H * W_dim * Cin
                                      + (size_t)h_in * W_dim * Cin
                                      + (size_t)w_in * Cin
                                      + cin_tile_start;
                        int kernel_idx = kh * 3 + kw;

                        // Inner loop: accumulate over Cin tile — dense FFMA sequence
                        for (int cin_local = 0; cin_local < cin_tile_len; cin_local++) {
                            float input_val = X[x_base + cin_local];
                            accumulator += smem_W[kernel_idx][cin_local][cout_local] * input_val;  // FFMA
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- Store output ----
    if (c_out < Cout && hw_idx < H * W_dim) {
        size_t y_flat = (size_t)sample_n * H * W_dim * Cout
                      + (size_t)h_out * W_dim * Cout
                      + (size_t)w_out * Cout
                      + c_out;
        Y[y_flat] = accumulator;
    }
}

// -----------------------------------------------------------------------
// Kernel: conv2d_1x1_nhwc
//
// 1×1 convolution: no spatial window, just a channel mixing linear layer.
// This is identical to a matrix multiply:
//   Y[n, hw, c_out] = bias[c_out] + sum_{c_in} X[n, hw, c_in] * W[c_out, c_in]
//
// Grid:  (N, ceil(H*W / TILE_HW), ceil(Cout / TILE_C))
// Block: (TILE_HW, TILE_C, 1)
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(BLOCK_THREADS)
void conv2d_1x1_nhwc(
    const float * __restrict__ X,
    const float * __restrict__ W,
    const float * __restrict__ bias,
    float       * __restrict__ Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    __shared__ float smem_W1[CIN_TILE][TILE_C];

    int sample_n   = blockIdx.x;
    int hw_base    = blockIdx.y * TILE_HW;
    int cout_base  = blockIdx.z * TILE_C;

    int hw_local   = threadIdx.x;
    int cout_local = threadIdx.y;

    int hw_idx = hw_base + hw_local;
    int c_out  = cout_base + cout_local;

    float accumulator = 0.0f;
    if (bias != NULL && c_out < Cout) {
        accumulator = bias[c_out];
    }

    for (int cin_tile_start = 0; cin_tile_start < Cin; cin_tile_start += CIN_TILE) {
        int cin_tile_len = min(CIN_TILE, Cin - cin_tile_start);

        // Cooperative weight load: W[c_out, c_in] tiled
        int linear_thread = threadIdx.y * TILE_HW + threadIdx.x;
        for (int load_idx = linear_thread; load_idx < CIN_TILE * TILE_C; load_idx += BLOCK_THREADS) {
            int cin_local  = load_idx / TILE_C;
            int cout_loc   = load_idx % TILE_C;
            int global_cin  = cin_tile_start + cin_local;
            int global_cout = cout_base + cout_loc;
            float weight_val = 0.0f;
            if (global_cin < Cin && global_cout < Cout) {
                weight_val = W[(size_t)global_cout * Cin + global_cin];
            }
            smem_W1[cin_local][cout_loc] = weight_val;
        }
        __syncthreads();

        if (c_out < Cout && hw_idx < H * W_dim) {
            size_t x_base = (size_t)sample_n * H * W_dim * Cin
                          + (size_t)hw_idx * Cin
                          + cin_tile_start;
            for (int cin_local = 0; cin_local < cin_tile_len; cin_local++) {
                accumulator += smem_W1[cin_local][cout_local] * X[x_base + cin_local];  // FFMA
            }
        }
        __syncthreads();
    }

    if (c_out < Cout && hw_idx < H * W_dim) {
        size_t y_flat = (size_t)sample_n * H * W_dim * Cout
                      + (size_t)hw_idx * Cout
                      + c_out;
        Y[y_flat] = accumulator;
    }
}
