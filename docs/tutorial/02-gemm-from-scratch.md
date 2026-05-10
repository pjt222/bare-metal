# Chapter 02 — GEMM from Scratch

> Foundational chapter. Chapters 03 (INT8) and 04 (pipelining) assume this
> material. Chapter 05 (Flash Attention) is built on top of HGEMM in here.

GEMM (general matrix-matrix multiply) is the most important kernel in
modern deep learning. Every transformer attention head, every linear
layer, every conv2d (when implemented via im2col or implicit GEMM) is
ultimately a GEMM. If your GEMM is slow, your model is slow. If you do
not understand how a fast GEMM works, you do not understand how a fast
model works.

This chapter walks through five GEMM kernels in order of increasing
sophistication, ending at a 16-warp HMMA kernel that achieves **31,910
GFLOPS** on RTX 3070 Ti — about 18% of the FP16 Tensor Core peak. Each
version exposes a specific bottleneck that the next version solves.

## The problem

Compute `C = A · B` where `A ∈ ℝ^(M×K)`, `B ∈ ℝ^(K×N)`, `C ∈ ℝ^(M×N)`.
The natural triple-loop:

```
for m in 0..M:
    for n in 0..N:
        for k in 0..K:
            C[m, n] += A[m, k] * B[k, n]
```

Total work: `2·M·N·K` floating-point operations (one multiply-add per inner
loop iteration counts as 2 ops). For 4096³ that is 137 billion ops.

The challenge: at peak FP32 throughput on GA104 (21.7 TFLOPS), 137 GOps
takes 6.3 ms. At peak FP16 Tensor Core throughput (174 TFLOPS), it takes
0.79 ms. The kernel either gets close to one of these bounds or it does
not. Most naive kernels achieve 1-3% of peak.

## Version 1 — Naive SGEMM (`kernels/gemm/sgemm/naive.cu`)

**Setup**: one thread per output element. Each thread loops over k and
accumulates `C[m, n]` directly.

```cpp
__global__ void naive_sgemm(...) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
}
```

The SASS is a single FFMA in a loop:

```sass
FFMA R0, R3, R4, R0 ;
```

**Performance**: 461 GFLOPS at M=N=K=1024. About 2% of FP32 peak.

**Why so slow**: each FFMA reads from DRAM. There is no caching of A rows
or B columns across threads. For 1024³, each element of A is read N=1024
times and each element of B is read M=1024 times. Total DRAM traffic:
~32 GB. At 608 GB/s peak, that is 53 ms of bandwidth — 100× more than
the compute time. The kernel is bandwidth-bound by 100×.

**Lesson**: redundant DRAM reads dominate naive GEMMs. Every byte of A
and B is read O(M+N) times, but only needs to be read once. Reducing
that reread factor is the entire art of GEMM optimization.

## Version 2 — Tiled SGEMM (`kernels/gemm/sgemm/tiled.cu`)

**Setup**: each thread block computes a `TILE_M × TILE_N` output block
(e.g., 32×32) cooperatively. The block iterates over k in chunks of
`TILE_K`. Each iteration:

1. Cooperatively load `A[row_block, k_block:k_block+TILE_K]` into smem
2. Cooperatively load `B[k_block:k_block+TILE_K, col_block]` into smem
3. `__syncthreads()`
4. Compute partial product: each thread accumulates one output element
   from the smem tiles
5. `__syncthreads()`

Critically: each smem tile is read by every thread in the block. A 32×32
A-tile is read 32 times during compute (once per k-step), but loaded from
DRAM only once. The DRAM reread factor drops from O(N) per A element to
O(N / TILE_N).

**SASS for the tiled version** uses LDS (load shared) for the inner loop
instead of LDG (load global). LDS has ~20 cycle latency, LDG has ~300+
cycle latency for an L2 miss. Same instruction count, but each one is
~15× faster.

**Performance**: 428 GFLOPS at 1024×1024 (slightly slower than naive due
to small-problem overhead), 1031 GFLOPS at 2048×2048 (where the tiling
benefit shows). Both still ~5% of peak.

**What is still wrong**:
- One accumulator per thread. Each thread does only `TILE_K` FFMAs per
  k-block, then `__syncthreads`. The compute/data-load ratio is too low.
- 32-bit LDS, not 128-bit. Each load fetches one float instead of four.
- No double-buffering — the `__syncthreads` stalls the warp.
- 1 block per SM at this thread count → low occupancy.

The lesson: simple tiling moves the bottleneck from DRAM bandwidth to
shared memory bandwidth and instruction issue rate, but does not eliminate
the bottleneck. The next move is to give each thread *more work per load*.

## Version 3 — Register-blocked SGEMM (`kernels/gemm/sgemm/register_blocked.cu`)

**Setup**: each thread computes an `RM × RN` block of outputs (e.g., 8×8 =
64 output elements per thread). The thread loads `RM` values of A and `RN`
values of B from smem per k-step, then does `RM × RN` FFMAs in a fully
unrolled inner loop.

```cpp
float acc[RM][RN] = {0.0f};
for (int kk = 0; kk < TILE_K; kk++) {
    float a_reg[RM], b_reg[RN];
    #pragma unroll
    for (int i = 0; i < RM; i++) a_reg[i] = smem_A[row_in_block + i][kk];
    #pragma unroll
    for (int j = 0; j < RN; j++) b_reg[j] = smem_B[kk][col_in_block + j];
    #pragma unroll
    for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++)
            acc[i][j] += a_reg[i] * b_reg[j];
}
```

Now each k-step does 8 LDS + 64 FFMA — a compute/load ratio of 8:1
instead of 1:1. The FFMA pipeline can issue without stalling on LDS.

**Performance**: SGEMM register-blocked reaches ~5,000 GFLOPS at 4096³.
Still only ~23% of FP32 peak. The remaining gap is occupancy (1 block per
SM at large register counts) and the synchronous LDG of A/B tiles between
k-blocks.

**Lesson**: register tiling raises the compute/load ratio. This is the
mechanism by which a kernel reaches above 10% of peak. Below 10% almost
always means insufficient register tiling.

This is the practical ceiling of FP32 GEMM on Ampere with hand-written
CUDA C++. The next 8× requires switching to FP16 Tensor Cores.

## Version 4 — HGEMM (FP16 Tensor Core baseline, `kernels/gemm/hgemm/hgemm.cu`)

**Setup**: same tiled structure but using the WMMA API. Each warp owns a
16×16 output tile (one HMMA-sized fragment). Per k-step:

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);

for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A_tile + k, K);
    wmma::load_matrix_sync(b_frag, B_tile + k * N, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}

wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
```

The compiler translates `wmma::mma_sync` into one `mma.sync.aligned.m16n8k16`
PTX instruction, which becomes **two HMMA.16816.F32** SASS instructions.
Each HMMA does 2048 FP16 MACs — 32 lanes × 16×8×16 / 32 = 64 MACs per lane.

Equivalence: one HMMA = 64 FFMAs worth of work in 16 cycles = 4× faster.
Two HMMAs per `mma_sync` = 8× faster than the equivalent FFMA loop.

**SASS pattern**:
```sass
LDSM.16.M88.4 R[regs], [smem]    ; ldmatrix.x4 — load 4 8×8 fragments
HMMA.16816.F32 R[acc], R[a], R[b], R[acc]
HMMA.16816.F32 R[acc], R[a], R[b], R[acc]   ; second half of mma.sync
```

**Performance**: ~7,853 GFLOPS at 4096³ on the basic HGEMM. About 4.5% of
FP16 TC peak. Better than register-blocked SGEMM in absolute terms, but a
smaller fraction of its peak.

**Why so much room left**: each warp computes only 16×16 output. With 4
warps per block and 1 block per SM, that is 32×32 output per SM-cycle —
not enough to amortize tile load cost. The next step is to give each
warp more output fragments and run more blocks per SM.

## Version 5 — 16-warp HGEMM (`kernels/gemm/hgemm/hgemm_16warp.cu`)

**Setup**: this is where the project's GEMM peaks. Block size grows to
`BM=128, BN=128, BK=32`. Sixteen warps per block (512 threads), arranged
as a 4×4 warp grid where each warp owns 32×32 output. Per warp: 4 WMMA
fragments (`2 × 2` arrangement of 16×16 sub-tiles).

Smem layout with explicit padding for bank-conflict elimination:
- `smem_A[2][BM × (BK + 8)]` = 2 × 128 × 40 × 2 = 20.5 KB (double-buffered)
- `smem_B[2][BK × (BN + 8)]` = 2 × 32 × 136 × 2 = 17 KB

Total ~38 KB → **2 blocks/SM** (16 warps × 2 = 32 warps/SM, full occupancy).

The doubled smem enables synchronous double-buffering: tile N+1 loads
into the inactive buffer while compute runs on tile N's buffer. No
cp.async needed at this stage; classic LDG → STS double-buffer suffices.

**Per-warp inner loop**: 4 WMMA tiles × 2 K-steps = 8 mma_sync calls = 16
HMMA instructions. The HMMA pipeline is now the bottleneck, not memory
latency.

**Performance**:

| Matrix | GFLOPS | vs FP32 tiled | vs FP16 peak |
|---|---|---|---|
| 512³ | 5,438 | 5.3× | 3.1% |
| 2048³ | ~22,000 | ~22× | 12.6% |
| **4096³** | **31,910** | **41×** | **18.3%** |

At 4096³ this kernel reaches 18.3% of the FP16 Tensor Core peak. CUTLASS
mature kernels reach 60-80% of peak with deeper optimization (cp.async +
software pipelining, swizzled smem layouts, persistent grids), but 18% is
already strong for hand-written code without an inner-loop autotuner.

**The decisive optimizations**:

1. **Larger block tile (128×128)** — each block does 16,384 output elements, amortizing tile load
2. **16 warps per block** — full SM occupancy (32 warps/SM at 2 blocks)
3. **+8 padding** on smem rows — eliminates bank conflicts on `ldmatrix.x4` (per the rule `stride mod 32 ≠ 0` for FP16 strides; see chapter 03 for the underlying analysis)
4. **Double-buffered LDG → STS** — overlap tile load with HMMA via classic two-buffer pattern
5. **Pipelined smem loads** — issue all `ldmatrix` for a k-step before the first HMMA

## Performance summary

| variant | matrix size | GFLOPS | % of peak | bottleneck |
|---|---|---|---|---|
| Naive SGEMM | 1024³ | 461 | 2% (FP32) | DRAM bandwidth |
| Tiled SGEMM | 2048³ | 1,031 | 5% (FP32) | smem bandwidth + sync |
| Register-blocked SGEMM | 4096³ | ~5,000 | ~23% (FP32) | occupancy + LDG sync |
| HGEMM (basic WMMA) | 4096³ | 7,853 | 4.5% (FP16 TC) | tile load amortization |
| **16-warp HGEMM** | **4096³** | **31,910** | **18.3%** (FP16 TC) | **HMMA pipeline + DRAM** |

Each version is a genuine 2-8× over the previous. The aggregate: 70× from
naive SGEMM to 16-warp HGEMM, primarily through three structural changes
(tiling, register blocking, switch to Tensor Cores) and one occupancy
maximization (block tile size + warp count + padding + double-buffer).

## What this chapter teaches

GEMM optimization is not about any single trick. It is about *layered
amortization*: each level of the memory hierarchy must amortize the cost
of the level below it.

- Tiling amortizes DRAM traffic across smem (per-block factor of `TILE_K`)
- Register blocking amortizes smem traffic across registers (per-thread factor of `RM × RN`)
- WMMA fragments amortize register loads across HMMAs (per-warp factor of 2048 MACs/HMMA)
- Multiple warps amortize HMMA latency across warp scheduling
- Multiple blocks per SM amortize block-level startup overhead

Each layer's amortization factor multiplies. A naive kernel has all
factors ≈ 1, so it pays full cost everywhere. A 16-warp HGEMM has factors
~32, ~64, ~2048, ~4, ~2 = ~16M-fold amortization.

The four laws (chapter 06) all show up here:
1. **Feed Tensor Cores**: every HMMA must have its A/B/C ready when the slot opens — that's what double-buffering and `ldmatrix` pipelining do
2. **Read each byte of DRAM once**: tile loads ensure each A/B element is read exactly `K/TILE_K` times from DRAM, not `M·N` times
3. **Fill warp schedulers**: 16 warps × 2 blocks/SM = 32 warps/SM is the maximum useful warp count on Ampere
4. **Stay below the smem cliff**: the BM=128/BN=128/BK=32 + padding = 38 KB layout was chosen specifically to fit 2 blocks/SM (50 KB cliff would force 1 block/SM and lose half the occupancy)

## How to run it yourself

```bash
cd /mnt/d/dev/p/bare-metal/kernels/gemm/sgemm

# Build SGEMM variants
nvcc --cubin -arch=sm_86 -O2 -o naive.sm_86.cubin naive.cu
nvcc --cubin -arch=sm_86 -O2 -o tiled.sm_86.cubin tiled.cu
nvcc --cubin -arch=sm_86 -O2 -o register_blocked.sm_86.cubin register_blocked.cu

nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 1024 1024 1024
./bench 4096 4096 4096

cd ../hgemm

# Build HGEMM variants
nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu
nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin hgemm_16warp.cu

nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 4096 4096 4096   # expect ~31900 GFLOPS for the 16-warp variant
```

## Inspecting SASS

```bash
# Naive: one FFMA in a loop
cuobjdump -sass naive.sm_86.cubin | grep -A1 'naive_sgemm' | head -20

# 16-warp HGEMM: lots of HMMA with LDSM interleaved
cuobjdump -sass hgemm_16warp.sm_86.cubin | grep -E 'HMMA|LDSM' | head -40
cuobjdump -sass hgemm_16warp.sm_86.cubin | grep HMMA | wc -l   # how many HMMA total?
```

## Source files

- `kernels/gemm/sgemm/naive.cu`, `tiled.cu`, `register_blocked.cu` (FP32 progression)
- `kernels/gemm/hgemm/hgemm.cu` (basic WMMA HGEMM)
- `kernels/gemm/hgemm/hgemm_16warp.cu` (the 31910 GFLOPS kernel)
- `kernels/gemm/hgemm/hgemm_256x128.cu`, `hgemm_tiled.cu`, `hgemm_tiled_direct.cu` (intermediate variants)
- `kernels/gemm/hgemm/README.md` (deeper SASS-level walkthrough)

## Cross-references

- Chapter 03 — INT8 Tensor Cores (the IMMA equivalent of HMMA)
- Chapter 04 — Software Pipelining (cp.async on top of HGEMM, regime-dependent)
- Chapter 05 — Flash Attention (uses HGEMM as the QK^T and PV building block)
- Chapter 06 — The Four Laws (every GEMM lesson reinforces all four)
- `docs/gpu_reflections.md` Insights 1, 7 — DRAM reread analysis
- `docs/ampere_sass_reference.md` — HMMA / LDSM / control codes
