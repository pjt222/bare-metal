# Phase 2: SGEMM — Matrix Multiply in SASS

## Overview

SGEMM (Single-precision GEneral Matrix Multiply) is the most important GPU kernel.
Everything in transformer and diffusion models reduces to matrix multiplies.
Understanding its SASS is the key to understanding GPU performance.

## Three Stages

| Stage | File | What you learn |
|---|---|---|
| A | `naive.cu` | Simplest GEMM SASS — the multiply-accumulate loop |
| B | `tiled.cu` | Shared memory tiling, LDS/STS, BAR.SYNC, FFMA pipeline |
| C | `hand_tuned.cuasm` | Stall count tuning, latency hiding, software pipelining |

## Benchmark Results (RTX 3070 Ti Laptop, sm_86)

| Matrix | naive | tiled | vs peak |
|---|---|---|---|
| 1024×1024 | 461 GFLOPS | 428 GFLOPS | ~2% |
| 2048×2048 | 996 GFLOPS | 1031 GFLOPS | ~5% |

At 2048× the tiled kernel pulls ahead. At 4096× the gap widens further.
Both are still far from the 21700 GFLOPS theoretical peak — that requires
register blocking, vectorized loads, and software pipelining (Stage C).

## Key SASS Patterns in the Tiled Kernel

### 1. Thread Block Barrier
```sass
BAR.SYNC.DEFER_BLOCKING 0x0    ; all threads wait here after loading tiles
```
Generated from `__syncthreads()`. Every shared memory tile load is bounded by two of these.

### 2. Shared Memory Loads (LDS)
The compiler pre-fetches all 32 A and 32 B values from the tile before computing:
```sass
LDS R29, [R10]         ; A[thread_row][0]
LDS R28, [R7]          ; B[0][thread_col]
LDS R33, [R10+0x84]    ; A[thread_row][1]   (0x84 = 132 = 33*4 with padding)
LDS R34, [R7+0x4]      ; B[1][thread_col]
...                    ; 32 pairs total
```
The `+0x84` offset (not `+0x80`) shows the **+1 padding** we added to avoid bank conflicts:
- Without padding: `tile_a[TILE_SIZE][TILE_SIZE]` → stride 32 × 4 = 128 bytes → 32 banks × 4 bytes → conflict-free actually
- With padding: `tile_a[TILE_SIZE][TILE_SIZE+1]` → stride 33 × 4 = 132 bytes → offset 0x84 per row

### 3. FFMA — the compute kernel
The compiler emits exactly **32 FFMA instructions** (one per `k` in the inner loop):
```sass
FFMA R28, R29, R28, R20    ; acc = A[row][0] * B[0][col] + 0
FFMA R34, R33, R34, R28    ; acc = A[row][1] * B[1][col] + acc
FFMA R33, R27, R26, R34    ; ...
...                         ; 32 total — inner loop fully unrolled
```
The `#pragma unroll` on the inner loop caused full unrolling: no branch, no loop counter — 32 raw FFMA instructions.

**This is the hot loop.** Everything else in GEMM optimization is about feeding these FFMAs with data fast enough that they never stall.

### 4. Interleaved LDS + FFMA (latency hiding)
Notice the pattern: the compiler **interleaves** LDS and FFMA instructions:
```sass
LDS R29, [R10]           ; load A[row][0]    ← starts LDS
LDS R28, [R7]            ; load B[0][col]    ← starts LDS
LDS R33, [R10+0x84]      ; load A[row][1]
LDS R34, [R7+0x4]        ; load B[1][col]
...more LDS...
FFMA R28, R29, R28, R20  ; ← first FFMA (LDS results ready by now)
LDS R16, [R10+0x318]     ; load A[row][12]   ← next tile loads
LDS R17, [R7+0x18]
FFMA R34, R33, R34, R28  ; second FFMA
...
```
This is **latency hiding**: LDS has ~20 cycle latency. By issuing many LDS instructions before the first FFMA, the data is ready by the time FFMA needs it.

## What's Wrong With the Tiled Kernel (Why Only 5% of Peak?)

1. **One accumulator register** (`accumulator`) — only 1 output value per thread.
   High-performance GEMM uses 4×4 or 8×8 register tiles: each thread computes
   16–64 output elements, dramatically increasing compute/byte ratio.

2. **32-bit loads** (LDS) not 128-bit (LDS.128).
   Loading 4 floats at once (`LDS.128`) reduces instruction count 4×.

3. **No global memory vectorization** — loads from A and B use 32-bit LDG, not 128-bit.

4. **No double-buffering** — the BAR.SYNC stalls the warp while waiting for
   all threads to load. Double-buffering would load the next tile while computing the current.

5. **Occupancy limited** by 32×32 block (1024 threads) × 1 block/SM = 1024 threads,
   but sm_86 can hold 1536 threads → 66% occupancy.

## The Path to Stage C (Hand-Tuned SASS)

Approaching peak requires combining all of these:
- **Register tiling**: 8×8 output elements per thread → 64 FFMA per inner loop
- **LDG.E.128**: vector global loads (4 floats per instruction)
- **LDS.128**: vector shared loads
- **Double-buffering**: overlap global loads with FFMA computation
- **Stall count tuning**: reduce S values in control codes to eliminate idle cycles
- **Software prefetch**: issue next iteration's global loads early

This is what `hand_tuned.cuasm` implements. Study the tiled SASS first,
then look at how the control codes change in the hand-tuned version.

## Build Commands

```bash
# Compile both kernels
nvcc --cubin -arch=sm_86 -O2 -o naive.sm_86.cubin naive.cu
nvcc --cubin -arch=sm_86 -O2 -o tiled.sm_86.cubin tiled.cu

# Disassemble to study SASS
cuobjdump -sass naive.sm_86.cubin > naive.sm_86.sass
cuobjdump -sass tiled.sm_86.cubin > tiled.sm_86.sass

# Count FFMA instructions (should be 32 for tiled with TILE_SIZE=32)
grep -c FFMA tiled.sm_86.sass

# Build and run benchmark
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 1024 1024 1024    # fast test
./bench 4096 4096 4096    # more representative
```
