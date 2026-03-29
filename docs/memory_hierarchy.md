# GA104 Memory Hierarchy (RTX 3070 Ti)

Understanding the memory system is essential for writing performant SASS.
Every data movement decision in a hand-tuned kernel comes down to this hierarchy.

## Overview

```
Registers (fastest, per-thread)
    |
    v
Shared Memory / L1 Cache (per SM, ~1 cycle)
    |
    v
L2 Cache (shared across all SMs, ~200 cycles)
    |
    v
GDDR6X VRAM (8 GB, ~400 cycles)
    |
    v [PCIe]
System RAM
```

## Registers

- **64K x 32-bit registers per SM** (256 KB)
- Divided equally among resident warps
- 48 warps max → 64K / 48 = 1365 registers per warp → 42 per thread
- But you can use up to **255 registers per thread** — at the cost of occupancy
- No load/store needed — register access is instantaneous

**In SASS**: `R0`–`R255` are the 32-bit general-purpose registers.
`RZ` is the hardwired zero register.

Pairs of registers hold 64-bit values (pointers, FP64): `R2:R3` means R2 is low, R3 is high.

Uniform registers `UR0`–`UR63` are warp-uniform (same value across all threads) — useful
for loop counters and addresses that are identical for all threads.

## Shared Memory / L1 Cache

- **128 KB per SM** on GA104
- Configurable split between L1 cache and explicit shared memory
- Max **99 KB** usable as explicit shared memory per thread block (kernel attribute)
- Remaining becomes L1 cache for global memory accesses
- Access latency: ~20 cycles

**SASS instructions**: `STS` (store to shared), `LDS` (load from shared)

**Bank conflicts**: 32 banks, 4 bytes each. A warp of 32 threads accessing
32 different banks → no conflict, all served in 1 cycle.
N threads accessing the same bank → N cycles (serialized).

**Avoiding conflicts in GEMM**: pad shared memory arrays by 1 element,
or use swizzled indexing patterns.

**Async copies** (`LDGSTS` on Ampere): copy directly from global to shared
without going through registers. Lets you overlap data transfer with computation.

## L2 Cache

- **4 MB** on GA104
- Shared across all 48 SMs
- Access latency: ~200 cycles
- Bandwidth: ~400 GB/s (internal)

When you write `LDG.E R4, [R2]`, the GPU first checks L2.
If the data is there (cache hit), you get ~200 cycles latency.
If not (cache miss), you wait for GDDR6X → ~400 cycles.

L2 is transparent — no SASS instructions to explicitly control it,
but you can use cache hints: `LDG.E.CONSTANT` (read-only, cache aggressively)
vs `LDG.E.LAST_USE` (evict after this load).

## GDDR6X VRAM

- **8 GB** on RTX 3070 Ti
- Bandwidth: **608 GB/s** theoretical peak
- Effective latency: ~400 cycles (full roundtrip when L2 misses)

The 608 GB/s is achievable only with:
1. Coalesced accesses: 32 threads access 32 consecutive 4-byte values → 1 transaction
2. 128-bit loads: `LDG.E.128` reads 16 bytes per thread → maximizes bandwidth
3. No bank conflicts or serialization

**Arithmetic intensity** = FLOPs / bytes accessed.
To be compute-bound (not memory-bound), you need enough FLOPs per byte:
- RTX 3070 Ti FP32 peak: 21.7 TFLOPS
- VRAM bandwidth: 608 GB/s
- Required intensity: 21.7e12 / 608e9 ≈ **36 FLOPs/byte**

A naive GEMM reads A and B once for each output → low intensity → memory-bound.
A tiled GEMM reuses tiles from shared memory → high intensity → compute-bound.

## Memory Layout for GEMM

For an M×K × K×N matrix multiply with tile size T:

```
Global memory:
  A: M×K floats  (row-major: A[row][col] = A[row*K + col])
  B: K×N floats  (col-major for B^T, or handle transposition in indexing)
  C: M×N floats  (row-major output)

Shared memory per block (tile T×T):
  As: T×T floats  (tile of A) = T*T*4 bytes
  Bs: T×T floats  (tile of B) = T*T*4 bytes

With T=32:  2 * 32*32*4 = 8 KB  (well within 99 KB limit)
With T=64:  2 * 64*64*4 = 32 KB (still fine)
With T=128: 2 * 128*128*4 = 128 KB (exceeds limit — need FP16 or double-buffer tricks)
```

## Constant Memory

- **64 KB total**, 8 banks of 8 KB each
- Bank 0 (`c[0x0][...]`): kernel arguments — automatically populated by the driver
- Access: broadcast to all threads in a warp if they all read the same address (1 cycle)
- Access: serialized if threads read different addresses (32 cycles)

Use constant memory for: loop counts, matrix dimensions, shared scaling factors.
Never use it for per-thread varying data.

## Register Spilling

When a kernel uses more registers than available (given its occupancy target),
the compiler **spills** registers to **local memory** — a per-thread region in VRAM.

Spilling appears in SASS as:
```sass
STL [R0+0x10], R4    ; spill R4 to local memory
LDL R4, [R0+0x10]   ; reload R4 from local memory
```

Spilling is very expensive — effectively an LDG/STG to VRAM.
Avoid by: reducing register usage, accepting lower occupancy,
or restructuring the algorithm to process smaller tiles.

## Practical Implications for ML Kernels

| Kernel | Bottleneck | Key optimization |
|---|---|---|
| Small SGEMM (M<64) | Memory bandwidth | Ensure coalesced loads |
| Large SGEMM (M>512) | Compute | Tile into shared memory, maximize FFMA IPC |
| HGEMM (Tensor Core) | Compute | HMMA instruction, 128-bit loads, no bank conflicts |
| Softmax (per row) | Memory bandwidth | Fuse row max + normalize into single pass |
| LayerNorm | Memory bandwidth | Fuse mean + variance into single pass |
| Flash Attention | Memory bandwidth | Fuse QK^T + softmax + V into single CUDA kernel |
| Conv2d (large) | Compute | im2col + HGEMM |
| Conv2d (small 3x3) | Memory | Direct convolution with shared memory halo |
