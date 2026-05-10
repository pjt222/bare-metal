# Phase 2: Softmax — SHFL.BFLY + MUFU.EX2 + MUFU.RCP

## Why Softmax

Every attention layer computes softmax over the score matrix (QK^T).
For a sequence length N, each row is a vector of N floats — we need:
1. Row max (reduction)
2. `exp(x - max)` for each element
3. Row sum (reduction)
4. Normalize by dividing (multiply by reciprocal)

At the SASS level, this is a showcase for three underused hardware units:

| Operation | SASS Instruction | Notes |
|---|---|---|
| Warp reduction | `SHFL.BFLY` | 5 shuffles for 32-element reduce |
| Fast exponentiation | `MUFU.EX2` | 2^x in hardware, one cycle |
| Fast reciprocal | `MUFU.RCP` | 1/x in hardware, one cycle |

## MUFU.EX2 — The Key Instruction

The GPU has no direct `exp` hardware — but it has `2^x` via `MUFU.EX2`.

```
exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408...)
```

In CUDA C: `exp2f(x * LOG2E)` compiles to:
```sass
FMUL  R3, R10, 1.4426950408889634   ; scale by log2(e)
MUFU.EX2 R3, R3                     ; 2^R3 in hardware
```

With `-use_fast_math`, `expf(x)` also maps to `MUFU.EX2`. We use `exp2f` explicitly to guarantee the instruction regardless of compile flags.

## SHFL.BFLY — Warp Butterfly Reduction

A warp (32 threads) reduces to a single value in 5 `SHFL.BFLY` instructions:

```sass
SHFL.BFLY PT, R12, R11, 0x10, 0x1f   ; exchange lane XOR 16
SHFL.BFLY PT, R13, R12,  0x8, 0x1f   ; exchange lane XOR  8
SHFL.BFLY PT, R14, R13,  0x4, 0x1f   ; exchange lane XOR  4
SHFL.BFLY PT, R15, R14,  0x2, 0x1f   ; exchange lane XOR  2
SHFL.BFLY PT, R6,  R15,  0x1, 0x1f   ; exchange lane XOR  1
```

After 5 rounds, every lane holds the reduction result (max or sum).
The `0x1f` mask means "all 32 lanes participate."

For multi-warp blocks, warp results are written to shared memory and the first warp reduces them.

## Numerical Stability

Raw `softmax(x) = exp(x) / sum(exp(x))` overflows for large x (exp(89) = FLT_MAX).

The stable form: **subtract max first.**
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

The max cancels algebraically: the result is identical. In practice, `exp(x_i - max)` is always in `[0, 1]` — no overflow.

## Measured Results

| Configuration | GB/s | % of 608 GB/s peak |
|---|---|---|
| 65536 rows × 32 (warp kernel) | 145 | 24% |
| 65536 rows × 128 (block kernel) | 300 | 49% |
| 16384 rows × 512 (block kernel) | 410 | 67% |

Wider rows improve bandwidth utilization because:
- Longer vectors → better memory coalescing
- More work per block → better GPU occupancy
- Reduction overhead amortized over more elements

For attention with head dimension d=64 (typical), `softmax_block` with 128 threads handles exactly one attention head's score vector per block.

## Two Kernels

### `softmax_warp` — row_width ≤ 32
- One warp per row, pure `SHFL.BFLY`, no shared memory
- Cleanest SASS to study: reduction is 5 SHFL instructions, nothing else
- Launch: `grid=(num_rows,1,1)`, `block=(32,1,1)`

### `softmax_block` — row_width ≤ `ELEMENTS_PER_THREAD × BLOCK_SIZE` = 512
- One block per row, `BLOCK_SIZE=128` threads, each handles 4 elements
- Inter-warp reduction via `NUM_WARPS_IN_BLOCK=4`-entry shared memory array
- Launch: `grid=(num_rows,1,1)`, `block=(128,1,1)`

## Inspecting the SASS

```bash
nvcc --cubin -arch=sm_86 -O2 -o softmax.sm_86.cubin softmax.cu
cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'
```

Expected output:
```
SHFL.BFLY PT, R12, R11, 0x10, 0x1f    ← max reduction step 1
SHFL.BFLY PT, R13, R12,  0x8, 0x1f    ← max reduction step 2
...
MUFU.EX2 R3, R10                        ← exp via 2^x hardware unit
MUFU.RCP R7, R0                         ← 1/sum via reciprocal hardware unit
```

## Build

```bash
# Compile kernel cubin
nvcc --cubin -arch=sm_86 -O2 -o softmax.sm_86.cubin softmax.cu

# Inspect key instructions
cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'

# Build and run bench
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 65536 32     # warp kernel (pure SHFL, no shared mem)
./bench 65536 128    # block kernel (128 threads, 4 elements/thread)
./bench 16384 512    # wider rows
```

## What's Next: LayerNorm

LayerNorm applies softmax's reduction pattern to **variance computation**:
1. Mean of the row: sum reduction → divide by N
2. Variance: sum of `(x - mean)^2` → `MUFU.RSQ` for reciprocal sqrt
3. Normalize: `(x - mean) * rsqrt(var + eps)`

The `MUFU.RSQ` instruction (reciprocal square root) is the key new piece.
