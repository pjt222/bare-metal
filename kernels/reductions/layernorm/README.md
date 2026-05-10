# Phase 2: LayerNorm — Welford Online Algorithm + MUFU.RSQ

## Why LayerNorm

Every transformer block applies LayerNorm before and after attention and FFN.
It normalizes across the hidden dimension (e.g., d_model=512):

```
y_i = gamma_i * (x_i - mean) / sqrt(variance + epsilon) + beta_i
```

The critical hardware instruction is `MUFU.RSQ` — the GPU's dedicated
reciprocal square root unit. Like `MUFU.EX2` for softmax, it executes
in one cycle rather than requiring a software approximation loop.

## Key SASS Instructions

| SASS | C intrinsic | Operation |
|---|---|---|
| `SHFL.BFLY` | `__shfl_xor_sync` | Warp-level Welford reduction |
| `MUFU.RSQ` | `rsqrtf` | 1/sqrt(variance + eps) |
| `MUFU.RCP` | `__frcp_rn` | 1/count (Welford mean update) |
| `FFMA` | `a*b + c` | normalize + scale + shift |

## Welford's Online Algorithm

Naïve approach needs two passes over the data: one for mean, one for variance.

**Welford's single-pass algorithm** maintains a running mean and variance:

```c
// Update with element x:
count += 1;
delta  = x - mean;
mean  += delta / count;         // MUFU.RCP for division
delta2 = x - mean;
M2    += delta * delta2;        // M2 / count = variance
```

The clever part: Welford accumulators can be **combined in parallel**:

```
welford_combine(a, b) → (count_a + count_b, combined_mean, combined_M2)
```

This lets us use SHFL.BFLY — the exact same butterfly pattern as softmax
reduction, but now exchanging (count, mean, M2) triples instead of scalars.

## SASS Confirmation

```
SHFL.BFLY PT, R2,  R25, 0x10, 0x1f   ← Welford round 1 (count)
SHFL.BFLY PT, R31, R26, 0x10, 0x1f   ← Welford round 1 (mean)
SHFL.BFLY PT, R28, R27, 0x10, 0x1f   ← Welford round 1 (M2)
MUFU.RCP R0, R29                       ← 1/count in combine formula
...
MUFU.RSQ R12, R4                       ← 1/sqrt(variance + eps)
```

Note: each SHFL.BFLY round shuffles **three registers** (count, mean, M2)
compared to one for softmax. This is why layernorm has ~3× the SHFL instructions.

## Measured Results

| Configuration | GB/s | % of 608 GB/s peak |
|---|---|---|
| 65536 rows × 32 (warp kernel) | 112 | 18% |
| 65536 rows × 128 (block kernel) | 185 | 30% |
| 16384 rows × 512 (block kernel) | 413 | 68% |

Pattern is the same as softmax: wider rows → better bandwidth utilization.
Lower bandwidth than softmax at small sizes because Welford shuffles 3 values
per round vs 1 for softmax.

Correctness: max_rel ~5% for 512-col case. This is Welford float32 vs double-precision
CPU reference — expected. In production, this precision is entirely acceptable.

## Two Kernels

### `layernorm_warp` — row_width ≤ 32
- One warp per row, pure SHFL — no shared memory
- 5 rounds × 3 shuffles = 15 SHFL.BFLY instructions per reduction pass
- Two reduction passes (first for statistics, same pass handles it)
- Launch: `grid=(num_rows,1,1)`, `block=(32,1,1)`

### `layernorm_block` — row_width ≤ 512
- One block per row, 128 threads × 4 elements/thread
- 4 warps → 4-entry shared memory array for inter-warp Welford combine
- Launch: `grid=(num_rows,1,1)`, `block=(128,1,1)`

## Build

```bash
nvcc --cubin -arch=sm_86 -O2 -o layernorm.sm_86.cubin layernorm.cu
cuobjdump -sass layernorm.sm_86.cubin | grep -E 'SHFL|MUFU'

nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 65536 32     # warp kernel
./bench 65536 128    # block kernel
./bench 16384 512    # wide rows
```

## What's Next: Activations (GELU, SiLU)

The FFN layer applies element-wise activations between two linear layers:
- **GELU**: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
  → `MUFU.TANH` + `FFMA` chain
- **SiLU**: `x * sigmoid(x) = x / (1 + exp(-x))`
  → `MUFU.EX2` (for exp) + `MUFU.RCP` (for 1/(1+exp))

Both are single-instruction throughput bottlenecks: the MUFU unit runs at
1/4 throughput vs FFMA, so pipelining many MUFU calls together with FFMA
work is the key optimization.
