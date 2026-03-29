# Phase 2: Activations — MUFU.TANH / MUFU.EX2 / MUFU.RCP

## Key SASS Instructions

| Kernel | SASS | Hardware unit |
|---|---|---|
| `gelu_kernel` | `MUFU.TANH` | Hardware tanh (requires `--use_fast_math`) |
| `silu_kernel` | `MUFU.EX2` + `MUFU.RCP` | 2^x unit + reciprocal unit |
| `gelu_fast` | `MUFU.EX2` + `MUFU.RCP` | Same as SiLU |
| `relu_kernel` | `FMNMX` | Float min/max comparison |

All MUFU instructions share a single hardware execution unit that runs at
**1/4 the throughput of FFMA** — one MUFU per 4 cycles, vs 1 FFMA per cycle.

## The Surprising Result: All Activations Are Equally Fast

```
relu_kernel     0.340 ms    394.5 GB/s  FMNMX only
gelu_fast       0.341 ms    393.2 GB/s  MUFU.EX2 + MUFU.RCP
silu_kernel     0.341 ms    393.9 GB/s  MUFU.EX2 + MUFU.RCP
gelu_kernel     0.340 ms    394.5 GB/s  MUFU.TANH + FFMA
```

Despite MUFU running at 1/4 throughput, **all four activations hit the same ~394 GB/s**.
Why? Because element-wise activations on large arrays are **memory bandwidth bound**.

Peak bandwidth: 608 GB/s. Achieved: 394 GB/s (65% of peak).
Each element requires 1 load + 1 store = 8 bytes. The GPU finishes the MUFU
work in the cycles spent waiting for the next memory transaction.

**Lesson**: Don't micro-optimize activation compute for large standalone kernels.
The bottleneck is always the load/store, not the arithmetic. MUFU optimization
matters when activations are **fused** into GEMM/attention kernels — where
compute and memory are already balanced, and extra MUFU work becomes visible.

## MUFU.TANH — Requires `--use_fast_math`

`tanhf(x)` in standard CUDA compiles to a multi-instruction software approximation:
```sass
; Without --use_fast_math: software tanh via exp(2x):
FMUL  R0, R6, 2.0               ; 2x
MUFU.EX2 R0, R0                  ; exp(2x) via 2^(2x*log2e)
...                               ; several FFMA instructions
```

With `--use_fast_math`, tanhf maps to the single dedicated unit:
```sass
MUFU.TANH R6, R6                 ; one-cycle hardware tanh
```

The fast math version has ~1 ULP precision loss — acceptable for activations
in neural networks (training uses fp16 anyway, inference doesn't need exact tanh).

## GELU vs Fast-GELU vs SiLU

| Activation | Formula | MUFU instructions |
|---|---|---|
| GELU (exact) | `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` | 1× MUFU.TANH |
| Fast GELU | `x * sigmoid(1.702 * x)` | 1× MUFU.EX2 + 1× MUFU.RCP |
| SiLU/Swish | `x * sigmoid(x)` | 1× MUFU.EX2 + 1× MUFU.RCP |

In practice (from our benchmark): all three have identical throughput.
Choose based on the accuracy requirement of the model, not performance.

## Build

```bash
# --use_fast_math required for MUFU.TANH emission in gelu_kernel
nvcc --cubin -arch=sm_86 -O2 --use_fast_math -o activations.sm_86.cubin activations.cu

# Verify MUFU instructions
cuobjdump -sass activations.sm_86.cubin | grep MUFU
# → MUFU.TANH  (gelu_kernel)
# → MUFU.EX2   (silu_kernel, gelu_fast)
# → MUFU.RCP   (silu_kernel, gelu_fast)

# Bench
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 16777216
```

## What Comes Next: Phase 3 Flash Attention

We now have all the building blocks:
- **HMMA.16816.F32** (HGEMM) — for QK^T and AV matrix multiplies
- **SHFL.BFLY + MUFU.EX2 + MUFU.RCP** (softmax) — for score normalization
- **MUFU.RSQ** (layernorm) — for pre/post-attention normalization
- **MUFU.TANH / MUFU.EX2** (activations) — for FFN non-linearities

Flash Attention fuses the softmax directly into the QK^T HMMA loop:
instead of writing the full N×N score matrix to VRAM, we keep scores
in shared memory / registers and apply online softmax tile by tile.
This is a 10-20× memory bandwidth reduction for long sequences.
