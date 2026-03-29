# Phase 2: IGEMM — INT8 Tensor Core Matrix Multiply

## Why INT8

INT8 Tensor Cores deliver 4× the throughput of FP16 on the same silicon:

| Precision | Peak Throughput | SASS Instruction |
|-----------|----------------|------------------|
| FP32 FFMA | 21.7 TFLOPS | `FFMA` |
| FP16 Tensor | 174 TFLOPS | `HMMA.16816.F32` |
| INT8 Tensor | 696 TOPS | `IMMA.16816.S8.S8` |

INT8 is the inference workhorse: weights and activations quantized to 8-bit, accumulated in INT32, then dequantized to FP32 for the next layer.

## Measured Results (RTX 3070 Ti Laptop, sm_86)

| Matrix Size | Time | TOPS | % of INT8 Peak | vs FP16 HGEMM |
|-------------|------|------|-----------------|---------------|
| 512³ | 0.058 ms | 4,638 | 0.7% | — |
| 4096³ | 12.7 ms | 10,828 | 1.6% | 1.38× |

FP16 HGEMM reference: 7,853 GFLOPS at 4096³ (4.5% of FP16 peak).

**Why only 1.6% of INT8 peak?** Same reason as the naive HGEMM — no shared memory tiling. The kernel loads A and B tiles directly from global memory via `wmma::load_matrix_sync`. At 4096³, memory traffic dominates. INT8 helps by reading half the bytes per element (1 vs 2), giving the 1.38× speedup. A tiled version with shared memory would achieve much higher utilization.

## The Key SASS Instruction

```sass
IMMA.16816.S8.S8 Rd, Ra.ROW, Rb.COL, Rc
```

- **Shape**: 16×8×16 at the hardware level (two instructions cover the 16×16×16 WMMA tile)
- **Operands**: Ra holds INT8 A-matrix rows, Rb holds INT8 B-matrix columns
- **Accumulation**: Rd/Rc are INT32 (4× wider than inputs)
- **Count**: 10 IMMA instructions per kernel (K-loop body + boundary handling)

## The WMMA API → PTX → SASS Chain

```
CUDA C++:  wmma::mma_sync(acc_int32, a_int8, b_int8, acc_int32)
    ↓
PTX:       mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
    ↓
SASS:      IMMA.16816.S8.S8  (×2 per PTX instruction → ×4 per WMMA call)
```

## Dequantization Path

After the K-loop, INT32 accumulator is converted to FP32:

```sass
I2FP.F32.S32 R16, R10    ; INT32 → FP32 (×8 per thread = 16×16 tile)
FMUL R5, R9, R6          ; multiply by dequant_scale (×8 per thread)
```

8 I2FP + 9 FMUL instructions handle the dequantization for the 16×16 output tile.

## Quantization

Symmetric per-tensor quantization (simplest correct approach):

```
scale = max(|tensor|) / 127
quantized = clamp(round(value / scale), -128, 127)
dequantized output: C_fp32 = C_int32 * scale_a * scale_b
```

For `fill_random` data in [-1, 1]: `scale ≈ 1/127 ≈ 0.00787`.

## Correctness

Tested against CPU FP32 SGEMM reference:

| Size | max_abs | max_rel | Tolerance | Result |
|------|---------|---------|-----------|--------|
| 512³ | 1.90e-01 | 4.45e+04 | abs=0.5, rel=0.1 | PASS |

The max_abs of 0.19 is the quantization error budget — each input value is rounded to the nearest multiple of `scale ≈ 0.008`, and these rounding errors accumulate over K=512 multiply-adds. The high max_rel comes from near-zero reference values where even small absolute error produces large relative error (handled by AND logic in `check_fp32`).

## Building

```bash
# Kernel cubin
nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu

# Benchmark binary
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common

# Run correctness
./bench 512 512 512

# Run performance
./bench 4096 4096 4096

# Inspect SASS
cuobjdump -sass igemm.sm_86.cubin | grep IMMA
cuobjdump -sass igemm.sm_86.cubin | grep I2F
```
