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

| Kernel | 4096³ Time | TOPS | % INT8 Peak | vs Naive |
|--------|-----------|------|-------------|----------|
| Naive (global loads) | 12.6 ms | 10,897 | 1.6% | 1.00× |
| **Tiled 64×64 (smem)** | **9.1 ms** | **15,145** | **2.2%** | **1.39×** |
| Register-blocked 128×128 | 10.8 ms | 12,760 | 1.8% | 1.17× |

FP16 HGEMM reference: 7,853 GFLOPS at 4096³. Tiled INT8 is 1.93× faster.

**Why tiled 64×64 beats register-blocked 128×128:** The 128×128 kernel's inner loop has 16 mma_sync calls per K-step (vs 4 for 64×64). This long IMMA chain reduces warp switching frequency. With only 8 warps/SM (4 warps × 2 blocks), the scheduler can't fully hide Tensor Core pipeline latency (S08 stalls). The 64×64 version's shorter inner loop allows faster warp interleaving — same lesson as the cp.async and Bc=128 Flash Attention experiments.

**Why only 2.2% of INT8 peak?** Without shared memory, the naive kernel is fully bandwidth-bound. Tiling reduces redundant global loads 4× (from ~8 GB to ~2 GB at 4096³), but 2 GB at 608 GB/s still takes ~3.3 ms — far from the 0.2 ms compute minimum. Larger tiles (128×128) hurt because the longer inner loop can't be hidden by 8 warps. The path to higher utilization requires either more warps per SM or a fundamentally different approach (persistent kernels, software pipelining).

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
