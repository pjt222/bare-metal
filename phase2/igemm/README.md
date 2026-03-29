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
| Tiled 64×64 (compiler) | 9.1 ms | 15,078 | 2.2% | 1.39× |
| **Tiled 64×64 (hand-tuned S02)** | **9.0 ms** | **15,320** | **2.2%** | **1.41×** |
| Tiled 64×64 (aggressive S01) | 9.1 ms | 15,144 | 2.2% | 1.39× |
| Register-blocked 128×128 | 10.8 ms | 12,760 | 1.8% | 1.17× |

FP16 HGEMM reference: 7,853 GFLOPS at 4096³. Tiled INT8 is 1.93× faster.

**Why tiled 64×64 beats register-blocked 128×128:** The 128×128 kernel's inner loop has 16 mma_sync calls per K-step (vs 4 for 64×64). This long IMMA chain reduces warp switching frequency. With only 8 warps/SM (4 warps × 2 blocks), the scheduler can't fully hide Tensor Core pipeline latency (S08 stalls). The 64×64 version's shorter inner loop allows faster warp interleaving — same lesson as the cp.async and Bc=128 Flash Attention experiments.

**Why only 2.2% of INT8 peak?** Without shared memory, the naive kernel is fully bandwidth-bound. Tiling reduces redundant global loads 4× (from ~8 GB to ~2 GB at 4096³), but 2 GB at 608 GB/s still takes ~3.3 ms — far from the 0.2 ms compute minimum. Larger tiles (128×128) hurt because the longer inner loop can't be hidden by 8 warps. The path to higher utilization requires either more warps per SM or a fundamentally different approach (persistent kernels, software pipelining).

## CuAssembler Hand-Tuning (Issue #2)

### IMMA Pipeline Analysis

Disassembling `igemm_tiled.sm_86.cubin` to `.cuasm` reveals the IMMA stall pattern in the K-loop inner body:

| Pattern | Stall | Count | Cause |
|---------|-------|-------|-------|
| Same A-fragment, new B-fragment | S04 | 8× | Compiler-conservative operand availability |
| Different A-fragment (R2↔R36) | S01 | 8× | No conflict — pipeline accepts immediately |

**Key finding: IMMA uses S04/S01, NOT S08 like HMMA.**

The HMMA (FP16) Tensor Core pipeline has a hardware-fixed S08 minimum between consecutive instructions (Phase 5 finding). IMMA (INT8) behaves differently:
- S01 throughput is achievable (verified correct at 512³ and 4096³)
- The compiler's S04 is conservative — S02 is the optimal stall count
- The S04 pattern comes from register read port contention (two IMMAs reading the same A-fragment register), not from Tensor Core pipeline depth

### Hand-Tuning Results (4096³)

| Variant | Stall | Time (ms) | TOPS | vs Compiler |
|---------|-------|-----------|------|-------------|
| Compiler (baseline) | S04 | 9.12 | 15,078 | — |
| **Hand-tuned (best)** | **S02** | **8.97** | **15,320** | **+1.6%** |
| Aggressive | S01 | 9.08 | 15,144 | +0.4% |

- **S02 wins**: enough scheduler slack for warp interleaving, less waste than S04
- **S01 too aggressive at scale**: eliminates scheduling slack, causing warp contention at 4096³ (though correct and faster at small 512³)
- **Gains are modest** because the kernel is memory-bound: 2 GB global loads at 608 GB/s (~3.3 ms floor) vs 9.1 ms total

### Inner Loop Cycle Budget

The K-loop body (BK=32, 2 WMMA K-steps) contains:
- 112 instructions, 305 total stall cycles = **417 issue cycles per K-tile**
- 16 IMMA instructions contribute only 40 of those 305 stall cycles
- The B-fragment assembly chain (LDS.U8 → IMAD → PRMT) dominates non-IMMA stalls
- For K=4096: 128 tiles × 417 cycles = ~53K cycles (~34 µs) for compute alone

### Files

- `igemm_tiled.sm_86.cuasm` — CuAssembler disassembly (annotated inner loop)
- `igemm_tiled_handtuned.cuasm` — S04→S02 on IMMA instructions
- `igemm_tiled_aggressive.cuasm` — S04→S01 on IMMA instructions
- `igemm_tiled_handtuned.sm_86.cubin` — Best hand-tuned binary
- `igemm_tiled_aggressive.sm_86.cubin` — Aggressive hand-tuned binary

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
