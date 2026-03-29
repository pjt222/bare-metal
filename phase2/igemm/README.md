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
| Naive (global loads) | 12.5 ms | 10,980 | 1.6% | 1.00× |
| Tiled 64×64 (compiler) | 9.1 ms | 15,121 | 2.2% | 1.38× |
| Tiled 64×64 (hand-tuned S02) | 9.0 ms | 15,341 | 2.2% | 1.40× |
| Tiled 64×64 (aggressive S01) | 9.1 ms | 15,170 | 2.2% | 1.38× |
| Register-blocked 128×128 | 10.8 ms | 12,741 | 1.8% | 1.16× |
| Pipelined LDG (double-buffer) | 7.6 ms | 18,054 | 2.6% | 1.64× |
| **Pipelined cp.async (double-buffer)** | **6.6 ms** | **20,688** | **3.0%** | **1.88×** |

FP16 HGEMM reference: 7,853 GFLOPS at 4096³. Pipelined cp.async INT8 is 2.63× faster.

**Why tiled 64×64 beats register-blocked 128×128:** The 128×128 kernel's inner loop has 16 mma_sync calls per K-step (vs 4 for 64×64). This long IMMA chain reduces warp switching frequency. With only 8 warps/SM (4 warps × 2 blocks), the scheduler can't fully hide Tensor Core pipeline latency (S08 stalls). The 64×64 version's shorter inner loop allows faster warp interleaving — same lesson as the Bc=128 Flash Attention experiment.

**Why only 3.0% of INT8 peak?** The tiled kernel without pipelining was fully bandwidth-bound: 2 GB at 608 GB/s = ~3.3 ms floor vs 9.0 ms total. Software pipelining (double-buffered smem) overlaps LDG for tile N+1 with IMMA compute on tile N, reclaiming ~2.4 ms of pipeline bubble. The remaining gap to the 0.2 ms compute minimum is still dominated by DRAM bandwidth — the kernel achieves ~308 GB/s effective bandwidth (51% of 608 GB/s peak).

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

## Software Pipelining (Issue #5)

### Double-Buffered K-Loop

The tiled kernel's K-loop is fully sequential: load tile → sync → compute → sync. Software pipelining overlaps global loads for tile N+1 with IMMA compute on tile N using double-buffered shared memory:

```
smem_a[2][BM*BK] + smem_b[2][BK*BN] = 2 × (2 KB + 2 KB) = 8 KB total
```

Two variants tested:

1. **LDG (register prefetch)**: LDG tile N+1 → registers, IMMA on buf[N%2], sync, STS registers → buf[(N+1)%2], sync
2. **cp.async (LDGSTS)**: LDGSTS tile N+1 → buf[(N+1)%2] (async, bypasses registers), IMMA on buf[N%2], wait+sync

### Results (Issue #5)

| Variant | 4096³ Time | TOPS | vs Hand-tuned |
|---------|-----------|------|---------------|
| Hand-tuned baseline (S02) | 9.0 ms | 15,341 | — |
| **Pipelined LDG** | **7.6 ms** | **18,054** | **+17.7%** |
| **Pipelined cp.async** | **6.6 ms** | **20,688** | **+34.9%** |

Across sizes:

| Size | Hand-tuned | Pipelined LDG | cp.async | LDG gain | cp.async gain |
|------|-----------|---------------|----------|----------|---------------|
| 512³ | 3,913 | 6,596 | 7,524 | +68.6% | +92.3% |
| 1024³ | 9,740 | 13,253 | 14,495 | +36.1% | +48.8% |
| 4096³ | 15,341 | 18,054 | 20,688 | +17.7% | +34.9% |

### Why cp.async Helps Here (Contradicts Phase 5 Flash Attention Finding)

Phase 5 found cp.async 4-5% **slower** for Flash Attention at 8 warps/SM. The explanation was that warp interleaving already hid DRAM latency, so cp.async only added commit/wait overhead.

For IGEMM, cp.async is **+35% faster**. The difference: **compute phase length**.

- **Flash Attention**: 64 HMMA per tile → long compute phase → 8 warps can interleave enough to hide ~300-cycle DRAM latency
- **IGEMM**: 8 IMMA per tile (4 mma_sync × 2 K-steps) → short compute phase → 8 warps generate ~128 compute cycles total, insufficient to hide ~300-cycle DRAM latency

When the compute/load ratio is low, warp interleaving alone cannot fill the load latency gap. cp.async provides additional hiding by decoupling the load from the warp's instruction stream — the memory subsystem works independently while IMMA executes.

**Updated rule**: cp.async benefits scale inversely with compute/load ratio. Short inner loops (IGEMM) benefit; long inner loops (Flash Attention) don't.

### SASS Structure (cp.async variant)

```
Main loop body:
  8× LDGSTS.E [smem], [global]    ← issue all 8 async copies (non-blocking)
  __pipeline_commit()              ← mark end of async group
  16× IMMA.16816.S8.S8            ← compute on CURRENT buffer (overlaps with LDGSTS)
  __pipeline_wait_prior(0)         ← wait for all copies to complete
  BAR.SYNC                         ← ensure all threads see new smem data
```

### Files

- `igemm_pipelined.cu` — LDG register-prefetch double-buffer variant
- `igemm_pipelined_cpasync.cu` — cp.async (LDGSTS) double-buffer variant
- `igemm_pipelined.sm_86.cubin` — LDG variant binary
- `igemm_pipelined_cpasync.sm_86.cubin` — cp.async variant binary

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
