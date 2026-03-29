# Phase 2: HGEMM — Tensor Core Matrix Multiply

## Why HGEMM

Every transformer and diffusion model inference run uses Tensor Cores.
The HMMA instruction gives 8× more throughput than FFMA for the same power budget.

| Mode | Peak (RTX 3070 Ti) | Instruction |
|---|---|---|
| FP32 (FFMA) | 21.7 TFLOPS | `FFMA R, R, R, R` |
| FP16 Tensor Core | 174 TFLOPS | `HMMA.16816.F32 R, R, R, R` |

## Measured Results

| Matrix | GFLOPS | vs FP32 tiled |
|---|---|---|
| 512×512×512 | 5,438 | 5.3× |
| 4096×4096×4096 | 7,853 | 7.6× |

At 4096³ we get 4.5% of the 174 TFLOPS theoretical peak.
The gap is closed by: larger tiles, double-buffering, `LDG.E.128` loads — all achievable in SASS.

## The Key SASS Instruction

```sass
HMMA.16816.F32 R12, R8, R16, R12 ;
```

This single instruction makes a **warp-wide** 16×8×16 matrix multiply:
- `R8`  — A fragment: 8 half-precision values (rows of a 16×16 sub-matrix, distributed across 32 threads)
- `R16` — B fragment: 8 half-precision values
- `R12` — C accumulator in, D accumulator out: 8 FP32 values
- All 32 threads in the warp execute it simultaneously → 16×8×16 = 2048 FP16 MACs per instruction

On Ampere, the WMMA `mma.sync.aligned.m16n8k16` PTX instruction → 2 HMMA SASS instructions per warp per 16-step k-tile.

## The WMMA API → PTX → SASS Chain

```
C++ (WMMA API)         PTX                    SASS
────────────────       ─────────────────────  ──────────────────────────────
wmma::load_matrix_sync → ldmatrix.sync.m8n8  → LDSM (load shared matrix)
wmma::mma_sync         → mma.sync.aligned    → HMMA.16816.F32
wmma::store_matrix_sync→ stmatrix.sync       → STS + STG
```

## Fragment Register Layout (sm_86)

The WMMA API hides the register layout, but at SASS level each warp's threads
hold specific rows/columns of the matrix fragment. This is what makes HMMA different
from FFMA — the data must be in the exact register layout the hardware expects.

For `m16n16k16` on Ampere:
- **A fragment** (16×16, FP16, row-major): each of the 32 threads holds 8 FP16 values covering 2 rows × 4 columns of the 16×16 tile
- **B fragment** (16×16, FP16, row-major): each thread holds 8 FP16 values
- **C/D accumulator** (16×16, FP32): each thread holds 8 FP32 values

The compiler automatically handles this layout when you use the WMMA API.
To write HMMA in raw SASS, you'd need to replicate this exact layout — see
`cuobjdump -sass hgemm.sm_86.cubin` and trace which registers hold which elements.

## Correctness Note

FP16 is less precise than FP32 — the same GEMM computed in FP16 vs FP32 will differ:
- FP32: ~5 decimal digits of precision
- FP16: ~3 decimal digits of precision

For a 512×512×512 matrix multiply, each output element sums 512 products. Rounding errors
accumulate, giving `max_rel` errors in the thousands (near-zero reference elements).
The absolute error (`max_abs`) is the meaningful measure — we saw `2.42e-02` (0.024).

## Build

```bash
# Compile kernel cubin
nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu

# See HMMA instructions
cuobjdump -sass hgemm.sm_86.cubin | grep HMMA

# Produce editable .cuasm for SASS study
python3 ../../scripts/build.py disasm hgemm.sm_86.cubin

# Build bench
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common

# Correctness check (small matrix)
./bench 512 512 512

# Performance (larger matrices reveal the real throughput)
./bench 4096 4096 4096
```

## What's Next: Flash Attention

HGEMM is the core of Flash Attention. The key insight of Flash Attention:
- Standard attention: write full N×N score matrix to VRAM → bandwidth limited
- Flash Attention: tile Q,K,V and keep scores in shared memory / registers → no VRAM write

Our hand-tuned Flash Attention kernel (Phase 3) will use HMMA for the QK^T and AV
multiplications, with online softmax interleaved in the same kernel loop.

## Pitfalls

See `docs/troubleshooting.md`:
- `BenchTimer` destructor segfault when called after `cuCtxDestroy` — scope RAII objects carefully
- FP16 correctness: use tolerance 0.1 (not 1e-3) for `check_fp32` when comparing to FP32 reference
