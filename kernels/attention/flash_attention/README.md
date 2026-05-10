# kernels/attention/flash_attention — Online softmax without N² memory

## Overview

Standard scaled dot-product attention computes:

```
O = softmax(Q K^T / sqrt(d)) V
```

The naive implementation writes the full `[seq_len × seq_len]` score matrix to HBM. Flash Attention avoids this using an **online softmax recurrence** that maintains running statistics and never materializes the N×N matrix.

## Memory Comparison

| Implementation | Score matrix | Total HBM | seq_len=1024 |
|---|---|---|---|
| Naive | O(N²) written to VRAM | Q+K+V + N² | 48 MB + 4 MB = 52 MB |
| Flash Attention | **never written** | Q+K+V + O | 48 MB + 16 MB = 64 MB Q/K/V + O |
| Ratio | — | ~5× less | grows with N |

The benefit grows quadratically with sequence length — the key property for long-context transformers.

---

## Online Softmax Recurrence

For each query row `q`:

```
State: running_max m = -inf, running_sum l = 0, output o[0..d-1] = 0

For each KV tile (K_j, V_j):
  1. s[k] = scale * Q[q] · K_j[k]           # dot product per KV position
  2. tile_max = max(s[k])
  3. new_max  = max(m, tile_max)
  4. rescale  = exp(m - new_max)             # shrink old accumulators
     o = o * rescale
     l = l * rescale
  5. For each k in tile:
     w = exp(s[k] - new_max)                # MUFU.EX2
     l += w
     o += w * V_j[k]                        # FFMA × d/warp_size
  6. m = new_max

Final: O[q] = o / l                         # MUFU.RCP
```

This is **mathematically identical** to computing the full softmax over all N KV positions and then multiplying by V. The rescaling in steps 4–5 ensures no numerical overflow regardless of input magnitude.

---

## SASS Instructions Produced

```
cuobjdump -sass flash_attn.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX|FFMA'
```

| Instruction | Count | Source |
|---|---|---|
| `SHFL.BFLY` | 320 | Warp dot product reduction (5 offsets × 32 KV × 2 kernels) |
| `MUFU.EX2` | 66 | `exp2f(x * LOG2E)` — attention weights + rescale factors |
| `MUFU.RCP` | 10 | `__frcp_rn(running_sum)` — final normalization |
| `FFMA` | 276 | `output_reg += weight * V_tile[kv][...]` |

**SHFL.BFLY pattern** for each Q·K dot product:
```
SHFL.BFLY PT, Rx, Ry, 0x10, 0x1f   ; offset 16 (lanes 0↔16, 1↔17, ...)
SHFL.BFLY PT, Rx, Ry, 0x8,  0x1f   ; offset  8
SHFL.BFLY PT, Rx, Ry, 0x4,  0x1f   ; offset  4
SHFL.BFLY PT, Rx, Ry, 0x2,  0x1f   ; offset  2
SHFL.BFLY PT, Rx, Ry, 0x1,  0x1f   ; offset  1
```
After these 5 instructions all 32 lanes hold the same dot product value — no shared memory needed.

---

## Kernel Design

### `flash_attn_1warp`

| Parameter | Value |
|---|---|
| Head dimension `D_HEAD` | 64 |
| KV tile size `BLOCK_KV` | 32 (= warp size) |
| Elements per thread | 2 (= D_HEAD / WARP_SIZE) |
| Grid | `(seq_len, 1, 1)` |
| Block | `(32, 1, 1)` — one warp per query row |
| Shared memory | 2 × 32 × 64 × 4 = 16 KB |

Each warp:
- Holds Q row in registers (2 floats per lane)
- Loads K tile + V tile into 16 KB shared memory
- Computes 32 dot products via SHFL.BFLY reduction
- Applies online softmax update
- Accumulates weighted V into 2-float-per-lane output registers
- Stores final normalized output

**Coalesced loading pattern**: thread `lane` loads indices `lane, lane+32, lane+64, ...`
giving 32 consecutive threads accessing consecutive global memory → 128-byte coalesced transactions.

### `flash_attn_multihead`

Same algorithm, parameterized by `(batch, head, q_idx)` via 3D grid:

```
Grid:  (seq_len, num_heads, batch_size)
Block: (32, 1, 1)
```

---

## Benchmark Results (RTX 3070 Ti Laptop, sm_86)

### Phase 3a — `flash_attn_1warp` (scalar, 1 warp per query row, BLOCK_KV=32)

| Config | seq_len | Measured BW* | Ideal BW** | ms |
|---|---|---|---|---|
| batch=1, heads=1 | 512 | 12.6 GB/s | 1.5 GB/s | 0.35 |
| batch=16, heads=16 | 512 | 17.8 GB/s | 2.1 GB/s | 64 |
| batch=8, heads=8 | 1024 | 20.8 GB/s | 1.3 GB/s | 53 |
| batch=4, heads=8 | 2048 | 20.8 GB/s | 0.6 GB/s | 105 |

### Phase 3b — `flash_attn_4warp` (4 warps share KV tile, BLOCK_KV=64)

| Config | seq_len | Measured BW | Ideal BW | ms | Speedup vs v1 |
|---|---|---|---|---|---|
| batch=8, heads=8 | 1024 | 30.3 GB/s | 3.6 GB/s | 19 | **2.8×** |
| batch=4, heads=8 | 2048 | 29.4 GB/s | 1.8 GB/s | 38 | **2.8×** |

SASS instruction counts (flash_wmma.sm_86.cubin):
- `SHFL.BFLY`: 321  — same butterfly pattern, BLOCK_KV=64 scores × 5 offsets
- `MUFU.EX2`:   65  — exp2 for 64 weights + 1 rescale per tile
- `MUFU.RCP`:    5  — final normalization
- `FFMA`:      264  — weighted V accumulation

\* Measured BW counts K and V reads × (seq_len/BLOCK_KV) iterations from global memory.
\*\* Ideal BW assumes K/V resident in L2 cache — only counts one pass through Q/K/V/O.

**Why so far from the 608 GB/s bandwidth ceiling?**

This kernel re-reads K and V tiles from **global memory** on every KV iteration. For `seq_len=1024`, that is `32` passes over the K and V tensors. The L2 cache (4 MB) is too small to hold the K/V tensors for multi-head workloads (16 heads × 1024 × 64 × 4 = 4 MB per tensor), so most L2 cache misses to DRAM.

The fix (next steps):
1. **Larger `BLOCK_KV`**: reduces the number of K/V load passes — quadratic improvement since more queries also reuse each K/V load
2. **WMMA Tensor Cores**: replace scalar dot products with HMMA.16816 — the QK^T computation is a matrix-matrix multiply, not just dot products
3. ~~**LDGSTS pipelining**~~: Implemented in Phase 5; measured 4-5% slower — warp interleaving at 8 warps already hides DRAM latency for Flash Attention's long compute phase (64 HMMA/tile). cp.async benefits only short compute phases (see IGEMM pipelining, +35%)

---

## Correctness

Tested against CPU numerically stable naive attention for all configurations:

| seq_len | max_abs | max_rel | Result |
|---|---|---|---|
| 512 | 1.27e-07 | 5.18e-03 | ✅ PASS |
| 1024 | 1.71e-07 | 2.16e-02 | ✅ PASS |
| 2048 | 1.70e-07 | 1.04e-01 | ✅ PASS |

The `max_rel` increases with `seq_len` because the online softmax has more accumulation rounding, and near-zero output elements produce high relative errors (see correctness check section in troubleshooting.md — AND logic: only fails if both absolute and relative are large). `max_abs` stays below 2e-7 across all sizes.

---

## Next: Flash Attention with Tensor Cores

The current scalar kernel uses warp dot products for Q·K scores. To exploit Tensor Cores:

```
flash_attn_wmma.cu:
  - Q, K, V as FP16
  - Block: 128 threads = 4 warps
  - Br = 64 (query tile per block, each warp handles 16 rows)
  - Bc = 64 (KV tile)
  - QK^T via WMMA: [16×64] @ [64×64]^T = [16×64] using 4 × HMMA.16816.F32
  - Online softmax: same SHFL.BFLY, MUFU.EX2 recurrence
  - PV via WMMA: [16×64] @ [64×64] = [16×64] using 4 × HMMA.16816.F32
  - Larger Br → more queries reuse each K/V tile → better occupancy
```

Expected speedup: 8× from Tensor Cores for the matmul steps + better K/V tile reuse.

---

## Building

```bash
# Kernel cubin (for SASS inspection)
nvcc --cubin -arch=sm_86 -O2 -o flash_attn.sm_86.cubin flash_attn.cu

# Benchmark binary
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common

# Run
./bench 512                    # single-head, seq_len=512
./bench 1024 8 8               # batch=8, heads=8, seq_len=1024
./bench 2048 4 8               # batch=4, heads=8, seq_len=2048

# Inspect SASS
cuobjdump -sass flash_attn.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX|FFMA'
```

## Persistent Kernel Grid (Issue #3)

A persistent-grid variant (`flash_attn_persistent.cu`) launches `num_sms × 2` blocks that loop over work tiles via `atomicAdd`. Each block grabs tiles and processes them until all tiles are done.

### Results

**batch=1, heads=8 (small batch — issue target):**

| seq | Tiles | br16 (ms) | Persistent (ms) | Speedup |
|-----|-------|-----------|-----------------|---------|
| 256 | 32 | 0.055 | 0.085 | 0.65× |
| 512 | 64 | 0.159 | 0.161 | 0.98× |
| 1024 | 128 | 0.579 | 0.580 | 1.00× |

**batch=8, heads=8 (many tiles):**

| seq | Tiles | br16 (ms) | Persistent (ms) | Speedup |
|-----|-------|-----------|-----------------|---------|
| 256 | 256 | 0.246 | 0.249 | 0.99× |
| 512 | 512 | 0.938 | 0.849 | **1.10×** |
| 1024 | 1024 | 3.108 | 2.819 | **1.10×** |

### Analysis

The persistent kernel helps at **moderate-to-large tile counts** (512+), NOT at the small-batch regime the issue originally targeted:

- **When tiles << grid (32 tiles, 92 blocks):** 60 blocks launch, allocate 48 KB smem, then immediately exit. This wastes SM resources and makes things worse (0.65×).
- **When tiles ≈ grid (128 tiles, 92 blocks):** roughly tied — each block processes ~1.4 tiles, minimal tail effect.
- **When tiles >> grid (512-1024 tiles, 92 blocks):** +10% improvement. The standard grid's last "wave" underutilizes SMs (e.g., 1024 mod 92 = 12 blocks on 46 SMs). The persistent grid eliminates this tail effect.

The benefit is **tail-wave elimination**, not SM utilization improvement at small batch.

---

## Phase 3c — Br=16 HMMA (Best Result)

The scalar kernels above are succeeded by `flash_attn_br16_regpv.cu`, which uses HMMA.16816.F32 Tensor Cores with Br=16, Bc=64:

| Config | Time | GFLOPS | vs Scalar |
|--------|------|--------|-----------|
| seq=1024, batch=8, heads=8 | **2.81 ms** | **6,112** | **18.9×** |
| seq=1024, batch=8, heads=8 (Bc=128) | 3.37 ms | 5,098 | 15.7× |

SASS: 64 HMMA per block (32 for QK^T + 32 for PV), SHFL.BFLY for online softmax reduction, MUFU.EX2 for attention weights.

Bc=128 crosses the 50 KB smem cliff → 1 block/SM occupancy regression. Bc=64 stays under cliff → 2 blocks/SM optimal. See [`docs/gpu_reflections.md`](../docs/gpu_reflections.md) for full analysis.

---

## Phase 3d — flash_attn_br16_v2 (Current Canonical, Issue #29)

The `flash_attn_br16_v2.cu` kernel is the result of a three-stage refactor of
`flash_attn_br16_regpv` (issue #29, 2026-05-07). It is now the canonical
high-performance Flash Attention kernel.

### Performance (RTX 3070 Ti, sm_86)

| Config | Baseline (regpv) | v2 | Speedup |
|--------|------------------|-----|---------|
| seq=512, batch=16, heads=16 | 2.80 ms | **1.76 ms** | **1.59×** |
| seq=1024, batch=8, heads=8 | 2.45 ms | **1.75 ms** | **1.40×** |
| seq=2048, batch=4, heads=8 | 4.87 ms | **3.46 ms** | **1.41×** |
| seq=4096, batch=2, heads=8 | 10.12 ms | **6.86 ms** | **1.47×** |

GFLOPS plateau: ~10 TFLOPS effective (5.7% of FP16 Tensor Core peak 174 TFLOPS).

### What Changed

1. **Lean per-thread softmax state**: `running_max[16]` + `running_sum[16]`
   replaced with 4 floats/thread (`my_max_lo/hi`, `my_sum_lo/hi`). Saves ~24
   regs/thread by eliminating broadcast-identical redundancy across the 32
   lanes of a warp.

2. **Q register cache**: query tile fragments hoisted out of the KV-base loop.
   Compiler did not auto-hoist them (verified via SASS). Saves 16x redundant
   L2 reads at seq=1024.

3. **smem_work elimination**: the 16 KB FP32 round-trip buffer between Phase B
   (QK^T) and Phase C (softmax) is replaced by on-fragment `__shfl_xor_sync`
   reductions within each WMMA row group. FP16 weights write directly to a
   small 8 KB smem region for Phase D PV; final pv_accum writes direct to
   global O. **LDS+STS instruction count: 238 → 30 in cubin.**

### SASS

```bash
cuobjdump -sass flash_br16_v2.sm_86.cubin | grep -E 'HMMA|LDSM|SHFL'
```

| Instruction | Count |
|---|---|
| HMMA | 64 (32 QK^T + 32 PV per kv iter, unchanged) |
| LDSM (ldmatrix) | 48 (16 K + 16 W + 16 V, unchanged) |
| LDS+STS | 30 (was 238 in regpv baseline) |
| SHFL | added: intra-group reductions for max/sum |

### Build

```bash
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2.sm_86.cubin flash_attn_br16_v2.cu
nvcc -arch=sm_86 -O2 -o bench_br16_regpv_pad bench_br16_regpv_pad.cu \
     -lcuda -I../../kernels/_common
./bench_br16_regpv_pad 1024 8 8
```

See [`docs/gpu_reflections.md`](../../docs/gpu_reflections.md) Observation P
for the full optimization narrative, and
[`docs/fragment_shfl_reductions.md`](../../docs/fragment_shfl_reductions.md)
for the reusable pattern documentation.
