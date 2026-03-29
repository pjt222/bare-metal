# CONTINUE_HERE.md — Bare-Metal GPU Project

*Created after Phase 5 optimization session. Read this first at the start of the next session.*

---

## Where We Are

**All of Phase 5 is complete.** Five optimization hypotheses from `docs/gpu_reflections.md`
were implemented and benchmarked. Each one taught something important.

### Quick Status Table

| Phase | Status | Highlight |
|-------|--------|-----------|
| 0 | ✅ Done | Environment, CuAssembler, nvcc, WSL |
| 1 | ✅ Done | Vector add: FADD→FMUL in SASS |
| 2 | ✅ Done | SGEMM, HGEMM, Softmax, LayerNorm, Activations |
| 3 | ✅ Done | Flash Attention: scalar → 4-warp → Br=16 HMMA (19× speedup) |
| 4 | ✅ Done | Diffusion primitives: Conv2d, GroupNorm, ResBlock, Cross-Attn |
| 5 | ✅ Done | 5 optimization experiments, all documented with postmortems |

---

## Phase 5 Findings — What We Learned

### ✅ Step 1 — im2col + WMMA Conv2d
**24× speedup over direct conv2d.** The definitive win of this project.
- File: `phase4/conv2d/conv2d_im2col.cu`
- SASS: LDGSTS.E.BYPASS.128 + HMMA.16816.F32 confirmed
- GFLOPS: 9,355 GEMM alone / 7,379 combined

### ✅ Step 2 — cp.async Double-Buffering (cross-attn + self-attn)
**Mixed/negative results — warp interleaving already hides latency.**
- Cross-attn pipelined: 1.54× faster at 16×16 (cold L2, 2 KV iters), SLOWER at larger sizes
- Self-attn pipelined: consistently 4–5% SLOWER at all seq_len
- Root cause: 2 blocks/SM × 4 warps = 8 warps. When block 0's warps stall on DRAM,
  block 1's warps execute HMMA. cp.async adds commit/wait overhead with no net gain.
- Files: `phase3/flash_attention/flash_attn_br16_pipeline.cu`, `phase4/cross_attention/cross_attn_pipelined.cu`

### ✅ Step 3 — Bc=128 Flash Attention
**17–20% SLOWER.** The occupancy cliff killed it.
- 80 KB smem → 1 block/SM → 4 warps (vs 2 blocks → 8 warps at 48 KB)
- 128 HMMA/tile confirmed in SASS (2× the 64 of Bc=64) — arithmetic worked, occupancy didn't
- **The 48 KB / 64 KB boundary is load-bearing on GA104** (128 KB per SM → 2× 64 KB)
- File: `phase3/flash_attention/flash_attn_br16_bc128.cu`

### ✅ Step 4 — CuAssembler HMMA Stall Analysis
**S08 between HMMAs is hardware-constrained, not conservative.**
- Disassembled flash_br16 and conv2d_im2col via CuAssembler
- S08 reflects the Ampere TC pipeline depth (1 HMMA per 8 cycles per warp)
- Cannot reduce to S01 between consecutive HMMAs — structural hazard on TC unit
- The S01 pairs in the WMMA GEMM (LDSM-interleaved) use independent register groups
  and are already the tightest schedulable arrangement
- CuAssembler editing not applicable to these kernels; FFMA-heavy kernels are obsolete
- `.cuasm` files: `phase3/flash_attention/flash_br16.cuasm`, `phase4/conv2d/wmma_gemm.cuasm`

### ✅ Step 5 — Implicit GEMM (no col buffer)
**1.07–1.89× speedup over explicit im2col+GEMM. Bit-perfect.**
- Eliminates 23–47 MB col buffer DRAM traffic by computing indices on-the-fly
- Key insight: precomputed coordinate tables in shared memory (+960 bytes)
  reduce per-element arithmetic from 6 integer divisions to 2 adds + 2 compares
- Speedup scales with col buffer size vs L2 (4 MB):

  | Config | col buf | Speedup |
  |--------|---------|---------|
  | 64×64, Cin=Cout=320   | 23.6 MB | 1.07× |
  | 32×32, Cin=Cout=640   | 11.8 MB | 1.77× |
  | 128×128, Cin=Cout=160 | 47.2 MB | **1.89×** |
  | 64×64, Cin=Cout=64    |  4.7 MB | 1.34× |

- Files: `phase4/conv2d/conv2d_implicit_gemm.cu`, `phase4/conv2d/bench_implicit_gemm.cu`

---

## The Four Laws of Making GA104 Happy (Updated from Three)

1. **Feed the Tensor Cores continuously.** Overlap data loading (cp.async) with
   HMMA. But check first — 8+ active warps may already do this for free.

2. **Read each byte of DRAM exactly once.** Every re-read multiplies effective
   traffic. im2col converts 9× re-reads into 1×; implicit GEMM eliminates the
   col buffer round-trip entirely.

3. **Fill the warp schedulers.** 32 warps/SM is ideal; 8 is sufficient for latency
   hiding. When you can't reach 8, you have a structural problem — fix smem or grid.

4. **Never cross the 64 KB smem cliff.** On GA104 (128 KB/SM):
   - ≤ 64 KB → 2 blocks/SM → 8 warps → warp interleaving works
   - > 64 KB → 1 block/SM → 4 warps → DRAM stalls exposed, performance regresses

---

## Key Pitfalls Discovered This Session

```
Bc=128 flash attn: 80 KB smem → 1 block/SM → 4 warps → 17-20% SLOWER than Bc=64
HMMA S08 stall: hardware TC pipeline constraint, cannot reduce to S01 (structural hazard)
Implicit GEMM naive: 6 divs/element × 1024 elements = 6144 divs/K-tile → 94× overhead vs precomputed
Implicit GEMM with tables: 64 K-dim divs + 256 M-dim divs (amortized) → fast enough to beat explicit
cp.async self-attn: always slower than baseline when 2 blocks/SM are present (warp interleaving wins)
```

---

## Next Steps (Prioritized)

### 🎯 Top Priority — Split-Q Flash Attention
**Goal:** 1.5–2× speedup at scale by reusing K/V in L2 across blocks.

**Current grid:** `(q_tiles, heads, batch)` — each block reads ALL K/V tiles.
At seq=1024 with 16 q-tiles, the same K/V is loaded 16× from DRAM (L2 evicted between waves).

**Proposed grid:** `(kv_tiles, heads, batch)` — each block reads ONE K/V tile,
accumulates partial softmax sums over all Q positions.

This is the "split-K" variant of Flash Attention:
```
Block i processes: KV_tile[i] × ALL Q tiles
  → loads K/V once per block (stays in L2 while Q tiles iterate)
  → accumulates partial {m_i, l_i, o_i} for each Q row
Final reduction: merge partial {m_i, l_i, o_i} across kv_tiles dimension
```

**Complexity:** requires a second-pass reduction kernel (or atomic accumulation).
The partial softmax merging formula:
```
m_new = max(m_a, m_b)
l_new = exp(m_a - m_new) * l_a + exp(m_b - m_new) * l_b
o_new = (exp(m_a - m_new) * l_a * o_a + exp(m_b - m_new) * l_b * o_b) / l_new
```

**Implementation plan:**
1. `flash_attn_split_q.cu` — split-Q kernel: outputs partial {m, l, o} per KV tile
2. `flash_attn_split_q_reduce.cu` — reduction kernel: merges partials across kv_tiles
3. `bench_split_q.cu` — compare vs flash_attn_br16 at seq=256/512/1024/2048

**Expected gain:** L2 reuse of K/V across concurrent blocks → fewer DRAM reads at large batch×heads.

---

### 🔬 Secondary — INT8 IMMA Path
Explore INT8 Tensor Core (`IMMA.16816`) for ~4× throughput over FP16.

- Requires quantization: FP16 → INT8 per-tensor or per-channel scale+zero-point
- IMMA accumulates in INT32, requires rescaling to FP32 output
- Use case: inference (not training) — activations bounded, weights quantizable
- Expected: 4× GFLOPS (from 174 TFLOPS FP16 → 696 TOPS INT8)

**Not yet started.** Complexity: quantization correctness + scale factor management.

---

### 🧪 Exploratory — Persistent Kernel Grid
For flash attention at small batch sizes (batch=1, heads=8):
- Many SMs are underutilized (only 8 blocks = 8/48 SM utilization at seq=256)
- Persistent kernel: fewer blocks, each block processes multiple KV tiles in sequence
- Reduces kernel launch overhead and improves SM utilization
- Pairs well with split-Q (reduces number of reduction passes needed)

---

## File Structure — New Files This Session

```
phase3/flash_attention/
  flash_attn_br16_pipeline.cu   ← cp.async self-attn (4-5% SLOWER — documented)
  bench_pipeline.cu              ← baseline vs pipeline comparison
  flash_attn_br16_bc128.cu      ← Bc=128 kernel (17-20% SLOWER — documents occupancy cliff)
  bench_bc128.cu                 ← Bc=64 vs Bc=128 comparison
  flash_br16.cuasm               ← CuAssembler disassembly (S08 analysis done)
  flash_br16_pipeline.sm_86.cubin
  flash_br16_bc128.sm_86.cubin

phase4/conv2d/
  conv2d_implicit_gemm.cu       ← implicit GEMM (no col buffer, 1.07-1.89× faster)
  bench_implicit_gemm.cu         ← explicit vs implicit comparison across 4 configs
  wmma_gemm.cuasm                ← CuAssembler disassembly (inner loop already tight)
  conv2d_direct.cuasm            ← CuAssembler disassembly (FFMA loop — obsolete kernel)
  conv2d_implicit_gemm.sm_86.cubin

docs/
  gpu_reflections.md             ← Updated with Steps 2-5 postmortems + 4th Law
```

---

## Build Commands Cheat Sheet

```bash
# WSL prefix for all CUDA commands:
wsl -e bash -c 'export PATH=/usr/local/cuda/bin:$PATH && cd /mnt/d/dev/p/bare-metal/... && ...'

# Compile kernel to cubin:
nvcc --cubin -arch=sm_86 -O2 -o kernel.sm_86.cubin kernel.cu

# Compile benchmark executable:
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common

# Count HMMA instructions:
cuobjdump -sass kernel.sm_86.cubin | grep HMMA | wc -l

# Disassemble to cuasm (for CuAssembler analysis):
cd /mnt/d/dev/p/bare-metal
PATH=/usr/local/cuda/bin:$PATH python3 -c "
import sys
sys.path.insert(0, 'tools/CuAssembler')
from CuAsm.CubinFile import CubinFile
CubinFile('path/to/kernel.cubin').saveAsCuAsm('kernel.cuasm')
"
```

---

## Benchmark Results Summary (All Kernels)

### Flash Attention (seq=1024, batch=8, heads=8, D=64)
| Kernel | Time | GFLOPS | Notes |
|--------|------|--------|-------|
| flash_attn scalar (1-warp) | 53 ms | ~330 | Baseline |
| flash_attn_4warp | 19 ms | ~920 | 2.8× |
| flash_attn_br16 (Bc=64) | 2.81 ms | 6,112 | **19× — current best** |
| flash_attn_br16_pipeline | 2.96 ms | 5,795 | 4-5% slower (cp.async overhead) |
| flash_attn_bc128 (Bc=128) | 3.37 ms | 5,098 | 17-20% slower (occupancy cliff) |

### Conv2d (N=1, H=W=64, Cin=Cout=320, 3×3 same pad)
| Kernel | Time | GFLOPS | Notes |
|--------|------|--------|-------|
| conv2d_nhwc (direct) | ~25 ms | ~300 | Baseline |
| conv2d_im2col (explicit) | 1.21 ms | 6,262 | 24× over direct |
| conv2d_implicit_gemm | 1.13 ms | 6,687 | **1.07× over explicit = 25× over direct** |

### Cross-Attention (SD 64×64 spatial, seq_q=4096, seq_kv=77, heads=8)
| Kernel | Time | GFLOPS | Notes |
|--------|------|--------|-------|
| cross_attn_br16 (baseline) | 0.246 ms | 2,624 | Current best for large spatial |
| cross_attn_pipelined | 0.293 ms | 2,202 | 19% slower at this config |

---

## Reading Order for Next Session

1. **This file** (you're reading it)
2. `docs/gpu_reflections.md` — full story, all 5 postmortems, 4 laws
3. `phase3/flash_attention/flash_attn_br16.cu` — the best attention kernel, starting point for split-Q
4. The Flash Attention paper (Dao et al. 2022) — Section 3.1 describes the split-K formulation
   that split-Q is derived from

---

*Last updated: end of Phase 5 optimization session, 2026-03-27.*
*All experiments ran on RTX 3070 Ti Laptop (GA104, sm_86, CUDA 12.8, WSL Ubuntu).*
