# Chapter 06 — The Four Laws of Making GA104 Happy

> Distilled from 18 empirical observations across 5 phases of kernel work.
> Source: `docs/gpu_reflections.md` and `CLAUDE.md`.

After ~80 kernel variants, several thousand benchmark runs, and many failed
optimizations, four principles dominate. They are not abstract — each was
arrived at through a specific failure that produced the same lesson.

This chapter states each law, then walks through one or two case studies
that illustrate it. Read this chapter top-down: the laws first, then the
evidence.

## The Laws

1. **Feed Tensor Cores continuously** — overlap loads with HMMA/IMMA. The
   benefit of cp.async depends on warp count, not on the operation itself.

2. **Read each byte of DRAM exactly once** — the L2 cache is too small for
   serious workloads (4 MB on GA104). Every redundant DRAM pass is wasted.

3. **Fill the warp schedulers** — 8 warps/SM is the floor, 12 is the sweet
   spot, beyond 16 yields diminishing returns. Below 8 is a structural problem.

4. **Never cross the 50 KB smem cliff** — `>50 KB/block → 1 block/SM →
   exposed DRAM stalls`. GA104 max smem is 100 KB; the cliff is at 100/2.

These four are not independent. Each interacts with the others. The most
useful intuition is that they form a co-dependent system: violating one
often forces violating another.

## Law 1 — Feed Tensor Cores continuously

### The naive expectation

`cp.async` (the GA10x asynchronous global→smem load) lets a warp issue a DRAM
load and continue executing without stalling on it. The issuing warp does not
block; HMMA can overlap with the in-flight load. Conventional wisdom: always
use cp.async for big tile loads.

### The empirical reality

cp.async helps when the **compute window per tile is short**, hurts when it's
long. From observation 14 in `gpu_reflections.md`:

| compute per tile | cp.async impact |
|---|---|
| 8 IMMA per tile (short) | **+35%** (helps a lot) |
| 64 HMMA per tile (long, FA Phase B) | **-5%** (hurts slightly) |

When the compute window is long enough to fully drain DRAM latency on its
own, the additional cp.async machinery (commit, wait_group barriers) costs
more than it saves. When the compute window is short, cp.async is the only
way to keep Tensor Cores fed.

### Case study: Flash Attention pipeline

Original `flash_attn_br16_pipeline.cu`: cp.async + 64 KB smem (double-buffered
K and V tiles + 16 KB smem_work + 16 KB smem_pv). At 64 KB the kernel runs
**1 block/SM = 4 warps**. Compute per tile: 64 HMMA. Result: lost 4-5% to
cp.async overhead.

Conclusion at the time: "self-attention has too much compute per tile to
benefit from cp.async". Negative result documented in observation 14.

After the smem_work elimination work in issue #29, the same kernel fits in
40 KB (double-buffered K + V tiles + 8 KB FP16 weight_smem). At 40 KB, it
runs **2 blocks/SM = 8 warps**. Same 64 HMMA per tile.

Now cp.async wins **+15-41%** across sizes. See observation Q.

The compute-vs-load ratio is the same. What changed is occupancy: 8 warps
have enough scheduler slots to overlap the cp.async machinery with HMMA
pipeline drain. 4 warps don't.

### Lesson

The compute/load ratio determines whether cp.async is even applicable. The
warp count determines whether it can be scheduled. Both must align for the
optimization to win. Negative results from one regime do not generalize to
another.

## Law 2 — Read each byte of DRAM exactly once

### The naive expectation

DRAM is fast. Modern HBM2 hits 608 GB/s on GA104. Most kernels are compute-
bound; bandwidth doesn't matter.

### The empirical reality

The L2 cache is **4 MB**. Any tensor larger than 4 MB cannot be reused from
L2 for a second pass — it goes back to DRAM. At typical workloads (Conv2d on
SD UNet, LLM attention KV cache), tensors are 16-100 MB. Every "read it
again" pattern is paying full DRAM cost twice.

### Case study: explicit im2col vs implicit GEMM

Explicit im2col writes a `[N×OH×OW, Cin×kH×kW]` column buffer to DRAM, then
a separate GEMM kernel reads it back. At SD UNet sizes (16384 × 1440):

- col buffer: 47.2 MB, exceeds L2 by 12×
- 2 DRAM passes × 47.2 MB = 94.4 MB extra bandwidth per Conv2d
- At 608 GB/s peak, that's 155 μs of pure bandwidth tax per call

The implicit GEMM variant computes im2col coordinates on the fly inside the
WMMA tile loader. Same compute, same arithmetic — but **0 MB col buffer**.
Result: 1.87× speedup at SD config (observation 4 in gpu_reflections.md).

### Case study: ResBlock conv2d (issue #83)

Phase 4 ResBlock used `conv2d_nhwc` (direct conv) which reads input X **9
times per output element** (once per kernel position). At N=1 C=320 H=W=32:
input is 1.3 MB. 9× reads = 12 MB per Conv2d call. Total ResBlock = ~50 MB
of avoidable DRAM traffic.

Swapping to `implicit_gemm_conv` (which reads X exactly once via implicit
im2col): **7× speedup**, 13 ms → 1.86 ms. The win is bigger than the L2
math suggests because the FFMA-based direct conv was also compute-inefficient
(no Tensor Cores).

### Lesson

Before optimizing FLOPS, count DRAM passes. A kernel reading every byte
9 times is not a 9× compute problem; it's a 9× bandwidth problem disguised
as compute. The fix is structural (implicit GEMM, fused passes), not
microscopic (better tiling).

## Law 3 — Fill the warp schedulers

### The naive expectation

GA104 has 4 warp schedulers per SM. So 4 warps per SM is enough to keep them
all busy.

### The empirical reality

The warp schedulers issue one instruction per warp per cycle, but most
instructions have multi-cycle latencies that the scheduler must hide:

- HMMA.16816.F32: 8-cycle pipeline (S08 stall between consecutive HMMAs from same warp)
- LDS.32 (single-bank): 2-3 cycles
- LDS with bank conflict: 2-3 cycles × replay factor (up to 8× for 8-way conflict)
- LDG: 200-400 cycles for an L2 miss

To hide all of this, you need many ready warps. Empirical sweet spots:

| warps/SM | regime |
|---|---|
| 4 | structural problem — many stalls exposed |
| 8 | floor — enough to hide most latencies |
| 12 | sweet spot — bank conflicts and HMMA pipeline drain hidden cleanly |
| 16+ | diminishing returns — extra warps fight for register file |

### Case study: smem padding lost despite eliminating bank conflicts

In the Flash Attention regpv kernel (issue #29 phase 1), the K/V tile and
smem_work overlay both had 8-way ldmatrix bank conflicts (row stride 128 B
mod 32 = 0). Conventional wisdom says padding by +8 halfs (stride 144 B,
mod 32 = 16) eliminates the conflict.

Padding the kernel did eliminate the bank conflicts — but it pushed smem
from 32 KB to 35 KB, dropping occupancy from 3 blocks/SM (12 warps) to
2 blocks/SM (8 warps). Result across all sizes: **-20 to -32% slower**.

At 12 warps, the bank conflict replays were already hidden by the warp
scheduler interleaving. Going to 8 warps exposed them — and exposed the
HMMA pipeline drain — more than padding eliminated. Observation O.

The mechanical fix (eliminate the smem allocation entirely via fragment-shfl
reductions, observation P) preserved 12 warps and delivered +40%.

### Case study: persistent grid loses at high occupancy

`flash_attn_persistent.cu` originally won +10% at large tile counts on the
8-warp baseline kernel. The win came from eliminating the tail wave (when
total tiles isn't a multiple of grid size).

After the smem_work elimination work, the v2 baseline runs 12 warps/SM. The
v2 persistent variant — same persistent loop, same atomicAdd — **lost 0-8%**
across sizes. Why?

At 12 warps/SM, the standard grid wave is short enough that tail-wave
elimination provides no headroom. Meanwhile, `atomicAdd` on the global tile
counter contends across blocks, and at higher SM utilization there are more
blocks contending. The optimization that won at 8 warps lost at 12.

### Lesson

Optimizations are occupancy-dependent. An optimization that wins at low warp
count may lose at high. When tracking new optimizations, always record the
warps/SM it was measured at. When introducing a kernel change that affects
occupancy, re-run the catalog of "established" optimizations — their
status may have flipped.

## Law 4 — Never cross the 50 KB smem cliff

### The expectation

GA104 max smem per SM is 100 KB. cuOccupancyMaxActiveBlocksPerMultiprocessor
returns the right number based on smem usage and register pressure. Trust it.

### The reality

The `100 KB / N blocks` math is not continuous. There are step functions
where the next-smaller block-per-SM count is the correct answer:

- ≤ 32 KB/block → 3 blocks/SM (smem 96 KB, just under 100)
- 33-50 KB/block → 2 blocks/SM (smem 66-100 KB)
- 51-100 KB/block → **1 block/SM** (smem 51-100 KB; this is the cliff)

The cliff at 50 KB is where things turn ugly: 1 block/SM means at most 4
warps (with 128 threads/block), which violates law 3.

### Case study: Bc=128 in Flash Attention

`flash_attn_br16_bc128.cu` doubled the KV tile size from 64 to 128. Larger
tiles → fewer iterations → ostensibly less overhead.

Smem at Bc=128: 16 KB K + 16 KB V + 32 KB scores = 64 KB. Crossed the cliff.
Result: 1 block/SM = 4 warps. Compared to Bc=64 at 2 blocks/SM = 8 warps:
**0.83× speed (-17%)**. The supposedly-better tile size lost.

This is documented as the "Bc=128 cliff regression" in observation 6 of
gpu_reflections.md. The lesson is sharper than it looks: bigger tiles are
not always better; tile sizing must be subordinated to the cliff.

### Case study: smem_work elimination lifted Bc=128 viability

After issue #29's smem_work elimination, a Bc=128 variant becomes possible:
16 KB K + 16 KB V + 16 KB FP16 weight_smem = 48 KB. Just under cliff. Now
2 blocks/SM holds. Whether the larger Bc actually wins compute-wise is
empirical (and not yet measured), but the structural barrier is gone.

### Lesson

Treat 50 KB as a hard structural limit, not a soft target. When smem
allocation approaches the cliff, the question is not "can we squeeze under"
but "can we eliminate one of the allocations entirely". Padding adds smem;
it should not be applied if the kernel is already near 30-32 KB (per-block
allocation for 3 blocks/SM) or near 50 KB (for 2 blocks/SM).

## Synthesis

The four laws are co-dependent:

- Law 1 needs Law 3: cp.async only wins when warp count is high enough
  to schedule it
- Law 2 enables Law 4: structural BW reduction (kill DRAM passes) often
  reduces smem requirements (no col buffer, no FP32 round-trip)
- Law 3 enables Law 1 and limits Law 4: more warps = more cp.async win =
  more smem-per-block budget = ironically less smem-per-block-needed
- Law 4 limits Law 3: smem cliff bounds the warps you can have

Every successful optimization in this repo respects all four. Every failed
optimization (Bc=128, naive padding, persistent at 12 warps, cp.async at
4 warps) violates at least one.

When designing a new kernel, work the laws in order:
1. **Law 4 first**: budget your smem under 32 KB (3 blocks) or under 50 KB
   (2 blocks). Anything else is structural failure.
2. **Law 3 second**: aim for ≥ 8 warps/SM. Anything less indicates
   register or smem mismanagement.
3. **Law 2 third**: count DRAM passes. Eliminate redundant reads via
   register caching, implicit GEMM, fragment-shfl reductions.
4. **Law 1 last**: only after the first three are satisfied does cp.async
   pipelining become a productive optimization.

Doing them in any other order produces optimizations that look good at first
and lose to a baseline once a co-dependent law is violated.

## Further Reading

- `docs/gpu_reflections.md` — full set of observations with measurements
- `docs/fragment_shfl_reductions.md` — pattern that resolves Law 2 + Law 4
  conflicts in Tensor Core kernels with online reductions
- `docs/memory_hierarchy.md` — GA104 cache and bank details
- `docs/control_codes.md` — stall codes and pipeline behavior
- `CLAUDE.md` — short-form statement of the laws
- The chapter outlines (01-05) — case studies of each law in action
