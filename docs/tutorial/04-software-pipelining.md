# Chapter 04 — Software Pipelining

> Read chapter 02 (GEMM) first if you have not. This chapter assumes you
> understand smem-tiled GEMM with WMMA fragments.

The dominant performance question on Ampere is not "are my Tensor Cores
fast" — they are fast. The question is "are my Tensor Cores idle waiting
for DRAM". Software pipelining is the answer to that question, but the
right answer depends on warp count, compute-per-tile ratio, and smem
budget. This chapter walks through when `cp.async` (the Ampere
asynchronous copy primitive) wins, when it loses, and how to predict the
outcome before implementing.

## The synchronous baseline

A standard tiled GEMM has the structure:

```cpp
for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Phase A: cooperative DRAM → smem load
    // Each thread does some LDG + STS pairs
    LDG.E.128 R[regs], [global_A + offset];
    STS [smem_A + smem_offset], R[regs];
    __syncthreads();

    // Phase B: WMMA compute on smem
    wmma::load_matrix_sync(a_frag, smem_A, BK);
    wmma::load_matrix_sync(b_frag, smem_B, BN);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
}
```

The LDG instruction takes ~300 cycles for an L2 miss. During those 300
cycles, the issuing warp is stalled. The warp scheduler tries to find
*another* warp to run, and if there are enough warps, the LDG latency is
hidden by overlap.

This is the latency-hiding mechanism that makes GPUs work at all. Adding
cp.async on top is asking "is the hiding the warp scheduler does enough,
or do we need more?"

## The cp.async primitive

`cp.async` is a PTX instruction (SASS: `LDGSTS.E.BYPASS.128`) that
copies data from global memory directly to shared memory, *bypassing the
register file*, asynchronously. The issuing warp continues executing
without waiting for the copy to land.

```ptx
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;  // 16 bytes async
cp.async.commit_group;                                    // mark end of group
cp.async.wait_group 1;                                    // wait until ≤1 in-flight
```

CUDA C++ wrapper (`<cuda_pipeline.h>`):

```cpp
__pipeline_memcpy_async(smem_dst, gmem_src, 16);
__pipeline_commit();
__pipeline_wait_prior(1);   // wait until at most 1 group is still in flight
```

Key properties:
- Copies bypass L1 cache (cg = "cache global", goes through L2 only)
- Issuing warp does not stall on the copy
- Multiple cp.async copies can be batched into a "group" via `commit_group`
- `wait_group N` waits until at most N groups remain in flight

The asynchrony matters because it lets you *issue* the next tile's load
*before* you finish computing on the current tile. The warp doing compute
is also the warp that issued the prefetch — the prefetch sits in flight
while compute runs.

## Double-buffered pipeline structure

The textbook usage is double buffering. Two smem tiles, A and B. Compute
reads from one while prefetch writes to the other. Roles flip each iter:

```cpp
// Prologue: prefetch tile 0 into buf[0]
cp_async_load(buf[0], global[0]);
__pipeline_commit();
__pipeline_wait_prior(0);
__syncthreads();

for (int i = 0; i < num_tiles - 1; i++) {
    int cur = i & 1;
    int nxt = (i + 1) & 1;

    // Issue prefetch for next tile
    cp_async_load(buf[nxt], global[i + 1]);
    __pipeline_commit();

    // Compute on current tile (overlapped with prefetch in flight)
    wmma_compute(buf[cur]);

    // Wait for the prefetch we issued at the top
    __pipeline_wait_prior(0);
    __syncthreads();
}

// Epilogue: compute on the last tile
wmma_compute(buf[(num_tiles - 1) & 1]);
```

Smem footprint doubles: 2× the K-tile space. This is the cost. The win is
that compute and load run concurrently across the iteration boundary.

## When cp.async wins, and when it does not

The decisive factor is *whether the warp scheduler had slack to begin with*.
On Ampere with 4 warp schedulers per SM, the scheduler can issue 4
instructions per cycle if there are 4 ready warps. If there are 8 active
warps total (typical for a 4-warp block at 2 blocks/SM), 4 of them can
hide while 4 stall on LDG.

Three regimes emerge from the empirical data in this project:

### Regime 1 — warp scheduler slack already sufficient (cp.async loses)

When the kernel has enough warps that the scheduler can find replacements
for any stalled warp, cp.async machinery costs more than its overlap
benefit.

| kernel | warps/SM | compute/tile | cp.async result |
|---|---|---|---|
| HGEMM 64×64 (`hgemm.cu`) | 8 | 64 HMMA | -5% |
| Cross-attn 64×64 (original) | 8 | 16 HMMA | -16% |
| Original `flash_attn_br16_pipeline` | 4 (1 block/SM) | 64 HMMA | -5% |

The original FA pipeline (4 warps/SM) is the most counter-intuitive. With
*only 4 warps*, you might expect cp.async to be most beneficial — fewest
warps to scheduler-hide loads. But at 4 warps the kernel is *also* below
the threshold where the prefetch can be productively overlapped: there
aren't enough warps to issue cp.async, do compute, and have any instruction
slot left for anything else. The cp.async commit/wait machinery itself
costs cycles that aren't hidden.

This is **Observation 14** in the project notes, refined by **Observation Q**
this session.

### Regime 2 — short compute, 8+ warps (cp.async wins big)

When there are 8+ warps but the per-tile compute is short, the scheduler
runs out of work *during* a load, and cp.async fills the gap.

| kernel | warps/SM | compute/tile | cp.async result |
|---|---|---|---|
| IGEMM 64×64 BK=32 | 8 | 8 IMMA | **+35%** |
| FA v2_pipeline | 8 | 64 HMMA | **+14-41%** |

The IGEMM case is the cleanest illustration. With only 8 IMMA per tile,
the compute phase is short — about 100 cycles. The LDG stall is 300+
cycles. Even with 8 warps to overlap, the load-compute imbalance leaves
exposed bubbles. cp.async fills them.

The FA v2_pipeline case is the surprise of this session. The kernel
*originally* lost 5% to cp.async at 4 warps/SM. After issue #29's smem
reduction (32 KB → 24 KB), the same kernel structure ran at 8 warps/SM
(2 blocks/SM). cp.async on top of the *same compute kernel* now wins
+14-41%. The load and compute did not change. The warp count did.

The lesson is simple but easily missed: **cp.async benefit depends on the
warps available to overlap with, not on the kernel's algorithmic shape**.
Same kernel, same compute pattern, different warp count → opposite
result.

### Regime 3 — too few warps to use cp.async productively

Below 8 warps/SM, cp.async generally cannot recover its overhead. The
machinery (commit_group + wait_group + extra smem for double buffers)
costs more than its overlap saves. This is the regime the original FA
pipeline lived in.

The fix is not to fight cp.async at low warp count — it is to *raise the
warp count*. Reducing smem (via pattern from chapter 05's fragment-shfl
reductions, or via removing redundant buffers) is usually the right
upstream change.

## A worked example: phase2 IGEMM

`phase2/igemm/igemm_pipelined.cu` is the synchronous baseline. The async
version is `igemm_pipelined_cpasync.cu`. Both share BM=64, BN=64, BK=32,
4 warps/block, ~16 KB smem, 4 blocks/SM = 16 warps/SM.

The compute phase is tight: per tile, 4 mma_sync calls × 2 K-steps = 8
IMMA per warp. About 100-cycle compute phase. The DRAM load is 16 KB per
block per tile = ~32 cache lines, ~300 cycle worst case.

Compute/load ratio is ~1:3. Even with 16 warps, load > compute means the
scheduler has slack at the *start* of each tile that cp.async can fill.

Measured: cp.async wins ~35% on this kernel.

The lesson from this case is that *high warp count alone* is not enough
to make cp.async lose. The compute/load ratio matters. When compute is
short relative to load, cp.async wins even at high warp count.

## A worked example: Flash Attention (this session)

The before/after for the FA v2_pipeline is the cleanest case study in the
project. Same kernel structure, two smem footprints, two opposite results.

| variant | smem | blocks/SM | warps/SM | compute/tile | cp.async result |
|---|---|---|---|---|---|
| Original FA pipeline | 64 KB | 1 | 4 | 64 HMMA | **-5%** |
| FA v2_pipeline (this session) | 40 KB | 2 | 8 | 64 HMMA | **+14-41%** |

The compute did not change. The warp count doubled because the smem
budget allowed an extra block per SM. cp.async flipped from a 5% loss
to a 41% win at the largest sizes.

This is the most important practical takeaway in the chapter: **before
adding cp.async, reduce smem to maximize warp count**. Adding cp.async to
a kernel that's already at 1 block/SM is asking it to recover from the
worst regime. Adding cp.async to a kernel at 2-3 blocks/SM is giving it
the regime it likes.

## A counter-example: bigger tile with cp.async

What about combining cp.async with a bigger tile? FA Bc=128 with cp.async
double-buffer would need 32 K + 32 V (double-buffered) + 16 weight = 80 KB
smem. That crosses the 50 KB cliff → 1 block/SM (4 warps). Predicted
catastrophic; would need actual measurement to confirm magnitude. But
based on regime 3 analysis, expecting 2× regression vs no-cp.async Bc=64.

The mistake to avoid: adding cp.async on top of *every* optimization. It
amplifies wins in good regimes but amplifies losses in bad ones. If your
optimization pushes you below 8 warps/SM, cp.async will pay the
amplification cost on the loss side.

## wait_group depth: a small tuning knob

`cp.async.wait_group N` waits until at most N groups remain in flight.
Common choices:

- `wait_group 0` — wait until *all* in-flight cp.async are done (full
  synchronization)
- `wait_group 1` — wait until at most 1 group is in flight (the typical
  double-buffer choice — keep one prefetch in flight while computing)
- `wait_group 2+` — keep more prefetches in flight (triple-buffering)

Triple-buffering on GA104 typically shows diminishing returns past 2
in-flight groups, because:

- Each in-flight group consumes smem (need 3× tile space instead of 2×)
- The DRAM transaction queue itself has limited depth
- The scheduler has limited ports

`phase2/igemm/igemm_pipelined_cpasync_bk64.cu` increases the K-tile size
(BK=64 instead of BK=32, doubling smem traffic per tile but halving outer
iterations) — a different lever than triple-buffering. Worth comparing.

## How to diagnose whether cp.async would help

Before implementing, estimate:

1. **Current warps/SM**: blocks/SM × warps/block
2. **Compute per tile**: count HMMA/IMMA instructions, multiply by ~16
   cycles each (HMMA on sm_86 has S08 stall between consecutive issues)
3. **DRAM bytes per tile**: tile_size × element_bytes
4. **Compute-to-load cycles ratio**: compute_cycles / (DRAM_bytes / DRAM_BW × clock_rate)

The decision rule from this project's experience:

| warps/SM | compute < load? | cp.async likely... |
|---|---|---|
| ≥ 8 | yes (compute bottleneck) | **wins** (e.g., IGEMM, FA v2) |
| ≥ 8 | no (load bottleneck) | break-even or slight win |
| 4-7 | yes | break-even or slight loss |
| ≤ 4 | either | **loses** (e.g., original FA pipeline) |

The cleanest rule: **if you are at 1 block/SM, fix smem first; do not add
cp.async**.

## Performance summary (cp.async results from this project)

| kernel | warps/SM | compute/tile | cp.async Δ | source |
|---|---|---|---|---|
| HGEMM 64×64 | 8 | 64 HMMA | -5% | observation 2 |
| Cross-attn 64×64 (orig) | 8 | 16 HMMA | -16% | issue #5 |
| Self-attn (FA orig pipeline) | 4 | 64 HMMA | -5% | observation 14 |
| **IGEMM 64×64 BK=32** | 8 | 8 IMMA | **+35%** | observation 14 |
| **FA v2_pipeline** | 8 | 64 HMMA | **+14 to +41%** | this session, observation Q |

The two bold rows are the regimes where cp.async pays. They share warps/SM
≥ 8, and either short compute (IGEMM) or recently-freed warps via smem
reduction (FA v2). The other three rows are the regimes where it loses
(too few warps, or too much compute already, or both).

## How to run it yourself

```bash
cd /mnt/d/dev/p/bare-metal/phase2/igemm

# Build sync and async variants
nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined.sm_86.cubin igemm_pipelined.cu
nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined_cpasync.sm_86.cubin igemm_pipelined_cpasync.cu

# (use phase2/igemm/bench.cu to compare; +35% expected on cpasync)

cd ../../phase3/flash_attention

# Build the v2 + v2_pipeline pair
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2.sm_86.cubin flash_attn_br16_v2.cu
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2_pipeline.sm_86.cubin flash_attn_br16_v2_pipeline.cu

# Compare
nvcc -arch=sm_86 -O2 -o bench_v2_variants bench_v2_variants.cu -lcuda -I../../kernels/_common
./bench_v2_variants 1024 8 8
./bench_v2_variants 1024 8 8   # second run is steady-state

# Expect ~10000 GFLOPS for v2, ~11500 for v2_pipeline (about +14%)
```

## What this chapter teaches

cp.async is not a universal optimization. It is a tool for hiding load
latency *when there is exposed load latency to hide*. The exposed-latency
condition requires:

1. Enough warps that the warp scheduler has slack to issue cp.async
   alongside compute (8+ warps/SM)
2. A compute/load ratio that leaves bubbles even with that slack (short
   compute or many DRAM bytes per tile)

If those conditions are not met, cp.async is dead weight. Worse: in
regime 3 (≤4 warps/SM), it can amplify a regression.

The most valuable thing this chapter teaches is the *pattern of when to
re-test cp.async*. Same kernel can flip from "cp.async loses 5%" to
"cp.async wins 41%" purely by changing smem footprint to gain a block per
SM. Whenever you change anything that affects occupancy, re-run
cp.async — it may have flipped. This is law 3 (fill the warp schedulers)
in disguise.

## Source files

- `phase2/igemm/igemm_pipelined.cu` (sync baseline, IGEMM)
- `phase2/igemm/igemm_pipelined_cpasync.cu` (cp.async double-buffer, IGEMM)
- `phase2/igemm/igemm_pipelined_cpasync_bk64.cu` (BK=64 variant)
- `phase2/igemm/igemm_pipelined_cpasync_perchannel.cu` (per-channel scale fusion)
- `phase3/flash_attention/flash_attn_br16_pipeline.cu` (original, lost at 4 warps)
- `phase3/flash_attention/flash_attn_br16_v2_pipeline.cu` (won after smem reduction)

## Cross-references

- Postmortems: `docs/gpu_reflections.md` Insights 2, 14; Observation Q
- Issues: #5, #78
- Chapter 05 — Flash Attention (the deep case study where cp.async flipped)
- Chapter 06 — The Four Laws (Law 1 + Law 3)
