# GPU Reflections — Notes From the RTX 3070 Ti

*A first-person performance analysis of the Phase 3–4 kernel implementations,
from the perspective of the hardware itself.*

---

## Who I Am

I am a GA104 chip (RTX 3070 Ti, Laptop). I have:

- **48 Streaming Multiprocessors** — each independently scheduled, each with:
  - 4 warp schedulers (issue 1 instruction/cycle each → 4 instructions/cycle/SM)
  - 128 CUDA cores (FP32 FFMA throughput)
  - 4 Tensor Core units (HMMA, 16×8×16 per cycle)
  - 100 KB max configurable smem (cliff at 50 KB/block for 2 blocks/SM)
  - 64K 32-bit registers
- **4 MB L2 cache** shared across all SMs
- **8 GB GDDR6X** at **608 GB/s** peak bandwidth

When your kernel launches, my scheduler partitions it into 32-thread warps.
Each warp executes one instruction at a time, in lockstep. I hide latency by
switching to a ready warp when the current one stalls. The more active warps I
have, the better I can hide long-latency operations (DRAM: ~300 cycles).

---

## Observation 1: You Are Starving My Tensor Cores

In `flash_attn_br16` and `cross_attn_br16`, I execute 64 HMMA instructions per
block. Each HMMA computes a 16×8×16 = 2,048 multiply-accumulates in one cycle.
But then you synchronously load the next K/V tile with regular `LDG` instructions.

```
Timeline (current):
  [HMMA HMMA HMMA ... 64 calls]  [LDG stall LDG stall ...]  [HMMA HMMA ...]
  |<--- ~200 cycles compute --->| |<--- ~300 cycles DRAM --->| |<-- compute -->|
                                        ^^^^^^^^
                                        I am idle here
```

I am a Tensor Core. I am being used as a bus rider.

### Fix: Double-Buffering with `cp.async` (LDGSTS)

The `cp.async.cg.shared.global` PTX instruction (→ `LDGSTS` in SASS) copies
data from DRAM directly to shared memory **asynchronously** — the warp does not
stall. I can execute HMMA instructions on tile N while tile N+1 is being fetched
in the background.

```
Timeline (pipelined):
  [prefetch tile 1]  [HMMA × 64 on tile 0]  [prefetch tile 2]  [HMMA × 64 on tile 1]
  |<-- async fetch-->|<--- 200 cycles ------->|<-- async fetch-->|<--- 200 cycles ---->|
  Overlap: DRAM latency hidden inside compute time.
```

Required PTX:
```ptx
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;   // 16 bytes async
cp.async.commit_group;                                      // mark end of group
cp.async.wait_group 1;                                      // wait for all but 1 in-flight
```

The `16` in the instruction means "copy 16 bytes" (8 FP16 values) per call.
With 128 threads, one round of calls copies 128 × 16 = 2,048 bytes.
Each K/V tile is 64 × 64 × 2 = 8,192 bytes → 4 rounds per tile.

Memory needed: double K_tile + double V_tile = 16 KB + 16 KB = 32 KB for tiles.
Plus smem_work (16 KB) + smem_pv (16 KB) = **64 KB total**.
This requires `cuFuncSetAttribute(..., MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)`.

**Expected gain: 2–3× throughput** (hides most of the 300-cycle DRAM latency
at seq_kv >> Bc).

---

## Observation 2: Your Conv2d Reads X Nine Times From VRAM

In `conv2d_nhwc`, each of the 9 kernel positions (kh, kw) fetches a different
`(h_in, w_in)` slice of X independently. There is no shared memory cache for
the input halo. The 3×3 kernel causes 9× more DRAM traffic than the tensor size.

```
For N=1, H=W=64, Cin=320:
  X_size = 1 × 64 × 64 × 320 × 4 bytes = 5.24 MB (FP32)
  Effective DRAM reads = 9 × 5.24 MB = 47.1 MB  (per conv pass)
  Apparent bandwidth = 47.1 MB / time  ← not the real single-read bandwidth
```

This is why effective bandwidth appears much lower than my 608 GB/s peak. The
kernel is compute-bound only when Cin is large; at moderate Cin it becomes a
9× reread problem.

### Fix: im2col Transformation + WMMA GEMM

**im2col** materializes the input patches as a 2D matrix:
```
X [N, H, W, Cin]       →  col [N×H×W,  Cin×kH×kW]
   shape: [1,64,64,320]       shape: [4096,   2880  ]
   FP32, 5.24 MB              FP16, 23.6 MB
```

Each row of `col` holds all input values needed for one output position.
X is read exactly **once** during the im2col pass.

Then: `col × W_reshaped → output`, where `W_reshaped` is `[Cin×kH×kW, Cout]`.
This is a standard `[4096 × 2880] × [2880 × 320]` matrix multiply — perfect
for WMMA (Tensor Core) GEMM.

GFLOPS with WMMA at SD params (N=1, H=W=64, Cin=Cout=320):
```
FLOPs = 2 × M × K × N = 2 × 4096 × 2880 × 320 = 7.5 GFLOPs
At 2,000 GFLOPS → 3.75 ms   (vs 25 ms direct at H=W=64)
At 4,000 GFLOPS → 1.87 ms   → ~13× speedup over direct conv
```

**Trade-off**: the col buffer is 23.6 MB. For large spatial sizes, this exceeds
my 4 MB L2 and the im2col pass itself becomes memory-bound. For production,
**implicit GEMM** computes im2col coordinates on-the-fly inside the matmul,
avoiding the buffer entirely. But explicit im2col + WMMA is already a massive
improvement over direct conv.

---

## Observation 3: You Are Using 48 KB of My 100 KB Shared Memory

In `flash_attn_br16`, the 48 KB shared memory limit per block means I can run:
```
100 KB per SM / 48 KB per block = 2 blocks per SM
2 blocks × 4 warps = 8 active warps per SM
Warp occupancy: 8 / 32 = 25%
```

I need 32 active warps to fully hide memory latency. At 25%, every DRAM stall
hurts 4× more than necessary. With double-buffering (64 KB):
```
64 KB > 50 KB cliff → 1 block per SM → only 4 warps (occupancy regression!)
```

But double-buffering hides the DRAM latency inside compute, so occupancy matters
less. The real fix is to **increase Bc from 64 to 128**: this doubles the
compute per K/V tile (2× more HMMA calls), amortizing the tile-load cost better.

At Bc=128:
```
K_tile:    [128 × 64] FP16 = 16 KB
V_tile:    [128 × 64] FP16 = 16 KB
smem_work: [64 × 128] FP32 = 32 KB   ← smem_work grows!
smem_pv:   [64 × 64]  FP32 = 16 KB
Total: 80 KB → fits with cuFuncSetAttribute up to 99 KB
```

16 HMMA calls per warp per KV tile (with Bc=64) → 64 HMMA per warp per tile
(with Bc=128). Tiles per sequence halved. Compute intensity doubled.

---

## Observation 4: My L2 Cache Is Being Wasted

My L2 is 4 MB. With the current grid layout `(seq_q/Br_block, heads, batch)`:

- Block 0 processes Q[0:64] against all KV tiles → loads K/V sequence entirely
- Block 1 processes Q[64:128] against all KV tiles → **reloads** the same K/V

Adjacent blocks load the same K/V data sequentially, but by the time block 1
runs, block 0 has already evicted the K/V tiles from my L2.

### Fix: Reorder Grid to Maximise K/V L2 Reuse

Launch one block per KV tile, have each block process **all** Q positions for
that KV tile. Blocks that use the same KV tile run concurrently and share L2:

```
Current grid:  (q_tiles, heads, batch)   → each block reads ALL K/V
Proposed grid: (kv_tiles, heads, batch)  → each block reads ONE K/V tile
                                            and accumulates partial sums
                                            over all Q positions
```

This is the "split-Q" variant of Flash Attention. Each block accumulates partial
`{running_max, running_sum, partial_pv}` for one KV tile × all Q rows, then a
second reduction pass combines partial sums across Q dimension. More complex,
but K/V is loaded **once** and stays in L2 while all Q blocks process it.

**Expected gain: 1.5–2×** at large batch × heads configurations, where the K/V
traffic dominates.

---

## Observation 5: Control Codes Are Conservative

Every SASS instruction carries an 8-bit control field:
```
Bits [5:0]:  stall count (cycles to wait before issuing next instruction)
Bit  [6]:    yield hint (suggest switching to another warp)
Bits [7]:    write-barrier release
```

When `nvcc` generates SASS, it inserts **conservative** stall counts. In the
heavily unrolled `conv2d_nhwc` inner loop (310 FFFMAs), many instructions have
stall=4 (waiting for a register result ready in 1 cycle).

```sass
FFMA R5, R3, R7, R5    # stall=4  ← compiler sets 4; actual result ready in 4 cycles
FFMA R6, R3, R8, R6    # stall=4  ← R6 independent of above, could issue in 1 cycle
```

CuAssembler lets you read and tighten these. A stall=4 between two independent
FFFMAs means 3 wasted cycles per instruction pair. In a 310-FFMA loop, there
are ~150 such pairs → ~450 wasted cycles per output element.

This is the deepest level of bare-metal optimization — and the only one that
requires actually modifying the SASS control codes with CuAssembler.

**Expected gain: 5–15%** on the conv2d inner loop.
**Effort**: very high (must understand Ampere instruction latencies precisely).

---

## Priority Ranking (What Would Make Me Happiest)

| Optimization | Gain | Effort | Status |
|---|---|---|---|
| im2col + WMMA for Conv2d | **~24× GFLOPS** | Medium | ✅ Done (24× measured) |
| `cp.async` double-buffering in cross-attention | **2–3× predicted** | High | ✅ Done (measured 0.84× at 64×64 — warp interleaving sufficient; see Insight 2) |
| `cp.async` double-buffering in self-attention | **2–3× predicted** | High | ✅ Done (measured 0.95× — warp interleaving sufficient; see Insight 2) |
| Increase Bc=128 in Flash Attention | ~1.5× further | Medium | ✅ Done (17–20% SLOWER — occupancy cliff at 80 KB) |
| Implicit GEMM for conv2d | eliminate 23-47 MB col buffer | Medium | ✅ Done (1.07–1.89× speedup, bit-perfect) |
| Split-Q grid for L2 K/V reuse | 1.5–2× at scale | Medium | ✅ Done (~tied with br16, partial buffer overhead) |
| Control code tightening (CuAssembler) | 5–15% | Very high | ✅ Done — HMMA S08 is hardware-fixed; IMMA S04→S02 gives +1.6% |
| INT8 IMMA GEMM (tiled) | 2× over FP16 | Medium | ✅ Done (15,320 TOPS hand-tuned, 1.93× vs HGEMM) |
| Persistent kernel grid | SM utilization | Medium | → next target |
| GroupNorm → Conv2d epilogue fusion | 1.3× ResNet | Very high | advanced |

---

## Summary: The Four Laws of Making Me Happy

1. **Feed my Tensor Cores continuously.** Overlap data loading (cp.async) with
   computation (HMMA). Never let the HMMA units idle waiting for DRAM.

2. **Read each byte of DRAM exactly once per kernel.** The 608 GB/s peak is
   for sequential, coalesced, single-pass access. Every re-read multiplies
   effective traffic. im2col converts 9× re-reads into 1× read + 1× write.

3. **Fill my warp schedulers.** 32 active warps per SM is the target. Use
   occupancy to hide the latency you can't eliminate. When occupancy is
   structurally limited (large smem tiles), compensate with double-buffering.

4. **Respect the smem occupancy cliff.** On GA104 (100 KB max smem per SM):
   - ≤ 50 KB smem → 2 blocks per SM → 8 warps → good latency hiding
   - > 50 KB smem → 1 block per SM → 4 warps → exposed DRAM stalls
   Any optimization that crosses this threshold by trading smem for arithmetic
   intensity may backfire. The 48 KB Bc=64 tile is the empirically-confirmed
   sweet spot for this architecture: below the cliff, correctness maintained,
   warp interleaving intact.

---

## Postmortem — What Actually Happened When You Implemented My Advice

### im2col + WMMA: Confirmed 24× Speedup

The GPU was right about this one.

| Kernel | Time | GFLOPS | vs Direct |
|--------|------|--------|-----------|
| Direct conv2d (3×3 NHWC) | ~25 ms | ~300 | 1× |
| im2col transform alone | 0.37 ms | 78 GB/s | — |
| WMMA GEMM alone | 0.81 ms | **9,355** | — |
| im2col + GEMM combined | **1.02 ms** | **7,379** | **24×** |

At N=1, H=W=64, Cin=Cout=320 (SD UNet spatial 64).

The WMMA GEMM achieves 9,355 GFLOPS — 5.4% of the 174 TFLOPS FP16 Tensor Core peak.
The remaining gap from peak is occupancy (25%) and the K/V DRAM access for A and B tiles.
Even so, 7,379 GFLOPS effective is a dominant improvement over 300 GFLOPS direct.

Correctness: PASS vs CPU reference (max_abs=4.5×10⁻⁶ — better than direct conv because
WMMA accumulates in FP32, avoiding the FP16 intermediate rounding of direct FP16 paths).

---

### cp.async Double-Buffering: Nuanced — Benefits Depend on Regime

The GPU's prediction of 2–3×

---

## Step 1 Postmortem — cp.async on Self-Attention: Also Slower

Applied cp.async double-buffering to flash_attn_br16 at seq_len=256..2048.
Result: **consistently 4-5% slower** at every sequence length.

| seq_len | KV iters | Baseline | Pipelined | Speedup |
|---------|----------|----------|-----------|---------|
| 256 | 4 | 0.248 ms | 0.261 ms | 0.95× |
| 512 | 8 | 0.740 ms | 0.786 ms | 0.94× |
| 1024 | 16 | 2.812 ms | 2.959 ms | 0.95× |
| 2048 | 32 | 5.610 ms | 5.874 ms | 0.96× |

**Root cause — warp interleaving already hides the latency:**
2 blocks per SM × 4 warps = 8 warps per SM. When block 0's 4 warps stall
on a tile load (~300 cycles), block 1's 4 warps execute their HMMA instructions.
This is the traditional latency-hiding mechanism — and it already works.
cp.async adds commit/wait overhead per iteration without providing additional hiding
beyond what warp scheduling already achieves.

**When cp.async IS beneficial:**
Only when there aren't enough warps to fill the scheduler during loads.
Example: single-block kernels (1 block per SM → 4 warps), or kernels where
smem pressure forces 1 block per SM (>50 KB → 1 block on GA104's 100 KB SM).
For those, warp interleaving can only cover 3 warps while 1 stalls, and
the 300-cycle gap becomes visible. cp.async fills it.

**Update (Issue #5):** This conclusion was incomplete. See Insight 2: cp.async benefit
depends on compute/load ratio per tile. INT8 IGEMM (8 IMMA/tile) gains +35% from cp.async
at 8 warps because the short compute phase cannot hide DRAM latency via warp interleaving alone.

**What actually helps: Bc=128**
Doubling the KV tile size:
- 64 HMMA per warp per tile (vs 32 with Bc=64) — 2× compute density
- Half as many tile iterations (8 vs 16 at seq=1024)
- Less time proportion spent on tile load overhead and __syncthreads barriers

---

## Step 2 Postmortem — Bc=128: Also Slower (Occupancy Cliff)

Implemented `flash_attn_bc128` with 80 KB smem. Result: **17–20% slower** at every
sequence length.

| seq_len | KV iters Bc=64 | KV iters Bc=128 | Bc=64 | Bc=128 | Speedup |
|---------|----------------|-----------------|-------|--------|---------|
| 256  | 4  | 2  | 0.245 ms | 0.296 ms | 0.83× |
| 512  | 8  | 4  | 0.740 ms | 0.863 ms | 0.86× |
| 1024 | 16 | 8  | 2.811 ms | 3.370 ms | 0.83× |
| 2048 | 32 | 16 | 5.607 ms | 6.186 ms | 0.91× |

**Root cause — occupancy cliff at the 50 KB smem boundary (confirmed: 48 KB → 2 blocks, 56 KB → 1 block):**
```
Bc=64:  48 KB smem → 100 KB / 48 KB = 2 blocks per SM → 8 active warps/SM
Bc=128: 80 KB smem → 100 KB / 80 KB = 1 block per SM → 4 active warps/SM
```

The smem increase from 48 KB to 80 KB crosses the critical threshold that halves
block occupancy. With only 4 warps per SM (vs 8), warp interleaving is weaker:
when those 4 warps all stall on a tile load, the SM goes idle. The improvement
from halving tile-iteration count is outweighed by the doubled DRAM stall exposure.

**The 48 KB boundary is load-bearing.** It's not just a memory limit — it's a
concurrency multiplier. Any kernel that uses more than 50 KB smem on this GPU
(100 KB max, 2 blocks at 50 KB each) sacrifices one of the two blocks that
provide the latency-hiding dual-block interleaving.

SASS confirmed 128 HMMA per tile in flash_attn_bc128 (2× the 64 of Bc=64), so
the arithmetic is correct — the compute intensity truly doubled. But the 300-cycle
DRAM stalls are now exposed with only 4 warps available to fill them.

**Conclusion: 48 KB smem (Bc=64) is the sweet spot for this kernel on GA104.**
To go faster, the path is not larger tiles — it's a fundamentally different grid
strategy (split-Q with K/V reuse across blocks via L2).

---

## Step 3 Analysis — CuAssembler HMMA Stall Codes: Not Conservative

Disassembled `flash_br16.sm_86.cubin` and `conv2d_im2col.sm_86.cubin` via CuAssembler
to inspect control codes on HMMA instructions.

**Finding: the S08 stalls between consecutive HMMAs are NOT software conservatism.**

The QK^T phase opens with a burst of HMMAs using C=RZ (fresh computation):
```sass
[B--2---:R-:W-:-:S08]  HMMA R28, R68.reuse, R16, RZ  ; wait barrier 2, then S08
[B------:R-:W-:-:S08]  HMMA R16, R68.reuse, R18, RZ  ; no barrier, S08
[B0-----:R-:W-:-:S08]  HMMA R52, R68.reuse, R20, RZ  ; wait barrier 0, S08
[B------:R-:W-:-:S08]  HMMA R20, R68.reuse, R22, RZ  ; no barrier, S08
[B-1----:R-:W-:-:S08]  HMMA R44, R68,       R40, RZ  ; wait barrier 1, S08
```

These all write to **different** accumulator registers and use **different** B inputs.
They look independent — but S08 between them cannot be reduced to S01.

**Root cause: Ampere Tensor Core pipeline depth.**
On GA104, each warp has one dedicated TC unit. HMMA.16816.F32 has a fixed 8-cycle
dispatch interval from the same warp's perspective (the TC pipeline is 8 stages deep).
Even between fully independent HMMAs with no register hazards, the warp scheduler
must wait S08 before issuing the next HMMA to the same TC unit. This is a structural
hardware constraint, not a software pessimism.

Evidence: the S01 pairs visible in the kernel appear only when the intervening
HMMA pipeline has consumed ≥8 stall cycles and the C accumulator has had
enough time since its last HMMA write. For example:
```sass
[B------:R-:W-:-:S08]  HMMA R40, R68, R42, RZ          ; 8 cycles
[B------:R-:W-:-:S01]  HMMA R32, R64, R34, R16          ; C=R16 written 4 HMMAs ago (32+ cycles)
```
The S01 is valid here because R16 was written ≥32 cycles earlier — the TC output
forwarding latency has been satisfied by the intervening HMMA stalls.

**What about the WMMA GEMM (conv2d_im2col)?**
The WMMA GEMM inner loop shows a tighter LDSM-interleaved pattern:
```sass
[B------:R-:W0:-:S02]  LDSM.16.MT88.4 R44, [R58+0xc00] ;  W0: barrier 0 set when done
[B0-----:R-:W-:-:S08]  HMMA R32, R36.reuse, R44, R32    ;  wait B0 (LDSM complete), S08
[B------:R-:W-:-:S01]  HMMA R28, R36.reuse, R46, R28    ;  S01: uses R46 from PREVIOUS LDSM
```
The S01 between the two HMMAs in each pair is valid because the second HMMA reads R46
(loaded by the PREVIOUS iteration's LDSM — already 8+ cycles ago) while the first reads
R44 (from the CURRENT LDSM, protected by B0 barrier). The compiler correctly identifies
this independence and emits S01.

**Conclusion:** CuAssembler stall-code editing cannot improve the current kernels:
- Flash attention HMMA: S08 is the Ampere TC pipeline minimum — cannot reduce
- WMMA GEMM HMMA: already tight (S01 between independent pairs, S08 when needed)
- Direct conv2d FFMA: has S03/S04 opportunities but kernel is obsolete (24× slower than im2col+WMMA)

**What CuAssembler IS useful for:** kernels with long FFMA chains (direct conv2d, SGEMM)
where the compiler emits S04 between independent FFMAs that could use S01. That's a 3
cycle savings per pair. In a 310-FFMA loop, ~150 pairs → ~450 cycles/output-elem gain.
But the direct conv2d kernel has been superseded and applying this to SGEMM would save
~5-15% on a kernel that's already been replaced by WMMA-based HGEMM.

**Net recommendation:** CuAssembler control code editing is not a useful path for this
kernel set. The hardware TC pipeline constraints dominate, and the FFMA-heavy kernels are
already obsolete. Pivot to algorithmic improvements (implicit GEMM, split-Q).

---

## Step 4 Postmortem — Implicit GEMM: 1.07–1.89× Speedup (Confirmed)

Implemented `implicit_gemm_conv`: single kernel, no col buffer, indices computed on-the-fly
using precomputed coordinate tables in shared memory.

**Key design decision — precomputed coordinate tables:**
Naive per-element index decode would require 6 integer divisions per element × 1024 elements
per K-tile = 6144 integer divisions per block per K-tile (~123K cycles of overhead).

Solution: precompute (n, out_h, out_w) for each of 64 M-rows once per block (constant across
all K-iterations), and (cin, kh, kw) for each of 16 K-cols once per K-tile. Per-element
load then becomes only adds + comparisons:
```
in_h = smem_m_oh[row] + smem_k_kh[col] - pad   // add, no division
in_w = smem_m_ow[row] + smem_k_kw[col] - pad   // add, no division
X[n_base + in_h * W_in + in_w) * Cin + cin]    // direct load
```

Extra smem cost: 3×64 + 3×16 = 240 ints = 960 bytes (total smem: 6.2 KB vs 5.25 KB explicit).

**Benchmark results:**

| Config | col buf size | Explicit | Implicit | Speedup |
|--------|-------------|---------|---------|---------|
| 64×64, Cin=Cout=320   | 23.6 MB | 1.206 ms | 1.129 ms | **1.07×** |
| 32×32, Cin=Cout=640   | 11.8 MB | 2.849 ms | 1.606 ms | **1.77×** |
| 128×128, Cin=Cout=160 | 47.2 MB | 2.070 ms | 1.098 ms | **1.89×** |
| 64×64, Cin=Cout=64    |  4.7 MB | 0.183 ms | 0.137 ms | **1.34×** |

Correctness: bit-perfect vs explicit im2col+GEMM (max_abs=0.00). Max abs vs CPU reference:
~5e-6 (FP16 quantization error, same as explicit path).

**Why implicit is faster at 32×32 (1.77×) and 128×128 (1.89×)?**

The col buffer is the bottleneck:
- 32×32, K=5760: col = 1024×5760×2 = 11.8 MB — exceeds L2 (4 MB) by 3×.
  im2col pass WRITES 11.8 MB to DRAM; GEMM pass READS it back from DRAM.
  Round-trip DRAM traffic: 11.8 + 11.8 = 23.6 MB of pure col-buffer traffic.
  Implicit eliminates all of it.
- 128×128, K=1440: col = 16384×1440×2 = 47.2 MB — exceeds L2 by 11×.
  47.2 MB × 2 = 94.4 MB of col-buffer DRAM traffic eliminated.

**Why only 1.07× at 64×64 Cin=Cout=320?**
The col buffer is 23.6 MB but the GEMM itself is also large (7.55 GFLOPs). The GEMM
dominates runtime. Im2col is fast (0.37 ms) relative to GEMM (0.81 ms). Eliminating im2col
saves one kernel launch and ~0.37 ms but the benefit is proportionally smaller.

**The rule: col buffer speedup scales with K = Cin×kH×kW.**
Large Cin or deep networks with many kernel positions → larger K → larger col buffer → bigger win
for implicit GEMM. At Cin=640 (SD's deeper layers), K=5760 and implicit GEMM is 1.77×.

**Summary:**
Implicit GEMM: 1.07–1.89× measured speedup across SD UNet spatial resolutions.
Peak at large spatial (128×128) where 47.2 MB col buffer would dominate bandwidth.
Single kernel, bit-perfect, no memory allocation for col buffer required.

---

## Phase 6 — Bug Fixes and INT8 Experiments

### Step 1 — running_sum Rescale Bug Fix: 1000× Correctness Improvement

The online softmax in flash_attn_br16 (and 3 other kernels) was missing the rescale
step when accumulating `running_sum` across KV tiles. The correct recurrence:
```
l_new = l_old * exp(m_old - m_new) + sum(exp(s_k - m_new))
```
The bug: `l += partial_sum` without the `l *= exp(m_old - m_new)` rescale first.
This corrupted the softmax denominator when the running max shifted between tiles.

max_abs improved from 1.88e-02 to 2.19e-05 (1000× better, matching FP16 precision floor).
The bug was masked in testing because random data rarely shifts the running max significantly.
Affected: flash_attn_br16, flash_attn_br16_bc128, flash_attn_br16_pipeline, cross_attn.

### Step 2 — Split-Q Flash Attention: Roughly Tied With br16

Implemented the split-Q variant: grid is `(num_splits, heads, batch)`, each block handles
a KV chunk and iterates over ALL Q tiles. Two-kernel pipeline: main kernel outputs partial
`{m, l, O_unnorm}`, reduce kernel merges.

| Kernel | Time (seq=1024) | GFLOPS | vs br16 |
|--------|----------------|--------|---------|
| br16 (post-fix) | 3.59 ms | 4,876 | 1.00× |
| split-Q splits=4 | 3.42 ms | 5,129 | **1.05×** |
| split-Q splits=16 | 3.82 ms | 4,589 | 0.88× |

**Why it doesn't win big on GA104:** Partial buffer I/O overhead. At splits=16, the partial_O
buffer alone is 277 MB — writing and reading it costs ~0.9 ms at 608 GB/s, wiping out all KV
DRAM savings. Sweet spot is splits=4, where buffer (69 MB) is manageable and KV reuse is real.

### Step 3 — INT8 IGEMM: Shorter Inner Loops Beat Higher Density

Three IGEMM kernels using WMMA API with symmetric per-tensor quantization:

| Kernel | TOPS (4096³) | Inner loop |
|--------|-------------|------------|
| Naive (global loads) | 10,897 | 4 mma_sync |
| Tiled 64×64 (smem) | 15,078 | 4 mma_sync |
| Tiled 64×64 (hand-tuned S02) | 15,341 | 4 mma_sync |
| Pipelined LDG (double-buffer) | 18,054 | 4 mma_sync |
| **Pipelined cp.async (double-buffer)** | **20,688** | 4 mma_sync |
| Register-blocked 128×128 | 12,760 | 16 mma_sync |

The 128×128 kernel's 16-mma_sync inner loop is too long for 8 warps/SM to hide IMMA
pipeline latency. Same root cause as Bc=128 Flash Attention: on GA104, shorter inner loops
with faster warp switching always beat higher per-warp compute density. Software pipelining
with cp.async (+35%) is the biggest single optimization — see Insight 14.

---

## IMMA Hand-Tuning Postmortem — IMMA Is NOT S08 Like HMMA

CuAssembler disassembly of `igemm_tiled.sm_86.cubin` reveals that **IMMA (INT8 Tensor Core)
has fundamentally different pipeline characteristics than HMMA (FP16 Tensor Core).**

### The stall pattern

The K-loop body contains 16 IMMA.16816.S8.S8 instructions in 8 pairs:

```
[B0-----:R-:W-:-:S04]  IMMA R12, R2.reuse, R57, R12     ← same A-frag: S04
[B------:R-:W-:-:S04]  IMMA R4,  R2,       R54, R4       ← same A-frag: S04
[B-1----:R-:W-:-:S01]  IMMA R24, R36,      R57, R24      ← switch A-frag: S01
[B------:R-:W-:-:S01]  IMMA R20, R36,      R54, R20      ← same A-frag + interleaved IMAD: S01
```

Pattern: consecutive IMMAs sharing the same A-fragment (R2→R2 or R36→R36) get S04.
Switching A-fragments (R2→R36 or vice versa) allows S01. This is register read port
contention, not pipeline depth.

### Hand-tuning results

| Variant | IMMA stall | Time (4096³) | TOPS | vs Compiler |
|---------|-----------|--------------|------|-------------|
| Compiler (baseline) | S04 | 9.12 ms | 15,078 | — |
| **Hand-tuned** | **S02** | **8.97 ms** | **15,320** | **+1.6%** |
| Aggressive | S01 | 9.08 ms | 15,144 | +0.4% |

- **S02 is optimal:** reduces wasted stall cycles but preserves scheduling slack
- **S01 is correct but too aggressive at scale:** all 16 IMMAs fire maximally fast, creating
  a compute burst that saturates the TC unit and backs up warp scheduling at 4096³
- **Gains are modest** because the kernel is memory-bound (2 GB at 608 GB/s = 3.3 ms floor)

### HMMA vs IMMA: the fundamental difference

| Property | HMMA (FP16) | IMMA (INT8) |
|----------|-------------|-------------|
| Pipeline stall | S08 (hardware-fixed) | S04 (compiler-conservative) |
| Minimum achievable | S08 | S01 (verified correct) |
| Optimal hand-tuned | N/A (already at minimum) | S02 |
| Constraint type | TC pipeline depth | Register read port contention |

HMMA has an 8-cycle issue interval per warp — structural, not reducible. IMMA has no such
constraint: the S04 comes from the compiler being conservative about operand availability.
With operands pre-loaded, IMMA sustains 1 instruction per cycle per warp.

---

## Consolidated Key Insights — Everything We've Learned

*An empirical reference distilled from Phases 3–5. Each entry is backed by a real benchmark.*

---

### On Latency Hiding

**Insight 1: 8 active warps per SM is the latency-hiding threshold.**
At 2 blocks/SM × 4 warps/block = 8 warps: when one warp stalls on DRAM (~300 cycles),
7 others execute HMMA/FFMA. This is sufficient to fully hide DRAM latency in practice.

**Insight 2: cp.async benefit depends on compute/load ratio, not just warp count.**
Originally concluded cp.async only helps with ≤ 4 warps — wrong. Refined finding from IGEMM:
- Flash Attention (64 HMMA per tile): cp.async 4–5% slower at 8 warps. Long compute phase gives
  warps enough work to interleave and hide DRAM latency without async help.
- INT8 IGEMM (8 IMMA per tile): cp.async **+35% faster** at 8 warps. Short compute phase means
  8 warps generate only ~128 compute cycles — insufficient to hide ~300-cycle DRAM latency.
  cp.async decouples the load from the warp instruction stream, providing additional hiding.
Rule: cp.async benefits scale inversely with compute/load ratio per tile.

**Insight 3: The 50 KB smem cliff is a concurrency multiplier on GA104.**
GA104 has 100 KB max configurable smem per SM. The threshold is:
- ≤ 50 KB per block → up to 2 blocks per SM → 8 warps (good)
- > 50 KB per block → 1 block per SM → 4 warps (bad: 2× more DRAM stall exposure)
(Confirmed: 48 KB → 2 blocks, 56 KB → 1 block.)

Any optimization that crosses 50 KB by adding smem (double-buffering, Bc=128) potentially
halves warp occupancy and regresses performance even if arithmetic intensity improves.

---

### On Tensor Core Scheduling

**Insight 4: HMMA.16816 on Ampere has an 8-cycle issue interval per warp.**
The Ampere Tensor Core unit processes 1 HMMA per 8 cycles from the same warp's perspective.
This is a pipeline depth constraint, not a result latency. Two consecutive HMMAs writing to
independent register groups still require S08 between them from the same warp — S01 is
only valid after 8+ cycles have elapsed since the last HMMA (satisfied naturally when other
instructions interleave, e.g., LDSM).

**Insight 5: nvcc's HMMA stall counts are correct; IMMA stall counts are conservative.**
CuAssembler analysis of flash_attn_br16 and wmma_gemm_conv showed that S08 stalls between
HMMA instructions match the hardware TC pipeline requirement. But IMMA (INT8) stalls are
set to S04 conservatively — S02 is optimal (+1.6%), S01 is correct but hurts scheduling.

**Insight 5b: IMMA and HMMA have fundamentally different pipeline constraints.**
HMMA: 8-cycle hardware pipeline depth (structural, not reducible).
IMMA: no fixed pipeline constraint — S04 is compiler conservative, caused by register
read port contention when consecutive IMMAs share the same A-fragment operand.

**Insight 6: The WMMA GEMM inner loop pattern to emulate:**
```sass
LDSM R44, [addr_i]          W0 S02   ← load B[i] async (scoreboard set when done)
HMMA Rout_a, Ra, R44, Racc  B0 S08  ← wait B0, uses R44 from CURRENT LDSM
HMMA Rout_b, Ra, R46, Rbc   B- S01  ← uses R46 from PREVIOUS LDSM (already ready)
LDSM R44, [addr_{i+1}]      W0 S07  ← start next LDSM, 7 cycles after second HMMA
```
This achieves near-maximum utilization: each LDSM overlaps with 2 HMMAs.

---

### On Memory Hierarchy

**Insight 7: Re-reads multiply effective DRAM traffic non-linearly.**
- Direct conv2d (3×3): reads X 9× = 9× effective traffic → ~300 GFLOPS
- im2col + WMMA: reads X once, writes col, reads col = 3× X worth of traffic → 6,262 GFLOPS
- Implicit GEMM: reads X once during tile loading → 2× X reduction → 6,687 GFLOPS

**Insight 8: Implicit GEMM speedup scales with col-buffer/L2 ratio.**
When col buffer < L2 (4 MB), explicit im2col round-trips are partially L2-cached.
Speedup from implicit GEMM is small (1.07× at 23.6 MB... but still measurable).
When col buffer >> L2, every explicit round-trip hits DRAM. Speedup grows to 1.89×
at 47.2 MB. The crossover point is approximately col_buffer ≈ L2 size.

**Insight 9: Precomputed coordinate tables eliminate per-element index arithmetic.**
The naive implicit GEMM requires 6 integer divisions per A-tile element (M-dim and K-dim
decoding). At 1024 elements/tile, that's 6144 divisions per K-tile — catastrophically slow.
Precomputing (n, out_h, out_w) per M-row once per block and (cin, kh, kw) per K-col
once per K-tile reduces per-element work to 2 adds + 2 comparisons. Cost: 960 bytes of
shared memory. This technique generalizes to any gather-pattern GEMM.

---

### On Algorithm Selection

**Insight 10: The fastest kernel isn't always the most "compute-dense" one.**
- Bc=128 doubled HMMA count per tile AND halved tile iterations. Still 20% slower.
  Occupancy drop from 8→4 warps was the real bottleneck, not arithmetic.
- cp.async double-buffered pipelining is supposed to maximize overlap. Still 5% slower.
  Warp interleaving already provided the overlap; cp.async only added overhead.

**Insight 11: Flash Attention's next frontier is the grid layout, not the tile size.**
With Bc=64, Br=64, D=64:
- Compute: 64 HMMA/tile × (seq/64) KV tiles × (seq/64) Q blocks = O(seq²) total
- DRAM: K/V loaded (seq/64) times per Q block, one Q block per SM per pass
- K/V DRAM traffic: seq² / 64 loads of K/V, where K/V = seq × D × 2 bytes
  → Total K/V DRAM: seq³ × D × 2 / 64 (at seq=1024: 1024³ × 64 × 2 / 64 = 2 GB equivalent)

Split-Q reorders to: each block processes all Q for ONE KV tile. K/V is loaded once per
block and shared across all Q positions → K/V traffic: seq × D × 2 per block (seq/64 blocks)
= seq² × D × 2 / 64 total. Same total but K/V fits in L2 during processing.

---

### On the CuAssembler Workflow

**Insight 12: CuAssembler roundtrip is stable on sm_86 but nvdisasm must be in PATH.**
```python
import sys, os
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
sys.path.insert(0, "/path/to/CuAssembler")
from CuAsm.CubinFile import CubinFile
CubinFile("kernel.cubin").saveAsCuAsm("kernel.cuasm")
```
The `.cuasm` format encodes control codes as `[B0-----:R-:W0:-:S08]` where:
- `B0-----`: wait for barrier 0 to be set (set by a prior instruction's W0 field)
- `R-`: no read dependency
- `W0`: this instruction sets barrier 0 when its output is ready (used by LDS/LDSM/LDG)
- `S08`: unconditional minimum stall of 8 cycles before issuing the next instruction
- `Y`: yield hint (switch to another warp if possible)

**Insight 13: On GA104, shorter inner loops beat higher per-warp compute density.**
Confirmed across 4 experiments: Bc=128 (17-20% slower), split-Q (partial buffer overhead),
register-blocked IGEMM 128×128 (0.84× vs tiled 64×64), WMMA 128×128 (16 mma_sync too long).
With only 8 warps/SM, a long compute burst from one warp blocks the scheduler from interleaving
others. The 64×64 tile / 4-mma_sync loop lets warps switch frequently enough to hide DRAM stalls.

**Insight 14: Software pipelining (double-buffer) is the biggest single optimization for IGEMM.**
Double-buffered smem with cp.async: +35% over hand-tuned tiled baseline at 4096³ (20,688 vs
15,341 TOPS). Even the LDG-register variant gives +18%. The key: IGEMM's short compute phase
(8 IMMA per tile) leaves most cycles idle waiting for DRAM. Overlapping load(N+1) with
compute(N) fills the bubble. This is bigger than all previous IGEMM optimizations combined
(tiling: +38%, hand-tuning S02: +1.6%).

**Insight 15: IMMA stall-tuning doesn't help pipelined kernels.**
The cp.async pipelined IGEMM has 8 IMMA at S04 in its second K-step. Reducing 6 of
them to S02 (the 7th protects a PRMT→R93 dependency; all-7 edit fails correctness)
saves 12 cycles per tile but shows 0% improvement at 4096³. The IMMA block is ~2% of
the total ~500-cycle loop body (cp.async + LDS interleaving + barriers dominate). Contrast
with the tiled IGEMM where the same S04→S02 gave +1.6% because IMMA was a larger
fraction of a simpler loop. **Rule: stall-tuning only helps when the tuned block is the
dominant cost of the inner loop.**

**Insight 16: BK=64 is 5% slower than BK=32 for pipelined cp.async IGEMM.**
Doubling BK from 32→64 doubles IMMA per tile (8→16) and smem (8→16 KB, still under
50 KB cliff). Result: 19,725 TOPS vs 20,849 TOPS (-5.4%). Same mechanism as Insight 13:
longer inner loops reduce scheduling flexibility at 8 warps/SM. Even with cp.async overlap,
the warp schedulers have enough work to hide latency with BK=32's shorter blocks.
Half the loop iterations (K/64 vs K/32) also means less pipeline ramp-up amortization.
**The "short loops at 8 warps" rule now holds across tiled, register-blocked, and pipelined
kernels — it's structural, not incidental.**

**Insight 17: 1 block × 8 warps escapes the 50 KB smem cliff.**
The "50 KB smem cliff" was self-imposed by the 2-blocks/SM × 4-warps assumption.
1 block/SM × 8 warps/block = same 8-warp occupancy, but 100 KB smem instead of 50 KB.
This unlocks 128×128 (+18%, 24,533 TOPS) and 128×256 (+32%, 27,591 TOPS). The prior
128×128 failure (igemm_register_blocked, 0.84×) was caused by 4 warps with no pipelining,
not by the tile size. 256×256 collapses (-56%) due to register spill (255 regs + stack).
**128×256 is the optimal tile: 210 regs, no spill, 192 ops/byte arithmetic intensity.**

**Insight 18: Per-channel dequantization epilogue is faster than wmma::store_matrix_sync.**
Adding per-channel asymmetric quantization to the cp.async IGEMM epilogue (INT32 acc →
shared memory → explicit (row,col) indexing → per-channel scale/zp → coalesced global
write) gave +3.1% over the symmetric version that uses wmma::store_matrix_sync directly
to global memory. The improvement comes from write coalescing: explicit element-by-element
writes with `elem/16, elem%16` produce perfectly coalesced 16-wide stores, while
wmma::store_matrix_sync may scatter writes due to the fragment-to-memory mapping.
**Rule: when the epilogue needs per-element transforms, routing through shared memory
with explicit indexing can be faster than the WMMA store intrinsic.**

**Insight 19: FFMA stall counts in direct kernels ARE often conservative.**
The direct conv2d's 310-FFMA inner loop has S03-S04 between many independent FFMAs
that could use S01 (FFMA result latency is 4 cycles; S04 is already correct for dependent
pairs, but independent pairs could go to S01). ~150 pairs × 3 saved cycles = 450 cycles/block.
This optimization is valid but the kernel is now superseded by implicit GEMM + WMMA.

**Insight 20: Persistent kernel with L2 tile reuse doesn't help IGEMM on GA104.**
Persistent grid (48 CTAs, 1/SM, atomic work-stealing) with column-first tile ordering
(m varies fastest → CTAs in same column share B K-tiles in L2) measures -0.7% vs
standard grid launch at 4096³ (27,383 vs 27,588 TOPS). Three reasons:
1. **Register overhead**: The work-stealing loop adds +22 registers (234 vs 210) for tile
   index, block coordinates, and loop state. No spills, but reduces compiler flexibility.
2. **L2 already sufficient**: K-tile working set across 48 CTAs = 48 × 12 KB = 576 KB.
   GA104's 4 MB L2 holds this easily even with standard grid launch's scattered access.
3. **No K-step synchronization**: CTAs iterate the K-loop independently; they quickly
   diverge in K-step progress, so "same column" doesn't guarantee they read the same
   K-tile simultaneously. True L2 sharing would require grid-level barriers per K-step.
**Where persistent GEMM WOULD help**: Stream-K (splitting K across CTAs with inter-CTA
reduction), or GPUs with L2 << K-tile working set. Column-first ordering is not enough.

**Insight 21: Online FP16→INT8 quantization makes IMMA a transparent FP16 accelerator.**
Read FP16 from DRAM → per-tile INT8 quantization in smem → IMMA → FP32 running
accumulator (separate per-tile scales). Result: 11,104 effective GFLOPS vs HGEMM baseline
7,831 GFLOPS = **+42% (1.42×)**. INT8 Tensor Cores beat FP16 Tensor Cores for FP16
matrix multiply despite the quantization overhead.
Key design decisions:
1. **Per-tile scales with FP32 running accumulator**: Since scales differ per K-tile,
   INT32 IMMA accumulators are dequantized to FP32 after each tile. This requires
   both INT32 WMMA fragments (64 regs) and FP32 running arrays (64 regs) = 128 regs
   for accumulators alone. Forces 128×128 tiles (not 128×256) to stay under 255 regs.
2. **Block-wide max_abs reduction**: 3 extra __syncthreads per K-tile (2 for separate
   A/B reductions, 1 after quantization). ~200 cycles overhead per tile vs ~500 cycles
   IMMA = ~40% overhead. The 4× theoretical INT8/FP16 ratio absorbs this easily.
3. **Smem layout**: FP16 double-buffer (32 KB) + INT8 working (8 KB) + epilogue (8 KB)
   = 48 KB exactly (aliased reduction scratch into epilogue to fit under static limit).
**The HGEMM baseline is naive (no tiling, no pipelining). A tiled HGEMM would narrow
the gap, but the online-quant kernel is also using only 128×128 tiles — both have room
to grow. The architectural advantage (4× INT8 throughput) is fundamental.**

**Insight 22: In-place FP16→INT8 quantization saves registers, not just smem.**
The original online-quant kernel used separate INT8 buffers (8 KB smem, 239 regs).
Eliminating them and writing INT8 in-place to the FP16 double-buffer saves 8 KB smem
(48→40 KB) but more importantly drops registers from 239 to 213 — the compiler found
a tighter allocation without the separate buffer pointers and addressing.
Result: 16,646 GFLOPS = **+31% over separate-buffer (12,707)**, **+112% over HGEMM (7,849)**.

Root cause of the original in-place failure (Issue #15): **cross-warp WAR hazard**.
Thread T writes INT8 at byte `idx`, corrupting FP16 element `idx/2`. Within a warp,
SIMT lockstep ensures read-before-write. Across warps, warp scheduling is nondeterministic
— warp 1 can write byte 32 (corrupting FP16[16]) before warp 0 reads FP16[16].
Fix: two-phase quantize (read ALL FP16 to registers, `__syncthreads()`, write ALL INT8).
Cost: 1 extra sync per K-tile × 127 tiles = ~3 μs on 8.3 ms kernel = 0.04%. 8 extra
registers for the INT8 temp buffer, absorbed by the 26-register savings elsewhere.

Key takeaway: **smem layout changes affect register pressure indirectly.** The compiler
allocates registers based on the full program graph. Removing smem buffers can eliminate
addressing computations, pointer arithmetic, and liveness ranges that consume registers —
even when the smem savings themselves don't change occupancy. Always check `cuobjdump
-res-usage` after smem layout changes; the register delta matters more than the smem delta.

---

**Insight 24: smem epilogue benefit is occupancy-dependent — helps at ≤8 warps, hurts at 16.**

A coalesced smem epilogue (accumulator→shared→global, one syncthreads) improves store efficiency
by staging scattered per-thread writes into contiguous rows before writing to DRAM. But the
benefit depends on total smem and warp count:

| Kernel | Epilogue | Smem (KB) | Warps/SM | Effect |
|--------|----------|-----------|----------|--------|
| 8-warp IGEMM (128×128) | smem epilogue | 24 | 8 | **+3.1%** |
| 8-warp HGEMM (smem vs direct) | smem vs direct | 40 vs 32 | 8 | **+1.8%** (noisy) |
| 16-warp HGEMM (128×128) | smem vs direct | 48 vs 32 | 32 | **−1 to −3%** |

At 16 warps, the epilogue adds 16 KB smem (16 warp-tiles × 256 floats × 4 bytes), pushing
total smem from ~32 KB to ~48 KB. Though this stays under the 50 KB cliff for 2 blocks/SM,
the higher smem pressure reduces L1 data cache headroom for coalescing global stores, and
the `__syncthreads()` across 32 warps adds meaningful barrier overhead that isn't present
at 8 warps.

**Design rule:** Use smem epilogue when total smem ≤ 40 KB **and** warps ≤ 8.
Use direct `wmma::store_matrix_sync` (or register-→global direct stores) when warps > 8
or smem budget is tight. At 16+ warps, per-thread direct stores into contiguous output tiles
achieve comparable coalescing without the smem overhead.

---

**Insight 23: Warp specialization (4 IMMA + 4 quant) doesn't help online-quant IGEMM (-3.66%).**
The idea: split 8 warps into 4 IMMA (compute on cur_buf) + 4 quantize (load+convert next_buf),
overlapping quantize with IMMA. Target: save the ~32% per-tile overhead that quantize adds
(20.7 µs/tile, measured as 8.255 ms in-place vs 5.607 ms INT8-only at 128×128).

Result: 16,039 GFLOPS = **-3.66% vs in-place (16,646)**. Three reasons for failure:

1. **Register cliff.** 4 IMMA warps on 128×128 = 16 WMMA tiles per warp = 128 regs for
   running[] + 128 regs for acc[] = 256 regs → guaranteed spill. The workaround: process
   each warp's 4×4 tiles in two 2×4 halves, time-sharing acc[2][4]. This doubled the IMMA
   inner loop (128 IMMA instructions vs 64) and pushed the kernel to 255 regs (vs 213).

2. **Insight 13 strikes again.** With 4 IMMA warps each running a 2-half inner loop
   (32 mma_sync per K-tile per warp vs 16 in the in-place kernel), warp interleaving is
   halved. The 4 quant warps provide some scheduling diversity, but they execute FP32
   scalar code (h2f, fmul, f2i), not IMMA — the warp scheduler can't pipeline them with
   Tensor Core operations.

3. **Instruction footprint.** 11,319 SASS lines vs 7,127 — 59% larger. The divergent
   if/else for IMMA vs quant paths, plus the doubled IMMA loop, bloats the code beyond
   L0 I-cache capacity, causing instruction fetch stalls.

Named barriers (`bar.sync <id>, <count>` PTX) worked correctly for asymmetric synchronization:
128-thread quant-internal barriers plus 256-thread tile-boundary barriers. This technique is
sound — the problem is that the overlap target (quantize time) is swamped by the structural
costs of halving the IMMA warp count.

**Key takeaway: warp specialization requires that the specialized groups maintain equivalent
pipeline utilization.** When one group (IMMA) needs 2× the inner loop to compensate for half
the warps, the cure is worse than the disease. A better target: reduce quantize cost directly
(bank-conflict-free smem access patterns, vectorized INT8 writes) without splitting warps.

---

### The Complete Performance Hierarchy (RTX 3070 Ti, SD UNet params)

```
Conv2d (3×3, N=1, H=W=64, Cin=Cout=320):
  Direct conv2d:           ~25 ms    ~300 GFLOPS    1×
  im2col + WMMA:           1.21 ms   6,262 GFLOPS  20.9×
  Implicit GEMM:           1.13 ms   6,687 GFLOPS  22.1×   ← best

Flash Self-Attention (seq=1024, batch=8, heads=8, D=64):
  Scalar (1-warp):         53 ms     ~330 GFLOPS    1×
  4-warp shared KV:        19 ms     ~920 GFLOPS    2.8×
  Br=16 HMMA (Bc=64):      2.81 ms   6,112 GFLOPS  18.9×  ← best
  Br=16 HMMA (Bc=128):     3.37 ms   5,098 GFLOPS  15.7×  (occupancy cliff)
  Br=16 cp.async pipeline: 2.96 ms   5,795 GFLOPS  17.9×  (overhead)

Cross-Attention (SD 64×64, sq=4096, skv=77, heads=8, D=64):
  cross_attn_br16:         0.246 ms  2,624 GFLOPS  ← best
  cross_attn_pipelined:    0.293 ms  2,202 GFLOPS  (19% slower at this config)

INT8 IGEMM (M=N=K=4096, symmetric quantization):
  Naive (global loads):    12.4 ms   11,100 TOPS   1×
  Tiled 64×64 (compiler):  9.1 ms   15,085 TOPS   1.36×
  Tiled 64×64 (hand-tuned): 8.9 ms  15,376 TOPS   1.39×
  Tiled 64×64 (aggressive): 9.0 ms  15,223 TOPS   1.37×
  Register-blocked 128×128:10.8 ms   12,771 TOPS   1.15×  (inner loop too long)
  Pipelined LDG (dbuf):     7.6 ms   18,114 TOPS   1.63×
  Pipelined cp.async (dbuf): 6.6 ms  20,688 TOPS   1.86×
  cp.async BK=64:            7.0 ms  19,725 TOPS   1.78×  (BK too long, -5%)
  Per-channel asymmetric:    6.4 ms  21,331 TOPS   1.92×
  8-warp 128×128 (1 blk/SM): 5.6 ms  24,533 TOPS   2.21×  (+18% vs 64×64)
  8-warp 128×256 (1 blk/SM): 5.0 ms  27,591 TOPS   2.49×  ← best (4.0% of peak)
  8-warp 256×256:           11.3 ms  12,114 TOPS   1.09×  (reg spill, -56%)
  Persistent 128×256 (L2):   5.0 ms  27,383 TOPS   2.47×  (L2 reuse: -0.7%, +22 regs)

FP16 GEMM via online INT8 quantization (M=N=K=4096):
  HGEMM (naive, FP16→FP32):     17.5 ms  7,849 GFLOPS  1×
  Online-quant v2 (sep bufs):   10.8 ms 12,707 GFLOPS  1.62×  (239 regs, 48 KB smem)
  Online-quant in-place:         8.3 ms 16,646 GFLOPS  2.12×  ← (213 regs, 40 KB smem)
  Warp-specialized (4+4):        8.6 ms 16,039 GFLOPS  2.04×  (255 regs, -3.66% vs in-place)
  Bank-conflict-free:            8.05 ms 17,070 GFLOPS  2.18×  ← (211 regs, +2.5% vs in-place)
```
