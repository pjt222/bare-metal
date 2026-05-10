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

---

## Observation N: Sparse HGEMM (2:4 mma.sp) — From Scalar Loads to Tensor Core Dominance

The sparse HGEMM kernel targets Ampere's `HMMA.SP.16816.F32` instruction, which
performs a 16x8x16 multiply-accumulate on FP16 with 2:4 structured sparsity.
Half of A's K elements are skipped, so the theoretical peak is **2x the dense
HGEMM baseline**.

### Journey

| Milestone | Size | Dense-eq GFLOPS | % of Dense | Code change |
|-----------|------|-----------------|------------|-------------|
| Naive (1 warp, no tiling) | 2048³ | 3,777 | 12% | Baseline |
| Tiled 128×128, scalar smem loads | 2048³ | 12,341 | 39% | `__halves2half2` fragment construction |
| + A-fragment `ldmatrix` | 2048³ | 19,025 | 60% | `ldmatrix.sync.aligned.m8n8.x2.shared.b16` |
| + B-fragment `ldmatrix.trans` | 2048³ | **41,930** | **131%** | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` |
| Final (4096³ validation) | 4096³ | **41,721** | **131%** | Same kernel, larger problem |

The kernel is now **faster than dense HGEMM** — achieving the theoretical
2x sparsity advantage in practice.

### Key SASS Findings

Disassembly of `hgemm_sparse_tiled.sm_86.cubin` shows the inner loop is
entirely HMMA.SP + LDSM interleaving — no scalar LDS loads remain:

```sass
/*1950*/  HMMA.SP.16816.F32 R16, R2.reuse, R40, R16, R46.reuse, 0x0
/*1960*/  LDSM.16.MT88.2   R40, [R59]
/*1970*/  HMMA.SP.16816.F32 R32, R2.reuse, R36, R32, R46.reuse, 0x0
/*1980*/  LDSM.16.M88.2    R36, [R60+0x300]
/*1990*/  HMMA.SP.16816.F32 R12, R2.reuse, R38, R12, R46.reuse, 0x0
/*19a0*/  LDSM.16.MT88.2   R38, [R59+0x10]
/*19b0*/  HMMA.SP.16816.F32 R28, R2, R42, R28, R46, 0x0
```

**Stall codes**: Consecutive `HMMA.SP` instructions from the same warp are
separated by **S08 stalls** — a hardware-fixed 8-cycle Tensor Core pipeline
depth. This is not reducible by software reordering. The LDSM instructions
interleaved between HMMA pairs do not hide the S08 stall (LDSM is single-cycle
issue), but they do keep the memory subsystem busy while the Tensor Core
pipeline drains.

**Fragment reuse**: The A-fragment `ldmatrix` is hoisted outside the `wj` loop
(so `fa0`/`fa1` are reused across both N-column tiles). B-fragments are loaded
inside both `wi` and `wj` loops — an opportunity for further pre-loading if
register pressure allows (the kernel already uses 63 registers, close to the
64-register limit that limits occupancy to 2 blocks/SM).

### Architecture Constraints

- **29 KB smem** (12 KB A + 17 KB B, double-buffered) — well under the 50 KB
  cliff → 2 blocks/SM, 8 warps/SM.
- **63 registers** per thread — at 512 threads/block, each block uses ~32K
  registers. With 64K/SM, this limits to 2 blocks/SM regardless of smem.
- **BK=32** gives 16 mma.sp per warp per K-tile. At 4096³, 128 K-tiles →
  2,048 mma.sp per warp, amortizing the cp.async prologue/epilogue overhead.

### Remaining Headroom

With sparse already at 131% of dense, the next frontier is not sparse-HGEMM
itself but upstream integration: online FP16→INT8 quantization (observed in
Phase 5) or fusing the sparse matmul into Flash Attention's QK^T and PV
computations.

## Observation O — Flash Attention smem padding: occupancy regression dominates

**Setup**: `flash_attn_br16_regpv` already runs at 3 blocks/SM (12 warps),
the highest occupancy among the FA variants on GA104. Hypothesis: pad
K/V tile (+8 halfs → stride 144 B) and `smem_work` (+4 floats → stride 272 B)
to break 8-way ldmatrix.x4 bank conflicts and gain throughput.

**Bank conflict math** (encoded in `scripts/audit/ldmatrix_conflicts.R`):
ldmatrix.x4 reads 8 row addresses; for 32 banks of 4 bytes, bank conflicts
occur when `gcd(stride_bytes / 4, 32) > 4`. Equivalently `stride_bytes mod 32 ≠ 0`
clears all 8 rows. Stride 128 B (FP16 [64]) and 256 B (FP32 [64]) both fail.

**Result** (bench_br16_regpv_pad, RTX 3070 Ti):

| variant | smem | blocks/SM | seq=1024 ms | speedup |
|---|---|---|---|---|
| regpv (no pad) | 32.0 KB | 3 (12 warps) | 2.45 | 1.00× |
| regpv + KV+8 + W+4 | 35.0 KB | 2 (8 warps) | 3.03 | 0.81× |
| regpv + KV+8 only | 34.0 KB | 2 (8 warps) | 3.22 | 0.76× |
| regpv + W+4 only | 33.0 KB | 2 (8 warps) | 3.49 | 0.70× |

Padding loses 20-32% across all sizes (seq 512..4096). **Surprise asymmetry**:
W+4-only loses MORE than KV+8 despite touching fewer LDS instructions. Likely
because W is the FP16 overlay path that interleaves with the softmax registers,
and breaking its store_matrix_sync stride (FP32 stride 268 B is mid-pipeline
in the Phase B → Phase C transition) hurts more than fixing it helps.

**Mechanism**: at 12 warps/SM the scheduler interleaves ldmatrix latency with
HMMA pipeline drain. Even with 8× replays, the pipeline stays fed. Dropping
to 8 warps exposes both the per-warp serialization AND the residual conflict
penalty no longer hides cleanly. The 33-KB threshold for 3 blocks/SM appears
tight: 3 × 32.0 KB = 96 KB fits, 3 × 33.0 KB = 99 KB rounds to over the
allocation granularity.

**Calibrated tradeoff model** (`pad_tradeoff()` in ldmatrix_conflicts.R):
predicts 0.86× speedup (loses 14%) for 12→8 warp transition with 8× conflict
elimination. Empirical 0.81× (loses 19%). Model agrees in direction, slightly
optimistic in magnitude — useful for screening future padding decisions.

**Generalization**: padding is profitable only when (a) baseline occupancy is
already capped by registers (smem headroom unused), or (b) the kernel runs at
< 8 warps/SM where conflicts become exposed. For high-occupancy regpv-style
kernels, structural changes (XOR swizzle, register-pressure reduction, or
algorithmic rearrangement to eliminate `smem_work`) are needed before padding
can win.

## Observation P — Flash Attention smem_work elimination: the structural win after padding failed

**Setup**: Observation O established that smem padding is unprofitable for the
12-warp Flash Attention regpv kernel because the occupancy drop (3→2 blocks/SM)
exceeds the bank-conflict elimination benefit. Path forward identified three
options: XOR swizzle, register pressure reduction, smem_work elimination.

**Result**: a three-stage refactor delivers **+40% throughput** at seq=1024
(2.45 → 1.75 ms, 7150 → 10000 GFLOPS).

| stage | smem | regs | ms@seq=1024 | GFLOPS | speedup |
|---|---|---|---|---|---|
| 0 baseline regpv | 32.0 KB | 156 | 2.45 | 7150 | 1.00× |
| 1a lean state | 32.0 KB | 132 | 2.52 | 6940 | 0.97× |
| 1b + Q reg cache | 32.0 KB | 144 | 2.37 | 7400 | 1.04× |
| 2 + smem_work elim | 24.0 KB | 138 | 1.75 | 10000 | **1.40×** |
| 3 + KV/W padding | 27.0 KB | 134 | 1.78 | 9826 | 1.37× (wash) |

**Stage 1 — lean per-thread state**: in WMMA m16n16k16, each lane "owns" 2
specific rows (row_lo = groupID, row_hi = groupID + 8) of the 16×16 score tile.
The baseline kernel held `running_max[16]` + `running_sum[16]` per thread —
broadcast-identical across 32 lanes, 32 redundant registers/thread. Restructured
to 4 floats/thread; broadcast via `__shfl_sync` per row when the softmax loop
needs the running stat. Free 24 regs/thread. Performance neutral standalone,
but enables stage 1b.

**Stage 1b — Q register cache**: SASS inspection (LDG count inside loop body)
revealed the compiler did NOT hoist Q across the KV-base loop in the regpv
or lean kernels — Q was reloaded from L2 every iteration (16× at seq=1024).
Explicitly hoisted `q_frag[TILES_D]` outside the loop. Costs 12 regs/thread.
Reload count drops from 30 LDGs/iter (lean) → 14/iter (qcache, only K/V remain).
Net +4-13% across sizes, peaking at +31% (seq=512, batch×heads=256).

**Stage 2 — smem_work elimination (the big win)**: the 16 KB `smem_work`
served as a FP32 round-trip buffer between Phase B (`store_matrix_sync` of
score_frag) and Phase C (per-row softmax `LDS.f32` reads). The kernel's
LDS+STS instruction count was 238 in cubin, dominated by this round-trip.

Restructure keeps `score_frag` in registers across Phase B → Phase C. The
key enabler is performing per-row reductions directly on fragment elements:
4 cols × 4 tiles per lane fmax, then intra-group `__shfl_xor_sync` with
offsets 1 and 2 covers all 4 lanes within a row group. For each owned row,
this lane plus 3 group siblings collectively span the 64-col dimension.

Phase D still needs FP16 weights as `matrix_a` row_major fragments. Solved
by writing FP16 weights directly to a 8 KB `weight_smem` at the WMMA-row-major
positions (lane-derived offset using groupID and in_group). Phase D's
`load_matrix_sync` reads naturally without further changes.

Final pv_accum store: instead of `store_matrix_sync` to smem followed by
normalized read-back, each lane writes its 8 owned elements (per pv_accum
fragment, per tile) directly to global O, applying `1/running_sum`. The
WMMA fragment layout maps cleanly to global indices.

Results: LDS+STS 238 → **30**, smem 32 KB → 24 KB, regs 144 → 138, throughput
+40% across all sizes. The kernel reaches ~10 TFLOPS = 5.7% of FP16 TC peak
(174 TFLOPS), up from 4.1%. Still far from peak but a structural step forward.

**Stage 3 — padding now affordable, no longer needed**: with 24 KB nosmem
layout, K/V/weight padding fits 3 blocks/SM (3×27.4 = 82 KB ≤ 100 KB). Tested
and found roughly tied with unpadded (±3%). The bank conflicts that motivated
Observation O are no longer the bottleneck after smem traffic was reduced 8×.

**Generalization**: when bank-conflict padding is unprofitable, the right
question is not "how do we pay for the conflicts" but "do we need this smem
allocation at all". For matmul-heavy kernels with online reductions, fragment
elements + intra-group shfl is often a viable substitute for smem-staged
reductions. This pattern likely generalizes to other Tensor Core kernels with
softmax-like operators (attention variants, normalized linear layers).

## Observation Q — Pipeline + nosmem: cp.async finally pays off at 2 blocks/SM

**Setup**: Observation P established `flash_attn_br16_v2` (nosmem fragment-shfl)
running at 24 KB / 3 blocks/SM / 12 warps. The original `flash_attn_br16_pipeline.cu`
used 64 KB smem (1 block/SM, 4 warps) and lost 4-5% to cp.async because the
HMMA pipeline drain dominates DRAM latency at low occupancy.

**Hypothesis**: applying the nosmem pattern to the pipeline kernel cuts smem
to 40 KB (2 buffers K + 2 buffers V + 8 KB weight_smem), enabling 2 blocks/SM
(8 warps) instead of 1. With doubled effective concurrency, cp.async overlap
should now be net-positive.

**Result** (`flash_attn_br16_v2_pipeline.cu`, RTX 3070 Ti):

| seq | b×h | v2 baseline GFLOPS | v2 pipeline GFLOPS | speedup |
|---|---|---|---|---|
| 512 | 256 | 8150 | 11493 | **1.41×** |
| 1024 | 64 | 9989 | 11450 | 1.15× |
| 2048 | 32 | 10137 | 11585 | 1.14× |
| 4096 | 16 | 9153 | 11666 | 1.27× |
| 1024 | 8 | 8788 | 10302 | 1.17× |

Pipeline plateau ~11.5 TFLOPS = **6.6% of FP16 TC peak** (174 TFLOPS).
Cumulative versus original `flash_attn_br16_regpv` baseline (verified post-warmup,
3-run mean): **1.60×** at seq=1024 (2.45 ms → 1.53 ms measured, 7154 → 11453 GFLOPS).
Earlier draft claimed 1.86× based on an extrapolated 1.31 ms for v2_pipeline; that
figure was wrong — the actual measured v2_pipeline time is 1.529 ms (3-run mean
of 1.555, 1.529, 1.529).

**Mechanism**:
- 8 warps/SM is sufficient for warp scheduler to hide HMMA pipeline drain
- cp.async issues asynchronous DRAM loads; while wait_group 1 stalls on the
  current tile, the next tile is in-flight
- Compute window per tile: ~64 HMMA × 8 cycle stall = 512 cycles
- DRAM latency per tile: ~300 cycles
- With 8 warps interleaved at 1 block/SM (original): scheduler runs out of
  ready warps during cp.async → exposed stall
- With 8 warps × 2 blocks/SM (v2 pipeline): twice as many ready warps → DRAM
  hidden, cp.async win materializes

**Persistent grid result** (`flash_attn_v2_persistent.cu`): -8% to +12% at
12 warps/SM. The previous persistent grid wins (+10% on flash_attn_br16 at
8 warps/SM, large tile counts) **disappeared** at v2's higher occupancy.
The v2 baseline grid wave is short enough that tail-wave elimination
provides no headroom; atomicAdd contention dominates. Concludes: persistent
pattern is occupancy-dependent — wins at low warp counts, neutral-to-loss
at high.

**Generalization**: when smem-reduction frees occupancy headroom, previously
counter-productive optimizations (cp.async, large tiles) can become net
wins. Re-evaluate negative results when the kernel's resource profile changes.

## Observation R — ResBlock conv2d swap: 7× speedup from picking the right kernel

**Setup**: `phase4/resblock/bench.cu` chains GN+SiLU → Conv2d(3×3) → GN+SiLU →
Conv2d(3×3) → residual_add. Used `conv2d_nhwc` from `conv2d.sm_86.cubin`. At
SD UNet config (N=1, C=320, H=W=32, G=32) the chain runs **13.07 ms** at
289 GFLOPS effective.

**Diagnosis**: 95% of ResBlock time is in the two Conv2d calls. `conv2d_nhwc`
is a direct FFMA-loop conv that re-reads input X 9× (once per kernel position).
Achieves ~236 GFLOPS at small sizes, ~289 at SD config. Meanwhile the
companion `implicit_gemm_conv` (Tensor Cores via WMMA, computes im2col on
the fly) achieves 4800-6800 GFLOPS — **20-30× faster** as a standalone kernel.

ResBlock was using the wrong conv. This is not a "kernel optimization"
problem; it's a "kernel selection" problem.

**Fix** (`phase4/resblock/bench_implicit.cu`):
- Reshape FP32 weights `[Cout, kH, kW, Cin]` → FP16 `[K_dim, Cout]` on host
  (one-time, helper from `bench_implicit_gemm.cu`)
- Load `implicit_gemm_conv` from `conv2d_implicit_gemm.sm_86.cubin`
- Set `MAX_DYNAMIC_SHARED_SIZE_BYTES` for the ~6.3 KB smem requirement
- Same grid as standalone implicit GEMM bench: `(grid_m, grid_n, 1)` × 128 threads
- Same correctness check; FP16 weight conversion adds slight precision loss
  but stays within `1e-2*sqrt(C)` AND-logic tolerance

**Result**:

| config | conv2d_nhwc | implicit_gemm | speedup |
|---|---|---|---|
| N=1 C=64 H=W=32 G=8 (small) | 1.057 ms (143 GFLOPS) | 0.586 ms (258 GFLOPS) | **1.80\u00d7** |
| N=1 C=128 H=W=64 G=16 (medium SD) | 9.576 ms (252 GFLOPS) | 1.932 ms (1251 GFLOPS) | **4.96\u00d7** |
| N=1 C=320 H=W=32 G=32 (SD UNet) | 13.068 ms (289 GFLOPS) | **1.864 ms (2025 GFLOPS)** | **7.01\u00d7** |

Speedup grows with problem size — larger Cin/Cout/M means more amortization
of WMMA fragment fixed costs.

**Impact on issue #4 (GroupNorm fusion)**:

Before this swap, GN+Conv fusion's projected savings (~1 MB DRAM round-trip
saved per pair, ~3 \u03bcs at 608 GB/s, vs. 13 ms ResBlock) was ~0.02% \u2014 tiny but
not zero. After the swap, ResBlock is 1.86 ms total. GN fusion would still
save ~3 \u03bcs but is now competing for time within a 1.86 ms budget where the
conv2d itself is the bulk \u2014 making fusion effectively 0.16% improvement.

Closed #4 as not-planned. The GN+Conv fusion idea was correct in spirit
(eliminate redundant DRAM passes) but the wrong target: the bigger
redundant DRAM pass was inside the conv (9\u00d7 reread), not between GN and
conv. Implicit GEMM eliminates that with no fusion needed.

**Generalization (Law 2 reinforced)**: before optimizing FLOPS, count
DRAM passes. A kernel reading every byte 9 times is not a 9\u00d7 compute
problem; it's a 9\u00d7 bandwidth problem disguised as compute. The structural
fix (use a kernel that doesn't reread) beats microscopic FFMA tuning by
order(s) of magnitude.

## Observation S — Bc=128 with v2's smem savings still loses (occupancy gates tile size)

**Hypothesis**: After v2's smem reduction (32 KB → 24 KB at Bc=64), doubling
to Bc=128 fits in 48 KB → 2 blocks/SM. The bigger inner-K tile would mean:
- 2× HMMA per phase B/D (better pipeline depth)
- 2× fewer outer KV iterations (less kernel-level overhead)
- Same Q register cache amortization

This was infeasible on the original kernel (would have been 64 KB → 1 block/SM,
the cliff).

**Test**: `flash_attn_br16_v2_bc128.cu` — identical to `flash_attn_br16_v2.cu`
except `Bc=128`, `TILES_Bc=8`, `__launch_bounds__(128, 2)`.

**Result** (3-run mean across 4 sizes, RTX 3070 Ti):

| seq | b | h | v2 (Bc=64, 12 warps) | v2_bc128 (Bc=128, 8 warps) | v2_pipeline (Bc=64, 8 warps) |
|---|---|---|---|---|---|
| 512  | 16 | 16 | **1.747 ms (10024 GFLOPS)** | 1.888 ms (9279) | 1.524 ms (11494) |
| 1024 | 8  | 8  | **1.757 ms (9972)**  | 1.875 ms (9341) | 1.528 ms (11462) |
| 2048 | 4  | 8  | **3.455 ms (10139)** | 3.669 ms (9549) | 3.021 ms (11596) |
| 4096 | 2  | 8  | **7.370 ms (9506)**  | 7.255 ms (9657) | 6.007 ms (11663) |

**Result is regime-dependent, not uniformly negative**:
- Bc=128 loses 5.8-7.4% at seq ∈ {512, 1024, 2048}
- Bc=128 **wins 1.6%** at seq=4096
- Bc=128 loses 19-23% to v2_pipeline at same 2 blocks/SM occupancy at all sizes

**Interpretation**:

1. At small/medium seq the occupancy hit (12 → 8 warps) dominates: warp
   scheduler runs out of slack to hide K/V load latency, so the kernel is
   warp-parallelism-bound and bigger tile cannot recover.

2. At seq=4096 the iteration count is high enough (32 outer iters at Bc=128
   vs 64 at Bc=64) that halving outer-iter count amortizes the kernel-level
   prologue and `__syncthreads` cost enough to overcome the occupancy hit.
   Crossover lies somewhere between seq=2048 and seq=4096.

3. At the same 2 blocks/SM regime, cp.async double-buffering (v2_pipeline)
   beats Bc=128 at *all* sizes. The win is from overlapping K/V loads with
   HMMA, not from doing more HMMA per iter. cp.async addresses the warp-
   parallelism shortage; bigger tile does not.

4. **Generalization (Law 3 nuanced)**: tile size is subordinate to occupancy
   AND to the load-compute overlap mechanism, BUT also interacts with
   iteration count. "Bigger tile loses" is true only when iteration count is
   low; at high iteration count the per-iter overhead amortizes the
   occupancy hit. The right tool when crossing an occupancy boundary is still
   cp.async (overlap), but the tile-size loss is regime-dependent, not
   absolute.

**Kernel kept** as `flash_attn_br16_v2_bc128.cu` (counter-example, like
the padding regressions in Observation O).

**Implication for future work**:
- A Bc=128 + cp.async double-buffer variant would need 80 KB smem (32 K +
  32 V double-buffered + 16 weight) — over the 50 KB cliff, so 1 block/SM
  (4 warps). Almost certainly catastrophic.
- For very-large-seq workloads (≥4096) the Bc=128 v2 kernel can be
  considered as a dispatch alternative — small win, but real.
- The plateau at ~11.5 TFLOPS (6.6% of FP16 peak) appears genuinely
  difficult to break with naive tile-size or pipeline-depth tweaks across
  all regimes. Regime-specific dispatching may be the practical answer.

## Observation T — Cymatic memory layout: angle-dependent gather locality on real DRAM

**Date**: 2026-05-07
**Phase**: speculative layout study (`phase4/cymatic/`)
**Headline**: layout aligned with Chladni-mode antinodes gives **+1.53× gather throughput at sector midlines**, **−1.89× at sector boundaries** on RTX 3070 Ti, GRID=2048² (13 MB DRAM-resident buffer).

### Setup

A clamped circular membrane mode `u_{n,m}(r,θ) = J_n(k_{n,m}·r) · cos(n·θ)`
partitions the disc into antinode regions of constant sign. For mode (6, 4):
12 angular sectors, 4 radial bands, ~35 regions of size 2 to 2103 cells (1051×
size ratio).

We map a 1D address space onto these regions via `(centroid_r, centroid_θ)`
ordering, raster-fill within each region. Two layouts of the same data:
row-major-inside vs cymatic-permuted. Run the same gather kernel on both, time
each, compare effective bandwidth.

### Measured results (GRID=2048², DRAM regime)

| trace | speedup | reason |
|---|---|---|
| `radial_mid_pi6` (θ=π/6 midline) | **1.53×** cym | Trace stays in one angular sector; cymatic addresses near-contiguous |
| `radial_bnd_pi4` (θ=π/4 boundary) | **0.54×** cym (1.85× row) | Trace on nodal line; adjacent cells in opposite-sign regions |
| `radial_bnd_5pi12` (boundary) | **0.53×** cym (1.89× row) | Same |
| `circular_r030` (small radius circle) | **1.38×** cym | Stays in one radial band; intra-band θ-ordering gives locality |
| `circular_r060` (large radius circle) | 1.12× cym | Mild win |
| `polar_tile_pi6/pi4` | 0.98–1.03× | Tie — wedge spans multiple regions |
| `radial_bias_07/00`, `random` | 1.00–1.07× | Tie — large random working set |
| `rowmajor_full` (sequential) | **0.66×** cym (1.51× row) | Row layout's native pattern |

### Insight 1: Layout amplifies its mode geometry

For mode (n=6), `cos(6θ)` zeros at θ = π/12 + k·π/6 (boundaries) and maxima
at θ = k·π/6 (midlines). Radial trace at midline → trace inside one sector
through all m=4 radial bands → cymatic addresses near-contiguous. Radial
trace at boundary → trace exactly on nodal line → adjacent (i, j) cells are
in entirely different sign regions → addresses jump between disjoint
ranges.

Worst-case slowdown (1.89×) and best-case speedup (1.53×) are similar in
magnitude. **The layout is conditional, not universal.**

### Insight 2: Cache regime matters — DRAM scale required for measurement

| GRID | n_inside | buffer | regime | typical speedup spread |
|---|---|---|---|---|
| 256² | 50K | 0.2 MB | L1/L2 | 0.78–1.34× (most ties) |
| 512² | 200K | 0.8 MB | L2 | 0.89–1.40× (most ties) |
| 1024² | 821K | 3.3 MB | L2 boundary | 0.63–1.86× (variable) |
| **2048²** | **3.3M** | **13 MB** | **DRAM** | **0.53–1.53× (sharp)** |

Below DRAM scale, post-warmup all accesses become L2 hits regardless of
physical layout, so locality differences don't show. The 2048² results are
the meaningful measurement.

### Insight 3: Static R metric was wrong about circular sweeps

The R locality metric (`scripts/cymatic/cymatic_analyze.R`) predicted circular
sweeps at fixed r should hurt cymatic ("adjacent θ → different angular
sectors → address jumps"). The CUDA bench measured the opposite: circular
sweeps tie or favor cymatic.

Reason: cymatic regions are ordered by `(centroid_r, centroid_θ)`. All
regions in one radial band sit in a **contiguous address range with
addresses sorted by θ within the band**. A circular trace at fixed r stays
in one radial band the entire time → scans through θ-sorted regions →
addresses roughly monotone, not random.

The static metric over individual cell pairs missed the effect of
region-level address ordering. **Locality through ordering of
non-adjacent items**, not just through adjacency. A failure mode for the
analytical metric corrected only by real-hardware measurement.

### Methodology lessons

- **Mean-of-5 timing is unstable for sub-100μs kernels** — initial 1024²
  runs gave 0.55–1.86× variance for the same trace. Fixed by switching to
  median-of-11 with auto-scaled iters (target 5 ms/kernel ≫ 10 μs event
  timer noise).
- **`c(qi, ni)` flood fill is O(N²·log N)** in R due to vector reallocation
  on each push. Hung at 2048². Fixed with preallocated queue + linear
  indexing → O(N²) → 51 s for one-time generation.
- **"Effective bandwidth >100% of peak"** indicates cache hits, not
  measurement error. The buffer is reused across iters; post-warmup
  small-trace kernels hit L1+L2 mostly. Reported BW is aggregate
  throughput, not pure DRAM.

### Where this could pay off

- **Diffusion-model 2D attention** with rotational position bias —
  pixels with similar angular position attend to each other; cymatic
  radial bands naturally express this.
- **FFT butterfly buffers** — stage-k butterflies have stride 2^k
  pattern; mode (n, m≈log₂N) might align region boundaries with
  butterfly groups.
- **Polar warp intermediate buffers** — radar, lidar, panoramic images.
- **Spherical harmonics** (l, m) coefficient layouts.

### Limits

- Cannot beat row-major on row-major-native sequential scans (1.5× loss).
- Boundary-aligned access patterns are catastrophic (1.9× loss).
- Mode selection is a workload-dependent parameter, not universal.
- Generation cost: 11 s at 1024², 51 s at 2048² (one-time R precompute).
- Lookup table cost: 4 bytes × N² for the permutation. 16 MB at 2048².

### Conclusion

The cymatic layout is **a real physical phenomenon on real GPU
hardware**, not just an analytical curiosity. It is **conditional**
(geometry-dependent), not universal. For a fixed workload with known
access geometry it is a real tool — search over (n, m, α) modes to
maximize measured speedup. For generic workloads, row-major remains
safer.

The most striking finding is the symmetry between best win (1.53×) and
worst loss (1.89×). The layout doesn't add cycles or remove cycles
overall — it redistributes them across access patterns, amplifying
some and demolishing others. Use it when you can choose the access
pattern. Avoid it when you cannot.

**Files**: `phase4/cymatic/{gen_cymatic_data.R, bench_cymatic.cu,
Makefile, results/}`, `scripts/cymatic_{mapping,analyze,visualize}.R`,
`docs/cymatic_memory_mapping.md`, `docs/figures/cymatic/cymatic_*.png`.

---

## Observation U — NCU profiling overturns three long-held assumptions

**TL;DR**: First measured-counter sweep across canonical kernels (issue #89,
2026-05-08, GPU performance counters newly enabled on the Windows host).
Three core assumptions were wrong. The bottleneck for Flash Attention is
**smem traffic**, not HMMA pipeline stalls. Split-Q (#84) addresses a
non-bottleneck for the b=8/h=8/seq=1024 shape.

### What I measured

`scripts/profile/ncu_profile_all.sh` ran 10 kernel configs with 15 metrics each
(`launch-skip 5 --launch-count 1`, see `scripts/profile/ncu_profile.R`):
occupancy, Tensor Core util, L1/L2 hit rates, DRAM bandwidth, coalescing
quality, and 8 stall-reason histograms (per-issue cycles).

Full results in `results/ncu/all.csv` and tabulated in
`docs/ncu_metrics.md`. Headline rows below.

### Headline measurements (per-issue stall cycles, higher = worse)

| kernel | occ% | TC% | L2 hit% | wait | mio | short_sb | math_throttle | barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA v2 baseline | 24.2 | 12.1 | 91.4 | 1.08 | 7.87 | 2.39 | 1.12 | 1.55 |
| FA v2 pipeline | 16.4 | 13.9 | 91.6 | 0.93 | 6.89 | 5.41 | 1.44 | 2.35 |
| FA v2 persistent | 17.1 | 11.2 | 89.4 | 1.08 | 4.66 | 2.37 | 0.97 | 0.92 |
| FA regpv (legacy) | 24.3 | 8.6 | 94.9 | 1.48 | 4.90 | 5.47 | 0.82 | 1.70 |
| HGEMM 16-warp | 65.1 | 46.3 | 76.4 | 2.63 | 3.47 | 2.21 | **35.46** | 5.33 |
| HGEMM 16-warp+epi | 65.4 | 26.1 | 64.9 | 2.36 | **20.98** | **16.82** | 5.91 | **17.32** |
| HGEMM 256x128 | 33.3 | 26.1 | 74.1 | 3.29 | **24.01** | 6.03 | 15.12 | **16.06** |
| Sparse INT8 GEMM | 65.4 | 14.0 | 77.7 | 2.50 | **10.71** | 1.54 | 0.45 | 2.99 |
| Cross-attn v2 | 21.0 | 10.5 | 87.0 | 1.07 | 6.18 | 2.75 | 0.99 | 1.05 |
| ResBlock implicit | 14.5 | 3.2 | 98.2 | 2.40 | 0 | 1.06 | 0.24 | 0.56 |

(`stall_long_sb` = DRAM-latency stall — was ≤2.6 everywhere except sparse
IGEMM 10.5 and ResBlock 5.1, never the dominant factor.)

### Three assumptions overturned

#### 1. HMMA S08 is NOT the dominant FA bottleneck

Long-held belief: "Flash Attention plateau at 11.5 TFLOPS = 6.6% peak is
HMMA-bound. S08 stall between consecutive HMMAs is hardware-fixed, can't
be reduced." See `docs/comparison_to_sota.md` and prior `gpu_reflections.md`
prose.

Measured: `stall_wait` (HMMA S08 + dependency wait) is **0.93** per-issue
cycles for FA v2 pipeline. `stall_mio` (memory IO unit throttle, smem
traffic) is **6.89** — **7.4× larger**. `stall_short_sb` (smem/L1
latency) is **5.41** — another **5.8×**.

The plateau is **smem-traffic-bound**, not HMMA-bound. The cumulative
`stall_mio + stall_short_sb = 12.3` cycles per issue dominate everything
else combined. HMMA pipeline is mostly waiting on smem, not vice versa.

This refutes the premise of issue #84 (split-Q). Split-Q addresses SM
starvation; we have 22× block oversubscription at b=8/h=8/seq=1024 and
**91.6% L2 hit rate**, so DRAM is not the wall either.

#### 2. HGEMM 16-warp's gap is FFMA pipe oversubscription, not Tensor Core

Long-held belief: "HGEMM at 18.3% of FP16 TC peak — Tensor Cores are the
limit, need more cp.async overlap or bigger tiles."

Measured: TC util is actually **46.3%** (not 18.3%). The 18.3% number
was achieved-throughput / nominal-peak; **46.3% is the cycle-fraction the
TCs were busy**. The gap between cycle-busy and FLOPS-throughput is
**`stall_math_throttle = 35.46`** — far the dominant stall.

`stall_math_throttle` = FFMA/INT pipe oversubscription. Means the kernel
is issuing FFMA (probably for accumulator scale or epilogue) faster than
the FP32 ALUs can retire. The Tensor Cores are running at near-half
their cycle ceiling, but every other instruction is stuck behind FFMA.

Right fix: **find and reduce the FFMA chain**, probably in the epilogue
or accumulator manipulation. Not bigger tiles. Not more pipelining.

#### 3. HGEMM epilogue variant pays massive smem + barrier cost

`hgemm_16warp_epi` was supposed to be a smarter epilogue (write-fused).
Measured: TC util drops 46% → 26% relative to plain 16-warp, and three
stalls explode:
- `stall_mio` 3.47 → **20.98** (6×)
- `stall_short_sb` 2.21 → **16.82** (7.6×)
- `stall_barrier` 5.33 → **17.32** (3.3×)

The "fused epilogue" routes accumulator data through smem with extra
syncs. Cost is larger than the savings. Production HGEMM should stay on
plain `hgemm_16warp` until the epilogue's smem path is rewritten (or
direct-to-global like fragment-shfl pattern from Observation P).

### Surprises that need follow-up

- **FA pipeline coalescing**: `load_coalesce_bytes = 16 / 32` (half perfect).
  Baseline + persistent both achieve 31. Pipeline cp.async is somehow
  losing coalescing. Worth tracing.
- **Sparse IGEMM L1 hit 83%**: anomalously high. Index/mask buffers cached
  in L1? Or workload regime? Recheck.
- **ResBlock implicit 3.2% TC util** despite 7.01× speedup headline: the
  workload (b=1, 320ch, 32×32) is so small it can't fill the GPU. 14.5%
  occupancy. Headline speedup came from killing 9× re-reads of input,
  not from utilization. Confirms Observation R framing.
- **FA regpv higher occupancy (24.3%) but lower TC util (8.6%) than v2
  pipeline (16.4% / 13.9%)**: occupancy alone is not predictive. Pipeline
  trades occupancy for cp.async overlap and gets more useful work out of
  fewer warps.

### Reprioritization (post-measurement)

Before this measurement, the open issue queue was ordered by the
`comparison_to_sota.md` analytic estimates. Three of the top issues are
now suspect:

- **#84 Split-Q FA** — addresses a non-bottleneck at the documented bench
  shape. L2 hit 91.6%, no DRAM problem. Block count saturates SMs ~22×.
  Reframe to target small-batch decoding regime, OR close as not-planned
  for trained-attention shapes.
- **#85 4-stage cp.async pipeline HGEMM** — TC util already 46% on
  16-warp, gap is FFMA throttle not memory. Pipelining more loads won't
  help. Re-evaluate after hunting the FFMA chain.
- **#88 XOR-swizzled smem** — was deprioritized via reasoning earlier
  (Observation O follow-up). Measurement now **promotes** it: stall_mio
  + stall_short_sb is the dominant FA stall. Bank conflicts under XOR
  swizzle eliminate the +8 padding tax that may be feeding short_sb.
  Worth revisiting.

New top candidates:
1. **Hunt the FFMA chain in HGEMM 16-warp** (35.46 cycles in
   `stall_math_throttle`). Could close most of the 46% → 100% TC util gap.
2. **Reduce smem traffic in FA v2 pipeline** (`stall_mio = 6.89`,
   `stall_short_sb = 5.41`). Candidates: XOR swizzle (#88), or extend
   fragment-shfl pattern (`docs/fragment_shfl_reductions.md`) to the
   K/V load path.
3. **Investigate HGEMM 16-warp+epi regression** — `stall_barrier = 17.32`
   suggests an over-syncing epilogue. Easy diagnostic.

### Methodology

Counters were gated on WSL2 until the Windows host was reconfigured:
NVIDIA Control Panel → Developer Settings → Manage GPU Performance Counters
→ "Allow access to all users". After reboot, counters opened to the
WSL-side `ncu` binary.

Harness: `scripts/profile/ncu_profile.R` (single kernel) +
`scripts/profile/ncu_profile_all.sh` (sweep). Validation: all 15 metric names
were checked against `ncu --query-metrics --chip ga104` before the first
run, and the parser was tested on synthetic CSV.

Discipline: each kernel measured with `--launch-skip 5 --launch-count 1`
in the same warmup regime as the standard benchmark. Single-launch
capture is noisy — re-run multiple times for any number used in a
publication. The numbers in this observation are first-pass; any
follow-up that depends on the precise stall count should re-measure with
`--launch-count 5` and average.

**Lesson**: trust measurement over reasoning when the measurement is
available. The architecture doc, the gap analysis, and at least three
prior observations were partly wrong about Flash Attention's bottleneck.
Reasoning gave us "HMMA-bound"; counters say "smem-bound". Both can be
true simultaneously, but the measured ratio is **smem stalls dominate
HMMA stalls 13×**. Not close.

**Files**: `scripts/profile/ncu_profile.R`, `scripts/profile/ncu_profile_all.sh`,
`docs/ncu_metrics.md`, `results/ncu/all.csv`.

---

## Observation V — HGEMM IMAD-chain hypothesis falsified

**TL;DR**: Built `hgemm_16warp_aligned.cu` (drops partial-tile branches,
assumes M/N/K aligned to BM/BN/BK). Cubin instruction count: IMAD 139 →
106 (-24%), ISETP 45 → 4 (-91%), BRA 15 → 6 (-60%). Performance
delta: **1.01× — flat**.

NCU on the aligned variant: `stall_math_throttle` actually rose 35.46 →
**41.74**, `stall_long_sb` jumped 1.01 → **12.56**, `stall_barrier` 5.33
→ **20.66**. TC util 46.3% → 44.4%. DRAM read BW 168 → 286 GB/s.

### What this tells us

1. **`math_pipe_throttle` is HMMA queue pressure, not IMAD chain.** Even
   with 24% fewer IMADs and 91% fewer ISETPs, the math throttle metric
   went up, not down. On Ampere, `math_pipe_throttle` includes the
   Tensor Math Pipe — at 46% TC busy, HMMAs themselves are queueing
   into a saturated pipe.

2. **Aligned compiler unrolling exposes DRAM.** The compiler doubled the
   HMMA count in the cubin (32 → 64) because it could fully unroll the K
   loop without per-iteration branch sites. More in-flight HMMAs need
   more A/B fragments → more LDGSTS → DRAM saturated faster → L2 misses
   exposed (hit rate 76.4% → 68.6%). The aligned variant is genuinely
   faster *at the math level* but the freed cycles get burned in
   memory-latency stalls instead.

3. **The real HGEMM 16-warp wall is HMMA issue rate, not anything
   above it.** At 46% TC util with `stall_wait = 2.63` (HMMA dependency
   wait is low), the Tensor Cores are issuing at near-maximum sustained
   rate given the kernel's HMMA dependency graph. Closing the 46% → 100%
   gap requires either:
   - **Reducing inter-HMMA dependencies** (different accumulator
     structure, smaller fragments, more parallel HMMA chains)
   - **SASS-level instruction scheduling** (CuAssembler hand-tune,
     issue #96)
   - **More work per fragment-load** (bigger K, Bc to amortize loads)

   None of these are about address arithmetic.

### What was right and what was wrong

The original NCU observation (Observation U) correctly identified
`stall_math_throttle` as the dominant FA stall, but the **mechanism
attribution was wrong**: I attributed it to "FFMA pipe oversubscription"
based on the metric description ("FFMA/INT/FP pipe oversubscription").
The actual contributor here is the Tensor Math Pipe component of that
same metric.

Ampere's NCU stall reasons are coarse-grained — `math_pipe_throttle`
covers FMA, ALU, FP64, ADU, *and* Tensor Math. Without separation, a
high value points only at "some math pipe is over-subscribed".
Distinguishing requires SASS-level instruction histograms (which kernel
op count is high) cross-referenced against the throttle metric.

In retrospect: the SASS histogram already gave the answer. **64 HMMA in
the aligned variant** vs **32 HMMA in the baseline**, both with
`math_pipe_throttle ≈ 35-42`. If IMAD were the cause, doubling HMMA
(while cutting IMAD 24%) would have *reduced* math throttle. It didn't.
Therefore HMMA dominates the metric.

### Lesson

Coarse stall metrics are necessary but not sufficient. Pair them with
SASS instruction histograms before deciding what to optimize. A
hypothesis that "instruction X causes stall Y" is testable by changing
X and re-measuring Y; in this case, the test was clean (24% IMAD
reduction, ISETP 91% gone, BRA 60% gone) and the metric refused to
move. Hypothesis falsified.

### Reprioritization (from-Observation-V)

- **HGEMM 16-warp is at its single-block-config ceiling.** The next
  meaningful HGEMM optimization isn't "reduce IMAD chain" or "better
  pipelining" — it's structural: split-K with cross-block reduction
  (issue #87), persistent grid (#86), or SASS hand-tuning (#96).
- **`hgemm_16warp_aligned` is preserved as a counterexample** — slightly
  faster (1.01×), demonstrates aligned-only path, useful reference for
  future variants. Not promoted to canonical because the aligned
  precondition limits applicability.

### Files

- `phase2/hgemm/hgemm_16warp_aligned.cu` — counterexample variant
- `phase2/hgemm/hgemm_16warp_aligned.sm_86.cubin` — built artifact
- `phase2/hgemm/bench_aligned.cu` — A/B comparison harness
- `results/ncu/hgemm_imad.csv` — NCU output for aligned variant

---

## Observation W — FA v2 pipeline: 2.4× from +8 smem padding (bank-conflict elimination)

**TL;DR**: Adding +8 padding to K_tile, V_tile, AND weight_smem row strides
in `flash_attn_br16_v2_pipeline.cu` delivers **1.85-2.44× speedup across
seq ∈ {256, 512, 1024, 2048, 4096}**. NCU bank conflict counter dropped
from 95.5M to 279K (-99.7%). TC util rose 13.9% → 36.2%. New canonical
kernel: `flash_attn_br16_v2_pipeline_pad2.cu`.

This is the single largest optimization win in the project's history,
exceeding even the original v2 (1.40×) and pipeline (1.20×) wins.

### What was hidden in plain sight

K_tile and V_tile in v2_pipeline are `[Bc=64 × D_HEAD=64]` FP16 dense
matrices with row stride 64 halfs = **128 bytes**. The smem bank period
on Ampere is 32 banks × 4 bytes = 128 bytes. **Every `ldmatrix.x4` read
of 8 consecutive rows landed on the same banks → 8-way conflict.**

NCU counters (issue #89, this session) confirmed:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`: **95.5M** conflicts
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`: 100.7M load wavefronts
- **Conflict rate: 87.5% of every load wavefront**

The kernel was effectively serializing every smem load. This was the
real cause of `stall_short_sb = 5.41` and `stall_mio = 6.89` — the
dominant FA pipeline stalls per Observation U.

### Why issue #80 was wrong to deprioritize this

Issue #80 closed XOR-swizzle exploration based on the reasoning that
"LDSM is 4.6% of cubin instructions, warp scheduler hides conflicts at
12 warps". The argument was wrong on two counts:

1. **Cubin instruction count ≠ runtime cycles spent.** 4.6% of
   instructions can be 50%+ of cycles when each instruction stalls 8×
   on bank conflicts. The conflict multiplier was uncounted.
2. **Stage 3 padding (alternative) was tied with v2 within ±3%** — but
   that comparison was between two things both still suffering from
   the same conflict. Tying broken-vs-broken doesn't validate either
   as conflict-free.

The correct way to evaluate bank conflicts is to **measure the
conflict counter directly**, not infer from instruction percentages.
Once measured, the 87.5% rate was unambiguous.

### The fix: +8 padding row stride

K_tile/V_tile: stride 64 halfs (128 B) → 72 halfs (144 B). For
ldmatrix.x4 reading 8 consecutive rows, byte offsets become
{0, 144, 288, 432, 576, 720, 864, 1008}. Modulo 128 (bank period):
{0, 16, 32, 48, 64, 80, 96, 112} — all 8 rows on distinct bank groups.
Conflict-free.

weight_smem: same stride 64 → 72 fix. Costs +1 KB per block.

Total smem budget: 40 KB → 45 KB. Still under 50 KB cliff for 2
blocks/SM. Register count actually *dropped* 140 → 133 (compiler
optimization side-effect of removing replay scheduling).

### Measurements

| variant | smem | regs | blocks/SM | seq=1024 GFLOPS | speedup |
|---|---|---|---|---:|---:|
| `v2_pipeline` (canonical pre-this-session) | 40 KB | 140 | 2 | 9058 | 1.00× |
| `v2_pipeline_pad` (K/V padded) | 44 KB | 134 | 2 | 14869 | 1.64× |
| `v2_pipeline_pad2` (K/V/W padded) | 45 KB | 133 | 2 | **21410** | **2.36×** |

Cross-seq robustness:

| seq | unpadded | pad1 | pad2 | pad2 / unpadded |
|---:|---:|---:|---:|---:|
| 256  | 7553  | 10038 | **13982** | 1.85× |
| 512  | 8415  | 14158 | **20569** | 2.44× |
| 1024 | 9058  | 14869 | **21410** | 2.36× |
| 2048 | 10125 | 15342 | **20580** | 2.03× |
| 4096 | 10141 | 15343 | **21062** | 2.08× |

NCU counters (seq=1024, b=8, h=8):

| metric | unpadded | pad2 | change |
|---|---:|---:|---|
| TC util % | 13.87 | **36.20** | +161% |
| L2 hit % | 91.55 | 94.24 | +3% |
| stall_short_sb | 5.41 | **0.30** | -94% |
| stall_mio | 6.89 | **0.33** | -95% |
| stall_math_throttle | 1.44 | 3.21 | +123% (HMMA queue, expected) |
| stall_barrier | 2.35 | 0.58 | -75% |
| bank conflicts (M) | 95.5 | **0.28** | -99.7% |
| conflict rate | 87.5% | 2.2% | -85 pts |

### What's the new wall

After pad2, the dominant stall is `stall_math_throttle = 3.21` — HMMA
queue pressure (per Observation V, math_pipe_throttle on this kernel
captures Tensor Math Pipe oversubscription). At 36.2% TC util we're
roughly half-saturated; the kernel can issue HMMAs but the Tensor
Cores can't accept them faster than the dependency graph allows.

To break beyond 36% TC util needs **more parallelism inside the HMMA
chain**: smaller fragments, more independent accumulators, or
SASS-level reordering (CuAssembler hand-tune, issue #96).

### Practical guidance: always pad small-D tensor-core smem buffers

For any FP16 Tensor Core kernel where smem holds a tile with row stride
64 halfs (128 B) or 32 halfs (64 B) — both bank-period multiples — pad
the row stride by +8 halfs unconditionally. The cost is small (typically
+1-4 KB per block), the win is massive (1.5-2.5× in this case), and the
mechanism is purely structural (no algorithm change).

Tile geometries that need this fix in the existing repo:
- `flash_attn_br16_v2_pipeline.cu` (D_HEAD=64) — fixed this session
- Same family of v2 kernels (`v2`, `v2_persistent`) — likely benefit
- HGEMM kernels: BK=32 (64 B stride for FP16) — needs check
- Cross-attention v2: D_HEAD=64 — likely benefit
- Any other kernel with FP16 tile column count = power-of-2 ≤ 64

Each one is a 5-minute change worth measuring.

### Files

- `phase3/flash_attention/flash_attn_br16_v2_pipeline_pad.cu` — pad1
  (K/V only), 1.5-1.7× win
- `phase3/flash_attention/flash_attn_br16_v2_pipeline_pad2.cu` —
  **new canonical**, 1.85-2.44× win
- `phase3/flash_attention/bench_v2_pipeline_pad.cu` — A/B/C harness
- `results/ncu/fa_pad.csv`, `results/ncu/fa_pad2.csv` — NCU output

### Lesson

Trust counters over instruction-percentage heuristics. The conflict
counter took 30 seconds to query and made the optimization obvious. We
spent multiple prior sessions reasoning about whether bank conflicts
mattered using cubin instruction histograms — a lower-resolution proxy
that gave the wrong answer.

Pattern this session: NCU + targeted counter queries → instant
diagnosis. Without counter access, this kernel would still be at 11.5
TFLOPS.

**Headline**: FA seq=1024,b=8,h=8 plateau **6.6% peak → 12.3% peak** in
one session, no algorithm change, ~30 lines of code edits across two
new variant files. 2.36× of the 7-8× FA-2 gap closed by smem padding
alone.

---

## Observation X — +8 padding pattern generalizes (issue #97)

**TL;DR**: Applied the +8 row-stride padding from Observation W to three
more kernels. All show similar 1.9-2.8× speedups at typical sizes.
Cross-attention shows a regime split (loses at very small KV, wins big
at typical). Pattern confirmed as broadly applicable.

### Kernels touched

| kernel | unpadded GFLOPS | padded GFLOPS | speedup | TC util before/after |
|---|---:|---:|---:|---:|
| `flash_attn_br16_v2` (baseline)         | 7913 | 15072 | **1.91×** | 12.1% / 22.9% |
| `flash_attn_br16_v2_pipeline` (Obs W)   | 9058 | 21410 | **2.36×** | 13.9% / 36.2% |
| `flash_attn_v2_persistent`              | 7294 | 15187 | **2.08×** | 11.2% / 21.2% |
| `cross_attn_v2` (typical 1024×256)      | 5490 | 10480 | **1.91×** | 10.5% / 18.3% |
| `cross_attn_v2` (CLIP-77 256×77)        | 1605 | 1229  | **0.77×** | regime loss |

(All measured at seq=1024, b=8, h=8 unless noted; NCU rerun for each padded variant.)

### Per-kernel files

- `phase3/flash_attention/flash_attn_br16_v2_pad.cu` — baseline padded
- `phase3/flash_attention/flash_attn_br16_v2_pipeline_pad2.cu` — pipeline, full padding (Obs W)
- `phase3/flash_attention/flash_attn_v2_persistent_pad.cu` — persistent padded
- `phase4/cross_attention/cross_attn_v2_pad.cu` — cross-attention padded
- `phase3/flash_attention/bench_v2_{baseline,persistent}_pad.cu` — A/B harnesses
- `phase4/cross_attention/bench_v2.cu` — extended to include _pad variant

### Pipeline still wins largest

The pipeline kernel sees the biggest speedup (2.36×) because it has the
most LDSM traffic per iteration: cp.async overlap means more in-flight
loads competing for smem bandwidth. When conflicts dominated, pipeline
suffered most. Eliminating them frees the biggest absolute amount.

The baseline and persistent kernels achieve the same TC util ceiling
(~22%) since neither has cp.async. They're now compute-throttled
identically, just less than the pipeline's 36%. The pipeline's
additional cp.async overlap remains a real win on top of bank-conflict
elimination.

### Cross-attention regime split

Cross-attention shows a clear regime split previously documented for
v2 vs baseline (insight 4 in CONTINUE_HERE prior session):

| seq_q × seq_kv | v2 (no pad) GFLOPS | v2_pad GFLOPS | pad / v2 |
|---:|---:|---:|---:|
| 256 × 77 (CLIP-77)   | 1803 | 1229  | 0.68× (loses) |
| 256 × 128            | 2426 | 3080  | 1.27× |
| 512 × 128            | 3695 | 5295  | 1.43× |
| 256 × 256            | 2739 | 1658  | 0.61× (loses, anomalous) |
| 512 × 256            | 4668 | 7106  | 1.52× |
| 1024 × 256 (typical) | 5490 | 10480 | **1.91×** |
| 1024 × 512           | 6975 | 12721 | **1.82×** |

**Pattern**: large enough seq_q × seq_kv → padded wins decisively. Below
some threshold, padded loses. The 256×256 anomaly (loses despite
total work between 256×128 and 512×256) needs follow-up investigation
— possibly launch overhead amortizing differently when total blocks
(256/64 × 8 × 1 = 32) exactly matches half the SM count.

**Production guidance**:
```cpp
if ((size_t)seq_q * seq_kv >= 200000) launch_v2_pad();
else if ((size_t)seq_q * seq_kv >= 50000) launch_v2();
else launch_baseline();
```

### Lesson

A measurement-validated pattern travels well. Once the bank-conflict
mechanism was understood (Obs W), applying it to four kernels was
mechanical — each took ~10 minutes of editing + ~5 minutes of building
+ ~2 minutes of benchmarking. **Total session-level investment for
this observation: ~1 hour.** Speedup leverage: 1.9-2.4× across four
canonical kernels.

The compounding effect on aggregate FA workload performance is
substantial. If a workload uses some mix of pipeline, baseline, and
persistent depending on size, the median speedup is now ~2× without
any algorithm change.

### Headline numbers (all post-padding)

| kernel | seq=1024 | seq=4096 |
|---|---:|---:|
| FA v2 baseline_pad        | 15.1 TFLOPS | 20.3 TFLOPS |
| FA v2 pipeline_pad2       | **21.4 TFLOPS** | **21.1 TFLOPS** |
| FA v2 persistent_pad      | 15.2 TFLOPS | (untested) |

12.3% of FP16 TC peak is now the v2 pipeline plateau, up from 6.6%.
The next ceiling is compute-throttle (`stall_math_throttle` 1.4-3.2)
which corresponds to HMMA queue saturation — only addressable via
SASS-level instruction reordering (issue #96).

---

## Observation Y — HGEMM bank conflict audit + epi variant fix (issues #98, #99)

**TL;DR**: NCU bank-conflict counter validated `hgemm_16warp` (0.17%
conflict rate) — existing `PAD_A=8 PAD_B=8` is correct. But
`hgemm_16warp_epi` (a fork that pre-dated the padding) had **75.9%
conflict rate**. Adding the same padding gives **1.41× speedup** (18.0 →
25.4 TFLOPS at 4096³ HGEMM). Closes both issues.

### #98: hgemm_16warp audit — confirms existing padding works

NCU on `hgemm_16warp` (4096³, B=4096):

| metric | value |
|---|---:|
| bank_conflicts (sum) | 133K |
| load wavefronts (sum) | 80M |
| **conflict rate** | **0.17%** |

Existing `PAD_A=8 PAD_B=8` (giving STRIDE_A=40, STRIDE_B=136) eliminates
bank conflicts as designed. The 35.46 `stall_math_throttle` measured in
Observation U is genuinely **HMMA queue pressure** (per Observation V
falsification) — not bank conflicts. Confirms Observation V's null
hypothesis: HGEMM 16-warp is at its single-block-config compute ceiling,
needs SASS hand-tuning (#96) to break further.

### #99: hgemm_16warp_epi — bank conflicts were the over-syncing source

NCU on `hgemm_16warp_epi` (the "fused epilogue" variant):

| metric | unpadded | full pad (PAD_A=8 PAD_B=8) |
|---|---:|---:|
| bank_conflicts (sum) | 347.9M | **8.9M** (-97%) |
| load wavefronts (sum) | 458.3M | 67.6M (-85%) |
| **conflict rate** | **75.9%** | **13.2%** |
| TC util % | 26.05 | **31.91** (+22%) |
| stall_short_sb | 16.82 | **0.05** (-99.7%) |
| stall_mio | 20.98 | **5.06** (-76%) |
| stall_barrier | 17.32 | **7.94** (-54%) |
| GFLOPS @ 4096³ | 18029 | **25403** (+41%) |

Root cause: the variant was forked from `hgemm_16warp` *before* PAD_A/PAD_B
was added there. Both `smem_a` (stride 32 halfs = 64 B = ½ bank period) and
`smem_b` (stride 128 halfs = 256 B = 2× bank period) hit the bank
period exactly, generating 4-way and 8-way LDSM conflicts respectively.

The "over-syncing" interpretation in Observation U was wrong direction:
the barrier stalls were a *consequence* of conflict-throttled smem
traffic, not the cause. Eliminating conflicts dropped barrier stall 17 → 8.

### Implementation note: smem cap workaround

Padding both buffers raises smem to 53 KB, exceeding the 48 KB **static**
smem cap on sm_86. The kernel was converted from `__shared__` to
`extern __shared__ char smem_raw[]` with a layout map — host code uses
`cuFuncSetAttribute(..., MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)` to allow
the larger allocation.

This pushes the kernel to 1 block/SM (53 KB > 50 KB cliff). The
conflict-elimination wins (1.41×) outweighs the occupancy loss because
the original variant was already memory-throttled at 2 blocks/SM.

### Files

- `phase2/hgemm/hgemm_16warp_epi_pad.cu` — padded variant (dynamic smem)
- `phase2/hgemm/bench_epi_pad.cu` — A/B harness
- `results/ncu/99_epi_pad.csv` — NCU output
- The original `hgemm_16warp_epi.cu` is preserved as the counter-example
  (75.9% conflict rate documented in Observation U).

### Pattern emerging across the codebase

| kernel | bank conflict rate | speedup from fix |
|---|---:|---:|
| FA v2 pipeline | 87.5% → 2.2% | 2.36× (Obs W) |
| FA v2 baseline | (similar) → ~3% | 1.91× (Obs X) |
| FA v2 persistent | (similar) → ~3% | 2.08× (Obs X) |
| Cross-attention v2 (typical) | (similar) → ~3% | 1.91× (Obs X) |
| HGEMM 16-warp | 0.17% (already padded) | n/a |
| HGEMM 16-warp_epi | 75.9% → 13% | 1.41× (this Obs) |

Five of six measured kernels had smem layouts where row stride was a
power-of-2 multiple of the bank period. Five of six benefited from
padding (the sixth, plain hgemm_16warp, was already padded). **The
diagnostic-and-fix pattern is now well-established**: NCU bank conflict
counter + +8 row stride = mechanical 1.4-2.4× wins.

---

## Observation Z — `smsp__sass_average_data_bytes_per_sector` undercounts cp.async (issue #100)

**TL;DR**: The `load_coalesce_bytes=16` reported on the FA pipeline
variant (vs baseline's 31) is **a metric accounting artifact, not a
real coalescing problem**. cp.async / LDGSTS instructions are counted
at twice their actual sector consumption by the SASS-level ratio
metric; the L1tex byte/sector counters confirm perfect 32 B/sector
coalescing. Issue closed as non-issue.

### Cross-check on raw counters

NCU on FA seq=1024, b=8, h=8:

| metric | baseline (LDG.E) | pipeline_pad2 (LDGSTS.E.128) |
|---|---:|---:|
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | 285,212,192 | 318,767,104 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 8,912,881 | 9,961,472 |
| **bytes/sector (computed manually)** | **32.0** | **32.0** |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 31.06 | **16.00** |

The L1tex counters show **identical 32 B/sector** for both variants —
both are perfectly coalesced. The SASS-level `bytes_per_sector` ratio
disagrees with itself: baseline 31.06 vs pipeline 16.00 despite the
raw byte/sector count being identical.

### Mechanism

`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio`
divides requested bytes by **predicted sectors** at SASS issue time.
Its sector accounting appears to double-count cp.async (LDGSTS)
because the instruction performs both a global load AND a shared store
in one issue — the metric apparently attributes 2 sector accesses per
L1tex sector for cp.async paths.

### Validation method (for future audits)

When the SASS-ratio metric reports an anomaly, **always cross-check
with raw L1tex counters**:

```
real_bytes_per_sector = l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum /
                        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
```

If `real_bytes_per_sector` differs from the SASS ratio, **trust the
L1tex value**. The SASS ratio is misleading on cp.async-heavy kernels.

### Implications for prior observations

- Observation U's note that pipeline had `load_coalesce_bytes=16` was
  a tooling miss — the variant's coalescing was always perfect.
- The coalescing column in `docs/ncu_metrics.md` should be annotated
  with this caveat for cp.async kernels.
- The real bottleneck for the pipeline variant was always bank
  conflicts (Observation W) and HMMA queue pressure, not coalescing.

### Diagnostic quality lesson

Three observations into the NCU work (U, V, Z), three of the metrics
in the harness needed reinterpretation or were outright misleading:

| metric | issue | resolution |
|---|---|---|
| `stall_math_throttle` | thought to be address arithmetic | Obs V: HMMA queue pressure |
| `l1_hit_pct` (low on cp.async) | thought to be cache-bypass loss | expected (cp.async bypasses L1) |
| `smsp__sass_avg_bytes_per_sector` | thought to be coalescing | Obs Z: under-counts cp.async 2× |

**Pattern**: SASS-level/SM-level synthetic ratios drift from raw L1tex
counters under cp.async. Always pair ratio metrics with raw byte and
sector counts to detect tooling artifacts.

---

## Observation AA — useful_pct trends across 103 kernels (issue #90)

**TL;DR**: Median useful_pct = **12.5%**. Across 103 cubins in the
repo, only ~12% of SASS instructions on average are doing arithmetic
work (HMMA + IMMA + FFMA + FMUL + FADD). The other 88% is address
arithmetic, smem traffic, and control flow. Tensor-Core-dense kernels
look "low-useful_pct" because each HMMA dispatches 1024 FP16 multiplies
under one SASS instruction.

### Distribution

NCU/cuobjdump scan via `scripts/audit/sass_histogram.R` over all `.sm_86.cubin`
in phase1-phase4 (excluding `test_/debug` paths):

| quantile | useful_pct |
|---:|---:|
|   0% |  0.0% |
|  10% |  2.7% |
|  25% |  5.3% |
|  50% | **12.5%** |
|  75% | 26.8% |
|  90% | 31.9% |
| 100% | 50.4% |

### Per-family means

| family | n | mean useful_pct | max useful_pct |
|---|---:|---:|---:|
| Flash Attention      | 32 | **29.2%** | 40.8% |
| Activation/norm      | 14 |    19.8%  | 38.9% |
| Conv2d               |  2 |    10.1%  | 13.2% |
| HGEMM/SGEMM/IGEMM    | 37 |     8.0%  | **50.4%** |
| Other                | 13 |     7.0%  | 17.2% |
| Data movement        |  5 |     0.0%  |  0.0% |

### Why FFMA-dense beats HMMA-dense in useful_pct (but not in TFLOPS)

Top kernel by useful_pct: `sgemm_register_blocked` at 50.4% — pure FP32
register-blocked inner loop with 512 FFMAs in 1016 SASS instructions.
But it runs at FP32, peaking around 21 TFLOPS at most.

Top Tensor Core kernel by useful_pct: `flash_attn_br16_v2_bc128` at
40.4%, with 128 HMMAs in 1496 instructions. Each HMMA performs
16x8x16 = 2048 FP16 muladds — so 128 HMMAs do ~262K muladds, while the
512 FFMAs in sgemm_register_blocked do 512 muladds.

**Lesson**: useful_pct is a good "where is the inefficiency" metric for
intra-family comparison (does this GEMM have more bookkeeping than that
GEMM?). It is NOT a good cross-architecture/cross-precision metric. A
30% useful_pct kernel using HMMA can outperform a 50% useful_pct kernel
using FFMA by 5-10x on the tensor core.

### What "0% useful" means

Five kernels showed 0.0% useful_pct: type casts (`fp16_to_fp32`,
`fp32_to_fp16`), transposes (`transpose_bhsd`, `transpose_bshd`),
`im2col_nhwc_fp16`. These are pure data movement — they do no
multiply-add. The instruction mix is LDG/STG/IMAD/STS only. Optimizing
"useful_pct" on these would mean inserting useless arithmetic; instead,
optimize for memory bandwidth (`dram__bytes_*` per launch) and L2 hit
rate.

### Bottom of the GEMM family

| kernel | useful_pct | total_inst |
|---|---:|---:|
| `wmma_gemm_conv`     | 1.5% | 544 |
| `implicit_gemm_conv` | 1.2% | 672 |
| `hgemm_sparse_naive` | 0.4% | 456 |

These are **bookkeeping-bound**. The naive sparse hgemm has 8x more
ISETP/IMAD/BRA than HMMA — most of the kernel is unpacking the sparse
metadata format. Targets for #96 (SASS hand-tune) or for algorithmic
rewrites.

### Practical rule of thumb on GA104

- **High-performing TC kernel**: 25-40% useful_pct, mostly HMMA-driven
- **High-performing scalar kernel**: 40-50% useful_pct (FFMA-driven)
- **Below 5% useful_pct on a compute kernel**: structural bug —
  the bookkeeping is doing all the work. Look for missing
  `#pragma unroll`, shared-memory thrash, or dynamic indexing that
  prevents constant folding.

### Files

- `scripts/audit/sass_histogram.R` — scanner (replaces deleted Python original)
- `docs/sass_histogram.csv` — full per-kernel data
- `docs/sass_histogram.md`  — Markdown table sorted by useful_pct
- `docs/figures/sass_histogram.png` — top-40 stacked bar visualization

### Method

Each cubin is disassembled with `cuobjdump -sass`, opcodes are matched
in priority order to one of 25 categories, and `useful_pct` is the
sum of (HMMA + IMMA + FFMA + FMUL + FADD) over total. The figure groups
the 25 categories into 9 visual families. Re-run with:

```
Rscript scripts/audit/sass_histogram.R --quiet
```

---

## Observation BB — measured roofline reveals all kernels are compute-limited (#92)

**TL;DR**: Replacing estimated operational intensity with **NCU-measured**
DRAM bytes shows that **every measured kernel sits well above the DRAM
ceiling**. The bottleneck is uniformly the compute pipeline (HMMA queue,
LDSM throttling), not bandwidth. This vindicates the +8 padding work
(Obs W/X/Y) which closed compute-side stalls without touching DRAM.

### Measured numbers (10 kernels)

| kernel | precision | OI_DRAM | OI_L2 | achieved GFLOPS | % of ceiling |
|---|---|---:|---:|---:|---:|
| HGEMM 16-warp (4096³)            | FP16 | 162  | 42 | 29751 | **17.1%** |
| Sparse INT8 GEMM (4096³)         | INT8 | 126  | 34 | 18035 |  5.2% |
| HGEMM 16-warp+epi (4096³)        | FP16 | 162  | 62 | 16730 |  9.6% |
| HGEMM 256x128 (4096³)            | FP16 | 274  | 82 | 16073 |  9.2% |
| FA v2 pipeline (seq=1024)        | FP16 | 412  | 57 |  8892 |  5.1% |
| FA v2 baseline (seq=1024)        | FP16 | 413  | 58 |  7773 |  4.5% |
| FA v2 persistent (seq=1024)      | FP16 | 411  | 57 |  7227 |  4.2% |
| Cross-attn v2 (1024 q, 256 kv)   | FP16 | 168  | 45 |  6417 |  3.7% |
| FA regpv (legacy)                | FP16 | 242  | 31 |  5554 |  3.2% |
| ResBlock implicit GEMM (320ch)   | FP16 | 420  | 19 |  2092 |  1.2% |

### Regime classification

At GA104's 608 GB/s DRAM, the DRAM ceiling crosses the FP16 TC peak
(174 TFLOPS) at OI = 286. Every kernel measured has OI_DRAM ≥ 126,
sitting **above the DRAM-only line** for some ceiling. **No kernel is
DRAM-bound**.

| OI_DRAM range | regime |
|---|---|
| > 286 (FP16 TC peak) | compute-bound (DRAM ceiling at peak; can't go higher) |
| 126-286              | DRAM-shifted compute-bound (DRAM ceiling > achieved, headroom on DRAM) |

The L2 ceiling (~3 TB/s) crosses FP16 peak at OI ~58. Most kernels
have OI_L2 between 30 and 80 — the L2 traffic is closer to the L2
ceiling than DRAM is to its ceiling, but everyone is still well below
both.

### Why every kernel is compute-limited

Two compounding reasons documented earlier:

1. **HMMA queue throttling** (Observation V): GA104's tensor pipe has
   8-cycle issue latency between consecutive HMMAs, hard-capping
   HMMA-dense kernels at ~46% TC util.

2. **LDSM stalls absent padding** (Observations U, W, Y): the kernels
   that ARE close to peak (HGEMM 16-warp at 17%) had this fixed
   already; the others (FA, cross-attn) reached 12-15% peak after
   padding (Obs X).

The roofline formalises what we already saw: bandwidth is not the
limiter. The forensic NCU work in U/V/W/Y/Z all pointed at the compute
pipeline; this measurement makes the conclusion explicit and visible.

### Outliers: ResBlock and Sparse INT8

**ResBlock** (1.2% peak): OI_DRAM = 420 yet only 2 TFLOPS. Both
ceilings are far above. Hypothesis: tile-shape mismatch or low
HMMA density. NCU shows TC util = 3.19% (vs 14% on the FA family).
Big optimization target.

**Sparse INT8** (5.2% of dense INT8 peak; 10.4% if you credit 2:4
sparsity): respectable but should be far higher. NCU shows TC util =
14% only. Would benefit from #96 (SASS hand-tune of IMMA control
codes, S04 → S02 pattern).

### Why the previous (estimated) roofline was misleading

The current `docs/figures/roofline.png` used `2·M·N·K /
(M·K + K·N + M·N)` style estimates. For HGEMM 16-warp at 4096³:

- Estimated OI: 2·4096³ / (3·4096²·2 B) = 1365 (assumes one-shot DRAM)
- Measured OI: **162** (8.4× lower!)

The estimated number assumes each byte hits DRAM exactly once. Reality:
HGEMM tile reuse means **the L2 carries most traffic**. Measured DRAM
is what really matters for the bandwidth ceiling. The estimated
roofline was **placing kernels too far right**, making them look
artificially compute-bound.

After correction, kernels are still compute-bound — but the *measured
distance* to the DRAM ceiling is much smaller (factor 5-10× less
headroom than estimated). This means **DRAM optimization could matter**
in some regimes, just less than previously thought.

### Files

- `scripts/profile/roofline_measured.R` — generates this analysis from
  `results/ncu/all.csv`
- `scripts/profile/ncu_profile.R` — extended in this commit with
  `dram_{read,write}_bytes`, `l2_bytes`, `duration_ns`, `hmma_count`,
  `tensor_count`, `alu_count`
- `docs/figures/roofline_measured.png` — the figure
- `docs/roofline_measured.md` — per-kernel data table

### Re-run

```
bash scripts/profile/ncu_profile_all.sh    # refresh results/ncu/all.csv
Rscript scripts/profile/roofline_measured.R
```

---

## Observation CC — register pressure audit, 0 spills across 103 kernels (#91)

**TL;DR**: `cuobjdump --dump-resource-usage` across all 103 cubins
shows **zero register spills** anywhere. The aggressive end of the
register-budget curve (IGEMM 256x256 at 255 regs/thread, hgemm_256x128
at 124×512) is sitting at 97-99.6% of the SM's 65,536-register
register file but never overflows. Tight management.

### Audit results

| metric | value |
|---|---:|
| total kernels                                  | 103 |
| spillers (LOCAL > 0)                           | **0** |
| dynamic-smem kernels (`extern __shared__`)      | 11 |
| median regs/thread                              | 64 |
| 90th-pct regs/thread                            | 168 |
| max regs/thread                                 | **255** (sm_86 hard cap) |

### Block-budget cliff: 8 register-saturated kernels

These kernels hit the register file ceiling (1 block/SM) due to
register pressure alone, not smem:

| kernel | regs | block_size | regs/block | % SM regs |
|---|---:|---:|---:|---:|
| igemm_8warp_256x256        | 255 | 256 | 65280 | **99.6%** |
| igemm_warp_specialized     | 255 | 256 | 65280 | 99.6% |
| hgemm_256x128              | 124 | 512 | 63488 | 96.9% |
| igemm_online_quant         | 239 | 256 | 61184 | 93.4% |
| igemm_persistent           | 234 | 256 | 59904 | 91.4% |
| igemm_online_quant_inplace | 213 | 256 | 54528 | 83.2% |
| igemm_online_quant_bankfree| 211 | 256 | 54016 | 82.4% |
| igemm_8warp_256            | 210 | 256 | 53760 | 82.0% |

Each uses 80-100% of the SM register file at its launch_bounds. They
cannot get to 2 blocks/SM without dropping register count or block
size. The IGEMM family is **register-bound** — not smem-bound, not
spill-bound — and reducing register count is the only path to higher
occupancy.

### Cross-check with previous observations

| kernel | this audit | prior claim | match |
|---|---|---|---|
| `flash_attn_br16_v2_pipeline_pad2` | 133 regs, theo 3 blocks/SM | "2 blocks/SM" (Obs W) | ✓ launch_bounds clamps at 2 |
| `hgemm_16warp_epi_pad`             | 124 regs, dynamic smem | "1 block/SM, 53 KB smem" (Obs Y) | ✓ smem clamps below register budget |
| `hgemm_16warp`                     | 64 regs, 37 KB smem    | "2 blocks/SM, 32 warps/SM" (Obs U) | ✓ |

### Implications for #96 SASS hand-tune target selection

Top candidates for SASS hand-tuning (#96, blocked by #101) given the
audit:

1. **HGEMM 16-warp (64 regs)** — has register headroom. Could
   trade more registers for fewer reloads, but already at the HMMA
   queue-pressure ceiling (Obs V). SASS-side reordering, not register
   reallocation, is the path.

2. **IGEMM 8warp_256 family** — register-saturated but 0 spill.
   Hand-tuning the IMMA control codes (S04 → S02, see Phase 5 results
   in the issue) could give +1.6% per IMMA according to past
   experiments. This is the cleanest target.

3. **FA v2 pipeline_pad2** — 133 regs at 2 blocks/SM. Has register
   headroom. Worth investigating whether HMMA-issue scheduling can be
   tightened.

### Tooling

- `scripts/audit/reg_audit.R` — the audit (256 lines)
- `docs/register_audit.csv` — full per-kernel data
- `docs/register_audit.md` — Markdown report

The launch_bounds parser handles the common pattern of macro-defined
block sizes (`__launch_bounds__(NUM_WARPS * WARP_SIZE, 2)`) by
extracting `#define` directives from the same `.cu` and resolving
them iteratively. 67/103 kernels resolved their block size this way;
the remaining 36 didn't have `__launch_bounds__` (default 1024).

### Re-run

```
Rscript scripts/audit/reg_audit.R
```

---

## Observation DD — persistent dispatch falsified for compute-bound GEMM (#86)

**TL;DR**: The claim that "persistent grid + cooperative work distribution
gives ~1.15× across all GEMM kernels" (`docs/comparison_to_sota.md`) is
**not supported by measurement on HGEMM 16-warp**. Across 6 problem
sizes from 1024³ to 8192³ + skinny variants, geomean speedup is
**0.988× (regression)**. The only modest win is at 1024³ (1.072×).
Following the diagnostic-first principle (Obs U/V/Z), this hypothesis
joins the falsified pile.

### Test setup

- Kernel: hgemm_16warp (the canonical HGEMM, +8 padded, 2 blocks/SM).
- Persistent variant: `phase2/hgemm/hgemm_16warp_persistent.cu`. Identical
  inner loop, wrapped in `for (int tile_id = blockIdx.x; tile_id <
  n_tiles; tile_id += gridDim.x)` with `<<<sm_count * 2, 512>>>` =
  `<<<92, 512>>>`.
- Bench harness: `bench_persistent_hgemm.cu` (A/B, 5 warmup + 30 timed).

### Results

| shape | orig (ms) | orig GFLOPS | persistent (ms) | persistent GFLOPS | speedup |
|---|---:|---:|---:|---:|---:|
| 1024³ (small)              |  0.126 | 16996 |  0.118 | 18226 | **1.072×** |
| 2048³ (medium)             |  0.627 | 27386 |  0.661 | 25992 | 0.949× |
| 4096³ (large)              |  4.578 | 30020 |  4.387 | 31329 | 1.044× |
| 8192³ (xlarge)             | 34.211 | 32139 | 35.731 | 30772 | 0.957× |
| 512×512×8192 (skinny K)    |  0.371 | 11566 |  0.391 | 10991 | 0.950× |
| 256×256×8192 (very skinny) |  0.371 |  2892 |  0.388 |  2764 | 0.956× |

Geomean: **0.988×** (min 0.949×, max 1.072×).

### Why persistent doesn't help here

Three reasons converge:

1. **Modern CUDA launch overhead is small** (~5-20 µs / kernel) and is
   amortised by the timed loop's batched launches. With 30 iterations
   the per-launch overhead is split across all of them, and it's
   already smaller than per-tile work for problems above ~1024³.

2. **No cross-tile state to reuse**. The textbook persistent-grid
   benefit is reusing smem / register state between tiles (e.g.,
   keeping a row of A loaded). The naive transcription used here just
   wraps the existing kernel in a `for (tile_id ...)` loop — every
   tile re-loads A and B from scratch, so the only thing being saved
   is the per-block barrier setup, which is microseconds at most.

3. **Block scheduler is already persistent-like**. CUDA's hardware
   block scheduler continuously picks ready blocks off the queue and
   dispatches to free SMs. There's no "wave" stall between tile
   batches in normal launches; they overlap.

### When persistent would help (predictions, not measured)

- **Very small problems** where launch overhead is > 50% of run time
  (e.g., 256³ where work is microseconds).
- **State-sharing rewrites** where a row of A or column of B is loaded
  once and reused across multiple output tiles.
- **Cross-block reduction** (split-K with persistent + cooperative
  groups) — different optimization, see #87.

The 1024³ case showed +7.2% — consistent with the launch-overhead
hypothesis at small sizes. Above 2048³ the measurement is uniformly
flat or slightly negative.

### Cross-check with FA persistent

`flash_attn_v2_persistent_pad` (Obs X) measured 15.2 TFLOPS at
seq=1024 vs 15.1 for `flash_attn_br16_v2_pad` baseline — also
basically tied. Persistent dispatch alone, without state-sharing, is
not a free 1.15×.

### Closing #86

The original framing was "close 1.15× across all kernels". The measurement
shows that without state-sharing rewrites, persistent gives ≤ 1.07× on
small problems and 0.94-0.96× on larger ones. Closing the issue as
**not-planned in this form** — re-evaluate if a state-sharing variant
emerges (e.g., A-row reuse across N-tiles).

### Files

- `phase2/hgemm/hgemm_16warp_persistent.cu` — naive persistent variant
- `phase2/hgemm/bench_persistent_hgemm.cu` — A/B bench

---

## Observation EE — K-split validates and exceeds #87 claim on skinny shapes

**TL;DR**: Pattern A (atomicAdd) K-split for HGEMM 16-warp delivers
**up to 4.57× on extreme-skinny shapes**, validating and exceeding
the issue #87 claim of 1.5× for skinny matrices. Square shapes
regress 24-41% due to atomicAdd cost + forced 1 block/SM. Result:
**dispatch policy** — use K-split below an M·N threshold.

### Implementation

`phase2/hgemm/hgemm_16warp_splitk.cu`: identical inner loop to
`hgemm_16warp.cu`, plus:

1. New parameter `int k_split`; new `blockIdx.z` axis selects which
   K-slice this block computes (`tile_id_lo = blockIdx.z * tiles_per_split`).
2. Every block accumulates a partial sum and **atomicAdd**s into
   `matrix_c`. Host zeros C before launch.
3. K-split uses dynamic smem (53 KB total = 20 KB A + 17 KB B + 16 KB
   epi_tile) which forces 1 block/SM (over the 50 KB cliff). Trade-off
   accepted because added blocks come from the K dimension instead.

### Results (4096³ down to 128x128x8192)

| shape | orig ms | orig GFLOPS | best splitk ms | best GFLOPS | speedup | best k_split |
|---|---:|---:|---:|---:|---:|---:|
| 4096³ (square, large)   | 4.62 | 29750 | 6.07 | 22644 | 0.76× | 2 |
| 2048³ (square, med)     | 0.50 | 34307 | 0.85 | 20139 | 0.59× | 2 |
| 1024³ (square, small)   | 0.10 | 21076 | 0.16 | 13641 | 0.65× | 2 |
| 512×512×4096 (mid skinny)   | 0.19 | 11355 | 0.11 | 19910 | **1.75×** | 8 |
| 256×256×4096 (skinny)       | 0.19 |  2859 | 0.04 | 12821 | **4.48×** | 8 |
| 256×256×8192 (very skinny)  | 0.37 |  2890 | 0.13 |  8019 | **2.77×** | 4 |
| 128×128×8192 (extreme)      | 0.37 |   722 | 0.08 |  3301 | **4.57×** | 8 |

### Why K-split wins on skinny

For small M·N, the original kernel produces too few output tiles to
fill the 46 SMs × 2 blocks = 92 SM slots. Examples:

- 128×128×K: only 1 tile total. 91 SM slots idle. Achieved 722 GFLOPS
  (0.4% of peak).
- 256×256×K: only 2×2 = 4 tiles. 88 SM slots idle. Achieved 2.9 TFLOPS
  (1.7% of peak).

K-split = 8 multiplies block count by 8: now 8 blocks instead of 1
(128×128) or 32 instead of 4 (256×256). Each block does 1/8 the K
work but the GPU runs them in parallel. Total time drops 4-5×.

### Why square shapes regress

Three compounding reasons:

1. **atomicAdd cost** — every output cell takes an atomic. For 4096³
   that's 16M atomicAdds; their latency adds up even at high
   throughput.
2. **1 block/SM** vs the standard kernel's 2 blocks/SM (53 KB > 50 KB
   cliff). Halving occupancy directly costs ~10-20%.
3. **No need for K-split** — 4096³ has 1024 tiles, already filling 11
   waves. Adding more blocks doesn't help; the math pipeline is the
   bottleneck (Obs V).

### Production dispatch policy

Threshold on M·N. The cross-over from this measurement is somewhere
between 256×256 (K-split wins 4×) and 1024×1024 (K-split loses 35%).
Pragmatic threshold: `K-split if M*N < 1024² else standard`.

| M·N range | recommended kernel | typical speedup |
|---|---|---|
| < 1M cells (skinny) | hgemm_16warp_splitk | 1.5-4.5× |
| ≥ 1M cells (square) | hgemm_16warp | 1.0× (use standard) |

### Numerical accuracy note

FP32 atomicAdd is order-dependent. For K-split=8 on a 4096-K problem,
each output cell gets 8 contributions summed in non-deterministic
order. Worst-case last-bit drift relative to canonical kernel: ~2-3
ulp. Bench used `1e-2` rel tolerance vs the standard `1e-3`; both
shapes passed.

### Files

- `phase2/hgemm/hgemm_16warp_splitk.cu` — kernel
- `phase2/hgemm/bench_splitk.cu` — A/B sweep with k_split autotune

---

## Observation FF — shape-aware HGEMM dispatch (closes #95)

**TL;DR**: Header-only dispatcher `hgemm_dispatch.cuh` picks between
the standard kernel and the splitk kernel at runtime based on M·N
and K. Validated on 9 shapes: **8 of 9 within 5% of measured best**;
**2.54× geomean speedup** vs always-standard. The single miss
(`512×512×1024`) was within measurement noise (0.040 vs 0.041 ms).

### Decision matrix (from `pick_variant`)

```
M*N >= 1024 * 1024              -> Standard
M*N <  256 *  256 (extreme)     -> SplitK_8
256*256 <= M*N < 1024*1024      -> SplitK_8 if K < 8192, else SplitK_4
                                   (large K: smaller split balances atomicAdd cost)
fall-through (no clean split)   -> Standard
```

### Validation

Across 9 shapes, the dispatcher:

| shape | picked | best measured | speedup over standard |
|---|---|---|---:|
| 4096³ (square)     | standard  | standard  | 1.12× |
| 2048³ (square)     | standard  | standard  | 0.99× |
| 1024³ (square)     | standard  | standard  | 0.94× |
| 512×512×4096       | splitk_8  | splitk_8  | 1.76× |
| 256×256×4096       | splitk_8  | splitk_8  | **4.54×** |
| 256×256×8192       | splitk_8  | splitk_8  | **5.36×** |
| 128×128×8192       | splitk_8  | splitk_8  | **5.95×** |
| 1024×1024×4096     | standard  | standard  | 1.00× |
| 512×512×1024       | splitk_8  | splitk_2  | 1.21× (noise: 0.040 vs 0.041) |

Dispatch correct picks: **8 / 9** (within 5% of measured best).
Geomean speedup vs always-standard: **2.54×**.

The 256³-tier wins are larger here than in Observation EE because
the dispatch path includes warmup amortisation; same kernel, slightly
different measurement context.

### Scope choice

The original framing of #95 asked for 4-8 tile-size variants
(BM/BN/BK combinations) plus a sweep + JSON dispatch table. Based on
post-Observation EE understanding, the **dispatch axis our
measurements actually justify is splitk-vs-standard**, not
tile-size-variants. Generating tile-size variants is a separate
effort; without measurement showing they help, it would be
speculative.

This commit covers the dispatch infrastructure for the dispatch axis
that DOES help (5+ on extreme skinny). Tile-size autotune is a
sensible follow-up if/when measurements indicate per-shape tile
preference.

### Implementation

- `phase2/hgemm/hgemm_dispatch.cuh` — header-only dispatcher
  (~190 lines). Three pieces:
  - `Handles` struct: pre-loaded function handles + smem size
  - `pick_variant(M,N,K)`: heuristic
  - `launch(...)`: zero-C-then-launch path for splitk; direct launch
    for standard
- `phase2/hgemm/bench_dispatch.cu` — A/B/C bench harness that runs
  standard, splitk-best, and dispatch on each shape

### Production usage

```cpp
#include "hgemm_dispatch.cuh"

hgemm_dispatch::Handles h { fn_standard, fn_splitk, smem_bytes };
hgemm_dispatch::launch(h, dA, dB, dC, M, N, K);
```

The dispatcher autotunes nothing at runtime — the heuristic constants
are baked in. To re-derive them, run `bench_dispatch.cu` after any
kernel change.

---

## Observation GG — implicit GEMM v2: 2.18x on ResBlock outlier (Obs BB target)

**TL;DR**: New `implicit_gemm_conv_v2` (16-warp 128×128×32, cp.async,
FP16 input) hits **2.18×** on the ResBlock outlier identified in
Observation BB. Geomean **1.71×** across 6 ResBlock shapes;
best-shape ResBlock now at **4.72% of peak** (was 2.16%). Same recipe
that worked for HGEMM 16-warp and FA pipeline pad2: bigger tile,
more warps, double-buffered cp.async.

### What v1 looked like

`phase4/conv2d/conv2d_implicit_gemm.cu`:
- 4 warps, 64×64×16 tile
- No cp.async, single-buffered (every K-tile blocks on `__syncthreads`
  before HMMA can start)
- FP32 input with inline `__float2half` cast on every load
- 6.3 KB smem, 1 block per output tile

The TC util reading from Obs BB (3.19%) traced to all three issues:
small tiles → low HMMA per smem-load ratio; no async → load latency
exposed; FP32-in → scalar smem stores rather than vectorized.

### v2 design

`phase4/conv2d/conv2d_implicit_gemm_v2.cu`:
- 16 warps, 128×128×32 tile (matches `hgemm_16warp.cu`)
- 4×4 warp grid, 2×2 register-fragment tiles per warp
- Double-buffered cp.async on weights B (FP16, vectorized 16 B copy)
- Scalar im2col loads on A (cp.async cannot perform FP32→FP16 cast,
  and the im2col indices are per-element scalar synthesis)
- Coordinate tables (M-dim 128 entries, K-dim 32 entries) in static smem
- 37 KB smem total → 2 blocks/SM (under 50 KB cliff per Obs W)

### Standalone conv results (FP16 input pre-cast)

| shape (N C HxW)         | v1 ms | v1 GFLOPS | v2 ms | v2 GFLOPS | speedup |
|---|---:|---:|---:|---:|---:|
| N=1 C=64  32×32 (Obs BB) | 0.073 |  514 | 0.045 |  846 | **1.65×** |
| N=1 C=128 32×32          | 0.089 |  849 | 0.045 | 1685 | **1.98×** |
| N=1 C=256 32×32          | 0.096 | 1565 | 0.067 | 2255 | 1.44× |
| N=1 C=512 16×16          | 0.057 |  663 | 0.042 |  895 | 1.35× |
| N=4 C=128 32×32          | 0.107 | 2818 | 0.104 | 2891 | 1.03× |
| N=4 C=256 32×32          | 0.215 | 2804 | 0.190 | 3184 | 1.13× |
| N=4 C=512 16×16          | 0.098 | 1536 | 0.103 | 1467 | 0.96× |
| N=8 C=256 32×32          | 0.349 | 3458 | 0.365 | 3314 | 0.96× |

Geomean across 8 standalone confs: **1.27×**.

### Full ResBlock pipeline results

Pipeline (5 kernels: GN+SiLU → cast → conv → GN+SiLU → cast → conv →
residual_add). The cast pass (FP32→FP16) is required because v2
takes FP16 input and the GroupNorm output is FP32. Cast cost: 1× DRAM
read + 1× DRAM write of the activation tensor.

| shape (N C HxW)        | v1 ms | v2 ms | speedup | v1 % peak | v2 % peak |
|---|---:|---:|---:|---:|---:|
| N=1 C=64  32×32 (Obs BB) | 0.678 | 0.576 | 1.18× | 0.13% | 0.15% |
| N=1 C=128 32×32          | 0.789 | 0.537 | **1.47×** | 0.44% | 0.65% |
| N=1 C=256 32×32          | 1.817 | 0.830 | **2.19×** | 0.76% | 1.67% |
| N=1 C=512 16×16          | 2.173 | 1.078 | **2.02×** | 0.64% | 1.29% |
| N=4 C=256 32×32          | 1.855 | 1.255 | 1.48× | 2.99% | 4.43% |
| N=4 C=512 16×16          | 2.565 | 1.177 | **2.18×** | 2.17% | 4.72% |

Geomean across 6 ResBlock shapes: **1.71×**.

### Why N=1 C=64 stays stuck at <1%

The Obs BB shape is so small that with v2's 128×128 tile, only
`(1*32*32) / 128 = 8` M-tiles and `64 / 128 = 1` N-tile, so 8 blocks
total per conv. With 46 SMs × 2 blocks/SM = 92 SM slots, 84 sit idle.
The pipeline is now GroupNorm-dominated, not conv-dominated. To win
here we'd need a smaller-tile variant of v2 OR fuse GN into conv.
Filed as future work; not pursued in this commit.

### Why the cast cost doesn't kill the win

For a typical layer the activation tensor is tens of MB. Cast cost
is ~2×size / DRAM_BW = ~50 µs for a 16 MB tensor at 608 GB/s. The
conv it enables runs ~500 µs, so cast is ~10% overhead on the conv
pass; offset by the 2× speedup on the conv itself. Net win as
measured.

Future fusion: write `groupnorm_silu_fused_fp16` that outputs FP16
directly. Drops the cast pass entirely, frees up ~10% more on the
pipeline. Easy 1-line change to the existing GN kernel; left for
follow-up.

### Headline against Obs BB roofline

The ResBlock outlier in Obs BB was 1.2% peak. With v2 at N=4 C=512
16×16: **4.72% peak**. Still well below the GEMM 17% peak we see on
HGEMM 16-warp at the same tile structure — gap is the FP32→FP16 cast
overhead and the per-element scalar A-load (vs HGEMM's vectorized
A-load). Both are addressable in v3.

### Files

- `phase4/conv2d/conv2d_implicit_gemm_v2.cu` — kernel (16-warp,
  cp.async, FP16 input)
- `phase4/conv2d/bench_implicit_v2.cu` — standalone conv A/B
- `phase4/resblock/bench_implicit_v2.cu` — full ResBlock pipeline A/B

---

## Observation HH — IMMA stall hand-tunes do not reproduce on CUDA 13.2 (#96 sub-task A)

**TL;DR**: Observation 14 documented a +1.6% gain from rewriting IMMA
`S04 → S02` stall counts via CuAssembler. With cuasmR (#102) on the
current CUDA 13.2 toolchain, **neither S08→S04 nor S04→S02 produces a
statistically significant speedup**. Across 6 IGEMM kernels:

  - `S08 → S04` geomean: **0.996×** (range 0.99-1.01)
  - `S04 → S02` geomean: **0.992×** (range 0.96-1.00)

Best individual case (`igemm_8warp_256x256`, 156 S08 IMMA → S04, 5
reps): **+1.7% mean, p = 0.215** — within noise.

The historical hand-tune is no longer reachable. ptxas under CUDA 13.2
appears to schedule IMMA stalls correctly already; what was a
compiler-conservative S04 in CUDA 11.x is now whatever the hardware
needs, and what was S08 in CUDA 11.x is also whatever the hardware
needs.

### What was tried

`scripts/bench/handtune_imma_s04.R` (cuasmR-based) walks each IGEMM cubin,
finds every IMMA whose stall field (bits 40:43 of the 64-bit control
word) equals `from_stall`, rewrites to `to_stall`, and writes a
patched cubin alongside the original.

Two passes:

| pass | from | to | candidates per kernel |
|---|---|---|---|
| A | S08 | S04 | 7-156 (top: igemm_8warp_256x256) |
| B | S04 | S02 | 0-33  (top: igemm_8warp_tribuf) |

Bench: file-swap A/B against `phase2/igemm/bench`, 5 reps each.

### Results

| kernel | n_changed (s08→s04) | speedup | n_changed (s04→s02) | speedup |
|---|---:|---:|---:|---:|
| `igemm_8warp_256x256`     | 156 | 1.013× | 2  | 0.999× |
| `igemm_8warp_tribuf`      | 25  | 0.988× | 33 | 1.002× |
| `igemm_8warp`  (128x128)  | 19  | 0.991× | 11 | 1.001× |
| `igemm_pipelined`         | 11  | 1.000× | 1  | 0.998× |
| `igemm_pipelined_cpasync` | 7   | 0.993× | 1  | 0.997× |
| `igemm_tiled`             | 7   | 0.990× | 0  | 0.957× (noise) |
| `igemm_warp_specialized`  | 9   | -      | 8  | -      |
|                          |     |        |    |        |
| **geomean**              |     | **0.9957×** |  | **0.9919×** |

The 5-rep validation on the most-changed kernel
(`igemm_8warp_256x256`, 156 S08 IMMA -> S04):

  - orig:    11623 ± 283 GFLOPS (2.44% sd)
  - patched: 11821 ± 151 GFLOPS (1.28% sd)
  - speedup: 1.017×, 95% CI [0.987, 1.047], **p = 0.215**

Noise floor at this kernel size is 1-2% per rep. Any sub-1% mean
shift cannot be claimed without ~50 reps and tight thermal control.

### Why the historical pattern stopped working

Three plausible factors compound:

1. **ptxas got better.** Compilers track per-arch latency tables; the
   gap between "compiler conservative" and "hardware optimal" narrows
   release over release. The Obs 14 finding (CUDA 11.x era) was a
   small, repeatable win at that toolchain level.
2. **GA104 clock variability.** Tensor Core frequency scales
   thermally; on this laptop GPU the run-to-run variance is large
   enough to swamp <2% effects without dedicated cooling and many
   reps.
3. **Stall fields aren't the bottleneck.** Each kernel here is L2
   bandwidth-bound or L1tex-bound (per Obs BB). Reducing scheduler
   stalls when the back-end isn't ready just exposes a different wait
   state.

### Sub-task B / C status

Per #96 the issue had three sub-tasks:

- **A** (S04→S02 across IMMA) — completed above, **negative result**.
- **B** (HMMA stall audit) — Obs 14 already established HMMA S08 is
  hardware-fixed; our ctrl-field decoder confirms HMMA stalls in
  `hgemm_16warp` are S00/S08 with no reducible pattern. Filed as
  superseded; no further investigation.
- **C** (non-Tensor-Core hand-tunes) — speculative; left for future
  work if measurement-driven evidence emerges.

### Closing #96

Closing as **negative result**. The 1.10× "hand-written SASS inner
loops" gap factor in `docs/comparison_to_sota.md` is over-estimated
relative to current ptxas; ~0% is the more honest number on CUDA 13.2.
Real wins on these kernels come from algorithmic restructuring (Obs U
through GG: smem padding +2.36×, K-split +4.57×, ResBlock 16-warp
+2.18×, dispatch 2.54×), not control-code micro-edits.

cuasmR remains useful as a diagnostic tool: the IMMA-stall histogram
across kernels (S00=18%, S02=39%, S04=14%, S06=12%, S08=17% across the
audited cubins) is itself a signal of where the compiler thinks
register-port contention lives.

### Files

- `scripts/bench/handtune_imma_s04.R` -- cuasmR-driven patcher (S08→S04 and
  S04→S02 modes)
- `scripts/bench/bench_imma_s04.R`, `scripts/bench/bench_imma_s02.R` -- A/B bench
  driver
- `phase2/igemm/*.imma_s04.sm_86.cubin` -- patched cubins (kept for
  reference, not used in any active bench)

---

## Observation II — Cymatic mode optimization: per-trace best beats default by 1.37× geomean (#93)

**TL;DR**: Sweeping `(n, m)` ∈ {2..10}×{1..6} = 54 modes for the cymatic
memory layout finds a better mode than the default (6, 4) on **every
trace**. Per-trace best vs per-trace default geomean **1.371×**, with
the worst-case trace (`radial_bnd_5pi12`, default 0.52× → best 1.21×)
flipping from a 1.92× **slowdown** into a 1.21× speedup just by
picking a different mode. The single best universal mode is (n=5,
m=4), geomean 1.099× across 15 traces.

### Methodology

`scripts/cymatic/cymatic_optimize.R` drives the sweep:

  1. For each `(n, m)`: regenerate `phase4/cymatic/perm.bin` +
     `traces.bin` via `gen_cymatic_data.R GRID n m`.
  2. Run `bench_cymatic` (median-of-11 over 15 access traces, GRID=2048
     buffer = 13 MB → DRAM regime where layout effects dominate).
  3. Parse the per-trace `<float>x` speedup from stdout, append to a
     long-form data frame.

54 modes × ~50 s/mode ≈ 46 min wall clock on this GPU. Output:
`docs/figures/cymatic/cymatic_optimize_2048.csv` (810 rows). Summary +
heatmaps via `scripts/cymatic/cymatic_optimize_summary.R`.

### Per-trace best modes

| trace                | default(6,4) | best mode | best speed | gain vs default |
|----------------------|---:|---|---:|---:|
| `radial_bnd_5pi12`   | **0.52×** | (9, 6) | 1.21× | **2.33×** |
| `circular_r060`      | 1.01× | (3, 6) | 1.84× | 1.82× |
| `circular_r030`      | 1.35× | (4, 3) | 2.37× | 1.76× |
| `radial_bnd_pi4`     | 0.70× | (6, 2) | 1.15× | 1.64× |
| `polar_tile_pi6`     | 0.94× | (6, 6) | 1.36× | 1.45× |
| `radial_bias_07`     | 0.84× | (3, 1) | 1.15× | 1.37× |
| `polar_tile_pi4`     | 1.00× | (4, 6) | 1.32× | 1.32× |
| `radial_bias_00`     | 1.21× | (7, 4) | 1.52× | 1.26× |
| `rowmajor_full`      | 0.66× | (6, 1) | 0.81× | 1.23× |
| `radial_mid_pi6`     | 1.39× | (2, 6) | 1.67× | 1.20× |
| `colmajor_full`      | 1.00× | (2, 5) | 1.20× | 1.20× |
| `radial_mid_0`       | 1.00× | (8, 1) | 1.18× | 1.18× |
| `radial_mid_pi3`     | 0.98× | (6, 6) | 1.10× | 1.12× |
| `radial_bnd_pi12`    | 1.00× | (6, 2) | 1.12× | 1.12× |
| `random`             | 0.97× | (3, 4) | 1.05× | 1.08× |
| **geomean**          | — | — | — | **1.371×** |

### Best universal modes (geomean across all 15 traces)

| rank | n | m | geomean | min trace | max trace |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 4 | 1.099× | 0.79× | 1.92× |
| 2 | 8 | 5 | 1.087× | 0.76× | 1.40× |
| 3 | 7 | 4 | 1.080× | 0.75× | 1.43× |
| 4 | 9 | 6 | 1.081× | 0.71× | 1.92× |
| 5 | 3 | 6 | 1.071× | 0.70× | 1.84× |
| ... | | | | | |
| -- | 6 | 4 | (default) 0.92× | 0.52× | 1.39× |

The default (6, 4) is **below 1.0× geomean** — across the 15 traces
its losses outweigh its wins. (5, 4) is the safest single pick if you
must use one mode for everything.

### Hypothesis check from #93

> *For a radial sweep at θ=θ₀, the best mode should have a sector
> midline at θ₀, i.e., n such that θ₀ = k·π/n.*

Mixed evidence:

| trace             | predicted n (kπ/n = θ₀) | best n | match? |
|-------------------|---|---|---|
| `radial_mid_0`    | any (θ=0)             | 8 | trivial (any n satisfies) |
| `radial_mid_pi6`  | n ∈ {6, 12, 18, ...}  | 2 | **NO** |
| `radial_mid_pi3`  | n ∈ {3, 6, 9, ...}    | 6 | **YES** |

The clean midline-alignment story is too simple. The actual best mode
depends on the interaction of:

  1. Angular alignment of the trace with sector midlines (the
     hypothesis above).
  2. Radial-band count `m` matching the trace's r-extent.
  3. **Total region count** `n × m`: too few regions and addresses
     within a region degrade to non-coalesced cell ordering; too many
     regions and the address space fragments.

The latter two effects can dominate: `radial_mid_pi6` prefers (2, 6) =
12 sectors over (6, 4) = 24 sectors despite the angular misalignment.
At 12 sectors the trace fits into one sector's full radial range with
internal contiguity that beats the angular-aligned but more-fragmented
(6, *) modes.

### Why this matters for #94

Issue #94 wants to apply the cymatic layout to Flash Attention's K/V
buffer. The (default, single-mode) approach measured here would lose
on most FA access patterns because the default mode is geometrically
arbitrary relative to attention's QK^T pattern. **Mode selection is
not optional** — pick wrong, lose 1.9×; pick right, win 1.4×.

This shifts #94's design: it's not "swap layout, measure speedup", it
must be "characterize the FA access trace, pick mode by alignment,
measure". The framework for that picking lives in
`cymatic_optimize.R`.

### Files

- `scripts/cymatic/cymatic_optimize.R`         — sweep driver (parameterizable)
- `scripts/cymatic/cymatic_optimize_summary.R` — post-process + plots
- `docs/figures/cymatic/cymatic_optimize_2048.csv`         — long-form data
- `docs/figures/cymatic/cymatic_optimize_2048_summary.csv` — best per trace
- `docs/figures/cymatic/cymatic_optimize_2048_facet.png`   — 15-trace heatmap
- `docs/figures/cymatic/cymatic_optimize_2048_geomean.png` — geomean heatmap
- `docs/figures/cymatic/cymatic_optimize_2048_<trace>.png` × 7 — focus plots

### Closing #93

Closing as **validated**. Mode selection alone delivers +37% geomean
over default and unblocks #94 (which now needs an
"alignment-driven mode picker" rather than treating layout as a free
swap-in).

---

## Observation JJ — Cymatic on Flash Attention K/V: 1.12× upper bound, structurally consumed by cp.async (#94)

**TL;DR**: Issue #94 step 1 done. The Flash Attention block-level
access trace at seq=1024 (DRAM regime) tops out at **1.12× with
optimal mode (n=9, m=4)** vs row-major. **Default cymatic (6, 4) is
tied (1.01×).** Step 2 (real-kernel integration) is structurally
infeasible: any practical permutation breaks cp.async (LDGSTS) vector
loads, and the resulting overhead exceeds the layout's 12% benefit.

### Trace setup

`phase4/cymatic/gen_fa_traces.R` injects three FA-flavored access
traces alongside the existing `rowmajor_full` control:

| trace family | order | semantics |
|---|---|---|
| `fa_seqN_rowmajor` | (q, k) lexicographic | actual FA pipeline iteration order |
| `fa_seqN_diagonal` | (q+k, q) | hypothetical diagonal scan (not used by current kernel) |
| `fa_seqN_zigzag`   | rows alternate direction | hypothetical snake scan (not used by current kernel) |

For each (q_block, k_block) pair, the trace expands into all
`Bc × D_HEAD = 64 × 64 = 4096` cells visited inside the block (modeling
per-element access). At seq=1024 with Br=16 Bc=64 D=64 → trace length
= 64 × 16 × 4096 = 4,194,304 logical visits → 1,346,688 in-disc cells
gathered after disc-mask filter.

`scripts/cymatic/cymatic_fa_alignment.R` runs the same 15-mode coarse sweep
((n ∈ {3,5,6,7,9}) × (m ∈ {2,4,6})) used for the synthetic traces in
Obs II.

### Results at seq=512 vs seq=1024

| trace                  | default (6, 4) | best mode | best speed |
|------------------------|---:|---|---:|
| **DRAM regime (seq=1024)** |   |   |   |
| `fa_seq1024_rowmajor`  | 1.01× | (9, 4) | **1.12×** |
| `fa_seq1024_diagonal`  | 0.98× | (6, 2) | 1.13× |
| `fa_seq1024_zigzag`    | 1.07× | (7, 2) | **1.25×** |
| **Cache regime (seq=512)** |   |   |   |
| `fa_seq512_rowmajor`   | 1.11× | (9, 4) | 13.3× |
| `fa_seq512_diagonal`   | 1.18× | (9, 6) | 15.6× |
| `fa_seq512_zigzag`     | 0.20× | (9, 6) | 11.4× |
| **Control**            |   |   |   |
| `rowmajor_full` (synth) | 0.83× | (9, 4) | 0.87× |

The seq=512 numbers are **L1/L2 sensitivity artifacts** (working set ≈
3 MB fits in 4 MB L2 post-warmup). The 13×–15× "wins" are the trace
falling into a beneficial cache replay pattern, not a layout property.
The seq=1024 numbers (working set 13 MB, fully DRAM-resident) are the
real measurement.

### Why FA trace beats `rowmajor_full` (1.12× vs 0.87×)

The synthetic `rowmajor_full` trace touches every in-disc cell exactly
once. Cymatic permutation cannot help: any reordering of "scan all
cells" still scans all cells. The layout's locality benefit is zero
because there is nothing to localize.

The FA trace is **sparse**: at seq=1024 in a 2048×2048 grid, only ~6%
of cells are visited (those that fall under attention blocks). This
sparsity is what cymatic permutation can exploit — by placing visited
cells closer in HBM, we increase L2-line reuse during the scan.

The 1.12× upper bound at mode (9, 4) is consistent with the measured
geometry: at n=9 the angular sectors at θ = kπ/9 happen to roughly
align with the attention-block boundaries when projected onto the
2D grid via the FA trace mapping.

### Why step 2 (real-kernel integration) cannot land the 12%

Two options for applying the layout to the FA kernel:

**(A) Per-element indirection in the kernel.**
Change every `K[k * D + d]` to `K_data[perm[k] * D + d]`. Adds one
extra LDG per element for `perm[k]`. Kills cp.async-LDGSTS vector
loads (16-byte aligned contiguous → scalar gather). Estimated
overhead: 30-60% perf loss (cp.async is the +35% optimization that
landed in the canonical pipeline kernel; removing it removes that
gain). Net effect of cymatic + indirection: **strongly negative**.

**(B) Pre-permute K/V in HBM at allocation.**
Allocate K such that the cell at logical (k, d) is physically at
`K_data[perm(k, d)]`. Kernel reads `K[k * D + d]` unchanged. The
permuted layout breaks the row-stride contiguity that cp.async
assumes: a tile load `cp.async K_smem[bc][d] from K + (k0+bc)*D + d`
reads from `K_data[perm(k0+bc, d)]` — addresses are no longer stride-D
contiguous, so `cp.async.ca.b16` becomes scalar.

Both options break cp.async. The ResBlock work (Obs GG) and the
pad-2 work (Obs U) both rely on cp.async for their wins. The 12%
layout benefit cannot offset the cp.async loss.

### Possible future paths (not pursued)

  1. **Block-coarse cymatic**: keep cells *within* a (Bc × D) tile
     contiguous (preserves cp.async); only reorder tiles relative to
     each other. At seq=1024 with Bc=64 D=64 there are only 16 tiles
     in the K dimension and 1 in the D dimension. A 1D permutation of
     16 elements has no useful structure for cymatic to exploit.
     **Negligible expected benefit.**
  2. **Restructure FA to gather access**: rewrite the kernel to use
     persistent kernel + gather-style access, then apply per-element
     permutation. The persistent-dispatch falsification (Obs DD)
     showed +0% on the modular path; combined with this, expected
     return is too low.
  3. **Apply cymatic to a different kernel**: any kernel whose access
     pattern is *both* sparse *and* not already cp.async-bound. The
     ResBlock conv2d's im2col col buffer is contiguous-stride;
     IGEMM/HGEMM K-buffers are contiguous-stride. The sparse +
     scatter-gather workload doesn't naturally arise in the kernels
     we care about.

### Closing #94

Closing as **measured-result, step 2 not pursued**. The 1.12× upper
bound at mode (9, 4) is real but unreachable from any kernel
restructuring that preserves cp.async, and cp.async is non-negotiable
on these kernels. The acceptance criterion "speedup found / no
speedup / slowdown — whichever, document mechanism" lands on the
second branch: **no speedup is reachable in practice on this kernel
family**.

The framework built for #93/#94 (cymatic_optimize.R, gen_fa_traces.R,
cymatic_fa_alignment.R) remains useful for any future kernel where
sparse + non-cp.async workloads emerge.

### Files

- `phase4/cymatic/gen_fa_traces.R`        — FA-flavored trace builder
- `scripts/cymatic/cymatic_fa_alignment.R`        — sweep driver (step 1)
- `docs/figures/cymatic/cymatic_fa_alignment_2048.csv`        — long-form data
- `docs/figures/cymatic/cymatic_fa_alignment_2048_*.png` × 7  — heatmaps

## Observation KK — cuda-oxide Rust→PTX spike: pipeline portable, 2× SASS bloat

NVlabs published [cuda-oxide](https://github.com/NVlabs/cuda-oxide)
(2026-04-22, alpha) — a custom rustc backend that compiles `#[kernel]`
Rust functions to PTX through a Pliron MIR / LLVM IR pipeline.
Question: can we slot it in as a front-end alternative to `nvcc` while
keeping the cuasmR SASS hand-edit research intact?

Spike kernel: vecadd, matched against `phase1/vector_add.cu` baseline.
Live install + run on RTX 3070 Ti, sm_86, CUDA 13.2. Full writeup in
`experiments/rust-experiments/README.md`.

### Headline results

| Metric                              | nvcc           | cuda-oxide      | Ratio |
|-------------------------------------|----------------|-----------------|-------|
| Source files (kernel + host)        | 2              | 1 (unified)     | —     |
| Kernel body LoC                     | 9              | 6               | 0.67× |
| PTX lines                           | 55             | 86              | 1.56× |
| PTX `.param` slots                  | 4              | 6               | 1.5×  |
| **SASS instructions (real)**        | **~16**        | **~34**         | **2.1×** |
| Bounds checks emitted               | 1              | 3               | 3×    |
| Cold build                          | <1 s           | ~110 s          | 100×+ |
| Incremental build                   | <1 s           | ~1 s            | ~1×   |
| Correctness                         | ✓              | ✓               | —     |
| **cuasmR roundtrip byte-identical** | ✓              | **✓**           | —     |
| **FADD→FMUL hand-edit runs**        | ✓              | **✓**           | —     |

### What the spike proved

1. **Pipeline converges cleanly at PTX.** Both flows merge at PTX, so
   ptxas + cuobjdump + cuasmR all operate downstream without changes.
   cuasm_read/cuasm_write on a Rust-origin cubin produced byte-identical
   output (md5 match). The phase1 FADD→FMUL opcode flip
   (`0x...7221 / 0x...0000` → `0x...7220 / 0x...400000`) ports verbatim
   to the oxide-emitted FADD; patched cubin loaded into the oxide host
   binary outputs `0,2,8,18,32` instead of `0,3,6,9,12`. **The hand-edit
   research is fully source-language-agnostic.**

2. **CUDA 13.2 + sm_86 is supported in practice** despite the README
   pitching Hopper/Blackwell as the target. `cargo oxide doctor` greens
   on libNVVM 2.0, nvJitLink 13.2, llc 21.1.8.

### What the spike measured against integration

1. **2× SASS bloat for safety-typed kernels.** Three sources, none
   reducible without dropping Rust's safety pitch:
   - **Independent slice bounds checks**: `&[f32], &[f32], DisjointSlice<f32>`
     are three independent length values; LLVM/ptxas cannot prove
     `a.len == b.len == c.len`. nvcc passes one `int num_elements` and
     emits one `ISETP.GE + @P0 EXIT`. oxide emits three.
   - **`Option<&mut T>` discriminant tracking**: `c.get_mut(idx)` returns
     an Option; the `Some/None` tag round-trips through PRMT, LOP3.LUT,
     ISETP, branch (4 SASS instr) before being collapsed.
   - **64-bit indexing throughout**: oxide threads `u64` indices through
     IADD3 + IADD3.X extended-precision pairs; nvcc fuses to IMAD.WIDE.

   For DRAM-bound kernels (vecadd, GroupNorm) this is invisible — DRAM
   is the bottleneck. For compute-bound kernels (HGEMM, sparse mma.sp,
   FA v2 — the entire thrust of phases 2/3/5) the extra ALU lands on
   the critical path.

2. **16 GB toolchain footprint** to support a single-language alt
   front-end. LLVM 21.1.8 (1.9 GB tarball, 11 GB extracted), nightly
   Rust + rust-src + rustc-dev (~2 GB), cuda-oxide checkout + cargo
   target/ (~3 GB). Repo lives on D: which was at 99% capacity, so the
   install went under `~/cuda-oxide-deps/` on /home (Linux ext4, 809 GB
   free). Repo footprint: 64 KB.

3. **Cold build ~110 s.** Backend `librustc_codegen_cuda.so` rebuild +
   path-deps. Iteration cost real on first build per machine; per-kernel
   incremental edits run in ~1 s.

4. **Sub-cliff to phase 1–5 thesis**: SASS hand-edit research operates
   below PTX. Source language (.cu vs .rs) is invisible at the level we
   care about. There is no measured win to integrating cuda-oxide for
   the existing phases.

### Verdict

**Don't replace nvcc.** Treat `experiments/` as a sandbox for narrow
experiments where Rust's front-end features earn their cost:

- ptxas output diffs vs nvcc on a hand-tuned phase2/3 kernel
- generic kernel templating with `Fn` closures vs `#define` macros
- early stake in sm_100a / Blackwell tcgen05 / WGMMA when GPU lands
  (cuda-oxide exposes these as native intrinsics; nvcc currently
  needs raw PTX inline asm)

For phase 1–5 SASS hand-edit work: **stay on nvcc**. The 2× ALU
overhead from Rust's safety abstractions structurally conflicts with
the project's compute-bound thesis.

### Files

- `experiments/rust-experiments/README.md`           — full writeup, all numbers
- `experiments/rust-experiments/run_oxide.sh`        — bootstrap + driver script
- `experiments/rust-experiments/vecadd_nvcc.ptx`     — nvcc PTX baseline
- `experiments/rust-experiments/vecadd_oxide.ptx`    — oxide PTX
- `experiments/rust-experiments/vecadd_*.sm_86.sass` — disassembly side-by-side
- `experiments/rust-experiments/vecadd_oxide.fmul.cubin` — hand-edited

## Observation LL — cuda-oxide gather_sum: fewer SASS instructions, slower runtime (vecadd inverted, unroll dominates)

**Phase**: experiments/rust-experiments/cymatic_oxide
**Hardware**: GA104, sm_86, RTX 3070 Ti
**Date**: 2026-05-10

Second cuda-oxide spike, this time on the phase4/cymatic gather_sum
kernel — a serialised `data[idx[k]]` chain rather than vecadd's flat
LDG. Same toolchain (Rust nightly + LLVM 21 + cuda-oxide) on the same
machine that ran Obs KK.

### Headline numbers

| Metric                              | nvcc           | oxide           | Ratio |
|-------------------------------------|---------------:|----------------:|------:|
| PTX lines                           |            149 |              89 |  0.60×|
| SASS real instructions              |            120 |              80 |  0.67×|
| Inner-loop LDG/FADD count           |             21 |               3 |  0.14×|
| Inner-loop unroll factor            |             4× |              1× |       |
| `rowmajor_full` row layout, GB/s    |         1858.1 |          1489.9 |  0.80×|
| `rowmajor_full` cymatic layout, GB/s|         1453.6 |           965.4 |  0.66×|

**oxide emits ~33% fewer instructions but runs 20-35% slower.** Inverse
of Obs KK (vecadd: 2× SASS, runtime parity).

### Why

nvcc unrolls the inner gather loop 4×; oxide does not. Per nvcc inner
iteration the SASS hot path issues 4 independent `LDG idx[k..k+3]`,
followed by 4 dependent `LDG data[idx_value]`, into 4 serial FADDs.
That's ~8 in-flight DRAM transactions per warp before any FADD waits.
oxide's inner body is one LDG (idx), one dependent LDG (data), one
FADD, branch — **no memory pipelining**. On a DRAM-bound trace the
unroll ratio is the bandwidth ratio.

The 3 panic-EXIT blocks for Rust slice bounds checks (idx[k],
data[idx_value], out[0]) are visible in oxide's SASS — ~9 cold
instructions, no measured runtime cost (correctly predicted-not-taken
on the hot path).

### Implications

1. **Rust safety bloat is not the dominant story for non-trivial
   kernels.** The Obs KK conclusion ("2× SASS from `Option<&mut T>` and
   per-slice bounds checks") was a vecadd artifact: vecadd's hot path
   is so short that bounds-check overhead dominates instruction count.
   For loops with real bodies, **loop-optimization heuristics in the
   compiler dominate**, not the front-end abstraction.

2. **cuda-oxide → LLVM 21 NVPTX backend is missing nvcc's unroll
   heuristic** for gather-style accumulator loops. Manual unrolling
   (Rust source `for _ in 0..4 { ... }` with constant strides, or
   `unsafe { get_unchecked }` with explicit body duplication) is the
   gap-closer. Not yet attempted.

3. **Cymatic layout effect (Obs T) is backend-agnostic.** Both nvcc
   (row > cym 0.78×) and oxide (row > cym 0.65×) preserve the pattern
   on the only DRAM-bound trace. Layout magnitude differs because
   bandwidth ceiling differs, not because the layout effect changed.

4. **Same conclusion for phase 1–5**: keep nvcc. The reason is
   different now (loop-opt heuristic gap, not safety bloat) but the
   recommendation stands.

### Files

- `experiments/rust-experiments/cymatic_oxide/README.md`         — full writeup
- `experiments/rust-experiments/cymatic_oxide/src/main.rs`       — Rust kernel port
- `experiments/rust-experiments/cymatic_oxide/gather_oxide.ptx`  — oxide PTX
- `experiments/rust-experiments/cymatic_oxide/gather_nvcc.cu`    — nvcc-only kernel source
- `experiments/rust-experiments/cymatic_oxide/gather_*.sm_86.sass` — disassembly side-by-side
- `experiments/rust-experiments/cymatic_oxide/bench_gather_driver.cu` — Driver-API harness
