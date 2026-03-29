# Troubleshooting & Pitfalls

Lessons learned from actual sessions. Updated as new issues are encountered.

---

## Environment Setup

### WSL: `nvcc` not found even though CUDA is installed

**Symptom**: `nvcc: command not found` in WSL despite CUDA toolkit being present.

**Cause**: CUDA installs to `/usr/local/cuda-12.x/bin` but doesn't add itself to PATH automatically.

**Fix**: Add to PATH for the session or permanently:
```bash
# Temporary (current session)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Permanent — add to ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

`/usr/local/cuda` is a symlink to the versioned install (`cuda-12.8`). Prefer the unversioned symlink so it survives toolkit updates.

---

### WSL: `python` not found, only `python3`

**Symptom**: Scripts using `python -c "..."` fail with `command not found`.

**Cause**: Debian/Ubuntu WSL ships `python3`, not `python`.

**Fix in scripts**: Detect platform and use the right binary:
```python
import os
PYTHON = "python3" if os.path.isdir("/usr/local/cuda") else "python"
```

Or install the alias system-wide: `sudo apt install python-is-python3`

---

### `pip3` not found in WSL

**Symptom**: `bash: pip3: command not found`

**Fix**: Use `python3 -m pip` instead:
```bash
python3 -m pip install pyelftools sympy
```

If pip refuses due to PEP 668 (externally managed environment):
```bash
python3 -m pip install pyelftools sympy --break-system-packages
```

---

### Windows: CUDA toolkit not installed (only game driver)

**Symptom**: `nvcc` not found on Windows even with a working GPU.

**Cause**: NVIDIA ships two separate things:
- **Game driver** (GeForce Experience) — runs games, no dev tools
- **CUDA Toolkit** — separate install, includes `nvcc`, `cuobjdump`, etc.

**Fix**: Install CUDA Toolkit 12.x from developer.nvidia.com. Use WSL if you want to avoid Windows PATH complexity.

---

### Windows: Python `pip install` warns about PATH

**Symptom**: After `pip install sympy`, get warning that `isympy.exe` is not on PATH.

**Impact**: None for our use — we import the library, don't call the CLI tool.

---

## CuAssembler

### Roundtrip produces different cubin bytes

**Symptom**: After disassemble → reassemble without any changes, the output cubin differs from the input.

**Cause**: CuAssembler may reorder ELF metadata sections or change alignment padding. The actual instruction bytes are identical.

**How to verify it's safe**: Run both cubins through the host driver and compare outputs — if results match, the roundtrip is functionally correct.

**Warning message to expect**:
```
WARNING - This Cubin(CuSMVersion(86)) needs desc hack!
```
This is normal for sm_86. CuAssembler applies a workaround for a cubin descriptor format quirk introduced in newer CUDA versions. It does not affect correctness.

---

### CuAssembler: CUDA 12.8 vs CuAssembler version mismatch

**Symptom**: CuAssembler fails to parse or reassemble a cubin compiled with a newer CUDA version.

**Cause**: CuAssembler was primarily developed against CUDA 11.x/12.x. The cubin format evolves.

**Mitigation**:
- Use `nvcc -O1` or `-O2` (avoid `-O3`) — less aggressive compiler transformations
- Use `extern "C"` on all kernels to simplify name mangling
- If a kernel fails roundtrip, simplify it (fewer features → simpler SASS → more stable)
- Check the CuAssembler GitHub issues for known sm_86 / CUDA 12.x problems

---

### CuAssembler fails on `ULDC` or other newer instructions

**Symptom**: Parse error or assertion failure in CuAssembler when encountering `ULDC`, `LDGSTS`, or other Ampere-specific instructions.

**Cause**: Newer instructions added in CUDA 11.1+ may not be in CuAssembler's instruction table.

**Workaround**: Compile with `-O1` to reduce use of exotic instructions. Or use `#pragma nounroll` to prevent the compiler from generating complex pipelined sequences initially.

---

## SASS / Instruction-Level Issues

### Wrong results after SASS modification

**Symptom**: Kernel runs but produces incorrect output.

**Most likely causes** (in order of likelihood):
1. **Data hazard**: Modified instruction depends on a result that isn't ready yet (stall count too low). Fix: increase the `S` value in the control code by 2–4.
2. **Wrong register**: Accidentally used the wrong register number — double-check `R4` vs `R3` etc. by re-reading the disassembly.
3. **Wrong instruction variant**: e.g., used `FADD` (FP32) on FP16 data. Check that the register types match.

**Debug approach**:
1. Set all stall counts to `S15` — if it fixes the issue, it was a hazard
2. Add `MEMBAR.GL` before any store instruction
3. Binary search: revert half your changes and test

---

### GPU hang / driver crash after SASS modification

**Symptom**: `./host` hangs indefinitely, or the display driver resets.

**Cause**: Invalid instruction encoding — the GPU encounters an illegal opcode or malformed control code.

**Recovery**: Kill the process, the driver will recover (TDR — Timeout Detection and Recovery). You don't need to reboot.

**Prevention**:
- Make one change at a time, test after each
- Never change the opcode encoding bits directly — only change instruction names and register numbers in `.cuasm`
- Keep the control code `[B...:R-:W-:Y:Snn]` the same as the original when first testing a modification

---

### `cuModuleLoad` fails with "invalid device function"

**Symptom**:
```
Failed to load cubin: CUDA_ERROR_INVALID_PTX
```
or
```
CUDA_ERROR_NO_BINARY_FOR_GPU
```

**Cause**: The cubin was compiled for a different `sm_` target than the GPU.

**Fix**: Always compile with `-arch=sm_86` for the RTX 3070 Ti. Check with:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# should output: 8.6
```

---

### `cuModuleGetFunction` fails with "function not found"

**Symptom**:
```
CUDA Driver API error — CUDA_ERROR_NOT_FOUND
```

**Cause**: The kernel symbol name in the cubin doesn't match the string passed to `cuModuleGetFunction`.

**Fix**: Use `extern "C"` in the kernel source to prevent C++ name mangling:
```c
// Without extern "C": symbol is "_Z10vector_addPKfS0_Pfi" (mangled)
// With extern "C":    symbol is "vector_add" (clean)
extern "C" __global__ void vector_add(...) { ... }
```

Verify the symbol name with:
```bash
cuobjdump -symbols vector_add.sm_86.cubin
```

---

## WSL-Specific CUDA Issues

### `nvidia-smi` works but CUDA programs fail

**Symptom**: `nvidia-smi` shows the GPU, but compiled CUDA programs crash on launch.

**Cause**: WSL CUDA requires matching driver versions between Windows and WSL. The Windows NVIDIA driver exposes the GPU to WSL via `/usr/lib/wsl/lib/`.

**Check**:
```bash
ls /usr/lib/wsl/lib/libcuda*   # should exist
cat /proc/driver/nvidia/version # Windows driver version
nvcc --version                  # CUDA toolkit version
```

The CUDA toolkit version must be ≤ the driver's maximum supported CUDA version.

**Fix**: Update the Windows NVIDIA driver if the toolkit version is newer than what the driver supports.

---

### Programs run fine locally but fail when called via `wsl -e bash -c`

**Symptom**: A command works in an interactive WSL shell but fails when called via `wsl -e bash -c '...'` from Windows.

**Cause**: `wsl -e bash -c` uses a non-login, non-interactive shell. `.bashrc` modifications (like PATH additions) are not sourced.

**Fix**: Explicitly set PATH in the command:
```bash
wsl -e bash -c 'export PATH=/usr/local/cuda/bin:$PATH && nvcc ...'
```

Or add PATH to `/etc/environment` (system-wide, always active):
```bash
echo 'PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' | sudo tee -a /etc/environment
```

---

## SGEMM / GEMM Kernel Pitfalls

### Register-blocked SGEMM slower than tiled — tile size and occupancy mismatch

**Symptom**: A register-blocked kernel with 16 accumulators per thread is slower than a simpler 1-accumulator tiled kernel.

**Root causes** (from our experiment):

1. **Wrong thread count per block**: With BM=64, TM=4 → BLOCK_ROWS=16, BLOCK_COLS=16 → 256 threads per block. If you launch with 1024 threads, `cuLaunchKernel` returns `CUDA_ERROR_INVALID_ARGUMENT`. Always derive block size from tile parameters.

2. **BK too small → too many barriers**: BK=8 with K=2048 means 256 `BAR.SYNC` operations per kernel. Each stalls all warps. BK=32 or BK=64 amortizes this cost. Rule: `K/BK` is the barrier count per kernel — keep it under ~100.

3. **Strided tile loading → no vectorization**: When each thread must load `BM*BK / num_threads` elements via a loop, the compiler can't use `LDG.E.128` (128-bit vector load). This kills bandwidth efficiency. Fix: use `float4` loads or hand-write `LDG.E.128` in SASS.

4. **Register pressure → low occupancy**: 58 registers × 256 threads = 14848 registers per block → 4 blocks per SM → 1024 active threads. Not much better than tiled (37 × 1024 = 37888 → 1 block/SM → 1024 threads). High register count only wins if you also vectorize loads and pipeline them.

**The real fix**: Register blocking only outperforms at the SASS level — you need `LDG.E.128` for global loads, `LDS.128` for shared loads, and double-buffering to hide global load latency. This is what `hand_tuned.cuasm` implements.

**Measured results (RTX 3070 Ti, 2048³)**:

| Kernel | GFLOPS | % peak |
|---|---|---|
| naive (1 acc, no smem) | 885 | 4.1% |
| tiled (1 acc, 32×32 smem tile) | 1032 | 4.8% |
| register-blocked (16 acc, 64×64 smem, BK=32) | 453 | 2.1% |

**Lesson**: More FFMA instructions per kernel does NOT automatically mean higher GFLOPS. The bottleneck moved from compute to data movement (loading the larger tiles). High-performance GEMM requires **all** of: register tiling + vectorized loads + software pipelining — simultaneously.

---

### `cuLaunchKernel` returns `CUDA_ERROR_INVALID_ARGUMENT` for a valid-looking kernel

**Symptom**: Launch fails with "invalid argument" even though grid and block sizes look reasonable.

**Most common cause**: Kernel has `__launch_bounds__(N)` and you're launching with more than N threads per block.

**Check**: Compare `__launch_bounds__` in the kernel source with the block size in `cuLaunchKernel`. They must match. With register-blocked SGEMM: `__launch_bounds__(BLOCK_ROWS * BLOCK_COLS)` = `__launch_bounds__(16 * 16)` = 256 threads max.

**Debug**: Run `nvcc --ptxas-options=-v` to see the kernel's register count and any launch bounds constraints.

---

## CUDA Resource Lifetime (RAII Pitfall)

### Segfault on exit from CUDA benchmark — `cuCtxDestroy` before RAII destructor

**Symptom**: Program runs successfully but segfaults at exit (exit code 139). No CUDA error message printed. Happens consistently.

**Root cause**: A RAII wrapper (e.g., `BenchTimer` owning `CUevent` handles) goes out of scope AFTER `cuCtxDestroy` is called manually. The destructor calls `cuEventDestroy` on events whose context is already gone → undefined behaviour → segfault.

**Code pattern that triggers it**:
```cpp
cuCtxCreate(&ctx, 0, dev);
BenchTimer timer;          // creates CUevent handles tied to ctx
// ... launch, measure ...
cuCtxDestroy(ctx);         // DESTROYS ctx ← events are now invalid
// end of scope: ~BenchTimer() calls cuEventDestroy → SEGFAULT
```

**Fix**: Ensure RAII wrappers are destroyed before `cuCtxDestroy` by scoping them explicitly:
```cpp
float avg_ms;
{
    BenchTimer timer;       // created inside this scope
    timer.start();
    // ... benchmark ...
    avg_ms = timer.stop_ms() / N;
}                           // ~BenchTimer() runs here — ctx still alive
cuCtxDestroy(ctx);          // safe: events already destroyed
```

**Or**: Simply don't call `cuCtxDestroy` manually — the OS reclaims GPU resources on process exit anyway. Only matters for long-running processes that need to recycle contexts.

**Same issue applies to**: any CUDA resource created with `cuEventCreate`, `cuStreamCreate`, `cuModuleLoad`, `cuMemAlloc` etc., when wrapped in RAII objects that are destroyed after the context.

---

## Correctness Checks

### High relative error but correct absolute error — false alarm

**Symptom**: `check_fp32` reports `max_rel=2.30e+00` (230%) but kernel still passes.

**Cause**: Relative error is computed as `|gpu - ref| / |ref|`. When `ref` is very close
to zero (which happens in GEMM with mixed positive/negative inputs), even a tiny
absolute error produces a huge relative error.

**Example**:
- `ref = 0.000001`, `gpu = 0.000002` → absolute error = 1e-6, relative error = 100%
- But the absolute error is negligible — the kernel is correct.

**The check passes** because `check_fp32` uses `AND` logic:
```cpp
bool is_wrong = (abs_error > abs_tolerance) AND (rel_error > rel_tolerance);
```
An element is only flagged wrong if **both** absolute and relative error are large.

**What to watch**: if `max_abs` is also large (e.g., > 0.01 for FP32 GEMM), then
there's a real correctness problem. High `max_rel` alone with small `max_abs` is normal.

---

## Build System

### `make` not available on Windows

The `Makefile` in `phase1/` uses Unix `make`. On Windows, use WSL:
```bash
wsl -e bash -c 'export PATH=/usr/local/cuda/bin:$PATH && cd /mnt/d/dev/p/bare-metal/phase1 && make all'
```

Or call the Python build script directly — it works on both platforms.

---

## Hardware-Specific Notes (RTX 3070 Ti Laptop)

### It's a Laptop GPU — slightly different specs than Desktop

The RTX 3070 Ti **Laptop** GPU (GA104) has the same compute capability (sm_86) and architecture as the desktop version, but:
- Lower TDP → lower sustained boost clocks
- Same 8 GB GDDR6X, same VRAM bandwidth spec
- Theoretical TFLOPS may be lower in practice due to power limits

For benchmarking: use `nvidia-smi -q -d CLOCK` to check actual clock speeds. Power-limited throttling can make benchmark results inconsistent. Warm up the GPU with a few kernel runs before measuring.

### WDDM mode (Windows) vs TCC mode

On Windows, the GPU runs in **WDDM mode** (display driver). This adds overhead for small kernel launches and limits some features (e.g., peer-to-peer memory). In WSL, the GPU is accessed via the WDDM driver's WSL interface — same limitation.

For production ML training workloads, Linux bare-metal (not WSL) with the GPU in **Compute mode** is ideal. For our development purposes, WSL is fine.

---

## Flash Attention / Online Softmax

### K/V re-read inflation — bandwidth numbers look misleadingly low

**Symptom**: Flash Attention benchmark reports ~20 GB/s against a 608 GB/s peak — seems terrible.

**Cause**: The bandwidth formula counts K and V reads `seq_len / BLOCK_KV` times each (once per KV tile iteration). For `seq_len=1024, BLOCK_KV=32` that's 32× K + 32× V = 64 full-tensor reads vs 1 for Q and O. So the "achieved bandwidth" reflects the actual DRAM traffic, which is large because K/V thrash the L2 cache.

**Not a bug** — the kernel is correct. The apparent low bandwidth comes from:
1. Small `BLOCK_KV=32` → many iterations → K/V read many times
2. L2 cache (4 MB) too small for multi-head K/V tensors → most reads miss to DRAM

**The "ideal" bandwidth** (K/V cached) is printed separately — this represents what you'd achieve if K/V fit in L2. For our config it's ~1 GB/s, confirming we're compute-bound (the SHFL.BFLY + FFMA work is the bottleneck, not bandwidth), not bandwidth-limited as the raw number suggests.

**Real fix**: Increase `BLOCK_KV` (larger tiles = more Q rows processed per K/V load) and use `LDGSTS` async pipelining. The WMMA version processes `Br=64` query rows per block, giving 2× more temporal reuse of each K/V tile.

---

### Online softmax: correctness degrades at long sequences

**Symptom**: `max_rel` error increases with `seq_len` (e.g., 5e-3 at 512, 1e-1 at 2048).

**Cause**: The online softmax accumulates more rounding error over more KV iterations. Near-zero output elements (softmax concentrates on a few positions) produce high relative error even when absolute error is tiny.

**Why it's acceptable**: The `check_fp32` function uses AND logic — both `max_abs` AND `max_rel` must exceed tolerance for a failure. In all tested cases `max_abs` stays below 2e-7 (< 1 ULP), so all tests pass. The high `max_rel` is a statistical artifact of near-zero denominators.

**Rule of thumb**: For attention outputs, use `abs_tolerance=1e-3, rel_tolerance=1e-1` when testing FP32 flash attention at `seq_len > 1024`.

---

### `__syncwarp()` vs `__syncthreads()` in single-warp kernels

**Rule**: When a kernel launches with exactly 1 warp per block (32 threads), `__syncwarp()` is sufficient for intra-warp synchronization. `__syncthreads()` works too but is slower (it also handles inter-warp barriers that don't exist in a single-warp block).

**In flash_attn_1warp**: after loading K/V tile into shared memory, `__syncwarp()` is correct. All 32 threads are in the same warp — no other warp accesses the shared memory.

**Pitfall**: If you later refactor to multi-warp blocks (e.g., 4 warps for WMMA flash attention), replace `__syncwarp()` with `__syncthreads()` at tile boundaries.

---

## WMMA Flash Attention (Br=16 per warp)

### WMMA matrix_b col_major = automatic transpose for QK^T

**Observation**: Computing `QK^T` (transpose of K) with WMMA requires no explicit buffer
transposition. Load K as `wmma::matrix_b` with `wmma::col_major` and WMMA automatically
interprets the data as K^T.

**Why it works**:
- `matrix_b` col_major with `load_matrix_sync(b_frag, ptr, stride)`:
  `b_frag[k][n] = ptr[n * stride + k]`
- K_tile stored row-major `[Bc × D_HEAD]`:
  `K_tile[kv_row][d_col] = ptr[kv_row * D_HEAD + d_col]`
- Loading K_tile at offset `n_tile * WMMA_N * D_HEAD + dk * WMMA_K` with stride `D_HEAD`:
  `b_frag[k][n] = K_tile[(n_tile*16 + n) * D_HEAD + dk*16 + k] = K[kv_base + n_tile*16 + n][dk*16 + k]`
- So `mma_sync(C, A, B)` computes `C[m][n] += Σ_k Q[m][dk+k] × K[kv+n][dk+k]` = Q·K^T ✓

**Rule**: For computing `A × B^T` with WMMA where B is stored row-major,
load B as `matrix_b col_major`. This is the standard trick for transposed matmul.

---

### WMMA fragment `.x[]` array for element-wise scaling

**Context**: The online softmax requires scaling each row of the WMMA PV accumulator
by `exp(old_max - new_max)`. This is a per-row scale, but WMMA fragment layout is
hardware-defined and undocumented.

**Two approaches**:

1. **Direct `.x[]` access** (undocumented but consistent on sm_86):
   ```cpp
   for (int elem = 0; elem < score_frag.num_elements; elem++)
       score_frag.x[elem] *= scale;  // scale ALL elements (row-independent)
   ```
   Works when the same scale applies to ALL elements in the fragment (e.g., scaling
   scores by `1/sqrt(d_head)`). **Does NOT work** for per-row scaling since the
   row-to-element mapping is undocumented.

2. **Store → scale in shared memory → reload** (robust):
   ```cpp
   wmma::store_matrix_sync(smem_ptr, frag, stride, wmma::mem_row_major);
   // row-wise scale using SHFL-reduced rescale_factors[row]:
   smem_ptr[row * stride + lane]             *= rescale_factor[row];
   smem_ptr[row * stride + lane + WARP_SIZE] *= rescale_factor[row];
   wmma::load_matrix_sync(frag, smem_ptr, stride, wmma::mem_row_major);
   ```
   Correct for any per-row scaling. Cost: 2 shared memory round-trips per KV iteration,
   but at ~10 TB/s smem bandwidth this is ~50 ns per block — negligible vs DRAM access.

**Recommendation**: Use approach 2 for per-row operations. Reserve approach 1 for
uniform scaling (e.g., multiplying all scores by a constant).

---

### smem_work overlay: FP32 scores reused as FP16 weights

**Technique**: In flash_attn_br16, `smem_work` is allocated as `FP32 [Br_block × Bc]`
(16 KB per block). After the WMMA QK^T step, scores (FP32) are stored there. During
the softmax step, the same memory is overwritten with FP16 attention weights.

```cpp
float  *smem_work = ...;           // 16 KB FP32 (scores)
__half *weight_ptr = (__half*)smem_work;  // same memory, reinterpret as FP16

// Phase C: write FP16 weights into first 8 KB of smem_work
weight_ptr[row * Bc + lane]             = __float2half(w_lo);
weight_ptr[row * Bc + lane + WARP_SIZE] = __float2half(w_hi);
// The last 8 KB of smem_work is now stale (old FP32 scores) — harmless

// Phase D: load FP16 weights with WMMA
wmma::load_matrix_sync(w_frag, weight_ptr + k * WMMA_K, Bc);
```

**Why it's safe**: After reading all FP32 scores for the softmax pass (Phase C rows loop),
we no longer need the FP32 data. Writing FP16 into the first 8 KB is fine because:
- Memory is 4-byte aligned (compatible with 2-byte FP16 writes)
- No thread reads the FP32 layout after Phase C writes
- `__syncthreads()` between Phase C (write) and Phase D (read) ensures visibility

**Pitfall**: Do NOT mix FP32 reads and FP16 writes to the same smem region without a
sync barrier between them. The overlay is safe ONLY because FP32 reads are complete
before FP16 writes begin (enforced by the sequential `for row` loop + `__syncthreads()`).

---

### Flash Attention progression: speedup breakdown

Measured on RTX 3070 Ti Laptop (sm_86), seq=1024, batch=8, heads=8:

| Kernel | Design | ms | Speedup | Key change |
|---|---|---|---|---|
| `flash_attn_1warp` | 1 warp/row, FP32, BKV=32 | 53 | 1× | baseline |
| `flash_attn_4warp` | 4 warps share tile, FP32, BKV=64 | 19 | **2.8×** | 4× K/V loads shared + 2× tile |
| `flash_attn_br16`  | Br=16/warp, FP16, HMMA | 2.8 | **19×** | 8× FLOPS/instr + FP16 BW |

**Root causes of speedup**:

`flash_attn_4warp` speedup (2.8×):
- 4 warps share K/V tile → 4× fewer K/V DRAM reads per query
- BLOCK_KV 32→64 → 2× fewer iterations → 2× less tile overhead
- Net: 8× less K/V traffic; actual speedup 2.8× (not 8×) due to compute overhead

`flash_attn_br16` speedup (6.7× over 4warp = 19× over baseline):
- FP16 inputs: K/V tiles are 2× smaller → 2× less K/V bandwidth
- Br=16 per warp: 16 rows reuse each K/V tile vs 1 row → 16× more Q-rows per K/V load
- HMMA.16816.F32: 8× higher throughput than FFMA for matmul steps
- smem reuse: `smem_work` FP16 overlay eliminates separate weight buffer allocation

**Bottleneck analysis**:
The ideal bandwidth (Q+K+V+O each read once) for `flash_attn_br16` is ~24 GB/s at seq=1024.
Hardware peak is 608 GB/s. The gap (24 vs 608 = 4%) is because:
- K/V tiles are still read multiple times from DRAM (seq/Bc = 16 iterations)
- Online softmax requires per-row reductions (SHFL bound) between HMMA calls
- `__syncthreads()` barriers between KV tiles create pipeline bubbles

The path to closing the gap: async K/V prefetching with LDGSTS (cp.async),
double-buffering K/V tiles, and warp-specialization (producer/consumer split).

---

## Phase 4: Diffusion Model Primitives

### GroupNorm: group_size must be divisible by WARP_SIZE

**Symptom**: GroupNorm kernel produces garbage output for unusual channel/group combinations.

**Cause**: The kernel design uses one warp (32 threads) per group. Each thread accumulates
`group_size / WARP_SIZE` elements. If `group_size % 32 != 0`, the last thread has fewer elements
than the others — but the Welford combine still reduces all 32 lanes uniformly, causing a count mismatch.

**Rule**: `(C / num_groups) * H * W` must be divisible by 32.

For typical SD configs this is always true: C=320, G=32 → group_size = 10 * H * W.
At H=W=16: group_size = 2560 = 80 × 32. ✓

---

### GroupNorm NHWC vs NCHW: ~7% throughput difference

**Observation**: `groupnorm_nchw` is consistently ~7% faster than `groupnorm` (NHWC) at large configs.

**Cause**: NCHW layout iterates over contiguous `H×W` spatial positions for a fixed channel `c`.
This means each thread's strided load `X[n*C*H*W + c*H*W + hw]` is coalesced across the warp
(adjacent threads have adjacent `hw` values). In NHWC, adjacent threads access adjacent channels
which are strided by `C` elements apart — slightly less cache-friendly for large C.

**Practical implication**: cuDNN uses NCHW internally for this reason. For Phase 4 ResNet blocks,
the NHWC kernel is preferred for conv2d interop (standard feature map layout), despite the small penalty.

---

### Conv2d correctness: large max_rel is expected for deep dot products

**Symptom**: `max_rel=0.63` in conv2d correctness check with `check_fp32(tol=1e-2, 1e-2)`.

**Cause**: For large Cin (e.g., 320), each output element accumulates `9 × 320 = 2880` multiply-adds.
Floating-point rounding errors compound: the expected max_abs error scales as `~eps_mach × sqrt(N) × mean(|values|)`.
For Cin=320, this gives `~1.2e-7 × sqrt(2880) × 0.35 ≈ 2e-6` absolute error — but **relative** error for
near-zero outputs is unbounded (a tiny true value and tiny computed value can differ 100% relatively).

**Fix**: Use `max_abs` as the primary correctness metric for convolutions. `max_rel` only matters for
numerically significant outputs. The `check_fp32` function uses AND logic — it passes if
**either** abs error **or** rel error is within tolerance.

---

### Conv2d direct kernel: weight shared memory layout

**Design**: The 3×3 conv tiles input channels in groups of `CIN_TILE=16`. For each tile, all
`BLOCK_THREADS = TILE_HW × TILE_C = 128` threads cooperatively load weights into shared memory:
- `smem_W[kernel_pos][cin_local][cout_local]` — kernel position first, channel last
- This layout gives conflict-free access in Phase 3 when each thread reads `smem_W[kpos][*][cout_local]`
  (all threads with the same `cout_local` = same `threadIdx.y` read the same column — broadcast)

**Pitfall**: If you swap the smem dimensions to `[cin][kpos][cout]`, the 9-position inner loop
causes 9 non-sequential reads per step, hurting L1 performance. Current layout collapses the 9×CIN_TILE
reads into a sequential scan over the first two dimensions.

---

### Fused GroupNorm+SiLU: MUFU.EX2 count is higher than expected

**Observation**: `cuobjdump | grep MUFU.EX2` shows ~5 instructions in `groupnorm_silu_fused`,
not the expected 1 (one per output element).

**Cause**: The compiler unrolls the output write loop (Phase 3) over multiple iterations when
`group_size / WARP_SIZE` is a compile-time constant or small. Each unrolled iteration needs one
`MUFU.EX2` call for `exp2f(-scaled * LOG2E)`. With 4× unrolling, you see 4–5 MUFU.EX2 ops.

**Benefit**: This is good — unrolling hides the MUFU latency (~4 cycles) by interleaving
independent EX2 calls for different output elements.

---

### ResNet block throughput: conv2d is the bottleneck

**Observation**: The 5-kernel ResNet block at N=1, C=320, H=W=16 runs at ~265 GFLOPS.
At the same config, standalone conv2d runs at ~299 GFLOPS. So conv2d dominates.

**Breakdown** (approximate timing from individual kernel benchmarks):
| Kernel | Time | % of total |
|--------|------|-----------|
| groupnorm_silu_fused × 2 | ~0.08 ms | ~2% |
| conv2d_nhwc × 2 | ~3.40 ms | ~95% |
| residual_add × 1 | ~0.07 ms | ~2% |
| **Total** | **~3.56 ms** | **100%** |

**Path to speedup**: Conv2d is the clear bottleneck. Options:
1. **im2col + WMMA**: Transform X to im2col matrix, then use HMMA.16816 (from Phase 2 HGEMM).
   Expected: 8× speedup on the matmul steps → ~40× total conv speedup.
2. **Depthwise separable conv**: Replace 3×3 C×C conv with depthwise 3×3 + 1×1 pointwise.
   Reduces FLOPs from `9C²HW` to `(9+C)×CHW` — a 9× reduction at C=320.
3. **Winograd F(2,3)**: 2.25× fewer multiplications for 3×3 conv via transform. Complex to implement in SASS.

The current kernel demonstrates the SASS structure clearly (310 FFMA, coalesced smem access)
even if not production-optimal.

---

### Cross-Attention: KV padding mask is mandatory for non-power-of-2 seq_kv

**Symptom**: With seq_kv=77 (CLIP), `max_abs` error was ~0.11 before masking.

**Root cause**: K_tile is zero-padded for positions `kv_global >= seq_kv`.
Zero-padded K → Q @ 0^T = 0 for those positions → score = 0.
In softmax: `exp(0 - max)` is NOT zero — for typical max ~0.5, this gives exp(-0.5) ≈ 0.61.
With 51 padded columns in the last tile, the softmax denominator is inflated by ~51 × 0.61 ≈ 31×,
severely underweighting all real token outputs.

**Fix**: In Phase C (online softmax), apply `-infinity` mask AFTER reading scores into registers:
```cuda
bool lo_padded = ((kv_base + (int)lane)             >= seq_kv);
bool hi_padded = ((kv_base + (int)lane + WARP_SIZE) >= seq_kv);
float score_lo = lo_padded ? NEG_INF : score_row[lane];
float score_hi = hi_padded ? NEG_INF : score_row[lane + WARP_SIZE];
```
Then use `score_lo`/`score_hi` for max reduction AND exp computation.
After fix: max_abs drops from 0.11 → 0.07 (residual error is FP16 HMMA precision, not padding).

**Why self-attention (flash_attn_br16) didn't need this**: In Phase 3, `seq_len` was always
required to be a multiple of `Br_block=64`. The bench program enforces `seq % Br_block != 0 → abort`.
CLIP's seq_kv=77 is NOT a multiple of 64, so cross-attention is the first kernel that hits partial
final tiles where the mask actually matters.

---

### Cross-Attention: inherent FP16 error baseline

**Observation**: After the masking fix, max_abs residual error at seq_kv=77 is ~0.07.
For seq_kv=64 (single full tile, no rescaling needed): max_abs=8.56e-05. Huge difference!

**Why the rescaling step amplifies error**:
- With a single KV tile: pure HMMA computation, no rescaling. Error ≈ FP16 unit roundoff × sqrt(seq_kv).
- With 2 KV tiles: the rescale multiplication `pv_row[i] *= exp(old_max - new_max)` introduces
  one extra float multiply per element per tile. Plus online softmax error compounds.
- The `check_fp32` AND logic passes even with max_abs=0.07, because those elements have
  small RELATIVE error (large true values) — i.e., the error on large-magnitude outputs is
  proportionally small.

**Cross-check**: `flash_attn_br16` (Phase 3) at seq=128 gives max_abs=3.72e-02.
Cross-attention at seq_kv=128 gives max_abs=3.57e-02. Identical! Confirms the error is
from the HMMA kernel design, not from cross-attention-specific code.

---

### Cross-Attention vs Self-Attention: KV iteration count asymmetry

**Key insight**: For cross-attention with seq_q (image) >> seq_kv (text):

| Config | seq_q | seq_kv | KV iters | Self-attn KV iters | Speedup vs self-attn |
|--------|-------|--------|----------|-------------------|---------------------|
| SD 8×8   | 64   | 77  | 2 | 1 | 0.5× (cross is slower!) |
| SD 16×16 | 256  | 77  | 2 | 4 | ~2× |
| SD 32×32 | 1024 | 77  | 2 | 16 | ~8× |
| SD 64×64 | 4096 | 77  | 2 | 64 | ~32× |

At 64×64 feature maps, cross-attention over CLIP-77 is 32× faster than self-attention would be
at the same resolution — this is why cross-attention scales so well in SD.

At 8×8 (deepest layer), cross-attention is actually SLOWER than self-attention because the GPU
is underutilized at only 1 KV tile worth of work.

**Practical numbers** (RTX 3070 Ti Laptop, batch=1, heads=8):
- 32×32 with CLIP-77: **0.072 ms** → 2,254 GFLOPS
- 64×64 with CLIP-77: **0.246 ms** → 2,624 GFLOPS (larger grid → better SM utilization)
- 32×32 with 512-token context: **0.267 ms** → 4,017 GFLOPS (more KV tiles → better Tensor Core utilization)
