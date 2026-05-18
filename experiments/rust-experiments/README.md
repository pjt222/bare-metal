# experiments/rust-experiments — cuda-oxide vs nvcc spike

End-to-end comparison: `Rust → cuda-oxide → PTX → ptxas → SASS → cuasmR patch → run`
vs
`CUDA C++ → nvcc → PTX → ptxas → SASS → cuasmR patch → run` (phase1 baseline).

**Status: live results captured on GA104, sm_86, RTX 3070 Ti.**

## Headline results

| Metric                              | nvcc baseline  | cuda-oxide      | Ratio |
|-------------------------------------|----------------|-----------------|-------|
| Source files (kernel + host)        | 2              | 1 (unified)     | —     |
| Kernel body LoC                     | 9              | 6               | 0.67× |
| PTX lines                           | 55             | 86              | 1.56× |
| PTX `.param` slots                  | 4              | 6               | 1.5×  |
| SASS instructions (real, no padding)| ~16            | ~34             | 2.1×  |
| Bounds checks emitted               | 1              | 3               | 3×    |
| Final cubin size (bytes)            | (similar)      | 3624            | —     |
| End-to-end build time (cold)        | <1 s           | ~110 s          | 100×+ |
| End-to-end build time (incremental) | <1 s           | ~1 s            | ~1×   |
| Correctness                         | ✓              | ✓               | —     |
| cuasmR roundtrip byte-identical     | ✓              | ✓               | —     |
| FADD→FMUL hand-edit runs            | ✓              | ✓               | —     |

## What was actually run

1. `cargo oxide doctor` — all green (Rust nightly, CUDA 13.2, libNVVM 2.0,
   nvJitLink 13.2, libdevice, llc 21.1.8, clang 21).
2. `cargo oxide run vecadd` — `✓ SUCCESS: All 1024 elements correct!`
   on the 3070 Ti.
3. `ptxas -arch=sm_86 -O2 vecadd_oxide.ptx -o vecadd_oxide.sm_86.cubin` — clean.
4. `cuobjdump -sass` on both cubins — saved.
5. `cuasm_read` + `cuasm_write` on Rust-origin cubin — md5 matches input.
6. `cuasm_set` slot 30: `FADD R7,R2,R5` → `FMUL R7,R2,R5`
   (`0x...7221 / 0x...0000` → `0x...7220 / 0x...400000`, same delta as phase1).
7. Patched cubin loaded into oxide-built host binary — outputs `2, 8, 18, 32`
   instead of `3, 6, 9, 12`. **Multiplication confirmed via SASS edit.**

## Files

| File                               | Origin                                | Use                                         |
|------------------------------------|---------------------------------------|---------------------------------------------|
| `vecadd_oxide/src/main.rs`         | cuda-oxide vecadd example             | Rust kernel + host (single source)          |
| `vecadd_oxide/Cargo.toml`          | cuda-oxide vecadd example             | Crate manifest (path-deps to cuda-oxide)    |
| `vecadd_nvcc.ptx`                  | `nvcc -ptx -arch=sm_86 -O1` of phase1 | Baseline PTX from CUDA C++                  |
| `vecadd_oxide.ptx`                 | `cargo oxide run vecadd`              | PTX from Rust                               |
| `vecadd_oxide.sm_86.cubin`         | `ptxas -arch=sm_86 -O2`               | SASS-compiled Rust kernel                   |
| `vecadd_oxide.sm_86.sass`          | `cuobjdump -sass`                     | Disassembled Rust SASS                      |
| `vecadd_nvcc.sm_86.sass`           | `cuobjdump -sass` of phase1 cubin     | Disassembled C++ SASS                       |
| `vecadd_oxide.roundtrip.cubin`     | `cuasm_read` + `cuasm_write`          | Byte-identity proof                         |
| `vecadd_oxide.fmul.cubin`          | `cuasm_set` slot 30                   | Hand-edited Rust-origin cubin               |
| `run_oxide.sh`                     | this spike                            | Driver script (now obsolete; deps satisfied)|

## Source comparison

### CUDA C++ (phase1)

```c
extern "C" __global__ void vector_add(
    const float * __restrict__ input_a,
    const float * __restrict__ input_b,
    float * __restrict__ output_c,
    int num_elements
) {
    int element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index < num_elements)
        output_c[element_index] = input_a[element_index] + input_b[element_index];
}
```

### Rust (cuda-oxide vecadd)

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

Both express the same intent. Rust source is shorter and safer-looking
(no `__restrict__`, no manual bounds, slice types carry length).

## Pipeline comparison

```
nvcc:        .cu  ──nvcc──▶ PTX ──ptxas──▶ SASS ──cuasmR──▶ patched SASS
cuda-oxide:  .rs  ──rustc-codegen-cuda──▶ Pliron MIR ──▶ LLVM IR
                  ──llc-21──▶ PTX ──ptxas──▶ SASS ──cuasmR──▶ patched SASS
```

Both pipelines converge at PTX. Everything downstream (ptxas, cuobjdump,
cuasmR byte patches, control-code edits) operates on the cubin and is
**source-language-agnostic**. Confirmed by byte-identical roundtrip and
working FADD→FMUL hand-edit on Rust-origin cubin.

## SASS bloat root cause

Rust kernel emits ~2× SASS for vecadd. Source of bloat (per
`vecadd_oxide.sm_86.sass`):

| Bloat source                                    | nvcc | oxide | Cost          |
|-------------------------------------------------|------|-------|---------------|
| Independent slice bounds checks (a, b, c)       | 1    | 3     | 4 ISETP + 2 EXIT |
| `Option<&mut T>` discriminant tracking          | 0    | 1     | PRMT, LOP3.LUT, ISETP, branch (4 instr) |
| 64-bit address arithmetic everywhere            | n/a  | n/a   | IADD3 + IADD3.X pairs vs nvcc IMAD.WIDE  |
| Unfused register copies                         | 0    | 2     | IMAD.MOV.U32 R3, R9 etc.                 |

For vecadd the kernel is DRAM-bandwidth-bound, so the extra ALU is free.
For compute-bound kernels (GEMM, attention) this 2× ALU overhead would
land on the critical path and matter.

PTX shape diff (oxide):
```ptx
.entry vecadd(
    .param .u64 .ptr .align 1 vecadd_param_0,   // a.ptr
    .param .u64                vecadd_param_1,  // a.len  ← nvcc has nothing here
    .param .u64 .ptr .align 1 vecadd_param_2,   // b.ptr
    .param .u64                vecadd_param_3,  // b.len  ← nor here
    .param .u64 .ptr .align 1 vecadd_param_4,   // c.ptr
    .param .u64                vecadd_param_5   // c.len  ← only num_elements in nvcc
)
```

nvcc passes one `int num_elements`. cuda-oxide passes three independent
`u64` lengths because slices are independent values at the type level.
ptxas + LLVM cannot prove `a.len == b.len == c.len`, so all three bounds
checks survive into SASS. **Workaround for production**: take a single
`(ptr, len)` for inputs and pre-validate equal lengths host-side, or use
fixed-size arrays.

## Toolchain footprint

Installed under `~/cuda-oxide-deps/` (NOT on D: — see disk constraint).
Symlinked into `experiments/rust-experiments/llvm21` and
`experiments/rust-experiments/cuda-oxide` (not committed to repo; rebuild
locally per `run_oxide.sh`).

| Component              | Source                             | Disk    |
|------------------------|------------------------------------|---------|
| LLVM 21.1.8 (clang+llc)| `LLVM-21.1.8-Linux-X64.tar.xz`     | 11 GB   |
| Rust nightly-2026-04-03| `rustup` (rust-toolchain.toml)     | ~2 GB   |
| cuda-oxide checkout    | `git clone NVlabs/cuda-oxide`      | ~30 MB  |
| cuda-oxide build target| `cargo build` artifacts            | ~3 GB   |
| **Total**              |                                    | **~16 GB** |

Required env (set by `run_oxide.sh`):
```bash
export PATH="$HOME/cuda-oxide-deps/llvm21/bin:$PATH"
export LIBCLANG_PATH="$HOME/cuda-oxide-deps/llvm21/lib"
export CUDA_OXIDE_LLC="$HOME/cuda-oxide-deps/llvm21/bin/llc"
```

## Compatibility surprises (positive)

1. **CUDA 13.2 works** despite cuda-oxide README claiming "12.x+". libNVVM
   2.0 and nvJitLink 13.2 both detected by `cargo oxide doctor`.
2. **ptxas accepts oxide PTX** even though oxide emits `.target sm_80` and
   we ask for sm_86 — ptxas auto-promotes virtual targets.
3. **cuobjdump + cuasmR are blind to provenance**. Byte-identical roundtrip
   on Rust-origin cubin, identical hand-edit pattern (FADD → FMUL with
   same opcode bit flip + same control-bit set).

## Compatibility risks (negative)

1. **Cold build = ~110 s** (rustc-dev component, codegen backend, all path
   deps). Iteration cost is real for first-time-per-machine. Incremental
   is fine (~1 s for kernel-only edits).
2. **sm_80 target hardcoded somewhere** in oxide's NVPTX path — not a
   blocker (ptxas re-targets), but means oxide may not exploit sm_86-only
   features (none material for this kernel; matters for sm_90+ TMA/WGMMA).
3. **Toolchain weight**: 16 GB resident, plus an external LLVM tarball
   that must be kept in sync with `cargo oxide doctor`'s expectations.
4. **Rust safety abstractions cost real SASS instructions.** For
   compute-bound kernels you'd write `unsafe`-style or use
   `cuda-device::DisjointSlice::get_unchecked_mut` (if exposed) to drop
   the bounds checks. Defeats the safety pitch.
5. **Multiple `inline asm exit;` blocks** generated for slice index panics.
   These are dead-code basic blocks in the cubin (`$L__BB0_8/9/10` in
   oxide PTX). Visible bloat, no runtime cost, but uglier disassembly.

## Verdict (revised after live measurement)

Spike confirms everything from the structural analysis and adds two
empirical findings:

**Pipeline converges cleanly at PTX.** cuasmR + nvdisasm + cuobjdump
all work on Rust-origin cubins with zero changes. Hand-edit research is
fully portable.

**CUDA 13.2 + sm_86 is supported in practice** despite cuda-oxide
targeting Hopper/Blackwell as the headline. Ampere is not abandoned.

**2× SASS bloat for safety-typed kernels.** For DRAM-bound kernels
this is invisible. For HMMA/IMMA-throughput kernels (the entire point of
this project's later phases) this is a structural cost that grows with
kernel complexity, not amortized away.

**16 GB toolchain footprint** to support what is currently a
single-language alternative for the kernel front-end.

### Recommended integration (if any)

Don't wholesale-replace nvcc. Instead, treat `experiments/` as a permanent
sandbox for **specific experiments**:

1. **Compare ptxas output quality**: when hand-tuning a phase2/3 kernel,
   compile it both ways and diff PTX/SASS. Sometimes oxide's MIR-derived
   PTX surfaces optimization opportunities nvcc misses (and vice versa).
2. **Generic kernel templating**: oxide's monomorphization-of-closures
   feature could replace ad-hoc `#define`-style template macros in C++
   for sweeping kernel parameters. Worth trying on a softmax/layernorm
   sweep.
3. **Track sm_100a / Blackwell support**: when the next GPU lands, oxide's
   tcgen05 / WGMMA support is more accessible than raw PTX intrinsics in
   nvcc inline assembly. Stake a claim now on the workflow.

For everything else — phase 1–5 SASS hand-edit research — **keep using
nvcc**. The 2× SASS instruction overhead is the dealbreaker for the
project's core thesis.
