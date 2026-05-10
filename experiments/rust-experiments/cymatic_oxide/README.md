# cymatic_oxide — gather_sum kernel: nvcc vs cuda-oxide

Second cuda-oxide spike, this time on a **non-trivial gather kernel** (the
kernels/memory_layout/cymatic memory-traffic benchmark) rather than vecadd. Results
invert the vecadd story in one dimension and confirm it in another.

## TL;DR

| Backend | SASS instr | Inner-loop LDG/FADD | row_GB/s | cym_GB/s | row/oxide row |
|---------|-----------:|--------------------:|---------:|---------:|--------------:|
| nvcc    |        120 |        21 (unrolled 4×) | 1858 | 1454 | 1.00× |
| oxide   |         80 |        3 (no unroll)    | 1490 | 965  | **0.80×** |

(`rowmajor_full` trace, n_inside=3.29M, 13.16 MB buffer — DRAM-bound.)

**oxide emits 33% fewer SASS instructions but runs 20–35% slower.**
Inverse of vecadd, where oxide had 2× the SASS at runtime parity.

The cymatic-vs-row layout pattern (Obs T) survives both backends:
`rowmajor_full` is row > cym (~0.65–0.78× cym/row) on both nvcc and oxide.
Layout effect is real and backend-agnostic.

## Why oxide is shorter on SASS but slower at runtime

**nvcc unrolls the inner gather loop 4×**, oxide does not.

nvcc inner body (snippet):
```
LDG.E.CONSTANT R16, [R2.64]
LDG.E.CONSTANT R16, [R16.64]
FADD R13, R13, R16
LDG.E.CONSTANT R16, [R4.64]
LDG.E.CONSTANT R18, [R6.64]
LDG.E.CONSTANT R16, [R16.64]
LDG.E.CONSTANT R18, [R18.64]
FADD R13, R13, R16
FADD R13, R13, R18
... (continues, 4 indep iterations issued together)
```

Per single inner-loop iteration nvcc gets:
- 4 independent `LDG idx[k..k+3]` issued together
- 4 dependent `LDG data[idx[i]]` issued back-to-back
- 4 serial `FADD`s into the accumulator

→ ~8 in-flight DRAM transactions per warp before any FADD waits.

oxide inner body (entire loop):
```
@%p3  bra panic
shl.b64 ... ; add ; ld.b32 ; add.rn.f32
add.s64 (k += stride)
bra inner_loop
```

→ 1 LDG, 1 dependent LDG, 1 FADD, branch back. **No memory pipelining.**

The 4× unroll in nvcc trades extra ALU instructions (~7 vs 3 FADD,
14 vs 2 IMAD.WIDE) for ~4× memory-level parallelism. On a DRAM-bound
gather this is the ratio that drives bandwidth. **The unroll heuristic,
not the front-end language, is the deciding factor.**

oxide also emits 3 panic blocks (3 ISETP+BRA+EXIT for the bounds checks
on `idx[k]`, `data[idx_value]`, and `out[0]`). These are dead-code on the
hot path (~9 cold instructions) — measured impact ≈ 0.

## Headline results

```
=== gather_sum runtime bench === (RTX 3070 Ti, sm_86, iters=200, runs=11)

trace                  n          row_ms     row_GB/s     cym_ms     cym_GB/s   row/cym
----------                  -------         ------     --------     ------     --------   -------

nvcc:
fa_seq1024_rowmajor    961728     0.0021       1806.1     0.0022       1788.9    0.99x
fa_seq1024_diagonal    961728     0.0022       1772.1     0.0021       1832.6    1.03x
fa_seq1024_zigzag      961728     0.0021       1823.7     0.0021       1801.8    0.99x
rowmajor_full         3289344     0.0071       1858.1     0.0091       1453.6    0.78x

oxide:
fa_seq1024_rowmajor    961728     0.0026       1470.4     0.0026       1484.9    1.01x
fa_seq1024_diagonal    961728     0.0026       1479.0     0.0026       1484.9    1.00x
fa_seq1024_zigzag      961728     0.0026       1484.9     0.0026       1482.4    1.00x
rowmajor_full         3289344     0.0088       1489.9     0.0136        965.4    0.65x
```

The 961k FA traces fit in L2 (3.7 MB) after warmup, so they all top out
at the same bandwidth ceiling; differences disappear into cache-effect
noise. Only `rowmajor_full` (13.16 MB > 4 MB L2) is DRAM-bound and shows
the real unroll effect.

## Files

| File                          | Origin                                  |
|-------------------------------|-----------------------------------------|
| `src/main.rs`                 | Rust port of kernels/memory_layout/cymatic gather_sum |
| `Cargo.toml`                  | Path-deps to cuda-oxide crates          |
| `gather_oxide.ptx`            | `cargo oxide run cymatic_gather`        |
| `gather_oxide.sm_86.cubin`    | `ptxas -arch=sm_86 -O2`                 |
| `gather_oxide.sm_86.sass`     | `cuobjdump -sass`                       |
| `gather_nvcc.cu`              | Source-identical extract from phase4   |
| `gather_nvcc.ptx`             | `nvcc -arch=sm_86 -O2 -ptx`             |
| `gather_nvcc.sm_86.cubin`     | `nvcc --cubin`                          |
| `gather_nvcc.sm_86.sass`      | `cuobjdump -sass`                       |
| `bench_gather_driver.cu`      | Driver-API harness (loads either cubin) |
| `bench_gather_driver`         | Built binary                            |
| `perm.bin`, `traces.bin`      | Symlinked / copied from kernels/memory_layout/cymatic  |

## How to reproduce

```bash
# 1. Build oxide kernel (drops gather_oxide.ptx)
cd ~/cuda-oxide-deps/cuda-oxide
export PATH="$HOME/cuda-oxide-deps/llvm21/bin:$PATH" \
       LIBCLANG_PATH="$HOME/cuda-oxide-deps/llvm21/lib" \
       CUDA_OXIDE_LLC="$HOME/cuda-oxide-deps/llvm21/bin/llc" \
       LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cargo oxide run cymatic_gather  # exits with MISMATCH (smoke test reads
                                # only thread-0's accumulator, matching
                                # the C++ kernel's atomicAdd guard); PTX
                                # still emitted before launch.

# 2. Compile both cubins
cd /mnt/d/dev/p/bare-metal/experiments/rust-experiments/cymatic_oxide
nvcc -arch=sm_86 -O2 --cubin gather_nvcc.cu -o gather_nvcc.sm_86.cubin
ptxas -arch=sm_86 -O2 gather_oxide.ptx       -o gather_oxide.sm_86.cubin

# 3. Build the driver harness once
nvcc -arch=sm_86 -O2 -std=c++17 -o bench_gather_driver \
     bench_gather_driver.cu -lcuda

# 4. Run on the same data files (kernels/memory_layout/cymatic/{perm,traces}.bin)
./bench_gather_driver gather_nvcc.sm_86.cubin  nvcc  perm.bin traces.bin
./bench_gather_driver gather_oxide.sm_86.cubin oxide perm.bin traces.bin
```

## Verdict

Updates the experiments/ verdict. The 2× SASS bloat conclusion from the vecadd
spike was a kernel-shape artifact:

- **vecadd**: pointer-bound, single-LDG body. Bounds-check bloat
  dominates SASS instruction count. Runtime parity (memory-bound either
  way).
- **cymatic gather_sum**: serialised `data[idx[k]]` chain. nvcc's loop
  unroller is the dominant factor. oxide emits *fewer* instructions but
  runs slower because each iteration has only one in-flight LDG.

So the real take-away from cuda-oxide on this Ampere project is not
"safety types add SASS"; it's **"oxide doesn't (yet) match nvcc's loop
optimization heuristics, even when LLVM is involved"**. Force-unrolling
the Rust source (`#[unroll(4)]`-equivalent loop hints, manual unroll, or
`unsafe` get_unchecked + explicit unroll) is the path to parity.

Cymatic layout effect (Obs T) is **independent of backend** — confirmed
on a second compiler. That's the more important phase4 finding.

For phase 1–5 SASS hand-edit research the conclusion stands: keep using
nvcc. cuda-oxide is now better characterised as a research surface for
**(a)** front-end ergonomics, **(b)** Hopper/Blackwell intrinsics on
sm_90+, and **(c)** comparing optimization heuristics across PTX
producers. Not a drop-in replacement.
