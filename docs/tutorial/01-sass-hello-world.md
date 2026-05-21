# Chapter 01 — SASS Hello World

> The first chapter. Read this if you have never modified GPU machine code
> directly. After this chapter the rest of the series makes sense.

The premise of this entire project is that you can read and modify the
actual machine code that runs on a GPU. Not PTX (the intermediate
assembly), not nvcc-compiled binary you have to trust. The real SASS
instructions, decoded, edited, reassembled, and run.

This chapter walks through the smallest possible such modification:
take a 5-line CUDA vector-add kernel, find the `FADD` instruction in its
SASS, change it to `FMUL`, reassemble, and watch the GPU multiply
instead of add.

If this works on your machine, every other technique in this series is
within reach. If it does not, the toolchain needs fixing before any
optimization work is meaningful.

## The kernel

```cpp
extern "C" __global__ void vector_add(
    const float *input_a, const float *input_b,
    float *output_c, int num_elements
) {
    int element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index < num_elements)
        output_c[element_index] = input_a[element_index] + input_b[element_index];
}
```

Five lines. One thread per output element. The thread loads its A and B,
adds them, stores the result. Branchless modulo bounds check.

What does the GPU actually execute?

## The toolchain

Three tools do the work:

- **`nvcc --cubin`** — compile CUDA C++ to a cubin (CUDA binary).
- **`cuobjdump -sass`** — disassemble cubin to human-readable SASS.
- **`CuAssembler`** — disassemble cubin to *editable* `.cuasm` text, hand-edit, reassemble back to cubin.

`cuobjdump` is read-only. CuAssembler is the round-trippable one. The
difference matters: `cuobjdump` shows you SASS but loses information
(some control codes are not displayed). CuAssembler preserves enough to
reassemble identically. If you only need to *read* SASS, use `cuobjdump`.
If you need to *modify and run*, use CuAssembler.

The build script `scripts/build.R` wraps CuAssembler invocations:

```bash
Rscript scripts/build.R compile  vector_add.cu               # .cu → .cubin
Rscript scripts/build.R disasm   vector_add.sm_86.cubin       # .cubin → .cuasm (editable)
Rscript scripts/build.R assemble vector_add.sm_86.cuasm       # .cuasm → .cubin (after edits)
Rscript scripts/build.R roundtrip vector_add.cu               # all three, verifies bit-identical
```

The roundtrip step is the toolchain smoke test. If it fails, your
CUDA-CuAssembler version pairing is incompatible — fix that before
anything else. With CUDA 13.2 and the bundled `tools/CuAssembler/`,
roundtrip should pass on a fresh checkout.

## Step 1 — Compile and disassemble

```bash
cd phase1
nvcc --cubin -arch=sm_86 -O1 -o vector_add.sm_86.cubin vector_add.cu
cuobjdump -sass vector_add.sm_86.cubin
```

The output is approximately (registers vary, addresses are illustrative):

```
/*0000*/ MOV       R1, c[0x0][0x28] ;
/*0010*/ S2R       R3, SR_TID.X ;
/*0020*/ S2R       R4, SR_CTAID.X ;
/*0030*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x0] ;
/*0040*/ IMAD      R4, R4, R1, R3 ;
/*0050*/ ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x160], PT ;
/*0060*/ @P0 EXIT ;
/*0070*/ HFMA2.MMA R3, -RZ, RZ, 0, 0 ;
/*0080*/ IMAD.WIDE R2, R4, 0x4, c[0x0][0x168] ;
/*0090*/ LDG.E     R2, [R2] ;
/*00a0*/ IMAD.WIDE R6, R4, 0x4, c[0x0][0x170] ;
/*00b0*/ LDG.E     R6, [R6] ;
/*00c0*/ FADD      R0, R2, R6 ;       <-- THE OPERATION
/*00d0*/ IMAD.WIDE R4, R4, 0x4, c[0x0][0x178] ;
/*00e0*/ STG.E     [R4], R0 ;
/*00f0*/ EXIT ;
```

Reading top-down:

- `MOV R1, c[0x0][0x28]` — load the stack frame pointer (boilerplate)
- `S2R R3, SR_TID.X` — read special register `threadIdx.x` into R3
- `S2R R4, SR_CTAID.X` — read `blockIdx.x` into R4
- `IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x0]` — load `blockDim.x` from constant memory bank 0 offset 0x0
- `IMAD R4, R4, R1, R3` — `R4 = blockIdx.x * blockDim.x + threadIdx.x` (this is `element_index`)
- `ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x160], PT` — set predicate `P0 = (R4 >= num_elements)`
- `@P0 EXIT` — if predicate is true (out of bounds), exit the thread
- `IMAD.WIDE R2, R4, 0x4, c[0x0][0x168]` — pointer arithmetic: `R2:R3 = &input_a[R4]` (×4 because `sizeof(float)`)
- `LDG.E R2, [R2]` — load 32-bit float at `[R2:R3]` into R2 (overwrites the address)
- `IMAD.WIDE R6, R4, 0x4, c[0x0][0x170]` — `R6:R7 = &input_b[R4]`
- `LDG.E R6, [R6]` — load `input_b` value into R6
- `FADD R0, R2, R6` — `R0 = R2 + R6` — **the operation we will modify**
- `IMAD.WIDE R4, R4, 0x4, c[0x0][0x178]` — `R4:R5 = &output_c[element_index]`
- `STG.E [R4], R0` — store R0 to global memory at `[R4]`
- `EXIT` — thread done

That is the entire kernel: 14 SASS instructions, one of which is the
arithmetic. The rest is index calculation and memory access.

## A note on constant memory

Kernel arguments do not arrive in registers. The CUDA driver loads them
into the GPU's constant memory before the kernel launches. Bank 0 is
reserved for kernel arguments and built-ins.

| Offset | Content |
|---|---|
| `c[0x0][0x0]`   | `blockDim.x` |
| `c[0x0][0x160]` | `num_elements` (4th arg) |
| `c[0x0][0x168]` | `input_a` pointer (1st arg) |
| `c[0x0][0x170]` | `input_b` pointer (2nd arg) |
| `c[0x0][0x178]` | `output_c` pointer (3rd arg) |

Exact offsets depend on the compiler version. Read your own disassembly
to confirm before assuming. The sizes are predictable: `int` = 4 bytes,
pointer = 8 bytes, with alignment padding.

## Step 2 — Disassemble for editing

```bash
Rscript scripts/build.R disasm kernels/tutorial/vector_add.sm_86.cubin
# produces kernels/tutorial/vector_add.sm_86.cuasm
```

The `.cuasm` file is similar to the `cuobjdump` output but includes
**control codes** in front of each instruction:

```
[B------:R-:W-:Y:S04]  FADD R0, R2, R6 ;
```

The control code says:

| field | meaning |
|---|---|
| `B------` | Barrier wait mask (which of 6 scoreboards to wait on) |
| `R-` | Read dependency scoreboard slot (`-` = none) |
| `W-` | Write dependency scoreboard slot |
| `Y` | Yield hint (`Y` = give up warp slot if stalled) |
| `S04` | Stall count: 4 cycles before issuing this instruction |

For a simple opcode swap you do not need to touch the control code.
Stall count tuning is for performance optimization (chapter 03 covers
the IMMA S04→S02 case), not correctness.

## Step 3 — The modification

Open `vector_add.sm_86.cuasm` in any text editor. Find:

```
[B------:R-:W-:Y:S04]  FADD R0, R2, R6 ;
```

Change `FADD` to `FMUL`:

```
[B------:R-:W-:Y:S04]  FMUL R0, R2, R6 ;
```

Save as `vector_add.sm_86.modified.cuasm`. That is the entire edit.

## Step 4 — Reassemble

```bash
Rscript scripts/build.R assemble kernels/tutorial/vector_add.sm_86.modified.cuasm
# produces kernels/tutorial/vector_add.sm_86.modified.reassembled.cubin
```

CuAssembler regenerates the cubin's SASS section. The encoded
instruction bytes for FMUL are different from FADD (different opcode in
the instruction word), but the surrounding metadata is unchanged.

## Step 5 — Run both

```bash
nvcc -arch=sm_86 -o host.exe kernels/tutorial/host.cu -lcuda

./host.exe kernels/tutorial/vector_add.sm_86.cubin                       # original — adds
./host.exe kernels/tutorial/vector_add.sm_86.modified.reassembled.cubin  # modified — multiplies
```

Expected output:

```
Original (FADD):
a[i]=1.0   b[i]=10.0   c[i]=11.0    OK   (1+10=11)
a[i]=2.0   b[i]=20.0   c[i]=22.0    OK
...
PASS: All 32 elements correct.

Modified (FMUL):
a[i]=1.0   b[i]=10.0   c[i]=10.0    OK   (1×10=10)
a[i]=2.0   b[i]=20.0   c[i]=40.0    OK   (2×20=40)
...
PASS: All 32 elements correct.
FMUL modification confirmed — GPU is multiplying, not adding!
```

The GPU just executed *your* hand-edited SASS. No nvcc magic, no
intermediate, no library wrapping. The bytes you wrote are the bytes
that ran.

## What this proves

Three things:

1. **The toolchain works**. nvcc → cubin → CuAssembler → editable text →
   cubin → execution is a complete loop. Every later chapter relies on
   this.
2. **You can read SASS**. The 14 instructions of vector_add are not
   intimidating. Larger kernels are just more of the same vocabulary —
   GEMM in chapter 02 is mostly LDG, LDS, FFMA, and bookkeeping; HMMA
   kernels add a few more opcodes.
3. **The hardware does what you tell it**. There is no compiler
   re-optimizing your hand edits, no JIT, no abstraction layer. SASS
   is the program. Modify SASS, the GPU executes the modification.

For the remaining chapters this last point matters most. When chapter 03
talks about hand-editing IMMA stall counts S04→S02 for +1.6%
performance, that edit is the same kind of operation you just did:
modify a control code in `.cuasm`, reassemble, run. Different opcode,
same workflow.

## When the toolchain breaks

The most common failure mode is a CUDA / CuAssembler version mismatch.
CuAssembler is a third-party project (`tools/CuAssembler/` in this
repo) that targets specific CUDA versions. If the `roundtrip` step
fails:

```bash
Rscript scripts/build.R roundtrip kernels/tutorial/vector_add.cu
# error: assembled cubin differs from original
```

The fix is usually one of:

- Update CuAssembler to the latest commit (it tracks new CUDA versions on a delay)
- Downgrade CUDA toolkit to a version CuAssembler supports
- Use a different sm_xx target (the one in this project, sm_86, is
  well-supported)

Without a working roundtrip, all further hand-edit work in this series
will fail in confusing ways. Do not skip the roundtrip check.

## How this chapter relates to the rest

Chapters 02-05 each apply this workflow to progressively more complex
kernels. The hand-edits change from "correctness demonstration" to
"performance tuning":

- Chapter 02 ends at HGEMM with ~32,000 GFLOPS — most of the win is in
  CUDA C++ (tile sizes, register blocking), but a few percentage points
  came from SASS-level analysis of which loads are interleaved with which
  HMMAs.
- Chapter 03 covers the IMMA S04→S02 hand-edit explicitly: same
  `[control_code]  OPCODE` modification pattern as you did to FADD,
  applied to control codes for performance instead of opcodes for
  correctness.
- Chapter 04 (cp.async pipelining) and 05 (Flash Attention) include
  examples where reading the SASS revealed *why* a kernel was slow —
  the C++ compiler was reloading Q from L2 every iteration despite the
  fragment being loop-invariant. The fix was a CUDA-level explicit
  cache, but the diagnosis was SASS-level.

The thread connecting all of these: **you cannot tune what you cannot
read**. The first step in any GPU optimization is `cuobjdump -sass`. The
optional second step (more often useful than expected) is hand-editing
the `.cuasm`.

## How to run it yourself

```bash
cd /mnt/d/dev/p/bare-metal

# Verify toolchain
Rscript scripts/verify_setup.R

# Compile
nvcc --cubin -arch=sm_86 -O1 -o kernels/tutorial/vector_add.sm_86.cubin kernels/tutorial/vector_add.cu

# Roundtrip check (must pass before hand-editing)
Rscript scripts/build.R roundtrip kernels/tutorial/vector_add.cu

# Disassemble for editing
Rscript scripts/build.R disasm kernels/tutorial/vector_add.sm_86.cubin

# Edit the .cuasm: FADD → FMUL on the line after the LDG of input_b
# (use any text editor)

# Reassemble
Rscript scripts/build.R assemble kernels/tutorial/vector_add.sm_86.modified.cuasm

# Compare
nvcc -arch=sm_86 -o kernels/tutorial/host.exe kernels/tutorial/host.cu -lcuda
./kernels/tutorial/host.exe kernels/tutorial/vector_add.sm_86.cubin                        # adds
./kernels/tutorial/host.exe kernels/tutorial/vector_add.sm_86.modified.reassembled.cubin   # multiplies
```

## Source files

- `kernels/tutorial/vector_add.cu` (the 5-line kernel)
- `kernels/tutorial/host.cu` (CPU driver, allocates buffers, launches kernel, checks output)
- `kernels/tutorial/README.md` (the deeper SASS-level walkthrough this chapter is based on)
- `tools/CuAssembler/` (third-party SASS assembler)
- `scripts/build.R` (compile / disasm / assemble / roundtrip wrapper)

## Cross-references

- Chapter 02 — GEMM from Scratch (next: SASS for matrix multiply)
- Chapter 03 — INT8 Tensor Cores (the IMMA S04 → S02 hand-edit)
- `docs/ampere_sass_reference.md` — the full instruction set reference
- `docs/control_codes.md` — control code field meanings in detail
