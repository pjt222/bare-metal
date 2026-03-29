# Phase 1: Hello World — Vector Add in SASS

## Goal

Write the simplest possible GPU kernel, compile it to native SASS machine code,
understand every instruction, hand-modify it, and prove the modification runs.

This is the foundational workflow for everything else in this project.

## The Kernel

```c
extern "C" __global__ void vector_add(
    const float *input_a, const float *input_b,
    float *output_c, int num_elements
) {
    int element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index < num_elements)
        output_c[element_index] = input_a[element_index] + input_b[element_index];
}
```

Five lines of C. But what does the GPU actually execute?

## Step 1: Compile and Disassemble

```bash
# Compile to cubin (GPU binary)
nvcc --cubin -arch=sm_86 -O1 -o vector_add.sm_86.cubin vector_add.cu

# Disassemble to human-readable SASS
cuobjdump -sass vector_add.sm_86.cubin

# Or use the build script (also produces editable .cuasm)
python ../scripts/build.py all vector_add.cu
```

## Step 2: Understanding the SASS Output

The disassembly will look approximately like this (exact registers vary):

```
/*0000*/   MOV R1, c[0x0][0x28] ;              // load stack pointer
/*0010*/   S2R R3, SR_TID.X ;                  // R3 = threadIdx.x
/*0020*/   S2R R4, SR_CTAID.X ;                // R4 = blockIdx.x
/*0030*/   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x0] ; // R1 = blockDim.x (from constant memory)
/*0040*/   IMAD R4, R4, R1, R3 ;               // R4 = blockIdx.x * blockDim.x + threadIdx.x (element_index)
/*0050*/   ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x160], PT ; // P0 = (element_index >= num_elements)
/*0060*/   @P0 EXIT ;                           // if out-of-bounds: exit (predicated)
/*0070*/   HFMA2.MMA R3, -RZ, RZ, 0, 0 ;       // (compiler artifact)
/*0080*/   IMAD.WIDE R2, R4, 0x4, c[0x0][0x168] ; // R2:R3 = &input_a[element_index]
/*0090*/   LDG.E R2, [R2] ;                    // R2 = input_a[element_index] (global load)
/*00a0*/   IMAD.WIDE R6, R4, 0x4, c[0x0][0x170] ; // R6:R7 = &input_b[element_index]
/*00b0*/   LDG.E R6, [R6] ;                    // R6 = input_b[element_index] (global load)
/*00c0*/   FADD R0, R2, R6 ;                   // R0 = input_a[i] + input_b[i]  <-- THE OPERATION
/*00d0*/   IMAD.WIDE R4, R4, 0x4, c[0x0][0x178] ; // R4:R5 = &output_c[element_index]
/*00e0*/   STG.E [R4], R0 ;                    // output_c[element_index] = R0 (global store)
/*00f0*/   EXIT ;                              // return
```

### Instruction Reference

| Instruction | Meaning |
|---|---|
| `S2R Rx, SR_TID.X` | Load special register `threadIdx.x` into Rx |
| `S2R Rx, SR_CTAID.X` | Load special register `blockIdx.x` into Rx |
| `IMAD Rd, Ra, Rb, Rc` | Rd = Ra * Rb + Rc (integer multiply-add) |
| `IMAD.WIDE Rd, Ra, imm, Rc` | 64-bit: Rd:Rd+1 = Ra * imm + Rc (pointer arithmetic) |
| `ISETP.GE.AND P, PT, Ra, Rb, PT` | Predicate: P = (Ra >= Rb) |
| `@P0 EXIT` | Conditional exit if predicate P0 is true |
| `LDG.E Rd, [Ra]` | Load 32-bit float from global memory at address Ra |
| `FADD Rd, Ra, Rb` | Rd = Ra + Rb (float add) — THIS IS WHAT WE MODIFY |
| `STG.E [Ra], Rb` | Store 32-bit float Rb to global memory at address Ra |
| `EXIT` | End the thread |

### Constant Memory `c[0x0][...]`

Arguments passed to the kernel live in constant memory bank 0.
The offsets depend on the kernel signature:

| Offset | Content |
|---|---|
| `c[0x0][0x0]` | blockDim.x |
| `c[0x0][0x168]` | pointer to input_a |
| `c[0x0][0x170]` | pointer to input_b |
| `c[0x0][0x178]` | pointer to output_c |
| `c[0x0][0x160]` | num_elements |

*(Exact offsets depend on compiler version — read your own disassembly)*

### Control Codes

Each SASS instruction is preceded by a control code in the `.cuasm` format:
```
[B------:R-:W0:Y:S04]
```

| Field | Meaning |
|---|---|
| `B------` | Barrier wait mask (which of 6 scoreboards to wait on) |
| `R-` / `R0` | Read dependency scoreboard slot |
| `W-` / `W0` | Write dependency scoreboard slot |
| `Y` / `-` | Yield hint (give up warp slot) |
| `S04` | Stall count: how many cycles to stall before this instruction |

You generally don't need to change control codes for a simple modification.
Only touch them when tuning latency scheduling.

## Step 3: The Modification

Open `vector_add.sm_86.cuasm` in any text editor.

Find the line containing `FADD`:
```
[B------:R-:W-:Y:S04]  FADD R0, R2, R6 ;
```

Change `FADD` to `FMUL`:
```
[B------:R-:W-:Y:S04]  FMUL R0, R2, R6 ;
```

Save the file as `vector_add.sm_86.modified.cuasm`.

## Step 4: Reassemble

```bash
python ../scripts/build.py assemble vector_add.sm_86.modified.cuasm
# produces: vector_add.sm_86.modified.reassembled.cubin
```

## Step 5: Build and Run the Host Driver

```bash
# Build the host program
nvcc -arch=sm_86 -o host.exe host.cu -lcuda

# Run with original cubin (FADD — should add)
host.exe vector_add.sm_86.cubin

# Run with modified cubin (FMUL — should multiply)
host.exe vector_add.sm_86.modified.reassembled.cubin multiply
```

## Expected Results

**Original (FADD)**:
```
a[i]=1.0   b[i]=10.0   c[i]=11.0   expected=11.0   OK
a[i]=2.0   b[i]=20.0   c[i]=22.0   expected=22.0   OK
...
PASS: All 32 elements correct.
```

**Modified (FMUL)**:
```
a[i]=1.0   b[i]=10.0   c[i]=10.0   expected=10.0   OK   (1*10=10)
a[i]=2.0   b[i]=20.0   c[i]=40.0   expected=40.0   OK   (2*20=40)
...
PASS: All 32 elements correct.
FMUL modification confirmed — GPU is multiplying, not adding!
```

## Before You Hand-Edit: Run the Roundtrip Test

The roundtrip test verifies that CuAssembler can disassemble and reassemble
the cubin without changes, producing a working binary. **Do this first.**

```bash
python ../scripts/build.py roundtrip vector_add.cu
```

If this fails, CuAssembler has a compatibility issue with your CUDA version.
Check `setup.md` and ensure you're using CUDA 12.x.

## What's Next

Once you've confirmed FMUL works, you've proven the full workflow:
- Write CUDA → compile → SASS → hand-modify → reassemble → run

Phase 2 applies this to ML primitives: SGEMM, HGEMM, softmax, layernorm.
The key difference: those kernels are much more complex and the modifications
are about *performance*, not just correctness.
