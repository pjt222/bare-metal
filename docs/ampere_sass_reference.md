# Ampere SASS Quick Reference (sm_86)

Quick reference for native GPU instructions on RTX 3070 Ti (GA104, Ampere).
These are reverse-engineered — NVIDIA does not publish official SASS docs.

## Integer Arithmetic

| Instruction | Operation | Notes |
|---|---|---|
| `IMAD Rd, Ra, Rb, Rc` | Rd = Ra * Rb + Rc | 32-bit integer multiply-add |
| `IMAD.WIDE Rd, Ra, imm, Rc` | Rd:Rd+1 = Ra * imm + Rc | 64-bit result — used for pointer arithmetic |
| `IMAD.MOV.U32 Rd, RZ, RZ, Ra` | Rd = Ra | Move (idiom: multiply by zero and add) |
| `IADD3 Rd, Ra, Rb, Rc` | Rd = Ra + Rb + Rc | 3-input integer add |
| `ISETP.cond.AND P, PT, Ra, Rb, PT` | P = (Ra cond Rb) | Integer set predicate |

Conditions for ISETP: `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` (with `.U32` for unsigned)

## Float Arithmetic (FP32)

| Instruction | Operation | Notes |
|---|---|---|
| `FADD Rd, Ra, Rb` | Rd = Ra + Rb | FP32 add |
| `FMUL Rd, Ra, Rb` | Rd = Ra * Rb | FP32 multiply |
| `FFMA Rd, Ra, Rb, Rc` | Rd = Ra * Rb + Rc | FP32 fused multiply-add — core of GEMM |
| `FMNMX Rd, Ra, Rb, P` | Rd = P ? min(Ra,Rb) : max(Ra,Rb) | FP32 min/max |
| `FSETP.cond.AND P, PT, Ra, Rb, PT` | P = (Ra cond Rb) | FP32 set predicate |
| `MUFU.op Rd, Ra` | Multi-function unit | See table below |

## Multi-Function Unit (MUFU)

Hardware approximations — ~1-4 ULP error, ~16 cycles latency.

| Variant | Operation | Use case |
|---|---|---|
| `MUFU.RCP Rd, Ra` | Rd = 1.0 / Ra | Softmax normalization, attention scale |
| `MUFU.RSQ Rd, Ra` | Rd = 1.0 / sqrt(Ra) | LayerNorm, GroupNorm |
| `MUFU.SQRT Rd, Ra` | Rd = sqrt(Ra) | Norms |
| `MUFU.EX2 Rd, Ra` | Rd = 2^Ra | exp(x) = 2^(x * log2(e)) |
| `MUFU.LG2 Rd, Ra` | Rd = log2(Ra) | log(x) = log2(x) / log2(e) |
| `MUFU.SIN Rd, Ra` | Rd = sin(Ra * pi) | Positional encoding |
| `MUFU.COS Rd, Ra` | Rd = cos(Ra * pi) | Positional encoding |
| `MUFU.TANH Rd, Ra` | Rd = tanh(Ra) | GELU activation (added Turing+) |

**Computing exp(x) via MUFU.EX2:**
```sass
FMUL Rd, Ra, 1.4426950408f   ; Ra * log2(e)
MUFU.EX2 Rd, Rd               ; 2^(Ra * log2(e)) = e^Ra
```

## Float16 / Half Precision

| Instruction | Operation | Notes |
|---|---|---|
| `HADD2 Rd, Ra, Rb` | FP16x2 add | Two FP16 ops packed in one instruction |
| `HMUL2 Rd, Ra, Rb` | FP16x2 multiply | |
| `HFMA2 Rd, Ra, Rb, Rc` | FP16x2 fused multiply-add | |

## Tensor Core (HMMA) — the performance critical instruction

Warp-wide matrix multiply. All 32 threads in a warp participate.

| Instruction | Shape | Precision |
|---|---|---|
| `HMMA.16816.F16 Rd, Ra, Rb, Rc` | 16x8x16 | FP16 in, FP16 out |
| `HMMA.16816.F32 Rd, Ra, Rb, Rc` | 16x8x16 | FP16 in, FP32 out |
| `HMMA.16816.BF16 Rd, Ra, Rb, Rc` | 16x8x16 | BF16 in, FP32 out |
| `IMMA.8816.S8 Rd, Ra, Rb, Rc` | 8x8x16 | INT8 in, INT32 out |

Each thread holds fragments of the matrices:
- A fragment: 8 FP16 values (rows of 16x16 sub-tile)
- B fragment: 8 FP16 values
- C/D accumulator: 4 FP32 values (or 8 FP16 values)

**Key**: These registers must be in the exact layout expected by hardware.
Reverse-engineer by compiling a `wmma` kernel and inspecting which registers
the compiler assigns to the `mma.sync` PTX instruction.

## Memory Instructions

### Global Memory

| Instruction | Width | Operation |
|---|---|---|
| `LDG.E Rd, [Ra]` | 32-bit | Load float from global |
| `LDG.E.64 Rd, [Ra]` | 64-bit | Load 2 floats (vector) |
| `LDG.E.128 Rd, [Ra]` | 128-bit | Load 4 floats (vector) — preferred for bandwidth |
| `STG.E [Ra], Rb` | 32-bit | Store float to global |
| `STG.E.128 [Ra], Rb` | 128-bit | Store 4 floats — preferred |
| `LDGSTS Rd, [Ra], [Rb]` | async | Async global→shared copy (Ampere) |

Use `.128` loads/stores when possible — 1 transaction vs 4.
Requires 16-byte aligned addresses.

### Shared Memory

| Instruction | Width | Operation |
|---|---|---|
| `LDS.U.32 Rd, [Ra]` | 32-bit | Load from shared |
| `LDS.U.128 Rd, [Ra]` | 128-bit | Load 4 floats from shared |
| `STS.32 [Ra], Rb` | 32-bit | Store to shared |
| `STS.128 [Ra], Rb` | 128-bit | Store 4 floats to shared |

**Bank conflicts**: Shared memory has 32 banks (4-byte each).
Threads in a warp accessing the same bank → serialized.
Fix with padding or swizzling the access pattern.

### Constant Memory

Kernel arguments live in bank 0: `c[0x0][offset]`

| Offset range | Content |
|---|---|
| `c[0x0][0x0]` – `c[0x0][0x7]` | blockDim.{x,y,z}, gridDim.{x,y,z} |
| `c[0x0][0x140]+` | kernel parameters (pointer, scalars) |

## Control Flow

| Instruction | Operation |
|---|---|
| `@Px INSTR` | Execute INSTR only if predicate Px is true |
| `@!Px INSTR` | Execute INSTR only if predicate Px is false |
| `EXIT` | End the thread |
| `BRA label` | Unconditional branch |
| `@Px BRA label` | Conditional branch |
| `BAR.SYNC N` | Block-level barrier — all threads wait at barrier N |
| `MEMBAR.GL` | Memory fence (global) |
| `MEMBAR.CTA` | Memory fence (CTA = cooperative thread array = block) |

## Warp-Level Primitives

| Instruction | Operation |
|---|---|
| `SHFL.BFLY Rd, Ra, mask, clamp` | XOR butterfly shuffle — used for reductions |
| `SHFL.IDX Rd, Ra, lane, clamp` | Broadcast from specific lane |
| `REDUX.ADD Rd, Ra` | Warp-wide reduction (Ampere+) |

**Warp reduction pattern using SHFL.BFLY** (sum 32 values):
```sass
SHFL.BFLY PT, R0, R0, 0x10, 0x1f   ; swap with lane ^ 16
FADD R0, R0, R0                      ; add
SHFL.BFLY PT, R0, R0, 0x8, 0x1f    ; swap with lane ^ 8
FADD R0, R0, R0
SHFL.BFLY PT, R0, R0, 0x4, 0x1f    ; swap with lane ^ 4
FADD R0, R0, R0
SHFL.BFLY PT, R0, R0, 0x2, 0x1f    ; swap with lane ^ 2
FADD R0, R0, R0
SHFL.BFLY PT, R0, R0, 0x1, 0x1f    ; swap with lane ^ 1
FADD R0, R0, R0
; R0 in lane 0 now holds the warp sum
```

## Special Registers (SR_*)

| Register | Value |
|---|---|
| `SR_TID.X` | threadIdx.x |
| `SR_TID.Y` | threadIdx.y |
| `SR_TID.Z` | threadIdx.z |
| `SR_CTAID.X` | blockIdx.x |
| `SR_CTAID.Y` | blockIdx.y |
| `SR_CTAID.Z` | blockIdx.z |
| `SR_LANEID` | Lane within warp (0–31) |
| `SR_WARPID` | Warp within SM |
| `SR_NTHREADS` | blockDim.x * blockDim.y * blockDim.z |
| `RZ` | Always zero (register zero) |
| `PT` | Always true predicate |

## Control Codes

Every instruction in `.cuasm` format has a control code:
```
[B------:R-:W0:Y:S04]  FADD R0, R2, R6 ;
```

| Field | Range | Meaning |
|---|---|---|
| `B------` | 6 bits | Barrier stall mask: wait for these scoreboards |
| `R-` / `R0`–`R5` | 3 bits | Read dependency: assign to scoreboard slot |
| `W-` / `W0`–`W5` | 3 bits | Write dependency: assign to scoreboard slot |
| `Y` / `-` | 1 bit | Yield: allow warp scheduler to switch warps |
| `S00`–`S15` | 4 bits | Stall count: extra cycles to pause before executing |

**Common stall counts by instruction class:**

| Instruction type | Typical stall | Reason |
|---|---|---|
| `IMAD` | S01 | 1 cycle |
| `LDS` | S02 | 2 cycles |
| `FFMA` | S01–S04 | 1–4 cycles |
| `LDG` | S15+ or barrier | ~200 cycle global latency |
| `HMMA` | S00–S01 | Low latency if properly pipelined |
| `MUFU` | S04 | ~4 cycle |

Setting stalls too low → incorrect results. Too high → wasted cycles.
The compiler's stall choices are a starting point; reducing them (carefully) is how
you extract performance in hand-tuned SASS.

## RTX 3070 Ti (sm_86) Key Limits

| Resource | Limit per SM | Limit per Block |
|---|---|---|
| Threads | 1536 | 1024 |
| Warps | 48 | 32 |
| Registers | 65536 x 32-bit | — |
| Shared Memory | 128 KB | 99 KB |
| Thread Blocks | 16 | — |
| 32-bit Regs/Thread | 255 | — |

**Occupancy calculation**: More registers per thread → fewer warps can run simultaneously.
For GEMM, 128 registers/thread → 8 warps/SM (16% occupancy). That's fine —
latency hiding via software pipelining is more important than occupancy for compute-bound kernels.
