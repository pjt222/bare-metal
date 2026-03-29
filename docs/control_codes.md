# SASS Control Codes — Ampere (sm_86)

Control codes are the scheduling metadata attached to every SASS instruction.
They tell the GPU's warp scheduler about data dependencies and when it's safe
to execute each instruction.

## Format

In CuAssembler `.cuasm` files, each instruction looks like:
```
[B------:R-:W0:Y:S04]  FFMA R8, R4, R6, R8 ;
```

The `[...]` block is the control code. It has 5 fields:

```
[BBBBBB:Rx:Wx:Y:Snn]
 ^^^^^^ ^^ ^^ ^ ^^^
 |      |  |  | +-- Stall count (S00..S15)
 |      |  |  +---- Yield hint (Y or -)
 |      |  +------- Write scoreboard (W0..W5 or W-)
 |      +---------- Read scoreboard (R0..R5 or R-)
 +----------------- Barrier mask (6 bits, B or -)
```

## Stall Count (S00..S15)

The number of cycles the warp scheduler will stall before issuing this instruction.

- `S00` — no stall (issue immediately)
- `S01` — stall 1 cycle
- `S15` — stall 15 cycles (maximum)

Setting stall too low causes **data hazards** → wrong results or GPU hangs.
Setting stall too high wastes cycles.

**How to find the right value**: stall = instruction_latency - pipeline_depth.
For most instructions, ptxas sets conservative stalls. Reducing them is safe
if you can fill the gap with independent instructions (latency hiding).

Typical instruction latencies on Ampere:

| Instruction | Latency | ptxas stall |
|---|---|---|
| `IMAD` | ~6 cycles | S06 |
| `FADD`, `FMUL` | ~4 cycles | S04 |
| `FFMA` | ~4 cycles | S04 |
| `MUFU.*` | ~16 cycles | S04 (with scoreboard) |
| `LDS` | ~20 cycles | S02 (with scoreboard) |
| `LDG` | ~200 cycles | S00 + barrier |
| `HMMA` | ~32 cycles | S00 + barrier (pipelined) |
| `BAR.SYNC` | variable | — |

## Write Scoreboard (W0..W5)

Assigns this instruction's output to a dependency tracking slot.
Later instructions that read this value reference the same slot via the
barrier mask.

- `W-` — no scoreboard tracking (immediate, e.g. `MOV`)
- `W0`..`W5` — 6 available scoreboard slots

Long-latency instructions (global loads, MUFU) use scoreboards so the
scheduler knows when the result is ready without stalling immediately.

## Barrier Mask (BBBBBB)

6-bit mask. Each bit corresponds to a scoreboard slot (0..5).
If bit N is set (`B` instead of `-`), this instruction stalls until
scoreboard slot N signals completion.

Example:
```
; Global load — assigns to scoreboard 0
[B------:R-:W0:-:S00]  LDG.E R4, [R2] ;

; ... (issue other independent instructions here to hide latency)

; Use the loaded value — wait for scoreboard 0
[B0-----:R-:W-:Y:S04]  FADD R8, R4, R6 ;
```

The `B0-----` means "wait for scoreboard slot 0 before executing."

## Read Scoreboard (R0..R5)

Similar to write scoreboard but for read-after-write hazards on shared memory.
Less commonly used in hand-written kernels.

## Yield Hint (Y or -)

`Y` — suggest to the warp scheduler that it can switch to another warp.
`-` — hint to keep executing this warp (reduces context-switch overhead).

Use `Y` at natural stall points (after issuing a long-latency load).
The scheduler uses this as a hint, not a hard requirement.

## Example: Optimizing an FFMA Loop

**Before optimization** (compiler output — conservative stalls):
```sass
[B------:R-:W0:-:S00]  LDG.E.128 R4, [R2] ;    // load 4 floats from A tile
[B------:R-:W1:-:S00]  LDG.E.128 R8, [R6] ;    // load 4 floats from B tile
[B0-----:R-:W-:Y:S15]  FFMA R12, R4, R8, R12 ; // wait 15 cycles for R4
[B------:R-:W-:-:S04]  FFMA R13, R5, R9, R13 ;
[B------:R-:W-:-:S04]  FFMA R14, R6, R10, R14 ;
[B------:R-:W-:-:S04]  FFMA R15, R7, R11, R15 ;
```

**After optimization** (software-pipelined — issue next load while computing):
```sass
; Issue load for NEXT tile early
[B------:R-:W0:-:S00]  LDG.E.128 R4, [R2] ;        // load tile N
[B------:R-:W1:-:S00]  LDG.E.128 R20, [R2+0x100] ; // load tile N+1 (next iteration)

; Compute on tile N-1 (already in registers from previous iter)
; These fill the latency gap of the tile N loads above
[B------:R-:W-:-:S04]  FFMA R12, R16, R18, R12 ;
[B------:R-:W-:-:S04]  FFMA R13, R17, R19, R13 ;

; Now use tile N (should be ready by now)
[B0-----:R-:W-:Y:S04]  FFMA R12, R4, R8, R12 ;     // B0: wait for tile N
```

This is **double-buffering** — you're always computing on one tile while
loading the next. It completely hides the global memory latency when done correctly.

## Debugging Control Code Issues

If your hand-modified kernel produces wrong results:
1. First try setting all stalls to `S15` — if it works, you have a hazard
2. Add `MEMBAR.GL` after global stores if results are inconsistent
3. Check scoreboard assignments: every long-latency op needs `W0..W5`
   and every consumer needs the matching `B` bit set

If the kernel hangs or crashes the driver:
1. You likely have an illegal instruction encoding
2. Restore the original control code from the unmodified `.cuasm`
3. Check CuAssembler's output for warnings during assembly
