# cuasmR — R-native SASS hand-edit toolchain

Replaces the upstream Python [`cloudcores/CuAssembler`](https://github.com/cloudcores/CuAssembler) for the `.cubin → edit → .cubin` workflow on this project (#102).

**Why custom**: upstream is ~2 years stale and breaks on every CUDA major-version bump. CUDA 13.2 changed the cubin `e_flags` layout (#101) — `sm_version` moved from byte[0] to byte[1], `vsm_version` removed entirely. Patching upstream Python had high drift cost for a project whose policy is R-primary.

**Why R works**: nvdisasm produces a clean structured listing for every supported `sm_*`. cuasmR parses that listing, indexes instructions by their file offset in the `.text` section, and patches at the byte level. We never re-encode instruction text → bytes ourselves; new opcodes come from nvdisasm of a sibling `.cu`.

## Architecture

```
.cubin --[nvdisasm + ELF parse]--> cuasm object (R list)
                                        |
                                  cuasm_set / data.frame edits
                                        |
                                        v
                               cuasm_write (byte-patch)
                                        |
                                        v
                                     .cubin
```

The `cuasm` object holds the original raw bytes plus a parsed view:
- `$path`     -- source cubin path
- `$raw`      -- raw byte vector (~ thousands of KB)
- `$sections` -- ELF section header table (data.frame)
- `$kernels`  -- one row per `.text.<kernel>` section (kernel, offset, size)
- `$insns`    -- one row per 16-byte SASS instruction slot
- `$arch`     -- decoded sm/vsm/layout from `e_flags`

Each instruction row has `instr_hex` (the 64-bit SASS instruction word) and `ctrl_hex` (the 64-bit control word — stalls, yield, scoreboards, reuse bits). These are the only bytes we ever change.

## Install

The package lives at `R/cuasmR/` in this repo. From the repo root:

```bash
Rscript scripts/install_cuasmR.R
```

This installs into the active renv library. After editing `R/cuasmR/R/*.R`, re-run the same script.

## Read / inspect / round-trip

```r
library(cuasmR)

obj <- cuasm_read("phase2/hgemm/hgemm_16warp.sm_86.cubin")

cuasm_kernels(obj)           # data.frame of .text.<kernel> sections
cuasm_insns(obj, "hgemm_16warp")[1:10, ]   # first 10 SASS instructions

cuasm_save_cuasm(obj, "/tmp/hgemm.cuasm")  # human-readable text dump

cuasm_roundtrip_check("phase2/hgemm/hgemm_16warp.sm_86.cubin")
# [1] TRUE     (read -> write produces byte-identical cubin)
```

## Hand-edit workflow

The supported edit operations are:

1. **Patch the control code** of a slot (stall count, yield bit, read/write barriers, register reuse).
2. **Replace the instruction word** of a slot with another encoding obtained from a sibling cubin.

Both are byte-level: write a new 64-bit hex string into `instr_hex` or `ctrl_hex`, call `cuasm_write`. Everything outside the `.text.<kernel>` sections is preserved verbatim.

### Example: FADD → FMUL on Phase 1 vector_add

```r
library(cuasmR)

# 1. Read original cubin
obj <- cuasm_read("kernels/tutorial/vector_add.sm_86.cubin")

# 2. Find the FADD slot
fadd <- subset(cuasm_insns(obj, "vector_add"), grepl("^FADD", text))
print(fadd)
#    kernel  slot address              text         instr_hex            ctrl_hex
# 14 vector_add 13   00d0  FADD R9, R4, R3 0x...4097221   0x004fca0000000000

# 3. Compile a sibling kernel that does multiplication, read its FMUL encoding
#    (nvcc --cubin -arch=sm_86 -O1 vector_mul.cu; cuasm_read; ...)
#    Result: FMUL R9, R4, R3 has instr_hex = 0x0000000304097220
#                                 ctrl_hex = 0x004fca0000400000

# 4. Patch and write
obj <- cuasm_set(obj, kernel = "vector_add", slot = 13,
                 instr_hex = "0x0000000304097220",
                 ctrl_hex  = "0x004fca0000400000")
cuasm_write(obj, "kernels/tutorial/vector_add.fmul.cubin")

# 5. Verify by re-disassembling
obj2 <- cuasm_read("kernels/tutorial/vector_add.fmul.cubin")
subset(cuasm_insns(obj2, "vector_add"), slot == 13)
#  kernel  slot address              text         instr_hex            ctrl_hex
#  vector_add 13   00d0  FMUL R9, R4, R3 0x...4097220   0x004fca0000400000
```

## What cuasmR does NOT do

- **Encode instruction text → bytes.** Upstream's `CuInsAssembler` does this via per-SM regex tables (~200 KB of pickled state per architecture). We don't need this for control-code edits or family-swap operations. If you want to insert a brand-new instruction with novel operands, write it as a sibling `.cu`, compile, and copy the encoding.
- **Restructure ELF sections.** `cuasm_write` only changes bytes inside existing `.text.<kernel>` slots. It cannot insert/remove instructions, change section sizes, or reorder symbols.
- **Validate semantics.** A patched control code that violates dependency rules will simply produce wrong results at runtime. Pair every hand-edit with a bench + correctness check (the `kernels/_common/check.h` `check_fp32` pattern).

## Round-trip guarantees

`cuasm_roundtrip_check(path)` verifies that reading and writing without edits produces a byte-identical cubin. We test this on:

| target                                                | kernels | insns | layout | roundtrip |
|---|---:|---:|---|:---:|
| `kernels/tutorial/vector_add.sm_86.cubin`                       | 1 |   32  | cuda13 | ✓ |
| `phase2/hgemm/hgemm_16warp.sm_86.cubin`               | 1 |  544  | cuda12 | ✓ |
| `phase2/hgemm/hgemm_16warp_splitk.sm_86.cubin`        | 1 | 1504  | cuda13 | ✓ |
| `phase3/.../flash_attn_br16_v2_pipeline_pad2.cubin`   | 1 | 1256  | cuda13 | ✓ |
| `phase4/conv2d/conv2d_implicit_gemm_v2.sm_86.cubin`   | 1 | 1024  | cuda13 | ✓ |
| `phase4/resblock/resblock.sm_86.cubin`                | 2 | 1232  | cuda12 | ✓ |

Mix of CUDA 12.x and 13.x cubins, single- and multi-kernel cubins.

## Tests

```bash
Rscript -e 'library(testthat); library(cuasmR); test_dir("R/cuasmR/tests/testthat")'
```

Three checks:
1. Byte-identical roundtrip on `kernels/tutorial/vector_add`.
2. `e_flags` decoder accepts both legacy and new layouts.
3. `cuasm_set` patches at most 8 bytes (one 64-bit word) per call.

## CLI

`scripts/build.R` uses cuasmR for the `disasm` and `roundtrip` subcommands:

```bash
Rscript scripts/build.R compile   kernels/tutorial/vector_add.cu
Rscript scripts/build.R disasm    kernels/tutorial/vector_add.sm_86.cubin
Rscript scripts/build.R roundtrip kernels/tutorial/vector_add.cu
```

The `assemble` subcommand is a stub — point users at the `cuasm_read → cuasm_set → cuasm_write` R script pattern shown above. Text-to-bytes assembly is intentionally out of scope.

## Comparison to upstream CuAssembler

|                      | upstream Python | cuasmR (this repo) |
|---|---|---|
| Decode `.cubin`              | re-implements ELF + opcode regex tables | nvdisasm + minimal ELF reader |
| Encode `.cuasm`              | reverse of decode (per-arch repos) | not supported (use sibling `.cu`) |
| CUDA 12.x compatible         | yes                            | yes |
| CUDA 13.x compatible         | broken (#101)                  | yes |
| Roundtrip byte-identical     | mostly (depends on metadata)   | yes (preserves all non-text bytes) |
| Maintenance surface          | ~5000 lines Python              | ~400 lines R |
| Drift on CUDA major bump     | breaks                         | tracks nvdisasm output (stable) |

## See also

- Issue #101 — original CUDA 13.2 break report
- Issue #102 — this replacement
- `R/cuasmR/R/elf.R` — `e_flags` layout decoder (handles both formats)
- `R/cuasmR/R/disasm.R` — nvdisasm output parser
- `R/cuasmR/R/api.R` — exported functions
