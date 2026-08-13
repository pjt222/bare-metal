#!/usr/bin/env bash
# run_oxide.sh — drive cuda-oxide vecadd through Rust → PTX → ptxas → SASS → cuasmR roundtrip
#
# This script assumes the live install completed successfully (see README.md
# "Toolchain footprint" section). All deps live under ~/cuda-oxide-deps/
# OFF the D: drive (D: was at 99% capacity; everything went to /home).
#
# To rebuild on a fresh machine, run BOOTSTRAP first (commented block below),
# then this script.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS="$HOME/cuda-oxide-deps"
OXIDE="$DEPS/cuda-oxide"
LLVM="$DEPS/llvm21"

# ============================================================================
# BOOTSTRAP (run once per machine; ~15 GB on /home, ~10 min total)
# ============================================================================
# mkdir -p "$DEPS" && cd "$DEPS"
# # 1. LLVM 21 (clang + llc with NVPTX) — 1.9 GB download, 11 GB extracted
# curl -L -o llvm21.tar.xz \
#   https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz
# mkdir -p llvm21 && tar xJf llvm21.tar.xz -C llvm21 --strip-components=1
# rm llvm21.tar.xz
# # 2. cuda-oxide checkout
# git clone --depth 1 https://github.com/NVlabs/cuda-oxide.git
# # 3. nightly Rust auto-installed by rustup via cuda-oxide/rust-toolchain.toml
# (cd cuda-oxide && rustup show)
# ============================================================================

# env required by cuda-oxide
export PATH="$LLVM/bin:$PATH"
export LIBCLANG_PATH="$LLVM/lib"
export CUDA_OXIDE_LLC="$LLVM/bin/llc"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"  # WSL libcuda passthrough

# 1) preflight
( cd "$OXIDE" && cargo oxide doctor ) || {
  echo "doctor failed — fix prerequisites first"; exit 1;
}

# 2) compile + run vecadd (Rust → PTX → driver-side JIT → GPU)
( cd "$OXIDE" && cargo oxide run vecadd )

# 3) capture emitted PTX
PTX="$OXIDE/crates/rustc-codegen-cuda/examples/vecadd/vecadd.ptx"
[ -f "$PTX" ] || { echo "vecadd.ptx not found at $PTX"; exit 1; }
cp "$PTX" "$SCRIPT_DIR/vecadd_oxide.ptx"

# 4) PTX → SASS via ptxas (matches kernels/tutorial/2 pipeline)
ptxas -arch=sm_86 -O2 "$SCRIPT_DIR/vecadd_oxide.ptx" \
    -o "$SCRIPT_DIR/vecadd_oxide.sm_86.cubin"

# 5) disassemble both for diff
cuobjdump -sass "$SCRIPT_DIR/vecadd_oxide.sm_86.cubin" \
    > "$SCRIPT_DIR/vecadd_oxide.sm_86.sass"
cuobjdump -sass "$REPO_ROOT/kernels/tutorial/vector_add.sm_86.cubin" \
    > "$SCRIPT_DIR/vecadd_nvcc.sm_86.sass"

# 6) cuasmR roundtrip — must run from REPO_ROOT for renv .Rprofile to load
( cd "$REPO_ROOT" && Rscript -e '
    library(cuasmR)
    obj <- cuasm_read("experiments/rust-experiments/vecadd_oxide.sm_86.cubin")
    cuasm_write(obj, "experiments/rust-experiments/vecadd_oxide.roundtrip.cubin")
    cat("kernels:", paste(obj$kernels$kernel, collapse=", "), "\n")
    cat("instructions:", nrow(obj$insns), "\n")
  ' )

# 7) byte-identity check (proves cuasmR survives Rust-origin cubins)
if cmp -s "$SCRIPT_DIR/vecadd_oxide.sm_86.cubin" \
          "$SCRIPT_DIR/vecadd_oxide.roundtrip.cubin"; then
  echo "✓ cuasmR roundtrip byte-identical"
else
  echo "✗ roundtrip differs — investigate cuasmR coverage of oxide-emitted sections"
  exit 1
fi

# 8) FADD → FMUL hand-edit on Rust-origin cubin
( cd "$REPO_ROOT" && Rscript -e '
    library(cuasmR)
    obj <- cuasm_read("experiments/rust-experiments/vecadd_oxide.sm_86.cubin")
    fadd <- subset(obj$insns, grepl("FADD", text))
    if (nrow(fadd) != 1) stop("expected exactly 1 FADD")
    # delta from phase1: opcode last digit 1->0, ctrl bit 22 (0x400000) set.
    #
    # The control word is 64-bit, so it must NOT go through strtoi(): that
    # returns int32, and the real word 0x004fe20000000000 overflowed to NA.
    # sprintf("%016x", NA) then yielded the literal "0x              NA",
    # which passes every length check downstream and is written out as
    # EIGHT ZERO BYTES -- zeroing the control word instead of setting one
    # bit in it. The committed vecadd_oxide.fmul.cubin carries the correct
    # 0x004fe20000400000, so it predates this bug; regenerating with the
    # old line would have silently corrupted it (#198).
    new_instr <- sub("1$", "0", fadd$instr_hex[1])
    new_ctrl  <- hex64_bit_set(fadd$ctrl_hex[1], 22L)
    obj <- cuasm_set(obj, kernel = fadd$kernel[1], slot = fadd$slot[1],
                     instr_hex = new_instr, ctrl_hex = new_ctrl)
    cuasm_write(obj, "experiments/rust-experiments/vecadd_oxide.fmul.cubin")
    cat("patched FADD slot", fadd$slot[1], "->", new_instr, "/", new_ctrl, "\n")
  ' )

# 9) verify the patched cubin disassembles as FMUL
echo "--- patched cubin disassembly ---"
cuobjdump -sass "$SCRIPT_DIR/vecadd_oxide.fmul.cubin" | grep -A1 -E 'FMUL|FADD' | head -4

# 9b) the patch must differ from the source in EXACTLY the two intended bytes
#     (#198). Step 7 only guards the roundtrip file, so a corrupt
#     regeneration of the .fmul file passed unnoticed -- which is precisely
#     how the strtoi overflow above could have overwritten a committed,
#     known-good artifact with one whose control word was zeroed.
#
#     Expected: the opcode byte and the control bit. `cmp -l` prints one
#     line per differing byte, so the count IS the assertion.
n_diff=$(cmp -l "$SCRIPT_DIR/vecadd_oxide.sm_86.cubin" \
                "$SCRIPT_DIR/vecadd_oxide.fmul.cubin" 2>/dev/null | wc -l)
if [ "$n_diff" -eq 2 ]; then
  echo "✓ patched cubin differs from source in exactly 2 bytes (opcode + ctrl bit)"
else
  echo "✗ patched cubin differs in $n_diff bytes, expected 2"
  echo "  a zeroed control word shows up here as 8 extra differing bytes."
  cmp -l "$SCRIPT_DIR/vecadd_oxide.sm_86.cubin" \
         "$SCRIPT_DIR/vecadd_oxide.fmul.cubin" | head -12
  exit 1
fi

echo
echo "Done. Compare:"
echo "  diff -u $SCRIPT_DIR/vecadd_nvcc.sm_86.sass $SCRIPT_DIR/vecadd_oxide.sm_86.sass"
echo "Run patched kernel (proves edit takes effect):"
echo "  - edit $OXIDE/crates/rustc-codegen-cuda/examples/vecadd/src/main.rs"
echo "    change load_module_from_file(\"vecadd.ptx\") to (\"vecadd.fmul.cubin\")"
echo "  - cp $SCRIPT_DIR/vecadd_oxide.fmul.cubin \\"
echo "       $OXIDE/crates/rustc-codegen-cuda/examples/vecadd/vecadd.fmul.cubin"
echo "  - cd $OXIDE && cargo oxide run vecadd"
echo "    expected: outputs 0,2,8,18,32 (multiplication, not addition)"
