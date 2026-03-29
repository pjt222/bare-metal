"""
build.py — Automate the bare-metal SASS workflow:

    compile   → produce .cubin from .cu
    disasm    → produce .cuasm from .cubin  (human-editable)
    assemble  → produce .cubin from .cuasm  (after hand-editing)
    run       → launch modified cubin via CUDA Driver API test harness

Usage:
    python scripts/build.py compile  phase1/vector_add.cu
    python scripts/build.py disasm   phase1/vector_add.cubin
    python scripts/build.py assemble phase1/vector_add_modified.cuasm
    python scripts/build.py roundtrip phase1/vector_add.cu   # compile + disasm + reassemble + verify identical
    python scripts/build.py all      phase1/vector_add.cu    # full workflow
"""

import subprocess
import sys
import os
import shutil
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUASSEMBLER_PATH = os.path.join(REPO_ROOT, "tools", "CuAssembler")
SM_ARCH = "sm_86"

# WSL: CUDA tools live in /usr/local/cuda/bin — add to PATH if not already there
CUDA_BIN = "/usr/local/cuda/bin"
if os.path.isdir(CUDA_BIN) and CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = CUDA_BIN + ":" + os.environ.get("PATH", "")


def ensure_cuassembler():
    if CUASSEMBLER_PATH not in sys.path:
        sys.path.insert(0, CUASSEMBLER_PATH)
    try:
        from CuAsm.CubinFile import CubinFile  # noqa: F401
    except ImportError:
        print(f"ERROR: CuAssembler not found at {CUASSEMBLER_PATH}")
        print("Run: git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler")
        sys.exit(1)


def run(cmd, cwd=None):
    """Run a shell command, print it, stream output. Exit on failure."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def cmd_compile(source_path, output_path=None, extra_flags=""):
    """Compile .cu to .cubin targeting sm_86."""
    source_path = os.path.abspath(source_path)
    if output_path is None:
        output_path = source_path.replace(".cu", f".{SM_ARCH}.cubin")
    output_path = os.path.abspath(output_path)

    print(f"\n[compile] {source_path} -> {output_path}")
    run(f'nvcc --cubin -arch={SM_ARCH} -O2 {extra_flags} -o "{output_path}" "{source_path}"')
    print(f"  -> {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def cmd_disasm(cubin_path, output_path=None, annotate=True):
    """Disassemble .cubin to .cuasm using CuAssembler."""
    ensure_cuassembler()
    from CuAsm.CubinFile import CubinFile

    cubin_path = os.path.abspath(cubin_path)
    if output_path is None:
        output_path = cubin_path.replace(".cubin", ".cuasm")
    output_path = os.path.abspath(output_path)

    print(f"\n[disasm] {cubin_path} -> {output_path}")
    cubin_file = CubinFile(cubin_path)
    cubin_file.saveAsCuAsm(output_path)
    print(f"  -> {output_path} ({os.path.getsize(output_path)} bytes)")

    # Also produce raw nvdisasm output for reference
    raw_sass_path = cubin_path.replace(".cubin", ".sass")
    sass_result = subprocess.run(
        f'cuobjdump -sass "{cubin_path}"',
        shell=True, capture_output=True, text=True
    )
    if sass_result.returncode == 0:
        with open(raw_sass_path, "w") as sass_file:
            sass_file.write(sass_result.stdout)
        print(f"  -> {raw_sass_path} (raw SASS reference)")

    return output_path


def cmd_assemble(cuasm_path, output_path=None):
    """Reassemble .cuasm back to .cubin using CuAssembler."""
    ensure_cuassembler()
    from CuAsm.CuAsmParser import CuAsmParser

    cuasm_path = os.path.abspath(cuasm_path)
    if output_path is None:
        # e.g. vector_add.sm_86.cuasm -> vector_add.sm_86.modified.cubin
        output_path = cuasm_path.replace(".cuasm", ".reassembled.cubin")
    output_path = os.path.abspath(output_path)

    print(f"\n[assemble] {cuasm_path} -> {output_path}")
    parser = CuAsmParser()
    parser.parse(cuasm_path)
    parser.saveAsCubin(output_path)
    print(f"  -> {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def cmd_roundtrip(source_path):
    """
    Full roundtrip test: compile -> disasm -> reassemble -> compare cubins.
    This must succeed before any hand-editing is attempted.
    """
    import filecmp

    print(f"\n[roundtrip] Testing CuAssembler stability on {source_path}")
    print("  This compiles, disassembles, reassembles, and checks the result matches.")
    print()

    cubin_path = cmd_compile(source_path)
    cuasm_path = cmd_disasm(cubin_path)
    reassembled_cubin = cmd_assemble(cuasm_path)

    # Compare the two cubins
    original_size = os.path.getsize(cubin_path)
    reassembled_size = os.path.getsize(reassembled_cubin)
    print(f"\n[roundtrip] Comparing cubins:")
    print(f"  Original:     {cubin_path} ({original_size} bytes)")
    print(f"  Reassembled:  {reassembled_cubin} ({reassembled_size} bytes)")

    if filecmp.cmp(cubin_path, reassembled_cubin, shallow=False):
        print("  RESULT: IDENTICAL — CuAssembler roundtrip is stable. Safe to hand-edit.")
    else:
        print("  RESULT: DIFFERENT — Cubins differ.")
        print("  This may be OK (CuAssembler may reorder some metadata).")
        print("  Run both cubins and compare outputs to verify correctness.")

    return cuasm_path


def cmd_all(source_path):
    """Compile + disasm. Produces editable .cuasm for hand-editing."""
    print(f"\n[all] Full workflow for {source_path}")
    print("  After this completes, hand-edit the .cuasm file, then run:")
    print("  python scripts/build.py assemble <path_to_modified.cuasm>")
    print()
    cubin_path = cmd_compile(source_path)
    cuasm_path = cmd_disasm(cubin_path)
    print(f"\n  Edit: {cuasm_path}")
    print(f"  Then: python scripts/build.py assemble {cuasm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="bare-metal GPU SASS build automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("command", choices=["compile", "disasm", "assemble", "roundtrip", "all"])
    parser.add_argument("input", help="Input file (.cu, .cubin, or .cuasm depending on command)")
    parser.add_argument("-o", "--output", help="Output file path (optional)")
    parser.add_argument("--flags", default="", help="Extra nvcc flags for compile step")

    args = parser.parse_args()

    if args.command == "compile":
        cmd_compile(args.input, args.output, args.flags)
    elif args.command == "disasm":
        cmd_disasm(args.input, args.output)
    elif args.command == "assemble":
        cmd_assemble(args.input, args.output)
    elif args.command == "roundtrip":
        cmd_roundtrip(args.input)
    elif args.command == "all":
        cmd_all(args.input)


if __name__ == "__main__":
    main()
