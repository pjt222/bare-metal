#!/usr/bin/env python3
"""
tools/cuasm_helper.py - Minimal Python helper for CuAssembler-only steps.

build.R shells out to this for the two operations that have no R equivalent:
  disasm   <cubin> <cuasm>     # CubinFile.saveAsCuAsm
  assemble <cuasm> <cubin>     # CuAsmParser.saveAsCubin

CuAssembler is a Python library (cloudcores/CuAssembler). Everything else
in build.R (compile, roundtrip orchestration, file comparison) stays in R.

Usage:
    python3 tools/cuasm_helper.py disasm   <input.cubin> <output.cuasm>
    python3 tools/cuasm_helper.py assemble <input.cuasm> <output.cubin>
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUASSEMBLER_PATH = os.path.join(REPO_ROOT, "tools", "CuAssembler")
sys.path.insert(0, CUASSEMBLER_PATH)


def disasm(cubin_path, cuasm_path):
    from CuAsm.CubinFile import CubinFile
    CubinFile(cubin_path).saveAsCuAsm(cuasm_path)


def assemble(cuasm_path, cubin_path):
    from CuAsm.CuAsmParser import CuAsmParser
    parser = CuAsmParser()
    parser.parse(cuasm_path)
    parser.saveAsCubin(cubin_path)


def main():
    if len(sys.argv) != 4:
        sys.stderr.write(__doc__)
        sys.exit(2)
    op, src, dst = sys.argv[1:4]
    if op == "disasm":
        disasm(src, dst)
    elif op == "assemble":
        assemble(src, dst)
    else:
        sys.stderr.write(f"unknown op: {op}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
