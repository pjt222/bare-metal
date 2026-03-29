"""
verify_setup.py — Check that the bare-metal GPU dev environment is ready.

Run from the repo root:
    python scripts/verify_setup.py
"""

import subprocess
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUASSEMBLER_PATH = os.path.join(REPO_ROOT, "tools", "CuAssembler")

# WSL: CUDA tools live in /usr/local/cuda/bin — add to PATH if not already there
CUDA_BIN = "/usr/local/cuda/bin"
if os.path.isdir(CUDA_BIN) and CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = CUDA_BIN + ":" + os.environ.get("PATH", "")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def run_command(cmd):
    """Run a shell command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Timed out"
    except Exception as exception:
        return False, "", str(exception)


def check_command(label, cmd, expected_substring=None):
    success, stdout, stderr = run_command(cmd)
    output = stdout or stderr
    if not success:
        print(f"{FAIL} {label}")
        print(f"       Command: {cmd}")
        print(f"       Output:  {output}")
        return False
    if expected_substring and expected_substring not in output:
        print(f"{FAIL} {label} — expected '{expected_substring}' in output")
        print(f"       Output: {output}")
        return False
    print(f"{PASS} {label}")
    if output:
        first_line = output.splitlines()[0]
        print(f"       {first_line}")
    return True


PYTHON = "python3" if os.path.isdir("/usr/local/cuda") else "python"


def check_python_import(label, import_statement):
    success, stdout, stderr = run_command(f'{PYTHON} -c "{import_statement}"')
    if not success:
        print(f"{FAIL} {label}")
        print(f"       {stderr}")
        return False
    print(f"{PASS} {label}")
    return True


def check_cuassembler_path():
    if os.path.isdir(CUASSEMBLER_PATH):
        print(f"{PASS} CuAssembler directory exists: {CUASSEMBLER_PATH}")
        return True
    print(f"{FAIL} CuAssembler not found at: {CUASSEMBLER_PATH}")
    print(f"       Run: git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler")
    return False


def check_gpu_info():
    success, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader"
    )
    if not success:
        print(f"{FAIL} nvidia-smi — GPU not detected or driver not installed")
        return False
    print(f"{PASS} GPU detected:")
    for line in stdout.splitlines():
        name, compute_cap, memory = [part.strip() for part in line.split(",")]
        print(f"       Name:             {name}")
        print(f"       Compute Cap:      sm_{compute_cap.replace('.', '')}")
        print(f"       Memory:           {memory}")
        if "3070" not in name and "86" not in compute_cap:
            print(f"  {INFO} Expected RTX 3070 Ti (sm_86) — got {name} (sm_{compute_cap})")
    return True


def main():
    print("=" * 60)
    print("  bare-metal GPU — Environment Verification")
    print("  Target: RTX 3070 Ti (GA104, sm_86, Ampere)")
    print("=" * 60)
    print()

    results = []

    print("-- CUDA Toolchain --")
    results.append(check_command("nvcc", "nvcc --version", "release"))
    results.append(check_command("cuobjdump", "cuobjdump --version", "cuobjdump"))
    results.append(check_command("nvdisasm", "nvdisasm --version", "nvdisasm"))
    print()

    print("-- GPU Driver --")
    results.append(check_gpu_info())
    print()

    print("-- Python Dependencies --")
    results.append(check_python_import("pyelftools", "import elftools; print('ok')"))
    results.append(check_python_import("sympy", "import sympy; print('ok')"))
    print()

    print("-- CuAssembler --")
    cuasm_dir_ok = check_cuassembler_path()
    results.append(cuasm_dir_ok)
    if cuasm_dir_ok:
        # Add CuAssembler to path for this check
        cuasm_in_path = CUASSEMBLER_PATH in sys.path
        if not cuasm_in_path:
            sys.path.insert(0, CUASSEMBLER_PATH)
        results.append(
            check_python_import(
                "CuAssembler import",
                f"import sys; sys.path.insert(0, r'{CUASSEMBLER_PATH}'); from CuAsm.CubinFile import CubinFile; print('CubinFile ok')"
            )
        )
        results.append(
            check_python_import(
                "CuAsmParser import",
                f"import sys; sys.path.insert(0, r'{CUASSEMBLER_PATH}'); from CuAsm.CuAsmParser import CuAsmParser; print('CuAsmParser ok')"
            )
        )
    print()

    print("-- sm_86 Compilation Test --")
    # Write a tiny test kernel and compile it
    test_cu_path = os.path.join(REPO_ROOT, "_verify_test.cu")
    test_cubin_path = os.path.join(REPO_ROOT, "_verify_test.cubin")
    try:
        with open(test_cu_path, "w") as f:
            f.write('extern "C" __global__ void test_kernel(float *x) { x[threadIdx.x] = 1.0f; }\n')
        success, stdout, stderr = run_command(
            f'nvcc --cubin -arch=sm_86 -o "{test_cubin_path}" "{test_cu_path}"'
        )
        if success and os.path.exists(test_cubin_path):
            print(f"{PASS} nvcc --cubin -arch=sm_86 compiles successfully")
            results.append(True)
            # Also check disassembly works
            success2, sass_out, _ = run_command(f'cuobjdump -sass "{test_cubin_path}"')
            if success2 and "SASS" in sass_out or "code" in sass_out.lower() or len(sass_out) > 10:
                print(f"{PASS} cuobjdump disassembly works")
                results.append(True)
            else:
                print(f"{FAIL} cuobjdump disassembly produced unexpected output")
                results.append(False)
        else:
            print(f"{FAIL} nvcc --cubin -arch=sm_86 failed")
            print(f"       {stderr}")
            results.append(False)
    finally:
        for path in [test_cu_path, test_cubin_path]:
            if os.path.exists(path):
                os.remove(path)
    print()

    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    if passed == total:
        print(f"{PASS} All {total} checks passed — ready for bare-metal GPU work!")
        print()
        print("  Next step: read phase1/README.md and run the vector_add hello world")
    else:
        print(f"{FAIL} {total - passed}/{total} checks failed — fix issues above before proceeding")
        print()
        print("  See setup.md for installation instructions")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
