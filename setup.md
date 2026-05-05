# Environment Setup

> **Development happens in WSL2.** The Windows NVIDIA driver exposes the GPU to WSL via `/usr/lib/wsl/lib/`. All build scripts, benchmarks, and SASS tooling assume a Linux environment.

## Quick Start (WSL2 — Recommended)

### 1. Install CUDA Toolkit in WSL

```bash
# Ubuntu 24.04
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

Verify:
```bash
nvcc --version
cuobjdump --version
nvdisasm --version
nvidia-smi
```

> **Why CUDA 12.8?** This project targets sm_86 (RTX 3070 Ti). CuAssembler was developed against CUDA 11.x–12.x; the cubin format may differ in 13.x. The driver is forward-compatible, so 12.8 kernels run on newer drivers.

### 2. Python Dependencies

```bash
python3 -m pip install pyelftools sympy
```

If pip warns about PEP 668 (externally managed environment):
```bash
python3 -m pip install pyelftools sympy --break-system-packages
```

### 3. CuAssembler

```bash
git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler
```

Add to `~/.bashrc` (or set per-session):
```bash
export PYTHONPATH="$PWD/tools/CuAssembler:$PYTHONPATH"
```

### 4. Verify Everything

```bash
python3 scripts/verify_setup.py
```

All checks should pass before proceeding to Phase 1.

---

## Appendix: Native Windows Setup

> ⚠️ **Not recommended.** Use WSL2 unless you have a specific reason to build on Windows natively.

### CUDA Toolkit (Windows)
Download from: https://developer.nvidia.com/cuda-downloads  
Default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\`

### Build Tools
nvcc on Windows requires MSVC (`cl.exe`). Install **Visual Studio 2022 Build Tools** with the **Desktop development with C++** workload, then use the **Developer Command Prompt for VS 2022**.

### PYTHONPATH (Windows cmd)
```cmd
set PYTHONPATH=D:\dev\p\bare-metal\tools\CuAssembler;%PYTHONPATH%
```

---

## Troubleshooting

**nvcc not found in WSL**:  
```bash
export PATH=/usr/local/cuda/bin:$PATH
```
Add to `~/.bashrc` to make permanent.

**WSL: GPU detected by `nvidia-smi` but CUDA programs fail**:  
Check driver/toolkit version compatibility:
```bash
cat /proc/driver/nvidia/version   # Windows driver version
nvcc --version                    # CUDA toolkit version
```
The toolkit version must be ≤ the driver's maximum supported CUDA version.

**CuAssembler import fails**:  
Ensure `tools/CuAssembler/` is in `sys.path`. `scripts/build.py` and `scripts/verify_setup.py` handle this automatically.

**GPU not detected**:  
Run `nvidia-smi`. If this fails, reinstall the Windows NVIDIA driver (WSL uses the Windows driver, not a separate Linux driver).
