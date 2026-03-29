# Environment Setup

## 1. CUDA Toolkit 12.6

Download from: https://developer.nvidia.com/cuda-12-6-0-download-archive
Select: Windows 11 > x86_64 > exe (local)

After install, verify these are on PATH (open a new terminal):
```
nvcc --version
cuobjdump --version
nvdisasm --version
```

Default install path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\`

> **Why 12.6 and not 13.x?** CuAssembler was developed against CUDA 11.x/12.x.
> The cubin format may differ in 13.x. Driver is forward-compatible so 12.6 runs fine.

## 2. Visual Studio 2022 Build Tools

nvcc requires the MSVC C++ compiler (`cl.exe`).

Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
Workload to select: **Desktop development with C++**

After install, use the **Developer Command Prompt for VS 2022** (or add cl.exe to PATH).

## 3. Python Dependencies

```bash
pip install pyelftools sympy
```

## 4. CuAssembler

```bash
cd D:\dev\p\bare-metal
git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler
```

Add to PYTHONPATH (add to your shell profile or set per-session):
```bash
set PYTHONPATH=D:\dev\p\bare-metal\tools\CuAssembler;%PYTHONPATH%
```

## 5. Verify Everything

```bash
cd D:\dev\p\bare-metal
python scripts/verify_setup.py
```

All checks should pass before proceeding to Phase 1.

## Troubleshooting

**nvcc not found**: Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin` to PATH.

**cl.exe not found**: Open a Developer Command Prompt for VS 2022, or run:
`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

**CuAssembler import fails**: Check PYTHONPATH includes `D:\dev\p\bare-metal\tools\CuAssembler`.

**GPU not detected**: Run `nvidia-smi` — if this fails, reinstall the NVIDIA driver.
