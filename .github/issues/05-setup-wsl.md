---
title: "Update setup.md for WSL-first development"
labels: ["documentation", "good-first-issue"]
---

## Problem
`setup.md` references Windows paths (`D:\dev\p\bare-metal`) and Visual Studio Build Tools. Actual development happens in WSL:
- `.gitignore` and scripts assume Unix shell (`export PATH=...`)
- `verify_setup.py` detects `/usr/local/cuda` (WSL convention)
- `nvcc` in WSL uses Windows driver via `/usr/lib/wsl/lib/`

Windows setup instructions are stale and misleading.

## Proposed Changes

### 1. Restructure as WSL-first
```markdown
## Quick Start (WSL — recommended)

1. Install WSL2 + Ubuntu 24.04
2. Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
3. Verify: `nvcc --version` and `nvidia-smi`
4. Clone CuAssembler: `git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler`
5. Run: `python3 scripts/verify_setup.py`
```

### 2. Move Windows instructions to appendix
```markdown
## Appendix: Native Windows Setup
> Not recommended. Use WSL unless you have a specific reason.
```

### 3. Fix CuAssembler PYTHONPATH
Current: `set PYTHONPATH=D:\...` (Windows cmd)
Fix: `export PYTHONPATH=/path/to/repo/tools/CuAssembler:$PYTHONPATH`

### 4. Remove outdated sections
- Remove Visual Studio Build Tools / `cl.exe` references (not needed in WSL)
- Remove Developer Command Prompt references

### 5. Add CUDA 12.8 compatibility note
Document why 12.8: CuAssembler stability on sm_86.

## Acceptance Criteria
- [ ] New contributor goes from zero to `verify_setup.py` passing in <10 commands
- [ ] Windows references moved to clearly marked appendix
- [ ] All paths use forward slashes / Unix conventions
- [ ] Python commands use `python3` consistently

## Effort
Low — documentation only.
