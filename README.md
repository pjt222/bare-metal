# bare-metal GPU

Hand-optimized SASS assembly kernels targeting **RTX 3070 Ti (GA104, sm_86, Ampere)**.

No cuBLAS. No cuDNN. No PyTorch. Just NVIDIA GPU assembly.

## Goal

Go as close to bare metal as NVIDIA allows:
- Write, disassemble, modify, and reassemble native GPU machine code (SASS)
- Build ML primitives from scratch: GEMM, softmax, attention, convolution
- Target: Flash Attention and diffusion model kernels in hand-tuned SASS

## Hardware

| Property | Value |
|---|---|
| GPU | RTX 3070 Ti (GA104) |
| Architecture | Ampere |
| Compute Capability | sm_86 |
| CUDA Cores | 6144 (48 SMs x 128) |
| Tensor Cores | 3rd gen (FP16, BF16, TF32, INT8) |
| VRAM | 8 GB GDDR6X |
| FP32 Peak | ~21.7 TFLOPS |
| FP16 Tensor Peak | ~174 TFLOPS |
| Shared Memory/SM | 128 KB (up to 99 KB per block) |
| Registers/SM | 64K x 32-bit |

## The Accessible Stack

```
CUDA C/C++          <- you write this
     |
     v nvcc
PTX (Virtual ISA)   <- documented, portable, stable ABI
     |
     v ptxas (driver JIT)
SASS (Native ISA)   <- sm_86, undocumented, reverse-engineered  <-- WE WORK HERE
     |
     v [SIGNATURE WALL - cryptographic, cannot cross]
Driver / Firmware   <- locked
```

## Toolchain

- **nvcc** — CUDA compiler (CUDA 12.x)
- **cuobjdump** — disassemble cubin to SASS
- **nvdisasm** — raw disassembly with control codes
- **CuAssembler** — Python tool: modify and reassemble SASS (`tools/CuAssembler/`)
- **CUDA Driver API** — load cubin directly, bypass nvcc link step

## Setup

See [setup.md](setup.md) for environment installation.

Run `python scripts/verify_setup.py` to confirm everything is working.

## Phases

| Phase | Description | Status | Highlight |
|---|---|---|---|
| 0 | Environment setup — CUDA 12.8, CuAssembler, WSL | ✅ Done | CuAssembler roundtrip verified |
| 1 | Hello World: vector add, FADD→FMUL hand-modification | ✅ Done | First SASS edit proven correct |
| 2 | ML primitives: SGEMM, HGEMM, softmax, layernorm, activations | ✅ Done | 7,853 GFLOPS HGEMM via HMMA.16816.F32 |
| 3 | Flash Attention: scalar → 4-warp → Br=16 HMMA (19× speedup) | ✅ Done | 2.8 ms at seq=1024 with Tensor Cores |
| 4 | Diffusion UNet: timestep emb, GroupNorm, Conv2d, ResNet, cross-attn | ✅ Done | Full SASS primitive inventory |
| 5 | Sparse GEMM, INT8 quantization, optimized epilogues | ✅ Done | 41,930 sparse-equiv GFLOPS |

## Phase 5 — Sparse & Quantized GEMM (Complete)

| Kernel | Highlight | Peak Result |
|--------|-----------|-------------|
| Sparse HGEMM 2:4 | `mma.sp` with ldmatrix | 41,721 dense-equiv GFLOPS |
| Sparse IGEMM 2:4 | INT8 sparse Tensor Cores | 39,674 dense-equiv TOPS at 2048³ |
| Online FP16→INT8 | Quantize on-the-fly in kernel | 16,646 GFLOPS (2.1× vs naive HGEMM) |
| Bank-conflict-free INT8 | Optimized smem epilogue | 17,070 GFLOPS |

> Phase 5 represents the optimization frontier: sparse patterns, INT8 quantization, and tuned epilogues. See [`docs/gpu_reflections.md`](docs/gpu_reflections.md) for the full empirical analysis.

## Current Best Results (RTX 3070 Ti Laptop)

### GEMM
| Kernel | Size | Performance | % Peak |
|--------|------|-------------|--------|
| HGEMM 16-warp | 4096³ | **31,910 GFLOPS** | 18.3% |
| HGEMM sparse 2:4 | 2048³ | **41,930** dense-equiv GFLOPS | — |
| IGEMM 128×256 (1 blk/SM) | 4096³ | **27,591 TOPS** | 4.0% |
| IGEMM pipelined cp.async | 4096³ | **20,688 TOPS** | 3.0% |
| IGEMM 128×256 (1 blk/SM) | 4096³ | **27,591 TOPS** | 4.0% |
| Online FP16→INT8 quant | 4096³ | **17,070 GFLOPS** | 9.6% |
| Sparse INT8 mma.sp | 2048³ | **39,674 dense-equiv TOPS** | — |

### Flash Attention
| Kernel | Config | Time | GFLOPS |
|--------|--------|------|--------|
| Flash Attention Br=16 regpv | seq=1024, b=8, h=8 | **2.81 ms** | **6,112** |

### Diffusion UNet
| Kernel | Config | Time | Performance |
|--------|--------|------|-------------|
| Implicit GEMM conv2d | 64×64, Cin=Cout=320 | **1.13 ms** | **6,687 GFLOPS** |

## Phase 4 Results Summary

All components of a Stable Diffusion UNet block implemented and verified:

| Kernel | SASS Instructions | Performance |
|--------|------------------|-------------|
| Timestep embedding | `MUFU.SIN`, `MUFU.COS`, `MUFU.EX2` | 153 GB/s at d=512, batch=1024 |
| GroupNorm (NHWC) | `SHFL.BFLY`, `MUFU.RSQ`, `MUFU.RCP` | 28–74 GB/s |
| GroupNorm+SiLU fused | above + `MUFU.EX2` for SiLU | reads X once (saves 1 tensor pass) |
| Conv2d 3×3 direct | `FFMA` (310 per pass, 9×unrolled) | 265–299 GFLOPS |
| Conv2d implicit GEMM | `HMMA.16816.F32`, precomputed coords | 6,687 GFLOPS (22× over direct) |
| ResNet Block | all of the above + `FADD` | 265 GFLOPS at SD 320ch 16×16 |
| Cross-attention | `HMMA.16816.F32`, `SHFL.BFLY`, `MUFU.EX2` | 2,255 GFLOPS at 32×32 with CLIP-77 |

## Key References

- [CuAssembler](https://github.com/cloudcores/CuAssembler) — the SASS assembler we use
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — cuobjdump, nvdisasm docs
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) — performance optimization
- [docs/ampere_sass_reference.md](docs/ampere_sass_reference.md) — quick SASS instruction reference
- [docs/control_codes.md](docs/control_codes.md) — stall counts, barriers, yield
- [docs/memory_hierarchy.md](docs/memory_hierarchy.md) — GA104 memory system
