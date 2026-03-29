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

## Phase 4 Results Summary

All components of a Stable Diffusion UNet block implemented and verified:

| Kernel | SASS Instructions | Performance |
|--------|------------------|-------------|
| Timestep embedding | `MUFU.SIN`, `MUFU.COS`, `MUFU.EX2` | 153 GB/s at d=512, batch=1024 |
| GroupNorm (NHWC) | `SHFL.BFLY`, `MUFU.RSQ`, `MUFU.RCP` | 28–74 GB/s |
| GroupNorm+SiLU fused | above + `MUFU.EX2` for SiLU | reads X once (saves 1 tensor pass) |
| Conv2d 3×3 NHWC | `FFMA` (310 per pass, 9×unrolled) | 265–299 GFLOPS |
| ResNet Block | all of the above + `FADD` | 265 GFLOPS at SD 320ch 16×16 |
| Cross-attention | `HMMA.16816.F32`, `SHFL.BFLY`, `MUFU.EX2` | 2,255 GFLOPS at 32×32 with CLIP-77 |

## Key References

- [CuAssembler](https://github.com/cloudcores/CuAssembler) — the SASS assembler we use
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — cuobjdump, nvdisasm docs
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) — performance optimization
- [docs/ampere_sass_reference.md](docs/ampere_sass_reference.md) — quick SASS instruction reference
- [docs/control_codes.md](docs/control_codes.md) — stall counts, barriers, yield
- [docs/memory_hierarchy.md](docs/memory_hierarchy.md) — GA104 memory system
