/**
 * graph-data.js
 *
 * Complete knowledge graph of the bare-metal GPU research project.
 * Each node is a topic with summary, formulas, and performance metrics.
 * Edges show how topics connect (builds-on, uses, enables, optimizes, applies-to).
 *
 * Layout: roughly left-to-right by phase, concepts below, hardware at bottom.
 */

export const nodes = [
  // ===== Phase 1: SASS Proof of Concept =====
  {
    id: 'vector-add',
    label: 'Vector Add',
    phase: 'phase1',
    summary: 'SASS hand-editing proof of concept. Compiled a vector add kernel, disassembled to .cuasm with CuAssembler, changed FADD→FMUL in the binary, reassembled. Proved the compile→disasm→edit→reassemble pipeline works.',
    formulas: [
      'C[i] = A[i] + B[i]  →  C[i] = A[i] × B[i]',
      'Pipeline: .cu → nvcc → .cubin → CuAssembler → .cuasm → edit → .cubin',
    ],
    metrics: 'CuAssembler roundtrip verified',
    x: -1100, y: 0,
  },

  // ===== Phase 2: ML Primitives =====
  {
    id: 'sgemm',
    label: 'SGEMM',
    phase: 'phase2',
    summary: 'FP32 scalar GEMM. Three stages: naive (global loads), tiled (shared memory + BAR.SYNC), hand-tuned (.cuasm stall code editing). Tiled kernel uses 32×32 smem tiles with FFMA inner loop.',
    formulas: [
      'C[i][j] = Σ_k A[i][k] · B[k][j]',
      'GFLOPS = 2·M·N·K / (t_ms / 1000) / 1e9',
    ],
    metrics: '1,031 GFLOPS at 2048² (~5% of 21.7 TFLOPS peak)',
    x: -700, y: -300,
  },
  {
    id: 'hgemm',
    label: 'HGEMM (Dense)',
    phase: 'phase2',
    summary: 'FP16 Tensor Core GEMM using HMMA.16816.F32. 128×128 tiles, BK=32, double-buffered cp.async, ldmatrix fragment loading. Each HMMA warp-wide instruction does 16×8×16 = 2048 FP16 MACs.',
    formulas: [
      'mma.sync.aligned.m16n8k16.f32.f16',
      'HMMA.16816.F32 Rd, Ra, Rb, Rc  (warp-wide)',
      '2048 MACs/instruction × 2 HMMA/mma_sync',
    ],
    metrics: '32,197 GFLOPS at 4096³ (93% FP16 TC peak)',
    x: -700, y: 0,
  },
  {
    id: 'softmax',
    label: 'Softmax',
    phase: 'phase2',
    summary: 'Online softmax using SHFL.BFLY warp butterfly reduction (5 shuffles for 32-lane reduce), MUFU.EX2 for exp via 2^(x·log₂e), and MUFU.RCP for reciprocal normalization.',
    formulas: [
      'softmax(xᵢ) = exp(xᵢ - max) / Σⱼ exp(xⱼ - max)',
      'exp(x) = 2^(x · 1.4427) → FMUL + MUFU.EX2',
      '5× SHFL.BFLY → full 32-lane reduction',
    ],
    metrics: '410 GB/s (67% of 608 GB/s DRAM peak)',
    x: -700, y: 300,
  },
  {
    id: 'layernorm',
    label: 'LayerNorm',
    phase: 'phase2',
    summary: 'Warp-cooperative layer normalization using Welford single-pass online algorithm. Computes mean and variance simultaneously, merges via parallel Welford combine. Uses MUFU.RSQ for 1/√(var+ε).',
    formulas: [
      'yᵢ = γᵢ · (xᵢ - μ) / √(σ² + ε) + βᵢ',
      'Welford: δ = x - μ; μ += δ/n; M₂ += δ·(x - μ)',
      'MUFU.RSQ → 1/√(var + ε) in one cycle',
    ],
    metrics: '413 GB/s (68% peak, single-pass Welford)',
    x: -700, y: 150,
  },
  {
    id: 'activations',
    label: 'Activations',
    phase: 'phase2',
    summary: 'SiLU, GELU, ReLU via MUFU special function unit. Surprising result: all activations hit ~394 GB/s regardless of compute cost — element-wise ops are memory-bandwidth-bound. MUFU runs at ¼ FFMA throughput but is hidden by memory latency.',
    formulas: [
      'SiLU(x) = x · σ(x) = x / (1 + 2^(-x·log₂e))',
      'GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))',
      'MUFU.EX2 + MUFU.RCP (SiLU) | MUFU.TANH (GELU)',
    ],
    metrics: '394 GB/s — all activations equal (65% DRAM BW peak)',
    x: -700, y: -150,
  },
  {
    id: 'hgemm-sparse',
    label: 'Sparse HGEMM',
    phase: 'phase2',
    summary: '2:4 structured sparsity using HMMA.SP.16816. A matrix compressed to 50% with metadata selecting 2-of-4 elements per group. Smem-tiled with ldmatrix A-fragment loading. A tiles are half-size (128×16 compressed vs 128×32 dense).',
    formulas: [
      'mma.sp.sync.aligned.m16n8k16.f32.f16',
      'Metadata 0x4444: positions {0,1} per group of 4',
      'A_compressed = 128×16 (half of 128×32)',
    ],
    metrics: '19,025 dense-equiv GFLOPS (59% of dense, +54% with ldmatrix)',
    x: -350, y: 0,
  },
  {
    id: 'igemm',
    label: 'INT8 IGEMM',
    phase: 'phase2',
    summary: 'INT8 Tensor Core GEMM using IMMA.16816.S8.S8 for 4× throughput over FP16. Pipelined with cp.async double-buffering. Key finding: IMMA sustains S01 stall (vs HMMA S08) — shorter pipeline, more throughput-sensitive to scheduling.',
    formulas: [
      'IMMA.16816.S8.S8 → INT8×INT8→INT32',
      'Dequantize: I2FP.F32.S32 + FMUL(scale)',
      '696 TOPS peak (4× over FP16 TC)',
    ],
    metrics: '20,688 TOPS (cp.async pipelined, +88% over naive)',
    x: -350, y: -300,
  },

  // ===== Phase 3: Flash Attention =====
  {
    id: 'flash-attention',
    label: 'Flash Attention',
    phase: 'phase3',
    summary: 'Memory-efficient attention: never materializes N×N score matrix. Three variants: scalar 1-warp → 4-warp shared KV → Br=16 HMMA tiled. Online softmax recurrence maintains running max/sum across KV tiles. 19× speedup from scalar to HMMA.',
    formulas: [
      'O = softmax(QK^T / √d) · V',
      'Online: m_new = max(m_old, tile_max)',
      'Rescale: o = o · exp(m_old - m_new)',
      'Final: O[q] = o / l  (MUFU.RCP)',
    ],
    metrics: '7,160 GFLOPS (Br=16 HMMA, 19× over scalar)',
    x: 0, y: 0,
  },

  // ===== Phase 4: Diffusion UNet =====
  {
    id: 'timestep-emb',
    label: 'Timestep Embedding',
    phase: 'phase4',
    summary: 'Sinusoidal positional encoding for diffusion timesteps. Maps scalar t ∈ [0,1000) to d_model-dim vector. Requires --use_fast_math for MUFU.SIN/COS hardware units; without it, compiles to slow software polynomial.',
    formulas: [
      'emb[2i] = sin(t · exp(-log(10000) · i / d/2))',
      'emb[2i+1] = cos(t · exp(-log(10000) · i / d/2))',
      'MUFU.SIN, MUFU.COS, MUFU.EX2',
    ],
    metrics: '153 GB/s (compute-bound on MUFU)',
    x: 400, y: -300,
  },
  {
    id: 'groupnorm',
    label: 'GroupNorm',
    phase: 'phase4',
    summary: 'Group normalization for diffusion UNet. Normalizes over (C/G)×H×W per (sample,group). Uses parallel Welford (same as LayerNorm). NCHW layout ~7% faster than NHWC. Fused with SiLU in ResBlock.',
    formulas: [
      'μ[n,g] = mean(X[n, group_g, :, :])',
      'Y = γ · (X - μ) / √(σ² + ε) + β',
      'Welford merge: O(1) per combine',
    ],
    metrics: '73.7 GB/s (NCHW, warp Welford + MUFU.RSQ)',
    x: 400, y: -150,
  },
  {
    id: 'conv2d',
    label: 'Conv2d',
    phase: 'phase4',
    summary: '3×3 NHWC convolution. Direct kernel with 9× unrolled inner loop → 310 FFMA in SASS. Reads input 9× (no smem halo cache). Path to higher perf: im2col transforms conv into GEMM, enabling Tensor Cores.',
    formulas: [
      'Y[n,h,w,c_out] = Σ_{kh,kw,c_in} W·X[n,h+kh-1,w+kw-1,c_in]',
      'im2col: [N·H·W, C·kh·kw] × [C·kh·kw, C_out]',
      '310 FFMA per conv pass (fully unrolled)',
    ],
    metrics: '299 GFLOPS (FP32 direct), im2col+WMMA: 24× faster',
    x: 400, y: 0,
  },
  {
    id: 'resblock',
    label: 'ResBlock',
    phase: 'phase4',
    summary: 'Full Stable Diffusion residual block: GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv→Add. Fused GroupNorm+SiLU eliminates one full tensor read/write. Conv2d dominates at 95% of runtime.',
    formulas: [
      'x_out = x + conv(silu(gn(conv(silu(gn(x))))))',
      'Fused: gn_silu reads X once → stats + normalize + SiLU',
      'SiLU: x / (1 + exp2f(-x·log₂e))  →  MUFU.EX2 + MUFU.RCP',
    ],
    metrics: '265 GFLOPS (5 kernels, conv2d = 95% of time)',
    x: 400, y: 150,
  },
  {
    id: 'cross-attention',
    label: 'Cross-Attention',
    phase: 'phase4',
    summary: 'Image-to-text cross-attention for UNet conditioning. Q from spatial features, K/V from CLIP text tokens (seq=77). Same Flash Attention algorithm but mandatory KV padding mask — exp(0-max) ≈ 0.61 inflates softmax by 31× without -∞ masking.',
    formulas: [
      'A = softmax(Q_img · K_text^T / √d)  [seq_q × 77]',
      'O = A · V_text  [seq_q × d_head]',
      'Padding fix: score = padded ? -∞ : score',
    ],
    metrics: '4,017 GFLOPS (512-token context)',
    x: 400, y: 300,
  },

  // ===== Cross-Cutting Concepts =====
  {
    id: 'four-laws',
    label: 'Four Laws of GA104',
    phase: 'concept',
    summary: '1. Feed Tensor Cores continuously — overlap loads with HMMA. 2. Read each byte of DRAM exactly once — im2col, streaming. 3. Fill warp schedulers — 32 warps/SM ideal, 8 minimum. 4. Never cross the 50 KB smem cliff.',
    formulas: [],
    metrics: 'Core optimization principles',
    x: -100, y: 500,
  },
  {
    id: 'smem-cliff',
    label: '50 KB Smem Cliff',
    phase: 'concept',
    summary: 'GA104 has 100 KB smem/SM. At ≤50 KB/block → 2 blocks/SM → 8 warps (good). At >50 KB → 1 block/SM → 4 warps → exposed DRAM stalls. Measured: 48 KB = 2 blocks ✓, 56 KB = 1 block → 2× regression.',
    formulas: [
      'Cliff = 100 KB / 2 = 50 KB per block',
      '≤50 KB → 2 blocks/SM → 8+ warps',
      '>50 KB → 1 block/SM → 4 warps (2× slower)',
    ],
    metrics: 'Occupancy collapse at >50 KB',
    x: 200, y: 500,
  },
  {
    id: 'cp-async',
    label: 'cp.async Pipelining',
    phase: 'concept',
    summary: 'LDGSTS: async global→shared copy bypassing register file. Double-buffered: load tile N+1 while computing tile N. Benefits depend on compute/load ratio: helpful for short loops (8 IMMA/tile: +35%), harmful for long loops (64 HMMA/tile: -5%).',
    formulas: [
      '__pipeline_memcpy_async(smem, gmem, 16)',
      'LDGSTS.E.128 (16 bytes, bypasses RF)',
      'Double-buffer: smem[2][BM×BK]',
    ],
    metrics: '+35% for IGEMM, -5% for Flash Attention',
    x: -400, y: 500,
  },
  {
    id: 'double-buffering',
    label: 'Double Buffering',
    phase: 'concept',
    summary: 'Overlap load and compute phases using two smem buffers. While warps compute on buffer[0], cp.async fills buffer[1]. Swap each K-tile iteration. Essential for hiding DRAM latency in bandwidth-bound kernels.',
    formulas: [
      'smem[2][BM × STRIDE]  (ping-pong)',
      'Load buf[1-cur] while compute buf[cur]',
      '__pipeline_commit(); __pipeline_wait_prior(1);',
    ],
    metrics: 'Hides DRAM latency when compute/load ratio is short',
    x: -400, y: 650,
  },
  {
    id: 'warp-reduction',
    label: 'Warp Reduction',
    phase: 'concept',
    summary: 'SHFL.BFLY butterfly pattern: 5 shuffle instructions reduce 32 lanes to a single value. No shared memory needed. Used in softmax, layernorm, groupnorm, and Flash Attention for max/sum/Welford reductions.',
    formulas: [
      'SHFL.BFLY offsets: 16, 8, 4, 2, 1',
      '5 instructions → full 32-lane reduce',
      'All lanes hold result (broadcast)',
    ],
    metrics: 'Zero smem, used in 6+ kernels',
    x: -100, y: 650,
  },
  {
    id: 'tensor-core-scheduling',
    label: 'Tensor Core Scheduling',
    phase: 'concept',
    summary: 'HMMA.16816 has S08 hardware pipeline minimum between back-to-back instructions — irreducible. IMMA sustains S01/S02 (shorter pipeline). Fragment loads between HMMAs must complete in ≤8 cycles or stalls grow (S12 = 4 excess cycles).',
    formulas: [
      'HMMA: S08 minimum (hardware constraint)',
      'IMMA: S01-S02 achievable (verified)',
      'Excess stalls: S12 - S08 = 4 wasted cycles',
    ],
    metrics: 'S08 irreducible for HMMA; S02 optimal for IMMA',
    x: 200, y: 650,
  },

  // ===== Hardware =====
  {
    id: 'ga104',
    label: 'GA104 (RTX 3070 Ti)',
    phase: 'hardware',
    summary: '48 SMs, 128 CUDA cores/SM, 64K registers/SM, 100 KB shared memory/SM. Ampere architecture (sm_86). L2 cache: 4 MB. DRAM bandwidth: 608 GB/s. The target hardware for all 5 phases of this research.',
    formulas: [
      'FP32 peak: 21.7 TFLOPS (FFMA)',
      'FP16 TC peak: 174 TFLOPS (HMMA)',
      'INT8 TC peak: 696 TOPS (IMMA)',
      'DRAM BW: 608 GB/s | L2: 4 MB',
    ],
    metrics: 'Ampere sm_86, all kernels target this GPU',
    x: 50, y: 850,
  },
];

export const edges = [
  // ===== Phase evolution (builds-on) =====
  { source: 'vector-add', target: 'sgemm', label: 'SASS editing pipeline → matrix multiply', type: 'builds-on' },
  { source: 'sgemm', target: 'hgemm', label: 'FP32 FFMA → FP16 Tensor Core (HMMA)', type: 'builds-on' },
  { source: 'hgemm', target: 'hgemm-sparse', label: 'Dense HMMA → 2:4 sparse HMMA.SP', type: 'builds-on' },
  { source: 'hgemm', target: 'igemm', label: 'FP16 HMMA → INT8 IMMA (4× throughput)', type: 'builds-on' },
  { source: 'flash-attention', target: 'cross-attention', label: 'Self-attention → cross-attention + KV masking', type: 'builds-on' },

  // ===== Building blocks (uses) =====
  { source: 'hgemm', target: 'flash-attention', label: 'GEMM computes QK^T and PV products', type: 'uses' },
  { source: 'softmax', target: 'flash-attention', label: 'Online softmax enables KV tiling', type: 'enables' },
  { source: 'hgemm', target: 'conv2d', label: 'im2col transforms conv → GEMM', type: 'uses' },
  { source: 'layernorm', target: 'groupnorm', label: 'Same parallel Welford algorithm', type: 'builds-on' },
  { source: 'groupnorm', target: 'resblock', label: 'Fused GroupNorm + SiLU', type: 'uses' },
  { source: 'activations', target: 'resblock', label: 'SiLU via MUFU.EX2 + MUFU.RCP', type: 'uses' },
  { source: 'conv2d', target: 'resblock', label: 'Conv2d = 95% of ResBlock runtime', type: 'uses' },
  { source: 'flash-attention', target: 'cross-attention', label: 'Same online softmax algorithm', type: 'uses' },
  { source: 'warp-reduction', target: 'softmax', label: 'SHFL.BFLY for row max and sum', type: 'uses' },
  { source: 'warp-reduction', target: 'layernorm', label: 'SHFL.BFLY for Welford merge', type: 'uses' },

  // ===== Optimization techniques (optimizes) =====
  { source: 'cp-async', target: 'hgemm', label: 'LDGSTS double-buffered smem loading', type: 'optimizes' },
  { source: 'cp-async', target: 'igemm', label: '+35% TOPS (short inner loop benefits)', type: 'optimizes' },
  { source: 'cp-async', target: 'hgemm-sparse', label: 'Double-buffered A_comp + B tiles', type: 'optimizes' },
  { source: 'double-buffering', target: 'cp-async', label: 'Ping-pong smem enables async loads', type: 'enables' },

  // ===== Hardware constraints (applies-to) =====
  { source: 'smem-cliff', target: 'hgemm', label: 'BK=32 keeps smem at 48 KB → 2 blocks/SM', type: 'applies-to' },
  { source: 'smem-cliff', target: 'flash-attention', label: '48 KB smem → 3 blocks/SM for Br=16', type: 'applies-to' },
  { source: 'smem-cliff', target: 'hgemm-sparse', label: '29 KB smem → 2 blocks/SM ✓', type: 'applies-to' },
  { source: 'tensor-core-scheduling', target: 'hgemm', label: 'S08 minimum between HMMA', type: 'applies-to' },
  { source: 'tensor-core-scheduling', target: 'igemm', label: 'IMMA: S02 optimal (not S08)', type: 'applies-to' },
  { source: 'tensor-core-scheduling', target: 'hgemm-sparse', label: 'HMMA.SP also S08 (same pipeline)', type: 'applies-to' },

  // ===== Principles (enables) =====
  { source: 'ga104', target: 'four-laws', label: 'Hardware constraints → optimization laws', type: 'enables' },
  { source: 'four-laws', target: 'smem-cliff', label: 'Law 4: never cross 50 KB', type: 'enables' },
  { source: 'four-laws', target: 'tensor-core-scheduling', label: 'Law 1: feed Tensor Cores continuously', type: 'enables' },
  { source: 'ga104', target: 'smem-cliff', label: '100 KB smem/SM → 50 KB cliff', type: 'enables' },
];
