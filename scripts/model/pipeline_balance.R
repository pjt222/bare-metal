#!/usr/bin/env Rscript
# pipeline_balance.R — Estimate pipeline balance: load-vs-compute overlap for GEMM kernels
# Models cp.async effectiveness, determines if kernel is load-bound or compute-bound.
# Usage: Rscript scripts/model/pipeline_balance.R <BM> <BN> <BK> <warps> <mma_per_tile>
#   mma_per_tile: number of mma ops per k-step per warp (e.g. 4 for 2x2 tiles)

cat("=== Pipeline Balance Model (sm_86) ===\n\n")

# Instruction latencies (cycles @ 1.8 GHz)
L_CYCLES <- list(
  cp_async = 150,      # cp.async latency to smem visible
  lds_u8 = 25,         # LDS.U8 scalar load
  lds_32 = 25,         # LDS.32 / LDSM
  ldmatrix = 25,       # ldmatrix (same as LDS)
  mma_sp = 32,         # mma.sp INT8
  mma = 16,            # regular mma FP16
  ffma = 4,            # FFMA (non-Tensor)
  prmt = 4,            # PRMT shuffle
  syncthreads = 20     # __syncthreads(pipeline) overhead
)

# Throughput: instrs per cycle per SM (theoretical)
T_PER_SM <- list(
  cp_async = 8,        # per SM, many outstanding
  lds = 16,            # LDS units per SM
  mma = 4,             # Tensor Core per SM
  ffma = 128,          # FP32 cores
  prmt = 16            # PRMT ops per SM
)

DRAM_BW <- 608.0       # GB/s
SM_COUNT <- 48

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 5) {
  BM <- as.integer(args[1]); BN <- as.integer(args[2]); BK <- as.integer(args[3])
  warps <- as.integer(args[4]); mma_per_tile <- as.integer(args[5])
} else {
  cat("Interactive mode (enter values):\n")
  BM <- as.integer(readline("BM (tile rows): "))
  BN <- as.integer(readline("BN (tile cols): "))
  BK <- as.integer(readline("BK (tile depth): "))
  warps <- as.integer(readline("Warps per block: "))
  mma_per_tile <- as.integer(readline("MMA ops per k-step per warp: "))
}

k_steps <- BK / 32  # for INT8 WMMA_K=32
n_mma_total <- mma_per_tile * k_steps * warps

# Compute cycles: how long to execute all mma ops
# Each mma takes ~32 cycles, 4 mma/SM/cycle → each SM can issue 4 mma/cycle
# But we need to account for occupancy
sm_warps <- warps  # assuming 1 block/SM worst case
sm_mma_per_cycle <- T_PER_SM$mma * (sm_warps / 48)  # scale by warp occupancy
compute_cycles <- n_mma_total * L_CYCLES$mma_sp / max(1, sm_mma_per_cycle)

# Load cycles for A: ldmatrix per warp per k-step
# Each warp does 2 ldmatrix.x2 per k-step (2 tiles * 2 regs)
lds_a_per_kstep <- warps * 2 * 2  # warps * tiles_m * x2 factor
lds_a_total <- lds_a_per_kstep * k_steps
a_load_cycles <- lds_a_total * L_CYCLES$ldmatrix / T_PER_SM$lds

# Load cycles for B: depends on load pattern
# Scalar LDS.U8: 8 per fragment per k-step per warp
# LDS.32: 1 per fragment per k-step per warp
# Each warp does: WARP_TILES_M * WARP_TILES_N * 2 (left/right) * 2 (b0/b1)
b_frag_per_kstep <- mma_per_tile * 4  # 2 sub-tiles * 2 b-regs per mma
b_scalar_loads <- b_frag_per_kstep * 8  # 8 LDS.U8 per fragment
b_scalar_cycles <- b_scalar_loads * k_steps * warps * L_CYCLES$lds_u8 / T_PER_SM$lds

b_lds32_loads <- b_frag_per_kstep * 1   # 1 LDS.32 per fragment
b_lds32_cycles <- b_lds32_loads * k_steps * warps * L_CYCLES$lds_32 / T_PER_SM$lds

# cp.async cycles (global→smem for next tile while computing current)
a_bytes <- BM * (BK / 2)  # compressed A
b_bytes <- BK * BN        # dense B
meta_bytes <- 128 * 4     # metadata
total_load_bytes <- a_bytes + b_bytes + meta_bytes

# cp.async throughput: ~608 GB/s DRAM = ~608e9/1.8e9 = ~338 bytes/cycle per device
# Per SM: 338 / 48 = ~7 bytes/cycle
bytes_per_cycle_sm <- DRAM_BW / (1.8 * SM_COUNT)  # 1.8 GHz
async_cycles <- total_load_bytes / bytes_per_cycle_sm

cat(sprintf("--- Config: BM=%d BN=%d BK=%d warps=%d mma/tile=%d ---\n",
            BM, BN, BK, warps, mma_per_tile))
cat(sprintf("K-steps per tile: %d\n", k_steps))
cat(sprintf("Total MMA ops/block: %d\n", n_mma_total))
cat("\n--- Compute vs Load Balance ---\n")
cat(sprintf("  Compute cycles (mma):      %8.0f\n", compute_cycles))
cat(sprintf("  A load cycles (ldmatrix):  %8.0f\n", a_load_cycles))
cat(sprintf("  B load cycles (scalar):    %8.0f\n", b_scalar_cycles))
cat(sprintf("  B load cycles (LDS.32):    %8.0f\n", b_lds32_cycles))
cat(sprintf("  cp.async cycles (global):  %8.0f\n", async_cycles))

compute_bound <- compute_cycles > async_cycles
cat(sprintf("\n  Kernel is %s-bound\n", if (compute_bound) "COMPUTE" else "MEMORY"))

# CP.async benefit analysis
cat("\n--- cp.async Effectiveness ---\n")
cp_async_benefit <- compute_cycles - async_cycles
cat(sprintf("  Compute - load gap: %.0f cycles\n", cp_async_benefit))
if (cp_async_benefit > 500) {
  cat("  cp.async helps: compute significantly longer than loads\n")
} else if (cp_async_benefit > 0) {
  cat("  cp.async marginally helpful: small compute surplus\n")
} else if (cp_async_benefit > -500) {
  cat("  cp.async neutral: loads nearly match compute\n")
} else {
  cat("  cp.async harmful: loads exceed compute, adding latency\n")
}

# Swizzle overhead estimation
cat("\n--- Swizzle Overhead (for LDSM kernels) ---\n")
swizzle_threads <- warps * 32
swizzle_elems <- BK * BN
swizzle_iters <- ceiling(swizzle_elems / swizzle_threads)
swizzle_cycles <- swizzle_iters * 4 + 2 * L_CYCLES$syncthreads  # store loop + 2 syncs
cat(sprintf("  Transpose elements: %d\n", swizzle_elems))
cat(sprintf("  Per-thread work:    %d stores\n", swizzle_iters))
if (compute_cycles > 0) {
  cat(sprintf("  Estimated cycles:   %.0f (%.1f%% of compute)\n",
              swizzle_cycles, 100 * swizzle_cycles / compute_cycles))
} else {
  cat(sprintf("  Estimated cycles:   %.0f (compute_cycles=0)\n", swizzle_cycles))
}

break_even_m <- if (swizzle_cycles > 0 && compute_cycles > 0) {
  target <- swizzle_cycles / (0.05 * compute_cycles)
  target * BM
} else { Inf }

cat(sprintf("  Break-even problem size: ~%.0fx%.0f (swizzle < 5%% of compute)\n",
            break_even_m, break_even_m * BN / BM))

# Wave analysis
cat("\n--- Wave Efficiency ---\n")
grid_x <- ceiling(4096 / BN)
grid_y <- ceiling(4096 / BM)
total_blocks <- grid_x * grid_y
occ_blocks <- min(2, floor(50 / (BM * BN * BK / 1e6)))  # rough
blocks_per_device <- occ_blocks * 48
waves <- total_blocks / blocks_per_device
cat(sprintf("  Grid %dx%d = %d blocks for 4096x4096\n", grid_x, grid_y, total_blocks))
cat(sprintf("  Concurrent blocks: %d (%d per SM x 48 SMs)\n", blocks_per_device, occ_blocks))
cat(sprintf("  Waves: %.2f (tail waste: %.1f%%)\n", waves, 100 * (1 - waves %% 1)))
