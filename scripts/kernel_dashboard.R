#!/usr/bin/env Rscript
# kernel_dashboard.R — Combined analysis dashboard for a kernel configuration
# Runs occupancy, roofline, pipeline balance, and smem layout analysis in one shot.
# Usage: Rscript scripts/kernel_dashboard.R <M> <N> <K> <BM> <BN> <BK> <warps> <mma_per_tile> [kernel_type]

cat("╔════════════════════════════════════════════════════════════════╗\n")
cat("║         GA104 Kernel Configuration Dashboard                   ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n\n")

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 8) {
  M <- as.numeric(args[1]); N <- as.numeric(args[2]); K <- as.numeric(args[3])
  BM <- as.integer(args[4]); BN <- as.integer(args[5]); BK <- as.integer(args[6])
  warps <- as.integer(args[7]); mma_per_tile <- as.integer(args[8])
  kernel <- ifelse(length(args) >= 9, args[9], "dense_gemm")
} else {
  # Default: current sparse INT8 config
  cat("Using defaults (sparse INT8 GEMM 4096³, BM=128 BN=128 BK=64 warps=16):\n")
  cat("Usage: Rscript scripts/kernel_dashboard.R <M> <N> <K> <BM> <BN> <BK> <warps> <mma_per_tile> [kernel_type]\n\n")
  M <- 4096; N <- 4096; K <- 4096
  BM <- 128; BN <- 128; BK <- 64; warps <- 16; mma_per_tile <- 4
  kernel <- "sparse_gemm"
}

# ======================================================================
# 1. OCCUPANCY
# ======================================================================
cat("━━━ 1. OCCUPANCY ━━━\n")
threads <- warps * 32
# Default register count (actual from cubin: sparse=64, dense=64-96)
# Override by passing reg_count as arg 9
reg_default <- if (kernel == "sparse_gemm") 64 else 96
regs <- ifelse(length(args) >= 10, as.integer(args[10]), reg_default)
pad <- 16
if (kernel == "sparse_gemm") {
  stride_a <- BK/2 + pad
  smem_a <- BM * stride_a
  smem_b <- BK * (BN + pad)
  smem_meta <- 128 * 4
} else {
  stride_a <- BK + pad
  smem_a <- BM * stride_a
  smem_b <- BK * (BN + pad)
  smem_meta <- 0
}
smem_total <- 2 * (smem_a + smem_b + smem_meta)

MAX_REGS <- 65536; MAX_SMEM <- 102400; MAX_WARPS <- 48
MAX_BLOCKS <- 32; MAX_THREADS <- 1536; WARP_SIZE <- 32

warps_pb <- ceiling(threads / WARP_SIZE)
blocks_regs <- floor(MAX_REGS / (regs * threads))
blocks_smem <- floor(MAX_SMEM / smem_total)
blocks_warps <- floor(MAX_WARPS / warps_pb)
blocks_threads <- min(MAX_BLOCKS, floor(MAX_THREADS / threads))

all_blocks <- c(blocks_regs, blocks_smem, blocks_warps, blocks_threads)
active_blocks <- min(all_blocks)
active_warps <- active_blocks * warps_pb
occ_pct <- 100 * active_warps / MAX_WARPS
bottleneck <- c("regs","smem","warps","threads")[which.min(all_blocks)]

cat(sprintf("  Config:  BM=%d BN=%d BK=%d warps=%d threads=%d\n", BM, BN, BK, warps, threads))
cat(sprintf("  Registers/thread: %d → %d regs/block\n", regs, regs * threads))
cat(sprintf("  Shared memory:    %.1f KB (A=%.1f B=%.1f meta=%.1f)\n",
            smem_total/1024, smem_a/1024, smem_b/1024, smem_meta/1024))
cat(sprintf("  Limit from regs:  %d blocks\n", blocks_regs))
cat(sprintf("  Limit from smem:  %d blocks\n", blocks_smem))
cat(sprintf("  Limit from warps: %d blocks\n", blocks_warps))
cat(sprintf("  Limit from threads:%d blocks\n", blocks_threads))
cat(sprintf("  ★ Active:        %d blocks/SM, %d warps/SM (%.1f%% occupancy)\n",
            active_blocks, active_warps, occ_pct))
cat(sprintf("  Bottleneck:      %s %s\n", bottleneck,
            if (smem_total > 51200) "⚠️ OVER 50KB CLIFF" else ""))

# ======================================================================
# 2. ROOFLINE
# ======================================================================
cat("\n━━━ 2. ROOFLINE ━━━\n")
FP32_PEAK <- 21.7; FP16_PEAK <- 174.0; INT8_PEAK <- 348.0
DRAM_BW <- 608.0

if (kernel == "sparse_gemm") {
  flops <- 2.0 * M * N * (K / 2)
  bytes <- 1.0 * M * (K/2) + 2.0 * K * N + 4.0 * M * N
  peak <- INT8_PEAK
} else if (kernel == "dense_gemm") {
  flops <- 2.0 * M * N * K
  bytes <- 2.0 * (M*K + K*N + M*N)
  peak <- FP16_PEAK
} else {
  flops <- 2.0 * M * N * K
  bytes <- 2.0 * (M*K + K*N + M*N)
  peak <- FP32_PEAK
}
oi <- flops / bytes
roof <- min(peak, DRAM_BW * oi / 1000)
cat(sprintf("  Problem: %.0f×%.0f×%.0f\n", M, N, K))
cat(sprintf("  FLOPs:   %.2e\n", flops))
cat(sprintf("  Bytes:   %.2e\n", bytes))
cat(sprintf("  OI:      %.2f FLOP/byte\n", oi))
cat(sprintf("  Roof:    %.1f TFLOPS (%.1f%% of %s peak)\n",
            roof, 100*roof/peak, if (kernel=="sparse_gemm") "INT8" else "FP16"))

# ======================================================================
# 3. PIPELINE BALANCE
# ======================================================================
cat("\n━━━ 3. PIPELINE BALANCE ━━━\n")
k_steps <- BK / 32
n_mma <- mma_per_tile * k_steps * warps

# Simplified: check if compute exceeds async load
a_bytes <- BM * (if (kernel=="sparse_gemm") BK/2 else BK)
b_bytes <- BK * BN
meta_bytes <- if (kernel=="sparse_gemm") 512 else 0
total_bytes <- a_bytes + b_bytes + meta_bytes

# Time per tile (rough)
cycles_mma <- n_mma * (if (kernel=="sparse_gemm") 32 else 16)
bytes_per_cycle <- DRAM_BW / (1.8 * 48)  # ~7 bytes/cycle per SM
cycles_load <- total_bytes / bytes_per_cycle

cat(sprintf("  MMA ops/tile:     %d (%d steps × %d warps × %d/tile)\n",
            n_mma, k_steps, warps, mma_per_tile))
cat(sprintf("  Compute cycles:   ~%.0f\n", cycles_mma))
cat(sprintf("  Load cycles:      ~%.0f (%.1f KB via cp.async)\n",
            cycles_load, total_bytes/1024))

if (cycles_mma > cycles_load * 1.2) {
  cat("  ★ COMPUTE-BOUND: overlapping loads hides latency\n")
} else if (cycles_mma > cycles_load * 0.8) {
  cat("  ★ BALANCED: loads and compute roughly matched\n")
} else {
  cat("  ⚠️ MEMORY-BOUND: loads exceed compute, occupancy critical\n")
}

# ======================================================================
# 4. SMEM LAYOUT (quick bank-conflict check)
# ======================================================================
cat("\n━━━ 4. SMEM BANK CONFLICTS ━━━\n")
bank_of <- function(addr) { bitwAnd(bitwShiftR(as.integer(addr), 2), 31) }

# For column-major reformat layout (the LDSM case)
lane <- 0:31; gid <- lane %/% 4; frag_tid <- lane %% 4

# B access patterns
col_stride <- BK  # =64 for our default
check_pattern <- function(nc, kg) {
  addrs <- nc * col_stride + kg * 4
  banks <- bank_of(addrs)
  tab <- as.integer(table(factor(banks, levels=0:31)))
  conflicts <- sum(pmax(tab - 1, 0))
  max_share <- max(tab)
  list(conflicts=conflicts, max=max_share, aligned=all(addrs %% 4 == 0))
}

p1 <- check_pattern(gid, frag_tid)        # b0 left
p2 <- check_pattern(gid, frag_tid + 4)    # b1 left
p3 <- check_pattern(8 + gid, frag_tid)    # b0 right
p4 <- check_pattern(8 + gid, frag_tid + 4) # b1 right

cat(sprintf("  Layout: col-major, stride=BK=%d\n", col_stride))
cat(sprintf("  b0_left:   conflicts=%d max_share=%d aligned=%s\n",
            p1$conflicts, p1$max, p1$aligned))
cat(sprintf("  b1_left:   conflicts=%d max_share=%d aligned=%s\n",
            p2$conflicts, p2$max, p2$aligned))
cat(sprintf("  b0_right:  conflicts=%d max_share=%d aligned=%s\n",
            p3$conflicts, p3$max, p3$aligned))
cat(sprintf("  b1_right:  conflicts=%d max_share=%d aligned=%s\n",
            p4$conflicts, p4$max, p4$aligned))

if (p1$conflicts + p2$conflicts + p3$conflicts + p4$conflicts == 0) {
  cat("  ★ ZERO CONFLICTS — layout is optimal\n")
} else {
  cat(sprintf("  ⚠️ %d total conflicts — consider padding or swizzle\n",
              p1$conflicts + p2$conflicts + p3$conflicts + p4$conflicts))
}

# ======================================================================
# 5. SUMMARY
# ======================================================================
cat("\n━━━ 5. SUMMARY ━━━\n")
grid_x <- ceiling(N / BN); grid_y <- ceiling(M / BM)
total_blocks <- grid_x * grid_y
dev_blocks <- active_blocks * 48
waves <- total_blocks / dev_blocks

cat(sprintf("  Grid:        %d × %d = %d blocks\n", grid_x, grid_y, total_blocks))
cat(sprintf("  Device cap:  %d concurrent blocks (%d/SM × 48 SMs)\n", dev_blocks, active_blocks))
cat(sprintf("  Waves:       %.2f (tail waste: %.1f%%)\n", waves, 100 * (1 - waves %% 1)))
cat(sprintf("  Occupancy:   %.1f%% (%s-bottlenecked)\n", occ_pct, bottleneck))
cat(sprintf("  Roofline:    %.1f TFLOPS @ OI=%.2f\n", roof, oi))

# Go/no-go
if (occ_pct < 25) {
  cat("  ⚠️  LOW OCCUPANCY — reduce registers or increase parallelism\n")
}
if (smem_total > 51200) {
  cat("  ⚠️  50KB CLIFF — occupancy collapse, restructure smem\n")
}
if (waves < 1.5) {
  cat("  ⚠️  INSUFFICIENT WAVES — grid too small for device\n")
}
if (p1$conflicts + p2$conflicts > 0) {
  cat("  ⚠️  BANK CONFLICTS — add padding to smem layout\n")
}
if (occ_pct >= 25 && smem_total <= 51200 && waves >= 1.5 &&
    p1$conflicts + p2$conflicts == 0) {
  cat("  ✅ CONFIG LOOKS HEALTHY\n")
}

cat("\n")
