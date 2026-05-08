#!/usr/bin/env Rscript
# config_optimizer.R — Grid-search optimal kernel config for GA104 sm_86
# Given M,N,K and kernel family, search BM/BN/BK/warp configs for best occupancy + roofline.
# Usage: Rscript scripts/config_optimizer.R <M> <N> <K> [kernel_type]
#   kernel_type: dense_gemm | sparse_gemm | flash_attn | conv2d

cat("=== GA104 Kernel Config Optimizer ===\n\n")

# Hardware constants (sm_86)
MAX_REGS_SM <- 65536
MAX_SMEM_SM <- 102400
MAX_WARPS_SM <- 48
MAX_BLOCKS_SM <- 32
MAX_THREADS_SM <- 1536
WARP_SIZE <- 32
SM_COUNT <- 48

FP16_PEAK <- 174.0   # TFLOPS
INT8_PEAK <- 348.0   # TFLOPS
DRAM_BW <- 608.0     # GB/s

# Bank-of helper
bank_of <- function(addr) { bitwAnd(bitwShiftR(as.integer(addr), 2), 31) }

# Occupancy calculator (per config)
occupancy <- function(threads, regs_per_thread, smem_bytes) {
  warps <- ceiling(threads / WARP_SIZE)
  blocks_regs <- floor(MAX_REGS_SM / (regs_per_thread * threads))
  blocks_smem <- floor(MAX_SMEM_SM / smem_bytes)
  blocks_warps <- floor(MAX_WARPS_SM / warps)
  blocks_threads <- floor(MAX_THREADS_SM / threads)
  blocks_hard <- min(MAX_BLOCKS_SM, blocks_threads)

  active <- min(blocks_regs, blocks_smem, blocks_warps, blocks_hard)
  active_warps <- active * warps
  occ_pct <- 100 * active_warps / MAX_WARPS_SM
  bottleneck <- c("regs","smem","warps","threads")[
    which.min(c(blocks_regs, blocks_smem, blocks_warps, blocks_hard))]
  list(active_blocks = active, active_warps = active_warps, occupancy = occ_pct,
       bottleneck = bottleneck, smem_kb = smem_bytes / 1024)
}

# Roofline prediction
dense_gemm_roof <- function(M, N, K, time_ms) {
  flops <- 2.0 * M * N * K
  bytes <- 2.0 * (M*K + K*N + M*N)  # fp16
  oi <- flops / bytes
  peak <- min(FP16_PEAK, DRAM_BW * oi / 1000)
  achieved <- flops / (time_ms / 1000) / 1e12
  list(oi = oi, peak_tf = peak, achieved_tf = achieved, efficiency = achieved / peak)
}

sparse_gemm_roof <- function(M, N, K, time_ms) {
  eff_flops <- 2.0 * M * N * (K / 2)
  bytes <- 1.0 * M * (K/2) + 2.0 * K * N + 4.0 * M * N
  oi <- eff_flops / bytes
  peak <- min(INT8_PEAK, DRAM_BW * oi / 1000)
  achieved <- eff_flops / (time_ms / 1000) / 1e12
  list(oi = oi, peak_tf = peak, achieved_tf = achieved, efficiency = achieved / peak)
}

# Config generator
generate_configs <- function(kernel_type) {
  configs <- data.frame()

  if (kernel_type %in% c("dense_gemm", "sparse_gemm")) {
    for (bm in c(64, 128, 192, 256)) {
      for (bn in c(64, 128, 192, 256)) {
        for (bk in c(32, 64, 128)) {
          for (warps in c(4, 8, 16, 32)) {
            threads <- warps * 32
            if (threads > 1024) next
            # Estimate registers per thread (heuristic: ~128-192 for GEMM)
            regs <- if (kernel_type == "sparse_gemm") 160 else 128
            # Estimate smem
            # A: BM * (BK/2 for sparse or BK for dense) + pad
            # B: BK * BN + pad
            pad <- 16
            if (kernel_type == "sparse_gemm") {
              stride_a <- bk/2 + pad
              smem_a <- bm * stride_a
              smem_b <- bk * (bn + pad)
              smem_meta <- 128 * 4  # 128 uint32
            } else {
              stride_a <- bk + pad
              smem_a <- bm * stride_a
              smem_b <- bk * (bn + pad)
              smem_meta <- 0
            }
            smem_total <- 2 * (smem_a + smem_b + smem_meta)  # double-buffered

            occ <- occupancy(threads, regs, smem_total)
            configs <- rbind(configs, data.frame(
              bm = bm, bn = bn, bk = bk, warps = warps, threads = threads,
              regs = regs, smem_a = smem_a, smem_b = smem_b, smem_total = smem_total / 1024,
              occ_blocks = occ$active_blocks, occ_warps = occ$active_warps,
              occ_pct = occ$occupancy, bottleneck = occ$bottleneck,
              smem_kb = occ$smem_kb
            ))
          }
        }
      }
    }
  } else if (kernel_type == "flash_attn") {
    # Simpler search for flash attention
    for (br in c(32, 64, 128)) {
      for (bc in c(32, 64, 128)) {
        for (warps in c(4, 8, 16)) {
          threads <- warps * 32
          if (threads > 1024) next
          # Flash attn smem: Q tile + K tile + V tile + softmax accum
          smem_total <- 2 * (br * 64 + bc * 64 + br * bc * 4 + br * 4)
          occ <- occupancy(threads, 128, smem_total)
          configs <- rbind(configs, data.frame(
            bm = br, bn = bc, bk = 64, warps = warps, threads = threads,
            regs = 128, smem_a = br * (64 + 16), smem_b = bc * (64 + 16),
            smem_total = smem_total / 1024,
            occ_blocks = occ$active_blocks, occ_warps = occ$active_warps,
            occ_pct = occ$occupancy, bottleneck = occ$bottleneck,
            smem_kb = occ$smem_kb
          ))
        }
      }
    }
  }

  configs
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3) {
  M <- as.numeric(args[1]); N <- as.numeric(args[2]); K <- as.numeric(args[3])
  kernel <- ifelse(length(args) >= 4, args[4], "dense_gemm")
} else {
  M <- 4096; N <- 4096; K <- 4096
  kernel <- "dense_gemm"
  cat("Using defaults: M=4096 N=4096 K=4096 kernel=dense_gemm\n")
  cat("Usage: Rscript scripts/config_optimizer.R <M> <N> <K> [kernel_type]\n\n")
}

cat(sprintf("Searching configs for %s @ %.0fx%.0fx%.0f\n", kernel, M, N, K))
configs <- generate_configs(kernel)
cat(sprintf("Generated %d candidate configs\n\n", nrow(configs)))

# Filter: must fit under cliff and have >1 block/SM
viable <- subset(configs, occ_blocks >= 2 & smem_kb <= 50)
cat(sprintf("Viable configs (>=2 blocks/SM, <=50KB): %d\n", nrow(viable)))

if (nrow(viable) == 0) {
  cat("No viable configs! Relaxing to 1 block/SM...\n")
  viable <- subset(configs, smem_kb <= 100)
}

# Score: prefer high occupancy, avoid regs bottleneck, smaller tiles = more parallelism
viable$parallelism <- (M / viable$bm) * (N / viable$bn)
viable$score <- viable$occ_pct * log(viable$parallelism + 1) -
  ifelse(viable$bottleneck == "regs", 10, 0) -
  ifelse(viable$bottleneck == "smem", 5, 0)

# Top 10
top <- head(viable[order(-viable$score), ], 10)

cat(sprintf("\n%-8s %-8s %-8s %-8s %-8s %-10s %-8s %-10s %-12s %-10s\n",
            "BM", "BN", "BK", "warps", "threads", "smem(KB)", "blocks", "warps/SM", "occ%", "bottleneck"))
cat(strrep("=", 100), "\n", sep = "")

for (i in seq_len(nrow(top))) {
  r <- top[i, ]
  cat(sprintf("%-8d %-8d %-8d %-8d %-8d %-10.1f %-8d %-10d %-11.1f %-10s\n",
              r$bm, r$bn, r$bk, r$warps, r$threads,
              r$smem_kb, r$occ_blocks, r$occ_warps, r$occ_pct, r$bottleneck))
}

# Best config detail
best <- top[1, ]
cat(sprintf("\n★ BEST CONFIG: BM=%d BN=%d BK=%d warps=%d threads=%d\n",
            best$bm, best$bn, best$bk, best$warps, best$threads))
cat(sprintf("  Shared memory: %.1f KB (A=%.1f B=%.1f KB)\n",
            best$smem_kb, best$smem_a/1024, best$smem_b/1024))
cat(sprintf("  Occupancy: %.1f%% (%d blocks/SM, %d warps/SM)\n",
            best$occ_pct, best$occ_blocks, best$occ_warps))
cat(sprintf("  Bottleneck: %s\n", best$bottleneck))

# Grid dimensions
grid_x <- ceiling(N / best$bn)
grid_y <- ceiling(M / best$bm)
cat(sprintf("  Grid: %d x %d = %d blocks (%.1f waves on device)\n",
            grid_x, grid_y, grid_x * grid_y, (grid_x * grid_y) / (best$occ_blocks * SM_COUNT)))

# Suggested next configs to try for tuning ladder
cat("\n--- Suggested tuning ladder ---\n")
for (i in 1:min(5, nrow(top))) {
  r <- top[i, ]
  grid_x <- ceiling(N / r$bn)
  grid_y <- ceiling(M / r$bm)
  total_blocks <- grid_x * grid_y
  waves <- total_blocks / (r$occ_blocks * SM_COUNT)
  cat(sprintf("  Run %d: BM=%d BN=%d BK=%d warps=%d — grid=%dx%d waves=%.2f\n",
              i, r$bm, r$bn, r$bk, r$warps, grid_x, grid_y, waves))
}
