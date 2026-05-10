#!/usr/bin/env Rscript
# occupancy_calc.R — GA104 sm_86 occupancy calculator
# Given block params, compute theoretical occupancy and identify bottlenecks.
# Usage: Rscript scripts/model/occupancy_calc.R <threads_per_block> <regs_per_thread> <smem_per_block_kb>
#        or interactive: Rscript scripts/model/occupancy_calc.R

cat("=== GA104 (sm_86) Occupancy Calculator ===\n\n")

# GA104 hardware limits (Ampere)
SM_COUNT <- 48
MAX_REGS_SM <- 65536
MAX_SMEM_SM <- 102400  # 100 KB
MAX_WARPS_SM <- 48
MAX_BLOCKS_SM <- 32
WARP_SIZE <- 32
MAX_THREADS_SM <- 1536
MAX_THREADS_BLOCK <- 1024

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3) {
  threads <- as.integer(args[1])
  regs <- as.integer(args[2])
  smem_kb <- as.numeric(args[3])
} else {
  cat("Interactive mode (enter values):\n")
  threads <- as.integer(readline("Threads per block: "))
  regs <- as.integer(readline("Registers per thread: "))
  smem_kb <- as.numeric(readline("Shared memory per block (KB): "))
}

smem <- smem_kb * 1024
warps_per_block <- ceiling(threads / WARP_SIZE)

# Limit from registers
blocks_regs <- floor(MAX_REGS_SM / (regs * threads))
# Limit from shared memory
blocks_smem <- floor(MAX_SMEM_SM / smem)
# Limit from warps
blocks_warps <- floor(MAX_WARPS_SM / warps_per_block)
# Limit from threads
blocks_threads <- floor(MAX_THREADS_SM / threads)
# Hard block limit
blocks_hard <- min(MAX_BLOCKS_SM, blocks_threads)

blocks_from_regs <- max(0, blocks_regs)
blocks_from_smem <- max(0, blocks_smem)
blocks_from_warps <- max(0, blocks_warps)
blocks_from_threads <- max(0, blocks_threads)

active_blocks <- min(blocks_from_regs, blocks_from_smem, blocks_from_warps, blocks_hard)
if (active_blocks <= 0) {
  cat("\n⚠️  CONFIGURATION IMPOSSIBLE — exceeds hardware limits\n")
  cat(sprintf("  Registers allow: %d blocks\n", blocks_from_regs))
  cat(sprintf("  Shared mem allows: %d blocks\n", blocks_from_smem))
  cat(sprintf("  Warps allow: %d blocks\n", blocks_from_warps))
  active_blocks <- 0; active_warps <- 0; occupancy <- 0
  bottleneck <- "IMPOSSIBLE"
} else {
  active_warps <- active_blocks * warps_per_block
  occupancy <- 100 * active_warps / MAX_WARPS_SM
  bottleneck <- c("regs","smem","warps","threads")[
    which.min(c(blocks_from_regs, blocks_from_smem, blocks_from_warps, blocks_hard))]
}

cat(sprintf("Threads/block:  %d (%d warps)\n", threads, warps_per_block))
cat(sprintf("Regs/thread:    %d (total %d per block)\n", regs, regs * threads))
cat(sprintf("Smem/block:     %.0f bytes (%.1f KB)\n", smem, smem_kb))
cat("\n--- Per-SM Limits ---\n")
cat(sprintf("  Max blocks/SM:    %d\n", MAX_BLOCKS_SM))
cat(sprintf("  Limit from regs:  %d blocks (%d warps)\n", blocks_from_regs, blocks_from_regs * warps_per_block))
cat(sprintf("  Limit from smem:  %d blocks (%d warps)\n", blocks_from_smem, blocks_from_smem * warps_per_block))
cat(sprintf("  Limit from warps: %d blocks (%d warps)\n", blocks_from_warps, blocks_from_warps * warps_per_block))
cat(sprintf("  Limit from threads:%d blocks (%d warps)\n", blocks_from_threads, blocks_from_threads * warps_per_block))
cat(sprintf("\nActive blocks/SM: %d\n", active_blocks))
cat(sprintf("Active warps/SM:  %d / %d\n", active_warps, MAX_WARPS_SM))
cat(sprintf("Occupancy:        %.1f%%\n", occupancy))
cat(sprintf("Bottleneck:       %s\n", bottleneck))

# Occupancy vs smem cliff
cliff <- if (smem_kb > 50) {
  cat("\n⚠️  50 KB CLIFF CROSSED — only 1 block/SM possible\n")
  ceiling(50)
} else {
  floor(50)
}

cat(sprintf("\n--- Full device (%d SMs) ---\n", SM_COUNT))
cat(sprintf("  Concurrent blocks: %d\n", active_blocks * SM_COUNT))
cat(sprintf("  Concurrent warps:  %d\n", active_warps * SM_COUNT))
cat(sprintf("  Concurrent threads:%d\n", active_blocks * threads * SM_COUNT))
