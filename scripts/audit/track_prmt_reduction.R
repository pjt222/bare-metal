#!/usr/bin/env Rscript
# track_prmt_reduction.R — Track PRMT reduction progress for issue #66
#
# Run after building cubin:
#   Rscript scripts/audit/track_prmt_reduction.R phase2/igemm/igemm_sparse_tiled.sm_86.cubin

cat("=== PRMT Reduction Tracker (Issue #66) ===\n\n")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("Usage: Rscript track_prmt_reduction.R <cubin_path>\n")
  quit(status = 1)
}

cubin <- args[1]
# Extract PRMT count using cuobjdump
prmt_cmd <- paste("cuobjdump -sass", cubin, "| grep -c PRMT")
prmt_count <- as.integer(system(prmt_cmd, intern = TRUE, ignore.stderr = TRUE))

# Extract LDS.U8 count
lds_cmd <- paste("cuobjdump -sass", cubin, "| grep -c 'LDS.U8\\|LDS.U.8'")
lds_count <- as.integer(system(lds_cmd, intern = TRUE, ignore.stderr = TRUE))

# Extract LDS.32 count
lds32_cmd <- paste("cuobjdump -sass", cubin, "| grep -c 'LDS.32\\|LDS.U.32'")
lds32_count <- as.integer(system(lds32_cmd, intern = TRUE, ignore.stderr = TRUE))

# Extract HMMA/IMMA count
mma_cmd <- paste("cuobjdump -sass", cubin, "| grep -c 'MMA\\|IMMA'")
mma_count <- as.integer(system(mma_cmd, intern = TRUE, ignore.stderr = TRUE))

cat(sprintf("Kernel: %s\n", basename(cubin)))
cat(sprintf("  PRMT:     %d\n", prmt_count))
cat(sprintf("  LDS.U8:   %d\n", lds_count))
cat(sprintf("  LDS.32:   %d\n", lds32_count))
cat(sprintf("  MMA/IMMA: %d\n", mma_count))

# Baseline and targets
baseline_prmt <- 160
target_prmt <- 112  # 30% reduction

cat("\n--- Progress ---\n")
cat(sprintf("  Baseline PRMT:  %d\n", baseline_prmt))
cat(sprintf("  Current PRMT:   %d\n", prmt_count))
cat(sprintf("  Reduction:      %.1f%% (target: ≥30%%)\n",
            100 * (baseline_prmt - prmt_count) / baseline_prmt))
cat(sprintf("  Status:         %s\n",
            ifelse(prmt_count <= target_prmt, "TARGET MET ✓", "NEED MORE REDUCTION")))

# Estimate: each B fragment eliminated saves 1 PRMT for LDS.32 alignment
# Currently: 8 LDS.U8 per fragment → 1 LDS.32
# Savings per fragment pair (b0+b1 for left+right): ~4 PRMT fragments
# Total fragments per tile per warp: WARP_TILES_M * WARP_TILES_N * 2 (left/right) * 2 (b0/b1) = 16 per k_step
# Per full BK tile: BK/WMMA_K = 2 k_steps, so 32 fragments total
# 32 * 4 = 128 LDS.U8 currently → would become 32 LDS.32
# PRMT saved: ~128

cat("\n--- Theoretical ceiling ---\n")
cat(sprintf("  If all scalar B loads → LDS.32: PRMT → ~%d\n", baseline_prmt - 128))
cat(sprintf("  If using real ldmatrix (not inline asm): PRMT → ~%d\n", baseline_prmt - 160))
