#!/usr/bin/env Rscript
# find_optimal_smem_layout.R — Find optimal INT8 B-smear layout for LDS.32
# The key constraint: each thread needs 4 INT8 from 4 consecutive K-rows at
# the same n_col, AND these 4 bytes must be contiguous in shared memory for
# LDS.32 to work.
#
# Layout: smem_b[k_group][n_col][row_in_group]
#         addr = k_group * (BN + pad) * 4 + n_col * 4 + byte
#         LDS.32 at addr gives bytes for this k_group and n_col.
#
# Bank conflict occurs when two threads in a warp hit the same bank.
# Bank = ((addr / 4) %% 32)

cat("=== Optimal B-smem Layout for igemm_sparse_tiled ===\n\n")

NUM_BANKS <- 32
BN <- 128
BK <- 64
KGROUPS <- BK / 4  # 16 groups of 4 K-rows

bank_of <- function(addr) { bitwAnd(bitwShiftR(addr, 2), 31) }

# Lane parameters
lane <- 0:31
gid <- lane %/% 4
tid_frag <- lane %% 4

base_cols <- seq(0, BN-1, by=32)  # each warp handles 32 cols

# Search over n_col_pad (extra bytes per n_col for bank shift)
# stride = (BN + pad) * 4 bytes per k_group
cat("n_pad | stride | max_bank | conflicts | total_KB\n")
best <- list(conflicts=999, pad=0, stride=0)

for (n_pad in 0:32) {
  stride_n <- (BN + n_pad) * 4
  
  all_conflicts <- 0
  max_bank_share <- 0
  
  for (base in base_cols) {
    # b0 access: k_group0 = tid_frag (implicitly, k_step=0)
    #            n_col = base + gid
    k_group0 <- tid_frag  # 0..3
    k_group1 <- tid_frag + 4  # 4..7 for b1 at same k_step
    
    for (kg in list(k_group0, k_group1)) {
      n_col <- base + gid
      addr <- kg * stride_n + n_col * 4
      banks <- bank_of(addr)
      tab <- as.integer(table(factor(banks, levels=0:31)))
      all_conflicts <- all_conflicts + sum(pmax(tab - 1, 0))
      max_bank_share <- max(max_bank_share, max(tab))
    }
    
    # b0/b1 at k_step=32: k_group = 8 + tid_frag and 12 + tid_frag
    for (kg in list(tid_frag + 8, tid_frag + 12)) {
      n_col <- base + gid
      addr <- kg * stride_n + n_col * 4
      banks <- bank_of(addr)
      tab <- as.integer(table(factor(banks, levels=0:31)))
      all_conflicts <- all_conflicts + sum(pmax(tab - 1, 0))
      max_bank_share <- max(max_bank_share, max(tab))
    }
  }
  
  total_kb <- (KGROUPS * stride_n) / 1024
  
  if (all_conflicts < best$conflicts || 
      (all_conflicts == best$conflicts && total_kb < best$total)) {
    best <- list(conflicts=all_conflicts, pad=n_pad, stride=stride_n, 
                 max_share=max_bank_share, total=total_kb)
  }
  
  if (all_conflicts <= 16 || n_pad %% 4 == 0) {
    cat(sprintf(" %-4d | %-6d | %-8d | %-9d | %-8.1f\n",
                n_pad, stride_n, max_bank_share, all_conflicts, total_kb))
  }
}

cat(sprintf("\n=== BEST: n_pad=%d, stride=%d bytes/k_group ===\n",
            best$pad, best$stride))
cat(sprintf("  Total smem per buffer: %.1f KB (%d bytes)\n",
            best$total, KGROUPS * best$stride))
cat(sprintf("  2x buffered + A + meta = %.0f KB (cliff: 50 KB)\n",
            2 * best$total + 12 + 1))
cat(sprintf("  Conflicts per 4-pattern: %d\n", best$conflicts))

# Verify addresses are 4-byte aligned
cat("\n=== Alignment verification ===\n")
stride <- best$stride
for (base in c(0, 32, 64, 96)) {
  for (kg in c(0, 4, 8, 12)) {
    addr <- kg * stride + (base + gid) * 4
    cat(sprintf("base=%d kg=%d: all_aligned=%s\n",
                base, kg, all(addr %% 4 == 0)))
  }
}
