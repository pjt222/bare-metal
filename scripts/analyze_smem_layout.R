#!/usr/bin/env Rscript
# analyze_smem_layout.R — Shared memory bank conflict analyzer (v2)
# Now supports: xor-swizzle patterns, multi-pattern stress test, conflict heatmap.
#
# The COPRIME-TO-32 RULE (issue #75):
#   For inline-asm scalar shared loads (LDS.32, LDS.64, LDS.128), bank conflicts
#   are avoided when stride_bytes / 4 is ODD (coprime to 32).
#   Example: BK_STRIDE=68 → 68/4=17 (odd) → conflicts=0.
#
#   This rule does NOT apply to WMMA/ldmatrix because those require
#   16-byte aligned leading dimensions (8-half multiple), which forces
#   stride/4 to be even (multiple of 4). For WMMA, use XOR swizzle instead.
#
# Usage: Rscript scripts/analyze_smem_layout.R [BK] [BN] [pattern]
#   pattern: "col_major" | "row_major" | "swizzle" | "kgroup_n"

cat("=== Shared Memory Bank Conflict Analyzer (sm_86) ===\n\n")

NUM_BANKS <- 32
BANK_WIDTH <- 4L
bank_of <- function(addr) { bitwAnd(bitwShiftR(addr, 2), 31) }

args <- commandArgs(trailingOnly = TRUE)
BK <- if (length(args) >= 1) as.integer(args[1]) else 64
BN <- if (length(args) >= 2) as.integer(args[2]) else 128
pattern <- if (length(args) >= 3) args[3] else "col_major"

lane <- 0:31
gid <- lane %/% 4         # 0..7, which fragment
frag_tid <- lane %% 4     # 0..3, which k_group within fragment

k_group0 <- frag_tid      # b0
k_group1 <- frag_tid + 4  # b1
k_group2 <- frag_tid + 8  # for WMMA_K=32, k_step=32
k_group3 <- frag_tid + 12

conflict_score <- function(addrs) {
  banks <- bank_of(addrs)
  tab <- as.integer(table(factor(banks, levels = 0:31)))
  sum(pmax(tab - 1, 0))  # 0 = ideal
}

max_share <- function(addrs) {
  banks <- bank_of(addrs)
  tab <- as.integer(table(factor(banks, levels = 0:31)))
  max(tab)
}

analyze_pattern <- function(pat, bk, bn, pad = 0) {
  col_stride <- switch(pat,
    "col_major" = bk + pad,
    "row_major" = bn + pad,
    "kgroup_n"  = (bn + pad) * 4,   # k_group * stride + n_col * 4
    "swizzle"   = bk + pad,         # with xor below
    bk + pad
  )

  # Addresses for 4 B-fragment access patterns (per-lane, 32 threads)
  # All 4 patterns occur in COMPUTE_TILE for INT8 sparse GEMM
  make_addrs <- function(nc, kg) {
    if (pat == "kgroup_n") {
      kg * (bn + pad) * 4 + nc * 4
    } else if (pat == "swizzle") {
      # XOR swizzle: row XOR (col >> log2(banks))
      shift <- 3  # log2(8) = 3, since gid 0..7 maps to 8 banks
      row <- kg
      col <- nc
      swiz_row <- bitwXor(row, bitwShiftR(col, shift))
      swiz_row * col_stride + col
    } else if (pat == "row_major") {
      kg * col_stride + nc
    } else {
      # col_major
      nc * col_stride + kg * 4
    }
  }

  n_col_left  <- gid       # b_base_col = 0
  n_col_right <- 8 + gid   # b_base_col + 8

  a1 <- make_addrs(n_col_left,  k_group0)
  a2 <- make_addrs(n_col_left,  k_group1)
  a3 <- make_addrs(n_col_right, k_group0)
  a4 <- make_addrs(n_col_right, k_group1)
  a5 <- make_addrs(n_col_left,  k_group2)
  a6 <- make_addrs(n_col_left,  k_group3)
  a7 <- make_addrs(n_col_right, k_group2)
  a8 <- make_addrs(n_col_right, k_group3)

  all_a <- c(a1, a2, a3, a4, a5, a6, a7, a8)  # 256 addresses = 8 warp-wide accesses

  list(
    pad = pad,
    stride = col_stride,
    conflicts_b0l = conflict_score(a1), max_b0l = max_share(a1),
    conflicts_b1l = conflict_score(a2), max_b1l = max_share(a2),
    conflicts_b0r = conflict_score(a3), max_b0r = max_share(a3),
    conflicts_b1r = conflict_score(a4), max_b1r = max_share(a4),
    conflicts_b2l = conflict_score(a5), max_b2l = max_share(a5),
    conflicts_b3l = conflict_score(a6), max_b3l = max_share(a6),
    conflicts_b2r = conflict_score(a7), max_b2r = max_share(a7),
    conflicts_b3r = conflict_score(a8), max_b3r = max_share(a8),
    total_conflicts = conflict_score(all_a),
    all_aligned = all(all_a %% 4 == 0),
    total_bytes = if (pat == "kgroup_n") (bk / 4) * (bn + pad) * 4 else bn * col_stride
  )
}

cat(sprintf("BK=%d BN=%d pattern=%s\n\n", BK, BN, pattern))

# Padding sweep
cat(sprintf("%-4s | %-8s | %-6s | %s | %s\n",
            "pad", "stride", "conf", "max_sh", "align"))
cat(strrep("-", 60), "\n", sep = "")

best <- list(conflicts = Inf, pad = 0)
for (pad in 0:min(32, BK)) {
  r <- analyze_pattern(pattern, BK, BN, pad)
  flag <- if (r$total_conflicts == 0) "★" else ""
  if (r$total_conflicts < best$conflicts) {
    best <- list(conflicts = r$total_conflicts, pad = pad, stride = r$stride, result = r)
  }
  if (pad %% 4 == 0 || r$total_conflicts == 0) {
    cat(sprintf("%-4d | %-8d | %-6d | %-6d | %-5s %s\n",
                pad, r$stride, r$total_conflicts, max(r$max_b0l, r$max_b1l, r$max_b0r, r$max_b1r),
                if (r$all_aligned) "Y" else "N", flag))
  }
}

cat(sprintf("\n★ BEST: pad=%d, stride=%d, total_conflicts=%d\n",
            best$pad, best$stride, best$conflicts))
r <- best$result
cat(sprintf("  Per-pattern: b0l=%d b1l=%d b0r=%d b1r=%d b2l=%d b3l=%d b2r=%d b3r=%d\n",
            r$conflicts_b0l, r$conflicts_b1l, r$conflicts_b0r, r$conflicts_b1r,
            r$conflicts_b2l, r$conflicts_b3l, r$conflicts_b2r, r$conflicts_b3r))
cat(sprintf("  Buffer size: %.1f KB\n", r$total_bytes / 1024))

# Memory budget check
total_kb <- 2 * r$total_bytes / 1024 + 12 + 1 + 16  # 2x reformat + A + meta + temp
cat(sprintf("  Total smem (with temp buf): %.1f KB %s\n",
            total_kb, if (total_kb <= 50) "✓ under cliff" else paste0("⚠ OVER 50KB CLIFF by ", round(total_kb - 50, 1), "KB")))

# Heatmap: bank share per pattern (ASCII)
cat("\n--- Bank Share Heatmap (best config) ---\n")
for (pat_name in c("b0_left", "b1_left", "b0_right", "b1_right")) {
  cat(sprintf("%-10s: ", pat_name))
  addrs <- switch(pat_name,
    "b0_left"  = { n_col_left  <- gid; kg <- k_group0; if (pattern == "kgroup_n") kg * (BN + best$pad) * 4 + n_col_left * 4 else n_col_left * best$stride + kg * 4 },
    "b1_left"  = { n_col_left  <- gid; kg <- k_group1; if (pattern == "kgroup_n") kg * (BN + best$pad) * 4 + n_col_left * 4 else n_col_left * best$stride + kg * 4 },
    "b0_right" = { n_col_right <- 8 + gid; kg <- k_group0; if (pattern == "kgroup_n") kg * (BN + best$pad) * 4 + n_col_right * 4 else n_col_right * best$stride + kg * 4 },
    "b1_right" = { n_col_right <- 8 + gid; kg <- k_group1; if (pattern == "kgroup_n") kg * (BN + best$pad) * 4 + n_col_right * 4 else n_col_right * best$stride + kg * 4 }
  )
  banks <- bank_of(addrs)
  tab <- as.integer(table(factor(banks, levels = 0:31)))
  for (i in 0:31) {
    v <- tab[i + 1]
    cat(if (v == 0) "." else if (v == 1) "1" else if (v == 2) "2" else "#")
  }
  cat("\n")
}
