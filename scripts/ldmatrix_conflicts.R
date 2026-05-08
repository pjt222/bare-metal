#!/usr/bin/env Rscript
# ldmatrix_conflicts.R — Generic ldmatrix.x4 bank conflict calculator + Flash Attention smem helper
#
# Source from other scripts:
#   source("scripts/ldmatrix_conflicts.R")
#
# CLI usage:
#   Rscript scripts/ldmatrix_conflicts.R <row_elems> <elem_bytes> [num_rows]
#   Rscript scripts/ldmatrix_conflicts.R 64 2 8        # FP16 [*][64], 8-row ldmatrix
#   Rscript scripts/ldmatrix_conflicts.R 64 4 8        # FP32 [*][64] (overlay analysis)
#   Rscript scripts/ldmatrix_conflicts.R --flash-attn  # Flash Attention preset analysis
#
# === Background ===
# ldmatrix.sync.aligned.m8n8.x4.shared.b16 issues 8 row addresses (one per
# thread group). For zero bank conflicts in a 32-bank * 4-byte memory:
#
#   condition: gcd(stride_bytes / 4, 32) <= 4
#   equivalent: stride_bytes mod 32 != 0  (when stride is multiple of 4)
#
# Common safe pads (FP16, base stride 64 halfs = 128 bytes):
#   +1 half  (130 B): mod 32 = 2  ✓
#   +2 halfs (132 B): mod 32 = 4  ✓
#   +4 halfs (136 B): mod 32 = 8  ✓
#   +8 halfs (144 B): mod 32 = 16 ✓  ← phase2 hgemm choice (also 16-byte aligned)
#   +16 halfs (160 B): mod 32 = 0 ✗ (back to conflict)
#
# WMMA load_matrix_sync requires 16-byte aligned leading dimension for FP16:
#   stride_halfs % 8 == 0  →  pad_halfs % 8 == 0  →  pad ∈ {8, 16, 24, ...}
#   Of these, only +8 halfs (144 B) avoids conflicts.

NUM_BANKS <- 32L
BANK_WIDTH <- 4L

# ---- Core: bank id of a byte address ----
bank_of <- function(addr_bytes) {
  bitwAnd(bitwShiftR(as.integer(addr_bytes), 2), NUM_BANKS - 1L)
}

# ---- ldmatrix.x4 conflict analysis ----
# Returns list: distinct_banks (out of num_rows), period, conflict_factor (max replay), ok (bool).
ldmatrix_x4_conflict <- function(stride_bytes, num_rows = 8L) {
  if (stride_bytes %% BANK_WIDTH != 0) {
    stop(sprintf("stride_bytes=%d not multiple of %d (LDS alignment)", stride_bytes, BANK_WIDTH))
  }
  q <- stride_bytes %/% BANK_WIDTH
  # gcd via Euclid
  gcd <- function(a, b) { while (b != 0L) { t <- b; b <- a %% b; a <- t }; a }
  g <- gcd(q, NUM_BANKS)
  period <- NUM_BANKS %/% g
  banks <- ((0L:(num_rows - 1L)) * q) %% NUM_BANKS
  distinct <- length(unique(banks))
  conflict_factor <- num_rows %/% distinct + (if (num_rows %% distinct != 0) 1L else 0L)
  list(
    stride_bytes = stride_bytes,
    q = q,
    gcd = g,
    period = period,
    num_rows = num_rows,
    distinct_banks = distinct,
    conflict_factor = conflict_factor,
    ok = (conflict_factor == 1L),
    banks = banks
  )
}

# ---- Find minimal padding (in elements) to clear conflicts ----
# row_elems: payload elements per row (e.g. 64 halfs)
# elem_bytes: bytes per element (2 for FP16, 4 for FP32)
# num_rows: rows accessed by ldmatrix (default 8 for x4)
# wmma_aligned: if TRUE, restrict pad such that final stride is 16-byte aligned
find_min_pad <- function(row_elems, elem_bytes, num_rows = 8L,
                         wmma_aligned = TRUE, max_pad = 32L) {
  for (pad in 0L:max_pad) {
    stride_bytes <- (row_elems + pad) * elem_bytes
    if (wmma_aligned && (stride_bytes %% 16L != 0L)) next
    r <- ldmatrix_x4_conflict(stride_bytes, num_rows)
    if (r$ok) return(list(pad = pad, stride_elems = row_elems + pad,
                          stride_bytes = stride_bytes, analysis = r))
  }
  NULL
}

# ---- Flash Attention smem budget ----
# Returns total smem bytes, breakdown, occupancy estimates.
flash_attn_smem <- function(Bc = 64L, D_HEAD = 64L, Br_BLOCK = 64L,
                            kv_pad_halfs = 0L, score_pad_floats = 0L,
                            reg_pv = TRUE, max_smem_per_sm = 102400L,
                            cliff_kb = 50L) {
  stride_kv_halfs   <- D_HEAD + kv_pad_halfs
  stride_score_flts <- Bc + score_pad_floats

  bytes_K <- Bc * stride_kv_halfs * 2L
  bytes_V <- Bc * stride_kv_halfs * 2L
  bytes_W <- Br_BLOCK * stride_score_flts * 4L
  bytes_PV <- if (reg_pv) 0L else Br_BLOCK * D_HEAD * 4L

  total <- bytes_K + bytes_V + bytes_W + bytes_PV
  total_kb <- total / 1024.0

  blocks_per_sm_smem  <- max_smem_per_sm %/% total
  under_cliff         <- total_kb <= cliff_kb
  blocks_at_cliff     <- if (under_cliff) 2L else 1L

  list(
    stride_kv_halfs    = stride_kv_halfs,
    stride_score_flts  = stride_score_flts,
    bytes_K = bytes_K, bytes_V = bytes_V,
    bytes_W = bytes_W, bytes_PV = bytes_PV,
    total = total, total_kb = total_kb,
    blocks_per_sm_smem = blocks_per_sm_smem,
    under_50kb_cliff = under_cliff,
    blocks_at_cliff = blocks_at_cliff,
    kv_conflict     = ldmatrix_x4_conflict(stride_kv_halfs * 2L),
    score_conflict_fp16 = ldmatrix_x4_conflict(stride_score_flts * 4L)  # FP16 overlay reads
  )
}

# ---- Padding-vs-occupancy tradeoff predictor ----
# Empirical model calibrated to flash_attn_br16_regpv_pad results (2026-05-07).
#
# Two competing effects when padding flips occupancy down:
#   (a) ldmatrix conflicts vanish: replay overhead removed
#   (b) fewer warps/SM: scheduler can hide less latency
#
# Key empirical finding: at ≥ 8 warps/SM, bank-conflict replays are largely
# hidden by warp interleaving (effective overhead ~5-15% of nominal).
# Below 8 warps, conflicts become exposed and replay multiplier matters.
#
# Occupancy throughput follows ~sqrt(warps/4) up to scheduler saturation
# at ~12 warps. Beyond 12, returns are flat.
pad_tradeoff <- function(conflict_factor, ldsm_per_iter, hmma_per_iter,
                         warps_no_pad, warps_pad,
                         ldsm_lat = 10, hmma_lat = 8) {
  # Effective conflict overhead: hidden when warps >= 8
  effective_overhead <- function(replay_k, warps) {
    if (warps >= 12) 0.05 * (replay_k - 1)      # almost fully hidden
    else if (warps >= 8) 0.15 * (replay_k - 1)  # mostly hidden
    else if (warps >= 4) 0.50 * (replay_k - 1)  # partially exposed
    else replay_k - 1                            # fully exposed
  }

  # Throughput factor from warp count.
  # Empirically, dropping 12→8 warps for HMMA-dense FA ≈ 0.75x throughput
  # (calibrated against bench_br16_regpv_pad: kv8+w4 measured 0.81x relative).
  occ_throughput <- function(warps) {
    if (warps >= 12) 1.0
    else if (warps >= 8) 0.70 + 0.075 * (warps - 8)   # 8→0.70, 12→1.0
    else if (warps >= 4) 0.40 + 0.075 * (warps - 4)
    else 0.25 * warps / 4
  }

  base_cycles    <- ldsm_per_iter * ldsm_lat + hmma_per_iter * hmma_lat
  ldsm_overhead_nopad <- ldsm_per_iter * ldsm_lat * effective_overhead(conflict_factor, warps_no_pad)
  ldsm_overhead_pad   <- ldsm_per_iter * ldsm_lat * effective_overhead(1, warps_pad)

  total_pad   <- (base_cycles + ldsm_overhead_pad)   / occ_throughput(warps_pad)
  total_nopad <- (base_cycles + ldsm_overhead_nopad) / occ_throughput(warps_no_pad)

  list(
    cycles_per_iter_pad   = total_pad,
    cycles_per_iter_nopad = total_nopad,
    speedup_pad_over_nopad = total_nopad / total_pad,
    pad_wins = total_pad < total_nopad,
    overhead_pad   = ldsm_overhead_pad,
    overhead_nopad = ldsm_overhead_nopad,
    occ_pad   = occ_throughput(warps_pad),
    occ_nopad = occ_throughput(warps_no_pad)
  )
}

# ---- Pretty-print helpers ----
print_conflict <- function(r, label = "") {
  cat(sprintf("  %-30s stride=%d B (q=%d, gcd=%d, period=%d) → %d/%d distinct banks, %dx replay %s\n",
              label, r$stride_bytes, r$q, r$gcd, r$period,
              r$distinct_banks, r$num_rows, r$conflict_factor,
              if (r$ok) "✓" else "✗"))
}

print_flash_attn <- function(b, label = "") {
  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  K/V tile stride:   %d halfs (%d bytes)\n", b$stride_kv_halfs, b$stride_kv_halfs * 2L))
  cat(sprintf("  smem_work stride:  %d floats (%d bytes)\n", b$stride_score_flts, b$stride_score_flts * 4L))
  cat(sprintf("  Bytes: K=%d V=%d W=%d PV=%d → total=%d B (%.2f KB)\n",
              b$bytes_K, b$bytes_V, b$bytes_W, b$bytes_PV, b$total, b$total_kb))
  cat(sprintf("  Cliff (50 KB): %s\n", if (b$under_50kb_cliff) "✓ under" else "✗ OVER"))
  cat(sprintf("  Blocks/SM (smem-limited): %d   (cliff-limited: %d)\n",
              b$blocks_per_sm_smem, b$blocks_at_cliff))
  print_conflict(b$kv_conflict,           "K/V ldmatrix.x4 (FP16):")
  print_conflict(b$score_conflict_fp16,   "weight overlay ldmatrix (FP16):")
}

# ===================================================================
# CLI entry point
# ===================================================================
if (!interactive() && sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) >= 1 && args[1] == "--flash-attn") {
    cat("=== Flash Attention smem layout sweep (Bc=64, D_HEAD=64, Br_BLOCK=64, regpv) ===\n")

    cat("\n-- Baseline (no padding) --\n")
    print_flash_attn(flash_attn_smem(kv_pad_halfs = 0L, score_pad_floats = 0L),
                     "kv_pad=0 halfs, score_pad=0 floats")

    cat("\n-- Padded (recommended) --\n")
    print_flash_attn(flash_attn_smem(kv_pad_halfs = 8L, score_pad_floats = 1L),
                     "kv_pad=8 halfs, score_pad=1 float")

    cat("\n-- Padding-vs-occupancy tradeoff (FA regpv, seq=1024, batch*heads=64) --\n")
    cat("  ldsm_per_iter=48 (16 K + 16 W + 16 V), hmma_per_iter=32, conflict=8x\n")
    t1 <- pad_tradeoff(conflict_factor = 8, ldsm_per_iter = 48, hmma_per_iter = 32,
                       warps_no_pad = 12, warps_pad = 8)
    cat(sprintf("    no-pad cycles/iter ≈ %.0f (12 warps, 8x replay)\n", t1$cycles_per_iter_nopad))
    cat(sprintf("    pad    cycles/iter ≈ %.0f (8 warps, no replay)\n", t1$cycles_per_iter_pad))
    cat(sprintf("    predicted speedup from padding: %.2fx %s\n",
                t1$speedup_pad_over_nopad,
                if (t1$pad_wins) "(pad wins)" else "(pad LOSES — occupancy drop dominates)"))
    cat("  → empirical: pad LOSES 20-32% (validated 2026-05-07, bench_br16_regpv_pad)\n")

    cat("\n-- Pad search: minimal K/V pad (WMMA-aligned) --\n")
    p1 <- find_min_pad(64L, 2L, num_rows = 8L, wmma_aligned = TRUE)
    cat(sprintf("  Min WMMA-aligned pad for FP16 stride 64: +%d halfs → stride %d halfs (%d B)\n",
                p1$pad, p1$stride_elems, p1$stride_bytes))

    cat("\n-- Pad search: minimal smem_work pad (FP32, FP16 overlay) --\n")
    cat("  (FP32 store_matrix_sync needs no alignment; FP16 ldmatrix overlay does)\n")
    for (sp in 0:8) {
      stride_b <- (64L + sp) * 4L
      r <- ldmatrix_x4_conflict(stride_b)
      cat(sprintf("    score_pad=+%d floats (stride=%d B): %s (%dx replay)\n",
                  sp, stride_b, if (r$ok) "✓" else "✗", r$conflict_factor))
    }
    quit("no")
  }

  if (length(args) < 2) {
    cat("Usage:\n")
    cat("  Rscript scripts/ldmatrix_conflicts.R <row_elems> <elem_bytes> [num_rows]\n")
    cat("  Rscript scripts/ldmatrix_conflicts.R --flash-attn\n")
    quit("no", status = 1)
  }

  row_elems  <- as.integer(args[1])
  elem_bytes <- as.integer(args[2])
  num_rows   <- if (length(args) >= 3) as.integer(args[3]) else 8L

  cat(sprintf("=== ldmatrix.x4 conflict sweep: row_elems=%d, elem_bytes=%d, num_rows=%d ===\n\n",
              row_elems, elem_bytes, num_rows))
  cat(sprintf("%-6s | %-12s | %-12s | %-7s | %-8s | status\n",
              "pad", "stride_elems", "stride_bytes", "distinct", "replay"))
  cat(strrep("-", 70), "\n", sep = "")

  best_pad <- NA_integer_
  for (pad in 0L:32L) {
    stride_bytes <- (row_elems + pad) * elem_bytes
    if (stride_bytes %% BANK_WIDTH != 0L) next
    r <- ldmatrix_x4_conflict(stride_bytes, num_rows)
    flag <- if (r$ok) "✓" else "✗"
    if (r$ok && is.na(best_pad)) best_pad <- pad
    cat(sprintf("%-6d | %-12d | %-12d | %-7d | %-8d | %s\n",
                pad, row_elems + pad, stride_bytes, r$distinct_banks, r$conflict_factor, flag))
  }

  if (!is.na(best_pad)) {
    cat(sprintf("\n★ Minimal conflict-free pad: +%d elements (stride=%d elems, %d B)\n",
                best_pad, row_elems + best_pad, (row_elems + best_pad) * elem_bytes))
  } else {
    cat("\n✗ No conflict-free pad found in [0, 32]\n")
  }
}
