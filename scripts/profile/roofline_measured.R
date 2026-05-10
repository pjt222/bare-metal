#!/usr/bin/env Rscript
# roofline_measured.R - Per-kernel measured roofline for #92.
#
# Reads NCU sweep output (results/ncu/all.csv) which now includes
# byte and duration counters, computes achieved TFLOPS and measured
# operational intensity, and plots them against DRAM, L2, and
# precision-specific compute ceilings.
#
# Usage:
#   Rscript scripts/profile/roofline_measured.R
#   Rscript scripts/profile/roofline_measured.R --in results/ncu/all.csv \
#                                       --out docs/figures/roofline_measured.png
#
# All inputs are NCU-measured. The previous roofline figure (figures/roofline.png)
# was based on theoretical operational intensity from algebraic estimates;
# the measured version reflects what the L2/DRAM hierarchy actually delivered.

library(ggplot2)
library(scales)

# Project-wide theme + viridis palettes (audit follow-up).
for (p in c("scripts/audit/_theme.R",
            "../audit/_theme.R",
            "../../scripts/audit/_theme.R")) {
    if (file.exists(p)) { source(p); break }
}

# ----------------------------------------------------------------------
# GA104 hardware ceilings (RTX 3070 Ti Laptop, sm_86)
# ----------------------------------------------------------------------
DRAM_PEAK_GBs   <- 608          # 608 GB/s
L2_PEAK_GBs     <- 3000         # ~3 TB/s effective on GA104 (rough)
FP16_TC_PEAK    <- 174e3        # 174 TFLOPS as GFLOPS
FP32_PEAK       <-  21.7e3      # 21.7 TFLOPS
INT8_TC_PEAK    <- 348e3        # 348 TOPS as GOPS

OPS_PER_TENSOR_INST <- 4096     # 16x8x16 muladds * 2 ops = 4096

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(
    in_path  = "results/ncu/all.csv",
    out_path = "docs/figures/roofline_measured.png",
    md_out   = "docs/roofline_measured.md"
  )
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--in")     { out$in_path  <- argv[i+1]; i <- i + 2 }
    else if (a == "--out")    { out$out_path <- argv[i+1]; i <- i + 2 }
    else if (a == "--md-out") { out$md_out   <- argv[i+1]; i <- i + 2 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: roofline_measured.R [--in PATH] [--out PATH] [--md-out PATH]\n")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  out
}

# ----------------------------------------------------------------------
# Compute derived metrics from the NCU sweep CSV.
# ----------------------------------------------------------------------
compute_metrics <- function(d) {
  d$dram_bytes  <- d$dram_read_bytes + d$dram_write_bytes
  d$duration_s  <- d$duration_ns * 1e-9

  # Total FLOPS / OPS achieved during the kernel.
  # tensor_count covers both HMMA and IMMA (NCU counts in same bucket).
  d$tensor_ops  <- d$tensor_count * OPS_PER_TENSOR_INST

  # Achieved throughput as GFLOPS (or GOPS for INT8 kernels).
  d$gflops_achieved <- d$tensor_ops / d$duration_s / 1e9

  # Operational intensity: ops per DRAM byte (and per L2 byte).
  d$oi_dram <- d$tensor_ops / d$dram_bytes
  d$oi_l2   <- d$tensor_ops / d$l2_bytes

  # Precision tag: HMMA-dominant -> FP16, no HMMA but tensor -> INT8/IMMA.
  d$precision <- ifelse(d$hmma_count > 0.9 * d$tensor_count, "FP16",
                  ifelse(d$tensor_count > 0,                 "INT8",
                                                              "FP32"))
  d
}

# ----------------------------------------------------------------------
# Build the roofline plot.
# ----------------------------------------------------------------------
make_roofline <- function(d, out_path) {
  # OI range for ceiling lines.
  oi_grid <- 10 ^ seq(-1, 4, length.out = 200)

  ceilings <- rbind(
    data.frame(oi   = oi_grid,
               gf   = pmin(DRAM_PEAK_GBs * oi_grid, FP16_TC_PEAK),
               tier = "DRAM (608 GB/s) -> FP16 TC peak",
               stringsAsFactors = FALSE),
    data.frame(oi   = oi_grid,
               gf   = pmin(L2_PEAK_GBs   * oi_grid, FP16_TC_PEAK),
               tier = "L2 (~3 TB/s)   -> FP16 TC peak",
               stringsAsFactors = FALSE),
    data.frame(oi   = oi_grid,
               gf   = pmin(DRAM_PEAK_GBs * oi_grid, INT8_TC_PEAK),
               tier = "DRAM (608 GB/s) -> INT8 TC peak",
               stringsAsFactors = FALSE)
  )

  # Three ceilings + three precisions - viridis_d at fixed positions so
  # ceilings and kernels share the same colour family.
  vir5 <- viridisLite::viridis(5, begin = 0.05, end = 0.95)
  pal <- c(
    "DRAM (608 GB/s) -> FP16 TC peak" = vir5[1],   # deep purple
    "L2 (~3 TB/s)   -> FP16 TC peak"  = vir5[3],   # teal
    "DRAM (608 GB/s) -> INT8 TC peak" = vir5[5]    # yellow
  )

  pal_kernels <- c(FP16 = vir5[1], INT8 = vir5[5], FP32 = vir5[3])

  # Single combined plot showing both DRAM-bounded and L2-bounded.
  g <- ggplot() +
    geom_line(data = ceilings,
              aes(x = oi, y = gf, color = tier, linetype = tier),
              linewidth = 0.7) +
    geom_point(data = d,
               aes(x = oi_dram, y = gflops_achieved, fill = precision),
               shape = 21, size = 3.2, stroke = 0.4, color = "black") +
    geom_text(data = d,
              aes(x = oi_dram, y = gflops_achieved, label = label),
              hjust = -0.08, vjust = 0.4, size = 2.6) +
    scale_x_log10(limits = c(0.5, 5000),
                  breaks = c(1, 10, 100, 1000),
                  labels = label_comma()) +
    scale_y_log10(limits = c(50, 350000),
                  breaks = c(100, 1000, 10000, 100000),
                  labels = label_comma()) +
    scale_color_manual(values = pal, name = "Ceiling") +
    scale_linetype_manual(values = c(
      "DRAM (608 GB/s) -> FP16 TC peak" = "solid",
      "L2 (~3 TB/s)   -> FP16 TC peak"  = "dashed",
      "DRAM (608 GB/s) -> INT8 TC peak" = "dotted"
    ), name = "Ceiling") +
    scale_fill_manual(values = pal_kernels, name = "Kernel precision") +
    annotation_logticks(sides = "lb", short = unit(0.05, "cm"),
                        mid = unit(0.10, "cm"), long = unit(0.15, "cm")) +
    labs(
      title    = "Measured roofline (RTX 3070 Ti, GA104)",
      subtitle = sprintf("%d kernels, NCU-measured DRAM bytes + tensor-pipe ops",
                          nrow(d)),
      x = "Operational intensity (ops / DRAM byte)",
      y = "Achieved throughput (GFLOPS for FP16, GOPS for INT8)",
      caption = paste(
        "DRAM bytes = dram__bytes_{read,write}.sum",
        "Tensor ops = smsp__inst_executed_pipe_tensor.sum * 4096",
        "Each point: one kernel from the NCU sweep (results/ncu/all.csv)",
        sep = "  -  "
      )
    ) +
    theme_baremetal(base_size = 10) +
    theme(
      legend.position  = "bottom",
      legend.direction = "vertical",
      legend.box       = "horizontal"
    ) +
    guides(color    = guide_legend(order = 1),
           linetype = guide_legend(order = 1),
           fill     = guide_legend(order = 2))

  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  bm_save(g, out_path, width = 10.5, height = 8)
}

# ----------------------------------------------------------------------
# Markdown summary table.
# ----------------------------------------------------------------------
write_markdown <- function(d, md_path) {
  dir.create(dirname(md_path), showWarnings = FALSE, recursive = TRUE)
  con <- file(md_path, "w"); on.exit(close(con), add = TRUE)

  writeLines(c(
    "# Measured Roofline",
    "",
    "> Detail view. Canonical entry point for all per-kernel comparisons:",
    "> [`docs/kernels.md`](kernels.md).",
    "",
    "Auto-generated by `scripts/profile/roofline_measured.R`.",
    "",
    "Operational intensity is **measured**, not estimated:",
    "",
    "  OI_DRAM = (tensor_count * 4096) / (dram_read_bytes + dram_write_bytes)",
    "",
    "Tensor ops = HMMA (FP16) + IMMA (INT8) instructions, each contributing",
    "16x8x16 muladds = 4096 ops per warp instruction.",
    "",
    "![Measured roofline](figures/roofline_measured.png)",
    "",
    "## Per-kernel data",
    "",
    "| kernel | precision | OI_DRAM | OI_L2 | achieved GFLOPS | DRAM (MB/launch) | duration (us) |",
    "|---|---|---:|---:|---:|---:|---:|"
  ), con)

  d_sorted <- d[order(-d$gflops_achieved), , drop = FALSE]
  for (i in seq_len(nrow(d_sorted))) {
    r <- d_sorted[i, ]
    writeLines(sprintf(
      "| %s | %s | %.1f | %.1f | %.0f | %.1f | %.0f |",
      r$label, r$precision, r$oi_dram, r$oi_l2,
      r$gflops_achieved, r$dram_bytes / 1024^2,
      r$duration_ns / 1e3
    ), con)
  }

  writeLines(c(
    "",
    "## Regime classification",
    "",
    "Below the DRAM ceiling line: **bandwidth-bound**.",
    "Between DRAM and L2 ceiling: **bandwidth-bound** but L2 is helping.",
    "Above L2 ceiling: **compute-bound** (cannot be helped by more bandwidth).",
    "",
    sprintf("Total tensor-active kernels measured: **%d**", nrow(d))
  ), con)
}

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  d <- read.csv(args$in_path, stringsAsFactors = FALSE)
  cat(sprintf("Read %d kernels from %s\n", nrow(d), args$in_path))

  required <- c("label", "dram_read_bytes", "dram_write_bytes", "l2_bytes",
                "duration_ns", "hmma_count", "tensor_count", "alu_count")
  missing <- setdiff(required, names(d))
  if (length(missing)) {
    stop("CSV missing required columns: ", paste(missing, collapse = ", "),
         ". Re-run scripts/profile/ncu_profile_all.sh after the #92 metric extension.")
  }

  d <- compute_metrics(d)

  # Filter: only kernels with tensor activity (HMMA or IMMA > 0).
  # Pure scalar / data-movement kernels have a different roofline regime.
  d_tc <- d[d$tensor_count > 0, , drop = FALSE]
  cat(sprintf("Tensor-active kernels: %d (skipping %d scalar)\n",
              nrow(d_tc), nrow(d) - nrow(d_tc)))

  cat("\n=== summary ===\n")
  print(d_tc[, c("label", "precision", "oi_dram", "oi_l2",
                  "gflops_achieved")], digits = 4, row.names = FALSE)

  make_roofline(d_tc, args$out_path)
  cat(sprintf("\nWrote roofline figure to %s\n", args$out_path))

  write_markdown(d_tc, args$md_out)
  cat(sprintf("Wrote markdown summary to %s\n", args$md_out))
}

if (sys.nframe() == 0L) main()
