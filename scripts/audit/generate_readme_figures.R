#!/usr/bin/env Rscript
# generate_readme_figures.R
#
# Produces all imagery linked from README.md. Run once after benchmarks
# are up to date. Outputs go to docs/figures/.
#
# Generated figures:
#   1. performance_overview.png  — bar chart of headline kernels, % of peak
#   2. roofline.png              — operational intensity vs achievable TFLOPS
#   3. fa_waterfall.png          — Flash Attention progression (regpv → v2_pipeline)
#   4. cymatic_speedup_grid.png  — cymatic vs row-major across traces × grid sizes
#   5. phase_progression.png     — cumulative speedup through phases 1-5

suppressPackageStartupMessages({
  for (pkg in c("ggplot2", "scales", "viridisLite", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("install.packages('%s')", pkg))
    }
  }
  library(ggplot2)
  library(jsonlite)
  library(scales)
})

# Project-wide theme + viridis palettes (audit follow-up).
source("scripts/audit/_theme.R")

OUT <- "docs/figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Hardware peaks (RTX 3070 Ti, GA104 sm_86)
FP32_PEAK_TFLOPS    <- 21.7
FP16_TC_PEAK_TFLOPS <- 174.0
INT8_TC_PEAK_TFLOPS <- 348.0
DRAM_BW_GBPS        <- 608.0

# ---- 1. Headline performance overview --------------------------------------

perf <- data.frame(
  kernel = c(
    "Naive SGEMM",
    "Tiled SGEMM",
    "Register-blocked SGEMM",
    "HGEMM (basic WMMA)",
    "HGEMM 16-warp",
    "Sparse HGEMM 2:4",
    "IGEMM 128x256",
    "Sparse INT8 mma.sp",
    "Conv2d implicit GEMM",
    "Flash Attn v2_pipeline",
    "ResBlock implicit conv"
  ),
  gflops = c(
    461,    # naive sgemm
    1031,   # tiled sgemm
    5000,   # register blocked sgemm
    7853,   # hgemm wmma
    31910,  # hgemm 16warp
    41721,  # sparse hgemm
    27591,  # igemm 128x256
    39674,  # sparse int8
    6687,   # conv2d implicit
    11453,  # FA v2_pipeline
    2025    # resblock implicit
  ),
  peak = c(
    FP32_PEAK_TFLOPS    * 1000,
    FP32_PEAK_TFLOPS    * 1000,
    FP32_PEAK_TFLOPS    * 1000,
    FP16_TC_PEAK_TFLOPS * 1000,
    FP16_TC_PEAK_TFLOPS * 1000,
    FP16_TC_PEAK_TFLOPS * 1000,
    INT8_TC_PEAK_TFLOPS * 1000,
    INT8_TC_PEAK_TFLOPS * 1000,
    FP16_TC_PEAK_TFLOPS * 1000,
    FP16_TC_PEAK_TFLOPS * 1000,
    FP16_TC_PEAK_TFLOPS * 1000
  ),
  category = c(
    "FP32", "FP32", "FP32",
    "FP16 TC", "FP16 TC", "FP16 TC sparse",
    "INT8 TC", "INT8 TC sparse",
    "FP16 TC", "FP16 TC", "FP16 TC"
  )
)
perf$pct_peak <- 100 * perf$gflops / perf$peak
perf$kernel  <- factor(perf$kernel, levels = perf$kernel)

p1 <- ggplot(perf, aes(x = kernel, y = gflops, fill = category)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%s\n%.1f%% peak",
                                 format(gflops, big.mark = ",", trim = TRUE),
                                 pct_peak)),
            hjust = -0.05, size = 2.9, lineheight = 0.95,
            colour = BM_DARK_FG) +
  coord_flip() +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.32))) +
  scale_fill_bm_disc() +
  labs(
    title = "Kernel Performance Headline",
    subtitle = "RTX 3070 Ti (GA104, sm_86) — measured GFLOPS; sparse = dense-equivalent",
    x = NULL, y = "GFLOPS",
    fill = NULL,
    caption = "Each kernel labelled with achieved GFLOPS and percent of its precision-class peak (FP32 = 21.7 TFLOPS, FP16 TC = 174, INT8 TC = 348)"
  ) +
  theme_baremetal() +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 9))

bm_save(p1, file.path(OUT, "performance_overview.png"),
        width = 10, height = 6)
cat("[fig] wrote performance_overview.png\n")

# ---- 2. Roofline ----------------------------------------------------------

# Operational intensity (FLOP / byte) for headline kernels
roof <- data.frame(
  kernel = c(
    "HGEMM 16-warp 4096^3",
    "Sparse HGEMM 2:4 2048^3",
    "Conv2d implicit GEMM",
    "Flash Attn v2_pipeline",
    "Tiled SGEMM 2048^3",
    "GroupNorm SD 320ch",
    "ResBlock implicit"
  ),
  oi_flops_per_byte = c(
    341,    # hgemm 4096^3 dense
    682,    # sparse hgemm (effective oi 2x)
    100,    # conv2d implicit
    256,    # FA Br=16
    85,     # tiled sgemm
    1.25,   # groupnorm
    50      # resblock
  ),
  achieved_tflops = c(
    31.91,
    41.72,
     6.69,
    11.45,
     1.03,
     0.05,    # ~50 GB/s memory-bound
     2.03
  )
)

# Build roofline curves: compute ceiling = peak; memory ceiling = bw * oi
oi_seq <- 10^seq(-1, 4, length.out = 200)
fp16_roof <- data.frame(
  oi = oi_seq,
  perf = pmin(FP16_TC_PEAK_TFLOPS, DRAM_BW_GBPS / 1000 * oi_seq),
  roof = "FP16 TC peak"
)
fp32_roof <- data.frame(
  oi = oi_seq,
  perf = pmin(FP32_PEAK_TFLOPS, DRAM_BW_GBPS / 1000 * oi_seq),
  roof = "FP32 peak"
)
mem_line <- data.frame(
  oi = oi_seq,
  perf = DRAM_BW_GBPS / 1000 * oi_seq,
  roof = "DRAM BW (608 GB/s)"
)

p2 <- ggplot() +
  geom_line(data = fp16_roof, aes(x = oi, y = perf, colour = roof),
            linewidth = 1.0) +
  geom_line(data = fp32_roof, aes(x = oi, y = perf, colour = roof),
            linewidth = 1.0, linetype = "dashed") +
  geom_line(data = mem_line, aes(x = oi, y = perf, colour = roof),
            linewidth = 0.6, linetype = "dotted") +
  geom_point(data = roof,
             aes(x = oi_flops_per_byte, y = achieved_tflops),
             size = 3.5, colour = BM_VIRIDIS_HIGH) +
  geom_text(data = roof,
            aes(x = oi_flops_per_byte, y = achieved_tflops, label = kernel),
            vjust = -1.0, size = 2.8, colour = BM_DARK_FG) +
  scale_x_log10(breaks = c(0.1, 1, 10, 100, 1000),
                labels = c("0.1", "1", "10", "100", "1000")) +
  scale_y_log10(breaks = c(0.01, 0.1, 1, 10, 100, 174),
                labels = c("0.01", "0.1", "1", "10", "100", "174")) +
  scale_colour_bm_disc() +
  labs(
    title = "Roofline — GA104 RTX 3070 Ti",
    subtitle = "Operational intensity (FLOP/byte) vs achieved TFLOPS",
    x = "Operational intensity (FLOP / byte, log)",
    y = "TFLOPS (log)",
    colour = NULL,
    caption = "Three viridis-coloured ceilings: FP16 Tensor Core (solid), FP32 (dashed), DRAM bandwidth (dotted). Points: measured kernels at viridis high-end yellow."
  ) +
  theme_baremetal() +
  theme(legend.position = "bottom")

bm_save(p2, file.path(OUT, "roofline.png"),
        width = 9, height = 6)
cat("[fig] wrote roofline.png\n")

# ---- 3. Flash Attention waterfall -----------------------------------------

fa <- data.frame(
  step = c(
    "regpv (baseline)",
    "lean state",
    "Q reg cache",
    "smem_work elim (v2)",
    "cp.async (v2_pipeline)"
  ),
  gflops = c(7154, 7600, 8800, 9998, 11453),
  ms     = c(2.448, 2.305, 1.991, 1.752, 1.529)
)
fa$cumulative <- fa$gflops / fa$gflops[1]
fa$step <- factor(fa$step, levels = fa$step)

# Cumulative speedup also encoded as fill colour - viridis sequential maps
# the optimisation arc onto perceptual order.
p3 <- ggplot(fa, aes(x = step, y = gflops, fill = cumulative)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%s GFLOPS\n(%.2f ms, %.2fx)",
                                 format(gflops, big.mark = ",", trim = TRUE),
                                 ms, cumulative)),
            vjust = -0.4, size = 3.0, lineheight = 0.95,
            colour = BM_DARK_FG) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  scale_fill_bm_seq(name = "cumulative\nspeedup") +
  labs(
    title = "Flash Attention Optimization Waterfall",
    subtitle = "seq=1024, batch=8, heads=8 — cumulative 1.60x from regpv to v2_pipeline",
    x = NULL, y = "GFLOPS",
    caption = "Per-step gain: lean state +6%, Q reg cache +16%, smem_work elimination +14%, cp.async +14%. Bar fill encodes cumulative speedup (viridis)."
  ) +
  theme_baremetal() +
  theme(axis.text.x = element_text(angle = 18, hjust = 1, size = 9))

bm_save(p3, file.path(OUT, "fa_waterfall.png"),
        width = 10, height = 5.5)
cat("[fig] wrote fa_waterfall.png\n")

# ---- 4. Cymatic speedup grid ----------------------------------------------

# Parse the captured results files
parse_bench <- function(path, grid_n) {
  if (!file.exists(path)) {
    return(data.frame(trace = character(), grid = integer(), speedup = numeric()))
  }
  lines <- readLines(path)
  body  <- lines[grepl("^[a-z]", lines) &
                 !grepl("^Device|^Grid|^Traces|^trace ", lines)]
  out <- data.frame(trace = character(), speedup = numeric())
  for (ln in body) {
    parts <- strsplit(trimws(ln), "\\s+")[[1]]
    if (length(parts) < 9) next
    tr   <- parts[1]
    spd  <- as.numeric(sub("x$", "", parts[length(parts)]))
    if (!is.na(spd)) {
      out <- rbind(out, data.frame(trace = tr, speedup = spd))
    }
  }
  out$grid <- grid_n
  out
}

cym_files <- list(
  list(path = "results/cymatic/grids/grid256_results.txt",  grid =  256),
  list(path = "results/cymatic/grids/grid512_results.txt",  grid =  512),
  list(path = "results/cymatic/grids/grid1024_results.txt", grid = 1024),
  list(path = "results/cymatic/grids/grid2048_results.txt", grid = 2048)
)
cym_df <- do.call(rbind, lapply(cym_files, function(x) parse_bench(x$path, x$grid)))

if (nrow(cym_df) > 0) {
  cym_df$grid_label <- factor(
    paste0(cym_df$grid, "²"),
    levels = c("256²", "512²", "1024²", "2048²")
  )
  cym_df$category <- "neutral"
  cym_df$category[cym_df$speedup >= 1.20] <- "cymatic wins"
  cym_df$category[cym_df$speedup <= 0.83] <- "cymatic loses"
  cym_df$category <- factor(cym_df$category,
                              levels = c("cymatic wins", "neutral", "cymatic loses"))

  # Order traces by 2048 speedup for readability
  ord <- cym_df[cym_df$grid == 2048, ]
  ord <- ord[order(-ord$speedup), ]
  cym_df$trace <- factor(cym_df$trace, levels = rev(ord$trace))

  # Continuous speedup encoded directly with diverging viridis-rooted scale
  # (purple <-> white <-> yellow, midpoint = 1.0). Drops the 3-bucket
  # categorical fill in favour of finer perceptual gradient.
  p4 <- ggplot(cym_df, aes(x = trace, y = speedup, fill = speedup)) +
    geom_hline(yintercept = 1.0, colour = "grey60", linetype = "dashed") +
    geom_col(width = 0.75) +
    coord_flip() +
    facet_wrap(~ grid_label, nrow = 1) +
    scale_fill_bm_div(midpoint = 1.0,
                      name = "speedup\n(>1 = cymatic wins)") +
    scale_y_continuous(breaks = c(0.5, 1.0, 1.5, 2.0)) +
    labs(
      title = "Cymatic Memory Layout — Speedup vs Row-Major (per access trace)",
      subtitle = "RTX 3070 Ti — Mode (n=6, m=4), gather-sum kernel, median of 11 runs",
      x = NULL, y = "Speedup (row_ms / cym_ms; >1 = cymatic wins)",
      caption = "Diverging viridis-derived fill centred at 1.0. 2048² is DRAM regime; smaller grids are L2-resident with locality differences hidden."
    ) +
    theme_baremetal() +
    theme(legend.position = "right",
          axis.text.y = element_text(size = 8))

  # Cymatic figures live under docs/figures/cymatic/.
  cym_dir <- file.path(OUT, "cymatic")
  dir.create(cym_dir, showWarnings = FALSE, recursive = TRUE)
  bm_save(p4, file.path(cym_dir, "cymatic_speedup_grid.png"),
          width = 12, height = 5.5)
  cat("[fig] wrote cymatic_speedup_grid.png\n")
} else {
  cat("[fig] cymatic results not found, skipping cymatic_speedup_grid.png\n")
}

# ---- 5. Phase progression -------------------------------------------------

phases <- data.frame(
  phase = factor(c(
    "Phase 1\nVector Add",
    "Phase 2\nNaive SGEMM",
    "Phase 2\nTiled SGEMM",
    "Phase 2\nReg-blocked",
    "Phase 2\nHGEMM basic",
    "Phase 2\nHGEMM 16-warp",
    "Phase 5\nSparse 2:4"
  ), levels = c(
    "Phase 1\nVector Add",
    "Phase 2\nNaive SGEMM",
    "Phase 2\nTiled SGEMM",
    "Phase 2\nReg-blocked",
    "Phase 2\nHGEMM basic",
    "Phase 2\nHGEMM 16-warp",
    "Phase 5\nSparse 2:4"
  )),
  gflops = c(NA, 461, 1031, 5000, 7853, 31910, 41721),
  cum_speedup = c(NA, 1.0, 2.24, 10.85, 17.03, 69.22, 90.51)
)
phases <- phases[!is.na(phases$gflops), ]

p5 <- ggplot(phases, aes(x = phase, y = cum_speedup, group = 1)) +
  geom_line(linewidth = 0.8, colour = BM_VIRIDIS_MID) +
  geom_point(aes(size = gflops, fill = cum_speedup), colour = BM_DARK_FG,
             shape = 21) +
  geom_text(aes(label = sprintf("%.1fx\n%s GFLOPS",
                                 cum_speedup,
                                 format(gflops, big.mark = ",", trim = TRUE))),
            vjust = -1.5, size = 2.9, lineheight = 0.95,
            colour = BM_DARK_FG) +
  scale_y_log10() +
  scale_size_continuous(range = c(3, 9), labels = comma) +
  scale_fill_bm_seq(guide = "none") +
  expand_limits(y = 200) +
  labs(
    title = "Cumulative Kernel Speedup Through the Phases",
    subtitle = "Naive SGEMM → 2:4 Sparse HGEMM (~91× cumulative)",
    x = NULL, y = "Cumulative speedup (log scale)",
    size = "GFLOPS",
    caption = "Each layer of amortization (tiling → register blocking → Tensor Cores → 16-warp occupancy → sparsity) compounds. Point fill encodes cumulative speedup (viridis)."
  ) +
  theme_baremetal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(size = 8))

bm_save(p5, file.path(OUT, "phase_progression.png"),
        width = 10, height = 5.5)
cat("[fig] wrote phase_progression.png\n")

cat("\n[fig] all figures written to ", OUT, "/\n", sep = "")

# ---- 6. Gap to measured local references -----------------------------------

.metric_from_entry <- function(entry) {
  if (!is.null(entry$gflops)) return(as.numeric(entry$gflops))
  if (!is.null(entry$tops)) {
    value <- as.numeric(entry$tops)
    if (!is.na(value) && value > 1000) value <- value / 1000.0
    return(value)
  }
  NA_real_
}

project_baselines <- fromJSON(file.path("data", "baselines.json"), simplifyVector = FALSE)
reference_baselines <- fromJSON(file.path("data", "reference_baselines.json"), simplifyVector = FALSE)

gap_rows <- list()
for (ref_kernel in names(reference_baselines$kernels)) {
  entry <- reference_baselines$kernels[[ref_kernel]]
  cfg_names <- setdiff(names(entry), c("exe", "library"))
  for (cfg in cfg_names) {
    ref_cfg <- entry[[cfg]]
    if (is.null(ref_cfg$project_kernel) || is.null(ref_cfg$project_config) || is.null(ref_cfg$peak)) next
    project_entry <- project_baselines$kernels[[ref_cfg$project_kernel]]
    if (is.null(project_entry)) next
    project_cfg <- project_entry[[ref_cfg$project_config]]
    if (is.null(project_cfg)) next

    ours <- .metric_from_entry(project_cfg)
    ref <- .metric_from_entry(ref_cfg)
    peak <- as.numeric(ref_cfg$peak)
    if (is.na(ours) || is.na(ref) || is.na(peak) || peak <= 0) next

    gap_rows[[length(gap_rows) + 1L]] <- data.frame(
      kernel = ref_cfg$label,
      ours_pct = 100 * ours / peak,
      ref_pct = 100 * ref / peak,
      stringsAsFactors = FALSE
    )
  }
}

if (length(gap_rows)) {
  gap <- do.call(rbind, gap_rows)
  gap$gap_factor <- gap$ref_pct / gap$ours_pct
  gap_long <- rbind(
    data.frame(kernel = gap$kernel, pct = gap$ours_pct,
               which = "this project (measured)"),
    data.frame(kernel = gap$kernel, pct = gap$ref_pct,
               which = "local reference (measured)")
  )
  gap_long$kernel <- factor(gap_long$kernel, levels = rev(gap$kernel))
  gap_long$which  <- factor(gap_long$which,
                            levels = c("this project (measured)",
                                       "local reference (measured)"))

  p6 <- ggplot(gap_long, aes(x = kernel, y = pct, fill = which)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.7) +
    geom_text(data = gap,
              aes(x = kernel, y = pmax(ours_pct, ref_pct) + 4,
                  label = sprintf("%.2fx ref/ours", gap_factor)),
              inherit.aes = FALSE,
              hjust = 0, size = 3.0, colour = BM_DARK_FG_MUTED, fontface = "italic") +
    coord_flip() +
    scale_y_continuous(limits = c(0, 105),
                       expand = expansion(mult = c(0, 0.02)),
                       labels = function(x) paste0(x, "%")) +
    scale_fill_bm_disc() +
    labs(
      title = "Measured Local Reference Gap",
      subtitle = "Percent of hardware peak — this project vs locally measured reference libraries",
      x = NULL, y = "% of precision-class peak",
      fill = NULL,
      caption = "Only workloads with locally installed reference-library harnesses are shown. Missing stacks are excluded rather than estimated."
    ) +
    theme_baremetal() +
    theme(legend.position = "bottom",
          axis.text.y = element_text(size = 9))

  bm_save(p6, file.path(OUT, "gap_to_sota.png"),
          width = 11, height = 5.5)
  cat("[fig] wrote gap_to_sota.png\n")
} else {
  cat("[fig] local reference baselines not found, skipping gap_to_sota.png\n")
}
