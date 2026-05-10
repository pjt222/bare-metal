#!/usr/bin/env Rscript
# cymatic_replot.R
#
# Re-render every cymatic per-trace heatmap from already-captured CSVs,
# without re-running the (expensive, GPU-bound) benchmark sweep. Used
# after a theme / palette refactor to pick up the new look without
# burning hours on bench replays.
#
# Reads:
#   docs/figures/cymatic/cymatic_optimize_2048.csv
#   docs/figures/cymatic/cymatic_fa_alignment_2048.csv
#
# Writes:
#   docs/figures/cymatic/cymatic_optimize_2048_<trace>.png
#   docs/figures/cymatic/cymatic_fa_alignment_2048_<trace>.png

suppressMessages({
    library(ggplot2); library(dplyr)
})

# Project-wide theme + viridis palettes.
source("scripts/audit/_theme.R")

fig_dir <- "docs/figures/cymatic"
grid_n <- 2048L

# ---- per-(n, m)-mode sweep ----
opt_csv <- file.path(fig_dir, sprintf("cymatic_optimize_%d.csv", grid_n))
if (file.exists(opt_csv)) {
    data <- read.csv(opt_csv, stringsAsFactors = FALSE)
    focus_traces <- intersect(
        c("radial_mid_pi6", "radial_bnd_pi4", "radial_bnd_5pi12",
          "circular_r030",  "circular_r060", "polar_tile_pi6",
          "rowmajor_full"),
        unique(data$trace))
    for (tr in focus_traces) {
        sub <- data[data$trace == tr, ]
        p <- ggplot(sub, aes(x = factor(n), y = factor(m), fill = speedup)) +
            geom_tile(color = "white") +
            geom_text(aes(label = sprintf("%.2f", speedup)), size = 3.0) +
            scale_fill_bm_div(midpoint = 1.0,
                              name = "speedup\n(>1 = cymatic wins)") +
            labs(title = sprintf("%s (grid=%d^2): cymatic speedup over (n, m)",
                                  tr, grid_n),
                 x = "n (angular frequency)",
                 y = "m (radial bands)") +
            theme_baremetal() +
            theme(panel.grid = element_blank())
        out_png <- file.path(fig_dir,
                             sprintf("cymatic_optimize_%d_%s.png", grid_n, tr))
        bm_save(p, out_png, width = 7.5, height = 4.0)
        cat(sprintf("[replot] wrote %s\n", out_png))
    }
} else {
    cat(sprintf("[replot] %s not found, skipping optimize set\n", opt_csv))
}

# ---- FA alignment sweep ----
fa_csv <- file.path(fig_dir, sprintf("cymatic_fa_alignment_%d.csv", grid_n))
if (file.exists(fa_csv)) {
    data <- read.csv(fa_csv, stringsAsFactors = FALSE)
    for (tr in unique(data$trace)) {
        sub <- data[data$trace == tr, ]
        p <- ggplot(sub, aes(x = factor(n), y = factor(m), fill = speedup)) +
            geom_tile(color = "white") +
            geom_text(aes(label = sprintf("%.2f", speedup)), size = 3) +
            scale_fill_bm_div(midpoint = 1.0, name = "speedup") +
            labs(title = sprintf("FA trace %s @ grid=%d^2: cymatic vs row-major",
                                  tr, grid_n),
                 x = "n", y = "m") +
            theme_baremetal() +
            theme(panel.grid = element_blank())
        out_png <- file.path(fig_dir,
                             sprintf("cymatic_fa_alignment_%d_%s.png",
                                     grid_n, tr))
        bm_save(p, out_png, width = 7, height = 4)
        cat(sprintf("[replot] wrote %s\n", out_png))
    }
} else {
    cat(sprintf("[replot] %s not found, skipping FA set\n", fa_csv))
}

cat("\n[replot] done.\n")
