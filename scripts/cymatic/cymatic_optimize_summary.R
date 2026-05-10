#!/usr/bin/env Rscript
# cymatic_optimize_summary.R -- post-process the (n, m) sweep results.
# Reads docs/figures/cymatic/cymatic_optimize_2048.csv, produces:
#   - summary CSV (best mode per trace, gain vs default)
#   - facet plot of all 15 traces' speedup heatmaps
#   - geomean-per-mode bar chart

suppressMessages({
    library(ggplot2); library(dplyr); library(tidyr)
})

# Project-wide theme + viridis palettes (audit follow-up).
for (p in c("scripts/audit/_theme.R",
            "../audit/_theme.R",
            "../../scripts/audit/_theme.R")) {
    if (file.exists(p)) { source(p); break }
}

csv_path <- "docs/figures/cymatic/cymatic_optimize_2048.csv"
data <- read.csv(csv_path, stringsAsFactors = FALSE)
default_n <- 6L; default_m <- 4L

# ---- Summary table ----
best <- data |>
    group_by(trace) |>
    summarise(
        best_n     = n[which.max(speedup)],
        best_m     = m[which.max(speedup)],
        best_speed = max(speedup),
        worst_n    = n[which.min(speedup)],
        worst_m    = m[which.min(speedup)],
        worst_speed = min(speedup),
        default_speed = ifelse(any(n == default_n & m == default_m),
                               speedup[n == default_n & m == default_m][1],
                               NA_real_),
        .groups = "drop"
    ) |>
    mutate(gain_vs_default = best_speed / default_speed,
           range_ratio     = best_speed / worst_speed) |>
    arrange(desc(gain_vs_default))

write.csv(best, "docs/figures/cymatic/cymatic_optimize_2048_summary.csv",
          row.names = FALSE)
cat(sprintf("[opt] wrote summary CSV (%d traces)\n", nrow(best)))

# ---- Facet plot: all traces, speedup over (n, m) ----
p_facet <- ggplot(data, aes(x = factor(n), y = factor(m), fill = speedup)) +
    geom_tile(color = "white", linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.2f", speedup)), size = 1.7) +
    scale_fill_bm_div(midpoint = 1.0, name = "speedup",
                      limits = c(0.4, 2.5)) +
    facet_wrap(~ trace, ncol = 4) +
    labs(title = "Cymatic memory layout: speedup vs row-major over (n, m)",
         subtitle = "GRID=2048 (13 MB DRAM), 54 modes / trace, 15 traces. Default mode (n=6, m=4) not best on any trace.",
         x = "n (angular frequency)",
         y = "m (radial bands)") +
    theme_baremetal(base_size = 9) +
    theme(panel.grid = element_blank())
bm_save(p_facet, "docs/figures/cymatic/cymatic_optimize_2048_facet.png",
        width = 14, height = 9)
cat("[opt] wrote facet PNG\n")

# ---- Geomean bar chart ----
mode_geo <- data |>
    group_by(n, m) |>
    summarise(geomean = exp(mean(log(speedup))),
              min_sp  = min(speedup),
              max_sp  = max(speedup),
              .groups = "drop") |>
    arrange(desc(geomean))

p_geo <- ggplot(mode_geo, aes(x = factor(n), y = factor(m), fill = geomean)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", geomean)), size = 3.0) +
    scale_fill_bm_div(midpoint = 1.0, name = "geomean") +
    labs(title = "Geomean speedup per mode (across 15 traces)",
         subtitle = sprintf("Best geomean: n=%d m=%d (%.3fx). Default n=6 m=4: %.3fx.",
                            mode_geo$n[1], mode_geo$m[1], mode_geo$geomean[1],
                            mode_geo$geomean[mode_geo$n == default_n &
                                             mode_geo$m == default_m]),
         x = "n", y = "m") +
    theme_baremetal() +
    theme(panel.grid = element_blank())
bm_save(p_geo, "docs/figures/cymatic/cymatic_optimize_2048_geomean.png",
        width = 7, height = 5)
cat("[opt] wrote geomean PNG\n")

# ---- Print headline summary ----
cat("\n== Best mode per trace (sorted by gain over default) ==\n")
print(best, n = Inf)

cat("\nHeadline numbers:\n")
cat(sprintf(
    "  default (n=6, m=4) wins on no trace; per-trace optima range %.2fx-%.2fx\n",
    min(best$best_speed), max(best$best_speed)))
cat(sprintf("  largest gain over default: %.2fx (%s, mode %d/%d)\n",
            max(best$gain_vs_default),
            best$trace[which.max(best$gain_vs_default)],
            best$best_n[which.max(best$gain_vs_default)],
            best$best_m[which.max(best$gain_vs_default)]))
cat(sprintf(
    "  geomean of (best per-trace) over (default per-trace): %.3fx\n",
    exp(mean(log(best$best_speed / best$default_speed),
             na.rm = TRUE))))
cat(sprintf("  best single mode by geomean across all 15 traces: n=%d m=%d (%.3fx)\n",
            mode_geo$n[1], mode_geo$m[1], mode_geo$geomean[1]))

cat("\n[opt] done.\n")
