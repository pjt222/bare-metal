#!/usr/bin/env Rscript
# cymatic_optimize.R -- sweep (n, m) modes and tabulate per-trace speedups
# (issue #93).
#
# For each (n, m) in the grid:
#   1. Run kernels/memory_layout/cymatic/gen_cymatic_data.R GRID n m   -> rewrites perm.bin + traces.bin
#   2. Run kernels/memory_layout/cymatic/bench                  -> table of per-trace speedups
#   3. Parse and stash row per trace
# Output: docs/figures/cymatic/cymatic_optimize_<grid>.csv (long form) and a heatmap
# image per trace into docs/figures/cymatic/cymatic_optimize_<grid>_<trace>.png.

suppressMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
})

# Project-wide theme + viridis palettes (audit follow-up).
for (p in c("scripts/audit/_theme.R",
            "../audit/_theme.R",
            "../../scripts/audit/_theme.R")) {
    if (file.exists(p)) { source(p); break }
}

WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
    cur <- Sys.getenv("LD_LIBRARY_PATH")
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                    paste(WSL_CUDA_LIB, cur, sep = ":")
                                 else WSL_CUDA_LIB)
}
Sys.setenv(PATH = paste("/usr/local/cuda/bin", Sys.getenv("PATH"), sep = ":"))

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
args   <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 2048L
n_grid <- if (length(args) >= 2) eval(parse(text = args[2])) else 2:10
m_grid <- if (length(args) >= 3) eval(parse(text = args[3])) else 1:6

cat(sprintf("[opt] grid=%d  n in {%s}  m in {%s}  total configs=%d\n",
            grid_n, paste(n_grid, collapse=","), paste(m_grid, collapse=","),
            length(n_grid) * length(m_grid)))

repo_root  <- normalizePath(".")
cym_dir    <- file.path(repo_root, "phase4", "cymatic")
gen_script <- file.path(cym_dir, "gen_cymatic_data.R")
bench_bin  <- file.path(cym_dir, "bench")
fig_dir    <- file.path(repo_root, "docs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

stopifnot(file.exists(gen_script), file.exists(bench_bin))

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# Parse the per-trace table out of bench stdout. Each data row looks
# roughly like:
#   "radial_mid_pi6             1603          0.000         48.2     7.9%   ...   1.00x"
# We grab the leading non-whitespace token (trace name) and the trailing
# "<float>x" speedup.
parse_bench_out <- function(out_lines) {
    # Trace name may include digits (radial_mid_pi6, circular_r030 etc.).
    # Identify body rows by trailing "<float>x" speedup token.
    speedup_re <- "([0-9]+\\.[0-9]+)x\\s*$"
    body <- out_lines[grepl(speedup_re, out_lines)]
    # Skip any header / divider lines that happen to end in "x"
    body <- body[grepl("^\\s*[a-zA-Z]", body)]
    if (!length(body)) return(NULL)
    name_re <- "^\\s*([a-zA-Z][a-zA-Z0-9_]*)"
    name    <- sub(name_re, "\\1", regmatches(body, regexpr(name_re, body)))
    sp_m    <- regmatches(body, regexpr(speedup_re, body))
    speedup <- as.numeric(sub("x\\s*$", "", sp_m))
    keep    <- !is.na(name) & !is.na(speedup) & nzchar(name)
    data.frame(trace = name[keep], speedup = speedup[keep],
               stringsAsFactors = FALSE)
}

run_one <- function(n, m) {
    setwd(cym_dir); on.exit(setwd(repo_root), add = TRUE)
    # Regenerate input files for this mode
    gen_out <- system2("Rscript", c("gen_cymatic_data.R",
                                    grid_n, n, m),
                       stdout = TRUE, stderr = TRUE)
    if (!file.exists("perm.bin") || !file.exists("traces.bin")) {
        cat("  [warn] gen failed for n=", n, " m=", m, "\n", sep="")
        return(NULL)
    }
    out <- system2("./bench", stdout = TRUE, stderr = TRUE)
    df  <- parse_bench_out(out)
    if (is.null(df)) {
        cat("  [warn] bench parse failed for n=", n, " m=", m, "\n", sep="")
        return(NULL)
    }
    df$n <- n; df$m <- m
    df
}

# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
all_rows <- list()
t0 <- Sys.time()
total <- length(n_grid) * length(m_grid)
done  <- 0L
for (nn in n_grid) for (mm in m_grid) {
    done <- done + 1L
    cat(sprintf("[%2d/%2d] n=%d m=%d ... ", done, total, nn, mm))
    flush.console()
    df <- run_one(nn, mm)
    if (!is.null(df)) {
        all_rows[[length(all_rows) + 1]] <- df
        # Brief summary: best & worst trace this mode
        best_idx  <- which.max(df$speedup)
        worst_idx <- which.min(df$speedup)
        cat(sprintf("best=%s %.2fx, worst=%s %.2fx, geomean=%.2fx\n",
                    df$trace[best_idx],  df$speedup[best_idx],
                    df$trace[worst_idx], df$speedup[worst_idx],
                    exp(mean(log(df$speedup)))))
    } else {
        cat("FAIL\n")
    }
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
cat(sprintf("\n[opt] sweep complete in %.1fs (%.1f s/config)\n",
            elapsed, elapsed / max(1, done)))

if (!length(all_rows)) {
    stop("No successful configs.")
}
data <- do.call(rbind, all_rows)

# --------------------------------------------------------------------------
# Save CSV
# --------------------------------------------------------------------------
csv_path <- file.path(fig_dir, sprintf("cymatic_optimize_%d.csv", grid_n))
write.csv(data, csv_path, row.names = FALSE)
cat(sprintf("[opt] wrote %s (%d rows, %d traces, %d modes)\n",
            csv_path, nrow(data),
            length(unique(data$trace)),
            length(unique(paste(data$n, data$m)))))

# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
default_n <- 6L; default_m <- 4L

cat("\n== Top mode per trace (by speedup) ==\n")
top_per_trace <- data %>%
    group_by(trace) %>%
    arrange(desc(speedup)) %>%
    slice_head(n = 5) %>%
    ungroup()
print(top_per_trace, n = Inf)

cat("\n== Best mode per trace vs default (n=6, m=4) ==\n")
best <- data %>%
    group_by(trace) %>%
    summarise(
        best_n     = n[which.max(speedup)],
        best_m     = m[which.max(speedup)],
        best_speed = max(speedup),
        default_speed = {
            row <- speedup[n == default_n & m == default_m]
            if (length(row)) row[1] else NA_real_
        },
        gain_vs_default = best_speed / default_speed,
        .groups = "drop"
    )
print(best, n = Inf)

cat(sprintf("\n[opt] traces where best mode > default by >5%%: %d / %d\n",
            sum(best$gain_vs_default > 1.05, na.rm = TRUE), nrow(best)))

# --------------------------------------------------------------------------
# Plot heatmap per trace (subset of "interesting" traces)
# --------------------------------------------------------------------------
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
        labs(title = sprintf("%s (grid=%d²): cymatic speedup over (n, m)",
                              tr, grid_n),
             x = "n (angular frequency)",
             y = "m (radial bands)") +
        theme_baremetal() +
        theme(panel.grid = element_blank())
    out_png <- file.path(fig_dir,
                         sprintf("cymatic_optimize_%d_%s.png", grid_n, tr))
    bm_save(p, out_png, width = 7.5, height = 4.0)
    cat(sprintf("[opt] wrote %s\n", out_png))
}

cat("\n[opt] done.\n")
