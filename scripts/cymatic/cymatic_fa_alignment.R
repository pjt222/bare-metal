#!/usr/bin/env Rscript
# cymatic_fa_alignment.R -- (Issue #94 step 1)
#
# Sweep cymatic (n, m) modes against Flash Attention block-level access
# traces. Uses phase4/cymatic/gen_fa_traces.R for trace generation, then
# the same bench harness. Mirrors cymatic_optimize.R but with FA
# traces instead of the synthetic radial/circular set.

suppressMessages({
    library(ggplot2); library(dplyr)
})

WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
    cur <- Sys.getenv("LD_LIBRARY_PATH")
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                    paste(WSL_CUDA_LIB, cur, sep = ":")
                                 else WSL_CUDA_LIB)
}
Sys.setenv(PATH = paste("/usr/local/cuda/bin", Sys.getenv("PATH"), sep = ":"))

args   <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 2048L
n_grid <- if (length(args) >= 2) eval(parse(text = args[2])) else c(3,5,6,7,9)
m_grid <- if (length(args) >= 3) eval(parse(text = args[3])) else c(2,4,6)

cat(sprintf("[fa_align] grid=%d  n in {%s}  m in {%s}  total=%d\n",
            grid_n, paste(n_grid, collapse=","), paste(m_grid, collapse=","),
            length(n_grid) * length(m_grid)))

repo_root <- normalizePath(".")
cym_dir   <- file.path(repo_root, "phase4", "cymatic")
gen_fa    <- file.path(cym_dir, "gen_fa_traces.R")
bench_bin <- file.path(cym_dir, "bench")
fig_dir   <- file.path(repo_root, "docs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(gen_fa), file.exists(bench_bin))

parse_bench_out <- function(out_lines) {
    speedup_re <- "([0-9]+\\.[0-9]+)x\\s*$"
    body <- out_lines[grepl(speedup_re, out_lines)]
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
    system2("Rscript", c("gen_fa_traces.R", grid_n, n, m),
            stdout = TRUE, stderr = TRUE)
    out <- system2("./bench", stdout = TRUE, stderr = TRUE)
    df  <- parse_bench_out(out)
    if (!is.null(df)) { df$n <- n; df$m <- m }
    df
}

all_rows <- list()
t0 <- Sys.time()
total <- length(n_grid) * length(m_grid); done <- 0L
for (nn in n_grid) for (mm in m_grid) {
    done <- done + 1L
    cat(sprintf("[%2d/%2d] n=%d m=%d ... ", done, total, nn, mm))
    flush.console()
    df <- run_one(nn, mm)
    if (!is.null(df)) {
        all_rows[[length(all_rows) + 1]] <- df
        bi <- which.max(df$speedup); wi <- which.min(df$speedup)
        cat(sprintf("best=%s %.2fx  worst=%s %.2fx  geomean=%.2fx\n",
                    df$trace[bi], df$speedup[bi],
                    df$trace[wi], df$speedup[wi],
                    exp(mean(log(df$speedup)))))
    } else cat("FAIL\n")
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
cat(sprintf("\n[fa_align] sweep complete in %.1fs (%.1f s/config)\n",
            elapsed, elapsed / max(1, done)))

if (!length(all_rows)) stop("No successful configs.")
data <- do.call(rbind, all_rows)

csv_path <- file.path(fig_dir, sprintf("cymatic_fa_alignment_%d.csv", grid_n))
write.csv(data, csv_path, row.names = FALSE)
cat(sprintf("[fa_align] wrote %s\n", csv_path))

cat("\n== Best mode per FA trace ==\n")
best <- data |>
    group_by(trace) |>
    summarise(best_n = n[which.max(speedup)],
              best_m = m[which.max(speedup)],
              best_speed  = max(speedup),
              worst_speed = min(speedup),
              .groups = "drop") |>
    arrange(desc(best_speed))
print(as.data.frame(best))

cat("\n== Speedup matrix per trace ==\n")
for (tr in unique(data$trace)) {
    sub <- data[data$trace == tr, ]
    cat(sprintf("\n  %s:\n", tr))
    mat <- with(sub, tapply(speedup, list(m, n), function(x) mean(x)))
    print(round(mat, 2))
}

# Heatmap per trace
for (tr in unique(data$trace)) {
    sub <- data[data$trace == tr, ]
    p <- ggplot(sub, aes(x = factor(n), y = factor(m), fill = speedup)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.2f", speedup)), size = 3) +
        scale_fill_gradient2(midpoint = 1.0, low = "steelblue",
                             mid = "white", high = "firebrick",
                             name = "speedup") +
        labs(title = sprintf("FA trace %s @ grid=%d²: cymatic vs row-major",
                              tr, grid_n),
             x = "n", y = "m") +
        theme_minimal(base_size = 11) +
        theme(panel.grid = element_blank())
    ggsave(file.path(fig_dir,
                     sprintf("cymatic_fa_alignment_%d_%s.png", grid_n, tr)),
           p, width = 7, height = 4, dpi = 130)
}
cat("\n[fa_align] done.\n")
