#!/usr/bin/env Rscript
# cymatic_analyze.R
#
# Compare cymatic vs row-major mapping under several access patterns and
# report locality metrics (expected cache lines touched, average jump size).
#
# Access patterns tested:
#   - radial sweep:        addresses along a fixed θ as r increases
#   - circular sweep:      addresses along a fixed r as θ varies
#   - polar tile:          all cells with (r, θ) inside a small wedge
#   - random radial bias:  draw cells weighted by exp(-(r-r0)^2/σ^2)
#
# Two metrics:
#   (a) cache_lines_touched: sum over consecutive pairs in trace, count
#       1 + (1 if address[k+1]/L != address[k]/L else 0); L = cache line size in cells.
#   (b) mean_jump:           mean |address[k+1] - address[k]|; lower = more
#       coalesced, higher = more strided.
#
# Compares cymatic mapping to row-major (i*N + j) on the same trace.
#
# Usage:
#   Rscript scripts/cymatic/cymatic_analyze.R cymatic_mapping.rds [cache_line_cells=32]

if (!exists("cymatic_mapping")) {
  args_full <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args_full, value = TRUE)
  script_dir <- if (length(file_arg)) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  } else {
    normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  }
  source(normalizePath(file.path(script_dir, "cymatic_mapping.R"), winslash = "/", mustWork = TRUE))
}

# ---- Trace generators ------------------------------------------------------

trace_radial <- function(grid, theta_target = 0) {
  # cells closest to the half-line θ = theta_target, ordered by r ascending
  th <- grid$theta
  r  <- grid$r
  inside <- grid$inside
  d_theta <- pmin(abs(th - theta_target), 2 * pi - abs(th - theta_target))
  band <- inside & d_theta < (2 * pi / max(grid$grid_n, 1))
  cells <- which(band, arr.ind = TRUE)
  cells <- cells[order(r[cells]), , drop = FALSE]
  cells
}

trace_circular <- function(grid, r_target = 0.6) {
  r <- grid$r; th <- grid$theta; inside <- grid$inside
  band <- inside & abs(r - r_target) < (1 / grid$grid_n)
  cells <- which(band, arr.ind = TRUE)
  cells <- cells[order(th[cells]), , drop = FALSE]
  cells
}

trace_polar_tile <- function(grid, r0 = 0.5, dr = 0.15,
                              theta0 = pi / 4, dtheta = pi / 6) {
  r <- grid$r; th <- grid$theta; inside <- grid$inside
  d_theta <- pmin(abs(th - theta0), 2 * pi - abs(th - theta0))
  sel <- inside & abs(r - r0) < dr & d_theta < dtheta
  cells <- which(sel, arr.ind = TRUE)
  cells <- cells[order(r[cells], th[cells]), , drop = FALSE]
  cells
}

trace_radial_bias <- function(grid, r0 = 0.7, sigma = 0.15, n_samples = 1000) {
  r <- grid$r; inside <- grid$inside
  cand <- which(inside, arr.ind = TRUE)
  weights <- exp(-((r[cand] - r0)^2) / (2 * sigma^2))
  idx <- sample(seq_len(nrow(cand)), n_samples, replace = TRUE, prob = weights)
  cand[idx, , drop = FALSE]
}

# ---- Address resolvers -----------------------------------------------------

cymatic_addr_of <- function(cells, mapping) {
  mapping$address[cells]
}

rowmajor_addr_of <- function(cells, grid) {
  # Address contiguous over active cells in row-major order
  inside <- grid$inside
  raster <- which(inside, arr.ind = TRUE)
  raster <- raster[order(raster[, 1], raster[, 2]), , drop = FALSE]
  key <- function(c) paste0(c[, 1], "_", c[, 2])
  ix <- match(key(cells), key(raster))
  ix
}

# ---- Metrics ---------------------------------------------------------------

cache_lines_touched <- function(addr, line_cells = 32) {
  addr <- addr[!is.na(addr)]
  if (length(addr) == 0) return(0)
  length(unique((addr - 1) %/% line_cells))
}

mean_jump <- function(addr) {
  addr <- addr[!is.na(addr)]
  if (length(addr) < 2) return(NA_real_)
  mean(abs(diff(addr)))
}

run_one_trace <- function(name, cells, mapping, line_cells) {
  addr_cym <- cymatic_addr_of(cells, mapping)
  addr_row <- rowmajor_addr_of(cells, mapping$grid)
  data.frame(
    pattern    = name,
    n_cells    = nrow(cells),
    cym_lines  = cache_lines_touched(addr_cym, line_cells),
    row_lines  = cache_lines_touched(addr_row, line_cells),
    cym_jump   = mean_jump(addr_cym),
    row_jump   = mean_jump(addr_row),
    line_cells = line_cells
  )
}

analyze <- function(mapping, line_cells = 32) {
  grid <- mapping$grid
  results <- rbind(
    run_one_trace("radial_sweep",    trace_radial(grid, 0),                 mapping, line_cells),
    run_one_trace("radial_sweep_45", trace_radial(grid, pi/4),              mapping, line_cells),
    run_one_trace("circular_r0.6",   trace_circular(grid, 0.6),             mapping, line_cells),
    run_one_trace("circular_r0.3",   trace_circular(grid, 0.3),             mapping, line_cells),
    run_one_trace("polar_tile_pi4",  trace_polar_tile(grid),                mapping, line_cells),
    run_one_trace("radial_bias_0.7", trace_radial_bias(grid, 0.7, 0.15),    mapping, line_cells),
    run_one_trace("radial_bias_0.0", trace_radial_bias(grid, 0.0, 0.20),    mapping, line_cells)
  )
  results$cym_vs_row_lines <- results$row_lines / pmax(results$cym_lines, 1L)
  results$cym_vs_row_jump  <- results$row_jump  / pmax(results$cym_jump, 1)
  results
}

# ---- Region locality summary ----------------------------------------------

region_geometry_summary <- function(mapping) {
  ord <- mapping$region_order
  data.frame(
    n_regions          = nrow(ord),
    cells_total        = sum(ord$cell_count),
    region_size_min    = min(ord$cell_count),
    region_size_p25    = quantile(ord$cell_count, 0.25),
    region_size_median = median(ord$cell_count),
    region_size_p75    = quantile(ord$cell_count, 0.75),
    region_size_max    = max(ord$cell_count),
    size_ratio_max_min = max(ord$cell_count) / max(min(ord$cell_count), 1),
    cells_per_region_mean = mean(ord$cell_count)
  )
}

# ---- CLI -------------------------------------------------------------------

.is_main_analyze <- function() {
  if (interactive()) return(FALSE)
  if (sys.nframe() > 1L) return(FALSE)
  invoked <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  if (length(invoked) == 0L) return(TRUE)
  basename(invoked) == "cymatic_analyze.R"
}

if (.is_main_analyze()) {
  args <- commandArgs(trailingOnly = TRUE)
  rds_path   <- if (length(args) >= 1) args[1] else "cymatic_mapping.rds"
  line_cells <- if (length(args) >= 2) as.integer(args[2]) else 32L
  if (!file.exists(rds_path)) {
    stop(sprintf("missing %s — run scripts/cymatic/cymatic_mapping.R first", rds_path))
  }
  mapping <- readRDS(rds_path)

  cat("\n=== Region geometry ===\n")
  print(region_geometry_summary(mapping))

  cat(sprintf("\n=== Locality (cache_line = %d cells) ===\n", line_cells))
  res <- analyze(mapping, line_cells)
  print(res, row.names = FALSE)

  out_dir <- "kernels/memory_layout/cymatic"
  if (!dir.exists(out_dir)) out_dir <- "."
  out_path <- file.path(out_dir, "cymatic_locality.csv")
  write.csv(res, out_path, row.names = FALSE)
  cat(sprintf("\n[cymatic] wrote %s\n", out_path))
  cat("[cymatic] interpretation: cym_vs_row_lines > 1 ⇒ cymatic wins on this pattern\n")
  cat("[cymatic]                  cym_vs_row_jump  > 1 ⇒ cymatic produces shorter strides\n")
}
