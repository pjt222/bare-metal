#!/usr/bin/env Rscript
# gen_cymatic_data.R
#
# Generate binary inputs for the CUDA cymatic benchmark:
#   perm.bin    — int32[n_inside]: cymatic address (0-indexed) for each
#                 row-major-inside cell, plus header (grid_n, n_inside).
#   traces.bin  — packed access traces, one per pattern. Each trace is a
#                 list of row-major-inside indices (0-indexed). The CUDA
#                 bench reads them and gathers from data[trace[k]] for
#                 row-major and data[perm[trace[k]]] for cymatic.
#
# Usage:
#   Rscript gen_cymatic_data.R [grid_n=1024] [n=6] [m=4] [n2=0] [m2=0] [alpha2=0]
#
# Run from phase4/cymatic/. Sources ../../scripts/cymatic_mapping.R and
# ../../scripts/cymatic_analyze.R for the field math and trace generators.

source("../../scripts/cymatic_mapping.R")
source("../../scripts/cymatic_analyze.R")

args <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 1024L
n1     <- if (length(args) >= 2) as.integer(args[2]) else 6L
m1     <- if (length(args) >= 3) as.integer(args[3]) else 4L
n2     <- if (length(args) >= 4) as.integer(args[4]) else 0L
m2     <- if (length(args) >= 5) as.integer(args[5]) else 0L
alpha2 <- if (length(args) >= 6) as.numeric(args[6]) else 0.0

modes <- list(list(n = n1, m = m1, alpha = 1.0))
if (n2 > 0L && m2 > 0L && alpha2 != 0) {
  modes[[length(modes) + 1L]] <- list(n = n2, m = m2, alpha = alpha2)
}

cat(sprintf("[gen] computing mapping grid_n=%d ...\n", grid_n))
mapping <- cymatic_mapping(grid_n = grid_n, modes = modes)

# ---- Row-major-inside ordering --------------------------------------------

inside <- mapping$grid$inside
inside_cells <- which(inside, arr.ind = TRUE)
ord <- order(inside_cells[, 1], inside_cells[, 2])
inside_cells <- inside_cells[ord, , drop = FALSE]
n_inside <- nrow(inside_cells)
stopifnot(n_inside == mapping$total_cells)

# Map (i, j) -> row-major-inside index for trace conversion
ij_key <- paste(inside_cells[, 1], inside_cells[, 2], sep = "_")
rmi_lookup <- setNames(seq_len(n_inside) - 1L, ij_key)

ij_to_rmi <- function(cells) {
  k <- paste(cells[, 1], cells[, 2], sep = "_")
  out <- rmi_lookup[k]
  out[!is.na(out)]
}

# ---- Permutation ----------------------------------------------------------

perm <- mapping$address[inside_cells] - 1L  # 0-indexed
stopifnot(!any(is.na(perm)))
stopifnot(min(perm) == 0L, max(perm) == n_inside - 1L)
stopifnot(length(unique(perm)) == n_inside)  # bijection check

cat(sprintf("[gen] grid=%d  n_inside=%d  buffer=%.1f MB (float)\n",
            grid_n, n_inside, n_inside * 4 / 1e6))

con <- file("perm.bin", "wb")
writeBin(as.integer(grid_n),   con, size = 4)
writeBin(as.integer(n_inside), con, size = 4)
writeBin(as.integer(perm),     con, size = 4)
close(con)
cat("[gen] wrote perm.bin\n")

# ---- Trace generation -----------------------------------------------------

cat("[gen] generating traces ...\n")
grid <- mapping$grid

build_trace <- function(name, cells) {
  rmi <- ij_to_rmi(cells)
  list(name = name, rmi = as.integer(rmi))
}

# Mode (n=6) has angular sector midlines at theta = k * pi/6 (cos(6 theta) = +-1)
# and boundaries at theta = pi/12 + k * pi/6 (cos(6 theta) = 0). Sweep both to
# expose the layout's angular dependency.
traces <- list(
  build_trace("radial_mid_0",     trace_radial(grid, 0)),                # midline
  build_trace("radial_mid_pi6",   trace_radial(grid, pi / 6)),            # midline
  build_trace("radial_mid_pi3",   trace_radial(grid, pi / 3)),            # midline
  build_trace("radial_bnd_pi12",  trace_radial(grid, pi / 12)),           # boundary
  build_trace("radial_bnd_pi4",   trace_radial(grid, pi / 4)),            # boundary
  build_trace("radial_bnd_5pi12", trace_radial(grid, 5 * pi / 12)),       # boundary
  build_trace("circular_r060",    trace_circular(grid, 0.60)),
  build_trace("circular_r030",    trace_circular(grid, 0.30)),
  build_trace("polar_tile_pi6",   trace_polar_tile(grid, r0 = 0.5, dr = 0.20,
                                                   theta0 = pi / 6, dtheta = pi / 6)),
  build_trace("polar_tile_pi4",   trace_polar_tile(grid, r0 = 0.5, dr = 0.20,
                                                   theta0 = pi / 4, dtheta = pi / 6)),
  build_trace("radial_bias_07",   trace_radial_bias(grid, r0 = 0.7, sigma = 0.15,
                                                    n_samples = 65536)),
  build_trace("radial_bias_00",   trace_radial_bias(grid, r0 = 0.0, sigma = 0.20,
                                                    n_samples = 65536)),
  build_trace("rowmajor_full",    inside_cells),
  build_trace("colmajor_full",    inside_cells[order(inside_cells[, 2],
                                                     inside_cells[, 1]), ])
)

# Add a "shuffled" baseline (random permutation) for worst-case reference.
set.seed(42)
shuffled <- sample(seq_len(n_inside) - 1L, min(65536L, n_inside))
traces[[length(traces) + 1L]] <- list(name = "random", rmi = as.integer(shuffled))

# Write traces.bin
con <- file("traces.bin", "wb")
writeBin(as.integer(length(traces)), con, size = 4)
for (tr in traces) {
  name_bytes <- charToRaw(tr$name)
  writeBin(as.integer(length(name_bytes)), con, size = 4)
  writeBin(name_bytes, con)
  writeBin(as.integer(length(tr$rmi)),     con, size = 4)
  writeBin(tr$rmi, con, size = 4)
}
close(con)
cat(sprintf("[gen] wrote traces.bin (%d traces)\n", length(traces)))

# Print summary
cat("\n=== Trace sizes ===\n")
for (tr in traces) {
  cat(sprintf("  %-20s n=%d\n", tr$name, length(tr$rmi)))
}

# Save mapping rds for reproducibility
saveRDS(mapping, "mapping.rds")
cat("[gen] saved mapping.rds\n")
