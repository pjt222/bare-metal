#!/usr/bin/env Rscript
# gen_cymatic_data.R
#
# Generate binary inputs for the CUDA cymatic benchmark:
#   perm.bin    — int32[n_inside]: cymatic address (0-indexed) for each
#                 row-major-active cell, plus header (grid_n, n_inside).
#   traces.bin  — packed access traces, one per pattern. Each trace is a
#                 list of row-major-active indices (0-indexed). The CUDA
#                 bench reads them and gathers from data[trace[k]] for
#                 row-major and data[perm[trace[k]]] for cymatic.
#
# Usage:
#   Rscript gen_cymatic_data.R [grid_n=1024] [n=6] [m=4] [n2=0] [m2=0] [alpha2=0] [domain=disc|square|overlayed]
#
# Resolves repo paths from the script location so it works both from the
# kernel directory and from the repo root.

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
source(normalizePath(file.path(script_dir, "..", "..", "..", "scripts", "cymatic", "cymatic_bench_io.R"),
                     winslash = "/", mustWork = TRUE))
cymatic_source("cymatic_mapping.R")
cymatic_source("cymatic_analyze.R")

args <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 1024L
n1     <- if (length(args) >= 2) as.integer(args[2]) else 6L
m1     <- if (length(args) >= 3) as.integer(args[3]) else 4L
n2     <- if (length(args) >= 4) as.integer(args[4]) else 0L
m2     <- if (length(args) >= 5) as.integer(args[5]) else 0L
alpha2 <- if (length(args) >= 6) as.numeric(args[6]) else 0.0
active_domain <- normalize_cymatic_domain(if (length(args) >= 7) args[7] else "disc")

modes <- list(list(n = n1, m = m1, alpha = 1.0))
if (n2 > 0L && m2 > 0L && alpha2 != 0) {
  modes[[length(modes) + 1L]] <- list(n = n2, m = m2, alpha = alpha2)
}

out_dir <- cymatic_kernel_dir()

cat(sprintf("[gen] computing mapping grid_n=%d domain=%s ...\n", grid_n, active_domain))
mapping <- cymatic_mapping(grid_n = grid_n, modes = modes, active_domain = active_domain)

# ---- Row-major-active ordering --------------------------------------------

inside_index <- build_inside_index(mapping)
inside_cells <- inside_index$inside_cells
n_inside <- inside_index$n_inside
stopifnot(n_inside == mapping$total_cells)

trace_to_rmi <- function(cells) {
  cells_to_rmi(cells, inside_index$rmi_lookup)
}

# ---- Permutation ----------------------------------------------------------

perm <- extract_perm(mapping, inside_cells)

cat(sprintf("[gen] grid=%d  n_inside=%d  buffer=%.1f MB (float)\n",
            grid_n, n_inside, n_inside * 4 / 1e6))

write_default_and_domain_artifact(write_perm_bin, "perm", "bin", active_domain,
                                  grid_n = grid_n, n_inside = n_inside, perm = perm)
cat(sprintf("[gen] wrote perm.bin and perm_%s.bin\n", active_domain))

# ---- Trace generation -----------------------------------------------------

cat("[gen] generating traces ...\n")
grid <- mapping$grid

build_trace <- function(name, cells) {
  rmi <- trace_to_rmi(cells)
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
write_default_and_domain_artifact(write_traces_bin, "traces", "bin", active_domain,
                                  traces = traces)
cat(sprintf("[gen] wrote traces.bin and traces_%s.bin (%d traces)\n",
            active_domain, length(traces)))

# Print summary
cat("\n=== Trace sizes ===\n")
for (tr in traces) {
  cat(sprintf("  %-20s n=%d\n", tr$name, length(tr$rmi)))
}

# Save mapping rds for reproducibility
saveRDS(mapping, file.path(out_dir, "mapping.rds"))
saveRDS(mapping, file.path(out_dir, sprintf("mapping_%s.rds", active_domain)))
cat(sprintf("[gen] saved mapping.rds and mapping_%s.rds\n", active_domain))
