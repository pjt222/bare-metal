#!/usr/bin/env Rscript
# cymatic_mapping.R
#
# Cymatic-inspired vector → memory mapping.
#
# Idea: a clamped circular membrane has standing-wave modes u_{n,m}(r,θ) =
#   J_n(k_{n,m} r) · cos(n θ),    k_{n,m} R = j_{n,m}
# where j_{n,m} is the m-th positive zero of the Bessel function J_n.
# The sign of u partitions the disc into "antinode regions" — bounded patches
# separated by nodal circles and nodal diameters. Each region is a stable
# island of in-phase oscillation.
#
# We use those regions as memory blocks. A 1-D address space is then a
# traversal of regions plus cells within them. Properties:
#   - Regions vary in size: small near the center for high (n,m), large near
#     the rim, mixed under mode superposition.
#   - Vectors with rotational / radial access patterns (FFT, polar warp,
#     radial-bias attention, spherical harmonics, diffusion 2D maps) gain
#     locality vs naive row-major.
#   - The mapping is deterministic and computable per (mode, resolution).
#
# Outputs:
#   - region map matrix (grid_n × grid_n, region_id ∈ {0, 1, ...}; 0 = outside disc)
#   - per-cell linear address (row-major over occupied cells, in region order)
#   - region table: region_id, cell_count, centroid_r, centroid_theta, area_fraction
#
# Usage:
#   Rscript scripts/cymatic_mapping.R [grid_n=128] [n=4] [m=3] [n2=0] [m2=0] [alpha=0]
# Example single mode (4, 3):  Rscript scripts/cymatic_mapping.R 128 4 3
# Example mixed (4,3)+0.6*(2,5): Rscript scripts/cymatic_mapping.R 128 4 3 2 5 0.6

# ---- Bessel zeros ----------------------------------------------------------

bessel_zeros_J <- function(n, max_m, search_max = 200, step = 0.01) {
  # First max_m positive zeros of J_n(x) by sign-change scan + uniroot refine.
  zeros <- numeric(max_m)
  found <- 0
  prev_x <- 1e-3
  prev_val <- besselJ(prev_x, n)
  x <- prev_x + step
  while (found < max_m && x < search_max) {
    val <- besselJ(x, n)
    if (!is.na(val) && sign(val) != sign(prev_val) && prev_val != 0) {
      z <- tryCatch(
        uniroot(function(t) besselJ(t, n), c(prev_x, x))$root,
        error = function(e) NA_real_
      )
      if (!is.na(z)) {
        found <- found + 1
        zeros[found] <- z
      }
    }
    prev_x <- x
    prev_val <- val
    x <- x + step
  }
  if (found < max_m) {
    stop(sprintf("only found %d zeros of J_%d up to x=%.1f", found, n, search_max))
  }
  zeros
}

# ---- Mode field ------------------------------------------------------------

chladni_mode <- function(r, theta, n, m, R_disc = 1, zeros_cache = NULL) {
  # u_{n,m}(r, θ) on a clamped disc of radius R_disc.
  if (is.null(zeros_cache)) zeros_cache <- bessel_zeros_J(n, m)
  k <- zeros_cache[m] / R_disc
  besselJ(k * r, n) * cos(n * theta)
}

mixed_mode_field <- function(r, theta, modes) {
  # modes: list of list(n=..., m=..., alpha=...)
  out <- 0
  for (md in modes) {
    z <- bessel_zeros_J(md$n, md$m)
    out <- out + md$alpha * chladni_mode(r, theta, md$n, md$m, zeros_cache = z)
  }
  out
}

# ---- Disc grid -------------------------------------------------------------

build_disc_grid <- function(grid_n, R_disc = 1) {
  # Cartesian grid_n × grid_n covering [-R, R]^2; mark cells inside the disc.
  xs <- seq(-R_disc, R_disc, length.out = grid_n)
  ys <- seq(-R_disc, R_disc, length.out = grid_n)
  Xg <- matrix(rep(xs, each = grid_n), grid_n, grid_n)
  Yg <- matrix(rep(ys, times = grid_n), grid_n, grid_n)
  Rg <- sqrt(Xg^2 + Yg^2)
  Tg <- atan2(Yg, Xg)
  inside <- Rg <= R_disc * (1 - 0.5 / grid_n)  # margin to avoid edge spikes
  list(x = Xg, y = Yg, r = Rg, theta = Tg, inside = inside,
       grid_n = grid_n, R_disc = R_disc)
}

# ---- Region labeling (4-connected flood fill on sign of field) ------------

label_regions <- function(field, inside) {
  # 4-connected components of cells that share sign(field) AND are inside.
  # Uses linear indexing + preallocated queue so it is O(N^2), not O(N^2 log N).
  grid_n <- nrow(field)
  sgn <- sign(field)
  sgn[!inside] <- 0
  label <- integer(grid_n * grid_n)              # flat, 0 = unlabeled
  sgn_flat <- as.integer(sgn)                    # flat, -1/0/1
  total <- length(label)

  # Preallocate queue once; cap at total cells (any region <= grid)
  q <- integer(total)
  next_id <- 0L

  for (start in seq_len(total)) {
    if (sgn_flat[start] == 0L || label[start] != 0L) next
    next_id <- next_id + 1L
    target <- sgn_flat[start]
    label[start] <- next_id
    q[1L] <- start
    head <- 1L
    tail <- 1L
    while (head <= tail) {
      idx <- q[head]; head <- head + 1L
      # row, col in 1-based:  i = ((idx-1) %% grid_n) + 1, j = ((idx-1) %/% grid_n) + 1
      im1 <- idx - 1L
      i <- im1 %% grid_n
      j <- im1 %/% grid_n
      # neighbors as flat indices
      # up: (i-1, j) -> idx - 1, if i > 0
      if (i > 0L) {
        nb <- idx - 1L
        if (label[nb] == 0L && sgn_flat[nb] == target) {
          label[nb] <- next_id; tail <- tail + 1L; q[tail] <- nb
        }
      }
      # down: (i+1, j) -> idx + 1, if i < grid_n-1
      if (i < grid_n - 1L) {
        nb <- idx + 1L
        if (label[nb] == 0L && sgn_flat[nb] == target) {
          label[nb] <- next_id; tail <- tail + 1L; q[tail] <- nb
        }
      }
      # left: (i, j-1) -> idx - grid_n, if j > 0
      if (j > 0L) {
        nb <- idx - grid_n
        if (label[nb] == 0L && sgn_flat[nb] == target) {
          label[nb] <- next_id; tail <- tail + 1L; q[tail] <- nb
        }
      }
      # right: (i, j+1) -> idx + grid_n, if j < grid_n-1
      if (j < grid_n - 1L) {
        nb <- idx + grid_n
        if (label[nb] == 0L && sgn_flat[nb] == target) {
          label[nb] <- next_id; tail <- tail + 1L; q[tail] <- nb
        }
      }
    }
  }
  dim(label) <- c(grid_n, grid_n)
  list(label = label, n_regions = next_id)
}

# ---- Region ordering -------------------------------------------------------

order_regions_radial <- function(label, grid) {
  # Sort regions by (centroid_r, centroid_theta) — radial-then-angular sweep.
  # This makes vectors that lie in adjacent radial shells map to nearby
  # addresses, which matches expected access locality for radial workloads.
  ids <- sort(unique(as.integer(label[label > 0L])))
  centers <- t(sapply(ids, function(id) {
    sel <- label == id
    c(mean(grid$r[sel]), mean(grid$theta[sel]), sum(sel))
  }))
  colnames(centers) <- c("r_centroid", "theta_centroid", "cell_count")
  ord <- order(centers[, "r_centroid"], centers[, "theta_centroid"])
  data.frame(
    region_id = ids[ord],
    new_id    = seq_along(ids),
    r_centroid     = centers[ord, "r_centroid"],
    theta_centroid = centers[ord, "theta_centroid"],
    cell_count     = centers[ord, "cell_count"]
  )
}

# ---- Address assignment ----------------------------------------------------

assign_addresses <- function(label, region_order) {
  # Walk regions in region_order; within each region, assign addresses in
  # row-major Cartesian sweep over the region's cells. Returns a matrix the
  # same shape as label, with linear addresses (NA outside disc / nodal cells).
  grid_n <- nrow(label)
  addr <- matrix(NA_integer_, grid_n, grid_n)
  ptr <- 0L
  id_to_new <- setNames(region_order$new_id, region_order$region_id)
  for (rid in region_order$region_id) {
    cells <- which(label == rid, arr.ind = TRUE)  # (i, j) pairs, row-major-ish
    # Sort by (i, j) — Cartesian raster within region.
    ord <- order(cells[, 1], cells[, 2])
    cells <- cells[ord, , drop = FALSE]
    for (k in seq_len(nrow(cells))) {
      ptr <- ptr + 1L
      addr[cells[k, 1], cells[k, 2]] <- ptr
    }
  }
  list(address = addr, total_cells = ptr)
}

# ---- Top-level compute -----------------------------------------------------

cymatic_mapping <- function(grid_n = 128,
                             modes = list(list(n = 4, m = 3, alpha = 1.0)),
                             R_disc = 1) {
  cat(sprintf("[cymatic] grid=%d  modes=", grid_n))
  for (md in modes) {
    cat(sprintf("(n=%d, m=%d, α=%.2f) ", md$n, md$m, md$alpha))
  }
  cat("\n")

  grid <- build_disc_grid(grid_n, R_disc)
  field <- mixed_mode_field(grid$r, grid$theta, modes)
  field[!grid$inside] <- 0

  cat("[cymatic] labeling regions ...\n")
  lab <- label_regions(field, grid$inside)
  cat(sprintf("[cymatic] %d regions\n", lab$n_regions))

  ord <- order_regions_radial(lab$label, grid)
  addr <- assign_addresses(lab$label, ord)
  cat(sprintf("[cymatic] %d cells addressed (disc area)\n", addr$total_cells))

  # Region size statistics
  sizes <- ord$cell_count
  cat(sprintf("[cymatic] region size: min=%d  median=%d  mean=%.1f  max=%d\n",
              as.integer(min(sizes)), as.integer(median(sizes)),
              mean(sizes), as.integer(max(sizes))))
  cat(sprintf("[cymatic] size ratio max/min = %.1fx\n",
              max(sizes) / max(min(sizes), 1)))

  list(
    grid          = grid,
    field         = field,
    label         = lab$label,
    n_regions     = lab$n_regions,
    region_order  = ord,
    address       = addr$address,
    total_cells   = addr$total_cells,
    modes         = modes
  )
}

# ---- Mapping table writer --------------------------------------------------

write_mapping_table <- function(mapping, path = "cymatic_mapping.csv") {
  # Per-cell row: address, i, j, x, y, r, theta, region_id_new, region_id_orig
  grid <- mapping$grid
  cells <- which(!is.na(mapping$address), arr.ind = TRUE)
  rid_orig <- mapping$label[cells]
  rid_new  <- mapping$region_order$new_id[match(rid_orig, mapping$region_order$region_id)]
  df <- data.frame(
    address     = as.integer(mapping$address[cells]),
    i           = as.integer(cells[, 1]),
    j           = as.integer(cells[, 2]),
    x           = grid$x[cells],
    y           = grid$y[cells],
    r           = grid$r[cells],
    theta       = grid$theta[cells],
    region_id   = as.integer(rid_new),
    region_orig = as.integer(rid_orig)
  )
  df <- df[order(df$address), ]
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("[cymatic] wrote %s (%d rows)\n", path, nrow(df)))
  invisible(df)
}

# ---- CLI -------------------------------------------------------------------

# Only run CLI when this file IS the entry script (not when sourced).
.is_main <- function() {
  if (interactive()) return(FALSE)
  if (sys.nframe() > 1L) return(FALSE)
  invoked <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  if (length(invoked) == 0L) return(TRUE)
  basename(invoked) == "cymatic_mapping.R"
}

if (.is_main()) {
  args <- commandArgs(trailingOnly = TRUE)
  grid_n <- if (length(args) >= 1) as.integer(args[1]) else 128L
  n1     <- if (length(args) >= 2) as.integer(args[2]) else 4L
  m1     <- if (length(args) >= 3) as.integer(args[3]) else 3L
  n2     <- if (length(args) >= 4) as.integer(args[4]) else 0L
  m2     <- if (length(args) >= 5) as.integer(args[5]) else 0L
  alpha2 <- if (length(args) >= 6) as.numeric(args[6]) else 0.0

  modes <- list(list(n = n1, m = m1, alpha = 1.0))
  if (n2 > 0L && m2 > 0L && alpha2 != 0) {
    modes[[length(modes) + 1L]] <- list(n = n2, m = m2, alpha = alpha2)
  }

  mapping <- cymatic_mapping(grid_n = grid_n, modes = modes)

  # Write to phase4/cymatic/ (audit Tier 4: stop polluting repo root).
  out_dir <- "phase4/cymatic"
  if (!dir.exists(out_dir)) {
    # Fall back to CWD if invoked from inside phase4/cymatic/ already.
    out_dir <- "."
  }
  csv_path <- file.path(out_dir, "cymatic_mapping.csv")
  rds_path <- file.path(out_dir, "cymatic_mapping.rds")
  write_mapping_table(mapping, csv_path)
  saveRDS(mapping, rds_path)
  cat(sprintf("[cymatic] wrote %s\n[cymatic] wrote %s\n", csv_path, rds_path))
}
