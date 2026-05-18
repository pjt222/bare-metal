#!/usr/bin/env Rscript
# cymatic_visualize.R
#
# Visualize one or two cymatic memory mappings produced by cymatic_mapping.R.
# Default mode writes field/region/address/size panels for one mapping.
# Overlay mode writes a domain comparison plot for two mappings.
#
# Usage:
#   Rscript scripts/cymatic/cymatic_visualize.R cymatic_mapping.rds [out_prefix]
#   Rscript scripts/cymatic/cymatic_visualize.R --overlay disc.rds square.rds [out_prefix]

suppressPackageStartupMessages({
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("install.packages('ggplot2')")
  }
  library(ggplot2)
})

# Source mapping helpers; assume cwd is project root or scripts/.
for (p in c("scripts/cymatic/cymatic_mapping.R", "cymatic_mapping.R",
            "./cymatic_mapping.R")) {
  if (file.exists(p)) { source(p); break }
}

# Project-wide theme + viridis palettes (audit follow-up).
for (p in c("scripts/audit/_theme.R",
            "../audit/_theme.R",
            "../../scripts/audit/_theme.R")) {
  if (file.exists(p)) { source(p); break }
}

plot_field <- function(mapping, title = "Cymatic field u(r,θ)") {
  grid <- mapping$grid
  df <- data.frame(
    x = as.vector(grid$x),
    y = as.vector(grid$y),
    u = as.vector(mapping$field),
    inside = as.vector(grid$inside)
  )
  df <- df[df$inside, ]
  # Field is a signed standing wave centred at 0; viridis-rooted diverging
  # palette (purple <-> white <-> yellow) preserves sign legibility while
  # staying in the project's colour family.
  ggplot(df, aes(x = x, y = y, fill = u)) +
    geom_raster() +
    scale_fill_bm_div(midpoint = 0) +
    coord_equal() +
    labs(title = title, x = NULL, y = NULL, fill = "u") +
    theme_baremetal()
}

plot_regions <- function(mapping,
                          title = sprintf("Antinode regions (%d)", mapping$n_regions)) {
  grid <- mapping$grid
  rid_orig <- as.vector(mapping$label)
  rid_new <- mapping$region_order$new_id[
    match(rid_orig, mapping$region_order$region_id)
  ]
  df <- data.frame(
    x = as.vector(grid$x),
    y = as.vector(grid$y),
    region = rid_new,
    inside = as.vector(grid$inside)
  )
  df <- df[df$inside & !is.na(df$region), ]
  # spread region_id over hue for visual distinction (cyclic palette)
  df$hue <- (df$region * 137L) %% mapping$n_regions  # golden-angle-ish dispersion
  # Categorical region IDs: viridis_c with the dispersed `hue` column gives
  # neighbouring regions visibly different colours while staying perceptually
  # uniform. Was rainbow(64) - dropped because rainbow misleads quantitative
  # readers (banding at yellow/cyan + non-uniform luminance).
  ggplot(df, aes(x = x, y = y, fill = hue)) +
    geom_raster() +
    scale_fill_bm_seq(option = "turbo") +
    coord_equal() +
    labs(title = title, x = NULL, y = NULL, fill = "region (perm)") +
    theme_baremetal() +
    theme(legend.position = "none")
}

plot_addresses <- function(mapping,
                            title = sprintf("Linear address mapping (%d cells)",
                                            mapping$total_cells)) {
  grid <- mapping$grid
  df <- data.frame(
    x    = as.vector(grid$x),
    y    = as.vector(grid$y),
    addr = as.vector(mapping$address)
  )
  df <- df[!is.na(df$addr), ]
  # Linear address: sequential viridis (option "viridis") - was "C" (plasma)
  # to match the project default.
  ggplot(df, aes(x = x, y = y, fill = addr)) +
    geom_raster() +
    scale_fill_bm_seq() +
    coord_equal() +
    labs(title = title, x = NULL, y = NULL, fill = "addr") +
    theme_baremetal()
}

plot_region_size_histogram <- function(mapping,
                                         title = "Region size distribution") {
  ord <- mapping$region_order
  ggplot(ord, aes(x = cell_count)) +
    geom_histogram(bins = 30, fill = BM_VIRIDIS_MID, colour = "white") +
    scale_x_log10() +
    labs(title = title, x = "cells per region (log10)", y = "frequency") +
    theme_baremetal()
}

plot_domain_overlay <- function(disc_mapping, square_mapping,
                                title = "Disc vs inscribed-square active domains") {
  stopifnot(identical(dim(disc_mapping$grid$inside), dim(square_mapping$grid$inside)))
  grid <- disc_mapping$grid
  disc_inside <- as.vector(disc_mapping$grid$inside)
  square_inside <- as.vector(square_mapping$grid$inside)
  cls <- ifelse(disc_inside & square_inside, "shared",
                ifelse(disc_inside, "disc_only", ifelse(square_inside, "square_only", "outside")))
  df <- data.frame(
    x = as.vector(grid$x),
    y = as.vector(grid$y),
    cls = factor(cls, levels = c("shared", "disc_only", "square_only", "outside"))
  )
  df <- df[df$cls != "outside", ]

  ggplot(df, aes(x = x, y = y, fill = cls)) +
    geom_raster() +
    scale_fill_manual(
      values = c(
        shared = BM_VIRIDIS_MID,
        disc_only = BM_VIRIDIS_HIGH,
        square_only = BM_VIRIDIS_LOW
      ),
      breaks = c("shared", "disc_only", "square_only"),
      labels = c("shared cells", "disc-only ring", "square-only")
    ) +
    coord_equal() +
    labs(title = title, x = NULL, y = NULL, fill = NULL) +
    theme_baremetal()
}

plot_quad <- function(mapping, out_prefix) {
  # Always write the four individual PNGs - they are referenced directly
  # from docs/cymatic_memory_mapping.md (cymatic_regions.png, _addresses.png
  # etc). Quad PNG is a bonus combined view when patchwork is available.
  ggsave(paste0(out_prefix, "_field.png"),     plot_field(mapping),
         width = 5, height = 5, dpi = 120)
  ggsave(paste0(out_prefix, "_regions.png"),   plot_regions(mapping),
         width = 5, height = 5, dpi = 120)
  ggsave(paste0(out_prefix, "_addresses.png"), plot_addresses(mapping),
         width = 5, height = 5, dpi = 120)
  ggsave(paste0(out_prefix, "_sizes.png"),     plot_region_size_histogram(mapping),
         width = 5, height = 4, dpi = 120)
  cat(sprintf("[cymatic] wrote 4 individual PNGs with prefix '%s'\n",
              out_prefix))

  if (requireNamespace("patchwork", quietly = TRUE)) {
    library(patchwork)
    p <- (plot_field(mapping) | plot_regions(mapping)) /
         (plot_addresses(mapping) | plot_region_size_histogram(mapping))
    ggsave(paste0(out_prefix, "_quad.png"), p,
           width = 11, height = 10, dpi = 120)
    cat(sprintf("[cymatic] wrote %s_quad.png\n", out_prefix))
  }
}

# ---- CLI -------------------------------------------------------------------

.is_main_viz <- function() {
  if (interactive()) return(FALSE)
  if (sys.nframe() > 1L) return(FALSE)
  invoked <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  if (length(invoked) == 0L) return(TRUE)
  basename(invoked) == "cymatic_visualize.R"
}

if (.is_main_viz()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1 && args[1] == "--overlay") {
    if (length(args) < 3) {
      stop("overlay mode requires: --overlay <disc_rds> <square_rds> [out_prefix]")
    }
    disc_rds <- args[2]
    square_rds <- args[3]
    out_prefix <- if (length(args) >= 4) args[4] else {
      if (dir.exists("docs/figures/cymatic")) "docs/figures/cymatic/cymatic_domain"
      else "cymatic_domain"
    }
    disc_mapping <- readRDS(disc_rds)
    square_mapping <- readRDS(square_rds)
    p <- plot_domain_overlay(disc_mapping, square_mapping)
    ggsave(paste0(out_prefix, "_overlay.png"), p, width = 6, height = 6, dpi = 140)
    cat(sprintf("[cymatic] wrote %s_overlay.png\n", out_prefix))
    quit(save = "no", status = 0)
  }

  rds_path   <- if (length(args) >= 1) args[1] else "cymatic_mapping.rds"
  # Default writes into docs/figures/cymatic/cymatic_*.png to match the
  # Output directory. Override with second positional arg to change.
  default_prefix <- if (dir.exists("docs/figures/cymatic")) {
    "docs/figures/cymatic/cymatic"
  } else {
    "cymatic"
  }
  out_prefix <- if (length(args) >= 2) args[2] else default_prefix
  if (!file.exists(rds_path)) {
    stop(sprintf("missing %s — run scripts/cymatic/cymatic_mapping.R first", rds_path))
  }
  mapping <- readRDS(rds_path)
  plot_quad(mapping, out_prefix)
}
