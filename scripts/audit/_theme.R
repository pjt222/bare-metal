# scripts/audit/_theme.R
# ----------------------------------------------------------------------
# Project-wide ggplot2 theme + viridis palette defaults.
#
# Source from any plotting script:
#   source(file.path(REPO_ROOT, "scripts/audit/_theme.R"))
#   p + theme_baremetal() + scale_fill_bm_seq()
#
# Provides:
#   theme_baremetal(base_size = 11)   - common theme (theme_minimal base)
#   scale_*_bm_seq(...)               - sequential viridis (option "viridis")
#   scale_*_bm_disc(...)              - discrete viridis (option "viridis")
#   scale_*_bm_div(midpoint = 1.0)    - diverging: viridis-end purple <-> yellow
#                                       through white. Used for ratio plots
#                                       centered on 1.0 (speedup vs baseline).
#   bm_save(p, path, w, h, ...)       - ggsave with project dpi default
#
# Design notes
#
#   * theme_minimal as base because grid lines on white read better than the
#     boxed defaults; baseline plots are mostly cartesian bars / heatmaps.
#   * viridis for everything sequential / discrete - perceptually uniform,
#     colour-blind friendly, prints in greyscale.
#   * "Diverging viridis" is a contradiction (viridis is sequential by design).
#     We approximate it with gradient2(low = viridis_purple, mid = white,
#     high = viridis_yellow) so the look stays in the same colour family.
#   * dpi defaults to 130 because every figure here ends up in markdown at
#     ~600px wide, and 130 dpi gives sharp text without inflating PNG size.
# ----------------------------------------------------------------------

suppressPackageStartupMessages({
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 required")
  }
  if (!requireNamespace("viridisLite", quietly = TRUE)) {
    stop("viridisLite required")
  }
})

# Project-wide constants pulled from one of the existing plotting scripts
# so we don't change rendered sizes accidentally.
BM_DPI               <- 130L
BM_BASE_SIZE_DEFAULT <- 11L
BM_FONT_FAMILY       <- ""   # "" = default sans, keeps PNGs portable

# Viridis endpoint colours (opt "viridis"). Cached to avoid loading the
# package on every plot call.
BM_VIRIDIS_LOW    <- viridisLite::viridis(3)[1]  # deep purple "#440154"
BM_VIRIDIS_MID    <- viridisLite::viridis(3)[2]  # teal/green  "#21908C"
BM_VIRIDIS_HIGH   <- viridisLite::viridis(3)[3]  # yellow      "#FDE725"

# Dark theme. Same minimal base, inverted background + text + grid colours.
# Background "grey10" (#1a1a1a) reads near-black without crushing to pure black,
# so antialiased text edges still blend. Grids visible but recessive.
BM_DARK_BG_PLOT   <- "grey10"   # outer canvas
BM_DARK_BG_PANEL  <- "grey14"   # plot panel, slightly lighter for separation
BM_DARK_BG_STRIP  <- "grey20"   # facet strip
BM_DARK_FG        <- "grey90"   # primary text / axis
BM_DARK_FG_MUTED  <- "grey70"   # subtitle
BM_DARK_FG_FAINT  <- "grey60"   # caption
BM_DARK_GRID_MAJ  <- "grey30"
BM_DARK_GRID_MIN  <- "grey22"

theme_baremetal <- function(base_size = BM_BASE_SIZE_DEFAULT,
                            base_family = BM_FONT_FAMILY) {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      plot.background  = ggplot2::element_rect(fill = BM_DARK_BG_PLOT,
                                                colour = NA),
      panel.background = ggplot2::element_rect(fill = BM_DARK_BG_PANEL,
                                                colour = NA),
      plot.title       = ggplot2::element_text(face = "bold",
                                               colour = BM_DARK_FG,
                                               size = base_size + 1),
      plot.subtitle    = ggplot2::element_text(colour = BM_DARK_FG_MUTED,
                                               size = base_size - 1),
      plot.caption     = ggplot2::element_text(colour = BM_DARK_FG_FAINT,
                                               size = base_size - 3,
                                               hjust = 0),
      plot.caption.position = "plot",
      axis.title       = ggplot2::element_text(colour = BM_DARK_FG,
                                               size = base_size - 1),
      axis.text        = ggplot2::element_text(colour = BM_DARK_FG_MUTED,
                                               size = base_size - 2),
      legend.title     = ggplot2::element_text(colour = BM_DARK_FG,
                                               size = base_size - 1),
      legend.text      = ggplot2::element_text(colour = BM_DARK_FG_MUTED,
                                               size = base_size - 2),
      legend.background = ggplot2::element_rect(fill = BM_DARK_BG_PLOT,
                                                 colour = NA),
      legend.key       = ggplot2::element_rect(fill = BM_DARK_BG_PLOT,
                                                colour = NA),
      legend.position  = "right",
      panel.grid.minor = ggplot2::element_line(colour = BM_DARK_GRID_MIN,
                                               linewidth = 0.25),
      panel.grid.major = ggplot2::element_line(colour = BM_DARK_GRID_MAJ,
                                               linewidth = 0.35),
      strip.text       = ggplot2::element_text(face = "bold",
                                               colour = BM_DARK_FG,
                                               size = base_size - 1),
      strip.background = ggplot2::element_rect(fill = BM_DARK_BG_STRIP,
                                               colour = NA)
    )
}

# Sequential viridis - continuous data
scale_fill_bm_seq <- function(...,
                              option = "viridis",
                              direction = 1) {
  ggplot2::scale_fill_viridis_c(option = option, direction = direction, ...)
}
scale_color_bm_seq <- function(...,
                               option = "viridis",
                               direction = 1) {
  ggplot2::scale_color_viridis_c(option = option, direction = direction, ...)
}
scale_colour_bm_seq <- scale_color_bm_seq

# Discrete viridis - categorical data
scale_fill_bm_disc <- function(...,
                               option = "viridis",
                               direction = 1,
                               begin = 0.05,
                               end   = 0.95) {
  ggplot2::scale_fill_viridis_d(option = option, direction = direction,
                                begin = begin, end = end, ...)
}
scale_color_bm_disc <- function(...,
                                option = "viridis",
                                direction = 1,
                                begin = 0.05,
                                end   = 0.95) {
  ggplot2::scale_color_viridis_d(option = option, direction = direction,
                                 begin = begin, end = end, ...)
}
scale_colour_bm_disc <- scale_color_bm_disc

# Diverging palette - ratio plots centred on `midpoint` (e.g. speedup vs
# baseline = 1.0). Endpoints from viridis to keep visual coherence.
# Diverging midpoint flipped to a dark grey so the neutral region blends with
# the dark panel background instead of punching a white hole through it.
scale_fill_bm_div <- function(midpoint = 1.0, ...,
                              low  = BM_VIRIDIS_LOW,
                              mid  = "grey20",
                              high = BM_VIRIDIS_HIGH) {
  ggplot2::scale_fill_gradient2(midpoint = midpoint,
                                low = low, mid = mid, high = high, ...)
}
scale_color_bm_div <- function(midpoint = 1.0, ...,
                               low  = BM_VIRIDIS_LOW,
                               mid  = "grey20",
                               high = BM_VIRIDIS_HIGH) {
  ggplot2::scale_color_gradient2(midpoint = midpoint,
                                 low = low, mid = mid, high = high, ...)
}
scale_colour_bm_div <- scale_color_bm_div

# Project-default ggsave wrapper. Forces dpi unless the caller overrides.
bm_save <- function(plot, filename, width, height, dpi = BM_DPI, ...) {
  ggplot2::ggsave(filename = filename, plot = plot,
                  width = width, height = height, dpi = dpi, ...)
}
