#!/usr/bin/env Rscript
# compare_reference.R - Join project baselines to locally measured reference
# baselines and print a reproducible comparison table.

library(jsonlite)

.repo_root <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  start <- if (length(fa)) normalizePath(dirname(sub("^--file=", "", fa[1])))
           else normalizePath(getwd())
  cur <- start
  repeat {
    if (file.exists(file.path(cur, ".git")) || file.exists(file.path(cur, "renv.lock"))) break
    parent <- dirname(cur)
    if (parent == cur) break
    cur <- parent
  }
  cur
}

parse_args <- function(argv) {
  out <- list(csv = NULL)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (a == "--csv") {
      out$csv <- argv[i + 1]
      i <- i + 2
    } else if (a %in% c("-h", "--help")) {
      cat("Usage: compare_reference.R [--csv path]\n")
      quit(status = 0)
    } else {
      stop("unknown arg: ", a)
    }
  }
  out
}

metric_value <- function(entry) {
  if (!is.null(entry$gflops)) return(list(value = as.numeric(entry$gflops), unit = "GFLOPS"))
  if (!is.null(entry$tops)) {
    value <- as.numeric(entry$tops)
    # Historical project baselines store INT8 throughput in GOPS-scale numbers
    # (e.g. 20227 for 20.227 TOPS). Normalize to TOPS for human-readable
    # comparison against the local reference pipeline.
    if (!is.na(value) && value > 1000) value <- value / 1000.0
    return(list(value = value, unit = "TOPS"))
  }
  list(value = NA_real_, unit = "UNKNOWN")
}

fmt_metric <- function(value, unit) {
  if (is.na(value)) return("n/a")
  if (unit == "TOPS") return(sprintf("%.2f TOPS", value))
  sprintf("%.0f GFLOPS", value)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  project_path <- file.path(.repo_root, "data", "baselines.json")
  reference_path <- file.path(.repo_root, "data", "reference_baselines.json")

  project <- fromJSON(project_path, simplifyVector = FALSE)
  reference <- fromJSON(reference_path, simplifyVector = FALSE)

  rows <- list()
  for (ref_kernel in names(reference$kernels)) {
    entry <- reference$kernels[[ref_kernel]]
    cfg_names <- setdiff(names(entry), c("exe", "library"))
    for (cfg in cfg_names) {
      ref_cfg <- entry[[cfg]]
      if (is.null(ref_cfg$project_kernel) || is.null(ref_cfg$project_config)) next
      project_entry <- project$kernels[[ref_cfg$project_kernel]]
      if (is.null(project_entry)) next
      project_cfg <- project_entry[[ref_cfg$project_config]]
      if (is.null(project_cfg)) next

      ref_metric <- metric_value(ref_cfg)
      project_metric <- metric_value(project_cfg)
      ratio <- project_metric$value / ref_metric$value
      gap <- ref_metric$value / project_metric$value

      rows[[length(rows) + 1L]] <- data.frame(
        label = if (!is.null(ref_cfg$label)) ref_cfg$label else paste(ref_cfg$project_kernel, cfg),
        library = if (!is.null(entry$library)) entry$library else "reference",
        ours = fmt_metric(project_metric$value, project_metric$unit),
        reference = fmt_metric(ref_metric$value, ref_metric$unit),
        pct_of_reference = sprintf("%.1f%%", 100 * ratio),
        gap = sprintf("%.2fx", gap),
        stringsAsFactors = FALSE
      )
    }
  }

  if (!is.null(reference$unsupported) && length(reference$unsupported)) {
    for (item in reference$unsupported) {
      rows[[length(rows) + 1L]] <- data.frame(
        label = item$label,
        library = if (!is.null(item$library)) item$library else "unavailable",
        ours = "not compared",
        reference = "not measured locally",
        pct_of_reference = "n/a",
        gap = if (!is.null(item$reason)) item$reason else "unavailable",
        stringsAsFactors = FALSE
      )
    }
  }

  out <- do.call(rbind, rows)
  cat("Local reference comparison\n")
  cat(sprintf("Project baselines:   %s\n", project$recorded_date))
  cat(sprintf("Reference baselines: %s\n\n", reference$recorded_date))
  print(out, row.names = FALSE, right = FALSE)

  if (!is.null(args$csv)) {
    write.csv(out, args$csv, row.names = FALSE)
    cat(sprintf("\n[compare_reference] wrote %s\n", args$csv))
  }
}

if (sys.nframe() == 0L) main()
