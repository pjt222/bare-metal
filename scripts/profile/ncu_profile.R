#!/usr/bin/env Rscript
# ncu_profile.R - Wrap `ncu` for one kernel + bench, emit CSV + markdown row.
#
# Usage:
#   Rscript scripts/profile/ncu_profile.R \
#       --kernel flash_attn_br16_v2_pipeline \
#       --bench phase3/flash_attention/bench_v2_variants \
#       --args "1024 8 8" \
#       --label "FA v2 pipeline (seq=1024, b=8, h=8)" \
#       --launch-skip 5 --launch-count 1 \
#       --out results/ncu/fa_v2_pipeline.csv
#
# Permissions: NCU needs GPU performance counters enabled. On WSL2/Windows host:
#   NVIDIA Control Panel -> Desktop menu -> Enable Developer Settings ->
#   Manage GPU Performance Counters -> "Allow access to all users". Reboot.
#
# On Linux native:
#   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

# (no library() loads needed — base R only)

# ----------------------------------------------------------------------
# Default metric set per issue #89 acceptance criteria.
# Each row: ncu metric name, short column label, higher_is_better flag.
# ----------------------------------------------------------------------
METRICS <- list(
  c("sm__warps_active.avg.pct_of_peak_sustained_active",
    "occupancy_pct", "TRUE"),
  c("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "tc_util_pct", "TRUE"),
  c("l1tex__t_sector_hit_rate.pct",
    "l1_hit_pct", "TRUE"),
  c("lts__t_sector_hit_rate.pct",
    "l2_hit_pct", "TRUE"),
  c("dram__bytes_read.sum.per_second",
    "dram_read_bw", "TRUE"),
  c("dram__bytes_write.sum.per_second",
    "dram_write_bw", "TRUE"),
  c("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "load_coalesce_bytes", "TRUE"),
  # Stall reason histogram
  c("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_long_sb", "FALSE"),
  c("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "stall_short_sb", "FALSE"),
  c("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "stall_wait", "FALSE"),
  c("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "stall_mio", "FALSE"),
  c("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "stall_lg_throttle", "FALSE"),
  c("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "stall_barrier", "FALSE"),
  c("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "stall_math_throttle", "FALSE"),
  c("smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "stall_tex_throttle", "FALSE"),
  # Roofline inputs (#92): byte totals + duration for measured OI.
  c("dram__bytes_read.sum",
    "dram_read_bytes", "FALSE"),
  c("dram__bytes_write.sum",
    "dram_write_bytes", "FALSE"),
  c("lts__t_bytes.sum",
    "l2_bytes", "FALSE"),
  c("gpu__time_duration.sum",
    "duration_ns", "FALSE"),
  # HMMA / ALU counts for compute-side roofline.
  c("smsp__inst_executed_pipe_tensor_op_hmma.sum",
    "hmma_count", "TRUE"),
  c("smsp__inst_executed_pipe_tensor.sum",
    "tensor_count", "TRUE"),    # HMMA + IMMA + ...
  c("smsp__inst_executed_pipe_alu.sum",
    "alu_count", "TRUE")
)

CUDA_BIN <- "/usr/local/cuda/bin"
NCU      <- file.path(CUDA_BIN, "ncu")

# WSL GPU library path. R prepends its own libdir to LD_LIBRARY_PATH which
# masks /usr/lib/wsl/lib where libcuda.so is exposed. Subprocess fails with
# 'no CUDA-capable device is detected' unless we override this.
WSL_CUDA_LIB <- "/usr/lib/wsl/lib"

ensure_path <- function() {
  cur_path <- Sys.getenv("PATH")
  if (!grepl(CUDA_BIN, cur_path, fixed = TRUE)) {
    Sys.setenv(PATH = paste(CUDA_BIN, cur_path, sep = ":"))
  }
  if (dir.exists(WSL_CUDA_LIB)) {
    cur_ld <- Sys.getenv("LD_LIBRARY_PATH")
    if (!grepl(WSL_CUDA_LIB, cur_ld, fixed = TRUE)) {
      Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur_ld))
                                     paste(WSL_CUDA_LIB, cur_ld, sep = ":")
                                   else WSL_CUDA_LIB)
    }
  }
}

# ----------------------------------------------------------------------
# Argument parsing (base R)
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  defaults <- list(
    kernel       = NULL,
    bench        = NULL,
    args         = "",
    label        = NULL,
    launch_skip  = 5L,
    launch_count = 1L,
    out          = "results/ncu/profile.csv",
    dry_run      = FALSE
  )
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    take_val <- function() argv[i + 1]
    if      (a == "--kernel")        { defaults$kernel       <- take_val();        i <- i + 2 }
    else if (a == "--bench")         { defaults$bench        <- take_val();        i <- i + 2 }
    else if (a == "--args")          { defaults$args         <- take_val();        i <- i + 2 }
    else if (a == "--label")         { defaults$label        <- take_val();        i <- i + 2 }
    else if (a == "--launch-skip")   { defaults$launch_skip  <- as.integer(take_val()); i <- i + 2 }
    else if (a == "--launch-count")  { defaults$launch_count <- as.integer(take_val()); i <- i + 2 }
    else if (a == "--out")           { defaults$out          <- take_val();        i <- i + 2 }
    else if (a == "--dry-run")       { defaults$dry_run      <- TRUE;              i <- i + 1 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript scripts/profile/ncu_profile.R --kernel KNAME --bench PATH",
          "                                     --label LABEL [--args 'ARGS']",
          "                                     [--launch-skip N] [--launch-count M]",
          "                                     [--out PATH] [--dry-run]",
          sep = "\n")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  for (req in c("kernel", "bench", "label")) {
    if (is.null(defaults[[req]])) stop("missing required arg: --", req)
  }
  defaults
}

# ----------------------------------------------------------------------
# NCU runner
# ----------------------------------------------------------------------
run_ncu <- function(kernel, bench, bench_args,
                    launch_skip, launch_count) {
  metric_arg <- paste(vapply(METRICS, `[`, character(1), 1), collapse = ",")
  cmd_args <- c(
    "--csv",
    "--kernel-name", kernel,
    "--launch-skip", as.character(launch_skip),
    "--launch-count", as.character(launch_count),
    "--metrics", metric_arg,
    "--target-processes", "all",
    bench,
    bench_args
  )
  out <- tryCatch(
    system2(NCU, cmd_args, stdout = TRUE, stderr = TRUE),
    error = function(e) {
      stop("ncu invocation failed: ", conditionMessage(e))
    }
  )
  status <- attr(out, "status")
  if (!is.null(status) && status != 0L) {
    cat(out, sep = "\n", file = stderr())
    stop(sprintf("ncu failed (rc=%d). Check ERR_NVGPUCTRPERM if counter permission error.",
                 status))
  }
  out
}

# ----------------------------------------------------------------------
# Parse NCU CSV: collapse multiple captured launches by averaging.
# Returns a named list: metric -> list(value, unit)
# ----------------------------------------------------------------------
parse_ncu_csv <- function(raw) {
  # Locate header line (first one starting with "ID" or '"ID"')
  hdr_idx <- which(grepl('^"?ID"?(,|$)', raw))
  if (!length(hdr_idx)) stop("Could not find CSV header in ncu output")
  body <- raw[hdr_idx[1]:length(raw)]
  body_text <- paste(body, collapse = "\n")

  con <- textConnection(body_text)
  on.exit(close(con), add = TRUE)
  df <- tryCatch(
    read.csv(con, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) stop("CSV parse failed: ", conditionMessage(e))
  )
  if (!"Metric Name" %in% names(df) || !"Metric Value" %in% names(df)) {
    stop("Expected 'Metric Name' and 'Metric Value' columns in ncu CSV")
  }

  out <- list()
  for (metric in unique(df[["Metric Name"]])) {
    if (is.na(metric) || metric == "") next
    sel <- df[df[["Metric Name"]] == metric, , drop = FALSE]
    raw_vals <- gsub(",", "", sel[["Metric Value"]], fixed = TRUE)
    nums <- suppressWarnings(as.numeric(raw_vals))
    finite_v <- nums[is.finite(nums)]
    avg <- if (length(finite_v)) mean(finite_v) else NA_real_
    unit <- if ("Metric Unit" %in% names(sel)) sel[["Metric Unit"]][1] else ""
    if (is.na(unit)) unit <- ""
    out[[metric]] <- list(value = avg, unit = unit)
  }
  out
}

# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------
fmt_val <- function(v) {
  if (is.na(v)) return("NaN")
  # Match Python's '%.4g' formatting (no width padding, %g-style precision).
  sprintf("%.4g", v)
}

# Quote a CSV cell if it contains comma, double-quote, or newline.
# Doubles internal quotes per RFC 4180.
csv_escape <- function(x) {
  if (grepl('[,"\n]', x)) {
    paste0('"', gsub('"', '""', x, fixed = TRUE), '"')
  } else x
}

write_csv_row <- function(path, label, parsed) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  is_new <- !file.exists(path)
  con <- file(path, "a")
  on.exit(close(con), add = TRUE)

  if (is_new) {
    cols <- c("label", vapply(METRICS, `[`, character(1), 2))
    writeLines(paste(cols, collapse = ","), con)
  }
  cells <- c(csv_escape(label),
             vapply(METRICS, function(m) {
               entry <- parsed[[m[1]]]
               if (is.null(entry)) "NaN" else fmt_val(entry$value)
             }, character(1)))
  writeLines(paste(cells, collapse = ","), con)
}

print_markdown <- function(label, parsed) {
  cat(sprintf("\n### %s\n\n", label))
  cat("| metric | value |\n")
  cat("|---|---|\n")
  for (m in METRICS) {
    entry <- parsed[[m[1]]]
    val <- if (is.null(entry)) NA_real_ else entry$value
    unit <- if (is.null(entry)) "" else entry$unit
    v_str <- fmt_val(val)
    if (nzchar(unit)) v_str <- paste(v_str, unit)
    cat(sprintf("| `%s` | %s |\n", m[2], v_str))
  }
}

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
main <- function() {
  ensure_path()
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (!file.exists(args$bench) ||
      file.access(args$bench, mode = 1L) != 0L) {
    stop(sprintf("Bench binary not found or not executable: %s", args$bench))
  }

  bench_args <- if (nzchar(args$args)) strsplit(args$args, "\\s+")[[1]] else character(0)

  if (args$dry_run) {
    metric_arg <- paste(vapply(METRICS, `[`, character(1), 1), collapse = ",")
    cmd <- c(NCU, "--csv", "--kernel-name", args$kernel,
             "--launch-skip", args$launch_skip,
             "--launch-count", args$launch_count,
             "--metrics", metric_arg,
             "--target-processes", "all",
             args$bench, bench_args)
    cat(paste(cmd, collapse = " "), "\n")
    return(invisible())
  }

  raw <- run_ncu(args$kernel, args$bench, bench_args,
                 args$launch_skip, args$launch_count)
  parsed <- parse_ncu_csv(raw)
  print_markdown(args$label, parsed)
  write_csv_row(args$out, args$label, parsed)
  cat(sprintf("\nAppended row to %s\n", args$out))
}

if (sys.nframe() == 0L) main()
