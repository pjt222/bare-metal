#!/usr/bin/env Rscript
# bench_flash_all.R - Unified Flash Attention benchmark harness.
# Mirrors bench_flash_all.py.
#
# Discovers and runs all FA bench executables in kernels/attention/flash_attention/,
# producing a comparison table.
#
# Usage:
#   Rscript scripts/bench/bench_flash_all.R                    # default 1024 8 8
#   Rscript scripts/bench/bench_flash_all.R 1024 8 8
#   Rscript scripts/bench/bench_flash_all.R --build 1024 8 8

# (uses base R only — no library() loads needed)

# WSL CUDA libpath fix.
.WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(.WSL_CUDA_LIB) &&
    !grepl(.WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  .cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(.cur))
                                  paste(.WSL_CUDA_LIB, .cur, sep = ":")
                                else .WSL_CUDA_LIB)
}

REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}
FLASH_DIR <- file.path(REPO_ROOT, "phase3", "flash_attention")

discover_benches <- function() {
  files <- list.files(FLASH_DIR, full.names = TRUE)
  hits <- list()
  for (f in files) {
    bn <- basename(f)
    # Match Python: starts with 'bench', regular file, no extension.
    if (startsWith(bn, "bench") && file.exists(f) && !file.info(f)$isdir &&
        !grepl("\\.", bn)) {
      hits[[bn]] <- f
    }
  }
  hits
}

run_bench <- function(exe_path, args) {
  out <- tryCatch(
    suppressWarnings(system2(exe_path, args, stdout = TRUE, stderr = TRUE,
                             timeout = 120)),
    error = function(e) {
      attr(character(0), "status") <- 1L
      character(0)
    }
  )
  status <- attr(out, "status")
  rc <- if (is.null(status)) 0L else as.integer(status)
  output <- paste(out, collapse = "\n")

  m <- list(raw = output, returncode = rc)
  hit <- regmatches(output,
                    regexec("([0-9.]+)\\s*ms.*?([0-9][0-9,.]*)\\s*GFLOPS",
                            output, perl = TRUE, ignore.case = TRUE))[[1]]
  if (length(hit) >= 3) {
    m$ms     <- as.numeric(hit[2])
    m$gflops <- as.numeric(gsub(",", "", hit[3], fixed = TRUE))
  }
  m$check <- if (grepl("PASS", output, fixed = TRUE)) "PASS"
             else if (grepl("FAIL", output, fixed = TRUE)) "FAIL"
             else "?"
  m
}

print_table <- function(results, seq_len, batch, heads) {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("  Flash Attention Comparison  (seq=%d, batch=%d, heads=%d)\n",
              seq_len, batch, heads))
  cat(strrep("=", 70), "\n", sep = "")
  cat(sprintf("  %-28s %10s %12s %8s\n", "Variant", "Time(ms)", "GFLOPS", "Check"))
  cat(sprintf("  %-28s %10s %12s %8s\n",
              strrep("-", 28), strrep("-", 10), strrep("-", 12), strrep("-", 8)))

  ms_vec <- vapply(results, function(r) if (!is.null(r$ms)) r$ms else NA_real_,
                   numeric(1))
  best_ms <- if (any(is.finite(ms_vec))) min(ms_vec, na.rm = TRUE) else 0

  for (r in results) {
    if (is.null(r$ms)) {
      check <- if (!is.null(r$check)) r$check else "SKIP"
      cat(sprintf("  %-28s %10s %12s %8s\n", r$name, "-", "-", check))
      next
    }
    marker <- if (abs(r$ms - best_ms) < 0.001) "*" else " "
    cat(sprintf(" %s%-27s %10.3f %12.0f %8s\n",
                marker, r$name, r$ms,
                if (!is.null(r$gflops)) r$gflops else 0,
                if (!is.null(r$check)) r$check else "?"))
  }
  cat(strrep("=", 70), "\n", sep = "")
  cat("  * = fastest variant\n\n")
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) && argv[1] %in% c("-h", "--help")) {
    cat("Usage: bench_flash_all.R [SEQ_LEN] [BATCH] [HEADS] [--build]\n")
    cat("Default: 1024 8 8\n")
    quit(status = 0)
  }
  do_build <- "--build" %in% argv
  argv <- argv[argv != "--build"]
  seq_len <- if (length(argv) >= 1) as.integer(argv[1]) else 1024L
  batch   <- if (length(argv) >= 2) as.integer(argv[2]) else 8L
  heads   <- if (length(argv) >= 3) as.integer(argv[3]) else 8L

  benches <- discover_benches()
  if (!length(benches)) {
    cat("No bench executables found in kernels/attention/flash_attention/\n")
    if (do_build) {
      cat("Attempting to build...\n")
      system2("make", c("-C", dirname(FLASH_DIR), "phase3"))
      benches <- discover_benches()
    }
    if (!length(benches)) {
      cat("Run: make phase3\n"); quit(status = 1)
    }
  }

  bench_args <- as.character(c(seq_len, batch, heads))

  preferred_order <- c(
    "bench", "bench_br16", "bench_br16_regpv",
    "bench_br16_bc128", "bench_br16_pipeline",
    "bench_fused", "bench_persistent",
    "bench_split_q", "bench_wmma"
  )

  ordered <- list()
  for (nm in preferred_order) {
    if (!is.null(benches[[nm]])) {
      ordered[[nm]] <- benches[[nm]]
      benches[[nm]] <- NULL
    }
  }
  remaining <- sort(names(benches))
  for (nm in remaining) ordered[[nm]] <- benches[[nm]]

  cat(sprintf("Running %d Flash Attention variants...\n", length(ordered)))
  results <- list()
  for (nm in names(ordered)) {
    cat(sprintf("\n--- %s ---\n", nm))
    exe <- ordered[[nm]]
    if (!file.exists(exe)) {
      results[[length(results) + 1L]] <- list(name = nm, check = "NOT_FOUND")
      next
    }
    metrics <- run_bench(exe, bench_args)
    metrics$name <- nm
    results[[length(results) + 1L]] <- metrics
    if (!is.null(metrics$ms)) {
      cat(sprintf("  %.3f ms  %.0f GFLOPS  [%s]\n",
                  metrics$ms,
                  if (!is.null(metrics$gflops)) metrics$gflops else 0,
                  metrics$check))
    }
  }

  print_table(results, seq_len, batch, heads)

  fails <- sum(vapply(results,
                      function(r) isTRUE(r$check == "FAIL"),
                      logical(1)))
  quit(status = if (fails > 0L) 1L else 0L)
}

if (sys.nframe() == 0L) main()
