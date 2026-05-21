#!/usr/bin/env Rscript
# scripts/probe/probe_clock_lock.R
#
# Probe whether `nvidia-smi -lgc` (locked graphics clock) actually
# works on this machine. This is the open question that decides
# whether clock-locking is a usable lever for the benchmark pipeline
# (see docs/benchmark_methodology.md and docs/rebaseline_protocol.md).
#
# WSL2 passthrough can ACCEPT the -lgc command syntactically and then
# silently no-op. So a zero exit code is NOT proof. This probe sets a
# lock, reads the clock back, and only reports success if the clock
# actually moved to the requested value.
#
# Privileged: nvidia-smi clock control needs root. Run with sudo:
#   sudo Rscript scripts/probe/probe_clock_lock.R
#
# The probe always resets the clock (-rgc) on exit, including on
# error, so it never leaves the GPU clamped. All console output is
# also tee'd to scripts/probe/probe_clock_lock.log.

.WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(.WSL_CUDA_LIB) &&
    !grepl(.WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  .cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(.cur))
                                  paste(.WSL_CUDA_LIB, .cur, sep = ":")
                                else .WSL_CUDA_LIB)
}

# Resolve the nvidia-smi binary by absolute path. Under `sudo` the
# PATH is reset to a secure_path that does NOT include the WSL CUDA
# directory, so a bare "nvidia-smi" name fails with exit 127. Prefer
# whatever is on PATH; fall back to the known WSL location.
.NVSMI <- local({
  on_path <- Sys.which("nvidia-smi")
  if (nzchar(on_path)) return(unname(on_path))
  wsl <- file.path(.WSL_CUDA_LIB, "nvidia-smi")
  if (file.exists(wsl)) wsl else "nvidia-smi"
})

# Log file beside this script, resolved from the Rscript --file= arg.
.LOG <- local({
  fa <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(fa)) {
    file.path(dirname(normalizePath(sub("^--file=", "", fa[[1]]))),
              "probe_clock_lock.log")
  } else {
    "probe_clock_lock.log"
  }
})

# Run nvidia-smi; return list(out=lines, status=exit code).
# An exit 127 means the binary itself could not be launched.
.smi <- function(args) {
  out <- tryCatch(
    suppressWarnings(system2(.NVSMI, args, stdout = TRUE, stderr = TRUE)),
    error = function(e) {
      res <- character(0)
      attr(res, "status") <- 127L
      res
    }
  )
  st <- attr(out, "status"); if (is.null(st)) st <- 0L
  list(out = out, status = as.integer(st))
}

# Current SM clock in MHz, or NA.
read_sm_clock <- function() {
  r <- .smi(c("--query-gpu=clocks.current.sm",
              "--format=csv,noheader,nounits"))
  if (r$status != 0L || !length(r$out)) return(NA_integer_)
  suppressWarnings(as.integer(trimws(r$out[[1]])))
}

# Supported graphics clocks, descending.
supported_clocks <- function() {
  r <- .smi(c("-q", "-d", "SUPPORTED_CLOCKS"))
  if (r$status != 0L) return(integer(0))
  gfx <- grep("^\\s*Graphics\\s*:", r$out, perl = TRUE, value = TRUE)
  mhz <- suppressWarnings(as.integer(sub(".*:\\s*([0-9]+)\\s*MHz.*", "\\1", gfx)))
  sort(unique(mhz[!is.na(mhz)]), decreasing = TRUE)
}

# Returns the probe exit status (0 works / 1 no nvidia-smi /
# 2 -lgc rejected / 3 silent no-op). The caller passes it to quit()
# AFTER this function returns, so on.exit (the -rgc reset) is
# guaranteed to run — quit() does not reliably run on.exit handlers
# of enclosing frames.
main <- function() {
  logcon <- file(.LOG, open = "wt")
  sink(logcon, split = TRUE)
  on.exit({ sink(); close(logcon) }, add = TRUE)

  cat("=== clock-lock probe (nvidia-smi -lgc) ===\n\n")
  cat(sprintf("nvidia-smi : %s\n", .NVSMI))
  cat(sprintf("log file   : %s\n\n", .LOG))

  clocks <- supported_clocks()
  if (!length(clocks)) {
    cat("ERROR: cannot read supported clocks — nvidia-smi unavailable.\n")
    cat(sprintf("  Tried binary: %s\n", .NVSMI))
    cat("  If running under sudo, the PATH is reset — this probe\n")
    cat("  resolves nvidia-smi by absolute path, so a failure here\n")
    cat("  means the driver itself is not responding.\n")
    return(1L)
  }

  # Target a mid-range supported clock — distinct from idle and from
  # max boost, so a readback can unambiguously confirm the lock.
  target <- clocks[[ceiling(length(clocks) / 2)]]
  cat(sprintf("Supported SM clocks: %d MHz (max) .. %d MHz (min), %d steps\n",
              max(clocks), min(clocks), length(clocks)))
  cat(sprintf("Lock target        : %d MHz\n\n", target))

  before <- read_sm_clock()
  cat(sprintf("SM clock before lock : %s MHz\n",
              if (is.na(before)) "?" else before))

  # Attempt the lock.
  lock <- .smi(c("-lgc", sprintf("%d,%d", target, target)))
  cat(sprintf("\n-lgc exit status     : %d\n", lock$status))
  if (length(lock$out)) cat("  ", paste(lock$out, collapse = "\n   "), "\n", sep = "")

  reset_done <- FALSE
  reset <- function() {
    if (reset_done) return(invisible())
    r <- .smi("-rgc")
    cat(sprintf("\n-rgc reset status    : %d%s\n", r$status,
                if (r$status == 0L) " (clock unlocked)" else " (RESET MAY HAVE FAILED — check manually)"))
    reset_done <<- TRUE
  }
  on.exit(reset(), add = TRUE, after = FALSE)

  if (lock$status != 0L) {
    cat("\nVERDICT: -lgc was REJECTED (non-zero exit).\n")
    if (any(grepl("Permission|root|privile", lock$out, ignore.case = TRUE)))
      cat("  Cause: insufficient privilege — re-run with sudo.\n")
    else
      cat("  Cause: not permitted on this platform (likely WSL passthrough).\n")
    cat("  Clock-locking is NOT available; the pipeline must rely on\n")
    cat("  cooldown + retry instead. See docs/benchmark_methodology.md.\n")
    return(2L)
  }

  # Exit 0 is not proof. Sample the clock a few times and check it
  # actually settled at the target.
  cat("\nReading clock back (lock with exit 0 — verifying it took effect):\n")
  samples <- integer(0)
  for (i in 1:5) {
    Sys.sleep(0.4)
    s <- read_sm_clock()
    samples <- c(samples, s)
    cat(sprintf("  sample %d: %s MHz\n", i, if (is.na(s)) "?" else s))
  }
  valid <- samples[!is.na(samples)]
  # Locked clock can read slightly off the request on an idle GPU;
  # accept within one clock step (~15 MHz on Ampere).
  tol <- 20L
  locked_ok <- length(valid) > 0 && all(abs(valid - target) <= tol)

  cat("\n")
  if (locked_ok) {
    cat(sprintf("VERDICT: clock-lock WORKS — SM clock held at ~%d MHz (target %d).\n",
                as.integer(median(valid)), target))
    cat("  Clock-locking is a usable lever for reproducible benchmarking.\n")
    return(0L)
  } else {
    cat("VERDICT: -lgc returned exit 0 but the clock did NOT lock.\n")
    cat(sprintf("  Requested %d MHz; observed %s MHz.\n", target,
                if (length(valid)) paste(range(valid), collapse = "-") else "?"))
    cat("  This is the WSL silent-no-op case: the command is accepted\n")
    cat("  but ignored. Clock-locking is NOT effective here — rely on\n")
    cat("  cooldown + retry. See docs/benchmark_methodology.md.\n")
    return(3L)
  }
}

if (sys.nframe() == 0L) quit(status = main(), runLast = FALSE)
