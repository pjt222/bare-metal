#!/usr/bin/env Rscript
# scripts/probe/probe_clock_lock.R
#
# Probe whether `nvidia-smi -lgc` (locked graphics clock) actually
# works on this machine. This is the open question that decides
# whether clock-locking is a usable lever for the benchmark pipeline
# (see docs/benchmark_methodology.md).
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
# error, so it never leaves the GPU clamped.

.WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(.WSL_CUDA_LIB) &&
    !grepl(.WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  .cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(.cur))
                                  paste(.WSL_CUDA_LIB, .cur, sep = ":")
                                else .WSL_CUDA_LIB)
}

# Run nvidia-smi; return list(out=lines, status=exit code).
.smi <- function(args) {
  out <- tryCatch(
    suppressWarnings(system2("nvidia-smi", args, stdout = TRUE, stderr = TRUE)),
    error = function(e) { attr(character(0), "status") <- 127L; character(0) }
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

main <- function() {
  cat("=== clock-lock probe (nvidia-smi -lgc) ===\n\n")

  clocks <- supported_clocks()
  if (!length(clocks)) {
    cat("ERROR: cannot read supported clocks — nvidia-smi unavailable.\n")
    quit(status = 1)
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
  on.exit(reset(), add = TRUE)

  if (lock$status != 0L) {
    cat("\nVERDICT: -lgc was REJECTED (non-zero exit).\n")
    if (any(grepl("Permission|root|privile", lock$out, ignore.case = TRUE)))
      cat("  Cause: insufficient privilege — re-run with sudo.\n")
    else
      cat("  Cause: not permitted on this platform (likely WSL passthrough).\n")
    cat("  Clock-locking is NOT available; the pipeline must rely on\n")
    cat("  cooldown + retry instead. See docs/benchmark_methodology.md.\n")
    quit(status = 2)
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
    quit(status = 0)
  } else {
    cat("VERDICT: -lgc returned exit 0 but the clock did NOT lock.\n")
    cat(sprintf("  Requested %d MHz; observed %s MHz.\n", target,
                if (length(valid)) paste(range(valid), collapse = "-") else "?"))
    cat("  This is the WSL silent-no-op case: the command is accepted\n")
    cat("  but ignored. Clock-locking is NOT effective here — rely on\n")
    cat("  cooldown + retry. See docs/benchmark_methodology.md.\n")
    quit(status = 3)
  }
}

if (sys.nframe() == 0L) main()
