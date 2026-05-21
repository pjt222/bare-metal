#!/usr/bin/env Rscript
# scripts/probe/probe_gpu_power.R
#
# Read-only probe of the GPU's power and clock envelope. Answers the
# questions that decide the benchmark-pipeline strategy (see
# docs/benchmark_methodology.md):
#
#   - What is the board power limit, and is there any headroom?
#   - Which graphics clocks does the GPU advertise as supported?
#   - Is the GPU throttled right now, and why?
#   - What does nvidia-smi report for GPU/display mode? (WSL caveat)
#
# Read-only: queries nvidia-smi, changes nothing, needs no privilege.
#
# Usage:
#   Rscript scripts/probe/probe_gpu_power.R            # human report
#   Rscript scripts/probe/probe_gpu_power.R --json     # machine output

# WSL CUDA passthrough: nvidia-smi needs libnvidia-ml.so from
# /usr/lib/wsl/lib; the R subprocess often inherits a stripped
# LD_LIBRARY_PATH. Same guard as scripts/bench/bench_meta.R.
.WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(.WSL_CUDA_LIB) &&
    !grepl(.WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  .cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(.cur))
                                  paste(.WSL_CUDA_LIB, .cur, sep = ":")
                                else .WSL_CUDA_LIB)
}

# Run nvidia-smi, return stdout lines or NULL on failure.
.smi <- function(args) {
  res <- tryCatch(
    suppressWarnings(system2("nvidia-smi", args, stdout = TRUE, stderr = TRUE)),
    error = function(e) NULL
  )
  if (is.null(res)) return(NULL)
  status <- attr(res, "status")
  if (!is.null(status) && status != 0L) return(NULL)
  res
}

# Pull "Label : Value" from an nvidia-smi -q block; first match wins.
.field <- function(lines, label) {
  hit <- grep(sprintf("^\\s*%s\\s*:", label), lines, perl = TRUE, value = TRUE)
  if (!length(hit)) return(NA_character_)
  trimws(sub("^[^:]*:\\s*", "", hit[[1]]))
}

# ---- power envelope ---------------------------------------------------------
probe_power <- function() {
  lines <- .smi(c("-q", "-d", "POWER"))
  if (is.null(lines)) return(NULL)
  num <- function(s) suppressWarnings(as.numeric(sub("\\s*W$", "", s)))
  list(
    current_w = num(.field(lines, "Current Power Limit")),
    default_w = num(.field(lines, "Default Power Limit")),
    min_w     = num(.field(lines, "Min Power Limit")),
    max_w     = num(.field(lines, "Max Power Limit")),
    draw_w    = num(.field(lines, "Average Power Draw"))
  )
}

# ---- supported graphics clocks ---------------------------------------------
probe_supported_clocks <- function() {
  lines <- .smi(c("-q", "-d", "SUPPORTED_CLOCKS"))
  if (is.null(lines)) return(integer(0))
  gfx <- grep("^\\s*Graphics\\s*:", lines, perl = TRUE, value = TRUE)
  mhz <- suppressWarnings(as.integer(sub(".*:\\s*([0-9]+)\\s*MHz.*", "\\1", gfx)))
  sort(unique(mhz[!is.na(mhz)]), decreasing = TRUE)
}

# ---- current clock + throttle state ----------------------------------------
probe_state <- function() {
  fields <- c("clocks.current.sm", "clocks.current.memory",
              "clocks.max.sm", "temperature.gpu", "power.draw", "pstate",
              "clocks_throttle_reasons.active")
  res <- .smi(c(sprintf("--query-gpu=%s", paste(fields, collapse = ",")),
                "--format=csv,noheader,nounits"))
  if (is.null(res) || !length(res)) return(NULL)
  v <- trimws(strsplit(res[[1]], ",", fixed = TRUE)[[1]])
  if (length(v) != length(fields)) return(NULL)
  out <- as.list(v); names(out) <- fields
  out
}

# Decode throttle bitmask -> reason names (subset; see bench_meta.R).
.throttle_bits <- c(GpuIdle = 0x1, ApplicationsClocksSet = 0x2,
                    SwPowerCap = 0x4, HwSlowdown = 0x8, SyncBoost = 0x10,
                    SwThermalSlowdown = 0x20, HwThermalSlowdown = 0x40,
                    HwPowerBrakeSlowdown = 0x80, DisplayClocksSetting = 0x100)
decode_throttle <- function(hex_str) {
  v <- suppressWarnings(strtoi(sub("^0x", "", hex_str), base = 16L))
  if (is.na(v) || v == 0) return(character(0))
  names(Filter(function(b) bitwAnd(v, b) != 0L, .throttle_bits))
}

# ---- GPU / display mode (WSL is blind to the MUX — see methodology doc) ----
probe_mode <- function() {
  lines <- .smi("-q")
  if (is.null(lines)) return(NULL)
  list(
    display_active   = .field(lines, "Display Active"),
    display_attached = .field(lines, "Display Attached"),
    virtualization   = .field(lines, "Virtualization Mode")
  )
}

# ---- report -----------------------------------------------------------------
main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  as_json <- "--json" %in% argv

  power   <- probe_power()
  clocks  <- probe_supported_clocks()
  state   <- probe_state()
  mode    <- probe_mode()

  if (is.null(power) && is.null(state)) {
    cat("ERROR: nvidia-smi unavailable — no GPU probe possible.\n")
    quit(status = 1)
  }

  throttle <- if (!is.null(state))
    decode_throttle(state[["clocks_throttle_reasons.active"]]) else character(0)

  if (as_json) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      cat("ERROR: --json needs the jsonlite package.\n"); quit(status = 1)
    }
    obj <- list(timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
                power = power, supported_clocks_mhz = clocks,
                state = state, throttle = throttle, mode = mode)
    cat(jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE, null = "null"), "\n")
    quit(status = 0)
  }

  cat("=== GPU power & clock probe ===\n\n")
  if (!is.null(power)) {
    cat(sprintf("Power limit : current %.0f W | default %.0f W | max %.0f W | min %.0f W\n",
                power$current_w, power$default_w, power$max_w, power$min_w))
    headroom <- power$max_w - power$current_w
    if (is.finite(headroom) && headroom <= 0)
      cat("            : at the VBIOS ceiling — no headroom (nvidia-smi -pl cannot raise it)\n")
    else
      cat(sprintf("            : %.0f W of headroom below the max limit\n", headroom))
  }
  if (length(clocks))
    cat(sprintf("\nSupported SM clocks (MHz): %s\n  highest %d | lowest %d | %d steps\n",
                paste(head(clocks, 12), collapse = " "),
                max(clocks), min(clocks), length(clocks)))
  if (!is.null(state))
    cat(sprintf("\nNow: SM %s MHz (max %s) | mem %s MHz | %s C | %s W | pstate %s\n",
                state[["clocks.current.sm"]], state[["clocks.max.sm"]],
                state[["clocks.current.memory"]], state[["temperature.gpu"]],
                state[["power.draw"]], state[["pstate"]]))
  cat(sprintf("Throttle   : %s\n",
              if (length(throttle)) paste(throttle, collapse = ",") else "none"))
  if (!is.null(mode)) {
    cat(sprintf("\nDisplay active=%s attached=%s | virtualization=%s\n",
                mode$display_active, mode$display_attached, mode$virtualization))
    cat("NOTE: under WSL nvidia-smi cannot observe the laptop MUX state.\n")
    cat("      GPU mode (hybrid/dGPU) must be recorded from the Windows host.\n")
  }
  cat("\nSee docs/benchmark_methodology.md for what these numbers imply.\n")
}

if (sys.nframe() == 0L) main()
