# bench_meta.R -- GPU + host state capture for benchmark runs.
#
# Wraps nvidia-smi and /proc readers into a structured snapshot, so a
# benchmark harness (bench_regress.R, grid_measure.R, ...) can attach
# measurement-time metadata to each result and refuse to compare runs
# at incompatible system states.
#
# Migrated from scripts/bench/bench_meta.R (issue #134). The WSL CUDA
# LD_LIBRARY_PATH guard that used to run at source-time now lives in
# .onLoad() (zzz.R) so it fires when the package is attached.

# ---- nvidia-smi field set ---------------------------------------------------
#
# Picked fields that matter for "is this measurement comparable to a
# baseline recorded earlier?":
#
#   clocks.current.sm       -- boost vs base clock; the dominant driver
#                             of throughput variance on a laptop GPU.
#   clocks.current.memory   -- memory clock; affects DRAM-bound kernels.
#   temperature.gpu         -- flags incoming thermal throttle.
#   power.draw              -- flags power-cap throttle.
#   pstate                  -- P0 (max) vs P2 / P5 / P8 (idle steps).
#   clocks_throttle_reasons.active   -- the canonical "is the kernel
#                                       being penalised right now"
#                                       bitmask (see decode below).
#   utilization.gpu         -- sanity check that the kernel actually ran.
#   utilization.memory      -- DRAM bus pressure during the launch.
#   enforced.power.limit    -- the cap the platform is ACTUALLY applying
#                             right now (issue #207). On a laptop this is
#                             set by the OEM/Windows power policy, not by
#                             us, and it moves without warning.
#   power.default_limit     -- the card's default cap. The pair matters,
#                             not either alone: `enforced < default` means
#                             the platform is holding the GPU below its
#                             rated envelope, which makes the measurement
#                             incomparable to a baseline recorded at full
#                             power. `power.draw` cannot show this -- it
#                             reports what the GPU drew, never what it was
#                             allowed to draw.
#
# Both come from the SAME nvidia-smi invocation as the fields above, so
# recording them costs nothing (measured: 0.148s for the query either way).
.NVIDIA_SMI_FIELDS <- c(
  "clocks.current.sm",
  "clocks.current.memory",
  "temperature.gpu",
  "power.draw",
  "pstate",
  "clocks_throttle_reasons.active",
  "utilization.gpu",
  "utilization.memory",
  "enforced.power.limit",
  "power.default_limit"
)

# Throttle-reason bitmask. From NVIDIA documentation; constants are
# stable across driver versions on Ampere.
.THROTTLE_BITS <- c(
  "GpuIdle"               = 0x0001,
  "ApplicationsClocksSet" = 0x0002,
  "SwPowerCap"            = 0x0004,
  "HwSlowdown"            = 0x0008,
  "SyncBoost"             = 0x0010,
  "SwThermalSlowdown"     = 0x0020,
  "HwThermalSlowdown"     = 0x0040,
  "HwPowerBrakeSlowdown"  = 0x0080,
  "DisplayClocksSetting"  = 0x0100
)

# Throttle states that make a measurement *unfair* (the GPU was being
# held below its capability when we recorded the number). GpuIdle is
# benign at moments between launches; the rest indicate the kernel
# itself was constrained.
.UNFAIR_THROTTLES <- c(
  "ApplicationsClocksSet",
  "SwPowerCap",
  "HwSlowdown",
  "SwThermalSlowdown",
  "HwThermalSlowdown",
  "HwPowerBrakeSlowdown"
)

# ---- low-level helpers ------------------------------------------------------

# Run nvidia-smi with a CSV query; return a single-row data.frame with
# the requested fields parsed (numeric where possible). On failure
# returns NULL so callers can gracefully skip metadata when no GPU
# is present (e.g. on a CI runner without a card).
.nvidia_smi_query <- function(fields = .NVIDIA_SMI_FIELDS) {
  query <- paste(fields, collapse = ",")
  res <- tryCatch(
    suppressWarnings(system2(
      "nvidia-smi",
      args = c(sprintf("--query-gpu=%s", query),
               "--format=csv,noheader,nounits"),
      stdout = TRUE, stderr = TRUE
    )),
    error = function(e) NULL
  )
  if (is.null(res) || !length(res)) return(NULL)
  status <- attr(res, "status")
  if (!is.null(status) && status != 0L) return(NULL)

  vals <- strsplit(res[[1]], ",", fixed = TRUE)[[1]]
  vals <- trimws(vals)
  if (length(vals) != length(fields)) return(NULL)

  # Numeric coercion for everything except the throttle hex string and pstate
  out <- as.list(vals)
  names(out) <- fields
  for (k in names(out)) {
    v <- out[[k]]
    if (k == "pstate" || k == "clocks_throttle_reasons.active") next
    n <- suppressWarnings(as.numeric(v))
    if (!is.na(n)) out[[k]] <- n
  }
  out
}

#' Decode an nvidia-smi throttle-reason bitmask.
#'
#' @param hex_str Hex string from
#'   \code{clocks_throttle_reasons.active} (e.g. \code{"0x0000000000000004"}).
#' @return Character vector of active reason names; empty vector means
#'   "no throttle".
#' @export
decode_throttle <- function(hex_str) {
  if (is.null(hex_str) || !nzchar(hex_str)) return(character(0))
  v <- suppressWarnings(strtoi(sub("^0x", "", hex_str), base = 16L))
  if (is.na(v)) return(character(0))
  if (v == 0) return(character(0))
  active <- character(0)
  for (name in names(.THROTTLE_BITS)) {
    if (bitwAnd(v, .THROTTLE_BITS[[name]]) != 0L) active <- c(active, name)
  }
  active
}

# Is the platform holding the GPU below its rated power envelope?
# (issue #207). Returns TRUE / FALSE / NA. NA means "could not tell"
# -- a driver reporting [N/A] for either limit, or a non-numeric value.
# NA is deliberately NOT folded into FALSE: "we do not know whether this
# session was capped" and "this session was not capped" are different
# claims, and only the second one licenses a comparison to a baseline.
.power_capped <- function(enforced, default) {
  if (is.null(enforced) || is.null(default)) return(NA)
  if (!is.numeric(enforced) || !is.numeric(default)) return(NA)
  if (is.na(enforced) || is.na(default) || default <= 0) return(NA)
  # 1 W of slack: some drivers report the pair a hair apart when uncapped.
  enforced < (default - 1)
}

# Read /proc/loadavg into a list. Linux/WSL only; returns NULL on
# other OSes.
.read_loadavg <- function() {
  path <- "/proc/loadavg"
  if (!file.exists(path)) return(NULL)
  raw <- tryCatch(readLines(path, n = 1, warn = FALSE), error = function(e) NULL)
  if (is.null(raw)) return(NULL)
  parts <- strsplit(raw, "\\s+")[[1]]
  if (length(parts) < 3) return(NULL)
  list(
    load_1m  = as.numeric(parts[1]),
    load_5m  = as.numeric(parts[2]),
    load_15m = as.numeric(parts[3])
  )
}

# Detect AC vs battery on a laptop. Returns "ac", "battery", or
# "unknown" (no /sys/class/power_supply, or no AC adapter present).
.read_ac_state <- function() {
  base <- "/sys/class/power_supply"
  if (!dir.exists(base)) return("unknown")
  ac_dirs <- list.files(base, pattern = "^A(C|DP|CAD)", full.names = TRUE)
  if (!length(ac_dirs)) return("unknown")
  for (d in ac_dirs) {
    f <- file.path(d, "online")
    if (file.exists(f)) {
      v <- tryCatch(as.integer(readLines(f, n = 1, warn = FALSE)),
                    error = function(e) NA_integer_)
      if (!is.na(v)) return(if (v == 1L) "ac" else "battery")
    }
  }
  "unknown"
}

# Laptop GPU mode (hybrid vs dGPU/MUX). WSL2 nvidia-smi cannot observe
# the MUX state -- `display_active` reads Disabled in both modes -- so the
# mode is NOT auto-detected. It is taken from the BARE_METAL_GPU_MODE
# environment variable, set explicitly by whoever records the run.
# Accepted: "hybrid", "dgpu". Anything else (including unset) -> "unknown".
# Never guess from display_active. See issue #126.
.read_gpu_mode <- function() {
  v <- tolower(trimws(Sys.getenv("BARE_METAL_GPU_MODE", unset = "")))
  if (v %in% c("hybrid", "dgpu")) v else "unknown"
}

# Driver + CUDA versions for provenance.
.read_versions <- function() {
  drv <- tryCatch({
    r <- system2("nvidia-smi",
                 c("--query-gpu=driver_version", "--format=csv,noheader"),
                 stdout = TRUE, stderr = FALSE)
    trimws(r[[1]])
  }, error = function(e) NA_character_)
  cuda <- tryCatch({
    r <- system2("nvcc", "--version", stdout = TRUE, stderr = FALSE)
    m <- regmatches(paste(r, collapse = " "),
                    regexec("release\\s+([0-9.]+)",
                            paste(r, collapse = " "), perl = TRUE))[[1]]
    if (length(m) >= 2) m[2] else NA_character_
  }, error = function(e) NA_character_)
  list(driver = drv, cuda = cuda)
}

# ---- public API -------------------------------------------------------------

#' Snapshot GPU + host state at the current instant.
#'
#' Cheap (~50ms for the nvidia-smi spawn). Safe to call before AND
#' after each bench launch.
#'
#' @return A list with shape:
#'   \describe{
#'     \item{gpu}{\code{clock_sm}, \code{clock_mem}, \code{temp_c},
#'       \code{power_w}, \code{pstate}, \code{throttle_hex},
#'       \code{throttle} (decoded character vector), \code{util_gpu},
#'       \code{util_mem}, \code{power_limit_w} (the cap the platform is
#'       enforcing), \code{power_limit_default_w} (the card's default),
#'       \code{power_capped} (\code{TRUE}/\code{FALSE}/\code{NA} --
#'       \code{NA} means the driver did not report the limits, which is
#'       NOT the same as "not capped"; see issue #207)}
#'     \item{host}{\code{loadavg}, \code{ac_state}, \code{gpu_mode}}
#'     \item{iso_time}{ISO 8601 timestamp}
#'   }
#'   Returns \code{NULL} if nvidia-smi fails (no GPU / not on PATH);
#'   callers should treat that as "no metadata available, run anyway".
#' @export
capture_gpu_state <- function() {
  gpu_raw <- .nvidia_smi_query()
  if (is.null(gpu_raw)) return(NULL)

  gpu <- list(
    clock_sm     = gpu_raw[["clocks.current.sm"]],
    clock_mem    = gpu_raw[["clocks.current.memory"]],
    temp_c       = gpu_raw[["temperature.gpu"]],
    power_w      = gpu_raw[["power.draw"]],
    pstate       = gpu_raw[["pstate"]],
    throttle_hex = gpu_raw[["clocks_throttle_reasons.active"]],
    throttle     = decode_throttle(gpu_raw[["clocks_throttle_reasons.active"]]),
    util_gpu     = gpu_raw[["utilization.gpu"]],
    util_mem     = gpu_raw[["utilization.memory"]],
    # Platform power envelope (issue #207). Either may be non-numeric on a
    # driver that reports [N/A]; .power_capped() treats that as "unknown",
    # never as "not capped".
    power_limit_w         = gpu_raw[["enforced.power.limit"]],
    power_limit_default_w = gpu_raw[["power.default_limit"]]
  )
  gpu$power_capped <- .power_capped(gpu$power_limit_w, gpu$power_limit_default_w)

  list(
    gpu      = gpu,
    host     = list(loadavg = .read_loadavg(),
                    ac_state = .read_ac_state(),
                    gpu_mode = .read_gpu_mode()),
    iso_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  )
}

# Windows "power mode" overlay GUIDs. These sit on top of the power
# scheme and are what the taskbar slider / Settings > Power sets.
.POWER_OVERLAYS <- c(
  "961cc777-2547-4f9d-8174-7d86181b8a7a" = "best-power-efficiency",
  "00000000-0000-0000-0000-000000000000" = "balanced",
  "ded574b5-45a0-4f42-8737-46345c09c238" = "best-performance",
  "3af9b8d9-7c97-431d-ad78-34a8bfea439f" = "better-performance"
)

#' Capture the host power policy that governs the GPU's power envelope.
#'
#' Session-level provenance for issue #207. On a laptop the dGPU's
#' \code{enforced.power.limit} is set by the platform, and on Windows the
#' lever is the power-mode \emph{overlay} -- which Windows changes on its
#' own (battery state, "automatic improvements", OEM utilities). The
#' overlay is the field that explains \emph{why} a cap moved, so a session
#' whose numbers look wrong can be attributed instead of re-litigated.
#'
#' Costs a \code{powershell.exe} spawn (~1s), far too slow for the
#' per-sample path -- call it ONCE per measurement session and attach the
#' result to the run record, alongside the per-sample
#' \code{power_limit_w} that \code{\link{capture_gpu_state}} records.
#'
#' @return \code{list(overlay_guid, overlay_name, ac_overlay_guid,
#'   dc_overlay_guid, source)}. \code{overlay_name} is a short slug
#'   (\code{"best-power-efficiency"}, \code{"balanced"},
#'   \code{"best-performance"}, \code{"better-performance"}) or the raw
#'   GUID when unrecognised. Every field is \code{NA_character_} with
#'   \code{source = "unavailable"} off Windows/WSL -- never a guess.
#' @export
capture_power_policy <- function() {
  unavailable <- list(overlay_guid = NA_character_,
                      overlay_name = NA_character_,
                      ac_overlay_guid = NA_character_,
                      dc_overlay_guid = NA_character_,
                      source = "unavailable")

  ps <- Sys.which("powershell.exe")
  if (!nzchar(ps)) return(unavailable)

  key <- paste0("HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\",
                "PowerSchemes")
  cmd <- sprintf(
    paste0("$p = Get-ItemProperty '%s' -ErrorAction SilentlyContinue; ",
           "if ($p) { \"$($p.ActiveOverlayAcPowerScheme)|",
           "$($p.ActiveOverlayDcPowerScheme)\" }"), key)

  res <- tryCatch(
    suppressWarnings(system2(ps, c("-NoProfile", "-NonInteractive",
                                   "-Command", shQuote(cmd)),
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) NULL)
  if (is.null(res) || !length(res)) return(unavailable)

  line <- trimws(paste(res, collapse = ""))
  parts <- strsplit(line, "|", fixed = TRUE)[[1]]
  if (length(parts) < 1L || !nzchar(parts[[1]])) return(unavailable)

  ac <- tolower(trimws(parts[[1]]))
  dc <- if (length(parts) >= 2L) tolower(trimws(parts[[2]])) else NA_character_

  # The AC overlay is the one that governs a benchmark run -- every
  # gated measurement requires AC (see require_ac in classify_meta).
  nm <- unname(.POWER_OVERLAYS[ac])
  list(overlay_guid    = ac,
       overlay_name    = if (is.na(nm)) ac else nm,
       ac_overlay_guid = ac,
       dc_overlay_guid = dc,
       source          = "windows-registry")
}

#' Decide whether a measurement is comparable to a baseline.
#'
#' @param pre,post \code{\link{capture_gpu_state}()} snapshots taken
#'   before / after the bench.
#' @param valid_when Optional list with any of:
#'   \describe{
#'     \item{require_no_throttle}{logical (default TRUE)}
#'     \item{allow_throttle}{character vector of names that ARE OK
#'       (default \code{"GpuIdle"})}
#'     \item{min_clock_sm}{numeric MHz floor}
#'     \item{max_temp_c}{numeric ceiling}
#'     \item{require_ac}{logical (laptop)}
#'   }
#' @return \code{list(ok, reasons, summary)}. \code{ok} is \code{NA} if
#'   \code{pre} or \code{post} is \code{NULL} (no GPU detected).
#' @export
classify_meta <- function(pre, post, valid_when = list()) {
  if (is.null(pre) || is.null(post)) {
    return(list(ok = NA, reasons = "no GPU metadata captured",
                summary = "meta unavailable"))
  }

  defaults <- list(
    require_no_throttle = TRUE,
    allow_throttle      = c("GpuIdle"),
    min_clock_sm        = NULL,
    max_temp_c          = NULL,
    require_ac          = FALSE
  )
  cfg <- utils::modifyList(defaults, as.list(valid_when))

  reasons <- character(0)

  if (cfg$require_no_throttle) {
    bad_throttle_pre  <- setdiff(pre$gpu$throttle,  cfg$allow_throttle)
    bad_throttle_post <- setdiff(post$gpu$throttle, cfg$allow_throttle)
    bad <- unique(c(bad_throttle_pre, bad_throttle_post))
    if (length(bad)) {
      reasons <- c(reasons,
        sprintf("throttle active during run: %s", paste(bad, collapse = ",")))
    }
  }
  if (!is.null(cfg$min_clock_sm)) {
    # Use post-run clock -- the one the kernel actually saw at the end. An
    # NA clock (nvidia-smi parse miss) is treated as below the floor:
    # we could not confirm the kernel ran at speed, so the sample is
    # unfair. NA-safe -- a bare `NA < x` would error in `if`.
    clk_v <- post$gpu$clock_sm
    if (is.na(clk_v) || clk_v < cfg$min_clock_sm) {
      reasons <- c(reasons,
        sprintf("clock_sm=%s MHz < required %d MHz",
                if (is.na(clk_v)) "NA" else as.character(as.integer(clk_v)),
                as.integer(cfg$min_clock_sm)))
    }
  }
  if (!is.null(cfg$max_temp_c)) {
    if (post$gpu$temp_c > cfg$max_temp_c) {
      reasons <- c(reasons,
        sprintf("temp=%d\u00b0C > max %d\u00b0C",
                as.integer(post$gpu$temp_c), as.integer(cfg$max_temp_c)))
    }
  }
  if (cfg$require_ac && pre$host$ac_state == "battery") {
    reasons <- c(reasons, "running on battery")
  }

  ok <- length(reasons) == 0L
  gpu_mode <- if (!is.null(post$host$gpu_mode)) post$host$gpu_mode else "unknown"
  summary <- sprintf("clk=%d/%d MHz  temp=%d\u00b0C  power=%.1fW  pstate=%s  %s  gpu_mode=%s%s",
                     as.integer(post$gpu$clock_sm),
                     as.integer(post$gpu$clock_mem),
                     as.integer(post$gpu$temp_c),
                     post$gpu$power_w,
                     post$gpu$pstate,
                     if (length(post$gpu$throttle))
                       paste0("throttle=[", paste(post$gpu$throttle, collapse = ","), "]")
                     else "throttle=none",
                     gpu_mode,
                     # Appended only when the platform is capping (issue #207),
                     # so existing summaries are unchanged on a normal session.
                     if (isTRUE(post$gpu$power_capped))
                       sprintf("  POWER-CAPPED=%.0f/%.0fW",
                               post$gpu$power_limit_w,
                               post$gpu$power_limit_default_w)
                     else "")

  list(ok = ok, reasons = reasons, summary = summary)
}

#' One-line GPU-state summary for terminal output.
#'
#' @param pre,post \code{\link{capture_gpu_state}()} snapshots.
#' @return A single string; \code{"(no GPU meta)"} when either snapshot
#'   is \code{NULL}.
#' @export
summarise_meta <- function(pre, post) {
  if (is.null(pre) || is.null(post)) return("(no GPU meta)")
  cls <- classify_meta(pre, post)
  cls$summary
}
