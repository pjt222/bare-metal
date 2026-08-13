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
#   power.default_limit     -- the card's DEFAULT cap.
#   power.max_limit         -- the VBIOS ceiling.
#
# All three are needed, and the reason is specific to this machine.
# `docs/benchmark_methodology.md` records it as
# `Current 150 W / Default 115 W / Max 150 W`: an OEM performance mode
# has already raised the enforced limit 115 -> 150, so the NORMAL state
# here is `enforced == max`, and 115 W is a *degraded fallback*, not the
# envelope. Comparing against `default` alone would therefore read the
# entire [115, 150) band -- including that documented fallback, a 23%
# envelope cut -- as "not capped". Comparing against `max` alone
# false-positives on hardware whose normal operating point IS the
# default (most desktops). So both comparisons are recorded, and neither
# is collapsed into a single verdict here.
#
# `power.draw` cannot substitute for any of this: it reports what the GPU
# drew, never what it was allowed to draw.
#
# All three come from the SAME nvidia-smi invocation as the fields above,
# so recording them costs nothing (measured: 0.148s for the query either
# way).
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
  "power.default_limit",
  "power.max_limit"
)

# Throttle reasons, as BIT INDICES (not masks). From NVIDIA
# documentation; constants are stable across driver versions on Ampere.
#
# Indices rather than masks because the mask is a 64-bit word and the
# only safe way to test it is one hex digit at a time -- see
# hex64_bit_get() in binio.R and issue #208.
.THROTTLE_BITS <- c(
  "GpuIdle"               = 0L,
  "ApplicationsClocksSet" = 1L,
  "SwPowerCap"            = 2L,
  "HwSlowdown"            = 3L,
  "SyncBoost"             = 4L,
  "SwThermalSlowdown"     = 5L,
  "HwThermalSlowdown"     = 6L,
  "HwPowerBrakeSlowdown"  = 7L,
  "DisplayClocksSetting"  = 8L
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
#' @return Character vector of active reason names. \code{character(0)}
#'   means "no throttle". \code{NA_character_} means the mask could not be
#'   parsed -- deliberately NOT \code{character(0)}, see below.
#' @export
decode_throttle <- function(hex_str) {
  if (is.null(hex_str) || length(hex_str) != 1L || is.na(hex_str) ||
      !nzchar(hex_str)) {
    return(character(0))
  }

  # The mask is a 64-bit word. The previous implementation ran it through
  # strtoi(base = 16L), which is int32: any mask with a bit at or above
  # 2^31 returned NA, and the NA branch returned character(0) -- the same
  # value that means "no throttle". A throttled run therefore decoded as a
  # clean one and was compared against a baseline (#208). Testing one hex
  # digit at a time cannot overflow, and a mask we cannot parse now returns
  # NA_character_ instead, which classify_meta's setdiff() surfaces as an
  # unrecognised reason -- so the sample is rejected rather than accepted.
  ok <- TRUE
  active <- character(0)
  for (name in names(.THROTTLE_BITS)) {
    set <- tryCatch(hex64_bit_get(hex_str, .THROTTLE_BITS[[name]]),
                    error = function(e) { ok <<- FALSE; FALSE })
    if (!ok) break
    if (isTRUE(set)) active <- c(active, name)
  }
  if (!ok) return(NA_character_)
  active
}

# Is the enforced power limit below a given reference envelope?
# (issue #207). Returns TRUE / FALSE / NA. NA means "could not tell"
# -- a driver reporting [N/A] for either value, or a non-numeric one.
#
# NA is deliberately NOT folded into FALSE: "we do not know whether this
# session was capped" and "this session was not capped" are different
# claims, and only the second one licenses a comparison to a baseline.
#
# The caller supplies the reference because there is no single right one
# (see the field-set comment above): `default` is unambiguous on any
# hardware but blind to a laptop whose normal state is above default,
# and `max` catches that but false-positives where default IS the normal
# operating point.
# Normalise one nvidia-smi watt field to numeric-or-NA. The driver
# reports unsupported fields as the STRING "[N/A]", which
# .nvidia_smi_query leaves as a character value; storing that in a
# numeric column gives the JSONL a field that is a number on one machine
# and a string on another, and every consumer has to defend against it.
# A value we cannot read is NA, which serialises to null (issue #207).
.as_watts <- function(x) {
  if (is.null(x)) return(NA_real_)
  if (is.numeric(x)) return(as.numeric(x))
  v <- suppressWarnings(as.numeric(x))
  if (is.na(v)) NA_real_ else v
}

# Watts for display: a plain number, or "?" when unknown. Never let an
# NA reach sprintf("%.0f"), which would render the literal "NA" inside a
# field that otherwise always carries digits.
.fmt_w <- function(x) {
  v <- .as_watts(x)
  if (is.na(v)) "?" else format(round(v))
}

.power_below <- function(enforced, reference) {
  if (is.null(enforced) || is.null(reference)) return(NA)
  if (!is.numeric(enforced) || !is.numeric(reference)) return(NA)
  if (is.na(enforced) || is.na(reference) || reference <= 0) return(NA)
  # 1 W of slack: some drivers report the pair a hair apart when uncapped.
  enforced < (reference - 1)
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
    # Platform power envelope (issue #207). Any of these may be
    # non-numeric on a driver that reports [N/A]; .power_below() treats
    # that as "unknown", never as "not capped".
    power_limit_w         = .as_watts(gpu_raw[["enforced.power.limit"]]),
    power_limit_default_w = .as_watts(gpu_raw[["power.default_limit"]]),
    power_limit_max_w     = .as_watts(gpu_raw[["power.max_limit"]])
  )
  # Two references, recorded separately rather than collapsed. On this
  # machine the normal state is enforced == max == 150 W with default at
  # 115 W, so `below_max` is the operationally meaningful one and
  # `below_default` marks the more severe fallback.
  gpu$power_below_default <- .power_below(gpu$power_limit_w,
                                          gpu$power_limit_default_w)
  gpu$power_below_max     <- .power_below(gpu$power_limit_w,
                                          gpu$power_limit_max_w)

  list(
    gpu      = gpu,
    host     = list(loadavg = .read_loadavg(),
                    ac_state = .read_ac_state(),
                    gpu_mode = .read_gpu_mode()),
    iso_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  )
}

# Session cache for readings that are session-scoped by nature and
# expensive to take (issue #207). Not exported; cleared by restarting R
# or via the `refresh` argument.
.meta_cache <- new.env(parent = emptyenv())

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
#' @param ac_state Which overlay governs: \code{"ac"} (default),
#'   \code{"battery"}, or \code{"unknown"}. Windows keeps a separate
#'   overlay per power source and they routinely differ -- this machine
#'   has held \code{best-performance} on AC and
#'   \code{best-power-efficiency} on DC simultaneously -- so the caller
#'   must say which one applied. Pass
#'   \code{capture_gpu_state()$host$ac_state}. Do NOT assume AC: nothing
#'   in this repo forces a gated measurement onto AC
#'   (\code{require_ac} defaults to \code{FALSE} and no baseline sets it).
#' @param refresh Recompute instead of returning the cached value.
#' @return \code{list(overlay_guid, overlay_name, ac_overlay_guid,
#'   dc_overlay_guid, governing, source)}. \code{overlay_name} is a short
#'   slug (\code{"best-power-efficiency"}, \code{"balanced"},
#'   \code{"best-performance"}, \code{"better-performance"}) or the raw
#'   GUID when unrecognised. Every field is \code{NA_character_} with
#'   \code{source = "unavailable"} off Windows/WSL, when the query fails,
#'   or when the value read is not GUID-shaped -- never a guess.
#' @export
capture_power_policy <- function(ac_state = "ac", refresh = FALSE) {
  unavailable <- list(overlay_guid = NA_character_,
                      overlay_name = NA_character_,
                      ac_overlay_guid = NA_character_,
                      dc_overlay_guid = NA_character_,
                      governing = NA_character_,
                      source = "unavailable")

  # Memoised for the lifetime of the session. The overlay is read once
  # per measurement session by design, but the gate's own test fixture
  # runs bench_regress.R ~10 times inside one `make test-r`, and each
  # spawn measured 1.376s on this box -- ~15s added to the slowest
  # blocking pre-push step, whose comment records it was deliberately
  # cut from 79s by memoising exactly this kind of child process.
  cache_key <- paste0("policy_", ac_state)
  if (!refresh && !is.null(.meta_cache[[cache_key]]))
    return(.meta_cache[[cache_key]])

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
                             stdout = TRUE, stderr = FALSE,
                             # Bounded like every other child this package
                             # spawns (run_bench passes one too). tryCatch
                             # catches errors, not hangs, and this call sits
                             # on a blocking pre-push step: a wedged interop
                             # handler would hang `git push` with no output.
                             timeout = 10)),
    error = function(e) NULL)
  if (is.null(res) || !length(res)) return(unavailable)
  # Non-zero exit means the value we are holding is not a reading.
  # .nvidia_smi_query already checks this; this call did not.
  st <- attr(res, "status")
  if (!is.null(st) && st != 0L) return(unavailable)

  # One line per value, not a blind paste: Windows PowerShell writes the
  # WARNING stream to stdout, localized, and -ErrorAction does not cover
  # it. Gluing every line together produced a real observed corruption --
  # "warnung: noise" concatenated onto the GUID -- which then passed the
  # consumer's `source != "unavailable"` gate and was stored as provenance.
  toks <- trimws(unlist(strsplit(paste(res, collapse = "\n"), "[\n|]")))
  toks <- toks[nzchar(toks)]
  guid_re <- "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
  guids <- tolower(toks)[grepl(guid_re, tolower(toks))]
  if (!length(guids)) return(unavailable)

  ac <- guids[[1]]
  dc <- if (length(guids) >= 2L) guids[[2]] else NA_character_

  # Which overlay actually governed this run. Windows applies the DC
  # overlay on battery, and they differ in practice.
  governing <- if (identical(ac_state, "battery") && !is.na(dc)) dc else ac
  nm <- unname(.POWER_OVERLAYS[governing])

  out <- list(overlay_guid    = governing,
              overlay_name    = if (is.na(nm)) governing else nm,
              ac_overlay_guid = ac,
              dc_overlay_guid = dc,
              governing       = if (identical(ac_state, "battery")) "dc" else "ac",
              source          = "windows-registry")
  .meta_cache[[cache_key]] <- out
  out
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
                     # Appended only when the enforced limit is below the
                     # VBIOS ceiling (issue #207), so a session at the normal
                     # enforced == max operating point reads exactly as before.
                     # All three numbers are printed rather than a verdict:
                     # 50/115/150 (platform clamp) and 115/115/150 (the
                     # documented fallback to default) are different problems
                     # and the reader can tell them apart at a glance.
                     if (isTRUE(post$gpu$power_below_max) ||
                         isTRUE(post$gpu$power_below_default))
                       sprintf("  POWER-LIMIT=%s/%s/%sW",
                               .fmt_w(post$gpu$power_limit_w),
                               .fmt_w(post$gpu$power_limit_default_w),
                               .fmt_w(post$gpu$power_limit_max_w))
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
