# bench_measure.R -- per-sample validation, sample collection, and median
# reporting shared by the measurement harnesses (issue #134).
#
# Consolidates: grid_measure.R's inline validation, rebaseline_measure.R's
# classify_sample + collect_samples + measure_config medians, and
# bench_regress.R's measure_clock_locked validation + sample loop + median.

#' Decide whether one benchmark sample is valid.
#'
#' Single per-sample verdict shared by every harness. The validity
#' \emph{decision} is preserved exactly across callers; the reject-reason
#' strings are canonicalised (downstream `grid_collect.R` histograms them
#' tolerantly).
#'
#' Checks, in order: non-zero exit (\code{rc}); missing throughput
#' (parse failure); then \code{\link{classify_meta}(pre, post, valid_when)}
#' for throttle / \code{min_clock_sm} floor / \code{max_temp_c} /
#' \code{require_ac}; then a two-sided observed-clock band.
#'
#' \strong{Throttle scope.} \code{classify_meta} inspects BOTH \code{pre}
#' and \code{post} throttle. A caller that validates on the post-run
#' snapshot only (grid_measure) must pass its post snapshot as \emph{both}
#' \code{pre} and \code{post} -- then the pre/post union collapses to
#' post-only, matching its original \code{throttle_str(post)} semantics.
#' rebaseline / bench_regress pass their real \code{pre}, \code{post}.
#'
#' @param rc Integer child exit status (0 = ok).
#' @param throughput Parsed throughput; \code{NA}/\code{NULL} = parse fail.
#' @param pre,post \code{\link{capture_gpu_state}()} snapshots.
#' @param valid_when List forwarded to \code{classify_meta} (e.g.
#'   \code{list(allow_throttle = "GpuIdle", min_clock_sm = 1300)}).
#' @param clock_band Optional two-sided MHz band \code{c(lo, hi)} the
#'   observed SM clock must fall within (the host-side clock-lock check);
#'   \code{NULL} (native regime) skips it.
#' @return \code{list(ok, reason)}: \code{ok} is \code{TRUE}/\code{FALSE},
#'   \code{reason} is \code{NA_character_} when valid else a short string.
#' @export
validate_sample <- function(rc, throughput, pre, post,
                            valid_when = list(), clock_band = NULL) {
  if (!is.null(rc) && !is.na(rc) && rc != 0L)
    return(list(ok = FALSE, reason = sprintf("crash(exit=%d)", as.integer(rc))))
  if (is.null(throughput) || is.na(throughput))
    return(list(ok = FALSE, reason = "parse-fail"))

  cls <- classify_meta(pre, post, valid_when)
  if (is.na(cls$ok))
    return(list(ok = FALSE, reason = "no-gpu-meta"))
  if (isFALSE(cls$ok))
    return(list(ok = FALSE,
                reason = sprintf("unfair(%s)", paste(cls$reasons, collapse = ";"))))

  if (!is.null(clock_band)) {
    clk <- post$gpu$clock_sm
    if (is.null(clk) || is.na(clk) || clk < clock_band[[1]] || clk > clock_band[[2]])
      return(list(ok = FALSE,
                  reason = sprintf("clock %s MHz outside band %d-%d",
                                   if (is.null(clk) || is.na(clk)) "NA"
                                   else as.character(as.integer(clk)),
                                   as.integer(clock_band[[1]]),
                                   as.integer(clock_band[[2]]))))
  }
  list(ok = TRUE, reason = NA_character_)
}

#' Collect N valid samples, retrying up to a cap.
#'
#' The loop-until-N-valid pattern shared by rebaseline_measure.R and
#' bench_regress.R's clock-locked path. Generic over how a sample is
#' produced and validated: \code{sample_fn} runs one benchmark and
#' returns whatever object the caller needs; \code{validate_fn} maps that
#' object to a \code{\link{validate_sample}}-style \code{list(ok, reason)}.
#' Only \emph{valid} samples are kept, and they are returned \strong{whole}
#' (so a caller can pick a representative sample and carry its metadata
#' forward). Warmup stays with the caller (counts differ per harness).
#'
#' grid_measure.R does NOT use this -- it records every attempt (valid or
#' not) to JSONL with no retry, and shares only \code{\link{validate_sample}}.
#'
#' @param sample_fn Niladic function returning one run result.
#' @param validate_fn Function of the run result returning
#'   \code{list(ok, reason)}.
#' @param n_valid Target count of valid samples.
#' @param max_attempts Hard cap on total attempts.
#' @param on_sample Optional callback \code{function(attempt, ok, sample,
#'   reason)} for per-attempt logging.
#' @return \code{list(samples, rejected, attempts, complete)}:
#'   \code{samples} is the list of valid run results,
#'   \code{rejected} the reject-reason strings, \code{attempts} the total
#'   tries, \code{complete} whether \code{n_valid} was reached.
#' @export
collect_valid_samples <- function(sample_fn, validate_fn, n_valid,
                                  max_attempts, on_sample = NULL) {
  samples  <- list()
  rejected <- character(0)
  attempt  <- 0L
  while (length(samples) < n_valid && attempt < max_attempts) {
    attempt <- attempt + 1L
    s <- sample_fn()
    v <- validate_fn(s)
    if (isTRUE(v$ok)) {
      samples[[length(samples) + 1L]] <- s
    } else {
      rejected <- c(rejected, v$reason)
    }
    if (!is.null(on_sample)) on_sample(attempt, isTRUE(v$ok), s, v$reason)
  }
  list(samples = samples, rejected = rejected, attempts = attempt,
       complete = length(samples) >= n_valid)
}

#' Median + spread summary of a set of valid samples.
#'
#' Shared by rebaseline_measure.R's \code{measure_config} and
#' bench_regress.R's clock-locked median. Takes plain numeric vectors so
#' it is agnostic to how the caller stores samples.
#'
#' @param throughput Numeric vector of valid-sample throughputs.
#' @param ms Numeric vector of valid-sample timings (ms).
#' @param clk Optional integer vector of observed SM clocks.
#' @return \code{list(median_throughput, median_ms, tput_lo, tput_hi,
#'   clk_lo, clk_hi, n)}. Clock fields are \code{NA} when \code{clk} is
#'   omitted. \code{n} is the throughput count.
#' @export
report_median_metrics <- function(throughput, ms, clk = NULL) {
  list(
    median_throughput = stats::median(throughput, na.rm = TRUE),
    median_ms         = stats::median(ms, na.rm = TRUE),
    tput_lo           = if (length(throughput)) min(throughput, na.rm = TRUE) else NA_real_,
    tput_hi           = if (length(throughput)) max(throughput, na.rm = TRUE) else NA_real_,
    clk_lo            = if (!is.null(clk) && length(clk)) min(clk, na.rm = TRUE) else NA_integer_,
    clk_hi            = if (!is.null(clk) && length(clk)) max(clk, na.rm = TRUE) else NA_integer_,
    n                 = length(throughput)
  )
}
