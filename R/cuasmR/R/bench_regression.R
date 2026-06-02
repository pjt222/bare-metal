# bench_regression.R -- baseline regression decision (issue #134).
#
# The final stage of the measurement API: compare a measured sample
# against a recorded baseline and classify it OK / IMPROVED / REGRESSION,
# refusing the comparison (SKIPPED) when the GPU was in an unfair state.

#' Classify a measured sample against its baseline.
#'
#' A run that crashed is a regression (CRASH). A run taken under an unfair
#' GPU state (throttle / power cap / below a clock floor — per the entry's
#' \code{valid_when}) is \strong{not} comparable and returns SKIPPED, not
#' REGRESSION. Otherwise the throughput ratio versus baseline is bucketed
#' by \code{tolerance} into REGRESSION / OK / IMPROVED.
#'
#' @param current Measured metrics: \code{returncode}, \code{throughput},
#'   \code{unit}, and optionally \code{meta_pre}/\code{meta_post}
#'   (\code{\link{capture_gpu_state}()} snapshots; \code{NULL} off-GPU
#'   skips the fairness check).
#' @param baseline Baseline entry: \code{gflops} or \code{tops} (matched
#'   to \code{current$unit}), and optional \code{valid_when}.
#' @param tolerance Fractional band; \code{ratio < 1 - tolerance} is a
#'   regression, \code{> 1 + tolerance} an improvement.
#' @param default_valid_when Fairness criteria used when the baseline
#'   entry has no \code{valid_when} of its own.
#' @return \code{list(is_reg, msg)} and, for the unfair case,
#'   \code{skipped = TRUE}.
#' @export
check_regression <- function(current, baseline, tolerance,
                             default_valid_when = list()) {
  if (!is.null(current$returncode) && current$returncode != 0L) {
    return(list(is_reg = TRUE,
                msg = sprintf("CRASH (exit=%d)", current$returncode)))
  }

  # Refuse to compare if the GPU was in an unfair state during the run
  # (thermal throttle, sw power cap, etc). Reported as SKIPPED, not
  # REGRESSION — the measurement isn't comparable to baseline regardless
  # of what the number says.
  if (!is.null(current$meta_pre) && !is.null(current$meta_post)) {
    valid_when <- if (!is.null(baseline$valid_when)) baseline$valid_when
                  else default_valid_when
    cls <- classify_meta(current$meta_pre, current$meta_post, valid_when)
    if (isFALSE(cls$ok)) {
      return(list(is_reg = FALSE, skipped = TRUE,
                  msg = sprintf("SKIPPED (%s) [%s]",
                                paste(cls$reasons, collapse = "; "),
                                cls$summary)))
    }
  }

  unit <- if (!is.null(current$unit)) current$unit else "GFLOPS"
  baseline_val <- baseline[[tolower(unit)]]
  if (is.null(baseline_val)) baseline_val <- baseline$gflops
  if (is.null(baseline_val)) baseline_val <- baseline$tops
  if (is.null(baseline_val)) baseline_val <- 0
  current_val <- if (!is.null(current$throughput)) current$throughput else 0

  if (baseline_val == 0 || current_val == 0) {
    return(list(is_reg = TRUE,
                msg = sprintf("NO_DATA (baseline=%g, current=%g)",
                              baseline_val, current_val)))
  }
  ratio <- current_val / baseline_val
  if (ratio < (1.0 - tolerance)) {
    list(is_reg = TRUE,
         msg = sprintf("REGRESSION %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  } else if (ratio > (1.0 + tolerance)) {
    list(is_reg = FALSE,
         msg = sprintf("IMPROVED %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  } else {
    list(is_reg = FALSE,
         msg = sprintf("OK %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  }
}
