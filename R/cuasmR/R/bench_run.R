# bench_run.R -- run a benchmark executable once, snapshotting GPU state.
#
# The single per-sample primitive shared by every measurement harness
# (bench_regress.R, grid_measure.R, rebaseline_measure.R,
# clock_lock_sweep.R). Migrated from those scripts' near-identical
# run_bench() / run_one_sample() / inline-runner copies (issue #134).

#' Run a benchmark executable once, snapshotting GPU state around it.
#'
#' Captures \code{\link{capture_gpu_state}()} immediately before and
#' after the launch, so a caller can later refuse to compare a run taken
#' under throttle / on battery / at the wrong clock.
#'
#' The benchmark resolves its cubin with a \emph{cwd-relative} filename
#' (\code{cuModuleLoad}), so it must run from the executable's own
#' directory. This function does \strong{not} change the working
#' directory — the caller is responsible for \code{setwd()} into the exe
#' dir (typically once around a warmup + sample loop). Pass an absolute
#' \code{exe} path so the launch itself is cwd-independent.
#'
#' @param exe Path to the benchmark executable (absolute recommended).
#' @param args Character vector of command-line arguments (coerced with
#'   \code{as.character}).
#' @param timeout Seconds before the child is killed; \code{0} (default)
#'   means no limit, matching base \code{\link{system2}}.
#' @return \code{list(out, pre, post, rc)} where \code{out} is the
#'   captured stdout+stderr \emph{line vector}, \code{pre}/\code{post}
#'   are \code{capture_gpu_state()} snapshots (\code{NULL} off-GPU), and
#'   \code{rc} is the integer exit status (\code{0} on success). A
#'   benchmark that catches \code{SIGINT} exits \code{130}; callers that
#'   want single-Ctrl+C cancellation should test \code{rc == 130L}.
#'   \code{out} stays a line vector (not a collapsed string) so the
#'   throughput parser can scan candidate lines.
#' @export
run_bench <- function(exe, args, timeout = 0) {
  pre <- capture_gpu_state()
  out <- tryCatch(
    suppressWarnings(system2(exe, as.character(args),
                             stdout = TRUE, stderr = TRUE, timeout = timeout)),
    error = function(e) {
      attr(character(0), "status") <- 1L
      character(0)
    }
  )
  post <- capture_gpu_state()
  status <- attr(out, "status")
  list(out = out, pre = pre, post = post,
       rc = if (is.null(status)) 0L else as.integer(status))
}
