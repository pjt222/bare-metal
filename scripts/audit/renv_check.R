#!/usr/bin/env Rscript
# renv_check.R -- deliberate renv project-synchronization check (#162).
#
# This is the RELOCATED half of the #162 fix. renv's autoloader used to
# run this same check on every R startup, costing ~31 s per process on
# this repo's 9p mount (/mnt/d) and making the GPU-free test suite take
# an hour. `.Rprofile` sets options(renv.config.synchronized.check = FALSE)
# to stop that; this script is where the check now lives, so drift is still
# caught -- once per push instead of dozens of times per workflow.
#
# The check itself is UNWEAKENED. Dependency discovery still scans the
# whole project: no `.renvignore` was added, and `snapshot.type` stays
# "implicit". Speeding the check up by making it scan less would trade
# correctness for time, which is the wrong direction -- a future R script
# under kernels/ or viz/ must still register as a dependency.
#
# Usage:
#   Rscript scripts/audit/renv_check.R          # report + exit 0/1
#   Rscript scripts/audit/renv_check.R --quiet  # only print on failure
#
# Exit codes:
#   0  project consistent (or renv absent -- nothing to check)
#   1  lockfile and library disagree

args   <- commandArgs(trailingOnly = TRUE)
quiet  <- "--quiet" %in% args

if (!requireNamespace("renv", quietly = TRUE)) {
  cat("renv not available -- skipping project sync check.\n")
  quit(status = 0)
}

repo_root <- {
  a  <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  start <- if (length(fa)) normalizePath(dirname(sub("^--file=", "", fa[1])))
           else            normalizePath(getwd())
  cur <- start
  repeat {
    if (file.exists(file.path(cur, "renv.lock"))) break
    parent <- dirname(cur)
    if (parent == cur) { cur <- start; break }
    cur <- parent
  }
  cur
}
old_wd <- setwd(repo_root); on.exit(setwd(old_wd), add = TRUE)

t0 <- Sys.time()
# renv::status() reports by printing; capture it so --quiet can stay
# silent on success, and so the synchronized flag can be read directly
# rather than scraped from the text.
out <- utils::capture.output(st <- renv::status(project = repo_root))
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# renv >= 1.0 returns a list with a `synchronized` flag. Fall back to the
# printed output if that field ever disappears, rather than silently
# passing: an unreadable result must NOT read as "consistent".
ok <- {
  if (is.list(st) && !is.null(st$synchronized)) isTRUE(st$synchronized)
  else any(grepl("No issues found", out, fixed = TRUE))
}

if (ok) {
  if (!quiet)
    cat(sprintf("renv: project consistent (lockfile == library), checked in %.1fs\n",
                elapsed))
  quit(status = 0)
}

cat("\n")
cat(strrep("=", 72), "\n", sep = "")
cat("  renv: PROJECT OUT OF SYNC (lockfile != library)\n")
cat(strrep("=", 72), "\n", sep = "")
cat(paste(out, collapse = "\n"), "\n\n")
cat("Resolve with ONE of:\n")
cat("  Rscript -e 'renv::restore()'    # library <- lockfile (adopt the lockfile)\n")
cat("  Rscript -e 'renv::snapshot()'   # lockfile <- library (record what you installed)\n")
cat("\n")
cat("If the offender is `cuasmR [Local != unknown]`, it was installed with\n")
cat("base install.packages() instead of renv::install(), which omits the\n")
cat("RemoteType stamp. Re-run: Rscript scripts/install_cuasmR.R  (see #133/#162).\n")
quit(status = 1)
