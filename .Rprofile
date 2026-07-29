# .Rprofile -- activates the project's renv library on R startup.

# Do NOT run renv's project-synchronization check on every R startup (#162).
#
# renv's autoloader calls renv::load(), which calls
# renv_load_report_synchronized() -> status() -> a FULL project dependency
# scan. Rprof attributes 92% of renv::load() to that path; its self-time is
# dominated by filesystem stats (file.exists, file.info, dir.exists,
# Sys.readlink), which are expensive on this repo's 9p mount (/mnt/d, NTFS
# via WSL). Measured on AC power, a script that does nothing:
#
#     Rscript -e 'invisible(1)'      33.2 s  (check on)
#     Rscript -e 'invisible(1)'       2.0 s  (check off)
#
# ~16x, paid per process. It made the GPU-free test suite take an hour.
#
# This RELOCATES the check, it does not remove it. The same check runs
# deliberately via scripts/audit/renv_check.R, wired into the pre-push hook,
# `make setup` and `make renv-check` -- once per push instead of once per
# process. Dependency discovery is left deliberately UNWEAKENED: no
# .renvignore, and snapshot.type stays "implicit", so a new R script
# anywhere in the tree still registers as a dependency. Speeding the check
# up by making it scan less would trade correctness for time.
#
# Set here rather than in a project .Renviron on purpose: a project
# .Renviron SUPPRESSES the user's ~/.Renviron entirely (R reads only one),
# which would silently drop a contributor's personal settings. An option
# set here also does not leak into child processes as an env var. Both
# files are skipped by `Rscript --no-init-file` -- harmless, because that
# flag also skips renv activation, so there is no autoloader cost to avoid.
options(renv.config.synchronized.check = FALSE)

source("renv/activate.R")
