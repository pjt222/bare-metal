#!/usr/bin/env Rscript
# install_cuasmR.R -- (re)install the local cuasmR package into renv.
#
# Run after editing R/cuasmR/.
#
# MUST use renv::install(), not install.packages(). Both put a working
# cuasmR in the library, but only renv::install() stamps the installed
# DESCRIPTION with the `RemoteType: local` / `RemoteUrl: ./R/cuasmR`
# metadata that renv matches against the lockfile record. Installed via
# base `install.packages()`, the package reads back as source "unknown"
# and `renv::status()` reports:
#
#     The following package(s) are out of sync [lockfile != library]:
#     - cuasmR   [0.2.0: Local != unknown]
#
# ...forever, on an otherwise healthy project. Issue #133 fixed that
# discrepancy by hand in 2026-05-22 (588c773/c364f49) but left this
# script on install.packages(), so `make setup` silently reverted the fix
# and the project was out of sync again by 2026-07-29 (#162). Changing
# the script is the root fix; re-running the manual command is not.

repo_root <- {
    args <- commandArgs(trailingOnly = FALSE)
    fa   <- grep("^--file=", args, value = TRUE)
    if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
    else            normalizePath(getwd())
}
pkg_dir <- file.path(repo_root, "R", "cuasmR")
if (!dir.exists(pkg_dir)) stop("cuasmR source not found at ", pkg_dir)

if (!requireNamespace("renv", quietly = TRUE))
    stop("renv is not available; run `make setup` from the repo root first.")

# Relative path: renv records RemoteUrl verbatim, and an absolute path
# would bake this machine's layout into the lockfile on the next
# snapshot. The lockfile already carries "./R/cuasmR".
old_wd <- setwd(repo_root); on.exit(setwd(old_wd), add = TRUE)

cat(sprintf("[install] cuasmR from %s (via renv::install)\n", pkg_dir))
renv::install("./R/cuasmR")

# Prove the stamp landed. Without this, a future switch back to
# install.packages() -- or an renv change -- would silently reintroduce
# the permanent out-of-sync state, and the only symptom would be a
# `renv::status()` nobody runs.
desc <- file.path(.libPaths()[1], "cuasmR", "DESCRIPTION")
if (!file.exists(desc))
    stop("cuasmR not found in the library after install: ", desc)
fields <- read.dcf(desc)
if (!"RemoteType" %in% colnames(fields))
    stop("cuasmR installed WITHOUT renv remote metadata (no RemoteType in ",
         desc, "). renv::status() will report it permanently out of sync. ",
         "Do not fall back to install.packages() here -- see the header.")
cat(sprintf("[verify] RemoteType=%s RemoteUrl=%s\n",
            fields[1, "RemoteType"],
            if ("RemoteUrl" %in% colnames(fields)) fields[1, "RemoteUrl"] else "<none>"))

cat("[verify] library(cuasmR) ...\n")
library(cuasmR)
cat(sprintf("[ok] cuasmR %s loaded\n",
            as.character(packageVersion("cuasmR"))))
