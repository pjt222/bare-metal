#!/usr/bin/env Rscript
# install_cuasmR.R -- (re)install the local cuasmR package into renv.
#
# Run after editing R/cuasmR/.

repo_root <- {
    args <- commandArgs(trailingOnly = FALSE)
    fa   <- grep("^--file=", args, value = TRUE)
    if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
    else            normalizePath(getwd())
}
pkg_dir <- file.path(repo_root, "R", "cuasmR")
if (!dir.exists(pkg_dir)) stop("cuasmR source not found at ", pkg_dir)

cat(sprintf("[install] cuasmR from %s\n", pkg_dir))
install.packages(pkg_dir, repos = NULL, type = "source", quiet = TRUE)

cat("[verify] library(cuasmR) ...\n")
library(cuasmR)
cat(sprintf("[ok] cuasmR %s loaded\n",
            as.character(packageVersion("cuasmR"))))
