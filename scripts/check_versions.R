#!/usr/bin/env Rscript
# scripts/check_versions.R — assert version strings agree across docs and renv.lock
#
# Checks:
#   1. CUDA version is consistent across README.md, AGENTS.md, SETUP.md
#   2. R version in AGENTS.md / SETUP.md matches the pin in renv.lock
#
# Run from the repo root:
#   Rscript scripts/check_versions.R
#
# Exit 0 = all versions agree.
# Exit 1 = at least one mismatch; details printed to stdout.

suppressPackageStartupMessages({
  library(jsonlite)
})

REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}

PASS <- "\033[92m[PASS]\033[0m"
FAIL <- "\033[91m[FAIL]\033[0m"
INFO <- "\033[94m[INFO]\033[0m"

# ---------------------------------------------------------------------------
# Helper: extract all CUDA major.minor strings from text
# Matches patterns: "CUDA 13.2", "cuda 13.2", "release 13.2", "cuda-13.2"
# ---------------------------------------------------------------------------
extract_cuda_versions <- function(text) {
  # Capture the version number that follows cuda/release keywords
  patterns <- c(
    "(?i)cuda[\\s-]+(\\d+\\.\\d+)",
    "(?i)release[\\s]+(\\d+\\.\\d+)"
  )
  found <- character(0)
  for (pat in patterns) {
    m <- gregexpr(pat, text, perl = TRUE)
    raw_matches <- regmatches(text, m)[[1]]
    if (length(raw_matches)) {
      # Extract just the version number portion
      nums <- sub(".*?(\\d+\\.\\d+)$", "\\1", raw_matches, perl = TRUE)
      found <- c(found, nums)
    }
  }
  unique(found)
}

# ---------------------------------------------------------------------------
# Helper: extract R major.minor strings from text
# Matches: "R 4.6.0", "R 4.6", "R-4.6.0"
# ---------------------------------------------------------------------------
extract_r_versions <- function(text) {
  m <- gregexpr("(?i)\\bR[\\s-]+(\\d+\\.\\d+(?:\\.\\d+)?)", text, perl = TRUE)
  raw_matches <- regmatches(text, m)[[1]]
  if (!length(raw_matches)) return(character(0))
  nums <- sub(".*?(\\d+\\.\\d+(?:\\.\\d+)?)$", "\\1", raw_matches, perl = TRUE)
  unique(nums)
}

# ---------------------------------------------------------------------------
# Read files
# ---------------------------------------------------------------------------
read_doc <- function(relative_path) {
  full <- file.path(REPO_ROOT, relative_path)
  if (!file.exists(full)) {
    cat(sprintf("%s File not found: %s\n", INFO, relative_path))
    return(NULL)
  }
  paste(readLines(full, warn = FALSE), collapse = "\n")
}

readme_text  <- read_doc("README.md")
agents_text  <- read_doc("AGENTS.md")
setup_text   <- read_doc("SETUP.md")
renv_path    <- file.path(REPO_ROOT, "renv.lock")

# ---------------------------------------------------------------------------
# Parse renv.lock for pinned R version
# ---------------------------------------------------------------------------
renv_r_version <- NA_character_
if (file.exists(renv_path)) {
  lock <- tryCatch(fromJSON(renv_path), error = function(e) NULL)
  if (!is.null(lock) && !is.null(lock$R$Version)) {
    renv_r_version <- lock$R$Version
  }
} else {
  cat(sprintf("%s renv.lock not found at %s\n", INFO, renv_path))
}

# ---------------------------------------------------------------------------
# Check 1: CUDA version consistency across docs
# ---------------------------------------------------------------------------
cat(strrep("=", 60), "\n")
cat("  bare-metal -- Version Consistency Check\n")
cat(strrep("=", 60), "\n\n")

results <- logical()

cat("-- CUDA version --\n")

cuda_sources <- list(
  "README.md" = if (!is.null(readme_text)) extract_cuda_versions(readme_text) else character(0),
  "AGENTS.md" = if (!is.null(agents_text)) extract_cuda_versions(agents_text) else character(0),
  "SETUP.md"  = if (!is.null(setup_text))  extract_cuda_versions(setup_text)  else character(0)
)

# Report what was found in each file
for (src_name in names(cuda_sources)) {
  versions <- cuda_sources[[src_name]]
  if (!length(versions)) {
    cat(sprintf("  %s  %-12s  (no CUDA version found)\n", INFO, src_name))
  } else {
    cat(sprintf("  found in %-12s: %s\n", src_name, paste(versions, collapse = ", ")))
  }
}

# Collect all unique CUDA versions across all docs
all_cuda_versions <- unique(unlist(cuda_sources))

if (length(all_cuda_versions) == 0L) {
  cat(sprintf("%s No CUDA version strings found in any doc\n\n", INFO))
  results <- c(results, TRUE)  # nothing to disagree
} else if (length(all_cuda_versions) == 1L) {
  cat(sprintf("%s CUDA version consistent: %s\n\n", PASS, all_cuda_versions))
  results <- c(results, TRUE)
} else {
  cat(sprintf("%s CUDA version MISMATCH across docs:\n", FAIL))
  for (src_name in names(cuda_sources)) {
    v <- cuda_sources[[src_name]]
    if (length(v)) cat(sprintf("       %-12s: %s\n", src_name, paste(v, collapse = ", ")))
  }
  cat("  Expected all docs to agree on a single CUDA release.\n\n")
  results <- c(results, FALSE)
}

# ---------------------------------------------------------------------------
# Check 2: R version — docs vs renv.lock pin
# ---------------------------------------------------------------------------
cat("-- R version --\n")

r_sources <- list(
  "AGENTS.md" = if (!is.null(agents_text)) extract_r_versions(agents_text) else character(0),
  "SETUP.md"  = if (!is.null(setup_text))  extract_r_versions(setup_text)  else character(0)
)

if (!is.na(renv_r_version)) {
  cat(sprintf("  renv.lock pin   : %s\n", renv_r_version))
} else {
  cat(sprintf("  %s  renv.lock R version not parsed\n", INFO))
}

for (src_name in names(r_sources)) {
  versions <- r_sources[[src_name]]
  if (!length(versions)) {
    cat(sprintf("  %-12s     : (no R version found)\n", src_name))
  } else {
    cat(sprintf("  %-12s     : %s\n", src_name, paste(versions, collapse = ", ")))
  }
}

# Build the set of all R version strings seen, normalised to major.minor
normalize_r <- function(v) sub("^(\\d+\\.\\d+).*", "\\1", v)

all_r_doc_versions <- unique(unlist(r_sources))

if (!is.na(renv_r_version)) {
  combined_r <- unique(c(normalize_r(renv_r_version), sapply(all_r_doc_versions, normalize_r)))
} else {
  combined_r <- unique(sapply(all_r_doc_versions, normalize_r))
}

if (length(combined_r) <= 1L) {
  cat(sprintf("%s R version consistent", PASS))
  if (!is.na(renv_r_version)) cat(sprintf(": %s (renv.lock + docs)", renv_r_version))
  cat("\n\n")
  results <- c(results, TRUE)
} else {
  cat(sprintf("%s R version MISMATCH:\n", FAIL))
  if (!is.na(renv_r_version))
    cat(sprintf("       renv.lock       : %s\n", renv_r_version))
  for (src_name in names(r_sources)) {
    v <- r_sources[[src_name]]
    if (length(v)) cat(sprintf("       %-12s: %s\n", src_name, paste(v, collapse = ", ")))
  }
  cat("  Expected docs and renv.lock to agree on the same R major.minor.\n\n")
  results <- c(results, FALSE)
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed <- sum(results)
total  <- length(results)
cat(strrep("=", 60), "\n")
if (passed == total) {
  cat(sprintf("%s All %d version checks passed\n", PASS, total))
} else {
  cat(sprintf("%s %d/%d version checks failed -- see details above\n",
              FAIL, total - passed, total))
}
cat(strrep("=", 60), "\n")

if (sys.nframe() == 0L) {
  quit(status = if (passed == total) 0L else 1L)
}
