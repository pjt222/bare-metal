#!/usr/bin/env Rscript
# scripts/audit/roxygen_check.R -- the cuasmR docs are generated (#225).
#
# R/cuasmR/man/*.Rd and R/cuasmR/NAMESPACE are generated from the roxygen
# blocks in R/cuasmR/R/*.R. There is one copy to edit: the roxygen block.
# The roxygen2 that generates them is the one renv.lock pins, and
# DESCRIPTION's RoxygenNote must name that same version. Both are enforced
# below, so a regeneration on any machine with the restored library
# produces the same bytes.
#
# Run from the repo root:
#   Rscript scripts/audit/roxygen_check.R           # check (make roxygen-check, CI)
#   Rscript scripts/audit/roxygen_check.R --write   # regenerate (make roxygen)
#
# The check regenerates into a temporary copy of the package and compares
# man/, NAMESPACE and DESCRIPTION with the tree, byte for byte, and refuses
# any man page without roxygen2's header. It never writes to the tree.
#
# Exit status: 0 = the tree matches what roxygen2 generates; 1 = drift
# (every differing file is named, with a diff); 2 = could not run.

package_dir <- "R/cuasmR"

fail_to_run <- function(...) {
  message("roxygen_check: ", ...)
  quit(save = "no", status = 2L)
}

cli_args <- commandArgs(trailingOnly = TRUE)
write_mode <- "--write" %in% cli_args
unknown_args <- setdiff(cli_args, "--write")
if (length(unknown_args)) {
  fail_to_run("unknown argument: ", unknown_args[[1]],
              "\nusage: Rscript scripts/audit/roxygen_check.R [--write]")
}

if (!file.exists(file.path(package_dir, "DESCRIPTION"))) {
  fail_to_run("run from the repo root (", package_dir,
              "/DESCRIPTION not found from ", getwd(), ")")
}
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  fail_to_run("roxygen2 is not installed. It is pinned in renv.lock: ",
              "run `make setup` (renv::restore()).")
}

roxygen_version <- as.character(utils::packageVersion("roxygen2"))
roxygen_note <- unname(read.dcf(file.path(package_dir, "DESCRIPTION"),
                                fields = "RoxygenNote")[1, 1])

# The installed roxygen2 must be the one renv.lock pins, in both modes:
# another version regenerates different bytes (and --write would rewrite
# every page and RoxygenNote with it). To upgrade, change the pin first
# (renv::install("roxygen2@<version>"), then snapshot), then `make roxygen`.
locked_version <- tryCatch(
  jsonlite::fromJSON("renv.lock", simplifyVector = FALSE)$Packages$roxygen2$Version,
  error = function(e) NULL)
if (is.null(locked_version)) {
  fail_to_run("renv.lock has no roxygen2 record, so there is no pinned version to check against.")
}
if (!identical(roxygen_version, locked_version)) {
  fail_to_run(sprintf(paste0(
    "roxygen2 %s is installed, but renv.lock pins %s. Run `make setup` ",
    "(renv::restore()). To upgrade roxygen2, change the pin in renv.lock first."),
    roxygen_version, locked_version))
}

if (write_mode) {
  regenerated <- tryCatch({
    roxygen2::roxygenise(package_dir)
    TRUE
  }, error = function(e) {
    message("roxygen_check: roxygen2::roxygenise() failed: ", conditionMessage(e))
    FALSE
  })
  if (!regenerated) quit(save = "no", status = 2L)
  message(sprintf("roxygen_check: regenerated %s with roxygen2 %s.",
                  package_dir, roxygen_version))
  quit(save = "no", status = 0L)
}

same_bytes <- function(path_a, path_b) {
  identical(readBin(path_a, "raw", file.size(path_a)),
            readBin(path_b, "raw", file.size(path_b)))
}

# Every regular file under man/, as paths relative to the package. Recursive,
# so a subdirectory such as man/figures is compared file by file instead of
# being handed to readBin() as if it were a file.
man_files <- function(root) {
  found <- list.files(file.path(root, "man"), recursive = TRUE, all.files = TRUE, no.. = TRUE)
  found <- found[!dir.exists(file.path(root, "man", found))]
  file.path("man", found)
}

# Check mode. A function, so that on.exit() runs: a top-level on.exit() in
# an Rscript does nothing. Returns the exit status instead of calling quit(),
# which would skip the cleanup.
run_check <- function() {
  scratch_root <- tempfile("roxygen_check_")
  dir.create(scratch_root)
  on.exit(unlink(scratch_root, recursive = TRUE), add = TRUE)
  if (!file.copy(package_dir, scratch_root, recursive = TRUE)) {
    message("roxygen_check: could not copy ", package_dir, " to ", scratch_root)
    return(2L)
  }
  generated_dir <- file.path(scratch_root, basename(package_dir))

  # roxygen2's messages are captured, not printed, so a passing check stays
  # one line; they are printed on any failure, because they carry the
  # explanation (e.g. "Skipping NAMESPACE" for a header-less file).
  roxygen_messages <- character()
  print_roxygen_messages <- function() {
    if (length(roxygen_messages)) {
      cat("roxygen2 said:\n")
      cat(sprintf("  %s\n", sub("\n+$", "", roxygen_messages)), sep = "")
    }
  }
  regenerate_ok <- tryCatch({
    withCallingHandlers(
      roxygen2::roxygenise(generated_dir),
      message = function(m) {
        roxygen_messages <<- c(roxygen_messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      })
    TRUE
  }, error = function(e) {
    print_roxygen_messages()
    message("roxygen_check: roxygen2::roxygenise() failed: ", conditionMessage(e))
    FALSE
  })
  if (!regenerate_ok) return(2L)

  compared_files <- c(sort(union(man_files(package_dir), man_files(generated_dir))),
                      "NAMESPACE", "DESCRIPTION")
  page_count <- sum(grepl("\\.Rd$", compared_files))

  drift <- character()

  # A RoxygenNote that names another version than the pinned roxygen2 is
  # drift in either direction. A newer note would otherwise pass: roxygen2
  # only remarks on it and leaves DESCRIPTION alone.
  if (!identical(roxygen_note, roxygen_version)) {
    drift <- c(drift, sprintf(paste0(
      "DESCRIPTION: RoxygenNote is %s, but the pinned roxygen2 is %s; ",
      "regenerate with `make roxygen` and commit"), roxygen_note, roxygen_version))
  }

  # roxygen2 overwrites (or deletes) a file only when it carries roxygen2's
  # header, and leaves a header-less file alone: a hand-written page or
  # NAMESPACE would then survive regeneration in both copies and compare
  # equal. Every page and the NAMESPACE must be generated, so refuse any
  # without the header.
  first_line_of <- function(relative_file) {
    readLines(file.path(package_dir, relative_file), n = 1L, warn = FALSE)
  }
  rd_header <- "% Generated by roxygen2: do not edit by hand"
  for (rd_file in grep("\\.Rd$", man_files(package_dir), value = TRUE)) {
    if (!identical(first_line_of(rd_file), rd_header)) {
      drift <- c(drift, sprintf("%s: no roxygen2 header, so it is hand-written; document it with a roxygen block instead", rd_file))
    }
  }
  namespace_header <- "# Generated by roxygen2: do not edit by hand"
  if (file.exists(file.path(package_dir, "NAMESPACE")) &&
      !identical(first_line_of("NAMESPACE"), namespace_header)) {
    drift <- c(drift, "NAMESPACE: no roxygen2 header, so it is hand-written; declare exports and imports with roxygen tags instead")
  }

  for (relative_file in compared_files) {
    tree_file <- file.path(package_dir, relative_file)
    generated_file <- file.path(generated_dir, relative_file)
    if (!file.exists(generated_file)) {
      drift <- c(drift, sprintf("%s: in the tree, but roxygen2 does not generate it (hand-written or stale)", relative_file))
    } else if (!file.exists(tree_file)) {
      drift <- c(drift, sprintf("%s: generated by roxygen2, missing from the tree", relative_file))
    } else if (!same_bytes(tree_file, generated_file)) {
      drift <- c(drift, sprintf("%s: differs from what roxygen2 generates", relative_file))
      diff_output <- suppressWarnings(system2("diff", c("-u", shQuote(tree_file), shQuote(generated_file)),
                                              stdout = TRUE, stderr = TRUE))
      cat(diff_output, sep = "\n")
    }
  }

  if (length(drift)) {
    cat(sprintf("\nroxygen_check: FAIL -- %d file(s) out of step with the roxygen blocks (roxygen2 %s, RoxygenNote %s):\n",
                length(drift), roxygen_version, roxygen_note))
    cat(sprintf("  - %s\n", drift), sep = "")
    cat("Edit the roxygen block in R/cuasmR/R/*.R, never the .Rd, then run `make roxygen` and commit the result.\n")
    print_roxygen_messages()
    return(1L)
  }

  cat(sprintf("roxygen_check: OK -- %d man page(s), NAMESPACE and DESCRIPTION match roxygen2 %s output.\n",
              page_count, roxygen_version))
  0L
}

quit(save = "no", status = run_check())
