#!/usr/bin/env Rscript
# scripts/audit/roxygen_check.R -- the cuasmR docs are generated (#225).
#
# R/cuasmR/man/*.Rd and R/cuasmR/NAMESPACE are generated from the roxygen
# blocks in R/cuasmR/R/*.R. There is one copy to edit: the roxygen block.
#
# Run from the repo root:
#   Rscript scripts/audit/roxygen_check.R           # check  (make roxygen-check; CI runs the script)
#   Rscript scripts/audit/roxygen_check.R --write   # regenerate in place (make roxygen)
#
# Both modes first require the installed roxygen2 to be the version
# renv.lock pins, so every machine with the restored library generates
# with the same roxygen2. (R itself is not pinned locally; see #230.)
#
# The check regenerates into a temporary copy of the package, never
# writing to the tree, and fails on:
#   - any file under man/, or NAMESPACE or DESCRIPTION, that differs
#     byte for byte from what roxygen2 generates;
#   - any man page or NAMESPACE without roxygen2's header (roxygen2 leaves
#     such files alone, so they would otherwise compare equal);
#   - a DESCRIPTION RoxygenNote that is not the pinned version.
#
# Exit status: 0 = the tree matches; 1 = drift (each problem named, with
# a diff and roxygen2's messages); 2 = could not run (wrong roxygen2,
# unreadable renv.lock, roxygen2 error, any unexpected error). Through
# make, both 1 and 2 surface as make's own exit status 2. One failure is
# outside this script's reach: a renv.lock that is not valid JSON breaks
# renv's own activation (.Rprofile sources renv/activate.R) before the
# script runs, and R exits 1.

package_dir <- "R/cuasmR"

fail_to_run <- function(...) {
  message("roxygen_check: ", ...)
  quit(save = "no", status = 2L)
}

# The roxygen2 version renv.lock pins. Each way this can fail gets its own
# message, so "no pinned version" is never reported for a lock that is
# missing, unparseable, or unreadable for want of jsonlite.
read_locked_roxygen_version <- function() {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    fail_to_run("jsonlite is not installed, so renv.lock cannot be read. Run `make setup` (renv::restore()).")
  }
  if (!file.exists("renv.lock")) fail_to_run("renv.lock not found in ", getwd())
  lock <- tryCatch(jsonlite::fromJSON("renv.lock", simplifyVector = FALSE),
                   error = function(e) fail_to_run("renv.lock could not be parsed: ", conditionMessage(e)))
  version <- lock$Packages$roxygen2$Version
  if (is.null(version)) {
    fail_to_run("renv.lock has no roxygen2 record, so there is no pinned version to check against.")
  }
  version
}

main <- function() {
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

  roxygen_version <<- as.character(utils::packageVersion("roxygen2"))
  roxygen_note <<- unname(read.dcf(file.path(package_dir, "DESCRIPTION"),
                                   fields = "RoxygenNote")[1, 1])

  # The installed roxygen2 must be the one renv.lock pins, in both modes:
  # another version regenerates different bytes (and --write would rewrite
  # every page and RoxygenNote with it). To upgrade, follow AGENTS.md
  # ("cuasmR's man pages are generated").
  locked_version <- read_locked_roxygen_version()
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
    if (!regenerated) return(2L)
    message(sprintf("roxygen_check: regenerated %s with roxygen2 %s.",
                    package_dir, roxygen_version))
    return(0L)
  }
  run_check()
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

  # roxygen2's messages are captured rather than streamed. They are printed
  # in full on any failure, because they carry the explanation (e.g.
  # "Skipping NAMESPACE" for a header-less file). On a pass, everything
  # except the routine "Loading" line is printed too, so a tag warning
  # (e.g. a misspelt @retrun) is not hidden by a matching regeneration.
  roxygen_messages <- character()
  print_roxygen_messages <- function(skip_routine = FALSE) {
    shown <- sub("\n+$", "", roxygen_messages)
    if (skip_routine) shown <- shown[!grepl("^\\S*\\s*Loading ", shown)]
    if (length(shown)) {
      cat("roxygen2 said:\n")
      cat(sprintf("  %s\n", shown), sep = "")
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
  page_count <- sum(grepl("\\.[Rr]d$", compared_files))

  drift <- character()

  # A RoxygenNote that names another version than the pinned roxygen2 is
  # drift in either direction. A newer note would otherwise pass: roxygen2
  # only remarks on it and leaves DESCRIPTION alone, which is also why
  # `make roxygen` cannot repair it.
  if (!identical(roxygen_note, roxygen_version)) {
    if (utils::compareVersion(roxygen_note, roxygen_version) < 0) {
      drift <- c(drift, sprintf(paste0(
        "DESCRIPTION: RoxygenNote is %s, older than the pinned roxygen2 %s; ",
        "`make roxygen` updates it, then commit"), roxygen_note, roxygen_version))
    } else {
      drift <- c(drift, sprintf(paste0(
        "DESCRIPTION: RoxygenNote is %s, newer than the pinned roxygen2 %s, so the docs ",
        "were regenerated with another roxygen2. `make roxygen` will not lower it: either set ",
        "RoxygenNote back to %s by hand and run `make roxygen`, or upgrade the pin ",
        "(AGENTS.md, cuasmR's man pages)"), roxygen_note, roxygen_version, roxygen_version))
    }
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
  for (rd_file in grep("\\.[Rr]d$", man_files(package_dir), value = TRUE)) {
    if (!identical(first_line_of(rd_file), rd_header)) {
      drift <- c(drift, sprintf("%s: no roxygen2 header, so it is hand-written; document it with a roxygen block instead", rd_file))
    }
  }
  namespace_header <- "# Generated by roxygen2: do not edit by hand"
  if (file.exists(file.path(package_dir, "NAMESPACE")) &&
      !identical(first_line_of("NAMESPACE"), namespace_header)) {
    drift <- c(drift, paste0(
      "NAMESPACE: no roxygen2 header, so it is hand-written, and `make roxygen` will skip it too. ",
      "Declare its exports and imports with roxygen tags, delete NAMESPACE (or restore its header ",
      "line), then run `make roxygen`"))
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
    cat(sprintf("\nroxygen_check: FAIL -- %d problem(s) with the generated docs (roxygen2 %s, RoxygenNote %s):\n",
                length(drift), roxygen_version, roxygen_note))
    cat(sprintf("  - %s\n", drift), sep = "")
    cat("Fix each as described. Edit roxygen blocks in R/cuasmR/R/*.R, never an .Rd; `make roxygen` regenerates.\n")
    print_roxygen_messages()
    return(1L)
  }

  print_roxygen_messages(skip_routine = TRUE)
  cat(sprintf("roxygen_check: OK -- %d man page(s), NAMESPACE and DESCRIPTION match roxygen2 %s output.\n",
              page_count, roxygen_version))
  0L
}

# Any error not handled above (a malformed DESCRIPTION, an unreadable
# file) is "could not run", exit 2, not R's default status 1, which the
# contract reserves for drift. on.exit() in run_check() still runs while
# the error unwinds.
roxygen_version <- NA_character_
roxygen_note <- NA_character_
exit_status <- tryCatch(main(), error = function(e) {
  message("roxygen_check: could not run: ", conditionMessage(e))
  2L
})
quit(save = "no", status = exit_status)
