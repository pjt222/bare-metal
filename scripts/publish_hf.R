#!/usr/bin/env Rscript
# publish_hf.R -- publish the GA104 kernel corpus to the Hugging Face
# dataset repo pjt222/ga104-cuda-kernels (WS4, issue #109).
#
# Run from the repo root:
#   Rscript scripts/publish_hf.R            # full build + upload
#   Rscript scripts/publish_hf.R --dry-run  # resolve manifest only
#
# Stages (full run):
#   1. verify    -- toolchain + GPU present (scripts/verify_setup.R)
#   2. token     -- load HF_TOKEN from .env or the environment
#   3. build     -- make clean && make all && make disasm
#   4. manifest  -- assert every expected cubin/sass exists and is
#                   current; cross-check coverage vs data/baselines.json
#   5. checksums -- write SHA256SUMS over every cubin
#   6. card      -- render hf/README.md into the staging dir
#   7. upload    -- hf repo create + hf upload
#
# --dry-run prints the resolved manifest and the hf commands WITHOUT
# building, cleaning, rendering, or uploading. It is safe on a tree
# that has never been built.
#
# (uses base R only -- no library() loads needed)

# --------------------------------------------------------------------
# Repo root + WSL path/library fixups (mirrors scripts/verify_setup.R)
# --------------------------------------------------------------------
REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}

CUDA_BIN <- "/usr/local/cuda/bin"
if (dir.exists(CUDA_BIN) && !grepl(CUDA_BIN, Sys.getenv("PATH"), fixed = TRUE)) {
  Sys.setenv(PATH = paste(CUDA_BIN, Sys.getenv("PATH"), sep = ":"))
}
WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur)) paste(WSL_CUDA_LIB, cur, sep = ":") else WSL_CUDA_LIB)
}

PASS <- "\033[92m[PASS]\033[0m"
FAIL <- "\033[91m[FAIL]\033[0m"
INFO <- "\033[94m[INFO]\033[0m"

DATASET_REPO <- "pjt222/ga104-cuda-kernels"

# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------
die <- function(...) {
  cat(sprintf("%s %s\n", FAIL, paste0(..., collapse = "")))
  quit(status = 1L)
}

info <- function(...) cat(sprintf("%s %s\n", INFO, paste0(..., collapse = "")))
ok   <- function(...) cat(sprintf("%s %s\n", PASS, paste0(..., collapse = "")))

# Run a shell command, streaming output; stop the script on non-zero.
run_or_die <- function(cmd, label) {
  info(sprintf("%s: %s", label, cmd))
  status <- system(cmd)
  if (status != 0L) die(sprintf("%s failed (exit %d)", label, status))
}

# --------------------------------------------------------------------
# Stage 2: load HF_TOKEN
#
# Resolution order: an HF_TOKEN already in the environment wins; the
# repo-root .env is only a fallback. .env is parsed as KEY=VALUE lines
# (blank lines and # comments skipped, surrounding quotes stripped).
# --------------------------------------------------------------------
load_hf_token <- function() {
  token <- Sys.getenv("HF_TOKEN", unset = "")
  if (nzchar(token)) {
    info("HF_TOKEN found in the environment.")
    return(token)
  }
  env_path <- file.path(REPO_ROOT, ".env")
  if (file.exists(env_path)) {
    for (line in readLines(env_path, warn = FALSE)) {
      trimmed <- trimws(line)
      if (!nzchar(trimmed) || startsWith(trimmed, "#")) next
      eq <- regexpr("=", trimmed, fixed = TRUE)
      if (eq < 1L) next
      key <- trimws(substr(trimmed, 1L, eq - 1L))
      val <- trimws(substr(trimmed, eq + 1L, nchar(trimmed)))
      val <- gsub('^["\']|["\']$', "", val)
      if (key == "HF_TOKEN" && nzchar(val)) {
        info("HF_TOKEN loaded from .env")
        return(val)
      }
    }
  }
  die("No HF_TOKEN found. Set it in the environment or copy .env.example ",
      "to .env and paste a write-scoped token (see https://hf.co/settings/tokens).")
}

# --------------------------------------------------------------------
# Manifest construction
#
# The manifest is the explicit set of paths uploaded to the dataset
# repo, each as (src = path on disk, dest = path in the repo). Sources
# keep their repo-relative path; the regenerated cubin/sass set is
# placed under a generated/ prefix.
# --------------------------------------------------------------------

# Researcher-facing docs to publish (excludes CONTINUE_HERE.md and the
# session-scoped / build-internal docs).
DOC_FILES <- c(
  "inventory.md", "comparison_to_sota.md", "gpu_reflections.md",
  "index.md", "roofline_measured.md", "sass_histogram.md",
  "register_audit.md", "ampere_sass_reference.md", "control_codes.md",
  "memory_hierarchy.md", "cuasm_r.md"
)

# Tracked kernel sources: every .cu/.cuh under kernels/.
kernel_sources <- function() {
  list.files(file.path(REPO_ROOT, "kernels"),
             pattern = "\\.(cu|cuh)$", recursive = TRUE, full.names = TRUE)
}

# Tracked hand-tuned cubins. Discovered by glob, not hardcoded -- the
# .gitignore !*_handtuned.sm_86.cubin rule tracks any number of these.
handtuned_cubins <- function() {
  list.files(file.path(REPO_ROOT, "kernels"),
             pattern = "_handtuned\\.sm_86\\.cubin$",
             recursive = TRUE, full.names = TRUE)
}

# Build-output artifacts excluded from the published corpus -- they
# are real output of tracked sources but are not kernels of record:
#   - *.imma_s02/s04 cubins: negative-result hand-tune experiments
#     (#96 sub-task A; also gitignored as such).
#   - test_*/verify_* cubins/sass: correctness-test and layout-probe
#     binaries, not kernels.
# The matching .cu sources stay in the manifest (canonical source
# tree); only their generated/ build artifacts are dropped.
CORPUS_EXCLUDE <- c(
  "\\.imma_s0[24]\\.sm_86\\.(cubin|sass)$",
  "(^|/)(test|verify)_[^/]*\\.sm_86\\.(cubin|sass)$"
)

# Regenerated full cubin/sass set produced by `make all && make disasm`.
# Hand-tuned cubins are excluded here -- they ship under their tracked
# kernels/ path (their canonical home), not the generated/ supplement.
# CORPUS_EXCLUDE artifacts are dropped from the published supplement.
generated_artifacts <- function() {
  cubins <- list.files(file.path(REPO_ROOT, "kernels"),
                        pattern = "\\.sm_86\\.cubin$",
                        recursive = TRUE, full.names = TRUE)
  cubins <- cubins[!grepl("_handtuned\\.sm_86\\.cubin$", cubins)]
  sass <- list.files(file.path(REPO_ROOT, "kernels"),
                     pattern = "\\.sm_86\\.sass$",
                     recursive = TRUE, full.names = TRUE)
  arts <- c(cubins, sass)
  for (pat in CORPUS_EXCLUDE) arts <- arts[!grepl(pat, arts)]
  arts
}

# Tracked handtuned cubin paths, per git -- restored after `make clean`
# wipes them (see restore_handtuned_cubins).
tracked_handtuned <- function() {
  out <- tryCatch(
    system2("git", c("-C", shQuote(REPO_ROOT), "ls-files",
                     "*_handtuned.sm_86.cubin"), stdout = TRUE),
    error = function(e) character(0))
  out <- out[nzchar(out)]
  if (length(out)) file.path(REPO_ROOT, out) else character(0)
}

# `make clean` deletes every *.sm_86.cubin, including the hand-tuned
# cubins -- and `make all` cannot regenerate them (they come from a
# .cuasm hand-edit, not a .cu source). Restore them from git so the
# corpus is complete.
restore_handtuned_cubins <- function() {
  tracked <- tracked_handtuned()
  if (length(tracked) == 0L) {
    info("no tracked hand-tuned cubins to restore.")
    return(invisible())
  }
  rel <- vapply(tracked,
                function(p) sub(paste0("^", REPO_ROOT, "/"), "", p),
                character(1))
  status <- system2("git", c("-C", shQuote(REPO_ROOT), "checkout",
                              "HEAD", "--", shQuote(rel)))
  if (status != 0L) die("failed to restore tracked hand-tuned cubins.")
  missing <- tracked[!file.exists(tracked)]
  if (length(missing)) {
    die(sprintf("hand-tuned cubin missing after restore: %s",
                paste(missing, collapse = ", ")))
  }
  ok(sprintf("restored %d tracked hand-tuned cubin(s) from git.",
             length(tracked)))
}

# Build the (src, dest) manifest. Pure path arithmetic -- no file
# existence required, so this is safe to call during --dry-run.
build_manifest <- function() {
  rel <- function(p) sub(paste0("^", REPO_ROOT, "/"), "", p)
  m <- list()
  add <- function(src, dest) m[[length(m) + 1L]] <<- list(src = src, dest = dest)

  for (f in kernel_sources())   add(f, rel(f))
  for (f in handtuned_cubins()) add(f, rel(f))
  for (f in generated_artifacts()) add(f, file.path("generated", rel(f)))

  add(file.path(REPO_ROOT, "data"), "data")            # whole dir

  for (d in DOC_FILES) {
    p <- file.path(REPO_ROOT, "docs", d)
    add(p, file.path("docs", d))
  }

  add(file.path(REPO_ROOT, "AGENTS.md"),  "AGENTS.md")
  add(file.path(REPO_ROOT, "LICENSE"),    "LICENSE")
  add(file.path(REPO_ROOT, "SHA256SUMS"), "SHA256SUMS")

  m
}

# --------------------------------------------------------------------
# Stage 4: manifest assertion
#
# Every expected cubin/sass must exist and be newer than the .cu it
# was compiled from; every kernel tracked in data/baselines.json must
# have its cubin present. Any gap aborts -- a stale or near-empty
# snapshot must never upload.
# --------------------------------------------------------------------
assert_corpus_current <- function() {
  cubins <- list.files(file.path(REPO_ROOT, "kernels"),
                        pattern = "\\.sm_86\\.cubin$",
                        recursive = TRUE, full.names = TRUE)
  if (length(cubins) == 0L) {
    die("No .sm_86.cubin files under kernels/ -- the corpus did not build.")
  }
  # Hand-tuned cubins are checked separately below: they have no .cu
  # source and `make disasm` produces no .sass for them, so the
  # cu/sass currency checks below do not apply.
  generated_cubins <- cubins[!grepl("_handtuned\\.sm_86\\.cubin$", cubins)]

  problems <- character()

  for (cubin in generated_cubins) {
    cu <- sub("\\.sm_86\\.cubin$", ".cu", cubin)
    sass <- sub("\\.cubin$", ".sass", cubin)
    if (!file.exists(sass)) {
      problems <- c(problems, sprintf("missing disassembly: %s", sass))
      next
    }
    if (file.exists(cu) &&
        file.info(cubin)$mtime < file.info(cu)$mtime) {
      problems <- c(problems,
                    sprintf("stale cubin (older than source): %s", cubin))
    }
    if (file.info(sass)$mtime < file.info(cubin)$mtime) {
      problems <- c(problems,
                    sprintf("stale disassembly (older than cubin): %s", sass))
    }
  }

  # Coverage cross-check: every kernel keyed in baselines.json must
  # have produced a cubin.
  baselines_path <- file.path(REPO_ROOT, "data", "baselines.json")
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    die("package 'jsonlite' is required (run `make setup`).")
  }
  baselines <- jsonlite::fromJSON(baselines_path, simplifyVector = FALSE)
  for (kernel_cu in names(baselines$kernels)) {
    expected_cubin <- file.path(REPO_ROOT,
                                sub("\\.cu$", ".sm_86.cubin", kernel_cu))
    if (!file.exists(expected_cubin)) {
      problems <- c(problems,
                    sprintf("baselines.json kernel has no cubin: %s", kernel_cu))
    }
  }

  # Every cubin git tracks as hand-tuned must be present -- `make clean`
  # deletes them and `make all` cannot rebuild them.
  for (ht in tracked_handtuned()) {
    if (!file.exists(ht)) {
      problems <- c(problems,
                    sprintf("tracked hand-tuned cubin missing: %s", ht))
    }
  }

  if (length(problems)) {
    cat(sprintf("%s manifest assertion failed -- %d problem(s):\n",
                FAIL, length(problems)))
    for (p in problems) cat(sprintf("       - %s\n", p))
    quit(status = 1L)
  }
  ok(sprintf("manifest assertion passed (%d cubins, all current, ",
             length(cubins)),
     sprintf("%d baselines.json kernels covered).",
             length(baselines$kernels)))
}

# --------------------------------------------------------------------
# Stage 5: SHA256SUMS over every cubin
# --------------------------------------------------------------------
write_sha256sums <- function() {
  cubins <- c(
    list.files(file.path(REPO_ROOT, "kernels"),
               pattern = "\\.sm_86\\.cubin$",
               recursive = TRUE, full.names = TRUE)
  )
  cubins <- sort(cubins)
  rel <- function(p) sub(paste0("^", REPO_ROOT, "/"), "", p)
  lines <- vapply(cubins, function(p) {
    digest_out <- system2("sha256sum", shQuote(p), stdout = TRUE)
    hex <- sub("\\s.*$", "", digest_out[1])
    sprintf("%s  %s", hex, rel(p))
  }, character(1))
  out_path <- file.path(REPO_ROOT, "SHA256SUMS")
  writeLines(lines, out_path)
  ok(sprintf("wrote SHA256SUMS (%d cubins) -> %s", length(cubins), out_path))
}

# --------------------------------------------------------------------
# Stage 6: render the dataset card
# --------------------------------------------------------------------
render_card <- function(staging_dir) {
  template_path <- file.path(REPO_ROOT, "hf", "README.md")
  if (!file.exists(template_path)) {
    die("dataset card template not found at hf/README.md")
  }
  template <- paste(readLines(template_path, warn = FALSE), collapse = "\n")
  commit <- tryCatch(
    trimws(system2("git", c("-C", shQuote(REPO_ROOT), "rev-parse", "HEAD"),
                    stdout = TRUE)),
    error = function(e) "unknown")
  build_date <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  rendered <- gsub("{{COMMIT_SHA}}", commit,     template, fixed = TRUE)
  rendered <- gsub("{{BUILD_DATE}}", build_date, rendered, fixed = TRUE)
  out_path <- file.path(staging_dir, "README.md")
  writeLines(rendered, out_path)
  ok(sprintf("rendered dataset card (commit %s, %s)",
             substr(commit, 1, 12), build_date))
}

# --------------------------------------------------------------------
# Stage 7 helpers: assemble the staging dir + the hf commands
# --------------------------------------------------------------------
hf_commands <- function(staging_dir) {
  c(
    sprintf("hf repo create %s --type dataset", DATASET_REPO),
    sprintf("hf upload %s %s --repo-type dataset",
            DATASET_REPO, shQuote(staging_dir))
  )
}

stage_manifest <- function(manifest, staging_dir) {
  for (entry in manifest) {
    src <- entry$src
    dest <- file.path(staging_dir, entry$dest)
    if (!file.exists(src)) {
      die(sprintf("manifest entry missing on disk: %s", src))
    }
    if (dir.exists(src)) {
      dir.create(dest, recursive = TRUE, showWarnings = FALSE)
      file.copy(list.files(src, full.names = TRUE), dest,
                recursive = TRUE)
    } else {
      dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
      file.copy(src, dest, overwrite = TRUE)
    }
  }
}

# --------------------------------------------------------------------
# Dry-run report
# --------------------------------------------------------------------
print_dry_run <- function(manifest) {
  cat(strrep("=", 64), "\n")
  cat("  publish_hf.R --dry-run -- no build, no render, no upload\n")
  cat(strrep("=", 64), "\n\n")

  rel <- function(p) sub(paste0("^", REPO_ROOT, "/"), "", p)
  exists_count <- 0L
  missing_count <- 0L

  cat("Resolved upload manifest (src on disk -> dest in dataset repo):\n\n")
  for (entry in manifest) {
    present <- file.exists(entry$src)
    if (present) exists_count <- exists_count + 1L
    else         missing_count <- missing_count + 1L
    tag <- if (present) "  ok " else "MISS "
    cat(sprintf("  [%s] %-44s -> %s\n", tag, rel(entry$src), entry$dest))
  }

  ht <- handtuned_cubins()
  gen <- generated_artifacts()
  cat(sprintf("\n  hand-tuned cubins found: %d\n", length(ht)))
  cat(sprintf("  generated cubin/sass artifacts on disk: %d\n", length(gen)))
  cat(sprintf("  manifest entries: %d present, %d missing on disk\n",
              exists_count, missing_count))
  if (missing_count > 0L) {
    cat(sprintf("\n  %s missing entries are expected on an unbuilt tree;\n",
                INFO))
    cat("        a full run regenerates them before the manifest assertion.\n")
  }

  cat("\nhf commands a full run would issue:\n\n")
  for (cmd in hf_commands("<staging-dir>")) cat(sprintf("  %s\n", cmd))

  cat(sprintf("\nDataset URL: https://huggingface.co/datasets/%s\n",
              DATASET_REPO))
  cat(strrep("=", 64), "\n")
}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) && argv[1] %in% c("-h", "--help")) {
    cat("Usage: publish_hf.R [--dry-run]\n")
    cat("  (no args)   build the corpus and upload to Hugging Face\n")
    cat("  --dry-run   print the resolved manifest + hf commands only\n")
    quit(status = 0L)
  }
  dry_run <- "--dry-run" %in% argv

  cat(strrep("=", 64), "\n")
  cat("  bare-metal -- publish kernel corpus to Hugging Face\n")
  cat(sprintf("  Target dataset: %s\n", DATASET_REPO))
  cat(strrep("=", 64), "\n\n")

  if (dry_run) {
    # Dry-run: pure path arithmetic, no build, no token, no upload.
    print_dry_run(build_manifest())
    quit(status = 0L)
  }

  # --- Stage 1: verify toolchain + GPU -----------------------------
  info("Stage 1/7: verifying toolchain + GPU ...")
  verify_script <- file.path(REPO_ROOT, "scripts", "verify_setup.R")
  status <- system2("Rscript", shQuote(verify_script))
  if (status != 0L) die("environment verification failed (see above).")

  # --- Stage 2: HF token -------------------------------------------
  info("Stage 2/7: resolving HF_TOKEN ...")
  token <- load_hf_token()
  Sys.setenv(HF_TOKEN = token)

  # --- Stage 3: materialize the corpus -----------------------------
  info("Stage 3/7: rebuilding the corpus (make clean && make all && make disasm) ...")
  run_or_die(sprintf("make -C %s clean",  shQuote(REPO_ROOT)), "make clean")
  run_or_die(sprintf("make -C %s all",    shQuote(REPO_ROOT)), "make all")
  run_or_die(sprintf("make -C %s disasm", shQuote(REPO_ROOT)), "make disasm")
  # `make clean` also wiped the tracked hand-tuned cubins; restore them.
  restore_handtuned_cubins()

  # --- Stage 4: manifest assertion ---------------------------------
  info("Stage 4/7: asserting the corpus is complete and current ...")
  assert_corpus_current()

  # --- Stage 5: checksums ------------------------------------------
  info("Stage 5/7: writing SHA256SUMS ...")
  write_sha256sums()

  # --- Stage 6: stage files + render the card ----------------------
  info("Stage 6/7: staging files + rendering the dataset card ...")
  staging_dir <- tempfile("hf_publish_")
  dir.create(staging_dir)
  on.exit(unlink(staging_dir, recursive = TRUE), add = TRUE)
  manifest <- build_manifest()
  stage_manifest(manifest, staging_dir)
  render_card(staging_dir)
  ok(sprintf("staged %d manifest entries -> %s",
             length(manifest), staging_dir))

  # --- Stage 7: create the repo + upload ---------------------------
  info("Stage 7/7: creating the dataset repo + uploading ...")
  # hf repo create is idempotent: tolerate an existing repo.
  # hf >= 1.x rejects a 'datasets/' prefix together with --type; pass
  # the bare repo id and let --type select the dataset namespace.
  create_cmd <- sprintf("hf repo create %s --type dataset 2>&1",
                        DATASET_REPO)
  create_out <- suppressWarnings(system(create_cmd, intern = TRUE))
  if (!is.null(attr(create_out, "status")) &&
      attr(create_out, "status") != 0L &&
      !any(grepl("already (exists|created)|409", create_out,
                 ignore.case = TRUE))) {
    die(sprintf("hf repo create failed:\n%s",
                paste(create_out, collapse = "\n")))
  }
  ok("dataset repo ready.")

  run_or_die(sprintf("hf upload %s %s --repo-type dataset",
                     DATASET_REPO, shQuote(staging_dir)),
             "hf upload")

  cat("\n")
  ok("publish complete.")
  cat(sprintf("  Dataset URL: https://huggingface.co/datasets/%s\n",
              DATASET_REPO))
  0L
}

if (sys.nframe() == 0L) {
  rc <- main()
  quit(status = if (is.null(rc)) 0L else rc)
}
