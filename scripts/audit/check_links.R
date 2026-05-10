#!/usr/bin/env Rscript
# scripts/audit/check_links.R — verify markdown link targets exist
#
# Scans every README.md (excluding renv/, .git/, viz/.../node_modules/)
# for relative links of the form [text](path) and reports broken ones.
# Skips http(s) URLs, mailto:, anchors-only (#foo), and absolute paths.

suppressPackageStartupMessages({
  library(stringr)
})

repo_root <- normalizePath(".")

readmes <- list.files(
  ".",
  pattern = "^README\\.md$",
  recursive = TRUE,
  full.names = TRUE
)
readmes <- readmes[!grepl("renv/|\\.git/|node_modules/|R/cuasmR/", readmes)]

# Markdown link regex: [text](target)  — non-greedy text, no nested ]( allowed
# Captures the target.
link_re <- "\\[([^\\]]*)\\]\\(([^\\)]+)\\)"

broken <- list()
total_links <- 0
total_files <- 0

for (md in readmes) {
  total_files <- total_files + 1
  txt <- readLines(md, warn = FALSE)
  body <- paste(txt, collapse = "\n")

  matches <- str_match_all(body, link_re)[[1]]
  if (nrow(matches) == 0) next

  md_dir <- dirname(md)

  for (i in seq_len(nrow(matches))) {
    target <- matches[i, 3]
    total_links <- total_links + 1

    # Strip anchor fragment
    target_path <- sub("#.*$", "", target)

    # Skip externals + anchors-only + absolute paths
    if (target_path == "") next  # pure-anchor link [text](#section)
    if (grepl("^https?://", target_path)) next
    if (grepl("^mailto:", target_path)) next
    if (grepl("^ftp://", target_path)) next
    if (substr(target_path, 1, 1) == "/") next  # absolute path — skip
    if (grepl("^[a-zA-Z]+:", target_path)) next  # other URI schemes

    # Resolve relative to the README dir
    resolved <- file.path(md_dir, target_path)
    if (!file.exists(resolved)) {
      broken[[length(broken) + 1L]] <- list(
        readme = sub(paste0("^", repo_root, "/?"), "", normalizePath(md, mustWork = FALSE)),
        target = target,
        text   = matches[i, 2]
      )
    }
  }
}

cat(sprintf("Scanned %d README files, %d links.\n", total_files, total_links))
cat(sprintf("Broken: %d\n", length(broken)))

if (length(broken) > 0) {
  cat("\n=== broken links ===\n")
  by_file <- split(broken, sapply(broken, function(x) x$readme))
  for (f in names(by_file)) {
    cat(sprintf("\n%s\n", f))
    for (b in by_file[[f]]) {
      cat(sprintf("  -> %s   [text: %s]\n", b$target, b$text))
    }
  }
  quit(status = 1)
}
