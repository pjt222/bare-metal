#!/usr/bin/env Rscript

# Shared helpers for the cymatic benchmark generators.

cymatic_repo_root <- function() {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  start <- if (length(fa)) normalizePath(dirname(sub("^--file=", "", fa[1])), winslash = "/", mustWork = TRUE)
           else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  cur <- start
  repeat {
    if (file.exists(file.path(cur, ".git")) || file.exists(file.path(cur, "renv.lock"))) {
      return(cur)
    }
    parent <- dirname(cur)
    if (identical(parent, cur)) return(start)
    cur <- parent
  }
}

cymatic_source <- function(script_name) {
  source(file.path(cymatic_repo_root(), "scripts", "cymatic", script_name))
}

cymatic_kernel_dir <- function() {
  file.path(cymatic_repo_root(), "kernels", "memory_layout", "cymatic")
}

normalize_cymatic_domain <- function(domain = NULL) {
  domain <- tolower(if (is.null(domain) || !nzchar(domain)) "disc" else domain)
  if (!domain %in% c("disc", "square", "overlayed")) {
    stop(sprintf("unknown cymatic domain '%s' (expected 'disc', 'square', or 'overlayed')", domain))
  }
  domain
}

domain_artifact_path <- function(name, domain, ext) {
  file.path(cymatic_kernel_dir(), sprintf("%s_%s.%s", name, normalize_cymatic_domain(domain), ext))
}

build_inside_index <- function(mapping) {
  inside_cells <- which(mapping$grid$inside, arr.ind = TRUE)
  ord <- order(inside_cells[, 1], inside_cells[, 2])
  inside_cells <- inside_cells[ord, , drop = FALSE]
  keys <- paste(inside_cells[, 1], inside_cells[, 2], sep = "_")
  list(
    inside_cells = inside_cells,
    n_inside = nrow(inside_cells),
    rmi_lookup = setNames(seq_len(nrow(inside_cells)) - 1L, keys)
  )
}

cells_to_rmi <- function(cells, rmi_lookup) {
  keys <- paste(cells[, 1], cells[, 2], sep = "_")
  out <- unname(rmi_lookup[keys])
  out[!is.na(out)]
}

extract_perm <- function(mapping, inside_cells) {
  perm <- mapping$address[inside_cells] - 1L
  stopifnot(!any(is.na(perm)))
  stopifnot(min(perm) == 0L, max(perm) == length(perm) - 1L)
  stopifnot(length(unique(perm)) == length(perm))
  as.integer(perm)
}

write_perm_bin <- function(path, grid_n, n_inside, perm) {
  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.integer(grid_n), con, size = 4)
  writeBin(as.integer(n_inside), con, size = 4)
  writeBin(as.integer(perm), con, size = 4)
}

write_traces_bin <- function(path, traces) {
  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.integer(length(traces)), con, size = 4)
  for (tr in traces) {
    stopifnot(is.list(tr), !is.null(tr$name), !is.null(tr$rmi))
    name_bytes <- charToRaw(tr$name)
    writeBin(as.integer(length(name_bytes)), con, size = 4)
    writeBin(name_bytes, con)
    writeBin(as.integer(length(tr$rmi)), con, size = 4)
    writeBin(as.integer(tr$rmi), con, size = 4)
  }
}

write_default_and_domain_artifact <- function(writer, base_name, ext, domain, ...) {
  domain <- normalize_cymatic_domain(domain)
  writer(file.path(cymatic_kernel_dir(), sprintf("%s.%s", base_name, ext)), ...)
  writer(domain_artifact_path(base_name, domain, ext), ...)
}
