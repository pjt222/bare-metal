#!/usr/bin/env Rscript
# gen_fa_traces.R
#
# Variant of gen_cymatic_data.R that injects Flash Attention block-level
# access traces in addition to (or in place of) the default trace set.
# Used by Issue #94 to measure whether cymatic helps real FA workloads.
#
# Usage:
#   Rscript gen_fa_traces.R [grid_n=1024] [n=6] [m=4] [domain=disc|square|overlayed]
#
# FA configuration assumed: Br=16 (queries/tile), Bc=64 (keys/tile).
# At seq=N: Nq = N/Br, Nk = N/Bc query/key tiles. Each FA iteration is
# one (q_tile, k_tile) pair. Trace = sequence of those pairs.
#
# We map (q_tile_idx, k_tile_idx) -> grid cell (i, j). The grid must be
# at least max(Nq, Nk) on each axis. Repo paths are resolved from the
# script location so invocation CWD does not matter.

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
script_dir <- if (length(file_arg)) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
} else {
    normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
source(normalizePath(file.path(script_dir, "..", "..", "..", "scripts", "cymatic", "cymatic_bench_io.R"),
                     winslash = "/", mustWork = TRUE))
cymatic_source("cymatic_mapping.R")
cymatic_source("cymatic_analyze.R")

args   <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 1024L
n1     <- if (length(args) >= 2) as.integer(args[2]) else 6L
m1     <- if (length(args) >= 3) as.integer(args[3]) else 4L
active_domain <- normalize_cymatic_domain(if (length(args) >= 4) args[4] else "disc")

out_dir <- cymatic_kernel_dir()

cat(sprintf("[gen_fa] grid=%d  mode=(%d, %d)  domain=%s\n", grid_n, n1, m1, active_domain))
mapping <- cymatic_mapping(grid_n = grid_n,
                           modes = list(list(n=n1, m=m1, alpha=1.0)),
                           active_domain = active_domain)

# ---- Row-major-inside ordering --------------------------------------------
inside_index <- build_inside_index(mapping)
inside_cells <- inside_index$inside_cells
n_inside <- inside_index$n_inside

trace_to_rmi <- function(cells) {
    cells_to_rmi(cells, inside_index$rmi_lookup)
}

perm <- extract_perm(mapping, inside_cells)

write_default_and_domain_artifact(write_perm_bin, "perm", "bin", active_domain,
                                  grid_n = grid_n, n_inside = n_inside, perm = perm)
cat(sprintf("[gen_fa] wrote perm.bin and perm_%s.bin  n_inside=%d  buffer=%.1f MB\n",
            active_domain, n_inside, n_inside * 4 / 1e6))

# ---- FA trace generators --------------------------------------------------

# Block-level FA: each iteration is one (q_block_idx, k_block_idx) pair.
# Per query block, scan all key blocks (causal mask off; full attention).
# Each (q, k) pair is mapped to D_HEAD adjacent cells in the row direction
# to model the per-element access within one block.
fa_block_trace <- function(seq_len, Br, Bc, D_head, grid_n,
                           order = c("rowmajor", "diagonal", "zigzag")) {
    order <- match.arg(order)
    Nq <- as.integer(seq_len / Br)
    Nk <- as.integer(seq_len / Bc)

    pairs <- switch(order,
        rowmajor = {
            ix <- expand.grid(q = 0:(Nq - 1), k = 0:(Nk - 1))
            ix[order(ix$q, ix$k), , drop = FALSE]
        },
        diagonal = {
            ix <- expand.grid(q = 0:(Nq - 1), k = 0:(Nk - 1))
            ix[order((ix$q + ix$k), ix$q), , drop = FALSE]
        },
        zigzag = {
            rows <- lapply(0:(Nq - 1), function(q) {
                ks <- if (q %% 2 == 0) 0:(Nk - 1) else (Nk - 1):0
                data.frame(q = q, k = ks)
            })
            do.call(rbind, rows)
        }
    )

    # For each (q, k) pair, the kernel touches Bc * D_head elements of K.
    # Model as a sequence of D_head cells at row k along the j axis,
    # repeated for each row in the K block.
    cells <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(idx) {
        q <- pairs$q[idx]; k <- pairs$k[idx]
        ki <- (k * Bc + seq.int(0, Bc - 1)) %% grid_n
        kj <- seq.int(0, D_head - 1L)
        cbind(rep(ki, each = D_head), rep(kj, times = Bc))
    }))
    storage.mode(cells) <- "integer"
    # Shift to 1-indexed within (1..grid_n)
    cells <- cells + 1L
    cells
}

# Configurations to test. seq=1024, Br=16, Bc=64, D=64 -> Nq=64, Nk=16.
# That fits comfortably in a 1024-grid (block addresses are 0..1023).
seqs   <- c(512L, 1024L)
orders <- c("rowmajor", "diagonal", "zigzag")
Br <- 16L; Bc <- 64L; D_head <- 64L

build_trace <- function(name, cells) {
    rmi <- trace_to_rmi(cells)
    list(name = name, rmi = as.integer(rmi))
}

traces <- list()
for (s in seqs) for (ord in orders) {
    cells <- fa_block_trace(s, Br, Bc, D_head, grid_n, order = ord)
    nm <- sprintf("fa_seq%d_%s", s, ord)
    tr <- build_trace(nm, cells)
    traces[[length(traces) + 1L]] <- tr
    cat(sprintf("  %-30s n=%d cells (raw=%d, in-disc kept)\n",
                nm, length(tr$rmi), nrow(cells)))
}

# Add the rowmajor_full control trace for direct comparison
traces[[length(traces) + 1L]] <- build_trace("rowmajor_full", inside_cells)
cat(sprintf("  %-30s n=%d cells (control)\n", "rowmajor_full", n_inside))

# ---- Write traces.bin -----------------------------------------------------
write_default_and_domain_artifact(write_traces_bin, "traces", "bin", active_domain,
                                  traces = traces)
cat(sprintf("[gen_fa] wrote traces.bin and traces_%s.bin (%d FA + 1 control trace)\n",
            active_domain, length(traces) - 1L))
