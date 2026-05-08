# api.R -- public functions exported by the package.

#' Read a CUDA cubin file and decode its SASS.
#'
#' @param cubin_path Path to the cubin (e.g. produced by
#'   \code{nvcc --cubin -arch=sm_86 ...}).
#' @return An object of class \code{"cuasm"} -- a list with elements
#'   \code{path}, \code{raw} (raw byte vector), \code{sections}
#'   (data.frame), \code{kernels} (data.frame), \code{insns}
#'   (data.frame), \code{arch} (sm_version etc.).
#' @export
cuasm_read <- function(cubin_path) {
    if (!file.exists(cubin_path)) stop("cubin not found: ", cubin_path)
    raw <- readBin(cubin_path, "raw", n = file.info(cubin_path)$size)

    sections <- parse_sections(raw)
    arch     <- parse_e_flags(raw)

    text_secs <- sections[grep("^\\.text\\.", sections$name), , drop = FALSE]
    text_secs$kernel <- sub("^\\.text\\.", "", text_secs$name)
    rownames(text_secs) <- NULL

    insns <- parse_nvdisasm(run_nvdisasm(cubin_path))

    # Cross-check observed instruction counts vs section sizes / 16.
    for (k in unique(insns$kernel)) {
        sec <- text_secs[text_secs$kernel == k, , drop = FALSE]
        if (nrow(sec) != 1)
            stop("kernel ", k, ": ", nrow(sec), " matching .text sections")
        n_obs <- sum(insns$kernel == k)
        n_exp <- sec$size / 16
        if (n_obs != n_exp) {
            warning(sprintf(
                "kernel %s: nvdisasm produced %d insns but section size %d => %d slots",
                k, n_obs, sec$size, n_exp))
        }
    }

    structure(
        list(path     = normalizePath(cubin_path),
             raw      = raw,
             sections = sections,
             kernels  = text_secs[, c("kernel", "offset", "size")],
             insns    = insns,
             arch     = arch),
        class = "cuasm")
}

#' Show all ELF sections in a cuasm object.
#' @param obj A \code{cuasm} object.
#' @return data.frame with index, name, type, offset, size.
#' @export
cuasm_sections <- function(obj) {
    stopifnot(inherits(obj, "cuasm"))
    obj$sections
}

#' List the kernel \code{.text} sections in a cubin.
#' @param obj A \code{cuasm} object.
#' @return data.frame with kernel, offset (file offset), size.
#' @export
cuasm_kernels <- function(obj) {
    stopifnot(inherits(obj, "cuasm"))
    obj$kernels
}

#' Get decoded SASS instructions, optionally filtered to one kernel.
#' @param obj A \code{cuasm} object.
#' @param kernel Optional kernel name (string).
#' @return data.frame with kernel, slot, address, text, instr_hex, ctrl_hex.
#' @export
cuasm_insns <- function(obj, kernel = NULL) {
    stopifnot(inherits(obj, "cuasm"))
    if (is.null(kernel)) return(obj$insns)
    obj$insns[obj$insns$kernel == kernel, , drop = FALSE]
}

#' Patch a single instruction at (kernel, slot).
#'
#' Either or both of \code{instr_hex} and \code{ctrl_hex} may be supplied.
#' Hex strings should be 16 hex digits with optional "0x" prefix
#' (i.e. one 64-bit word).
#'
#' @param obj A \code{cuasm} object.
#' @param kernel Kernel name (string).
#' @param slot 0-based instruction slot within the kernel.
#' @param instr_hex Optional new instruction word.
#' @param ctrl_hex Optional new control word.
#' @return Updated \code{cuasm} object.
#' @export
cuasm_set <- function(obj, kernel, slot, instr_hex = NULL, ctrl_hex = NULL) {
    stopifnot(inherits(obj, "cuasm"))
    rows <- which(obj$insns$kernel == kernel & obj$insns$slot == slot)
    if (length(rows) != 1)
        stop("cuasm_set: no match (kernel=", kernel, ", slot=", slot, ")")
    if (!is.null(instr_hex)) obj$insns$instr_hex[rows] <- instr_hex
    if (!is.null(ctrl_hex))  obj$insns$ctrl_hex [rows] <- ctrl_hex
    obj
}

#' Write a (possibly patched) cubin to disk.
#'
#' Bytes outside the kernel \code{.text} sections are copied verbatim
#' from the input. Within each \code{.text} section, the 16-byte
#' instruction slots are written from \code{obj$insns}.
#'
#' @param obj A \code{cuasm} object.
#' @param out_path Destination path.
#' @return out_path (invisibly).
#' @export
cuasm_write <- function(obj, out_path) {
    stopifnot(inherits(obj, "cuasm"))
    out <- obj$raw

    for (i in seq_len(nrow(obj$kernels))) {
        k    <- obj$kernels$kernel[i]
        base <- obj$kernels$offset[i]
        size <- obj$kernels$size[i]
        n    <- size / 16

        rows <- which(obj$insns$kernel == k)
        if (length(rows) != n) {
            stop(sprintf("kernel %s: %d insns vs %d slots in .text",
                         k, length(rows), n))
        }
        rows <- rows[order(obj$insns$slot[rows])]
        for (j in seq_along(rows)) {
            r       <- rows[j]
            slot_off <- base + (j - 1L) * 16L
            ibs <- hex64_to_bytes(obj$insns$instr_hex[r])
            cbs <- hex64_to_bytes(obj$insns$ctrl_hex [r])
            out[(slot_off + 1L):(slot_off + 8L)]  <- ibs
            out[(slot_off + 9L):(slot_off + 16L)] <- cbs
        }
    }

    writeBin(out, out_path)
    invisible(out_path)
}

#' Verify byte-identical roundtrip: read -> write -> diff.
#' @param cubin_path Path to a cubin file.
#' @return TRUE iff the rewritten cubin is byte-identical to the source.
#' @export
cuasm_roundtrip_check <- function(cubin_path) {
    obj <- cuasm_read(cubin_path)
    scratch <- tempfile(fileext = ".cubin")
    on.exit(unlink(scratch), add = TRUE)
    cuasm_write(obj, scratch)
    raw_in  <- readBin(cubin_path, "raw", n = file.info(cubin_path)$size)
    raw_out <- readBin(scratch,    "raw", n = file.info(scratch)$size)
    identical(raw_in, raw_out)
}

#' Save a human-readable .cuasm-style text dump.
#' @param obj A \code{cuasm} object.
#' @param out_path Destination path (typically \code{.cuasm}).
#' @return out_path (invisibly).
#' @export
cuasm_save_cuasm <- function(obj, out_path) {
    stopifnot(inherits(obj, "cuasm"))
    con <- file(out_path, "w"); on.exit(close(con))

    cat("// cuasm v0 -- generated by cuasmR\n", file = con)
    cat(sprintf("// source: %s\n", obj$path), file = con)
    cat(sprintf("// arch: sm_%d (layout %s, e_flags=0x%08x)\n\n",
                obj$arch$sm_version, obj$arch$layout, obj$arch$e_flags),
        file = con)

    for (i in seq_len(nrow(obj$kernels))) {
        k <- obj$kernels$kernel[i]
        cat(sprintf(".section .text.%s,\"ax\",@progbits\n", k), file = con)
        cat(sprintf("//   file offset 0x%x  size %d  (%d insns)\n",
                    obj$kernels$offset[i], obj$kernels$size[i],
                    obj$kernels$size[i] / 16), file = con)
        rows <- which(obj$insns$kernel == k)
        rows <- rows[order(obj$insns$slot[rows])]
        for (r in rows) {
            cat(sprintf("    /*%4s*/ slot=%-3d  %-50s ; /* %s */ /* %s */\n",
                        obj$insns$address[r], obj$insns$slot[r],
                        obj$insns$text[r],
                        obj$insns$instr_hex[r], obj$insns$ctrl_hex[r]),
                file = con)
        }
        cat("\n", file = con)
    }
    invisible(out_path)
}
