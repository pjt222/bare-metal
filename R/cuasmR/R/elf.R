# elf.R -- minimal ELF64 reader, scoped to what we need from cubins.

# Parse the ELF64 section header table from a raw byte vector.
# Returns a data.frame with one row per section.
parse_sections <- function(raw) {
    e_shoff     <- read_u64_le(raw, 40)
    e_shentsize <- read_u16_le(raw, 58)
    e_shnum     <- read_u16_le(raw, 60)
    e_shstrndx  <- read_u16_le(raw, 62)

    shstrtab_off  <- read_u64_le(raw, e_shoff + e_shentsize * e_shstrndx + 24)
    shstrtab_size <- read_u64_le(raw, e_shoff + e_shentsize * e_shstrndx + 32)
    strs <- raw[(shstrtab_off + 1):(shstrtab_off + shstrtab_size)]

    # Allocate vectors up front, then assemble at the end. This avoids the
    # data.frame[r, ] <- list(...) reassignment which calls as.character()
    # on each list element and corrupted ".text.vector_add" -> 'vector_add,"ax"@progbits'
    # in a previous iteration of this code (the loop wrote a name for the
    # WRONG section because of factor coercion in default data.frame).
    idx <- integer(e_shnum)
    nm  <- character(e_shnum)
    off <- numeric(e_shnum)
    sz  <- numeric(e_shnum)
    ty  <- numeric(e_shnum)

    for (i in 0:(e_shnum - 1)) {
        base <- e_shoff + e_shentsize * i
        idx[i + 1] <- i
        nm [i + 1] <- read_cstr(strs, read_u32_le(raw, base))
        ty [i + 1] <- read_u32_le(raw, base + 4)
        off[i + 1] <- read_u64_le(raw, base + 24)
        sz [i + 1] <- read_u64_le(raw, base + 32)
    }
    data.frame(index = idx, name = nm, type = ty,
               offset = off, size = sz,
               stringsAsFactors = FALSE)
}

# CUDA 12.x vs 13.x e_flags layout decoder.
# Returns a list(sm_version, vsm_version, layout = "cuda12" | "cuda13").
parse_e_flags <- function(raw) {
    ef    <- read_u32_le(raw, 48)
    byte0 <- bitwAnd(ef,                0xFF)
    byte1 <- bitwAnd(bitwShiftR(ef,  8), 0xFF)
    byte2 <- bitwAnd(bitwShiftR(ef, 16), 0xFF)

    known <- c(35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75,
               80, 86, 87, 89, 90)

    if (byte0 %in% known) {
        list(sm_version = byte0, vsm_version = byte2, layout = "cuda12",
             e_flags = ef)
    } else if (byte1 %in% known) {
        list(sm_version = byte1, vsm_version = byte1, layout = "cuda13",
             e_flags = ef)
    } else {
        stop(sprintf(
            "cuasmR: unrecognized e_flags = 0x%08x (byte0=%d byte1=%d byte2=%d)",
            ef, byte0, byte1, byte2))
    }
}
