# binio.R -- low-level binary I/O helpers (internal, not exported).

read_u16_le <- function(raw, off) {
    bs <- as.integer(raw[(off + 1):(off + 2)])
    bs[1] + bitwShiftL(bs[2], 8)
}

read_u32_le <- function(raw, off) {
    bs <- as.integer(raw[(off + 1):(off + 4)])
    bs[1] + bitwShiftL(bs[2], 8) + bitwShiftL(bs[3], 16) + bitwShiftL(bs[4], 24)
}

read_u64_le <- function(raw, off) {
    # ELF offsets fit in IEEE-754 double mantissa (53 bits) so this is fine.
    lo <- read_u32_le(raw, off)
    hi <- read_u32_le(raw, off + 4)
    lo + hi * 4294967296
}

read_cstr <- function(raw, off) {
    p   <- off + 1
    end <- p
    while (end <= length(raw) && raw[end] != as.raw(0)) end <- end + 1
    if (end == p) "" else rawToChar(raw[p:(end - 1)])
}

# 64-bit word at off, returned as a 16-char "0x...." lowercase hex string.
read_u64hex <- function(raw, off) {
    bs <- raw[(off + 1):(off + 8)]
    paste0("0x", paste(rev(sprintf("%02x", as.integer(bs))), collapse = ""))
}

# Convert "0x...." or "...." hex string to 8 raw bytes (little-endian).
hex64_to_bytes <- function(hex) {
    # Type gate first: tolower()/sub() coerce silently, so an unquoted 0x1337 --
    # valid R for the double 4919 -- would stringify to "4919", clear the hex
    # check, and write a different value than the caller meant. See #170.
    if (!is.character(hex) || length(hex) != 1L) {
        stop("hex64_to_bytes: need a length-1 character string, got ",
             class(hex)[1L], " of length ", length(hex))
    }
    if (is.na(hex)) stop("hex64_to_bytes: hex is NA")
    s <- sub("^0x", "", tolower(hex))
    if (nchar(s) > 16) stop("hex64_to_bytes: too long: ", hex)
    # Reject non-hex BEFORE padding. A malformed 16-char token -- e.g. the
    # "              NA" that sprintf("%016x", NA) yields -- clears the length
    # check, makes every pair NA, and as.raw() coerces it to eight 00 bytes,
    # silently zeroing an instruction word. See #170, #197.
    if (!grepl("^[0-9a-f]{1,16}$", s)) stop("hex64_to_bytes: not hex: ", hex)
    # Zero-pad explicitly: flag = "0" is a NUMERIC format flag, so formatC()
    # right-justifies a character argument with SPACES.
    s <- paste0(strrep("0", 16L - nchar(s)), s)
    pairs <- substring(s, seq(1, 15, 2), seq(2, 16, 2))
    as.raw(rev(strtoi(pairs, 16L)))
}
