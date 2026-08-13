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

# ---------------------------------------------------------------------------
# 64-bit hex word bit/nibble editing (issues #198, #208).
#
# THE RULE: never hand a 64-bit SASS word to strtoi(). strtoi(x, 16L)
# returns an int32, so anything above 2,147,483,647 silently becomes NA --
# that is EVERY 64-bit word, and every 32-bit half whose leading hex digit
# is >= 8. The NA then propagates through bitwOr/bitwAnd, and
# sprintf("%016x", NA) yields the literal text "0x              NA", a
# 16-character token that passes every downstream length check and is
# written out as eight zero bytes. That is how a live FADD instruction
# word got zeroed while the test still passed (#170/#197), and how
# run_oxide.sh came to disagree with its own committed artifact (#198).
#
# These helpers edit the word AS DIGITS. A single hex digit is 0..15, so
# the only value ever passed to strtoi is one character wide and cannot
# overflow. Bit 0 is the LSB; bit 63 is the top bit of the leading digit.

# Validate + normalise to exactly 16 lowercase hex digits, no "0x".
.hex64_norm <- function(hex) {
    if (!is.character(hex) || length(hex) != 1L) {
        stop(".hex64_norm: need a length-1 character string, got ",
             class(hex)[1L], " of length ", length(hex))
    }
    if (is.na(hex)) stop(".hex64_norm: hex is NA")
    s <- sub("^0x", "", tolower(hex))
    if (nchar(s) > 16L) stop(".hex64_norm: too long: ", hex)
    if (!grepl("^[0-9a-f]{1,16}$", s)) stop(".hex64_norm: not hex: ", hex)
    paste0(strrep("0", 16L - nchar(s)), s)
}

.hex64_check_bit <- function(bit) {
    if (!is.numeric(bit) || length(bit) != 1L || is.na(bit) ||
        bit != as.integer(bit) || bit < 0L || bit > 63L) {
        stop("bit must be a single integer in 0..63, got: ",
             paste(format(bit), collapse = ","))
    }
    as.integer(bit)
}

#' Read one bit of a 64-bit hex word.
#'
#' @param hex A 64-bit word as a hex string, with or without \code{"0x"}.
#' @param bit Bit index, 0 (LSB) to 63.
#' @return \code{TRUE} or \code{FALSE}.
#' @export
hex64_bit_get <- function(hex, bit) {
    s   <- .hex64_norm(hex)
    bit <- .hex64_check_bit(bit)
    # Which hex digit holds this bit, counted from the LEFT (1-based).
    pos <- 16L - (bit %/% 4L)
    # One character: strtoi cannot overflow here, by construction.
    d <- strtoi(substr(s, pos, pos), 16L)
    bitwAnd(d, bitwShiftL(1L, bit %% 4L)) != 0L
}

#' Set or clear one bit of a 64-bit hex word.
#'
#' @param hex A 64-bit word as a hex string, with or without \code{"0x"}.
#' @param bit Bit index, 0 (LSB) to 63.
#' @param value \code{TRUE}/1 to set, \code{FALSE}/0 to clear.
#' @return The modified word as a 16-digit \code{"0x...."} string.
#' @export
hex64_bit_set <- function(hex, bit, value = TRUE) {
    s   <- .hex64_norm(hex)
    bit <- .hex64_check_bit(bit)
    # Reject anything as.logical() cannot parse, rather than letting
    # isTRUE(NA) collapse it to FALSE. hex64_bit_set(w, 22, "yes") would
    # otherwise CLEAR bit 22 -- the exact opposite of the caller's intent,
    # silently, in code that edits machine instructions.
    if (length(value) != 1L) stop("value must be a single TRUE/FALSE")
    on <- suppressWarnings(as.logical(value))
    if (is.na(on))
        stop("value must be a single non-NA TRUE/FALSE, got: ",
             paste(format(value), collapse = ","))
    pos <- 16L - (bit %/% 4L)
    d   <- strtoi(substr(s, pos, pos), 16L)
    mask <- bitwShiftL(1L, bit %% 4L)
    d <- if (on) bitwOr(d, mask) else bitwAnd(d, bitwNot(mask))
    substr(s, pos, pos) <- sprintf("%x", bitwAnd(d, 0xFL))
    paste0("0x", s)
}

#' Read one nibble (4-bit field) of a 64-bit hex word.
#'
#' A nibble IS one hex digit, so a field that happens to be nibble-aligned
#' -- like the SASS stall count in bits 40..43 -- needs no arithmetic on
#' the word at all.
#'
#' @param hex A 64-bit word as a hex string.
#' @param nibble Nibble index, 0 (bits 0..3) to 15 (bits 60..63).
#' @return Integer 0..15.
#' @export
hex64_nibble_get <- function(hex, nibble) {
    s <- .hex64_norm(hex)
    n <- .hex64_check_bit(nibble * 4L) %/% 4L
    pos <- 16L - n
    strtoi(substr(s, pos, pos), 16L)
}

#' Set one nibble (4-bit field) of a 64-bit hex word.
#'
#' @param hex A 64-bit word as a hex string.
#' @param nibble Nibble index, 0 (bits 0..3) to 15 (bits 60..63).
#' @param value Integer 0..15.
#' @return The modified word as a 16-digit \code{"0x...."} string.
#' @export
hex64_nibble_set <- function(hex, nibble, value) {
    s <- .hex64_norm(hex)
    n <- .hex64_check_bit(nibble * 4L) %/% 4L
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        value != as.integer(value) || value < 0L || value > 15L) {
        stop("value must be a single integer in 0..15, got: ",
             paste(format(value), collapse = ","))
    }
    pos <- 16L - n
    substr(s, pos, pos) <- sprintf("%x", as.integer(value))
    paste0("0x", s)
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
