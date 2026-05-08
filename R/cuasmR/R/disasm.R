# disasm.R -- nvdisasm wrapper + parser.

# Run nvdisasm with --print-instruction-encoding and return raw lines.
run_nvdisasm <- function(cubin_path) {
    nvd <- Sys.which("nvdisasm")
    if (nvd == "") stop("cuasmR: nvdisasm not on PATH")
    out <- system2(nvd,
                   args = c("--print-instruction-encoding", shQuote(cubin_path)),
                   stdout = TRUE, stderr = TRUE)
    if (!is.null(attr(out, "status")) && attr(out, "status") != 0) {
        stop("cuasmR: nvdisasm failed:\n", paste(out, collapse = "\n"))
    }
    out
}

# Parse nvdisasm output into a data.frame of instructions.
# Columns: kernel, slot (0-based), address (hex string), text, instr_hex, ctrl_hex.
parse_nvdisasm <- function(lines) {
    # Section header for a kernel:
    #   .section  .text.<name>,"ax",@progbits
    # We capture <name> only -- a strict regex avoids matching .text.<name>
    # references inside other directives.
    sec_re  <- "^\\s*\\.section\\s+\\.text\\.([A-Za-z_][A-Za-z0-9_]*)\\s*,"

    # Instruction line (Ampere): two halves split across two lines
    #   /*0030*/   IMAD R6, R6, c[0x0][0x0], R3 ;     /* 0x0000000006067a24 */
    #                                                /* 0x001fca00078e0203 */
    insn_re <- "^\\s*/\\*([0-9a-fA-F]+)\\*/\\s+(.*?);\\s*/\\*\\s*(0x[0-9a-fA-F]+)\\s*\\*/\\s*$"
    ctrl_re <- "^\\s*/\\*\\s*(0x[0-9a-fA-F]+)\\s*\\*/\\s*$"

    kernels   <- character(0)
    slots     <- integer(0)
    addrs     <- character(0)
    texts     <- character(0)
    instrhx   <- character(0)
    ctrlhx    <- character(0)

    cur_kernel     <- NA_character_
    pending        <- NULL
    slot_in_kernel <- 0L

    for (line in lines) {
        m_sec <- regmatches(line, regexec(sec_re, line))[[1]]
        if (length(m_sec) >= 2) {
            cur_kernel     <- m_sec[2]
            slot_in_kernel <- 0L
            pending        <- NULL
            next
        }
        if (is.na(cur_kernel)) next

        m_in <- regmatches(line, regexec(insn_re, line))[[1]]
        if (length(m_in) >= 4) {
            pending <- list(addr = m_in[2],
                            text = trimws(m_in[3]),
                            ihex = m_in[4])
            next
        }
        if (!is.null(pending)) {
            m_ct <- regmatches(line, regexec(ctrl_re, line))[[1]]
            if (length(m_ct) >= 2) {
                kernels <- c(kernels, cur_kernel)
                slots   <- c(slots,   slot_in_kernel)
                addrs   <- c(addrs,   pending$addr)
                texts   <- c(texts,   pending$text)
                instrhx <- c(instrhx, pending$ihex)
                ctrlhx  <- c(ctrlhx,  m_ct[2])
                slot_in_kernel <- slot_in_kernel + 1L
                pending <- NULL
            }
        }
    }
    data.frame(kernel = kernels, slot = slots, address = addrs,
               text = texts, instr_hex = instrhx, ctrl_hex = ctrlhx,
               stringsAsFactors = FALSE)
}
