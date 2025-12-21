#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("usage: qbeta_ref.R <input.tsv>")
}
input_path <- args[1]

df <- read.table(input_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

if (!all(c("p", "a", "b", "lower_tail", "log_p") %in% names(df))) {
  stop("input must have columns: p, a, b, lower_tail, log_p")
}

vals <- mapply(function(p, a, b, lower_tail, log_p) {
  qbeta(p, a, b, lower.tail = lower_tail, log.p = log_p)
}, df$p, df$a, df$b, df$lower_tail, df$log_p)

out <- format(vals, digits = 17, scientific = TRUE)
writeLines(out)
