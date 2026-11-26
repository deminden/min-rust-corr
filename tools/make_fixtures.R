#!/usr/bin/env Rscript
# make_fixtures.R — generate black-box reference fixtures for mincorr tests.
#
# Uses only public R APIs (stats, WGCNA::bicor/bicorAndPvalue, HellCor::HellCor)
# as numerical oracles. Writes small deterministic TSV fixtures into
# tests/fixtures/ plus MANIFEST.json (R/package versions, seed).
#
# Usage: Rscript tools/make_fixtures.R [out_dir]
# Sections needing WGCNA or HellCor are skipped (and recorded as skipped in the
# manifest) when the package is not installed.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "tests/fixtures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
SEED <- 20261124L

fmt <- function(v) ifelse(is.na(v), "NaN", sprintf("%.17g", v))

write_matrix <- function(m, name) {
  path <- file.path(out_dir, paste0(name, ".tsv"))
  con <- file(path, "w")
  rn <- if (is.null(rownames(m))) paste0("r", seq_len(nrow(m))) else rownames(m)
  writeLines(paste(c("id", paste0("c", seq_len(ncol(m)))), collapse = "\t"), con)
  for (i in seq_len(nrow(m))) {
    writeLines(paste(c(rn[i], fmt(m[i, ])), collapse = "\t"), con)
  }
  close(con)
}

# ---------------------------------------------------------------- datasets --
set.seed(SEED)
mk <- function(r, c, gen) matrix(gen(r * c), nrow = r, ncol = c)
datasets <- list()
datasets$normal_30x40 <- mk(30, 40, rnorm)
datasets$normal_8x200 <- mk(8, 200, rnorm)
datasets$lognormal_20x77 <- mk(20, 77, rlnorm)
datasets$ties_15x30 <- mk(15, 30, function(k) as.numeric(sample(1:5, k, replace = TRUE)))
out <- mk(15, 50, rnorm)
hit <- sample(length(out), round(0.05 * length(out)))
out[hit] <- out[hit] + sample(c(-1, 1), length(hit), replace = TRUE) * runif(length(hit), 8, 20)
datasets$outliers_15x50 <- out
cr <- mk(6, 20, rnorm)
cr[1, ] <- 3
cr[2, ] <- -1.5
cr[3, ] <- c(rep(2, 16), 5, 6, -1, 2.5) # MAD = 0, variance > 0
cr[4, ] <- c(rep(0, 15), 1, 2, 3, 4, 5) # MAD = 0, variance > 0
datasets$constant_rows_6x20 <- cr
na <- mk(12, 40, rnorm)
na[sample(length(na), round(0.05 * length(na)))] <- NA
datasets$na_12x40 <- na
datasets$tiny_5x3 <- mk(5, 3, rnorm)
datasets$tiny_5x4 <- mk(5, 4, rnorm)
x <- sort(runif(50, 0.1, 3))
x <- sample(x)
datasets$monotone_pairs <- rbind(x, exp(x), -x^3, log(x), 1 / x, x^2 - 2 * x, rnorm(50))

# ------------------------------------------------------------ correlations --
pair_p <- function(m, method) {
  n <- nrow(m)
  p <- matrix(NA_real_, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) {
    if (i == j) next
    ok <- is.finite(m[i, ]) & is.finite(m[j, ])
    p[i, j] <- tryCatch(
      suppressWarnings(cor.test(m[i, ok], m[j, ok], method = method, exact = FALSE)$p.value),
      error = function(e) NA_real_)
  }
  p
}

skipped <- character(0)
have_wgcna <- requireNamespace("WGCNA", quietly = TRUE)
have_hellcor <- requireNamespace("HellCor", quietly = TRUE)
if (!have_wgcna) skipped <- c(skipped, "bicor (WGCNA not installed)")
if (!have_hellcor) skipped <- c(skipped, "hellcor (HellCor not installed)")

for (nm in names(datasets)) {
  m <- datasets[[nm]]
  write_matrix(m, paste0(nm, "__input"))
  for (method in c("pearson", "spearman", "kendall")) {
    r <- suppressWarnings(cor(t(m), method = method, use = "pairwise.complete.obs"))
    write_matrix(r, paste0(nm, "__", method, "_r"))
    write_matrix(pair_p(m, method), paste0(nm, "__", method, "_p"))
  }
  if (have_wgcna) {
    for (mpo in c(1, 0.1, 0.05)) {
      tag <- if (mpo == 1) "bicor" else sprintf("bicor_mpo%03d", round(mpo * 100))
      r <- suppressWarnings(WGCNA::bicor(t(m), use = "pairwise.complete.obs", maxPOutliers = mpo))
      write_matrix(r, paste0(nm, "__", tag, "_r"))
    }
    bp <- suppressWarnings(WGCNA::bicorAndPvalue(t(m), use = "pairwise.complete.obs"))
    write_matrix(bp$p, paste0(nm, "__bicor_p"))
  }
  if (have_hellcor && !anyNA(m) && ncol(m) >= 4) {
    fa <- names(formals(HellCor::HellCor))
    alphas <- if ("alpha" %in% fa) c(6, 3, 10) else 6
    for (a in alphas) {
      h <- matrix(NA_real_, nrow(m), nrow(m))
      for (i in seq_len(nrow(m))) for (j in seq_len(nrow(m))) {
        if (j <= i) next
        res <- tryCatch(
          if (a == 6) HellCor::HellCor(m[i, ], m[j, ]) else HellCor::HellCor(m[i, ], m[j, ], alpha = a),
          error = function(e) NULL)
        v <- if (is.null(res)) NA_real_ else if (is.list(res) && !is.null(res$Hcor)) res$Hcor else as.numeric(unlist(res))[1]
        h[i, j] <- v
        h[j, i] <- v
      }
      tag <- if (a == 6) "hellcor" else sprintf("hellcor_alpha%02d", a)
      write_matrix(h, paste0(nm, "__", tag, "_r"))
    }
  }
}

# ------------------------------------------------------------------- beta --
ab <- rbind(cbind(c(0.5, 1, 2, 3, 6, 10, 50), c(0.5, 1, 2, 3, 6, 10, 50)),
            c(2, 5), c(0.7, 9))
rows <- list()
for (k in seq_len(nrow(ab))) {
  a <- ab[k, 1]; b <- ab[k, 2]
  for (n in c(5, 77, 1000)) {
    ps <- c((1:n) / (n + 1), (seq(1.5, n - 0.5, by = 1)) / (n + 1))
    q <- qbeta(ps, a, b)
    rows[[length(rows) + 1]] <- data.frame(a = a, b = b, p = ps, q = q,
                                           cdf = pbeta(q, a, b), pdf = dbeta(q, a, b))
  }
  xs <- (1:199) / 200
  rows[[length(rows) + 1]] <- data.frame(a = a, b = b, p = NA, q = xs,
                                         cdf = pbeta(xs, a, b), pdf = dbeta(xs, a, b))
}
bt <- do.call(rbind, rows)
con <- file(file.path(out_dir, "beta_grid.tsv"), "w")
writeLines("a\tb\tp\tx\tcdf\tpdf", con)
writeLines(paste(fmt(bt$a), fmt(bt$b), fmt(bt$p), fmt(bt$q), fmt(bt$cdf), fmt(bt$pdf), sep = "\t"), con)
close(con)

# ---------------------------------------------------------------- ranks --
rk <- t(apply(datasets$ties_15x30, 1, rank, ties.method = "average"))
write_matrix(rk, "ties_15x30__ranks")

# ------------------------------------------------------------- manifest --
pv <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else NA
q <- function(s) if (is.na(s)) "null" else paste0("\"", s, "\"")
manifest <- paste0("{\n",
  "  \"generator\": \"tools/make_fixtures.R\",\n",
  "  \"seed\": ", SEED, ",\n",
  "  \"r_version\": ", q(R.version.string), ",\n",
  "  \"WGCNA\": ", q(pv("WGCNA")), ",\n",
  "  \"HellCor\": ", q(pv("HellCor")), ",\n",
  "  \"datasets\": [", paste0("\"", names(datasets), "\"", collapse = ", "), "],\n",
  "  \"skipped\": [", paste0("\"", skipped, "\"", collapse = ", "), "]\n",
  "}\n")
if (length(skipped) == 0) manifest <- sub("\\[\"\"\\]", "[]", manifest)
writeLines(manifest, file.path(out_dir, "MANIFEST.json"), sep = "")
cat("fixtures written to", out_dir, "\n")
