#!/usr/bin/env Rscript
# pvalue_parity_check.R
# Compact Rust-vs-R p-value parity checks:
# - Pearson / Spearman / Kendall / Bicor: numeric parity
# - Hellcor: Monte Carlo null distribution behavior

lib_path <- file.path(getwd(), "r_libs")
if (!dir.exists(lib_path)) dir.create(lib_path, recursive = TRUE)
.libPaths(c(lib_path, .libPaths()))

install_if_missing <- function(pkg, bioc = FALSE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, " ...")
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = lib_path)
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE, lib = lib_path)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org", lib = lib_path)
    }
  }
}

invisible(lapply(c("data.table", "matrixStats"), install_if_missing))
install_if_missing("WGCNA", bioc = TRUE)
install_if_missing("HellCor")

suppressPackageStartupMessages({
  library(data.table)
  library(matrixStats)
  library(WGCNA)
  library(HellCor)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript pvalue_parity_check.R <data_file> [subset_n=40] [hellcor_mc_samples=1000] [seed=42]")
}
data_file <- args[1]
subset_n <- if (length(args) >= 2) as.integer(args[2]) else 40L
hellcor_mc_samples <- if (length(args) >= 3) as.integer(args[3]) else 1000L
seed <- if (length(args) >= 4) as.integer(args[4]) else 42L

if (subset_n < 3) stop("subset_n must be >= 3")
if (hellcor_mc_samples < 10) stop("hellcor_mc_samples must be >= 10")

cat("Loading:", data_file, "\n")
dt <- fread(data_file, data.table = FALSE)
row_ids <- dt[[1]]
mat <- as.matrix(dt[, -1, drop = FALSE])
rownames(mat) <- row_ids

ord <- order(rownames(mat))
mat <- mat[ord, , drop = FALSE]
row_ids <- rownames(mat)

if (subset_n > nrow(mat)) stop("subset_n exceeds number of rows")
subset_ids <- row_ids[seq_len(subset_n)]
mat_sub <- mat[subset_ids, , drop = FALSE]
n_samples <- ncol(mat_sub)

cat(sprintf("Subset: %d rows x %d samples\n", nrow(mat_sub), ncol(mat_sub)))

input_basename <- function(path) {
  b <- tools::file_path_sans_ext(basename(path))
  if (grepl("\\.tar$", b)) b <- sub("\\.tar$", "", b)
  b
}

write_subset_file <- function(ids, file) {
  writeLines(ids, con = file, useBytes = TRUE)
}

run_rust_method <- function(input_path, method, subset_file, mc_samples = NULL, seed = NULL) {
  cmd <- c(
    "run", "-q", "-p", "mincorr-cli", "--",
    input_path, method,
    "--subset-file", subset_file,
    "--with-pvalues"
  )
  if (method == "hellcor") {
    cmd <- c(cmd, "--pvalue-mode", "mc", "--mc-samples", as.character(mc_samples), "--mc-seed", as.character(seed))
  }
  out <- system2("cargo", cmd, stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status")
  if (!is.null(status) && status != 0) {
    stop(sprintf("Rust run failed for %s:\n%s", method, paste(out, collapse = "\n")))
  }
}

load_rust_matrix_from_tar <- function(tar_path, suffix_pattern) {
  if (!file.exists(tar_path)) stop("Missing tar output: ", tar_path)
  files <- untar(tar_path, list = TRUE)
  target <- files[grepl(suffix_pattern, files)]
  if (length(target) == 0) stop("No file matching ", suffix_pattern, " in ", tar_path)
  target <- target[1]
  exdir <- tempfile("rust_mat_")
  dir.create(exdir, recursive = TRUE)
  untar(tar_path, files = target, exdir = exdir)
  dt <- fread(
    file.path(exdir, target),
    data.table = FALSE,
    sep = "\t",
    na.strings = c("NaN", "nan"),
    check.names = FALSE
  )
  m <- as.matrix(dt[, -1, drop = FALSE])
  rownames(m) <- dt[[1]]
  if (nrow(m) == ncol(m) && length(dt[[1]]) == ncol(m)) {
    colnames(m) <- dt[[1]]
  } else {
    colnames(m) <- colnames(dt)[-1]
  }
  m
}

pairwise_n <- function(x) {
  valid <- !is.na(x)
  n <- valid %*% t(valid)
  storage.mode(n) <- "double"
  n
}

p_from_corr_t <- function(r, n_pair) {
  p <- matrix(NA_real_, nrow = nrow(r), ncol = ncol(r))
  dimnames(p) <- dimnames(r)
  ok <- is.finite(r) & is.finite(n_pair) & n_pair > 2
  if (any(ok)) {
    rr <- pmax(pmin(r[ok], 1), -1)
    denom <- pmax(1 - rr^2, .Machine$double.eps)
    tstat <- abs(rr) * sqrt((n_pair[ok] - 2) / denom)
    p[ok] <- 2 * pt(-tstat, df = n_pair[ok] - 2)
  }
  diag(p) <- 0
  p
}

r_pearson_p <- function(x) {
  r <- cor(t(x), method = "pearson", use = "pairwise.complete.obs")
  dimnames(r) <- list(rownames(x), rownames(x))
  p_from_corr_t(r, pairwise_n(x))
}

r_spearman_p <- function(x) {
  xr <- matrixStats::rowRanks(x, ties.method = "average")
  r <- cor(t(xr), method = "pearson", use = "pairwise.complete.obs")
  dimnames(r) <- list(rownames(x), rownames(x))
  p_from_corr_t(r, pairwise_n(xr))
}

r_kendall_p <- function(x) {
  n <- nrow(x)
  p <- matrix(NA_real_, n, n)
  rownames(p) <- rownames(x)
  colnames(p) <- rownames(x)
  diag(p) <- 0
  for (i in seq_len(n)) {
    if (i < n) {
      for (j in (i + 1):n) {
        # asymptotic p-value (matches Rust mode intent)
        ct <- suppressWarnings(cor.test(x[i, ], x[j, ], method = "kendall", exact = FALSE))
        p[i, j] <- ct$p.value
        p[j, i] <- ct$p.value
      }
    }
  }
  p
}

r_bicor_p <- function(x) {
  bp <- WGCNA::bicorAndPvalue(t(x), use = "pairwise.complete.obs")
  p <- NULL
  if (!is.null(bp$p)) p <- bp$p
  if (is.null(p) && !is.null(bp$pvalue)) p <- bp$pvalue
  if (is.null(p) && !is.null(bp$pValues)) p <- bp$pValues
  if (is.null(p)) stop("Could not find p-value matrix in WGCNA::bicorAndPvalue output")
  p
}

compare_upper <- function(method, rust_p, r_p) {
  rust_p <- rust_p[rownames(r_p), colnames(r_p), drop = FALSE]
  idx <- upper.tri(r_p)
  rv <- r_p[idx]
  uv <- rust_p[idx]
  ok <- is.finite(rv) & is.finite(uv)
  if (!any(ok)) {
    cat(method, ": no finite pairs to compare\n")
    return(invisible(NULL))
  }
  d <- uv[ok] - rv[ok]
  cat(sprintf(
    "%-8s p-value parity: n=%d, MAE=%.3e, RMSE=%.3e, max_abs=%.3e, cor=%.6f\n",
    method, length(d), mean(abs(d)), sqrt(mean(d^2)), max(abs(d)), cor(rv[ok], uv[ok])
  ))
}

subset_file <- tempfile("subset_ids_", fileext = ".txt")
write_subset_file(subset_ids, subset_file)
base <- input_basename(data_file)

methods <- c("pearson", "spearman", "kendall", "bicor", "hellcor")
for (m in methods) {
  run_rust_method(data_file, m, subset_file, mc_samples = hellcor_mc_samples, seed = seed)
}

rust_tar <- function(method) {
  sprintf("%s_subsetfile%d_%s_correlations.tar.gz", base, subset_n, method)
}

rust_p_pearson <- load_rust_matrix_from_tar(rust_tar("pearson"), "_pvalues\\.tsv$")
rust_p_spearman <- load_rust_matrix_from_tar(rust_tar("spearman"), "_pvalues\\.tsv$")
rust_p_kendall <- load_rust_matrix_from_tar(rust_tar("kendall"), "_pvalues\\.tsv$")
rust_p_bicor <- load_rust_matrix_from_tar(rust_tar("bicor"), "_pvalues\\.tsv$")
rust_p_hellcor <- load_rust_matrix_from_tar(rust_tar("hellcor"), "_pvalues\\.tsv$")

r_p_pearson <- r_pearson_p(mat_sub)
r_p_spearman <- r_spearman_p(mat_sub)
r_p_kendall <- r_kendall_p(mat_sub)
r_p_bicor <- r_bicor_p(mat_sub)

compare_upper("pearson", rust_p_pearson, r_p_pearson)
compare_upper("spearman", rust_p_spearman, r_p_spearman)
compare_upper("kendall", rust_p_kendall, r_p_kendall)
compare_upper("bicor", rust_p_bicor, r_p_bicor)

# Hellcor MC distribution behavior:
# 1) Rust p-values on independent uniforms should be close to Uniform(0,1).
# 2) R HellCor p-values built from a shared null MC distribution should behave similarly.
set.seed(seed + 1L)
n_indep <- max(24L, min(40L, subset_n))
indep <- matrix(runif(n_indep * n_samples), nrow = n_indep, ncol = n_samples)
rownames(indep) <- sprintf("U%03d", seq_len(n_indep))

indep_tsv <- tempfile("hellcor_indep_", fileext = ".tsv")
fwrite(
  data.frame(row_id = rownames(indep), indep, check.names = FALSE),
  indep_tsv,
  sep = "\t",
  quote = FALSE
)
indep_subset_file <- tempfile("indep_subset_ids_", fileext = ".txt")
write_subset_file(rownames(indep), indep_subset_file)

run_rust_method(indep_tsv, "hellcor", indep_subset_file, mc_samples = hellcor_mc_samples, seed = seed)
indep_base <- input_basename(indep_tsv)
rust_indep_tar <- sprintf("%s_subsetfile%d_hellcor_correlations.tar.gz", indep_base, n_indep)
rust_indep_p <- load_rust_matrix_from_tar(rust_indep_tar, "_pvalues\\.tsv$")
rust_indep_vals <- rust_indep_p[upper.tri(rust_indep_p)]
rust_indep_vals <- rust_indep_vals[is.finite(rust_indep_vals)]

ks_rust <- suppressWarnings(ks.test(jitter(rust_indep_vals, amount = 1e-12), "punif"))

set.seed(seed + 2L)
r_null <- replicate(
  hellcor_mc_samples,
  HellCor::HellCor(
    runif(n_samples), runif(n_samples),
    Kmax = 20L, Lmax = 20L, K = 0L, L = 0L,
    alpha = 6.0, pval.comp = FALSE, conf.level = NULL, C.version = TRUE
  )$Hcor
)
r_null <- sort(r_null[is.finite(r_null)])

pair_idx <- utils::combn(seq_len(n_indep), 2)
r_hell_p <- numeric(ncol(pair_idx))
for (k in seq_len(ncol(pair_idx))) {
  i <- pair_idx[1, k]
  j <- pair_idx[2, k]
  obs <- HellCor::HellCor(
    indep[i, ], indep[j, ],
    Kmax = 20L, Lmax = 20L, K = 0L, L = 0L,
    alpha = 6.0, pval.comp = FALSE, conf.level = NULL, C.version = TRUE
  )$Hcor
  ge <- sum(r_null >= obs)
  r_hell_p[k] <- (ge + 1) / (length(r_null) + 1)
}
r_hell_p <- r_hell_p[is.finite(r_hell_p)]
ks_r <- suppressWarnings(ks.test(jitter(r_hell_p, amount = 1e-12), "punif"))

min_len <- min(length(rust_indep_vals), length(r_hell_p))
set.seed(seed + 3L)
rust_cmp <- sample(rust_indep_vals, min_len)
r_cmp <- sample(r_hell_p, min_len)
ks_between <- suppressWarnings(ks.test(jitter(rust_cmp, amount = 1e-12), jitter(r_cmp, amount = 1e-12)))

cat("\nHellcor MC distribution behavior (independence):\n")
cat(sprintf("Rust  KS vs U(0,1): D=%.4f, p=%.4g, n=%d\n", ks_rust$statistic, ks_rust$p.value, length(rust_indep_vals)))
cat(sprintf("R     KS vs U(0,1): D=%.4f, p=%.4g, n=%d\n", ks_r$statistic, ks_r$p.value, length(r_hell_p)))
cat(sprintf("Rust vs R two-sample KS: D=%.4f, p=%.4g, n=%d\n", ks_between$statistic, ks_between$p.value, min_len))

cat("\nDone.\n")
