args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript hellcor_compare.R <input.tsv[.gz]> <output_matrix.tsv> <output_debug_prefix> [i] [j] [max_rows] [workers]")
}

input_path <- args[1]
matrix_out <- args[2]
debug_prefix <- args[3]
debug_i <- if (length(args) >= 4) as.integer(args[4]) else NA
debug_j <- if (length(args) >= 5) as.integer(args[5]) else NA
max_rows <- if (length(args) >= 6) as.integer(args[6]) else NA
workers <- if (length(args) >= 7) as.integer(args[7]) else NA

lib_path <- file.path(getwd(), "r_libs")
.libPaths(c(lib_path, .libPaths()))
if (!requireNamespace("HellCor", quietly = TRUE)) {
  stop("HellCor package not available in r_libs; install it first.")
}

read_matrix <- function(path) {
  if (grepl("\\.gz$", path)) {
    con <- gzfile(path, "rt")
    on.exit(close(con), add = TRUE)
    df <- read.delim(con, check.names = FALSE)
  } else {
    df <- read.delim(path, check.names = FALSE)
  }
  rownames(df) <- df[[1]]
  as.matrix(df[, -1, drop = FALSE])
}

compute_debug <- function(x, y, kmax, lmax, alpha, prefix) {
  n <- length(x)
  u1 <- rank(x, ties.method = "average") / (n + 1)
  u2 <- rank(y, ties.method = "average") / (n + 1)
  t1 <- qbeta(u1, alpha, alpha)
  t2 <- qbeta(u2, alpha, alpha)
  weight <- sqrt(dbeta(t1, alpha, alpha) * dbeta(t2, alpha, alpha))

  R1 <- rep(Inf, n)
  R2 <- rep(Inf, n)
  II1 <- rep(NA_integer_, n)

  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      d <- sqrt((t1[i] - t1[j])^2 + (t2[i] - t2[j])^2)
      if (d < R1[i]) {
        R2[i] <- R1[i]
        R1[i] <- d
        II1[i] <- j
      } else if (d < R2[i]) {
        R2[i] <- d
      }
      if (d < R1[j]) {
        R2[j] <- R1[j]
        R1[j] <- d
        II1[j] <- i
      } else if (d < R2[j]) {
        R2[j] <- d
      }
    }
  }
  R1 <- R1 * weight
  R2 <- R2 * weight

  cte1 <- 2 * sqrt(n - 1) / n
  max_k <- max(kmax, lmax)
  polys <- orthopolynom::legendre.polynomials(max_k, normalized = TRUE)
  bfuncs <- lapply(polys, function(y) eval(parse(text = paste("function(x)", y))))
  bfuncs[[1]] <- function(x) rep(sqrt(2) / 2, length(x))

  leg1 <- lapply(0:kmax, function(k) sqrt(2) * bfuncs[[k + 1]](2 * u1 - 1))
  leg2 <- lapply(0:lmax, function(l) sqrt(2) * bfuncs[[l + 1]](2 * u2 - 1))

  betahat <- matrix(0, nrow = kmax + 1, ncol = lmax + 1)
  for (k in 0:kmax) {
    for (l in 0:lmax) {
      betahat[k + 1, l + 1] <- cte1 * sum(R1 * leg1[[k + 1]] * leg2[[l + 1]])
    }
  }

  write.table(u1, paste0(prefix, "_u1.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(u2, paste0(prefix, "_u2.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(t1, paste0(prefix, "_t1.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(t2, paste0(prefix, "_t2.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(R1, paste0(prefix, "_R1.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(R2, paste0(prefix, "_R2.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(betahat, paste0(prefix, "_betahat.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}

mat <- read_matrix(input_path)
mat <- mat[order(rownames(mat)), , drop = FALSE]
n_all <- nrow(mat)
if (!is.na(max_rows) && max_rows > 0 && max_rows < n_all) {
  mat <- mat[seq_len(max_rows), , drop = FALSE]
}
n_rows <- nrow(mat)
out <- matrix(NA_real_, nrow = n_rows, ncol = n_rows)
rownames(out) <- rownames(mat)
colnames(out) <- rownames(mat)

if (!is.na(workers) && workers > 1) {
  if (.Platform$OS.type == "windows") {
    warning("Parallel workers ignored on Windows; falling back to sequential.")
    workers <- NA
  }
}

if (!is.na(workers) && workers > 1) {
  pairs <- do.call(
    rbind,
    lapply(seq_len(n_rows - 1), function(i) cbind(i, (i + 1):n_rows))
  )
  results <- parallel::mclapply(
    seq_len(nrow(pairs)),
    function(idx) {
      i <- pairs[idx, 1]
      j <- pairs[idx, 2]
      res <- HellCor::HellCor(mat[i, ], mat[j, ], Kmax = 20L, Lmax = 20L, K = 0L, L = 0L,
                              alpha = 6.0, pval.comp = FALSE, conf.level = NULL, C.version = TRUE)
      list(i = i, j = j, val = res$Hcor)
    },
    mc.cores = workers
  )
  for (i in seq_len(n_rows)) {
    out[i, i] <- 1.0
  }
  for (entry in results) {
    out[entry$i, entry$j] <- entry$val
    out[entry$j, entry$i] <- entry$val
  }
} else {
  for (i in seq_len(n_rows)) {
    out[i, i] <- 1.0
    if (i < n_rows) {
      for (j in (i + 1):n_rows) {
        res <- HellCor::HellCor(mat[i, ], mat[j, ], Kmax = 20L, Lmax = 20L, K = 0L, L = 0L,
                                alpha = 6.0, pval.comp = FALSE, conf.level = NULL, C.version = TRUE)
        out[i, j] <- res$Hcor
        out[j, i] <- res$Hcor
      }
    }
  }
}

write.table(
  cbind(rowname = rownames(out), out),
  matrix_out,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

if (!is.na(debug_i) && !is.na(debug_j)) {
  if (!requireNamespace("orthopolynom", quietly = TRUE)) {
    stop("orthopolynom package is required for debug dumps.")
  }
  compute_debug(mat[debug_i + 1, ], mat[debug_j + 1, ], 20, 20, 6.0, debug_prefix)
}
