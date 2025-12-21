#!/usr/bin/env Rscript
# investigate_diffs.R - Compare Rust vs R correlation matrices

lib_path <- file.path(getwd(), "r_libs")
if (!dir.exists(lib_path)) {
  dir.create(lib_path, recursive = TRUE)
}
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


invisible(lapply(c("data.table", "DescTools", "stringr", "pcaPP", "matrixStats"), install_if_missing))
install_if_missing("WGCNA", bioc = TRUE)
install_if_missing("HellCor")

suppressPackageStartupMessages({
  library(data.table)
  library(DescTools)  # For Lin's CCC
  library(stringr)
  library(WGCNA)
  library(HellCor)
  library(pcaPP)  # for cor.fk fast Kendall
  library(matrixStats)
})

# Require expression file path as first CLI argument
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript investigate_diffs.R <data_file> [threads] [methods]")
}
data_file <- args[1]
threads <- if (length(args) >= 2) as.integer(args[2]) else NA
methods <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else NULL
if (!is.null(methods)) {
  methods <- tolower(trimws(methods))
}

if (!is.na(threads) && threads > 0) {
  allowWGCNAThreads(nThreads = threads)
} else {
  allowWGCNAThreads()
}

# Extract input basename for constructing Rust output filenames
input_basename <- tools::file_path_sans_ext(basename(data_file))
# Remove .tar if present (for .tar.gz files)
if (grepl("\\.tar$", input_basename)) {
  input_basename <- sub("\\.tar$", "", input_basename)
}

cat("Loading data from", data_file, "...\n")
expr_dt <- fread(data_file, data.table = FALSE)
row_ids <- expr_dt[[1]]
mat <- as.matrix(expr_dt[, -1])
rownames(mat) <- row_ids

# Sort like Rust does
ord <- order(row_ids)
mat <- mat[ord, ]
row_ids <- row_ids[ord]

cat("Matrix dimensions:", nrow(mat), "rows x", ncol(mat), "columns\n")

timing_results <- list()

# Pearson correlation with WGCNA
if (is.null(methods) || "pearson" %in% methods) {
  cat("- Computing Pearson correlations...")
  start_time <- Sys.time()
  r_pearson <- cor(t(mat), method = "pearson", use = "pairwise.complete.obs")
  end_time <- Sys.time()
  timing_results$pearson <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(" completed in", round(timing_results$pearson, 3), "seconds\n")
}

# Spearman correlation via fast Pearson on ranked data
if (is.null(methods) || "spearman" %in% methods) {
  cat("- Computing Spearman correlations...")
  start_time <- Sys.time()
  # Rank each row across columns efficiently
  rank_mat <- matrixStats::rowRanks(mat, ties.method = "average")
  r_spearman <- cor(t(rank_mat), method = "pearson", use = "pairwise.complete.obs")
  end_time <- Sys.time()
  timing_results$spearman <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(" completed in", round(timing_results$spearman, 3), "seconds\n")
}


if (is.null(methods) || "kendall" %in% methods) {
  cat("- Computing Kendall correlations...")
  start_time <- Sys.time()
  # Use fast Kendall tau from pcaPP (O(n log n))
  r_kendall <- cor.fk(t(mat))
  end_time <- Sys.time()
  timing_results$kendall <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(" completed in", round(timing_results$kendall, 3), "seconds\n")
}

if (is.null(methods) || "bicor" %in% methods) {
  cat("- Computing bicor correlations...")
  start_time <- Sys.time()
  r_bicor <- bicor(t(mat), use = "pairwise.complete.obs")
  end_time <- Sys.time()
  timing_results$bicor <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(" completed in", round(timing_results$bicor, 3), "seconds\n")
}

if (is.null(methods) || "hellcor" %in% methods) {
  cat("- Computing hellcor correlations...")
  start_time <- Sys.time()
  n_rows <- nrow(mat)
  r_hellcor <- matrix(NA_real_, nrow = n_rows, ncol = n_rows)
  rownames(r_hellcor) <- rownames(mat)
  colnames(r_hellcor) <- rownames(mat)

  if (!is.na(threads) && threads > 1 && .Platform$OS.type != "windows") {
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
      mc.cores = threads
    )
    for (i in seq_len(n_rows)) {
      r_hellcor[i, i] <- 1.0
    }
    for (entry in results) {
      r_hellcor[entry$i, entry$j] <- entry$val
      r_hellcor[entry$j, entry$i] <- entry$val
    }
  } else {
    for (i in seq_len(n_rows)) {
      r_hellcor[i, i] <- 1.0
      if (i < n_rows) {
        for (j in (i + 1):n_rows) {
          res <- HellCor::HellCor(mat[i, ], mat[j, ], Kmax = 20L, Lmax = 20L, K = 0L, L = 0L,
                                  alpha = 6.0, pval.comp = FALSE, conf.level = NULL, C.version = TRUE)
          r_hellcor[i, j] <- res$Hcor
          r_hellcor[j, i] <- res$Hcor
        }
      }
    }
  }
  end_time <- Sys.time()
  timing_results$hellcor <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(" completed in", round(timing_results$hellcor, 3), "seconds\n")
}

r_cors <- list()
if (exists("r_pearson")) r_cors$pearson <- r_pearson
if (exists("r_spearman")) r_cors$spearman <- r_spearman
if (exists("r_kendall")) r_cors$kendall <- r_kendall
if (exists("r_bicor")) r_cors$bicor <- r_bicor
if (exists("r_hellcor")) r_cors$hellcor <- r_hellcor

cat("\nTimings\n")
for (method in names(timing_results)) {
  cat(sprintf("%-10s: %8.3f seconds\n", stringr::str_to_title(method), timing_results[[method]]))
}

# Load & compare each Rust output
for (type in names(r_cors)) {
  tryCatch({
    cat("\n=== Comparing", type, "correlations ===\n")
    
    rust_tgz <- sprintf("%s_%s_correlations.tar.gz", input_basename, type)
    if (!file.exists(rust_tgz)) {
      cat("Skipping, Rust output missing:", rust_tgz, "\n")
      next
    }

    # Load Rust matrix
    tmpd <- tempdir()
    rust_csv_path <- untar(rust_tgz, list = TRUE)[1]
    untar(rust_tgz, files = rust_csv_path, exdir = tmpd)
    rust_dt <- fread(file.path(tmpd, rust_csv_path), sep = "\t", na.strings = "NaN", data.table = FALSE)
    
    rust_mat <- as.matrix(rust_dt[,-1])
    rownames(rust_mat) <- rust_dt[[1]]
    colnames(rust_mat) <- rust_dt[[1]]
    rust_mat <- rust_mat[row_ids, row_ids] # Ensure same order as R matrix
    
    r_mat <- r_cors[[type]]
    r_vals <- r_mat[upper.tri(r_mat)]
    rust_vals <- rust_mat[upper.tri(rust_mat)]
    
    # NaN pattern check
    na_pattern_equal <- identical(is.na(r_vals), is.na(rust_vals))
    mismatches <- sum(is.na(r_vals) != is.na(rust_vals))
    cat(sprintf("NaN pattern identical between Rust and R: %s (mismatches: %d)\n",
                if (na_pattern_equal) "YES" else "NO", mismatches))

    # Remove NaNs for paired comparison of numeric values
    valid_idx <- !is.na(r_vals) & !is.na(rust_vals)
    r_vals_valid <- r_vals[valid_idx]
    rust_vals_valid <- rust_vals[valid_idx]

    if (length(r_vals_valid) == 0) {
        cat("No valid numeric pairs to compare after removing NAs.\n")
        next
    }
    
    diff_vals <- rust_vals_valid - r_vals_valid
    avg_vals <- (rust_vals_valid + r_vals_valid) / 2
    
    mae <- mean(abs(diff_vals))
    rmse <- sqrt(mean(diff_vals^2))
    cor_of_cors <- cor(r_vals_valid, rust_vals_valid, method = "pearson")
    ccc <- CCC(r_vals_valid, rust_vals_valid)$rho.c$est
    
    cat("\nRust vs R numerical differences\n")
    cat(sprintf("Mean Absolute Error (MAE): %.4g\n", mae))
    cat(sprintf("Root Mean Squared Error (RMSE): %.4g\n", rmse))
    cat(sprintf("Correlation of correlations: %.6f\n", cor_of_cors))
    cat(sprintf("Lin's Concordance Correlation Coefficient: %.6f\n", ccc))

  }, error = function(e) {
    cat("\nError processing", type, "correlations:", conditionMessage(e), "\n")
  })
} 
