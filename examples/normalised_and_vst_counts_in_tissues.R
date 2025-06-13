# ---- command line arguments ----
args <- commandArgs(trailingOnly = TRUE)

# Parse arguments
keyword <- NULL
normalised <- TRUE
vst <- TRUE

for (arg in args) {
  if (grepl("^keyword=", arg)) {
    keyword <- sub("^keyword=", "", arg)
  } else if (grepl("^normalised=", arg)) {
    normalised <- as.logical(sub("^normalised=", "", arg))
  } else if (grepl("^vst=", arg)) {
    vst <- as.logical(sub("^vst=", "", arg))
  }
}

# Validate arguments
if (!normalised && !vst) {
  stop("Error: Both normalised=FALSE and vst=FALSE. At least one output type must be enabled.")
}

# ---- package management helper ----
install_if_missing <- function(pkg, bioc = FALSE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, " ...")
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}

# Install required packages
invisible(lapply(c("data.table"), install_if_missing))
install_if_missing("DESeq2", bioc = TRUE)


suppressPackageStartupMessages({
  library(DESeq2)
  library(data.table)
})

# Set folder paths
data_folder <- "data"
output_folder_norm <- "data/Normalised_counts_tissues"
output_folder_vst <- "data/Normalised_counts_tissues_vst"

# Ensure data directory exists and download GTEx gene reads file if missing
dir.create(data_folder, showWarnings = FALSE, recursive = TRUE)

gene_reads_url <- "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz"
gene_reads_dest <- file.path(data_folder, "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz")

if (!file.exists(gene_reads_dest)) {
  message("Downloading GTEx gene reads file (~900 MB).")
  download.file(gene_reads_url, destfile = gene_reads_dest, mode = "wb")
  message("Download completed: ", gene_reads_dest)
}

annotation_url <- "https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
annotation_dest <- file.path(data_folder, "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt")

if (!file.exists(annotation_dest)) {
  message("Downloading GTEx annotation file (~36 MB).")
  download.file(annotation_url, destfile = annotation_dest, mode = "wb")
  message("Download completed: ", annotation_dest)
}

# Create output directories if they don't exist
if (normalised) {
  dir.create(output_folder_norm, showWarnings = FALSE)
}
if (vst) {
  dir.create(output_folder_vst, showWarnings = FALSE)
}

# Load input data
gct_file <- file.path(data_folder, "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz")
annotation_file <- annotation_dest

# Read GCT file (ignoring the first two rows for metadata)
gct_data <- fread(gct_file, skip = 2)
gct_data <- as.data.frame(gct_data)

# Read annotation file
annotations <- fread(annotation_file)
annotations <- as.data.frame(annotations)

# Ensure all GCT samples have corresponding annotation rows
sample_ids <- colnames(gct_data)[3:ncol(gct_data)]  # Exclude Name and Description columns
annotations <- annotations[annotations$SAMPID %in% sample_ids, ]

# Verify all GCT samples have annotations
if (!all(sample_ids %in% annotations$SAMPID)) {
  stop("Some GCT file samples do not have corresponding rows in the annotation file.")
}

# Split data by tissue
tissues <- unique(annotations$SMTSD)

# Filter tissues by keyword if provided
if (!is.null(keyword)) {
  tissues <- tissues[grepl(keyword, tissues, ignore.case = TRUE)]
  if (length(tissues) == 0) {
    stop("No tissues found containing keyword: ", keyword)
  }
  message("Filtering tissues by keyword '", keyword, "': ", length(tissues), " tissues found")
}

tissue_data <- split(annotations, annotations$SMTSD)
tissue_data <- tissue_data[names(tissue_data) %in% tissues]

# Function to normalize counts and save results
normalize_counts <- function(tissue, tissue_df, gct_data) {
  tissue_samples <- tissue_df$SAMPID
  tissue_counts <- gct_data[, c("Name", "Description", tissue_samples), drop = FALSE]
  
  # Extract count matrix and colData
  count_matrix <- as.matrix(tissue_counts[, -(1:2)])  # Remove Name and Description
  rownames(count_matrix) <- tissue_counts$Name
  colData <- data.frame(row.names = tissue_samples, condition = rep("control", length(tissue_samples)))
  
  # Create DESeq2 dataset and normalize
  dds <- DESeqDataSetFromMatrix(countData = count_matrix, colData = colData, design = ~1)
  dds <- DESeq(dds)
  dds <- estimateSizeFactors(dds)
  
  # Write normalised counts if requested
  if (normalised) {
    norm_counts <- counts(dds, normalized = TRUE)
    norm_file <- file.path(output_folder_norm, paste0(tissue, "_normalised_counts.tsv.gz"))
    write.table(norm_counts, file = gzfile(norm_file), sep = "\t", col.names = TRUE, quote = FALSE)
  }
  
  # Write VST-transformed data if requested
  if (vst) {
    vsd <- vst(dds)
    vst_counts <- assay(vsd)
    vst_file <- file.path(output_folder_vst, paste0(tissue, "_vst_counts.tsv.gz"))
    write.table(vst_counts, file = gzfile(vst_file), sep = "\t", col.names = TRUE, quote = FALSE)
  }
}

# Run normalisation sequentially (because of deseq2) 
for (tissue in names(tissue_data)) {
  tissue_df <- tissue_data[[tissue]]
  message("Processing ", tissue, " (", nrow(tissue_df), " samples)...")
  normalize_counts(tissue, tissue_df, gct_data)
}
