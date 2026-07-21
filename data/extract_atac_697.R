#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(GenomicRanges)
  library(GenomeInfoDb)
  library(IRanges)
  library(JASPAR2022)
  library(TFBSTools)
  library(motifmatchr)
  library(SummarizedExperiment)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(irlba)
})

args <- commandArgs(trailingOnly = TRUE)
BASE_DIR <- if (length(args) >= 1) {
  normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
OUT_DIR <- if (length(args) >= 2) {
  normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
} else {
  file.path(BASE_DIR, "unified_atac_features")
}
MATRIX_DIR <- file.path(OUT_DIR, "unified_peak_matrices")

FIXED_PEAK_WIDTH <- 500L
MIN_CELL_FRAC <- 0.001
MAX_CELL_FRAC <- 0.80
MAX_FEATURE_PEAKS <- 100000L
LSI_DIMS <- 50L
TFIDF_SCALE <- 10000

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MATRIX_DIR, recursive = TRUE, showWarnings = FALSE)

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", timestamp(), sprintf(...)))
  flush.console()
}

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path, call. = FALSE)
  path
}

write_mtx_gz <- function(mat, path) {
  tmp_path <- sub("\\.gz$", "", path)
  Matrix::writeMM(mat, file = tmp_path)
  in_con <- file(tmp_path, open = "rb")
  out_con <- gzfile(path, open = "wb")
  on.exit({
    close(in_con)
    close(out_con)
    unlink(tmp_path)
  }, add = TRUE)
  repeat {
    chunk <- readBin(in_con, what = "raw", n = 1024 * 1024)
    if (length(chunk) == 0) break
    writeBin(chunk, out_con)
  }
}

write_table_gz <- function(x, path, sep = "\t", col.names = TRUE, row.names = FALSE) {
  con <- gzfile(path, open = "wt")
  on.exit(close(con), add = TRUE)
  utils::write.table(
    x,
    file = con,
    sep = sep,
    quote = FALSE,
    col.names = col.names,
    row.names = row.names
  )
}

read_lines_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con), add = TRUE)
  readLines(con, warn = FALSE)
}

make_sample_table <- function(base_dir) {
  feature_files <- list.files(
    base_dir,
    pattern = "^GSM[0-9]+_ATAC_.*_features\\.txt\\.gz$",
    full.names = TRUE
  )
  if (length(feature_files) == 0) {
    stop("No ATAC feature files found in: ", base_dir, call. = FALSE)
  }
  sample <- sub("^GSM[0-9]+_ATAC_(.*)_features\\.txt\\.gz$", "\\1", basename(feature_files))
  tab <- data.table(
    sample = sample,
    feature_file = feature_files,
    matrix_file = sub("_features\\.txt\\.gz$", "_matrix.mtx.gz", feature_files),
    barcode_file = sub("_features\\.txt\\.gz$", "_barcodes.txt.gz", feature_files)
  )
  tab <- tab[order(sample)]
  invisible(lapply(tab$matrix_file, stop_if_missing))
  invisible(lapply(tab$barcode_file, stop_if_missing))
  tab
}

parse_peak_ids <- function(peak_ids) {
  parts <- tstrsplit(peak_ids, "-", fixed = TRUE)
  if (length(parts) < 3) {
    stop("Peak IDs must look like chr-start-end.", call. = FALSE)
  }
  peak_dt <- data.table(
    peak_id = peak_ids,
    chr = parts[[1]],
    start = suppressWarnings(as.integer(parts[[2]])),
    end = suppressWarnings(as.integer(parts[[3]]))
  )
  peak_dt <- peak_dt[!is.na(start) & !is.na(end) & start > 0 & end >= start]
  peak_dt
}

fixed_peak_granges <- function(feature_file, fixed_width = FIXED_PEAK_WIDTH) {
  peak_ids <- read_lines_gz(feature_file)
  peak_dt <- parse_peak_ids(peak_ids)
  gr <- makeGRangesFromDataFrame(
    peak_dt,
    seqnames.field = "chr",
    start.field = "start",
    end.field = "end",
    keep.extra.columns = TRUE
  )
  gr <- keepStandardChromosomes(gr, pruning.mode = "coarse")
  seqlevelsStyle(gr) <- "UCSC"
  GenomeInfoDb::seqinfo(gr) <- GenomeInfoDb::seqinfo(BSgenome.Hsapiens.UCSC.hg38)[seqlevels(gr)]
  resize(gr, width = fixed_width, fix = "center")
}

make_peak_ids <- function(gr) {
  paste0(as.character(seqnames(gr)), "-", start(gr), "-", end(gr))
}

project_to_consensus <- function(mat, sample_gr, consensus_gr, sample_name) {
  hits <- findOverlaps(sample_gr, consensus_gr, ignore.strand = TRUE)
  if (length(hits) == 0) {
    stop("No overlaps between sample peaks and consensus peaks for sample: ", sample_name, call. = FALSE)
  }
  map <- sparseMatrix(
    i = subjectHits(hits),
    j = queryHits(hits),
    x = 1,
    dims = c(length(consensus_gr), length(sample_gr))
  )
  projected <- map %*% mat
  projected@x[projected@x > 0] <- 1
  projected
}

compute_tfidf <- function(mat, scale_factor = TFIDF_SCALE) {
  mat <- as(mat, "dgCMatrix")
  n_cells <- ncol(mat)
  cell_depth <- Matrix::colSums(mat)
  cell_depth[cell_depth <= 0] <- 1
  tf <- t(t(mat) / cell_depth)
  df <- Matrix::rowSums(mat > 0)
  idf <- log(1 + n_cells / (1 + df))
  tfidf <- Diagonal(x = as.numeric(idf)) %*% tf
  tfidf@x <- log1p(tfidf@x * scale_factor)
  list(tfidf = as(tfidf, "dgCMatrix"), idf = idf, depth = cell_depth)
}

save_dense_matrix_gz <- function(mat, path, row_id_name = "cell_id") {
  dt <- as.data.table(as.data.frame(mat, check.names = FALSE))
  dt[, (row_id_name) := rownames(mat)]
  setcolorder(dt, c(row_id_name, setdiff(names(dt), row_id_name)))
  write_table_gz(dt, path, sep = ",", col.names = TRUE, row.names = FALSE)
}

sample_table <- make_sample_table(BASE_DIR)
log_msg("Found %d ATAC samples: %s", nrow(sample_table), paste(sample_table$sample, collapse = ", "))

log_msg("Reading ATAC peak coordinates and building fixed-width consensus peaks.")
sample_gr_list <- setNames(vector("list", nrow(sample_table)), sample_table$sample)
for (i in seq_len(nrow(sample_table))) {
  sample <- sample_table$sample[i]
  gr <- fixed_peak_granges(sample_table$feature_file[i])
  sample_gr_list[[sample]] <- gr
  log_msg("%s: %d fixed peaks", sample, length(gr))
}

all_fixed_peaks <- do.call(c, unname(sample_gr_list))
consensus_gr <- reduce(all_fixed_peaks, ignore.strand = TRUE, min.gapwidth = 1L)
consensus_gr <- keepStandardChromosomes(consensus_gr, pruning.mode = "coarse")
consensus_gr <- consensus_gr[width(consensus_gr) >= 100 & width(consensus_gr) <= 5000]
consensus_gr <- sortSeqlevels(sort(consensus_gr))
consensus_ids <- make_peak_ids(consensus_gr)
log_msg("Consensus peak regions before cell-frequency filtering: %d", length(consensus_gr))

log_msg("First pass: project each sample to consensus peaks and estimate accessibility frequency.")
total_cells <- 0L
consensus_cell_count <- numeric(length(consensus_gr))
consensus_access_sum <- numeric(length(consensus_gr))
sample_cell_counts <- integer(nrow(sample_table))

for (i in seq_len(nrow(sample_table))) {
  sample <- sample_table$sample[i]
  mat <- readMM(sample_table$matrix_file[i])
  mat <- as(mat, "dgCMatrix")
  barcodes <- read_lines_gz(sample_table$barcode_file[i])
  if (ncol(mat) != length(barcodes)) {
    stop("Barcode count does not match matrix columns for sample: ", sample, call. = FALSE)
  }
  proj <- project_to_consensus(mat, sample_gr_list[[sample]], consensus_gr, sample)
  bin_proj <- proj
  bin_proj@x[bin_proj@x > 0] <- 1
  consensus_cell_count <- consensus_cell_count + as.numeric(Matrix::rowSums(bin_proj > 0))
  consensus_access_sum <- consensus_access_sum + as.numeric(Matrix::rowSums(proj))
  sample_cell_counts[i] <- ncol(proj)
  total_cells <- total_cells + ncol(proj)
  rm(mat, proj, bin_proj)
  gc(verbose = FALSE)
  log_msg("%s: projected %d cells", sample, sample_cell_counts[i])
}

cell_frac <- consensus_cell_count / total_cells
keep <- which(cell_frac >= MIN_CELL_FRAC & cell_frac <= MAX_CELL_FRAC)
if (length(keep) > MAX_FEATURE_PEAKS) {
  ord <- order(consensus_cell_count[keep], consensus_access_sum[keep], decreasing = TRUE)
  keep <- keep[ord[seq_len(MAX_FEATURE_PEAKS)]]
  keep <- sort(keep)
}
filtered_gr <- consensus_gr[keep]
filtered_ids <- consensus_ids[keep]
filtered_stats <- data.table(
  peak_id = filtered_ids,
  chr = as.character(seqnames(filtered_gr)),
  start = start(filtered_gr),
  end = end(filtered_gr),
  width = width(filtered_gr),
  cell_count = consensus_cell_count[keep],
  cell_fraction = cell_frac[keep],
  accessibility_sum = consensus_access_sum[keep]
)
write_table_gz(filtered_stats, file.path(OUT_DIR, "consensus_peaks_filtered.tsv.gz"))
log_msg("Filtered consensus peaks retained: %d", length(filtered_gr))

log_msg("Second pass: write unified peak matrices and build combined matrix.")
projected_list <- setNames(vector("list", nrow(sample_table)), sample_table$sample)
cell_ids <- character(0)

for (i in seq_len(nrow(sample_table))) {
  sample <- sample_table$sample[i]
  mat <- readMM(sample_table$matrix_file[i])
  mat <- as(mat, "dgCMatrix")
  barcodes <- read_lines_gz(sample_table$barcode_file[i])
  proj <- project_to_consensus(mat, sample_gr_list[[sample]], filtered_gr, sample)
  proj <- as(proj, "dgCMatrix")
  sample_cell_ids <- paste(sample, barcodes, sep = "_")
  colnames(proj) <- sample_cell_ids
  rownames(proj) <- filtered_ids
  projected_list[[sample]] <- proj
  cell_ids <- c(cell_ids, sample_cell_ids)

  write_mtx_gz(proj, file.path(MATRIX_DIR, paste0(sample, "_matrix.mtx.gz")))
  write_table_gz(data.table(cell_id = sample_cell_ids, barcode = barcodes), file.path(MATRIX_DIR, paste0(sample, "_barcodes.tsv.gz")))
  log_msg("%s: saved unified matrix with %d peaks x %d cells", sample, nrow(proj), ncol(proj))
  rm(mat)
  gc(verbose = FALSE)
}

combined_mat <- do.call(cbind, projected_list)
rownames(combined_mat) <- filtered_ids
colnames(combined_mat) <- cell_ids
write_table_gz(data.table(cell_id = cell_ids), file.path(OUT_DIR, "all_unified_cell_ids.tsv.gz"))
log_msg("Combined unified ATAC matrix: %d peaks x %d cells", nrow(combined_mat), ncol(combined_mat))

log_msg("Computing TF-IDF and %d-dimensional LSI features.", LSI_DIMS)
tfidf_res <- compute_tfidf(combined_mat)
tfidf <- tfidf_res$tfidf
lsi_n <- min(LSI_DIMS, nrow(tfidf) - 1L, ncol(tfidf) - 1L)
if (lsi_n != LSI_DIMS) stop("Insufficient cells or peaks to compute 50 LSI components.", call. = FALSE)
svd <- irlba::irlba(t(tfidf), nv = lsi_n, nu = lsi_n)
lsi <- svd$u %*% diag(svd$d, nrow = lsi_n)
rownames(lsi) <- cell_ids
colnames(lsi) <- paste0("LSI_", seq_len(ncol(lsi)))
save_dense_matrix_gz(lsi, file.path(OUT_DIR, "atac_lsi_50.csv.gz"))
saveRDS(
  list(
    peak_id = filtered_ids,
    idf = tfidf_res$idf,
    depth = tfidf_res$depth,
    svd_v = svd$v,
    svd_d = svd$d,
    scale = TFIDF_SCALE
  ),
  file = file.path(OUT_DIR, "lsi_model.rds")
)
log_msg("Saved LSI features.")

log_msg("Scanning filtered consensus peaks for JASPAR2022 TF motifs.")
opts <- list(species = 9606, collection = "CORE", matrixtype = "PWM")
pwm_list <- getMatrixSet(JASPAR2022, opts)
motif_names <- sapply(pwm_list, function(x) name(x))
motif_names <- ifelse(motif_names == "" | is.na(motif_names), names(pwm_list), motif_names)
keep_motif <- !duplicated(motif_names)
pwm_list <- pwm_list[keep_motif]
motif_names <- motif_names[keep_motif]
if (length(motif_names) != 647L) {
  stop("Expected 647 unique JASPAR2022 motif names, found ", length(motif_names), call. = FALSE)
}

motif_matches <- matchMotifs(
  pwms = pwm_list,
  subject = filtered_gr,
  genome = BSgenome.Hsapiens.UCSC.hg38,
  out = "matches"
)
motif_mat <- SummarizedExperiment::assay(motif_matches)
motif_mat <- as(motif_mat, "dgCMatrix")
motif_mat@x[] <- 1
rownames(motif_mat) <- filtered_ids
colnames(motif_mat) <- motif_names
write_mtx_gz(motif_mat, file.path(OUT_DIR, "motif_peak_by_tf.mtx.gz"))
write_table_gz(data.table(tf = motif_names), file.path(OUT_DIR, "motif_names.tsv.gz"))
log_msg("Motif peak-by-TF matrix: %d peaks x %d TFs", nrow(motif_mat), ncol(motif_mat))

log_msg("Computing global TF motif activity from TF-IDF peak matrix.")
tf_activity <- as.matrix(t(tfidf) %*% motif_mat)
tf_activity <- scale(tf_activity)
tf_activity[is.na(tf_activity)] <- 0
rownames(tf_activity) <- cell_ids
colnames(tf_activity) <- motif_names
save_dense_matrix_gz(tf_activity, file.path(OUT_DIR, "global_tf_tfidf_motif_activity_zscore.csv.gz"))
log_msg("Saved TF motif activity features.")

atac_697 <- cbind(tf_activity, lsi)
colnames(atac_697) <- c(paste0("TF_", motif_names), colnames(lsi))
save_dense_matrix_gz(atac_697, file.path(OUT_DIR, "ATAC_697.csv.gz"))
log_msg("Saved combined ATAC matrix: %d cells x %d features", nrow(atac_697), ncol(atac_697))

summary_dt <- data.table(
  item = c(
    "base_dir",
    "output_dir",
    "samples",
    "total_cells",
    "fixed_peak_width",
    "raw_fixed_peaks",
    "consensus_peaks_before_filter",
    "consensus_peaks_after_filter",
    "min_cell_frac",
    "max_cell_frac",
    "max_feature_peaks",
    "lsi_dims",
    "tf_motifs"
  ),
  value = c(
    BASE_DIR,
    OUT_DIR,
    paste(sample_table$sample, collapse = ","),
    as.character(total_cells),
    as.character(FIXED_PEAK_WIDTH),
    as.character(length(all_fixed_peaks)),
    as.character(length(consensus_gr)),
    as.character(length(filtered_gr)),
    as.character(MIN_CELL_FRAC),
    as.character(MAX_CELL_FRAC),
    as.character(MAX_FEATURE_PEAKS),
    as.character(ncol(lsi)),
    as.character(length(motif_names))
  )
)
write_table_gz(summary_dt, file.path(OUT_DIR, "summary.tsv.gz"))
saveRDS(
  list(
    sample_table = sample_table,
    consensus_peaks = filtered_gr,
    sample_cell_counts = setNames(sample_cell_counts, sample_table$sample),
    parameters = list(
      fixed_peak_width = FIXED_PEAK_WIDTH,
      min_cell_frac = MIN_CELL_FRAC,
      max_cell_frac = MAX_CELL_FRAC,
      max_feature_peaks = MAX_FEATURE_PEAKS,
      lsi_dims = LSI_DIMS,
      tfidf_scale = TFIDF_SCALE
    )
  ),
  file = file.path(OUT_DIR, "feature_build_metadata.rds")
)

log_msg("Done. Outputs are in: %s", OUT_DIR)
