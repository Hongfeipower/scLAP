#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(Matrix)
})

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", timestamp(), sprintf(...)))
  flush.console()
}

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_dir <- file.path(root, "scATAC", "archr_paired4008_downstream")
table_dir <- file.path(out_dir, "tables")
pb_dir <- file.path(out_dir, "pseudobulk_edgeR")

dir.create(pb_dir, recursive = TRUE, showWarnings = FALSE)

matrix_file <- file.path(table_dir, "PeakMatrix_pseudobulk_celltype_by_library.mtx")
columns_file <- file.path(table_dir, "PeakMatrix_pseudobulk_celltype_by_library_columns.csv")
qc_file <- file.path(table_dir, "paired4008_archr_qc.csv")
peak_file <- file.path(table_dir, "PeakSet_consensus_metadata.csv")
marker_file <- file.path(table_dir, "marker_peaks_by_RNA_cell_type_markers_annotated.csv")

stopifnot(
  file.exists(matrix_file), file.exists(columns_file), file.exists(qc_file),
  file.exists(peak_file), file.exists(marker_file)
)

min_target_cells <- 10L
min_other_cells <- 20L
min_libraries <- 3L
min_peak_count <- 10L
min_peak_samples <- 3L
fdr_cutoff <- 0.05
logfc_cutoff <- 1

log_msg("Reading pseudobulk PeakMatrix.")
peak_counts <- readMM(matrix_file)
column_meta <- fread(columns_file)
qc <- fread(qc_file)
peak_meta <- fread(peak_file)
archr_markers <- fread(marker_file)

stopifnot(ncol(peak_counts) == nrow(column_meta))
stopifnot(nrow(peak_counts) == nrow(peak_meta))
setnames(peak_meta, "idx", "archr_idx")
peak_meta[, peak_row := .I]

column_meta[, c("RNA_cell_type", "FragmentLibrary") := tstrsplit(
  pseudobulk_group, "__", fixed = TRUE
)]

qc <- qc[analysis_qc_pass == TRUE & RNA_cell_type != "unclassified"]
cell_counts <- qc[, .(cells = .N), by = .(RNA_cell_type, FragmentLibrary)]
fwrite(cell_counts, file.path(pb_dir, "pseudobulk_cell_counts_by_celltype_library.csv"))

analysis_types <- sort(unique(qc$RNA_cell_type))
summary_rows <- list()
design_rows <- list()

for (target in analysis_types) {
  log_msg("Testing %s vs other labeled cells.", target)
  target_counts <- cell_counts[RNA_cell_type == target, .(
    FragmentLibrary, target_cells = cells
  )]
  other_counts <- cell_counts[RNA_cell_type != target, .(
    other_cells = sum(cells)
  ), by = FragmentLibrary]
  eligible <- merge(target_counts, other_counts, by = "FragmentLibrary")
  eligible <- eligible[
    target_cells >= min_target_cells & other_cells >= min_other_cells
  ]
  setorder(eligible, FragmentLibrary)

  if (nrow(eligible) < min_libraries) {
    log_msg("Skipping %s: only %d eligible libraries.", target, nrow(eligible))
    summary_rows[[target]] <- data.table(
      cell_type = target,
      status = "skipped_insufficient_libraries",
      eligible_libraries = nrow(eligible),
      target_cells = sum(eligible$target_cells),
      other_cells = sum(eligible$other_cells),
      tested_peaks = NA_integer_,
      significant_peaks = NA_integer_,
      significant_open_in_target = NA_integer_,
      significant_closed_in_target = NA_integer_,
      archr_marker_overlap = NA_integer_
    )
    next
  }

  compare_columns <- list()
  compare_meta <- list()
  for (i in seq_len(nrow(eligible))) {
    lib <- eligible$FragmentLibrary[[i]]
    target_col <- which(
      column_meta$RNA_cell_type == target &
        column_meta$FragmentLibrary == lib
    )
    other_cols <- which(
      column_meta$RNA_cell_type != target &
        column_meta$RNA_cell_type != "unclassified" &
        column_meta$FragmentLibrary == lib
    )
    stopifnot(length(target_col) == 1L, length(other_cols) >= 1L)
    compare_columns[[length(compare_columns) + 1L]] <- peak_counts[, target_col, drop = FALSE]
    compare_meta[[length(compare_meta) + 1L]] <- data.table(
      cell_type = target, FragmentLibrary = lib, group = "Target",
      cells = eligible$target_cells[[i]]
    )
    compare_columns[[length(compare_columns) + 1L]] <- Matrix::rowSums(
      peak_counts[, other_cols, drop = FALSE]
    )
    compare_meta[[length(compare_meta) + 1L]] <- data.table(
      cell_type = target, FragmentLibrary = lib, group = "Other",
      cells = eligible$other_cells[[i]]
    )
  }

  compare_matrix <- do.call(cbind, compare_columns)
  sample_meta <- rbindlist(compare_meta)
  sample_meta[, sample_id := paste(FragmentLibrary, group, sep = "__")]
  colnames(compare_matrix) <- sample_meta$sample_id
  sample_meta[, contrast := paste0(target, "_vs_other_labeled")]
  design_rows[[target]] <- sample_meta

  keep <- Matrix::rowSums(compare_matrix >= min_peak_count) >= min_peak_samples
  tested_counts <- as.matrix(compare_matrix[keep, , drop = FALSE])
  rownames(tested_counts) <- which(keep)

  sample_meta[, group := factor(group, levels = c("Other", "Target"))]
  sample_meta[, FragmentLibrary := factor(FragmentLibrary)]
  design <- model.matrix(~ FragmentLibrary + group, data = sample_meta)
  stopifnot(qr(design)$rank == ncol(design))

  y <- DGEList(counts = tested_counts)
  y <- calcNormFactors(y, method = "TMM")
  y <- estimateDisp(y, design = design, robust = TRUE)
  fit <- glmQLFit(y, design = design, robust = TRUE)
  qlf <- glmQLFTest(fit, coef = "groupTarget")
  tab <- as.data.table(topTags(qlf, n = Inf, sort.by = "PValue")$table, keep.rownames = "peak_row")
  tab[, peak_row := as.integer(peak_row)]
  tab[, abs_logFC := abs(logFC)]
  tab <- merge(tab, peak_meta, by = "peak_row", all.x = TRUE)
  tab[, comparison := paste0(target, "_vs_other_labeled")]
  tab[, significant := FDR <= fdr_cutoff & abs(logFC) >= logfc_cutoff]
  tab[, direction := fifelse(
    significant & logFC >= logfc_cutoff, "open_in_target",
    fifelse(significant & logFC <= -logfc_cutoff, "closed_in_target", "not_significant")
  )]

  archr_target <- unique(archr_markers[group == target, .(peak_id)])
  tab[, archr_singlecell_marker := peak_id %chin% archr_target$peak_id]
  setorder(tab, FDR, -abs_logFC)

  safe_target <- gsub("[^A-Za-z0-9]+", "_", target)
  fwrite(tab, file.path(pb_dir, paste0("edgeR_", safe_target, "_vs_other_all.csv.gz")))
  fwrite(
    tab[significant == TRUE],
    file.path(pb_dir, paste0("edgeR_", safe_target, "_vs_other_significant.csv"))
  )
  fwrite(
    head(tab, 200L),
    file.path(pb_dir, paste0("edgeR_", safe_target, "_vs_other_top200.csv"))
  )

  summary_rows[[target]] <- data.table(
    cell_type = target,
    status = "tested",
    eligible_libraries = nrow(eligible),
    target_cells = sum(eligible$target_cells),
    other_cells = sum(eligible$other_cells),
    tested_peaks = nrow(tab),
    significant_peaks = sum(tab$significant),
    significant_open_in_target = sum(tab$direction == "open_in_target"),
    significant_closed_in_target = sum(tab$direction == "closed_in_target"),
    archr_marker_overlap = sum(tab$significant & tab$archr_singlecell_marker)
  )
}

summary_table <- rbindlist(summary_rows, fill = TRUE)
fwrite(summary_table, file.path(pb_dir, "library_aware_edgeR_summary.csv"))
fwrite(rbindlist(design_rows, fill = TRUE), file.path(pb_dir, "library_aware_edgeR_sample_design.csv"))

malignant_all_file <- file.path(pb_dir, "edgeR_Malignant_vs_other_all.csv.gz")
if (file.exists(malignant_all_file)) {
  malignant <- fread(malignant_all_file)
  story_genes <- c("MET", "CAV1", "CAV2", "CAPZA2", "LINC01510", "GPC3", "AFP", "EPCAM")
  story <- malignant[nearestGene %chin% story_genes]
  setorder(story, FDR, -abs_logFC)
  fwrite(story, file.path(pb_dir, "Malignant_MET_CAV1_storyline_peaks.csv"))
  fwrite(
    story[significant == TRUE],
    file.path(pb_dir, "Malignant_MET_CAV1_storyline_significant_peaks.csv")
  )
}

writeLines(
  c(
    "# Library-aware pseudobulk ATAC validation",
    "",
    "The analysis uses pseudo-pair-guided RNA cell-type labels and therefore",
    "remains a validation of internal consistency, not independent ground truth.",
    "",
    "For each cell type, fragment libraries with at least 10 target cells and",
    "20 other labeled cells are retained. Within each eligible library, target",
    "peak counts are compared with the sum of all other labeled cell types.",
    "edgeR TMM normalization and quasi-likelihood GLMs are fitted with fragment",
    "library as a blocking factor. Significant peaks satisfy FDR <= 0.05 and",
    "absolute log2 fold-change >= 1.",
    "",
    "The main HCC storyline table is:",
    "`Malignant_MET_CAV1_storyline_significant_peaks.csv`."
  ),
  file.path(pb_dir, "README_library_aware_pseudobulk.md")
)

log_msg("Library-aware pseudobulk analysis complete.")
