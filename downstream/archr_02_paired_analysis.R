#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ArchR)
  library(data.table)
  library(Matrix)
  library(SummarizedExperiment)
})

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", timestamp(), sprintf(...)))
  flush.console()
}

flatten_markers <- function(marker_list) {
  groups <- names(marker_list)
  rows <- lapply(groups, function(group) {
    x <- as.data.frame(marker_list[[group]])
    if (nrow(x) == 0) return(NULL)
    x$feature <- rownames(x)
    x$group <- group
    x
  })
  rbindlist(rows, fill = TRUE)
}

export_marker_features <- function(se_marker, prefix, cut_off = "FDR <= 0.05 & Log2FC >= 0.5") {
  saveRDS(se_marker, paste0(prefix, "_SummarizedExperiment.rds"))
  markers <- getMarkers(seMarker = se_marker, cutOff = cut_off)
  flat <- flatten_markers(markers)
  fwrite(flat, paste0(prefix, "_markers.csv"))
  invisible(markers)
}

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_dir <- file.path(root, "scATAC", "archr_paired4008_downstream")
table_dir <- file.path(out_dir, "tables")
plot_dir <- file.path(out_dir, "plots")
selected_project_dir <- file.path(out_dir, "ArchRProject_paired4008_qcpass")
arrow_file <- file.path(out_dir, "arrows", "GSE227265_allEligible.arrow")
peak_rds <- file.path(out_dir, "consensus_peakset_allEligible_byLibrary.rds")
pair_file <- file.path(root, "scATAC", "final_pairing_tables", "recommended_mutualTop10_q75_pseudo_pairs_4008.csv")

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot(file.exists(arrow_file))
stopifnot(file.exists(peak_rds))
stopifnot(file.exists(pair_file))

addArchRThreads(threads = 12)
addArchRGenome("hg38")
set.seed(42)

pairs <- fread(pair_file)
stopifnot(nrow(pairs) == 4008L)
stopifnot(uniqueN(pairs$scATAC_cell_id) == 4008L)
pairs[, barcode := sub("^.*#", "", scATAC_cell_id)]
stopifnot(uniqueN(pairs$barcode) == 4008L)

proj <- ArchRProject(
  ArrowFiles = arrow_file,
  outputDirectory = selected_project_dir,
  copyArrows = FALSE
)
archr_cells <- getCellNames(proj)
archr_barcode <- sub("^.*#", "", archr_cells)
pair_idx <- match(archr_barcode, pairs$barcode)
selected_cells <- archr_cells[!is.na(pair_idx)]
stopifnot(length(selected_cells) == 4008L)
proj <- proj[selected_cells, ]

selected_barcodes <- sub("^.*#", "", getCellNames(proj))
meta_idx <- match(selected_barcodes, pairs$barcode)
stopifnot(!anyNA(meta_idx))
meta <- pairs[meta_idx]

proj <- addCellColData(
  ArchRProj = proj, data = meta$cell_type, name = "RNA_cell_type",
  cells = getCellNames(proj), force = TRUE
)
proj <- addCellColData(
  ArchRProj = proj, data = meta$scRNA_dataset, name = "RNA_dataset",
  cells = getCellNames(proj), force = TRUE
)
proj <- addCellColData(
  ArchRProj = proj, data = meta$scRNA_sample, name = "RNA_sample",
  cells = getCellNames(proj), force = TRUE
)
proj <- addCellColData(
  ArchRProj = proj, data = meta$latent_cosine, name = "latent_cosine",
  cells = getCellNames(proj), force = TRUE
)
proj <- addCellColData(
  ArchRProj = proj,
  data = paste0("Lib_", sub("^.*-", "", selected_barcodes)),
  name = "FragmentLibrary",
  cells = getCellNames(proj),
  force = TRUE
)

qc <- as.data.frame(getCellColData(proj, select = c(
  "TSSEnrichment", "nFrags", "RNA_cell_type", "RNA_dataset",
  "RNA_sample", "latent_cosine", "FragmentLibrary"
)))
qc$cell_id <- rownames(qc)
qc$barcode <- sub("^.*#", "", qc$cell_id)
qc$analysis_qc_pass <- qc$TSSEnrichment >= 4 & qc$nFrags >= 1000
fwrite(qc, file.path(table_dir, "paired4008_archr_qc.csv"))

qc_summary <- as.data.table(qc)[, .(
  input_cells = .N,
  qc_pass_cells = sum(analysis_qc_pass),
  qc_pass_fraction = mean(analysis_qc_pass),
  median_TSS = median(TSSEnrichment),
  median_nFrags = median(nFrags)
), by = RNA_cell_type]
fwrite(qc_summary, file.path(table_dir, "paired4008_archr_qc_by_celltype.csv"))

qc_cells <- qc$cell_id[qc$analysis_qc_pass]
log_msg("Selected pseudo-pairs: %d; ArchR QC-pass cells: %d", nrow(qc), length(qc_cells))
proj <- proj[qc_cells, ]

peak_set <- readRDS(peak_rds)
proj <- addPeakSet(ArchRProj = proj, peakSet = peak_set, force = TRUE)
proj <- addPeakMatrix(ArchRProj = proj, ceiling = 4, binarize = FALSE, force = TRUE)

proj <- addIterativeLSI(
  ArchRProj = proj,
  useMatrix = "TileMatrix",
  name = "IterativeLSI",
  iterations = 2,
  clusterParams = list(resolution = 0.2, sampleCells = min(10000, nCells(proj)), n.start = 10),
  varFeatures = 25000,
  dimsToUse = 1:30,
  force = TRUE
)
proj <- addUMAP(
  ArchRProj = proj,
  reducedDims = "IterativeLSI",
  name = "UMAP",
  nNeighbors = 30,
  minDist = 0.5,
  metric = "cosine",
  force = TRUE
)

plot_umap <- plotEmbedding(
  ArchRProj = proj,
  colorBy = "cellColData",
  name = "RNA_cell_type",
  embedding = "UMAP"
)
plotPDF(
  plotList = plot_umap,
  name = "paired4008_qcpass_ATAC_UMAP_RNA_cell_type.pdf",
  ArchRProj = proj,
  addDOC = FALSE,
  width = 7,
  height = 6
)

labeled_cells <- getCellNames(proj)[proj$RNA_cell_type != "unclassified"]
proj_labeled <- proj[labeled_cells, ]
log_msg("Labeled QC-pass cells used for differential analyses: %d", nCells(proj_labeled))

bias_vars <- c("TSSEnrichment", "log10(nFrags)")
marker_peaks <- getMarkerFeatures(
  ArchRProj = proj_labeled,
  useMatrix = "PeakMatrix",
  groupBy = "RNA_cell_type",
  bias = bias_vars,
  testMethod = "wilcoxon"
)
export_marker_features(
  marker_peaks,
  file.path(table_dir, "marker_peaks_by_RNA_cell_type"),
  "FDR <= 0.05 & Log2FC >= 0.5"
)

analysis_types <- sort(unique(as.character(proj_labeled$RNA_cell_type)))
if ("Malignant" %in% analysis_types && length(analysis_types) > 1) {
  malignant_vs_other <- getMarkerFeatures(
    ArchRProj = proj_labeled,
    useMatrix = "PeakMatrix",
    groupBy = "RNA_cell_type",
    useGroups = "Malignant",
    bgdGroups = setdiff(analysis_types, "Malignant"),
    bias = bias_vars,
    testMethod = "wilcoxon"
  )
  export_marker_features(
    malignant_vs_other,
    file.path(table_dir, "marker_peaks_Malignant_vs_other_labeled"),
    "FDR <= 0.05 & Log2FC >= 0.5"
  )
}

marker_gene_scores <- getMarkerFeatures(
  ArchRProj = proj_labeled,
  useMatrix = "GeneScoreMatrix",
  groupBy = "RNA_cell_type",
  bias = bias_vars,
  testMethod = "wilcoxon"
)
export_marker_features(
  marker_gene_scores,
  file.path(table_dir, "marker_GeneScores_by_RNA_cell_type"),
  "FDR <= 0.05 & Log2FC >= 0.5"
)

key_genes <- c(
  "GPC3", "AFP", "EPCAM", "KRT19", "KRT8", "KRT18", "ALB",
  "CD68", "C1QC", "APOE", "LYZ", "MARCO",
  "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRA",
  "PECAM1", "VWF", "KDR", "EMCN",
  "CD3D", "CD3E", "IL7R", "CCL5", "NKG7",
  "CD79A", "MS4A1", "CD74", "CD37",
  "KRT7", "PROM1"
)
gene_score_se <- getMatrixFromProject(proj_labeled, useMatrix = "GeneScoreMatrix")
gene_score_mat <- assay(gene_score_se)
gene_meta <- as.data.frame(rowData(gene_score_se))
gene_col <- intersect(c("name", "symbol", "gene_name", "geneName"), colnames(gene_meta))[1]
gene_symbols <- if (is.na(gene_col)) rownames(gene_score_se) else as.character(gene_meta[[gene_col]])
keep_gene <- !is.na(gene_symbols) & gene_symbols != "" & !duplicated(gene_symbols)
gene_score_mat <- gene_score_mat[keep_gene, , drop = FALSE]
gene_symbols <- gene_symbols[keep_gene]
present_key_genes <- intersect(key_genes, gene_symbols)
cell_types <- as.character(proj_labeled$RNA_cell_type)
key_gene_rows <- lapply(present_key_genes, function(gene) {
  values <- as.numeric(gene_score_mat[match(gene, gene_symbols), ])
  data.table(
    gene = gene,
    RNA_cell_type = sort(unique(cell_types)),
    mean_GeneScore = vapply(sort(unique(cell_types)), function(x) mean(values[cell_types == x]), numeric(1)),
    median_GeneScore = vapply(sort(unique(cell_types)), function(x) median(values[cell_types == x]), numeric(1))
  )
})
fwrite(rbindlist(key_gene_rows), file.path(table_dir, "key_marker_gene_accessibility_GeneScore_by_celltype.csv"))

peak_se <- getMatrixFromProject(proj_labeled, useMatrix = "PeakMatrix")
peak_mat <- assay(peak_se)
peak_meta <- as.data.frame(rowData(peak_se))
peak_meta$peak_id <- rownames(peak_se)
fwrite(peak_meta, file.path(table_dir, "PeakMatrix_peak_metadata.csv"))

pseudobulk_group <- paste0(
  as.character(proj_labeled$RNA_cell_type),
  "__",
  as.character(proj_labeled$FragmentLibrary)
)
design <- sparse.model.matrix(~ 0 + pseudobulk_group)
colnames(design) <- sub("^pseudobulk_group", "", colnames(design))
pseudobulk_mat <- peak_mat %*% design
writeMM(pseudobulk_mat, file.path(table_dir, "PeakMatrix_pseudobulk_celltype_by_library.mtx"))
fwrite(
  data.table(column_index = seq_len(ncol(pseudobulk_mat)), pseudobulk_group = colnames(pseudobulk_mat)),
  file.path(table_dir, "PeakMatrix_pseudobulk_celltype_by_library_columns.csv")
)

saveRDS(marker_peaks, file.path(out_dir, "marker_peaks_by_RNA_cell_type.rds"))
saveArchRProject(ArchRProj = proj, outputDirectory = selected_project_dir, load = FALSE)
log_msg("Stage 2 complete.")
