#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ArchR)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(data.table)
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

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_dir <- file.path(root, "scATAC", "archr_paired4008_downstream")
table_dir <- file.path(out_dir, "tables")
selected_project_dir <- file.path(out_dir, "ArchRProject_paired4008_qcpass")
marker_peak_rds <- file.path(out_dir, "marker_peaks_by_RNA_cell_type.rds")

stopifnot(dir.exists(selected_project_dir))
stopifnot(file.exists(marker_peak_rds))
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

addArchRThreads(threads = 12)
addArchRGenome("hg38")
set.seed(42)

proj <- loadArchRProject(path = selected_project_dir, showLogo = FALSE)
labeled_cells <- getCellNames(proj)[proj$RNA_cell_type != "unclassified"]
proj_labeled <- proj[labeled_cells, ]
log_msg("Labeled QC-pass cells used in Stage 3: %d", nCells(proj_labeled))

proj_labeled <- addGroupCoverages(
  ArchRProj = proj_labeled,
  groupBy = "RNA_cell_type",
  useLabels = FALSE,
  minCells = 20,
  maxCells = 500,
  maxFragments = 25 * 10^6,
  minReplicates = 2,
  maxReplicates = 5,
  force = TRUE
)

proj_labeled <- addMotifAnnotations(
  ArchRProj = proj_labeled,
  motifSet = "homer",
  name = "Motif",
  force = TRUE
)
proj_labeled <- addBgdPeaks(proj_labeled)
proj_labeled <- addDeviationsMatrix(
  ArchRProj = proj_labeled,
  peakAnnotation = "Motif",
  force = TRUE
)

bias_vars <- c("TSSEnrichment", "log10(nFrags)")
marker_motifs <- getMarkerFeatures(
  ArchRProj = proj_labeled,
  useMatrix = "MotifMatrix",
  groupBy = "RNA_cell_type",
  bias = bias_vars,
  testMethod = "wilcoxon"
)
saveRDS(marker_motifs, file.path(out_dir, "marker_motif_deviations_by_RNA_cell_type.rds"))
motif_markers <- getMarkers(marker_motifs, cutOff = "FDR <= 0.05 & MeanDiff >= 0.5")
fwrite(
  flatten_markers(motif_markers),
  file.path(table_dir, "marker_motif_deviations_by_RNA_cell_type.csv")
)

marker_peaks <- readRDS(marker_peak_rds)
motif_enrichment <- peakAnnoEnrichment(
  seMarker = marker_peaks,
  ArchRProj = proj_labeled,
  peakAnnotation = "Motif",
  cutOff = "FDR <= 0.05 & Log2FC >= 0.5"
)
saveRDS(motif_enrichment, file.path(out_dir, "peak_motif_enrichment_by_RNA_cell_type.rds"))
for (assay_name in assayNames(motif_enrichment)) {
  x <- as.data.frame(assay(motif_enrichment, assay_name))
  x$motif <- rownames(x)
  fwrite(x, file.path(table_dir, paste0("peak_motif_enrichment_", assay_name, ".csv")))
}

proj_labeled <- addCoAccessibility(
  ArchRProj = proj_labeled,
  reducedDims = "IterativeLSI",
  dimsToUse = 1:30,
  corCutOff = 0.5,
  maxDist = 1e5
)
coaccess <- getCoAccessibility(
  ArchRProj = proj_labeled,
  corCutOff = 0.5,
  resolution = 1,
  returnLoops = FALSE
)
fwrite(as.data.frame(coaccess), file.path(table_dir, "coaccessibility_links_cor_ge_0.5.csv"))

key_genes <- c(
  "GPC3", "AFP", "EPCAM", "KRT19", "KRT8", "KRT18", "ALB",
  "CD68", "C1QC", "APOE", "COL1A1", "FAP",
  "PECAM1", "VWF", "CD3D", "CD3E", "MS4A1", "CD79A"
)
browser_tracks <- plotBrowserTrack(
  ArchRProj = proj_labeled,
  groupBy = "RNA_cell_type",
  geneSymbol = key_genes,
  upstream = 50000,
  downstream = 50000
)
plotPDF(
  plotList = browser_tracks,
  name = "BrowserTracks_key_marker_genes_by_RNA_cell_type.pdf",
  ArchRProj = proj_labeled,
  addDOC = FALSE,
  width = 9,
  height = 6
)

saveArchRProject(ArchRProj = proj_labeled, outputDirectory = selected_project_dir, load = FALSE)
log_msg("Stage 3 complete.")
