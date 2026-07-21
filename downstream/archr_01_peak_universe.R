#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ArchR)
  library(data.table)
  library(GenomicRanges)
})

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", timestamp(), sprintf(...)))
  flush.console()
}

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_dir <- file.path(root, "scATAC", "archr_paired4008_downstream")
arrow_dir <- file.path(out_dir, "arrows")
all_project_dir <- file.path(out_dir, "ArchRProject_allEligible")
table_dir <- file.path(out_dir, "tables")
fragment_file <- file.path(root, "scATAC", "GSE227265_fragments_AllSamples.tsv.gz")
whitelist_file <- file.path(root, "scATAC", "GSE227265_cell_whitelist.csv")
macs2 <- if (length(args) >= 2) args[[2]] else Sys.which("macs2")
arrow_file <- file.path(arrow_dir, "GSE227265_allEligible.arrow")
peak_rds <- file.path(out_dir, "consensus_peakset_allEligible_byLibrary.rds")
peak_bed <- file.path(table_dir, "consensus_peakset_allEligible_byLibrary.bed")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(arrow_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(fragment_file))
stopifnot(file.exists(whitelist_file))
stopifnot(file.exists(macs2))

addArchRThreads(threads = 12)
addArchRGenome("hg38")
set.seed(42)

whitelist <- fread(whitelist_file)
stopifnot("barcode" %in% colnames(whitelist))
valid_barcodes <- unique(as.character(whitelist$barcode))
log_msg("Feature-eligible ATAC whitelist: %d cells", length(valid_barcodes))

if (!file.exists(arrow_file)) {
  log_msg("Creating ArrowFile from the full fragment file.")
  old_wd <- getwd()
  setwd(arrow_dir)
  arrow_files <- createArrowFiles(
    inputFiles = c(GSE227265 = fragment_file),
    sampleNames = "GSE227265_allEligible",
    outputNames = "GSE227265_allEligible",
    validBarcodes = valid_barcodes,
    minTSS = 0,
    minFrags = 0,
    maxFrags = 1e+07,
    addTileMat = TRUE,
    addGeneScoreMat = TRUE,
    force = TRUE,
    threads = getArchRThreads()
  )
  setwd(old_wd)
  log_msg("ArrowFile created: %s", paste(arrow_files, collapse = ", "))
} else {
  log_msg("Reusing existing ArrowFile: %s", arrow_file)
}

proj <- ArchRProject(
  ArrowFiles = arrow_file,
  outputDirectory = all_project_dir,
  copyArrows = FALSE
)

archr_cells <- getCellNames(proj)
barcodes <- sub("^.*#", "", archr_cells)
fragment_library <- paste0("Lib_", sub("^.*-", "", barcodes))
proj <- addCellColData(
  ArchRProj = proj,
  data = fragment_library,
  cells = archr_cells,
  name = "FragmentLibrary",
  force = TRUE
)

qc <- as.data.frame(getCellColData(proj, select = c("TSSEnrichment", "nFrags")))
qc$cell_id <- rownames(qc)
qc$barcode <- sub("^.*#", "", qc$cell_id)
qc$FragmentLibrary <- paste0("Lib_", sub("^.*-", "", qc$barcode))
fwrite(qc, file.path(table_dir, "allEligible_arrow_qc.csv"))
log_msg("Arrow project contains %d cells across %d fragment libraries", nrow(qc), length(unique(qc$FragmentLibrary)))

if (!file.exists(peak_rds)) {
  log_msg("Building fragment-library pseudo-bulk coverages from all eligible ATAC cells.")
  proj <- addGroupCoverages(
    ArchRProj = proj,
    groupBy = "FragmentLibrary",
    useLabels = FALSE,
    minCells = 40,
    maxCells = 500,
    maxFragments = 25 * 10^6,
    minReplicates = 2,
    maxReplicates = 5,
    force = TRUE
  )

  log_msg("Calling a group-independent consensus peak universe with MACS2.")
  proj <- addReproduciblePeakSet(
    ArchRProj = proj,
    groupBy = "FragmentLibrary",
    peakMethod = "Macs2",
    pathToMacs2 = macs2,
    maxPeaks = 200000,
    reproducibility = "2",
    force = TRUE
  )
  peak_set <- getPeakSet(proj)
  saveRDS(peak_set, peak_rds)
} else {
  log_msg("Reusing existing consensus peak set: %s", peak_rds)
  peak_set <- readRDS(peak_rds)
}

peak_df <- data.frame(
  chr = as.character(seqnames(peak_set)),
  start = start(peak_set) - 1L,
  end = end(peak_set),
  peak_id = paste0(seqnames(peak_set), ":", start(peak_set), "-", end(peak_set)),
  stringsAsFactors = FALSE
)
fwrite(peak_df, peak_bed, sep = "\t", col.names = FALSE)
log_msg("Consensus peak universe contains %d peaks", nrow(peak_df))
log_msg("Stage 1 complete.")
