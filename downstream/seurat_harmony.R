#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(harmony)
  library(Matrix)
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
project_dir <- if (length(args) >= 1) {
  normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

data_dir <- file.path(project_dir, "data", "paired_4008_full_raw_counts")
pair_path <- file.path(
  project_dir,
  "results",
  "final_pairing_tables",
  "recommended_mutualTop10_q75_pseudo_pairs_4008.csv"
)
out_dir <- file.path(project_dir, "downstream_fullraw_seurat")
fig_dir <- file.path(out_dir, "figures")
table_dir <- file.path(out_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

read_source <- function(source_name) {
  source_dir <- file.path(data_dir, source_name)
  message("Reading raw-count subset: ", source_name)
  counts <- ReadMtx(
    mtx = file.path(source_dir, "matrix.mtx.gz"),
    cells = file.path(source_dir, "barcodes.tsv.gz"),
    features = file.path(source_dir, "genes.tsv.gz"),
    cell.column = 1,
    feature.column = 2,
    cell.sep = "\t",
    feature.sep = "\t",
    unique.features = TRUE,
    strip.suffix = FALSE
  )
  rownames(counts) <- make.unique(gsub("_", "-", rownames(counts), fixed = TRUE))
  obj <- CreateSeuratObject(
    counts = counts,
    assay = "RNA",
    project = source_name,
    min.cells = 0,
    min.features = 0
  )
  obj$raw_count_source <- source_name
  obj
}

source_names <- c("GSE125449_Set1", "GSE125449_Set2", "GSE151530", "GSE189903")
objects <- lapply(source_names, read_source)
obj <- merge(objects[[1]], y = objects[-1], merge.data = FALSE)

pairs <- read.csv(pair_path, check.names = FALSE, stringsAsFactors = FALSE)
pairs$seurat_cell_id <- gsub("#", "__", pairs$scRNA_cell_id, fixed = TRUE)
pairs <- pairs[match(colnames(obj), pairs$seurat_cell_id), , drop = FALSE]
if (any(is.na(pairs$scRNA_cell_id))) {
  stop("Merged Seurat object could not be aligned to the 4,008-pair metadata table.")
}
if (ncol(obj) != 4008 || anyDuplicated(colnames(obj))) {
  stop("Expected exactly 4,008 unique scRNA cells after raw-count extraction.")
}
rownames(pairs) <- pairs$seurat_cell_id
obj <- AddMetaData(obj, metadata = pairs)
obj$cell_type <- factor(
  obj$cell_type,
  levels = c("Malignant", "HPC-like", "TAM", "CAF", "TEC", "T cell", "B cell", "unclassified")
)
obj$scRNA_dataset <- factor(obj$scRNA_dataset)
obj$scRNA_sample <- factor(obj$scRNA_sample)
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

qc <- obj@meta.data %>%
  as.data.frame() %>%
  tibble::rownames_to_column("scRNA_cell_id_meta") %>%
  select(
    scRNA_cell_id_meta,
    scATAC_cell_id,
    scRNA_cell_id,
    scRNA_dataset,
    scRNA_sample,
    cell_type,
    latent_cosine,
    nCount_RNA,
    nFeature_RNA,
    percent.mt
  )
write.csv(qc, file.path(table_dir, "paired4008_fullraw_qc_per_cell.csv"), row.names = FALSE)
qc_summary <- qc %>%
  group_by(scRNA_dataset, cell_type) %>%
  summarise(
    n_cells = n(),
    median_nCount_RNA = median(nCount_RNA),
    median_nFeature_RNA = median(nFeature_RNA),
    median_percent_mt = median(percent.mt),
    .groups = "drop"
  )
write.csv(qc_summary, file.path(table_dir, "paired4008_fullraw_qc_summary.csv"), row.names = FALSE)

message("Running standard Seurat normalization and HVG selection.")
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
obj <- JoinLayers(obj)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 3000, verbose = FALSE)
write.csv(
  data.frame(gene = VariableFeatures(obj)),
  file.path(table_dir, "paired4008_fullraw_seurat_hvg3000.csv"),
  row.names = FALSE
)

obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = 40, verbose = FALSE)
set.seed(20260531)
obj <- harmony::RunHarmony(
  object = obj,
  group.by.vars = c("scRNA_dataset", "scRNA_sample"),
  reduction.use = "pca",
  dims.use = 1:30,
  reduction.save = "harmony",
  verbose = TRUE
)
obj <- RunUMAP(
  obj,
  reduction = "harmony",
  dims = 1:30,
  reduction.name = "umap_harmony_fullraw",
  verbose = FALSE
)

cell_type_palette <- c(
  "Malignant" = "#009E73",
  "HPC-like" = "#7A9A01",
  "TAM" = "#0072B2",
  "CAF" = "#D89000",
  "TEC" = "#AA66CC",
  "T cell" = "#00A9B7",
  "B cell" = "#E66B5B",
  "unclassified" = "#E84DAD"
)
p_cell_type <- DimPlot(
  obj,
  reduction = "umap_harmony_fullraw",
  group.by = "cell_type",
  cols = cell_type_palette,
  label = FALSE,
  pt.size = 0.5
) +
  ggtitle("Paired 4,008 scRNA cells: full-gene Seurat HVGs + Harmony") +
  theme_classic(base_size = 12)
p_dataset <- DimPlot(
  obj,
  reduction = "umap_harmony_fullraw",
  group.by = "scRNA_dataset",
  pt.size = 0.5
) +
  ggtitle("Harmony UMAP by source dataset") +
  theme_classic(base_size = 12)
ggsave(file.path(fig_dir, "paired4008_fullraw_harmony_umap_celltype.png"), p_cell_type, width = 9, height = 6.8, dpi = 320)
ggsave(file.path(fig_dir, "paired4008_fullraw_harmony_umap_celltype.pdf"), p_cell_type, width = 9, height = 6.8)
ggsave(file.path(fig_dir, "paired4008_fullraw_harmony_umap_dataset.png"), p_dataset, width = 9, height = 6.8, dpi = 320)
ggsave(file.path(fig_dir, "paired4008_fullraw_harmony_umap_dataset.pdf"), p_dataset, width = 9, height = 6.8)
ggsave(file.path(fig_dir, "paired4008_fullraw_harmony_umap_overview.pdf"), p_cell_type + p_dataset, width = 15, height = 6.8)

emb <- Embeddings(obj, "umap_harmony_fullraw")
umap_table <- cbind(
  obj@meta.data,
  UMAP_1 = emb[, 1],
  UMAP_2 = emb[, 2]
)
write.csv(umap_table, file.path(table_dir, "paired4008_fullraw_harmony_umap_coordinates.csv"), row.names = FALSE)

message("Computing annotation-based differential-expression markers.")
Idents(obj) <- "cell_type"
markers <- FindAllMarkers(
  obj,
  assay = "RNA",
  slot = "data",
  only.pos = TRUE,
  test.use = "wilcox",
  logfc.threshold = 0.25,
  min.pct = 0.10,
  return.thresh = 0.05
)
write.csv(markers, file.path(table_dir, "paired4008_fullraw_celltype_FindAllMarkers.csv"), row.names = FALSE)
top_markers <- markers %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 50) %>%
  ungroup()
write.csv(top_markers, file.path(table_dir, "paired4008_fullraw_celltype_top50_markers.csv"), row.names = FALSE)

known_markers <- tibble::tribble(
  ~expected_cell_type, ~gene, ~marker_group, ~marker_note,
  "Malignant", "AFP", "HCC malignant", "alpha-fetoprotein HCC marker",
  "Malignant", "GPC3", "HCC malignant", "glypican-3 HCC marker",
  "Malignant", "ALB", "HCC malignant", "hepatocyte lineage marker",
  "Malignant", "AKR1B10", "HCC malignant", "HCC-associated malignant-cell marker",
  "Malignant", "EPCAM", "HCC malignant", "epithelial/progenitor-like tumor marker",
  "Malignant", "KRT19", "HCC malignant", "KRT19-positive aggressive HCC program",
  "Malignant", "KRT8", "HCC malignant", "epithelial/hepatocyte tumor marker",
  "Malignant", "KRT18", "HCC malignant", "epithelial/hepatocyte tumor marker",
  "Malignant", "MKI67", "HCC malignant", "cycling malignant-cell marker",
  "Malignant", "TOP2A", "HCC malignant", "cycling malignant-cell marker",
  "HPC-like", "KRT7", "HPC-like", "cholangiocyte/progenitor marker",
  "HPC-like", "KRT19", "HPC-like", "hepatic progenitor/cholangiocyte marker",
  "HPC-like", "EPCAM", "HPC-like", "epithelial progenitor marker",
  "HPC-like", "SOX9", "HPC-like", "hepatic progenitor/cholangiocyte marker",
  "HPC-like", "PROM1", "HPC-like", "stem/progenitor marker",
  "HPC-like", "SLC12A2", "HPC-like", "progenitor/cholangiocyte-like marker",
  "TAM", "CD68", "HCC TAM", "macrophage marker",
  "TAM", "LYZ", "HCC TAM", "myeloid/macrophage marker",
  "TAM", "C1QA", "HCC TAM", "C1Q-positive TAM marker",
  "TAM", "C1QB", "HCC TAM", "C1Q-positive TAM marker",
  "TAM", "C1QC", "HCC TAM", "C1Q-positive TAM marker",
  "TAM", "APOE", "HCC TAM", "lipid-associated TAM marker",
  "TAM", "SPP1", "HCC TAM", "SPP1-positive tumor-promoting TAM marker",
  "TAM", "TREM2", "HCC TAM", "immunosuppressive TAM marker",
  "TAM", "MARCO", "HCC TAM", "tumor-associated macrophage marker",
  "CAF", "COL1A1", "HCC CAF", "fibroblast extracellular-matrix marker",
  "CAF", "COL1A2", "HCC CAF", "fibroblast extracellular-matrix marker",
  "CAF", "COL3A1", "HCC CAF", "fibroblast extracellular-matrix marker",
  "CAF", "DCN", "HCC CAF", "fibroblast marker",
  "CAF", "LUM", "HCC CAF", "fibroblast marker",
  "CAF", "FAP", "HCC CAF", "activated CAF marker",
  "CAF", "ACTA2", "HCC CAF", "myofibroblast CAF marker",
  "CAF", "TAGLN", "HCC CAF", "myofibroblast CAF marker",
  "CAF", "PDGFRB", "HCC CAF", "perivascular fibroblast marker",
  "TEC", "PECAM1", "HCC TEC", "endothelial-cell marker",
  "TEC", "VWF", "HCC TEC", "endothelial-cell marker",
  "TEC", "KDR", "HCC TEC", "VEGF-receptor/endothelial marker",
  "TEC", "ENG", "HCC TEC", "activated endothelial marker",
  "TEC", "PLVAP", "HCC TEC", "tumor endothelial marker",
  "TEC", "EMCN", "HCC TEC", "vascular endothelial marker",
  "TEC", "CLEC14A", "HCC TEC", "tumor endothelial marker",
  "T cell", "CD3D", "HCC T cell", "T-cell receptor-complex marker",
  "T cell", "CD3E", "HCC T cell", "T-cell receptor-complex marker",
  "T cell", "TRAC", "HCC T cell", "T-cell receptor marker",
  "T cell", "CD8A", "HCC T cell", "cytotoxic T-cell marker",
  "T cell", "CD8B", "HCC T cell", "cytotoxic T-cell marker",
  "T cell", "IL7R", "HCC T cell", "memory/helper-like T-cell marker",
  "T cell", "NKG7", "HCC T cell", "cytotoxic lymphocyte marker",
  "T cell", "GZMB", "HCC T cell", "cytotoxic lymphocyte marker",
  "T cell", "PDCD1", "HCC T cell", "exhaustion checkpoint marker",
  "T cell", "CTLA4", "HCC T cell", "checkpoint/Treg marker",
  "T cell", "LAG3", "HCC T cell", "exhaustion checkpoint marker",
  "T cell", "TIGIT", "HCC T cell", "exhaustion checkpoint marker",
  "T cell", "HAVCR2", "HCC T cell", "exhaustion checkpoint marker",
  "T cell", "TOX", "HCC T cell", "exhaustion-associated transcription factor",
  "T cell", "FOXP3", "HCC T cell", "Treg marker",
  "T cell", "IL2RA", "HCC T cell", "Treg activation marker",
  "B cell", "MS4A1", "HCC B cell", "B-cell marker",
  "B cell", "CD79A", "HCC B cell", "B-cell receptor marker",
  "B cell", "CD79B", "HCC B cell", "B-cell receptor marker",
  "B cell", "CD74", "HCC B cell", "antigen-presenting B-cell marker",
  "B cell", "MZB1", "HCC B cell", "plasma-cell marker",
  "B cell", "JCHAIN", "HCC B cell", "plasma-cell marker",
  "B cell", "IGKC", "HCC B cell", "immunoglobulin-light-chain marker"
)
write.csv(known_markers, file.path(table_dir, "literature_hcc_celltype_marker_gene_list.csv"), row.names = FALSE)

markers2 <- markers %>% mutate(gene_upper = toupper(gene))
known2 <- known_markers %>% mutate(gene_upper = toupper(gene))
marker_hits <- markers2 %>%
  inner_join(known2, by = "gene_upper", relationship = "many-to-many") %>%
  mutate(
    gene = gene.x,
    matches_expected_cell_type = cluster == expected_cell_type
  ) %>%
  select(
    gene,
    cluster,
    expected_cell_type,
    matches_expected_cell_type,
    avg_log2FC,
    pct.1,
    pct.2,
    p_val_adj,
    marker_group,
    marker_note
  ) %>%
  arrange(desc(matches_expected_cell_type), expected_cell_type, desc(avg_log2FC))
write.csv(marker_hits, file.path(table_dir, "paired4008_fullraw_DE_literature_marker_hits.csv"), row.names = FALSE)

data_layer <- GetAssayData(obj, assay = "RNA", layer = "data")
known_present <- known_markers %>% filter(gene %in% rownames(data_layer))
avg_rows <- list()
for (ct in levels(obj$cell_type)) {
  cells <- colnames(obj)[obj$cell_type == ct]
  for (gene in unique(known_present$gene)) {
    values <- as.numeric(data_layer[gene, cells])
    avg_rows[[length(avg_rows) + 1]] <- data.frame(
      cell_type = ct,
      gene = gene,
      avg_logexpr = mean(values),
      pct_expr = mean(values > 0),
      stringsAsFactors = FALSE
    )
  }
}
known_expr <- bind_rows(avg_rows) %>%
  left_join(known_markers, by = "gene", relationship = "many-to-many") %>%
  mutate(is_expected_cell_type = cell_type == expected_cell_type)
write.csv(
  known_expr,
  file.path(table_dir, "paired4008_fullraw_literature_marker_expression_by_celltype.csv"),
  row.names = FALSE
)

dot_genes <- c(
  "AFP", "GPC3", "ALB", "AKR1B10", "EPCAM", "KRT19", "TOP2A",
  "KRT7", "SOX9", "PROM1", "SLC12A2",
  "C1QA", "C1QB", "C1QC", "SPP1", "TREM2", "MARCO",
  "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP",
  "PECAM1", "VWF", "KDR", "ENG", "PLVAP", "EMCN",
  "CD3D", "CD3E", "CD8A", "NKG7", "GZMB", "PDCD1", "CTLA4", "TOX",
  "MS4A1", "CD79A", "MZB1", "JCHAIN", "IGKC"
)
dot_genes <- dot_genes[dot_genes %in% rownames(data_layer)]
p_dot <- DotPlot(obj, features = dot_genes, group.by = "cell_type", assay = "RNA") +
  RotatedAxis() +
  ggtitle("Primary liver cancer-related marker expression in paired 4,008 RNA cells") +
  theme_classic(base_size = 10)
ggsave(file.path(fig_dir, "paired4008_fullraw_hcc_marker_dotplot.png"), p_dot, width = 16, height = 5.7, dpi = 320)
ggsave(file.path(fig_dir, "paired4008_fullraw_hcc_marker_dotplot.pdf"), p_dot, width = 16, height = 5.7)

feature_plot_genes <- c("AFP", "GPC3", "AKR1B10", "KRT19", "SPP1", "C1QA", "COL1A2", "PLVAP")
feature_plot_genes <- feature_plot_genes[feature_plot_genes %in% rownames(data_layer)]
if (length(feature_plot_genes) > 0) {
  p_features <- FeaturePlot(
    obj,
    reduction = "umap_harmony_fullraw",
    features = feature_plot_genes,
    ncol = 4,
    order = TRUE
  )
  ggsave(file.path(fig_dir, "paired4008_fullraw_hcc_marker_featureplots.png"), p_features, width = 14, height = 8, dpi = 320)
  ggsave(file.path(fig_dir, "paired4008_fullraw_hcc_marker_featureplots.pdf"), p_features, width = 14, height = 8)
}

write.csv(
  as.data.frame(table(obj$cell_type, obj$scRNA_dataset)),
  file.path(table_dir, "paired4008_fullraw_celltype_by_dataset_counts.csv"),
  row.names = FALSE
)
saveRDS(obj, file.path(out_dir, "paired4008_fullraw_seurat_harmony.rds"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "R_sessionInfo.txt"))
message("Done. Output: ", out_dir)
