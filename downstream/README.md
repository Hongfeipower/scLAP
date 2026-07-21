# Downstream workflows

The scripts are kept in execution order without additional nested directories.

1. `extract_paired_rna_counts.py` extracts full raw RNA counts for selected pseudo-pairs.
2. `seurat_harmony.R` performs RNA normalization, Harmony integration, UMAP visualization and marker analysis.
3. `archr_01_peak_universe.R` builds the all-eligible-cell peak universe.
4. `archr_02_paired_analysis.R` analyzes the selected pseudo-paired ATAC cells using transferred RNA labels.
5. `archr_03_motif_coaccess.R` performs motif enrichment, co-accessibility and track analysis.
6. `archr_04_pseudobulk_edger.R` performs library-aware pseudobulk differential accessibility.

The ArchR scripts retain the manuscript analysis settings. Pass the project root as the first command-line argument; `archr_01_peak_universe.R` optionally accepts the MACS2 path as its second argument. Peak calling uses the full eligible fragment population; cell-type differential peaks, motif enrichment and co-accessibility use the selected pseudo-paired cells.
