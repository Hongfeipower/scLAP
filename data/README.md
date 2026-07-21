# Data and feature construction

## Included example

| File | Shape | Description |
|---|---:|---|
| `Fontan1_RNA_2000.csv.gz` | 2,253 x 2,000 | Log-normalized RNA HVGs |
| `Fontan1_ATAC_697.csv.gz` | 2,253 x 697 | 647 TF motif activities + LSI1-50 |
| `genes_2000.csv` | 2,000 | Ordered RNA schema |
| `atac_features_697.csv` | 697 | Ordered ATAC schema |

Both matrices use `cell_id` as the first column and contain the same paired cells in the same order.

## RNA

For training-set HVG selection:

```bash
python data/extract_rna_2000.py \
  --input sample1_raw.h5ad sample2_raw.h5ad \
  --sample Sample1 Sample2 \
  --select-hvg \
  --genes-out data/genes_2000.csv \
  --output training_RNA_2000.csv.gz
```

For an external RNA dataset, omit `--select-hvg`; the script then applies the fixed `genes_2000.csv` schema.

## Paired peak matrices

```bash
Rscript data/extract_atac_697.R path/to/raw_peak_matrices path/to/output_features
Rscript data/export_lsi_model.R path/to/output_features path/to/output_features
```

The input directory is expected to contain matched `GSM*_ATAC_*_features.txt.gz`, `*_matrix.mtx.gz` and `*_barcodes.txt.gz` files. The R workflow constructs a fixed-width consensus peak universe, computes TF-IDF, scans JASPAR2022 motifs, and writes the directly usable `ATAC_697.csv.gz` matrix containing 647 motif-activity features plus 50 LSI components.

## Fragment projection

```bash
python data/project_fragments_to_atac_697.py \
  --fragment fragments.tsv.gz \
  --train-feature-dir path/to/output_features \
  --lsi-model path/to/output_features/lsi_projection_model.bin \
  --out-dir path/to/projected_features
```

External fragments are counted against the training-derived consensus peaks and projected with the training-derived motif and LSI definitions.
