# scLAP
 a liver-adapted contrastive shared-latent framework for pseudo-pairing single-cell transcriptomes and chromatin accessibility profiles

 
```text
scLAP_GitHub_release/
|-- data/          Example inputs and RNA/ATAC feature construction
|-- model/         Published configuration, preprocessing statistics and metrics
|-- downstream/    Seurat/Harmony and ArchR analysis
|-- train.py       Paired co-assay training
|-- evaluate.py    Translation and exact-pair retrieval evaluation
|-- pseudo_pair.py Mutual-TopK pseudo-pair construction
|-- environment.yml
`-- README.md
```

## Input

scLAP uses cell-by-feature matrices with cell IDs in the first column:

- RNA: 2,000 log-normalized highly variable genes.
- ATAC: 647 TF motif-activity features followed by 50 LSI components.

The included Fontan1 example contains 2,253 paired cells from GSE223843. RNA and ATAC rows have identical cell IDs and order. This compact sample is intended for code testing; the manuscript model was trained on all six paired liver co-assay samples.

## Installation

```bash
conda env create -f environment.yml
conda activate sclap
```

## Quick start

Run a short end-to-end demonstration:

