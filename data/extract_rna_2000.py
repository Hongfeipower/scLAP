#!/usr/bin/env python3
"""Create the log-normalized 2,000-gene RNA input used by scLAP."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent


def load_inputs(paths, sample_names):
    objects = []
    for position, path in enumerate(paths):
        adata = sc.read_h5ad(path)
        sample = sample_names[position] if sample_names else Path(path).stem
        adata.obs_names = [f"{sample}_{cell}" for cell in adata.obs_names.astype(str)]
        adata.obs["sample"] = sample
        objects.append(adata)
    return sc.concat(objects, join="outer", merge="same") if len(objects) > 1 else objects[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC, library-normalize, log1p-transform and select/reindex RNA genes for scLAP."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Raw-count H5AD file(s).")
    parser.add_argument("--sample", nargs="+", help="Sample names matching --input order.")
    parser.add_argument("--output", required=True, help="Output cell-by-gene CSV/CSV.GZ.")
    parser.add_argument("--gene-list", default=str(SCRIPT_DIR / "genes_2000.csv"))
    parser.add_argument(
        "--select-hvg",
        action="store_true",
        help="Derive 2,000 HVGs from these training cells instead of applying --gene-list.",
    )
    parser.add_argument("--genes-out", default="")
    parser.add_argument("--skip-qc", action="store_true")
    args = parser.parse_args()

    if args.sample and len(args.sample) != len(args.input):
        raise ValueError("--sample must contain one name per --input file")
    adata = load_inputs(args.input, args.sample)
    adata.var_names_make_unique()

    if not args.skip_qc:
        symbols = adata.var_names.astype(str).str.upper()
        adata.var["mt"] = symbols.str.startswith("MT-")
        adata.var["ribo"] = symbols.str.startswith(("RPS", "RPL"))
        adata.var["hb"] = symbols.str.match(r"^HB(?!P)")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True
        )
        keep = (
            (adata.obs["n_genes_by_counts"] > 500)
            & (adata.obs["n_genes_by_counts"] < 6000)
            & (adata.obs["total_counts"] < 20000)
            & (adata.obs["pct_counts_mt"] < 15)
            & (adata.obs["pct_counts_ribo"] < 1.5)
            & (adata.obs["pct_counts_hb"] < 0.6)
        )
        adata = adata[keep].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if args.select_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        genes = adata.var_names[adata.var["highly_variable"]].astype(str).tolist()
    else:
        genes = pd.read_csv(args.gene_list).iloc[:, 0].astype(str).tolist()
        missing = [gene for gene in genes if gene not in adata.var_names]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} required genes; first: {missing[:10]}")

    selected = adata[:, genes].X
    selected = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    output = pd.DataFrame(selected, index=adata.obs_names.astype(str), columns=genes)
    output.index.name = "cell_id"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, float_format="%.7g")
    if args.genes_out:
        pd.DataFrame({"gene": genes}).to_csv(args.genes_out, index=False)
    print(f"Wrote {output.shape[0]:,} cells x {output.shape[1]:,} genes to {output_path}")


if __name__ == "__main__":
    main()
