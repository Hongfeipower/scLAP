#!/usr/bin/env python3
import argparse
import csv
import gzip
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_lines_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\r\n") for line in handle if line.strip()]


def parse_header(handle):
    banner = handle.readline().rstrip("\r\n")
    if not banner.startswith("%%MatrixMarket"):
        raise ValueError(f"Not a MatrixMarket file: {banner}")
    comments = []
    line = handle.readline()
    while line.startswith("%"):
        comments.append(line.rstrip("\r\n"))
        line = handle.readline()
    n_rows, n_cols, n_values = map(int, line.split())
    return banner, comments, n_rows, n_cols, n_values


def extract_one(
    source_name: str,
    dataset: str,
    matrix_path: Path,
    genes_path: Path,
    barcodes_path: Path,
    requested_ids,
    out_root: Path,
):
    out_dir = out_root / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    barcodes = read_lines_gz(barcodes_path)
    barcode_to_raw_col = {barcode: i + 1 for i, barcode in enumerate(barcodes)}
    requested_barcodes = [cell_id.split("#", 1)[1] for cell_id in requested_ids]
    selected = [(cell_id, barcode) for cell_id, barcode in zip(requested_ids, requested_barcodes) if barcode in barcode_to_raw_col]
    if not selected:
        log(f"{source_name}: no requested cells found; skip")
        return []

    raw_to_out = {
        barcode_to_raw_col[barcode]: out_col
        for out_col, (_, barcode) in enumerate(selected, start=1)
    }
    selected_full_ids = [cell_id for cell_id, _ in selected]
    log(f"{source_name}: extracting {len(selected_full_ids)} cells from {matrix_path.name}")

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as source:
        banner, comments, n_rows, n_cols, n_values = parse_header(source)
        log(f"{source_name}: source shape={n_rows}x{n_cols}; nnz={n_values:,}")
        with tempfile.NamedTemporaryFile(
            mode="wt",
            encoding="utf-8",
            delete=False,
            suffix=".body.txt.gz",
            dir=out_dir,
        ) as tmp_plain:
            tmp_body_path = Path(tmp_plain.name)

        selected_nnz = 0
        with gzip.open(tmp_body_path, "wt", encoding="utf-8", newline="") as body:
            for row_number, line in enumerate(source, start=1):
                parts = line.split()
                if len(parts) != 3:
                    continue
                raw_col = int(parts[1])
                out_col = raw_to_out.get(raw_col)
                if out_col is not None:
                    body.write(f"{parts[0]} {out_col} {parts[2]}\n")
                    selected_nnz += 1
                if row_number % 20_000_000 == 0:
                    log(f"{source_name}: scanned {row_number:,}/{n_values:,} matrix entries")

    output_matrix = out_dir / "matrix.mtx.gz"
    with gzip.open(output_matrix, "wt", encoding="utf-8", newline="") as output:
        output.write(f"{banner}\n")
        for comment in comments:
            output.write(f"{comment}\n")
        output.write(f"{n_rows} {len(selected_full_ids)} {selected_nnz}\n")
        with gzip.open(tmp_body_path, "rt", encoding="utf-8") as body:
            shutil.copyfileobj(body, output)
    tmp_body_path.unlink()

    shutil.copyfile(genes_path, out_dir / "genes.tsv.gz")
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        for cell_id in selected_full_ids:
            handle.write(f"{cell_id.replace('#', '__')}\n")
    pd.DataFrame(
        {
            "source_name": source_name,
            "dataset": dataset,
            "scRNA_cell_id": selected_full_ids,
            "seurat_cell_id": [cell_id.replace("#", "__") for cell_id in selected_full_ids],
            "original_barcode": [barcode for _, barcode in selected],
            "original_matrix_column": [barcode_to_raw_col[barcode] for _, barcode in selected],
        }
    ).to_csv(out_dir / "selected_cells.csv", index=False)
    log(f"{source_name}: wrote {len(selected_full_ids)} cells; selected nnz={selected_nnz:,}")
    return selected_full_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--raw-rna-dir", required=True)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    raw_rna_dir = Path(args.raw_rna_dir)
    pair_path = project_dir / "results/final_pairing_tables/recommended_mutualTop10_q75_pseudo_pairs_4008.csv"
    out_root = project_dir / "data/paired_4008_full_raw_counts"
    out_root.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(pair_path)

    sources = [
        ("GSE125449_Set1", "GSE125449"),
        ("GSE125449_Set2", "GSE125449"),
        ("GSE151530", "GSE151530"),
        ("GSE189903", "GSE189903"),
    ]
    selected_all = []
    manifest_rows = []
    for source_name, dataset in sources:
        requested_ids = pairs.loc[pairs["scRNA_dataset"] == dataset, "scRNA_cell_id"].tolist()
        selected = extract_one(
            source_name=source_name,
            dataset=dataset,
            matrix_path=raw_rna_dir / f"{source_name}_matrix.mtx.gz",
            genes_path=raw_rna_dir / f"{source_name}_genes.tsv.gz",
            barcodes_path=raw_rna_dir / f"{source_name}_barcodes.tsv.gz",
            requested_ids=requested_ids,
            out_root=out_root,
        )
        selected_all.extend(selected)
        manifest_rows.append({"source_name": source_name, "dataset": dataset, "selected_cells": len(selected)})

    expected = set(pairs["scRNA_cell_id"])
    observed = set(selected_all)
    missing = sorted(expected - observed)
    duplicates = len(selected_all) - len(observed)
    manifest = pd.DataFrame(manifest_rows)
    manifest.loc[len(manifest)] = ["TOTAL", "ALL", len(selected_all)]
    manifest.to_csv(out_root / "extraction_manifest.csv", index=False)
    pd.DataFrame({"missing_scRNA_cell_id": missing}).to_csv(out_root / "missing_cells.csv", index=False)

    log(f"Selected total rows={len(selected_all):,}; unique cells={len(observed):,}; duplicates={duplicates:,}")
    if missing or duplicates or len(observed) != len(expected):
        raise RuntimeError(
            f"Extraction validation failed: missing={len(missing)}, duplicates={duplicates}, "
            f"observed={len(observed)}, expected={len(expected)}"
        )
    log(f"Extraction complete: {out_root}")


if __name__ == "__main__":
    main()
