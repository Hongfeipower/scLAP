import argparse
import gzip
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import io, sparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "processed_atac"

MAX_PEAK_WIDTH = 5000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def infer_dataset_name(path: Path) -> str:
    name = path.name
    m = re.match(r"(GSE\d+)", name)
    if m:
        return m.group(1)
    return name.replace(".fragments.tsv.gz", "").replace(".tsv.gz", "")


def load_lsi_model(path: Path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype="<i4", count=4)
        n_idf, n_rows, n_cols, n_d = map(int, header)
        idf = np.fromfile(f, dtype="<f8", count=n_idf).astype(np.float32)
        svd_v = np.fromfile(f, dtype="<f8", count=n_rows * n_cols).reshape((n_rows, n_cols), order="F").astype(np.float32)
        svd_d = np.fromfile(f, dtype="<f8", count=n_d).astype(np.float32)
        scale = float(np.fromfile(f, dtype="<f8", count=1)[0])
    return idf, svd_v, svd_d, scale


def load_peaks(path: Path):
    peaks = pd.read_csv(path, sep="\t")
    by_chr = {}
    for chrom, sub in peaks.groupby("chr", sort=False):
        order = np.argsort(sub["start"].to_numpy())
        starts = sub["start"].to_numpy(np.int64)[order]
        ends = sub["end"].to_numpy(np.int64)[order]
        idx = sub.index.to_numpy(np.int64)[order]
        by_chr[str(chrom)] = (starts, ends, idx)
    return peaks, by_chr


def load_motif(path: Path):
    raw = io.mmread(path).tocoo()
    return sparse.csc_matrix(
        (np.ones(raw.nnz, dtype=np.float32), (raw.row, raw.col)),
        shape=raw.shape,
    )


class BinaryPairWriter:
    def __init__(self, temp_dir: Path, flush_events: int):
        self.temp_dir = temp_dir
        self.flush_events = flush_events
        self.row_path = temp_dir / "overlap_rows.int32.bin"
        self.col_path = temp_dir / "overlap_cols.int32.bin"
        self.rows = []
        self.cols = []
        self.n_events = 0
        self.row_handle = open(self.row_path, "wb")
        self.col_handle = open(self.col_path, "wb")

    def add(self, hit_peaks, col: int):
        n = len(hit_peaks)
        self.rows.extend(hit_peaks.tolist())
        self.cols.extend([col] * n)
        if len(self.rows) >= self.flush_events:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        rows = np.asarray(self.rows, dtype=np.int32)
        cols = np.asarray(self.cols, dtype=np.int32)
        rows.tofile(self.row_handle)
        cols.tofile(self.col_handle)
        self.n_events += int(rows.size)
        self.rows.clear()
        self.cols.clear()
        log(f"flushed overlap events={self.n_events:,}")

    def close(self):
        self.flush()
        self.row_handle.close()
        self.col_handle.close()


def load_whitelist(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "barcode" not in df.columns or "cell_id" not in df.columns:
        raise ValueError(f"Whitelist must contain barcode and cell_id columns: {path}")
    mapping = dict(zip(df["barcode"].astype(str), df["cell_id"].astype(str)))
    log(f"loaded whitelist {path}: {len(mapping):,} barcodes")
    return mapping


def stream_fragments_to_overlap_bins(fragment_path: Path, peaks_by_chr, temp_dir: Path, dataset_name: str, args, whitelist):
    temp_dir.mkdir(parents=True, exist_ok=True)
    writer = BinaryPairWriter(temp_dir, args.flush_events)
    barcode_to_col = {}
    n_frag = 0
    n_overlap_events = 0
    n_frag_with_overlap = 0
    chrom_seen = set()

    log(f"{dataset_name}: streaming {fragment_path}")
    with gzip.open(fragment_path, "rt") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            if chrom not in peaks_by_chr:
                n_frag += 1
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            barcode = parts[3]
            if whitelist is not None and barcode not in whitelist:
                n_frag += 1
                if args.max_fragments and n_frag >= args.max_fragments:
                    break
                if n_frag % args.report_every == 0:
                    log(
                        f"{dataset_name}: fragments={n_frag:,}, cells={len(barcode_to_col):,}, "
                        f"overlap_fragments={n_frag_with_overlap:,}, overlap_events={n_overlap_events:,}"
                    )
                continue
            chrom_seen.add(chrom)
            starts, ends, peak_idx = peaks_by_chr[chrom]
            left = np.searchsorted(starts, start - MAX_PEAK_WIDTH, side="left")
            right = np.searchsorted(starts, end, side="right")
            if right > left:
                local = np.nonzero(ends[left:right] >= start)[0]
                if local.size:
                    col = barcode_to_col.get(barcode)
                    if col is None:
                        col = len(barcode_to_col)
                        barcode_to_col[barcode] = col
                    hit_peaks = peak_idx[left:right][local]
                    writer.add(hit_peaks, col)
                    n_overlap_events += int(len(hit_peaks))
                    n_frag_with_overlap += 1
            n_frag += 1
            if args.max_fragments and n_frag >= args.max_fragments:
                break
            if n_frag % args.report_every == 0:
                log(
                    f"{dataset_name}: fragments={n_frag:,}, cells={len(barcode_to_col):,}, "
                    f"overlap_fragments={n_frag_with_overlap:,}, overlap_events={n_overlap_events:,}"
                )

    writer.close()
    barcodes = [None] * len(barcode_to_col)
    for barcode, col in barcode_to_col.items():
        barcodes[col] = whitelist[barcode] if whitelist is not None else f"{dataset_name}#{barcode}"
    stats = {
        "dataset": dataset_name,
        "fragments": int(n_frag),
        "cells_with_peak_overlap": int(len(barcode_to_col)),
        "fragments_with_peak_overlap": int(n_frag_with_overlap),
        "overlap_events": int(n_overlap_events),
        "chromosomes_seen": sorted(chrom_seen),
        "row_bin": str(writer.row_path),
        "col_bin": str(writer.col_path),
    }
    log(f"{dataset_name}: stream done; cells={len(barcodes):,}; overlap_events={n_overlap_events:,}")
    return barcodes, stats, writer.row_path, writer.col_path, writer.n_events


def build_peak_cell_from_bins(row_path: Path, col_path: Path, n_events: int, n_peaks: int, n_cells: int):
    log(f"building sparse peak-cell matrix from {n_events:,} overlap events")
    rows = np.memmap(row_path, dtype=np.int32, mode="r", shape=(n_events,))
    cols = np.memmap(col_path, dtype=np.int32, mode="r", shape=(n_events,))
    data = np.ones(n_events, dtype=np.uint8)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n_peaks, n_cells)).tocsc()
    mat.sum_duplicates()
    mat.data[:] = 1
    mat.eliminate_zeros()
    log(f"built peak matrix {mat.shape}, nnz={mat.nnz:,}")
    return mat


def compute_features(dataset_name: str, peak_cell, barcodes, idf, svd_v, motif, motif_names, scale):
    depth = np.asarray(peak_cell.sum(axis=0)).ravel().astype(np.float32)
    keep = depth > 0
    if not np.all(keep):
        peak_cell = peak_cell[:, keep]
        depth = depth[keep]
        barcodes = [b for b, k in zip(barcodes, keep) if k]
    depth[depth <= 0] = 1.0

    tfidf = peak_cell.astype(np.float32, copy=True)
    tfidf = tfidf @ sparse.diags(1.0 / depth, format="csc")
    tfidf = sparse.diags(idf, format="csr") @ tfidf
    tfidf.data = np.log1p(tfidf.data * scale).astype(np.float32)

    log(f"{dataset_name}: projecting LSI")
    lsi = (tfidf.T @ svd_v).astype(np.float32)

    log(f"{dataset_name}: computing motif activity")
    tf_activity = (tfidf.T @ motif).toarray().astype(np.float32)
    mean = tf_activity.mean(axis=0, keepdims=True)
    std = tf_activity.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    tf_activity = (tf_activity - mean) / std

    tf_df = pd.DataFrame(tf_activity, columns=[f"TF_{name}" for name in motif_names])
    lsi_df = pd.DataFrame(lsi, columns=[f"LSI_{i + 1}" for i in range(lsi.shape[1])])
    out = pd.concat([pd.Series(barcodes, name="cell_id"), tf_df, lsi_df], axis=1)
    feature_stats = {
        "cells": int(len(barcodes)),
        "mean_depth": float(depth.mean()),
        "median_depth": float(np.median(depth)),
        "min_depth": float(depth.min()),
        "max_depth": float(depth.max()),
    }
    return out, feature_stats


def main():
    parser = argparse.ArgumentParser(
        description="Project fragment files into the training-derived scLAP ATAC697 feature space."
    )
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-feature-dir", type=Path, required=True)
    parser.add_argument("--lsi-model", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--whitelist", type=Path, default=None)
    parser.add_argument("--report-every", type=int, default=5_000_000)
    parser.add_argument("--flush-events", type=int, default=5_000_000)
    parser.add_argument("--max-fragments", type=int, default=0)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.out_dir / "_tmp_overlap_bins"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    dataset_name = args.dataset_name or infer_dataset_name(args.fragment)
    peaks, peaks_by_chr = load_peaks(args.train_feature_dir / "consensus_peaks_filtered.tsv.gz")
    motif = load_motif(args.train_feature_dir / "motif_peak_by_tf.mtx.gz")
    motif_names = pd.read_csv(args.train_feature_dir / "motif_names.tsv.gz", sep="\t").iloc[:, 0].astype(str).tolist()
    idf, svd_v, svd_d, scale = load_lsi_model(args.lsi_model)
    if len(idf) != len(peaks) or svd_v.shape[0] != len(peaks) or motif.shape[0] != len(peaks):
        raise ValueError(f"Dimension mismatch: peaks={len(peaks)}, idf={len(idf)}, svd_v={svd_v.shape}, motif={motif.shape}")
    log(f"Loaded training model: peaks={len(peaks)}, motifs={len(motif_names)}, lsi_dims={svd_v.shape[1]}, scale={scale}")

    whitelist = load_whitelist(args.whitelist)
    barcodes, stream_stats, row_path, col_path, n_events = stream_fragments_to_overlap_bins(
        args.fragment, peaks_by_chr, temp_dir, dataset_name, args, whitelist
    )
    peak_cell = build_peak_cell_from_bins(row_path, col_path, n_events, len(peaks), len(barcodes))
    features, feature_stats = compute_features(
        dataset_name, peak_cell, barcodes, idf, svd_v, motif, motif_names, scale
    )

    suffix = "_test" if args.max_fragments else ""
    out_path = args.out_dir / f"scATAC_newTF647_LSI50_features{suffix}.csv.gz"
    features.to_csv(out_path, index=False, compression="gzip")
    meta = {
        "feature_file": str(out_path),
        "shape": [int(features.shape[0]), int(features.shape[1])],
        "columns": {
            "cell_id": 1,
            "tf_motif_activity": len(motif_names),
            "lsi": int(svd_v.shape[1]),
        },
        "stream_stats": stream_stats,
        "feature_stats": feature_stats,
        "note": "TF motif activity is z-scored within the scATAC dataset; LSI is projected with the co-seq training LSI model.",
    }
    meta_path = args.out_dir / f"scATAC_feature_metadata{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {out_path} shape={features.shape}")

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        log(f"removed temp dir {temp_dir}")


if __name__ == "__main__":
    main()
