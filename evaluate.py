#!/usr/bin/env python3
"""Evaluate a trained scLAP checkpoint on paired RNA-ATAC cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import SharedLatentPairingModel, evaluate, log, row_zscore


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_matrix(path: Path, columns) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    missing = [column for column in columns if column not in matrix.columns]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} required features; first: {missing[:5]}")
    return matrix.loc[:, columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report translation and exact-pair retrieval metrics for paired cells."
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--checkpoint", default="runs/Fontan1_demo/best_model.pt")
    parser.add_argument(
        "--stats",
        default="",
        help="Preprocessing NPZ; defaults to preprocess_stats.npz beside the checkpoint.",
    )
    parser.add_argument("--rna", default="data/Fontan1_RNA_2000.csv.gz")
    parser.add_argument("--atac", default="data/Fontan1_ATAC_697.csv.gz")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    checkpoint_path = resolve(root, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first or pass --checkpoint."
        )
    stats_path = resolve(root, args.stats) if args.stats else checkpoint_path.parent / "preprocess_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    stats = np.load(stats_path, allow_pickle=True)
    genes = stats["genes"].astype(str).tolist()
    feature_cols = stats["feature_cols"].astype(str).tolist()
    rna = load_matrix(resolve(root, args.rna), genes)
    atac = load_matrix(resolve(root, args.atac), feature_cols)
    common = rna.index.intersection(atac.index, sort=False)
    if common.empty:
        raise ValueError("RNA and ATAC inputs do not share cell IDs")
    rna = rna.loc[common]
    atac = atac.loc[common]

    x = atac.to_numpy(np.float32)
    x = ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    y = row_zscore(rna.to_numpy(np.float32))

    if args.split == "all":
        local_idx = np.arange(len(common), dtype=np.int64)
    else:
        split_ids = set(stats["cell_ids"][stats[f"{args.split}_idx"]].astype(str))
        local_idx = np.flatnonzero(np.asarray([cell_id in split_ids for cell_id in common]))
        if len(local_idx) == 0:
            raise ValueError(f"No input cells belong to the saved {args.split} split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    if not checkpoint_args:
        args_path = checkpoint_path.parent / "args.json"
        checkpoint_args = json.loads(args_path.read_text()) if args_path.exists() else {}
    model = SharedLatentPairingModel(
        atac_dim=len(feature_cols),
        rna_dim=len(genes),
        hidden=checkpoint_args.get("hidden", 1024),
        h_dim=checkpoint_args.get("h_dim", 512),
        z_dim=checkpoint_args.get("z_dim", 128),
        enc_blocks=checkpoint_args.get("enc_blocks", 4),
        dec_blocks=checkpoint_args.get("dec_blocks", 2),
        dropout=checkpoint_args.get("dropout", 0.15),
        activation=checkpoint_args.get("activation", "gelu"),
    ).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    log(f"Evaluating {len(local_idx):,} {args.split} cells on {device}")
    metrics = evaluate(model, x, y, local_idx, args.batch_size, device, args.split)
    output_path = resolve(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
