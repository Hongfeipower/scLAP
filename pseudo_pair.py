#!/usr/bin/env python3
"""Construct mutual-TopK, thresholded, one-to-one scLAP pseudo-pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import SharedLatentPairingModel, row_zscore


PROJECT_ROOT = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_matrix(path: Path, columns) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    missing = [column for column in columns if column not in matrix.columns]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} required features; first: {missing[:5]}")
    return matrix.loc[:, columns]


def read_rna(paths, genes):
    frames = []
    sources = []
    for path in paths:
        frame = read_matrix(path, genes)
        source = path.name.split("_", 1)[0]
        frames.append(frame)
        sources.extend([source] * len(frame))
        log(f"RNA {path.name}: {frame.shape}")
    matrix = pd.concat(frames, axis=0)
    if matrix.index.has_duplicates:
        duplicated = matrix.index[matrix.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            "RNA cell IDs must be globally unique across input files; duplicate examples: "
            f"{duplicated}"
        )
    return matrix, np.asarray(sources, dtype=object)


@torch.no_grad()
def encode(model, values, modality, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        if modality == "atac":
            _, latent = model.encode_atac(batch)
        else:
            _, latent = model.encode_rna(batch)
        chunks.append(latent.cpu().numpy())
    return np.vstack(chunks).astype(np.float32, copy=False)


@torch.no_grad()
def exact_topk(query, reference, topk, chunk_size, device, label):
    k = min(topk, reference.shape[0])
    indices = np.empty((query.shape[0], k), dtype=np.int32)
    scores = np.empty((query.shape[0], k), dtype=np.float32)
    reference_t = torch.from_numpy(reference.T).to(device)
    log(f"{label}: query={query.shape[0]:,}, reference={reference.shape[0]:,}, topK={k}")
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        similarity = torch.from_numpy(query[start:end]).to(device) @ reference_t
        values, positions = torch.topk(similarity, k=k, dim=1)
        indices[start:end] = positions.cpu().numpy()
        scores[start:end] = values.cpu().numpy()
    return indices, scores


def mutual_edges(atac_to_rna, atac_scores, rna_to_atac, threshold):
    reverse = [dict(zip(row.tolist(), range(1, len(row) + 1))) for row in rna_to_atac]
    edges = []
    for atac_index, (rna_row, score_row) in enumerate(zip(atac_to_rna, atac_scores)):
        for atac_rank, (rna_index, score) in enumerate(zip(rna_row, score_row), start=1):
            rna_rank = reverse[int(rna_index)].get(atac_index)
            if rna_rank is not None and float(score) >= threshold:
                edges.append((atac_index, int(rna_index), float(score), atac_rank, rna_rank))
    return edges


def greedy_unique(edges, n_atac, n_rna):
    used_atac = np.zeros(n_atac, dtype=bool)
    used_rna = np.zeros(n_rna, dtype=bool)
    selected = []
    for edge in sorted(edges, key=lambda value: (-value[2], value[0], value[1])):
        atac_index, rna_index = edge[:2]
        if used_atac[atac_index] or used_rna[rna_index]:
            continue
        used_atac[atac_index] = True
        used_rna[rna_index] = True
        selected.append(edge)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply scLAP using mutual TopK, a cosine threshold and greedy one-to-one matching."
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--checkpoint", default="model/best_model.pt")
    parser.add_argument("--stats", default="model/preprocess_stats.npz")
    parser.add_argument("--atac", required=True, help="Unpaired ATAC697 CSV/CSV.GZ.")
    parser.add_argument("--rna", required=True, nargs="+", help="One or more RNA2000 CSV/CSV.GZ files.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.815542)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--output", default="pseudo_pairs.csv")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    checkpoint_path = resolve(root, args.checkpoint)
    stats_path = resolve(root, args.stats)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. See model/README.md."
        )
    stats = np.load(stats_path, allow_pickle=True)
    genes = stats["genes"].astype(str).tolist()
    feature_cols = stats["feature_cols"].astype(str).tolist()

    atac = read_matrix(resolve(root, args.atac), feature_cols)
    if atac.index.has_duplicates:
        raise ValueError("ATAC cell IDs must be unique")
    rna_paths = [resolve(root, value) for value in args.rna]
    rna, rna_sources = read_rna(rna_paths, genes)
    x = ((atac.to_numpy(np.float32) - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    y = row_zscore(rna.to_numpy(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
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
    log(f"Loaded model on {device}; ATAC={len(atac):,}, RNA={len(rna):,}")

    z_atac = encode(model, x, "atac", args.batch_size, device)
    z_rna = encode(model, y, "rna", args.batch_size, device)
    atac_to_rna, atac_scores = exact_topk(
        z_atac, z_rna, args.topk, args.chunk_size, device, "ATAC-to-RNA"
    )
    rna_to_atac, _ = exact_topk(
        z_rna, z_atac, args.topk, args.chunk_size, device, "RNA-to-ATAC"
    )
    edges = mutual_edges(atac_to_rna, atac_scores, rna_to_atac, args.threshold)
    pairs = greedy_unique(edges, len(atac), len(rna))

    rows = []
    atac_ids = atac.index.to_numpy(str)
    rna_ids = rna.index.to_numpy(str)
    for atac_index, rna_index, score, atac_rank, rna_rank in pairs:
        rows.append(
            {
                "atac_cell_id": atac_ids[atac_index],
                "rna_cell_id": rna_ids[rna_index],
                "rna_source": rna_sources[rna_index],
                "cosine_similarity": score,
                "atac_to_rna_rank": atac_rank,
                "rna_to_atac_rank": rna_rank,
            }
        )
    output_path = resolve(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_columns = [
        "atac_cell_id", "rna_cell_id", "rna_source", "cosine_similarity",
        "atac_to_rna_rank", "rna_to_atac_rank",
    ]
    pd.DataFrame(rows, columns=pair_columns).to_csv(output_path, index=False)
    summary = {
        "n_atac": len(atac),
        "n_rna": len(rna),
        "topk": args.topk,
        "threshold": args.threshold,
        "mutual_edges_passing_threshold": len(edges),
        "greedy_unique_pairs": len(pairs),
    }
    Path(str(output_path) + ".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.save_embeddings:
        np.savez_compressed(
            Path(str(output_path) + ".embeddings.npz"),
            z_atac=z_atac,
            z_rna=z_rna,
            atac_ids=atac_ids,
            rna_ids=rna_ids,
            rna_sources=rna_sources,
        )
    print(json.dumps(summary, indent=2))
    log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
