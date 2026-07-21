#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rna(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_RNA_2000_common_cells.csv"))
    files += sorted(data_dir.glob("*_RNA_2000_common_cells.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No RNA files found in {data_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, index_col=0)
        frames.append(df)
        log(f"RNA {path.name}: {df.shape}")
    rna = pd.concat(frames, axis=0)
    return rna[~rna.index.duplicated(keep="first")]


def load_matrix(path: Path, label: str) -> pd.DataFrame:
    """Load a cell-by-feature CSV/CSV.GZ whose first column is the cell ID."""
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix[~matrix.index.duplicated(keep="first")]
    log(f"{label} {path.name}: {matrix.shape}")
    return matrix


def load_features(feature_dir: Path) -> pd.DataFrame:
    tf_path = feature_dir / "global_tf_tfidf_motif_activity_zscore.csv.gz"
    lsi_path = feature_dir / "atac_lsi_50.csv.gz"
    if not tf_path.exists():
        raise FileNotFoundError(tf_path)
    if not lsi_path.exists():
        raise FileNotFoundError(lsi_path)
    tf = pd.read_csv(tf_path).set_index("cell_id")
    lsi = pd.read_csv(lsi_path).set_index("cell_id")
    tf.columns = [f"TF_{c}" for c in tf.columns]
    lsi.columns = [str(c) if str(c).startswith("LSI_") else f"LSI_{c}" for c in lsi.columns]
    features = pd.concat([tf, lsi], axis=1, join="inner")
    features = features[~features.index.duplicated(keep="first")]
    log(f"ATAC TF features: {tf.shape}; LSI features: {lsi.shape}; combined: {features.shape}")
    return features


def stratified_split(cell_ids, seed: int, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, cid in enumerate(cell_ids):
        sample = str(cid).split("_", 1)[0]
        groups.setdefault(sample, []).append(i)
    train, val, test = [], [], []
    for _, idx in sorted(groups.items()):
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train:n_train + n_val].tolist())
        test.extend(idx[n_train + n_val:].tolist())
    return np.array(train), np.array(val), np.array(test)


def row_zscore(y: np.ndarray, eps=1e-6) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    mu = y.mean(axis=1, keepdims=True)
    sd = y.std(axis=1, keepdims=True)
    return (y - mu) / np.maximum(sd, eps)


def pearson_rows(pred: np.ndarray, true: np.ndarray, eps=1e-8) -> np.ndarray:
    pred = pred - pred.mean(axis=1, keepdims=True)
    true = true - true.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=1) * (true * true).sum(axis=1))
    return ((pred * true).sum(axis=1) / np.maximum(denom, eps)).astype(np.float64)


def pearson_cols(pred: np.ndarray, true: np.ndarray, eps=1e-8) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=0) * (true * true).sum(axis=0))
    return ((pred * true).sum(axis=0) / np.maximum(denom, eps)).astype(np.float64)


def make_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "mish":
        return nn.Mish()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, activation: str, ff_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, h_dim: int, blocks: int, dropout: float, activation: str):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden),
            make_activation(activation),
            nn.Dropout(dropout),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden, dropout, activation))
        layers.extend([nn.LayerNorm(hidden), nn.Linear(hidden, h_dim), make_activation(activation)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, h_dim: int, hidden: int, out_dim: int, blocks: int, dropout: float, activation: str):
        super().__init__()
        layers = [nn.Linear(h_dim, hidden), make_activation(activation), nn.Dropout(dropout)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden, dropout, activation))
        layers.extend([nn.LayerNorm(hidden), nn.Linear(hidden, out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float):
    return GradientReverse.apply(x, lambd)


class SharedLatentPairingModel(nn.Module):
    def __init__(
        self,
        atac_dim: int,
        rna_dim: int,
        hidden: int,
        h_dim: int,
        z_dim: int,
        enc_blocks: int,
        dec_blocks: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.atac_encoder = Encoder(atac_dim, hidden, h_dim, enc_blocks, dropout, activation)
        self.rna_encoder = Encoder(rna_dim, hidden, h_dim, enc_blocks, dropout, activation)
        self.atac_decoder = Decoder(h_dim, hidden, atac_dim, dec_blocks, dropout, activation)
        self.rna_decoder = Decoder(h_dim, hidden, rna_dim, dec_blocks, dropout, activation)
        self.proj_atac = nn.Sequential(nn.LayerNorm(h_dim), nn.Linear(h_dim, z_dim))
        self.proj_rna = nn.Sequential(nn.LayerNorm(h_dim), nn.Linear(h_dim, z_dim))
        self.domain_disc = nn.Sequential(
            nn.Linear(h_dim, max(64, h_dim // 2)),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, h_dim // 2), 2),
        )

    def encode_atac(self, x):
        h = self.atac_encoder(x)
        z = F.normalize(self.proj_atac(h), dim=1)
        return h, z

    def encode_rna(self, y):
        h = self.rna_encoder(y)
        z = F.normalize(self.proj_rna(h), dim=1)
        return h, z

    def forward(self, x_atac, y_rna=None, grl_lambda=0.0):
        h_a, z_a = self.encode_atac(x_atac)
        out = {
            "h_atac": h_a,
            "z_atac": z_a,
            "atac_recon": self.atac_decoder(h_a),
            "rna_pred": self.rna_decoder(h_a),
        }
        if y_rna is not None:
            h_r, z_r = self.encode_rna(y_rna)
            out.update(
                {
                    "h_rna": h_r,
                    "z_rna": z_r,
                    "rna_recon": self.rna_decoder(h_r),
                    "rna_to_atac": self.atac_decoder(h_r),
                }
            )
            if grl_lambda > 0:
                h = torch.cat([h_a, h_r], dim=0)
                out["domain_logits"] = self.domain_disc(grad_reverse(h, grl_lambda))
        return out


def corr_loss_torch(pred, true, eps=1e-8):
    pred = pred - pred.mean(dim=1, keepdim=True)
    true = true - true.mean(dim=1, keepdim=True)
    num = (pred * true).sum(dim=1)
    den = torch.sqrt((pred * pred).sum(dim=1) * (true * true).sum(dim=1)).clamp_min(eps)
    return 1.0 - (num / den).mean()


def symmetric_infonce(z_a, z_r, temperature: float):
    logits = z_a @ z_r.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_ar = F.cross_entropy(logits, labels)
    loss_ra = F.cross_entropy(logits.T, labels)
    with torch.no_grad():
        ranks_ar = (logits > logits.diag()[:, None]).sum(dim=1) + 1
        ranks_ra = (logits.T > logits.diag()[:, None]).sum(dim=1) + 1
        top1 = 0.5 * ((ranks_ar <= 1).float().mean() + (ranks_ra <= 1).float().mean())
        top10 = 0.5 * ((ranks_ar <= 10).float().mean() + (ranks_ra <= 10).float().mean())
    return 0.5 * (loss_ar + loss_ra), logits, top1.item(), top10.item()


def balanced_batch_loss(logits):
    logits = logits.float()
    p_row = torch.softmax(logits, dim=1)
    p_col = torch.softmax(logits, dim=0)
    col_mass = p_row.sum(dim=0)
    row_mass = p_col.sum(dim=1)
    return ((col_mass - 1.0) ** 2).mean() + ((row_mass - 1.0) ** 2).mean()


def sinkhorn_identity_loss(logits, epsilon: float = 0.05, iters: int = 5, eps: float = 1e-8):
    logits = logits.float()
    q = torch.exp((logits - logits.max()) / epsilon)
    q = q / q.sum().clamp_min(eps)
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(eps)
    return -torch.log(torch.diag(q).clamp_min(eps)).mean()


def hard_negative_margin_loss(logits, margin: float):
    logits = logits.float()
    n = logits.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    pos = logits.diag()
    neg_row = logits.masked_fill(eye, -1e4).max(dim=1).values
    neg_col = logits.masked_fill(eye, -1e4).max(dim=0).values
    return 0.5 * (F.relu(margin + neg_row - pos).mean() + F.relu(margin + neg_col - pos).mean())


def same_sample_hard_negative_loss(logits, sample_labels, margin: float):
    logits = logits.float()
    labels = sample_labels.to(logits.device)
    n = logits.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    if not same.any():
        return logits.new_tensor(0.0)
    pos = logits.diag()
    neg_row = logits.masked_fill(~same, -1e4).max(dim=1).values
    neg_col = logits.masked_fill(~same, -1e4).max(dim=0).values
    valid_row = same.any(dim=1)
    valid_col = same.any(dim=0)
    row_loss = F.relu(margin + neg_row[valid_row] - pos[valid_row]).mean() if valid_row.any() else logits.new_tensor(0.0)
    col_loss = F.relu(margin + neg_col[valid_col] - pos[valid_col]).mean() if valid_col.any() else logits.new_tensor(0.0)
    return 0.5 * (row_loss + col_loss)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(h_a, h_r, inv_weight: float, var_weight: float, cov_weight: float, eps: float = 1e-4):
    h_a = h_a.float()
    h_r = h_r.float()
    inv = F.mse_loss(h_a, h_r)

    std_a = torch.sqrt(h_a.var(dim=0) + eps)
    std_r = torch.sqrt(h_r.var(dim=0) + eps)
    var = 0.5 * (F.relu(1.0 - std_a).mean() + F.relu(1.0 - std_r).mean())

    h_a = h_a - h_a.mean(dim=0)
    h_r = h_r - h_r.mean(dim=0)
    cov_a = (h_a.T @ h_a) / max(h_a.shape[0] - 1, 1)
    cov_r = (h_r.T @ h_r) / max(h_r.shape[0] - 1, 1)
    cov = (off_diagonal(cov_a).pow(2).sum() / h_a.shape[1]) + (off_diagonal(cov_r).pow(2).sum() / h_r.shape[1])
    total = inv_weight * inv + var_weight * var + cov_weight * cov
    return total, inv, var, cov


@torch.no_grad()
def predict_arrays(model, x, y, batch_size, device):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pred, za, zr = [], [], []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb, yb)
        pred.append(out["rna_pred"].detach().cpu().numpy())
        za.append(out["z_atac"].detach().cpu().numpy())
        zr.append(out["z_rna"].detach().cpu().numpy())
    return np.vstack(pred), np.vstack(za), np.vstack(zr)


def retrieval_metrics(z_query, z_ref, direction: str):
    sim = z_query.astype(np.float32) @ z_ref.astype(np.float32).T
    n = sim.shape[0]
    diag = sim[np.arange(n), np.arange(n)]
    ranks = (sim > diag[:, None]).sum(axis=1) + 1
    best = np.argmax(sim, axis=1)
    counts = np.bincount(best, minlength=n)
    p = counts[counts > 0].astype(np.float64) / max(counts.sum(), 1)
    entropy = -np.sum(p * np.log(p)) if len(p) else 0.0
    sorted_counts = np.sort(counts.astype(np.float64))
    gini = 0.0
    if sorted_counts.sum() > 0:
        idx = np.arange(1, n + 1)
        gini = float((2 * np.sum(idx * sorted_counts) / (n * sorted_counts.sum())) - (n + 1) / n)
    return {
        "direction": direction,
        "n": int(n),
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "top50": float((ranks <= 50).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_foscttm": float(((ranks - 1) / max(n - 1, 1)).mean()),
        "median_foscttm": float(np.median((ranks - 1) / max(n - 1, 1))),
        "top1_unique_ref": int(np.count_nonzero(counts)),
        "top1_max_ref_hits": int(counts.max()) if n else 0,
        "top1_effective_ref": float(np.exp(entropy)),
        "top1_hit_gini": gini,
    }, ranks, best, sim


def mnn_metrics(best_ar, best_ra):
    pairs = []
    for i, j in enumerate(best_ar):
        if best_ra[j] == i:
            pairs.append((i, j))
    if not pairs:
        return {"mnn_pairs": 0, "mnn_precision_true_pair": float("nan"), "mnn_recall_true_pair": 0.0}
    true = sum(1 for i, j in pairs if i == j)
    return {
        "mnn_pairs": int(len(pairs)),
        "mnn_precision_true_pair": float(true / len(pairs)),
        "mnn_recall_true_pair": float(true / len(best_ar)),
    }


def evaluate(model, x, y, idx, batch_size, device, label: str):
    pred, za, zr = predict_arrays(model, x[idx], y[idx], batch_size, device)
    cell_p = pearson_rows(pred, y[idx])
    gene_p = pearson_cols(pred, y[idx])
    mse = float(np.mean((pred - y[idx]) ** 2))
    mae = float(np.mean(np.abs(pred - y[idx])))
    ss_res = float(np.sum((pred - y[idx]) ** 2))
    ss_tot = float(np.sum((y[idx] - y[idx].mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    ar, ranks_ar, best_ar, _ = retrieval_metrics(za, zr, "ATAC_to_RNA")
    ra, ranks_ra, best_ra, _ = retrieval_metrics(zr, za, "RNA_to_ATAC")
    mnn = mnn_metrics(best_ar, best_ra)
    return {
        "label": label,
        "n": int(len(idx)),
        "translation": {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "cell_pearson_mean": float(cell_p.mean()),
            "cell_pearson_median": float(np.median(cell_p)),
            "gene_pearson_mean": float(gene_p.mean()),
            "gene_pearson_median": float(np.median(gene_p)),
        },
        "retrieval": {
            "ATAC_to_RNA": ar,
            "RNA_to_ATAC": ra,
            "MNN": mnn,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train scLAP on paired RNA2000 and ATAC697 cell-by-feature matrices."
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON file whose model/loss settings become defaults; explicit CLI options win.",
    )
    parser.add_argument("--rna-file", default="data/Fontan1_RNA_2000.csv.gz")
    parser.add_argument("--atac-file", default="data/Fontan1_ATAC_697.csv.gz")
    parser.add_argument(
        "--data-dir",
        default="",
        help="Directory of *_RNA_2000_common_cells.csv[.gz] files for a full multi-sample run.",
    )
    parser.add_argument(
        "--feature-dir",
        default="",
        help="Directory containing the split 647-TF and LSI50 feature files for a full run.",
    )
    parser.add_argument("--run-name", default="Fontan1_demo")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--h-dim", type=int, default=512)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--pretrain-rna-epochs", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--freeze-rna-after-pretrain", action="store_true")
    parser.add_argument("--freeze-rna-encoder", action="store_true")
    parser.add_argument("--freeze-rna-proj", action="store_true")
    parser.add_argument("--freeze-rna-decoder", action="store_true")
    parser.add_argument("--freeze-atac-decoder", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--w-trans", type=float, default=1.0)
    parser.add_argument("--w-trans-corr", type=float, default=0.5)
    parser.add_argument("--w-rna-recon", type=float, default=0.25)
    parser.add_argument("--w-rna-recon-corr", type=float, default=0.10)
    parser.add_argument("--w-atac-recon", type=float, default=0.10)
    parser.add_argument("--w-rna-to-atac", type=float, default=0.05)
    parser.add_argument("--w-pair-cos", type=float, default=0.2)
    parser.add_argument("--w-infonce", type=float, default=0.4)
    parser.add_argument("--w-hardneg", type=float, default=0.05)
    parser.add_argument("--w-same-sample-hardneg", type=float, default=0.0)
    parser.add_argument("--w-balance", type=float, default=0.05)
    parser.add_argument("--w-sinkhorn", type=float, default=0.0)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--w-vicreg-inv", type=float, default=0.0)
    parser.add_argument("--w-vicreg-var", type=float, default=0.0)
    parser.add_argument("--w-vicreg-cov", type=float, default=0.0)
    parser.add_argument("--w-cycle", type=float, default=0.05)
    parser.add_argument("--w-adv", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(args.root) / config_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        explicit = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        path_keys = {
            "root", "config", "rna_file", "atac_file", "data_dir", "feature_dir",
            "run_name", "out_root", "init_checkpoint",
        }
        for key, value in config.items():
            option = "--" + key.replace("_", "-")
            if key not in path_keys and hasattr(args, key) and option not in explicit:
                setattr(args, key, value)

    seed_everything(args.seed)
    root = Path(args.root)
    out_dir = root / args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    frozen_requested = any(
        (args.freeze_rna_encoder, args.freeze_rna_proj, args.freeze_rna_decoder, args.freeze_atac_decoder)
    )
    if frozen_requested and not args.init_checkpoint and args.pretrain_rna_epochs == 0:
        raise ValueError(
            "Frozen strict-v2 components require --init-checkpoint (or an explicit pretraining stage)."
        )

    if args.rna_file and args.atac_file:
        rna_path = Path(args.rna_file)
        atac_path = Path(args.atac_file)
        if not rna_path.is_absolute():
            rna_path = root / rna_path
        if not atac_path.is_absolute():
            atac_path = root / atac_path
        rna = load_matrix(rna_path, "RNA")
        features = load_matrix(atac_path, "ATAC")
    elif args.data_dir and args.feature_dir:
        data_dir = Path(args.data_dir)
        feature_dir = Path(args.feature_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        if not feature_dir.is_absolute():
            feature_dir = root / feature_dir
        rna = load_rna(data_dir)
        features = load_features(feature_dir)
    else:
        raise ValueError(
            "Provide both --rna-file/--atac-file or both --data-dir/--feature-dir."
        )
    common = rna.index.intersection(features.index, sort=False)
    rna = rna.loc[common]
    features = features.loc[common]
    genes = list(rna.columns)
    feature_cols = list(features.columns)
    cell_ids = np.asarray(common)
    log(f"Aligned cells={len(common)}; ATAC dim={features.shape[1]}; RNA dim={rna.shape[1]}")
    if len(common) == 0:
        raise ValueError("RNA and ATAC inputs do not share any cell IDs")
    if features.shape[1] != 697 or rna.shape[1] != 2000:
        raise ValueError(
            f"Expected ATAC697 and RNA2000 inputs; received {features.shape[1]} and {rna.shape[1]}"
        )

    train_idx, val_idx, test_idx = stratified_split(cell_ids, args.seed)
    log(f"Split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    x = features.to_numpy(np.float32)
    y = row_zscore(rna.to_numpy(np.float32))
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = x[train_idx].std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x = ((x - x_mean) / x_std).astype(np.float32)

    np.savez_compressed(
        out_dir / "preprocess_stats.npz",
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cell_ids=cell_ids,
        genes=np.asarray(genes),
        feature_cols=np.asarray(feature_cols),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    model = SharedLatentPairingModel(
        atac_dim=x.shape[1],
        rna_dim=y.shape[1],
        hidden=args.hidden,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(
            f"Loaded init checkpoint {ckpt_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    sample_names = np.asarray([str(cid).split("_", 1)[0] for cid in cell_ids])
    sample_levels = {s: i for i, s in enumerate(sorted(set(sample_names)))}
    sample_codes = np.asarray([sample_levels[s] for s in sample_names], dtype=np.int64)

    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(sample_codes[train_idx]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.pretrain_rna_epochs > 0:
        log(f"Pretraining RNA autoencoder for {args.pretrain_rna_epochs} epochs")
        rna_ds = TensorDataset(torch.from_numpy(y[train_idx]))
        rna_loader = DataLoader(
            rna_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        pre_params = list(model.rna_encoder.parameters()) + list(model.rna_decoder.parameters()) + list(model.proj_rna.parameters())
        pre_opt = torch.optim.AdamW(pre_params, lr=args.pretrain_lr, weight_decay=args.weight_decay)
        for pe in range(1, args.pretrain_rna_epochs + 1):
            model.train()
            total = 0.0
            total_corr = 0.0
            n_pre = 0
            for (yb,) in rna_loader:
                yb = yb.to(device, non_blocking=True)
                pre_opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    h_r, z_r = model.encode_rna(yb)
                    y_rec = model.rna_decoder(h_r)
                    rec = F.smooth_l1_loss(y_rec, yb)
                    rec_corr = corr_loss_torch(y_rec, yb)
                    loss_pre = rec + 0.25 * rec_corr
                scaler.scale(loss_pre).backward()
                scaler.unscale_(pre_opt)
                torch.nn.utils.clip_grad_norm_(pre_params, 5.0)
                scaler.step(pre_opt)
                scaler.update()
                bs = yb.shape[0]
                total += rec.detach().item() * bs
                total_corr += rec_corr.detach().item() * bs
                n_pre += bs
            if pe == 1 or pe % 5 == 0 or pe == args.pretrain_rna_epochs:
                log(f"pretrain_rna_epoch={pe:03d} recon={total / max(n_pre, 1):.4f} corr_loss={total_corr / max(n_pre, 1):.4f}")

    if args.freeze_rna_after_pretrain:
        for module in (model.rna_encoder, model.proj_rna, model.rna_decoder):
            for param in module.parameters():
                param.requires_grad_(False)
        log("Frozen RNA encoder/projection/RNA decoder after pretraining")

    freeze_targets = [
        ("RNA encoder", args.freeze_rna_encoder, model.rna_encoder),
        ("RNA projection", args.freeze_rna_proj, model.proj_rna),
        ("RNA decoder", args.freeze_rna_decoder, model.rna_decoder),
        ("ATAC decoder", args.freeze_atac_decoder, model.atac_decoder),
    ]
    for name, should_freeze, module in freeze_targets:
        if should_freeze:
            for param in module.parameters():
                param.requires_grad_(False)
            log(f"Frozen {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freeze options")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history = []
    best_score = -1e9
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        n_seen = 0
        adv_lambda = args.w_adv * min(1.0, epoch / max(1, args.epochs // 3))
        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(xb, yb, grl_lambda=adv_lambda if args.w_adv > 0 else 0.0)
                trans_mse = F.smooth_l1_loss(out["rna_pred"], yb)
                trans_corr = corr_loss_torch(out["rna_pred"], yb)
                rna_recon = F.smooth_l1_loss(out["rna_recon"], yb)
                rna_recon_corr = corr_loss_torch(out["rna_recon"], yb)
                atac_recon = F.smooth_l1_loss(out["atac_recon"], xb)
                rna_to_atac = F.smooth_l1_loss(out["rna_to_atac"], xb)
                pair_cos = 1.0 - (out["z_atac"] * out["z_rna"]).sum(dim=1).mean()
                infonce, logits, batch_top1, batch_top10 = symmetric_infonce(out["z_atac"], out["z_rna"], args.temperature)
                hardneg = hard_negative_margin_loss(logits, args.margin)
                same_sample_hardneg = same_sample_hard_negative_loss(logits, sb, args.margin)
                balance = balanced_batch_loss(logits)
                sinkhorn = sinkhorn_identity_loss(logits, args.sinkhorn_epsilon, args.sinkhorn_iters)
                vicreg, vicreg_inv, vicreg_var, vicreg_cov = vicreg_loss(
                    out["h_atac"], out["h_rna"], args.w_vicreg_inv, args.w_vicreg_var, args.w_vicreg_cov
                )
                h_pred, z_pred = model.encode_rna(out["rna_pred"])
                cycle = 1.0 - (z_pred * out["z_atac"]).sum(dim=1).mean()
                adv = xb.new_tensor(0.0)
                if args.w_adv > 0 and "domain_logits" in out:
                    domain_labels = torch.cat([
                        torch.zeros(xb.shape[0], dtype=torch.long, device=device),
                        torch.ones(yb.shape[0], dtype=torch.long, device=device),
                    ])
                    adv = F.cross_entropy(out["domain_logits"], domain_labels)
                loss = (
                    args.w_trans * trans_mse
                    + args.w_trans_corr * trans_corr
                    + args.w_rna_recon * rna_recon
                    + args.w_rna_recon_corr * rna_recon_corr
                    + args.w_atac_recon * atac_recon
                    + args.w_rna_to_atac * rna_to_atac
                    + args.w_pair_cos * pair_cos
                    + args.w_infonce * infonce
                    + args.w_hardneg * hardneg
                    + args.w_same_sample_hardneg * same_sample_hardneg
                    + args.w_balance * balance
                    + args.w_sinkhorn * sinkhorn
                    + vicreg
                    + args.w_cycle * cycle
                    + args.w_adv * adv
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            bs = xb.shape[0]
            n_seen += bs
            batch_vals = {
                "loss": loss.detach().item(),
                "trans_mse": trans_mse.detach().item(),
                "trans_corr_loss": trans_corr.detach().item(),
                "rna_recon": rna_recon.detach().item(),
                "atac_recon": atac_recon.detach().item(),
                "pair_cos_loss": pair_cos.detach().item(),
                "infonce": infonce.detach().item(),
                "hardneg": hardneg.detach().item(),
                "same_sample_hardneg": same_sample_hardneg.detach().item(),
                "balance": balance.detach().item(),
                "sinkhorn": sinkhorn.detach().item(),
                "vicreg": vicreg.detach().item(),
                "vicreg_inv": vicreg_inv.detach().item(),
                "vicreg_var": vicreg_var.detach().item(),
                "vicreg_cov": vicreg_cov.detach().item(),
                "cycle": cycle.detach().item(),
                "adv": adv.detach().item(),
                "batch_top1": batch_top1,
                "batch_top10": batch_top10,
            }
            for k, v in batch_vals.items():
                totals[k] = totals.get(k, 0.0) + v * bs
        scheduler.step()
        train_loss = {k: v / max(n_seen, 1) for k, v in totals.items()}

        val_metrics = evaluate(model, x, y, val_idx, args.eval_batch_size, device, "val")
        val_ar = val_metrics["retrieval"]["ATAC_to_RNA"]
        val_ra = val_metrics["retrieval"]["RNA_to_ATAC"]
        val_trans = val_metrics["translation"]
        score = (
            val_ar["top10"]
            + val_ra["top10"]
            + 0.5 * (val_ar["top1"] + val_ra["top1"])
            - 0.5 * (val_ar["mean_foscttm"] + val_ra["mean_foscttm"])
            + 0.05 * val_trans["cell_pearson_mean"]
        )
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_loss,
            "val": val_metrics,
            "score": float(score),
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
        log(
            "epoch={:03d} loss={:.4f} val_cellP={:.4f} val_top1={:.4f}/{:.4f} "
            "val_top10={:.4f}/{:.4f} val_foscttm={:.4f}/{:.4f} val_MNNp={} score={:.4f}".format(
                epoch,
                train_loss["loss"],
                val_trans["cell_pearson_mean"],
                val_ar["top1"],
                val_ra["top1"],
                val_ar["top10"],
                val_ra["top10"],
                val_ar["mean_foscttm"],
                val_ra["mean_foscttm"],
                val_metrics["retrieval"]["MNN"]["mnn_precision_true_pair"],
                score,
            )
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "score": best_score,
                    "genes": genes,
                    "feature_cols": feature_cols,
                },
                out_dir / "best_model.pt",
            )
            (out_dir / "best_val_metrics.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False))
        if epoch - best_epoch >= args.patience:
            log(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}; best_score={best_score:.4f}")
            break

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    final = {
        "best_epoch": int(ckpt["epoch"]),
        "best_score": float(ckpt["score"]),
        "train": evaluate(model, x, y, train_idx, args.eval_batch_size, device, "train"),
        "val": evaluate(model, x, y, val_idx, args.eval_batch_size, device, "val"),
        "test": evaluate(model, x, y, test_idx, args.eval_batch_size, device, "test"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
    log("FINAL " + json.dumps(final["test"], indent=2, ensure_ascii=False))
    log(f"Run complete: {out_dir}")


if __name__ == "__main__":
    main()
