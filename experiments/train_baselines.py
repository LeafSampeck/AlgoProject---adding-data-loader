"""
Train the two baselines (and, if present, the proposed model) under the SHARED
training configuration in config.py, and save checkpoints that
compare_models.py / confusion_matrices.py can read.

Training protocol (identical for every model -- report it in Section IV):
    * optimizer      : AdamW(lr, weight_decay) from DEFAULT_CONFIG
    * loss           : cross-entropy, optionally class-weighted (inverse frequency
                       on the TRAIN split only, so no information leaks from val/test)
    * model selection: best validation macro-F1; early stopping after
                       `early_stop_patience` epochs without improvement
    * seeds          : python / numpy / torch seeded before the model AND the data
                       loaders are built, so shuffling order is reproducible too

Outputs:
    checkpoints/<model>_k<horizon>_seed<seed>.pt     best-on-validation weights
    results/history_<model>_seed<seed>.json          per-epoch loss / val metrics

Person 1 owns the canonical train.py. This script implements the same protocol
so Person 2 can train baselines in parallel; if train.py diverges (e.g. a
different scheduler), make the two agree before the final runs.

Usage (from the repo root):
    python -m experiments.train_baselines --data synthetic --epochs 3   # smoke test
    python -m experiments.train_baselines --data fi2010 --seeds 42 43 44
    python -m experiments.train_baselines --data fi2010 --models cnn cnn_std_attn cnn_lin_attn
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from data.synthetic import get_dataloaders
from experiments.common import (checkpoint_path, compute_metrics, get_device,
                                predict, save_checkpoint, set_seed)
from models import build_model
from models.common import count_parameters

BASELINES = ["cnn", "cnn_std_attn"]


def train_labels(loader):
    """All training labels, without touching the inputs when the dataset exposes y."""
    ds = loader.dataset
    if hasattr(ds, "y"):
        return torch.as_tensor(ds.y).cpu().numpy()
    return np.concatenate([y.numpy() for _, y in loader])


def class_weights(labels, num_classes):
    """Inverse-frequency weights normalised to mean 1 (sklearn's 'balanced')."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32), counts.astype(int).tolist()


def fit(model_name, config, data_source, ckpt_dir="checkpoints", out_dir="results"):
    set_seed(config["seed"])
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(data_source, config)
    model = build_model(model_name, config).to(device)

    if config["use_class_weights"]:
        weights, counts = class_weights(train_labels(train_loader), config["num_classes"])
        print(f"  train class counts {counts} -> weights {weights.numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                            weight_decay=config["weight_decay"])

    path = checkpoint_path(model_name, config, ckpt_dir)
    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    print(f"  params {count_parameters(model):,}  device {device}  -> {path}")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        val = compute_metrics(*predict(model, val_loader, device))
        rec = {"epoch": epoch, "train_loss": total / n, "seconds": time.time() - t0,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(f"  epoch {epoch:3d}  loss {rec['train_loss']:.4f}  "
              f"val acc {val['accuracy']:.4f}  val macro-F1 {val['macro_f1']:.4f}  "
              f"({rec['seconds']:.1f}s)")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            save_checkpoint(path, model_name, config, model, epoch, val)
        else:
            bad_epochs += 1
            if bad_epochs >= config["early_stop_patience"]:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"history_{model_name}_seed{config['seed']}.json"), "w") as f:
        json.dump({"model": model_name, "data": data_source, "config": config,
                   "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                   "history": history}, f, indent=2)
    return best_f1, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "fi2010"], default="synthetic")
    ap.add_argument("--models", nargs="+", default=BASELINES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[DEFAULT_CONFIG["seed"]])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override DEFAULT_CONFIG['epochs'] (smoke tests only)")
    args = ap.parse_args()

    for name in args.models:
        for seed in args.seeds:
            cfg = dict(DEFAULT_CONFIG, seed=seed)
            if args.epochs is not None:
                cfg["epochs"] = args.epochs
            print(f"== {name}  seed {seed}  data {args.data} ==")
            f1, ep = fit(name, cfg, args.data)
            print(f"   best val macro-F1 {f1:.4f} at epoch {ep}\n")


if __name__ == "__main__":
    main()
