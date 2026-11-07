"""
Helpers shared by Person 2's experiment scripts.

Checkpoint format (agree this with Person 1 so train.py saves the same thing):
    {
        "model_name": "cnn" | "cnn_std_attn" | "cnn_lin_attn",
        "config": {...DEFAULT_CONFIG used for training...},
        "state_dict": model.state_dict(),
        "epoch": int,
        "val_metrics": {"accuracy": ..., "macro_f1": ...},
    }
saved to checkpoints/<model_name>_k<horizon>_seed<seed>.pt
"""

import os
import random

import numpy as np
import torch

from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(model_name, config, ckpt_dir="checkpoints"):
    return os.path.join(ckpt_dir, f"{model_name}_k{config['horizon_k']}_seed{config['seed']}.pt")


def save_checkpoint(path, model_name, config, model, epoch, val_metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_name": model_name, "config": dict(config),
                "state_dict": model.state_dict(), "epoch": epoch,
                "val_metrics": val_metrics}, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_pred) as numpy arrays over a whole loader."""
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        logits = model(X.to(device))
        ps.append(logits.argmax(dim=1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

"""
def compute_metrics(y_true, y_pred):
    Use Person 3's evaluation/metrics.py when present so every number in the
    paper comes from one definition; fall back to sklearn until then.
    try:
        from evaluation.metrics import compute_metrics as shared
        return shared(y_true, y_pred)
    except ImportError:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        return {"accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(p), "macro_recall": float(r),
                "macro_f1": float(f1)}
"""