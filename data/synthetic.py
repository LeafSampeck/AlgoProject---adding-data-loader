"""
A synthetic data generator that generates LOB-shaped data.

Shapes match the real pipeline exactly:
    X: (N, seq_len, 40) float32     y: (N,) int64 in {0: Up, 1: Stationary, 2: Down}

The labels carry a weak, learnable signal (order-book volume imbalance in the
last few steps), so a training run that is working should beat ~33% accuracy.
These numbers mean nothing about real markets: never report them in the paper.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticLOBDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=100, n_features=40, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, seq_len, n_features)).astype(np.float32)
        # FI-2010 layout per level: [ask_p, ask_v, bid_p, bid_v] x 10 levels.
        ask_vol = X[:, -10:, 1::4].mean(axis=(1, 2))
        bid_vol = X[:, -10:, 3::4].mean(axis=(1, 2))
        imbalance = bid_vol - ask_vol + 0.1 * rng.standard_normal(n_samples)
        lo, hi = np.quantile(imbalance, [1 / 3, 2 / 3])
        y = np.where(imbalance > hi, 0, np.where(imbalance < lo, 2, 1)).astype(np.int64)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_synthetic_dataloaders(batch_size=64, seq_len=100, n_train=2000, n_val=400,
                              n_test=400, seed=0):
    train = SyntheticLOBDataset(n_train, seq_len, seed=seed)
    val = SyntheticLOBDataset(n_val, seq_len, seed=seed + 1)
    test = SyntheticLOBDataset(n_test, seq_len, seed=seed + 2)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size))


def get_dataloaders(source, config):
    """Single switch the experiment scripts use.

    source="synthetic" -> the fake data above.
    source="fi2010"    -> Person 3's loader. Expected to exist as
                          data.fi2010_loader.get_dataloaders(config)
                          returning (train_loader, val_loader, test_loader),
                          with a chronological split and test loader shuffle=False.
    """
    if source == "synthetic":
        return get_synthetic_dataloaders(config["batch_size"], config["seq_len"],
                                         seed=config["seed"])
    if source == "fi2010":
        try:
            from data.fi2010_loader import get_dataloaders as fi_get
        except ImportError as e:
            raise ImportError(
                "data/fi2010_loader.py (Person 3) is not in the repo yet, or does not "
                "define get_dataloaders(config). Use --data synthetic until it lands."
            ) from e
        return fi_get(config)
    raise ValueError(f"unknown data source {source!r}")
