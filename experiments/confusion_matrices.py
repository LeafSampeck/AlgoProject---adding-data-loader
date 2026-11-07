"""
Confusion matrices for Section VII (predictive results).

Reads the predictions saved by compare_models.py (so the matrices always match
the table), and writes:
    figures/confusion_<model>.pdf          one per model
    figures/confusion_all.pdf              1x3 panel for the paper (fits one IEEE column
                                           if you use figure* across both columns)
    results/per_class_report.txt           per-class precision / recall / F1

Rows are true labels, columns are predictions, values are row-normalized
(i.e., per-class recall on the diagonal). Raw counts are printed in each cell.

Usage (from the repo root, after compare_models.py):
    python -m experiments.confusion_matrices
    python -m experiments.confusion_matrices --seed 42
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, DEFAULT_CONFIG, MODEL_DISPLAY_NAMES
from models import MODEL_NAMES

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})


def load_preds(name, seed, results_dir="results"):
    path = os.path.join(results_dir, f"predictions_{name}_seed{seed}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["y_true"], d["y_pred"]


def draw(ax, cm, title):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    found = {n: p for n in MODEL_NAMES if (p := load_preds(n, args.seed)) is not None}
    if not found:
        print("No predictions found. Run experiments.compare_models first.")
        return

    report_lines = []
    for name, (y_true, y_pred) in found.items():
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(2.6, 2.4))
        draw(ax, cm, MODEL_DISPLAY_NAMES[name])
        fig.tight_layout()
        fig.savefig(f"figures/confusion_{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        report_lines += [f"== {MODEL_DISPLAY_NAMES[name]} (seed {args.seed}) ==",
                         classification_report(y_true, y_pred, labels=[0, 1, 2],
                                               target_names=CLASS_NAMES, digits=4,
                                               zero_division=0), ""]

    fig, axes = plt.subplots(1, len(found), figsize=(2.4 * len(found), 2.4))
    axes = np.atleast_1d(axes)
    for ax, (name, (y_true, y_pred)) in zip(axes, found.items()):
        im = draw(ax, confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
                  MODEL_DISPLAY_NAMES[name])
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    fig.savefig("figures/confusion_all.pdf", bbox_inches="tight")
    plt.close(fig)

    with open("results/per_class_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print("Wrote figures/confusion_*.pdf and results/per_class_report.txt")


if __name__ == "__main__":
    main()