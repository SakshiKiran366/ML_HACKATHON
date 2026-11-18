"""Comparison plots built ONLY from recorded results (never fabricated).

    python src/evaluation/plots.py [--results results] [--figs results/figures]
Loss curves are drawn only for models that have a real training history (boosting).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar(ax, labels, vals, errs=None, title="", ylabel=""):
    ax.bar(labels, vals, yerr=errs, capsize=4, color="#4C72B0")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figs", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    figs = Path(a.figs or res / "figures"); figs.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(res / "experiments_summary.csv")
    tag = s["data"].iloc[0]
    sup = f"[{tag} DATA]" if tag != "REAL" else ""

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, s.model, s.cv_macro_f1_top1_mean, s.cv_macro_f1_top1_std,
        f"CV macro-F1 (top-1 decision), grouped 5-fold {sup}", "macro F1")
    fig.tight_layout(); fig.savefig(figs / "cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bar(axes[0], s.model, s.test_top1, None, f"Student-test Top-1 accuracy {sup}", "accuracy")
    bar(axes[1], s.model, s.test_macro_f1_top1, None, f"Student-test macro-F1 (top-1) {sup}", "macro F1")
    fig.tight_layout(); fig.savefig(figs / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, s.model, s.test_log_loss, None, f"Student-test log loss {sup}", "log loss (lower better)")
    fig.tight_layout(); fig.savefig(figs / "student_test_log_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(s)); w = 0.38
    ax.bar(x - w / 2, s.train_top1, w, label="train (diagnostic only)")
    ax.bar(x + w / 2, s.test_top1, w, label="student-test")
    ax.set_xticks(x); ax.set_xticklabels(s.model, rotation=20); ax.legend()
    ax.set_title(f"Over/underfitting check: train vs student-test Top-1 {sup}")
    fig.tight_layout(); fig.savefig(figs / "train_vs_test_top1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, s.model, s.infer_us_per_row, None, f"Inference time per row (us) {sup}", "microseconds")
    fig.tight_layout(); fig.savefig(figs / "inference_time.png", dpi=150); plt.close(fig)

    cms = json.loads((res / "confusion_matrices.json").read_text())
    fig, axes = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.4))
    for ax, (name, d) in zip(np.atleast_1d(axes), cms.items()):
        m = np.array(d["student_test"])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(name, fontsize=9); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
    fig.suptitle(f"Student-test confusion matrices (threshold 0.5) {sup}", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "confusion_matrices.png", dpi=150); plt.close(fig)

    hist = json.loads((res / "training_histories.json").read_text())
    for name, h in hist.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["train_loss"], label="train loss")
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="validation loss (held-out trips)")
        ax.set_xlabel("round" if name == "hist_gb" else "epoch"); ax.set_ylabel("log loss")
        ax.set_title(f"{name} training history {sup}"); ax.legend()
        fig.tight_layout(); fig.savefig(figs / f"loss_curve_{name}.png", dpi=150); plt.close(fig)
    print("figures written to", figs)


if __name__ == "__main__":
    main()
