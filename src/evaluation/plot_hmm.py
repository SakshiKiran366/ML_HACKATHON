"""Figures for the classifier-only vs HMM comparison, drawn only from results/hmm_*.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2] / "results"
(R / "figures").mkdir(exist_ok=True)
res = pd.read_csv(R / "hmm_results.csv")
names = {"classifier_only": "Classifier only", "viterbi_simple_teammate": "Viterbi (hand-set)",
         "hmm_default_params": "HMM (default)", "hmm_tuned_params": "HMM (tuned)"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ["top1", "route_f1", "reachable_continuity"],
                          ["Observation Top-1 accuracy", "Route-level F1 (arc sets)", "Continuity (network-reachable)"]):
    w = 0.38
    for i, part in enumerate(["train_oof", "student_test"]):
        d = res[res.partition == part].set_index("method").reindex(list(names))
        ax.bar(np.arange(len(names)) + (i - 0.5) * w, d[col], w, label=part.replace("_", " "))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names.values(), rotation=20, fontsize=8)
    lo = res[col].min(); ax.set_ylim(max(0, lo - 0.03), min(1.0, res[col].max() + 0.015)); ax.set_title(title, fontsize=10)
axes[0].legend(fontsize=8)
fig.suptitle("Classifier-only vs HMM decoding (train = out-of-fold, 80 trips; student-test = 10 trips)", fontsize=10)
fig.tight_layout(); fig.savefig(R / "figures" / "hmm_comparison.png", dpi=150)
pt = pd.read_csv(R / "hmm_per_trip_student_test.csv")
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(pt))
ax.bar(x - 0.2, pt.top1_classifier, 0.4, label="classifier only"); ax.bar(x + 0.2, pt.top1_hmm, 0.4, label="classifier + HMM")
ax.set_xticks(x); ax.set_xticklabels(pt.trip_id.astype(str)); ax.set_ylim(0.6, 1.0); ax.set_xlabel("student-test trip")
ax.set_ylabel("Top-1 accuracy"); ax.legend(); ax.set_title("Per-trip Top-1: HMM helps every student-test trip")
fig.tight_layout(); fig.savefig(R / "figures" / "hmm_per_trip.png", dpi=150)
print("saved figures; HMM improves Top-1 on", int((pt.top1_hmm > pt.top1_classifier).sum()), "of", len(pt), "trips")
