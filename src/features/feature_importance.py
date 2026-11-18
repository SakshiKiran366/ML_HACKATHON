"""Feature importance on the leak-free candidate table (train partition only, trip-grouped 5-fold CV).

Permutation importance = drop in observation-level Top-1 accuracy when one feature's values are
shuffled on held-out trips. Also writes the feature correlation matrix.

    python src/features/feature_importance.py
Outputs: results/feature_importance.csv, results/figures/feature_importance.png,
         results/figures/feature_correlation.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import common as C

N_REPEATS, N_OBS = 3, 4000


def main():
    df = C.load_table(C.TABLES)
    train = df[df.partition == "train"].reset_index(drop=True)
    feats = C.feature_columns(train)
    X, y = train[feats].to_numpy(np.float32), train.label.to_numpy()
    trips = train.trip_id.to_numpy()
    rng = np.random.default_rng(C.SEED)
    drops = {f: [] for f in feats}; base_acc = []
    for k, (tr_i, va_i) in enumerate(C.grouped_folds(trips, 5)):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=C.SEED).fit(X[tr_i], y[tr_i])
        ev = train.iloc[va_i].reset_index(drop=True)
        keys = ev[["trip_id", "obs_idx"]].drop_duplicates().sample(
            min(N_OBS, ev[["trip_id", "obs_idx"]].drop_duplicates().shape[0]), random_state=C.SEED + k)
        sub = ev.merge(keys, on=["trip_id", "obs_idx"]).reset_index(drop=True)
        Xs = sub[feats].to_numpy(np.float32)
        base = C.evaluate(sub, m.predict_proba(Xs)[:, 1])["top1_accuracy"]
        base_acc.append(base)
        for j, f in enumerate(feats):
            for _ in range(N_REPEATS):
                Xp = Xs.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                drops[f].append(base - C.evaluate(sub, m.predict_proba(Xp)[:, 1])["top1_accuracy"])
        print(f"fold {k}: base top1 on sampled held-out observations = {base:.4f}")
    out = pd.DataFrame({"feature": feats,
                        "perm_importance_top1_mean": [np.mean(drops[f]) for f in feats],
                        "perm_importance_top1_std": [np.std(drops[f]) for f in feats]})
    out = out.sort_values("perm_importance_top1_mean", ascending=False)
    C.RESULTS.mkdir(exist_ok=True); (C.RESULTS / "figures").mkdir(exist_ok=True)
    out.to_csv(C.RESULTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    o = out.iloc[::-1]
    ax.barh(o.feature, o.perm_importance_top1_mean, xerr=o.perm_importance_top1_std, color="#4C72B0")
    ax.set_xlabel("drop in Top-1 accuracy when shuffled"); ax.set_title("Permutation importance (grouped 5-fold CV, train trips)")
    fig.tight_layout(); fig.savefig(C.RESULTS / "figures" / "feature_importance.png", dpi=150); plt.close(fig)
    corr = train[feats].sample(min(200_000, len(train)), random_state=C.SEED).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    fig.colorbar(im); ax.set_title("Spearman correlation of features (train sample)")
    fig.tight_layout(); fig.savefig(C.RESULTS / "figures" / "feature_correlation.png", dpi=150); plt.close(fig)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
