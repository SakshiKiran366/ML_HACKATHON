"""Refit the best recorded Optuna configuration on all training rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "hist_gb_tuned.joblib")
    a = p.parse_args()
    df = pd.read_parquet(ROOT / "tables" / "candidates_train.parquet")
    meta = {"trip_id", "obs_idx", "arc_id", "arc_from", "arc_to", "label", "partition"}
    features = [c for c in df.columns if c not in meta]
    X = df[features].to_numpy(np.float32); y = df.label.to_numpy()
    best = json.loads((ROOT / "results" / "optuna_best.json").read_text())
    params = best["best_params"]
    trials = pd.read_csv(ROOT / "results" / "optuna_trials.csv")
    row = trials.loc[trials.value.idxmax()]
    rounds = max(1, int(round(row["user_attrs_mean_best_round"])))
    model = HistGradientBoostingClassifier(max_iter=rounds, early_stopping=False,
            random_state=42, **params).fit(X, y)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "name": "hist_gb_tuned",
                 "params": params, "max_iter": rounds}, a.out)
    print(f"Saved tuned model from all {df.trip_id.nunique()} train trips: rounds={rounds}, path={a.out}")

if __name__ == "__main__": main()
