"""Fit one chosen zoo model on all training trips and save the API artifact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import common as C
from zoo import registry

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, default=C.TABLES / "candidates_train.parquet")
    p.add_argument("--model", choices=registry().keys(), default="hist_gb")
    p.add_argument("--out", type=Path, default=C.ROOT / "artifacts" / "final_model.joblib")
    args = p.parse_args()
    df = pd.read_parquet(args.table)
    if "partition" in df and not (df.partition == "train").all():
        df = df[df.partition == "train"]
    feats = C.feature_columns(df)
    X, y = df[feats].to_numpy(np.float32), df.label.to_numpy()
    _family, factory, config = registry()[args.model]
    model = factory().fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats, "name": args.model, "config": config}, args.out)
    print(f"Saved {args.model} artifact to {args.out} from {df.trip_id.nunique()} training trips")

if __name__ == "__main__": main()
