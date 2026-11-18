"""Build the candidate table for the train and student_test partitions (evaluator data not touched).

    python src/features/build_table.py [--limit N]
Writes tables/candidates_train.parquet, tables/candidates_student_test.parquet and
results/candidate_diagnostics.csv (candidate miss rate, label-alignment quality per trip).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P

ROOT = Path(__file__).resolve().parents[2]


def one(split, trip_dir):
    t0 = time.perf_counter()
    track, nodes, arcs, route = P.read_trip(trip_dir)
    df, st = P.build_trip(track, nodes, arcs, route, trip_id=int(trip_dir.name), partition=split)
    st.update({"trip_id": int(trip_dir.name), "partition": split, "n_rows": len(df),
               "seconds": time.perf_counter() - t0})
    return split, df, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "tables"))
    a = ap.parse_args()
    jobs = []
    for split in ("train", "student_test"):
        dirs = sorted(d for d in (ROOT / "dataset" / split).iterdir() if d.is_dir())
        jobs += [(split, d) for d in (dirs[: a.limit] if a.limit else dirs)]
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(s, d) for s, d in jobs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "student_test"):
        parts = [df for s, df, _ in res if s == split]
        if parts:
            tab = pd.concat(parts, ignore_index=True)
            tab.to_parquet(out / f"candidates_{split}.parquet", compression="zstd", index=False)
            print(split, tab.shape, f"{(out / f'candidates_{split}.parquet').stat().st_size / 1e6:.1f} MB")
    diag = pd.DataFrame([st for _, _, st in res])
    (ROOT / "results").mkdir(exist_ok=True)
    diag.to_csv(ROOT / "results" / "candidate_diagnostics.csv", index=False)
    cols = ["n_points", "n_rows", "unref_dropped_frac", "label_found_frac", "aligned_d_mean", "aligned_within30",
            "miss_at_1", "miss_at_5", "miss_at_8", "miss_at_12", "miss_at_16", "seconds"]
    print(diag.groupby("partition")[cols].mean().T.round(4))
    print("worst label_found_frac:\n", diag.nsmallest(5, "label_found_frac")[["partition", "trip_id", "label_found_frac", "aligned_d_mean"]])


if __name__ == "__main__":
    main()
