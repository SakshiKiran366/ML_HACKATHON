"""Compare classifier-only and Viterbi map matching on permitted student-test trips."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.main import load_model, predict, read_trip  # noqa: E402
import pipeline as P  # noqa: E402

def continuity(frame):
    f = frame.sort_values("obs_idx")
    if len(f) < 2: return 1.0
    adjacent_obs = np.diff(f.obs_idx.to_numpy()) == 1
    linked = ((f.arc_id.to_numpy()[1:] == f.arc_id.to_numpy()[:-1]) |
              (f.arc_to.to_numpy()[:-1] == f.arc_from.to_numpy()[1:]))
    return float(linked[adjacent_obs].mean()) if adjacent_obs.any() else 1.0

def main():
    load_model()
    rows = []
    for folder in sorted((ROOT / "dataset" / "student_test").iterdir()):
        if not folder.is_dir(): continue
        tid = folder.name
        folder, track, nodes, arcs = read_trip(tid)
        frame, decoded, _ = predict(track, nodes, arcs)
        route = P.read_trip(folder)[3]
        if route is None: continue
        if "score" not in frame: raise RuntimeError("prediction scores missing")
        raw = frame.loc[frame.groupby("obs_idx").score.idxmax()].sort_values("obs_idx")
        clean, _ = P.clean_track(track)
        lon0, lat0 = clean[:, 0].mean(), clean[:, 1].mean()
        points = P.project(clean[:, :2], lon0, lat0)
        projected = P.project(nodes, lon0, lat0)
        A, B = projected[arcs[:, 0]], projected[arcs[:, 1]]
        truth, _ = P.align_labels(points, clean[:, 2], A, B, route)
        def accuracy(pred):
            obs = pred.obs_idx.to_numpy(dtype=int)
            ok = obs < len(truth)
            return float(np.mean(pred.arc_id.to_numpy()[ok] == truth[obs[ok]])) if ok.any() else 0.0
        rows.append({"trip_id": tid, "classifier_accuracy": accuracy(raw), "viterbi_accuracy": accuracy(decoded),
                     "classifier_continuity": continuity(raw), "viterbi_continuity": continuity(decoded),
                     "n_observations": int(len(decoded))})
    out = pd.DataFrame(rows)
    if out.empty: raise RuntimeError("No student-test trips with reference routes found")
    out.to_csv(ROOT / "results" / "decoder_comparison.csv", index=False)
    summary = out.select_dtypes(include="number").mean().to_dict()
    pd.DataFrame([summary]).to_csv(ROOT / "results" / "decoder_comparison_summary.csv", index=False)
    print(out.to_string(index=False))
    print("\nMean metrics:", summary)

if __name__ == "__main__": main()
