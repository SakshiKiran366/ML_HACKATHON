"""Reusable reconstruction pipeline: raw trip files -> candidates -> classifier -> HMM/Viterbi.

This is the function the FastAPI backend calls. It reads ONLY .track/.nodes/.arcs (never .route), so
ground truth cannot enter the prediction path. Scoring against reference data is a separate step.

    from predict import load_bundle, reconstruct_trip
    bundle = load_bundle()
    result = reconstruct_trip(Path("dataset/student_test/00000052"), bundle)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("features", "hmm"):
    sys.path.insert(0, str(ROOT / "src" / sub))
import hmm as H
import pipeline as P


def load_bundle(artifacts: Path = ROOT / "artifacts") -> dict:
    m = joblib.load(artifacts / "hist_gb_tuned.joblib")
    cfg = json.loads((artifacts / "hmm_config.json").read_text())
    return {"model": m["model"], "features": m["features"], "hmm": cfg["params"], "config": cfg}


def reconstruct_trip(trip_dir: Path, bundle: dict) -> dict:
    t0 = time.perf_counter()
    tid = trip_dir.name
    rd = lambda ext, **kw: pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, **kw).to_numpy()
    track, nodes, arcs = rd("track").astype(float), rd("nodes").astype(float), rd("arcs").astype(int)  # NO .route
    df, st = P.build_trip(track, nodes, arcs, route=None, trip_id=int(tid), partition="inference", return_geom=True)
    t1 = time.perf_counter()
    prob = bundle["model"].predict_proba(df[bundle["features"]].to_numpy(np.float32))[:, 1]
    t2 = time.perf_counter()
    g = st["geom"]
    graph = H.TripGraph(arcs, g["L"])
    tr = H.trellis_from_table(graph, g["P"], g["T"], df, prob)
    _, hmm_arcs = tr.decode(**bundle["hmm"])
    clf_arcs = tr.classifier_only()
    t3 = time.perf_counter()
    clean, _ = P.clean_track(track)

    def polyline(chosen):
        seq = [int(a) for i, a in enumerate(chosen) if i == 0 or a != chosen[i - 1]]
        pts = []
        for a in seq:
            for n in arcs[a]:
                c = [float(nodes[n, 0]), float(nodes[n, 1])]
                if not pts or pts[-1] != c:
                    pts.append(c)
        return pts

    used = np.unique(df["arc_id"].to_numpy())
    return {
        "trip_id": tid, "obs_idx": tr.obs_idx, "classifier_arc": [int(a) for a in clf_arcs],
        "hmm_arc": [int(a) for a in hmm_arcs],
        "observations": clean[:, :2].tolist(),
        "predicted_route": polyline(hmm_arcs), "classifier_route": polyline(clf_arcs),
        "road_network": [[nodes[arcs[a, 0]].tolist(), nodes[arcs[a, 1]].tolist()] for a in used],
        "strict_continuity_hmm": H.strict_continuity(hmm_arcs, graph),
        "strict_continuity_classifier": H.strict_continuity(clf_arcs, graph),
        "timings_ms": {"candidates_features": 1000 * (t1 - t0), "classifier": 1000 * (t2 - t1),
                       "hmm": 1000 * (t3 - t2), "total": 1000 * (t3 - t0)},
    }
