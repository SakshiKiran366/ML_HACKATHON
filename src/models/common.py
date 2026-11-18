"""Shared harness for PathSnap model experiments.

Data contract (one row per GPS-observation x candidate-arc pair):
    trip_id, obs_idx, arc_id, arc_from, arc_to, label, partition, <feature columns...>
`partition` is "train" or "student_test". Evaluator data is never loaded here.
Every column that is not in META_COLS is treated as a model feature.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold

META_COLS = ["trip_id", "obs_idx", "arc_id", "arc_from", "arc_to", "label", "partition"]
SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


# --------------------------------------------------------------------------- data
def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path) if path.suffix in (".csv", ".gz") else pd.read_parquet(path)
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"table missing required columns: {missing}")
    bad = set(df["partition"].unique()) - {"train", "student_test"}
    if bad:
        raise ValueError(f"unexpected partitions {bad}; evaluator data must never enter this pipeline")
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def make_synthetic(n_train=80, n_test=10, k=8, seed=SEED) -> pd.DataFrame:
    """SYNTHETIC stand-in table, only to exercise the harness. Never report its numbers."""
    rng = np.random.default_rng(seed)
    rows = []
    trip_ids = list(range(n_train + n_test))
    for t in trip_ids:
        n_obs = int(rng.integers(150, 700))
        arc_cur = 0.0
        for o in range(n_obs):
            arc_cur += rng.uniform(0.0, 0.12)
            true_arc = int(arc_cur)
            miss = rng.random() < 0.04
            cands = {true_arc + d for d in (-2, -1, 1, 2, 3)} - {true_arc}
            cands = [c for c in cands if c >= 0]
            while len(cands) < k - 1:
                cands.append(int(rng.integers(0, true_arc + 10)))
            cands = list(dict.fromkeys(cands))[: k - 1]
            if not miss:
                cands.append(true_arc)
            for a in cands:
                is_true = a == true_arc and not miss
                dist = abs(rng.normal(6 if is_true else 25, 8 if is_true else 18))
                rows.append((t, o, a, a, a + 1, int(is_true),
                             "train" if t < n_train else "student_test",
                             dist, abs(rng.normal(12 if is_true else 60, 25)),
                             rng.normal(0 if is_true else 0.6, 0.5),
                             rng.uniform(5, 60), rng.uniform(40, 300), rng.uniform(0, 1),
                             dist + rng.normal(0, 5), float(rng.random() < (0.9 if is_true else 0.3))))
    cols = META_COLS + ["dist_m", "heading_diff_deg", "heading_score", "speed_mps",
                        "arc_len_m", "proj_frac", "prev_dist_m", "connected_prev"]
    df = pd.DataFrame(rows, columns=cols)
    df["dist_rank"] = df.groupby(["trip_id", "obs_idx"])["dist_m"].rank(method="first")
    df["dist_gap"] = df["dist_m"] - df.groupby(["trip_id", "obs_idx"])["dist_m"].transform("min")
    return df


# ------------------------------------------------------------------------ splits
def grouped_folds(trip_ids: np.ndarray, n_splits: int = 5):
    """GroupKFold by trip (no trip split across folds). sklearn balances fold sizes."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(trip_ids))
    yield from gkf.split(dummy, dummy, groups=trip_ids)


def inner_val_split(trip_ids: np.ndarray, frac=0.12, seed=SEED):
    """Hold out whole trips for early stopping (never split a trip)."""
    uniq = np.unique(trip_ids)
    rng = np.random.default_rng(seed)
    val_trips = set(rng.choice(uniq, size=max(1, int(len(uniq) * frac)), replace=False))
    is_val = np.array([t in val_trips for t in trip_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]


# ----------------------------------------------------------------------- metrics
def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    """df: rows with META_COLS; p: P(label=1) per row."""
    d = df[["trip_id", "obs_idx", "arc_id", "arc_from", "arc_to", "label"]].copy().reset_index(drop=True)
    d["p"] = np.asarray(p, dtype=float)
    y = d["label"].to_numpy()
    pc = np.clip(d["p"].to_numpy(), 1e-7, 1 - 1e-7)
    yhat = (pc >= 0.5).astype(int)
    out = {
        "n_rows": len(d),
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "confusion_matrix_thr0.5": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }
    # observation-level: pick highest-scoring candidate per GPS observation
    key = ["trip_id", "obs_idx"]
    idx = d.groupby(key)["p"].idxmax()
    pred = d.loc[idx].sort_values(key)
    present = d.groupby(key)["label"].max().loc[pred.set_index(key).index].to_numpy()
    top1_hit = pred["label"].to_numpy()
    out["n_obs"] = len(pred)
    out["top1_accuracy"] = float(top1_hit.mean())  # misses count as wrong
    out["top1_given_present"] = float(top1_hit.sum() / max(present.sum(), 1))
    out["candidate_miss_rate"] = float(1.0 - present.mean())
    # top-1 as classification: chosen candidate = 1, all others = 0
    ytop = np.zeros(len(d), dtype=int)
    ytop[idx.to_numpy()] = 1
    out["macro_f1_top1"] = f1_score(y, ytop, average="macro", zero_division=0)
    out["precision_top1"] = precision_score(y, ytop, zero_division=0)
    out["recall_top1"] = recall_score(y, ytop, zero_division=0)
    # route continuity: consecutive observations -> same arc or head-to-tail connected arcs
    pr = pred.reset_index(drop=True)
    same_trip = pr["trip_id"].to_numpy()[1:] == pr["trip_id"].to_numpy()[:-1]
    same_arc = pr["arc_id"].to_numpy()[1:] == pr["arc_id"].to_numpy()[:-1]
    linked = pr["arc_to"].to_numpy()[:-1] == pr["arc_from"].to_numpy()[1:]
    ok = (same_arc | linked)[same_trip]
    out["route_continuity"] = float(ok.mean()) if len(ok) else float("nan")
    # sequence-level: per-trip top-1 mean/std
    per_trip = pr.groupby("trip_id")["label"].mean()
    out["trip_top1_mean"] = float(per_trip.mean())
    out["trip_top1_std"] = float(per_trip.std(ddof=0))
    return out


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
