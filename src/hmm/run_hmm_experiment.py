"""Classifier-only vs classifier + HMM/Viterbi, with decoder tuning (train trips only).

    python src/hmm/run_hmm_experiment.py [--quick] [--trials 40]

1. out-of-fold (trip-grouped 5-fold) classifier probabilities on the 80 train trips  -> decoder tuning data
2. final classifier fit on the 80 train trips -> probabilities on the 10 student-test trips
3. decoder parameters tuned (Optuna TPE) on the train out-of-fold probabilities ONLY
4. report on student-test: classifier-only vs classifier+HMM
Evaluator data is never loaded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
for sub in ("models", "features", "hmm"):
    sys.path.insert(0, str(ROOT / "src" / sub))
import common as C
import hmm as H
import pipeline as P
from viterbi import decode as simple_decode  # teammate's connectivity-only Viterbi
from zoo import HGBModel

PROC = ROOT / "output" / "processed"


def load_params():
    f = C.RESULTS / "optuna_best.json"
    if f.exists():
        return json.loads(f.read_text())["best_params"]
    print("!! optuna_best.json not found: using default HGB parameters")
    return {}


def oof_probs(train, feats, params, quick):
    cache = PROC / ("oof_probs_quick.npy" if quick else "oof_probs.npy")
    if cache.exists():
        return np.load(cache)
    X, y, trips = train[feats].to_numpy(np.float32), train.label.to_numpy(), train.trip_id.to_numpy()
    out = np.zeros(len(train))
    for k, (tr_i, va_i) in enumerate(C.grouped_folds(trips, 5)):
        sub, val = C.inner_val_split(trips[tr_i])
        m = HGBModel(**params).fit(X[tr_i[sub]], y[tr_i[sub]], X[tr_i[val]], y[tr_i[val]])
        out[va_i] = m.predict_proba(X[va_i])
        print(f"  OOF fold {k} done")
    PROC.mkdir(parents=True, exist_ok=True); np.save(cache, out)
    return out


def trip_dir(partition, tid):
    return ROOT / "dataset" / partition / f"{int(tid):08d}"


def build_trellis(partition, tid, rows, prob):
    track, nodes, arcs, _ = P.read_trip(trip_dir(partition, tid))
    track, _ = P.clean_track(track)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    Pm, T = P.project(track[:, :2], lon0, lat0), track[:, 2]
    N = P.project(nodes, lon0, lat0)
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    g = H.TripGraph(arcs, np.linalg.norm(B - A, axis=1))
    tr = H.trellis_from_table(g, Pm, T, rows, prob)
    truth = rows[rows.label == 1].drop_duplicates("obs_idx").set_index("obs_idx").arc_id
    return tid, g, tr, truth, rows, np.asarray(prob)


def make_trellises(df, probs, partition, n_jobs=6):
    jobs = []
    for tid, rows in df.groupby("trip_id", sort=True):
        jobs.append(delayed(build_trellis)(partition, tid, rows.reset_index(drop=True), probs[rows.index.to_numpy()]))
    return Parallel(n_jobs=n_jobs)(jobs)


def score(trellises, params=None):
    """params None -> classifier-only. Returns aggregate + per-trip metrics."""
    hit = tot = 0; strict = []; reach = []; rp = []; rr = []; per_trip = []
    for tid, g, tr, truth, rows, prob in trellises:
        if params is None:
            chosen = np.array(tr.classifier_only())
        elif params == "simple":  # teammate decoder: hand-set constants, not tuned
            chosen = rows.arc_id.to_numpy()[simple_decode(rows, prob)]
        else:
            _, chosen = tr.decode(**params); chosen = np.array(chosen)
        if len(chosen) == 0:
            continue
        t_arc = truth.reindex(tr.obs_idx).to_numpy()
        ok = (chosen == np.nan_to_num(t_arc, nan=-1)) & ~np.isnan(t_arc)
        hit += int(ok.sum()); tot += len(chosen)
        s = H.strict_continuity(chosen, g); strict.append(s)
        reach.append(tr.reachable_fraction(chosen))
        ref, pred = set(truth.to_numpy().tolist()), set(chosen.tolist())
        p = len(ref & pred) / max(len(pred), 1); r = len(ref & pred) / max(len(ref), 1)
        rp.append(p); rr.append(r)
        per_trip.append({"trip_id": tid, "n_obs": len(chosen), "top1": float(ok.mean()), "strict_continuity": s,
                         "route_precision": p, "route_recall": r})
    pp, rrm = float(np.mean(rp)), float(np.mean(rr))
    return {"top1": hit / tot, "trip_top1_mean": float(np.mean([x["top1"] for x in per_trip])),
            "strict_continuity": float(np.nanmean(strict)), "reachable_continuity": float(np.nanmean(reach)),
            "route_precision": pp, "route_recall": rrm,
            "route_f1": 2 * pp * rrm / max(pp + rrm, 1e-9), "n_obs": tot}, pd.DataFrame(per_trip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    df = C.load_table(C.TABLES)
    feats = C.feature_columns(df)
    train = df[df.partition == "train"].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)
    if a.quick:
        train = train[train.trip_id.isin(sorted(train.trip_id.unique())[:15])].reset_index(drop=True)
        test = test[test.trip_id.isin(sorted(test.trip_id.unique())[:4])].reset_index(drop=True)
    params = load_params()
    t0 = time.time()
    print("1/4 out-of-fold probabilities on train trips"); p_oof = oof_probs(train, feats, params, a.quick)
    print("2/4 final classifier -> student-test probabilities")
    X, y, trips = train[feats].to_numpy(np.float32), train.label.to_numpy(), train.trip_id.to_numpy()
    import joblib
    shipped = joblib.load(ROOT / "artifacts" / "hist_gb_tuned.joblib")  # the model the API serves (train rows only)
    assert shipped["features"] == feats, "feature mismatch between table and shipped model"
    p_te = shipped["model"].predict_proba(test[feats].to_numpy(np.float32))[:, 1]
    print("3/4 building HMM trellises"); tr_tr = make_trellises(train, p_oof, "train"); tr_te = make_trellises(test, p_te, "student_test")
    print(f"    built in {time.time() - t0:.0f}s")
    base_tr, _ = score(tr_tr); base_te, base_pt = score(tr_te)
    print("classifier-only  train(OOF):", {k: round(v, 4) for k, v in base_tr.items()})

    def objective(trial):
        prm = dict(alpha=trial.suggest_float("alpha", 0.3, 3.0, log=True), beta=trial.suggest_float("beta", 5, 150, log=True),
                   p_unreach=trial.suggest_float("p_unreach", 1, 40), p_back=trial.suggest_float("p_back", 1, 40),
                   p_uturn=trial.suggest_float("p_uturn", 0, 20))
        return score(tr_tr, prm)[0]["top1"]
    print("4/4 tuning decoder on train OOF probabilities (TPE,", a.trials, "trials)")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.enqueue_trial(H.DEFAULT_PARAMS)
    study.optimize(objective, n_trials=a.trials)
    best = {k: float(v) for k, v in study.best_params.items()}
    default_tr, _ = score(tr_tr, H.DEFAULT_PARAMS); best_tr, _ = score(tr_tr, best)
    default_te, _ = score(tr_te, H.DEFAULT_PARAMS); best_te, hmm_pt = score(tr_te, best)
    simple_tr, _ = score(tr_tr, "simple"); simple_te, _ = score(tr_te, "simple")
    rows = [("classifier_only", "train_oof", base_tr), ("classifier_only", "student_test", base_te),
            ("viterbi_simple_teammate", "train_oof", simple_tr), ("viterbi_simple_teammate", "student_test", simple_te),
            ("hmm_default_params", "train_oof", default_tr), ("hmm_default_params", "student_test", default_te),
            ("hmm_tuned_params", "train_oof", best_tr), ("hmm_tuned_params", "student_test", best_te)]
    res = pd.DataFrame([{"method": m, "partition": p, **v} for m, p, v in rows])
    suffix = "_quick" if a.quick else ""
    C.RESULTS.mkdir(exist_ok=True)
    res.to_csv(C.RESULTS / f"hmm_results{suffix}.csv", index=False)
    study.trials_dataframe().to_csv(C.RESULTS / f"hmm_trials{suffix}.csv", index=False)
    pt = base_pt.merge(hmm_pt, on="trip_id", suffixes=("_classifier", "_hmm"))
    pt.to_csv(C.RESULTS / f"hmm_per_trip_student_test{suffix}.csv", index=False)
    C.dump_json({"params": best, "tuned_on": "train out-of-fold probabilities", "gap_s": H.GAP_S, "reach_m": H.REACH_M,
                 "back_tol_m": H.BACK_TOL_M, "n_trials": a.trials, "seed": C.SEED}, C.RESULTS / f"hmm_params{suffix}.json")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False)); print("best params", best)


if __name__ == "__main__":
    main()
