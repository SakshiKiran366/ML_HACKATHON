"""Bayesian hyperparameter optimisation (Optuna TPE) for HistGradientBoosting.

Run this on the model chosen from the five-model comparison (experiments_summary.csv).
Objective : maximise mean observation-level Top-1 accuracy over trip-grouped K-fold CV on the 80
            training trips. Boosting rounds are early-stopped on held-out TRIPS inside each fold.
Never used in the search: student-test trips (only for the final tuned-vs-untuned comparison)
and evaluator trips (never loaded).

    python src/tuning/optuna_hgb.py --trials 30 --folds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import common as C
from zoo import HGBModel

MAX_ITER = 300
SPACE = {"learning_rate": "log-uniform [0.03, 0.3]", "max_leaf_nodes": "int [15, 127]",
         "max_depth": "int [4, 16] or None", "min_samples_leaf": "log-int [10, 200]",
         "l2_regularization": "log-uniform [1e-3, 30]", "max_features": "uniform [0.5, 1.0]",
         "max_iter": f"fixed {MAX_ITER}, best round chosen on held-out trips"}


def fold_eval(params, X, y, trips, tr_i, ev_i, ev_df):
    """Fit on tr_i (early stopping on held-out trips of tr_i), score ev_i at the best round."""
    sub, val = C.inner_val_split(trips[tr_i])
    fit_i, es_i = tr_i[sub], tr_i[val]
    m = HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=False,
                                       random_state=C.SEED, **params).fit(X[fit_i], y[fit_i])
    va = [log_loss(y[es_i], p[:, 1], labels=[0, 1]) for p in m.staged_predict_proba(X[es_i])]
    best = int(np.argmin(va)) + 1
    for k, p in enumerate(m.staged_predict_proba(X[ev_i]), start=1):
        if k == best:
            return C.evaluate(ev_df, p[:, 1]), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(C.TABLES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(C.RESULTS))
    a = ap.parse_args()
    df = C.load_table(a.table)
    feats = C.feature_columns(df)
    train = df[df.partition == "train"].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)
    X, y = train[feats].to_numpy(np.float32), train.label.to_numpy()
    trips = train.trip_id.to_numpy()
    folds = list(C.grouped_folds(trips, a.folds))

    def objective(trial):
        depth = trial.suggest_int("max_depth", 4, 16)
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            max_depth=depth,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 30, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0))
        accs, lls, bests = [], [], []
        for tr_i, ev_i in folds:
            met, best = fold_eval(params, X, y, trips, tr_i, ev_i, train.iloc[ev_i])
            accs.append(met["top1_accuracy"]); lls.append(met["log_loss"]); bests.append(best)
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(accs)))
        trial.set_user_attr("mean_best_round", float(np.mean(bests)))
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                study_name="hgb_top1")
    with C.Timer() as t_search:
        study.optimize(objective, n_trials=a.trials)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    C.dump_json(SPACE, out / "optuna_search_space.json")
    best = study.best_params
    print("best CV top-1", study.best_value, best)

    # untuned vs tuned: same folds, same student-test trips, same metrics
    rows = []
    Xte = test[feats].to_numpy(np.float32)
    for label, params in (("untuned", {}), ("tuned", best)):
        accs, f1s, lls = [], [], []
        for tr_i, ev_i in folds:
            sub, val = C.inner_val_split(trips[tr_i])
            m = HGBModel(**params).fit(X[tr_i[sub]], y[tr_i[sub]], X[tr_i[val]], y[tr_i[val]])
            met = C.evaluate(train.iloc[ev_i], m.predict_proba(X[ev_i]))
            accs.append(met["top1_accuracy"]); f1s.append(met["macro_f1_top1"]); lls.append(met["log_loss"])
        sub, val = C.inner_val_split(trips)
        with C.Timer() as tf:
            m = HGBModel(**params).fit(X[sub], y[sub], X[val], y[val])
        with C.Timer() as ti:
            p = m.predict_proba(Xte)
        te = C.evaluate(test, p)
        rows.append({"model": f"hist_gb_{label}", "params": params or "defaults",
                     "cv_top1_mean": np.mean(accs), "cv_macro_f1_top1_mean": np.mean(f1s),
                     "cv_log_loss_mean": np.mean(lls), "test_top1": te["top1_accuracy"],
                     "test_macro_f1_top1": te["macro_f1_top1"], "test_macro_f1": te["macro_f1"],
                     "test_log_loss": te["log_loss"], "test_route_continuity": te["route_continuity"],
                     "fit_s": tf.seconds, "infer_us_per_row": 1e6 * ti.seconds / len(Xte)})
        if label == "tuned":
            import joblib
            C.ARTIFACTS.mkdir(exist_ok=True)
            joblib.dump({"model": m.m, "features": feats, "params": best}, C.ARTIFACTS / "hist_gb_tuned.joblib")
    cmp = pd.DataFrame(rows)
    cmp["search_trials"] = a.trials; cmp["search_seconds"] = t_search.seconds
    cmp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    C.dump_json({"best_params": best, "best_cv_top1": study.best_value, "n_trials": a.trials,
                 "sampler": "TPE", "seed": C.SEED, "search_seconds": t_search.seconds}, out / "optuna_best.json")
    print(cmp.T)


if __name__ == "__main__":
    main()
