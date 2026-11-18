"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (XGBoost).

Objective : maximise mean observation-level Top-1 accuracy over trip-grouped 5-fold CV on the 80
            training trips (log loss is recorded per trial as a secondary check).
Student-test and evaluator trips are NEVER used inside the search.
After the search: refit best config on all training trips, evaluate on student-test, and compare with
the untuned model under identical data/metrics.

    python src/tuning/optuna_xgb.py --table data/candidates.parquet --trials 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import common as C
from zoo import XGBModel

SPACE = {  # documented search space (also written to results/optuna_search_space.json)
    "learning_rate": "log-uniform [0.01, 0.3]", "max_depth": "int [3, 10]",
    "min_child_weight": "log-uniform [1, 20]", "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.5, 1.0]", "reg_lambda": "log-uniform [1e-2, 20]",
    "reg_alpha": "log-uniform [1e-3, 5]", "gamma": "uniform [0, 3]"}


def fit_eval(params, Xtr, ytr, tr_df, Xev, ev_df):
    sub, val = C.inner_val_split(tr_df.trip_id.to_numpy())
    m = XGBModel(**params)
    with C.Timer() as tf:
        m.fit(Xtr[sub], ytr[sub], Xtr[val], ytr[val])
    with C.Timer() as ti:
        p = m.predict_proba(Xev)
    return m, C.evaluate(ev_df, p), tf.seconds, ti.seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--trials", type=int, default=60); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(C.RESULTS))
    a = ap.parse_args()
    df = C.make_synthetic() if a.synthetic else C.load_table(a.table)
    tag = "SYNTHETIC" if a.synthetic else "REAL"
    if a.synthetic:
        print("!! SYNTHETIC DATA: harness test only")
    feats = C.feature_columns(df)
    train = df[df.partition == "train"].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)
    X, y = train[feats].to_numpy(np.float32), train.label.to_numpy()
    folds = list(C.grouped_folds(train.trip_id.to_numpy(), a.folds))

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            gamma=trial.suggest_float("gamma", 0, 3))
        accs, lls = [], []
        for tr_i, va_i in folds:
            _, met, _, _ = fit_eval(params, X[tr_i], y[tr_i], train.iloc[tr_i].reset_index(drop=True),
                                    X[va_i], train.iloc[va_i])
            accs.append(met["top1_accuracy"]); lls.append(met["log_loss"])
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(accs)))
        return float(np.mean(accs))

    sampler = optuna.samplers.TPESampler(seed=C.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_top1")
    with C.Timer() as t_search:
        study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    C.dump_json(SPACE, out / "optuna_search_space.json")

    best = study.best_params
    print("best", study.best_value, best)
    # tuned vs untuned: identical CV folds + student-test
    rows = {}
    for label, params in (("untuned", {}), ("tuned", best)):
        accs, f1s, lls, ft, inf = [], [], [], [], []
        for tr_i, va_i in folds:
            _, met, tf, ti = fit_eval(params, X[tr_i], y[tr_i], train.iloc[tr_i].reset_index(drop=True),
                                      X[va_i], train.iloc[va_i])
            accs.append(met["top1_accuracy"]); f1s.append(met["macro_f1"]); lls.append(met["log_loss"])
        m, te, tf, ti = fit_eval(params, X, y, train, test[feats].to_numpy(np.float32), test)
        rows[label] = {"model": f"xgboost_{label}", "params": params or "defaults", "data": tag,
                       "cv_top1_mean": np.mean(accs), "cv_macro_f1_mean": np.mean(f1s),
                       "cv_log_loss_mean": np.mean(lls), "test_top1": te["top1_accuracy"],
                       "test_macro_f1": te["macro_f1"], "test_log_loss": te["log_loss"],
                       "test_route_continuity": te["route_continuity"], "fit_s": tf,
                       "infer_us_per_row": 1e6 * ti / len(test)}
        if label == "tuned":
            C.ARTIFACTS.mkdir(exist_ok=True)
            m.m.save_model(str(C.ARTIFACTS / ("xgb_tuned_SYNTHETIC.json" if a.synthetic else "xgb_tuned.json")))
    cmp = pd.DataFrame(rows.values())
    cmp["search_trials"] = a.trials; cmp["search_seconds"] = t_search.seconds
    cmp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    C.dump_json({"best_params": best, "best_cv_top1": study.best_value, "n_trials": a.trials,
                 "sampler": "TPE", "seed": C.SEED, "search_seconds": t_search.seconds, "data": tag},
                out / "optuna_best.json")
    print(cmp.T)


if __name__ == "__main__":
    main()
