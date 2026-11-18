"""Train + compare the five models with trip-grouped CV and student-test evaluation.

    python src/models/run_experiments.py --table data/candidates.parquet
    python src/models/run_experiments.py --synthetic        # harness smoke test ONLY

Selection data: 80 training trips (grouped CV). Student-test trips: development comparison only.
Evaluator data is never loaded.
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common as C
from zoo import registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", default="logreg,knn,random_forest,xgboost,mlp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(C.RESULTS))
    args = ap.parse_args()

    if args.synthetic:
        print("!" * 70, "\n!! SYNTHETIC DATA: harness test only. Do NOT report these numbers.\n" + "!" * 70)
        df, tag = C.make_synthetic(), "SYNTHETIC"
    else:
        df, tag = C.load_table(args.table), "REAL"
    feats = C.feature_columns(df)
    train = df[df.partition == "train"].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)
    print(f"[{tag}] features={feats}\n train trips={train.trip_id.nunique()} rows={len(train)} | "
          f"student_test trips={test.trip_id.nunique()} rows={len(test)}")
    assert set(train.trip_id).isdisjoint(set(test.trip_id)), "trip appears in two partitions!"

    X_tr, y_tr = train[feats].to_numpy(np.float32), train["label"].to_numpy()
    X_te, y_te = test[feats].to_numpy(np.float32), test["label"].to_numpy()
    reg = registry()
    out = C.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_rows, summary_rows, histories, cms = [], [], {}, {}
    folds = list(C.grouped_folds(train.trip_id.to_numpy(), args.folds))

    for name in args.models.split(","):
        family, factory, cfg = reg[name]
        print(f"\n=== {name} ({family})\n    {cfg}")
        fold_metrics, fold_fit, fold_inf = [], [], []
        for k, (tr_i, va_i) in enumerate(folds):
            sub, val = C.inner_val_split(train.trip_id.to_numpy()[tr_i])
            tr_i_fit, tr_i_es = tr_i[sub], tr_i[val]
            m = factory()
            with C.Timer() as t_fit:
                m.fit(X_tr[tr_i_fit], y_tr[tr_i_fit], X_tr[tr_i_es], y_tr[tr_i_es])
            with C.Timer() as t_inf:
                p = m.predict_proba(X_tr[va_i])
            met = C.evaluate(train.iloc[va_i], p)
            fold_metrics.append(met); fold_fit.append(t_fit.seconds); fold_inf.append(t_inf.seconds)
            fold_rows.append({"model": name, "fold": k, "fit_s": t_fit.seconds,
                              "infer_s": t_inf.seconds, **{a: b for a, b in met.items()
                                                           if not isinstance(b, list)}})
            print(f"    fold {k}: top1={met['top1_accuracy']:.4f} f1_top1={met['macro_f1_top1']:.4f} "
                  f"logloss={met['log_loss']:.4f} fit={t_fit.seconds:.1f}s")
        # final fit on all 80 training trips, scored on the 10 student-test trips
        sub, val = C.inner_val_split(train.trip_id.to_numpy())
        m = factory()
        with C.Timer() as t_fit:
            m.fit(X_tr[sub], y_tr[sub], X_tr[val], y_tr[val])
        with C.Timer() as t_inf:
            p_te = m.predict_proba(X_te)
        te = C.evaluate(test, p_te)
        p_tr = m.predict_proba(X_tr)  # training-set score, only to diagnose over/underfitting
        trm = C.evaluate(train, p_tr)
        if m.history is not None:
            histories[name] = m.history
        cms[name] = {"student_test": te["confusion_matrix_thr0.5"]}
        cv = pd.DataFrame(fold_metrics)
        summary_rows.append({
            "model": name, "family": family, "config": cfg, "seed": C.SEED, "data": tag,
            "cv_top1_mean": cv.top1_accuracy.mean(), "cv_top1_std": cv.top1_accuracy.std(ddof=0),
            "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(ddof=0),
            "cv_macro_f1_top1_mean": cv.macro_f1_top1.mean(), "cv_macro_f1_top1_std": cv.macro_f1_top1.std(ddof=0),
            "cv_log_loss_mean": cv.log_loss.mean(), "cv_route_continuity_mean": cv.route_continuity.mean(),
            "test_top1": te["top1_accuracy"], "test_macro_f1": te["macro_f1"],
            "test_macro_f1_top1": te["macro_f1_top1"], "test_precision_macro": te["precision_macro"],
            "test_recall_macro": te["recall_macro"], "test_log_loss": te["log_loss"],
            "test_route_continuity": te["route_continuity"], "test_candidate_miss_rate": te["candidate_miss_rate"],
            "test_trip_top1_mean": te["trip_top1_mean"], "test_trip_top1_std": te["trip_top1_std"],
            "train_top1": trm["top1_accuracy"], "train_log_loss": trm["log_loss"],
            "fit_s_final": t_fit.seconds, "infer_s_student_test": t_inf.seconds,
            "infer_us_per_row": 1e6 * t_inf.seconds / len(X_te)})
        print(f"    FINAL student_test: top1={te['top1_accuracy']:.4f} macroF1={te['macro_f1']:.4f} "
              f"logloss={te['log_loss']:.4f} cont={te['route_continuity']:.4f} "
              f"(train top1 {trm['top1_accuracy']:.4f})")

    pd.DataFrame(fold_rows).to_csv(out / "experiments_folds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "experiments_summary.csv", index=False)
    C.dump_json(histories, out / "training_histories.json")
    C.dump_json(cms, out / "confusion_matrices.json")
    C.dump_json({"time": datetime.now().isoformat(timespec="seconds"), "data": tag, "seed": C.SEED,
                 "features": feats, "folds": args.folds, "python": sys.version.split()[0],
                 "platform": platform.platform(), "models": args.models.split(",")},
                out / "run_meta.json")
    print(f"\nSaved results to {out}  [{tag}]")


if __name__ == "__main__":
    main()
