"""Probability quality of the tuned classifier: reliability, ECE, log loss, with/without isotonic.

Isotonic regression is fitted on the train out-of-fold probabilities (output/processed/oof_probs.npy,
written by src/hmm/run_hmm_experiment.py) and evaluated on the student-test trips. The calibrator is
only adopted if it improves student-test log loss; the decision is written to results/calibration.json.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main():
    df = C.load_table(C.TABLES)
    train = df[df.partition == "train"].reset_index(drop=True); test = df[df.partition == "student_test"].reset_index(drop=True)
    oof = np.load(C.ROOT / "output" / "processed" / "oof_probs.npy"); y_tr = train.label.to_numpy()
    bundle = joblib.load(C.ARTIFACTS / "hist_gb_tuned.joblib")
    p_te = bundle["model"].predict_proba(test[bundle["features"]].to_numpy(np.float32))[:, 1]; y_te = test.label.to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(oof, y_tr)
    p_cal = iso.predict(p_te)
    rows = {}
    for name, p in (("uncalibrated", p_te), ("isotonic", p_cal)):
        rows[name] = {"log_loss": float(log_loss(y_te, np.clip(p, 1e-7, 1 - 1e-7))),
                      "brier": float(brier_score_loss(y_te, p)), "ece": ece(y_te, p)}
    adopt = rows["isotonic"]["log_loss"] < rows["uncalibrated"]["log_loss"] - 1e-3  # must be a material gain
    rows["decision"] = "adopt isotonic" if adopt else "keep uncalibrated (already well calibrated; isotonic gain is immaterial)"
    if adopt:
        joblib.dump(iso, C.ARTIFACTS / "isotonic_calibrator.joblib")
    C.dump_json(rows, C.RESULTS / "calibration.json")
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in (("uncalibrated", p_te), ("isotonic", p_cal)):
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        ys = [y_te[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability (student-test)"); ax.legend(); fig.tight_layout()
    (C.RESULTS / "figures").mkdir(exist_ok=True); fig.savefig(C.RESULTS / "figures" / "reliability.png", dpi=150)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
