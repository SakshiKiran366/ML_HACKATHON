"""Collect the inference artifacts the API needs into artifacts/ (run after tuning + HMM experiment).

    artifacts/hist_gb_tuned.joblib   classifier + feature list + tuned params   (written by optuna_hgb.py)
    artifacts/hmm_config.json        decoder parameters + constants             (written here)
    artifacts/model_card.json        what the model is and where the evidence is (written here)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    hp = json.loads((C.RESULTS / "hmm_params.json").read_text())
    C.dump_json({"params": hp["params"], "gap_s": hp["gap_s"], "reach_m": hp["reach_m"], "back_tol_m": hp["back_tol_m"],
                 "tuned_on": hp["tuned_on"], "seed": hp["seed"]}, C.ARTIFACTS / "hmm_config.json")
    best = json.loads((C.RESULTS / "optuna_best.json").read_text())
    s = pd.read_csv(C.RESULTS / "experiments_summary.csv")
    cmp = pd.read_csv(C.RESULTS / "tuned_vs_untuned.csv")
    C.dump_json({
        "model": "HistGradientBoostingClassifier", "task": "P(candidate arc is the reference arc | GPS obs)",
        "features": json.loads((C.RESULTS / "run_meta.json").read_text())["features"],
        "hyperparameters": best["best_params"], "tuning": {"method": "Optuna TPE", "trials": best["n_trials"],
        "objective": "grouped-CV Top-1 accuracy on 80 train trips", "best_cv_top1": best["best_cv_top1"]},
        "selected_because": "best CV Top-1 and log loss among five models (results/experiments_summary.csv)",
        "model_comparison": s[["model", "cv_top1_mean", "test_top1", "test_log_loss"]].round(4).to_dict("records"),
        "tuned_vs_untuned": cmp[["model", "cv_top1_mean", "test_top1", "test_log_loss"]].round(4).to_dict("records"),
        "decoder": "HMM/Viterbi, see src/hmm/hmm.py", "evaluator_data_used": False,
    }, C.ARTIFACTS / "model_card.json")
    print("artifacts:", sorted(p.name for p in C.ARTIFACTS.iterdir()))


if __name__ == "__main__":
    main()
