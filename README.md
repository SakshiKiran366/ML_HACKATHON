# PathSnap: Distributed Map-Matching & Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon. Noisy 1 Hz GPS points are matched to a road
network: a classifier scores candidate road segments (arcs) for every GPS point, then an HMM with
Viterbi decoding reconstructs a coherent route. Served by a FastAPI backend with a separate UI.

## Deployment roles (two laptops)

| Laptop | Role | Runs |
|---|---|---|
| **Laptop 1 (this machine)** | Model + API | trained classifier, HMM/Viterbi, road-network data, evaluation against server-side reference routes, FastAPI logs shown to evaluator |
| Laptop 2 | UI | lists the ten evaluation trips, sends the trip ID to the API, shows route and results |

Ground-truth labels stay on Laptop 1 and never reach the model or the UI.

## Repository layout

```
data/               raw train + student_test trips (evaluator data is NOT in this repo)
src/models/         common.py (harness + metrics), zoo.py (5 models), run_experiments.py
src/tuning/         optuna_xgb.py (Bayesian optimisation)
src/evaluation/     plots.py (comparison plots from recorded results)
src/hmm/            (planned) HMM + Viterbi
api/  ui/           (planned) FastAPI backend and evaluator UI
results/            experiments_*.csv, optuna_trials.csv, figures/  (real runs only)
artifacts/          final model, calibrator, configs
```

## Model experiments (Models and Training Guidelines)

Five models, one per family: Logistic Regression (linear), distance-weighted KNN (instance-based),
Random Forest (bagging ensemble), XGBoost (boosting, GPU), MLP (neural network, torch).
Only one model is selected as final; models are not stacked or blended.

- Training and CV use the 80 train trips with trip-grouped 5-fold CV. Student-test trips are used only for development comparison.
- Primary selection metric: observation-level Top-1 accuracy (grouped CV), log loss as tie-breaker.
- Also logged: macro F1, precision, recall, confusion matrix, candidate miss rate, route continuity, training and inference time, seed, config.
- Tuning: Optuna TPE on grouped CV of the train trips only.

```bash
pip install -r requirements.txt
python src/models/run_experiments.py --table data/candidates.parquet
python src/tuning/optuna_xgb.py --table data/candidates.parquet --trials 60
python src/evaluation/plots.py
```

Input table contract (one row per GPS observation x candidate arc):
`trip_id, obs_idx, arc_id, arc_from, arc_to, label, partition (train|student_test), <feature columns>`.

`--synthetic` runs a built-in fake table to smoke-test the code. Its numbers are meaningless and must
never be reported.

## Integrity

Evaluator data is never used for training, feature selection, model selection or tuning. No hardcoded
outputs. AI assistance was used in building this code and is disclosed per event policy.
