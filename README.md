# PathSnap: Distributed Map-Matching & Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon. Noisy 1 Hz GPS points are matched to a road
network: a classifier scores candidate road segments (arcs) for every GPS point, then an HMM with
Viterbi decoding reconstructs a coherent route. Served by a FastAPI backend with a separate UI.

## Deployment roles (two laptops)

| Laptop | Role | Runs |
|---|---|---|
| **Laptop 1** | Model + API | trained classifier, HMM/Viterbi, road-network data, evaluation against server-side reference routes, FastAPI logs shown to evaluator |
| **Laptop 2** | UI | lists the ten evaluation trips, sends the trip ID to the API, shows route and results |

Ground-truth labels stay on Laptop 1 and never reach the model or the UI.

## Repository layout

```
dataset/            raw train + student_test trips (evaluator data is NOT in this repo)
src/0[1-7]*.py      dataset exploration scripts (inventory, validation, trajectories, network, routes, maps)
src/08, src/09      SUPERSEDED (label leak), see docs/FEATURE_PIPELINES.md
src/features/       pipeline.py (candidates, labels, features), build_table.py
src/models/         common.py (harness + metrics), zoo.py (5 models), run_experiments.py
src/tuning/         optuna_hgb.py (Bayesian optimisation)
src/evaluation/     plots.py (comparison plots from recorded results)
docs/               FEATURE_PIPELINES.md (canonical pipeline + leak finding), API_CONTRACT.md (backend <-> UI)
src/hmm/            (planned) HMM + Viterbi
frontend/           React + Vite + Leaflet evaluation UI
tables/             candidate tables (parquet, one file per partition)
results/            experiment logs/CSVs, candidate diagnostics, figures
artifacts/          final model and configs
output/             exploration reports (teammates' EDA outputs)
```

## Frontend

The browser application lives in [`frontend/`](frontend/). See [`frontend/README.md`](frontend/README.md)
for installation, API configuration, the provisional FastAPI contract, and deployment instructions.

## Pipeline

1. **Candidate generation** (`src/features/pipeline.py`): trips are projected to local metres. Every arc within 60 m of a GPS point is a candidate, with the nearest 16 kept using a spatial index.
2. **Labels**: a monotone Viterbi alignment of GPS points to the reference route is used only for training labels and server-side scoring. Ground truth remains server-side.
3. **Features**: distance, position along arc, arc length, distance rank/gap, GPS speed and heading agreement, time gaps, neighbouring GPS distances, adjacency, and node degrees.

```bash
pip install -r requirements.txt
python src/features/build_table.py
python src/models/run_experiments.py
python src/evaluation/plots.py
python src/features/feature_importance.py     # leak-free permutation importance
python src/tuning/optuna_hgb.py --trials 30
```

## Model experiments

Models include Logistic Regression, distance-weighted KNN, Decision Tree, Random Forest, and HistGradientBoosting. One model is selected as final; models are not stacked or blended.

- Training and CV use the 80 train trips with trip-grouped 5-fold CV.
- Student-test trips are used only for development comparison.
- Primary selection metric is observation-level Top-1 accuracy, with log loss as tie-breaker.
- Also logged: macro F1, precision, recall, confusion matrix, candidate miss rate, route continuity, training and inference time, seed, and config.

Table contract (one row per GPS observation x candidate arc):
`trip_id, obs_idx, arc_id, arc_from, arc_to, label, partition (train|student_test), <feature columns>`.

## Integrity

Evaluator data is never used for training, feature selection, model selection or tuning. No hardcoded
outputs. AI assistance was used in building this code and is disclosed per event policy.
