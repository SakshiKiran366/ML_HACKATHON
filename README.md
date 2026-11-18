# PathSnap: Distributed Map-Matching & Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon. Noisy 1 Hz GPS points are matched to a road
network: a classifier scores candidate road segments (arcs) for every GPS point, then an HMM with
Viterbi decoding reconstructs a coherent route. Served by a FastAPI backend with a separate UI.

## Deployment roles (two laptops)

| Laptop | Role | Runs |
|---|---|---|
| **Laptop 1** | Model + API | trained classifier, HMM/Viterbi, road-network data, evaluation against server-side reference routes, FastAPI logs shown to evaluator |
| Laptop 2 | UI | lists the ten evaluation trips, sends the trip ID to the API, shows route and results |

Ground-truth labels stay on Laptop 1 and never reach the model or the UI.

## Repository layout

```
dataset/            raw train + student_test trips (evaluator data is NOT in this repo)
src/0*.py           dataset exploration scripts (inventory, validation, trajectories, network, routes, maps)
src/features/       pipeline.py (candidates, labels, features), build_table.py
src/models/         common.py (harness + metrics), zoo.py (5 models), run_experiments.py
src/tuning/         optuna_hgb.py (Bayesian optimisation)
src/evaluation/     plots.py (comparison plots from recorded results)
src/hmm/            (planned) HMM + Viterbi
api/  ui/           (planned) FastAPI backend and evaluator UI
tables/             candidate tables (parquet, one file per partition)
results/            experiment logs/CSVs, candidate diagnostics, figures
artifacts/          final model and configs
output/             exploration reports (teammates' EDA outputs)
```

## Pipeline

1. **Candidate generation** (`src/features/pipeline.py`): trips are projected to local metres (they lie in different cities). Every arc within 60 m of a GPS point is a candidate (nearest 16 kept) using a spatial index; arcs longer than 150 m are brute-forced.
2. **Labels**: a monotone Viterbi alignment of GPS points to the reference route (stay or move forward along the route, distance + heading cost, free forward jumps across GPS gaps). Used only for training labels and server-side scoring. Observations more than 60 m from their reference arc (breaks in the supplied route, e.g. train trip 2) are dropped and counted in `results/candidate_diagnostics.csv`.
3. **Features** (all computable without `.route`, so the API can reuse them): distance, position along arc, arc length, distance rank/gap, GPS speed and heading agreement, time gaps, distance of neighbouring GPS points to the same arc, adjacency to the neighbouring points' nearest arcs, node degrees.

```bash
pip install -r requirements.txt
python src/features/build_table.py                 # tables/candidates_*.parquet + diagnostics
python src/models/run_experiments.py               # five models, grouped CV + student-test
python src/evaluation/plots.py                     # figures from recorded results
python src/tuning/optuna_hgb.py --trials 30        # Bayesian optimisation of the selected model
```

## Model experiments (Models and Training Guidelines)

Five models, one per family named in the guidelines: Logistic Regression (linear), distance-weighted
KNN (instance-based), Decision Tree (tree-based), Random Forest (ensemble), HistGradientBoosting
(boosting). One model is selected as final; models are not stacked or blended.

- Training and CV use the 80 train trips with trip-grouped 5-fold CV. Student-test trips are used only for development comparison.
- Primary selection metric: observation-level Top-1 accuracy (grouped CV), log loss as tie-breaker.
- Also logged: macro F1, precision, recall, confusion matrix, candidate miss rate, route continuity, training and inference time, seed, config.

Table contract (one row per GPS observation x candidate arc):
`trip_id, obs_idx, arc_id, arc_from, arc_to, label, partition (train|student_test), <feature columns>`.

## Integrity

Evaluator data is never used for training, feature selection, model selection or tuning. No hardcoded
outputs. AI assistance was used in building this code and is disclosed per event policy.
