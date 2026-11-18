# PathSnap: Technical Report (team review draft)

This report is generated from the experiment files in this repository. Please review it, make sure every team member understands the methods, and add screenshots from the actual two-laptop demonstration before submission. No evaluator-test data was available in this workspace.

## System and data
PathSnap matches noisy GPS observations to candidate directed road segments. The input files are whitespace-separated: `.track` contains longitude, latitude and seconds; `.nodes` contains longitude/latitude; `.arcs` contains directed node pairs; `.route` contains reference arc IDs. The checked-in development partitions contain 80 training trips and 10 student-test trips. The separate evaluator partition was not present and was not used.

The feature pipeline cleans non-finite and non-increasing timestamp rows, projects each trip to local metres, searches for nearby arcs, and calculates distance, direction, motion, time-gap, neighbouring-point and network-connectivity features. Training labels align GPS points monotonically to the supplied reference route. Observations more than 60 m from the aligned route segment are excluded as unreliable. The student-test labels are used only for development comparison.

## Model comparison
The project trained five distinct classifiers and used five-fold GroupKFold by trip. The primary selection metric was observation-level Top-1 accuracy; log loss was used as a probability-quality tie-breaker. Student-test results are reported as development comparison, not as a substitute for the reserved evaluator partition.

| Model | CV Top-1 | CV log loss | Student-test Top-1 | Student-test macro F1 | Student-test log loss |
|---|---:|---:|---:|---:|---:|
| logreg | 0.8425 | 0.1941 | 0.8404 | 0.9115 | 0.1817 |
| knn | 0.8301 | 0.1380 | 0.8291 | 0.9052 | 0.1189 |
| decision_tree | 0.8435 | 0.1736 | 0.8509 | 0.9173 | 0.1466 |
| random_forest | 0.8519 | 0.0909 | 0.8615 | 0.9232 | 0.0788 |
| hist_gb | 0.8539 | 0.0807 | 0.8582 | 0.9214 | 0.0711 |

The recorded comparison selects **hist_gb**: it has the highest mean grouped-CV Top-1 (0.8539) and the lowest CV log loss (0.0807) among these five models. The final API artifact is saved under `artifacts/final_model.joblib`.

## Bayesian optimisation

- hist_gb_untuned — CV Top-1 0.8538; student-test Top-1 0.8574; student-test log loss 0.0721.
- hist_gb_tuned — CV Top-1 0.8549; student-test Top-1 0.8586; student-test log loss 0.0690.
- TPE completed 10 trials. Best training-CV Top-1: 0.8548. Search details and trials are in `results/optuna_best.json`, `results/optuna_trials.csv`, and `results/optuna_search_space.json`.

## HMM and Viterbi

The hidden state is a candidate directed road arc for each GPS observation. Classifier probabilities supply emission scores. Viterbi transitions favour staying on an arc or moving head-to-tail to a connected arc; projected along-arc distance, expected travel distance and GPS gap duration adjust transition costs. The decoder returns one road segment per observation. Long gaps soften connectivity penalties.

On the ten student-test trips, classifier-only mean point accuracy was 0.8778 and Viterbi accuracy was 0.9087. Mean directed route continuity changed from 0.9513 to 0.9945. Per-trip results are in `results/decoder_comparison.csv`.

## Phase checklist

- Phases 1-6: project setup, dataset exploration, preprocessing, candidates/features and reference-route labels are implemented and recorded in the source and existing exploration outputs.
- Phases 7-8: the five classifiers are separated into model modules; grouped CV and student-test comparisons are recorded.
- Phase 9: the Bayesian search and tuned-versus-untuned measurements are retained with the Optuna trial CSV.
- Phases 10-11: Viterbi decoding is implemented and compared; Folium route maps can be regenerated from the prediction-map script.
- Phase 12: the FastAPI service is connected to the pushed React UI contract; the UI production build and TypeScript check pass.
- Phases 13-14: startup and LAN instructions and this report are prepared. The actual other-laptop connection, organizer evaluator data and required demonstration screenshots still need to be supplied/checked by the team.

## API, UI and deployment

The FastAPI service implements `/health`, `/model`, `/trips`, `/trips/{trip_id}` and `/reconstruct`. On the development dataset, functional calls completed for all ten student-test trips. The API predicts first; it reads a reference route only afterward to calculate server-side metrics. Route labels are not returned to the browser. The React/Vite UI connects to the API through its settings panel and renders returned GPS, candidate-road context, prediction and metrics.

For a two-laptop demonstration, run the API on laptop 1 with `uvicorn api.main:app --host 0.0.0.0 --port 8000`; configure laptop 2's UI to use the backend's reachable LAN address. At event time, set `PATHSNAP_TRIPS_DIR` to the organizer-provided evaluator-trip directory. That directory was not available here, so a final cross-laptop evaluator run and the required live screenshots remain team actions.

## Reproducibility and disclosure

The five-model comparison is in `results/experiments_summary.csv` and `results/experiments_folds.csv`; model configurations and random seed are recorded there and in the run metadata. The trained estimator, feature list and model configuration are in `artifacts/final_model.joblib`. AI assistance was used for implementation and report drafting; the team must review the text, verify claims against its understanding, and disclose assistance under the event policy.
