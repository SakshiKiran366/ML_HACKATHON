# Results summary (all numbers come from files in `results/`; evaluator data was never used)

## 1. Five-model comparison (`results/experiments_summary.csv`)
Trip-grouped 5-fold CV on the 80 train trips; student-test = 10 development trips. Primary metric:
observation-level Top-1 accuracy, log loss as tie-breaker. HistGradientBoosting has the best CV Top-1
(0.8539) and best log loss, but the gaps between the top models are inside the fold-to-fold spread (~0.046),
so the margin is narrow.

## 2. Bayesian optimisation (`results/optuna_best.json`, `optuna_trials.csv`, `tuned_vs_untuned.csv`)
Optuna TPE, 10 trials, grouped CV Top-1: tuned 0.8549 vs untuned 0.8538 (student-test 0.8586 vs 0.8574).
The improvement is about 0.001, i.e. within noise; the default configuration was already near-optimal.

## 3. Probability quality (`results/calibration.json`, `figures/reliability.png`)
Student-test log loss 0.0690, Brier 0.0202, expected calibration error 0.0054. Isotonic calibration changes
log loss by 0.00014, so it is **not** adopted.

## 4. Decoding: classifier-only vs HMM/Viterbi (`results/hmm_results.csv`, `hmm_per_trip_student_test.csv`)
Same probabilities for every row (train = out-of-fold, student-test = shipped model `artifacts/hist_gb_tuned.joblib`).
Decoder parameters were tuned (Optuna TPE, 40 trials) on the **train out-of-fold** probabilities only.

| Method | Train OOF Top-1 | Student-test Top-1 | Student-test route F1 | Student-test strict continuity | Student-test network-reachable continuity |
|---|---:|---:|---:|---:|---:|
| Classifier only | 0.8583 | 0.8586 | 0.8967 | 0.9484 | 0.9857 |
| Viterbi, hand-set constants (`src/hmm/viterbi.py`) | 0.8905 | 0.8885 | 0.9516 | 0.9943 | 0.9900 |
| HMM, default parameters (`src/hmm/hmm.py`) | 0.8929 | 0.8902 | 0.9386 | 0.9705 | 0.9959 |
| **HMM, tuned parameters** | **0.9107** | **0.9086** | 0.9463 | 0.9769 | 0.9900 |

- The tuned HMM raises Top-1 by 5.0 points over classifier-only on student-test and improves all 10 trips
  (`figures/hmm_per_trip.png`). The train out-of-fold and student-test gains agree (0.9107 vs 0.9086), so the
  tuning did not overfit.
- The hand-set Viterbi has better route F1 and strict (directly head-to-tail) continuity; it never leaves a
  directly connected arc. The tuned HMM has higher Top-1 and route precision (0.960 vs 0.936) but lower recall
  (0.933 vs 0.968). With 1 Hz sampling a fast vehicle skips short arcs, so strict continuity understates
  network consistency; on the network-reachability measure the two decoders tie (0.990).
- Cases where the HMM did not fully help: trip 77 (Top-1 0.704 to 0.805, still the weakest). Its vehicle is slow
  (median 5.5 m/s) on a dense network (12.6 candidates per point vs 4.4 for the easiest trip), so parallel and
  reverse-direction arcs stay ambiguous. Labels for this trip are clean (0 dropped, 100% of true arcs among candidates).
- Error analysis before the HMM (student-test, classifier-only): 13.7% wrong, of which 41.7% reverse-direction twin,
  42.1% adjacent arc, 16.2% other; 57.8% wrong when GPS heading is unreliable vs 12.5% otherwise.

## 5. Known data limitations
- Train trips 33 and 54: the reference route is 80-130 m from the GPS track (systematic offset); 52% and 88% of their
  observations are dropped as unreliable (`results/candidate_diagnostics.csv`).
- Supplied reference routes can have breaks (e.g. train trip 2), so labels are reference labels, not verified truth.
- Candidate parameters (60 m radius, 16 per point) were fixed in advance; candidate miss rate is 0.9% train, 0.1% student-test.
- Exact-duplicate arcs exist in some networks (an irreducible ambiguity).

## 6. Reproduce
```
python src/features/build_table.py
python src/models/run_experiments.py
python src/tuning/optuna_hgb.py --trials 10
python src/hmm/run_hmm_experiment.py --trials 40
python src/models/export_artifacts.py && python src/models/calibration_check.py
python src/evaluation/plots.py && python src/evaluation/plot_hmm.py
python src/viz/folium_map.py --partition student_test --trip 00000052
```
`src/predict.py: reconstruct_trip()` runs the full pipeline (candidates, features, classifier, HMM) from `.track/.nodes/.arcs`
only. The API currently decodes with `src/hmm/viterbi.py`; switching it to `reconstruct_trip` would use the tuned HMM.
