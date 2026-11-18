# Feature pipelines: which one is canonical, and why

Two candidate/feature pipelines exist in this repo. **Use `src/features/pipeline.py` for everything
(models, tuning, HMM, API). Do not use `src/08_candidate_features.py` or `src/09_feature_selection.py`
for any reported result.**

## The problem in `src/08_candidate_features.py`

Lines 211-218: when the reference-route arc is not among the K nearest arcs, the script **inserts the
true arc into the candidate set using the `.route` label** (`forced_route_candidates`). The candidate set
therefore depends on the ground truth. That cannot be reproduced at inference time (the API has no
route), and it breaks the rule that labels must not enter the prediction pipeline.

Measured with the script's own `trip_rows()` (K=8):

| Trip | Observations | True arc forced in by label | Mean distance of true arc |
|---|---|---|---|
| 33 | 5,844 | 3,856 (66.0%) | 80.9 m |
| 54 | 19,313 | 12,732 (65.9%) | 131.8 m |
| 5 | 1,045 | 0 | 5.4 m |
| 3 | 821 | 7 (0.9%) | 5.4 m |

Consequences:
- Candidate miss rate is hidden (always 0 after forcing) and the model sees rows no real candidate
  generator could produce. The "outlier trips 33 and 54 pushed to rank 9 (appended)" explanation in
  `output/reports/feature_selection_summary.json` is this artefact.
- The feature importances and the 0.789 CV Top-1 in `src/09_feature_selection.py` were computed on that
  table, so they should not be reported.
- Labels are the nearest route arc per point (no ordering constraint), which is unreliable on loops and
  overlapping roads. Heading is the bearing to the next point only, with no speed gating.

Genuine finding from that analysis: **trips 33 and 54 have a large systematic offset between the GPS
track and the supplied reference route** (true arc 80-130 m from the points). Mention it as a data
limitation (Dataset Exploration PDF, section 6).

## The canonical pipeline (`src/features/pipeline.py`)

| Step | Method |
|---|---|
| Candidates | all arcs within 60 m (nearest 16), label-free, KD-tree + brute force for long arcs |
| Labels | monotone Viterbi alignment of GPS points to the route (loops/overlaps handled); only used for labels and scoring |
| Unreliable reference | observations > 60 m from their aligned reference arc are dropped and counted (`results/candidate_diagnostics.csv`) |
| Features | 25 features: geometry, heading (speed-gated), speed, time gaps, neighbouring-point context, connectivity, node degrees; all computable without `.route` |
| Speed | whole dataset in about 20 s |

Candidate miss rate with 16 candidates: 0.9% train, 0.1% student-test.

## Feature importance (valid, leak-free): `results/feature_importance.csv`

Permutation importance on held-out trips (trip-grouped 5-fold CV, train partition), measured as the
drop in observation-level Top-1 accuracy. Top results: `cos_heading` 0.229, `dist_gap` 0.088,
`proj_frac` 0.012, `abs_heading_deg` 0.007. Heading agreement is the dominant signal, which is why
two-way roads (twin arcs with identical geometry) are the main source of errors. Individually small
importances for correlated features are understated (shuffling one leaves its correlated partner).
Figures: `results/figures/feature_importance.png`, `feature_correlation.png`.
Regenerate with `python src/features/feature_importance.py`.
