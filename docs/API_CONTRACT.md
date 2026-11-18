# PathSnap API contract (Laptop 1 backend <-> Laptop 2 frontend)

The backend runs on **Laptop 1** (FastAPI). The frontend (`frontend/`) runs on Laptop 2 and sets
`VITE_API_BASE_URL=http://<laptop-1-LAN-IP>:8000` (or via the in-app API settings).
Field names below match what `frontend/src/services/api.ts` already accepts. All coordinates are
`[longitude, latitude]`.

## Hard rules
- Ground-truth (`.route`) data is read only inside server-side scoring. It is **never** returned by any endpoint.
- The UI sends only a trip id. The server runs the full pipeline (candidates, features, classifier, HMM/Viterbi) every time. No cached or hardcoded answers.
- Every request is logged to the console (method, path, trip id, status, latency) so the evaluator can see it.

## Server configuration (environment variables)
| Variable | Meaning | Default |
|---|---|---|
| `PATHSNAP_TRIP_DIR` | folder with trip folders `<id>/<id>.{track,nodes,arcs,route}` | `dataset/student_test` (development) |
| `PATHSNAP_CORS_ORIGINS` | allowed frontend origins, comma separated | `*` for development |

At evaluation time `PATHSNAP_TRIP_DIR` is pointed at the folder holding the ten evaluator trips
(kept out of git: `.gitignore` excludes `dataset/evaluator*/`). The trip list is read from that folder,
so no ids are hardcoded. The folder must contain each trip's `.route` file for server-side scoring.

## Endpoints

### `GET /health`
`{"ok": true, "message": "PathSnap API ready", "model": "hist_gb", "trips_available": 10}`

### `GET /trips`
`{"trips": [{"id": "00000000", "label": "Trip 00000000", "observation_count": 1635}, ...]}`

### `GET /trips/{trip_id}`
Permitted inputs only, for display:
```
{"id": "00000000",
 "observations": [[lon, lat], ...],
 "road_network": [[[lon, lat], [lon, lat]], ...],   // only arcs within 60 m of some GPS point
 "start": [lon, lat], "end": [lon, lat]}
```
`road_network` is limited to candidate arcs (not the full network, which can have 100k arcs).
Errors: `404` unknown trip.

### `POST /reconstruct`   body `{"trip_id": "00000000"}`
Runs the full pipeline and scores against the server-side reference.
```
{"id": "00000000",
 "observations": [[lon, lat], ...],
 "road_network": [...],
 "predicted_route": [[lon, lat], ...],     // polyline of the matched arcs in travel order
 "metrics": {
   "observations_scored": 1630,
   "top1_classifier_only": 0.86, "top1_hmm": 0.93,
   "route_continuity_classifier_only": 0.95, "route_continuity_hmm": 0.99,
   "candidate_miss_rate": 0.001,
   "reference_dropped_observations": 5},
 "processing_time_ms": 842}
```
`metrics` values are numbers or strings (the frontend renders them as a table). Errors: `404` unknown
trip, `422` bad body, `500` with `{"detail": "..."}`.

## Open items
- The numeric values above are examples of the shape only; real values come from the running backend.
- HMM/Viterbi decoding is required before `top1_hmm` and `route_continuity_hmm` exist; until then the
  backend would return classifier-only values and must label them as such.
