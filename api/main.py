"""PathSnap inference API. Ground-truth routes are read only on the server."""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "hmm"))
import pipeline as P  # noqa: E402
from viterbi import decode  # noqa: E402

app = FastAPI(title="PathSnap API", version="1.0.0", description="Server-side GPS map matching")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
TRIP_ROOT = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "dataset" / "student_test")).expanduser().resolve()
USING_DEV_DATA = "PATHSNAP_TRIPS_DIR" not in os.environ
MODEL_PATH = ROOT / "artifacts" / "final_model.joblib"
model_bundle = None

class ReconstructRequest(BaseModel):
    trip_id: str

def trip_dir(trip_id: str) -> Path:
    # IDs are numeric in the supplied dataset; avoid arbitrary filesystem paths.
    if not trip_id.isdigit() or len(trip_id) > 8:
        raise HTTPException(404, "Trip not found")
    path = TRIP_ROOT / f"{int(trip_id):08d}"
    if not path.is_dir():
        raise HTTPException(404, "Trip not found")
    return path

def read_trip(trip_id: str):
    folder = trip_dir(trip_id)
    track, nodes, arcs, _route = P.read_trip(folder, include_route=False)
    return folder, track, nodes, arcs

def predict(track, nodes, arcs):
    # API inference receives track/network only; reference route is never passed to feature/model code.
    frame, _ = P.build_trip(track, nodes, arcs, route=None, trip_id=0, partition="student_test")
    if frame.empty:
        raise HTTPException(422, "No road candidates could be generated for this trip")
    if model_bundle is None:
        # Predictable startup experience when a trained artifact has not been created yet.
        frame["score"] = -frame["dist_m"].to_numpy()
    else:
        model = model_bundle["model"]
        expected = model_bundle["features"]
        raw = model.predict_proba(frame[expected].to_numpy(dtype=np.float32))
        frame["score"] = np.asarray(raw)[:, 1] if np.asarray(raw).ndim == 2 else np.asarray(raw)
    chosen = decode(frame, frame.score.to_numpy())
    selected = frame.iloc[chosen].sort_values("obs_idx")
    # Return aligned segment vertices so the UI can render an actual road route.
    route_nodes = []
    for row in selected.itertuples():
        route_nodes.append([int(row.arc_from), int(row.arc_to)])
    return frame, selected, route_nodes

def geometry(track, nodes, selected, route_nodes):
    # Dataset node/arc indices are in geographic coordinates (lon, lat).
    gps = [[float(lon), float(lat)] for lon, lat in track[:, :2]]
    road = []
    seen = set()
    for u, v in route_nodes:
        edge = (u, v)
        if edge in seen: continue
        seen.add(edge)
        road.append([[float(nodes[u, 0]), float(nodes[u, 1])], [float(nodes[v, 0]), float(nodes[v, 1])]])
    predicted = [[float(nodes[int(u), 0]), float(nodes[int(u), 1])] for u, _ in route_nodes]
    if route_nodes:
        predicted.append([float(nodes[int(route_nodes[-1][1]), 0]), float(nodes[int(route_nodes[-1][1]), 1])])
    return gps, road, predicted

@app.get("/health")
def health():
    return {"ok": True, "message": "PathSnap inference API ready", "model_loaded": model_bundle is not None}

@app.get("/trips")
def trips():
    items = []
    for folder in sorted((p for p in TRIP_ROOT.iterdir() if p.is_dir()), key=lambda p: p.name):
        try:
            track, *_ = P.read_trip(folder, include_route=False)
            label = "Student test" if USING_DEV_DATA else "Evaluation trip"
            items.append({"id": folder.name, "label": label, "observation_count": len(track)})
        except Exception:
            continue
    return {"trips": items}

@app.get("/trips/{trip_id}")
def get_trip(trip_id: str):
    _folder, track, nodes, arcs = read_trip(trip_id)
    gps = [[float(lon), float(lat)] for lon, lat in track[:, :2]]
    candidates, _ = P.build_trip(track, nodes, arcs, route=None, trip_id=0, partition="student_test")
    arc_ids = candidates.sort_values("dist_m").arc_id.drop_duplicates().head(1000).to_numpy(dtype=int)
    road = [[[float(nodes[arcs[i, 0], 0]), float(nodes[arcs[i, 0], 1])],
             [float(nodes[arcs[i, 1], 0]), float(nodes[arcs[i, 1], 1])]] for i in arc_ids]
    return {"id": trip_id, "observations": gps, "road_network": road,
            "start": gps[0] if gps else None, "end": gps[-1] if gps else None}

@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest):
    start = time.perf_counter()
    _folder, track, nodes, arcs = read_trip(req.trip_id)
    frame, selected, route_nodes = predict(track, nodes, arcs)
    gps, road, predicted = geometry(track, nodes, selected, route_nodes)
    # Ground truth remains server-side and never appears in the response.
    metrics = {"matched_observations": int(selected.obs_idx.nunique()), "candidate_count": int(len(frame))}
    # Reference routes are consulted only inside the backend to score the result.
    route = P.read_trip(_folder)[3]
    if route is not None and len(track):
        clean, _ = P.clean_track(track)
        lon0, lat0 = clean[:, 0].mean(), clean[:, 1].mean()
        points = P.project(clean[:, :2], lon0, lat0)
        projected_nodes = P.project(nodes, lon0, lat0)
        A, B = projected_nodes[arcs[:, 0]], projected_nodes[arcs[:, 1]]
        aligned, _ = P.align_labels(points, clean[:, 2], A, B, route)
        ordered = selected.sort_values("obs_idx")
        obs = ordered.obs_idx.to_numpy(dtype=int)
        valid = obs < len(aligned)
        pred_arc = ordered.arc_id.to_numpy()[valid]
        truth_arc = aligned[obs[valid]]
        metrics["accuracy"] = round(float(np.mean(pred_arc == truth_arc) * 100), 2) if len(pred_arc) else 0.0
        metrics["route_similarity"] = metrics["accuracy"]
        raw_best = frame.loc[frame.groupby("obs_idx").score.idxmax()].sort_values("obs_idx")
        raw_obs = raw_best.obs_idx.to_numpy(dtype=int)
        raw_valid = raw_obs < len(aligned)
        metrics["classifier_accuracy"] = round(float(np.mean(raw_best.arc_id.to_numpy()[raw_valid] == aligned[raw_obs[raw_valid]]) * 100), 2) if raw_valid.any() else 0.0
        links = (ordered.arc_id.to_numpy()[1:] == ordered.arc_id.to_numpy()[:-1]) | (ordered.arc_to.to_numpy()[:-1] == ordered.arc_from.to_numpy()[1:])
        consecutive = np.diff(obs) == 1
        metrics["route_continuity"] = round(float(links[consecutive].mean() * 100), 2) if consecutive.any() else 100.0
    elapsed = round((time.perf_counter() - start) * 1000)
    metrics["processing_time_ms"] = elapsed
    return {"id": req.trip_id, "observations": gps, "road_network": road, "predicted_route": predicted,
            "start": gps[0] if gps else None, "end": gps[-1] if gps else None,
            "processing_time_ms": elapsed, "metrics": metrics}

@app.on_event("startup")
def load_model():
    global model_bundle
    if MODEL_PATH.exists():
        import joblib
        model_bundle = joblib.load(MODEL_PATH)

@app.get("/model")
def model_status():
    return {"loaded": model_bundle is not None,
            "name": model_bundle.get("name") if model_bundle else None,
            "config": (model_bundle.get("config") or model_bundle.get("params")) if model_bundle else None,
            "max_iter": model_bundle.get("max_iter") if model_bundle else None}
