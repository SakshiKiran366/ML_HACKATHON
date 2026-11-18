"""Create a Folium map of GPS, reference, and model-predicted routes for a dev trip."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.main import load_model, predict, read_trip  # noqa: E402
import pipeline as P  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trip", default="00000000")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "maps" / "prediction_map.html")
    a = p.parse_args()
    load_model()
    folder, track, nodes, arcs = read_trip(a.trip)
    _, selected, route_nodes = predict(track, nodes, arcs)
    route = P.read_trip(folder)[3]
    gps = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    pred_nodes = []
    if route_nodes:
        pred_nodes = [[float(nodes[u, 1]), float(nodes[u, 0])] for u, _ in route_nodes]
        pred_nodes.append([float(nodes[route_nodes[-1][1], 1]), float(nodes[route_nodes[-1][1], 0])])
    reference = []
    if route is not None:
        for arc_id in route:
            u, v = arcs[int(arc_id)]
            if not reference: reference.append([float(nodes[u, 1]), float(nodes[u, 0])])
            reference.append([float(nodes[v, 1]), float(nodes[v, 0])])
    center = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    fg_pred = folium.FeatureGroup(name="Viterbi prediction", show=True)
    fg_ref = folium.FeatureGroup(name="Reference route (server-side dev data)", show=False)
    step = max(1, len(gps) // 150)
    for pt in gps[::step]: folium.CircleMarker(pt, radius=2, color="#e6a23c", fill=True, fill_opacity=0.75).add_to(fg_gps)
    if gps: folium.PolyLine(gps, color="#e6a23c", weight=2, opacity=0.8).add_to(fg_gps)
    if len(pred_nodes) > 1: folium.PolyLine(pred_nodes, color="#14a878", weight=5, opacity=0.95, tooltip="Model prediction").add_to(fg_pred)
    if len(reference) > 1: folium.PolyLine(reference, color="#4378bf", weight=4, opacity=0.85, tooltip="Reference route").add_to(fg_ref)
    for fg in (fg_gps, fg_pred, fg_ref): fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    png = ROOT.parents[1] / "outputs" / f"PathSnap_Map_{a.trip}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.scatter(track[::step, 0], track[::step, 1], s=5, c="#e6a23c", alpha=0.8, label="GPS observations", zorder=3)
    ax.plot(track[:, 0], track[:, 1], color="#e6a23c", linewidth=1, alpha=0.5)
    if len(pred_nodes) > 1:
        ax.plot([p[1] for p in pred_nodes], [p[0] for p in pred_nodes], color="#14a878", linewidth=2.2, label="Viterbi prediction", zorder=2)
    if len(reference) > 1:
        ax.plot([p[1] for p in reference], [p[0] for p in reference], color="#4378bf", linewidth=1.8, label="Reference route", zorder=1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(f"PathSnap route reconstruction · trip {a.trip}")
    ax.grid(alpha=0.2); ax.legend(loc="best", frameon=True); fig.tight_layout(); fig.savefig(png); plt.close(fig)
    print(f"Saved {a.out} for trip {a.trip} ({len(selected)} decoded GPS observations)")
    print(f"Saved {png}")

if __name__ == "__main__": main()
