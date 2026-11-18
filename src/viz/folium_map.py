"""Interactive Folium map for one trip: GPS points, road network (candidate arcs), reference route,
classifier-only route and classifier+HMM route, with layer control and a legend.

    python src/viz/folium_map.py --partition student_test --trip 00000052
The reference route (.route) is drawn only for development partitions (never part of prediction).
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from predict import load_bundle, reconstruct_trip


def build_map(trip_dir: Path, bundle: dict, out_html: Path):
    res = reconstruct_trip(trip_dir, bundle)
    tid = trip_dir.name
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes"); arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=int)
    route = np.loadtxt(trip_dir / f"{tid}.route", dtype=int, ndmin=1)
    ll = lambda lonlat: [lonlat[1], lonlat[0]]  # folium wants lat, lon
    obs = res["observations"]
    m = folium.Map(location=ll(obs[len(obs) // 2]), zoom_start=15, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (candidate arcs)", show=False)
    for a, b in res["road_network"]:
        folium.PolyLine([ll(a), ll(b)], color="#888888", weight=2, opacity=0.6).add_to(net)
    gps = folium.FeatureGroup("GPS observations")
    folium.PolyLine([ll(p) for p in obs], color="#1f4fd8", weight=1, opacity=0.4).add_to(gps)
    for i, p in enumerate(obs):
        if i % 3 == 0:
            folium.CircleMarker(ll(p), radius=2, color="#1f4fd8", fill=True, fill_opacity=0.9,
                                tooltip=f"obs {i}").add_to(gps)
    ref = folium.FeatureGroup("Reference route (development only)")
    folium.PolyLine([ll(nodes[n]) for a in route for n in arcs[a]], color="#12a150", weight=7, opacity=0.55).add_to(ref)
    clf = folium.FeatureGroup("Classifier-only route", show=False)
    folium.PolyLine([ll(p) for p in res["classifier_route"]], color="#f28c00", weight=4, dash_array="6 6").add_to(clf)
    hmm = folium.FeatureGroup("Classifier + HMM route")
    folium.PolyLine([ll(p) for p in res["predicted_route"]], color="#d62728", weight=4).add_to(hmm)
    for g in (net, gps, ref, clf, hmm):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;"
              "border:1px solid #888;font-size:12px'><b>Trip %s</b><br>"
              "<span style='color:#1f4fd8'>&#9679;</span> GPS observations<br>"
              "<span style='color:#12a150'>&#9473;</span> Reference route<br>"
              "<span style='color:#f28c00'>&#9473;</span> Classifier-only route<br>"
              "<span style='color:#d62728'>&#9473;</span> Classifier + HMM route<br>"
              "<span style='color:#888'>&#9473;</span> Road network (candidate arcs)</div>" % tid)
    m.get_root().html.add_child(folium.Element(legend))
    lats = [p[1] for p in obs]; lons = [p[0] for p in obs]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test"); ap.add_argument("--trip", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = ROOT / "dataset" / a.partition / a.trip
    out = Path(a.out) if a.out else ROOT / "results" / "maps" / f"map_{a.trip}.html"
    r = build_map(d, load_bundle(), out)
    print("wrote", out, "| timings ms", {k: round(v) for k, v in r["timings_ms"].items()})
