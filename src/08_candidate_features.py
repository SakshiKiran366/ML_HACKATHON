"""Build observation-by-candidate features for the map-matching models.

WARNING (superseded, do not use for reported results): trip_rows() inserts the true route arc into the
candidate set when it is missing (see the `candidate_misses` block), which leaks the label into
candidate generation. Use src/features/pipeline.py instead. Details: docs/FEATURE_PIPELINES.md.

The raw dataset stores one road graph and one GPS track per trip.  This
script converts those files into a long CSV with one row per GPS observation
and nearby directed arc.  Labels are derived from the route file for the
development partitions; the route is never used as a model feature.

Run from the repository root:

    python src/08_candidate_features.py --k 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_ROOT, OUTPUT_DIR

EARTH_RADIUS_M = 6_371_008.8
DEFAULT_K = 8
CELL_SIZE_M = 250.0
SEARCH_RADII_M = (250.0, 500.0, 1_000.0, 2_000.0, 5_000.0)


def read_points(path: Path, columns: int) -> list[tuple[float, ...]]:
    points = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) < columns:
                raise ValueError(f"{path}:{line_number} has fewer than {columns} values")
            points.append(tuple(float(value) for value in values[:columns]))
    return points


def read_arcs(path: Path) -> list[tuple[int, int]]:
    rows = read_points(path, 2)
    return [(int(source), int(target)) for source, target in rows]


def read_route(path: Path) -> list[int]:
    return [int(row[0]) for row in read_points(path, 1)]


def xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    scale = math.pi / 180.0
    return (
        (lon - lon0) * scale * EARTH_RADIUS_M * math.cos(lat0 * scale),
        (lat - lat0) * scale * EARTH_RADIUS_M,
    )


def bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, start)
    lon2, lat2 = map(math.radians, end)
    value = math.degrees(math.atan2(
        math.sin(lon2 - lon1) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1),
    ))
    return (value + 360.0) % 360.0


def angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        fraction = 0.0
    else:
        fraction = max(
            0.0,
            min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
        )
    closest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1]), fraction


def cell(value: float) -> int:
    return math.floor(value / CELL_SIZE_M)


def cells_for_bbox(min_x: float, max_x: float, min_y: float, max_y: float):
    for gx in range(cell(min_x), cell(max_x) + 1):
        for gy in range(cell(min_y), cell(max_y) + 1):
            yield gx, gy


def build_arc_records(nodes, arcs, lon0, lat0):
    records = []
    for arc_id, (source, target) in enumerate(arcs):
        if source < 0 or target < 0 or source >= len(nodes) or target >= len(nodes):
            continue
        start = xy(nodes[source][0], nodes[source][1], lon0, lat0)
        end = xy(nodes[target][0], nodes[target][1], lon0, lat0)
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        records.append({
            "arc_id": arc_id,
            "arc_from": source,
            "arc_to": target,
            "start": start,
            "end": end,
            "length_m": length,
            "bearing_deg": bearing(nodes[source], nodes[target]),
        })
    return records


def build_index(records):
    index = {}
    for record in records:
        start, end = record["start"], record["end"]
        for key in cells_for_bbox(
            min(start[0], end[0]),
            max(start[0], end[0]),
            min(start[1], end[1]),
            max(start[1], end[1]),
        ):
            index.setdefault(key, []).append(record)
    return index


def nearby_arcs(point, index, records, k):
    candidates = {}
    gx, gy = cell(point[0]), cell(point[1])
    for radius in SEARCH_RADII_M:
        steps = max(1, math.ceil(radius / CELL_SIZE_M))
        for ix in range(gx - steps, gx + steps + 1):
            for iy in range(gy - steps, gy + steps + 1):
                for record in index.get((ix, iy), ()):
                    candidates[record["arc_id"]] = record
        scored = []
        for record in candidates.values():
            distance, fraction = point_segment_distance(point, record["start"], record["end"])
            scored.append((distance, fraction, record))
        scored.sort(key=lambda item: (item[0], item[2]["arc_id"]))
        if len(scored) >= k:
            return scored[:k], scored
    scored = []
    for record in records:
        distance, fraction = point_segment_distance(point, record["start"], record["end"])
        scored.append((distance, fraction, record))
    scored.sort(key=lambda item: (item[0], item[2]["arc_id"]))
    return scored[:k], scored


def trip_rows(record_dir: Path, partition: str, k: int):
    trip_id = record_dir.name
    prefix = record_dir / trip_id
    nodes = read_points(prefix.with_suffix(".nodes"), 2)
    arcs = read_arcs(prefix.with_suffix(".arcs"))
    track = read_points(prefix.with_suffix(".track"), 3)
    route = read_route(prefix.with_suffix(".route"))

    lon0 = sum(point[0] for point in nodes) / len(nodes)
    lat0 = sum(point[1] for point in nodes) / len(nodes)
    records = build_arc_records(nodes, arcs, lon0, lat0)
    by_id = {record["arc_id"]: record for record in records}
    index = build_index(records)
    route_records = [by_id[arc_id] for arc_id in route if arc_id in by_id]
    route_index = build_index(route_records)

    projected = [xy(point[0], point[1], lon0, lat0) for point in track]
    gps_bearings = []
    speeds = []
    time_gaps = []
    for i, point in enumerate(track):
        if i + 1 < len(track):
            next_point = track[i + 1]
            gps_bearings.append(bearing(point[:2], next_point[:2]))
        elif i:
            gps_bearings.append(gps_bearings[-1])
        else:
            gps_bearings.append(0.0)
        if i == 0:
            speeds.append(0.0)
            time_gaps.append(0.0)
        else:
            gap = max(0.0, track[i][2] - track[i - 1][2])
            distance = math.hypot(
                projected[i][0] - projected[i - 1][0],
                projected[i][1] - projected[i - 1][1],
            )
            time_gaps.append(gap)
            speeds.append(distance / gap if gap > 0 else 0.0)

    all_rows = []
    previous_candidates = {}
    candidate_misses = 0
    for obs_idx, point in enumerate(projected):
        selected, all_scored = nearby_arcs(point, index, records, k)
        route_selected, _ = nearby_arcs(point, route_index, route_records, 1) if route_records else ([], [])
        true_arc_id = route_selected[0][2]["arc_id"] if route_selected else None
        candidate_ids = {item[2]["arc_id"] for item in selected}
        if true_arc_id is not None and true_arc_id not in candidate_ids:
            candidate_misses += 1
            true_record = by_id[true_arc_id]
            true_distance, true_fraction = point_segment_distance(
                point, true_record["start"], true_record["end"]
            )
            selected.append((true_distance, true_fraction, true_record))
            selected.sort(key=lambda item: (item[0], item[2]["arc_id"]))

        current_candidates = {}
        minimum_distance = selected[0][0] if selected else 0.0
        for distance, fraction, record in selected:
            previous = previous_candidates.get(record["arc_id"])
            connected = int(
                previous is not None
                and (
                    record["arc_from"] == previous["arc_to"]
                    or record["arc_id"] == previous["arc_id"]
                )
            )
            current_candidates[record["arc_id"]] = record
            all_rows.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "arc_id": record["arc_id"],
                "arc_from": record["arc_from"],
                "arc_to": record["arc_to"],
                "label": int(record["arc_id"] == true_arc_id) if true_arc_id is not None else "",
                "partition": partition,
                "dist_m": distance,
                "heading_diff_deg": angle_difference(gps_bearings[obs_idx], record["bearing_deg"]),
                "heading_score": (1.0 + math.cos(math.radians(
                    angle_difference(gps_bearings[obs_idx], record["bearing_deg"])
                ))) / 2.0,
                "speed_mps": speeds[obs_idx],
                "time_gap_s": time_gaps[obs_idx],
                "arc_len_m": record["length_m"],
                "proj_frac": fraction,
                "dist_rank": len(current_candidates),
                "dist_gap": distance - minimum_distance,
                "prev_dist_m": previous["distance"] if previous is not None else "",
                "connected_prev": connected,
            })
        previous_candidates = {
        arc_id: {"arc_id": arc_id, "arc_to": record["arc_to"], "distance": distance}
            for distance, _, record in selected
            for arc_id in [record["arc_id"]]
        }
    return all_rows, candidate_misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "processed" / "candidate_features.csv")
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be at least 2")

    rows = []
    summary = []
    for partition in ("train", "student_test"):
        split_dir = DATASET_ROOT / partition
        for record_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            trip_rows_data, misses = trip_rows(record_dir, partition, args.k)
            rows.extend(trip_rows_data)
            summary.append({
                "partition": partition,
                "trip_id": record_dir.name,
                "observations": len(set(row["obs_idx"] for row in trip_rows_data)),
                "candidate_rows": len(trip_rows_data),
                "forced_route_candidates": misses,
            })
            print(f"[PROCESSED] {partition}: {record_dir.name} rows={len(trip_rows_data)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.out.with_name("candidate_features_summary.json")
    report_path.write_text(json.dumps({
        "k": args.k,
        "rows": len(rows),
        "features": fieldnames[7:],
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Wrote trip summary to {report_path}")


if __name__ == "__main__":
    main()
