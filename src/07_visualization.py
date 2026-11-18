
import matplotlib.pyplot as plt
import folium

from pathlib import Path

from config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR


PLOT_DIR = OUTPUT_DIR / "visualizations"
MAP_DIR = OUTPUT_DIR / "maps"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)


def load_coordinates(file_path, expected_columns):
    coordinates = []

    with file_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:
            if not line.strip():
                continue

            values = line.split()

            if len(values) != expected_columns:
                continue

            coordinates.append((
                float(values[0]),
                float(values[1])
            ))

    return coordinates


def find_first_training_trip():
    trips = sorted(
        path for path in TRAIN_DIR.iterdir()
        if path.is_dir()
    )

    return trips[0] if trips else None


def plot_coordinate_distribution(trips):
    longitudes = []
    latitudes = []

    for trip_dir in trips:

        track_file = (
            trip_dir / f"{trip_dir.name}.track"
        )

        if not track_file.exists():
            continue

        coordinates = load_coordinates(track_file, 3)

        for lon, lat in coordinates:
            longitudes.append(lon)
            latitudes.append(lat)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        longitudes,
        latitudes,
        s=1,
        alpha=0.3
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GPS Coordinate Distribution")
    plt.grid(alpha=0.3)

    output_file = PLOT_DIR / "gps_distribution.png"

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_file}")


def plot_representative_trajectory(trip_dir):
    track_file = (
        trip_dir / f"{trip_dir.name}.track"
    )

    coordinates = load_coordinates(track_file, 3)

    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]

    plt.figure(figsize=(10, 7))

    plt.plot(
        longitudes,
        latitudes,
        linewidth=0.8
    )

    plt.scatter(
        longitudes[0],
        latitudes[0],
        color="green",
        label="Start",
        s=40
    )

    plt.scatter(
        longitudes[-1],
        latitudes[-1],
        color="red",
        label="End",
        s=40
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"GPS Trajectory - {trip_dir.name}")
    plt.legend()
    plt.grid(alpha=0.3)

    output_file = (
        PLOT_DIR / f"trajectory_{trip_dir.name}.png"
    )

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_file}")


def create_interactive_map(trip_dir):
    record_id = trip_dir.name

    track_file = trip_dir / f"{record_id}.track"
    nodes_file = trip_dir / f"{record_id}.nodes"
    arcs_file = trip_dir / f"{record_id}.arcs"

    gps = load_coordinates(track_file, 3)
    nodes = load_coordinates(nodes_file, 2)

    center_lat = sum(p[1] for p in gps) / len(gps)
    center_lon = sum(p[0] for p in gps) / len(gps)

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        control_scale=True
    )

    # GPS trajectory layer
    gps_points = [
        (lat, lon) for lon, lat in gps
    ]

    folium.PolyLine(
        gps_points,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="GPS trajectory"
    ).add_to(map_object)

    # Start and end markers
    folium.Marker(
        gps_points[0],
        tooltip="Trajectory start",
        icon=folium.Icon(color="green")
    ).add_to(map_object)

    folium.Marker(
        gps_points[-1],
        tooltip="Trajectory end",
        icon=folium.Icon(color="red")
    ).add_to(map_object)

    # Sample road-network arcs to keep map responsive
    with arcs_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        arc_lines = [
            line for line in file if line.strip()
        ]

    step = max(1, len(arc_lines) // 3000)

    for arc_id in range(0, len(arc_lines), step):

        values = arc_lines[arc_id].split()

        if len(values) != 2:
            continue

        from_node = int(values[0])
        to_node = int(values[1])

        if not (
            0 <= from_node < len(nodes)
            and 0 <= to_node < len(nodes)
        ):
            continue

        lon1, lat1 = nodes[from_node]
        lon2, lat2 = nodes[to_node]

        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color="gray",
            weight=1,
            opacity=0.35
        ).add_to(map_object)

    output_file = (
        MAP_DIR / f"map_{record_id}.html"
    )

    map_object.save(str(output_file))

    print(f"[SAVED] {output_file}")


def main():
    train_trips = sorted(
        path for path in TRAIN_DIR.iterdir()
        if path.is_dir()
    )

    test_trips = sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    all_trips = train_trips + test_trips

    print("\n========== DATA VISUALIZATION ==========")

    plot_coordinate_distribution(all_trips)

    representative_trip = find_first_training_trip()

    if representative_trip is None:
        print("[ERROR] No training trip found.")
        return

    print(
        f"\nRepresentative trip: {representative_trip.name}"
    )

    plot_representative_trajectory(representative_trip)

    create_interactive_map(representative_trip)

    print("\nVisualization completed successfully.")


if __name__ == "__main__":
    main()
