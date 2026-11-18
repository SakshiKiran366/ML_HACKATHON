
import csv
import math

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR


EARTH_RADIUS_KM = 6371.0088


def haversine(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance in kilometres."""

    lon1, lat1, lon2, lat2 = map(
        math.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(
        math.sqrt(a)
    )


def load_coordinates(file_path, columns):
    """Load longitude and latitude from a whitespace file."""

    coordinates = []

    with file_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:
            if not line.strip():
                continue

            values = line.split()

            longitude = float(values[0])
            latitude = float(values[1])

            coordinates.append((longitude, latitude))

    return coordinates


def calculate_spatial_features(coordinates, prefix):
    """Calculate geographic extent and centroid."""

    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]

    lon_min = min(longitudes)
    lon_max = max(longitudes)
    lat_min = min(latitudes)
    lat_max = max(latitudes)

    centroid_lon = sum(longitudes) / len(longitudes)
    centroid_lat = sum(latitudes) / len(latitudes)

    width_km = haversine(
        lon_min, centroid_lat,
        lon_max, centroid_lat
    )

    height_km = haversine(
        centroid_lon, lat_min,
        centroid_lon, lat_max
    )

    return {
        f"{prefix}_point_count": len(coordinates),
        f"{prefix}_longitude_min": lon_min,
        f"{prefix}_longitude_max": lon_max,
        f"{prefix}_latitude_min": lat_min,
        f"{prefix}_latitude_max": lat_max,
        f"{prefix}_centroid_longitude": centroid_lon,
        f"{prefix}_centroid_latitude": centroid_lat,
        f"{prefix}_width_km": width_km,
        f"{prefix}_height_km": height_km,
    }


def calculate_overlap(gps, network):
    """Calculate intersection of GPS and network bounding boxes."""

    lon_min = max(
        gps["gps_longitude_min"],
        network["network_longitude_min"]
    )

    lon_max = min(
        gps["gps_longitude_max"],
        network["network_longitude_max"]
    )

    lat_min = max(
        gps["gps_latitude_min"],
        network["network_latitude_min"]
    )

    lat_max = min(
        gps["gps_latitude_max"],
        network["network_latitude_max"]
    )

    if lon_min >= lon_max or lat_min >= lat_max:
        return 0.0

    gps_area = (
        (gps["gps_longitude_max"] - gps["gps_longitude_min"])
        * (gps["gps_latitude_max"] - gps["gps_latitude_min"])
    )

    intersection_area = (
        (lon_max - lon_min) * (lat_max - lat_min)
    )

    if gps_area == 0:
        return 0.0

    return intersection_area / gps_area * 100


def analyse_trip(record_dir):
    record_id = record_dir.name

    track_file = record_dir / f"{record_id}.track"
    nodes_file = record_dir / f"{record_id}.nodes"

    if not track_file.is_file() or not nodes_file.is_file():
        return None

    gps_coordinates = load_coordinates(track_file, 3)
    network_coordinates = load_coordinates(nodes_file, 2)

    if not gps_coordinates or not network_coordinates:
        return None

    gps_features = calculate_spatial_features(
        gps_coordinates, "gps"
    )

    network_features = calculate_spatial_features(
        network_coordinates, "network"
    )

    centroid_offset = haversine(
        gps_features["gps_centroid_longitude"],
        gps_features["gps_centroid_latitude"],
        network_features["network_centroid_longitude"],
        network_features["network_centroid_latitude"]
    )

    overlap_percentage = calculate_overlap(
        gps_features, network_features
    )

    return {
        "record_id": record_id,
        **gps_features,
        **network_features,
        "centroid_offset_km": centroid_offset,
        "gps_bbox_overlap_percentage": overlap_percentage,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    partitions = [
        ("train", TRAIN_DIR),
        ("student_test", TEST_DIR),
    ]

    for split_name, split_dir in partitions:

        if not split_dir.exists():
            continue

        for record_dir in sorted(split_dir.iterdir()):

            if not record_dir.is_dir():
                continue

            features = analyse_trip(record_dir)

            if features is None:
                print(
                    f"[WARNING] Skipping {record_dir.name}"
                )
                continue

            features["split"] = split_name
            results.append(features)

            print(
                f"[PROCESSED] {split_name}: {record_dir.name}"
            )

    if not results:
        print("[ERROR] No geographical features generated.")
        return

    output_path = (
        REPORT_DIR / "geographical_features.csv"
    )

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== GEOGRAPHICAL FEATURES ==========")
    print(f"Trips processed: {len(results)}")

    print(
        "Average GPS width (km): "
        f"{sum(r['gps_width_km'] for r in results) / len(results):.3f}"
    )

    print(
        "Average GPS height (km): "
        f"{sum(r['gps_height_km'] for r in results) / len(results):.3f}"
    )

    print(
        "Average centroid offset (km): "
        f"{sum(r['centroid_offset_km'] for r in results) / len(results):.3f}"
    )

    print("\nGenerated report:")
    print(output_path)


if __name__ == "__main__":
    main()
