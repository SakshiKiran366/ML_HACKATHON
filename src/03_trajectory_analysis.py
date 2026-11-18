
import csv
import math
import statistics

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR


EARTH_RADIUS_KM = 6371.0088


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate geographical distance between two GPS points."""

    lon1, lat1, lon2, lat2 = map(
        math.radians, [lon1, lat1, lon2, lat2]
    )

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def load_track(file_path):
    """Read longitude, latitude and timestamp from a track file."""

    observations = []

    with file_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:
            if not line.strip():
                continue

            longitude, latitude, timestamp = map(
                float, line.split()
            )

            observations.append(
                (longitude, latitude, timestamp)
            )

    return observations


def analyse_trajectory(observations):
    """Calculate temporal and geographical trajectory statistics."""

    longitudes = [point[0] for point in observations]
    latitudes = [point[1] for point in observations]
    timestamps = [point[2] for point in observations]

    intervals = []
    distances = []
    speeds = []

    for i in range(1, len(observations)):
        previous = observations[i - 1]
        current = observations[i]

        time_difference = current[2] - previous[2]
        intervals.append(time_difference)

        distance = haversine_distance(
            previous[0],
            previous[1],
            current[0],
            current[1],
        )

        distances.append(distance)

        if time_difference > 0:
            speeds.append(
                distance / time_difference * 3600
            )

    positive_intervals = [
        interval for interval in intervals if interval > 0
    ]

    non_increasing_intervals = sum(
        interval <= 0 for interval in intervals
    )

    gaps_over_1 = sum(interval > 1 for interval in intervals)
    gaps_over_5 = sum(interval > 5 for interval in intervals)
    gaps_over_10 = sum(interval > 10 for interval in intervals)

    result = {
        "observation_count": len(observations),
        "longitude_min": min(longitudes),
        "longitude_max": max(longitudes),
        "latitude_min": min(latitudes),
        "latitude_max": max(latitudes),
        "timestamp_start": timestamps[0],
        "timestamp_end": timestamps[-1],
        "duration_seconds": timestamps[-1] - timestamps[0],
        "interval_count": len(intervals),
        "non_increasing_intervals": non_increasing_intervals,
        "gaps_over_1_second": gaps_over_1,
        "gaps_over_5_seconds": gaps_over_5,
        "gaps_over_10_seconds": gaps_over_10,
        "max_gap_seconds": max(intervals) if intervals else 0,
        "mean_interval_seconds": (
            statistics.mean(intervals) if intervals else 0
        ),
        "median_interval_seconds": (
            statistics.median(intervals) if intervals else 0
        ),
        "exactly_1_second_count": sum(
            interval == 1 for interval in intervals
        ),
        "positive_interval_count": len(positive_intervals),
        "total_distance_km": sum(distances),
        "maximum_observed_speed_kmh": (
            max(speeds) if speeds else 0
        ),
    }

    result["exactly_1_second_percentage"] = (
        result["exactly_1_second_count"]
        / len(intervals) * 100
        if intervals else 0
    )

    return result


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

            track_file = (
                record_dir / f"{record_dir.name}.track"
            )

            if not track_file.is_file():
                print(f"[WARNING] Missing track: {record_dir.name}")
                continue

            observations = load_track(track_file)

            if not observations:
                print(f"[WARNING] Empty track: {record_dir.name}")
                continue

            stats = analyse_trajectory(observations)

            results.append({
                "split": split_name,
                "record_id": record_dir.name,
                **stats,
            })

    if not results:
        print("[ERROR] No trajectories available for analysis.")
        return

    output_path = REPORT_DIR / "trajectory_analysis.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    total_trips = len(results)
    total_observations = sum(
        row["observation_count"] for row in results
    )

    total_intervals = sum(
        row["interval_count"] for row in results
    )

    total_exactly_1 = sum(
        row["exactly_1_second_count"] for row in results
    )

    total_irregular = sum(
        row["gaps_over_1_second"] for row in results
    )

    total_non_increasing = sum(
        row["non_increasing_intervals"] for row in results
    )

    print("\n========== TRAJECTORY ANALYSIS ==========")
    print(f"Trips analysed              : {total_trips}")
    print(f"GPS observations             : {total_observations}")
    print(f"Timestamp intervals          : {total_intervals}")
    print(f"Exactly 1-second intervals  : {total_exactly_1}")

    print(
        f"Exactly 1-second percentage : "
        f"{total_exactly_1 / total_intervals * 100:.2f}%"
        if total_intervals else
        "Exactly 1-second percentage : N/A"
    )

    print(f"Gaps over 1 second           : {total_irregular}")
    print(f"Non-increasing intervals     : {total_non_increasing}")

    print("\nGenerated report:")
    print(output_path)


if __name__ == "__main__":
    main()
