
import csv
import math

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR


def validate_file(file_path, expected_columns):
    stats = {
        "total_rows": 0,
        "valid_rows": 0,
        "malformed_rows": 0,
        "non_numeric_rows": 0,
        "non_finite_rows": 0,
        "invalid_longitude": 0,
        "invalid_latitude": 0,
        "invalid_timestamp": 0,
    }

    longitudes = []
    latitudes = []
    timestamps = []

    with file_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:
            if not line.strip():
                continue

            stats["total_rows"] += 1
            values = line.split()

            if len(values) != expected_columns:
                stats["malformed_rows"] += 1
                continue

            try:
                numbers = [float(value) for value in values]
            except ValueError:
                stats["non_numeric_rows"] += 1
                continue

            if not all(math.isfinite(value) for value in numbers):
                stats["non_finite_rows"] += 1
                continue

            longitude, latitude = numbers[0], numbers[1]

            if not -180 <= longitude <= 180:
                stats["invalid_longitude"] += 1

            if not -90 <= latitude <= 90:
                stats["invalid_latitude"] += 1

            if expected_columns == 3:
                timestamp = numbers[2]
                timestamps.append(timestamp)

                if timestamp < 0:
                    stats["invalid_timestamp"] += 1

            longitudes.append(longitude)
            latitudes.append(latitude)
            stats["valid_rows"] += 1

    if longitudes:
        stats["longitude_min"] = min(longitudes)
        stats["longitude_max"] = max(longitudes)
        stats["latitude_min"] = min(latitudes)
        stats["latitude_max"] = max(latitudes)
    else:
        stats["longitude_min"] = None
        stats["longitude_max"] = None
        stats["latitude_min"] = None
        stats["latitude_max"] = None

    # Check whether swapping longitude and latitude
    # would also satisfy the geographical ranges.
    stats["swapped_order_valid_rows"] = sum(
        -90 <= lon <= 90 and -180 <= lat <= 180
        for lon, lat in zip(longitudes, latitudes)
    )

    if timestamps:
        stats["timestamp_min"] = min(timestamps)
        stats["timestamp_max"] = max(timestamps)
    else:
        stats["timestamp_min"] = None
        stats["timestamp_max"] = None

    return stats


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    partitions = [
        ("train", TRAIN_DIR),
        ("student_test", TEST_DIR),
    ]

    for split_name, split_dir in partitions:

        if not split_dir.exists():
            print(f"[WARNING] Missing directory: {split_dir}")
            continue

        for record_dir in sorted(split_dir.iterdir()):

            if not record_dir.is_dir():
                continue

            for extension, expected_columns in [
                (".nodes", 2),
                (".track", 3),
            ]:

                file_path = (
                    record_dir / f"{record_dir.name}{extension}"
                )

                if not file_path.is_file():
                    continue

                stats = validate_file(
                    file_path, expected_columns
                )

                results.append({
                    "split": split_name,
                    "record_id": record_dir.name,
                    "file_type": extension[1:],
                    **stats,
                })

    if not results:
        print("[ERROR] No coordinate files found.")
        return

    output_path = REPORT_DIR / "coordinate_validation.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    totals = {
        key: sum(row[key] for row in results)
        for key in [
            "total_rows",
            "valid_rows",
            "malformed_rows",
            "non_numeric_rows",
            "non_finite_rows",
            "invalid_longitude",
            "invalid_latitude",
            "invalid_timestamp",
        ]
    }

    print("\n========== COORDINATE VALIDATION ==========")
    print(f"Files inspected       : {len(results)}")

    for key, value in totals.items():
        print(f"{key.replace('_', ' ').title():24}: {value}")

    print("\nGenerated report:")
    print(output_path)


if __name__ == "__main__":
    main()
