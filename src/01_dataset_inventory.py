
import csv
import json

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR

EXPECTED_FILES = [".arcs", ".nodes", ".route", ".track"]


def count_rows(file_path):
    """Count non-empty lines in a file."""
    if not file_path.exists():
        return 0

    with file_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:
        return sum(1 for line in file if line.strip())


def inspect_split(split_name, split_dir):
    records = []

    if not split_dir.exists():
        print(f"[WARNING] Directory not found: {split_dir}")
        return records

    record_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir()
    )

    for record_dir in record_dirs:
        record = {
            "split": split_name,
            "record_id": record_dir.name,
        }

        missing_files = []

        for extension in EXPECTED_FILES:
            file_path = record_dir / f"{record_dir.name}{extension}"
            exists = file_path.is_file()

            file_key = extension[1:]

            record[f"{file_key}_exists"] = exists
            record[f"{file_key}_size_bytes"] = (
                file_path.stat().st_size if exists else 0
            )
            record[f"{file_key}_rows"] = (
                count_rows(file_path) if exists else 0
            )

            if not exists:
                missing_files.append(extension)

        record["missing_files"] = ";".join(missing_files)
        record["record_complete"] = not missing_files

        records.append(record)

    return records


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = (
        inspect_split("train", TRAIN_DIR)
        + inspect_split("student_test", TEST_DIR)
    )

    if not all_records:
        print("[ERROR] No record directories found.")
        return

    csv_path = REPORT_DIR / "dataset_inventory.csv"
    json_path = REPORT_DIR / "dataset_inventory.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=all_records[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_records)

    summary = {
        "total_records": len(all_records),
        "training_records": sum(
            r["split"] == "train" for r in all_records
        ),
        "student_test_records": sum(
            r["split"] == "student_test" for r in all_records
        ),
        "complete_records": sum(
            r["record_complete"] for r in all_records
        ),
        "incomplete_records": sum(
            not r["record_complete"] for r in all_records
        ),
        "records": all_records,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== DATASET INVENTORY ==========")
    print(f"Total records        : {summary['total_records']}")
    print(f"Training records     : {summary['training_records']}")
    print(f"Student test records : {summary['student_test_records']}")
    print(f"Complete records     : {summary['complete_records']}")
    print(f"Incomplete records   : {summary['incomplete_records']}")

    print("\nGenerated reports:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
