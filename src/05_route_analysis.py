
import csv

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR


def load_arcs(arcs_file):
    """
    Load road-network arcs.
    The zero-based row index represents arc_id.
    """

    arcs = []

    with arcs_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 2:
                raise ValueError(
                    f"Invalid arc format at line {line_number}"
                )

            from_node = int(values[0])
            to_node = int(values[1])

            arcs.append((from_node, to_node))

    return arcs


def load_route(route_file):
    """Load chronological ground-truth arc IDs."""

    route_ids = []
    malformed_rows = 0

    with route_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 1:
                malformed_rows += 1
                continue

            try:
                route_ids.append(int(values[0]))
            except ValueError:
                malformed_rows += 1

    return route_ids, malformed_rows


def analyse_route(arcs, route_ids):
    """Validate route references and arc-to-arc continuity."""

    invalid_ids = [
        arc_id for arc_id in route_ids
        if arc_id < 0 or arc_id >= len(arcs)
    ]

    valid_ids = [
        arc_id for arc_id in route_ids
        if 0 <= arc_id < len(arcs)
    ]

    continuity_checks = 0
    continuity_breaks = 0

    # Check only genuinely consecutive route entries.
    # Do not bridge over invalid arc IDs.
    for i in range(1, len(route_ids)):

        previous_id = route_ids[i - 1]
        current_id = route_ids[i]

        if not (
            0 <= previous_id < len(arcs)
            and 0 <= current_id < len(arcs)
        ):
            continue

        continuity_checks += 1

        previous_destination = arcs[previous_id][1]
        current_origin = arcs[current_id][0]

        if previous_destination != current_origin:
            continuity_breaks += 1

    unique_arc_count = len(set(route_ids))

    return {
        "route_length": len(route_ids),
        "valid_arc_references": len(valid_ids),
        "invalid_arc_references": len(invalid_ids),
        "continuity_checks": continuity_checks,
        "continuity_breaks": continuity_breaks,
        "continuity_percentage": (
            (continuity_checks - continuity_breaks)
            / continuity_checks * 100
            if continuity_checks else 0
        ),
        "unique_arc_ids": unique_arc_count,
        "repeated_arc_occurrences": (
            len(route_ids) - unique_arc_count
        ),
        "first_arc_id": route_ids[0] if route_ids else None,
        "last_arc_id": route_ids[-1] if route_ids else None,
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

            record_id = record_dir.name

            arcs_file = record_dir / f"{record_id}.arcs"
            route_file = record_dir / f"{record_id}.route"

            if not arcs_file.is_file() or not route_file.is_file():
                print(f"[WARNING] Missing files: {record_id}")
                continue

            arcs = load_arcs(arcs_file)

            route_ids, malformed_rows = load_route(route_file)

            stats = analyse_route(arcs, route_ids)

            results.append({
                "split": split_name,
                "record_id": record_id,
                "arc_count": len(arcs),
                "malformed_route_rows": malformed_rows,
                **stats,
            })

            print(
                f"[PROCESSED] {split_name}: {record_id}"
            )

    if not results:
        print("[ERROR] No route records found.")
        return

    output_path = REPORT_DIR / "route_analysis.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== ROUTE ANALYSIS ==========")
    print(f"Trips analysed           : {len(results)}")

    print(
        f"Total route arc IDs      : "
        f"{sum(r['route_length'] for r in results)}"
    )

    print(
        f"Invalid arc references   : "
        f"{sum(r['invalid_arc_references'] for r in results)}"
    )

    print(
        f"Malformed route rows     : "
        f"{sum(r['malformed_route_rows'] for r in results)}"
    )

    print(
        f"Continuity checks        : "
        f"{sum(r['continuity_checks'] for r in results)}"
    )

    print(
        f"Continuity breaks        : "
        f"{sum(r['continuity_breaks'] for r in results)}"
    )

    print(
        f"Repeated arc occurrences : "
        f"{sum(r['repeated_arc_occurrences'] for r in results)}"
    )

    print("\nGenerated report:")
    print(output_path)


if __name__ == "__main__":
    main()
