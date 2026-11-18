
import csv

from config import TRAIN_DIR, TEST_DIR, REPORT_DIR


def analyse_network(record_dir):
    record_id = record_dir.name

    nodes_file = record_dir / f"{record_id}.nodes"
    arcs_file = record_dir / f"{record_id}.arcs"

    if not nodes_file.exists() or not arcs_file.exists():
        return None

    # Load geographical nodes.
    node_count = 0

    with nodes_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:
        for line in file:
            if line.strip():
                values = line.split()

                if len(values) != 2:
                    raise ValueError(
                        f"Malformed node in {nodes_file}: {line}"
                    )

                float(values[0])
                float(values[1])
                node_count += 1

    # Build graph structures.
    undirected = [set() for _ in range(node_count)]
    in_degree = [0] * node_count
    out_degree = [0] * node_count

    arc_count = 0
    valid_arcs = 0
    invalid_arcs = 0
    self_loops = 0
    duplicate_arcs = 0

    seen_arcs = set()

    with arcs_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line in file:
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 2:
                raise ValueError(
                    f"Malformed arc in {arcs_file}: {line}"
                )

            from_node = int(values[0])
            to_node = int(values[1])

            arc_count += 1

            if not (
                0 <= from_node < node_count
                and 0 <= to_node < node_count
            ):
                invalid_arcs += 1
                continue

            valid_arcs += 1

            if from_node == to_node:
                self_loops += 1

            edge = (from_node, to_node)

            if edge in seen_arcs:
                duplicate_arcs += 1

            seen_arcs.add(edge)

            out_degree[from_node] += 1
            in_degree[to_node] += 1

            undirected[from_node].add(to_node)
            undirected[to_node].add(from_node)

    # Identify isolated nodes.
    isolated_nodes = sum(
        in_degree[i] == 0 and out_degree[i] == 0
        for i in range(node_count)
    )

    # Find weakly connected components.
    visited = set()
    component_sizes = []

    for start_node in range(node_count):

        if start_node in visited:
            continue

        stack = [start_node]
        visited.add(start_node)
        component_size = 0

        while stack:
            current = stack.pop()
            component_size += 1

            for neighbour in undirected[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        component_sizes.append(component_size)

    component_sizes.sort(reverse=True)

    return {
        "record_id": record_id,
        "node_count": node_count,
        "arc_count": arc_count,
        "valid_arcs": valid_arcs,
        "invalid_arcs": invalid_arcs,
        "self_loops": self_loops,
        "duplicate_arcs": duplicate_arcs,
        "isolated_nodes": isolated_nodes,
        "weak_component_count": len(component_sizes),
        "largest_component_size": (
            component_sizes[0] if component_sizes else 0
        ),
        "smallest_component_size": (
            component_sizes[-1] if component_sizes else 0
        ),
        "average_in_degree": (
            sum(in_degree) / node_count if node_count else 0
        ),
        "average_out_degree": (
            sum(out_degree) / node_count if node_count else 0
        ),
        "maximum_in_degree": max(in_degree, default=0),
        "maximum_out_degree": max(out_degree, default=0),
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

            result = analyse_network(record_dir)

            if result is not None:
                result["split"] = split_name
                results.append(result)

                print(
                    f"[PROCESSED] {split_name}: "
                    f"{record_dir.name}"
                )

    if not results:
        print("[ERROR] No road networks found.")
        return

    output_path = REPORT_DIR / "network_analysis.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== NETWORK ANALYSIS ==========")
    print(f"Networks analysed : {len(results)}")
    print(f"Total nodes       : {sum(r['node_count'] for r in results)}")
    print(f"Total arcs        : {sum(r['arc_count'] for r in results)}")
    print(f"Valid arcs        : {sum(r['valid_arcs'] for r in results)}")
    print(f"Invalid arcs      : {sum(r['invalid_arcs'] for r in results)}")
    print(f"Isolated nodes    : {sum(r['isolated_nodes'] for r in results)}")
    print(
        f"Weak components  : "
        f"{sum(r['weak_component_count'] for r in results)}"
    )

    print("\nGenerated report:")
    print(output_path)


if __name__ == "__main__":
    main()
