import argparse
import json
import re
from pathlib import Path


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_layer_numbers(directory, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.jsonl$")
    layer_numbers = []
    for path in directory.glob(f"{prefix}_*.jsonl"):
        match = pattern.match(path.name)
        if match:
            layer_numbers.append(int(match.group(1)))
    return sorted(layer_numbers)


def load_winner_flags(layer_path, candidate):
    flags = {}
    for record in iter_jsonl(layer_path):
        node_id = record.get("node_id")
        winners = record.get("winner_to_cand")
        if winners is None:
            winners = record.get("winner_to_can", [])
        flags[node_id] = candidate in (winners or [])
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Report layer sizes and candidate-election edge counts for a streamed Meek graph."
    )
    parser.add_argument(
        "--stream-dir",
        type=Path,
        default=Path("data/meek_autosave/D4_profile_stream"),
        help="Path to the *_stream directory.",
    )
    parser.add_argument(
        "--candidate",
        type=int,
        default=7,
        help="Candidate id to detect transitions into winner_to_cand.",
    )
    args = parser.parse_args()

    layers_dir = args.stream_dir / "layers"
    edges_dir = args.stream_dir / "edges"

    layer_numbers = parse_layer_numbers(layers_dir, "layer")
    if not layer_numbers:
        raise SystemExit(f"No layer files found in {layers_dir}.")

    print("Layer vertex counts:")
    for layer in layer_numbers:
        layer_path = layers_dir / f"layer_{layer}.jsonl"
        count = count_jsonl(layer_path)
        print(f"layer {layer}: {count}")

    edge_layers = parse_layer_numbers(edges_dir, "edges")
    if not edge_layers:
        raise SystemExit(f"No edge files found in {edges_dir}.")

    total_edges = 0
    flags_cache = {}
    cache_order = []

    def get_flags(layer):
        if layer in flags_cache:
            return flags_cache[layer]
        layer_path = layers_dir / f"layer_{layer}.jsonl"
        if not layer_path.exists():
            return None
        flags = load_winner_flags(layer_path, args.candidate)
        flags_cache[layer] = flags
        cache_order.append(layer)
        if len(cache_order) > 2:
            old = cache_order.pop(0)
            flags_cache.pop(old, None)
        return flags

    print(f"Edges electing candidate {args.candidate}:")
    for layer in edge_layers:
        edges_path = edges_dir / f"edges_{layer}.jsonl"
        from_flags = get_flags(layer)
        to_flags = get_flags(layer + 1)
        if from_flags is None or to_flags is None:
            continue
        for record in iter_jsonl(edges_path):
            from_id = record.get("from")
            to_id = record.get("to")
            if from_id is None or to_id is None:
                continue
            if not from_flags.get(from_id, False) and to_flags.get(to_id, False):
                total_edges += 1

    print(total_edges)


if __name__ == "__main__":
    main()
