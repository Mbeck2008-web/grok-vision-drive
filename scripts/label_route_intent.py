#!/usr/bin/env python3
"""Write intent.jsonl hindsight turn commands from state.jsonl + a road graph.

Training / eval only. Refuses to overwrite gt.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.route_hindsight import label_sequence  # noqa: E402
from python.sensors.route_intent import Graph  # noqa: E402


def load_label_graph(path: Path, level: str | None) -> Graph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        name = level or next(iter(data["levels"]))
        return Graph.from_fixture(data, name)
    nodes = {str(k): (float(v[0]), float(v[1])) for k, v in data["nodes"].items()}
    edges = tuple((str(e["a"]), str(e["b"]), bool(e.get("oneway", False))) for e in data["edges"])
    return Graph(nodes=nodes, edges=edges)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hindsight junction commands for --route-cond")
    parser.add_argument("--state", required=True, help="state.jsonl from a recording")
    parser.add_argument("--graph", required=True, help="Road graph JSON (junctions_synthetic format OK)")
    parser.add_argument("--level", default=None, help="Level key when --graph has multiple levels")
    parser.add_argument("--out", required=True, help="intent.jsonl sidecar (not gt.jsonl)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.name == "gt.jsonl":
        print("label_route_intent: refusing to overwrite gt.jsonl", file=sys.stderr)
        return 2
    state = Path(args.state)
    graph = load_label_graph(Path(args.graph), args.level)
    rows = []
    with state.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    labels = label_sequence(graph, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ("follow", "left", "right", "straight", "mask")}
    reasons: dict[str, int] = {}
    junction_frames = 0
    with out.open("w", encoding="utf-8") as dest:
        for row in labels:
            cmd = str(row.get("cmd") or "follow")
            if cmd not in counts:
                cmd = "mask"
            counts[cmd] += 1
            if cmd in ("left", "right", "straight"):
                junction_frames += 1
            if cmd == "mask":
                why = str(row.get("reason") or "unknown")
                reasons[why] = reasons.get(why, 0) + 1
            dest.write(json.dumps(row, sort_keys=True) + "\n")
    parts = " ".join(f"{name}={counts[name]}" for name in ("follow", "left", "right", "straight", "mask"))
    line = f"[GVD] route-intent: {parts} junction_frames={junction_frames}"
    if reasons:
        line += " mask_reasons: " + " ".join(f"{name}={reasons[name]}" for name in sorted(reasons))
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
