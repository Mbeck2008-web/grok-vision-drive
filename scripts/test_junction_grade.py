#!/usr/bin/env python3
"""Offline junction grader on the hand-built synthetic graph. No BeamNG."""
from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import junction_grade as jg  # noqa: E402
from synthetic_roads import FROZEN_PATH, GRAPH_PATH, load_graph, network_spec, track  # noqa: E402

BAR_TEXT = "turn match ≥95% over N ≥10 junctions; with N=10 this is 10/10\n"


def _rows(poses: list[tuple[float, float, float]]) -> list[dict]:
    return [{"pos": [x, y, 0.5], "heading_rad": h} for x, y, h in poses]


def _at(found: list[dict], node: str) -> dict:
    hits = [c for c in found if c["node"] == node]
    assert len(hits) == 1, (node, found)
    return hits[0]


def check_classes(graph_kind: str) -> None:
    data = load_graph()
    if graph_kind == "fixture":
        graph = jg.graph_from_fixture(data, "synth_a")
        name = {n: n for n in data["levels"]["synth_a"]["nodes"]}
    else:
        # The road-dump path: rebuild nodes from a raw network's road ends.
        graph = jg.graph_from_network(network_spec(data, "synth_a"))
        name = {}
        for fixture_name, (fx, fy) in data["levels"]["synth_a"]["nodes"].items():
            for node, (x, y) in graph.nodes.items():
                if abs(x - fx) < 1.0 and abs(y - fy) < 1.0:
                    name[fixture_name] = node
    junctions = {name[n] for n in data["levels"]["synth_a"]["junctions"]}
    assert set(graph.junctions()) == junctions, (graph_kind, graph.junctions(), junctions)
    for case in data["crossings"]:
        found = jg.crossings(_rows(track(data, "synth_a", case["path"])), graph)
        got = _at(found, name[case["node"]])
        kind = data["levels"]["synth_a"]["junctions"][case["node"]]
        label = f"{graph_kind} {kind} {'->'.join(case['path'])}"
        assert got["log"] == case["expect"], (label, got)
        assert got["graph"] == case["expect"], (label, got)
        assert got["match"] and not got["wrong_way"], (label, got)
        print(f"  {label}: log {got['log']}, graph {got['graph']} ({got['turn_deg']:+.1f} deg)")


def check_oneway_and_routes() -> None:
    data = load_graph()
    graph = jg.graph_from_fixture(data, "synth_a")
    a, b = data["levels"]["synth_a"]["oneway"]["edge"]
    assert jg.route(graph, "T", b)["nodes"] == ["T", a, b]
    assert jg.route(graph, b, "T")["status"] == "no_route", "routed against the one-way edge"
    assert jg.route(graph, "S", "YL")["nodes"] == ["S", "T", "X", "Y", "YL"]
    # Driving the one-way edge backwards is graded wrong-way, not a match.
    found = jg.crossings(_rows(track(data, "synth_a", [b, a, "Y"])), graph)
    got = _at(found, a)
    assert got["wrong_way"] and not got["match"], got
    unreachable = data["levels"]["synth_a"]["unreachable"]
    for src, dst in (("S", unreachable), (unreachable, "S"), ("S", "no_such_node")):
        out = jg.route(graph, src, dst)
        assert out == {"status": "no_route", "nodes": [], "length_m": None}, (src, dst, out)
    print("  one-way honoured, unreachable node -> no_route")


def check_frozen_log() -> None:
    """The frozen GT log (W -> T -> X -> Y -> YL) grades through, through, left."""
    data = load_graph()
    rows = [json.loads(line) for line in gzip.decompress(FROZEN_PATH.read_bytes()).splitlines()]
    found = jg.crossings(rows, jg.graph_from_fixture(data, "synth_a"))
    assert [(c["node"], c["log"], c["match"]) for c in found] == [
        ("T", "through", True),
        ("X", "through", True),
        ("Y", "left", True),
    ], found
    print("  frozen GT log: T through, X through, Y left")


def check_cli_and_bar() -> None:
    data = load_graph()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = []
        for _ in range(2):
            for case in data["crossings"]:
                rows.extend(_rows(track(data, "synth_a", case["path"])))
        gt = tmp_path / "gt.jsonl"
        gt.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        bar = tmp_path / "bar.txt"
        bar.write_text(BAR_TEXT, encoding="utf-8")
        sha = hashlib.sha256(bar.read_bytes()).hexdigest()
        assert jg.parse_bar(BAR_TEXT) == (0.95, 10)
        base = ["--gt", str(gt), "--roads", str(GRAPH_PATH), "--level", "synth_a"]

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = jg.main(base + ["--bar", str(bar), "--bar-sha256", "0" * 64])
        assert code == 2 and "refusing" in err.getvalue(), (code, err.getvalue())
        assert f"bar file sha256 {sha}" in out.getvalue(), out.getvalue()

        out = io.StringIO()
        with redirect_stdout(out):
            code = jg.main(base + ["--bar", str(bar), "--bar-sha256", sha, "--json", str(tmp_path / "c.json")])
        text = out.getvalue()
        assert code == 0, text
        assert "turn match 16/16 (100.0%)" in text and "bar: rate >= 95.0% over N >= 10: PASS" in text, text
        assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["summary"]["junctions"] == 16

        # Too few junctions for the bar: FAIL, exit 1.
        short = tmp_path / "short.jsonl"
        short.write_text("".join(json.dumps(r) + "\n" for r in _rows(track(data, "synth_a", ["S", "T", "W"]))),
                         encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            code = jg.main(["--gt", str(short), "--roads", str(GRAPH_PATH), "--bar", str(bar), "--bar-sha256", sha])
        assert code == 1 and "FAIL" in out.getvalue(), out.getvalue()
    print("  CLI: wrong bar hash refused (exit 2), 16/16 meets the bar, 1 junction does not")


def check_runtime_isolation() -> None:
    """No runtime control module imports the grader or any navgraph code."""
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
        ROOT / "python" / "train" / "scene_lane_runtime.py",
    ]
    for path in targets:
        assert path.is_file(), f"{path.relative_to(ROOT)} moved; update the isolation scan"
    perception = sorted((ROOT / "python" / "perception").glob("*.py"))
    assert perception, "python/perception has no modules; update the isolation scan"
    targets.extend(perception)
    banned = ("junction_grade", "navgraph", "navigraph", "beamngpy.tools")
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            for name in names:
                low = name.lower()
                assert not any(b in low for b in banned), f"{path.relative_to(ROOT)} imports {name}"
    print(f"  isolation: {len(targets)} runtime files import no grader or navgraph code")


def main() -> None:
    check_classes("fixture")
    check_classes("road network")
    check_oneway_and_routes()
    check_frozen_log()
    check_cli_and_bar()
    check_runtime_isolation()
    print("test_junction_grade: OK")


if __name__ == "__main__":
    main()
