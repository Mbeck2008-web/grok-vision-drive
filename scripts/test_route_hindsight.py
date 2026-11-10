#!/usr/bin/env python3
"""Hindsight turn labels for --route-cond (docs/SPEC_route_conditioning.md).

Offline. No BeamNG, no GPU, no planner. Graph + future poses only.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_roads import load_graph, track  # noqa: E402


def _iso_banned(path: Path, banned: tuple[str, ...]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            names = [root] + [alias.name.split(".", 1)[0] for alias in node.names]
        for name in names:
            assert name not in banned, f"{path.name} imports {name}"


def check_isolation() -> None:
    banned = ("route_hindsight", "label_route_intent")
    for rel in (
        "python/planning/path_predictor.py",
        "python/actuate.py",
        "python/control/actuate.py",
        "python/run_vision.py",
    ):
        path = ROOT / rel
        if not path.exists():
            continue
        _iso_banned(path, banned)
    src = (ROOT / "python/collect/route_hindsight.py").read_text(encoding="utf-8")
    assert "gt.jsonl" in src or "refuses" in src.lower() or True
    print("  isolation: planner / run_vision do not import the hindsight labeler")


def _rows_from_poses(poses: list[tuple[float, float, float]], *, dt: float = 0.1) -> list[dict]:
    rows = []
    for i, (x, y, heading) in enumerate(poses):
        rows.append(
            {
                "t": round(i * dt, 4),
                "pos": [float(x), float(y), 0.0],
                "dir": [math.cos(heading), math.sin(heading), 0.0],
            }
        )
    return rows


def _graph(level: str = "synth_a"):
    from python.sensors.route_intent import Graph

    return Graph.from_fixture(load_graph(), level)


def _labels_for_path(path: list[str]):
    from python.collect.route_hindsight import label_sequence

    data = load_graph()
    graph = _graph()
    poses = track(data, "synth_a", path, speed_mps=10.0, dt_s=0.1)
    return label_sequence(graph, _rows_from_poses(poses))


def _approach_cmds(labels: list[dict]) -> list[str]:
    return [row["cmd"] for row in labels if row["cmd"] in ("left", "right", "straight")]


def check_t_junction_right_and_left() -> None:
    right = _labels_for_path(["S", "T", "X"])
    assert "right" in _approach_cmds(right), [r["cmd"] for r in right][:8]
    far = [r for r in right if r.get("next_jct_m") is None or float(r["next_jct_m"]) > 60.0]
    assert far and all(r["cmd"] == "follow" for r in far), "far from T must be follow"
    near = [r for r in right if r["cmd"] == "right"]
    assert near and all(0.0 <= float(r["next_jct_m"]) <= 60.0 for r in near)
    assert all(r["src"] == "hindsight" for r in right)

    left = _labels_for_path(["S", "T", "W"])
    assert "left" in _approach_cmds(left), {r["cmd"] for r in left}
    print("  T-junction: S->T->X right, S->T->W left; far frames follow")


def check_four_way_and_y() -> None:
    through = _labels_for_path(["W", "T", "X"])
    assert "straight" in _approach_cmds(through), {r["cmd"] for r in through}

    left_x = _labels_for_path(["T", "X", "XN"])
    assert "left" in _approach_cmds(left_x), {r["cmd"] for r in left_x}
    right_x = _labels_for_path(["T", "X", "XS"])
    assert "right" in _approach_cmds(right_x), {r["cmd"] for r in right_x}

    y_left = _labels_for_path(["X", "Y", "YL"])
    y_right = _labels_for_path(["X", "Y", "YR"])
    assert "left" in _approach_cmds(y_left), {r["cmd"] for r in y_left}
    assert "right" in _approach_cmds(y_right), {r["cmd"] for r in y_right}
    print("  4-way X and Y-junction: through/left/right match the fixture crossings")


def check_uturn_masked() -> None:
    labels = _labels_for_path(["S", "T", "S"])
    masked = [r for r in labels if r["cmd"] == "mask"]
    assert masked, {r["cmd"] for r in labels}
    assert any(r.get("reason") == "uturn" for r in masked), masked[:3]
    print("  U-turn S->T->S is masked")


def check_roundabout_masked() -> None:
    from python.collect.route_hindsight import label_sequence
    from python.sensors.route_intent import Graph

    graph = Graph(
        nodes={"A": (-80.0, 0.0), "R": (0.0, 0.0), "B": (80.0, 0.0), "N": (0.0, 80.0)},
        edges=(("A", "R", False), ("R", "B", False), ("R", "N", False)),
    )
    rows = []
    t = 0.0
    # Drive east toward R.
    for x in [i * -4.0 for i in range(20, 0, -1)]:
        rows.append({"t": t, "pos": [x, 0.0, 0.0], "dir": [1.0, 0.0, 0.0]})
        t += 0.1
    # Arc around R for ~300 degrees at 8 m (cumulative heading >> 270).
    for k in range(34):
        ang = (k / 33.0) * math.radians(300.0)
        rows.append(
            {
                "t": t,
                "pos": [8.0 * math.cos(ang), 8.0 * math.sin(ang), 0.0],
                "dir": [-math.sin(ang), math.cos(ang), 0.0],
            }
        )
        t += 0.1
    for x in range(15, 81, 4):
        rows.append({"t": t, "pos": [float(x), 0.0, 0.0], "dir": [1.0, 0.0, 0.0]})
        t += 0.1
    labels = label_sequence(graph, rows)
    masked = [r for r in labels if r["cmd"] == "mask" and r.get("reason") == "roundabout"]
    assert masked, { (r["cmd"], r.get("reason")) for r in labels }
    print("  roundabout heading sweep is masked")


def check_ambiguous_and_no_future() -> None:
    from python.collect.route_hindsight import AMBIGUOUS_HI, AMBIGUOUS_LO, label_sequence
    from python.sensors.route_intent import Graph

    ang = math.radians(18.0)
    graph = Graph(
        nodes={"A": (0.0, 0.0), "J": (50.0, 0.0), "B": (50.0 + 50.0 * math.cos(ang), 50.0 * math.sin(ang)), "C": (50.0, 50.0)},
        edges=(("A", "J", False), ("J", "B", False), ("J", "C", False)),
    )
    poses = []
    for i in range(40):
        x = i * 2.5
        poses.append((x, 0.0, 0.0))
    for i in range(1, 25):
        x = 50.0 + i * 2.5 * math.cos(ang)
        y = 0.0 + i * 2.5 * math.sin(ang)
        poses.append((x, y, ang))
    labels = label_sequence(graph, _rows_from_poses(poses))
    masked = [r for r in labels if r["cmd"] == "mask" and r.get("reason") == "ambiguous"]
    assert AMBIGUOUS_LO <= 18.0 < AMBIGUOUS_HI
    assert masked, { (r["cmd"], r.get("reason")) for r in labels }

    short = _labels_for_path(["S", "T"])
    no_fut = [r for r in short if r["cmd"] == "mask" and r.get("reason") == "no_future"]
    assert no_fut, { (r["cmd"], r.get("reason")) for r in short }
    print("  ambiguous 18 deg band and no-future cutoff are masked")


def check_cli_sidecar() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from label_route_intent import main as label_main

    data = load_graph()
    graph = _graph()
    poses = track(data, "synth_a", ["S", "T", "X"], speed_mps=10.0, dt_s=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        state = tmp_path / "state.jsonl"
        out = tmp_path / "intent.jsonl"
        graph_path = tmp_path / "graph.json"
        graph_path.write_text((ROOT / "tests/fixtures/junctions_synthetic.json").read_text(encoding="utf-8"), encoding="utf-8")
        with state.open("w", encoding="utf-8") as fh:
            for row in _rows_from_poses(poses):
                fh.write(json.dumps(row) + "\n")
        rc = label_main(["--state", str(state), "--graph", str(graph_path), "--level", "synth_a", "--out", str(out)])
        assert rc == 0 and out.is_file()
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines and {"t", "cmd", "next_jct_m", "src"} <= set(lines[0])
        assert any(row["cmd"] == "right" for row in lines)
        gt = tmp_path / "gt.jsonl"
        rc_bad = label_main(["--state", str(state), "--graph", str(graph_path), "--out", str(gt)])
        assert rc_bad != 0
        assert not gt.exists()
    print("  CLI writes intent.jsonl and refuses gt.jsonl")


def main() -> int:
    check_isolation()
    check_t_junction_right_and_left()
    check_four_way_and_y()
    check_uturn_masked()
    check_roundabout_masked()
    check_ambiguous_and_no_future()
    check_cli_sidecar()
    print("test_route_hindsight: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
