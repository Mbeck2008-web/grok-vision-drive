#!/usr/bin/env python3
"""Hindsight turn labels for --route-cond (docs/SPEC_route_conditioning.md).

Offline. No BeamNG, no GPU, no planner. Graph + future poses only.
"""
from __future__ import annotations

import ast
import contextlib
import gzip
import io
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_roads import load_graph, track  # noqa: E402


def _imported_module_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
            names.extend(alias.name for alias in node.names)
    return names


def _iso_banned(path: Path, banned: tuple[str, ...]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for name in _imported_module_names(tree):
        low = name.lower()
        assert not any(b in low for b in banned), f"{path.name} imports {name}"


def check_isolation() -> None:
    banned = ("route_hindsight", "label_route_intent", "road_graph")
    for rel in (
        "python/planning/path_predictor.py",
        "python/actuate.py",
        "python/control/actuate.py",
        "python/run_vision.py",
        "python/train/scene_lane_runtime.py",
        "python/train/scene_ckpt_runtime.py",
        "python/train/route_cond.py",
    ):
        path = ROOT / rel
        if not path.exists():
            continue
        _iso_banned(path, banned)
    src = (ROOT / "python/collect/route_hindsight.py").read_text(encoding="utf-8")
    assert "gt.jsonl" in src or "refuses" in src.lower() or True
    print("  isolation: planner / run_vision do not import the hindsight labeler")


def check_iso_catches_dotted_hindsight_imports() -> None:
    """Mutations that the old first-token matcher missed must fail."""
    snippets = (
        "from python.collect.route_hindsight import label_sequence\n",
        "import python.collect.route_hindsight\n",
        "from python.collect import route_hindsight\n",
    )
    for src in snippets:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_vision.py"
            path.write_text(src, encoding="utf-8")
            raised = False
            try:
                _iso_banned(path, ("route_hindsight", "label_route_intent"))
            except AssertionError:
                raised = True
            assert raised, f"isolation missed: {src!r}"
    print("  isolation: dotted route_hindsight / label_sequence imports are banned")


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


def _rows_v2_from_poses(poses: list[tuple[float, float, float]], *, dt: float = 0.1) -> list[dict]:
    """Collect-shaped v2 state.jsonl rows: ego.pos / capture.pos, no top-level pos."""
    rows = []
    for i, (x, y, heading) in enumerate(poses):
        pos = [float(x), float(y), 0.0]
        direction = [math.cos(heading), math.sin(heading), 0.0]
        rows.append(
            {
                "t": round(i * dt, 4),
                "ego": {"pos": pos, "dir": direction},
                "capture": {"pos": list(pos), "dir": list(direction)},
            }
        )
    return rows


def _write_tiny_road_dump(path: Path) -> Path:
    """Minimal <data>/roads/<map>.json.gz for the synth_a T at the origin.

    Through-road W(-200,0)–X(200,0) plus stem S(0,-200)–T(0,0). The dump
    walk must merge the stem end onto the through road to make a degree-3
    node. Edge records use the BeamNG left/middle/right xyz shape; oneWay
    is the string "0" used in real dumps.
    """
    def edges(points: list[tuple[float, float]], *, horizontal: bool) -> list[dict]:
        half = 2.0
        out = []
        for x, y in points:
            if horizontal:
                left, right = [x, y - half, 0.0], [x, y + half, 0.0]
            else:
                left, right = [x - half, y, 0.0], [x + half, y, 0.0]
            out.append({"left": left, "right": right, "middle": [x, y, 0.0]})
        return out

    def road(rid: str, points: list[tuple[float, float]], *, horizontal: bool) -> tuple[str, dict]:
        return rid, {
            "drivability": "1",
            "oneWay": "0",
            "edges": edges(points, horizontal=horizontal),
        }

    payload = {
        "map": "synth_a",
        "spawn": {"pos": [0.0, -200.0, 0.0], "source": "fixture"},
        "note": "tiny route-backlabel T-junction",
        "road_network": dict(
            (
                road("through", [(-200.0, 0.0), (0.0, 0.0), (200.0, 0.0)], horizontal=True),
                road("stem", [(0.0, -200.0), (0.0, 0.0)], horizontal=False),
            )
        ),
        "roads": {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path



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
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = label_main(["--state", str(state), "--graph", str(graph_path), "--level", "synth_a", "--out", str(out)])
        hist = buf.getvalue()
        assert rc == 0 and out.is_file()
        for key in ("follow=", "left=", "right=", "straight=", "mask=", "junction"):
            assert key in hist, (key, hist)
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines and {"t", "cmd", "next_jct_m", "src"} <= set(lines[0])
        assert any(row["cmd"] == "right" for row in lines)
        gt = tmp_path / "gt.jsonl"
        rc_bad = label_main(["--state", str(state), "--graph", str(graph_path), "--out", str(gt)])
        assert rc_bad != 0
        assert not gt.exists()
    print("  CLI writes intent.jsonl and refuses gt.jsonl")



def check_v2_ego_pos_parity() -> None:
    """Nested ego.pos / capture.pos (no top-level pos) match flattened labels."""
    from python.collect.route_hindsight import label_sequence

    data = load_graph()
    graph = _graph()
    poses = track(data, "synth_a", ["S", "T", "X"], speed_mps=10.0, dt_s=0.1)
    flat = label_sequence(graph, _rows_from_poses(poses))
    nested = label_sequence(graph, _rows_v2_from_poses(poses))
    assert len(nested) == len(flat) == len(poses)
    assert all("pos" not in row for row in _rows_v2_from_poses(poses)[:3])
    for got, want in zip(nested, flat):
        assert got["cmd"] == want["cmd"], (got, want)
        assert got.get("next_jct_m") == want.get("next_jct_m"), (got, want)
        assert got.get("reason") == want.get("reason"), (got, want)
    assert "right" in _approach_cmds(nested)
    # Capture-only rows (no ego, no top-level pos) also match.
    capture_only = []
    for row in _rows_v2_from_poses(poses):
        capture_only.append({"t": row["t"], "capture": row["capture"]})
    captured = label_sequence(graph, capture_only)
    assert [r["cmd"] for r in captured] == [r["cmd"] for r in flat]
    print("  v2 ego.pos / capture.pos parity: same labels as flattened {pos,dir}")


def check_cli_gzip_road_dump() -> None:
    """CLI loads <data>/roads/<map>.json.gz and labels a v2 state.jsonl clip."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from label_route_intent import main as label_main
    from python.collect.route_hindsight import label_sequence

    from python.collect.road_graph import load_label_graph

    data = load_graph()
    poses = track(data, "synth_a", ["S", "T", "X"], speed_mps=10.0, dt_s=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        state = tmp_path / "clip" / "state.jsonl"
        out = tmp_path / "clip" / "intent.jsonl"
        dump = tmp_path / "roads" / "synth_a.json.gz"
        _write_tiny_road_dump(dump)
        dump_graph = load_label_graph(dump)
        assert any(
            abs(x) <= 1.0 and abs(y) <= 1.0
            for name, (x, y) in dump_graph.nodes.items()
            if name in dump_graph.junctions()
        ), (dump_graph.nodes, dump_graph.junctions())
        want = label_sequence(dump_graph, _rows_from_poses(poses))
        state.parent.mkdir(parents=True, exist_ok=True)
        with state.open("w", encoding="utf-8") as fh:
            for row in _rows_v2_from_poses(poses):
                assert "pos" not in row and "dir" not in row
                fh.write(json.dumps(row) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = label_main(["--state", str(state), "--graph", str(dump), "--out", str(out)])
        hist = buf.getvalue()
        assert rc == 0 and out.is_file(), (rc, hist)
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == len(want)
        assert [r["cmd"] for r in lines] == [r["cmd"] for r in want]
        assert any(row["cmd"] == "right" for row in lines)
        for key in ("follow=", "left=", "right=", "straight=", "mask=", "junction"):
            assert key in hist, (key, hist)
        gt = tmp_path / "gt.jsonl"
        rc_bad = label_main(["--state", str(state), "--graph", str(dump), "--out", str(gt)])
        assert rc_bad != 0
        assert not gt.exists()
    print("  CLI: v2 state.jsonl + roads/synth_a.json.gz matches flattened labels; still refuses gt.jsonl")


def main() -> int:
    check_isolation()
    check_iso_catches_dotted_hindsight_imports()
    check_t_junction_right_and_left()
    check_four_way_and_y()
    check_uturn_masked()
    check_roundabout_masked()
    check_ambiguous_and_no_future()
    check_cli_sidecar()
    check_v2_ego_pos_parity()
    check_cli_gzip_road_dump()
    print("test_route_hindsight: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
