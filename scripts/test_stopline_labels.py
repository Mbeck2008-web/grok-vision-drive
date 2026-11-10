#!/usr/bin/env python3
"""Stop-line labels are separate from lane_gt. No planner, no viz, no GPU."""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.debug.painted_lanes import HEADING_MAX_DEG, LOOKAHEADS_M, PaintedLanes  # noqa: E402

CAR = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
HALF = 1.75


def _line(pts, material: str = "line_white") -> dict:
    return {"material": material, "nodes": [[float(x), float(y), 0.0, 0.15] for x, y in pts]}


def _through_road() -> list[dict]:
    ys = list(np.arange(-10.0, 40.1, 2.0))
    return [
        _line([(-HALF, y) for y in ys]),
        _line([(HALF, y) for y in ys]),
    ]


def check_isolation() -> None:
    banned = ("label_stoplines", "stopline")
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
    ]
    targets.extend(sorted((ROOT / "python" / "perception").glob("*.py")))
    targets.extend(sorted((ROOT / "python" / "viz").glob("*.py")))
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
                assert "label_stoplines" not in low, f"{path.name} imports {name}"
                if path.parent.name in {"planning", "control", "perception", "viz"}:
                    assert "stopline" not in low, f"{path.name} imports {name}"
    print(f"  isolation: {len(targets)} runtime files do not import the stopline writer")


def check_heading_roles() -> None:
    from python.debug.painted_lanes import paint_heading_role

    assert paint_heading_role(0.0) == "lane"
    assert paint_heading_role(HEADING_MAX_DEG) == "lane"
    assert paint_heading_role(50.0) == "mask"
    assert paint_heading_role(60.0) == "stop_line"
    assert paint_heading_role(90.0) == "stop_line"
    print("  heading roles: <=45 lane, 45-60 mask, >=60 stop_line")


def check_through_road_lane_gt_unchanged() -> None:
    paint = PaintedLanes(_through_road())
    lanes = paint.lane_at(*CAR)
    assert len(lanes) == len(LOOKAHEADS_M)
    for y, got in zip(LOOKAHEADS_M, lanes):
        assert got is not None, (y, lanes)
        assert abs(got[0] + HALF) < 0.1 and abs(got[1] - HALF) < 0.1, (y, got)
    bars = paint.stop_lines_ahead(*CAR)
    assert bars == [], bars
    print("  through-road: lane_at unchanged, no stop_line")


def check_stop_bar_is_labelled_not_a_lane() -> None:
    decals = _through_road() + [_line([(-2.0, 15.0), (2.0, 15.0)])]
    paint = PaintedLanes(decals)
    lanes = paint.lane_at(*CAR)
    for y, got in zip(LOOKAHEADS_M, lanes):
        assert got is not None, (y, lanes)
        assert abs(got[0] + HALF) < 0.1 and abs(got[1] - HALF) < 0.1, (y, got)
    bars = paint.stop_lines_ahead(*CAR)
    assert len(bars) == 1, bars
    bar = bars[0]
    assert bar["src"] == "stop_line"
    assert abs(float(bar["y_m"]) - 15.0) < 0.3, bar
    assert bar.get("valid") is True
    print("  stop bar at y=15 is src=stop_line and is not scored as a lane")


def check_diagonal_guide_is_masked() -> None:
    # ~50 deg off +y: dy, dx such that angle from heading is 50 deg
    ang = math.radians(50.0)
    # segment direction (sin 50, cos 50) relative to +y? heading +y=(0,1),
    # a direction at 50 deg is (sin 50, cos 50)
    dx, dy = math.sin(ang), math.cos(ang)
    decals = _through_road() + [_line([(dx * t, 20.0 + dy * t) for t in (-3.0, 3.0)])]
    paint = PaintedLanes(decals)
    lanes = paint.lane_at(*CAR)
    assert lanes[0] is not None
    bars = paint.stop_lines_ahead(*CAR)
    assert bars == [], bars
    masked = paint.masked_cross_paint(*CAR)
    assert masked, masked
    assert all(row["src"] == "mask" for row in masked), masked
    print("  ~50 deg crossing paint is mask, not a lane and not a stop_line")


def check_writer_refuses_gt_jsonl() -> None:
    import label_stoplines as ls

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "state.jsonl").write_text("{}\n", encoding="utf-8")
        code = ls.main([str(root), "--out", "gt.jsonl"])
        assert code == 2
    print("  writer refuses to overwrite gt.jsonl")


def check_cli_sidecar() -> None:
    import label_stoplines as ls

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state = root / "state.jsonl"
        # One synthetic pose. The CLI labels from --decals JSON, not a level zip.
        state.write_text(
            json.dumps({"t": 1.0, "map": "synth", "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}) + "\n",
            encoding="utf-8",
        )
        decals = _through_road() + [_line([(-2.0, 15.0), (2.0, 15.0)])]
        decal_path = root / "decals.json"
        decal_path.write_text(json.dumps(decals), encoding="utf-8")
        out = root / "stoplines.jsonl"
        code = ls.main([str(root), "--decals", str(decal_path), "--out", out.name])
        assert code == 0, code
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows and rows[0]["t"] == 1.0
        found = [b for b in rows[0]["bars"] if b["src"] == "stop_line"]
        assert found and abs(float(found[0]["y_m"]) - 15.0) < 0.3, rows
        assert not out.with_name("gt.jsonl").exists()
    print("  CLI writes stoplines.jsonl beside state and leaves gt.jsonl alone")


def main() -> None:
    check_isolation()
    check_heading_roles()
    check_through_road_lane_gt_unchanged()
    check_stop_bar_is_labelled_not_a_lane()
    check_diagonal_guide_is_masked()
    check_writer_refuses_gt_jsonl()
    check_cli_sidecar()
    print("test_stopline_labels: OK")


if __name__ == "__main__":
    main()
