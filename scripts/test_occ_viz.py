#!/usr/bin/env python3
"""Sparse occ + SceneNet outputs round-trip into a valid C++ viz message.

Spec: docs/specs/occ_sparse.md ("Viz schema"). Perception outputs only, no GT.
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402


def _road_occ():
    """Drivable band |x| <= 3.5 m, y 0..20 m on the ground layer; one car cell; some free queries."""
    from python.viz.occ_scene import GRID_MIN, GRID_SHAPE, VOXEL_M

    ids, logits = [], []
    iz = int((0.0 - GRID_MIN[2]) / VOXEL_M) - 1  # z in [-0.5, 0)
    for iy in range(int((0 - GRID_MIN[1]) / VOXEL_M), int((20 - GRID_MIN[1]) / VOXEL_M)):
        for ix in range(int((-3.5 - GRID_MIN[0]) / VOXEL_M), int((3.5 - GRID_MIN[0]) / VOXEL_M)):
            ids.append(np.ravel_multi_index((ix, iy, iz), GRID_SHAPE))
            logits.append([4.0, -2.0, 3.0, -2.0])
    car = np.ravel_multi_index((int((2.0 - GRID_MIN[0]) / VOXEL_M), int((10 - GRID_MIN[1]) / VOXEL_M), iz + 2), GRID_SHAPE)
    ids.append(car)
    logits.append([4.0, -2.0, -2.0, 3.0])
    for j in range(30):  # free: never drawn
        ids.append(np.ravel_multi_index((j, 100, 7), GRID_SHAPE))
        logits.append([-4.0, 3.0, -2.0, -2.0])
    return np.asarray(ids), np.asarray(logits, np.float32)


def _pred(frame: int = 0, dt: float = 0.1, ego_v: float = 10.0):
    """Ego at 10 m/s. Five objects whose ego-frame tracks give the five legend states."""
    lanes = np.stack([np.stack([np.full(16, x), np.linspace(0, 30, 16)], 1) for x in (-1.8, 1.8, 5.4)])
    k = frame * dt
    tracks = [  # (x, y, yaw from +x; pi/2 = ego heading)
        (0.0, 20.0, math.pi / 2),                           # ground +10 m/s ahead: same_way
        (-3.6, 40.0 - 22.0 * k, -math.pi / 2),               # ground -12 m/s: oncoming
        (-4.0 + 6.0 * k, 15.0 - ego_v * k, 0.0),             # ground (6, 0): crossing
        (3.6, 1.0, math.pi / 2),                             # beside ego at ego speed: alongside
        (5.5, 30.0 - ego_v * k, math.pi / 2),                # ground 0: parked
    ]
    objects = np.zeros((5, 8), np.float32)
    for i, (x, y, yaw) in enumerate(tracks):
        objects[i, [0, 1, 2]] = (x, y, yaw)
    cls = np.full((5, 6), -3.0, np.float32)
    cls[:, 1] = 3.0
    return {
        "lanes": lanes, "lane_valid": np.array([3.0, 3.0, -3.0]), "lane_paint": np.array([0.9, 0.1, 0.1]),
        "curbs": np.stack([np.stack([np.full(16, -5.0), np.linspace(0, 30, 16)], 1)]), "curb_valid": np.array([3.0]),
        "objects": objects, "object_class": cls, "object_valid": np.full(5, 3.0),
        "signs": np.array([[4.0, 18.0, 0, 1]], np.float32), "sign_class": np.array([[3.0, -3, -3, -3]]),
        "sign_valid": np.array([3.0]), "stop_line": np.array([18.0, 3.0]),
        "crosswalks": np.array([[-4.0, 19.0, 4.0, 19.0]]),
        "occ": _road_occ(),
    }


def check_round_trip_and_edge_positions() -> None:
    from python.viz.occ_scene import MotionTracker, occ_to_viz, validate_message

    tracker = MotionTracker()
    for f in range(4):
        msg = json.loads(json.dumps(occ_to_viz(_pred(f), tracker=tracker, seq=f, t=0.1 * f, ego_speed_mps=10.0)))
    assert validate_message(msg) == [], validate_message(msg)
    assert [ln["paint"] for ln in msg["lanes"]] == ["yellow", "white"], "P(yellow) picks paint; invalid lane dropped"
    occ_edges = [e for e in msg["edges"] if e.get("src") == "occ"]
    assert len(occ_edges) == 2 and sum(e.get("src") == "curb" for e in msg["edges"]) == 1, msg["edges"]
    for e, want_x in zip(sorted(occ_edges, key=lambda e: e["pts"][0][0]), (-3.5, 3.5)):
        xs = np.array([p[0] for p in e["pts"]])
        ys = np.array([p[1] for p in e["pts"]])
        assert np.abs(xs - want_x).max() <= 0.25, (want_x, xs)
        assert abs(ys.min() - 0.5) <= 1.0 and abs(ys.max() - 19.5) <= 1.0, (ys.min(), ys.max())
        assert np.all(np.diff(ys) > 0), "edge points ordered forward"
    assert {c["cls"] for c in msg["occ"]} == {"drivable", "vehicle"}
    assert len(msg["occ"]) == 14 * 40 + 1, "free queries never drawn"
    motions = [t.get("motion") for t in msg["tracks"]]
    assert motions == ["same_way", "oncoming", "crossing", "alongside", "parked"], motions
    assert len({t["id"] for t in msg["tracks"]}) == 5
    assert msg["stop_lines"] == [{"pts": [[-1.8, 18.0], [1.8, 18.0]]}]
    assert len(msg["crosswalks"]) == 1 and msg["signs"][0]["cls"] == "stop_sign"
    print(f"  round trip: {len(msg['occ'])} occ cells, occ edges at x=+-3.5 (tol 0.25 m), 5 motion states over 4 frames")


def check_motion_needs_history() -> None:
    from python.viz.occ_scene import MotionTracker, occ_to_viz

    one = occ_to_viz(_pred(0))
    assert {t["motion"] for t in one["tracks"]} == {"unknown"}, "no tracker -> no single-frame guess"
    tr = MotionTracker()
    seen = [[t["motion"] for t in occ_to_viz(_pred(f), tracker=tr, t=0.1 * f, ego_speed_mps=10.0)["tracks"]] for f in range(3)]
    assert set(seen[0]) == {"unknown"} and set(seen[1]) == {"unknown"}, seen
    assert "unknown" not in seen[2], seen
    print("  motion: unknown until 3 frames over 0.2 s; never from one frame")


def check_red_only_edges_and_validator() -> None:
    from python.viz.occ_scene import build_message, occ_to_viz, scene_state, validate_message

    msg = occ_to_viz(_pred(0))
    assert {ln["paint"] for ln in msg["lanes"]} <= {"white", "yellow"}, "lanes are never red"
    assert all("paint" not in e for e in msg["edges"]), "edges and curbs are the only red, by kind not paint"
    reds = [k for k, v in _walk(msg) if v == "red"]
    assert reds == [], f"no red colour anywhere but the renderer's edges: {reds}"
    p = _pred(0)
    p.pop("lane_paint")
    assert {ln["paint"] for ln in build_message(scene_state(p))["lanes"]} == {"white"}, "no paint head -> white"
    extra = dict(msg, future_field={"x": 1})
    assert validate_message(extra) == [], "unknown fields are ignored"
    for bad in ({"lanes": [{"pts": [[0, 0], [0, 1]], "paint": "red", "kind": "detected"}]},
                {"lanes": [{"pts": [[0, 0], [0, 1]], "paint": "purple", "kind": "detected"}]},
                {"edges": [{"pts": [[0, 0], [0, 1]], "kind": "detected", "paint": "white"}]},
                {"edges": [{"pts": [[0, 0], [0, 1]], "kind": "lane"}]},
                {"occ": [{"x": 0.0, "y": 0.0, "z": 0.0, "cls": "sky"}]},
                {"tracks": [{"x": 0.0, "y": 0.0, "motion": "flying"}]}):
        assert validate_message(dict(msg, **bad)), f"validator must reject {bad}"
    gone = dict(msg)
    gone.pop("path")
    assert validate_message(gone), "missing v1 key rejected"
    ids = np.arange(3000) * 19
    big = build_message(scene_state(_pred(0), occ=(ids, np.tile([[4.0, 3.0, -2.0, -2.0]], (3000, 1)))))
    size = len(json.dumps(big, separators=(",", ":")).encode())
    assert len(big["occ"]) <= 800 and size < 60000, (len(big["occ"]), size)
    print("  red: only edges/curbs; lanes white/yellow; v1 required keys checked; unknown keys ignored; <60 kB")


def _walk(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "state":  # sign lamp state, not a draw colour
                continue
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    else:
        yield key, obj


def check_mirrors_and_no_gt() -> None:
    import python.viz.occ_scene as vs

    src = (ROOT / "python/viz/occ_scene.py").read_text(encoding="utf-8")
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in mods if "train" in m or "label" in m or "raw_frames" in m or m == "torch"}, mods
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_check("occ_viz_mirrors", "PyTorch is not installed")
        return
    import python.train.occ_sparse as occ
    import python.train.scene_net as sn

    assert vs.OCC_CLASSES == occ.OCC_CLASSES and vs.GRID_SHAPE == tuple(occ.GRID_SHAPE) and vs.VOXEL_M == occ.VOXEL_M
    assert vs.OBJECT_CLASSES == sn.OBJECT_CLASSES and vs.SIGN_CLASSES == sn.SIGN_CLASSES
    ids = np.arange(0, 57344, 97)
    assert np.allclose(vs.voxel_centers(ids), occ.voxel_centers(ids))
    head = occ.SparseOccHead(32)
    with torch.no_grad():
        lg = head(torch.zeros(1, 32), torch.as_tensor(ids[None]))[0].numpy()
    cells = vs.occ_cells(ids, lg, thresh=0.0)
    assert all(c["cls"] in vs.OCC_CLASSES for c in cells)
    print("  mirrors match occ_sparse/scene_net; occ_scene imports no torch, labels or raw frames")


def main() -> int:
    check_round_trip_and_edge_positions()
    check_motion_needs_history()
    check_red_only_edges_and_validator()
    check_mirrors_and_no_gt()
    print("test_occ_viz: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
