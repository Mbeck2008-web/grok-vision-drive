#!/usr/bin/env python3
"""Offline label pictures. Missing fields still render. Runtime does not import them."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _labels():
    return {
        "t": 12.5,
        "ego": {"speed_mps": 11.2},
        "calib": {"pitch": 0.10, "yaw": 1.20, "roll": 0.0},
        "fps": {"vision": 18.0, "draw": 17.5},
        "flags": {"WET_ROAD": 0.79, "RAINING": 0.10},
        "path_conf": 0.82,
        "engage": "ON",
        "cipv_id": 1,
        "path_ego": [{"x": 0.0, "y": float(y)} for y in range(6, 36, 2)],
        "lanes": [
            {"points": [{"x": -1.8, "y": float(y)} for y in range(6, 36, 2)]},
            {"points": [{"x": 1.8, "y": float(y)} for y in range(6, 36, 2)]},
        ],
        "edges": [{"points": [{"x": 4.0, "y": float(y)} for y in range(8, 30, 2)]}],
        "stop_lines": [{"y": 14.0, "x0": -3.0, "x1": 3.0}],
        "tracks": [
            {"id": 1, "class": "vehicle", "x": 0.2, "y": 18.0, "yaw": 1.57, "in_path": True},
            {"id": 2, "class": "truck", "x": -3.5, "y": 16.0, "yaw": 1.57},
            {"id": 3, "class": "pedestrian", "x": 4.5, "y": 12.0, "yaw": 1.2},
        ],
        "signs": [{"cls": "stop_sign", "x": -4.0, "y": 20.0}, {"cls": "traffic_light", "x": 2.0, "y": 24.0, "state": "green"}],
        "cones": [{"x": 3.2, "y": 10.0}],
    }


def _is_gray(p):
    return abs(int(p[0]) - int(p[1])) < 2 and abs(int(p[1]) - int(p[2])) < 2


def test_pixels_and_missing() -> None:
    from python.offline.label_review import class_letter, render_label_frame
    from python.viz.early_vector import project_main_xyz

    assert class_letter("vehicle") == "C"
    assert class_letter("truck") == "V"
    assert class_letter("pedestrian") == "G"
    assert class_letter("cone") == "F"
    assert class_letter("stop_sign") == "K"

    src = np.zeros((720, 1280, 3), dtype=np.uint8)
    src[:, :] = (30, 80, 140)
    painted = render_label_frame(src, _labels())
    assert painted.shape == src.shape
    assert _is_gray(painted[8, 8]), painted[8, 8]
    # A far corner of the sky stays gray: no label geometry there.
    assert _is_gray(painted[8, 400]) or int(painted[8, 400].max()) > 0

    def at(x, y, z=0.02):
        uv = project_main_xyz(x, y, z, 1280, 720)
        assert uv is not None, (x, y, z)
        u, v = uv
        return painted[v, u]

    lane = at(-1.8, 12.0)
    assert int(lane[1]) > 200 and int(lane[0]) < 40 and int(lane[2]) < 40, lane
    curb = at(4.0, 14.0)
    assert int(curb[0]) > 200 and int(curb[1]) < 40, curb
    stop = at(0.0, 14.0, 0.04)
    assert int(stop[2]) > 200 and int(stop[1]) < 40, stop
    # In-path car is red. Yaw is forward, so the side corner is half a width out.
    red = at(0.2 + 0.925, 18.0)
    assert int(red[2]) > 200, red
    # Truck is not in path: yellow wire at a side corner.
    yel = at(-3.5 - 1.25, 16.0)
    assert int(yel[1]) > 200 and int(yel[2]) > 200 and int(yel[0]) < 40, yel
    apex = at(3.2, 10.0, 0.75)
    assert int(apex[2]) > 140 and int(apex[1]) > 100, apex
    # Stop icon near the top center.
    icon = painted[36, 1280 // 2]
    assert int(icon[2]) > 200, icon
    # Hot flag is green somewhere in the top-left readout.
    hud = painted[0:120, 0:220]
    hot = (hud[:, :, 1] > 200) & (hud[:, :, 0] < 40) & (hud[:, :, 2] < 40)
    assert int(hot.sum()) > 8
    assert int(painted[-12, 10].max()) > 180, "MAIN label"

    empty = render_label_frame(src, {})
    assert empty.shape == src.shape
    assert _is_gray(empty[200, 200])
    icon = empty[20:52, 620:660]
    saturated = (icon[:, :, 2] > 200) & (icon[:, :, 1] < 40)
    assert int(saturated.sum()) == 0, "no stop icon without a stop"
    none_frame = render_label_frame(None, {"tracks": [{"class": "vehicle"}]})
    assert none_frame.ndim == 3
    print("label colors, missing fields, stop icon OK")


def test_folder_and_runtime_fence() -> None:
    from python.offline.label_review import encode_mp4, load_recording, render_label_frame

    for rel in (
        "python/run_vision.py",
        "python/viz/stage.py",
        "python/viz/early_vector.py",
        "python/viz/debug_board.py",
        "python/viz/showcase.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "label_review" not in text, rel
    for path in (ROOT / "python" / "runtime").glob("*.py"):
        assert "label_review" not in path.read_text(encoding="utf-8"), path.name

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames"
        frames.mkdir()
        img = np.full((240, 320, 3), 90, dtype=np.uint8)
        cv2.imwrite(str(frames / "000.png"), img)
        cv2.imwrite(str(frames / "001.png"), img)
        (root / "labels.jsonl").write_text(
            json.dumps({"ego": {"speed_mps": 1.0}}) + "\n",
            encoding="utf-8",
        )
        pairs = load_recording(root)
        assert len(pairs) == 2
        assert pairs[1][1] == {}
        painted = [render_label_frame(f, lab) for f, lab in pairs]
        out = root / "clip.mp4"
        assert encode_mp4(painted, out, fps=10.0)
        assert out.stat().st_size > 100
    print("recording folder, short labels, runtime fence OK")


def main() -> None:
    test_pixels_and_missing()
    test_folder_and_runtime_fence()
    print("test_viz_label_review: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"test_viz_label_review: FAIL — {exc}")
        sys.exit(1)
