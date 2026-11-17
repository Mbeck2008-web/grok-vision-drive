#!/usr/bin/env python3
"""Generate cpp_dots_stream.mp4 and the five cpp_dots_*.png stills.

The stream is a synthetic ~8 s drive (25 Hz) through the publisher, with
world-fixed beads and explicit appear / disappear events so fades are visible.
Stills reuse the gallery scenes at 1280x720, chase camera, HUD only on dusk.

    python scripts/render_cpp_dots_stream.py \
        --exe cpp/viz/build/linux-release/gvd_viz \
        --out /workspace/gvd-ui
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.viz.cpp_publisher import write_scene_file  # noqa: E402

import importlib.util

_gal_spec = importlib.util.spec_from_file_location(
    "render_viz_cpp_gallery", ROOT / "scripts" / "render_viz_cpp_gallery.py"
)
_gal = importlib.util.module_from_spec(_gal_spec)
assert _gal_spec.loader is not None
_gal_spec.loader.exec_module(_gal)
dusk = _gal.dusk
highway_curve = _gal.highway_curve
intersection = _gal.intersection
multilane = _gal.multilane
parking = _gal.parking
render = _gal.render

FPS = 25
DURATION_S = 8.0
YAW = 1.5707963  # facing +Y


def _line(x: float, y0: float, y1: float, step: float = 2.0) -> list[dict]:
    pts, y = [], float(y0)
    while y <= y1 + 1e-6:
        pts.append({"x": x, "y": round(y, 2), "cls": "white", "prob": 1.0, "observed": True})
        y += step
    return pts


def _curb(x: float, y0: float, y1: float, step: float = 0.8) -> list[dict]:
    pts, y = [], float(y0)
    while y <= y1 + 1e-6:
        pts.append({"x": x, "y": round(y, 2), "cls": "curb", "prob": 1.0, "observed": True})
        y += step
    return pts


def stream_states() -> list[dict]:
    """World-fixed road in ego frame at t=0, plus appear/disappear events.

    Ego pose advances along +Y so the car stays screen-locked and the beads
    stream past. A yellow lane appears at t=2.0 s and is gone after t=6.0 s
    (fade in / fade out). A mid-block curb gap is filled in at t=3.5 s and
    one far-ahead white bead is removed at t=5.0 s.
    """
    n = int(DURATION_S * FPS)
    states = []
    base_white = _line(-1.8, -20.0, 160.0, 2.0) + _line(1.8, -20.0, 160.0, 2.0)
    base_curb = _curb(-7.0, -20.0, 160.0) + _curb(7.0, -20.0, 160.0)
    for i, d in enumerate(base_white + base_curb):
        d["id"] = i + 1
    appear_pts = []
    y = -20.0
    aid = 10_000
    while y <= 160.0 + 1e-6:
        appear_pts.append({"id": aid, "x": 0.0, "y": round(y, 2), "cls": "yellow",
                           "prob": 0.9, "observed": True})
        aid += 1
        y += 2.0
    disappear_id = 40  # a white bead well ahead at t=0

    for f in range(n):
        t = f / float(FPS)
        ego_y = 18.0 * t  # 18 m/s
        dots = [dict(d) for d in base_white + base_curb]
        if t >= 2.0:
            # appear: yellow centre line fades in and stays until 6.0 s
            if t < 6.0:
                dots.extend(dict(d) for d in appear_pts)
        # disappear: drop one persistent white bead from 5.0 s
        if t >= 5.0:
            dots = [d for d in dots if d.get("id") != disappear_id]
        # a late curb fill-in at t=3.5 (new ids, must fade in)
        if t >= 3.5:
            dots.append({"id": 20_000, "x": -7.0, "y": 90.4, "cls": "curb", "prob": 1.0, "observed": True})
        tracks = [{"id": 1, "class": "vehicle", "x": 3.6, "y": 14.0, "yaw": YAW,
                   "length": 4.5, "width": 1.85, "height": 1.5}]
        states.append({
            "engaged": True,
            "path_conf": 0.9,
            "ego": {"speed_mps": 18.0, "x": 0.0, "y": ego_y, "yaw": YAW, "yaw_rate_rps": 0.0},
            "planner": {"cipv_id": 1, "aeb": "off", "target_v": 18.0},
            "path_ego": [{"x": 0.0, "y": float(y)} for y in (2.0, 8.0, 16.0)],
            "dots": dots,
            "tracks": tracks,
            "lanes_ext": [],
            "road_edges": [],
            "signs": [],
        })
    return states


def encode_mp4(frames_prefix: Path, out: Path, fps: int = FPS) -> None:
    # frames are written as prefix_0000.png
    pattern = str(frames_prefix) + "_%04d.png"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-i", pattern,
        "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ]
    subprocess.run(cmd, check=True)


STILLS = (
    ("intersection", intersection, "neon", "cpp_dots_1_intersection.png"),
    ("multilane", multilane, "neon", "cpp_dots_2_multilane.png"),
    ("parking", parking, "neon", "cpp_dots_3_parking.png"),
    ("dusk", dusk, "dusk", "cpp_dots_4_dusk.png"),
    ("highway_curve", highway_curve, "neon", "cpp_dots_5_highway_curve.png"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "cpp/viz/build/linux-release/gvd_viz"))
    ap.add_argument("--out", default="/workspace/gvd-ui")
    ap.add_argument("--skip-stills", action="store_true")
    ap.add_argument("--skip-stream", action="store_true")
    args = ap.parse_args()
    exe, out = Path(args.exe), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not exe.is_file():
        sys.exit(f"gvd_viz not found: {exe}")

    if not args.skip_stills:
        with tempfile.TemporaryDirectory() as tmp:
            for name, fn, palette, fname in STILLS:
                sf = Path(tmp) / f"{name}.jsonl"
                write_scene_file(sf, [fn()])
                dest = out / fname
                render(exe, sf, dest, (1280, 720), palette, lens="chase", hud=(name == "dusk"))
                print(dest)

    if not args.skip_stream:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            sf = tmp_p / "stream.jsonl"
            n = write_scene_file(sf, stream_states())
            prefix = tmp_p / "f"
            subprocess.run(
                [str(exe), "--offline", str(sf), "--png-seq", str(prefix),
                 "--size", "1280x720", "--lens", "chase", "--palette", "neon"],
                check=True,
            )
            mp4 = out / "cpp_dots_stream.mp4"
            encode_mp4(prefix, mp4)
            print(f"{mp4} ({n} frames, appear at 2.0 s / 3.5 s, disappear at 5.0 s / 6.0 s)")


if __name__ == "__main__":
    main()
