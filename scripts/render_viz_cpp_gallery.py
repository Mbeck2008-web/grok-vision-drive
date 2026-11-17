#!/usr/bin/env python3
"""Render the C++ visualizer (gvd_viz) on fixed scenes matched to the reference
screenshots, and build side-by-side compare sheets.

Offline sample frames only: these are hand-built perception states, not a live
capture. Each state goes through the real publisher (build_scene_message) so the
renderer sees exactly what --viz-cpp would send.

    python scripts/render_viz_cpp_gallery.py --exe cpp/viz/build/linux-release/gvd_viz \
        --refs /workspace/gvd-viz-refs/fsd-early-beta --out /workspace/gvd-ui --round 1
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.viz.cpp_publisher import write_scene_file  # noqa: E402


def line(fn, y0, y1, step=2.0):
    out, y = [], float(y0)
    while y <= y1 + 1e-6:
        out.append({"x": float(fn(y)), "y": round(y, 2)})
        y += step
    return out


def xline(y, x0, x1, step=2.0):
    return [{"x": float(x), "y": float(y)} for x in np.arange(x0, x1 + 1e-6, step)]


def corner(x_side, y_turn, radius, far_x, y_from=-16.0):
    """Curb up the road, rounding into a cross street."""
    s = 1.0 if x_side > 0 else -1.0
    pts = [{"x": x_side, "y": float(y)} for y in np.arange(y_from, y_turn, 0.5)]
    for k in range(1, 10):
        t = (math.pi / 2) * k / 9
        pts.append({"x": x_side + s * radius * (1 - math.cos(t)), "y": y_turn + radius * math.sin(t)})
    end = pts[-1]
    pts += [{"x": float(x), "y": end["y"]} for x in np.arange(end["x"] + s * 0.5, far_x, s * 0.5)]
    return pts


def corner_back(x_side, y_turn, radius, far_x, y_to=120.0):
    """Far-side curb: from the cross street back up the road ahead."""
    s = 1.0 if x_side > 0 else -1.0
    pts = [{"x": float(x), "y": y_turn} for x in np.arange(far_x, x_side + s * radius, -s * 0.5)]
    for k in range(0, 10):
        t = (math.pi / 2) * (1 - k / 9)
        pts.append({"x": x_side + s * radius * (1 - math.cos(t)), "y": y_turn + radius * (1 - math.sin(t))})
    pts += [{"x": x_side, "y": float(y)} for y in np.arange(pts[-1]["y"] + 0.5, y_to, 0.5)]
    return pts


_CLS_SIZE = {
    "vehicle": (4.5, 1.85, 1.5), "car": (4.5, 1.85, 1.5), "parked": (4.5, 1.9, 1.5),
    "truck": (7.6, 2.5, 3.0), "bus": (10.5, 2.6, 3.2), "trailer": (12.0, 2.5, 3.4),
    "bike": (1.8, 0.6, 1.5), "bicycle": (1.8, 0.6, 1.5), "cyclist": (1.8, 0.6, 1.6),
    "motorcycle": (2.1, 0.7, 1.4), "pedestrian": (0.7, 0.7, 1.7), "ped": (0.7, 0.7, 1.7),
    "person": (0.7, 0.7, 1.7), "obstacle": (4.4, 1.9, 1.6), "debris": (1.2, 0.8, 0.6),
    "cone": (0.4, 0.4, 0.75),
}


def car(i, x, y, yaw=1.5708, cls="vehicle", **kw):
    l, w, h = _CLS_SIZE.get(cls, (4.5, 1.85, 1.5))
    d = {"id": i, "class": cls, "x": x, "y": y, "yaw": yaw, "length": l, "width": w, "height": h}
    d.update(kw)
    return d


def lane(pts, paint="white", kind="detected"):
    return {"points": pts, "kind": kind, "paint": paint, "side": "left", "index": 0}


def base(speed, tracks, lanes, edges, signs=(), path=None, lead=1, set_v=None):
    return {
        "engaged": True,
        "path_conf": 0.9,
        "ego": {"speed_mps": speed},
        "planner": {"cipv_id": lead, "aeb": "off", "target_v": set_v if set_v is not None else speed},
        "path_ego": path or [],
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": list(signs),
    }


def intersection():
    """ref1: four lanes into a signalised cross street."""
    lanes = []
    for x, paint in ((-5.4, "yellow"), (-1.8, "white"), (1.8, "white"), (5.4, "white")):
        lanes.append(lane(line(lambda y, x=x: x, -16, 22), paint))
        lanes.append(lane(line(lambda y, x=x: x, 42, 120), paint))
    for y, paint in ((29.0, "white"), (32.0, "white"), (35.0, "white")):
        lanes.append(lane(xline(y, -70, 70), paint))
    edges = [
        {"points": corner(-9.0, 20.0, 6.0, -70.0), "kind": "detected", "cls": "curb"},
        {"points": corner(9.0, 20.0, 6.0, 70.0), "kind": "detected", "cls": "curb"},
        {"points": corner_back(-9.0, 40.0, 5.0, -70.0), "kind": "detected", "cls": "curb"},
        {"points": corner_back(9.0, 40.0, 5.0, 70.0), "kind": "detected", "cls": "curb"},
    ]
    tracks = [
        car(1, 0.0, 11.0),
        car(2, 3.6, 5.5), car(3, 3.6, 12.0), car(4, 3.7, 18.5),
        car(5, -3.6, 17.0),
        car(6, -9.0, 8.0, cls="truck", length=6.0, width=2.4, height=2.6, yaw=1.2),
        car(7, -7.4, 4.0, cls="bike", yaw=1.57),
        car(8, -16.0, 31.0, yaw=0.0, cls="truck", length=7.0, width=2.4, height=2.4),
        car(9, 14.0, 33.5, yaw=math.pi),
        car(10, -1.8, 52.0, yaw=-1.5708),
    ]
    signs = [
        {"cls": "traffic_light", "x": 9.6, "y": 40.0, "state": "red"},
        {"cls": "traffic_light", "x": -1.8, "y": 42.0, "state": "red"},
    ]
    return base(6.5, tracks, lanes, edges, signs, line(lambda y: 0.0, 2, 9, 1.0))


def multilane():
    """ref2: wide multi-lane approach, light row ahead, cross traffic at the far sides."""
    lanes = []
    for x, paint in ((-12.6, "white"), (-9.0, "yellow"), (-8.6, "yellow"), (-5.4, "white"), (-1.8, "white"),
                     (1.8, "white"), (5.4, "white"), (9.0, "white")):
        lanes.append(lane(line(lambda y, x=x: x - 0.003 * y * y if x < -8 else x, -16, 38), paint))
    for y in (44.0, 47.0, 50.0):
        lanes.append(lane(xline(y, -60, 60), "white"))
    edges = [
        {"points": corner(-15.0, 32.0, 7.0, -70.0), "kind": "detected", "cls": "curb"},
        {"points": corner(12.6, 34.0, 7.0, 70.0), "kind": "detected", "cls": "curb"},
    ]
    tracks = [
        car(1, 0.0, 10.0, height=1.7),
        car(2, -3.6, 1.0, length=5.0, width=2.1, height=2.4),
        car(3, 3.6, 2.5, length=4.8, width=2.0, height=1.9), car(4, 3.6, 17.0), car(5, 7.2, 14.0),
        car(6, -3.6, 15.0, height=1.8), car(7, -3.6, 24.0), car(8, 3.6, 25.0), car(9, 0.3, 27.0),
        car(10, -10.5, 22.0, cls="truck", length=4.8, width=2.2, height=2.2, yaw=1.3),
        car(11, -22.0, 47.0, yaw=0.0), car(12, 22.0, 46.0, yaw=math.pi), car(13, 28.0, 48.5, yaw=math.pi),
    ]
    signs = [
        {"cls": "traffic_light", "x": -8.0, "y": 54.0, "state": "red"},
        {"cls": "traffic_light", "x": -3.5, "y": 54.0, "state": "red"},
        {"cls": "traffic_light", "x": 0.5, "y": 54.0, "state": "green"},
        {"cls": "traffic_light", "x": 4.5, "y": 54.0, "state": "green"},
        {"cls": "traffic_light", "x": 8.5, "y": 54.0, "state": "red"},
        {"cls": "traffic_light", "x": 13.0, "y": 34.0, "state": "red"},
    ]
    return base(9.0, tracks, lanes, edges, signs, line(lambda y: 0.0, 2, 11, 1.0), set_v=11.0)


def parking():
    """ref3: parking aisle, parked rows, path curving into a bay."""
    tracks, tid = [], 1
    for side, x in ((-1, -7.5), (1, 7.5)):
        for k, y in enumerate(np.arange(6.0, 56.0, 3.4)):
            if (side == 1 and k == 3) or (side == -1 and k in (5, 9)) or (side == 1 and k == 10):
                continue  # open bays
            tracks.append(car(tid, float(x + side * 0.3 * (k % 2)), float(y), yaw=0.0 + 0.06 * ((k % 3) - 1),
                              cls="parked", length=4.6, width=1.9, height=1.5))
            tid += 1
    tracks.append(car(90, -3.0, 14.0, cls="pedestrian"))
    edges = [{"points": line(lambda y: -11.0, -16, 52, 0.5), "kind": "detected", "cls": "curb"},
             {"points": line(lambda y: 11.0, -16, 52, 0.5), "kind": "detected", "cls": "curb"}]
    bay = 6.0 + 3.4 * 3
    path = [{"x": 0.0, "y": float(y)} for y in np.arange(2.0, bay - 4.0, 1.0)]
    for k in range(1, 10):
        t = (math.pi / 2) * k / 9
        path.append({"x": 5.0 * (1 - math.cos(t)), "y": bay - 4.0 + 4.0 * math.sin(t)})
    path.append({"x": 6.5, "y": bay})
    lanes = [lane(line(lambda y: 2.2, 0, 8, 0.6), "white")]
    return base(2.5, tracks, lanes, edges, (), path, lead=-1)


def dusk():
    """ref4: residential T-junction at dusk (overlay palette)."""
    lanes = [lane(line(lambda y: -0.2 - 0.004 * y * y, -10, 40, 1.5), "yellow"),
             lane(line(lambda y: 3.2 + 0.002 * y * y, -10, 40, 1.5), "white"),
             lane(xline(30.0, -40.0, -6.0, 1.5), "white")]
    edges = [{"points": corner(-6.5, 22.0, 6.0, -45.0, -10.0), "kind": "detected", "cls": "curb"},
             {"points": corner(7.5, 24.0, 6.0, 45.0, -10.0), "kind": "detected", "cls": "curb"},
             {"points": xline(42.0, -45.0, 45.0, 0.5), "kind": "detected", "cls": "curb"}]
    tracks = [car(1, -14.0, 34.0, yaw=0.0), car(2, 9.0, 36.0, yaw=math.pi), car(3, 15.0, 37.0, yaw=math.pi),
              car(4, 21.0, 37.5, yaw=math.pi), car(5, -2.5, 46.0, cls="pedestrian"),
              car(6, 9.0, 8.0, yaw=0.2, cls="parked", length=4.8, width=2.0, height=1.6)]
    path = line(lambda y: 0.9 + 0.012 * y * y if y > 6 else 0.9, 2, 36, 1.0)
    return base(5.8, tracks, lanes, edges, [{"cls": "stop_sign", "x": -6.8, "y": 26.0}], path, lead=-1, set_v=11.0)


def highway_curve():
    """Long sweeping highway curve, three lanes plus shoulders."""
    k = 1.0 / 220.0
    def off(x):
        return lambda y, x=x: x + 0.5 * k * y * y
    lanes = [lane(line(off(-5.4), -16, 150, 2.0), "yellow"), lane(line(off(-1.8), -16, 150, 2.0)),
             lane(line(off(1.8), -16, 150, 2.0)), lane(line(off(5.4), -16, 150, 2.0))]
    edges = [{"points": line(off(-7.0), -16, 150, 0.5), "kind": "detected", "cls": "curb"},
             {"points": line(off(7.2), -16, 150, 0.5), "kind": "detected", "cls": "curb"}]
    def yaw_at(y):
        return math.atan2(1.0, k * y)
    tracks = [car(1, off(0.0)(26.0), 26.0, yaw=yaw_at(26.0)), car(2, off(3.6)(12.0), 12.0, yaw=yaw_at(12.0)),
              car(3, off(-3.6)(40.0), 40.0, yaw=yaw_at(40.0), cls="truck"),
              car(4, off(3.6)(55.0), 55.0, yaw=yaw_at(55.0)), car(5, off(0.0)(80.0), 80.0, yaw=yaw_at(80.0)),
              car(6, off(-3.6)(95.0), 95.0, yaw=yaw_at(95.0))]
    return base(29.0, tracks, lanes, edges, (), line(off(0.0), 2, 22, 1.5), set_v=31.0)


# scene -> (ref file or None, palette, chase lens args for the ref match, extra angles)
SCENES = {
    "intersection": (intersection, "ref1_intersection_neon.png", "neon"),
    "multilane": (multilane, "ref2_multilane_lights.png", "neon"),
    "parking": (parking, "ref3_parkinglot_orange.png", "parking"),
    "dusk": (dusk, "ref4_overlay_dusk_orange.png", "dusk"),
    "highway_curve": (highway_curve, None, "neon"),
}
# Per-scene ref framing (eye, look, fov) tuned against the refs, round by round.
REF_LENS = {
    "intersection": ("0,-14.3,9", "0,6.3,0", 38.0),
    "multilane": ("0,-16,8", "0,12,0", 34.0),
    "parking": ("0,-12,8", "0,6,0", 42.0),
    "dusk": ("0,-22.9,8.6", "0,21.8,0", 45.0),
    "highway_curve": ("0,-16,8", "0,12,0", 34.0),
}
# The ref crops show no speed HUD except the dusk overlay.
REF_HUD = {"dusk"}
ANGLES = ("chase", "low", "high", "side", "top")


def render(exe: Path, scene_file: Path, out: Path, size: tuple[int, int], palette: str, lens: str | None = None,
           eye=None, look=None, fov=None, hud: bool = True) -> None:
    cmd = [str(exe), "--offline", str(scene_file), "--png", str(out), "--size", f"{size[0]}x{size[1]}",
           "--palette", palette]
    if not hud:
        cmd.append("--no-hud")
    if lens:
        cmd += ["--lens", lens]
    if eye:
        cmd += ["--eye", eye, "--look", look, "--fov", str(fov)]
    subprocess.run(cmd, check=True)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 9 * len(text)), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_w(img: np.ndarray, w: int) -> np.ndarray:
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "cpp/viz/build/linux-release/gvd_viz"))
    ap.add_argument("--refs", default="/workspace/gvd-viz-refs/fsd-early-beta")
    ap.add_argument("--out", default="/workspace/gvd-ui")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--panel-w", type=int, default=760)
    args = ap.parse_args()
    exe, refs, out = Path(args.exe), Path(args.refs), Path(args.out)
    shots = out / f"cpp_r{args.round}"
    shots.mkdir(parents=True, exist_ok=True)
    rows, angle_rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (fn, ref_name, palette) in SCENES.items():
            sf = Path(tmp) / f"{name}.jsonl"
            write_scene_file(sf, [fn()])
            ref = cv2.imread(str(refs / ref_name)) if ref_name else None
            if ref is not None:
                aspect = ref.shape[1] / ref.shape[0]
            else:
                aspect = 2.15
            w = 1280
            h = int(round(w / aspect)) // 2 * 2
            eye, look, fov = REF_LENS[name]
            png = shots / f"{name}_ref_match.png"
            render(exe, sf, png, (w, h), palette, eye=eye, look=look, fov=fov, hud=name in REF_HUD)
            mine = cv2.imread(str(png))
            pw = args.panel_w
            left = label(fit_w(ref, pw), f"REF {ref_name}") if ref is not None else label(
                np.zeros_like(fit_w(mine, pw)), "no reference (highway curve)")
            right = label(fit_w(mine, pw), f"C++ gvd_viz r{args.round}: {name}")
            hh = max(left.shape[0], right.shape[0])
            pad = lambda im: cv2.copyMakeBorder(im, 0, hh - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            rows.append(np.hstack([pad(left), np.full((hh, 6, 3), 60, np.uint8), pad(right)]))
            tiles = []
            for ang in ANGLES:
                p = shots / f"{name}_{ang}.png"
                render(exe, sf, p, (960, 540), palette, lens=ang)
                tiles.append(label(fit_w(cv2.imread(str(p)), 384), f"{name} / {ang}"))
            angle_rows.append(np.hstack(tiles))
    sep = lambda w: np.full((6, w, 3), 60, np.uint8)
    sheet = rows[0]
    for r in rows[1:]:
        sheet = np.vstack([sheet, sep(sheet.shape[1]), r])
    cv2.imwrite(str(out / f"cpp_compare_r{args.round}.png"), sheet)
    angles = angle_rows[0]
    for r in angle_rows[1:]:
        angles = np.vstack([angles, sep(angles.shape[1]), r])
    cv2.imwrite(str(out / f"cpp_angles_r{args.round}.png"), angles)
    print(out / f"cpp_compare_r{args.round}.png")
    print(out / f"cpp_angles_r{args.round}.png")


if __name__ == "__main__":
    main()
