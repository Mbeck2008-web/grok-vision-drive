#!/usr/bin/env python3
"""Write offline sample frames for the vector stage. Not a live capture."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ui():
    from python.viz.stage import VizUI

    ui = VizUI()
    ui.show_nerd = False
    ui.style = "early"
    return ui


def _pts(fn, y0, y1, step=2.0):
    out = []
    y = float(y0)
    while y <= y1 + 1e-6:
        out.append({"x": float(fn(y)), "y": round(y, 2)})
        y += step
    return out


def _curb_corner(x_side, y_turn, radius, far_x):
    """Curb running up the road, rounding into the cross street."""
    sgn = 1.0 if x_side > 0 else -1.0
    pts = [{"x": x_side, "y": round(y, 2)} for y in np.arange(-16.0, y_turn, 1.0)]
    for k in range(1, 9):
        t = (math.pi / 2) * k / 8
        pts.append({"x": x_side + sgn * radius * (1 - math.cos(t)), "y": y_turn + radius * math.sin(t)})
    end = pts[-1]
    for x in np.arange(end["x"] + sgn, far_x, sgn * 1.0):
        pts.append({"x": float(x), "y": end["y"]})
    return pts


def _curb_back(x_side, y_turn, radius, far_x):
    """Far-side curb: from the cross street back onto the road ahead."""
    sgn = 1.0 if x_side > 0 else -1.0
    pts = [{"x": float(x), "y": y_turn} for x in np.arange(far_x, x_side + sgn * radius, -sgn * 1.0)]
    for k in range(0, 9):
        t = (math.pi / 2) * (1 - k / 8)
        pts.append({"x": x_side + sgn * radius * (1 - math.cos(t)), "y": y_turn + radius * math.sin(t)})
    end = pts[-1]
    pts += [{"x": x_side, "y": float(y)} for y in np.arange(end["y"] + 1.0, 110.0, 1.0)]
    return pts


def intersection():
    """Four-lane approach to a signalised cross street, like the ref1 frame."""
    lanes = []
    for x, idx, paint in ((-5.4, -2, "yellow"), (-1.8, -1, "white"), (1.8, 1, "white"), (5.4, 2, "white")):
        for y0, y1 in ((-16, 21), (40, 110)):
            lanes.append({
                "points": _pts(lambda y, x=x: x, y0, y1),
                "kind": "detected",
                "index": idx,
                "side": "left" if x < 0 else "right",
                # Real paint: the far-left line is the yellow centre line.
                "paint": paint,
            })
    # Cross street lanes, left to right across the view.
    for y, paint in ((28.5, "white"), (31.5, "yellow"), (34.5, "white")):
        lanes.append({
            "points": [{"x": float(x), "y": y} for x in np.arange(-60.0, 61.0, 2.0)],
            "kind": "detected",
            "index": 3,
            "side": "right",
            "paint": paint,
        })
    edges = [
        {"points": _curb_corner(-9.0, 19.0, 6.0, -60.0), "kind": "detected", "side": "left"},
        {"points": _curb_corner(9.0, 19.0, 6.0, 60.0), "kind": "detected", "side": "right"},
        {"points": _curb_back(-9.0, 38.0, 4.0, -60.0), "kind": "detected", "side": "left"},
        {"points": _curb_back(9.0, 38.0, 4.0, 60.0), "kind": "detected", "side": "right"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": 0.0, "y": 11, "yaw": 1.57},
        {"id": 2, "class": "vehicle", "x": 3.6, "y": 6, "yaw": 1.57},
        {"id": 3, "class": "vehicle", "x": 3.6, "y": 13, "yaw": 1.57},
        {"id": 4, "class": "vehicle", "x": 3.7, "y": 20, "yaw": 1.57},
        {"id": 5, "class": "vehicle", "x": -3.6, "y": 15, "yaw": 1.57},
        {"id": 6, "class": "truck", "x": -18.0, "y": 30.0, "yaw": 0.0},
        {"id": 7, "class": "vehicle", "x": 14.0, "y": 33.0, "yaw": math.pi},
        {"id": 8, "class": "pedestrian", "x": -10.5, "y": 23.0, "yaw": 0.0},
        {"id": 9, "class": "bike", "x": -7.6, "y": 12.0, "yaw": 1.57},
        {"id": 10, "class": "vehicle", "x": -1.8, "y": 46.0, "yaw": -1.57},
    ]
    signs = [
        {"cls": "traffic_light", "x": -1.8, "y": 40, "state": "red"},
        {"cls": "traffic_light", "x": 1.8, "y": 40, "state": "red"},
        {"cls": "traffic_light", "x": 9.5, "y": 39, "state": "red"},
        {"cls": "stop_sign", "x": -9.6, "y": 21.5},
    ]
    path = _pts(lambda y: 0.0, 2, 20, 1.5)
    return {
        "engaged": True,
        "path_conf": 0.9,
        "ego": {"speed_mps": 6.5},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 6.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": path,
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": signs,
    }


def multilane():
    lanes = []
    for x, idx in ((-9.0, -3), (-5.4, -2), (-1.8, -1), (1.8, 1), (5.4, 2), (9.0, 3)):
        lanes.append({
            "points": _pts(lambda y, x=x: x + 0.4 * math.sin(y / 22.0), -16, 140),
            "kind": "detected",
            "index": idx,
            "side": "left" if x < 0 else "right",
            "paint": "line_yellow" if idx == -3 else "line_white",
        })
    edges = [
        {"points": _pts(lambda y: -10.2, -16, 140), "kind": "detected"},
        {"points": _pts(lambda y: 12.6, -16, 140), "kind": "detected"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": 0.1, "y": 14, "yaw": 1.57},
        {"id": 2, "class": "vehicle", "x": 3.6, "y": 12, "yaw": 1.57},
        {"id": 3, "class": "vehicle", "x": -3.6, "y": 18, "yaw": 1.57},
        {"id": 4, "class": "vehicle", "x": 3.7, "y": 26, "yaw": 1.57},
        {"id": 5, "class": "truck", "x": -7.2, "y": 22, "yaw": 1.57},
        {"id": 6, "class": "vehicle", "x": 7.2, "y": 24, "yaw": 1.57},
        {"id": 7, "class": "pedestrian", "x": 11.2, "y": 10, "yaw": 0.2},
        {"id": 8, "class": "vehicle", "x": 7.2, "y": 7, "yaw": 1.57},
        {"id": 9, "class": "vehicle", "x": 0.2, "y": 34, "yaw": 1.57},
        {"id": 10, "class": "vehicle", "x": -3.7, "y": 40, "yaw": 1.57},
        {"id": 11, "class": "bike", "x": -11.5, "y": 30, "yaw": 1.57},
    ]
    signs = [
        {"cls": "traffic_light", "x": -3.0, "y": 48, "state": "green"},
        {"cls": "traffic_light", "x": 0.2, "y": 50, "state": "green"},
        {"cls": "traffic_light", "x": 3.2, "y": 48, "state": "red"},
    ]
    return {
        "engaged": True,
        "path_conf": 0.88,
        "ego": {"speed_mps": 16.0},
        "planner": {"cipv_id": 1, "target_v": 15.0, "aeb": "off"},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": _pts(lambda y: 0.35 * math.sin(y / 16.0), 2, 55, 1.5),
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": signs,
    }


def parking():
    """Parking aisle: parked rows both sides, path curving into a bay (ref3)."""
    tracks = []
    tid = 1
    for side, x in ((-1, -6.6), (1, 6.6)):
        for k, y in enumerate(np.arange(3.0, 44.0, 2.8)):
            if side == 1 and k == 5:
                continue  # the open bay the path turns into
            tracks.append({
                "id": tid,
                "class": "obstacle",
                "x": float(x + side * 0.3 * (k % 2)),
                "y": float(y),
                "yaw": 0.08 * ((k % 3) - 1),
                "length": 4.6,
                "width": 1.9,
                "height": 1.5,
            })
            tid += 1
    tracks.append({"id": 90, "class": "pedestrian", "x": -3.2, "y": 22.0, "yaw": 0.0})
    tracks.append({"id": 99, "class": "vehicle", "x": -0.4, "y": 30.0, "yaw": 1.57})
    edges = []
    for x in (-9.4, 9.4):
        edges.append({"points": _pts(lambda y, x=x: x, -16, 46, 1.0), "kind": "detected"})
    bay_y = 3.0 + 2.8 * 5
    path = [{"x": 0.0, "y": float(y)} for y in np.arange(2.0, bay_y - 4.0, 1.2)]
    for k in range(1, 9):
        t = (math.pi / 2) * k / 8
        path.append({"x": 4.0 * (1 - math.cos(t)), "y": bay_y - 4.0 + 4.0 * math.sin(t)})
    path.append({"x": 6.0, "y": bay_y})
    return {
        "engaged": False,
        "path_conf": 0.7,
        "ego": {"speed_mps": 3.2},
        "planner": {"cipv_id": 99, "target_v": 4.0, "aeb": "off"},
        "path_ego": path,
        "lanes_ext": [
            {"points": _pts(lambda y: -3.0, -16, 44), "kind": "detected", "index": -1, "paint": "white"},
        ],
        "road_edges": edges,
        "tracks": tracks,
        "signs": [],
    }


def _lane_rec(fn, y0, y1, idx, paint="white", step=2.0):
    return {
        "points": _pts(fn, y0, y1, step),
        "kind": "detected",
        "index": idx,
        "side": "left" if idx < 0 else "right",
        "paint": paint,
    }


def _base_state(speed, lead, path, lanes, edges, tracks, signs=(), engaged=True):
    return {
        "engaged": engaged,
        "path_conf": 0.9,
        "ego": {"speed_mps": speed},
        "planner": {"cipv_id": lead, "aeb": "off", "target_v": speed},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": path,
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": list(signs),
    }


def highway_curve():
    """Long sweeping left-hand highway curve: three lanes our way, a median, two the other way."""
    def bend(x0):
        # Curvature grows with distance (about 1/400 per metre ahead).
        return lambda y: x0 - (max(0.0, y) ** 2) / 800.0

    lanes = [_lane_rec(bend(x), -16, 150, idx, paint) for x, idx, paint in (
        (-5.4, -2, "yellow"), (-1.8, -1, "white"), (1.8, 1, "white"), (5.4, 2, "white"),
        (-9.6, -3, "yellow"), (-13.2, -3, "white"),
    )]
    edges = [
        {"points": _pts(bend(-6.2), -16, 150), "kind": "detected"},
        {"points": _pts(bend(-8.8), -16, 150), "kind": "detected"},
        {"points": _pts(bend(-17.4), -16, 150), "kind": "detected"},
        {"points": _pts(bend(7.6), -16, 150), "kind": "detected"},
    ]
    f = bend(0.0)

    def car(tid, cls, dx, y, back=False):
        yaw = (-1.57 if back else 1.57) - y / 400.0
        return {"id": tid, "class": cls, "x": f(y) + dx, "y": y, "yaw": yaw}

    tracks = [
        car(1, "vehicle", 0.0, 17), car(2, "vehicle", 3.6, 8), car(3, "truck", 3.6, 30),
        car(4, "vehicle", -3.6, 24), car(5, "vehicle", 0.0, 46), car(6, "vehicle", -11.4, 20, True),
        car(7, "vehicle", -11.4, 50, True), car(8, "vehicle", -15.0, 34, True),
    ]
    path = _pts(lambda y: f(y), 2, 70, 2.0)
    return _base_state(27.0, 1, path, lanes, edges, tracks)


def merge_ramp():
    """On-ramp from the right tapering into the right lane at y~58 (gore line + ramp car)."""
    def gore(y):
        return 1.8 + 9.0 * max(0.0, (58.0 - y) / 74.0) ** 1.5

    lanes = [
        _lane_rec(lambda y: -5.4, -16, 140, -2, "yellow"),
        _lane_rec(lambda y: -1.8, -16, 140, -1),
        _lane_rec(lambda y: 1.8, 58, 140, 1),
        _lane_rec(gore, -16, 58, 1),  # ramp gore / merge line
        _lane_rec(lambda y: 5.4, 58, 140, 2),
    ]
    edges = [
        {"points": _pts(lambda y: -8.6, -16, 140), "kind": "detected"},
        {"points": _pts(lambda y: max(5.6, gore(y) + 3.8), -16, 140), "kind": "detected"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": 0.0, "y": 18, "yaw": 1.57},
        {"id": 2, "class": "vehicle", "x": gore(30) + 1.9, "y": 30, "yaw": 1.75},
        {"id": 3, "class": "vehicle", "x": gore(8) + 1.9, "y": 8, "yaw": 1.8},
        {"id": 4, "class": "vehicle", "x": -3.6, "y": 10, "yaw": 1.57},
        {"id": 5, "class": "truck", "x": 3.6, "y": 66, "yaw": 1.57},
        {"id": 6, "class": "vehicle", "x": -3.6, "y": 40, "yaw": 1.57},
    ]
    path = _pts(lambda y: 0.0, 2, 60, 2.0)
    return _base_state(22.0, 1, path, lanes, edges, tracks)


def t_junction():
    """Road ends at a T: stop bar + zebra, cross traffic, path turning left."""
    lanes = [
        _lane_rec(lambda y: 0.0, -16, 20, -1, "yellow"),
        {"points": [{"x": float(x), "y": 32.0} for x in np.arange(-70.0, 71.0, 2.0)],
         "kind": "detected", "index": 3, "paint": "yellow"},
    ]
    edges = [
        {"points": _curb_corner(-4.2, 22.0, 4.0, -70.0), "kind": "detected"},
        {"points": _curb_corner(4.2, 22.0, 4.0, 70.0), "kind": "detected"},
        {"points": [{"x": float(x), "y": 39.5} for x in np.arange(-70.0, 71.0, 1.0)], "kind": "detected"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": -16.0, "y": 34.5, "yaw": 0.0},
        {"id": 2, "class": "vehicle", "x": 12.0, "y": 29.5, "yaw": math.pi},
        {"id": 3, "class": "truck", "x": 30.0, "y": 34.5, "yaw": 0.0},
        {"id": 4, "class": "pedestrian", "x": -6.5, "y": 21.0, "yaw": 0.0},
        {"id": 5, "class": "bike", "x": 6.0, "y": 24.0, "yaw": math.pi},
    ]
    path = [{"x": -1.9, "y": float(y)} for y in np.arange(2.0, 26.0, 1.5)]
    for k in range(1, 10):
        t = (math.pi / 2) * k / 9
        path.append({"x": -1.9 - 7.6 * (1 - math.cos(t)), "y": 26.0 + 7.6 * math.sin(t) * 0.45})
    signs = [{"cls": "stop_sign", "x": -4.8, "y": 19.0}]
    return _base_state(4.0, None, path, lanes, edges, tracks, signs)


GALLERY_LENSES = ("chase", "low", "high")
WIDE = (1548, 720)  # ~2.15:1, the framing of the early-beta stills


def gallery(dest=None):
    """Every scene x 3 lenses at 1280x720, plus a ~2.15:1 chase frame each."""
    from python.viz.early_vector import render_early_stage
    from python.viz.stage import render_stage

    dest = dest or (ROOT / "docs" / "images" / "viz_early_fsd" / "gallery")
    dest.mkdir(parents=True, exist_ok=True)
    ui = _ui()
    scenes = {
        "intersection": intersection(),
        "highway_curve": highway_curve(),
        "multilane": multilane(),
        "parking": parking(),
        "merge_ramp": merge_ramp(),
        "t_junction": t_junction(),
    }
    written = []
    for name, st in scenes.items():
        for lens in GALLERY_LENSES:
            frame = render_early_stage(st, ui, width=1280, height=720, lens=lens)
            path = dest / f"{name}_{lens}.png"
            cv2.imwrite(str(path), frame)
            written.append(path)
        frame = render_early_stage(st, ui, width=WIDE[0], height=WIDE[1], lens="chase")
        path = dest / f"{name}_wide.png"
        cv2.imwrite(str(path), frame)
        written.append(path)
    ov = _ui()
    ov.overlay = True
    for tag, (w, h) in (("", (1280, 720)), ("_wide", WIDE)):
        frame = render_stage(overlay_scene(), ui=ov, width=w, height=h, main_frame=dusk_frame(h, w))
        path = dest / f"overlay_dusk{tag}.png"
        cv2.imwrite(str(path), frame)
        written.append(path)
    for path in written:
        print(path)
    return written


def overlay_scene():
    """Multilane traffic pushed 14 m further out: the hood camera is low, so the
    early-beta overlay stills show small, distant boxes (ref4)."""
    st = multilane()
    st["tracks"] = [dict(t, y=float(t["y"]) + 14.0) for t in st["tracks"]]
    st["ego"] = {"speed_mps": 6.0}
    return st


def dusk_frame(h=720, w=1280):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        sky = np.array([40, 50, 70]) * (1.0 - t) + np.array([18, 22, 30]) * t
        img[y, :] = sky.astype(np.uint8)
    # Road plane-ish dark band.
    img[int(h * 0.42):, :] = (28, 32, 36)
    return img


def main() -> None:
    from python.viz.stage import render_stage

    dest = ROOT / "docs" / "images" / "viz_early_fsd"
    dest.mkdir(parents=True, exist_ok=True)
    ui = _ui()
    scenes = {
        "intersection": intersection(),
        "multilane_lights": multilane(),
        "parking": parking(),
    }
    for name, st in scenes.items():
        frame = render_stage(st, ui=ui, width=1280, height=800)
        path = dest / f"{name}.png"
        cv2.imwrite(str(path), frame)
        print(path, frame.shape)
    ov = _ui()
    ov.overlay = True
    frame = render_stage(overlay_scene(), ui=ov, width=1280, height=720, main_frame=dusk_frame())
    path = dest / "overlay_dusk.png"
    cv2.imwrite(str(path), frame)
    print(path, frame.shape)
    board_dest = ROOT / "docs" / "images" / "viz_debug"
    board_dest.mkdir(parents=True, exist_ok=True)
    frame = debug_board_sample()
    path = board_dest / "board.png"
    cv2.imwrite(str(path), frame)
    print(path, frame.shape)


def debug_board_sample():
    """Debug board after 60 synthetic ticks so the graphs have a trace."""
    from python.viz.stage import render_stage

    ui = _ui()
    ui.debug_board = True
    main = np.zeros((240, 320, 3), dtype=np.uint8)
    main[:] = (60, 50, 45)
    main[:110] = (70, 60, 90)
    st = multilane()
    st["flags"] = {"WET_ROAD": 0.8, "RAINING": 0.1}
    st["calib"] = {"pitch": 0.1, "yaw": 1.2, "roll": 0.0}
    frame = None
    for i in range(60):
        st["ego"] = {
            "speed_mps": 14.0 + 3.0 * math.sin(i / 12.0),
            "yaw_rate": 0.05 * math.sin(i / 9.0),
            "steer": 0.2 * math.sin(i / 9.0),
            "throttle": 0.3 + 0.2 * math.sin(i / 7.0),
            "brake": max(0.0, -0.3 * math.sin(i / 7.0)),
        }
        st["t"] = i / 12.0
        frame = render_stage(st, ui=ui, width=1280, height=720, main_frame=main, cam_frames={"main": main})
    return frame


if __name__ == "__main__":
    if "--gallery" in sys.argv[1:]:
        gallery()
    else:
        main()
        gallery()
