#!/usr/bin/env python3
"""Recorder road-edge heights for labels v4.

The map file keeps left/right world z from the DecalRoad edges. A frame
points at that file. Old dumps with no edge height still load as None.
The polyline is not copied onto every tick (state.jsonl has a 3.5 GiB cap).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.control_log import TRAFFIC_CHUNK, parse_traffic, road_tag  # noqa: E402
from python.collect.road_dump import (  # noqa: E402
    load_road_dump,
    network_from_polys,
    persist_recording_roads,
    road_gt_for_row,
    write_road_dump,
)
from python.collect.road_loader import reset_shared  # noqa: E402
from python.debug.gt_geom import RoadPoly  # noqa: E402


class _Hit:
    def __init__(self) -> None:
        self.road_id = "kept"
        self.offset_m = 0.25
        self.heading_err_rad = -0.04
        self.lane_width_m = 3.5


def _spec() -> dict:
    edges = []
    for k in range(4):
        y = 4.0 * k
        edges.append(
            {
                "middle": [0.0, y, 6.0],
                "left": [-1.75, y, 5.0],
                "right": [1.75, y, 7.0],
            }
        )
    return {"kept": {"drivability": 1, "edges": edges}}


class _Bng:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.calls = 0

    def get_road_network(self, include_edges: bool = True, drivable_only: bool = True):
        self.calls += 1
        assert include_edges is True and drivable_only is True
        return self.spec


def check_recording_keeps_edge_z_and_old_dumps_stay_none() -> None:
    reset_shared()
    spec = _spec()
    bng = _Bng(spec)
    # A poly without heights must not replace the game payload.
    bare = RoadPoly("other", [(0.0, 0.0), (0.0, 12.0)], 4.0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = persist_recording_roads(root, "west_coast_usa", bng, [bare])
        assert path is not None and path.name == "west_coast_usa.json.gz"
        assert bng.calls == 1
        rows = [
            {
                "i": index,
                "map": "west_coast_usa",
                "road": {
                    "road_id": "kept",
                    "edge_z_file": "roads/west_coast_usa.json.gz",
                },
            }
            for index in range(3)
        ]
        for row in rows:
            got = road_gt_for_row(root, row)
            assert got is not None
            assert len(got.left_z) == len(got.xy) == len(got.right_z)
            assert got.left_z == [5.0, 5.0, 5.0, 5.0]
            assert got.right_z == [7.0, 7.0, 7.0, 7.0]
            assert got.z == [6.0, 6.0, 6.0, 6.0]

        old = RoadPoly("old", [(0.0, 0.0), (0.0, 10.0)], 7.0)
        old_path = persist_recording_roads(root, "legacy", None, [old])
        assert old_path is not None
        loaded = road_gt_for_row(root, {"map": "legacy", "road": {"road_id": "old"}})
        assert loaded is not None and loaded.left_z is None and loaded.right_z is None
        assert road_gt_for_row(root, {"road": {"edge_z_file": "roads/west_coast_usa.json.gz"}}) is None
        assert road_gt_for_row(
            root, {"map": "west_coast_usa", "road": {"road_id": "kept", "edge_z_file": None}}
        ) is None
        # The id is not in the file edge_z_file names. The other map is not searched.
        assert road_gt_for_row(
            root,
            {"map": "west_coast_usa", "road": {"road_id": "kept", "edge_z_file": "roads/legacy.json.gz"}},
        ) is None

        # A hand-written dump whose points are xy only is the same None.
        xy_only = {
            "bare": {
                "edges": [
                    {"middle": [0.0, 0.0], "left": [-1.0, 0.0], "right": [1.0, 0.0]},
                    {"middle": [0.0, 8.0], "left": [-1.0, 8.0], "right": [1.0, 8.0]},
                ]
            }
        }
        raw = root / "roads" / "xy.json.gz"
        write_road_dump(raw, level="xy", spawn=None, spawn_source="", road_network=xy_only)
        polys, _payload = load_road_dump(raw)
        assert polys and polys[0].left_z is None and polys[0].right_z is None
    print("  recording keeps left_z/right_z per edge point; old dumps stay None")


def check_frame_does_not_copy_the_polyline() -> None:
    poly = RoadPoly(
        "kept",
        [(0.0, float(i)) for i in range(40)],
        7.0,
        z=[1.0] * 40,
        left_z=[1.1] * 40,
        right_z=[0.9] * 40,
    )
    tag = road_tag(_Hit(), 2, poly)
    assert tag is not None
    assert tag["edge_z"] is True
    assert tag["road_width_m"] == 7.0
    assert tag["road_offset_m"] == 0.25
    assert tag["heading_err_rad"] == -0.04
    assert "left_z" not in tag and "right_z" not in tag
    blob = json.dumps(tag)
    assert len(blob) < 400, len(blob)
    built = network_from_polys([poly])
    assert built["kept"]["edges"][3]["left"][2] == 1.1
    assert built["kept"]["edges"][3]["right"][2] == 0.9
    print(f"  road tag is {len(blob)} bytes and does not repeat the edge polyline")


def check_traffic_model_is_recorded_without_a_class_guess() -> None:
    assert "jbeam" in TRAFFIC_CHUNK
    for banned in ("truck", "bus", "pedestrian", "bike"):
        assert banned not in TRAFFIC_CHUNK, banned
    car = {
        "name": "parked",
        "model": "etk800",
        "pos": [1, 2, 3],
        "vel": [0, 0, 0],
        "dir": [1, 0, 0],
    }
    parsed = parse_traffic(json.dumps([car]))
    assert parsed is not None and parsed[0]["model"] == "etk800"
    nameless = dict(car)
    del nameless["model"]
    plain = parse_traffic(json.dumps([nameless]))
    assert plain is not None and "model" not in plain[0]
    empty = parse_traffic(json.dumps([{**car, "model": ""}]))
    assert empty is not None and "model" not in empty[0]
    print("  traffic model is the jbeam string; a missing model is omitted")


def check_disk_delta_against_the_two_hour_baseline() -> None:
    """Per-frame JSON growth, plus the six map dumps. Images stay the 96 GB / 2 h baseline."""
    old_road = {"road_id": "889854.0", "lanes_here": 2, "lane_width_m": 3.5000123456789}
    new_road = {
        **old_road,
        "road_width_m": 7.000042337134482,
        "road_offset_m": 0.17768413419996776,
        "heading_err_rad": -0.04123456789012345,
        "edge_z": True,
        "edge_z_file": "roads/west_coast_usa.json.gz",
    }
    top = {
        "road_width_m": new_road["road_width_m"],
        "road_offset_m": new_road["road_offset_m"],
        "heading_err_rad": new_road["heading_err_rad"],
    }
    before = {"road": old_road}
    after = {"road": new_road, **top}
    delta = len(json.dumps(after)) - len(json.dumps(before))
    frames = 8 * 3600 * 10
    json_bytes = delta * frames
    dumps = sum(path.stat().st_size for path in (ROOT / "tests" / "fixtures" / "roads").glob("*.json.gz"))
    assert dumps == 9862547, dumps
    baseline_gb = 96.0 / 2.0 * 8.0
    extra_gb = (json_bytes + dumps) / 1e9
    print(
        f"  disk per_frame={delta} B rows_8h={json_bytes / 1e6:.2f} MB "
        f"dumps={dumps / 1e6:.2f} MB extra={extra_gb:.4f} GB "
        f"8h={baseline_gb + extra_gb:.2f} GB (baseline {baseline_gb:.0f} GB)"
    )


def _curb_kind(x: float, y: float, z: float, road: RoadPoly):
    """Class of one main-camera point, and its view-Z range in metres."""
    import scripts.test_obstacle_labels as tol
    from python.train.obstacles import _unproject, label_obstacle_frame, view_depth_m

    cam = tol._cam("main")
    depth = np.zeros((cam.height, cam.width), np.uint16)
    hit = view_depth_m(cam, x, y, z, cam.width, cam.height)
    assert hit is not None, (x, y, z)
    u, v, metres = hit
    ui, vi = int(round(u)), int(round(v))
    assert 0 <= ui < cam.width and 0 <= vi < cam.height, (ui, vi, x, y)
    depth[vi, ui] = np.uint16(max(1, int(round(float(metres) * 100.0))))
    values, valid = tol._paint(3.5)
    got = label_obstacle_frame(
        depth,
        pose=cam,
        lane_gt=values,
        lane_gt_valid=valid,
        road=road,
        ego_pos=(0.0, 0.0, 0.0),
        ego_dir=(0.0, 1.0, 0.0),
        ego_up=(0.0, 0.0, 1.0),
        cam="main",
        t=1.0,
    )
    ex, ey, _ez, ok = _unproject(cam, depth)
    assert bool(ok[vi, ui])
    kind = tol._exact(got, float(ex[vi, ui]), float(ey[vi, ui]))
    return kind, float(metres)


def _flat_road(*, left: float | None, right: float | None, width: float = 3.5) -> RoadPoly:
    import scripts.test_obstacle_labels as tol

    road = tol._road(lambda _y: 0.0, width=width, edges=True)
    count = len(road.z or [])
    road.left_z = None if left is None else [left] * count
    road.right_z = None if right is None else [right] * count
    return road


def check_near_curb_blocked_far_curb_unknown() -> None:
    """0.10 m on main inside 17.34 m is blocked. 0.12 m at 30 m is unknown."""
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN

    half = 3.5 / 2.0
    road = _flat_road(left=0.0, right=0.0)
    # 8 m and 12 m stay inside 17.34 m after the pixel-center ray. 16 m quantizes
    # under curb_eps + pad(r), so it is not the blocked sample.
    for y in (8.0, 12.0):
        kind, metres = _curb_kind(half + 0.3, y, 0.10, road)
        assert metres <= 17.34, (y, metres)
        assert kind == BLOCKED, (y, kind, metres)
    bare = _flat_road(left=None, right=None)
    missing, _metres = _curb_kind(half + 0.3, 8.0, 0.10, bare)
    assert missing == UNKNOWN, missing
    assert missing != DRIVABLE
    far, far_m = _curb_kind(half + 0.3, 30.0, 0.12, road)
    assert far_m > 17.34, far_m
    assert far == UNKNOWN, (far, far_m)
    assert far != DRIVABLE
    print("  0.10 m curb on main at r<=17.34 is blocked; 0.12 m at 30 m is unknown")


def check_drop_left_z_only() -> None:
    """A left-side 0.10 m curb needs left_z. right_z alone leaves it unknown."""
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN

    x = -(3.5 / 2.0 + 0.3)
    both = _curb_kind(x, 8.0, 0.10, _flat_road(left=0.0, right=0.0))[0]
    assert both == BLOCKED, both
    dropped = _curb_kind(x, 8.0, 0.10, _flat_road(left=None, right=0.0))[0]
    assert dropped == UNKNOWN, dropped
    assert dropped != DRIVABLE
    print("  drop left_z only: left curb blocked with it, unknown without it")


def check_drop_right_z_only() -> None:
    """A right-side 0.10 m curb needs right_z. left_z alone leaves it unknown."""
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN

    x = 3.5 / 2.0 + 0.3
    both = _curb_kind(x, 8.0, 0.10, _flat_road(left=0.0, right=0.0))[0]
    assert both == BLOCKED, both
    dropped = _curb_kind(x, 8.0, 0.10, _flat_road(left=0.0, right=None))[0]
    assert dropped == UNKNOWN, dropped
    assert dropped != DRIVABLE
    print("  drop right_z only: right curb blocked with it, unknown without it")


def check_swap_left_right() -> None:
    """The high edge is the left one. Swapping the sides turns that surface into a curb."""
    from python.train.obstacles import BLOCKED, DRIVABLE

    x = -(3.5 / 2.0 + 0.25)
    high_left = _flat_road(left=0.20, right=0.0)
    on_edge = _curb_kind(x, 8.0, 0.20, high_left)[0]
    assert on_edge == DRIVABLE, on_edge
    swapped = _flat_road(left=0.0, right=0.20)
    assert _curb_kind(x, 8.0, 0.20, swapped)[0] == BLOCKED
    print("  swap left/right: the raised left edge is drivable; the swap is blocked")


def check_centerline_z_is_not_the_edge() -> None:
    """A point on a raised edge is the road. Centerline z alone would call it a curb."""
    from python.train.obstacles import BLOCKED, DRIVABLE

    x = 3.5 / 2.0 + 0.3
    raised = _flat_road(left=0.15, right=0.15)
    assert raised.z is not None and raised.z[0] == 0.0
    on_edge = _curb_kind(x, 8.0, 0.15, raised)[0]
    assert on_edge == DRIVABLE, on_edge
    assert on_edge != BLOCKED
    center_only = _flat_road(left=0.0, right=0.0)
    assert _curb_kind(x, 8.0, 0.15, center_only)[0] == BLOCKED
    print("  centerline z is not the edge: a point on the raised edge stays drivable")


def check_pad_025_on_edge_z() -> None:
    """pad(r) blocks a 0.10 m near curb. A flat 0.25 m pad would leave it unknown."""
    from python.train.obstacles import BLOCKED, UNKNOWN

    x = 3.5 / 2.0 + 0.3
    road = _flat_road(left=0.0, right=0.0)
    assert _curb_kind(x, 8.0, 0.10, road)[0] == BLOCKED
    # The same face at 30 m is past the 0.02 m curb pad and stays unknown.
    assert _curb_kind(x, 30.0, 0.12, road)[0] == UNKNOWN
    print("  pad(r) blocks the near 0.10 m curb; 0.25 m would not, and 30 m stays unknown")


def check_ignore_edge_z_file() -> None:
    """The labeler reads this frame's edge_z_file. The other map's flat dump is not a fallback."""
    import label_obstacles as cli
    import scripts.test_obstacle_labels as tol
    from python.collect.road_dump import write_road_dump
    from python.train.obstacles import BLOCKED, UNKNOWN, view_depth_m

    cam = tol._cam("main")
    x, y, z = 3.5 / 2.0 + 0.3, 8.0, 0.10
    hit = view_depth_m(cam, x, y, z, cam.width, cam.height)
    assert hit is not None
    u, v, metres = hit
    ui, vi = int(round(u)), int(round(v))
    depth = np.zeros((cam.height, cam.width), np.uint16)
    depth[vi, ui] = np.uint16(max(1, int(round(float(metres) * 100.0))))
    values, valid = tol._paint(3.5)

    def edges(z_edge: float | None) -> list[dict]:
        out = []
        for station in range(-20, 81, 5):
            left = [-1.75, float(station)] if z_edge is None else [-1.75, float(station), z_edge]
            right = [1.75, float(station)] if z_edge is None else [1.75, float(station), z_edge]
            out.append({"middle": [0.0, float(station), 0.0], "left": left, "right": right})
        return out

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw" / "main_depth"
        raw.mkdir(parents=True)
        np.save(raw / "000000.npy", depth)
        write_road_dump(
            root / "roads" / "flat.json.gz",
            level="flat",
            spawn=None,
            spawn_source="",
            road_network={"kept": {"drivability": 1, "edges": edges(None)}},
        )
        write_road_dump(
            root / "roads" / "edged.json.gz",
            level="edged",
            spawn=None,
            spawn_source="",
            road_network={"kept": {"drivability": 1, "edges": edges(0.0)}},
        )
        row = {
            "i": 0,
            "t": 1.0,
            "map": "flat",
            "road": {"road_id": "kept", "edge_z_file": "roads/edged.json.gz"},
            "ego": {"pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
            "tech": {"lane_gt": values.tolist(), "lane_gt_valid": valid.tolist()},
        }
        (root / "state.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert cli.main([str(root), "--pose-width", "640", "--pose-height", "480"]) == 0
        saved = json.loads((root / "obstacles.jsonl").read_text(encoding="utf-8").splitlines()[0])
        from python.train.obstacles import _unproject

        ex, ey, _ez, _ok = _unproject(cam, depth)
        kind = tol._exact(saved, float(ex[vi, ui]), float(ey[vi, ui]))
        assert kind == BLOCKED, kind
        # No id: the depth fit runs, and this one point has no slope.
        row["road"] = {"edge_z_file": "roads/edged.json.gz"}
        (root / "obstacles.jsonl").unlink()
        (root / "state.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert cli.main([str(root), "--pose-width", "640", "--pose-height", "480"]) == 0
        saved = json.loads((root / "obstacles.jsonl").read_text(encoding="utf-8").splitlines()[0])
        kind = tol._exact(saved, float(ex[vi, ui]), float(ey[vi, ui]))
        assert kind == UNKNOWN, kind
    print("  edge_z_file selects the dump; a missing id does not fall back")


def main() -> None:
    check_recording_keeps_edge_z_and_old_dumps_stay_none()
    check_frame_does_not_copy_the_polyline()
    check_traffic_model_is_recorded_without_a_class_guess()
    check_disk_delta_against_the_two_hour_baseline()
    check_near_curb_blocked_far_curb_unknown()
    check_drop_left_z_only()
    check_drop_right_z_only()
    check_swap_left_right()
    check_centerline_z_is_not_the_edge()
    check_pad_025_on_edge_z()
    check_ignore_edge_z_file()
    print("test_labels_v4_edge_z: OK")


if __name__ == "__main__":
    main()
