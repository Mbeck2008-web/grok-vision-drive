#!/usr/bin/env python3
"""Obstacles sidecar: world-Z clearance over RoadPoly, corridor, training head.

The blocked test is world Z of the unprojected point minus interpolated
RoadPoly.z, compared with OBSTACLE_Z_EPS_M. Ego z and ego pitch are not the
road surface. The corridor is the painted lane about the painted center, or
road_width_m about road_offset_m the way pack_road_edge builds it.
"""
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

from python.debug.gt_geom import RoadPoly  # noqa: E402
from python.debug.lane_targets import pack_lane_gt  # noqa: E402
from python.train.sensor_aux import pack_road_edge  # noqa: E402


class _Pose:
    """Vehicle-frame camera. Same fields the labeler reads off cameras.yaml."""

    def __init__(self) -> None:
        self.cam_id = "main"
        self.x = 0.0
        self.y = 1.20
        self.z = 1.26
        self.yaw_deg = 0.0
        self.pitch_deg = -6.0
        self.fov_v_deg = 34.0
        self.width = 120
        self.height = 90


POSE = _Pose()


def _axes(forward, up):
    fx, fy, fz = (float(v) for v in forward)
    ux, uy, uz = (float(v) for v in up)
    right = (fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux)
    return right, (fx, fy, fz), (ux, uy, uz)


def _to_ego(world, pos, right, forward, up):
    rel = tuple(float(world[i]) - float(pos[i]) for i in range(3))
    return tuple(
        rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2]
        for axis in (right, forward, up)
    )


def _flat_axes():
    return _axes((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _paint(width=3.5, offset=0.0, heading=0.0):
    return pack_lane_gt(
        lane_offset_m=offset,
        heading_err_rad=heading,
        lane_width_m=width,
        off_road=False,
    )


def _road(z_at, width=8.0, *, edges=True):
    ys = list(range(-20, 81, 5))
    z = [float(z_at(float(y))) for y in ys]
    return RoadPoly(
        road_id="synth",
        xy=[(0.0, float(y)) for y in ys],
        width_m=float(width),
        z=z,
        left_z=list(z) if edges else None,
        right_z=list(z) if edges else None,
    )


def _with_slopes(road, left_slope: float, right_slope: float):
    """Edge heights for per-side slopes. Positive slope raises the ego-right side."""
    half = float(road.width_m) * 0.5
    road.left_z = [None if z is None else float(z) + float(left_slope) * (-half) for z in road.z]
    road.right_z = [None if z is None else float(z) + float(right_slope) * half for z in road.z]
    return road


def _splat(depth, ego_xyz):
    from python.train.obstacles import view_depth_m

    hit = view_depth_m(POSE, ego_xyz[0], ego_xyz[1], ego_xyz[2], depth.shape[1], depth.shape[0])
    assert hit is not None, ego_xyz
    u, v, metres = hit
    ui, vi = int(round(u)), int(round(v))
    assert 0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0], (ui, vi, ego_xyz)
    cm = int(round(float(metres) * 100.0))
    assert cm > 0, (cm, ego_xyz)
    if depth[vi, ui] == 0 or cm < int(depth[vi, ui]):
        depth[vi, ui] = np.uint16(cm)
    return vi, ui


def _blank(h, w):
    return np.zeros((h, w), np.uint16)


def _cell(x, y):
    from python.train.obstacles import GRID_H, GRID_W, X_MAX, X_MIN, Y_MAX, Y_MIN

    col = int(math.floor((float(x) - X_MIN) / (X_MAX - X_MIN) * GRID_W))
    row = int(math.floor((float(y) - Y_MIN) / (Y_MAX - Y_MIN) * GRID_H))
    return row, col


def _exact(record, x, y):
    row, col = _cell(x, y)
    mask = np.asarray(record["mask"], dtype=np.int64).reshape(record["grid_h"], record["grid_w"])
    return int(mask[row, col])


def _at(record, x, y):
    """Class at the ego cell. A point on a grid line may round into a neighbor."""
    from python.train.obstacles import BLOCKED, DRIVABLE, GRID_W

    row, col = _cell(x, y)
    mask = np.asarray(record["mask"], dtype=np.int64).reshape(record["grid_h"], record["grid_w"])
    assert 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1], (row, col, x, y)
    found = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = row + dr, col + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and int(mask[rr, cc]) != 0:
                found.append(int(mask[rr, cc]))
    if BLOCKED in found:
        kind = BLOCKED
    elif DRIVABLE in found:
        kind = DRIVABLE
    else:
        kind = int(mask[row, col])
    return kind, row, col, GRID_W


def _label(depth, *, lane=None, road=None, pos=(0.0, 0.0, 0.0), forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0), **kw):
    from python.train.obstacles import label_obstacle_frame

    values, valid = lane if lane is not None else (None, None)
    return label_obstacle_frame(
        depth,
        pose=POSE,
        lane_gt=values,
        lane_gt_valid=valid,
        road=road,
        ego_pos=pos,
        ego_dir=forward,
        ego_up=up,
        cam="main",
        t=1.0,
        **kw,
    )


def check_pitched_and_raised_ego_uses_world_z() -> None:
    """A mixed frame marks the wrong cells.

    The vehicle is 4 m above a road whose world Z is about 5, and it is
    pitched nose-down 15 deg. The road sample's ego z is above the epsilon
    (a level-ego comparison would call it an obstacle) and the wall sample's
    ego z is under the epsilon (that comparison would miss it). World Z minus
    interpolated RoadPoly.z does the opposite.
    """
    from python.train.obstacles import BLOCKED, DRIVABLE, OBSTACLE_Z_EPS_M, UNKNOWN

    pitch = math.radians(-15.0)
    forward = (0.0, math.cos(pitch), math.sin(pitch))
    up = (0.0, -math.sin(pitch), math.cos(pitch))
    right, forward, up = _axes(forward, up)
    pos = (0.0, 0.0, 9.0)

    def surface(y):
        return 5.0 + 0.02 * float(y)

    road_w = (0.0, 18.0, surface(18.0))
    wall_w = (0.0, 10.0, surface(10.0) + float(OBSTACLE_Z_EPS_M) + 1.0)
    road_e = _to_ego(road_w, pos, right, forward, up)
    wall_e = _to_ego(wall_w, pos, right, forward, up)
    assert road_e[2] > float(OBSTACLE_Z_EPS_M), road_e
    assert wall_e[2] <= float(OBSTACLE_Z_EPS_M), wall_e
    assert abs(road_w[2] - surface(18.0)) < 1e-9
    assert wall_w[2] - surface(10.0) > float(OBSTACLE_Z_EPS_M)

    depth = _blank(POSE.height, POSE.width)
    _splat(depth, road_e)
    _splat(depth, wall_e)
    got = _label(depth, lane=_paint(), road=_road(surface), pos=pos, forward=forward, up=up)
    assert got["mask_src"] == "geom", got["mask_src"]
    road_cls, rr, rc, _gw = _at(got, road_e[0], road_e[1])
    wall_cls, wr, wc, _gw = _at(got, wall_e[0], wall_e[1])
    assert (rr, rc) != (wr, wc), ((rr, rc), (wr, wc))
    assert road_cls == DRIVABLE, (road_cls, road_e, road_w)
    assert wall_cls == BLOCKED, (wall_cls, wall_e, wall_w)

    # Level vehicle, origin lifted off the road. Ego z of the wall is negative.
    flat_r, flat_f, flat_u = _flat_axes()
    high = (0.0, 0.0, 8.0)
    road2_w = (0.0, 12.0, 5.0)
    wall2_w = (0.0, 18.0, 5.0 + float(OBSTACLE_Z_EPS_M) + 1.0)
    road2_e = _to_ego(road2_w, high, flat_r, flat_f, flat_u)
    wall2_e = _to_ego(wall2_w, high, flat_r, flat_f, flat_u)
    assert wall2_e[2] < 0.0 < float(OBSTACLE_Z_EPS_M)
    assert road2_e[2] < -float(OBSTACLE_Z_EPS_M)
    depth2 = _blank(POSE.height, POSE.width)
    _splat(depth2, road2_e)
    _splat(depth2, wall2_e)
    got2 = _label(
        depth2,
        lane=_paint(),
        road=_road(lambda _y: 5.0),
        pos=high,
        forward=flat_f,
        up=flat_u,
    )
    assert _at(got2, road2_e[0], road2_e[1])[0] == DRIVABLE, road2_e
    assert _at(got2, wall2_e[0], wall2_e[1])[0] == BLOCKED, wall2_e
    print("  pitched and z-offset ego: blocked is world Z minus RoadPoly.z")


def _recovered_z(ego) -> float:
    from python.train.obstacles import _unproject

    depth = _blank(POSE.height, POSE.width)
    vi, ui = _splat(depth, ego)
    _x, _y, z, _valid = _unproject(POSE, depth)
    return float(z[vi, ui])


def _assert_row_fields(record) -> None:
    """clear_eps_m, curb_eps_m, and edge_band_m are on every obstacles row."""
    assert record["clear_eps_m"] == 0.15, record.get("clear_eps_m")
    assert record["curb_eps_m"] == 0.08, record.get("curb_eps_m")
    assert record["edge_band_m"] == 0.5, record.get("edge_band_m")


def check_epsilon_is_world_clearance() -> None:
    from python.train.obstacles import BLOCKED, DRIVABLE, OBSTACLE_Z_EPS_M, UNKNOWN

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    below = (0.0, 8.0, -0.5)
    on = (0.0, 14.0, float(OBSTACLE_Z_EPS_M) * 0.5)
    over = (0.0, 22.0, float(OBSTACLE_Z_EPS_M) + 0.5)
    depth = _blank(POSE.height, POSE.width)
    egos = []
    for world in (below, on, over):
        ego = _to_ego(world, pos, right, forward, up)
        _splat(depth, ego)
        egos.append(ego)
    got = _label(depth, lane=_paint(), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
    assert _at(got, *egos[0][:2])[0] == DRIVABLE, egos[0]
    assert _at(got, *egos[1][:2])[0] == DRIVABLE, egos[1]
    assert _at(got, *egos[2][:2])[0] == BLOCKED, egos[2]
    # Literal 0.15. A test that only reads the constant still passes at 0.30.
    assert float(OBSTACLE_Z_EPS_M) == 0.15
    near = (0.0, 8.0, 0.20)
    depth_pin = _blank(POSE.height, POSE.width)
    pin_ego = _to_ego(near, pos, right, forward, up)
    _splat(depth_pin, pin_ego)
    pinned = _label(depth_pin, lane=_paint(), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
    assert _at(pinned, pin_ego[0], pin_ego[1])[0] != DRIVABLE, (pin_ego, near)
    print(f"  world clearance epsilon is {OBSTACLE_Z_EPS_M} m")


def check_corridor_matches_paint_then_pack_road_edge() -> None:
    """Painted lane width about the painted center. Else pack_road_edge."""
    from python.train.obstacles import BLOCKED, UNKNOWN

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    road = _road(lambda _y: 0.0, width=8.0)
    outside_paint = (2.5, 12.0, 1.0)
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, outside_paint)
    painted = _label(depth, lane=_paint(3.5), road=road, pos=pos, forward=forward, up=up)
    assert _at(painted, outside_paint[0], outside_paint[1])[0] == UNKNOWN, painted["mask_src"]

    wide = _label(
        depth,
        lane=(None, None),
        road=road,
        pos=pos,
        forward=forward,
        up=up,
        road_width_m=8.0,
        road_offset_m=0.0,
        road_heading_err_rad=0.0,
    )
    assert _at(wide, outside_paint[0], outside_paint[1])[0] == BLOCKED

    values, valid = pack_road_edge(offset_m=1.5, heading_err_rad=0.0, road_width_m=6.0)
    # y=10 m is lookahead index 1: left, right.
    left, right_x = float(values[2]), float(values[3])
    assert valid[2] > 0.5 and valid[3] > 0.5
    center = 0.5 * (left + right_x)
    assert abs(center - (-1.5)) < 1e-5, center
    inside = (center + 0.25, 12.0, 1.0)
    outside = (right_x + 0.6, 12.0, 1.0)
    depth2 = _blank(POSE.height, POSE.width)
    _splat(depth2, inside)
    _splat(depth2, outside)
    edged = _label(
        depth2,
        lane=(None, None),
        road=road,
        pos=pos,
        forward=forward,
        up=up,
        road_width_m=6.0,
        road_offset_m=1.5,
        road_heading_err_rad=0.0,
    )
    assert _at(edged, inside[0], inside[1])[0] == BLOCKED, (left, right_x)
    assert _at(edged, outside[0], outside[1])[0] == UNKNOWN, (left, right_x)
    print("  corridor is painted width, else pack_road_edge center +/- half width")


def check_wall_pole_empty_and_missing_depth() -> None:
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    road = _road(lambda _y: 0.0)
    lane = _paint()

    def scene(points):
        depth = _blank(POSE.height, POSE.width)
        for point in points:
            _splat(depth, point)
        return _label(depth, lane=lane, road=road, pos=pos, forward=forward, up=up)

    ground = [(x, y, 0.0) for y in (6.0, 10.0, 18.0, 26.0) for x in (-1.0, 0.0, 1.0)]
    wall = [(x, 15.0, z) for x in (-0.8, 0.0, 0.8) for z in (0.6, 1.2)]
    got = scene(ground + wall)
    assert got["mask_src"] == "geom"
    assert _at(got, 0.0, 15.0)[0] == BLOCKED
    assert _at(got, 0.0, 10.0)[0] == DRIVABLE
    assert got["clear_valid"] is True
    assert abs(float(got["clear_m"]) - 15.0) < 0.75, got["clear_m"]

    pole = [(0.3 + dx, 12.0, z) for dx in (-0.15, 0.0, 0.15) for z in (0.6, 1.4)]
    got_p = scene(ground + pole)
    assert _at(got_p, 0.3, 12.0)[0] == BLOCKED
    assert _at(got_p, 0.0, 10.0)[0] == DRIVABLE
    assert got_p["clear_valid"] is True
    assert abs(float(got_p["clear_m"]) - 12.0) < 0.75, got_p["clear_m"]

    both = scene(ground + pole + [(0.0, 20.0, 1.2)])
    assert abs(float(both["clear_m"]) - 12.0) < 0.75, both["clear_m"]

    empty = scene(ground)
    mask = np.asarray(empty["mask"], dtype=np.int64)
    assert int((mask == BLOCKED).sum()) == 0, int((mask == BLOCKED).sum())
    assert _at(empty, 0.0, 10.0)[0] == DRIVABLE
    assert empty["clear_valid"] is False
    assert empty["clear_m"] is None

    missing = _label(_blank(POSE.height, POSE.width), lane=lane, road=road, pos=pos, forward=forward, up=up)
    none_depth = _label(None, lane=lane, road=road, pos=pos, forward=forward, up=up)
    for record in (missing, none_depth):
        mask = np.asarray(record["mask"], dtype=np.int64)
        assert int(mask.max()) == UNKNOWN, record["mask_src"]
        assert record["clear_valid"] is False
        assert record["clear_m"] is None
    print("  wall, pole, empty road, and missing depth")


def check_missing_geometry_stays_unknown() -> None:
    from python.train.obstacles import UNKNOWN

    right, forward, up = _flat_axes()
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, (0.0, 12.0, 1.2))
    no_z = RoadPoly("no_z", xy=[(0.0, -10.0), (0.0, 40.0)], width_m=8.0, z=None)
    got = _label(depth, lane=_paint(), road=no_z, pos=(0.0, 0.0, 0.0), forward=forward, up=up)
    assert int(np.asarray(got["mask"]).max()) == UNKNOWN

    no_corridor = _label(
        depth,
        lane=(None, None),
        road=_road(lambda _y: 0.0),
        pos=(0.0, 0.0, 0.0),
        forward=forward,
        up=up,
    )
    assert int(np.asarray(no_corridor["mask"]).max()) == UNKNOWN
    print("  missing RoadPoly.z or corridor stays unknown")


def check_clearance_masked_without_near_centers() -> None:
    values, valid = _paint()
    valid = valid.copy()
    valid[0] = 0.0
    valid[3] = 0.0
    right, forward, up = _flat_axes()
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, (0.0, 15.0, 1.2))
    got = _label(
        depth,
        lane=(values, valid),
        road=_road(lambda _y: 0.0),
        pos=(0.0, 0.0, 0.0),
        forward=forward,
        up=up,
    )
    assert got["clear_valid"] is False
    assert got["clear_m"] is None
    print("  clearance masked when the 5 m and 10 m centers are invalid")


def check_semantic_blocked_needs_palette_proof() -> None:
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN, proven_blocked_names

    classes = {
        "kind": "gvd_sem_classes",
        "train_only": True,
        "classes": {"0": "unknown", "1": "ROAD", "2": "WALL", "3": "PROP", "4": "TREE"},
    }
    assert proven_blocked_names(classes, None) == frozenset()
    assert proven_blocked_names(classes, ["PROP", "OBJECT", "not-in-file"]) == frozenset()
    assert proven_blocked_names(classes, ["WALL", "PROP"]) == frozenset({"WALL"})
    assert "TREE" not in proven_blocked_names(classes, None)

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    # WALL sitting on the surface: geometry says drivable, a proven name says blocked.
    # Tall PROP: geometry says blocked, an unproven coarse name stays unknown in sem mode.
    depth = _blank(POSE.height, POSE.width)
    sem = np.zeros(depth.shape, np.uint8)
    wall_px = _splat(depth, (0.0, 16.0, 0.0))
    prop_px = _splat(depth, (0.0, 24.0, 1.5))
    road_px = _splat(depth, (0.0, 8.0, 0.0))
    sem[wall_px] = 2
    sem[prop_px] = 3
    sem[road_px] = 1
    kw = dict(pos=pos, forward=forward, up=up, road=_road(lambda _y: 0.0), lane=_paint(), sem=sem, sem_classes=classes)

    geom = _label(depth, **kw)
    assert geom["mask_src"] == "geom"
    assert _at(geom, 0.0, 16.0)[0] == DRIVABLE
    assert _at(geom, 0.0, 24.0)[0] == BLOCKED

    proved = _label(depth, blocked_names=["WALL"], **kw)
    assert proved["mask_src"] == "sem"
    assert _at(proved, 0.0, 16.0)[0] == BLOCKED
    assert _at(proved, 0.0, 24.0)[0] == UNKNOWN
    assert _at(proved, 0.0, 8.0)[0] == DRIVABLE

    coarse = _label(depth, blocked_names=["PROP", "OBJECT"], **kw)
    assert coarse["mask_src"] == "geom"

    dead = _blank(POSE.height, POSE.width)
    sem_dead = np.full(dead.shape, 2, np.uint8)
    masked = _label(
        dead,
        blocked_names=["WALL"],
        sem=sem_dead,
        sem_classes=classes,
        lane=_paint(),
        road=_road(lambda _y: 0.0),
        pos=pos,
        forward=forward,
        up=up,
    )
    assert int(np.asarray(masked["mask"]).max()) == UNKNOWN
    print("  semantic blocked cells only after palette proof; depth 0 stays unknown")


def check_writer_and_missing_sidecar() -> None:
    import label_obstacles as cli
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN
    from python.train.scene_net import labels_for_row

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, (0.0, 10.0, 0.0))
    _splat(depth, (0.0, 16.0, 1.2))
    values, valid = _paint()
    edges = []
    for y in range(-20, 81, 5):
        edges.append(
            {
                "middle": [0.0, float(y), 0.0],
                "left": [-3.5, float(y), 0.0],
                "right": [3.5, float(y), 0.0],
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw" / "main_depth"
        raw.mkdir(parents=True)
        np.save(raw / "000000.npy", depth)
        row = {
            "t": 1.25,
            "i": 0,
            "ego": {"pos": list(pos), "dir": list(forward), "up": list(up)},
            "tech": {
                "lane_gt": values.tolist(),
                "lane_gt_valid": valid.tolist(),
                "road_width_m": 7.0,
                "road_offset_m": 0.0,
                "heading_err_rad": 0.0,
            },
        }
        row["map"] = "synth"
        row["road"] = {"road_id": "synth", "edge_z_file": "roads/synth.json.gz"}
        (root / "state.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        from python.collect.road_dump import write_road_dump

        write_road_dump(
            root / "roads" / "synth.json.gz",
            level="synth",
            spawn=None,
            spawn_source="",
            road_network={"synth": {"drivability": 1, "edges": edges}},
        )
        code = cli.main([str(root), "--pose-width", "120", "--pose-height", "90"])
        assert code == 0, code
        out = root / "obstacles.jsonl"
        assert out.is_file()
        saved = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert saved["t"] == 1.25 and saved["cam"] == "main"
        assert saved["grid_h"] == 64 and saved["grid_w"] == 64
        assert saved["x_min"] == -16.0 and saved["x_max"] == 16.0
        assert saved["y_min"] == 0.0 and saved["y_max"] == 40.0
        _assert_row_fields(saved)
        assert _at(saved, 0.0, 10.0)[0] == DRIVABLE
        assert _at(saved, 0.0, 16.0)[0] == BLOCKED
        assert not (root / "gt.jsonl").exists()

        code = cli.main([str(root), "--out", "gt.jsonl"])
        assert code == 2
        assert not (root / "gt.jsonl").exists()

        labels = labels_for_row({"i": 3, "t": 9.0, "tech": {}}, root=root, obstacles=True)
        assert float(labels["obstacle_present"]) == 0.0
        assert int(labels["obstacle_mask"].max()) == UNKNOWN
        assert float(labels["obstacle_clear_valid"]) == 0.0
        loaded = labels_for_row({"i": 0, "t": 1.25, "tech": {}}, root=root, obstacles=True)
        assert float(loaded["obstacle_present"]) == 1.0
        assert int(loaded["obstacle_mask"][_cell(0.0, 16.0)]) == BLOCKED
        plain = labels_for_row({"i": 0, "t": 1.25, "tech": {}})
        assert "obstacle_mask" not in plain
    print("  writer stores obstacles.jsonl and a missing sidecar stays masked")


def _visible(x: float, y: float, z: float) -> bool:
    from python.train.obstacles import view_depth_m

    hit = view_depth_m(POSE, x, y, z, POSE.width, POSE.height)
    if hit is None:
        return False
    u, v, _metres = hit
    return 0.0 <= u < POSE.width and 0.0 <= v < POSE.height


def _surface_points(xs, ys, z_at) -> list[tuple[float, float, float]]:
    points = []
    for y in ys:
        for x in xs:
            z = float(z_at(float(x)))
            if _visible(float(x), float(y), z):
                points.append((float(x), float(y), z))
    return points


def _geom_road(points, road_width: float, *, left_slope: float = 0.0, right_slope: float = 0.0):
    right, forward, up = _flat_axes()
    depth = _blank(POSE.height, POSE.width)
    for point in points:
        _splat(depth, point)
    road = _with_slopes(_road(lambda _y: 0.0, width=road_width), left_slope, right_slope)
    return _label(
        depth,
        lane=(None, None),
        road=road,
        pos=(0.0, 0.0, 0.0),
        forward=forward,
        up=up,
        road_width_m=float(road_width),
        road_offset_m=0.0,
        road_heading_err_rad=0.0,
    )


def check_bump_survives_depth_quantization() -> None:
    """A 0.10 m bump is not blocked, including at 30 m.

    uint16 centimetres plus the pixel-center ray make a flat road at 27.4 m
    come back near z 0.026, and a 0.12 m bump there come back above 0.15.
    On this pose ``pad(r)`` is the depth term only. A short bump stays under the
    blocked bar. A far bump is not drivable.
    """
    from python.train.obstacles import BLOCKED, DRIVABLE, depth_range_pad_m

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    road = _road(lambda _y: 0.0)
    lane = _paint()

    def kind(world):
        depth = _blank(POSE.height, POSE.width)
        ego = _to_ego(world, pos, right, forward, up)
        _splat(depth, ego)
        got = _label(depth, lane=lane, road=road, pos=pos, forward=forward, up=up)
        return _at(got, ego[0], ego[1])[0], got

    flat_z = _recovered_z((0.0, 27.4, 0.0))
    bump_z = _recovered_z((0.0, 27.4, 0.12))
    assert abs(flat_z - 0.026) < 0.005, flat_z
    assert bump_z > 0.15, bump_z
    pad = depth_range_pad_m(np.array([0.0, 8.0, 27.4, 30.0]), POSE)
    assert float(pad[-1]) > float(pad[1]) > 0.005
    assert float(np.max(pad)) <= 0.25
    near, _got = kind((0.0, 8.0, 0.10))
    far, far_row = kind((0.0, 30.0, 0.10))
    quant, _qrow = kind((0.0, 27.4, 0.12))
    real, _rrow = kind((0.0, 27.4, 0.40))
    assert near != BLOCKED, near
    assert far != DRIVABLE, far
    assert quant != DRIVABLE, (quant, bump_z)
    assert real != DRIVABLE, real
    _assert_row_fields(far_row)
    print("  pad(r): a near 0.10 m bump is not blocked; a far quantized bump is not drivable")


def check_crown_and_bank_not_blocked() -> None:
    """A lateral slope from left_z/right_z is the road, including wide corridors.

    Centerline Z alone marks the high side blocked. Edge heights supply the slope.
    """
    from python.train.obstacles import BLOCKED, DRIVABLE

    def assert_surface(points, road_width: float, *, left_slope: float, right_slope: float) -> None:
        got = _geom_road(points, road_width, left_slope=left_slope, right_slope=right_slope)
        _assert_row_fields(got)
        mask = np.asarray(got["mask"], dtype=np.int64)
        assert int((mask == BLOCKED).sum()) == 0, int((mask == BLOCKED).sum())
        for x, y, _z in points:
            kind = _at(got, x, y)[0]
            assert kind != BLOCKED, (x, y, kind, left_slope, right_slope)

    bank_w = 8.0
    bank = _surface_points(
        [i * 0.5 for i in range(-8, 9)],
        (12.0, 16.0),
        lambda x: 0.06 * x,
    )
    assert len(bank) >= 8 and max(abs(p[0]) for p in bank) > 2.5
    assert_surface(bank, bank_w, left_slope=0.06, right_slope=0.06)

    crown_w = 16.0
    crown = _surface_points(
        [i * 0.5 for i in range(-16, 17)],
        (16.0, 20.0, 24.0),
        lambda x: -0.02 * abs(x),
    )
    assert len(crown) >= 8 and max(abs(p[0]) for p in crown) > 2.5
    assert_surface(crown, crown_w, left_slope=0.02, right_slope=-0.02)

    # 2% cross-slope across a corridor wider than 5 m. The high side is
    # above 0.15 m of centerline Z and still the road.
    slope_w = 20.0
    slope = _surface_points(
        [i * 0.5 for i in range(0, 19)],
        (16.0, 20.0, 24.0, 28.0),
        lambda x: 0.02 * x,
    )
    assert len(slope) >= 8 and max(p[0] for p in slope) > 7.5
    assert_surface(slope, slope_w, left_slope=0.02, right_slope=0.02)

    # One high edge point, no edge z and no 20-inlier fit: unknown, not blocked.
    from python.train.obstacles import UNKNOWN

    lone = _surface_points((3.5,), (12.0,), lambda x: 0.06 * x)
    assert lone
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, lone[0])
    bare = _label(
        depth,
        lane=(None, None),
        road=_road(lambda _y: 0.0, width=bank_w, edges=False),
        pos=(0.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        road_width_m=bank_w,
        road_offset_m=0.0,
        road_heading_err_rad=0.0,
    )
    assert _at(bare, lone[0][0], lone[0][1])[0] == UNKNOWN, lone
    print("  2% crown and 6% bank: 0 in-corridor blocks on a wide corridor")


def check_heading_matches_pack_road_edge() -> None:
    """Nonzero heading uses the same sin term as pack_road_edge.

    Flipping the sign of sin moves the center to the other side of the car.
    """
    from python.debug.lane_targets import LOOKAHEADS_M
    from python.train.obstacles import BLOCKED, UNKNOWN

    heading = 0.30
    offset = 0.5
    width = 6.0
    y = float(LOOKAHEADS_M[1])
    values, valid = pack_road_edge(offset_m=offset, heading_err_rad=heading, road_width_m=width)
    left, right_x = float(values[2]), float(values[3])
    assert float(valid[2]) > 0.5 and float(valid[3]) > 0.5
    packed = 0.5 * (left + right_x)
    expect = -offset + y * math.sin(heading)
    flipped = -offset - y * math.sin(heading)
    assert abs(packed - expect) < 1e-4, (packed, expect)
    assert abs(packed - flipped) > 1.0, (packed, flipped)

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    road = _road(lambda _y: 0.0, width=width)
    # y=16 keeps both the true center and the sign-flipped center in the image,
    # and the flipped center is outside this corridor's edge band.
    y_img = 16.0
    center = -offset + y_img * math.sin(heading)
    other = -offset - y_img * math.sin(heading)
    half = width * 0.5
    assert abs(center - other) > half + 0.5
    inside = (center, y_img, 1.2)
    outside = (other, y_img, 1.2)
    assert _visible(*inside) and _visible(*outside), (inside, outside)
    depth = _blank(POSE.height, POSE.width)
    _splat(depth, inside)
    _splat(depth, outside)
    got = _label(
        depth,
        lane=(None, None),
        road=road,
        pos=pos,
        forward=forward,
        up=up,
        road_width_m=width,
        road_offset_m=offset,
        road_heading_err_rad=heading,
    )
    assert _at(got, inside[0], inside[1])[0] == BLOCKED, (center, other)
    assert _at(got, outside[0], outside[1])[0] == UNKNOWN, (center, other)
    print("  nonzero heading matches pack_road_edge; a sin sign flip misses the corridor")


def check_edge_heights_are_kept() -> None:
    """left_z and right_z survive parsing, the loader, and a dump.

    A middle with no height stays missing. The slope uses the raw edge heights.
    A record with no edge height leaves the fields None.
    """
    from python.collect.road_dump import load_road_dump, write_road_dump
    from python.collect.road_loader import polys_from_network
    from python.debug.gt_geom import parse_road_edges
    from python.train.obstacles import BLOCKED

    kept = [
        {"middle": [0.0, 4.0 * k, 6.0], "left": [-1.75, 4.0 * k, 5.0], "right": [1.75, 4.0 * k, 7.0]}
        for k in range(4)
    ]
    poly = parse_road_edges("kept", kept)
    assert poly is not None
    assert poly.z == [6.0, 6.0, 6.0, 6.0]
    assert poly.left_z == [5.0, 5.0, 5.0, 5.0]
    assert poly.right_z == [7.0, 7.0, 7.0, 7.0]
    bare = [
        {"middle": [0.0, 4.0 * k], "left": [-1.75, 4.0 * k, 5.0], "right": [1.75, 4.0 * k, 7.0]}
        for k in range(3)
    ]
    got = parse_road_edges("bare", bare)
    assert got is not None and got.z is None
    assert got.left_z == [5.0, 5.0, 5.0] and got.right_z == [7.0, 7.0, 7.0]
    slope = (got.right_z[0] - got.left_z[0]) / got.width_m
    assert abs(slope - (2.0 / 3.5)) < 1e-6, slope
    low = [
        {"middle": [0.0, 4.0 * k], "left": [-1.75, 4.0 * k, -1.0], "right": [1.75, 4.0 * k, 1.0]}
        for k in range(4)
    ]
    low_poly = parse_road_edges("low", low)
    assert low_poly is not None and low_poly.z is None
    plane = 0.0 + 0.5 * ((1.0 - (-1.0)) / low_poly.width_m)
    tall = (0.5, 8.0, plane + 0.8)
    depth = _blank(POSE.height, POSE.width)
    ego = _to_ego(tall, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    vi, ui = _splat(depth, ego)
    from python.train.obstacles import _unproject
    ex, ey, _ez, _ok = _unproject(POSE, depth)
    labeled = _label(
        depth, lane=_paint(3.5), road=low_poly, pos=(0.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
    )
    assert _exact(labeled, float(ex[vi, ui]), float(ey[vi, ui])) == BLOCKED, (float(ex[vi, ui]), float(ey[vi, ui]))
    loaded = polys_from_network({"bare": {"edges": bare}})
    assert loaded and loaded[0].left_z == [5.0, 5.0, 5.0] and loaded[0].right_z == [7.0, 7.0, 7.0]
    missing_edges = [
        {"middle": [0.0, 0.0, 1.0], "left": [-1.0, 0.0], "right": [1.0, 0.0]},
        {"middle": [0.0, 10.0, 1.0], "left": [-1.0, 10.0], "right": [1.0, 10.0]},
    ]
    missing = parse_road_edges("old", missing_edges)
    assert missing is not None and missing.left_z is None and missing.right_z is None
    plain = RoadPoly("old-ctor", [(0.0, 0.0), (0.0, 10.0)], 7.0)
    assert plain.left_z is None and plain.right_z is None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "map.json.gz"
        write_road_dump(
            path,
            level="synth",
            spawn=None,
            spawn_source="",
            road_network={"bare": {"edges": bare}},
        )
        polys, _payload = load_road_dump(path)
    assert polys and polys[0].left_z == [5.0, 5.0, 5.0] and polys[0].right_z == [7.0, 7.0, 7.0]
    print("  left_z/right_z survive parse, loader, and dump; missing edges stay None")


def check_depth_fit_keeps_curb_blocked() -> None:
    """Depth-fit 0.10 m curb stays out of drivable. Edge-z blocks it where pad <= 0.02.

    Seeds are paved cells with |h_centerline| <= 0.05 m, in a 5-row window.
    Four cells per row times five rows is the 20-inlier minimum. Taller points
    on the same cells are excluded by that gate. Dropping it lets them outvote
    the road, and nothing stays blocked.
    """
    from python.train.obstacles import BLOCKED, DRIVABLE

    classes = {
        "kind": "gvd_sem_classes",
        "train_only": True,
        "classes": {"0": "unknown", "1": "ROAD"},
    }
    half = 3.5 / 2.0
    depth = _blank(POSE.height, POSE.width)
    sem = np.zeros(depth.shape, np.uint8)

    def paint(point) -> None:
        sem[_splat(depth, point)] = 1

    # Four lateral bins inside the corridor. The road pixel is one sample.
    # Three taller pixels share its BEV cell and lie on h = 0.55 x. Nine rows
    # so a middle row's 5-row window holds 4 x 5 = 20 road cells.
    bins = (
        (0.08, 0.20, 0.32, 0.44),
        (0.58, 0.70, 0.82, 0.94),
        (1.08, 1.20, 1.32, 1.44),
        (1.54, 1.60, 1.66, 1.72),
    )
    for row in range(14, 23):
        y = (row + 0.5) * 0.625
        for xs in bins:
            paint((xs[1], y, 0.0))
            for x in (xs[0], xs[2], xs[3]):
                paint((x, y, 0.55 * x))
    curb = (half + 0.3, (18 + 0.5) * 0.625, 0.10)
    paint(curb)
    got = _label(
        depth,
        lane=_paint(3.5),
        road=_road(lambda _y: 0.0, edges=False),
        pos=(0.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        sem=sem,
        sem_classes=classes,
    )
    assert _exact(got, curb[0], curb[1]) != DRIVABLE, curb
    assert int((np.asarray(got["mask"]) == BLOCKED).sum()) > 0
    # Edge-z path, where pad(r) <= 0.02: the same 0.10 m curb is blocked.
    from python.train.obstacles import BLOCKED as _BLOCKED
    edge_kind, _counts = _one_point(_cam("main"), 1.75 + 0.3, 8.0, 0.10, edges=True, seeds=False)
    assert edge_kind == _BLOCKED, edge_kind
    print("  depth fit: 0.10 m curb is not drivable; edge-z curb at 8 m is blocked")


def _cam(kind: str):
    if kind == "main":
        return _Cam(x=0.0, y=1.20, z=1.26, yaw_deg=0.0, pitch_deg=-6.0, fov_v_deg=34.0, width=640, height=480)
    return _Cam(x=-0.03, y=1.20, z=1.24, yaw_deg=0.0, pitch_deg=-8.0, fov_v_deg=88.7, width=640, height=360)


class _Cam:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def check_interim_pad_band() -> None:
    """pad(r) on the edge-z path, pad(r)+fit_tol on the depth fit.

    A 0.16 m or 0.20 m face on the fit path is never drivable at 12/20/27.4/30 m
    on main and wide. pad=0 on the edge-z path makes one of those faces drivable.
    """
    import python.train.obstacles as obstacles
    from python.train.obstacles import DRIVABLE, depth_range_pad_m, label_obstacle_frame, view_depth_m, _unproject

    main, wide = _cam("main"), _cam("wide")
    published = (
        (main, 17.34, 0.019996, True),
        (main, 17.35, 0.020004, False),
        (main, 27.4, 0.0287, False),
        (main, 30.0, 0.0309, False),
        (wide, 5.54, 0.019976, True),
        (wide, 30.0, 0.0861, False),
    )
    for cam, dist, want, under in published:
        got = float(depth_range_pad_m(np.array([dist]), cam)[0])
        assert abs(got - want) < 5e-4, (dist, got, want)
        assert (got <= 0.02) is under, (dist, got)

    classes = {"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "unknown", "1": "ROAD"}}

    def splat(depth, sem, cam, x, y, z, paved):
        hit = view_depth_m(cam, x, y, z, cam.width, cam.height)
        assert hit is not None, (x, y, z, cam.width)
        u, v, metres = hit
        ui, vi = int(round(u)), int(round(v))
        assert 0 <= ui < cam.width and 0 <= vi < cam.height, (ui, vi, x, y, z)
        depth[vi, ui] = np.uint16(max(1, int(round(float(metres) * 100.0))))
        if paved and sem is not None:
            sem[vi, ui] = 1
        return vi, ui

    def label_at(cam, depth, sem, vi, ui, *, edges):
        values, valid = _paint(3.5)
        got = label_obstacle_frame(
            depth,
            pose=cam,
            lane_gt=values,
            lane_gt_valid=valid,
            road=_road(lambda _y: 0.0, edges=edges),
            ego_pos=(0.0, 0.0, 0.0),
            ego_dir=(0.0, 1.0, 0.0),
            ego_up=(0.0, 0.0, 1.0),
            sem=sem,
            sem_classes=classes if sem is not None else None,
            cam="probe",
            t=1.0,
        )
        ex, ey, _ez, _ok = _unproject(cam, depth)
        return _exact(got, float(ex[vi, ui]), float(ey[vi, ui]))

    def fallback_face(cam, y, height):
        depth = np.zeros((cam.height, cam.width), np.uint16)
        sem = np.zeros(depth.shape, np.uint8)
        center = int(math.floor(float(y) / 0.625))
        for row in range(center - 4, center + 5):
            yy = (row + 0.5) * 0.625
            if yy < 2.0:
                continue
            for x in (0.25, 0.75, 1.25, 1.60):
                splat(depth, sem, cam, x, yy, 0.0, True)
        # Edge-band cell, so a paved seed in the corridor cannot vote the face drivable.
        vi, ui = splat(depth, sem, cam, 1.75 + 0.3, float(y), float(height), False)
        return label_at(cam, depth, sem, vi, ui, edges=False)

    def edge_face(cam, y, height):
        depth = np.zeros((cam.height, cam.width), np.uint16)
        vi, ui = splat(depth, None, cam, 0.40, float(y), float(height), False)
        return label_at(cam, depth, None, vi, ui, edges=True)

    for cam in (main, wide):
        for y in (12.0, 20.0, 27.4, 30.0):
            for height in (0.16, 0.20):
                kind = fallback_face(cam, y, height)
                assert kind != DRIVABLE, (cam.width, y, height, kind)
    saved = obstacles.depth_range_pad_m
    obstacles.depth_range_pad_m = lambda depth_m, pose=None: np.zeros(np.shape(depth_m), dtype=np.float64)
    try:
        opened = [edge_face(cam, y, height) for cam in (main, wide) for y in (12.0, 20.0, 27.4, 30.0) for height in (0.16, 0.20)]
    finally:
        obstacles.depth_range_pad_m = saved
    assert any(kind == DRIVABLE for kind in opened), opened
    print("  fit path: 0.16/0.20 m faces never drivable; pad=0 on edge-z is")


def check_edge_band_and_row_fields() -> None:
    """A 0.10 m curb in the edge band is never drivable. A 0.05 m lip is not blocked.

    The three epsilon fields are on a full row and on every empty row.
    """
    from python.train.obstacles import DRIVABLE, UNKNOWN

    right, forward, up = _flat_axes()
    pos = (0.0, 0.0, 0.0)
    half = 3.5 / 2.0
    x = half + 0.3
    # y=10 is where a recovered 0.10 m face clears curb_eps + the depth pad.
    y = 10.0
    assert x < half + 0.5

    def one(z: float):
        depth = _blank(POSE.height, POSE.width)
        ego = _to_ego((x, y, z), pos, right, forward, up)
        _splat(depth, ego)
        got = _label(depth, lane=_paint(3.5), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
        return got, ego

    curb, curb_e = one(0.10)
    lip, lip_e = one(0.05)
    assert _at(curb, curb_e[0], curb_e[1])[0] != DRIVABLE, curb_e
    assert _at(lip, lip_e[0], lip_e[1])[0] != DRIVABLE, lip_e
    inside = _blank(POSE.height, POSE.width)
    in_ego = _to_ego((0.0, 8.0, 0.0), pos, right, forward, up)
    _splat(inside, in_ego)
    inner = _label(inside, lane=_paint(3.5), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
    assert _at(inner, in_ego[0], in_ego[1])[0] == DRIVABLE, in_ego

    none_depth = _label(None, lane=_paint(), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
    zeros = _label(_blank(POSE.height, POSE.width), lane=_paint(), road=_road(lambda _y: 0.0), pos=pos, forward=forward, up=up)
    no_corridor = _label(
        inside,
        lane=(None, None),
        road=_road(lambda _y: 0.0),
        pos=pos,
        forward=forward,
        up=up,
    )
    for record in (curb, lip, inner, none_depth, zeros, no_corridor):
        _assert_row_fields(record)
    assert int(np.asarray(none_depth["mask"]).max()) == UNKNOWN
    print("  edge band: 0.10 m and 0.05 m at half+0.3 m are not drivable; eps fields on every row")


def check_head_stays_out_of_export() -> None:
    from python.train.cam_strip import HEAD_NAMES as CAM_HEADS
    from python.train.cam_strip import parse_heads
    from python.train.obstacles import CLEARANCE_LOSS_WEIGHT, PROP_CHECK_PASSED
    from python.train.scene_net import HEAD_NAMES, OUTPUT_FIELDS

    assert "obstacles" not in HEAD_NAMES
    assert "obstacles" not in CAM_HEADS
    assert "obstacle_mask" not in OUTPUT_FIELDS
    assert "obstacles" not in parse_heads("")
    assert parse_heads("lane_gt,obstacles") == ("lane_gt", "obstacles")
    assert PROP_CHECK_PASSED is False
    assert float(CLEARANCE_LOSS_WEIGHT) == 0.0

    import torch

    from python.train.scene_net import (
        LANE_POINTS,
        LAMP_STATES,
        N_CURBS,
        N_LANES,
        N_OBJECTS,
        N_SIGNS,
        OBJECT_CLASSES,
        OBJECT_FEAT,
        SIGN_CLASSES,
        SIGN_FEAT,
        SceneNet,
        ScenePrediction,
        scene_loss,
    )

    plain = SceneNet(cams=("main",))
    assert not getattr(plain, "obstacles", False)
    assert not any("obstacle" in name for name in plain.state_dict())
    opt = SceneNet(cams=("main",), obstacles=True)
    assert getattr(opt, "obstacles", False)
    extra = sum(p.numel() for p in opt.parameters()) - sum(p.numel() for p in plain.parameters())
    cin = int(opt.stem.out_channels)
    assert extra == cin * 3 + 3 + cin + 1, extra
    assert extra == 516, extra
    opt.eval()
    strip = torch.zeros(1, 3, opt.canvas_h, opt.canvas_w)
    pred, _state = opt.forward_batch(strip, torch.zeros(1), opt.initial_state())
    assert tuple(pred.obstacle_mask.shape) == (1, 3, 64, 64)
    assert tuple(pred.obstacle_clear.shape) == (1, 1)
    assert "obstacle_mask" not in pred.as_dict()

    try:
        import onnx
        from python.train.scene_net import export_scene_onnx
    except ImportError:
        from gvd_test_env import skip_check

        skip_check("obstacle_onnx", "onnx is not installed")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = export_scene_onnx(Path(tmp) / "obs.onnx", opt, heads=("lane_gt",))
            model = onnx.load(str(path))
            outputs = [item.name for item in model.graph.output]
        assert list(outputs[: len(OUTPUT_FIELDS)]) == list(OUTPUT_FIELDS)
        assert not any("obstacle" in name for name in outputs)

    raw = torch.zeros(1, 3, 64, 64)
    raw[:, 1] = 4.0
    logits = raw.detach().requires_grad_(True)
    clear = torch.zeros(1, 1, requires_grad=True)
    blank = ScenePrediction(
        lanes=torch.zeros(1, N_LANES, LANE_POINTS, 2),
        lane_valid=torch.zeros(1, N_LANES),
        curbs=torch.zeros(1, N_CURBS, LANE_POINTS, 2),
        curb_valid=torch.zeros(1, N_CURBS),
        objects=torch.zeros(1, N_OBJECTS, OBJECT_FEAT),
        object_class=torch.zeros(1, N_OBJECTS, len(OBJECT_CLASSES)),
        object_valid=torch.zeros(1, N_OBJECTS),
        signs=torch.zeros(1, N_SIGNS, SIGN_FEAT),
        sign_class=torch.zeros(1, N_SIGNS, len(SIGN_CLASSES)),
        sign_state=torch.zeros(1, N_SIGNS, len(LAMP_STATES)),
        sign_valid=torch.zeros(1, N_SIGNS),
        lane_gt=torch.zeros(1, 14),
        obstacle_mask=logits,
        obstacle_clear=clear,
    )
    mask = torch.zeros(64, 64, dtype=torch.long)
    mask[10, 10] = 2
    target = {
        "obstacle_mask": mask,
        "obstacle_clear": torch.tensor([12.0]),
        "obstacle_clear_valid": torch.tensor([1.0]),
        "obstacle_present": torch.tensor([1.0]),
    }
    loss = scene_loss(blank, target, heads=("obstacles",))
    float(loss["total"])
    loss["total"].backward()
    assert logits.grad is not None
    assert float(logits.grad[0, :, 0, 0].abs().sum()) == 0.0
    assert float(logits.grad[0, :, 10, 10].abs().sum()) > 0.0
    known_ce = torch.nn.functional.cross_entropy(logits[0, :, 10, 10], torch.tensor(2))
    assert abs(float(loss["obstacle"]) - float(known_ce)) < 1e-5
    import inspect
    from python.train.scene_net import _scene_loss_body
    assert "torch.where" in inspect.getsource(_scene_loss_body)
    assert clear.grad is None or float(clear.grad.abs().sum()) == 0.0
    assert float(loss["obstacle_clear"]) == 0.0

    absent = dict(target)
    absent["obstacle_present"] = torch.tensor([0.0])
    logits2 = torch.zeros(1, 3, 64, 64, requires_grad=True)
    blank.obstacle_mask = logits2
    missed = scene_loss(blank, absent, heads=("obstacles",))
    assert float(missed["obstacle"]) == 0.0
    assert bool(missed["total"].requires_grad)
    missed["total"].backward()
    assert logits2.grad is not None and float(logits2.grad.abs().sum()) == 0.0

    unknown = dict(target)
    unknown["obstacle_mask"] = torch.zeros(64, 64, dtype=torch.long)
    unknown["obstacle_present"] = torch.tensor([1.0])
    logits3 = torch.zeros(1, 3, 64, 64, requires_grad=True)
    blank.obstacle_mask = logits3
    unk = scene_loss(blank, unknown, heads=("obstacles",))
    assert float(unk["obstacle"]) == 0.0
    assert bool(unk["total"].requires_grad)
    unk["total"].backward()
    assert float(logits3.grad.abs().sum()) == 0.0

    opt.train()
    live = opt.forward_batch(torch.rand(1, 3, opt.canvas_h, opt.canvas_w), torch.zeros(1), opt.initial_state())[0]
    trained = scene_loss(live, target, heads=("obstacles",))
    trained["total"].backward()
    assert opt.obstacle_mask_head.weight.grad is not None
    assert float(opt.obstacle_mask_head.weight.grad.abs().sum()) > 0.0
    assert opt.obstacle_clear_head.weight.grad is None or float(opt.obstacle_clear_head.weight.grad.abs().sum()) == 0.0
    print("  training-only head is opt-in, clearance weight stays 0, unknown cells do not train")


def check_runtime_does_not_import_the_labeler() -> None:
    targets = [ROOT / "python" / "run_vision.py"]
    for folder in ("control", "planning", "perception", "viz"):
        targets.extend(sorted((ROOT / "python" / folder).glob("*.py")))
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
                assert "label_obstacles" not in low, f"{path.name} imports {name}"
                if "obstacle" in low:
                    raise AssertionError(f"{path.name} imports {name}")
    print(f"  runtime files do not import the obstacle labeler ({len(targets)} files)")


def _classes_road():
    return {"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "unknown", "1": "ROAD"}}


def _one_point(cam, x, y, z, *, edges, seeds=False):
    """Class of one point. ``seeds`` paints a paved station so the depth fit can run."""
    from python.train.obstacles import label_obstacle_frame, view_depth_m, _unproject

    depth = np.zeros((cam.height, cam.width), np.uint16)
    sem = np.zeros(depth.shape, np.uint8) if seeds or not edges else None

    def splat(px, py, pz, paved):
        hit = view_depth_m(cam, px, py, pz, cam.width, cam.height)
        assert hit is not None, (px, py, pz)
        u, v, metres = hit
        ui, vi = int(round(u)), int(round(v))
        assert 0 <= ui < cam.width and 0 <= vi < cam.height, (ui, vi, px, py, pz)
        depth[vi, ui] = np.uint16(max(1, int(round(float(metres) * 100.0))))
        if paved and sem is not None:
            sem[vi, ui] = 1
        return vi, ui

    if seeds:
        center = int(math.floor(float(y) / 0.625))
        for row in range(center - 4, center + 5):
            yy = (row + 0.5) * 0.625
            if yy < 2.0:
                continue
            for sx in (0.25, 0.75, 1.25, 1.60):
                splat(sx, yy, 0.0, True)
    vi, ui = splat(x, y, z, False)
    values, valid = _paint(3.5)
    got = label_obstacle_frame(
        depth,
        pose=cam,
        lane_gt=values,
        lane_gt_valid=valid,
        road=_road(lambda _y: 0.0, edges=edges),
        ego_pos=(0.0, 0.0, 0.0),
        ego_dir=(0.0, 1.0, 0.0),
        ego_up=(0.0, 0.0, 1.0),
        sem=sem,
        sem_classes=_classes_road() if sem is not None else None,
        cam="probe",
        t=1.0,
    )
    ex, ey, _ez, _ok = _unproject(cam, depth)
    kind = _exact(got, float(ex[vi, ui]), float(ey[vi, ui]))
    mask = np.asarray(got["mask"])
    counts = {int(k): int(v) for k, v in zip(*np.unique(mask, return_counts=True))}
    return kind, counts


def check_path_class_counts() -> None:
    """Edge-z uses pad(r). The depth fit adds fit_tol. Flat road inside 17 m is drivable."""
    from python.train.obstacles import BLOCKED, DRIVABLE, UNKNOWN

    main = _cam("main")
    half = 3.5 / 2.0
    names = {UNKNOWN: "unknown", DRIVABLE: "drivable", BLOCKED: "blocked"}
    rows = []

    def add(path, what, y, z, x, *, edges, seeds):
        kind, counts = _one_point(main, x, y, z, edges=edges, seeds=seeds)
        rows.append((path, what, y, names[kind], counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)))
        return kind

    for y in (8.0, 12.0, 16.0):
        assert add("edge-z", "flat", y, 0.0, 0.4, edges=True, seeds=False) == DRIVABLE, y
    assert add("edge-z", "curb 0.10", 8.0, 0.10, half + 0.3, edges=True, seeds=False) == BLOCKED
    assert add("edge-z", "curb 0.10", 12.0, 0.10, half + 0.3, edges=True, seeds=False) == BLOCKED
    assert add("edge-z", "curb 0.20", 8.0, 0.20, half + 0.3, edges=True, seeds=False) == BLOCKED
    assert add("edge-z", "curb 0.35", 8.0, 0.35, half + 0.3, edges=True, seeds=False) == BLOCKED
    assert add("edge-z", "curb 0.10", 30.0, 0.10, half + 0.3, edges=True, seeds=False) != BLOCKED
    assert add("fit", "flat", 8.0, 0.0, 0.4, edges=False, seeds=True) == DRIVABLE
    assert add("fit", "curb 0.10", 8.0, 0.10, half + 0.3, edges=False, seeds=True) == UNKNOWN
    assert add("fit", "curb 0.20", 8.0, 0.20, half + 0.3, edges=False, seeds=True) == UNKNOWN
    assert add("fit", "curb 0.35", 8.0, 0.35, half + 0.3, edges=False, seeds=True) == BLOCKED
    print("  path counts (main): path object y class unk drv blk")
    for path, what, y, kind, unk, drv, blk in rows:
        print(f"    {path:7} {what:10} y={y:4.0f} {kind:9} {unk:4} {drv:3} {blk:2}")


def check_fit_levers() -> None:
    """Each fallback lever fails on its own: seed 0.15, whole frame, min inliers 1, no refit."""
    from python.train.obstacles import (
        BLOCKED,
        FIT_MIN_INLIERS,
        FIT_SEED_M,
        UNKNOWN,
        WINDOW_ROWS,
        _refit_cell_slope,
        label_obstacle_frame,
        view_depth_m,
        _unproject,
    )

    assert FIT_SEED_M == 0.05 and FIT_MIN_INLIERS == 20 and WINDOW_ROWS == 5
    lateral = np.array([0.25] * 40 + [1.75] * 4, dtype=np.float64)
    height = np.array([0.05] * 40 + [0.0] * 4, dtype=np.float64)
    rows = np.array(list(range(40)) + list(range(4)))
    refit = _refit_cell_slope(lateral, height, rows)
    assert refit is not None and abs(refit - 0.20) < 1e-9, refit

    cam = _cam("main")
    classes = _classes_road()

    def scene(points):
        depth = np.zeros((cam.height, cam.width), np.uint16)
        sem = np.zeros(depth.shape, np.uint8)
        mark = None
        for x, y, z, paved, query in points:
            hit = view_depth_m(cam, x, y, z, cam.width, cam.height)
            assert hit is not None, (x, y, z)
            u, v, metres = hit
            ui, vi = int(round(u)), int(round(v))
            assert 0 <= ui < cam.width and 0 <= vi < cam.height, (ui, vi, x, y, z)
            depth[vi, ui] = np.uint16(max(1, int(round(float(metres) * 100.0))))
            if paved:
                sem[vi, ui] = 1
            if query:
                mark = (vi, ui)
        values, valid = _paint(3.5)
        got = label_obstacle_frame(
            depth, pose=cam, lane_gt=values, lane_gt_valid=valid,
            road=_road(lambda _y: 0.0, edges=False),
            ego_pos=(0.0, 0.0, 0.0), ego_dir=(0.0, 1.0, 0.0), ego_up=(0.0, 0.0, 1.0),
            sem=sem, sem_classes=classes, cam="probe", t=1.0,
        )
        ex, ey, _ez, _ok = _unproject(cam, depth)
        vi, ui = mark
        return _exact(got, float(ex[vi, ui]), float(ey[vi, ui]))

    def seeds_at(rows_y, xs, z):
        out = []
        for y in rows_y:
            for x in xs:
                out.append((x, y, z, True, False))
        return out

    y_local = (16 + 0.5) * 0.625
    local = seeds_at([y_local], (0.25, 0.75, 1.25, 1.60), 0.0)
    far = []
    for row in range(40, 49):
        far.extend(seeds_at([(row + 0.5) * 0.625], (0.25, 0.75, 1.25, 1.60), 0.0))
    tall = (1.25, y_local + 0.12, 0.55, False, True)
    # One row cannot reach 20 inliers, so the tall face stays unknown.
    assert scene(local + [tall]) == UNKNOWN
    # Far rows are outside the 5-row window. The whole frame would fit them and block the face.
    assert scene(local + far + [tall]) == UNKNOWN

    y = 8.0
    center = int(math.floor(y / 0.625))
    road = []
    poison = []
    for row in range(center - 4, center + 5):
        yy = (row + 0.5) * 0.625
        for x in (0.25, 0.75, 1.25, 1.60):
            road.append((x, yy, 0.0, True, False))
            for dx in (-0.10, 0.10):
                poison.append((x + dx, yy, 0.12, True, False))
    marker = (1.60, y + 0.15, 0.45, False, True)
    assert scene(poison + road + [marker]) == BLOCKED
    print("  fit levers: seed 0.05, 5-row window, 20 inliers, and the refit each hold")


def main() -> None:
    check_pitched_and_raised_ego_uses_world_z()
    check_epsilon_is_world_clearance()
    check_corridor_matches_paint_then_pack_road_edge()
    check_edge_heights_are_kept()
    check_depth_fit_keeps_curb_blocked()
    check_edge_band_and_row_fields()
    check_crown_and_bank_not_blocked()
    check_bump_survives_depth_quantization()
    check_interim_pad_band()
    check_path_class_counts()
    check_fit_levers()
    check_heading_matches_pack_road_edge()
    check_wall_pole_empty_and_missing_depth()
    check_missing_geometry_stays_unknown()
    check_clearance_masked_without_near_centers()
    check_semantic_blocked_needs_palette_proof()
    check_writer_and_missing_sidecar()
    check_head_stays_out_of_export()
    check_runtime_does_not_import_the_labeler()
    print("test_obstacle_labels: OK")


if __name__ == "__main__":
    main()
