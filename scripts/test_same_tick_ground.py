#!/usr/bin/env python3
"""Same-tick eight cameras, and side/rear pixels on the ground (no BeamNG)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.lanes import (  # noqa: E402
    _ego_point,
    camera_pose,
    lanes_from_view,
    missing_pose_ids,
    posed_lanes_from_image,
    project_pose_pixel,
)
from python.perception.stitch360 import STITCH_ORDER, stitch_frames  # noqa: E402
from python.planning.path_predictor import predict_path  # noqa: E402
from python.sensors.cameras import (  # noqa: E402
    CAM_IDS,
    BeamNGPyBackend,
    CamHealth,
    bundle_tick_frames,
    load_camera_config,
)
from python.viz.stage import drawn_lane_records  # noqa: E402


def _nearest(cid: str, x: float, y: float, step: int = 4) -> tuple[int, int]:
    pose = camera_pose(cid)
    assert pose is not None, cid
    best: tuple[int, int] | None = None
    best_d = 1e9
    for v in range(0, pose.height, step):
        for u in range(0, pose.width, step):
            ground = project_pose_pixel(pose, u, v)
            if ground is None:
                continue
            dist = (ground["x"] - x) ** 2 + (ground["y"] - y) ** 2
            if dist < best_d:
                best_d = dist
                best = (u, v)
    assert best is not None, (cid, x, y)
    return best


def _paint(cid: str, specs: list[tuple[float, list[float], tuple[int, int, int]]]) -> np.ndarray:
    pose = camera_pose(cid)
    assert pose is not None
    img = np.full((pose.height, pose.width, 3), (40, 40, 42), dtype=np.uint8)
    import cv2

    for world_x, ys, color in specs:
        pts = [_nearest(cid, world_x, y) for y in ys]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 10)
    return img


def _check_poses_and_pixels() -> None:
    missing = missing_pose_ids()
    assert missing == (), missing
    pillar = camera_pose("pillarL")
    rear = camera_pose("rear")
    right = camera_pose("pillarR")
    assert pillar is not None and rear is not None and right is not None
    pu, pv = pillar.width // 2, pillar.height // 2
    side = project_pose_pixel(pillar, pu, pv)
    assert side is not None
    assert side["x"] < -3.0, side
    forward = _ego_point(pu, pv, pillar.width, pillar.height, 35.0)
    assert abs(forward["x"]) <= 3.0 and forward["y"] >= 2.0
    assert abs(side["x"] - forward["x"]) > 1.0 or abs(side["y"] - forward["y"]) > 1.0
    ru, rv = rear.width // 2, rear.height // 2
    behind = project_pose_pixel(rear, ru, rv)
    assert behind is not None
    assert behind["y"] < 0.0, behind
    assert behind["y"] < rear.y, behind
    forced = _ego_point(ru, rv, rear.width, rear.height, 35.0)
    assert forced["y"] >= 2.0
    assert behind["y"] < 0.0
    other = project_pose_pixel(right, right.width // 2, right.height // 2)
    assert other is not None and other["x"] > 3.0, other


def _check_lanes_and_path() -> None:
    pillar = _paint("pillarL", [(-6.0, [0.0, 2.0, 4.0, 6.0, 8.0], (245, 245, 245))])
    split = _paint(
        "pillarL",
        [
            (-8.0, [0.0, 2.0, 4.0, 6.0], (245, 245, 245)),
            (-4.0, [0.0, 2.0, 4.0, 6.0], (245, 245, 245)),
        ],
    )
    pieces = posed_lanes_from_image(split, "pillarL")
    assert len(pieces) >= 2, pieces
    means = [sum(p["x"] for p in poly) / len(poly) for poly in pieces]
    assert max(means) - min(means) > 0.9, means

    rear = _paint(
        "rear",
        [
            (-1.6, [-12.0, -8.0, -5.0, -3.2], (245, 245, 245)),
            (1.6, [-12.0, -8.0, -5.0, -3.2], (0, 220, 220)),
        ],
    )
    frames: dict[str, np.ndarray] = {}
    for cid in STITCH_ORDER:
        pose = camera_pose(cid)
        assert pose is not None
        frames[cid] = np.full((pose.height, pose.width, 3), (30, 30, 32), dtype=np.uint8)
    frames["pillarL"] = pillar
    frames["rear"] = rear
    stitch = stitch_frames(frames)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    fit = lanes_from_view(blank, stitch.bgr)
    xs = [p["x"] for poly in fit.lanes_bev for p in poly]
    ys = [p["y"] for poly in fit.lanes_bev for p in poly]
    assert any(x < -3.0 for x in xs), xs
    assert any(y < -1.0 for y in ys), ys
    plan = predict_path(lanes_bev=fit.lanes_bev, lane_conf=fit.conf, ego_speed_mps=5.0)
    assert plan.drivable is True, plan.prediction
    assert plan.path_ego
    assert any(p["y"] < 0.0 for p in plan.path_ego), plan.path_ego[:3]

    beside_left = [{"x": -6.0, "y": float(y)} for y in (0, 4, 8, 12)]
    beside_right = [{"x": -2.6, "y": float(y)} for y in (0, 4, 8, 12)]
    beside = predict_path(lanes_bev=[beside_left, beside_right], lane_conf=0.8, ego_speed_mps=5.0)
    assert beside.drivable is True
    assert beside.path_ego[0]["x"] < -3.0, beside.path_ego[0]
    rear_left = [{"x": -1.6, "y": float(y)} for y in (-8, -6, -4, -2)]
    rear_right = [{"x": 1.6, "y": float(y)} for y in (-8, -6, -4, -2)]
    rear_plan = predict_path(lanes_bev=[rear_left, rear_right], lane_conf=0.8, ego_speed_mps=5.0)
    assert rear_plan.drivable is True
    assert rear_plan.prediction != "roundabout", rear_plan.prediction
    assert min(p["y"] for p in rear_plan.path_ego) < 0.0


def _check_drawn_past_path() -> None:
    pts = [{"x": -1.7, "y": float(y)} for y in (-6.0, -2.0, 4.0, 18.0, 48.0)]
    path = [{"x": 0.0, "y": float(y)} for y in range(0, 31, 2)]
    state = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "path_ego": path,
        "path_width": 3.5,
        "path_debug_preview": False,
        "viz_smoke": False,
        "lanes_ext": [{"points": list(pts), "kind": "detected", "index": -1, "side": "left"}],
        "road_edges": [],
        "tracks": [],
        "signs": [],
        "missing_state_keys": ["live cameras"],
    }
    drawn = drawn_lane_records(state)
    got = [float(p["y"]) for p in drawn[0]["points"]]
    assert max(got) == 48.0, got
    assert min(got) == -6.0, got


def _check_same_tick_stitch() -> None:
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    grab_at = src.find("bundle = backend.grab()")
    stitch_at = src.find("stitch_frames(bundle_tick_frames(bundle))")
    assert 0 <= grab_at < stitch_at

    reads: list[str] = []
    blank = {"on": False}
    colour = {"n": 0}

    class Cam:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.is_streaming = True
            self.update_priority = 0.0
            self.resolution = kwargs.get("resolution", (8, 8))

        def get_update_priority(self):
            return self.update_priority

        def set_update_priority(self, p):
            self.update_priority = float(p)

        def set_max_pending_requests(self, n):
            return None

        def stream_raw(self):
            reads.append(self.name)
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            if not blank["on"]:
                img[0, 0, 1] = 40 + (colour["n"] % 200)
                img[1, 1, 1] = 20
            return {"colour": img}

        def poll(self):
            raise AssertionError("shared-memory cameras stream_raw")

        def send_ad_hoc_poll_request(self):
            raise AssertionError("ad-hoc is not this read")

        def is_ad_hoc_poll_request_ready(self, request_id):
            return False

        def collect_ad_hoc_poll_request(self, request_id):
            return None

        def remove(self):
            return None

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = Cam
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    try:
        import python.control.actuate as act

        prev = act.soft_esc_sensors_every_tick()
        engage = act.engage_path()
        prev_bytes = engage.read_bytes() if engage.is_file() else None
        act.note_soft_esc_engaged(False)
        act.write_engage_flag(False)
        be = BeamNGPyBackend(
            config=load_camera_config(),
            tech_config={
                "wait_vehicle_s": 0,
                "cameras": {
                    "attach": True,
                    "rgb_only": True,
                    "update_s": 0.067,
                    "shared_memory": True,
                    "streaming": True,
                },
            },
        )

        def _connect(explicit=True):
            be.session.vehicle = object()
            be.session.bng = object()
            return True

        be.session.connect = _connect  # type: ignore[method-assign]
        be.session.attach_vehicle_sensors = lambda: {}  # type: ignore[method-assign]
        be.open()
        reads.clear()
        colour["n"] = 3
        first = be.grab()
        assert reads[:8] == [f"gvd_{cid}" for cid in CAM_IDS], reads
        assert len(reads) == 8
        for cid in CAM_IDS:
            assert first.health[cid] == CamHealth.OK, cid
        stamps = [float(first.timestamps[cid]) for cid in CAM_IDS]
        assert max(stamps) - min(stamps) < 1e-6
        tick = bundle_tick_frames(first)
        assert set(CAM_IDS).issubset(tick)
        st = stitch_frames(tick)
        assert all(sec.present for sec in st.sectors), [sec.cam_id for sec in st.sectors if not sec.present]

        colour["n"] = 9
        reads.clear()
        second = be.grab()
        assert reads == [f"gvd_{cid}" for cid in CAM_IDS]
        st2 = stitch_frames(bundle_tick_frames(second))
        assert not np.array_equal(st.bgr, st2.bgr)

        sentinel = np.full((8, 8, 3), 17, dtype=np.uint8)
        sentinel[0, 0, 2] = 200
        for cid in CAM_IDS:
            be._cache_frames[cid] = sentinel.copy()
            be._cache_ts[cid] = 1.0
        blank["on"] = True
        poisoned = be.grab()
        blank["on"] = False
        for cid in CAM_IDS:
            assert cid not in poisoned.frames, cid
            assert poisoned.health[cid] == CamHealth.MISSING
        empty = stitch_frames(bundle_tick_frames(poisoned))
        assert not any(sec.present for sec in empty.sectors)
        assert int(empty.bgr.max()) != 200

        odd = first
        odd.timestamps["rear"] = float(odd.timestamps["main"]) + 1.0
        kept = bundle_tick_frames(odd)
        assert "rear" not in kept
        assert "main" in kept
        dropped = stitch_frames(kept)
        rear_sec = next(sec for sec in dropped.sectors if sec.cam_id == "rear")
        assert rear_sec.present is False
        act.note_soft_esc_engaged(prev)
        if prev_bytes is None:
            engage.unlink(missing_ok=True)
        else:
            engage.write_bytes(prev_bytes)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def main() -> None:
    _check_poses_and_pixels()
    _check_lanes_and_path()
    _check_drawn_past_path()
    _check_same_tick_stitch()
    print("test_same_tick_ground: OK")


if __name__ == "__main__":
    main()
