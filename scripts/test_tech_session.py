#!/usr/bin/env python3
"""Offline Tech session / vehicle-data / camera-frame conversion (no live BeamNG)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.sensors.cameras import (  # noqa: E402
    CAM_IDS,
    CAMERA_HZ_TARGET,
    DEFAULT_FAR_M,
    DEFAULT_NEAR_M,
    DEFAULT_UPDATE_PRIORITY,
    DEFAULT_UPDATE_S,
    FORWARD_CAM_IDS,
    LIVE_NARROW_HITCH_AFTER_S,
    NARROW_FAR_LIVE_HITCH_M,
    NARROW_GRAB_DIV,
    NARROW_GRAB_PHASE,
    ON_DEMAND_UPDATE_S,
    PILLAR_SPREAD_STEP,
    REAR_CAM_IDS,
    REAR_GRAB_DIV,
    REAR_GRAB_PHASE,
    REPEAT_GRAB_DIV,
    REPEAT_GRAB_PHASE,
    SIDE_CAM_IDS,
    SIDE_GRAB_DIV,
    SIDE_GRAB_PHASE,
    WIDE_GRAB_DIV,
    beamng_camera_sensor_kwargs,
    camera_clip_planes,
    camera_grab_div,
    camera_grab_due,
    camera_grab_phase,
    camera_update_priority,
    camera_update_s,
    colour_to_bgr,
    far_hitch_ladder,
    grab_due,
    grab_is_poll_free,
    grab_phase_of,
    grab_wheel,
    invert_update_priority,
    iter_clip_attach_attempts,
    live_narrow_far_m,
    load_camera_config,
    priority_highest_is_zero,
    read_camera_colour,
    read_camera_update_priority,
    resolve_backend_name,
    yaw_pitch_to_dir_up,
)
from python.sensors.tech import (  # noqa: E402
    FORBIDDEN_SENSORS,
    TechSession,
    VehicleData,
    apply_env_overrides,
    apply_nav_missing,
    bearing_deg,
    gvd_to_bng_vehicle,
    haversine_m,
    heading_from_world_dir,
    latest_gps_reading,
    load_tech_config,
    nav_heading_from_sources,
    nav_snapshot,
    path_ego_to_world,
    wrap180,
)


def check_frame_convert() -> None:
    # GVD windshield-forward (0, 1.20, 1.26) looking +Y → BeamNG (0, -1.20, 1.26) looking -Y.
    pos = gvd_to_bng_vehicle(0.0, 1.20, 1.26)
    assert pos == (0.0, -1.20, 1.26), pos
    d0, u0 = yaw_pitch_to_dir_up(0.0, 0.0)
    bd = gvd_to_bng_vehicle(*d0)
    # BeamNGpy Camera default forward is (0, -1, 0).
    assert abs(bd[0] - 0.0) < 1e-9 and abs(bd[1] - (-1.0)) < 1e-9, (d0, bd)
    rear_d, _ = yaw_pitch_to_dir_up(180.0, 0.0)
    br = gvd_to_bng_vehicle(*rear_d)
    assert br[1] > 0.5, br  # rear look is +Y in BeamNG vehicle space
    # involution
    assert gvd_to_bng_vehicle(*gvd_to_bng_vehicle(0.5, -1.0, 2.0)) == (0.5, -1.0, 2.0)


def check_path_world() -> None:
    path = [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 10.0, "z": 0.0}]
    # Vehicle at origin facing +Y world.
    w = path_ego_to_world(path, (1.0, 2.0, 3.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert w is not None and len(w) == 2
    assert abs(w[1]["x"] - 1.0) < 1e-6 and abs(w[1]["y"] - 12.0) < 1e-6
    assert path_ego_to_world(path, None, (0, 1, 0)) is None


def check_tech_yaml() -> None:
    cfg = load_tech_config()
    assert cfg.get("host") == "localhost"
    assert int(cfg.get("port") or 0) == 25252
    assert cfg.get("launch") is False
    assert float(cfg.get("socket_timeout")) == 10.0
    assert str(cfg.get("beamngpy_pin")) == "1.36"
    sensors = cfg.get("sensors") or {}
    assert sensors.get("electrics") is True
    assert sensors.get("damage") is True
    assert sensors.get("gps") is True
    for bad in FORBIDDEN_SENSORS:
        assert not sensors.get(bad), bad
    assert "gps" not in FORBIDDEN_SENSORS
    assert "lidar" not in FORBIDDEN_SENSORS
    cams = cfg.get("cameras") or {}
    assert cams.get("rgb_only") is True
    assert cams.get("convert_gvd_frame") is True
    raw_yaml = (ROOT / "config" / "tech.yaml").read_text(encoding="utf-8")
    assert "GVD_TECH_SOCKET_TIMEOUT" in raw_yaml
    text = raw_yaml.lower()
    assert "lidar" in text
    assert "vision" in text or "optional" in text
    assert "nav hint" in text or "map pin" in text
    nav = cfg.get("nav") or {}
    assert "pin_lat" in nav and "pin_lon" in nav
    gps = cfg.get("gps") or {}
    assert gps.get("ref_lon") is not None and gps.get("ref_lat") is not None
    assert list(gps.get("pos_m") or [])[:3] == [0.0, 0.0, 0.0] or list(gps.get("pos_m") or [])[:3] == [0.0, 0.0, 1.7]
    rig = load_camera_config()
    ids = [c.get("id") for c in (rig.get("cameras") or [])]
    assert ids == list(CAM_IDS), ids
    assert abs(float(rig.get("near_m") or 0) - DEFAULT_NEAR_M) < 1e-9
    by_id = {c["id"]: c for c in (rig.get("cameras") or [])}
    assert float(by_id["narrow"]["far_m"]) == 800
    assert float(by_id["main"]["far_m"]) == 300
    assert float(by_id["narrow"]["far_m"]) > float(by_id["main"]["far_m"])
    assert float(by_id["main"]["far_m"]) != 800
    assert float(by_id["wide"]["far_m"]) == 300
    hitch = rig.get("hitch") or {}
    assert int(hitch.get("side_grab_div")) == SIDE_GRAB_DIV
    assert int(hitch.get("repeat_grab_div")) == REPEAT_GRAB_DIV
    assert int(hitch.get("repeat_grab_phase")) == REPEAT_GRAB_PHASE
    assert int(hitch.get("rear_grab_div")) == REAR_GRAB_DIV
    assert int(hitch.get("main_grab_div")) == 1
    assert int(hitch.get("wide_grab_div")) == WIDE_GRAB_DIV
    assert int(hitch.get("narrow_grab_div")) == NARROW_GRAB_DIV
    assert int(hitch.get("narrow_grab_phase")) == NARROW_GRAB_PHASE
    assert int(hitch.get("wide_grab_phase")) == 0
    assert int(hitch.get("side_grab_phase")) == SIDE_GRAB_PHASE
    assert int(hitch.get("rear_grab_phase")) == REAR_GRAB_PHASE
    for cid in ("pillarL", "pillarR", "repeatL", "repeatR"):
        assert float(by_id[cid]["far_m"]) == 100, cid
        assert float(by_id[cid]["requested_update_time"]) < 0, cid
    assert camera_grab_div("pillarL", hitch) == SIDE_GRAB_DIV == 16
    assert camera_grab_div("pillarR", hitch) == 16
    assert camera_grab_div("repeatL", hitch) == REPEAT_GRAB_DIV == 16
    assert camera_grab_div("repeatR", hitch) == 16
    assert float(by_id["rear"]["far_m"]) == 100
    assert float(by_id["rear"]["requested_update_time"]) < 0
    assert camera_grab_div("rear", hitch) == REAR_GRAB_DIV == 16
    assert abs(float(by_id["main"]["requested_update_time"]) - 0.067) < 1e-9
    # Wide and narrow stay on-demand. The hitch poll is Python-side, not an auto GPU update.
    assert float(by_id["wide"]["requested_update_time"]) < 0
    assert float(by_id["narrow"]["requested_update_time"]) < 0
    assert camera_grab_div("main", hitch) == 1
    assert camera_grab_div("wide", hitch) == WIDE_GRAB_DIV == 16
    assert camera_grab_div("narrow", hitch) == NARROW_GRAB_DIV == 16
    assert camera_grab_phase("wide", hitch) == 0
    assert camera_grab_phase("narrow", hitch) == NARROW_GRAB_PHASE == 1
    assert camera_grab_phase("pillarL", hitch) == SIDE_GRAB_PHASE
    assert camera_grab_phase("pillarR", hitch) == (SIDE_GRAB_PHASE + PILLAR_SPREAD_STEP) % SIDE_GRAB_DIV
    for div_key in ("wide_grab_div", "narrow_grab_div", "side_grab_div", "repeat_grab_div", "rear_grab_div"):
        assert int(hitch.get(div_key)) == 16, div_key
    assert (int(hitch.get("wide_grab_phase")), int(hitch.get("narrow_grab_phase")), int(hitch.get("side_grab_phase")), int(hitch.get("repeat_grab_phase")), int(hitch.get("rear_grab_phase"))) == (0, 1, 2, 5, 6)
    assert camera_grab_phase("rear", hitch) == REAR_GRAB_PHASE == 6
    assert abs(float(by_id["main"].get("update_priority", 0)) - 0.0) < 1e-9
    assert float(by_id["narrow"].get("update_priority", 0)) > float(by_id["main"].get("update_priority", 0))
    assert float(by_id["pillarL"].get("update_priority", 0)) >= float(by_id["narrow"].get("update_priority", 0))
    for spec in rig.get("cameras") or []:
        res = list(spec.get("live_res") or [])
        assert res and max(int(res[0]), int(res[1])) == 640, spec.get("id")


def check_env_overrides(monkey: dict[str, str]) -> None:
    import os

    old = {k: os.environ.get(k) for k in ("GVD_BEAMNG_HOST", "GVD_BEAMNG_PORT", "BNG_HOME")}
    try:
        os.environ["GVD_BEAMNG_HOST"] = "127.0.0.8"
        os.environ["GVD_BEAMNG_PORT"] = "3333"
        os.environ["BNG_HOME"] = "C:/Tech"
        out = apply_env_overrides({"host": "localhost", "port": 25252, "home": ""})
        assert out["host"] == "127.0.0.8" and out["port"] == 3333 and out["home"] == "C:/Tech"
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_forbidden_attach() -> None:
    session = TechSession({"sensors": {"lidar": True, "electrics": False, "damage": False, "gforces": False}, "wait_vehicle_s": 0})
    session.vehicle = object()
    session.bng = object()
    try:
        attached = session.attach_vehicle_sensors()
    except ValueError as e:
        raise AssertionError(f"lidar attach must be allowed, got {e}") from e
    assert attached.get("lidar") in (True, False)

    banned = TechSession({"sensors": {"ultrasonic": True}, "wait_vehicle_s": 0})
    banned.vehicle = object()
    try:
        banned.attach_vehicle_sensors()
        raise AssertionError("ultrasonic must be refused")
    except ValueError as e:
        assert "ultrasonic" in str(e).lower()


def check_gps_not_forbidden() -> None:
    session = TechSession({"sensors": {"gps": True, "electrics": False, "damage": False, "gforces": False}, "wait_vehicle_s": 0})
    session.vehicle = object()
    session.bng = object()
    try:
        attached = session.attach_vehicle_sensors()
    except ValueError as e:
        raise AssertionError(f"GPS must be allowed, got {e}") from e
    # Live GPS needs BeamNGpy; offline we only require no refuse.
    assert attached.get("gps") in (True, False)


def check_gps_geometry() -> None:
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert abs(d - 111194.9) < 250, d
    assert abs(bearing_deg(0.0, 0.0, 1.0, 0.0) - 0.0) < 1e-6
    assert abs(bearing_deg(0.0, 0.0, 0.0, 1.0) - 90.0) < 1e-6
    assert abs(wrap180(190.0) - (-170.0)) < 1e-9
    assert heading_from_world_dir((0.0, 1.0, 0.0)) == 0.0  # +Y north
    assert abs(heading_from_world_dir((1.0, 0.0, 0.0)) - 90.0) < 1e-6  # +X east
    bulk = [{"time": 1.0, "lat": 53.0, "lon": 8.8}, {"time": 2.0, "lat": 53.1, "lon": 8.81}]
    got = latest_gps_reading(bulk)
    assert got is not None and got["lat"] == 53.1
    assert latest_gps_reading([{"lat": 1.0, "lon": 2.0}])["lon"] == 2.0
    assert latest_gps_reading({"lat": 53.09, "lon": 8.81, "x": 1, "y": 2})["lat"] == 53.09


def check_ego_fb_null_safe() -> None:
    """Tech attach leaves ego_fb=None; heading fill must not crash or require engage."""
    assert nav_heading_from_sources(vdata=None, ego_fb=None) is None
    vd = VehicleData(dir=(0.0, 1.0, 0.0), gps_heading_deg=12.5)
    assert nav_heading_from_sources(vdata=vd, ego_fb=None) == 12.5
    vd2 = VehicleData(dir=(0.0, 1.0, 0.0))
    assert nav_heading_from_sources(vdata=vd2, ego_fb=None) == 0.0

    class _Fb:
        dir = (1.0, 0.0, 0.0)

    assert abs(nav_heading_from_sources(vdata=None, ego_fb=_Fb()) - 90.0) < 1e-6
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "ego_fb.dir" not in rv
    assert "nav_heading_from_sources" in rv
    cams = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    assert "independent of Alt+G" in cams


def check_gps_poll_and_pin() -> None:
    class Box(dict):
        def __contains__(self, k):
            return dict.__contains__(self, k)

        def __iter__(self):
            return dict.__iter__(self)

    class FakeSensors(Box):
        def poll(self):
            return self

    class FakeGPS:
        def __init__(self):
            self.removed = False

        def poll(self):
            return [{"time": 1.0, "lon": 8.81, "lat": 53.09, "x": 10.0, "y": 20.0}]

        def remove(self):
            self.removed = True

    class FakeVeh:
        vid = "etk_player"
        options = {"model": "etk800"}
        state = {"pos": (10.0, 20.0, 1.0), "dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0), "vel": (0, 8, 0)}
        sensors = FakeSensors(
            electrics={"wheelspeed": 8.0, "steering_input": 0.2, "throttle_input": 0.1, "brake_input": 0.0, "gear": 3, "rpm": 2200},
            damage={"damage": 0.05},
            gforces={"gx": 0.1, "gy": 0.2, "gz": 0.0},
        )

    session = TechSession(
        {
            "wait_vehicle_s": 0,
            "sensors": {"electrics": True, "damage": True, "gforces": True, "gps": True},
            "nav": {"pin_lat": 53.10, "pin_lon": 8.82, "pin_name": "west gate"},
        }
    )
    session.vehicle = FakeVeh()
    session.attached = {"electrics": True, "damage": True, "gforces": True, "gps": True}
    gps = FakeGPS()
    session._gps = gps
    data = session.poll()
    assert data.lat == 53.09 and data.lon == 8.81
    assert data.gps_x == 10.0 and data.pin_name == "west gate"
    assert data.range_m is not None and data.range_m > 0
    assert data.bearing_deg is not None
    # Facing +Y (north); pin is east-ish of current lon → relative bearing to the right.
    assert data.bearing_rel_deg is not None and data.bearing_rel_deg > 0
    snap = nav_snapshot(data)
    assert snap["mode"] == "hint" and snap["drive_to_pin"] is False
    assert snap["pin"]["name"] == "west gate"
    miss = apply_nav_missing(["tracks", "nav"], snap)
    assert "nav drive-to-pin" in miss and "nav" not in miss
    assert nav_snapshot(None)["mode"] == "missing"
    session.close()
    assert gps.removed is True


def check_pin_env_override() -> None:
    import os

    keys = ("GVD_NAV_PIN_LAT", "GVD_NAV_PIN_LON", "GVD_NAV_PIN_NAME", "GVD_GPS")
    old = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["GVD_NAV_PIN_LAT"] = "53.2"
        os.environ["GVD_NAV_PIN_LON"] = "8.9"
        os.environ["GVD_NAV_PIN_NAME"] = "dock"
        os.environ["GVD_GPS"] = "1"
        out = apply_env_overrides({"sensors": {"gps": False}, "nav": {}})
        assert out["sensors"]["gps"] is True
        assert out["nav"]["pin_lat"] == 53.2 and out["nav"]["pin_lon"] == 8.9
        assert out["nav"]["pin_name"] == "dock"
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_poll_mock() -> None:
    class Box(dict):
        def __contains__(self, k):
            return dict.__contains__(self, k)

        def __iter__(self):
            return dict.__iter__(self)

    class FakeSensors(Box):
        def poll(self):
            return self

    class FakeVeh:
        vid = "etk_player"
        options = {"model": "etk800"}
        state = {"pos": (10.0, 20.0, 1.0), "dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0), "vel": (0, 8, 0)}
        sensors = FakeSensors(
            electrics={"wheelspeed": 8.0, "steering_input": 0.2, "throttle_input": 0.1, "brake_input": 0.0, "gear": 3, "rpm": 2200},
            damage={"damage": 0.05},
            gforces={"gx": 0.1, "gy": 0.2, "gz": 0.0},
        )

    session = TechSession({"wait_vehicle_s": 0, "sensors": {"electrics": True, "damage": True, "gforces": True}})
    session.vehicle = FakeVeh()
    session.attached = {"electrics": True, "damage": True, "gforces": True}
    data = session.poll()
    assert isinstance(data, VehicleData)
    assert data.connected and data.vid == "etk_player" and data.model == "etk800"
    assert data.speed_mps == 8.0 and data.steering_input == 0.2
    assert data.damage == 0.05 and data.pose_ok
    assert data.accel is not None and abs(data.accel - math.hypot(0.1, 0.2)) < 1e-6
    ox, oy, oz = session.origin_offset_gvd()
    assert (ox, oy, oz) == (0.0, 0.0, 0.0)
    pos, direction, up = session.camera_mount([0.0, 1.2, 1.26], (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert pos == (0.0, -1.2, 1.26) and direction == (0.0, -1.0, 0.0)


def check_camera_clip_planes() -> None:
    """Per-id far_m + requested_update_time wired as near_far_planes and update time."""
    cfg = load_camera_config()
    defaults = {"near_m": cfg.get("near_m", DEFAULT_NEAR_M)}
    want_far = {
        "narrow": 800.0,
        "main": 300.0,
        "wide": 300.0,
        "pillarL": 100.0,
        "pillarR": 100.0,
        "repeatL": 100.0,
        "repeatR": 100.0,
        "rear": 100.0,
    }
    want_rate = {
        "narrow": 0.067,
        "main": 0.067,
        "wide": 0.067,
        "pillarL": ON_DEMAND_UPDATE_S,
        "pillarR": ON_DEMAND_UPDATE_S,
        "repeatL": ON_DEMAND_UPDATE_S,
        "repeatR": ON_DEMAND_UPDATE_S,
        "rear": ON_DEMAND_UPDATE_S,
    }
    assert DEFAULT_FAR_M == want_far
    assert DEFAULT_UPDATE_S == want_rate
    assert want_far["narrow"] > want_far["main"]
    assert want_far["main"] != 800.0
    assert want_far["narrow"] != want_far["main"]
    assert want_far["narrow"] >= NARROW_FAR_LIVE_HITCH_M
    assert NARROW_FAR_LIVE_HITCH_M >= want_far["main"]
    for spec in cfg.get("cameras") or []:
        cid = spec["id"]
        near, far = camera_clip_planes(spec, defaults=defaults)
        assert abs(near - 0.05) < 1e-9, cid
        assert far == want_far[cid], (cid, far)
        # Yaml on-demand wins for wide/narrow. A missing key still uses the 0.067 role default.
        if cid in ("wide", "narrow"):
            assert camera_update_s(spec, cid=cid) == ON_DEMAND_UPDATE_S, cid
        else:
            assert abs(camera_update_s(spec, cid=cid) - want_rate[cid]) < 1e-9, cid
        other = "rear" if cid == "narrow" else "narrow"
        assert camera_clip_planes({"id": other}, defaults=defaults)[1] == want_far[other]

    # missing yaml keys still use locked per-id defaults (forward never silent 100 m, never both-800)
    assert camera_clip_planes({"id": "narrow"}) == (0.05, 800.0)
    assert camera_clip_planes({"id": "main"}) == (0.05, 300.0)
    assert camera_clip_planes({"id": "wide"}) == (0.05, 300.0)
    assert camera_clip_planes({"id": "pillarL"}) == (0.05, 100.0)
    assert camera_clip_planes({"id": "rear"}) == (0.05, 100.0)
    assert abs(camera_update_s({"id": "narrow"}) - 0.067) < 1e-9
    assert abs(camera_update_s({"id": "main"}) - 0.067) < 1e-9
    assert abs(camera_update_s({"id": "wide"}) - 0.067) < 1e-9
    assert camera_update_s({"id": "pillarL"}) == ON_DEMAND_UPDATE_S
    assert camera_update_s({"id": "rear"}) == ON_DEMAND_UPDATE_S
    assert camera_update_s({"id": "rear", "requested_update_time": -1}) == ON_DEMAND_UPDATE_S
    # forward 100 m (BeamNGpy default) is clamped to the role far — not kept, not a 400 hitch
    assert camera_clip_planes({"id": "narrow", "far_m": 100})[1] == 800.0
    assert camera_clip_planes({"id": "main", "far_m": 100})[1] == 300.0
    assert camera_clip_planes({"id": "wide", "far_m": 100})[1] == 300.0
    # sides/rear lock at 100; 150 clamps down (rung-2 hitch)
    assert camera_clip_planes({"id": "pillarL", "far_m": 100})[1] == 100.0
    assert camera_clip_planes({"id": "rear", "far_m": 100})[1] == 100.0
    assert camera_clip_planes({"id": "pillarL", "far_m": 150})[1] == 100.0
    assert camera_clip_planes({"id": "rear", "far_m": 150})[1] == 100.0
    # both-800 rejected: main 800 clamps to 300; narrow stays 800
    assert camera_clip_planes({"id": "main", "far_m": 800})[1] == 300.0
    assert camera_clip_planes({"id": "narrow", "far_m": 800})[1] == 800.0
    # live hitch floor: explicit 400 is legal and still ≥ main
    assert camera_clip_planes({"id": "narrow", "far_m": 400})[1] == 400.0
    assert camera_clip_planes({"id": "narrow", "far_m": 400})[1] >= camera_clip_planes({"id": "main"})[1]
    # near_far_planes pair still clamped to the role band
    assert camera_clip_planes({"id": "main", "far_m": 300, "near_far_planes": [0.05, 600]}) == (0.05, 300.0)
    assert camera_clip_planes({"id": "narrow", "far_m": 2000})[1] == 800.0

    assert far_hitch_ladder("narrow", 800.0) == (800.0,)
    assert far_hitch_ladder("narrow", 400.0) == (400.0,)
    assert far_hitch_ladder("main", 300.0) == (300.0,)
    assert far_hitch_ladder("wide", 300.0) == (300.0,)
    assert far_hitch_ladder("pillarL", 100.0) == (100.0,)
    assert far_hitch_ladder("rear", 100.0) == (100.0,)
    assert far_hitch_ladder("rear", 150.0) == (100.0,)
    assert far_hitch_ladder("main", 800.0) == (300.0,)
    assert far_hitch_ladder("narrow", 100.0) == (800.0,)

    assert camera_grab_div("main") == 1
    assert camera_grab_div("main", {"main_grab_div": 4}) == 1  # clamped every-tick
    assert camera_grab_div("main", {"main_grab_div": 2}) == 1
    assert camera_grab_div("wide") == WIDE_GRAB_DIV == 16
    assert camera_grab_div("narrow") == NARROW_GRAB_DIV == 16
    assert camera_grab_div("narrow", {"narrow_grab_div": 16}) == 16
    assert camera_grab_div("narrow", {"narrow_grab_div": 8}) == 8
    assert camera_grab_div("narrow", {"narrow_grab_div": 4}) == 4
    assert camera_grab_div("narrow", {"narrow_grab_div": 3}) == 3
    assert camera_grab_div("narrow", {"narrow_grab_div": 1}) == 2  # floor ÷2
    assert camera_grab_div("pillarL") == SIDE_GRAB_DIV == 16
    assert camera_grab_div("repeatL") == REPEAT_GRAB_DIV == 16
    assert camera_grab_div("rear") == REAR_GRAB_DIV == 16
    assert camera_grab_div("rear", {"rear_grab_div": 16, "side_grab_div": 16}) == 16
    assert grab_due(0, 2, 0) and not grab_due(1, 2, 0)
    assert grab_due(1, 2, 1) and not grab_due(0, 2, 1)
    hitch = load_camera_config().get("hitch") or {}
    wheel = {
        0: {"main", "wide"},
        1: {"main", "narrow"},
        2: {"main", "pillarL"},
        3: {"main", "pillarR"},
        4: {"main"},
        5: {"main", "repeatL"},
        6: {"main", "rear"},
        7: {"main", "repeatR"},
    }
    for slot in range(8, 16):
        wheel[slot] = {"main"}
    forbidden = {"narrow", "main", "pillarL", "pillarR"}
    for i in range(32):
        assert not (
            camera_grab_due("wide", i, hitch) and camera_grab_due("narrow", i, hitch)
        ), i
        assert camera_grab_due("main", i, hitch)
        wide_due = camera_grab_due("wide", i, hitch)
        narrow_due = camera_grab_due("narrow", i, hitch)
        if wide_due:
            assert i % 2 == 0, i
        if narrow_due:
            assert i % 2 == 1, i
        side_rear = [
            cid
            for cid in (*SIDE_CAM_IDS, *REAR_CAM_IDS)
            if camera_grab_due(cid, i, hitch)
        ]
        reads = [cid for cid in CAM_IDS if camera_grab_due(cid, i, hitch)]
        forwards = [cid for cid in ("main", "wide", "narrow") if cid in reads]
        # main is the only stream_raw; wide/narrow are polls and never both due.
        assert forwards == ["main"] or set(forwards) <= {"main", "wide", "narrow"}
        assert "main" in forwards
        assert not ("wide" in forwards and "narrow" in forwards)
        assert len([cid for cid in forwards if cid != "main"]) <= 1
        assert len(side_rear) <= 1, (i, side_rear)
        assert len(reads) <= 2, (i, reads)
        assert not forbidden.issubset(set(reads)), (i, reads)
        assert set(reads) == wheel[i % 16], (i, reads)
        if camera_grab_due("rear", i, hitch):
            sides_on_rear = [cid for cid in side_rear if cid in SIDE_CAM_IDS]
            assert sides_on_rear == [], (i, sides_on_rear)
        if camera_grab_due("narrow", i, hitch):
            assert side_rear == []
            assert "rear" not in side_rear
            assert not camera_grab_due("wide", i, hitch)
    assert [i for i in range(16) if camera_grab_due("wide", i, hitch)] == [0]
    assert [i for i in range(16) if camera_grab_due("narrow", i, hitch)] == [1]
    assert [i for i in range(16) if camera_grab_due("pillarL", i, hitch)] == [2]
    assert [i for i in range(16) if camera_grab_due("pillarR", i, hitch)] == [3]
    assert [i for i in range(16) if camera_grab_due("repeatL", i, hitch)] == [5]
    assert [i for i in range(16) if camera_grab_due("repeatR", i, hitch)] == [7]
    rear_hits = [i for i in range(16) if camera_grab_due("rear", i, hitch)]
    assert rear_hits == [6], rear_hits
    for i in rear_hits:
        sides = [cid for cid in SIDE_CAM_IDS if camera_grab_due(cid, i, hitch)]
        assert sides == [], (i, sides)
    assert camera_grab_phase("pillarL", hitch) == SIDE_GRAB_PHASE
    assert camera_grab_phase("pillarR", hitch) == 3
    assert camera_grab_phase("repeatL", hitch) == REPEAT_GRAB_PHASE
    assert camera_grab_phase("repeatR", hitch) == (REPEAT_GRAB_PHASE + 2) % REPEAT_GRAB_DIV
    assert camera_grab_due("main", 0, hitch) and camera_grab_due("wide", 0, hitch)
    assert not camera_grab_due("narrow", 0, hitch)
    hitch3 = dict(hitch)
    hitch3["narrow_grab_div"] = 3
    n3 = 0
    for i in range(24):
        w = camera_grab_due("wide", i, hitch3)
        n = camera_grab_due("narrow", i, hitch3)
        assert not (w and n), i
        if n:
            n3 += 1
            assert not w
    # phase 1 ÷3 collides with wide ÷16 on tick 16; the wide guard drops that one.
    assert n3 == 7
    assert CAMERA_HZ_TARGET == 10.0
    assert invert_update_priority(0.0) == 1.0
    assert live_narrow_far_m(4.0, 800.0, elapsed_s=0.0, unique_n=10) == 800.0  # warmup
    assert live_narrow_far_m(4.0, 800.0, elapsed_s=LIVE_NARROW_HITCH_AFTER_S, unique_n=10) == NARROW_FAR_LIVE_HITCH_M
    assert live_narrow_far_m(12.0, 800.0, elapsed_s=LIVE_NARROW_HITCH_AFTER_S, unique_n=10) == 800.0
    assert NARROW_FAR_LIVE_HITCH_M == 400.0
    assert NARROW_FAR_LIVE_HITCH_M >= 300.0
    assert far_hitch_ladder("narrow", 800.0, unique_hz=4.0) == (800.0, NARROW_FAR_LIVE_HITCH_M)
    assert far_hitch_ladder("narrow", 800.0, unique_hz=12.0) == (800.0,)

    assert abs(camera_update_priority({"id": "main"}, cid="main") - 0.0) < 1e-9
    assert camera_update_priority({"id": "narrow"}, cid="narrow") > camera_update_priority({"id": "main"}, cid="main")
    assert DEFAULT_UPDATE_PRIORITY["rear"] >= DEFAULT_UPDATE_PRIORITY["narrow"]

    narrow_tries = iter_clip_attach_attempts({"id": "narrow", "far_m": 800, "requested_update_time": 0.067})
    assert narrow_tries == [(0.05, 800.0, 0.067)]
    main_tries = iter_clip_attach_attempts({"id": "main", "far_m": 300, "requested_update_time": 0.067})
    assert main_tries == [(0.05, 300.0, 0.067)]
    wide_tries = iter_clip_attach_attempts({"id": "wide", "far_m": 300})
    assert wide_tries == [(0.05, 300.0, 0.067)]
    side = iter_clip_attach_attempts({"id": "pillarL", "far_m": 100, "requested_update_time": -1}, update_s=0.067)
    assert side == [(0.05, 100.0, ON_DEMAND_UPDATE_S)]
    # global 0.067 cannot clobber side/rear on-demand -1
    side_default = iter_clip_attach_attempts({"id": "pillarL"}, update_s=0.067)
    assert side_default == [(0.05, 100.0, ON_DEMAND_UPDATE_S)]
    rear_default = iter_clip_attach_attempts({"id": "rear"}, update_s=0.067)
    assert rear_default == [(0.05, 100.0, ON_DEMAND_UPDATE_S)]

    kw = beamng_camera_sensor_kwargs(
        pos=(0.0, -1.2, 1.26),
        direction=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_v=21.4,
        resolution=(640, 480),
        update_s=0.067,
        near_m=0.05,
        far_m=800.0,
        shmem=True,
        streaming=False,  # caller cannot turn streaming off
        rgb_only=True,
        update_priority=0.5,
    )
    assert kw["near_far_planes"] == (0.05, 800.0)
    assert kw["requested_update_time"] == 0.067
    assert kw["is_streaming"] is True
    assert kw["update_priority"] == 0.5
    assert kw["is_render_depth"] is False
    assert kw["is_render_annotations"] is False
    assert kw["resolution"] == (640, 480)
    assert "near_far_planes" in kw

    import numpy as np

    rgb = colour_to_bgr(np.zeros((8, 8, 3), dtype=np.uint8))
    assert rgb is not None and rgb.shape == (8, 8, 3)
    raw4 = np.zeros((8, 8, 4), dtype=np.uint8).tobytes()
    bgr = colour_to_bgr(raw4, (8, 8))
    assert bgr is not None and bgr.shape == (8, 8, 3)

    src = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    assert "near_far_planes" in src
    assert "beamng_camera_sensor_kwargs" in src
    assert "Camera(f\"gvd_{cid}\", bng, vehicle, **kwargs)" in src
    assert "requested_update_time" in src
    assert "stream_raw" in src
    assert "read_camera_colour" in src
    assert "poll_camera_colour" not in src
    assert "NARROW_FAR_LIVE_HITCH_M" in src
    assert "live_narrow_far_m" in src
    assert "unique_gpu_n" in src
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "loop_hz=args.hz" not in rv
    assert "loop_hz=loop_hz_ema" in rv
    assert "gpu_vram_used_gb" in rv
    assert "unique_gpu_n" in rv
    assert "unique_frame_hz_inst" in rv
    assert "if main is not None:" not in rv
    hw = (ROOT / "python" / "runtime" / "hw_probe.py").read_text(encoding="utf-8")
    assert "GPU_VRAM_CACHE_S" in hw
    assert "def ema_hz" in hw
    assert "NVIDIA_SMI_TIMEOUT_S" in hw
    assert "timeout=3" not in hw
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "BeamNGpy #199" in readme or "BeamNGpy/issues/199" in readme
    assert "narrow > main" in readme
    assert "@ -1" in readme or "@ **-1**" in readme
    assert "0.13 half-rate" not in readme
    assert "0.267" not in readme
    schema = (ROOT / "docs" / "gvd_state_schema.md").read_text(encoding="utf-8")
    assert "`grab_ms`" in schema
    assert "unique GPU-frame" in schema
    for key in (
        "grab_phase",
        "grab_poll_free",
        "sensors_poll_ms",
        "poll_gps_ms",
        "poll_gps_sent",
        "electrics_ms",
    ):
        assert f"`{key}`" in schema, key


def check_beamngpy_open_passes_near_far() -> None:
    """BeamNGPyBackend.open() constructs Camera(..., near_far_planes=(0.05, far_m)) per id."""
    import sys
    import types

    from python.sensors.cameras import BeamNGPyBackend

    captured: list[tuple[str, dict]] = []

    class FakeCamera:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            captured.append((name, kwargs))
            self.update_priority = float(kwargs.get("update_priority", 0.0))

        def get_update_priority(self):
            return self.update_priority

        def remove(self):
            return None

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = FakeCamera
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    try:
        be = BeamNGPyBackend(
            config=load_camera_config(),
            tech_config={
                "wait_vehicle_s": 0,
                "cameras": {"attach": True, "rgb_only": True, "update_s": 0.067, "shared_memory": True, "streaming": True},
            },
        )

        def _connect(explicit=True):
            be.session.vehicle = object()
            be.session.bng = object()
            return True

        be.session.connect = _connect  # type: ignore[method-assign]
        be.session.attach_vehicle_sensors = lambda: {}  # type: ignore[method-assign]
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            be.open()
        log = buf.getvalue()
        assert "hitch steps narrow:" in log
        assert "far_m=800@update_s=-1" in log
        assert "hitch steps rear:" in log
        assert "far_m=100@update_s=-1" in log
        assert "hitch steps pillarL:" in log
        assert "not resolution" in log
        assert "grab_div main=1" in log
        assert "wide=16" in log
        assert "narrow=16" in log
        assert "stream_raw main" in log
        assert "rear=16" in log
        assert "depth/semantic OFF" in log
        names = [n for n, _ in captured]
        assert names == [f"gvd_{c}" for c in CAM_IDS], names
        by = {n: kw for n, kw in captured}
        assert by["gvd_narrow"]["near_far_planes"] == (0.05, 800.0)
        assert by["gvd_main"]["near_far_planes"] == (0.05, 300.0)
        assert by["gvd_wide"]["near_far_planes"] == (0.05, 300.0)
        assert by["gvd_narrow"]["requested_update_time"] == ON_DEMAND_UPDATE_S
        assert by["gvd_main"]["requested_update_time"] == 0.067
        assert by["gvd_wide"]["requested_update_time"] == ON_DEMAND_UPDATE_S
        assert by["gvd_main"]["update_priority"] == 0.0
        assert by["gvd_narrow"]["update_priority"] > by["gvd_main"]["update_priority"]
        assert read_camera_update_priority(be._sensors["main"]) == 0.0
        assert priority_highest_is_zero(be._sensors["main"], 0.0)
        assert invert_update_priority(0.0) == 1.0
        assert CAMERA_HZ_TARGET == 10.0
        for cid in SIDE_CAM_IDS | REAR_CAM_IDS:
            assert by[f"gvd_{cid}"]["near_far_planes"] == (0.05, 100.0), cid
            assert by[f"gvd_{cid}"]["requested_update_time"] == ON_DEMAND_UPDATE_S, cid
            assert by[f"gvd_{cid}"]["is_streaming"] is True, cid
            assert by[f"gvd_{cid}"]["is_render_depth"] is False
            assert by[f"gvd_{cid}"]["resolution"][0] >= 1
        assert by["gvd_narrow"]["is_render_depth"] is False
        assert by["gvd_narrow"]["is_streaming"] is True
        assert by["gvd_narrow"]["resolution"] == (640, 480)
        assert be._clip_planes["narrow"] == (0.05, 800.0)
        assert be._clip_planes["main"] == (0.05, 300.0)
        assert be._update_s["rear"] == ON_DEMAND_UPDATE_S
        assert be._side_grab_div == 16
        assert be._rear_grab_div == 16
        assert be._grab_div["main"] == 1
        assert be._grab_div["wide"] == 16
        assert be._grab_div["narrow"] == 16
        hitch_by = {cid: (near, far, rate) for cid, near, far, rate in be._hitch_steps}
        assert hitch_by["narrow"] == (0.05, 800.0, ON_DEMAND_UPDATE_S)
        assert hitch_by["main"] == (0.05, 300.0, 0.067)
        assert hitch_by["wide"] == (0.05, 300.0, ON_DEMAND_UPDATE_S)
        assert hitch_by["pillarL"][1] == 100.0 and hitch_by["pillarL"][2] == ON_DEMAND_UPDATE_S
        assert hitch_by["rear"][1] == 100.0 and hitch_by["rear"][2] == ON_DEMAND_UPDATE_S
        # always explicit near_far_planes; forward never 100; never both-800
        for name, kw in captured:
            assert "near_far_planes" in kw, name
            assert kw["is_streaming"] is True, name
            far = kw["near_far_planes"][1]
            if name in ("gvd_narrow", "gvd_main", "gvd_wide"):
                assert far != 100.0, name
            else:
                assert far == 100.0, name
        assert by["gvd_narrow"]["near_far_planes"][1] != by["gvd_main"]["near_far_planes"][1]
        assert by["gvd_main"]["near_far_planes"][1] != 800.0
        assert be._ok
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def check_beamngpy_side_grab_half_rate() -> None:
    """Engage and Soft Esc: main stream_raw every tick. Companions poll on the hitch.

    Soft Esc does not burst all seven and does not skip the wheel. At most one
    companion is polled per tick (÷16). A tick that skips a cam keeps the last
    non-blank frame and health OK. An all-zero buffer is not a picture: the
    slot stays missing until a real frame arrives. capture_note says
    soft_esc_colour=hitch, not main-only. A live engage file or debug
    force-engage is the same hitch (the note omits the soft-esc tag).
    """
    import sys
    import types

    import numpy as np

    import python.control.actuate as act
    from python.sensors.cameras import (
        BeamNGPyBackend,
        CamHealth,
        REAR_CAM_IDS,
        SIDE_CAM_IDS,
        soft_esc_colour_main_only,
    )

    streams: dict[str, int] = {}
    polls: dict[str, int] = {}
    # Names in this set return an all-zero colour buffer. Every other read
    # sets one non-zero pixel so it counts as a picture.
    blank_colour: set[str] = set()

    class FakeCamera:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.is_streaming = True
            self.update_priority = float(kwargs.get("update_priority", 0.0))
            self.resolution = kwargs.get("resolution", (8, 8))
            streams[name] = 0
            polls[name] = 0

        def get_update_priority(self):
            return self.update_priority

        def set_update_priority(self, p):
            self.update_priority = float(p)

        def stream_raw(self):
            streams[self.name] += 1
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            if self.name not in blank_colour:
                img[0, 0, 0] = (streams[self.name] % 255) + 1
            return {"colour": img}

        def poll(self):
            polls[self.name] += 1
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            if self.name not in blank_colour:
                img[0, 0, 1] = (polls[self.name] % 255) + 1
            return {"colour": img}

        def stream(self):
            raise AssertionError("forwards must use stream_raw, not stream()")

        def remove(self):
            return None

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = FakeCamera
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    prev_latch = act.soft_esc_sensors_every_tick()
    engage = act.engage_path()
    prev_bytes = engage.read_bytes() if engage.is_file() else None
    try:
        be = BeamNGPyBackend(
            config=load_camera_config(),
            tech_config={
                "wait_vehicle_s": 0,
                "cameras": {"attach": True, "rgb_only": True, "update_s": 0.067, "shared_memory": True, "streaming": True},
            },
        )

        def _connect(explicit=True):
            be.session.vehicle = object()
            be.session.bng = object()
            return True

        be.session.connect = _connect  # type: ignore[method-assign]
        be.session.attach_vehicle_sensors = lambda: {}  # type: ignore[method-assign]
        be.open()
        assert len(be._sensors) == 8
        assert set(be._sensors) == set(CAM_IDS)
        assert read_camera_update_priority(be._sensors["main"]) == 0.0
        assert priority_highest_is_zero(be._sensors["main"], 0.0)
        # Engage hitch colour. The latch alone keeps companion polls.
        act.note_soft_esc_engaged(True)
        act.write_engage_flag(False)
        assert soft_esc_colour_main_only() is False
        n = 16
        last = None
        wheel = {
            0: {"main", "wide"},
            1: {"main", "narrow"},
            2: {"main", "pillarL"},
            3: {"main", "pillarR"},
            4: {"main"},
            5: {"main", "repeatL"},
            6: {"main", "rear"},
            7: {"main", "repeatR"},
        }
        for slot in range(8, 16):
            wheel[slot] = {"main"}
        poll_ids = SIDE_CAM_IDS | REAR_CAM_IDS | {"wide", "narrow"}
        for i in range(n):
            s0 = dict(streams)
            p0 = dict(polls)
            bundle = be.grab()
            last = bundle
            assert bundle.grab_phase == i
            assert bundle.grab_poll_free is grab_is_poll_free(i, be._hitch)
            assert bundle.grab_ms >= 0.0
            streamed = [cid for cid in ("main", "wide", "narrow") if streams[f"gvd_{cid}"] > s0.get(f"gvd_{cid}", 0)]
            assert streamed == ["main"], (i, streamed)
            assert "wide" not in streamed and "narrow" not in streamed
            if i == 0:
                for cid in poll_ids:
                    due = camera_grab_due(cid, 0, be._hitch)
                    assert (polls[f"gvd_{cid}"] == 1) == due, (cid, due, polls[f"gvd_{cid}"])
                assert bundle.health["narrow"] == CamHealth.MISSING  # not read yet
                assert bundle.health["narrow"] != CamHealth.STALE
            polled = [
                cid
                for cid in poll_ids
                if polls[f"gvd_{cid}"] > p0.get(f"gvd_{cid}", 0)
            ]
            colour = set(streamed) | set(polled)
            assert len(streamed) <= 1
            assert len(colour) <= 2, (i, colour)
            assert colour == wheel[i], (i, colour)
            assert bundle.grab_poll_free is (colour == {"main"}), (i, colour, bundle.grab_poll_free)
            assert not {"main", "wide"}.issubset(set(streamed))
            assert not ("wide" in polled and "narrow" in polled)
            if camera_grab_due("wide", i, be._hitch):
                assert i % 2 == 0
            if camera_grab_due("narrow", i, be._hitch):
                assert i % 2 == 1
            if camera_grab_due("rear", i, be._hitch):
                assert [cid for cid in polled if cid in SIDE_CAM_IDS] == [], (i, polled)
            for cid in CAM_IDS:
                assert bundle.health[cid] != CamHealth.STALE, (i, cid, bundle.health[cid])
        assert last is not None
        assert last.health["main"] == CamHealth.OK
        assert last.health["narrow"] == CamHealth.OK
        assert last.health["wide"] == CamHealth.OK  # grab_i=7 skip keeps the last good frame
        assert last.grab_ms >= 0.0
        assert last.unique_gpu_n >= 1  # incrementing FakeCamera
        for cid in SIDE_CAM_IDS:
            assert last.health[cid] == CamHealth.OK, cid
            assert cid in last.frames
        for cid in REAR_CAM_IDS:
            assert last.health[cid] == CamHealth.OK, cid  # skip is not STALE
            assert cid in last.frames
        assert streams["gvd_main"] == n
        assert polls["gvd_main"] == 0
        assert streams["gvd_wide"] == 0
        assert streams["gvd_narrow"] == 0
        assert polls["gvd_wide"] == n // WIDE_GRAB_DIV
        assert polls["gvd_narrow"] == n // NARROW_GRAB_DIV
        for cid in ("pillarL", "pillarR"):
            assert polls[f"gvd_{cid}"] == n // SIDE_GRAB_DIV, (cid, polls[f"gvd_{cid}"])
            assert streams[f"gvd_{cid}"] == 0, cid
        for cid in ("repeatL", "repeatR"):
            assert polls[f"gvd_{cid}"] == n // REPEAT_GRAB_DIV, (cid, polls[f"gvd_{cid}"])
            assert streams[f"gvd_{cid}"] == 0, cid
        for cid in REAR_CAM_IDS:
            assert polls[f"gvd_{cid}"] == n // REAR_GRAB_DIV, (cid, polls[f"gvd_{cid}"])
            assert streams[f"gvd_{cid}"] == 0, cid
        assert be._clip_planes["narrow"][1] > be._clip_planes["main"][1]
        assert "rear_div=16" in last.note
        assert "side_div=16" in last.note
        assert "wide_div=16" in last.note
        assert "narrow_div=16" in last.note
        assert "main_div=1" in last.note
        assert "soft_esc_colour=main" not in last.note

        # Soft Esc: same ÷16 hitch. Last non-blank frame stays on a skip tick.
        act.note_soft_esc_engaged(False)
        act.write_engage_flag(False)
        assert act.soft_esc_sensors_every_tick() is False
        assert soft_esc_colour_main_only() is True
        s_soft = dict(streams)
        for i in range(n):
            gi = be._grab_i
            due = [cid for cid in poll_ids if camera_grab_due(cid, gi, be._hitch)]
            before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            bundle = be.grab()
            assert bundle.grab_phase == gi % 16
            assert len(due) <= 1, (gi, due)
            assert bundle.grab_poll_free is (len(due) == 0)
            assert "soft_esc_colour=hitch" in bundle.note
            assert "soft_esc_colour=main" not in bundle.note
            assert "soft_esc_warm" not in bundle.note
            assert bundle.health["main"] == CamHealth.OK
            assert streams["gvd_main"] == s_soft["gvd_main"] + i + 1
            assert polls["gvd_main"] == 0
            for cid in CAM_IDS:
                if cid == "main":
                    continue
                assert polls[f"gvd_{cid}"] == before_p[cid] + (1 if cid in due else 0), cid
                assert streams[f"gvd_{cid}"] == s_soft[f"gvd_{cid}"], cid
                assert bundle.health[cid] == CamHealth.OK, cid
                assert cid in bundle.frames
                assert int(bundle.frames[cid].max()) > 0
        assert len(be._sensors) == 8
        assert set(be._sensors) == set(CAM_IDS)
        assert camera_grab_due("wide", n, be._hitch)
        assert not camera_grab_due("narrow", n, be._hitch)

        # Dropping wide does not colour it until its hitch slot, and does not
        # poll the other six on that same tick. Until then the slot is missing,
        # not a black OK. After the slot, later ticks keep the last frame.
        be._cache_frames.pop("wide", None)
        be._cache_ts.pop("wide", None)
        be._frame_sig.pop("wide", None)
        seen_wide = False
        for _step in range(16):
            gi = be._grab_i
            due = [cid for cid in poll_ids if camera_grab_due(cid, gi, be._hitch)]
            before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            cold = be.grab()
            assert "soft_esc_colour=hitch" in cold.note
            assert "soft_esc_warm" not in cold.note
            assert len(due) <= 1
            for cid in poll_ids:
                assert polls[f"gvd_{cid}"] == before_p[cid] + (1 if cid in due else 0), cid
            assert cold.health["main"] == CamHealth.OK
            if "wide" in due:
                assert cold.health["wide"] == CamHealth.OK
                assert int(cold.frames["wide"].max()) > 0
                assert cold.grab_poll_free is False
                seen_wide = True
            elif not seen_wide:
                assert cold.health["wide"] == CamHealth.MISSING
                assert "wide" not in cold.frames
            else:
                assert cold.health["wide"] == CamHealth.OK
                assert "wide" in cold.frames
                assert int(cold.frames["wide"].max()) > 0
            for cid in poll_ids:
                if cid == "wide":
                    continue
                assert cold.health[cid] == CamHealth.OK, cid
                assert cid in cold.frames
        assert seen_wide
        assert streams["gvd_wide"] == 0
        assert len(be._sensors) == 8

        # Rising edge: latch still false. A live engage file colours the due companion.
        guard = 0
        while [
            cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, be._grab_i, be._hitch)
        ] != ["narrow"]:
            guard += 1
            assert guard <= 16
            be.grab()
        act.write_engage_flag(True)
        assert act.soft_esc_sensors_every_tick() is False
        assert soft_esc_colour_main_only() is False
        gi = be._grab_i
        due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
        assert due == ["narrow"], (gi, due)
        narrow_polls = polls["gvd_narrow"]
        wide_polls = polls["gvd_wide"]
        main_streams = streams["gvd_main"]
        edge = be.grab()
        assert polls["gvd_narrow"] == narrow_polls + 1
        assert streams["gvd_narrow"] == 0
        assert streams["gvd_main"] == main_streams + 1
        assert edge.health["narrow"] == CamHealth.OK
        assert edge.grab_poll_free is False
        assert "soft_esc_colour=hitch" not in edge.note
        assert "soft_esc_colour=main" not in edge.note
        assert "soft_esc_warm" not in edge.note
        assert polls["gvd_wide"] == wide_polls  # this slot is narrow, not wide
        assert edge.health["wide"] == CamHealth.OK
        assert len(be._sensors) == 8

        act.write_engage_flag(False)
        act.note_soft_esc_engaged(False)
        assert soft_esc_colour_main_only() is True

        def _cam_ok(bundle) -> int:
            hs = bundle.health_str()
            return sum(1 for cid in CAM_IDS if hs.get(cid) == "ok")

        def _drop_companion_caches() -> None:
            for cid in CAM_IDS:
                if cid == "main":
                    continue
                be._cache_frames.pop(cid, None)
                be._cache_ts.pop(cid, None)
                be._frame_sig.pop(cid, None)

        # Cold Soft Esc: one companion per hitch slot, then last-frame 8/8.
        # Not a single-tick warm of all seven.
        _drop_companion_caches()
        cold_before = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
        cold_streams = {cid: streams[f"gvd_{cid}"] for cid in CAM_IDS}
        got: set[str] = set()
        for step in range(16):
            gi = be._grab_i
            due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
            warmed = be.grab()
            assert len(due) <= 1, (gi, due)
            assert "soft_esc_colour=hitch" in warmed.note
            assert "soft_esc_warm" not in warmed.note
            assert streams["gvd_main"] == cold_streams["main"] + step + 1
            got.update(due)
            for cid in CAM_IDS:
                if cid == "main":
                    assert warmed.health[cid] == CamHealth.OK
                    continue
                assert streams[f"gvd_{cid}"] == cold_streams[cid], cid
                if cid in got:
                    assert warmed.health[cid] == CamHealth.OK, cid
                    assert cid in warmed.frames
                    assert int(warmed.frames[cid].max()) > 0
                else:
                    assert warmed.health[cid] == CamHealth.MISSING, cid
                    assert cid not in warmed.frames
        for cid in CAM_IDS:
            if cid == "main":
                continue
            assert polls[f"gvd_{cid}"] == cold_before[cid] + 1, cid
        assert _cam_ok(warmed) == 8
        # Next tick keeps every last frame. Only the due companion is polled again.
        steady_before = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
        gi = be._grab_i
        due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
        steady = be.grab()
        assert "soft_esc_colour=hitch" in steady.note
        assert "soft_esc_warm" not in steady.note
        assert steady.grab_poll_free is (len(due) == 0)
        assert _cam_ok(steady) == 8
        for cid in CAM_IDS:
            assert cid in steady.frames, cid
            assert int(steady.frames[cid].max()) > 0
            if cid == "main":
                continue
            assert polls[f"gvd_{cid}"] == steady_before[cid] + (1 if cid in due else 0), cid
            assert streams[f"gvd_{cid}"] == cold_streams[cid], cid
            if cid not in due:
                assert cid not in steady.unique_gpu_ids
        assert len(be._sensors) == 8

        def _advance_to_companion_slot() -> None:
            # Soft Esc only. A force-engage frame would hitch-colour these grabs.
            assert soft_esc_colour_main_only() is True
            guard = 0
            while not any(
                cid != "main" and camera_grab_due(cid, be._grab_i, be._hitch) for cid in CAM_IDS
            ):
                guard += 1
                assert guard <= 16
                held = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
                skipped = be.grab()
                assert skipped.grab_poll_free is True
                assert "soft_esc_colour=hitch" in skipped.note
                assert _cam_ok(skipped) == 8
                for cid in CAM_IDS:
                    if cid == "main":
                        continue
                    assert polls[f"gvd_{cid}"] == held[cid], cid

        def _force_edge_grab():
            gi = be._grab_i
            due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
            assert len(due) == 1, (gi, due)
            before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            before_s = {cid: streams[f"gvd_{cid}"] for cid in CAM_IDS}
            bundle = be.grab()
            return due, before_p, before_s, bundle

        def _assert_force_edge(due, before_p, before_s, bundle) -> None:
            assert soft_esc_colour_main_only() is True
            assert "soft_esc_colour=main" not in bundle.note
            assert "soft_esc_warm=1" not in bundle.note
            assert bundle.grab_poll_free is False
            assert streams["gvd_main"] == before_s["main"] + 1
            for cid in CAM_IDS:
                if cid == "main":
                    continue
                assert streams[f"gvd_{cid}"] == before_s[cid], cid
                assert bundle.health[cid] == CamHealth.OK, cid
                if cid in due:
                    assert polls[f"gvd_{cid}"] == before_p[cid] + 1, cid
                else:
                    assert polls[f"gvd_{cid}"] == before_p[cid], cid
            # The slot after a companion is not always poll-free (0 then 1).
            # Walk to the next main-only tick. Last frames stay painted.
            guard = 0
            while not grab_is_poll_free(be._grab_i, be._hitch):
                guard += 1
                assert guard <= 16
                gi = be._grab_i
                due_now = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
                mid_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
                mid = be.grab()
                assert mid.grab_poll_free is False
                assert "soft_esc_colour=hitch" in mid.note
                assert _cam_ok(mid) == 8
                for cid in CAM_IDS:
                    if cid == "main":
                        continue
                    assert polls[f"gvd_{cid}"] == mid_p[cid] + (1 if cid in due_now else 0), cid
                    assert cid in mid.frames
            quiet_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            quiet_main = streams["gvd_main"]
            quiet = be.grab()
            assert quiet.grab_poll_free is True
            assert "soft_esc_colour=hitch" in quiet.note
            assert _cam_ok(quiet) == 8
            assert streams["gvd_main"] == quiet_main + 1
            for cid in CAM_IDS:
                if cid == "main":
                    continue
                assert polls[f"gvd_{cid}"] == quiet_p[cid], cid
                assert cid in quiet.frames
                assert int(quiet.frames[cid].max()) > 0

        # Debug --force-engage / force_engage: rising edge hitch-colours the due
        # companion while the latch is still false and the engage file is absent.
        def _via_args():
            args = type("A", (), {"force_engage": True})()
            assert args.force_engage is True
            assert act.soft_esc_sensors_every_tick() is False
            assert soft_esc_colour_main_only() is False
            return _force_edge_grab()

        _advance_to_companion_slot()
        _assert_force_edge(*_via_args())

        def _via_ui():
            ui = type("U", (), {})()
            ui.debug = type("D", (), {"force_engage": True})()
            assert soft_esc_colour_main_only() is False
            return _force_edge_grab()

        _advance_to_companion_slot()
        _assert_force_edge(*_via_ui())

        _advance_to_companion_slot()
        saved_argv = list(sys.argv)
        sys.argv = [*saved_argv, "--force-engage"]
        try:
            assert soft_esc_colour_main_only() is False
            argv_edge = _force_edge_grab()
        finally:
            sys.argv = saved_argv
        _assert_force_edge(*argv_edge)

        # Cold debug engage is hitch colour, not a Soft Esc warm of all seven.
        def _cold_force():
            args = type("A", (), {"force_engage": True})()
            assert soft_esc_colour_main_only() is False
            _drop_companion_caches()
            gi = be._grab_i
            due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
            assert len(due) == 1, (gi, due)
            before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            bundle = be.grab()
            return due[0], before_p, bundle

        _advance_to_companion_slot()

        due_cid, before_p, forced_cold = _cold_force()
        assert soft_esc_colour_main_only() is True
        assert polls[f"gvd_{due_cid}"] == before_p[due_cid] + 1
        assert forced_cold.health[due_cid] == CamHealth.OK
        assert "soft_esc_warm=1" not in forced_cold.note
        assert "soft_esc_colour=main" not in forced_cold.note
        for cid in CAM_IDS:
            if cid in ("main", due_cid):
                continue
            assert polls[f"gvd_{cid}"] == before_p[cid], cid
            assert forced_cold.health[cid] == CamHealth.MISSING, cid

        # Cold all-zero stream_raw and poll. Zeros stay missing. After one real
        # frame, the next zero read keeps that picture and does not go black.
        blank_colour.update(f"gvd_{cid}" for cid in CAM_IDS)
        _drop_companion_caches()
        be._cache_frames.pop("main", None)
        be._cache_ts.pop("main", None)
        be._frame_sig.pop("main", None)
        for _step in range(8):
            gi = be._grab_i
            due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
            main_n = streams["gvd_main"]
            before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
            coldz = be.grab()
            assert streams["gvd_main"] == main_n + 1
            assert polls["gvd_main"] == 0
            assert "soft_esc_colour=hitch" in coldz.note
            assert "cam_main" not in coldz.frames
            for cid in CAM_IDS:
                assert coldz.health[cid] == CamHealth.MISSING, (cid, coldz.health[cid])
                assert cid not in coldz.frames
            if due:
                assert polls[f"gvd_{due[0]}"] == before_p[due[0]] + 1
        blank_colour.clear()
        gi = be._grab_i
        due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
        live = be.grab()
        assert live.health["main"] == CamHealth.OK
        assert int(live.frames["main"].max()) > 0
        kept = {"main"}
        main_pic = live.frames["main"].copy()
        kept_pic = {"main": main_pic}
        if due:
            assert live.health[due[0]] == CamHealth.OK
            assert int(live.frames[due[0]].max()) > 0
            kept.add(due[0])
            kept_pic[due[0]] = live.frames[due[0]].copy()
        for cid in CAM_IDS:
            if cid not in kept:
                assert live.health[cid] == CamHealth.MISSING, cid
                assert cid not in live.frames
        blank_colour.update(f"gvd_{cid}" for cid in CAM_IDS)
        gi = be._grab_i
        due_zero = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, gi, be._hitch)]
        main_n = streams["gvd_main"]
        before_p = {cid: polls[f"gvd_{cid}"] for cid in CAM_IDS}
        heldz = be.grab()
        assert streams["gvd_main"] == main_n + 1
        assert "soft_esc_colour=hitch" in heldz.note
        for cid, pic in kept_pic.items():
            assert heldz.health[cid] == CamHealth.OK, cid
            assert cid in heldz.frames
            assert np.array_equal(heldz.frames[cid], pic)
        for cid in CAM_IDS:
            if cid in kept:
                continue
            assert heldz.health[cid] == CamHealth.MISSING, cid
            assert cid not in heldz.frames
        if due_zero:
            assert polls[f"gvd_{due_zero[0]}"] == before_p[due_zero[0]] + 1
        blank_colour.clear()
        for _step in range(16):
            be.grab()

        # A failed read with no picture stays missing and is retried on the next
        # wide slot. It is not frozen, and it is not reported OK.
        fail_n = {"n": 0}

        class FailWide:
            is_streaming = True
            resolution = (8, 8)

            def poll(self):
                fail_n["n"] += 1
                return None

            def stream_raw(self):
                raise AssertionError("wide must not stream_raw")

        saved_wide = be._sensors["wide"]
        be._sensors["wide"] = FailWide()
        be._cache_frames.pop("wide", None)
        be._cache_ts.pop("wide", None)

        def _due_now() -> list[str]:
            return [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, be._grab_i, be._hitch)]

        guard = 0
        while "wide" not in _due_now():
            guard += 1
            assert guard <= 16
            be.grab()
        missed = be.grab()
        assert missed.health["wide"] == CamHealth.MISSING
        assert "wide" not in missed.frames
        assert fail_n["n"] == 1
        assert "soft_esc_colour=hitch" in missed.note
        assert "soft_esc_warm" not in missed.note
        guard = 0
        while "wide" not in _due_now():
            guard += 1
            assert guard <= 16
            be.grab()
            assert fail_n["n"] == 1
        again = be.grab()
        assert again.health["wide"] == CamHealth.MISSING
        assert fail_n["n"] == 2
        assert "soft_esc_colour=hitch" in again.note
        be._sensors["wide"] = saved_wide
        painted = np.zeros((8, 8, 3), dtype=np.uint8)
        painted[0, 0, 1] = 40
        be._cache_frames["wide"] = painted
        be._cache_ts["wide"] = 0.0

        # Main never polls. Wide/narrow poll and do not stream_raw.
        class NoStream:
            is_streaming = True
            resolution = (8, 8)

            def poll(self):
                raise AssertionError("main must not poll()")

        assert read_camera_colour(NoStream(), cid="main", resolution=(8, 8)) is None
        assert "main" in FORWARD_CAM_IDS

        class WidePoll:
            is_streaming = True
            resolution = (8, 8)

            def stream_raw(self):
                raise AssertionError("wide must not stream_raw")

            def poll(self):
                return {"colour": np.zeros((8, 8, 3), dtype=np.uint8)}

        assert read_camera_colour(WidePoll(), cid="wide", resolution=(8, 8)) is not None
        assert read_camera_colour(WidePoll(), cid="narrow", resolution=(8, 8)) is not None

        # failed stream_raw → STALE. A scheduled skip above stayed OK.
        class FailRaw:
            is_streaming = True
            resolution = (8, 8)

            def stream_raw(self):
                return None

            def poll(self):
                raise AssertionError("main must not poll()")

        be._sensors["main"] = FailRaw()
        failed = be.grab()
        assert failed.health["main"] == CamHealth.STALE
        assert "main" in failed.frames  # last good pixels stay
        assert "main" not in failed.unique_gpu_ids
        be._cache_frames.pop("main", None)
        be._cache_ts.pop("main", None)
        be._frame_sig.pop("main", None)
        empty = be.grab()
        assert empty.health["main"] == CamHealth.MISSING
        assert "main" not in empty.frames
        assert "cam_main" not in empty.frames
        assert "main" not in empty.unique_gpu_ids

        # cache re-show is not a unique GPU frame
        class SameRaw:
            is_streaming = True
            resolution = (8, 8)

            def stream_raw(self):
                img = np.zeros((8, 8, 3), dtype=np.uint8)
                img[1, 1, 1] = 30  # same bytes every call; a zero buffer is not a frame
                return {"colour": img}

        be._sensors["main"] = SameRaw()
        be._frame_sig.pop("main", None)
        first = be.grab()
        second = be.grab()
        assert "main" in first.unique_gpu_ids
        assert "main" not in second.unique_gpu_ids

        # live unique-Hz hitch 800→400 (warmup already elapsed)
        import time as _time

        be._open_mono = _time.monotonic() - LIVE_NARROW_HITCH_AFTER_S - 0.1
        be._unique_n = 10
        be._unique_hz_ema = 4.0
        be._narrow_live_hitched = False
        be._maybe_live_narrow_hitch()
        assert be._narrow_live_hitched
        assert abs(be._clip_planes["narrow"][1] - NARROW_FAR_LIVE_HITCH_M) < 1e-9
        assert be._clip_planes["narrow"][1] >= be._clip_planes["main"][1]
    finally:
        act.note_soft_esc_engaged(prev_latch)
        if prev_bytes is None:
            engage.unlink(missing_ok=True)
        else:
            engage.write_bytes(prev_bytes)
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def check_soft_esc_segment_timers() -> None:
    """Soft Esc Tip #2 timers, plus Tip #4 coalesce while disengaged.

    Hitch schedule stays put. engaged=false reuses last-good inside 200 ms.
    Engage polls ``vehicle.sensors.poll`` every grab.
    """
    import time

    hitch = load_camera_config().get("hitch")
    assert grab_wheel(hitch) == 16
    assert grab_wheel(None) == 16
    hitch_slots = [i for i in range(16) if not grab_is_poll_free(i, hitch)]
    free_slots = [i for i in range(16) if grab_is_poll_free(i, hitch)]
    assert hitch_slots == [0, 1, 2, 3, 5, 6, 7], hitch_slots
    assert free_slots == [4, 8, 9, 10, 11, 12, 13, 14, 15], free_slots
    for i in range(32):
        assert grab_phase_of(i, hitch) == i % 16
        assert grab_is_poll_free(i, None) is grab_is_poll_free(i, hitch)

    from python.sensors.cameras import StubBackend

    stub = StubBackend().grab()
    assert stub.grab_phase == -1 and stub.grab_poll_free is True
    assert stub.grab_ms == 0.0

    class Box(dict):
        def __contains__(self, k):
            return dict.__contains__(self, k)

        def __iter__(self):
            return dict.__iter__(self)

    class FakeSensors(Box):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.n = 0

        def poll(self):
            self.n += 1
            time.sleep(0.02)
            return self

    class FakeGPS:
        def __init__(self) -> None:
            self.n = 0
            self.removed = False

        def poll(self):
            self.n += 1
            time.sleep(0.02)
            return [{"time": 1.0, "lon": 8.81, "lat": 53.09, "x": 1.0, "y": 2.0}]

        def remove(self) -> None:
            self.removed = True

    class FakeVeh:
        vid = "etk_player"
        options = {"model": "etk800"}
        state = {"pos": (0.0, 0.0, 0.0), "dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0), "vel": (0.0, 1.0, 0.0)}

        def __init__(self) -> None:
            self.sensors = FakeSensors(
                electrics={"wheelspeed": 3.0, "steering_input": 0.0, "throttle_input": 0.0, "brake_input": 0.0}
            )

    session = TechSession({"wait_vehicle_s": 0, "sensors": {"electrics": True, "gps": True}})
    veh = FakeVeh()
    session.vehicle = veh
    session.attached = {"electrics": True, "gps": True}
    gps = FakeGPS()
    session._gps = gps
    import python.control.actuate as act
    from python.sensors.tech import SOFT_ESC_SENSOR_POLL_S

    act.note_soft_esc_engaged(False)
    try:
        assert act.soft_esc_sensors_every_tick() is False
        first = session.poll()
        assert first.sensors_poll_ms >= 10.0, first.sensors_poll_ms
        assert first.poll_gps_sent is True
        assert first.poll_gps_ms >= 10.0, first.poll_gps_ms
        assert gps.n == 1 and veh.sensors.n == 1
        assert first.lat == 53.09 and first.sensors.get("gps") == "ok"
        # Soft Esc engaged=false: second grab inside 200 ms does not poll.
        second = session.poll()
        assert second.poll_gps_sent is False
        assert second.poll_gps_ms == 0.0
        assert second.sensors_poll_ms == 0.0
        assert gps.n == 1  # no PollGPSGE on a coalesced grab
        assert veh.sensors.n == 1  # no second vehicle.sensors.poll
        assert second.sensors.get("gps") == "stale"
        assert second.lat == 53.09 and second.speed_mps == 3.0
        assert "coalesced" in second.note
        nav = nav_snapshot(second)
        assert nav["gps"]["lat"] == 53.09 and "not a new fix" in nav["note"]
        el = act.read_electrics(veh)
        assert el is not None and el["wheelspeed"] == 3.0
        assert veh.sensors.n == 1  # electrics reused the republished last-good map
        # Window elapsed → one ego poll. GPS period still holds.
        session._sensors_poll_mono = time.monotonic() - (SOFT_ESC_SENSOR_POLL_S + 0.01)
        third = session.poll()
        assert veh.sensors.n == 2
        assert third.sensors_poll_ms >= 10.0
        assert gps.n == 1 and third.poll_gps_sent is False and third.poll_gps_ms == 0.0
        assert third.lat == 53.09 and third.sensors.get("gps") == "stale"
        # Pin the GPS window so the Engage grabs below stay inside it.
        session._gps_mono = time.monotonic()
        # Engage: Tip #1, one sensors.poll per grab. No Soft Esc coalesce.
        driver = act.BeamNGPyActuator(veh)
        driver.note_engaged(True)
        assert act.soft_esc_sensors_every_tick() is True
        fourth = session.poll()
        fifth = session.poll()
        assert veh.sensors.n == 4
        assert fourth.sensors_poll_ms >= 10.0 and fifth.sensors_poll_ms >= 10.0
        assert gps.n == 1  # GPS window unchanged while Engage polls every grab
        driver.note_engaged(False)
        assert act.soft_esc_sensors_every_tick() is False
        sixth = session.poll()
        assert veh.sensors.n == 4 and sixth.sensors_poll_ms == 0.0
        assert sixth.poll_gps_sent is False and gps.n == 1
        session.close()
        assert gps.removed is True
    finally:
        act.note_soft_esc_engaged(False)

    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    for key in ("grab_phase", "grab_poll_free", "sensors_poll_ms", "poll_gps_ms", "electrics_ms"):
        assert f'st["{key}"]' in rv, key
    assert "[GVD] seg " in rv
    assert "camera_hz=cam_hz_ema" in rv
    assert "unique_frame_hz_inst" in rv
    assert "loop_hz=args.hz" not in rv


def check_nvidia_smi_cache_and_honest_hz() -> None:
    from python.runtime import hw_probe as hw

    hw._gpu_vram_cached_at = 0.0
    hw._gpu_vram_cached_gb = 0.0
    calls = {"n": 0}

    def _fake_check_output(*_a, **_k):
        calls["n"] += 1
        assert float(_k.get("timeout", 99)) <= 2.0 + 1e-9
        return "2048\n"

    old = __import__("subprocess").check_output
    __import__("subprocess").check_output = _fake_check_output  # type: ignore[method-assign]
    try:
        a = hw.gpu_vram_used_gb(now=10.0, force=True)
        b = hw.gpu_vram_used_gb(now=10.2)
        assert abs(a - 2.0) < 1e-9
        assert b == a
        assert calls["n"] == 1
        c = hw.gpu_vram_used_gb(now=10.51)
        assert calls["n"] == 2
        assert c == a
    finally:
        __import__("subprocess").check_output = old  # type: ignore[method-assign]
        hw._gpu_vram_cached_at = 0.0
        hw._gpu_vram_cached_gb = 0.0
    assert abs(hw.ema_hz(0.0, 4.0) - 4.0) < 1e-9
    mixed = hw.ema_hz(4.0, 12.0)
    assert 4.0 < mixed < 12.0
    assert mixed < 10.0  # honest; not clamped to ≥10
    assert abs(hw.unique_frame_hz_inst(0, 0.1)) < 1e-12
    assert abs(hw.unique_frame_hz_inst(1, 0.1) - 10.0) < 1e-9
    assert abs(hw.unique_frame_hz_inst(8, 0.1) - 10.0) < 1e-9  # one tick, not 80 Hz
    assert hw.NVIDIA_SMI_TIMEOUT_S <= 2.0


def check_auto_backend_not_tech_without_env() -> None:
    import os

    old = os.environ.get("GVD_BEAMNG")
    try:
        os.environ.pop("GVD_BEAMNG", None)
        name = resolve_backend_name("auto")
        assert name != "beamngpy", name
        os.environ["GVD_BEAMNG"] = "1"
        assert resolve_backend_name("auto") == "beamngpy"
        assert resolve_backend_name("beamngpy") == "beamngpy"
        assert resolve_backend_name("window") == "window"
    finally:
        if old is None:
            os.environ.pop("GVD_BEAMNG", None)
        else:
            os.environ["GVD_BEAMNG"] = old


def check_connect_without_beamngpy() -> None:
    session = TechSession({"wait_vehicle_s": 0, "host": "127.0.0.1", "port": 1})
    ok = session.connect(explicit=True)
    # Either beamngpy missing, or connect refused — never a fake vehicle.
    assert ok is False
    assert session.vehicle is None


def check_no_chrome() -> None:
    import re

    chrome = re.compile(r"tesla|\bfsd\b|full self[- ]driving|autopilot", re.I)
    for rel in ("python/sensors/tech.py", "play_gvd_tech.bat", "config/tech.yaml"):
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        # comments/docs may name the honesty rule; scan play_gvd_tech prints only via the bat.
        if rel.endswith(".bat"):
            assert not chrome.search(text), rel


def _hold_env(**extra: str | None):
    import os
    from contextlib import contextmanager

    keys = (
        "GVD_DOCS_DIR",
        "GVD_PRODUCT",
        "GVD_BEAMNG",
        "GVD_BACKEND",
        "GVD_TECH_LAUNCH",
        "GVD_BEAMNGPY_PIN",
        "LOCALAPPDATA",
        "BNG_HOME",
        "BEAMNG_HOME",
        "USERPROFILE",
        "HOME",
    )

    @contextmanager
    def _ctx():
        old = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                if k in extra:
                    val = extra[k]
                    if val is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = val
                elif k in ("GVD_TECH_LAUNCH", "GVD_BEAMNGPY_PIN", "GVD_DOCS_DIR", "BNG_HOME", "BEAMNG_HOME"):
                    os.environ.pop(k, None)
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _ctx()


def _plant_link(folder, *, lua_bus: str, age_s: float = 0.0) -> None:
    import json
    import os
    import time

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "gvd_link.json"
    path.write_text(json.dumps({"lua_bus": lua_bus}), encoding="utf-8")
    if age_s:
        old = time.time() - age_s
        os.utime(path, (old, old))


def _check_launch_fail_prints_argv(open_tech_beamngpy) -> None:
    """Launch failure prints BeamNGpy's real argv. Attach failure does not."""
    import io
    import sys
    import types
    from contextlib import redirect_stdout

    seen: dict = {}

    class FakeBNG:
        def __init__(self, host, port, **kwargs):
            seen["kwargs"] = dict(kwargs)
            self.binary = "Bin64/BeamNG.tech.x64.exe"
            self.gfx = None
            self.quit_on_close = True
            self.last_command_line = None

        def open(self, launch=True, **kwargs):
            seen["binary"] = self.binary
            seen["gfx"] = self.gfx
            seen["quit_on_close"] = self.quit_on_close
            if launch:
                self.last_command_line = (
                    "/opt/techhome/BeamNG.tech.exe -nosteam -tcom -tport 25252 "
                    "-console -tcom-listen-ip 127.0.0.1 -gfx dx11"
                )
                raise RuntimeError("crash")

        def disconnect(self) -> None:
            return None

    class FakeEarly(FakeBNG):
        def open(self, launch=True, **kwargs):
            seen["early_binary"] = self.binary
            seen["early_gfx"] = self.gfx
            raise RuntimeError("missing exe")

    class FakeAttach(FakeBNG):
        def open(self, launch=True, **kwargs):
            raise RuntimeError("attach down")

    mod = types.ModuleType("beamngpy")
    mod.BeamNGpy = FakeBNG
    old = sys.modules.get("beamngpy")
    sys.modules["beamngpy"] = mod
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy("127.0.0.1", 25252, home="/opt/techhome", user=None, launch=True)
            except RuntimeError as exc:
                assert "crash" in str(exc)
            else:
                raise AssertionError("launch should fail")
        out = buf.getvalue()
        assert seen["kwargs"]["binary"] == "BeamNG.tech.exe"
        assert seen["kwargs"]["gfx"] == "dx11"
        assert seen["kwargs"]["quit_on_close"] is False
        assert seen["binary"] == "BeamNG.tech.exe"
        assert seen["gfx"] == "dx11"
        assert seen["quit_on_close"] is False
        assert (
            "argv: /opt/techhome/BeamNG.tech.exe -nosteam -tcom -tport 25252 "
            "-console -tcom-listen-ip 127.0.0.1 -gfx dx11"
        ) in out
        assert "BeamNG.tech.x64.exe" not in out
        assert "Bin64" not in out

        mod.BeamNGpy = FakeEarly
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy("127.0.0.1", 25252, home="/opt/techhome", user=None, launch=True)
            except RuntimeError as exc:
                assert "missing exe" in str(exc)
            else:
                raise AssertionError("early launch should fail")
        out = buf.getvalue()
        assert seen["early_binary"] == "BeamNG.tech.exe"
        assert seen["early_gfx"] == "dx11"
        assert "/opt/techhome/BeamNG.tech.exe" in out
        assert "-tcom" in out and "-console" in out and "-gfx dx11" in out
        assert "BeamNG.tech.x64.exe" not in out
        assert "Bin64" not in out

        mod.BeamNGpy = FakeAttach
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy("127.0.0.1", 25252, home="/opt/techhome", user=None, launch=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError("attach should fail")
        assert "launch failed" not in buf.getvalue()
    finally:
        if old is None:
            sys.modules.pop("beamngpy", None)
        else:
            sys.modules["beamngpy"] = old


def _check_hello_timeout() -> None:
    """Blocking open() must REFUSE inside socket_timeout on 1.35.1 and 1.36."""
    import inspect
    import io
    import os
    import socket
    import sys
    import tempfile
    import threading
    import time
    import types
    from contextlib import redirect_stdout
    from pathlib import Path

    from python.sensors.tech import (
        TECH_SOCKET_TIMEOUT_S,
        TechHelloTimeout,
        TechSession,
        drop_tech_beamngpy,
        load_tech_config,
        open_tech_beamngpy,
        resolve_socket_timeout,
        tech_hold_gate,
    )

    block_s = 3.0
    cap = 0.4
    os.environ.pop("GVD_TECH_SOCKET_TIMEOUT", None)
    seen: dict = {}
    instances: list = []

    class FakeBNG:
        """1.36-shaped: accepts socket_timeout, then blocks in open()."""

        def __init__(self, host, port, **kwargs):
            self.kwargs = dict(kwargs)
            self.binary = "Bin64/BeamNG.tech.x64.exe"
            self.gfx = None
            self.socket_timeout = kwargs.get("socket_timeout")
            self.quit_on_close = kwargs.get("quit_on_close", True)
            self.last_command_line = None
            self.events: list[str] = []
            self.host = host
            self.port = port
            instances.append(self)

        def reconnect(self) -> None:
            """1.35.1-shaped: a closed Hello socket opens another blocking recv."""
            self.events.append("reconnect")
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = socket.create_connection((self.host, int(self.port)))
            self._sock.settimeout(None)

        def open(self, launch=True, **kwargs):
            seen["launch"] = launch
            seen["socket_timeout"] = self.socket_timeout
            seen["quit_on_close"] = self.quit_on_close
            seen["binary"] = self.binary
            seen["open_returned"] = False
            self._sock = socket.create_connection((self.host, int(self.port)))
            try:
                while True:
                    try:
                        self._sock.settimeout(None)
                        data = self._sock.recv(1)
                        if not data:
                            raise OSError("eof")
                    except OSError:
                        self.reconnect()
            finally:
                seen["open_returned"] = True

        def disconnect(self) -> None:
            self.events.append("disconnect")
            sock = getattr(self, "_sock", None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        def close(self) -> None:
            self.events.append("close")

        def quit_beamng(self) -> None:
            self.events.append("quit")

    class Fake135(FakeBNG):
        """1.35.1-shaped: constructor rejects socket_timeout; open() still blocks."""

        def __init__(self, host, port, **kwargs):
            if "socket_timeout" in kwargs or "gfx" in kwargs or "binary" in kwargs:
                raise TypeError("unexpected kw")
            super().__init__(host, port, **kwargs)

    def _bounded(elapsed: float, limit: float) -> None:
        assert elapsed >= limit * 0.8, elapsed
        assert elapsed < limit + 0.2, elapsed
        assert elapsed < block_s * 0.9, elapsed

    def _hello_stopped() -> None:
        assert seen.get("open_returned") is True
        alive = [t.name for t in threading.enumerate() if t.name == "gvd-hello" and t.is_alive()]
        assert not alive, alive

    mod = types.ModuleType("beamngpy")
    mod.BeamNGpy = FakeBNG
    old = sys.modules.get("beamngpy")
    sys.modules["beamngpy"] = mod
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    held_socks: list[socket.socket] = []
    stop_accept = threading.Event()

    def _accept_hold() -> None:
        """Accept so the backlog cannot stall connect(); hold so client recv blocks."""
        srv.settimeout(0.2)
        while not stop_accept.is_set():
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            held_socks.append(conn)

    acceptor = threading.Thread(target=_accept_hold, name="gvd-accept", daemon=True)
    acceptor.start()
    try:
        assert "phase=Hello REFUSE" not in inspect.getsource(TechSession.connect)
        assert float(load_tech_config().get("socket_timeout")) == TECH_SOCKET_TIMEOUT_S
        assert resolve_socket_timeout(load_tech_config()) == TECH_SOCKET_TIMEOUT_S

        os.environ["GVD_TECH_SOCKET_TIMEOUT"] = str(cap)
        assert resolve_socket_timeout(load_tech_config()) == cap
        buf = io.StringIO()
        started = time.monotonic()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy(
                    "127.0.0.1",
                    port,
                    home="/opt/techhome",
                    user=None,
                    launch=True,
                )
            except TechHelloTimeout as exc:
                assert "Hello timeout" in str(exc)
            else:
                raise AssertionError("Hello timeout should refuse")
        elapsed = time.monotonic() - started
        _bounded(elapsed, cap)
        out = buf.getvalue()
        assert seen["launch"] is False
        assert seen["socket_timeout"] == cap
        assert seen["quit_on_close"] is False
        assert seen["binary"] == "BeamNG.tech.exe"
        assert instances[-1].kwargs["socket_timeout"] == cap
        assert instances[-1].events == ["disconnect"]
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events
        _hello_stopped()
        assert f"phase=attach 127.0.0.1:{port} LISTENING launch=False" in out
        assert f"socket_timeout={cap:g}s" in out
        assert out.count("phase=Hello REFUSE") == 1
        assert "Not a missing vehicle" in out
        assert "no vehicle spawned" not in out
        assert "launch failed" not in out
        os.environ.pop("GVD_TECH_SOCKET_TIMEOUT", None)

        mod.BeamNGpy = Fake135
        buf = io.StringIO()
        started = time.monotonic()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy(
                    "127.0.0.1",
                    port,
                    home="/opt/techhome",
                    user=None,
                    launch=False,
                    socket_timeout=cap,
                )
            except TechHelloTimeout as exc:
                assert "Hello timeout" in str(exc)
            else:
                raise AssertionError("1.35.1 Hello should refuse")
        _bounded(time.monotonic() - started, cap)
        out = buf.getvalue()
        assert "socket_timeout" not in instances[-1].kwargs
        assert instances[-1].events == ["disconnect"]
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events
        _hello_stopped()
        assert out.count("phase=Hello REFUSE") == 1
        assert f"socket_timeout={cap:g}s" in out

        mod.BeamNGpy = FakeBNG
        with tempfile.TemporaryDirectory() as td:
            yaml_path = Path(td) / "tech.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "host: 127.0.0.1",
                        f"port: {port}",
                        "launch: true",
                        'beamngpy_pin: "1.36"',
                        "home: /opt/techhome",
                        f"socket_timeout: {cap}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_tech_config(yaml_path)
            assert float(cfg["socket_timeout"]) == cap
            assert resolve_socket_timeout(cfg) == cap
            buf = io.StringIO()
            started = time.monotonic()
            with _hold_env():
                with redirect_stdout(buf):
                    gate = tech_hold_gate(
                        cfg,
                        mod_present=True,
                        beamngpy_version="1.36.1",
                    )
            _bounded(time.monotonic() - started, cap)
        out = buf.getvalue()
        assert gate.hello == "timeout"
        assert gate.vehicle_spawned is False
        assert gate.will_launch is False
        assert not gate.ok and not gate.proceed
        assert "Hello timeout" in gate.note
        assert "no vehicle spawned" not in gate.note
        assert "vehicle=Hello-timeout" in gate.line
        assert "phase=wait-gate" in gate.line
        assert out.count("phase=Hello REFUSE") == 1
        assert "phase=wait-gate" in out
        assert "no vehicle spawned" not in out
        assert "disconnect" in instances[-1].events
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events
        _hello_stopped()

        buf = io.StringIO()
        started = time.monotonic()
        with redirect_stdout(buf):
            session = TechSession(
                {
                    "host": "127.0.0.1",
                    "port": port,
                    "launch": True,
                    "wait_vehicle_s": 0,
                    "socket_timeout": cap,
                    "home": "/opt/techhome",
                }
            )
            assert session.connect(explicit=True) is False
        _bounded(time.monotonic() - started, cap)
        out = buf.getvalue()
        assert out.count("phase=Hello REFUSE") == 1
        assert "Hello timeout" in session.note
        assert "no vehicle" not in session.note
        assert session.bng is None
        assert "disconnect" in instances[-1].events
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events
        _hello_stopped()

        buf = io.StringIO()
        with _hold_env():
            with redirect_stdout(buf):
                down = tech_hold_gate(
                    {
                        "host": "127.0.0.1",
                        "port": 9,
                        "launch": False,
                        "beamngpy_pin": "1.36",
                        "socket_timeout": 7.5,
                    },
                    port_up=False,
                    mod_present=True,
                    beamngpy_version="1.36.1",
                )
        text = buf.getvalue()
        assert "phase=attach 127.0.0.1:9 down launch=False socket_timeout=7.5s" in text
        assert "phase=Hello skipped" in text
        assert down.hello == "skipped"
        assert "no vehicle spawned" in down.note

        class FakeEmpty(FakeBNG):
            def open(self, launch=True, **kwargs):
                return None

            def get_current_vehicles(self):
                return {}

        mod.BeamNGpy = FakeEmpty
        with _hold_env():
            with redirect_stdout(io.StringIO()):
                empty = tech_hold_gate(
                    {
                        "host": "127.0.0.1",
                        "port": port,
                        "launch": False,
                        "beamngpy_pin": "1.36",
                        "home": "/opt/techhome",
                        "socket_timeout": cap,
                    },
                    mod_present=True,
                    beamngpy_version="1.36.1",
                )
        assert empty.hello == "ok"
        assert empty.vehicle_spawned is False
        assert "no vehicle spawned" in empty.note
        assert "Hello timeout" not in empty.note
        assert "vehicle=no" in empty.line
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events

        class FakeBoom(FakeBNG):
            def open(self, launch=True, **kwargs):
                raise RuntimeError("Hello version mismatch")

        mod.BeamNGpy = FakeBoom
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                open_tech_beamngpy(
                    "127.0.0.1",
                    port,
                    home=None,
                    user=None,
                    launch=False,
                    socket_timeout=cap,
                )
            except RuntimeError as exc:
                assert "version mismatch" in str(exc)
            else:
                raise AssertionError("Hello mismatch should disconnect and raise")
        boom_out = buf.getvalue()
        assert instances[-1].events == ["disconnect"]
        assert "close" not in instances[-1].events and "quit" not in instances[-1].events
        assert "phase=Hello REFUSE" not in boom_out

        class FakeLive(FakeBNG):
            def __init__(self, host, port, **kwargs):
                super().__init__(host, port, **kwargs)
                veh = types.SimpleNamespace(vid="veh", is_connected=lambda: True)
                self._veh = veh
                self.vehicles = types.SimpleNamespace(get_current=lambda: {"veh": veh})

            def open(self, launch=True, **kwargs):
                seen["opens"] = seen.get("opens", 0) + 1

            def get_current_vehicles(self):
                return {"veh": self._veh}

        mod.BeamNGpy = FakeLive
        seen["opens"] = 0
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus = root / "AppData" / "Local" / "BeamNG" / "BeamNG.tech" / "current" / "Documents" / "GVD"
            moved = root / "moved" / "Documents" / "GVD"
            moved.mkdir(parents=True)
            _plant_link(bus, lua_bus=str(moved), age_s=0.0)
            live_cfg = {
                "host": "127.0.0.1",
                "port": port,
                "launch": False,
                "beamngpy_pin": "1.36",
                "socket_timeout": cap,
                "home": "/opt/techhome",
                "wait_vehicle_s": 0,
            }
            with _hold_env(
                LOCALAPPDATA=str(root / "AppData" / "Local"),
                USERPROFILE=str(root / "Users" / "Name"),
                HOME=str(root),
                GVD_PRODUCT="tech",
                GVD_BEAMNG="1",
                GVD_DOCS_DIR=str(bus),
            ):
                with redirect_stdout(io.StringIO()) as hold_buf:
                    held = tech_hold_gate(
                        live_cfg,
                        mod_present=True,
                        beamngpy_version="1.36.1",
                        retain=True,
                    )
                assert held.proceed, held.note
                assert seen["opens"] == 1
                assert f"folder={moved}" in held.line
                assert "disconnect" not in instances[-1].events
                buf = io.StringIO()
                with redirect_stdout(buf):
                    session = TechSession(live_cfg)
                    assert session.connect(explicit=True) is True
                reuse_out = buf.getvalue()
                assert seen["opens"] == 1, seen["opens"]
                assert f"phase=Hello reuse 127.0.0.1:{port}" in reuse_out
                assert "phase=Hello REFUSE" not in reuse_out
                assert session.bng is instances[-1]
                session.close()
                assert instances[-1].events == ["disconnect"]
                assert "close" not in instances[-1].events and "quit" not in instances[-1].events
                assert "phase=Hello" in hold_buf.getvalue()
    finally:
        drop_tech_beamngpy()
        stop_accept.set()
        for held in held_socks:
            try:
                held.close()
            except OSError:
                pass
        os.environ.pop("GVD_TECH_SOCKET_TIMEOUT", None)
        srv.close()
        if old is None:
            sys.modules.pop("beamngpy", None)
        else:
            sys.modules["beamngpy"] = old


def check_tech_hold_gate() -> None:
    """Offline: bus match / freshness, one starter, pin, and Esc/q disconnect."""
    import inspect
    import os
    import socket
    import tempfile
    from pathlib import Path

    from python.runtime import paths
    from python.sensors.tech import (
        TECH_GFX,
        TECH_ROOT_EXE,
        TechHoldGate,
        TechSession,
        exit_if_tech_connect_failed,
        park_tech_beamngpy,
        beamngpy_launch_argv,
        beamngpy_pin_ok,
        human_one_starter,
        open_tech_beamngpy,
        real_launch_argv,
        release_tech_beamngpy,
        research_port_listening,
        resolve_tech_launch,
        tech_hold_gate,
        tech_key_status,
        tech_mod_present,
    )

    assert TECH_ROOT_EXE == "BeamNG.tech.exe"
    assert TECH_GFX == "dx11"
    assert human_one_starter(None) == "BeamNG.tech.exe -tcom -console -gfx dx11"
    starter = human_one_starter("/opt/techhome")
    assert starter == "/opt/techhome/BeamNG.tech.exe -tcom -console -gfx dx11"
    argv = beamngpy_launch_argv("/opt/techhome", 25252, None)
    assert argv[0] == "/opt/techhome/BeamNG.tech.exe"
    assert argv[1:6] == ["-nosteam", "-tcom", "-tport", "25252", "-console"]
    assert argv[argv.index("-gfx") + 1] == "dx11"
    assert "-userpath" not in argv
    assert "Bin64" not in " ".join(argv)
    assert "BeamNG.tech.x64.exe" not in " ".join(argv)
    with_user = beamngpy_launch_argv("/opt/techhome", 25252, "/opt/user")
    assert with_user[-2:] == ["-userpath", "/opt/user"]
    assert "tech_key" not in inspect.getsource(TechHoldGate.ok.fget)
    assert "tech_key" not in inspect.getsource(TechHoldGate.proceed.fget)
    assert inspect.getsource(TechHoldGate.proceed.fget).strip().endswith("return self.ok and self.pin_ok")
    assert resolve_tech_launch({"launch": True}, port_listening=True)[0] is False
    assert "double-start" in resolve_tech_launch({"launch": True}, port_listening=True)[1]
    assert resolve_tech_launch({"launch": False}, port_listening=True)[0] is False
    assert resolve_tech_launch({"launch": True}, port_listening=False)[0] is True
    assert resolve_tech_launch({"launch": False}, port_listening=False)[0] is False
    assert beamngpy_pin_ok("1.36.2", "1.36")
    assert beamngpy_pin_ok("1.35.4", "1.35")
    assert not beamngpy_pin_ok("1.35.0", "1.36")
    assert not beamngpy_pin_ok("1.26.0", "1.36")
    assert not beamngpy_pin_ok(None, "1.36")

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    live_port = srv.getsockname()[1]
    try:
        assert research_port_listening("127.0.0.1", live_port)
    finally:
        srv.close()
    assert not research_port_listening("127.0.0.1", live_port)

    class _Bng:
        def __init__(self, *, boom: bool = False) -> None:
            self.quit_on_close = True
            self.events: list[str] = []
            self.boom = boom

        def disconnect(self) -> None:
            self.events.append("disconnect")
            if self.boom:
                raise RuntimeError("socket dead")

        def close(self) -> None:
            self.events.append("close")

        def quit_beamng(self) -> None:
            self.events.append("quit")

    alive = _Bng()
    session = TechSession({"wait_vehicle_s": 0, "launch": True})
    session.bng = alive
    session.close()
    assert alive.events == ["disconnect"]
    assert alive.quit_on_close is False
    dead = _Bng(boom=True)
    session.bng = dead
    session.close()
    assert dead.events == ["disconnect"]
    assert "close" not in dead.events and "quit" not in dead.events
    orphan = _Bng()
    release_tech_beamngpy(orphan)
    assert orphan.events == ["disconnect"]
    close_src = inspect.getsource(TechSession.close)
    close_body = close_src.split('"""', 2)[-1]
    assert ".close(" not in close_body
    assert "quit_beamng" not in close_body
    assert "taskkill" not in close_body
    connect_src = inspect.getsource(TechSession.connect)
    assert "resolve_tech_launch" in connect_src
    assert "quit_on_close=False" in inspect.getsource(open_tech_beamngpy)
    open_src = inspect.getsource(open_tech_beamngpy)
    assert 'binary": TECH_ROOT_EXE' in open_src or "binary=TECH_ROOT_EXE" in open_src or '"binary": TECH_ROOT_EXE' in open_src
    assert "TECH_GFX" in open_src
    assert "beamngpy launch failed. argv:" in open_src

    bat = (ROOT / "play_gvd_tech.bat").read_text(encoding="utf-8")
    assert 'set "GVD_TECH_LAUNCH=0"' in bat
    assert 'set "GVD_TECH_LAUNCH=1"' not in bat
    assert 'start ""' not in bat
    assert r'BeamNG.tech.exe" -tcom -console -gfx dx11' in bat
    assert "BeamNG.tech.x64.exe" not in bat
    assert r"Bin64\BeamNG" not in bat
    assert "--tech-hold" in bat
    assert "quit BeamNG" not in bat
    req = (ROOT / "requirements-beamng.txt").read_text(encoding="utf-8")
    assert "beamngpy>=1.35,<1.37" in req
    assert "beamngpy>=1.26" not in req
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert rv.find("tech_hold_gate(") < rv.find("make_backend(")
    assert "tech_hold_gate(retain=True)" in rv
    assert rv.find("backend.open()") < rv.find("exit_if_tech_connect_failed(")
    assert "--tech-hold" in rv
    assert 'ord("q"), 27' in rv
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_dev = readme.split("### Dev install (BeamNG.tech)", 1)[1].split("**Tech hold prove**", 1)[0]
    assert "--tech-hold" in readme_dev
    assert "python python/run_vision.py --tech-hold" not in readme_dev
    assert "Soft Esc parked" in readme
    assert "Tech hold prove" in readme
    assert "quit_on_close" in readme
    assert "Unique-frame Hz" in readme and "not" in readme

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        la = root / "AppData" / "Local"
        bus = la.joinpath("BeamNG", "BeamNG.tech", "current", "Documents", "GVD")
        moved = root / "moved" / "Documents" / "GVD"
        moved.mkdir(parents=True)
        mod = la.joinpath("BeamNG", "BeamNG.tech", "current", "mods", "unpacked", "gvd", "lua", "ge", "extensions", "gvd")
        mod.mkdir(parents=True)
        (mod / "main.lua").write_text("-- gvd\n", encoding="utf-8")
        home = root / "tech-install"
        (home / "Bin64").mkdir(parents=True)
        (home / "Bin64" / "BeamNG.tech.x64.exe").write_bytes(b"")
        (home / "Bin64" / "tech.key").write_text("key", encoding="utf-8")
        assert tech_key_status(str(home)) is False
        (home / "tech.key").write_text("", encoding="utf-8")
        assert tech_key_status(str(home)) is False
        (home / "tech.key").write_text(" \n\t", encoding="utf-8")
        assert tech_key_status(str(home)) is False
        (home / "tech.key").write_text("key\n", encoding="utf-8")
        assert tech_key_status(str(home)) is True
        assert tech_key_status("") is None
        assert tech_key_status(None) is None
        cfg = {
            "host": "127.0.0.1",
            "port": 9,
            "launch": False,
            "beamngpy_pin": "1.36",
            "home": str(home),
        }
        with _hold_env(
            LOCALAPPDATA=str(la),
            USERPROFILE=str(root / "Users" / "Name"),
            HOME=str(root),
            GVD_PRODUCT="tech",
            GVD_BEAMNG="1",
            GVD_DOCS_DIR=str(bus),
            BNG_HOME=str(home),
        ):
            assert tech_mod_present()
            base = dict(port_up=True, vehicle_spawned=True, mod_present=True, beamngpy_version="1.36.1")
            gate = tech_hold_gate(cfg, **base)
            assert gate.link == "MISMATCH" and not gate.ok, gate.note
            assert not gate.lua_fresh and not gate.buses_same
            assert gate.lua_bus == ""
            assert "lua_bus=stale" in gate.line and "folder=--" in gate.line

            _plant_link(bus, lua_bus=str(moved), age_s=2.5)
            stale = tech_hold_gate(cfg, **base)
            assert not stale.lua_fresh and not stale.buses_same and stale.link == "MISMATCH"
            assert stale.lua_age_s is not None and stale.lua_age_s >= 1.0
            assert not stale.ok and not stale.proceed
            assert stale.lua_bus == ""
            assert "lua_bus=stale" in stale.line and f"folder={moved}" in stale.line

            _plant_link(bus, lua_bus="Documents/GVD", age_s=0.0)
            bad = tech_hold_gate(cfg, **base)
            assert not bad.lua_fresh and bad.link == "MISMATCH"
            assert "folder=Documents/GVD" in bad.line

            _plant_link(bus, lua_bus=str(moved), age_s=0.0)
            ok = tech_hold_gate(cfg, **base)
            assert ok.lua_fresh and ok.buses_same and ok.link == "ok", ok.note
            assert ok.ok and ok.proceed, ok.note
            assert paths.buses_same_folder(ok.python_bus, ok.lua_bus)
            assert "lua_bus=fresh" in ok.line and f"folder={moved}" in ok.line
            assert ok.lua_age_s is not None and ok.lua_age_s < 1.0
            assert ok.will_launch is False
            assert ok.tech_key is True
            assert ok.hello == "skipped"
            assert "phase=wait-gate" in ok.line
            assert "Hello-timeout" not in ok.line
            (home / "tech.key").write_text("", encoding="utf-8")
            unlocked = tech_hold_gate(cfg, **base)
            assert unlocked.tech_key is False
            assert unlocked.ok and unlocked.proceed, unlocked.note
            (home / "tech.key").write_text("key\n", encoding="utf-8")

            down = tech_hold_gate(cfg, port_up=False, vehicle_spawned=True, mod_present=True, beamngpy_version="1.36.1")
            assert not down.port_listening and not down.ok and down.will_launch is False
            # port down must not probe a vehicle, and must still build the gate.
            skipped = tech_hold_gate(cfg, port_up=False, beamngpy_version="1.36.1")
            assert skipped.vehicle_spawned is False and not skipped.ok

            launch_cfg = dict(cfg, launch=True)
            attached = tech_hold_gate(launch_cfg, **base)
            assert attached.will_launch is False and attached.proceed
            solo = tech_hold_gate(
                launch_cfg, port_up=False, vehicle_spawned=False, mod_present=True, beamngpy_version="1.36.1"
            )
            assert solo.will_launch is True and not solo.ok

            mismatch_pin = tech_hold_gate(cfg, port_up=True, vehicle_spawned=True, mod_present=True, beamngpy_version="1.35.2")
            assert mismatch_pin.ok and not mismatch_pin.pin_ok and not mismatch_pin.proceed

            no_mod = tech_hold_gate(cfg, port_up=True, vehicle_spawned=True, mod_present=False, beamngpy_version="1.36")
            assert not no_mod.mod_present and not no_mod.ok
            no_veh = tech_hold_gate(cfg, port_up=True, vehicle_spawned=False, mod_present=True, beamngpy_version="1.36")
            assert not no_veh.vehicle_spawned and not no_veh.ok

    _check_launch_fail_prints_argv(open_tech_beamngpy)
    _check_hello_timeout()

    # connect refuses a dead port instead of launching (launch stays false).
    with _hold_env():
        quiet = TechSession({"wait_vehicle_s": 0, "host": "127.0.0.1", "port": 1, "launch": False})
        assert quiet.connect(explicit=True) is False
        assert quiet.vehicle is None
        assert "not LISTENING" in quiet.note

    import io
    from contextlib import redirect_stdout

    class _Parked:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.port = 25252

        def disconnect(self) -> None:
            self.events.append("disconnect")

    parked = _Parked()
    park_tech_beamngpy(parked)

    class _Ok:
        connect_failed = False

    exit_if_tech_connect_failed(_Ok(), "beamngpy")

    class _Window:
        connect_failed = True
        session = type("S", (), {"note": "boom"})()

    exit_if_tech_connect_failed(_Window(), "window")
    assert parked.events == []

    class _Bad:
        connect_failed = True
        session = type("S", (), {"note": "beamngpy connect failed (boom)"})()

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            exit_if_tech_connect_failed(_Bad(), "beamngpy")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("post-hold connect fail should exit 1")
    assert "REFUSE:" in buf.getvalue()
    assert "Post-hold connect failed" in buf.getvalue()
    assert parked.events == ["disconnect"]
    assert "close" not in parked.events


def main() -> None:
    check_frame_convert()
    check_path_world()
    check_tech_yaml()
    check_env_overrides({})
    check_forbidden_attach()
    check_gps_not_forbidden()
    check_gps_geometry()
    check_ego_fb_null_safe()
    check_gps_poll_and_pin()
    check_pin_env_override()
    check_poll_mock()
    check_camera_clip_planes()
    check_beamngpy_open_passes_near_far()
    check_beamngpy_side_grab_half_rate()
    check_soft_esc_segment_timers()
    check_nvidia_smi_cache_and_honest_hz()
    check_auto_backend_not_tech_without_env()
    check_connect_without_beamngpy()
    check_tech_hold_gate()
    check_no_chrome()
    print("test_tech_session: OK")


def test_tech_session() -> None:
    main()


if __name__ == "__main__":
    main()
