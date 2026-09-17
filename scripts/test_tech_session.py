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
    DEFAULT_FAR_M,
    DEFAULT_NEAR_M,
    DEFAULT_UPDATE_S,
    REAR_CAM_IDS,
    REAR_GRAB_DIV,
    SIDE_CAM_IDS,
    SIDE_GRAB_DIV,
    beamng_camera_sensor_kwargs,
    camera_clip_planes,
    camera_grab_div,
    camera_update_s,
    far_hitch_ladder,
    iter_clip_attach_attempts,
    load_camera_config,
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
    text = (ROOT / "config" / "tech.yaml").read_text(encoding="utf-8").lower()
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
    assert int(hitch.get("rear_grab_div")) == REAR_GRAB_DIV
    for cid in ("pillarL", "pillarR", "repeatL", "repeatR"):
        assert float(by_id[cid]["far_m"]) == 100, cid
        assert abs(float(by_id[cid]["requested_update_time"]) - 0.13) < 1e-9, cid
        assert camera_grab_div(cid, hitch) == 2, cid
    assert float(by_id["rear"]["far_m"]) == 100
    assert abs(float(by_id["rear"]["requested_update_time"]) - 0.267) < 1e-9
    assert camera_grab_div("rear", hitch) == 4
    for cid in ("narrow", "main", "wide"):
        assert abs(float(by_id[cid]["requested_update_time"]) - 0.067) < 1e-9, cid
        assert camera_grab_div(cid, hitch) == 1, cid
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
        "pillarL": 0.13,
        "pillarR": 0.13,
        "repeatL": 0.13,
        "repeatR": 0.13,
        "rear": 0.267,
    }
    assert DEFAULT_FAR_M == want_far
    assert DEFAULT_UPDATE_S == want_rate
    assert want_far["narrow"] > want_far["main"]
    assert want_far["main"] != 800.0
    assert want_far["narrow"] != want_far["main"]
    for spec in cfg.get("cameras") or []:
        cid = spec["id"]
        near, far = camera_clip_planes(spec, defaults=defaults)
        assert abs(near - 0.05) < 1e-9, cid
        assert far == want_far[cid], (cid, far)
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
    assert abs(camera_update_s({"id": "pillarL"}) - 0.13) < 1e-9
    assert abs(camera_update_s({"id": "rear"}) - 0.267) < 1e-9
    # forward 100 m (BeamNGpy default) is clamped to the role far — not kept
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
    # near_far_planes pair still clamped to the role band
    assert camera_clip_planes({"id": "main", "far_m": 300, "near_far_planes": [0.05, 600]}) == (0.05, 300.0)
    assert camera_clip_planes({"id": "narrow", "far_m": 2000})[1] == 800.0

    assert far_hitch_ladder("narrow", 800.0) == (800.0,)
    assert far_hitch_ladder("main", 300.0) == (300.0,)
    assert far_hitch_ladder("wide", 300.0) == (300.0,)
    assert far_hitch_ladder("pillarL", 100.0) == (100.0,)
    assert far_hitch_ladder("rear", 100.0) == (100.0,)
    assert far_hitch_ladder("rear", 150.0) == (100.0,)
    assert far_hitch_ladder("main", 800.0) == (300.0,)
    assert far_hitch_ladder("narrow", 100.0) == (800.0,)

    assert camera_grab_div("narrow") == 1
    assert camera_grab_div("pillarL") == 2
    assert camera_grab_div("rear") == 4
    assert camera_grab_div("rear", {"rear_grab_div": 4, "side_grab_div": 2}) == 4

    narrow_tries = iter_clip_attach_attempts({"id": "narrow", "far_m": 800, "requested_update_time": 0.067})
    assert narrow_tries == [(0.05, 800.0, 0.067)]
    main_tries = iter_clip_attach_attempts({"id": "main", "far_m": 300, "requested_update_time": 0.067})
    assert main_tries == [(0.05, 300.0, 0.067)]
    wide_tries = iter_clip_attach_attempts({"id": "wide", "far_m": 300})
    assert wide_tries == [(0.05, 300.0, 0.067)]
    side = iter_clip_attach_attempts({"id": "pillarL", "far_m": 100, "requested_update_time": 0.13}, update_s=0.067)
    assert side == [(0.05, 100.0, 0.13)]
    # global 0.067 cannot clobber side half-rate or rear ÷4
    side_default = iter_clip_attach_attempts({"id": "pillarL"}, update_s=0.067)
    assert side_default == [(0.05, 100.0, 0.13)]
    rear_default = iter_clip_attach_attempts({"id": "rear"}, update_s=0.067)
    assert rear_default == [(0.05, 100.0, 0.267)]

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
        streaming=True,
        rgb_only=True,
    )
    assert kw["near_far_planes"] == (0.05, 800.0)
    assert kw["requested_update_time"] == 0.067
    assert kw["is_render_depth"] is False
    assert kw["is_render_annotations"] is False
    assert kw["resolution"] == (640, 480)
    assert "near_far_planes" in kw

    src = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    assert "near_far_planes" in src
    assert "beamng_camera_sensor_kwargs" in src
    assert "Camera(f\"gvd_{cid}\", bng, vehicle, **kwargs)" in src
    assert "requested_update_time" in src
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "BeamNGpy #199" in readme or "BeamNGpy/issues/199" in readme
    assert "narrow > main" in readme
    assert "0.267" in readme


def check_beamngpy_open_passes_near_far() -> None:
    """BeamNGPyBackend.open() constructs Camera(..., near_far_planes=(0.05, far_m)) per id."""
    import sys
    import types

    from python.sensors.cameras import BeamNGPyBackend

    captured: list[tuple[str, dict]] = []

    class FakeCamera:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            captured.append((name, kwargs))

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
        assert "far_m=800@update_s=0.067" in log
        assert "hitch steps rear:" in log
        assert "far_m=100@update_s=0.267" in log
        assert "hitch steps pillarL:" in log
        assert "not resolution" in log
        assert "side_grab_div=2" in log
        assert "rear_grab_div=4" in log
        assert "depth/semantic OFF" in log
        names = [n for n, _ in captured]
        assert names == [f"gvd_{c}" for c in CAM_IDS], names
        by = {n: kw for n, kw in captured}
        assert by["gvd_narrow"]["near_far_planes"] == (0.05, 800.0)
        assert by["gvd_main"]["near_far_planes"] == (0.05, 300.0)
        assert by["gvd_wide"]["near_far_planes"] == (0.05, 300.0)
        assert by["gvd_narrow"]["requested_update_time"] == 0.067
        assert by["gvd_main"]["requested_update_time"] == 0.067
        assert by["gvd_wide"]["requested_update_time"] == 0.067
        for cid in SIDE_CAM_IDS:
            assert by[f"gvd_{cid}"]["near_far_planes"] == (0.05, 100.0), cid
            assert abs(by[f"gvd_{cid}"]["requested_update_time"] - 0.13) < 1e-9, cid
            assert by[f"gvd_{cid}"]["is_render_depth"] is False
            assert by[f"gvd_{cid}"]["resolution"][0] >= 1
        for cid in REAR_CAM_IDS:
            assert by[f"gvd_{cid}"]["near_far_planes"] == (0.05, 100.0), cid
            assert abs(by[f"gvd_{cid}"]["requested_update_time"] - 0.267) < 1e-9, cid
            assert by[f"gvd_{cid}"]["is_render_depth"] is False
        assert by["gvd_narrow"]["is_render_depth"] is False
        assert by["gvd_narrow"]["resolution"] == (640, 480)
        assert be._clip_planes["narrow"] == (0.05, 800.0)
        assert be._clip_planes["main"] == (0.05, 300.0)
        assert abs(be._update_s["rear"] - 0.267) < 1e-9
        assert be._side_grab_div == 2
        assert be._rear_grab_div == 4
        hitch_by = {cid: (near, far, rate) for cid, near, far, rate in be._hitch_steps}
        assert hitch_by["narrow"] == (0.05, 800.0, 0.067)
        assert hitch_by["main"] == (0.05, 300.0, 0.067)
        assert hitch_by["wide"] == (0.05, 300.0, 0.067)
        assert hitch_by["pillarL"][1] == 100.0 and abs(hitch_by["pillarL"][2] - 0.13) < 1e-9
        assert hitch_by["rear"][1] == 100.0 and abs(hitch_by["rear"][2] - 0.267) < 1e-9
        # always explicit near_far_planes; forward never 100; never both-800
        for name, kw in captured:
            assert "near_far_planes" in kw, name
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
    """Sides poll every 2nd grab; rear every 4th; forward every grab. No resolution change."""
    import sys
    import types

    import numpy as np

    from python.sensors.cameras import BeamNGPyBackend, CamHealth, REAR_CAM_IDS, SIDE_CAM_IDS

    polls: dict[str, int] = {}

    class FakeCamera:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.is_streaming = False
            polls[name] = 0

        def poll(self):
            polls[self.name] += 1
            return {"colour": np.zeros((8, 8, 3), dtype=np.uint8)}

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
        be.open()
        n = 8
        last = None
        for _ in range(n):
            last = be.grab()
        assert last is not None
        assert last.health["main"] == CamHealth.OK
        assert last.health["narrow"] == CamHealth.OK
        for cid in SIDE_CAM_IDS | REAR_CAM_IDS:
            assert last.health[cid] == CamHealth.OK, cid
            assert f"gvd_{cid}" in last.frames or cid in last.frames
        assert polls["gvd_narrow"] == n
        assert polls["gvd_main"] == n
        assert polls["gvd_wide"] == n
        for cid in SIDE_CAM_IDS:
            assert polls[f"gvd_{cid}"] == n // 2, (cid, polls[f"gvd_{cid}"])
        for cid in REAR_CAM_IDS:
            assert polls[f"gvd_{cid}"] == n // 4, (cid, polls[f"gvd_{cid}"])
        assert be._clip_planes["narrow"][1] > be._clip_planes["main"][1]
        assert "rear_div=4" in last.note
        assert "side_div=2" in last.note
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


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
    check_auto_backend_not_tech_without_env()
    check_connect_without_beamngpy()
    check_no_chrome()
    print("test_tech_session: OK")


if __name__ == "__main__":
    main()
