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
    check_gps_poll_and_pin()
    check_pin_env_override()
    check_poll_mock()
    check_auto_backend_not_tech_without_env()
    check_connect_without_beamngpy()
    check_no_chrome()
    print("test_tech_session: OK")


if __name__ == "__main__":
    main()
