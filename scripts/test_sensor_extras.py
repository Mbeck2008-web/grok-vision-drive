#!/usr/bin/env python3
"""Offline extras bus: IMU/GPS/LiDAR/radar/Foxglove. Planner stays vision-only."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import EgoFeedback, read_ego_feedback  # noqa: E402
from python.sensors.extras import (  # noqa: E402
    ExtraSensors,
    apply_sensor_env,
    load_sensors_config,
    parse_lidar_points,
    parse_radar_returns,
    world_xy_to_ll,
)
from python.sensors.tech import DEFAULT_REF_LAT, DEFAULT_REF_LON, EARTH_R_M, FORBIDDEN_SENSORS, TechSession  # noqa: E402
from python.viz.foxglove_bridge import FoxgloveBridge  # noqa: E402


def check_world_xy_to_ll() -> None:
    lat, lon = world_xy_to_ll(0.0, 0.0)
    assert abs(lat - DEFAULT_REF_LAT) < 1e-9 and abs(lon - DEFAULT_REF_LON) < 1e-9
    lat2, lon2 = world_xy_to_ll(0.0, EARTH_R_M * math.pi / 180.0)
    assert abs(lat2 - (DEFAULT_REF_LAT + 1.0)) < 1e-6, lat2
    lat3, lon3 = world_xy_to_ll(100.0, 0.0)
    assert lon3 > DEFAULT_REF_LON and abs(lat3 - DEFAULT_REF_LAT) < 1e-9


def check_sensors_yaml_defaults() -> None:
    cfg = load_sensors_config()
    assert cfg.get("drive_uses") == "vision"
    assert cfg.get("imu") is True
    assert cfg.get("gps") is True
    assert cfg.get("lidar") is False
    assert cfg.get("radar") is False
    assert cfg.get("lidar_lua") is False
    fox = cfg.get("foxglove") or {}
    assert fox.get("enabled") is False
    assert int(fox.get("port") or 0) == 8765


def check_env_overrides() -> None:
    old = {k: os.environ.get(k) for k in ("GVD_LIDAR", "GVD_RADAR", "GVD_FOXGLOVE", "GVD_FOXGLOVE_PORT")}
    try:
        os.environ["GVD_LIDAR"] = "1"
        os.environ["GVD_RADAR"] = "1"
        os.environ["GVD_FOXGLOVE"] = "1"
        os.environ["GVD_FOXGLOVE_PORT"] = "9001"
        out = apply_sensor_env({"drive_uses": "vision", "lidar": False, "radar": False})
        assert out["lidar"] is True and out["radar"] is True
        assert out["foxglove"]["enabled"] is True
        assert out["foxglove"]["port"] == 9001
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_parse_lidar_radar() -> None:
    pts = parse_lidar_points({"pointCloud": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert pts == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    pts2 = parse_lidar_points([{"x": 0, "y": 1, "z": 2}, [3, 4, 5]])
    assert pts2[0] == (0.0, 1.0, 2.0) and pts2[1] == (3.0, 4.0, 5.0)
    hits = parse_radar_returns({"returns": [{"range": 12.0, "azimuth": 0.1, "velocity": -2.0}]})
    assert hits and hits[0]["distance"] == 12.0 and hits[0]["az"] == 0.1


def check_ego_feedback_extras() -> None:
    docs = Path.home() / "Documents" / "GVD"
    docs.mkdir(parents=True, exist_ok=True)
    p = docs / "gvd_ego.json"
    p.write_text(
        '{"speed_mps":4.0,"steering_input":0.1,"throttle_input":0.2,"brake_input":0.0,'
        '"applied_seq":3,"applying":true,"gx":0.05,"gy":0.1,"gz":9.8,"yaw_rate":0.02,'
        '"pos":{"x":1.0,"y":2.0,"z":3.0}}',
        encoding="utf-8",
    )
    fb = read_ego_feedback()
    assert fb is not None
    assert fb.speed_mps == 4.0 and fb.applied_seq == 3
    assert fb.gx == 0.05 and fb.gy == 0.1 and abs((fb.gz or 0) - 9.8) < 1e-9
    assert fb.yaw_rate == 0.02
    assert fb.pos == (1.0, 2.0, 3.0)


def check_poll_lua_pose_gps() -> None:
    fb = EgoFeedback(gx=0.1, gy=0.0, gz=9.8, yaw_rate=0.01, pos=(0.0, 0.0, 0.5))
    bundle = ExtraSensors({"drive_uses": "vision", "imu": True, "gps": True, "lidar": False, "radar": False}).poll(
        ego_fb=fb
    )
    assert bundle.imu.source == "lua"
    assert bundle.gps.ok and bundle.gps.source == "lua_pose"
    assert abs((bundle.gps.lat or 0) - DEFAULT_REF_LAT) < 1e-6
    assert bundle.lidar.source == "missing" and bundle.lidar.note == "off"
    assert bundle.drive_uses == "vision"
    h = bundle.health()
    assert h["drive_uses"] == "vision" and h["lidar"] == "missing"


def check_foxglove_disabled_without_sdk() -> None:
    fox = FoxgloveBridge({"foxglove": {"enabled": False}})
    assert fox.ok is False and fox.note == "disabled"
    fox2 = FoxgloveBridge({"foxglove": {"enabled": True, "host": "127.0.0.1", "port": 8765}})
    # Either the SDK is missing (honest skip) or a server started — both are fine offline.
    assert fox2.enabled is True
    fox2.close()


def check_perception_stays_vision() -> None:
    src = (ROOT / "python" / "perception" / "pipeline.py").read_text(encoding="utf-8").lower()
    assert "lidar" not in src
    assert "radar" not in src
    assert "foxglove" not in src


def check_lidar_flag_no_valueerror() -> None:
    session = TechSession(
        {"sensors": {"lidar": True, "electrics": False, "damage": False, "gforces": False}, "wait_vehicle_s": 0}
    )
    session.vehicle = object()
    session.bng = object()
    attached = session.attach_vehicle_sensors()
    assert attached.get("lidar") in (True, False)
    assert "lidar" not in FORBIDDEN_SENSORS
    banned = TechSession({"sensors": {"ultrasonic": True}, "wait_vehicle_s": 0})
    banned.vehicle = object()
    try:
        banned.attach_vehicle_sensors()
        raise AssertionError("ultrasonic must still be refused")
    except ValueError as e:
        assert "ultrasonic" in str(e).lower()


def check_foxglove_reqs_not_in_retail_glob() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_release_zip", ROOT / "scripts" / "make_release_zip.py")
    assert spec and spec.loader
    rel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rel)
    assert "requirements-foxglove.txt" not in rel.INCLUDE_GLOBS
    assert "requirements-foxglove.txt" not in rel.INCLUDE_FILES


def main() -> None:
    check_world_xy_to_ll()
    check_sensors_yaml_defaults()
    check_env_overrides()
    check_parse_lidar_radar()
    check_ego_feedback_extras()
    check_poll_lua_pose_gps()
    check_foxglove_disabled_without_sdk()
    check_perception_stays_vision()
    check_lidar_flag_no_valueerror()
    check_foxglove_reqs_not_in_retail_glob()
    print("test_sensor_extras: OK")


if __name__ == "__main__":
    main()
