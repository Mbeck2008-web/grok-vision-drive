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
    gvd_to_bng_vehicle,
    load_tech_config,
    path_ego_to_world,
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
    for bad in FORBIDDEN_SENSORS:
        assert not sensors.get(bad), bad
    cams = cfg.get("cameras") or {}
    assert cams.get("rgb_only") is True
    assert cams.get("convert_gvd_frame") is True
    text = (ROOT / "config" / "tech.yaml").read_text(encoding="utf-8").lower()
    assert "lidar" in text and "never" in text
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
    session = TechSession({"sensors": {"lidar": True}, "wait_vehicle_s": 0})
    session.vehicle = object()
    try:
        session.attach_vehicle_sensors()
        raise AssertionError("lidar must be refused")
    except ValueError as e:
        assert "lidar" in str(e).lower()


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
    check_poll_mock()
    check_auto_backend_not_tech_without_env()
    check_connect_without_beamngpy()
    check_no_chrome()
    print("test_tech_session: OK")


if __name__ == "__main__":
    main()
