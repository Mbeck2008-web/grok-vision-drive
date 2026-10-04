#!/usr/bin/env python3
"""Companion colour must not SendAdHocRequestCamera.

That request renders on the game view and steps exposure for one frame
(bright blue shadows). The supervisor tiles do not show that flash.
Companions are read with stream_raw, on the same hitch as before.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.sensors.cameras import (  # noqa: E402
    CAM_IDS,
    COMPANION_OFFSCREEN_UPDATE_S,
    ON_DEMAND_UPDATE_S,
    BeamNGPyBackend,
    CamHealth,
    beamng_camera_sensor_kwargs,
    camera_grab_due,
    load_camera_config,
)


def main() -> None:
    off = beamng_camera_sensor_kwargs(
        pos=(0.0, 0.0, 1.2),
        direction=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_v=50.0,
        resolution=(64, 48),
        update_s=ON_DEMAND_UPDATE_S,
        near_m=0.05,
        far_m=100.0,
        shmem=True,
        streaming=True,
        rgb_only=True,
    )
    assert off["requested_update_time"] == COMPANION_OFFSCREEN_UPDATE_S
    assert off["requested_update_time"] > 0
    assert off["is_streaming"] is True

    sent: list[str] = []
    polled: list[str] = []
    streamed: list[str] = []

    class Cam:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.is_streaming = True
            self.update_priority = float(kwargs.get("update_priority", 0.0))
            self.resolution = kwargs.get("resolution", (8, 8))

        def get_update_priority(self):
            return self.update_priority

        def set_update_priority(self, p):
            self.update_priority = float(p)

        def set_max_pending_requests(self, n):
            return None

        def stream_raw(self):
            streamed.append(self.name)
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[0, 0, 0] = 40
            img[0, 0, 2] = 30
            return {"colour": img}

        def poll(self):
            polled.append(self.name)
            raise AssertionError("companion must not PollCamera")

        def send_ad_hoc_poll_request(self):
            sent.append(self.name)
            raise AssertionError("SendAdHocRequestCamera flashes the game view")

        def is_ad_hoc_poll_request_ready(self, request_id):
            raise AssertionError("ad-hoc ready check is not the companion read")

        def collect_ad_hoc_poll_request(self, request_id):
            raise AssertionError("ad-hoc collect is not the companion read")

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
        assert set(be._sensors) == set(CAM_IDS)
        for cid in CAM_IDS:
            if cid == "main":
                assert be._sensors[cid].kwargs["requested_update_time"] == 0.067
            else:
                assert be._sensors[cid].kwargs["requested_update_time"] == COMPANION_OFFSCREEN_UPDATE_S
        hitch = load_camera_config().get("hitch") or {}
        seen: set[str] = set()
        for i in range(16):
            bundle = be.grab()
            due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, i, hitch)]
            assert len(due) <= 1, (i, due)
            for cid in due:
                assert bundle.health[cid] == CamHealth.OK, (i, cid)
                assert int(bundle.frames[cid].max()) > 0
                seen.add(cid)
            assert bundle.health["main"] == CamHealth.OK
        assert seen == {cid for cid in CAM_IDS if cid != "main"}, seen
        assert sent == []
        assert polled == []
        assert streamed, "stream_raw never ran"
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
    src = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    grab = src.split("def grab(self)", 1)[1].split("\n    def ", 1)[0]
    assert "_kick_adhoc" not in grab
    assert "send_ad_hoc_poll_request" not in grab
    print("test_companion_request: OK")


if __name__ == "__main__":
    main()
