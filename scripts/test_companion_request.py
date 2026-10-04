#!/usr/bin/env python3
"""Companion colour must not SendAdHocRequestCamera.

That request renders on the game view and steps exposure for one frame
(bright blue shadows). The supervisor tiles do not show that flash.
Companions are read with stream_raw, on the same tick as main.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.sensors.cameras import (  # noqa: E402
    _reading_colour,
    CAM_IDS,
    COMPANION_OFFSCREEN_UPDATE_S,
    ON_DEMAND_UPDATE_S,
    BeamNGPyBackend,
    CamHealth,
    beamng_camera_sensor_kwargs,
    camera_grab_due,
    colour_to_bgr,
    companion_frame_is_flash,
    recover_viewport_tone,
    load_camera_config,
)


def _road(color: tuple[int, int, int]) -> np.ndarray:
    img = np.full((48, 64, 3), color, dtype=np.uint8)
    img[30:46, 28:36] = (245, 245, 245)
    return img


def _publish(be: BeamNGPyBackend, cid: str, frame: np.ndarray) -> dict:
    frames: dict = {}
    be._publish_read(cid, frame, 0.0, frames, {}, {}, [])
    return frames


def _check_flash_frame_does_not_land() -> None:
    # Settled gray-blue road, then the one bad frame: brighter and bluer together.
    settled = _road((108, 102, 98))
    flash = _road((210, 140, 70))
    assert companion_frame_is_flash(flash, settled) is True
    # A channel swap of the settled road is not that frame. Gray stays gray.
    gray = np.full((48, 64, 3), 140, dtype=np.uint8)
    swapped = colour_to_bgr(gray)
    assert swapped is not None
    assert abs(int(swapped[20, 20, 0]) - int(swapped[20, 20, 2])) <= 1
    assert companion_frame_is_flash(settled[:, :, ::-1].copy(), settled) is False
    # Brighter but still gray is a real lighting change, not the blue flash.
    brighter = _road((140, 138, 136))
    assert companion_frame_is_flash(brighter, settled) is False
    be = BeamNGPyBackend.__new__(BeamNGPyBackend)
    be.long_side = 640
    be._cache_frames = {"wide": settled.copy()}
    be._cache_ts = {"wide": 0.0}
    be._frame_sig = {}
    landed = _publish(be, "wide", flash)
    assert "wide" not in landed
    assert np.array_equal(be._cache_frames["wide"], settled)
    quiet = _road((112, 106, 102))
    landed2 = _publish(be, "wide", quiet)
    assert np.array_equal(landed2["wide"], quiet)
    # Main is not this gate.
    be._cache_frames["main"] = settled.copy()
    be._cache_ts["main"] = 0.0
    main_landed = _publish(be, "main", flash)
    assert np.array_equal(main_landed["main"], flash)


def _check_washed_companion_is_toned_and_stored() -> None:
    """Flash is the buffer before the curve. A washed road is not that flash.

    Asphalt (228, 220, 210) against a settled road is under the blue bar.
    After the curve the blue excess clears the bar, and the old gate dropped
    the companion. The frame that lands is the tone-matched road.
    """
    settled = _road((108, 102, 98))
    washed = _road((228, 220, 210))
    rgb = np.ascontiguousarray(washed[:, :, ::-1])
    raw = colour_to_bgr(rgb, tone=False)
    assert raw is not None
    assert tuple(int(v) for v in raw[10, 10]) == (228, 220, 210)
    assert companion_frame_is_flash(raw, settled) is False
    toned = recover_viewport_tone(raw)
    assert tuple(int(v) for v in toned[10, 10]) == (163, 141, 117)
    assert companion_frame_is_flash(toned, settled) is True
    read = _reading_colour({"colour": rgb}, None)
    assert read is not None and np.array_equal(read, raw)
    be = BeamNGPyBackend.__new__(BeamNGPyBackend)
    be.long_side = 640
    be._cache_frames = {"wide": settled.copy(), "main": settled.copy()}
    be._cache_ts = {"wide": 0.0, "main": 0.0}
    be._frame_sig = {}
    wide = _publish(be, "wide", read)
    assert np.array_equal(wide["wide"], toned)
    assert tuple(int(v) for v in be._cache_frames["wide"][10, 10]) == (163, 141, 117)
    main = _publish(be, "main", read)
    assert np.array_equal(main["main"], toned)


def main() -> None:
    _check_flash_frame_does_not_land()
    _check_washed_companion_is_toned_and_stored()
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
    assert COMPANION_OFFSCREEN_UPDATE_S == 0.067
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
        bundle = be.grab()
        due = [cid for cid in CAM_IDS if cid != "main" and camera_grab_due(cid, 0, hitch)]
        assert set(due) == {cid for cid in CAM_IDS if cid != "main"}, due
        stamps = []
        for cid in CAM_IDS:
            assert bundle.health[cid] == CamHealth.OK, cid
            assert int(bundle.frames[cid].max()) > 0
            stamps.append(float(bundle.timestamps[cid]))
        assert max(stamps) - min(stamps) < 1e-6
        assert "same_tick=8" in bundle.note
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
