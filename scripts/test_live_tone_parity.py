#!/usr/bin/env python3
"""Live camera frames are the frames the recorder saves.

The recorder stores each colour buffer as the sensor returned it
(``colour_to_bgr(..., tone=False)``). The scene net is trained on those
pictures, so the live grab must hand the model, the lane fit, the strip,
the CAMS tiles, and the inset the same bytes. A road sample on either side
of the old washed-road gate (188) must not flip frames between a raw and a
darkened picture, and the companion flash check compares raw with raw.

``GVD_LIVE_TONE=1`` is the only way to get the old washed-road curve on
the live frames. It is off by default.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.data.strip_writer import compose_strip  # noqa: E402
from python.sensors.cameras import (  # noqa: E402
    BeamNGPyBackend,
    _reading_colour,
    colour_to_bgr,
    companion_frame_is_flash,
    recover_viewport_tone,
)
from scripts.collect_scene_lanes import _read_camera  # noqa: E402

W, H = 640, 480
TONE_ENV = "GVD_LIVE_TONE"


def _road_bgr(green: int) -> np.ndarray:
    """Warm low-saturation road, luma = green + 0.654, one white stripe."""
    img = np.full((H, W, 3), (green - 10, green, green + 6), dtype=np.uint8)
    img[300:440, 400:412] = (250, 250, 250)
    return img


def _rgba(bgr: np.ndarray) -> bytes:
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = bgr[:, :, ::-1]
    rgba[:, :, 3] = 255
    return rgba.tobytes()


def _backend() -> BeamNGPyBackend:
    be = BeamNGPyBackend.__new__(BeamNGPyBackend)
    be.long_side = 640
    be._cache_frames = {}
    be._cache_ts = {}
    be._frame_sig = {}
    be._late_unique = []
    return be


def _live(be: BeamNGPyBackend, cid: str, raw: bytes) -> tuple[dict, str]:
    """One sensor reply through the live grab: _reading_colour -> _publish_read."""
    bgr = _reading_colour({"colour": raw}, (W, H))
    frames: dict = {}
    be._publish_read(cid, bgr, 0.0, frames, {}, {}, [])
    return frames, str(be._last_status)


def _recorder(raw: bytes) -> np.ndarray:
    """The same sensor reply through the recorder read (tone=False)."""
    cam = types.SimpleNamespace(resolution=(W, H), stream_raw=lambda: {"colour": raw})
    image, reason, _ = _read_camera("main", cam, {})
    assert image is not None, reason
    return image


def _clear_flag() -> None:
    os.environ.pop(TONE_ENV, None)


def check_live_matches_recorder() -> None:
    """A washed road (sample 189, over the old gate) lands raw on every live reader."""
    _clear_flag()
    raw_bgr = _road_bgr(188)
    raw = _rgba(raw_bgr)
    saved = _recorder(raw)
    assert np.array_equal(saved, raw_bgr)
    # The old curve would rewrite this frame; parity has to be real here.
    assert not np.array_equal(recover_viewport_tone(saved), saved)
    be = _backend()
    for cid in ("main", "wide", "repeatL", "rear"):
        frames, status = _live(be, cid, raw)
        assert status == "ok", (cid, status)
        live = frames[cid]
        assert live.dtype == saved.dtype and live.shape == saved.shape
        assert np.array_equal(live, saved), (cid, float(live.mean()), float(saved.mean()))
        if cid == "main":
            assert np.array_equal(frames["cam_main"], saved)
        # Live scene strip (run_vision compose_strip(bundle.frames)) vs recorder strip.
        assert np.array_equal(compose_strip(frames), compose_strip({cid: saved})), cid
    print("  live == recorder: main, wide, repeatL, rear, and the composed strip")


def check_straddle_sequence_is_raw_and_kept() -> None:
    """Road sample 187/189 frame to frame: every frame raw, none dropped as a flash."""
    _clear_flag()
    be = _backend()
    seq = [188, 186, 188, 188, 186, 186, 188, 186]  # luma 188.65 / 186.65
    mixed, dropped, curved_ref = [], [], []
    for i, green in enumerate(seq):
        raw_bgr = _road_bgr(green)
        raw = _rgba(raw_bgr)
        saved = _recorder(raw)
        for cid in ("main", "repeatR"):
            frames, status = _live(be, cid, raw)
            if status == "flash_filtered" or cid not in frames:
                dropped.append((i, cid, green))
                continue
            if not np.array_equal(frames[cid], saved):
                mixed.append((i, cid, green, round(float(frames[cid].mean()) - float(saved.mean()), 1)))
            # Flash reference is the raw buffer, not a curved copy.
            if not np.array_equal(be._cache_frames[cid], saved):
                curved_ref.append((i, cid, green))
    assert not mixed and not dropped and not curved_ref, (
        f"corrected frames: {mixed}; dropped by the flash filter: {dropped}; "
        f"flash reference not raw: {curved_ref}"
    )
    print(f"  {len(seq)} straddling frames x 2 cams: all raw, none dropped")


def check_warm_washed_companion_keeps_landing() -> None:
    """A warm washed road (sample ~210, red over blue) is not the blue flash.

    The old path held the curved frame as the flash reference. The curve
    widens red-over-blue, so the next raw buffer of the same road read as
    brighter and bluer by 18+ and was dropped, and the reference never moved
    again: that companion stopped updating after its first frame.
    """
    _clear_flag()
    be = _backend()
    raw_bgr = np.full((H, W, 3), (196, 210, 220), dtype=np.uint8)
    raw_bgr[300:440, 400:412] = (250, 250, 250)
    raw = _rgba(raw_bgr)
    statuses, landed = [], []
    for _ in range(4):
        frames, status = _live(be, "repeatR", raw)
        statuses.append(status)
        landed.append(status == "ok" and np.array_equal(frames["repeatR"], raw_bgr))
    assert statuses == ["ok"] * 4, statuses
    assert all(landed), landed
    print("  warm washed companion: 4 of 4 frames land, raw")


def check_flash_still_filtered_raw_vs_raw() -> None:
    """The one bright-blue companion frame is still not stored."""
    _clear_flag()
    be = _backend()
    settled = np.full((H, W, 3), (108, 102, 98), dtype=np.uint8)
    flash = np.full((H, W, 3), (210, 140, 70), dtype=np.uint8)
    assert companion_frame_is_flash(flash, settled) is True
    frames, status = _live(be, "wide", _rgba(settled))
    assert status == "ok" and np.array_equal(frames["wide"], settled)
    frames, status = _live(be, "wide", _rgba(flash))
    assert status == "flash_filtered" and "wide" not in frames
    assert np.array_equal(be._cache_frames["wide"], settled)
    print("  bright-blue companion flash still filtered (raw vs raw)")


def check_tone_flag_off_by_default() -> None:
    """GVD_LIVE_TONE is opt-in: unset / 0 -> raw; 1 -> old curve on the stored frame."""
    from python.sensors.cameras import live_tone, live_tone_enabled

    raw_bgr = _road_bgr(188)
    toned = recover_viewport_tone(raw_bgr)
    assert not np.array_equal(toned, raw_bgr)
    _clear_flag()
    assert live_tone_enabled() is False
    assert live_tone(raw_bgr) is raw_bgr
    for off in ("", "0", "false", "no"):
        os.environ[TONE_ENV] = off
        assert live_tone_enabled() is False, off
    try:
        os.environ[TONE_ENV] = "1"
        assert live_tone_enabled() is True
        assert np.array_equal(live_tone(raw_bgr), toned)
        be = _backend()
        frames, status = _live(be, "repeatL", _rgba(raw_bgr))
        assert status == "ok" and np.array_equal(frames["repeatL"], toned)
        # Opted in, the flash check still holds the raw buffer.
        assert np.array_equal(be._cache_frames["repeatL"], raw_bgr)
        frames, status = _live(be, "repeatL", _rgba(raw_bgr))
        assert status == "ok", status
    finally:
        _clear_flag()
    print("  GVD_LIVE_TONE off by default; =1 tones the stored frame, flash ref stays raw")


def main() -> None:
    check_live_matches_recorder()
    check_straddle_sequence_is_raw_and_kept()
    check_warm_washed_companion_keeps_landing()
    check_flash_still_filtered_raw_vs_raw()
    check_tone_flag_off_by_default()
    print("test_live_tone_parity: OK")


if __name__ == "__main__":
    main()
