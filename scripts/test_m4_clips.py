#!/usr/bin/env python3
"""Offline M4 clip recorder checks (Spec verify + Critic fixes)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.data.record import RING_CAP_BYTES, ClipRecorder, choose_encoder, clips_dir


def main() -> None:
    # Encoder preference
    assert choose_encoder(prefer="qsv", allow_nvenc=False, available={"h264_qsv", "libx264", "h264_nvenc"}) == "h264_qsv"
    assert choose_encoder(prefer="qsv", allow_nvenc=False, available={"libx264", "h264_nvenc"}) == "libx264"
    # never default nvenc
    assert choose_encoder(prefer="qsv", allow_nvenc=False, available={"h264_nvenc"}) == "none"
    # --encode nvenc when QSV missing → nvenc wins (not silent libx264)
    assert choose_encoder(prefer="nvenc", allow_nvenc=True, available={"libx264", "h264_nvenc"}) == "h264_nvenc"
    assert choose_encoder(prefer="qsv", allow_nvenc=True, available={"libx264", "h264_nvenc"}) == "h264_nvenc"
    # QSV still wins over nvenc when present
    assert choose_encoder(prefer="nvenc", allow_nvenc=True, available={"h264_qsv", "h264_nvenc"}) == "h264_qsv"

    rec = ClipRecorder(dry_run=True, encode_prefer="cpu", max_bytes=5_000_000, pre_s=0.5, post_s=0.1, hz=10)
    fake = np.zeros((120, 160, 3), dtype=np.uint8)
    t0 = time.time()
    for i in range(20):
        ok = rec.push(fake, {"engaged": True, "planner": {"aeb": "off", "ttc_lead": 5}, "ego": {}}, now=t0 + i * 0.05)
        assert ok is True
    out = rec.flush("manual", now=t0 + 1.0)
    assert out is not None and out.is_dir(), out
    assert (out / "state.jsonl").is_file()
    assert (out / "meta.json").is_file()
    assert rec.last_clip_trigger == "manual"
    assert clips_dir().exists()

    # disengage + AEB edge/cooldown
    rec2 = ClipRecorder(dry_run=True, max_bytes=2_000_000)
    assert rec2.note_engage(True) is None
    assert rec2.note_engage(False) == "disengage"
    assert rec2.check_aeb({"aeb": "brake"}) == "aeb_brake"
    assert rec2.check_aeb({"aeb": "brake"}) is None  # latched — no flood
    assert rec2.check_aeb({"aeb": "off"}) is None
    assert rec2.check_aeb({"aeb": "brake"}) == "aeb_brake"  # re-edge
    assert rec2.check_aeb({"aeb": "off", "ttc_lead": 1.0}) == "near_miss_ttc"
    assert rec2.check_aeb({"aeb": "off", "ttc_lead": 1.0}) is None

    # ring cap: after pushes, ring_bytes <= max_bytes
    tiny = ClipRecorder(dry_run=True, max_bytes=80_000, pre_s=30, post_s=30)
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(200):
        tiny.push(big, {"engaged": False}, now=t0 + i * 0.01)
    assert tiny.ring_bytes <= tiny.max_bytes, (tiny.ring_bytes, tiny.max_bytes)

    assert RING_CAP_BYTES == 2 * 1024 * 1024 * 1024
    print("test_m4_clips: OK")


if __name__ == "__main__":
    main()
