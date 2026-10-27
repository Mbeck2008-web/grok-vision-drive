#!/usr/bin/env python3
"""Each camera frame carries its real capture window, and each tick its timing line.

``_poll_cameras`` stamps every camera with ``time.monotonic()`` right before
and right after its BeamNG call (``mono_before`` / ``mono_after``), keeps
``time.time()`` as its own ``wall`` field, and records ``sensor_t`` when
BeamNG returns one. ``call_s`` is the call's duration on ``time.perf_counter()``
(the monotonic pair can read equal on Windows, whose monotonic clock ticks
every 15.6 ms). The writer stores all five per frame in state.jsonl.

``TickLog`` (ticks.jsonl) splits a tick into its timed parts and keeps the
sim clock next to wall time. These checks cover the bookkeeping only; they
say nothing about why a real tick was slow.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect.tick_log import TICKS_FILE, TickLog, slow_cause  # noqa: E402
from python.data.raw_frames import RawFrameWriter  # noqa: E402


class _Cam:
    def __init__(self, sensor_t: float | None, delay_s: float = 0.002) -> None:
        self.sensor_t = sensor_t
        self.delay_s = delay_s
        self.resolution = (8, 6)
        self.calls = 0

    def stream_raw(self) -> object:
        self.calls += 1
        time.sleep(self.delay_s)
        image = np.full((6, 8, 3), 40 + self.calls % 50, dtype=np.uint8)
        raw: dict = {"colour": image}
        if self.sensor_t is not None:
            raw["timestamp"] = self.sensor_t + 0.1 * self.calls
        return raw


class _PollCam(_Cam):
    stream_raw = None  # type: ignore[assignment]

    def poll(self) -> object:
        return _Cam.stream_raw(self)


def check_windows_ordered_and_monotonic() -> None:
    res = time.get_clock_info("monotonic").resolution
    cams = {"main": _Cam(sensor_t=5.0), "wide": _PollCam(sensor_t=None), "rear": _Cam(sensor_t=None)}
    ticks = []
    for _ in range(6):
        frames, stamps, reasons = collect._poll_cameras(cams)
        assert reasons == {}, reasons
        assert set(frames) == set(cams)
        ticks.append(stamps)
        time.sleep(0.001)
    prev_after = None
    for stamps in ticks:
        for cid in cams:  # poll order
            st = stamps[cid]
            before, after = st["mono_before"], st["mono_after"]
            assert isinstance(before, float) and isinstance(after, float), st
            assert before <= after, st
            call_s = st["call_s"]
            assert isinstance(call_s, float), st
            assert call_s >= cams[cid].delay_s * 0.5, st  # the call's own duration, precise clock
            # The perf window sits inside the monotonic one; a coarse monotonic clock can lose
            # up to one resolution step of it.
            assert after - before >= call_s - res - 1e-6, (st, res)
            if prev_after is not None:
                assert prev_after <= before, (prev_after, st)  # monotonic across cameras and ticks
            prev_after = after
            assert isinstance(st["wall"], float) and abs(st["wall"] - time.time()) < 60.0, st
        assert stamps["main"]["sensor_t"] is not None
        assert stamps["wide"]["sensor_t"] is None and stamps["rear"]["sensor_t"] is None
    main_t = [s["main"]["sensor_t"] for s in ticks]
    assert main_t == sorted(main_t), main_t
    print(f"capture windows: {len(ticks)} ticks x {len(cams)} cams, before<=after, monotonic, "
          f"main window {ticks[0]['main']['mono_after'] - ticks[0]['main']['mono_before']:.4f} s, "
          f"main call_s {ticks[0]['main']['call_s']:.4f} s (monotonic resolution {res:g} s)")


def check_failed_call_still_has_window() -> None:
    class _Boom:
        resolution = (8, 6)

        def stream_raw(self) -> object:
            raise RuntimeError("socket closed")

    window: dict = {}
    image, reason, _raw = collect._read_camera("main", _Boom(), window)
    assert image is None and "RuntimeError" in reason
    assert window["mono_before"] <= window["mono_after"], window
    assert window["perf_before"] <= window["perf_after"], window


def check_writer_stores_windows() -> None:
    cams = {"main": _Cam(sensor_t=2.0), "wide": _Cam(sensor_t=None)}
    pose = [
        {"id": "main", "pos_m": [0, 1, 1], "yaw_deg": 0.0, "pitch_deg": 0.0, "fov_v": 50.0,
         "near_m": 0.05, "far_m": 300.0, "resolution": [8, 6]},
        {"id": "wide", "pos_m": [0, 1, 1], "yaw_deg": 0.0, "pitch_deg": 0.0, "fov_v": 80.0,
         "near_m": 0.05, "far_m": 300.0, "resolution": [8, 6]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        writer = RawFrameWriter(Path(tmp), pose)
        for _ in range(3):
            frames, stamps, _reasons = collect._poll_cameras(cams)
            writer.write(frames, t=time.time(), timestamps=stamps)
        rows = [json.loads(line) for line in (Path(tmp) / "state.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    last = None
    for row in rows:
        for cid in ("main", "wide"):
            cam = row["cams"][cid]
            assert cam["mono_before"] <= cam["mono_after"], cam
            assert isinstance(cam["call_s"], float) and cam["call_s"] >= cams[cid].delay_s * 0.5, cam
            assert isinstance(cam["t"], float)
            if last is not None:
                assert cam["mono_before"] >= last
            last = cam["mono_after"]
        assert row["cams"]["main"]["sensor_t"] is not None and row["cams"]["wide"]["sensor_t"] is None
    # A stamp without call_s (an older collector) stores None; the other keys are unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        writer = RawFrameWriter(Path(tmp), pose)
        frames, stamps, _reasons = collect._poll_cameras(cams)
        old = {cid: {k: v for k, v in st.items() if k != "call_s"} for cid, st in stamps.items()}
        writer.write(frames, t=time.time(), timestamps=old)
        row = json.loads((Path(tmp) / "state.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["cams"]["main"]["call_s"] is None, row["cams"]["main"]
    assert row["cams"]["main"]["mono_before"] == round(stamps["main"]["mono_before"], 6)
    print("writer: mono_before/mono_after/call_s/t/sensor_t per frame in state.jsonl (call_s None when absent)")


def check_tick_log() -> None:
    assert slow_cause(0.15, {"poll_s": 0.1}, 0.1) is None
    assert slow_cause(0.3, {"poll_s": 0.05, "submit_s": 0.2, "sleep_s": 0.9}, 0.1) == "submit_s"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / TICKS_FILE
        log = TickLog(path, 0.1, flush_every=1)
        log.start(100.0)
        log.add("poll_s", 0.03)
        log.add("veh_s", 0.01)
        log.note(sim_t=50.0, q=2)
        log.add("sleep_s", 0.05)
        log.end(100.09)
        first = log.start(100.1)  # 0.01 s of loop top
        assert first is not None and abs(first["dt"] - 0.1) < 1e-9, first
        assert abs(first["top_s"] - 0.01) < 1e-6 and abs(first["other_s"]) < 1e-6, first
        assert "cause" not in first and "sim_dt" not in first
        log.add("poll_s", 0.31)
        log.note(sim_t=50.1)
        log.end(100.45)
        second = log.start(100.5)
        assert abs(second["dt"] - 0.4) < 1e-9 and second["cause"] == "poll_s", second
        # Wall 0.4 s against 0.1 s of sim time: the sim did not take the time.
        assert abs(second["sim_dt"] - 0.1) < 1e-6, second
        log.note(sim_t=None)
        log.end(100.6)
        log.close()
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["k"] for x in lines] == [0, 1, 2], lines
    assert lines[2]["sim_t"] is None and "sim_dt" not in lines[2]
    print(f"tick log: {len(lines)} lines, slow tick cause={lines[1]['cause']} sim_dt={lines[1]['sim_dt']}")


def check_vehicle_sim_time() -> None:
    class _Vehicle:
        state = {"pos": [1.0, 2.0, 3.0], "dir": [0.0, 1.0, 0.0], "vel": [0.0, 0.0, 0.0], "time": 12.5}
        sensors = None

    assert collect._read_vehicle(_Vehicle())["sim_t"] == 12.5

    class _NoTime(_Vehicle):
        state = {"pos": [1.0, 2.0, 3.0], "dir": [0.0, 1.0, 0.0]}

    assert collect._read_vehicle(_NoTime())["sim_t"] is None
    print("vehicle sim_t from the State sensor's time, None when absent")


def main() -> None:
    check_windows_ordered_and_monotonic()
    check_failed_call_still_has_window()
    check_writer_stores_windows()
    check_tick_log()
    check_vehicle_sim_time()
    print("test_capture_window OK")


if __name__ == "__main__":
    main()
