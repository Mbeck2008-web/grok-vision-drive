#!/usr/bin/env python3
"""Collector stops: every early exit logs one ``collect stop reason=`` line, writes
``stop_reason`` to collect.json, exits non-zero, and leaves a stopped heartbeat.
Also the rec3 regression: a segment that never drove must not end the run.

Offline; reuses the mocks in test_collect_spawn_fix.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.collect_scene_lanes as collect  # noqa: E402
import scripts.test_collect_spawn_fix as sf  # noqa: E402


def _check_stop(code, text: str, out: Path, reason: str, want_code) -> dict:
    lines = [x for x in text.splitlines() if x.startswith("collect stop reason=")]
    assert len(lines) == 1, (reason, lines, text[-2500:])
    assert lines[0].startswith(f"collect stop reason={reason} code={want_code} "), (lines, text[-2000:])
    assert code == want_code, (reason, code, text[-2000:])
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == reason and status["stop_code"] == want_code, status
    beat = json.loads((out.parent / "status" / collect.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert beat["state"] == "stopped" and beat["reason"] == reason and beat["code"] == want_code, beat
    assert beat["pid"] == os.getpid() and beat["last_line"] and "done_s" in beat and "target_s" in beat, beat
    shared = out.parent / "status" / "collect.json"
    if shared.is_file():
        assert json.loads(shared.read_text(encoding="utf-8"))["stop_reason"] == reason
    return status


def check_requested_and_target() -> None:
    code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 120}, ["--no-excursions"])
    _check_stop(code, text, out, "requested", 0)  # the stop file: deliberate, exit 0
    code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999}, ["--no-excursions", "--minutes", "1"])
    _check_stop(code, text, out, "target_reached", 0)
    print("requested / target_reached ok")


def _never_drives_after(n0: int):
    return lambda n: 10.0 if n <= n0 else 0.0


def check_rec3_segment_end_keeps_running() -> None:
    """rec3: #10 ended, #11 never drove, its stationary end stopped the run (exit 0, no reason).
    Now the run moves on to the next segment; only N stuck ends in a row stop it, logged."""
    paint = sf._Paint(lambda x, y: x > 1000.0)
    extra = ["--stall-respawn-s", "0", "--segment-min", "1", "--minutes", "60", "--no-progress-ends", "3"]
    code, text, out = sf._run_west_coast(paint, sf._west_pools(), extra, 99999, speed_at=_never_drives_after(100))
    ends = [x for x in text.splitlines() if x.startswith("collect segment end ")]
    assert ends[0] == "collect segment end west_coast_usa#0 stationary", ends
    assert "collect segment end west_coast_usa#1 stationary: no recorded row since the last segment end" in text, ends
    assert "1 in a row); moving to the next segment" in text and "collect segment west_coast_usa#2" in text, text[-3000:]
    assert " ended" not in "\n".join(x for x in text.splitlines() if x.startswith("collect stop "))
    status = _check_stop(code, text, out, "no_progress", collect.STOP_EXIT_EARLY)
    assert status["stop_detail"].startswith("3 segment ends in a row with no recorded row"), status
    # Off: --no-progress-ends 0 never stops on it; the stop file ends the run.
    code, text0, out0 = sf._run_west_coast(paint, sf._west_pools(), extra[:-1] + ["0"], 1500, speed_at=_never_drives_after(100))
    assert "no_progress" not in text0 and text0.count("moving to the next segment") >= 4, text0[-2000:]
    _check_stop(code, text0, out0, "requested", 0)
    # Default stall respawn: 3 respawns with no row, then the stationary path ends the segment.
    code, text2, out2 = sf._run_west_coast(paint, sf._west_pools(), ["--segment-min", "1", "--minutes", "60", "--no-progress-ends", "2"],
                                           99999, speed_at=_never_drives_after(100))
    assert 2 <= text2.count("collect stall_respawn west_coast_usa#0") <= 4, text2[-3000:]
    assert "collect segment end west_coast_usa#0 stationary" in text2
    _check_stop(code, text2, out2, "no_progress", collect.STOP_EXIT_EARLY)
    print(f"rec3 regression ok: {len(ends)} segment ends, then reason=no_progress exit 6; stall respawn capped")


def check_schedule_wrap() -> None:
    """A used-up schedule (no segment boundary left before target_s) wraps instead of stopping."""
    paint = sf._Paint(lambda x, y: x > 1000.0)
    real = collect.continue_after_segment_end
    collect.continue_after_segment_end = lambda done, cut, *a, **k: ("stop", cut + 60.0, 0.0)
    extra = ["--stall-respawn-s", "0", "--segment-min", "1", "--minutes", "60", "--no-progress-ends", "3"]
    try:
        code, text, out = sf._run_west_coast(paint, sf._west_pools(), extra, 99999, speed_at=_never_drives_after(100))
    finally:
        collect.continue_after_segment_end = real
    wraps = [x for x in text.splitlines() if x.startswith("collect schedule used up at done_s=")]
    assert len(wraps) == 3 and all("wrapping the map rotation (cut 60s -> 0)" in x for x in wraps), wraps
    assert text.count("collect segment west_coast_usa#0") >= 3, text[-2000:]  # reloaded, fresh spawn each time
    _check_stop(code, text, out, "no_progress", collect.STOP_EXIT_EARLY)
    print(f"schedule wrap ok: {len(wraps)} wraps, run kept going")


def check_level_load_failed() -> None:
    import python.collect.spawn_mix as spawn_mix
    from python.collect.level_session import OpenedLevel
    from python.debug.gt_geom import RoadPoly

    calls = {"n": 0}

    def enter(bng, level, cfg, cams):
        calls["n"] += 1
        if calls["n"] >= 2:
            return OpenedLevel(vehicle=None, roads=[], spawn=None, cameras={}, error="level load timed out")
        return OpenedLevel(vehicle=bng.vehicles.vehicle, roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
                           spawn=(0.4, 12.0, 0.5), cameras={"main": sf._Cam()})

    previous = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load)
    collect.enter_map = enter
    collect.painted_for = lambda level: sf._Paint(lambda x, y: x > 1000.0)
    pools = sf._west_pools()
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))
    try:
        code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999},
                                            ["--maps", "west_coast_usa", "--no-vary-env", "--no-excursions",
                                             "--segment-min", "1", "--minutes", "3"])
    finally:
        collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load = previous
    _check_stop(code, text, out, "level_load_failed", 2)
    print("level_load_failed ok")


def check_camera_outage() -> None:
    old = sf._Cam.poll
    sf._Cam.poll = lambda self: {}
    try:
        code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999}, ["--no-excursions"])
    finally:
        sf._Cam.poll = old
    _check_stop(code, text, out, "camera_outage", 4)
    print("camera_outage ok")


def check_disk_low() -> None:
    import python.data.data_root as data_root

    real = data_root.check_data_path
    calls = {"n": 0}

    def guard(path):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise data_root.DataDriveError("data disk below 20 GB free")
        return real(path)

    data_root.check_data_path = guard
    try:
        code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999}, ["--no-excursions"])
    finally:
        data_root.check_data_path = real
    _check_stop(code, text, out, "disk_low", 3)
    print("disk_low ok")


def check_smoke_failed() -> None:
    real = collect.decode_issues
    collect.decode_issues = lambda root: ["unreadable 000001.png"]
    try:
        code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999},
                                            ["--no-excursions", "--smoke-s", "2"])
    finally:
        collect.decode_issues = real
    _check_stop(code, text, out, "smoke_failed", 5)
    print("smoke_failed ok")


def check_exception() -> None:
    real = collect._read_vehicle
    calls = {"n": 0}

    def boom(vehicle):
        calls["n"] += 1
        if calls["n"] > 30:
            raise RuntimeError("sensor exploded")
        return real(vehicle)

    collect._read_vehicle = boom
    buf_text = ""
    root = Path(tempfile.mkdtemp())
    try:
        sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999}, ["--no-excursions"], root)
    except RuntimeError as exc:
        assert "sensor exploded" in str(exc)
    else:
        raise AssertionError("the exception must propagate")
    finally:
        collect._read_vehicle = real
    out = root / "scene"
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "exception" and "RuntimeError: sensor exploded" in status["stop_detail"], status
    assert status["stop_code"] is None
    beat = json.loads((root / "status" / collect.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert beat["state"] == "stopped" and beat["reason"] == "exception", beat
    del buf_text
    print("exception ok (re-raised, stop_reason recorded)")


def check_session_open_failed() -> None:
    import io

    import python.sensors.tech as tech

    real = tech.open_tech_beamngpy
    root = Path(tempfile.mkdtemp())

    def refuse(*a, **k):
        raise ConnectionRefusedError("BeamNG not listening")

    tech.open_tech_beamngpy = refuse
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(["--minutes", "5", "--data-dir", str(root), "--out", str(root / "scene"),
                             "--host", "127.0.0.1", "--port", "1"])
    finally:
        sys.stdout = old
        tech.open_tech_beamngpy = real
    _check_stop(code, buf.getvalue(), root / "scene", "session_open_failed", 2)
    print("session_open_failed ok")


def check_heartbeat_interval() -> None:
    root = Path(tempfile.mkdtemp())
    run = collect.RunStop()
    run.out, run.progress_path, run.target_s, run.status_root = root / "scene", root / "scene" / "collect.json", 600.0, root / "status"
    clock = {"t": 100.0}
    real = collect.time.monotonic
    collect.time.monotonic = lambda: clock["t"]
    path = root / "status" / collect.HEARTBEAT_FILE
    try:
        run.last_line = "collect driving x#0"
        run.beat(1.0)
        first = json.loads(path.read_text(encoding="utf-8"))
        clock["t"] += 29.0
        run.beat(2.0)
        assert json.loads(path.read_text(encoding="utf-8"))["done_s"] == 1.0  # under 30 s: not rewritten
        clock["t"] += 1.0
        run.beat(3.0)
        again = json.loads(path.read_text(encoding="utf-8"))
    finally:
        collect.time.monotonic = real
    assert first["state"] == "running" and first["reason"] is None and first["target_s"] == 600.0 and first["last_line"] == "collect driving x#0"
    assert again["done_s"] == 3.0 and set(first) >= {"pid", "t", "time", "done_s", "target_s", "last_line", "state", "reason", "segment"}
    print(f"heartbeat ok: {sorted(first)}")


def main() -> None:
    check_heartbeat_interval()
    check_requested_and_target()
    check_rec3_segment_end_keeps_running()
    check_schedule_wrap()
    check_level_load_failed()
    check_camera_outage()
    check_disk_low()
    check_smoke_failed()
    check_exception()
    check_session_open_failed()
    print("collect stops ok")


if __name__ == "__main__":
    main()
