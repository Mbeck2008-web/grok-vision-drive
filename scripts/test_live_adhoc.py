#!/usr/bin/env python3
"""Live hold-last side cameras: scheduled ad-hoc polls (fake beamngpy camera).

* the schedule is the recorder's ``CameraSchedule`` (imported) at the
  recorder's ``rates_table`` / ``CAMERA_HZ``; every send is a due phase.
* ages come from the request id's collect stamp (wall time after the
  collect), through ``cam_hold.held_sector``: 1/15 s ticks, 1.5 s cap; the
  same pixels on every render still age from the new collect (no pixel
  hash); a reply for another request id is dropped.
* a poll that is not collected within the timeout is black with age -1,
  even inside the 1.5 s cap.
* a respawn drops every held side frame and in-flight request: every side
  camera is black / -1 until a poll sent after the respawn completes.
* a slow poll never blocks the tick: grab() stays bounded and main is read
  every tick.
* the default grab sends no ad-hoc request and has no ages.
* window path: main only (age 0), the others -1; the runtime accepts a
  gvd_cam_age=1 model on the BeamNG path and refuses it elsewhere.
* scripts/live_age_capture.py: PASS / FAIL on a synthetic log.
"""
# gvd-ci: requires torch
from __future__ import annotations

import functools
import io
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from python.data.cam_hold import CAM_AGE_ORDER, HOLD_CAP_S, held_sector, quantise_age  # noqa: E402
from python.data.raw_frames import CAMERA_HZ, CAMERA_PHASE_S, CameraSchedule  # noqa: E402
from python.sensors import live_hold  # noqa: E402
from python.sensors.live_hold import ScheduledAdHoc, bundle_ages, recorder_schedule  # noqa: E402

PIX = np.zeros((48, 64, 4), dtype=np.uint8)
PIX[..., 0] = 90
PIX[..., 1] = 100
PIX[..., 2] = 110
PIX[..., 3] = 255
PIX[30:46, 28:36, :3] = 240


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class FakeCam:
    """beamngpy Camera's ad-hoc API. Same pixels every time."""

    def __init__(self, name: str, *, ready: bool = True, delay_s: float = 0.0) -> None:
        self.name = name
        self.ready = ready
        self.delay_s = float(delay_s)
        self.next_rid = 100 + 1000 * abs(hash(name)) % 7
        self.sent: list[int] = []
        self.collected: list[int] = []

    def stream_raw(self) -> dict:
        return {"colour": PIX.copy()}

    def send_ad_hoc_poll_request(self) -> int:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.next_rid += 1
        self.sent.append(self.next_rid)
        return self.next_rid

    def is_ad_hoc_poll_request_ready(self, rid: int) -> bool:
        if self.delay_s:
            time.sleep(self.delay_s)
        return bool(self.ready) and rid in self.sent

    def collect_ad_hoc_poll_request(self, rid: int) -> dict:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.collected.append(rid)
        return {"colour": PIX.copy()}


def _sync_io(cid, kind, fn, budget):
    return fn()


def _prepare(cid, images):
    return np.ascontiguousarray(images["colour"][..., :3]) if isinstance(images, dict) else None


def check_recorder_schedule() -> None:
    assert live_hold.CameraSchedule is CameraSchedule  # imported from the recorder's module
    from python.collect.auto_mode import rates_table

    sched = recorder_schedule()
    assert sched.rates == {k: float(v) for k, v in rates_table("", "").items()} == {k: float(v) for k, v in CAMERA_HZ.items()}
    assert sched.phases == {k: float(v) for k, v in CAMERA_PHASE_S.items()}
    clock = Clock()
    cams = {cid: FakeCam(cid) for cid in CAM_AGE_ORDER if cid != "main"}
    live = ScheduledAdHoc(cams, clock=clock, wall=clock)
    mirror = recorder_schedule()
    period = 1.0 / 15.0
    sends = {cid: 0 for cid in cams}
    for _k in range(int(20.0 / period)):
        before = {cid: len(cam.sent) for cid, cam in cams.items()}
        live.step(cams, _sync_io, lambda v: False, _prepare)
        due = mirror.tick(clock.t, [c for c in live.cams if c in cams])
        got = sorted(cid for cid, cam in cams.items() if len(cam.sent) > before[cid])
        assert got == sorted(due), (clock.t, got, due)
        for cid in got:
            sends[cid] += 1
        clock.t += period
    for cid, n in sends.items():
        want = 20.0 * CAMERA_HZ[cid]
        assert abs(n - want) <= 0.1 * want + 2, (cid, n, want)
    assert "main" not in live.cams
    print(f"schedule: sends == the recorder CameraSchedule each tick over 20 s; per-cam sends {sends} OK")


def check_request_id_ages() -> None:
    clock = Clock()
    cams = {"pillarL": FakeCam("pillarL")}
    live = ScheduledAdHoc(cams, clock=clock, wall=clock)
    period = 1.0 / 15.0
    main = _prepare("main", {"colour": PIX})
    seen = []
    for _k in range(60):
        n_before = len(cams["pillarL"].collected)
        live.step(cams, _sync_io, lambda v: False, _prepare)
        now = clock.t + 0.003  # main's stamp, a little after the companion collect
        frames, ages = live.compose(now, main)
        k = CAM_AGE_ORDER.index("pillarL")
        if len(cams["pillarL"].collected) > n_before:
            last_wall = clock.t
            assert float(ages[k]) == 0.0 and "pillarL" in frames  # same pixels, still a new render
        if "pillarL" in frames:
            want = held_sector("pillarL", last_wall, frames["pillarL"], now)[1]
            assert abs(float(ages[k]) - want) < 1e-6 and abs(want - quantise_age(now - last_wall)) < 1e-6
            seen.append(round(float(ages[k]) * 15))
        assert float(ages[CAM_AGE_ORDER.index("main")]) == 0.0
        assert all(float(ages[CAM_AGE_ORDER.index(c)]) == -1.0 for c in CAM_AGE_ORDER if c not in ("main", "pillarL"))
        clock.t += period
    assert max(seen) >= 6 and 0 in seen, seen  # 2 Hz on a 15 Hz tick: ages 0..7 ticks
    # A reply for another request id is dropped; the pending one still lands.
    cams["pillarL"].ready = False
    for _k in range(10):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        clock.t += period
        if live.pending:
            break
    rid = live.pending["pillarL"]["rid"]
    assert live.deliver("pillarL", rid - 1, PIX[..., :3], clock.t) is False and live.dropped_ids >= 1
    assert "pillarL" in live.pending
    assert live.deliver("pillarL", rid, PIX[..., :3], clock.t) is True
    print(f"request-id ages: collect stamp + held_sector (ticks seen {sorted(set(seen))}); other ids dropped OK")


def check_timeout_black() -> None:
    clock = Clock()
    cams = {"repeatL": FakeCam("repeatL")}
    live = ScheduledAdHoc(cams, clock=clock, wall=clock)
    period = 1.0 / 15.0
    main = _prepare("main", {"colour": PIX})
    k = CAM_AGE_ORDER.index("repeatL")
    for _k in range(5):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        clock.t += period
    frames, ages = live.compose(clock.t, main)
    assert "repeatL" in frames and 0.0 <= float(ages[k]) <= HOLD_CAP_S
    cams["repeatL"].ready = False  # the next request never completes
    timed_out_at = None
    for _k in range(40):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        frames, ages = live.compose(clock.t, main)
        if live.timeouts and timed_out_at is None:
            timed_out_at = clock.t
            assert "repeatL" not in frames and float(ages[k]) == -1.0
        clock.t += period
    assert timed_out_at is not None and live.timeouts_by_cam.get("repeatL", 0) >= 1
    # The timed-out request's late reply is not used.
    late_rid = cams["repeatL"].sent[-1] if not live.pending else live.pending["repeatL"]["rid"] - 1
    live.late("repeatL", "sched_collect", (late_rid, {"colour": PIX}, clock.t), _prepare)
    frames, ages = live.compose(clock.t, main)
    if "repeatL" not in live.pending or live.pending["repeatL"]["rid"] != late_rid:
        assert "repeatL" not in frames and float(ages[k]) == -1.0
    print(f"timeout: a request not collected in {live.timeout_s:g} s is black / -1 inside the cap OK")


def check_respawn_reset() -> None:
    from types import SimpleNamespace

    from python.sensors.cameras import CameraFrameBundle
    from python.sensors.live_hold import RespawnWatch, scrub_after_respawn

    clock = Clock()
    side = ("pillarL", "repeatL", "rear")
    cams = {cid: FakeCam(cid) for cid in side}
    live = ScheduledAdHoc(cams, clock=clock, wall=clock)
    period = 1.0 / 15.0
    main = _prepare("main", {"colour": PIX})
    for _k in range(40):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        clock.t += period
    frames, ages = live.compose(clock.t, main)
    assert set(side) <= set(frames), sorted(frames)
    # One request is in flight at the respawn.
    cams["pillarL"].ready = False
    for _k in range(10):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        clock.t += period
        if "pillarL" in live.pending:
            break
    stale_rid = live.pending["pillarL"]["rid"]
    before = {cid: max(cam.sent) for cid, cam in cams.items()}
    live.reset()  # the respawn
    frames, ages = live.compose(clock.t, main)
    assert set(frames) == {"main"} and all(
        float(ages[CAM_AGE_ORDER.index(c)]) == -1.0 for c in CAM_AGE_ORDER if c != "main"
    ), (sorted(frames), ages)
    # The pre-respawn request's reply is dropped, ready or not.
    live.late("pillarL", "sched_collect", (stale_rid, {"colour": PIX}, clock.t), _prepare)
    assert "pillarL" not in live.compose(clock.t, main)[0]
    cams["pillarL"].ready = True
    first_back: dict[str, int] = {}
    for _k in range(40):
        live.step(cams, _sync_io, lambda v: False, _prepare)
        frames, ages = live.compose(clock.t, main)
        for cid in side:
            if cid in frames and cid not in first_back:
                first_back[cid] = cams[cid].collected[-1]
                assert first_back[cid] > before[cid], (cid, first_back[cid], before[cid])
                assert float(ages[CAM_AGE_ORDER.index(cid)]) == 0.0
            if cid not in first_back:
                assert cid not in frames and float(ages[CAM_AGE_ORDER.index(cid)]) == -1.0
        clock.t += period
    assert set(first_back) == set(side), first_back

    # Backend: reset_live_hold() then the next grab has every side camera -1.
    be = _backend({"main": FakeCam("main"), **{cid: FakeCam(cid) for cid in side}})
    be.enable_scheduled_adhoc()
    for _k in range(30):
        bundle = be.grab()
        time.sleep(1.0 / 30.0)
    assert any(a >= 0 for c, a in zip(CAM_AGE_ORDER, bundle.cam_ages) if c != "main")
    be.reset_live_hold()
    bundle = be.grab()
    assert set(bundle.held_frames) == {"main"}
    assert all(a == -1.0 for c, a in zip(CAM_AGE_ORDER, bundle.cam_ages) if c != "main"), bundle.cam_ages
    be.close()

    # The grab of the respawn tick ran before the edge was seen: scrubbed to main only.
    img = PIX[..., :3].copy()
    tick = CameraFrameBundle(frames={"main": img}, cam_ages=tuple([0.2] * 8), held_frames={"main": img, "pillarL": img})
    scrub_after_respawn(tick)
    assert set(tick.held_frames) == {"main"}
    assert all(a == -1.0 for c, a in zip(CAM_AGE_ORDER, tick.cam_ages) if c != "main")

    # Respawn edge: vehicle change, reconnect, a jump no car could drive; not a fast straight.
    watch = RespawnWatch()
    v = lambda **kw: SimpleNamespace(**{"connected": True, "vid": "ego", "pos": (0.0, 0.0, 0.0), "speed_mps": 0.0, **kw})  # noqa: E731
    assert watch.observe(v(), 0.1) is None
    x = 0.0
    for _k in range(50):
        x += 45.0 * 0.1
        assert watch.observe(v(pos=(x, 0.0, 0.0), speed_mps=45.0), 0.1) is None
    assert watch.observe(v(pos=(x + 400.0, 0.0, 0.0)), 0.1).startswith("jump")
    assert watch.observe(v(pos=(x + 400.0, 0.0, 0.0), vid="other"), 0.1) == "vehicle"
    assert watch.observe(v(connected=False, pos=None, vid="other"), 0.1) is None
    assert watch.observe(v(pos=(x + 400.0, 0.0, 0.0), vid="other"), 0.1) == "reconnect"
    assert watch.count == 3

    # run_vision: the edge resets the backend hold and scrubs this tick's grab.
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    i = rv.index("respawn = respawn_watch.observe(vdata, tick_dt)")
    block = rv[i : i + 400]
    assert "backend.reset_live_hold()" in block and "scrub_after_respawn(bundle)" in block
    assert rv.index("bundle = backend.grab()") < i < rv.index("scene_guard.observe(scene_lane, strip")
    print("respawn: no pre-respawn frame or request survives; side cams -1 until a post-respawn poll lands OK")


def _backend(cams: dict):
    from python.sensors.cameras import BeamNGPyBackend

    be = BeamNGPyBackend()
    be._ok = True
    be._sensors = dict(cams)
    return be


def check_tick_not_blocked() -> None:
    from python.sensors.cameras import CamHealth

    slow = {cid: FakeCam(cid, delay_s=0.3) for cid in ("pillarL", "pillarR", "repeatL", "repeatR", "rear", "narrow", "wide")}
    main = FakeCam("main")
    be = _backend({"main": main, **slow})
    be.enable_scheduled_adhoc()
    durations = []
    main_ok = 0
    end = time.monotonic() + 2.0
    n = 0
    while time.monotonic() < end:
        t0 = time.perf_counter()
        bundle = be.grab()
        durations.append(time.perf_counter() - t0)
        n += 1
        if bundle.health.get("main") == CamHealth.OK:
            main_ok += 1
        assert bundle.cam_ages is not None and len(bundle.cam_ages) == 8
        time.sleep(1.0 / 15.0)
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    assert main_ok == n, (main_ok, n)
    assert max(durations) < 0.05, durations[-5:]
    assert sum(len(c.sent) for c in slow.values()) >= 1
    # Every companion is black: nothing came back inside the timeout.
    assert all(a == -1.0 for c, a in zip(CAM_AGE_ORDER, bundle.cam_ages) if c != "main")
    be.close()
    print(f"non-blocking: {n} grabs with 0.3 s polls, max {max(durations) * 1000:.1f} ms p95 {p95 * 1000:.1f} ms, main every tick OK")

    fast = {cid: FakeCam(cid) for cid in ("pillarL", "repeatL")}
    be = _backend({"main": FakeCam("main"), **fast})
    be.enable_scheduled_adhoc()
    got = set()
    for _k in range(40):
        bundle = be.grab()
        for c, a in zip(CAM_AGE_ORDER, bundle.cam_ages):
            if a >= 0:
                got.add(c)
        assert set(bundle.held_frames) >= {"main"}
        time.sleep(1.0 / 15.0)
    assert {"main", "pillarL", "repeatL"} <= got, got
    be.close()
    print("backend: fast fake polls land in held_frames with ages OK")


def check_default_unchanged() -> None:
    cams = {"main": FakeCam("main"), "pillarL": FakeCam("pillarL")}
    be = _backend(cams)
    for _k in range(20):
        bundle = be.grab()
        assert bundle.cam_ages is None and bundle.held_frames == {}
    assert cams["pillarL"].sent == []
    held, ages = bundle_ages(bundle)
    assert set(held) == {"main"} and float(ages[CAM_AGE_ORDER.index("main")]) == 0.0
    assert all(float(a) == -1.0 for c, a in zip(CAM_AGE_ORDER, ages) if c != "main")
    be.close()
    print("default grab: no ad-hoc request, no ages; main-only ages from bundle_ages OK")


def check_runtime_gate() -> None:
    import torch

    from python.sensors.cameras import BeamNGPyBackend, CameraFrameBundle, WindowBackend
    from python.train.scene_lane_runtime import SceneLaneGuard, SceneLaneRuntime
    from python.train.scene_net import SceneNet, export_scene_onnx

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "age.onnx"
        with redirect_stdout(io.StringIO()):
            export_scene_onnx(path, SceneNet(cams=("main",), cam_age=True), cams=("main",), heads=("lane_gt",), cam_age=True)
        beam = SceneLaneGuard()
        rt = beam.load(functools.partial(SceneLaneRuntime, feeds_cam_age=bool(getattr(BeamNGPyBackend, "can_feed_cam_age", False))), path)
        assert beam.refused is None and rt is not None and rt.cam_age
        win = SceneLaneGuard()
        got = win.load(functools.partial(SceneLaneRuntime, feeds_cam_age=bool(getattr(WindowBackend, "can_feed_cam_age", False))), path)
        assert got is None and win.refused is not None
        main = np.full((192, 256, 3), 90, dtype=np.uint8)
        bundle = CameraFrameBundle(frames={"main": main}, backend="window")
        held, ages = bundle_ages(bundle)
        from python.data.strip_writer import compose_strip

        lane, conf = beam.observe(rt, compose_strip(held), 1.0 / 15.0, engaged=True, ages=ages)
        assert beam.hold is None and conf is not None
    print("runtime: gvd_cam_age=1 accepted on the BeamNG path, refused on window; window ages main-only OK")


def check_capture_script() -> None:
    from live_age_capture import compare, parse_log

    train = {cam: {"median": 0.2, "p95": 0.467} for cam in CAM_AGE_ORDER if cam != "main"}
    lines = []
    rng = np.random.RandomState(0)
    for k in range(900):
        ages = []
        for cam in CAM_AGE_ORDER:
            if cam == "main":
                ages.append(f"{cam}:0.000")
            else:
                ages.append(f"{cam}:{(k % 8) / 15.0:.3f}")
        lines.append(f"[GVD] cam_age t={k} tick_ms={20 + rng.rand() * 10:.2f} loop_dt_ms=66.7 scheduled=1 timeouts=0 ages={','.join(ages)}")
    live = parse_log(lines)
    result = compare(live, train)
    assert result["pass"] and result["tick_pass"] and result["scheduled_ticks"] == 900, json.dumps(result)[:400]
    late = compare(live, {cam: {"median": 0.6, "p95": 0.9} for cam in train})
    assert not late["pass"]
    slow = parse_log([line.replace("tick_ms=2", "tick_ms=9") for line in lines])
    assert not compare(slow, train)["tick_pass"]
    print("live_age_capture: PASS within 1 / 2 ticks and tick p95 <= 67 ms; FAIL otherwise OK")


def main() -> None:
    check_recorder_schedule()
    check_request_id_ages()
    check_timeout_black()
    check_tick_not_blocked()
    check_respawn_reset()
    check_default_unchanged()
    check_runtime_gate()
    check_capture_script()
    print("test_live_adhoc: all OK")


if __name__ == "__main__":
    main()
