#!/usr/bin/env python3
"""Live hold-last side cameras: streamed every tick, the recorder's storage filter, wall-time ages.

* the filter is the recorder's ``CameraSchedule`` (imported) at the
  recorder's ``rates_table`` / ``CAMERA_HZ``.
* wall time, not ticks: one wall-time frame sequence at 185 ms and at 310 ms
  ticks (and on jittered ticks) gives the same ages within quantisation, and
  live's stored set and ages equal the recorder's storage decisions fed
  through the trainer's ``cam_hold.hold_plan``.
* the BeamNG grab streams all 8 cameras every tick and never sends an ad-hoc
  request; the default grab has no ages.
* a respawn / reset drops every held side frame: side cameras are black / -1
  until a frame stored after it.
* reset detection: a 0.8 g stop never fires; an in-place reset (11 -> 0.27
  m/s, 0.54 m) does; id / reconnect / jump still do; a locked wheel and a
  stale sample do not.
* stale ego: a coalesced poll_vehicle is stale; engaged past one tick it
  takes the existing heartbeat_stale stop (shadow_tick), wired before it.
* window path main only; the runtime accepts a gvd_cam_age=1 model on BeamNG
  only; scripts/live_age_capture.py bars.
"""
# gvd-ci: requires torch
from __future__ import annotations

import functools
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from python.data.cam_hold import CAM_AGE_ORDER, HOLD_TICK_S, hold_plan  # noqa: E402
from python.data.raw_frames import CAMERA_HZ, CameraSchedule  # noqa: E402
from python.sensors import live_hold  # noqa: E402
from python.sensors.live_hold import (  # noqa: E402
    FRONT_CAMS,
    SIDE_CAMS,
    LiveStreamHold,
    RespawnWatch,
    StaleEgo,
    bundle_ages,
    recorder_schedule,
    scrub_after_respawn,
    vehicle_data_stale,
)

PIX = np.zeros((48, 64, 4), dtype=np.uint8)
PIX[..., 0] = 90
PIX[..., 1] = 100
PIX[..., 2] = 110
PIX[..., 3] = 255
PIX[30:46, 28:36, :3] = 240
ALL = tuple(CAM_AGE_ORDER)


class FakeCam:
    """beamngpy Camera: stream_raw shared memory, plus the ad-hoc API (which must never be called)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.streams = 0
        self.adhoc_calls = 0

    def stream_raw(self) -> dict:
        self.streams += 1
        return {"colour": PIX.copy()}

    def send_ad_hoc_poll_request(self) -> int:
        self.adhoc_calls += 1
        return 1

    def is_ad_hoc_poll_request_ready(self, rid: int) -> bool:
        self.adhoc_calls += 1
        return True

    def collect_ad_hoc_poll_request(self, rid: int) -> dict:
        self.adhoc_calls += 1
        return {"colour": PIX.copy()}


def _backend(cams: dict):
    from python.sensors.cameras import BeamNGPyBackend

    be = BeamNGPyBackend()
    be._ok = True
    be._sensors = dict(cams)
    return be


def check_recorder_schedule() -> None:
    assert live_hold.CameraSchedule is CameraSchedule  # imported from the recorder's module
    from python.collect.auto_mode import rates_table

    sched = recorder_schedule()
    assert sched.rates == {k: float(v) for k, v in rates_table("", "").items()} == {k: float(v) for k, v in CAMERA_HZ.items()}
    assert set(FRONT_CAMS) | set(SIDE_CAMS) == set(ALL) and not set(FRONT_CAMS) & set(SIDE_CAMS)
    print(f"schedule: the recorder's CameraSchedule at rates_table {dict(sched.rates)} OK")


# ---- wall time, not ticks --------------------------------------------------------


def _ticks(period: float, seconds: float, *, jitter: float = 0.0, seed: int = 0) -> list[float]:
    rng = np.random.RandomState(seed)
    out, t = [], 0.0
    while t < seconds:
        out.append(t)
        t += period * (1.0 + (rng.uniform(-jitter, jitter) if jitter else 0.0))
    return out


def _run(tick_times: list[float], *, mono0: float = 5000.0, wall0: float = 1.7e9) -> dict:
    """Live (LiveStreamHold) and the recorder (CameraSchedule -> stored rows -> cam_hold.hold_plan) on the same ticks.

    Every camera's read on a tick has wall stamp ``wall0 + t + k * 0.5 ms`` (main first),
    the monotonic clock is ``mono0 + t + 4 ms`` after the reads, like the recorder.
    """
    live = LiveStreamHold()
    rec = recorder_schedule()
    rows, live_ages, live_taken, nows = [], [], [], []
    for i, t in enumerate(tick_times):
        walls = {cam: wall0 + t + 0.0005 * k for k, cam in enumerate(("main",) + tuple(c for c in ALL if c != "main"))}
        frames = {cam: ("px", cam, i) for cam in ALL}
        mono = mono0 + t + 0.004
        held, ages = live.step(frames, walls, mono)
        live_ages.append(np.asarray(ages, dtype=np.float64))
        live_taken.append(set(live.last_taken))
        nows.append(walls["main"])
        # The recorder: read all 8, tick(mono, list(frames)), store the due ones with cams.<cam>.t = wall.
        due = rec.tick(mono, list(frames))
        rows.append({"i": i, "t": round(walls["main"], 6), "dt_s": None if i == 0 else 0.1, "segment": 0, "session": "s",
                     "cams": {cam: {"t": walls[cam]} for cam in due}})
        for cam in FRONT_CAMS:
            assert held.get(cam) == frames[cam]  # fronts are this tick's read at age 0
    plan = hold_plan(rows)
    train_ages = [np.asarray([plan[i][cam][1] for cam in ALL], dtype=np.float64) for i in range(len(rows))]
    return {"live": live_ages, "train": train_ages, "taken": live_taken, "rows": rows, "now": nows, "wall0": wall0}


def _grid_age(cam: str, now_rel: float, lead: float) -> float:
    """Wall-time age of ``cam`` at ``now``: now - the latest grid phase <= now + lead (the schedule's rounding)."""
    from python.data.raw_frames import CAMERA_PHASE_S

    period = 1.0 / float(CAMERA_HZ[cam])
    first = float(np.fmod(max(0.0, CAMERA_PHASE_S.get(cam, 0.0)), period))
    n = np.floor((now_rel + min(lead, period / 2.0) + 1e-9 - first) / period)
    return now_rel - (first + n * period)


def check_wall_time_ages() -> None:
    # Quantisation (half a 1/15 s step) plus the per-camera read offsets inside a tick (<= 4 ms here).
    q = HOLD_TICK_S / 2.0 + 0.004 + 1e-6
    side = [ALL.index(c) for c in SIDE_CAMS]
    results = {}
    for name, ticks, jittered in (
        ("185ms", _ticks(0.185, 30.0), False),
        ("310ms", _ticks(0.310, 30.0), False),
        ("jitter", _ticks(0.185, 30.0, jitter=0.5, seed=3), True),
    ):
        got = _run(ticks)
        # 1) the recorder's storage decisions and the trainer's ages, exactly.
        for i, row in enumerate(got["rows"]):
            assert got["taken"][i] == set(row["cams"]), (name, i)
            assert np.allclose(got["live"][i], got["train"][i]), (name, i, got["live"][i], got["train"][i])
        # 2) a function of wall time, not of the tick count: within half a tick + quantisation of the grid age.
        worst = 0.0
        for i, t in enumerate(ticks):
            if t < 1.6:
                continue  # the 1 Hz cameras' first grid phase
            prev_gap = ticks[i] - ticks[i - 1]
            gaps = [ticks[j] - ticks[j - 1] for j in range(max(1, i - 3), min(len(ticks), i + 2))]
            # A phase is served within half a tick of it at a fixed rate; on jittered
            # ticks the serving tick can be up to one interval away.
            half = (1.0 if jittered else 0.5) * max(gaps)
            for k in side:
                cam = ALL[k]
                age = got["live"][i][k]
                assert age >= 0.0, (name, cam, t)
                want = _grid_age(cam, t, prev_gap / 2.0)
                worst = max(worst, abs(age - want) - half)
                assert abs(age - want) <= half + q, (name, cam, t, age, want)
        results[name] = got
        print(f"wall time {name}: {len(ticks)} ticks, live stored set == recorder's, ages == trainer hold_plan, "
              f"|age - grid age| <= half tick + 1/30 s (worst excess {max(0.0, worst) * 1000:.1f} ms) OK")
    # 3) Sampled distributions. Pointwise the age is the wall-time grid age within half
    # a tick (2); the recorder's nearest-tick rounding serves a phase up to half a tick
    # early or late, so the age a slow loop samples is not tick-rate free: at 310 ms
    # ticks a 2 Hz pillar is stored on most ticks. Live at 15 Hz vs the recorder's
    # 185 ms rows must still meet the capture bars (median 1 tick, p95 2 ticks).
    stats = {}
    for name, per, jit in (("66.7ms", 1.0 / 15.0, 0.1), ("185ms", 0.185, 0.3), ("310ms", 0.310, 0.3)):
        # Jittered like real loops (rec5c's 185 ms median tick is not a fixed period).
        arr = np.asarray(_run(_ticks(per, 300.0, jitter=jit, seed=1))["live"][30:])
        stats[name] = {ALL[k]: (float(np.median(arr[:, k])), float(np.percentile(arr[:, k], 95))) for k in side}
    for cam in SIDE_CAMS:
        (m_live, p_live), (m_rec, p_rec) = stats["66.7ms"][cam], stats["185ms"][cam]
        assert abs(m_live - m_rec) <= HOLD_TICK_S + 1e-6 and abs(p_live - p_rec) <= 2 * HOLD_TICK_S + 1e-6, (cam, stats)
    fmt = {n: {c: f"{m:.3f}/{p:.3f}" for c, (m, p) in v.items()} for n, v in stats.items()}
    print(f"wall time: median/p95 ages {fmt}; 15 Hz live vs 185 ms rows within 1 / 2 ticks OK "
          "(310 ms differs by the schedule's half-tick rounding, reported)")


# ---- backend ---------------------------------------------------------------------


def check_backend_streams() -> None:
    cams = {cid: FakeCam(cid) for cid in ALL}
    be = _backend(cams)
    for _k in range(5):
        bundle = be.grab()
        assert bundle.cam_ages is None and bundle.held_frames == {}  # default grab: no hold, no ages
    be.enable_live_hold()
    seen_side = set()
    n = 0
    for _k in range(45):
        bundle = be.grab()
        n += 1
        ages = dict(zip(ALL, bundle.cam_ages))
        for cam in FRONT_CAMS:
            assert ages[cam] == 0.0 and cam in bundle.held_frames
        for cam in SIDE_CAMS:
            if ages[cam] >= 0:
                seen_side.add(cam)
                assert cam in bundle.held_frames
        time.sleep(1.0 / 15.0)
    assert seen_side == set(SIDE_CAMS), seen_side
    assert all(c.adhoc_calls == 0 for c in cams.values()), {k: c.adhoc_calls for k, c in cams.items()}
    assert all(c.streams == n + 5 for c in cams.values()), {k: c.streams for k, c in cams.items()}
    be.close()
    print(f"backend: all 8 cameras streamed on each of {n + 5} grabs, 0 ad-hoc calls; held sides carry ages OK")


def check_respawn_reset() -> None:
    cams = {cid: FakeCam(cid) for cid in ALL}
    be = _backend(cams)
    hold = be.enable_live_hold()
    for _k in range(30):
        be.grab()
        time.sleep(1.0 / 15.0)
    bundle = be.grab()
    assert all(a >= 0 for c, a in zip(ALL, bundle.cam_ages) if c in ("pillarL", "pillarR"))
    t_reset = time.time()
    be.reset_live_hold()
    scrub_after_respawn(bundle)
    assert set(bundle.held_frames) == {"main"}
    assert all(a == -1.0 for c, a in zip(ALL, bundle.cam_ages) if c != "main")
    firsts: dict[str, float] = {}
    for _k in range(40):
        bundle = be.grab()
        for cam, age in zip(ALL, bundle.cam_ages):
            if cam in SIDE_CAMS and age >= 0 and cam not in firsts:
                last = hold.hold.last(cam)
                assert last is not None and last[0] >= t_reset, cam  # stored after the reset
                firsts[cam] = last[0]
        time.sleep(1.0 / 15.0)
    assert {"pillarL", "pillarR"} <= set(firsts)
    be.close()
    print(f"respawn: every side camera black / -1 until a frame stored after the reset ({sorted(firsts)}) OK")


# ---- reset detection -------------------------------------------------------------


def _v(speed: float, pos_x: float, *, vid: str = "ego", connected: bool = True, wheel: float | None = None, note: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        connected=connected, vid=vid, pos=(pos_x, 0.0, 0.0), vel=(speed, 0.0, 0.0),
        speed_mps=speed if wheel is None else wheel, note=note,
    )


def check_reset_detection() -> None:
    for dt in (1.0 / 15.0, 0.185, 0.31):
        # Hard stop at a light: 0.8 g from 15 m/s to 0, integrated.
        w = RespawnWatch()
        mono, x, v = 100.0, 0.0, 15.0
        assert w.observe(_v(v, x), None, mono=mono) is None
        while v > 0.0:
            mono += dt
            v_new = max(0.0, v - 0.8 * 9.81 * dt)
            x += 0.5 * (v + v_new) * dt
            v = v_new
            assert w.observe(_v(v, x), dt, mono=mono) is None, (dt, v)
        for _k in range(5):
            mono += dt
            assert w.observe(_v(0.0, x), dt, mono=mono) is None
        # A locked wheel: wheelspeed 0 at once, the body still at 11 m/s.
        w = RespawnWatch()
        assert w.observe(_v(11.0, 0.0), None, mono=0.0) is None
        assert w.observe(_v(10.8, 11.0 * dt, wheel=0.0), dt, mono=dt) is None
        # In-place reset like the real capture: 10.99 -> 0.27 m/s in one tick, moved 0.54 m.
        w = RespawnWatch()
        assert w.observe(_v(10.99, 50.0), None, mono=0.0) is None
        got = w.observe(_v(0.27, 50.54), dt, mono=dt)
        assert got is not None and got.startswith("reset"), (dt, got)
        assert w.count == 1
    # The stale (coalesced) sample in between is skipped, and dt spans the fresh samples.
    w = RespawnWatch()
    assert w.observe(_v(10.0, 0.0), None, mono=0.0) is None
    stale = _v(10.0, 0.0, note="soft esc: sensors.poll coalesced; last-good (not a new GPS fix)")
    assert w.observe(stale, 0.066, mono=0.066) is None
    assert w.observe(_v(9.5, 1.3), 0.066, mono=0.132) is None  # 0.5 m/s in 132 ms: fine
    # Existing triggers stay.
    w = RespawnWatch()
    w.observe(_v(5.0, 0.0, vid="a"), None, mono=0.0)
    assert w.observe(_v(5.0, 0.3, vid="b"), 0.066, mono=0.066) == "vehicle"
    w.observe(_v(5.0, 0.3, vid="b", connected=False), 0.066, mono=0.132)
    assert w.observe(_v(5.0, 0.6, vid="b"), 0.066, mono=0.198) == "reconnect"
    got = w.observe(_v(5.0, 80.0, vid="b"), 0.066, mono=0.264)
    assert got is not None and got.startswith("jump")
    print("reset detection: 0.8 g stops and a locked wheel never fire; the in-place reset (10.99 -> 0.27 m/s, "
          "0.54 m) fires at 67 / 185 / 310 ms; id / reconnect / jump kept OK")


# ---- stale ego -------------------------------------------------------------------


def check_stale_ego() -> None:
    from python.runtime.shadow import shadow_tick
    from python.sensors.cameras import BeamNGPyBackend
    from python.sensors.tech import VehicleData

    tech_src = (ROOT / "python" / "sensors" / "tech.py").read_text(encoding="utf-8")
    assert 'tag = "soft esc: sensors.poll coalesced; last-good (not a new GPS fix)"' in tech_src
    assert vehicle_data_stale(VehicleData(connected=True, note="soft esc: sensors.poll coalesced; last-good (not a new GPS fix)"))
    assert not vehicle_data_stale(VehicleData(connected=True, speed_mps=3.0)) and not vehicle_data_stale(None)
    # The producer: camera IO in flight -> poll_vehicle skips sensors.poll.
    be = BeamNGPyBackend()
    be.session = SimpleNamespace(vehicle=object(), _last_vehicle_data=None, _last_sensor_map=None, poll=lambda: VehicleData(connected=True))
    be._socket_io_busy = lambda: True
    assert vehicle_data_stale(be.poll_vehicle())
    be._socket_io_busy = lambda: False
    assert not vehicle_data_stale(be.poll_vehicle())
    # Counting: one stale tick is allowed while engaged; the second in a row blocks.
    fresh, stale = VehicleData(connected=True), VehicleData(connected=True, note="camera io in flight; sensors.poll skipped")
    ego = StaleEgo()
    seq = [stale, fresh, stale, stale, stale, fresh]
    blocks = []
    for vd in seq:
        ego.observe(vd)
        blocks.append((ego.blocks_drive(True), ego.blocks_drive(False)))
    assert [b[0] for b in blocks] == [False, False, False, True, True, False], blocks
    assert not any(b[1] for b in blocks) and ego.total == 4
    # The existing path heartbeat_ok=False takes: stop + disengage (heartbeat_stale).

    class _E2E:
        def forward(self, *a, **k):
            return SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0, accel=0.0, ok=False)

    path = [{"x": 0.0, "y": float(y)} for y in range(2, 30, 2)]
    tick = shadow_tick(policy="modular", engaged=True, heartbeat_ok=False, path_debug_preview=False, allow_preview_drive=False,
                       path_ego=path, planner={"target_v": 5.0, "aeb": "off"}, ego_speed_mps=5.0, lane_conf=1.0, path_conf=1.0,
                       seq=7, e2e_policy=_E2E())
    assert tick.applied.reason == "heartbeat_stale" and tick.should_disengage and tick.applied.brake > 0
    # Wiring: the stale check sets heartbeat_ok before shadow_tick, and run_vision's heartbeat_stale branch disengages.
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    i_obs = src.index("ego_stale = stale_ego.observe(vdata)")
    i_hb = src.index("            heartbeat_ok = True\n            if stale_ego.blocks_drive(engaged):")
    i_set = src.index("                heartbeat_ok = False", i_hb)
    i_tick = src.index("tick = shadow_tick(", i_hb)
    assert i_obs < i_hb < i_set < i_tick
    assert src.count("stale_ego.observe(") == 1 and "if cam_age" not in src[i_obs - 60:i_obs]  # counted on every tick
    print("stale ego: coalesced / skipped polls are stale; engaged, the 2nd stale tick in a row sets heartbeat_ok=False "
          "-> the existing heartbeat_stale stop + disengage; disengaged it only counts OK")


# ---- runtime / window / grader -----------------------------------------------------


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
        assert set(rt.last_timers) == {"prep_ms", "scene_ms", "post_ms"} and rt.last_timers["scene_ms"] > 0
    print("runtime: gvd_cam_age=1 accepted on BeamNG, refused on window; window ages main-only; prep/scene/post timers OK")


def _capture_lines(*, hold: bool, n: int = 900, flash_every: int = 0, jump_every: int = 0, skip_at: int = -1,
                   dt_hi: bool = False, scene_hi: bool = False, engaged_at: int = -1, untrained: bool = True,
                   stale_at: tuple[int, ...] = (), no_detect_at: int = -1, detector: str = "yolov8n-onnx") -> list[str]:
    rng = np.random.RandomState(0)
    lines = [
        f"[GVD] cam-age model gvd_cam_age={1 if hold else 0} cams={8 if hold else 1} capture_untrained={int(untrained) if hold else 0} "
        f"hold={'on' if hold else 'off'} engage_disabled=1",
        "[GVD] cam-age providers scene=CPUExecutionProvider",
        f"[GVD] cam-age providers detector={detector}:CUDAExecutionProvider+CPUExecutionProvider",
    ]
    for k in range(n):
        ages = ",".join(
            f"{cam}:0.000" if cam in FRONT_CAMS else (f"{cam}:{(k % 8) / 15.0:.3f}" if hold else f"{cam}:-1")
            for cam in CAM_AGE_ORDER
        )
        luma = 0.4 + (0.2 if jump_every and k % jump_every == 0 else 0.0)
        flash = int(bool(flash_every) and k % flash_every == 0)
        dt = (230 if dt_hi else 140) + rng.rand() * 20
        scene = (200 if scene_hi else 40) + rng.rand() * 10
        det = "0.000" if k == no_detect_at else f"{8 + rng.rand():.3f}"
        lines.append(
            f"[GVD] cam_age t={1000 + k / 15.0:.3f} tick_ms={60 + rng.rand() * 10:.2f} loop_dt_ms={dt:.2f} hold={int(hold)} "
            f"baseline={int(not hold)} engaged={int(k >= engaged_at >= 0)} engage_disabled=1 would_skip={int(k == skip_at)} "
            f"stale_ego={int(k in stale_at)} stale_ego_n=0 reset_n=0 main_luma={luma:.4f} main_flash={flash} "
            f"compose_ms=1.200 prep_ms=2.000 scene_ms={scene:.3f} post_ms=0.300 detect_ms={det} plan_ms=1.500 "
            f"infer_ms=12.000 aeb=off ages={ages}"
        )
    return lines


def check_capture_script() -> None:
    from live_age_capture import DT_P50_MAX_MS, DT_P95_MAX_MS, grade, parse_log, table

    assert (DT_P50_MAX_MS, DT_P95_MAX_MS) == (162.0, 215.0)
    train = {cam: {"median": 0.2, "p95": 0.467} for cam in CAM_AGE_ORDER if cam != "main"}
    for cam in ("narrow", "wide"):
        train[cam] = {"median": 0.0, "p95": 0.067}
    base = parse_log(_capture_lines(hold=False, flash_every=100, jump_every=100))
    on = parse_log(_capture_lines(hold=True, flash_every=100, jump_every=100))
    result = grade(on, base, train)
    assert result["pass"], table(result)
    names = set(result["bars"])
    assert names == {"model", "engage_off", "mode", "duration", "compute", "ages", "loop_dt", "scene_ms",
                     "would_skip", "stale_ego", "main_flash", "main_luma_jump"}, names
    assert "OVERALL PASS" in table(result)
    good = dict(flash_every=100, jump_every=100)
    failures = {
        "ages": (dict(hold=True, **good), {cam: {"median": 0.6, "p95": 0.9} for cam in train}),
        "loop_dt": (dict(hold=True, dt_hi=True, **good), train),
        "scene_ms": (dict(hold=True, scene_hi=True, **good), train),
        "would_skip": (dict(hold=True, skip_at=300, **good), train),
        "main_flash": (dict(hold=True, flash_every=50, jump_every=100), train),
        "main_luma_jump": (dict(hold=True, flash_every=100, jump_every=40), train),
        "model": (dict(hold=True, untrained=False, **good), train),
        "compute": (dict(hold=True, no_detect_at=17, **good), train),
    }
    for bar, (kw, tr) in failures.items():
        got = grade(parse_log(_capture_lines(**kw)), base, tr)
        assert not got["pass"] and not got["bars"][bar]["pass"], (bar, table(got))
        others = [name for name, b in got["bars"].items() if not b["pass"] and name != bar]
        assert not others, (bar, others)
    # stale ego: counted while disengaged (PASS, reported); engaged past one tick FAILs (with engage_off).
    got = grade(parse_log(_capture_lines(hold=True, stale_at=(5, 6, 7), **good)), base, train)
    assert got["bars"]["stale_ego"]["pass"] and "on=3/900" in got["bars"]["stale_ego"]["detail"]
    got = grade(parse_log(_capture_lines(hold=True, stale_at=(500, 501), engaged_at=400, **good)), base, train)
    assert not got["bars"]["stale_ego"]["pass"] and not got["bars"]["engage_off"]["pass"]
    # a different detector fails compute; swapped logs fail mode; short logs fail duration; looser dt bars pass.
    assert not grade(parse_log(_capture_lines(hold=True, detector="yolox-onnx", **good)), base, train)["bars"]["compute"]["pass"]
    assert not grade(on, on, train)["bars"]["mode"]["pass"]
    assert not grade(parse_log(_capture_lines(hold=True, n=300, **good)), base, train)["bars"]["duration"]["pass"]
    assert grade(parse_log(_capture_lines(hold=True, dt_hi=True, **good)), base, train, dt_p50_ms=300, dt_p95_ms=400)["pass"]
    print(f"live_age_capture: {len(names)} bars PASS on a good pair; each FAILs alone on its fault; dt bars 162 / 215 ms OK")


def check_telemetry_helpers() -> None:
    from python.sensors.live_hold import main_exposure

    rng = np.random.RandomState(1)
    dim = rng.randint(40, 90, (192, 256, 3)).astype(np.uint8)
    luma, flash = main_exposure(dim, None)
    want = (0.114 * dim[..., 0].mean() + 0.587 * dim[..., 1].mean() + 0.299 * dim[..., 2].mean()) / 255.0
    assert abs(luma - want) < 1e-6 and not flash
    blue = dim.astype(np.int32) + np.array([120, 50, 30])
    assert main_exposure(np.clip(blue, 0, 255).astype(np.uint8), dim)[1] is True
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "would_skip_control = bool(cam_age_log) and camera_ge_socket_busy()\n            if engaged and (cmd.reason" in src
    assert "if args.no_engage:\n                engaged = False" in src
    for key in ("compose_ms=", "prep_ms=", "scene_ms=", "post_ms=", "detect_ms=", "plan_ms=", "stale_ego=", "aeb="):
        assert key in src, key
    # Engage gates actuation only: perception (detector + planner + AEB) runs before and outside any engaged branch.
    i_perc = src.index("            pout = perc.tick(\n                main,")
    assert "if engaged" not in src[src.rindex("\n", 0, i_perc - 1):i_perc]
    from python.perception.pipeline import PerceptionOut

    assert {"detect_ms", "plan_ms"} <= set(PerceptionOut.__dataclass_fields__)
    print("telemetry: main luma / flash, timers, stale ego, would-skip; perception runs every tick OK")


def check_perception_timers() -> None:
    from python.perception.pipeline import ModularPerception

    perc = ModularPerception(allow_synthetic=True)
    frame = np.full((192, 256, 3), 90, dtype=np.uint8)
    for k in range(3):
        out = perc.tick(frame, stitch_bgr=None, ego_speed_mps=5.0, steer_deg=0.0, yaw_rate=0.0, dt_s=1.0 / 15.0)
        assert out.detect_ms > 0.0 and out.plan_ms > 0.0 and out.infer_ms >= out.detect_ms + out.plan_ms - 1e-6
        assert "aeb" in out.planner
    print(f"perception: detect_ms {out.detect_ms:.3f} / plan_ms {out.plan_ms:.3f} on every tick, aeb in planner OK")


def main() -> None:
    check_recorder_schedule()
    check_wall_time_ages()
    check_backend_streams()
    check_respawn_reset()
    check_reset_detection()
    check_stale_ego()
    check_runtime_gate()
    check_capture_script()
    check_telemetry_helpers()
    check_perception_timers()
    print("test_live_hold: all OK")


if __name__ == "__main__":
    main()
