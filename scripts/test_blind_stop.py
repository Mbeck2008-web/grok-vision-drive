#!/usr/bin/env python3
"""A dark or stale camera must not keep driving on remembered lanes.

No BeamNG. Lane fit is stubbed so only the pipeline, the planner, and the
shadow gate run. Time is injected; nothing sleeps.
"""
from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.e2e import make_e2e  # noqa: E402
from python.perception.lanes import LaneResult  # noqa: E402
from python.perception.pipeline import PRIOR_LANE_MAX_S, ModularPerception  # noqa: E402
import python.perception.pipeline as pipeline  # noqa: E402
from python.runtime.shadow import (  # noqa: E402
    CAMERA_LOSS_DISENGAGE_S,
    WINDSHIELD_RECOVER_S,
    VisionLoss,
    hold_for_vision_loss,
    shadow_tick,
    vision_loss_reason,
)
from python.sensors.cameras import (  # noqa: E402
    CREEP_MPS,
    BeamNGPyBackend,
    CamHealth,
    WindowBackend,
    note_grab_ego_speed,
)
from python.viz.debug_draw import cabin_drive_word  # noqa: E402


def _line(x: float) -> list[dict[str, float]]:
    return [{"x": x, "y": float(y)} for y in (0, 10, 20, 36)]


def _live() -> LaneResult:
    lanes = [_line(-1.8), _line(1.8)]
    return LaneResult(conf=0.9, lanes_bev=lanes, curvature=0.0, forward_n=len(lanes))


def _blank() -> LaneResult:
    return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)


def _drive(planner: dict, *, ego: float, age: float | None, seq: int):
    return shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=True,
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner=planner,
        ego_speed_mps=ego,
        lane_conf=0.9,
        path_conf=0.8,
        seq=seq,
        e2e_policy=make_e2e(),
        main_bgr=None,
        cfg=None,
        vision_loss_age_s=age,
    )


def check_unchanged_buffer_is_missing() -> None:
    be = BeamNGPyBackend.__new__(BeamNGPyBackend)
    be.long_side = 64
    be._cache_frames = {}
    be._cache_ts = {}
    be._frame_sig = {}
    be._stuck_n = {}
    be._good_ts = {}
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[2, 2] = (10, 20, 30)
    note_grab_ego_speed(0.0)
    frames: dict = {}
    health: dict = {}
    ids: list[str] = []
    be._publish_read("main", img, 1.0, frames, {}, health, ids)
    assert "main" in frames and "main" in ids and health["main"] == CamHealth.OK
    # One repeat is still the frame. It is not a unique GPU id.
    note_grab_ego_speed(12.0)
    frames2: dict = {}
    health2: dict = {}
    ids2: list[str] = []
    be._publish_read("main", img.copy(), 1.1, frames2, {}, health2, ids2)
    assert "main" in frames2 and "cam_main" in frames2
    assert "main" not in ids2
    assert health2["main"] == CamHealth.OK
    # Still inside the prior window, even after several grabs.
    for i, ts in enumerate((1.2, 1.4, 1.6)):
        held: dict = {}
        held_h: dict = {}
        be._publish_read("main", img.copy(), ts, held, {}, held_h, [])
        assert "main" in held and held_h["main"] == CamHealth.OK, (i, ts)
    # Moving, stuck, and older than the prior window: this sector is empty.
    assert 1.0 + PRIOR_LANE_MAX_S < 1.0 + PRIOR_LANE_MAX_S + 0.05
    lost: dict = {}
    lost_h: dict = {}
    lost_ids: list[str] = []
    be._publish_read("main", img.copy(), 1.0 + PRIOR_LANE_MAX_S + 0.05, lost, {}, lost_h, lost_ids)
    assert "main" not in lost and "cam_main" not in lost
    assert "main" not in lost_ids
    assert lost_h["main"] == CamHealth.MISSING
    # Stopped traffic does not turn a stuck buffer into a loss.
    note_grab_ego_speed(CREEP_MPS)
    parked: dict = {}
    parked_h: dict = {}
    be._publish_read("main", img.copy(), 3.0, parked, {}, parked_h, [])
    assert "main" in parked and parked_h["main"] == CamHealth.OK
    # All-zero is unrendered, not a stuck picture.
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    empty: dict = {}
    empty_h: dict = {}
    be._publish_read("main", blank, 4.0, empty, {}, empty_h, [])
    assert "main" not in empty and empty_h["main"] == CamHealth.MISSING
    nxt = img.copy()
    nxt[2, 3] = (11, 21, 31)
    frames3: dict = {}
    ids3: list[str] = []
    be._publish_read("main", nxt, 5.0, frames3, {}, {}, ids3)
    assert "main" in frames3 and "main" in ids3


def check_window_stuck_buffer() -> None:
    """Retail window capture uses the same stuck-frame rule as Tech."""
    win = WindowBackend.__new__(WindowBackend)
    win.long_side = 32
    win._logged = True
    win._region = {"left": 0, "top": 0, "width": 16, "height": 16}
    win._title_match = True
    win._note = "test"
    win._impl = "stub"
    win._mss = None
    win._frame_sig = {}
    win._stuck_n = {}
    win._good_ts = {}
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[1, 1] = (8, 9, 10)

    class Cam:
        def __init__(self) -> None:
            self.img = img

        def grab(self):
            return self.img.copy()

    win._cam = Cam()
    win._apply_window_rect = lambda _rect: None  # type: ignore[method-assign]
    note_grab_ego_speed(0.0)
    first = win.grab()
    assert first.health["main"] == CamHealth.OK
    assert "main" in first.frames
    assert "main" in first.unique_gpu_ids
    note_grab_ego_speed(12.0)
    second = win.grab()
    assert second.health["main"] == CamHealth.OK
    assert "main" in second.frames
    assert "main" not in second.unique_gpu_ids
    for _ in range(3):
        again = win.grab()
        assert again.health["main"] == CamHealth.OK
    win._good_ts["main"] = __import__("time").time() - (PRIOR_LANE_MAX_S + 0.2)
    lost = win.grab()
    assert lost.health["main"] == CamHealth.MISSING
    assert "main" not in lost.frames
    note_grab_ego_speed(0.4)
    kept = win.grab()
    assert kept.health["main"] == CamHealth.OK
    assert "main" in kept.frames
    win._cam.img = np.zeros((16, 16, 3), dtype=np.uint8)
    blank = win.grab()
    assert blank.health["main"] == CamHealth.MISSING
    assert blank.frames == {}


def check_live_camera_lost_is_not_ignored() -> None:
    assert vision_loss_reason({"lanes_source": "live", "camera_lost": True}) == "camera_lost"
    assert vision_loss_reason({"lanes_source": "live", "camera_lost": False}) is None
    assert vision_loss_reason({"lanes_source": "unseen", "camera_lost": True}) is None
    assert vision_loss_reason({"lanes_source": "unseen"}) is None
    assert vision_loss_reason({"lanes_source": "prior"}) == "lanes_lost"
    assert vision_loss_reason({"lanes_source": "none", "camera_lost": True}) == "camera_lost"


def check_blind_hold_keeps_tech_brake() -> None:
    """Two ticks: moving service brake, then parking brake. Release only after stop.

    Tech ``hold`` calls ``vehicle.control`` with ``release=False`` while Engage is off.
    """
    from python.control.actuate import BeamNGPyActuator, stop_command
    from python.runtime.shadow import blind_brake_command as _blind

    class Veh:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def control(self, **kw):
            self.calls.append(kw)

    loss = VisionLoss()
    loss.note_disengage()
    assert loss.dropped is True
    moving, hold = hold_for_vision_loss(
        stop_command(seq=1, reason="not_engaged"), ego_speed_mps=12.0, seq=1
    )
    assert hold is True
    assert moving.reason == "camera_lost"
    assert abs(moving.brake - 0.6) < 1e-9 and moving.throttle == 0.0 and moving.steer == 0.0
    veh = Veh()
    act = BeamNGPyActuator(veh)
    act.note_engaged(False)
    sent = act.hold(moving)
    assert sent.applied is True
    kw = veh.calls[-1]
    assert kw["throttle"] == 0.0 and kw["steering"] == 0.0
    assert abs(kw["brake"] - 0.6) < 1e-6
    assert kw["parkingbrake"] == 0.0
    assert not (kw["brake"] == 0.0 and kw["parkingbrake"] == 0.0)
    creeping, hold2 = hold_for_vision_loss(moving, ego_speed_mps=0.4, seq=2)
    assert hold2 is True and creeping.brake >= 0.99
    sent2 = act.hold(creeping)
    assert sent2.applied is True
    kw2 = veh.calls[-1]
    assert kw2["parkingbrake"] == 1.0 and kw2["brake"] == 0.0 and kw2["throttle"] == 0.0
    stopped, hold3 = hold_for_vision_loss(creeping, ego_speed_mps=0.0, seq=3)
    assert hold3 is False
    released = act.stop(seq=3, reason="camera_lost")
    assert released.brake == 0.0
    kw3 = veh.calls[-1]
    assert kw3["brake"] == 0.0 and kw3["parkingbrake"] == 0.0 and kw3["throttle"] == 0.0
    # The helper used above is the same brake the shadow gate builds.
    law = _blind(ego_speed_mps=12.0, seq=1, reason="camera_lost")
    assert abs(law.brake - moving.brake) < 1e-9


def check_dark_camera_brakes_then_disengages() -> None:
    mode = {"live": True}

    def fake(_main, _stitch=None):
        return _live() if mode["live"] else _blank()

    old = pipeline.lanes_from_view
    pipeline.lanes_from_view = fake
    try:
        perc = ModularPerception(allow_synthetic=False, detector_id="empty")
        picture = np.zeros((8, 8, 3), dtype=np.uint8)
        picture[0, 0] = (30, 30, 30)
        live = perc.tick(picture, now=0.0, ego_speed_mps=5.0)
        assert live.planner["lanes_source"] == "live", live.planner
        assert float(live.planner["target_v"]) > 5.0, live.planner
        driving = _drive(live.planner, ego=5.0, age=None, seq=1)
        assert driving.applied.throttle > 0.0 and driving.applied.brake == 0.0, driving.applied
        assert driving.should_disengage is False

        mode["live"] = False
        clock = VisionLoss()
        # Still inside the prior window: the plan remembers the lanes, the car does not.
        remembered = perc.tick(None, now=0.2, ego_speed_mps=18.0)
        assert remembered.planner["lanes_source"] == "prior", remembered.planner
        assert remembered.planner["prediction"] == "context", remembered.planner
        assert remembered.planner["camera_lost"] is True
        assert float(remembered.planner["target_v"]) > 5.0, remembered.planner
        age = clock.age("prior", 0.2)
        assert age == 0.0
        braking = _drive(remembered.planner, ego=18.0, age=age, seq=2)
        assert braking.applied.reason == "camera_lost", braking.applied
        assert braking.applied.throttle == 0.0 and braking.applied.steer == 0.0, braking.applied
        assert abs(braking.applied.brake - 0.6) < 1e-9, braking.applied
        assert braking.applied.brake < 1.0
        assert braking.should_disengage is False
        assert cabin_drive_word({"engaged": True, "cmd_reason": "camera_lost"})[0] == "HOLD"

        # Brake is already on. Disengage once the loss has lasted 1 s, still under 2 s.
        assert CAMERA_LOSS_DISENGAGE_S <= 1.0
        later = perc.tick(None, now=0.2 + 0.9, ego_speed_mps=18.0)
        age_later = clock.age(str(later.planner["lanes_source"]), 0.2 + 0.9)
        assert age_later is not None and age_later < CAMERA_LOSS_DISENGAGE_S
        hold = _drive(later.planner, ego=18.0, age=age_later, seq=3)
        assert hold.should_disengage is False and hold.applied.brake == 0.6

        drop_t = 0.2 + CAMERA_LOSS_DISENGAGE_S
        assert drop_t - 0.2 <= 2.0
        dropped = perc.tick(None, now=drop_t, ego_speed_mps=18.0)
        # Priors expire at 0.75 s, so this tick is "none", not a forever context plan.
        assert drop_t > PRIOR_LANE_MAX_S
        assert dropped.planner["lanes_source"] == "none", dropped.planner
        assert dropped.planner["prediction"] == "none", dropped.planner
        assert float(dropped.planner["target_v"]) == 0.0, dropped.planner
        age_drop = clock.age("none", drop_t)
        assert age_drop is not None and age_drop >= CAMERA_LOSS_DISENGAGE_S
        done = _drive(dropped.planner, ego=18.0, age=age_drop, seq=4)
        assert done.should_disengage is True
        assert done.veto_reason == "camera_lost"
        assert done.applied.reason == "camera_lost"
        assert done.applied.throttle == 0.0 and abs(done.applied.brake - 0.6) < 1e-9

        # Lanes gone, picture still there: lanes_lost, same brake, not camera_lost.
        fit_fail = perc.tick(picture, now=drop_t, ego_speed_mps=18.0)
        assert fit_fail.planner["camera_lost"] is False
        assert fit_fail.planner["lanes_source"] == "none"
        named = _drive(fit_fail.planner, ego=18.0, age=age_drop, seq=5)
        assert named.applied.reason == "lanes_lost", named.applied
        assert named.should_disengage is True
        assert named.applied.brake < 1.0

        # A backend that has never seen lanes is not this gate.
        fresh = ModularPerception(allow_synthetic=False, detector_id="empty")
        unseen = fresh.tick(None, now=0.0)
        assert unseen.planner["lanes_source"] == "unseen", unseen.planner
        plain = _drive(unseen.planner, ego=5.5, age=None, seq=6)
        assert plain.applied.reason != "camera_lost"
        assert plain.should_disengage is False
    finally:
        pipeline.lanes_from_view = old


def check_side_lanes_are_not_a_recovery() -> None:
    """A live lanes_source from side pose lines does not end a dark episode.

    Recovery is a healthy windshield, held for ``WINDSHIELD_RECOVER_S``.
    Clearing the loss clock does not clear the brake hold.
    """
    clock = VisionLoss()
    clock.note_disengage()
    assert clock.dropped and clock.hold_until_stop
    age = clock.age("live", 10.0, camera_lost=True, windshield_live=False)
    assert clock.dropped and clock.hold_until_stop
    assert age == 0.0
    # One healthy windshield frame is still inside the recover window.
    held = clock.age("live", 10.2, camera_lost=False, windshield_live=True)
    assert clock.dropped and held is not None
    assert WINDSHIELD_RECOVER_S >= 0.5
    done = clock.age("live", 10.2 + WINDSHIELD_RECOVER_S, camera_lost=False, windshield_live=True)
    assert done is None and clock.dropped is False
    assert clock.hold_until_stop is True

    def posed(_main, _stitch=None):
        return LaneResult(conf=0.8, lanes_bev=[_line(6.0)], curvature=0.0, forward_n=0)

    def live_fit(_main, _stitch=None):
        return _live()

    old = pipeline.lanes_from_view
    pipeline.lanes_from_view = live_fit
    try:
        perc = ModularPerception(allow_synthetic=False, detector_id="empty")
        picture = np.zeros((8, 8, 3), dtype=np.uint8)
        picture[0, 0] = (30, 30, 30)
        live = perc.tick(picture, now=0.0, ego_speed_mps=12.0)
        assert live.planner["lanes_source"] == "live", live.planner
        assert live.planner["windshield_live"] is True
        assert live.planner["camera_lost"] is False
        pipeline.lanes_from_view = posed
        dark = perc.tick(None, now=0.1, ego_speed_mps=12.0)
        assert dark.planner["lanes_source"] != "live", dark.planner
        assert dark.planner["lanes_source"] == "prior", dark.planner
        assert dark.planner["camera_lost"] is True
        assert dark.planner["windshield_live"] is False
    finally:
        pipeline.lanes_from_view = old


def check_stale_heartbeat_dark_camera_brakes() -> None:
    """A stale stamp while the camera is dark brakes. It does not release.

    Stopped, the heartbeat gate may still stop. Moving, blind braking wins.
    """
    planner = {
        "lanes_source": "live",
        "camera_lost": True,
        "windshield_live": False,
        "target_v": 20.0,
        "aeb": "off",
    }
    moving = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=False,
        path_debug_preview=False,
        allow_preview_drive=True,
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner=planner,
        ego_speed_mps=12.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=1,
        e2e_policy=make_e2e(),
        main_bgr=None,
        cfg=None,
        vision_loss_age_s=0.2,
    )
    assert moving.applied.reason == "camera_lost", moving.applied
    assert moving.applied.brake > 0.3 and moving.applied.throttle == 0.0
    assert moving.should_disengage is True
    assert moving.veto_reason == "camera_lost"
    stopped = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=False,
        path_debug_preview=False,
        allow_preview_drive=True,
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner=planner,
        ego_speed_mps=0.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=2,
        e2e_policy=make_e2e(),
        main_bgr=None,
        cfg=None,
        vision_loss_age_s=0.2,
    )
    assert stopped.applied.reason == "heartbeat_stale", stopped.applied
    assert stopped.should_disengage is True


class _TickScript:
    """Shared grab clock for one ``run_vision.main`` pass."""

    n = 0
    phase = "boot"
    speed = 0.0
    tech = False
    dark_mono: float | None = None
    rows: list[tuple] = []


def _fresh_bus(speed: float) -> None:
    from python.control.actuate import ego_path
    from python.runtime import paths as gvd_paths

    pin = gvd_paths.predicted_gvd_docs_path()
    pin.mkdir(parents=True, exist_ok=True)
    folder = pin if pin.is_absolute() else pin.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"lua_bus": str(folder)})
    (folder / "gvd_link.json").write_text(payload, encoding="utf-8")
    ego_path().write_text(
        json.dumps(
            {
                "speed_mps": speed,
                "steering_input": 0.0,
                "throttle_input": 0.0,
                "brake_input": 0.0,
                "applied_seq": 0,
                "applying": False,
                "player_device": False,
                "lua_bus": str(folder),
            }
        ),
        encoding="utf-8",
    )


def _run_supervisor_ticks(*, tech: bool) -> tuple[list[tuple], str]:
    """Drive ``run_vision.main``. Does not call ``hold`` or ``stop`` itself.

    Live windshield, then a dark main with a posed side line, then a stop.
    """
    import python.control.actuate as actuate
    import python.run_vision as rv
    from python.runtime.paths import bus_identity
    from python.sensors.cameras import CameraFrameBundle, CamHealth, StubBackend

    _TickScript.n = 0
    _TickScript.phase = "boot"
    _TickScript.speed = 0.0
    _TickScript.tech = tech
    _TickScript.dark_mono = None
    _TickScript.rows = []

    class _Veh:
        def __init__(self) -> None:
            self.electrics = {
                "wheelspeed": 12.0,
                "steering_input": 0.0,
                "throttle_input": 0.0,
                "brake_input": 0.0,
            }
            self.sensors = self

        def poll(self):
            return {"electrics": dict(self.electrics)}

        def control(self, **kw):
            _TickScript.rows.append(
                ("control", _TickScript.phase, _TickScript.speed, time.monotonic(), dict(kw))
            )

        def set_shift_mode(self, _mode):
            return None

        def queue_lua_command(self, _chunk, _response=False):
            return True

    orig_init = StubBackend.__init__
    orig_grab = StubBackend.grab
    orig_write = actuate.atomic_write_json
    orig_lanes = pipeline.lanes_from_view

    def init(self, long_side=None):
        orig_init(self, long_side)
        if _TickScript.tech:
            self.vehicle = _Veh()

    def grab(self):
        _TickScript.n += 1
        n = _TickScript.n
        if n <= 8:
            phase, speed = "live", 12.0
        elif n <= 50:
            phase, speed = "dark", 12.0
        else:
            phase, speed = "stop", 0.0
        if phase == "dark" and _TickScript.dark_mono is None:
            _TickScript.dark_mono = time.monotonic()
        _TickScript.phase = phase
        _TickScript.speed = speed
        veh = getattr(self, "vehicle", None)
        if veh is not None:
            veh.electrics["wheelspeed"] = speed
        _fresh_bus(speed)
        if phase == "live":
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[0, 0] = (40, 50, 60)
            stamp = time.time()
            return CameraFrameBundle(
                frames={"main": img},
                timestamps={"main": stamp},
                health={"main": CamHealth.OK},
                backend="stub",
                unique_gpu_n=1,
                unique_gpu_ids=("main",),
            )
        return CameraFrameBundle(
            frames={},
            timestamps={},
            health={"main": CamHealth.MISSING},
            backend="stub",
            note="main dark",
            unique_gpu_n=0,
        )

    def lanes(_main, _stitch=None):
        if _TickScript.phase == "live":
            return _live()
        # Side pose only. forward_n stays 0 so this is not a windshield recovery.
        return LaneResult(conf=0.7, lanes_bev=[_line(6.0)], curvature=0.0, forward_n=0)

    def write_json(path, payload, *args, **kwargs):
        ok = orig_write(path, payload, *args, **kwargs)
        if Path(path).name == "gvd_cmd.json" and isinstance(payload, dict):
            _TickScript.rows.append(
                ("cmd", _TickScript.phase, _TickScript.speed, time.monotonic(), dict(payload))
            )
        if Path(path).name == "gvd_engage.json" and isinstance(payload, dict):
            _TickScript.rows.append(
                ("engage", _TickScript.phase, _TickScript.speed, time.monotonic(), dict(payload))
            )
        return ok

    _fresh_bus(12.0)
    ident = bus_identity()
    assert ident.matched, ident.note
    StubBackend.__init__ = init
    StubBackend.grab = grab
    actuate.atomic_write_json = write_json
    pipeline.lanes_from_view = lanes
    argv = sys.argv
    buf = io.StringIO()
    sys.argv = [
        "run_vision",
        "--backend", "stub",
        "--hz", "20",
        "--max-ticks", "62",
        "--force-engage",
        "--allow-preview-drive",
        "--detector", "empty",
    ]
    try:
        with redirect_stdout(buf):
            rv.main()
    except BaseException as exc:
        tail = buf.getvalue()[-2500:]
        raise AssertionError(f"run_vision.main failed: {exc!r}\n{tail}") from exc
    finally:
        sys.argv = argv
        StubBackend.__init__ = orig_init
        StubBackend.grab = orig_grab
        actuate.atomic_write_json = orig_write
        pipeline.lanes_from_view = orig_lanes
    return list(_TickScript.rows), buf.getvalue()


def _late_rows(rows: list[tuple], kind: str) -> list[dict]:
    start = _TickScript.dark_mono
    assert start is not None, "dark phase never started"
    out = []
    for key, phase, speed, mono, payload in rows:
        if key != kind or phase != "dark" or speed <= 1.0:
            continue
        if mono - start >= 1.05:
            out.append(payload)
    return out


def check_supervisor_tick_holds_brake() -> None:
    """Retail and Tech supervisor ticks keep braking after the drop, while moving.

    The loop is ``run_vision.main``. Side lanes stay fitted and the main
    picture is gone. A stopped car is the only release.
    """
    retail, retail_log = _run_supervisor_ticks(tech=False)
    assert "DISENGAGED: camera_lost" in retail_log, retail_log[-1500:]
    late_cmd = _late_rows(retail, "cmd")
    assert len(late_cmd) >= 2, [(c.get("engaged"), c.get("brake"), c.get("reason")) for c in late_cmd]
    for cmd in late_cmd:
        assert cmd.get("engaged") is True, cmd
        assert float(cmd.get("brake") or 0.0) > 0.2, cmd
        assert float(cmd.get("throttle") or 0.0) == 0.0, cmd
        assert cmd.get("reason") in ("camera_lost", "lanes_lost"), cmd
    late_engage = _late_rows(retail, "engage")
    assert all(item.get("engaged") is not False for item in late_engage), late_engage
    # The car then stops. The release is allowed only on that phase.
    stopped_cmd = [p for k, phase, _s, _m, p in retail if k == "cmd" and phase == "stop"]
    assert stopped_cmd, "no command after the stop"
    assert any(p.get("engaged") is False for p in stopped_cmd), stopped_cmd[-1]

    tech, tech_log = _run_supervisor_ticks(tech=True)
    assert "DISENGAGED: camera_lost" in tech_log, tech_log[-1500:]
    late_ctl = _late_rows(tech, "control")
    assert len(late_ctl) >= 2, late_ctl
    for kw in late_ctl:
        brake = float(kw.get("brake") or 0.0)
        park = float(kw.get("parkingbrake") or 0.0)
        assert brake > 0.2 or park > 0.5, kw
        assert not (brake == 0.0 and park == 0.0), kw


def main() -> None:
    check_unchanged_buffer_is_missing()
    check_window_stuck_buffer()
    check_live_camera_lost_is_not_ignored()
    check_blind_hold_keeps_tech_brake()
    check_dark_camera_brakes_then_disengages()
    check_side_lanes_are_not_a_recovery()
    check_stale_heartbeat_dark_camera_brakes()
    check_supervisor_tick_holds_brake()
    print("test_blind_stop: OK")


if __name__ == "__main__":
    main()
