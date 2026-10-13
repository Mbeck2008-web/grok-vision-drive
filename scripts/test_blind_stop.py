#!/usr/bin/env python3
"""A dark or stale camera must not keep driving on remembered lanes.

No BeamNG. Lane fit is stubbed so only the pipeline, the planner, and the
shadow gate run. Time is injected; nothing sleeps.
"""
from __future__ import annotations

import sys
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
    return LaneResult(conf=0.9, lanes_bev=[_line(-1.8), _line(1.8)], curvature=0.0)


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


def main() -> None:
    check_unchanged_buffer_is_missing()
    check_window_stuck_buffer()
    check_live_camera_lost_is_not_ignored()
    check_blind_hold_keeps_tech_brake()
    check_dark_camera_brakes_then_disengages()
    print("test_blind_stop: OK")


if __name__ == "__main__":
    main()
