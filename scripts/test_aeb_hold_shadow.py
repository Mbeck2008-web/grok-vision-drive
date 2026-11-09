#!/usr/bin/env python3
"""AEB on shadow/e2e must hold the brake, not release after disengage.

On --policy shadow and --policy e2e, evaluate_veto returns aeb_brake and
shadow_tick sets should_disengage with stop_command (brake 1). The supervisor
used to clear Engage and then call actuator.stop(), which releases the pedals
while not engaged. Modular keeps brake 1 because it never takes that path.

    py -3 scripts/test_aeb_hold_shadow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    BeamNGPyActuator,
    CmdJsonActuator,
    DriveCommand,
    latch_aeb_park_hold,
)
from python.control.e2e import E2EIntent  # noqa: E402
from python.runtime.shadow import ShadowConfig, shadow_tick  # noqa: E402


class _Veh:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def control(self, **kw) -> None:
        self.calls.append(kw)

    def set_shift_mode(self, mode: str) -> None:
        del mode


class _FixedE2E:
    def forward(self, *args, **kwargs) -> E2EIntent:
        del args, kwargs
        return E2EIntent(steer=0.31, throttle=0.4, brake=0.0, accel=0.4, ok=True, backend="onnx")


def _path() -> list[dict[str, float]]:
    return [{"x": 0.0, "y": float(i)} for i in range(12)]


def _aeb_tick(policy: str):
    return shadow_tick(
        policy=policy,
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10.0, "aeb": "brake"},
        ego_speed_mps=12.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=6,
        e2e_policy=_FixedE2E(),
        main_bgr=None,
        wide_bgr=None,
        cfg=ShadowConfig(),
    )


def _handoff(act, tick):
    """The supervisor order: latch AEB hold, drop Engage, then stop()."""
    latch_aeb_park_hold(act, tick.veto_reason)
    engaged = False if tick.should_disengage else True
    act.note_engaged(engaged)
    if engaged and tick.applied.reason == "ok":
        return act.apply(tick.applied)
    return act.stop(seq=tick.applied.seq, reason=tick.applied.reason)


def check_shadow_e2e_command_is_a_hold() -> None:
    for policy in ("shadow", "e2e"):
        tick = _aeb_tick(policy)
        assert tick.veto_reason == "aeb_brake", (policy, tick.veto_reason)
        assert tick.applied.brake == 1.0 and tick.applied.throttle == 0.0, tick.applied
        assert tick.applied.steer == 0.0, tick.applied
        assert tick.applied.reason.startswith("veto:"), tick.applied.reason
        print(f"{policy}: AEB command is a full hold (no steer/throttle) OK")


def check_tech_stop_after_aeb_disengage_holds() -> None:
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "latch_aeb_park_hold" in src, "run_vision must latch the AEB hold before dropping Engage"

    for policy in ("shadow", "e2e"):
        tick = _aeb_tick(policy)
        veh = _Veh()
        act = BeamNGPyActuator(veh)
        act.note_engaged(True)
        act.apply(DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=1, reason="ok"))
        _handoff(act, tick)
        last = veh.calls[-1]
        assert float(last.get("throttle") or 0.0) == 0.0, last
        assert float(last.get("parkingbrake") or 0.0) == 1.0, (policy, last)
        assert float(last.get("brake") or 0.0) == 0.0, last
        # Later not_engaged ticks must keep the hold, not release.
        act.stop(seq=tick.applied.seq + 1, reason="not_engaged")
        later = veh.calls[-1]
        assert float(later.get("parkingbrake") or 0.0) == 1.0, (policy, later)
        print(f"{policy}: Tech AEB handoff holds, does not release OK")


def check_cmd_json_aeb_sets_hold_park() -> None:
    tick = _aeb_tick("shadow")
    act = CmdJsonActuator()
    act.note_engaged(True)
    latch_aeb_park_hold(act, tick.veto_reason)
    act.note_engaged(False)
    act.stop(seq=tick.applied.seq, reason=tick.applied.reason)
    path = __import__("python.control.actuate", fromlist=["cmd_path"]).cmd_path()
    payload = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
    assert payload.get("hold_park") is True, payload
    assert float(payload.get("brake") or 0.0) == 1.0, payload
    assert payload.get("engaged") is False, payload
    print("cmd_json: AEB handoff writes hold_park with brake 1 OK")


def main() -> None:
    check_shadow_e2e_command_is_a_hold()
    check_tech_stop_after_aeb_disengage_holds()
    check_cmd_json_aeb_sets_hold_park()
    print("test_aeb_hold_shadow: OK")


if __name__ == "__main__":
    main()
