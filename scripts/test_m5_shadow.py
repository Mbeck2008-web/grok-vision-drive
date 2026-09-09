#!/usr/bin/env python3
"""Offline M5 shadow / E2E checks (Spec verify)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.e2e import E2E_H, E2E_W, E2EIntent, make_e2e
from python.runtime.shadow import ShadowConfig, shadow_tick


class _FixedE2E:
    """Deterministic E2E stand-in so disagreement asserts always fire."""

    def __init__(self, steer: float = 0.0, accel: float = 0.0) -> None:
        self.steer = float(steer)
        self.accel = float(accel)
        self.backend = "stub"

    def forward(self, *args, **kwargs) -> E2EIntent:
        a = self.accel
        thr = a if a >= 0 else 0.0
        brk = -a if a < 0 else 0.0
        return E2EIntent(
            steer=self.steer,
            accel=a,
            throttle=thr,
            brake=brk,
            ok=True,
            backend="stub",
            reason="fixed",
        )


def _path(n: int = 12, x: float = 0.1):
    return [{"x": x, "y": float(i), "z": 0.0} for i in range(n)]


def main() -> None:
    e2e = make_e2e()
    main = np.zeros((240, 320, 3), dtype=np.uint8)
    wide = np.zeros((240, 320, 3), dtype=np.uint8)
    main[:, :, 1] = 40
    wide[:, :, 2] = 40

    # stub forward no crash
    out = e2e.forward(main, wide, speed_mps=8.0, steer_deg=2.0)
    assert out.ok, out
    assert -1.0 <= out.steer <= 1.0
    assert -1.0 <= out.accel <= 1.0
    assert e2e.backend in ("stub", "onnx")

    cfg = ShadowConfig(lane_conf_min=0.25, steer_disagree_max=0.55, path_conf_min=0.15)

    # disengaged → no actuate; shadow fields still written
    tick = shadow_tick(
        policy="shadow",
        engaged=False,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=1,
        e2e_policy=e2e,
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick.applied.reason == "not_engaged"
    assert tick.applied.throttle == 0.0 and tick.applied.brake == 1.0
    assert "steer" in tick.shadow and "throttle" in tick.shadow and "brake" in tick.shadow

    # veto low conf → no e2e apply (engaged e2e policy)
    tick2 = shadow_tick(
        policy="e2e",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=0.05,  # below min
        path_conf=0.8,
        seq=2,
        e2e_policy=e2e,
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick2.veto_reason == "low_lane_conf", tick2.veto_reason
    assert tick2.e2e_ok is False
    assert tick2.applied.throttle == 0.0 and tick2.applied.brake == 1.0
    assert tick2.applied.reason.startswith("veto:")
    assert tick2.should_disengage is True

    # heartbeat veto
    tick3 = shadow_tick(
        policy="e2e",
        engaged=True,
        heartbeat_ok=False,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=3,
        e2e_policy=e2e,
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick3.veto_reason == "heartbeat_stale"
    assert tick3.applied.reason == "heartbeat_stale"

    # disagreement veto: fixed E2E steer=0 vs modular steer~1 from skewed path — always assert
    tick4 = shadow_tick(
        policy="e2e",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(x=3.0),  # large lateral → modular steer ~1.0
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=4,
        e2e_policy=_FixedE2E(steer=0.0),
        main_bgr=main,
        wide_bgr=wide,
        cfg=ShadowConfig(lane_conf_min=0.25, steer_disagree_max=0.05, path_conf_min=0.15),
    )
    assert abs(tick4.modular.steer - tick4.e2e.steer) > 0.05, (
        tick4.modular.steer,
        tick4.e2e.steer,
    )
    assert tick4.veto_reason == "disagreement", tick4.veto_reason
    assert tick4.e2e_ok is False
    assert tick4.applied.throttle == 0.0 and tick4.applied.brake == 1.0
    assert tick4.applied.reason.startswith("veto:")

    # engaged shadow + healthy → apply modular (not crash); shadow written
    tick5 = shadow_tick(
        policy="shadow",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(x=0.0),
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=5,
        e2e_policy=e2e,
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick5.veto_reason == "none"
    assert tick5.applied.reason == "ok"
    assert tick5.shadow["steer"] == tick5.e2e.steer

    # AEB brake → veto E2E (do not apply E2E actuators)
    tick_aeb = shadow_tick(
        policy="e2e",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(x=0.0),
        planner={"target_v": 10, "aeb": "brake"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=6,
        e2e_policy=_FixedE2E(steer=0.1, accel=0.5),
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick_aeb.veto_reason == "aeb_brake", tick_aeb.veto_reason
    assert tick_aeb.e2e_ok is False
    assert tick_aeb.applied.throttle == 0.0 and tick_aeb.applied.brake == 1.0
    assert tick_aeb.applied.reason.startswith("veto:")
    assert tick_aeb.should_disengage is True

    # AEB warn → also veto E2E
    tick_warn = shadow_tick(
        policy="e2e",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(x=0.0),
        planner={"target_v": 10, "aeb": "warn"},
        ego_speed_mps=5.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=7,
        e2e_policy=_FixedE2E(steer=0.1, accel=0.5),
        main_bgr=main,
        wide_bgr=wide,
        cfg=cfg,
    )
    assert tick_warn.veto_reason == "aeb_warn", tick_warn.veto_reason
    assert tick_warn.e2e_ok is False
    assert tick_warn.applied.throttle == 0.0 and tick_warn.applied.brake == 1.0
    assert tick_warn.applied.reason.startswith("veto:")

    # train module imports without weights
    from python.train.train_e2e import smoke_import

    info = smoke_import()
    assert info["ok"] is True

    # dims
    assert E2E_W == 320 and E2E_H == 180

    print("test_m5_shadow: OK")


if __name__ == "__main__":
    main()
