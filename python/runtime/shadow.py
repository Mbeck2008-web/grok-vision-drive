"""M5 shadow mode: always compute modular + e2e; actuate only when engaged + modular OK.

Default policy_modular = safety supervisor (may veto E2E).
Disengaged → no actuate; shadow fields still written.
Modular veto → hold/disengage; optional clip trigger via recorder.
Dead-man / heartbeat gates unchanged (caller passes heartbeat_ok).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python.control.actuate import DriveCommand, plan_command, stop_command
from python.control.e2e import E2EIntent, E2EPolicy


@dataclass
class ShadowConfig:
    lane_conf_min: float = 0.25
    steer_disagree_max: float = 0.55
    path_conf_min: float = 0.15


@dataclass
class ShadowTick:
    modular: DriveCommand
    e2e: E2EIntent
    applied: DriveCommand
    shadow: dict[str, float] = field(default_factory=dict)
    e2e_ok: bool = True
    veto_reason: str = "none"
    should_disengage: bool = False
    clip_trigger: str | None = None
    policy: str = "shadow"


def load_shadow_config(cfg: dict[str, Any] | None = None) -> ShadowConfig:
    cfg = cfg or {}
    e2e = cfg.get("e2e") or {}
    veto = cfg.get("veto") or cfg.get("shadow") or {}
    return ShadowConfig(
        lane_conf_min=float(veto.get("lane_conf_min", 0.25)),
        steer_disagree_max=float(veto.get("steer_disagree_max", 0.55)),
        path_conf_min=float(veto.get("path_conf_min", 0.15)),
    )


def modular_intent(
    *,
    path_ego: list[dict[str, float]] | None,
    planner: dict[str, Any] | None,
    ego_speed_mps: float,
    seq: int,
) -> DriveCommand:
    return plan_command(
        path_ego=path_ego,
        planner=planner,
        ego_speed_mps=ego_speed_mps,
        seq=seq,
    )


def evaluate_veto(
    *,
    modular: DriveCommand,
    e2e: E2EIntent,
    lane_conf: float,
    path_conf: float,
    heartbeat_ok: bool,
    path_debug_preview: bool,
    allow_preview_drive: bool,
    cfg: ShadowConfig,
    policy: str,
    planner: dict[str, Any] | None = None,
) -> str:
    """Return veto_reason or 'none'. Modular supervisor always may veto E2E."""
    if not heartbeat_ok:
        return "heartbeat_stale"
    # AEB from modular planner: never apply E2E actuators under brake (and warn).
    aeb = str((planner or {}).get("aeb") or "off").lower()
    if aeb == "brake":
        return "aeb_brake"
    if aeb == "warn":
        return "aeb_warn"
    if lane_conf < cfg.lane_conf_min:
        return "low_lane_conf"
    if path_conf < cfg.path_conf_min:
        return "low_path_conf"
    if path_debug_preview and not allow_preview_drive and policy in ("e2e", "shadow"):
        # E2E may still propose; modular preview block applies when applying modular path.
        # For e2e apply path we still veto on preview unless allowed.
        if policy == "e2e":
            return "preview_blocked"
    if not e2e.ok:
        return "e2e_forward_fail"
    disagree = abs(float(modular.steer) - float(e2e.steer))
    if disagree > cfg.steer_disagree_max:
        return "disagreement"
    return "none"


def shadow_tick(
    *,
    policy: str,
    engaged: bool,
    heartbeat_ok: bool,
    path_debug_preview: bool,
    allow_preview_drive: bool,
    path_ego: list[dict[str, float]] | None,
    planner: dict[str, Any] | None,
    ego_speed_mps: float,
    lane_conf: float,
    path_conf: float,
    seq: int,
    e2e_policy: E2EPolicy,
    main_bgr: Any = None,
    wide_bgr: Any = None,
    steer_deg: float = 0.0,
    cfg: ShadowConfig | None = None,
) -> ShadowTick:
    """Compute modular + e2e every tick; choose apply command per policy + gates."""
    cfg = cfg or ShadowConfig()
    policy = (policy or "modular").lower()

    modular = modular_intent(
        path_ego=path_ego,
        planner=planner,
        ego_speed_mps=ego_speed_mps,
        seq=seq,
    )
    e2e = e2e_policy.forward(
        main_bgr,
        wide_bgr,
        speed_mps=ego_speed_mps,
        steer_deg=steer_deg,
    )

    # Shadow fields always reflect E2E proposal (and modular pedals for comparison).
    shadow = {
        "steer": float(e2e.steer),
        "throttle": float(e2e.throttle),
        "brake": float(e2e.brake),
        "modular_steer": float(modular.steer),
        "modular_throttle": float(modular.throttle),
        "modular_brake": float(modular.brake),
        "accel": float(e2e.accel),
    }

    veto = evaluate_veto(
        modular=modular,
        e2e=e2e,
        lane_conf=float(lane_conf),
        path_conf=float(path_conf),
        heartbeat_ok=heartbeat_ok,
        path_debug_preview=path_debug_preview,
        allow_preview_drive=allow_preview_drive,
        cfg=cfg,
        policy=policy,
        planner=planner,
    )

    should_disengage = False
    clip_trigger = None
    e2e_ok = veto == "none" and bool(e2e.ok)

    # Disengaged → never actuate (shadow still written above).
    if not engaged:
        applied = stop_command(seq=seq, reason="not_engaged")
        return ShadowTick(
            modular=modular,
            e2e=e2e,
            applied=applied,
            shadow=shadow,
            e2e_ok=e2e_ok,
            veto_reason=veto if veto != "none" else "none",
            should_disengage=False,
            clip_trigger=None,
            policy=policy,
        )

    if not heartbeat_ok:
        applied = stop_command(seq=seq, reason="heartbeat_stale")
        return ShadowTick(
            modular=modular,
            e2e=e2e,
            applied=applied,
            shadow=shadow,
            e2e_ok=False,
            veto_reason="heartbeat_stale",
            should_disengage=True,
            clip_trigger="disengage",
            policy=policy,
        )

    # Modular-only policy: classic plan path; preview gate.
    if policy == "modular":
        if path_debug_preview and not allow_preview_drive:
            applied = stop_command(seq=seq, reason="preview_blocked")
        else:
            applied = DriveCommand(
                steer=modular.steer,
                throttle=modular.throttle,
                brake=modular.brake,
                seq=seq,
                reason="ok",
            )
        return ShadowTick(
            modular=modular,
            e2e=e2e,
            applied=applied,
            shadow=shadow,
            e2e_ok=e2e_ok,
            veto_reason="none",
            should_disengage=False,
            clip_trigger=None,
            policy=policy,
        )

    # E2E or shadow: modular is safety supervisor — veto holds / disengages.
    if veto != "none":
        should_disengage = veto in (
            "low_lane_conf",
            "disagreement",
            "e2e_forward_fail",
            "low_path_conf",
            "aeb_brake",
        )
        clip_trigger = "disengage" if should_disengage else None
        applied = stop_command(seq=seq, reason=f"veto:{veto}")
        return ShadowTick(
            modular=modular,
            e2e=e2e,
            applied=applied,
            shadow=shadow,
            e2e_ok=False,
            veto_reason=veto,
            should_disengage=should_disengage,
            clip_trigger=clip_trigger,
            policy=policy,
        )

    if policy == "e2e":
        if path_debug_preview and not allow_preview_drive:
            # Already covered by veto for e2e; belt-and-suspenders.
            applied = stop_command(seq=seq, reason="preview_blocked")
        else:
            applied = DriveCommand(
                steer=float(e2e.steer),
                throttle=float(e2e.throttle),
                brake=float(e2e.brake),
                seq=seq,
                reason="ok",
            )
        return ShadowTick(
            modular=modular,
            e2e=e2e,
            applied=applied,
            shadow=shadow,
            e2e_ok=True,
            veto_reason="none",
            should_disengage=False,
            clip_trigger=None,
            policy=policy,
        )

    # shadow: write E2E shadow fields but apply modular (supervisor drives).
    if path_debug_preview and not allow_preview_drive:
        applied = stop_command(seq=seq, reason="preview_blocked")
    else:
        applied = DriveCommand(
            steer=modular.steer,
            throttle=modular.throttle,
            brake=modular.brake,
            seq=seq,
            reason="ok",
        )
    return ShadowTick(
        modular=modular,
        e2e=e2e,
        applied=applied,
        shadow=shadow,
        e2e_ok=True,
        veto_reason="none",
        should_disengage=False,
        clip_trigger=None,
        policy=policy,
    )
