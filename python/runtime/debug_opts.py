"""Session debug knobs for the GVD VISION nerd DRIVE tab.

These mutate the live supervisor: gates, actuator channels, AEB/CIPV, perception.
They do not invent cameras or feed LiDAR into the planner. Sim toy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python.control.actuate import DriveCommand

POLICIES = ("session", "modular", "e2e", "shadow")

# Rows drawn on the DRIVE tab, in order. kind=header is a label, not a control.
DEBUG_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "gates", "kind": "header", "label": "GATES — live drive"},
    {
        "id": "allow_preview",
        "kind": "bool",
        "label": "drive on preview path",
        "attr": "allow_preview",
        "hint": "bypass path_debug_preview block",
    },
    {
        "id": "force_engage",
        "kind": "bool",
        "label": "force engage",
        "attr": "force_engage",
        "hint": "debug: treat as engaged without Alt+G",
    },
    {
        "id": "ignore_override",
        "kind": "bool",
        "label": "ignore player override",
        "attr": "ignore_override",
        "hint": "wheel/pedals will not disengage",
    },
    {
        "id": "ignore_veto_disengage",
        "kind": "bool",
        "label": "veto holds, no disengage",
        "attr": "ignore_veto_disengage",
        "hint": "modular veto still brakes; engage stays",
    },
    {
        "id": "policy",
        "kind": "enum",
        "label": "policy",
        "attr": "policy",
        "values": list(POLICIES),
    },
    {"id": "act", "kind": "header", "label": "ACTUATORS"},
    {"id": "steer_on", "kind": "bool", "label": "steer channel", "attr": "steer_on"},
    {"id": "throttle_on", "kind": "bool", "label": "throttle channel", "attr": "throttle_on"},
    {"id": "brake_on", "kind": "bool", "label": "brake channel", "attr": "brake_on"},
    {
        "id": "steer_gain",
        "kind": "float",
        "label": "steer gain",
        "attr": "steer_gain",
        "lo": 0.0,
        "hi": 2.5,
        "step": 0.1,
    },
    {
        "id": "max_steer",
        "kind": "float",
        "label": "max |steer|",
        "attr": "max_steer",
        "lo": 0.1,
        "hi": 1.0,
        "step": 0.05,
    },
    {"id": "invert_steer", "kind": "bool", "label": "invert steer", "attr": "invert_steer"},
    {"id": "hold_brake", "kind": "bool", "label": "hold brake", "attr": "hold_brake"},
    {"id": "freeze_cmd", "kind": "bool", "label": "freeze last cmd", "attr": "freeze_cmd"},
    {"id": "plan", "kind": "header", "label": "PLANNER / AEB"},
    {"id": "aeb_on", "kind": "bool", "label": "AEB", "attr": "aeb_on"},
    {
        "id": "aeb_ttc",
        "kind": "float",
        "label": "AEB TTC s",
        "attr": "aeb_ttc",
        "lo": 0.4,
        "hi": 3.0,
        "step": 0.1,
    },
    {"id": "cipv_on", "kind": "bool", "label": "CIPV / lead", "attr": "cipv_on"},
    {
        "id": "speed_cap",
        "kind": "float",
        "label": "speed cap m/s",
        "attr": "speed_cap",
        "lo": 2.0,
        "hi": 30.0,
        "step": 1.0,
    },
    {
        "id": "cruise_mps",
        "kind": "float_opt",
        "label": "cruise override",
        "attr": "cruise_mps",
        "lo": 0.0,
        "hi": 25.0,
        "step": 1.0,
        "off": None,
        "on_default": 8.0,
    },
    {
        "id": "corridor_width",
        "kind": "float",
        "label": "corridor width m",
        "attr": "corridor_width",
        "lo": 1.2,
        "hi": 4.0,
        "step": 0.1,
    },
    {
        "id": "lane_conf_min",
        "kind": "float",
        "label": "veto lane_conf min",
        "attr": "lane_conf_min",
        "lo": 0.0,
        "hi": 0.95,
        "step": 0.05,
    },
    {"id": "perc", "kind": "header", "label": "PERCEPTION"},
    {
        "id": "lanes_on",
        "kind": "bool",
        "label": "use lane paint",
        "attr": "lanes_on",
        "hint": "off → preview path (needs drive-on-preview to move)",
    },
    {
        "id": "detector_on",
        "kind": "bool",
        "label": "use detector tracks",
        "attr": "detector_on",
        "hint": "off → no CIPV/AEB from YOLO",
    },
)

# Overlay toggles (GVD VISION VIZ tab). Keys 1–5 map onto the first five debug layers.
VIZ_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "viz_head", "kind": "header", "label": "OVERLAY — stage layers"},
    {
        "id": "viz_dense",
        "kind": "bool",
        "label": "dense overlay",
        "attr": "viz_dense",
        "hint": "occupancy + ids + vel + boxes + FOV + cost",
    },
    {"id": "viz_path", "kind": "bool", "label": "corridor ribbon", "attr": "viz_path"},
    {"id": "viz_lanes", "kind": "bool", "label": "lane paint", "attr": "viz_lanes"},
    {"id": "viz_ghosts", "kind": "bool", "label": "track boxes", "attr": "viz_ghosts"},
    {"id": "viz_signs", "kind": "bool", "label": "signs / lights", "attr": "viz_signs"},
    {"id": "viz_forecast", "kind": "bool", "label": "forecast fans", "attr": "viz_forecast"},
    {"id": "viz_pip", "kind": "bool", "label": "cam_main PIP", "attr": "viz_pip"},
    {"id": "viz_hud", "kind": "bool", "label": "dense HUD", "attr": "viz_hud"},
    {"id": "viz_dbg", "kind": "header", "label": "DEBUG LAYERS (keys 1–5)"},
    {
        "id": "viz_occ",
        "kind": "bool",
        "label": "1 occupancy (from tracks)",
        "attr": "viz_occ",
        "hint": "not a learned grid",
    },
    {"id": "viz_boxes", "kind": "bool", "label": "2 detector boxes in PIP", "attr": "viz_boxes"},
    {"id": "viz_lane_poly", "kind": "bool", "label": "3 lane polynomials", "attr": "viz_lane_poly"},
    {"id": "viz_frustums", "kind": "bool", "label": "4 camera FOV", "attr": "viz_frustums"},
    {"id": "viz_cost", "kind": "bool", "label": "5 planner samples", "attr": "viz_cost"},
    {"id": "viz_more", "kind": "header", "label": "ANNOTATION"},
    {"id": "viz_ids", "kind": "bool", "label": "track ids", "attr": "viz_ids"},
    {"id": "viz_vel", "kind": "bool", "label": "velocity ticks", "attr": "viz_vel"},
    {"id": "viz_cams", "kind": "bool", "label": "camera strip", "attr": "viz_cams"},
)

CONTROL_ROWS = tuple(r for r in DEBUG_ROWS if r.get("kind") != "header")
VIZ_CONTROL_ROWS = tuple(r for r in VIZ_ROWS if r.get("kind") != "header")

# Key 1–5 ↔ overlay attrs. Dense overlay turns the annotation stack on together.
LAYER_ATTR = {1: "viz_occ", 2: "viz_boxes", 3: "viz_lane_poly", 4: "viz_frustums", 5: "viz_cost"}
DENSE_ATTRS = ("viz_occ", "viz_boxes", "viz_ids", "viz_vel", "viz_frustums", "viz_cost")


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class DebugOpts:
    """Live knobs. Defaults match stock supervisor behaviour (preview blocked, Alt+G engage)."""

    allow_preview: bool = False
    force_engage: bool = False
    ignore_override: bool = False
    ignore_veto_disengage: bool = False
    policy: str = "session"
    steer_on: bool = True
    throttle_on: bool = True
    brake_on: bool = True
    steer_gain: float = 1.0
    max_steer: float = 1.0
    invert_steer: bool = False
    hold_brake: bool = False
    freeze_cmd: bool = False
    aeb_on: bool = True
    aeb_ttc: float = 1.2
    cipv_on: bool = True
    speed_cap: float = 18.0
    cruise_mps: float | None = None
    corridor_width: float = 2.0
    lane_conf_min: float = 0.25
    lanes_on: bool = True
    detector_on: bool = True
    viz_dense: bool = False
    viz_path: bool = True
    viz_lanes: bool = True
    viz_ghosts: bool = True
    viz_signs: bool = True
    viz_forecast: bool = True
    viz_pip: bool = True
    viz_hud: bool = True
    viz_occ: bool = False
    viz_boxes: bool = False
    viz_lane_poly: bool = False
    viz_frustums: bool = False
    viz_cost: bool = False
    viz_ids: bool = False
    viz_vel: bool = False
    viz_cams: bool = False
    _frozen: DriveCommand | None = field(default=None, repr=False, compare=False)

    def effective_policy(self, session_policy: str) -> str:
        if self.policy in ("modular", "e2e", "shadow"):
            return self.policy
        return session_policy

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_preview": bool(self.allow_preview),
            "force_engage": bool(self.force_engage),
            "ignore_override": bool(self.ignore_override),
            "ignore_veto_disengage": bool(self.ignore_veto_disengage),
            "policy": str(self.policy),
            "steer_on": bool(self.steer_on),
            "throttle_on": bool(self.throttle_on),
            "brake_on": bool(self.brake_on),
            "steer_gain": float(self.steer_gain),
            "max_steer": float(self.max_steer),
            "invert_steer": bool(self.invert_steer),
            "hold_brake": bool(self.hold_brake),
            "freeze_cmd": bool(self.freeze_cmd),
            "aeb_on": bool(self.aeb_on),
            "aeb_ttc": float(self.aeb_ttc),
            "cipv_on": bool(self.cipv_on),
            "speed_cap": float(self.speed_cap),
            "cruise_mps": self.cruise_mps,
            "corridor_width": float(self.corridor_width),
            "lane_conf_min": float(self.lane_conf_min),
            "lanes_on": bool(self.lanes_on),
            "detector_on": bool(self.detector_on),
            "viz_dense": bool(self.viz_dense),
            "viz_path": bool(self.viz_path),
            "viz_lanes": bool(self.viz_lanes),
            "viz_ghosts": bool(self.viz_ghosts),
            "viz_signs": bool(self.viz_signs),
            "viz_forecast": bool(self.viz_forecast),
            "viz_pip": bool(self.viz_pip),
            "viz_hud": bool(self.viz_hud),
            "viz_occ": bool(self.viz_occ),
            "viz_boxes": bool(self.viz_boxes),
            "viz_lane_poly": bool(self.viz_lane_poly),
            "viz_frustums": bool(self.viz_frustums),
            "viz_cost": bool(self.viz_cost),
            "viz_ids": bool(self.viz_ids),
            "viz_vel": bool(self.viz_vel),
            "viz_cams": bool(self.viz_cams),
            "drive_uses": "vision",
        }

    def apply_dense(self) -> None:
        for attr in DENSE_ATTRS:
            setattr(self, attr, bool(self.viz_dense))

    def format_value(self, row: dict[str, Any]) -> str:
        kind = row.get("kind")
        attr = str(row.get("attr") or "")
        val = getattr(self, attr, None)
        if kind == "bool":
            return "ON" if val else "off"
        if kind == "enum":
            return str(val)
        if kind == "float_opt":
            if val is None:
                return "off"
            return f"{float(val):.1f}"
        if kind == "float":
            step = float(row.get("step") or 0.1)
            if step >= 1.0:
                return f"{float(val):.0f}"
            return f"{float(val):.2f}".rstrip("0").rstrip(".")
        return str(val)

    def toggle(self, row: dict[str, Any]) -> None:
        kind = row.get("kind")
        attr = str(row.get("attr") or "")
        if kind == "bool":
            setattr(self, attr, not bool(getattr(self, attr)))
            if attr == "freeze_cmd" and not self.freeze_cmd:
                self._frozen = None
            if attr == "viz_dense":
                self.apply_dense()
            return
        if kind == "enum":
            values = list(row.get("values") or POLICIES)
            cur = str(getattr(self, attr) or values[0])
            i = values.index(cur) if cur in values else 0
            setattr(self, attr, values[(i + 1) % len(values)])
            return
        if kind == "float_opt":
            cur = getattr(self, attr)
            if cur is None:
                setattr(self, attr, float(row.get("on_default") or row.get("lo") or 0.0))
            else:
                setattr(self, attr, None)
            return
        if kind == "float":
            self.nudge(row, +1)

    def nudge(self, row: dict[str, Any], direction: int) -> None:
        kind = row.get("kind")
        attr = str(row.get("attr") or "")
        step = float(row.get("step") or 0.1) * (1 if direction >= 0 else -1)
        if kind == "enum":
            values = list(row.get("values") or POLICIES)
            cur = str(getattr(self, attr) or values[0])
            i = values.index(cur) if cur in values else 0
            setattr(self, attr, values[(i + (1 if direction >= 0 else -1)) % len(values)])
            return
        if kind == "bool":
            if direction != 0:
                self.toggle(row)
            return
        if kind == "float_opt":
            cur = getattr(self, attr)
            if cur is None:
                if direction > 0:
                    setattr(self, attr, float(row.get("on_default") or 8.0))
                return
            nxt = float(cur) + step
            lo = float(row.get("lo") or 0.0)
            if nxt < lo - 1e-9 and direction < 0:
                setattr(self, attr, None)
                return
            setattr(self, attr, _clip(nxt, lo, float(row.get("hi") or 25.0)))
            return
        if kind == "float":
            cur = float(getattr(self, attr) or 0.0)
            setattr(self, attr, _clip(cur + step, float(row.get("lo") or 0.0), float(row.get("hi") or 1.0)))


def control_index(sel: int) -> int:
    n = len(CONTROL_ROWS)
    if n <= 0:
        return 0
    return sel % n


def row_at(sel: int) -> dict[str, Any]:
    return CONTROL_ROWS[control_index(sel)]


def viz_control_index(sel: int) -> int:
    n = len(VIZ_CONTROL_ROWS)
    if n <= 0:
        return 0
    return sel % n


def viz_row_at(sel: int) -> dict[str, Any]:
    return VIZ_CONTROL_ROWS[viz_control_index(sel)]


def apply_to_perception(opts: DebugOpts, pout: Any) -> Any:
    """Rewrite perception/planner outputs the supervisor is about to act on."""
    pl = dict(pout.planner or {})
    if not opts.detector_on:
        pout.tracks = []
        pout.tracks_n = 0
        pout.objects_n = 0
        pout.signs = []
        if hasattr(pout, "dets"):
            pout.dets = []
        pl["cipv_id"] = None
        pl["ttc_lead"] = None
        pl["aeb"] = "off"
    if not opts.cipv_on:
        pl["cipv_id"] = None
        pl["ttc_lead"] = None
        pl["aeb"] = "off"
    if not opts.lanes_on:
        pout.path_debug_preview = True
        pout.lane_conf = 0.0
        pout.path_conf = min(float(pout.path_conf or 0.0), 0.35)
    if not opts.aeb_on:
        pl["aeb"] = "off"
    elif opts.cipv_on and opts.detector_on:
        ttc = pl.get("ttc_lead")
        try:
            ttc_f = float(ttc) if ttc is not None else None
        except (TypeError, ValueError):
            ttc_f = None
        if ttc_f is not None:
            if ttc_f < opts.aeb_ttc:
                pl["aeb"] = "brake"
            elif ttc_f < opts.aeb_ttc + 1.3:
                pl["aeb"] = "warn"
            else:
                pl["aeb"] = "off"
    if pl.get("aeb") == "brake" and opts.aeb_on:
        pl["target_v"] = 0.0
    elif opts.cruise_mps is not None:
        pl["target_v"] = float(opts.cruise_mps)
    try:
        tv = float(pl.get("target_v") or 0.0)
    except (TypeError, ValueError):
        tv = 0.0
    pl["target_v"] = min(tv, float(opts.speed_cap))
    pl["corridor_width"] = float(opts.corridor_width)
    pout.path_width = float(opts.corridor_width)
    pout.planner = pl
    return pout


def apply_to_command(opts: DebugOpts, cmd: DriveCommand) -> DriveCommand:
    """Rewrite the command that will be applied this tick."""
    if opts.freeze_cmd and opts._frozen is not None:
        frozen = opts._frozen
        return DriveCommand(
            steer=frozen.steer,
            throttle=frozen.throttle,
            brake=frozen.brake,
            seq=cmd.seq,
            applied=False,
            reason=cmd.reason if cmd.reason != "ok" else "ok",
        )
    steer = float(cmd.steer)
    throttle = float(cmd.throttle)
    brake = float(cmd.brake)
    if opts.invert_steer:
        steer = -steer
    steer = _clip(steer * float(opts.steer_gain), -1.0, 1.0)
    cap = max(0.0, float(opts.max_steer))
    steer = _clip(steer, -cap, cap)
    if not opts.steer_on:
        steer = 0.0
    if not opts.throttle_on:
        throttle = 0.0
    if not opts.brake_on:
        brake = 0.0
    if opts.hold_brake:
        throttle = 0.0
        brake = 1.0
    out = DriveCommand(
        steer=steer,
        throttle=throttle,
        brake=brake,
        seq=cmd.seq,
        applied=cmd.applied,
        reason=cmd.reason,
    )
    if opts.freeze_cmd:
        opts._frozen = DriveCommand(
            steer=out.steer, throttle=out.throttle, brake=out.brake, seq=out.seq, reason=out.reason
        )
    else:
        opts._frozen = None
    return out
