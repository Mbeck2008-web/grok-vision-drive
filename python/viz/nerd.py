"""Monospace nerd overlay (toggle V). Tabs: LIVE telemetry / DRIVE / VIZ / keys."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from python.runtime.debug_opts import DEBUG_ROWS, VIZ_ROWS, DebugOpts

BG = (16, 13, 12)
FG = (212, 204, 200)
ICE = (212, 196, 158)
DIM = (120, 120, 120)


def scene_note(s: dict[str, Any]) -> str:
    """Honest lane/edge/sign tally for the OpenCV stage (and the nerd row)."""
    det = pred = stub = 0
    for ln in s.get("lanes_ext") or s.get("lanes") or []:
        if not isinstance(ln, dict):
            det += 1
            continue
        kind = str(ln.get("kind") or "detected")
        if kind == "detected":
            det += 1
        elif kind == "stub":
            stub += 1
        else:
            pred += 1
    bits: list[str] = []
    if det or pred or stub:
        parts = []
        if det:
            parts.append(f"{det} seen")
        if pred:
            parts.append(f"{pred} pred")
        if stub:
            parts.append(f"{stub} stub")
        bits.append("lanes " + "+".join(parts))
    else:
        bits.append("no lane paint")
    edges = s.get("road_edges") or s.get("edges") or []
    if edges:
        only_pred = all(
            (not isinstance(e, dict)) or str(e.get("kind") or "predicted") != "detected"
            for e in edges
        )
        bits.append("edges pred" if only_pred else "edges")
    signs = poles = 0
    for item in s.get("signs") or []:
        if str(item.get("cls") or item.get("class") or "") == "pole":
            poles += 1
        else:
            signs += 1
    if signs:
        bits.append(f"{signs} sign" + ("" if signs == 1 else "s"))
    if poles:
        bits.append(f"{poles} pole" + ("" if poles == 1 else "s"))
    return " · ".join(bits)


def render_panel(
    state: dict[str, Any],
    h: int = 720,
    w: int = 420,
    show_help: bool = False,
    ui: Any = None,
) -> np.ndarray:
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    tab = "keys" if show_help else "live"
    opts = DebugOpts()
    sel = 0
    viz_sel = 0
    hits: list[dict[str, Any]] = []
    if ui is not None:
        tab = str(getattr(ui, "nerd_tab", None) or tab)
        if getattr(ui, "show_help", False) and tab == "live":
            tab = "keys"
        opts = getattr(ui, "debug", None) or opts
        sel = int(getattr(ui, "debug_sel", 0) or 0)
        viz_sel = int(getattr(ui, "viz_sel", 0) or 0)
    y = 22
    cv2.putText(img, "GVD  ·  VISION", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ICE, 1, cv2.LINE_AA)
    y = _draw_tabs(img, tab, w, y + 10, hits)
    if tab == "keys":
        lines = _help_lines()
        y += 8
        for line in lines:
            col = DIM if line.startswith(" ") or line.startswith("keys") else FG
            cv2.putText(img, line[:58], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            y += 16
            if y > h - 24:
                break
    elif tab == "drive":
        y = _draw_knob_tab(
            img, opts, sel, y + 6, h, w, hits, DEBUG_ROWS,
            intro="wires into live gates / actuators / AEB",
        )
    elif tab == "viz":
        y = _draw_knob_tab(
            img, opts, viz_sel, y + 6, h, w, hits, VIZ_ROWS,
            intro="stage layers — occupancy is from tracks, not a net",
        )
    else:
        y += 8
        for line in _lines(state):
            col = DIM if line.startswith(" ") else FG
            cv2.putText(img, line[:58], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            y += 16
            if y > h - 24:
                break
    cv2.putText(
        img,
        "D drive  G viz  [ ] tab  click  V hide",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        DIM,
        1,
        cv2.LINE_AA,
    )
    if ui is not None:
        ui.nerd_hits = hits
        ui.nerd_tab = tab
    return img


def _draw_tabs(
    img: np.ndarray, tab: str, w: int, y: int, hits: list[dict[str, Any]]
) -> int:
    labels = (("live", "LIVE"), ("drive", "DRIVE"), ("viz", "VIZ"), ("keys", "KEYS"))
    x = 12
    for tid, label in labels:
        tw = 8 + 9 * len(label)
        rect = (x, y, x + tw, y + 20)
        on = tid == tab
        bg = ICE if on else (40, 36, 34)
        fg = (16, 13, 12) if on else FG
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), bg, -1)
        cv2.putText(
            img, label, (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, fg, 1, cv2.LINE_AA
        )
        hits.append({"kind": "tab", "id": tid, "rect": rect})
        x += tw + 8
    # remainder of tab row
    return y + 26


def _draw_knob_tab(
    img: np.ndarray,
    opts: DebugOpts,
    sel: int,
    y0: int,
    h: int,
    w: int,
    hits: list[dict[str, Any]],
    rows: tuple,
    *,
    intro: str,
) -> int:
    y = y0
    cv2.putText(img, intro[:58], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, DIM, 1, cv2.LINE_AA)
    y += 16
    ctrl_i = 0
    n_ctrl = sum(1 for r in rows if r.get("kind") != "header")
    sel_i = (sel % n_ctrl) if n_ctrl else 0
    for row in rows:
        if y > h - 36:
            break
        kind = row.get("kind")
        if kind == "header":
            cv2.putText(
                img,
                str(row.get("label") or ""),
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                ICE,
                1,
                cv2.LINE_AA,
            )
            y += 16
            continue
        chosen = ctrl_i == sel_i
        if chosen:
            cv2.rectangle(img, (10, y - 12), (w - 10, y + 4), (42, 36, 32), -1)
        label = str(row.get("label") or row.get("id"))
        val = opts.format_value(row)
        col = ICE if chosen else FG
        cv2.putText(img, label[:28], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        val_x = w - 88
        minus = (val_x - 16, y - 11, val_x - 2, y + 3)
        plus = (w - 22, y - 11, w - 10, y + 3)
        val_rect = (val_x, y - 11, w - 24, y + 3)
        row_rect = (10, y - 12, w - 10, y + 4)
        cv2.putText(img, "-", (minus[0] + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, val[:8], (val_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        cv2.putText(img, "+", (plus[0] + 1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM, 1, cv2.LINE_AA)
        hits.append({"kind": "row", "i": ctrl_i, "part": "row", "rect": row_rect})
        hits.append({"kind": "row", "i": ctrl_i, "part": "minus", "rect": minus})
        hits.append({"kind": "row", "i": ctrl_i, "part": "value", "rect": val_rect})
        hits.append({"kind": "row", "i": ctrl_i, "part": "plus", "rect": plus})
        y += 16
        ctrl_i += 1
    cv2.putText(
        img,
        "j/k select  h/l nudge  Enter toggle",
        (16, min(h - 28, y + 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        DIM,
        1,
        cv2.LINE_AA,
    )
    return y


def hit_test(hits: list[dict[str, Any]] | None, x: int, y: int) -> dict[str, Any] | None:
    for item in reversed(hits or []):
        rect = item.get("rect") or (0, 0, 0, 0)
        if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            return item
    return None


def _help_lines() -> list[str]:
    return [
        "keys:",
        "  V  toggle this nerd panel",
        "  D  DRIVE tab (gates / actuators / AEB)",
        "  G  VIZ tab (overlay layers)",
        "  [ ] cycle LIVE / DRIVE / VIZ / KEYS",
        "  j/k  select row   h/l nudge",
        "  Enter / click  toggle",
        "  0  clean cabin (stage only)",
        "  1  occupancy (from tracks)",
        "  2  detector boxes in PIP",
        "  3  lane polynomials",
        "  4  camera FOV wedges",
        "  5  planner cost samples",
        "  T  toggle BEV debug / chase 3/4 bird",
        "  C  manual clip",
        "  q  quit",
        "",
        "DRIVE writes the command this tick.",
        "force engage is debug-only. Sim toy.",
    ]


def _vehicle_line(s: dict[str, Any]) -> str:
    """Tech vehicle row; empty on retail so the panel stays one-cam honest."""
    veh = s.get("vehicle") if isinstance(s.get("vehicle"), dict) else {}
    cap = str(s.get("capture_backend") or "")
    if cap != "beamngpy" and not veh.get("connected") and not veh.get("vid"):
        return ""
    if not veh.get("connected"):
        note = veh.get("note") or "not connected"
        return f"vehicle {note}"
    dmg = veh.get("damage")
    dmg_s = f"{float(dmg):.2f}" if dmg is not None else "?"
    pose = "pose ok" if veh.get("pose_ok") else "pose missing"
    return (
        f"vehicle {veh.get('vid') or '?'} {veh.get('model') or ''} "
        f"dmg={dmg_s} {pose}"
    ).strip()


def _nav_line(s: dict[str, Any]) -> str:
    """Tech GPS row; empty on retail so the panel stays one-cam honest."""
    nav = s.get("nav") if isinstance(s.get("nav"), dict) else {}
    cap = str(s.get("capture_backend") or "")
    mode = str(nav.get("mode") or "missing")
    gps = nav.get("gps") if isinstance(nav.get("gps"), dict) else {}
    if mode != "hint" or not gps.get("ok"):
        if cap == "beamngpy":
            return "nav GPS missing"
        return ""
    try:
        loc = f"{float(gps.get('lat')):.5f},{float(gps.get('lon')):.5f}"
    except (TypeError, ValueError):
        loc = "?"
    bits = [f"nav hint {loc}"]
    pin = nav.get("pin") if isinstance(nav.get("pin"), dict) else None
    if pin:
        name = str(pin.get("name") or "pin")
        rng = nav.get("range_m")
        if rng is not None:
            try:
                bits.append(f"{name} {float(rng):.0f}m")
            except (TypeError, ValueError):
                bits.append(name)
        else:
            bits.append(name)
        rel = nav.get("bearing_rel_deg")
        if rel is not None:
            try:
                bits.append(f"brg {float(rel):+.0f}deg")
            except (TypeError, ValueError):
                pass
        bits.append("not routing")
    return " ".join(bits)


def _extras_line(s: dict[str, Any]) -> str:
    """IMU/GPS/LiDAR/radar/Foxglove bus; empty-ish if the block is absent."""
    extra = s.get("sensors") if isinstance(s.get("sensors"), dict) else {}
    if not extra:
        return ""
    imu = extra.get("imu") or "missing"
    gps = extra.get("gps") or "missing"
    lidar = extra.get("lidar") or "missing"
    radar = extra.get("radar") or "missing"
    fox = extra.get("foxglove") or "off"
    drive = extra.get("drive_uses") or "vision"
    return f"extras imu={imu} gps={gps} lidar={lidar} radar={radar} fox={fox} drive={drive}"


def _lines(s: dict[str, Any]) -> list[str]:
    ego = s.get("ego") or {}
    pl = s.get("planner") or {}
    cams = s.get("cam_health") or {}
    cam_s = " ".join(f"{k}:{str(v)[:1]}" for k, v in cams.items())
    cap = s.get("capture_backend") or "?"
    cap_note = s.get("capture_note") or ""
    v = float(ego.get("speed_mps") or 0)
    mph = v * 2.23694
    miss = s.get("missing_state_keys") or []
    lead = None
    for tr in s.get("tracks") or []:
        if str(tr.get("class", "vehicle")) == "vehicle" and float(tr.get("y", 0)) > 5:
            lead = tr
            break
    sel = []
    if lead:
        # miss distance stub at 1/2/3s along ego x=0 corridor
        for t in (1, 2, 3):
            pred_y = float(lead.get("y", 0)) + float(lead.get("speed_mps", 0)) * t
            miss_m = abs(float(lead.get("x", 0)))
            sel.append(f"miss@{t}s={miss_m:.1f}m (lead y~{pred_y:.0f})")
    veh_line = _vehicle_line(s)
    nav_line = _nav_line(s)
    extras_line = _extras_line(s)
    return [
        f"policy: {s.get('policy', '?')}   engage: {s.get('engaged')}",
        f"disengage: {s.get('disengage_reason', 'none')}",
        f"loop {s.get('loop_hz', 0):.1f}Hz  cam {s.get('camera_hz', 0):.1f}Hz",
        f"infer {s.get('infer_ms', 0):.1f}ms  viz {s.get('viz_ms', 0):.1f}ms  hb {s.get('heartbeat_ms', 0):.0f}ms",
        f"VRAM {s.get('gpu_vram_used_gb', 0):.1f}/{s.get('gpu_vram_total_gb', 11):.0f} GB  {s.get('gpu_name', '')}",
        f"cams {cam_s}",
        f"capture {cap} {cap_note}".rstrip(),
        f"rss {float(s.get('rss_mb') or 0):.0f} MB",
        f"lane {s.get('lane_conf', 0):.2f}  bev {s.get('bev_coverage', 0):.0%}  obj {s.get('objects_n', 0)}  trk {s.get('tracks_n', 0)}",
        f"ego {v:.1f} m/s ({mph:.0f} mph)  steer {ego.get('steer_deg', 0):.1f}",
        f"thr {ego.get('throttle', 0):.2f}  brk {ego.get('brake', 0):.2f}  yawr {ego.get('yaw_rate', 0):.2f}",
        f"planner w={pl.get('corridor_width', 0):.2f}  curv={pl.get('curvature', 0):.4f}  vt={pl.get('target_v', 0):.1f}",
        f"TTC {pl.get('ttc_lead')}   AEB {pl.get('aeb')}",
        f"path_conf {s.get('path_conf', 0):.2f}  width {s.get('path_width', 0):.2f}  preview={s.get('path_debug_preview')}",
        f"scene {s.get('viz_scene_note') or scene_note(s)}",
        f"cmd {s.get('actuator', '?')} {s.get('cmd_reason', '?')} applied={s.get('cmd_applied')} ego={s.get('ego_source', '?')}",
        *([veh_line] if veh_line else []),
        *([nav_line] if nav_line else []),
        *([extras_line] if extras_line else []),
        f"clip {s.get('last_clip_trigger', 'none')}",
        *sel,
        *([_debug_line(s)] if _debug_line(s) else []),
        "missing: " + (", ".join(miss) if miss else "none"),
    ]


def _debug_line(s: dict[str, Any]) -> str:
    dbg = s.get("debug") if isinstance(s.get("debug"), dict) else None
    if not dbg:
        return ""
    cap = dbg.get("speed_cap")
    try:
        cap_s = f"{float(cap):.0f}"
    except (TypeError, ValueError):
        cap_s = "?"
    return (
        f"debug pol={dbg.get('policy')} preview={dbg.get('allow_preview')} "
        f"aeb={dbg.get('aeb_on')} cap={cap_s}"
    )
