"""Offline stop-sign / traffic-light events and metrics (docs/specs/trainer_2.md, PR 6b).

Cuts decided by Michael 2026-11-25:

* a **candidate approach** is a GT stop sign, stop line, or red/yellow light on
  the ego path within ``WINDOW_M`` (40 m), with ego speed >= ``ENTRY_V_MPS``
  (2 m/s) at first entry into that window. Yellow as a scored stop event is a
  spec proposal. Green and ``unknown`` lamps are never stop events;
* a stop is **held** only when ego speed stays below ``STOP_V_MPS`` (0.5 m/s)
  for >= ``HOLD_S`` (0.5 s) before the stop line + ``LINE_TOL_M`` (1 m). An
  approach without that hold is a **miss**;
* **false brake**: ``ctl_brake > BRAKE_ON`` (0.3) with no stop event within
  40 m, counted per onset and per km. **Green-light false brake**: the same on
  a GT green light with no other stop event;
* **stop distance**: signed metres to the stop line at the start of the held
  full stop (median, p95);
* ``stop_side_angle`` slice: stop-sign events whose bearing from ego at first
  GT visibility is 22-37.5 deg, reported on its own.

Input rows (one per frame, in time order per ``source``)::

    {"t": s, "v": m/s, "ctl_brake": 0..1, "split": "train"|"holdout",
     "source": recording, "stops": [{"id", "cls", "x", "y", "state"?}, ...]}

``x``/``y`` are ego-frame metres (y forward = distance to the stop line /
sign). Only holdout rows are scored; an approach with any non-holdout row in
its window is not scored. GT here is a training/eval label only: this module
imports nothing from ``python.*`` and no live path imports it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

STOP_V_MPS = 0.5
HOLD_S = 0.5
LINE_TOL_M = 1.0
WINDOW_M = 40.0
ENTRY_V_MPS = 2.0
BRAKE_ON = 0.3
SIDE_ANGLE_DEG = (22.0, 37.5)
STOP_CLASSES = ("stop_sign", "stop_line")
LIGHT_CLASS = "traffic_light"
STOP_LAMPS = ("red", "yellow")
MAX_DT_S = 1.0
_EPS = 1e-6

__all__ = [
    "BRAKE_ON",
    "ENTRY_V_MPS",
    "HOLD_S",
    "LINE_TOL_M",
    "SIDE_ANGLE_DEG",
    "STOP_V_MPS",
    "WINDOW_M",
    "StopEventError",
    "bearing_deg",
    "is_stop_target",
    "score",
]


class StopEventError(ValueError):
    """Rows are not scoreable (missing split, time or speed)."""


def bearing_deg(target: Mapping[str, Any]) -> float:
    """Unsigned bearing of a target from ego heading, degrees (0 = dead ahead)."""
    return math.degrees(math.atan2(abs(float(target["x"])), float(target["y"])))


def is_stop_target(target: Mapping[str, Any]) -> bool:
    cls = str(target.get("cls") or "")
    if cls in STOP_CLASSES:
        return True
    return cls == LIGHT_CLASS and str(target.get("state") or "unknown") in STOP_LAMPS


def _in_window(target: Mapping[str, Any]) -> bool:
    y = float(target["y"])
    return -LINE_TOL_M <= y <= WINDOW_M


def _check(row: Mapping[str, Any]) -> None:
    for key in ("t", "v", "split"):
        if key not in row or row[key] is None:
            raise StopEventError(f"row {row.get('source')!r} i={row.get('i')!r} has no {key!r}")


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _slice_names(ev: Mapping[str, Any]) -> list[str]:
    names = []
    if ev["cls"] == LIGHT_CLASS:
        names.append(f"{ev['state']}_light")
    else:
        names.append(ev["cls"])
        if ev["cls"] == "stop_sign" and SIDE_ANGLE_DEG[0] <= ev["first_bearing_deg"] <= SIDE_ANGLE_DEG[1]:
            names.append("stop_side_angle")
    return names


def _events_for_source(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    first_seen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for tgt in row.get("stops") or []:
            first_seen.setdefault(str(tgt["id"]), tgt)
    events = []
    for sid, first in first_seen.items():
        entry = None
        for k, row in enumerate(rows):
            tgt = next((s for s in row.get("stops") or [] if str(s["id"]) == sid), None)
            if tgt is not None and is_stop_target(tgt) and 0.0 < float(tgt["y"]) <= WINDOW_M:
                entry = k
                break
        if entry is None:
            continue
        entry_tgt = next(s for s in rows[entry]["stops"] if str(s["id"]) == sid)
        if float(rows[entry]["v"]) < ENTRY_V_MPS:
            continue
        window: list[tuple[Mapping[str, Any], float]] = []
        for row in rows[entry:]:
            tgt = next((s for s in row.get("stops") or [] if str(s["id"]) == sid), None)
            if tgt is None or float(tgt["y"]) < -LINE_TOL_M:
                break
            window.append((row, float(tgt["y"])))
        if any(str(r["split"]) != "holdout" for r, _y in window):
            continue
        held_at = None
        run_start = None
        for k, (row, y) in enumerate(window):
            if float(row["v"]) < STOP_V_MPS:
                if run_start is None:
                    run_start = k
                nxt_t = float(window[k + 1][0]["t"]) if k + 1 < len(window) else None
                still = nxt_t is not None and float(window[k + 1][0]["v"]) < STOP_V_MPS
                if not still:
                    t0 = float(window[run_start][0]["t"])
                    t1 = nxt_t if nxt_t is not None else float(row["t"])
                    if t1 - t0 >= HOLD_S - _EPS:
                        held_at = run_start
                        break
                    run_start = None
            else:
                run_start = None
        state = str(entry_tgt.get("state") or "unknown") if entry_tgt.get("cls") == LIGHT_CLASS else None
        ev = {
            "source": str(rows[entry].get("source") or ""),
            "id": sid,
            "cls": str(entry_tgt.get("cls")),
            "state": state,
            "t_entry": float(rows[entry]["t"]),
            "v_entry": float(rows[entry]["v"]),
            "first_bearing_deg": bearing_deg(first),
            "held": held_at is not None,
            "stop_distance_m": None if held_at is None else window[held_at][1],
        }
        ev["slices"] = _slice_names(ev)
        events.append(ev)
    return events


def score(rows: Iterable[Mapping[str, Any]], *, brake_key: str = "ctl_brake") -> dict[str, Any]:
    """Events, misses, false brakes, stop distance and slices over the holdout rows."""
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        _check(row)
        by_source.setdefault(str(row.get("source") or ""), []).append(row)
    detail: list[dict[str, Any]] = []
    false_brakes = 0
    green_false = 0
    metres = 0.0
    for src_rows in by_source.values():
        src_rows = sorted(src_rows, key=lambda r: float(r["t"]))
        detail.extend(_events_for_source(src_rows))
        prev_on = False
        prev_t = None
        for row in src_rows:
            hold = str(row["split"]) == "holdout"
            t = float(row["t"])
            if hold and prev_t is not None:
                dt = t - prev_t
                if 0.0 < dt <= MAX_DT_S:
                    metres += max(0.0, float(row["v"])) * dt
            prev_t = t
            brake = row.get(brake_key)
            on = brake is not None and float(brake) > BRAKE_ON
            onset = on and not prev_on
            prev_on = on
            if not (hold and onset):
                continue
            stops = row.get("stops") or []
            if any(is_stop_target(s) and _in_window(s) for s in stops):
                continue
            false_brakes += 1
            if any(s.get("cls") == LIGHT_CLASS and s.get("state") == "green" and _in_window(s) for s in stops):
                green_false += 1
    km = metres / 1000.0
    slices: dict[str, dict[str, int]] = {
        name: {"events": 0, "missed": 0}
        for name in ("stop_sign", "stop_side_angle", "stop_line", "red_light", "yellow_light")
    }
    for ev in detail:
        for name in ev["slices"]:
            slot = slices.setdefault(name, {"events": 0, "missed": 0})
            slot["events"] += 1
            slot["missed"] += int(not ev["held"])
    dists = [float(ev["stop_distance_m"]) for ev in detail if ev["held"]]
    return {
        "events": len(detail),
        "missed": sum(1 for ev in detail if not ev["held"]),
        "slices": slices,
        "false_brakes": false_brakes,
        "green_false_brakes": green_false,
        "km": km,
        "false_brakes_per_km": (false_brakes / km) if km > 0 else float("nan"),
        "green_false_brakes_per_km": (green_false / km) if km > 0 else float("nan"),
        "stop_distance_m": {
            "n": len(dists),
            "median": _quantile(dists, 0.5),
            "p95": _quantile(dists, 0.95),
        },
        "detail": detail,
    }
