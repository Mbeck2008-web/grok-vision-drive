"""Stage 2b offline scoring: the holdout pass's predictions -> ``stop_events.score`` (docs/specs/trainer_2.md, 2b / PR 6b).

Offline only. GT here (signs / lamps, recorded speed and time) is a label and an eval input, never a model input;
no live path imports this module. The event definitions, cuts and the scorer are ``python/train/stop_events.py``
(Michael's 2026-11-25 cuts), unchanged. This module only builds its input from the holdout pass:

* **per-event stop ids**: ``signs.jsonl`` entries carry ``id`` (``signal_labels.sign_id``, from the world position, so
  it is the same on every tick). An entry without one is not a trackable event: it is skipped and counted, and a
  holdout with no id'd entry has no events (``StopNotMeasured``).
* **the held-stop speed key**: ``pred_v``. Each holdout row carries the model's predicted speed profile and
  ``score(..., speed_key="pred_v")`` judges the held stop on it. The recorded ``v`` still decides which approaches exist
  and the holdout ``km``, so a model faces exactly the events (and per-km denominator) that any other profile does.
* **the predicted speed profile (``pred_v``)** is integrated from the ctl head's own outputs at h = 0
  (``ctl_throttle``, ``ctl_brake``): ``dv/dt = THROTTLE_ACCEL * throttle - BRAKE_DECEL * brake``, floored at 0, from
  the recorded speed at the start of an approach, open-loop at the recorded row times. It is not re-anchored to the
  recorded speed while an approach lasts, so a recorded human stop never credits the model: the model has to brake
  to a stop and keep braking (or at least not drive off) for the 0.5 s hold. Outside an approach ``pred_v`` is the
  recorded ``v`` (not scored there). The two constants are **PROPOSAL, pending Michael** (``BRAKE_DECEL`` is the
  repo's measured full-pedal decel in ``gt_lane_loop.PARK_DECEL_MPS2`` = 4.5 m/s^2; ``THROTTLE_ACCEL`` has no
  measurement in the repo yet).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from python.train import stop_events as se

# PROPOSAL pending Michael: see the module docstring.
BRAKE_DECEL_MPS2 = 4.5
THROTTLE_ACCEL_MPS2 = 2.0
# Row of ``ctl`` (3, K): steer, throttle, brake; column 0 is h = 0 (now).
_THROTTLE, _BRAKE, _H0 = 1, 2, 0
# Stage metric names (trainer2_recipe.STAGE_PRIMARY_METRICS["2b"]) and what they count.
STOP_SLICES = ("stop_sign", "stop_line")
LIGHT_SLICES = ("red_light", "yellow_light")


class StopNotMeasured(ValueError):
    """The holdout cannot score stage 2b (no usable events / rows). Says exactly why; nothing is made up."""


def _get(row: Mapping[str, Any], *path: str) -> Any:
    cur: Any = row
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _signs_of(row: Mapping[str, Any]) -> list[Any]:
    tech = row.get("tech")
    signs = tech.get("signs") if isinstance(tech, Mapping) and "signs" in tech else row.get("signs")
    return list(signs) if isinstance(signs, list) else []


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stop_rows(pairs: Iterable[tuple[Mapping[str, Any], Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """``stop_events.score`` rows from ``(row, ctl (3, K))`` pairs of the holdout pass, plus what was skipped.

    ``t`` is the recording's sim time (``ego.sim_t``: the hold is judged in sim seconds), ``v`` the recorded speed
    (``ego.speed_mps``), ``ctl_brake`` / ``ctl_throttle`` the prediction at h = 0. A row without a finite sim time or
    speed cannot be placed in time and is refused loudly, not dropped. Rows come out in time order per source.
    """
    out: list[dict[str, Any]] = []
    skipped = {"no_id": 0, "no_pose": 0, "rows": 0}
    for n, (row, ctl) in enumerate(pairs):
        t, v = _finite(_get(row, "ego", "sim_t")), _finite(_get(row, "ego", "speed_mps"))
        if t is None or v is None or v < 0.0:
            raise StopNotMeasured(
                f"holdout row {row.get('i')!r} has no finite ego.sim_t / ego.speed_mps (the stop events are placed "
                "in sim time at the recorded speed)")
        ctl = np.asarray(ctl, dtype=np.float64)
        thr, brk = _finite(ctl[_THROTTLE][_H0]), _finite(ctl[_BRAKE][_H0])
        if thr is None or brk is None:
            raise StopNotMeasured(f"holdout row {row.get('i')!r}: the ctl prediction is not finite")
        stops = []
        for sign in _signs_of(row):
            if not isinstance(sign, Mapping):
                continue
            sid, cls = sign.get("id"), sign.get("cls")
            x, y = _finite(sign.get("x")), _finite(sign.get("y"))
            if not sid or not cls:
                skipped["no_id"] += 1
                continue
            if x is None or y is None:
                skipped["no_pose"] += 1
                continue
            stops.append({"id": str(sid), "cls": str(cls), "x": x, "y": y, "state": sign.get("state")})
        out.append({
            "i": row.get("i"), "t": t, "v": v, "split": "holdout",
            "source": str(row.get("source") or row.get("source_root") or ""),
            "ctl_brake": min(1.0, max(0.0, brk)), "ctl_throttle": min(1.0, max(0.0, thr)), "stops": stops,
        })
        skipped["rows"] += 1
    out.sort(key=lambda r: (r["source"], r["t"]))
    return out, skipped


def _approach_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """``[(first, last)]`` row indices of every scored approach, merged where they overlap (one car, one profile)."""
    spans: list[tuple[int, int]] = []
    for ev in se._events_for_source(list(rows)):
        sid = ev["id"]
        first = next(k for k, r in enumerate(rows) if float(r["t"]) >= float(ev["t_entry"]) - 1e-9)
        last, last_t = first, float(rows[first]["t"])
        for k in range(first, len(rows)):
            tgt = next((s for s in rows[k].get("stops") or [] if str(s["id"]) == sid), None)
            if tgt is None:
                if float(rows[k]["t"]) - last_t > se.MAX_DT_S:
                    break
                continue
            if float(tgt["y"]) < -se.LINE_TOL_M:
                break
            last, last_t = k, float(rows[k]["t"])
        spans.append((first, last))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def add_predicted_speed(rows: list[dict[str, Any]]) -> None:
    """Set ``pred_v`` on every row (see the module docstring). Rows are time-ordered per source."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source") or ""), []).append(row)
    for src_rows in by_source.values():
        for row in src_rows:
            row["pred_v"] = float(row["v"])
        for first, last in _approach_spans(src_rows):
            speed = float(src_rows[first]["v"])
            src_rows[first]["pred_v"] = speed
            for k in range(first + 1, last + 1):
                dt = float(src_rows[k]["t"]) - float(src_rows[k - 1]["t"])
                if 0.0 < dt <= se.MAX_DT_S:
                    accel = THROTTLE_ACCEL_MPS2 * src_rows[k]["ctl_throttle"] - BRAKE_DECEL_MPS2 * src_rows[k]["ctl_brake"]
                    speed = max(0.0, speed + accel * dt)
                src_rows[k]["pred_v"] = speed


def _count(res: Mapping[str, Any], names: Sequence[str], key: str) -> int:
    return sum(int(res["slices"].get(n, {}).get(key, 0)) for n in names)


def measure(pairs: Iterable[tuple[Mapping[str, Any], Any]]) -> dict[str, Any]:
    """The stage-2b numbers for one holdout pass. A metric with nothing to count is ``None``, never 0.

    ``missed_stop`` / ``missed_light``: scored stop-sign (+ stop-line) / red-or-yellow-light events with no held full
    stop in the predicted profile (spec: must be 0). ``false_brake``: ``ctl_brake`` > 0.3 onsets with no stop event
    within 40 m, per recorded holdout km. ``stop_side_angle`` and the other slices are in ``slices``.
    """
    rows, skipped = stop_rows(pairs)
    if not rows:
        raise StopNotMeasured("the holdout pass produced no ctl-scored rows to place stop events on")
    add_predicted_speed(rows)
    res = se.score(rows, speed_key="pred_v")
    stop_n, light_n = _count(res, STOP_SLICES, "events"), _count(res, LIGHT_SLICES, "events")
    return {
        "missed_stop": float(_count(res, STOP_SLICES, "missed")) if stop_n else None,
        "missed_light": float(_count(res, LIGHT_SLICES, "missed")) if light_n else None,
        "false_brake": res["false_brakes_per_km"],
        "stop_events": stop_n, "light_events": light_n, "stop_km": res["km"],
        "stop_side_angle_events": int(res["slices"].get("stop_side_angle", {}).get("events", 0)),
        "stop_side_angle_missed": int(res["slices"].get("stop_side_angle", {}).get("missed", 0)),
        "green_false_brake": res["green_false_brakes_per_km"],
        "stop_distance_m": res["stop_distance_m"],
        "skipped": skipped,
    }


def recorded_events(pairs: Iterable[tuple[Mapping[str, Any], Any]]) -> dict[str, int]:
    """Scored event counts of a holdout from the recorded rows alone (before any epoch trains)."""
    rows, _ = stop_rows(pairs)
    for r in rows:
        r["pred_v"] = r["v"]
    res = se.score(rows, speed_key="pred_v") if rows else {"slices": {}}
    return {"stop": _count(res, STOP_SLICES, "events"), "light": _count(res, LIGHT_SLICES, "events"), "rows": len(rows)}
