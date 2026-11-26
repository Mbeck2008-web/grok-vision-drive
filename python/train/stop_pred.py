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
* **the predicted speed profile (``pred_v``)** is integrated from the ctl head's own outputs at h = 0 and a fitted
  longitudinal map, open-loop at the recorded row times::

      dv/dt = K_THR * ctl_throttle - K_BRK * ctl_brake - K_DRAG * v        (v floored at 0)

  ``v`` starts at the recorded speed at the first row of an approach (t0, the same entry row the 2 m/s gate uses) and
  after t0 the profile reads only the ctl predictions and its own state: never the recorded speed, a wheel speed or
  an acceleration, so a recorded human stop never credits the model. ``K_THR``, ``K_BRK``, ``K_DRAG`` (all >= 0) are
  **fitted by ``fit_motion`` on manifest train rows only**, before the metric runs (``K_BRK`` must come out > 0 or
  the stage is not measured). ``fit_motion`` refuses a fit set that holds a holdout row id.
* **brake = 0** (and throttle = 0): the car **coasts**, ``dv/dt = -K_DRAG * v``, and a stopped car stays stopped (no
  creep): holding a stop needs no brake, leaving it needs throttle. Throttle > 0 adds ``K_THR * throttle`` and so
  ends a hold. This is written in ``docs/specs/trainer_2.md`` (stage 2b) and pinned by ``check_brake_zero_behaviour``.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from python.train import stop_events as se

# fit_motion: which consecutive train rows are usable, and how many it takes to fit a map at all.
FIT_MAX_DT_S = 0.25  # same as ctl_targets.COND_MAX_DT_S: a longer step is a hole, not an acceleration
FIT_MIN_DT_S = 0.02
FIT_MIN_ROWS = 30
FIT_MIN_BRAKE_ROWS = 10
FIT_BRAKE_ON = 0.1  # a fit row with brake_ai above this is a braking row
FIT_MAX_ACCEL_MPS2 = 15.0  # an |dv/dt| above this is a speed glitch (respawn), not a car

# pred_v of a row that is in no approach: it is never inside a stop window; any value >= 0.5 m/s is 'not a stop'.
NOT_SCORED_V = 1000.0
# Row of ``ctl`` (3, K): steer, throttle, brake; column 0 is h = 0 (now).
_THROTTLE, _BRAKE, _H0 = 1, 2, 0
# Stage metric names (trainer2_recipe.STAGE_PRIMARY_METRICS["2b"]) and what they count.
STOP_SLICES = ("stop_sign", "stop_line")
LIGHT_SLICES = ("red_light", "yellow_light")


class StopNotMeasured(ValueError):
    """The holdout cannot score stage 2b (no usable events / rows / fit). Says exactly why; nothing is made up."""


class Motion:
    """The fitted longitudinal map ``dv/dt = k_thr*throttle - k_brk*brake - k_drag*v`` and what it was fitted on."""

    __slots__ = ("k_thr", "k_brk", "k_drag", "n_rows", "n_brake_rows", "ids")

    def __init__(self, k_thr: float, k_brk: float, k_drag: float, *, n_rows: int = 0, n_brake_rows: int = 0,
                 ids: frozenset = frozenset()) -> None:
        for name, value in (("k_thr", k_thr), ("k_brk", k_brk), ("k_drag", k_drag)):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise StopNotMeasured(f"motion {name}={value!r} must be a finite number >= 0")
        if float(k_brk) <= 0.0:
            raise StopNotMeasured("motion k_brk must be > 0: braking has to slow the car")
        self.k_thr, self.k_brk, self.k_drag = float(k_thr), float(k_brk), float(k_drag)
        self.n_rows, self.n_brake_rows, self.ids = int(n_rows), int(n_brake_rows), frozenset(ids)

    def accel(self, throttle: float, brake: float, v: float) -> float:
        return self.k_thr * throttle - self.k_brk * brake - self.k_drag * v

    def as_dict(self) -> dict[str, Any]:
        return {"k_thr": self.k_thr, "k_brk": self.k_brk, "k_drag": self.k_drag, "n_rows": self.n_rows,
                "n_brake_rows": self.n_brake_rows}


def fit_motion(train_rows: Iterable[Mapping[str, Any]], *, holdout_ids: Iterable[Any]) -> Motion:
    """Fit ``Motion`` on recorded **train** rows only: the speed change between consecutive rows against the recorded AI
    pedals (``pedals.throttle_ai`` / ``pedals.brake_ai``) and the speed.

    ``train_rows`` are full-recording rows of the manifest's train rows (``i``, ``segment``, ``ego.sim_t``,
    ``ego.speed_mps``, ``pedals.*``); ``holdout_ids`` are the run's holdout row ids. A row whose id is a holdout id is a
    leak into the fit and is **refused** (``StopNotMeasured``), never silently dropped. Non-negative least squares
    (a negative coefficient is set to 0 and the rest refitted); ``k_brk`` must come out > 0 on enough braking rows.
    """
    hold = {str(i) for i in holdout_ids}
    rows = list(train_rows)
    leaked = sorted(str(r.get("i")) for r in rows if str(r.get("i")) in hold)
    if leaked:
        raise StopNotMeasured(f"the motion fit set holds {len(leaked)} holdout row id(s) (e.g. {leaked[:3]}): fit on train rows only")
    rows.sort(key=lambda r: (str(r.get("segment")), _finite(_get(r, "ego", "sim_t")) or 0.0))
    xs: list[list[float]] = []
    ys: list[float] = []
    braking = 0
    for a, b in zip(rows, rows[1:]):
        if a.get("segment") != b.get("segment"):
            continue
        ta, tb = _finite(_get(a, "ego", "sim_t")), _finite(_get(b, "ego", "sim_t"))
        va, vb = _finite(_get(a, "ego", "speed_mps")), _finite(_get(b, "ego", "speed_mps"))
        thr, brk = _finite(_get(a, "pedals", "throttle_ai")), _finite(_get(a, "pedals", "brake_ai"))
        if None in (ta, tb, va, vb, thr, brk) or va < 0.0 or vb < 0.0:
            continue
        if vb <= 0.0:
            continue  # at or onto a standstill the speed is floored: the step says nothing about the pedals
        if _get(a, "veh", "reverse") or (_finite(_get(a, "pedals", "parkingbrake_ai")) or 0.0) > 0.5:
            continue
        dt = tb - ta
        if not FIT_MIN_DT_S <= dt <= FIT_MAX_DT_S:
            continue
        accel = (vb - va) / dt
        if abs(accel) > FIT_MAX_ACCEL_MPS2:
            continue
        xs.append([thr, brk, va])
        ys.append(accel)
        braking += int(brk > FIT_BRAKE_ON)
    if len(xs) < FIT_MIN_ROWS or braking < FIT_MIN_BRAKE_ROWS:
        raise StopNotMeasured(
            f"the motion fit needs >= {FIT_MIN_ROWS} usable train row pairs with >= {FIT_MIN_BRAKE_ROWS} braking ones "
            f"(brake_ai > {FIT_BRAKE_ON}); have {len(xs)} / {braking}")
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    # accel = k_thr*thr - k_brk*brk - k_drag*v  ->  columns (thr, -brk, -v); all three coefficients are >= 0
    design = x * np.asarray([1.0, -1.0, -1.0])
    active = [0, 1, 2]
    coef = np.zeros(3)
    while active:
        sol, *_ = np.linalg.lstsq(design[:, active], y, rcond=None)
        if (sol >= 0.0).all():
            coef[active] = sol
            break
        active.pop(int(np.argmin(sol)))
    if coef[1] <= 0.0:
        raise StopNotMeasured("the train rows show no deceleration under braking (fitted k_brk <= 0): no brake map to fit")
    return Motion(coef[0], coef[1], coef[2], n_rows=len(xs), n_brake_rows=braking,
                  ids=frozenset(str(r.get("i")) for r in rows))


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
    """``[(first, last)]`` row indices of every scored approach, merged where they overlap (one car, one profile).

    An approach starts where ``stop_events`` finds its entry (the target within 40 m at a recorded speed >= 2 m/s: that
    gate is t0) and ends at the stop line + 1 m, from the GT target positions. Nothing after t0 reads the recorded
    speed: the rows are scored here with a stub profile that is a held stop everywhere, so a lamp-turned-green rule
    (``RED_TO_GREEN_PROCEED_IS_MISS`` False) can never drop an approach because of what the recorded car did.
    """
    stub = [dict(r, pred_v=0.0) for r in rows]
    spans: list[tuple[int, int]] = []
    for ev in se._events_for_source(stub, "pred_v"):
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


def add_predicted_speed(rows: list[dict[str, Any]], motion: Motion) -> None:
    """Set ``pred_v`` on every row (see the module docstring). Rows are time-ordered per source.

    The only recorded speed read is ``v`` of the first row of an approach (t0). After t0 the update uses
    ``ctl_throttle`` / ``ctl_brake``, the row times and the profile's own state. A row outside every approach is never
    in a stop window; it gets ``NOT_SCORED_V`` (a speed that is not a stop), not the recorded speed.
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source") or ""), []).append(row)
    for src_rows in by_source.values():
        for row in src_rows:
            row["pred_v"] = NOT_SCORED_V
        for first, last in _approach_spans(src_rows):
            speed = float(src_rows[first]["v"])
            src_rows[first]["pred_v"] = speed
            for k in range(first + 1, last + 1):
                dt = float(src_rows[k]["t"]) - float(src_rows[k - 1]["t"])
                if 0.0 < dt <= se.MAX_DT_S:
                    speed = max(0.0, speed + motion.accel(src_rows[k]["ctl_throttle"], src_rows[k]["ctl_brake"], speed) * dt)
                src_rows[k]["pred_v"] = speed


def _count(res: Mapping[str, Any], names: Sequence[str], key: str) -> int:
    return sum(int(res["slices"].get(n, {}).get(key, 0)) for n in names)


def measure(pairs: Iterable[tuple[Mapping[str, Any], Any]], motion: Motion) -> dict[str, Any]:
    """The stage-2b numbers for one holdout pass. A metric with nothing to count is ``None``, never 0.

    ``missed_stop`` / ``missed_light``: scored stop-sign (+ stop-line) / red-or-yellow-light events with no held full
    stop in the predicted profile (spec: must be 0). ``false_brake``: ``ctl_brake`` > 0.3 onsets with no stop event
    within 40 m, per recorded holdout km. ``stop_side_angle`` and the other slices are in ``slices``.
    """
    rows, skipped = stop_rows(pairs)
    if not rows:
        raise StopNotMeasured("the holdout pass produced no ctl-scored rows to place stop events on")
    add_predicted_speed(rows, motion)
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
        "skipped": skipped, "motion": motion.as_dict(),
    }


def recorded_events(pairs: Iterable[tuple[Mapping[str, Any], Any]]) -> dict[str, int]:
    """Scored event counts of a holdout from the recorded rows alone (before any epoch trains)."""
    rows, _ = stop_rows(pairs)
    for r in rows:
        r["pred_v"] = 0.0  # every approach counts as held: only the number of events is wanted here
    res = se.score(rows, speed_key="pred_v") if rows else {"slices": {}}
    return {"stop": _count(res, STOP_SLICES, "events"), "light": _count(res, LIGHT_SLICES, "events"), "rows": len(rows)}


def preflight(train_rows: Iterable[Mapping[str, Any]], hold_rows: Iterable[Mapping[str, Any]], *,
              default_root: Any = None) -> dict[str, Any]:
    """Can stage 2b be scored on this split? Answered from the recorded rows alone, before any stage trains.

    ``train_rows`` / ``hold_rows`` are the run's manifest split (``legacy_split_rows``). Joins the ``signs.jsonl`` ids
    onto copies of the holdout rows (``default_root`` is the strip root of a single-root run), then needs >= 1
    stop-sign/line and >= 1 red/yellow-light event in the holdout and a motion fit on the train rows. Raises
    ``StopNotMeasured`` with the reason; a run whose plan has stage 2b calls this first so SSL / stage 1 / stage 2 are
    not trained for hours before 2b finds the recording cannot be scored.
    """
    try:
        return _preflight(train_rows, hold_rows, default_root)
    except StopNotMeasured as exc:
        if "stage 2b" in str(exc):
            raise
        raise StopNotMeasured(f"stage 2b cannot be scored: {exc}") from exc


def _preflight(train_rows: Iterable[Mapping[str, Any]], hold_rows: Iterable[Mapping[str, Any]], default_root: Any) -> dict[str, Any]:
    from python.train.intersection_labels import attach_sign_labels

    held = [dict(r) for r in hold_rows]
    held_ids = {r.get("i") for r in held}
    attach_sign_labels(held, default_root=default_root)
    found = recorded_events((r, np.zeros((3, 1))) for r in held)
    if not found["stop"] or not found["light"]:
        raise StopNotMeasured(
            f"stage 2b cannot be scored on this holdout: {found['stop']} stop-sign/line and {found['light']} "
            "red/yellow-light events (needs >= 1 of each: a miss count with no events would be a vacuous 0). "
            "Record more intersections with signals.jsonl + export_signal_labels.py, or move the holdout")
    motion = fit_motion([r for r in train_rows if r.get("i") not in held_ids], holdout_ids=held_ids)
    return {"events": found, "motion": motion}
