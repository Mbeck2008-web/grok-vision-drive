"""Trainer 2.0 decision rules: the per-iteration merge bar and the 2.0 naming bar.

docs/specs/trainer_2.md, "Decision rules (two tiers)". Thresholds decided by
Michael 2026-11-25:

* **iteration bar** (vs the previous shipped/best model on the matched
  holdout): lane RMS >= 10% better; 0 missed stop-sign / red-light brakes,
  the 22-37.5 deg ``stop_side_angle`` slice included; false-brake rate and
  green-light false-brake rate no worse; every labelable scenario slice no
  regression. A slice with no baseline counterpart is reported, not a win;
* **2.0 naming bar**: the iteration bar, lane RMS >= 25% better on the
  matched holdout, and 30-minute day **and** night drives on the held-out
  map (named in ``split_manifest.json``) with random AI traffic, run on
  Desktop-IQU45, camera-only, live VRAM <= 4 GB, with **zero interventions**.
  Anything short ships as 1.x.

Intervention = takeover, collision, missed stop sign, missed red light, curb
strike, **and** any AEB or hand-coded stop-rule activation. The last part is a
**PROPOSAL pending Michael** (GVD call 2026-11-25, not his words): it is the
named config ``BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS`` (and the
``count_backstop`` argument), default True. With False, ``aeb`` / ``hand_rule``
are still tallied and reported under ``backstop`` but do not count toward
``total``. The drive records come from the IQU45 harness (spec Dependencies
6-8); without them the name is never 2.0.

Scenario slices
---------------
A slice is ``{"value": number, "direction": "higher_is_better" |
"lower_is_better", "limit": number}``. The direction is required (a bare number
is refused: assuming "higher is better" let a cut-in false-brake rate rise
0.5 -> 4.0/km and collisions 0 -> 3 pass). ``limit`` is the slice's own pass
criterion (higher: ``value >= limit``; lower: ``value <= limit``, e.g. collisions
``<= 0``). Iteration bar: no regression vs the baseline in the slice's
direction. **2.0 bar: every slice on the card must itself pass its ``limit``**
(a slice with no limit, or none reported at all, fails it), on top of the
iteration bar.

Not measured is not passing: a ``None`` or non-finite (NaN/inf) lane RMS,
false-brake rate, slice value, drive length or VRAM fails its check. ``None``
is what ``stop_events.score`` returns for a rate with no distance (km = 0).

Pure Python. Training/eval only; nothing on a live path imports it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

ITER_RMS_GAIN = 0.10
NAME_RMS_GAIN = 0.25
DRIVE_MINUTES = 30.0
MAX_LIVE_VRAM_GB = 4.0
DRIVE_SOURCE = "iqu45"
TIMES_OF_DAY = ("day", "night")
CORE_INTERVENTION_KINDS = ("takeover", "collision", "missed_stop_sign", "missed_red_light", "curb_strike")
BACKSTOP_KINDS = ("aeb", "hand_rule")
INTERVENTION_KINDS = CORE_INTERVENTION_KINDS + BACKSTOP_KINDS
# PROPOSAL pending Michael (GVD call 2026-11-25, interpreting "zero interventions"): an AEB or hand-coded
# stop-rule activation during the 2.0 drive counts as an intervention. False = tallied and reported only.
BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS = True
HIGHER = "higher_is_better"
LOWER = "lower_is_better"
DIRECTIONS = (HIGHER, LOWER)
_TOL = 1e-12

__all__ = [
    "BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS",
    "BACKSTOP_KINDS",
    "CORE_INTERVENTION_KINDS",
    "DIRECTIONS",
    "DRIVE_MINUTES",
    "HIGHER",
    "INTERVENTION_KINDS",
    "LOWER",
    "ITER_RMS_GAIN",
    "MAX_LIVE_VRAM_GB",
    "NAME_RMS_GAIN",
    "iteration_bar",
    "naming_bar",
    "tally_interventions",
]


def _num(value: Any) -> float | None:
    """A finite float, or None for None / non-numbers / NaN / inf (not measured)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gain(new: Any, prev: Any) -> float | None:
    """Relative lane-RMS gain, or None when the new RMS is not a finite number (never a pass)."""
    base = _num(prev)
    if base is None or not base > 0:
        raise ValueError(f"baseline lane RMS must be a finite number > 0, got {prev!r}")
    got = _num(new)
    return None if got is None else (base - got) / base


def _require_card(card: Mapping[str, Any], what: str) -> None:
    for key in ("lane_rms", "stop", "slices", "ssl_pretrain"):
        if key not in card:
            raise ValueError(f"{what} run card has no {key!r} (the card must state SSL on/off)")


def _slices(card: Mapping[str, Any], what: str) -> dict[str, tuple[float | None, str, float | None]]:
    """``{name: (value, direction, limit)}``; a slice without a direction is refused."""
    out: dict[str, tuple[float | None, str, float | None]] = {}
    for name, rec in dict(card["slices"] or {}).items():
        if not isinstance(rec, Mapping):
            raise ValueError(
                f"{what} slice {name!r} is {rec!r}: a slice is "
                f"{{'value', 'direction': {HIGHER!r}|{LOWER!r}, 'limit'?}}; a bare number has no direction"
            )
        direction = rec.get("direction")
        if direction not in DIRECTIONS:
            raise ValueError(f"{what} slice {name!r} direction {direction!r} not in {DIRECTIONS}")
        out[str(name)] = (_num(rec.get("value")), str(direction), _num(rec.get("limit")))
    return out


def _worse(direction: str, new: float | None, base: float | None) -> bool:
    """True when ``new`` regresses against ``base`` (or either is not a finite number)."""
    if new is None or base is None:
        return True
    return not (new >= base - _TOL) if direction == HIGHER else not (new <= base + _TOL)


def _meets_limit(direction: str, value: float | None, limit: float | None) -> bool:
    if value is None or limit is None:
        return False
    return value >= limit - _TOL if direction == HIGHER else value <= limit + _TOL


def iteration_bar(new: Mapping[str, Any], prev: Mapping[str, Any]) -> dict[str, Any]:
    """Per-iteration merge bar of ``new`` against the previous shipped/best ``prev``."""
    _require_card(new, "new")
    _require_card(prev, "baseline")
    fail: list[str] = []
    gain = _gain(new["lane_rms"], prev["lane_rms"])
    if gain is None or gain < ITER_RMS_GAIN - _TOL:
        fail.append("lane_rms")
    stop = new["stop"]
    if int(stop["missed"]) != 0:
        fail.append("missed_stop")
    for name in ("stop_side_angle", "red_light"):
        rec = (stop.get("slices") or {}).get(name)
        if rec is None:
            fail.append(f"{name}: slice missing from the stop scores")
        elif int(rec.get("missed", 0)) != 0:
            fail.append(name)
    for key, label in (("false_brakes_per_km", "false_brake_rate"), ("green_false_brakes_per_km", "green_false_brake_rate")):
        got, base = _num(stop.get(key)), _num(prev["stop"].get(key))
        if got is None:
            fail.append(f"{label}: not measured (None/NaN, e.g. km = 0)")
        elif base is None:
            fail.append(f"{label}: baseline not measured (None/NaN)")
        elif not got <= base + _TOL:
            fail.append(label)
    new_slices = _slices(new, "new")
    prev_slices = _slices(prev, "baseline")
    no_baseline = sorted(set(new_slices) - set(prev_slices))
    for name, (base, direction, _limit) in sorted(prev_slices.items()):
        if name not in new_slices:
            fail.append(f"slice:{name}")
            continue
        value, new_dir, _lim = new_slices[name]
        if new_dir != direction:
            fail.append(f"slice:{name} direction {new_dir} != baseline {direction}")
        elif _worse(direction, value, base):
            fail.append(f"slice:{name}")
    return {"pass": not fail, "fail": fail, "lane_rms_gain": gain, "no_baseline": no_baseline,
            "ssl_pretrain": {"new": bool(new["ssl_pretrain"]), "baseline": bool(prev["ssl_pretrain"])}}


def tally_interventions(
    events: Iterable[Mapping[str, Any]],
    *,
    count_backstop: bool = BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS,
) -> dict[str, int]:
    """Count drive events per intervention kind, plus ``backstop`` and ``total``.

    An event with a missing or unknown ``kind`` raises (a typo such as
    ``"Takeover"`` must not read as a clean drive). ``total`` counts the core
    kinds, and ``aeb`` / ``hand_rule`` only when ``count_backstop`` (PROPOSAL
    pending Michael); ``backstop`` is their count either way.
    """
    out = {kind: 0 for kind in INTERVENTION_KINDS}
    for ev in events:
        kind = ev.get("kind")
        if kind is None or str(kind).strip() == "":
            raise ValueError(f"drive event without kind: {dict(ev)!r}")
        if str(kind) not in out:
            raise ValueError(f"unknown drive event kind {kind!r}; expected one of {INTERVENTION_KINDS}")
        out[str(kind)] += 1
    out["backstop"] = sum(out[k] for k in BACKSTOP_KINDS)
    kinds = INTERVENTION_KINDS if count_backstop else CORE_INTERVENTION_KINDS
    out["total"] = sum(out[k] for k in kinds)
    return out


def naming_bar(
    new: Mapping[str, Any],
    prev: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    drives: Iterable[Mapping[str, Any]] | None,
    count_backstop: bool = BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS,
) -> dict[str, Any]:
    """``{"name": "2.0"|"1.x", "pass", "fail", ...}``. No IQU45 drive records means 1.x.

    Beyond the iteration bar: lane RMS >= 25% better, **every scenario slice on
    the new card passes its own ``limit``** (and at least one slice is
    reported), and the day and night drives. ``count_backstop`` is the
    PROPOSAL pending Michael on AEB / hand-rule activations.
    """
    fail: list[str] = []
    it = iteration_bar(new, prev)
    if not it["pass"]:
        fail.append("iteration_bar")
    if it["lane_rms_gain"] is None or it["lane_rms_gain"] < NAME_RMS_GAIN - _TOL:
        fail.append("lane_rms_25")
    new_slices = _slices(new, "new")
    if not new_slices:
        fail.append("slices_pass: no scenario slice reported")
    for name, (value, direction, limit) in sorted(new_slices.items()):
        if limit is None:
            fail.append(f"slices_pass: {name} has no pass limit")
        elif not _meets_limit(direction, value, limit):
            fail.append(f"slices_pass: {name} {value!r} fails {direction} limit {limit!r}")
    held_map = split_manifest.get("heldout_map")
    if held_map is None or str(held_map).strip() == "":
        fail.append("heldout_map: split manifest names none")
    tallies = []
    seen_tod: set[str] = set()
    for drive in list(drives or []):
        tag = f"{drive.get('time_of_day')}@{drive.get('map')}"
        if str(drive.get("source")) != DRIVE_SOURCE:
            fail.append(f"iqu45: {tag} drive not from {DRIVE_SOURCE}")
            continue
        if held_map is None or str(drive.get("map")) != str(held_map):
            fail.append(f"heldout_map: {tag} not on held-out map {held_map!r}")
        minutes = _num(drive.get("minutes"))
        if minutes is None or not minutes >= DRIVE_MINUTES - _TOL:
            fail.append(f"drive_minutes: {tag} {drive.get('minutes')!r} < {DRIVE_MINUTES}")
        if drive.get("random_traffic") is not True:
            fail.append(f"random_traffic: {tag}")
        if drive.get("camera_only") is not True:
            fail.append(f"camera_only: {tag}")
        vram = _num(drive.get("live_vram_gb"))
        if vram is None or not vram <= MAX_LIVE_VRAM_GB + _TOL:
            fail.append(f"live_vram: {tag} {drive.get('live_vram_gb')!r} GB")
        tally = tally_interventions(drive.get("events") or [], count_backstop=count_backstop)
        tallies.append({"drive": tag, **tally})
        if tally["total"] != 0:
            fail.append(f"interventions: {tag} {tally['total']}")
        seen_tod.add(str(drive.get("time_of_day")))
    for tod in TIMES_OF_DAY:
        if tod not in seen_tod:
            fail.append(f"{tod}_drive_missing")
    return {"name": "1.x" if fail else "2.0", "pass": not fail, "fail": fail, "iteration": it, "interventions": tallies}
