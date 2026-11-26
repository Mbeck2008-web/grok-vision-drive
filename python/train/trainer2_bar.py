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
strike, **and** any AEB or hand-coded stop-rule activation (GVD call
2026-11-25, a proposal pending Michael). The drive records come from the
IQU45 harness (spec Dependencies 6-8); without them the name is never 2.0.

Pure Python. Training/eval only; nothing on a live path imports it.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

ITER_RMS_GAIN = 0.10
NAME_RMS_GAIN = 0.25
DRIVE_MINUTES = 30.0
MAX_LIVE_VRAM_GB = 4.0
DRIVE_SOURCE = "iqu45"
TIMES_OF_DAY = ("day", "night")
INTERVENTION_KINDS = (
    "takeover",
    "collision",
    "missed_stop_sign",
    "missed_red_light",
    "curb_strike",
    "aeb",
    "hand_rule",
)
_TOL = 1e-12

__all__ = [
    "DRIVE_MINUTES",
    "INTERVENTION_KINDS",
    "ITER_RMS_GAIN",
    "MAX_LIVE_VRAM_GB",
    "NAME_RMS_GAIN",
    "iteration_bar",
    "naming_bar",
    "tally_interventions",
]


def _gain(new: float, prev: float) -> float:
    if not prev > 0:
        raise ValueError(f"baseline lane RMS must be > 0, got {prev!r}")
    return (float(prev) - float(new)) / float(prev)


def _require_card(card: Mapping[str, Any], what: str) -> None:
    for key in ("lane_rms", "stop", "slices", "ssl_pretrain"):
        if key not in card:
            raise ValueError(f"{what} run card has no {key!r} (the card must state SSL on/off)")


def iteration_bar(new: Mapping[str, Any], prev: Mapping[str, Any]) -> dict[str, Any]:
    """Per-iteration merge bar of ``new`` against the previous shipped/best ``prev``."""
    _require_card(new, "new")
    _require_card(prev, "baseline")
    fail: list[str] = []
    gain = _gain(new["lane_rms"], prev["lane_rms"])
    if gain < ITER_RMS_GAIN - _TOL:
        fail.append("lane_rms")
    stop = new["stop"]
    if int(stop["missed"]) != 0:
        fail.append("missed_stop")
    for name in ("stop_side_angle", "red_light"):
        if int(((stop.get("slices") or {}).get(name) or {}).get("missed", 0)) != 0:
            fail.append(name)
    for key, label in (("false_brakes_per_km", "false_brake_rate"), ("green_false_brakes_per_km", "green_false_brake_rate")):
        if float(stop[key]) > float(prev["stop"][key]) + _TOL:
            fail.append(label)
    no_baseline = sorted(set(new["slices"]) - set(prev["slices"]))
    for name, base in sorted(prev["slices"].items()):
        if name not in new["slices"] or float(new["slices"][name]) < float(base) - _TOL:
            fail.append(f"slice:{name}")
    return {"pass": not fail, "fail": fail, "lane_rms_gain": gain, "no_baseline": no_baseline,
            "ssl_pretrain": {"new": bool(new["ssl_pretrain"]), "baseline": bool(prev["ssl_pretrain"])}}


def tally_interventions(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count drive events per intervention kind (others ignored) plus ``total``."""
    out = {kind: 0 for kind in INTERVENTION_KINDS}
    for ev in events:
        kind = ev.get("kind")
        if kind is None or str(kind).strip() == "":
            raise ValueError(f"drive event without kind: {dict(ev)!r}")
        if str(kind) in out:
            out[str(kind)] += 1
    out["total"] = sum(out[k] for k in INTERVENTION_KINDS)
    return out


def naming_bar(
    new: Mapping[str, Any],
    prev: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    drives: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """``{"name": "2.0"|"1.x", "pass", "fail", ...}``. No IQU45 drive records means 1.x."""
    fail: list[str] = []
    it = iteration_bar(new, prev)
    if not it["pass"]:
        fail.append("iteration_bar")
    if it["lane_rms_gain"] < NAME_RMS_GAIN - _TOL:
        fail.append("lane_rms_25")
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
        if float(drive.get("minutes") or 0.0) < DRIVE_MINUTES - _TOL:
            fail.append(f"drive_minutes: {tag} {drive.get('minutes')} < {DRIVE_MINUTES}")
        if drive.get("random_traffic") is not True:
            fail.append(f"random_traffic: {tag}")
        if drive.get("camera_only") is not True:
            fail.append(f"camera_only: {tag}")
        vram = drive.get("live_vram_gb")
        if vram is None or float(vram) > MAX_LIVE_VRAM_GB + _TOL:
            fail.append(f"live_vram: {tag} {vram!r} GB")
        tally = tally_interventions(drive.get("events") or [])
        tallies.append({"drive": tag, **tally})
        if tally["total"] != 0:
            fail.append(f"interventions: {tag} {tally['total']}")
        seen_tod.add(str(drive.get("time_of_day")))
    for tod in TIMES_OF_DAY:
        if tod not in seen_tod:
            fail.append(f"{tod}_drive_missing")
    return {"name": "1.x" if fail else "2.0", "pass": not fail, "fail": fail, "iteration": it, "interventions": tallies}
