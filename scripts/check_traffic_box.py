#!/usr/bin/env python3
"""IQU45 smoke check: decide whether traffic ``box`` is live or spawn-stuck.

``TRAFFIC_CHUNK`` records ``box`` from Lua ``getSpawnWorldOOBB``. The method
name contains "Spawn"; this repo has no first-party BeamNG source that states
whether ``getCenter`` / ``getAxis`` follow the live pose or stay at spawn.
This script is the proof required before a long recording. It decides on the
IQU45 smoke recording; do not assume the result from the Lua name.

Companions are deleted and recreated under the same ``gvd_traffic_*`` names
at each ego spawn (``python/collect/spawn_mix.py``). Samples are therefore
split on a ``veh_meta`` row (a new ego spawn) or a ``pos`` jump of more than
``JUMP_M``. A teleport is neither motion nor a centre change.

A spawn segment is scored on its moving boxed samples (box present and
displacement from that segment's first ``pos`` is greater than ``MOVE_M``):

* ``|box.ctr - pos|`` must be <= ``|half|`` (Euclidean) + ``CTR_POS_SLACK_M``.
* One box axis must have ``|dot|`` >= ``AXIS_DOT_MIN`` with ``dir``, and
  one with ``up``.
* LIVE requires at least ``MIN_MOVING_BOXED`` such samples in a segment,
  all passing both tests, and no violating sample in any segment.

If the box is spawn-stuck or axes stay at spawn: keep only ``half`` and
build ``ctr`` / ``ax`` from live ``pos`` / ``dir`` / ``up``, or switch the
Lua read to a proven live OOBB call, then re-smoke. Do not start the long
run on an unproven box.

Usage::

    py -3 scripts/check_traffic_box.py <recording_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# A companion that travelled this far (world metres, from the first pos of
# its current spawn segment) is "moving" in that segment.
MOVE_M = 10.0
# A pos jump this far between successive samples for one name starts a new
# spawn segment. Traffic is 2 Hz, so this is well above one tick of travel.
JUMP_M = 50.0
# LIVE needs at least this many moving boxed samples in one spawn segment.
MIN_MOVING_BOXED = 6
# |ctr - pos| limit on a moving boxed sample: |half| + this slack (metres).
CTR_POS_SLACK_M = 0.5
# On a moving boxed sample, one axis must align with dir and one with up.
AXIS_DOT_MIN = 0.98
NAME_PREFIX = "gvd_traffic_"


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(value: Any) -> tuple[float, float, float] | None:
    vec = value if isinstance(value, tuple) and len(value) == 3 else _vec3(value)
    if vec is None:
        return None
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _half_norm(box: Mapping[str, Any] | None) -> float | None:
    if not isinstance(box, Mapping):
        return None
    half = _vec3(box.get("half"))
    if half is None or any(h <= 0.0 for h in half):
        return None
    return math.sqrt(half[0] ** 2 + half[1] ** 2 + half[2] ** 2)


def ctr_pos_limit(box: Mapping[str, Any] | None) -> float | None:
    """Max allowed ``|box.ctr - pos|`` on a moving boxed sample."""
    norm = _half_norm(box)
    if norm is None:
        return None
    return norm + CTR_POS_SLACK_M


def axes_aligned(box: Mapping[str, Any] | None, direction: Any, up: Any) -> bool:
    """True when one axis matches ``dir`` and one matches ``up`` (|dot| threshold)."""
    if not isinstance(box, Mapping):
        return False
    axes = box.get("ax")
    if not isinstance(axes, (list, tuple)) or len(axes) < 3:
        return False
    dir_u, up_u = _unit(direction), _unit(up)
    if dir_u is None or up_u is None:
        return False
    units = [_unit(a) for a in axes[:3]]
    if any(a is None for a in units):
        return False
    has_dir = any(abs(_dot(a, dir_u)) >= AXIS_DOT_MIN for a in units)
    has_up = any(abs(_dot(a, up_u)) >= AXIS_DOT_MIN for a in units)
    return has_dir and has_up


def resolve_state(path: Path) -> Path:
    """``<dir>/state.jsonl`` or a jsonl file itself."""
    path = path.expanduser()
    if path.is_dir():
        return path / "state.jsonl"
    return path


def load_state(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def split_segments(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[list[dict[str, Any]]]], int, int]:
    """Per-name spawn segments. ``veh_meta`` or a pos jump starts a new one."""
    by_name: dict[str, list[list[dict[str, Any]]]] = {}
    last_pos: dict[str, tuple[float, float, float]] = {}
    last_gen: dict[str, int] = {}
    spawn_gen = 0
    n_rows = 0
    n_traffic = 0
    for row in rows:
        n_rows += 1
        if "veh_meta" in row:
            spawn_gen += 1
        traffic = row.get("traffic")
        if not isinstance(traffic, list):
            continue
        n_traffic += 1
        for car in traffic:
            if not isinstance(car, Mapping):
                continue
            name = str(car.get("name") or "")
            if not name.startswith(NAME_PREFIX):
                continue
            sample = dict(car)
            pos = _vec3(sample.get("pos"))
            segs = by_name.setdefault(name, [])
            new_seg = not segs or last_gen.get(name) != spawn_gen
            if (
                not new_seg
                and pos is not None
                and name in last_pos
                and _dist(pos, last_pos[name]) > JUMP_M
            ):
                new_seg = True
            if new_seg:
                segs.append([])
            segs[-1].append(sample)
            last_gen[name] = spawn_gen
            if pos is not None:
                last_pos[name] = pos
    return by_name, n_rows, n_traffic


def _segment_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    first_pos: tuple[float, float, float] | None = None
    first_ctr: tuple[float, float, float] | None = None
    pos_span = 0.0
    ctr_span = 0.0
    max_ctr_pos = 0.0
    n_box = 0
    n_ctr_pos = 0
    n_moving_boxed = 0
    spawn_stuck = False
    axes_stuck = False
    for sample in samples:
        pos = _vec3(sample.get("pos"))
        if pos is None:
            continue
        if first_pos is None:
            first_pos = pos
        pos_span = max(pos_span, _dist(pos, first_pos))
        box = sample.get("box") if isinstance(sample.get("box"), Mapping) else None
        ctr = _vec3(box.get("ctr") if box is not None else None)
        if ctr is None or box is None:
            continue
        n_box += 1
        if first_ctr is None:
            first_ctr = ctr
        ctr_span = max(ctr_span, _dist(ctr, first_ctr))
        ctr_pos = _dist(ctr, pos)
        max_ctr_pos = max(max_ctr_pos, ctr_pos)
        n_ctr_pos += 1
        if _dist(pos, first_pos) <= MOVE_M:
            continue
        n_moving_boxed += 1
        limit = ctr_pos_limit(box)
        if limit is None or ctr_pos > limit:
            spawn_stuck = True
        if not axes_aligned(box, sample.get("dir"), sample.get("up")):
            axes_stuck = True
    n = len(samples)
    moving = first_pos is not None and pos_span > MOVE_M
    unproven = moving and n_moving_boxed < MIN_MOVING_BOXED and not spawn_stuck and not axes_stuck
    # A moving stretch with no boxed samples at all is unproven (not stuck).
    if moving and n_box == 0:
        unproven = True
    return {
        "ticks": n,
        "moved_m": pos_span,
        "box_nonnull": (n_box / n) if n else 0.0,
        "max_ctr_pos_m": max_ctr_pos if n_ctr_pos else None,
        "ctr_span_m": ctr_span if n_box else None,
        "n_box": n_box,
        "n_moving_boxed": n_moving_boxed,
        "moving": moving,
        "spawn_stuck": spawn_stuck,
        "axes_stuck": axes_stuck,
        "unproven": unproven,
        "proven_live": (
            moving
            and n_moving_boxed >= MIN_MOVING_BOXED
            and not spawn_stuck
            and not axes_stuck
        ),
    }


def _merge_stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(s["ticks"] for s in segments)
    n_box = sum(s["n_box"] for s in segments)
    moved = sum(s["moved_m"] for s in segments)
    ctr_pos_vals = [s["max_ctr_pos_m"] for s in segments if s["max_ctr_pos_m"] is not None]
    ctr_span_vals = [s["ctr_span_m"] for s in segments if s["ctr_span_m"] is not None]
    n_moving_boxed = sum(s["n_moving_boxed"] for s in segments)
    moving = any(s["moving"] for s in segments)
    spawn_stuck = any(s["spawn_stuck"] for s in segments)
    axes_stuck = any(s["axes_stuck"] for s in segments)
    proven_live = any(s["proven_live"] for s in segments)
    unproven = (not spawn_stuck) and (not axes_stuck) and (not proven_live)
    return {
        "ticks": n,
        "moved_m": moved,
        "box_nonnull": (n_box / n) if n else 0.0,
        "max_ctr_pos_m": max(ctr_pos_vals) if ctr_pos_vals else None,
        "ctr_span_m": max(ctr_span_vals) if ctr_span_vals else None,
        "n_box": n_box,
        "n_moving_boxed": n_moving_boxed,
        "n_spawns": len(segments),
        "moving": moving,
        "spawn_stuck": spawn_stuck,
        "axes_stuck": axes_stuck,
        "unproven": unproven,
        "proven_live": proven_live and not spawn_stuck and not axes_stuck,
    }


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-name stats for ``gvd_traffic_*`` cars found in ``state.jsonl`` rows."""
    by_name, n_rows, n_traffic = split_segments(rows)
    cars = {
        name: _merge_stats([_segment_stats(seg) for seg in segs])
        for name, segs in sorted(by_name.items())
    }
    stuck = [name for name, stats in cars.items() if stats["spawn_stuck"]]
    axes = [name for name, stats in cars.items() if stats["axes_stuck"] and name not in stuck]
    proven = [name for name, stats in cars.items() if stats["proven_live"]]
    unproven = [name for name, stats in cars.items() if stats["unproven"] and stats["moving"]]
    if stuck:
        status = "SPAWN-STUCK"
        ok = False
    elif axes:
        status = "AXES-STUCK"
        ok = False
    elif not proven:
        status = "UNPROVEN"
        ok = False
    else:
        status = "LIVE"
        ok = True
    return {
        "ticks": n_rows,
        "traffic_rows": n_traffic,
        "cars": cars,
        "moving": sorted(name for name, stats in cars.items() if stats["moving"]),
        "stuck": stuck,
        "axes": axes,
        "unproven": unproven,
        "proven": proven,
        "status": status,
        "ok": ok,
    }


def _tag(stats: Mapping[str, Any]) -> str:
    if stats["spawn_stuck"]:
        return "SPAWN-STUCK"
    if stats["axes_stuck"]:
        return "AXES-STUCK"
    if stats["proven_live"]:
        return "LIVE"
    if stats["moving"]:
        return "UNPROVEN"
    return "PARKED"


def format_report(summary: Mapping[str, Any], source: str) -> str:
    lines = [
        f"check_traffic_box: {source}  ticks={summary['ticks']}  "
        f"traffic_rows={summary['traffic_rows']}"
    ]
    cars = summary["cars"]
    if not cars:
        lines.append(f"  (no {NAME_PREFIX}* cars in traffic tags)")
    for name, stats in cars.items():
        max_cp = stats["max_ctr_pos_m"]
        ctr_span = stats["ctr_span_m"]
        max_s = "none" if max_cp is None else f"{max_cp:.3f}"
        span_s = "none" if ctr_span is None else f"{ctr_span:.3f}"
        lines.append(
            f"  {name}  ticks={stats['ticks']}  spawns={stats['n_spawns']}  "
            f"moved_m={stats['moved_m']:.3f}  box_nonnull={stats['box_nonnull']:.3f}  "
            f"moving_boxed={stats['n_moving_boxed']}  max_|ctr-pos|_m={max_s}  "
            f"ctr_span_m={span_s}  {_tag(stats)}"
        )
    if summary["ok"]:
        n = len(summary["proven"])
        lines.append(
            f"PASS: {n} {NAME_PREFIX}* car(s) met the live-box bar on the "
            "IQU45 smoke check "
            f"(|ctr-pos| <= |half|+{CTR_POS_SLACK_M:g} m on "
            f">= {MIN_MOVING_BOXED} moving boxed samples in a spawn segment, "
            f"axes |dot|>= {AXIS_DOT_MIN:g} with dir and up; samples split "
            "on veh_meta or a pos jump). The smoke run decides this; do not "
            "assume it from the Lua name. Long run may proceed on this evidence."
        )
    elif summary["stuck"]:
        names = ", ".join(summary["stuck"])
        lines.append(
            f"FAIL: {names} |box.ctr - pos| exceeded |half|+{CTR_POS_SLACK_M:g} m "
            "on a moving boxed sample. Do not start the long run. Keep only "
            "half and build ctr/ax from pos/dir/up, or switch to a live OOBB "
            "call, then re-smoke."
        )
    elif summary["axes"]:
        names = ", ".join(summary["axes"])
        lines.append(
            f"FAIL: {names} box axes were not aligned with dir/up "
            f"(|dot| < {AXIS_DOT_MIN:g}) on a moving boxed sample. Do not "
            "start the long run. Keep only half and build ctr/ax from "
            "pos/dir/up, or switch to a live OOBB call, then re-smoke."
        )
    elif not summary["moving"]:
        lines.append(
            f"FAIL: no {NAME_PREFIX}* car moved more than {MOVE_M:g} m in a "
            "spawn segment; box is unproven. Do not start the long run."
        )
    else:
        lines.append(
            f"FAIL: no {NAME_PREFIX}* car had >= {MIN_MOVING_BOXED} moving "
            f"boxed samples (travel > {MOVE_M:g} m) in one spawn segment; "
            "box is unproven. Do not start the long run."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="IQU45 smoke check: decide if traffic box is live before a long run."
    )
    ap.add_argument("recording", type=Path, help="Recording dir or state.jsonl")
    args = ap.parse_args(argv)
    state = resolve_state(args.recording)
    if not state.is_file():
        print(f"check_traffic_box: missing {state}", file=sys.stderr)
        return 2
    rows = load_state(state)
    if not rows:
        print(f"check_traffic_box: empty {state}", file=sys.stderr)
        return 2
    summary = summarize(rows)
    print(format_report(summary, str(state)))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
