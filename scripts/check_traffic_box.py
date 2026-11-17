#!/usr/bin/env python3
"""IQU45 smoke check: does traffic ``box.ctr`` follow the moving car?

``TRAFFIC_CHUNK`` records ``box`` from Lua ``getSpawnWorldOOBB``. The method
name contains "Spawn"; this repo has no first-party BeamNG source that states
whether ``getCenter`` / ``getAxis`` follow the live pose or stay at spawn.
This script is the proof required before a long recording.

For each moving ``gvd_traffic_*`` car in a recording's ``state.jsonl`` it
prints the box non-null rate and max ``|box.ctr - pos|``. It exits non-zero
(do not start the long run) when:

* no ``gvd_traffic_*`` car travelled more than ``MOVE_M``, or a moving car
  has no box (unproven);
* a car travelled more than ``MOVE_M`` while ``box.ctr`` stayed within
  ``SPAWN_STUCK_M`` of its first reported centre (spawn-stuck).

If ctr is spawn-stuck: keep only ``half`` and build ``ctr`` / ``ax`` from
live ``pos`` / ``dir`` / ``up``, or switch the Lua read to a proven live
OOBB call, then re-smoke. Do not start the long run on an unproven box.

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

# A companion that travelled this far (world metres, from its first pos)
# is "moving". Parked cars are ignored for the live-box proof.
MOVE_M = 10.0
# If that car's box.ctr never leaves this radius of its first reported
# centre, the box is treated as stuck at spawn.
SPAWN_STUCK_M = 1.0
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


def _car_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    first_pos: tuple[float, float, float] | None = None
    first_ctr: tuple[float, float, float] | None = None
    pos_span = 0.0
    ctr_span = 0.0
    max_ctr_pos = 0.0
    n_box = 0
    n_ctr_pos = 0
    for sample in samples:
        pos = _vec3(sample.get("pos"))
        if pos is None:
            continue
        if first_pos is None:
            first_pos = pos
        pos_span = max(pos_span, _dist(pos, first_pos))
        box = sample.get("box")
        ctr = _vec3(box.get("ctr") if isinstance(box, Mapping) else None)
        if ctr is None:
            continue
        n_box += 1
        if first_ctr is None:
            first_ctr = ctr
        ctr_span = max(ctr_span, _dist(ctr, first_ctr))
        max_ctr_pos = max(max_ctr_pos, _dist(ctr, pos))
        n_ctr_pos += 1
    n = len(samples)
    moving = first_pos is not None and pos_span > MOVE_M
    spawn_stuck = (
        moving
        and first_ctr is not None
        and ctr_span < SPAWN_STUCK_M
    )
    unproven = moving and n_box == 0
    return {
        "ticks": n,
        "moved_m": pos_span,
        "box_nonnull": (n_box / n) if n else 0.0,
        "max_ctr_pos_m": max_ctr_pos if n_ctr_pos else None,
        "ctr_span_m": ctr_span if n_box else None,
        "moving": moving,
        "spawn_stuck": spawn_stuck,
        "unproven": unproven,
    }


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-name stats for ``gvd_traffic_*`` cars found in ``state.jsonl`` rows."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    n_rows = 0
    n_traffic = 0
    for row in rows:
        n_rows += 1
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
            by_name.setdefault(name, []).append(dict(car))
    cars = {name: _car_stats(samples) for name, samples in sorted(by_name.items())}
    moving = {name: stats for name, stats in cars.items() if stats["moving"]}
    stuck = [name for name, stats in moving.items() if stats["spawn_stuck"]]
    unproven = [name for name, stats in moving.items() if stats["unproven"]]
    if stuck:
        status = "SPAWN-STUCK"
        ok = False
    elif not moving:
        status = "UNPROVEN"
        ok = False
    elif unproven:
        status = "UNPROVEN"
        ok = False
    else:
        status = "LIVE"
        ok = True
    return {
        "ticks": n_rows,
        "traffic_rows": n_traffic,
        "cars": cars,
        "moving": sorted(moving),
        "stuck": stuck,
        "unproven": unproven,
        "status": status,
        "ok": ok,
    }


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
        if stats["spawn_stuck"]:
            tag = "SPAWN-STUCK"
        elif stats["unproven"]:
            tag = "UNPROVEN"
        elif stats["moving"]:
            tag = "LIVE"
        else:
            tag = "PARKED"
        max_s = "none" if max_cp is None else f"{max_cp:.3f}"
        span_s = "none" if ctr_span is None else f"{ctr_span:.3f}"
        lines.append(
            f"  {name}  ticks={stats['ticks']}  moved_m={stats['moved_m']:.3f}  "
            f"box_nonnull={stats['box_nonnull']:.3f}  max_|ctr-pos|_m={max_s}  "
            f"ctr_span_m={span_s}  {tag}"
        )
    if summary["ok"]:
        n = len(summary["moving"])
        lines.append(
            f"PASS: {n} moving {NAME_PREFIX}* car(s); box.ctr follows pos. "
            "Long run may proceed on this evidence."
        )
    elif summary["stuck"]:
        names = ", ".join(summary["stuck"])
        lines.append(
            f"FAIL: {names} box.ctr stayed within {SPAWN_STUCK_M:g} m of its "
            f"first centre while the car moved more than {MOVE_M:g} m. "
            "Do not start the long run. Keep only half and build ctr/ax from "
            "pos/dir/up, or switch to a live OOBB call, then re-smoke."
        )
    elif not summary["moving"]:
        lines.append(
            f"FAIL: no {NAME_PREFIX}* car moved more than {MOVE_M:g} m; "
            "box is unproven. Do not start the long run."
        )
    else:
        names = ", ".join(summary["unproven"])
        lines.append(
            f"FAIL: {names} moved more than {MOVE_M:g} m but reported no box; "
            "box is unproven. Do not start the long run."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-check traffic box.ctr against pos before a long run."
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
