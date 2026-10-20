#!/usr/bin/env python3
"""Record scene-net strips while BeamNG's AI drives.

Tech is already running. This attaches to port 25252 (config/tech.yaml),
does not launch the game, and does not call BeamNGpy.close().

The AI mode is ``span`` with lane keeping and a speed limit that changes
every three minutes. ``--waypoints`` switches to manual targets, one name
at a time, still traffic-free. The level is whatever is loaded. Start on
west_coast_usa. Reloading a map is a separate ``--reload-level`` and
replaces the current session.

Each tick writes one lossless file per camera, exactly the array BeamNG
returned (PNG for uint8, otherwise ``.npy``). Cameras stay in separate
folders. ``state.jsonl`` holds the raw timestamps, ``dt_s``, and the camera
pose and intrinsics. Lane targets go in ``gt.jsonl``. Nothing here resizes,
stitches, or draws on the frames. The trainer does that when it loads.

The scene net does not assume a fixed rate. Ground-truth lane targets are the
road network at 5/10/20/30 m plus heading error, same geometry as the
live log's 8 m sample. Uncompressed uint8 for the yaml rig at 10 Hz is
about 231.7 GB/h. The run stops if the data disk drops under 20 GB free.

The output directory is resumable. A second start appends strips and keeps
counting until ``--minutes`` is reached.

Excursions (weaves, lane changes, recoveries) are on unless
``--no-excursions``. ``--maps`` rotates whole segments. The first minute
is a smoke collect: if those PNGs do not decode, the process stops.
``--check-maps`` loads each level as a scenario and exits without recording.

    py -3 scripts\\collect_scene_lanes.py --check-maps --list-levels
    py -3 scripts\\collect_scene_lanes.py --check-maps --maps west_coast_usa,italy,east_coast_usa,utah,jungle_rock_island
    py -3 scripts\\collect_scene_lanes.py --minutes 120 --maps west_coast_usa,italy,east_coast_usa,utah,jungle_rock_island --segment-min 12 --smoke-s 60

West-coast spawns skip roads whose next 300 m has painted lanes on less than
``--min-paint-cov`` of the probes (default 0.5, 0 disables). ``--weave-guard``
keeps weaves off unpainted stretches and junctions, and ends an excursion when
its 60 m ``ai.setLine`` runs out.

Row fields in ``state.jsonl`` written by this collector, besides the frame data:

* ``settle``: true for the first 3 s of recorded rows after a spawn, a
  teleport, a recover, or a level load. Steering is still saturated there.
* ``recover``: true on the first row after a recover teleport (one row).
* ``stall``, ``stall_s``: the car is under 1 m/s while the AI should move,
  and for how long so far. A stop at a sign is a short stall.
* ``impact``, ``impact_decel_mps2``: the speed fell faster than 15 m/s^2
  since the previous tick. The excursion is aborted; 2 s stopped respawns.
* ``spawn_type``, ``spawn_paint_cov``: the west-coast spawn kind and its
  measured paint coverage ahead (null when not measured).
* ``ego.up``: the State sensor's up vector. ``ego.damage``: Damage sensor total.

``impacts.jsonl`` beside ``state.jsonl`` has one ``{"kind": "impact", "t",
"i", "pre_t0", "pre_i0", ...}`` line per impact and one ``{"kind":
"impact_end", "t", "how"}`` line when it ends (``respawn``, ``drove_on``, or
the abort reason). ``t`` matches the row ``t`` exactly; ``i`` is the row index
the writer was expected to assign (``i_row`` ``impact``). When the impact
tick itself is not written (the same tick aborts on damage, or the writer
queue is full), ``i`` is the last written row before it and ``i_row`` is
``previous``; ``t`` is still the impact time. ``pre_t0`` is the first row
of the 3 s run-up (PRE_IMPACT_S). ``abort_t0`` is the row t where the
current abort phase began when the impact lands in an abort, else null.
The trainer (``--mask-events``) masks from the earliest of those to the
last row before the next respawn (recover / new settle run), past
``impact_end``; rows are masked, never deleted.
``impacts.jsonl`` is not a complete crash list: it only sees a longitudinal
speed drop, so a sideswipe or a scrape that keeps the speed is missed, and a
gap between ticks over 0.5 s (a stalled loop, a teleport) is never scored.

The spawn deck logs ``select_ms`` on every spawn and a ``collect spawn mix``
line with the per-kind counts, cross-kind and best-seen fallbacks each
minute. The same summary is ``spawn_mix`` in both collect.json files.

``--min-live-yield`` (default 0, off) watches the share of moving, non-settle
rows whose painted label is valid 10 m ahead, over the last 60 s of them.
Under the threshold it cuts: on west coast it takes the next deck spawn,
elsewhere it skips to the next scheduled segment. Rows already written stay;
the clock is not rolled back, so no recorded minute is lost and the cut does
not count toward the 20% recover budget. Each cut is a ``collect low_yield``
line and a ``cuts.jsonl`` record with the reason, position, the road the car
is on and the spawn road. After 3 cuts in a row the guard pauses until a full
window reads at or above the threshold, so it cannot respawn forever.
Trade-off: the guard keeps only roads the current painted labeler already
handles (bridges whose edge line is not a decal, tunnels, dirt). It biases
the data toward those roads, and a labeler fix later would have found
labels on the stretches it cut.

The deck also counts how often each spawn road is picked and its share of
recorded seconds: ``road_repeats`` in both collect.json files (carried over
when a run resumes from collect.json) and ``collect spawn repeats`` lines at
the end of the run. A road picked 3 or more times is flagged. On resume the
roads collect.json already lists are skipped, so the reseeded deck does not
start over on them. ``collect spawn types`` lines at the end of the run (and
``spawn_types`` in collect.json) count picks as city/highway/ramp/bridge/
tunnel, deck spawns and recheck respawns apart.

Spawn safety (west coast):

* When every deck road is used, a ``collect deck exhausted ...`` line is
  printed once, ``deck_exhausted`` goes into both collect.json files, and the
  car keeps driving its current spawn: no respawn, no stop. A stall then
  falls back to the old stationary recover.
* ``--traffic-ahead-m`` / ``--traffic-behind-m`` (60 / 40) place the two
  companion cars that far ahead and behind along the road walk, facing the
  direction of travel. A walk that cannot reach the distance uses the
  farthest point it reached; under 8 m the car is not spawned. Each spawn
  logs a ``collect traffic`` line, with any fallback.
* ``--stall-respawn-s`` (20; 0 is the old 10 s stationary abort and recover)
  respawns from the deck after that many seconds under 1 m/s (the
  stationary detector), outside the first 3 s after a spawn. The 10 s
  stationary abort waits for it. Each one is a ``collect stall_respawn`` line
  and a ``cuts.jsonl`` record with ``reason=stall_respawn``. Three stall
  respawns in a row with no recorded row fall back to the stationary abort.

Stops: every run that started recording ends with exactly one ``collect stop
reason=<reason> code=<exit> done_s=.. target_s=..`` line, ``stop_reason`` /
``stop_detail`` / ``stop_code`` in both collect.json files, and a non-zero
exit when it stops before ``target_s`` (the stop file, ``reason=requested``,
keeps exit 0). A segment end never stops the run: a segment that never drove
moves on to the next one, and a used-up schedule wraps the map rotation;
only ``--no-progress-ends`` (5) segment ends in a row with no recorded row
stop it, ``reason=no_progress``, exit 6.

Code pin: the first log line is ``collect code_sha=<sha> code_dirty=<true|false|
unknown> argv=<json list>`` (``git rev-parse HEAD`` and ``git status
--porcelain`` in this checkout; ``unknown`` when git fails). The same
``code_sha`` / ``code_dirty`` / ``argv`` go into both collect.json files and
the heartbeat.

Heartbeat: ``<GVD status>/collect_heartbeat.json`` (E:/GVD_data/status on
the driving PC), rewritten every 30 s and once at the stop: ``pid``, ``t``,
``time``, ``done_s``, ``target_s``, ``segment``, ``last_line``, ``state``
(running or stopped), ``reason``, ``code``, ``out``.

``--recheck-cov-min N`` (default 0, off; west coast with ``--min-paint-cov``
above 0) reruns the spawn paint-coverage probe every N minutes of a spawn,
from the nearest road to the car along its heading. The AI leaves the 300 m
it was picked on within minutes. Under ``--min-paint-cov`` the car takes the
next deck spawn. Rows are never dropped. Every recheck, kept or not, is a
``collect recheck_cov`` line and a ``cuts.jsonl`` record with
``reason=recheck_cov``.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.raw_frames import controls_from_state
from python.collect.spawn_mix import clamp_speed_range, traffic_count

SPEEDS_MPS = (8.0, 11.0, 14.0, 17.0, 10.0)
SPEED_HOLD_S = 180.0
WAYPOINT_HOLD_S = 180.0
DISK_CHECK_S = 10.0
# No picture from any camera for this long is a dead attach, not a slow disk.
ALL_CAM_STALE_S = 10.0


def speed_at(elapsed_s: float, speeds: tuple[float, ...] = SPEEDS_MPS) -> float:
    if not speeds:
        return 10.0
    index = int(max(0.0, float(elapsed_s)) // SPEED_HOLD_S) % len(speeds)
    return float(speeds[index])


def waypoint_at(elapsed_s: float, names: tuple[str, ...]) -> str | None:
    if not names:
        return None
    index = int(max(0.0, float(elapsed_s)) // WAYPOINT_HOLD_S) % len(names)
    return names[index]


def ai_commands(speed_mps: float, waypoint: str | None, *, traffic: bool = False) -> list[str]:
    """Vehicle-Lua lines. Span is traffic-free. Traffic uses the traffic AI."""
    speed = float(speed_mps)
    if traffic and not waypoint:
        return [
            'ai.setMode("traffic")',
            'ai.driveInLane("on")',
            'ai.setSpeedMode("limit")',
            f"ai.setSpeed({speed})",
        ]
    if waypoint:
        safe = str(waypoint).replace("\\", "").replace('"', "")
        return [
            'ai.setMode("manual")',
            'ai.driveInLane("on")',
            'ai.setSpeedMode("limit")',
            f"ai.setSpeed({speed})",
            f'ai.setTarget("{safe}")',
        ]
    return [
        'ai.setMode("span")',
        'ai.driveInLane("on")',
        'ai.setSpeedMode("limit")',
        f"ai.setSpeed({speed})",
    ]


def directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += int(path.stat().st_size)
            except OSError:
                continue
    return total


def format_disk_line(
    elapsed_s: float,
    nbytes: int,
    free_bytes: int | None = None,
    *,
    ceiling_gb_per_hour: float | None = None,
    hz: float | None = None,
) -> str:
    minutes = max(elapsed_s, 1e-6) / 60.0
    mb = nbytes / (1024.0 * 1024.0)
    measured_gb_h = (nbytes / max(elapsed_s, 1e-6)) * 3600.0 / (1024.0 ** 3)
    line = (
        f"collect t={elapsed_s / 60.0:.1f} min disk={mb:.1f} MB "
        f"rate={mb / minutes:.2f} MB/min measured={measured_gb_h:.2f} GB/h"
    )
    if ceiling_gb_per_hour is not None:
        line += f" ceiling={float(ceiling_gb_per_hour):.1f} GB/h"
    if free_bytes is not None:
        line += f" free={free_bytes / (1024.0 ** 3):.1f} GB"
    if hz is not None:
        line += f" hz={float(hz):.2f}"
    return line


def tick_sleep_s(elapsed_tick_s: float, period_s: float = 0.1) -> float:
    """Sleep only the remainder of the plan period. A slow tick does not add another."""
    return max(0.0, float(period_s) - max(0.0, float(elapsed_tick_s)))


def achieved_hz(frames: int, elapsed_s: float) -> float:
    return float(frames) / max(float(elapsed_s), 1e-6)


_SEGMENT_ENV = (("", ""), ("0.35", "clear"), ("0.7", ""), ("0.2", "rain"), ("0.55", "fog"))


def segment_env(index: int) -> tuple[str, str]:
    """Weather and time of day for this segment. The first segment leaves the sky."""
    return _SEGMENT_ENV[int(index) % len(_SEGMENT_ENV)]


def format_queue_line(stats: dict | None) -> str:
    stats = stats or {}
    return (
        f"queue={int(stats.get('queue_depth') or 0)} "
        f"peak={int(stats.get('queue_peak') or 0)} "
        f"dropped={int(stats.get('dropped_frames') or 0)}"
    )


def load_progress(path: Path) -> float:
    if not path.is_file():
        return 0.0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    try:
        return max(0.0, float(raw.get("done_s") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def save_progress(path: Path, done_s: float, target_s: float, extra: dict[str, Any] | None = None) -> None:
    """``done_s``/``target_s`` (what a resume reads), plus ``extra`` keys for the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"done_s": float(done_s), "target_s": float(target_s)}
    payload.update(dict(extra or {}))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record AI driving strips with GT lane targets")
    parser.add_argument("--minutes", type=float, default=45.0, help="Stop after this much driving. 0 with --until-disk runs to the 20 GB floor.")
    parser.add_argument("--until-disk", action="store_true", help="Ignore the clock. Stop only under 20 GB free.")
    parser.add_argument("--map", default="", help="Level to load. Empty keeps the current level.")
    parser.add_argument("--route", default="span", choices=["span", "waypoints", "random"])
    parser.add_argument(
        "--speed-min", type=float, default=None,
        help="Clamp every spawn-type speed range to this floor. Unset: the spawn type's range (8 m/s off the deck).",
    )
    parser.add_argument(
        "--speed-max", type=float, default=None,
        help="Clamp every spawn-type speed range to this ceiling. Unset: the spawn type's range (17 m/s off the deck).",
    )
    parser.add_argument("--cameras", default="", help="Comma list. Empty keeps every camera at its plan rate.")
    parser.add_argument("--rates", default="", help="Overrides, for example main=10,rear=1")
    parser.add_argument(
        "--traffic",
        nargs="?",
        const=1,
        default=None,
        type=int,
        help="Companion AI cars ahead and behind. West coast defaults to 2. 0 disables them.",
    )
    parser.add_argument(
        "--respawn-min",
        type=float,
        default=0.0,
        help="On west_coast_usa, teleport to a new stratified spawn after this many minutes.",
    )
    parser.add_argument("--time-of-day", default="", help="0-1 day fraction, or an hour 0-24. Empty leaves the sky.")
    parser.add_argument("--weather", default="", help="clear, rain, or fog. Empty leaves the weather.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Strip folder. Default: GVD_DATA_DIR/scene_lanes or E:\\GVD_data\\scene_lanes.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Override GVD_DATA_DIR for this process. Refuses C: and D:.",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--waypoints",
        default="",
        help="Comma-separated BeamNG waypoint names. Empty stays in span mode.",
    )
    parser.add_argument(
        "--reload-level",
        default="",
        help="Load this level before driving. Empty keeps the level that is already running.",
    )
    parser.add_argument(
        "--maps",
        default="",
        help="Comma-separated levels to rotate. Empty keeps the current level.",
    )
    parser.add_argument(
        "--segment-min",
        type=float,
        default=12.0,
        help="Minutes per segment. Weighted time is split into chunks of this length.",
    )
    parser.add_argument(
        "--map-weights",
        default="",
        help=(
            "Time shares, for example west_coast_usa=0.6. Unlisted maps split the rest. "
            "Each share is cut into --segment-min chunks and rotated. Empty rotates equally."
        ),
    )
    parser.add_argument("--no-excursions", action="store_true", help="Stay centered. Default weaves and lane changes.")
    parser.add_argument("--smoke-s", type=float, default=60.0, help="Collect this many seconds, check PNG decode, then continue. 0 skips the check.")
    parser.add_argument("--no-vary-env", action="store_true", help="Do not change weather or time of day on each segment.")
    parser.add_argument(
        "--check-maps",
        action="store_true",
        help="Load each map, print spawn, road edges, and cameras, then exit.",
    )
    parser.add_argument(
        "--dump-roads",
        action="store_true",
        help="Load each map and write E:\\GVD_data\\roads\\<map>.json.gz, then exit.",
    )
    parser.add_argument(
        "--list-levels",
        action="store_true",
        help="With --check-maps, print installed level names and which have a spawn table, then exit.",
    )
    parser.add_argument(
        "--min-paint-cov",
        type=float,
        default=0.5,
        help=(
            "West coast: skip a spawn whose next 300 m has a painted lane 10 m ahead on less than "
            "this share of the 10 m probes. 0 disables."
        ),
    )
    parser.add_argument(
        "--weave-guard",
        action="store_true",
        help=(
            "Start no weave with paint on under half the lookaheads or a side road within 40 m, "
            "and end an excursion before its 60 m line runs out."
        ),
    )
    parser.add_argument(
        "--min-live-yield",
        type=float,
        default=0.0,
        help=(
            "Cut the spawn (west coast) or the segment when under this share of the last 60 s of "
            "moving rows has a painted label 10 m ahead. 0 (default) is off. Frames are kept."
        ),
    )
    parser.add_argument(
        "--traffic-ahead-m",
        type=float,
        default=60.0,
        help="Companion car ahead of the ego, in meters along the road. Unreachable: the farthest valid spot.",
    )
    parser.add_argument(
        "--traffic-behind-m",
        type=float,
        default=40.0,
        help="Companion car behind the ego, in meters along the road. Unreachable: the farthest valid spot.",
    )
    parser.add_argument(
        "--no-impact-respawn",
        action="store_true",
        help=(
            "Keep driving after a logged impact (old behaviour). Default: respawn from the deck on the "
            "next tick (recover in place off the west coast deck), so a damaged car is never recorded."
        ),
    )
    parser.add_argument(
        "--stall-respawn-s",
        type=float,
        default=20.0,
        help=(
            "West coast: respawn from the deck after this many seconds stationary (under 1 m/s), "
            "outside the 3 s settle. 0 keeps the 10 s stationary abort and recover."
        ),
    )
    parser.add_argument(
        "--no-progress-ends",
        type=int,
        default=5,
        help="Stop (reason=no_progress, exit 6) after this many segment ends in a row with no recorded row. 0 never stops.",
    )
    parser.add_argument(
        "--recheck-cov-min",
        type=float,
        default=0.0,
        help=(
            "West coast: every N minutes, rerun the spawn paint-coverage probe from the car's road and "
            "heading, and respawn from the deck when it is under --min-paint-cov. 0 (default) is off. "
            "Frames are kept."
        ),
    )
    return parser


def recheck_due(recheck_s: float, done_s: float, mark_s: float) -> bool:
    """True when ``recheck_s`` (> 0) seconds of driving passed since the spawn or the last recheck."""
    return float(recheck_s) > 0.0 and float(done_s) - float(mark_s) >= float(recheck_s) - 1e-9


def stall_respawn_due(stall_respawn_s: float, slow_s: float, since_spawn_s: float, can_respawn: bool) -> bool:
    """True when a stall respawn should fire: on, a deck to draw from, past the settle, stalled long enough."""
    from python.collect.liveness import SETTLE_FLAG_S

    return (
        float(stall_respawn_s) > 0.0
        and bool(can_respawn)
        and float(since_spawn_s) >= SETTLE_FLAG_S
        and float(slow_s) >= float(stall_respawn_s) - 1e-9
    )


def defer_stationary(why: str, stall_respawn_s: float, slow_s: float, can_respawn: bool) -> str:
    """Hold the 10 s ``stationary`` abort while a later stall respawn can still handle it."""
    if why == "stationary" and float(stall_respawn_s) > 0.0 and can_respawn and float(slow_s) < float(stall_respawn_s):
        return ""
    return why


def recheck_plan(cov: float | None, min_cov: float) -> str:
    """``respawn`` when a scored recheck is under ``min_cov``, else ``keep`` (no road found keeps)."""
    if cov is None:
        return "keep"
    return "respawn" if float(cov) < float(min_cov) else "keep"


def recheck_record(
    *,
    t: float,
    i: int | None,
    segment: str,
    map_name: str | None,
    spawn_type: str | None,
    spawn_road: str | None,
    road: str | None,
    pos: list[float] | None,
    cov: float | None,
    min_cov: float,
    recheck_min: float,
    action: str,
    done_s: float,
    road_type: str | None = None,
    new_spawn_road: str | None = None,
    new_road_type: str | None = None,
) -> dict[str, Any]:
    """One ``cuts.jsonl`` line for a coverage recheck. ``new_*`` name the respawn, if any."""
    return {
        "kind": "recheck",
        "reason": "recheck_cov",
        "t": round(float(t), 6),
        "i": i,
        "segment": segment,
        "map": map_name or None,
        "spawn_type": spawn_type,
        "spawn_road": spawn_road,
        "road": road,
        "pos": pos,
        "paint_cov": None if cov is None else round(float(cov), 4),
        "min_paint_cov": float(min_cov),
        "recheck_min": float(recheck_min),
        "action": action,
        "done_s": round(float(done_s), 3),
        "road_type": road_type,
        "new_spawn_road": new_spawn_road,
        "new_road_type": new_road_type,
    }


def paint_share(tech: dict[str, Any] | None) -> float:
    """Share of the painted lookaheads (5/10/20/30 m) with a valid lane center."""
    from python.debug.lane_targets import LOOKAHEADS_M, SLOT_CENTER, lookahead_slot

    valid = list((tech or {}).get("lane_gt_valid") or [])
    idx = [lookahead_slot(k, SLOT_CENTER) for k in range(len(LOOKAHEADS_M))]
    slots = [valid[i] for i in idx if i < len(valid)]
    try:
        return sum(1.0 for value in slots if float(value) > 0.5) / float(len(idx))
    except (TypeError, ValueError):
        return 0.0


def weave_guard_reason(share: float, fork_m: float | None, min_share: float, junction_m: float) -> str:
    """``open``, ``paint_ahead`` (too little paint ahead), or ``junction_ahead``."""
    if float(share) < float(min_share):
        return "paint_ahead"
    if fork_m is not None and float(fork_m) < float(junction_m):
        return "junction_ahead"
    return "open"


def label_valid_at(tech: dict[str, Any] | None, ahead_m: float = 10.0) -> bool:
    """True when the label's center slot at ``ahead_m`` is valid."""
    from python.debug.lane_targets import LOOKAHEADS_M, SLOT_CENTER, lookahead_slot

    try:
        k = [float(v) for v in LOOKAHEADS_M].index(float(ahead_m))
        return float(((tech or {}).get("lane_gt_valid") or [])[lookahead_slot(k, SLOT_CENTER)]) > 0.5
    except (IndexError, TypeError, ValueError):
        return False


def low_yield_plan(can_respawn: bool, boundary: float | None, done_s: float, cut_s: float) -> tuple[str, float]:
    """``(action, schedule_cut)`` for a low-yield cut.

    ``respawn``: west coast, the next deck road. ``segment_end``: skip the
    rest of this scheduled piece; like a hard segment end, ``done_s`` stays,
    so the skipped minutes are recorded on later pieces. ``keep``: nothing to
    cut to (no deck and no map rotation).
    """
    if can_respawn:
        return "respawn", float(cut_s)
    if boundary is None:
        return "keep", float(cut_s)
    return "segment_end", float(cut_s) + max(0.0, float(boundary) - (float(done_s) + float(cut_s)))


def polyline_length(points: list[tuple[float, float]] | None) -> float:
    pts = list(points or [])
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def parse_map_weights(text: str) -> dict[str, float]:
    """``west_coast_usa=0.6,italy=0.08`` into a dict. Empty text is no weighting."""
    weights: dict[str, float] = {}
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"map weight {item!r} needs name=share")
        name, raw = item.split("=", 1)
        name = name.strip()
        share = float(raw)
        if not name or share < 0.0 or not math.isfinite(share):
            raise ValueError(f"map weight {item!r} is not a positive share")
        weights[name] = share
    return weights


def share_maps(maps: tuple[str, ...], weights: dict[str, float]) -> list[tuple[str, float]]:
    """Time fraction for each map, in ``maps`` order, summing to 1.

    A listed weight is that map's share. Maps without a weight split whatever
    is left. Weights that already sum past 1, with nothing left to split, are
    scaled down.
    """
    explicit = {name: float(weights[name]) for name in maps if name in weights}
    missing = [name for name in maps if name not in weights]
    used = sum(explicit.values())
    if missing:
        left = max(0.0, 1.0 - used)
        each = left / len(missing)
        shares = [(name, explicit.get(name, each)) for name in maps]
        total = sum(share for _name, share in shares)
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            shares = [(name, share / total) for name, share in shares]
        return shares
    scale = used if used > 0.0 else 1.0
    return [(name, explicit.get(name, 0.0) / scale) for name in maps]


# A second stall, tip, or damage hit ends the segment. The remainder of
# that segment is reassigned, not lost. The same reason on three maps, or
# real stall and settle seconds past this much of the plan, end the run.
HARD_SEGMENT_REASONS = ("stationary", "tipped", "damage", "impact")
IMPACTS_FILE = "impacts.jsonl"
CUTS_FILE = "cuts.jsonl"
# An excursion line is WINDOW_M (60 m) long. With --weave-guard the move ends
# this far before the line runs out, instead of driving past its end.
LINE_END_MARGIN_M = 5.0
# A weave with --weave-guard needs this share of the painted lookaheads.
WEAVE_GUARD_PAINT = 0.5
# Rows recorded before saying the Damage sensor never reported.
DAMAGE_WARN_ROWS = 300
SEGMENT_FAIL_MAPS = 3
RECOVER_LOST_FRACTION = 0.20


def segment_fail_action(
    reason: str,
    map_name: str,
    failed: dict[str, set[str]],
    lost_s: float,
    skip_s: float,
    target_s: float,
) -> str:
    """``exit`` or ``continue`` after a segment's second hard abort.

    Records ``map_name`` under ``reason``. Three maps with that reason, or
    real stall and settle seconds past 20% of a finite plan, exit.
    ``skip_s`` is the remainder handed to later segments. It is not lost
    time, so one short segment cannot fail a long plan.
    """
    del skip_s
    bucket = failed.setdefault(str(reason), set())
    bucket.add(map_name or "current")
    if len(bucket) >= SEGMENT_FAIL_MAPS:
        return "exit"
    planned = float(target_s)
    if math.isfinite(planned) and planned > 0.0 and float(lost_s) > RECOVER_LOST_FRACTION * planned:
        return "exit"
    return "continue"


def weighted_pieces(
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float],
    total_s: float | None,
) -> list[tuple[str, float]]:
    """Chunks of at most ``segment_s`` that sum to each map's share.

    The heaviest map alternates with the others, so a 60% west-coast share
    is several short visits rather than one block at the front. A map whose
    share is already shorter than ``segment_s`` is a single chunk.
    """
    shares = share_maps(maps, weights)
    if total_s is not None and math.isfinite(float(total_s)) and float(total_s) > 0.0:
        cycle = float(total_s)
    else:
        positive = [share for _name, share in shares if share > 1e-9]
        small = min(positive) if positive else 1.0
        cycle = max(1.0, float(segment_s)) / small
    span = max(1.0, float(segment_s))
    budgets = [(name, max(0.0, share) * cycle) for name, share in shares]
    budgets = [(name, dur) for name, dur in budgets if dur > 0.05]
    if not budgets:
        return [(maps[0], span)]

    def _chunks(name: str, budget: float) -> list[tuple[str, float]]:
        pieces: list[tuple[str, float]] = []
        left = float(budget)
        while left > 0.05 and len(pieces) < 10000:
            take = min(span, left)
            if left - take <= 0.05:
                take = left
            pieces.append((name, take))
            left -= take
        return pieces

    heavy_at = max(range(len(budgets)), key=lambda i: (budgets[i][1], -i))
    heavy = _chunks(budgets[heavy_at][0], budgets[heavy_at][1])
    others: list[tuple[str, float]] = []
    for index, (name, dur) in enumerate(budgets):
        if index == heavy_at:
            continue
        others.extend(_chunks(name, dur))
    pieces: list[tuple[str, float]] = []
    heavy_i = 0
    other_i = 0
    while heavy_i < len(heavy) or other_i < len(others):
        if heavy_i < len(heavy):
            pieces.append(heavy[heavy_i])
            heavy_i += 1
        if other_i < len(others):
            pieces.append(others[other_i])
            other_i += 1
    return pieces


def segment_at(
    elapsed_s: float,
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None = None,
    total_s: float | None = None,
) -> tuple[str, int]:
    """Map name and segment index. Empty maps stay on the loaded level.

    Without weights, each map gets ``segment_s`` and the list repeats.
    With weights, ``weighted_pieces`` rotates those chunks across the plan.
    The index keeps climbing when the cycle repeats.
    """
    if not maps:
        return "", 0
    elapsed = max(0.0, float(elapsed_s))
    if not weights:
        span = max(1.0, float(segment_s))
        index = int(elapsed // span)
        return maps[index % len(maps)], index
    pieces = weighted_pieces(maps, segment_s, weights, total_s)
    cycle_len = sum(dur for _name, dur in pieces) or 1.0
    which = int(elapsed // cycle_len)
    pos = elapsed - which * cycle_len
    acc = 0.0
    for i, (name, dur) in enumerate(pieces):
        acc += dur
        if pos < acc or i == len(pieces) - 1:
            return name, which * len(pieces) + i
    return pieces[-1][0], which * len(pieces) + len(pieces) - 1


def segment_boundary(
    elapsed_s: float,
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None = None,
    total_s: float | None = None,
) -> float | None:
    """Schedule time where the current segment index changes. None with no maps."""
    if not maps:
        return None
    elapsed = max(0.0, float(elapsed_s))
    if not weights:
        span = max(1.0, float(segment_s))
        return (int(elapsed // span) + 1) * span
    pieces = weighted_pieces(maps, segment_s, weights, total_s)
    cycle_len = sum(dur for _name, dur in pieces) or 1.0
    which = int(elapsed // cycle_len)
    pos = elapsed - which * cycle_len
    acc = 0.0
    for _name, dur in pieces:
        acc += dur
        if pos < acc:
            return which * cycle_len + acc
    return (which + 1) * cycle_len


def segment_changed(loaded_map: str, loaded_index: int, map_name: str, seg_index: int) -> bool:
    """True when this piece needs a fresh level load.

    Two west-coast chunks are different segments even though the level name
    matches. An empty name is the single-level collect and does not reload.
    """
    return bool(map_name) and (map_name != loaded_map or seg_index != loaded_index)


def continue_after_segment_end(
    done_s: float,
    cut_s: float,
    lost_s: float,
    reason: str,
    map_name: str,
    failed: dict[str, set[str]],
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None,
    total_s: float,
) -> tuple[str, float, float]:
    """``(action, new_cut_s, skipped_s)`` after a segment's second hard abort.

    ``action`` is ``exit``, ``stop`` (nothing left to rotate to), or
    ``continue``. The unspent remainder is added to the schedule cut so the
    next lookup starts the following piece. ``done_s`` stays put: those
    minutes are still to be collected, on the later segments.
    """
    action = segment_fail_action(reason, map_name, failed, lost_s, 0.0, total_s)
    if action == "exit":
        return "exit", cut_s, 0.0
    planned = None
    if math.isfinite(float(total_s)) and float(total_s) > 0.0:
        planned = float(total_s)
    boundary = segment_boundary(done_s + cut_s, maps, segment_s, weights, planned)
    if boundary is None:
        return "stop", cut_s, 0.0
    skipped = max(0.0, float(boundary) - (float(done_s) + float(cut_s)))
    return "continue", float(cut_s) + skipped, skipped


def decode_issues(root: Path) -> list[str]:
    """Problems that mean the smoke capture cannot be trained on."""
    from python.data.raw_frames import list_raw_images, png_signature_ok

    if not (Path(root) / "state.jsonl").is_file():
        return ["smoke wrote no state.jsonl"]
    images = [path for path in list_raw_images(root) if path.suffix.lower() == ".png"]
    if not images:
        return ["smoke wrote no png frames"]
    bad = [path.name for path in images if not png_signature_ok(path)]
    if bad:
        return [f"unreadable {name}" for name in bad[:8]]
    return []


def start_freeroam(bng: Any, level: str) -> bool:
    """Ask the running session to load ``level``. Does not close BeamNGpy.

    Opening the level does not spawn a car. ``enter_map`` waits, then builds
    a scenario if ``get_player_vehicle_id`` reports that there is no vehicle.
    """
    from python.collect.level_session import queue_level

    return queue_level(bng, level) == ""


def drop_cameras(cams: dict[str, Any] | None) -> None:
    """Detach the previous level's sensors. This does not quit the game."""
    for cam in list((cams or {}).values()):
        fn = getattr(cam, "remove", None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception:
            continue


def road_for(roads: list[Any], hit: object) -> Any:
    if hit is None:
        return None
    want = str(getattr(hit, "road_id", "") or "")
    for road in roads:
        if str(getattr(road, "road_id", "")) == want:
            return road
    return None


def painted_for(level: str) -> Any:
    """Painted line decals for ``level`` (cached dump, else the level zip under BNG_HOME)."""
    from python.data.data_root import roads_dir
    from python.debug.painted_lanes import PaintedLanes, load_painted_decals

    try:
        lanes = PaintedLanes(load_painted_decals(str(level or ""), roads_dir()))
    except Exception as exc:
        print(f"collect painted lanes {level}: {type(exc).__name__}: {exc}", flush=True)
        lanes = PaintedLanes([])
    print(f"collect painted lanes {level} segments={len(lanes.segs)}", flush=True)
    return lanes


def road_lanes(roads: list[Any], hit: object) -> int | None:
    road = road_for(roads, hit)
    if road is None:
        return None
    from python.debug.gt_geom import lane_count

    try:
        return int(lane_count(float(road.width_m)))
    except (TypeError, ValueError):
        return None


def _queue(vehicle: Any, command: str) -> None:
    fn = getattr(vehicle, "queue_lua_command", None)
    if callable(fn):
        try:
            fn(command)
            return
        except TypeError:
            fn(command, False)
            return
    raise RuntimeError("vehicle has no queue_lua_command")


def _player_vehicle(bng: Any) -> Any:
    """Player vehicle, or ``RuntimeError`` carrying the BeamNGpy text.

    ``BNGError: There is no vehicle`` used to escape because only
    ``RuntimeError`` was caught. The text is preserved either way.
    """
    from python.collect.level_session import player_vehicle, session_text, is_session_error

    try:
        veh, note = player_vehicle(bng)
    except Exception as exc:
        if is_session_error(exc):
            raise RuntimeError(session_text(exc)) from exc
        raise
    if veh is None:
        raise RuntimeError(note or "no vehicle in the running session")
    return veh


def enter_map(
    bng: Any,
    level: str,
    cfg: dict[str, Any],
    cams: dict[str, Any] | None,
) -> Any:
    """Drop the old cameras, start a scenario, then attach every camera.

    The returned object's ``error`` is empty on success and is safe to store
    in ``status/collect.json``. Roads are read only after the scenario starts.
    """
    from python.collect.level_session import OpenedLevel, begin_level, is_session_error, session_text

    drop_cameras(cams)
    try:
        opened = begin_level(bng, level)
    except Exception as exc:
        if is_session_error(exc):
            return OpenedLevel(error=session_text(exc))
        raise
    if opened.error or opened.vehicle is None:
        opened.error = opened.error or "no vehicle after load"
        return opened
    if not opened.roads:
        opened.error = opened.error or "no road network"
        return opened
    try:
        opened.cameras = _attach_cameras(bng, opened.vehicle, cfg)
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, RuntimeError):
            opened.error = session_text(exc)
            return opened
        raise
    print(f"collect cameras {len(opened.cameras)} ({', '.join(sorted(opened.cameras))})", flush=True)
    # vehicle.state is the State sensor (pos/dir/vel). Steering and pedals live
    # in Electrics, which this collector never attached, so wheel.steer was null.
    try:
        from beamngpy.sensors import Electrics  # type: ignore

        opened.vehicle.sensors.attach("electrics", Electrics())
    except Exception as exc:  # already attached, or an old BeamNGpy
        print(f"collect electrics attach: {type(exc).__name__}: {exc}", flush=True)
    # Damage is its own sensor, not a State or Electrics field. Without it
    # ego.damage was null on every row and the damage abort never fired.
    try:
        from beamngpy.sensors import Damage  # type: ignore

        opened.vehicle.sensors.attach("damage", Damage())
    except Exception as exc:  # already attached, or an old BeamNGpy
        print(f"collect damage attach: {type(exc).__name__}: {exc}", flush=True)
    return opened


def format_check_line(level: str, opened: Any) -> str:
    """One pre-flight line: spawn, edge count, camera count."""
    spawn = getattr(opened, "spawn", None)
    pos = "na" if not spawn else f"({float(spawn[0]):.1f},{float(spawn[1]):.1f},{float(spawn[2]):.1f})"
    source = str(getattr(opened, "spawn_source", "") or "na")
    edges = len(getattr(opened, "roads", None) or [])
    cameras = len(getattr(opened, "cameras", None) or {})
    error = str(getattr(opened, "error", "") or "")
    if not error and (edges < 1 or cameras < 1):
        error = "missing edges or cameras"
    line = f"collect check {level} spawn={pos} source={source} edges={edges} cameras={cameras}"
    if error:
        line += f" error={error}"
    return line


def run_map_checks(
    bng: Any,
    levels: tuple[str, ...] | list[str],
    cfg: dict[str, Any],
    skipped: list[str] | None = None,
) -> int:
    """Load each level, report spawn/edges/cameras, and exit. No recording."""
    from python.data.data_root import logs_dir, use_data_disk

    failed = 0
    cams: dict[str, Any] = {}
    lines = [f"collect skip level {name} (not in this Tech install)" for name in (skipped or [])]
    for level in levels:
        opened = enter_map(bng, level, cfg, cams)
        cams = dict(getattr(opened, "cameras", None) or {})
        line = format_check_line(level, opened)
        print(line, flush=True)
        lines.append(line)
        if opened.error or not opened.roads or not opened.cameras:
            failed += 1
    drop_cameras(cams)
    if use_data_disk():
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        path = logs_dir() / f"check_maps_{stamp}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"collect check log {path}", flush=True)
    return 2 if failed else 0


def run_road_dumps(
    bng: Any,
    levels: tuple[str, ...] | list[str],
    cfg: dict[str, Any],
    skipped: list[str] | None = None,
) -> int:
    """Load each level and write ``roads/<map>.json.gz``. No recording."""
    from python.collect.road_dump import write_road_dump
    from python.data.data_root import roads_dir, use_data_disk
    from python.debug.gt_logger import raw_road_network

    if not use_data_disk():
        print("collect roads need GVD_DATA_DIR (E:\\GVD_data on the driving PC)", flush=True)
        return 2
    failed = 0
    cams: dict[str, Any] = {}
    for name in skipped or []:
        print(f"collect skip level {name} (not in this Tech install)", flush=True)
    for level in levels:
        opened = enter_map(bng, level, cfg, cams)
        cams = dict(getattr(opened, "cameras", None) or {})
        spec, roads, note = raw_road_network(bng)
        if opened.error:
            note = opened.error
        path = write_road_dump(
            roads_dir() / f"{level}.json.gz",
            level=level,
            spawn=getattr(opened, "spawn", None),
            spawn_source=str(getattr(opened, "spawn_source", "") or ""),
            road_network=spec,
            roads=roads,
            note=note,
        )
        edges = len(getattr(opened, "roads", None) or [])
        print(f"collect roads {level} edges={edges} file={path}", flush=True)
        if opened.error or edges < 1 or spec is None:
            failed += 1
    drop_cameras(cams)
    return 2 if failed else 0


def list_installed_levels(bng: Any) -> int:
    """Print every level Tech reports, and whether this repo has a spawn pose for it."""
    from python.collect.level_session import KNOWN_SPAWNS, installed_levels

    names, note = installed_levels(bng)
    if not names:
        print(f"collect level list unavailable ({note})", flush=True)
        return 2
    print(f"collect levels {len(names)}", flush=True)
    for name in sorted(names):
        pose = "known" if name in KNOWN_SPAWNS else "no-spawn-table"
        print(f"collect level {name} {pose}", flush=True)
    return 0


def record_session_failure(status_root: Any, message: str) -> None:
    """Write the session error where ``gvd_status`` already looks."""
    from python.debug.remote_status import write_collect_status

    text = str(message or "session error")
    _write_collect_crash(status_root, text, {})
    write_collect_status(
        status_root,
        {
            "kind": "collect",
            "error": text,
            "driver": "beamng_ai",
            "automatic": True,
            "updated": time.time(),
        },
    )


def _colour(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("colour", "color", "image"):
            if payload.get(key) is not None:
                return payload[key]
    return payload


def camera_outage(
    since: float | None,
    now: float,
    got_any: bool,
    reasons: dict[str, str],
) -> tuple[float | None, str | None]:
    """Start or finish the all-cameras-dead timer.

    ``got_any`` clears it. Otherwise the clock runs from the first empty
    tick, and after ``ALL_CAM_STALE_S`` the message names each camera.
    """
    if got_any:
        return None, None
    start = float(now) if since is None else float(since)
    if float(now) - start <= ALL_CAM_STALE_S:
        return start, None
    detail = ", ".join(f"{cid}={reasons.get(cid, 'missing')}" for cid in sorted(reasons))
    return start, f"every camera stale for >{ALL_CAM_STALE_S:.0f}s ({detail})"


def attached_frames_ready(cams: dict[str, Any], frames: dict[str, Any]) -> bool:
    """True when every attached rig camera returned a picture.

    The first tick often has all-zero buffers. Those are already dropped by
    the reader, so a camera that has not rendered yet is simply absent.
    """
    from python.sensors.cameras import CAM_IDS

    expected = [str(cid) for cid in cams if str(cid) in CAM_IDS]
    if not expected:
        return False
    return all(cid in frames for cid in expected)


STOP_EXIT_EARLY = 6
ETK800 = "etk800"
from python.collect.excursion import ETK800_HALF_WIDTH_M  # noqa: E402
EGO_WIDTH_RANGE_M = (1.2, 3.0)


def bbox_width_m(box: Any) -> float | None:
    """Vehicle width from ``Vehicle.get_bbox()`` corners: left-right distance, front or rear (larger)."""
    if not isinstance(box, dict):
        return None
    best: float | None = None
    for end in ("front", "rear"):
        a, b = box.get(f"{end}_bottom_left"), box.get(f"{end}_bottom_right")
        try:
            d = math.dist([float(v) for v in list(a)[:3]], [float(v) for v in list(b)[:3]])
        except (TypeError, ValueError):
            continue
        if math.isfinite(d):
            best = d if best is None else max(best, d)
    return best


def vehicle_model(vehicle: Any) -> str | None:
    m = getattr(vehicle, "model", None)
    if m:
        return str(m)
    opts = getattr(vehicle, "options", None)
    if isinstance(opts, dict) and opts.get("model"):
        return str(opts["model"])
    return None


def vehicle_half_width(vehicle: Any) -> tuple[float | None, str]:
    """Ego half width for the weave lane-edge rule, read once per spawned vehicle.

    ``get_bbox()`` first (BeamNG's world bbox, which includes the mirrors).
    When that fails or is out of EGO_WIDTH_RANGE_M, an etk800 uses its known
    1.8 m and says so; any other model returns None and the run stops with
    ``vehicle_width``. There is no silent default.
    """
    fn = getattr(vehicle, "get_bbox", None)
    if callable(fn):
        try:
            width = bbox_width_m(fn())
        except Exception as exc:
            note = f"bbox failed: {type(exc).__name__}: {exc}"
        else:
            lo, hi = EGO_WIDTH_RANGE_M
            if width is not None and lo <= width <= hi:
                return width * 0.5, f"source=bbox width={width:.2f}m half={width * 0.5:.2f}m"
            note = f"bbox width {'na' if width is None else format(width, '.2f') + 'm'} outside {lo:g}-{hi:g} m"
    else:
        note = "no get_bbox"
    model = vehicle_model(vehicle)
    if model == ETK800:
        half = ETK800_HALF_WIDTH_M
        return half, f"source=etk800 width={2 * half:.2f}m half={half:.2f}m ({note})"
    return None, f"{note}; model={model or 'unknown'} is not {ETK800}, no known width"


def code_pin(repo: Path | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    """``code_sha``, ``code_dirty`` (bool, or "unknown") and ``argv`` for this run."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    sha: str = "unknown"
    dirty: Any = "unknown"
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15
        )
        if head.returncode == 0 and head.stdout.strip():
            sha = head.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=30
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    full = [sys.executable] + (list(sys.argv) if argv is None else [str(Path(__file__).resolve())] + [str(a) for a in argv])
    return {"code_sha": sha, "code_dirty": dirty, "argv": full}


def pin_line(pin: dict[str, Any]) -> str:
    dirty = pin.get("code_dirty")
    dirty_txt = "unknown" if not isinstance(dirty, bool) else ("true" if dirty else "false")
    return f"collect code_sha={pin.get('code_sha')} code_dirty={dirty_txt} argv={json.dumps(pin.get('argv') or [])}"
STALL_RESPAWN_STREAK = 3
HEARTBEAT_FILE = "collect_heartbeat.json"
HEARTBEAT_S = 30.0


class _LastLine:
    """stdout pass-through that keeps the last non-empty printed line."""

    def __init__(self, inner: Any, run: "RunStop") -> None:
        self._inner = inner
        self._run = run
        self._part = ""

    def write(self, text: str) -> int:
        self._part += str(text)
        if "\n" in self._part:
            lines = [x for x in self._part.split("\n")[:-1] if x.strip()]
            if lines:
                self._run.last_line = lines[-1][:500]
            self._part = self._part.rsplit("\n", 1)[1]
        return self._inner.write(text)

    def flush(self) -> None:
        self._inner.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RunStop:
    """Why a recording run stopped, and the heartbeat a poller can read."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.detail = ""
        self.out: Path | None = None
        self.progress_path: Path | None = None
        self.target_s: float | None = None
        self.status_root: Path | None = None
        self.tool = False
        self.last_line = ""
        self.segment: str | None = None
        self.done_s = 0.0
        self.pin: dict[str, Any] = {}
        self._beat_m: float | None = None

    def attach(self, out: Path, progress_path: Path, target_s: float, done_s: float) -> None:
        self.out, self.progress_path, self.target_s, self.done_s = Path(out), Path(progress_path), float(target_s), float(done_s)
        try:
            from python.debug.remote_status import status_dir

            self.status_root = status_dir()
        except Exception:
            self.status_root = None

    def stop(self, reason: str, detail: str = "") -> None:
        """The first reason sticks."""
        if self.reason is None:
            self.reason, self.detail = str(reason), str(detail or "")

    def heartbeat_path(self) -> Path | None:
        return None if self.status_root is None else Path(self.status_root) / HEARTBEAT_FILE

    def beat(self, done_s: float, *, force: bool = False, state: str = "running", code: int | None = None) -> None:
        self.done_s = float(done_s)
        now = time.monotonic()
        if not force and self._beat_m is not None and now - self._beat_m < HEARTBEAT_S:
            return
        self._beat_m = now
        path = self.heartbeat_path()
        if path is None:
            return
        target = self.target_s
        payload = {
            "kind": "collect_heartbeat",
            "pid": os.getpid(),
            "t": time.time(),
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "done_s": round(float(done_s), 3),
            "target_s": None if target is None or target == float("inf") else float(target),
            "segment": self.segment,
            "last_line": self.last_line,
            "state": state,
            "reason": self.reason if state == "stopped" else None,
            "detail": self.detail if state == "stopped" else None,
            "code": code,
            "out": None if self.out is None else str(self.out),
            **self.pin,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            print(f"collect heartbeat write failed: {exc}", flush=True)

    @staticmethod
    def _merge(path: Path, extra: dict[str, Any]) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.update(extra)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"collect stop_reason write failed for {path}: {exc}", flush=True)

    def finish(self, code: int | None) -> int | None:
        """Exactly one ``collect stop reason=`` line; stop_reason in collect.json; non-zero if early."""
        if self.tool or self.progress_path is None or self.target_s is None:
            return code
        done = max(float(self.done_s), load_progress(self.progress_path))
        early = done < float(self.target_s) - 1e-6
        reason = self.reason or ("target_reached" if not early else "unknown")
        if early and code == 0 and reason != "requested":
            code = STOP_EXIT_EARLY
        target_txt = "inf" if self.target_s == float("inf") else f"{self.target_s:.1f}"
        tail = f" {self.detail}" if self.detail else ""
        print(f"collect stop reason={reason} code={code} done_s={done:.1f} target_s={target_txt}{tail}", flush=True)
        extra = {"stop_reason": reason, "stop_detail": self.detail or None, "stop_code": code, "stopped": time.time(), **self.pin}
        if not self.progress_path.is_file():
            extra = {"done_s": done, "target_s": float(self.target_s), **extra}
        self._merge(self.progress_path, extra)
        if self.status_root is not None and (Path(self.status_root) / "collect.json").is_file():
            self._merge(Path(self.status_root) / "collect.json", extra)
        self.reason = reason
        self.beat(done, force=True, state="stopped", code=code)
        return code


def main(argv: list[str] | None = None) -> int:
    """Run the collector; every recording stop is logged once, recorded, and early stops exit non-zero."""
    run = RunStop()
    old = sys.stdout
    sys.stdout = _LastLine(old, run)
    try:
        run.pin = code_pin(argv=argv)
        print(pin_line(run.pin), flush=True)
        try:
            code = _main(argv, run)
        except BaseException as exc:
            run.stop("exception", f"{type(exc).__name__}: {exc}")
            run.finish(None)
            raise
        return run.finish(code)  # type: ignore[return-value]
    finally:
        sys.stdout = old


def _main(argv: list[str] | None, run: RunStop) -> int:
    args = build_parser().parse_args(argv)
    from python.data.data_root import (
        DataDriveError,
        check_data_path,
        ensure_cache_dirs,
        scene_lanes_dir,
    )
    from python.data.raw_frames import (
        CAMERA_HZ,
        PNG_RATIO_MAX,
        PNG_RATIO_MEAN,
        RawFrameWriter,
        camera_due,
        camera_sidecar,
        capture_plan,
        frame_nbytes,
        plan_totals,
        yaw_rate_from_dirs,
    )
    from python.data.raw_queue import RawWriteQueue
    from python.debug.gt_geom import RoadFollow, follow_label, heading_from_dir
    from python.debug.gt_logger import load_road_polys
    from python.debug.lane_targets import tech_from_hit, tech_from_painted
    from python.debug.painted_lanes import PaintedLanes, load_painted_decals
    from python.sensors.cameras import load_camera_config
    from python.sensors.tech import (
        gvd_to_bng_vehicle,
        load_tech_config,
        open_tech_beamngpy,
        release_tech_beamngpy,
        widen_socket_timeout,
    )

    cfg = load_tech_config()
    host = args.host or str(cfg.get("host") or "localhost")
    port = int(args.port if args.port is not None else cfg.get("port") or 25252)
    if args.data_dir:
        os.environ["GVD_DATA_DIR"] = str(args.data_dir)
    try:
        ensure_cache_dirs()
    except DataDriveError as exc:
        print(f"collect {exc}", file=sys.stderr)
        return 2
    from python.collect.auto_mode import environment_commands, rates_table, speeds_for, stop_file

    names = tuple(part.strip() for part in str(args.waypoints).split(",") if part.strip())
    if args.route == "span":
        names = ()
    elif args.route == "random" and names:
        import random

        bag = list(names)
        random.Random(1).shuffle(bag)
        names = tuple(bag)
    map_names = tuple(part.strip() for part in str(args.maps).split(",") if part.strip())
    try:
        map_weights = parse_map_weights(args.map_weights)
    except ValueError as exc:
        print(f"collect {exc}", file=sys.stderr)
        return 2
    if (args.check_maps or args.dump_roads) and not map_names:
        from python.collect.auto_mode import DEFAULT_MAPS

        map_names = tuple(DEFAULT_MAPS)
    segment_s = max(60.0, float(args.segment_min) * 60.0)
    excursions_on = not bool(args.no_excursions)
    vary_env = bool(map_names) and not bool(args.no_vary_env)
    # Off the spawn deck (other maps) the range is 8-17, clamped to the flags like a spawn type.
    flags_set = args.speed_min is not None or args.speed_max is not None
    base_range, _src, _empty = clamp_speed_range((8.0, 17.0), args.speed_min, args.speed_max)
    speeds = speeds_for(base_range[0], base_range[1])
    cam_rates = rates_table(args.rates, args.cameras)
    until_disk = bool(args.until_disk) or float(args.minutes) <= 0.0
    if args.map:
        args.reload_level = args.map
    out = Path(args.out) if args.out is not None else scene_lanes_dir()
    try:
        check_data_path(out)
    except DataDriveError as exc:
        print(f"collect {exc}", file=sys.stderr)
        return 2
    progress_path = out / "collect.json"
    done_s = load_progress(progress_path)
    target_s = float("inf") if until_disk else max(60.0, float(args.minutes) * 60.0)
    flag = stop_file(out)
    if flag.is_file():
        flag.unlink()
    run.attach(out, progress_path, target_s, done_s)
    if done_s >= target_s:
        print(f"collect already reached {done_s / 60.0:.1f} min in {out}")
        run.stop("target_reached", "already reached before this run")
        return 0
    cam_cfg = load_camera_config()
    plan = capture_plan(cam_cfg, PNG_RATIO_MAX)
    totals = plan_totals(plan)
    ceiling_gb = float(totals["png_gb_per_hour"])
    print(
        f"collect resume={done_s / 60.0:.1f} min target={target_s / 60.0:.1f} min out={out} "
        f"mode={'waypoints ' + ','.join(names) if names else 'span traffic-free'}",
        flush=True,
    )
    print(
        f"collect png ratio measured mean={PNG_RATIO_MEAN:.3f} max={PNG_RATIO_MAX:.3f} "
        f"on 85 real frames (tests/fixtures/gt_run*.jpg)",
        flush=True,
    )
    for row in plan:
        print(
            f"collect cam {row['id']} {row['width']}x{row['height']}x{row['channels']} "
            f"@ {row['hz']:.0f} Hz raw={row['uncompressed_bytes']} B "
            f"png={row['png_gb_per_hour']:.2f} GB/h",
            flush=True,
        )
    print(
        f"collect plan png={totals['png_gb_per_hour']:.1f} GB/h "
        f"uncompressed={totals['uncompressed_gb_per_hour']:.1f} GB/h "
        f"stop under 20 GB free",
        flush=True,
    )
    from python.collect.level_session import installed_levels, is_session_error, select_maps, session_text
    from python.debug.remote_status import status_dir as _status_dir

    status_root = _status_dir()
    try:
        bng = open_tech_beamngpy(
            host, port, home=cfg.get("home") or None, user=cfg.get("user") or None, launch=False,
        )
    except Exception as exc:
        if not is_session_error(exc):
            raise
        record_session_failure(status_root, session_text(exc))
        print(f"collect stop {session_text(exc)}", flush=True)
        run.stop("session_open_failed", session_text(exc))
        return 2
    widen_socket_timeout(bng, 120.0)
    writer = None
    try:
        skipped_levels: list[str] = []
        if map_names:
            installed, list_note = installed_levels(bng)
            if installed is None:
                print(f"collect level list unavailable ({list_note}); trying the requested maps", flush=True)
            else:
                kept, skipped_levels = select_maps(map_names, installed)
                for name in skipped_levels:
                    print(f"collect skip level {name} (not in this Tech install)", flush=True)
                map_names = tuple(kept)
                if not map_names and not args.list_levels:
                    message = "none of the requested maps are in this Tech install"
                    record_session_failure(status_root, message)
                    print(f"collect stop {message}", flush=True)
                    run.stop("no_maps", message)
                    return 2
        elif args.reload_level:
            print(f"collect reload level={args.reload_level}", flush=True)
        if args.list_levels or args.dump_roads or args.check_maps:
            run.tool = True
        if args.list_levels:
            return list_installed_levels(bng)
        if args.dump_roads:
            code = run_road_dumps(bng, map_names, cam_cfg, skipped_levels)
            if code != 0:
                record_session_failure(status_root, "dump-roads failed")
            return code
        if args.check_maps:
            code = run_map_checks(bng, map_names, cam_cfg, skipped_levels)
            if code != 0:
                record_session_failure(status_root, "check-maps failed")
            return code
        # A map rotation enters the level on the first segment, after the
        # writer exists. A single-level run keeps the scenario already started.
        vehicle = None
        roads: list[Any] = []
        painted = PaintedLanes([])
        cams: dict[str, Any] = {}
        spawn_pos: tuple[float, float, float] | None = None
        if not map_names:
            opened = enter_map(bng, str(args.reload_level or ""), cam_cfg, {})
            vehicle, roads, cams, err = opened.vehicle, opened.roads, opened.cameras, opened.error
            painted = painted_for(str(args.reload_level or ""))
            if err or vehicle is None or not roads:
                message = err or "no vehicle after load"
                record_session_failure(status_root, message)
                print(f"collect stop {message}", flush=True)
                run.stop("level_load_failed", message)
                return 2
            if opened.spawn is not None:
                spawn_pos = (float(opened.spawn[0]), float(opened.spawn[1]), float(opened.spawn[2]))
        writer = RawWriteQueue(RawFrameWriter(out, camera_sidecar(cam_cfg)))
        from python.data.strip_writer import STRIP_ORDER
        from python.debug.remote_status import CollectMeter, status_dir, write_collect_status

        meter = CollectMeter(STRIP_ORDER)
        status_root = status_dir()
        seed_bytes = directory_bytes(out)
        last_print = time.monotonic()
        last_disk = last_print
        from python.collect.excursion import (
            ACTIVE,
            Excursion,
            ExcursionClock,
            CRUISE_S,
            WEAVE_STATS_NOTES,
            WeaveStats,
            excursion_commands,
            excursion_path,
            gate_reason,
            near_junction,
            overspeed,
            plan_speed,
            ramp_heading_reached,
            targets_for,
        )

        clock = ExcursionClock()
        from python.collect.liveness import (
            ImpactWatch,
            Liveness,
            SETTLE_S,
            SettleWindow,
            TIP_DIR_Z_LIVE,
            YieldGuard,
            car_is_driving,
        )
        from python.collect.excursion import JUNCTION_AHEAD_M, fork_ahead_m

        live = Liveness()
        impact = ImpactWatch()
        settle = SettleWindow()
        yield_guard = YieldGuard(float(getattr(args, "min_live_yield", 0.0) or 0.0))
        last_row_i: int | None = None
        line_left_m: float | None = None
        spawn_cov: float | None = None
        spawn_road: str | None = None
        damage_seen = False
        damage_warned = False
        row_base = getattr(getattr(writer, "writer", None), "_n", None)
        weave_paint_min = float(args.min_paint_cov) if float(args.min_paint_cov) > 0.0 else WEAVE_GUARD_PAINT
        segment_aborts = 0
        recover_lost_s = 0.0
        schedule_cut = 0.0
        last_end_done = -1.0
        stuck_ends = 0
        no_progress_max = max(0, int(getattr(args, "no_progress_ends", 0) or 0))
        segment_fail_maps: dict[str, set[str]] = {}
        cams_ready = False
        segment_open = False
        follow = RoadFollow()
        label_hit = None
        mark_recover = False
        gate_note = ""
        gate_log_t = 0.0
        warmup_since: float | None = None
        last_ai: tuple[Any, ...] | None = None
        held_tag: Excursion | None = None
        # Wall time t of the first tick of the current abort phase. An impact
        # inside an abort names it (abort_t0) so the whole abort is masked.
        abort_t0: float | None = None
        # Ego half width from the bbox (or etk800), read at each level load.
        ego_half_w: float | None = None
        ego_width: dict[str, Any] = {}
        width_for: Any = None
        # Seconds the painted lane at the car has been missing during a weave.
        paint_lost_s = 0.0
        PAINT_LOST_S = 0.5
        impact_respawn_on = not bool(getattr(args, "no_impact_respawn", False))
        impact_respawn_due = False
        impact_respawns: dict[str, int] = {"impacts": 0, "deck": 0, "recover": 0, "abort_same_tick": 0}
        loaded_map = ""
        loaded_index = -1
        speed_span = [speeds]
        # Effective speed range per segment (collect.json speed_ranges) and the
        # weave / lane change / recover counters (collect.json weave_stats).
        speed_ranges: dict[str, list[dict[str, Any]]] = {}
        weave_stats = WeaveStats()
        # Planning speed and phase of the move in progress (None between moves).
        plan_v: float | None = None
        plan_phase = ""
        plan_overspeed_noted = False
        # The lane change in progress: road covered since its line was sent,
        # against the planned ramp (None between lane changes).
        lane_change_run: dict[str, float] | None = None
        spawn_deck = None
        spawn_kind = ""
        spawn_mark = -1.0e9
        respawn_s = max(0.0, float(args.respawn_min)) * 60.0
        recheck_min = max(0.0, float(getattr(args, "recheck_cov_min", 0.0) or 0.0))
        recheck_s = recheck_min * 60.0
        recheck_mark = 0.0
        stall_respawn_s = max(0.0, float(getattr(args, "stall_respawn_s", 0.0) or 0.0))
        # Loop seconds since the last spawn, recover or level load (the
        # driving clock stops while the car stands, so settle cannot be used).
        since_spawn = [0.0]
        deck_exhausted: dict[str, Any] = {}
        # Stall respawns in a row with no recorded row, and done_s at the last one.
        stall_streak: list[float] = [0, -1.0]
        tick_dt = 0.1
        last_cam: dict[str, float] = {}
        prev_dir: list[float] | None = None
        prev_dir_t: float | None = None
        last_error = ""
        cam_reasons: dict[str, str] = {}
        if vehicle is not None:
            for command in environment_commands(args.time_of_day, args.weather):
                try:
                    _queue(vehicle, command)
                except Exception as exc:
                    last_error = session_text(exc) if is_session_error(exc) else f"{type(exc).__name__}: {exc}"

        def _used_bytes() -> int:
            extra = 0
            try:
                extra = int(writer.stats().get("bytes_written") or 0)
            except Exception:
                extra = 0
            return int(seed_bytes) + extra

        def _publish(free: int | None, speed_now: float, route: str | None) -> None:
            payload = meter.snapshot(
                elapsed_s=done_s,
                disk_bytes=_used_bytes(),
                free_bytes=free,
                ai_speed_mps=speed_now,
                ai_mode=("manual" if route else "span"),
                ai_waypoint=route,
                out=str(out),
                target_s=target_s,
            )
            payload["raw_png_gb_per_hour"] = float(ceiling_gb)
            payload["raw_uncompressed_gb_per_hour"] = float(totals["uncompressed_gb_per_hour"])
            payload.update(writer.stats())
            payload["driver"] = "beamng_ai"
            payload["automatic"] = True
            payload["error"] = last_error
            payload["cam_drop_reason"] = dict(cam_reasons)
            payload["until_disk"] = bool(until_disk)
            payload.update(run.pin)
            if spawn_deck is not None:
                payload["spawn_mix"] = spawn_deck.summary()
                payload["road_repeats"] = spawn_deck.road_repeats()
                payload["spawn_types"] = spawn_deck.type_summary()
            if deck_exhausted:
                payload["deck_exhausted"] = dict(deck_exhausted)
            if yield_guard.enabled:
                payload["live_yield"] = yield_guard.summary()
            if target_s == float("inf"):
                payload["target_s"] = None
            write_collect_status(status_root, payload)

        def _hz() -> float:
            try:
                written = int(writer.stats().get("written") or 0)
            except Exception:
                written = 0
            return achieved_hz(written if written > 0 else meter.frames, done_s)

        _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
        if status_root is not None:
            print(f"collect status={status_root / 'collect.json'}", flush=True)
        all_stale_since: float | None = None
        last_reason_line = ""
        smoke_ok = done_s > 0.0 or float(args.smoke_s) <= 0.0
        session_faults = 0

        def _abort(reason: str) -> int | None:
            """Respawn once. A second hard abort ends the segment and continues."""
            nonlocal segment_aborts, last_error, clock, last_ai, held_tag, prev_dir, prev_dir_t, spawn_pos
            nonlocal segment_open, follow, label_hit, mark_recover, done_s, recover_lost_s
            nonlocal schedule_cut, last_end_done, stuck_ends, loaded_index
            segment_aborts += 1
            print(f"collect abort {reason} segment={segment_id}", flush=True)
            last_error = f"{segment_id} {reason}"
            stall_s = max(0.0, float(live.slow_s)) if reason in ("stationary", "impact") else 0.0
            if segment_aborts >= 2 and reason in HARD_SEGMENT_REASONS:
                recover_lost_s += stall_s
                action, schedule_cut, _skipped = continue_after_segment_end(
                    done_s,
                    schedule_cut,
                    recover_lost_s,
                    reason,
                    map_name or segment_id,
                    segment_fail_maps,
                    map_names,
                    segment_s,
                    map_weights or None,
                    target_s,
                )
                print(f"collect segment end {segment_id} {reason}", flush=True)
                if action == "exit":
                    n_maps = len(segment_fail_maps.get(reason, ()))
                    if n_maps >= SEGMENT_FAIL_MAPS:
                        message = f"{reason} on {n_maps} maps"
                    else:
                        planned = target_s if target_s < float("inf") else 1.0
                        share = recover_lost_s / planned
                        message = f"recover lost {share:.0%} of the plan"
                    last_error = f"{segment_id} {message}"
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    _write_collect_crash(status_root, last_error, {})
                    print(f"collect stop {last_error}", flush=True)
                    run.stop("segment_fail", message)
                    return 2
                # A segment end never ends the run. rec3 (6a07d65) stopped here
                # with exit 0: segment #11 never drove, so done_s still equalled
                # #10's end and the old guard returned "ended".
                no_progress = last_end_done >= 0.0 and done_s <= last_end_done + 1e-3
                stuck_ends = stuck_ends + 1 if no_progress else 0
                if no_progress:
                    print(
                        f"collect segment end {segment_id} {reason}: no recorded row since the last segment end "
                        f"(done_s={done_s:.1f}, {stuck_ends} in a row); moving to the next segment",
                        flush=True,
                    )
                    if no_progress_max > 0 and stuck_ends >= no_progress_max:
                        message = f"{stuck_ends} segment ends in a row with no recorded row (last {segment_id} {reason})"
                        last_error = message
                        _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                        _publish(None, speed, waypoint)
                        _write_collect_crash(status_root, message, {})
                        run.stop("no_progress", message)
                        return STOP_EXIT_EARLY
                if action == "stop":
                    print(
                        f"collect schedule used up at done_s={done_s:.1f} < target; wrapping the map rotation "
                        f"(cut {schedule_cut:.0f}s -> 0)",
                        flush=True,
                    )
                    schedule_cut = 0.0
                    loaded_index = -1  # reload: a fresh level and spawn, not the stuck pose
                last_end_done = done_s
                segment_aborts = 0
                segment_open = False
                return None
            from python.collect.level_session import place_vehicle, recover_points, snap_to_road

            here = None
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                here = (float(pos[0]), float(pos[1]), float(pos[2]))
            placed = None
            for near in recover_points(here, spawn_pos):
                placed = snap_to_road(bng, near)
                if placed is not None:
                    break
            if placed is None:
                message = f"{segment_id} aborted: {reason}; no road pose to respawn"
            else:
                moved, mrot = placed
                shift = place_vehicle(bng, vehicle, moved, mrot)
                if shift:
                    message = f"{segment_id} aborted: {reason}; respawn failed: {shift}"
                else:
                    print(
                        f"collect recover {segment_id} ({moved[0]:.1f},{moved[1]:.1f},{moved[2]:.1f})",
                        flush=True,
                    )
                    time.sleep(SETTLE_S)
                    recover_lost_s += stall_s + float(SETTLE_S)
                    live.reset()
                    clock = ExcursionClock()
                    last_ai = None
                    held_tag = None
                    prev_dir = None
                    prev_dir_t = None
                    follow = RoadFollow()
                    label_hit = None
                    segment_open = False
                    spawn_pos = moved
                    mark_recover = True
                    impact.reset()
                    yield_guard.reset()
                    settle.start(done_s)
                    since_spawn[0] = 0.0
                    return None
            last_error = message
            _save_progress(done_s, done_s if target_s == float("inf") else target_s)
            _publish(None, speed, waypoint)
            _write_collect_crash(status_root, message, {})
            print(f"collect stop {message}", flush=True)
            run.stop("recover_failed", message)
            return 2

        def _speed_range(kind: str, kind_range: tuple[float, float], default_src: str) -> tuple[float, ...]:
            """Clamp a range to the flags, log it, record it for this segment, and return its speed cycle."""
            (lo, hi), src, empty = clamp_speed_range(kind_range, args.speed_min, args.speed_max)
            if src == "spawn_type":
                src = default_src
            if empty:
                print(f"collect speed_clamp kind={kind} range={kind_range[0]:.0f}-{kind_range[1]:.0f} "
                      f"flags={args.speed_min}-{args.speed_max} -> {lo:g}-{hi:g} (no overlap: flag range)", flush=True)
            print(f"collect speed_range seg={segment_id} kind={kind} range={lo:g}-{hi:g} src={src}", flush=True)
            entry = {"kind": kind, "range": [lo, hi], "src": src, "done_s": round(float(done_s), 3)}
            rows = speed_ranges.setdefault(segment_id, [])
            if not rows or {k: v for k, v in rows[-1].items() if k != "done_s"} != {
                k: v for k, v in entry.items() if k != "done_s"
            }:
                rows.append(entry)
            return speeds_for(lo, hi)

        def _apply_spawn(force: bool, source: str = "spawn") -> None:
            """Teleport west coast to the next unused stratified spawn."""
            nonlocal spawn_deck, spawn_kind, spawn_mark, spawn_pos, last_ai, segment_open, speed, spawn_cov
            nonlocal spawn_road, recheck_mark
            if map_name != "west_coast_usa" or vehicle is None:
                return
            if not force and (respawn_s <= 0.0 or done_s - spawn_mark < respawn_s):
                return
            if spawn_deck is None:
                from python.collect.spawn_mix import SpawnDeck

                dump = None
                try:
                    from python.data.data_root import roads_dir

                    candidate = roads_dir() / "west_coast_usa.json.gz"
                    if candidate.is_file():
                        dump = candidate
                except Exception:
                    dump = None
                if dump is None:
                    bundled = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
                    if bundled.is_file():
                        dump = bundled
                if dump is None:
                    return
                load_t0 = time.perf_counter()
                cov_min = max(0.0, float(args.min_paint_cov))
                spawn_deck = SpawnDeck.load(
                    dump,
                    painted=painted if cov_min > 0.0 and painted else None,
                    min_paint_cov=cov_min,
                    log=lambda text: print(f"collect {text}", flush=True),
                )
                print(
                    f"collect spawn deck paint filter "
                    f"{'min_cov=' + format(cov_min, '.2f') if spawn_deck.filtering() else 'off'} "
                    f"load_ms={(time.perf_counter() - load_t0) * 1000.0:.0f}",
                    flush=True,
                )
                try:
                    prior = json.loads(progress_path.read_text(encoding="utf-8")).get("road_repeats")
                except Exception:
                    prior = None
                seeded = spawn_deck.seed_repeats(prior)
                if seeded:
                    skipped_n = spawn_deck.skip_recorded(prior)
                    print(
                        f"collect spawn repeats resumed roads={seeded} from {progress_path.name}; "
                        f"skipping {skipped_n} already-picked roads",
                        flush=True,
                    )
                if recheck_s > 0.0:
                    print(
                        f"collect recheck_cov every {recheck_min:g} min "
                        f"{'on' if spawn_deck.filtering() else 'off (needs --min-paint-cov > 0 and painted lanes)'}",
                        flush=True,
                    )
            if spawn_deck.exhausted:
                return
            picked = spawn_deck.take(source)
            if picked is None:
                mix = spawn_deck.summary()
                deck_exhausted.update(
                    {
                        "exhausted": True,
                        "segment": segment_id,
                        "source": source,
                        "done_s": round(float(done_s), 3),
                        "picks": mix["picks"],
                        "used": mix["used"],
                        "roads": mix["roads"],
                        "spawn_road": spawn_road,
                    }
                )
                print(
                    f"collect deck exhausted {segment_id} source={source} picks={mix['picks']} "
                    f"used={mix['used']}/{mix['roads']} roads; keeping the current spawn "
                    f"road={spawn_road or 'na'} (no respawn)",
                    flush=True,
                )
                _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                return
            from python.collect.level_session import place_vehicle
            from python.collect.spawn_mix import traffic_lua

            shift = place_vehicle(bng, vehicle, picked.pos, picked.rot)
            if shift:
                print(f"collect spawn {segment_id} {picked.kind} failed {shift}", flush=True)
                return
            spawn_kind = picked.kind
            spawn_cov = picked.paint_cov
            spawn_road = str(picked.road_id)
            spawn_mark = done_s
            recheck_mark = done_s
            spawn_pos = picked.pos
            impact.reset()
            yield_guard.reset()
            settle.start(done_s)
            since_spawn[0] = 0.0
            speed_span[0] = _speed_range(picked.kind, (float(picked.speed[0]), float(picked.speed[1])), "spawn_type")
            speed = speed_at(done_s, tuple(speed_span[0]))
            last_ai = None
            live.reset()
            segment_open = False
            count = traffic_count(args.traffic, map_name)
            ge = getattr(bng, "queue_lua_command", None)
            if count and callable(ge):
                slots, notes = spawn_deck.traffic_slots(
                    picked, count, float(args.traffic_ahead_m), float(args.traffic_behind_m)
                )
                for note in notes:
                    print(f"collect {note}", flush=True)
                placed = " ".join(f"{where}={along:.1f}m" for where, _p, _r, along in slots) or "none"
                print(
                    f"collect traffic {segment_id} placed {placed} "
                    f"(want ahead={float(args.traffic_ahead_m):.0f}m behind={float(args.traffic_behind_m):.0f}m along the road)",
                    flush=True,
                )
                for command in traffic_lua(picked, len(slots), slots):
                    ge(command)
            print(
                f"collect spawn {segment_id} {picked.kind} "
                f"({picked.pos[0]:.1f},{picked.pos[1]:.1f},{picked.pos[2]:.1f}) "
                f"speed={picked.speed[0]:.0f}-{picked.speed[1]:.0f} traffic={count} "
                f"paint_cov={'na' if picked.paint_cov is None else format(picked.paint_cov, '.2f')} "
                f"select_ms={spawn_deck.last_ms:.0f}",
                flush=True,
            )

        def _can_stall_respawn() -> bool:
            return (
                stall_respawn_s > 0.0
                and not (stall_streak[0] >= STALL_RESPAWN_STREAK and done_s <= stall_streak[1] + 1e-3)
                and map_name == "west_coast_usa"
                and vehicle is not None
                and spawn_deck is not None
                and not spawn_deck.exhausted
            )

        def _stall_respawn(t_now: float) -> bool:
            """Respawn a stalled car from the deck. False when no spawn was taken."""
            stalled = float(live.slow_s)
            kind_was, road_was = spawn_kind or None, spawn_road
            here = None if pos is None else [round(float(v), 2) for v in list(pos)[:3]]
            stall_streak[0] = stall_streak[0] + 1 if done_s <= stall_streak[1] + 1e-3 else 1
            stall_streak[1] = float(done_s)
            picks_before = spawn_deck.summary()["picks"]
            _apply_spawn(True, "stall")
            took = spawn_deck.summary()["picks"] > picks_before
            where = "na" if here is None else f"({here[0]:.1f},{here[1]:.1f},{here[2]:.1f})"
            print(
                f"collect stall_respawn {segment_id} {kind_was or '-'} stalled={stalled:.1f}s "
                f">= {stall_respawn_s:.0f}s pos={where} spawn_road={road_was or 'na'} -> "
                f"{('respawn road=' + str(spawn_road) + ' type=' + str(spawn_deck.last_type)) if took else 'keep (deck exhausted)'}",
                flush=True,
            )
            append_jsonl(
                out / CUTS_FILE,
                {
                    "kind": "stall",
                    "reason": "stall_respawn",
                    "t": round(float(t_now), 6),
                    "i": last_row_i,
                    "segment": segment_id,
                    "map": map_name or None,
                    "spawn_type": kind_was,
                    "spawn_road": road_was,
                    "pos": here,
                    "stalled_s": round(stalled, 2),
                    "stall_respawn_s": stall_respawn_s,
                    "action": "respawn" if took else "keep",
                    "new_spawn_road": spawn_road if took else None,
                    "new_road_type": spawn_deck.last_type if took else None,
                    "done_s": round(float(done_s), 3),
                },
            )
            return took

        def _impact_respawn(t_now: float) -> bool:
            """Respawn from the deck right after a logged impact. False when no spawn was taken."""
            here = None if pos is None else [round(float(v), 2) for v in list(pos)[:3]]
            kind_was, road_was = spawn_kind or None, spawn_road
            took = False
            if map_name == "west_coast_usa" and vehicle is not None and spawn_deck is not None and not spawn_deck.exhausted:
                picks_before = spawn_deck.summary()["picks"]
                _apply_spawn(True, "impact")
                took = spawn_deck.summary()["picks"] > picks_before
            impact_respawns["deck" if took else "recover"] += 1
            where = "na" if here is None else f"({here[0]:.1f},{here[1]:.1f},{here[2]:.1f})"
            print(
                f"collect impact_respawn {segment_id} n={impact_respawns['deck'] + impact_respawns['recover']} "
                f"pos={where} spawn_road={road_was or 'na'} -> "
                f"{('respawn road=' + str(spawn_road) + ' type=' + str(spawn_deck.last_type)) if took else 'recover in place'}",
                flush=True,
            )
            append_jsonl(
                out / CUTS_FILE,
                {
                    "kind": "impact",
                    "reason": "impact_respawn",
                    "t": round(float(t_now), 6),
                    "i": last_row_i,
                    "segment": segment_id,
                    "map": map_name or None,
                    "spawn_type": kind_was,
                    "spawn_road": road_was,
                    "pos": here,
                    "action": "respawn" if took else "recover",
                    "new_spawn_road": spawn_road if took else None,
                    "done_s": round(float(done_s), 3),
                },
            )
            _save_progress(done_s, done_s if target_s == float("inf") else target_s)
            return took

        def _spawn_extra() -> dict[str, Any] | None:
            extra: dict[str, Any] = dict(run.pin)
            if spawn_deck is not None:
                extra["spawn_mix"] = spawn_deck.summary()
                extra["road_repeats"] = spawn_deck.road_repeats()
                extra["spawn_types"] = spawn_deck.type_summary()
            if deck_exhausted:
                extra["deck_exhausted"] = dict(deck_exhausted)
            if yield_guard.enabled:
                extra["live_yield"] = yield_guard.summary()
            extra["impact_respawn"] = {"on": impact_respawn_on, **impact_respawns}
            extra["speed_ranges"] = {k: list(v) for k, v in speed_ranges.items()}
            extra["weave_stats"] = weave_stats.summary()
            extra["notes"] = {"weave_stats": dict(WEAVE_STATS_NOTES)}
            if ego_width:
                extra["ego_width"] = dict(ego_width)
            return extra or None

        def _save_progress(done: float, target: float) -> None:
            save_progress(progress_path, done, target, _spawn_extra())

        def _say_repeats() -> None:
            if spawn_deck is not None:
                for line in spawn_deck.repeat_lines() + spawn_deck.type_lines():
                    print(f"collect {line}", flush=True)
                if deck_exhausted:
                    print(
                        f"collect deck exhausted at {deck_exhausted['done_s']:.0f}s ({deck_exhausted['segment']}, "
                        f"{deck_exhausted['used']}/{deck_exhausted['roads']} roads used)",
                        flush=True,
                    )

        def _impact_end(how: str, t_end: float) -> None:
            """Close an open impact in the sidecar so the trainer knows where the mask ends."""
            if not impact.open:
                return
            append_jsonl(
                out / IMPACTS_FILE,
                {
                    "kind": "impact_end",
                    "t": round(float(t_end), 6),
                    "how": str(how),
                    "segment": segment_id,
                    "since_impact_s": round(float(impact.since_impact or 0.0), 3),
                },
            )
            impact.close()

        while done_s < target_s:
            run.beat(done_s)
            if stop_file(out).is_file():
                _save_progress(done_s, target_s if target_s < float("inf") else done_s)
                _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
                _say_repeats()
                print("collect stop requested", flush=True)
                run.stop("requested", f"stop file {stop_file(out)}")
                return 0
            elapsed = done_s
            speed = speed_at(elapsed, tuple(speed_span[0]))
            waypoint = waypoint_at(elapsed, names)
            map_name, seg_index = segment_at(
                elapsed + schedule_cut,
                map_names,
                segment_s,
                map_weights or None,
                None if target_s == float("inf") else target_s,
            )
            segment_id = f"{map_name}#{seg_index}" if map_name else "current#0"
            run.segment = segment_id
            if not map_names and segment_id not in speed_ranges:
                # A single-level run has no segment load: record its range once here.
                speed_span[0] = _speed_range("level", (8.0, 17.0), "flag" if flags_set else "default")
                speed = speed_at(elapsed, tuple(speed_span[0]))
            if map_name == loaded_map and seg_index == loaded_index:
                _apply_spawn(False)
            if segment_changed(loaded_map, loaded_index, map_name, seg_index):
                print(f"collect segment {segment_id}", flush=True)
                opened = enter_map(bng, map_name, cam_cfg, cams)
                vehicle, roads, cams, err = opened.vehicle, opened.roads, opened.cameras, opened.error
                painted = painted_for(map_name)
                if err or vehicle is None or not roads or not cams:
                    last_error = err or "no vehicle after load"
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    _write_collect_crash(status_root, last_error, {})
                    print(f"collect stop {last_error}", flush=True)
                    run.stop("level_load_failed", last_error)
                    return 2
                loaded_map = map_name
                loaded_index = seg_index
                last_ai = None
                held_tag = None
                clock = ExcursionClock()
                live.reset()
                segment_aborts = 0
                cams_ready = False
                segment_open = False
                follow = RoadFollow()
                label_hit = None
                gate_note = ""
                warmup_since = None
                if opened.spawn is not None:
                    spawn_pos = (float(opened.spawn[0]), float(opened.spawn[1]), float(opened.spawn[2]))
                prev_dir = None
                prev_dir_t = None
                last_cam = {}
                impact.reset()
                yield_guard.reset()
                settle.start(done_s)
                since_spawn[0] = 0.0
                line_left_m = None
                if vary_env:
                    tod, sky = segment_env(seg_index)
                    sky_commands = environment_commands(tod, sky)
                else:
                    sky_commands = environment_commands(args.time_of_day, args.weather)
                # A new level starts on the off-deck range; a deck spawn below replaces it.
                speed_span[0] = _speed_range("level", (8.0, 17.0), "flag" if flags_set else "default")
                speed = speed_at(done_s, tuple(speed_span[0]))
                _apply_spawn(True)
                for command in sky_commands:
                    try:
                        _queue(vehicle, command)
                    except Exception as exc:
                        last_error = session_text(exc) if is_session_error(exc) else f"{type(exc).__name__}: {exc}"
                        _write_collect_crash(status_root, last_error, {})
            if excursions_on and vehicle is not None and width_for is not vehicle:
                # Once per spawned vehicle (session open or level load).
                width_for = vehicle
                ego_half_w, width_note = vehicle_half_width(vehicle)
                ego_width = {"half_m": None if ego_half_w is None else round(float(ego_half_w), 3),
                             "note": width_note, "segment": segment_id}
                print(f"collect vehicle width {segment_id} {width_note}", flush=True)
                if ego_half_w is None:
                    last_error = f"vehicle width unknown: {width_note}"
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    _write_collect_crash(status_root, last_error, {})
                    print(f"collect stop {last_error}", flush=True)
                    run.stop("vehicle_width", width_note)
                    return 2
            tick_mark = time.monotonic()
            try:
                frames, stamps, cam_reasons = _poll_cameras(cams)
            except Exception as exc:
                if not is_session_error(exc):
                    raise
                last_error = session_text(exc)
                _publish(None, speed, waypoint)
                _write_collect_crash(status_root, last_error, {})
                print(f"collect session {last_error}", flush=True)
                session_faults += 1
                if session_faults >= 3:
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    run.stop("session_faults", f"3 camera polls in a row failed: {last_error}")
                    return 2
                time.sleep(0.5)
                continue
            session_faults = 0
            reason_line = ", ".join(f"{cid}={cam_reasons[cid]}" for cid in sorted(cam_reasons))
            if reason_line and reason_line != last_reason_line:
                print(f"collect cam drop {reason_line}", flush=True)
                last_reason_line = reason_line
            all_stale_since, outage = camera_outage(
                all_stale_since, time.monotonic(), bool(frames), cam_reasons,
            )
            if outage:
                last_error = outage
                _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                _publish(None, speed, waypoint)
                _write_collect_crash(status_root, outage, cam_reasons)
                print(f"collect stop {outage}", flush=True)
                run.stop("camera_outage", outage)
                return 4
            sample = _read_vehicle(vehicle)
            pos = sample.get("pos")
            direction = sample.get("dir")
            now = time.time()
            if sample.get("yaw_rate") is None:
                sample["yaw_rate"] = yaw_rate_from_dirs(
                    prev_dir, direction, 0.0 if prev_dir_t is None else now - prev_dir_t,
                )
            if direction is not None:
                prev_dir = direction
                prev_dir_t = now
            dir_z = None
            if isinstance(direction, (list, tuple)) and len(direction) >= 3:
                try:
                    dir_z = float(direction[2])
                except (TypeError, ValueError):
                    dir_z = None
            up_vec = sample.get("up")
            up_z = None
            if isinstance(up_vec, (list, tuple)) and len(up_vec) >= 3:
                try:
                    up_z = float(up_vec[2])
                except (TypeError, ValueError):
                    up_z = None
            try:
                speed_mps_f = None if sample.get("speed_mps") is None else float(sample.get("speed_mps"))
            except (TypeError, ValueError):
                speed_mps_f = None
            # Every shaped move is planned at max(measured, target).
            v_plan, v_meas, v_src = plan_speed(speed_mps_f, speed)
            v_plan = speed if v_plan is None else v_plan
            try:
                damage_f = None if sample.get("damage") is None else float(sample.get("damage"))
            except (TypeError, ValueError):
                damage_f = None
            if not cams_ready:
                if attached_frames_ready(cams, frames):
                    cams_ready = True
                    warmup_since = None
                    print(f"collect cameras live {sorted(cams)}", flush=True)
                else:
                    if warmup_since is None:
                        warmup_since = time.monotonic()
                    elif time.monotonic() - warmup_since > ALL_CAM_STALE_S:
                        waiting = ", ".join(cid for cid in cams if cid not in frames) or "none"
                        message = f"cameras did not all render within {ALL_CAM_STALE_S:.0f}s ({waiting})"
                        last_error = message
                        _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                        _publish(None, speed, waypoint)
                        _write_collect_crash(status_root, message, cam_reasons)
                        print(f"collect stop {message}", flush=True)
                        run.stop("camera_warmup", message)
                        return 4
                    time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                    tick_dt = max(1e-3, time.monotonic() - tick_mark)
                    continue
            mono = time.monotonic()
            due: dict[str, Any] = {}
            due_stamps: dict[str, Any] = {}
            held: list[tuple[str, float | None]] = []
            for cid, image in frames.items():
                if not camera_due(cid, mono, last_cam, cam_rates):
                    continue
                held.append((cid, last_cam.get(cid)))
                last_cam[cid] = mono
                due[cid] = image
                if cid in stamps:
                    due_stamps[cid] = stamps[cid]
            if meter.frames == 0 and due:
                print(
                    f"collect first tick cams={sorted(due)} bytes={frame_nbytes(due)}",
                    flush=True,
                )
            heading = None
            if direction is not None:
                heading = heading_from_dir(direction[0], direction[1])
            hit = None
            if pos is not None and roads:
                hit, label_hit, _refused = follow_label(
                    roads,
                    float(pos[0]),
                    float(pos[1]),
                    heading,
                    follow,
                    speed_mps_f or 0.0,
                    label_hit,
                )
            driving = car_is_driving(speed_mps_f, dir_z, up_z, TIP_DIR_Z_LIVE)
            if not segment_open:
                if driving:
                    segment_open = True
                    live.reset()
                    clock = ExcursionClock()
                    # A new segment or a respawn: the next refusal is a new blocked stretch.
                    weave_stats.end_stretch()
                    lane_change_run = None
                    last_ai = None
                    held_tag = None
                    print(f"collect driving {segment_id}", flush=True)
                else:
                    if last_ai is None:
                        for command in ai_commands(speed, waypoint, traffic=False):
                            try:
                                _queue(vehicle, command)
                            except Exception as exc:
                                last_error = f"{type(exc).__name__}: {exc}"
                        last_ai = (round(speed, 3), waypoint, "cruise", 1, map_name)
                        print(
                            f"collect ai speed={speed:.0f} m/s route={waypoint or map_name or 'span'} "
                            f"excursion=cruise",
                            flush=True,
                        )
                    since_spawn[0] += tick_dt
                    why = live.update(
                        tick_dt,
                        speed_mps_f,
                        dir_z,
                        damage_f,
                        ai_should_move=True,
                        up_z=up_z,
                        tip_z=TIP_DIR_Z_LIVE,
                    )
                    why = defer_stationary(why, stall_respawn_s, live.slow_s, _can_stall_respawn())
                    if why in ("", "stationary") and stall_respawn_due(
                        stall_respawn_s, live.slow_s, since_spawn[0], _can_stall_respawn()
                    ):
                        if not _stall_respawn(now):
                            why = "stationary"
                        else:
                            time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                            tick_dt = max(1e-3, time.monotonic() - tick_mark)
                            continue
                    if why:
                        code = _abort(why)
                        if code is not None:
                            return code
                        time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                        tick_dt = max(1e-3, time.monotonic() - tick_mark)
                        continue
                    time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                    tick_dt = max(1e-3, time.monotonic() - tick_mark)
                    continue
            # A crash shows as a speed drop between two ticks. Stop the move now
            # and respawn if the car stays stopped, instead of the 10 s stall.
            impact_decel = impact.update(tick_dt, speed_mps_f)
            impact_event: dict[str, Any] | None = None
            if impact_decel is not None:
                pre_t, pre_i = impact.pre_window()
                impact_event = {
                    "kind": "impact",
                    "t": round(float(now), 6),
                    "i": None,
                    "pre_t0": None if pre_t is None else round(float(pre_t), 6),
                    "pre_i0": pre_i,
                    "decel_mps2": round(float(impact_decel), 2),
                    "speed_before_mps": impact.speed_before,
                    "speed_after_mps": speed_mps_f,
                    "segment": segment_id,
                    "excursion": clock.phase,
                    "abort_t0": (
                        round(float(abort_t0), 6) if clock.phase == "abort" and abort_t0 is not None else None
                    ),
                    "pos": pos,
                }
                before_txt = "na" if impact.speed_before is None else format(impact.speed_before, ".1f")
                after_txt = "na" if speed_mps_f is None else format(speed_mps_f, ".1f")
                print(
                    f"collect impact {segment_id} decel={impact_decel:.1f} m/s2 "
                    f"v={before_txt}->{after_txt} excursion={clock.phase}",
                    flush=True,
                )
                if clock.phase in ACTIVE:
                    clock.force_abort("impact")
                line_left_m = None
                impact_respawns["impacts"] += 1
            elif impact.drove_off():
                _impact_end("drove_on", now)
            if clock.phase not in ACTIVE:
                plan_v, plan_phase, plan_overspeed_noted = None, "", False
            elif plan_v is not None and clock.phase == plan_phase and overspeed(speed_mps_f, plan_v):
                # Faster than the line was shaped for. A weave or lane change goes
                # back to the lane now. A recover is already the way back: aborting
                # it hands the return to the unshaped lane keeper, so it is logged.
                if clock.phase in ("weave", "lane_change"):
                    print(f"collect excursion {segment_id} {clock.phase} abort overspeed "
                          f"v_meas={speed_mps_f:.1f} v_plan={plan_v:.1f}", flush=True)
                    weave_stats.abort(clock.phase, "overspeed", plan_v)
                    clock.force_abort("overspeed")
                    line_left_m = None
                    plan_v, plan_phase = None, ""
                elif not plan_overspeed_noted:
                    print(f"collect excursion {segment_id} recover overspeed v_meas={speed_mps_f:.1f} "
                          f"v_plan={plan_v:.1f} (logged; the recover continues)", flush=True)
                    weave_stats.overspeed_logged += 1
                    plan_overspeed_noted = True
            if lane_change_run is not None and clock.phase == "lane_change":
                lane_change_run["covered_m"] += max(0.0, float(speed_mps_f or 0.0)) * tick_dt
            if line_left_m is not None:
                if clock.phase not in ACTIVE:
                    line_left_m = None
                else:
                    line_left_m -= max(0.0, float(speed_mps_f or 0.0)) * tick_dt
                    if line_left_m <= LINE_END_MARGIN_M:
                        print(f"collect excursion {segment_id} {clock.phase} line end, back to lane", flush=True)
                        clock.force_abort("line_end")
                        line_left_m = None
            # Label from the painted lines the camera sees. The nav-spline hit
            # still drives the excursion controller below, but it is not a label:
            # road_invisible AI splines split into equal 1/1 lanes are not the paint.
            if painted:
                tech = tech_from_painted(
                    painted.lane_at(pos, direction) if pos is not None and direction is not None else []
                )
            else:
                tech = tech_from_hit(None, None, None, None)
            lanes_here = road_lanes(roads, hit)
            # The painted lane at the car (y = 0), vehicle frame: the weave's
            # lane edges and width. Same DecalRoad source as the labels.
            paint_lane_now: tuple[float, float] | None = None
            if painted and pos is not None and direction is not None:
                got = painted.lane_at(pos, direction, ys=(0.0,))
                paint_lane_now = got[0] if got else None
            if clock.phase == "weave" and paint_lane_now is None:
                paint_lost_s += tick_dt
            else:
                paint_lost_s = 0.0
            paint_w_now = None if paint_lane_now is None else float(paint_lane_now[1]) - float(paint_lane_now[0])
            # A missing label is an ambiguous tick, not a junction. Aborting the
            # weave on it turned every masked sample into a failed excursion.
            junction = False if hit is None else near_junction(
                hit,
                roads,
                None if pos is None else float(pos[0]),
                None if pos is None else float(pos[1]),
                heading,
            )
            window_ok = True
            start_kind: str | None = None
            if (
                excursions_on
                and not waypoint
                and driving
                and clock.phase == "cruise"
                and clock.phase_t + tick_dt >= CRUISE_S
                and pos is not None
            ):
                kind = clock.peek_kind(lanes_here)
                gate = gate_reason(
                    roads, hit, float(pos[0]), float(pos[1]), heading, kind, v_plan,
                    weave_lane_w=paint_w_now, half_width_m=ego_half_w,
                )
                if gate != "open" and kind != "weave":
                    # A lane change needs a second lane and a clear stretch.
                    # An in-lane weave does not.
                    weave = gate_reason(
                        roads, hit, float(pos[0]), float(pos[1]), heading, "weave", v_plan,
                        weave_lane_w=paint_w_now, half_width_m=ego_half_w,
                    )
                    kind = "weave"
                    gate = weave
                if kind == "weave" and gate == "open" and paint_lane_now is None:
                    gate = "no_paint"  # no painted lane at the car: no lane edge to weave inside
                if args.weave_guard and gate == "open":
                    gate = weave_guard_reason(
                        paint_share(tech),
                        fork_ahead_m(roads, hit, float(pos[0]), float(pos[1]), heading),
                        weave_paint_min,
                        JUNCTION_AHEAD_M,
                    )
                start_kind = kind
                window_ok = gate == "open"
                if not window_ok and not clock.reason:
                    weave_stats.refuse(gate, v_plan)  # one attempt; a blocked stretch counts once
                now_gate = time.monotonic()
                if gate != gate_note or now_gate - gate_log_t >= 5.0:
                    word = "open" if gate == "open" else f"blocked {gate}"
                    print(f"collect gate {segment_id} {kind} {word} v_plan={v_plan:.1f}", flush=True)
                    gate_note = gate
                    gate_log_t = now_gate
            if excursions_on and not waypoint and driving:
                paved = road_for(roads, hit) if hit is not None else None
                move = clock.update(
                    tick_dt,
                    offset_m=None if hit is None else hit.lane_offset_m,
                    heading_rad=None if hit is None else hit.heading_err_rad,
                    lane_width_m=None if hit is None else hit.lane_width_m,
                    off_road=bool(hit.off_road) if hit is not None else False,
                    near_junction=junction,
                    traffic=False,
                    n_lanes=lanes_here,
                    window_ok=window_ok,
                    start_kind=start_kind,
                    road_offset_m=None if hit is None else hit.offset_m,
                    road_width_m=None if paved is None else paved.width_m,
                    speed_mps=v_plan,
                    # Weave abort (lane_edge): a body edge (vehicle-frame x =
                    # +/- ego_half_w around the state pos ref node) more than
                    # EDGE_M past the painted lane line at the car
                    # (PaintedLanes.lane_at, y = 0: level DecalRoad paint). Not
                    # the navgraph lane split, which is the whole AI corridor.
                    half_width_m=ego_half_w,
                    paint_lane=paint_lane_now,
                    paint_lost=paint_lost_s >= PAINT_LOST_S,
                )
            else:
                move = Excursion("cruise", 0.0, 0.0, 1, "")
            if lane_change_run is not None and clock.phase != "lane_change":
                if clock.phase == "recover":
                    # LANE_CHANGE_S is up. A slow car has not reached the end of the ramp.
                    covered, ramp = lane_change_run["covered_m"], lane_change_run["ramp_m"]
                    complete = covered + 1e-6 >= ramp
                    if not complete:
                        meas_txt = "na" if speed_mps_f is None else format(speed_mps_f, ".1f")
                        print(f"collect excursion {segment_id} lane_change_incomplete covered={covered:.0f}m "
                              f"ramp={ramp:.0f}m v_meas={meas_txt} v_plan={lane_change_run['v_plan']:.1f}", flush=True)
                    weave_stats.lane_change_end(complete)
                lane_change_run = None
            resolved = held_tag if held_tag is not None and held_tag.phase == move.phase else move
            path = None
            ai_key = (round(speed, 3), waypoint, resolved.phase, resolved.direction, map_name)
            if (
                driving
                and last_ai != ai_key
                and excursions_on
                and not waypoint
                and move.phase in ACTIVE
                and pos is not None
            ):
                path_info: dict[str, Any] = {}
                path, used = excursion_path(
                    roads,
                    float(pos[0]),
                    float(pos[1]),
                    heading,
                    move,
                    hit,
                    v_plan,
                    paint_lane=paint_lane_now,
                    half_width_m=ego_half_w,
                    info=path_info,
                )
                road = road_for(roads, hit)
                spec = None if path is None else targets_for(
                    move.phase,
                    used,
                    None if hit is None else hit.lane_width_m,
                    lane_id=0 if hit is None else int(getattr(hit, "lane_id", 0) or 0),
                    road_width_m=None if road is None else float(road.width_m),
                    speed_mps=v_plan,
                    weave_lane_w=paint_w_now,
                    half_width_m=ego_half_w,
                )
                if spec is not None and move.phase == "weave" and path_info.get("amp_m") is not None:
                    amp_sent = float(path_info["amp_m"])
                    spec = (int(used) * amp_sent, -int(used) * math.atan(amp_sent * 2.0 * math.pi / float(path_info["wave_m"])))
                elif spec is not None and path_info.get("heading_rad") is not None:
                    # Cosine ramp: the tag heading is the sent line's peak heading.
                    spec = (float(spec[0]), math.copysign(float(path_info["heading_rad"]), float(spec[1])))
                speeds_txt = (f"v_plan={v_plan:.1f} v_meas={'na' if v_meas is None else format(v_meas, '.1f')} "
                              f"v_target={speed:.1f}")
                if v_src == "target_fallback":
                    print(f"collect {move.phase} {segment_id} no measured speed; v_plan=v_target={speed:.1f}", flush=True)
                if path is not None and move.phase == "weave" and path_info:
                    print(
                        f"collect weave {segment_id} {speeds_txt} amp={path_info.get('amp_m', 0.0):.2f}m "
                        f"wave={path_info.get('wave_m', 0.0):.0f}m line={path_info.get('length_m', 0.0):.0f}m "
                        f"lat_acc={path_info.get('lat_acc_mps2', 0.0):.2f} m/s2 "
                        f"(with road {path_info.get('total_lat_acc_mps2', 0.0):.2f}) shrunk={path_info.get('shrunk', 0)}",
                        flush=True,
                    )
                elif path is not None and path_info:
                    print(
                        f"collect {move.phase} {segment_id} {speeds_txt} delta={path_info.get('delta_m', 0.0):.2f}m "
                        f"ramp={path_info.get('ramp_m') or 0.0:.0f}m line={path_info.get('length_m', 0.0):.0f}m "
                        f"lat_acc={path_info.get('lat_acc_mps2', 0.0):.2f} m/s2 "
                        f"(with road {path_info.get('total_lat_acc_mps2', 0.0):.2f})",
                        flush=True,
                    )
                if path is None or spec is None:
                    why_refused = str(path_info.get("refused") or ("narrow" if path is not None else "no_path"))
                    print(f"collect {move.phase} {segment_id} refused {why_refused} {speeds_txt} "
                          f"lat_acc={path_info.get('lat_acc_mps2', 0.0):.2f} m/s2", flush=True)
                    weave_stats.refuse(why_refused, v_plan)
                    resolved = clock.force_abort("no_path")
                    path = None
                    plan_v, plan_phase = None, ""
                else:
                    if plan_v is None or plan_phase != move.phase:
                        wave_sent = float(path_info.get("wave_m") or 0.0)
                        weave_stats.start(
                            move.phase, v_plan, direction=int(used),
                            one_sided=move.phase == "weave" and wave_sent > 0.0
                            and float(path_info.get("length_m") or 0.0) <= 0.5 * wave_sent + 1.0,
                        )
                        plan_overspeed_noted = False
                        if move.phase == "lane_change" and path_info.get("ramp_m") is not None:
                            lane_change_run = {
                                "covered_m": 0.0,
                                "ramp_m": float(path_info["ramp_m"]),
                                "delta_m": float(path_info.get("delta_m") or 0.0),
                                "v_plan": float(v_plan),
                            }
                    plan_v, plan_phase = v_plan, move.phase
                    clock.direction = int(used)
                    resolved = Excursion(move.phase, float(spec[0]), float(spec[1]), int(used), move.reason)
                ai_key = (round(speed, 3), waypoint, resolved.phase, resolved.direction, map_name)
            send_ai = last_ai != ai_key
            if not driving and not waypoint and last_ai is not None:
                send_ai = False
            if send_ai:
                commands = (
                    excursion_commands(
                        speed,
                        resolved.phase,
                        resolved.direction,
                        traffic=False,
                        path=path,
                    )
                    if excursions_on and not waypoint
                    else ai_commands(speed, waypoint, traffic=False)
                )
                for command in commands:
                    try:
                        _queue(vehicle, command)
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                last_ai = ai_key
                held_tag = resolved
                if args.weave_guard:
                    sent_line = path if excursions_on and not waypoint and resolved.phase in ACTIVE else None
                    line_left_m = polyline_length(sent_line) if sent_line else None
                print(
                    f"collect ai speed={speed:.0f} m/s route={waypoint or map_name or 'span'} "
                    f"excursion={resolved.phase}",
                    flush=True,
                )
            move = resolved
            if lane_change_run is not None and move.phase == "lane_change":
                # The label heading is the peak the sent line has reached so far:
                # the full ramp peak from the ramp midpoint on, less for a car that
                # is short of it (a lane change that ends incomplete).
                reached = ramp_heading_reached(
                    lane_change_run["delta_m"], lane_change_run["ramp_m"], lane_change_run["covered_m"]
                )
                move = dataclasses.replace(move, target_heading_rad=math.copysign(reached, move.target_heading_rad))
            if clock.phase == "abort":
                if abort_t0 is None:
                    abort_t0 = now
            else:
                abort_t0 = None
            since_spawn[0] += tick_dt
            why = live.update(
                tick_dt,
                speed_mps_f,
                dir_z,
                damage_f,
                ai_should_move=not bool(waypoint),
                up_z=up_z,
                tip_z=TIP_DIR_Z_LIVE,
            )
            why = defer_stationary(why, stall_respawn_s, live.slow_s, _can_stall_respawn())
            if impact_respawn_due:
                # Logged impact on the previous tick (its row is written):
                # respawn now instead of driving on damaged.
                impact_respawn_due = False
                if not why:
                    _impact_end("impact_respawn", now)
                    if _impact_respawn(now):
                        time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                        tick_dt = max(1e-3, time.monotonic() - tick_mark)
                        continue
                    why = "impact"
            if (
                why in ("", "stationary")
                and not impact.should_respawn()
                and stall_respawn_due(stall_respawn_s, live.slow_s, since_spawn[0], _can_stall_respawn())
            ):
                _impact_end("stall_respawn", now)
                if _stall_respawn(now):
                    time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                    tick_dt = max(1e-3, time.monotonic() - tick_mark)
                    continue
                why = "stationary"
            if not why and impact.should_respawn():
                print(
                    f"collect impact {segment_id} stopped {impact.stall_s:.1f}s after the impact, respawning",
                    flush=True,
                )
                why = "impact"
            if why:
                if impact_event is not None:
                    # This tick aborts before its row is written: name the
                    # last written row so every impact maps to a real row.
                    impact_event["i"] = last_row_i
                    impact_event["i_row"] = "previous"
                    append_jsonl(out / IMPACTS_FILE, impact_event)
                    impact_event = None
                    if impact_respawn_on:
                        impact_respawns["abort_same_tick"] += 1  # the abort below respawns it
                _impact_end("respawn" if why == "impact" else why, now)
                for cid, prev in held:
                    if prev is None:
                        last_cam.pop(cid, None)
                    else:
                        last_cam[cid] = prev
                code = _abort(why)
                if code is not None:
                    return code
                time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
                tick_dt = max(1e-3, time.monotonic() - tick_mark)
                continue
            tags = {
                "segment": segment_id,
                "map": map_name or None,
                "excursion": move.as_tag(),
            }
            if spawn_kind:
                tags["spawn_type"] = spawn_kind
                tags["spawn_paint_cov"] = None if spawn_cov is None else round(float(spawn_cov), 3)
            if mark_recover:
                tags["recover"] = True
                mark_recover = False
            if settle.active(done_s):
                tags["settle"] = True
            if live.slow_s > 0.0:
                tags["stall"] = True
                tags["stall_s"] = round(float(live.slow_s), 2)
            if impact_decel is not None:
                tags["impact"] = True
                tags["impact_decel_mps2"] = round(float(impact_decel), 2)
            row_i = None if row_base is None else int(row_base) + int(getattr(writer, "enqueued", 0) or 0)
            queued = writer.submit(
                due,
                t=now,
                timestamps=due_stamps,
                ego={
                    "speed_mps": speed_mps_f,
                    "yaw_rate": sample.get("yaw_rate"),
                    "damage": damage_f,
                    "pos": pos,
                    "dir": direction,
                    "up": up_vec,
                    "x": None if pos is None else pos[0],
                    "y": None if pos is None else pos[1],
                },
                wheel={"steer": sample.get("steer")},
                pedals={"throttle": sample.get("throttle"), "brake": sample.get("brake")},
                tech=tech,
                driver="beamng_ai",
                tags=tags,
            )
            if impact_event is not None:
                impact_event["i"] = row_i if queued else last_row_i
                impact_event["i_row"] = "impact" if queued else "previous"
                append_jsonl(out / IMPACTS_FILE, impact_event)
                impact_respawn_due = impact_respawn_on
            yield_fire = False
            if queued:
                impact.note_row(now, row_i)
                last_row_i = row_i
                if spawn_deck is not None and spawn_road:
                    spawn_deck.note_time(spawn_road, min(0.5, tick_dt))
                yield_fire = yield_guard.update(
                    tick_dt, speed_mps_f, label_valid_at(tech, 10.0), settle.active(done_s)
                )
            if damage_f is not None:
                damage_seen = True
            elif not damage_seen and not damage_warned and meter.frames >= DAMAGE_WARN_ROWS:
                print(
                    f"collect damage sensor not reporting after {meter.frames} rows; damage abort inactive",
                    flush=True,
                )
                damage_warned = True
            if not queued:
                for cid, prev in held:
                    if prev is None:
                        last_cam.pop(cid, None)
                    else:
                        last_cam[cid] = prev
                print(
                    f"collect drop writer queue full cams={sorted(due)} "
                    f"{format_queue_line(writer.stats())}",
                    flush=True,
                )
            mask = tech.get("lane_gt_valid") or []
            try:
                gt_ok = float(sum(mask)) > 0.0
            except TypeError:
                gt_ok = False
            meter.note(set(frames), gt_ok)
            if yield_fire:
                # Cut the spawn or segment, never its rows. See --min-live-yield.
                rate = yield_guard.rate or 0.0
                yield_guard.cut()
                here = None if pos is None else [round(float(v), 2) for v in list(pos)[:3]]
                road_here = None if hit is None else (str(getattr(hit, "road_id", "") or "") or None)
                kind_was, road_was = spawn_kind or None, spawn_road
                can_respawn = map_name == "west_coast_usa" and vehicle is not None and spawn_deck is not None
                boundary = None if can_respawn else segment_boundary(
                    done_s + schedule_cut,
                    map_names,
                    segment_s,
                    map_weights or None,
                    None if target_s == float("inf") else target_s,
                )
                plan, new_cut = low_yield_plan(can_respawn, boundary, done_s, schedule_cut)
                if plan == "respawn":
                    mark = spawn_mark
                    _apply_spawn(True)
                    if spawn_mark == mark:
                        plan = "keep"
                elif plan == "segment_end":
                    schedule_cut = new_cut
                    segment_open = False
                where = "na" if here is None else f"({here[0]:.1f},{here[1]:.1f},{here[2]:.1f})"
                print(
                    f"collect low_yield {segment_id} {kind_was or '-'} valid10={rate:.2f} "
                    f"< {yield_guard.min_yield:.2f} over {yield_guard.window_s:.0f}s moving pos={where} "
                    f"road={road_here or 'na'} spawn_road={road_was or 'na'} "
                    f"cut {yield_guard.in_a_row}/{yield_guard.max_cuts} -> {plan}",
                    flush=True,
                )
                append_jsonl(
                    out / CUTS_FILE,
                    {
                        "kind": "low_yield",
                        "t": round(float(now), 6),
                        "i": last_row_i,
                        "segment": segment_id,
                        "map": map_name or None,
                        "spawn_type": kind_was,
                        "spawn_road": road_was,
                        "road": road_here,
                        "pos": here,
                        "valid10": round(float(rate), 4),
                        "min_live_yield": yield_guard.min_yield,
                        "window_s": yield_guard.window_s,
                        "cut_in_a_row": yield_guard.in_a_row,
                        "action": plan,
                        "done_s": round(float(done_s), 3),
                    },
                )
                if yield_guard.paused:
                    print(
                        f"collect low_yield paused after {yield_guard.in_a_row} cuts in a row; "
                        f"cuts again after a {yield_guard.window_s:.0f}s window at or above "
                        f"{yield_guard.min_yield:.2f}",
                        flush=True,
                    )
            elif (
                queued
                and recheck_due(recheck_s, done_s, recheck_mark)
                and spawn_deck is not None
                and spawn_deck.filtering()
                and map_name == "west_coast_usa"
                and vehicle is not None
                and not settle.active(done_s)
            ):
                # Respawn when the road the AI wandered onto is unpainted. Never drops rows.
                recheck_mark = done_s
                here = None if pos is None else [round(float(v), 2) for v in list(pos)[:3]]
                cov_now, road_now = spawn_deck.coverage_from(pos, direction)
                kind_was, road_was = spawn_kind or None, spawn_road
                type_now = None if road_now is None else spawn_deck.road_type(road_now, pos)
                plan = recheck_plan(cov_now, spawn_deck.min_paint_cov)
                new_road = new_type = None
                if plan == "respawn":
                    mark = spawn_mark
                    _apply_spawn(True, "recheck")
                    if spawn_mark == mark:
                        plan = "keep"
                    else:
                        new_road, new_type = spawn_road, spawn_deck.last_type
                where = "na" if here is None else f"({here[0]:.1f},{here[1]:.1f},{here[2]:.1f})"
                respawned = "" if new_road is None else f" new_road={new_road} new_type={new_type}"
                print(
                    f"collect recheck_cov {segment_id} {kind_was or '-'} "
                    f"paint_cov={'na' if cov_now is None else format(cov_now, '.2f')} "
                    f"min={spawn_deck.min_paint_cov:.2f} road={road_now or 'na'} road_type={type_now or 'na'} "
                    f"spawn_road={road_was or 'na'} pos={where}{respawned} -> {plan}",
                    flush=True,
                )
                append_jsonl(
                    out / CUTS_FILE,
                    recheck_record(
                        t=now,
                        i=last_row_i,
                        segment=segment_id,
                        map_name=map_name,
                        spawn_type=kind_was,
                        spawn_road=road_was,
                        road=road_now,
                        pos=here,
                        cov=cov_now,
                        min_cov=spawn_deck.min_paint_cov,
                        recheck_min=recheck_min,
                        action=plan,
                        done_s=done_s,
                        road_type=type_now,
                        new_spawn_road=new_road,
                        new_road_type=new_type,
                    ),
                )
            time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
            tick_dt = max(1e-3, time.monotonic() - tick_mark)
            done_s += tick_dt
            if not smoke_ok and done_s >= float(args.smoke_s):
                if not writer.wait_empty(30.0):
                    print("collect smoke failed ['writer did not finish']", flush=True)
                    run.stop("smoke_failed", "writer did not finish")
                    return 5
                issues = decode_issues(out)
                if issues:
                    print(f"collect smoke failed {issues}", flush=True)
                    run.stop("smoke_failed", str(issues))
                    return 5
                hz_now = _hz()
                print(
                    f"collect smoke ok t={done_s:.0f}s frames={meter.frames} hz={hz_now:.2f}",
                    flush=True,
                )
                smoke_ok = True
            now_m = time.monotonic()
            if now_m - last_disk >= DISK_CHECK_S:
                nbytes = _used_bytes()
                try:
                    free = check_data_path(out)
                except DataDriveError as exc:
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    print(
                        format_disk_line(done_s, nbytes, ceiling_gb_per_hour=ceiling_gb, hz=_hz()),
                        flush=True,
                    )
                    print(format_queue_line(writer.stats()), flush=True)
                    print(f"collect stop {exc}", flush=True)
                    run.stop("disk_low", str(exc))
                    return 3
                last_disk = now_m
                if now_m - last_print >= 60.0:
                    print(
                        format_disk_line(
                            done_s,
                            nbytes,
                            free,
                            ceiling_gb_per_hour=ceiling_gb,
                            hz=_hz(),
                        ),
                        flush=True,
                    )
                    print(format_queue_line(writer.stats()), flush=True)
                    if spawn_deck is not None:
                        print(f"collect {spawn_deck.summary_line()}", flush=True)
                    _save_progress(done_s, done_s if target_s == float("inf") else target_s)
                    _publish(free, speed, waypoint)
                    last_print = now_m
        _save_progress(done_s, target_s if target_s < float("inf") else done_s)
        _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
        print(format_disk_line(done_s, _used_bytes(), ceiling_gb_per_hour=ceiling_gb, hz=_hz()), flush=True)
        print(format_queue_line(writer.stats()), flush=True)
        if spawn_deck is not None:
            print(f"collect {spawn_deck.summary_line()}", flush=True)
        _say_repeats()
        run.stop("target_reached")
        run.done_s = float(done_s)
        return 0
    finally:
        if writer is not None:
            writer.close()
        release_tech_beamngpy(bng)


def _attach_cameras(bng: Any, vehicle: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    from beamngpy.sensors import Camera  # type: ignore

    from python.sensors.cameras import (
        CAM_IDS,
        beamng_camera_sensor_kwargs,
        camera_clip_planes,
        camera_update_priority,
        camera_update_s,
        yaw_pitch_to_dir_up,
    )
    from python.sensors.tech import gvd_to_bng_vehicle

    origin = cfg.get("origin_offset") or [0.0, 0.0, 0.0]
    attached: dict[str, Any] = {}
    for spec in cfg.get("cameras") or []:
        if not isinstance(spec, dict):
            continue
        cid = str(spec.get("id") or "")
        if cid not in CAM_IDS:
            continue
        pos = list(spec.get("pos_m") or [0.0, 1.2, 1.2])
        pos = [float(pos[0]) + float(origin[0]), float(pos[1]) + float(origin[1]), float(pos[2]) + float(origin[2])]
        yaw = float(spec.get("yaw_deg") or 0.0)
        pitch = float(spec.get("pitch_deg") or 0.0)
        direction, up = yaw_pitch_to_dir_up(yaw, pitch)
        res = spec.get("live_res") or [640, 480]
        near_m, far_m = camera_clip_planes(spec, cid=cid, defaults={"near_m": cfg.get("near_m", 0.05)})
        kwargs = beamng_camera_sensor_kwargs(
            pos=gvd_to_bng_vehicle(*pos),
            direction=gvd_to_bng_vehicle(*direction),
            up=gvd_to_bng_vehicle(*up),
            fov_v=float(spec.get("fov_v") or 70.0),
            resolution=(int(res[0]), int(res[1])),
            update_s=camera_update_s(spec, cid=cid),
            near_m=near_m,
            far_m=far_m,
            shmem=True,
            streaming=True,
            rgb_only=True,
            update_priority=camera_update_priority(spec, cid=cid),
            cid=cid,
        )
        attached[cid] = Camera(f"gvd_collect_{cid}", bng, vehicle, **kwargs)
    if "main" not in attached:
        raise RuntimeError("main camera did not attach")
    return attached


def _camera_resolution(cam: Any) -> tuple[int, int] | None:
    res = getattr(cam, "resolution", None)
    if res is None:
        return None
    try:
        width, height = int(res[0]), int(res[1])
    except (TypeError, ValueError, IndexError):
        return None
    if width < 1 or height < 1:
        return None
    return width, height


def _read_camera(cid: str, cam: Any) -> tuple[Any, str, Any]:
    """One picture, why it was dropped, and the raw reading for its timestamp.

    Live grab reads shared memory with ``stream_raw``. ``poll`` is a
    game-engine round trip per camera; eight of those, plus the 0.1 s
    sleep, is the ~2.3 Hz collect loop. A streaming camera has no picture
    in the poll reply.
    """
    import numpy as np

    from python.sensors.cameras import colour_to_bgr, frame_is_unrendered

    resolution = _camera_resolution(cam)
    raw: Any = None
    how = "poll"
    if callable(getattr(cam, "stream_raw", None)):
        how = "stream_raw"
        try:
            raw = cam.stream_raw()
        except Exception as exc:
            return None, f"stream_raw {type(exc).__name__}: {exc}", None
    elif callable(getattr(cam, "poll", None)):
        try:
            raw = cam.poll()
        except Exception as exc:
            return None, f"poll {type(exc).__name__}: {exc}", None
    else:
        return None, "no stream_raw or poll", None
    if raw is None:
        return None, f"{how} returned None", None
    if isinstance(raw, np.ndarray):
        image = colour_to_bgr(raw, resolution, tone=False)
    elif isinstance(raw, dict):
        image = colour_to_bgr(_colour(raw), resolution, tone=False)
    else:
        return None, f"{how} returned {type(raw).__name__}", raw
    if image is None:
        keys = ",".join(sorted(raw)) if isinstance(raw, dict) else ""
        return None, f"{how} had no colour array" + (f" keys={keys}" if keys else ""), raw
    if frame_is_unrendered(image):
        return None, f"{how} unrendered (all zeros)", raw
    return image, "", raw


def _poll_cameras(cams: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Pictures as the sensor returned them, plus why each camera was dropped."""
    frames: dict[str, Any] = {}
    stamps: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for cid, cam in cams.items():
        image, reason, raw = _read_camera(cid, cam)
        if image is None:
            reasons[str(cid)] = reason or "missing"
            continue
        frames[cid] = image
        sensor_t = None
        if isinstance(raw, dict):
            for key in ("timestamp", "time", "t"):
                if raw.get(key) is not None:
                    sensor_t = raw.get(key)
                    break
        stamps[cid] = {"wall": time.time(), "sensor_t": sensor_t}
    return frames, stamps, reasons


def _write_collect_crash(directory: Any, message: str, reasons: dict[str, str]) -> None:
    if directory is None:
        return
    path = Path(directory) / "collect_crash.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message.rstrip() + "\n" + json.dumps(reasons, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_vehicle(vehicle: Any) -> dict[str, Any]:
    """One sensor poll. Controls are whatever the vehicle state actually holds."""
    try:
        if hasattr(vehicle, "poll_sensors"):
            vehicle.poll_sensors()
    except Exception:
        pass
    state = getattr(vehicle, "state", None)
    if not isinstance(state, dict):
        try:
            state = vehicle.sensors.poll() if hasattr(vehicle, "sensors") else None
        except Exception:
            state = None
    state = dict(state) if isinstance(state, dict) else None
    try:
        sensors = getattr(vehicle, "sensors", None)
        el = sensors["electrics"] if sensors is not None else None
        data = getattr(el, "data", el)
        if state is not None and data is not None:
            state["electrics"] = {str(k): data[k] for k in data}
    except Exception:
        pass
    try:
        sensors = getattr(vehicle, "sensors", None)
        dmg = sensors["damage"] if sensors is not None else None
        dmg_data = getattr(dmg, "data", dmg)
        if state is not None and state.get("damage") is None and dmg_data is not None:
            total = dmg_data.get("damage")
            if total is not None:
                state["damage"] = float(total)
    except Exception:
        pass
    sample = controls_from_state(state)
    up = state.get("up") if isinstance(state, dict) else None
    if isinstance(up, (list, tuple)) and len(up) >= 3:
        try:
            sample["up"] = [float(up[0]), float(up[1]), float(up[2])]
        except (TypeError, ValueError):
            pass
    if sample.get("pos") is None and isinstance(state, dict):
        pos = state.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            sample["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
    return sample


if __name__ == "__main__":
    raise SystemExit(main())
