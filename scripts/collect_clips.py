"""Edge-case clip mode: many 30-60 s lossless clips per session, each in its own new folder.

Short clips are the edge-case batch (30-60 s). Optional ``--long-clips N`` mixes
N 10-15 minute regular-driving clips through the batch (default 0, so current
behaviour is unchanged). A highway-loop long clip is planned from the map's road
dump (python/collect/long_route.py): the AI spawns at one end of a highway facing
along it, is driven to the far end (rolling ai.setLine windows), then loops back
over non-highway roads (dirt where the dump's road material says unpaved) to a
different route each time, the direction alternating through the batch. The
route goes to the collector as ``--route-plan``; afterwards the recorded ego pose
is scored against it (clip_tags.json ``route_measure``, session.json
``long_route_totals``). Maps with no highway (or no plannable route) roam
freely with ``--route span`` and are not skipped. Nothing here has been run
against a live BeamNG.

    py -3 scripts\\collect_clips.py --clips 5 --out E:\\GVD_data\\clips_v2 \\
        --maps west_coast_usa,italy,east_coast_usa --max-traffic 6 --seed 1 --chase-cam

Every clip is one ``collect_scene_lanes.py`` process (raw lossless PNG, all
eight cameras with training-only depth/sem, the Hirochi eSBR, BeamNG.tech
in Vulkan). The plan per clip (edge case, map, traffic, weather, time of
day, length, spawn seed) comes from ``python/collect/clip_mode.py``. Each
clip gets ``clip_tags.json``; a clip that fails the quick check is marked
``pretrain_only`` and kept. The session folder ``<out>/clips_<stamp>`` and
every clip folder are created new; nothing existing is reused or replaced.
Extra arguments after ``--`` go to every collector run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect.clip_mode import (  # noqa: E402
    DEFAULT_LONG_CLIP_MAX_S,
    DEFAULT_LONG_CLIP_MIN_S,
    DEFAULT_LONG_CLIPS,
    DEFAULT_LONG_HIGHWAY_FRAC,
    EDGE_CASES,
    GFX,
    MAX_CLIP_S,
    MIN_CLIP_S,
    run_clips,
)
from python.collect.tech_watchdog import (  # noqa: E402
    DEFAULT_TECH_FAIL_RELAUNCH,
    DEFAULT_TECH_RELAUNCH_CAP,
    TechWatchdog,
    resolve_tech_exe,
)

COLLECTOR = ROOT / "scripts" / "collect_scene_lanes.py"
DEFAULT_MAPS = "west_coast_usa,italy,east_coast_usa"


def _maps(text: str) -> tuple[str, ...]:
    out = tuple(m.strip() for m in str(text).split(",") if m.strip())
    if not out:
        raise argparse.ArgumentTypeError("--maps needs at least one map")
    return out


def _long_clip_s(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--long-clip-s needs MIN,MAX")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--long-clip-s needs MIN,MAX as numbers") from exc
    if not (lo > 0.0 and hi >= lo):
        raise argparse.ArgumentTypeError("--long-clip-s needs 0 < MIN <= MAX")
    return lo, hi


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Root folder; a new clips_<stamp> session goes under it.")
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Name the session folder clips_<id> and write it to session.json as session_id "
            "(the supervisor passes one per launch so it reads its own session, not the newest). "
            "Default: a timestamp."
        ),
    )
    parser.add_argument("--clips", type=int, default=20, help="Clips in this session.")
    parser.add_argument("--seed", type=int, default=0, help="Session seed: the same seed gives the same plans.")
    parser.add_argument("--maps", type=_maps, default=_maps(DEFAULT_MAPS), help="Comma-separated map pool.")
    parser.add_argument("--max-traffic", type=int, default=6, help="Most companion cars per clip (heavy_traffic uses all).")
    parser.add_argument("--min-s", type=float, default=MIN_CLIP_S, help="Shortest clip, seconds (>= 30).")
    parser.add_argument("--max-s", type=float, default=MAX_CLIP_S, help="Longest clip, seconds (<= 60).")
    parser.add_argument(
        "--long-clips",
        type=int,
        default=DEFAULT_LONG_CLIPS,
        metavar="N",
        help=(
            "How many clips in this batch are long regular-driving runs "
            f"(default {DEFAULT_LONG_CLIPS}: off; short 30-60 s clips stay "
            "the edge cases. The supervisor must pass this flag). About 5-10 "
            "is the usual ask; that is a recommendation, not a cap (N cannot "
            "exceed --clips). They are 10-15 min (see --long-clip-s), spread "
            "by --seed. When the map has a highway, the AI spawns at one end, "
            "is driven to the far end, then loops back over non-highway roads "
            "(dirt where the map has it), a different route each time with "
            "the direction alternating; how far it got is recorded from the "
            "GT pose (route_measure). Maps with no highway are not skipped: "
            "the AI roams freely (span). Tagged 'long' (and 'long_highway' "
            "only for a highway-first route)."
        ),
    )
    parser.add_argument(
        "--long-clip-s",
        type=_long_clip_s,
        default=_long_clip_s(f"{DEFAULT_LONG_CLIP_MIN_S:g},{DEFAULT_LONG_CLIP_MAX_S:g}"),
        metavar="MIN,MAX",
        help=(
            "Duration range in seconds for --long-clips "
            f"(default {DEFAULT_LONG_CLIP_MIN_S:g},{DEFAULT_LONG_CLIP_MAX_S:g}: "
            "10-15 minutes of regular driving)."
        ),
    )
    parser.add_argument(
        "--long-clip-highway-frac",
        type=float,
        default=DEFAULT_LONG_HIGHWAY_FRAC,
        metavar="FRAC",
        help=(
            "Fraction of long clips that use a highway-first route when the "
            f"session pool has a highway map (default {DEFAULT_LONG_HIGHWAY_FRAC:g}). "
            "The rest, and any clip on a map with no highway, roam freely "
            "(span, seeded spawn). 0-1 inclusive."
        ),
    )
    parser.add_argument("--edge-cases", default="", help=f"Comma-separated subset of {','.join(EDGE_CASES)}.")
    parser.add_argument("--chase-cam", action="store_true", help="Also record the recording-only chase camera.")
    parser.add_argument(
        "--gfx", default=GFX,
        help=f"Renderer recorded in clip_tags.json and session.json (default {GFX}, Vulkan). The collector "
             f"launches BeamNG.tech with -gfx {GFX}; an already-running BeamNG keeps its own renderer.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python that runs the collector.")
    parser.add_argument(
        "--tech-fail-relaunch",
        type=int,
        default=DEFAULT_TECH_FAIL_RELAUNCH,
        metavar="N",
        help=(
            "After N consecutive session_open_failed or level_load_failed clips, "
            "stop attaching, kill hung Tech, relaunch, and wait for Hello "
            f"(default {DEFAULT_TECH_FAIL_RELAUNCH}; 0=off)."
        ),
    )
    parser.add_argument(
        "--tech-relaunch-cap",
        type=int,
        default=DEFAULT_TECH_RELAUNCH_CAP,
        metavar="N",
        help=(
            f"Max Tech relaunches in this batch (default {DEFAULT_TECH_RELAUNCH_CAP}). "
            "A further trigger halts the session with stop_reason=tech_relaunch_limit. "
            "Other session.json stop_reason values: tech_relaunch_unavailable (no known exe), "
            "tech_relaunch_hello_failed (relaunch ran, Hello did not succeed), "
            "tech_relaunch_port_stuck (research port still LISTENING after the 20 s "
            "kill wait; no second Tech is launched)."
        ),
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Pause file checked before each clip (default <out>/STOP). When it exists the running "
             "clip finishes and is tagged, no new clip starts, and session.json gets "
             "stop_reason=stop_file. Pass an empty string to disable.",
    )
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    elif extra:
        parser.error(f"unknown arguments {extra}; pass collector arguments after --")
    if args.clips < 1:
        parser.error("--clips must be at least 1")
    if args.long_clips < 0:
        parser.error("--long-clips must be 0 or more")
    if args.long_clips > args.clips:
        parser.error("--long-clips cannot exceed --clips")
    if args.max_traffic < 0:
        parser.error("--max-traffic must be 0 or more")
    if not (MIN_CLIP_S <= args.min_s <= args.max_s <= MAX_CLIP_S):
        parser.error(f"clip length must satisfy {MIN_CLIP_S:.0f} <= --min-s <= --max-s <= {MAX_CLIP_S:.0f}")
    if not (0.0 <= float(args.long_clip_highway_frac) <= 1.0):
        parser.error("--long-clip-highway-frac must be between 0 and 1")
    if args.tech_fail_relaunch < 0:
        parser.error("--tech-fail-relaunch must be 0 or more")
    if args.tech_relaunch_cap < 1:
        parser.error("--tech-relaunch-cap must be at least 1")
    cases = tuple(c.strip() for c in args.edge_cases.split(",") if c.strip())
    bad = [c for c in cases if c not in EDGE_CASES]
    if bad:
        parser.error(f"unknown edge case(s) {bad}")
    args.edge_cases = cases or None
    if args.stop_file is None:
        args.stop_file = args.out / "STOP"
    elif not str(args.stop_file).strip():
        args.stop_file = None
    else:
        args.stop_file = Path(args.stop_file)
    args.extra = extra
    return args


def collector_command(argv: list[str], *, python: str | None = None) -> list[str]:
    return [python or sys.executable, str(COLLECTOR), *argv]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    def runner(clip_argv: list[str]) -> int:
        return subprocess.run(collector_command(clip_argv, python=args.python), cwd=str(ROOT)).returncode

    watchdog = TechWatchdog(
        fail_relaunch=args.tech_fail_relaunch,
        relaunch_cap=args.tech_relaunch_cap,
        exe=resolve_tech_exe(),
    )

    def after_clip(clip: Path, plan, code: int, qc) -> str | None:
        del plan, qc
        return watchdog.after_clip(clip, exit_code=code)

    session = run_clips(
        args.out,
        clips=args.clips,
        seed=args.seed,
        maps=args.maps,
        max_traffic=args.max_traffic,
        runner=runner,
        min_s=args.min_s,
        max_s=args.max_s,
        edge_cases=args.edge_cases,
        chase=args.chase_cam,
        extra=args.extra,
        gfx=args.gfx,
        after_clip=after_clip,
        stop_file=args.stop_file,
        long_clips=args.long_clips,
        long_min_s=args.long_clip_s[0],
        long_max_s=args.long_clip_s[1],
        long_highway_frac=args.long_clip_highway_frac,
        session_id=args.session_id,
    )
    print(f"session {session}")
    if watchdog.halt_reason:
        print(f"session stop reason={watchdog.halt_reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
