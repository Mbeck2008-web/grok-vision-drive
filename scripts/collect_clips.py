"""Edge-case clip mode: many 30-60 s lossless clips per session, each in its own new folder.

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

from python.collect.clip_mode import EDGE_CASES, GFX, MAX_CLIP_S, MIN_CLIP_S, run_clips  # noqa: E402
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Root folder; a new clips_<stamp> session goes under it.")
    parser.add_argument("--clips", type=int, default=20, help="Clips in this session.")
    parser.add_argument("--seed", type=int, default=0, help="Session seed: the same seed gives the same plans.")
    parser.add_argument("--maps", type=_maps, default=_maps(DEFAULT_MAPS), help="Comma-separated map pool.")
    parser.add_argument("--max-traffic", type=int, default=6, help="Most companion cars per clip (heavy_traffic uses all).")
    parser.add_argument("--min-s", type=float, default=MIN_CLIP_S, help="Shortest clip, seconds (>= 30).")
    parser.add_argument("--max-s", type=float, default=MAX_CLIP_S, help="Longest clip, seconds (<= 60).")
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
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    elif extra:
        parser.error(f"unknown arguments {extra}; pass collector arguments after --")
    if args.clips < 1:
        parser.error("--clips must be at least 1")
    if args.max_traffic < 0:
        parser.error("--max-traffic must be 0 or more")
    if not (MIN_CLIP_S <= args.min_s <= args.max_s <= MAX_CLIP_S):
        parser.error(f"clip length must satisfy {MIN_CLIP_S:.0f} <= --min-s <= --max-s <= {MAX_CLIP_S:.0f}")
    if args.tech_fail_relaunch < 0:
        parser.error("--tech-fail-relaunch must be 0 or more")
    if args.tech_relaunch_cap < 1:
        parser.error("--tech-relaunch-cap must be at least 1")
    cases = tuple(c.strip() for c in args.edge_cases.split(",") if c.strip())
    bad = [c for c in cases if c not in EDGE_CASES]
    if bad:
        parser.error(f"unknown edge case(s) {bad}")
    args.edge_cases = cases or None
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
    )
    print(f"session {session}")
    if watchdog.halt_reason:
        print(f"session stop reason={watchdog.halt_reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
