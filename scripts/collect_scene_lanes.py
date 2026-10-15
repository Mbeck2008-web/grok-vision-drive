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

    py -3 scripts\\collect_scene_lanes.py --check-maps --maps west_coast_usa,italy,east_coast_usa,utah,jungle_rock_island
    py -3 scripts\\collect_scene_lanes.py --minutes 120 --maps west_coast_usa,italy,east_coast_usa,utah,jungle_rock_island --segment-min 12 --smoke-s 60 --speed-min 8 --speed-max 17
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.raw_frames import controls_from_state

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


def save_progress(path: Path, done_s: float, target_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"done_s": float(done_s), "target_s": float(target_s)}, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record AI driving strips with GT lane targets")
    parser.add_argument("--minutes", type=float, default=45.0, help="Stop after this much driving. 0 with --until-disk runs to the 20 GB floor.")
    parser.add_argument("--until-disk", action="store_true", help="Ignore the clock. Stop only under 20 GB free.")
    parser.add_argument("--map", default="", help="Level to load. Empty keeps the current level.")
    parser.add_argument("--route", default="span", choices=["span", "waypoints", "random"])
    parser.add_argument("--speed-min", type=float, default=8.0)
    parser.add_argument("--speed-max", type=float, default=17.0)
    parser.add_argument("--cameras", default="", help="Comma list. Empty keeps every camera at its plan rate.")
    parser.add_argument("--rates", default="", help="Overrides, for example main=10,rear=1")
    parser.add_argument("--traffic", action="store_true", help="Traffic AI. Off is span, traffic-free.")
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
    parser.add_argument("--segment-min", type=float, default=12.0, help="Minutes on each map segment.")
    parser.add_argument("--no-excursions", action="store_true", help="Stay centered. Default weaves and lane changes.")
    parser.add_argument("--smoke-s", type=float, default=60.0, help="Collect this many seconds, check PNG decode, then continue. 0 skips the check.")
    parser.add_argument("--no-vary-env", action="store_true", help="Do not change weather or time of day on each segment.")
    parser.add_argument(
        "--check-maps",
        action="store_true",
        help="Load each map, print spawn, road edges, and cameras, then exit.",
    )
    return parser


def segment_at(elapsed_s: float, maps: tuple[str, ...], segment_s: float) -> tuple[str, int]:
    """Map name and segment index. Empty maps stay on the loaded level."""
    if not maps:
        return "", 0
    span = max(1.0, float(segment_s))
    index = int(max(0.0, float(elapsed_s)) // span)
    return maps[index % len(maps)], index


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


def run_map_checks(bng: Any, levels: tuple[str, ...] | list[str], cfg: dict[str, Any]) -> int:
    """Load each level, report spawn/edges/cameras, and exit. No recording."""
    failed = 0
    cams: dict[str, Any] = {}
    for level in levels:
        opened = enter_map(bng, level, cfg, cams)
        cams = dict(getattr(opened, "cameras", None) or {})
        print(format_check_line(level, opened), flush=True)
        if opened.error or not opened.roads or not opened.cameras:
            failed += 1
    drop_cameras(cams)
    return 2 if failed else 0


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


def main(argv: list[str] | None = None) -> int:
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
    from python.debug.gt_geom import heading_from_dir, project_road
    from python.debug.gt_logger import load_road_polys
    from python.debug.lane_targets import tech_from_hit
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
    if args.check_maps and not map_names:
        from python.collect.auto_mode import DEFAULT_MAPS

        map_names = tuple(DEFAULT_MAPS)
    segment_s = max(60.0, float(args.segment_min) * 60.0)
    excursions_on = not bool(args.no_excursions)
    vary_env = bool(map_names) and not bool(args.no_vary_env)
    speeds = speeds_for(args.speed_min, args.speed_max)
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
    if done_s >= target_s:
        print(f"collect already reached {done_s / 60.0:.1f} min in {out}")
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
        return 2
    widen_socket_timeout(bng, 90.0)
    writer = None
    try:
        if map_names:
            installed, list_note = installed_levels(bng)
            if installed is None:
                print(f"collect level list unavailable ({list_note}); trying the requested maps", flush=True)
            else:
                kept, skipped = select_maps(map_names, installed)
                for name in skipped:
                    print(f"collect skip level {name} (not in this Tech install)", flush=True)
                map_names = tuple(kept)
                if not map_names:
                    message = "none of the requested maps are in this Tech install"
                    record_session_failure(status_root, message)
                    print(f"collect stop {message}", flush=True)
                    return 2
        elif args.reload_level:
            print(f"collect reload level={args.reload_level}", flush=True)
        if args.check_maps:
            code = run_map_checks(bng, map_names, cam_cfg)
            if code != 0:
                record_session_failure(status_root, "check-maps failed")
            return code
        # A map rotation enters the level on the first segment, after the
        # writer exists. A single-level run keeps the scenario already started.
        vehicle = None
        roads: list[Any] = []
        cams: dict[str, Any] = {}
        if not map_names:
            opened = enter_map(bng, str(args.reload_level or ""), cam_cfg, {})
            vehicle, roads, cams, err = opened.vehicle, opened.roads, opened.cameras, opened.error
            if err or vehicle is None or not roads:
                message = err or "no vehicle after load"
                record_session_failure(status_root, message)
                print(f"collect stop {message}", flush=True)
                return 2
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
            excursion_commands,
            excursion_path,
            near_junction,
            targets_for,
        )

        clock = ExcursionClock()
        last_ai: tuple[Any, ...] | None = None
        held_tag: Excursion | None = None
        loaded_map = ""
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
                ai_mode=("manual" if route else "traffic" if args.traffic else "span"),
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
        while done_s < target_s:
            if stop_file(out).is_file():
                save_progress(progress_path, done_s, target_s if target_s < float("inf") else done_s)
                _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
                print("collect stop requested", flush=True)
                return 0
            elapsed = done_s
            speed = speed_at(elapsed, speeds)
            waypoint = waypoint_at(elapsed, names)
            map_name, seg_index = segment_at(elapsed, map_names, segment_s)
            segment_id = f"{map_name}#{seg_index}" if map_name else "current#0"
            if map_name and map_name != loaded_map:
                print(f"collect segment {segment_id}", flush=True)
                opened = enter_map(bng, map_name, cam_cfg, cams)
                vehicle, roads, cams, err = opened.vehicle, opened.roads, opened.cameras, opened.error
                if err or vehicle is None or not roads or not cams:
                    last_error = err or "no vehicle after load"
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    _write_collect_crash(status_root, last_error, {})
                    print(f"collect stop {last_error}", flush=True)
                    return 2
                loaded_map = map_name
                last_ai = None
                held_tag = None
                clock = ExcursionClock()
                prev_dir = None
                prev_dir_t = None
                last_cam = {}
                if vary_env:
                    tod, sky = segment_env(seg_index)
                    sky_commands = environment_commands(tod, sky)
                else:
                    sky_commands = environment_commands(args.time_of_day, args.weather)
                for command in sky_commands:
                    try:
                        _queue(vehicle, command)
                    except Exception as exc:
                        last_error = session_text(exc) if is_session_error(exc) else f"{type(exc).__name__}: {exc}"
                        _write_collect_crash(status_root, last_error, {})
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
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
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
                save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                _publish(None, speed, waypoint)
                _write_collect_crash(status_root, outage, cam_reasons)
                print(f"collect stop {outage}", flush=True)
                return 4
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
            heading = None
            if direction is not None:
                heading = heading_from_dir(direction[0], direction[1])
            hit = None
            if pos is not None:
                hit = project_road(roads, pos[0], pos[1], heading)
            tech = tech_from_hit(
                None if hit is None else hit.lane_offset_m,
                None if hit is None else hit.heading_err_rad,
                None if hit is None else hit.lane_width_m,
                None if hit is None else hit.off_road,
            )
            if excursions_on and not waypoint:
                move = clock.update(
                    tick_dt,
                    offset_m=None if hit is None else hit.lane_offset_m,
                    heading_rad=None if hit is None else hit.heading_err_rad,
                    lane_width_m=None if hit is None else hit.lane_width_m,
                    off_road=bool(hit.off_road) if hit is not None else False,
                    near_junction=near_junction(hit),
                    traffic=bool(args.traffic),
                    n_lanes=road_lanes(roads, hit),
                )
            else:
                move = Excursion("cruise", 0.0, 0.0, 1, "")
            resolved = held_tag if held_tag is not None and held_tag.phase == move.phase else move
            path = None
            ai_key = (round(speed, 3), waypoint, resolved.phase, resolved.direction, map_name)
            if last_ai != ai_key and excursions_on and not waypoint and move.phase in ACTIVE and pos is not None:
                path, used = excursion_path(
                    roads,
                    float(pos[0]),
                    float(pos[1]),
                    heading,
                    move,
                    hit,
                    speed,
                )
                road = road_for(roads, hit)
                spec = None if path is None else targets_for(
                    move.phase,
                    used,
                    None if hit is None else hit.lane_width_m,
                    lane_id=0 if hit is None else int(getattr(hit, "lane_id", 0) or 0),
                    road_width_m=None if road is None else float(road.width_m),
                )
                if path is None or spec is None:
                    resolved = clock.force_abort("no_path")
                    path = None
                else:
                    clock.direction = int(used)
                    resolved = Excursion(move.phase, float(spec[0]), float(spec[1]), int(used), move.reason)
                ai_key = (round(speed, 3), waypoint, resolved.phase, resolved.direction, map_name)
            if last_ai != ai_key:
                commands = (
                    excursion_commands(
                        speed,
                        resolved.phase,
                        resolved.direction,
                        traffic=bool(args.traffic),
                        path=path,
                    )
                    if excursions_on and not waypoint
                    else ai_commands(speed, waypoint, traffic=bool(args.traffic))
                )
                for command in commands:
                    try:
                        _queue(vehicle, command)
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                last_ai = ai_key
                held_tag = resolved
                print(
                    f"collect ai speed={speed:.0f} m/s route={waypoint or map_name or 'span'} "
                    f"excursion={resolved.phase}",
                    flush=True,
                )
            move = resolved
            queued = writer.submit(
                due,
                t=now,
                timestamps=due_stamps,
                ego={
                    "speed_mps": sample.get("speed_mps"),
                    "yaw_rate": sample.get("yaw_rate"),
                    "pos": pos,
                    "dir": direction,
                    "x": None if pos is None else pos[0],
                    "y": None if pos is None else pos[1],
                },
                wheel={"steer": sample.get("steer")},
                pedals={"throttle": sample.get("throttle"), "brake": sample.get("brake")},
                tech=tech,
                driver="beamng_ai",
                tags={
                    "segment": segment_id,
                    "map": map_name or None,
                    "excursion": move.as_tag(),
                },
            )
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
            time.sleep(tick_sleep_s(time.monotonic() - tick_mark, 0.1))
            tick_dt = max(1e-3, time.monotonic() - tick_mark)
            done_s += tick_dt
            if not smoke_ok and done_s >= float(args.smoke_s):
                if not writer.wait_empty(30.0):
                    print("collect smoke failed ['writer did not finish']", flush=True)
                    return 5
                issues = decode_issues(out)
                if issues:
                    print(f"collect smoke failed {issues}", flush=True)
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
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    print(
                        format_disk_line(done_s, nbytes, ceiling_gb_per_hour=ceiling_gb, hz=_hz()),
                        flush=True,
                    )
                    print(format_queue_line(writer.stats()), flush=True)
                    print(f"collect stop {exc}", flush=True)
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
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                    _publish(free, speed, waypoint)
                    last_print = now_m
        save_progress(progress_path, done_s, target_s if target_s < float("inf") else done_s)
        _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
        print(format_disk_line(done_s, _used_bytes(), ceiling_gb_per_hour=ceiling_gb, hz=_hz()), flush=True)
        print(format_queue_line(writer.stats()), flush=True)
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
    sample = controls_from_state(state if isinstance(state, dict) else None)
    if sample.get("pos") is None and isinstance(state, dict):
        pos = state.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            sample["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
    return sample


if __name__ == "__main__":
    raise SystemExit(main())
