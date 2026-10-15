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

    .venv\\Scripts\\python scripts\\collect_scene_lanes.py --minutes 45
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

SPEEDS_MPS = (8.0, 11.0, 14.0, 17.0, 10.0)
SPEED_HOLD_S = 180.0
WAYPOINT_HOLD_S = 180.0
DISK_CHECK_S = 10.0


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
    return line


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
    return parser


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
    api = getattr(bng, "vehicles", None)
    vehicles: dict[str, Any] = {}
    if api is not None and hasattr(api, "get_current"):
        cur = api.get_current()
        if isinstance(cur, dict):
            vehicles = {str(k): v for k, v in cur.items() if v is not None}
    vid = None
    if api is not None and hasattr(api, "get_player_vehicle_id"):
        info = api.get_player_vehicle_id()
        if isinstance(info, dict):
            vid = info.get("vid") or info.get("name")
        elif info:
            vid = info
    veh = vehicles.get(str(vid)) if vid else None
    if veh is None and vehicles:
        veh = next(iter(vehicles.values()))
    if veh is None:
        raise RuntimeError("no vehicle in the running session")
    if hasattr(veh, "connect"):
        try:
            if not getattr(veh, "is_connected", lambda: True)():
                veh.connect(bng)
        except Exception:
            pass
    return veh


def _colour(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("colour", "color", "image"):
            if payload.get(key) is not None:
                return payload[key]
    return payload


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
        controls_from_state,
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
    bng = open_tech_beamngpy(
        host, port, home=cfg.get("home") or None, user=cfg.get("user") or None, launch=False,
    )
    widen_socket_timeout(bng, 90.0)
    writer = None
    try:
        if args.reload_level:
            print(f"collect reload level={args.reload_level}", flush=True)
            # The running level is replaced. The socket stays open.
            loader = getattr(getattr(bng, "scenario", None), "load_level", None) or getattr(bng, "load_level", None)
            # Prefer the control queue so we do not invent a scenario object.
            queue = getattr(bng, "control", None)
            send = getattr(queue, "queue_lua_command", None) or getattr(bng, "queue_lua_command", None)
            if callable(send):
                send(f'freeroam_freeroam.startFreeroam("{args.reload_level}")')
                time.sleep(8.0)
            elif callable(loader):
                loader(args.reload_level)
        vehicle = _player_vehicle(bng)
        roads, note = load_road_polys(bng)
        if not roads:
            print(f"collect no road network ({note})", file=sys.stderr)
            return 2
        cams = _attach_cameras(bng, vehicle, cam_cfg)
        writer = RawWriteQueue(RawFrameWriter(out, camera_sidecar(cam_cfg)), max_depth=8)
        from python.data.strip_writer import STRIP_ORDER
        from python.debug.remote_status import CollectMeter, status_dir, write_collect_status

        meter = CollectMeter(STRIP_ORDER)
        status_root = status_dir()
        last_print = time.monotonic()
        last_disk = last_print
        last_ai: tuple[float, str | None] | None = None
        last_cam: dict[str, float] = {}
        prev_dir: list[float] | None = None
        prev_dir_t: float | None = None
        last_error = ""
        for command in environment_commands(args.time_of_day, args.weather):
            try:
                _queue(vehicle, command)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        def _publish(free: int | None, speed_now: float, route: str | None) -> None:
            payload = meter.snapshot(
                elapsed_s=done_s,
                disk_bytes=directory_bytes(out),
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
            payload["until_disk"] = bool(until_disk)
            if target_s == float("inf"):
                payload["target_s"] = None
            write_collect_status(status_root, payload)

        _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
        if status_root is not None:
            print(f"collect status={status_root / 'collect.json'}", flush=True)
        while done_s < target_s:
            if stop_file(out).is_file():
                save_progress(progress_path, done_s, target_s if target_s < float("inf") else done_s)
                _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
                print("collect stop requested", flush=True)
                return 0
            tick_mark = time.monotonic()
            elapsed = done_s
            speed = speed_at(elapsed, speeds)
            waypoint = waypoint_at(elapsed, names)
            if last_ai != (speed, waypoint):
                for command in ai_commands(speed, waypoint, traffic=bool(args.traffic)):
                    _queue(vehicle, command)
                last_ai = (speed, waypoint)
                print(f"collect ai speed={speed:.0f} m/s route={waypoint or 'span'}", flush=True)
            frames, stamps = _poll_cameras(cams)
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
            )
            if not queued:
                for cid, prev in held:
                    if prev is None:
                        last_cam.pop(cid, None)
                    else:
                        last_cam[cid] = prev
                print(f"collect drop {format_queue_line(writer.stats())}", flush=True)
            mask = tech.get("lane_gt_valid") or []
            try:
                gt_ok = float(sum(mask)) > 0.0
            except TypeError:
                gt_ok = False
            meter.note(set(frames), gt_ok)
            time.sleep(0.1)
            done_s += time.monotonic() - tick_mark
            now_m = time.monotonic()
            if now_m - last_disk >= DISK_CHECK_S:
                nbytes = directory_bytes(out)
                try:
                    free = check_data_path(out)
                except DataDriveError as exc:
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                    _publish(None, speed, waypoint)
                    print(format_disk_line(done_s, nbytes, ceiling_gb_per_hour=ceiling_gb), flush=True)
                    print(format_queue_line(writer.stats()), flush=True)
                    print(f"collect stop {exc}", flush=True)
                    return 3
                last_disk = now_m
                if now_m - last_print >= 60.0:
                    print(format_disk_line(done_s, nbytes, free, ceiling_gb_per_hour=ceiling_gb), flush=True)
                    print(format_queue_line(writer.stats()), flush=True)
                    save_progress(progress_path, done_s, done_s if target_s == float("inf") else target_s)
                    _publish(free, speed, waypoint)
                    last_print = now_m
        save_progress(progress_path, done_s, target_s if target_s < float("inf") else done_s)
        _publish(None, speed_at(done_s, speeds), waypoint_at(done_s, names))
        print(format_disk_line(done_s, directory_bytes(out), ceiling_gb_per_hour=ceiling_gb), flush=True)
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


def _poll_cameras(cams: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pictures as the sensor returned them, plus the wall time of that poll."""
    import numpy as np

    frames: dict[str, Any] = {}
    stamps: dict[str, Any] = {}
    for cid, cam in cams.items():
        try:
            payload = cam.poll()
        except Exception:
            continue
        image = _colour(payload)
        if not isinstance(image, np.ndarray) or image.size == 0:
            continue
        frames[cid] = image
        sensor_t = None
        if isinstance(payload, dict):
            for key in ("timestamp", "time", "t"):
                if payload.get(key) is not None:
                    sensor_t = payload.get(key)
                    break
        stamps[cid] = {"wall": time.time(), "sensor_t": sensor_t}
    return frames, stamps


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
