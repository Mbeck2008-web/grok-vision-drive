#!/usr/bin/env python3
"""Headless lane-keep check against a running BeamNG.tech.

Attaches with launch=False. Does not start BeamNG and does not stop it.
Disconnect uses the research socket only (quit_on_close stays false).

Loads west_coast_usa, spawns an etk800 at the BeamNGpy highway example
pose, starts the supervisor with ground-truth logging, engages with
extensions.gvd_main.engage(), waits, disengages, and runs the report.
Exit 1 when the engaged in-lane percentage is below --min-in-lane.
Exit 2 when Tech or the log never showed up.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# BeamNGpy's west_coast_usa example: highway, etk800, heading along the road.
DEFAULT_POS = (-717.121, 101.0, 118.675)
DEFAULT_ROT = (0.0, 0.0, 0.3826834, 0.9238795)


def _lua(bng, command: str) -> None:
    for owner in (bng, getattr(bng, "control", None)):
        if owner is None:
            continue
        fn = getattr(owner, "queue_lua_command", None)
        if not callable(fn):
            continue
        try:
            fn(command)
            return
        except TypeError:
            fn(command, False)
            return
    raise RuntimeError("BeamNGpy has no queue_lua_command")


def _load_level(bng) -> None:
    from beamngpy import Scenario, Vehicle  # type: ignore

    scenario = Scenario("west_coast_usa", "gvd_gt_lane", description="GVD lane ground truth")
    vehicle = Vehicle("gvd_ego", model="etk800")
    scenario.add_vehicle(vehicle, pos=DEFAULT_POS, rot_quat=DEFAULT_ROT)
    scenario.make(bng)
    load = getattr(getattr(bng, "scenario", None), "load", None)
    start = getattr(getattr(bng, "scenario", None), "start", None)
    if callable(load) and callable(start):
        load(scenario)
        start()
        return
    bng.load_scenario(scenario)
    bng.start_scenario()


def _log_says_crash(path: Path) -> bool:
    """True when the logger already recorded a damage jump or a stuck car."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    prev = None
    for line in lines[-40:]:
        line = line.strip()
        if not line:
            continue
        try:
            import json
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        delta = row.get("damage_delta")
        if isinstance(delta, (int, float)) and float(delta) >= 400.0:
            return True
        damage = row.get("damage")
        if prev is not None and isinstance(damage, (int, float)) and damage - prev >= 400.0:
            return True
        if isinstance(damage, (int, float)):
            prev = float(damage)
    return False


def _newest_log(since: float) -> Path | None:
    from python.data.data_root import logs_dir

    folder = logs_dir()
    if not folder.is_dir():
        return None
    found = [p for p in folder.glob("gt_*.jsonl") if p.stat().st_mtime >= since - 1.0]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Live lane-keep check on a running BeamNG.tech")
    ap.add_argument("--seconds", type=float, default=30.0, help="Engaged seconds")
    ap.add_argument("--min-in-lane", type=float, default=80.0, help="Required engaged in-lane percent")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--lane-source",
        default=os.environ.get("GVD_LANE_SOURCE", "hough"),
        choices=["hough", "scene", "gt"],
        help="gt feeds the road-network center at 8 m as lane_x.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from python.debug.gt_summary import (
        distance_failure,
        format_report,
        load_jsonl,
        render_plot,
        summarize,
    )
    from python.sensors.tech import (
        load_tech_config,
        open_tech_beamngpy,
        release_tech_beamngpy,
        widen_socket_timeout,
    )

    cfg = load_tech_config()
    host = args.host or str(cfg.get("host") or "localhost")
    port = int(args.port if args.port is not None else cfg.get("port") or 25252)
    try:
        bng = open_tech_beamngpy(
            host, port, home=cfg.get("home") or None, user=cfg.get("user") or None, launch=False,
        )
    except Exception as exc:
        print(f"[GVD] live lane test: Tech attach failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("[GVD] BeamNG.tech was not launched and was not stopped.", file=sys.stderr)
        return 2

    sup: subprocess.Popen | None = None
    started = time.time()
    try:
        try:
            # Hello stays short. Loading west_coast_usa takes about 30 s and
            # the 10 s socket timeout aborted it.
            widen_socket_timeout(bng, 90.0)
            _load_level(bng)
        except Exception as exc:
            print(f"[GVD] live lane test: scenario load failed ({type(exc).__name__}: {exc})", file=sys.stderr)
            return 2
        time.sleep(2.0)
        started = time.time()
        env = os.environ.copy()
        env["GVD_GT_LOG"] = "1"
        env["GVD_DATA_DIR"] = os.environ.get("GVD_DATA_DIR", r"E:\GVD_data")
        env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(Path(env["GVD_DATA_DIR"]) / "tmp")
        env["TORCH_HOME"] = str(Path(env["GVD_DATA_DIR"]) / "cache" / "torch")
        env["GVD_TECH_LAUNCH"] = "0"
        env["GVD_BEAMNG"] = "1"
        env["GVD_BACKEND"] = "beamngpy"
        env["GVD_LANE_SOURCE"] = str(args.lane_source)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        sup = subprocess.Popen(
            [
                sys.executable, "-u", str(ROOT / "python" / "run_vision.py"),
                "--backend", "beamngpy", "--gt-log", "--hz", "10", "--encode", "cpu",
                "--lane-source", str(args.lane_source),
            ],
            cwd=str(ROOT),
            env=env,
        )
        log = None
        deadline = time.time() + 45.0
        while time.time() < deadline:
            log = _newest_log(started)
            if log is not None and log.stat().st_size > 0:
                break
            if sup.poll() is not None:
                print(f"[GVD] supervisor exited {sup.returncode} before a ground-truth log", file=sys.stderr)
                return 2
            time.sleep(0.5)
        if log is None:
            print("[GVD] no gt_*.jsonl under the data logs folder after 45 s", file=sys.stderr)
            return 2
        _lua(bng, "extensions.gvd_main.engage()")
        print(f"[GVD] engaged for {args.seconds:.0f}s  log={log}", flush=True)
        end = time.time() + max(0.0, float(args.seconds))
        while time.time() < end:
            time.sleep(0.5)
            if _log_says_crash(log):
                print("[GVD] crash or stuck in the ground-truth log; stopping early", flush=True)
                break
        _lua(bng, "extensions.gvd_main.disengage()")
        time.sleep(0.5)
    finally:
        if sup is not None and sup.poll() is None:
            sup.terminate()
            try:
                sup.wait(timeout=8)
            except subprocess.TimeoutExpired:
                sup.kill()
        release_tech_beamngpy(bng)

    log = _newest_log(started)
    if log is None:
        print("[GVD] ground-truth log disappeared", file=sys.stderr)
        return 2
    rows = load_jsonl(log)
    summary = summarize(rows)
    print(format_report(summary, source=str(log)))
    png = log.with_suffix(".png")
    try:
        render_plot(rows, png)
        print(f"plot={png}")
    except Exception as exc:
        print(f"plot=failed {type(exc).__name__}: {exc}", file=sys.stderr)
    failed = False
    reason = distance_failure(summary)
    if reason is not None:
        print(f"[GVD] FAIL {reason}", file=sys.stderr)
        failed = True
    pct = summary.get("in_lane_pct")
    if pct is None or float(pct) < float(args.min_in_lane):
        print(f"[GVD] FAIL in_lane_pct={pct} need>={args.min_in_lane}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
