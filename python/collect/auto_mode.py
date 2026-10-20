"""Automatic Mode options shared by the CLI and the supervisor button.

The process that records is ``scripts/collect_scene_lanes.py``. Every tick
is ``driver=beamng_ai``. Output stays on the data disk.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from python.data.raw_frames import CAMERA_HZ
from python.debug.remote_status import COLLECT_NAME, read_json, status_dir

MAPS = ("", "west_coast_usa", "italy", "east_coast_usa", "utah", "jungle_rock_island")
# Levels that ship with BeamNG.tech 0.39. A multi-map run rotates these.
DEFAULT_MAPS = ("west_coast_usa", "italy", "east_coast_usa", "utah", "jungle_rock_island")
ROUTES = ("span", "waypoints", "random")
LANE_CAMERAS = "main,narrow,wide"
CAMERA_PRESETS = ("", LANE_CAMERAS)
WEATHERS = ("", "clear", "rain", "fog")


# Shown in the panel. The collector gets --speed-min/--speed-max only when
# the user set a speed (``speed_set``); otherwise the spawn-type ranges apply
# (city 8-14, highway 20-30, ramp 10-18).
SPEED_MIN_DEFAULT = 8.0
SPEED_MAX_DEFAULT = 17.0


@dataclass
class AutoOptions:
    map_name: str = ""
    route: str = "span"
    waypoints: str = ""
    speed_min: float = SPEED_MIN_DEFAULT
    speed_max: float = SPEED_MAX_DEFAULT
    minutes: float = 45.0
    until_disk: bool = False
    cameras: str = ""
    rates: str = ""
    traffic: bool = False
    time_of_day: str = ""
    weather: str = ""
    out: Path | None = None
    excursions: bool = True
    maps: str = ""
    segment_min: float = 12.0
    smoke_s: float = 60.0
    vary_env: bool = True
    speed_set: bool = False

    def __post_init__(self) -> None:
        # A speed given here (not the defaults) is the user's: pass it on.
        if float(self.speed_min) != SPEED_MIN_DEFAULT or float(self.speed_max) != SPEED_MAX_DEFAULT:
            self.speed_set = True

    def cycle(self, ident: str) -> None:
        if ident == "map":
            self.map_name = _next(MAPS, self.map_name)
        elif ident == "route":
            self.route = _next(ROUTES, self.route)
        elif ident == "cameras":
            self.cameras = _next(CAMERA_PRESETS, self.cameras)
        elif ident == "traffic":
            self.traffic = not self.traffic
        elif ident == "disk":
            self.until_disk = not self.until_disk
        elif ident == "weather":
            self.weather = _next(WEATHERS, self.weather)
        elif ident == "time":
            order = ("", "0.25", "0.5", "0.75")
            self.time_of_day = _next(order, self.time_of_day)
        elif ident == "excursion":
            self.excursions = not self.excursions
        elif ident == "maps":
            joined = ",".join(DEFAULT_MAPS)
            self.maps = "" if self.maps == joined else joined
        elif ident == "speed+":
            self.speed_max = min(30.0, float(self.speed_max) + 1.0)
            self.speed_set = True
        elif ident == "speed-":
            self.speed_max = max(float(self.speed_min), float(self.speed_max) - 1.0)
            self.speed_set = True
        elif ident == "minutes+":
            self.minutes = min(180.0, float(self.minutes) + 15.0)
        elif ident == "minutes-":
            self.minutes = max(0.0, float(self.minutes) - 15.0)


def _next(values: tuple[str, ...], current: str) -> str:
    items = list(values)
    try:
        i = items.index(str(current))
    except ValueError:
        i = -1
    return items[(i + 1) % len(items)]


def speeds_for(low: float, high: float) -> tuple[float, ...]:
    lo = float(min(low, high))
    hi = float(max(low, high))
    if hi - lo < 0.5:
        return (lo,)
    mid = (lo + hi) / 2.0
    return (lo, mid, hi, mid)


def rates_table(rates: str, cameras: str) -> dict[str, float]:
    """Hz per camera. A camera left out of ``cameras`` is off (0 Hz)."""
    table = {str(k): float(v) for k, v in CAMERA_HZ.items()}
    wanted = [part.strip() for part in str(cameras or "").split(",") if part.strip()]
    if wanted:
        keep = set(wanted)
        for cid in list(table):
            if cid not in keep:
                table[cid] = 0.0
    for part in str(rates or "").split(","):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            table[key] = max(0.0, float(raw))
        except ValueError:
            continue
    return table


def environment_commands(time_of_day: str, weather: str) -> list[str]:
    """Best-effort BeamNG environment lines. Empty when the option is blank."""
    commands: list[str] = []
    text = str(time_of_day or "").strip()
    if text:
        try:
            tod = float(text)
        except ValueError:
            tod = None
        if tod is not None:
            if tod > 1.0:
                tod = tod / 24.0
            tod = min(1.0, max(0.0, tod))
            commands.append(f"core_environment.setTimeOfDay({{time={tod:.4f}}})")
    sky = str(weather or "").strip().replace("\\", "").replace('"', "")
    if sky:
        commands.append(f'core_weather.setWeather("{sky}")')
    return commands


def stop_file(out: Path | str) -> Path:
    return Path(out) / "STOP"


def collect_argv(opts: AutoOptions, script: Path | str, python: str | None = None) -> list[str]:
    """Command line for the raw collector. Does not start it."""
    cmd = [python or sys.executable, str(script)]
    if opts.until_disk or float(opts.minutes) <= 0.0:
        cmd += ["--until-disk"]
    else:
        cmd += ["--minutes", str(float(opts.minutes))]
    if opts.map_name:
        cmd += ["--map", str(opts.map_name)]
    cmd += ["--route", str(opts.route or "span")]
    if opts.waypoints:
        cmd += ["--waypoints", str(opts.waypoints)]
    if opts.speed_set:
        # Only a speed the user set. Without one the spawn-type ranges apply.
        cmd += ["--speed-min", str(float(opts.speed_min)), "--speed-max", str(float(opts.speed_max))]
    if opts.cameras:
        cmd += ["--cameras", str(opts.cameras)]
    if opts.rates:
        cmd += ["--rates", str(opts.rates)]
    if opts.traffic:
        cmd.append("--traffic")
    if opts.time_of_day:
        cmd += ["--time-of-day", str(opts.time_of_day)]
    if opts.weather:
        cmd += ["--weather", str(opts.weather)]
    if not opts.excursions:
        cmd.append("--no-excursions")
    if opts.maps:
        cmd += ["--maps", str(opts.maps), "--segment-min", str(float(opts.segment_min))]
        if not opts.vary_env:
            cmd.append("--no-vary-env")
    if float(opts.smoke_s) > 0.0:
        cmd += ["--smoke-s", str(float(opts.smoke_s))]
    if opts.out is not None:
        cmd += ["--out", str(opts.out)]
    return cmd


def format_status(payload: dict[str, Any] | None, *, running: bool, error: str = "") -> list[str]:
    """Lines for the supervisor panel. Missing fields stay visible as na."""
    payload = payload or {}
    elapsed = float(payload.get("elapsed_s") or 0.0) / 60.0
    disk = float(payload.get("disk_bytes") or 0.0) / (1024.0 ** 3)
    free = payload.get("disk_free_bytes")
    free_s = "na" if free is None else f"{float(free) / (1024.0 ** 3):.1f} GB"
    ai = payload.get("ai") if isinstance(payload.get("ai"), dict) else {}
    word = "recording" if running else "stopped"
    lines = [
        f"Automatic Mode  {word}",
        f"minutes {elapsed:.1f}",
        f"frames {payload.get('frames', 0)}",
        f"disk {disk:.2f} GB",
        f"free {free_s}",
        f"AI {ai.get('mode') or 'na'} {ai.get('speed_mps') if ai.get('speed_mps') is not None else 'na'} m/s",
        "driver=beamng_ai",
    ]
    note = error or str(payload.get("error") or "")
    lines.append(f"error {note or 'none'}")
    return lines


class AutoSession:
    """One collector process. Stop writes STOP and ends the process."""

    def __init__(self, script: Path | str | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.script = Path(script) if script is not None else root / "scripts" / "collect_scene_lanes.py"
        self.options = AutoOptions()
        self.proc: subprocess.Popen[Any] | None = None
        self.error = ""

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, popen: Callable[..., Any] | None = None) -> bool:
        if self.running():
            self.error = "already running"
            return False
        out = self.options.out
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            flag = stop_file(out)
            if flag.is_file():
                flag.unlink()
        launcher = popen or subprocess.Popen
        try:
            self.proc = launcher(collect_argv(self.options, self.script))
        except OSError as exc:
            self.error = str(exc)
            self.proc = None
            return False
        self.error = ""
        return True

    def stop(self) -> None:
        out = self.options.out
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            stop_file(out).write_text("stop\n", encoding="utf-8")
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
        self.error = ""

    def status_lines(self) -> list[str]:
        folder = status_dir()
        payload = None if folder is None else read_json(folder / COLLECT_NAME)
        code = None if self.proc is None else self.proc.poll()
        extra = self.error
        if code not in (None, 0) and not extra:
            extra = f"exit {code}"
        return format_status(payload, running=self.running(), error=extra)
