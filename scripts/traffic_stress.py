#!/usr/bin/env python3
"""Step AI traffic counts and record what the collector can sustain.

For each count (default 0, 4, 8, 12, 16, 24, 32) the live mode holds
about 60 s and writes achieved recorder Hz, frame-drop fraction, tick
time p50/p95, nvidia-smi VRAM, and BeamNG / supervisor RSS. Escalation
stops when Hz drops under 9, drops exceed 1%, or VRAM headroom is under
1 GB. A JSON and Markdown table go under --out.

This script attaches to a BeamNG.tech already running on --host/--port.
It never launches or closes BeamNG. Do not run it from CI; use --stub
for the offline tests, and run live on the dev PC (Desktop-IQU45).

Example (dev PC, after BeamNG.tech is up and the usual collect attach works):

    py -3 scripts\\traffic_stress.py --hold-s 60 --maps west_coast_usa --out E:\\GVD_data\\traffic_stress
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

MappingLike = Mapping[str, Any]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_COUNTS = (0, 4, 8, 12, 16, 24, 32)
STOP_HZ_MIN = 9.0
STOP_DROP_MAX = 0.01
STOP_VRAM_HEADROOM_GB = 1.0
DEFAULT_VRAM_TOTAL_GB = 11.0  # GTX 1080 Ti, when nvidia-smi has no total


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile. ``None`` on an empty series."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (float(p) / 100.0) * (len(xs) - 1)
    i = int(math.floor(rank))
    j = min(i + 1, len(xs) - 1)
    frac = rank - i
    return xs[i] * (1.0 - frac) + xs[j] * frac


def should_stop(
    *,
    hz: float,
    drop_frac: float,
    vram_used_gb: float,
    vram_total_gb: float,
    hz_min: float = STOP_HZ_MIN,
    drop_max: float = STOP_DROP_MAX,
    headroom_gb: float = STOP_VRAM_HEADROOM_GB,
) -> str | None:
    """Why to stop escalating, or ``None`` if this step is still healthy."""
    if float(hz) < float(hz_min):
        return "hz"
    if float(drop_frac) > float(drop_max):
        return "drops"
    if float(vram_total_gb) - float(vram_used_gb) < float(headroom_gb):
        return "vram"
    return None


def _tick_percentiles(sample: MappingLike) -> tuple[float | None, float | None]:
    if "tick_p50_s" in sample or "tick_p95_s" in sample:
        return (
            None if sample.get("tick_p50_s") is None else float(sample["tick_p50_s"]),
            None if sample.get("tick_p95_s") is None else float(sample["tick_p95_s"]),
        )
    pair = sample.get("tick_s")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return float(pair[0]), float(pair[1])
    times = sample.get("tick_times") or sample.get("dts")
    if times:
        return percentile(list(times), 50), percentile(list(times), 95)
    return None, None



def _row_from_sample(count: int, hold_s: float, sample: MappingLike) -> dict[str, Any]:
    p50, p95 = _tick_percentiles(sample)
    return {
        "traffic_count": int(count),
        "hold_s": float(hold_s),
        "hz": float(sample.get("hz") or 0.0),
        "drop_frac": float(sample.get("drop_frac") or 0.0),
        "tick_p50_s": p50,
        "tick_p95_s": p95,
        "vram_used_gb": float(sample.get("vram_used_gb") or 0.0),
        "vram_total_gb": float(sample.get("vram_total_gb") or DEFAULT_VRAM_TOTAL_GB),
        "beamng_rss_mb": sample.get("beamng_rss_mb"),
        "supervisor_rss_mb": sample.get("supervisor_rss_mb"),
        "stop": None,
    }


def run_plan(
    counts: Iterable[int],
    measure: Callable[[int], MappingLike],
    hold_s: float = 60.0,
    *,
    hz_min: float = STOP_HZ_MIN,
    drop_max: float = STOP_DROP_MAX,
    headroom_gb: float = STOP_VRAM_HEADROOM_GB,
) -> list[dict[str, Any]]:
    """Measure each count until a stop rule fires. The failing step is kept."""
    rows: list[dict[str, Any]] = []
    for count in counts:
        sample = dict(measure(int(count)))
        row = _row_from_sample(int(count), hold_s, sample)
        reason = should_stop(
            hz=float(row["hz"]),
            drop_frac=float(row["drop_frac"]),
            vram_used_gb=float(row["vram_used_gb"]),
            vram_total_gb=float(row["vram_total_gb"]),
            hz_min=hz_min,
            drop_max=drop_max,
            headroom_gb=headroom_gb,
        )
        row["stop"] = reason
        rows.append(row)
        if reason is not None:
            break
    return rows


def write_report(rows: Sequence[MappingLike], json_path: Path, md_path: Path) -> None:
    """Write the step table as JSON and GitHub-flavored Markdown."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    stopped = next((row for row in rows if row.get("stop")), None)
    healthy = [int(row["traffic_count"]) for row in rows if not row.get("stop")]
    payload = {
        "hold_s": rows[0]["hold_s"] if rows else 60.0,
        "counts": [int(row["traffic_count"]) for row in rows],
        "max_safe_count": None if not healthy else int(healthy[-1]),
        "stopped_at": None if stopped is None else int(stopped["traffic_count"]),
        "stop": None if stopped is None else stopped["stop"],
        "rows": list(rows),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cols = (
        "traffic_count",
        "hz",
        "drop_frac",
        "tick_p50_s",
        "tick_p95_s",
        "vram_used_gb",
        "vram_total_gb",
        "beamng_rss_mb",
        "supervisor_rss_mb",
        "stop",
    )
    lines = [
        "# Traffic stress",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col)
            if val is None:
                cells.append("")
            elif isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    if stopped is not None:
        lines.append("")
        lines.append(
            f"Stopped escalating at {stopped['traffic_count']} cars "
            f"({stopped['stop']})."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nvidia_vram() -> tuple[float, float]:
    """``(used_gb, total_gb)``. Total falls back to 11 GB. Zeros if smi is missing."""
    used = 0.0
    total = DEFAULT_VRAM_TOTAL_GB
    try:
        from python.runtime.hw_probe import gpu_vram_used_gb

        used = float(gpu_vram_used_gb(force=True))
    except Exception:
        used = 0.0
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
        ).strip().splitlines()[0]
        tot = float(out)
        if tot > 64:
            tot = tot / 1024.0
        if tot > 0:
            total = tot
    except Exception:
        pass
    return used, total


def _process_rss_mb(match: Callable[[str], bool]) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    total = 0.0
    found = False
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = str(proc.info.get("name") or "")
            cmd = " ".join(str(p) for p in (proc.info.get("cmdline") or []))
            if match(name) or match(cmd):
                rss = proc.info.get("memory_info")
                if rss is not None:
                    total += float(rss.rss) / (1024.0 * 1024.0)
                    found = True
        except (psutil.Error, TypeError, ValueError):
            continue
    return total if found else None


def sample_rss() -> tuple[float | None, float | None]:
    """BeamNG.tech / BeamNG.drive RSS, and this process plus a supervisor if present."""
    beam = _process_rss_mb(
        lambda text: "beamng" in text.lower() or "BeamNG" in text
    )
    try:
        from python.sensors.frame_slot import process_rss_mb

        self_rss = float(process_rss_mb())
    except Exception:
        self_rss = 0.0
    other = _process_rss_mb(
        lambda text: "run_vision.py" in text or "gvd_status" in text or "play_gvd" in text
    )
    supervisor = self_rss + (other or 0.0)
    return beam, supervisor


def _parse_counts(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step AI traffic counts and record collector load (does not launch or close BeamNG)."
    )
    parser.add_argument("--hold-s", type=float, default=60.0, help="Seconds to record at each count. Default 60.")
    parser.add_argument(
        "--counts",
        default=",".join(str(n) for n in DEFAULT_COUNTS),
        help="Comma-separated AI car counts. Default 0,4,8,12,16,24,32.",
    )
    parser.add_argument("--maps", default="west_coast_usa", help="Passed through to the collector in live mode.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Folder for traffic_stress.json and traffic_stress.md.",
    )
    parser.add_argument("--hz-min", type=float, default=STOP_HZ_MIN)
    parser.add_argument("--drop-max", type=float, default=STOP_DROP_MAX)
    parser.add_argument("--vram-headroom-gb", type=float, default=STOP_VRAM_HEADROOM_GB)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Do not attach to BeamNG. Read --stub-metrics JSON (or a built-in sequence) and write the report.",
    )
    parser.add_argument(
        "--stub-metrics",
        type=Path,
        default=None,
        help="JSON object mapping traffic_count -> metric dict, for --stub.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Override GVD_DATA_DIR for live collector children. Refuses C: and D:.",
    )
    return parser


def _stub_measure(fixtures: dict[int, dict[str, Any]]) -> Callable[[int], dict[str, Any]]:
    def measure(count: int) -> dict[str, Any]:
        if count not in fixtures:
            raise KeyError(f"no stub metrics for traffic_count={count}")
        return dict(fixtures[count])

    return measure


def _default_stub_fixtures() -> dict[int, dict[str, Any]]:
    # Healthy enough to walk the whole default list in a dry CLI run.
    rows = {}
    for n in DEFAULT_COUNTS:
        rows[n] = dict(
            hz=12.0 - 0.05 * n,
            drop_frac=0.0,
            tick_s=(0.08 + 0.001 * n, 0.10 + 0.002 * n),
            vram_used_gb=6.0 + 0.05 * n,
            vram_total_gb=11.0,
            beamng_rss_mb=3500.0 + 10.0 * n,
            supervisor_rss_mb=800.0,
        )
    return rows


def live_measure_factory(args: argparse.Namespace) -> Callable[[int], dict[str, Any]]:
    """Build a measure() that runs the collector for --hold-s at each count.

    The child is ``scripts/collect_scene_lanes.py``. It attaches to the
    already-running Tech session and does not launch or close it. Each
    step uses a fresh --out subfolder so previous frames are not deleted.
    """
    import subprocess
    import time

    collect = ROOT / "scripts" / "collect_scene_lanes.py"
    out_root = Path(args.out)
    hold_s = max(60.0, float(args.hold_s))  # collector floors --minutes at 1
    minutes = hold_s / 60.0

    def measure(count: int) -> dict[str, Any]:
        step_dir = out_root / f"n{int(count):02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(collect),
            "--minutes",
            f"{minutes:g}",
            "--maps",
            str(args.maps),
            "--segment-min",
            f"{minutes:g}",
            "--traffic",
            str(int(count)),
            "--smoke-s",
            "0",
            "--no-excursions",
            "--out",
            str(step_dir),
        ]
        if args.host:
            cmd += ["--host", str(args.host)]
        if args.port is not None:
            cmd += ["--port", str(args.port)]
        if args.data_dir:
            cmd += ["--data-dir", str(args.data_dir)]
        print(f"traffic_stress step traffic={count} cmd={' '.join(cmd)}", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        held = max(1e-3, time.monotonic() - t0)
        if proc.returncode not in (0, 6, 7):
            # 6 is no_progress, 7 is file_cap; still parse whatever was written.
            print(f"traffic_stress collector exit {proc.returncode} at traffic={count}", flush=True)
        return _metrics_from_step(step_dir, held)

    return measure


def _metrics_from_step(step_dir: Path, held_s: float) -> dict[str, Any]:
    """Hz, drops, and tick percentiles from a collector output folder."""
    from python.collect.tick_log import TICKS_FILE
    from python.data.strip_writer import read_state_jsonl

    state_path = step_dir / "state.jsonl"
    rows = read_state_jsonl(state_path) if state_path.is_file() else []
    frames = len(rows)
    ticks_path = step_dir / TICKS_FILE
    tick_rows = read_state_jsonl(ticks_path) if ticks_path.is_file() else []
    dts = [float(r["dt"]) for r in tick_rows if r.get("dt") is not None]
    dropped = 0
    for row in tick_rows:
        if row.get("row") is False:
            dropped += 1
        # queue-full drops also show up as collect drop lines; ticks.note(row=False)
    total_ticks = len(tick_rows) or (frames + dropped)
    drop_frac = 0.0 if total_ticks <= 0 else dropped / float(total_ticks)
    hz = frames / float(held_s) if held_s > 0 else 0.0
    # Prefer the recorded span when state has timestamps (ignores process teardown).
    if len(rows) >= 2 and rows[0].get("t") is not None and rows[-1].get("t") is not None:
        span = float(rows[-1]["t"]) - float(rows[0]["t"])
        if span > 0.5:
            hz = (frames - 1) / span
    used, total = _nvidia_vram()
    beam, supervisor = sample_rss()
    p50, p95 = (percentile(dts, 50), percentile(dts, 95)) if dts else (None, None)
    return {
        "hz": hz,
        "drop_frac": drop_frac,
        "tick_p50_s": p50,
        "tick_p95_s": p95,
        "vram_used_gb": used,
        "vram_total_gb": total,
        "beamng_rss_mb": beam,
        "supervisor_rss_mb": supervisor,
        "frames": frames,
        "dropped": dropped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        counts = _parse_counts(args.counts)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if not counts or any(n < 0 for n in counts):
        parser.error("counts must be a comma list of integers >= 0")
        return 2
    out = args.out
    if out is None:
        out = Path.cwd() / "traffic_stress"
    args.out = Path(out)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stub:
        if args.stub_metrics is not None:
            raw = json.loads(Path(args.stub_metrics).read_text(encoding="utf-8"))
            fixtures = {int(k): dict(v) for k, v in raw.items()}
        else:
            fixtures = _default_stub_fixtures()
        measure = _stub_measure(fixtures)
        mode = "stub"
    else:
        print(
            "traffic_stress live: attaching via collect_scene_lanes; "
            "will not launch or close BeamNG",
            flush=True,
        )
        measure = live_measure_factory(args)
        mode = "live"
    rows = run_plan(
        counts,
        measure,
        hold_s=float(args.hold_s),
        hz_min=float(args.hz_min),
        drop_max=float(args.drop_max),
        headroom_gb=float(args.vram_headroom_gb),
    )
    write_report(rows, args.out / "traffic_stress.json", args.out / "traffic_stress.md")
    report = args.out / "traffic_stress.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "max_safe_count": payload.get("max_safe_count"),
                "stopped_at": payload.get("stopped_at"),
                "stop": payload.get("stop"),
            }
        ),
        flush=True,
    )
    stopped = next((row for row in rows if row.get("stop")), None)
    if stopped is None:
        print(
            f"traffic_stress {mode} finished {len(rows)} steps, no stop rule; "
            f"max_safe_count={payload.get('max_safe_count')} -> {args.out}",
            flush=True,
        )
    else:
        print(
            f"traffic_stress {mode} stopped at {stopped['traffic_count']} cars "
            f"({stopped['stop']}); max_safe_count={payload.get('max_safe_count')} -> {args.out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
