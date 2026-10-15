#!/usr/bin/env python3
"""One screen of whatever is running, or whatever ran last.

    py -3 scripts\\gvd_status.py

Reads ``E:\\GVD_data\\status`` and ``E:\\GVD_data\\eval``. Prints training
progress, the latest metrics line, a crash file if one exists, collection,
the live lane line, and free disk. Exit 0 even when nothing has run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.data_root import data_root, free_bytes, use_data_disk  # noqa: E402
from python.debug.remote_status import (  # noqa: E402
    COLLECT_NAME,
    CRASH_NAME,
    EVAL_DIR_NAME,
    HEARTBEAT_NAME,
    LIVE_NAME,
    METRICS_NAME,
    VALIDATE_DIR,
    read_json,
    status_dir,
    tail_jsonl,
)


def _gb(nbytes: int | None) -> str:
    if nbytes is None:
        return "na"
    return f"{nbytes / (1024.0 ** 3):.1f}"


def _line(name: str, text: str) -> str:
    return f"{name}: {text}"


def summarize(directory: Path | None, eval_directory: Path | None) -> list[str]:
    lines: list[str] = []
    if directory is None:
        lines.append("data_dir: off (set GVD_DATA_DIR to an absolute path, or run on Windows)")
        lines.append("nothing has run")
        return lines
    root = directory.parent
    lines.append(f"data_dir: {root}")
    try:
        lines.append(f"free_gb: {_gb(free_bytes(root))}")
    except OSError as exc:
        lines.append(f"free_gb: unreadable ({exc})")
    collect = read_json(directory / COLLECT_NAME)
    if collect is None:
        lines.append(_line("collect", "none"))
    else:
        lines.append(
            _line(
                "collect",
                f"frames={collect.get('frames')} hz={collect.get('hz')} "
                f"gt_valid_frac={collect.get('gt_valid_frac')} "
                f"stale={collect.get('stale_cams')} "
                f"free_gb={_gb(collect.get('disk_free_bytes'))} "
                f"elapsed_s={collect.get('elapsed_s')} ai={collect.get('ai')}",
            )
        )
    heart = read_json(directory / HEARTBEAT_NAME)
    if heart is None:
        lines.append(_line("train", "none"))
    else:
        lines.append(
            _line(
                "train",
                f"step={heart.get('step')} epoch={heart.get('epoch')} "
                f"loss={heart.get('loss')} eta_s={heart.get('eta_s')} "
                f"lr={heart.get('lr')} device={heart.get('device')}",
            )
        )
    metrics = tail_jsonl(directory / METRICS_NAME, 1)
    if not metrics:
        lines.append(_line("metrics", "none"))
    else:
        last = metrics[-1]
        lines.append(
            _line(
                "metrics",
                f"kind={last.get('kind')} step={last.get('step')} loss={last.get('loss')} "
                f"heads={last.get('heads')} grad_norm={last.get('grad_norm')} "
                f"nan={last.get('nan')} inf={last.get('inf')} "
                f"r={last.get('r')} rms={last.get('rms')}",
            )
        )
    crash = directory / CRASH_NAME
    if crash.is_file():
        text = crash.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        head = text[0] if text else "empty"
        lines.append(_line("crash", f"{crash} {head}"))
    else:
        lines.append(_line("crash", "none"))
    live = read_json(directory / LIVE_NAME)
    if live is None:
        lines.append(_line("live", "none"))
    else:
        lines.append(
            _line(
                "live",
                f"lane_x={live.get('lane_x')} model_lane_x={live.get('model_lane_x')} "
                f"model_conf={live.get('model_conf')} gt_center_x={live.get('gt_center_x')} "
                f"src={live.get('lane_src')}",
            )
        )
    report = read_json(directory / VALIDATE_DIR / "report.json")
    if report is None:
        lines.append(_line("validate", "none"))
    else:
        lines.append(
            _line(
                "validate",
                f"rows={report.get('rows')} jpegs={report.get('jpegs')} issues={report.get('issue_count')}",
            )
        )
    summary = None if eval_directory is None else read_json(eval_directory / "summary.json")
    if summary is None:
        lines.append(_line("eval", "none"))
    else:
        lines.append(
            _line(
                "eval",
                f"n={summary.get('n')} r={summary.get('r')} rms={summary.get('rms')} bias={summary.get('bias')}",
            )
        )
    if (
        collect is None
        and heart is None
        and not metrics
        and not crash.is_file()
        and live is None
        and report is None
        and summary is None
    ):
        lines.append("nothing has run")
    return lines


def main(argv: list[str] | None = None) -> int:
    del argv
    directory = status_dir()
    if directory is None and use_data_disk():
        directory = data_root() / "status"
    eval_directory = None if directory is None else directory.parent / EVAL_DIR_NAME
    for line in summarize(directory, eval_directory):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
