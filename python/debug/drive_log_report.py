"""Score a drive log: model 8 m lane_x against logged GT, and opposite-sign steer.

``python -m python.debug.drive_log_report <jsonl>``

GT is whatever ``--drive-log-gt`` stored. This tool does not read BeamNG and
does not change a command.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(raw, dict):
            rows.append(raw)
        else:
            bad += 1
    return rows, bad


def _f(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def model_lane_x(row: dict[str, Any]) -> float | None:
    """The model's 8 m center. The path sample is the fallback when the net field is absent."""
    got = _f(row, "model_lane_x")
    if got is not None:
        return got
    return _f(row, "lane_x")


def signs_oppose(left: float | None, right: float | None) -> bool:
    """True when both are nonzero and their signs differ. Zero has no sign to oppose."""
    if left is None or right is None or left == 0.0 or right == 0.0:
        return False
    return (left > 0.0) != (right > 0.0)


def same_sign(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == 0.0 and right == 0.0
    return (left > 0.0) == (right > 0.0)


def _num(value: float | None) -> str:
    if value is None:
        return "na"
    return f"{value:+.3f}"


def format_report(rows: list[dict[str, Any]]) -> str:
    errors: list[float] = []
    agree = 0
    opposite: list[dict[str, Any]] = []
    for row in rows:
        pred = model_lane_x(row)
        gt = _f(row, "gt_lane_x")
        if pred is not None and gt is not None:
            errors.append(pred - gt)
            if same_sign(pred, gt):
                agree += 1
        if signs_oppose(_f(row, "steer_cmd"), _f(row, "lane_x")):
            opposite.append(row)
    lines = [f"ticks {len(rows)}", f"scored {len(errors)}"]
    if errors:
        mean = sum(errors) / len(errors)
        rms = math.sqrt(sum(err * err for err in errors) / len(errors))
        pct = 100.0 * agree / len(errors)
        lines.append(f"model lane_x - gt_lane_x: mean {mean:+.3f} m, rms {rms:.3f} m")
        lines.append(f"sign agreement {pct:.1f}% ({agree}/{len(errors)})")
    else:
        lines.append("model lane_x - gt_lane_x: mean n/a, rms n/a")
        lines.append("sign agreement n/a (0/0)")
    lines.append(f"opposite-sign ticks: {len(opposite)}")
    for row in opposite:
        lane = _f(row, "lane_x")
        law = _f(row, "steer_law")
        note = ""
        if law is not None and lane is not None and not signs_oppose(law, lane):
            note = " post_law"
        yaw = _f(row, "launch_yaw_rad")
        yaw_s = "na" if yaw is None else f"{yaw:+.3f}"
        lines.append(
            f"  frame_id={row.get('frame_id')} wall={row.get('wall')} "
            f"lane_x={_num(lane)} steer_cmd={_num(_f(row, 'steer_cmd'))} "
            f"steer_why={row.get('steer_why')} steer_law={_num(law)} "
            f"launch_yaw_rad={yaw_s}{note}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        print("usage: python -m python.debug.drive_log_report <jsonl>", file=sys.stderr)
        return 2 if len(args) != 1 else 0
    path = Path(args[0])
    if not path.is_file():
        print(f"drive log not found: {path}", file=sys.stderr)
        return 2
    rows, bad = load_jsonl(path)
    if bad:
        print(f"skipped {bad} bad lines", file=sys.stderr)
    sys.stdout.write(format_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
