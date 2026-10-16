"""Files a remote shell can read. No window.

Everything here is written under the data disk, in ``status/``
(``E:\\GVD_data\\status`` on the driving PC). When that disk is not in
use the writers do nothing, so a Linux test does not create an ``E:``
folder in the repo.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

HEARTBEAT_S = 30.0
CHECKPOINT_S = 180.0
METRICS_EVERY = 10
SANITY_LANE_LOSS = 0.05
SANITY_STEPS = 200
DRY_RUN_STEPS = 2
STALE_MISS_TICKS = 5

COLLECT_NAME = "collect.json"
HEARTBEAT_NAME = "train_heartbeat.json"
METRICS_NAME = "metrics.jsonl"
CRASH_NAME = "train_crash.txt"
LIVE_NAME = "live.json"
VALIDATE_DIR = "validate"
EVAL_DIR_NAME = "eval"


def status_dir(explicit: Path | str | None = None) -> Path | None:
    """Directory for status files, or None when the data disk is off."""
    if explicit is not None:
        return Path(explicit)
    from python.data.data_root import data_root, use_data_disk

    if not use_data_disk():
        return None
    return data_root() / "status"


def eval_dir(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    from python.data.data_root import data_root, use_data_disk

    if not use_data_disk():
        return None
    return data_root() / EVAL_DIR_NAME


def write_json(path: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def append_jsonl(path: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return path


def read_json(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def tail_jsonl(path: Path | str | None, n: int = 5) -> list[dict[str, Any]]:
    if path is None:
        return []
    file = Path(path)
    if not file.is_file():
        return []
    lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, int(n)) :]:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            out.append(raw)
    return out


def json_num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def sanity_reached(lane_gt_loss: float | None, limit: float = SANITY_LANE_LOSS) -> bool:
    """True when the lane head has been driven near zero."""
    got = json_num(lane_gt_loss)
    if got is None:
        return False
    return got < float(limit)


def gpu_util_percent() -> float | None:
    """GPU utilization, or None when nvidia-smi is not there."""
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    return json_num(line[0].strip())


class CollectMeter:
    """Counts for the once-a-minute collection status file."""

    def __init__(self, cam_ids: tuple[str, ...]) -> None:
        self.cam_ids = tuple(cam_ids)
        self.frames = 0
        self.ticks = 0
        self.gt_valid = 0
        self.dropped = {cid: 0 for cid in self.cam_ids}
        self._streak = {cid: 0 for cid in self.cam_ids}

    def note(self, present: set[str] | Mapping[str, Any], gt_valid: bool) -> None:
        self.ticks += 1
        self.frames += 1
        if gt_valid:
            self.gt_valid += 1
        have = set(present)
        for cid in self.cam_ids:
            if cid in have:
                self._streak[cid] = 0
            else:
                self.dropped[cid] += 1
                self._streak[cid] += 1

    def stale(self) -> list[str]:
        return [cid for cid, n in self._streak.items() if n >= STALE_MISS_TICKS]

    def snapshot(
        self,
        *,
        elapsed_s: float,
        disk_bytes: int,
        free_bytes: int | None,
        ai_speed_mps: float | None,
        ai_mode: str,
        ai_waypoint: str | None,
        out: str,
        target_s: float,
    ) -> dict[str, Any]:
        elapsed = max(float(elapsed_s), 1e-6)
        ticks = max(self.ticks, 1)
        return {
            "kind": "collect",
            "updated": time.time(),
            "out": out,
            "elapsed_s": float(elapsed_s),
            "target_s": float(target_s),
            "frames": int(self.frames),
            "hz": float(self.frames) / elapsed,
            "dropped_cams": dict(self.dropped),
            "stale_cams": self.stale(),
            "gt_valid": int(self.gt_valid),
            "gt_ticks": int(self.ticks),
            "gt_valid_frac": float(self.gt_valid) / float(ticks),
            "disk_bytes": int(disk_bytes),
            "disk_free_bytes": None if free_bytes is None else int(free_bytes),
            "ai": {
                "speed_mps": json_num(ai_speed_mps),
                "mode": str(ai_mode),
                "waypoint": ai_waypoint,
            },
        }


def write_collect_status(directory: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if directory is None:
        return None
    return write_json(Path(directory) / COLLECT_NAME, payload)


def write_heartbeat(directory: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if directory is None:
        return None
    return write_json(Path(directory) / HEARTBEAT_NAME, payload)


def append_metrics(directory: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if directory is None:
        return None
    return append_jsonl(Path(directory) / METRICS_NAME, payload)


def write_crash(directory: Path | None, traceback_text: str, batch: Mapping[str, Any]) -> Path | None:
    if directory is None:
        return None
    path = Path(directory) / CRASH_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    body = traceback_text.rstrip() + "\n---\n" + json.dumps(dict(batch), indent=2, sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8")
    return path


_live_wall = 0.0


def note_live(payload: Mapping[str, Any], *, min_s: float = 1.0) -> Path | None:
    """Latest live lane numbers. Rewritten about once a second."""
    global _live_wall
    directory = status_dir()
    if directory is None:
        return None
    now = time.monotonic()
    if _live_wall and (now - _live_wall) < float(min_s):
        return None
    _live_wall = now
    row = dict(payload)
    row["kind"] = "live"
    row["updated"] = time.time()
    return write_json(directory / LIVE_NAME, row)
