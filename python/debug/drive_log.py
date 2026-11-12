"""Per-tick drive log for a live scene-ckpt run.

``--drive-log`` writes one JSON object per supervisor tick. ``--drive-log-gt``
adds the painted-lane center at 8 m, from the same lane-target code the
trainer scores (``tech_from_painted`` then ``lane_x_from_vector``). That
number is stored on the row and nowhere else: it is not a steer input, a
pedal input, or a visualizer overlay.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

from python.debug.lane_targets import (
    ORIGIN_CENTRELINE,
    ORIGIN_REF_NODE,
    lane_center_is_valid,
    lane_x_from_vector,
    tech_from_painted,
)

# BeamNG electrics ``steering_input``. ±1 is full lock, ±30 deg at the road wheel.
APPLIED_STEER_UNIT = "steering_input (±1 = ±30 deg road wheel)"

# One row, every tick. Missing measurements are JSON null, not omitted.
ROW_FIELDS = (
    "wall",
    "frame_id",
    "frame_age_s",
    "lane_x",
    "lane_y",
    "steer_cmd",
    "steer_why",
    "throttle",
    "brake",
    "gear",
    "speed_mps",
    "yaw_rad",
    "pos",
    "applied_steer",
    "applied_steer_unit",
    "loop_hz",
    "camera_hz",
    "model_lane_x",
    "steer_law",
    "launch_yaw_rad",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _scalar(value: Any) -> Any:
    """A JSON number or string, or None. Bool is not a gear or a frame id."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _pos(value: Any) -> list[float] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    if len(value) < 2:
        return None
    out: list[float] = []
    for item in list(value)[:3]:
        num = _finite(item)
        if num is None:
            return None
        out.append(num)
    return out


def build_drive_row(
    obs: Mapping[str, Any] | None,
    *,
    include_gt: bool = False,
    gt_lane_x: float | None = None,
    gt_origin: str | None = None,
) -> dict[str, Any]:
    """One JSON-ready row. ``include_gt`` is the only way ``gt_lane_x`` appears."""
    src = obs or {}
    row: dict[str, Any] = {}
    for key in ROW_FIELDS:
        if key == "pos":
            row[key] = _pos(src.get(key))
        elif key in ("gear", "steer_why", "frame_id"):
            row[key] = _scalar(src.get(key))
        elif key == "applied_steer_unit":
            continue
        else:
            row[key] = _finite(src.get(key))
    applied = row.get("applied_steer")
    if applied is None:
        row["applied_steer_unit"] = None
    else:
        unit = src.get("applied_steer_unit")
        row["applied_steer_unit"] = str(unit) if isinstance(unit, str) and unit.strip() else APPLIED_STEER_UNIT
    if include_gt:
        row["gt_lane_x"] = _finite(gt_lane_x)
        row["gt_origin"] = str(gt_origin) if gt_origin else None
    return row


def lane_x_8m_from_paint(
    painted: Any,
    pos: Any,
    direction: Any,
    *,
    tracker: Any = None,
    origin_right_m: float = 0.0,
    origin: str = ORIGIN_REF_NODE,
    ref_right_of_origin_m: float = 0.0,
) -> float | None:
    """Training-label center at 8 m, or None when either paint line is missing.

    ``lane_at`` plus ``tech_from_painted`` plus ``lane_x_from_vector`` is the
    collector's label path. A slot with no paint is invalid, not a centered lane.
    """
    if painted is None or pos is None or direction is None:
        return None
    edges = painted.lane_at(
        pos,
        direction,
        tracker=tracker,
        origin_right_m=float(origin_right_m),
    )
    tech = tech_from_painted(
        edges,
        origin=origin,
        ref_right_of_origin_m=float(ref_right_of_origin_m),
    )
    if not lane_center_is_valid(tech.get("lane_gt_valid")):
        return None
    return lane_x_from_vector(tech.get("lane_gt"))


class PaintLaneGt:
    """Load one map's painted lines and read the 8 m center. Log only."""

    def __init__(self, origin: str | None = None, painted: Any = None) -> None:
        asked = str(origin or "").strip()
        self._centreline = ORIGIN_CENTRELINE in asked
        self._paint = painted
        self._tracker = None
        self._loaded = painted is not None
        self._level = ""
        self._origin_right_m = 0.0
        self._ref_right = 0.0
        self._offset_ready = not self._centreline
        self._prev: tuple[float, float] | None = None
        self._level_try = 0.0
        self._noted = False
        if painted is not None:
            from python.debug.painted_lanes import LaneTracker

            self._tracker = LaneTracker()

    def origin_label(self) -> str:
        if self._centreline and not self._offset_ready:
            return "body_centreline_unavailable"
        if self._centreline:
            return ORIGIN_CENTRELINE
        return ORIGIN_REF_NODE

    def load(self, bng: Any) -> None:
        """Read the map's painted lines once, before the drive loop.

        A research-socket call stays out of the tick so this flag cannot
        change the next command's ``dt``.
        """
        self._ensure(bng, socket_busy=False, vdata=None)

    def sample(self, vdata: Any, bng: Any = None, *, socket_busy: bool = False) -> tuple[float | None, str]:
        """``(gt_lane_x, gt_origin)``. A miss is ``(None, origin)``.

        ``bng`` and ``socket_busy`` are accepted so a caller can pass the
        live session, and ignored: paint was loaded by ``load``.
        """
        del bng, socket_busy
        self._resolve_offset(vdata)
        label = self.origin_label()
        if self._paint is None or vdata is None or (self._centreline and not self._offset_ready):
            return None, label
        pos = getattr(vdata, "pos", None)
        direction = getattr(vdata, "dir", None)
        if pos is None or direction is None:
            return None, label
        self._maybe_reset(pos)
        got = lane_x_8m_from_paint(
            self._paint,
            pos,
            direction,
            tracker=self._tracker,
            origin_right_m=float(self._origin_right_m),
            origin=ORIGIN_CENTRELINE if self._centreline else ORIGIN_REF_NODE,
            ref_right_of_origin_m=float(self._ref_right),
        )
        return got, label

    def _maybe_reset(self, pos: Any) -> None:
        if self._tracker is None:
            return
        try:
            here = (float(pos[0]), float(pos[1]))
        except (TypeError, ValueError, IndexError):
            return
        if self._prev is not None and math.hypot(here[0] - self._prev[0], here[1] - self._prev[1]) > 80.0:
            self._tracker.reset()
        self._prev = here

    def _resolve_offset(self, vdata: Any) -> None:
        if not self._centreline or self._offset_ready or vdata is None:
            return
        model = str(getattr(vdata, "model", "") or "").strip()
        if not model:
            return
        home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME") or ""
        zip_path = Path(home) / "content" / "vehicles" / f"{model}.zip" if home else None
        if zip_path is None or not zip_path.is_file():
            return
        try:
            from python.debug.vehicle_origin import ref_right_of_centre_m

            ref, _src = ref_right_of_centre_m(model, zip_path)
        except (OSError, ValueError, KeyError):
            return
        self._ref_right = float(ref)
        self._origin_right_m = -float(ref)
        self._offset_ready = True

    def _ensure(self, bng: Any, *, socket_busy: bool, vdata: Any) -> None:
        self._resolve_offset(vdata)
        if self._loaded or socket_busy or bng is None:
            return
        now = time.time()
        if self._level_try and (now - self._level_try) < 2.0:
            return
        self._level_try = now
        try:
            # current_level is re-exported by gt_logger. road_loader stays
            # imported only by gt_logger and level_session.
            from python.debug.gt_logger import current_level
            from python.debug.painted_lanes import LaneTracker, PaintedLanes, load_painted_decals

            level = current_level(bng) or ""
            if not level:
                return
            roads = None
            try:
                from python.data.data_root import roads_dir, use_data_disk

                if use_data_disk():
                    roads = roads_dir()
            except Exception:
                roads = None
            paint = PaintedLanes(load_painted_decals(level, roads))
        except Exception as exc:
            if not self._noted:
                print(f"[GVD] drive-log GT paint: {exc}", flush=True)
                self._noted = True
            return
        self._level = level
        self._loaded = True
        if not paint:
            if not self._noted:
                print(f"[GVD] drive-log GT: no painted lines for {level}", flush=True)
                self._noted = True
            return
        self._paint = paint
        self._tracker = LaneTracker()
        print(f"[GVD] drive-log GT paint: {level}", flush=True)


class DriveLog:
    """Append-only JSONL. Each ``write`` is flushed so a crash keeps the rows."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(f"{self.path} is a directory")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(dict(row), allow_nan=False, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def record_drive_tick(
    log: DriveLog,
    cmd: Any,
    obs: Mapping[str, Any] | None,
    *,
    include_gt: bool = False,
    gt_lane_x: float | None = None,
    gt_origin: str | None = None,
) -> dict[str, Any]:
    """Append one row. ``cmd`` is read to prove it was not rewritten, then left as it was."""
    before = (float(cmd.steer), float(cmd.throttle), float(cmd.brake), str(cmd.reason))
    row = build_drive_row(obs, include_gt=include_gt, gt_lane_x=gt_lane_x, gt_origin=gt_origin)
    log.write(row)
    after = (float(cmd.steer), float(cmd.throttle), float(cmd.brake), str(cmd.reason))
    if after != before:
        raise RuntimeError("drive log mutated the command")
    return row
