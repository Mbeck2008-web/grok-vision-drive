"""Offline training-time frame quality: wrong-way and overexposed-road drops.

Wrong-way uses recorded ego pose plus a saved road dump (stored tangent,
never flipped). Overexposure uses the lower half of the training image
(any RGB channel >= 250). Neither signal is a model input; deploy stays
camera-only. See ``docs/SPEC_frame_quality_filter.md``.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

SHOULDER_M = 1.5
CENTER_DEADZONE_M = 0.75
ALONG_DEG = 60.0
ALONG_COS = math.cos(math.radians(ALONG_DEG))
CELL_M = 40.0
CLIP_VALUE = 250
# Suggested cut when the flag is turned on. Parser default stays -1 (off).
DEFAULT_MAX_ROAD_CLIP = 0.20


class FrameQualityError(ValueError):
    """A road dump needed by ``--drop-wrong-way`` is missing or unreadable."""


@dataclass
class QualityRoad:
    road_id: str
    xy: list[tuple[float, float]]
    width_m: float
    oneway: bool
    lanes_left: int
    lanes_right: int
    lanes: int
    drivability: float


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _oneway(road: Mapping[str, Any]) -> bool:
    raw = road.get("oneWay")
    if raw is None:
        raw = road.get("oneway")
    if isinstance(raw, bool):
        return raw
    return str(raw or "0").strip().lower() in {"1", "true", "yes", "on"}


def _middles_from_edges(edges: Any) -> tuple[list[tuple[float, float]], list[float]]:
    xy: list[tuple[float, float]] = []
    widths: list[float] = []
    if not isinstance(edges, list):
        return xy, widths
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        mid = edge.get("middle") or edge.get("mid") or edge.get("center")
        left = edge.get("left")
        right = edge.get("right")
        mx = my = None
        if isinstance(mid, (list, tuple)) and len(mid) >= 2:
            mx, my = float(mid[0]), float(mid[1])
        elif (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) >= 2
            and len(right) >= 2
        ):
            mx = 0.5 * (float(left[0]) + float(right[0]))
            my = 0.5 * (float(left[1]) + float(right[1]))
        if mx is None or my is None:
            continue
        xy.append((mx, my))
        if (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) >= 2
            and len(right) >= 2
        ):
            span = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
            if span >= 0.5:
                widths.append(span)
    return xy, widths


def load_quality_roads(path: Path | str) -> list[QualityRoad]:
    """Centerlines and one-way flags from a ``road_dump.write_road_dump`` file."""
    file = Path(path)
    if not file.is_file():
        raise FrameQualityError(f"road dump not found: {file}")
    opener: Callable
    if file.suffix == ".gz" or file.name.endswith(".json.gz"):
        opener = gzip.open
    else:
        opener = open
    with opener(file, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    network = payload.get("road_network") if isinstance(payload, dict) else None
    if isinstance(network, dict) and "roads" in network and isinstance(network["roads"], dict):
        network = network["roads"]
    if not isinstance(network, dict):
        return []
    out: list[QualityRoad] = []
    for rid, road in network.items():
        if not isinstance(road, dict):
            continue
        driv = _num(road.get("drivability", road.get("drivable", 1)), 1.0)
        if driv <= 0.0:
            continue
        xy, widths = _middles_from_edges(road.get("edges") or road.get("nodes"))
        if len(xy) < 2:
            continue
        if widths:
            widths.sort()
            width = widths[len(widths) // 2]
        else:
            width = 7.0
        if width < 2.0:
            width = 3.5
        left = _int(road.get("lanesLeft"))
        right = _int(road.get("lanesRight"))
        out.append(
            QualityRoad(
                road_id=str(rid),
                xy=xy,
                width_m=float(width),
                oneway=_oneway(road),
                lanes_left=left,
                lanes_right=right,
                lanes=left + right,
                drivability=driv,
            )
        )
    return out


def build_road_grid(roads: Iterable[QualityRoad]) -> dict[tuple[int, int], list[QualityRoad]]:
    grid: dict[tuple[int, int], list[QualityRoad]] = {}
    for road in roads:
        placed: set[tuple[int, int]] = set()

        def add(px: float, py: float, current: QualityRoad = road) -> None:
            ix, iy = int(math.floor(px / CELL_M)), int(math.floor(py / CELL_M))
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    slot = (ix + ox, iy + oy)
                    if slot in placed:
                        continue
                    placed.add(slot)
                    grid.setdefault(slot, []).append(current)

        for px, py in road.xy:
            add(px, py)
        for (x0, y0), (x1, y1) in zip(road.xy, road.xy[1:]):
            span = math.hypot(x1 - x0, y1 - y0)
            if span <= CELL_M:
                continue
            steps = int(span // CELL_M)
            for step in range(1, steps + 1):
                t = step / (steps + 1)
                add(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return grid


def _project_stored(road: QualityRoad, x: float, y: float) -> dict[str, Any] | None:
    best: tuple[float, int, float, float, float, float, float] | None = None
    pts = road.xy
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        t = ((x - ax) * dx + (y - ay) * dy) / length2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        cx, cy = ax + t * dx, ay + t * dy
        dist = math.hypot(x - cx, y - cy)
        if best is None or dist < best[0]:
            best = (dist, i, cx, cy, dx, dy, length2)
    if best is None:
        return None
    dist, _i, cx, cy, dx, dy, length2 = best
    flen = math.sqrt(length2) or 1.0
    fx, fy = dx / flen, dy / flen
    rx, ry = fy, -fx
    offset = (x - cx) * rx + (y - cy) * ry
    return {"dist_m": dist, "offset_m": offset, "tx": fx, "ty": fy, "road": road}


def pose_from_row(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """World ``(x, y, heading_rad)`` from a ``state.jsonl`` row. None if missing."""
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    pos = ego.get("pos") if isinstance(ego, Mapping) else None
    if pos is None:
        pos = row.get("pos")
    direc = None
    if isinstance(ego, Mapping):
        direc = ego.get("dir") or ego.get("forward")
    if direc is None:
        direc = row.get("dir") or row.get("forward")
    heading = None
    if isinstance(ego, Mapping) and ego.get("heading_rad") is not None:
        heading = ego.get("heading_rad")
    if heading is None:
        heading = row.get("heading_rad")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if heading is None:
        if not (isinstance(direc, (list, tuple)) and len(direc) >= 2):
            return None
        try:
            heading = math.atan2(float(direc[1]), float(direc[0]))
        except (TypeError, ValueError):
            return None
    try:
        heading = float(heading)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
        return None
    return x, y, heading


def classify_wrong_way(
    x: float,
    y: float,
    heading_rad: float,
    roads: list[QualityRoad],
    grid: dict[tuple[int, int], list[QualityRoad]] | None = None,
) -> dict[str, Any]:
    """``verdict`` is ``wrong_way``, ``ok``, or ``unknown`` (see the spec)."""
    if not roads:
        return {"verdict": "unknown", "reason": "off_network"}
    if grid is None:
        grid = build_road_grid(roads)
    vx, vy = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    key = (int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M)))
    hits: list[dict[str, Any]] = []
    for road in grid.get(key, ()):
        hit = _project_stored(road, x, y)
        if hit is None:
            continue
        if hit["dist_m"] > road.width_m * 0.5 + SHOULDER_M:
            continue
        align = hit["tx"] * vx + hit["ty"] * vy
        hit = dict(hit)
        hit["align"] = align
        hit["along_abs"] = abs(align)
        hits.append(hit)
    if not hits:
        return {"verdict": "unknown", "reason": "off_network"}
    hits.sort(key=lambda item: (item["dist_m"], -item["along_abs"]))
    best = hits[0]
    road = best["road"]
    align = float(best["align"])
    offset = float(best["offset_m"])
    effective_oneway = bool(road.oneway) or (road.lanes_left == 0 and road.lanes_right >= 1)
    out: dict[str, Any] = {
        "verdict": "unknown",
        "reason": "",
        "road_id": road.road_id,
        "oneway": road.oneway,
        "effective_oneway": effective_oneway,
        "offset_m": offset,
        "align": align,
        "dist_m": best["dist_m"],
        "width_m": road.width_m,
    }
    if effective_oneway:
        if align < -ALONG_COS:
            out["verdict"] = "wrong_way"
            out["reason"] = "oneway_against"
        elif align > ALONG_COS:
            out["verdict"] = "ok"
            out["reason"] = "oneway_with"
        else:
            out["reason"] = "oneway_oblique"
        return out
    if abs(align) < ALONG_COS:
        out["reason"] = "twoway_oblique"
        return out
    if abs(offset) < CENTER_DEADZONE_M:
        out["reason"] = "twoway_centerline"
        return out
    if offset * align < 0.0:
        out["verdict"] = "wrong_way"
        out["reason"] = "twoway_opposing_lane"
    else:
        out["verdict"] = "ok"
        out["reason"] = "twoway_correct_side"
    return out


def classify_row(
    row: Mapping[str, Any],
    roads: list[QualityRoad],
    grid: dict[tuple[int, int], list[QualityRoad]] | None = None,
) -> dict[str, Any]:
    pose = pose_from_row(row)
    if pose is None:
        return {"verdict": "unknown", "reason": "no_pose"}
    return classify_wrong_way(pose[0], pose[1], pose[2], roads, grid)


def drop_wrong_way_rows(
    rows: list[Mapping[str, Any]],
    roads: list[QualityRoad],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep rows that are not ``wrong_way``. ``unknown`` and ``ok`` stay."""
    grid = build_road_grid(roads)
    kept: list[dict[str, Any]] = []
    stats = {"wrong_way": 0, "ok": 0, "unknown": 0}
    for row in rows:
        got = classify_row(row, roads, grid)
        verdict = str(got.get("verdict") or "unknown")
        stats[verdict] = int(stats.get(verdict, 0)) + 1
        if verdict == "wrong_way":
            continue
        kept.append(dict(row))
    return kept, stats


def road_clip_fraction(bgr: Any, clip_value: int = CLIP_VALUE) -> float:
    """Share of lower-half pixels with any RGB channel ``>= clip_value``."""
    frame = np.asarray(bgr)
    if frame.ndim != 3 or frame.shape[0] < 2 or frame.size == 0:
        return 0.0
    lower = frame[frame.shape[0] // 2 :]
    if lower.size == 0:
        return 0.0
    return float(np.any(lower >= int(clip_value), axis=-1).mean())


def drop_overexposed_rows(
    rows: list[Mapping[str, Any]],
    max_road_clip: float,
    image_loader: Callable[[Any, int], Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop rows whose lower-half clip fraction exceeds ``max_road_clip``."""
    kept: list[dict[str, Any]] = []
    over = 0
    skipped = 0
    for row in rows:
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            skipped += 1
            kept.append(dict(row))
            continue
        root = row.get("source_root")
        try:
            image = image_loader(root, index)
            frac = road_clip_fraction(image)
        except Exception:
            skipped += 1
            kept.append(dict(row))
            continue
        if frac > float(max_road_clip):
            over += 1
            continue
        kept.append(dict(row))
    return kept, {"overexposed": over, "clip_skipped": skipped}


def resolve_road_dump(
    rows: list[Mapping[str, Any]],
    road_dump: Path | str | None = None,
) -> Path:
    """Explicit path, or ``roads_dir() / <map>.json.gz`` from the rows."""
    if road_dump:
        path = Path(road_dump)
        if not path.is_file():
            raise FrameQualityError(f"--road-dump not found: {path}")
        return path
    maps = sorted({str(row.get("map") or "") for row in rows if str(row.get("map") or "")})
    if not maps:
        raise FrameQualityError(
            "--drop-wrong-way needs a road dump: pass --road-dump or record a map name on each row"
        )
    if len(maps) > 1:
        raise FrameQualityError(
            f"--drop-wrong-way found multiple maps {maps}; pass --road-dump for the network to use"
        )
    from python.data.data_root import roads_dir

    path = roads_dir() / f"{maps[0]}.json.gz"
    if not path.is_file():
        raise FrameQualityError(
            f"--drop-wrong-way needs {path} (or pass --road-dump); no dump was found for map {maps[0]!r}"
        )
    return path


def apply_frame_quality(
    rows: list[Mapping[str, Any]],
    *,
    drop_wrong_way: bool = False,
    max_road_clip: float = -1.0,
    road_dump: Path | str | None = None,
    roads: list[QualityRoad] | None = None,
    image_loader: Callable[[Any, int], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the enabled filters. Both off returns copies of the same rows."""
    stats = {"wrong_way": 0, "ok": 0, "unknown": 0, "overexposed": 0, "clip_skipped": 0}
    out = [dict(row) for row in rows]
    if drop_wrong_way:
        use = roads
        if use is None:
            use = load_quality_roads(resolve_road_dump(out, road_dump))
        out, ww = drop_wrong_way_rows(out, use)
        stats.update(ww)
    if float(max_road_clip) >= 0.0:
        if image_loader is None:
            from python.train.scene_loader import load_canvas

            def image_loader(root: Any, index: int) -> Any:
                if root in (None, ""):
                    raise FrameQualityError("overexposure drop needs a recording root on the row")
                return load_canvas(root, index)

        out, clip_stats = drop_overexposed_rows(out, float(max_road_clip), image_loader)
        stats.update(clip_stats)
    return out, stats


def describe_quality(stats: Mapping[str, int], before: int) -> str:
    after = before - int(stats.get("wrong_way", 0)) - int(stats.get("overexposed", 0))
    return (
        f"before={int(before)} wrong_way={int(stats.get('wrong_way', 0))} "
        f"overexposed={int(stats.get('overexposed', 0))} kept={after} "
        f"ok={int(stats.get('ok', 0))} unknown={int(stats.get('unknown', 0))}"
    )
