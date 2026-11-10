"""Offline training-time frame quality: wrong-way and overexposed-road drops.

Wrong-way uses recorded ego pose plus a saved road dump. The legal
tangent follows node order, reversed when ``flipDirection`` is true; it
is never flipped just to match the car. Overexposure uses the lower half
of each selected training-camera sector (any RGB channel >= 250). A row
is dropped when the maximum of those per-cam fractions exceeds T. Missing
cameras are omitted, so black pads on the full 2272×192 canvas do not
dilute the fraction. Neither signal is a model input; deploy stays
camera-only. See ``docs/SPEC_frame_quality_filter.md``.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

SHOULDER_M = 1.5
CENTER_DEADZONE_M = 0.75
ALONG_DEG = 60.0
ALONG_COS = math.cos(math.radians(ALONG_DEG))
CELL_M = 40.0
# Discard a snap onto another deck (bridge / overpass). About one storey.
MAX_DZ_M = 3.0
CLIP_VALUE = 250
# Parser default for --max-road-clip. Pass -1 to disable.
DEFAULT_MAX_ROAD_CLIP = 0.20
# Per-cam lower-half clip of the selected training cameras; drop on the max.
CLIP_AGGREGATE = "max"
ROAD_CLIP_METRIC = "road_clip_v1"
ROAD_CLIP_SIDECAR = "road_clip_v1.jsonl"


class FrameQualityError(ValueError):
    """A road dump needed by the wrong-way filter is missing or unreadable."""


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
    flip: bool = False
    z: list[float | None] | None = None


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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "0").strip().lower() in {"1", "true", "yes", "on"}


def _oneway(road: Mapping[str, Any]) -> bool:
    raw = road.get("oneWay")
    if raw is None:
        raw = road.get("oneway")
    return _truthy(raw)


def _flip_direction(road: Mapping[str, Any]) -> bool:
    raw = road.get("flipDirection")
    if raw is None:
        raw = road.get("flip_direction")
    return _truthy(raw)


def _mid_z(raw: Any) -> float | None:
    if not (isinstance(raw, (list, tuple)) and len(raw) >= 3):
        return None
    try:
        value = float(raw[2])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _middles_from_edges(edges: Any) -> tuple[list[tuple[float, float]], list[float], list[float | None]]:
    xy: list[tuple[float, float]] = []
    widths: list[float] = []
    zs: list[float | None] = []
    if not isinstance(edges, list):
        return xy, widths, zs
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        mid = edge.get("middle") or edge.get("mid") or edge.get("center")
        left = edge.get("left")
        right = edge.get("right")
        mx = my = None
        mz = _mid_z(mid)
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
            lz, rz = _mid_z(left), _mid_z(right)
            if mz is None and lz is not None and rz is not None:
                mz = 0.5 * (lz + rz)
        if mx is None or my is None:
            continue
        xy.append((mx, my))
        zs.append(mz)
        if (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) >= 2
            and len(right) >= 2
        ):
            span = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
            if span >= 0.5:
                widths.append(span)
    return xy, widths, zs


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
        xy, widths, zs = _middles_from_edges(road.get("edges") or road.get("nodes"))
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
                flip=_flip_direction(road),
                z=None if all(value is None for value in zs) else zs,
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


def _project_stored(
    road: QualityRoad,
    x: float,
    y: float,
    z: float | None = None,
) -> dict[str, Any] | None:
    best: tuple[float, int, float, float, float, float, float, float] | None = None
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
            best = (dist, i, t, cx, cy, dx, dy, length2)
    if best is None:
        return None
    dist, i, t, cx, cy, dx, dy, length2 = best
    road_z = None
    if road.z is not None and i < len(road.z):
        z0 = road.z[i]
        z1 = road.z[i + 1] if i + 1 < len(road.z) else None
        if z0 is not None and z1 is not None:
            road_z = z0 + t * (z1 - z0)
        elif z0 is not None:
            road_z = z0
        elif z1 is not None:
            road_z = z1
    dz = None
    if z is not None and road_z is not None:
        dz = float(z) - float(road_z)
        if abs(dz) > MAX_DZ_M:
            return None
    flen = math.sqrt(length2) or 1.0
    fx, fy = dx / flen, dy / flen
    # Node order is the geometric tangent. flipDirection reverses legal travel
    # without rewriting the stored edges.
    if road.flip:
        fx, fy = -fx, -fy
    rx, ry = fy, -fx
    offset = (x - cx) * rx + (y - cy) * ry
    dist3 = dist if dz is None else math.hypot(dist, dz)
    return {
        "dist_m": dist,
        "dist3_m": dist3,
        "offset_m": offset,
        "tx": fx,
        "ty": fy,
        "dz_m": dz,
        "road": road,
    }


def pose_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float | None] | None:
    """World ``(x, y, heading_rad, z_or_none)`` from a ``state.jsonl`` row."""
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
    z = None
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        try:
            z_val = float(pos[2])
        except (TypeError, ValueError):
            z_val = None
        else:
            z = z_val if math.isfinite(z_val) else None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
        return None
    return x, y, heading, z


def classify_wrong_way(
    x: float,
    y: float,
    heading_rad: float,
    roads: list[QualityRoad],
    grid: dict[tuple[int, int], list[QualityRoad]] | None = None,
    z: float | None = None,
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
        hit = _project_stored(road, x, y, z)
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
    hits.sort(key=lambda item: (item["dist3_m"], -item["along_abs"]))
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
        "flip": road.flip,
        "dz_m": best.get("dz_m"),
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
    return classify_wrong_way(pose[0], pose[1], pose[2], roads, grid, z=pose[3])


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


def driving_clip_cams(cams: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Selected cameras that the clip metric may use (STRIP_ORDER only)."""
    from python.data.strip_writer import STRIP_ORDER

    if not cams:
        return tuple(STRIP_ORDER)
    wanted = {str(name) for name in cams}
    return tuple(name for name in STRIP_ORDER if name in wanted)


def row_clip_source(row: Mapping[str, Any]) -> str:
    """Cache / sidecar key: multi-root ``source``, else resolved ``source_root``."""
    if row.get("source") not in (None, ""):
        return str(row["source"])
    root = row.get("source_root")
    if root in (None, ""):
        return ""
    return str(Path(root).resolve())


def _finite_frac(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def load_clip_sidecar(path: Path | str) -> dict[tuple[str, int], dict[str, float]]:
    """Load per-cam fractions keyed by (source, i). Wrong-version lines are skipped."""
    file = Path(path)
    out: dict[tuple[str, int], dict[str, float]] = {}
    if not file.is_file():
        return out
    with file.open("rt", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or str(payload.get("v") or "") != ROAD_CLIP_METRIC:
                continue
            try:
                index = int(payload.get("i"))
            except (TypeError, ValueError):
                continue
            fracs = payload.get("frac")
            if not isinstance(fracs, dict):
                continue
            clean = {str(cam): float(val) for cam, val in fracs.items() if _finite_frac(val)}
            if not clean:
                continue
            key = (str(payload.get("source") or ""), index)
            cur = dict(out.get(key) or {})
            cur.update(clean)
            out[key] = cur
    return out


def save_clip_sidecar(path: Path | str, store: Mapping[tuple[str, int], Mapping[str, float]]) -> None:
    """Rewrite a sidecar atomically. Keys are (source, i)."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    with tmp.open("wt", encoding="utf-8") as handle:
        for (source, index), fracs in sorted(store.items(), key=lambda item: (item[0][0], item[0][1])):
            payload = {
                "v": ROAD_CLIP_METRIC,
                "source": source,
                "i": int(index),
                "frac": {str(cam): float(val) for cam, val in fracs.items()},
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, file)


def _cache_sector(cache: Any, cam: str, index: int, row: Mapping[str, Any]) -> Any:
    if cache is None or not hasattr(cache, "sector"):
        return None
    source_key = row.get("source")
    kwargs = {"source": source_key} if source_key is not None else {}
    try:
        return cache.sector(cam, int(index), **kwargs)
    except TypeError:
        return cache.sector(cam, int(index))
    except Exception:
        return None


def _measure_cam_clip(
    row: Mapping[str, Any],
    cam: str,
    index: int,
    *,
    cache: Any = None,
    cam_loader: Callable[[Any, int, str], Any] | None = None,
) -> tuple[float | None, str]:
    """Return (fraction, origin) for one driving camera. origin is cache/raw/miss."""
    image = _cache_sector(cache, cam, index, row)
    origin = "cache"
    if image is None:
        origin = "raw"
        root = row.get("source_root")
        if cam_loader is not None:
            try:
                image = cam_loader(root, index, cam)
            except Exception:
                image = None
        else:
            if root in (None, ""):
                return None, "miss"
            from python.data.raw_frames import load_raw_camera

            try:
                image = load_raw_camera(root, cam, index)
            except Exception:
                image = None
    if image is None:
        return None, "miss"
    return road_clip_fraction(image), origin


def drop_overexposed_rows(
    rows: list[Mapping[str, Any]],
    max_road_clip: float,
    *,
    cams: tuple[str, ...] | list[str] | None = None,
    cache: Any = None,
    cam_loader: Callable[[Any, int, str], Any] | None = None,
    clip_store: dict[tuple[str, int], dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop rows whose max per-cam lower-half clip (selected cams) exceeds T.

    Only cameras in ``STRIP_ORDER`` that were selected (``--cams``) and actually
    present are measured. Absent cameras are omitted, so a washed main sector
    still trips T=0.20 when the other sectors are black or missing. The
    aggregate is the maximum of those per-cam fractions (not a canvas-wide
    mean, and not main-only unless ``--cams main``).
    """
    selected = driving_clip_cams(cams)
    kept: list[dict[str, Any]] = []
    over = 0
    skipped = 0
    decoded = 0
    cache_hits = 0
    sidecar_hits = 0
    store = clip_store if clip_store is not None else {}
    for row in rows:
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            skipped += 1
            kept.append(dict(row))
            continue
        source = row_clip_source(row)
        key = (source, index)
        known = dict(store.get(key) or {})
        need = [cam for cam in selected if cam not in known]
        if selected and not need:
            sidecar_hits += 1
        new_fracs: dict[str, float] = {}
        for cam in need:
            frac, origin = _measure_cam_clip(
                row, cam, index, cache=cache, cam_loader=cam_loader
            )
            if frac is None:
                continue
            new_fracs[cam] = float(frac)
            if origin == "raw":
                decoded += 1
            elif origin == "cache":
                cache_hits += 1
        if new_fracs:
            known.update(new_fracs)
            store[key] = known
        values = [known[cam] for cam in selected if cam in known]
        if not values:
            skipped += 1
            kept.append(dict(row))
            continue
        if max(values) > float(max_road_clip):
            over += 1
            continue
        kept.append(dict(row))
    return kept, {
        "overexposed": over,
        "clip_skipped": skipped,
        "clip_decoded": decoded,
        "clip_cache_hits": cache_hits,
        "clip_sidecar_hits": sidecar_hits,
    }


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


def _warn_quality(message: str, warn: Callable[[str], None] | None = None) -> None:
    if warn is not None:
        warn(message)
        return
    print(f"[GVD] frame-quality: {message}", flush=True)


def apply_frame_quality(
    rows: list[Mapping[str, Any]],
    *,
    drop_wrong_way: bool = False,
    max_road_clip: float = -1.0,
    road_dump: Path | str | None = None,
    roads: list[QualityRoad] | None = None,
    image_loader: Callable[[Any, int], Any] | None = None,
    require_road_dump: bool = False,
    warn: Callable[[str], None] | None = None,
    cams: tuple[str, ...] | list[str] | None = None,
    cache: Any = None,
    cam_loader: Callable[[Any, int, str], Any] | None = None,
    clip_sidecar: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the enabled filters. Both off returns copies of the same rows.

    When ``drop_wrong_way`` is set and no road dump can be loaded, raise
    ``FrameQualityError`` if ``require_road_dump`` is true (the user passed
    ``--drop-wrong-way``). Otherwise log one warning and keep wrong-way
    frames as unknown; the road-clip filter still runs when T >= 0.

    Clip is measured per selected training camera (never ``main_depth`` /
    ``main_sem``, never the full compose_strip canvas). Values are persisted
    in ``clip_sidecar`` (``road_clip_v1``, keyed by source+id) so a later
    build does not re-decode. When a frame cache is passed, sectors are
    read from it instead of raw PNG files.
    """
    stats = {
        "wrong_way": 0,
        "ok": 0,
        "unknown": 0,
        "overexposed": 0,
        "clip_skipped": 0,
        "dump_missing": 0,
        "clip_decoded": 0,
        "clip_cache_hits": 0,
        "clip_sidecar_hits": 0,
    }
    out = [dict(row) for row in rows]
    if drop_wrong_way:
        use = roads
        if use is None:
            try:
                use = load_quality_roads(resolve_road_dump(out, road_dump))
            except (FrameQualityError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                if require_road_dump:
                    if isinstance(exc, FrameQualityError):
                        raise
                    raise FrameQualityError(f"road dump unreadable: {exc}") from exc
                stats["dump_missing"] = 1
                stats["unknown"] = len(out)
                err = exc if isinstance(exc, FrameQualityError) else f"road dump unreadable: {exc}"
                if float(max_road_clip) >= 0.0:
                    msg = (
                        f"no road dump for this map; keeping wrong-way frames as unknown. "
                        f"The road-clip filter still runs (T={float(max_road_clip):g}). "
                        f"Pass --drop-wrong-way to fail, or --road-dump PATH. ({err})"
                    )
                else:
                    msg = (
                        f"no road dump for this map; keeping wrong-way frames as unknown. "
                        f"Pass --drop-wrong-way to fail, or --road-dump PATH. ({err})"
                    )
                _warn_quality(msg, warn)
                use = None
        if use is not None:
            out, ww = drop_wrong_way_rows(out, use)
            stats.update(ww)
    if float(max_road_clip) >= 0.0:
        store: dict[tuple[str, int], dict[str, float]] = {}
        side = Path(clip_sidecar) if clip_sidecar else None
        if side is not None:
            store.update(load_clip_sidecar(side))
        # image_loader (the old compose_strip canvas path) is ignored. Clip is
        # per selected camera via cam_loader, the frame cache, or load_raw_camera.
        out, clip_stats = drop_overexposed_rows(
            out,
            float(max_road_clip),
            cams=cams,
            cache=cache,
            cam_loader=cam_loader,
            clip_store=store,
        )
        stats.update(clip_stats)
        if side is not None:
            save_clip_sidecar(side, store)
    return out, stats


def describe_quality(stats: Mapping[str, int], before: int) -> str:
    after = before - int(stats.get("wrong_way", 0)) - int(stats.get("overexposed", 0))
    missing = " dump_missing=1 (wrong-way kept)" if int(stats.get("dump_missing", 0)) else ""
    decoded = ""
    if any(int(stats.get(key, 0)) for key in ("clip_decoded", "clip_cache_hits", "clip_sidecar_hits")):
        decoded = (
            f" clip_decoded={int(stats.get('clip_decoded', 0))}"
            f" clip_cache={int(stats.get('clip_cache_hits', 0))}"
            f" clip_sidecar={int(stats.get('clip_sidecar_hits', 0))}"
        )
    return (
        f"before={int(before)} wrong_way={int(stats.get('wrong_way', 0))} "
        f"overexposed={int(stats.get('overexposed', 0))} kept={after} "
        f"ok={int(stats.get('ok', 0))} unknown={int(stats.get('unknown', 0))}{missing}{decoded} "
        f"(filtered-run metrics are not comparable to an unfiltered run)"
    )
