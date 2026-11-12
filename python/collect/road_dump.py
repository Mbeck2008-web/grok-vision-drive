"""Save and reload a live Tech road network.

Replays that rebuild the road from the driven path have no side roads, so
they open weaves the live graph rejects. The dump is the raw
``get_road_network`` payload plus the spawn, one gzip per map.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from python.debug.gt_geom import RoadPoly
from python.debug.gt_logger import polys_from_network


def json_safe(value: Any) -> Any:
    """Lists and dicts only. BeamNG tuples and numpy scalars become those."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "tolist", None)
    if callable(item):
        return json_safe(item())
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number


def write_road_dump(
    path: Path | str,
    *,
    level: str,
    spawn: tuple[float, float, float] | None,
    spawn_source: str,
    road_network: Any,
    roads: Any = None,
    note: str = "",
) -> Path:
    """Write ``E:\\GVD_data\\roads\\<map>.json.gz`` (or the path given)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map": str(level),
        "spawn": None
        if spawn is None
        else {"pos": [float(v) for v in spawn[:3]], "source": str(spawn_source or "")},
        "note": str(note or ""),
        "road_network": json_safe(road_network),
        "roads": json_safe(roads),
    }
    with gzip.open(dest, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return dest


def level_file_name(level: str) -> str:
    """A single path segment. An empty or hostile name becomes ``current``."""
    name = str(level or "").strip() or "current"
    if name in {".", ".."} or any(ch in name for ch in ("/", "\\", "\0")):
        return "current"
    return name


def _xyz(x: float, y: float, z: float | None) -> list[float]:
    if z is None:
        return [float(x), float(y)]
    return [float(x), float(y), float(z)]


def _height_at(values: list[float | None] | None, index: int) -> float | None:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def network_from_polys(polys: list[RoadPoly]) -> dict[str, Any]:
    """A ``get_road_network`` shaped dict. Edge z is kept; a missing height stays off the point.

    Left and right are offset in x by half the width. ``parse_road_edges`` reads
    their z, not that lateral offset, and the width is the left-right span.
    """
    spec: dict[str, Any] = {}
    for poly in polys:
        if len(poly.xy) < 2:
            continue
        half = float(poly.width_m) * 0.5
        edges = []
        for index, (x, y) in enumerate(poly.xy):
            edges.append(
                {
                    "middle": _xyz(x, y, _height_at(poly.z, index)),
                    "left": _xyz(x - half, y, _height_at(poly.left_z, index)),
                    "right": _xyz(x + half, y, _height_at(poly.right_z, index)),
                }
            )
        spec[str(poly.road_id)] = {"drivability": 1, "edges": edges}
    return spec


def persist_recording_roads(
    root: Path | str,
    level: str,
    bng: Any = None,
    polys: list[RoadPoly] | None = None,
) -> Path | None:
    """Write ``<recording>/roads/<level>.json.gz`` once per map.

    The cached ``get_road_network`` payload is the rendered DecalRoad edges
    (left/middle/right xyz). A session that only has parsed polys is written
    from those, so ``left_z`` / ``right_z`` still round-trip. Nothing here
    runs per tick, and a failed read does not raise.
    """
    spec = None
    note = ""
    if bng is not None:
        try:
            # gt_logger is the allowed road_loader reader. This file must not import it.
            from python.debug.gt_logger import polys_from_network, raw_road_network

            got, _legacy, got_note = raw_road_network(bng)
        except Exception:
            got, got_note, polys_from_network = None, "", None
        if got is not None and polys_from_network is not None and polys_from_network(got):
            spec = got
            note = str(got_note or "")
    if spec is None and polys:
        spec = network_from_polys(list(polys))
    if spec is None:
        return None
    name = level_file_name(level)
    dest = Path(root) / "roads" / f"{name}.json.gz"
    return write_road_dump(
        dest,
        level=name,
        spawn=None,
        spawn_source="",
        road_network=spec,
        note=note,
    )


def persist_painted_decals(root: Path | str, level: str) -> Path | None:
    """Copy lane-line decals next to the recording. Stop bars are that paint, turned across the lane.

    Missing zip and missing cache return None. The collect keeps going.
    """
    from python.data.data_root import roads_dir, use_data_disk
    from python.debug.painted_lanes import load_painted_decals

    name = level_file_name(level)
    try:
        decals = load_painted_decals(name, roads_dir() if use_data_disk() else None)
    except Exception:
        return None
    if not decals:
        return None
    dest = Path(root) / "roads" / f"{name}_lines.json.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as handle:
        json.dump(decals, handle)
    return dest


def _road_dict(row: dict[str, Any]) -> dict[str, Any]:
    road = row.get("road")
    if isinstance(road, dict):
        return road
    tags = row.get("tags")
    if isinstance(tags, dict) and isinstance(tags.get("road"), dict):
        return tags["road"]
    return {}


def _safe_relative(rel: str) -> bool:
    path = Path(rel)
    return bool(rel) and not rel.startswith(("/", "\\")) and ".." not in path.parts


def road_gt_for_row(root: Path | str, row: dict[str, Any]) -> RoadPoly | None:
    """This frame's road, from ``road.edge_z_file`` or ``roads/<map>.json.gz``.

    A missing road id is no road (the depth fit runs). The id is looked up
    only in that one file. Another map's dump is not searched, and there is
    no ``roads[0]`` fallback. ``edge_z_file: null`` (the dump was not written)
    is no road even when ``map`` is set.
    """
    base = Path(root)
    road = _road_dict(row)
    raw_id = road.get("road_id", row.get("road_id"))
    if raw_id is None or not str(raw_id).strip():
        return None
    wanted = str(raw_id)
    if "edge_z_file" in road:
        rel = road.get("edge_z_file")
        if not isinstance(rel, str) or not _safe_relative(rel.strip()):
            return None
        path = base / rel.strip()
    else:
        level = row.get("map")
        if not isinstance(level, str) or not level.strip():
            return None
        path = base / "roads" / f"{level_file_name(level)}.json.gz"
    if not path.is_file():
        return None
    polys, _payload = load_road_dump(path)
    for poly in polys:
        if poly.road_id == wanted:
            return poly
    return None


def recorded_edge_z_file(root: Path | str, relative: str | None) -> str | None:
    """The relative dump path when that file is on disk, else None."""
    if not isinstance(relative, str) or not _safe_relative(relative.strip()):
        return None
    rel = relative.strip()
    if (Path(root) / rel).is_file():
        return rel
    return None


def load_road_dump(path: Path | str) -> tuple[list[RoadPoly], dict[str, Any]]:
    """Parsed centerlines and the raw payload."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return [], {}
    spec = payload.get("road_network")
    polys = polys_from_network(spec)
    if not polys and isinstance(payload.get("roads"), dict):
        # A get_roads-only dump has no edges. There is nothing to drive on.
        polys = []
    return polys, payload
