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
