"""Training-only vehicle boxes from traffic rows.

The recorder stores each companion's world ``pos`` / ``vel`` / ``dir`` and,
when Electrics answered, that vehicle's ``brake_lights``. This module turns
one row into ego-frame object labels. It does not draw, plan, or export.

A null traffic sample scores nothing: there is no interpolation, and
``unseen_s`` is not a reason to invent a box. Length and width are left
unset. Brake lights are written beside the box. ``brake_lights: null`` is
what the recorder stores when Electrics cannot attach to a Lua-spawned
``gvd_traffic_*`` scenetree object. That null is NaN and its mask stays 0,
including when the IQU45 flag is on, so it is not "lights off" and it cannot
enter a loss. A real 0 is the only off. The head stays shut until that poll
(``BRAKE_LIGHT_HEAD``).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from python.collect.signal_labels import world_to_ego
from python.train.scene_net import N_OBJECTS, OBJECT_CLASSES, OBJECT_FEAT, labels_from_tech

# Off until a live poll on IQU45 shows brake_lights on a traffic vehicle.
# The objects head does not grow a channel for this, and it is not an ONNX output.
BRAKE_LIGHT_HEAD = False

_ABSENT = object()
_CAR_MODELS = frozenset({"etk800"})
_BRAKE_KEYS = ("brake_lights", "brakelights")


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _traffic_field(row: Mapping[str, Any]) -> Any:
    if "traffic" in row:
        return row.get("traffic")
    tags = row.get("tags")
    if isinstance(tags, Mapping) and "traffic" in tags:
        return tags.get("traffic")
    return _ABSENT


def _ego_pose(row: Mapping[str, Any]) -> tuple[Any, Any]:
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else None
    if isinstance(ego, Mapping):
        return ego.get("pos"), ego.get("dir")
    return row.get("pos"), row.get("dir")


def object_class_name(obj: Mapping[str, Any]) -> str:
    """Class index name. A missing model is ``vehicle``, never truck/bus/bike.

    ``etk800`` and a ``gvd_traffic_*`` name with no model are the spawned
    companions, which are cars. Any other model string is not a guess.
    """
    for key in ("cls", "class"):
        raw = obj.get(key)
        if isinstance(raw, str):
            name = raw.strip().lower().replace("-", "_").replace(" ", "_")
            if name in OBJECT_CLASSES:
                return name
    model = obj.get("model")
    if isinstance(model, str) and model.strip():
        name = model.strip().lower()
        if name in OBJECT_CLASSES:
            return name
        if name in _CAR_MODELS:
            return "car"
        return "vehicle"
    ident = str(obj.get("name") or "")
    if ident.startswith("gvd_traffic"):
        return "car"
    return "vehicle"


def _yaw(direction: Any, ego_pos: Any, ego_dir: Any) -> float | None:
    """Traffic ``dir`` in the ego frame. ``atan2(forward, right)`` is in (-pi, pi]."""
    delta = _pair(direction)
    base = _pair(ego_pos)
    if delta is None or base is None:
        return None
    origin = world_to_ego(base, ego_pos, ego_dir)
    tip = world_to_ego((base[0] + delta[0], base[1] + delta[1]), ego_pos, ego_dir)
    if origin is None or tip is None:
        return None
    right = tip[0] - origin[0]
    forward = tip[1] - origin[1]
    if math.hypot(right, forward) < 1e-8:
        return None
    return math.atan2(forward, right)


def _speed(vel: Any) -> float | None:
    """Horizontal speed. Vertical velocity is not part of the label."""
    flat = _pair(vel)
    if flat is None:
        return None
    return math.hypot(flat[0], flat[1])


def _brake(obj: Mapping[str, Any]) -> tuple[float, bool]:
    """(value, present). Null is masked, not off.

    PR1 writes ``brake_lights: null`` when Electrics fails on a scenetree
    ``gvd_traffic_*``. That is NaN with present False. A finite 0 is off.
    """
    sources: list[Mapping[str, Any]] = [obj]
    nested = obj.get("veh")
    if isinstance(nested, Mapping):
        sources.append(nested)
    for source in sources:
        for key in _BRAKE_KEYS:
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, bool):
                return (1.0 if value else 0.0), True
            if value is None:
                return math.nan, False
            try:
                out = float(value)
            except (TypeError, ValueError):
                return math.nan, False
            if not math.isfinite(out):
                return math.nan, False
            return out, True
    return 0.0, False


def _empty() -> dict[str, np.ndarray]:
    return {
        "objects": np.zeros((N_OBJECTS, OBJECT_FEAT), dtype=np.float32),
        "object_class": np.zeros((N_OBJECTS,), dtype=np.int64),
        "object_valid": np.zeros((N_OBJECTS,), dtype=np.float32),
        "brake_lights": np.zeros((N_OBJECTS,), dtype=np.float32),
        "brake_lights_valid": np.zeros((N_OBJECTS,), dtype=np.float32),
    }


def _boxes(row: Mapping[str, Any], traffic: list[Any]) -> list[dict[str, Any]]:
    ego_pos, ego_dir = _ego_pose(row)
    boxes: list[dict[str, Any]] = []
    for item in traffic:
        if len(boxes) >= N_OBJECTS:
            break
        if not isinstance(item, dict):
            continue
        pos = _pair(item.get("pos"))
        if pos is None:
            continue
        xy = world_to_ego(pos, ego_pos, ego_dir)
        yaw = _yaw(item.get("dir"), ego_pos, ego_dir)
        speed = _speed(item.get("vel"))
        if xy is None or yaw is None or speed is None:
            continue
        lamp, seen = _brake(item)
        # No length/width: the 4.5 m x 1.8 m fill is not a measurement.
        boxes.append(
            {
                "cls": object_class_name(item),
                "x": xy[0],
                "y": xy[1],
                "yaw": yaw,
                "speed_mps": speed,
                "partial": 0.0,
                "unseen_s": 0.0,
                "_brake": lamp,
                "_brake_seen": seen,
            }
        )
    return boxes


def traffic_object_labels(
    row: Mapping[str, Any] | None,
    *,
    brake_light_head: bool | None = None,
) -> dict[str, np.ndarray]:
    """Object tensors for one state row.

    No ``traffic`` key leaves the dict empty so painted tracks stay as they
    are. ``traffic: null`` (the off-ticks of the 2 Hz probe) scores nothing.
    Brake-light values are stored either way. A null field is NaN.
    ``brake_lights_valid`` is 1 only for a finite field when the head flag
    is on. Null stays 0, so no brake-light loss can use it.
    """
    row = row or {}
    field = _traffic_field(row)
    if field is _ABSENT:
        return {}
    if not isinstance(field, list):
        return _empty()
    boxes = _boxes(row, field)
    if not boxes:
        return _empty()
    packed = labels_from_tech({"objects": boxes})
    lamps = np.zeros((N_OBJECTS,), dtype=np.float32)
    seen = np.zeros((N_OBJECTS,), dtype=np.float32)
    for slot, box in enumerate(boxes):
        lamps[slot] = float(box["_brake"])
        if box["_brake_seen"]:
            seen[slot] = 1.0
    open_head = BRAKE_LIGHT_HEAD if brake_light_head is None else bool(brake_light_head)
    return {
        "objects": packed["objects"],
        "object_class": packed["object_class"],
        "object_valid": packed["object_valid"],
        "brake_lights": lamps,
        "brake_lights_valid": seen if open_head else np.zeros_like(seen),
    }
