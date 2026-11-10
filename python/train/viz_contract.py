"""What the visualizer is allowed to draw.

Everything drawn comes from the cameras and the live model that is driving.
BeamNG ground truth, game poses, sensors, the map, the road network, and the
AI path are training labels or forbidden inputs. They are not drawn at
runtime, including as a debug fallback. A field with no model output, no
checkpoint id, or confidence below ``MIN_CONF`` is not drawable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from python.train.detect_targets import LOOKAHEADS_M, OBJ_CLASSES

MIN_CONF = 0.40
MODEL_SRC = "model"
# Sources that must never become a drawable, even if a caller also sets src=model.
FORBIDDEN_SRC = frozenset(
    {
        "ai",
        "ai_path",
        "beamng",
        "beamng_gt",
        "debug",
        "game",
        "game_pose",
        "gt",
        "map",
        "pose",
        "road",
        "sensor",
    }
)


class VizContractError(ValueError):
    """A drawable is not an output of the deployed model."""


def _sigmoid(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _confidence(field: Mapping[str, Any]) -> float:
    try:
        number = float(field.get("confidence"))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _source_names(field: Mapping[str, Any]) -> set[str]:
    names = set()
    for key in ("src", "label_src", "source"):
        raw = field.get(key)
        if isinstance(raw, str) and raw.strip():
            names.add(raw.strip().lower())
    return names


def is_model_drawable(field: Any) -> bool:
    """True when ``field`` is a high-confidence output of one model head."""
    if not isinstance(field, Mapping):
        return False
    names = _source_names(field)
    if MODEL_SRC not in names:
        return False
    if names & FORBIDDEN_SRC:
        return False
    head = field.get("head")
    ckpt = field.get("ckpt")
    if not isinstance(head, str) or not head.strip():
        return False
    if not isinstance(ckpt, str) or not ckpt.strip():
        return False
    if _confidence(field) < MIN_CONF:
        return False
    return True


def require_model_drawable(field: Any) -> dict[str, Any]:
    """Return ``field`` or raise. Foreign, untagged, and low-confidence records raise."""
    if not is_model_drawable(field):
        raise VizContractError("refusing a drawable that is not a live model output")
    return dict(field)


# Nested keys that are never model geometry, even inside a valid head record.
_NEST_DROP = frozenset(
    {
        "gt_poly",
        "path_ego",
        "path_world",
        "road_edges",
        "road_network",
    }
)
_GEOM_KEYS = frozenset({"x", "y", "z", "side", "yaw", "length", "width", "height", "cls"})


def _drop_key(key: str) -> bool:
    return key in _NEST_DROP or key in FORBIDDEN_SRC


def _is_geometry_point(field: Mapping[str, Any]) -> bool:
    """An x/y sample with no source tag. Not a path or a ground-truth record."""
    if _source_names(field):
        return False
    return bool(set(field) & {"x", "y", "side"}) and set(field) <= _GEOM_KEYS | {"side"}


def _clean_drawable(field: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a model record and drop foreign children, recursively."""
    out: dict[str, Any] = {}
    for key, value in field.items():
        name = str(key)
        if _drop_key(name):
            continue
        cleaned = _clean_child(value)
        if cleaned is None:
            continue
        out[name] = cleaned
    return out


def _clean_child(value: Any) -> Any | None:
    if isinstance(value, Mapping):
        if _source_names(value) & FORBIDDEN_SRC:
            return None
        if is_model_drawable(value):
            return _clean_drawable(value)
        if _is_geometry_point(value):
            return {str(k): v for k, v in value.items()}
        nested: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _drop_key(name):
                continue
            cleaned = _clean_child(item)
            if cleaned is None:
                continue
            if isinstance(item, Mapping) and not is_model_drawable(item) and not _is_geometry_point(item):
                if not isinstance(cleaned, dict) or not cleaned:
                    continue
            nested[name] = cleaned
        return nested or None
    if isinstance(value, list):
        items: list[Any] = []
        for item in value:
            if isinstance(item, Mapping):
                if _source_names(item) & FORBIDDEN_SRC:
                    continue
                if is_model_drawable(item):
                    items.append(_clean_drawable(item))
                    continue
                if _is_geometry_point(item):
                    items.append({str(k): v for k, v in item.items()})
                    continue
                continue
            cleaned = _clean_child(item)
            if cleaned is not None:
                items.append(cleaned)
        return items
    return value


def renderable_state(payload: Any) -> dict[str, Any]:
    """The only dict the visualizer may draw from.

    Untagged numbers, ground truth, map roads, AI paths, and low-confidence
    model records are omitted. A valid parent does not keep a nested
    ground-truth polyline or ``path_world``. An empty result means draw nothing.
    """
    if not isinstance(payload, Mapping):
        return {}
    kept: dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key)
        if _drop_key(name):
            continue
        if isinstance(value, Mapping):
            if is_model_drawable(value):
                kept[name] = _clean_drawable(value)
                continue
            nested = renderable_state(value)
            if nested:
                kept[name] = nested
        elif isinstance(value, list):
            items = [
                _clean_drawable(item) if isinstance(item, Mapping) else item
                for item in value
                if is_model_drawable(item)
            ]
            if items:
                kept[name] = items
    return kept


def _batch(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim >= 2:
        return arr[0]
    return arr


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def package_lane(*, ckpt: str, lane_x: Any, confidence: Any) -> dict[str, Any]:
    """Tag the lane center. Low confidence clears the metre so nothing is drawn."""
    conf = _finite(confidence) or 0.0
    x = _finite(lane_x)
    return {
        "src": MODEL_SRC,
        "head": "lane_gt",
        "ckpt": str(ckpt),
        "confidence": conf,
        "lane_x": None if conf < MIN_CONF else x,
    }


def package_detect(
    *,
    ckpt: str,
    edge: Any = None,
    edge_conf: Any = None,
    obj_box: Any = None,
    obj_conf: Any = None,
    obj_class: Any = None,
    brake: Any = None,
    brake_conf: Any = None,
) -> dict[str, Any] | None:
    """Tag model heads. Geometry is empty unless confidence is at least ``MIN_CONF``.

    ``edge_conf``, ``obj_conf``, and ``brake_conf`` are logits. Missing heads
    are omitted. This function never reads ground truth.
    """
    if all(item is None for item in (edge, obj_box, brake)):
        return None
    out: dict[str, Any] = {}
    ckpt_id = str(ckpt)
    if edge is not None and edge_conf is not None:
        values = np.asarray(edge, dtype=np.float64).reshape(-1)
        conf = _sigmoid(np.asarray(edge_conf, dtype=np.float64).reshape(-1)[0])
        points: list[dict[str, float]] = []
        if conf >= MIN_CONF:
            for index, y_m in enumerate(LOOKAHEADS_M):
                if values.size < 2 * index + 2:
                    break
                left = _finite(values[2 * index])
                right = _finite(values[2 * index + 1])
                if left is not None:
                    points.append({"side": "left", "x": left, "y": float(y_m)})
                if right is not None:
                    points.append({"side": "right", "x": right, "y": float(y_m)})
        out["edge"] = {
            "src": MODEL_SRC,
            "head": "edge",
            "ckpt": ckpt_id,
            "confidence": conf,
            "points": points,
        }
    if obj_box is not None and obj_conf is not None and obj_class is not None:
        boxes = _batch(obj_box)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        confs = np.asarray(obj_conf, dtype=np.float64).reshape(-1)
        logits = _batch(obj_class)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        items: list[dict[str, Any]] = []
        n = min(int(boxes.shape[0]), int(confs.shape[0]), int(logits.shape[0]))
        for index in range(n):
            conf = _sigmoid(confs[index])
            if conf < MIN_CONF:
                continue
            row = boxes[index]
            if row.shape[0] < 7 or not np.isfinite(row[:2]).all():
                continue
            cls_index = int(np.argmax(logits[index]))
            cls_index = max(0, min(len(OBJ_CLASSES) - 1, cls_index))
            item = {
                "src": MODEL_SRC,
                "head": "obj",
                "ckpt": ckpt_id,
                "confidence": conf,
                "cls": OBJ_CLASSES[cls_index],
                "x": float(row[0]),
                "y": float(row[1]),
                "z": _finite(row[2]),
                "yaw": _finite(row[3]),
                "length": _finite(row[4]),
                "width": _finite(row[5]),
                "height": _finite(row[6]),
            }
            items.append(item)
        out["objects"] = items
    if brake is not None and brake_conf is not None:
        conf = _sigmoid(np.asarray(brake_conf, dtype=np.float64).reshape(-1)[0])
        if conf >= MIN_CONF:
            prob = _sigmoid(np.asarray(brake, dtype=np.float64).reshape(-1)[0])
            out["brake"] = {
                "src": MODEL_SRC,
                "head": "brakelight",
                "ckpt": ckpt_id,
                "confidence": conf,
                "on": bool(prob >= 0.5),
            }
    return out
