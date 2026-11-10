"""Training-only labels for the opt-in edge, object, and brake-light heads.

BeamNG ground truth is a training label. These records are tagged
``label_src=beamng_gt`` and are not drawable. The live visualizer may draw
only the deployed model's outputs (see ``python/train/viz_contract.py``).

What a recorded row can actually supply:

* Road edges: an explicit ego-frame polyline (``edge_left`` / ``edge_right``)
  or, failing that, ``road_width_m`` plus ``road_offset_m``. ``lane_width_m``
  and painted ``lane_gt`` are never read. A missing side stays masked. The
  packer does not invent the other side from one line.
* Objects: traffic ``pos``/``dir`` in world, converted with the ego pose, or
  an ego-frame ``x``/``y``. Size stays masked unless ``length_m``,
  ``width_m``, and ``height_m`` are on the record. Cones come only from
  ``static_objects`` or ``cones`` boxes already on the row.
* Lead brake lights: a ``brakelights`` field on a traffic vehicle ahead.
  The ego car's own ``brakelights`` is not the lead. The collector's traffic
  probe stores pos/vel/dir only, so this label is masked on today's strips.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LOOKAHEADS_M = (5.0, 10.0, 20.0, 30.0)
EDGE_DIM = 2 * len(LOOKAHEADS_M)
OBJ_SLOTS = 8
OBJ_FEAT = 7  # x, y, z, yaw, length, width, height
OBJ_CLASSES = ("vehicle", "cone")
LABEL_SRC = "beamng_gt"
FEAT_NAMES = ("x", "y", "z", "yaw", "length", "width", "height")

_SIDECAR = "detect.jsonl"
_CACHE: dict[str, dict[int, dict[str, Any]]] = {}


def clear_detect_cache() -> None:
    _CACHE.clear()


def wants_row(row: Mapping[str, Any] | None) -> bool:
    del row
    return True


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _xyz(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"), value.get("z"))
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3].copy()


def _unit(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def world_to_ego(
    pos: Any,
    ego_pos: Any,
    ego_dir: Any,
    *,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float] | None:
    """World point to ego metres: +x right, +y forward, +z up.

    Returns None when the ego pose is missing or the forward axis is parallel
    to up. It does not assume the car sits at the origin.
    """
    point = _xyz(pos)
    origin = _xyz(ego_pos)
    forward = _unit(_xyz(ego_dir)) if _xyz(ego_dir) is not None else None
    if point is None or origin is None or forward is None:
        return None
    up_vec = _unit(np.asarray(up, dtype=np.float64))
    if up_vec is None:
        return None
    right = _unit(np.cross(forward, up_vec))
    if right is None:
        return None
    lifted = np.cross(right, forward)
    rel = point - origin
    return (float(np.dot(rel, right)), float(np.dot(rel, forward)), float(np.dot(rel, lifted)))


def _yaw_world(direction: Any, ego_dir: Any) -> float | None:
    car = _unit(_xyz(direction)) if _xyz(direction) is not None else None
    forward = _unit(_xyz(ego_dir)) if _xyz(ego_dir) is not None else None
    if car is None or forward is None:
        return None
    up_vec = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = _unit(np.cross(forward, up_vec))
    if right is None:
        return None
    return float(math.atan2(float(np.dot(car, right)), float(np.dot(car, forward))))


def _dig(row: Mapping[str, Any], *keys: str) -> Any:
    road = row.get("road") if isinstance(row.get("road"), Mapping) else {}
    tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else {}
    for src in (row, road, tech):
        if not isinstance(src, Mapping):
            continue
        for key in keys:
            if key in src and src.get(key) is not None:
                return src.get(key)
    return None


def _pairs(value: Any) -> list[tuple[float, float]]:
    """Ego-frame (x, y) samples. Accepts ``[x, y]`` or ``{"x", "y"}``."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            x = _finite(item.get("x"))
            y = _finite(item.get("y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = _finite(item[0])
            y = _finite(item[1])
        else:
            continue
        if x is None or y is None:
            continue
        out.append((x, y))
    return out


def _sample_side(points: list[tuple[float, float]], y_m: float) -> float | None:
    """x at ``y_m`` inside the labeled span. No extrapolation."""
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: item[1])
    ys = [item[1] for item in ordered]
    xs = [item[0] for item in ordered]
    if y_m < ys[0] or y_m > ys[-1]:
        return None
    # Collapse duplicate y so np.interp's xp is strictly increasing.
    keep_x: list[float] = []
    keep_y: list[float] = []
    for x, y in zip(xs, ys):
        if keep_y and abs(y - keep_y[-1]) < 1e-6:
            keep_x[-1] = x
            continue
        keep_x.append(x)
        keep_y.append(y)
    if len(keep_y) < 2:
        return None
    return float(np.interp(y_m, np.asarray(keep_y), np.asarray(keep_x)))


def _edge_from_width(row: Mapping[str, Any]) -> tuple[list[float | None], list[float], str]:
    width = _finite(_dig(row, "road_width_m"))
    offset = _finite(_dig(row, "road_offset_m", "offset_m"))
    if width is None or offset is None or width <= 0.0:
        return [None] * EDGE_DIM, [0.0] * EDGE_DIM, "absent"
    if _dig(row, "off_road") is True:
        return [None] * EDGE_DIM, [0.0] * EDGE_DIM, "absent"
    from python.train.sensor_aux import pack_road_edge

    heading = _finite(_dig(row, "heading_err_rad", "heading_rad"))
    values, valid = pack_road_edge(
        offset_m=offset,
        heading_err_rad=heading,
        road_width_m=width,
        off_road=False,
    )
    packed = [None if not math.isfinite(float(v)) else float(v) for v in values]
    mask = [float(v) for v in valid]
    return packed, mask, "road_width"


def _edge_from_polyline(row: Mapping[str, Any]) -> tuple[list[float | None], list[float]] | None:
    left = _pairs(_dig(row, "edge_left", "edge_left_xy"))
    right = _pairs(_dig(row, "edge_right", "edge_right_xy"))
    if not left and not right:
        return None
    values: list[float | None] = [None] * EDGE_DIM
    valid = [0.0] * EDGE_DIM
    for index, y_m in enumerate(LOOKAHEADS_M):
        lx = _sample_side(left, y_m)
        rx = _sample_side(right, y_m)
        if lx is not None:
            values[2 * index] = lx
            valid[2 * index] = 1.0
        if rx is not None:
            values[2 * index + 1] = rx
            valid[2 * index + 1] = 1.0
    return values, valid


def edge_label(row: Mapping[str, Any]) -> dict[str, Any]:
    """Left/right road edge at 5/10/20/30 m. Not paint. Not ``lane_gt``."""
    poly = _edge_from_polyline(row)
    if poly is not None:
        values, valid = poly
        method = "polyline" if any(valid) else "absent"
    else:
        values, valid, method = _edge_from_width(row)
    return {
        "kind": "edge",
        "label_src": LABEL_SRC,
        "train_only": True,
        "method": method,
        "values": values,
        "valid": valid,
    }


def _size(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    length = _finite(obj.get("length_m", obj.get("length")))
    width = _finite(obj.get("width_m", obj.get("width")))
    height = _finite(obj.get("height_m", obj.get("height")))
    if length is None or width is None or height is None:
        return None
    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        return None
    return length, width, height


def _class_name(obj: Mapping[str, Any]) -> str:
    raw = str(obj.get("cls") or obj.get("class") or obj.get("name") or "").lower()
    if "cone" in raw:
        return "cone"
    return "vehicle"


def _blank_slot() -> dict[str, Any]:
    return {"cls": "", "box": [None] * OBJ_FEAT, "valid": [0.0] * OBJ_FEAT}


def _slot_from_ego(
    *,
    cls: str,
    x: float | None,
    y: float | None,
    z: float | None,
    yaw: float | None,
    size: tuple[float, float, float] | None,
) -> dict[str, Any] | None:
    if x is None or y is None:
        return None
    box: list[float | None] = [x, y, z, yaw, None, None, None]
    valid = [1.0, 1.0, 1.0 if z is not None else 0.0, 1.0 if yaw is not None else 0.0, 0.0, 0.0, 0.0]
    if size is not None:
        box[4], box[5], box[6] = size
        valid[4] = valid[5] = valid[6] = 1.0
    return {"cls": cls, "box": box, "valid": valid}


def _object_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    ego_pos = ego.get("pos") if isinstance(ego, Mapping) else None
    if ego_pos is None:
        ego_pos = row.get("pos")
    ego_dir = None
    if isinstance(ego, Mapping):
        ego_dir = ego.get("dir", ego.get("direction"))
    if ego_dir is None:
        ego_dir = row.get("dir")
    found: list[tuple[float, dict[str, Any]]] = []

    def _push(slot: dict[str, Any] | None) -> None:
        if slot is None:
            return
        y = slot["box"][1]
        found.append((abs(float(y)), slot))

    traffic = row.get("traffic")
    if isinstance(traffic, list):
        for car in traffic:
            if not isinstance(car, Mapping):
                continue
            size = _size(car)
            if "x" in car and "y" in car and "pos" not in car:
                slot = _slot_from_ego(
                    cls=_class_name(car),
                    x=_finite(car.get("x")),
                    y=_finite(car.get("y")),
                    z=_finite(car.get("z")) if "z" in car else None,
                    yaw=_finite(car.get("yaw")) if "yaw" in car else None,
                    size=size,
                )
                _push(slot)
                continue
            if "pos" not in car:
                continue
            ego_xyz = world_to_ego(car.get("pos"), ego_pos, ego_dir)
            if ego_xyz is None:
                continue
            yaw = _yaw_world(car.get("dir"), ego_dir)
            slot = _slot_from_ego(
                cls=_class_name(car),
                x=ego_xyz[0],
                y=ego_xyz[1],
                z=ego_xyz[2],
                yaw=yaw,
                size=size,
            )
            _push(slot)

    static = row.get("static_objects")
    if not isinstance(static, list):
        static = row.get("cones")
    if isinstance(static, list):
        for obj in static:
            if not isinstance(obj, Mapping):
                continue
            size = _size(obj)
            if "pos" in obj:
                ego_xyz = world_to_ego(obj.get("pos"), ego_pos, ego_dir)
                if ego_xyz is None:
                    continue
                yaw = _finite(obj.get("yaw"))
                if yaw is None:
                    yaw = _yaw_world(obj.get("dir"), ego_dir)
                slot = _slot_from_ego(
                    cls=_class_name(obj) if _class_name(obj) == "cone" or obj.get("cls") or obj.get("class") else "cone",
                    x=ego_xyz[0],
                    y=ego_xyz[1],
                    z=ego_xyz[2],
                    yaw=yaw,
                    size=size,
                )
            else:
                slot = _slot_from_ego(
                    cls=_class_name(obj) if (obj.get("cls") or obj.get("class") or "cone" in str(obj.get("name", "")).lower()) else "cone",
                    x=_finite(obj.get("x")),
                    y=_finite(obj.get("y")),
                    z=_finite(obj.get("z")) if "z" in obj else None,
                    yaw=_finite(obj.get("yaw")) if "yaw" in obj else None,
                    size=size,
                )
            _push(slot)
    found.sort(key=lambda item: item[0])
    return [slot for _dist, slot in found[:OBJ_SLOTS]]


def object_label(row: Mapping[str, Any]) -> dict[str, Any]:
    slots = _object_candidates(row)
    while len(slots) < OBJ_SLOTS:
        slots.append(_blank_slot())
    return {
        "label_src": LABEL_SRC,
        "train_only": True,
        "listed": _object_lists_present(row),
        "slots": slots[:OBJ_SLOTS],
    }


def _object_lists_present(row: Mapping[str, Any]) -> bool:
    """True when this frame recorded traffic or static objects, even if empty."""
    return any(isinstance(row.get(key), list) for key in ("traffic", "static_objects", "cones"))


def _brake_value(obj: Mapping[str, Any]) -> float | None:
    for key in ("brakelights", "brake_lights", "brake_light"):
        if key in obj:
            number = _finite(obj.get(key))
            if number is None:
                return None
            return 1.0 if number >= 0.5 else 0.0
    return None


def brake_label(row: Mapping[str, Any]) -> dict[str, Any]:
    """Lead-vehicle brake lamp. Ego ``brakelights`` is ignored."""
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    ego_pos = ego.get("pos") if isinstance(ego, Mapping) else row.get("pos")
    ego_dir = None
    if isinstance(ego, Mapping):
        ego_dir = ego.get("dir", ego.get("direction"))
    if ego_dir is None:
        ego_dir = row.get("dir")
    best: tuple[float, float] | None = None
    traffic = row.get("traffic") if isinstance(row.get("traffic"), list) else []
    saw_field = False
    for car in traffic:
        if not isinstance(car, Mapping):
            continue
        lamp = _brake_value(car)
        if lamp is None:
            continue
        saw_field = True
        if "x" in car and "y" in car and "pos" not in car:
            y = _finite(car.get("y"))
        else:
            converted = world_to_ego(car.get("pos"), ego_pos, ego_dir)
            y = None if converted is None else converted[1]
        if y is None or y <= 0.0:
            continue
        if best is None or y < best[0]:
            best = (y, lamp)
    if best is None:
        note = (
            "no lead vehicle with brakelights in the recording"
            if not saw_field
            else "no lead ahead with brakelights"
        )
        return {"label_src": LABEL_SRC, "train_only": True, "on": None, "valid": 0.0, "note": note}
    return {
        "label_src": LABEL_SRC,
        "train_only": True,
        "on": best[1],
        "valid": 1.0,
        "note": "lead brakelights from the recorded traffic vehicle",
    }


def label_record(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """One training row. Does not copy ``lane_gt`` and does not mutate ``row``."""
    source = dict(row or {})
    return {
        "i": source.get("i"),
        "t": source.get("t"),
        "label_src": LABEL_SRC,
        "train_only": True,
        "edge": edge_label(source),
        "objects": object_label(source),
        "brake": brake_label(source),
    }


def _load_sidecar(root: Path | str | None) -> dict[int, dict[str, Any]]:
    if root is None:
        return {}
    path = Path(root) / _SIDECAR
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    table: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and rec.get("i") is not None:
                table[int(rec["i"])] = rec
    _CACHE[key] = table
    return table


def _num_list(value: Any, n: int) -> list[float]:
    raw = list(value or [])
    out = []
    for index in range(n):
        item = raw[index] if index < len(raw) else None
        number = _finite(item)
        out.append(0.0 if number is None else float(number))
    return out


def arrays_from_record(rec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Loss tensors. Paint ``lane_gt`` is not a key."""
    edge = rec.get("edge") if isinstance(rec.get("edge"), Mapping) else {}
    values = _num_list(edge.get("values"), EDGE_DIM)
    valid = _num_list(edge.get("valid"), EDGE_DIM)
    objects = rec.get("objects") if isinstance(rec.get("objects"), Mapping) else {}
    slots = objects.get("slots") if isinstance(objects.get("slots"), list) else []
    box = np.zeros((OBJ_SLOTS, OBJ_FEAT), dtype=np.float32)
    mask = np.zeros((OBJ_SLOTS, OBJ_FEAT), dtype=np.float32)
    cls = np.zeros((OBJ_SLOTS,), dtype=np.int64)
    for index in range(OBJ_SLOTS):
        slot = slots[index] if index < len(slots) and isinstance(slots[index], Mapping) else {}
        raw_box = slot.get("box") if isinstance(slot, Mapping) else None
        raw_valid = slot.get("valid") if isinstance(slot, Mapping) else None
        for feat in range(OBJ_FEAT):
            number = None
            if isinstance(raw_box, (list, tuple)) and feat < len(raw_box):
                number = _finite(raw_box[feat])
            flag = 0.0
            if isinstance(raw_valid, (list, tuple)) and feat < len(raw_valid):
                flag = 1.0 if (_finite(raw_valid[feat]) or 0.0) > 0.5 else 0.0
            if number is None:
                flag = 0.0
                number = 0.0
            box[index, feat] = float(number)
            mask[index, feat] = flag
        name = str(slot.get("cls") or "") if isinstance(slot, Mapping) else ""
        cls[index] = OBJ_CLASSES.index(name) if name in OBJ_CLASSES else 0
    brake = rec.get("brake") if isinstance(rec.get("brake"), Mapping) else {}
    on = _finite(brake.get("on"))
    brake_valid = 1.0 if (_finite(brake.get("valid")) or 0.0) > 0.5 and on is not None else 0.0
    listed = 1.0 if objects.get("listed") is True or objects.get("listed") == 1 or objects.get("listed") == 1.0 else 0.0
    return {
        "edge": np.asarray(values, dtype=np.float32),
        "edge_valid": np.asarray(valid, dtype=np.float32),
        "obj_box": box,
        "obj_valid": mask,
        "obj_class": cls,
        "brake": np.asarray(0.0 if on is None else on, dtype=np.float32),
        "brake_valid": np.asarray(brake_valid, dtype=np.float32),
        "obj_listed": np.asarray(listed, dtype=np.float32),
    }


def training_arrays(row: Mapping[str, Any] | None, root: Path | str | None = None) -> dict[str, np.ndarray]:
    """Sidecar ``detect.jsonl`` wins. Otherwise the row is labeled in place.

    Either way the arrays are training targets. They are not model outputs.
    """
    source = row or {}
    rec = None
    if root is not None and source.get("i") is not None:
        rec = _load_sidecar(root).get(int(source["i"]))
    if rec is None:
        rec = label_record(source)
    return arrays_from_record(rec)


def export_root(root: Path | str) -> int:
    """Write ``detect.jsonl`` beside ``state.jsonl``. Returns the row count."""
    folder = Path(root)
    src = folder / "state.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"{src} is not a recording")
    lines = []
    count = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        lines.append(json.dumps(label_record(row), sort_keys=True))
        count += 1
    (folder / _SIDECAR).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return count
