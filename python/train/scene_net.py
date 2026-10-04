"""One scene net for the 2272×192 strip.

The net predicts lanes, curbs, cars, signs, and lights. It has no steer head.
``predict_path`` turns the scene into the path. The same weights take whatever
step time the recorder stored: each forward consumes the real seconds since
the previous strip. FP32 on compute capability older than 7.0 (Pascal 6.1).
Automatic mixed precision at 7.0 and newer. The exported ONNX graph is FP32
either way.

Memory is 15 seconds of real time. It is not a fixed step count. A short
skip adds the recent tokens whose age is still inside ``RESIDUAL_S`` seconds.
That skip is not a fixed 8 steps, and it is not a second model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from python.data.strip_writer import CANVAS_H, CANVAS_W, SECTORS

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None

MEMORY_S = 15.0
STEM_CHANNELS = (32, 64, 128, 128)
SECTOR_FEAT = 64
TOKEN_DIM = 512
GRU_HIDDEN = 512
# The skip covers this many seconds of real time, not a fixed step count.
RESIDUAL_S = 1.5
# The ring is sized for a 60 Hz step so a faster-than-5 Hz strip still fits
# the whole span. Slower steps simply leave more of the ring unused.
RESIDUAL_MIN_DT = 1.0 / 60.0
RESIDUAL_TOKENS = int(math.ceil(RESIDUAL_S / RESIDUAL_MIN_DT))
# Dropout on the vector that enters the GRU. Not on the hidden state.
GRU_IN_DROPOUT = 0.1
N_LANES = 6
LANE_POINTS = 16
N_CURBS = 2
N_OBJECTS = 16
OBJECT_FEAT = 8
N_SIGNS = 8
SIGN_FEAT = 4

# Columns of objects[16, 8]: x, y, yaw, length, width, speed, partial, unseen_s.
# Columns of signs[8, 4]: x, y, misses, seen fraction.
OBJECT_CLASSES = ("vehicle", "car", "truck", "bus", "pedestrian", "bike")
SIGN_CLASSES = ("stop_sign", "yield", "traffic_light", "speed_limit")
LAMP_STATES = ("unknown", "red", "yellow", "green")

_CLASS_ALIASES = {
    "stop": "stop_sign",
    "yield_sign": "yield",
    "light": "traffic_light",
    "trafficlight": "traffic_light",
    "speed": "speed_limit",
    "speed_sign": "speed_limit",
    "amber": "yellow",
    "r": "red",
    "g": "green",
    "y": "yellow",
}

# Keyword names of python.planning.path_predictor.predict_path.
# There is no curb argument and no steer argument. A valid curb is one more
# polyline in lanes_bev, which is the road-line list that planner already pairs.
PREDICT_PATH_ARGS = (
    "lanes_bev",
    "lane_conf",
    "curvature",
    "tracks",
    "signs",
    "ego_speed_mps",
    "prior_lanes",
    "length_m",
    "step_m",
)

OUTPUT_FIELDS = (
    "lanes",
    "lane_valid",
    "curbs",
    "curb_valid",
    "objects",
    "object_class",
    "object_valid",
    "signs",
    "sign_class",
    "sign_state",
    "sign_valid",
)

LOSS_TERMS = ("points", "class", "lamps", "valid")

_BAD_REASONS = frozenset({"player_steer", "player_brake", "player_throttle"})

SCENE_ONNX_NAME = "e2e_scene.onnx"


def scene_head_sizes() -> tuple[tuple[str, int], ...]:
    """Linear heads on the GRU hidden state. Names match the exported net."""
    return (
        ("lane_head", N_LANES * LANE_POINTS * 2),
        ("lane_valid_head", N_LANES),
        ("curb_head", N_CURBS * LANE_POINTS * 2),
        ("curb_valid_head", N_CURBS),
        ("object_head", N_OBJECTS * OBJECT_FEAT),
        ("object_class_head", N_OBJECTS * len(OBJECT_CLASSES)),
        ("object_valid_head", N_OBJECTS),
        ("sign_head", N_SIGNS * SIGN_FEAT),
        ("sign_class_head", N_SIGNS * len(SIGN_CLASSES)),
        ("sign_state_head", N_SIGNS * len(LAMP_STATES)),
        ("sign_valid_head", N_SIGNS),
    )


def conv2d_parameter_count(
    in_channels: int,
    out_channels: int,
    kernel: int = 3,
    *,
    bias: bool = True,
) -> int:
    """Weights in one conv. Bias adds one value per output channel."""
    weights = int(out_channels) * int(in_channels) * int(kernel) * int(kernel)
    if bias:
        weights += int(out_channels)
    return weights


def linear_parameter_count(in_features: int, out_features: int, *, bias: bool = True) -> int:
    """Weights in one linear layer."""
    weights = int(out_features) * int(in_features)
    if bias:
        weights += int(out_features)
    return weights


def gru_parameter_count(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    *,
    bias: bool = True,
    bidirectional: bool = False,
) -> int:
    """Weights in a PyTorch GRU.

    Each layer has three gates. ``weight_ih`` is ``3H x input``, ``weight_hh``
    is ``3H x H``, and bias adds two vectors of length ``3H`` when enabled.
    """
    directions = 2 if bidirectional else 1
    total = 0
    hidden = int(hidden_size)
    for layer in range(int(num_layers)):
        layer_in = int(input_size) if layer == 0 else hidden * directions
        gates = 3 * hidden
        per_direction = gates * layer_in + gates * hidden
        if bias:
            per_direction += gates + gates
        total += per_direction * directions
    return total


def architecture_parameter_count() -> int:
    """Trainable weights of the exported scene net, from the layer sizes.

    The stem is shared, so it is counted once. ReLU, the residual, and the
    sigmoids add no weights. This is the count before any tensor exists.
    """
    total = 0
    cin = 3
    for cout in STEM_CHANNELS:
        total += conv2d_parameter_count(cin, int(cout), 3, bias=True)
        cin = int(cout)
    total += linear_parameter_count(STEM_CHANNELS[-1], SECTOR_FEAT, bias=True)
    total += gru_parameter_count(TOKEN_DIM + 1, GRU_HIDDEN, num_layers=1, bias=True)
    for _name, out in scene_head_sizes():
        total += linear_parameter_count(GRU_HIDDEN, out, bias=True)
    return total


def count_module_parameters(module: Any) -> int:
    """Sum trainable parameter elements on a built module."""
    total = 0
    for param in module.parameters():
        if not bool(getattr(param, "requires_grad", True)):
            continue
        total += int(param.numel())
    return total


def default_onnx_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / SCENE_ONNX_NAME


def torch_ready() -> bool:
    return torch is not None


def split_capability(capability: Any) -> tuple[int, int] | None:
    """Parse a compute capability. ``None`` means there is no GPU capability."""
    if capability is None:
        return None
    if isinstance(capability, str):
        capability = float(capability.strip())
    if isinstance(capability, (tuple, list)):
        if len(capability) < 2:
            return None
        return int(capability[0]), int(capability[1])
    value = float(capability)
    if not math.isfinite(value):
        return None
    major = int(math.floor(value + 1e-9))
    minor = int(round((value - major) * 10.0))
    if minor >= 10:
        major += 1
        minor -= 10
    if minor < 0:
        minor = 0
    return major, minor


def amp_enabled(capability: Any) -> bool:
    """True only for compute capability 7.0 or newer.

    Pascal (6.1) and anything older stay FP32. A missing capability stays FP32.
    """
    parts = split_capability(capability)
    if parts is None:
        return False
    return parts >= (7, 0)


def precision_for_capability(capability: Any) -> str:
    """``amp`` at compute 7.0+, otherwise ``fp32``."""
    return "amp" if amp_enabled(capability) else "fp32"


def letterbox_rect(
    src_w: int,
    src_h: int,
    dst_w: int = 256,
    dst_h: int = CANVAS_H,
) -> tuple[int, int, int, int]:
    """Fitted width, height, and top-left pad for a letterbox into ``dst``."""
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    nw = max(1, min(dst_w, int(round(src_w * scale))))
    nh = max(1, min(dst_h, int(round(src_h * scale))))
    x0 = (dst_w - nw) // 2
    y0 = (dst_h - nh) // 2
    return nw, nh, x0, y0


def step_dt(dt_s: float | None) -> float:
    """Seconds the GRU consumes. The first strip has no previous step.

    The value is the recorded gap. It is not rewritten to 0.2 s.
    """
    if dt_s is None:
        return 0.0
    return float(dt_s)


def residual_ages(
    dts: list[float],
    span_s: float = RESIDUAL_S,
    capacity: int = RESIDUAL_TOKENS,
) -> list[float]:
    """Age of each ring slot after consuming the real step times.

    Empty slots start older than ``span_s``. Each step ages the ring by that
    step's seconds, drops the oldest slot, and appends the new token at age 0.
    """
    empty = float(span_s) + 1.0
    ages = [empty] * int(capacity)
    for dt in dts:
        step = float(dt)
        ages = [age + step for age in ages]
        ages = ages[1:] + [0.0]
    return ages


def residual_keep_count(
    dts: list[float],
    span_s: float = RESIDUAL_S,
    capacity: int = RESIDUAL_TOKENS,
) -> int:
    """How many ring slots are still inside the skip span."""
    return sum(1 for age in residual_ages(dts, span_s, capacity) if age <= float(span_s))


def memory_windows(rows: list[Mapping[str, Any]], memory_s: float = MEMORY_S) -> list[list[Mapping[str, Any]]]:
    """Group strips into windows of at most ``memory_s`` real seconds.

    Every row is kept. Nothing is inserted or dropped to hit a fixed rate.
    A row that would push the window past ``memory_s`` starts the next window.
    """
    if not rows:
        return []
    windows: list[list[Mapping[str, Any]]] = []
    cur: list[Mapping[str, Any]] = [rows[0]]
    t0 = float(rows[0]["t"])
    for row in rows[1:]:
        t = float(row["t"])
        if t - t0 > float(memory_s):
            windows.append(cur)
            cur = [row]
            t0 = t
        else:
            cur.append(row)
    windows.append(cur)
    return windows


def _finite(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _marked_bad(blob: Any) -> bool:
    if not isinstance(blob, dict):
        return False
    if blob.get("bad") is True:
        return True
    for key in ("override", "reason", "disengage_reason"):
        reason = str(blob.get(key) or "").strip().lower()
        if reason in _BAD_REASONS:
            return True
    return False


def moment_ok(wheel: Any, pedals: Any) -> bool:
    """False when wheel or pedals mark a bad moment.

    In-range controls are kept. These channels are a gate, not a loss target.
    """
    if wheel is None:
        wheel = {}
    if pedals is None:
        pedals = {}
    if not isinstance(wheel, dict) or not isinstance(pedals, dict):
        return False
    if _marked_bad(wheel) or _marked_bad(pedals):
        return False
    steer = None
    for key in ("steer", "steering_input", "steering"):
        if key in wheel and wheel[key] is not None:
            steer = _finite(wheel[key])
            if steer is None:
                return False
            break
    if steer is not None and abs(steer) > 1.0:
        return False
    for key in ("throttle", "brake"):
        if key not in pedals or pedals[key] is None:
            continue
        val = _finite(pedals[key])
        if val is None or val < 0.0 or val > 1.0:
            return False
    return True


def loss_mask(row: Mapping[str, Any]) -> bool:
    """True when this strip contributes scene loss. Wheel and pedals are the gate."""
    if not moment_ok(row.get("wheel"), row.get("pedals")):
        return False
    tech = row.get("tech")
    return isinstance(tech, dict) and bool(tech)


def _has_labels(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def live_strip_fields(
    *,
    lanes: Any = None,
    tracks: Any = None,
    signs: Any = None,
    curbs: Any = None,
    steer: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    override_reason: Any = None,
) -> dict[str, Any]:
    """Scene labels and the wheel/pedal gate for one live strip.

    ``tech`` is present only when this tick has lanes, tracks, signs, or curbs,
    which is what ``loss_mask`` requires before a step is supervised. A player
    override reason is stored on the wheel or the pedals so that moment is gated.
    """
    tech: dict[str, Any] = {}
    if _has_labels(lanes):
        tech["lanes"] = lanes
    if _has_labels(tracks):
        tech["tracks"] = tracks
    if _has_labels(signs):
        tech["signs"] = signs
    if _has_labels(curbs):
        tech["curbs"] = curbs
    wheel: dict[str, Any] = {"steer": float(steer or 0.0)}
    pedals: dict[str, Any] = {
        "throttle": float(throttle or 0.0),
        "brake": float(brake or 0.0),
    }
    reason = str(override_reason or "").strip().lower()
    if reason in _BAD_REASONS:
        if reason == "player_steer":
            wheel["reason"] = reason
        else:
            pedals["reason"] = reason
    return {"wheel": wheel, "pedals": pedals, "tech": tech or None}


def _class_index(value: Any, table: tuple[str, ...]) -> int | None:
    if isinstance(value, str):
        key = _CLASS_ALIASES.get(value.strip().lower(), value.strip().lower())
        try:
            return table.index(key)
        except ValueError:
            return None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(table):
        return None
    return idx


def _resample_poly(poly: Any, n: int = LANE_POINTS) -> np.ndarray | None:
    pts: list[tuple[float, float]] = []
    for p in poly or []:
        if isinstance(p, dict):
            x, y = p.get("x"), p.get("y")
        else:
            try:
                x, y = p[0], p[1]
            except (TypeError, IndexError):
                continue
        if x is None or y is None:
            continue
        try:
            pts.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, arr[:, 0]), np.interp(dst, src, arr[:, 1])], axis=1)


def _as_point_array(value: Any, n_slots: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(points [n,16,2], valid [n])`` when ``value`` is already that tensor."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 3 and arr.shape[1] == LANE_POINTS and arr.shape[2] == 2:
        out = np.zeros((n_slots, LANE_POINTS, 2), dtype=np.float64)
        valid = np.zeros((n_slots,), dtype=np.float64)
        n = min(n_slots, arr.shape[0])
        out[:n] = arr[:n]
        valid[:n] = 1.0
        return out, valid
    return None


def _polys_to_slots(value: Any, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    packed = _as_point_array(value, n_slots)
    if packed is not None and not isinstance(value, (list, tuple)):
        return packed
    if packed is not None and isinstance(value, np.ndarray):
        return packed
    points = np.zeros((n_slots, LANE_POINTS, 2), dtype=np.float64)
    valid = np.zeros((n_slots,), dtype=np.float64)
    # A list of polylines, or a numeric array that was not already [n,16,2].
    if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[1:] == (LANE_POINTS, 2):
        return packed if packed is not None else (points, valid)
    polys = value if isinstance(value, (list, tuple)) else []
    slot = 0
    for poly in polys:
        if slot >= n_slots:
            break
        sampled = _resample_poly(poly, LANE_POINTS)
        if sampled is None:
            continue
        points[slot] = sampled
        valid[slot] = 1.0
        slot += 1
    return points, valid


def _apply_valid(valid: np.ndarray, given: Any) -> np.ndarray:
    if given is None:
        return valid
    arr = np.asarray(given, dtype=np.float64).reshape(-1)
    out = np.zeros_like(valid)
    n = min(len(out), len(arr))
    out[:n] = arr[:n]
    return out


def _object_matrix(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((N_OBJECTS, OBJECT_FEAT), dtype=np.float64)
    classes = np.zeros((N_OBJECTS,), dtype=np.int64)
    valid = np.zeros((N_OBJECTS,), dtype=np.float64)
    if value is None:
        return points, classes, valid
    arr = None
    try:
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.ndim == 2 and candidate.shape[1] >= OBJECT_FEAT and np.issubdtype(np.asarray(value).dtype, np.number):
            arr = candidate
    except (TypeError, ValueError):
        arr = None
    if arr is not None:
        n = min(N_OBJECTS, arr.shape[0])
        points[:n] = arr[:n, :OBJECT_FEAT]
        valid[:n] = 1.0
        return points, classes, valid
    slot = 0
    for obj in value or []:
        if slot >= N_OBJECTS:
            break
        if not isinstance(obj, dict):
            continue
        cls = _class_index(obj.get("cls", obj.get("class", "vehicle")), OBJECT_CLASSES)
        if cls is None:
            continue
        partial = obj.get("partial", 0.0)
        if isinstance(partial, bool):
            partial = 1.0 if partial else 0.0
        speed = obj.get("speed_mps", obj.get("speed", 0.0))
        points[slot] = (
            float(obj.get("x", 0.0)),
            float(obj.get("y", 0.0)),
            float(obj.get("yaw", obj.get("heading", 0.0))),
            float(obj.get("length", obj.get("length_m", 4.5))),
            float(obj.get("width", obj.get("width_m", 1.8))),
            float(speed or 0.0),
            float(partial or 0.0),
            float(obj.get("unseen_s", 0.0) or 0.0),
        )
        classes[slot] = cls
        valid[slot] = 1.0
        slot += 1
    return points, classes, valid


def _sign_matrix(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((N_SIGNS, SIGN_FEAT), dtype=np.float64)
    classes = np.zeros((N_SIGNS,), dtype=np.int64)
    states = np.zeros((N_SIGNS,), dtype=np.int64)
    valid = np.zeros((N_SIGNS,), dtype=np.float64)
    if value is None:
        return points, classes, states, valid
    arr = None
    try:
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.ndim == 2 and candidate.shape[1] >= SIGN_FEAT and np.issubdtype(np.asarray(value).dtype, np.number):
            arr = candidate
    except (TypeError, ValueError):
        arr = None
    if arr is not None:
        n = min(N_SIGNS, arr.shape[0])
        points[:n] = arr[:n, :SIGN_FEAT]
        valid[:n] = 1.0
        return points, classes, states, valid
    slot = 0
    for sign in value or []:
        if slot >= N_SIGNS:
            break
        if not isinstance(sign, dict):
            continue
        cls = _class_index(sign.get("cls", sign.get("class", "")), SIGN_CLASSES)
        if cls is None:
            continue
        state = _class_index(sign.get("state") or "unknown", LAMP_STATES)
        if state is None:
            state = 0
        seen = sign.get("seen_fraction", sign.get("seen", 1.0))
        points[slot] = (
            float(sign.get("x", 0.0)),
            float(sign.get("y", 0.0)),
            float(sign.get("misses", 0.0) or 0.0),
            float(1.0 if seen is None else seen),
        )
        classes[slot] = cls
        states[slot] = state
        valid[slot] = 1.0
        slot += 1
    return points, classes, states, valid


def _index_vector(value: Any, table: tuple[str, ...], n: int, fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback
    if isinstance(value, np.ndarray) and value.ndim == 2:
        idx = np.argmax(value, axis=-1).astype(np.int64)
        out = np.zeros((n,), dtype=np.int64)
        m = min(n, len(idx))
        out[:m] = idx[:m]
        return out
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        out = fallback.copy()
        for i, item in enumerate(value[:n]):
            parsed = _class_index(item, table)
            if parsed is not None:
                out[i] = parsed
        return out
    try:
        arr = np.asarray(value).reshape(-1)
    except (TypeError, ValueError):
        return fallback
    out = fallback.copy()
    for i, item in enumerate(arr[:n]):
        if isinstance(item, str):
            parsed = _class_index(item, table)
        else:
            parsed = _class_index(int(item), table) if np.ndim(item) == 0 else None
            if parsed is None:
                try:
                    parsed = int(item)
                except (TypeError, ValueError):
                    parsed = None
                if parsed is None or parsed < 0 or parsed >= len(table):
                    parsed = None
        if parsed is not None:
            out[i] = parsed
    return out


def labels_from_tech(tech: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    """Fixed-shape scene labels from a Tech dict, or zeros when labels are absent."""
    tech = tech or {}
    lanes, lane_valid = _polys_to_slots(tech.get("lanes", tech.get("lanes_bev")), N_LANES)
    curbs, curb_valid = _polys_to_slots(tech.get("curbs"), N_CURBS)
    if "lane_valid" in tech:
        lane_valid = _apply_valid(lane_valid, tech.get("lane_valid"))
    if "curb_valid" in tech:
        curb_valid = _apply_valid(curb_valid, tech.get("curb_valid"))
    objects, object_class, object_valid = _object_matrix(tech.get("objects", tech.get("tracks")))
    if "object_class" in tech:
        object_class = _index_vector(tech.get("object_class"), OBJECT_CLASSES, N_OBJECTS, object_class)
    if "object_valid" in tech:
        object_valid = _apply_valid(object_valid, tech.get("object_valid"))
    signs, sign_class, sign_state, sign_valid = _sign_matrix(tech.get("signs"))
    if "sign_class" in tech:
        sign_class = _index_vector(tech.get("sign_class"), SIGN_CLASSES, N_SIGNS, sign_class)
    if "sign_state" in tech:
        sign_state = _index_vector(tech.get("sign_state"), LAMP_STATES, N_SIGNS, sign_state)
    if "sign_valid" in tech:
        sign_valid = _apply_valid(sign_valid, tech.get("sign_valid"))
    return {
        "lanes": lanes.astype(np.float32),
        "lane_valid": lane_valid.astype(np.float32),
        "curbs": curbs.astype(np.float32),
        "curb_valid": curb_valid.astype(np.float32),
        "objects": objects.astype(np.float32),
        "object_class": object_class.astype(np.int64),
        "object_valid": object_valid.astype(np.float32),
        "signs": signs.astype(np.float32),
        "sign_class": sign_class.astype(np.int64),
        "sign_state": sign_state.astype(np.int64),
        "sign_valid": sign_valid.astype(np.float32),
    }


def _field(scene: Any, name: str, default: Any = None) -> Any:
    if isinstance(scene, dict):
        return scene.get(name, default)
    return getattr(scene, name, default)


def _numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _slot_name(value: Any, table: tuple[str, ...]) -> str:
    if isinstance(value, str):
        key = _CLASS_ALIASES.get(value.strip().lower(), value.strip().lower())
        if key in table:
            return key
        return table[0]
    arr = _numpy(value)
    if arr.ndim == 0:
        idx = int(arr)
    else:
        idx = int(np.argmax(arr))
    idx = max(0, min(len(table) - 1, idx))
    return table[idx]


def _poly_dicts(points: np.ndarray) -> list[dict[str, float]]:
    out = []
    for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        out.append({"x": float(x), "y": float(y), "z": 0.0})
    return out


def scene_to_predict_kwargs(scene: Any, *, ego_speed_mps: float = 0.0) -> dict[str, Any]:
    """Map scene tensors onto ``predict_path`` keywords.

    The returned keys are a subset of ``PREDICT_PATH_ARGS``. Curbs are extra
    polylines in ``lanes_bev``. Steering is unchanged because this dict is not
    a steer command and ``predict_path`` does not take one.
    """
    lanes = _numpy(_field(scene, "lanes"))
    lane_valid = _numpy(_field(scene, "lane_valid")).reshape(-1)
    curbs = _numpy(_field(scene, "curbs"))
    curb_valid = _numpy(_field(scene, "curb_valid")).reshape(-1)
    objects = _numpy(_field(scene, "objects"))
    object_class = _numpy(_field(scene, "object_class"))
    object_valid = _numpy(_field(scene, "object_valid")).reshape(-1)
    signs = _numpy(_field(scene, "signs"))
    sign_class = _field(scene, "sign_class")
    sign_state = _field(scene, "sign_state")
    sign_valid_raw = _field(scene, "sign_valid", None)
    sign_valid = (
        np.ones((signs.shape[0],), dtype=np.float64)
        if sign_valid_raw is None
        else _numpy(sign_valid_raw).reshape(-1)
    )

    polylines: list[list[dict[str, float]]] = []
    for i, flag in enumerate(lane_valid):
        if float(flag) <= 0.5:
            continue
        polylines.append(_poly_dicts(lanes[i]))
    for i, flag in enumerate(curb_valid):
        if float(flag) <= 0.5:
            continue
        polylines.append(_poly_dicts(curbs[i]))

    tracks = []
    for i, flag in enumerate(object_valid):
        if float(flag) <= 0.5:
            continue
        row = objects[i]
        tracks.append(
            {
                "id": int(i),
                "cls": _slot_name(object_class[i], OBJECT_CLASSES),
                "x": float(row[0]),
                "y": float(row[1]),
                "yaw": float(row[2]),
                "length": float(row[3]),
                "width": float(row[4]),
                "speed_mps": float(row[5]),
                "partial": bool(float(row[6]) >= 0.5),
                "unseen_s": float(row[7]),
            }
        )

    sign_rows = []
    for i, flag in enumerate(sign_valid):
        if float(flag) <= 0.5:
            continue
        row = signs[i]
        sign_rows.append(
            {
                "cls": _slot_name(sign_class[i], SIGN_CLASSES),
                "x": float(row[0]),
                "y": float(row[1]),
                "misses": int(round(max(0.0, float(row[2])))),
                "seen_fraction": float(row[3]),
                "state": _slot_name(sign_state[i], LAMP_STATES),
            }
        )

    conf = float(np.mean(np.asarray(lane_valid, dtype=np.float64))) if lane_valid.size else 0.0
    kwargs = {
        "lanes_bev": polylines,
        "lane_conf": conf,
        "tracks": tracks,
        "signs": sign_rows,
        "ego_speed_mps": float(ego_speed_mps),
    }
    extra = set(kwargs) - set(PREDICT_PATH_ARGS)
    if extra:
        raise RuntimeError(f"scene adapter emitted unknown planner args: {sorted(extra)}")
    return kwargs


def assign_min_cost(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Each row and each column is used at most once.

    Rectangular costs assign ``min(rows, cols)`` pairs. Pairs are ``(row, col)``
    in the original matrix. A non-finite cost never enters the search: NaN does
    not compare as less-than, and an all-NaN matrix would not return.
    """
    matrix = np.array(cost, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    finite = np.isfinite(matrix)
    if not bool(finite.any()):
        return []
    if not bool(finite.all()):
        finite_vals = matrix[finite]
        span = float(np.max(finite_vals) - np.min(finite_vals))
        penalty = float(np.max(finite_vals)) + span + 1.0
        if not np.isfinite(penalty):
            return []
        matrix[~finite] = penalty
    n_rows, n_cols = matrix.shape
    transposed = False
    work = matrix
    if n_rows > n_cols:
        work = matrix.T.copy()
        n_rows, n_cols = work.shape
        transposed = True
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    parent = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)
    for i in range(1, n_rows + 1):
        parent[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(parent[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = float(minv[j])
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[parent[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if parent[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            parent[j0] = parent[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs: list[tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if parent[j] == 0:
            continue
        row = int(parent[j] - 1)
        col = j - 1
        if transposed:
            pairs.append((col, row))
        else:
            pairs.append((row, col))
    return pairs


@dataclass
class ScenePrediction:
    lanes: Any
    lane_valid: Any
    curbs: Any
    curb_valid: Any
    objects: Any
    object_class: Any
    object_valid: Any
    signs: Any
    sign_class: Any
    sign_state: Any
    sign_valid: Any

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in OUTPUT_FIELDS}


@dataclass
class SceneState:
    """GRU hidden plus a residual ring.

    ``mask`` stores each token's age in seconds. A slot older than
    ``RESIDUAL_S`` is left out of the skip. The ring length is only the
    storage cap.
    """

    h: Any
    tokens: Any
    mask: Any


def _scene_loss_body(pred: ScenePrediction, target: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is not installed")
    import torch.nn.functional as functional

    device = pred.lanes.device
    dtype = torch.float32
    zero = torch.zeros((), device=device, dtype=dtype)
    points = zero.clone()
    classes = zero.clone()
    lamps = zero.clone()
    valid = zero.clone()

    def bce(score: Any, goal: Any) -> Any:
        # Valid scores are probabilities. Cast out of autocast so the loss stays FP32.
        return functional.binary_cross_entropy(
            score.float().clamp(1e-6, 1.0 - 1e-6), goal.float()
        )

    lane_v = target["lane_valid"].to(device=device, dtype=dtype).reshape(-1)
    valid = valid + bce(pred.lane_valid, lane_v)
    lane_on = lane_v > 0.5
    if bool(lane_on.any()):
        gt_lanes = target["lanes"].to(device=device, dtype=dtype)
        points = points + functional.smooth_l1_loss(pred.lanes[lane_on].float(), gt_lanes[lane_on])

    curb_v = target["curb_valid"].to(device=device, dtype=dtype).reshape(-1)
    valid = valid + bce(pred.curb_valid, curb_v)
    curb_on = curb_v > 0.5
    if bool(curb_on.any()):
        gt_curbs = target["curbs"].to(device=device, dtype=dtype)
        points = points + functional.smooth_l1_loss(pred.curbs[curb_on].float(), gt_curbs[curb_on])

    points, classes, valid = _match_continuous(
        functional,
        pred.objects,
        pred.object_class,
        pred.object_valid,
        target["objects"].to(device=device, dtype=torch.float32),
        target["object_class"].to(device=device, dtype=torch.long),
        target["object_valid"].to(device=device, dtype=dtype).reshape(-1),
        points,
        classes,
        valid,
        bce,
        state_logits=None,
        gt_state=None,
        lamps=None,
    )
    points, classes, valid, lamps = _match_continuous(
        functional,
        pred.signs,
        pred.sign_class,
        pred.sign_valid,
        target["signs"].to(device=device, dtype=torch.float32),
        target["sign_class"].to(device=device, dtype=torch.long),
        target["sign_valid"].to(device=device, dtype=dtype).reshape(-1),
        points,
        classes,
        valid,
        bce,
        state_logits=pred.sign_state,
        gt_state=target["sign_state"].to(device=device, dtype=torch.long),
        lamps=lamps,
    )
    total = points + classes + lamps + valid
    out = {"points": points, "class": classes, "lamps": lamps, "valid": valid, "total": total}
    if set(LOSS_TERMS) - set(out):
        raise RuntimeError("scene loss is missing a term")
    return out


def _match_continuous(
    functional: Any,
    pred_cont: Any,
    pred_class: Any,
    pred_valid: Any,
    gt_cont: Any,
    gt_class: Any,
    gt_valid: Any,
    points: Any,
    classes: Any,
    valid: Any,
    bce: Any,
    *,
    state_logits: Any,
    gt_state: Any,
    lamps: Any,
) -> tuple[Any, Any, Any] | tuple[Any, Any, Any, Any]:
    if torch is None:
        raise ImportError("PyTorch is not installed")
    device = pred_cont.device
    dtype = pred_cont.dtype
    gt_on = torch.where(gt_valid > 0.5)[0]
    pred_xy = pred_cont[:, :2].detach().cpu().numpy()
    pairs: list[tuple[int, int]] = []
    if int(gt_on.numel()) > 0:
        gt_xy = gt_cont[gt_on][:, :2].detach().cpu().numpy()
        cost = np.abs(pred_xy[:, None, :] - gt_xy[None, :, :]).sum(axis=-1)
        pairs = assign_min_cost(cost)
    target_valid = torch.zeros(pred_cont.shape[0], device=device, dtype=dtype)
    if pairs:
        pred_i = torch.tensor([p for p, _g in pairs], device=device, dtype=torch.long)
        local_g = torch.tensor([g for _p, g in pairs], device=device, dtype=torch.long)
        gt_i = gt_on[local_g]
        target_valid[pred_i] = 1
        points = points + functional.smooth_l1_loss(pred_cont[pred_i].float(), gt_cont[gt_i].float())
        classes = classes + functional.cross_entropy(pred_class[pred_i].float(), gt_class[gt_i])
        if state_logits is not None and gt_state is not None and lamps is not None:
            lamps = lamps + functional.cross_entropy(state_logits[pred_i].float(), gt_state[gt_i])
    valid = valid + bce(pred_valid, target_valid)
    if lamps is None:
        return points, classes, valid
    return points, classes, valid, lamps


def scene_loss(pred: ScenePrediction, target: Mapping[str, Any]) -> dict[str, Any]:
    """Smooth-L1 on points, cross-entropy on class and lamps, match on objects and signs.

    Wheel and pedals are not terms.
    """
    return _scene_loss_body(pred, target)


if torch is not None:

    class SharedStem(nn.Module):
        """Shared stride-2 stem: channels 32, 64, 128, 128, then 64 floats."""

        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            cin = 3
            for cout in STEM_CHANNELS:
                # Batch 1 is a legal step, so this is not batch norm.
                # GroupNorm is not added either: the stem stays free of
                # normalization parameters so a paused checkpoint still loads.
                layers.append(nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=False))
                cin = cout
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(STEM_CHANNELS[-1], SECTOR_FEAT)

        def forward(self, x: Any) -> Any:
            feat = self.conv(x).mean(dim=(2, 3))
            return self.fc(feat)

    class SceneNet(nn.Module):
        """Strip in, planner tensors out. Hidden size 512. No steer head."""

        def __init__(self) -> None:
            super().__init__()
            if TOKEN_DIM != len(SECTORS) * SECTOR_FEAT:
                raise RuntimeError("8 sector features must concatenate to the 512-d token")
            self.stem = SharedStem()
            # The dropped tensor is the GRU input. ``state.h`` is not dropped.
            self.gru_in_drop = nn.Dropout(p=GRU_IN_DROPOUT)
            self.gru = nn.GRU(TOKEN_DIM + 1, GRU_HIDDEN, num_layers=1, batch_first=True)
            hidden = GRU_HIDDEN
            for name, out in scene_head_sizes():
                setattr(self, name, nn.Linear(hidden, out))

        def initial_state(self, device: Any = None, dtype: Any = None, batch: int = 1) -> SceneState:
            device = device or torch.device("cpu")
            dtype = dtype or torch.float32
            width = max(1, int(batch))
            empty_age = float(RESIDUAL_S) + 1.0
            return SceneState(
                h=torch.zeros(1, width, GRU_HIDDEN, device=device, dtype=dtype),
                tokens=torch.zeros(width, RESIDUAL_TOKENS, TOKEN_DIM, device=device, dtype=dtype),
                mask=torch.full((width, RESIDUAL_TOKENS), empty_age, device=device, dtype=dtype),
            )

        def forward(self, strip: Any, dt: Any, state: SceneState) -> tuple[ScenePrediction, SceneState]:
            """One strip, seconds since the previous strip, and the GRU state.

            Planner tensors have no batch dimension. A training batch uses
            ``forward_batch`` and keeps the leading dimension.
            """
            image = _strip_nchw(strip)
            pred, new_state = self._forward_batched(image, dt, state)
            return _prediction_at(pred, 0), new_state

        def forward_batch(self, strip: Any, dt: Any, state: SceneState) -> tuple[ScenePrediction, SceneState]:
            """Several strips in one step. Outputs keep the batch dimension."""
            image = _strip_nchw(strip, allow_batch=True)
            return self._forward_batched(image, dt, state)

        def _forward_batched(self, image: Any, dt: Any, state: SceneState) -> tuple[ScenePrediction, SceneState]:
            state = _state_on(state, image)
            token = self._token(image)
            dt_col = _dt_column(dt, image)
            # The step time is the value that arrived. It is not rewritten to a fixed period.
            # Dropout is on the token. dt and the recurrent state are not dropped.
            dropped = self.gru_in_drop(token)
            gru_out, h_new = self.gru(torch.cat([dropped, dt_col], dim=-1), state.h)
            # Same ring update as residual_ages: age by this dt, drop the oldest, append age 0.
            width = int(image.shape[0])
            aged = state.mask + dt_col.reshape(width, 1)
            zeros = torch.zeros(width, 1, device=image.device, dtype=image.dtype)
            new_mask = torch.cat([aged[:, 1:], zeros], dim=1)
            new_tokens = torch.cat([state.tokens[:, 1:], token], dim=1)
            in_span = (new_mask <= float(RESIDUAL_S)).to(dtype=image.dtype)
            denom = in_span.sum(dim=1, keepdim=True).clamp(min=1.0)
            residual = (new_tokens * in_span.unsqueeze(-1)).sum(dim=1) / denom
            feat = gru_out[:, 0, :] + residual
            return _heads_batch(self, feat), SceneState(h=h_new, tokens=new_tokens, mask=new_mask)

        def _token(self, image: Any) -> Any:
            parts = []
            for sec in SECTORS:
                crop = image[:, :, :, sec.x0 : sec.x1]
                if sec.wide:
                    crop = _letterbox_wide(crop)
                parts.append(self.stem(crop))
            return torch.cat(parts, dim=-1).unsqueeze(1)

    def _heads_batch(net: SceneNet, feat: Any) -> ScenePrediction:
        width = int(feat.shape[0])
        return ScenePrediction(
            lanes=net.lane_head(feat).view(width, N_LANES, LANE_POINTS, 2),
            lane_valid=torch.sigmoid(net.lane_valid_head(feat)).view(width, N_LANES),
            curbs=net.curb_head(feat).view(width, N_CURBS, LANE_POINTS, 2),
            curb_valid=torch.sigmoid(net.curb_valid_head(feat)).view(width, N_CURBS),
            objects=net.object_head(feat).view(width, N_OBJECTS, OBJECT_FEAT),
            object_class=net.object_class_head(feat).view(width, N_OBJECTS, len(OBJECT_CLASSES)),
            object_valid=torch.sigmoid(net.object_valid_head(feat)).view(width, N_OBJECTS),
            signs=net.sign_head(feat).view(width, N_SIGNS, SIGN_FEAT),
            sign_class=net.sign_class_head(feat).view(width, N_SIGNS, len(SIGN_CLASSES)),
            sign_state=net.sign_state_head(feat).view(width, N_SIGNS, len(LAMP_STATES)),
            sign_valid=torch.sigmoid(net.sign_valid_head(feat)).view(width, N_SIGNS),
        )

    def _prediction_at(pred: ScenePrediction, index: int) -> ScenePrediction:
        return ScenePrediction(**{name: getattr(pred, name)[index] for name in OUTPUT_FIELDS})

    def _letterbox_wide(crop: Any) -> Any:
        import torch.nn.functional as functional

        _b, _c, src_h, src_w = crop.shape
        nw, nh, x0, y0 = letterbox_rect(int(src_w), int(src_h), 256, CANVAS_H)
        resized = functional.interpolate(crop, size=(nh, nw), mode="bilinear", align_corners=False)
        out = crop.new_zeros(crop.shape[0], crop.shape[1], CANVAS_H, 256)
        out[:, :, y0 : y0 + nh, x0 : x0 + nw] = resized
        return out

    def _strip_nchw(strip: Any, *, allow_batch: bool = False) -> Any:
        if not torch.is_tensor(strip):
            strip = torch.from_numpy(np.ascontiguousarray(strip))
        if strip.dim() == 3 and strip.shape[0] == 3:
            strip = strip.unsqueeze(0)
        elif strip.dim() == 3 and strip.shape[-1] == 3:
            strip = strip.permute(2, 0, 1).unsqueeze(0)
        elif strip.dim() == 4 and strip.shape[-1] == 3:
            strip = strip.permute(0, 3, 1, 2)
        width = int(strip.shape[0]) if strip.dim() == 4 else 0
        spatial_ok = strip.dim() == 4 and int(strip.shape[1]) == 3 and tuple(strip.shape[-2:]) == (CANVAS_H, CANVAS_W)
        batch_ok = width == 1 if not allow_batch else width >= 1
        if not spatial_ok or not batch_ok:
            want = f"Bx3x{CANVAS_H}x{CANVAS_W}" if allow_batch else f"1x3x{CANVAS_H}x{CANVAS_W}"
            raise ValueError(f"strip must be {want}, got {tuple(strip.shape)}")
        if strip.dtype == torch.uint8:
            strip = strip.float() / 255.0
        else:
            strip = strip.float()
            if float(strip.detach().max()) > 1.5:
                strip = strip / 255.0
        return strip

    def _dt_column(dt: Any, image: Any) -> Any:
        width = int(image.shape[0])
        if not torch.is_tensor(dt):
            if isinstance(dt, (list, tuple)):
                dt = torch.tensor(list(dt), device=image.device, dtype=image.dtype)
            else:
                value = 0.0 if dt is None else float(dt)
                dt = torch.full((width,), value, device=image.device, dtype=image.dtype)
        else:
            dt = dt.to(device=image.device, dtype=image.dtype)
        dt = dt.reshape(-1)
        if int(dt.numel()) == 1 and width != 1:
            dt = dt.expand(width)
        if int(dt.numel()) != width:
            raise ValueError(f"dt length {int(dt.numel())} does not match batch {width}")
        return dt.reshape(width, 1, 1)

    def _state_on(state: SceneState, image: Any) -> SceneState:
        width = int(image.shape[0])
        moved = SceneState(
            h=state.h.to(device=image.device, dtype=image.dtype),
            tokens=state.tokens.to(device=image.device, dtype=image.dtype),
            mask=state.mask.to(device=image.device, dtype=image.dtype),
        )
        if int(moved.h.shape[1]) != width or int(moved.tokens.shape[0]) != width or int(moved.mask.shape[0]) != width:
            raise ValueError(f"state batch does not match strips {width}")
        return moved

    class _OnnxScene(nn.Module):
        def __init__(self, net: SceneNet) -> None:
            super().__init__()
            self.net = net

        def forward(self, strip: Any, dt: Any, h: Any, tokens: Any, mask: Any) -> tuple[Any, ...]:
            pred, state = self.net(strip, dt, SceneState(h=h, tokens=tokens, mask=mask))
            fields = tuple(getattr(pred, name) for name in OUTPUT_FIELDS)
            return fields + (state.h, state.tokens, state.mask)

    def _trace_scene_onnx(model: Any, args: tuple[Any, ...], path_str: str) -> None:
        """Trace the FP32 graph. The module and ``args`` are already on CPU."""
        wrapper = _OnnxScene(model).eval()
        output_names = list(OUTPUT_FIELDS) + ["h", "tokens", "mask"]
        export_kwargs = dict(
            input_names=["strip", "dt", "h", "tokens", "mask"],
            output_names=output_names,
            opset_version=17,
        )
        try:
            torch.onnx.export(wrapper, args, path_str, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(wrapper, args, path_str, **export_kwargs)

else:

    class SceneNet:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("PyTorch is not installed")

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ImportError("PyTorch is not installed")

        def forward_batch(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ImportError("PyTorch is not installed")

    def _trace_scene_onnx(model: Any, args: tuple[Any, ...], path_str: str) -> None:
        raise ImportError("PyTorch is not installed")


_SCENE_COUNT: tuple[int, bool] | None = None


def scene_parameter_count(net: Any = None) -> tuple[int, bool]:
    """Return ``(trainable count, approximate)`` for the exported scene net.

    A built module is summed, and that count is exact. With no module and no
    PyTorch, the count is ``architecture_parameter_count`` and approximate is
    True, because the weights do not exist yet.
    """
    global _SCENE_COUNT
    if net is not None:
        return count_module_parameters(net), False
    if _SCENE_COUNT is None:
        if torch is not None:
            _SCENE_COUNT = (count_module_parameters(SceneNet()), False)
        else:
            _SCENE_COUNT = (architecture_parameter_count(), True)
    return _SCENE_COUNT


def cpu_export_inputs(net: Any, zeros: Any) -> tuple[Any, tuple[Any, ...]]:
    """Move ``net`` to CPU and build the tracer examples on CPU.

    Training may have left the module on CUDA. The tracer is given CPU
    examples, so the module has to be on CPU too or the export raises.
    """
    model = net.to("cpu")
    model = model.eval()
    strip = zeros(1, 3, CANVAS_H, CANVAS_W, device="cpu")
    dt = zeros(1, device="cpu")
    state = model.initial_state(device="cpu")
    return model, (strip, dt, state.h, state.tokens, state.mask)


def export_scene_onnx(
    path: Path | str,
    net: Any = None,
    *,
    zeros: Any = None,
    trace: Any = None,
) -> Path:
    """Write the FP32 graph. Mixed precision does not change this file.

    The module and the example inputs are moved to CPU before the tracer runs.
    ``zeros`` and ``trace`` default to PyTorch. A test can pass both and skip
    a GPU.
    """
    path = Path(path)
    if net is None or zeros is None or trace is None:
        if torch is None:
            raise ImportError("PyTorch is not installed")
    if net is None:
        net = SceneNet()
    if zeros is None:
        zeros = torch.zeros
    path.parent.mkdir(parents=True, exist_ok=True)
    model, args = cpu_export_inputs(net, zeros)
    if trace is None:
        trace = _trace_scene_onnx
    trace(model, args, str(path))
    return path
