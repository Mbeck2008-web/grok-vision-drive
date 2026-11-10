"""Train-only sign and lamp labels from beamngpy traffic_signals.

Nothing here is drawn or sent to the planner. Missing APIs and unknown
lamps stay masked; they are never filled in.
"""
from __future__ import annotations

import math
import re
from typing import Any

SIGN_CLASSES = ("stop_sign", "yield", "traffic_light", "speed_limit")
LAMP_STATES = ("unknown", "red", "yellow", "green")

_CLASS_ALIASES = {
    "stop": "stop_sign",
    "stop_sign": "stop_sign",
    "yield": "yield",
    "yield_sign": "yield",
    "light": "traffic_light",
    "trafficlight": "traffic_light",
    "traffic_light": "traffic_light",
    "speed": "speed_limit",
    "speed_sign": "speed_limit",
    "speed_limit": "speed_limit",
}
_LAMP_ALIASES = {
    "unknown": "unknown",
    "red": "red",
    "yellow": "yellow",
    "amber": "yellow",
    "green": "green",
    "r": "red",
    "y": "yellow",
    "g": "green",
}


def _norm_cls(raw: Any) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    direct = _CLASS_ALIASES.get(key)
    if direct is not None:
        return direct
    # Instance names like "trafficLight 1", "trafficLight1", "stop 2".
    stripped = re.sub(r"[\s_]*\d+$", "", key).strip("_")
    return _CLASS_ALIASES.get(stripped)


def _norm_lamp(raw: Any, *, cls: str) -> str:
    if cls != "traffic_light":
        return "unknown"
    key = str(raw or "").strip().lower()
    return _LAMP_ALIASES.get(key, "unknown")


def read_map_node_signals(bng: Any) -> tuple[Any, str | None]:
    """Call ``traffic_signals.get_map_node_signals`` if it exists."""
    api = getattr(bng, "traffic_signals", None)
    if api is None:
        return None, "missing traffic_signals"
    fn = getattr(api, "get_map_node_signals", None)
    if fn is None or not callable(fn):
        return None, "missing get_map_node_signals"
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 — a live dump must not invent labels
        return None, f"get_map_node_signals failed: {exc}"


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        if value.get("x") is None or value.get("y") is None:
            return None
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None:
        return float(value[0]), float(value[1])
    return None


def world_to_ego(
    world_xy: tuple[float, float],
    ego_pos: Any,
    ego_dir: Any,
) -> tuple[float, float] | None:
    """Vehicle frame: x right, y forward. Same basis as painted_lanes."""
    pos = _xy(ego_pos)
    direction = _xy(ego_dir)
    if pos is None or direction is None:
        return None
    fx, fy = direction
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        return None
    fx, fy = fx / norm, fy / norm
    rx, ry = fy, -fx
    dx, dy = world_xy[0] - pos[0], world_xy[1] - pos[1]
    return dx * rx + dy * ry, dx * fx + dy * fy


def _looks_like_entry(item: Any) -> bool:
    return isinstance(item, dict) and any(
        key in item
        for key in ("instance", "name", "cls", "class", "type", "pos", "position", "state", "action")
    )


def _recognized_item(item: Any) -> bool:
    if _looks_like_entry(item):
        return True
    if isinstance(item, dict):
        return all(_recognized_item(value) for value in item.values())
    if isinstance(item, (list, tuple)):
        return all(_recognized_item(value) for value in item)
    return False


def recognized_payload(raw: Any) -> bool:
    """True for an empty or documented traffic_signals dump, else False."""
    if raw is None:
        return True
    if isinstance(raw, (list, tuple)):
        return len(raw) == 0 or all(_recognized_item(item) for item in raw)
    if isinstance(raw, dict):
        return len(raw) == 0 or _recognized_item(raw)
    return False


def _as_signal_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if any(k in raw for k in ("pos", "position", "cls", "name", "type")):
            return [raw]
        out: list[Any] = []
        for value in raw.values():
            out.extend(_as_signal_list(value))
        return out
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            out.extend(_as_signal_list(item))
        return out
    return []


def labels_from_signals(raw: Any, ego_pos: Any, ego_dir: Any) -> list[dict[str, Any]]:
    """Ego-frame signs. Unknown class or pose is dropped, not guessed."""
    out: list[dict[str, Any]] = []
    for item in _as_signal_list(raw):
        if not isinstance(item, dict):
            continue
        cls = _norm_cls(
            item.get("cls")
            or item.get("class")
            or item.get("instance")
            or item.get("name")
            or item.get("type")
        )
        if cls is None or cls not in SIGN_CLASSES:
            continue
        pos = _xy(item.get("pos") or item.get("position") or item.get("xyz"))
        if pos is None:
            continue
        ego = world_to_ego(pos, ego_pos, ego_dir)
        if ego is None:
            continue
        out.append(
            {
                "cls": cls,
                "x": float(ego[0]),
                "y": float(ego[1]),
                "state": _norm_lamp(item.get("state") or item.get("lamp") or item.get("signal"), cls=cls),
                "src": "traffic_signals",
            }
        )
    out.sort(key=lambda row: (row["y"], row["x"], row["cls"]))
    return out


def sidecar_row(
    *,
    t: Any,
    raw: Any,
    err: str | None,
    ego_pos: Any,
    ego_dir: Any,
) -> dict[str, Any]:
    if err:
        return {"err": str(err), "t": t}
    if not recognized_payload(raw):
        return {"err": "unrecognized traffic_signals payload", "t": t}
    return {"t": t, "signs": labels_from_signals(raw, ego_pos, ego_dir)}
