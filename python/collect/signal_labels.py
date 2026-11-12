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


def _tkey(stamp: Any) -> str:
    try:
        return f"{float(stamp):.4f}"
    except (TypeError, ValueError):
        return str(stamp)


def signal_payload(row: Any) -> tuple[Any, str | None]:
    """Split one ``signals.jsonl`` object into ``(raw, err)``.

    ``{t, err}`` is a missing API. ``{t, signals: payload}`` (or ``raw``) is a
    real dump. A row with neither is missing, not an empty network.
    """
    if not isinstance(row, dict):
        return None, "unrecognized traffic_signals payload"
    err = row.get("err")
    if err:
        return None, str(err)
    if "signals" in row:
        return row.get("signals"), None
    if "raw" in row:
        return row.get("raw"), None
    extra = {key: value for key, value in row.items() if key != "t"}
    if not extra:
        return None, "missing signals"
    return extra, None


def signs_for_state(state_row: Any, signal_row: Any) -> dict[str, Any]:
    """One ``signs.jsonl`` object. Missing data is an ``err`` row, never a lamp."""
    t = state_row.get("t") if isinstance(state_row, dict) else None
    if signal_row is None:
        return {"err": "missing signals", "t": t}
    raw, err = signal_payload(signal_row)
    if err:
        return {"err": err, "t": t}
    ego_pos = state_row.get("pos") if isinstance(state_row, dict) else None
    ego_dir = None
    if isinstance(state_row, dict):
        ego_dir = state_row.get("dir") if state_row.get("dir") is not None else state_row.get("direction")
    if _xy(ego_pos) is None or _xy(ego_dir) is None:
        return {"err": "missing ego pose", "t": t}
    return sidecar_row(t=t, raw=raw, err=None, ego_pos=ego_pos, ego_dir=ego_dir)


def label_signs(state_rows: Any, signal_rows: Any) -> list[dict[str, Any]]:
    """Join per-tick ``signals.jsonl`` onto ``state.jsonl`` by ``t``.

    The first signal row for a timestamp wins. A state row with no partner
    is masked. This does not stamp one dump onto every tick.
    """
    index: dict[str, Any] = {}
    for row in signal_rows or []:
        if isinstance(row, dict) and "t" in row:
            index.setdefault(_tkey(row.get("t")), row)
    out: list[dict[str, Any]] = []
    for state in state_rows or []:
        key = _tkey(state.get("t") if isinstance(state, dict) else None)
        out.append(signs_for_state(state, index.get(key)))
    return out
