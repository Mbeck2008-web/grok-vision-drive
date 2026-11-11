"""Labels v4 recorder extras: per-tick signals and traffic brake lights.

Training only. A missing signal API or a traffic Electrics attach that
cannot run is masked. Nothing here invents a lamp.

Lua-spawned ``gvd_traffic_*`` cars are scenetree objects
(``python/collect/spawn_mix.py``), not connected beamngpy Vehicles, so
``sensors.attach("electrics")`` may fail. The row then stores
``brake_lights: null`` and the reason is logged once. Connecting each
traffic vehicle via beamngpy, then attaching Electrics, is a follow-up
for after the IQU45 check shows the field stays null.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python.collect.signal_labels import read_map_node_signals

SIGNALS_FILE = "signals.jsonl"

# One line per failure kind per process. A 2 Hz poll must not repeat it.
_ELECTRICS_LOGGED: set[str] = set()
_FOLLOWUP = (
    "Follow-up: connect each traffic vehicle via beamngpy, then attach Electrics."
)


def signal_tick_row(t: Any, raw: Any, err: str | None) -> dict[str, Any]:
    """One ``signals.jsonl`` row. An error is ``{t, err}`` and has no lamps."""
    if err:
        return {"t": t, "err": str(err)}
    return {"t": t, "signals": raw}


def append_signal_tick(path: Path | str, bng: Any, t: Any) -> None:
    """Append one signal payload. A missing API or a bad write does not raise."""
    try:
        raw, err = read_map_node_signals(bng)
        text = json.dumps(signal_tick_row(t, raw, err), sort_keys=True)
    except Exception as exc:  # noqa: BLE001 — the collect keeps going
        try:
            text = json.dumps({"err": f"signals tick failed: {exc}", "t": t}, sort_keys=True)
        except Exception:
            return
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except Exception:
        return


def _log_once(kind: str, detail: str) -> None:
    if kind in _ELECTRICS_LOGGED:
        return
    _ELECTRICS_LOGGED.add(kind)
    print(f"collect traffic electrics: {detail}", flush=True)


def _object_name(veh: Any, vid: str) -> str:
    for attr in ("name", "vid"):
        value = getattr(veh, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(vid)


def _connected_by_name(bng: Any) -> dict[str, Any]:
    """Non-player beamngpy vehicles keyed by name. The player is never included."""
    api = getattr(bng, "vehicles", None)
    current: dict[str, Any] = {}
    try:
        getter = getattr(api, "get_current", None) if api is not None else None
        cur = getter() if callable(getter) else None
        if isinstance(cur, dict):
            current = {str(key): value for key, value in cur.items() if value is not None}
    except Exception:
        return {}
    player_ids: set[str] = set()
    player = None
    try:
        ident = getattr(api, "get_player_vehicle_id", None) if api is not None else None
        info = ident() if callable(ident) else None
        vid = info.get("vid") or info.get("name") if isinstance(info, dict) else info
        if vid:
            player_ids.add(str(vid))
            player = current.get(str(vid))
    except Exception:
        player = None
    found: dict[str, Any] = {}
    for vid, veh in current.items():
        if vid in player_ids or (player is not None and veh is player):
            continue
        found.setdefault(_object_name(veh, vid), veh)
    return found


def _try_attach(veh: Any) -> str | None:
    """``None`` when Electrics is attached. Otherwise a short reason."""
    sensors = getattr(veh, "sensors", None)
    attach = getattr(sensors, "attach", None) if sensors is not None else None
    if not callable(attach):
        return "no sensors.attach"
    try:
        if "electrics" in sensors:
            return None
    except Exception as exc:  # noqa: BLE001
        return f"electrics lookup failed: {type(exc).__name__}: {exc}"
    try:
        from beamngpy.sensors import Electrics  # type: ignore

        attach("electrics", Electrics())
    except Exception as exc:  # noqa: BLE001 — scenetree objects are not Vehicles
        return f"Electrics attach failed: {type(exc).__name__}: {exc}"
    return None


def _read_brake_lights(veh: Any) -> tuple[Any, str | None]:
    poll = getattr(veh, "poll_sensors", None)
    if callable(poll):
        try:
            poll()
        except Exception as exc:  # noqa: BLE001
            return None, f"electrics poll failed: {type(exc).__name__}: {exc}"
    sensors = getattr(veh, "sensors", None)
    try:
        el = sensors["electrics"] if sensors is not None else None
    except Exception as exc:  # noqa: BLE001
        return None, f"electrics read failed: {type(exc).__name__}: {exc}"
    data = getattr(el, "data", None)
    if not isinstance(data, dict):
        data = el if isinstance(el, dict) else None
    if not isinstance(data, dict):
        return None, "electrics data missing"
    if "brake_lights" in data:
        return data["brake_lights"], None
    if "brakelights" in data:
        return data["brakelights"], None
    return None, "brake_lights absent after Electrics attach"


def _brake_lights_for(name: str, handles: dict[str, Any]) -> Any:
    """This vehicle's lamp, or None when Electrics cannot be read. Never the ego lamp."""
    label = name or "(unnamed)"
    veh = handles.get(name)
    if veh is None:
        _log_once(
            "not_connected",
            f"no connected beamngpy vehicle for scenetree traffic ({label}); "
            f"brake_lights null. {_FOLLOWUP}",
        )
        return None
    try:
        why = _try_attach(veh)
        if why:
            _log_once(
                "attach",
                f"{why} for {label}; brake_lights null. {_FOLLOWUP}",
            )
            return None
        value, missing = _read_brake_lights(veh)
        if missing:
            _log_once("absent", f"{missing} for {label}; brake_lights null.")
            return None
        return value
    except Exception as exc:  # noqa: BLE001
        _log_once(
            "error",
            f"{type(exc).__name__}: {exc} for {label}; brake_lights null. {_FOLLOWUP}",
        )
        return None


def with_brake_lights(cars: list[dict[str, Any]] | None, bng: Any) -> list[dict[str, Any]]:
    """Copy each traffic row and store that vehicle's ``brake_lights`` (or null)."""
    try:
        handles = _connected_by_name(bng)
    except Exception:
        handles = {}
    out: list[dict[str, Any]] = []
    for car in list(cars or []):
        if not isinstance(car, dict):
            continue
        name = str(car.get("name") or "")
        out.append(
            {
                "name": name,
                "pos": car.get("pos"),
                "vel": car.get("vel"),
                "dir": car.get("dir"),
                "brake_lights": _brake_lights_for(name, handles),
            }
        )
    return out
