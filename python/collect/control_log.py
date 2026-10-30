"""Control-input fields for collector rows: what the BeamNG AI asked for and what the car did.

Pure helpers, no BeamNG import. The collector sends ``VE_PROBE`` (vehicle Lua) and
``TRAFFIC_CHUNK`` (game-engine Lua) with ``response=True`` and parses the replies
here. A probe value is ``None`` when the reply lacks it; it is never 0 by default.
The label recorded under ``*_ai`` is the AI controller's command, and that
controller uses information the cameras do not see (road geometry ahead, its
planned path, traffic).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

# Row key -> electrics names it is read from (beamngpy's renamed key first, raw Lua alias next).
ELECTRICS_KEEP: dict[str, tuple[str, ...]] = {
    "gear": ("gear",),
    "gear_index": ("gear_index", "gearIndex"),
    "is_shifting": ("is_shifting", "isShifting"),
    "rpm": ("rpm",),
    "engine_load": ("engine_load", "engineLoad"),
    "engine_throttle": ("engine_throttle", "engineThrottle"),
    "clutch_ratio": ("clutch_ratio", "clutchRatio"),
    "wheelspeed": ("wheelspeed",),
    "airspeed": ("airspeed",),
    "avg_wheel_av": ("avg_wheel_av", "avgWheelAV"),
    "abs_active": ("abs_active", "absActive"),
    "esc_active": ("esc_active", "escActive"),
    "tcs_active": ("tcs_active", "tcsActive"),
    "reverse": ("reverse",),
    "turnsignal": ("turnsignal",),
    "brakelights": ("brakelights", "brake_lights"),
    "ignition": ("ignition",),
    "running": ("running",),
}

_N = "local function n(x) x=tonumber(x);if x==nil or x~=x or x==math.huge or x==-math.huge then return nil end;return x end;"

# One vehicle-Lua call: input.lastInputs (sources + the ai command), raw and smoothed
# accelerations, and body angular rates. Each part is pcall'd so one failure nulls only itself.
VE_PROBE = (
    "local r={};" + _N
    + "pcall(function() local li=input and input.lastInputs;if type(li)=='table' then local s={};"
    "for k,_ in pairs(li) do s[#s+1]=tostring(k) end;table.sort(s);r.src=s;local a=li.ai;"
    "if type(a)=='table' then r.ai={steering=n(a.steering),throttle=n(a.throttle),brake=n(a.brake),"
    "parkingbrake=n(a.parkingbrake),clutch=n(a.clutch)} end end end);"
    "pcall(function() local s=sensors;if s then r.g={x=n(s.gx),y=n(s.gy),z=n(s.gz)};"
    "r.g2={x=n(s.gx2),y=n(s.gy2),z=n(s.gz2)} end end);"
    "pcall(function() r.w={yaw=n(obj:getYawAngularVelocity()),pitch=n(obj:getPitchAngularVelocity()),"
    "roll=n(obj:getRollAngularVelocity())} end);"
    "return jsonEncode(r)"
)

# Once per spawn: steering wheel lock (deg) and wheelbase (m): the spread of the wheel
# hub nodes along the car in the vehicle's own jbeam frame (y is along the car).
VE_META_PROBE = (
    "local r={};"
    "pcall(function() r.lock=tonumber(v.data.input.steeringWheelLock) end);"
    "pcall(function() local lo,hi=nil,nil;for _,w in pairs(v.data.wheels or {}) do "
    "local nd=v.data.nodes[w.node1];local y=nd and (nd.pos and nd.pos.y or nd.posY);"
    "if y then lo=math.min(lo or y,y);hi=math.max(hi or y,y) end end;"
    "if lo and hi then r.wb=hi-lo end end);"
    "return jsonEncode(r)"
)

TRAFFIC_EVERY_S = 0.5
TRAFFIC_NAMES = ("gvd_traffic_0", "gvd_traffic_1")
TRAFFIC_CHUNK = (
    "local out={};"
    + "".join(
        f"pcall(function() local o=scenetree.findObject('{name}');if o then "
        "local p=o:getPosition();local v=o:getVelocity();local d=o:getDirectionVector();"
        f"out[#out+1]={{name='{name}',pos={{p.x,p.y,p.z}},vel={{v.x,v.y,v.z}},dir={{d.x,d.y,d.z}}}} end end);"
        for name in TRAFFIC_NAMES
    )
    + "return jsonEncode(out)"
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _reply(reply: Any) -> Any:
    """The decoded Lua reply: a dict/list as sent, a JSON string decoded, anything else None."""
    if isinstance(reply, (bytes, bytearray)):
        reply = reply.decode("utf-8", "replace")
    if isinstance(reply, str):
        try:
            return json.loads(reply)
        except ValueError:
            return None
    return reply if isinstance(reply, (dict, list)) else None


def _part(blob: Any, key: str) -> Mapping[str, Any]:
    got = blob.get(key) if isinstance(blob, dict) else None
    return got if isinstance(got, dict) else {}


def electrics_subset(electrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whitelisted electrics, under the row's names. Missing channels are left out."""
    if not isinstance(electrics, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, names in ELECTRICS_KEEP.items():
        for name in names:
            if name in electrics:
                value = electrics[name]
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                out[key] = value
                break
    return out


def probe_nulls() -> dict[str, Any]:
    """Probe fields for a row without a probe (or with a failed one)."""
    return parse_ve_probe(None)


def parse_ve_probe(reply: Any) -> dict[str, Any]:
    blob = _reply(reply)
    ai, g, g2, w = (_part(blob, k) for k in ("ai", "g", "g2", "w"))
    src = blob.get("src") if isinstance(blob, dict) else None
    return {
        "steer_ai": _num(ai.get("steering")),
        "throttle_ai": _num(ai.get("throttle")),
        "brake_ai": _num(ai.get("brake")),
        "parkingbrake_ai": _num(ai.get("parkingbrake")),
        "clutch_ai": _num(ai.get("clutch")),
        "imu": {
            "gx": _num(g.get("x")),
            "gy": _num(g.get("y")),
            "gz": _num(g.get("z")),
            "g_smoothed": {"x": _num(g2.get("x")), "y": _num(g2.get("y")), "z": _num(g2.get("z"))},
            "yaw_rate": _num(w.get("yaw")),
            "pitch_rate": _num(w.get("pitch")),
            "roll_rate": _num(w.get("roll")),
        },
        "ai_in_src": [str(s) for s in src] if isinstance(src, list) else None,
    }


def parse_ve_meta(reply: Any) -> dict[str, float | None]:
    blob = _reply(reply)
    blob = blob if isinstance(blob, dict) else {}
    return {"steer_lock_deg": _num(blob.get("lock")), "wheelbase_m": _num(blob.get("wb"))}


def _vec(value: Any) -> list[float | None] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return [_num(v) for v in value[:3]]


def parse_traffic(reply: Any) -> list[dict[str, Any]] | None:
    """Traffic cars found by name. ``[]`` when none exist; ``None`` when the call gave nothing usable."""
    blob = _reply(reply)
    if isinstance(blob, dict) and not blob:
        return []  # an empty Lua table encodes as {}
    if not isinstance(blob, list):
        return None
    cars = []
    for car in blob:
        if isinstance(car, dict):
            cars.append({"name": str(car.get("name") or ""), "pos": _vec(car.get("pos")),
                         "vel": _vec(car.get("vel")), "dir": _vec(car.get("dir"))})
    return cars


_SET = re.compile(r'^\s*ai\.(setMode|setSpeedMode|setSpeed|setLine)\((.*)\)\s*$', re.S)


def ai_cmd_state(state: Mapping[str, Any] | None = None, command: str | None = None) -> dict[str, Any]:
    """The collector's last AI commands, updated by one vehicle-Lua line it sent."""
    out = dict(state) if state else {"mode": None, "set_speed_mps": None, "speed_mode": None, "line_active": False}
    match = _SET.match(str(command or ""))
    if not match:
        return out
    name, arg = match.group(1), match.group(2).strip()
    if name == "setMode":
        out["mode"] = arg.strip("\"'")
        out["line_active"] = False
    elif name == "setSpeedMode":
        out["speed_mode"] = arg.strip("\"'")
    elif name == "setSpeed":
        out["set_speed_mps"] = _num(arg)
    else:
        out["line_active"] = True
    return out


def road_tag(hit: Any, lanes_here: int | None) -> dict[str, Any] | None:
    """Road fields the collector already has for this tick. Fields it does not compute are left out."""
    if hit is None:
        return None
    out: dict[str, Any] = {}
    if getattr(hit, "road_id", None) is not None:
        out["road_id"] = str(hit.road_id)
    if lanes_here is not None:
        out["lanes_here"] = int(lanes_here)
    width = _num(getattr(hit, "lane_width_m", None))
    if width is not None:
        out["lane_width_m"] = width
    return out


def vehicle_fields(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Tick-rate fields from the State and electrics the collector already polls (no game call)."""
    state = state if isinstance(state, Mapping) else {}
    electrics = state.get("electrics") if isinstance(state.get("electrics"), Mapping) else {}
    return {
        "steer_angle_deg": _num(electrics.get("steering")),
        "throttle_applied": _num(electrics.get("throttle")),
        "brake_applied": _num(electrics.get("brake")),
        "parkingbrake": _num(electrics.get("parkingbrake_input")),
        "clutch": _num(electrics.get("clutch_input")),
        "veh": electrics_subset(electrics),
        "vel": _vec(state.get("vel")),
    }
