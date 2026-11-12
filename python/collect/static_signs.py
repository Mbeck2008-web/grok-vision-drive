"""Train-only stop and yield signs from static level meshes.

The live supervisor never imports this module. A collect run asks the
BeamNG scenetree once per map, writes ``static_signs.json``, and
``scripts/export_signal_labels.py`` projects those points into the ego
frame when it builds ``signs.jsonl``. A failed query is an ``absent``
marker. It is never filled in with a guessed sign.

Allowlist status
----------------
Verified, and not used as a mesh stem:

* ``TSStatic.shapeName`` and ``position`` are the static-mesh fields
  (`TSStatic` level-class doc, items.level.json doc).
* ``scenetree.findClassObjects`` / ``findObject`` is the GE lookup used
  for ``TSStatic`` (BeamNG forum "Moding scenario"; ``findObjectById``
  is also tried because some wrappers say the class scan returns ids).
* ``signals.json`` controller type ``signStop`` and instance names like
  ``stop 1`` are the traffic-signals API, already handled by
  ``labels_from_signals``. They are not mesh names, so they are not
  on this list.
* ``/art/shapes/objects/s_sign_drift.dae`` is a real west-coast
  ``TSStatic`` shape (items.level.json in BNG-BeamLegalRacing e790ff2).
  It is a drift marker. The ``s_sign_`` prefix is real; the stop and
  yield stems that copy it are not.

Unverified stems (every entry below is a guess; none was found in this
repo, in beamngpy, or in the BeamNG docs):

* shape or object name: ``s_sign_stop``, ``s_sign_yield``,
  ``stop_sign``, ``stopsign``, ``sign_stop``, ``yield_sign``,
  ``yieldsign``, ``sign_yield``
* shape basename only: ``stop``, ``yield`` (so ``stop.dae`` counts and
  ``stop_light.dae`` does not)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from python.collect.signal_labels import world_to_ego

STATIC_SIGNS_NAME = "static_signs.json"
DEDUPE_M = 2.0
_MATCH_CAP = 4000
_STEM = re.compile(r"^[a-z0-9_]+$")

# stem, class, verified, how. ``how`` is ``shape-or-name`` or ``shape-basename``.
SHAPE_PATTERNS: tuple[tuple[str, str, bool, str], ...] = (
    ("s_sign_stop", "stop_sign", False, "shape-or-name"),
    ("s_sign_yield", "yield", False, "shape-or-name"),
    ("stop_sign", "stop_sign", False, "shape-or-name"),
    ("stopsign", "stop_sign", False, "shape-or-name"),
    ("sign_stop", "stop_sign", False, "shape-or-name"),
    ("yield_sign", "yield", False, "shape-or-name"),
    ("yieldsign", "yield", False, "shape-or-name"),
    ("sign_yield", "yield", False, "shape-or-name"),
    ("stop", "stop_sign", False, "shape-basename"),
    ("yield", "yield", False, "shape-basename"),
)


def allowlist_records() -> list[dict[str, Any]]:
    return [
        {"stem": stem, "cls": cls, "verified": bool(verified), "match": how}
        for stem, cls, verified, how in SHAPE_PATTERNS
    ]


def allowlist_line() -> str:
    parts = [
        f"{stem}->{cls} {how} {'verified' if verified else 'guess'}"
        for stem, cls, verified, how in SHAPE_PATTERNS
    ]
    return "[GVD] stop-sign source allowlist: " + ", ".join(parts)


def _norm(text: Any) -> str:
    return str(text or "").replace("\\", "/").lower().replace("-", "_").replace(" ", "_")


def _basename(text: Any) -> str:
    base = _norm(text).rsplit("/", 1)[-1]
    base = re.sub(r"\.(dae|cdae|dts)$", "", base)
    return re.sub(r"_+\d+$", "", base)


def _bounded(text: str, stem: str) -> bool:
    start = 0
    while True:
        found = text.find(stem, start)
        if found < 0:
            return False
        before = text[found - 1] if found else ""
        after_at = found + len(stem)
        after = text[after_at] if after_at < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = found + 1


def classify_shape(shape: Any, name: Any = "") -> str | None:
    """Map one mesh path or object name onto ``stop_sign`` or ``yield``."""
    shape_n = _norm(shape)
    name_n = _norm(name)
    bases = {_basename(shape), _basename(name)}
    for stem, cls, _verified, how in SHAPE_PATTERNS:
        if how == "shape-basename":
            if _basename(shape) == stem:
                return cls
            continue
        if stem in bases or _bounded(shape_n, stem) or _bounded(name_n, stem):
            return cls
    return None


def _is_tsstatic(obj: dict[str, Any]) -> bool:
    kind = obj.get("class")
    if kind is None:
        kind = obj.get("className")
    if kind is None or str(kind).strip() == "":
        return True
    return str(kind).strip().lower() == "tsstatic"


def _pos(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    raw = obj.get("pos")
    if raw is None:
        raw = obj.get("position")
    if isinstance(raw, dict):
        if raw.get("x") is None or raw.get("y") is None:
            return None
        z = raw.get("z", 0.0)
        try:
            return float(raw["x"]), float(raw["y"]), float(z or 0.0)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2 and raw[0] is not None and raw[1] is not None:
        try:
            z = float(raw[2]) if len(raw) > 2 and raw[2] is not None else 0.0
            return float(raw[0]), float(raw[1]), z
        except (TypeError, ValueError):
            return None
    return None


def _sign_row(cls: str, ego: tuple[float, float]) -> dict[str, Any]:
    return {
        "cls": cls,
        "x": float(ego[0]),
        "y": float(ego[1]),
        "state": "unknown",
        "src": "scenetree",
    }


def labels_from_scenetree(objects: Any, ego_pos: Any, ego_dir: Any) -> list[dict[str, Any]]:
    """Ego-frame stop/yield rows. A non-matching mesh is dropped, not guessed."""
    out: list[dict[str, Any]] = []
    for obj in objects or []:
        if not isinstance(obj, dict) or not _is_tsstatic(obj):
            continue
        cls = classify_shape(obj.get("shapeName") or obj.get("shape") or "", obj.get("name") or "")
        if cls is None:
            continue
        pos = _pos(obj)
        if pos is None:
            continue
        ego = world_to_ego((pos[0], pos[1]), ego_pos, ego_dir)
        if ego is None:
            continue
        out.append(_sign_row(cls, ego))
    out.sort(key=lambda row: (row["y"], row["x"], row["cls"]))
    return out


def world_signs(objects: Any) -> list[dict[str, Any]]:
    """World-frame matches. Positions come from the object, never a default."""
    out: list[dict[str, Any]] = []
    for obj in objects or []:
        if not isinstance(obj, dict) or not _is_tsstatic(obj):
            continue
        shape = str(obj.get("shapeName") or obj.get("shape") or "")
        name = str(obj.get("name") or "")
        cls = classify_shape(shape, name)
        if cls is None:
            continue
        pos = _pos(obj)
        if pos is None:
            continue
        out.append(
            {
                "cls": cls,
                "name": name,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "shapeName": shape,
            }
        )
    out.sort(key=lambda row: (row["pos"][1], row["pos"][0], row["cls"], row["name"]))
    return out


def _lua_set(stems: list[str]) -> str:
    for stem in stems:
        if _STEM.fullmatch(stem) is None:
            raise ValueError(f"refusing stem in lua: {stem!r}")
    return "{" + ", ".join(f"['{stem}']=true" for stem in stems) + "}"


def lua_chunk() -> str:
    """GE Lua: return only TSStatic meshes the allowlist names."""
    long_stems = [stem for stem, _cls, _verified, how in SHAPE_PATTERNS if how == "shape-or-name"]
    exact_stems = [stem for stem, _cls, _verified, how in SHAPE_PATTERNS if how == "shape-basename"]
    return f"""
local long = {_lua_set(long_stems)}
local exact = {_lua_set(exact_stems)}
local function norm(s)
  s = tostring(s or ''):lower():gsub('\\\\','/'):gsub('%-','_'):gsub(' ','_')
  return s
end
local function basename(s)
  s = norm(s)
  s = s:match('([^/]+)$') or s
  s = s:gsub('%.dae$',''):gsub('%.cdae$',''):gsub('%.dts$','')
  s = s:gsub('_+%d+$','')
  return s
end
local function edge(ch)
  if ch == nil or ch == '' then return true end
  return ch:match('[a-z0-9]') == nil
end
local function bounded(text, stem)
  local i = 1
  while true do
    local s, e = string.find(text, stem, i, true)
    if not s then return false end
    local before = s > 1 and text:sub(s - 1, s - 1) or ''
    local after = text:sub(e + 1, e + 1)
    if edge(before) and edge(after) then return true end
    i = s + 1
  end
end
local function hit(shape, name)
  local shape_n, name_n = norm(shape), norm(name)
  local bshape = basename(shape)
  if exact[bshape] then return true end
  local bname = basename(name)
  for stem, _ in pairs(long) do
    if bshape == stem or bname == stem or bounded(shape_n, stem) or bounded(name_n, stem) then
      return true
    end
  end
  return false
end
if not scenetree or not scenetree.findClassObjects then
  return jsonEncode({{ok=false, err='scenetree.findClassObjects unavailable'}})
end
local ids = scenetree.findClassObjects('TSStatic')
if type(ids) ~= 'table' then
  return jsonEncode({{ok=false, err='scenetree.findClassObjects unavailable'}})
end
local out = {{}}
for _, id in pairs(ids) do
  local o = nil
  pcall(function()
    if scenetree.findObject then o = scenetree.findObject(tostring(id)) end
    if not o and scenetree.findObjectById then
      local n = tonumber(id)
      if n then o = scenetree.findObjectById(n) end
    end
  end)
  if o then
    local shape, name = '', ''
    pcall(function()
      if o.getField then shape = o:getField('shapeName', 0) or '' end
      if shape == '' and o.shapeName then shape = o.shapeName end
      if o.getName then name = o:getName() or '' else name = o.name or '' end
    end)
    shape, name = tostring(shape or ''), tostring(name or '')
    if hit(shape, name) then
      local p = nil
      pcall(function() p = o:getPosition() end)
      local x = p and tonumber(p.x) or nil
      local y = p and tonumber(p.y) or nil
      local z = p and tonumber(p.z) or 0
      if x and y then
        out[#out + 1] = {{name=name, shapeName=shape, class='TSStatic', pos={{x, y, z}}}}
      end
    end
  end
  if #out > {_MATCH_CAP} then
    return jsonEncode({{ok=false, err='scenetree match cap'}})
  end
end
return jsonEncode({{ok=true, objects=out}})
"""


def _decode(reply: Any) -> Any:
    if isinstance(reply, (bytes, bytearray)):
        reply = reply.decode("utf-8", "replace")
    if isinstance(reply, str):
        text = reply.strip()
        if not text:
            return None
        try:
            reply = json.loads(text)
        except ValueError:
            return None
    if isinstance(reply, dict) and not any(key in reply for key in ("ok", "objects", "err")):
        for key in ("result", "resp", "data", "return", "ret"):
            if key in reply:
                return _decode(reply[key])
    return reply


def parse_scenetree_reply(reply: Any) -> tuple[list[Any] | None, str | None]:
    """``(objects, None)`` or ``(None, err)``. An empty list is a real zero."""
    blob = _decode(reply)
    if not isinstance(blob, dict):
        return None, "unrecognized scenetree payload"
    if blob.get("ok") is not True:
        return None, str(blob.get("err") or "scenetree query failed")
    raw = blob.get("objects", [])
    if raw is None or raw == {}:
        raw = []
    if not isinstance(raw, list):
        return None, "unrecognized scenetree payload"
    return raw, None


def query_static_signs(bng: Any) -> tuple[list[Any] | None, str | None]:
    """One scenetree read. ``(None, err)`` on failure, never a made-up mesh."""
    from python.collect.level_session import is_session_error, session_text

    fn = None
    for owner in (getattr(bng, "control", None), bng):
        if owner is None:
            continue
        candidate = getattr(owner, "queue_lua_command", None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        return None, "no queue_lua_command"
    chunk = lua_chunk()
    try:
        try:
            reply = fn(chunk, True)
        except TypeError:
            reply = fn(chunk, response=True)
    except Exception as exc:  # noqa: BLE001 — a live dump must not invent labels
        if is_session_error(exc):
            return None, f"scenetree query failed: {session_text(exc)}"
        return None, f"scenetree query failed: {type(exc).__name__}: {exc}"
    if reply is None:
        return None, "scenetree query returned nothing"
    return parse_scenetree_reply(reply)


def _entry(objects: list[Any] | None, err: str | None, map_name: str) -> dict[str, Any]:
    if err or objects is None:
        reason = err or "scenetree query returned nothing"
        print(f"[GVD] stop-sign source: absent on {map_name}: {reason}", flush=True)
        return {"absent": True, "err": reason, "matches": 0, "signs": []}
    signs = world_signs(objects)
    if not signs:
        print(f"[GVD] stop-sign source: 0 matches on {map_name}", flush=True)
        return {"matches": 0, "signs": []}
    print(f"[GVD] stop-sign source: {len(signs)} matches on {map_name}", flush=True)
    return {"matches": len(signs), "signs": signs}


def load_static_catalog(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("maps"), dict):
        return None
    return data


def _write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_static_signs(bng: Any, map_name: str, path: Path) -> dict[str, Any]:
    """Enumerate one map and merge the result into ``static_signs.json``."""
    label = str(map_name or "").strip() or "loaded"
    print(allowlist_line(), flush=True)
    objects, err = query_static_signs(bng)
    catalog = load_static_catalog(Path(path)) or {"maps": {}}
    maps = catalog.get("maps")
    if not isinstance(maps, dict):
        maps = {}
    maps[label] = _entry(objects, err, label)
    catalog["allowlist"] = allowlist_records()
    catalog["maps"] = maps
    catalog["src"] = "scenetree"
    _write_catalog(Path(path), catalog)
    return catalog


def _is_signal(row: dict[str, Any]) -> bool:
    return str(row.get("src") or "traffic_signals") != "scenetree"


def _near(a: dict[str, Any], b: dict[str, Any], radius_m: float) -> bool:
    try:
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) <= float(radius_m)
    except (KeyError, TypeError, ValueError):
        return False


def merge_sign_labels(
    signal_signs: Any,
    static_signs: Any,
    radius_m: float = DEDUPE_M,
) -> list[dict[str, Any]]:
    """Keep signal rows. Drop a scenetree row of the same class within ``radius_m``."""
    kept = [dict(row) for row in signal_signs or [] if isinstance(row, dict)]
    for row in static_signs or []:
        if not isinstance(row, dict) or "x" not in row or "y" not in row:
            continue
        if any(_is_signal(prev) and prev.get("cls") == row.get("cls") and _near(prev, row, radius_m) for prev in kept):
            continue
        kept.append(dict(row))
    kept.sort(key=lambda row: (float(row.get("y") or 0.0), float(row.get("x") or 0.0), str(row.get("cls") or "")))
    return kept


def _state_map(state: Any) -> str:
    if not isinstance(state, dict):
        return ""
    tags = state.get("tags")
    if isinstance(tags, dict) and tags.get("map"):
        return str(tags.get("map") or "").strip()
    return str(state.get("map") or "").strip()


def _catalog_entry(state: Any, maps: dict[str, Any]) -> dict[str, Any] | None:
    name = _state_map(state)
    if name and isinstance(maps.get(name), dict):
        return maps[name]
    if not name and len(maps) == 1:
        only = next(iter(maps.values()))
        return only if isinstance(only, dict) else None
    return None


def _project_world(signs: Any, state: Any) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []
    ego_pos = state.get("pos")
    ego_dir = state.get("dir") if state.get("dir") is not None else state.get("direction")
    out: list[dict[str, Any]] = []
    for item in signs or []:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("cls") or "")
        if cls not in ("stop_sign", "yield"):
            continue
        pos = _pos(item)
        if pos is None:
            continue
        ego = world_to_ego((pos[0], pos[1]), ego_pos, ego_dir)
        if ego is None:
            continue
        out.append(_sign_row(cls, ego))
    return out


def merge_static_into_rows(
    sign_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    catalog: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Project one map's static signs onto each tick and dedupe against signals."""
    maps = catalog.get("maps") if isinstance(catalog, dict) else None
    if not isinstance(maps, dict):
        return list(sign_rows)
    out: list[dict[str, Any]] = []
    for index, sign_row in enumerate(sign_rows):
        state = state_rows[index] if index < len(state_rows) else None
        out.append(_merge_row(sign_row, state, maps))
    return out


def _merge_row(sign_row: Any, state: Any, maps: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sign_row, dict):
        return {"err": "unrecognized signs row"}
    entry = _catalog_entry(state, maps)
    if not isinstance(entry, dict):
        return dict(sign_row)
    if entry.get("absent") or entry.get("err"):
        row = dict(sign_row)
        row["static_signs"] = {"absent": True, "err": str(entry.get("err") or "absent")}
        return row
    world = entry.get("signs") or []
    if not world:
        return dict(sign_row)
    static_labels = _project_world(world, state)
    if not static_labels:
        return dict(sign_row)
    existing = sign_row.get("signs")
    if not isinstance(existing, list):
        existing = []
    row = dict(sign_row)
    if row.get("err"):
        row["signals_err"] = row.pop("err")
    row["signs"] = merge_sign_labels(existing, static_labels)
    return row
