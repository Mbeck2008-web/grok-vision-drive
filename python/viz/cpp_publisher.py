"""Publish perception-only scene messages to the C++ visualizer (``gvd_viz``).

Spec: docs/specs/viz_cpp.md. The supervisor calls :meth:`CppVizPublisher.publish`
once per tick. It never blocks and never raises:

* one UDP datagram to 127.0.0.1 on a non-blocking socket; a full socket
  buffer, a missing listener, or any OS error drops that frame (counted);
* the message is built from a whitelist of perception fields only. Ground
  truth (``path_world``, ``debug_gt`` overlays, ``*_gt`` / ``gt_*`` keys,
  labeler output) is never read, so it cannot reach the renderer;
* a missing ``gvd_viz`` executable is reported once and the publisher keeps
  sending (harmless) so a renderer started later still gets frames.

Schema v2 (``{"v": 2, "type": "gvd_scene", ...}``) is mirrored by
``cpp/viz/src/scene.cpp``. Bump ``SCHEMA_VERSION`` on any breaking change.
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
DEFAULT_PORT = 47800
# Loopback UDP payload cap (65507 = 65535 − 8 byte UDP header − 20 byte min IPv4).
UDP_MAX_PAYLOAD = 65507
# Stay under the UDP cap; the test pins this against UDP_MAX_PAYLOAD, not just itself.
MAX_DATAGRAM = 60000
MAX_TRACKS = 64
MAX_SIGNS = 16
MAX_LANES = 16
MAX_EDGES = 8
MAX_POINTS = 120
MAX_DOTS = 1024
# Supervisor ticks are ~15-30 Hz; never send faster than the renderer's 60 fps.
MIN_INTERVAL_S = 1.0 / 60.0

ROOT = Path(__file__).resolve().parents[2]

from python.viz.occ_out_v2_contract import LANE_PAINTS, OCC_OUT_VERSION  # noqa: E402
# Strict v1 enums (on-wire). Exact strings only; anything else is dropped or
# replaced with the documented default. See docs/specs/viz_cpp.md.
PAINT_VALUES = frozenset({"white", "yellow"})
KIND_VALUES = frozenset({"detected", "predicted"})
AEB_VALUES = frozenset({"off", "warn", "brake"})
TRACK_CLS = frozenset({
    "vehicle", "car", "pedestrian", "ped", "person", "bicycle", "bike", "cyclist",
    "motorcycle", "truck", "bus", "trailer", "obstacle", "parked", "debris", "cone",
})
SIGN_CLS = frozenset({"traffic_light", "stop_sign", "pole"})
SIGN_STATE = frozenset({"red", "yellow", "green", "unknown"})
# Top-level message keys. Tests pin this set so a new field is a schema decision.
MESSAGE_KEYS = frozenset(
    {"v", "type", "seq", "t", "ego", "lead_id", "aeb", "lanes", "edges", "dots", "tracks", "signs", "path", "path_conf"}
)
DOT_CLS = frozenset({"white", "yellow", "curb", "edge", "path"})
# Aliases accepted on the state side only; the wire always uses DOT_CLS.
DOT_CLS_ALIASES = {
    "white": "white", "lane": "white", "white_solid": "white", "white_dashed": "white",
    "stop_line": "white",
    "yellow": "yellow", "yellow_solid": "yellow", "yellow_dashed": "yellow",
    "curb": "curb",
    "edge": "edge", "road_edge": "edge",
    "path": "path",
}


def _f(v: Any, nd: int = 2) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd)


def _pts(src: Any, limit: int = MAX_POINTS) -> list[list[float]]:
    if isinstance(src, dict):
        src = src.get("points") or src.get("pts") or []
    out: list[list[float]] = []
    if not isinstance(src, (list, tuple)):
        return out
    for p in src:
        if isinstance(p, dict):
            x, y = _f(p.get("x")), _f(p.get("y"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _f(p[0]), _f(p[1])
        else:
            continue
        if x is None or y is None:
            continue
        out.append([x, y])
    if len(out) > limit:
        step = len(out) / float(limit)
        keep = [out[int(i * step)] for i in range(limit - 1)] + [out[-1]]
        out = keep
    return out


def lane_paint(record: Any) -> str:
    """``"yellow"`` only when ``paint`` is exactly ``"yellow"``; else ``"white"``.

    Exact key and exact value. Aliases (``paint_color``, ``colour``, ``line_color``,
    ``color``, ``material``, ``yellow: true``) and substring matches
    (``"not_yellow"``) are rejected. Live ``lanes_ext`` has no paint field yet,
    so live lanes are white. Index, side and kind never pick a colour.
    """
    if not isinstance(record, dict):
        return "white"
    return "yellow" if record.get("paint") == "yellow" else "white"


def _int(v: Any) -> int | None:
    """JSON-integer only: reject bools, floats (no truncation) and strings."""
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return int(v)


def _tracks(src: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for tr in src:
        if not isinstance(tr, dict):
            continue
        x, y = _f(tr.get("x")), _f(tr.get("y"))
        if x is None or y is None:
            continue
        cls = tr.get("class") if tr.get("class") is not None else tr.get("cls")
        if cls is None or cls == "":
            cls = "vehicle"
        cls = str(cls)
        if cls not in TRACK_CLS:
            continue
        row: dict[str, Any] = {"cls": cls, "x": x, "y": y}
        tid = _int(tr.get("id"))
        if tid is not None:
            row["id"] = tid
        yaw = _f(tr.get("yaw"), 3)
        if yaw is not None:
            row["yaw"] = yaw
        # Supervisor-state aliases (length/width/height/speed_mps) map to the
        # on-wire keys l/w/h/v. The datagram itself only ever has l/w/h/v.
        for src_key, key in (("l", "l"), ("w", "w"), ("h", "h"), ("v", "v"),
                             ("length", "l"), ("width", "w"), ("height", "h"), ("speed_mps", "v")):
            if key in row:
                continue
            val = _f(tr.get(src_key))
            if val is not None:
                row[key] = val
        # Live tracks do not carry a lead flag; lead is indicated by lead_id.
        # If the state (or occ_to_viz) sets lead=true, pass it through.
        if tr.get("lead") is True:
            row["lead"] = True
        out.append(row)
        if len(out) >= MAX_TRACKS:
            break
    return out


def _signs(src: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for s in src:
        if not isinstance(s, dict):
            continue
        x, y = _f(s.get("x")), _f(s.get("y"))
        cls = s.get("cls") if s.get("cls") is not None else s.get("class")
        cls = str(cls) if cls else ""
        if x is None or y is None or cls not in SIGN_CLS:
            continue
        raw_state = s.get("state")
        if raw_state is None or raw_state == "":
            state = "unknown"
        else:
            state = str(raw_state)
            if state not in SIGN_STATE:
                continue
        out.append({"cls": cls, "x": x, "y": y, "state": state})
        if len(out) >= MAX_SIGNS:
            break
    return out


def _list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []



def _seq(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            return list(tolist())
        except (TypeError, ValueError):
            return []
    return []


def _as_bool(v: Any) -> bool | None:
    """Accept a real bool (or a numpy bool scalar). Strings and 0/1 are invalid."""
    if type(v) is bool:
        return v
    item = getattr(v, "item", None)
    if callable(item):
        try:
            x = item()
        except (TypeError, ValueError):
            return None
        if type(x) is bool:
            return x
    return None


def _as_int(v: Any) -> int | None:
    """JSON / numpy integer, not bool, not truncated float."""
    if type(v) is bool:
        return None
    if type(v) is int:
        return v
    item = getattr(v, "item", None)
    if callable(item):
        try:
            x = item()
        except (TypeError, ValueError):
            return None
        if type(x) is int:
            return x
        if type(x) is bool:
            return None
    return None


def _ijk(raw: Any) -> list[int] | None:
    seq = _seq(raw)
    if len(seq) < 3:
        return None
    out = [_as_int(seq[0]), _as_int(seq[1]), _as_int(seq[2])]
    if any(x is None for x in out):
        return None
    return [int(out[0]), int(out[1]), int(out[2])]  # type: ignore[arg-type]


def _occ_v2(occ: Any) -> bool:
    if not isinstance(occ, dict):
        return False
    v = _as_int(occ.get("v"))
    typ = occ.get("type")
    return v == OCC_OUT_VERSION and typ == "occ_out"


def _dot_cls(raw: Any) -> str | None:
    """Map a paint / edge label to an on-wire DotCls. Integer occ class ids
    (other/drivable/vehicle) are not paint and return None."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None
    key = str(raw).strip().lower()
    return DOT_CLS_ALIASES.get(key)


def _one_dot(rec: Any, *, default_cls: str = "white", default_prob: float | None = None,
             default_obs: bool | None = None, default_id: int | None = None) -> dict[str, Any] | None:
    if not isinstance(rec, dict):
        return None
    x, y = _f(rec.get("x")), _f(rec.get("y"))
    if x is None or y is None:
        return None
    cls = _dot_cls(rec.get("cls") if rec.get("cls") is not None else rec.get("class")) or default_cls
    if cls not in DOT_CLS:
        return None
    if "prob" in rec:
        pr = _f(rec.get("prob"), 3)
    elif "conf" in rec:
        pr = _f(rec.get("conf"), 3)
    else:
        pr = None
    if pr is None:
        if default_prob is None:
            return None
        pr = default_prob
    pr = min(1.0, max(0.0, pr))
    if "observed" in rec:
        observed = _as_bool(rec.get("observed"))
        if observed is None:
            return None
    elif default_obs is None:
        return None
    else:
        observed = default_obs
    row: dict[str, Any] = {"x": x, "y": y, "cls": cls, "prob": pr, "observed": observed}
    tid = _int(rec.get("id"))
    if tid is None:
        tid = default_id
    if tid is not None:
        row["id"] = tid
    ijk = _ijk(rec.get("ijk"))
    if ijk is not None:
        row["ijk"] = ijk
    return row


def _dots_from_list(src: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in _list(src):
        d = _one_dot(rec)
        if d is not None:
            out.append(d)
        if len(out) >= MAX_DOTS:
            break
    return out


def _dots_from_vertices(pts: list[list[float]], cls: str, prob: float, observed: bool,
                        id0: int | None = None) -> list[dict[str, Any]]:
    """One dot per vertex. Never interpolates mid-segment samples."""
    out: list[dict[str, Any]] = []
    for i, p in enumerate(pts):
        if len(p) < 2:
            continue
        row: dict[str, Any] = {"x": p[0], "y": p[1], "cls": cls, "prob": float(prob), "observed": bool(observed)}
        if id0 is not None:
            row["id"] = id0 + i
        out.append(row)
    return out


def _dots_from_occ_out(occ: Any) -> list[dict[str, Any]]:
    """Dots from a canonical occ_out v2 message.

    Red curb / road-edge beads come only from ``edges[]``. Paint colour comes
    only from ``lanes[].paint``. ``v`` must be 2; anything else is rejected.
    Occupancy cells (cls in other/drivable/vehicle) are not paint and are not
    drawn as beads. A missing or invalid per-point ``conf``/``prob`` or
    ``observed`` drops that point instead of drawing it at full brightness.
    """
    if not _occ_v2(occ):
        return []
    out: list[dict[str, Any]] = []

    def _pt_xy(p: Any) -> tuple[float, float] | None:
        if isinstance(p, dict):
            x, y = _f(p.get("x")), _f(p.get("y"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _f(p[0]), _f(p[1])
        else:
            return None
        if x is None or y is None:
            return None
        return x, y

    def _push(x: float, y: float, cls: str, pr: float, obs: bool, ijk=None) -> bool:
        rec: dict[str, Any] = {"x": x, "y": y, "cls": cls, "prob": min(1.0, max(0.0, pr)), "observed": obs}
        if ijk is not None:
            rec["ijk"] = ijk
        out.append(rec)
        return len(out) >= MAX_DOTS

    for e in _seq(occ.get("edges")):
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind == "curb":
            cls = "curb"
        elif kind == "road_edge":
            cls = "edge"
        else:
            continue
        pts = _seq(e.get("pts"))
        confs = _seq(e.get("conf"))
        obs = _seq(e.get("observed"))
        for j, p in enumerate(pts):
            xy = _pt_xy(p)
            if xy is None:
                continue
            if j >= len(confs):
                continue
            pr = _f(confs[j], 3)
            if pr is None:
                continue
            if j >= len(obs):
                continue
            ob = _as_bool(obs[j])
            if ob is None:
                continue
            if _push(xy[0], xy[1], cls, pr, ob):
                return out

    for ln in _seq(occ.get("lanes")):
        if not isinstance(ln, dict):
            continue
        paint = ln.get("paint")
        if paint not in LANE_PAINTS:
            continue
        pts = _seq(ln.get("pts"))
        confs = _seq(ln.get("conf"))
        obs = _seq(ln.get("observed"))
        for j, p in enumerate(pts):
            xy = _pt_xy(p)
            if xy is None:
                continue
            if j >= len(confs):
                continue
            pr = _f(confs[j], 3)
            if pr is None:
                continue
            if j >= len(obs):
                continue
            ob = _as_bool(obs[j])
            if ob is None:
                continue
            if _push(xy[0], xy[1], str(paint), pr, ob):
                return out
    return out


def _thin_dots(dots: list[dict[str, Any]], limit: int, origin: tuple[float, float] = (0.0, 0.0)) -> list[dict[str, Any]]:
    """Drop farthest / lowest-prob dots. Never invent mid-segment samples."""
    if len(dots) <= limit:
        return dots
    ox, oy = origin

    def key(d: dict[str, Any]) -> tuple[float, float]:
        dx, dy = float(d["x"]) - ox, float(d["y"]) - oy
        return (-float(d.get("prob", 1.0)), dx * dx + dy * dy)

    ranked = sorted(dots, key=key)
    return ranked[:limit]


def _dots_to_world(dots: list[dict[str, Any]], ex: float, ey: float, yaw: float) -> list[dict[str, Any]]:
    """Supervisor / occ_out points are ego-frame; the wire uses world when a pose is set."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for d in dots:
        x, y = float(d["x"]), float(d["y"])
        dx = x * s + y * c
        dy = -x * c + y * s
        rec = dict(d)
        rec["x"], rec["y"] = round(ex + dx, 2), round(ey + dy, 2)
        out.append(rec)
    return out


def collect_dots(st: dict[str, Any]) -> list[dict[str, Any]]:

    """Independent dots from state. Preference: dots, then occ_out, then vertices."""
    if st.get("dots") is not None:
        return _thin_dots(_dots_from_list(st.get("dots")), MAX_DOTS)
    if st.get("occ_out") is not None:
        return _thin_dots(_dots_from_occ_out(st.get("occ_out")), MAX_DOTS)
    out: list[dict[str, Any]] = []
    for ln in _list(st.get("lanes_ext")):
        paint = lane_paint(ln)
        kind = "detected"
        if isinstance(ln, dict) and ln.get("kind") is not None:
            raw = str(ln.get("kind"))
            if raw not in KIND_VALUES:
                continue
            kind = raw
        pts = _pts(ln)
        out.extend(_dots_from_vertices(pts, paint, 0.45 if kind == "predicted" else 1.0, kind != "predicted"))
    for ed in _list(st.get("road_edges")):
        kind = "detected"
        if isinstance(ed, dict) and ed.get("kind") is not None:
            raw = str(ed.get("kind"))
            if raw not in KIND_VALUES:
                continue
            kind = raw
        cls = "curb" if (isinstance(ed, dict) and str(ed.get("cls") or ed.get("class") or "").lower() in {"curb"}) else "edge"
        pts = _pts(ed)
        out.extend(_dots_from_vertices(pts, cls, 0.6 if kind == "predicted" else 1.0, kind != "predicted"))
    path_pts = _pts(st.get("path_ego"))
    pc = _f(st.get("path_conf"))
    conf = 1.0 if pc is None else min(1.0, max(0.0, pc))
    if path_pts:
        out.extend(_dots_from_vertices(path_pts, "path", conf, True))
    return _thin_dots(out, MAX_DOTS)


def build_scene_message(state: dict[str, Any], *, seq: int = 0, t: float | None = None) -> dict[str, Any]:
    """Scene message from the supervisor state. Perception fields only.

    Reads: ``engaged``, ``ego.{speed_mps,x,y,yaw,yaw_rate,yaw_rate_rps}``, ``planner.{target_v,cipv_id,aeb}``,
    ``path_ego``, ``path_conf``, ``lanes_ext``, ``road_edges``, ``tracks``,
    ``signs``, ``dots``, ``occ_out``. Nothing else. Each lane / curb / edge
    vertex is forwarded as an independent dot; the publisher never resamples
    a polyline into mid-segment beads.
    """
    st = state if isinstance(state, dict) else {}
    ego = st.get("ego") if isinstance(st.get("ego"), dict) else {}
    pl = st.get("planner") if isinstance(st.get("planner"), dict) else {}
    ego_msg: dict[str, Any] = {"engaged": bool(st.get("engaged")), "speed_mps": _f(ego.get("speed_mps")) or 0.0}
    # yaw_rate is optional; used to ego-motion-compensate dots when there is no world pose.
    yr = _f(ego.get("yaw_rate_rps") if ego.get("yaw_rate_rps") is not None else ego.get("yaw_rate"), 4)
    if yr is not None:
        ego_msg["yaw_rate_rps"] = yr
    # Live ego x/y/yaw is not wired up yet (UNPROVEN). Only emit a pose when
    # all three fields are present and finite — nothing is invented.
    ex, ey, eyaw = _f(ego.get("x"), 3), _f(ego.get("y"), 3), _f(ego.get("yaw"), 4)
    have_pose = ex is not None and ey is not None and eyaw is not None
    if have_pose:
        ego_msg["x"], ego_msg["y"], ego_msg["yaw"] = ex, ey, eyaw
    dots = collect_dots(st)
    if have_pose:
        dots = _dots_to_world(dots, float(ex), float(ey), float(eyaw))
    msg: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "type": "gvd_scene",
        "seq": int(seq),
        "t": round(float(time.monotonic() if t is None else t), 4),
        "ego": ego_msg,
        "lanes": [],
        "edges": [],
        "dots": dots,
        "tracks": _tracks(_list(st.get("tracks"))),
        "signs": _signs(_list(st.get("signs"))),
        "path": {"pts": _pts(st.get("path_ego"))},
    }
    aeb = pl.get("aeb")
    msg["aeb"] = aeb if aeb in AEB_VALUES else "off"
    pc = _f(st.get("path_conf"))
    msg["path_conf"] = 1.0 if pc is None else min(1.0, max(0.0, pc))
    set_v = _f(pl.get("target_v"))
    if set_v is not None:
        msg["ego"]["set_mps"] = set_v
    lead = _int(pl.get("cipv_id"))
    if lead is not None:
        msg["lead_id"] = lead
    for ln in _list(st.get("lanes_ext")):
        if len(msg["lanes"]) >= MAX_LANES:
            break
        pts = _pts(ln)
        if len(pts) < 2:
            continue
        kind = "detected"
        if isinstance(ln, dict) and ln.get("kind") is not None:
            raw_kind = str(ln.get("kind"))
            if raw_kind not in KIND_VALUES:
                continue
            kind = raw_kind
        paint = lane_paint(ln)
        if isinstance(ln, dict) and ln.get("paint") is not None and ln.get("paint") not in PAINT_VALUES:
            continue
        msg["lanes"].append({"pts": pts, "paint": paint, "kind": kind})
    for ed in _list(st.get("road_edges")):
        if len(msg["edges"]) >= MAX_EDGES:
            break
        pts = _pts(ed)
        if len(pts) < 2:
            continue
        kind = "detected"
        if isinstance(ed, dict) and ed.get("kind") is not None:
            raw_kind = str(ed.get("kind"))
            if raw_kind not in KIND_VALUES:
                continue
            kind = raw_kind
        msg["edges"].append({"pts": pts, "kind": kind})
    return msg


def encode_message(msg: dict[str, Any], max_bytes: int = MAX_DATAGRAM) -> bytes | None:
    """Compact UTF-8 JSON that fits one datagram, thinning points if needed."""
    max_bytes = min(int(max_bytes), UDP_MAX_PAYLOAD)
    data = json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) <= max_bytes:
        return data
    thin = json.loads(data)
    ox = float(((thin.get("ego") or {}).get("x")) or 0.0)
    oy = float(((thin.get("ego") or {}).get("y")) or 0.0)
    for _ in range(8):
        dots = thin.get("dots") or []
        if len(dots) > 8:
            # drop farthest / lowest-prob; never invent mid-segment samples
            thin["dots"] = _thin_dots(dots, max(8, len(dots) // 2), (ox, oy))
        for group in ("lanes", "edges"):
            for rec in thin.get(group) or []:
                pts = rec.get("pts") or []
                if len(pts) > 4:
                    rec["pts"] = pts[::2] if pts[-1] in pts[::2] else pts[::2] + [pts[-1]]
        p = thin.get("path") or {}
        if len(p.get("pts") or []) > 4:
            p["pts"] = p["pts"][::2]
        data = json.dumps(thin, separators=(",", ":")).encode("utf-8")
        if len(data) <= max_bytes:
            return data
    thin["tracks"] = (thin.get("tracks") or [])[:16]
    data = json.dumps(thin, separators=(",", ":")).encode("utf-8")
    return data if len(data) <= max_bytes else None


def default_exe() -> Path | None:
    """The built ``gvd_viz`` under cpp/viz/build (Windows preset first)."""
    names = ("gvd_viz.exe", "gvd_viz") if sys.platform == "win32" else ("gvd_viz",)
    for sub in ("windows-msvc/Release", "windows-msvc", "linux-release", ""):
        for name in names:
            cand = ROOT / "cpp" / "viz" / "build" / sub / name
            if cand.is_file():
                return cand
    return None


class CppVizPublisher:
    """Fire-and-forget UDP publisher. ``publish`` never blocks or raises."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1", min_interval_s: float = MIN_INTERVAL_S):
        self.addr = (host, int(port))
        self.min_interval_s = float(min_interval_s)
        self.sent = 0
        self.dropped = 0
        self.too_big = 0
        self.errors = 0
        self.last_error = ""
        self.last_ms = 0.0
        self._seq = 0
        self._last_send = -1e9
        self._proc: subprocess.Popen | None = None
        self.note = ""
        self._sock: socket.socket | None = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setblocking(False)
            self._sock = s
        except OSError as exc:  # no socket: publisher stays a no-op
            self.note = f"viz-cpp: no UDP socket ({exc}); C++ viz off"
            self._sock = None

    # -- renderer process (optional) -------------------------------------
    def launch(self, exe: str | os.PathLike[str] | None = None, extra: list[str] | None = None) -> bool:
        """Start ``gvd_viz --listen PORT``. Missing exe: one note, returns False."""
        path = Path(exe) if exe else default_exe()
        if path is None or not Path(path).is_file():
            self.note = (
                f"viz-cpp: gvd_viz not found ({exe or 'cpp/viz/build'}); publishing anyway, "
                "start it by hand or build it (docs/specs/viz_cpp.md)"
            )
            return False
        try:
            self._proc = subprocess.Popen(
                [str(path), "--listen", str(self.addr[1]), *(extra or [])],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            self.note = f"viz-cpp: could not start {path}: {exc}"
            self._proc = None
            return False
        self.note = f"viz-cpp: started {path} on udp://127.0.0.1:{self.addr[1]}"
        return True

    # -- per tick ----------------------------------------------------------
    def publish(self, state: dict[str, Any], *, now: float | None = None) -> bool:
        """Send one scene. True if a datagram left; False if skipped/dropped."""
        t0 = time.perf_counter()
        try:
            if self._sock is None:
                return False
            mono = time.monotonic() if now is None else float(now)
            if mono - self._last_send < self.min_interval_s:
                return False
            # Increment on every send attempt, including frames later dropped
            # (EAGAIN / OSError / too-big). Rate-limited skips do not increment.
            self._seq += 1
            data = encode_message(build_scene_message(state, seq=self._seq, t=mono))
            if data is None:
                self.too_big += 1
                return False
            try:
                self._sock.sendto(data, self.addr)
            except (BlockingIOError, InterruptedError):
                self.dropped += 1
                return False
            except OSError as exc:  # refused / unreachable / reset: renderer absent
                self.dropped += 1
                self.last_error = str(exc)
                return False
            self._last_send = mono
            self.sent += 1
            return True
        except Exception as exc:  # never let the viz take down the control loop
            self.errors += 1
            self.last_error = repr(exc)
            return False
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000.0

    def status(self) -> dict[str, Any]:
        alive = self._proc is not None and self._proc.poll() is None
        return {"sent": self.sent, "dropped": self.dropped, "too_big": self.too_big, "errors": self.errors,
                "renderer_running": alive, "last_ms": round(self.last_ms, 3)}

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is not None:
            return  # already exited; poll() reaped it
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                pass
        except OSError:
            pass


def write_scene_file(path: str | os.PathLike[str], states: Iterable[dict[str, Any]]) -> int:
    """Offline: write one scene message per line (``gvd_viz --offline``)."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, st in enumerate(states):
            data = encode_message(build_scene_message(st, seq=i + 1, t=float(i) / 30.0))
            if data is None:
                continue
            fh.write(data.decode("utf-8") + "\n")
            n += 1
    return n
