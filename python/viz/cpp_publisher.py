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

Schema v1 (``{"v": 1, "type": "gvd_scene", ...}``) is mirrored by
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

SCHEMA_VERSION = 1
DEFAULT_PORT = 47800
# Loopback UDP carries up to 65507 bytes; stay well under it.
MAX_DATAGRAM = 60000
MAX_TRACKS = 64
MAX_SIGNS = 16
MAX_LANES = 16
MAX_EDGES = 8
MAX_POINTS = 120
# Supervisor ticks are ~15-30 Hz; never send faster than the renderer's 60 fps.
MIN_INTERVAL_S = 1.0 / 60.0

ROOT = Path(__file__).resolve().parents[2]
_PAINT_KEYS = ("paint", "paint_color", "color", "colour", "material", "line_color")
# Top-level message keys. Tests pin this set so a new field is a schema decision.
MESSAGE_KEYS = frozenset(
    {"v", "type", "seq", "t", "ego", "lead_id", "aeb", "lanes", "edges", "tracks", "signs", "path", "path_conf"}
)


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
    """``"yellow"`` only when the record carries yellow paint; else ``"white"``.

    Live ``lanes_ext`` has no paint field yet, so live lanes are white. Index,
    side and kind never pick a colour (no purple lanes).
    """
    if not isinstance(record, dict):
        return "white"
    for key in _PAINT_KEYS:
        raw = record.get(key)
        if isinstance(raw, str) and "yellow" in raw.lower():
            return "yellow"
    if record.get("yellow") is True:
        return "yellow"
    return "white"


def _int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _tracks(src: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for tr in src:
        if not isinstance(tr, dict):
            continue
        x, y = _f(tr.get("x")), _f(tr.get("y"))
        if x is None or y is None:
            continue
        row: dict[str, Any] = {"cls": str(tr.get("class") or tr.get("cls") or "vehicle"), "x": x, "y": y}
        tid = _int(tr.get("id"))
        if tid is not None:
            row["id"] = tid
        yaw = _f(tr.get("yaw"), 3)
        if yaw is not None:
            row["yaw"] = yaw
        for src_key, key in (("length", "l"), ("width", "w"), ("height", "h"), ("speed_mps", "v")):
            val = _f(tr.get(src_key))
            if val is not None:
                row[key] = val
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
        cls = str(s.get("cls") or s.get("class") or "")
        if x is None or y is None or not cls:
            continue
        out.append({"cls": cls, "x": x, "y": y, "state": str(s.get("state") or "unknown")})
        if len(out) >= MAX_SIGNS:
            break
    return out


def _list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def build_scene_message(state: dict[str, Any], *, seq: int = 0, t: float | None = None) -> dict[str, Any]:
    """Scene message from the supervisor state. Perception fields only.

    Reads: ``engaged``, ``ego.speed_mps``, ``planner.{target_v,cipv_id,aeb}``,
    ``path_ego``, ``path_conf``, ``lanes_ext``, ``road_edges``, ``tracks``,
    ``signs``. Nothing else.
    """
    st = state if isinstance(state, dict) else {}
    ego = st.get("ego") if isinstance(st.get("ego"), dict) else {}
    pl = st.get("planner") if isinstance(st.get("planner"), dict) else {}
    msg: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "type": "gvd_scene",
        "seq": int(seq),
        "t": round(float(time.monotonic() if t is None else t), 4),
        "ego": {"engaged": bool(st.get("engaged")), "speed_mps": _f(ego.get("speed_mps")) or 0.0},
        "aeb": str(pl.get("aeb") or "off"),
        "lanes": [],
        "edges": [],
        "tracks": _tracks(_list(st.get("tracks"))),
        "signs": _signs(_list(st.get("signs"))),
        "path": {"pts": _pts(st.get("path_ego"))},
        "path_conf": _f(st.get("path_conf")) if _f(st.get("path_conf")) is not None else 1.0,
    }
    set_v = _f(pl.get("target_v"))
    if set_v is not None:
        msg["ego"]["set_mps"] = set_v
    lead = _int(pl.get("cipv_id"))
    if lead is not None:
        msg["lead_id"] = lead
    for ln in _list(st.get("lanes_ext"))[:MAX_LANES]:
        pts = _pts(ln)
        if len(pts) < 2:
            continue
        kind = str(ln.get("kind") or "detected") if isinstance(ln, dict) else "detected"
        msg["lanes"].append({"pts": pts, "paint": lane_paint(ln), "kind": kind})
    for ed in _list(st.get("road_edges"))[:MAX_EDGES]:
        pts = _pts(ed)
        if len(pts) < 2:
            continue
        kind = str(ed.get("kind") or "detected") if isinstance(ed, dict) else "detected"
        msg["edges"].append({"pts": pts, "kind": kind})
    return msg


def encode_message(msg: dict[str, Any], max_bytes: int = MAX_DATAGRAM) -> bytes | None:
    """Compact UTF-8 JSON that fits one datagram, thinning points if needed."""
    data = json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) <= max_bytes:
        return data
    thin = json.loads(data)
    for _ in range(6):
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
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
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
