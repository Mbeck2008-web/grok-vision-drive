"""Smoke-run instrumentation for excursion lines (log and collect.json only, no control).

What the first live run with ``vehicle.ai.set_line`` has to show, per the
pass bar (KILL on a snap):

* at each weave start: the car's distance from the first line point, and the
  per-tick position delta, unexplained jump and speed for +/- SMOKE_WINDOW_S;
* the tick spacing for +/- SMOKE_WINDOW_S around every line ack and recheck
  (the ack and recheck must not stall the tick loop);
* per weave (acked rows only): the correlation of the measured yaw rate with
  the yaw rate the sent line commands at the car (speed x signed curvature of
  the line at its vertex nearest the car).

A "jump" is the part of a tick-to-tick move that speed and heading do not
explain: ``|dp - (v0 d0 + v1 d1) / 2 * dt|`` (``d`` the unit heading in x, y).
``driveUsingPath{script}`` teleports the car to the first line node
(ai.lua 6003-6008, ``vehicleSetPositionRotation``); that would show as a jump
at the weave start. Nothing here changes what the collector does.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable

SMOKE_WINDOW_S = 1.0
JUMP_M = 0.5
YAW_CORR_MIN = 0.5
KEEP = 200  # per-window records kept in collect.json (counts and maxima cover all of them)

PASS_BAR = (
    "0 line_not_active, 0 line_lost, no tick-to-tick position jump over 0.5 m and no snap to the first "
    "line point at weave start (a snap is a teleport: KILL), yaw rate vs commanded yaw rate correlation "
    ">= 0.5 in every weave"
)


def _xy(v: Any) -> tuple[float, float] | None:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            x, y = float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return None


def _unit(v: Any) -> tuple[float, float] | None:
    xy = _xy(v)
    if xy is None:
        return None
    n = math.hypot(*xy)
    return None if n < 1e-9 else (xy[0] / n, xy[1] / n)


def _num(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def jump_m(a: dict, b: dict) -> float | None:
    """Unexplained displacement between two ticks (see module doc)."""
    if a["pos"] is None or b["pos"] is None:
        return None
    dt = b["t"] - a["t"]
    dx, dy = b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1]
    v0, v1 = a["v"] or 0.0, b["v"] or 0.0
    if a["dir"] is not None and b["dir"] is not None:
        ex = 0.5 * (v0 * a["dir"][0] + v1 * b["dir"][0]) * dt
        ey = 0.5 * (v0 * a["dir"][1] + v1 * b["dir"][1]) * dt
        return math.hypot(dx - ex, dy - ey)
    return abs(math.hypot(dx, dy) - 0.5 * (v0 + v1) * dt)


def line_curvatures(points: list[tuple[float, float]]) -> list[float]:
    """Signed curvature (1/m, CCW positive) at each vertex; the ends take their neighbour's."""
    n = len(points)
    if n < 3:
        return [0.0] * n
    out = [0.0] * n
    for i in range(1, n - 1):
        (ax, ay), (bx, by), (cx, cy) = points[i - 1], points[i], points[i + 1]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        la, lb, lc = math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by), math.hypot(cx - ax, cy - ay)
        out[i] = 0.0 if la * lb * lc < 1e-12 else 2.0 * cross / (la * lb * lc)
    out[0], out[-1] = out[1], out[-2]
    return out


def corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5 or len(a) != len(b):
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa < 1e-12 or sb < 1e-12:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(float(v), nd)


class LineSmoke:
    """Feed it every tick and every line event; ``summary()`` goes into ``weave_stats.line_smoke``."""

    def __init__(self, window_s: float = SMOKE_WINDOW_S) -> None:
        self.window_s = float(window_s)
        self._ring: deque[dict] = deque()
        self._open: list[dict] = []
        self.starts: list[dict] = []
        self.events: list[dict] = []
        self.weaves: list[dict] = []
        self.n_starts = 0
        self.n_events: dict[str, int] = {}
        self.snaps = 0
        self.start_max_jump_m: float | None = None
        self.event_max_tick_s: float | None = None
        self.weave_max_jump_m: float | None = None
        self._line: dict | None = None
        self._prev_phase = ""

    # -- inputs
    def tick(
        self,
        t: float,
        pos: Any,
        direction: Any,
        speed: Any,
        yaw_rate: Any,
        phase_tag: str,
        segment: Any = None,
    ) -> list[str]:
        """One collector tick (``t`` monotonic s). Returns log lines for windows that closed."""
        cur = {"t": float(t), "pos": _xy(pos), "dir": _unit(direction), "v": _num(speed), "yaw": _num(yaw_rate),
               "phase": str(phase_tag or "")}
        prev = self._ring[-1] if self._ring else None
        if prev is not None and cur["phase"].startswith("weave"):
            j = jump_m(prev, cur)
            if j is not None and (self.weave_max_jump_m is None or j > self.weave_max_jump_m):
                self.weave_max_jump_m = j
        self._ring.append(cur)
        while self._ring and self._ring[0]["t"] < cur["t"] - self.window_s - 0.5:
            self._ring.popleft()
        self._weave_tick(cur)
        out: list[str] = []
        still = []
        for w in self._open:
            if cur["t"] >= w["t0"] - self.window_s - 1e-9:
                w["ticks"].append(cur)
            if cur["t"] > w["t0"] + self.window_s + 1e-9:
                out.append(self._close(w))
            else:
                still.append(w)
        self._open = still
        return out

    def _window(self, kind: str, t: float, segment: Any, **extra: Any) -> dict:
        lo = float(t) - self.window_s - 1e-9
        w = {"kind": kind, "t0": float(t), "segment": segment, "ticks": [x for x in self._ring if x["t"] >= lo], **extra}
        self._open.append(w)
        return w

    def line_start(self, t: float, pos: Any, points: list[tuple[float, float]], phase: str, segment: Any = None) -> None:
        """A line was sent (``set_line``) at ``t``; the car is at ``pos``."""
        self._end_weave()
        pts = [(float(x), float(y)) for x, y in points]
        self._line = {"points": pts, "kappa": line_curvatures(pts), "phase": str(phase), "segment": segment,
                      "yaw": [], "cmd": [], "t0": float(t)}
        if phase != "weave":
            return
        self.n_starts += 1
        here = _xy(pos)
        first = pts[0] if pts else None
        dist = None if here is None or first is None else math.hypot(here[0] - first[0], here[1] - first[1])
        self._window("weave_start", t, segment, first=first, dist_first_m=dist)

    def event(self, kind: str, t: float, segment: Any = None, detail: str = "") -> None:
        """``ack``, ``ack_fail``, ``recheck_sent``, ``recheck_ok``, ``line_lost``."""
        self.n_events[kind] = self.n_events.get(kind, 0) + 1
        self._window(kind, t, segment, detail=str(detail))

    def flush(self) -> list[str]:
        """Segment end / shutdown: close every open window with the ticks it has."""
        out = [self._close(w) for w in self._open]
        self._open = []
        self._end_weave()
        return out

    # -- yaw vs command
    def _weave_tick(self, cur: dict) -> None:
        line = self._line
        if cur["phase"] != "weave" and self._prev_phase == "weave":
            self._end_weave()
            line = None
        self._prev_phase = cur["phase"]
        if line is None or cur["phase"] != "weave" or cur["pos"] is None or cur["yaw"] is None or cur["v"] is None:
            return
        pts = line["points"]
        if not pts:
            return
        k = min(range(len(pts)), key=lambda i: (pts[i][0] - cur["pos"][0]) ** 2 + (pts[i][1] - cur["pos"][1]) ** 2)
        line["yaw"].append(cur["yaw"])
        line["cmd"].append(cur["v"] * line["kappa"][k])

    def _end_weave(self) -> None:
        line, self._line = self._line, None
        if not line or line["phase"] != "weave" or not line["yaw"]:
            return
        c = corr(line["yaw"], line["cmd"])
        self.weaves.append({"segment": line["segment"], "n": len(line["yaw"]), "yaw_cmd_corr": _r(c)})

    # -- windows
    def _close(self, w: dict) -> str:
        ticks = sorted(w["ticks"], key=lambda x: x["t"])
        rows = []
        for i, x in enumerate(ticks):
            prev = ticks[i - 1] if i else None
            dpos = None
            if prev is not None and prev["pos"] is not None and x["pos"] is not None:
                dpos = math.hypot(x["pos"][0] - prev["pos"][0], x["pos"][1] - prev["pos"][1])
            rows.append({
                "dt_s": _r(x["t"] - w["t0"]),
                "tick_s": None if prev is None else _r(x["t"] - prev["t"]),
                "dpos_m": _r(dpos),
                "jump_m": None if prev is None else _r(jump_m(prev, x)),
                "v_mps": _r(x["v"], 2),
            })
        jumps = [r["jump_m"] for r in rows if r["jump_m"] is not None]
        spacing = [r["tick_s"] for r in rows if r["tick_s"] is not None]
        max_jump = max(jumps) if jumps else None
        max_tick = max(spacing) if spacing else None
        rec = {"kind": w["kind"], "segment": w["segment"], "n_ticks": len(rows), "max_jump_m": max_jump, "max_tick_s": max_tick}
        if w["kind"] == "weave_start":
            rec["dist_first_m"] = _r(w.get("dist_first_m"))
            rec["snap"] = bool(max_jump is not None and max_jump > JUMP_M)
            rec["ticks"] = rows
            if rec["snap"]:
                self.snaps += 1
            if max_jump is not None and (self.start_max_jump_m is None or max_jump > self.start_max_jump_m):
                self.start_max_jump_m = max_jump
            if len(self.starts) < KEEP:
                self.starts.append(rec)
            dp = ",".join("na" if r["dpos_m"] is None else f"{r['dpos_m']:.2f}" for r in rows)
            vv = ",".join("na" if r["v_mps"] is None else f"{r['v_mps']:.1f}" for r in rows)
            dist = "na" if rec["dist_first_m"] is None else f"{rec['dist_first_m']:.2f}"
            mj = "na" if max_jump is None else f"{max_jump:.2f}"
            return (f"collect smoke weave_start {w['segment']} dist_first={dist}m max_jump={mj}m "
                    f"snap={'KILL' if rec['snap'] else 'no'} dpos=[{dp}] v=[{vv}]")
        rec["detail"] = w.get("detail", "")
        rec["tick_s"] = spacing
        if max_tick is not None and (self.event_max_tick_s is None or max_tick > self.event_max_tick_s):
            self.event_max_tick_s = max_tick
        if len(self.events) < KEEP:
            self.events.append(rec)
        sp = ",".join(f"{s:.3f}" for s in spacing)
        mt = "na" if max_tick is None else f"{max_tick:.3f}"
        return f"collect smoke {w['kind']} {w['segment']} {w.get('detail', '')} max_tick={mt}s tick=[{sp}]".replace("  ", " ")

    # -- output
    def summary(self, aborted: dict | None = None) -> dict[str, Any]:
        aborted = aborted or {}

        def count(reason: str) -> int:
            return sum(int((per or {}).get(reason, 0)) for per in aborted.values())

        corrs = [w["yaw_cmd_corr"] for w in self.weaves if w["yaw_cmd_corr"] is not None]
        lna, lost = count("line_not_active"), count("line_lost")
        max_jump = max([v for v in (self.start_max_jump_m, self.weave_max_jump_m) if v is not None], default=None)
        bar = {
            "line_not_active": lna,
            "line_lost": lost,
            "max_tick_jump_m": _r(max_jump),
            "snaps_at_weave_start": int(self.snaps),
            "weaves_with_corr": len(corrs),
            "yaw_cmd_corr_min": _r(min(corrs)) if corrs else None,
            "yaw_cmd_corr_median": _r(sorted(corrs)[len(corrs) // 2]) if corrs else None,
        }
        bar["pass"] = bool(
            lna == 0 and lost == 0 and self.snaps == 0
            and max_jump is not None and max_jump <= JUMP_M
            and corrs and min(corrs) >= YAW_CORR_MIN
        )
        return {
            "bar": PASS_BAR,
            "pass_bar": bar,
            "weave_starts": int(self.n_starts),
            "events": dict(self.n_events),
            "start_max_jump_m": _r(self.start_max_jump_m),
            "weave_max_jump_m": _r(self.weave_max_jump_m),
            "event_max_tick_s": _r(self.event_max_tick_s),
            "starts": list(self.starts),
            "event_windows": list(self.events),
            "weaves": list(self.weaves[:KEEP]),
        }

