"""Multi-frame object tracker; motion state comes from track displacement only.

Detections arrive in the ego frame each tick (x right, y forward, metres).
Ego speed and yaw rate (runtime kinematics, not ground truth) move them into
an odometry frame, where a track's displacement over its last frames gives
its ground velocity. A track seen in fewer than 3 frames, or over less than 0.2 s,
is ``unknown``: there is no single-frame motion guess.

States: ``parked`` (ground speed < 0.5 m/s over >= MIN_SPAN_S),
``same_way`` / ``alongside`` (heading within 35 deg of ego; alongside when
beside the ego), ``oncoming`` (within 35 deg of opposite), ``crossing``
(otherwise), ``unknown`` (fewer than 2 frames or a span shorter than
MIN_SPAN_S).

Timestamps must increase. A repeat or backwards stamp (clock jump), a gap
over ``max_gap_s`` (dropped frames) or non-finite ego kinematics reset the
tracker, so no velocity is ever computed across a discontinuity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from python.occ.classes import SRC_MOTION

MOTION_STATES = ("unknown", "parked", "same_way", "oncoming", "crossing", "alongside")
PARKED_MPS = 0.5
MIN_SPAN_S = 0.2
MIN_FRAMES = 3
SAME_DEG = 35.0
ALONGSIDE_Y = 6.0
ALONGSIDE_X = 1.2
MAX_TRACKS = 64
# Fastest object the gate allows for a track with no velocity yet (highway oncoming).
MAX_OBJ_MPS = 45.0


def _finite(*vals: Any) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


@dataclass
class _Track:
    tid: int
    cls: str
    hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, X, Y) odom
    det: dict[str, Any] = field(default_factory=dict)
    misses: int = 0
    vel: tuple[float, float] | None = None


class MotionTracker:
    def __init__(self, *, gate_m: float = 3.0, max_misses: int = 2, max_gap_s: float = 0.5, window: int = 6):
        self.gate_m = float(gate_m)
        self.max_misses = int(max_misses)
        self.max_gap_s = float(max_gap_s)
        self.window = max(2, int(window))
        self.resets = 0
        self._next_id = 1
        self._reset_state()

    def _reset_state(self) -> None:
        self.tracks: list[_Track] = []
        self.t: float | None = None
        self.px = 0.0
        self.py = 0.0
        self.th = math.pi / 2  # ego forward = +y at the odom origin

    def reset(self) -> None:
        self.resets += 1
        self._reset_state()

    def _to_odom(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.th), math.sin(self.th)
        return self.px + x * s + y * c, self.py - x * c + y * s

    def _to_ego_vec(self, vx: float, vy: float) -> tuple[float, float]:
        c, s = math.cos(self.th), math.sin(self.th)
        return vx * s - vy * c, vx * c + vy * s

    def _motion(self, tr: _Track) -> tuple[str, float | None, int]:
        h = tr.hist[-self.window :]
        n = len(h)
        if n < MIN_FRAMES:
            return "unknown", None, n
        t0, x0, y0 = h[0]
        t1, x1, y1 = h[-1]
        span = t1 - t0
        if span < MIN_SPAN_S:
            return "unknown", None, n
        vx, vy = (x1 - x0) / span, (y1 - y0) / span
        speed = math.hypot(vx, vy)
        tr.vel = self._to_ego_vec(vx, vy)  # ego-frame ground velocity (m/s), x right / y forward
        if speed < PARKED_MPS:
            return "parked", speed, n
        ex, ey = self._to_ego_vec(vx, vy)
        rel = math.degrees(math.atan2(ey, ex) - math.pi / 2)
        rel = (rel + 180.0) % 360.0 - 180.0
        if abs(rel) <= SAME_DEG:
            d = tr.det
            if abs(float(d["y"])) <= ALONGSIDE_Y and abs(float(d["x"])) >= ALONGSIDE_X:
                return "alongside", speed, n
            return "same_way", speed, n
        if abs(rel) >= 180.0 - SAME_DEG:
            return "oncoming", speed, n
        return "crossing", speed, n

    def update(
        self,
        t: float,
        detections: Sequence[Mapping[str, Any]],
        *,
        speed_mps: float | None = None,
        yaw_rate: float = 0.0,
        motion: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """``motion`` = (dx, dy, dyaw) since the last tick in the previous ego frame
        (the same estimate the temporal warp used: VO by default). Without it,
        ``speed_mps`` / ``yaw_rate`` are integrated (``--ego-from-vehicle``)."""
        use_motion = motion is not None
        if use_motion:
            try:
                mdx, mdy, mdyaw = (float(v) for v in motion)  # type: ignore[union-attr]
            except (TypeError, ValueError):
                mdx = mdy = mdyaw = math.nan
            ok = _finite(t, mdx, mdy, mdyaw)
        else:
            ok = speed_mps is not None and _finite(t, speed_mps, yaw_rate)
        if not ok:
            self.reset()
            return self._emit_unknown(detections)
        t = float(t)
        if self.t is not None:
            dt = t - self.t
            if dt <= 0.0 or dt > self.max_gap_s:
                self.reset()
            elif use_motion:
                c, s = math.cos(self.th), math.sin(self.th)
                self.px += mdx * s + mdy * c
                self.py += -mdx * c + mdy * s
                self.th += mdyaw
            else:
                self.th += float(yaw_rate) * dt
                self.px += float(speed_mps) * dt * math.cos(self.th)
                self.py += float(speed_mps) * dt * math.sin(self.th)
        self.t = t
        dets = [dict(d) for d in detections if isinstance(d, Mapping) and _finite(d.get("x"), d.get("y"))][:MAX_TRACKS]
        odom = [self._to_odom(float(d["x"]), float(d["y"])) for d in dets]
        pairs = []
        for ti, tr in enumerate(self.tracks):
            tl, X, Y = tr.hist[-1]
            dt_tr = max(0.0, t - tl)
            if len(tr.hist) >= 2:
                # Constant-velocity prediction; the gate stays gate_m around it.
                ta, xa, ya = tr.hist[-2]
                span = max(tl - ta, 1e-6)
                X += (X - xa) / span * dt_tr
                Y += (Y - ya) / span * dt_tr
                gate = self.gate_m
            else:
                gate = self.gate_m + MAX_OBJ_MPS * dt_tr
            for di, (ox, oy) in enumerate(odom):
                dist = math.hypot(ox - X, oy - Y)
                if dist <= gate and str(dets[di].get("cls", "vehicle")) == tr.cls:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            tr.hist.append((t, *odom[di]))
            tr.hist = tr.hist[-self.window :]
            tr.det = dets[di]
            tr.misses = 0
        keep = []
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
            if tr.misses <= self.max_misses:
                keep.append(tr)
        self.tracks = keep
        for di, d in enumerate(dets):
            if di in used_d or len(self.tracks) >= MAX_TRACKS:
                continue
            tr = _Track(self._next_id, str(d.get("cls", "vehicle")), [(t, *odom[di])], d)
            self._next_id += 1
            self.tracks.append(tr)
        out = []
        for tr in self.tracks:
            if tr.misses:
                continue
            state, speed, n = self._motion(tr)
            row = dict(tr.det)
            row.update({"id": tr.tid, "cls": tr.cls, "motion": state, "motion_frames": n, "motion_src": SRC_MOTION})
            if speed is not None:
                row["v"] = speed
                vel = getattr(tr, "vel", None)
                if vel is not None and all(math.isfinite(c) for c in vel):
                    row["vx"], row["vy"] = float(vel[0]), float(vel[1])
            out.append(row)
        return out

    def _emit_unknown(self, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for d in detections:
            if isinstance(d, Mapping) and _finite(d.get("x"), d.get("y")):
                row = dict(d)
                row.update({"id": None, "motion": "unknown", "motion_frames": 1, "motion_src": SRC_MOTION})
                out.append(row)
        return out[:MAX_TRACKS]
