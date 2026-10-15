"""Periodic lateral moves for Automatic Mode, still driven by BeamNG's AI.

The car stays on the road. Cruise is span / drive-in-lane. A weave is a
slow sine inside the current lane. A lane change follows a line into the
next lane. A recovery leaves the lane center by 0.5–1.5 m at about ±10°
and comes back. Junctions, traffic, and the road edge abort the move.
The phase is a tag on the tick. The pixels stay the raw camera frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from python.debug.gt_geom import CURVATURE_AHEAD_M, RoadPoly, lane_center_offsets

PHASES = ("cruise", "weave", "lane_change", "recover", "abort")
ACTIVE = ("weave", "lane_change", "recover")
CRUISE_S = 20.0
WEAVE_S = 12.0
LANE_CHANGE_S = 8.0
RECOVER_S = 12.0
ABORT_S = 3.0
# Stay this far inside the painted lane during a weave.
WEAVE_MARGIN_M = 0.5
WEAVE_AMP_M = 0.45
WEAVE_WAVE_M = 40.0
# A recovery starts this far from the lane center, then returns.
RECOVER_OFFSET_M = (0.5, 1.5)
HEADING_DEG = (5.0, 15.0)
HEADING_TARGET_DEG = 10.0
# Past the lane edge plus this margin, the move is no longer on the road.
EDGE_M = 0.2
PATH_STEP_M = 4.0
MIN_PATH_M = 25.0
# A lane change that never saw this much offset is still in progress.
RECOVER_SEEN_M = 0.4


@dataclass(frozen=True)
class Excursion:
    phase: str
    target_offset_m: float
    target_heading_rad: float
    direction: int
    reason: str

    def as_tag(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "target_offset_m": round(float(self.target_offset_m), 3),
            "target_heading_rad": round(float(self.target_heading_rad), 4),
            "direction": int(self.direction),
            "reason": self.reason,
        }


def _deg(value: float) -> float:
    return math.radians(float(value))


def _heading(direction: int, degrees: float | None = None) -> float:
    """Signed heading error. Positive is left of the road, so a right move is negative."""
    lo, hi = HEADING_DEG
    deg = HEADING_TARGET_DEG if degrees is None else abs(float(degrees))
    deg = min(hi, max(lo, deg))
    sign = -1.0 if int(direction) >= 0 else 1.0
    return sign * _deg(deg)


def unsafe_reason(
    *,
    offset_m: float | None,
    heading_rad: float | None,
    lane_width_m: float | None,
    off_road: bool,
    near_junction: bool,
    traffic: bool,
    phase: str,
) -> str:
    """Why an excursion must stop. Empty when the move may continue."""
    if off_road:
        return "off_road"
    if near_junction:
        return "junction"
    if traffic and phase != "abort":
        return "traffic"
    width = None if lane_width_m is None else float(lane_width_m)
    if offset_m is not None and width is not None and width > 0.0:
        if abs(float(offset_m)) > width * 0.5 + EDGE_M:
            return "road_edge"
    if phase != "cruise" and heading_rad is not None and abs(float(heading_rad)) > _deg(20.0):
        return "heading"
    return ""


class ExcursionClock:
    """Cruise, then weave / lane-change / recover, then cruise again."""

    def __init__(self) -> None:
        self.phase = "cruise"
        self.phase_t = 0.0
        self.direction = 1
        self._kind = 0
        self.reason = ""
        self._saw_offset = False

    def update(
        self,
        dt_s: float,
        *,
        offset_m: float | None,
        heading_rad: float | None,
        lane_width_m: float | None,
        off_road: bool = False,
        near_junction: bool = False,
        traffic: bool = False,
        n_lanes: int | None = None,
    ) -> Excursion:
        dt = max(0.0, float(dt_s))
        reason = unsafe_reason(
            offset_m=offset_m,
            heading_rad=heading_rad,
            lane_width_m=lane_width_m,
            off_road=off_road,
            near_junction=near_junction,
            traffic=traffic,
            phase=self.phase,
        )
        if reason and self.phase not in ("cruise", "abort"):
            self.phase = "abort"
            self.phase_t = 0.0
            self.reason = reason
            self._saw_offset = False
        elif reason and self.phase == "cruise":
            self.reason = reason
            self.phase_t = 0.0
        else:
            if self.phase == "cruise":
                self.reason = ""
            self.phase_t += dt
            self._advance(offset_m, heading_rad, lane_width_m, traffic, n_lanes=n_lanes)
        return self._sample(lane_width_m)

    def force_abort(self, reason: str) -> Excursion:
        """Stop a move that has no safe path. A second call does not restart the timer."""
        if self.phase != "abort":
            self.phase = "abort"
            self.phase_t = 0.0
            self.reason = str(reason or "unsafe")
            self._saw_offset = False
        return self._sample(None)

    def _advance(
        self,
        offset_m: float | None,
        heading_rad: float | None,
        lane_width_m: float | None,
        traffic: bool,
        n_lanes: int | None,
    ) -> None:
        if self.phase == "cruise" and self.phase_t >= CRUISE_S and not self.reason:
            self._start_next(lane_width_m, traffic, n_lanes)
        elif self.phase == "weave" and self.phase_t >= WEAVE_S:
            self._to_cruise()
        elif self.phase == "lane_change" and self.phase_t >= LANE_CHANGE_S:
            self.phase = "recover"
            self.phase_t = 0.0
            self.reason = ""
            self._saw_offset = False
        elif self.phase == "recover":
            if offset_m is not None and abs(float(offset_m)) >= RECOVER_SEEN_M:
                self._saw_offset = True
            centered = self._saw_offset and offset_m is not None and abs(float(offset_m)) < 0.3
            straight = heading_rad is None or abs(float(heading_rad)) < _deg(3.0)
            if (centered and straight) or self.phase_t >= RECOVER_S:
                self._to_cruise()
        elif self.phase == "abort" and self.phase_t >= ABORT_S:
            self._to_cruise()

    def _start_next(self, lane_width_m: float | None, traffic: bool, n_lanes: int | None) -> None:
        if traffic:
            self.phase_t = 0.0
            self.reason = "traffic"
            return
        kind = ("weave", "lane_change", "recover")[self._kind % 3]
        self._kind += 1
        if kind == "lane_change" and n_lanes is not None and int(n_lanes) < 2:
            kind = "weave"
        self.phase = kind
        self.phase_t = 0.0
        self.reason = ""
        self._saw_offset = False
        if kind == "lane_change":
            self.direction = -int(self.direction or 1)

    def _to_cruise(self) -> None:
        self.phase = "cruise"
        self.phase_t = 0.0
        self.reason = ""
        self._saw_offset = False

    def _sample(self, lane_width_m: float | None) -> Excursion:
        if self.phase in ("cruise", "abort"):
            return Excursion(self.phase, 0.0, 0.0, self.direction, self.reason or ("unsafe" if self.phase == "abort" else ""))
        spec = targets_for(self.phase, self.direction, lane_width_m, lane_id=0, road_width_m=None)
        if spec is None:
            return Excursion(self.phase, 0.0, 0.0, self.direction, self.reason)
        return Excursion(self.phase, spec[0], spec[1], self.direction, self.reason)


def targets_for(
    phase: str,
    direction: int,
    lane_width_m: float | None,
    *,
    lane_id: int = 0,
    road_width_m: float | None = None,
) -> tuple[float, float] | None:
    """Lane-center offset and heading for this phase, or None when it cannot stay on the road.

    Offset is positive to the right of the current lane center. Heading is
    positive when the path points left of the road. Without a measured road
    width the nominal tag assumes two lanes of ``lane_width_m``.
    """
    if phase not in ACTIVE:
        return (0.0, 0.0)
    lane_w = 3.5 if lane_width_m is None or float(lane_width_m) <= 0.0 else float(lane_width_m)
    road_w = float(road_width_m) if road_width_m is not None and float(road_width_m) > 0.0 else lane_w * 2.0
    direction = 1 if int(direction) >= 0 else -1
    centers = lane_center_offsets(road_w)
    if not centers:
        return None
    lane_i = max(0, min(int(lane_id), len(centers) - 1))
    lane_here = road_w / float(len(centers))
    if phase == "weave":
        half = max(0.0, lane_here * 0.5 - WEAVE_MARGIN_M)
        amp = min(WEAVE_AMP_M, half)
        if amp < 0.2:
            return None
        heading = math.atan(amp * 2.0 * math.pi / WEAVE_WAVE_M)
        return (direction * amp, -direction * heading)
    if phase == "lane_change":
        nxt = lane_i + direction
        if nxt < 0 or nxt >= len(centers):
            return None
        delta = float(centers[nxt] - centers[lane_i])
        if abs(delta) < 0.4:
            return None
        return (delta, _heading(direction))
    lo, hi = RECOVER_OFFSET_M
    room = lane_here * 0.5 - 0.3
    if room < lo:
        return None
    mag = min(hi, max(lo, min(1.0, room)))
    base = float(centers[lane_i])
    if abs(base + direction * mag) > road_w * 0.5 - EDGE_M:
        return None
    return (direction * mag, _heading(direction))


def excursion_commands(
    speed_mps: float,
    phase: str,
    direction: int,
    *,
    traffic: bool = False,
    path: list[tuple[float, float]] | None = None,
) -> list[str]:
    """AI lines for this phase. A path leaves lane-keeping; cruise puts it back."""
    speed = float(speed_mps)
    if phase in ACTIVE and path:
        return [
            'ai.setMode("manual")',
            'ai.driveInLane("off")',
            'ai.setSpeedMode("limit")',
            f"ai.setSpeed({speed})",
            lua_set_line(path, speed),
        ]
    mode = "traffic" if traffic else "span"
    lines = [
        f'ai.setMode("{mode}")',
        'ai.driveInLane("on")',
        'ai.setSpeedMode("limit")',
        f"ai.setSpeed({speed})",
    ]
    if phase == "abort":
        lines.append("ai.setAggression(0.2)")
    return lines


def lua_set_line(points: list[tuple[float, float]], speed_mps: float) -> str:
    """One ``ai.setLine`` call. ``cling`` keeps the line on the road surface."""
    speed = float(speed_mps)
    body = ",".join(
        f"{{pos={{{x:.2f},{y:.2f},0}},speed={speed:.2f}}}" for x, y in points
    )
    return f"ai.setLine({{{body}}}, true)"


def near_junction(hit: object) -> bool:
    """True when this polyline ends inside the curvature lookahead, or the hit is missing.

    ``project_road`` reports ``ahead_m`` as how far that walk got. A full walk
    is a continuing road, not a junction. The old ``< 20 m`` test fired on
    every healthy sample because the walk itself is only 12 m.
    """
    if hit is None:
        return True
    if bool(getattr(hit, "off_road", False)) or bool(getattr(hit, "discontinuous", False)):
        return True
    ahead = getattr(hit, "ahead_m", 0.0)
    try:
        ahead_m = float(ahead)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(ahead_m):
        return True
    return ahead_m < float(CURVATURE_AHEAD_M) - 0.05


def _locate(
    road: RoadPoly,
    x: float,
    y: float,
    heading: float | None,
) -> tuple[int, float, float, float, float] | None:
    pts = road.xy
    best: tuple[float, int, float, float, float, float] | None = None
    px, py = float(x), float(y)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / length2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        cx = ax + t * dx
        cy = ay + t * dy
        dist = math.hypot(px - cx, py - cy)
        flen = math.sqrt(length2)
        fx, fy = dx / flen, dy / flen
        if best is None or dist < best[0]:
            best = (dist, i, cx, cy, fx, fy)
    if best is None:
        return None
    _dist, i, cx, cy, fx, fy = best
    if heading is not None and math.isfinite(float(heading)):
        vx, vy = math.cos(float(heading)), math.sin(float(heading))
        if fx * vx + fy * vy < 0.0:
            fx, fy = -fx, -fy
    return i, cx, cy, fx, fy


def _walk(
    road: RoadPoly,
    x: float,
    y: float,
    heading: float | None,
    distance_m: float,
    step_m: float,
) -> list[tuple[float, float, float, float, float]]:
    """Centerline samples ``(x, y, fx, fy, s)`` looking forward along the road."""
    located = _locate(road, x, y, heading)
    if located is None:
        return []
    i, cx, cy, fx, fy = located
    pts = road.xy
    ax, ay = pts[i]
    bx, by = pts[i + 1]
    forward = (bx - ax) * fx + (by - ay) * fy >= 0.0
    idx = i + 1 if forward else i
    step_i = 1 if forward else -1
    samples = [(cx, cy, fx, fy, 0.0)]
    px, py = cx, cy
    travelled = 0.0
    next_at = max(1.0, float(step_m))
    distance = max(0.0, float(distance_m))
    guard = 0
    while next_at <= distance + 1e-6 and len(samples) < 100 and guard < 800:
        guard += 1
        if idx < 0 or idx >= len(pts):
            break
        nx, ny = pts[idx]
        dx, dy = nx - px, ny - py
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            idx += step_i
            continue
        ux, uy = dx / seg, dy / seg
        if travelled + seg + 1e-9 < next_at:
            px, py = nx, ny
            travelled += seg
            fx, fy = ux, uy
            idx += step_i
            continue
        need = next_at - travelled
        px = px + ux * need
        py = py + uy * need
        travelled = next_at
        fx, fy = ux, uy
        samples.append((px, py, fx, fy, travelled))
        next_at += max(1.0, float(step_m))
    return samples


def _road_of(roads: list[RoadPoly], hit: object) -> RoadPoly | None:
    want = str(getattr(hit, "road_id", "") or "")
    if want:
        for road in roads:
            if str(road.road_id) == want:
                return road
    return roads[0] if roads else None


def _lateral(phase: str, s_m: float, base: float, delta: float) -> float:
    """Road-center offset along the path. ``delta`` is the signed lane-center excursion."""
    if phase == "weave":
        wave = max(WEAVE_WAVE_M, 1.0)
        return base + delta * math.sin(2.0 * math.pi * float(s_m) / wave)
    heading = math.tan(_deg(HEADING_TARGET_DEG))
    ramp = max(8.0, abs(float(delta)) / max(heading, 1e-3))
    if phase == "lane_change":
        return base + float(delta) * min(1.0, float(s_m) / ramp)
    if float(s_m) < ramp:
        return base + float(delta) * (float(s_m) / ramp)
    if float(s_m) < ramp + 8.0:
        return base + float(delta)
    back = float(s_m) - (ramp + 8.0)
    return base + float(delta) * (1.0 - min(1.0, back / ramp))


def excursion_path(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading: float | None,
    move: Excursion,
    hit: object,
    speed_mps: float,
) -> tuple[list[tuple[float, float]] | None, int]:
    """World points for ``ai.setLine``, and the direction that stays on the road.

    The second value flips when the requested lane change would leave the
    pavement and the other side would not. ``(None, direction)`` means do
    not send a line.
    """
    if move.phase not in ACTIVE or hit is None or not roads:
        return None, int(move.direction)
    road = _road_of(roads, hit)
    if road is None:
        return None, int(move.direction)
    directions = [1 if int(move.direction) >= 0 else -1]
    if move.phase == "lane_change":
        directions.append(-directions[0])
    duration = {"weave": WEAVE_S, "lane_change": LANE_CHANGE_S, "recover": RECOVER_S}[move.phase]
    need = max(MIN_PATH_M, abs(float(speed_mps)) * duration)
    need = min(400.0, need)
    lane_id = int(getattr(hit, "lane_id", 0) or 0)
    lane_w = getattr(hit, "lane_width_m", None)
    for direction in directions:
        spec = targets_for(
            move.phase,
            direction,
            None if lane_w is None else float(lane_w),
            lane_id=lane_id,
            road_width_m=float(road.width_m),
        )
        if spec is None:
            continue
        delta = float(spec[0])
        centers = lane_center_offsets(float(road.width_m))
        if lane_id < 0 or lane_id >= len(centers):
            continue
        base = float(centers[lane_id])
        samples = _walk(road, x, y, heading, need, PATH_STEP_M)
        if len(samples) < 2 or samples[-1][4] + 1e-3 < need * 0.85:
            continue
        limit = float(road.width_m) * 0.5 - EDGE_M
        points: list[tuple[float, float]] = []
        ok = True
        for cx, cy, fx, fy, s_m in samples:
            lat = _lateral(move.phase, s_m, base, delta)
            if abs(lat) > limit + 1e-6:
                ok = False
                break
            rx, ry = fy, -fx
            points.append((cx + rx * lat, cy + ry * lat))
        if ok and len(points) >= 2:
            return points, direction
    return None, int(move.direction)
