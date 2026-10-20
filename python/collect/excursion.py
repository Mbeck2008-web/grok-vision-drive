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

from python.debug.gt_geom import CURVATURE_AHEAD_M, RoadPoly, lane_center_offsets, road_ahead_split

PHASES = ("cruise", "weave", "lane_change", "recover", "abort")
ACTIVE = ("weave", "lane_change", "recover")
CRUISE_S = 20.0
WEAVE_S = 12.0
LANE_CHANGE_S = 8.0
RECOVER_S = 12.0
ABORT_S = 3.0
# Stay this far inside the painted lane during a weave.
WEAVE_MARGIN_M = 0.5
# In-lane only. Large enough to see, small enough to stay off the next lane.
WEAVE_AMP_M = 0.6
WEAVE_AMP_LO_M = 0.3
WEAVE_AMP_HI_M = 0.8
WEAVE_WAVE_M = 40.0
# Speed-compensated weave. The wavelength is the distance covered in
# WEAVE_PERIOD_S, never under WEAVE_WAVE_M. The amplitude cap falls linearly
# from WEAVE_AMP_HI_M at WEAVE_SLOW_MPS to WEAVE_AMP_FAST_M at WEAVE_FAST_MPS.
# WEAVE_LAT_ACC_MPS2 guards the peak lateral acceleration of the sine,
# amp * (2*pi / wave)^2 * v^2: a larger amplitude is shrunk to it, and a weave
# that would need less than WEAVE_AMP_LO_M is not started. Highways keep
# weaving, slower and smaller. rec4 impacts were all in weave/abort.
WEAVE_PERIOD_S = 5.0
WEAVE_SLOW_MPS = 10.0
WEAVE_FAST_MPS = 22.0
WEAVE_AMP_FAST_M = 0.35
WEAVE_LAT_ACC_MPS2 = 1.0
# Ego body half-width for the weave lane-edge test. The collector always
# drives an etk800 (level_session.py); 1.8 m is path_predictor.VEHICLE_WIDTH.
# No bbox request: a live query that BeamNG may not answer is not used here.
EGO_HALF_WIDTH_M = 0.9
# A recovery starts this far from the lane center, then returns.
RECOVER_OFFSET_M = (0.5, 1.5)
HEADING_DEG = (5.0, 15.0)
HEADING_TARGET_DEG = 10.0
# Past the lane edge plus this margin, the move is no longer on the road.
EDGE_M = 0.2
# During a weave or lane change, the car may sit well off the lane center
# while still on the pavement. The navgraph left/right is the paint. West
# coast weaves were 0.5–0.6 m outside that line, still on the shoulder.
# Abort only once the car is a metre past the paint.
PAVED_PAST_M = 1.0
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


def _past_paved_edge(road_offset_m: float | None, road_width_m: float | None) -> bool:
    """True when the car is more than ``PAVED_PAST_M`` outside the median road span."""
    if road_offset_m is None or road_width_m is None:
        return False
    try:
        offset = abs(float(road_offset_m))
        width = float(road_width_m)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(offset) or not math.isfinite(width) or width <= 0.0:
        return False
    return offset > width * 0.5 + PAVED_PAST_M


def weave_wave_m(speed_mps: float | None) -> float:
    """Weave wavelength: ``max(WEAVE_WAVE_M, v * WEAVE_PERIOD_S)``. No speed: WEAVE_WAVE_M."""
    v = _speed(speed_mps)
    return WEAVE_WAVE_M if v is None else max(WEAVE_WAVE_M, v * WEAVE_PERIOD_S)


def weave_amp_cap_m(speed_mps: float | None) -> float:
    """0.8 m at or under 10 m/s, 0.35 m at or over 22 m/s, linear between."""
    v = _speed(speed_mps)
    if v is None or v <= WEAVE_SLOW_MPS:
        return WEAVE_AMP_HI_M
    if v >= WEAVE_FAST_MPS:
        return WEAVE_AMP_FAST_M
    k = (v - WEAVE_SLOW_MPS) / (WEAVE_FAST_MPS - WEAVE_SLOW_MPS)
    return WEAVE_AMP_HI_M + k * (WEAVE_AMP_FAST_M - WEAVE_AMP_HI_M)


def weave_lat_acc(amp_m: float, wave_m: float, speed_mps: float) -> float:
    """Peak lateral acceleration of ``amp * sin(2*pi*s/wave)`` driven at ``v``."""
    k = 2.0 * math.pi / max(float(wave_m), 1e-6)
    return float(amp_m) * k * k * float(speed_mps) ** 2


def weave_plan(speed_mps: float | None, room_m: float) -> tuple[float, float] | None:
    """``(amp_m, wave_m)`` for a weave at this speed, or None when it cannot start.

    ``room_m`` is the largest center offset the lane allows. The amplitude is
    min(WEAVE_AMP_M, room, speed cap), shrunk to the lateral-acceleration
    guard; under WEAVE_AMP_LO_M the weave is refused.
    """
    wave = weave_wave_m(speed_mps)
    amp = min(WEAVE_AMP_M, float(room_m), weave_amp_cap_m(speed_mps))
    v = _speed(speed_mps)
    if v is not None and v > 0.0:
        k = 2.0 * math.pi / wave
        amp = min(amp, WEAVE_LAT_ACC_MPS2 / (k * k * v * v))
    if amp < WEAVE_AMP_LO_M - 1e-9:
        return None
    return amp, wave


def _speed(speed_mps: float | None) -> float | None:
    if speed_mps is None:
        return None
    try:
        v = abs(float(speed_mps))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def body_past_lane_edge(
    lane_offset_m: float | None,
    lane_width_m: float | None,
    half_width_m: float = EGO_HALF_WIDTH_M,
) -> bool:
    """True when the ego body edge is more than EDGE_M outside its lane's painted edge.

    Measured FROM the body edge: the car's lateral offset from its lane
    center (``lane_offset_m``, the BeamNG vehicle position projected onto the
    navgraph road) plus or minus ``half_width_m``. Measured AGAINST the lane
    edge in WORLD road geometry: the navgraph RoadPoly lane center
    (``lane_center_offsets`` of the road width and lane count) plus or minus
    half the lane width. Not the painted-lane labels in the camera rows.
    """
    if lane_offset_m is None or lane_width_m is None:
        return False
    try:
        off = abs(float(lane_offset_m))
        width = float(lane_width_m)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(off) or not math.isfinite(width) or width <= 0.0:
        return False
    return off + float(half_width_m) > width * 0.5 + EDGE_M


def unsafe_reason(
    *,
    offset_m: float | None,
    heading_rad: float | None,
    lane_width_m: float | None,
    off_road: bool,
    near_junction: bool,
    traffic: bool,
    phase: str,
    road_offset_m: float | None = None,
    road_width_m: float | None = None,
    half_width_m: float | None = None,
) -> str:
    """Why an excursion must stop. Empty when the move may continue.

    A weave is aborted for ``lane_edge`` once its body edge is EDGE_M past
    the lane edge (``body_past_lane_edge``; ``offset_m`` / ``lane_width_m``
    come from the world road, and ``half_width_m`` None skips it). A weave,
    lane change or recovery is aborted for ``road_edge`` only when the car is
    past the paved edge. ``road_width_m`` is the median left-right span, not
    the widest junction node. Cruise still uses the lane half-width.
    """
    if off_road:
        return "off_road"
    if near_junction:
        return "junction"
    if traffic and phase != "abort":
        return "traffic"
    if phase == "weave" and half_width_m is not None and body_past_lane_edge(offset_m, lane_width_m, half_width_m):
        return "lane_edge"
    if phase in ("weave", "lane_change", "recover") and road_width_m is not None and road_offset_m is not None:
        if _past_paved_edge(road_offset_m, road_width_m):
            return "road_edge"
    else:
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
        self.speed_mps: float | None = None

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
        window_ok: bool = True,
        start_kind: str | None = None,
        road_offset_m: float | None = None,
        road_width_m: float | None = None,
        speed_mps: float | None = None,
        half_width_m: float | None = None,
    ) -> Excursion:
        dt = max(0.0, float(dt_s))
        self.speed_mps = speed_mps
        reason = unsafe_reason(
            offset_m=offset_m,
            heading_rad=heading_rad,
            lane_width_m=lane_width_m,
            off_road=off_road,
            near_junction=near_junction,
            traffic=traffic,
            phase=self.phase,
            road_offset_m=road_offset_m,
            road_width_m=road_width_m,
            half_width_m=half_width_m,
        )
        if reason and self.phase not in ("cruise", "abort"):
            self.phase = "abort"
            self.phase_t = 0.0
            self.reason = reason
            self._saw_offset = False
        elif reason and self.phase == "cruise" and reason != "junction":
            # A junction does not wipe the cruise timer. The window check
            # refuses to start while a real fork is close. The old test
            # treated every nearby road end as a junction and the timer
            # never reached 20 s, so the 6-minute run stayed in cruise.
            self.reason = reason
            self.phase_t = 0.0
        else:
            if self.phase == "cruise":
                self.reason = ""
            self.phase_t += dt
            self._advance(
                offset_m,
                heading_rad,
                lane_width_m,
                traffic,
                n_lanes=n_lanes,
                window_ok=window_ok,
                start_kind=start_kind,
            )
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
        window_ok: bool = True,
        start_kind: str | None = None,
    ) -> None:
        if self.phase == "cruise" and self.phase_t >= CRUISE_S and not self.reason:
            if not window_ok:
                # The timer elapsed on a curve or a junction. Look again shortly.
                self.phase_t = max(0.0, CRUISE_S - 2.0)
                return
            self._start_next(lane_width_m, traffic, n_lanes, chosen=start_kind)
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

    def peek_kind(self, n_lanes: int | None) -> str:
        """Mostly weaves. Every fourth start may be a lane change."""
        if self._kind % 4 == 3 and n_lanes is not None and int(n_lanes) >= 2:
            return "lane_change"
        return "weave"

    def _start_next(
        self,
        lane_width_m: float | None,
        traffic: bool,
        n_lanes: int | None,
        chosen: str | None = None,
    ) -> None:
        if traffic:
            self.phase_t = 0.0
            self.reason = "traffic"
            return
        kind = chosen if chosen in ("weave", "lane_change") else self.peek_kind(n_lanes)
        self._kind += 1
        if kind == "lane_change" and (n_lanes is None or int(n_lanes) < 2):
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
        spec = targets_for(self.phase, self.direction, lane_width_m, lane_id=0, road_width_m=None, speed_mps=self.speed_mps)
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
    lanes: int | None = None,
    speed_mps: float | None = None,
) -> tuple[float, float] | None:
    """Lane-center offset and heading for this phase, or None when it cannot stay on the road.

    A weave's amplitude and wavelength come from ``weave_plan(speed_mps)``.
    Its room keeps the center WEAVE_MARGIN_M inside the paint and the body
    (EGO_HALF_WIDTH_M) inside the lane, so the planned peak sits EDGE_M
    short of the ``lane_edge`` abort.

    Offset is positive to the right of the current lane center. Heading is
    positive when the path points left of the road. Without a measured road
    width the nominal tag assumes two lanes of ``lane_width_m``.
    """
    if phase not in ACTIVE:
        return (0.0, 0.0)
    lane_w = 3.5 if lane_width_m is None or float(lane_width_m) <= 0.0 else float(lane_width_m)
    road_w = float(road_width_m) if road_width_m is not None and float(road_width_m) > 0.0 else lane_w * 2.0
    direction = 1 if int(direction) >= 0 else -1
    centers = lane_center_offsets(road_w, lanes)
    if not centers:
        return None
    lane_i = max(0, min(int(lane_id), len(centers) - 1))
    lane_here = road_w / float(len(centers))
    if phase == "weave":
        half = max(0.0, min(lane_here * 0.5 - WEAVE_MARGIN_M, lane_here * 0.5 - EGO_HALF_WIDTH_M))
        if half < WEAVE_AMP_LO_M:
            return None
        plan = weave_plan(speed_mps, half)
        if plan is None:
            return None
        amp, wave = plan
        heading = math.atan(amp * 2.0 * math.pi / wave)
        return (direction * amp, -direction * heading)
    if phase == "lane_change":
        nxt = lane_i + direction
        if nxt < 0 or nxt >= len(centers):
            return None
        here = float(centers[lane_i])
        there = float(centers[nxt])
        # Right-hand traffic: the left half of the centerline is oncoming.
        # Crossing into it, or onto the paint at the road edge, is not a lane change.
        if there < -0.25 and here >= -0.25:
            return None
        if abs(there) > road_w * 0.5 - 0.8:
            return None
        delta = there - here
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


def near_junction(
    hit: object,
    roads: list[RoadPoly] | None = None,
    x: float | None = None,
    y: float | None = None,
    heading: float | None = None,
) -> bool:
    """True when a junction or a dead end sits inside the curvature lookahead.

    Without ``roads``, a polyline that ends inside that lookahead counts.
    With ``roads``, a piece boundary that has one colinear continuation does
    not. A second road leaving the tip does.
    """
    if hit is None:
        return True
    if roads and x is not None and y is not None:
        # Same rule as the weave gate: a piece that ends is not a junction
        # when a straight continuation exists. ``discontinuous`` on the hit
        # is only this edge running out, and that was aborting every weave
        # one or two ticks after the gate opened.
        road = _road_of(roads, hit)
        if road is None:
            return True
        span = _span(roads, road, float(x), float(y), heading, float(CURVATURE_AHEAD_M))
        return span[0] < float(CURVATURE_AHEAD_M) - 0.05
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


def _span(
    roads: list[RoadPoly],
    road: RoadPoly,
    x: float,
    y: float,
    heading: float | None,
    distance_m: float,
) -> tuple[float, bool, float]:
    """``(length, fork_m or None, heading_span)`` along this road and its continuations.

    A side road is recorded as ``fork_m`` but the walk follows the straight
    continuation. ``heading_span`` is the largest turn away from the start.
    """
    length, fork_at, span, _samples = _chain(roads, road, x, y, heading, distance_m, 8.0)
    return length, fork_at, span


def _chain(
    roads: list[RoadPoly],
    road: RoadPoly,
    x: float,
    y: float,
    heading: float | None,
    distance_m: float,
    step_m: float,
) -> tuple[float, bool, float, list[tuple[float, float, float, float, float]]]:
    """Walk forward across colinear pieces. A branch ends the walk."""
    samples: list[tuple[float, float, float, float, float]] = []
    visited: list[str] = []
    current: RoadPoly | None = road
    cx, cy = float(x), float(y)
    face = heading
    base = 0.0
    fork_at: float | None = None
    heading_span = 0.0
    h0: float | None = None
    while current is not None and len(visited) < 12 and base < float(distance_m) - 1e-3:
        if current.road_id in visited:
            break
        visited.append(current.road_id)
        piece = _walk_piece(current, cx, cy, face, float(distance_m) - base, step_m)
        if not piece:
            break
        for px, py, fx, fy, s_m in piece:
            samples.append((px, py, fx, fy, s_m + base))
            heading_now = math.atan2(fy, fx)
            if h0 is None:
                h0 = heading_now
            heading_span = max(heading_span, abs(math.atan2(math.sin(heading_now - h0), math.cos(heading_now - h0))))
        base = samples[-1][4]
        if base >= float(distance_m) - 1e-3:
            break
        cont, split = road_ahead_split(roads, current, samples[-1][0], samples[-1][1], math.atan2(samples[-1][3], samples[-1][2]))
        # A side road at this tip is a fork, but the straight continuation
        # is still the road. Stopping here made every west-coast node look
        # like a junction, so the weave gate never opened. The replay of
        # collect3 had no side roads, which is why it started five weaves.
        if split and fork_at is None:
            fork_at = base
        if cont is None:
            break
        cx, cy = samples[-1][0], samples[-1][1]
        face = math.atan2(samples[-1][3], samples[-1][2])
        current = cont
    length = samples[-1][4] if samples else 0.0
    return length, fork_at, heading_span, samples


# A usable excursion window: 60 m whose heading changes slower than this,
# and no fork of this road inside 40 m. 0.01/m is a 100 m radius.
WINDOW_M = 60.0
JUNCTION_AHEAD_M = 40.0
CURVATURE_MAX = 0.01


def gate_reason(
    roads: list[RoadPoly],
    hit: object,
    x: float,
    y: float,
    heading: float | None,
    phase: str,
    speed_mps: float = 0.0,
) -> str:
    """Why ``phase`` cannot start, or ``open``.

    Weave only needs the current lane. A lane change needs a second lane.
    Curvature is the heading change along the next 60 m divided by that
    distance. A nearby road that does not meet this tip is not a junction.
    ``speed_mps`` sizes the weave (``weave_plan``); the window is a distance.
    """
    if phase not in ACTIVE or hit is None or not roads:
        return "no_hit"
    if bool(getattr(hit, "off_road", False)) or bool(getattr(hit, "discontinuous", False)):
        return "off_road"
    road = _road_of(roads, hit)
    if road is None:
        return "no_hit"
    lane_w = getattr(hit, "lane_width_m", None)
    lane_id = int(getattr(hit, "lane_id", 0) or 0)
    directions = [1, -1] if phase == "lane_change" else [1]
    if not any(
        targets_for(
            phase,
            direction,
            None if lane_w is None else float(lane_w),
            lane_id=lane_id,
            road_width_m=float(road.width_m),
            speed_mps=speed_mps if speed_mps else None,
        )
        is not None
        for direction in directions
    ):
        return "narrow"
    length, fork_at, span = _span(roads, road, x, y, heading, WINDOW_M)
    # Weaves stay in the lane, so a driveway does not block them.
    # A lane change does not start when a fork is inside 40 m.
    if phase != "weave" and fork_at is not None and fork_at < JUNCTION_AHEAD_M:
        return "junction"
    if length + 1e-3 < WINDOW_M:
        return "short"
    if span / max(length, 1.0) >= CURVATURE_MAX:
        return "curve"
    return "open"


def fork_ahead_m(
    roads: list[RoadPoly],
    hit: object,
    x: float,
    y: float,
    heading: float | None,
    distance_m: float = WINDOW_M,
) -> float | None:
    """Distance to the first side road on this road's continuation, or None.

    The same walk ``gate_reason`` uses. The weave gate ignores it; the
    collector's ``--weave-guard`` does not.
    """
    if hit is None or not roads:
        return None
    road = _road_of(roads, hit)
    if road is None:
        return None
    _length, fork_at, _span_h = _span(roads, road, x, y, heading, distance_m)
    return None if fork_at is None else float(fork_at)


def route_is_open(
    roads: list[RoadPoly],
    hit: object,
    x: float,
    y: float,
    heading: float | None,
    phase: str,
    speed_mps: float,
) -> bool:
    """True when ``gate_reason`` is ``open``."""
    return gate_reason(roads, hit, x, y, heading, phase, speed_mps) == "open"


def _walk_piece(
    road: RoadPoly,
    x: float,
    y: float,
    heading: float | None,
    distance_m: float,
    step_m: float,
) -> list[tuple[float, float, float, float, float]]:
    """Centerline samples ``(x, y, fx, fy, s)`` on one polyline."""
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
    if travelled < distance - 0.05 and 0 <= idx < len(pts):
        guard2 = 0
        while travelled < distance - 0.05 and 0 <= idx < len(pts) and guard2 < 80:
            guard2 += 1
            nx, ny = pts[idx]
            dx, dy = nx - px, ny - py
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                idx += step_i
                continue
            ux, uy = dx / seg, dy / seg
            need = distance - travelled
            if seg >= need - 1e-9:
                samples.append((px + ux * need, py + uy * need, ux, uy, distance))
                travelled = distance
                break
            px, py = nx, ny
            travelled += seg
            fx, fy = ux, uy
            idx += step_i
    # A step of 8 m toward a 12 m lookahead can finish 3 cm short of the
    # target and leave that distance off the sample list. The junction test
    # then reads 8 m and aborts a weave that still has road ahead.
    if travelled > samples[-1][4] + 0.05:
        samples.append((px, py, fx, fy, travelled))
    return samples


def _road_of(roads: list[RoadPoly], hit: object) -> RoadPoly | None:
    want = str(getattr(hit, "road_id", "") or "")
    if want:
        for road in roads:
            if str(road.road_id) == want:
                return road
    return roads[0] if roads else None


def _lateral(phase: str, s_m: float, base: float, delta: float, wave_m: float = WEAVE_WAVE_M) -> float:
    """Road-center offset along the path. ``delta`` is the signed lane-center excursion."""
    if phase == "weave":
        wave = max(float(wave_m), 1.0)
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
    speed = _speed(speed_mps)
    wave = weave_wave_m(speed)
    # A long highway wave is sent as its first half: out and back to center.
    need = max(WINDOW_M, 0.5 * wave) if move.phase == "weave" else WINDOW_M
    lane_id = int(getattr(hit, "lane_id", 0) or 0)
    lane_w = getattr(hit, "lane_width_m", None)
    # The label counted lanes from lanesLeft/Right. Width/3.5 does not match
    # that on a 5 m two-lane road or a 10 m road, so the path aimed at a
    # different center, or refused the lane id and aborted no_path.
    n_lanes = int(getattr(road, "lanes", 0) or 0)
    lane_count = n_lanes if 1 <= n_lanes <= 8 else None
    _length, _fork, _span_h, samples = _chain(roads, road, x, y, heading, need, PATH_STEP_M)
    if len(samples) < 2 or samples[-1][4] + 1e-3 < need * 0.85:
        return None, int(move.direction)
    for direction in directions:
        spec = targets_for(
            move.phase,
            direction,
            None if lane_w is None else float(lane_w),
            lane_id=lane_id,
            road_width_m=float(road.width_m),
            lanes=lane_count,
            speed_mps=speed,
        )
        if spec is None:
            continue
        delta = float(spec[0])
        centers = lane_center_offsets(float(road.width_m), lane_count)
        if lane_id < 0 or lane_id >= len(centers):
            continue
        base = float(centers[lane_id])
        limit = float(road.width_m) * 0.5 - EDGE_M
        points: list[tuple[float, float]] = []
        ok = True
        for cx, cy, fx, fy, s_m in samples:
            lat = _lateral(move.phase, s_m, base, delta, wave)
            if abs(lat) > limit + 1e-6:
                ok = False
                break
            rx, ry = fy, -fx
            points.append((cx + rx * lat, cy + ry * lat))
        if ok and len(points) >= 2:
            return points, direction
    return None, int(move.direction)
