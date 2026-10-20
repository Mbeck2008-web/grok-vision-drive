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
# The planned weave keeps the body (half width) plus WEAVE_SLACK_M inside the
# painted lane edge: amp <= lane_w / 2 - half_width - WEAVE_SLACK_M. Under
# WEAVE_AMP_LO_M the weave is refused.
WEAVE_SLACK_M = 0.3
# etk800 half width (1.8 m, path_predictor.VEHICLE_WIDTH). The collector reads
# the vehicle bbox at spawn and uses this only, logged, when the bbox read
# fails and the model is etk800; any other model without a bbox stops the run.
ETK800_HALF_WIDTH_M = 0.9
EGO_HALF_WIDTH_M = ETK800_HALF_WIDTH_M  # default for library callers without a measured width
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
# Every shaped move (weave, lane change, recover) is planned at
# v_plan = max(measured, target): the AI target is not what the car does
# (rec5: a weave planned at 11 m/s with the car at 7.8; rec4 highway at 30).
# A lane change or recover with a planning speed eases in and out on a
# cosine ramp long enough to keep the sent line's lateral acceleration
# under MANEUVER_LAT_ACC_MPS2. A weave or lane change is aborted
# ``overspeed`` once the measured speed is OVERSPEED_FRAC over v_plan.
MANEUVER_LAT_ACC_MPS2 = WEAVE_LAT_ACC_MPS2
OVERSPEED_FRAC = 0.15
RAMP_MARGIN = 1.05
SPEED_BANDS = ((12.0, "<12"), (18.0, "12-18"), (24.0, "18-24"), (float("inf"), ">=24"))


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


def weave_length_m(speed_mps: float | None) -> float:
    """Length of the weave line that is sent: whole half-waves, the count nearest WINDOW_M, at least one.

    The line starts and ends on the lane center. At 5 m/s per metre of wave
    a half wave passes WINDOW_M (60 m) above 24 m/s; there the line is one
    half wave (2.5 v metres: out to one side and back), e.g. 82.5 m at 33 m/s.
    """
    half = 0.5 * weave_wave_m(speed_mps)
    k = max(1, int(math.floor(WINDOW_M / half + 0.5)))
    return k * half


def _menger(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Signed curvature through three points (1/m)."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    den = math.hypot(abx, aby) * math.hypot(bcx, bcy) * math.hypot(acx, acy)
    if den < 1e-9:
        return 0.0
    return 2.0 * (abx * bcy - aby * bcx) / den


def path_lat_acc(
    points: list[tuple[float, float]],
    centerline: list[tuple[float, float]],
    speed_mps: float,
) -> tuple[float, float]:
    """``(weave, total)`` peak lateral acceleration of a sent line driven at ``speed_mps``.

    ``total`` is v^2 * |curvature| of the line itself. ``weave`` is v^2 *
    |line curvature - road centerline curvature| at the same samples: what
    the weave adds on top of the road's own bend. The 1.0 m/s^2 limit is on
    ``weave``; the road's bend is the gate's curvature rule, not the weave's.
    """
    v2 = float(speed_mps) ** 2
    weave = total = 0.0
    n = min(len(points), len(centerline))
    for i in range(1, n - 1):
        kp = _menger(points[i - 1], points[i], points[i + 1])
        kr = _menger(centerline[i - 1], centerline[i], centerline[i + 1])
        weave = max(weave, v2 * abs(kp - kr))
        total = max(total, v2 * abs(kp))
    return weave, total


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


def plan_speed(measured_mps: float | None, target_mps: float | None) -> tuple[float | None, float | None, str]:
    """``(v_plan, v_meas, source)``: max(measured, target) when the speed is measured.

    ``source`` is ``measured`` or ``target`` (the larger one), or
    ``target_fallback`` when there is no measured speed.
    """
    meas = _speed(measured_mps)
    target = _speed(target_mps)
    if meas is None:
        return target, None, "target_fallback"
    if target is None or meas >= target:
        return meas, meas, "measured"
    return target, meas, "target"


def overspeed(measured_mps: float | None, v_plan_mps: float | None) -> bool:
    """True when the measured speed is more than OVERSPEED_FRAC above the planning speed."""
    meas, plan = _speed(measured_mps), _speed(v_plan_mps)
    if meas is None or plan is None or plan <= 0.0:
        return False
    return meas > plan * (1.0 + OVERSPEED_FRAC) + 1e-9


def speed_band(speed_mps: float | None) -> str:
    """``<12``, ``12-18``, ``18-24`` or ``>=24`` m/s; ``na`` without a speed."""
    v = _speed(speed_mps)
    if v is None:
        return "na"
    for top, name in SPEED_BANDS:
        if v < top:
            return name
    return SPEED_BANDS[-1][1]


def ramp_length_m(delta_m: float, speed_mps: float | None) -> float | None:
    """Cosine ramp length for a lane change / recover leg of ``delta_m``. None without a speed.

    ``delta * (1 - cos(pi s / L)) / 2`` peaks at heading atan(delta pi / 2L)
    and lateral acceleration delta (pi / L)^2 / 2 * v^2. L is the longer of
    the length that keeps the peak heading at HEADING_TARGET_DEG and the
    length that keeps the acceleration under MANEUVER_LAT_ACC_MPS2 (x RAMP_MARGIN).
    """
    v = _speed(speed_mps)
    if v is None:
        return None
    d = abs(float(delta_m))
    by_heading = max(8.0, d / max(math.tan(_deg(HEADING_TARGET_DEG)), 1e-3)) * math.pi / 2.0
    by_acc = math.pi * v * math.sqrt(d / (2.0 * MANEUVER_LAT_ACC_MPS2)) * RAMP_MARGIN
    return max(by_heading, by_acc)


def maneuver_length_m(phase: str, delta_m: float, speed_mps: float | None) -> float:
    """Road a lane change / recover line needs: WINDOW_M, or the ramps plus holds at this speed."""
    ramp = ramp_length_m(delta_m, speed_mps)
    if ramp is None:
        return WINDOW_M
    if phase == "lane_change":
        return max(WINDOW_M, ramp + 16.0)
    if phase == "recover":
        return max(WINDOW_M, 2.0 * ramp + 16.0)
    return WINDOW_M


def ramp_heading_reached(delta_m: float, ramp_m: float, covered_m: float) -> float:
    """Peak heading (rad, unsigned) of the cosine ramp ``delta (1 - cos(pi s / L)) / 2`` over s in [0, covered].

    The slope peaks at s = L / 2, so from there on this is the full ramp's
    peak, atan(delta pi / 2L). Before it, the heading the line has reached.
    """
    length = max(float(ramp_m), 1.0)
    s = min(max(float(covered_m), 0.0), 0.5 * length)
    return math.atan(abs(float(delta_m)) * math.pi / (2.0 * length) * math.sin(math.pi * s / length))


WEAVE_STATS_NOTES = {
    "started": "moves whose shaped line was sent, by phase (weave, lane_change, recover)",
    "refused": "blocked stretches, by the first refusal reason of the stretch: a run of consecutive "
               "refused start attempts, ended by an excursion start, the segment end or a respawn",
    "refused_attempts": "every refused start attempt, by reason (the gate retries about every 2 s)",
    "aborted": "moves that had started and were ended early, by phase then reason (overspeed)",
    "lane_change_complete": "lane changes whose recover began after the car covered the planned ramp",
    "lane_change_incomplete": "lane changes whose recover began (after LANE_CHANGE_S) before the car "
                              "covered the planned ramp; their label heading is the peak the line reached",
}


class WeaveStats:
    """Counters for collect.json ``weave_stats``: starts, refusals, aborts, speed bands, one-sided sides."""

    def __init__(self) -> None:
        self.started: dict[str, int] = {}
        self.refused: dict[str, int] = {}
        self.refused_attempts: dict[str, int] = {}
        self.aborted: dict[str, dict[str, int]] = {}
        self.bands: dict[str, dict[str, int]] = {
            name: {"started": 0, "refused": 0, "refused_attempts": 0, "aborted": 0} for _top, name in SPEED_BANDS
        }
        self.sides: dict[str, int] = {"left": 0, "right": 0}
        self.overspeed_logged = 0
        self.lane_change_complete = 0
        self.lane_change_incomplete = 0
        self._stretch = False

    def start(self, phase: str, v_plan: float | None, *, direction: int = 1, one_sided: bool = False) -> None:
        self.started[phase] = self.started.get(phase, 0) + 1
        self._stretch = False
        band = speed_band(v_plan)
        if band in self.bands:
            self.bands[band]["started"] += 1
        if phase == "weave" and one_sided:
            # Offset is positive to the right; the first half-wave goes to ``direction``.
            self.sides["right" if int(direction) >= 0 else "left"] += 1

    def refuse(self, reason: str, v_plan: float | None) -> None:
        """One refused attempt. ``refused`` counts it only when it opens a blocked stretch."""
        key = str(reason or "unknown")
        self.refused_attempts[key] = self.refused_attempts.get(key, 0) + 1
        band = speed_band(v_plan)
        if band in self.bands:
            self.bands[band]["refused_attempts"] += 1
        if self._stretch:
            return
        self._stretch = True
        self.refused[key] = self.refused.get(key, 0) + 1
        if band in self.bands:
            self.bands[band]["refused"] += 1

    def end_stretch(self) -> None:
        """Segment end or respawn: the next refusal opens a new stretch."""
        self._stretch = False

    def abort(self, phase: str, reason: str, v_plan: float | None) -> None:
        per = self.aborted.setdefault(str(phase), {})
        key = str(reason or "unknown")
        per[key] = per.get(key, 0) + 1
        band = speed_band(v_plan)
        if band in self.bands:
            self.bands[band]["aborted"] += 1

    def lane_change_end(self, complete: bool) -> None:
        if complete:
            self.lane_change_complete += 1
        else:
            self.lane_change_incomplete += 1

    def summary(self) -> dict[str, object]:
        return {
            "started": dict(self.started),
            "started_total": sum(self.started.values()),
            "refused": dict(self.refused),
            "refused_total": sum(self.refused.values()),
            "refused_attempts": dict(self.refused_attempts),
            "refused_attempts_total": sum(self.refused_attempts.values()),
            "aborted": {k: dict(v) for k, v in self.aborted.items()},
            "aborted_total": sum(sum(v.values()) for v in self.aborted.values()),
            "lane_change_complete": int(self.lane_change_complete),
            "lane_change_incomplete": int(self.lane_change_incomplete),
            "by_speed_band": {k: dict(v) for k, v in self.bands.items()},
            "half_wave_sides": dict(self.sides),
            "recover_overspeed_logged": int(self.overspeed_logged),
        }


def _speed(speed_mps: float | None) -> float | None:
    if speed_mps is None:
        return None
    try:
        v = abs(float(speed_mps))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def body_past_paint_edge(
    paint_lane: tuple[float, float] | None,
    half_width_m: float,
) -> bool:
    """True when either body edge is more than EDGE_M outside the painted lane.

    FROM: the ego body edges, x = -half_width and +half_width in the vehicle
    frame (x right, origin at the BeamNG state ``pos`` ref node, the body's
    lateral center). AGAINST: ``paint_lane`` = (left_x, right_x), the painted
    lane lines at the car (y = 0) from ``PaintedLanes.lane_at``: the level's
    DecalRoad line decals in world space, the same source the lane labels use.
    Not the navgraph: on west_coast_usa most navgraph roads are road_invisible
    AI splines whose width is the whole corridor, split into 1/1 lanes.
    """
    if paint_lane is None:
        return False
    left, right = float(paint_lane[0]), float(paint_lane[1])
    hw = float(half_width_m)
    return hw > right + EDGE_M or -hw < left - EDGE_M


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
    paint_lane: tuple[float, float] | None = None,
    paint_lost: bool = False,
) -> str:
    """Why an excursion must stop. Empty when the move may continue.

    A weave is aborted for ``lane_edge`` once a body edge is EDGE_M past the
    painted lane edge (``body_past_paint_edge``), and for ``no_paint`` when
    the collector has lost the painted lane at the car (``paint_lost``).
    ``half_width_m`` None skips both (library callers). A weave,
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
    if phase == "weave" and half_width_m is not None:
        if body_past_paint_edge(paint_lane, half_width_m):
            return "lane_edge"
        if paint_lost:
            return "no_paint"
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
        paint_lane: tuple[float, float] | None = None,
        paint_lost: bool = False,
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
            paint_lane=paint_lane,
            paint_lost=paint_lost,
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
        start = self.start_move(chosen, n_lanes)
        self._kind += 1
        self.phase = start.phase
        self.phase_t = 0.0
        self.reason = ""
        self._saw_offset = False
        self.direction = int(start.direction)

    def start_move(self, chosen: str | None, n_lanes: int | None) -> Excursion:
        """The phase and direction the next start takes, without starting it.

        ``_start_next`` uses this, so the gate (``sent_line_args`` with this
        move) judges the line the start will build: a lane change flips the
        last direction, a weave keeps it (``clock.direction`` is the side the
        last sent line used).
        """
        kind = chosen if chosen in ("weave", "lane_change") else self.peek_kind(n_lanes)
        if kind == "lane_change" and (n_lanes is None or int(n_lanes) < 2):
            kind = "weave"
        direction = -int(self.direction or 1) if kind == "lane_change" else self.direction
        return Excursion(kind, 0.0, 0.0, direction, "")

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
    weave_lane_w: float | None = None,
    half_width_m: float | None = None,
) -> tuple[float, float] | None:
    """Lane-center offset and heading for this phase, or None when it cannot stay on the road.

    A weave's amplitude and wavelength come from ``weave_plan(speed_mps)``
    with room ``lane_w / 2 - half_width - WEAVE_SLACK_M``; ``lane_w`` is the
    painted lane width (``weave_lane_w``) when given, else the navgraph lane
    (library callers). Under WEAVE_AMP_LO_M of room the weave is refused.

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
        lane_w_weave = float(weave_lane_w) if weave_lane_w is not None and float(weave_lane_w) > 0.0 else lane_here
        hw = EGO_HALF_WIDTH_M if half_width_m is None else float(half_width_m)
        room = lane_w_weave * 0.5 - hw - WEAVE_SLACK_M
        if room < WEAVE_AMP_LO_M - 1e-9:
            return None
        plan = weave_plan(speed_mps, room)
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
    """Vehicle-VM Lua for this phase. A path leaves lane-keeping; cruise puts it back.

    With a path these are only the lines sent *before* the line itself
    (``ai_line.pre_line_commands``). The line goes out with beamngpy's
    ``vehicle.ai.set_line`` (``ai_line.send_and_ack``) and nothing is sent
    after it: 0.39.4.0's ``ai`` has no ``setLine``, and a later
    ``setMode``/``setState`` would drop the route the line installs.
    Cruise and abort put lane-follow back: ``setMode`` span or traffic
    (rebuilds the route from the navgraph), ``driveInLane("on")``, speed.
    """
    speed = float(speed_mps)
    if phase in ACTIVE and path:
        from python.collect.ai_line import pre_line_commands

        return pre_line_commands()
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
            if samples and s_m < 1e-6:
                # A continuation's first sample is the previous tip again: same
                # s, a few degrees of heading apart. Offset by the lane center,
                # that pair is a lateral step over ~0 m, which path_lat_acc
                # read as 13-38 m/s^2 and refused the weave.
                continue
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


@dataclass(frozen=True)
class SentLineArgs:
    """Everything ``excursion_path`` reads to build a sent line, but the roads.

    ``phase`` and ``direction`` are the only parts of the move it reads. The
    collector builds one of these for the gate (with ``ExcursionClock.
    start_move``) and one for the start (with the clock's move), both through
    ``sent_line_args``, so the gate judges the line the start sends.
    """

    x: float
    y: float
    heading: float | None
    phase: str
    direction: int
    hit: object
    speed_mps: float | None
    paint_lane: tuple[float, float] | None
    half_width_m: float | None


def sent_line_args(
    x: float,
    y: float,
    heading: float | None,
    move: Excursion,
    hit: object,
    speed_mps: float | None,
    *,
    paint_lane: tuple[float, float] | None,
    half_width_m: float | None,
) -> SentLineArgs:
    """The one place the collector's gate and start build their path inputs."""
    return SentLineArgs(
        x=float(x),
        y=float(y),
        heading=None if heading is None else float(heading),
        phase=str(move.phase),
        direction=1 if int(move.direction) >= 0 else -1,
        hit=hit,
        speed_mps=None if speed_mps is None else float(speed_mps),
        paint_lane=None if paint_lane is None else (float(paint_lane[0]), float(paint_lane[1])),
        half_width_m=None if half_width_m is None else float(half_width_m),
    )


def build_sent_line(
    roads: list[RoadPoly], args: SentLineArgs, info: dict | None = None,
) -> tuple[list[tuple[float, float]] | None, int]:
    """``excursion_path`` on ``args``: what the gate checks and the start sends."""
    return excursion_path(
        roads, args.x, args.y, args.heading, Excursion(args.phase, 0.0, 0.0, args.direction, ""),
        args.hit, args.speed_mps,  # type: ignore[arg-type]
        paint_lane=args.paint_lane, half_width_m=args.half_width_m, info=info,
    )


def gate_reason(
    roads: list[RoadPoly],
    hit: object,
    x: float,
    y: float,
    heading: float | None,
    phase: str,
    speed_mps: float = 0.0,
    *,
    weave_lane_w: float | None = None,
    half_width_m: float | None = None,
    paint_lane: tuple[float, float] | None = None,
    line: SentLineArgs | None = None,
) -> str:
    """Why ``phase`` cannot start, or ``open``.

    Weave only needs the current lane. A lane change needs a second lane.
    Curvature is the heading change along the next 60 m divided by that
    distance. A nearby road that does not meet this tip is not a junction.
    ``speed_mps`` sizes the weave (``weave_plan``); a weave needs
    ``max(WINDOW_M, weave_length_m(v))`` of road. ``weave_lane_w`` /
    ``half_width_m`` are the painted lane width and the ego half width.

    The gate judges the line that would be sent, not a separate coarse one.
    The window walk uses the sent-line builder (``_chain`` at PATH_STEP_M,
    the walk ``excursion_path`` makes), and once that window is clear the
    line itself is built with ``excursion_path`` from the same inputs
    (``paint_lane`` is the painted lane at the car, as the collector passes
    it). A line it refuses closes the gate with that reason (no_path,
    narrow, road_edge, lat_acc, line_kink, line_gap). ``_span`` (8 m) is only
    for ``near_junction`` and ``fork_ahead_m``.

    ``line`` is the start's own path input (``sent_line_args`` with
    ``ExcursionClock.start_move``): its direction is the side the start will
    try first, which decides the line on an asymmetric bend. Without it
    (library callers) the line is built for direction +1 from the arguments
    here. The cheap checks (narrow, short, junction, curve) run first; the
    full line is built only once they pass.
    """
    if line is not None and line.phase != phase:
        raise ValueError(f"gate_reason({phase!r}) given a {line.phase!r} line")
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
            weave_lane_w=weave_lane_w if phase == "weave" else None,
            half_width_m=half_width_m,
        )
        is not None
        for direction in directions
    ):
        return "narrow"
    need = max(WINDOW_M, weave_length_m(speed_mps if speed_mps else None)) if phase == "weave" else WINDOW_M
    if phase != "weave" and speed_mps:
        for direction in directions:
            spec = targets_for(phase, direction, None if lane_w is None else float(lane_w), lane_id=lane_id,
                               road_width_m=float(road.width_m))
            if spec is not None:
                need = max(need, maneuver_length_m(phase, float(spec[0]), speed_mps))
    length, fork_at, span, _samples = _chain(roads, road, x, y, heading, need, PATH_STEP_M)
    # Weaves stay in the lane, so a driveway does not block them.
    # A lane change does not start when a fork is inside 40 m.
    if phase != "weave" and fork_at is not None and fork_at < JUNCTION_AHEAD_M:
        return "junction"
    if length + 1e-3 < need:
        return "short"
    if span / max(length, 1.0) >= CURVATURE_MAX:
        return "curve"
    if line is None:
        line = sent_line_args(x, y, heading, Excursion(phase, 0.0, 0.0, 1, ""), hit,
                              speed_mps if speed_mps else None,
                              paint_lane=paint_lane, half_width_m=half_width_m)
    info: dict = {}
    points, _used = build_sent_line(roads, line, info)
    if points is None:
        return str(info.get("refused") or "no_path")
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

    The 8 m ``_span`` walk (the same roads and joins ``gate_reason`` walks,
    coarser samples). The weave gate ignores it; the collector's
    ``--weave-guard`` does not.
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


def _lateral(
    phase: str, s_m: float, base: float, delta: float, wave_m: float = WEAVE_WAVE_M, ramp_m: float | None = None,
) -> float:
    """Road-center offset along the path. ``delta`` is the signed lane-center excursion.

    With ``ramp_m`` (a planning speed) a lane change / recover leg is a
    cosine ramp of that length; without it, the old linear ramp.
    """
    if phase == "weave":
        wave = max(float(wave_m), 1.0)
        return base + delta * math.sin(2.0 * math.pi * float(s_m) / wave)
    if ramp_m is not None:
        ramp = max(float(ramp_m), 1.0)
        s = float(s_m)

        def ease(u: float) -> float:
            u = 0.0 if u <= 0.0 else 1.0 if u >= 1.0 else u
            return 0.5 * (1.0 - math.cos(math.pi * u))

        if phase == "lane_change":
            return base + float(delta) * ease(s / ramp)
        if s < ramp + 8.0:
            return base + float(delta) * ease(s / ramp)
        return base + float(delta) * (1.0 - ease((s - ramp - 8.0) / ramp))
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


# Sent-line backstop. A line that survives every other check is still
# refused when its points jump. Both caps were measured on 65 rec5b excursion
# lines replayed over the west_coast_usa road dump (61 legitimate, 4 with a
# wrong join: rows 3281, 13317, 24338, 27739).
#
# LINE_KINK_M caps the second difference at each sample: the perpendicular
# distance from the line through its two neighbours, normalized to a 4 m step
# by 16 / (a * b), with each neighbour spacing a, b clamped to 1..4 m. The
# legitimate max was 0.367 m (all 61 lines), so 0.6 m is 1.6x that. The
# smallest jump seen was 0.92 m (row 24338). That jump sits at the very end
# of its line, where only the last segment shows it, so a future end-of-line
# jump could read smaller than 0.6 m and pass. A clean sideways step of d at
# 4 m spacing reads about d / 2 (d * 4 / sqrt(64 + d^2)), so steps under
# about 1.2 m pass this cap too.
LINE_KINK_M = 0.6
# LINE_GAP_M caps the distance between adjacent sent points. It catches only
# big jumps: the smallest jump seen was 5.65 m and the legitimate max gap was
# 4.99 m (PATH_STEP_M is 4 m; a piece end adds up to ~1 m). It is NOT a check
# for a second line laid beside the first: a parallel road 3.5 m over, joined
# with ordinary 4 m steps, passes it. LINE_KINK_M is the check for that.
LINE_GAP_M = 6.0
KINK_SPACING_M = (1.0, 4.0)


def line_kinks_m(points: list[tuple[float, float]]) -> list[float]:
    """Normalized second difference at each interior point (see LINE_KINK_M)."""
    lo, hi = KINK_SPACING_M
    out: list[float] = []
    for i in range(1, len(points) - 1):
        ax, ay = points[i - 1]
        bx, by = points[i]
        cx, cy = points[i + 1]
        ux, uy = cx - ax, cy - ay
        chord = math.hypot(ux, uy)
        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        if chord < 1e-9:
            dev = a  # the line folds back on itself
        else:
            dev = abs((bx - ax) * uy - (by - ay) * ux) / chord
        out.append(dev * hi * hi / (min(hi, max(lo, a)) * min(hi, max(lo, b))))
    return out


def sent_line_fault(points: list[tuple[float, float]]) -> tuple[str, float] | None:
    """``("line_kink", m)`` or ``("line_gap", m)`` when the sent line jumps, else None."""
    gaps = [math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)]
    if gaps and max(gaps) > LINE_GAP_M:
        return "line_gap", max(gaps)
    kinks = line_kinks_m(points)
    if kinks and max(kinks) > LINE_KINK_M:
        return "line_kink", max(kinks)
    return None


def excursion_path(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading: float | None,
    move: Excursion,
    hit: object,
    speed_mps: float,
    *,
    paint_lane: tuple[float, float] | None = None,
    half_width_m: float | None = None,
    info: dict | None = None,
) -> tuple[list[tuple[float, float]] | None, int]:
    """World points for the AI line (``vehicle.ai.set_line``), and the direction that stays on the road.

    The second value flips when the requested lane change would leave the
    pavement and the other side would not. ``(None, direction)`` means do
    not send a line.

    A weave line is ``weave_length_m(v)`` long. With ``paint_lane`` (painted
    lane at the car, vehicle frame) it is centred on the painted lane center
    and sized from the painted width. The sent line is then checked:
    ``path_lat_acc`` of the actual points over the road centerline must stay
    within WEAVE_LAT_ACC_MPS2; a line over it is rebuilt with the amplitude
    scaled down, and refused (None) once that is under WEAVE_AMP_LO_M.
    ``info`` gets amp_m, wave_m, length_m, lat_acc_mps2, total_lat_acc_mps2.
    A lane change / recover with a speed is a cosine ramp sized by
    ``ramp_length_m`` and checked the same way (lengthened, else refused);
    ``info`` gets delta_m, ramp_m, length_m, heading_rad and the accelerations.
    A refusal sets ``info["refused"]`` (no_path, narrow, road_edge, lat_acc,
    line_gap, line_kink). The last two are the sent-line backstop
    (``sent_line_fault``): a line that jumps between points is never sent;
    ``info`` then also gets ``line_fault_m``.
    """
    if move.phase not in ACTIVE or hit is None or not roads:
        if info is not None:
            info.update(refused="no_path")
        return None, int(move.direction)
    road = _road_of(roads, hit)
    if road is None:
        if info is not None:
            info.update(refused="no_path")
        return None, int(move.direction)
    directions = [1 if int(move.direction) >= 0 else -1]
    if move.phase == "lane_change":
        directions.append(-directions[0])
    speed = _speed(speed_mps)
    wave = weave_wave_m(speed)
    need = weave_length_m(speed) if move.phase == "weave" else WINDOW_M
    paint_w = None
    paint_base = None
    if move.phase == "weave" and paint_lane is not None:
        paint_w = float(paint_lane[1]) - float(paint_lane[0])
        off = getattr(hit, "offset_m", None)
        if off is not None:
            paint_base = float(off) + 0.5 * (float(paint_lane[0]) + float(paint_lane[1]))
    lane_id = int(getattr(hit, "lane_id", 0) or 0)
    lane_w = getattr(hit, "lane_width_m", None)
    # The label counted lanes from lanesLeft/Right. Width/3.5 does not match
    # that on a 5 m two-lane road or a 10 m road, so the path aimed at a
    # different center, or refused the lane id and aborted no_path.
    n_lanes = int(getattr(road, "lanes", 0) or 0)
    lane_count = n_lanes if 1 <= n_lanes <= 8 else None
    if move.phase != "weave" and speed is not None:
        # The ramps are sized at the planning speed: a fast lane change needs more road.
        for direction in directions:
            spec = targets_for(move.phase, direction, None if lane_w is None else float(lane_w),
                               lane_id=lane_id, road_width_m=float(road.width_m), lanes=lane_count)
            if spec is not None:
                need = max(need, maneuver_length_m(move.phase, float(spec[0]), speed))
    _length, _fork, _span_h, samples = _chain(roads, road, x, y, heading, need, PATH_STEP_M)
    if len(samples) < 2 or samples[-1][4] + 1e-3 < need * 0.85:
        if info is not None:
            info.update(refused="no_path", need_m=need, chain_m=samples[-1][4] if samples else 0.0)
        return None, int(move.direction)

    def _backstop(
        points: list[tuple[float, float]], direction: int,
    ) -> tuple[list[tuple[float, float]] | None, int]:
        # The line that would be sent, after every other check: refuse it if it jumps.
        fault = sent_line_fault(points)
        if fault is None:
            return points, direction
        if info is not None:
            info.update(refused=fault[0], line_fault_m=fault[1])
        return None, int(move.direction)

    refused = "narrow"
    for direction in directions:
        spec = targets_for(
            move.phase,
            direction,
            None if lane_w is None else float(lane_w),
            lane_id=lane_id,
            road_width_m=float(road.width_m),
            lanes=lane_count,
            speed_mps=speed,
            weave_lane_w=paint_w,
            half_width_m=half_width_m,
        )
        if spec is None:
            continue
        delta = float(spec[0])
        centers = lane_center_offsets(float(road.width_m), lane_count)
        if lane_id < 0 or lane_id >= len(centers):
            continue
        base = float(centers[lane_id]) if paint_base is None else paint_base
        limit = float(road.width_m) * 0.5 - EDGE_M
        centerline = [(cx, cy) for cx, cy, _fx, _fy, _s in samples]
        ramp = None if move.phase == "weave" else ramp_length_m(delta, speed)
        for _try in range(4):
            points: list[tuple[float, float]] = []
            ok = True
            for cx, cy, fx, fy, s_m in samples:
                lat = _lateral(move.phase, s_m, base, delta, wave, ramp)
                if abs(lat) > limit + 1e-6:
                    ok = False
                    break
                rx, ry = fy, -fx
                points.append((cx + rx * lat, cy + ry * lat))
            if not ok or len(points) < 2:
                refused = "road_edge"
                break
            if speed is None:
                return _backstop(points, direction)
            if move.phase != "weave":
                move_acc, total_acc = path_lat_acc(points, centerline, speed)
                legs = 1.0 if move.phase == "lane_change" else 2.0
                if move_acc <= MANEUVER_LAT_ACC_MPS2 + 1e-9:
                    if info is not None:
                        info.update(delta_m=delta, ramp_m=ramp, length_m=samples[-1][4],
                                    heading_rad=math.atan(abs(delta) * math.pi / (2.0 * float(ramp))),
                                    lat_acc_mps2=move_acc, total_lat_acc_mps2=total_acc, shrunk=_try)
                    return _backstop(points, direction)
                # Sampling slop over the analytic ramp: lengthen it, if the road allows.
                ramp = float(ramp) * math.sqrt(move_acc / MANEUVER_LAT_ACC_MPS2) * 1.03
                if legs * ramp + (0.0 if legs == 1.0 else 8.0) > samples[-1][4]:
                    if info is not None:
                        info.update(refused="lat_acc", lat_acc_mps2=move_acc)
                    return None, int(move.direction)
                continue
            weave_acc, total_acc = path_lat_acc(points, centerline, speed)
            if weave_acc <= WEAVE_LAT_ACC_MPS2 + 1e-9:
                if info is not None:
                    info.update(amp_m=abs(delta), wave_m=wave, length_m=samples[-1][4],
                                lat_acc_mps2=weave_acc, total_lat_acc_mps2=total_acc, shrunk=_try)
                return _backstop(points, direction)
            delta *= 0.97 * WEAVE_LAT_ACC_MPS2 / weave_acc
            if abs(delta) < WEAVE_AMP_LO_M - 1e-9:
                if info is not None:
                    info.update(refused="lat_acc", lat_acc_mps2=weave_acc)
                return None, int(move.direction)
        else:
            if speed is not None:
                refused = "lat_acc"
    if info is not None:
        info.setdefault("refused", refused)
    return None, int(move.direction)
