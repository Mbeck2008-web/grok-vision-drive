"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from python.perception.detect import Detection, pose_ok

# speed_mps on a track is road speed, not |Δrange/Δt|.
_SPEED_CAP = 40.0
# Constant-velocity Kalman on (range, range-rate). A ±2 m grab is about
# 1.2 m rms. 2 m of measurement sigma left the worst 30 m stop short of
# a 2 m margin; 1.5 m still treats that grab as noise and lets a real
# close through. Process noise is white acceleration so a lead that
# brakes can change its range-rate. There is no net-close gate and no
# flat-gap distance wall.
MEASURE_SIGMA_M = 1.5
ACCEL_SIGMA_MPS2 = 5.0
# Unknown range-rate on the first sample. The planner treats this as the
# prior standard deviation, not as a stopped car.
PRIOR_RATE_SIGMA_MPS = 7.0
# Road speed stays unpublished until the rate variance has collapsed.
# A 4 m step inside ±2 m still moves the mean, and publishing that as a
# road speed made time-to-collision a full brake. The brake law reads the
# mean and the sigma the whole time; only speed_mps waits.
_RATE_VALID_SIGMA = 2.2
# A same-lane lead can jump 4 m between grabs when each one is ±2 m off.
# The old 4 m gate missed that jump, left the first track coasting, and
# the coast kept the brake it had already armed.
_MATCH_GATE_M = 6.0
# Long enough for a route-gap slope a couple of seconds behind the latest grab.
_SAMPLE_KEEP_S = 2.6
_SAMPLE_KEEP = 48
# A same-track swap inside the match gate. 2.5 m and 4 m reset the route
# baseline. A 1.6 m step is smaller than this and has to fade on its own
# once the filtered rate is flat again.
_ROUTE_RESET_M = 2.2
# Do not snap the range filter on a ±2 m grab. Two grabs can differ by
# 4 m, and throwing those updates out left a real 2 m/s close looking
# like a pace. 5 m is past that grab and inside the match gate, so a
# same-track teleport still does not become a new closing speed.
_RATE_SNAP_M = 5.0
# Match the lead's road speed only after the filtered rate has stayed
# at least this fast for _CLOSE_HOLD_S. A correlated wander crosses
# 1 m/s of closing. It does not stay at 1.5.
_CLOSE_ARM_MPS = -1.5
_CLOSE_OPEN_MPS = 0.3


def lead_road_speed(ego_speed_mps: float, dy_m: float, dt_s: float) -> tuple[float, float]:
    """Road speed of a lead, and the signed range-rate that produced it.

    ``dy_m`` is this frame's forward range minus the last measured range.
    A closing gap is negative. Road speed is ego speed plus that rate.
    The absolute value of the rate is not a road speed: a stopped car
    approached at 12 m/s has range-rate −12 and road speed 0.
    """
    ego = max(0.0, float(ego_speed_mps))
    dt = max(1e-3, float(dt_s))
    if dy_m < 0.0:
        rate = -abs(float(dy_m) / dt)
    elif dy_m > 0.0:
        rate = abs(float(dy_m) / dt)
    else:
        rate = 0.0
    road = max(0.0, min(_SPEED_CAP, ego + rate))
    return float(road), float(rate)


def _kalman_update(
    range_m: float,
    rate_mps: float,
    p00: float,
    p01: float,
    p11: float,
    measured_m: float,
    dt_s: float,
) -> tuple[float, float, float, float, float]:
    """Joseph-form update of a constant-velocity range filter.

    State is forward range and its rate. A negative rate is closing.
    Process noise is white acceleration so a braking lead is allowed.
    """
    dt = max(1e-3, float(dt_s))
    range_m = range_m + dt * rate_mps
    q = ACCEL_SIGMA_MPS2 * ACCEL_SIGMA_MPS2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    p00 = p00 + 2.0 * dt * p01 + dt2 * p11 + q * dt4 / 4.0
    p01 = p01 + dt * p11 + q * dt3 / 2.0
    p11 = p11 + q * dt2
    meas = MEASURE_SIGMA_M * MEASURE_SIGMA_M
    innov_var = p00 + meas
    if innov_var < 1e-9:
        innov_var = 1e-9
    k0 = p00 / innov_var
    k1 = p01 / innov_var
    innov = float(measured_m) - range_m
    range_m = range_m + k0 * innov
    rate_mps = rate_mps + k1 * innov
    # (I - K H) P (I - K H)^T + K R K^T, H = [1, 0].
    a = 1.0 - k0
    b = -k1
    m00 = a * p00
    m01 = a * p01
    m10 = b * p00 + p01
    m11 = b * p01 + p11
    n00 = m00 * a + k0 * k0 * meas
    n01 = m00 * b + m01 + k0 * k1 * meas
    n10 = m10 * a + k1 * k0 * meas
    n11 = m10 * b + m11 + k1 * k1 * meas
    p01 = 0.5 * (n01 + n10)
    p00 = max(n00, 1e-9)
    p11 = max(n11, 1e-9)
    lim = math.sqrt(p00 * p11)
    if p01 > lim:
        p01 = lim
    elif p01 < -lim:
        p01 = -lim
    return float(range_m), float(rate_mps), float(p00), float(p01), float(p11)


@dataclass
class Track:
    id: int
    cls: str
    x: float
    y: float
    yaw: float
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    conf: float = 0.0
    # Last measured forward range. Range-rate uses this, not the coasted y.
    measured_y: float = 0.0
    range_rate_mps: float = 0.0
    range_rate_sigma: float = PRIOR_RATE_SIGMA_MPS
    range_rate_valid: bool = False
    # Seconds the filtered rate has stayed at least 1.5 m/s closing.
    # A short blip is not a lead we have to match yet.
    close_hold_s: float = 0.0
    # (time, measured forward range). Kept so a dropped window can still
    # report how far the range moved. The rate itself is the Kalman state.
    samples: list[tuple[float, float]] | None = None
    # (time, route-arc station) filled by the planner. A new track starts
    # empty, which is the reset on a track-id change. Not a shared default.
    route_gaps: list | None = None
    # [locked lead decel m/s², seconds the curvature has been hot,
    # cooldown-until]. The planner writes this. A swap clears it.
    drop_state: list | None = None
    # Filter state. None until the first measurement.
    _kf_r: float = 0.0
    _kf_rd: float = 0.0
    _kf_p00: float = 0.0
    _kf_p01: float = 0.0
    _kf_p11: float = 0.0
    _kf_n: int = 0
    _kf_t: float | None = None

    def __post_init__(self) -> None:
        if self.route_gaps is None:
            self.route_gaps = []
        if self.drop_state is None:
            self.drop_state = [0.0, 0.0, 0.0]


class IoUTracker:
    def __init__(self, max_misses: int = 5, gate_m: float = _MATCH_GATE_M) -> None:
        self.max_misses = max_misses
        self.gate_m = gate_m
        self._next_id = 1
        self.tracks: list[Track] = []
        self._last_t: float | None = None

    def update(
        self,
        dets: list[Detection],
        now: float,
        ego_speed_mps: float = 0.0,
    ) -> list[Track]:
        dt = 0.1 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now
        ego = max(0.0, min(_SPEED_CAP, float(ego_speed_mps)))
        # Drop non-finite poses before match or spawn. A NaN range is not a
        # missed car and it is not a new car.
        dets = [d for d in dets if pose_ok(d)]
        # predict in the ego frame. Road speed minus ego is range-rate.
        # The step stays small so a bad speed cannot leave the match gate.
        for tr in self.tracks:
            tr.x += 0.0  # lateral hold
            tr.y += (tr.speed_mps - ego) * dt * 0.15
            tr.age += 1
            tr.misses += 1

        unmatched = set(range(len(dets)))
        for tr in self.tracks:
            best_i, best_d = None, self.gate_m
            for i in list(unmatched):
                d = dets[i]
                if d.cls != tr.cls:
                    continue
                dist = ((d.x - tr.x) ** 2 + (d.y - tr.y) ** 2) ** 0.5
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is None:
                continue
            d = dets[best_i]
            unmatched.discard(best_i)
            if tr.samples is None:
                tr.samples = []
            if tr.route_gaps is None:
                tr.route_gaps = []
            if tr.drop_state is None:
                tr.drop_state = [0.0, 0.0, 0.0]
            # A swap stays one track (the match gate is 6 m) but it is not
            # the same gap. Drop the route baseline so the jump is not a brake.
            if tr.samples and abs(float(d.y) - float(tr.samples[-1][1])) >= _ROUTE_RESET_M:
                tr.route_gaps.clear()
                for i in range(len(tr.drop_state)):
                    tr.drop_state[i] = 0.0
            tr.samples.append((float(now), float(d.y)))
            _trim_samples(tr.samples, float(now))
            _measure_track(tr, float(d.y), float(now), ego)
            tr.measured_y = d.y
            tr.x, tr.y, tr.yaw = d.x, d.y, d.yaw
            tr.conf = d.conf
            tr.hits += 1
            tr.misses = 0

        for i in unmatched:
            d = dets[i]
            # No rate yet. Do not invent ego speed: a first sample is not
            # a measurement, and storing ego made a stopped lead look matched.
            self.tracks.append(
                Track(
                    id=self._next_id,
                    cls=d.cls,
                    x=d.x,
                    y=d.y,
                    yaw=d.yaw,
                    conf=d.conf,
                    speed_mps=0.0,
                    measured_y=d.y,
                    range_rate_mps=0.0,
                    range_rate_sigma=PRIOR_RATE_SIGMA_MPS,
                    range_rate_valid=False,
                    samples=[(float(now), float(d.y))],
                    _kf_r=float(d.y),
                    _kf_rd=0.0,
                    _kf_p00=MEASURE_SIGMA_M * MEASURE_SIGMA_M,
                    _kf_p01=0.0,
                    _kf_p11=PRIOR_RATE_SIGMA_MPS * PRIOR_RATE_SIGMA_MPS,
                    _kf_n=1,
                    _kf_t=float(now),
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return list(self.tracks)

    def as_dicts(self) -> list[dict]:
        return [
            {
                "id": t.id,
                "class": t.cls,
                "x": t.x,
                "y": t.y,
                "yaw": t.yaw,
                "speed_mps": t.speed_mps,
                "range_rate_mps": t.range_rate_mps,
                "range_rate_sigma": t.range_rate_sigma,
                "range_rate_valid": bool(t.range_rate_valid),
                # Positive when the retained window's range has decreased.
                "range_drop_m": _range_drop_m(t.samples),
                # Same objects the planner appends to. A copy would forget
                # the route gap on the next frame.
                "samples": t.samples if t.samples is not None else [],
                "route_gaps": t.route_gaps if t.route_gaps is not None else [],
                "drop_state": t.drop_state if t.drop_state is not None else [0.0, 0.0, 0.0],
                "close_hold_s": t.close_hold_s,
                "yaw_rate": t.yaw_rate,
                "hits": t.hits,
                "misses": t.misses,
                "age": t.age,
                "conf": t.conf,
            }
            for t in self.tracks
        ]


def _measure_track(tr: Track, measured_m: float, now: float, ego: float) -> None:
    """Fold one range into the track filter.

    The first sample stays invalid, rate 0, speed 0. A single grab is not
    a road speed. From the second sample the filter's rate is the road-speed
    correction, including a slow 2 m/s close that never moves a long window
    by several metres.
    """
    if tr._kf_n < 1 or tr._kf_t is None:
        tr._kf_r = float(measured_m)
        tr._kf_rd = 0.0
        tr._kf_p00 = MEASURE_SIGMA_M * MEASURE_SIGMA_M
        tr._kf_p01 = 0.0
        tr._kf_p11 = PRIOR_RATE_SIGMA_MPS * PRIOR_RATE_SIGMA_MPS
        tr._kf_n = 1
        tr._kf_t = float(now)
        tr.range_rate_valid = False
        tr.range_rate_mps = 0.0
        tr.range_rate_sigma = PRIOR_RATE_SIGMA_MPS
        tr.speed_mps = 0.0
        return
    dt = max(1e-3, float(now) - float(tr._kf_t))
    predicted = tr._kf_r + dt * tr._kf_rd
    # A settled track that jumps by more than a ±2 m grab can explain is a
    # teleport, not a new closing speed. Keep the rate. The new range is
    # the baseline. A 4 m swap is smaller than that and goes through the
    # filter, so a real close is not mistaken for a pace.
    if tr.range_rate_valid and abs(float(measured_m) - predicted) >= _RATE_SNAP_M:
        tr._kf_r = float(measured_m)
        tr._kf_p00 = max(tr._kf_p00, MEASURE_SIGMA_M * MEASURE_SIGMA_M)
        tr._kf_t = float(now)
        tr._kf_n += 1
        tr.range_rate_mps = float(tr._kf_rd)
        tr.range_rate_sigma = float(math.sqrt(max(0.0, tr._kf_p11)))
        tr.speed_mps = float(max(0.0, min(_SPEED_CAP, ego + tr.range_rate_mps)))
        _bump_close_hold(tr, dt)
        return
    tr._kf_r, tr._kf_rd, tr._kf_p00, tr._kf_p01, tr._kf_p11 = _kalman_update(
        tr._kf_r, tr._kf_rd, tr._kf_p00, tr._kf_p01, tr._kf_p11, measured_m, dt,
    )
    tr._kf_n += 1
    tr._kf_t = float(now)
    sigma = math.sqrt(max(0.0, tr._kf_p11))
    tr.range_rate_sigma = float(sigma)
    tr.range_rate_mps = float(tr._kf_rd)
    # The mean is real enough to brake with. It is a road speed only once
    # the uncertainty is inside the band a ±2 m grab can support.
    tr.range_rate_valid = sigma <= _RATE_VALID_SIGMA
    if tr.range_rate_valid:
        tr.speed_mps = float(max(0.0, min(_SPEED_CAP, ego + tr.range_rate_mps)))
        _bump_close_hold(tr, dt)
    else:
        tr.speed_mps = 0.0
        tr.close_hold_s = 0.0


def _bump_close_hold(tr: Track, dt: float) -> None:
    """Accumulate a close only while the filtered rate stays fast.

    A frame between the arm line and an opening does not wipe the hold.
    An opening does.
    """
    if tr.range_rate_mps <= _CLOSE_ARM_MPS:
        tr.close_hold_s = float(tr.close_hold_s) + dt
    elif tr.range_rate_mps > _CLOSE_OPEN_MPS:
        tr.close_hold_s = 0.0
    else:
        tr.close_hold_s = max(0.0, float(tr.close_hold_s) - dt)


def _trim_samples(samples: list[tuple[float, float]], now: float) -> None:
    keep = [
        (t, y)
        for t, y in samples
        if math.isfinite(t) and math.isfinite(y) and now - t <= _SAMPLE_KEEP_S
    ]
    if len(keep) > _SAMPLE_KEEP:
        keep = keep[-_SAMPLE_KEEP:]
    samples[:] = keep


def _range_drop_m(samples: list[tuple[float, float]] | None) -> float:
    if not samples or len(samples) < 2:
        return 0.0
    return float(samples[0][1] - samples[-1][1])
