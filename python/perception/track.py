"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from python.perception.detect import Detection

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
_SAMPLE_KEEP = 16


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
    # (time, measured forward range). Kept so a dropped window can still
    # report how far the range moved. The rate itself is the Kalman state.
    samples: list[tuple[float, float]] | None = None
    # Filter state. None until the first measurement.
    _kf_r: float = 0.0
    _kf_rd: float = 0.0
    _kf_p00: float = 0.0
    _kf_p01: float = 0.0
    _kf_p11: float = 0.0
    _kf_n: int = 0
    _kf_t: float | None = None


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
            tr.samples.append((float(now), float(d.y)))
            if len(tr.samples) > _SAMPLE_KEEP:
                tr.samples = tr.samples[-_SAMPLE_KEEP:]
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
    else:
        tr.speed_mps = 0.0


def _range_drop_m(samples: list[tuple[float, float]] | None) -> float:
    if not samples or len(samples) < 2:
        return 0.0
    return float(samples[0][1] - samples[-1][1])
