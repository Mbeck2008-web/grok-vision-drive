"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

from dataclasses import dataclass

from python.perception.detect import Detection

# speed_mps on a track is road speed, not |Δrange/Δt|.
_SPEED_CAP = 40.0
# Least squares over about a second. A 5-grab ramp inside ±2 m is the same
# shape as a real close, so a short fit never becomes a range-rate.
_RATE_MIN_N = 8
_RATE_MIN_SPAN_S = 0.7
_RATE_WINDOW = 16
# ±2 m of range noise can walk 18 m down to 14 m. A closing rate has to
# move farther than that band, and the slope has to beat its own stderr.
_CLOSE_NET_M = 4.5
_SLOPE_SIGMA = 3.0
# A same-lane lead can jump 4 m between grabs when each one is ±2 m off.
# The old 4 m gate missed that jump, left the first track coasting, and
# the coast kept the brake it had already armed.
_MATCH_GATE_M = 6.0


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


def _fit_range_rate(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Slope (m/s), RMSE (m), and slope stderr from a line through (t, range)."""
    n = len(samples)
    t0 = samples[0][0]
    ts = [s[0] - t0 for s in samples]
    ys = [s[1] for s in samples]
    mean_t = sum(ts) / n
    mean_y = sum(ys) / n
    var_t = sum((t - mean_t) ** 2 for t in ts)
    if var_t < 1e-8:
        return 0.0, 0.0, 0.0
    slope = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys)) / var_t
    sse = 0.0
    for t, y in zip(ts, ys):
        err = y - (mean_y + slope * (t - mean_t))
        sse += err * err
    rmse = (sse / n) ** 0.5
    dof = max(1, n - 2)
    se = ((sse / dof) / var_t) ** 0.5
    return float(slope), float(rmse), float(se)


def consistent_range_rate(samples: list[tuple[float, float]]) -> tuple[bool, float]:
    """Range-rate from a long least-squares line, or not valid yet.

    Five grabs of a 0.5 m ramp fit a clean slope and are still just ±2 m of
    noise on a lead that is not moving. The fit has to cover most of a
    second, the slope has to clear its stderr, and a closing call has to
    move the range by more than that noise band. Full braking does not use
    this flag: a short gap brakes from the range alone.
    """
    if len(samples) < _RATE_MIN_N:
        return False, 0.0
    window = samples[-_RATE_WINDOW:]
    if window[-1][0] - window[0][0] < _RATE_MIN_SPAN_S:
        return False, 0.0
    slope, _rmse, se = _fit_range_rate(window)
    if se > 1e-6 and abs(slope) < _SLOPE_SIGMA * se:
        return False, 0.0
    net = window[-1][1] - window[0][1]
    # A closing slope inside the ±2 m band is the ramp, not a lead we are
    # catching. An opening slope still has to actually increase the range.
    if slope < -1.0 and net > -_CLOSE_NET_M:
        return False, 0.0
    if slope > 1.0 and net < 1.0:
        return False, 0.0
    return True, float(slope)


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
    range_rate_valid: bool = False
    # (time, measured forward range). A closing rate needs a long line.
    # The planner still brakes from the gap while this is short.
    samples: list[tuple[float, float]] | None = None


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
            if len(tr.samples) > _RATE_WINDOW:
                tr.samples = tr.samples[-_RATE_WINDOW:]
            valid, rate = consistent_range_rate(tr.samples)
            tr.range_rate_valid = valid
            if valid:
                tr.range_rate_mps = float(rate)
                tr.speed_mps = float(max(0.0, min(_SPEED_CAP, ego + tr.range_rate_mps)))
            else:
                # No rate yet. Speed 0 would be a stopped lead. Leave it
                # invalid so the stop/AEB gate skips this track.
                tr.range_rate_mps = 0.0
                tr.speed_mps = 0.0
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
                    range_rate_valid=False,
                    samples=[(float(now), float(d.y))],
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
                "range_rate_valid": bool(t.range_rate_valid),
                "yaw_rate": t.yaw_rate,
                "hits": t.hits,
                "misses": t.misses,
                "age": t.age,
                "conf": t.conf,
            }
            for t in self.tracks
        ]
