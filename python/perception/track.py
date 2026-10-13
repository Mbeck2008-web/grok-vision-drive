"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

from dataclasses import dataclass

from python.perception.detect import Detection

# speed_mps on a track is road speed, not |Δrange/Δt|.
_SPEED_CAP = 40.0
# Least-squares range-rate over this many grabs. Fewer than this is not
# a measurement: speed 0 would look like a stopped lead to the AEB gate.
_RATE_MIN = 6
_RATE_WINDOW = 8


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


def _fit_range_rate(samples: list[tuple[float, float]]) -> float:
    """Signed range-rate (m/s) from a least-squares line through (t, range)."""
    n = len(samples)
    t0 = samples[0][0]
    ts = [s[0] - t0 for s in samples]
    ys = [s[1] for s in samples]
    mean_t = sum(ts) / n
    mean_y = sum(ys) / n
    var_t = sum((t - mean_t) ** 2 for t in ts)
    if var_t < 1e-8:
        return 0.0
    return sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys)) / var_t


def consistent_range_rate(samples: list[tuple[float, float]]) -> tuple[bool, float]:
    """Range-rate once several grabs agree on a line. Otherwise not valid.

    One ±2 m grab at 10 Hz is about ±20 m/s. A single smoothed step of that
    still looks like a hard close. The line through the window is the rate,
    and it is withheld until ``_RATE_MIN`` samples exist.
    """
    if len(samples) < _RATE_MIN:
        return False, 0.0
    return True, float(_fit_range_rate(samples[-_RATE_WINDOW:]))


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
    # (time, measured forward range). The AEB gate ignores the track until
    # this window can support a range-rate.
    samples: list[tuple[float, float]] | None = None


class IoUTracker:
    def __init__(self, max_misses: int = 5, gate_m: float = 4.0) -> None:
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
