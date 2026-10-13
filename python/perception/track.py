"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

from dataclasses import dataclass

from python.perception.detect import Detection

# speed_mps on a track is road speed, not |Δrange/Δt|.
_SPEED_CAP = 40.0
# A clean, even close can confirm in this many grabs. A noisier fit waits
# for ``_RATE_MIN``. Either way the slope has to be a line, not one step.
_RATE_FAST = 5
_RATE_MIN = 6
_RATE_WINDOW = 8
# ±2 m of grab noise sits near 1 m RMSE. A real range line does not.
_RMSE_MAX = 0.55
# Fast confirm: successive range steps match each other, and each step is
# a real close (15 Hz at 18 m/s is 1.2 m; 10 Hz at 9 m/s is 0.9 m).
_FAST_STEP_TOL = 0.20
_FAST_MIN_STEP_M = 0.45
_FAST_RMSE_MAX = 0.15


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


def _range_steps(ys: list[float]) -> list[float]:
    return [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]


def _steps_agree(ys: list[float], *, closing: bool) -> bool:
    """True when every step moves the same way by about the same amount."""
    ds = _range_steps(ys)
    if len(ds) < _RATE_FAST - 1:
        return False
    if closing:
        if any(d >= -_FAST_MIN_STEP_M for d in ds):
            return False
    elif any(d <= _FAST_MIN_STEP_M for d in ds):
        return False
    mean = sum(ds) / len(ds)
    span = _FAST_STEP_TOL * abs(mean)
    return all(abs(d - mean) <= span for d in ds)


def _one_step(ys: list[float]) -> bool:
    """True when one jump, not a steady trend, carries the range change.

    Three samples at +2 m then three at −2 m fit about −10 m/s. The range
    is flat on both sides of that single step.
    """
    ds = _range_steps(ys)
    if len(ds) < 2:
        return False
    net = ys[-1] - ys[0]
    if abs(net) < 0.5:
        return False
    biggest = max(ds, key=abs)
    others = abs(net) - abs(biggest)
    return abs(biggest) > 0.65 * abs(net) and others < 0.5 * abs(biggest)


def _regular_close(ys: list[float]) -> bool:
    """True when the range steps are a steady close, not a noisy walk.

    A random ±2 m walk can still sit on a straight line for six grabs.
    Those steps are uneven. A real close keeps moving the same way, including
    after the car is already braking and each step gets smaller.
    """
    ds = _range_steps(ys)
    if len(ds) < 4:
        return False
    # One grab jumping back out is noise. A brake curve only shrinks the step.
    if any(d > 0.15 for d in ds):
        return False
    mean = sum(ds) / len(ds)
    if mean >= -0.05:
        return False
    var = sum((d - mean) ** 2 for d in ds) / len(ds)
    return var ** 0.5 <= 0.30


def _trend_agrees(ys: list[float], slope: float) -> bool:
    ds = _range_steps(ys)
    if slope < 0.0:
        good = sum(d < -0.2 for d in ds)
        bad = sum(d > 0.2 for d in ds)
    else:
        good = sum(d > 0.2 for d in ds)
        bad = sum(d < -0.2 for d in ds)
    return good >= 3 and good > bad


def consistent_range_rate(samples: list[tuple[float, float]]) -> tuple[bool, float]:
    """Range-rate once the grabs agree on a line. Otherwise not valid.

    A +2/−2 step over six grabs fits about −10 m/s with no residual check,
    and that slope full-brakes a lead that is not closing. The slope is
    used only when the line is tight, or when every recent step is a real
    close of about the same size. One step does not confirm.
    """
    if len(samples) < _RATE_FAST:
        return False, 0.0
    window = samples[-_RATE_WINDOW:]
    ys = [y for _, y in window]
    # Even steps first, so a hard close is confirmed before six noisy grabs.
    if len(ys) >= _RATE_FAST and _steps_agree(ys[-_RATE_FAST:], closing=True):
        slope, rmse, _se = _fit_range_rate(window[-_RATE_FAST:])
        if slope < -3.0 and rmse <= _FAST_RMSE_MAX:
            return True, float(slope)
    if len(window) < _RATE_MIN:
        return False, 0.0
    slope, rmse, se = _fit_range_rate(window)
    if rmse > _RMSE_MAX or _one_step(ys):
        return False, 0.0
    if slope <= -1.0:
        if se > 1e-6 and abs(slope) < 3.0 * se:
            return False, 0.0
        if not _regular_close(ys):
            return False, 0.0
    elif slope >= 1.0:
        if se > 1e-6 and abs(slope) < 3.0 * se:
            return False, 0.0
        if not _trend_agrees(ys, slope):
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
