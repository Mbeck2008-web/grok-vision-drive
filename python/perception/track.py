"""IoU + constant-velocity tracker (BYTETrack-style drop after ~5 misses)."""

from __future__ import annotations

from dataclasses import dataclass, field

from python.perception.detect import Detection


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


class IoUTracker:
    def __init__(self, max_misses: int = 5, gate_m: float = 4.0) -> None:
        self.max_misses = max_misses
        self.gate_m = gate_m
        self._next_id = 1
        self.tracks: list[Track] = []
        self._last_t: float | None = None

    def update(self, dets: list[Detection], now: float) -> list[Track]:
        dt = 0.1 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now
        # predict
        for tr in self.tracks:
            tr.x += 0.0  # lateral hold
            tr.y += tr.speed_mps * dt * 0.15  # weak along-track if speed known
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
            # speed from y delta
            vy = (d.y - tr.y) / dt
            tr.speed_mps = 0.7 * tr.speed_mps + 0.3 * max(0.0, -vy)  # closing positive later in CIPV
            # store absolute along-road speed estimate from forward motion of box
            tr.speed_mps = float(max(0.0, min(40.0, abs(vy))))
            tr.x, tr.y, tr.yaw = d.x, d.y, d.yaw
            tr.conf = d.conf
            tr.hits += 1
            tr.misses = 0

        for i in unmatched:
            d = dets[i]
            self.tracks.append(
                Track(id=self._next_id, cls=d.cls, x=d.x, y=d.y, yaw=d.yaw, conf=d.conf, speed_mps=0.0)
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
                "yaw_rate": t.yaw_rate,
                "hits": t.hits,
                "age": t.age,
                "conf": t.conf,
            }
            for t in self.tracks
        ]
