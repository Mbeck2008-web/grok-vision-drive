"""Per-segment random clip length and traffic count for the collector.

``--segment-min-range`` and ``--traffic-range`` are off unless passed.
When both are off, scheduling and ``traffic_count`` stay exactly as they
were: ``ClipSchedule`` forwards to ``segment_at`` / ``segment_boundary``,
and ``resolve_traffic`` is ``traffic_count``.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

from python.collect.spawn_mix import traffic_count


def minutes_to_segment_s(minutes: float) -> float:
    """Seconds for a clip. Same 60 s floor as ``--segment-min``."""
    return max(60.0, float(minutes) * 60.0)


def parse_minute_range(values: Sequence[Any] | None) -> tuple[float, float] | None:
    """``(lo, hi)`` minutes, or ``None`` when the flag was not passed."""
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("segment-min-range needs LO HI")
    try:
        lo = float(values[0])
        hi = float(values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("segment-min-range bounds must be finite") from exc
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("segment-min-range bounds must be finite")
    if lo < 0.0:
        raise ValueError("segment-min-range LO is below 0")
    if lo > hi:
        raise ValueError("segment-min-range LO is above max")
    return lo, hi


def parse_int_range(values: Sequence[Any] | None) -> tuple[int, int] | None:
    """``(lo, hi)`` inclusive counts, or ``None`` when the flag was not passed."""
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("traffic-range needs LO HI")
    try:
        lo = int(values[0])
        hi = int(values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("traffic-range bounds must be finite") from exc
    if lo < 0:
        raise ValueError("traffic-range LO is below 0")
    if lo > hi:
        raise ValueError("traffic-range LO is above max")
    return lo, hi


def draw_segment_minutes(rng: random.Random, lo: float, hi: float) -> float:
    """Uniform clip length in ``[lo, hi]`` minutes."""
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        return lo_f
    return float(rng.uniform(lo_f, hi_f))


def draw_traffic_count(rng: random.Random, lo: int, hi: int) -> int:
    """Uniform inclusive integer in ``[lo, hi]``."""
    return int(rng.randint(int(lo), int(hi)))


def resolve_traffic(
    requested: int | None,
    map_name: str,
    traffic_range: tuple[int, int] | None,
    rng: random.Random,
) -> int:
    """Drawn count when a range is set; otherwise ``traffic_count``."""
    if traffic_range is None:
        return traffic_count(requested, map_name)
    return draw_traffic_count(rng, traffic_range[0], traffic_range[1])


def clip_row_tags(segment_min: float | None, traffic_n: int | None) -> dict[str, Any]:
    """State-row keys added only when the matching range flag is in use."""
    tags: dict[str, Any] = {}
    if segment_min is not None:
        tags["segment_min"] = round(float(segment_min), 4)
    if traffic_n is not None:
        tags["traffic_count"] = int(traffic_n)
    return tags


def validate_clip_flags(args: Any) -> tuple[tuple[float, float] | None, tuple[int, int] | None, int]:
    """Parse range flags. Raises ``ValueError`` on a bad range or both traffic flags."""
    minute_range = parse_minute_range(getattr(args, "segment_min_range", None))
    traffic_range = parse_int_range(getattr(args, "traffic_range", None))
    seed = int(getattr(args, "seed", 0) or 0)
    if traffic_range is not None and getattr(args, "traffic", None) is not None:
        raise ValueError("pass --traffic-range or --traffic, not both")
    return minute_range, traffic_range, seed


def _ranged_weighted_pieces(
    maps: tuple[str, ...],
    lo_s: float,
    hi_s: float,
    weights: dict[str, float],
    total_s: float | None,
    rng: random.Random,
) -> list[tuple[str, float]]:
    """Like ``weighted_pieces``, but each chunk is drawn in ``[lo_s, hi_s]``."""
    from scripts.collect_scene_lanes import share_maps

    shares = share_maps(maps, weights)
    if total_s is not None and math.isfinite(float(total_s)) and float(total_s) > 0.0:
        cycle = float(total_s)
    else:
        positive = [share for _name, share in shares if share > 1e-9]
        small = min(positive) if positive else 1.0
        mid = 0.5 * (float(lo_s) + float(hi_s))
        cycle = max(1.0, mid) / small
    budgets = [(name, max(0.0, share) * cycle) for name, share in shares]
    budgets = [(name, dur) for name, dur in budgets if dur > 0.05]
    if not budgets:
        return [(maps[0] if maps else "", max(1.0, float(lo_s)))]

    def _chunks(name: str, budget: float) -> list[tuple[str, float]]:
        pieces: list[tuple[str, float]] = []
        left = float(budget)
        while left > 0.05 and len(pieces) < 10000:
            take = float(rng.uniform(lo_s, hi_s)) if hi_s > lo_s else float(lo_s)
            take = min(take, left)
            if left - take <= 0.05:
                take = left
            pieces.append((name, take))
            left -= take
        return pieces

    heavy_at = max(range(len(budgets)), key=lambda i: (budgets[i][1], -i))
    heavy = _chunks(budgets[heavy_at][0], budgets[heavy_at][1])
    others: list[tuple[str, float]] = []
    for index, (name, dur) in enumerate(budgets):
        if index == heavy_at:
            continue
        others.extend(_chunks(name, dur))
    pieces: list[tuple[str, float]] = []
    heavy_i = 0
    other_i = 0
    while heavy_i < len(heavy) or other_i < len(others):
        if heavy_i < len(heavy):
            pieces.append(heavy[heavy_i])
            heavy_i += 1
        if other_i < len(others):
            pieces.append(others[other_i])
            other_i += 1
    return pieces


class ClipSchedule:
    """Fixed ``segment_s`` chunks, or a seeded random length per clip.

    When ``minute_range`` is ``None`` this is a thin wrapper around
    ``segment_at`` / ``segment_boundary``. When a range is set, each clip
    is drawn uniformly in that minute window (then floored at 60 s).
    """

    def __init__(
        self,
        maps: tuple[str, ...],
        segment_s: float,
        weights: dict[str, float] | None,
        total_s: float | None,
        *,
        minute_range: tuple[float, float] | Sequence[float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.maps = tuple(maps or ())
        self.segment_s = max(1.0, float(segment_s))
        self.weights = dict(weights) if weights else None
        self.total_s = total_s
        self.minute_range = parse_minute_range(minute_range) if minute_range is not None else None
        self.rng = rng if rng is not None else random.Random(0)
        self._pieces: list[tuple[str, float]] = []
        if self.minute_range is not None:
            lo, hi = self.minute_range
            self._lo_s = minutes_to_segment_s(lo)
            self._hi_s = minutes_to_segment_s(hi)

    def at(self, elapsed_s: float, maps: tuple[str, ...] | None = None) -> tuple[str, int]:
        use = self.maps if maps is None else tuple(maps or ())
        if self.minute_range is None:
            from scripts.collect_scene_lanes import segment_at

            return segment_at(elapsed_s, use, self.segment_s, self.weights, self.total_s)
        name, index, _boundary = self._lookup(elapsed_s)
        return name, index

    def boundary(self, elapsed_s: float, maps: tuple[str, ...] | None = None) -> float | None:
        use = self.maps if maps is None else tuple(maps or ())
        if self.minute_range is None:
            from scripts.collect_scene_lanes import segment_boundary

            return segment_boundary(elapsed_s, use, self.segment_s, self.weights, self.total_s)
        _name, _index, boundary = self._lookup(elapsed_s)
        return boundary

    def piece(self, index: int) -> tuple[str, int]:
        if self.minute_range is None:
            from scripts.collect_scene_lanes import segment_piece

            return segment_piece(index, self.maps, self.segment_s, self.weights, self.total_s)
        i = max(0, int(index))
        self._ensure_index(i)
        return self._pieces[i][0], i

    def duration_min(self, index: int) -> float:
        """Drawn length of clip ``index``, in minutes."""
        if self.minute_range is None:
            return self.segment_s / 60.0
        i = max(0, int(index))
        self._ensure_index(i)
        return self._pieces[i][1] / 60.0

    def _draw_s(self) -> float:
        lo, hi = self._lo_s, self._hi_s
        return float(self.rng.uniform(lo, hi)) if hi > lo else float(lo)

    def _append_unweighted(self) -> None:
        maps = self.maps or ("",)
        name = maps[len(self._pieces) % len(maps)]
        self._pieces.append((name, self._draw_s()))

    def _fill_weighted_cycle(self) -> None:
        more = _ranged_weighted_pieces(
            self.maps,
            self._lo_s,
            self._hi_s,
            self.weights or {},
            self.total_s,
            self.rng,
        )
        if not more:
            more = [("", self._draw_s())]
        self._pieces.extend(more)

    def _ensure_index(self, index: int) -> None:
        self._ensure_count(index + 1)

    def _ensure_count(self, n: int) -> None:
        if self.weights and not self._pieces:
            self._fill_weighted_cycle()
        while len(self._pieces) < n:
            if self.weights:
                # Finite weighted plans are one cycle; past the end, repeat
                # with new draws (until-disk).
                self._fill_weighted_cycle()
                if not self._pieces:
                    break
            else:
                self._append_unweighted()

    def _ensure_elapsed(self, elapsed_s: float) -> None:
        if self.weights and not self._pieces:
            self._fill_weighted_cycle()
        while sum(dur for _name, dur in self._pieces) <= elapsed_s:
            before = len(self._pieces)
            if self.weights:
                planned = self.total_s
                if planned is not None and math.isfinite(float(planned)) and sum(
                    dur for _name, dur in self._pieces
                ) >= float(planned) - 1e-9:
                    break
                self._fill_weighted_cycle()
            else:
                self._append_unweighted()
            if len(self._pieces) == before:
                break

    def _lookup(self, elapsed_s: float) -> tuple[str, int, float]:
        elapsed = max(0.0, float(elapsed_s))
        self._ensure_elapsed(elapsed)
        if not self._pieces:
            self._ensure_count(1)
        acc = 0.0
        for i, (name, dur) in enumerate(self._pieces):
            next_acc = acc + dur
            if elapsed < next_acc:
                return name, i, next_acc
            acc = next_acc
        name, dur = self._pieces[-1]
        return name, len(self._pieces) - 1, acc
