"""Per-segment random clip length and traffic count for the collector.

``--segment-min-range`` and ``--traffic-range`` are off unless passed.
When both are off, scheduling and ``traffic_count`` stay exactly as they
were: ``ClipSchedule`` forwards to ``segment_at`` / ``segment_boundary``,
and ``resolve_traffic`` is ``traffic_count``.

This module does not import ``scripts.collect_scene_lanes``.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

from python.collect.spawn_mix import traffic_count

# Inclusive ceiling for ``--traffic-range HI``.
TRAFFIC_RANGE_MAX = 32
# ``--segment-min-range LO`` must be at least one minute (before the 60 s floor).
SEGMENT_MIN_RANGE_FLOOR = 1.0


def minutes_to_segment_s(minutes: float) -> float:
    """Seconds for a clip. Same 60 s floor as ``--segment-min``."""
    return max(60.0, float(minutes) * 60.0)


def clip_rng(seed: int, index: int) -> random.Random:
    """Per-clip RNG for ``(seed, index)``.

    CPython 3.12 ``Random.seed`` rejects tuples, so the pair is encoded as
    ``"{seed}:{index}"``. Same pair, same stream.
    """
    return random.Random(f"{int(seed)}:{int(index)}")


def clip_seed_tag(seed: int, index: int) -> str:
    """Logged per-clip seed, matching ``clip_rng``."""
    return f"{int(seed)}:{int(index)}"


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
    if lo < SEGMENT_MIN_RANGE_FLOOR:
        raise ValueError("segment-min-range LO is below 1")
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
    if hi > TRAFFIC_RANGE_MAX:
        raise ValueError(f"traffic-range HI is above {TRAFFIC_RANGE_MAX}")
    return lo, hi


def maps_allow_traffic_range(maps: Sequence[str] | str | None) -> bool:
    """True when the run includes west_coast_usa (the only spawn-deck map)."""
    if maps is None:
        names: tuple[str, ...] = ()
    elif isinstance(maps, str):
        names = tuple(part.strip() for part in maps.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in maps if str(part).strip())
    return "west_coast_usa" in names


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
    if traffic_range is not None and not maps_allow_traffic_range(getattr(args, "maps", "")):
        raise ValueError(
            "traffic-range requires west_coast_usa in --maps "
            "(companion spawn uses the west-coast deck; other maps have no deck)"
        )
    return minute_range, traffic_range, seed


def share_maps(maps: tuple[str, ...], weights: dict[str, float]) -> list[tuple[str, float]]:
    """Time fraction for each map, in ``maps`` order, summing to 1.

    A listed weight is that map's share. Maps without a weight split whatever
    is left. Weights that already sum past 1, with nothing left to split, are
    scaled down.
    """
    explicit = {name: float(weights[name]) for name in maps if name in weights}
    missing = [name for name in maps if name not in weights]
    used = sum(explicit.values())
    if missing:
        left = max(0.0, 1.0 - used)
        each = left / len(missing)
        shares = [(name, explicit.get(name, each)) for name in maps]
        total = sum(share for _name, share in shares)
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            shares = [(name, share / total) for name, share in shares]
        return shares
    scale = used if used > 0.0 else 1.0
    return [(name, explicit.get(name, 0.0) / scale) for name in maps]


def weighted_pieces(
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float],
    total_s: float | None,
) -> list[tuple[str, float]]:
    """Chunks of at most ``segment_s`` that sum to each map's share.

    The heaviest map alternates with the others, so a 60% west-coast share
    is several short visits rather than one block at the front. A map whose
    share is already shorter than ``segment_s`` is a single chunk.
    """
    shares = share_maps(maps, weights)
    if total_s is not None and math.isfinite(float(total_s)) and float(total_s) > 0.0:
        cycle = float(total_s)
    else:
        positive = [share for _name, share in shares if share > 1e-9]
        small = min(positive) if positive else 1.0
        cycle = max(1.0, float(segment_s)) / small
    span = max(1.0, float(segment_s))
    budgets = [(name, max(0.0, share) * cycle) for name, share in shares]
    budgets = [(name, dur) for name, dur in budgets if dur > 0.05]
    if not budgets:
        return [(maps[0] if maps else "", span)]

    def _chunks(name: str, budget: float) -> list[tuple[str, float]]:
        pieces: list[tuple[str, float]] = []
        left = float(budget)
        while left > 0.05 and len(pieces) < 10000:
            take = min(span, left)
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


def segment_at(
    elapsed_s: float,
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None = None,
    total_s: float | None = None,
) -> tuple[str, int]:
    """Map name and segment index. Empty maps stay on the loaded level.

    Without weights, each map gets ``segment_s`` and the list repeats.
    With weights, ``weighted_pieces`` rotates those chunks across the plan.
    The index keeps climbing when the cycle repeats.
    """
    if not maps:
        return "", 0
    elapsed = max(0.0, float(elapsed_s))
    if not weights:
        span = max(1.0, float(segment_s))
        index = int(elapsed // span)
        return maps[index % len(maps)], index
    pieces = weighted_pieces(maps, segment_s, weights, total_s)
    cycle_len = sum(dur for _name, dur in pieces) or 1.0
    which = int(elapsed // cycle_len)
    pos = elapsed - which * cycle_len
    acc = 0.0
    for i, (name, dur) in enumerate(pieces):
        acc += dur
        if pos < acc or i == len(pieces) - 1:
            return name, which * len(pieces) + i
    return pieces[-1][0], which * len(pieces) + len(pieces) - 1


def segment_piece(
    index: int,
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None = None,
    total_s: float | None = None,
) -> tuple[str, int]:
    """Map name of segment ``index`` (the inverse of ``segment_at``'s index)."""
    if not maps:
        return "", 0
    i = max(0, int(index))
    if not weights:
        return maps[i % len(maps)], i
    pieces = weighted_pieces(maps, segment_s, weights, total_s)
    return pieces[i % len(pieces)][0], i


def segment_boundary(
    elapsed_s: float,
    maps: tuple[str, ...],
    segment_s: float,
    weights: dict[str, float] | None = None,
    total_s: float | None = None,
) -> float | None:
    """Schedule time where the current segment index changes. None with no maps."""
    if not maps:
        return None
    elapsed = max(0.0, float(elapsed_s))
    if not weights:
        span = max(1.0, float(segment_s))
        return (int(elapsed // span) + 1) * span
    pieces = weighted_pieces(maps, segment_s, weights, total_s)
    cycle_len = sum(dur for _name, dur in pieces) or 1.0
    which = int(elapsed // cycle_len)
    pos = elapsed - which * cycle_len
    acc = 0.0
    for _name, dur in pieces:
        acc += dur
        if pos < acc:
            return which * cycle_len + acc
    return (which + 1) * cycle_len


def normalize_clip_lengths(
    pieces: list[tuple[str, float]],
    lo_s: float,
    hi_s: float,
    *,
    allow_short_last: bool,
) -> list[tuple[str, float]]:
    """Make every clip satisfy ``LO <= len <= HI``, except a short last clip.

    Short leftovers are folded into the previous clip (or the next, if first).
    A piece longer than HI is split so both parts stay in range when possible.
    """
    lo_s = float(lo_s)
    hi_s = max(float(hi_s), lo_s)
    out = [(str(name), float(dur)) for name, dur in pieces if float(dur) > 0.05]
    if not out:
        return []

    def _merge_shorts() -> None:
        guard = 0
        while guard < 10000:
            guard += 1
            moved = False
            i = 0
            while i < len(out):
                name, dur = out[i]
                is_last = i == len(out) - 1
                if dur + 1e-9 >= lo_s or (allow_short_last and is_last and len(out) > 1):
                    i += 1
                    continue
                if i > 0:
                    prev_n, prev_d = out[i - 1]
                    out[i - 1] = (prev_n, prev_d + dur)
                    del out[i]
                    moved = True
                    continue
                if i + 1 < len(out):
                    next_n, next_d = out[i + 1]
                    out[i + 1] = (next_n, next_d + dur)
                    del out[i]
                    moved = True
                    continue
                break
            if not moved:
                return

    def _split_long() -> None:
        i = 0
        while i < len(out):
            name, dur = out[i]
            if dur <= hi_s + 1e-9:
                i += 1
                continue
            rest = dur - hi_s
            is_tail = i == len(out) - 1 and not out[i + 1:] if False else i == len(out) - 1
            # Prefer (hi, rest) when rest is a valid clip or a permitted short last.
            if rest + 1e-9 >= lo_s or (allow_short_last and is_tail and not out[i + 1 :]):
                out[i] = (name, hi_s)
                out.insert(i + 1, (name, rest))
                i += 1
                continue
            # Otherwise leave at least LO on the tail: (dur-LO, LO), if that head is in range.
            head = dur - lo_s
            if head + 1e-9 >= lo_s and head <= hi_s + 1e-9:
                out[i] = (name, head)
                out.insert(i + 1, (name, lo_s))
                i += 2
                continue
            # Head would still exceed HI: peel HI and keep splitting.
            out[i] = (name, hi_s)
            out.insert(i + 1, (name, rest))
            i += 1

    for _ in range(32):
        before = list(out)
        _merge_shorts()
        _split_long()
        if out == before:
            break
    return out


def _ranged_weighted_pieces(
    maps: tuple[str, ...],
    lo_s: float,
    hi_s: float,
    weights: dict[str, float],
    total_s: float | None,
    seed: int,
    start_index: int = 0,
) -> list[tuple[str, float]]:
    """Like ``weighted_pieces``, but each chunk is drawn in ``[lo_s, hi_s]``."""
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
        take = float(clip_rng(seed, start_index).uniform(lo_s, hi_s)) if hi_s > lo_s else float(lo_s)
        return [(maps[0] if maps else "", take)]

    next_index = [int(start_index)]

    def _draw() -> float:
        idx = next_index[0]
        next_index[0] = idx + 1
        rng = clip_rng(seed, idx)
        return float(rng.uniform(lo_s, hi_s)) if hi_s > lo_s else float(lo_s)

    def _chunks(name: str, budget: float) -> list[tuple[str, float]]:
        pieces: list[tuple[str, float]] = []
        left = float(budget)
        while left > 0.05 and len(pieces) < 10000:
            take = _draw()
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
    finite = total_s is not None and math.isfinite(float(total_s)) and float(total_s) > 0.0
    return normalize_clip_lengths(pieces, lo_s, hi_s, allow_short_last=bool(finite))


class ClipSchedule:
    """Fixed ``segment_s`` chunks, or a seeded random length per clip.

    When ``minute_range`` is ``None`` this is a thin wrapper around
    ``segment_at`` / ``segment_boundary``. When a range is set, each clip
    is drawn with ``clip_rng(seed, index)`` in that minute window (then
    floored at 60 s). Weighted leftovers are folded so every clip is in
    ``[LO, HI]`` except possibly the last clip of a finite run.
    """

    def __init__(
        self,
        maps: tuple[str, ...],
        segment_s: float,
        weights: dict[str, float] | None,
        total_s: float | None,
        *,
        minute_range: tuple[float, float] | Sequence[float] | None = None,
        seed: int = 0,
        rng: random.Random | None = None,
    ) -> None:
        del rng  # per-clip ``clip_rng(seed, index)`` is the only stream
        self.maps = tuple(maps or ())
        self.segment_s = max(1.0, float(segment_s))
        self.weights = dict(weights) if weights else None
        self.total_s = total_s
        self.minute_range = parse_minute_range(minute_range) if minute_range is not None else None
        self.seed = int(seed)
        self._pieces: list[tuple[str, float]] = []
        if self.minute_range is not None:
            lo, hi = self.minute_range
            self._lo_s = minutes_to_segment_s(lo)
            self._hi_s = minutes_to_segment_s(hi)

    def at(self, elapsed_s: float, maps: tuple[str, ...] | None = None) -> tuple[str, int]:
        use = self.maps if maps is None else tuple(maps or ())
        if self.minute_range is None:
            return segment_at(elapsed_s, use, self.segment_s, self.weights, self.total_s)
        name, index, _boundary = self._lookup(elapsed_s)
        return name, index

    def boundary(self, elapsed_s: float, maps: tuple[str, ...] | None = None) -> float | None:
        use = self.maps if maps is None else tuple(maps or ())
        if self.minute_range is None:
            return segment_boundary(elapsed_s, use, self.segment_s, self.weights, self.total_s)
        _name, _index, boundary = self._lookup(elapsed_s)
        return boundary

    def piece(self, index: int) -> tuple[str, int]:
        if self.minute_range is None:
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

    def _draw_s(self, index: int) -> float:
        lo, hi = self._lo_s, self._hi_s
        rng = clip_rng(self.seed, index)
        return float(rng.uniform(lo, hi)) if hi > lo else float(lo)

    def _append_unweighted(self) -> None:
        maps = self.maps or ("",)
        index = len(self._pieces)
        name = maps[index % len(maps)]
        self._pieces.append((name, self._draw_s(index)))

    def _fill_weighted_cycle(self) -> None:
        more = _ranged_weighted_pieces(
            self.maps,
            self._lo_s,
            self._hi_s,
            self.weights or {},
            self.total_s,
            self.seed,
            start_index=len(self._pieces),
        )
        if not more:
            more = [("", self._draw_s(len(self._pieces)))]
        self._pieces.extend(more)

    def _ensure_index(self, index: int) -> None:
        self._ensure_count(index + 1)

    def _ensure_count(self, n: int) -> None:
        if self.weights and not self._pieces:
            self._fill_weighted_cycle()
        while len(self._pieces) < n:
            if self.weights:
                planned = self.total_s
                if planned is not None and math.isfinite(float(planned)):
                    break
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
