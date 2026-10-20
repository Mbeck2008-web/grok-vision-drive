"""Lane targets from the painted line decals the camera actually sees.

The old label (gt_geom.follow_label) projects the car onto
``get_road_network(drivable_only=True)`` edges. On west_coast_usa 1294 of
those 1467 roads are ``road_invisible`` AI splines whose width is the nav
corridor, and whose lanesLeft/lanesRight default to 1/1. Splitting that
width into equal lanes gave 1.75-7.5 m lanes that do not sit on the paint.

Here the lane is bracketed by painted line decals (``line_white``,
``line_yellow``, ``line_dashed_*`` ...), in the vehicle frame (x right, y
forward, origin at the state ``pos`` ref node). The ego lane is the pair of
lines that contains the car at y = 0; each lookahead follows those same two
lines, so curves are modelled and no lookahead picks a different lane. A
``LaneTracker`` keeps that choice from tick to tick (hysteresis). No paint on
one side means the slot is invalid (NaN, mask 0), never a synthetic 3.5 m
guess.

Paint more than ``Z_MAX_M`` above or below the car (a bridge deck, an
underpass) and paint running more than ``HEADING_MAX_DEG`` off the car's
heading (stop lines, crosswalk stripes) is ignored.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOOKAHEADS_M = (5.0, 10.0, 20.0, 30.0)
LANE_MIN_M = 2.4
LANE_MAX_M = 5.0
DOUBLE_LINE_M = 0.6
SEARCH_M = 12.0
# Paint on another deck: a bridge overhead or the road under an overpass.
Z_MAX_M = 3.0
# A lane line runs with the car. Stop lines and crosswalk bars run across it.
HEADING_MAX_DEG = 45.0
_CELL = 25.0


def is_paint(material: Any) -> bool:
    m = str(material or "").lower()
    return m.startswith("line") or m.startswith("m_line")


def _catmull_rom(nodes: list[list[float]], k: int = 4) -> np.ndarray:
    """DecalRoad nodes are spline controls; densify so curves stay curved."""
    p = np.asarray([n[:3] for n in nodes], dtype=np.float64)
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        for t in np.linspace(0.0, 1.0, k, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return np.asarray(out)


def decals_from_level_zip(zip_path: Path | str) -> list[dict[str, Any]]:
    """Every DecalRoad in a level zip: material and nodes. Read-only."""
    out: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".level.json"):
                continue
            for line in z.read(name).decode("utf-8", "replace").splitlines():
                if '"DecalRoad"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                out.append({"material": obj.get("material"), "nodes": obj.get("nodes"),
                            "persistentId": obj.get("persistentId")})
    return out


def load_painted_decals(map_name: str, roads_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """``<roads_dir>/<map>_lines.json.gz`` first, else ``$BNG_HOME/content/levels/<map>.zip``."""
    if roads_dir is not None:
        cached = Path(roads_dir) / f"{map_name}_lines.json.gz"
        if cached.is_file():
            with gzip.open(cached, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")
    if home:
        zp = Path(home) / "content" / "levels" / f"{map_name}.zip"
        if zp.is_file():
            decals = [d for d in decals_from_level_zip(zp) if is_paint(d.get("material"))]
            if roads_dir is not None:
                Path(roads_dir).mkdir(parents=True, exist_ok=True)
                with gzip.open(Path(roads_dir) / f"{map_name}_lines.json.gz", "wt", encoding="utf-8") as fh:
                    json.dump(decals, fh)
            return decals
    return []


class PaintedLanes:
    """Grid of painted segments for one map."""

    def __init__(self, decals: Iterable[dict[str, Any]]) -> None:
        segs = []
        for d in decals:
            nodes = d.get("nodes")
            if not is_paint(d.get("material")) or not nodes or len(nodes) < 2:
                continue
            p = _catmull_rom(nodes)
            segs.append(np.hstack([p[:-1], p[1:]]))
        self.segs = np.vstack(segs) if segs else np.zeros((0, 6))
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        mid = (self.segs[:, :2] + self.segs[:, 3:5]) * 0.5
        for j, (x, y) in enumerate(mid):
            grid[(int(x // _CELL), int(y // _CELL))].append(j)
        self.grid = {k: np.asarray(v) for k, v in grid.items()}

    def __bool__(self) -> bool:
        return len(self.segs) > 0

    def _frame(
        self,
        pos: Any,
        direction: Any,
        z_max_m: float | None,
        heading_max_deg: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Nearby paint segments in the vehicle frame: ``(p0, f, r, ax, ay, bx, by)``."""
        f = np.asarray([float(direction[0]), float(direction[1])])
        norm = float(np.hypot(*f))
        if norm < 1e-6:
            return None
        f /= norm
        r = np.asarray([f[1], -f[0]])
        cx, cy = int(float(pos[0]) // _CELL), int(float(pos[1]) // _CELL)
        idx = [self.grid[(cx + a, cy + b)] for a in range(-2, 3) for b in range(-2, 3) if (cx + a, cy + b) in self.grid]
        if not idx:
            return None
        s = self.segs[np.concatenate(idx)]
        keep = np.ones(len(s), dtype=bool)
        if z_max_m is not None and len(pos) >= 3 and pos[2] is not None:
            zmid = 0.5 * (s[:, 2] + s[:, 5])
            keep &= np.abs(zmid - float(pos[2])) <= float(z_max_m)
        if heading_max_deg is not None:
            d = s[:, 3:5] - s[:, :2]
            dl = np.hypot(d[:, 0], d[:, 1])
            cosang = np.abs(d @ f) / np.maximum(dl, 1e-9)
            keep &= (dl > 1e-6) & (cosang >= math.cos(math.radians(float(heading_max_deg))))
        s = s[keep]
        if not len(s):
            return None
        p0 = np.asarray([float(pos[0]), float(pos[1])])
        a, b = s[:, :2] - p0, s[:, 3:5] - p0
        ax, ay, bx, by = a @ r, a @ f, b @ r, b @ f
        return p0, f, r, ax, ay, bx, by

    @staticmethod
    def _lines_at(
        y: float, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray, near: float, reach: float,
    ) -> list[tuple[float, float]]:
        """Painted lines crossing ``y`` within ``reach`` of ``near``: ``(x, dx/dy)``, doubles merged."""
        m = ((ay - y) * (by - y) <= 0.0) & (np.abs(by - ay) > 1e-6)
        slope = (bx[m] - ax[m]) / (by[m] - ay[m])
        hits = ax[m] + (y - ay[m]) * slope
        sel = np.abs(hits - near) < reach
        order = np.argsort(hits[sel])
        xs, ss = hits[sel][order], slope[sel][order]
        merged: list[list[float]] = []  # x, slope, count
        for v, sl in zip(xs, ss):
            if merged and v - merged[-1][0] / merged[-1][2] <= DOUBLE_LINE_M:
                merged[-1][0] += float(v)
                merged[-1][1] += float(sl)
                merged[-1][2] += 1.0
            else:
                merged.append([float(v), float(sl), 1.0])
        return [(x / n, sl / n) for x, sl, n in merged]

    def lane_at(
        self,
        pos: Any,
        direction: Any,
        ys: tuple[float, ...] = LOOKAHEADS_M,
        *,
        z_max_m: float | None = Z_MAX_M,
        heading_max_deg: float | None = HEADING_MAX_DEG,
        tracker: LaneTracker | None = None,
    ) -> list[tuple[float, float] | None]:
        """``(left_x, right_x)`` of the painted ego lane at each ``y``, vehicle frame.

        The ego lane is the pair of adjacent painted lines that contains the
        car at y = 0 (the car's own position), ``LANE_MIN_M``-``LANE_MAX_M``
        wide. Each lookahead then follows those same two lines: each line is
        found where its own position and slope at the previous crossing put it
        (within ``FOLLOW_M``), so no lookahead re-picks a different pair. A
        lookahead where either line is missing, or the two are out of the
        width range, is None.

        ``tracker`` (a ``LaneTracker``, one per drive; reset it on a respawn or
        a new segment) keeps the previous tick's pair: it stays the ego lane
        until the car centre is more than ``SWITCH_MARGIN_M`` past one of its
        lines, or the pair is gone (see ``LaneTracker``). Without a tracker
        each call is stateless.

        ``z_max_m`` / ``heading_max_deg`` of None disable that filter
        (diagnostics only). ``pos`` without a z skips the z filter.
        """
        frame = self._frame(pos, direction, z_max_m, heading_max_deg)
        if frame is None:
            if tracker is not None:
                tracker.reset()
            return [None] * len(ys)
        p0, f, r, ax, ay, bx, by = frame
        lines0 = self._lines_at(0.0, ax, ay, bx, by, 0.0, SEARCH_M)
        anchor = _containing_pair(lines0)
        if tracker is not None:
            anchor = tracker._choose(anchor, lines0, p0, f, r)
        if anchor is None:
            return [None] * len(ys)
        out: list[tuple[float, float] | None] = []
        state = [_LineFollow(anchor[0]), _LineFollow(anchor[1])]
        for y in ys:
            y = float(y)
            got: list[float | None] = []
            for side in state:
                guess = side.predict(y)
                lines = self._lines_at(y, ax, ay, bx, by, guess, FOLLOW_M)
                if not lines:
                    got.append(None)
                    continue
                x, sl = min(lines, key=lambda v: abs(v[0] - guess))
                side.update(y, x, sl)
                got.append(x)
            left, right = got
            if left is None or right is None or not (LANE_MIN_M <= right - left <= LANE_MAX_M):
                out.append(None)
            else:
                out.append((left, right))
        return out


# Hysteresis margin for the ego-lane choice, fixed from the car before any
# replay census: half the ego half width, 0.5 * 1.01 m (2.02 m car) = 0.505 m.
# The pair the car was in stays the ego lane until the car centre (the state
# pos ref node, the body's lateral centre) is more than this past one of its
# lines. At that point three quarters of the body (1.515 of 2.02 m) is over
# the line. Short of it the car straddles the line, and a car straddling a
# line has not changed lanes: it keeps the lane it came from.
SWITCH_MARGIN_M = 0.505
# A line of the previous pair matches a line seen this tick when it is this
# close to where the previous tick's crossing predicts it (one tick is a few
# metres of travel). A quarter of the narrowest lane: a neighbour line, a
# lane width away, can never match.
MATCH_M = 0.6
# Each line is followed from one lookahead to the next within this of its
# predicted position (its last crossing, slope and slope change). Under half
# the narrowest lane, so the neighbour line cannot be taken.
FOLLOW_M = 1.0
# A line of the kept pair that is not seen at y = 0 (a gap between dashes, a
# decal piece ending) is carried from where it was last seen for at most this
# far behind the car: one 40 ft dash cycle (10 ft dash, 30 ft gap). Once both
# lines are unseen, or a line is carried farther, the pair is gone.
CARRY_M = 12.2


def _containing_pair(lines: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The adjacent line pair with left <= 0 <= right, if it is a lane wide."""
    for left, right in zip(lines, lines[1:]):
        if left[0] <= 0.0 <= right[0]:
            if LANE_MIN_M <= right[0] - left[0] <= LANE_MAX_M:
                return left, right
            return None
    return None


class _LineFollow:
    """One painted line walked forward from y = 0: last crossing, slope, slope change."""

    def __init__(self, start: tuple[float, float]) -> None:
        self.y, self.x, self.slope = 0.0, float(start[0]), float(start[1])
        self.curv = 0.0

    def predict(self, y: float) -> float:
        dy = float(y) - self.y
        return self.x + self.slope * dy + 0.5 * self.curv * dy * dy

    def update(self, y: float, x: float, slope: float) -> None:
        dy = float(y) - self.y
        if dy > 1e-6:
            self.curv = (float(slope) - self.slope) / dy
        self.y, self.x, self.slope = float(y), float(x), float(slope)


class LaneTracker:
    """The previous tick's ego-lane pair for ``PaintedLanes.lane_at``. One per drive.

    Each line is kept as the world point where it crossed y = 0 when it was
    last seen, and its world direction. Next tick it is projected onto the
    car's new y = 0 and matched (``MATCH_M``) to a line seen there.

    The kept pair is replaced by the pair that contains the car (or by none)
    only when the car centre is more than ``SWITCH_MARGIN_M`` past one of its
    lines, or the pair is gone: both lines unseen, a line carried more than
    ``CARRY_M`` behind the car, or the two out of the lane width range.
    ``reset()`` forgets it (a respawn, a new segment, a new map).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.left: tuple[float, float, float, float] | None = None
        self.right: tuple[float, float, float, float] | None = None

    def _project(
        self, line: tuple[float, float, float, float], p0: np.ndarray, f: np.ndarray, r: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """``(x, slope, behind_m)`` of a stored line on the car's y = 0 now."""
        qx, qy, dx, dy = line
        q = np.asarray([qx, qy]) - p0
        d = np.asarray([dx, dy])
        df = float(d @ f)
        if abs(df) < 1e-3:
            return None
        t = -float(q @ f) / df
        x = float((q + t * d) @ r)
        return x, float(d @ r) / df, -float(q @ f)

    def _choose(
        self,
        containing: tuple[tuple[float, float], tuple[float, float]] | None,
        lines0: list[tuple[float, float]],
        p0: np.ndarray,
        f: np.ndarray,
        r: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        kept = None
        if self.left is not None and self.right is not None:
            sides = []
            seen = 0
            for line in (self.left, self.right):
                got = self._project(line, p0, f, r)
                if got is None:
                    sides = []
                    break
                x, slope, behind = got
                match = min(lines0, key=lambda v: abs(v[0] - x), default=None)
                if match is not None and abs(match[0] - x) <= MATCH_M:
                    sides.append((match, True))
                    seen += 1
                elif behind <= CARRY_M:
                    sides.append(((x, slope), False))
                else:
                    sides = []
                    break
            if len(sides) == 2 and seen >= 1:
                (left, left_seen), (right, right_seen) = sides
                if LANE_MIN_M <= right[0] - left[0] <= LANE_MAX_M:
                    if left[0] <= SWITCH_MARGIN_M and right[0] >= -SWITCH_MARGIN_M:
                        kept = (left, left_seen), (right, right_seen)
        if kept is not None:
            (left, left_seen), (right, right_seen) = kept
            self._store(left, left_seen, right, right_seen, p0, f, r)
            return left, right
        if containing is None:
            self.left = self.right = None
            return None
        self._store(containing[0], True, containing[1], True, p0, f, r)
        return containing

    def _store(self, left, left_seen, right, right_seen, p0, f, r) -> None:
        def world(side: tuple[float, float]) -> tuple[float, float, float, float]:
            x, slope = side
            q = p0 + x * r
            d = f + slope * r
            d = d / float(np.hypot(*d))
            return float(q[0]), float(q[1]), float(d[0]), float(d[1])

        # A carried line keeps the point where it was last seen.
        self.left = world(left) if left_seen or self.left is None else self.left
        self.right = world(right) if right_seen or self.right is None else self.right


def heading_from_centers(c5: float | None, c10: float | None) -> float | None:
    """Same sign as ``gt_geom.ego_center_x``: positive when the car points left of the lane."""
    if c5 is None or c10 is None:
        return None
    return float(math.asin(max(-1.0, min(1.0, (float(c10) - float(c5)) / 5.0))))
