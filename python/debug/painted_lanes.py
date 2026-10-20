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
# The collector's car (etk800), bbox width as vehicle_half_width logs it.
EGO_WIDTH_M = 2.02
# A pair of lines is a lane when the car fits between them: narrower than
# the car's own width it is not a lane the car can be in (a gore wedge, a
# double line wider than DOUBLE_LINE_M, a taper's last metres). The old
# fixed 2.4 m had no source and dropped real narrow lanes (2.34 m at a lane
# end on west_coast_usa).
LANE_MIN_M = EGO_WIDTH_M
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
        walked forward in ``FOLLOW_STEP_M`` steps, at each step the paint
        nearest where its last crossing and slope put it (within
        ``FOLLOW_M``), across gaps of up to ``CARRY_M``. No lookahead re-picks
        a different pair. A lookahead where either line has no paint, or the
        two are out of the width range, is None.

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
        walk = (_LineWalk(anchor[0]), _LineWalk(anchor[1]))
        width = anchor[1][0] - anchor[0][0]
        y_at = 0.0
        for y in ys:
            y = float(y)
            # Step both lines forward to y; each step sees the paint once for both.
            while y_at < y - 1e-9:
                y_at = min(y, y_at + FOLLOW_STEP_M)
                lines = self._lines_at(y_at, ax, ay, bx, by, 0.0, math.inf)
                width = _step_pair(walk, y_at, lines, width)
            left, right = walk[0].here(y), walk[1].here(y)
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
# metres of travel). Well under half the narrowest lane (LANE_MIN_M / 2):
# a neighbour line, a lane width away, can never match.
MATCH_M = 0.6
# Each line is walked forward in steps of FOLLOW_STEP_M. At each step it is
# the paint nearest the straight extension of its last crossing, within
# FOLLOW_M. Paint the heading filter keeps runs at most HEADING_MAX_DEG
# (45 deg) off the car, so it moves at most tan(45 deg) * 1 m = 1.0 m
# sideways in one step; and 1.0 m is under half the narrowest lane
# (LANE_MIN_M / 2 = 1.01 m), so the neighbour line is never taken. A step
# with no paint there (a gap between decal pieces) carries the line straight
# on; after more than CARRY_M without paint the line is lost.
FOLLOW_STEP_M = 1.0
FOLLOW_M = 1.0
# A line that starts inside the lane splits it only when it runs with the
# lane: no steeper against the lane's lines than a 10:1 taper (0.1 m/m, the
# shortest lane-shift taper at city speed). Steeper paint crossing the lane
# (junction guide lines, hatching) is not a lane line.
SPLIT_SLOPE_MAX = 0.1
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


class _LineWalk:
    """One painted line walked forward from y = 0 (``lane_at``)."""

    def __init__(self, start: tuple[float, float]) -> None:
        self.y, self.x, self.slope = 0.0, float(start[0]), float(start[1])
        self.seen_y = 0.0  # last y with paint on this line
        self.lost = False

    def guess(self, y: float) -> float:
        return self.x + self.slope * (float(y) - self.y)

    def candidates(self, y: float, lines: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if self.lost:
            return []
        g = self.guess(y)
        return [v for v in lines if abs(v[0] - g) <= FOLLOW_M]

    def take(self, y: float, line: tuple[float, float] | None) -> None:
        if self.lost:
            return
        if line is not None:
            self.x, self.slope = float(line[0]), float(line[1])
            self.seen_y = float(y)
        else:
            self.x = self.guess(y)
            if float(y) - self.seen_y > CARRY_M:
                self.lost = True
        self.y = float(y)

    def here(self, y: float) -> float | None:
        """x at ``y`` when the line has paint there (the walk is at ``y``)."""
        if self.lost or abs(self.seen_y - float(y)) > 1e-9:
            return None
        return self.x


def _step_pair(
    walk: tuple[_LineWalk, _LineWalk], y: float, lines: list[tuple[float, float]], width: float,
) -> float:
    """Advance both lines of the lane to ``y``; returns the lane width there (or the last one).

    Each line takes the paint within FOLLOW_M of its straight extension.
    Where a line forks (a lane line and an exit or edge line leaving it,
    merged as one line until they are DOUBLE_LINE_M apart) both branches
    are candidates. A lane's two lines run parallel, so the lane's own line
    is the branch that runs parallel to the other side: the pair of
    candidates with the least slope difference is taken (nearest the
    extension on a tie, slopes compared to 0.01).
    """
    left, right = walk
    if not left.lost and not right.lost:
        # A lane that splits: a new line starts between the two (a lane
        # opening beside this one), runs with the lane (SPLIT_SLOPE_MAX), and
        # both parts are lanes. The ego lane is the part its own centre runs
        # into; the line on the other side moves to the new line.
        gl, gr = left.guess(y), right.guess(y)
        c = 0.5 * (gl + gr)
        along = 0.5 * (left.slope + right.slope)
        inner = [
            v for v in lines
            if gl + LANE_MIN_M <= v[0] <= gr - LANE_MIN_M and abs(v[1] - along) <= SPLIT_SLOPE_MAX
        ]
        if inner:
            split = min(inner, key=lambda v: abs(v[0] - c))
            if split[0] >= c:
                right.x, right.y, right.slope = split[0], y, split[1]
            else:
                left.x, left.y, left.slope = split[0], y, split[1]
    cl, cr = left.candidates(y, lines), right.candidates(y, lines)
    if cl and cr:
        gl, gr = left.guess(y), right.guess(y)
        lb, rb = min(
            ((a, b) for a in cl for b in cr if b[0] > a[0]),
            key=lambda p: (round(abs(p[1][1] - p[0][1]), 2), abs(p[0][0] - gl) + abs(p[1][0] - gr)),
            default=(None, None),
        )
        left.take(y, lb)
        right.take(y, rb)
        return width if lb is None or rb is None else rb[0] - lb[0]
    # One side has no paint here: the other keeps nearest its extension; the
    # width is kept from the last step where both were seen.
    for side, cand in ((left, cl), (right, cr)):
        g = side.guess(y)
        side.take(y, min(cand, key=lambda v: abs(v[0] - g)) if cand else None)
    return width


class LaneTracker:
    """The previous tick's ego-lane pair for ``PaintedLanes.lane_at``. One per drive.

    Each line is kept as the world point where it crossed y = 0 when it was
    last seen, and its world direction. Next tick it is projected onto the
    car's new y = 0 and matched (``MATCH_M``) to a line seen there.

    The kept pair is replaced by the pair that contains the car (or by none)
    only when the car centre is more than ``SWITCH_MARGIN_M`` past one of its
    lines, or the pair is no longer a lane: both lines unseen, a line
    carried more than ``CARRY_M`` behind the car, paint seen between the two
    (they are not adjacent), or the two out of the lane width range.
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
                # Still a lane: adjacent lines (no paint seen between them;
                # e.g. a lane line that tapers across under the car), a lane
                # wide.
                between = any(left[0] < x < right[0] and (x, sl) != left and (x, sl) != right for x, sl in lines0)
                if not between and LANE_MIN_M <= right[0] - left[0] <= LANE_MAX_M:
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
