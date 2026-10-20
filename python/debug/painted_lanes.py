"""Lane targets from the painted line decals the camera actually sees.

The old label (gt_geom.follow_label) projects the car onto
``get_road_network(drivable_only=True)`` edges. On west_coast_usa 1294 of
those 1467 roads are ``road_invisible`` AI splines whose width is the nav
corridor, and whose lanesLeft/lanesRight default to 1/1. Splitting that
width into equal lanes gave 1.75-7.5 m lanes that do not sit on the paint.

Here the lane is bracketed by the nearest painted line decals
(``line_white``, ``line_yellow``, ``line_dashed_*`` ...) left and right of
the car at each lookahead, in the vehicle frame (x right, y forward, origin
at the state ``pos`` ref node). Each lookahead is intersected separately, so
curves are modelled. No paint on one side means the slot is invalid (NaN,
mask 0), never a synthetic 3.5 m guess.

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
# The chosen lane's center must stay this close to the predicted center.
TRACK_MAX_M = 2.5
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

    def lane_at(
        self,
        pos: Any,
        direction: Any,
        ys: tuple[float, ...] = LOOKAHEADS_M,
        *,
        z_max_m: float | None = Z_MAX_M,
        heading_max_deg: float | None = HEADING_MAX_DEG,
    ) -> list[tuple[float, float] | None]:
        """``(left_x, right_x)`` of the painted ego lane at each ``y``, vehicle frame.

        The first lookahead takes the lane whose center is nearest the car.
        Each later one takes the lane nearest the center extrapolated from
        the previous two, so a curve that leaves the straight-ahead line is
        still followed. ``z_max_m`` / ``heading_max_deg`` of None disable that
        filter (diagnostics only). ``pos`` without a z skips the z filter.
        """
        f = np.asarray([float(direction[0]), float(direction[1])])
        norm = float(np.hypot(*f))
        if norm < 1e-6:
            return [None] * len(ys)
        f /= norm
        r = np.asarray([f[1], -f[0]])
        cx, cy = int(float(pos[0]) // _CELL), int(float(pos[1]) // _CELL)
        idx = [self.grid[(cx + a, cy + b)] for a in range(-2, 3) for b in range(-2, 3) if (cx + a, cy + b) in self.grid]
        if not idx:
            return [None] * len(ys)
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
            return [None] * len(ys)
        p0 = np.asarray([float(pos[0]), float(pos[1])])
        a, b = s[:, :2] - p0, s[:, 3:5] - p0
        ax, ay, bx, by = a @ r, a @ f, b @ r, b @ f
        out: list[tuple[float, float] | None] = []
        seen: list[tuple[float, float]] = []
        for y in ys:
            if len(seen) >= 2:
                (y0, c0), (y1, c1) = seen[-2], seen[-1]
                pred = c1 + (c1 - c0) * (float(y) - y1) / (y1 - y0)
            elif seen:
                pred = seen[-1][1]
            else:
                pred = 0.0
            m = ((ay - y) * (by - y) <= 0.0) & (np.abs(by - ay) > 1e-6)
            hits = ax[m] + (y - ay[m]) * (bx[m] - ax[m]) / (by[m] - ay[m])
            xs = np.sort(hits[np.abs(hits - pred) < SEARCH_M])
            merged: list[float] = []
            for v in xs:
                if merged and v - merged[-1] <= DOUBLE_LINE_M:
                    merged[-1] = 0.5 * (merged[-1] + float(v))
                else:
                    merged.append(float(v))
            best: tuple[float, float, float] | None = None
            for lx, rx in zip(merged, merged[1:]):
                if not (LANE_MIN_M <= rx - lx <= LANE_MAX_M):
                    continue
                miss = abs(0.5 * (lx + rx) - pred)
                if best is None or miss < best[0]:
                    best = (miss, lx, rx)
            if best is None or best[0] > TRACK_MAX_M:
                out.append(None)
                continue
            out.append((best[1], best[2]))
            seen.append((float(y), 0.5 * (best[1] + best[2])))
        return out


def heading_from_centers(c5: float | None, c10: float | None) -> float | None:
    """Same sign as ``gt_geom.ego_center_x``: positive when the car points left of the lane."""
    if c5 is None or c10 is None:
        return None
    return float(math.asin(max(-1.0, min(1.0, (float(c10) - float(c5)) / 5.0))))
