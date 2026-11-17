"""Occupancy classes, label sources and the viz colour rule.

Every output the occupancy stack emits carries ``src``: the training label
source that supervised it, or ``UNLABELED`` when no recorder signal backs it.
Red is reserved for road edges and curbs (Michael's lane rule).
"""
from __future__ import annotations

from typing import Mapping

# Semantic classes of an occupied voxel. 0 means "occupied, class not labeled".
UNKNOWN = 0
ROAD = 1
CURB = 2
VEHICLE = 3
BLOCKED = 4
CLASS_NAMES: tuple[str, ...] = ("unknown", "road", "curb", "vehicle", "blocked")
N_CLASSES = len(CLASS_NAMES)

UNLABELED = "UNLABELED"

# Where each class's ground truth comes from (training only).
SRC_OCC = "depth:main+wide"              # occupied vs free (depth hit / ray free space)
SRC_ROAD = "sem:drivable_allowlist"      # sem name in DRIVABLE_CLASS_ALLOWLIST
SRC_CURB = "depth+sem:road_boundary_height"
SRC_VEHICLE = "boxes:traffic"            # recorder traffic rows (pos/dir) -> ego boxes
SRC_BLOCKED = "depth:height_above_ground"
SRC_EDGE = "derived:occ_road_boundary"   # polylines from predicted ROAD cells
SRC_CURB_LINE = "derived:occ_curb_cells"
SRC_MOTION = "tracks:multi_frame"        # tracker displacement over >= 2 frames

CLASS_SRC: Mapping[int, str] = {
    UNKNOWN: SRC_OCC,
    ROAD: SRC_ROAD,
    CURB: SRC_CURB,
    VEHICLE: SRC_VEHICLE,
    BLOCKED: SRC_BLOCKED,
}

KNOWN_SOURCES = frozenset(
    {SRC_OCC, SRC_ROAD, SRC_CURB, SRC_VEHICLE, SRC_BLOCKED, SRC_EDGE, SRC_CURB_LINE, SRC_MOTION, UNLABELED}
)

# Outputs the visualizer can draw that have no recorder label source yet.
UNLABELED_OUTPUTS: tuple[str, ...] = ("lanes.paint", "crosswalks", "stop_lines", "signs")

# Class id -> RGB for occupied-cell rendering. Red only on CURB (edges are
# polylines and are drawn red by the renderer). VEHICLE is blue, as the
# early-beta car boxes.
CLASS_RGB: Mapping[int, tuple[int, int, int]] = {
    UNKNOWN: (90, 90, 90),
    ROAD: (40, 40, 48),
    CURB: (230, 30, 30),
    VEHICLE: (40, 120, 255),
    BLOCKED: (255, 150, 30),
}
RED_CLASSES = frozenset({CURB})
RED_POLYLINE_KINDS = frozenset({"road_edge", "curb"})


def is_red(rgb: tuple[int, int, int]) -> bool:
    """Reddish: red channel dominant and clearly above green and blue."""
    r, g, b = (int(v) for v in rgb)
    return r >= 150 and r >= 2 * g and r >= 2 * b


def class_src(cls: int) -> str:
    return CLASS_SRC.get(int(cls), UNLABELED)
