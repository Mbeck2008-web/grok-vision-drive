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
LANE_WHITE = 5
LANE_YELLOW = 6
STOP_LINE = 7
CLASS_NAMES: tuple[str, ...] = ("unknown", "road", "curb", "vehicle", "blocked", "lane_white", "lane_yellow", "stop_line")
PAINT_CLASSES = frozenset({LANE_WHITE, LANE_YELLOW, STOP_LINE})
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
SRC_MOTION = "tracks:multi_frame"        # tracker displacement over >= 3 frames spanning >= 0.2 s
SRC_PAINT = "decals:lane_line_material"  # painted_lanes DecalRoad paint, colour from material name
SRC_STOP = "decals:stop_lines"           # painted_lanes.stop_lines_ahead (label_stoplines sidecar)
SRC_FLOW = "boxes:traffic_vel+static"    # per-cell ground velocity: traffic rows' vel; static world = 0
SRC_OBJ_HEAD = "boxes:traffic"           # BEV object-centre head
SRC_LANE_LINE = "derived:occ_paint_cells"
SRC_STOP_LINE = "derived:occ_stop_cells"

CLASS_SRC: Mapping[int, str] = {
    UNKNOWN: SRC_OCC,
    ROAD: SRC_ROAD,
    CURB: SRC_CURB,
    VEHICLE: SRC_VEHICLE,
    BLOCKED: SRC_BLOCKED,
    LANE_WHITE: SRC_PAINT,
    LANE_YELLOW: SRC_PAINT,
    STOP_LINE: SRC_STOP,
}

KNOWN_SOURCES = frozenset(
    {SRC_OCC, SRC_ROAD, SRC_CURB, SRC_VEHICLE, SRC_BLOCKED, SRC_EDGE, SRC_CURB_LINE, SRC_MOTION, SRC_PAINT, SRC_STOP,
     SRC_FLOW, SRC_OBJ_HEAD, SRC_LANE_LINE, SRC_STOP_LINE, UNLABELED}
)

# Outputs the visualizer can draw that have no recorder label source yet.
# Crosswalk stripes: painted_lanes only has a masked "cross paint" bucket, not a crosswalk label.
UNLABELED_OUTPUTS: tuple[str, ...] = ("crosswalks", "signs")

# Class id -> RGB for occupied-cell rendering. Red only on CURB (edges are
# polylines and are drawn red by the renderer). VEHICLE is blue, as the
# early-beta car boxes.
CLASS_RGB: Mapping[int, tuple[int, int, int]] = {
    UNKNOWN: (90, 90, 90),
    ROAD: (40, 40, 48),
    CURB: (230, 30, 30),
    VEHICLE: (40, 120, 255),
    BLOCKED: (255, 150, 30),
    LANE_WHITE: (235, 235, 235),
    LANE_YELLOW: (250, 205, 40),
    STOP_LINE: (245, 245, 245),
}
RED_CLASSES = frozenset({CURB})
RED_POLYLINE_KINDS = frozenset({"road_edge", "curb"})


def is_red(rgb: tuple[int, int, int]) -> bool:
    """Reddish: red channel dominant and clearly above green and blue."""
    r, g, b = (int(v) for v in rgb)
    return r >= 150 and r >= 2 * g and r >= 2 * b


def class_src(cls: int) -> str:
    return CLASS_SRC.get(int(cls), UNLABELED)
