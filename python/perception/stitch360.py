"""Flat 360° strip of the rig frames already delivered on one tick.

``CameraFrameBundle.frames`` holds every camera that has a real picture on
that grab. Companions are polled on a hitch, and the bundle keeps the last
real frame, so the eight slots are together even when only one of them is
new this tick.

The strip is left to right around the car:

    repeatL, pillarL, wide, main, narrow, pillarR, repeatR, rear

Look yaw runs from the left repeater, across the nose, to the right
repeater, then the rear plate. The three yaw-0 windshield cameras stay in
mount order: wide is the left offset, narrow the right. There is no overlap
calibration and no pose warp, so each camera is its own sector and the seams
are empty gaps. The join from rear back to repeatL is not drawn.

The horizontal spans already add up to more than a circle, because narrow,
main, and wide share yaw 0 and the rear plate covers both repeater aims.
A small field-of-view or yaw nudge would not turn those seams into one
viewpoint, so this strip does not retune the rig. Pillar yaw stays ±78°
and the repeater mounts stay at Y 1.30.

A missing or unrendered camera leaves that sector empty. It is not filled
from another camera.

Repeater inboard edges show the ego body (image left on repeatL, image right
on repeatR). Those columns are cleared and marked. They are not another
vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from python.sensors.cameras import frame_is_unrendered

# Fixed left-to-right order around the car. Not the hitch order.
STITCH_ORDER: tuple[str, ...] = (
    "repeatL",
    "pillarL",
    "wide",
    "main",
    "narrow",
    "pillarR",
    "repeatR",
    "rear",
)

SECTOR_W = 160
SECTOR_H = 96
GAP_PX = 8

# Inboard edge of a repeater frame. A body-volume projection into the
# ±160° repeater (fov_v 55.4, 4:3) stays inside this fraction. It is the
# frame edge, not a measured mesh.
EGO_EDGE_FRAC = 0.25
EGO_EDGE: dict[str, str] = {"repeatL": "left", "repeatR": "right"}

STITCH_NOTE = (
    "Fixed left-to-right order around the car "
    "(repeatL, pillarL, wide, main, narrow, pillarR, repeatR, rear). "
    "Seams are empty gaps: no overlap calibration and no pose warp. "
    "A missing camera stays an empty sector. "
    "Repeater inboard edges are ego body and are not another vehicle."
)


@dataclass(frozen=True)
class Sector360:
    cam_id: str
    x0: int
    x1: int
    present: bool
    ego_x0: int | None = None
    ego_x1: int | None = None


@dataclass
class Stitch360:
    bgr: np.ndarray
    order: tuple[str, ...]
    sectors: tuple[Sector360, ...]
    gap_px: int
    ego_body: np.ndarray
    note: str

    def allows_vehicle(self, x: int, y: int) -> bool:
        """False on repeater ego-body pixels, gaps, and empty sectors.

        A present camera's other pixels may be some other object. The ego
        body in a repeater sector is never another vehicle.
        """
        h, w = self.ego_body.shape[:2]
        if y < 0 or x < 0 or y >= h or x >= w:
            return False
        if bool(self.ego_body[y, x]):
            return False
        for sec in self.sectors:
            if sec.x0 <= x < sec.x1:
                return bool(sec.present)
        return False


def ego_columns(cam_id: str, width: int) -> tuple[int, int] | None:
    """Inboard column range ``[x0, x1)`` for a repeater, else None."""
    edge = EGO_EDGE.get(cam_id)
    if edge is None or width <= 0:
        return None
    n = max(1, int(round(float(width) * EGO_EDGE_FRAC)))
    n = min(n, int(width))
    if edge == "left":
        return (0, n)
    return (int(width) - n, int(width))


def _as_bgr(img: Any) -> np.ndarray | None:
    if frame_is_unrendered(img):
        return None
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[-1] > 3:
        arr = np.ascontiguousarray(arr[:, :, :3])
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if frame_is_unrendered(arr):
        return None
    return arr


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        sh, sw = img.shape[:2]
        ys = np.linspace(0, max(sh - 1, 0), height).astype(np.int32)
        xs = np.linspace(0, max(sw - 1, 0), width).astype(np.int32)
        return np.ascontiguousarray(img[ys][:, xs])


def _frame_for(frames: Mapping[str, Any], cam_id: str) -> Any:
    img = frames.get(cam_id)
    if cam_id == "main" and frame_is_unrendered(img):
        img = frames.get("cam_main")
    return img


def stitch_frames(frames: Mapping[str, Any] | Any | None) -> Stitch360:
    """Lay delivered frames into one strip. Missing slots stay empty."""
    if frames is not None and not isinstance(frames, Mapping) and hasattr(frames, "frames"):
        frames = getattr(frames, "frames")
    src: Mapping[str, Any] = frames if isinstance(frames, Mapping) else {}

    n = len(STITCH_ORDER)
    width = n * SECTOR_W + (n - 1) * GAP_PX
    bgr = np.zeros((SECTOR_H, width, 3), dtype=np.uint8)
    ego = np.zeros((SECTOR_H, width), dtype=bool)
    sectors: list[Sector360] = []
    x = 0
    for cam_id in STITCH_ORDER:
        raw = _as_bgr(_frame_for(src, cam_id))
        ego_span: tuple[int, int] | None = None
        present = raw is not None
        if raw is not None:
            placed = _resize(raw, SECTOR_W, SECTOR_H)
            ego_span = ego_columns(cam_id, SECTOR_W)
            if ego_span is not None:
                x0e, x1e = ego_span
                placed[:, x0e:x1e] = 0
                ego[:, x + x0e : x + x1e] = True
            bgr[:, x : x + SECTOR_W] = placed
        sectors.append(
            Sector360(
                cam_id=cam_id,
                x0=x,
                x1=x + SECTOR_W,
                present=present,
                ego_x0=None if ego_span is None else x + ego_span[0],
                ego_x1=None if ego_span is None else x + ego_span[1],
            )
        )
        x += SECTOR_W + GAP_PX
    return Stitch360(
        bgr=bgr,
        order=STITCH_ORDER,
        sectors=tuple(sectors),
        gap_px=GAP_PX,
        ego_body=ego,
        note=STITCH_NOTE,
    )
