"""Synthetic road frames for the learned-lane property tests (no BeamNG).

Every frame here is built from parameters, so hypothesis can shrink it. The
``no-paint`` families never contain a thin painted stripe: blank, uniform,
sky over asphalt, per-pixel noise, and edge-only roads (grass, dirt, or a
wide bright concrete curb / sidewalk band on either side). ``painted`` draws
thin white or yellow stripes on asphalt for the liveness checks.
"""
from __future__ import annotations

import cv2
import numpy as np

ASPHALT = (70, 72, 74)
GRASS = (40, 120, 60)
DIRT = (60, 100, 130)
YELLOW = (40, 190, 230)
WHITE = (235, 235, 235)


def uniform(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = np.asarray(bgr, dtype=np.uint8)
    return img


def noise(h: int, w: int, seed: int, lo: int = 0, hi: int = 256) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(lo, hi, size=(h, w, 3), dtype=np.uint8)


def sky_over_asphalt(h: int, w: int, horizon: float, sky_v: int) -> np.ndarray:
    img = uniform(h, w, ASPHALT)
    top = int(round(h * horizon))
    if top > 0:
        ramp = np.linspace(sky_v, max(0, sky_v - 40), top, dtype=np.float32)
        img[:top, :, 0] = np.clip(ramp[:, None] + 25, 0, 255).astype(np.uint8)
        img[:top, :, 1] = np.clip(ramp[:, None] + 5, 0, 255).astype(np.uint8)
        img[:top, :, 2] = np.clip(ramp[:, None], 0, 255).astype(np.uint8)
    return img


def _road_poly(h: int, w: int, horizon: float, half_bottom: float, half_top: float, shift: float) -> np.ndarray:
    top = int(round(h * horizon))
    cx = w * (0.5 + shift)
    return np.array(
        [[
            (int(cx - half_bottom * w), h - 1),
            (int(cx - half_top * w), top),
            (int(cx + half_top * w), top),
            (int(cx + half_bottom * w), h - 1),
        ]],
        dtype=np.int32,
    )


def edge_only(
    h: int,
    w: int,
    *,
    horizon: float,
    half_bottom: float,
    half_top: float,
    shift: float,
    side: str,
    curb_v: int,
    curb_px: int,
) -> np.ndarray:
    """Asphalt road on a non-paint background, with optional wide curb bands.

    ``side`` is the background: ``grass``, ``dirt`` or ``concrete`` (a bright,
    low-saturation sidewalk with value ``curb_v``). ``curb_px`` > 0 adds a
    bright concrete curb band, ``curb_px`` wide in the image, outside both
    road edges. No thin stripe is drawn, so there is no paint in the frame.
    A band narrower than ``PAINT_MAX_WIDTH_PX`` is a thin bright line in
    pixels, i.e. indistinguishable from paint, so the tests draw wider ones.
    """
    bg = {"grass": GRASS, "dirt": DIRT, "concrete": (curb_v, curb_v, curb_v)}[side]
    img = sky_over_asphalt(h, w, horizon, 200)
    top = int(round(h * horizon))
    img[top:] = np.asarray(bg, dtype=np.uint8)
    poly = _road_poly(h, w, horizon, half_bottom, half_top, shift)
    if curb_px > 0:
        # A constant-width band (curb_px wide in the image) outside each road edge.
        outer = poly.copy()
        outer[0][0][0] -= int(curb_px)
        outer[0][1][0] -= int(curb_px)
        outer[0][2][0] += int(curb_px)
        outer[0][3][0] += int(curb_px)
        cv2.fillPoly(img, outer, (curb_v, curb_v, curb_v))
    cv2.fillPoly(img, poly, ASPHALT)
    return img


def painted(
    h: int,
    w: int,
    *,
    horizon: float,
    half_bottom: float,
    half_top: float,
    shift: float,
    stripe_px: int,
    yellow_left: bool,
) -> np.ndarray:
    """Asphalt road with a thin painted line on each lane edge."""
    img = edge_only(h, w, horizon=horizon, half_bottom=half_bottom, half_top=half_top,
                    shift=shift, side="grass", curb_v=0, curb_px=0)
    poly = _road_poly(h, w, horizon, half_bottom * 0.8, half_top * 0.8, shift)[0]
    left = YELLOW if yellow_left else WHITE
    cv2.line(img, tuple(int(v) for v in poly[0]), tuple(int(v) for v in poly[1]), left, int(stripe_px))
    cv2.line(img, tuple(int(v) for v in poly[3]), tuple(int(v) for v in poly[2]), WHITE, int(stripe_px))
    return img
