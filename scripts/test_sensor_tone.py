#!/usr/bin/env python3
"""Sun-blown colour buffers match the viewport picture before anyone reads them.

The old path stored a channel swap. A near-white road stayed white paint, so
a thin stripe was not a separate region. A highlight knee (140, keep 35%)
still leaves that road light and drops the stripe through the paint cut.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.lanes import estimate_lanes, lane_paint_mask  # noqa: E402
from python.perception.stitch360 import forward_lane_view, stitch_frames  # noqa: E402
from python.runtime.shadow import ShadowConfig  # noqa: E402
from python.sensors.cameras import CAM_IDS, colour_to_bgr, recover_viewport_tone  # noqa: E402
from python.viz.debug_draw import (  # noqa: E402
    CAMS_PREVIEW_DIM,
    CAMS_PREVIEW_SLOTS,
    cam_tile_rects,
    clamp_front_overexpose,
    draw_cam_tiles,
)


def main() -> None:
    gate = ShadowConfig().lane_conf_min
    blown = _highway(
        asphalt=(228, 228, 228),
        yellow=(188, 230, 242),
        white=(252, 252, 252),
        sky=(242, 232, 214),
        width=3,
    )
    # Washed-out behavior: the road itself is white paint, so the stripe is
    # not a separate region. Identity fails the contract below.
    assert int(blown[360, 320, 0]) > 210
    assert int(lane_paint_mask(blown)[360, 320]) == 255, "raw road should be white paint"
    knee = _highlight_knee(blown)
    assert int(knee[360, 320, 0]) > 160, "knee 140 / keep 35% does not reach mid gray"
    assert int(lane_paint_mask(knee)[_stripe_yx(blown, (252, 252, 252))]) == 0

    held = blown.copy()
    corrected = recover_viewport_tone(blown)
    assert np.array_equal(blown, held), "tone must not rewrite the caller's buffer"
    via = colour_to_bgr(np.ascontiguousarray(blown[:, :, ::-1]))
    assert via is not None and np.array_equal(via, corrected)
    _assert_game_road(corrected, blown, gate)

    close = _highway(
        asphalt=(240, 240, 240),
        yellow=(214, 236, 246),
        white=(248, 248, 248),
        sky=(246, 240, 232),
        width=3,
    )
    close_out = colour_to_bgr(np.ascontiguousarray(close[:, :, ::-1]))
    assert close_out is not None
    _assert_game_road(close_out, close, gate, white=(248, 248, 248))

    # Already mid gray: not crushed, and yellow/white paint still counts.
    flat = np.full((48, 64, 3), 140, dtype=np.uint8)
    assert np.array_equal(recover_viewport_tone(flat), flat)
    clear = _highway(
        asphalt=(128, 126, 124),
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        sky=(175, 185, 195),
        width=4,
    )
    assert np.array_equal(recover_viewport_tone(clear), clear)
    clear_fit = estimate_lanes(clear)
    assert clear_fit.conf >= gate and len(clear_fit.lanes_bev) >= 2, clear_fit

    # Hazy sky stays hazy. It must not become white paint, and an unmarked
    # road must not grow a lane.
    haze = _highway(
        asphalt=(168, 170, 172),
        yellow=(140, 190, 205),
        white=(198, 198, 200),
        sky=(175, 185, 195),
        width=2,
    )
    assert np.array_equal(recover_viewport_tone(haze), haze)
    assert int(lane_paint_mask(haze)[8, 8]) == 0
    haze_fit = estimate_lanes(haze)
    assert haze_fit.conf >= gate and len(haze_fit.lanes_bev) >= 2, haze_fit
    bare = _highway(
        asphalt=(168, 170, 172),
        yellow=(140, 190, 205),
        white=(198, 198, 200),
        sky=(175, 185, 195),
        width=2,
        stripes=False,
    )
    assert np.array_equal(recover_viewport_tone(bare), bare)
    assert int(lane_paint_mask(bare)[8, 8]) == 0
    bare_fit = estimate_lanes(bare)
    assert bare_fit.conf < gate and bare_fit.lanes_bev == [], bare_fit

    zeros = np.zeros((16, 16, 3), dtype=np.uint8)
    assert np.array_equal(recover_viewport_tone(zeros), zeros)
    assert int(zeros.max()) == 0
    empty = colour_to_bgr(np.zeros((8, 8, 4), dtype=np.uint8).tobytes(), (8, 8))
    assert empty is not None and int(empty.max()) == 0

    # Same corrected frame for the lane fit and the CAMS tile. The preview
    # dim is only a scale on an already mid-gray picture.
    assert np.array_equal(clamp_front_overexpose(corrected, "main"), corrected)
    _assert_cams_reads_corrected(corrected)
    _assert_missing_slot_stays_empty(corrected)
    _assert_stitch_uses_corrected(corrected)
    again = recover_viewport_tone(corrected)
    assert np.array_equal(again, corrected)
    # A bright gray hood is not the road. Mid-gray asphalt stays, and the
    # lane fit does not drop a lane or grow two more. colour_to_bgr is the
    # store path for sensor frames; recover_viewport_tone is the window grab.
    _assert_midgray_hood_stays((128, 118, 108), (236, 236, 236))
    _assert_midgray_hood_stays((140, 132, 124), (220, 220, 220))
    # Sky (190, 200, 210) sits a few levels over the gate. The 42%–93% band
    # is mostly that sky, so its 40th percentile used to open the curve,
    # crush asphalt (128, 118, 108) to about (40, 32, 25), paint the sky,
    # and grow the fit from 2 lanes to 4. A dark hood reproduces it. The
    # road sample leaves this frame mid gray at 2 lanes.
    _assert_midgray_under_bright_sky()
    print("test_sensor_tone: OK")


def _assert_game_road(
    frame: np.ndarray,
    source: np.ndarray,
    gate: float,
    *,
    white: tuple[int, int, int] = (252, 252, 252),
) -> None:
    road = frame[360, 320]
    assert 80 <= int(road.min()) and int(road.max()) <= 160, tuple(int(v) for v in road)
    assert int(lane_paint_mask(frame)[360, 320]) == 0, "pavement must not stay white paint"
    assert int(lane_paint_mask(frame)[30, 320]) == 0, "sky must not become white paint"
    stripe_at = _stripe_yx(source, white)
    assert int(lane_paint_mask(frame)[stripe_at]) == 255, stripe_at
    yellow_at = _yellow_yx(source)
    assert int(lane_paint_mask(frame)[yellow_at]) == 255, yellow_at
    fit = estimate_lanes(frame)
    assert fit.conf >= gate and len(fit.lanes_bev) >= 2, fit


def _assert_cams_reads_corrected(corrected: np.ndarray) -> None:
    tile_h, tile_w = 72, 96
    undimmed, label, ok = _one_tile(corrected, tile_w, tile_h, dim=None)
    assert ok and label == "ok"
    road = undimmed[48:68, 36:60]
    assert 80 <= float(road.mean()) <= 160, float(road.mean())
    dimmed, dim_label, dim_ok = _one_tile(corrected, tile_w, tile_h, dim=CAMS_PREVIEW_DIM)
    assert dim_ok and dim_label == "ok"
    dim_road = float(dimmed[48:68, 36:60].mean())
    assert 50 <= dim_road <= 150, dim_road
    assert dim_road < float(road.mean())


def _one_tile(frame: np.ndarray, width: int, height: int, dim: float | None):
    from python.viz.debug_draw import _paint_cam_tile

    return _paint_cam_tile(width, height, "main", frame, "ok", dropped=False, dim=dim)


def _assert_missing_slot_stays_empty(corrected: np.ndarray) -> None:
    canvas = np.full((220, 330, 3), (1, 2, 3), dtype=np.uint8)
    frames = {cid: corrected for cid in CAM_IDS if cid != "rear"}
    health = {cid: "ok" for cid in CAM_IDS}
    draw_cam_tiles(
        canvas,
        frames,
        health,
        x0=0,
        y0=0,
        width=330,
        height=220,
        cols=3,
        rows=3,
        ids=CAMS_PREVIEW_SLOTS,
        dim=CAMS_PREVIEW_DIM,
    )
    rects = cam_tile_rects(0, 0, 330, 220, 3, 3)
    center = rects[4]
    cx, cy, cw, ch = center
    assert tuple(int(v) for v in canvas[cy + ch // 2, cx + cw // 2]) == (1, 2, 3)
    rear = rects[7]
    rx, ry, rw, rh = rear
    sample = canvas[ry + rh // 2, rx + rw - 8]
    assert int(sample.max()) < 40, tuple(int(v) for v in sample)
    assert "rear" not in frames


def _assert_stitch_uses_corrected(corrected: np.ndarray) -> None:
    frames = {cid: corrected for cid in CAM_IDS if cid != "rear"}
    stitched = stitch_frames(frames)
    rear = next(sec for sec in stitched.sectors if sec.cam_id == "rear")
    assert rear.present is False
    assert int(stitched.bgr[:, rear.x0 : rear.x1].max()) == 0
    band = forward_lane_view(stitched.bgr)
    med = float(np.median(band))
    assert 60 <= med <= 170, med


def _highlight_knee(img: np.ndarray, knee: float = 140.0, keep: float = 0.35) -> np.ndarray:
    x = img.astype(np.float32)
    y = np.where(x <= knee, x, knee + keep * (x - knee))
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def _assert_midgray_hood_stays(
    asphalt: tuple[int, int, int],
    hood: tuple[int, int, int],
) -> None:
    raw = _highway(
        asphalt=asphalt,
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        sky=(175, 185, 195),
        width=4,
        hood=hood,
    )
    before = estimate_lanes(raw)
    assert before.conf >= 0.9 and len(before.lanes_bev) == 2, before
    via = colour_to_bgr(np.ascontiguousarray(raw[:, :, ::-1]))
    window = recover_viewport_tone(raw)
    assert via is not None
    assert np.array_equal(via, raw), tuple(int(v) for v in via[360, 320])
    assert np.array_equal(window, raw), tuple(int(v) for v in window[360, 320])
    assert tuple(int(v) for v in via[360, 320]) == asphalt
    after = estimate_lanes(via)
    assert after.conf >= 0.9 and len(after.lanes_bev) == 2, after


def _assert_midgray_under_bright_sky() -> None:
    raw = _highway(
        asphalt=(128, 118, 108),
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        sky=(190, 200, 210),
        width=4,
        hood=(35, 35, 38),
    )
    before = estimate_lanes(raw)
    assert before.conf >= 0.9 and len(before.lanes_bev) == 2, before
    via = colour_to_bgr(np.ascontiguousarray(raw[:, :, ::-1]))
    window = recover_viewport_tone(raw)
    assert via is not None
    road = tuple(int(v) for v in via[360, 320])
    assert np.array_equal(via, raw), road
    assert np.array_equal(window, raw), tuple(int(v) for v in window[360, 320])
    assert road == (128, 118, 108)
    assert int(lane_paint_mask(via)[30, 320]) == 0, "sky must stay off the paint mask"
    after = estimate_lanes(via)
    assert after.conf >= 0.9 and len(after.lanes_bev) == 2, after


def _highway(
    *,
    asphalt: tuple[int, int, int],
    yellow: tuple[int, int, int],
    white: tuple[int, int, int],
    sky: tuple[int, int, int],
    width: int,
    stripes: bool = True,
    hood: tuple[int, int, int] = (35, 35, 38),
) -> np.ndarray:
    cv2 = __import__("cv2")
    w, h = 640, 480
    img = np.full((h, w, 3), sky, dtype=np.uint8)
    vp = (int(w * 0.5), int(h * 0.46))
    cv2.fillPoly(
        img,
        [np.array([[int(w * 0.02), h - 1], vp, [int(w * 0.98), h - 1]], np.int32)],
        asphalt,
    )

    def _draw(xb: float, color: tuple[int, int, int]) -> None:
        a = (int(xb), h - 8)
        b = (int(w * 0.5 + (xb / w - 0.5) * 18), int(h * 0.46) + 4)
        cv2.line(img, a, b, color, width)

    if stripes:
        _draw(w * 0.28, yellow)
        _draw(w * 0.72, white)
    cv2.rectangle(img, (0, int(h * 0.93)), (w, h), hood, -1)
    return img


def _stripe_yx(img: np.ndarray, white: tuple[int, int, int]) -> tuple[int, int]:
    sel = (
        (img[:, :, 0] >= white[0] - 1)
        & (img[:, :, 1] >= white[1] - 1)
        & (img[:, :, 2] >= white[2] - 1)
    )
    sel[:230, :] = False
    sel[440:, :] = False
    ys, xs = np.where(sel)
    assert len(xs) > 0, "white stripe missing"
    mid = len(xs) // 2
    return int(ys[mid]), int(xs[mid])


def _yellow_yx(img: np.ndarray) -> tuple[int, int]:
    sel = (img[:, :, 0] < 220) & (img[:, :, 1] > 220) & (img[:, :, 2] > 230)
    sel[:230, :] = False
    sel[440:, :] = False
    ys, xs = np.where(sel)
    assert len(xs) > 0, "yellow stripe missing"
    mid = len(xs) // 2
    return int(ys[mid]), int(xs[mid])


if __name__ == "__main__":
    main()
