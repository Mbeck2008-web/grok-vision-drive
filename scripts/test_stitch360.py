#!/usr/bin/env python3
"""360° strip: camera order, empty sectors, ego body, and the E2E view."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.e2e import E2E_H, E2E_W, E2EIntent, E2EPolicy
from python.perception.lanes import estimate_lanes
from python.perception.stitch360 import (
    EGO_EDGE_FRAC,
    GAP_PX,
    SECTOR_H,
    SECTOR_W,
    STITCH_ORDER,
    ego_columns,
    stitch_frames,
)
from python.runtime.shadow import ShadowConfig, shadow_tick


def _solid(color: tuple[int, int, int], h: int = 48, w: int = 80) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _colors() -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for i, cid in enumerate(STITCH_ORDER):
        out[cid] = (20 + i * 25, 30 + i * 17, 40 + i * 13)
    vals = list(out.values())
    assert len(set(vals)) == len(vals)
    assert all(max(c) > 0 for c in vals)
    return out


def _center(st, sec) -> tuple[int, int, int]:
    x = (sec.x0 + sec.x1) // 2
    y = SECTOR_H // 2
    pix = st.bgr[y, x]
    return (int(pix[0]), int(pix[1]), int(pix[2]))


def _resized(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(img, (E2E_W, E2E_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def main() -> None:
    colors = _colors()
    frames = {cid: _solid(colors[cid]) for cid in STITCH_ORDER}
    st = stitch_frames(frames)

    assert st.order == STITCH_ORDER
    assert [sec.cam_id for sec in st.sectors] == list(STITCH_ORDER)
    assert st.bgr.shape[0] == SECTOR_H
    assert st.bgr.shape[1] == len(STITCH_ORDER) * SECTOR_W + (len(STITCH_ORDER) - 1) * GAP_PX
    xs = [sec.x0 for sec in st.sectors]
    assert xs == sorted(xs)
    assert all(b > a for a, b in zip(xs, xs[1:]))
    for sec in st.sectors:
        assert _center(st, sec) == colors[sec.cam_id], (sec.cam_id, _center(st, sec))
        assert sec.present

    # Seams are empty. They are not a blend of the cameras on either side.
    left, right = st.sectors[0], st.sectors[1]
    gap = st.bgr[:, left.x1 : right.x0]
    assert gap.shape[1] == GAP_PX
    assert int(gap.max()) == 0
    assert "no overlap" in st.note

    # A missing camera keeps its sector and stays empty.
    partial = dict(frames)
    del partial["pillarR"]
    hole = stitch_frames(partial)
    assert [sec.x0 for sec in hole.sectors] == xs
    missing = next(sec for sec in hole.sectors if sec.cam_id == "pillarR")
    assert not missing.present
    block = hole.bgr[:, missing.x0 : missing.x1]
    assert int(block.max()) == 0
    for cid, col in colors.items():
        if cid == "pillarR":
            continue
        assert not np.any(np.all(block == np.array(col, dtype=np.uint8), axis=2))
    # Neighbours keep their own pixels.
    for sec in hole.sectors:
        if sec.cam_id == "pillarR":
            continue
        assert _center(hole, sec) == colors[sec.cam_id]

    # Unrendered black is a gap, not a picture copied from main.
    blank = dict(frames)
    blank["wide"] = np.zeros_like(frames["wide"])
    unrendered = stitch_frames(blank)
    wide = next(sec for sec in unrendered.sectors if sec.cam_id == "wide")
    assert not wide.present
    assert int(unrendered.bgr[:, wide.x0 : wide.x1].max()) == 0

    # cam_main fills the main sector only.
    alias = stitch_frames({"cam_main": frames["main"]})
    for sec in alias.sectors:
        if sec.cam_id == "main":
            assert sec.present and _center(alias, sec) == colors["main"]
        else:
            assert not sec.present

    # Repeater inboard edge is ego body, not another vehicle.
    y = SECTOR_H // 2
    left_n = ego_columns("repeatL", SECTOR_W)
    right_n = ego_columns("repeatR", SECTOR_W)
    assert left_n is not None and right_n is not None
    assert left_n[1] - left_n[0] == max(1, int(round(SECTOR_W * EGO_EDGE_FRAC)))
    rep_l = next(sec for sec in st.sectors if sec.cam_id == "repeatL")
    rep_r = next(sec for sec in st.sectors if sec.cam_id == "repeatR")
    assert rep_l.ego_x0 == rep_l.x0 and rep_l.ego_x1 == rep_l.x0 + left_n[1]
    assert rep_r.ego_x1 == rep_r.x1 and rep_r.ego_x0 == rep_r.x1 - (right_n[1] - right_n[0])
    assert not st.allows_vehicle(rep_l.ego_x0, y)
    assert st.bgr[y, rep_l.ego_x0].tolist() == [0, 0, 0]
    assert st.allows_vehicle(rep_l.x1 - 2, y)
    assert _center(st, rep_l) == colors["repeatL"]
    assert not st.allows_vehicle(rep_r.ego_x1 - 1, y)
    assert st.bgr[y, rep_r.ego_x1 - 1].tolist() == [0, 0, 0]
    assert st.allows_vehicle(rep_r.x0 + 2, y)
    main_sec = next(sec for sec in st.sectors if sec.cam_id == "main")
    assert main_sec.ego_x0 is None
    assert int(st.ego_body[:, main_sec.x0 : main_sec.x1].sum()) == 0
    assert st.allows_vehicle((main_sec.x0 + main_sec.x1) // 2, y)
    # An empty repeater is a gap, not an ego-body mask.
    no_rep = stitch_frames({cid: img for cid, img in frames.items() if cid not in ("repeatL", "repeatR")})
    assert int(no_rep.ego_body.sum()) == 0

    empty = stitch_frames(None)
    assert all(not sec.present for sec in empty.sectors)
    assert int(empty.bgr.max()) == 0
    assert not empty.allows_vehicle(0, 0)

    # The hardcoded model reads the stitch, not the main crop.
    main = _solid((0, 255, 0), h=36, w=64)
    wide = _solid((255, 0, 0), h=36, w=64)
    pol = E2EPolicy(model_path=ROOT / "models" / "no_such_e2e.onnx", seed=3)
    assert pol.backend == "stub"
    seen: dict[str, np.ndarray | str] = {}
    real = pol._forward_numpy

    def _wrap(feats):
        seen["source"] = feats["source"]
        seen["cams"] = np.array(feats["cams"], copy=True)
        return real(feats)

    pol._forward_numpy = _wrap
    out = pol.forward(main, wide, stitch_bgr=st.bgr, speed_mps=4.0, steer_deg=1.0)
    assert out.ok
    assert -1.0 <= out.steer <= 1.0 and -1.0 <= out.accel <= 1.0
    assert pol.last_source == "stitch"
    assert seen["source"] == "stitch"
    expected = _resized(st.bgr)
    assert pol.last_view is not None
    assert np.allclose(pol.last_view, expected)
    cams = seen["cams"]
    assert isinstance(cams, np.ndarray)
    assert np.allclose(cams[0], expected.transpose(2, 0, 1))
    assert np.allclose(cams[1], expected.transpose(2, 0, 1))
    main_view = _resized(main)
    assert not np.allclose(pol.last_view, main_view)
    pol.forward(np.full_like(main, 9), wide, stitch_bgr=st.bgr)
    assert pol.last_view is not None and np.allclose(pol.last_view, expected)
    other = np.full_like(st.bgr, (8, 16, 32))
    pol.forward(main, wide, stitch_bgr=other)
    assert pol.last_view is not None and not np.allclose(pol.last_view, expected)
    assert np.allclose(pol.last_view, _resized(other))

    # Callers that still pass only main and wide keep that path.
    legacy = pol.forward(main, wide, speed_mps=2.0)
    assert legacy.ok
    assert pol.last_source == "main_wide"
    assert pol.last_view is not None and np.allclose(pol.last_view, main_view)
    assert seen["source"] == "main_wide"
    legacy_cams = seen["cams"]
    assert isinstance(legacy_cams, np.ndarray)
    assert np.allclose(legacy_cams[0], main_view.transpose(2, 0, 1))
    assert np.allclose(legacy_cams[1], _resized(wide).transpose(2, 0, 1))

    class _Rec:
        backend = "stub"

        def __init__(self) -> None:
            self.stitch = None

        def forward(self, main_bgr, wide_bgr=None, **kwargs) -> E2EIntent:
            self.stitch = kwargs.get("stitch_bgr")
            return E2EIntent(ok=True, backend="stub", reason="rec")

    rec = _Rec()
    shadow_tick(
        policy="shadow",
        engaged=False,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=[{"x": 0.0, "y": 1.0, "z": 0.0}, {"x": 0.0, "y": 2.0, "z": 0.0}],
        planner={"target_v": 8, "aeb": "off"},
        ego_speed_mps=3.0,
        lane_conf=0.9,
        path_conf=0.8,
        seq=1,
        e2e_policy=rec,
        main_bgr=main,
        wide_bgr=wide,
        stitch_bgr=st.bgr,
        cfg=ShadowConfig(),
    )
    assert rec.stitch is st.bgr

    # Main-only lane fit still accepts a main frame. It does not read the strip.
    fit = estimate_lanes(main)
    assert fit.conf == 0.0 and fit.lanes_bev == []
    fit_none = estimate_lanes(None)
    assert fit_none.conf == 0.0 and fit_none.lanes_bev == []

    print("test_stitch360: OK")


if __name__ == "__main__":
    main()
