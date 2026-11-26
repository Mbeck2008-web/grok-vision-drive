#!/usr/bin/env python3
"""GT-named offline tests for Reality View step 8(1): the C-from-GT quality rubric.

Spec: docs/specs/reality_view_3dgs.md, "8. Kill or graduate (quality rubric)"
item (1) and "Step 8 quality rubric". This file consumes sim GT, so it is
GT-named (``check_no_sim_gt_on_live_path``: a GT fixture is only consumed by a
test named GT). Everything is generated in-process from a tiny analytic scene:
no committed binaries, no RNG, no wall clock, numpy only.

Checks:
- check_useful_view_metrics            (C-from-GT, spec thresholds)
- check_useful_view_checklist          (C-from-GT, six yes/no items)
- check_depth_buffer_required          (no rendered-depth buffer => fail, not skip)
- check_metric_boundaries              (inclusive thresholds, one step either side)
- check_rubric_is_gt_named_only        (nothing outside GT-named files imports it)
- check_spec_constants_match_doc       (module thresholds parse equal to the spec table)
- check_ssim_reference                 (closed-form Wang SSIM; no library on this box)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.viz import rv_contract as rvc  # noqa: E402
from python.viz import rv_rubric_gt as rub  # noqa: E402

W, H = 160, 120
EGO_FOOTPRINT = (4.5, 1.9)


# --------------------------------------------------------------------------
# Tiny analytic world (ego frame: x forward, y left, z up; camera frame is
# OpenCV: x right, y down, z forward; depth is z-depth, metres).
# --------------------------------------------------------------------------
VEH_MIN = np.array([12.0, -0.95, 0.0])
VEH_MAX = np.array([16.5, 0.95, 1.5])
VEH_CENTER = (VEH_MIN + VEH_MAX) / 2.0
WALL_X = 60.0
ROAD_HALF_WIDTH = 5.0


def cam_pose(pos, pitch_deg: float) -> np.ndarray:
    """Camera->ego 4x4 for a camera at ``pos`` looking +x, pitched down."""
    th = np.deg2rad(pitch_deg)
    z = np.array([np.cos(th), 0.0, -np.sin(th)])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, pos
    return t


def intrinsics(f: float) -> np.ndarray:
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rays(k: np.ndarray, t_ego_cam: np.ndarray):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    dc = np.stack([(u + 0.5 - k[0, 2]) / k[0, 0], (v + 0.5 - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)
    d = dc @ t_ego_cam[:3, :3].T
    return t_ego_cam[:3, 3], d


def _cast(o: np.ndarray, d: np.ndarray):
    """Nearest hit along o + t d for each ray: (t, which) with which 0 ground, 1 box, 2 wall."""
    inf = np.inf
    down = d[..., 2] < -1e-9
    t_ground = np.where(down, -o[2] / np.where(down, d[..., 2], 1.0), inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (VEH_MIN - o) / d
        t2 = (VEH_MAX - o) / d
    tn = np.minimum(t1, t2)
    tf = np.maximum(t1, t2)
    t_in = np.max(np.nan_to_num(tn, nan=-inf), axis=-1)
    t_out = np.min(np.nan_to_num(tf, nan=inf), axis=-1)
    hit_box = (t_in <= t_out) & (t_out > 0) & (t_in > 0)
    t_box = np.where(hit_box, t_in, inf)
    fwd = d[..., 0] > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = np.where(fwd, (WALL_X - o[0]) / np.where(fwd, d[..., 0], 1.0), inf)
    pw = o + d * np.where(np.isfinite(t_wall), t_wall, 0.0)[..., None]
    t_wall = np.where((pw[..., 2] >= 0) & (pw[..., 2] <= 12.0), t_wall, inf)
    ts = np.stack([t_ground, t_box, t_wall], axis=-1)
    return np.min(ts, axis=-1), np.argmin(ts, axis=-1)


def raycast(k: np.ndarray, t_ego_cam: np.ndarray):
    """Analytic GT: depth (NaN = sky), rgb, road/vehicle masks. Ground + box + wall."""
    o, d = _rays(k, t_ego_cam)
    t, which = _cast(o, d)
    hit = np.isfinite(t)
    depth = np.where(hit, t, np.nan)  # direction z_cam == 1, so t is z-depth
    p = o + d * np.where(hit, t, 0.0)[..., None]
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    ground = hit & (which == 0)
    road = ground & (np.abs(p[..., 1]) < ROAD_HALF_WIDTH)
    grass = ground & ~road
    stripe = (np.floor(p[..., 0] / 2.0).astype(int) % 2).astype(np.float64)
    rgb[road] = np.stack([85 + 25 * stripe, 85 + 25 * stripe, 90 + 25 * stripe], axis=-1)[road]
    rgb[grass] = np.array([60.0, 120.0, 55.0])
    veh = hit & (which == 1)
    shade = np.where(p[..., 2] > VEH_MAX[2] - 1e-6, 1.0, np.where(p[..., 0] < VEH_MIN[0] + 1e-6, 0.8, 0.6))
    rgb[veh] = (np.array([200.0, 40.0, 40.0]) * shade[..., None])[veh]
    wall = hit & (which == 2)
    rgb[wall] = np.array([110.0, 130.0, 190.0])
    rgb[~hit] = np.array([170.0, 200.0, 235.0])
    return depth, np.clip(rgb, 0, 255).astype(np.uint8), road, veh


def unproject(depth: np.ndarray, k: np.ndarray, t_ego_cam: np.ndarray):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    z = depth.astype(np.float64)
    pc = np.stack([(u + 0.5 - k[0, 2]) / k[0, 0] * z, (v + 0.5 - k[1, 2]) / k[1, 1] * z, z], axis=-1)
    return pc @ t_ego_cam[:3, :3].T + t_ego_cam[:3, 3]


FOG = np.array([170, 200, 235], dtype=np.uint8)  # background / fog fill = sky colour


def splat_render(points_ego, colors, k, t_ego_cam, footprints_m):
    """Tiny C-style depth-billboard render with a z-buffer (test-only reference)."""
    inv = np.linalg.inv(t_ego_cam)
    pc = points_ego @ inv[:3, :3].T + inv[:3, 3]
    z = pc[:, 2]
    ok = z > 0.3
    pc, z, colors, fp = pc[ok], z[ok], colors[ok], footprints_m[ok]
    u = pc[:, 0] / z * k[0, 0] + k[0, 2]
    v = pc[:, 1] / z * k[1, 1] + k[1, 2]
    size = np.maximum(1, np.ceil(fp * k[0, 0] / z).astype(int))
    zbuf = np.full((H, W), np.inf)
    u0 = np.floor(u - size / 2.0 + 0.5).astype(int)
    v0 = np.floor(v - size / 2.0 + 0.5).astype(int)
    kmax = int(size.max()) if size.size else 1
    for dy in range(kmax):
        for dx in range(kmax):
            m = (size > dy) & (size > dx)
            uu, vv = u0[m] + dx, v0[m] + dy
            ins = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            np.minimum.at(zbuf, (vv[ins], uu[ins]), z[m][ins])
    rgb = np.tile(FOG, (H, W, 1))
    for dy in range(kmax):
        for dx in range(kmax):
            m = (size > dy) & (size > dx)
            uu, vv = u0[m] + dx, v0[m] + dy
            ins = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv, zz, cc = uu[ins], vv[ins], z[m][ins], colors[m][ins]
            win = zz <= zbuf[vv, uu]
            rgb[vv[win], uu[win]] = cc[win]
    covered = np.isfinite(zbuf)
    depth = np.where(covered, zbuf, 0.0).astype(np.float32)
    return rgb, depth, covered.astype(np.float32)


SRC_K = intrinsics(45.0)                      # wide rig camera, ~122 deg hfov
SRC_T = cam_pose([0.0, 0.0, 1.5], 15.0)
CHASE_K = intrinsics(70.0)                    # chase / held-out camera behind and above the rig
CHASE_T = cam_pose([-1.5, 0.0, 2.2], 8.0)


def build_scene():
    """Source rig camera + chase camera GT, and the C-from-GT render."""
    sd, srgb, _, _ = raycast(SRC_K, SRC_T)
    cd, crgb, croad, cveh = raycast(CHASE_K, CHASE_T)
    valid = np.isfinite(sd)
    pts = unproject(np.where(valid, sd, 1.0), SRC_K, SRC_T)[valid]
    cols = srgb[valid]
    fp = (sd[valid] / SRC_K[0, 0]) * 1.5
    rgb, depth, opacity = splat_render(pts, cols, CHASE_K, CHASE_T, fp)
    # unobserved = chase pixels whose world point the source camera cannot see:
    # outside its frustum, or occluded (exact analytic ray from the source camera).
    pw = unproject(np.where(np.isfinite(cd), cd, 1.0), CHASE_K, CHASE_T)
    inv = np.linalg.inv(SRC_T)
    ps = pw @ inv[:3, :3].T + inv[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        us = ps[..., 0] / ps[..., 2] * SRC_K[0, 0] + SRC_K[0, 2]
        vs = ps[..., 1] / ps[..., 2] * SRC_K[1, 1] + SRC_K[1, 2]
    inside = (ps[..., 2] > 0) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    t_first, _ = _cast(SRC_T[:3, 3], pw - SRC_T[:3, 3])  # p = o + 1 * (p - o)
    seen = inside & (t_first >= 1.0 - 1e-6)
    unobserved = np.isfinite(cd) & ~seen
    gt = rub.GtFrame(
        depth=cd.astype(np.float32),
        rgb=crgb,
        intrinsics=CHASE_K,
        t_ego_cam=CHASE_T,
        road_mask=croad,
        vehicle_mask=cveh,
        unobserved_mask=unobserved,
        vehicle_center_ego=VEH_CENTER.copy(),
    )
    return gt, rgb, depth, opacity, (pts, cols, fp)


def make_out(rgb, depth, *, seq=1, ts=1.0):
    return rvc.RvOutputSpec(
        schema_version=rvc.RV_SCHEMA_VERSION,
        seq=seq,
        timestamp_s=ts,
        pose_4x4=CHASE_T.astype(np.float32),
        rgb=rgb,
        depth=depth.astype(np.float32),
    )


def good_harness(n=5):
    mesh = []
    render = []
    for i in range(n):
        p = np.eye(4)
        p[0, 3] = 1.5 * i
        mesh.append(p)
        q = p.copy()
        q[1, 3] = 0.1  # 10 cm lateral, well inside the mesh footprint
        render.append(q)
    return rub.Harness(render_ego_poses=np.stack(render), mesh_ego_poses=np.stack(mesh), ego_footprint_m=EGO_FOOTPRINT)


def _items_failed(res):
    return sorted(k for k, v in res.items.items() if not v)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_useful_view_metrics() -> None:
    """Spec Step 8 table: AbsRel<=0.20, delta<1.25>=0.70, coverage>=50%, PSNR>=16, SSIM>=0.45."""
    gt, rgb, depth, opacity, _ = build_scene()
    view = rub.RenderedView(out=make_out(rgb, depth), opacity=opacity)
    res = rub.score_metrics(view, gt)
    assert res.passed, f"C-from-GT canned clip must pass the metrics: {res.failures} {res.values}"
    v = res.values
    assert v["abs_rel"] <= 0.20 and v["delta_125"] >= 0.70 and v["coverage"] >= 0.50, v
    assert v["psnr_db"] >= 16.0 and v["ssim"] >= 0.45, v
    # regression pins on this deterministic fixture (my implementation's values, not spec numbers)
    assert abs(v["ssim"] - 0.8506) < 0.005 and abs(v["psnr_db"] - 17.64) < 0.05 and abs(v["abs_rel"] - 0.0229) < 0.002, v
    # deterministic: same inputs -> identical values
    assert rub.score_metrics(view, gt).values == v

    # each metric must be able to fail on its own
    # AbsRel only: constant 25% over-estimate keeps coverage; delta stays 1.25 boundary -> use 1.3
    bad = rub.score_metrics(rub.RenderedView(out=make_out(rgb, depth * 1.3), opacity=opacity), gt)
    assert not bad.passed and not bad.items["abs_rel"] and not bad.items["delta_125"], bad.items
    assert bad.items["coverage"] and bad.items["psnr_db"], bad.items
    # coverage only: blank every other column of the render (depth + opacity), RGB untouched
    d2, o2 = depth.copy(), opacity.copy()
    d2[:, 1::2] = 0.0
    o2[:, 1::2] = 0.0
    sparse = rub.score_metrics(rub.RenderedView(out=make_out(rgb, d2), opacity=o2), gt)
    assert not sparse.items["coverage"] and sparse.items["abs_rel"] and sparse.items["delta_125"], sparse.items
    assert not sparse.passed
    # PSNR / SSIM only: perfect geometry, destroyed colours
    wrong_rgb = np.roll(np.flipud(rgb), 7, axis=1)
    bad_rgb = rub.score_metrics(rub.RenderedView(out=make_out(wrong_rgb, depth), opacity=opacity), gt)
    assert not bad_rgb.items["psnr_db"] or not bad_rgb.items["ssim"], bad_rgb.values
    assert bad_rgb.items["abs_rel"] and bad_rgb.items["coverage"] and not bad_rgb.passed
    # a depth buffer that is perfect but only on pixels outside 0.5-80 m GT range is not credit
    print("PASS check_useful_view_metrics", {k: round(float(x), 4) for k, x in v.items()})


def check_useful_view_checklist() -> None:
    """All six spec checklist items must be yes on the same clip; each can fail alone."""
    gt, rgb, depth, opacity, (pts, cols, fp) = build_scene()
    view = rub.RenderedView(out=make_out(rgb, depth), opacity=opacity)
    harness = good_harness()
    res = rub.score_checklist(view, gt, harness, repo_root=ROOT)
    assert res.passed, f"checklist must be all-yes on the canned clip: {_items_failed(res)} {res.failures}"
    assert sorted(res.items) == sorted(rub.CHECKLIST_ITEMS), sorted(res.items)
    assert len(rub.CHECKLIST_ITEMS) == 6

    # 1. road not a surface: keep only every 9th road pixel
    d, o = depth.copy(), opacity.copy()
    spray = gt.road_mask & ((np.arange(W)[None, :] % 3 != 0) | (np.arange(H)[:, None] % 3 != 0))
    d[spray], o[spray] = 0.0, 0.0
    r = rub.score_checklist(rub.RenderedView(out=make_out(rgb, d), opacity=o), gt, harness, repo_root=ROOT)
    assert _items_failed(r) == ["road_surface"], _items_failed(r)

    # 2. ego pose: one tick jumps 5.0 m laterally (no overlap with the mesh any more)
    jump = good_harness()
    rp = jump.render_ego_poses.copy()
    rp[3, 1, 3] += 5.0
    r = rub.score_checklist(view, gt, rub.Harness(rp, jump.mesh_ego_poses, EGO_FOOTPRINT), repo_root=ROOT)
    assert _items_failed(r) == ["ego_pose_lines_up"], _items_failed(r)
    # a 4.9 m lateral offset is below the 5 m jump gate but the footprints no longer overlap
    rp2 = jump.render_ego_poses.copy()
    rp2[:, 1, 3] = 4.9
    r = rub.score_checklist(view, gt, rub.Harness(rp2, jump.mesh_ego_poses, EGO_FOOTPRINT), repo_root=ROOT)
    assert _items_failed(r) == ["ego_pose_lines_up"], _items_failed(r)

    # 3a. vehicle in the wrong place: shift the vehicle's GT centre 3.5 m (> 3 m)
    far = rub.GtFrame(**{**gt.__dict__, "vehicle_center_ego": gt.vehicle_center_ego + np.array([0.0, 3.5, 0.0])})
    r = rub.score_checklist(view, far, harness, repo_root=ROOT)
    assert _items_failed(r) == ["vehicle_placed_once"], (_items_failed(r), r.values)
    # 3a'. the 3.0 m gate is inclusive and exact: put the GT centre 2.95 / 3.05 m from the rendered one
    est = np.median(rub._unproject_px(depth.astype(np.float64), gt, gt.vehicle_mask & (opacity >= 0.1))[:, :2], axis=0)
    for off_m, want_ok in ((2.95, True), (3.05, False)):
        c = np.array([est[0] + off_m, est[1], gt.vehicle_center_ego[2]])
        g3 = rub.GtFrame(**{**gt.__dict__, "vehicle_center_ego": c})
        r = rub.score_checklist(view, g3, harness, repo_root=ROOT)
        assert ("vehicle_placed_once" not in _items_failed(r)) is want_ok, (off_m, r.values)
    # 3a''. robust centre: 20% of the vehicle's pixels far behind it must not drag the estimate out
    d, vv = depth.copy(), np.argwhere(gt.vehicle_mask & (opacity >= 0.1))
    for (yy_, xx_) in vv[: len(vv) // 5]:
        d[yy_, xx_] += 30.0
    r = rub.score_checklist(rub.RenderedView(out=make_out(rgb, d), opacity=opacity), gt, harness, repo_root=ROOT)
    assert "vehicle_placed_once" not in _items_failed(r), r.values
    # 1b. road is a surface only if its depth is right too: coverage kept, road depth 1.5x too far
    d = depth.copy()
    d[gt.road_mask] *= 1.5
    r = rub.score_checklist(rub.RenderedView(out=make_out(rgb, d), opacity=opacity), gt, harness, repo_root=ROOT)
    assert "road_surface" in _items_failed(r), r.values
    # 3b. drawn twice: a second copy of the vehicle splats 8 m to the side
    vm_src = (pts[:, 0] >= VEH_MIN[0] - 0.01) & (pts[:, 0] <= VEH_MAX[0] + 0.01) & (np.abs(pts[:, 1]) <= 0.96) & (pts[:, 2] > 0.01)
    ghost_pts = pts[vm_src] + np.array([0.0, 4.0, 0.0])
    all_pts = np.concatenate([pts, ghost_pts])
    all_cols = np.concatenate([cols, cols[vm_src]])
    all_fp = np.concatenate([fp, fp[vm_src]])
    grgb, gdepth, gop = splat_render(all_pts, all_cols, CHASE_K, CHASE_T, all_fp)
    r = rub.score_checklist(rub.RenderedView(out=make_out(grgb, gdepth), opacity=gop), gt, harness, repo_root=ROOT)
    assert "vehicle_placed_once" in _items_failed(r), (_items_failed(r), r.values)

    # 4. unobserved painted as a fake wall: opaque surface at 5 m over the unobserved pixels
    d, o = depth.copy(), opacity.copy()
    d[gt.unobserved_mask], o[gt.unobserved_mask] = 5.0, 1.0
    r = rub.score_checklist(rub.RenderedView(out=make_out(rgb, d), opacity=o), gt, harness, repo_root=ROOT)
    assert "unobserved_not_wall" in _items_failed(r), _items_failed(r)
    # hatch / fog (RGB painted, opacity 0) is allowed
    assert gt.unobserved_mask.sum() > 0
    h_rgb = rgb.copy()
    h_rgb[gt.unobserved_mask] = (200, 200, 200)
    r = rub.score_checklist(rub.RenderedView(out=make_out(h_rgb, depth), opacity=opacity), gt, harness, repo_root=ROOT)
    assert "unobserved_not_wall" not in _items_failed(r), _items_failed(r)

    # 4b. the opaque fraction gate is <= 1% of the (eroded) unobserved pixels
    eroded = rub._erode(gt.unobserved_mask, rub.UNOBSERVED_EDGE_ERODE_PX)
    idx = np.argwhere(eroded)
    n_ok = int(0.01 * len(idx))
    for n_paint, want_ok in ((n_ok, True), (n_ok + 1, False)):
        d, o = depth.copy(), opacity.copy()
        d[eroded], o[eroded] = 0.0, 0.0
        for (yy_, xx_) in idx[:n_paint]:
            d[yy_, xx_], o[yy_, xx_] = 5.0, 1.0
        r = rub.score_checklist(rub.RenderedView(out=make_out(rgb, d), opacity=o), gt, harness, repo_root=ROOT)
        assert ("unobserved_not_wall" not in _items_failed(r)) is want_ok, (n_paint, len(idx), r.values)

    # 5. GT on a non-GT-named path
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "python" / "viz").mkdir(parents=True)
        (root / "python" / "viz" / "live_renderer.py").write_text("from python.viz import rv_rubric_gt\n", encoding="utf-8")
        r = rub.score_checklist(view, gt, harness, repo_root=root)
    assert _items_failed(r) == ["no_sim_gt_on_live_path"], _items_failed(r)

    # 5b. a live input validator that accepts GT depth keys fails the item
    orig_validate = rvc.validate_rv_input
    rvc.validate_rv_input = lambda spec: None
    try:
        r = rub.score_checklist(view, gt, harness, repo_root=ROOT)
    finally:
        rvc.validate_rv_input = orig_validate
    assert _items_failed(r) == ["no_sim_gt_on_live_path"], _items_failed(r)

    # 6. stale indicator: a packer that never sets the stale flag
    r = rub.score_checklist(view, gt, harness, repo_root=ROOT, pack=lambda out, age_ms: rvc.pack_output_slot(out, age_ms=0.0))
    assert _items_failed(r) == ["stale_indicator"], _items_failed(r)
    print("PASS check_useful_view_checklist", sorted(res.items))


def check_depth_buffer_required() -> None:
    """A render with no rendered-depth buffer fails AbsRel and delta; it does not skip or raise."""
    gt, rgb, depth, opacity, _ = build_scene()
    out = make_out(rgb, depth)
    nodepth = rvc.RvOutputSpec(**{**out.__dict__, "depth": None})
    res = rub.score_metrics(rub.RenderedView(out=nodepth, opacity=opacity), gt)
    assert not res.passed and not res.items["abs_rel"] and not res.items["delta_125"], res.items
    assert res.values["abs_rel"] is None and res.values["delta_125"] is None
    assert any("depth buffer" in f for f in res.failures), res.failures
    # wrong shape / dtype depth is also "no usable buffer"
    for bad in (depth[:-1], depth.astype(np.float64)):
        o2 = rvc.RvOutputSpec(**{**out.__dict__, "depth": bad})
        r2 = rub.score_metrics(rub.RenderedView(out=o2, opacity=opacity), gt)
        assert not r2.passed and not r2.items["abs_rel"] and not r2.items["delta_125"], r2.items
    # an all-hole depth buffer is a buffer, but scores as failing, not as vacuous success
    r3 = rub.score_metrics(rub.RenderedView(out=make_out(rgb, np.zeros_like(depth)), opacity=opacity), gt)
    assert not r3.passed and not r3.items["abs_rel"] and not r3.items["coverage"], r3.items
    # SHM path: a slot packed through the contract round-trips and scores identically ...
    slot = rvc.pack_output_slot(out, age_ms=10.0)
    via_slot = rub.score_slot_metrics(slot, opacity, gt)
    assert via_slot.passed and via_slot.values == rub.score_metrics(rub.RenderedView(out=out, opacity=opacity), gt).values
    # ... and a slot without RV_FLAG_HAS_DEPTH fails the two metrics instead of raising
    buf = bytearray(slot)
    flags = int.from_bytes(buf[12:16], "little") & ~rvc.RV_FLAG_HAS_DEPTH
    buf[12:16] = flags.to_bytes(4, "little")
    r4 = rub.score_slot_metrics(bytes(buf), opacity, gt)
    assert not r4.passed and not r4.items["abs_rel"] and not r4.items["delta_125"], r4.items
    # a truncated slot (no depth payload) fails the same way
    r5 = rub.score_slot_metrics(bytes(slot[: rvc.SLOT_HEADER_SIZE + W * H * 3]), opacity, gt)
    assert not r5.passed and not r5.items["abs_rel"], r5.items
    print("PASS check_depth_buffer_required")


def _flat_gt(depth_value=10.0):
    d = np.full((H, W), depth_value, dtype=np.float32)
    rgb = np.full((H, W, 3), 100, dtype=np.uint8)
    z = np.zeros((H, W), dtype=bool)
    return rub.GtFrame(
        depth=d, rgb=rgb, intrinsics=CHASE_K, t_ego_cam=CHASE_T, road_mask=z, vehicle_mask=z.copy(),
        unobserved_mask=z.copy(), vehicle_center_ego=np.zeros(3),
    )


def check_metric_boundaries() -> None:
    """Inclusive spec thresholds, tested one step either side with exact integer fractions."""
    gt = _flat_gt(10.0)
    rgb = gt.rgb
    ones = np.ones((H, W), dtype=np.float32)

    def run(depth, opacity=ones, rgb_=rgb):
        return rub.score_metrics(rub.RenderedView(out=make_out(rgb_, depth), opacity=opacity), gt)

    # AbsRel: constant scale s gives AbsRel == |s-1|
    assert run(np.full((H, W), 11.9, np.float32)).items["abs_rel"]
    assert not run(np.full((H, W), 12.1, np.float32)).items["abs_rel"]
    assert run(np.full((H, W), 8.1, np.float32)).items["abs_rel"]
    assert not run(np.full((H, W), 7.9, np.float32)).items["abs_rel"]
    # delta<1.25: exactly 70% of pixels good (ratio 1.0), 30% bad (ratio 2.0)
    n = H * W
    k = int(round(0.70 * n))
    d = np.full(n, 20.0, np.float32)
    d[:k] = 10.0
    assert run(d.reshape(H, W)).items["delta_125"], "70.0% must pass (>= 0.70)"
    d[k - 1] = 20.0
    assert not run(d.reshape(H, W)).items["delta_125"], "69.99% must fail"
    # delta strictly < 1.25: ratio exactly 1.25 does not count as good (float32-exact values)
    ex = np.full((H, W), 12.5, np.float32)
    assert run(ex).values["delta_125"] == 0.0
    # coverage: exactly 50% of valid pixels covered passes, one fewer fails
    half = n // 2
    o = np.zeros(n, np.float32)
    o[:half] = 1.0
    dd = np.where(o > 0, 10.0, 0.0).astype(np.float32)
    assert run(dd.reshape(H, W), o.reshape(H, W)).items["coverage"], "50.0% must pass"
    o[half - 1] = 0.0
    dd = np.where(o > 0, 10.0, 0.0).astype(np.float32)
    assert not run(dd.reshape(H, W), o.reshape(H, W)).items["coverage"]
    # opacity threshold is >= 0.1
    assert run(np.full((H, W), 10.0, np.float32), ones * 0.1).items["coverage"]
    assert not run(np.full((H, W), 10.0, np.float32), ones * 0.0999).items["coverage"]
    # GT range: pixels with GT depth outside 0.5-80 m (AbsRel) / 0.5-40 m (coverage) are not scored
    far = rub.GtFrame(**{**gt.__dict__, "depth": np.full((H, W), 90.0, np.float32)})
    r = rub.score_metrics(rub.RenderedView(out=make_out(rgb, np.full((H, W), 90.0, np.float32)), opacity=ones), far)
    assert not r.items["abs_rel"] and not r.items["coverage"], "no valid GT pixels must fail, not pass vacuously"
    mid = rub.GtFrame(**{**gt.__dict__, "depth": np.full((H, W), 60.0, np.float32)})
    r = rub.score_metrics(rub.RenderedView(out=make_out(rgb, np.full((H, W), 60.0, np.float32)), opacity=ones), mid)
    assert r.items["abs_rel"] and not r.items["coverage"], (r.items, "60 m counts for AbsRel only")
    # PSNR: identical -> inf passes; MSE just under / over 16 dB
    assert run(np.full((H, W), 10.0, np.float32)).values["psnr_db"] == float("inf")
    mse16 = 255.0 ** 2 / 10 ** 1.6
    amp_ok = int(np.floor(np.sqrt(mse16)))  # uniform error of this many levels stays >= 16 dB
    for amp, want in ((amp_ok, True), (amp_ok + 2, False)):
        shifted = np.clip(rgb.astype(int) + amp, 0, 255).astype(np.uint8)
        assert run(np.full((H, W), 10.0, np.float32), rgb_=shifted).items["psnr_db"] is want, amp
    # SSIM of identical images is 1.0 (and in [-1, 1] always)
    assert abs(run(np.full((H, W), 10.0, np.float32)).values["ssim"] - 1.0) < 1e-12
    # AbsRel is inclusive: exactly 0.20 passes (single scored pixel so the mean is exact)
    one = np.zeros((H, W), np.float32)
    one[5, 5] = 1.0
    d_one = np.where(one > 0, 12.0, 0.0).astype(np.float32)
    r = run(d_one, one)
    assert r.values["abs_rel"] == 0.2 and r.items["abs_rel"], r.values
    d_one = np.where(one > 0, 12.01, 0.0).astype(np.float32)
    assert not run(d_one, one).items["abs_rel"]
    # GT range edges, inclusive: AbsRel 0.5-80 m, coverage 0.5-40 m
    for gv, absrel_scored, cov_scored in ((0.5, True, True), (0.49, False, False), (40.0, True, True),
                                          (40.5, True, False), (80.0, True, False), (80.5, False, False)):
        g2 = rub.GtFrame(**{**gt.__dict__, "depth": np.full((H, W), gv, np.float32)})
        r = rub.score_metrics(rub.RenderedView(out=make_out(rgb, np.full((H, W), gv, np.float32)), opacity=ones), g2)
        assert (r.values["abs_rel"] is not None) is absrel_scored, (gv, r.values)
        assert (r.values["coverage"] is not None) is cov_scored, (gv, r.values)
    # SSIM: perfect depth, same-mean images whose contrast is scaled; 0.2 -> ~0.42 fails, 0.25 -> ~0.50 passes
    yy, xx = np.mgrid[0:H, 0:W]
    base = 128 + 60 * np.sin(0.7 * xx) * np.cos(0.5 * yy)
    g_rgb = np.round(np.stack([base] * 3, -1)).astype(np.uint8)
    tex = rub.GtFrame(**{**gt.__dict__, "rgb": g_rgb})
    flat10 = np.full((H, W), 10.0, np.float32)
    for alpha, want, approx in ((0.2, False, 0.423), (0.25, True, 0.503)):
        r_rgb = np.round(np.clip(128 + (g_rgb.astype(np.float64) - 128) * alpha, 0, 255)).astype(np.uint8)
        r = rub.score_metrics(rub.RenderedView(out=make_out(r_rgb, flat10), opacity=ones), tex)
        assert r.items["ssim"] is want and r.items["psnr_db"], (alpha, r.values)
        assert abs(r.values["ssim"] - approx) < 0.003, (alpha, r.values["ssim"])
    # morphology helpers must not wrap around the image border
    m = np.zeros((6, 6), dtype=bool)
    m[0, 0] = True
    dil = rub._dilate(m, 1)
    assert dil[:2, :2].all() and dil.sum() == 4, "dilation wrapped around the border"
    full = np.ones((6, 6), dtype=bool)
    assert rub._erode(full, 1).all(), "erosion treated the border as background"
    print("PASS check_metric_boundaries")


def check_rubric_is_gt_named_only() -> None:
    """The rubric module and this test are GT-named; no non-GT file under python/ imports the rubric."""
    assert "gt" in Path(rub.__file__).stem.split("_"), rub.__file__
    assert "gt" in Path(__file__).stem.split("_"), __file__
    assert rub.find_non_gt_importers(ROOT) == [], rub.find_non_gt_importers(ROOT)
    # supervisor / safety-path modules must not import it either (covered by the scan, pinned here)
    for rel in ("python/run_vision.py",):
        assert "rv_rubric_gt" not in (ROOT / rel).read_text(encoding="utf-8")
    # the live input contract still rejects GT depth keys
    out_keys = ("depth_gt", "gt_depth", "sim_depth", "aux_depth")
    rgb = np.zeros((8, 8, 3), np.uint8)
    cam = rvc.RvCameraInput("main", rgb, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32))
    for key in out_keys:
        spec = rvc.RvInputSpec(rvc.RV_SCHEMA_VERSION, 0, 0.0, np.eye(4, dtype=np.float32), {"main": cam},
                               {key: np.zeros((8, 8), np.float32)})
        try:
            rvc.validate_rv_input(spec)
        except rvc.RvSchemaError:
            continue
        raise AssertionError(f"live input accepted GT key {key!r}")
    print("PASS check_rubric_is_gt_named_only")


# Constants whose values are the published Step 8 table / checklist numbers,
# plus the Redesigner proposal pins. Each name must appear in the proposal.
_SPEC_CONSTANTS: tuple[str, ...] = (
    "ABS_REL_MAX",
    "DELTA_125_MIN",
    "COVERAGE_MIN",
    "COVERAGE_OPACITY_MIN",
    "PSNR_MIN_DB",
    "SSIM_MIN",
    "GT_DEPTH_RANGE_M",
    "GT_COVERAGE_RANGE_M",
    "VEHICLE_CENTER_ERR_MAX_M",
    "STALE_AGE_MS",
    "SSIM_WINDOW",
    "UNOBSERVED_OPAQUE_MAX_FRAC",
    "GHOST_MIN_FRAC_OF_VEHICLE",
    "GHOST_HEIGHT_MIN_M",
    "GHOST_DEPTH_TOL_M",
    "GHOST_EDGE_DILATE_PX",
    "GHOST_OPEN_PX",
    "UNOBSERVED_EDGE_ERODE_PX",
    "ROAD_MIN_COVERAGE",
    "ROAD_ABS_REL_MAX",
)


def _spec_sections() -> tuple[str, str]:
    """Published Step 8 rubric (table + checklist) and the Redesigner proposal."""
    text = (ROOT / "docs/specs/reality_view_3dgs.md").read_text(encoding="utf-8")
    _, found, rest = text.partition("## Step 8 quality rubric (proposed, unmeasured)")
    assert found, "missing Step 8 quality rubric section"
    published, found, proposal = rest.partition("spec proposal (Redesigner, not Michael's)")
    assert found, "missing Redesigner step 8(1) spec proposal"
    proposal = proposal.split("\n## ", 1)[0]
    return published, proposal


def _table_row(section: str, prefix: str) -> list[str]:
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and prefix in cells[0]:
            return cells
    raise AssertionError(f"missing table row starting with {prefix!r}")


def check_spec_constants_match_doc() -> None:
    """Module thresholds equal the published spec table; every constant is named in the proposal."""
    published, proposal = _spec_sections()

    absrel = _table_row(published, "AbsRel")
    assert "0.5–80 m" in absrel[0], absrel
    assert "0.20" in absrel[1], absrel
    assert rub.ABS_REL_MAX == 0.20

    delta = _table_row(published, "δ < 1.25")
    assert "0.70" in delta[1], delta
    assert rub.DELTA_125_MIN == 0.70

    cov = _table_row(published, "Coverage")
    assert "0.5–40 m" in cov[0], cov
    assert "0.1" in cov[0], cov  # opacity ≥ 0.1
    assert "50%" in cov[1], cov
    assert rub.COVERAGE_MIN == 0.50
    assert rub.COVERAGE_OPACITY_MIN == 0.1

    psnr = _table_row(published, "Held-out view PSNR")
    assert "16 dB" in psnr[1] and "0.45" in psnr[1], psnr
    assert rub.PSNR_MIN_DB == 16.0
    assert rub.SSIM_MIN == 0.45

    rate = _table_row(published, "Rate / age")
    assert "200 ms" in rate[1], rate
    assert rub.STALE_AGE_MS == 200.0

    assert "≤ **3 m**" in published
    assert rub.VEHICLE_CENTER_ERR_MAX_M == 3.0
    assert rub.GT_DEPTH_RANGE_M == (0.5, 80.0)
    assert rub.GT_COVERAGE_RANGE_M == (0.5, 40.0)
    assert rub.SSIM_WINDOW == 7
    assert rub.ROAD_MIN_COVERAGE == rub.COVERAGE_MIN == 0.50
    assert rub.ROAD_ABS_REL_MAX == rub.ABS_REL_MAX == 0.20
    assert rub.UNOBSERVED_OPAQUE_MAX_FRAC == 0.01
    assert rub.GHOST_MIN_FRAC_OF_VEHICLE == 0.25
    assert rub.GHOST_HEIGHT_MIN_M == 0.5
    assert rub.GHOST_DEPTH_TOL_M == 1.0
    assert rub.GHOST_EDGE_DILATE_PX == 2
    assert rub.GHOST_OPEN_PX == 1
    assert rub.UNOBSERVED_EDGE_ERODE_PX == 2

    # proposal repeats the published numbers unchanged and names every constant
    for needle in (
        "covered",
        "opacity ≥ 0.1",
        "Zero scored pixels",
        "Inclusive",
        "Strict `< 1.25`",
        "> 200 ms",
        "rv_contract.pack_output_slot",
        "≥ 50%",
        "≤ 0.20",
        "footprint overlap",
        "Median",
        "≤ 3 m",
        "0.5 m",
        "max(1 m, 20%",
        "2 px dilation",
        "1 px morphological opening",
        "25%",
        "1%",
        "2 px-eroded",
        "opacity=None",
        "OpenCV",
        "Wang",
        "7×7",
        "K1 = 0.01",
        "K2 = 0.03",
        "L = 255",
    ):
        assert needle in proposal, needle
    for name in _SPEC_CONSTANTS:
        assert hasattr(rub, name), name
        assert name in proposal, name
    print("PASS check_spec_constants_match_doc")


def check_ssim_reference() -> None:
    """Closed-form Wang SSIM checks. No reference library was present on this box.

    Searched 2026-11-25: scikit-image, pytorch-msssim, piq, torchmetrics, and
    kornia were all missing from /workspace/venv-pr87b-ci-torch (the suite
    interpreter), /workspace/swarm/swarm-3/venv-bev, and the system Python.
    OpenCV 5.0.0 is installed but has no SSIM entry point (no cv2.quality).
    There is therefore no library cross-check; values below are analytic.

    Formula (Wang et al.): uniform 7×7 window, valid region, no Gaussian,
    K1=0.01, K2=0.03, L=255, population moments (E[x^2]−μ^2), per-channel
    then averaged. C1=(K1 L)^2 = 6.5025, C2=(K2 L)^2 = 58.5225 (float64
    of those expressions: 6.502500000000001 and 58.522499999999994).
    """
    # no library to import — if one appears later this still must not depend on it
    import importlib
    for name in ("skimage", "pytorch_msssim", "piq", "torchmetrics", "kornia"):
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        # A library showed up; we still do not call it (not a dep). Record that.
        break

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    assert rub.SSIM_WINDOW == 7

    def rgb(fill) -> np.ndarray:
        return np.full((16, 16, 3), fill, dtype=np.uint8)

    # identical images -> 1.0 (already implied by the formula when x=y)
    assert rub.ssim(rgb(80), rgb(80)) == 1.0

    # constant vs constant reduces to (2ab + C1) / (a^2 + b^2 + C1); variances are 0
    # provenance: analytic reduction of Wang SSIM, any window size, any image size >= 7
    a, b = 100, 150
    expect = (2 * a * b + c1) / (a * a + b * b + c1)  # 0.923092310530793
    assert abs(expect - 0.923092310530793) < 1e-15
    assert abs(rub.ssim(rgb(a), rgb(b)) - 0.923092310530793) < 1e-12

    # near the 0.45 gate: constants 255 vs 61
    # (2*255*61 + C1) / (255^2 + 61^2 + C1) = 0.45258719855324536
    expect = (2 * 255 * 61 + c1) / (255 * 255 + 61 * 61 + c1)
    assert abs(expect - 0.45258719855324536) < 1e-15
    assert abs(rub.ssim(rgb(255), rgb(61)) - 0.45258719855324536) < 1e-12
    assert 0.45 < 0.45258719855324536 < 0.46

    # hand-computed 7×7 window stats (single window = the whole image).
    # X is 0,1,...,48 in row-major order; Y is 24 everywhere.
    # sum(X)=1176, μx=1176/49=24; μy=24.
    # sum(X^2)=38024, E[X^2]=38024/49=776, σx^2=776-24^2=200.
    # σy^2=0, σxy=0 (Y constant, μx=μy).
    # SSIM = C2 / (σx^2 + C2) = 58.5225 / 258.5225 = 0.2263729462619308
    x = np.arange(49, dtype=np.uint8).reshape(7, 7)
    y = np.full((7, 7), 24, dtype=np.uint8)
    xr = np.stack([x, x, x], axis=-1)
    yr = np.stack([y, y, y], axis=-1)
    expect = c2 / (200.0 + c2)
    assert abs(expect - 0.2263729462619308) < 1e-15
    assert abs(rub.ssim(xr, yr) - 0.2263729462619308) < 1e-12
    print("PASS check_ssim_reference", {
        "const_100_vs_150": 0.923092310530793,
        "const_255_vs_61_near_0.45": 0.45258719855324536,
        "ramp0_48_vs_24": 0.2263729462619308,
    })


def main() -> None:
    check_useful_view_metrics()
    check_useful_view_checklist()
    check_depth_buffer_required()
    check_metric_boundaries()
    check_rubric_is_gt_named_only()
    check_spec_constants_match_doc()
    check_ssim_reference()
    print("test_rv_rubric_gt: ok")


if __name__ == "__main__":
    main()
