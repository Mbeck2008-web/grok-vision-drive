#!/usr/bin/env python3
"""Sparse occupancy net: camera-only, sparse queries, and it learns.

* torch projection == numpy ``project`` (same pixel, same frustum mask).
* ``forward(images, pts)``: no depth / semantic / box argument.
* ``predict`` is coarse-to-fine over 3 levels: exact query counts per level
  (all hot, top cold, middle cold, capped); a cold level stops the descent.
* overfit: on a tiny synthetic two-camera scene (built through shards, the
  recovery-mix sampler, and ``occ_train.train``) the loss falls by more than
  60% and the query IoU ends above 0.85 and 0.25 over its start; the
  trained net's coarse-to-fine ``predict`` scores observed-cell IoU > 0.6.
* ``sample_queries``: occupied queries sit in occupied cells, free queries
  never do, coarse ``any`` targets match the label cells.
"""
# gvd-ci: requires torch
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from python.data.occ_shards import ShardWriter  # noqa: E402
from python.perception.occ_geometry import DEFAULT_GRID, project, rig_cams  # noqa: E402
from python.perception.occ_sparse import LEVELS, SparseOccNet, _capped_hot, _column_ground_lvl, visible_mask  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_train import _class_balance, _unoccupied_points, prep_image, sample_queries, train  # noqa: E402
from synthetic_occ import SEM_CLASSES, render  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
G = DEFAULT_GRID

# f=4 uphill: replace the max_below proxy (Skeptic pr237v9 (b)).
# Tight floors are the measured v9 values; b91b957 fails the building
# column counts (184/208 at +10% noise-free). These are strictly stronger
# than the old 88/138/44/85/66 caps, not a numeric widening of them.
F4_UPHILL_TRUTH = {
    (0.10, 0.0):  dict(road_ok=880, road_n=880, bld_ok=208, bld_n=208, wall_min=176, wall_eq=True),
    (0.15, 0.0):  dict(road_ok=640, road_n=664, bld_ok=128, bld_n=136, wall_min=120, wall_eq=True),
    (0.10, 0.01): dict(road_ok=868, road_n=880, bld_ok=197, bld_n=208, wall_min=170, wall_eq=False),
    (0.15, 0.01): dict(road_ok=637, road_n=664, bld_ok=120, bld_n=136, wall_min=115, wall_eq=False),
    (0.10, 0.02): dict(road_ok=798, road_n=880, bld_ok=188, bld_n=208, wall_min=169, wall_eq=False),
    (0.15, 0.02): dict(road_ok=602, road_n=664, bld_ok=107, bld_n=136, wall_min=115, wall_eq=False),
}


def _assert_f4_uphill_truth(name, hot, ph, lab, idx, cap, cm, want) -> None:
    """Per-column band vs true ground, walls kept, leftover-only below pool."""
    lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
    ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
    iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
    ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
    col = ix0 * (int(iy0.max()) + 1) + iy0
    gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)[col]
    in_bld = (np.abs(hot[:, 0]) > 8) & (np.abs(hot[:, 0]) < 14) & (hot[:, 1] > 0) & (hot[:, 1] < 45)
    tg: dict[int, int] = {}
    is_bld: dict[int, bool] = {}
    for c, l, lb, b in zip(col, lvl, lab, in_bld):
        c = int(c)
        if lb == "ground":
            tg[c] = int(l)
        if b:
            is_bld[c] = True
    gl_c = {int(c): int(g) for c, g in zip(col, gl)}
    road = [c for c in tg if not is_bld.get(c, False)]
    bld = [c for c in tg if is_bld.get(c, False)]
    road_ok = sum(gl_c[c] == tg[c] for c in road)
    bld_ok = sum(gl_c[c] == tg[c] for c in bld)
    assert len(road) == want["road_n"] and len(bld) == want["bld_n"], (
        name, "column totals", len(road), len(bld), want)
    assert road_ok >= want["road_ok"], (name, "road band", road_ok, want["road_ok"], len(road))
    assert bld_ok >= want["bld_ok"], (name, "building band", bld_ok, want["bld_ok"], len(bld))
    n_wall_tot = int((lab == "wall").sum())
    n_wall_kept = int((lab[idx] == "wall").sum())
    assert n_wall_kept >= want["wall_min"], (name, "walls kept", n_wall_kept, want["wall_min"])
    if want["wall_eq"]:
        assert n_wall_kept == n_wall_tot, (name, "all walls must be kept", n_wall_kept, n_wall_tot)
    wall_below = int(((lab == "wall") & (lvl < gl)).sum())
    assert wall_below == 0, (name, "true wall in below pool", wall_below)
    kept_band = int((lvl[idx] == gl[idx]).sum())
    kept_struct = int((lab[idx] == "wall").sum())
    kept_below_pool = int((lvl[idx] < gl[idx]).sum())
    leftover = cap - kept_band - kept_struct
    assert kept_below_pool <= leftover, (
        name, "below pool exceeds leftover", kept_below_pool, leftover,
        kept_band, kept_struct, cap)
    print(f"capped_hot {name} truth: road {road_ok}/{len(road)} "
          f"building {bld_ok}/{len(bld)} walls {n_wall_kept}/{n_wall_tot} "
          f"below-pool {kept_below_pool} leftover {leftover} OK")


def check_projection() -> None:
    cams = [c.scaled(64) for c in rig_cams(CFG)]
    net = SparseOccNet(cams)
    rng = np.random.default_rng(0)
    pts = rng.uniform(G.lo, G.hi, size=(4000, 3))
    tp = torch.from_numpy(pts).double()
    net = net.double()
    any_seen = np.zeros(len(pts), bool)
    for k, cam in enumerate(cams):
        uv, z, ok = project(cam, pts)
        u, v, tok = (t.numpy() for t in net.project(k, tp))
        assert ok.sum() > 50, (cam.cam_id, ok.sum())
        assert np.array_equal(tok, ok), (cam.cam_id, (tok != ok).sum())
        assert np.allclose(u[ok], uv[ok, 0], atol=1e-6) and np.allclose(v[ok], uv[ok, 1], atol=1e-6)
        any_seen |= ok
    feats = [torch.zeros((1, 64, c.height // 8, c.width // 8), dtype=torch.float64) for c in cams]
    seen = net.query(feats, tp.unsqueeze(0))["seen"][0].numpy() > 0
    assert np.array_equal(seen, any_seen)
    print("projection: torch frustum == numpy frustum on all 8 cams OK")


def check_camera_only() -> None:
    sig = inspect.signature(SparseOccNet.forward)
    assert list(sig.parameters) == ["self", "images", "pts", "cam_w"], list(sig.parameters)
    sig = inspect.signature(SparseOccNet.predict)
    names = [n for n in sig.parameters if n != "self"]
    assert not any(w in n for n in names for w in ("depth", "sem", "box", "label", "gt")), names
    print("camera-only: forward(images, pts, cam_w), predict has no GT argument OK")


def check_sparse_predict() -> None:
    cams = [c.scaled(64) for c in rig_cams(CFG)]
    torch.manual_seed(0)
    net = SparseOccNet(cams, decode_unseen=False).eval()  # frustum-only top stage: exact visible counts (unseen: test_occ_unseen)
    imgs = [torch.rand(1, 3, c.height, c.width) for c in cams]
    top = G.coarse(LEVELS[0])
    vis = int(visible_mask(cams, top.centers_of_keys(np.arange(top.dense_cells))).sum())
    assert 0 < vis < top.dense_cells

    def bias(any0: float, any1: float) -> None:
        with torch.no_grad():
            net.head[-1].weight.zero_()
            net.head[-1].bias.zero_()
            net.head[-1].bias[1] = any0
            net.head[-1].bias[2] = any1

    bias(10.0, 10.0)  # every cell hot at both levels: worst case, uncapped
    U = 10 ** 9  # uncapped top stage for the exact-count checks; the capped sweep is checked below
    hot = net.predict(imgs, max_points=10 ** 9, max_coarse=U)
    assert hot.n_coarse == vis + vis * 8, (hot.n_coarse, vis)
    assert hot.n_fine == vis * 64, (hot.n_fine, vis)
    bias(-10.0, 10.0)  # top level cold: nothing below it is queried
    cold = net.predict(imgs, max_coarse=U)
    assert cold.n_coarse == vis and cold.n_fine == 0 and cold.keys.size == 0, (cold.n_coarse, cold.n_fine)
    assert cold.queried_ratio() < 0.02, cold.queried_ratio()
    bias(10.0, -10.0)  # middle level cold: no fine query
    mid = net.predict(imgs, max_coarse=U)
    assert mid.n_fine == 0 and mid.n_coarse == vis + min(vis, 65536 // 8) * 8, (mid.n_coarse, vis)
    bias(10.0, 10.0)
    capped = net.predict(imgs, max_points=8192, max_coarse=U)
    assert capped.n_fine <= 8192 and capped.n_coarse <= vis + 8192, (capped.n_coarse, capped.n_fine)
    assert capped.queried_ratio() < 0.1, capped.queried_ratio()
    # absolute cap on the top stage: <= cap per frame, every visible coarse cell revisited
    net.reset_temporal()
    bias(-10.0, -10.0)
    cap = 1000
    seen_top: set = set()
    frames = int(np.ceil(vis / cap)) + 1
    for _ in range(frames):
        pr = net.predict(imgs, max_coarse=cap)
        assert pr.stage_queries[0] <= cap, pr.stage_queries
        seen_top.update(map(tuple, np.round(net._coarse_pts, 2).tolist()))
    assert len(seen_top) == vis, (len(seen_top), vis, frames)
    # hot top cells are carried into the next frame first
    net.reset_temporal()
    bias(10.0, -10.0)
    net.predict(imgs, max_coarse=cap)
    first = {tuple(p) for p in np.round(net._coarse_pts, 2).tolist()}
    net.predict(imgs, max_coarse=cap)
    second = {tuple(p) for p in np.round(net._coarse_pts, 2).tolist()}
    assert len(first & second) >= cap // 2, len(first & second)
    net.reset_temporal()
    print(f"top-stage cap: {cap}/frame, all {vis} visible coarse cells swept in {frames} frames, hot cells carried OK")
    print(f"sparse predict: cold {cold.n_queried} pts ({cold.queried_ratio():.4f} of dense), "
          f"capped {capped.n_queried} ({capped.queried_ratio():.4f}), uncapped all-hot {hot.n_queried} OK")


def check_queries() -> None:
    cams = {c.cam_id: c.scaled(96) for c in rig_cams(CFG)}
    d, s, _ = render(cams["main"])
    lab = build_occ_labels({"main": (d, s)}, cams, sem_classes=SEM_CLASSES)
    q = sample_queries(lab.keys, lab.cls, lab.free, np.random.default_rng(0), n_occ=300, n_free=300, n_coarse=100)
    k = G.keys_of(q["pts"].astype(np.float64))
    occ_m, free_m = q["occ"] == 1, (q["occ"] == 0) & (q["occ_w"] == 1)
    assert occ_m.sum() == 300 and free_m.sum() == 300 and (q["occ_w"] == 0).sum() == 100
    assert np.isin(k[occ_m], lab.keys).all(), "occupied query left its cell"
    assert not np.isin(k[free_m], lab.keys).any(), "free query in an occupied cell"
    assert q["any"].shape == (700, len(LEVELS))
    for li, f in enumerate(LEVELS):
        cg = G.coarse(f)
        occ_l = np.unique(cg.keys_of(G.centers_of_keys(lab.keys)))
        assert np.array_equal(q["any"][:, li] > 0.5, np.isin(cg.keys_of(q["pts"].astype(np.float64)), occ_l)), f
        assert (q["any"][occ_m, li] == 1).all() and 0 < q["any"][q["occ_w"] == 0, li].mean() < 1
    # a coarser level is hot wherever a finer one is
    assert np.all(q["any"][:, 0] >= q["any"][:, 1])
    assert (q["cls"][occ_m] >= 0).all() and (q["cls"][~occ_m] == -1).all()
    print("queries: occ in cells, free outside, coarse any == label cells OK")


def check_overfit(tmp: Path) -> None:
    ids = ("main", "wide")
    cams = [c.scaled(64) for c in rig_cams(CFG) if c.cam_id in ids]
    full = {c.cam_id: c for c in rig_cams(CFG)}
    with ShardWriter(tmp / "sh") as w:
        for i, (box_y, rec) in enumerate([(10.0, False), (10.0, False), (10.0, False), (10.0, True)]):
            imgs, planes = {}, {}
            for cid in ids:
                cam = full[cid].scaled(128)
                d, s, bgr = render(cam)
                imgs[cid], planes[cid] = bgr, (d, s)
            lab = build_occ_labels(planes, {c: full[c].scaled(128) for c in ids}, sem_classes=SEM_CLASSES,
                                   rng=np.random.default_rng(i))
            w.write(f"s{i}", imgs, lab.as_arrays(), {"recovery": rec, "t": float(i)})
    torch.manual_seed(0)
    res = train(tmp / "sh", cams, steps=60, batch_size=4, lr=3e-3, device="cpu", seed=0,
                n_occ=256, n_free=256, n_coarse=64)
    h = res["history"]
    first = float(np.mean([x["loss"] for x in h[:5]]))
    last = float(np.mean([x["loss"] for x in h[-5:]]))
    iou0 = float(np.mean([x["iou"] for x in h[:5]]))
    iou1 = float(np.mean([x["iou"] for x in h[-5:]]))
    assert all(x["recovery_frac"] == 0.25 for x in h), {x["recovery_frac"] for x in h}
    print(f"overfit: loss {first:.3f} -> {last:.3f} ({last / first:.2f}x), iou {iou0:.3f} -> {iou1:.3f}, "
          f"params {res['params']}, peak rss {res['peak_rss_mb']:.0f} MB")
    assert last < 0.4 * first, (first, last)
    assert iou1 > 0.85 and iou1 > iou0 + 0.25, (iou0, iou1)
    print("overfit: loss fell > 60%, IoU > 0.85 OK")
    # The trained net's sparse predict on the training scene, cell level.
    net = res["net"].eval()
    net.decode_unseen = False  # camera-only labels here: unseen cells were never supervised (unseen decode: test_occ_unseen)
    imgs = [torch.from_numpy(prep_image(render(full[c.cam_id].scaled(128))[2], c)).unsqueeze(0) for c in cams]
    net.reset_temporal()
    # Capped top stage (2048 of the visible coarse cells). check_sparse_predict
    # needs 7 frames to sweep all 5044 visible coarse cells; 3 frames left the
    # observed-cell IoU platform-dependent (0.515 / 14,022 queries on CI vs
    # 0.722 / 20,517 locally). Run a full sweep, still under the 0.25 dense cap.
    for _ in range(8):
        pred = net.predict(imgs)
    lab = build_occ_labels({c: render(full[c].scaled(128))[:2] for c in ids}, {c: full[c].scaled(128) for c in ids},
                           sem_classes=SEM_CLASSES, stride=1)
    observed = np.union1d(lab.keys, G.keys_of(lab.free.astype(np.float64)))
    seen_pred = np.intersect1d(pred.keys, observed)
    inter = np.intersect1d(seen_pred, lab.keys).size
    union = np.union1d(seen_pred, lab.keys).size
    print(f"predict: {pred.keys.size} cells predicted ({seen_pred.size} in observed cells), label {lab.keys.size}, "
          f"observed-cell IoU {inter / union:.3f}, "
          f"queried {pred.n_queried} = {pred.queried_ratio():.4f} of dense, stored {pred.keys.size / G.dense_cells:.4f}")
    assert inter / union > 0.6, inter / union
    assert pred.queried_ratio() < 0.25, pred.queried_ratio()
    print("predict: observed-cell IoU > 0.6 from sparse queries (< 25% of dense) OK")



def check_capped_hot_spreads() -> None:
    """Later-stage caps must not spend the whole budget on the highest-p region.

    Off-road ground / buildings are slightly easier occupied cells than the
    road; a global top-k by probability then never refines the road.
    """
    # one height band, two xy regions: a large high-p field and a smaller lower-p "road"
    n_hi, n_lo = 2000, 200
    hi = np.stack([np.full(n_hi, -10.0), np.linspace(-20.0, 40.0, n_hi), np.full(n_hi, 0.2)], 1)
    lo = np.stack([np.full(n_lo, 10.0), np.linspace(-20.0, 40.0, n_lo), np.full(n_lo, 0.2)], 1)
    hot = np.concatenate([hi, lo])
    p = np.concatenate([np.full(n_hi, 0.90), np.full(n_lo, 0.80)])
    cap = 256
    idx = _capped_hot(hot, p, cap, z_min=-2.0, cell_m=0.4)
    assert idx.size == cap, idx.size
    picked = hot[idx]
    n_road = int((picked[:, 0] > 0).sum())
    n_field = int((picked[:, 0] < 0).sum())
    assert n_field > 0 and n_road > 0, (n_field, n_road)
    assert n_road >= 16, n_road  # a fair xy share, not a couple of leftovers
    # height split is kept: a thinner upper band still gets slots when the ground is huge
    ground = np.stack([np.linspace(-20.0, 20.0, 4000), np.zeros(4000), np.full(4000, 0.2)], 1)
    posts = np.stack([np.linspace(-5.0, 5.0, 80), np.full(80, 15.0), np.full(80, 3.0)], 1)
    hot2 = np.concatenate([ground, posts])
    p2 = np.concatenate([np.full(4000, 0.95), np.full(80, 0.70)])
    idx2 = _capped_hot(hot2, p2, 256, z_min=-2.0, cell_m=0.4)
    n_up = int((hot2[idx2, 2] > 1.0).sum())
    assert n_up > 0, n_up
    print(f"capped_hot: cap={cap} field/road {n_field}/{n_road}; ground+posts upper kept {n_up} OK")


def check_capped_hot_below_ground() -> None:
    """Skeptic probe: a hot below-ground fill must not starve above-ground structure."""
    f = 2
    cm = G.voxel_m * f
    xs = G.x_min + cm * (np.arange(64) + 0.5)
    ys = G.y_min + cm * (np.arange(80) + 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    gz = G.z_min + cm * 2.5  # ground layer index 2 at the 2x stage

    def layer(z, mask=None):
        m = np.ones_like(X, bool) if mask is None else mask
        return np.stack([X[m], Y[m], np.full(int(m.sum()), z)], 1)

    bld = ((np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)) | ((X > -4) & (X < 4) & (Y > 35) & (Y < 40))
    rng = np.random.default_rng(0)
    for p_bg, p_st in ((0.9, 0.7), (0.6, 0.7), (0.0, 0.7)):
        parts = [(layer(gz), 0.95, "ground")]
        if p_bg > 0:
            parts += [(layer(G.z_min + cm * 0.5), p_bg, "below"), (layer(G.z_min + cm * 1.5), p_bg, "below")]
        parts += [(layer(gz + cm * k, bld), p_st, "struct") for k in (1, 2, 3, 4)]
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) + rng.normal(0, 0.01, len(p[0])) for p in parts])
        lab = np.concatenate([[p[2]] * len(p[0]) for p in parts])
        idx = _capped_hot(hot, ph, 2048, G.z_min, cm)
        n_struct = int((lab[idx] == "struct").sum())
        assert idx.size <= 2048, idx.size
        assert n_struct > 0, (p_bg, n_struct, dict(zip(*np.unique(lab[idx], return_counts=True))))
        print(f"capped_hot below-ground p={p_bg}: structure kept {n_struct} / cap 2048 OK")

    # Ground coverage 0.8 / 0.95 plus one full below-ground layer must not
    # flip the ground band to the more populous below layer (count-based
    # picker). Structure must keep its slots.
    for gfrac in (0.80, 0.95):
        for p_bg in (0.6, 0.9):
            gm = rng.random(X.size) < gfrac
            parts = [(layer(gz, gm), 0.95, "ground"),
                     (layer(G.z_min + cm * 0.5), p_bg, "below")]
            parts += [(layer(gz + cm * k, bld), 0.7, "struct") for k in (1, 2, 3, 4)]
            hot = np.concatenate([p[0] for p in parts])
            ph = np.concatenate([np.full(len(p[0]), p[1]) + rng.normal(0, 0.01, len(p[0])) for p in parts])
            lab = np.concatenate([[p[2]] * len(p[0]) for p in parts])
            idx = _capped_hot(hot, ph, 2048, G.z_min, cm)
            n_struct = int((lab[idx] == "struct").sum())
            kept = dict(zip(*np.unique(lab[idx], return_counts=True)))
            assert n_struct > 0, (gfrac, p_bg, n_struct, kept)
            print(f"capped_hot ground {gfrac:.2f} + 1 below p={p_bg}: structure kept {n_struct} {kept} OK")



def check_capped_hot_follows_ground() -> None:
    """Ground band follows the local ground per xy column at production LEVELS.

    Production only caps at LEVELS (4, 2): cell 1.6 m and 0.8 m. OccRuntime
    max_points=16384 with 8 children gives cap=2048 at both stages. A 3%
    downhill and a -0.7 m ground offset must keep the (far) road; a below
    layer adjacent to ground must not take band slots (exact band counts).
    """
    for f, cap in ((4, 2048), (4, 512), (2, 2048)):
        cm = G.voxel_m * f
        nx = int(round((G.x_max - G.x_min) / cm))
        ny = int(round((G.y_max - G.y_min) / cm))
        xs = G.x_min + cm * (np.arange(nx) + 0.5)
        ys = G.y_min + cm * (np.arange(ny) + 0.5)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        bld = ((np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)) | (
            (X > -4) & (X < 4) & (Y > 35) & (Y < 40))

        def zc(z):
            z = np.asarray(z, dtype=np.float64)
            return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

        def run(name: str, gz, ground_mask=None) -> None:
            gz = np.broadcast_to(np.asarray(gz, dtype=np.float64), X.shape)
            gm = np.ones(X.size, bool) if ground_mask is None else ground_mask
            parts = [(np.stack([X[gm], Y[gm], zc(gz[gm])], 1), 0.95,
                      np.where(Y[gm] > 20.0, "groundFAR", "groundNEAR"))]
            # Adjacent below-ground layer only under ground columns, so a
            # hole cannot promote below to the band. Those below cells
            # must not take band slots (N6).
            parts.append((np.stack([X[gm], Y[gm], zc(gz[gm] - cm)], 1), 0.9,
                          np.full(int(gm.sum()), "below")))
            for k in (1, 2, 3):
                parts.append((np.stack([X[bld], Y[bld], zc(gz[bld] + k * cm)], 1),
                              0.7, np.full(int(bld.sum()), "struct")))
            hot = np.concatenate([p[0] for p in parts])
            # Fixed p, no noise: a 0.05 gap must be enough for the column
            # max to pick ground over the adjacent below layer (N6).
            ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
            lab = np.concatenate([p[2] for p in parts])
            keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
            hot, ph, lab = hot[keep], ph[keep], lab[keep]
            idx = _capped_hot(hot, ph, cap, G.z_min, cm)
            assert idx.size <= cap, (name, f, idx.size)
            picked = lab[idx]
            n_g = int(np.isin(picked, ("groundFAR", "groundNEAR")).sum())
            n_far = int((picked == "groundFAR").sum())
            n_st = int((picked == "struct").sum())
            n_bel = int((picked == "below").sum())
            n_g_tot = int(np.isin(lab, ("groundFAR", "groundNEAR")).sum())
            n_st_tot = int((lab == "struct").sum())
            n_bel_tot = int((lab == "below").sum())
            # Exact pool sizes: band is cap//2, all ground; structure next;
            # adjacent below only gets leftover. Unused budget returns to ground.
            n_low = min(n_g_tot, cap // 2) if n_g_tot else 0
            n_up = min(n_st_tot, cap - n_low)
            n_dn = min(n_bel_tot, cap - n_low - n_up)
            leftover = cap - (n_low + n_up + n_dn)
            n_low = min(n_g_tot, n_low + leftover)
            leftover = cap - (n_low + n_up + n_dn)
            n_up = min(n_st_tot, n_up + leftover)
            leftover = cap - (n_low + n_up + n_dn)
            n_dn = min(n_bel_tot, n_dn + leftover)
            kept = dict(zip(*np.unique(picked, return_counts=True)))
            assert n_g == n_low, (name, f, n_g, n_low, kept)
            assert n_st == n_up, (name, f, n_st, n_up, kept)
            assert n_bel == n_dn, (name, f, n_bel, n_dn, kept)
            assert n_far > 0, (name, f, "far road starved", kept)
            assert n_st > 0, (name, f, "structure starved", kept)
            print(f"capped_hot f={f} cell={cm}m cap={cap} {name}: "
                  f"ground {n_g} (far {n_far}) struct {n_st} below {n_bel} OK")

        run("flat z=0.2", 0.2)
        run("offset z=-0.7", 0.2 - 0.7)
        run("3% downhill", 0.2 - 0.03 * np.maximum(Y, 0.0))
        # Holed ground, n_ground < cap//2, structure fills the rest: picked
        # below must be 0. A widened band (N6) would pull the adjacent below
        # layer into the band and spend those slots on below.
        # Structure stays on the building footprint (not only on ground
        # columns): a roof must not become the band in a hole.
        if cap <= 1024 or f == 2:
            gm = (np.arange(X.size) % 8) == 0  # 12.5% of columns
            run("holed 12.5% (N6)", 0.2, ground_mask=gm)



def check_capped_hot_surface_fill() -> None:
    """A below-ground fill must not steal the band on equal or slightly higher p.

    Production 2x stage, cap 2048: equal p, both saturated at 1.0, fill 0.97
    vs ground 0.95, and cold/empty ground with a hot fill. Structure stays
    at the geometric-cap share (1024). Downhill and offset stay in
    check_capped_hot_follows_ground.
    """
    f, cap = 2, 2048
    cm = G.voxel_m * f
    nx = int(round((G.x_max - G.x_min) / cm))
    ny = int(round((G.y_max - G.y_min) / cm))
    xs = G.x_min + cm * (np.arange(nx) + 0.5)
    ys = G.y_min + cm * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)

    def zc(z):
        z = np.asarray(z, dtype=np.float64)
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

    def layer(z, mask=None):
        m = np.ones(X.size, bool) if mask is None else mask
        return np.stack([X[m], Y[m], np.full(int(m.sum()), zc(z))], 1)

    def run(name, parts, expect_ground=True):
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
        hot, ph, lab = hot[keep], ph[keep], lab[keep]
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        kept = dict(zip(*np.unique(lab[idx], return_counts=True)))
        n_st = int((lab[idx] == "struct").sum())
        n_g = int((lab[idx] == "ground").sum())
        assert n_st == 1024, (name, n_st, kept)
        if expect_ground:
            assert n_g == 1024, (name, n_g, kept)
        else:
            assert n_g == 0, (name, n_g, kept)
        print(f"capped_hot surface {name}: {kept} OK")

    struct = [(layer(0.2 + k * cm, bld), 0.7, "struct") for k in (1, 2, 3)]
    for pb, pg, name in ((0.95, 0.95, "equal p=0.95"),
                         (1.0, 1.0, "saturated p=1.0"),
                         (0.97, 0.95, "fill 0.97 vs ground 0.95")):
        run(name, [(layer(0.2), pg, "ground"), (layer(0.2 - cm), pb, "below"),
                   (layer(0.2 - 2 * cm), pb, "below")] + struct)
    run("cold ground, hot fill p=0.9",
        [(layer(0.2 - cm), 0.9, "below"), (layer(0.2 - 2 * cm), 0.9, "below")] + struct,
        expect_ground=False)


def check_capped_hot_structure_surface() -> None:
    """Saturated structure and buildings with no ground under them.

    The band is chosen from the bottom of the column, so a wall or car
    roof at the same p as the ground cannot take the band (Skeptic
    pr237v5 roof_probe A/C/E). Structure must keep at least the cfb2a42
    share; the below-ground pool must be empty.
    """
    f, cap = 2, 2048
    cm = G.voxel_m * f
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]

    def zc(z):
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

    def layer(z, m):
        return np.stack([X[m], Y[m], np.full(int(m.sum()), zc(z))], 1)

    all_xy = np.ones_like(X, bool)
    road = np.abs(X) < 4
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    cars = road & (np.floor(Y / 6) % 2 == 0) & (Y > 5)
    nob = ~bld

    def run(name, parts, min_struct, allow_below=False):
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        kept = {str(k): int(v) for k, v in zip(*np.unique(lab[idx], return_counts=True))}
        n_st = int(kept.get("wall", 0) + kept.get("car", 0))
        n_bel = int(kept.get("below", 0))
        assert n_st >= min_struct, (name, n_st, min_struct, kept)
        if not allow_below:
            assert n_bel == 0, (name, n_bel, kept)
        print(f"capped_hot structure {name}: {kept} OK")

    walls = [(layer(0.2 + k * cm, bld), 1.0, "wall") for k in (1, 2, 3)]
    car_l = [(layer(0.2 + k * cm, cars), 1.0, "car") for k in (1, 2)]
    # A: all saturated, no fill. cfb2a42: wall 874 + car 150 = 1024, below 0.
    run("A all saturated 1.0, no fill",
        [(layer(0.2, all_xy), 1.0, "ground")] + walls + car_l, 1024)
    # B: saturated + fill 1.0. cfb2a42 kept structure and below=0.
    run("B all saturated 1.0 + below fill 1.0",
        [(layer(0.2, all_xy), 1.0, "ground"), (layer(0.2 - cm, all_xy), 1.0, "below")]
        + walls + car_l, 1024)
    # C: ground .95, walls .92, cars .93, fill .9 (all inside a 0.05 margin
    # of some neighbour). cfb2a42: car 520 + wall 504 = 1024, below 0.
    run("C ground .95, walls .92, cars .93, fill .9",
        [(layer(0.2, all_xy), 0.95, "ground"), (layer(0.2 - cm, all_xy), 0.9, "below")]
        + [(layer(0.2 + k * cm, bld), 0.92, "wall") for k in (1, 2, 3)]
        + [(layer(0.2 + k * cm, cars), 0.93, "car") for k in (1, 2)], 1024)
    # E: no ground under buildings. cfb2a42: wall 1024, below 0.
    run("E no ground under buildings, walls .8, fill .9",
        [(layer(0.2, nob), 0.95, "ground"), (layer(0.2 - cm, nob), 0.9, "below")]
        + [(layer(0.2 + k * cm, bld), 0.8, "wall") for k in (1, 2, 3)], 1024)
    # F: E + cars .97 on the road. cfb2a42: car 520 + wall 504 = 1024, below 0.
    run("F no ground under buildings + cars .97",
        [(layer(0.2, nob), 0.95, "ground"), (layer(0.2 - cm, nob), 0.9, "below")]
        + [(layer(0.2 + k * cm, bld), 0.8, "wall") for k in (1, 2, 3)]
        + [(layer(0.2 + k * cm, cars), 0.97, "car") for k in (1, 2)], 1024)


def check_capped_hot_grade_and_onelayer() -> None:
    """Steep uphill, one-layer objects, a wide no-ground building, and a weak hole fill.

    The band follows a local ground plane (up to a 15% grade over the 12 m
    window) so far ground on a hill stays the band and structure keeps the
    cfb2a42 / 6a530c5 share. A one-layer car or guardrail that sits on
    real ground goes to the above-ground pool, not the band. A >=20 m
    building with no ground under it stays structure (kills P4, radius
    4 m). A hole whose only seed is a weak below-ground fill inherits
    the surrounding ground (kills P8, dropping raw >= env).
    """
    def zc(z, cm):
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

    def grid_xy(cm):
        xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
        ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
        return [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]

    def lay(X, Y, zfun, m, cm):
        z = zfun(X[m], Y[m])
        return np.stack([X[m], Y[m], zc(z, cm)], 1)

    def run_parts(name, parts, cap, cm, min_counts, max_below=0, truth=None):
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
        hot, ph, lab = hot[keep], ph[keep], lab[keep]
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        kept = {str(k): int(v) for k, v in zip(*np.unique(lab[idx], return_counts=True))}
        for k, nmin in min_counts.items():
            assert int(kept.get(k, 0)) >= nmin, (name, k, kept, nmin)
        if truth is not None:
            _assert_f4_uphill_truth(name, hot, ph, lab, idx, cap, cm, truth)
        else:
            n_bel = int(kept.get("below", 0))
            assert n_bel <= max_below, (name, n_bel, max_below, kept)
        print(f"capped_hot {name}: {kept} OK")
        return kept

    # --- uphill + one-layer (Skeptic attack6 / attack6_base) ---
    for f, cap, want in (
        (2, 2048, {"up10_wall": 900, "up15_wall": 650, "car": 150, "rail": 64,
                    "up10_below": 128, "up15_below": 140}),
        (4, 512, {"up10_wall": 176, "up15_wall": 120, "car": 40, "rail": 32}),
    ):
        cm = G.voxel_m * f
        X, Y = grid_xy(cm)
        ALL = np.ones_like(X, bool)
        road = np.abs(X) < 4
        bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
        rail = np.abs(np.abs(X) - 5) < cm * 0.6
        cars = road & (np.floor(Y / 6) % 2 == 0) & (Y > 5)
        flat = lambda x, y: np.full(x.shape, 0.2)
        # 1-layer cars on saturated road + 3-layer buildings (cfb2a42 / 6a530c5).
        run_parts(
            f"f={f} 1-layer cars 1.0",
            [(lay(X, Y, flat, ALL, cm), 1.0, "ground")]
            + [(lay(X, Y, lambda x, y, k=k: 0.2 + k * cm + 0 * x, bld, cm), 1.0, "wall") for k in (1, 2, 3)]
            + [(lay(X, Y, lambda x, y: 0.2 + cm + 0 * x, cars, cm), 1.0, "car")],
            cap, cm, {"car": want["car"]},
        )
        # 1-layer guardrails + fill (cfb2a42 kept 64 / 32 rails).
        run_parts(
            f"f={f} 1-layer guardrails 1.0",
            [(lay(X, Y, flat, ALL, cm), 1.0, "ground"),
             (lay(X, Y, lambda x, y: 0.2 - cm + 0 * x, ALL, cm), 1.0, "below")]
            + [(lay(X, Y, lambda x, y, k=k: 0.2 + k * cm + 0 * x, bld, cm), 1.0, "wall") for k in (1, 2, 3)]
            + [(lay(X, Y, lambda x, y: 0.2 + cm + 0 * x, rail, cm), 1.0, "rail")],
            cap, cm, {"rail": want["rail"]},
        )
        for g, wkey, bkey in ((0.10, "up10_wall", "up10_below"), (0.15, "up15_wall", "up15_below")):
            gz = lambda x, y, g=g: 0.2 + g * np.maximum(y, 0)
            extra = ({"truth": F4_UPHILL_TRUTH[(g, 0.0)]} if f == 4
                     else {"max_below": want[bkey]})
            run_parts(
                f"f={f} uphill {g * 100:+.0f}%",
                [(lay(X, Y, gz, ALL, cm), 0.95, "ground"),
                 (lay(X, Y, lambda x, y, g=g: gz(x, y) - cm, ALL, cm), 0.9, "below")]
                + [(lay(X, Y, lambda x, y, k=k, g=g: gz(x, y) + k * cm, bld, cm), 0.95, "wall") for k in (1, 2, 3)],
                cap, cm, {"wall": want[wkey]},
                **extra,
            )

    # --- P4: >=20 m-wide building, no ground under it ---
    f, cap = 2, 2048
    cm = G.voxel_m * f
    X, Y = grid_xy(cm)
    # 20 m+ wide in x (fits in the 51.2 m grid); the centre is ~10 m
    # from the side, so a 12 m window still sees ground and a 4 m
    # window does not.
    bld = (X >= 5.0) & (X < G.x_max) & (Y >= 5.0) & (Y < 45.0)
    width = float(X[bld].max() - X[bld].min()) + cm
    assert width >= 20.0, (width, cm, X[bld].min(), X[bld].max())
    nob = ~bld
    run_parts(
        "P4 20 m+ building, no ground under it",
        [(lay(X, Y, lambda x, y: np.full(x.shape, 0.2), nob, cm), 0.95, "ground"),
         (lay(X, Y, lambda x, y: np.full(x.shape, 0.2 - cm), nob, cm), 0.9, "below")]
        + [(lay(X, Y, lambda x, y, k=k: np.full(x.shape, 0.2 + k * cm), bld, cm), 0.8, "wall") for k in (1, 2, 3)],
        cap, cm, {"wall": 1024},
        max_below=0,
    )

    # --- P8: hole whose only seed is a weak below-ground fill ---
    f, cap = 2, 2048
    cm = G.voxel_m * f
    X, Y = grid_xy(cm)
    ALL = np.ones_like(X, bool)
    hole = (np.abs(X) < 3.0) & (Y > 10.0) & (Y < 22.0)
    ground = ALL & ~hole
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    parts = (
        [(lay(X, Y, lambda x, y: np.full(x.shape, 0.2), ground, cm), 0.95, "ground"),
         (lay(X, Y, lambda x, y: np.full(x.shape, 0.2 - cm), hole, cm), 0.60, "holefill"),
         (lay(X, Y, lambda x, y: np.full(x.shape, 0.2 - cm), ground, cm), 0.90, "below")]
        + [(lay(X, Y, lambda x, y, k=k: np.full(x.shape, 0.2 + k * cm), bld, cm), 0.95, "wall") for k in (1, 2, 3)]
    )
    hot = np.concatenate([p[0] for p in parts])
    ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
    lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
    keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
    hot, ph, lab = hot[keep], ph[keep], lab[keep]
    lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
    ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
    iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
    ix0 = ix - int(ix.min())
    iy0 = iy - int(iy.min())
    col = ix0 * (int(iy0.max()) + 1) + iy0
    gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)
    hole_cols = np.unique(col[lab == "holefill"])
    ground_lvl = int(np.median(lvl[lab == "ground"]))
    fill_lvl = int(np.median(lvl[lab == "holefill"]))
    assert fill_lvl < ground_lvl, (fill_lvl, ground_lvl)
    assert hole_cols.size >= 8, hole_cols.size
    # The hole inherits the surrounding ground; a weak fill must not become
    # the band (P8 would set gl to the fill layer).
    assert np.all(gl[hole_cols] == ground_lvl), (
        "P8 hole fill became the band",
        int((gl[hole_cols] == fill_lvl).sum()),
        int(hole_cols.size),
        fill_lvl,
        ground_lvl,
    )
    print(f"capped_hot P8 weak hole fill: {hole_cols.size} hole cols inherit ground {ground_lvl}, not fill {fill_lvl} OK")




def check_capped_hot_p_margin_and_robust_env() -> None:
    """p-margin ground-at-envelope and a robust 12 m envelope (Skeptic pr237v7).

    has_env_layer must not require an exact p match: a one-layer car at
    p 0.901-0.949 on a road at 0.90 must stay in the above-ground pool
    (6cdfb8d put the band on 640/640 car columns at 0.8 m and 160/160
    at 1.6 m). Existing f=2 / f=4 floors plus the cold-ground wall must
    still hold with p noise at sigma 0.01 and 0.02 (6cdfb8d cold-ground
    wall 124 and 1; v6 held 1022 at sigma 0.01). The 12 m envelope must
    not follow one low outlier seed (6cdfb8d: 1004/5120 columns at
    level 0 on the sigma 0.01 two-fill scene) or jump a layer when
    sparse cars are more than BAND_MARGIN hotter than the road
    (6cdfb8d: 4480/4480 road columns at 0.8 m, 1120/1120 at 1.6 m).
    """
    def car_scene(cell, nx, ny, proad, pcar, every):
        ix, iy = [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")]
        ground_lvl = 4
        car = (ix % every == 0)
        col = np.concatenate([ix * 1000 + iy, (ix * 1000 + iy)[car]])
        lvl = np.concatenate([np.full(ix.size, ground_lvl), np.full(int(car.sum()), ground_lvl + 1)])
        p = np.concatenate([np.full(ix.size, proad), np.full(int(car.sum()), pcar)]).astype(np.float32)
        xy = (np.concatenate([ix, ix[car]]), np.concatenate([iy, iy[car]]))
        band = _column_ground_lvl(col, lvl, p, xy[0], xy[1], cell)
        u = np.unique(col)
        r = band[u]
        n_car = int(car.sum())
        n_road = int((~car).sum())
        at_car = int((r[car] == ground_lvl + 1).sum())
        road_wrong = int((r[~car] != ground_lvl).sum())
        return at_car, n_car, road_wrong, n_road

    # --- Major 1: cars slightly hotter than the road must not take the band ---
    for cell, nx, ny, n_car_exp, n_road_exp in ((0.8, 64, 80, 640, 4480), (1.6, 32, 40, 160, 1120)):
        for d in (0.001, 0.010, 0.030, 0.049):
            at_car, n_car, road_wrong, n_road = car_scene(cell, nx, ny, 0.90, 0.90 + d, 9)
            assert n_car == n_car_exp and n_road == n_road_exp, (cell, d, n_car, n_road)
            assert at_car == 0, (
                f"car p={0.90+d:.3f} took the band at {cell} m",
                at_car, n_car, road_wrong, n_road,
            )
            assert road_wrong == 0, (cell, d, road_wrong, n_road)
            print(f"p-margin car {cell} m road 0.90 car {0.90+d:.3f}: "
                  f"band@car {at_car}/{n_car}, road wrong {road_wrong}/{n_road} OK")

    # --- Major 2: sparse cars hotter than the road by more than 0.05 ---
    for cell, nx, ny, n_car_exp, n_road_exp in ((0.8, 64, 80, 640, 4480), (1.6, 32, 40, 160, 1120)):
        for pcar in (0.95, 0.96):
            at_car, n_car, road_wrong, n_road = car_scene(cell, nx, ny, 0.90, pcar, 9)
            assert n_car == n_car_exp and n_road == n_road_exp, (cell, pcar, n_car, n_road)
            assert road_wrong == 0, (
                f"hot cars p={pcar} moved the road band at {cell} m",
                road_wrong, n_road, at_car, n_car,
            )
            assert at_car == 0, (
                f"hot cars p={pcar} took the band at {cell} m",
                at_car, n_car,
            )
            print(f"robust-env hot car {cell} m road 0.90 car {pcar:.2f}: "
                  f"band@car {at_car}/{n_car}, road wrong {road_wrong}/{n_road} OK")

    # --- Major 2: one low outlier must not drag ~1000 columns (cold_dbg 1004/5120) ---
    f, cm = 2, G.voxel_m * 2
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    zc = lambda z: G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
    ALL = np.ones_like(X, bool)
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)

    def layers_two_fill():
        def L(z, m):
            return np.stack([X[m], Y[m], np.full(int(m.sum()), zc(z))], 1)
        parts = [(L(0.2 - cm, ALL), 0.9), (L(0.2 - 2 * cm, ALL), 0.9)]
        parts += [(L(0.2 + k * cm, bld), 0.7) for k in (1, 2, 3)]
        hot = np.concatenate([p[0] for p in parts])
        return hot, parts

    hot, parts = layers_two_fill()
    rng = np.random.default_rng(1)
    ph = np.clip(
        np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
        + rng.normal(0, 0.01, len(hot)),
        0, 1,
    ).astype(np.float32)
    lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
    ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
    iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
    ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
    col = ix0 * (int(iy0.max()) + 1) + iy0
    gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)[col]
    # first fill layer is one cell per column (5120 columns)
    t = gl[: X.size]
    n0 = int((t == 0).sum())
    assert t.size == 5120, t.size
    assert n0 == 0, ("low outlier dragged columns to level 0", n0, t.size)
    print(f"robust-env two-fill sigma 0.01: {n0}/{t.size} columns at level 0 OK")

    # --- Major 1: p noise on the f=2 / f=4 floors; cold-ground wall near v6 ---
    def run_parts(name, parts, cap, cm, min_counts, max_below=0, sig=0.0, seed=1, truth=None):
        rng = np.random.default_rng(seed)
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float64)
        if sig > 0:
            ph = np.clip(ph + rng.normal(0, sig, len(ph)), 0, 1)
        ph = ph.astype(np.float32)
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
        hot, ph, lab = hot[keep], ph[keep], lab[keep]
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        kept = {str(k): int(v) for k, v in zip(*np.unique(lab[idx], return_counts=True))}
        for k, nmin in min_counts.items():
            assert int(kept.get(k, 0)) >= nmin, (name, k, kept, nmin)
        if truth is not None:
            _assert_f4_uphill_truth(name, hot, ph, lab, idx, cap, cm, truth)
        else:
            n_bel = int(kept.get("below", 0))
            assert n_bel <= max_below, (name, n_bel, max_below, kept)
        print(f"capped_hot {name}: {kept} OK")
        return kept

    def grid_xy(cm):
        xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
        ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
        return [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]

    def zc_cm(z, cm):
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

    def lay(X, Y, zfun, m, cm):
        z = zfun(X[m], Y[m])
        return np.stack([X[m], Y[m], zc_cm(z, cm)], 1)

    # Cars/rails keep the noise-free floors even with p noise. Uphill
    # walls under noise are pinned at seed 1 near the noise-free floors
    # (b91b957 pinned them at about half: f=2 397 / 471, f=4 94 / 106).
    # Cold-ground wall must stay at the cap share (1024 / 256), near
    # v6's 1022, not 6cdfb8d's 124 / 1.
    for f, cap, want in (
        (2, 2048, {"car": 150, "rail": 64, "cold_wall": 1024,
                    0.01: {"up10_wall": 963, "up15_wall": 744, "up10_below": 129, "up15_below": 124},
                    0.02: {"up10_wall": 808, "up15_wall": 693, "up10_below": 130, "up15_below": 149}}),
        (4, 512, {"car": 40, "rail": 32, "cold_wall": 256,
                   0.01: {"up10_wall": 170, "up15_wall": 115},
                   0.02: {"up10_wall": 169, "up15_wall": 115}}),
    ):
        cm = G.voxel_m * f
        X, Y = grid_xy(cm)
        ALL = np.ones_like(X, bool)
        road = np.abs(X) < 4
        bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
        rail = np.abs(np.abs(X) - 5) < cm * 0.6
        cars = road & (np.floor(Y / 6) % 2 == 0) & (Y > 5)
        flat = lambda x, y: np.full(x.shape, 0.2)
        for sig in (0.01, 0.02):
            tag = f"f={f} sig={sig}"
            run_parts(
                f"{tag} 1-layer cars 1.0",
                [(lay(X, Y, flat, ALL, cm), 1.0, "ground")]
                + [(lay(X, Y, lambda x, y, k=k: 0.2 + k * cm + 0 * x, bld, cm), 1.0, "wall") for k in (1, 2, 3)]
                + [(lay(X, Y, lambda x, y: 0.2 + cm + 0 * x, cars, cm), 1.0, "car")],
                cap, cm, {"car": want["car"]}, sig=sig,
            )
            run_parts(
                f"{tag} 1-layer guardrails 1.0",
                [(lay(X, Y, flat, ALL, cm), 1.0, "ground"),
                 (lay(X, Y, lambda x, y: 0.2 - cm + 0 * x, ALL, cm), 1.0, "below")]
                + [(lay(X, Y, lambda x, y, k=k: 0.2 + k * cm + 0 * x, bld, cm), 1.0, "wall") for k in (1, 2, 3)]
                + [(lay(X, Y, lambda x, y: 0.2 + cm + 0 * x, rail, cm), 1.0, "rail")],
                cap, cm, {"rail": want["rail"]}, sig=sig,
            )
            uw = want[sig]
            for g, wkey, bkey in ((0.10, "up10_wall", "up10_below"), (0.15, "up15_wall", "up15_below")):
                gz = lambda x, y, g=g: 0.2 + g * np.maximum(y, 0)
                extra = ({"truth": F4_UPHILL_TRUTH[(g, sig)]} if f == 4
                         else {"max_below": uw[bkey]})
                run_parts(
                    f"{tag} uphill {g * 100:+.0f}%",
                    [(lay(X, Y, gz, ALL, cm), 0.95, "ground"),
                     (lay(X, Y, lambda x, y, g=g: gz(x, y) - cm, ALL, cm), 0.9, "below")]
                    + [(lay(X, Y, lambda x, y, k=k, g=g: gz(x, y) + k * cm, bld, cm), 0.95, "wall") for k in (1, 2, 3)],
                    cap, cm, {"wall": uw[wkey]}, sig=sig, **extra,
                )
            # cold-ground wall must stay near the v6 level (~1022 of 1024 at f=2)
            run_parts(
                f"{tag} cold ground wall",
                [(lay(X, Y, lambda x, y: 0.2 - cm + 0 * x, ALL, cm), 0.9, "below"),
                 (lay(X, Y, lambda x, y: 0.2 - 2 * cm + 0 * x, ALL, cm), 0.9, "below")]
                + [(lay(X, Y, lambda x, y, k=k: 0.2 + k * cm + 0 * x, bld, cm), 0.7, "wall") for k in (1, 2, 3)],
                cap, cm, {"wall": want["cold_wall"]}, max_below=cap, sig=sig,
            )


def check_capped_hot_exposed_ground() -> None:
    """Buried ground is judged against the exposed ground at its level (Skeptic pr237v8).

    v8 decided "real ground at the envelope" from a 0.049 p margin to the
    column max, so on a grade the downhill-lagged 12 m envelope landed on
    the fill and any fill within 0.049 (or a noisy 0.05 gap) took the band.
    Pins: +10% fill .92/.93 keep the noise-free walls; the standard .90
    fill at sigma 0.005 / 0.01 stays near the noise-free per-column
    result (b91b957: 1251 / 1315 of 5120 columns one layer low); noisy
    crest / terrace / kink walls stay near noise-free; a dense lot of hot
    cars (p 1.0 on road .95) keeps the road band. VOTE_P_MIN is pinned at
    both edges (road 0.66 and 0.80 vote, the P8 0.60 fill does not), and
    a low-p grid at the 16384-column cap stays a few MiB (b91b957: 1517).
    """
    import tracemalloc

    def grid_xy(cm):
        xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
        ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
        return [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]

    def zc(z, cm):
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)

    def lay(X, Y, zfun, m, cm):
        return np.stack([X[m], Y[m], zc(zfun(X[m], Y[m]), cm)], 1)

    def noisy(parts, sig, seed):
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float64)
        if sig > 0:
            ph = np.clip(ph + np.random.default_rng(seed).normal(0, sig, len(ph)), 0, 1)
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
        return hot[keep], ph.astype(np.float32)[keep], lab[keep]

    def kept_of(parts, cap, cm, sig=0.0, seed=1):
        hot, ph, lab = noisy(parts, sig, seed)
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        return {str(k): int(v) for k, v in zip(*np.unique(lab[idx], return_counts=True))}

    def scene(X, Y, gz, cm, bld, pf=0.9):
        return ([(lay(X, Y, gz, np.ones_like(X, bool), cm), 0.95, "ground"),
                 (lay(X, Y, lambda x, y: gz(x, y) - cm, np.ones_like(X, bool), cm), pf, "below")]
                + [(lay(X, Y, lambda x, y, k=k: gz(x, y) + k * cm, bld, cm), 0.95, "wall") for k in (1, 2, 3)])

    up10 = lambda x, y: 0.2 + 0.10 * np.maximum(y, 0)
    crest = lambda x, y: 0.2 + 0.08 * np.minimum(np.maximum(y, 0), 25) - 0.08 * np.maximum(y - 25, 0)
    kink15 = lambda x, y: 0.2 + 0.15 * np.maximum(y - 20, 0)
    for f, cap, want in (
        (2, 2048, {"fill_wall": 988, "fill_below": 128,
                   ("up10", 0.005): 1014, ("up10", 0.01): 963, ("crest", 0.01): 927, ("terrace", 0.01): 981}),
        (4, 512, {"fill_wall": 176, "fill_below": 100,
                  ("up10", 0.005): 168, ("up10", 0.01): 170, ("kink15", 0.01): 215}),
    ):
        cm = G.voxel_m * f
        X, Y = grid_xy(cm)
        bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
        terr = lambda x, y, cm=cm: 0.2 + cm * (x >= 0)
        for pf in (0.92, 0.93):
            k = kept_of(scene(X, Y, up10, cm, bld, pf), cap, cm)
            assert k.get("wall", 0) >= want["fill_wall"], (f, pf, k)
            assert k.get("below", 0) <= want["fill_below"], (f, pf, k)
            print(f"exposed-ground f={f} uphill +10% fill {pf}: {k} OK")
        for (name, sig), wmin in [(k, v) for k, v in want.items() if isinstance(k, tuple)]:
            gz = {"up10": up10, "crest": crest, "kink15": kink15, "terrace": terr}[name]
            k = kept_of(scene(X, Y, gz, cm, bld), cap, cm, sig=sig)
            assert k.get("wall", 0) >= wmin, (f, name, sig, k, wmin)
            if f == 2:
                assert k.get("below", 0) <= 129, (f, name, sig, k)
            print(f"exposed-ground f={f} {name} sigma {sig}: {k} OK")

    # per-column: uphill +10% at 0.8 m, standard fill; columns put one layer low
    f, cm = 2, G.voxel_m * 2
    X, Y = grid_xy(cm)
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    col_want = {0.0: (24, 30), 0.005: (28, 41), 0.01: (91, 39), 0.02: (340, 40)}
    for sig in (0.0, 0.005, 0.01, 0.02):
        hot, ph, lab = noisy(scene(X, Y, up10, cm, bld), sig, 1)
        lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
        ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
        iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
        ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
        col = ix0 * (int(iy0.max()) + 1) + iy0
        gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)
        g = lab == "ground"
        low = int((gl[col[g]] == lvl[g] - 1).sum())
        high = int((gl[col[g]] > lvl[g]).sum())
        assert low <= col_want[sig][0] and high <= col_want[sig][1], (sig, low, high, col_want[sig])
        print(f"exposed-ground per-column +10% sigma {sig}: one layer low {low}, high {high} "
              f"of {int(g.sum())} ground columns OK")

    # dense lot of hot cars
    for f, cap in ((2, 2048), (4, 512)):
        cm = G.voxel_m * f
        X, Y = grid_xy(cm)
        ALL = np.ones_like(X, bool)
        lot = (np.abs(X) < 20) & (Y > 5) & (Y < 45)
        cars = lot & ((np.floor((Y - 5) / 2.5) % 5) < 3)
        bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
        flat = lambda x, y: np.full(x.shape, 0.2)
        parts = ([(lay(X, Y, flat, ALL, cm), 0.95, "ground"),
                  (lay(X, Y, lambda x, y: flat(x, y) - cm, ALL, cm), 0.9, "below"),
                  (lay(X, Y, lambda x, y: flat(x, y) + cm, cars, cm), 1.0, "car")]
                 + [(lay(X, Y, lambda x, y, k=k: flat(x, y) + k * cm, bld & ~lot, cm), 0.95, "wall") for k in (1, 2, 3)])
        for sig, gmin in ((0.0, cap // 2), (0.01, 920 if f == 2 else 225)):
            k = kept_of(parts, cap, cm, sig=sig)
            assert k.get("below", 0) == 0 and k.get("ground", 0) >= gmin, (f, sig, k, gmin)
            print(f"exposed-ground dense hot lot (p 1.0 on road .95) f={f} sigma {sig}: {k} OK")

    # VOTE_P_MIN pinned at both edges: a far raised road at p 0.66 / 0.80,
    # more than 12 m from any stronger voter, must vote for its own band
    # (a 0.82 floor would leave it to inherit the near road two layers down).
    cell, nx, ny = 0.8, 40, 80
    ix, iy = [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")]
    near, far = iy < 19, iy >= 35
    m = near | far
    ix, iy, near, far = ix[m], iy[m], near[m], far[m]
    for pr in (0.66, 0.80):
        colv = np.concatenate([ix * 1000 + iy, ix * 1000 + iy])
        lvl = np.concatenate([np.where(far, 6, 4), np.where(far, 5, 3)]).astype(np.int64)
        p = np.concatenate([np.where(far, pr, 0.9), np.full(ix.size, 0.5)]).astype(np.float32)
        band = _column_ground_lvl(colv, lvl, p, np.concatenate([ix, ix]), np.concatenate([iy, iy]), cell)
        bf = band[(ix * 1000 + iy)[far]]
        assert np.all(bf == 6), ("far road did not vote", pr, int((bf != 6).sum()), int(far.sum()))
        print(f"exposed-ground VOTE_P_MIN: far road p {pr} votes its own band {int(far.sum())}/{int(far.sum())} OK")

    # Low-p grid at the 16384-column cap (b91b957 q x g inherit: 1517 MiB).
    peaks = []
    for cell, n in ((0.8, 64), (0.2, 90), (0.2, 128)):
        ix, iy = [a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij")]
        colg = np.concatenate([ix * n + iy] * 2)
        lvl = np.concatenate([np.full(ix.size, 5), np.full(ix.size, 4)]).astype(np.int64)
        strip = ix < max(1, n // 12)
        p = np.concatenate([np.where(strip, 0.9, 0.6), np.full(ix.size, 0.55)]).astype(np.float32)
        tracemalloc.start()
        out = _column_ground_lvl(colg, lvl, p, np.concatenate([ix] * 2), np.concatenate([iy] * 2), cell)
        peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
        tracemalloc.stop()
        assert np.all(out[ix * n + iy] == 5), (cell, n)
    assert max(peaks) <= 16.0, peaks
    print(f"exposed-ground low-p grid growth 4096/8100/16384 columns: peak "
          f"{', '.join(f'{x:.1f}' for x in peaks)} MiB OK")


def check_capped_hot_inherit_and_vote_edges() -> None:
    """Kill inherit/vote-floor mutants X4, X8, X9, X10, X13 (Skeptic pr237v9)."""
    def levels_on(n, cell, layers):
        ix, iy = [a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij")]
        col, lv, pp = [], [], []
        for mask, lvl, p in layers:
            m = mask(ix, iy)
            col.append(ix[m] * n + iy[m])
            lv.append(np.full(int(m.sum()), lvl) if np.isscalar(lvl) else lvl(ix[m], iy[m]))
            pp.append(np.full(int(m.sum()), p))
        col = np.concatenate(col)
        lvl = np.concatenate(lv).astype(np.int64)
        p = np.concatenate(pp).astype(np.float32)
        return _column_ground_lvl(col, lvl, p, col // n, col % n, cell), ix, iy, n

    # X13: P8-shaped hole whose fill is p=0.63, inside (0.60, 0.65].
    # 0.65 leaves the fill silent (hole inherits ground). 0.62 lets it vote.
    f, cap, cm = 2, 2048, G.voxel_m * 2
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    def zc(z):
        return G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
    def lay(z, m):
        return np.stack([X[m], Y[m], np.full(int(m.sum()), zc(z))], 1)
    ALL = np.ones_like(X, bool)
    hole = (np.abs(X) < 3.0) & (Y > 10.0) & (Y < 22.0)
    ground = ALL & ~hole
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    parts = (
        [(lay(0.2, ground), 0.95, "ground"),
         (lay(0.2 - cm, hole), 0.63, "holefill"),
         (lay(0.2 - cm, ground), 0.90, "below")]
        + [(lay(0.2 + k * cm, bld), 0.95, "wall") for k in (1, 2, 3)]
    )
    hot = np.concatenate([p[0] for p in parts])
    ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
    lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
    keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
    hot, ph, lab = hot[keep], ph[keep], lab[keep]
    lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
    ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
    iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
    ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
    col = ix0 * (int(iy0.max()) + 1) + iy0
    gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)
    hole_cols = np.unique(col[lab == "holefill"])
    ground_lvl = int(np.median(lvl[lab == "ground"]))
    fill_lvl = int(np.median(lvl[lab == "holefill"]))
    assert fill_lvl < ground_lvl
    assert hole_cols.size >= 8
    assert np.all(gl[hole_cols] == ground_lvl), (
        "X13: p=0.63 hole fill became the band",
        int((gl[hole_cols] == fill_lvl).sum()), int(hole_cols.size), fill_lvl, ground_lvl,
    )
    print(f"capped_hot X13 vote floor: {hole_cols.size} hole cols inherit ground "
          f"{ground_lvl}, not p=0.63 fill {fill_lvl} OK")
    # Isolated far p=0.63 lower hole (>12 m from p=0.90 ground) must also not vote.
    ixh, iyh = [a.ravel() for a in np.meshgrid(np.arange(40), np.arange(80), indexing="ij")]
    near_h, far_h = iyh < 19, iyh >= 35
    mh = near_h | far_h
    ixh, iyh, near_h, far_h = ixh[mh], iyh[mh], near_h[mh], far_h[mh]
    colh = np.concatenate([ixh * 1000 + iyh, ixh * 1000 + iyh])
    lvlh = np.concatenate([np.where(far_h, 3, 4), np.where(far_h, 2, 3)]).astype(np.int64)
    phh = np.concatenate([np.where(far_h, 0.63, 0.90), np.full(ixh.size, 0.50)]).astype(np.float32)
    bh = _column_ground_lvl(colh, lvlh, phh, np.concatenate([ixh, ixh]), np.concatenate([iyh, iyh]), 0.8)
    assert np.all(bh[(ixh * 1000 + iyh)[far_h]] == 4), "X13: isolated p=0.63 hole voted"

    # X8 / X10: a low-y ground strip, a high-y hole of weak fill more than
    # 12 m away. Leftover inherit must walk in +y (X8 drops that pass) and
    # must not fall back to the hole's own fill seed (X10).
    n, cell = 64, 0.8
    band, ix, iy, n = levels_on(n, cell, [
        (lambda x, y: y <= 40, 4, 0.95),
        (lambda x, y: y > 40, 3, 0.60),
    ])
    hole = iy > 40
    hcols = ix[hole] * n + iy[hole]
    assert np.all(band[hcols] == 4), (
        "X8/X10: high-y hole did not inherit the low-y ground",
        int((band[hcols] == 3).sum()), int(hcols.size),
    )
    print(f"capped_hot X8/X10 +y inherit: {int(hcols.size)} hole cols inherit "
          f"ground 4, not fill 3 OK")

    # X9: leftover inherit of a slightly high plane uses floor(z+0.25), not
    # +0.5, so a 0.6-layer error cannot jump a layer (same bias as the inlier
    # snap). Voters at x<5 step from level 4 (y<32) to 5 (y>=32); leftover
    # x>=40 is >12 m away and has cells at both 4 and 5. At the first 8
    # leftover rows after the step, +0.25 stays on 4 and +0.5 jumps to 5.
    band, ix, iy, n = levels_on(n, cell, [
        (lambda x, y: (x < 5) & (y < 32), 4, 0.95),
        (lambda x, y: (x < 5) & (y >= 32), 5, 0.95),
        (lambda x, y: x >= 40, 4, 0.50),
        (lambda x, y: x >= 40, 5, 0.50),
    ])
    edge = (ix >= 40) & (iy >= 32) & (iy <= 39)
    ecols = ix[edge] * n + iy[edge]
    assert ecols.size == 8 * 24, ecols.size  # 8 y * 24 leftover x
    assert np.all(band[ecols] == 4), (
        "X9: leftover inherit used +0.5 and jumped a layer at the step",
        int((band[ecols] == 5).sum()), int(ecols.size),
    )
    print(f"capped_hot X9 inherit floor(z+0.25): {int(ecols.size)} leftover "
          f"cols at the 4/5 step stay on 4 OK")

    # X4: non-voting 0.60 hole fill at level 4 next to a 0.90 road at level 5
    # over a 0.75 buried fill (with exposed 0.90 ground at level 4 nearby) must
    # not seed ref at level 4 and falsely promote the buried fill.
    band, ix, iy, n = levels_on(n, cell, [
        (lambda x, y: y < 12, 4, 0.90),
        (lambda x, y: (y >= 12) & (x < 32), 5, 0.90),
        (lambda x, y: (y >= 12) & (x < 32), 4, 0.75),
        (lambda x, y: (y >= 12) & (x >= 32), 4, 0.60),
    ])
    sub = (iy >= 14) & (ix < 32) & (ix >= 20)
    rcols = ix[sub] * n + iy[sub]
    assert np.all(band[rcols] == 5), (
        "X4: non-voter seed promoted buried 0.75 fill",
        int((band[rcols] == 4).sum()), int(rcols.size),
    )
    print(f"capped_hot X4 non-voter seed: {int(rcols.size)} road cols stay on 5 OK")


def check_capped_hot_low_p_grade_and_saturated_fill() -> None:
    """Pin Major 2 (low-p road, relative vote), Major 3 (saturated/0.01 fill on grade), Minor 6 (f=4 noisy grade)."""
    def levels_gt(n, cell, layers):
        ix, iy = [a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij")]
        col, lv, pp, gt = [], [], [], np.full(n * n, -999)
        for mask, lvl, p, is_gnd in layers:
            m = mask(ix, iy)
            c = ix[m] * n + iy[m]
            col.append(c)
            lv.append(np.full(int(m.sum()), lvl) if np.isscalar(lvl) else lvl(ix[m], iy[m]))
            pp.append(np.full(int(m.sum()), p) if np.isscalar(p) else p(ix[m], iy[m]))
            if is_gnd:
                gt[c] = lv[-1]
        col = np.concatenate(col)
        lvl = np.concatenate(lv).astype(np.int64)
        p = np.concatenate(pp).astype(np.float32)
        band = _column_ground_lvl(col, lvl, p, col // n, col % n, cell)
        return band[np.arange(n * n)], gt, ix, iy

    # Major 2: corner of strong road (p=0.90), rest p=0.60 road one layer up.
    n, cell = 64, 0.8
    band, gt, ix, iy = levels_gt(n, cell, [
        (lambda x, y: (x < 6) & (y < 6), 4, 0.90, True),
        (lambda x, y: ~((x < 6) & (y < 6)), 5, 0.60, True),
        (lambda x, y: ~((x < 6) & (y < 6)), 4, 0.55, False),
    ])
    dx = np.where(ix < 6, 0, ix - 5)
    dy = np.where(iy < 6, 0, iy - 5)
    far = np.maximum(dx, dy) > 15
    assert int((band[far] == 5).sum()) == int(far.sum()), (int((band[far] == 5).sum()), int(far.sum()))
    print(f"Major 2 corner +1 step p.60: far cols {int(far.sum())}/{int(far.sum())} at own seed OK")

    # Major 2: +10% grade, road 0.95 for x<32 and 0.62 beyond (fill 0.90/0.57).
    band, gt, _, _ = levels_gt(n, cell, [
        (lambda x, y: x >= 0, lambda x, y: 4 + (0.10 * y).astype(int),
         lambda x, y: np.where(x < 32, 0.95, 0.62), True),
        (lambda x, y: x >= 0, lambda x, y: 3 + (0.10 * y).astype(int),
         lambda x, y: np.where(x < 32, 0.90, 0.57), False),
    ])
    assert int((band != gt).sum()) == 0, int((band != gt).sum())
    print("Major 2 +10% grade, half p.62: 0/4096 wrong OK")

    # Major 2: road p=0.63, fill 0.58, one-layer cars p=0.66 every 9th row.
    band, _, _, _ = levels_gt(n, cell, [
        (lambda x, y: x >= 0, 4, 0.63, True),
        (lambda x, y: x >= 0, 3, 0.58, False),
        (lambda x, y: (y % 9) == 0, 5, 0.66, False),
    ])
    assert int((band == 5).sum()) == 0, int((band == 5).sum())
    print("Major 2 road p.63 + cars p.66: no column at the car layer OK")

    # Major 3: fill within 0.01 of ground, or saturated, on a grade.
    def kept_walls(f, g, pg, pf):
        cm = G.voxel_m * f
        cap = 2048 if f == 2 else 512
        xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
        ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
        X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
        zc = lambda z: G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
        bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
        gz = lambda x, y: 0.2 + g * np.maximum(y, 0)
        parts = [
            (np.stack([X, Y, zc(gz(X, Y))], 1), pg, "ground"),
            (np.stack([X, Y, zc(gz(X, Y) - cm)], 1), pf, "below"),
        ] + [
            (np.stack([X[bld], Y[bld], zc(gz(X[bld], Y[bld]) + k * cm)], 1), pg, "wall")
            for k in (1, 2, 3)
        ]
        hot = np.concatenate([p[0] for p in parts])
        ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
        lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
        hot, ph, lab = hot[keep], ph[keep], lab[keep]
        idx = _capped_hot(hot, ph, cap, G.z_min, cm)
        return int((lab[idx] == "wall").sum()), int((lab == "wall").sum())

    w, tot = kept_walls(2, 0.05, 0.95, 0.94)
    assert w >= 1000, (w, tot)
    print(f"Major 3 +5% .95/.94: walls {w}/{tot} OK")

    w, tot = kept_walls(2, 0.10, 1.0, 1.0)
    assert w >= 1000, (w, tot)
    print(f"Major 3 +10% saturated 1.0/1.0: walls {w}/{tot} OK")

    # Minor 6: f=4 noisy uphill grade must not place band on the first wall layer.
    f, cm = 4, G.voxel_m * 4
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    zc = lambda z: G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    for g in (0.10, 0.15):
        gz = lambda x, y, g=g: 0.2 + g * np.maximum(y, 0)
        parts = [
            (np.stack([X, Y, zc(gz(X, Y))], 1), 0.95, "ground"),
            (np.stack([X, Y, zc(gz(X, Y) - cm)], 1), 0.9, "below"),
        ] + [
            (np.stack([X[bld], Y[bld], zc(gz(X[bld], Y[bld]) + k * cm)], 1), 0.95, "wall")
            for k in (1, 2, 3)
        ]
        for sig in (0.01, 0.02):
            rng = np.random.default_rng(1)
            hot = np.concatenate([p[0] for p in parts])
            ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float64)
            ph = np.clip(ph + rng.normal(0, sig, len(ph)), 0, 1).astype(np.float32)
            lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
            keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
            hot, ph, lab = hot[keep], ph[keep], lab[keep]
            lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
            ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
            iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
            ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
            col = ix0 * (int(iy0.max()) + 1) + iy0
            gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)[col]
            in_bld = (np.abs(hot[:, 0]) > 8) & (np.abs(hot[:, 0]) < 14) & (hot[:, 1] > 0) & (hot[:, 1] < 45)
            tg, is_bld = {}, {}
            for c, l, lb, b in zip(col, lvl, lab, in_bld):
                c = int(c)
                if lb == "ground":
                    tg[c] = int(l)
                if b:
                    is_bld[c] = True
            gl_c = {int(c): int(g_) for c, g_ in zip(col, gl)}
            bld_cs = [c for c in tg if is_bld.get(c, False)]
            high = sum(gl_c[c] > tg[c] for c in bld_cs)
            assert high == 0, (g, sig, high, len(bld_cs))
            print(f"Minor 6 f=4 +{int(g*100)}% sigma={sig}: {high}/{len(bld_cs)} building cols high OK")


def check_class_balance() -> None:
    """_class_balance splits evenly across classes; a 99:1 pool must not stay 99:1."""
    rng = np.random.default_rng(0)
    cls = np.concatenate([np.zeros(990, np.int64), np.ones(10, np.int64)])
    idxs = np.arange(cls.size)
    # 20 independent draws of 100: balanced should land near 50/50; off (uniform) ~1
    minority = []
    for i in range(20):
        pick = _class_balance(idxs, cls, np.random.default_rng(i), 100)
        minority.append(int((cls[pick] == 1).sum()))
    assert min(minority) >= 30, minority  # 50/50 with 10 available, resampled
    # single-class pool still returns n indices (and must not ZeroDivisionError)
    one = _class_balance(np.arange(8), np.zeros(8, np.int64), rng, 5)
    assert one.size == 5
    print(f"class_balance: minority counts in 20 draws of 100 from 99:1 {min(minority)}-{max(minority)} OK")


def check_unoccupied_not_occluded_structure() -> None:
    """Free unoccupied samples must not sit in occluded above-ground structure (not in keys)."""
    rng = np.random.default_rng(0)
    # occupied: a full ground layer at z cell 5 (~0.2 m) plus a few visible upper posts
    nx, ny, nz = G.shape
    gz = 5
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ground = G.key_of_cell(np.stack([ix.ravel(), iy.ravel(), np.full(ix.size, gz)], 1))
    posts = G.keys_of(np.stack([np.linspace(-5, 5, 20), np.full(20, 15.0), np.full(20, 3.0)], 1))
    posts = posts[posts >= 0]
    keys = np.unique(np.concatenate([ground, posts]))
    cls = np.zeros(keys.size, np.uint8)
    # planted occluded structure above ground, NOT in keys
    hidden = G.keys_of(np.stack([np.linspace(8, 12, 40), np.full(40, 20.0), np.full(40, 2.5)], 1))
    hidden = hidden[(hidden >= 0) & (~np.isin(hidden, keys))]
    assert hidden.size >= 10, hidden.size
    free = np.zeros((0, 3), np.float64)
    q = sample_queries(keys, cls, free, rng, n_occ=50, n_free=80, n_coarse=0)
    free_m = (q["occ"] == 0) & (q["occ_w"] == 1)
    free_pts = q["pts"][free_m].astype(np.float64)
    free_k = G.keys_of(free_pts)
    assert not np.isin(free_k, hidden).any(), int(np.isin(free_k, hidden).sum())
    # and we actually sampled below-ground unoccupied centres (kills n_empty=0)
    zc = G.cell_of(free_pts)[:, 2] if free_pts.size else np.zeros(0, np.int64)
    n_below = int((zc < gz).sum()) if free_pts.size else 0
    assert n_below >= 20, n_below
    # none of the free samples are above the ground layer
    assert not (zc > gz).any(), int((zc > gz).sum())
    print(f"unoccupied free: {n_below} below-ground, 0 in planted occluded structure, 0 above ground OK")


def main() -> int:
    # pin to 1: CI runners and this box disagree on thread count, and the
    # overfit predict IoU was 0.515 / 14,022 queries on CI vs 0.722 / 20,517 here.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    check_projection()
    check_camera_only()
    check_sparse_predict()
    check_capped_hot_spreads()
    check_capped_hot_below_ground()
    check_capped_hot_follows_ground()
    check_capped_hot_surface_fill()
    check_capped_hot_structure_surface()
    check_capped_hot_grade_and_onelayer()
    check_capped_hot_p_margin_and_robust_env()
    check_capped_hot_exposed_ground()
    check_capped_hot_inherit_and_vote_edges()
    check_capped_hot_low_p_grade_and_saturated_fill()
    check_class_balance()
    check_unoccupied_not_occluded_structure()
    check_queries()
    with tempfile.TemporaryDirectory() as d:
        check_overfit(Path(d))
    print("test_occ_net: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
