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
from python.perception.occ_sparse import LEVELS, SparseOccNet, _capped_hot, visible_mask  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_train import _class_balance, _unoccupied_points, prep_image, sample_queries, train  # noqa: E402
from synthetic_occ import SEM_CLASSES, render  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
G = DEFAULT_GRID


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
        if cap <= 1024 or f == 2:
            # 12.5% of columns have ground; structure sits on those same
            # columns only (enough cells to fill the non-band budget) so a
            # hole cannot promote a roof to the band. Picked below must be 0.
            # A widened band (N6b) would pull the adjacent below into the
            # band and spend those slots on below.
            # Stride so n_ground < cap//2 but n_ground + in-grid struct >= cap
            # (at 1.6 m the third struct layer can clip z_max).
            step = 6 if cap <= 1024 else 8
            gm = (np.arange(X.size) % step) == 0
            gz = np.broadcast_to(np.asarray(0.2, dtype=np.float64), X.shape)
            parts = [(np.stack([X[gm], Y[gm], zc(gz[gm])], 1), 0.95,
                      np.where(Y[gm] > 20.0, "groundFAR", "groundNEAR")),
                     (np.stack([X[gm], Y[gm], zc(gz[gm] - cm)], 1), 0.9,
                      np.full(int(gm.sum()), "below"))]
            for k in (1, 2, 3):
                parts.append((np.stack([X[gm], Y[gm], zc(gz[gm] + k * cm)], 1),
                              0.7, np.full(int(gm.sum()), "struct")))
            # reuse run() by passing through a temporary ground_mask path:
            # build via a one-off that uses the same exact-count asserts
            def run_parts(name, parts):
                hot = np.concatenate([p[0] for p in parts])
                ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
                lab = np.concatenate([p[2] for p in parts])
                keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
                hot, ph, lab = hot[keep], ph[keep], lab[keep]
                idx = _capped_hot(hot, ph, cap, G.z_min, cm)
                picked = lab[idx]
                n_g = int(np.isin(picked, ("groundFAR", "groundNEAR")).sum())
                n_st = int((picked == "struct").sum())
                n_bel = int((picked == "below").sum())
                n_g_tot = int(np.isin(lab, ("groundFAR", "groundNEAR")).sum())
                n_st_tot = int((lab == "struct").sum())
                n_bel_tot = int((lab == "below").sum())
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
                assert n_bel == n_dn == 0, (name, f, n_bel, n_dn, kept)
                assert n_g == n_g_tot, (name, f, "ground dropped", kept)
                print(f"capped_hot f={f} cell={cm}m cap={cap} {name}: "
                      f"ground {n_g} struct {n_st} below {n_bel} OK")
            run_parts("holed (N6)", parts)



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
    check_class_balance()
    check_unoccupied_not_occluded_structure()
    check_queries()
    with tempfile.TemporaryDirectory() as d:
        check_overfit(Path(d))
    print("test_occ_net: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
