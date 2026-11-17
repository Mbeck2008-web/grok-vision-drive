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
from python.train.occ_train import prep_image, sample_queries, train  # noqa: E402
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
    for _ in range(3):  # capped top stage (2048 of the visible coarse cells): let the sweep + hot carry settle
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


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    check_projection()
    check_camera_only()
    check_sparse_predict()
    check_capped_hot_spreads()
    check_queries()
    with tempfile.TemporaryDirectory() as d:
        check_overfit(Path(d))
    print("test_occ_net: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
