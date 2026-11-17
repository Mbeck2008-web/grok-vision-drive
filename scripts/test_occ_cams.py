#!/usr/bin/env python3
"""Sparse occupancy net: reduced cameras, per-camera age, sparsity, checkpoints.

* presets ``min`` (main+wide) and ``front3`` (main+wide+narrow): an 8-camera net
  with the other cameras masked gives the same outputs as the same weights
  built on only those cameras, and ``predict`` runs with ``None`` images for
  the missing ones;
* per-camera age at the rig pattern 10/10/10/2/2/1/1/1 Hz: a stale camera's
  weight is below a fresh one's, a fully stale (or missing) camera has weight 0
  and perturbing its pixels does not change the fused output, while
  perturbing a partly stale camera does (it is down-weighted, not dropped);
* sparsity begins at the query set: the only dense tensors are the 2-D
  per-camera feature maps; no 3-D activation in ``predict`` has as many rows as
  the dense grid (hooked and counted);
* checkpoints: input size, camera count/preset and the grid box are loadable
  knobs (same weights, forward works); a width or depth change fails with
  ``CheckpointMismatch`` naming the knob, never a partial load.
"""
# gvd-ci: requires torch
from __future__ import annotations

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

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, rig_cams  # noqa: E402
from python.perception.occ_sparse import (  # noqa: E402
    AGE_MAX_S, CAM_PRESETS, PROFILES, CheckpointMismatch, SparseOccNet, cam_weights, load_checkpoint, save_checkpoint,
)
from python.train.occ_train import train_cam_weights  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
RIG_HZ = {"narrow": 10.0, "main": 10.0, "wide": 10.0, "pillarL": 2.0, "pillarR": 2.0, "repeatL": 1.0, "repeatR": 1.0, "rear": 1.0}


def _cams(long_side: int = 64):
    return [c.scaled(long_side) for c in rig_cams(CFG)]


def _pts(n: int = 3000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    lo, hi = torch.tensor(DEFAULT_GRID.lo, dtype=torch.float32), torch.tensor(DEFAULT_GRID.hi, dtype=torch.float32)
    return (torch.rand((1, n, 3), generator=g) * (hi - lo) + lo)


def check_presets() -> None:
    cams = _cams()
    torch.manual_seed(0)
    full = SparseOccNet(cams).eval()
    imgs = [torch.rand(1, 3, c.height, c.width) for c in cams]
    pts = _pts()
    for preset in ("min", "front3"):
        keep = CAM_PRESETS[preset]
        w = cam_weights(full.cam_ids, preset=preset)
        assert w.tolist() == [1.0 if c in keep else 0.0 for c in full.cam_ids], (preset, w)
        sub_cams = [c for c in cams if c.cam_id in keep]
        sub = SparseOccNet(sub_cams).eval()
        sub.load_state_dict(full.state_dict())
        sub_imgs = [im for c, im in zip(cams, imgs) if c.cam_id in keep]
        masked_imgs = [im if c.cam_id in keep else None for c, im in zip(cams, imgs)]
        with torch.no_grad():
            a = full(imgs, pts, w)
            b = sub(sub_imgs, pts)
            c_ = full(masked_imgs, pts)
            seen_all = float(full(imgs, pts)["seen"].mean())
        for k in ("occ", "any", "cls", "seen"):
            assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(c_[k], b[k], atol=1e-5), (preset, k)
        seen = float(b["seen"].mean())
        assert 0.0 < seen < seen_all, (preset, seen, seen_all)
        p_sub = sub.predict(sub_imgs, max_points=4096)
        p_full = full.predict(masked_imgs, max_points=4096)
        assert p_full.n_coarse == p_sub.n_coarse and np.array_equal(p_full.keys, p_sub.keys), preset
        print(f"preset {preset} ({'+'.join(keep)}): masked 8-cam net == {len(keep)}-cam net, "
              f"seen {seen:.3f} (8 cams {seen_all:.3f}), predict {p_sub.n_queried} pts OK")


def check_ages() -> None:
    cams = _cams()
    ids = tuple(c.cam_id for c in cams)
    # Tick at t = 0.95 s on the rig pattern: each cam's last frame is floor(t*hz)/hz.
    t = 0.95
    ages = {c: t - np.floor(t * hz) / hz for c, hz in RIG_HZ.items()}
    w = cam_weights(ids, ages_s=ages)
    wd = dict(zip(ids, w.tolist()))
    fresh = min(wd["main"], wd["wide"], wd["narrow"])
    assert fresh > 0.85, wd
    assert 0.0 < wd["pillarL"] < fresh and 0.0 < wd["rear"] < wd["pillarL"], wd
    # one tick later the 1 Hz cams have a new frame: their weight resets to ~fresh
    t2 = 1.05
    w2 = dict(zip(ids, cam_weights(ids, ages_s={c: t2 - np.floor(t2 * hz) / hz for c, hz in RIG_HZ.items()}).tolist()))
    assert w2["rear"] > 0.85 and w2["rear"] > wd["rear"], (w2, wd)
    # older than the max age, missing, NaN or negative: weight 0
    w3 = dict(zip(ids, cam_weights(ids, ages_s={"main": 0.0, "wide": AGE_MAX_S + 0.01, "narrow": float("nan"), "rear": -1.0}).tolist()))
    assert w3["main"] == 1.0 and w3["wide"] == 0.0 and w3["narrow"] == 0.0 and w3["rear"] == 0.0 and w3["pillarL"] == 0.0, w3
    # behaviour: perturb a fully stale cam -> no change; a partly stale cam -> change
    torch.manual_seed(1)
    net = SparseOccNet(cams).eval()
    imgs = [torch.rand(1, 3, c.height, c.width) for c in cams]
    pts = _pts(4000, 1)
    k_stale, k_part = ids.index("rear"), ids.index("pillarL")
    ages_mix = dict(ages, rear=AGE_MAX_S + 0.5)
    wm = cam_weights(ids, ages_s=ages_mix)
    assert wm[k_stale] == 0.0 and 0.0 < wm[k_part] < wm[ids.index("main")], wm
    with torch.no_grad():
        base = net(imgs, pts, wm)
        pert = list(imgs)
        pert[k_stale] = torch.ones_like(imgs[k_stale])
        same = net(pert, pts, wm)
        assert torch.equal(base["occ"], same["occ"]) and torch.equal(base["cls"], same["cls"]), "a fully stale cam leaked into fusion"
        pert = list(imgs)
        pert[k_part] = 1.0 - imgs[k_part]
        moved = (net(pert, pts, wm)["occ"] - base["occ"]).abs().max().item()
        w_one = wm.copy(); w_one[k_part] = 1.0
        reweighted = (net(imgs, pts, w_one)["occ"] - base["occ"]).abs().max().item()
    assert reweighted > 1e-4, "a partly stale cam must count less than a fresh one in the fused mean"
    assert moved > 1e-4, moved
    calls: list[int] = []
    hk = net.encoder.register_forward_hook(lambda _m, _i, _o: calls.append(1))
    pred = net.predict(imgs, cam_w=wm, max_points=4096)
    hk.remove()
    assert len(calls) == int((wm > 0).sum()) == 7, (len(calls), wm)  # the stale cam is not even encoded
    pred_pert = net.predict([im if k != k_stale else torch.zeros_like(im) for k, im in enumerate(imgs)], cam_w=wm, max_points=4096)
    assert np.array_equal(pred.keys, pred_pert.keys) and pred.n_queried == pred_pert.n_queried
    # training draws the same kind of weights: main+wide always 1, others dropped or aged
    tw = train_cam_weights(ids, 256, 0.3, np.random.default_rng(0))
    assert tw.shape == (256, 8) and np.all(tw[:, ids.index("main")] == 1.0) and np.all(tw[:, ids.index("wide")] == 1.0)
    drop = (tw[:, ids.index("rear")] == 0.0).mean()
    assert 0.2 < drop < 0.4 and tw.min() >= 0.0 and tw.max() <= 1.0, drop
    print(f"ages @ t=0.95 s (10/10/10/2/2/1/1/1 Hz): main {wd['main']:.2f} pillarL {wd['pillarL']:.2f} rear {wd['rear']:.2f}; "
          f"fully stale cam pixels change nothing, partly stale cam moves occ by {moved:.4f} OK")


def check_sparsity_begins() -> None:
    cams = _cams()
    torch.manual_seed(0)
    net = SparseOccNet(cams).eval()
    with torch.no_grad():
        net.head[-1].weight.zero_(); net.head[-1].bias.zero_(); net.head[-1].bias[1:3] = 10.0  # all hot: worst case
    imgs = [torch.rand(1, 3, c.height, c.width) for c in cams]
    rows: list[int] = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: rows.append(int(o.shape[1]))) for m in net.head]
    fmaps: list[tuple[int, ...]] = []
    hooks.append(net.encoder.register_forward_hook(lambda _m, _i, o: fmaps.append(tuple(o.shape))))
    pred = net.predict(imgs, max_points=16384)
    for h in hooks:
        h.remove()
    dense = DEFAULT_GRID.dense_cells
    assert max(rows) <= 16384 + 8 and max(rows) < dense / 10, (max(rows), dense)
    assert all(len(s) == 4 for s in fmaps) and len(fmaps) == 8  # 2-D maps per camera, no 3-D volume
    print(f"sparsity begins at the query set: largest 3-D query batch {max(rows)} rows vs dense grid {dense}; "
          f"queried {pred.n_queried} ({pred.queried_ratio():.4f} of dense); dense tensors are 8 2-D feature maps only OK")


def check_checkpoints(tmp: Path) -> None:
    cams = _cams(64)
    torch.manual_seed(0)
    a = SparseOccNet(cams).eval()
    path = tmp / "a.pt"
    save_checkpoint(a, path, profile="1080ti")
    pts = _pts(500, 2)
    loadable = {
        "input 32 px": SparseOccNet(_cams(32)),
        "cams min": SparseOccNet([c for c in cams if c.cam_id in CAM_PRESETS["min"]]),
        "grid 0.2 m": SparseOccNet(cams, grid=OccGrid(voxel_m=0.2)),
    }
    for name, net in loadable.items():
        meta = load_checkpoint(net.eval(), path)
        assert meta == {"profile": "1080ti"}, meta
        for k, v in a.state_dict().items():
            assert torch.equal(net.state_dict()[k], v), (name, k)
        imgs = [torch.rand(1, 3, c.height, c.width) for c in net.cams]
        with torch.no_grad():
            out = net(imgs, pts)
        assert out["occ"].shape == (1, 500) and torch.isfinite(out["occ"]).all(), name
    retrain = {
        "cheap profile": (SparseOccNet(cams, config="cheap"), "channels"),
        "hidden 96": (SparseOccNet(cams, hidden=96), "hidden"),
        "feat 48": (SparseOccNet(cams, feat=48), "feat"),
        "depth +1 stage": (SparseOccNet(cams, channels=(32, 64, 96, 96)), "channels"),
    }
    for name, (net, knob) in retrain.items():
        before = {k: v.clone() for k, v in net.state_dict().items()}
        try:
            load_checkpoint(net, path)
        except CheckpointMismatch as e:
            assert knob in str(e) and "retrain" in str(e), (name, str(e))
        else:
            raise AssertionError(f"{name}: loaded across a width/depth change")
        assert all(torch.equal(before[k], v) for k, v in net.state_dict().items()), f"{name}: partial load"
    torch.save({"state_dict": a.state_dict()}, tmp / "bare.pt")
    try:
        load_checkpoint(SparseOccNet(cams), tmp / "bare.pt")
    except CheckpointMismatch:
        pass
    else:
        raise AssertionError("a checkpoint without a spec loaded")
    assert all(p["compute"] == "fp32" for p in PROFILES.values())
    from python.perception.occ_sparse import memory_estimate, profile_cams
    for prof, pr in PROFILES.items():
        pc = profile_cams(CFG, prof)
        assert sorted(c.cam_id for c in pc) == sorted(CAM_PRESETS[pr["cams"]]), prof
        est = memory_estimate(pc, config=pr["config"])
        assert est["params_formula"] == sum(p.numel() for p in SparseOccNet(pc, config=pr["config"]).parameters()), prof
        k2 = memory_estimate(pc, config=pr["config"], temporal_k=2)
        assert abs(k2["feature_cache_mb"] - 2 * est["feature_cache_mb"]) < 1e-9
    fc = {c.cam_id: c.width for c in profile_cams(CFG, "front_cluster")}
    assert fc == {"main": 800, "wide": 640}, fc
    print(f"checkpoints: loads across {', '.join(loadable)}; refuses {', '.join(retrain)} with no partial load OK")


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    check_presets()
    check_ages()
    check_sparsity_begins()
    with tempfile.TemporaryDirectory() as d:
        check_checkpoints(Path(d))
    print("test_occ_cams: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
