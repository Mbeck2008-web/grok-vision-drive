#!/usr/bin/env python3
"""Visible vs occluded road metrics on SYNTHETIC intersection / roundabout clips (offline, CPU).

Trains a SparseOccNet on map-labelled synthetic clips (``tests/synthetic_scenes``,
labels from ``occ_map_labels`` over the whole grid), then runs the causal
``OccRuntime`` over held-out clips with the VO memory ON and OFF (ablation).
Optionally also trains a camera-masked baseline (``--baseline``: the old
loss, occluded cells weight 0). Every number is from this run; synthetic,
not BeamNG.

Metrics per clip kind, split by 8-camera ray-cast GT visibility of the cell:

* ``iou_visible`` / ``iou_occluded``: drivable IoU on the ground layer;
* ``edge_chamfer_visible`` / ``_occluded``: GT road outline points vs the
  boundary of the predicted drivable cells (completeness, metres, capped 5 m)
  plus predicted boundary -> GT (accuracy), mean of the two;
* lane chamfer: not reported (no lane head; paint / lane labels UNLABELED);
* the window is the occupancy grid (x +-25.6, y -12.8..51.2 m): metrics are
  inside it; ``road_beyond_window`` = share of GT centerline length within
  80 m of the ego that lies outside it.

    python scripts/eval_occ_unseen.py --steps 300 --out /tmp/unseen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID, rig_cams  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_map_labels import build_map_labels, merge_targets, visible_keys, _segments  # noqa: E402
import synthetic_scenes as ss  # noqa: E402
from synthetic_occ import GROUND_Z  # noqa: E402

G = DEFAULT_GRID
CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
KINDS = {"intersection": ss.intersection, "roundabout": ss.roundabout}


def frame_labels(scene, rcams):
    rend = {cid: ss.render(cam, scene) for cid, cam in rcams.items()}
    planes = {cid: r[:2] for cid, r in rend.items()}
    lab = build_occ_labels(planes, rcams, sem_classes=ss.SEM_CLASSES, stride=1)
    vis = visible_keys(planes, rcams, stride=2)
    ml = build_map_labels(scene.roads, ground_z=GROUND_Z)
    keys, cls = merge_targets(lab.keys, lab.cls, ml)
    return {"images": {cid: r[2] for cid, r in rend.items()}, "keys": keys, "cls": cls, "free": lab.free,
            "vis": vis, "ml": ml, "lab": lab}


def build_shards(out_dir, seeds, rcams, frames, dt):
    from python.data.occ_shards import ShardWriter

    n = 0
    with ShardWriter(out_dir) as w:
        for kind, fn in KINDS.items():
            for s in seeds:
                for t, sc in ss.clip(fn(s), frames=frames, dt=dt):
                    f = frame_labels(sc, rcams)
                    w.write(f"{kind}{s}f{int(round(t * 100)):04d}", f["images"],
                            {"keys": f["keys"], "cls": f["cls"], "free": f["free"], "vis": f["vis"]},
                            {"recovery": n % 4 == 0, "synthetic": kind})
                    n += 1
    return n


def boundary(drv_xy_cells: set) -> np.ndarray:
    out = []
    for ix, iy in drv_xy_cells:
        if any((ix + dx, iy + dy) not in drv_xy_cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            out.append((ix, iy))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def _cxy(cells):
    v = G.voxel_m
    return np.stack([G.x_min + (cells[:, 0] + 0.5) * v, G.y_min + (cells[:, 1] + 0.5) * v], 1)


def _nn(a, b, cap=5.0):
    if not len(a):
        return np.zeros(0)
    if not len(b):
        return np.full(len(a), cap)
    out = np.empty(len(a))
    for i in range(0, len(a), 512):
        d = np.sqrt(((a[i:i + 512, None, :] - b[None]) ** 2).sum(-1)).min(1)
        out[i:i + 512] = np.minimum(d, cap)
    return out


def frame_metrics(rt, f, scene):
    keys, cls, prob, obs = rt.last_cells
    gk = f["ml"].ground_k
    c = G.cell_of_key(keys) if keys.size else np.zeros((0, 3), np.int64)
    pd = {(int(a), int(b)) for (a, b, _), k in zip(c, cls) if k == 1}
    pobs = {}
    for (a, b, _), o, k in zip(c, obs, cls):
        if k == 1:
            pobs[(int(a), int(b))] = pobs.get((int(a), int(b)), False) or bool(o)
    gd = {(int(a), int(b)) for a, b, _ in G.cell_of_key(f["ml"].drivable_keys)}
    nx, ny, _ = G.shape
    vis_g = {(int(a), int(b)) for a, b, z in G.cell_of_key(f["vis"]) if z == gk}
    res = {}
    allc = {(a, b) for a in range(nx) for b in range(ny)}
    for split, S in (("visible", vis_g), ("occluded", allc - vis_g)):
        P, D = pd & S, gd & S
        u = len(P | D)
        res[f"iou_{split}"] = len(P & D) / u if u else float("nan")
        res[f"gt_cells_{split}"] = len(D)
    gte = np.concatenate(f["ml"].edges) if f["ml"].edges else np.zeros((0, 2))
    gte_cells = np.floor((gte - [G.x_min, G.y_min]) / G.voxel_m).astype(int)
    gvis = np.array([(int(a), int(b)) in vis_g for a, b in gte_cells], bool)
    pb = boundary(pd)
    pbxy = _cxy(pb) if pb.size else np.zeros((0, 2))
    pbo = np.array([pobs.get((int(a), int(b)), False) for a, b in pb], bool)
    for split, gm, pm in (("visible", gvis, pbo), ("occluded", ~gvis, ~pbo)):
        comp = _nn(gte[gm], pbxy)
        acc = _nn(pbxy[pm], gte)
        res[f"edge_chamfer_{split}"] = float((comp.mean() if comp.size else np.nan) / 2 + (acc.mean() if acc.size else np.nan) / 2)
        res[f"edge_pts_{split}"] = int(gm.sum())
    pts = []
    for r in scene.roads:
        a, b = _segments(r)
        for p, q in zip(a, b):
            L = float(np.hypot(*(q - p)))
            for t in np.arange(0, L, 0.5):
                pts.append(p + (q - p) * t / max(L, 1e-9))
    P = np.asarray(pts)
    near = np.hypot(P[:, 0], P[:, 1]) <= 80.0
    inside = (P[:, 0] > G.x_min) & (P[:, 0] < G.x_max) & (P[:, 1] > G.y_min) & (P[:, 1] < G.y_max)
    res["road_beyond_window"] = float(1.0 - (near & inside).sum() / max(1, near.sum()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-seeds", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--render-long", type=int, default=96)
    ap.add_argument("--net-long", type=int, default=64)
    ap.add_argument("--config", default="cheap")
    ap.add_argument("--baseline", action="store_true", help="also train the camera-masked (visible_only) baseline")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch

    from python.perception.occ_out import OccRuntime
    from python.perception.occ_sparse import PROFILES, SparseOccNet, memory_estimate, profile_cams
    from python.train import occ_train

    torch.set_num_threads(a.threads)
    rcams = {c.cam_id: c.scaled(a.render_long) for c in rig_cams(CFG)}
    ncams = [c.scaled(a.net_long) for c in rig_cams(CFG)]
    t0 = time.time()
    report = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "synthetic": True, "args": vars(a)}
    with tempfile.TemporaryDirectory() as d:
        n = build_shards(Path(d), range(100, 100 + a.train_seeds), rcams, a.frames, a.dt)
        report["train_samples"] = n
        report["shard_build_s"] = time.time() - t0
        models = {"map_unmasked": False}
        if a.baseline:
            models["camera_masked"] = True
        nets = {}
        for name, vo in models.items():
            orig = occ_train.sample_queries
            if vo:
                occ_train.sample_queries = lambda *x, **k: orig(*x, **{**k, "visible_only": True})  # noqa: E731
            try:
                t1 = time.time()
                torch.manual_seed(0)
                net = SparseOccNet(ncams, config=a.config)
                r = occ_train.train(d, ncams, steps=a.steps, batch_size=a.batch, net=net, seed=0)
            finally:
                occ_train.sample_queries = orig
            h = r["history"]
            report[f"train_{name}"] = {"s": time.time() - t1, "loss_first5": float(np.mean([x["loss"] for x in h[:5]])),
                                       "loss_last5": float(np.mean([x["loss"] for x in h[-5:]])), "params": r["params"]}
            nets[name] = r["net"].eval()
    rows = {}
    for kind, fn in KINDS.items():
        for s in range(900, 900 + a.eval_seeds):
            scene = fn(s)
            clip = ss.clip(scene, frames=a.frames, dt=a.dt)
            labs = [frame_labels(sc, rcams) for _, sc in clip]
            imgs = [[torch.from_numpy(occ_train.prep_image(f["images"][c.cam_id], c))[None] for c in ncams] for f in labs]
            for name, net in nets.items():
                for mem in (True, False):
                    if name != "map_unmasked" and not mem:
                        continue
                    net.reset_temporal()
                    rt = OccRuntime(net, memory=mem, max_points=16384, max_coarse=2048)
                    per = []
                    for (t, sc), f, im in zip(clip, labs, imgs):
                        with torch.no_grad():
                            rt.step(im, t)
                        per.append(frame_metrics(rt, f, sc))
                    key = f"{name}{'' if mem else '_no_memory'}"
                    rows.setdefault(kind, {}).setdefault(key, []).append(per[-1])  # last frame: memory has history
                    rows[kind].setdefault("_beyond", {})[s] = per[-1]["road_beyond_window"]
    summ = {}
    for kind, byk in rows.items():
        summ[kind] = {"road_beyond_window_per_clip": byk.pop("_beyond")}
        for key, lst in byk.items():
            summ[kind][key] = {m: float(np.nanmean([x[m] for x in lst])) for m in lst[0] if m != "road_beyond_window"}
    report["metrics_last_frame_mean"] = summ
    est = {f"batch{b}": memory_estimate(profile_cams(CFG, "1080ti"), config=PROFILES["1080ti"]["config"],
                                        max_points=PROFILES["1080ti"]["max_points"], batch=b)["total_mb"] for b in (1, a.batch)}
    report["vram_est_mb_cpu_1080ti"] = est
    report["runner_batch"] = {"offline_eval_and_live": 1, "training": a.batch}
    report["wall_s"] = time.time() - t0
    txt = json.dumps(report, indent=1, default=float)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
