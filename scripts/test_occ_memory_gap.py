#!/usr/bin/env python3
"""Memory gap: a road leg seen early, then hidden, is recovered only with memory.

Tiny deterministic SYNTHETIC scene, cameras main + wide. Two variants that
differ only by a right-hand road leg (A: leg, B: no leg). Frames 0..K see the
near leg; from frame K+1 a wall (a stopped truck) hides it. A bus parked
across the leg hides its far part in every frame (only inferable from the
scene, so it needs the unmasked loss and the scene-context decoder).

Score on the last frame, over the whole leg footprint (cells no live camera
reaches at that frame):

    score = (IoU_drivable(A) + specificity(B)) / 2

With memory off the frames after K are pixel-identical in A and B, so the net
gives the same cells the same class in both: IoU(A) <= recall(A) = 1 -
specificity(B), hence score <= 0.5 (up to query-coverage differences). The
test asserts score(memory on) - score(memory off) >= 0.25 and score(on) >= 0.75.

Mutants checked in-process (each must break the gap):
* lift ablated: camera features zeroed in the decoder;
* memory disabled: OccMemory returns this tick's cells only;
* loss masked to visible: the camera-masked baseline loss.
"""
# gvd-ci: requires torch
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID, rig_cams  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_map_labels import MapRoad, build_map_labels, merge_targets, visible_keys  # noqa: E402
import synthetic_scenes as ss  # noqa: E402
from synthetic_occ import GROUND_Z  # noqa: E402

G = DEFAULT_GRID
CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
CAM_IDS = ("main", "wide")
RENDER, NET = 64, 48
FRAMES, K, DT, SPEED, Y0 = 6, 2, 0.25, 10.0, -8.0  # ego starts at y = Y0; frames 0..K see the leg, K+1.. hidden
YC, LEG_W = 18.0, 6.0
LEG_X0, LEG_X1 = 4.4, 24.0                   # scored footprint: behind the wall line
WALL = ((3.5, 11.0, 0.0), (4.3, 40.0, 6.0))    # static: hides the leg once the ego has driven past its start
BUS = ((9.0, 2.0, 0.0), (10.0, YC + 6.0, 6.0))  # always: hides the far leg
BLDG = [((-16.0, 4.0, 0.0), (-8.0, 12.0, 6.0)), ((-15.0, 22.0, 0.0), (-7.0, 32.0, 5.0)),
        ((-14.0, 38.0, 0.0), (-6.0, 46.0, 7.0))]
# VO needs structure across the motion: faces parallel to it are sampled at ego-fixed points
# by the pixel rays and register best at zero shift (measured: GT-label VO 0.0 m vs true 1.0 m).
POSTS = [((-5.6, y, 0.0), (-4.8, y + 0.8, 4.0)) for y in (5.0, 8.5, 13.0, 16.5, 21.5, 25.0, 30.5, 36.0, 41.0)]


def scene(leg: bool, hidden: bool) -> ss.Scene:
    roads = [MapRoad(np.array([[0.0, -20.0], [0.0, 80.0]]), 7.0)]
    if leg:
        roads.append(MapRoad(np.array([[3.0, YC], [30.0, YC]]), LEG_W))
    boxes = list(BLDG) + POSTS + [BUS, WALL]
    return ss.Scene("leg" if leg else "noleg", roads, boxes, SPEED)


def clip(leg: bool):
    return [(i * DT, scene(leg, i > K).shifted(Y0 + SPEED * i * DT)) for i in range(FRAMES)]


def leg_cells(dy: float) -> np.ndarray:
    """Ground-layer keys of the leg footprint (both variants), ego frame after dy."""
    ml = build_map_labels([MapRoad(np.array([[3.0, YC - dy], [30.0, YC - dy]]), LEG_W)], ground_z=GROUND_Z)
    k = ml.drivable_keys
    x = G.centers_of_keys(k)[:, 0]
    return k[(x >= LEG_X0) & (x <= LEG_X1)]


def cams():
    rc = {c.cam_id: c.scaled(RENDER) for c in rig_cams(CFG) if c.cam_id in CAM_IDS}
    nc = [c.scaled(NET) for c in rig_cams(CFG) if c.cam_id in CAM_IDS]
    return rc, nc


def frame(sc, rcams):
    rend = {cid: ss.render(cam, sc) for cid, cam in rcams.items()}
    planes = {cid: r[:2] for cid, r in rend.items()}
    lab = build_occ_labels(planes, rcams, sem_classes=ss.SEM_CLASSES, stride=1)
    vis = visible_keys(planes, rcams, stride=1)
    ml = build_map_labels(sc.roads, ground_z=GROUND_Z)
    keys, cls = merge_targets(lab.keys, lab.cls, ml)
    return {"images": {cid: r[2] for cid, r in rend.items()}, "keys": keys, "cls": cls, "free": lab.free,
            "vis": vis, "ml": ml}


def build(d: Path, rcams) -> list:
    from python.data.occ_shards import ShardWriter

    frames, n = [], 0
    with ShardWriter(d) as w:
        for leg in (True, False):
            for t, sc in clip(leg):
                f = frame(sc, rcams)
                frames.append((leg, t, f))
                w.write(f"{'A' if leg else 'B'}{int(round(t * 100)):04d}", f["images"],
                        {"keys": f["keys"], "cls": f["cls"], "free": f["free"], "vis": f["vis"]},
                        {"recovery": n % 4 == 0, "synthetic": "memory_gap"})
                n += 1
    return frames


def train_net(d: Path, ncams, *, steps: int, visible_only: bool = False):
    import torch

    from python.perception.occ_sparse import SparseOccNet
    from python.train import occ_train

    orig = occ_train.sample_queries
    if visible_only:
        occ_train.sample_queries = lambda *x, **k: orig(*x, **{**k, "visible_only": True})  # noqa: E731
    try:
        torch.manual_seed(0)
        net = SparseOccNet(ncams, config="cheap")
        r = occ_train.train(d, ncams, steps=steps, batch_size=4, lr=3e-3, net=net, seed=0)
    finally:
        occ_train.sample_queries = orig
    h = r["history"]
    return r["net"].eval(), (float(np.mean([x["loss"] for x in h[:5]])), float(np.mean([x["loss"] for x in h[-5:]])))


def run_clip(net, ncams, frames, *, leg: bool, memory: bool, trace=None):
    import torch

    from python.perception.occ_out import OccRuntime
    from python.train.occ_train import prep_image

    net.reset_temporal()
    rt = OccRuntime(net, memory=memory, max_points=16384, max_coarse=5120)
    if trace is not None:
        rt.trace = trace
    oks = []
    for lg, t, f in frames:
        if lg != leg:
            continue
        im = [torch.from_numpy(prep_image(f["images"][c.cam_id], c))[None] for c in ncams]
        with torch.no_grad():
            m = rt.step(im, t)
        oks.append(bool(m["ego"]["ok"]))
    keys, cls, prob, obs = rt.last_cells
    return keys, cls, oks


def target_sets(frames):
    """Leg footprint keys in the LAST ego frame: seen early (a camera reached it in frames 0..K) and
    hidden at the last frame; and never-seen (no camera reached it in any frame)."""
    lastdy = Y0 + SPEED * frames[FRAMES - 1][1]
    L = leg_cells(lastdy)
    hits = []
    for lg, t, f in frames[:FRAMES]:
        c = G.centers_of_keys(L) + np.array([0.0, lastdy - (Y0 + SPEED * t), 0.0])  # last-frame cell -> that frame
        hits.append(np.isin(G.keys_of(c), f["vis"]))
    seen = np.any(hits[:K + 1], axis=0) & ~hits[-1]
    never = ~np.any(hits, axis=0)
    return L[seen], L[never]


def score(net, ncams, frames, *, memory: bool, trace=None):
    """{"seen_hidden": (score, IoU(A), spec(B)), "never_seen": ...} on the last frame."""
    S, N = target_sets(frames)
    out = {}
    for leg in (True, False):
        keys, cls, oks = run_clip(net, ncams, frames, leg=leg, memory=memory, trace=trace)
        out[leg] = (keys[cls == 1], oks)
    res = {}
    for part, T in (("seen_hidden", S), ("never_seen", N)):
        iou = float(np.isin(T, out[True][0]).mean())   # drivable within the footprint / footprint
        spec = float(1.0 - np.isin(T, out[False][0]).mean())
        res[part] = ((iou + spec) / 2, iou, spec, int(T.size))
    res["vo_ok"] = (out[True][1], out[False][1])
    return res
