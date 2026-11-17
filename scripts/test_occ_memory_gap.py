#!/usr/bin/env python3
"""Hidden-leg memory proof for s2 OccMemory, ported from swarm s4's protocol.

s4 (gvd/occ-sparse-s4, scripts/test_occ_ablation.py) isolated memory from VO:
two ticks, a hide at tick 1, a Youden J = recall - false-alarm gate on hidden
leg cells, and in-process mutants that must fail the gate. That test targeted
s4's differentiable feature memory (python.occ). This port uses the same
protocol on s2's APIs: SparseOccNet + OccRuntime (predict, cap, observed-mask,
OccMemory). Ego warp is the vehicle-signal path (camera VO has test_occ_ego).
Threads are pinned and the RNGs are seeded so the overfit is repeatable.

Scene (world metres, ego drives +y): a straight road, an optional right-hand
asphalt leg, and a long step that leaves that patch behind every camera at tick 1. The visible leg is painted a high-contrast colour (not the default
gray asphalt) so the cheap stride-8 encoder at 128 px can tell it from grass;
the thing under test is still OccMemory, not color discrimination. Tick 0: the scored cells are visible. Tick 1: they sit outside every camera frustum (checked by project). Whether the leg exists at tick 1 is then
only knowable from tick 0, i.e. from OccMemory.

The proof runs through ``OccRuntime.step`` (net ``predict`` + cap +
observed-mask + OccMemory). Ego warp uses the vehicle-signal path
(``wheelspeed_mps = STEP_M/DT``) so a 2-tick cheap overfit is not also a
VO test (camera VO has ``test_occ_ego``). Query positions and the observed flag are
not taken from GT; held-out starts sit more than one 0.4 m voxel from any
train start. Lift ablation is retrained with features zeroed; a loss-ablation
mutant drops the visible-leg queries from the training loss.

Score at tick 1 on those hidden cells, on held-out start positions:

    recall      = fraction called drivable (occ prob >= 0.5 and cls == 1) when the leg is present
    false_alarm = the same fraction when the leg is absent
    J           = recall - false_alarm

GATE (fixed before the first run): J(memory on) >= 0.5 and
J(memory on) - J(memory off) >= 0.3.

Mutants (each must fail the gate):
* lift ablated: retrained with camera features zeroed in the decoder
* memory disabled: OccMemory not applied
* zero warp: memory is updated with fwd=0 (the s4/s2 failure mode where VO
  locks to an ego-fixed sheet and the carried leg misses its tick-1 cell)
* loss ablated: visible-leg queries dropped from the training loss
"""
# gvd-ci: requires torch
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID, pixel_rays, project, rig_cams  # noqa: E402
from python.perception.occ_out import OccMemory  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_map_labels import MapRoad, build_map_labels, merge_targets, visible_keys  # noqa: E402
from python.train.occ_train import occ_loss, prep_image, sample_queries  # noqa: E402
import synthetic_scenes as ss  # noqa: E402
from synthetic_occ import GROUND_Z  # noqa: E402

G = DEFAULT_GRID
CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
CAM_IDS = ("main", "wide")
RENDER, NET = 128, 128
STEP_M = 8.0  # longer than the leg in y; after this step the patch is behind every camera
# s4's hidden-leg rectangle and occluder, in world metres (right of the main road)
LEG = {"x0": 2.5, "x1": 12.0, "y0": 7.0, "y1": 9.0}  # short in y; a 8 m step puts it behind the cameras
# Gray asphalt vs grass is only ~1% of a 128 px frame and dies in the stride-8
# encoder (4-5 feature cells, tiny activation). This BGR paint keeps the same
# footprint but is visible to the cheap net so tick-0 A vs B class can train.
LEG_BGR = np.array([0, 230, 255], np.uint8)
HID_CLS_W = 2.0  # extra CE on the leg-rectangle queries (same labels as below)
OCC = ((2.5, 8.0, 0.0), (14.0, 11.0, 4.0))  # in front of the leg; far enough that t1 camera is not inside it
# Held-out starts are > one 0.4 m voxel from every train start.
TRAIN_STARTS = (0.0, 1.0, 2.0)
EVAL_STARTS = (0.5, 1.5)  # 0.5 m > one 0.4 m voxel from every train start
STEPS = 160
DT = 0.1    # OccRuntime dt; vehicle ego recovers STEP_M metres in this interval
# High-contrast paint on the tick-1 wall so the cheap stride-8 encoder can see it
# (the default building grey is almost invisible on main and muted on wide).
OCC_BGR = np.array([0, 0, 255], np.uint8)
P_WALL = 0.0  # t1 is behind the cameras; no wall overfit
GATE_J = 0.5
GATE_GAP = 0.3


def cams():
    rc = {c.cam_id: c.scaled(RENDER) for c in rig_cams(CFG) if c.cam_id in CAM_IDS}
    nc = [c.scaled(NET) for c in rig_cams(CFG) if c.cam_id in CAM_IDS]
    return rc, nc


def world(leg: bool, occluder: bool) -> ss.Scene:
    roads = [MapRoad(np.array([[0.0, -20.0], [0.0, 80.0]]), 7.0)]
    if leg:
        roads.append(MapRoad(np.array([[LEG["x0"], (LEG["y0"] + LEG["y1"]) / 2.0],
                                       [LEG["x1"], (LEG["y0"] + LEG["y1"]) / 2.0]]),
                             LEG["y1"] - LEG["y0"]))
    boxes = [OCC] if occluder else []
    return ss.Scene("memgap", roads, boxes, 0.0)


def frame(ego_y: float, leg: bool, occluder: bool, rcams):
    sc = world(leg, occluder).shifted(ego_y)
    rend = {cid: ss.render(cam, sc) for cid, cam in rcams.items()}
    planes = {cid: r[:2] for cid, r in rend.items()}
    lab = build_occ_labels(planes, rcams, sem_classes=ss.SEM_CLASSES, stride=1)
    vis = visible_keys(planes, rcams, stride=1)
    ml = build_map_labels(sc.roads, ground_z=GROUND_Z)
    keys, cls = merge_targets(lab.keys, lab.cls, ml)
    depths = {cid: r[0] for cid, r in rend.items()}
    images = {cid: r[2] for cid, r in rend.items()}
    if leg:
        images = _paint_leg(images, depths, rcams, ego_y)
    if occluder:
        images = _paint_occ(images, {cid: r[1] for cid, r in rend.items()})
    return {"images": images, "depths": depths, "keys": keys, "cls": cls, "free": lab.free, "vis": vis, "ml": ml}


def _paint_leg(images, depths, rcams, ego_y):
    """Recolor ground pixels inside the LEG rectangle (ego frame at ``ego_y``)."""
    out = {cid: img.copy() for cid, img in images.items()}
    for cam in rcams.values():
        img, depth = out[cam.cam_id], depths[cam.cam_id]
        vs, us = np.mgrid[0:cam.height, 0:cam.width]
        rays = pixel_rays(cam, us.reshape(-1), vs.reshape(-1))
        origin = np.asarray(cam.pos, dtype=np.float64)
        t_m = depth.reshape(-1).astype(np.float64) / 100.0
        pts = origin + rays * t_m[:, None]
        valid = (depth.reshape(-1) > 0) & (depth.reshape(-1) < 65535)
        hit = (valid
               & (pts[:, 0] >= LEG["x0"]) & (pts[:, 0] <= LEG["x1"])
               & (pts[:, 1] >= LEG["y0"] - ego_y) & (pts[:, 1] <= LEG["y1"] - ego_y)
               & (np.abs(pts[:, 2] - GROUND_Z) < 0.6))
        if hit.any():
            img.reshape(-1, 3)[hit] = LEG_BGR
    return out


def _paint_occ(images, sems):
    """Recolor building / occluder pixels so the wall is visible to the cheap net."""
    from synthetic_scenes import SEM_BUILDING

    out = {cid: img.copy() for cid, img in images.items()}
    for cid, img in out.items():
        m = sems[cid] == SEM_BUILDING
        if m.any():
            img[m] = OCC_BGR
    return out


def occ_box_keys(ego_y: float) -> np.ndarray:
    """Elevated cells inside the tick-1 occluder, in the ego frame at ``ego_y``."""
    (x0, y0, z0), (x1, y1, z1) = OCC
    y0, y1 = y0 - ego_y, y1 - ego_y
    xs = np.arange(x0 + 0.2, x1, G.voxel_m)
    ys = np.arange(y0 + 0.2, y1, G.voxel_m)
    zs = np.arange(max(z0, G.z_min) + 0.2, z1, G.voxel_m)
    if not (len(xs) and len(ys) and len(zs)):
        return np.zeros(0, np.int64)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
    k = G.keys_of(pts)
    return np.unique(k[k >= 0])


def occ_face_keys(ego_y: float) -> np.ndarray:
    """One-voxel front slab of the occluder (the face that blocks camera rays)."""
    (x0, y0, z0), (x1, _y1, z1) = OCC
    y = y0 + 0.5 * G.voxel_m - ego_y
    xs = np.arange(x0 + 0.2, x1, G.voxel_m)
    zs = np.arange(max(z0, 0.2) + 0.2, z1, G.voxel_m)
    if not (len(xs) and len(zs)):
        return np.zeros(0, np.int64)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    pts = np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], 1)
    k = G.keys_of(pts)
    return np.unique(k[k >= 0])


def wall_hit_keys(ego_y: float, rcams) -> np.ndarray:
    """Cells where camera rays to the t1 leg rectangle meet the wall front plane."""
    (x0, y0, z0), (x1, _y1, z1) = OCC
    plane_y = y0 + 0.5 * G.voxel_m - ego_y
    base = leg_keys(ego_y)
    if not base.size:
        return base
    pts = G.centers_of_keys(base)
    hits = []
    for cam in rcams.values():
        o = np.asarray(cam.pos, np.float64)
        d = pts - o
        den = np.where(np.abs(d[:, 1]) < 1e-9, 1e-9, d[:, 1])
        typ = (plane_y - o[1]) / den
        hit = o + typ[:, None] * d
        ok = (typ > 0.0) & (typ < 1.0) & (hit[:, 0] >= x0) & (hit[:, 0] <= x1) & (hit[:, 2] >= z0) & (hit[:, 2] <= z1)
        if ok.any():
            hits.append(hit[ok])
    if not hits:
        return np.zeros(0, np.int64)
    k = G.keys_of(np.concatenate(hits))
    return np.unique(k[k >= 0])


def above_leg_keys(ego_y: float, n_z: int = 4) -> np.ndarray:
    """Empty air immediately above the leg rectangle (must not occlude the ground cells)."""
    base = leg_keys(ego_y)
    if not base.size:
        return base
    c = G.centers_of_keys(base)
    ks = [G.keys_of(c + np.array([0.0, 0.0, i * G.voxel_m])) for i in range(1, n_z + 1)]
    k = np.unique(np.concatenate(ks))
    return k[k >= 0]


def leg_keys(ego_y: float) -> np.ndarray:
    """Ground-layer keys of the axis-aligned leg rectangle in the ego frame at ``ego_y``.

    Built from the rectangle itself (not a MapRoad stadium): the labeler would
    grow a 6 m-wide centreline into the main road and those cells are never hidden.
    """
    ml = build_map_labels(
        [MapRoad(np.array([[LEG["x0"], (LEG["y0"] + LEG["y1"]) / 2.0 - ego_y],
                           [LEG["x1"], (LEG["y0"] + LEG["y1"]) / 2.0 - ego_y]]),
                 LEG["y1"] - LEG["y0"])],
        ground_z=GROUND_Z,
    )
    c = G.centers_of_keys(ml.drivable_keys)
    m = ((c[:, 0] >= LEG["x0"]) & (c[:, 0] <= LEG["x1"]) &
         (c[:, 1] >= LEG["y0"] - ego_y) & (c[:, 1] <= LEG["y1"] - ego_y))
    return ml.drivable_keys[m]


def depth_seen(rcams, depths, pts: np.ndarray) -> np.ndarray:
    """True where some camera's depth is within 0.5 m of the point (s4 _seen)."""
    vis = np.zeros(len(pts), bool)
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    for cam in rcams.values():
        uv, z, ok = project(cam, pts)
        d = depths[cam.cam_id]
        u = np.clip(np.nan_to_num(uv[:, 0]).astype(np.int64), 0, cam.width - 1)
        v = np.clip(np.nan_to_num(uv[:, 1]).astype(np.int64), 0, cam.height - 1)
        hit = d[v, u].astype(np.float64) / 100.0  # recorder depth is uint16 cm
        valid = (d[v, u] > 0) & (d[v, u] < 65535)
        vis |= ok & valid & np.isfinite(hit) & (np.abs(hit - z) < 0.5)
    return vis


def hidden_leg(rcams, start: float) -> np.ndarray:
    """Tick-1 ego-frame leg keys visible at tick 0 and outside every camera frustum at tick 1.

    After STEP_M the patch is behind the cameras, so observed_mask is false
    with no predicted occluder and no forced flag. OccRuntime memory is then
    the only way to label those cells.
    """
    from python.perception.occ_geometry import project

    f0 = frame(start, True, False, rcams)
    k0, k1 = leg_keys(start), leg_keys(start + STEP_M)
    s0 = depth_seen(rcams, f0["depths"], G.centers_of_keys(k0))
    pts1 = G.centers_of_keys(k1)
    in_f = np.zeros(len(pts1), bool)
    for cam in rcams.values():
        _, _, ok = project(cam, pts1)
        in_f |= ok
    # keys are a rigid +y shift of one voxel grid; pair by warp of centres
    warped = G.keys_of(G.centers_of_keys(k0) - np.array([0.0, STEP_M, 0.0]))
    keep = s0 & np.isin(warped, k1[~in_f])
    hid = warped[keep]
    hid = hid[hid >= 0]
    if hid.size < 30:
        raise AssertionError(f"start {start}: only {hid.size} hidden-and-seen-early leg cells "
                             f"(tick-0 visible {int(s0.sum())}, tick-1 in frustum {int(in_f.sum())})")
    return hid


def query_cells(net, ncams, images, keys, lift_zero=False):
    """Run the net at cell centres; return keys, cls, prob for occupied predictions."""
    import torch

    if keys.size == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.float32))
    pts = G.centers_of_keys(keys).astype(np.float32)
    im = [torch.from_numpy(prep_image(images[c.cam_id], c))[None] for c in ncams]
    orig = net.query
    if lift_zero:
        def _q(feats, p, cam_w=None):
            return orig([f * 0 for f in feats], p, cam_w)
        net.query = _q
    try:
        with torch.no_grad():
            out = net(im, torch.from_numpy(pts)[None])
    finally:
        net.query = orig
    import torch as _t
    prob = _t.sigmoid(out["occ"][0].float()).cpu().numpy()
    cls = out["cls"][0].float().argmax(-1).cpu().numpy().astype(np.uint8)
    return keys.astype(np.int64), cls, prob.astype(np.float32)


def _air_free(keys: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Above-ground empty cell centres (true free air, not occluded structure)."""
    if n <= 0:
        return np.zeros((0, 3), np.float32)
    nx, ny, nz = G.shape
    gz = int(np.floor((0.0 - G.z_min) / float(G.voxel_m)))
    z0 = min(nz, gz + 2)
    if z0 >= nz:
        return np.zeros((0, 3), np.float32)
    n_draw = int(n) * 6
    cell = np.stack([
        rng.integers(0, nx, size=n_draw),
        rng.integers(0, ny, size=n_draw),
        rng.integers(z0, nz, size=n_draw),
    ], axis=1)
    cand = G.key_of_cell(cell)
    occ = np.asarray(keys, np.int64)
    if occ.size:
        cand = cand[~np.isin(cand, occ)]
    cand = cand[cand >= 0]
    if cand.size == 0:
        return np.zeros((0, 3), np.float32)
    if cand.size > n:
        cand = cand[:n]
    return G.centers_of_keys(cand).astype(np.float32)


def train_visible(rcams, ncams, *, steps: int, lift_zero: bool = False, drop_hid: bool = False):
    """Overfit the query head on tick-0 (leg visible, no occluder) at the train starts."""
    import torch

    from python.perception.occ_sparse import SparseOccNet

    torch.manual_seed(0)
    torch.set_num_threads(4)
    # Camera-only overfit: unseen decode would fill phantom occluders and
    # mark the visible leg unobserved, so memory would never see it.
    net = SparseOccNet(ncams, config="cheap", decode_unseen=False)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    cache = {}
    for s in TRAIN_STARTS:
        for leg in (True, False):
            cache[(s, leg, 0)] = frame(s, leg, False, rcams)
            cache[(s, leg, 1)] = frame(s + STEP_M, leg, True, rcams)
    rng = np.random.default_rng(0)
    hist = []
    net.train()
    orig = net.query
    if lift_zero:
        def _q(feats, p, cam_w=None):
            return orig([f * 0 for f in feats], p, cam_w)
        net.query = _q
    try:
        for it in range(steps):
            s = TRAIN_STARTS[int(rng.integers(len(TRAIN_STARTS)))]
            leg = bool(rng.integers(2))
            # Tick 0 for A vs B class. Occcluder queries are added separately below
            # so we do not flood the head with elevated "other".
            # Learn A vs B first; mixing the red wall from step 0 collapses class.
            late = it >= int(steps * 0.75)
            wall = bool(late and P_WALL > 0 and rng.random() < P_WALL)

            tick = 1 if wall else 0
            f = cache[(s, leg, tick)]
            qrng = np.random.default_rng((it, int(s * 10), int(leg), tick))
            if wall:
                # Wall-only batch on the tick-1 image: do not sample the rest of
                # the t1 scene (that flooded t0 with phantom occluders).
                box = np.unique(np.concatenate([
                    occ_face_keys(s + STEP_M),
                    wall_hit_keys(s + STEP_M, rcams),
                ]))
                pts_b = G.centers_of_keys(box).astype(np.float32) if box.size else np.zeros((0, 3), np.float32)
                n_b = int(pts_b.shape[0])
                nlev = len(__import__("python.perception.occ_sparse", fromlist=["LEVELS"]).LEVELS)
                q = {
                    "pts": pts_b,
                    "occ": np.ones(n_b, np.float32),
                    "occ_w": np.ones(n_b, np.float32),
                    "cls": np.zeros(n_b, np.int64),
                    "any": np.ones((n_b, nlev), np.float32),
                    "any_w": np.ones((n_b, nlev), np.float32),
                }
                hits = wall_hit_keys(s + STEP_M, rcams)
                if hits.size and n_b:
                    q["occ_w"][np.isin(box, hits)] = 4.0
            else:
                q = sample_queries(f["keys"], f["cls"], f["free"], qrng,
                                   vis_keys=f["vis"], n_occ=256, n_free=256, n_coarse=64)
                air = _air_free(f["keys"], qrng, 256)
                # Contrast: the would-be wall face and the air above the leg are
                # empty at tick 0 (the wall has not appeared). This keeps t0
                # observed and makes the wall image-conditional.
                ghost = np.unique(np.concatenate([
                    occ_face_keys(s),
                    wall_hit_keys(s, rcams),
                ]))
                above = above_leg_keys(s)
                extra_neg = [air] if air.size else []
                if ghost.size:
                    extra_neg.append(G.centers_of_keys(ghost).astype(np.float32))
                if above.size:
                    extra_neg.append(G.centers_of_keys(above).astype(np.float32))
                if extra_neg:
                    neg = np.concatenate(extra_neg).astype(np.float32)
                    n_a = int(neg.shape[0])
                    q["pts"] = np.concatenate([q["pts"], neg])
                    q["occ"] = np.concatenate([q["occ"], np.zeros(n_a, np.float32)])
                    q["occ_w"] = np.concatenate([q["occ_w"], np.ones(n_a, np.float32)])
                    q["cls"] = np.concatenate([q["cls"], np.full(n_a, -1, np.int64)])
                    q["any"] = np.concatenate([q["any"], np.zeros((n_a, q["any"].shape[1]), np.float32)])
                    q["any_w"] = np.concatenate([q["any_w"], np.ones((n_a, q["any"].shape[1]), np.float32)])
            # s4: also put the (visible) tick-0 leg rectangle in the batch so the A/B
            # class difference is actually trained, not swamped by the main road
            hid0 = np.zeros(0, np.int64) if (drop_hid or tick != 0) else leg_keys(s)
            if hid0.size:
                extra = G.centers_of_keys(hid0)
                n = extra.shape[0]
                # labels: occupied; class is 1 iff this episode has the leg
                q["pts"] = np.concatenate([q["pts"], extra.astype(np.float32)])
                q["occ"] = np.concatenate([q["occ"], np.ones(n, np.float32)])
                q["occ_w"] = np.concatenate([q["occ_w"], np.ones(n, np.float32)])
                q["cls"] = np.concatenate([q["cls"], np.full(n, 1 if leg else 0, np.int64)])
                any_extra = np.ones((n, q["any"].shape[1]), np.float32)
                q["any"] = np.concatenate([q["any"], any_extra])
                q["any_w"] = np.concatenate([q["any_w"], np.ones_like(any_extra)])
            batch = {k: torch.from_numpy(np.stack([q[k]])) for k in ("pts", "occ", "occ_w", "any", "any_w", "cls")}
            batch["images"] = [torch.from_numpy(prep_image(f["images"][c.cam_id], c))[None] for c in ncams]
            out = net(batch["images"], batch["pts"])
            loss, stats = occ_loss(out, batch)
            if hid0.size:
                import torch.nn.functional as F
                l_hid = F.cross_entropy(
                    out["cls"][0, -n:].float(),
                    torch.full((n,), 1 if leg else 0, dtype=torch.int64),
                )
                loss = loss + HID_CLS_W * l_hid
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hist.append(stats)
    finally:
        net.query = orig
    net.eval()
    return net, hist


def _call(keys, cls, prob, hid):
    """Fraction of ``hid`` called drivable."""
    if hid.size == 0 or keys.size == 0:
        return 0.0
    order = np.argsort(keys)
    ks, cs, ps = keys[order], cls[order], prob[order]
    pos = np.clip(np.searchsorted(ks, hid), 0, ks.size - 1)
    hit = ks[pos] == hid
    return float(((hit) & (cs[pos] == 1) & (ps[pos] >= 0.5)).mean())


def _images(ncams, images):
    import torch

    from python.train.occ_train import prep_image

    return [torch.from_numpy(prep_image(images[c.cam_id], c))[None] for c in ncams]


def _apply_lift(net):
    orig = net.query

    def _q(feats, p, cam_w=None):
        return orig([f * 0 for f in feats], p, cam_w)

    net.query = _q
    return orig


def evaluate(net, ncams, hid, cache, *, memory: bool, lift_zero: bool = False, zero_warp: bool = False):
    """recall, false_alarm, J over held-out starts, via OccRuntime (predict + cap + observed + memory)."""
    from python.perception.occ_out import OccRuntime

    rec, fa, vo = [], [], []
    orig = _apply_lift(net) if lift_zero else None
    # Official OccRuntime vehicle-ego path: a known warp, not GT keys or a forced observed flag.
    speed = 0.0 if zero_warp else (STEP_M / DT)
    vehicle = {"wheelspeed_mps": speed, "imu_yaw_rate": 0.0}
    try:
        for s in EVAL_STARTS:
            h = hid[s]
            for leg, bucket in ((True, rec), (False, fa)):
                f0, f1 = cache[(s, leg)]
                # One-tick cap would miss a side-road that the direct query already
                # classifies; warm the coarse sweep / hot carry on tick-0 images
                # (production would have prior ticks) then step OccRuntime.
                im0, im1 = _images(ncams, f0["images"]), _images(ncams, f1["images"])
                import torch
                net.eval()
                with torch.no_grad():
                    for _ in range(4):
                        net.predict(im0, max_points=65536, max_coarse=16384)
                rt = OccRuntime(net, from_vehicle=True, memory=memory, max_points=65536, max_coarse=16384,
                                timer=lambda: 0.0)
                rt.step(im0, 0.0, vehicle=vehicle)
                msg1 = rt.step(im1, DT, vehicle=vehicle)
                keys, cls, prob, _obs = rt.last_cells
                ego = msg1.get("ego") or {}
                vo.append({"ok": bool(ego.get("ok")), "source": ego.get("source"),
                           "speed": float(ego.get("speed_mps") or 0.0),
                           "fwd": float(ego.get("speed_mps") or 0.0) * DT})
                bucket.append(_call(keys, cls, prob, h))
    finally:
        if orig is not None:
            net.query = orig
    recall, false_alarm = float(np.mean(rec)), float(np.mean(fa))
    return {"recall": recall, "false_alarm": false_alarm, "J": recall - false_alarm, "vo": vo}


def _fmt(tag, on, off) -> str:
    return (f"{tag}: memory on recall {on['recall']:.3f} false-alarm {on['false_alarm']:.3f} J {on['J']:.3f} | "
            f"memory off recall {off['recall']:.3f} false-alarm {off['false_alarm']:.3f} J {off['J']:.3f} | "
            f"gap {on['J'] - off['J']:+.3f}")


def _gate(on, off) -> bool:
    return on["J"] >= GATE_J and (on["J"] - off["J"]) >= GATE_GAP


def check_memory_gap() -> None:
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    rcams, ncams = cams()
    t0 = time.perf_counter()
    hid = {s: hidden_leg(rcams, s) for s in TRAIN_STARTS + EVAL_STARTS}
    gap = min(abs(a - b) for a in EVAL_STARTS for b in TRAIN_STARTS)
    assert gap > G.voxel_m, (gap, G.voxel_m)
    print(f"A0 scene: {len(TRAIN_STARTS)} train / {len(EVAL_STARTS)} held-out starts x leg on/off; "
          f"hidden leg cells per start {min(h.size for h in hid.values())}-{max(h.size for h in hid.values())} "
          f"(visible at tick 0, hidden in all {len(rcams)} cams at tick 1)")
    cache = {}
    for s in EVAL_STARTS:
        for leg in (True, False):
            cache[(s, leg)] = (frame(s, leg, False, rcams), frame(s + STEP_M, leg, False, rcams))

    net, hist = train_visible(rcams, ncams, steps=STEPS)
    first = float(np.mean([h["loss"] for h in hist[:5]]))
    last = float(np.mean([h["loss"] for h in hist[-5:]]))
    print(f"train: steps={STEPS} loss {first:.3f} -> {last:.3f} in {time.perf_counter() - t0:.1f}s")

    def _vo(tag, res):
        ok = [v["ok"] for v in res.get("vo") or []]
        fwds = [v["fwd"] for v in res.get("vo") or []]
        if fwds:
            print(f"{tag} ego: ok {sum(ok)}/{len(ok)}, fwd m {min(fwds):.2f}..{max(fwds):.2f} (true {STEP_M})")

    on = evaluate(net, ncams, hid, cache, memory=True)
    off = evaluate(net, ncams, hid, cache, memory=False)
    _vo("baseline on", on)
    _vo("baseline off", off)
    print(_fmt("baseline", on, off) + f" | gate {'PASS' if _gate(on, off) else 'FAIL'}")
    assert _gate(on, off), f"baseline fails the memory gate: on={on} off={off}"

    # Mutants: each must fail the same gate (s4 A2). Lift is retrained with features zeroed.
    lift_net, _ = train_visible(rcams, ncams, steps=STEPS, lift_zero=True)
    abl_on = evaluate(lift_net, ncams, hid, cache, memory=True, lift_zero=True)
    abl_off = evaluate(lift_net, ncams, hid, cache, memory=False, lift_zero=True)
    print(_fmt("mutant lift ablated (retrained)", abl_on, abl_off))
    assert not _gate(abl_on, abl_off), f"lift-ablated mutant still passed the gate: {abl_on} {abl_off}"
    print("mutant lift ablated: gate broken OK")

    print(_fmt("mutant memory disabled", off, off))
    assert not _gate(off, off), "memory-disabled mutant should not pass the gate"
    print("mutant memory disabled: gate broken OK")

    zw_on = evaluate(net, ncams, hid, cache, memory=True, zero_warp=True)
    zw_off = off
    print(_fmt("mutant zero warp", zw_on, zw_off))
    assert not _gate(zw_on, zw_off), f"zero-warp mutant still passed the gate: {zw_on} {zw_off}"
    print("mutant zero warp: gate broken OK")

    # Loss ablation: drop the visible-leg queries from the training loss (s4 M4 analogue).
    drop_net, _ = train_visible(rcams, ncams, steps=STEPS, drop_hid=True)
    drop_on = evaluate(drop_net, ncams, hid, cache, memory=True)
    drop_off = evaluate(drop_net, ncams, hid, cache, memory=False)
    print(_fmt("mutant loss ablated (no hid queries)", drop_on, drop_off))
    assert not _gate(drop_on, drop_off), f"loss-ablation mutant still passed the gate: {drop_on} {drop_off}"
    print("mutant loss ablated: gate broken OK")


def main() -> int:
    check_memory_gap()
    print("test_occ_memory_gap: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
