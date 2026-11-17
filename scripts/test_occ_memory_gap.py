#!/usr/bin/env python3
"""Hidden-leg memory proof for s2 OccMemory, ported from swarm s4's protocol.

s4 (gvd/occ-sparse-s4, scripts/test_occ_ablation.py) isolated memory from VO:
two ticks, an occluder that appears at tick 1, true synthetic motion for the
warp, a Youden J = recall - false-alarm gate on hidden leg cells, and
in-process mutants that must fail the gate. That test targeted s4's
differentiable feature memory (python.occ). This port uses the same protocol
on s2's APIs: SparseOccNet queries + OccMemory (key/prob store, camera-only
runtime). Motion is injected as the true (0, STEP_M, 0) ego shift; a VO-driven
run is printed but not gated (VO has its own tests in test_occ_ego.py).

Scene (world metres, ego drives +y): a straight road, an optional right-hand
asphalt leg, and at tick 1 only a wide occluder face between the cameras and
the leg. The visible leg is painted a high-contrast colour (not the default
gray asphalt) so the cheap stride-8 encoder at 128 px can tell it from grass;
the thing under test is still OccMemory, not color discrimination. Tick 0: the scored cells are visible. Tick 1: they are hidden in
every camera (checked by depth). Whether the leg exists at tick 1 is then
only knowable from tick 0, i.e. from OccMemory.

Score at tick 1 on those hidden cells, on held-out start positions:

    recall      = fraction called drivable (occ prob >= 0.5 and cls == 1) when the leg is present
    false_alarm = the same fraction when the leg is absent
    J           = recall - false_alarm

GATE (fixed before the first run): J(memory on) >= 0.5 and
J(memory on) - J(memory off) >= 0.3.

Mutants (each must fail the gate):
* lift ablated: camera features zeroed in the decoder (eval)
* memory disabled: OccMemory not applied
* zero warp: memory is updated with fwd=0 (the s4/s2 failure mode where VO
  locks to an ego-fixed sheet and the carried leg misses its tick-1 cell)
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
STEP_M = 2.4  # longer than the leg in y, so the zero-warp mutant cannot overlap the scored cells
# s4's hidden-leg rectangle and occluder, in world metres (right of the main road)
LEG = {"x0": 2.5, "x1": 14.0, "y0": 11.2, "y1": 13.2}  # short in y so a 2.4 m step leaves no zero-warp overlap
# Gray asphalt vs grass is only ~1% of a 128 px frame and dies in the stride-8
# encoder (4-5 feature cells, tiny activation). This BGR paint keeps the same
# footprint but is visible to the cheap net so tick-0 A vs B class can train.
LEG_BGR = np.array([0, 230, 255], np.uint8)
HID_CLS_W = 2.0  # extra CE on the leg-rectangle queries (same labels as below)
OCC = ((2.5, 5.0, 0.0), (14.0, 8.0, 4.0))  # wide face in front of the leg; appears at tick 1 only
TRAIN_STARTS = (0.0, 0.5, 1.0, 1.5, 2.0)
EVAL_STARTS = (0.25, 0.75, 1.25, 1.75)
STEPS = 160
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
    """Tick-1 ego-frame leg keys visible at tick 0 and hidden in every camera at tick 1."""
    f0 = frame(start, True, False, rcams)
    f1 = frame(start + STEP_M, True, True, rcams)
    k0, k1 = leg_keys(start), leg_keys(start + STEP_M)
    s0 = depth_seen(rcams, f0["depths"], G.centers_of_keys(k0))
    s1 = depth_seen(rcams, f1["depths"], G.centers_of_keys(k1))
    # keys are a rigid +y shift of one voxel grid; pair by warp of centres
    warped = G.keys_of(G.centers_of_keys(k0) - np.array([0.0, STEP_M, 0.0]))
    keep = s0 & np.isin(warped, k1[~s1])
    hid = warped[keep]
    hid = hid[hid >= 0]
    # the scored set is hidden by construction; a few rectangle edge cells may still
    # peek around the occluder and are simply left out of the gate
    if hid.size < 30:
        raise AssertionError(f"start {start}: only {hid.size} hidden-and-seen-early leg cells "
                             f"(tick-0 visible {int(s0.sum())}, tick-1 still visible {int(s1.sum())})")
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


def train_visible(rcams, ncams, *, steps: int, lift_zero: bool = False):
    """Overfit the query head on tick-0 (leg visible, no occluder) at the train starts."""
    import torch

    from python.perception.occ_sparse import SparseOccNet

    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    net = SparseOccNet(ncams, config="cheap")
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    cache = {}
    for s in TRAIN_STARTS:
        for leg in (True, False):
            cache[(s, leg)] = frame(s, leg, False, rcams)
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
            f = cache[(s, leg)]
            q = sample_queries(f["keys"], f["cls"], f["free"], np.random.default_rng((it, int(s * 10), int(leg))),
                               vis_keys=f["vis"], n_occ=256, n_free=256, n_coarse=64)
            # s4: also put the (visible) leg rectangle in the batch so the A/B
            # class difference is actually trained, not swamped by the main road
            hid0 = leg_keys(s)
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


def evaluate(net, ncams, rcams, hid, cache, *, memory: bool, lift_zero: bool = False, fwd: float = STEP_M):
    """recall, false_alarm, J over held-out starts. ``fwd`` is the memory warp (metres)."""
    rec, fa = [], []
    for s in EVAL_STARTS:
        h = hid[s]
        for leg, bucket in ((True, rec), (False, fa)):
            f0, f1 = cache[(s, leg)]
            # Tick 0 memory is only the observed-leg support (hid cells warped back).
            # Including the whole map would also store the net's spatially smeared
            # class around the leg, so a zero warp could still land on hid and
            # the mutant would not test the warp.
            k0 = G.keys_of(G.centers_of_keys(h) + np.array([0.0, STEP_M, 0.0]))
            k0 = k0[k0 >= 0]
            keys, cls, prob = query_cells(net, ncams, f0["images"], k0, lift_zero=lift_zero)
            # geometric observed (tick 0: the hid cells were visible)
            pts = G.centers_of_keys(keys) if keys.size else np.zeros((0, 3))
            obs = depth_seen(rcams, f0["depths"], pts) if keys.size else np.zeros(0, bool)
            if memory:
                mem = OccMemory(G)
                mem.step(keys, cls, prob, obs, ok=True, fwd=0.0, dyaw=0.0, dt=None, cams=[], occl=np.zeros(0, np.int64))
                k1 = np.unique(np.concatenate([f1["keys"], h]))
                k1 = k1[k1 >= 0]
                keys1, cls1, prob1 = query_cells(net, ncams, f1["images"], k1, lift_zero=lift_zero)
                pts1 = G.centers_of_keys(keys1) if keys1.size else np.zeros((0, 3))
                obs1 = depth_seen(rcams, f1["depths"], pts1) if keys1.size else np.zeros(0, bool)
                # hid cells are geometrically hidden: force unobserved so memory is kept
                if keys1.size and h.size:
                    obs1[np.isin(keys1, h)] = False
                keys, cls, prob, obs = mem.step(
                    keys1, cls1, prob1, obs1, ok=True, fwd=float(fwd), dyaw=0.0, dt=0.1, cams=[], occl=np.zeros(0, np.int64),
                )
            else:
                k1 = np.unique(np.concatenate([f1["keys"], h]))
                k1 = k1[k1 >= 0]
                keys, cls, prob = query_cells(net, ncams, f1["images"], k1, lift_zero=lift_zero)
            bucket.append(_call(keys, cls, prob, h))
    recall, false_alarm = float(np.mean(rec)), float(np.mean(fa))
    return {"recall": recall, "false_alarm": false_alarm, "J": recall - false_alarm}


def _fmt(tag, on, off) -> str:
    return (f"{tag}: memory on recall {on['recall']:.3f} false-alarm {on['false_alarm']:.3f} J {on['J']:.3f} | "
            f"memory off recall {off['recall']:.3f} false-alarm {off['false_alarm']:.3f} J {off['J']:.3f} | "
            f"gap {on['J'] - off['J']:+.3f}")


def _gate(on, off) -> bool:
    return on["J"] >= GATE_J and (on["J"] - off["J"]) >= GATE_GAP


def check_memory_gap() -> None:
    rcams, ncams = cams()
    t0 = time.perf_counter()
    hid = {s: hidden_leg(rcams, s) for s in TRAIN_STARTS + EVAL_STARTS}
    print(f"A0 scene: {len(TRAIN_STARTS)} train / {len(EVAL_STARTS)} held-out starts x leg on/off; "
          f"hidden leg cells per start {min(h.size for h in hid.values())}-{max(h.size for h in hid.values())} "
          f"(visible at tick 0, hidden in all {len(rcams)} cams at tick 1)")
    cache = {}
    for s in EVAL_STARTS:
        for leg in (True, False):
            cache[(s, leg)] = (frame(s, leg, False, rcams), frame(s + STEP_M, leg, True, rcams))

    net, hist = train_visible(rcams, ncams, steps=STEPS)
    first = float(np.mean([h["loss"] for h in hist[:5]]))
    last = float(np.mean([h["loss"] for h in hist[-5:]]))
    print(f"train: steps={STEPS} loss {first:.3f} -> {last:.3f} in {time.perf_counter() - t0:.1f}s")

    on = evaluate(net, ncams, rcams, hid, cache, memory=True)
    off = evaluate(net, ncams, rcams, hid, cache, memory=False)
    print(_fmt("baseline", on, off) + f" | gate {'PASS' if _gate(on, off) else 'FAIL'}")
    assert _gate(on, off), f"baseline fails the memory gate: on={on} off={off}"

    # Mutants: each must fail the same gate (s4 A2).
    abl_on = evaluate(net, ncams, rcams, hid, cache, memory=True, lift_zero=True)
    abl_off = evaluate(net, ncams, rcams, hid, cache, memory=False, lift_zero=True)
    print(_fmt("mutant lift ablated", abl_on, abl_off))
    assert not _gate(abl_on, abl_off), f"lift-ablated mutant still passed the gate: {abl_on} {abl_off}"
    print("mutant lift ablated: gate broken OK")

    print(_fmt("mutant memory disabled", off, off))
    assert not _gate(off, off), "memory-disabled mutant should not pass the gate"
    print("mutant memory disabled: gate broken OK")

    zw_on = evaluate(net, ncams, rcams, hid, cache, memory=True, fwd=0.0)
    zw_off = off
    print(_fmt("mutant zero warp", zw_on, zw_off))
    assert not _gate(zw_on, zw_off), f"zero-warp mutant still passed the gate: {zw_on} {zw_off}"
    print("mutant zero warp: gate broken OK")


def main() -> int:
    check_memory_gap()
    print("test_occ_memory_gap: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
