#!/usr/bin/env python3
# gvd-ci: requires torch
"""Sparse occupancy net: hidden-road-leg memory proof, with ablation mutants that must fail it.

Critic rule (Skeptic): a synthetic proof must be one a broken model fails.
Overfit the real sparse net (``OccSparseNet``, no test-only model) on a tiny
synthetic junction: the ego drives forward on the straight road; in each
episode a side-road leg is either present (asphalt) or absent (grass). Tick 0:
no occluder, the leg area is in view. Tick 1 (ego +0.5 m): a wide occluder
face stands between the forward cameras and the leg, so every leg cell used
in the gate is geometrically hidden in every camera (checked by ray depth).
Whether the leg exists at tick 1 is then only knowable from tick 0, i.e. from
the temporal memory.

Score at tick 1 on the hidden leg cells, on held-out start positions:
road call = occ prob > 0.5 and argmax class == ROAD;
recall = road calls on leg-present episodes, false alarm = road calls on
leg-absent episodes, J = recall - false alarm (Youden).

GATE (fixed before the first run): J(memory on) >= 0.5 and
J(memory on) - J(memory off) >= 0.3, where memory off = the same trained
weights run with ``memory=None`` at tick 1.

MUTANTS (each retrains from the same seed with the ablation in place, then
runs the same gate; the gate must FAIL for each = mutant killed):
  M1 lift_off        the lift returns no points (no observed cells, nothing to remember)
  M2 lift_feat_zero  lifted point features zeroed (positions kept; pooled camera features remain)
  M3 memory_off      the memory argument is dropped inside every tick (train and eval)
  M4 loss_occluded   hidden (occluded) leg cells dropped from the training loss

Motion between the two ticks is supplied (true synthetic motion, (0, 0.5, 0)):
this isolates memory + warp from VO, which has its own tests in
test_occ_sparse_net.py; a VO-driven run is printed but not gated.
Grass ground keeps class UNKNOWN as a target here (index 0) instead of the
production "ignore" so "not road" is learnable; documented test-only change.
CPU only, seeded, one intra-op thread.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from occ_net_fixtures import small_profile, small_rig  # noqa: E402
from python.occ import classes as C  # noqa: E402
from python.occ.grid import VoxelGrid  # noqa: E402
from python.occ.labels import CamFrame, voxel_labels  # noqa: E402
from python.occ.model import OccSparseNet, make_queries, occ_loss  # noqa: E402
from python.occ.synthetic import SEM_CLASSES, scene  # noqa: E402

torch.set_num_threads(1)
G = VoxelGrid()
IDS = ("main", "wide", "narrow")
LS = int(os.environ.get("GVD_ABL_LS", "128"))
STEPS = int(os.environ.get("GVD_ABL_STEPS", "160"))
BATCH = 2
BINS = int(os.environ.get("GVD_ABL_BINS", "1"))
STEP_M = 0.5                                       # ego forward motion between tick 0 and tick 1
LEG = {"x0": 3.5, "x1": 11.0, "y0": 14.0, "y1": 20.0}   # side-road leg, world metres (right of the road)
OCC = {"x": 5.5, "y0": 8.0, "w": 10.0, "h": 3.0, "l": 1.0}  # occluder face (world y0), appears at tick 1
if os.environ.get("GVD_ABL_LEG") == "near":
    LEG = {"x0": 3.5, "x1": 10.0, "y0": 9.0, "y1": 16.0}
    OCC = {"x": 5.0, "y0": 5.5, "w": 10.0, "h": 3.0, "l": 1.0}
TRAIN_STARTS = (0.0, 0.5, 1.0, 1.5, 2.0)
EVAL_STARTS = (0.25, 0.75, 1.25, 1.75)
N_HID = 96                                         # hidden leg queries per tick-1 sample (fixed size)
GATE_J = 0.5
GATE_GAP = 0.3


def render(cams, ego_y: float, leg_on: bool, occ_on: bool):
    rgb, depth, frames = {}, {}, []
    occ = dict(OCC, y0=OCC["y0"] - 0.0)
    for cid, cam in cams.items():
        sc = scene(cam, ego_y=ego_y, box=occ, with_box=occ_on, leg=LEG if leg_on else None)
        rgb[cid], depth[cid] = sc["rgb"], sc["depth"]
        frames.append(CamFrame(cam, sc["depth"], sc["sem"]))
        paint, bars = sc["paint"], sc["stop_bars"]
    lab = voxel_labels(frames, SEM_CLASSES, [], paint=paint, stop_bars=bars, max_free=1500, seed=0)
    return rgb, depth, lab


def leg_points(ego_y: float) -> np.ndarray:
    """Fine ground-voxel centres of the leg rectangle in the ego frame at ``ego_y`` (z = 0.25, the [0, 0.5) layer)."""
    xs = np.arange(LEG["x0"] + 0.25, LEG["x1"], 0.5)
    ys = np.arange(LEG["y0"] + 0.25, LEG["y1"], 0.5)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel() - ego_y, np.full(xx.size, 0.25)], 1)


def _seen(cams, depth, pts_ground):
    """Per point: visible (depth hit within 0.5 m of the point) in at least one camera."""
    vis = np.zeros(len(pts_ground), bool)
    for cid, cam in cams.items():
        uv, z, ok = cam.project(pts_ground)
        u = np.clip(np.nan_to_num(uv[:, 0]).astype(np.int64), 0, cam.width - 1)
        v = np.clip(np.nan_to_num(uv[:, 1]).astype(np.int64), 0, cam.height - 1)
        hit = depth[cid][v, u]
        vis |= ok & np.isfinite(hit) & (np.abs(hit - z) < 0.5)
    return vis


def hidden_leg(cams, start: float):
    """Leg cell centres (tick-1 ego frame) visible in some camera at tick 0 and hidden in every camera at tick 1."""
    _, d0, _ = render(cams, start, True, False)
    _, d1, _ = render(cams, start + STEP_M, True, True)
    p0 = leg_points(start)
    p1 = leg_points(start + STEP_M)
    g0, g1 = p0.copy(), p1.copy()
    g0[:, 2] = 0.0
    g1[:, 2] = 0.0
    keep = _seen(cams, d0, g0) & ~_seen(cams, d1, g1)
    return p1[keep], int(_seen(cams, d1, g1).sum()), int(keep.sum()), len(p1)


def _queries(lab, depth, seed, hid=None, leg_on=False, drop_hidden=False):
    q = make_queries(lab, n_pos=256, n_neg=256, seed=seed)
    gnd = (q["occ_t"] > 0) & (q["cls_t"] < 0) & (q["pts"][:, 2] < 0.5)   # non-road ground (grass): learnable "not road"
    q["cls_t"][gnd] = 0
    if hid is not None:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(hid), N_HID, replace=len(hid) < N_HID)
        q["pts"] = np.concatenate([q["pts"], hid[sel].astype(np.float32)])
        q["occ_t"] = np.concatenate([q["occ_t"], np.ones(N_HID, np.float32)])
        q["cls_t"] = np.concatenate([q["cls_t"], np.full(N_HID, C.ROAD if leg_on else 0, np.int64)])
        q["flow_t"] = np.concatenate([q["flow_t"], np.zeros((N_HID, 2), np.float32)])
        q["mask"] = np.concatenate([q["mask"], np.full(N_HID, 0.0 if drop_hidden else 1.0, np.float32)])
        if drop_hidden:
            q["cls_t"][-N_HID:] = -1
    q["depth"] = depth
    return q


def _stack(qs, cams):
    keys = ("pts", "occ_t", "cls_t", "flow_t", "mask", "mid_occ", "mid_free", "boxes")
    out = {k: torch.from_numpy(np.stack([q[k] for q in qs])) for k in keys}
    for cid in cams:
        out[f"depth:{cid}"] = torch.from_numpy(np.stack([q["depth"][cid] for q in qs]).astype(np.float32))
    return out


def _tensors(rgbs, cams):
    imgs = {cid: torch.stack([torch.from_numpy(r[cid]).permute(2, 0, 1).float() / 255.0 for r in rgbs]) for cid in cams}
    return imgs, {cid: torch.zeros(len(rgbs)) for cid in cams}


# ------------------------------------------------------------------ mutants
@contextlib.contextmanager
def mutant(name: str | None):
    orig_lift, orig_step = OccSparseNet.lift, OccSparseNet.sparse_step
    if name == "lift_off":
        def lift(self, enc):
            pts, feats, ws, st = orig_lift(self, enc)
            return pts[:0], feats[:0], ws[:0], {"rays": st["rays"], "lifted": 0}
        OccSparseNet.lift = lift
    elif name == "lift_feat_zero":
        def lift(self, enc):
            pts, feats, ws, st = orig_lift(self, enc)
            return pts, feats * 0.0, ws, st
        OccSparseNet.lift = lift
    elif name == "memory_off":
        def sparse_step(self, images, cams, ages, motion=None, memory=None):
            return orig_step(self, images, cams, ages, None, None)
        OccSparseNet.sparse_step = sparse_step
    try:
        yield
    finally:
        OccSparseNet.lift, OccSparseNet.sparse_step = orig_lift, orig_step


class Data:
    """Cached renders, labels and hidden-leg sets for every (start, leg_on)."""

    def __init__(self, cams):
        self.cams = cams
        self.cache = {}
        self.hid = {}
        for s in TRAIN_STARTS + EVAL_STARTS:
            hid, vis1, n_hid, n_all = hidden_leg(cams, s)
            assert vis1 == 0, f"start {s}: {vis1} leg cells still visible at tick 1 (occluder does not hide the leg)"
            assert n_hid >= 40, f"start {s}: only {n_hid} hidden leg cells"
            self.hid[s] = hid
            for leg_on in (True, False):
                self.cache[(s, leg_on)] = (render(cams, s, leg_on, False), render(cams, s + STEP_M, leg_on, True))
        print(f"A0 scene: {len(TRAIN_STARTS)} train / {len(EVAL_STARTS)} held-out starts x leg on/off; hidden leg cells per start "
              f"{min(len(h) for h in self.hid.values())}-{max(len(h) for h in self.hid.values())} (visible at tick 0, hidden in all "
              f"{len(cams)} cams at tick 1)")


def train(data: Data, mut: str | None, seed: int = 0):
    cams = data.cams
    torch.manual_seed(seed)
    model = OccSparseNet(profile=small_profile(IDS, LS, k_max=32, bins_per_ray=BINS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    motion = torch.tensor([[0.0, STEP_M, 0.0]] * BATCH)
    hist = []
    model.train()
    for it in range(STEPS):
        eps = [(TRAIN_STARTS[int(rng.integers(len(TRAIN_STARTS)))], bool(rng.integers(2))) for _ in range(BATCH)]
        r0 = [data.cache[e][0] for e in eps]
        r1 = [data.cache[e][1] for e in eps]
        q0 = _stack([_queries(r[2], r[1], it * 7 + k) for k, r in enumerate(r0)], cams)
        q1 = _stack([_queries(r[2], r[1], it * 7 + k + 3, hid=data.hid[e[0]], leg_on=e[1], drop_hidden=(mut == "loss_occluded"))
                     for k, (r, e) in enumerate(zip(r1, eps))], cams)
        i0, a0 = _tensors([r[0] for r in r0], cams)
        i1, a1 = _tensors([r[0] for r in r1], cams)
        out0 = model(i0, cams, a0, q0["pts"], memory=None)
        out1 = model(i1, cams, a1, q1["pts"], memory=out0["memory"], motion=motion)
        loss = occ_loss(out0, q0)["total"] + occ_loss(out1, q1)["total"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
    model.eval()
    return model, hist


@torch.no_grad()
def evaluate(model, data: Data, starts=EVAL_STARTS, use_vo: bool = False):
    cams = data.cams
    calls = {("on", True): [], ("on", False): [], ("off", True): [], ("off", False): []}
    for s in starts:
        hid = torch.from_numpy(data.hid[s].astype(np.float32))[None]
        for leg_on in (True, False):
            (rgb0, _, _), (rgb1, _, _) = data.cache[(s, leg_on)]
            i0, a0 = _tensors([rgb0], cams)
            i1, a1 = _tensors([rgb1], cams)
            mem = model(i0, cams, a0, hid[:, :1], memory=None)["memory"]
            mot = None if use_vo else torch.tensor([[0.0, STEP_M, 0.0]])
            for arm, m in (("on", mem), ("off", None)):
                out = model(i1, cams, a1, hid, memory=m, motion=mot if m is not None else None)
                road = (torch.sigmoid(out["occ_logit"][0]) > 0.5) & (out["cls_logit"][0].argmax(-1) == C.ROAD)
                calls[(arm, leg_on)].append(road.float().numpy())
    res = {}
    for arm in ("on", "off"):
        rec = float(np.mean(np.concatenate(calls[(arm, True)])))
        fa = float(np.mean(np.concatenate(calls[(arm, False)])))
        res[arm] = {"recall": rec, "false_alarm": fa, "J": rec - fa}
    res["gap"] = res["on"]["J"] - res["off"]["J"]
    res["gate"] = bool(res["on"]["J"] >= GATE_J and res["gap"] >= GATE_GAP)
    return res


def _fmt(tag, res, hist, dt):
    on, off = res["on"], res["off"]
    return (f"{tag}: memory on recall {on['recall']:.3f} false-alarm {on['false_alarm']:.3f} J {on['J']:.3f} | "
            f"memory off recall {off['recall']:.3f} false-alarm {off['false_alarm']:.3f} J {off['J']:.3f} | gap {res['gap']:+.3f} | "
            f"gate {'PASS' if res['gate'] else 'FAIL'} | loss {np.mean(hist[:5]):.3f}->{np.mean(hist[-5:]):.3f}, {len(hist)} steps {dt:.0f} s CPU")


def main() -> int:
    t_all = time.time()
    cams = small_rig(IDS, LS)
    data = Data(cams)
    print(f"A0 gate: J(on) >= {GATE_J} and J(on) - J(off) >= {GATE_GAP}; {STEPS} steps x batch {BATCH}, {LS} px, cams {IDS}, CPU, seed 0")
    t0 = time.time()
    with mutant(None):
        model, hist = train(data, None)
        base = evaluate(model, data)
        vo = evaluate(model, data, use_vo=True)
        train_fit = evaluate(model, data, starts=TRAIN_STARTS)
    print(_fmt("A1 baseline (held-out starts)", base, hist, time.time() - t0), flush=True)
    print(f"A1 baseline on train starts: J on {train_fit['on']['J']:.3f} off {train_fit['off']['J']:.3f}; "
          f"VO-driven warp (not gated): J on {vo['on']['J']:.3f} (recall {vo['on']['recall']:.3f}, false-alarm {vo['on']['false_alarm']:.3f})", flush=True)
    killed = {}
    for k, name in enumerate(("lift_off", "lift_feat_zero", "memory_off", "loss_occluded"), start=1):
        t0 = time.time()
        with mutant(name):
            m, h = train(data, name)
            r = evaluate(m, data)
        killed[name] = not r["gate"]
        print(_fmt(f"M{k} {name}", r, h, time.time() - t0) + f" | mutant {'KILLED' if killed[name] else 'SURVIVED'}", flush=True)
    print(f"A2 mutants killed {sum(killed.values())}/{len(killed)}: {killed}; total {time.time() - t_all:.0f} s (CPU timing)")
    assert base["gate"], f"baseline fails the memory gate: {base}"
    survivors = [n for n, k in killed.items() if not k]
    assert not survivors, f"surviving mutants (gate passes with the ablation in place): {survivors}"
    print("occ memory ablation A0-A2 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
