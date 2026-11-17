#!/usr/bin/env python3
"""Unseen road structure: full-region map labels, unmasked loss, unseen decode, observed, memory, causality.

TRACER method: the loss mask and the decoder are instrumented and traced
before (camera-masked loss / no scene context) vs after (map labels over the
whole grid / scene context), on a synthetic intersection whose cross legs are
hidden behind corner buildings.

* labels: map drivable cells cover the occluded legs; the camera-ray
  visibility set does not; merged targets keep them;
* loss trace: occluded targets carry weight and get a non-zero gradient at the
  logits (before: weight 0, gradient exactly 0);
* decoder trace: a query no camera sees still depends on the images through
  the scene context (before: d out / d encoder == 0);
* predict: top-stage cells outside every frustum are decoded, with finite prob
  and a valid class (decode_unseen=False: none);
* occ_out v2: ``observed`` on every cell and edge point; occluded cells
  observed=False with finite prob / class; agreement with ray-cast GT;
* memory: a leg seen at t0, not predicted at t1 and occluded at t1 persists
  (VO-warped, observed=False); without memory it is gone; a dropped cell a
  camera sees at t1 is not carried; a seen cell also replaces this tick's
  unobserved live guess at the same key (observed cells always win);
* causality: frames after t perturbed or deleted -> occ_out at <= t byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception import occ_out as oo  # noqa: E402
from python.perception.occ_geometry import DEFAULT_GRID, rig_cams  # noqa: E402
from python.perception.occ_sparse import OccPrediction, cam_weights, visible_mask  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.train.occ_map_labels import build_map_labels, merge_targets, visible_keys  # noqa: E402
from python.train.occ_train import sample_queries  # noqa: E402
import synthetic_scenes as ss  # noqa: E402
from synthetic_occ import GROUND_Z  # noqa: E402

G = DEFAULT_GRID
CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
RCAMS = {c.cam_id: c.scaled(96) for c in rig_cams(CFG)}
SCENE = ss.intersection(7)
YC = float(SCENE.roads[1].xy[0, 1])


def labels(scene):
    planes = {cid: ss.render(cam, scene)[:2] for cid, cam in RCAMS.items()}
    lab = build_occ_labels(planes, RCAMS, sem_classes=ss.SEM_CLASSES, stride=1)
    vis = visible_keys(planes, RCAMS, stride=2)
    ml = build_map_labels(scene.roads, ground_z=GROUND_Z)
    keys, cls = merge_targets(lab.keys, lab.cls, ml)
    return lab, vis, ml, keys, cls


LAB, VIS, ML, KEYS, CLS = labels(SCENE)


def key_at(x, y, z=GROUND_Z):
    return int(G.keys_of(np.array([[x, y, z]]))[0])


def check_labels() -> None:
    drv = ML.drivable_keys
    occl = drv[~np.isin(drv, VIS)]
    cam_drv = LAB.keys[LAB.cls == 1]
    leg = key_at(20.0, YC)  # cross road, 20 m right of the junction, behind the corner building
    near = key_at(0.0, 6.0)
    assert leg in drv and leg not in VIS and leg not in LAB.keys, "occluded leg must be labelled from the map, not the camera"
    assert near in drv and near in VIS
    assert leg in KEYS and CLS[np.searchsorted(KEYS, leg)] == 1
    assert occl.size > 0.2 * drv.size and drv.size > 2 * cam_drv.size, (occl.size, drv.size, cam_drv.size)
    print(f"labels: map drivable {drv.size} cells over the whole grid vs camera-hit drivable {cam_drv.size}; "
          f"{occl.size} ({occl.size / drv.size:.0%}) not reached by any of 8 camera rays (occluded / out of view), "
          f"kept as targets; visibility set {VIS.size} keys (metric split only) OK")


def _batch(visible_only: bool):
    import torch

    from python.train.occ_train import collate, prep_image

    kw = {"visible_only": True} if visible_only else {}  # after = the training default (no flag)
    q = sample_queries(KEYS, CLS, LAB.free, np.random.default_rng(3), vis_keys=VIS, n_occ=768, n_free=256, n_coarse=128, **kw)
    q["images"] = [prep_image(ss.render(c, SCENE)[2], NCAMS[i]) for i, c in enumerate(RCAMS.values())]
    q["recovery"] = False
    return collate([q]), torch


NCAMS = [c.scaled(48) for c in rig_cams(CFG)]


def _net(**kw):
    import torch

    from python.perception.occ_sparse import SparseOccNet

    torch.manual_seed(0)
    return SparseOccNet(NCAMS, config="cheap", **kw)


def check_loss_trace() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        from gvd_test_env import skip_check
        skip_check("occ_unseen loss trace", "torch missing"); return
    from python.train.occ_train import occ_loss

    trace = {}
    for name, vo in (("before (camera-masked)", True), ("after (map, unmasked)", False)):
        b, torch = _batch(vo)
        net = _net()
        out = net(b["images"], b["pts"])
        out["occ"].retain_grad(); out["cls"].retain_grad(); out["any"].retain_grad()
        loss, _ = occ_loss(out, b)
        loss.backward()
        occ = (b["vis"][0] < 0.5) & (b["occ"][0] > 0.5)
        g_occ = out["occ"].grad[0].abs(); g_cls = out["cls"].grad[0].abs().sum(-1); g_any = out["any"].grad[0].abs().sum(-1)
        trace[name] = {"occluded_pos": int(occ.sum()), "mask_w": float(b["occ_w"][0][occ].sum()),
                       "cls_targets": int((b["cls"][0][occ] >= 0).sum()),
                       "grad_occ": float(g_occ[occ].sum()), "grad_cls": float(g_cls[occ].sum()), "grad_any": float(g_any[occ].sum()),
                       "grad_visible": float(g_occ[(b["vis"][0] > 0.5) & (b["occ_w"][0] > 0)].sum())}
    bef, aft = trace["before (camera-masked)"], trace["after (map, unmasked)"]
    assert bef["occluded_pos"] == aft["occluded_pos"] > 50, trace
    assert bef["mask_w"] == 0 and bef["grad_occ"] == 0 and bef["grad_cls"] == 0 and bef["grad_any"] == 0, bef
    assert aft["mask_w"] == aft["occluded_pos"] and aft["cls_targets"] == aft["occluded_pos"], aft
    assert aft["grad_occ"] > 0 and aft["grad_cls"] > 0 and aft["grad_any"] > 0 and aft["grad_visible"] > 0, aft
    for k, v in trace.items():
        print(f"loss trace {k}: {v}")
    print("loss trace: occluded GT is NOT masked out of the loss; it reaches the logits with non-zero gradient OK")


def check_decoder_trace() -> None:
    try:
        import torch
    except ImportError:
        from gvd_test_env import skip_check
        skip_check("occ_unseen decoder trace", "torch missing"); return
    from python.train.occ_train import prep_image

    w = cam_weights([c.cam_id for c in NCAMS], preset="min")
    on = [c for c, x in zip(NCAMS, w) if x > 0]
    cand = G.centers_of_keys(np.arange(0, G.dense_cells, 97))
    pts = cand[~visible_mask(on, cand)][:256]
    assert pts.shape[0] == 256
    imgs = [torch.from_numpy(prep_image(ss.render(c, SCENE)[2], NCAMS[i]))[None] for i, c in enumerate(RCAMS.values())]
    res = {}
    for name, ctx in (("before (no ctx)", False), ("after (ctx)", True)):
        net = _net(ctx=ctx)
        out = net(imgs, torch.from_numpy(pts).float()[None], torch.from_numpy(w))
        assert float(out["seen"].abs().max()) == 0.0
        out["occ"].sum().backward()
        g = float(net.encoder.net[0].weight.grad.abs().sum()) if net.encoder.net[0].weight.grad is not None else 0.0
        with torch.no_grad():
            alt = net([im.flip(-1) for im in imgs], torch.from_numpy(pts).float()[None], torch.from_numpy(w))["occ"]
        res[name] = (g, float((alt - out["occ"].detach()).abs().max()))
    assert res["before (no ctx)"] == (0.0, 0.0), res
    assert res["after (ctx)"][0] > 0 and res["after (ctx)"][1] > 0, res
    print(f"decoder trace on 256 queries no camera sees (main+wide only): d occ / d encoder, max change on flipped images: {res} "
          f"-> unseen queries decode from the scene, not only lifted observed features OK")


def check_predict_unseen() -> None:
    try:
        import torch
    except ImportError:
        from gvd_test_env import skip_check
        skip_check("occ_unseen predict", "torch missing"); return
    w = cam_weights([c.cam_id for c in NCAMS], preset="min")
    on = [c for c, x in zip(NCAMS, w) if x > 0]
    imgs = [torch.rand(1, 3, c.height, c.width) for c in NCAMS]
    out = {}
    for du in (False, True):
        net = _net(decode_unseen=du).eval()
        with torch.no_grad():
            net.head[-1].bias.zero_(); net.head[-1].bias[:3] = 10.0  # every stage hot
        p = net.predict(imgs, max_points=8192, max_coarse=4096, cam_w=w)
        vis = visible_mask(on, G.centers_of_keys(p.keys)) if p.keys.size else np.zeros(0, bool)
        out[du] = (p, vis)
    p, vis = out[True]
    # frustum-only: just fine children of border coarse cells fall outside; decode_unseen: whole regions
    p0, v0 = out[False]
    assert (~v0).sum() <= 0.05 * max(1, p0.keys.size) and (~vis).sum() > 10 * max(1, (~v0).sum()), ((~v0).sum(), p0.keys.size, (~vis).sum())
    assert np.isfinite(p.prob).all() and ((p.prob >= 0) & (p.prob <= 1)).all() and (p.cls < 3).all()
    print(f"predict: decode_unseen keeps {(~vis).sum()} of {p.keys.size} cells outside every active frustum "
          f"(frustum-only: {(~v0).sum()} border children of {p0.keys.size}), all finite prob in [0,1], valid class OK")


class StubNet:
    def __init__(self, preds, cams):
        self.cams, self.cam_ids, self.grid = cams, tuple(c.cam_id for c in cams), G
        self.preds = list(preds)

    def predict(self, images, *, max_points, max_coarse, cam_w):
        return self.preds.pop(0)


def _pred(keys, cls, drop=None):
    keys, cls = np.asarray(keys), np.asarray(cls)
    if drop is not None:
        m = ~np.isin(keys, drop); keys, cls = keys[m], cls[m]
    n = keys.size
    return OccPrediction(keys, np.full(n, 0.9, np.float32), cls.astype(np.uint8), 0, n, (0, 0, n))


CLIP = ss.clip(SCENE, frames=4, dt=0.25)
CLIP_LABELS = [labels(sc) for _, sc in CLIP]


def _runtime(preds, memory=True):
    cams = list(RCAMS.values())
    return oo.OccRuntime(StubNet(preds, cams), memory=memory, timer=lambda: 0.0)


def _imgs():
    return [object() for _ in RCAMS]


def check_observed() -> None:
    rt = _runtime([_pred(k, c) for _, _, _, k, c in CLIP_LABELS[:1]])
    m = rt.step(_imgs(), 0.0)
    obs = np.array(m["cells"]["observed"]); prob = np.array(m["cells"]["prob"]); cls = np.array(m["cells"]["cls"])
    assert obs.size == len(m["cells"]["ijk"]) and all(len(e["observed"]) == len(e["pts"]) for e in m["edges"])
    assert (~obs).sum() > 100 and obs.sum() > 100, obs.sum()
    assert np.isfinite(prob[~obs]).all() and (prob[~obs] > 0).all() and (cls[~obs] < 3).all()
    ijk = np.array(m["cells"]["ijk"])
    keys = G.key_of_cell(ijk)
    ground = ijk[:, 2] == ML.ground_k
    gt_vis = np.isin(keys, VIS)
    agree = float((obs[ground] == gt_vis[ground]).mean())
    assert agree > 0.8, agree
    # edge points track GT visibility of the ground cell under the point
    ep = np.array([p for e in m["edges"] for p in e["pts"]], float).reshape(-1, 2)
    eo = np.array([b for e in m["edges"] for b in e["observed"]], bool)
    ec = G.cell_of(np.concatenate([ep, np.zeros((len(ep), 1))], 1))
    ec[:, 2] = ML.ground_k
    e_agree = float((eo == np.isin(G.key_of_cell(ec), VIS)).mean())
    assert eo.size > 20 and e_agree > 0.85, (eo.size, e_agree)
    # preset "min" (forward cams only): edge points behind the ego are unobserved, ahead observed
    rt = oo.OccRuntime(StubNet([_pred(k, c) for _, _, _, k, c in CLIP_LABELS[:1]], list(RCAMS.values())),
                       preset="min", memory=False, timer=lambda: 0.0)
    m2 = rt.step(_imgs(), 0.0)
    fp = np.array([p for e in m2["edges"] for p in e["pts"]], float).reshape(-1, 2)
    fo = np.array([b for e in m2["edges"] for b in e["observed"]], bool)
    behind, ahead = fp[:, 1] < -3.0, (fp[:, 1] > 8.0) & (fp[:, 1] < 30.0)
    assert behind.sum() > 5 and not fo[behind].any(), (behind.sum(), fo[behind].sum())
    assert ahead.sum() > 5 and fo[ahead].mean() > 0.8, fo[ahead].mean()
    print(f"observed: {obs.sum()} observed / {(~obs).sum()} not (finite prob, valid class); ground cells agree with "
          f"8-camera ray-cast GT visibility {agree:.1%}; edge points agree {e_agree:.1%}; preset min: "
          f"{int(behind.sum())} edge points behind ego all unobserved, ahead {fo[ahead].mean():.0%} observed OK")


def check_memory() -> None:
    _, vis1, ml1, k1, c1 = CLIP_LABELS[1]
    drv1 = ml1.drivable_keys
    hidden = drv1[~np.isin(drv1, vis1)]          # occluded at t1
    shown = drv1[np.isin(drv1, vis1)][::7]       # visible at t1
    preds = [_pred(*CLIP_LABELS[0][3:]), _pred(k1, c1, drop=np.concatenate([hidden, shown]))]
    res = {}
    for mem in (True, False):
        rt = _runtime(list(preds), memory=mem)
        rt.step(_imgs(), CLIP[0][0])
        m = rt.step(_imgs(), CLIP[1][0])
        keys, kcls, _, obs = rt.last_cells  # every stored cell (the message keeps the nearest MAX_CELLS)
        carried = keys[~np.isin(keys, preds[1].keys) & (kcls == 1)]
        prec = float(np.isin(carried, ml1.keys[ml1.cls == 1]).mean()) if carried.size else None
        if mem:
            prec_mem = prec
        assert m["ego"]["ok"], m["ego"]
        res[mem] = (float(np.isin(hidden, keys).mean()), float(np.isin(shown, keys).mean()),
                    float(obs[np.isin(keys, hidden)].mean()) if np.isin(keys, hidden).any() else None)
    assert res[True][0] > 0.5 and res[False][0] == 0.0, res        # carried vs gone
    assert res[True][2] == 0.0, res                                 # carried cells are not observed
    assert res[True][1] < 0.1, res                                  # a seen-free cell is not carried
    assert prec_mem is not None and prec_mem > 0.97, prec_mem      # carried cells land on the true road (VO-warped)
    print(f"memory: occluded-at-t1 leg cells present with memory {res[True][0]:.0%} (observed=False), without {res[False][0]:.0%}; "
          f"visible-and-dropped cells carried {res[True][1]:.0%}; carried ground cells on true road {prec_mem:.1%} OK")


def check_memory_rules() -> None:
    from python.perception.occ_outputs import VEHICLE
    rng = np.random.default_rng(5)
    keys = np.unique(rng.integers(0, G.dense_cells, 600))
    cls = np.where(np.arange(keys.size) % 3 == 0, VEHICLE, 1).astype(np.uint8)
    prob = np.full(keys.size, 0.9, np.float32)
    obs = np.ones(keys.size, bool)
    occl = np.zeros(0, np.int64)
    e = np.zeros(0, np.int64)
    kw = dict(fwd=0.0, dyaw=0.0, cams=[], occl=occl)
    mem = oo.OccMemory(G)
    mem.step(keys, cls, prob, obs, ok=True, dt=None, **kw)
    assert mem.keys.size and not (mem.cls == VEHICLE).any(), np.unique(mem.cls)   # vehicles are never remembered
    K, C, _, _ = mem.step(e, e.astype(np.uint8), e.astype(np.float32), e.astype(bool), ok=True, dt=0.1, **kw)
    assert K.size and not (C == VEHICLE).any(), (K.size, np.unique(C))
    n_static = K.size
    # carried cells move with the ego: 2 m forward (+y), no yaw -> every remembered cell shifts 2 m back
    before = G.centers_of_keys(mem.keys)
    exp = G.keys_of(before - np.array([0.0, 2.0, 0.0]))
    exp = np.unique(exp[exp >= 0])
    K, _, _, _ = mem.step(e, e.astype(np.uint8), e.astype(np.float32), e.astype(bool), ok=True, dt=0.1,
                          fwd=2.0, dyaw=0.0, cams=[], occl=occl)
    assert exp.size > 100 and np.array_equal(np.sort(K), exp), (exp.size, K.size, np.isin(K, exp).mean())
    # yaw: 90 deg left turn -> (x, y) maps to (y, -x)
    before = G.centers_of_keys(mem.keys)
    exp = G.keys_of(np.stack([before[:, 1], -before[:, 0], before[:, 2]], 1))
    exp = np.unique(exp[exp >= 0])
    K, _, _, _ = mem.step(e, e.astype(np.uint8), e.astype(np.float32), e.astype(bool), ok=True, dt=0.1,
                          fwd=0.0, dyaw=np.pi / 2, cams=[], occl=occl)
    assert exp.size > 10 and np.array_equal(np.sort(K), exp), (exp.size, K.size, np.isin(K, exp).mean())
    mem.step(e, e.astype(np.uint8), e.astype(np.float32), e.astype(bool), ok=False, dt=0.1, **kw)
    assert mem.keys.size == 0, mem.keys.size                                    # VO lost -> memory cleared
    K, _, _, _ = mem.step(e, e.astype(np.uint8), e.astype(np.float32), e.astype(bool), ok=True, dt=0.1, **kw)
    assert K.size == 0, K.size                                                  # nothing resurfaces after VO recovers
    print(f"memory rules: {int((cls == VEHICLE).sum())} vehicle cells never carried ({n_static} static carried); "
          f"VO-warped (2 m forward shift, 90 deg yaw rotation) cell-exact; VO lost -> cleared, nothing resurfaces OK")



def check_memory_seen_beats_guess() -> None:
    """A tick-1 in-frustum, unobserved live guess must lose to a seen memory cell.

    OccRuntime path: stub predict + cap + observed_mask + OccMemory. Tick 0
    observes a drivable cell. Tick 1 emits the same key as a wrong class,
    still in a camera frustum, but an elevated occluder on the ray makes
    observed_mask False. The merge rule keeps the earlier observation.
    Mutant M6 (keep only memory keys that are missing from the live set)
    lets the live guess win.
    """
    from python.perception.occ_geometry import project
    from python.perception.occ_out import observed_mask, occluders

    cams = [c for c in RCAMS.values() if c.cam_id in ("main", "wide")]
    cam = next(c for c in cams if c.cam_id == "wide")
    origin = np.asarray(cam.pos, np.float64)
    tgt_pt = np.array([2.0, 10.0, GROUND_Z + 0.2])
    ref_pt = np.array([-4.0, 4.0, GROUND_Z + 0.2])  # off the ray; keeps a ground band
    tgt = int(G.keys_of(tgt_pt.reshape(1, 3))[0])
    ref = int(G.keys_of(ref_pt.reshape(1, 3))[0])
    assert tgt >= 0 and ref >= 0 and tgt != ref
    _, _, tgt_ok = project(cam, G.centers_of_keys(np.array([tgt])))
    assert bool(tgt_ok[0]), "target must be in the wide frustum at tick 1"

    # An elevated cell on the camera ray, not the last voxel before the target.
    gz = int(G.cell_of_key(np.array([ref]))[0, 2])
    vec = G.centers_of_keys(np.array([tgt]))[0] - origin
    dist = float(np.linalg.norm(vec))
    unit = vec / max(dist, 1e-9)
    cands = []
    for t in np.arange(0.4, dist - G.voxel_m, G.voxel_m * 0.5):
        k = int(G.keys_of((origin + unit * t).reshape(1, 3))[0])
        if k < 0 or k in (tgt, ref):
            continue
        if int(G.cell_of_key(np.array([k]))[0, 2]) >= gz + 1:
            cands.append(k)
    assert cands, "need an elevated occluder on the ray"
    k0, c0 = np.array([ref, tgt], np.int64), np.array([1, 1], np.uint8)
    occl_k = None
    obs1 = None
    for k in cands:
        k1 = np.array([ref, tgt, k], np.int64)
        c1 = np.array([1, 0, 0], np.uint8)
        obs1 = observed_mask(G.centers_of_keys(k1), cams, occluders(k1, c1, G), G)
        if not bool(obs1[k1 == tgt][0]):
            occl_k, k1, c1 = k, k1, c1
            break
    assert occl_k is not None, f"no on-ray occluder hid the target (tried {len(cands)})"
    assert int((k1 == tgt).sum()) == 1

    vehicle = {"wheelspeed_mps": 0.0, "imu_yaw_rate": 0.0}
    imgs = [object() for _ in cams]
    rt = oo.OccRuntime(StubNet([_pred(k0, c0), _pred(k1, c1)], cams),
                       from_vehicle=True, memory=True, timer=lambda: 0.0)
    m0 = rt.step(imgs, 0.0, vehicle=vehicle)
    keys, cls, _p, obs = rt.last_cells
    assert m0["ego"]["ok"] and bool(obs[keys == tgt][0]) and int(cls[keys == tgt][0]) == 1
    m1 = rt.step(imgs, 0.1, vehicle=vehicle)
    keys, cls, _p, obs = rt.last_cells
    assert m1["ego"]["ok"]
    assert tgt in keys, "remembered cell must still be in last_cells"
    assert int(cls[keys == tgt][0]) == 1, (int(cls[keys == tgt][0]), "seen memory should beat the unobserved live guess")
    assert not bool(obs[keys == tgt][0])

    rt_off = oo.OccRuntime(StubNet([_pred(k0, c0), _pred(k1, c1)], cams),
                           from_vehicle=True, memory=False, timer=lambda: 0.0)
    rt_off.step(imgs, 0.0, vehicle=vehicle)
    rt_off.step(imgs, 0.1, vehicle=vehicle)
    keys, cls, _p, obs = rt_off.last_cells
    assert int(cls[keys == tgt][0]) == 0 and not bool(obs[keys == tgt][0])
    print("memory seen-beats-guess: tick-1 in-frustum unobserved live guess (cls=other) "
          "replaced by the tick-0 observation (cls=drivable); without memory the guess stands OK")


def check_causality() -> None:
    preds = [_pred(k, c) for _, _, _, k, c in CLIP_LABELS]
    full = _runtime(list(preds))
    a = [oo.encode_occ_out(full.step(_imgs(), t)) for t, _ in CLIP]
    k = 1
    rng = np.random.default_rng(0)
    junk = []
    for _ in CLIP[k + 1:]:
        jk = np.unique(rng.integers(0, G.dense_cells, 5000))
        junk.append(_pred(jk, rng.integers(0, 3, jk.size)))
    pert = _runtime(list(preds[:k + 1]) + junk)
    b = [oo.encode_occ_out(pert.step(_imgs(), t)) for t, _ in CLIP]
    cut = _runtime(list(preds[:k + 1]))
    c = [oo.encode_occ_out(cut.step(_imgs(), t)) for t, _ in CLIP[:k + 1]]
    assert a[:k + 1] == b[:k + 1] == c and a[k + 1] != b[k + 1]
    res = "stub only"
    try:
        import torch

        net = _net().eval()
        rtA = oo.OccRuntime(net, timer=lambda: 0.0)
        imgs = [[torch.rand(1, 3, c.height, c.width, generator=torch.Generator().manual_seed(i * 10 + j)) for j, c in enumerate(NCAMS)]
                for i in range(4)]
        A = [oo.encode_occ_out(rtA.step(im, 0.25 * i)) for i, im in enumerate(imgs)]
        net.reset_temporal()
        rtB = oo.OccRuntime(net, timer=lambda: 0.0)
        B = [oo.encode_occ_out(rtB.step(im if i <= k else [x * 0 + 0.5 for x in im], 0.25 * i)) for i, im in enumerate(imgs)]
        assert A[:k + 1] == B[:k + 1], "future frames changed the past"
        res = "stub + SparseOccNet"
    except ImportError:
        pass
    print(f"causality ({res}): frames after t={CLIP[k][0]} s perturbed or deleted -> occ_out at <= t byte-identical "
          f"({len(a[k])} bytes) OK")


def main() -> int:
    check_labels()
    check_loss_trace()
    check_decoder_trace()
    check_predict_unseen()
    check_observed()
    check_memory()
    check_memory_rules()
    check_memory_seen_beats_guess()
    check_causality()
    print("test_occ_unseen: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
