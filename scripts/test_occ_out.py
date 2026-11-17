#!/usr/bin/env python3
"""occ_out v1 contract: every field the 4 debug panels need, strict validation.

A synthetic world (road strip + grass, roadside posts, a parked car facing
forward, an oncoming car) is driven by an ego at 12 m/s turning left at
0.1 rad/s, sampled at IRREGULAR frame timestamps. A stub net replays the
ego-frame occupied cells (what ``SparseOccNet.predict`` returns), so
``OccRuntime.step`` runs the real VO -> tracker -> outputs -> occ_out path.

* panels: every ``PANEL_FIELDS`` path resolves in the message, lanes empty
  (paint UNLABELED), graph series one value per tick;
* ego: source ``vo`` by default (``--ego-from-vehicle`` off), speed ~=
  |(dx, dy)|/dt and yaw_rate ~= dyaw/dt with dt from the frame timestamps,
  left turn positive; with the flag, the vehicle signals and source
  ``vehicle``;
* heading: unit ``[cos, sin]``; parked car ``box`` along the road, oncoming
  car ``velocity`` pointing -y;
* boxes2d: projections of the 3-D boxes, same id and class, inside the image,
  clipped at the border, none for STALE / ABSENT cameras;
* corridor: ego frame, closed, simple, <= cap verts, empty at low confidence;
* reject: unknown / missing fields at every level, NaN, out-of-range, caps;
* VRAM: ``peak_vram_mb`` None / UNPROVEN, CPU estimate only in
  ``vram_est_mb_cpu``, never aliased;
* flat: a 64x finer grid and a 30k-cell prediction give a message of the same
  capped size.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception import occ_out as oo  # noqa: E402
from python.perception.occ_ego import add_ego_args  # noqa: E402
from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, rig_cams  # noqa: E402
from python.perception.occ_sparse import OccPrediction  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
CAMS = [c.scaled(640) for c in rig_cams(CFG)]
V, W = 12.0, 0.1                       # ego m/s, rad/s (left)
TS = [0.0, 0.1, 0.15, 0.25, 0.45, 0.5, 0.6, 0.8, 0.9, 1.0, 1.12, 1.2]
ONC_V = 8.0                            # oncoming car, world m/s toward -y
# the recorder's measured per-camera rates (GVD-reported): 10/10/10/2/2/1/1/1 Hz
RATES = {"narrow": 10, "main": 10, "wide": 10, "pillarL": 2, "pillarR": 2, "repeatL": 1, "repeatR": 1, "rear": 1}


def pose(t: float):
    """Constant speed + yaw rate (CCW) from the origin heading +y."""
    h = W * t
    return (-(V / W) * (1 - math.cos(h)), (V / W) * math.sin(h), h)


def to_ego(q: np.ndarray, p) -> np.ndarray:
    px, py, h = p
    d = q - np.array([px, py])
    return np.stack([d @ np.array([math.cos(h), math.sin(h)]), d @ np.array([-math.sin(h), math.cos(h)])], 1)


def _rect(x0, x1, y0, y1, s=0.2):
    return np.array([[x, y] for x in np.arange(x0, x1 + 1e-6, s) for y in np.arange(y0, y1 + 1e-6, s)])


def world(t: float):
    rng = np.random.default_rng(5)
    posts = np.stack([rng.choice([-1, 1], 160) * rng.uniform(6.0, 22.0, 160), rng.uniform(-10, 80, 160)], 1)
    road = _rect(-3.6, 3.6, -10, 80, 0.4)
    grass = np.concatenate([_rect(-12, -4.0, -10, 80, 0.4), _rect(4.0, 12, -10, 80, 0.4)])
    parked = _rect(2.2, 3.4, 24.0, 28.4)                  # on the right of the lane, facing +y
    oy = 50.0 - ONC_V * t
    onc = _rect(-2.6, -1.4, oy - 2.2, oy + 2.2)          # left lane, driving -y
    return posts, road, grass, parked, onc


def frame(t: float, grid: OccGrid = DEFAULT_GRID) -> OccPrediction:
    posts, road, grass, parked, onc = world(t)
    p = pose(t)
    keys, cls = [], []
    for pts, c, zs in ((posts, 0, (0.2, 0.6, 1.0, 1.4)), (road, 1, (-0.2,)), (grass, 0, (-0.2,)),
                       (parked, 2, (0.2, 0.6, 1.0, 1.4)), (onc, 2, (0.2, 0.6, 1.0, 1.4))):
        e = to_ego(pts, p)
        for z in zs:
            xyz = np.concatenate([e, np.full((len(e), 1), z)], 1)
            ins = grid.inside(xyz)
            keys.append(grid.keys_of(xyz[ins])); cls.append(np.full(int(ins.sum()), c, np.uint8))
    k = np.concatenate(keys); c = np.concatenate(cls)
    k, idx = np.unique(k, return_index=True)
    n = int(k.size)
    return OccPrediction(k, np.full(n, 0.9, np.float32), c[idx], 0, n, (0, 0, n))


class StubNet:
    """Replays prepared predictions; same attributes OccRuntime reads from SparseOccNet."""

    def __init__(self, preds, grid: OccGrid = DEFAULT_GRID):
        self.cams, self.cam_ids, self.grid = CAMS, tuple(c.cam_id for c in CAMS), grid
        self.preds = list(preds)
        self.cam_w = []

    def predict(self, images, *, max_points, max_coarse, cam_w):
        self.cam_w.append(np.asarray(cam_w).copy())
        return self.preds.pop(0)


class NoImage:
    """Stand-in camera frame that raises on any read (shape, pixels, array, iteration)."""

    def __getattr__(self, name):
        raise AssertionError(f"camera frame read downstream of the net: .{name}")

    def __getitem__(self, k):
        raise AssertionError("camera frame indexed downstream of the net")

    def __array__(self, *a, **k):
        raise AssertionError("camera frame converted downstream of the net")

    def __len__(self):
        raise AssertionError("camera frame len() downstream of the net")

    def __iter__(self):
        raise AssertionError("camera frame iterated downstream of the net")


def ages_at(t: float):
    """Each camera's own frame age at tick t, from its recorder rate."""
    return {cid: round(t - math.floor(t * hz + 1e-9) / hz, 6) for cid, hz in RATES.items()}


def run(*, from_vehicle=False, vehicle_fn=None, preset=None, vram_est=None, ages_fn=None, memory=True):
    rt = oo.OccRuntime(StubNet([frame(t) for t in TS]), from_vehicle=from_vehicle, preset=preset, vram_est_mb_cpu=vram_est,
                       memory=memory)
    imgs = [NoImage() for _ in CAMS]  # raises on any pixel access: everything after the net is image-blind
    ages_fn = ages_fn or ages_at
    return rt, [rt.step(imgs, t, ages_s=ages_fn(t), vehicle=None if vehicle_fn is None else vehicle_fn(t)) for t in TS]


RUN = run()


def check_panels() -> None:
    rt, msgs = RUN
    for m in msgs:
        oo.validate_occ_out(json.loads(oo.encode_occ_out(m)))
    last = msgs[-1]
    missing = {p: [f for f in fs if not oo.field_present(last, f)] for p, fs in oo.PANEL_FIELDS.items()}
    assert not any(missing.values()), missing
    assert last["cells"]["ijk"] and len(last["edges"]) == 2 and last["corridor"]["pts"] and len(last["objects"]) == 2
    assert last["boxes2d"].get("main") and last["lanes"] == []
    assert set(last["cams"]) == set(oo.CAM_IDS)
    st = {c: v["status"] for c, v in last["cams"].items()}
    assert st["main"] == "LIVE" and all(v in oo.CAM_STATUS for v in st.values()), st
    assert [m["seq"] for m in msgs] == list(range(len(TS))) and [m["t"] for m in msgs] == TS
    confs = [x for e in last["edges"] for x in e["conf"]]
    assert float(np.median(confs)) == 0.9 and max(confs) <= 0.9, confs  # = the boundary cells' occupancy prob
    c = last["counts"]
    assert c["stored"] <= c["queried"] + oo.MAX_MEMORY and c["queried"] <= c["full"] and c["frustum"] < c["full"] and c["cells_out"] <= oo.MAX_CELLS
    from python.viz import occ_viz

    occ_viz.DROPPED.clear()
    vz = occ_viz.occ_to_viz(oo.occ_out_to_viz_inputs(last))
    assert not occ_viz.DROPPED and len(vz["edges"]) == 2 and len(vz["tracks"]) == 2 and vz["occ"]["cells"], dict(occ_viz.DROPPED)
    assert sorted(t["motion"] for t in vz["tracks"]) == ["oncoming", "parked"]
    print(f"panels: {', '.join(f'{p} {len(fs)} fields' for p, fs in oo.PANEL_FIELDS.items())} all present; "
          f"graphs {len(msgs)} ticks; stored {c['stored']} / queried {c['queried']} / frustum {c['frustum']} / full {c['full']}; "
          f"{len(oo.encode_occ_out(last))} bytes; -> occ_to_viz v1 with 0 dropped keys OK")


def _truth(i: int):
    """Per-frame ego motion from frame i-1 to i, in frame i-1's ego frame."""
    a, b = pose(TS[i - 1]), pose(TS[i])
    d = to_ego(np.array([[b[0], b[1]]]), a)[0]
    return float(np.hypot(*d)), b[2] - a[2], TS[i] - TS[i - 1]


def check_ego() -> None:
    a = add_ego_args(argparse.ArgumentParser()).parse_args([])
    assert a.ego_from_vehicle is False
    _, msgs = RUN
    assert msgs[0]["ego"]["dt_s"] is None and not msgs[0]["ego"]["ok"]
    def _vo_ok(msgs, tag: str) -> tuple[float, float]:
        dists, yaws = [], []
        held = []
        for i in range(1, len(TS)):
            e = msgs[i]["ego"]
            dist, dyaw, dt = _truth(i)
            assert e["source"] == "vo" and e["ok"], (tag, i, e)
            assert abs(e["dt_s"] - dt) < 1e-6, (tag, i, e["dt_s"], dt)
            dists.append(abs(e["speed_mps"] * dt - dist)); yaws.append(abs(e["yaw_rate_rps"] * dt - dyaw))
            assert abs(e["speed_mps"] - dist / dt) < max(2.0, 0.3 / dt), (tag, i, e, dist / dt)
            assert e["yaw_rate_rps"] > 0, (tag, i, e)  # left turn is positive
        assert not held
        assert max(dists) < 0.3 and max(yaws) < 0.012, (tag, dists, yaws)
        return max(dists), max(yaws)

    d0, y0 = _vo_ok(msgs, "recorder rates")
    _, fresh = run(ages_fn=lambda t: {c: 0.0 for c in RATES})
    d1, y1 = _vo_ok(fresh, "all cams fresh")
    front = ("main", "wide", "narrow")
    _, f3 = run(ages_fn=lambda t: {c: (0.0 if c in front else 1.0) for c in RATES})
    d2, y2 = _vo_ok(f3, "front 3 fresh")
    sp = np.array([m["ego"]["speed_mps"] for m in msgs[1:]]); yr = np.array([m["ego"]["yaw_rate_rps"] for m in msgs[1:]])
    assert abs(sp.mean() - V) < 1.0 and abs(yr.mean() - W) < 0.03, (sp, yr)
    # with --ego-from-vehicle the car's own signals, tagged vehicle
    _, vm = run(from_vehicle=True, vehicle_fn=lambda t: {"wheelspeed_mps": 11.5 + t, "imu_yaw_rate": 0.09})
    assert all(m["ego"]["source"] == "vehicle" and m["ego"]["ok"] for m in vm)
    assert abs(vm[-1]["ego"]["speed_mps"] - (11.5 + TS[-1])) < 1e-3 and vm[-1]["ego"]["yaw_rate_rps"] == 0.09
    print(f"ego: source vo by default, dt from timestamps {sorted({round(_truth(i)[2], 3) for i in range(1, len(TS))})}; "
          f"recorder/all-fresh/front-3 all 11/11 ok, max |speed*dt-dist| {d0:.3f}/{d1:.3f}/{d2:.3f} m, "
          f"|yaw_rate*dt-dyaw| {y0:.4f}/{y1:.4f}/{y2:.4f} rad; "
          f"mean {sp.mean():.2f} m/s (true {V}), {yr.mean():.3f} rad/s (true {W}, left +); vehicle flag -> source vehicle OK")



def check_vo_runtime_obs() -> None:
    """OccRuntime VO must use live-camera observed_mask (kills O1 and O3).

    O1 (obs=None): a dense ego-fixed sheet that observed_mask hides must not
    vote; VO still tracks 1.5 m of real motion. O3 (obs from stale cameras
    too): ego-fixed clutter only a stale rear camera can see must not vote.
    """
    g = DEFAULT_GRID
    rng = np.random.default_rng(5)
    posts = np.stack([rng.choice([-1, 1], 300) * rng.uniform(5, 22, 300),
                      rng.uniform(-10, 50, 300)], 1)
    sheet = np.c_[np.random.default_rng(9).uniform(-6, 6, 3000),
                  np.random.default_rng(9).uniform(0, 30, 3000),
                  np.full(3000, 1.2)]
    rear = np.c_[np.random.default_rng(11).uniform(-2, 2, 400),
                 np.random.default_rng(11).uniform(-8, -3, 400),
                 np.full(400, 1.0)]

    def pred(fwd: float, clutter: np.ndarray | None) -> OccPrediction:
        p = posts - np.array([0.0, fwd])
        keys, cls = [], []
        for z in (0.2, 0.6, 1.0, 1.4):
            xyz = np.column_stack([p, np.full(len(p), z)])
            ins = g.inside(xyz)
            keys.append(g.keys_of(xyz[ins])); cls.append(np.zeros(int(ins.sum()), np.uint8))
        gx, gy = np.meshgrid(np.arange(-4, 4, 0.4) + 0.2, np.arange(-10, 50, 0.4) + 0.2)
        road = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.2)])
        keys.append(g.keys_of(road)); cls.append(np.ones(len(road), np.uint8))
        if clutter is not None:
            ins = g.inside(clutter)
            keys.append(g.keys_of(clutter[ins])); cls.append(np.zeros(int(ins.sum()), np.uint8))
        k = np.concatenate(keys); c = np.concatenate(cls)
        ok = k >= 0; k, idx = np.unique(k[ok], return_index=True)
        n = int(k.size)
        return OccPrediction(k, np.full(n, 0.9, np.float32), c[ok][idx], 0, n, (0, 0, n))

    ages_fresh = {c: 0.0 for c in RATES}
    ages_front = {c: (0.0 if c in ("main", "wide", "narrow") else 1.0) for c in RATES}
    imgs = [NoImage() for _ in CAMS]

    def track(clutter, ages, tag, true_fwd=1.5, dt=0.2):
        net = StubNet([pred(0.0, clutter), pred(true_fwd, clutter)])
        rt = oo.OccRuntime(net, from_vehicle=False, memory=False)
        m0 = rt.step(imgs, 0.0, ages_s=ages)
        m1 = rt.step(imgs, dt, ages_s=ages)
        e = m1["ego"]
        err = abs(e["speed_mps"] * dt - true_fwd)
        assert e["source"] == "vo" and e["ok"], (tag, e)
        assert err <= 0.2, (tag, e, err)
        print(f"runtime VO {tag}: fwd {e['speed_mps']*dt:.2f} m (true {true_fwd}) ok err {err:.3f} OK")

    track(sheet, ages_fresh, "dense unobserved sheet (O1)")
    track(rear, ages_front, "stale-rear clutter only (O3)")


def check_heading() -> None:
    last = RUN[1][-1]
    objs = sorted(last["objects"], key=lambda o: o["x"])
    onc, parked = objs
    for o in objs:
        assert abs(o["heading"][0] ** 2 + o["heading"][1] ** 2 - 1) < 1e-6 and o["frames"] >= 3
    assert parked["motion"] == "parked" and parked["heading_src"] == "box", parked
    # box (PCA) axis is +-pi ambiguous: along the road either way (ego yawed W*t from the road)
    assert abs(abs(parked["heading"][1]) - 1) < 0.05, parked
    assert onc["motion"] == "oncoming" and onc["heading_src"] == "velocity" and onc["heading"][1] < -0.95, onc
    assert "yaw" not in onc  # heading is the only direction field
    print(f"heading: parked box axis {parked['heading']}, oncoming velocity {onc['heading']} OK")


def check_boxes2d() -> None:
    rt, msgs = RUN
    last = msgs[-1]
    objs = {o["id"]: o for o in last["objects"]}
    for cid, boxes in last["boxes2d"].items():
        cam = last["cams"][cid]
        assert cam["status"] == "LIVE", (cid, cam)
        for b in boxes:
            assert b["id"] in objs and b["cls"] == objs[b["id"]]["cls"]
            u0, v0, u1, v1 = b["box"]
            assert 0 <= u0 < u1 <= cam["res"][0] and 0 <= v0 < v1 <= cam["res"][1], (cid, b)
    assert set(objs) <= {b["id"] for b in last["boxes2d"]["main"]} | {b["id"] for b in last["boxes2d"].get("wide", [])}
    assert "rear" not in last["boxes2d"]  # both cars are ahead
    main = next(c for c in CAMS if c.cam_id == "main")
    # straddling the image edge -> clipped to the border; behind the camera -> none
    edge = {"x": 0.0, "y": 10.0, "z0": -1.0, "l": 4.0, "w": 60.0, "h": 1.5}
    bx = oo.project_box(main, oo.box_corners(edge, [0.0, 1.0]))
    assert bx is not None and bx[0] == 0.0 and bx[2] == float(main.width), bx
    assert oo.project_box(main, oo.box_corners(dict(edge, y=-20.0, w=2.0), [0.0, 1.0])) is None
    # masked cameras: ABSENT, no boxes
    _, mm = run(preset="min")
    m = mm[-1]
    assert {c for c, v in m["cams"].items() if v["status"] != "ABSENT"} == {"main", "wide"}, m["cams"]
    assert m["counts"]["frustum"] < last["counts"]["frustum"], (m["counts"], last["counts"])  # active cams only
    assert set(m["boxes2d"]) <= {"main", "wide"} and m["cams"]["rear"] == {"status": "ABSENT", "age_s": None, "weight": 0.0, "res": None}
    # a STALE camera (main frame 0.6 s old, weight exp(-0.6/0.5) = 0.30) gets no boxes, a fresh one does
    _, sm = run(ages_fn=lambda t: dict(ages_at(t), main=0.6, wide=0.0))
    s_last = sm[-1]
    assert s_last["cams"]["main"]["status"] == "STALE" and 0 < s_last["cams"]["main"]["weight"] < 0.5, s_last["cams"]["main"]
    assert "main" not in s_last["boxes2d"] and s_last["boxes2d"].get("wide"), sorted(s_last["boxes2d"])
    bad = copy.deepcopy(last); bad["boxes2d"]["rear"] = [dict(last["boxes2d"]["main"][0])]
    bad["cams"]["rear"] = {"status": "STALE", "age_s": 0.9, "weight": 0.1, "res": [640, 400]}
    _raises(bad, "STALE")
    bad = copy.deepcopy(last); bad["boxes2d"]["main"][0]["id"] = 999
    _raises(bad, "not an object")
    bad = copy.deepcopy(last); bad["boxes2d"]["main"][0]["box"][2] = 10_000.0
    _raises(bad, "not inside")
    print(f"boxes2d: {sum(len(v) for v in last['boxes2d'].values())} boxes in {sorted(last['boxes2d'])}, ids = track ids, "
          f"inside image, edge-clipped, none behind / for masked cams OK")


def check_corridor() -> None:
    last = RUN[1][-1]
    cor = last["corridor"]
    pts = cor["pts"]
    assert 4 <= len(pts) <= oo.MAX_CORRIDOR_VERTS and pts[0] != pts[-1] and oo.polygon_simple(pts) and cor["conf"] >= 0.5
    # ego frame: mapped back to the world with the true pose, every vertex sits on the road edge (|x| = 3.6-4.0)
    px, py, h = pose(TS[-1])
    e = np.asarray(pts, dtype=np.float64)
    wx = px + e[:, 0] * math.cos(h) - e[:, 1] * math.sin(h)
    err = np.abs(np.abs(wx) - 3.8)
    assert err.max() < 0.6 and e[:, 1].min() < 5, (err.max(), e[:, 1].min())
    left = [e for e in last["edges"] if e["side"] == "left"]
    assert oo.corridor(left) == {"pts": [], "conf": 0.0}       # one side only: empty, no default
    short = [dict(e, pts=e["pts"][:2]) for e in last["edges"]]
    assert oo.corridor(short)["pts"] == []                    # too little support: empty
    crossed = [{"side": "left", "pts": [[2.0, 0.0], [2.0, 20.0]]}, {"side": "right", "pts": [[-2.0, 0.0], [-2.0, 20.0]]}]
    assert oo.corridor(crossed)["pts"] == []                  # left right of right: empty
    big = oo.corridor([{"side": "left", "pts": [[-3.0, y] for y in range(0, 200)]},
                       {"side": "right", "pts": [[3.0, y] for y in range(0, 200)]}])
    assert len(big["pts"]) <= oo.MAX_CORRIDOR_VERTS and oo.polygon_simple(big["pts"])
    bow = copy.deepcopy(last); bow["corridor"] = {"pts": [[0, 0], [4, 4], [4, 0], [0, 4]], "conf": 0.9}
    _raises(bow, "self-intersecting")
    low = copy.deepcopy(last); low["corridor"]["conf"] = 0.2
    _raises(low, "confidence floor")
    shut = copy.deepcopy(last); shut["corridor"]["pts"] = pts[:-1] + [pts[0]]
    _raises(shut, "do not repeat")
    print(f"corridor: {len(pts)} verts (cap {oo.MAX_CORRIDOR_VERTS}), conf {cor['conf']}, simple, implicitly closed, "
          f"vertices within {err.max():.2f} m of the true road edge; "
          f"one-sided / short / crossed -> empty OK")


def _raises(msg, frag: str) -> None:
    try:
        oo.validate_occ_out(msg)
    except oo.OccOutError as e:
        assert frag in str(e), (frag, str(e))
        return
    raise AssertionError(f"validator accepted a message that should fail on {frag!r}")


def check_reject() -> None:
    good = RUN[1][-1]
    cases = [
        (lambda m: m.update(extra=1), "unknown fields"),
        (lambda m: m["ego"].update(gt_pose=[0, 0]), "unknown fields"),
        (lambda m: m["objects"][0].update(yaw=0.1), "unknown fields"),
        (lambda m: m["cams"]["main"].update(img=0), "unknown fields"),
        (lambda m: m["counts"].update(dense=1), "unknown fields"),
        (lambda m: m["edges"][0].update(color="red"), "unknown fields"),
        (lambda m: m.pop("corridor"), "missing"),
        (lambda m: m["objects"][0].pop("heading"), "missing"),
        (lambda m: m.update(v=3), "v/type"),
        (lambda m: m["cells"].pop("observed"), "missing"),               # v2 must
        (lambda m: m["edges"][0].pop("observed"), "missing"),
        (lambda m: m["cells"]["observed"].__setitem__(0, 1), "not a bool"),
        (lambda m: m["edges"][0]["observed"].pop(), "bools"),
        (lambda m: m["ego"].update(speed_mps=float("nan")), "not finite"),
        (lambda m: m["ego"].update(source="gt"), "not in"),
        (lambda m: m["cells"]["prob"].__setitem__(0, 1.5), "outside"),
        (lambda m: m["cells"]["cls"].__setitem__(0, 7), "unknown class"),
        (lambda m: m["cells"]["ijk"].__setitem__(0, [10_000, 0, 0]), "outside the grid"),
        (lambda m: m["objects"][0].update(heading=[1.5, 0.0]), "outside"),
        (lambda m: m["objects"][0].update(heading=[0.6, 0.6]), "cos^2"),
        (lambda m: m["objects"][0].update(motion="flying"), "not in"),
        (lambda m: m["objects"][1].update(id=m["objects"][0]["id"]), "duplicate"),
        (lambda m: m["counts"].update(stored=m["counts"]["full"] + 1), "stored"),
        (lambda m: m["counts"].update(objects=9), "counts.objects"),
        (lambda m: m["edges"][0]["conf"].pop(), "values for"),
        (lambda m: m.update(lanes=[{"paint": "red", "pts": [[0, 0], [0, 1]], "conf": [1, 1], "observed": [True, False]}]), "not in"),
        (lambda m: m["cams"]["rear"].update(status="ABSENT", age_s=0.5), "ABSENT"),
        (lambda m: m.update(peak_vram_mb=812.0), "MEASURED"),
        (lambda m: m.update(vram_status="MEASURED"), "UNPROVEN"),
        (lambda m: (m["cells"]["ijk"].extend([[0, 0, 0]] * oo.MAX_CELLS), m["cells"]["cls"].extend([0] * oo.MAX_CELLS),
                    m["cells"]["prob"].extend([0.5] * oo.MAX_CELLS)), "over cap"),
    ]
    for mut, frag in cases:
        m = copy.deepcopy(good)
        mut(m)
        _raises(m, frag)
    # version gate: v1 = v2 minus observed (no memory cells); v1 carrying observed is rejected
    v2 = run(memory=False)[1][-1]
    v1 = copy.deepcopy(v2); v1["v"] = 1; v1["cells"].pop("observed")
    for e in v1["edges"]:
        e.pop("observed")
    oo.validate_occ_out(v1)
    bad = copy.deepcopy(v2); bad["v"] = 1
    _raises(bad, "unknown fields")
    print(f"reject: {len(cases)} malformed messages (unknown / missing fields at 6 levels, NaN, ranges, enums, caps, VRAM) raise OK")


def check_vram() -> None:
    _, msgs = run(vram_est=1234.5)
    m = msgs[-1]
    assert m["peak_vram_mb"] is None and m["vram_status"] == "UNPROVEN" and m["vram_est_mb_cpu"] == 1234.5
    assert all(x["peak_vram_mb"] is None for x in RUN[1]) and RUN[1][-1]["vram_est_mb_cpu"] is None
    ok = copy.deepcopy(m); ok.update(peak_vram_mb=2048.0, vram_status="MEASURED")
    oo.validate_occ_out(ok)  # only a measured number may fill it, and the estimate stays separate
    assert ok["vram_est_mb_cpu"] == 1234.5
    print("vram: peak_vram_mb None/UNPROVEN, CPU estimate only in vram_est_mb_cpu, MEASURED needs an explicit status OK")


def check_flat() -> None:
    g = DEFAULT_GRID
    fine = OccGrid(g.x_min, g.x_max, g.y_min, g.y_max, g.z_min, g.z_max, g.voxel_m / 4.0)
    assert fine.dense_cells == 64 * g.dense_cells
    base = RUN[1][-1]

    def dense_pred(grid, n):
        rng = np.random.default_rng(0)
        k = np.unique(rng.integers(0, grid.dense_cells, n))
        return OccPrediction(k, np.full(k.size, 0.8, np.float32), rng.integers(0, 3, k.size).astype(np.uint8), 0, k.size, (0, 0, k.size))

    sizes = {}
    for name, grid in (("1x", g), ("64x", fine)):
        rt = oo.OccRuntime(StubNet([dense_pred(grid, 30_000)], grid=grid), frustum_cells=grid.dense_cells // 2)
        m = rt.step([NoImage() for _ in CAMS], 0.0)
        assert m["counts"]["cells_out"] == oo.MAX_CELLS and m["counts"]["full"] == grid.dense_cells
        sizes[name] = len(oo.encode_occ_out(m))
    assert sizes["64x"] <= 1.1 * sizes["1x"] and max(sizes.values()) <= oo.MAX_BYTES, sizes
    print(f"flat: 30k stored cells -> {oo.MAX_CELLS} out; bytes 1x grid {sizes['1x']} vs 64x grid {sizes['64x']} "
          f"(cap {oo.MAX_BYTES}); scene message {len(oo.encode_occ_out(base))} bytes OK")


def check_runtime_net() -> None:
    try:
        import torch
    except ImportError:
        from gvd_test_env import skip_check

        skip_check("occ_out runtime net", "torch missing")
        return
    from python.perception.occ_sparse import SparseOccNet

    cams = [c.scaled(64) for c in rig_cams(CFG)]
    net = SparseOccNet(cams).eval()
    rt = oo.OccRuntime(net, max_points=4096, max_coarse=512)
    for i, t in enumerate((0.0, 0.1, 0.2)):
        m = rt.step([torch.rand(1, 3, c.height, c.width) for c in cams], t, ages_s={c.cam_id: 0.0 for c in cams})
    assert m["seq"] == 2 and m["ego"]["source"] == "vo" and m["timing_ms"]["predict"] > 0
    print(f"runtime: untrained SparseOccNet -> occ_out v2 valid, {m['counts']['stored']} stored, "
          f"{m['timing_ms']['total']:.0f} ms total (CPU, in-code timer) OK")


def main() -> int:
    check_panels()
    check_ego()
    check_vo_runtime_obs()
    check_heading()
    check_boxes2d()
    check_corridor()
    check_reject()
    check_vram()
    check_flat()
    check_runtime_net()
    print("test_occ_out: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
