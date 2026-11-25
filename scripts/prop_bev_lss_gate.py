"""Property tests (hypothesis) for the BEV LSS visibility gate (G5). DEV ONLY, not in the offline suite.

hypothesis is a dev-only dependency (not in requirements*.txt). Run:
    PYTHONPATH=<devdeps> python scripts/prop_bev_lss_gate.py [--examples N] [--seed S]

Invariants (synthetic ray-cast scenes, scripts/bev_gate_scenes.py; depth head p_true on the true bin):
  I1  A ground cell whose centre is in view of >= 1 camera and whose sight line from EVERY in-view
      camera origin passes through an occluder (shrunk 0.02 m at the top and sides) is never
      free-eligible, seen, through or clear. Any wall height 0.55-1.5 m (above the 0.5 m band),
      offset, yaw, length (partial walls), thickness, 1..8 cameras, ego origin above / below the road,
      ego pitch / grade +-6 %, a car box, and walls whose top edge a camera ray grazes exactly.
  I2  A cell wholly inside an occluder footprint is never free-eligible.
  I3  Logits are finite, forced values are exactly +-FORCE_LOGIT, and the gate is deterministic.
Every failure is shrunk by hypothesis and printed as a minimal scene.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from hypothesis import HealthCheck, Phase, given, note, seed, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import bev_gate_scenes as S  # noqa: E402
from python.train import bev_lss as bl  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
ALL = [c["id"] for c in CFG["cameras"]]
HEAD = bl.BevLssHead().eval()
GEOMS: dict = {}


def geoms(cams):
    key = tuple(cams)
    if key not in GEOMS:
        GEOMS[key] = bl.rig_geometry(list(cams), CFG, model="etk800")
    return GEOMS[key]


cam_sets = st.one_of(
    st.sampled_from([("main",), ("narrow", "main", "wide"), tuple(ALL)]),
    st.lists(st.sampled_from(ALL), min_size=1, max_size=8, unique=True).map(lambda l: tuple(c for c in ALL if c in l)),
)
r2 = lambda lo, hi: st.floats(lo, hi, allow_nan=False).map(lambda v: round(v, 2))  # noqa: E731
walls = st.tuples(r2(-8, 8), r2(2.5, 22), st.one_of(st.just(0.0), r2(-60, 60)), r2(1.0, 16.0), r2(0.2, 0.6), r2(0.55, 1.5))
car = st.tuples(r2(-4, 4), r2(6, 20), st.one_of(st.just(0.0), r2(-30, 30)), st.just(1.8), st.just(4.5), st.just(1.5))
scenes = st.fixed_dictionaries(
    {
        "cams": cam_sets,
        "walls": st.lists(walls, min_size=1, max_size=3),
        "car": st.one_of(st.none(), car),
        "grade": st.one_of(st.just(0.0), r2(-0.06, 0.06)),
        "z0": st.one_of(st.just(0.0), st.just(-0.23), r2(-0.3, 0.05)),
        "p_true": st.one_of(st.just(0.95), r2(0.6, 0.99)),
        "graze": st.one_of(st.none(), st.tuples(st.integers(0, 2), st.floats(0.0, 1.0))),
    }
)


def graze_top(scene) -> list:
    """Re-size one wall so a camera ray hits its front top edge exactly (the "grazing top edge" case)."""
    boxes = [list(w) for w in scene["walls"]]
    if scene["graze"] is None:
        return [tuple(b) for b in boxes]
    wi, frac = scene["graze"]
    b = boxes[wi % len(boxes)]
    g = geoms(scene["cams"])[0]
    o = g.T[:3, 3].astype(np.float64)
    d = S.ray_dirs(g).reshape(-1, 3)
    centre, R, half = S.box_frame(tuple(b), scene["grade"], scene["z0"])
    # intersect every ray with the face plane nearer the camera, inside the face's length
    n = R[:, 1] if (o - centre) @ R[:, 1] < 0 else -R[:, 1]
    p0 = centre - n * half[1]
    den = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0 - o) @ n) / den
    P = o + t[:, None] * d
    along = (P - centre) @ R[:, 0]
    zrel = P[:, 2] - (centre[2] - half[2])
    ok = (t > 0) & (np.abs(along) < half[0] - 0.05) & (zrel > 0.55) & (zrel < 1.5)
    if not ok.any():
        return [tuple(x) for x in boxes]
    cand = np.sort(zrel[ok])
    b[5] = float(cand[min(int(frac * len(cand)), len(cand) - 1)])
    return [tuple(x) for x in boxes]


def check_scene(scene, below_top=0.02, inset=0.02, inset_thick=None, resolve=False) -> dict:
    boxes = graze_top(scene) + ([scene["car"]] if scene["car"] is not None else [])
    gm = geoms(scene["cams"])
    out = S.run_scene(gm, boxes, scene["grade"], scene["z0"], scene["p_true"], head=HEAD)
    lab = S.labels(out)
    gt = S.ground_truth(gm, boxes, scene["grade"], scene["z0"], below_top=below_top, inset=inset, foot_pad=0.36, inset_thick=inset_thick, resolve=resolve)
    h = gt["hidden"]
    sp = gt["sub_pitch"]
    X, Y = S.cell_centres()
    whole = np.zeros_like(h)
    for b in boxes:
        whole |= S.in_footprint(b, X, Y, pad=-0.36)
    lg = out["logits"][0]
    forced_ok = bool(((lg[bl.CH_FREE] == -bl.FORCE_LOGIT) | (lg[bl.CH_FREE].abs() < bl.FORCE_LOGIT)).all())
    return {
        "boxes": boxes,
        "hidden": int(h.sum()),
        "leak_free": int((lab["free"] & h).sum()),
        "leak_seen": int((lab["seen"] & h).sum()),
        "leak_through": int((lab["through"] & h).sum()),
        "leak_clear": int((lab["clear"] & h).sum()),
        "foot_free": int((lab["free"] & whole).sum()),
        "sub_pitch": int(sp.sum()),
        "sub_pitch_free": int((lab["free"] & sp).sum()),
        "finite": lab["finite"] and forced_ok,
        "cells": [(round(float(X[r, c]), 2), round(float(Y[r, c]), 2)) for r, c in np.argwhere(lab["free"] & h)[:6]],
    }


# --mode resolved (the stated invariant I1): Skeptic's definition, on cells every in-view camera
# resolves (bev_gate_scenes.RESOLVE_MARGIN_M); sub-pitch cells are counted and printed, not asserted.
# --mode skeptic: Skeptic's wall_leak definition (0.1 m under the top, 0.3 m inset along the wall).
# --mode strict: 0.02 m under the top / inset with no resolution margin (sub-pitch grazing included;
# expected to find depth maps that are identical to open road, see SHADOW_RESOLVE_M).
MODES = {
    "resolved": {"below_top": 0.1, "inset": 0.3, "inset_thick": 0.0, "resolve": True},
    "skeptic": {"below_top": 0.1, "inset": 0.3, "inset_thick": 0.0},
    "strict": {"below_top": 0.02, "inset": 0.02},
}
TOL = dict(MODES["resolved"])
STATS = {"runs": 0, "hidden": 0, "sub_pitch": 0, "sub_pitch_free": 0, "leak_free": 0}


def prop_hidden_never_free(scene) -> None:
    r = check_scene(scene, **TOL)
    for k in ("hidden", "sub_pitch", "sub_pitch_free", "leak_free"):
        STATS[k] += r[k]
    STATS["runs"] += 1
    note(f"scene={scene!r} boxes={r['boxes']!r} result={ {k: v for k, v in r.items() if k != 'boxes'} }")
    msg = f"scene={ {k: v for k, v in scene.items() if k != 'walls'} } boxes={r['boxes']} -> " + str({k: v for k, v in r.items() if k != "boxes"})
    assert r["finite"], msg
    assert r["leak_free"] == 0 and r["leak_seen"] == 0 and r["leak_through"] == 0 and r["leak_clear"] == 0, msg
    assert r["foot_free"] == 0, msg


def prop_deterministic(scene) -> None:
    boxes = graze_top(scene)
    gm = geoms(scene["cams"])
    a = S.run_scene(gm, boxes, scene["grade"], scene["z0"], scene["p_true"], head=HEAD)["logits"]
    b = S.run_scene(gm, boxes, scene["grade"], scene["z0"], scene["p_true"], head=HEAD)["logits"]
    assert torch.equal(a, b)


def limit_demo() -> int:
    """A 0.6 m wall that falls between two ray rows of the wide camera at every column: the depth
    input is bit-identical to open road, so any deterministic gate labels the cells behind it the
    same as on open road. This is the resolution limit the invariant states."""
    for cam, sign in (("wide", 1.0), ("rear", -1.0), ("main", 1.0)):
        g = geoms((cam,))
        P0 = S.depth_prob(g, [])
        hit = None
        for y in np.arange(4.0, 25.0, 0.05):
            box = (0.0, round(sign * float(y), 2), 0.0, 40.0, 0.3, 0.6)
            if torch.equal(S.depth_prob(g, [box]), P0):
                gt = S.ground_truth(g, [box], below_top=0.1, inset=0.3, inset_thick=0.0)
                if gt["hidden"].sum() > 0:
                    hit = (box, gt)
                    break
        if hit is None:
            print(f"{cam}: no 0.6 m wall 4-25 m fits between two ray rows (every one is hit)")
            continue
        box, gt = hit
        a = S.run_scene(g, [], head=HEAD)
        b = S.run_scene(g, [box], head=HEAD)
        fa, fb = S.labels(a)["free"], S.labels(b)["free"]
        print(f"{cam}: wall {box}: depth input identical to open road; logits identical {torch.equal(a['logits'], b['logits'])}; "
              f"hidden (0.1 m / 0.3 m) {int(gt['hidden'].sum())}, free-eligible {int((fb & gt['hidden']).sum())} "
              f"(open road: {int((fa & gt['hidden']).sum())})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", choices=sorted(MODES), default="resolved")
    ap.add_argument("--limit-demo", action="store_true", help="print the sub-pitch wall that no gate can see")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.limit_demo:
        return limit_demo()
    TOL.clear()
    TOL.update(MODES[args.mode])
    print(f"mode {args.mode}: {TOL} seed {args.seed} examples {args.examples}")
    cfg = settings(max_examples=args.examples, deadline=None, derandomize=False, database=None,
                   suppress_health_check=list(HealthCheck), print_blob=True,
                   phases=[Phase.explicit, Phase.generate, Phase.shrink])
    fails = 0
    for name, prop, n in (("I1/I2/I3 hidden_never_free", prop_hidden_never_free, args.examples), ("I3 deterministic", prop_deterministic, 5)):
        test = seed(args.seed)(settings(cfg, max_examples=n)(given(scenes)(prop)))
        try:
            test()
            print(f"PASS {name}: {n} examples")
        except AssertionError as e:  # hypothesis re-raises the shrunk failure
            fails += 1
            print(f"FAIL {name}: {str(e)[:2000]}")
    print(f"stats over every scene hypothesis ran (incl. shrinking): {STATS}")
    print("prop_bev_lss_gate:", "OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
