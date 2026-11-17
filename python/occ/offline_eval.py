"""Offline occluded-cell evaluation (labels / eval side only; never imported by the runtime).

Occluded GT for a tick = cells labelled from ALL ground-truth depth cameras
minus the cells the RUNTIME cameras' own GT depth labels (i.e. cells no
runtime camera ray can see this tick). The model's stored cells are scored
on that set: occupancy IoU = TP / (TP + FP + FN) with TP = predicted ∩
occluded-occupied, FP = predicted ∩ occluded-free, FN = occluded-occupied
not predicted.

Every number is printed next to two dumb baselines run on the SAME held-out
ticks, and an occluded score that does not beat the best baseline is
reported as 0 (Skeptic's rule):

* ``prior``  per-cell occupancy prior from the TRAIN split (cell occupied in
  >= 50% of train ticks' full GT),
* ``nomem``  the same network with its temporal memory reset every tick.

Modes
-----
``--synthetic``  SYNTHETIC only (clearly labelled): train and held-out
    scenes come from different layout families (``SYNTH_SPLIT``: road width,
    parked-car pose, start offsets) so held-out layouts are never trained on.
``--shards``     real recorder shards. Requires ``--split split.json``
    ({"train": {"<map/region>": "<glob>", ...}, "eval": {...}}; map/region
    names must be disjoint) and ``--ckpt``. Refuses to print an occluded
    number without a split. Real-clip numbers are UNPROVEN until GVD runs this.

Causal: each clip runs tick by tick through ``OccRuntime`` (no lookahead).

Commands for GVD (see docs/specs/occ_sparse.md, "Eval and baseline commands")::

    python -m python.occ.offline_eval --synthetic --train-steps 300
    python -m python.occ.offline_eval --shards E:/occ_shards --split split.json --ckpt occ_sparse.pt --profile main_wide
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from python.occ.camera import CameraModel, cameras_from_config
from python.occ.grid import DEFAULT_GRID, VoxelGrid
from python.occ.labels import CamFrame, voxel_labels

SYNTH_SPLIT = {
    "family": "synthetic straight-road layouts (python.occ.synthetic.scene)",
    "train": {"road_half_w": 3.5, "car": {"x": 0.0, "y0": 15.0}, "starts": [0.0, 1.0, 2.0, 3.0]},
    "eval": {"road_half_w": 5.0, "car": {"x": -2.0, "y0": 22.0}, "starts": [0.37, 4.61]},
}


def _flat(idx: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    nx, ny, nz = grid.shape
    idx = np.asarray(idx, np.int64).reshape(-1, 3)
    return (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]


def occluded_sets(full: Mapping[str, np.ndarray], visible: Mapping[str, np.ndarray], grid: VoxelGrid = DEFAULT_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Flat ids of occluded-occupied and occluded-free cells: full-GT labels not labelled by the runtime cameras."""
    vis = np.union1d(_flat(visible["occ_idx"], grid), _flat(visible["free_idx"], grid))
    occ = np.setdiff1d(_flat(full["occ_idx"], grid), vis)
    free = np.setdiff1d(np.setdiff1d(_flat(full["free_idx"], grid), vis), occ)
    return occ, free


def occluded_iou(pred_flat: np.ndarray, occ: np.ndarray, free: np.ndarray) -> dict[str, float]:
    tp = np.intersect1d(pred_flat, occ).size
    fp = np.intersect1d(pred_flat, free).size
    fn = occ.size - tp
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / den if den else float("nan"), "n_occ": int(occ.size)}


def class_prior(full_labels: Sequence[Mapping[str, np.ndarray]], grid: VoxelGrid = DEFAULT_GRID, thr: float = 0.5) -> np.ndarray:
    """Flat ids occupied in >= ``thr`` of the train ticks' full GT (the per-cell prior baseline)."""
    if not full_labels:
        return np.zeros(0, np.int64)
    ids, cnt = np.unique(np.concatenate([_flat(l["occ_idx"], grid) for l in full_labels]), return_counts=True)
    return ids[cnt >= thr * len(full_labels)]


def summarize(rows: Sequence[Mapping[str, float]]) -> float:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")


def score_table(model_iou: float, baselines: Mapping[str, float], split: Mapping[str, Any], label: str) -> dict[str, Any]:
    best = max((v for v in baselines.values() if math.isfinite(v)), default=float("nan"))
    beats = math.isfinite(model_iou) and math.isfinite(best) and model_iou > best
    return {"label": label, "split": split, "occluded_iou_model": model_iou, "baselines": dict(baselines),
            "best_baseline": best, "beats_baseline": bool(beats), "occluded_score": model_iou if beats else 0.0}


# ------------------------------------------------------------------ synthetic
def _synthetic_tick(rt_cams, gt_cams, ego_y, fam):
    from python.occ.synthetic import SEM_CLASSES, scene

    box = {"x": fam["car"]["x"], "y0": fam["car"]["y0"], "w": 1.8, "h": 1.4, "l": 4.0}
    rgb, vis_frames, full_frames, paint, bars, car = {}, [], [], [], [], None
    for cid, cam in gt_cams.items():
        sc = scene(cam, road_half_w=fam["road_half_w"], box=box, ego_y=ego_y)
        fr = CamFrame(cam, sc["depth"], sc["sem"])
        full_frames.append(fr)
        if cid in rt_cams:
            rgb[cid] = sc["rgb"]
            vis_frames.append(fr)
        paint, bars, car = sc["paint"], sc["stop_bars"], sc["box"]
    kw = dict(paint=paint, stop_bars=bars, max_free=4000, seed=0)
    full = voxel_labels(full_frames, SEM_CLASSES, [car] if car else [], **kw)
    vis = voxel_labels(vis_frames, SEM_CLASSES, [car] if car else [], **kw)
    return rgb, full, vis


def run_synthetic(model, rt_cams: Mapping[str, CameraModel], gt_cams: Mapping[str, CameraModel], *, ticks: int = 8, step_m: float = 0.8,
                  split: Mapping[str, Any] = SYNTH_SPLIT, grid: VoxelGrid = DEFAULT_GRID) -> dict[str, Any]:
    """Held-out SYNTHETIC occluded eval with the prior and no-memory baselines (same ticks)."""
    from python.occ.runtime import OccRuntime

    train_full = []
    for y0 in split["train"]["starts"]:
        for k in range(ticks):
            train_full.append(_synthetic_tick(rt_cams, gt_cams, y0 + step_m * k, split["train"])[1])
    prior = class_prior(train_full, grid)
    rows = {"model": [], "nomem": [], "prior": []}
    for y0 in split["eval"]["starts"]:
        rt = OccRuntime(model, rt_cams)
        for k in range(ticks):
            rgb, full, vis = _synthetic_tick(rt_cams, gt_cams, y0 + step_m * k, split["eval"])
            occ, free = occluded_sets(full, vis, grid)
            imgs = {c: (rgb[c], 0.1 * k) for c in rt_cams}
            h = rt.perceive(0.1 * k, imgs)
            if k == 0:
                continue  # first tick has no memory in either arm
            rows["model"].append(occluded_iou(_flat(h["cells"]["idx"], grid), occ, free))
            rt0 = OccRuntime(model, rt_cams)  # memory reset every tick (ablation)
            h0 = rt0.perceive(0.1 * k, imgs)
            rows["nomem"].append(occluded_iou(_flat(h0["cells"]["idx"], grid), occ, free))
            rows["prior"].append(occluded_iou(prior, occ, free))
    res = score_table(summarize(rows["model"]), {"prior": summarize(rows["prior"]), "nomem": summarize(rows["nomem"])},
                      {"train": split["train"], "eval": split["eval"], "family": split["family"]}, "SYNTHETIC held-out layout family")
    res["ticks_scored"] = len(rows["model"])
    res["occluded_cells_per_tick"] = float(np.mean([r["n_occ"] for r in rows["model"]])) if rows["model"] else 0.0
    return res


# ------------------------------------------------------------------ shards
def load_split(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        raise SystemExit("refusing to report occluded numbers without --split (held-out maps/regions, GVD posts it)")
    sp = json.loads(Path(path).read_text(encoding="utf-8"))
    tr, ev = set(sp.get("train", {})), set(sp.get("eval", {}))
    if not tr or not ev or tr & ev:
        raise SystemExit(f"split must name disjoint non-empty train/eval maps or regions, got train={sorted(tr)} eval={sorted(ev)}")
    return sp


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--shards")
    ap.add_argument("--split")
    ap.add_argument("--ckpt")
    ap.add_argument("--profile", default="main_wide")
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    if a.synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from occ_net_fixtures import small_profile, small_rig, train_joint

        from python.occ.model import OccSparseNet

        ids = ("main", "wide")
        rt_cams = small_rig(ids, 128)
        gt_cams = small_rig(("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear"), 128)
        torch.manual_seed(0)
        m = OccSparseNet(profile=small_profile(ids, 128, k_max=32))
        train_joint(m, rt_cams, steps=a.train_steps)
        print(json.dumps(run_synthetic(m, rt_cams, gt_cams), default=float, indent=1))
        return 0
    split = load_split(a.split)
    if not (a.shards and a.ckpt):
        raise SystemExit("--shards and --ckpt are required for the real-data eval")
    print(json.dumps({"status": "UNPROVEN", "note": "real-shard occluded eval: run on GVD's box with recordings; none exist offline",
                      "split": split, "shards": a.shards, "ckpt": a.ckpt, "files": {k: sorted(glob.glob(v))[:3] for k, v in split["eval"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
