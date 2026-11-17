#!/usr/bin/env python3
"""front_cluster (main+wide, main at higher res) vs front3 (main+wide+narrow).

Opt-in (not a suite test). Same synthetic train clips and the same held-out
clips for both rigs, same seed, same steps. Labels come from the rendered
depth+semantic of narrow+main+wide at 128 px (training GT, identical for both).
Each rig sees the RGB resized to its own input sizes. At ``--scale 0.125`` the
rigs are front_cluster main 100 px + wide 80 px vs front3 3 x 80 px (the 800/640
ratio of the real profiles, at 1/8 size so it trains on a CPU).

Reports per rig: input sizes, params, the arithmetic memory ESTIMATE at full
size (``memory_estimate``), and held-out query IoU (occupied label cells vs
free-ray points), plus vehicle-cell recall near (< 30 m) and far (>= 30 m).

    python scripts/compare_front_cluster.py --steps 150 --scale 0.125
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def scenes(seed: int, n: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = float(rng.uniform(10.0, 48.0)); x = float(rng.uniform(-4.0, 4.0))
        out.append(((x - 0.9, y, 0.2), (x + 0.9, y + 4.4, 1.8)))
    return out


def build(tmp: Path, name: str, boxes, cfg):
    import numpy as np

    from python.data.occ_shards import ShardWriter
    from python.perception.occ_geometry import rig_cams
    from python.train.occ_labels import build_occ_labels
    from synthetic_occ import SEM_CLASSES, render

    ids = ("narrow", "main", "wide")
    full = {c.cam_id: c.scaled(128) for c in rig_cams(cfg) if c.cam_id in ids}
    samples = []
    with ShardWriter(tmp / name) as w:
        for i, (lo, hi) in enumerate(boxes):
            imgs, planes = {}, {}
            for cid, cam in full.items():
                d, s, bgr = render(cam, box_lo=lo, box_hi=hi)
                imgs[cid], planes[cid] = bgr, (d, s)
            lab = build_occ_labels(planes, full, sem_classes=SEM_CLASSES, rng=np.random.default_rng(i))
            w.write(f"{name}{i:04d}", imgs, lab.as_arrays(), {"recovery": i % 4 == 0, "t": float(i)})
            samples.append((imgs, lab))
    return tmp / name, samples


def evaluate(net, cams, samples):
    import numpy as np
    import torch

    from python.perception.occ_geometry import DEFAULT_GRID as G
    from python.train.occ_train import prep_image

    tp = fp = fn = 0
    near = [0, 0]; far = [0, 0]
    with torch.no_grad():
        for imgs, lab in samples:
            x = [torch.from_numpy(prep_image(imgs[c.cam_id], c)).unsqueeze(0) for c in cams]
            occ = G.centers_of_keys(lab.keys)
            pts = np.concatenate([occ, lab.free.astype(np.float64)]).astype(np.float32)
            p = torch.sigmoid(net(x, torch.from_numpy(pts).unsqueeze(0))["occ"][0]).numpy() > 0.5
            t = np.zeros(len(pts), bool); t[: len(occ)] = True
            tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum())
            veh = lab.cls == 2
            for m, acc in ((veh & (occ[:, 1] < 30.0), near), (veh & (occ[:, 1] >= 30.0), far)):
                acc[0] += int(p[: len(occ)][m].sum()); acc[1] += int(m.sum())
    return {"query_iou": tp / max(1, tp + fp + fn), "veh_recall_near": near[0] / max(1, near[1]),
            "veh_recall_far": far[0] / max(1, far[1]), "veh_cells_near": near[1], "veh_cells_far": far[1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--heldout", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    import yaml

    from python.perception.occ_sparse import PROFILES, SparseOccNet, memory_estimate, profile_cams
    from python.train.occ_train import train

    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    res = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": platform.node(), "device": "cpu",
           "scale": a.scale, "steps": a.steps, "train_clips": a.train, "heldout_clips": a.heldout}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tr_dir, _ = build(tmp, "train", scenes(1, a.train), cfg)
        _, held = build(tmp, "held", scenes(2, a.heldout), cfg)
        for prof in ("front_cluster", "front3"):
            cams = profile_cams(cfg, prof, scale=a.scale)
            torch.manual_seed(0)
            net = SparseOccNet(cams, config=PROFILES[prof]["config"])
            t0 = time.perf_counter()
            out = train(tr_dir, cams, steps=a.steps, batch_size=4, lr=3e-3, device="cpu", seed=0, net=net,
                        n_occ=256, n_free=256, n_coarse=64)
            ev = evaluate(out["net"].eval(), cams, held)
            res[prof] = {
                "cams_trained": {c.cam_id: [c.width, c.height] for c in cams},
                "cams_full_size": {c.cam_id: [c.width, c.height] for c in profile_cams(cfg, prof)},
                "params": out["params"],
                "mem_estimate_full_size_mb_fp32": memory_estimate(profile_cams(cfg, prof), config=PROFILES[prof]["config"]),
                "train_s": time.perf_counter() - t0,
                "train_loss_first5_last5": [float(np.mean([h["loss"] for h in out["history"][:5]])),
                                            float(np.mean([h["loss"] for h in out["history"][-5:]]))],
                "heldout": ev,
            }
            print(json.dumps({prof: res[prof]}), flush=True)
    res["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
