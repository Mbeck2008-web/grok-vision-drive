#!/usr/bin/env python3
"""Sparse occupancy (docs/specs/occ_sparse.md): labels, shards, mix, net.

Spec-derived checks. Each one names the guarantee it pins.

Labels (numpy only):
* G1 sparse: labels are voxel index lists, far smaller than the dense grid.
* G2 depth 0 and the clamp value 65535 add nothing (no occupied, no free).
* G3 a flat wall at view-Z 10 m: every occupied voxel is at the wall, every
  free voxel is in front of it, nothing behind it is labelled (unknown is
  never written as free).
* G4 no voxel is both occupied and free.
* G5 semantics: an allowlisted road name is ``drivable``; class 0 and names
  off the allowlist are ignore (255), never guessed.
* G6 vehicle boxes: a box with no length/width marks only the 0.8 m core
  disk (no 4.5 x 1.8 fill), class ``vehicle``, flow = speed * heading in
  the ego frame; flow is NaN off boxes.
* G7 more cameras with depth add voxels (any of the 8 cams can supply depth).

Shards (numpy + stdlib):
* G8 WebDataset layout: plain tar, members ``<key>.<ext>``, round trip exact,
  roll over at max_bytes, label npz round trip.
* G9 recovery mix: 5% recoveries in, weighted draw lands in [0.20, 0.30];
  a recording with no recoveries is not padded with fake ones.

Net (torch):
* G10 runtime is camera-only: forward takes (images, queries), no depth.
* G11 a camera that cannot see a query contributes nothing.
* G12 tiny synthetic overfit: occupancy accuracy > 0.95 on the training set.
* G13 coarse-to-fine inference evaluates at most coarse + max_fine queries.
* G14 ONNX export (opset 17) runs in onnxruntime and matches torch.
* G15 budget: inference upper bound <= 4 GB at 8 x 640x480 fp32; training
  estimate with fp16 activations <= 11 GB at the default batch.
"""
from __future__ import annotations

import io
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from python.perception.lanes import CameraPose  # noqa: E402
from python.train import occ_sparse as occ  # noqa: E402
from python.data import occ_shards as shards  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}".rstrip(), flush=True)
    if not ok:
        FAILS.append(name)


def flat_pose(cid: str = "main", yaw: float = 0.0) -> CameraPose:
    return CameraPose(cid, 0.0, 0.0, 1.2, yaw, 0.0, 34.0, 64, 48)


GRID = occ.OccGrid()


def wall_labels(depth_m: float = 10.0, cams=("main",)):
    poses = {c: flat_pose(c, yaw=0.0 if c == "main" else 180.0) for c in cams}
    depth = {c: np.full((48, 64), int(depth_m * 100), np.uint16) for c in cams}
    return occ.build_occ_labels(depth, poses, GRID, stride=2)


def test_labels() -> None:
    lab = wall_labels()
    centers = GRID.centers(lab.occ_idx)
    fcent = GRID.centers(lab.free_idx)
    dense = int(np.prod(GRID.dims))
    check("G1 sparse", 0 < len(lab.occ_idx) + len(lab.free_idx) < dense // 10,
          f"occ={len(lab.occ_idx)} free={len(lab.free_idx)} dense={dense}")
    check("G1 int16 index lists", lab.occ_idx.dtype == np.int16 and lab.occ_idx.shape[1] == 3)
    # view-Z depth: the wall plane is y = 10 m in the ego frame for a level forward camera
    check("G3 occupied at wall", len(centers) > 0 and bool(np.all(np.abs(centers[:, 1] - 10.0) <= GRID.voxel_m)),
          f"y range {centers[:, 1].min() if len(centers) else 'na'}..{centers[:, 1].max() if len(centers) else 'na'}")
    check("G3 free in front", len(fcent) > 0 and bool(np.all(fcent[:, 1] < 10.0)),
          f"max free y {fcent[:, 1].max() if len(fcent) else 'na'}")
    beyond = np.concatenate([centers[:, 1], fcent[:, 1]]) > 10.0 + GRID.voxel_m
    check("G3 nothing behind wall", not bool(beyond.any()))
    occ_set = {tuple(v) for v in lab.occ_idx.tolist()}
    free_set = {tuple(v) for v in lab.free_idx.tolist()}
    check("G4 occ and free disjoint", not (occ_set & free_set))

    for bad in (0, occ.DEPTH_CM_MAX):
        d = {"main": np.full((48, 64), bad, np.uint16)}
        lab0 = occ.build_occ_labels(d, {"main": flat_pose()}, GRID, stride=2)
        check(f"G2 depth {bad} adds nothing", len(lab0.occ_idx) == 0 and len(lab0.free_idx) == 0)

    sem = {"main": np.zeros((48, 64), np.uint8)}
    sem["main"][:, :32] = 1
    sem["main"][:, 32:48] = 2
    names = {"0": "unknown", "1": "ROAD", "2": "BUILDINGS"}
    lab_s = occ.build_occ_labels({"main": np.full((48, 64), 1000, np.uint16)}, {"main": flat_pose()},
                                 GRID, stride=2, sem_by_cam=sem, sem_names=names)
    cls = lab_s.occ_cls
    xs = GRID.centers(lab_s.occ_idx)[:, 0]
    left = cls[xs < -0.6]
    right = cls[xs > 0.6]
    check("G5 allowlisted name is drivable", len(left) > 0 and bool(np.all(left == occ.CLS_DRIVABLE)))
    check("G5 off-allowlist and class 0 ignored", len(right) > 0 and bool(np.all(right == occ.CLS_IGNORE)))

    box = {"x": 0.0, "y": 20.0, "yaw": math.pi / 2, "speed": 5.0}
    lab_b = occ.build_occ_labels({}, {}, GRID, boxes=[box])
    bc = GRID.centers(lab_b.occ_idx)
    r = np.hypot(bc[:, 0], bc[:, 1] - 20.0) if len(bc) else np.zeros(0)
    check("G6 core disk only", len(bc) > 0 and bool(np.all(r <= occ.VEHICLE_CORE_M + 1e-6)), f"n={len(bc)}")
    check("G6 class vehicle", bool(np.all(lab_b.occ_cls == occ.CLS_VEHICLE)))
    check("G6 flow speed*heading", bool(np.allclose(lab_b.flow, [[0.0, 5.0]] * len(bc), atol=1e-5)))
    sized = dict(box, length=4.0, width=2.0)
    lab_sz = occ.build_occ_labels({}, {}, GRID, boxes=[sized])
    check("G6 sized box fills more", len(lab_sz.occ_idx) > len(lab_b.occ_idx))
    check("G6 flow NaN off boxes", bool(np.all(np.isnan(lab.flow))))

    two = wall_labels(cams=("main", "rear"))
    check("G7 more depth cams add voxels", len(two.occ_idx) > len(lab.occ_idx))


def test_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        w = shards.ShardWriter(Path(tmp), "occ", max_bytes=3000)
        rng = np.random.default_rng(0)
        samples = {}
        for k in range(6):
            key = f"rec0-{k:06d}"
            payload = {"main.png": rng.bytes(900), "json": b'{"i": %d}' % k}
            samples[key] = payload
            w.write(key, payload)
        paths = w.close()
        check("G8 rollover", len(paths) >= 2, f"{len(paths)} shards")
        back = {key: s for key, s in shards.iter_samples(paths)}
        check("G8 round trip", back == samples)
        with tarfile.open(paths[0]) as tf:
            names = tf.getnames()
        check("G8 key.ext members", all(n.count(".") >= 1 and n.split(".")[0].startswith("rec0-") for n in names))
    lab = wall_labels()
    blob = shards.pack_labels(lab)
    lab2 = shards.unpack_labels(blob)
    same = (np.array_equal(lab.occ_idx, lab2.occ_idx) and np.array_equal(lab.free_idx, lab2.free_idx)
            and np.array_equal(lab.occ_cls, lab2.occ_cls)
            and np.array_equal(np.isnan(lab.flow), np.isnan(lab2.flow)))
    check("G8 labels npz round trip", same)

    rec = np.zeros(1000, bool)
    rec[:50] = True
    wts = shards.recovery_mix_weights(rec, target=0.25)
    frac = float(wts[rec].sum() / wts.sum())
    draws = np.random.default_rng(1).choice(len(rec), size=20000, p=wts / wts.sum())
    drawn = float(rec[draws].mean())
    check("G9 mix target", abs(frac - 0.25) < 1e-6 and 0.20 <= drawn <= 0.30, f"exp={frac:.3f} drawn={drawn:.3f}")
    none = shards.recovery_mix_weights(np.zeros(10, bool), target=0.25)
    check("G9 no fake recoveries", bool(np.allclose(none, 1.0)))
    check("G9 recover tag", shards.is_recovery_row({"tags": {"excursion": {"phase": "recover"}}})
          and shards.is_recovery_row({"tags": {"recover": True}})
          and not shards.is_recovery_row({"tags": {"excursion": {"phase": "weave"}}}))


def test_net() -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        for g in ("G10", "G11", "G12", "G13", "G14", "G15"):
            skip_check(g, "torch not importable")
        return
    import inspect

    torch.manual_seed(0)
    poses = [flat_pose("main"), flat_pose("rear", yaw=180.0)]
    net = occ.SparseOccNet(poses, GRID, width=16)
    params = list(inspect.signature(net.forward).parameters)
    check("G10 camera-only forward", params == ["images", "queries"], str(params))

    img = torch.rand(1, 2, 3, 48, 64)
    q = torch.tensor([[[0.0, 10.0, 1.0], [0.0, -10.0, 1.0]]])
    with torch.no_grad():
        base = net(img, q)
        img2 = img.clone()
        img2[:, 1] = torch.rand(3, 48, 64)  # rear camera changes; it cannot see y=+10
        out2 = net(img2, q)
    check("G11 unseen camera adds nothing", torch.allclose(base["occ"][0, 0], out2["occ"][0, 0], atol=1e-6)
          and not torch.allclose(base["occ"][0, 1], out2["occ"][0, 1], atol=1e-6))

    # overfit: two scenes, wall at 8 m vs 16 m, the image tells which
    torch.manual_seed(0)
    netf = occ.SparseOccNet([flat_pose("main")], GRID, width=16)
    scenes = []
    for d, shade in ((8.0, 0.2), (16.0, 0.8)):
        lab = occ.build_occ_labels({"main": np.full((48, 64), int(d * 100), np.uint16)},
                                   {"main": flat_pose()}, GRID, stride=4)
        imgs = torch.full((1, 1, 3, 48, 64), shade)
        scenes.append((imgs, occ.sample_queries(lab, GRID, n_occ=96, n_free=96, rng=np.random.default_rng(0))))
    opt = torch.optim.Adam(netf.parameters(), lr=3e-3)
    first = last = None
    for step in range(300):
        tot = 0.0
        for imgs, b in scenes:
            out = netf(imgs, b["queries"][None])
            loss = occ.occ_loss(out, {k: v[None] for k, v in b.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        first = tot if first is None else first
        last = tot
    accs = []
    with torch.no_grad():
        for imgs, b in scenes:
            p = netf(imgs, b["queries"][None])["occ"][0]
            m = b["occ_mask"] > 0
            accs.append(float(((p[m] > 0) == (b["occ"][m] > 0.5)).float().mean()))
    check("G12 overfit", min(accs) > 0.95 and last < first * 0.2, f"loss {first:.3f}->{last:.3f} acc={accs}")

    res = occ.infer_sparse(netf, scenes[0][0], tau=0.5, max_fine=512)
    ncoarse = int(np.prod(GRID.coarse_dims))
    check("G13 query budget", res["evaluated"] <= ncoarse + 512 and len(res["idx"]) <= 512,
          f"evaluated={res['evaluated']} coarse={ncoarse}")
    fy = GRID.centers(res["idx"].numpy())[:, 1] if len(res["idx"]) else np.zeros(0)
    check("G13 fine cells near the 8 m wall", len(fy) > 0 and float(np.median(fy)) < 10.0,
          f"n={len(fy)} median_y={float(np.median(fy)) if len(fy) else 'na'}")

    try:
        import onnx
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        skip_check("G14", "onnx/onnxruntime not importable")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "occ.onnx"
            occ.export_onnx(netf, path, n_queries=64, image_hw=(48, 64))
            onnx.checker.check_model(str(path))
            qx = torch.tensor(GRID.centers(res["idx"][:64].numpy()) if len(res["idx"]) >= 64
                              else np.zeros((64, 3)), dtype=torch.float32)[None]
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            got = sess.run(None, {"images": scenes[0][0].numpy(), "queries": qx.numpy()})
            with torch.no_grad():
                ref = netf(scenes[0][0], qx)
            names = [o.name for o in sess.get_outputs()]
            check("G14 onnx parity", names[:1] == ["occ"] and np.allclose(got[0], ref["occ"].numpy(), atol=1e-4))

    full = occ.SparseOccNet(None, GRID)
    budget = occ.memory_budget(full, n_cams=8, hw=(480, 640))
    check("G15 infer <= 4 GB", budget["infer_bytes"] <= 4 * 2**30, f"{budget['infer_bytes'] / 2**30:.3f} GB")
    check("G15 train fp16 <= 11 GB", budget["train_fp16_bytes"] <= 11 * 2**30,
          f"{budget['train_fp16_bytes'] / 2**30:.3f} GB (batch {budget['batch']})")


def main() -> int:
    test_labels()
    test_shards()
    test_net()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("test_occ_sparse: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
