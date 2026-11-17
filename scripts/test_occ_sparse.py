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
from python.train import occ_labels as L  # noqa: E402

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
    return L.build_occ_labels(depth, poses, GRID, stride=2)


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

    for bad in (0, L.DEPTH_CM_MAX):
        d = {"main": np.full((48, 64), bad, np.uint16)}
        lab0 = L.build_occ_labels(d, {"main": flat_pose()}, GRID, stride=2)
        check(f"G2 depth {bad} adds nothing", len(lab0.occ_idx) == 0 and len(lab0.free_idx) == 0)

    sem = {"main": np.zeros((48, 64), np.uint8)}
    sem["main"][:, :32] = 1
    sem["main"][:, 32:48] = 2
    names = {"0": "unknown", "1": "ROAD", "2": "BUILDINGS"}
    lab_s = L.build_occ_labels({"main": np.full((48, 64), 1000, np.uint16)}, {"main": flat_pose()},
                                 GRID, stride=2, sem_by_cam=sem, sem_names=names)
    cls = lab_s.occ_cls
    xs = GRID.centers(lab_s.occ_idx)[:, 0]
    left = cls[xs < -0.6]
    right = cls[xs > 0.6]
    check("G5 allowlisted name is drivable", len(left) > 0 and bool(np.all(left == L.CLS_DRIVABLE)))
    check("G5 off-allowlist and class 0 ignored", len(right) > 0 and bool(np.all(right == L.CLS_IGNORE)))

    box = {"x": 0.0, "y": 20.0, "yaw": math.pi / 2, "speed": 5.0}
    lab_b = L.build_occ_labels({}, {}, GRID, boxes=[box])
    bc = GRID.centers(lab_b.occ_idx)
    r = np.hypot(bc[:, 0], bc[:, 1] - 20.0) if len(bc) else np.zeros(0)
    check("G6 core disk only", len(bc) > 0 and bool(np.all(r <= L.VEHICLE_CORE_M + 1e-6)), f"n={len(bc)}")
    check("G6 class vehicle", bool(np.all(lab_b.occ_cls == L.CLS_VEHICLE)))
    check("G6 flow speed*heading", bool(np.allclose(lab_b.flow, [[0.0, 5.0]] * len(bc), atol=1e-5)))
    sized = dict(box, length=4.0, width=2.0)
    lab_sz = L.build_occ_labels({}, {}, GRID, boxes=[sized])
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
    camera_only = {"images", "queries", "mem_feat", "mem_xf", "mem_valid", "cam_valid", "cam_age", "mem_age"}
    check("G10 camera-only forward (images + own cached features/timing; no pose/depth/label input)",
          params[:2] == ["images", "queries"] and set(params) <= camera_only, str(params))

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
        lab = L.build_occ_labels({"main": np.full((48, 64), int(d * 100), np.uint16)},
                                   {"main": flat_pose()}, GRID, stride=4)
        imgs = torch.full((1, 1, 3, 48, 64), shade)
        scenes.append((imgs, L.sample_queries(lab, GRID, n_occ=96, n_free=96, rng=np.random.default_rng(0),
                                              n_coarse=48), lab))
    opt = torch.optim.Adam(netf.parameters(), lr=3e-3)
    first = last = None
    for step in range(300):
        tot = 0.0
        for imgs, b, _ in scenes:
            out = netf(imgs, b["queries"][None])
            loss = L.occ_loss(out, {k: v[None] for k, v in b.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        first = tot if first is None else first
        last = tot
    accs = []
    with torch.no_grad():
        for imgs, b, _ in scenes:
            p = netf(imgs, b["queries"][None])["occ"][0]
            m = b["occ_mask"] > 0
            accs.append(float(((p[m] > 0) == (b["occ"][m] > 0.5)).float().mean()))
    check("G12 overfit", min(accs) > 0.95 and last < first * 0.2, f"loss {first:.3f}->{last:.3f} acc={accs}")

    res = occ.infer_sparse(netf, scenes[0][0], tau=0.5, max_fine=512)
    ncoarse = int(np.prod(GRID.coarse_dims))
    check("G13 query budget", res["evaluated"] <= ncoarse + 512 and len(res["idx"]) <= 512,
          f"evaluated={res['evaluated']} coarse={ncoarse}")
    from python.perception.occ_viz import occ_to_viz, validate_v1

    vmsg = occ_to_viz(res, grid=GRID, t=0.0)
    check("G20 net output -> valid v1 message", not validate_v1(vmsg) and len(vmsg["occ"]["cells"]) == len(res["idx"]),
          f"cells={len(vmsg['occ']['cells'])}")
    res = occ.infer_sparse(netf, scenes[0][0], tau=0.5)  # default fine budget for the quality check
    # space behind the first surface is unknown (never labelled), so judge only labelled cells:
    # returned cells must not be labelled-free, and they must find the wall
    lab0 = scenes[0][2]
    rows = lambda a: set(map(tuple, np.asarray(a, np.int64).reshape(-1, 3).tolist()))  # noqa: E731
    got, free0, wall0 = rows(res["idx"].numpy()), rows(lab0.free_idx), rows(lab0.occ_idx)
    fp = len(got & free0) / max(1, len(got))
    rec = len(got & wall0) / max(1, min(len(wall0), 512))
    check("G13 fine cells: <=5% in labelled free space, >=50% wall recall", len(got) > 0 and fp <= 0.05 and rec >= 0.5,
          f"n={len(got)} free_hit={fp:.3f} wall_recall={rec:.3f} wall={len(wall0)}")

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
            with torch.no_grad():
                f0 = netf.encode(scenes[1][0])
                mf = torch.stack([f0, f0 * 0.5], 1)
                mx = torch.tensor([[[1.0, 0, 0.3], [0, 1.0, -1.0]], [[1.0, 0, 0], [0, 1.0, -2.0]]])[None]
                mv = torch.tensor([[1.0, 0.0]])
                ref = netf(scenes[0][0], qx, mf, mx, mv)
            got = sess.run(None, {"images": scenes[0][0].numpy(), "queries": qx.numpy(), "mem_feat": mf.numpy(),
                                  "mem_xf": mx.numpy(), "mem_valid": mv[:, :, None].numpy(),
                                  "cam_valid": np.ones((1, 1), np.float32), "cam_age": np.zeros((1, 1), np.float32),
                                  "mem_age": np.zeros((1, 2, 1), np.float32)})
            names = [o.name for o in sess.get_outputs()]
            check("G14 onnx parity (with temporal memory)", names[:1] == ["occ"] and np.allclose(got[0], ref["occ"].numpy(), atol=1e-4))

    full = occ.SparseOccNet(None, GRID)
    budget = occ.memory_budget(full, n_cams=8, hw=(480, 640))
    check("G15 infer <= 4 GB", budget["infer_bytes"] <= 4 * 2**30, f"{budget['infer_bytes'] / 2**30:.3f} GB")
    check("G15 train fp16 <= 11 GB", budget["train_fp16_bytes"] <= 11 * 2**30,
          f"{budget['train_fp16_bytes'] / 2**30:.3f} GB (batch {budget['batch']})")


def test_runtime_isolation() -> None:
    """G16: the runtime net module never pulls ground-truth code."""
    import subprocess

    code = ("import sys; import python.train.occ_sparse as m; m.SparseOccNet; "
            "bad=[k for k in sys.modules if k.split('.')[-1] in "
            "('occ_labels','raw_frames','sensor_aux','vehicle_labels','occ_shards','train_occ')]; print(bad)")
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        code = code.replace(" m.SparseOccNet;", "")
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    check("G16 occ_sparse imports no GT modules", res.returncode == 0 and res.stdout.strip() == "[]",
          (res.stdout + res.stderr).strip()[-300:])
    src = (ROOT / "python" / "train" / "occ_sparse.py").read_text(encoding="utf-8")
    check("G16 no GT names in net source", not any(w in src for w in ("occ_labels", "raw_frames", "_depth", "_sem\"")))


def _fake_recording(root: Path, n: int, recover_every: int) -> Path:
    import json

    import cv2

    rec = root / "rec_fake"
    for cam in ("main", "main_depth", "main_sem"):
        (rec / "raw" / cam).mkdir(parents=True, exist_ok=True)
    (rec / "sem_classes.json").write_text(json.dumps(
        {"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "none", "7": "ASPHALT"}}), encoding="utf-8")
    rows = []
    for i in range(n):
        img = np.full((48, 64, 3), 40 + 10 * i, np.uint8)
        cv2.imwrite(str(rec / "raw" / "main" / f"{i:06d}.png"), img)
        cv2.imwrite(str(rec / "raw" / "main_depth" / f"{i:06d}.png"), np.full((48, 64), 800 + 100 * i, np.uint16))
        cv2.imwrite(str(rec / "raw" / "main_sem" / f"{i:06d}.png"), np.full((48, 64), 7, np.uint8))
        tags = {"excursion": {"phase": "recover"}} if recover_every and i % recover_every == 0 else {}
        rows.append({"i": i, "tags": tags, "pos": [0.0, 1.5 * i, 0.0], "dir": [0.0, 1.0, 0.0]})
    (rec / "state.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rec


def test_pipeline() -> None:
    """G17 shards-only loader; G18 sampler enforces the recovery mix; G19 one train step from shards."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        for g in ("G17", "G18", "G19"):
            skip_check(g, "torch not importable")
        return
    import builtins
    import shutil

    from python.train import train_occ as T

    poses = {"main": flat_pose("main")}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rec = _fake_recording(td, n=12, recover_every=12)  # 1 of 12 ticks is a recovery
        stats = T.build_shards(rec, td / "shards", poses=poses, stride=4, max_bytes=40_000)
        check("G17 build wrote shards", stats["written"] == 12 and stats["recoveries"] == 1
              and len(stats["shards"]) >= 2, str({k: v for k, v in stats.items() if k != "shards"}))
        shutil.rmtree(rec)  # the loader must not need the recording
        idx = shards.ShardIndex(sorted((td / "shards").glob("*.tar")))
        ds = T.OccShardDataset(idx, ["main"], GRID, long_side=64, n_occ=32, n_free=32)
        opened: list[str] = []
        real_open = builtins.open

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)

        builtins.open = spy
        try:
            item = ds[3]
        finally:
            builtins.open = real_open
        check("G17 loader reads only .tar", bool(opened) and all(f.endswith(".tar") for f in opened), str(opened[:4]))
        check("G17 image from shard bytes", float(item["cam_valid"][0]) == 1.0
              and abs(float(item["images"][0].mean()) - 70 / 255) < 0.02)
        check("G17 labels from shard", float(item["occ_mask"].sum()) > 0)
        try:
            shards.ShardIndex([td / "x.png"])
            png_ok = False
        except ValueError:
            png_ok = True
        check("G17 refuses non-tar input", png_ok)

        sampler = T.make_sampler(ds, 0.25, num_samples=8000, seed=1)
        drawn = np.array(list(iter(sampler)))
        frac = float(ds.recovery[drawn].mean())
        check("G18 sampler draws 20-30% recoveries", 0.20 <= frac <= 0.30 and float(ds.recovery.mean()) < 0.1,
              f"in={ds.recovery.mean():.3f} drawn={frac:.3f}")

        torch.manual_seed(0)
        net = occ.SparseOccNet([poses["main"]], GRID, width=8)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, sampler=sampler)
        batch = next(iter(loader))
        check("G19 temporal memory from shards", float(ds[5]["mem_valid"].sum()) == 2.0
              and float(ds[0]["mem_valid"].sum()) == 0.0
              and abs(float(ds[5]["mem_xf"][0, 1, 2]) - 1.5) < 1e-4, str(ds[5]["mem_xf"][0].tolist()))
        out = T.forward_with_memory(net, batch)
        loss = L.occ_loss(out, batch)
        loss.backward()
        grads = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
        check("G19 train step from shards", bool(torch.isfinite(loss)) and grads > 0, f"loss={float(loss):.4f}")


def test_temporal() -> None:
    """G22: temporal memory: ego warp, masking, motion learnable only through memory, backbone once per tick."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        skip_check("G22", "torch not importable")
        return
    xf = occ.relative_xf((10.0, 5.0, 0.0, 1.0), (10.0, 3.0, 0.0, 1.0))
    p = xf[:, :2] @ np.array([0.0, 4.0]) + xf[:, 2]
    xf2 = occ.relative_xf((0.0, 0.0, 0.0, 1.0), (0.0, -2.0, 1.0, 0.0))  # past heading +x world, 2 m behind
    p2 = xf2[:, :2] @ np.array([1.0, 4.0]) + xf2[:, 2]  # world (1,4) -> past ego: right=(0,-1), fwd=(1,0)
    check("G22 relative_xf (translation, rotation)", np.allclose(p, [0.0, 6.0], atol=1e-5)
          and np.allclose(p2, [-6.0, 1.0], atol=1e-5), f"{p.tolist()} {p2.tolist()}")

    torch.manual_seed(0)
    net = occ.SparseOccNet([flat_pose("main")], GRID, width=8, mem_len=2)
    img = torch.rand(1, 1, 3, 48, 64)
    q = torch.tensor([[[0.0, 10.0, 0.0], [1.0, 12.0, 0.5]]])
    junk = torch.randn((1, 2) + tuple(net.encode(img).shape[1:]))
    with torch.no_grad():
        a = net(img, q)["occ"]
        b = net(img, q, junk, torch.randn(1, 2, 2, 3), torch.zeros(1, 2))["occ"]
        c = net(img, q, junk, torch.tensor([[[[1.0, 0, 0], [0, 1.0, 0]]] * 2]), torch.ones(1, 2))["occ"]
    check("G22 invalid memory is ignored, valid memory is used", torch.allclose(a, b) and not torch.allclose(a, c))

    # motion only visible across frames: same current image, different past -> different flow target
    def fit(use_mem: bool) -> float:
        torch.manual_seed(0)
        m = occ.SparseOccNet([flat_pose("main")], GRID, width=8, mem_len=1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        cur = torch.rand(1, 1, 3, 48, 64, generator=torch.Generator().manual_seed(3))
        qq = torch.tensor([[[x, y, 0.0] for x in (-1.0, 0.0, 1.0) for y in (8.0, 10.0, 12.0)]])
        eye = torch.tensor([[[[1.0, 0, 0], [0, 1.0, -1.0]]]])
        g = torch.Generator().manual_seed(7)
        past = [torch.rand(1, 1, 3, 48, 64, generator=g) for _ in range(2)]  # textured: backbone normalizes flat images
        cases = [(0, 5.0), (1, 0.0)]  # (past frame, forward flow m/s)
        for _ in range(250):
            for shade, v in cases:
                mf = occ.encode_memory(m, past[shade][:, None])  # the training path (train_occ.forward_with_memory)
                kw = (mf, eye, torch.ones(1, 1)) if use_mem else ()
                out = m(cur, qq, *kw)["flow"]
                loss = ((out - torch.tensor([0.0, v])) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        err = 0.0
        with torch.no_grad():
            for shade, v in cases:
                mf = m.encode(past[shade])[:, None]
                kw = (mf, eye, torch.ones(1, 1)) if use_mem else ()
                err = max(err, float((m(cur, qq, *kw)["flow"][..., 1] - v).abs().max()))
        return err

    e_mem, e_none = fit(True), fit(False)
    check("G22 flow learnable only with temporal memory", e_mem < 0.5 and e_none > 2.0,
          f"err with memory={e_mem:.3f} without={e_none:.3f}")

    calls = []
    hnd = net.backbone.register_forward_hook(lambda *_: calls.append(1))
    mem = occ.FeatureMemory(2)
    try:
        for k in range(3):
            occ.infer_sparse(net, torch.rand(1, 1, 3, 48, 64), tau=0.0, max_fine=64, memory=mem,
                             ego_pose=(0.0, 1.5 * k, 0.0, 1.0), ego_from_vehicle=True)
    finally:
        hnd.remove()
    check("G22 infer: backbone once per tick, memory holds mem_len frames", len(calls) == 3 and len(mem.frames) == 2,
          f"backbone calls={len(calls)} frames={len(mem.frames)}")


def test_scaling() -> None:
    """G23: ego-aligned memory on a static scene, per-camera age, reduced-cam mode, hardware profiles,
    checkpoint compatibility across profiles."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        skip_check("G23", "torch not importable")
        return

    # -- static scene, ego drives forward and turns: every memory lookup must land on the same world point
    def world_to_ego(pose, w):
        x, y, fx, fy = pose
        d = np.asarray(w, np.float64) - [x, y]
        return np.array([d @ [fy, -fx], d @ [fx, fy]])

    def yawpose(x, y, deg):
        r = math.radians(deg)
        return (x, y, math.cos(r), math.sin(r))

    torch.manual_seed(0)
    net = occ.SparseOccNet([flat_pose("main")], GRID, width=8, mem_len=2)
    poses = [yawpose(0.0, 0.0, 90.0), yawpose(0.1, 1.5, 88.0), yawpose(0.3, 3.0, 85.0)]  # oldest .. now
    world = np.array([[1.0, 12.0], [-2.0, 9.0], [0.5, 20.0], [3.0, 15.0]])
    cur = np.stack([world_to_ego(poses[2], w) for w in world])
    q = torch.tensor(np.c_[cur, np.full(len(cur), 0.4)][None], dtype=torch.float32)
    seen: list = []
    orig = net._pool
    net._pool = lambda feat, xyz, cam_w=None: (seen.append(xyz.detach().clone()), orig(feat, xyz, cam_w))[1]
    img = torch.rand(1, 1, 3, 48, 64)
    mem = occ.FeatureMemory(2)
    try:
        for k in range(2):
            occ.infer_sparse(net, img, tau=0.0, max_fine=16, memory=mem, ego_pose=poses[k], ego_from_vehicle=True)
        feat = net.encode(img)
        mf, mx, mv, _ = mem.tensors(poses[2], feat)
        seen.clear()
        with torch.no_grad():
            net.decode(feat, q, mf, mx, mv)
    finally:
        net._pool = orig
    # seen[0]=current, seen[1]=newest past (pose 1), seen[2]=pose 0
    errs = []
    for k, past in ((1, poses[1]), (2, poses[0])):
        want = np.stack([world_to_ego(past, w) for w in world])
        errs.append(float(np.abs(seen[k][0, :, :2].numpy() - want).max()))
    check("G23 static scene: memory lookups stay on the same world point while ego moves/turns",
          len(seen) == 3 and max(errs) < 1e-3, f"max err m per past frame={['%.5f' % e for e in errs]}")

    # -- per-camera age: a stale camera is down-weighted, a very stale one is masked
    two = occ.SparseOccNet([flat_pose("main"), flat_pose("wide")], GRID, width=4, mem_len=1)
    fm = torch.ones(1, 2, 4, 6, 8)
    fm[:, 1] = 3.0
    pt = torch.tensor([[[0.0, 10.0, 0.0]]])

    def pooled(age_wide):
        with torch.no_grad():
            cw = two.cam_weight_for(torch.ones(1, 2), torch.tensor([[0.0, age_wide]]))
            return float(two._pool(fm, pt, cw)[0][0, 0, 0])

    vals = [pooled(a) for a in (0.0, 0.6, 2.0)]
    expect_mid = (1.0 + 3.0 * math.exp(-0.6 / occ.CAM_AGE_TAU_S)) / (1.0 + math.exp(-0.6 / occ.CAM_AGE_TAU_S))
    check("G23 per-cam age: fresh=mean, stale down-weighted, >max age masked",
          abs(vals[0] - 2.0) < 1e-4 and abs(vals[1] - expect_mid) < 1e-4 and abs(vals[2] - 1.0) < 1e-4,
          f"pooled={['%.4f' % v for v in vals]} expect 2.0/{expect_mid:.4f}/1.0")
    with torch.no_grad():
        f2 = torch.rand(1, 2, 4, 6, 8)
        m2 = torch.rand(1, 1, 2, 4, 6, 8)
        eye = torch.tensor([[[[1.0, 0, 0], [0, 1.0, 0]]]])
        q2 = torch.tensor([[[0.0, 10.0, 0.0], [1.0, 14.0, 0.4]]])
        stale = two.decode(f2, q2, m2, eye, torch.ones(1, 1, 2), mem_age=torch.tensor([[[0.0, 5.0]]]))["occ"]
        dropped = two.decode(f2, q2, m2, eye, torch.tensor([[[1.0, 0.0]]]))["occ"]
    check("G23 per-cam age in memory: a stale memory cam equals that cam missing", torch.allclose(stale, dropped))

    # -- reduced-cam mode: masking cams on the 8-cam net == the 2/3-cam profile with the same weights
    full, _, _ = occ.build_profile("1080ti")
    ids = list(full.cam_ids)
    torch.manual_seed(1)
    imgs = torch.rand(1, len(ids), 3, 72, 128)
    qq = torch.tensor([[[x, y, 0.4] for x in (-4.0, 0.0, 4.0) for y in (6.0, 15.0, 30.0)]])
    diffs = {}
    for prof in ("1080ti_3cam", "1080ti_2cam"):
        red, _, cfg = occ.build_profile(prof)
        occ.load_compatible(red, full.state_dict())
        keep = [ids.index(c) for c in red.cam_ids]
        mask = torch.zeros(1, len(ids))
        mask[0, keep] = 1.0
        with torch.no_grad():
            a = full(imgs, qq, cam_valid=mask)["occ"]
            b = red(imgs[:, keep], qq)["occ"]
        diffs[prof] = float((a - b).abs().max())
    check("G23 reduced-cam: main+wide+narrow and main+wide run, == masked 8-cam net",
          all(d < 1e-5 for d in diffs.values()) and set(occ.PROFILES["1080ti_2cam"]["cams"]) == {"main", "wide"},
          str({k: f"{v:.2e}" for k, v in diffs.items()}))

    # -- every profile builds, runs a dummy forward at its own resolution, reports params + estimate
    rep = {}
    for name, cfg in occ.PROFILES.items():
        n, g, c = occ.build_profile(name)
        ls = int(c["long_side"])
        hw = (ls * 9 // 16 // 8 * 8, ls)
        mb = occ.memory_budget(n, n_cams=len(n.cam_ids), hw=hw, n_queries=int(np.prod(g.coarse_dims)) + int(c["max_fine"]), batch=1)
        half = 0.5 if c["precision"] == "fp16" else 1.0
        rep[name] = (sum(p.numel() for p in n.parameters()), len(n.cam_ids), hw,
                     round(mb["infer_bytes"] * half / 2**30, 2), round(mb["mem_cache_bytes"] / 2**20, 1),
                     round(mb["train_fp32_bytes"] / 2**30, 2))
    print("G23 profiles (params, cams, hw, infer GiB, memory-cache MiB, train fp32 GiB b=1; CPU-hook ESTIMATES):",
          rep)
    check("G23 every profile builds and forwards; 1080ti estimate under the 3 GB budget (4 GB free - 1 GB)",
          set(rep) >= {"1080ti", "high"} and rep["1080ti"][3] < 3.0, str(rep))

    # -- checkpoint compatibility: loadable knobs load, shape knobs refuse cleanly
    sd = full.state_dict()
    okl = []
    for kw in ({"mem_len": 4}, {"mem_len": 1}):
        m = occ.SparseOccNet(occ.rig_poses(), GRID, width=48, hidden=128, **kw)
        occ.load_compatible(m, sd)
        okl.append(kw)
    fine = occ.SparseOccNet(occ.rig_poses(["main", "wide"]), occ.OccGrid(voxel_m=0.2, y_max=76.8), width=48,
                            hidden=128)
    occ.load_compatible(fine, sd)
    okl.append("cams+voxel+range")
    refused = []
    for prof_kw in ({"width": 128, "hidden": 256, "depth": 3}, {"width": 48, "hidden": 128, "depth": 2}):
        try:
            occ.load_compatible(occ.SparseOccNet(occ.rig_poses(), GRID, **prof_kw), sd)
        except ValueError as e:
            refused.append("retrain" in str(e))
    hi, _, _ = occ.build_profile("high")
    try:
        occ.load_compatible(hi, sd)
        refused.append(False)
    except ValueError:
        refused.append(True)
    check("G23 ckpt: loads across mem_len/cams/voxel/range; width/depth refuse with a clear error",
          len(okl) == 3 and refused == [True, True, True], f"loaded={okl} refused={refused}")


def _render(pose, h: int = 48, w: int = 64, fov_v: float = 34.0, cam_h: float = 1.2) -> np.ndarray:
    """Tiny ray caster for a static world: textured ground plane + distant skyline. (3,h,w) in [0,1]."""
    x, y, fx, fy = pose
    f = (h / 2) / math.tan(math.radians(fov_v) / 2)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = (np.arange(h) + 0.5 - h / 2) / f
    uu, vv = np.meshgrid(u, v)
    rx, ry = fy, -fx  # ego right in world
    dwx = fx + uu * rx
    dwy = fy + uu * ry
    img = np.zeros((3, h, w), np.float32)
    down = vv > 1e-3
    t = np.where(down, cam_h / np.maximum(vv, 1e-3), 0.0)
    gx, gy = x + t * dwx, y + t * dwy
    for c in range(3):
        g = (0.5 + 0.18 * np.sin(1.7 * gx + 0.9 * c) * np.cos(1.3 * gy + 0.4 * c)
             + 0.12 * np.sin(0.37 * gx * gy * 0.1 + 2.1 * gy + c) + 0.1 * np.sin(4.1 * gx - 3.3 * gy + c))
        az = np.arctan2(dwy, dwx)
        sky = 0.6 + 0.2 * np.sin(7.0 * az + c) * (vv < -0.05) + 0.1 * np.sin(23.0 * az)
        img[c] = np.where(down, g, sky)
    return np.clip(img, 0.0, 1.0)


def test_vo() -> None:
    """G24: runtime ego motion is camera-only (VO head); GT/vehicle pose only behind ego_from_vehicle;
    static scene stays put in world space with the VO-estimated pose, drift reported in metres."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        skip_check("G24", "torch not importable")
        return
    past = (3.0, -2.0, math.cos(1.2), math.sin(1.2))
    d = (0.3, 1.4, 0.07)
    cur = occ.pose_step(past, d)
    back = occ.delta_from_xf(occ.relative_xf(cur, past))
    check("G24 pose_step / relative_xf / delta_from_xf round trip", np.allclose(back, d, atol=1e-5), str(back))

    src = (ROOT / "python" / "train" / "occ_sparse.py").read_text(encoding="utf-8")
    bad = [w for w in ("vehicle_labels", "_ego_pose", "scenetree", "beamngpy", "state.jsonl") if w in src]
    torch.manual_seed(0)
    net = occ.SparseOccNet([flat_pose("main")], GRID, width=16, hidden=64, mem_len=2)
    mem = occ.FeatureMemory(2)
    img = torch.rand(1, 1, 3, 48, 64)
    refused = False
    try:
        occ.infer_sparse(net, img, tau=0.0, max_fine=16, memory=mem, ego_pose=(0.0, 0.0, 0.0, 1.0))
    except ValueError:
        refused = True
    check("G24 runtime: no GT/pose source in net module; external pose refused with flag off",
          not bad and refused, f"names={bad} refused={refused}")

    # train VO (and backbone) on random pose pairs of the synthetic static world
    rng = np.random.default_rng(0)

    def pair():
        p0 = (rng.uniform(-50, 50), rng.uniform(-50, 50), *(lambda a: (math.cos(a), math.sin(a)))(rng.uniform(0, 6.3)))
        dd = (rng.uniform(-0.2, 0.2), rng.uniform(0.0, 2.0), rng.uniform(-0.06, 0.06))
        return _render(p0), _render(occ.pose_step(p0, dd)), dd

    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    scale = torch.tensor([1.0, 1.0, 10.0])
    for step in range(500):
        batch = [pair() for _ in range(12)]
        a = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
        c = torch.from_numpy(np.stack([b[1] for b in batch]))[:, None]
        tgt = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        loss = torch.nn.functional.smooth_l1_loss(net.vo(net.vo_encode(c), net.vo_encode(a)) * scale, tgt * scale)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # held-out clip: 30 ticks, the ego drives ~40 m with gentle turns; runtime path, flag off
    true = [(10.0, 7.0, math.cos(1.0), math.sin(1.0))]
    for k in range(29):
        true.append(occ.pose_step(true[-1], (0.0, 1.0 + 0.5 * math.sin(k / 4), 0.03 * math.cos(k / 5))))
    mem = occ.FeatureMemory(2)
    est = []
    net.eval()
    with torch.no_grad():
        for p in true:
            occ.infer_sparse(net, torch.from_numpy(_render(p))[None, None], tau=0.0, max_fine=16, memory=mem)
            est.append(mem.odom)

    def to_start(poses):  # express each pose relative to the first, as (x, y)
        o = poses[0]
        return np.array([[(q[0] - o[0]) * o[3] - (q[1] - o[1]) * o[2], (q[0] - o[0]) * o[2] + (q[1] - o[1]) * o[3]]
                         for q in poses])

    te, ee = to_start(true), to_start(est)
    path = float(np.sum(np.linalg.norm(np.diff(te, axis=0), axis=1)))
    drift = float(np.linalg.norm(te[-1] - ee[-1]))
    still = float(np.linalg.norm(te[-1]))  # error if we assumed the ego never moved
    world = np.array([[2.0, 12.0], [-3.0, 20.0], [1.0, 8.0]])
    werr = []
    for k in range(2, len(true)):
        for j in (1, 2):
            xt = occ.relative_xf(true[k], true[k - j])
            xe = occ.relative_xf(est[k], est[k - j])
            werr.append(float(np.abs((world @ xt[:, :2].T + xt[:, 2]) - (world @ xe[:, :2].T + xe[:, 2])).max()))
    print(f"G24 VO clip: path {path:.1f} m, end drift {drift:.2f} m ({100 * drift / path:.1f}%), "
          f"stationary-assumption err {still:.1f} m, memory world-point err max {max(werr):.2f} m "
          f"mean {np.mean(werr):.3f} m")
    check("G24 static scene with VO pose (flag off): drift <10% of path and memory lookups stay within 0.5 m",
          drift < 0.10 * path and max(werr) < 0.5,
          f"drift={drift:.2f} m / {path:.1f} m, world err max={max(werr):.2f} m")


def test_sparsity_scaling() -> None:
    """G25: fully sparse end to end. Every tensor produced by any op in infer_sparse (backbone, coarse
    pass, fine pass, heads) is counted with a dispatch hook. With max_coarse/max_fine fixed, the total
    stays flat while the grid range/resolution grows, and no tensor is anywhere near grid-sized."""
    try:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:  # noqa: BLE001
        skip_check("G25", "torch dispatch mode not importable")
        return

    class Count(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.total = 0
            self.biggest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor):
                    self.total += t.numel()
                    self.biggest = max(self.biggest, t.numel())
            return out

    grids = [occ.OccGrid(), occ.OccGrid(y_max=102.4, x_min=-51.2, x_max=51.2), occ.OccGrid(voxel_m=0.2),
             occ.OccGrid(voxel_m=0.1, y_max=102.4, x_min=-51.2, x_max=51.2)]
    img = torch.rand(1, 1, 3, 48, 64)
    rows = []
    for g in grids:
        torch.manual_seed(0)
        net = occ.SparseOccNet([flat_pose("main")], g, width=16, mem_len=2)
        mem = occ.FeatureMemory(2)
        occ.infer_sparse(net, img, tau=0.0, max_coarse=4096, max_fine=4096, memory=mem)  # fill memory
        with Count() as c:
            res = occ.infer_sparse(net, img, tau=0.0, max_coarse=4096, max_fine=4096, memory=mem)
        rows.append((int(np.prod(g.dims)), int(np.prod(g.coarse_dims)), res["evaluated"], c.total, c.biggest))
    print("G25 (full cells, full coarse cells, queried, all-op tensor elements, largest tensor) "
          "tau=0 max_coarse=max_fine=4096, temporal on:", rows)
    totals = {r[3] for r in rows}
    check("G25 every tensor in infer_sparse (coarse pass included) flat while the grid grows 1x..%dx"
          % (rows[-1][0] // rows[0][0]), len(totals) == 1 and all(r[2] == 8192 for r in rows), str(rows))
    check("G25 no tensor scales with the grid: largest is the same on every grid (query x hidden, set by "
          "the caps) and below the big grid's coarse lattice",
          len({r[4] for r in rows}) == 1 and rows[-1][4] < rows[-1][1], str([r[4] for r in rows]))

    torch.manual_seed(0)
    net = occ.SparseOccNet([flat_pose("main")], GRID, width=16)
    res = occ.infer_sparse(net, torch.rand(1, 1, 3, 48, 64), tau=0.0)
    full = int(np.prod(GRID.dims))
    stored = len(res["idx"])
    print(f"G25 1080ti grid: full {full} cells, queried {res['evaluated']} ({100 * res['evaluated'] / full:.1f}%), "
          f"stored {stored} ({100 * stored / full:.2f}%); worst case (tau=0), "
          f"caps max_coarse={occ.DEFAULT_MAX_COARSE} max_fine={occ.DEFAULT_MAX_FINE}")
    check("G25 queried <= max_coarse + max_fine and stored <= queried at the 1080ti profile",
          res["evaluated"] <= occ.DEFAULT_MAX_COARSE + occ.DEFAULT_MAX_FINE and stored <= res["evaluated"],
          f"{res['evaluated']}/{stored}/{full}")
    import json
    import tracemalloc

    from python.perception import occ_out as oo
    ob, npk = [], []
    for g in grids:
        torch.manual_seed(0)
        net = occ.SparseOccNet([flat_pose("main")], g, width=16, mem_len=2)
        r = occ.infer_sparse(net, img, tau=0.0, max_coarse=4096, max_fine=4096)
        ob.append(len(json.dumps(oo.OccOutBuilder(net).build(r, t=0.0).to_dict())))
        tracemalloc.start()
        ci = occ.nearest_coarse_queries(g, 4096)
        occ.coarse_children(g, ci[:4096 // int(np.prod(g.coarse))])
        npk.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    print(f"G25 occ_out json bytes per grid {ob}; numpy index peak bytes per grid {npk}")
    check(f"G25 occ_out size flat (<=10 pct) while the grid grows {rows[-1][0] // rows[0][0]}x",
          max(ob) <= 1.1 * min(ob), str(ob))
    check("G25 numpy index arrays (coarse selection + children) flat (<=2x) while the grid grows",
          max(npk[1:]) <= 2 * min(npk[1:]) and max(npk) < 8 * 2**20, str(npk))
    far = occ.nearest_coarse_queries(occ.OccGrid(voxel_m=0.1, y_max=204.8, x_min=-102.4, x_max=102.4), 4096)
    check("G25 coarse selection is nearest-first and capped on a 230M-cell grid", len(far) == 4096, str(len(far)))


def test_occ_out() -> None:
    """G26: occ_out v1 per-tick record (all six viz tabs) from net outputs only; schema, caps, content."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        skip_check("G26", "torch not importable")
        return
    import json

    from python.perception import occ_out as oo

    torch.manual_seed(0)
    net = occ.SparseOccNet([flat_pose("main"), flat_pose("wide")], GRID, width=16, mem_len=2)
    b = oo.OccOutBuilder(net)
    mem = occ.FeatureMemory(2)
    errs, sizes, out = [], [], None
    for k in range(3):
        res = occ.infer_sparse(net, torch.rand(1, 2, 3, 48, 64), tau=0.0, memory=mem, max_fine=2048)
        out = b.build(res, t=0.1 * k).to_dict()
        errs += oo.validate_occ_out(out)
        sizes.append(len(json.dumps(out)))
    stages = set(out["debug"]["stage_ms"])
    need = {"backbone", "vo", "memory", "query_coarse", "query_fine", "head_occ", "head_sem", "head_flow", "head_box"}
    print(f"G26 live net 3 ticks: json bytes {sizes}, stages {sorted(stages)}, counts {out['debug']['counts']}, "
          f"cams {[c['cam_id'] for c in out['cams']]} pts {[len(c['depth']) for c in out['cams']]}, "
          f"mem_pts {len(out['ego']['memory_pts'])}, vram {out['debug']['peak_vram_mb']}")
    check("G26 net -> occ_out v1 validates every tick, per-stage ms for every stage/head, VO delta present",
          not errs and need <= stages and out["ego"]["delta"] is not None, str(errs[:5]) + str(need - stages))

    # synthetic head outputs with known content (no net): lanes, edges, objects, tracks, memory warp
    v = GRID.voxel_m
    cells, cls, boxv = [], [], []

    def add(xy, c, bx=None):
        for p in xy:
            cells.append([p[0], p[1], 0.0 if c != 1 else 0.6])
            cls.append(c)
            boxv.append(bx(p) if bx else [0, 0, 0, 0, 0, 1])
    ys = np.arange(4.0, 40.0, v)
    add([(1.8, y) for y in ys if (y - 4.0) % 12.0 < 3.0], 2)                     # white dashed: 3 m on, 9 m off
    add([(-1.8, y) for y in ys], 3)                                                # yellow solid
    add([(x, y) for x in np.arange(-3.6, 3.7, v) for y in np.arange(4.0, 20.0, v)], 0)  # drivable patch
    for cx, cy in ((6.0, 12.0), (-6.0, 25.0)):                                    # two cars 4.4 x 1.8, heading +y
        pts = [(cx + dx, cy + dy) for dx in np.arange(-0.8, 0.85, v) for dy in np.arange(-2.0, 2.05, v)]
        add(pts, 1, lambda p, cx=cx, cy=cy: [cx - p[0], cy - p[1], math.log(4.4), math.log(1.8), 1.0, 0.0])
    cen = np.array(cells)
    idx, ok = GRID.to_index(cen)
    idx = idx[ok]
    cls_a = np.array(cls)[ok]
    boxa = np.array(boxv, np.float32)[ok]
    # snap the box offsets to the snapped cell centres
    snapped = GRID.centers(idx)
    boxa[:, :2] += (cen[ok] - snapped)[:, :2]
    sem = np.full((len(idx), len(occ.OCC_CLASSES)), -6.0, np.float32)
    sem[np.arange(len(idx)), cls_a] = 6.0

    def fake(pose):
        return {"idx": torch.from_numpy(idx), "prob": torch.full((len(idx),), 0.9), "sem": torch.from_numpy(sem),
                "flow": torch.zeros(len(idx), 2), "box": torch.from_numpy(boxa), "evaluated": 15872,
                "ego_delta": [0.0, 2.0, 0.0], "ego_pose": pose, "stage_ms": {"backbone": 1.0}}
    b2 = oo.OccOutBuilder(net)
    ts = [0.0, 0.1, 0.25, 0.32]  # irregular tick times: dt must come from them
    outs = [b2.build(fake((0.0, 2.0 * k, 0.0, 1.0)), t=ts[k]).to_dict() for k in range(4)]
    o = outs[-1]
    types = sorted(ln["type"] for ln in o["lanes"])
    objs = sorted(((round(x["xyz"][0], 1), round(x["xyz"][1], 1), round(x["lwh"][0], 1), round(x["lwh"][1], 1),
                    x["track_id"]) for x in o["objects"]))
    ids0 = sorted(x["track_id"] for x in outs[0]["objects"])
    print(f"G26 synthetic: lane types {types}, objects (x, y, l, w, id) {objs}, first-tick ids {ids0}")
    check("G26 lanes: white dashed chained, yellow solid, two road edges, per-point conf",
          types.count("white_dashed") == 1 and types.count("yellow_solid") == 1 and types.count("road_edge") == 2
          and all(len(ln["conf"]) == len(ln["pts"]) for ln in o["lanes"]), str(types))
    check("G26 objects: box-head votes give each car's centre and 4.4 x 1.8, ids stable over ticks",
          len(objs) == 2 and all(abs(a - e) < 0.25 for r, ex in zip(objs, ((-6.0, 25.0), (6.0, 12.0))) for a, e in zip(r[:2], ex))
          and all(abs(r[2] - 4.4) < 0.1 and abs(r[3] - 1.8) < 0.1 for r in objs) and sorted(r[4] for r in objs) == ids0,
          str(objs))
    # memory points: ego drove 2 m/tick forward, scene fixed in ego coords means it moved with ego -> earlier
    # cells re-expressed now must sit 2 m (newest) behind where they were
    mp = np.array(o["ego"]["memory_pts"])
    cur_top = np.array(o["occ"]["xyz"])[: len(mp) // 2]
    shift = float(np.median(mp[: len(cur_top), 1] - cur_top[:, 1])) if len(cur_top) else float("nan")
    check("G26 memory points re-expressed through the ego pose (newest tick 2 m behind)",
          len(mp) > 0 and abs(shift + 2.0) < 1e-3, f"n={len(mp)} shift={shift}")
    check("G26 synthetic record validates", not oo.validate_occ_out(o), str(oo.validate_occ_out(o)))

    e = o["ego"]
    check("G26 ego speed/yaw rate = |(dx,dy)|/dt, dyaw/dt with dt from tick times; source vo",
          e["source"] == "vo" and abs(e["dt"] - 0.07) < 1e-6 and abs(e["speed_mps"] - 2.0 / 0.07) < 0.01
          and abs(e["yaw_rate"]) < 1e-9, f"{e['dt']} {e['speed_mps']} {e['yaw_rate']} {e['source']}")
    poly = np.array(e["corridor"])

    def self_x(pg):
        segs = [(pg[i], pg[(i + 1) % len(pg)]) for i in range(len(pg))]

        def cr(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        for i in range(len(segs)):
            for j in range(i + 2, len(segs)):
                if i == 0 and j == len(segs) - 1:
                    continue
                (p1, p2), (p3, p4) = segs[i], segs[j]
                if cr(p1, p2, p3) * cr(p1, p2, p4) < 0 and cr(p3, p4, p1) * cr(p3, p4, p2) < 0:
                    return True
        return False
    check("G26 corridor: closed ring >=3 vertices, capped, not self-intersecting, spans the drivable patch",
          3 <= len(poly) <= oo.OCC_OUT_CAPS["corridor"] and not self_x(poly) and poly[:, 0].min() < -3.5
          and poly[:, 0].max() > 3.5, f"n={len(poly)}")
    ids3 = {x["track_id"] for x in o["objects"]}
    b2d = [bx for cm in o["cams"] for bx in cm["boxes"]]
    check("G26 2D boxes are projections of the 3D boxes: same track ids, clipped to [-1,1]",
          len(b2d) >= 1 and {bx["track_id"] for bx in b2d} <= ids3
          and all(-1 <= v_ <= 1 for bx in b2d for v_ in bx["uv_min"] + bx["uv_max"]), str(b2d[:2]))
    b3 = oo.OccOutBuilder(net)
    om = b3.build(fake((0.0, 0.0, 0.0, 1.0)), t=0.0, cam_valid=[1.0, 0.0]).to_dict()
    check("G26 masked cam is absent (not zeros); reduced rig still validates",
          [c["cam_id"] for c in om["cams"]] == ["main"] and not oo.validate_occ_out(om))
    rv = dict(fake((0.0, 0.0, 0.0, 1.0)), ego_source="vehicle")
    check("G26 ego source says vehicle only when the runtime used --ego-from-vehicle",
          oo.OccOutBuilder(net).build(rv, t=0.0).to_dict()["ego"]["source"] == "vehicle")

    import copy

    def mut(f):
        x = copy.deepcopy(o)
        f(x)
        return oo.validate_occ_out(x)
    cases = {
        "unknown top key": lambda x: x.__setitem__("extra", 1),
        "unknown nested key": lambda x: x["ego"].__setitem__("speed_kph", 1.0),
        "unknown object key": lambda x: x["objects"][0].__setitem__("heading", 0.0),
        "missing field": lambda x: x["occ"].pop("flow"),
        "GT key": lambda x: x["debug"].__setitem__("gt_pose", [0, 0]),
        "NaN": lambda x: x["occ"]["xyz"][0].__setitem__(0, float("nan")),
        "inf": lambda x: x["ego"].__setitem__("speed_mps", float("inf")),
        "prob > 1": lambda x: x["occ"]["prob"].__setitem__(0, 1.5),
        "conf < 0": lambda x: x["lanes"][0]["conf"].__setitem__(0, -0.1),
        "unknown class": lambda x: x["occ"]["sem"].__setitem__(0, 99),
        "unknown lane type": lambda x: x["lanes"][0].__setitem__("type", "purple_dotted"),
        "wrong shape": lambda x: x["occ"]["idx"].__setitem__(0, [1, 2]),
        "count mismatch": lambda x: x["occ"]["counts"].__setitem__("stored", 1),
        "yaw not unit": lambda x: x["objects"][0].__setitem__("yaw_sin", 0.5),
        "over cap": lambda x: x.__setitem__("lanes", x["lanes"] * 9),
        "1-point polyline": lambda x: x["lanes"][0].update(pts=x["lanes"][0]["pts"][:1], conf=x["lanes"][0]["conf"][:1]),
        "vram as string": lambda x: x["debug"].__setitem__("peak_vram_mb", "UNPROVEN"),
        "bad source": lambda x: x["ego"].__setitem__("source", "gt"),
    }
    missed = [k for k, f in cases.items() if not mut(f)]
    check(f"G26 validator rejects all {len(cases)} contract violations", not missed, str(missed))
    from python.perception import occ_viz as ov
    vm = oo.occ_out_to_viz(o)
    check("G26 occ_out feeds gvd_scene v1 with v1 keys only (validates; lanes/edges/tracks carried; v2 delta listed)",
          not ov.validate_v1(vm) and set(vm) <= ov.V1_KEYS and len(vm["tracks"]) == len(o["objects"])
          and len(vm["lanes"]) == 2 and len(vm["edges"]) == 2 and len(oo.VIZ_V2_DELTA) > 0,
          f"v1 errs={ov.validate_v1(vm)}")
    check("G26 VRAM: peak_vram_mb None on CPU (UNPROVEN until IQU45 nvidia-smi); estimate in its own field",
          o["debug"]["peak_vram_mb"] is None and "cpu_mem_est_mb" in o["debug"]
          and oo.OccOutBuilder(net).build(fake(None), t=0.0, cpu_mem_est_mb=160.0).to_dict()["debug"]["peak_vram_mb"] is None)
    src = Path(oo.__file__).read_text(encoding="utf-8")
    check("G26 occ_out reads no GT/recording source and no images",
          not any(w in src for w in ("vehicle_labels", "occ_labels", "state.jsonl", "beamngpy", "cv2.imread")) and
          "images" not in src, "")


def test_viz_no_images() -> None:
    """G16b: the viz path consumes only net outputs: no torch/cv2/net module, no image argument."""
    import inspect
    import subprocess

    code = ("import sys; sys.path.insert(0, %r); import python.perception.occ_viz as V; "
            "print(sorted(m for m in ('torch', 'cv2', 'PIL', 'python.train.occ_sparse', 'python.train.train_occ') "
            "if m in sys.modules))" % str(ROOT))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    check("G16b viz imports no torch/cv2/net", res.returncode == 0 and res.stdout.strip() == "[]",
          (res.stdout + res.stderr).strip()[-200:])
    from python.perception import occ_viz as V

    params = [n for f in (V.occ_to_viz, V.viz_state_from_cells, V.viz_state_from_infer, V.OccTracker.update)
              for n in inspect.signature(f).parameters]
    src = Path(V.__file__).read_text(encoding="utf-8").lower()
    check("G16b no image inputs in the viz API", not any("im" in n for n in params)
          and "image" not in src and "imread" not in src and "pixel" not in src, str(params))

    class _Poison:
        """Stands in for camera frames: any access raises."""
        def __getattr__(self, k):
            raise AssertionError(f"viz touched images ({k})")

        def __array__(self, *a, **kw):
            raise AssertionError("viz touched images (array)")

        def __getitem__(self, k):
            raise AssertionError("viz touched images (index)")

        def __len__(self):
            raise AssertionError("viz touched images (len)")

    idx = np.array([[64, 56, 4], [65, 56, 4], [64, 57, 4], [65, 57, 4]], np.int64)
    sem = np.zeros((4, 7), np.float32)
    sem[:, 3] = 1.0
    out = {"idx": idx, "sem": sem, "occ": np.ones(4, np.float32),
           "images": _Poison(), "frames": _Poison(), "feat": _Poison()}
    try:
        msg = V.occ_to_viz(V.viz_state_from_infer(out, GRID), t=0.0)
        ok, err = V.validate_v1(msg) == [], ""
    except AssertionError as e:  # pragma: no cover - the failure we test for
        ok, err = False, str(e)
    check("G16b viz runs with poisoned image/feature keys present", ok, err)


def test_paint_labels() -> None:
    """G21: palette names map to classes only by exact list; thin paint wins its voxel at 25%."""
    from python.train import occ_sparse as O

    names = {"ASPHALT": O.CLS_DRIVABLE, "line_yellow": O.CLS_LANE_YELLOW, "m_line_white": O.CLS_LANE_WHITE,
             "line_dashed_short_yellow": O.CLS_LANE_YELLOW, "line_reflectors": O.CLS_IGNORE,
             "crosswalk": O.CLS_IGNORE, "curb": O.CLS_IGNORE, "grass": O.CLS_IGNORE}
    got = {n: L.class_for_name(n) for n in names}
    check("G21 palette name map", got == names, str(got))
    lin = np.array([5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])
    cls = np.array([0, 0, 0, O.CLS_LANE_WHITE] + [0] * 7 + [O.CLS_LANE_YELLOW], np.uint8)
    vox, out = L._vote(lin, cls)
    check("G21 paint share vote", list(vox) == [5, 9] and list(out) == [O.CLS_LANE_WHITE, O.CLS_DRIVABLE], str(out))


def _synthetic_scene_cells():
    from python.perception.occ_viz import CLASS_NAMES

    v = 0.4
    cells, cls, flow = [], [], []

    def add(xs, ys, z, name, f=(np.nan, np.nan)):
        for x in xs:
            for y in ys:
                cells.append([x, y, z])
                cls.append(CLASS_NAMES.index(name))
                flow.append(list(f))

    ys = np.arange(2.2, 30.0, v)
    add(np.arange(-3.4, 3.5, v), ys, -1.4, "drivable")
    add([-1.8], ys, -1.0, "lane_white")
    add([1.8], ys, -1.0, "lane_yellow")
    add(np.arange(-3.0, 0.0, v), [24.6], -1.0, "stop_line")
    add(np.arange(-3.0, 3.1, v), np.arange(27.0, 29.0, v), -0.6, "crosswalk")
    veh = {"same_way": ((0.0, 15.0), (0.0, 10.0)), "oncoming": ((3.6, 20.0), (0.0, -8.0)),
           "crossing": ((-9.0, 12.0), (6.0, 0.0)), "alongside": ((3.4, 1.0), (0.0, 5.0)),
           "parked": ((7.0, 8.0), (0.0, 0.0))}
    for (cx, cy), f in veh.values():
        c = GRID.centers(GRID.to_index(np.array([[cx, cy, 0.6]]))[0])[0]  # snap the centre, then whole cells
        k = np.arange(-2, 3) * v
        add(c[0] + k, c[1] + k, 0.6, "vehicle", f)
    idx, inside = GRID.to_index(np.array(cells))
    assert inside.all()
    return GRID.centers(idx), np.array(cls), np.array(flow), veh  # snapped to real cell centres


def test_viz() -> None:
    """G20: synthetic occupancy -> positions within tolerance -> v1 message; motion only from tracks over time."""
    import json

    from python.perception import occ_viz as V

    cells, cls, flow, veh = _synthetic_scene_cells()
    st = V.viz_state_from_cells(cells, cls)
    paints = sorted(ln["paint"] for ln in st["lanes_ext"])
    check("G20 paint white+yellow", paints == ["white", "yellow"], str(paints))
    for paint, x0 in (("white", -1.8), ("yellow", 1.8)):
        ln = [r for r in st["lanes_ext"] if r["paint"] == paint]
        pts = np.array(ln[0]["pts"]) if ln else np.zeros((0, 2))
        check(f"G20 {paint} line within 0.2 m of x={x0}, y 2.2..29.8",
              len(pts) > 0 and bool(np.all(np.abs(pts[:, 0] - x0) <= 0.2)) and pts[0, 1] <= 2.6
              and pts[-1, 1] >= 29.4 and bool(np.all(np.diff(pts[:, 1]) > 0)), f"n={len(pts)}")
    occ_edges = [e for e in st["road_edges"] if e["kind"] == "occ"]
    xs = sorted(round(float(np.mean([p[0] for p in e["pts"]])), 2) for e in occ_edges)
    check("G20 road edges within 0.05 m of x=+-3.6", len(xs) == 2 and abs(xs[0] + 3.6) <= 0.05
          and abs(xs[1] - 3.6) <= 0.05, str(xs))
    sl = np.array(st["stop_lines"][0]["pts"]) if st["stop_lines"] else np.zeros((0, 2))
    check("G20 stop line at y=24.6 x -3..-0.2", len(st["stop_lines"]) == 1 and bool(np.all(np.abs(sl[:, 1] - 24.6) <= 0.2))
          and sl[:, 0].min() <= -2.8 and sl[:, 0].max() >= -0.4)
    cw = np.array(st["crosswalks"][0]["pts"]) if st["crosswalks"] else np.zeros((0, 2))
    check("G20 crosswalk rectangle (cells x -3..3, y 27..28.6, +-1 voxel)", len(st["crosswalks"]) == 1 and len(cw) == 5
          and abs(cw[:, 0].min() + 3.0) <= 0.4 and abs(cw[:, 0].max() - 3.0) <= 0.4
          and abs(cw[:, 1].min() - 27.0) <= 0.4 and abs(cw[:, 1].max() - 28.6) <= 0.4, str(cw[:4].tolist()))
    dets = {(round(t["x"]), round(t["y"])) for t in st["tracks"]}
    check("G20 vehicle centroids", dets == {(round(c[0]), round(c[1])) for (c, _) in veh.values()}, str(dets))

    # motion over frames: move each synthetic car by its velocity while the ego drives 5 m/s forward
    ego_v, dt = 5.0, 0.1
    tracker = V.OccTracker()
    vcls = V.CLASS_NAMES.index("vehicle")
    vmask = cls == vcls
    msgs = []
    n_frames = 8
    for k in range(n_frames):
        t = k * dt
        moved = cells.copy()
        for (c0, v), name in zip(veh.values(), veh):
            sel = vmask & (np.hypot(cells[:, 0] - c0[0], cells[:, 1] - c0[1]) < 1.5)
            off = np.array([v[0] * t, v[1] * t - ego_v * t])  # ego-frame: the ego moved forward
            moved[sel, :2] += np.round(off / 0.4) * 0.4  # whole voxels, like real output (+-0.2 m quantisation)
        msgs.append(V.occ_to_viz(V.viz_state_from_cells(moved, cls), t=t, seq=k, ego_speed_mps=ego_v,
                                 tracker=tracker))
    check("G20 no motion from a single frame", all("motion" not in tr and "v" not in tr for tr in msgs[0]["tracks"]))
    last = msgs[-1]
    ids = {tr["id"] for tr in last["tracks"]}
    check("G20 ids stable across frames", ids == {tr["id"] for tr in msgs[0]["tracks"]} and len(ids) == len(veh))
    got = {}
    for tr in last["tracks"]:
        k0 = min(veh, key=lambda m: math.hypot(tr["x"] - (veh[m][0][0] + veh[m][1][0] * t),
                                               tr["y"] - (veh[m][0][1] + (veh[m][1][1] - ego_v) * t)))
        got[k0] = tr.get("motion")
    check("G20 motion states from tracks", got == {m: m for m in veh}, str(got))
    sw = [tr for tr in last["tracks"] if tr.get("motion") == "same_way"]
    check("G20 speed/heading from track (ego-compensated)", bool(sw) and abs(sw[0]["v"] - 10.0) < 0.3
          and abs(sw[0]["yaw"] - math.pi / 2) < 0.05, str(sw[:1]))
    pk = [tr for tr in last["tracks"] if tr.get("motion") == "parked"]
    check("G20 parked car still while ego drives", bool(pk) and pk[0]["v"] < 0.3)

    # Skeptic's case: a stationary car facing forward, ego at 10 m/s, must be parked, not same_way
    trk2 = V.OccTracker()
    vid = V.CLASS_NAMES.index("vehicle")
    pk_states = []
    for k in range(6):
        t = k * 0.1
        y0 = 20.2 - np.round(10.0 * t / 0.4) * 0.4  # ego closes in at 10 m/s, whole voxels
        xs, ys = np.arange(-2, 3) * 0.4 + 0.2, np.arange(-5, 6) * 0.4 + y0  # 2.0 m wide, 4.4 m long along y
        pc = np.array([[x, y, 0.6] for x in xs for y in ys])
        fr = V.viz_state_from_cells(pc, np.full(len(pc), vid))
        pk_states.append(V.occ_to_viz(fr, t=t, seq=k, ego_speed_mps=10.0, tracker=trk2)["tracks"])
    one = pk_states[-1][0] if len(pk_states[-1]) == 1 else {}
    check("G20 forward-facing stationary car -> parked", one.get("motion") == "parked" and one.get("v", 9) < 0.5
          and abs(abs(one.get("yaw", 0)) - math.pi / 2) < 0.1, str(one))

    errs = V.validate_v1(last)
    check("G20 v1 required fields", not errs, "; ".join(errs[:3]))
    back = json.loads(json.dumps(last, allow_nan=False))
    check("G20 JSON round trip", back == last and not V.validate_v1(back))
    extra = dict(back, future_field={"x": 1})
    check("G20 unknown fields ignored", not V.validate_v1(extra))
    bad = dict(back, v=2)
    check("G20 validator rejects v!=1", bool(V.validate_v1(bad)))

    reds = [(k, r) for k in ("lanes", "edges", "tracks") for r in back[k] if V.color_of(k, r) == "red"]
    reds += [(k, r) for k in ("crosswalks", "stop_lines") for r in back["occ"][k] if V.color_of(k, r) == "red"]
    check("G20 red only on edges/curbs", bool(back["edges"]) and all(k == "edges" for k, _ in reds)
          and all(V.color_of("edges", e) == "red" for e in back["edges"]))
    check("G20 lane colours = paint", all(V.color_of("lanes", r) == r["paint"] for r in back["lanes"]))
    try:
        from python.viz import cpp_publisher as P
    except Exception:  # noqa: BLE001 - lands with PR #232
        skip_check("G20 cpp_publisher v1", "python/viz/cpp_publisher.py not on this branch (PR #232)")
        return
    state = {"lanes_ext": back["lanes"], "road_edges": back["edges"],
             "tracks": [dict(tr, **{"class": tr["cls"], "speed_mps": tr.get("v")}) for tr in back["tracks"]]}
    msg = P.build_scene_message(state)
    check("G20 cpp_publisher v1 message", P.encode_message(msg) is not None
          and sorted(x["paint"] for x in msg["lanes"]) == ["white", "yellow"] and len(msg["tracks"]) == len(veh))


def main() -> int:
    test_labels()
    test_paint_labels()
    test_viz()
    test_viz_no_images()
    test_shards()
    test_net()
    test_runtime_isolation()
    test_temporal()
    test_scaling()
    test_vo()
    test_sparsity_scaling()
    test_occ_out()
    test_pipeline()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("test_occ_sparse: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
