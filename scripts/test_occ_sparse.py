#!/usr/bin/env python3
"""Sparse occupancy (training-only): labels from depth+sem of every train_gt
camera plus traffic rows, WebDataset tar shards, recovery mix, and an opt-in
query head that is not an ONNX output.

Offline. Synthetic depth planes and rows only. No BeamNG, no GPU.
Torch checks skip when PyTorch is not installed.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402


def _cam(cid: str, yaw: float, pos=(0.0, 1.2, 1.26), pitch: float = 0.0, res=(64, 48), fov_v: float = 40.0):
    return {"id": cid, "pos_m": list(pos), "yaw_deg": yaw, "pitch_deg": pitch, "fov_v": fov_v,
            "resolution": list(res), "far_m": 300.0}


def _plane(depth_m: float, res=(64, 48)) -> np.ndarray:
    return np.full((res[1], res[0]), int(round(depth_m * 100)), dtype=np.uint16)


def check_unproject_forward_and_side() -> None:
    from python.train.occ_sparse import unproject_depth

    pts, keep = unproject_depth(_plane(10.0), _cam("main", 0.0), stride=1)
    assert pts.shape[1] == 3 and keep.all()
    # view-Z 10 m straight ahead from y=1.2: every point sits on y = 11.2.
    assert np.allclose(pts[:, 1], 11.2, atol=1e-3), (pts[:, 1].min(), pts[:, 1].max())
    assert abs(float(np.median(pts[:, 0]))) < 0.2 and abs(float(np.median(pts[:, 2])) - 1.26) < 0.2
    side, _ = unproject_depth(_plane(10.0), _cam("pillarR", 90.0, pos=(0.0, 0.0, 1.0)), stride=1)
    assert np.allclose(side[:, 0], 10.0, atol=1e-3), "yaw 90 looks right (+x)"
    depth = _plane(10.0)
    depth[:, :32] = 0
    pts0, keep0 = unproject_depth(depth, _cam("main", 0.0), stride=1)
    assert keep0.sum() == depth.size // 2, "depth 0 is unknown, never a point"
    assert len(pts0) == int(keep0.sum())
    print("  unproject: view-Z plane lands on y=11.2, yaw 90 lands on +x, depth 0 dropped")


def check_labels_sparse_8cams() -> None:
    from python.train.occ_sparse import GRID_SHAPE, OCC_CLASSES, occ_labels_for_tick

    yaws = {"narrow": 0, "main": 0, "wide": 0, "pillarL": -78, "pillarR": 78,
            "repeatL": -150, "repeatR": 150, "rear": 180}
    cams = [_cam(c, y) for c, y in yaws.items()]
    planes = {c: {"depth_cm": _plane(8.0), "sem": np.zeros((48, 64), np.uint8)} for c in yaws}
    lab = occ_labels_for_tick(planes, cams, sem_names={0: "unknown"}, row=None, stride=4)
    occ, cls, free = lab["occ_idx"], lab["occ_cls"], lab["free_idx"]
    dense = int(np.prod(GRID_SHAPE))
    assert occ.dtype == np.int32 and free.dtype == np.int32 and cls.dtype == np.uint8
    assert 0 < len(occ) < dense // 20, f"sparse: {len(occ)} of {dense}"
    assert len(np.unique(occ)) == len(occ), "one entry per voxel"
    assert not np.intersect1d(occ, free).size, "a hit voxel is never free"
    assert set(cls.tolist()) == {OCC_CLASSES.index("unknown")}, "class 0 stays unknown, not drivable"
    ix, iy, _ = np.unravel_index(occ, GRID_SHAPE)
    from python.train.occ_sparse import voxel_centers
    centers = voxel_centers(occ)
    assert (centers[:, 1] < -2.0).any(), "rear camera fills behind the ego"
    assert (centers[:, 0] > 2.0).any() and (centers[:, 0] < -2.0).any(), "pillars fill both sides"
    assert lab["cams"] == sorted(yaws), lab["cams"]
    print(f"  8 cams: {len(occ)} occupied + {len(free)} free of {dense} voxels, 360 coverage")


def check_drivable_and_missing_cams() -> None:
    from python.train.occ_sparse import OCC_CLASSES, occ_labels_for_tick

    # Pitch down onto the ground: allowlisted ROAD pixels become drivable.
    cam = _cam("main", 0.0, pitch=-30.0)
    sem = np.full((48, 64), 7, np.uint8)
    lab = occ_labels_for_tick({"main": {"depth_cm": _plane(2.4), "sem": sem}}, [cam],
                              sem_names={0: "unknown", 7: "ROAD"}, row=None, stride=4)
    assert set(lab["occ_cls"].tolist()) == {OCC_CLASSES.index("drivable")}
    # A camera with no planes is skipped and named, never filled.
    lab2 = occ_labels_for_tick({"main": {"depth_cm": _plane(5.0), "sem": None}}, [cam, _cam("rear", 180)],
                               sem_names={}, row=None, stride=4)
    assert lab2["cams"] == ["main"]
    assert set(lab2["occ_cls"].tolist()) == {OCC_CLASSES.index("unknown")}, "no sem -> unknown"
    print("  ROAD -> drivable, missing sem -> unknown, missing cam -> skipped")


def check_vehicle_voxels() -> None:
    from python.train.occ_sparse import OCC_CLASSES, occ_labels_for_tick, voxel_centers

    row = {"pos": [100.0, 100.0], "dir": [0.0, 1.0],
           "traffic": [{"name": "gvd_traffic_0", "pos": [103.0, 110.0], "dir": [0.0, 1.0], "vel": [0.0, 5.0]}]}
    lab = occ_labels_for_tick({}, [], sem_names={}, row=row, stride=4)
    veh = OCC_CLASSES.index("vehicle")
    assert (lab["occ_cls"] == veh).all() and len(lab["occ_cls"]) > 0
    c = voxel_centers(lab["occ_idx"])
    assert np.allclose(c[:, 0], 3.25, atol=0.26) and np.allclose(c[:, 1], 10.25, atol=0.26), c[:2]
    assert len(np.unique(c[:, :2], axis=0)) == 1, "center column only: no invented 4.5x1.8 footprint"
    for null_row in ({**row, "traffic": None}, {"pos": [0, 0], "dir": [0, 1]}):
        assert len(occ_labels_for_tick({}, [], sem_names={}, row=null_row)["occ_idx"]) == 0
    print("  traffic row -> one vehicle column at ego xy; null traffic scores nothing")


def check_shards_roundtrip() -> None:
    from python.train.occ_sparse import OccShardWriter, read_occ_shards

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        samples = []
        for i in range(7):
            n = int(rng.integers(1, 50))
            samples.append({"key": f"rec0_{i:06d}",
                            "occ_idx": rng.integers(0, 1000, n).astype(np.int32),
                            "occ_cls": rng.integers(0, 3, n).astype(np.uint8),
                            "free_idx": rng.integers(1000, 2000, n).astype(np.int32),
                            "recover": bool(i % 3 == 0)})
        with OccShardWriter(tmp, max_bytes=4096) as writer:
            for s in samples:
                writer.add(s)
        shards = sorted(Path(tmp).glob("occ-*.tar"))
        assert len(shards) >= 2, "rotates before max_bytes"
        assert not list(Path(tmp).glob("*.png")), "no loose files"
        with tarfile.open(shards[0]) as tf:
            names = tf.getnames()
        assert names[0].endswith(".occ_idx.npy") and names[0].split(".")[0] == "rec0_000000", names[:4]
        back = list(read_occ_shards(shards))
        assert [b["key"] for b in back] == [s["key"] for s in samples]
        for a, b in zip(samples, back):
            for k in ("occ_idx", "occ_cls", "free_idx"):
                assert np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype
            assert a["recover"] == b["recover"]
    print(f"  shards: {len(shards)} WebDataset tars, exact round trip")


def check_recovery_mix() -> None:
    from python.train.occ_sparse import is_recovery_row, recovery_weights

    rows = [{"tags": {"excursion": {"phase": "recover"}}}] * 5 + [{"tags": {"excursion": {"phase": "cruise"}}}] * 95
    assert is_recovery_row(rows[0]) and not is_recovery_row(rows[-1]) and not is_recovery_row({})
    w = recovery_weights(rows, target=0.25)
    share = float(w[:5].sum() / w.sum())
    assert abs(share - 0.25) < 1e-6, share
    try:
        recovery_weights(rows, target=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("target outside 0.20-0.30 is refused")
    w0 = recovery_weights(rows[5:], target=0.25)
    assert np.allclose(w0, 1.0), "no recoveries: uniform, never invented"
    print("  recovery mix: 5% recover rows -> 25% of draws; 0.5 refused")


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    import torch

    return torch


def check_head_training_only() -> None:
    torch = _torch()
    if torch is None:
        skip_check("occ_head_training_only", "PyTorch is not installed")
        return
    from python.train.scene_net import OUTPUT_FIELDS, SceneNet

    plain = SceneNet(cams=("main",))
    headed = SceneNet(cams=("main",), occ_sparse=True)
    assert not any(k.startswith("occ_head") for k in plain.state_dict())
    assert any(k.startswith("occ_head") for k in headed.state_dict())
    assert ":occ_sparse=" in (headed.pause_arch or "")
    missing, unexpected = headed.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("occ_head") for k in missing)
    strip = torch.zeros(1, 3, int(headed.canvas_h), int(headed.canvas_w))
    pred, _ = headed.forward(strip, 0.1, headed.initial_state())
    assert pred.occ_feat is not None and list(pred.as_dict()) == list(OUTPUT_FIELDS)
    assert "occ_feat" not in OUTPUT_FIELDS
    base, _ = plain.forward(strip, 0.1, plain.initial_state())
    assert getattr(base, "occ_feat", None) is None
    q = torch.randint(0, 1000, (1, 64))
    logits = headed.occ_head(pred.occ_feat.reshape(1, -1), q)
    assert tuple(logits.shape) == (1, 64, 1 + 3)
    n = sum(p.numel() for p in headed.occ_head.parameters())
    assert n < 100_000, n
    print(f"  occ head opt-in ({n} params), plain ckpt loads, ONNX fields unchanged")


def check_loss_masks_and_overfit() -> None:
    torch = _torch()
    if torch is None:
        skip_check("occ_overfit", "PyTorch is not installed")
        return
    from python.train.occ_sparse import SparseOccHead, build_queries, occ_loss

    torch.manual_seed(0)
    head = SparseOccHead(32)
    empty = {"occ_idx": np.zeros(0, np.int32), "occ_cls": np.zeros(0, np.uint8), "free_idx": np.zeros(0, np.int32)}
    q, t_occ, t_cls, m = build_queries([empty], k=16, seed=0)
    loss = occ_loss(head(torch.zeros(1, 32), q), t_occ, t_cls, m)
    assert float(loss) == 0.0, "no labels -> zero loss, not a fake free sky"
    rng = np.random.default_rng(1)
    labels = []
    for _ in range(4):
        occ = np.unique(rng.integers(0, 50000, 40)).astype(np.int32)
        free = np.setdiff1d(np.unique(rng.integers(0, 50000, 60)), occ).astype(np.int32)
        labels.append({"occ_idx": occ, "occ_cls": rng.integers(0, 3, len(occ)).astype(np.uint8), "free_idx": free})
    feat = torch.randn(4, 32)
    q, t_occ, t_cls, m = build_queries(labels, k=128, seed=0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(400):
        opt.zero_grad()
        loss = occ_loss(head(feat, q), t_occ, t_cls, m)
        first = float(loss) if first is None else first
        loss.backward()
        opt.step()
    out = head(feat, q)
    pred_occ = (out[..., 0] > 0).float()
    acc = float(((pred_occ == t_occ).float() * m).sum() / m.sum())
    assert float(loss) < 0.1 * first and acc > 0.97, (first, float(loss), acc)
    print(f"  overfit: loss {first:.3f} -> {float(loss):.4f}, occ acc {acc:.3f}")


def check_build_reads_shards_not_pngs() -> None:
    """Recording dir -> shards; the loader then works with raw/ gone (no PNG fallback)."""
    import shutil
    import tempfile

    from python.data.raw_frames import RawFrameWriter, write_sem_classes
    from python.train.occ_sparse import build_occ_shards, read_occ_shards

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        writer = RawFrameWriter(root, [])
        sem = np.zeros((48, 64), np.uint8)
        sem[24:, :] = 2  # ROAD below the horizon
        for i in range(3):
            writer.write(
                {"main": np.full((48, 64, 3), 40, np.uint8), "main_depth": _plane(10.0), "main_sem": sem},
                t=float(i), timestamps={"main": 1.0 + i}, tech={},
            )
        write_sem_classes(root, {"CAR": (0, 0, 142), "ROAD": (128, 64, 128), "SKY": (70, 130, 180)})
        (root / "cameras.json").write_text(json.dumps({"cameras": [_cam("main", 0.0)]}), encoding="utf-8")
        paths = build_occ_shards(root, Path(tmp) / "shards")
        assert paths and all(str(p).endswith(".tar") for p in paths), paths
        shutil.rmtree(root)  # no PNGs left to fall back to
        got = list(read_occ_shards(paths))
        assert len(got) == 3, len(got)
        assert all(len(g["occ_idx"]) > 0 and g["cams"] == ["main"] for g in got), got[0]
    print("  build: recording -> 3 shard samples, loader reads them with raw/ deleted")


def check_live_entry_points_skip_occ() -> None:
    """Runtime stays camera-only: importing the live entry point never loads the GT labeler."""
    import os
    import subprocess

    code = "import sys, python.run_vision; print([m for m in sys.modules if 'occ_sparse' in m])"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
    for live in ("python/run_vision.py", "python/train/scene_live.py"):
        path = ROOT / live
        if path.is_file():
            assert "occ_sparse" not in path.read_text(encoding="utf-8"), live
    print("  live: run_vision / scene_live never import occ_sparse")


def main() -> int:
    check_unproject_forward_and_side()
    check_labels_sparse_8cams()
    check_drivable_and_missing_cams()
    check_vehicle_voxels()
    check_shards_roundtrip()
    check_recovery_mix()
    check_build_reads_shards_not_pngs()
    check_live_entry_points_skip_occ()
    check_head_training_only()
    check_loss_masks_and_overfit()
    print("test_occ_sparse: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
