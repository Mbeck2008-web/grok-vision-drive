#!/usr/bin/env python3
"""Sparse occupancy net: overfit, sparsity, memory estimate, camera-only runtime, fuzzed inputs.

# gvd-ci: requires torch

INVARIANTS (net side; I1-I9 are in test_occ_sparse_props.py)
 N1  overfit: on one synthetic scene the training loss falls to <= 50% of its
     start and occupancy accuracy on the training queries reaches >= 90%.
 N2  size: params < 3.2 M (YOLOv8n is 3.2 M); CPU estimate of inference at
     640x480 fp16 <= 4 GB and AMP training (batch 8) <= 11 GB. These are
     estimates from a CPU forward, not GPU measurements.
 N3  sparse: inference queries <= 20% of the dense grid and never evaluates
     a dense volume; output cells are unique int32 in-bounds indices with
     conf in [0, 1] and a known class.
 N4  fuzzed images (black, white, NaN, inf, odd sizes) never produce NaN
     outputs; images under 16 px or batch != 1 are refused.
 N5  camera-only: the runtime import closure never reaches labels, shards,
     sampler, build_shards, train_occ or any recorder/GT module, and the
     runtime step takes no depth/sem/box argument.
 N6  runtime -> viz: every frame validates against v1; frame 1 has no motion
     state; every track/edge carries a src.

    py -3 scripts/test_occ_sparse_net.py
"""
from __future__ import annotations

import ast
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from python.occ import classes as C  # noqa: E402
from python.occ.camera import CameraModel, cameras_from_config  # noqa: E402
from python.occ.labels import CamFrame, voxel_labels  # noqa: E402
from python.occ.model import OccSparseNet, make_queries, memory_estimate, occ_loss, param_count  # noqa: E402
from python.occ.synthetic import SEM_CLASSES, scene  # noqa: E402
from python.occ.viz_adapter import validate_v1  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
MAIN = cameras_from_config(ids=("main",))["main"]


def _train_batch(cam: CameraModel, n: int = 2):
    sc = scene(cam)
    lab = voxel_labels([CamFrame(cam, sc["depth"], sc["sem"])], SEM_CLASSES, [sc["box"]], max_free=1500, seed=0)
    q = make_queries(lab, n_pos=384, n_neg=384, seed=0)
    img = torch.from_numpy(sc["rgb"]).permute(2, 0, 1).float()[None] / 255.0
    images = {"main": img.repeat(n, 1, 1, 1)}
    qt = {k: torch.from_numpy(np.stack([v] * n)) for k, v in q.items()}
    return images, qt, lab


def test_overfit_loss_falls():
    cam = MAIN.scaled(128, 96)
    images, q, lab = _train_batch(cam)
    assert (lab["occ_cls"] == C.ROAD).any() and (lab["occ_cls"] == C.VEHICLE).any() and len(lab["free_idx"]) > 100
    model = OccSparseNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    t0 = time.time()
    for _ in range(150):
        out = model(images, {"main": cam}, q["pts"], coarse_pts=q["coarse_pts"])
        parts = occ_loss(out, q)
        opt.zero_grad()
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(parts["total"]))
    model.eval()
    with torch.no_grad():
        out = model(images, {"main": cam}, q["pts"], coarse_pts=q["coarse_pts"])
    m = q["mask"] > 0
    acc = float(((out["occ_logit"] > 0).float() == q["occ_t"])[m].float().mean())
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    print(f"overfit: loss {first:.3f} -> {last:.3f} ({last / first:.2%}), occ acc {acc:.3f}, {time.time() - t0:.1f}s CPU")
    assert all(math.isfinite(v) for v in losses)
    assert last <= 0.5 * first, (first, last)
    assert acc >= 0.9, acc


def test_size_and_memory_estimate():
    model = OccSparseNet()
    p = param_count(model)
    est = memory_estimate(model, {"main": MAIN}, height=480, width=640, n_queries=16384, train_batch=8, train_queries=2048)
    print(f"params {p:,}; CPU estimate: infer fp16 {est['infer_gb']:.2f} GB, AMP train b8 {est['train_gb']:.2f} GB (incl. {est['cuda_context_gb']} GB CUDA context allowance); not a GPU measurement")
    assert p < 3_200_000
    assert est["infer_gb"] <= 4.0 and est["train_gb"] <= 11.0


def _check_cells(cells, model):
    idx, cls, conf = cells["idx"], cells["cls"], cells["conf"]
    assert idx.dtype == np.int32 and idx.ndim == 2 and idx.shape[1] == 3
    assert ((idx >= 0) & (idx < np.array(model.grid.shape))).all()
    assert np.unique(model.grid.flat(idx)).size == idx.shape[0]
    assert cls.shape == (idx.shape[0],) and ((cls >= 0) & (cls < C.N_CLASSES)).all()
    assert np.isfinite(conf).all() and ((conf >= 0) & (conf <= 1)).all()


def test_sparse_inference():
    model = OccSparseNet().eval()
    res = model.infer({"main": torch.rand(1, 3, 480, 640)}, {"main": MAIN})
    st = res["stats"]
    frac = (st["fine_queries"] + st["coarse_queries"]) / st["n_dense"]
    print(f"queries: coarse {st['coarse_queries']} + fine {st['fine_queries']} of dense {st['n_dense']} ({frac:.1%}); cells out {len(res['cells']['idx'])}")
    assert frac <= 0.2
    _check_cells(res["cells"], model)
    # nothing above the coarse threshold -> no fine queries at all
    res0 = model.infer({"main": torch.rand(1, 3, 64, 64)}, {"main": MAIN}, coarse_thr=1.1)
    assert res0["stats"]["fine_queries"] == 0 and res0["cells"]["idx"].shape == (0, 3)


def test_fuzzed_images():
    model = OccSparseNet().eval()
    for img in (torch.zeros(1, 3, 48, 64), torch.ones(1, 3, 48, 64), torch.full((1, 3, 48, 64), float("nan")),
                torch.full((1, 3, 17, 333), float("inf")), torch.rand(1, 3, 1080, 1920)):
        res = model.infer({"main": img}, {"main": MAIN})
        _check_cells(res["cells"], model)
    for bad in (torch.zeros(1, 3, 8, 8), torch.zeros(2, 3, 64, 64), torch.zeros(3, 64, 64)):
        try:
            model.infer({"main": bad}, {"main": MAIN})
        except ValueError:
            continue
        raise AssertionError(f"accepted {tuple(bad.shape)}")
    from python.occ.runtime import OccRuntime

    rt = OccRuntime(model, {"main": MAIN})
    for bad in (np.zeros((48, 64), np.uint8), np.zeros((48, 64, 3), np.float32), np.zeros((48, 64, 4), np.uint8)):
        try:
            rt.step(0.0, {"main": bad}, speed_mps=1.0)
        except ValueError:
            continue
        raise AssertionError(f"runtime accepted {bad.dtype} {bad.shape}")


FORBIDDEN = ("python.occ.labels", "python.occ.shards", "python.occ.sampler", "python.occ.build_shards", "python.occ.train_occ",
             "python.occ.synthetic", "python.train", "python.collect", "python.data", "python.debug")


def _imports(mod: str) -> set[str]:
    path = ROOT / (mod.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return {m for m in out if m.startswith("python.")}


def test_runtime_camera_only():
    seen, todo = set(), ["python.occ.runtime"]
    while todo:
        m = todo.pop()
        if m in seen or not (ROOT / (m.replace(".", "/") + ".py")).is_file():
            continue
        seen.add(m)
        todo.extend(_imports(m))
    bad = sorted(m for m in seen if m.startswith(FORBIDDEN))
    print(f"runtime import closure: {sorted(seen)}")
    assert not bad, bad
    for m in seen:
        text = (ROOT / (m.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert "_depth\"" not in text and "_sem\"" not in text and "sem_classes" not in text, m
    from python.occ.runtime import OccRuntime

    params = set(inspect.signature(OccRuntime.step).parameters)
    assert params == {"self", "t", "images", "speed_mps", "yaw_rate", "engaged"}, params
    fwd = set(inspect.signature(OccSparseNet.forward).parameters)
    assert not any(k in fwd for k in ("depth", "sem", "boxes", "labels")), fwd


def test_runtime_to_viz():
    from python.occ.runtime import OccRuntime

    cam = MAIN.scaled(128, 96)
    images, q, _ = _train_batch(cam, n=1)
    model = OccSparseNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60):
        out = model(images, {"main": cam}, q["pts"], coarse_pts=q["coarse_pts"])
        loss = occ_loss(out, q)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    rt = OccRuntime(model, {"main": cam})
    rgb = scene(cam)["rgb"]
    for k in range(4):
        r = rt.step(0.1 * k, {"main": rgb}, speed_mps=0.0)
        msg = r["viz"]
        assert not validate_v1(msg), validate_v1(msg)
        for tr in msg["tracks"]:
            assert tr["src"] in C.KNOWN_SOURCES and (k > 0 or tr["motion"] == "unknown")
        for ed in msg["edges"]:
            assert ed["src"] in C.KNOWN_SOURCES and ed["type"] in C.RED_POLYLINE_KINDS
    print(f"runtime -> viz: {len(msg['occ']['cells'])} cells, {len(msg['edges'])} edges, {len(msg['tracks'])} tracks, valid v1")


def test_fp16_forward():
    from gvd_test_env import skip_check

    model = OccSparseNet().eval().half()
    try:
        res = model.infer({"main": torch.rand(1, 3, 96, 128).half()}, {"main": MAIN})
    except (RuntimeError, NotImplementedError) as exc:
        skip_check("fp16_forward", f"CPU half not supported here: {exc}")
        return
    _check_cells(res["cells"], model)


TESTS = ["test_overfit_loss_falls", "test_size_and_memory_estimate", "test_sparse_inference", "test_fuzzed_images",
         "test_runtime_camera_only", "test_runtime_to_viz", "test_fp16_forward"]


def main() -> int:
    for name in TESTS:
        globals()[name]()
        print(f"{name} OK", flush=True)
    print("occ_sparse net N1-N6 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
