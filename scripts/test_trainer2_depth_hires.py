#!/usr/bin/env python3
"""Trainer 2.0: 48x64 Reality View depth + far/sky class replacing the 12x16 aux_depth grid.

Checks (python/train/depth_hires.py, opt-in ``--aux-depth --heads ...,depth_hires``):
  check_hires_targets_far_and_masks   uint16 cm -> 48x64 metres / valid / far / far_valid
  check_loader_train_only_hires       loader + labels: 48x64 from main_depth, cache bypassed, 12x16 unchanged when off
  check_net_hires_shapes_camera_only  SceneNet depth_hires: (B,1,48,64) depth + far from the image only; token unchanged
  check_loss_learns_and_rejects       scene_loss trains the head (depth + far drop); shape / missing-far errors raise
  check_default_off_unchanged         default heads / nets carry no depth_hires; build_net round-trips the head
  check_cli_needs_aux_depth           --heads lane_gt,depth_hires without --aux-depth stops before any data read
  check_bench_within_live_budget      bench_depth_hires: 48x64 out, decoder < 50k params, est VRAM <= 4 GB (tiny, small)
Torch checks skip visibly without torch.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402

H, W = 192, 256  # main camera crop at scale 1


def _torch():
    try:
        import torch
    except ImportError:
        return None
    # Tiny nets: more intra-op threads only oversubscribe a shared CI / box CPU.
    torch.set_num_threads(2)
    return torch


def _scene_depth_cm() -> np.ndarray:
    """Sky (300 m far plane) in the top quarter, 12.5 m road band, 40 m block, a zero (invalid) patch."""
    cm = np.full((H, W), 1250, np.uint16)
    cm[: H // 4] = 30000
    cm[H // 2 :, : W // 2] = 4000
    cm[H - 16 :, W - 32 :] = 0
    return cm


def check_hires_targets_far_and_masks() -> None:
    from python.train.depth_hires import DEPTH_HI_H, DEPTH_HI_W, FAR_CLASS_M, depth_hires_targets

    assert (DEPTH_HI_H, DEPTH_HI_W) == (48, 64) and DEPTH_HI_H * DEPTH_HI_W == 16 * 12 * 16
    t = depth_hires_targets(_scene_depth_cm())
    for key in ("depth_m", "depth_valid", "far", "far_valid"):
        assert t[key].shape == (48, 64) and t[key].dtype == np.float32, (key, t[key].shape, t[key].dtype)
    sky = slice(0, 12)
    assert np.all(t["far"][sky] == 1.0) and np.all(t["depth_valid"][sky] == 0.0), "sky rows are far, not metric"
    assert np.all(t["far_valid"][sky] == 1.0)
    assert np.all(t["far"][12:] == 0.0), "no far below the sky"
    assert np.allclose(t["depth_m"][12:24], 12.5) and np.all(t["depth_valid"][12:24] == 1.0)
    assert np.allclose(t["depth_m"][24:40, :32], 40.0)
    dead = (slice(48 - 4, 48), slice(64 - 8, 64))
    assert np.all(t["depth_valid"][dead] == 0.0) and np.all(t["far_valid"][dead] == 0.0), "0 cm is unknown, not far"
    assert np.all(t["depth_m"][dead] == 0.0)
    assert not np.any((t["depth_valid"] > 0) & (t["far"] > 0)), "a cell is metric or far, never both"
    # Boundary: exactly FAR_CLASS_M is far; one cm under is metric.
    edge = np.full((H, W), int(FAR_CLASS_M * 100) - 1, np.uint16)
    edge[:, : W // 2] = int(FAR_CLASS_M * 100)
    e = depth_hires_targets(edge)
    assert np.all(e["far"][:, :32] == 1.0) and np.all(e["far"][:, 32:] == 0.0)
    assert np.allclose(e["depth_m"][:, 32:], FAR_CLASS_M - 0.01, atol=1e-3)
    # Mixed cell: 3/4 far is far; 1/4 far stays metric with the near mean.
    mixed = np.full((4, 4), 1000, np.uint16)
    mixed[:3] = 30000
    m = depth_hires_targets(np.tile(mixed, (48, 64)))
    assert np.all(m["far"] == 1.0)
    mixed2 = np.full((4, 4), 30000, np.uint16)
    mixed2[:3] = 1000
    m2 = depth_hires_targets(np.tile(mixed2, (48, 64)))
    assert np.all(m2["far"] == 0.0) and np.all(m2["depth_valid"] == 1.0) and np.allclose(m2["depth_m"], 10.0)
    for bad in (np.zeros((H, W), np.float32), np.zeros((H, W, 1), np.uint16)):
        try:
            depth_hires_targets(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad.dtype} {bad.shape} accepted")
    print(f"targets: 48x64; sky=far (no metric), 0 cm masked, far boundary at {FAR_CLASS_M} m, mixed cells by majority OK")


def _write_rec(root: Path, depth: np.ndarray) -> None:
    from python.data.raw_frames import RawFrameWriter, write_sem_classes

    sem = np.zeros(depth.shape, np.uint8)
    frames = {"main": np.full((*depth.shape, 3), 40, np.uint8), "main_depth": depth, "main_sem": sem}
    RawFrameWriter(root, []).write(frames, t=0.0, timestamps={"main": 1.0}, tech={})
    write_sem_classes(root, {"ROAD": (128, 64, 128), "SKY": (70, 130, 180)})


class _PoisonCache:
    """A 12x16 cache that must not serve a depth_hires read."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, index, source=None, cam="main"):
        self.calls += 1
        z = np.full((12, 16), 99.0, np.float32)
        return {"depth_m": z, "depth_valid": np.ones_like(z), "sem_id": np.zeros((12, 16), np.int64),
                "sem_valid": np.ones_like(z), "sem_classes": {}}


def check_loader_train_only_hires() -> None:
    from python.train.depth_hires import depth_hires_targets
    from python.train.sensor_aux import AUX_H, AUX_W, attach_aux_to_labels, load_sensor_aux_targets

    depth = _scene_depth_cm()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_rec(root, depth)
        cache = _PoisonCache()
        got = load_sensor_aux_targets(root, 0, cam="main", aux_cache=cache, depth_hires=True)
        assert cache.calls == 0, "depth_hires must bypass the 12x16 aux cache"
        want = depth_hires_targets(depth)
        assert np.array_equal(got["depth_hi_m"], want["depth_m"]) and np.array_equal(got["depth_hi_far"], want["far"])
        assert got["depth_m"].shape == (AUX_H, AUX_W), "the 12x16 depth is still computed (sem/back-compat)"
        off = load_sensor_aux_targets(root, 0, cam="main", aux_cache=cache)
        assert cache.calls == 1 and float(off["depth_m"][0, 0]) == 99.0, "off: cache path unchanged"
        assert "depth_hi_m" not in off
        lab_hi = attach_aux_to_labels({}, root=root, index=0, want_depth=True, want_depth_hires=True)
        assert lab_hi["aux_depth"].shape == (48, 64) and lab_hi["aux_depth_far"].shape == (48, 64)
        assert set(lab_hi) == {"aux_depth", "aux_depth_valid", "aux_depth_far", "aux_depth_far_valid"}, sorted(lab_hi)
        lab_lo = attach_aux_to_labels({}, root=root, index=0, want_depth=True)
        assert lab_lo["aux_depth"].shape == (AUX_H, AUX_W) and "aux_depth_far" not in lab_lo
        lab_sem = attach_aux_to_labels({}, root=root, index=0, want_sem=True, want_depth_hires=True)
        assert "aux_depth" not in lab_sem, "hires never turns depth on by itself"
        # Missing files: all masked, nothing invented.
        empty = load_sensor_aux_targets(root, 7, cam="main", depth_hires=True)
        assert all(float(empty[k].sum()) == 0.0 for k in ("depth_hi_m", "depth_hi_valid", "depth_hi_far", "depth_hi_far_valid"))
    from python.data.raw_frames import is_train_only_cam

    assert is_train_only_cam("main_depth") and not is_train_only_cam("main")
    print("loader: 48x64 from train-only main_depth, 12x16 cache bypassed only when hires, missing tick fully masked OK")


def _net(torch, hires: bool, scale: float = 1.0, seed: int = 0):
    from python.train.scene_net import SceneNet

    torch.manual_seed(seed)
    return SceneNet(cams=("main",), scale=scale, aux_sensors=True, depth_hires=hires).eval()


def _run(net, img):
    return net.forward_batch(img, 0.1, net.initial_state(batch=int(img.shape[0])))


def _image(torch, net, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, net.canvas_h, net.canvas_w, generator=g)


def _scene_image(torch, net):
    """A camera picture laid out like ``_scene_depth_cm`` (bright sky, two grey bands), batch of 2."""
    import cv2

    cm = _scene_depth_cm().astype(np.float32)
    grey = np.where(cm >= 25000, 0.95, np.where(cm >= 4000, 0.25, np.where(cm > 0, 0.55, 0.05))).astype(np.float32)
    crop_w = int(net.sectors[0].x1 - net.sectors[0].x0)
    grey = cv2.resize(grey, (crop_w, int(net.canvas_h)), interpolation=cv2.INTER_AREA)
    img = torch.zeros(2, 3, int(net.canvas_h), int(net.canvas_w))
    img[:, :, :, net.sectors[0].x0 : net.sectors[0].x1] = torch.from_numpy(grey)
    return img


def check_net_hires_shapes_camera_only() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_net_hires_shapes_camera_only", "torch not installed")
        return
    from python.train.depth_hires import ARCH_TOKEN, stem_pyramid

    for scale in (1.0, 0.5):
        net = _net(torch, True, scale)
        assert ARCH_TOKEN in net.pause_arch and not hasattr(net, "aux_depth_head"), "hires replaces the 1x1 12x16 head"
        img = _image(torch, net)
        with torch.no_grad():
            pred, _state = _run(net, img)
            pyr = stem_pyramid(net.stem, img[:, :, :, net.sectors[0].x0 : net.sectors[0].x1])
            ref = net.stem.features(img[:, :, :, net.sectors[0].x0 : net.sectors[0].x1])
        assert tuple(pred.aux_depth.shape) == (2, 1, 48, 64), (scale, tuple(pred.aux_depth.shape))
        assert tuple(pred.aux_depth_far.shape) == (2, 1, 48, 64)
        assert bool((pred.aux_depth > 0).all()), "metres are positive"
        assert tuple(pred.aux_sem_logits.shape[-2:]) == (12, 16)
        assert len(pyr) == 4 and torch.equal(pyr[-1], ref), "pyramid top is exactly stem.features"
        # Camera-only: forward takes the image (and nothing from BeamNG); the head is off the token path.
        low = _net(torch, False, scale)
        low.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("depth_hires_head.")}, strict=False)
        with torch.no_grad():
            pred_low, _ = _run(low, img)
        assert torch.equal(pred_low.lanes, pred.lanes) and torch.equal(pred_low.aux_sem_logits, pred.aux_sem_logits)
    import inspect

    params = list(inspect.signature(net.forward_batch).parameters)
    assert not any("depth" in p or "label" in p or "sensor" in p for p in params), params
    print("net: (B,1,48,64) depth + far at scale 1 and 0.5; pyramid top == stem.features; planner outputs unchanged OK")


def check_loss_learns_and_rejects() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_loss_learns_and_rejects", "torch not installed")
        return
    from python.train.depth_hires import depth_hires_targets
    from python.train.scene_net import scene_loss
    from python.train.sensor_aux import AuxTargetError

    net = _net(torch, True, 0.5).train()
    img = _scene_image(torch, net)
    t = depth_hires_targets(_scene_depth_cm())
    target = {
        "aux_depth": torch.from_numpy(np.stack([t["depth_m"]] * 2)),
        "aux_depth_valid": torch.from_numpy(np.stack([t["depth_valid"]] * 2)),
        "aux_depth_far": torch.from_numpy(np.stack([t["far"]] * 2)),
        "aux_depth_far_valid": torch.from_numpy(np.stack([t["far_valid"]] * 2)),
    }
    from python.train.depth_hires import depth_hires_loss

    from python.train.depth_hires import stem_pyramid

    # The decoder alone learns the map from the (fixed) stem pyramid: cheap on a loaded CPU box.
    with torch.no_grad():
        crop = img[:, :, :, net.sectors[0].x0 : net.sectors[0].x1]
        pyramid = [m.detach() for m in stem_pyramid(net.stem, crop)]
    opt = torch.optim.Adam(net.depth_hires_head.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        depth_m, far_logit = net.depth_hires_head(pyramid)
        d, f = depth_hires_loss(depth_m, far_logit, target)
        if first is None:
            first = (float(d.detach()), float(f.detach()))
        opt.zero_grad()
        (d + f).backward()
        opt.step()
    last = (float(d.detach()), float(f.detach()))
    assert last[0] < 0.5 * first[0] and last[1] < 0.5 * first[1], (first, last)
    # Wired into scene_loss as the aux_depth term (weight > 0), zero when weight is 0.
    pred, _ = _run(net, img)
    full = _full_target(torch, pred, target)
    out = scene_loss(pred, full, heads=("lane_gt",), aux_depth_weight=1.0)
    d2, f2 = depth_hires_loss(pred.aux_depth, pred.aux_depth_far, target)
    got, want = float(out["aux_depth"].detach()), float((d2 + f2).detach())
    assert abs(got - want) < 1e-5, (got, want)
    assert float(scene_loss(pred, full, heads=("lane_gt",), aux_depth_weight=0.0)["aux_depth"]) == 0.0
    no_far = {k: v for k, v in full.items() if not k.startswith("aux_depth_far")}
    for bad, why in ((no_far, "missing far"), ({**full, "aux_depth": full["aux_depth"][:, :12, :16]}, "12x16 target")):
        try:
            scene_loss(pred, bad, heads=("lane_gt",), aux_depth_weight=1.0)
        except AuxTargetError:
            continue
        raise AssertionError(f"{why} accepted")
    print(f"loss: depth {first[0]:.3f}->{last[0]:.3f}, far {first[1]:.3f}->{last[1]:.3f} in 60 decoder steps; scene_loss term matches; bad targets raise OK")


def _full_target(torch, pred, extra):
    """Zero scene targets shaped like ``pred`` (heads=lane_gt), plus the hires depth targets."""
    out = {}
    for name in ("lanes", "lane_valid", "curbs", "curb_valid", "objects", "object_valid", "signs", "sign_valid"):
        out[name] = torch.zeros_like(getattr(pred, name))
    for name in ("object_class", "sign_class", "sign_state"):
        out[name] = torch.zeros(getattr(pred, name).shape[:-1], dtype=torch.int64)
    out["lane_gt"] = torch.zeros(pred.lane_gt.shape[0], pred.lane_gt.shape[-1])
    out["lane_gt_valid"] = torch.zeros(pred.lane_gt.shape[0], pred.lane_gt.shape[-1])
    out.update(extra)
    return out


def check_default_off_unchanged() -> None:
    from python.train.cam_strip import HEAD_NAMES, parse_heads

    assert "depth_hires" not in HEAD_NAMES and "depth_hires" not in parse_heads("")
    assert parse_heads("lane_gt,depth_hires") == ("lane_gt", "depth_hires")
    torch = _torch()
    if torch is None:
        skip_check("check_default_off_unchanged", "torch not installed (net part)")
        return
    from python.train.epoch_ckpt import build_net
    from python.train.scene_net import SceneNet

    torch.manual_seed(0)
    plain = SceneNet(cams=("main",))
    torch.manual_seed(0)
    aux = SceneNet(cams=("main",), aux_sensors=True)
    for net in (plain, aux):
        assert net.depth_hires is False and not any(k.startswith("depth_hires_head.") for k in net.state_dict())
        assert ":depth_hi=" not in str(getattr(net, "pause_arch", ""))
    assert "aux_depth_head.weight" in aux.state_dict()
    with torch.no_grad():
        p, _ = _run(aux.eval(), torch.rand(1, 3, aux.canvas_h, aux.canvas_w))
    assert tuple(p.aux_depth.shape) == (1, 1, 12, 16) and p.aux_depth_far is None
    try:
        SceneNet(cams=("main",), depth_hires=True)
    except ValueError:
        pass
    else:
        raise AssertionError("depth_hires without aux_sensors accepted")
    hires = _net(torch, True)
    weights = {k: v.numpy() for k, v in hires.state_dict().items()}
    rebuilt = build_net({"cams": ["main"], "scale": 1.0, "backbone": "scratch"}, weights)
    assert rebuilt.depth_hires and set(rebuilt.state_dict()) == set(hires.state_dict())
    rebuilt_aux = build_net({"cams": ["main"], "scale": 1.0, "backbone": "scratch"},
                            {k: v.numpy() for k, v in aux.state_dict().items()})
    assert not rebuilt_aux.depth_hires
    print("default: no depth_hires head/arch/output unless named; 12x16 path intact; build_net round-trips both OK")


def check_cli_needs_aux_depth() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_cli_needs_aux_depth", "torch not installed")
        return
    from python.train.train_scene import train_directory

    with tempfile.TemporaryDirectory() as tmp:
        try:
            train_directory(Path(tmp) / "nope", epochs=1, lr=1e-3, capability=None, export_path=None,
                            heads=("lane_gt", "depth_hires"), aux_depth=False)
        except SystemExit as exc:
            assert "--aux-depth" in str(exc), exc
        else:
            raise AssertionError("depth_hires without --aux-depth trained")
    print("cli: --heads lane_gt,depth_hires without --aux-depth exits naming --aux-depth OK")


def check_bench_within_live_budget() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_bench_within_live_budget", "torch not installed")
        return
    import bench_depth_hires as bench

    for preset in ("tiny", "small"):
        row = bench.bench(preset, reps=1)
        assert row["output_shape"] == [1, 1, 48, 64], row["output_shape"]
        assert row["decoder_params"] < 50_000, row["decoder_params"]
        assert row["net_params_hires"] - row["net_params_12x16"] < 50_000
        assert row["est_total_vram_gb"] <= bench.LIVE_VRAM_GB, row
        assert row["decoder_activation_bytes"] < 4 * 1024 * 1024, row["decoder_activation_bytes"]
    print(f"bench: tiny/small decoder {row['decoder_params']} params, est VRAM {row['est_total_vram_gb']:.3f} GB <= 4 GB OK")


def main() -> int:
    checks = (
        check_hires_targets_far_and_masks,
        check_loader_train_only_hires,
        check_net_hires_shapes_camera_only,
        check_loss_learns_and_rejects,
        check_default_off_unchanged,
        check_cli_needs_aux_depth,
        check_bench_within_live_budget,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
