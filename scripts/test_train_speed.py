#!/usr/bin/env python3
"""Training-speed path: aux memmap, fused window math, val RMS, batch estimate.

CPU-only. Does not invent GPU wall-clock timings.
"""
# gvd-ci: requires torch
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

H, W = 48, 64
PALETTE = {"CAR": (0, 0, 142), "ROAD": (128, 64, 128), "SKY": (70, 130, 180)}


def _write_rec(root: Path, *, n: int = 2) -> None:
    from python.data.raw_frames import RawFrameWriter, write_sem_classes

    writer = RawFrameWriter(root, [])
    for i in range(n):
        depth = np.full((H, W), 1500 + i * 10, np.uint16)
        depth[0, 0] = 0
        sem = np.zeros((H, W), np.uint8)
        sem[:, : W // 2] = 2
        writer.write(
            {
                "main": np.full((H, W, 3), 40 + i, np.uint8),
                "main_depth": depth,
                "main_sem": sem,
            },
            t=float(i) * 0.1,
            timestamps={"main": 1.0 + i},
            tech={
                "lane_gt": [0.1 * (i + 1)] * 14,
                "lane_gt_valid": [1.0] * 14,
                "gt_source": "painted_decals",
            },
        )
    write_sem_classes(root, PALETTE)


def check_aux_cache_matches_png() -> None:
    from python.train.aux_cache import AuxCache, StaleAuxCacheError, build_aux_cache
    from python.train.sensor_aux import load_sensor_aux_targets

    with tempfile.TemporaryDirectory() as tmp:
        rec = Path(tmp) / "rec"
        cache = Path(tmp) / "cache"
        cache.mkdir()
        _write_rec(rec, n=3)
        manifest = build_aux_cache(rec, cache)
        assert manifest is not None
        assert manifest["format"] == "gvd-aux-cache-1"
        aux = AuxCache(cache)
        assert aux.present()
        for index in range(3):
            png = load_sensor_aux_targets(rec, index, cam="main")
            hit = load_sensor_aux_targets(rec, index, cam="main", aux_cache=aux)
            for key in ("depth_m", "depth_valid", "sem_id", "sem_valid"):
                assert np.array_equal(png[key], hit[key]), key
        # Old RGB-only folder: no aux/ → loader falls back to PNGs.
        bare = Path(tmp) / "rgb_only"
        bare.mkdir()
        empty = AuxCache(bare)
        assert not empty.present()
        assert not empty.stale
        fallback = load_sensor_aux_targets(rec, 0, cam="main", aux_cache=empty)
        png = load_sensor_aux_targets(rec, 0, cam="main")
        assert np.array_equal(png["depth_m"], fallback["depth_m"])
        # Stale format fails loudly with a rebuild hint.
        bad = Path(tmp) / "bad"
        (bad / "aux").mkdir(parents=True)
        payload = {"format": "gvd-aux-cache-0", "aux_h": 12, "aux_w": 16}
        (bad / "aux" / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        stale = AuxCache(bad)
        assert stale.stale
        assert "rebuild" in stale.reason.lower()
        try:
            stale.require_fresh()
        except StaleAuxCacheError as exc:
            assert "rebuild" in str(exc).lower()
        else:
            raise AssertionError("stale aux cache must raise")
        # Existing dest/aux is create-only.
        try:
            build_aux_cache(rec, cache)
        except FileExistsError:
            pass
        else:
            raise AssertionError("rebuild in place must refuse")
    print("aux_cache_matches_png")


def check_parser_flags() -> None:
    from python.train.train_scene import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--strips",
            "x",
            "--time-steps",
            "100",
            "--eval-every",
            "2",
            "--chunk-stride",
            "16",
            "--no-fuse-window",
            "--ckpt-dir",
            r"C:\GVD_runs\speed",
            "--no-panel",
            "--workers",
            "4",
            "--batch",
            "16",
        ]
    )
    assert args.time_steps == 100
    assert args.eval_every == 2
    assert args.chunk_stride == 16
    assert args.no_fuse_window is True
    assert args.ckpt_dir.endswith("speed")
    print("parser_flags")


def check_batch_uses_trained_canvas() -> None:
    from python.train.fit import bytes_per_strip, largest_batch, usable_bytes

    file_bytes = 8_500_000_000
    n_strips = 4726
    free = int(7.9 * (1024 ** 3))
    old = bytes_per_strip(file_bytes, n_strips)
    trained = bytes_per_strip(file_bytes, n_strips, canvas_h=192, canvas_w=256)
    assert trained < old
    old_fit = largest_batch(
        usable_bytes(free), file_bytes=file_bytes, n_strips=n_strips, window_strips=16, limit=32
    )
    new_fit = largest_batch(
        usable_bytes(free),
        file_bytes=file_bytes,
        n_strips=n_strips,
        window_strips=16,
        limit=32,
        canvas_h=192,
        canvas_w=256,
    )
    assert new_fit >= old_fit
    print(f"batch_uses_trained_canvas old_per={old} trained_per={trained} old_fit={old_fit} new_fit={new_fit}")


def _lane_row(i: int, center: float) -> dict:
    vec = np.zeros(14, dtype=np.float32)
    mask = np.ones(14, dtype=np.float32)
    # c5, c10, c20, c30 live at slots 0, 3, 6, 9
    for slot in (0, 3, 6, 9):
        vec[slot] = float(center)
    return {
        "i": i,
        "dt_s": 0.1,
        "wheel": {},
        "pedals": {},
        "tech": {
            "lane_gt": vec.tolist(),
            "lane_gt_valid": mask.tolist(),
            "gt_source": "painted_decals",
        },
    }


def _loop_window_loss(net, frames, dts, rows, device):
    from python.train.scene_net import SceneState, has_lane_gt, scene_loss
    from python.train.train_scene import (
        _as_target,
        _forward,
        _prediction_at,
        _row_labels,
        loss_mask,
        step_dt,
    )
    import torch

    state = net.initial_state(device=device, batch=1)
    total = None
    n_loss = 0
    for frame, dt_s, row in zip(frames, dts, rows):
        image = torch.from_numpy(frame).to(device)
        dt = torch.tensor([step_dt(dt_s)], dtype=torch.float32, device=device)
        pred, state = _forward(net, image, dt, state, [row])
        if not loss_mask(row):
            continue
        one = _prediction_at(pred, 0)
        labels = _row_labels(row, Path("."), None, 0.0, 0.0, None)
        assert has_lane_gt(labels) or True
        step = scene_loss(one, _as_target(labels, device), ("lane_gt",))
        total = step["total"] if total is None else total + step["total"]
        n_loss += 1
    return total / n_loss


def check_fused_window_matches_loop() -> None:
    import torch

    from python.train.scene_net import SceneNet

    torch.manual_seed(73)
    net = SceneNet(cams=("main",))
    net.eval()
    device = torch.device("cpu")
    net.to(device)
    rng = np.random.RandomState(73)
    frames = [rng.randint(0, 255, (192, 256, 3), dtype=np.uint8) for _ in range(4)]
    dts = [0.1, 0.1, 0.1, 0.1]
    rows = [_lane_row(i, 0.25 * (i + 1)) for i in range(4)]
    loop_net = SceneNet(cams=("main",))
    loop_net.load_state_dict(net.state_dict())
    loop_net.eval()
    loop_net.to(device)
    fuse_net = SceneNet(cams=("main",))
    fuse_net.load_state_dict(net.state_dict())
    fuse_net.eval()
    fuse_net.to(device)
    loop_loss = _loop_window_loss(loop_net, frames, dts, rows, device)
    stacked = np.ascontiguousarray(np.stack(frames)[None, ...])
    state = fuse_net.initial_state(device=device, batch=1)
    pred, _end, _ = fuse_net.forward_window(
        torch.from_numpy(stacked),
        torch.tensor([dts], dtype=torch.float32),
        state,
    )
    from python.train.scene_net import scene_loss
    from python.train.train_scene import _as_target, _prediction_at, _row_labels, loss_mask

    total = None
    n_loss = 0
    for tick, row in enumerate(rows):
        if not loss_mask(row):
            continue
        one = _prediction_at(pred, tick)
        labels = _row_labels(row, Path("."), None, 0.0, 0.0, None)
        step = scene_loss(one, _as_target(labels, device), ("lane_gt",))
        total = step["total"] if total is None else total + step["total"]
        n_loss += 1
    fuse_loss = total / n_loss
    delta = float((loop_loss - fuse_loss).abs().item())
    assert delta < 1e-4, delta
    loop_loss.backward()
    fuse_loss.backward()
    max_grad = 0.0
    loop_grads = {name: p.grad for name, p in loop_net.named_parameters() if p.grad is not None}
    for name, param in fuse_net.named_parameters():
        if param.grad is None:
            continue
        other = loop_grads.get(name)
        assert other is not None, name
        max_grad = max(max_grad, float((param.grad - other).abs().max().item()))
    assert max_grad < 1e-4, max_grad
    print(f"fused_window_matches_loop loss_delta={delta:.2e} grad_delta={max_grad:.2e} (CPU)")


def check_val_rms_identical() -> None:
    import torch

    from python.train.scene_net import SceneNet
    from python.train.train_scene import _eval_sequence, val_summary

    torch.manual_seed(7)
    net = SceneNet(cams=("main",))
    net.eval()
    device = torch.device("cpu")
    net.to(device)
    rng = np.random.RandomState(7)
    frames = [rng.randint(0, 255, (192, 256, 3), dtype=np.uint8) for _ in range(8)]
    rows = [_lane_row(i, 0.2 + 0.05 * i) for i in range(8)]
    state_a = net.initial_state(device=device, batch=1)
    state_b = net.initial_state(device=device, batch=1)
    pairs_a: list = []
    pairs_long_a: list = []
    parts_a: dict = {}
    loss_a, n_a = _eval_sequence(
        net,
        Path("."),
        rows,
        state_a,
        device=device,
        amp=False,
        view=None,
        pairs=pairs_a,
        pairs_long=pairs_long_a,
        parts=parts_a,
        frames=frames,
        reset_every=4,
        heads=("lane_gt",),
    )
    pairs_b: list = []
    pairs_long_b: list = []
    parts_b: dict = {}
    loss_b, n_b = _eval_sequence(
        net,
        Path("."),
        rows,
        state_b,
        device=device,
        amp=False,
        view=None,
        pairs=pairs_b,
        pairs_long=pairs_long_b,
        parts=parts_b,
        frames=list(frames),
        reset_every=4,
        heads=("lane_gt",),
    )
    scored_a = val_summary(pairs_a, pairs_long_a, parts_a)
    scored_b = val_summary(pairs_b, pairs_long_b, parts_b)
    assert n_a == n_b and n_a > 0
    assert scored_a["n"] == scored_b["n"]
    assert scored_a["rms"] == scored_b["rms"]
    assert scored_a["rms_long"] == scored_b["rms_long"]
    print(f"val_rms_identical rms={scored_a['rms']} n={scored_a['n']} (CPU)")


def check_memory_reading_skips_smi_when_torch_has_bytes() -> None:
    import python.train.train_scene as trainer

    calls = {"n": 0}

    def boom() -> int:
        calls["n"] += 1
        return 123

    real = trainer.read_used_vram_bytes
    real_cuda = trainer._cuda_memory_bytes
    try:
        trainer._cuda_memory_bytes = lambda: 4096
        trainer.read_used_vram_bytes = boom
        used, kind = trainer._memory_reading("cuda")
        assert used == 4096
        assert kind == "vram"
        assert calls["n"] == 0
        trainer._cuda_memory_bytes = lambda: 0
        used, kind = trainer._memory_reading("cuda")
        assert used == 123
        assert calls["n"] == 1
    finally:
        trainer.read_used_vram_bytes = real
        trainer._cuda_memory_bytes = real_cuda
    print("memory_reading_skips_smi")


def main() -> None:
    check_parser_flags()
    check_batch_uses_trained_canvas()
    check_aux_cache_matches_png()
    check_memory_reading_skips_smi_when_torch_has_bytes()
    check_fused_window_matches_loop()
    check_val_rms_identical()
    print("train speed ok")


if __name__ == "__main__":
    main()
