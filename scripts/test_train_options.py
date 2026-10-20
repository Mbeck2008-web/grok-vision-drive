#!/usr/bin/env python3
"""Camera subset, head selection, photometric augment, and the frame cache."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402

from python.data.strip_writer import CANVAS_H, CANVAS_W, SECTORS, STRIP_ORDER  # noqa: E402
from python.train.cam_strip import (  # noqa: E402
    canvas_width,
    extract_cams,
    metadata_dict,
    parse_cams,
    parse_heads,
    photometric_jitter,
    prepare_model_strip,
)
from python.train.frame_cache import CHUNK_BYTES, FrameCache, build_frame_cache, frames_per_chunk  # noqa: E402
from python.train.continual import epoch_windows, valid_fraction, window_difficulty  # noqa: E402
from python.train.scene_loader import loader_settings  # noqa: E402
from python.train.scene_net import HEAD_NAMES, checkpoint_arch, head_names  # noqa: E402
from python.train.train_scene import build_parser  # noqa: E402


def check_cams_and_heads() -> None:
    assert parse_cams("") == tuple(STRIP_ORDER)
    assert parse_cams("main") == ("main",)
    assert parse_cams("rear,main") == ("main", "rear")
    assert canvas_width(("main",)) == 256
    assert canvas_width(STRIP_ORDER) == CANVAS_W == 2272
    try:
        parse_cams("nope")
        raise AssertionError("unknown camera should fail")
    except ValueError:
        pass
    assert parse_heads("") == HEAD_NAMES
    assert parse_heads("lane_gt") == ("lane_gt",)
    assert head_names(("lane_gt",)) == frozenset({"lane_gt"})
    assert head_names(None) == frozenset(HEAD_NAMES)
    assert "cams=" not in checkpoint_arch()
    assert checkpoint_arch(("main",)).endswith("cams=main")
    props = metadata_dict(("main",), ("lane_gt",), height=CANVAS_H, width=256)
    assert props["gvd_cams"] == "main" and props["gvd_canvas_w"] == "256"
    parser = build_parser()
    args = parser.parse_args(
        ["--cams", "main", "--heads", "lane_gt", "--augment", "--cache", r"C:\GVD_cache"]
    )
    assert args.lr == 1e-3
    assert args.cams == "main" and args.heads == "lane_gt" and args.augment is True
    assert parser.parse_args(["--no-augment"]).augment is False
    sched = parser.parse_args([])
    assert sched.lr == 1e-3
    assert sched.batch == 0
    wider = parser.parse_args(["--batch", "8"])
    assert wider.batch == 8 and wider.lr == 1e-3
    assert sched.clip == 1.0 and sched.lane_loss == "huber" and sched.min_valid == 0.3
    assert sched.balance is False and sched.scale == 1.0 and sched.workers is None
    assert loader_settings(4, cuda=True) == {
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": True,
    }
    assert loader_settings(0, cuda=True)["persistent_workers"] is False
    print("cams and heads")


def _row(index: int, *, valid: bool, steer: float = 0.0, phase: str = "cruise") -> dict:
    mask = [1.0] * 14 if valid else [0.0] * 14
    values = [0.2, -1.5, 1.9, 0.4, -1.3, 2.1] + [0.0] * 8
    return {
        "i": index,
        "t": float(index),
        "tech": {"lane_gt": values, "lane_gt_valid": mask, "heading_err_rad": 0.2 if phase != "cruise" else 0.0},
        "wheel": {"steer": steer},
        "excursion": {"phase": phase},
    }


def check_epoch_windows() -> None:
    straight = [_row(i, valid=True) for i in range(8)]
    curve = [_row(i, valid=True, steer=0.8, phase="weave") for i in range(8, 16)]
    empty = [_row(i, valid=False) for i in range(16, 24)]
    plan = [
        {"kind": "train", "rows": straight, "incoming": "carry", "stride": 4},
        {"kind": "train", "rows": curve, "incoming": "carry", "stride": 4},
        {"kind": "train", "rows": empty, "incoming": "carry", "stride": 4},
        {"kind": "holdout", "rows": straight, "incoming": "reset", "stride": 0},
    ]
    assert valid_fraction(straight) == 1.0
    assert valid_fraction(empty) == 0.0
    assert window_difficulty(curve) > window_difficulty(straight)
    out = epoch_windows(plan, __import__("random").Random(0), min_valid=0.3, balance=True)
    train = [item for item in out if item["kind"] == "train"]
    hold = [item for item in out if item["kind"] == "holdout"]
    assert hold and out[-1]["kind"] == "holdout"
    assert all(item["incoming"] == "reset" for item in train)
    assert all(item["rows"] is not empty for item in train)
    assert sum(1 for item in train if item["rows"][0]["i"] >= 8) >= 2
    print("epoch windows")
    plan_steps = [
        {"kind": "train", "rows": [1]},
        {"kind": "train", "rows": [2]},
        {"kind": "holdout", "rows": [3]},
    ]
    packed = iter([{"frames": ["a"]}, {"frames": ["b"]}])
    got = []
    for item in plan_steps:
        if item["kind"] == "holdout":
            got.append(None)
            continue
        got.append(next(packed)["frames"])
    try:
        next(packed)
        raise AssertionError("cache loader outlasted the plan")
    except StopIteration:
        pass
    assert got == [["a"], ["b"], None]
    print("prefetch covers every train window")


def check_extract_and_jitter() -> None:
    full = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    main = next(sec for sec in SECTORS if sec.cam_id == "main")
    full[:, main.x0 : main.x1] = (8, 16, 24)
    packed = extract_cams(full, ("main",))
    assert packed.shape == (CANVAS_H, 256, 3)
    assert int(packed[0, 0, 0]) == 8
    again = prepare_model_strip(full, ("main",), 256)
    assert again is not None and again.shape[1] == 256
    same = prepare_model_strip(packed, ("main",), 256)
    assert same is not None and same.shape == packed.shape
    jitter = photometric_jitter(packed, np.random.RandomState(1))
    assert jitter.shape == packed.shape and jitter.dtype == np.uint8
    assert not np.array_equal(jitter, packed)
    print("extract and jitter")


def check_cache_chunks() -> None:
    frame = CANVAS_H * 256 * 3
    assert frames_per_chunk(CANVAS_H, 256) * frame <= CHUNK_BYTES
    assert CHUNK_BYTES < 4 * 1024 ** 3
    colour = np.zeros((32, 40, 3), dtype=np.uint8)
    colour[:, :] = (12, 24, 36)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "strips"
        cam = root / "raw" / "main"
        cam.mkdir(parents=True)
        import cv2

        ok, buf = cv2.imencode(".png", colour)
        assert ok
        (cam / "000000.png").write_bytes(buf.tobytes())
        (cam / "000001.png").write_bytes(buf.tobytes())
        (root / "state.jsonl").write_text(
            "\n".join(json.dumps({"i": index, "t": float(index)}) for index in (0, 1)) + "\n",
            encoding="utf-8",
        )
        dest = Path(tmp) / "cache"
        manifest = build_frame_cache(root, dest, ("main",))
        assert manifest["cams"] == ["main"]
        assert manifest["canvas_w"] == 256
        chunk = dest / manifest["chunks"][0]["file"]
        assert chunk.stat().st_size < 4 * 1024 ** 3
        cache = FrameCache(dest)
        assert cache.has(0) and cache.has(1) and not cache.has(2)
        image = cache.canvas(0)
        assert image is not None and image.shape == (CANVAS_H, 256, 3)
        assert int(image[CANVAS_H // 2, 128, 0]) > 0
    print("cache chunks")


def check_subset_forward_probe_and_export() -> None:
    """A one-camera token is 64-d. The GRU hidden state stays 512."""
    from python.train.scene_net import GRU_HIDDEN, SceneNet, export_scene_onnx, torch_ready
    from python.train.train_scene import _probe_cuda

    if not torch_ready():
        skip_check("subset_forward", "torch missing")
        return
    import torch

    def exercise(cams: tuple[str, ...], scale: float = 1.0) -> None:
        net = SceneNet(cams=cams, scale=scale).cpu().train()
        assert net.token_dim == len(cams) * 64
        assert net.token_dim != GRU_HIDDEN or net.skip is None
        if net.token_dim != GRU_HIDDEN:
            assert net.skip is not None and net.skip.out_features == GRU_HIDDEN
        image = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
        state = net.initial_state(device="cpu", batch=1)
        pred, _state = net.forward_batch(image, torch.zeros(1), state)
        assert tuple(pred.lane_gt.shape)[-1] == 14
        assert _probe_cuda(net, 1, 2, False, torch.device("cpu")) is True

    exercise(("main",))
    exercise(("main", "narrow"))
    exercise(("main",), scale=0.5)
    full = SceneNet().cpu()
    assert full.skip is None and full.token_dim == GRU_HIDDEN

    cached = torch.zeros(full.canvas_h, 256, 3)
    main = SceneNet(cams=("main",)).cpu().eval()
    assert cached.shape == (main.canvas_h, main.canvas_w, 3)
    pred, _state = main.forward(cached, torch.zeros(1), main.initial_state(device="cpu"))
    assert tuple(pred.lane_gt.shape) == (14,)

    captured: dict[str, object] = {}

    def _trace(model: object, args: tuple, path: str) -> None:
        del model
        captured["shape"] = tuple(args[0].shape)
        Path(path).write_bytes(b"not-a-real-onnx")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "main.onnx"
        export_scene_onnx(dest, cams=("main",), heads=("lane_gt",), zeros=torch.zeros, trace=_trace)
        assert captured["shape"] == (1, 3, main.canvas_h, main.canvas_w)
        assert dest.is_file()
        try:
            import onnx
        except ImportError:
            onnx = None
        if onnx is not None:
            real = Path(tmp) / "main_real.onnx"
            export_scene_onnx(real, SceneNet(cams=("main",)).cpu().eval(), heads=("lane_gt",))
            model = onnx.load(real)
            onnx.checker.check_model(model)
            meta = {item.key: item.value for item in model.metadata_props}
            assert meta["gvd_cams"] == "main"
            strip = model.graph.input[0]
            dims = [dim.dim_value for dim in strip.type.tensor_type.shape.dim]
            assert dims == [1, 3, main.canvas_h, main.canvas_w]
    print("subset forward and export shape")


def check_spawn_loader_reads_cache() -> None:
    """Windows starts DataLoader workers with spawn. A lambda collate does not pickle."""
    import multiprocessing as mp
    import pickle

    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.scene_net import torch_ready
    from python.train.train_scene import (
        _EpochWindows,
        _first_item,
        _prefetch_loader,
        _release_worker_handles,
    )

    if not torch_ready():
        skip_check("spawn_loader", "torch missing")
        return
    import cv2
    from torch.utils.data import DataLoader

    colour = np.zeros((16, 16, 3), dtype=np.uint8)
    colour[:, :] = (40, 80, 120)
    ok, encoded = cv2.imencode(".png", colour)
    assert ok
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "scene"
        cam = root / "raw" / "main"
        cam.mkdir(parents=True)
        lines = []
        for index in range(4):
            (cam / f"{index:06d}.png").write_bytes(encoded.tobytes())
            lines.append({"i": index, "t": float(index), "cams": {"main": {"file": f"raw/main/{index:06d}.png"}}})
        (root / "state.jsonl").write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
        cache_dir = Path(tmp) / "cache"
        build_frame_cache(root, cache_dir, ("main",))
        cache = FrameCache(cache_dir)
        items = [
            {"kind": "train", "rows": [lines[0], lines[1]], "stride": 2, "incoming": "carry"},
            {"kind": "train", "rows": [lines[2], lines[3]], "stride": 2, "incoming": "carry"},
            {"kind": "holdout", "rows": [lines[0]], "stride": 0},
        ]
        dataset = _EpochWindows(items[:2], root, ("main",), str(cache_dir), 1.0)
        pickle.dumps(_first_item)
        pickle.dumps(dataset)
        pickle.dumps(cache)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            collate_fn=_first_item,
            worker_init_fn=_release_worker_handles,
            multiprocessing_context=mp.get_context("spawn"),
        )
        seen = []
        for batch in loader:
            assert batch["incoming"] == "reset"
            assert len(batch["frames"]) == 2
            assert batch["frames"][0].shape == (192, 256, 3)
            seen.append(batch["rows"][0]["i"])
        assert sorted(seen) == [0, 2]
        del loader
        wrapped = _prefetch_loader(items, root, ("main",), cache, 1.0, workers=2, cuda=False)
        assert wrapped is not None and wrapped.collate_fn is _first_item
        assert len(wrapped.dataset) == 2
        del wrapped
    print("spawn loader")


def check_sanity_overfits_cached_main() -> None:
    """Sanity repeats one main-camera batch until lane_gt is under 0.05."""
    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.scene_net import torch_ready
    from python.train.train_scene import train_directory

    if not torch_ready():
        skip_check("sanity_overfit", "torch missing")
        return
    import cv2

    colour = np.zeros((24, 32, 3), dtype=np.uint8)
    colour[:, :] = (30, 60, 90)
    ok, encoded = cv2.imencode(".png", colour)
    assert ok
    lane = [0.25, -1.5, 2.0, 0.25, -1.5, 2.0, 0.25, -1.5, 2.0, 0.25, -1.5, 2.0, 0.0, 1.0]
    mask = [1.0] * 13 + [1.0]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "scene"
        cam = root / "raw" / "main"
        cam.mkdir(parents=True)
        state_lines = []
        gt_lines = []
        for index in range(8):
            (cam / f"{index:06d}.png").write_bytes(encoded.tobytes())
            state_lines.append(
                {
                    "i": index,
                    "t": float(index) * 0.1,
                    "dt_s": None if index == 0 else 0.1,
                    "segment": "west_coast_usa#0",
                    "cams": {"main": {"file": f"raw/main/{index:06d}.png"}},
                    "wheel": {"steer": 0.1},
                    "pedals": {"throttle": 0.2, "brake": 0.0},
                }
            )
            gt_lines.append(
                {
                    "i": index,
                    "t": float(index) * 0.1,
                    "tech": {"lane_gt": lane, "lane_gt_valid": mask, "heading_err_rad": 0.0},
                }
            )
        (root / "state.jsonl").write_text(
            "\n".join(json.dumps(row) for row in state_lines) + "\n", encoding="utf-8"
        )
        (root / "gt.jsonl").write_text(
            "\n".join(json.dumps(row) for row in gt_lines) + "\n", encoding="utf-8"
        )
        cache_dir = Path(tmp) / "cache"
        build_frame_cache(root, cache_dir, ("main",))
        cache = FrameCache(cache_dir)
        assert cache.has(0)
        image = cache.canvas(0)
        assert image is not None and image.shape == (192, 256, 3)
        stats = train_directory(
            root,
            epochs=1,
            lr=1e-3,
            capability=None,
            export_path=None,
            free_vram_bytes=0,
            free_ram_bytes=8_000_000_000,
            resume=False,
            sanity=True,
            dry_run=False,
            open_panel=False,
            autostart=True,
            cams=("main",),
            heads=("lane_gt",),
            cache=cache,
            augment=True,
            workers=0,
        )
        assert stats.get("sanity_ok") is True, stats
    print("sanity overfit")


def _bar(x: int, height: int = 192, width: int = 256) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    col = int(np.clip(x, 2, width - 3))
    frame[:, col - 2 : col + 3] = 255
    return frame


def _offset_row(index: int, offset: float) -> dict:
    values = [0.0] * 14
    mask = [0.0] * 14
    values[0] = float(offset)
    values[-1] = 1.0
    mask[0] = 1.0
    mask[-1] = 1.0
    return {
        "i": index,
        "t": float(index) * 0.1,
        "dt_s": 0.1,
        "wheel": {"steer": 0.0},
        "pedals": {"throttle": 0.2, "brake": 0.0},
        "tech": {"lane_gt": values, "lane_gt_valid": mask},
    }


def check_lane_position_learns() -> None:
    """64 distinct lane positions overfit, and a disjoint set still correlates.

    Global average pooling made every horizontal shift the same vector, so
    this fit stayed near r=0. The stem now keeps eight horizontal bins.
    """
    from python.debug.lane_targets import score_centers
    from python.train.continual import WEIGHT_DECAY
    from python.train.scene_net import HORIZ_BINS, SceneNet, scene_loss, torch_ready
    from python.train.train_scene import _optimizer_group, resolve_scene_batch

    assert resolve_scene_batch(0, 8) == 8
    assert resolve_scene_batch(4, 8) == 4
    assert resolve_scene_batch(0, 0) == 1
    if not torch_ready():
        skip_check("lane_position", "torch missing")
        return
    import torch
    import torch.nn.functional as functional

    np.random.seed(0)
    torch.manual_seed(0)
    width = 256
    net = SceneNet(cams=("main",))
    # Dropout stays on in the real trainer. It is off here so a short CPU
    # fit of 64 images is not a coin flip.
    net.gru_in_drop.p = 0.0
    left = torch.from_numpy(_bar(40)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    right = torch.from_numpy(_bar(200)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        pooled = []
        for image in (left, right):
            horiz = net.stem.conv(image).mean(dim=2)
            pooled.append(functional.adaptive_avg_pool1d(horiz, HORIZ_BINS).flatten())
    stem_cos = float(functional.cosine_similarity(pooled[0], pooled[1], dim=0))
    assert stem_cos < 0.95, stem_cos

    train_x = [20 + i * 3 for i in range(64)]
    hold_x = [21 + i * 12 for i in range(16)]
    assert len(set(train_x)) == 64
    assert set(hold_x).isdisjoint(train_x)

    def pack(xs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        images = np.stack([_bar(x) for x in xs])
        values = np.zeros((len(xs), 14), np.float32)
        mask = np.zeros((len(xs), 14), np.float32)
        offsets: list[float] = []
        for i, x in enumerate(xs):
            offset = (float(x) - width / 2.0) / (width / 4.0)
            values[i, 0] = offset
            values[i, -1] = 1.0
            mask[i, 0] = 1.0
            mask[i, -1] = 1.0
            offsets.append(offset)
        return images, values, mask, offsets

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    net.train()
    order = np.arange(64)
    last = 1.0
    for _epoch in range(20):
        np.random.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, 64, 8):
            chosen = [train_x[int(i)] for i in order[start : start + 8]]
            images, values, mask, _offsets = pack(chosen)
            state = net.initial_state(batch=len(chosen))
            pred, _state = net.forward_batch(torch.from_numpy(images), 0.1, state)
            loss = scene_loss(
                pred,
                {"lane_gt": torch.from_numpy(values), "lane_gt_valid": torch.from_numpy(mask)},
                ("lane_gt",),
                lane_loss="huber",
            )["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        last = total / steps
        if last < 0.05 and _epoch >= 3:
            break
    assert last < 0.05, last

    def predict(xs: list[int]) -> tuple[list[float], list[float]]:
        net.eval()
        images, _values, _mask, offsets = pack(xs)
        with torch.no_grad():
            state = net.initial_state(batch=len(xs))
            pred, _state = net.forward_batch(torch.from_numpy(images), 0.1, state)
            centers = [float(v) for v in pred.lane_gt[:, 0].detach()]
        return centers, offsets

    train_pred, train_true = predict(train_x)
    hold_pred, hold_true = predict(hold_x)
    train_score = score_centers(train_pred, train_true)
    hold_score = score_centers(hold_pred, hold_true)
    assert float(train_score["r"]) > 0.9, train_score
    assert float(hold_score["r"]) > 0.5, hold_score
    grad_net = SceneNet(cams=("main",))
    grad_net.train()
    images, values, mask, _offsets = pack(train_x[:8])
    state = grad_net.initial_state(batch=8)
    pred, _state = grad_net.forward_batch(torch.from_numpy(images), 0.1, state)
    loss = scene_loss(
        pred,
        {"lane_gt": torch.from_numpy(values), "lane_gt_valid": torch.from_numpy(mask)},
        ("lane_gt",),
    )["total"]
    loss.backward()
    stem_grad = float(grad_net.stem.conv[0].weight.grad.abs().sum())
    assert stem_grad > 0.0, stem_grad
    print(
        f"lane position loss={last:.4f} r={train_score['r']:.3f} "
        f"hold_r={hold_score['r']:.3f} stem_cos={stem_cos:.3f} stem_grad={stem_grad:.3f}"
    )

    # The trainer step has to see every window in the stack, not only index 0.
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    net.train()
    sequences = []
    frames = []
    targets = [1.2, 0.4, -0.4, -1.2]
    columns = [40, 90, 160, 210]
    for index, (offset, column) in enumerate(zip(targets, columns)):
        sequences.append([_offset_row(index, offset)])
        frames.append([_bar(column)])
    group_loss = None
    for _step in range(8):
        group_loss, n_sup, _carry, _end = _optimizer_group(
            net,
            opt,
            None,
            Path("."),
            sequences,
            device=torch.device("cpu"),
            amp=False,
            view=None,
            clip=1.0,
            heads=("lane_gt",),
            lane_loss="huber",
            frame_sets=frames,
        )
        assert n_sup == 4
    assert group_loss is not None and group_loss < 0.2, group_loss
    net.eval()
    with torch.no_grad():
        state = net.initial_state(batch=4)
        pred, _state = net.forward_batch(torch.from_numpy(np.stack([_bar(c) for c in columns])), 0.1, state)
        got = [float(v) for v in pred.lane_gt[:, 0].detach()]
    assert score_centers(got, targets)["r"] > 0.9, got
    print(f"group step loss={group_loss:.4f} r={score_centers(got, targets)['r']:.3f}")


def main() -> None:
    check_cams_and_heads()
    check_extract_and_jitter()
    check_cache_chunks()
    check_epoch_windows()
    check_subset_forward_probe_and_export()
    check_spawn_loader_reads_cache()
    check_sanity_overfits_cached_main()
    check_lane_position_learns()
    print("train options ok")


if __name__ == "__main__":
    main()
