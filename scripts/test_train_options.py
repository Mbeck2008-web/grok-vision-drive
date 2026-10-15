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
        print("subset forward skipped (torch missing)")
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


def main() -> None:
    check_cams_and_heads()
    check_extract_and_jitter()
    check_cache_chunks()
    check_epoch_windows()
    check_subset_forward_probe_and_export()
    print("train options ok")


if __name__ == "__main__":
    main()
