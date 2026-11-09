#!/usr/bin/env python3
"""Refuse a multi-camera scene-ckpt: this shadow path only has the main frame.

_main_canvas composes {"main": frame} then extract_cams for whatever cams the
checkpoint lists. Missing sectors stay black. Engage is already off; drawing
a lane from black side cameras is still the wrong image.

    py -3 scripts/test_scene_ckpt_multicam.py
"""
# gvd-ci: requires torch
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_multicam_refused_at_load() -> None:
    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import SceneCkptInputError, SceneCkptRuntime
    from python.train.scene_net import SceneNet

    torch.manual_seed(0)
    net = SceneNet(cams=("main", "wide")).cpu().eval()
    meta = {
        "epoch": 0,
        "step": 1,
        "cams": ["main", "wide"],
        "scale": 1.0,
        "backbone": "scratch",
        "heads": ["lane_gt"],
        "cam_age": False,
        "ctl": False,
        "arch": getattr(net, "pause_arch", None),
    }
    weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = epoch_ckpt_path(Path(tmp), Path(tmp) / "scene_lanes", "20261107T011144Z", 0, 1)
        write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
        try:
            SceneCkptRuntime(path)
        except SceneCkptInputError as exc:
            msg = str(exc).lower()
            assert "main" in msg and ("black" in msg or "camera" in msg or "cams" in msg), exc
            print(f"multi-cam ckpt refused at load: {exc}")
        else:
            raise AssertionError("eight/multi-camera checkpoint loaded; side cameras would be black")


def check_main_only_still_loads() -> None:
    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet

    torch.manual_seed(1)
    net = SceneNet(cams=("main",)).cpu().eval()
    meta = {
        "epoch": 0,
        "step": 1,
        "cams": ["main"],
        "scale": 1.0,
        "backbone": "scratch",
        "heads": ["lane_gt"],
        "cam_age": False,
        "ctl": False,
        "arch": getattr(net, "pause_arch", None),
    }
    weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = epoch_ckpt_path(Path(tmp), Path(tmp) / "scene_lanes", "20261107T011144Z", 0, 1)
        write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
        rt = SceneCkptRuntime(path)
        assert rt.cams == ("main",) or rt.cams == ("main",)
        print("main-only ckpt still loads OK")


def main() -> None:
    check_multicam_refused_at_load()
    check_main_only_still_loads()
    print("test_scene_ckpt_multicam: OK")


if __name__ == "__main__":
    main()
