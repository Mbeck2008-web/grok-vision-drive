#!/usr/bin/env python3
"""A multi-camera scene-ckpt loads: --scene-ckpt takes the live 8-camera frame dict.

Before Trainer 2.0 PR2b (#188) the shadow path only had the main frame, so any
checkpoint listing more than ``main`` was refused at load (side sectors would
have been black). ``_main_canvas`` now composes the whole live dict, so those
checkpoints load; scripts/test_scene_ckpt_8cam.py pins what they are fed.

    py -3 scripts/test_scene_ckpt_multicam.py
"""
# gvd-ci: requires torch
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_multicam_loads() -> None:
    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import SceneCkptRuntime
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
        rt = SceneCkptRuntime(path)
        assert tuple(rt.cams) == ("wide", "main"), rt.cams
        assert rt.canvas_h == 192 and rt.canvas_w > 256, (rt.canvas_h, rt.canvas_w)
        print(f"multi-cam ckpt loads: cams={list(rt.cams)} canvas {rt.canvas_h}x{rt.canvas_w}")


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
    check_multicam_loads()
    check_main_only_still_loads()
    print("test_scene_ckpt_multicam: OK")


if __name__ == "__main__":
    main()
