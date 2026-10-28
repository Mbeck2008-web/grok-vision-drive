#!/usr/bin/env python3
"""GVD TRAIN panel (spec v3, PR-A).

* params: the panel's params row is the trainable sum of the live net it was
  given (``set_net``), for the default net, a ``cam_age=True`` net and a
  one-camera net.
"""
# gvd-ci: requires torch
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _params_row(window) -> str:
    rows = [line for line in window.lines.splitlines() if line.startswith("params")]
    assert len(rows) == 1, window.lines
    return rows[0]


def check_params_live_net() -> None:
    import torch

    from python.train.scene_net import SceneNet
    from python.train.status_window import TrainWindow

    torch.manual_seed(0)
    configs = (
        ("cam_age=True", lambda: SceneNet(cam_age=True)),
        ("default", lambda: SceneNet()),
        ("cams=main", lambda: SceneNet(cams=("main",))),
    )
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "strips"
        empty.mkdir()
        for label, build in configs:
            net = build()
            want = sum(p.numel() for p in net.parameters() if p.requires_grad)
            window = TrainWindow()
            if hasattr(window, "set_net"):
                window.set_net(net)
            window.load_folder(empty)
            row = _params_row(window)
            assert f"({want})" in row, f"{label}: panel shows {row!r}, live net sum is {want}"
            assert "approximate" not in row, row
            print(f"params {label}: {row} == sum(p.numel()) {want} OK")


def main() -> None:
    check_params_live_net()
    print("test_train_panel: all OK")


if __name__ == "__main__":
    main()
