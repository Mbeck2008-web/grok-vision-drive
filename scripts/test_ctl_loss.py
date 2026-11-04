#!/usr/bin/env python3
"""B'2 control-head loss (``--ctl-loss``). Offline only.

* L1 loss: ``l1`` is the masked mean |pred - target| per channel on steer and
  throttle (L1, the same error the eval's MAE measures; per-row masked mean
  over horizons), summed; brake keeps the masked-mean smooth-L1 (beta 0.1).
  Masked slots do not pull (99 or NaN goals), an all-void mask is 0, an unknown
  kind raises.
* L2 default: ``huber`` (the default) is the old smooth-L1 on all three
  channels, and ``scene_loss`` without ``ctl_loss`` gives the same total as before.
* L3 CLI: ``--ctl-loss`` defaults to ``huber``, ``l1`` is refused without the
  ctl head (code 2, nothing written), and a tiny ``--ctl-cond --ctl-loss l1``
  seed-73 run writes ``ctl_loss`` into the checkpoint and ``gvd_ctl_loss`` into
  its ONNX.
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

import numpy as np  # noqa: E402


def _case():
    import torch

    torch.manual_seed(73)
    pred = torch.rand(4, 3, 6) * 0.8 - 0.2
    goal = torch.rand(4, 3, 6) * 0.8 - 0.2
    valid = (torch.rand(4, 3, 6) > 0.4).float()
    return pred, goal, valid


def _want(pred, goal, valid, l1_channels) -> float:
    want = 0.0
    for c in range(3):
        m = valid[:, c].numpy() > 0.5
        if not m.any():
            continue
        d = np.abs(pred[:, c].numpy() - goal[:, c].numpy())[m]
        per = d if c in l1_channels else np.where(d < 0.1, 0.5 * d * d / 0.1, d - 0.05)
        want += float(per.mean())
    return want


def check_l1_loss() -> None:
    import torch

    from python.train.scene_net import _ctl_loss

    pred, goal, valid = _case()
    got = float(_ctl_loss(pred, goal, valid, kind="l1"))
    want = _want(pred, goal, valid, (0, 1))
    assert abs(got - want) < 1e-6, (got, want)
    # Steer / throttle alone: exactly mean |pred - target| over the valid slots.
    only = valid.clone()
    only[:, 1:, :] = 0.0
    mae = float((pred[:, 0] - goal[:, 0]).abs()[valid[:, 0] > 0.5].mean())
    assert abs(float(_ctl_loss(pred, goal, only, kind="l1")) - mae) < 1e-6
    for fill in (99.0, float("nan")):
        moved = goal.clone()
        moved[valid < 0.5] = fill
        assert abs(float(_ctl_loss(pred, moved, valid, kind="l1")) - got) < 1e-6, f"masked slots must not pull ({fill})"
    assert float(_ctl_loss(pred, goal, torch.zeros_like(valid), kind="l1")) == 0.0
    try:
        _ctl_loss(pred, goal, valid, kind="mse")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown ctl loss must raise")
    print(f"L1 loss: l1 = masked mean |err| on steer/throttle + smooth-L1 brake ({got:.6f}), masked inert OK")


def check_l2_default() -> None:
    import torch

    from python.train.scene_net import SceneNet, _ctl_loss, labels_for_row, scene_loss
    from test_ctl_head import _row

    pred, goal, valid = _case()
    old = _want(pred, goal, valid, ())
    assert abs(float(_ctl_loss(pred, goal, valid)) - old) < 1e-6
    assert abs(float(_ctl_loss(pred, goal, valid, kind="huber")) - old) < 1e-6
    net = SceneNet(cams=("main",), ctl=True).eval()
    strip = torch.rand(1, 3, net.canvas_h, net.canvas_w)
    with torch.no_grad():
        p, _ = net.forward(strip, 0.1, net.initial_state())
    row = _row(0, 0.0)
    row["tech"] = {}
    row["ctl"] = np.array([[0.3] * 6, [0.9] * 6, [0.2] * 6], np.float32)
    row["ctl_valid"] = np.ones((3, 6), np.float32)
    target = {k: torch.as_tensor(v) for k, v in labels_for_row(row).items()}
    for key in ("object_class", "sign_class", "sign_state"):
        target[key] = target[key].long()
    heads = ("lane_gt", "ctl")
    default = scene_loss(p, target, heads)
    huber = scene_loss(p, target, heads, ctl_loss="huber")
    l1 = scene_loss(p, target, heads, ctl_loss="l1", ctl_weight=2.0)
    assert float(default["total"]) == float(huber["total"]) and float(default["ctl"]) == float(huber["ctl"])
    assert float(l1["ctl"]) != float(default["ctl"])
    assert abs(float(l1["total"]) - float(l1["lane_gt"]) - 2.0 * float(l1["ctl"])) < 1e-5
    print("L2 default: huber is the old smooth-L1 on all three channels; scene_loss default unchanged OK")


def check_l3_cli() -> None:
    import onnx
    import torch

    import export_epoch_onnx
    from python.train import train_scene
    from python.train.epoch_ckpt import load_epoch_ckpt
    from test_ctl_head import _fixture, _listing, _main_code

    args = train_scene.build_parser().parse_args(["--strips", "x"])
    assert args.ctl_loss == "huber", "--ctl-loss defaults to huber"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        strips = tmp / "rec"
        strips.mkdir()
        before = _listing(tmp)
        code, text = _main_code(["--strips", str(strips), "--no-panel", "--autostart", "--dry-run",
                                 "--heads", "lane_gt", "--ctl-loss", "l1"])
        assert code == 2 and "--ctl-loss l1 needs --heads lane_gt,ctl" in text, (code, text[-2000:])
        assert _listing(tmp) == before

    torch.manual_seed(73)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "data" / "rec"
        _fixture(root)
        ckpts = tmp / "run" / "ckpts"
        code, text = _main_code([
            "--strips", str(root), "--cams", "main", "--heads", "lane_gt,ctl", "--ctl-cond", "--ctl-loss", "l1",
            "--seed", "73", "--exact-epochs", "1", "--no-panel", "--autostart", "--workers", "0",
            "--min-valid", "0", "--holdout-blocks", "1", "--ckpt-dir", str(ckpts),
            "--pause-dir", str(tmp / "run" / "pause"),
        ])
        assert code == 0, (code, text[-3000:])
        assert "[GVD] ctl loss: steer/throttle=l1 brake=huber" in text, text[-3000:]
        ep0 = sorted(ckpts.glob("*.ckpt_ep0_step*.npz"))
        assert len(ep0) == 1
        meta = load_epoch_ckpt(ep0[0])["meta"]
        assert meta["ctl_loss"] == "l1" and meta["ctl_cond_dim"] == 5, meta
        out = tmp / "ep0.onnx"
        assert export_epoch_onnx.main(["--ckpt-dir", str(ckpts), "--epoch", "0", "--out", str(out)]) == 0
        props = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
        assert props.get("gvd_ctl_loss") == "l1" and props.get("gvd_ctl_cond") == "5", props
    print("L3 CLI: --ctl-loss default huber, l1 refused without ctl, --ctl-cond --ctl-loss l1 seed 73 run -> "
          "ckpt ctl_loss=l1, ONNX gvd_ctl_loss=l1 OK")


def main() -> None:
    check_l1_loss()
    check_l2_default()
    check_l3_cli()
    print("ctl loss: L1-L3 OK")


if __name__ == "__main__":
    main()
