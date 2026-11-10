#!/usr/bin/env python3
"""--route-cond model input, old-ckpt load, cmd-flip, trainer plumbing.

Not --ctl-cond. Offline. The net checks skip when PyTorch is missing.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402


def check_encoding_is_not_ctl_cond() -> None:
    from python.train.route_cond import ROUTE_CMDS, ROUTE_COND_DIM, encode_route, vec_from_nav

    assert ROUTE_CMDS == ("follow", "left", "right", "straight")
    assert ROUTE_COND_DIM == 5
    follow = encode_route("follow", None)
    left = encode_route("left", 30.0)
    mask = encode_route("mask", 12.0)
    assert follow.shape == (5,) and follow.dtype == np.float32
    assert follow.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert left.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert abs(float(left[4]) - 30.0 / 150.0) < 1e-6
    assert mask.tolist()[:4] == [1.0, 0.0, 0.0, 0.0], "mask encodes as follow"
    nav_left = vec_from_nav({"turn": "left", "next_jct_m": 12.0})
    assert nav_left.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    none = vec_from_nav({"turn": "none", "next_jct_m": None})
    assert none.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    missing = vec_from_nav(None)
    assert missing.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    src = (ROOT / "python/train/route_cond.py").read_text(encoding="utf-8")
    for needle in ("steer_ai", "throttle_ai", "brake_ai", "speed_mps", "ctl_cond"):
        assert needle not in src, needle
    print("  encode: 5-vector one-hot + distance; none/mask/missing -> follow; no ctl GT")


def check_runtime_uses_route_intent_schema() -> None:
    from python.train.route_cond import runtime_route_vec
    from python.sensors.route_intent import result_message

    vec = runtime_route_vec({}, result_message("right", 9.0))
    assert vec.tolist()[:4] == [0.0, 0.0, 1.0, 0.0]
    fallback = runtime_route_vec({}, None)
    assert fallback.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    print("  runtime_route_vec maps #203 turn/right to the model slot; missing -> follow")


def check_balance_and_sidecar() -> None:
    from python.train.continual import is_rare_row, oversample_repeats
    from python.train.route_cond import attach_route_cmds, per_command_metrics

    assert is_rare_row({"route_cmd": "left"})
    assert is_rare_row({"route_cmd": "straight"})
    assert not is_rare_row({"route_cmd": "follow"})
    assert not is_rare_row({"route_cmd": "mask"})
    assert oversample_repeats([{"route_cmd": "right"}]) > 1
    assert oversample_repeats([{"route_cmd": "follow"}]) == 1

    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "intent.jsonl").write_text(
            json.dumps({"t": 1.0, "cmd": "left", "next_jct_m": 20.0, "src": "hindsight"}) + "\n",
            encoding="utf-8",
        )
        rows = [{"t": 1.0, "source_root": str(root)}, {"t": 2.0, "source_root": str(root)}]
        counts = attach_route_cmds(rows)
        assert rows[0]["route_cmd"] == "left" and rows[1]["route_cmd"] == "follow"
        assert counts["left"] == 1 and counts["missing"] == 1
    grouped = per_command_metrics(
        [{"cmd": "left", "err": 2.0}, {"cmd": "left", "err": 4.0}, {"cmd": "follow", "err": 1.0}]
    )
    assert grouped["left"]["n"] == 2 and abs(grouped["left"]["mean_err"] - 3.0) < 1e-6
    assert grouped["follow"]["n"] == 1
    print("  junction cmds are rare for --balance; sidecar attach + per-command metrics")


def check_trainer_flag() -> None:
    # Do not import train_scene here: it pulls cv2 via status_window, which the
    # isolated suite hides. The plumbing is asserted from source instead.
    src = (ROOT / "python/train/train_scene.py").read_text(encoding="utf-8")
    assert '"--route-cond"' in src
    assert "This is not --ctl-cond" in src
    assert "action=\"store_true\"" in src[src.index("--route-cond") : src.index("--route-cond") + 400]
    assert "route_cond=bool(args.route_cond)" in src
    assert "route_cond=bool(route_cond)" in src
    assert "def row_route(" in src
    assert "encode_route" in src
    print("  trainer: --route-cond default off, help denies ctl-cond, flag is plumbed")


def check_run_vision_wires_under_flag() -> None:
    text = (ROOT / "python/run_vision.py").read_text(encoding="utf-8")
    assert "runtime_route_vec" in text or "vec_from_nav" in text
    assert "wants_route" in text
    tree = ast.parse(text, filename="run_vision.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("route_hindsight"):
            raise AssertionError("run_vision must not import the hindsight labeler")
    print("  run_vision feeds runtime_route_vec only when the model wants_route")


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def check_model_flag_shapes_ckpt_and_flip() -> None:
    torch = _torch()
    if torch is None:
        skip_check("scene_net_route_cond", "PyTorch is not installed")
        return
    from python.train.route_cond import encode_route
    from python.train.scene_net import SceneNet

    plain = SceneNet(cams=("main",))
    other = SceneNet(cams=("main",))
    assert getattr(plain, "route_cond", False) is False
    assert not hasattr(plain, "route_proj")
    other.load_state_dict(plain.state_dict())
    keyed = set(plain.state_dict())
    cond = SceneNet(cams=("main",), route_cond=True)
    assert cond.route_cond is True
    extra = set(cond.state_dict()) - keyed
    assert extra and all(name.startswith("route_proj") for name in extra)
    assert keyed == set(other.state_dict())
    missing, unexpected = cond.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert any(name.startswith("route_proj") for name in missing)
    arch = getattr(cond, "pause_arch", "") or ""
    assert ":route_cond=5" in arch

    h, w = int(cond.canvas_h), int(cond.canvas_w)
    strip = torch.zeros(1, 3, h, w)
    left = torch.from_numpy(encode_route("left", 25.0)).unsqueeze(0)
    right = torch.from_numpy(encode_route("right", 25.0)).unsqueeze(0)
    pred0, _ = plain.forward(strip, 0.1, plain.initial_state())
    pred_l, _ = cond.forward(strip, 0.1, cond.initial_state(), route=left)
    assert tuple(pred_l.lane_gt.shape) == tuple(pred0.lane_gt.shape)
    with torch.no_grad():
        cond.route_proj.weight.normal_(0.0, 1.0)
        cond.route_proj.bias.zero_()
    pred_l, _ = cond.forward(strip, 0.1, cond.initial_state(), route=left)
    pred_r, _ = cond.forward(strip, 0.1, cond.initial_state(), route=right)
    assert not torch.allclose(pred_l.lane_gt, pred_r.lane_gt), "flipping cmd must change lane_gt"
    print("  SceneNet: default unchanged, old ckpt loads, shapes match, left/right flip changes path")


def main() -> int:
    check_encoding_is_not_ctl_cond()
    check_runtime_uses_route_intent_schema()
    check_balance_and_sidecar()
    check_trainer_flag()
    check_run_vision_wires_under_flag()
    check_model_flag_shapes_ckpt_and_flip()
    print("test_route_cond: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
