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
    assert float(mask[4]) == 0.0, "masked frames must not keep a near-junction distance"
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


def check_train_runtime_encoding_contract() -> None:
    """Runtime must emit the same 5-vector the labeler trains on."""
    from python.sensors.route_intent import AMBIGUOUS_HI, AMBIGUOUS_LO, APPROACH_M, LOOKAHEAD_M
    from python.collect import route_hindsight as rh
    from python.train.route_cond import cmd_from_nav, encode_route, runtime_route_vec, vec_from_nav

    assert rh.APPROACH_M == APPROACH_M == 60.0
    assert rh.LOOKAHEAD_M == LOOKAHEAD_M == 80.0
    assert rh.AMBIGUOUS_LO == AMBIGUOUS_LO == 16.0
    assert rh.AMBIGUOUS_HI == AMBIGUOUS_HI == 24.0

    left_near = vec_from_nav({"turn": "left", "next_jct_m": 25.0})
    assert left_near.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert abs(float(left_near[4]) - 25.0 / 150.0) < 1e-6

    # Training never emits a turn one-hot past APPROACH_M (60 m). A 190 m left
    # would encode as dist 1.267 today; it must become follow like the labeler.
    cmd, dist = cmd_from_nav({"turn": "left", "next_jct_m": 190.0})
    assert cmd == "follow", (cmd, dist)
    far = encode_route(cmd, dist)
    assert far.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    # Past LOOKAHEAD_M the labeler never saw a junction, so dist is missing (1.0).
    assert abs(float(far[4]) - 1.0) < 1e-6

    mid = vec_from_nav({"turn": "left", "next_jct_m": 70.0})
    assert mid.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    assert abs(float(mid[4]) - 70.0 / 150.0) < 1e-6

    assert runtime_route_vec({}, {"turn": "left", "next_jct_m": 190.0}).tolist()[:4] == [
        1.0, 0.0, 0.0, 0.0
    ]
    print("  encoding: turn cmds only <=60 m; >80 m follow+missing; shared thresholds")


def check_left_25m_reaches_model_tensor() -> None:
    """A live #203 left at 25 m must reach the model as left with dist 25/150."""
    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic_roads import load_graph
    from python.sensors.route_intent import Graph, intent_from_pose
    from python.train.route_cond import runtime_route_vec

    graph = Graph.from_fixture(load_graph(), "synth_a")
    result = intent_from_pose(graph, (0.0, -25.0), goal="W")
    assert result["turn"] == "left", result
    assert result["next_jct_m"] is not None and abs(float(result["next_jct_m"]) - 25.0) < 0.51, result
    vec = runtime_route_vec({}, result)
    assert vec.tolist()[:4] == [0.0, 1.0, 0.0, 0.0], vec
    assert abs(float(vec[4]) - 25.0 / 150.0) < 1e-5, vec
    wiped = runtime_route_vec({"turn": "left", "next_jct_m": 25.0}, None)
    assert wiped.tolist()[:4] == [1.0, 0.0, 0.0, 0.0], "None intent still falls back to follow"
    print("  live left at 25 m encodes as left with dist 25/150")


def check_labeler_runtime_vectors_match() -> None:
    """Same pose sequence: hindsight encode_route == runtime_route_vec(intent_from_pose)."""
    import math

    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic_roads import load_graph
    from python.collect.route_hindsight import label_sequence
    from python.sensors.route_intent import Graph, intent_from_pose
    from python.train.route_cond import encode_route, runtime_route_vec

    data = load_graph()
    graph = Graph.from_fixture(data, "synth_a")
    # Centerline (not the 1.75 m lane offset track()) so graph remaining
    # distance matches along-track metres.
    poses = []
    for y in range(-200, 1, 1):
        poses.append((0.0, float(y), math.pi / 2.0))
    for x in range(-1, -201, -1):
        poses.append((float(x), 0.0, math.pi))
    rows = []
    for i, (x, y, heading) in enumerate(poses):
        rows.append(
            {
                "t": round(i * 0.1, 4),
                "pos": [float(x), float(y), 0.0],
                "dir": [math.cos(heading), math.sin(heading), 0.0],
            }
        )
    labels = label_sequence(graph, rows)
    assert len(labels) == len(poses)
    mismatches = []
    compared = 0
    for row, lab, pose in zip(rows, labels, poses):
        # Masked hindsight rows are follow+0; runtime missing/none is follow+1.0.
        if lab["cmd"] == "mask":
            continue
        train_vec = encode_route(lab["cmd"], lab["next_jct_m"])
        result = intent_from_pose(graph, (pose[0], pose[1]), goal="W", heading=pose[2])
        live_vec = runtime_route_vec({}, result)
        compared += 1
        if not np.allclose(train_vec, live_vec, atol=5e-3):
            mismatches.append((row["t"], lab, result, train_vec.tolist(), live_vec.tolist()))
    assert compared >= 200, compared
    assert any(lab["cmd"] == "left" for lab in labels)
    assert not mismatches, mismatches[:5]
    print(f"  labeler vs runtime: {compared} identical non-mask vectors on S->T->W")


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


def _call_passes_literal_none(tree: ast.AST, func_name: str, arg_index: int = 1) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name != func_name:
            continue
        if len(node.args) > arg_index and isinstance(node.args[arg_index], ast.Constant):
            if node.args[arg_index].value is None:
                return True
        for kw in node.keywords:
            if kw.arg in {"intent_result", "result"} and isinstance(kw.value, ast.Constant):
                if kw.value.value is None:
                    return True
    return False


def check_run_vision_wires_under_flag() -> None:
    text = (ROOT / "python/run_vision.py").read_text(encoding="utf-8")
    assert "runtime_route_vec" in text or "vec_from_nav" in text
    assert "wants_route" in text
    assert "intent_from_pose" in text, "run_vision must compute #203 intent from pose + pin"
    tree = ast.parse(text, filename="run_vision.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("route_hindsight"):
            raise AssertionError("run_vision must not import the hindsight labeler")
    assert not _call_passes_literal_none(tree, "runtime_route_vec"), (
        "runtime_route_vec(..., None) wipes a live turn to follow"
    )
    assert not _call_passes_literal_none(tree, "merge_nav_intent"), (
        "merge_nav_intent(..., None) wipes nav turn before the next tick"
    )
    print("  run_vision feeds runtime_route_vec from intent_from_pose; no None wipe")


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
    check_train_runtime_encoding_contract()
    check_left_25m_reaches_model_tensor()
    check_labeler_runtime_vectors_match()
    check_balance_and_sidecar()
    check_trainer_flag()
    check_run_vision_wires_under_flag()
    check_model_flag_shapes_ckpt_and_flip()
    print("test_route_cond: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
