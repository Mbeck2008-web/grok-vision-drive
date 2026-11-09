#!/usr/bin/env python3
"""Opt-in detect heads learn a tiny synthetic batch and stay out of the default net."""
# gvd-ci: requires torch
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_default_net_unchanged() -> None:
    import torch

    from python.train.scene_net import SceneNet

    torch.manual_seed(0)
    plain = SceneNet(cams=("main",))
    torch.manual_seed(0)
    flagged = SceneNet(cams=("main",), edge=False, detect_obj=False, brakelight=False)
    assert list(plain.state_dict()) == list(flagged.state_dict())
    for key in plain.state_dict():
        assert torch.equal(plain.state_dict()[key], flagged.state_dict()[key])
    assert not any(key.startswith("detect.") for key in plain.state_dict())
    extra = SceneNet(cams=("main",), edge=True, detect_obj=True, brakelight=True)
    names = list(extra.state_dict())
    assert any(key.startswith("detect.edge_head.") for key in names)
    assert any(key.startswith("detect.obj_head.") for key in names)
    assert any(key.startswith("detect.brake_head.") for key in names)
    print("SceneNet: detect heads add no parameters unless named OK")


def check_learns_and_mask_is_zero() -> None:
    import torch

    from python.train.detect_heads import DetectHeads, detect_terms, geometry_smooth
    from python.train.detect_targets import EDGE_DIM, OBJ_FEAT, OBJ_SLOTS
    from python.train.scene_net import scene_loss

    torch.manual_seed(0)
    head = DetectHeads(edge=True, obj=True, brakelight=True, hidden=16)
    feat = torch.randn(1, 16)
    edge = torch.tensor([-1.5, 1.6, -1.2, 1.7, -0.4, 2.1, 0.2, 2.4], dtype=torch.float32)
    box = torch.zeros(OBJ_SLOTS, OBJ_FEAT)
    valid = torch.zeros(OBJ_SLOTS, OBJ_FEAT)
    box[0] = torch.tensor([1.0, 8.0, 0.2, 0.3, 4.0, 1.8, 1.5])
    valid[0] = 1.0
    cls = torch.zeros(OBJ_SLOTS, dtype=torch.long)
    cls[0] = 1
    target = {
        "edge": edge,
        "edge_valid": torch.ones(EDGE_DIM),
        "obj_box": box,
        "obj_valid": valid,
        "obj_class": cls,
        "brake": torch.tensor(1.0),
        "brake_valid": torch.tensor(1.0),
    }
    opt = torch.optim.Adam(head.parameters(), lr=0.08)
    losses = []
    for _ in range(100):
        pred = SimpleNamespace(**head(feat))
        terms = detect_terms(pred, target, feat.device, torch.float32)
        loss = terms["edge"] + terms["obj"] + terms["brakelight"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
    with torch.no_grad():
        pred = head(feat)
        assert torch.allclose(pred["edge"], edge, atol=0.35), pred["edge"]

    masked = dict(target)
    masked["edge"] = torch.full((EDGE_DIM,), 50.0)
    masked["edge_valid"] = torch.zeros(EDGE_DIM)
    other = dict(masked)
    other["edge"] = torch.full((EDGE_DIM,), -50.0)
    pred = SimpleNamespace(**head(feat))
    geom_a, n_a = geometry_smooth(pred.edge, masked["edge"], masked["edge_valid"])
    geom_b, _n_b = geometry_smooth(pred.edge, other["edge"], other["edge_valid"])
    assert float(geom_a) == 0.0 and float(geom_b) == 0.0 and float(n_a) == 0.0
    # A fully masked target is a constant 0, so it cannot pull the head.
    assert geom_a.grad_fn is None and geom_b.grad_fn is None

    step = scene_loss(
        SimpleNamespace(lanes=torch.zeros(1), edge=pred.edge.detach(), edge_conf=pred.edge_conf.detach()),
        {"edge": masked["edge"], "edge_valid": masked["edge_valid"]},
        heads=("edge",),
    )
    assert float(step["edge_geom"]) == 0.0
    assert float(step["edge_n"]) == 0.0
    print(f"detect heads: loss {losses[0]:.3f} -> {losses[-1]:.3f}, masked geometry is 0 OK")


def check_runtime_tags_the_checkpoint() -> None:
    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet
    from python.train.viz_contract import renderable_state

    torch.manual_seed(2)
    net = SceneNet(cams=("main",), edge=True, detect_obj=True, brakelight=True).cpu().eval()
    frame = (np.random.RandomState(0).random((480, 640, 3)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        meta = {
            "epoch": 1,
            "step": 2,
            "cams": ["main"],
            "scale": 1.0,
            "backbone": "scratch",
            "heads": ["lane_gt", "edge", "obj", "brakelight"],
            "edge": True,
            "obj": True,
            "brakelight": True,
            "arch": net.pause_arch,
        }
        weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
        path = epoch_ckpt_path(Path(tmp), Path(tmp) / "scene_lanes", "20261109T000000Z", 1, 2)
        write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
        runtime = SceneCkptRuntime(path)
        pred = runtime.observe(frame, 0.1)
        state = pred.as_state()
        assert state["lane"]["src"] == "model"
        assert state["lane"]["head"] == "lane_gt"
        assert state["lane"]["ckpt"] == path.name
        edge = state["detect"]["edge"]
        assert edge["src"] == "model" and edge["head"] == "edge" and edge["ckpt"] == path.name
        poisoned = dict(state)
        poisoned["map_road"] = {
            "src": "map",
            "head": "edge",
            "ckpt": path.name,
            "confidence": 0.99,
            "points": [{"x": 0.0, "y": 5.0}],
        }
        kept = renderable_state(poisoned)
        assert "map_road" not in kept
        flat = json_sources(kept)
        assert flat
        assert set(flat) <= {"model"}
    print("runtime: drawable fields are tagged head + ckpt, map roads are dropped OK")


def json_sources(node) -> list[str]:
    found = []
    if isinstance(node, dict):
        if "src" in node:
            found.append(str(node["src"]))
        for value in node.values():
            found.extend(json_sources(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(json_sources(value))
    return found


def main() -> None:
    check_default_net_unchanged()
    check_learns_and_mask_is_zero()
    check_runtime_tags_the_checkpoint()
    print("test_detect_heads: all OK")


if __name__ == "__main__":
    main()
