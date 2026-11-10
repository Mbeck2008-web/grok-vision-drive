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
    assert float(step["edge"]) == 0.0
    assert step["edge"].grad_fn is None
    print(f"detect heads: loss {losses[0]:.3f} -> {losses[-1]:.3f}, masked geometry is 0 OK")


def check_masked_term_has_no_grad() -> None:
    """A fully masked batch does not train confidence as 'absent'."""
    import torch
    import torch.nn.functional as functional

    from python.train.detect_heads import CONF_WEIGHT, DetectHeads, detect_terms
    from python.train.detect_targets import EDGE_DIM, OBJ_FEAT, OBJ_SLOTS
    from python.train.scene_net import scene_loss

    assert CONF_WEIGHT == 0.1, CONF_WEIGHT
    torch.manual_seed(1)
    feat = torch.randn(1, 16, requires_grad=True)
    head = DetectHeads(edge=True, obj=True, brakelight=True, hidden=16)
    pred = SimpleNamespace(**head(feat))
    target = {
        "edge": torch.zeros(EDGE_DIM),
        "edge_valid": torch.zeros(EDGE_DIM),
        "obj_box": torch.zeros(OBJ_SLOTS, OBJ_FEAT),
        "obj_valid": torch.zeros(OBJ_SLOTS, OBJ_FEAT),
        "obj_class": torch.zeros(OBJ_SLOTS, dtype=torch.long),
        "obj_listed": torch.zeros(()),
        "brake": torch.zeros(()),
        "brake_valid": torch.zeros(()),
    }
    terms = detect_terms(pred, target, feat.device, torch.float32)
    loss = terms["edge"] + terms["obj"] + terms["brakelight"]
    for name in ("edge", "obj", "brakelight"):
        assert float(terms[name]) == 0.0, (name, float(terms[name]))
        assert terms[name].grad_fn is None, name
    assert loss.grad_fn is None
    if loss.requires_grad:
        loss.backward()
    for param in head.parameters():
        assert param.grad is None or int(torch.count_nonzero(param.grad)) == 0
    assert feat.grad is None or int(torch.count_nonzero(feat.grad)) == 0

    # Empty object slots are negatives only when the recorder sent the list.
    listed = dict(target)
    listed["obj_listed"] = torch.ones(())
    listed_terms = detect_terms(pred, listed, feat.device, torch.float32)
    assert listed_terms["obj"].requires_grad
    assert listed_terms["edge"].grad_fn is None

    # Partial mask: one valid metre is not averaged with the seven zeros.
    raw = torch.zeros(1, EDGE_DIM)
    raw[0, 0] = 1.0
    mask = torch.zeros(EDGE_DIM)
    mask[0] = 1.0
    from python.train.detect_heads import geometry_smooth

    geom, _n = geometry_smooth(raw, torch.zeros(EDGE_DIM), mask)
    assert abs(float(geom) - 0.5) < 1e-5, float(geom)

    edge = torch.zeros(EDGE_DIM, requires_grad=True)
    edge_conf = torch.zeros((), requires_grad=True)
    step = scene_loss(
        SimpleNamespace(lanes=torch.zeros(1), edge=edge, edge_conf=edge_conf),
        {"edge": torch.zeros(EDGE_DIM), "edge_valid": mask},
        heads=("edge",),
    )
    bare = functional.binary_cross_entropy_with_logits(edge_conf, torch.ones((), device=edge_conf.device))
    # scene_loss must add the whole term. Adding only *_geom drops this 0.1 * BCE.
    assert torch.allclose(step["total"], step["edge"]), (float(step["total"]), float(step["edge"]))
    assert not torch.allclose(step["total"], step["edge_geom"])
    assert torch.allclose(step["edge"] - step["edge_geom"], 0.1 * bare), (
        float(step["edge"] - step["edge_geom"]),
        float(0.1 * bare),
    )
    print("masked batch: whole detect term is detached 0; scene_loss trains the 0.1 conf term OK")


def check_runtime_tags_the_checkpoint() -> None:
    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet

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
        assert "map_road" not in state
        assert "beamng_gt" not in json_sources(state)
        if float(pred.confidence) >= 0.40:
            assert state["lane"]["src"] == "model"
            assert state["lane"]["ckpt"] == path.name
            assert state.get("centers")
        else:
            assert not state.get("centers")
            assert state.get("lane_x") is None
        edge = (state.get("detect") or {}).get("edge")
        if edge is not None:
            assert edge["src"] == "model" and edge["head"] == "edge" and edge["ckpt"] == path.name
            assert float(edge["confidence"]) >= 0.40
            assert "label_src" not in edge
    print("runtime: as_state keeps only drawable model fields OK")


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
    check_masked_term_has_no_grad()
    check_runtime_tags_the_checkpoint()
    print("test_detect_heads: all OK")


if __name__ == "__main__":
    main()
