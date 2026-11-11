#!/usr/bin/env python3
"""Labels v4 PR 4: road_edge stays opt-in; curbs wait for palette proof.

* ``road_edge`` is not a painted-lane label. Naming it on ``--heads`` is the
  only way it enters the loss.
* A class name that merely looks like a curb is not proof. ``curb_valid``
  stays 0 until that exact name is on the human allowlist and in
  ``sem_classes.json``.
* An empty ``--heads`` therefore leaves ``curbs`` out, so the zero mask is
  not a BCE target for "no curb".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402


def _paint_and_edge():
    from python.debug.lane_targets import pack_lane_gt
    from python.train.sensor_aux import pack_road_edge

    paint_v, paint_m = pack_lane_gt(
        lane_offset_m=-0.4, heading_err_rad=0.2, lane_width_m=3.5, off_road=False,
    )
    edge_v, edge_m = pack_road_edge(
        offset_m=1.25, heading_err_rad=0.0, road_width_m=11.0, off_road=False,
    )
    return paint_v, paint_m, edge_v, edge_m


def _edge_polylines(edge_v: np.ndarray) -> list[list[dict[str, float]]]:
    """Two polylines built from road-edge xs. Must not become curb labels."""
    ys = (0.0, 5.0, 10.0, 20.0, 30.0)
    left = [{"x": float(edge_v[0]), "y": float(y)} for y in ys]
    right = [{"x": float(edge_v[1]), "y": float(y)} for y in ys]
    return [left, right]


def _palette(*names: str) -> dict:
    classes = {"0": "unknown"}
    for i, name in enumerate(names, start=1):
        classes[str(i)] = name
    return {"kind": "gvd_sem_classes", "train_only": True, "classes": classes}


def check_allowlist_empty_and_curb_valid_zero() -> None:
    from python.train.scene_net import (
        CURB_NAME_ALLOWLIST,
        confirmed_curb_names,
        curb_targets,
        labels_for_row,
    )

    assert CURB_NAME_ALLOWLIST == frozenset(), CURB_NAME_ALLOWLIST
    palette = _palette("CURB", "kerb", "guardrail", "GUARD_RAIL", "ROAD")
    assert confirmed_curb_names(palette) == ()
    assert confirmed_curb_names(palette, allowlist=frozenset({"CURB"})) == ("CURB",)
    assert confirmed_curb_names(_palette("ROAD"), allowlist=frozenset({"CURB"})) == ()

    _paint_v, _paint_m, edge_v, _edge_m = _paint_and_edge()
    polys = _edge_polylines(edge_v)
    curbs, valid = curb_targets(palette, points=polys)
    assert curbs.shape == (2, 16, 2)
    assert valid.shape == (2,)
    assert float(valid.sum()) == 0.0
    assert float(np.abs(curbs).sum()) == 0.0

    # The gate opens only for the exact human-copied name, and still does not
    # copy the road-edge vector into the slots by itself.
    opened, opened_valid = curb_targets(palette, points=polys, allowlist=frozenset({"CURB"}))
    assert float(opened_valid.sum()) == 2.0
    assert float(np.abs(opened).sum()) > 0.0
    closed, closed_valid = curb_targets(
        _palette("ROAD"), points=polys, allowlist=frozenset({"CURB"}),
    )
    assert float(closed_valid.sum()) == 0.0
    assert float(np.abs(closed).sum()) == 0.0

    paint_v, paint_m, edge_v, edge_m = _paint_and_edge()
    row = {
        "tech": {
            "lane_gt": paint_v.tolist(),
            "lane_gt_valid": paint_m.tolist(),
            "gt_source": "painted_decals",
            "road_offset_m": 1.25,
            "road_width_m": 11.0,
            "heading_err_rad": 0.0,
            "off_road": False,
            "curbs": _edge_polylines(edge_v),
        }
    }
    labels = labels_for_row(row, road_edge=True)
    assert np.allclose(labels["lane_gt"], paint_v)
    assert np.allclose(labels["lane_gt_valid"], paint_m)
    assert float(labels["curb_valid"].sum()) == 0.0
    assert float(np.abs(labels["curbs"]).sum()) == 0.0
    assert "road_edge" in labels and float(labels["road_edge_valid"].sum()) == float(edge_m.sum())
    # No road-edge x was written into a paint slot.
    paint_slots = np.asarray(labels["lane_gt"], dtype=np.float64).reshape(-1)
    for value in np.asarray(edge_v, dtype=np.float64).reshape(-1):
        assert not np.any(np.isclose(paint_slots, value, rtol=0.0, atol=1e-4)), value
    print("allowlist empty, curb_valid 0, paint unchanged")


def check_road_edge_and_curbs_opt_in() -> None:
    import json
    import tempfile

    from python.train.cam_strip import HEAD_NAMES, parse_heads
    from python.train.scene_net import curb_palette_for_training, resolve_train_heads

    empty = resolve_train_heads("")
    assert "road_edge" not in empty
    assert "ctl" not in empty
    assert "curbs" not in empty
    assert "lane_gt" in empty and "lanes" in empty
    # Explicit lists still parse. road_edge never arrives from the empty default.
    assert parse_heads("") == HEAD_NAMES
    assert "road_edge" not in parse_heads("")
    named = resolve_train_heads("lane_gt,road_edge")
    assert named == ("lane_gt", "road_edge")
    assert "curbs" not in named
    # A confirmed name puts curbs back on an empty --heads. A lookalike does not.
    proven = resolve_train_heads("", _palette("CURB"), allowlist=frozenset({"CURB"}))
    assert "curbs" in proven and "road_edge" not in proven
    lookalike = resolve_train_heads("", _palette("CURB"))
    assert "curbs" not in lookalike
    # The production allowlist is empty, so a file that says CURB is still not proof.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sem_classes.json").write_text(json.dumps(_palette("CURB", "guardrail")), encoding="utf-8")
        assert curb_palette_for_training([root]) is None
        assert curb_palette_for_training([root, root]) is None
    assert curb_palette_for_training(None) is None
    assert curb_palette_for_training(()) is None
    # train_scene.main resolves heads through heads_for_run. Importing the
    # trainer pulls cv2; the source lock does not.
    trainer = (ROOT / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert "def heads_for_run(" in trainer
    assert "head_names = heads_for_run(args.heads, strip_roots)" in trainer
    print("road_edge opt-in, empty --heads omits curbs")


def check_empty_heads_does_not_bce_curb_valid() -> None:
    try:
        import torch
    except ImportError:
        skip_check("empty_heads_does_not_bce_curb_valid", "torch missing")
        return
    from python.train.scene_net import SceneNet, labels_for_row, resolve_train_heads, scene_loss

    paint_v, paint_m, edge_v, _edge_m = _paint_and_edge()
    labels = labels_for_row(
        {
            "tech": {
                "lane_gt": paint_v.tolist(),
                "lane_gt_valid": paint_m.tolist(),
                "curbs": _edge_polylines(edge_v),
            }
        }
    )
    assert float(labels["curb_valid"].sum()) == 0.0
    net = SceneNet(cams=("main",)).train()
    blank = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
    target = {key: torch.as_tensor(value) for key, value in labels.items()}
    for key in ("object_class", "sign_class", "sign_state"):
        target[key] = target[key].long()

    def fresh():
        net.zero_grad(set_to_none=True)
        pred, _state = net(blank, 0.1, net.initial_state())
        return pred

    heads = resolve_train_heads("")
    scene_loss(fresh(), target, heads)["total"].backward()
    grad = net.curb_valid_head.weight.grad
    assert grad is None or float(grad.abs().max()) == 0.0, grad

    # The same all-zero mask does BCE-train curb_valid when the head is named.
    # Empty --heads must not take that path. A second forward, so the first
    # backward has not already freed the graph.
    named = scene_loss(fresh(), target, ("curbs",))
    assert float(named["valid"].detach()) > 0.0
    named["total"].backward()
    named_grad = net.curb_valid_head.weight.grad
    assert named_grad is not None and float(named_grad.abs().max()) > 0.0
    print("empty --heads does not BCE-train curb_valid")


def main() -> None:
    check_allowlist_empty_and_curb_valid_zero()
    check_road_edge_and_curbs_opt_in()
    check_empty_heads_does_not_bce_curb_valid()
    print("test_labels_v4_road_edge: OK")


if __name__ == "__main__":
    main()
