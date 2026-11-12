#!/usr/bin/env python3
"""Labels v4 PR 4: road_edge stays opt-in; curbs wait for palette proof.

* ``road_edge`` is not a painted-lane label. Naming it on ``--heads`` is the
  only way it enters the loss.
* A class name that merely looks like a curb is not proof. ``curb_valid``
  stays 0. A name on ``CURB_NAME_ALLOWLIST`` does not write curb points.
* An empty ``--heads`` leaves ``curbs`` out even when that name is in
  ``sem_classes.json``, so the zero mask is not a BCE target for "no curb".
* Explicit ``--heads curbs`` raises until a labeler sets ``curb_valid`` from
  a ``CURB_NAME_ALLOWLIST`` class id, including when that name matches.
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
    # A recording root with a lookalike palette must not turn stuffed curbs on.
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sem_classes.json").write_text(
            json.dumps(_palette("CURB", "kerb", "guardrail")), encoding="utf-8",
        )
        rooted = labels_for_row(row, root=root, road_edge=True)
    assert float(rooted["curb_valid"].sum()) == 0.0
    assert float(np.abs(rooted["curbs"]).sum()) == 0.0
    assert np.allclose(rooted["lane_gt"], paint_v)
    assert "road_edge" in labels and float(labels["road_edge_valid"].sum()) == float(edge_m.sum())
    # No road-edge x was written into a paint slot.
    paint_slots = np.asarray(labels["lane_gt"], dtype=np.float64).reshape(-1)
    for value in np.asarray(edge_v, dtype=np.float64).reshape(-1):
        assert not np.any(np.isclose(paint_slots, value, rtol=0.0, atol=1e-4)), value
    print("allowlist empty, curb_valid 0, paint unchanged")


def _expect_curb_allowlist(fn) -> None:
    """``fn`` raises ValueError and the message names CURB_NAME_ALLOWLIST."""
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        assert "CURB_NAME_ALLOWLIST" in text, text
        return
    raise AssertionError("expected ValueError naming CURB_NAME_ALLOWLIST")


def check_explicit_curbs_without_allowlist_raises() -> None:
    from python.train.scene_net import resolve_train_heads
    from python.train.train_scene import heads_for_run

    _expect_curb_allowlist(lambda: resolve_train_heads("curbs"))
    _expect_curb_allowlist(lambda: resolve_train_heads("lane_gt,curbs"))
    _expect_curb_allowlist(lambda: heads_for_run("curbs"))
    _expect_curb_allowlist(lambda: heads_for_run("lane_gt,curbs"))
    # A named list that does not ask for curbs is unchanged.
    assert resolve_train_heads("lane_gt,road_edge") == ("lane_gt", "road_edge")
    print("explicit curbs without CURB_NAME_ALLOWLIST raises")


def check_explicit_curbs_with_name_match_raises() -> None:
    """A matching allowlist name is not a labeler. Explicit curbs still raises."""
    import json
    import tempfile

    import python.train.scene_net as scene_net
    from python.train.sensor_aux import read_sem_classes
    from python.train.scene_net import confirmed_curb_names, resolve_train_heads
    from python.train.train_scene import heads_for_run

    allow = frozenset({"CURB"})
    payload = {
        "kind": "gvd_sem_classes",
        "train_only": True,
        "classes": {"0": "unknown", "1": "CURB"},
        "colours": {"CURB": [180, 180, 180]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sem_classes.json").write_text(json.dumps(payload), encoding="utf-8")
        loaded = read_sem_classes(root)
        assert confirmed_curb_names(loaded, allowlist=allow) == ("CURB",)
        _expect_curb_labeler(lambda: resolve_train_heads("curbs", loaded, allowlist=allow))
        _expect_curb_labeler(lambda: resolve_train_heads("lane_gt,curbs", loaded, allowlist=allow))
        old = scene_net.CURB_NAME_ALLOWLIST
        scene_net.CURB_NAME_ALLOWLIST = allow
        try:
            _expect_curb_labeler(lambda: heads_for_run("curbs", [root]))
        finally:
            scene_net.CURB_NAME_ALLOWLIST = old
    print("explicit curbs raises when the allowlist name matches")


def _expect_curb_labeler(fn) -> None:
    """``fn`` raises until a labeler sets curb_valid from the class id."""
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        assert "CURB_NAME_ALLOWLIST" in text, text
        assert "curb_valid" in text, text
        return
    raise AssertionError("expected ValueError: no labeler sets curb_valid")


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
    # A confirmed name does not put curbs back on an empty --heads. No labeler
    # sets curb_valid from that class id. A lookalike does not either.
    proven = resolve_train_heads("", _palette("CURB"), allowlist=frozenset({"CURB"}))
    assert "curbs" not in proven and "road_edge" not in proven
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
    print("road_edge opt-in, empty --heads omits curbs")


def check_main_loss_heads_omit_curbs() -> None:
    """The heads tuple main hands to the loss omits curbs on an empty --heads.

    A source grep of ``heads_for_run`` stays green if main then trains with
    ``parse_heads(args.heads)``. This calls main and reads that tuple.
    """
    import tempfile

    import python.train.train_scene as train_scene
    from python.train.cam_strip import HEAD_NAMES

    seen: dict = {}
    original = train_scene.train_directory

    def capture(*_args, **kwargs):
        seen["heads"] = kwargs.get("heads")
        raise SystemExit(0)

    train_scene.train_directory = capture
    code = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rec"
            root.mkdir()
            try:
                train_scene.main(
                    [
                        "--strips", str(root),
                        "--cams", "main",
                        "--heads", "",
                        "--no-panel",
                        "--autostart",
                        "--no-resume",
                        "--dry-run",
                        "--workers", "0",
                        "--epochs", "1",
                    ]
                )
            except SystemExit as exc:
                code = exc.code
    finally:
        train_scene.train_directory = original
    assert code == 0, code
    want = tuple(name for name in HEAD_NAMES if name != "curbs")
    assert seen.get("heads") == want, seen.get("heads")
    print("train main loss heads omit curbs")


def check_default_onnx_outputs_are_exact() -> None:
    """Default export outputs are OUTPUT_FIELDS plus h, tokens, mask. Nothing else."""
    try:
        import onnx
    except ImportError:
        skip_check("default_onnx_outputs_are_exact", "onnx missing")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_check("default_onnx_outputs_are_exact", "torch missing")
        return
    import tempfile

    from python.train.scene_net import OUTPUT_FIELDS, SceneNet, export_scene_onnx

    net = SceneNet(cams=("main",))
    with tempfile.TemporaryDirectory() as tmp:
        path = export_scene_onnx(Path(tmp) / "plain.onnx", net, heads=("lane_gt",))
        model = onnx.load(str(path))
    outputs = [item.name for item in model.graph.output]
    assert outputs == list(OUTPUT_FIELDS) + ["h", "tokens", "mask"], outputs
    print("default ONNX outputs match OUTPUT_FIELDS + h, tokens, mask")


def check_allowlist_match_without_points_has_no_curb_grad() -> None:
    """A real palette hit is not a curb label when no class id wrote points."""
    import json
    import tempfile

    try:
        import torch
    except ImportError:
        skip_check("allowlist_match_without_points_has_no_curb_grad", "torch missing")
        return
    import python.train.scene_net as scene_net
    from python.train.sensor_aux import read_sem_classes
    from python.train.scene_net import (
        SceneNet,
        confirmed_curb_names,
        labels_for_row,
        resolve_train_heads,
        scene_loss,
    )

    paint_v, paint_m, edge_v, _edge_m = _paint_and_edge()
    payload = {
        "kind": "gvd_sem_classes",
        "train_only": True,
        "classes": {"0": "unknown", "1": "CURB"},
        "colours": {"CURB": [180, 180, 180]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sem_classes.json").write_text(json.dumps(payload), encoding="utf-8")
        loaded = read_sem_classes(root)
        assert confirmed_curb_names(loaded, allowlist=frozenset({"CURB"})) == ("CURB",)
        heads = resolve_train_heads("", loaded, allowlist=frozenset({"CURB"}))
        assert "curbs" not in heads, heads
        bare = labels_for_row(
            {"tech": {"lane_gt": paint_v.tolist(), "lane_gt_valid": paint_m.tolist()}},
            root=root,
        )
        assert float(bare["curb_valid"].sum()) == 0.0
        old = scene_net.CURB_NAME_ALLOWLIST
        scene_net.CURB_NAME_ALLOWLIST = frozenset({"CURB"})
        try:
            stuffed = labels_for_row(
                {
                    "tech": {
                        "lane_gt": paint_v.tolist(),
                        "lane_gt_valid": paint_m.tolist(),
                        "curbs": _edge_polylines(edge_v),
                    }
                },
                root=root,
            )
        finally:
            scene_net.CURB_NAME_ALLOWLIST = old
        assert float(stuffed["curb_valid"].sum()) == 0.0
        assert float(np.abs(stuffed["curbs"]).sum()) == 0.0
        labels = bare

    net = SceneNet(cams=("main",)).train()
    blank = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
    target = {key: torch.as_tensor(value) for key, value in labels.items()}
    for key in ("object_class", "sign_class", "sign_state"):
        target[key] = target[key].long()
    net.zero_grad(set_to_none=True)
    pred, _state = net(blank, 0.1, net.initial_state())
    scene_loss(pred, target, heads)["total"].backward()
    for param in (net.curb_head.weight, net.curb_valid_head.weight):
        grad = param.grad
        assert grad is None or float(grad.abs().max()) == 0.0, grad
    print("allowlist match with no curb points leaves curb grad at 0")


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
    curb_xy = net.curb_head.weight.grad
    assert curb_xy is None or float(curb_xy.abs().max()) == 0.0, curb_xy
    print("empty --heads does not BCE-train curb_valid")


def main() -> None:
    check_allowlist_empty_and_curb_valid_zero()
    check_explicit_curbs_without_allowlist_raises()
    check_explicit_curbs_with_name_match_raises()
    check_road_edge_and_curbs_opt_in()
    check_main_loss_heads_omit_curbs()
    check_default_onnx_outputs_are_exact()
    check_allowlist_match_without_points_has_no_curb_grad()
    check_empty_heads_does_not_bce_curb_valid()
    print("test_labels_v4_road_edge: OK")


if __name__ == "__main__":
    main()
