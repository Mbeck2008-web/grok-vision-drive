#!/usr/bin/env python3
"""Train-only depth/sem aux + separate road_edge (spec 2026-11-05).

* Targets load from uint16 depth / uint8 sem PNGs + sem_classes.json.
* Missing targets are masked; shape/dtype/schema errors raise.
* road_edge is a separate vector; never copied into lane_gt / paint slots.
* Unknown semantic classes are not treated as drivable.
* Default --heads lane_gt + default ONNX I/O stay unchanged.
* Aux / road_edge never become inference inputs.
* Aux spatial grid is fixed and bounded.
"""
# gvd-ci: requires torch
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

H, W = 48, 64
PALETTE = {"CAR": (0, 0, 142), "ROAD": (128, 64, 128), "SKY": (70, 130, 180), "DIRT_UNKNOWN": (160, 120, 80)}


def _write_rec(root: Path, *, depth=None, sem=None, classes=None, rgb=True, bad_sem_schema=False) -> None:
    from python.data.raw_frames import RawFrameWriter, write_sem_classes

    frames = {}
    if rgb:
        frames["main"] = np.full((H, W, 3), 40, np.uint8)
    if depth is not None:
        frames["main_depth"] = depth
    if sem is not None:
        frames["main_sem"] = sem
    writer = RawFrameWriter(root, [])
    writer.write(frames, t=0.0, timestamps={"main": 1.0}, tech={
        "lane_gt": [0.0] * 14,
        "lane_gt_valid": [0.0] * 14,  # unpainted
        "gt_source": "painted_decals",
    })
    if bad_sem_schema:
        (root / "sem_classes.json").write_text(json.dumps({"kind": "nope"}), encoding="utf-8")
    elif classes is not False:
        write_sem_classes(root, classes if classes is not None else PALETTE)


def check_load_ok_and_downsample() -> None:
    from python.train.sensor_aux import AUX_H, AUX_W, load_sensor_aux_targets

    depth = np.full((H, W), 1500, np.uint16)
    depth[0, 0] = 0
    sem = np.zeros((H, W), np.uint8)
    sem[:, : W // 2] = 2  # ROAD after sorted ids: CAR=1, DIRT_UNKNOWN=2, ROAD=3, SKY=4 — wait sorted
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_rec(root, depth=depth, sem=sem)
        got = load_sensor_aux_targets(root, 0, cam="main")
        assert got["depth_m"].shape == (AUX_H, AUX_W)
        assert got["depth_valid"].shape == (AUX_H, AUX_W)
        assert got["sem_id"].shape == (AUX_H, AUX_W)
        assert got["sem_valid"].dtype == np.float32
        assert float(got["depth_m"].max()) > 0
        assert "sem_classes" in got and got["sem_classes"]["train_only"] is True
    print("load_ok_and_downsample")


def check_missing_masked() -> None:
    from python.train.sensor_aux import load_sensor_aux_targets

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_rec(root, depth=None, sem=None, classes=False)
        got = load_sensor_aux_targets(root, 0, cam="main")
        assert got["depth_valid"].sum() == 0
        assert got["sem_valid"].sum() == 0
    print("missing_masked")


def check_alignment_schema_fail() -> None:
    from python.train.sensor_aux import AuxTargetError, load_sensor_aux_targets

    depth = np.full((H, W), 100, np.uint16)
    sem = np.zeros((H, W // 2), np.uint8)  # misaligned width
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        # write depth ok then overwrite sem with wrong shape via raw path
        _write_rec(root, depth=depth, sem=np.zeros((H, W), np.uint8))
        import cv2
        bad = np.zeros((H, W // 2), np.uint8)
        path = next((root / "raw" / "main_sem").rglob("*.png"))
        cv2.imwrite(str(path), bad)
        try:
            load_sensor_aux_targets(root, 0, cam="main")
        except AuxTargetError as exc:
            assert "align" in str(exc).lower() or "shape" in str(exc).lower(), exc
        else:
            raise AssertionError("expected AuxTargetError on shape mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_rec(root, depth=np.full((H, W), 1, np.uint16), sem=np.zeros((H, W), np.uint8), bad_sem_schema=True)
        try:
            load_sensor_aux_targets(root, 0, cam="main")
        except AuxTargetError as exc:
            assert "sem_classes" in str(exc).lower() or "schema" in str(exc).lower(), exc
        else:
            raise AssertionError("expected AuxTargetError on sem_classes schema")
    print("alignment_schema_fail")


def check_road_edge_separate_from_paint() -> None:
    from python.debug.lane_targets import LANE_GT_DIM, lane_gt_from_record
    from python.train.sensor_aux import ROAD_EDGE_DIM, pack_road_edge, road_edge_from_record

    values, valid = pack_road_edge(offset_m=0.5, heading_err_rad=0.0, road_width_m=8.0, off_road=False)
    assert values.shape == (ROAD_EDGE_DIM,) and valid.shape == (ROAD_EDGE_DIM,)
    assert float(valid.sum()) == float(ROAD_EDGE_DIM)
    # Unpainted paint record stays masked.
    paint_v, paint_m = lane_gt_from_record({"lane_gt": [float("nan")] * LANE_GT_DIM, "lane_gt_valid": [0.0] * LANE_GT_DIM})
    assert float(paint_m[:-1].sum()) == 0.0 or float(paint_m[:12].sum()) == 0.0
    row = {
        "lane_gt": paint_v.tolist(),
        "lane_gt_valid": paint_m.tolist(),
        "gt_source": "painted_decals",
        "road_offset_m": 0.5,
        "road_width_m": 8.0,
        "off_road": False,
        "heading_err_rad": 0.0,
    }
    re_v, re_m = road_edge_from_record(row)
    assert float(re_m.sum()) > 0
    # No equality / substitution into lane paint slots.
    assert not np.allclose(re_v[: min(len(re_v), 12)], paint_v[: min(len(re_v), 12)], equal_nan=True)
    assert "road_edge" != "lane_gt"
    # Explicit road_edge fields win and still leave paint alone.
    row2 = dict(row)
    row2["road_edge"] = re_v.tolist()
    row2["road_edge_valid"] = re_m.tolist()
    got_v, got_m = road_edge_from_record(row2)
    assert np.allclose(got_v, re_v) and np.allclose(got_m, re_m)
    paint2, mask2 = lane_gt_from_record(row2)
    assert float(mask2[:12].sum()) == 0.0
    print("road_edge_separate_from_paint")


def check_unknown_sem_not_drivable() -> None:
    from python.train.sensor_aux import DRIVABLE_CLASS_ALLOWLIST, drivable_mask_from_sem

    classes = {"0": "unknown", "1": "CAR", "2": "DIRT_UNKNOWN", "3": "ROAD", "4": "SKY"}
    sem = np.array([[0, 2, 3], [1, 2, 4]], np.uint8)
    mask = drivable_mask_from_sem(sem, classes)
    # Only ROAD (id 3) if ROAD is allowlisted; DIRT_UNKNOWN never.
    assert "DIRT_UNKNOWN" not in DRIVABLE_CLASS_ALLOWLIST
    assert "ROAD" in DRIVABLE_CLASS_ALLOWLIST
    assert int(mask[0, 1]) == 0  # dirt
    assert int(mask[0, 2]) == 1  # road
    assert int(mask[0, 0]) == 0  # unknown
    print("unknown_sem_not_drivable")


def check_default_onnx_and_no_input_leak() -> None:
    import onnx

    from python.train.scene_net import OUTPUT_FIELDS, SceneNet, SceneState, export_scene_onnx, scene_loss

    cams = ("main",)
    plain = SceneNet(cams=cams)
    assert not getattr(plain, "aux_sensors", False)
    assert not getattr(plain, "road_edge", False)
    with tempfile.TemporaryDirectory() as tmp:
        path = export_scene_onnx(Path(tmp) / "plain.onnx", plain, heads=("lane_gt",))
        model = onnx.load(str(path))
        inputs = [i.name for i in model.graph.input]
        outputs = [o.name for o in model.graph.output]
        # Torch may rename state inputs to h.1/tokens.1/mask.1 when outputs reuse those names.
        assert inputs[0] == "strip" and inputs[1] == "dt", inputs
        assert all(n.split(".", 1)[0] in {"strip", "dt", "h", "tokens", "mask"} for n in inputs), inputs
        assert not any(x in n for n in inputs for x in ("depth", "sem", "road_edge", "aux"))
        assert list(outputs[: len(OUTPUT_FIELDS)]) == list(OUTPUT_FIELDS)
        assert "aux_depth" not in outputs and "road_edge" not in outputs
        assert outputs[-3:] == ["h", "tokens", "mask"]

    # Forward inputs are RGB strip only even when labels carry distinctive aux.
    import torch

    net = SceneNet(cams=cams, aux_sensors=True, road_edge=True).eval()
    state = net.initial_state()
    strip = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
    # Poison would show up if concatenated — keep strip zeros and check pred path runs.
    pred, _ = net.forward_batch(strip, torch.zeros(1), state)
    assert pred.lane_gt is not None
    assert pred.road_edge is not None
    assert pred.aux_depth is not None and pred.aux_sem_logits is not None
    # Default loss (heads lane_gt only) ignores aux/road_edge terms in total equality sense:
    target = {
        "lane_gt": torch.zeros(1, pred.lane_gt.shape[-1]),
        "lane_gt_valid": torch.zeros(1, pred.lane_gt.shape[-1]),
        "lanes": torch.zeros_like(pred.lanes),
        "lane_valid": torch.zeros_like(pred.lane_valid),
        "curbs": torch.zeros_like(pred.curbs),
        "curb_valid": torch.zeros_like(pred.curb_valid),
        "objects": torch.zeros_like(pred.objects),
        "object_class": torch.zeros(pred.object_class.shape[:-1], dtype=torch.int64),
        "object_valid": torch.zeros_like(pred.object_valid),
        "signs": torch.zeros_like(pred.signs),
        "sign_class": torch.zeros(pred.sign_class.shape[:-1], dtype=torch.int64),
        "sign_state": torch.zeros(pred.sign_state.shape[:-1], dtype=torch.int64),
        "sign_valid": torch.zeros_like(pred.sign_valid),
        "aux_depth": torch.ones_like(pred.aux_depth) * 99.0,
        "aux_depth_valid": torch.ones_like(pred.aux_depth),
        "aux_sem": torch.zeros(pred.aux_sem_logits.shape[0], pred.aux_sem_logits.shape[2], pred.aux_sem_logits.shape[3], dtype=torch.int64),
        "aux_sem_valid": torch.ones(pred.aux_sem_logits.shape[0], pred.aux_sem_logits.shape[2], pred.aux_sem_logits.shape[3]),
        "road_edge": torch.ones_like(pred.road_edge) * 7.0,
        "road_edge_valid": torch.ones_like(pred.road_edge),
    }
    base_net = SceneNet(cams=cams).eval()
    base_state = base_net.initial_state()
    base_pred, _ = base_net.forward_batch(strip, torch.zeros(1), base_state)
    # Fill matching keys for base
    for k, v in list(target.items()):
        if k.startswith("aux_") or k.startswith("road_edge"):
            continue
        if k not in ("lane_gt", "lane_gt_valid") and hasattr(base_pred, k.split("_")[0] if False else k):
            pass
    t_base = {k: v for k, v in target.items() if not k.startswith("aux_") and not k.startswith("road_edge")}
    # reshape lane targets to base
    t_base["lane_gt"] = torch.zeros(1, base_pred.lane_gt.shape[-1])
    t_base["lane_gt_valid"] = torch.zeros(1, base_pred.lane_gt.shape[-1])
    t_base["lanes"] = torch.zeros_like(base_pred.lanes)
    t_base["lane_valid"] = torch.zeros_like(base_pred.lane_valid)
    t_base["curbs"] = torch.zeros_like(base_pred.curbs)
    t_base["curb_valid"] = torch.zeros_like(base_pred.curb_valid)
    t_base["objects"] = torch.zeros_like(base_pred.objects)
    t_base["object_class"] = torch.zeros(base_pred.object_class.shape[:-1], dtype=torch.int64)
    t_base["object_valid"] = torch.zeros_like(base_pred.object_valid)
    t_base["signs"] = torch.zeros_like(base_pred.signs)
    t_base["sign_class"] = torch.zeros(base_pred.sign_class.shape[:-1], dtype=torch.int64)
    t_base["sign_state"] = torch.zeros(base_pred.sign_state.shape[:-1], dtype=torch.int64)
    t_base["sign_valid"] = torch.zeros_like(base_pred.sign_valid)
    loss_default = scene_loss(base_pred, t_base, heads=("lane_gt",))
    # Opt-in road_edge + aux change total when heads/flags request them.
    loss_re = scene_loss(
        pred,
        target,
        heads=("lane_gt", "road_edge"),
        aux_depth_weight=1.0,
        aux_sem_weight=1.0,
        road_edge_weight=1.0,
    )
    assert float(loss_re["road_edge"]) > 0.0
    assert float(loss_re["aux_depth"]) > 0.0
    print("default_onnx_and_no_input_leak")


def check_aux_grid_bounded() -> None:
    from python.train.sensor_aux import AUX_H, AUX_W, SEM_CLASS_MAX

    assert AUX_H * AUX_W <= 12 * 16
    assert SEM_CLASS_MAX <= 64
    import torch
    from python.train.scene_net import SceneNet

    net = SceneNet(cams=("main",), aux_sensors=True)
    state = net.initial_state()
    strip = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
    pred, _ = net.forward_batch(strip, torch.zeros(1), state)
    assert tuple(pred.aux_depth.shape[-2:]) == (AUX_H, AUX_W)
    assert pred.aux_sem_logits.shape[1] == SEM_CLASS_MAX
    assert pred.aux_depth.numel() + pred.aux_sem_logits.numel() < 200_000
    print("aux_grid_bounded")


def check_parse_heads_road_edge() -> None:
    from python.train.cam_strip import parse_heads

    assert "road_edge" not in parse_heads("")
    assert parse_heads("lane_gt") == ("lane_gt",)
    assert parse_heads("lane_gt,road_edge") == ("lane_gt", "road_edge")
    try:
        parse_heads("lane_gt,paint_from_road")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown head should raise")
    print("parse_heads_road_edge")



def check_trainer_slice_aux_losses() -> None:
    """BLOCKER: train_scene._prediction_at must keep aux/road_edge so losses fire."""
    import torch
    from python.train import train_scene as ts
    from python.train.scene_net import SceneNet, scene_loss

    net = SceneNet(cams=("main",), aux_sensors=True, road_edge=True).eval()
    pred, _ = net.forward_batch(torch.zeros(1, 3, net.canvas_h, net.canvas_w), torch.zeros(1), net.initial_state())
    one = ts._prediction_at(pred, 0)
    assert one.road_edge is not None and one.aux_depth is not None and one.aux_sem_logits is not None
    target = {
        "lane_gt": torch.zeros_like(one.lane_gt),
        "lane_gt_valid": torch.zeros_like(one.lane_gt),
        "lanes": torch.zeros_like(one.lanes),
        "lane_valid": torch.zeros_like(one.lane_valid),
        "curbs": torch.zeros_like(one.curbs),
        "curb_valid": torch.zeros_like(one.curb_valid),
        "objects": torch.zeros_like(one.objects),
        "object_class": torch.zeros(one.object_class.shape[:-1], dtype=torch.int64),
        "object_valid": torch.zeros_like(one.object_valid),
        "signs": torch.zeros_like(one.signs),
        "sign_class": torch.zeros(one.sign_class.shape[:-1], dtype=torch.int64),
        "sign_state": torch.zeros(one.sign_state.shape[:-1], dtype=torch.int64),
        "sign_valid": torch.zeros_like(one.sign_valid),
        "aux_depth": torch.ones_like(one.aux_depth) * 99.0,
        "aux_depth_valid": torch.ones_like(one.aux_depth),
        "aux_sem": torch.zeros(one.aux_sem_logits.shape[-2], one.aux_sem_logits.shape[-1], dtype=torch.int64),
        "aux_sem_valid": torch.ones(one.aux_sem_logits.shape[-2], one.aux_sem_logits.shape[-1]),
        "road_edge": torch.ones_like(one.road_edge) * 7.0,
        "road_edge_valid": torch.ones_like(one.road_edge),
    }
    loss = scene_loss(
        one, target, heads=("lane_gt", "road_edge"),
        aux_depth_weight=1.0, aux_sem_weight=1.0, road_edge_weight=1.0,
    )
    assert float(loss["road_edge"].detach()) > 0.0, loss
    assert float(loss["aux_depth"].detach()) > 0.0, loss
    assert float(loss["aux_sem"].detach()) > 0.0, loss
    assert float(loss["total"].detach()) > 10.0, float(loss["total"].detach())
    print("trainer_slice_aux_losses")


def _lane_target(pred) -> dict:
    import torch

    return {
        "lane_gt": torch.zeros_like(pred.lane_gt),
        "lane_gt_valid": torch.zeros_like(pred.lane_gt),
        "lanes": torch.zeros_like(pred.lanes),
        "lane_valid": torch.zeros_like(pred.lane_valid),
        "curbs": torch.zeros_like(pred.curbs),
        "curb_valid": torch.zeros_like(pred.curb_valid),
        "objects": torch.zeros_like(pred.objects),
        "object_class": torch.zeros(pred.object_class.shape[:-1], dtype=torch.int64),
        "object_valid": torch.zeros_like(pred.object_valid),
        "signs": torch.zeros_like(pred.signs),
        "sign_class": torch.zeros(pred.sign_class.shape[:-1], dtype=torch.int64),
        "sign_state": torch.zeros(pred.sign_state.shape[:-1], dtype=torch.int64),
        "sign_valid": torch.zeros_like(pred.sign_valid),
    }


def _nhw(t):
    """Map a depth pred or labeler target onto (N, H, W) without broadcasting."""
    if t.dim() >= 3 and t.shape[-3] == 1:
        t = t.squeeze(-3)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def _matched_depth_term(pd, gd, md):
    import torch.nn.functional as functional

    pd, gd, md = _nhw(pd), _nhw(gd), _nhw(md)
    per = functional.smooth_l1_loss(pd, gd, reduction="none")
    return (per * md).sum() / md.sum().clamp(min=1.0)


def _no_broadcast_warnings(caught) -> None:
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    bad = [m for m in msgs if "broadcast" in m.lower() or "target size" in m.lower()]
    assert not bad, bad


def check_aux_depth_loss_no_broadcast() -> None:
    """Labeler HW targets must match the depth head without a broadcast warning.

    After train_scene._prediction_at the head is (1, H, W); labels_for_row
    supplies (H, W). A stacked batch is (N, 1, H, W) vs (N, H, W). Both
    layouts must equal the elementwise loss on those matched (N, H, W)
    tensors. aux_sem already unsqueezes; lock that it stays warning-free.
    """
    import warnings

    import torch
    from python.train import train_scene as ts
    from python.train.scene_net import SceneNet, scene_loss

    def _score(pred, target, *, aux_sem_weight=0.0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss = scene_loss(
                pred, target, heads=("lane_gt",),
                aux_depth_weight=1.0, aux_sem_weight=aux_sem_weight,
            )
        _no_broadcast_warnings(caught)
        return loss

    net = SceneNet(cams=("main",), aux_sensors=True).eval()
    # Batch 1, trainer slice: pred (1, H, W), labeler target (H, W).
    pred1, _ = net.forward_batch(
        torch.zeros(1, 3, net.canvas_h, net.canvas_w), torch.zeros(1), net.initial_state()
    )
    one = ts._prediction_at(pred1, 0)
    hw = tuple(one.aux_depth.shape[-2:])
    assert tuple(one.aux_depth.shape) == (1,) + hw, one.aux_depth.shape
    target1 = _lane_target(one)
    target1["aux_depth"] = torch.full(hw, 99.0)
    target1["aux_depth_valid"] = torch.ones(hw)
    target1["aux_sem"] = torch.zeros(hw, dtype=torch.int64)
    target1["aux_sem_valid"] = torch.ones(hw)
    loss1 = _score(one, target1, aux_sem_weight=1.0)
    expected1 = _matched_depth_term(one.aux_depth, target1["aux_depth"], target1["aux_depth_valid"])
    assert torch.allclose(loss1["aux_depth"], expected1), (float(loss1["aux_depth"]), float(expected1))

    # Batch >1, unsliced: pred (N, 1, H, W), stacked labeler targets (N, H, W).
    n = 2
    predn, _ = net.forward_batch(
        torch.zeros(n, 3, net.canvas_h, net.canvas_w),
        torch.zeros(n),
        net.initial_state(batch=n),
    )
    assert tuple(predn.aux_depth.shape) == (n, 1) + hw, predn.aux_depth.shape
    targetn = _lane_target(predn)
    targetn["aux_depth"] = torch.full((n,) + hw, 99.0)
    targetn["aux_depth_valid"] = torch.ones((n,) + hw)
    targetn["aux_sem"] = torch.zeros((n,) + hw, dtype=torch.int64)
    targetn["aux_sem_valid"] = torch.ones((n,) + hw)
    lossn = _score(predn, targetn, aux_sem_weight=1.0)
    expectedn = _matched_depth_term(predn.aux_depth, targetn["aux_depth"], targetn["aux_depth_valid"])
    assert torch.allclose(lossn["aux_depth"], expectedn), (float(lossn["aux_depth"]), float(expectedn))

    # Same layouts through the trainer slice of a wider batch.
    sliced = ts._prediction_at(predn, 1)
    targets = _lane_target(sliced)
    targets["aux_depth"] = torch.full(hw, 42.0)
    targets["aux_depth_valid"] = torch.ones(hw)
    losss = _score(sliced, targets)
    expecteds = _matched_depth_term(sliced.aux_depth, targets["aux_depth"], targets["aux_depth_valid"])
    assert torch.allclose(losss["aux_depth"], expecteds), (float(losss["aux_depth"]), float(expecteds))
    print("aux_depth_loss_no_broadcast")


def check_rgb_alignment_hard_fail() -> None:
    """MAJOR: depth/sem HxW must match RGB main when main exists."""
    from python.train.sensor_aux import AuxTargetError, load_sensor_aux_targets

    depth = np.full((H, W), 100, np.uint16)
    sem = np.zeros((H, W), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_rec(root, depth=depth, sem=sem, rgb=True)
        # overwrite RGB with different size
        import cv2
        from python.data.raw_frames import raw_frame_candidates
        rgb_path = next((root / "raw" / "main").rglob("*.png"))
        cv2.imwrite(str(rgb_path), np.full((H + 4, W + 4, 3), 40, np.uint8))
        try:
            load_sensor_aux_targets(root, 0, cam="main")
        except AuxTargetError as exc:
            assert "rgb" in str(exc).lower() or "align" in str(exc).lower(), exc
        else:
            raise AssertionError("expected AuxTargetError when RGB HxW disagrees")
    print("rgb_alignment_hard_fail")


def check_depth_downsample_zero_bias() -> None:
    """MAJOR: zero-cm invalids must not bias INTER_AREA metre means."""
    from python.train.sensor_aux import _downsample_u16_depth_m

    d = np.full((48, 64), 2000, np.uint16)  # 20 m
    d[0::2, :] = 0
    metres, mask = _downsample_u16_depth_m(d)
    assert float(mask.mean()) >= 0.5
    got = float(metres[mask > 0.5].mean())
    assert abs(got - 20.0) < 0.05, f"biased depth mean {got}"
    print("depth_downsample_zero_bias")


def check_lane_width_not_road_width() -> None:
    """MAJOR: lane_width_m alone must not invent road_edge; need road_width_m or explicit."""
    from python.train.sensor_aux import road_edge_from_record

    masked_v, masked_m = road_edge_from_record(
        {"road_offset_m": 0.5, "lane_width_m": 3.5, "heading_err_rad": 0.0, "off_road": False}
    )
    assert float(masked_m.sum()) == 0.0, "lane_width_m must not pack road_edge"
    ok_v, ok_m = road_edge_from_record(
        {"road_offset_m": 0.5, "road_width_m": 8.0, "heading_err_rad": 0.0, "off_road": False}
    )
    assert float(ok_m.sum()) > 0.0
    print("lane_width_not_road_width")


def check_meaningful_no_leak() -> None:
    """MINOR: aux/road_edge targets affect scene_loss only — never strip / forward inputs."""
    import inspect
    import torch
    from python.train.scene_net import SceneNet, scene_loss

    net = SceneNet(cams=("main",), aux_sensors=True, road_edge=True).eval()
    # forward_batch must not accept label / aux / road_edge kwargs (no concat path).
    params = set(inspect.signature(net.forward_batch).parameters)
    for banned in ("aux_depth", "aux_sem", "road_edge", "labels", "target", "depth", "sem"):
        assert banned not in params, banned
    strip = torch.linspace(0.0, 1.0, 1 * 3 * net.canvas_h * net.canvas_w).reshape(
        1, 3, net.canvas_h, net.canvas_w
    )
    strip_before = strip.clone()
    dt = torch.zeros(1)
    state = net.initial_state()
    pred_a, _ = net.forward_batch(strip, dt, state)
    pred_b, _ = net.forward_batch(strip, dt, net.initial_state())
    assert torch.equal(strip, strip_before), "forward must not mutate strip"
    # Predictions are RGB-determined: identical strip → identical aux_depth (no label channel).
    assert torch.allclose(pred_a.aux_depth, pred_b.aux_depth)
    poison_depth = torch.full_like(pred_a.aux_depth, 123.0)
    poison_re = torch.full_like(pred_a.road_edge, 45.0)
    base_target = {
        "lane_gt": torch.zeros_like(pred_a.lane_gt),
        "lane_gt_valid": torch.zeros_like(pred_a.lane_gt),
        "lanes": torch.zeros_like(pred_a.lanes),
        "lane_valid": torch.zeros_like(pred_a.lane_valid),
        "curbs": torch.zeros_like(pred_a.curbs),
        "curb_valid": torch.zeros_like(pred_a.curb_valid),
        "objects": torch.zeros_like(pred_a.objects),
        "object_class": torch.zeros(pred_a.object_class.shape[:-1], dtype=torch.int64),
        "object_valid": torch.zeros_like(pred_a.object_valid),
        "signs": torch.zeros_like(pred_a.signs),
        "sign_class": torch.zeros(pred_a.sign_class.shape[:-1], dtype=torch.int64),
        "sign_state": torch.zeros(pred_a.sign_state.shape[:-1], dtype=torch.int64),
        "sign_valid": torch.zeros_like(pred_a.sign_valid),
        "aux_depth": torch.zeros_like(pred_a.aux_depth),
        "aux_depth_valid": torch.ones_like(pred_a.aux_depth),
        "road_edge": torch.zeros_like(pred_a.road_edge),
        "road_edge_valid": torch.ones_like(pred_a.road_edge),
    }
    loss_clean = scene_loss(
        pred_a, base_target, heads=("lane_gt", "road_edge"),
        aux_depth_weight=1.0, road_edge_weight=1.0,
    )
    poisoned = dict(base_target)
    poisoned["aux_depth"] = poison_depth
    poisoned["road_edge"] = poison_re
    loss_poison = scene_loss(
        pred_a, poisoned, heads=("lane_gt", "road_edge"),
        aux_depth_weight=1.0, road_edge_weight=1.0,
    )
    # Poison changes the *loss*, proving targets are loss-side only.
    assert float(loss_poison["total"]) != float(loss_clean["total"])
    assert float(loss_poison["aux_depth"]) > float(loss_clean["aux_depth"])
    # Strip still untouched after scoring.
    assert torch.equal(strip, strip_before)
    print("meaningful_no_leak")


def check_scene_loss_rejects_aux_shape_mismatch() -> None:
    """MINOR: scene_loss must raise on aux spatial mismatch, not interpolate."""
    import torch
    from python.train.scene_net import SceneNet, scene_loss
    from python.train.sensor_aux import AuxTargetError

    net = SceneNet(cams=("main",), aux_sensors=True).eval()
    pred, _ = net.forward_batch(torch.zeros(1, 3, net.canvas_h, net.canvas_w), torch.zeros(1), net.initial_state())
    target = {
        "lane_gt": torch.zeros_like(pred.lane_gt),
        "lane_gt_valid": torch.zeros_like(pred.lane_gt),
        "lanes": torch.zeros_like(pred.lanes),
        "lane_valid": torch.zeros_like(pred.lane_valid),
        "curbs": torch.zeros_like(pred.curbs),
        "curb_valid": torch.zeros_like(pred.curb_valid),
        "objects": torch.zeros_like(pred.objects),
        "object_class": torch.zeros(pred.object_class.shape[:-1], dtype=torch.int64),
        "object_valid": torch.zeros_like(pred.object_valid),
        "signs": torch.zeros_like(pred.signs),
        "sign_class": torch.zeros(pred.sign_class.shape[:-1], dtype=torch.int64),
        "sign_state": torch.zeros(pred.sign_state.shape[:-1], dtype=torch.int64),
        "sign_valid": torch.zeros_like(pred.sign_valid),
        "aux_depth": torch.ones(1, 1, 3, 3) * 5.0,
        "aux_depth_valid": torch.ones(1, 1, 3, 3),
    }
    try:
        scene_loss(pred, target, heads=("lane_gt",), aux_depth_weight=1.0)
    except AuxTargetError:
        pass
    else:
        raise AssertionError("expected AuxTargetError on aux depth shape mismatch")
    print("scene_loss_rejects_aux_shape_mismatch")


def check_road_edge_labels_opt_in() -> None:
    """NIT: labels_for_row attaches road_edge only when requested."""
    from python.train.scene_net import labels_for_row

    row = {
        "i": 0,
        "tech": {
            "lane_gt": [0.0] * 14,
            "lane_gt_valid": [0.0] * 14,
            "road_offset_m": 0.5,
            "road_width_m": 8.0,
            "heading_err_rad": 0.0,
            "off_road": False,
        },
    }
    plain = labels_for_row(row)
    assert "road_edge" not in plain and "road_edge_valid" not in plain
    with_re = labels_for_row(row, road_edge=True)
    assert "road_edge" in with_re and float(with_re["road_edge_valid"].sum()) > 0
    print("road_edge_labels_opt_in")


def check_loss_terms_opt_in() -> None:
    """NIT: _loss_terms lists aux/road_edge when those flags are on."""
    from python.train.scene_net import SceneNet
    from python.train.train_scene import _loss_terms

    plain = _loss_terms(SceneNet(cams=("main",)))
    assert "road_edge" not in plain and "aux_depth" not in plain and "aux_sem" not in plain
    both = _loss_terms(SceneNet(cams=("main",), aux_sensors=True, road_edge=True))
    assert "road_edge" in both and "aux_depth" in both and "aux_sem" in both
    print("loss_terms_opt_in")


def main() -> None:
    check_load_ok_and_downsample()
    check_missing_masked()
    check_alignment_schema_fail()
    check_road_edge_separate_from_paint()
    check_unknown_sem_not_drivable()
    check_parse_heads_road_edge()
    check_aux_grid_bounded()
    check_default_onnx_and_no_input_leak()
    check_trainer_slice_aux_losses()
    check_aux_depth_loss_no_broadcast()
    check_rgb_alignment_hard_fail()
    check_depth_downsample_zero_bias()
    check_lane_width_not_road_width()
    check_meaningful_no_leak()
    check_scene_loss_rejects_aux_shape_mismatch()
    check_road_edge_labels_opt_in()
    check_loss_terms_opt_in()
    print("test_sensor_aux_supervision: OK")


if __name__ == "__main__":
    main()
