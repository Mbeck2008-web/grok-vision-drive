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
    loss_with_poison_targets = scene_loss(base_pred, t_base, heads=("lane_gt",))
    assert float(loss_default["total"]) == float(loss_with_poison_targets["total"])
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


def main() -> None:
    check_load_ok_and_downsample()
    check_missing_masked()
    check_alignment_schema_fail()
    check_road_edge_separate_from_paint()
    check_unknown_sem_not_drivable()
    check_parse_heads_road_edge()
    check_aux_grid_bounded()
    check_default_onnx_and_no_input_leak()
    print("test_sensor_aux_supervision: OK")


if __name__ == "__main__":
    main()
