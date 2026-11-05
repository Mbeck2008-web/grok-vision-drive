#!/usr/bin/env python3
"""E0: every exported model records the label frame it was trained in.

``gvd_gt_origin`` / ``gvd_gt_labeler`` come from the training rows' own
``tech.gt_origin`` / ``tech.gt_labeler`` (sorted, comma-joined when mixed;
``unknown`` when a labelled row does not say). An export with no training
rows says ``unknown``. The epoch checkpoint carries the frame, so an epoch
export stamps the same values. No BeamNG; temp dir only.

    py -3 scripts/test_gt_origin_stamp.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402

from python.debug.lane_targets import ORIGIN_CENTRELINE, ORIGIN_REF_NODE  # noqa: E402


def _row(origin: str | None, labeler: str | None = None) -> dict:
    tech: dict = {"lane_gt": [0.0]}
    if origin is not None:
        tech["gt_origin"] = origin
    if labeler is not None:
        tech["gt_labeler"] = labeler
    return {"i": 0, "tech": tech}


def check_frame_props() -> None:
    from python.train.cam_strip import GT_LABELER_META, GT_ORIGIN_META, gt_frame_props

    assert GT_ORIGIN_META == "gvd_gt_origin" and GT_LABELER_META == "gvd_gt_labeler"
    v103 = gt_frame_props([_row(ORIGIN_CENTRELINE, "v103"), _row(ORIGIN_CENTRELINE, "v103"), {"i": 1}])
    assert v103 == {GT_ORIGIN_META: ORIGIN_CENTRELINE, GT_LABELER_META: "v103"}, v103
    live = gt_frame_props([_row(ORIGIN_REF_NODE)])
    assert live == {GT_ORIGIN_META: ORIGIN_REF_NODE, GT_LABELER_META: "unknown"}, live
    mixed = gt_frame_props([_row(ORIGIN_REF_NODE), _row(ORIGIN_CENTRELINE, "v103"), _row(None)])
    assert mixed[GT_ORIGIN_META] == ",".join(sorted([ORIGIN_REF_NODE, ORIGIN_CENTRELINE, "unknown"])), mixed
    assert gt_frame_props([]) == {GT_ORIGIN_META: "unknown", GT_LABELER_META: "unknown"}
    print("frame props: single, live (no labeler), mixed, empty OK")


def check_export_stamps() -> None:
    import onnx

    from python.train.cam_strip import GT_LABELER_META, GT_ORIGIN_META, gt_frame_props
    from python.train.scene_net import SceneNet, export_scene_onnx

    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = export_scene_onnx(Path(tmp) / "plain.onnx", SceneNet(cams=("main",)), heads=("lane_gt",))
            framed = export_scene_onnx(
                Path(tmp) / "v103.onnx", SceneNet(cams=("main",)), heads=("lane_gt",),
                gt_frame=gt_frame_props([_row(ORIGIN_CENTRELINE, "v103")]),
            )
        p = {x.key: x.value for x in onnx.load(str(plain)).metadata_props}
        f = {x.key: x.value for x in onnx.load(str(framed)).metadata_props}
    assert p.get(GT_ORIGIN_META) == "unknown" and p.get(GT_LABELER_META) == "unknown", p
    assert f.get(GT_ORIGIN_META) == ORIGIN_CENTRELINE and f.get(GT_LABELER_META) == "v103", f
    print("export: unknown without rows, row frame with rows OK")


def check_epoch_export() -> None:
    import numpy as np
    import onnx

    from python.train.cam_strip import GT_LABELER_META, GT_ORIGIN_META
    from python.train.epoch_ckpt import export_epoch_onnx, write_epoch_ckpt
    from python.train.scene_net import SceneNet, checkpoint_arch

    net = SceneNet(cams=("main",))
    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    meta = {"run": "r", "epoch": 0, "step": 1, "strips": "s", "cams": ["main"], "cam_age": False, "ctl": False,
            "heads": ["lane_gt"], "scale": 1.0, "arch": checkpoint_arch(("main",)),
            GT_ORIGIN_META: ORIGIN_CENTRELINE, GT_LABELER_META: "v103"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        info = write_epoch_ckpt(tmp / "c.npz", weights, meta, min_free_bytes=0, strip_root=tmp / "strips")
        out = tmp / "ep0.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_epoch_onnx(Path(info["path"]), out)
        props = {x.key: x.value for x in onnx.load(str(out)).metadata_props}
    assert props.get(GT_ORIGIN_META) == ORIGIN_CENTRELINE and props.get(GT_LABELER_META) == "v103", props
    print("epoch export: checkpoint frame stamped OK")


def check_trainer_wiring() -> None:
    src = (ROOT / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert "gt_frame = gt_frame_props(rows)" in src
    assert src.count("gt_frame=gt_frame") >= 2, "both in-training exports pass the row frame"
    assert "**gt_frame," in src, "epoch checkpoint meta carries the frame"
    print("wiring: train_scene passes the row frame to export and epoch checkpoint OK")


def main() -> None:
    check_frame_props()
    check_trainer_wiring()
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("onnx") is None:
        skip_check("export_stamps", "torch/onnx missing (it exports a net)")
    else:
        check_export_stamps()
        check_epoch_export()
    print("test_gt_origin_stamp: OK")


if __name__ == "__main__":
    main()
