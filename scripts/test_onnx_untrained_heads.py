#!/usr/bin/env python3
"""ONNX exports mark the outputs of untrained heads, and the runtime refuses them.

An export always carries every ``OUTPUT_FIELDS`` output; with ``--heads
lane_gt`` the ``lanes`` / ``lane_valid`` (and curb, object, sign) outputs are
random-init. ``gvd_heads_trained`` / ``gvd_outputs_untrained`` say so, and
``require_trained`` / ``SceneLaneRuntime`` refuse to read them.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402
from python.train.cam_strip import (  # noqa: E402
    HEAD_NAMES,
    HEAD_OUTPUTS,
    UntrainedOutputError,
    metadata_dict,
    require_trained,
    untrained_outputs,
)


def test_metadata_marks_untrained_outputs() -> None:
    props = metadata_dict(("main",), ("lane_gt",), height=192, width=256)
    assert props["gvd_heads_trained"] == "lane_gt"
    bad = untrained_outputs(props)
    assert {"lanes", "lane_valid", "curbs", "curb_valid", "objects", "object_class", "object_valid",
            "signs", "sign_class", "sign_state", "sign_valid"} == set(bad)
    assert "lane_gt" not in bad
    both = metadata_dict(("main",), ("lanes", "lane_gt"), height=192, width=256)
    assert "lanes" not in untrained_outputs(both) and "curbs" in untrained_outputs(both)
    full = metadata_dict(("main",), None, height=192, width=256)
    assert untrained_outputs(full) == frozenset() and full["gvd_outputs_untrained"] == ""
    assert set(HEAD_OUTPUTS) == set(HEAD_NAMES)


def test_require_trained() -> None:
    props = metadata_dict(("main",), ("lane_gt",), height=192, width=256)
    require_trained(props, ["lane_gt"])
    for name in ("lanes", "lane_valid"):
        try:
            require_trained(props, [name])
            raise AssertionError(f"{name} is untrained and must be refused")
        except UntrainedOutputError as exc:
            assert name in str(exc)
    # Older exports: gvd_heads only is still read; no metadata marks nothing.
    assert "lanes" in untrained_outputs({"gvd_heads": "lane_gt"})
    assert untrained_outputs({}) == frozenset()


def _tiny_onnx(path: Path, outputs: list[str]) -> bool:
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except ImportError:
        return False
    x = helper.make_tensor_value_info("strip", TensorProto.FLOAT, [1, 3, 192, 256])
    outs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 192, 256]) for name in outputs]
    nodes = [helper.make_node("Identity", ["strip"], [name]) for name in outputs]
    graph = helper.make_graph(nodes, "tiny", [x], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return True


def test_runtime_refuses_untrained_lane_gt() -> None:
    try:
        import onnxruntime  # noqa: F401  # type: ignore
    except ImportError:
        skip_check("runtime_refuses_untrained_lane_gt", "onnxruntime not installed")
        return
    from python.train.cam_strip import stamp_onnx_metadata
    from python.train.scene_lane_runtime import SceneLaneRuntime

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.onnx"
        if not _tiny_onnx(path, ["lanes", "lane_gt"]):
            skip_check("runtime_refuses_untrained_lane_gt", "onnx not installed")
            return
        assert stamp_onnx_metadata(str(path), metadata_dict(("main",), ("lanes",), height=192, width=256))
        rt = SceneLaneRuntime(path)
        assert not rt.ready and rt.session is None
        assert "lane_gt" in rt.reason and "did not train" in rt.reason, rt.reason


def test_runtime_output_refuses_untrained() -> None:
    from python.train.scene_lane_runtime import SceneLaneRuntime

    rt = SceneLaneRuntime.__new__(SceneLaneRuntime)
    rt.untrained = untrained_outputs(metadata_dict(("main",), ("lane_gt",), height=192, width=256))
    rt._last = {"lanes": [0.0], "lane_gt": [1.0]}
    assert rt.output("lane_gt") == [1.0]
    try:
        rt.output("lanes")
        raise AssertionError("lanes is untrained and must be refused")
    except UntrainedOutputError:
        pass
    assert SceneLaneRuntime.READS == ("lane_gt",)


def main() -> None:
    test_metadata_marks_untrained_outputs()
    test_require_trained()
    test_runtime_refuses_untrained_lane_gt()
    test_runtime_output_refuses_untrained()
    print("test_onnx_untrained_heads: OK")


if __name__ == "__main__":
    main()
