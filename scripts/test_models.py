#!/usr/bin/env python3
"""Offline checks for the GVD MODEL catalog / loader (no weight download)."""
from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHROME_RE = re.compile(r"tesla|\bfsd\b|full self[- ]driving|autopilot", re.I)

from python.perception.detect import EmptyDetector, OnnxYoloDetector, SyntheticDetector
from python.perception.pipeline import ModularPerception
from python.runtime.debug_opts import DebugOpts
from python.runtime.models import (
    ModelRuntime,
    cycle_id,
    detector_choices,
    e2e_choices,
    ids_for,
    load_detector,
    load_e2e,
)
import python.runtime.models as models_mod


def test_catalog_shipped_n() -> None:
    det_ids = ids_for("detector")
    assert det_ids[0] == "auto"
    assert "synthetic" in det_ids and "empty" in det_ids
    assert "yolov8n-onnx" in det_ids
    n = next(c for c in detector_choices() if c.id == "yolov8n-onnx")
    assert n.path is not None and n.path.is_file()
    for stem in ("yolov8s-onnx", "yolov8m-onnx"):
        if stem in det_ids:
            path = next(c.path for c in detector_choices() if c.id == stem)
            assert path is not None and path.is_file()
    e2e_ids = ids_for("e2e")
    assert e2e_ids[:2] == ["auto", "stub"]
    assert cycle_id("detector", "auto", +1) in det_ids
    assert cycle_id("detector", "empty", +1) == "auto"
    assert cycle_id("e2e", "auto", +1) == "stub"


def test_load_fallbacks() -> None:
    det, miss = load_detector("synthetic")
    assert isinstance(det, SyntheticDetector) and "yolo_weights" in miss
    det, miss = load_detector("empty")
    assert isinstance(det, EmptyDetector)
    try:
        load_detector("not-a-net")
        raise AssertionError("unknown detector must raise")
    except ValueError:
        pass
    pol = load_e2e("stub")
    assert pol.backend == "stub"
    det, miss = load_detector("yolov8n-onnx")
    assert isinstance(det, OnnxYoloDetector) and det.name == "yolov8n-onnx" and miss == []
    import numpy as np

    blank = det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert isinstance(blank, list)
    try:
        load_e2e("missing-e2e-id")
        raise AssertionError("unknown e2e must raise")
    except ValueError:
        pass


def test_hot_swap() -> None:
    perc = ModularPerception(allow_synthetic=True)
    e2e = load_e2e("stub")
    opts = DebugOpts(detector_id="empty", e2e_id="stub")
    rt = ModelRuntime(detector_id="", e2e_id="")
    perc, e2e, notes = rt.sync(opts, perc, e2e, allow_synthetic=False)
    assert isinstance(perc.detector, EmptyDetector)
    assert any("detector -> empty" in n for n in notes)
    opts.detector_id = "synthetic"
    perc, e2e, notes = rt.sync(opts, perc, e2e, allow_synthetic=False)
    assert isinstance(perc.detector, SyntheticDetector)
    assert perc.detector.detect(None), "synthetic still yields smoke leads"


def test_missing_file_reverts(tmp_path: Path) -> None:
    old = models_mod.MODELS_DIR
    models_mod.MODELS_DIR = tmp_path
    try:
        fake = tmp_path / "yolov8n.onnx"
        fake.write_bytes(b"not-an-onnx")
        assert "yolov8n-onnx" in ids_for("detector")
        perc = ModularPerception(allow_synthetic=True)
        e2e = load_e2e("stub")
        opts = DebugOpts(detector_id="yolov8n-onnx", e2e_id="stub")
        rt = ModelRuntime(detector_id="synthetic", e2e_id="stub")
        kept = perc.detector.name
        perc, e2e, notes = rt.sync(opts, perc, e2e, allow_synthetic=True)
        assert opts.detector_id == "synthetic"
        assert perc.detector.name == kept
        assert any("failed" in n for n in notes)
    finally:
        models_mod.MODELS_DIR = old


def test_no_chrome() -> None:
    path = ROOT / "python" / "runtime" / "models.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                skip.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            assert not CHROME_RE.search(node.value), node.value


def main() -> None:
    test_catalog_shipped_n()
    test_load_fallbacks()
    test_hot_swap()
    with tempfile.TemporaryDirectory() as td:
        test_missing_file_reverts(Path(td))
    test_no_chrome()
    print("test_models: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_models: FAIL - {e}")
        sys.exit(1)
