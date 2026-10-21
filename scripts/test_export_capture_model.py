#!/usr/bin/env python3
"""scripts/export_capture_age_model.py: untrained 8-cam gvd_cam_age=1, create-only, never in the repo."""
# gvd-ci: requires torch
from __future__ import annotations

import functools
import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from export_capture_age_model import export  # noqa: E402


def _refused(fn) -> str:
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    raise AssertionError("not refused")


def main() -> None:
    import onnxruntime as ort

    from python.data.cam_hold import CAM_AGE_ORDER
    from python.sensors.cameras import BeamNGPyBackend
    from python.train.scene_lane_runtime import SceneLaneGuard, SceneLaneRuntime

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub" / "capture.onnx"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            got = export(out, seed=0)
        assert got == out.resolve() and out.is_file()
        assert sorted(p.name for p in out.parent.iterdir()) == ["capture.onnx"]  # no partial left
        meta = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
        assert meta.get("gvd_cam_age") == "1" and meta.get("gvd_capture_untrained") == "1", meta
        assert meta.get("gvd_cams") == ",".join(CAM_AGE_ORDER), meta.get("gvd_cams")
        guard = SceneLaneGuard()
        rt = guard.load(functools.partial(SceneLaneRuntime, feeds_cam_age=bool(BeamNGPyBackend.can_feed_cam_age)), out)
        assert guard.refused is None and rt is not None and rt.cam_age and len(rt.cams) == 8
        before = out.stat().st_mtime_ns
        assert "already exists" in _refused(lambda: export(out))
        assert out.stat().st_mtime_ns == before
        assert "inside the repository" in _refused(lambda: export(ROOT / "models" / "capture_test.onnx"))
        assert "inside the repository" in _refused(lambda: export(ROOT / "capture_test.onnx"))
        assert not (ROOT / "models" / "capture_test.onnx").exists() and not (ROOT / "capture_test.onnx").exists()
        assert ".onnx" in _refused(lambda: export(Path(tmp) / "x.bin"))
    print("export_capture_age_model: untrained gvd_cam_age=1 8-cam model, loads on BeamNG path; create-only, repo refused OK")
    print("test_export_capture_model: all OK")


if __name__ == "__main__":
    main()
