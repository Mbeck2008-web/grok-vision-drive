#!/usr/bin/env python3
"""Offline train/live preprocessing parity tool (``scripts/parity_check.py``).

* a synthetic frame, all 8 cams and main only: every stage is byte-equal, exit 0.
* the same check through the CLI on a recorded-folder layout (``--rec``).
* an injected mismatch in one live stage (canvas, model_canvas, nchw) is reported
  as MISMATCH on that stage with its max abs diff, the stages before it still
  match, and the exit code is 1.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import parity_check  # noqa: E402
from gvd_test_env import skip_check  # noqa: E402

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

STAGES = ("canvas", "model_canvas", "nchw")


def _by_stage(results: list) -> dict:
    return {r.stage: r for r in results}


def _check_all_match(cams: tuple[str, ...]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = parity_check.write_synthetic(Path(tmp) / "rec")
        res = _by_stage(parity_check.run(root, 0, cams))
    assert tuple(res) == STAGES, tuple(res)
    for stage in ("canvas", "model_canvas"):
        assert res[stage].ok and res[stage].max_abs == 0.0, res[stage]
    if HAVE_TORCH:
        assert res["nchw"].ok and res["nchw"].max_abs == 0.0, res["nchw"]
        assert res["nchw"].live_dtype == "float32", res["nchw"]
    else:
        assert res["nchw"].skipped
        skip_check("nchw stage", "torch is not installed")
    print(f"synthetic cams={','.join(cams)}: every stage byte-equal OK")


def check_synthetic_match() -> None:
    from python.data.strip_writer import STRIP_ORDER

    _check_all_match(tuple(STRIP_ORDER))
    _check_all_match(("main",))


def check_cli_recorded_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = parity_check.write_synthetic(Path(tmp) / "rec")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = parity_check.main(["--rec", str(root), "--index", "0", "--cams", "main"])
    text = out.getvalue()
    assert code == 0, text
    for stage in STAGES[:2]:
        assert f"{stage}: match" in text, text
    print("CLI --rec: exit 0, every stage reported OK")


@contextlib.contextmanager
def _patched(name: str, fn):
    old = getattr(parity_check, name)
    setattr(parity_check, name, fn)
    try:
        yield
    finally:
        setattr(parity_check, name, old)


def _inject(name: str, wrap, stage: str, want_diff: float) -> None:
    old = getattr(parity_check, name)
    with tempfile.TemporaryDirectory() as tmp:
        root = parity_check.write_synthetic(Path(tmp) / "rec")
        with _patched(name, wrap(old)):
            res = _by_stage(parity_check.run(root, 0, ("main",)))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = parity_check.main(["--rec", str(root), "--index", "0", "--cams", "main"])
    if stage == "nchw" and not HAVE_TORCH:
        skip_check("nchw injection", "torch is not installed")
        return
    bad = res[stage]
    assert not bad.ok and not bad.skipped, bad
    assert abs(bad.max_abs - want_diff) < 1e-6, (stage, bad.max_abs, want_diff)
    for before in STAGES[: STAGES.index(stage)]:
        assert res[before].ok, (stage, res[before])
    assert code == 1, out.getvalue()
    assert f"{stage}: MISMATCH" in out.getvalue(), out.getvalue()
    print(f"injected {name} change: {stage} MISMATCH max_abs={bad.max_abs:.6g}, exit 1 OK")


def check_injected_mismatch_named() -> None:
    # Pin the first pixel so the injected diff is known exactly.
    def bump_size(fn):
        def wrapped(*args, **kwargs):
            out = np.array(fn(*args, **kwargs), copy=True)
            v = int(out[0, 0, 0])
            out[0, 0, 0] = v + 3 if v < 250 else v - 3
            return out
        return wrapped

    _inject("compose_strip", bump_size, "canvas", 3.0)
    _inject("prepare_model_strip", bump_size, "model_canvas", 3.0)

    def scale_254(fn):
        def wrapped(strip, width=None):
            out = fn(strip, width=width)
            return None if out is None else out * np.float32(255.0 / 254.0)
        return wrapped

    with tempfile.TemporaryDirectory() as tmp:
        root = parity_check.write_synthetic(Path(tmp) / "rec")
        peak = float(parity_check.load_raw_cameras(root, 0)["main"].max())
    want = abs(np.float32(peak / 255.0) * np.float32(255.0 / 254.0) - np.float32(peak / 255.0))
    _inject("live_strip_nchw", scale_254, "nchw", float(want))


def main() -> None:
    check_synthetic_match()
    check_cli_recorded_layout()
    check_injected_mismatch_named()
    print("test_parity_check: OK")


if __name__ == "__main__":
    main()
