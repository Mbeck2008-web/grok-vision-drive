#!/usr/bin/env python3
"""Live strip scaling matches training (#140).

``scene_lane_runtime._strip_nchw`` must scale like the trainer's
``scene_net._strip_nchw``: uint8 input is always divided by 255, and the
``max > 1.5`` test applies to float input only.

* an all-ones uint8 strip gives max == 1/255 (it used to stay 1.0).
* a float strip in 0..1 is unchanged.
* a float strip in 0..255 is divided by 255.
* the float threshold is strict ``> 1.5``: a float strip whose max is exactly 1.5 is
  unchanged, and one whose max is 1.5001 is divided by 255.
* torch cross-check: equal to ``scene_net._strip_nchw`` on the same inputs
  (skipped with ``skip_check`` when torch is absent).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from python.data.strip_writer import CANVAS_H, CANVAS_W  # noqa: E402
from python.train.scene_lane_runtime import _strip_nchw  # noqa: E402

SHAPE = (CANVAS_H, CANVAS_W, 3)
# The float scaling threshold (``max > 1.5``) and a max just over it. Both are exact in float32
# terms: float32(1.5) == 1.5 and float32(1.5001) > 1.5.
AT_THRESHOLD = 1.5
OVER_THRESHOLD = 1.5001


def _float_with_max(rng: np.random.Generator, peak: float) -> np.ndarray:
    """A float32 strip in 0..1 with one pixel set to `peak`, so its max is exactly float32(peak)."""
    arr = rng.random(SHAPE, dtype=np.float32)
    arr[0, 0, 0] = np.float32(peak)
    assert float(arr.max()) == float(np.float32(peak))
    return arr


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(140)
    return {
        "uint8_ones": np.ones(SHAPE, dtype=np.uint8),
        "uint8_random": rng.integers(0, 256, SHAPE, dtype=np.uint8),
        "float_unit": rng.random(SHAPE, dtype=np.float32),
        "float_255": (rng.random(SHAPE, dtype=np.float32) * 255.0).astype(np.float32),
        "float_max_1_5": _float_with_max(rng, AT_THRESHOLD),
        "float_max_1_5001": _float_with_max(rng, OVER_THRESHOLD),
    }


def check_uint8_ones() -> None:
    out = _strip_nchw(np.ones(SHAPE, dtype=np.uint8))
    assert out is not None and out.shape == (1, 3, CANVAS_H, CANVAS_W), getattr(out, "shape", None)
    assert out.dtype == np.float32, out.dtype
    got = float(out.max())
    assert abs(got - 1.0 / 255.0) < 1e-7, f"uint8 all-ones max {got}, want {1.0 / 255.0}"
    print("ok uint8 all-ones -> max 1/255")


def check_float_unit_unchanged() -> None:
    src = _inputs()["float_unit"]
    out = _strip_nchw(src)
    assert out is not None
    np.testing.assert_array_equal(out[0], np.transpose(src, (2, 0, 1)))
    print("ok float 0..1 unchanged")


def check_float_255_divided() -> None:
    src = _inputs()["float_255"]
    out = _strip_nchw(src)
    assert out is not None
    np.testing.assert_allclose(out[0], np.transpose(src, (2, 0, 1)) / 255.0, rtol=0, atol=1e-6)
    print("ok float 0..255 divided by 255")


def check_float_threshold_boundary() -> None:
    at = _inputs()["float_max_1_5"]
    assert float(at.max()) == 1.5
    out = _strip_nchw(at)
    assert out is not None
    np.testing.assert_array_equal(out[0], np.transpose(at, (2, 0, 1)))

    over = _inputs()["float_max_1_5001"]
    assert float(over.max()) > 1.5
    out = _strip_nchw(over)
    assert out is not None
    np.testing.assert_allclose(out[0], np.transpose(over, (2, 0, 1)) / 255.0, rtol=0, atol=1e-6)
    print("ok float max == 1.5 unchanged, max 1.5001 divided by 255")


def check_torch_cross() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_check("torch_cross_check", "torch not installed")
        return
    from python.train.scene_net import _strip_nchw as train_strip

    for name, src in _inputs().items():
        live = _strip_nchw(src)
        ref = train_strip(src).detach().cpu().numpy()
        assert live is not None and live.shape == ref.shape, (name, getattr(live, "shape", None), ref.shape)
        np.testing.assert_allclose(live, ref, rtol=0, atol=1e-6, err_msg=name)
    print("ok live == trainer on all inputs")


def main() -> int:
    check_uint8_ones()
    check_float_unit_unchanged()
    check_float_255_divided()
    check_float_threshold_boundary()
    check_torch_cross()
    print("test_scene_strip_scale: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
