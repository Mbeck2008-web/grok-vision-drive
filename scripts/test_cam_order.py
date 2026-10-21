#!/usr/bin/env python3
"""Camera order: every stitch is in ``STRIP_ORDER`` whatever order ``cams`` arrives in.

Before this fix, ``train_directory(cams=("main", "pillarL"))`` (the Python
API; the CLI goes through ``parse_cams`` and was already canonical) stitched
main first, while the net, its export metadata and the offline eval use
``pillarL, main``. The ONNX then scored the swapped picture (r sign flip).

* ``canonical_cams`` / ``sectors_for`` / ``scaled_layout`` / ``extract_cams``
  give one layout for either order; unknown names are refused.
* a single camera (``--cams main``, v103) has one order only: its layout is
  the same before and after.
* a tiny CPU run called with ``("main", "pillarL")`` records ``pillarL,main``
  in the checkpoint, and every epoch's ONNX scores the logged rms / r within
  1e-4 on the same n.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from python.data.strip_writer import CANVAS_H, CANVAS_W, STRIP_ORDER  # noqa: E402
from python.train.cam_strip import (  # noqa: E402
    canonical_cams,
    cams_from_metadata,
    extract_cams,
    parse_cams,
    scaled_layout,
    sectors_for,
)


def check_layout() -> None:
    a = ("main", "pillarL")
    b = ("pillarL", "main")
    assert canonical_cams(a) == canonical_cams(b) == parse_cams("main,pillarL") == ("pillarL", "main")
    assert canonical_cams(("rear", "main", "rear")) == ("main", "rear")
    assert canonical_cams(None) == canonical_cams(()) == tuple(STRIP_ORDER)
    try:
        canonical_cams(("main", "bogus"))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown camera accepted")
    assert sectors_for(a) == sectors_for(b)
    assert scaled_layout(a, 1.0) == scaled_layout(b, 1.0)
    assert scaled_layout(a, 0.5) == scaled_layout(b, 0.5)
    rng = np.random.RandomState(0)
    full = rng.randint(0, 256, size=(CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    assert np.array_equal(extract_cams(full, a), extract_cams(full, b))
    assert cams_from_metadata({"gvd_cams": "main,pillarL"}) == ("pillarL", "main")
    print("layout: either order gives the STRIP_ORDER canvas; unknown names refused OK")


def check_single_cam_unchanged() -> None:
    # v103 trained with ``--cams main``: parse_cams gives ("main",), one order only.
    assert parse_cams("main") == canonical_cams(("main",)) == ("main",)
    (sec,) = sectors_for(("main",))
    assert (sec.cam_id, sec.x0, sec.x1) == ("main", 0, 256)
    print("single camera: --cams main has one order and the same 256-wide layout OK")


def check_train_direct_api_order() -> None:
    import torch

    import test_epoch_ckpt as fx
    from eval_onnx_holdout import evaluate
    from python.train.epoch_ckpt import export_epoch_onnx, load_epoch_ckpt

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        fx._fixture(root, side=True)
        torch.manual_seed(0)
        ckpts = Path(tmp) / "ck"
        with contextlib.redirect_stdout(io.StringIO()):
            fx._train(root, cams=("main", "pillarL"), ckpt_dir=ckpts, status_root=Path(tmp) / "_s")
        vals = fx._val_records(Path(tmp) / "_s")
        assert vals
        for rec in vals[:3] + vals[-1:]:
            path = ckpts / rec["ckpt"]
            assert load_epoch_ckpt(path)["meta"]["cams"] == ["pillarL", "main"]
            out = Path(tmp) / f"ep{rec['epoch']}.onnx"
            export_epoch_onnx(path, out)
            got = evaluate(root, out, holdout_blocks=1, eval_chunk=16)
            assert got["n"] == rec["n"], (got["n"], rec["n"])
            for key in ("rms", "r"):
                assert abs(float(got[key]) - float(rec[key])) < 1e-4, (key, got[key], rec[key])
            print(f"direct API ('main','pillarL') epoch={rec['epoch']}: logged r={rec['r']:.6f} onnx r={got['r']:.6f} n={got['n']}")
    print("train_directory: a non-STRIP_ORDER cams tuple trains, records and evaluates in STRIP_ORDER OK")


def main() -> None:
    check_layout()
    check_single_cam_unchanged()
    check_train_direct_api_order()
    print("test_cam_order: all OK")


if __name__ == "__main__":
    main()
