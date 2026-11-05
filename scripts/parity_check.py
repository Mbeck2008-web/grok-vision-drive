#!/usr/bin/env python3
"""Offline train/live preprocessing parity check for one frame.

Runs one recorded tick (``--rec ROOT --index N``) or a synthetic one
(``--synthetic``) through the live path and the training path, and reports
each stage as byte-equal or with its max abs diff:

* ``canvas``: live ``compose_strip`` (run_vision) vs training ``_load_bgr``
  (raw) and the frame cache canvas.
* ``model_canvas``: live ``prepare_model_strip`` (scene runtime) vs training
  ``_frame_row`` (augment off).
* ``nchw``: live ``scene_lane_runtime._strip_nchw`` (the ONNX ``strip`` feed)
  vs trainer ``scene_net._strip_nchw`` (needs torch, else skipped).

Offline only: no BeamNG and no GPU. Scale 1 only; the live runtime refuses a
scaled canvas. Exit code 0 when every stage matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.raw_frames import load_raw_cameras, read_state_jsonl  # noqa: E402
from python.data.strip_writer import STRIP_ORDER, compose_strip  # noqa: E402
from python.train.cam_strip import canvas_width, parse_cams, prepare_model_strip  # noqa: E402
from python.train.scene_lane_runtime import _strip_nchw as live_strip_nchw  # noqa: E402


@dataclass
class StageResult:
    stage: str
    ok: bool
    max_abs: float
    live_shape: tuple = ()
    train_shape: tuple = ()
    live_dtype: str = ""
    train_dtype: str = ""
    skipped: bool = False
    note: str = ""

    def line(self) -> str:
        if self.skipped:
            return f"{self.stage}: skipped ({self.note})"
        shapes = f"live {self.live_shape} {self.live_dtype} / train {self.train_shape} {self.train_dtype}"
        word = "match" if self.ok else "MISMATCH"
        extra = f" ({self.note})" if self.note else ""
        return f"{self.stage}: {word} max_abs={self.max_abs:.6g} {shapes}{extra}"


def compare(stage: str, live: Any, train: Any, note: str = "") -> StageResult:
    if live is None or train is None:
        missing = "live" if live is None else "train"
        return StageResult(stage, False, float("inf"), note=f"{missing} path gave no array")
    a, b = np.asarray(live), np.asarray(train)
    res = StageResult(stage, False, float("inf"), tuple(a.shape), tuple(b.shape), str(a.dtype), str(b.dtype), note=note)
    if a.shape != b.shape:
        return res
    res.max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    res.ok = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return res


def write_synthetic(root: Path, seed: int = 0) -> Path:
    """One tick of random raw camera PNGs (native camera sizes) plus ``state.jsonl``."""
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    for cam in STRIP_ORDER:
        folder = root / "raw" / cam
        folder.mkdir(parents=True, exist_ok=True)
        h, w = (180, 320) if cam in ("wide", "rear") else (150, 200)
        image = (rng.random((h, w, 3)) * 200 + 30).astype(np.uint8)
        assert cv2.imwrite(str(folder / "000000.png"), image)
    (root / "state.jsonl").write_text(json.dumps({"i": 0, "t": 0.0}) + "\n", encoding="utf-8")
    return root


def _row(root: Path, index: int) -> dict[str, Any]:
    for row in read_state_jsonl(root / "state.jsonl"):
        try:
            if int(row.get("i")) == int(index):
                return dict(row)
        except (TypeError, ValueError):
            continue
    raise SystemExit(f"index {index} not in {root / 'state.jsonl'}")


def run(root: Path | str, index: int, cams: tuple[str, ...]) -> list[StageResult]:
    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.train_scene import _frame_row, _load_bgr

    root = Path(root)
    row = _row(root, index)
    width = canvas_width(cams)
    # Live: run_vision stitches the camera frames, the runtime cuts and scales them.
    live_canvas = compose_strip(load_raw_cameras(root, index))
    live_model = prepare_model_strip(live_canvas, cams, width)
    live_nchw = None if live_model is None else live_strip_nchw(live_model, width=width)

    results: list[StageResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        build_frame_cache(root, Path(tmp) / "cache", ",".join(STRIP_ORDER))
        with FrameCache(Path(tmp) / "cache") as cache:
            raw_res = compare("canvas", live_canvas, _load_bgr(root, index), "train raw")
            cache_res = compare("canvas", live_canvas, cache.canvas(index), "train cache")
            canvas = raw_res if not raw_res.ok else cache_res
            canvas.note = "train raw and cache" if raw_res.ok and cache_res.ok else canvas.note
            results.append(canvas)
            got = _frame_row(root, row, cams=cams, cache=cache, augment=False)
    train_model = None if got is None else got[0]
    results.append(compare("model_canvas", live_model, train_model))
    try:
        from python.train.scene_net import _strip_nchw as train_strip_nchw
    except Exception as exc:  # torch missing
        results.append(StageResult("nchw", False, float("nan"), skipped=True, note=f"no torch: {type(exc).__name__}"))
        return results
    from python.data.strip_writer import CANVAS_H

    train_nchw = None if train_model is None else train_strip_nchw(train_model, width=width, height=CANVAS_H).numpy()
    results.append(compare("nchw", live_nchw, train_nchw))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rec", help="recording folder with raw/ and state.jsonl")
    src.add_argument("--synthetic", action="store_true", help="use one synthetic random tick")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--cams", default=",".join(STRIP_ORDER), help="model cameras (default all 8)")
    args = ap.parse_args(argv)
    cams = parse_cams(args.cams)
    with tempfile.TemporaryDirectory() as tmp:
        root = write_synthetic(Path(tmp) / "rec") if args.synthetic else Path(args.rec)
        results = run(root, args.index, cams)
    for res in results:
        print(res.line())
    ok = all(res.ok or res.skipped for res in results)
    print("parity: OK" if ok else "parity: MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
