#!/usr/bin/env python3
"""Pre-training safety fixes. CPU only, no torch.

(f) with all cameras (no --cams), a row without a decodable main picture is
    dropped (``dropped["no_main"]``), not stitched with a black front sector.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _png(path: Path, value: int = 128) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), value, dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def _row(i: int, t: float, cams: tuple[str, ...], segment: str = "s#0") -> dict:
    return {
        "i": i,
        "t": t,
        "segment": segment,
        "ego": {"speed_mps": 10.0, "damage": 0.0, "dir": [0.0, 1.0, 0.0]},
        "cams": {cam: {"file": f"raw/{cam}/{i:06d}.png"} for cam in cams},
    }


def _recording(tmp: Path, layout: list[tuple[str, ...]], *, step: float = 0.1) -> list[dict]:
    rows = []
    for i, cams in enumerate(layout):
        for cam in cams:
            _png(tmp / "raw" / cam / f"{i:06d}.png")
        rows.append(_row(i, 1000.0 + step * i, cams))
    return rows


def check_no_main_dropped() -> None:
    from python.train.scene_loader import select_training_rows

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        main = ("main", "narrow", "wide")
        side = ("pillarL", "pillarR")
        rows = _recording(tmp, [main, side, main, side, main])
        dropped: dict = {}
        kept, missing, _excluded = select_training_rows(tmp, rows, cams=None, dropped=dropped)
        assert [r["i"] for r in kept] == [0, 2, 4], [r["i"] for r in kept]
        assert missing == 2 and dropped.get("no_main") == 2, (missing, dropped)
        # A camera list without main keeps the side-only rows.
        kept_side, _m, _e = select_training_rows(tmp, rows, cams=("pillarL",), dropped={})
        assert [r["i"] for r in kept_side] == [1, 3], [r["i"] for r in kept_side]
    print("(f) no-main rows dropped with all cameras: kept [0, 2, 4], dropped no_main=2 OK")


def main() -> None:
    check_no_main_dropped()
    print("test_pretrain_safety: OK")


if __name__ == "__main__":
    main()
