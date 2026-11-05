#!/usr/bin/env python3
"""A3 (#150): class-wise NMS on the YOLOv8 ONNX path at IoU 0.45.

A fake session returns a synthetic ``[1, 84, N]`` head, so no model file and
no onnxruntime are needed.

    py -3 scripts/test_yolo_nms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.detect import OnnxYoloDetector  # noqa: E402

CAR, PERSON = 2, 0  # COCO ids


class _Input:
    name = "images"


class FakeSession:
    def __init__(self, rows: list[tuple[float, float, float, float, int, float]]) -> None:
        # Real heads have N >> 84 (8400); detect() orients the tensor on that. Pad with zero-score rows.
        head = np.zeros((84, max(len(rows), 128)), dtype=np.float32)
        for k, (cx, cy, w, h, cls_id, score) in enumerate(rows):
            head[:4, k] = (cx, cy, w, h)
            head[4 + cls_id, k] = score
        self.out = head[None, ...]

    def run(self, _names, _feed):
        return [self.out]


def _detector(rows) -> OnnxYoloDetector:
    det = object.__new__(OnnxYoloDetector)
    det.conf = 0.35
    det.nms_thr = 0.45  # __init__ is skipped (no onnxruntime); the default is checked below
    det.in_h = det.in_w = 640
    det.input_name = _Input.name
    det.session = FakeSession(rows)
    return det


def _detect(rows):
    return _detector(rows).detect(np.zeros((640, 640, 3), dtype=np.uint8))


def main() -> None:
    same = _detect([(320, 320, 100, 80, CAR, 0.9), (322, 318, 100, 80, CAR, 0.8), (318, 322, 98, 82, CAR, 0.6)])
    assert len(same) == 1 and abs(same[0].conf - 0.9) < 1e-6, same
    print("three overlapping same-class rows -> 1 detection (highest score kept) OK")
    apart = _detect([(100, 100, 50, 50, CAR, 0.9), (500, 500, 50, 50, CAR, 0.8)])
    assert len(apart) == 2, apart
    print("two disjoint rows -> 2 detections OK")
    mixed = _detect([(320, 320, 100, 80, CAR, 0.9), (320, 320, 100, 80, PERSON, 0.8)])
    assert sorted(d.cls for d in mixed) == ["pedestrian", "vehicle"], mixed
    print("different classes on the same box are not suppressed OK")
    assert OnnxYoloDetector.__init__.__defaults__[-1] == 0.45
    print("test_yolo_nms: OK")


if __name__ == "__main__":
    main()
