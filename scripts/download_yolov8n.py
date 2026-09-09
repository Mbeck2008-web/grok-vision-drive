#!/usr/bin/env python3
"""Download YOLOv8n and export ONNX @640 into models/ (gitignored).

Reliable one-liner (preferred):
  pip install ultralytics
  yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
  mv yolov8n.onnx models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", action="store_true", default=True)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    print("[GVD] loading yolov8n.pt via ultralytics…")
    model = YOLO("yolov8n.pt")
    if args.no_onnx:
        print("[GVD] skipped ONNX; run: yolo export model=yolov8n.pt format=onnx imgsz=640")
        return
    print("[GVD] exporting ONNX imgsz=640…")
    out = model.export(format="onnx", imgsz=640, simplify=True)
    out_p = Path(str(out))
    dest = MODELS / "yolov8n.onnx"
    if out_p.exists():
        shutil.copy2(out_p, dest)
        print(f"[GVD] wrote {dest}")
    else:
        print("[GVD] export returned no file — use the yolo export one-liner in the docstring")
    print("[GVD] weights are gitignored (*.pt / *.onnx).")


if __name__ == "__main__":
    main()
