#!/usr/bin/env python3
"""Download YOLOv8n and optionally export ONNX @640 into models/ (gitignored weights)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="store_true", help="Export models/yolov8n.onnx @ imgsz 640")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # downloads to cwd/cache then we copy intent
    pt = MODELS / "yolov8n.pt"
    # ultralytics caches; save explicitly
    model.save(str(pt)) if hasattr(model, "save") else None
    # Prefer writing via export path
    if not pt.exists():
        # model.ckpt path
        src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
        if src and src.exists():
            pt.write_bytes(src.read_bytes())
        else:
            # re-load downloads into models via export trick
            model = YOLO("yolov8n.pt")
            print(f"[GVD] ultralytics has yolov8n; place weights in {pt} if missing")
    if args.onnx:
        model = YOLO(str(pt) if pt.exists() else "yolov8n.pt")
        out = model.export(format="onnx", imgsz=640, simplify=True)
        out_p = Path(out)
        dest = MODELS / "yolov8n.onnx"
        if out_p.exists():
            dest.write_bytes(out_p.read_bytes())
            print(f"[GVD] wrote {dest}")
    print(f"[GVD] done. gitignore covers *.pt / *.onnx under models/")


if __name__ == "__main__":
    main()
