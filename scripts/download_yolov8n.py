#!/usr/bin/env python3
"""Download a YOLOv8 detect checkpoint and export ONNX @640 into models/ (gitignored).

Default is yolov8n (the GVD target). s/m/l/x work with the same ONNX detector
if you have the VRAM. Not a model zoo commit: weights stay local.

  PYTHONPATH=. python scripts/download_yolov8n.py --size n
  PYTHONPATH=. python scripts/download_yolov8n.py --size s
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SIZES = ("n", "s", "m", "l", "x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="n", choices=SIZES, help="YOLOv8 detect size (default n)")
    ap.add_argument("--onnx", action="store_true", default=True)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    stem = f"yolov8{args.size}"
    print(f"[GVD] loading {stem}.pt via ultralytics…")
    model = YOLO(f"{stem}.pt")
    if args.no_onnx:
        dest_pt = MODELS / f"{stem}.pt"
        # Ultralytics caches the pt; copy if we can find it next to cwd.
        cwd_pt = Path(f"{stem}.pt")
        if cwd_pt.is_file():
            shutil.copy2(cwd_pt, dest_pt)
            print(f"[GVD] wrote {dest_pt}")
        print(f"[GVD] skipped ONNX; run: yolo export model={stem}.pt format=onnx imgsz=640")
        return
    print("[GVD] exporting ONNX imgsz=640…")
    out = model.export(format="onnx", imgsz=640, simplify=True)
    out_p = Path(str(out))
    dest = MODELS / f"{stem}.onnx"
    if out_p.exists():
        shutil.copy2(out_p, dest)
        print(f"[GVD] wrote {dest}")
    else:
        print("[GVD] export returned no file — use the yolo export one-liner in the docstring")
    print("[GVD] weights are gitignored (*.pt / *.onnx).")


if __name__ == "__main__":
    main()
