#!/usr/bin/env python3
"""Download official Megvii YOLOX-s ONNX (Apache-2.0) into models/.

Optional second detector row. Retail zip still ships YOLOv8n only.

  PYTHONPATH=. python scripts/download_yolox.py

Source: Megvii-BaseDetection/YOLOX 0.1.1rc0 ``yolox_s.onnx`` (not AGPL).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
NOTICE = MODELS / "NOTICE.txt"

# Official pre-generated ONNX (Apache-2.0). decode_in_inference=False.
YOLOX_S_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
YOLOX_S_NAME = "yolox_s.onnx"
# Published 0.1.1rc0 asset; fail the write if GitHub serves something else.
YOLOX_S_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"

NOTICE_LINES = (
    "- models/yolox_s.onnx  -- optional, gitignored, not in the retail zip. "
    "Megvii YOLOX-s detect @640, official ONNX (0.1.1rc0). License: Apache-2.0 "
    "(Megvii-BaseDetection/YOLOX, Copyright (c) 2021-2022 Megvii Inc.). "
    "Download: PYTHONPATH=. python scripts/download_yolox.py",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_notice() -> None:
    NOTICE.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTICE.read_text(encoding="utf-8") if NOTICE.is_file() else ""
    if "yolox_s.onnx" in existing and "Apache-2.0" in existing:
        return
    block = "\n".join(NOTICE_LINES) + "\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    NOTICE.write_text(existing + block, encoding="utf-8")
    print(f"[GVD] updated {NOTICE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=YOLOX_S_URL, help="ONNX URL (official Megvii default)")
    ap.add_argument("--out", default=None, help="Destination (default: models/yolox_s.onnx)")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) if args.out else MODELS / YOLOX_S_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GVD] downloading Apache YOLOX-s ONNX → {dest}")
    req = urllib.request.Request(args.url, headers={"User-Agent": "GVD-download-yolox-s"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        digest = _sha256(tmp)
        if args.url == YOLOX_S_URL and digest != YOLOX_S_SHA256:
            tmp.unlink(missing_ok=True)
            print(f"[GVD] sha256 mismatch for official YOLOX-s ONNX: {digest}", file=sys.stderr)
            return 1
        tmp.replace(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[GVD] download failed: {e}", file=sys.stderr)
        return 1
    print(f"[GVD] wrote {dest} ({dest.stat().st_size} bytes, sha256={digest})")
    _ensure_notice()
    print("[GVD] YOLOX-s is optional; retail zip still ships models/yolov8n.onnx only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
