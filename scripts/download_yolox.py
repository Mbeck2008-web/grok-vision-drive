#!/usr/bin/env python3
"""Download official Megvii YOLOX ONNX (Apache-2.0) into models/.

The retail default is YOLOX-Nano, which is also shipped in git. Re-run this
to refresh it, or pass ``--size s`` (and other official sizes) for optional
gitignored extras.

  PYTHONPATH=. python scripts/download_yolox.py
  PYTHONPATH=. python scripts/download_yolox.py --size s

Source: Megvii-BaseDetection/YOLOX 0.1.1rc0 (Apache-2.0, not AGPL).
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
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

# Official pre-generated ONNX (Apache-2.0). decode_in_inference=False.
MODELS_BY_SIZE = {
    "nano": {
        "name": "yolox_nano.onnx",
        "url": f"{RELEASE}/yolox_nano.onnx",
        "sha256": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        "notice": (
            "- models/yolox_nano.onnx  -- shipped. Megvii YOLOX-Nano detect @416, "
            "official ONNX (0.1.1rc0). License: Apache-2.0 (Megvii-BaseDetection/YOLOX, "
            "Copyright (c) 2021-2022 Megvii Inc.). Re-download: PYTHONPATH=. python scripts/download_yolox.py"
        ),
    },
    "s": {
        "name": "yolox_s.onnx",
        "url": f"{RELEASE}/yolox_s.onnx",
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "notice": (
            "- models/yolox_s.onnx  -- optional, gitignored, not in the retail zip. "
            "Megvii YOLOX-s detect @640, official ONNX (0.1.1rc0). License: Apache-2.0 "
            "(Megvii-BaseDetection/YOLOX, Copyright (c) 2021-2022 Megvii Inc.). "
            "Download: PYTHONPATH=. python scripts/download_yolox.py --size s"
        ),
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_notice(spec: dict[str, str]) -> None:
    NOTICE.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTICE.read_text(encoding="utf-8") if NOTICE.is_file() else ""
    name = spec["name"]
    if name in existing and "Apache-2.0" in existing:
        return
    line = spec["notice"]
    block = line + "\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    NOTICE.write_text(existing + block, encoding="utf-8")
    print(f"[GVD] updated {NOTICE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--size",
        default="nano",
        choices=sorted(MODELS_BY_SIZE),
        help="Official YOLOX size (default nano, the shipped detector)",
    )
    ap.add_argument("--url", default=None, help="Override ONNX URL")
    ap.add_argument("--out", default=None, help="Destination (default: models/yolox_<size>.onnx)")
    args = ap.parse_args()
    spec = MODELS_BY_SIZE[args.size]
    url = args.url or spec["url"]
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) if args.out else MODELS / spec["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GVD] downloading Apache YOLOX-{args.size} ONNX → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "GVD-download-yolox"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        digest = _sha256(tmp)
        if args.url is None and digest != spec["sha256"]:
            tmp.unlink(missing_ok=True)
            print(f"[GVD] sha256 mismatch for official YOLOX-{args.size} ONNX: {digest}", file=sys.stderr)
            return 1
        tmp.replace(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[GVD] download failed: {e}", file=sys.stderr)
        return 1
    print(f"[GVD] wrote {dest} ({dest.stat().st_size} bytes, sha256={digest})")
    _ensure_notice(spec)
    if args.size == "nano":
        print("[GVD] YOLOX-Nano is the shipped default (Apache-2.0).")
    else:
        print("[GVD] this size is optional; retail zip ships models/yolox_nano.onnx only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
