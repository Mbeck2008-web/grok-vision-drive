"""Optional offline E2E trainer from Documents/GVD/clips.

Must import / smoke without weights. Does not commit weight blobs.
Toy PilotNet-scale — no transformers / ViT / BEV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def clips_root() -> Path:
    try:
        from python.runtime.state_io import gvd_docs_dir

        return gvd_docs_dir() / "clips"
    except Exception:
        return Path.home() / "Documents" / "GVD" / "clips"


def list_clip_dirs(root: Path | None = None) -> list[Path]:
    root = root or clips_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def smoke_import() -> dict:
    """Import path + stub forward without requiring clips or ONNX weights."""
    from python.control.e2e import E2E_H, E2E_W, make_e2e
    import numpy as np

    pol = make_e2e()
    main = np.zeros((E2E_H, E2E_W, 3), dtype=np.uint8)
    wide = np.zeros((E2E_H, E2E_W, 3), dtype=np.uint8)
    out = pol.forward(main, wide, speed_mps=5.0, steer_deg=0.0)
    return {
        "backend": pol.backend,
        "steer": out.steer,
        "accel": out.accel,
        "ok": out.ok,
        "clips": len(list_clip_dirs()),
    }


def train_epoch_stub(clip_dirs: list[Path], epochs: int = 1) -> dict:
    """No-op training loop placeholder — counts frames in state.jsonl if present."""
    n_frames = 0
    for d in clip_dirs:
        sj = d / "state.jsonl"
        if sj.is_file():
            n_frames += sum(1 for _ in sj.open("r", encoding="utf-8"))
    return {"epochs": int(epochs), "clips": len(clip_dirs), "frames": n_frames, "trained": False}


def main() -> None:
    ap = argparse.ArgumentParser(description="GVD M5 optional E2E offline train (toy)")
    ap.add_argument("--smoke", action="store_true", help="Import + stub forward; exit 0")
    ap.add_argument("--clips", type=str, default="", help="Clips root (default Documents/GVD/clips)")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    if args.smoke:
        info = smoke_import()
        print(json.dumps({"smoke": info}, indent=2))
        return

    root = Path(args.clips) if args.clips else clips_root()
    dirs = list_clip_dirs(root)
    if not dirs:
        print(f"[GVD] no clips under {root}; nothing to train (OK)")
        return
    stats = train_epoch_stub(dirs, epochs=args.epochs)
    print(json.dumps(stats, indent=2))
    print("[GVD] train_e2e: stub only — export ONNX to models/e2e_current.onnx yourself; weights not committed.")


if __name__ == "__main__":
    main()
