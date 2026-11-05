#!/usr/bin/env python3
"""Scene index skips training-only GT folders (raw/main_depth, raw/main_sem).

The trainer never reads them, so ``training_frame_bytes`` must not count,
stat, or open them. Counting them tripled the index walk on a recording
with depth and semantic GT.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import python.data.raw_frames as rf  # noqa: E402

N = 7


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        writer = rf.RawFrameWriter(root, [])
        for i in range(N):
            writer.write(
                {
                    "main": np.full((6, 8, 3), 90, np.uint8),
                    "main_depth": np.full((6, 8), 1000, np.uint16),
                    "main_sem": np.ones((6, 8), np.uint8),
                },
                t=float(i),
                timestamps={"main": 1.0 + i},
                tech={"lane": i},
            )
        gt_bytes = sum(p.stat().st_size for d in ("main_depth", "main_sem") for p in (root / "raw" / d).rglob("*") if p.is_file())
        assert gt_bytes > 0
        opened: list[Path] = []
        real = rf.png_signature_ok

        def spy(path):
            opened.append(Path(path))
            return real(path)

        rf.png_signature_ok = spy
        try:
            total, frames = rf.training_frame_bytes(root)
        finally:
            rf.png_signature_ok = real
        everything = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        print(f"frames={frames} total={total} everything={everything} gt_bytes={gt_bytes} opened={len(opened)}")
        assert frames == N, f"counted {frames} frames for {N} main captures"
        assert total == everything - gt_bytes, (total, everything, gt_bytes)
        bad = [p for p in opened if rf.is_train_only_cam(rf.raw_camera_of(p) or "")]
        assert not bad, f"opened {len(bad)} train-only files"
    print("scene index skips train-only GT folders")


if __name__ == "__main__":
    main()
