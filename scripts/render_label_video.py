#!/usr/bin/env python3
"""Offline label-review pictures. Training rows over the recorded camera.

    python scripts/render_label_video.py <recording_dir> --out clip.mp4

The live supervisor does not import this script. A folder may hold
``frames/`` or ``raw/main/`` plus ``labels.jsonl`` (or ``gt.jsonl`` /
``state.jsonl``). Missing label fields are left undrawn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw training labels on recorded frames")
    ap.add_argument("recording_dir", nargs="?", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument("--png-dir", default="", help="Also write numbered png frames here")
    args = ap.parse_args(argv)
    from python.offline.label_review import encode_mp4, load_recording, render_label_frame, write_frames

    folder = Path(args.recording_dir) if args.recording_dir else None
    if folder is None or not folder.is_dir():
        print("recording folder is required", file=sys.stderr)
        return 2
    pairs = load_recording(folder)
    if not pairs:
        print(f"no frames in {folder}", file=sys.stderr)
        return 2
    painted = [render_label_frame(frame, labels) for frame, labels in pairs]
    if args.png_dir:
        write_frames(pairs, Path(args.png_dir))
    if args.out:
        dest = Path(args.out)
        if dest.suffix.lower() == ".mp4":
            ok = encode_mp4(painted, dest, fps=float(args.fps))
            if not ok:
                print("mp4 encode failed", file=sys.stderr)
                return 1
            print(dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            import cv2
            cv2.imwrite(str(dest), painted[0])
            print(dest)
    else:
        print(f"frames {len(painted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
