#!/usr/bin/env python3
"""Render a GVD clip folder to a 30 fps libx264 mp4.

Each real tick (one JPEG + one state.jsonl row) is held for its logged
duration. No frames are interpolated between ticks. A gap longer than 0.5 s
is treated as a cut.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.viz.clip_render import render_clip_folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path, help="Folder with frames/ and state.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output mp4 (default: clip/gvd_clip.mp4)")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    dest = args.out if args.out is not None else args.clip / "gvd_clip.mp4"
    info = render_clip_folder(args.clip, dest, fps=args.fps)
    print(f"render_clip: {info['n_src']} ticks -> {info['n_out']} frames @ {args.fps:.0f} fps -> {info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
