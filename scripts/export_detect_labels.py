#!/usr/bin/env python3
"""Write ``detect.jsonl`` training labels beside a strip's ``state.jsonl``.

The file is BeamNG ground truth for the trainer. The live visualizer does not
read it. Missing road width, vehicle size, cones, or lead brake lights stay
masked. Nothing is invented from painted lane width.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.train.detect_targets import export_root


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_detect_labels.py <strip folder>")
    folder = Path(sys.argv[1])
    n = export_root(folder)
    print(f"wrote {n} rows to {folder / 'detect.jsonl'}")


if __name__ == "__main__":
    main()
