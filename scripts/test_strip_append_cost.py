#!/usr/bin/env python3
"""Per-strip state append stays flat as state.jsonl grows. No BeamNG."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import python.data.strip_writer as strip_writer  # noqa: E402
from python.data.strip_writer import STRIP_ORDER, StripWriter  # noqa: E402


def _bundle() -> dict[str, np.ndarray]:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :] = (90, 90, 90)
    return {cam: frame.copy() for cam in STRIP_ORDER}


def _stamps(frames: dict, t: float) -> dict[str, float]:
    return {cam: t for cam in frames}


def _times(writer: StripWriter, n: int, t0: float) -> list[float]:
    frames = _bundle()
    out: list[float] = []
    for i in range(n):
        t = t0 + i * 0.05
        started = time.perf_counter()
        writer.write(frames, t=t, timestamps=_stamps(frames, t))
        out.append(time.perf_counter() - started)
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    frames = _bundle()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = StripWriter(root)
        writer.write(frames, t=1.0, timestamps=_stamps(frames, 1.0))
        calls = {"n": 0}
        real = strip_writer.read_state_jsonl

        def _counting(path: Path):
            calls["n"] += 1
            return real(path)

        strip_writer.read_state_jsonl = _counting
        try:
            writer.write(frames, t=1.1, timestamps=_stamps(frames, 1.1))
        finally:
            strip_writer.read_state_jsonl = real
        assert calls["n"] == 0, calls
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        assert meta["n_strips"] == 2, meta["n_strips"]

    with tempfile.TemporaryDirectory() as empty_dir, tempfile.TemporaryDirectory() as full_dir:
        empty = StripWriter(empty_dir)
        full_root = Path(full_dir)
        state = full_root / "state.jsonl"
        state.parent.mkdir(parents=True, exist_ok=True)
        with state.open("w", encoding="utf-8") as handle:
            for i in range(4000):
                handle.write(json.dumps({"i": i, "t": float(i)}) + "\n")
        full = StripWriter(full_root)
        # Drop the first write of each (open + meta). Compare the rest.
        empty_dt = _times(empty, 5, 10.0)[1:]
        full_dt = _times(full, 5, 10.0)[1:]
        empty_med = _median(empty_dt)
        full_med = _median(full_dt)
        assert full_med < max(empty_med * 3.0, empty_med + 0.05), (empty_med, full_med)
        full.close()
        empty.close()
    print("test_strip_append_cost: OK")


if __name__ == "__main__":
    main()
