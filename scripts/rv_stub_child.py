#!/usr/bin/env python3
"""Stub Reality View children for ``test_rv_isolation.py`` (test process only).

Not a test script (no ``test_`` prefix), never imported by the supervisor.

* ``gen``: a deliberately slow generator. Each frame it unprojects a fixed
  depth map from a chase pose into points (``unproject``), packs them into
  float32 splat records (``pack_splats``), and publishes them to the
  generator slot. ``--slow-ms`` sleeps and ``--burn-ms`` spins the CPU per
  frame on top of that.
* ``viz``: a deliberately slow renderer. It reads the newest complete frame
  of the viz slot and "renders" it (a reduction), sleeping and burning per
  frame the same way.

The depth here is synthetic (a fixed ramp), not sim GT; step 2 owns the
real unproject prototype.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.runtime.rv_link import RvSlot  # noqa: E402

H, W = 120, 160


def unproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, pose: np.ndarray) -> np.ndarray:
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]].astype(np.float32)
    z = depth.astype(np.float32)
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    return (pts @ pose.T)[:, :3]


def pack_splats(points: np.ndarray, frame: int) -> bytes:
    rec = np.zeros((points.shape[0], 4), dtype=np.float32)
    rec[:, :3] = points
    rec[:, 3] = float(frame)
    return rec.tobytes()


def _burn(ms: float) -> None:
    end = time.perf_counter() + ms / 1000.0
    x = 0
    while time.perf_counter() < end:
        x += 1


def run_gen(slot: RvSlot, slow_ms: float, burn_ms: float) -> None:
    depth = np.linspace(2.0, 40.0, H * W, dtype=np.float32).reshape(H, W)
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = -6.0  # chase cam 6 m behind
    frame = 0
    while True:
        frame += 1
        payload = pack_splats(unproject(depth, 140.0, 140.0, W / 2, H / 2, pose), frame)
        slot.publish(payload[: slot.slot_bytes])
        _burn(burn_ms)
        time.sleep(slow_ms / 1000.0)


def run_viz(slot: RvSlot, slow_ms: float, burn_ms: float) -> None:
    out = bytearray(slot.slot_bytes)
    last = -1
    while True:
        got = slot.read_into(out, last)
        if got is not None:
            last, n = got
            np.frombuffer(out, dtype=np.float32, count=n // 4).sum()
        _burn(burn_ms)
        time.sleep(slow_ms / 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("gen", "viz"))
    ap.add_argument("--rv-shm", required=True)
    ap.add_argument("--slow-ms", type=float, default=0.0)
    ap.add_argument("--burn-ms", type=float, default=0.0)
    a = ap.parse_args()
    slot = RvSlot.attach(a.rv_shm)
    (run_gen if a.role == "gen" else run_viz)(slot, a.slow_ms, a.burn_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
