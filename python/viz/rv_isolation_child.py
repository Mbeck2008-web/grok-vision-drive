"""Reality View Step 3 isolation child: a deliberately slow generator or renderer.

Runs only as a separate process (``python -m python.viz.rv_isolation_child``),
started by :class:`python.viz.rv_isolation.RvHost`. It exists to prove the
supervisor cannot be stalled by its RV children:

* ``--mode slow_generator`` (infer-side stand-in): reads the frame handoff,
  "packs" for ``--gen-ms`` (sleep stand-in for unproject / splat packing, no
  math), then writes a valid packed output slot and stalls ``--stall-ms`` in the
  middle of that write (seq odd), so the supervisor sees torn / busy slots.
* ``--mode slow_renderer`` (viz-side stand-in): holds the input slot reader-busy
  flag for ``--render-ms`` per read, and after ``--hang-after`` reads hangs with
  the flag held, so every supervisor handoff hits backpressure.

Same parent-death watch as the #250 stub (pidfd / OpenProcess / ppid poll), so
the child exits when the supervisor dies. No safety-path imports, no GPU.
"""
from __future__ import annotations

import argparse
import os
import signal
import struct
import sys
import time

_POLL_S = 0.05


def _sleep_watch(seconds: float, watch, stop) -> bool:
    """Sleep up to ``seconds`` in short steps; True if the parent died or a stop was asked."""
    from python.viz.rv_renderer_stub import _parent_has_exited

    end = time.monotonic() + seconds
    while True:
        if stop() or _parent_has_exited(watch):
            return True
        left = end - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(_POLL_S, left))


def _packed_frame(i: int) -> bytes:
    import numpy as np

    from python.viz.rv_contract import RV_SCHEMA_VERSION, RvOutputSpec, pack_output_slot

    h, w = 48, 64
    rgb = np.full((h, w, 3), (i * 13) % 256, dtype=np.uint8)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    spec = RvOutputSpec(
        schema_version=RV_SCHEMA_VERSION, seq=i, timestamp_s=float(i), pose_4x4=np.eye(4, dtype=np.float32), rgb=rgb, depth=depth
    )
    return pack_output_slot(spec, age_ms=0.0)


def _write_stalled(slot, data: bytes, stall_s: float, watch, stop) -> bool:
    """Seqlock write that sleeps mid-write (seq odd) to stress the supervisor reader."""
    from python.viz.rv_isolation import RV_CTRL_SIZE

    mm = slot._mm
    seq = slot.seq() & ~1
    struct.pack_into("<Q", mm, 0, seq + 1)
    half = len(data) // 2
    mm[RV_CTRL_SIZE : RV_CTRL_SIZE + half] = data[:half]
    dead = _sleep_watch(stall_s, watch, stop)
    mm[RV_CTRL_SIZE + half : RV_CTRL_SIZE + len(data)] = data[half:]
    struct.pack_into("<Q", mm, 8, len(data))
    struct.pack_into("<Q", mm, 0, seq + 2)
    return dead


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GVD Reality View isolation child (test stand-in)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--mode", choices=("slow_generator", "slow_renderer"), required=True)
    ap.add_argument("--gen-ms", type=float, default=80.0)
    ap.add_argument("--stall-ms", type=float, default=30.0)
    ap.add_argument("--render-ms", type=float, default=250.0)
    ap.add_argument("--hang-after", type=int, default=3)
    args = ap.parse_args(argv)

    from python.viz.rv_renderer_stub import _close_parent_watch, _open_parent_watch, _parent_has_exited

    watch = _open_parent_watch(int(args.parent_pid))
    if watch is None or _parent_has_exited(watch):
        _close_parent_watch(watch)
        return 0

    stopped = [False]

    def _on_sig(_signum, _frame) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)
    stop = lambda: stopped[0]  # noqa: E731

    from python.viz.rv_isolation import RvSharedSlot

    in_slot = RvSharedSlot(args.in_path)
    out_slot = RvSharedSlot(args.out_path)
    buf = bytearray(in_slot.capacity)
    last_in = 0
    out_slot.set_armed_pid(os.getpid())
    try:
        i = 0
        while not stop():
            i += 1
            if args.mode == "slow_generator":
                n, last_in = in_slot.try_copy_into(buf, last_in)
                if _sleep_watch(args.gen_ms / 1000.0, watch, stop):
                    break
                if _write_stalled(out_slot, _packed_frame(i), args.stall_ms / 1000.0, watch, stop):
                    break
            else:
                in_slot.set_reader_busy(True)
                n, last_in = in_slot.try_copy_into(buf, last_in)
                hold = 3600.0 if i > args.hang_after else args.render_ms / 1000.0
                if _sleep_watch(hold, watch, stop):
                    break
                in_slot.set_reader_busy(False)
                if _sleep_watch(0.01, watch, stop):
                    break
    finally:
        _close_parent_watch(watch)
        in_slot.close()
        out_slot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
