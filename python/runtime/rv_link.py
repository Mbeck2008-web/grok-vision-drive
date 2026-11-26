"""Reality View link: the supervisor side of the RV child processes.

docs/specs/reality_view_3dgs.md, Feasibility plan step 3 (isolation). The
generator (unproject / splat packing) and the renderer are **child
processes**; this module only starts them, hands frames between their
shared-memory slots, and tells the main view when to degrade to the PR #242
placeholder. After ``actuator.apply`` the supervisor calls ``handoff()``,
which is one non-blocking memcpy (drop on a torn or missing frame). There
is no generation, sort, render or thread here.

Minimal interface on purpose: step 1 (``gvd/rv-contract``) owns the real
input / output / slot schema and the default-OFF flag, and this is not wired
into ``run_vision.py`` yet.

Slot layout (one ``SharedMemory`` block, two buffers, a seqlock each):
header ``<IIII`` magic, nbuf, slot_bytes, latest; then per buffer
``<QQI4x`` seq (odd while writing), frame id, length; then the buffers.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Any

MAGIC = 0x52564C31  # "RVL1"
_HDR = struct.Struct("<IIII")
_BUF = struct.Struct("<QQI4x")
_BUF_AT = (_HDR.size, _HDR.size + _BUF.size)
_DATA_AT = _HDR.size + 2 * _BUF.size


class RvSlot:
    """Double-buffered single-writer slot. Readers never block the writer."""

    def __init__(self, shm: shared_memory.SharedMemory, owner: bool) -> None:
        self.shm = shm
        self._mv = shm.buf  # cached: no property call on the post-apply path
        self.owner = owner
        magic, nbuf, self.slot_bytes, _ = _HDR.unpack_from(self._mv, 0)
        if magic != MAGIC or nbuf != 2:
            raise ValueError("not an RV slot")
        self._frame = 0

    @classmethod
    def create(cls, slot_bytes: int) -> "RvSlot":
        shm = shared_memory.SharedMemory(create=True, size=_DATA_AT + 2 * int(slot_bytes))
        _HDR.pack_into(shm.buf, 0, MAGIC, 2, int(slot_bytes), 0)
        for at in _BUF_AT:
            _BUF.pack_into(shm.buf, at, 0, 0, 0)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> "RvSlot":
        shm = shared_memory.SharedMemory(name=name)
        # The owner unlinks; a child's tracker must not unlink it when the child dies.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        return cls(shm, owner=False)

    @property
    def name(self) -> str:
        return self.shm.name

    def _data(self, i: int) -> int:
        return _DATA_AT + i * self.slot_bytes

    def begin_write(self) -> tuple[int, memoryview]:
        latest = _HDR.unpack_from(self._mv, 0)[3]
        i = 1 - latest
        seq, frame, n = _BUF.unpack_from(self._mv, _BUF_AT[i])
        _BUF.pack_into(self._mv, _BUF_AT[i], seq | 1, frame, n)
        return i, self._mv[self._data(i) : self._data(i) + self.slot_bytes]

    def commit(self, i: int, n: int, frame: int) -> None:
        seq = _BUF.unpack_from(self._mv, _BUF_AT[i])[0]
        _BUF.pack_into(self._mv, _BUF_AT[i], seq + 1, frame, n)
        magic, nbuf, sb, _ = _HDR.unpack_from(self._mv, 0)
        _HDR.pack_into(self._mv, 0, magic, nbuf, sb, i)

    def abort(self, i: int) -> None:
        seq, frame, n = _BUF.unpack_from(self._mv, _BUF_AT[i])
        _BUF.pack_into(self._mv, _BUF_AT[i], seq + 1, frame, n)

    def publish(self, payload: Any) -> int:
        """Writer side (child): copy ``payload`` in and make it the newest frame."""
        mv = memoryview(payload).cast("B")
        n = min(len(mv), self.slot_bytes)
        i, dst = self.begin_write()
        dst[:n] = mv[:n]
        dst.release()
        self._frame += 1
        self.commit(i, n, self._frame)
        return self._frame

    def _newest(self) -> tuple[int, int, int, int] | None:
        i = _HDR.unpack_from(self._mv, 0)[3]
        seq, frame, n = _BUF.unpack_from(self._mv, _BUF_AT[i])
        if seq & 1 or frame == 0:
            return None
        return i, seq, frame, n

    def read_into(self, out: Any, last_frame: int) -> tuple[int, int] | None:
        """Copy the newest complete frame newer than ``last_frame``; None = drop."""
        got = self._newest()
        if got is None or got[2] <= last_frame:
            return None
        i, seq, frame, n = got
        memoryview(out).cast("B")[:n] = self._mv[self._data(i) : self._data(i) + n]
        if _BUF.unpack_from(self._mv, _BUF_AT[i])[0] != seq:
            return None  # torn: the writer lapped us
        return frame, n

    def copy_newest_to(self, dst: "RvSlot", last_frame: int) -> int | None:
        """One memcpy from this slot's newest frame into ``dst``; returns the frame id or None."""
        got = self._newest()
        if got is None or got[2] <= last_frame:
            return None
        i, seq, frame, n = got
        n = min(n, dst.slot_bytes)
        j, buf = dst.begin_write()
        buf[:n] = self._mv[self._data(i) : self._data(i) + n]
        buf.release()
        if _BUF.unpack_from(self._mv, _BUF_AT[i])[0] != seq:
            dst.abort(j)
            return None
        dst.commit(j, n, frame)
        return frame

    def close(self) -> None:
        self._mv = None  # type: ignore[assignment]
        self.shm.close()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class RvLink:
    """Supervisor-side handle: gen child -> gen slot -> handoff -> viz slot -> viz child."""

    def __init__(self, gen_argv: list[str], viz_argv: list[str], slot_bytes: int, stale_s: float = 0.5) -> None:
        self.gen_argv = list(gen_argv)
        self.viz_argv = list(viz_argv)
        self.slot_bytes = int(slot_bytes)
        self.stale_s = float(stale_s)
        self.gen_slot: RvSlot | None = None
        self.viz_slot: RvSlot | None = None
        self._gen: subprocess.Popen | None = None
        self._viz: subprocess.Popen | None = None
        self._last_frame = 0
        self._last_handoff_t: float | None = None
        self.frames_handed = 0

    def start(self) -> None:
        self.gen_slot = RvSlot.create(self.slot_bytes)
        self.viz_slot = RvSlot.create(self.slot_bytes)
        quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        self._gen = subprocess.Popen([sys.executable, *self.gen_argv, "--rv-shm", self.gen_slot.name], **quiet)
        self._viz = subprocess.Popen([sys.executable, *self.viz_argv, "--rv-shm", self.viz_slot.name], **quiet)

    @property
    def gen_pid(self) -> int:
        return self._gen.pid if self._gen else -1

    @property
    def viz_pid(self) -> int:
        return self._viz.pid if self._viz else -1

    def alive_gen(self) -> bool:
        return self._gen is not None and self._gen.poll() is None

    def alive_viz(self) -> bool:
        return self._viz is not None and self._viz.poll() is None

    def handoff(self) -> bool:
        """Post-apply step: one non-blocking memcpy of the newest frame. Never waits."""
        if self.gen_slot is None or self.viz_slot is None:
            return False
        frame = self.gen_slot.copy_newest_to(self.viz_slot, self._last_frame)
        if frame is None:
            return False
        self._last_frame = frame
        self._last_handoff_t = time.monotonic()
        self.frames_handed += 1
        return True

    def degraded(self, now: float | None = None) -> bool:
        if not (self.alive_gen() and self.alive_viz()) or self._last_handoff_t is None:
            return True
        t = time.monotonic() if now is None else now
        return t - self._last_handoff_t > self.stale_s

    def main_view(self, state: dict[str, Any], width: int, height: int, now: float | None = None):
        """None while the viz child owns the view; otherwise the #242 placeholder."""
        if not self.degraded(now):
            return None
        from python.viz.view_select import render_reality_placeholder

        return render_reality_placeholder(state, width, height)

    def close(self) -> None:
        for p in (self._gen, self._viz):
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=5)
        for s in (self.gen_slot, self.viz_slot):
            if s is not None:
                s.close()
        self.gen_slot = self.viz_slot = None
