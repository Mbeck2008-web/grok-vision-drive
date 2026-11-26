"""Reality View Step 3: supervisor-side isolation harness (copy-only, degrade, no blocking).

Spec: docs/specs/reality_view_3dgs.md (Feasibility plan step 3, "Isolation",
"Degrade / disable"). The supervisor never renders, unprojects or packs splats.
After ``actuator.apply`` it calls :meth:`RvHost.after_apply`, which only:

1. polls the child with ``Popen.poll()`` (never ``wait``); a dead child flips
   the host to degraded once (one note) and the main view to the PR #242
   placeholder. Never an error, never a disengage.
2. hands the already-captured frame to the child by a memcpy into the bounded
   input slot; if the child holds the slot (reader busy) the frame is dropped.
3. copies the latest already-packed output slot out of shared memory into the
   supervisor's own buffer, validates the 128-byte header only, and keeps the
   previous good view if the slot is torn (writer mid-write), stale or invalid.

Every step is wrapped: no exception leaves ``after_apply``. Shared memory is a
file-backed ``mmap`` per direction with a seqlock control block (stdlib only,
same code on Linux and win32). This module imports no safety-path modules and
does no GPU selection; flag resolution stays in run_vision / rv_contract.
"""
from __future__ import annotations

import mmap
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from python.viz.rv_contract import (
    RV_DEFAULT_SLOT_BYTES,
    RV_FLAG_HAS_DEPTH,
    RV_RENDERER_MODULE,
    SLOT_HEADER_SIZE,
    RvClient,
    RvSchemaError,
    RvShmHeader,
)

RV_ISO_CHILD_MODULE = "python.viz.rv_isolation_child"

# Control block at offset 0 of each slot file: seq (even = stable, odd = writer
# mid-write), nbytes, reader_busy (consumer holds the slot), armed_pid (child
# writes its pid once its parent watch is armed), reserved.
RV_CTRL_STRUCT = struct.Struct("<QQIIQ")
RV_CTRL_SIZE = RV_CTRL_STRUCT.size
_SEQ = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_OFF_SEQ, _OFF_NBYTES, _OFF_BUSY, _OFF_ARMED = 0, 8, 16, 20


class RvSharedSlot:
    """One bounded single-writer / single-reader shared slot (file-backed mmap, seqlock).

    Neither side ever waits on the other: the writer drops when the reader holds
    the slot, the reader drops when the writer is mid-write or the seq changed
    under the copy.
    """

    def __init__(self, path: str, payload_bytes: int | None = None, *, create: bool = False) -> None:
        self.path = str(path)
        if create:
            if payload_bytes is None or int(payload_bytes) < SLOT_HEADER_SIZE:
                raise RvSchemaError(f"slot payload_bytes must be >= {SLOT_HEADER_SIZE}")
            with open(self.path, "wb") as fh:
                fh.truncate(RV_CTRL_SIZE + int(payload_bytes))
        total = os.path.getsize(self.path)
        if total <= RV_CTRL_SIZE:
            raise RvSchemaError(f"slot file {self.path} too small ({total} B)")
        self.capacity = total - RV_CTRL_SIZE
        self._fh = open(self.path, "r+b")
        self._mm = mmap.mmap(self._fh.fileno(), total)
        self.written = 0
        self.dropped = 0

    # -- control block ---------------------------------------------------
    def seq(self) -> int:
        return _SEQ.unpack_from(self._mm, _OFF_SEQ)[0]

    def reader_busy(self) -> bool:
        return _U32.unpack_from(self._mm, _OFF_BUSY)[0] != 0

    def set_reader_busy(self, busy: bool) -> None:
        _U32.pack_into(self._mm, _OFF_BUSY, 1 if busy else 0)

    def armed_pid(self) -> int:
        return _U32.unpack_from(self._mm, _OFF_ARMED)[0]

    def set_armed_pid(self, pid: int) -> None:
        _U32.pack_into(self._mm, _OFF_ARMED, int(pid) & 0xFFFFFFFF)

    # -- writer ----------------------------------------------------------
    def write(self, data: bytes | bytearray | memoryview) -> bool:
        """Copy ``data`` into the slot. Never waits: drops if the reader holds the slot."""
        n = len(data)
        if n > self.capacity:
            raise RvSchemaError(f"slot payload {n} exceeds slot capacity {self.capacity}")
        if self.reader_busy():
            self.dropped += 1
            return False
        seq = self.seq() & ~1
        _SEQ.pack_into(self._mm, _OFF_SEQ, seq + 1)
        self._mm[RV_CTRL_SIZE : RV_CTRL_SIZE + n] = data
        _SEQ.pack_into(self._mm, _OFF_NBYTES, n)
        _SEQ.pack_into(self._mm, _OFF_SEQ, seq + 2)
        self.written += 1
        return True

    # -- reader ----------------------------------------------------------
    def try_copy_into(self, dst: bytearray, last_seq: int) -> tuple[int, int]:
        """Copy the latest stable payload into ``dst``. Never waits.

        Returns ``(n, seq)``: n > 0 copied, n == 0 nothing new, n < 0 dropped
        (writer mid-write, seq changed under the copy, or oversize).
        """
        s1 = self.seq()
        if s1 == last_seq:
            return 0, s1
        if s1 & 1:
            return -1, last_seq
        n = _SEQ.unpack_from(self._mm, _OFF_NBYTES)[0]
        if n <= 0 or n > self.capacity or n > len(dst):
            return -1, last_seq
        mv = memoryview(self._mm)
        try:
            src = mv[RV_CTRL_SIZE : RV_CTRL_SIZE + n]
            try:
                dst[:n] = src
            finally:
                src.release()
        finally:
            mv.release()
        if self.seq() != s1:
            return -1, last_seq
        return n, s1

    def close(self) -> None:
        try:
            self._mm.close()
        except (BufferError, ValueError, OSError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass


@dataclass
class RvHostStats:
    handoff_ok: int = 0
    handoff_dropped: int = 0
    out_applied: int = 0
    out_dropped: int = 0
    out_rejected: int = 0
    errors: int = 0


class RvHost:
    """Supervisor-side Reality View host: child health poll, copy-only handoff, degrade."""

    def __init__(
        self,
        *,
        module: str = RV_RENDERER_MODULE,
        child_args: tuple[str, ...] = (),
        in_bytes: int = RV_DEFAULT_SLOT_BYTES,
        out_bytes: int = RV_DEFAULT_SLOT_BYTES,
        print_notes: bool = True,
    ) -> None:
        self._dir = tempfile.mkdtemp(prefix="gvd_rv_")
        self.in_slot_path = os.path.join(self._dir, "rv_in.slot")
        self.out_slot_path = os.path.join(self._dir, "rv_out.slot")
        self._in = RvSharedSlot(self.in_slot_path, in_bytes, create=True)
        self._out = RvSharedSlot(self.out_slot_path, out_bytes, create=True)
        self.slot_capacity = self._out.capacity
        # Two supervisor-owned buffers: copy into scratch, validate, then swap.
        self._scratch = bytearray(self._out.capacity)
        self._view = bytearray(self._out.capacity)
        for b in (self._scratch, self._view):  # pre-fault every page now, not on the first tick
            b[::4096] = bytes(len(range(0, len(b), 4096)))
        self._view_len = 0
        self._view_wh = (0, 0)
        self._last_out_seq = 0
        # The #250 stub takes no slot arguments; the isolation child does.
        slot_args: tuple[str, ...] = ()
        if module != RV_RENDERER_MODULE:
            slot_args = ("--in", self.in_slot_path, "--out", self.out_slot_path)
        self.client = RvClient(SLOT_HEADER_SIZE, module=module, extra_args=slot_args + tuple(child_args))
        self.stats = RvHostStats()
        self.degraded = False
        self.degrade_notes = 0
        self.note = ""
        self.print_notes = bool(print_notes)

    @property
    def child_pid(self) -> int | None:
        return self.client.child_pid

    def launch(self) -> int | None:
        pid = self.client.launch()
        self.note = self.client.note
        if pid is None:
            self._degrade("launch failed")
        return pid

    def wait_ready(self, timeout_s: float = 5.0) -> bool:
        """Startup only (never from the loop): wait until the child armed its parent watch."""
        deadline = time.monotonic() + float(timeout_s)
        pid = self.client.child_pid
        while time.monotonic() < deadline:
            if pid is not None and self._out.armed_pid() == (pid & 0xFFFFFFFF):
                return True
            if not self.client.is_alive():
                return False
            time.sleep(0.01)
        return False

    def _degrade(self, reason: str) -> None:
        if self.degraded:
            return
        self.degraded = True
        self._view_len = 0
        self.degrade_notes += 1
        self.note = f"reality-view: {reason}; main view on the placeholder, driving continues"
        if self.print_notes:
            try:
                print(f"[GVD] {self.note}", flush=True)
            except (OSError, ValueError):
                pass

    def after_apply(self, frame: bytes | bytearray | memoryview | None, *, seq: int) -> bool:
        """Post-``actuator.apply`` RV work: poll, copy frame in, copy packed output out.

        Never blocks, never raises. Returns False once degraded.
        """
        try:
            if self.degraded:
                return False
            if not self.client.is_alive():
                proc = self.client._proc
                rc = None if proc is None else proc.returncode
                self._degrade(f"child exited (rc={rc})")
                return False
            if frame is not None:
                if self._in.write(frame):
                    self.stats.handoff_ok += 1
                else:
                    self.stats.handoff_dropped += 1
            n, s = self._out.try_copy_into(self._scratch, self._last_out_seq)
            if n < 0:
                self.stats.out_dropped += 1
            elif n > 0:
                self._last_out_seq = s
                mv = memoryview(self._scratch)
                try:
                    hdr = RvShmHeader.unpack(mv[:SLOT_HEADER_SIZE])
                finally:
                    mv.release()
                if (
                    not hdr.flags & RV_FLAG_HAS_DEPTH
                    or hdr.width <= 0
                    or hdr.height <= 0
                    or SLOT_HEADER_SIZE + hdr.payload_bytes != n
                    or hdr.payload_bytes != hdr.width * hdr.height * 7
                ):
                    self.stats.out_rejected += 1
                else:
                    self._scratch, self._view = self._view, self._scratch
                    self._view_len = n
                    self._view_wh = (hdr.width, hdr.height)
                    self.stats.out_applied += 1
            return True
        except RvSchemaError:
            self.stats.out_rejected += 1
            return True
        except Exception as exc:  # noqa: BLE001 - RV never breaks the supervisor
            self.stats.errors += 1
            self._degrade(f"host error {exc!r}")
            return False

    def view_bytes(self) -> bytes:
        """Copy of the currently applied slot (header + payload); b'' if none / degraded."""
        return bytes(self._view[: self._view_len]) if not self.degraded else b""

    def main_view(self, state: dict[str, Any] | None, width: int, height: int) -> Any:
        """Viz-side composition (not on the after-apply path).

        Degraded or no applied frame -> the bit-identical PR #242 placeholder.
        Otherwise the applied RV RGB is pasted top-left over the placeholder.
        """
        from python.viz.view_select import render_reality_placeholder

        img = render_reality_placeholder(state, width, height)
        if self.degraded or self._view_len == 0:
            return img
        import numpy as np

        w, h = self._view_wh
        rgb = np.frombuffer(self._view, dtype=np.uint8, count=w * h * 3, offset=SLOT_HEADER_SIZE).reshape(h, w, 3)
        hh, ww = min(h, img.shape[0]), min(w, img.shape[1])
        img[:hh, :ww] = rgb[:hh, :ww]
        return img

    def close(self) -> None:
        """Shutdown only: stop the child and remove the slot files."""
        try:
            self.client.close()
        finally:
            self._in.close()
            self._out.close()
            shutil.rmtree(self._dir, ignore_errors=True)
