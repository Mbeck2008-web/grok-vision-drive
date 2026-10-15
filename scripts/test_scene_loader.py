#!/usr/bin/env python3
"""Scene loader stays in-process, and a spawned process can read raw_repaired."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.raw_frames import _png_bytes, load_training_bgr, read_image_file  # noqa: E402
from python.data.strip_writer import compose_strip  # noqa: E402
from python.train.fit import _windows_memory, read_free_ram_bytes, read_used_ram_bytes  # noqa: E402
from python.train.scene_loader import _spawn_load, default_num_workers  # noqa: E402
from python.train.train_scene import (  # noqa: E402
    _confirm_start,
    arm_dry_run_watchdog,
    disarm_dry_run_watchdog,
    train_directory,
)


class _Kernel:
    def __init__(self, avail: int, total: int, ok: int = 1) -> None:
        self.avail = avail
        self.total = total
        self.ok = ok

    def GlobalMemoryStatusEx(self, ref) -> int:
        status = ref._obj
        status.ullAvailPhys = self.avail
        status.ullTotalPhys = self.total
        return self.ok


class _HangWindow:
    available = True

    def consume_start(self) -> bool:
        return False

    def window_visible(self) -> bool:
        return True

    def pump(self) -> None:
        raise AssertionError("dry-run waited on the Start button")


def check_windows_memory_status() -> None:
    avail, total = 17_800_000_000, 31_900_000_000
    assert _windows_memory(_Kernel(avail, total)) == (avail, total)
    assert _windows_memory(_Kernel(avail, total, ok=0)) is None
    import python.train.fit as fit

    real_meminfo = fit._meminfo_kb
    real_windows = fit._windows_memory
    real_psutil = fit._psutil_memory
    try:
        fit._meminfo_kb = lambda _name: None
        fit._windows_memory = lambda kernel32=None: (avail, total)
        assert read_free_ram_bytes() == avail
        assert read_used_ram_bytes() == total - avail
        fit._windows_memory = lambda kernel32=None: None
        fit._psutil_memory = lambda: (4_000, 8_000)
        assert read_free_ram_bytes() == 4_000
        assert read_used_ram_bytes() == 4_000
    finally:
        fit._meminfo_kb = real_meminfo
        fit._windows_memory = real_windows
        fit._psutil_memory = real_psutil
    assert read_free_ram_bytes() > 0
    print(f"free ram {read_free_ram_bytes()}")


def check_workers_and_dry_run_start() -> None:
    assert default_num_workers() == 0
    assert "win32" in inspect.getsource(default_num_workers)
    assert _confirm_start(_HangWindow(), dry_run=True) is True
    body = inspect.getsource(train_directory)
    assert body.index("write_heartbeat(") < body.index("view.open()")
    assert "dump_traceback_later" in inspect.getsource(arm_dry_run_watchdog)
    arm_dry_run_watchdog(60.0)
    disarm_dry_run_watchdog()
    print("dry-run does not wait for Start")


def _corrupt(payload: bytes) -> bytes:
    return payload.replace(b"\n", b"\r\n")


def check_spawn_reads_repaired() -> None:
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 40
    colour[0, 0] = (8, 16, 24)
    encoded = _png_bytes(colour)
    assert encoded is not None
    expected = compose_strip({"main": colour})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw" / "main" / "000000.png"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(_corrupt(encoded))
        assert raw.read_bytes().startswith(b"\x89PNG\r\r\n\x1a")
        assert read_image_file(raw) is None
        repaired = root / "raw_repaired" / "main" / "000000.png"
        repaired.parent.mkdir(parents=True)
        repaired.write_bytes(encoded)
        loaded = load_training_bgr(root, 0)
        assert np.array_equal(loaded, expected)
        ctx = multiprocessing.get_context("spawn")
        os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        queue = ctx.Queue()
        proc = ctx.Process(target=_spawn_load, args=(str(root), 0, queue))
        proc.start()
        try:
            shape, pixel = queue.get(timeout=90)
        finally:
            proc.join(20)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
        assert proc.exitcode == 0, proc.exitcode
        assert shape == tuple(int(v) for v in expected.shape)
        assert pixel == int(expected.reshape(-1)[1])
        assert raw.read_bytes().startswith(b"\x89PNG\r\r\n\x1a")
    print(f"spawn loader shape={shape}")


def main() -> None:
    check_windows_memory_status()
    check_workers_and_dry_run_start()
    check_spawn_reads_repaired()
    print("scene loader ok")


if __name__ == "__main__":
    main()
