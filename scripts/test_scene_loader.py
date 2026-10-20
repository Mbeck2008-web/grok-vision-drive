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

from python.data.raw_frames import (  # noqa: E402
    _png_bytes,
    load_training_bgr,
    read_image_file,
    training_frame_bytes,
)
from python.data.strip_writer import compose_strip  # noqa: E402
from python.train.continual import (  # noqa: E402
    choose_chunk_length,
    chunk_vram_budget,
    remaining_until_holdout_can_stop,
    should_reset_state,
)
from python.train.fit import (  # noqa: E402
    _windows_memory,
    bytes_per_strip,
    largest_batch,
    read_free_ram_bytes,
    read_used_ram_bytes,
    step_bytes,
    usable_bytes,
)
from python.train.scene_loader import _spawn_load, default_num_workers, select_training_rows  # noqa: E402
from python.train.scene_net import MEMORY_S  # noqa: E402
from python.train.train_scene import (  # noqa: E402
    _confirm_start,
    arm_dry_run_watchdog,
    disarm_dry_run_watchdog,
    resolve_panel,
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


def _row(index: int, t: float, dt: float | None) -> dict:
    return {
        "i": index,
        "t": t,
        "dt_s": dt,
        "session": "rec",
        "cams": {"main": {"file": f"raw/main/{index:06d}.png"}},
    }


def check_leading_rows_without_frames() -> None:
    """The first state rows can be older than the first PNG. They are not a crash."""
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 50
    encoded = _png_bytes(colour)
    assert encoded is not None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        main = root / "raw" / "main"
        main.mkdir(parents=True)
        # Indices 0 and 1 were recorded before any camera file existed.
        (main / "000002.png").write_bytes(encoded)
        corrupt = encoded.replace(b"\n", b"\r\n")
        (main / "000003.png").write_bytes(corrupt)
        repaired = root / "raw_repaired" / "main"
        repaired.mkdir(parents=True)
        (repaired / "000003.png").write_bytes(encoded)
        rows = [
            _row(0, 0.0, None),
            _row(1, 0.1, 0.1),
            _row(2, 10.0, 0.1),
            _row(3, 40.0, 0.1),
        ]
        try:
            load_training_bgr(root, 0)
        except FileNotFoundError as exc:
            assert "000000" in str(exc)
            assert "raw" in str(exc)
        else:
            raise AssertionError("index 0 has no frame")
        kept, missing, excluded = select_training_rows(root, rows)
        assert missing == 2 and excluded == 0
        assert [int(row["i"]) for row in kept] == [2, 3]
        assert kept[0]["dt_s"] is None
        assert abs(float(kept[1]["dt_s"]) - 30.0) < 1e-9
        assert should_reset_state("rec", "rec", kept[1]["dt_s"], memory_s=MEMORY_S)
        assert [int(row["i"]) for row in kept[:4]] == [2, 3]
        loaded = load_training_bgr(root, 3)
        assert loaded.shape[2] == 3
        assert read_image_file(main / "000003.png") is None
    print("leading rows without frames skipped")


def check_frames_counted_once() -> None:
    colour = np.full((6, 8, 3), 9, dtype=np.uint8)
    encoded = _png_bytes(colour)
    assert encoded is not None
    other = np.full((20, 30, 3), 3, dtype=np.uint8)
    second = _png_bytes(other)
    assert second is not None and len(second) != len(encoded)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw" / "main"
        raw.mkdir(parents=True)
        fixed = root / "raw_repaired" / "main"
        fixed.mkdir(parents=True)
        (raw / "000000.png").write_bytes(encoded.replace(b"\n", b"\r\n"))
        (fixed / "000000.png").write_bytes(encoded)
        (raw / "000001.png").write_bytes(second)
        (fixed / "000001.png").write_bytes(encoded)
        (root / "state.jsonl").write_text("{}\n", encoding="utf-8")
        total, frames = training_frame_bytes(root)
        assert frames == 2
        assert total == len(encoded) + len(second) + len("{}\n".encode("utf-8"))
    print(f"frames counted once bytes={total}")


def check_panel_flags_and_vram_room() -> None:
    import python.train.train_scene as trainer

    real_display = trainer.interactive_display
    real_headless = trainer.headless_requested
    try:
        trainer.interactive_display = lambda: True
        trainer.headless_requested = lambda: False
        assert resolve_panel(no_panel=False, autostart=False) == (True, False)
        assert resolve_panel(no_panel=True, autostart=False) == (False, True)
        assert resolve_panel(no_panel=False, autostart=True) == (True, True)
        trainer.headless_requested = lambda: True
        assert resolve_panel(no_panel=False, autostart=False) == (False, True)
        trainer.headless_requested = lambda: False
        trainer.interactive_display = lambda: False
        assert resolve_panel(no_panel=False, autostart=False) == (False, True)
    finally:
        trainer.interactive_display = real_display
        trainer.headless_requested = real_headless
    assert _confirm_start(_HangWindow(), dry_run=False, autostart=True) is True
    assert _confirm_start(_HangWindow(), dry_run=False, sanity=True) is True
    # Rest of this pass, plus the forced second pass, plus two passes of the other rate.
    # 435 left in a 608-step epoch, one learning rate still waiting.
    job = remaining_until_holdout_can_stop(
        435,
        608,
        epoch=0,
        has_holdout=True,
        rates_after=1,
    )
    assert job == 435 + 608 + 2 * 608
    assert remaining_until_holdout_can_stop(10, 40, epoch=1, has_holdout=True, rates_after=0) == 10

    # 1080 Ti, about 7.9 GiB free, FP32, one repaired copy of the 45-minute set.
    free = int(7.9 * (1024 ** 3))
    file_bytes = 8_500_000_000
    n_strips = 4726
    per = bytes_per_strip(file_bytes, n_strips)
    chunk = choose_chunk_length(per, chunk_vram_budget(free))
    one = step_bytes(1, file_bytes=file_bytes, n_strips=n_strips, window_strips=chunk)
    room = largest_batch(
        usable_bytes(free),
        file_bytes=file_bytes,
        n_strips=n_strips,
        window_strips=chunk,
        limit=32,
    )
    assert chunk == 16
    assert one < usable_bytes(free)
    assert room > 1
    print(f"1080 Ti FP32 chunk={chunk} step={one} parallel_fit={room}")


def main() -> None:
    check_windows_memory_status()
    check_workers_and_dry_run_start()
    check_spawn_reads_repaired()
    check_leading_rows_without_frames()
    check_frames_counted_once()
    check_panel_flags_and_vram_room()
    print("scene loader ok")


if __name__ == "__main__":
    main()
