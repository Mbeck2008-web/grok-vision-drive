"""Pick a scene-training batch and device before the first step.

The driving file is the strip directory. Its size and the free VRAM decide
how many windows a step may hold. The batch shrinks until that step fits.
CPU is only the path left when the card cannot hold one step and system RAM
is larger, and that path is not recommended.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from python.data.strip_writer import CANVAS_H, CANVAS_W

# FP32 working set for one strip. Eight sectors each keep the decoded crop and
# four stride-2 maps, about 26 times the 192×2272 canvas. A recorded file
# larger than that canvas costs more. Mixed precision is not assumed here, so
# a Pascal card is not handed a batch that only fits in FP16.
STRIP_LIVE_COPIES = 26
# Weights, gradients, and Adam moments, with a little headroom.
MODEL_OVERHEAD_BYTES = 64 * 1024 * 1024
# A tenth of free memory stays with the allocator.
USABLE_NUM = 9
USABLE_DEN = 10

_FIT_FAIL = "the training step fits neither free VRAM nor a larger pool of CPU RAM"


class TrainFitError(RuntimeError):
    """The measured step does not fit in VRAM or in a larger pool of CPU RAM."""


@dataclass(frozen=True)
class FitPlan:
    device: str
    batch: int
    recommended: bool
    file_bytes: int
    n_strips: int
    free_vram_bytes: int
    free_ram_bytes: int
    window_strips: int
    step_bytes: int
    reason: str


def canvas_bytes() -> int:
    return int(CANVAS_H) * int(CANVAS_W) * 3


def bytes_per_strip(file_bytes: int, n_strips: int) -> int:
    measured = max(0, int(file_bytes)) // max(1, int(n_strips))
    return max(measured, canvas_bytes()) * STRIP_LIVE_COPIES


def step_bytes(batch: int, *, file_bytes: int, n_strips: int, window_strips: int) -> int:
    """Bytes one optimizer step is predicted to hold."""
    strips = max(1, int(batch)) * max(1, int(window_strips))
    return MODEL_OVERHEAD_BYTES + strips * bytes_per_strip(file_bytes, n_strips)


def usable_bytes(free_bytes: int) -> int:
    return max(0, int(free_bytes)) * USABLE_NUM // USABLE_DEN


def largest_batch(
    budget: int,
    *,
    file_bytes: int,
    n_strips: int,
    window_strips: int,
    limit: int,
) -> int:
    """Largest batch at least 1 whose step fits in ``budget``, or 0."""
    limit = max(0, int(limit))
    if limit < 1 or int(budget) <= 0:
        return 0
    per = max(1, int(window_strips)) * bytes_per_strip(file_bytes, n_strips)
    if MODEL_OVERHEAD_BYTES + per > int(budget):
        return 0
    room = (int(budget) - MODEL_OVERHEAD_BYTES) // per
    return max(0, min(limit, int(room)))


def shrink_batch(batch: int) -> int:
    """Next smaller batch after a step that did not fit. One shrinks to zero."""
    batch = int(batch)
    if batch <= 1:
        return 0
    return max(1, batch // 2)


def plan_fit(
    *,
    file_bytes: int,
    n_strips: int,
    free_vram_bytes: int,
    free_ram_bytes: int,
    n_windows: int,
    window_strips: int,
) -> FitPlan:
    """Choose device and batch from the driving file and free memory.

    A GPU batch that fits is recommended. CPU is returned only when batch 1
    does not fit in free VRAM and free CPU RAM is larger. That plan is marked
    not recommended.
    """
    limit = max(1, int(n_windows))
    strips = max(1, int(window_strips))
    common = dict(
        file_bytes=int(file_bytes),
        n_strips=max(1, int(n_strips)),
        free_vram_bytes=max(0, int(free_vram_bytes)),
        free_ram_bytes=max(0, int(free_ram_bytes)),
        window_strips=strips,
    )
    gpu_batch = 0
    if common["free_vram_bytes"] > 0:
        gpu_batch = largest_batch(
            usable_bytes(common["free_vram_bytes"]),
            file_bytes=common["file_bytes"],
            n_strips=common["n_strips"],
            window_strips=strips,
            limit=limit,
        )
    if gpu_batch >= 1:
        return FitPlan(
            device="cuda",
            batch=gpu_batch,
            recommended=True,
            step_bytes=step_bytes(gpu_batch, file_bytes=common["file_bytes"], n_strips=common["n_strips"], window_strips=strips),
            reason="gpu",
            **common,
        )
    if common["free_ram_bytes"] > common["free_vram_bytes"]:
        cpu_batch = largest_batch(
            usable_bytes(common["free_ram_bytes"]),
            file_bytes=common["file_bytes"],
            n_strips=common["n_strips"],
            window_strips=strips,
            limit=limit,
        )
        if cpu_batch >= 1:
            return FitPlan(
                device="cpu",
                batch=cpu_batch,
                recommended=False,
                step_bytes=step_bytes(cpu_batch, file_bytes=common["file_bytes"], n_strips=common["n_strips"], window_strips=strips),
                reason="cpu-not-recommended",
                **common,
            )
    raise TrainFitError(_FIT_FAIL)


def fit_with_probe(
    plan: FitPlan,
    probe: Callable[[str, int], bool],
    *,
    n_windows: int,
) -> FitPlan:
    """Shrink until ``probe(device, batch)`` says the step fits.

    Probe is the real check after the file and free VRAM have been measured.
    A GPU success stays recommended. When every GPU batch fails and CPU RAM is
    larger, the CPU plan is not recommended.
    """
    limit = max(1, int(n_windows))

    def accept(device: str, batch: int) -> FitPlan | None:
        batch = int(batch)
        while batch >= 1:
            if probe(device, batch):
                return FitPlan(
                    device=device,
                    batch=batch,
                    recommended=device == "cuda",
                    file_bytes=plan.file_bytes,
                    n_strips=plan.n_strips,
                    free_vram_bytes=plan.free_vram_bytes,
                    free_ram_bytes=plan.free_ram_bytes,
                    window_strips=plan.window_strips,
                    step_bytes=step_bytes(
                        batch,
                        file_bytes=plan.file_bytes,
                        n_strips=plan.n_strips,
                        window_strips=plan.window_strips,
                    ),
                    reason="gpu" if device == "cuda" else "cpu-not-recommended",
                )
            batch = shrink_batch(batch)
        return None

    if plan.device == "cuda":
        chosen = accept("cuda", plan.batch)
        if chosen is not None:
            return chosen
        if not (plan.free_ram_bytes > plan.free_vram_bytes):
            raise TrainFitError(_FIT_FAIL)
        cpu_batch = largest_batch(
            usable_bytes(plan.free_ram_bytes),
            file_bytes=plan.file_bytes,
            n_strips=plan.n_strips,
            window_strips=plan.window_strips,
            limit=limit,
        )
        if cpu_batch < 1:
            raise TrainFitError(_FIT_FAIL)
        chosen = accept("cpu", cpu_batch)
        if chosen is None:
            raise TrainFitError(_FIT_FAIL)
        return chosen

    chosen = accept(plan.device, plan.batch)
    if chosen is None:
        raise TrainFitError(_FIT_FAIL)
    return chosen


def measure_driving_file(root: Path | str) -> int:
    """Bytes on disk for the strip directory (meta, JPEGs, and state)."""
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return total


def _meminfo_kb(name: str) -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = name + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2:
                return int(float(parts[1]))
    return None


def read_free_ram_bytes() -> int:
    kb = _meminfo_kb("MemAvailable")
    if kb is None:
        return 0
    return max(0, kb) * 1024


def read_used_ram_bytes() -> int:
    total = _meminfo_kb("MemTotal")
    free = _meminfo_kb("MemAvailable")
    if total is None or free is None:
        return 0
    return max(0, total - free) * 1024


def _nvidia_mib(field: str) -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0]
        mib = float(line.split(",")[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0
    if mib < 0:
        return 0
    return int(mib * 1024 * 1024)


def read_free_vram_bytes() -> int:
    return _nvidia_mib("memory.free")


def read_used_vram_bytes() -> int:
    return _nvidia_mib("memory.used")
