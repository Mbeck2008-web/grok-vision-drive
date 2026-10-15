"""Load one training canvas. The scene net steps frames in order.

Windows starts extra processes with spawn. A spawned worker re-imports the
trainer while the parent already owns the CUDA context and the OpenCV
window, and that combination stalls. The default worker count stays 0 on
every OS for that reason, and on Windows in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

from python.data.raw_frames import load_training_bgr


def default_num_workers() -> int:
    """In-process loading.

    ``sys.platform == "win32"`` is called out because spawn is the only
    start method there. A non-zero count deadlocks behind CUDA and the
    status window. The GRU also consumes frames in file order, so more
    workers would not be a batch map.
    """
    if sys.platform == "win32":
        return 0
    return 0


def load_canvas(root: Path | str, index: int):
    """Stitch this tick. A text-mode PNG is read from ``raw_repaired``."""
    return load_training_bgr(root, index)


def _spawn_load(root: str, index: int, queue) -> None:
    """Entry point for a spawned process. Imports nothing from the parent."""
    image = load_canvas(root, index)
    queue.put(
        (
            tuple(int(v) for v in image.shape),
            int(image.reshape(-1)[1] if image.size > 1 else image.reshape(-1)[0]),
        )
    )
