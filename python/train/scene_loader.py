"""Load one training canvas. The scene net steps frames in order.

Windows starts extra processes with spawn. A spawned worker re-imports the
trainer while the parent already owns the CUDA context and the OpenCV
window, and that combination stalls. The default worker count stays 0 on
every OS for that reason, and on Windows in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

from python.data.raw_frames import load_training_bgr, row_has_decodable_frame


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


def _stamp(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if stamp != stamp or stamp in (float("inf"), float("-inf")):
        return None
    return stamp


def select_training_rows(
    root: Path | str,
    rows: list,
) -> tuple[list, int, int]:
    """Keep rows that have a decodable strip camera. Count the rest as missing.

    Stationary, tipped, and damage spans are a third count and are not
    trained. ``dt_s`` is rewritten from the timestamps of the kept rows. A
    hole where frames were never saved becomes one gap, and a gap longer
    than the GRU memory clears the state. The original state file is not
    rewritten.
    """
    from python.collect.liveness import bad_row_reasons

    bad = bad_row_reasons([row for row in rows if isinstance(row, dict)])
    kept: list = []
    missing = 0
    excluded = 0
    prev_t: float | None = None
    for row in rows:
        if not isinstance(row, dict):
            missing += 1
            continue
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            index = None
        if index is not None and index in bad:
            excluded += 1
            continue
        if not row_has_decodable_frame(root, row):
            missing += 1
            continue
        item = dict(row)
        stamp = _stamp(item.get("t"))
        if prev_t is None or stamp is None:
            item["dt_s"] = None
        else:
            item["dt_s"] = stamp - prev_t
        kept.append(item)
        if stamp is not None:
            prev_t = stamp
    return kept, missing, excluded


def _spawn_load(root: str, index: int, queue) -> None:
    """Entry point for a spawned process. Imports nothing from the parent."""
    image = load_canvas(root, index)
    queue.put(
        (
            tuple(int(v) for v in image.shape),
            int(image.reshape(-1)[1] if image.size > 1 else image.reshape(-1)[0]),
        )
    )
