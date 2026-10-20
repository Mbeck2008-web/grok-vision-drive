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
    """Prefetch workers for a window DataLoader.

    Windows defaults to 4 so PNG decode is not on the GPU thread.
    Other hosts stay in-process: a spawned worker that re-imports this
    package while the parent holds CUDA and the status window can stall.
    ``--workers`` overrides the default. ``persistent_workers`` stays on
    for the whole epoch when the count is above zero.
    """
    if sys.platform == "win32":
        return 4
    return 0


def loader_settings(workers: int, *, cuda: bool) -> dict[str, int | bool]:
    """DataLoader kwargs. ``pin_memory`` only helps a CUDA copy."""
    count = max(0, int(workers))
    return {
        "num_workers": count,
        "persistent_workers": count > 0,
        "pin_memory": bool(cuda) and count > 0,
    }


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
    cams: tuple[str, ...] | list[str] | None = None,
    dropped: dict | None = None,
) -> tuple[list, int, int]:
    """Keep rows that have a decodable strip camera. Count the rest as missing.

    ``cams`` is the trained camera list (see ``row_has_decodable_frame``):
    a row whose listed cameras, or whose ``main`` picture, did not decode is
    also left out. Those rows are counted in ``missing`` and, when
    ``dropped`` is a dict, in ``dropped["no_trained_cam"]``.

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
        if cams and not row_has_decodable_frame(root, row, cams):
            missing += 1
            if dropped is not None:
                dropped["no_trained_cam"] = int(dropped.get("no_trained_cam", 0)) + 1
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
