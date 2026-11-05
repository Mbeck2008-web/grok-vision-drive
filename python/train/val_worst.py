"""The N worst holdout frames of one val pass, by per-frame loss.

Read-only over values the val pass already computed: the per-frame loss,
the lane_gt prediction and the row's labels. Memory is a heap of at most N
entries. One compact JSON per epoch, plus optional top-down PNG thumbnails of
prediction vs GT lanes.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_N = 16
MAX_N = 256  # keeps one epoch's JSON in the tens of KB


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _vec(values: Any) -> list[float | None] | None:
    if values is None:
        return None
    return [_num(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


class WorstFrames:
    """Keep the ``n`` highest-loss frames. Ties keep the earlier frame."""

    def __init__(self, n: int) -> None:
        self.n = max(0, min(int(n), MAX_N))
        self._heap: list[tuple[float, int, dict[str, Any]]] = []
        self._seen = 0

    def note(self, loss: float, row: Mapping[str, Any], pred: Any, gt: Any, gt_mask: Any) -> None:
        if self.n <= 0:
            return
        value = float(loss)
        self._seen += 1
        # -seen: on equal loss the earlier frame ranks higher and stays.
        key = (value, -self._seen)
        if len(self._heap) >= self.n and key <= self._heap[0][:2]:
            return
        tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else {}
        item = {
            "loss": _num(value),
            "i": row.get("i"),
            "t": _num(row.get("t")),
            "segment": row.get("segment"),
            "gt_origin": str(tech.get("gt_origin") or "unknown"),
            "pred": _vec(pred),
            "gt": _vec(gt),
            "gt_valid": _vec(gt_mask),
        }
        entry = (value, -self._seen, item)
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heapreplace(self._heap, entry)

    def ranked(self) -> list[dict[str, Any]]:
        return [item for _v, _s, item in sorted(self._heap, key=lambda e: (-e[0], -e[1]))]

    def write(
        self,
        out_dir: Path | str,
        *,
        run: str,
        epoch: int,
        step: int,
        recording: str,
        cams: tuple[str, ...] | None,
        png: bool = False,
    ) -> Path | None:
        if self.n <= 0:
            return None
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"val_worst.run{run}.ep{int(epoch)}"
        frames = self.ranked()
        camera = ",".join(cams) if cams else "all"
        for rank, item in enumerate(frames):
            item["rank"] = rank
            item["recording"] = recording
            item["camera"] = camera
        payload = {
            "kind": "val_worst",
            "run": run,
            "epoch": int(epoch),
            "step": int(step),
            "n": len(frames),
            "scored": self._seen,
            "frames": frames,
        }
        if png:
            thumbs = out / stem
            thumbs.mkdir(exist_ok=True)
            for item in frames:
                name = f"rank{item['rank']:03d}_i{item['i']}.png"
                if write_thumbnail(thumbs / name, item.get("pred"), item.get("gt"), item.get("gt_valid")):
                    item["png"] = f"{stem}/{name}"
        path = out / f"{stem}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        return path


def write_thumbnail(path: Path, pred: Any, gt: Any, gt_valid: Any, size: tuple[int, int] = (120, 160)) -> bool:
    """Top-down lanes: GT green, prediction red; x -6..6 m, y 0..32 m ahead."""
    import cv2

    from python.debug.lane_targets import LOOKAHEADS_M, SLOT_CENTER, SLOT_LEFT, SLOT_RIGHT, lookahead_slot

    w, h = size
    img = np.full((h, w, 3), 24, dtype=np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round((x + 6.0) / 12.0 * (w - 1))), int(round((1.0 - y / 32.0) * (h - 1)))

    def draw(vec: Any, color: tuple[int, int, int], valid: Any = None) -> None:
        if vec is None:
            return
        for part in (SLOT_LEFT, SLOT_CENTER, SLOT_RIGHT):
            pts = []
            for k, y in enumerate(LOOKAHEADS_M):
                j = lookahead_slot(k, part)
                x = vec[j] if j < len(vec) else None
                ok = valid is None or (j < len(valid) and (valid[j] or 0) > 0.5)
                if x is not None and ok:
                    pts.append(px(float(x), float(y)))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, color, 1, cv2.LINE_AA)

    draw(gt, (80, 220, 80), gt_valid)
    draw(pred, (60, 60, 240))
    return bool(cv2.imwrite(str(path), img))
