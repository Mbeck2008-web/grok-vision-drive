"""Holdout depth AbsRel and semantic mIoU for the Trainer 2.0 curriculum (stage 1 exit metrics).

Pooled over every scored holdout tick, on the 12x16 train-only aux grid the aux heads
predict (the same grid and validity masks the aux loss uses). The sums are plain
numpy so they can be tested without torch.

* ``depth_absrel``: mean of ``|pred - gt| / gt`` over valid cells with ``gt > 0``.
* ``sem_miou``: mean over classes present in the target or the prediction (union > 0)
  of ``intersection / union``, from one confusion pooled over valid (``sem_id > 0``) cells.

Neither is measured for a ``depth_hires`` prediction (a different grid): ``update`` is
simply not called for one, and the metric then stays unmeasured (``None``).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class AuxEval:
    def __init__(self) -> None:
        self.rel_sum = 0.0
        self.rel_n = 0
        self.inter: dict[int, int] = {}
        self.union: dict[int, int] = {}
        self.sem_n = 0

    def reset(self) -> None:
        self.__init__()

    def update_depth(self, pred_m: Any, gt_m: Any, valid: Any) -> None:
        p = np.asarray(pred_m, dtype=np.float64).reshape(-1)
        g = np.asarray(gt_m, dtype=np.float64).reshape(-1)
        v = np.asarray(valid, dtype=np.float64).reshape(-1)
        if not (p.shape == g.shape == v.shape):
            raise ValueError(f"depth eval shapes differ: pred {p.shape} gt {g.shape} valid {v.shape}")
        keep = (v > 0.5) & (g > 0.0) & np.isfinite(g)
        if not bool(keep.any()):
            return
        rel = np.abs(p[keep] - g[keep]) / g[keep]
        self.rel_sum += float(rel.sum())
        self.rel_n += int(keep.sum())

    def update_sem(self, logits: Any, gt_ids: Any, valid: Any) -> None:
        lg = np.asarray(logits)
        if lg.ndim == 4:
            lg = lg[0]
        pred = lg.argmax(axis=0).reshape(-1)
        g = np.asarray(gt_ids).reshape(-1)
        v = np.asarray(valid, dtype=np.float64).reshape(-1)
        if not (pred.shape == g.shape == v.shape):
            raise ValueError(f"sem eval shapes differ: pred {pred.shape} gt {g.shape} valid {v.shape}")
        keep = v > 0.5
        if not bool(keep.any()):
            return
        p, t = pred[keep], g[keep]
        for cls in np.union1d(np.unique(p), np.unique(t)).tolist():
            pc, tc = p == cls, t == cls
            self.inter[int(cls)] = self.inter.get(int(cls), 0) + int((pc & tc).sum())
            self.union[int(cls)] = self.union.get(int(cls), 0) + int((pc | tc).sum())
        self.sem_n += int(keep.sum())

    def absrel(self) -> float | None:
        return None if self.rel_n <= 0 else self.rel_sum / self.rel_n

    def miou(self) -> float | None:
        classes = [c for c, u in self.union.items() if u > 0]
        if not classes:
            return None
        return float(np.mean([self.inter.get(c, 0) / self.union[c] for c in classes]))
