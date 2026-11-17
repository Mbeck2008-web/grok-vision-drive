"""Camera-only runtime step: images + ego kinematics -> occupancy outputs -> viz v1.

Imports nothing that reads depth, semantics, boxes or recorder rows. Not
wired into ``run_vision.py`` in this branch (opt-in integration is a follow-up).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from python.occ.camera import CameraModel
from python.occ.model import OccSparseNet
from python.occ.objects import objects_from_cells
from python.occ.polylines import edges_from_cells
from python.occ.tracks import MotionTracker
from python.occ.viz_adapter import occ_to_viz


class OccRuntime:
    def __init__(self, model: OccSparseNet, cams: Mapping[str, CameraModel], *, fp16: bool = False, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.cams = dict(cams)
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            self.model.half()
        self.tracker = MotionTracker()
        self.seq = 0

    def step(self, t: float, images: Mapping[str, np.ndarray], *, speed_mps: float, yaw_rate: float = 0.0, engaged: bool = False) -> dict[str, Any]:
        """``images``: cam id -> HxWx3 uint8 RGB. Returns ``{"outputs", "viz"}``."""
        imgs = {}
        for cid, arr in images.items():
            a = np.asarray(arr)
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
                raise ValueError(f"{cid}: need HxWx3 uint8, got {a.dtype} {a.shape}")
            imgs[cid] = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(self.device, self.dtype) / 255.0
        res = self.model.infer(imgs, self.cams)
        cells = res["cells"]
        grid = self.model.grid
        objs = objects_from_cells(cells["idx"], cells["cls"], cells["conf"], grid=grid)
        tracks = self.tracker.update(t, objs, speed_mps=speed_mps, yaw_rate=yaw_rate)
        edges = edges_from_cells(cells["idx"], cells["cls"], grid=grid)
        outputs = {"t": t, "cells": cells, "tracks": tracks, "edges": edges, "stats": res["stats"]}
        viz = occ_to_viz(outputs, seq=self.seq, ego={"engaged": engaged, "speed_mps": speed_mps}, grid=grid)
        self.seq += 1
        return {"outputs": outputs, "viz": viz}
