"""Camera-only runtime: per-camera images (+ their own timestamps) -> heads -> downstream.

``perceive`` is the only stage that touches images. Ego motion for the
temporal warp comes from the net's camera-only VO head by default. Wheel
speed / IMU yaw rate are read only when ``profile.ego_from_vehicle`` is on
(the ``--ego-from-vehicle`` flag, default off; those signals still need
GVD's confirmation). No ground-truth pose, scenetree or sim-truth field is
read here: ``perceive`` takes scalars, never a state dict. Not wired into
``run_vision.py`` in this branch.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from python.occ.camera import CameraModel
from python.occ.downstream import Downstream
from python.occ.model import OccSparseNet, ego_motion


class OccRuntime:
    def __init__(self, model: OccSparseNet, cams: Mapping[str, CameraModel], *, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.cams = dict(cams)
        self.device = device
        self.profile = model.profile
        self.dtype = torch.float16 if self.profile.infer_dtype == "fp16" else torch.float32
        if self.dtype == torch.float16:
            self.model.half()
        self.memory = None
        self.t: float | None = None
        self.resets = 0
        self.downstream = Downstream(model.grid, self.cams)

    def _prep(self, cid: str, arr: Any) -> torch.Tensor:
        a = np.asarray(arr)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 16 or a.shape[1] < 16:
            raise ValueError(f"{cid}: need HxWx3 uint8 (>=16 px), got {a.dtype} {a.shape}")
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255.0
        ls = self.profile.long_side(cid)
        s = ls / max(a.shape[0], a.shape[1])
        if abs(s - 1.0) > 1e-6:  # resolution knob: first lever before giving up same-tick sync
            x = F.interpolate(x, size=(max(16, round(a.shape[0] * s)), max(16, round(a.shape[1] * s))), mode="bilinear", antialias=True, align_corners=False)
        return x.to(self.device, self.dtype)

    def perceive(self, t: float, images: Mapping[str, tuple[Any, float]], *, speed_mps: float | None = None, yaw_rate: float | None = None) -> dict[str, Any]:
        """``images``: cam id -> (HxWx3 uint8 RGB, that camera's own timestamp).

        Returns head outputs only (no image, no tensor).
        """
        fresh_t = isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(float(t))
        dt = None if (self.t is None or not fresh_t) else float(t) - self.t
        if not fresh_t or dt is None or dt <= 0.0 or dt > self.profile.max_gap_s:
            if self.memory is not None:
                self.resets += 1
            self.memory = None
        motion = None
        if self.profile.ego_from_vehicle and self.memory is not None:
            if speed_mps is not None and yaw_rate is not None and math.isfinite(speed_mps) and math.isfinite(yaw_rate):
                motion = ego_motion(float(speed_mps), float(yaw_rate), float(dt))
            else:
                self.memory = None
        imgs, ages = {}, {}
        for cid, item in images.items():
            if cid not in self.cams or cid not in self.profile.cams:
                continue
            arr, t_cam = item
            imgs[cid] = self._prep(cid, arr)
            try:
                ages[cid] = float(t) - float(t_cam) if fresh_t else math.nan
            except (TypeError, ValueError):
                ages[cid] = math.nan
        res = self.model.infer(imgs, self.cams, ages, motion=motion, memory=self.memory)
        self.memory = res["memory"]
        self.t = float(t) if fresh_t else None
        return {"t": float(t) if fresh_t else 0.0, "cells": res["cells"], "objects": res["objects"], "motion_used": res["motion_used"],
                "stats": res["stats"], "vo": res["vo"], "vo_conf": res["vo_conf"], "vo_cov": res["vo_cov"], "memory_pts": res["memory_pts"],
                "cam_panels": res["cam_panels"], "timing_ms": res["timing_ms"], "ego_from_vehicle": bool(self.profile.ego_from_vehicle)}

    def step(self, t: float, images: Mapping[str, tuple[Any, float]], *, speed_mps: float | None = None, yaw_rate: float | None = None, engaged: bool = False) -> dict[str, Any]:
        heads = self.perceive(t, images, speed_mps=speed_mps, yaw_rate=yaw_rate)
        if self.profile.ego_from_vehicle and speed_mps is not None:
            heads["speed_mps"] = speed_mps
        heads["engaged"] = engaged
        out = self.downstream(heads)
        out["heads"] = heads
        return out
