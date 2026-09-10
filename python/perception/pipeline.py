"""Modular perception → planning tick for GVD M2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from python.perception.cipv import select_cipv
from python.perception.detect import STATIC_CLASSES, make_detector
from python.perception.lanes import estimate_lanes
from python.perception.track import IoUTracker
from python.planning.corridor import build_path_ego
from python.planning.speed import plan_speed

MAX_SIGNS = 8


@dataclass
class PerceptionOut:
    tracks: list[dict[str, Any]] = field(default_factory=list)
    objects_n: int = 0
    tracks_n: int = 0
    lane_conf: float = 0.0
    lanes_bev: list = field(default_factory=list)
    path_ego: list[dict[str, float]] = field(default_factory=list)
    path_width: float = 2.0
    path_conf: float = 0.0
    path_debug_preview: bool = True
    planner: dict[str, Any] = field(default_factory=dict)
    signs: list[dict[str, Any]] = field(default_factory=list)
    dets: list[dict[str, Any]] = field(default_factory=list)
    infer_ms: float = 0.0
    missing: list[str] = field(default_factory=list)
    detector_name: str = ""


class ModularPerception:
    def __init__(self, *, allow_synthetic: bool = True) -> None:
        self.detector, self.missing = make_detector(allow_synthetic=allow_synthetic)
        self.tracker = IoUTracker()

    def tick(
        self,
        main_bgr: np.ndarray | None,
        *,
        ego_speed_mps: float = 0.0,
        steer_deg: float = 0.0,
    ) -> PerceptionOut:
        t0 = time.perf_counter()
        dets = self.detector.detect(main_bgr)
        # Road furniture is detected but never tracked: tracks feed CIPV / AEB / ghosts,
        # and a stop sign is not a lead vehicle. It rides to the UI as `signs` instead.
        moving = [d for d in dets if d.cls not in STATIC_CLASSES]
        signs = [
            {
                "cls": d.cls,
                "x": round(float(d.x), 2),
                "y": round(float(d.y), 2),
                "conf": round(float(d.conf), 2),
                # No colour classifier in the stack yet — never guess the aspect.
                "state": "unknown" if d.cls == "traffic_light" else None,
            }
            for d in dets
            if d.cls in STATIC_CLASSES
        ][:MAX_SIGNS]
        # if detector didn't set ego x/y (onnx path does), leave as-is
        tracks = self.tracker.update(moving, time.time())
        track_dicts = self.tracker.as_dicts()
        lanes = estimate_lanes(main_bgr)
        cipv = select_cipv(track_dicts, path_width=2.0, ego_speed_mps=ego_speed_mps)
        corridor = build_path_ego(
            lanes_bev=lanes.lanes_bev,
            lane_conf=lanes.conf,
            curvature=lanes.curvature,
            cipv=cipv.track,
            steer_deg=steer_deg,
        )
        speed = plan_speed(
            ego_speed_mps=ego_speed_mps,
            curvature=corridor.curvature,
            ttc_lead=cipv.ttc_lead,
            aeb=cipv.aeb,
        )
        missing = list(self.missing)
        if main_bgr is None:
            missing.append("cam_main_frame")
        if lanes.conf <= 0:
            missing.append("lanes")
        # shrink missing when synthetic/weights work
        if self.detector.name != "empty" and "yolo_weights" in missing and self.detector.name == "synthetic":
            pass  # keep yolo_weights listed — honest
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return PerceptionOut(
            tracks=track_dicts,
            objects_n=len(dets),
            tracks_n=len(track_dicts),
            lane_conf=lanes.conf,
            lanes_bev=lanes.lanes_bev,
            path_ego=corridor.path_ego,
            path_width=corridor.path_width,
            path_conf=corridor.path_conf,
            path_debug_preview=not corridor.from_planner,
            signs=signs,
            dets=[
                {
                    "cls": d.cls,
                    "conf": round(float(d.conf), 2),
                    "xyxy": [round(float(v), 1) for v in d.xyxy],
                }
                for d in dets
            ],
            planner={
                "corridor_width": corridor.path_width,
                "curvature": corridor.curvature,
                "target_v": speed.target_v,
                "ttc_lead": speed.ttc_lead,
                "aeb": speed.aeb,
                "cipv_id": (cipv.track or {}).get("id"),
            },
            infer_ms=infer_ms,
            missing=missing,
            detector_name=self.detector.name,
        )
