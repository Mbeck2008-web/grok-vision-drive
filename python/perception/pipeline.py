"""Modular perception → planning tick for GVD M2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from python.perception.cipv import select_cipv
from python.perception.detect import STATIC_CLASSES
from python.perception.lanes import estimate_lanes
from python.perception.track import IoUTracker
from python.planning.path_predictor import predict_path

MAX_SIGNS = 8
SIGN_KEEP = 8
SIGN_MATCH_M = 3.0


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
    def __init__(self, *, allow_synthetic: bool = True, detector_id: str = "auto") -> None:
        from python.runtime.models import load_detector

        self.detector, self.missing = load_detector(detector_id, allow_synthetic=allow_synthetic)
        self.tracker = IoUTracker()
        self._prior_lanes: list = []
        self._prior_signs: list[dict[str, Any]] = []

    def set_detector(self, detector: Any, missing: list[str] | None = None) -> None:
        """Swap the live detector; reset tracks so CIPV ids do not stick to a dead net."""
        self.detector = detector
        self.missing = list(missing or [])
        self.tracker = IoUTracker()
        self._prior_lanes = []
        self._prior_signs = []

    def tick(
        self,
        main_bgr: np.ndarray | None,
        *,
        ego_speed_mps: float = 0.0,
        steer_deg: float = 0.0,
    ) -> PerceptionOut:
        # The wheel is for the override residual and the ego display. The route
        # is predicted from lanes, vehicles, and signs, then followed.
        del steer_deg
        t0 = time.perf_counter()
        dets = self.detector.detect(main_bgr)
        # Road furniture is detected but never tracked: tracks feed CIPV / AEB / ghosts,
        # and a stop sign is not a lead vehicle. It rides to the UI as `signs` instead.
        moving = [d for d in dets if d.cls not in STATIC_CLASSES]
        fresh_signs = []
        for d in dets:
            if d.cls not in STATIC_CLASSES:
                continue
            lamp = getattr(d, "state", None)
            if d.cls == "traffic_light" and lamp in (None, ""):
                lamp = "unknown"
            fresh_signs.append({
                "cls": d.cls,
                "x": round(float(d.x), 2),
                "y": round(float(d.y), 2),
                "conf": round(float(d.conf), 2),
                "state": lamp,
                "partial": bool(getattr(d, "partial", False)),
                "misses": int(getattr(d, "misses", 0) or 0),
            })
        signs = _hold_signs(self._prior_signs, fresh_signs)[:MAX_SIGNS]
        self._prior_signs = list(signs)
        # if detector didn't set ego x/y (onnx path does), leave as-is
        tracks = self.tracker.update(moving, time.time())
        track_dicts = self.tracker.as_dicts()
        lanes = estimate_lanes(main_bgr)
        prior_lanes = None
        if lanes.lanes_bev:
            self._prior_lanes = lanes.lanes_bev
        elif self._prior_lanes:
            prior_lanes = self._prior_lanes
        cipv = select_cipv(track_dicts, path_width=2.0, ego_speed_mps=ego_speed_mps)
        plan = predict_path(
            lanes_bev=lanes.lanes_bev,
            lane_conf=lanes.conf,
            curvature=lanes.curvature,
            tracks=track_dicts,
            signs=signs,
            ego_speed_mps=ego_speed_mps,
            prior_lanes=prior_lanes,
        )
        cipv_id = plan.cipv_id
        if cipv_id is None and cipv.track is not None:
            cipv_id = cipv.track.get("id")
        ttc = plan.ttc_lead if plan.ttc_lead is not None else cipv.ttc_lead
        aeb = plan.aeb
        target_v = plan.target_v
        if cipv.aeb == "brake" and aeb != "brake":
            aeb = "brake"
            target_v = 0.0
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
            path_ego=plan.path_ego,
            path_width=plan.path_width,
            path_conf=plan.path_conf,
            path_debug_preview=not plan.drivable,
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
                "corridor_width": plan.path_width,
                "curvature": plan.curvature,
                "target_v": target_v,
                "ttc_lead": ttc,
                "aeb": aeb,
                "cipv_id": cipv_id,
                "stop_reason": plan.stop_reason,
                "prediction": plan.prediction,
                "path_length_m": plan.path_length_m,
                "pred_brake": plan.pred_brake,
                "blinker": plan.blinker,
            },
            infer_ms=infer_ms,
            missing=missing,
            detector_name=self.detector.name,
        )


def _hold_signs(prior: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a sign that dropped out for a few frames, including a red lamp.

    A fresh light with no colour does not wipe a red we already held.
    ``misses`` counts frames since the last real detection. ``partial`` marks
    a sign we are only remembering.
    """
    used = [False] * len(prior)
    out: list[dict[str, Any]] = []
    for sign in fresh:
        best_i = None
        best_d = SIGN_MATCH_M
        for i, old in enumerate(prior):
            if used[i] or str(old.get("cls")) != str(sign.get("cls")):
                continue
            dx = float(old.get("x") or 0.0) - float(sign.get("x") or 0.0)
            dy = float(old.get("y") or 0.0) - float(sign.get("y") or 0.0)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_d:
                best_i, best_d = i, dist
        state = sign.get("state")
        if best_i is not None:
            used[best_i] = True
            held = prior[best_i].get("state")
            if str(sign.get("cls")) == "traffic_light" and state in (None, "", "unknown"):
                if held not in (None, "", "unknown"):
                    state = held
        out.append({
            "cls": sign.get("cls"),
            "x": sign.get("x"),
            "y": sign.get("y"),
            "conf": sign.get("conf"),
            "state": state,
            "partial": bool(sign.get("partial")),
            "misses": 0,
        })
    for i, old in enumerate(prior):
        if used[i]:
            continue
        misses = int(old.get("misses") or 0) + 1
        if misses > SIGN_KEEP:
            continue
        kept = dict(old)
        kept["misses"] = misses
        kept["partial"] = True
        out.append(kept)
    return out
