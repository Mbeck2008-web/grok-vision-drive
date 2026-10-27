"""Modular perception → planning tick for GVD M2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from python.perception.cipv import select_cipv
from python.perception.detect import STATIC_CLASSES
from python.perception.lanes import LaneResult, lanes_from_view
from python.perception.track import IoUTracker
from python.planning.path_predictor import (
    LaneTrack,
    _expose_pedals,
    predict_path,
    reckon_ego_x,
    reckon_path,
)

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
    # Parts of infer_ms, every tick: the detector, and lanes + CIPV + planner (incl. AEB).
    detect_ms: float = 0.0
    plan_ms: float = 0.0
    missing: list[str] = field(default_factory=list)
    detector_name: str = ""
    lane_source: str = "none"
    # Seconds since the windshield/pillar pixels last changed. 0 on a new frame.
    frame_age_s: float = 0.0


class ModularPerception:
    def __init__(self, *, allow_synthetic: bool = True, detector_id: str = "auto") -> None:
        from python.runtime.models import load_detector

        self.detector, self.missing = load_detector(detector_id, allow_synthetic=allow_synthetic)
        self.tracker = IoUTracker()
        self._prior_lanes: list = []
        self._prior_signs: list[dict[str, Any]] = []
        self._lane_track = LaneTrack()
        self._lane_sig: bytes | None = None
        self._lane_mono: float = 0.0
        self._taken_mono: float = 0.0
        self._last_plan: Any = None
        self._last_lanes: LaneResult | None = None

    def arm_lane_seed(self) -> None:
        """Drop any center fit before engage. The next pair may seed it.

        Pre-engage ticks still run the fitter. gt_run8 stored one_right at
        +5.41 m while idle, then engaged on one_right/hold and steered that
        value. Until a fresh pair is taken, the route stays at x=0.
        """
        self._lane_track.reset()
        self._lane_track.awaiting_pair = True
        self._prior_lanes = []
        self._lane_sig = None
        self._lane_mono = 0.0
        self._taken_mono = 0.0
        self._last_plan = None
        self._last_lanes = None

    def set_detector(self, detector: Any, missing: list[str] | None = None) -> None:
        """Swap the live detector; reset tracks so CIPV ids do not stick to a dead net."""
        self.detector = detector
        self.missing = list(missing or [])
        self.tracker = IoUTracker()
        self._prior_lanes = []
        self._prior_signs = []
        self._lane_track = LaneTrack()
        self._lane_sig = None
        self._lane_mono = 0.0
        self._taken_mono = 0.0
        self._last_plan = None
        self._last_lanes = None

    def tick(
        self,
        main_bgr: np.ndarray | None,
        *,
        stitch_bgr: np.ndarray | None = None,
        ego_speed_mps: float | None,
        steer_deg: float = 0.0,
        yaw_rate: float = 0.0,
        dt_s: float | None = None,
    ) -> PerceptionOut:
        # The wheel is for the override residual and the ego display. The route
        # is predicted from lanes, vehicles, and signs, then followed.
        del steer_deg
        t0 = time.perf_counter()
        dets = self.detector.detect(main_bgr)
        t_det = time.perf_counter()
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
        sig = _lane_view_signature(stitch_bgr, main_bgr)
        now = time.monotonic()
        fresh = sig is None or sig != self._lane_sig
        if fresh:
            self._lane_sig = sig
            self._lane_mono = now
        pixel_age = 0.0 if fresh else max(0.0, now - self._lane_mono)
        dt = 0.05 if dt_s is None or dt_s <= 0.0 else float(dt_s)
        if fresh or self._last_plan is None:
            lanes = lanes_from_view(main_bgr, stitch_bgr)
            self._last_lanes = lanes
        else:
            # Same windshield and pillar pixels. Do not treat the old fit as a
            # new measurement. Move the remembered route with the car.
            lanes = self._last_lanes or LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)
            if self._lane_track.center_x is not None:
                self._lane_track.center_x = reckon_ego_x(
                    self._lane_track.center_x, 8.0, yaw_rate=yaw_rate, dt=dt,
                )
        cipv = select_cipv(track_dicts, path_width=2.0, ego_speed_mps=ego_speed_mps)
        if fresh or self._last_plan is None:
            plan, self._prior_lanes = drive_plan_from_lanes(
                lanes.lanes_bev,
                self._prior_lanes,
                lane_conf=lanes.conf,
                curvature=lanes.curvature,
                tracks=track_dicts,
                signs=signs,
                ego_speed_mps=ego_speed_mps,
                lane_track=self._lane_track,
            )
            # A plan at an unknown speed has no speed fields: never hold it. The next
            # tick with a known speed fits fresh, even on the same pixels.
            self._last_plan = None if ego_speed_mps is None else plan
        else:
            path = reckon_path(
                self._last_plan.path_ego,
                speed_mps=ego_speed_mps,
                yaw_rate=yaw_rate,
                dt=dt,
            )
            plan = replace(
                self._last_plan,
                path_ego=path,
                lane_source=_mark_stale(self._last_plan.lane_source),
                drivable=len(path) >= 2,
            )
            # Pedals, path length and blinker from the moved path at this tick's
            # speed, through the same call a fresh plan ends with. A copy kept the
            # fresh tick's values: fitted at 0 m/s, pred_brake stayed 0 at 5.5 m/s.
            plan = _expose_pedals(plan, ego_speed_mps)
            self._last_plan = None if ego_speed_mps is None else plan
        src = str(plan.lane_source or "")
        held = any(token in src for token in ("hold", "stale", "prior"))
        if held or src in ("", "none"):
            if self._taken_mono <= 0.0:
                self._taken_mono = now
            decision_age = max(0.0, now - self._taken_mono)
        else:
            self._taken_mono = now
            decision_age = 0.0
        # Pixel identity resets only when the bytes repeat. A hold on a new
        # picture used to report age 0, because main's pixels change every tick.
        frame_age_s = max(pixel_age, decision_age)
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
        t_end = time.perf_counter()
        infer_ms = (t_end - t0) * 1000.0
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
                "lane_source": plan.lane_source,
            },
            infer_ms=infer_ms,
            detect_ms=(t_det - t0) * 1000.0,
            plan_ms=(t_end - t_det) * 1000.0,
            missing=missing,
            detector_name=self.detector.name,
            lane_source=plan.lane_source,
            frame_age_s=frame_age_s,
        )


def _mark_stale(source: str) -> str:
    text = str(source or "none")
    if "stale" in text:
        return text
    if text == "none":
        return "stale"
    return f"{text}/stale"


def _lane_view_signature(
    stitch_bgr: np.ndarray | None,
    main_bgr: np.ndarray | None,
) -> bytes | None:
    """Bytes of the cameras that feed the lane fit, subsampled.

    A new rear frame must not count as a new lane frame. Wide, main, narrow,
    and both pillars do. ``None`` means there is no picture yet.
    """
    from python.perception.stitch360 import GAP_PX, SECTOR_H, SECTOR_W, STITCH_ORDER

    parts: list[bytes] = []
    if isinstance(stitch_bgr, np.ndarray) and stitch_bgr.ndim >= 2:
        height, width = int(stitch_bgr.shape[0]), int(stitch_bgr.shape[1])
        expect = len(STITCH_ORDER) * SECTOR_W + (len(STITCH_ORDER) - 1) * GAP_PX
        if height == SECTOR_H and width == expect:
            for cid in ("pillarL", "wide", "main", "narrow", "pillarR"):
                index = STITCH_ORDER.index(cid)
                x0 = index * (SECTOR_W + GAP_PX)
                sector = stitch_bgr[:, x0 : x0 + SECTOR_W]
                parts.append(np.ascontiguousarray(sector[::8, ::8]).tobytes())
    if not parts and isinstance(main_bgr, np.ndarray) and main_bgr.size:
        parts.append(np.ascontiguousarray(main_bgr[::8, ::8]).tobytes())
    if not parts:
        return None
    return b"".join(parts)


def drive_plan_from_lanes(
    fresh: list,
    prior: list,
    *,
    lane_conf: float,
    curvature: float,
    tracks: list,
    signs: list,
    ego_speed_mps: float | None,
    lane_track: LaneTrack | None = None,
) -> tuple[Any, list]:
    """Route from the live lines, or from the last lines that actually routed.

    A non-empty frame that does not make a route used to replace the memory.
    The next tick then had nothing to follow (``lane_x=na`` and steer 0 on
    the PR 82 run). Predict from the last drivable lines instead, and only
    store a frame that produced a route.
    """
    stored = list(prior or [])
    live = list(fresh or [])
    prior_arg = stored if (not live and stored) else None
    seeded = lane_track is not None and lane_track.center_x is not None
    plan = predict_path(
        lanes_bev=live,
        lane_conf=lane_conf,
        curvature=curvature,
        tracks=tracks,
        signs=signs,
        ego_speed_mps=ego_speed_mps,
        prior_lanes=prior_arg,
        lane_track=lane_track,
        remember=bool(live) or not seeded,
    )
    from_live = bool(plan.drivable and live)
    used_prior = bool(plan.drivable and not live and prior_arg)
    if not plan.drivable and live and stored:
        plan = predict_path(
            lanes_bev=[],
            lane_conf=lane_conf,
            curvature=curvature,
            tracks=tracks,
            signs=signs,
            ego_speed_mps=ego_speed_mps,
            prior_lanes=stored,
            lane_track=lane_track,
            remember=False,
        )
        used_prior = bool(plan.drivable)
    if from_live:
        stored = live
    elif used_prior:
        # The prior is in the ego frame of the tick it was seen. Keeping it
        # while the car yaws freezes lane_x off to one side (the 47b430e run
        # held +2.83 m and steered into a building).
        stored = []
    return plan, stored


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
