"""Load a train_scene epoch checkpoint and run it live.

``python.run_vision --scene-ckpt`` is the only caller. Shadow (the default)
draws the overlay and writes state; the prediction must not reach the wheel
or pedals. ``--scene-ckpt-drive`` is an explicit opt-in: ``lane_x`` is fed
to the existing modular lateral controller. It never writes ctl-cond, never
bypasses heartbeat / AEB / the confidence veto, and inference stays on CPU
so BeamNG keeps the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from python.debug.lane_targets import (
    confidence_from_vector,
    lane_x_from_vector,
)
from python.runtime.frame_gate import STALE_CAPTION
from python.viz.scene_ribbon import (
    RIBBON_BGR,
    SCENE_RIBBON_MIN_CONF,
    paint_tapered_ribbon,
    scene_ckpt_caption,
    scene_ckpt_centers,
    scene_ckpt_display_gain,
    scene_ckpt_ribbon_polyline,
)

# Distinct from the existing ice-blue corridor (python.viz.stage.ICE).
ICE_BGR = (212, 196, 158)
SHADOW_BGR = (48, 32, 220)  # warm red, BGR


class SceneCkptInputError(ValueError):
    """Main-camera shape/channel mismatch, or a checkpoint this runtime refuses."""


def overlay_bgr(kind: str) -> tuple[int, int, int]:
    """Named overlay colors. ``ice`` is the existing viz corridor; ``shadow`` is this path."""
    if kind == "ice":
        return ICE_BGR
    if kind == "shadow":
        return SHADOW_BGR
    raise KeyError(kind)


DRIVE_SPEED_MPS = 15.0


def shadow_engage_allowed(*, drive: bool = False) -> bool:
    """Engage is blocked in shadow. ``--scene-ckpt-drive`` is the only opt-in."""
    return bool(drive)


def apply_scene_ckpt_to_command(cmd: Any, _pred: Any) -> Any:
    """Return ``cmd`` unchanged.

    Drive mode does not rewrite pedals here. ``lane_x`` is injected into the
    modular path before ``shadow_tick``, so AEB, the veto, and the speed
    controller still have the last word.
    """
    if cmd is None:
        return cmd
    try:
        return replace(cmd)
    except TypeError:
        return cmd


def apply_scene_ckpt_lane(pout: Any, pred: Any, *, drive: bool, min_conf: float = SCENE_RIBBON_MIN_CONF, no_conf_gate: bool = False) -> str:
    """Optionally replace ``pout``'s lane with the checkpoint prediction.

    Returns ``shadow`` when drive is off (``pout`` unchanged), ``lane`` when
    the prediction is usable, or ``stop`` when it is missing or low-confidence.
    """
    if not drive:
        return "shadow"
    from python.train.scene_lane_runtime import scene_lane_decision, straight_lane_path

    lane_x = None if pred is None else getattr(pred, "lane_x", None)
    try:
        conf = 0.0 if pred is None else float(getattr(pred, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if scene_lane_decision(lane_x, conf, min_conf=min_conf, no_conf_gate=no_conf_gate) != "lane":
        pout.path_ego = []
        pout.lane_conf = float(conf)
        try:
            pout.path_conf = min(float(getattr(pout, "path_conf", 0.0) or 0.0), float(conf))
        except (TypeError, ValueError):
            pout.path_conf = float(conf)
        pout.lane_source = "scene_ckpt/stop"
        pout.path_debug_preview = False
        return "stop"
    pout.path_ego = straight_lane_path(float(lane_x))
    pout.lane_conf = float(conf)
    pout.path_conf = float(conf)
    pout.lane_source = "scene_ckpt"
    pout.path_debug_preview = False
    return "lane"


def _scene_seen(pout: Any, pl: dict[str, Any]) -> bool:
    """True when modular already tracked a lead, a cut, or any detection/sign."""
    if pl.get("cipv_id") is not None:
        return True
    if pl.get("ttc_lead") is not None:
        return True
    if str(pl.get("aeb") or "off") == "brake":
        return True
    if str(pl.get("stop_reason") or "none") not in ("none", "", "None"):
        return True
    for attr in ("signs", "tracks", "dets"):
        if getattr(pout, attr, None):
            return True
    for attr in ("objects_n", "tracks_n"):
        try:
            if int(getattr(pout, attr, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _can_fill_empty_plan(
    *,
    unique_gpu_n: int | None,
    hitch: bool,
    frame_fresh: bool,
) -> bool:
    """Install cruise only on a fresh unique colour frame, not a hitch replay."""
    if hitch or not frame_fresh:
        return False
    if unique_gpu_n is None:
        return False
    try:
        return int(unique_gpu_n) > 0
    except (TypeError, ValueError):
        return False


def _cipv_id_of(track: Any) -> int | None:
    if track is None:
        return None
    ident = track.get("id") if isinstance(track, dict) else getattr(track, "id", None)
    try:
        return None if ident is None else int(ident)
    except (TypeError, ValueError):
        return None


def replan_scene_ckpt_path(pout: Any, *, ego_speed_mps: float | None) -> None:
    """Re-run scene cuts and CIPV against the installed scene-ckpt path.

    ``predict_path`` returns ``_empty()`` before ``_apply_scene`` when there
    are no BEV lanes, so a leaked ``target_v=0`` / ``stop_reason=none`` can
    hide a lead or a stop/red that modular already saw. This attaches those
    cuts to ``path_ego`` after ``apply_scene_ckpt_lane``. It never writes a
    cruise target when a lead, sign, object, or cut is present.
    """
    path = list(getattr(pout, "path_ego", None) or [])
    if len(path) < 2:
        return
    from python.perception.cipv import select_cipv
    from python.perception.pipeline import fuse_planner_cipv
    from python.planning.path_predictor import (
        _apply_scene,
        _limit_mps,
        _predict_occupants,
        _speed_flags,
    )

    pl = dict(getattr(pout, "planner", None) or {})
    tracks = list(getattr(pout, "tracks", None) or [])
    signs = list(getattr(pout, "signs", None) or [])
    try:
        width = float(getattr(pout, "path_width", None) or pl.get("corridor_width") or 2.0)
    except (TypeError, ValueError):
        width = 2.0
    try:
        curv = float(pl.get("curvature") or 0.0)
    except (TypeError, ValueError):
        curv = 0.0
    occupants = _predict_occupants(tracks)
    path, stop_reason, cut_id, ttc, _occupied, stop_dist, closing = _apply_scene(
        path, occupants, signs, width, ego_speed_mps=ego_speed_mps,
    )
    aeb, target, ttc = _speed_flags(
        curv=curv,
        stop_reason=stop_reason,
        ttc=ttc,
        stop_dist=stop_dist,
        closing_mps=closing,
        ego_speed_mps=ego_speed_mps,
        signs=signs,
        path=path,
    )
    cipv = select_cipv(tracks, path_width=width, ego_speed_mps=ego_speed_mps)
    aeb, ttc, target = fuse_planner_cipv(
        plan_aeb=aeb,
        plan_ttc=ttc,
        plan_target_v=target,
        plan_stop_reason=stop_reason,
        cipv=cipv,
    )
    # Standing at a stop/red/yield is a hold. Comfort sqrt(2*a*s) is for
    # approaching while moving; at ego ~= 0 it would invent a creep target.
    try:
        standing = ego_speed_mps is None or float(ego_speed_mps) <= 0.05
    except (TypeError, ValueError):
        standing = True
    if standing and stop_reason in ("stop_sign", "red_light", "yellow_light", "yield"):
        target = 0.0
    limit = _limit_mps(signs, path)
    if limit is not None:
        try:
            lim = max(0.0, float(limit))
        except (TypeError, ValueError):
            lim = None
        else:
            if target is None:
                target = lim
            else:
                try:
                    target = min(float(target), lim)
                except (TypeError, ValueError):
                    target = lim
    pout.path_ego = path
    if stop_reason not in ("none", "", "None", None):
        pl["stop_reason"] = stop_reason
    cipv_id = cut_id if cut_id is not None else _cipv_id_of(getattr(cipv, "track", None))
    if cipv_id is None:
        cipv_id = pl.get("cipv_id")
    if cipv_id is not None:
        pl["cipv_id"] = cipv_id
    # Never downgrade an upstream AEB brake when this tick has no tracks
    # (select_cipv at standstill returns aeb=off; _empty skipped _apply_scene).
    if aeb == "brake":
        pl["aeb"] = "brake"
    elif aeb is not None and str(pl.get("aeb") or "off") != "brake":
        pl["aeb"] = aeb
    if ttc is not None:
        pl["ttc_lead"] = ttc
    seen = _scene_seen(pout, pl)
    if seen:
        # Do not invent cruise toward something already seen. A real cut / AEB
        # / posted limit may write a slower target; a standstill CIPV stays 0.
        prev = pl.get("target_v")
        try:
            prev_f = None if prev is None else float(prev)
        except (TypeError, ValueError):
            prev_f = None
        kept_stop = str(pl.get("stop_reason") or "none")
        new_cut = stop_reason not in ("none", "", "None", None)
        if str(pl.get("aeb") or "off") == "brake" or kept_stop not in ("none", "", "None"):
            if not new_cut:
                # Existing cut/stop, no new geometry this tick: do not
                # replace a hold with the empty-path cruise from _speed_flags.
                if prev_f is None or prev_f <= 0.0:
                    pl["target_v"] = 0.0
            elif target is None:
                pl["target_v"] = 0.0 if prev_f is None or prev_f <= 0.0 else prev_f
            else:
                try:
                    tv = float(target)
                except (TypeError, ValueError):
                    tv = 0.0
                if prev_f is None or prev_f <= 0.0:
                    pl["target_v"] = tv
                else:
                    pl["target_v"] = min(prev_f, tv)
        elif limit is not None:
            try:
                pl["target_v"] = min(0.0 if prev_f is None or prev_f <= 0.0 else prev_f, max(0.0, float(limit)))
            except (TypeError, ValueError):
                pl["target_v"] = 0.0 if prev_f is None or prev_f <= 0.0 else prev_f
        elif prev_f is None or prev_f <= 0.0:
            pl["target_v"] = 0.0
    pout.planner = pl


def cap_drive_speed(
    pout: Any,
    speed_mps: float,
    *,
    unique_gpu_n: int | None = None,
    hitch: bool = False,
    frame_fresh: bool = True,
) -> None:
    """Conservative longitudinal cap. Does not raise an existing slower target.

    ``None`` means there is no speed plan. A literal ``0.0`` is a placeholder
    only when AEB is not braking, no scene cut is recorded, and nothing was
    seen (no ``cipv_id``, TTC, objects, tracks, or signs). A negative target
    is a hold at 0, not a cruise. The placeholder is filled only on a fresh
    unique colour frame (``unique_gpu_n > 0``, not a soft-esc hitch).
    """
    cap = float(speed_mps)
    pl = dict(getattr(pout, "planner", None) or {})
    raw = pl.get("target_v")
    aeb = str(pl.get("aeb") or "off")
    stop = str(pl.get("stop_reason") or "none")
    seen = _scene_seen(pout, pl)
    can_fill = (
        (not seen)
        and aeb != "brake"
        and stop in ("none", "", "None")
        and _can_fill_empty_plan(unique_gpu_n=unique_gpu_n, hitch=hitch, frame_fresh=frame_fresh)
    )
    if raw is None:
        if can_fill:
            pl["target_v"] = cap
        elif seen or aeb == "brake" or stop not in ("none", "", "None"):
            pl["target_v"] = 0.0
        pout.planner = pl
        return
    try:
        tv = float(raw)
    except (TypeError, ValueError):
        if can_fill:
            pl["target_v"] = cap
        else:
            pl["target_v"] = 0.0
        pout.planner = pl
        return
    if tv < 0.0:
        pl["target_v"] = 0.0
    elif tv == 0.0 and can_fill:
        pl["target_v"] = cap
    else:
        pl["target_v"] = min(tv, cap)
    pout.planner = pl


def prepare_scene_ckpt_drive_speed(
    pout: Any,
    speed_mps: float,
    *,
    ego_speed_mps: float | None,
    unique_gpu_n: int | None = None,
    hitch: bool = False,
    frame_fresh: bool = True,
) -> None:
    """Re-run scene cuts on the ckpt path, then cap only a still-empty plan."""
    replan_scene_ckpt_path(pout, ego_speed_mps=ego_speed_mps)
    cap_drive_speed(
        pout,
        speed_mps,
        unique_gpu_n=unique_gpu_n,
        hitch=hitch,
        frame_fresh=frame_fresh,
    )


def scene_ckpt_as_state(
    pred: Any,
    *,
    drive: bool,
    engaged: bool,
    steer: float | None,
    frame_stale: bool = False,
) -> dict[str, Any]:
    """Fields written to ``gvd_state.json`` and clip ``state.jsonl``."""
    if pred is not None and hasattr(pred, "as_state"):
        out = dict(pred.as_state())
    else:
        out = {"lane_x": None, "confidence": 0.0, "canvas": None}
    out["source"] = "scene_ckpt"
    out["shadow"] = not bool(drive)
    out["drive"] = bool(drive)
    out["engaged"] = bool(engaged)
    out["steer"] = None if steer is None else float(steer)
    out["frame_stale"] = bool(frame_stale)
    out["caption"] = STALE_CAPTION if frame_stale else ("DRIVE" if drive else "SHADOW")
    return out


@dataclass
class SceneCkptPred:
    lane_gt: np.ndarray | None = None
    lane_x: float | None = None
    confidence: float = 0.0
    heading_err_rad: float | None = None
    canvas_shape: tuple[int, ...] | None = None

    @classmethod
    def empty(cls) -> "SceneCkptPred":
        return cls()

    def as_state(self) -> dict[str, Any]:
        centers = scene_ckpt_centers(self.lane_gt)
        out = {
            "shadow": True,
            "source": "scene_ckpt",
            "lane_x": None if self.lane_x is None else float(self.lane_x),
            "confidence": float(self.confidence),
            "centers": centers,
            "caption": scene_ckpt_caption(self.confidence),
            "canvas": list(self.canvas_shape) if self.canvas_shape else None,
        }
        if self.heading_err_rad is not None:
            out["heading_err_rad"] = float(self.heading_err_rad)
        return out


def scene_ckpt_overlay_caption(
    pred: SceneCkptPred | None,
    *,
    drive: bool = False,
    frame_stale: bool = False,
) -> str:
    """Mode/safety tag drawn by ``draw_scene_ckpt_overlay``.

    Confidence is shown by ribbon opacity, not text. This returns only
    ``STALE``, ``DRIVE``, or ``SHADOW``. A stale frame is fail-closed
    (no ribbon) and is not a drive caption.
    """
    if frame_stale:
        return STALE_CAPTION
    return "DRIVE" if drive else "SHADOW"


def draw_scene_ckpt_overlay(
    frame: np.ndarray | None,
    pred: SceneCkptPred | None,
    *,
    drive: bool = False,
    frame_stale: bool = False,
) -> np.ndarray | None:
    """Paint the predicted lane-center path on a camera frame. Does not modify ``frame``.

    No picture: return ``frame`` unchanged (including None). Callers can always
    assign the result back without a None-check at the call site.
    """
    if frame is None:
        return frame
    import cv2

    out = np.array(frame, copy=True)
    if out.ndim != 3 or out.shape[-1] != 3:
        return out
    color = overlay_bgr("shadow")
    h, w = int(out.shape[0]), int(out.shape[1])
    # Mode/safety tag only (SHADOW / DRIVE / STALE). Confidence is ribbon opacity, not text.
    caption = scene_ckpt_overlay_caption(pred, drive=drive, frame_stale=frame_stale)
    cv2.putText(out, caption, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    # Solid swatch so tests (and a clip) can see the mark without anti-alias blending.
    # Confidence is shown by ribbon opacity, not text.
    cv2.rectangle(out, (12, 36), (40, 56), color, -1)
    if frame_stale or pred is None or pred.lane_gt is None:
        return out
    centers = scene_ckpt_centers(pred.lane_gt)
    # min_conf=0: low confidence still strokes, at display gain ~0.40.
    poly = scene_ckpt_ribbon_polyline(centers, pred.confidence, min_conf=0.0)

    def _project(x: float, y: float) -> tuple[int, int] | None:
        return project_main_ground(x, y, w, h)

    paint_tapered_ribbon(
        out, poly, _project, RIBBON_BGR, gain=scene_ckpt_display_gain(pred.confidence),
    )
    return out


def project_main_ground(x_m: float, y_m: float, width: int, height: int) -> tuple[int, int] | None:
    """Ego ground (x right, y forward) onto the main camera using cameras.yaml.

    Uses the recorded mount (pos, yaw, pitch) and vertical fov, scaled to the
    actual frame size. OpenCV ``v`` grows downward, so a farther point has a
    smaller ``v`` than a near point (it sits higher in the image).
    """
    from python.perception.lanes import camera_pose, pose_pixel_of_ground

    pose = camera_pose("main")
    if pose is None:
        return None
    uv = pose_pixel_of_ground(pose, float(x_m), float(y_m), width=int(width), height=int(height))
    if uv is None:
        return None
    return int(round(uv[0])), int(round(uv[1]))


def _project_ego(x_m: float, y_m: float, width: int, height: int) -> tuple[int, int]:
    """Back-compat wrapper: the windshield model, or (0, 0) if the point is behind the lens."""
    uv = project_main_ground(x_m, y_m, width, height)
    return (0, 0) if uv is None else uv


def _infer_flag(meta: Any, weights: Any, key: str, prefixes: tuple[str, ...], arch_token: str) -> bool:
    if isinstance(meta, dict) and meta.get(key) is True:
        return True
    if weights:
        for name in weights:
            text = str(name)
            if any(text.startswith(prefix) for prefix in prefixes):
                return True
    arch = str((meta or {}).get("arch") or "")
    return bool(arch_token) and arch_token in arch


class SceneCkptRuntime:
    """One CPU SceneNet loaded from an epoch ``.npz``, GRU state kept across ticks."""

    def __init__(self, path: Path | str, requested_preset: str | None = None) -> None:
        from python.train.epoch_ckpt import build_net, load_epoch_ckpt, load_into_net

        self.path = Path(path)
        loaded = load_epoch_ckpt(self.path)
        meta = loaded["meta"]
        weights = loaded["weights"]
        if bool(meta.get("cam_age")):
            raise SceneCkptInputError(
                f"{self.path.name}: cam-age checkpoints need per-camera hold ages; "
                "this shadow runtime does not feed them"
            )
        cond_dim = int(meta.get("ctl_cond_dim") or 0)
        if cond_dim:
            raise SceneCkptInputError(
                f"{self.path.name}: ctl cond model (ctl_cond_dim={cond_dim}, gvd_ctl_cond) "
                "needs the t-dt controls; this shadow runtime does not feed them"
            )
        self.meta = meta
        if requested_preset not in (None, "", "auto"):
            from python.train.model_size import refuse_preset_mismatch

            refuse_preset_mismatch(requested_preset, meta)
        self.net = build_net(meta, weights)
        load_into_net(self.net, weights)
        self.net = self.net.cpu().eval()
        net_cond = int(getattr(self.net, "cond_dim", 0) or 0)
        if net_cond:
            raise SceneCkptInputError(
                f"{self.path.name}: ctl cond model (cond_dim={net_cond}, gvd_ctl_cond) "
                "needs the t-dt controls; this shadow runtime does not feed them"
            )
        for param in self.net.parameters():
            if param.device.type != "cpu":
                raise SceneCkptInputError("scene-ckpt shadow must stay on CPU")
        self.cams = tuple(getattr(self.net, "cams", ()) or ())
        names = self.cams or ("main",)
        if names != ("main",):
            raise SceneCkptInputError(
                f"{self.path.name}: this shadow runtime only has the main camera; "
                f"refusing cams={list(names)} (missing sectors would be black)"
            )
        self.canvas_h = int(getattr(self.net, "canvas_h", 192))
        self.canvas_w = int(getattr(self.net, "canvas_w", 256))
        self.aux_sensors = bool(getattr(self.net, "aux_sensors", False))
        self.state = self.net.initial_state(device="cpu")

    @property
    def wants_route(self) -> bool:
        return bool(getattr(self.net, "route_cond", False))

    def observe(self, frame: Any, dt_s: float | None, route: Any = None) -> SceneCkptPred:
        arr = _require_main(frame)
        if arr is None:
            return SceneCkptPred.empty()
        canvas = _main_canvas(arr, self.cams, self.canvas_h, self.canvas_w, float(getattr(self.net, "scale", 1.0) or 1.0))
        import torch

        with torch.inference_mode():
            if self.wants_route:
                pred, self.state = self.net.forward(
                    canvas, 0.0 if dt_s is None else float(dt_s), self.state, route=route
                )
            else:
                pred, self.state = self.net.forward(canvas, 0.0 if dt_s is None else float(dt_s), self.state)
        raw = getattr(pred, "lane_gt", None)
        if raw is None:
            return SceneCkptPred(canvas_shape=tuple(int(x) for x in canvas.shape))
        vec = raw.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        heading = None
        if vec.size >= 2:
            raw_h = float(vec[-2])
            if raw_h == raw_h:
                heading = raw_h
        return SceneCkptPred(
            lane_gt=vec,
            lane_x=lane_x_from_vector(vec),
            confidence=confidence_from_vector(vec),
            heading_err_rad=heading,
            canvas_shape=tuple(int(x) for x in canvas.shape),
        )


def _require_main(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3 or int(arr.shape[-1]) != 3:
        raise SceneCkptInputError(
            f"main camera must be HxWx3 (live long side ~640), got shape {tuple(arr.shape)} "
            "(channel/shape check)"
        )
    return arr


def _main_canvas(
    frame: np.ndarray,
    cams: tuple[str, ...],
    canvas_h: int,
    canvas_w: int,
    scale: float,
) -> np.ndarray:
    from python.data.strip_writer import compose_strip
    from python.train.cam_strip import extract_cams, scale_canvas

    names = cams or ("main",)
    strip = compose_strip({"main": frame})
    packed = extract_cams(strip, names)
    if float(scale) < 0.999:
        packed = scale_canvas(packed, scale, names)
    if packed.ndim != 3 or packed.shape[-1] != 3:
        raise SceneCkptInputError(f"composed canvas has bad channels: {packed.shape}")
    if int(packed.shape[0]) != int(canvas_h) or int(packed.shape[1]) != int(canvas_w):
        raise SceneCkptInputError(
            f"composed canvas {packed.shape[0]}x{packed.shape[1]} does not match "
            f"the checkpoint canvas {canvas_h}x{canvas_w}"
        )
    return packed
