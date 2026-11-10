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
from python.viz.scene_ribbon import (
    RIBBON_BGR,
    SCENE_RIBBON_MIN_CONF,
    paint_tapered_ribbon,
    scene_ckpt_caption,
    scene_ckpt_centers,
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


def apply_scene_ckpt_lane(pout: Any, pred: Any, *, drive: bool, min_conf: float = SCENE_RIBBON_MIN_CONF) -> str:
    """Optionally replace ``pout``'s lane with the checkpoint prediction.

    Returns ``shadow`` when drive is off (``pout`` unchanged), ``lane`` when
    the prediction is usable, or ``stop`` when it is missing or low-confidence.
    """
    if not drive:
        return "shadow"
    from python.train.scene_lane_runtime import scene_lane_decision, straight_lane_path

    lane_x, conf = _lane_from_filtered_state(pred)
    if scene_lane_decision(lane_x, conf, min_conf=min_conf) != "lane":
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


def cap_drive_speed(pout: Any, speed_mps: float) -> None:
    """Conservative longitudinal cap. Does not raise an existing slower target."""
    cap = float(speed_mps)
    pl = dict(getattr(pout, "planner", None) or {})
    raw = pl.get("target_v")
    if raw is None:
        pl["target_v"] = cap
    else:
        try:
            pl["target_v"] = min(float(raw), cap)
        except (TypeError, ValueError):
            pl["target_v"] = cap
    pout.planner = pl


def _lane_from_filtered_state(pred: Any) -> tuple[float | None, float]:
    """Read the metre the filtered state still publishes.

    ``as_state`` drops centers and nulls ``lane_x`` below 0.40. This does not
    fall back to the raw prediction, so a suppressed center cannot steer.
    Missing centers or ``lane_x is None`` return no metre and do not raise.
    """
    if pred is None or not hasattr(pred, "as_state"):
        return None, 0.0
    try:
        state = pred.as_state()
    except Exception:
        return None, 0.0
    if not isinstance(state, dict):
        return None, 0.0
    try:
        conf = float(state.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    centers = state.get("centers")
    if not isinstance(centers, list) or not centers:
        return None, conf
    raw = state.get("lane_x")
    if isinstance(raw, dict):
        raw = raw.get("lane_x")
    if raw is None:
        return None, conf
    try:
        lane_x = float(raw)
    except (TypeError, ValueError):
        return None, conf
    return lane_x, conf


def scene_ckpt_as_state(
    pred: Any,
    *,
    drive: bool,
    engaged: bool,
    steer: float | None,
) -> dict[str, Any]:
    """Fields written to ``gvd_state.json`` and clip ``state.jsonl``."""
    if pred is not None and hasattr(pred, "as_state"):
        out = dict(pred.as_state())
    else:
        out = {"lane_x": None, "confidence": 0.0, "canvas": None}
    # The drive log stores the filtered metre, or null when the filter
    # suppressed it. It does not copy the raw prediction back in.
    published = out.get("lane_x")
    if isinstance(published, dict):
        out["lane_x"] = published.get("lane_x")
    elif "lane_x" not in out:
        out["lane_x"] = None
    out["source"] = "scene_ckpt"
    out["shadow"] = not bool(drive)
    out["drive"] = bool(drive)
    out["engaged"] = bool(engaged)
    out["steer"] = None if steer is None else float(steer)
    return out


@dataclass
class SceneCkptPred:
    lane_gt: np.ndarray | None = None
    lane_x: float | None = None
    confidence: float = 0.0
    heading_err_rad: float | None = None
    canvas_shape: tuple[int, ...] | None = None
    ckpt: str = ""
    detect: dict[str, Any] | None = None

    @classmethod
    def empty(cls) -> "SceneCkptPred":
        return cls()

    def as_state(self) -> dict[str, Any]:
        from python.train.viz_contract import MIN_CONF, MODEL_SRC, package_lane, renderable_state

        conf = float(self.confidence)
        ckpt = str(self.ckpt or "")
        body: dict[str, Any] = {
            "shadow": True,
            "source": "scene_ckpt",
            "confidence": conf,
            "caption": scene_ckpt_caption(conf),
            "canvas": list(self.canvas_shape) if self.canvas_shape else None,
        }
        # Only records that pass is_model_drawable survive renderable_state.
        # Below 0.40, package_lane nulls the metre and the record is omitted,
        # and centers are not published at all.
        if ckpt and conf >= float(MIN_CONF):
            body["centers"] = [
                {
                    "src": MODEL_SRC,
                    "head": "lane_gt",
                    "ckpt": ckpt,
                    "confidence": conf,
                    "x": float(center["x"]),
                    "y": float(center["y"]),
                }
                for center in scene_ckpt_centers(self.lane_gt)
            ]
            tagged = package_lane(ckpt=ckpt, lane_x=self.lane_x, confidence=conf)
            body["lane"] = tagged
            body["lane_x"] = tagged
        if self.detect:
            body["detect"] = self.detect
        filtered = renderable_state(body)
        filtered["shadow"] = True
        filtered["source"] = "scene_ckpt"
        filtered["confidence"] = conf
        filtered["caption"] = scene_ckpt_caption(conf)
        filtered["canvas"] = body["canvas"]
        # #191 heading is metadata, not a drawable. It is not a steering source.
        if self.heading_err_rad is not None:
            try:
                heading = float(self.heading_err_rad)
            except (TypeError, ValueError):
                heading = None
            if heading is not None and heading == heading:
                filtered["heading_err_rad"] = heading
        return filtered


def scene_ckpt_overlay_caption(pred: SceneCkptPred | None, *, drive: bool = False) -> str:
    """Caption drawn by ``draw_scene_ckpt_overlay``. Pure so tests can lock the kind.

    Low confidence is always ``LOW``, even when drive is on: the drive path has
    already returned stop below SCENE_RIBBON_MIN_CONF. High confidence is
    ``DRIVE`` only when drive is on; otherwise ``SHADOW``.
    """
    if pred is None:
        return "SHADOW"
    cap = scene_ckpt_caption(pred.confidence)  # SHADOW or LOW
    kind = "DRIVE" if (drive and cap == "SHADOW") else cap
    if pred.lane_x is not None and kind in ("SHADOW", "DRIVE"):
        return f"{kind} lane_x={float(pred.lane_x):+.2f}m conf={float(pred.confidence):.2f}"
    return f"{kind} conf={float(pred.confidence):.2f}"


def draw_scene_ckpt_overlay(
    frame: np.ndarray | None,
    pred: SceneCkptPred | None,
    *,
    drive: bool = False,
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
    caption = scene_ckpt_overlay_caption(pred, drive=drive)
    cv2.putText(out, caption, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    # Solid swatch so tests (and a clip) can see the mark without anti-alias blending.
    cv2.rectangle(out, (12, 36), (40, 56), color, -1)
    if pred is None or pred.lane_gt is None:
        return out
    if float(pred.confidence) < SCENE_RIBBON_MIN_CONF:
        return out
    centers = scene_ckpt_centers(pred.lane_gt)
    poly = scene_ckpt_ribbon_polyline(centers, pred.confidence)

    def _project(x: float, y: float) -> tuple[int, int] | None:
        return project_main_ground(x, y, w, h)

    paint_tapered_ribbon(out, poly, _project, RIBBON_BGR)
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

    def __init__(self, path: Path | str) -> None:
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

    def observe(self, frame: Any, dt_s: float | None) -> SceneCkptPred:
        arr = _require_main(frame)
        if arr is None:
            return SceneCkptPred.empty()
        canvas = _main_canvas(arr, self.cams, self.canvas_h, self.canvas_w, float(getattr(self.net, "scale", 1.0) or 1.0))
        import torch

        with torch.inference_mode():
            pred, self.state = self.net.forward(canvas, 0.0 if dt_s is None else float(dt_s), self.state)
        raw = getattr(pred, "lane_gt", None)
        detect = _detect_package(self.path.name, pred)
        shape = tuple(int(x) for x in canvas.shape)
        if raw is None:
            return SceneCkptPred(canvas_shape=shape, ckpt=self.path.name, detect=detect)
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
            canvas_shape=shape,
            ckpt=self.path.name,
            detect=detect,
        )


def _numpy_head(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if detach is None:
        return None
    return detach().float().cpu().numpy()


def _detect_package(ckpt: str, pred: Any) -> dict[str, Any] | None:
    """Model-head package. Missing heads stay omitted. Never reads ground truth."""
    if getattr(pred, "edge", None) is None and getattr(pred, "obj_box", None) is None and getattr(pred, "brake", None) is None:
        return None
    from python.train.viz_contract import package_detect

    return package_detect(
        ckpt=ckpt,
        edge=_numpy_head(getattr(pred, "edge", None)),
        edge_conf=_numpy_head(getattr(pred, "edge_conf", None)),
        obj_box=_numpy_head(getattr(pred, "obj_box", None)),
        obj_conf=_numpy_head(getattr(pred, "obj_conf", None)),
        obj_class=_numpy_head(getattr(pred, "obj_class", None)),
        brake=_numpy_head(getattr(pred, "brake", None)),
        brake_conf=_numpy_head(getattr(pred, "brake_conf", None)),
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
