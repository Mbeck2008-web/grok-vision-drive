"""Load a train_scene epoch checkpoint and run it live as shadow-only.

``python.run_vision --scene-ckpt`` is the only caller. The prediction is for
the overlay and ``gvd_state.json``; it must not reach the wheel or pedals.
Inference is CPU so BeamNG keeps the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from python.debug.lane_targets import (
    LIVE_LANE_Y_M,
    LOOKAHEADS_M,
    confidence_from_vector,
    lane_x_from_vector,
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


def shadow_engage_allowed() -> bool:
    """Engage is never allowed while a scene-ckpt shadow runtime is the point of the run."""
    return False


def apply_scene_ckpt_to_command(cmd: Any, _pred: Any) -> Any:
    """Return ``cmd`` unchanged. A scene-ckpt prediction never edits actuation."""
    if cmd is None:
        return cmd
    try:
        return replace(cmd)
    except TypeError:
        return cmd


@dataclass
class SceneCkptPred:
    lane_gt: np.ndarray | None = None
    lane_x: float | None = None
    confidence: float = 0.0
    canvas_shape: tuple[int, ...] | None = None

    @classmethod
    def empty(cls) -> "SceneCkptPred":
        return cls()

    def as_state(self) -> dict[str, Any]:
        return {
            "shadow": True,
            "source": "scene_ckpt",
            "lane_x": None if self.lane_x is None else float(self.lane_x),
            "confidence": float(self.confidence),
            "canvas": list(self.canvas_shape) if self.canvas_shape else None,
        }


def draw_scene_ckpt_overlay(frame: np.ndarray | None, pred: SceneCkptPred | None) -> np.ndarray | None:
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
    caption = "SHADOW"
    if pred is not None and pred.lane_x is not None:
        caption = f"SHADOW lane_x={float(pred.lane_x):+.2f}m conf={float(pred.confidence):.2f}"
    cv2.putText(out, caption, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    # Solid swatch so tests (and a clip) can see the mark without anti-alias blending.
    cv2.rectangle(out, (12, 36), (40, 56), color, -1)
    if pred is None or pred.lane_gt is None:
        return out
    vec = np.asarray(pred.lane_gt, dtype=np.float64).reshape(-1)
    pts: list[tuple[int, int]] = []
    for i, y_m in enumerate(LOOKAHEADS_M):
        if vec.size < i * 3 + 1:
            break
        pts.append(_project_ego(float(vec[i * 3]), float(y_m), w, h))
    extra = lane_x_from_vector(vec, LIVE_LANE_Y_M)
    if extra is not None:
        pts.append(_project_ego(float(extra), float(LIVE_LANE_Y_M), w, h))
        pts = sorted(set(pts), key=lambda p: p[1], reverse=True)
    drawn = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
    if len(drawn) >= 2:
        cv2.polylines(out, [np.array(drawn, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    for x, y in drawn:
        cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)
    return out


def _project_ego(x_m: float, y_m: float, width: int, height: int) -> tuple[int, int]:
    """Rough ground-plane sketch: x right, y forward, onto the camera frame."""
    y = max(float(y_m), 0.5)
    fx = 0.90 * float(width)
    u = int(round(width * 0.5 + (float(x_m) / y) * fx))
    v = int(round(height * 0.72 - (1.15 / y) * fx))
    return u, v


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
        if raw is None:
            return SceneCkptPred(canvas_shape=tuple(int(x) for x in canvas.shape))
        vec = raw.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        return SceneCkptPred(
            lane_gt=vec,
            lane_x=lane_x_from_vector(vec),
            confidence=confidence_from_vector(vec),
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
