"""Higher-resolution depth for Reality View: a light FPN decoder off the SharedStem.

Trainer 2.0 add-on (Michael 2026-11-26 12:50 AM): replace the 12x16
``aux_depth`` grid with a ``DEPTH_HI_H x DEPTH_HI_W`` (48x64) map from a small
top-down decoder over the main-sector stem pyramid (stride 16 -> 8 -> 4), plus
a sky/far class. Opt-in: ``train_scene --aux-depth --heads ...,depth_hires``.

* Supervision is BeamNG ``main_depth`` (uint16 planar-z cm), **training labels
  only**. Runtime stays camera-only: the head reads only the camera stem map.
* Depth cells: area mean of the valid metres (``0 < cm < FAR_CLASS_M``); a cell
  is valid when >= half its pixels are valid near depth. Far cells (sky / far
  plane / clamp): >= half the pixels at or past ``FAR_CLASS_M`` (250 m; main
  far plane is 300 m). Far cells get no metric target; the far logit learns them.
* Loss: smooth-L1 on log metres over valid cells + ``FAR_LOSS_WEIGHT`` x BCE on
  the far logit over cells that are mostly valid-or-far. ``FAR_CLASS_M``,
  ``FAR_LOSS_WEIGHT`` and the decoder width are spec proposals.

Not exported to ONNX in this PR (like the 12x16 head); the live Reality View
path picks it up in its own PR. ``scripts/bench_depth_hires.py`` measures the
decoder's CPU ms and parameter / activation memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEPTH_HI_H = 48
DEPTH_HI_W = 64
FAR_CLASS_M = 250.0
MIN_DEPTH_M = 0.1
FAR_LOSS_WEIGHT = 0.5
DECODER_WIDTH = 32
ARCH_TOKEN = f":depth_hi={DEPTH_HI_H}x{DEPTH_HI_W}"
HEAD_PREFIX = "depth_hires_head."

__all__ = [
    "ARCH_TOKEN",
    "DECODER_WIDTH",
    "DEPTH_HI_H",
    "DEPTH_HI_W",
    "FAR_CLASS_M",
    "FAR_LOSS_WEIGHT",
    "HEAD_PREFIX",
    "MIN_DEPTH_M",
    "build_depth_hires_head",
    "depth_hires_loss",
    "depth_hires_targets",
    "empty_targets",
    "stem_pyramid",
]


def empty_targets() -> dict[str, np.ndarray]:
    """All-masked targets (no file: nothing is invented)."""
    z = np.zeros((DEPTH_HI_H, DEPTH_HI_W), np.float32)
    return {"depth_m": z.copy(), "depth_valid": z.copy(), "far": z.copy(), "far_valid": z.copy()}


def depth_hires_targets(depth_cm: np.ndarray) -> dict[str, np.ndarray]:
    """``DEPTH_HI_H x DEPTH_HI_W`` metres, valid mask, far label and far mask from uint16 cm."""
    import cv2

    if depth_cm.ndim != 2 or depth_cm.dtype != np.uint16:
        raise ValueError(f"depth must be uint16 HxW cm, got {depth_cm.dtype} {getattr(depth_cm, 'shape', None)}")
    cm = depth_cm.astype(np.float32)
    far_px = (cm >= FAR_CLASS_M * 100.0).astype(np.float32)
    near_px = ((cm > 0.0) & (far_px < 0.5)).astype(np.float32)
    metres = np.where(near_px > 0.5, cm / 100.0, 0.0).astype(np.float32)
    size = (DEPTH_HI_W, DEPTH_HI_H)
    d = cv2.resize(metres, size, interpolation=cv2.INTER_AREA)
    v = cv2.resize(near_px, size, interpolation=cv2.INTER_AREA)
    f = cv2.resize(far_px, size, interpolation=cv2.INTER_AREA)
    mean = np.where(v > 1e-6, d / np.maximum(v, 1e-6), 0.0).astype(np.float32)
    far = (f >= 0.5).astype(np.float32)
    valid = ((v >= 0.5) & (far < 0.5)).astype(np.float32)
    far_valid = ((v + f) >= 0.5).astype(np.float32)
    return {"depth_m": mean, "depth_valid": valid, "far": far, "far_valid": far_valid}


def stem_pyramid(stem: Any, x: Any) -> list[Any]:
    """Every ReLU output of ``SharedStem.conv`` (strides 2, 4, 8, 16), same math as ``stem.features``."""
    import torch.nn as nn

    maps = []
    for layer in stem.conv:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            maps.append(x)
    return maps


def build_depth_hires_head(stem_channels: tuple[int, ...], width: int = DECODER_WIDTH) -> Any:
    """Top-down FPN over the last three stem maps -> (metres, far logit) at 48x64."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    chans = tuple(int(c) for c in stem_channels)
    if len(chans) < 3:
        raise ValueError(f"depth_hires needs >= 3 stem stages, got {chans}")

    class DepthHiresHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lat = nn.ModuleList([nn.Conv2d(c, int(width), kernel_size=1) for c in chans[-3:]])
            self.smooth = nn.Conv2d(int(width), int(width), kernel_size=3, padding=1)
            self.norm = nn.GroupNorm(4 if int(width) % 4 == 0 else 1, int(width))
            self.out = nn.Conv2d(int(width), 2, kernel_size=1)

        def forward(self, pyramid: list[Any]) -> tuple[Any, Any]:
            c2, c3, c4 = pyramid[-3:]
            p = self.lat[2](c4)
            p = functional.interpolate(p, size=c3.shape[-2:], mode="bilinear", align_corners=False) + self.lat[1](c3)
            p = functional.interpolate(p, size=c2.shape[-2:], mode="bilinear", align_corners=False) + self.lat[0](c2)
            p = torch.relu(self.norm(self.smooth(p)))
            out = self.out(p)
            if tuple(out.shape[-2:]) != (DEPTH_HI_H, DEPTH_HI_W):
                out = functional.interpolate(out, size=(DEPTH_HI_H, DEPTH_HI_W), mode="bilinear", align_corners=False)
            depth_m = functional.softplus(out[:, :1]) + MIN_DEPTH_M
            return depth_m, out[:, 1:2]

    return DepthHiresHead()


def _nhw(t: Any) -> Any:
    if t.dim() >= 3 and t.shape[-3] == 1:
        t = t.squeeze(-3)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def depth_hires_loss(depth_m: Any, far_logit: Any, target: dict[str, Any]) -> tuple[Any, Any]:
    """(depth term, far term). Masked means; a mask with no cells gives a zero term."""
    import torch
    import torch.nn.functional as functional

    pd, pf = _nhw(depth_m), _nhw(far_logit)
    gd, md = _nhw(target["aux_depth"]).to(pd), _nhw(target["aux_depth_valid"]).to(pd)
    gf, mf = _nhw(target["aux_depth_far"]).to(pd), _nhw(target["aux_depth_far_valid"]).to(pd)
    for name, t in (("depth goal", gd), ("depth mask", md), ("far goal", gf), ("far mask", mf), ("far logit", pf)):
        if tuple(t.shape) != tuple(pd.shape):
            raise ValueError(f"depth_hires {name} shape {tuple(t.shape)} != pred {tuple(pd.shape)}")
    zero = pd.sum() * 0.0
    depth_term = zero
    if bool((md > 0.5).any()):
        err = functional.smooth_l1_loss(
            torch.log(pd.clamp(min=MIN_DEPTH_M)), torch.log(gd.clamp(min=MIN_DEPTH_M)), reduction="none"
        )
        depth_term = (err * md).sum() / md.sum().clamp(min=1.0)
    far_term = zero
    if bool((mf > 0.5).any()):
        bce = functional.binary_cross_entropy_with_logits(pf, gf, reduction="none")
        far_term = FAR_LOSS_WEIGHT * (bce * mf).sum() / mf.sum().clamp(min=1.0)
    return depth_term, far_term
