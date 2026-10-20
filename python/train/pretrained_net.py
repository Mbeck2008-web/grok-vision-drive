"""Single-frame ImageNet baseline for the scene trainer (``--backbone``).

A torchvision backbone with ImageNet weights reads each frame on its own.
There is no GRU and no skip: the lane heads read per-frame features, so the
score says what one picture carries, without any memory.

Recorded frames are BGR uint8 (``python/sensors/cameras.py`` flips the
BeamNG RGB buffer to BGR before the PNG is written, and OpenCV decodes it
back as BGR). The trainer hands the net that BGR canvas scaled to 0-1.
The net, not the cache, swaps it to RGB and applies the ImageNet mean and
std, so recorded frames and the frame cache stay untouched.

The ``SceneState`` the trainer passes is returned unchanged. It is only
there so the call signature matches ``SceneNet``.
"""

from __future__ import annotations

from typing import Any

from python.train.scene_net import (
    GRU_HIDDEN,
    HORIZ_BINS,
    RESIDUAL_S,
    RESIDUAL_TOKENS,
    SceneState,
    scene_head_sizes,
)

BACKBONES = ("scratch", "imagenet-resnet18", "mobilenetv3")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The trainer's canvas is BGR. Index 2 (R), 1 (G), 0 (B) makes RGB.
BGR_TO_RGB = (2, 1, 0)

import torch
from torch import nn


def _resnet18(weights: bool) -> tuple[nn.Module, nn.Module | None, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if weights else None)
    body = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
    )
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), model.fc)
    return body, classifier, 512


def _mobilenetv3(weights: bool) -> tuple[nn.Module, nn.Module | None, int]:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if weights else None)
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), model.classifier)
    return model.features, classifier, 576


def imagenet_categories(name: str) -> list[str]:
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1 if name == "imagenet-resnet18" else MobileNet_V3_Small_Weights.IMAGENET1K_V1
    return list(weights.meta["categories"])


class ImageNetPreprocess(nn.Module):
    """BGR 0-1 in, RGB ImageNet-normalized out. Lives inside the net."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, bgr01: Any) -> Any:
        rgb = bgr01[:, list(BGR_TO_RGB), :, :]
        return (rgb - self.mean.to(rgb.dtype)) / self.std.to(rgb.dtype)


class PretrainedSceneNet(nn.Module):
    """Same call surface as ``SceneNet`` (forward, forward_batch, initial_state).

    Each sector goes through the shared backbone. Its last feature map is
    pooled to ``HORIZ_BINS`` columns (left/right position survives, unlike a
    global average), projected to ``GRU_HIDDEN`` and fed to the same linear
    heads as ``SceneNet``. BatchNorm stays in eval mode (ImageNet statistics):
    the trainer's per-tick batch is a handful of windows, too few for batch
    statistics.
    """

    def __init__(
        self,
        cams: tuple[str, ...] | None = None,
        scale: float = 1.0,
        backbone: str = "imagenet-resnet18",
        *,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        from python.train.cam_strip import parse_cams, scaled_layout

        if backbone == "imagenet-resnet18":
            body, classifier, channels = _resnet18(pretrained)
        elif backbone == "mobilenetv3":
            body, classifier, channels = _mobilenetv3(pretrained)
        else:
            raise ValueError(f"unknown backbone {backbone!r}; pick one of {BACKBONES[1:]}")
        self.backbone_name = backbone
        self.cams = parse_cams(",".join(cams) if cams else "")
        self.scale = 1.0 if float(scale) >= 0.999 else float(scale)
        self.sectors, self.canvas_h, self.canvas_w = scaled_layout(self.cams, self.scale)
        self.freeze = bool(freeze)
        self.preprocess = ImageNetPreprocess()
        self.backbone = body
        # Kept out of the module tree so it is neither trained nor saved:
        # only the one-frame check script uses it for top-5 classes.
        self.__dict__["_classifier"] = classifier
        self.token_dim = len(self.sectors) * channels * HORIZ_BINS
        self.proj = nn.Sequential(nn.Linear(self.token_dim, GRU_HIDDEN), nn.ReLU(inplace=False))
        for name, out in scene_head_sizes():
            setattr(self, name, nn.Linear(GRU_HIDDEN, out))
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        # Pause files from this net must not load into SceneNet, or back.
        self.pause_arch = f"{backbone}:{'frozen' if self.freeze else 'tuned'}:cams=" + ",".join(self.cams)

    def train(self, mode: bool = True) -> "PretrainedSceneNet":
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def initial_state(self, device: Any = None, dtype: Any = None, batch: int = 1) -> SceneState:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        width = max(1, int(batch))
        return SceneState(
            h=torch.zeros(1, width, GRU_HIDDEN, device=device, dtype=dtype),
            tokens=torch.zeros(width, RESIDUAL_TOKENS, 1, device=device, dtype=dtype),
            mask=torch.full((width, RESIDUAL_TOKENS), float(RESIDUAL_S) + 1.0, device=device, dtype=dtype),
        )

    def sector_maps(self, image: Any) -> list[Any]:
        """Backbone feature maps for each sector of a Bx3xHxW BGR 0-1 canvas."""
        x = self.preprocess(image)
        return [self.backbone(x[:, :, :, sec.x0 : sec.x1]) for sec in self.sectors]

    def classify(self, image: Any) -> Any:
        """ImageNet logits of the first sector (sanity check only)."""
        return self._classifier.to(image.device)(self.sector_maps(image)[0])

    def features(self, image: Any) -> Any:
        import torch.nn.functional as functional

        parts = [functional.adaptive_avg_pool2d(fm, (1, HORIZ_BINS)).flatten(1) for fm in self.sector_maps(image)]
        return self.proj(torch.cat(parts, dim=-1))

    def forward(self, strip: Any, dt: Any, state: SceneState) -> tuple[Any, SceneState]:
        from python.train.scene_net import _prediction_at, _strip_nchw

        image = _strip_nchw(strip, width=self.canvas_w, height=self.canvas_h)
        pred, new_state = self._forward_batched(image, dt, state)
        return _prediction_at(pred, 0), new_state

    def forward_batch(self, strip: Any, dt: Any, state: SceneState) -> tuple[Any, SceneState]:
        from python.train.scene_net import _strip_nchw

        image = _strip_nchw(strip, allow_batch=True, width=self.canvas_w, height=self.canvas_h)
        return self._forward_batched(image, dt, state)

    def _forward_batched(self, image: Any, dt: Any, state: SceneState) -> tuple[Any, SceneState]:
        from python.train.scene_net import _heads_batch

        # Single frame: dt and the state are ignored and passed through.
        return _heads_batch(self, self.features(image)), state


def build_scene_net(
    cams: tuple[str, ...] | None,
    scale: float,
    backbone: str = "scratch",
    *,
    freeze: bool = False,
) -> nn.Module:
    """``SceneNet`` for ``scratch``, else the single-frame ImageNet baseline."""
    if backbone in (None, "", "scratch"):
        from python.train.scene_net import SceneNet

        return SceneNet(cams=cams, scale=scale)
    return PretrainedSceneNet(cams=cams, scale=scale, backbone=backbone, freeze=freeze)
