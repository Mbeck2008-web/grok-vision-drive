"""Opt-in edge, object, and brake-light heads on the scene-net feature.

Off unless the trainer names them. A default ``SceneNet`` does not construct
this module, so its checkpoint stays the same. Losses are masked: a missing
label adds no geometry gradient.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from python.train.detect_targets import EDGE_DIM, OBJ_CLASSES, OBJ_FEAT, OBJ_SLOTS

CONF_WEIGHT = 0.1


class DetectHeads(torch.nn.Module):
    """Linear heads on the GRU feature. Logits, not probabilities."""

    def __init__(
        self,
        *,
        edge: bool = False,
        obj: bool = False,
        brakelight: bool = False,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.use_edge = bool(edge)
        self.use_obj = bool(obj)
        self.use_brake = bool(brakelight)
        width = int(hidden)
        if self.use_edge:
            self.edge_head = torch.nn.Linear(width, EDGE_DIM + 1)
        if self.use_obj:
            self.obj_head = torch.nn.Linear(width, OBJ_SLOTS * (OBJ_FEAT + 1 + len(OBJ_CLASSES)))
        if self.use_brake:
            self.brake_head = torch.nn.Linear(width, 2)

    def forward(self, feat: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.use_edge:
            raw = self.edge_head(feat)
            out["edge"] = raw[..., :EDGE_DIM]
            out["edge_conf"] = raw[..., EDGE_DIM]
        if self.use_obj:
            width = int(feat.shape[0])
            raw = self.obj_head(feat).view(width, OBJ_SLOTS, OBJ_FEAT + 1 + len(OBJ_CLASSES))
            out["obj_box"] = raw[..., :OBJ_FEAT]
            out["obj_conf"] = raw[..., OBJ_FEAT]
            out["obj_class"] = raw[..., OBJ_FEAT + 1 :]
        if self.use_brake:
            raw = self.brake_head(feat)
            out["brake"] = raw[..., 0]
            out["brake_conf"] = raw[..., 1]
        return out


def _zero(device: Any, dtype: Any) -> Any:
    return torch.zeros((), device=device, dtype=dtype)


def _batch_rows(pred: Any, tail: int) -> Any:
    """(tail,) or (B, tail) → (B, tail)."""
    if pred.dim() == tail:
        return pred.unsqueeze(0)
    return pred


def _target(value: Any, device: Any, dtype: Any) -> Any | None:
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=dtype)


def _long(value: Any, device: Any) -> Any | None:
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.long)


def geometry_smooth(pred: Any, goal: Any, mask: Any) -> tuple[Any, Any]:
    """Masked smooth-L1. A fully masked target is a constant 0 with no gradient."""
    device = pred.device
    dtype = pred.dtype
    if goal is None or mask is None:
        return _zero(device, dtype), _zero(device, dtype)
    goal_t = torch.as_tensor(goal, device=device, dtype=dtype)
    mask_t = torch.as_tensor(mask, device=device, dtype=dtype)
    pred_b = pred.reshape(pred.shape[0], -1) if pred.dim() > 1 else pred.reshape(1, -1)
    goal_b = goal_t.reshape(pred_b.shape)
    mask_b = mask_t.reshape(pred_b.shape)
    labeled = (mask_b > 0.5).any(dim=1).to(dtype).sum()
    if not bool((mask_b > 0.5).any()):
        return _zero(device, dtype), labeled
    diff = functional.smooth_l1_loss(pred_b, torch.nan_to_num(goal_b, nan=0.0), reduction="none")
    loss = (diff * mask_b).sum() / mask_b.sum().clamp(min=1.0)
    return loss, labeled


def _conf_bce(logit: Any, present: Any) -> Any:
    return functional.binary_cross_entropy_with_logits(logit.float(), present.float())


def detect_terms(pred: Any, target: Mapping[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    """Per-head training terms. ``*_geom`` ignores unlabeled rows. ``*_n`` counts them."""
    zero = _zero(device, dtype)
    out: dict[str, Any] = {}
    edge = getattr(pred, "edge", None)
    if edge is None:
        out["edge"] = zero
        out["edge_geom"] = zero
        out["edge_n"] = zero
    else:
        pred_e = _batch_rows(edge.to(device=device, dtype=dtype), 1)
        if pred_e.dim() == 1:
            pred_e = pred_e.unsqueeze(0)
        geom, n = geometry_smooth(pred_e, target.get("edge"), target.get("edge_valid"))
        conf_logit = getattr(pred, "edge_conf", None)
        conf = zero
        if conf_logit is not None and target.get("edge_valid") is not None:
            valid = _target(target.get("edge_valid"), device, dtype)
            assert valid is not None
            present = (valid.reshape(pred_e.shape[0], -1) > 0.5).any(dim=1).to(dtype)
            conf = _conf_bce(conf_logit.reshape(-1).to(device=device, dtype=dtype), present)
        out["edge_geom"] = geom
        out["edge_n"] = n
        out["edge"] = geom + CONF_WEIGHT * conf
    box = getattr(pred, "obj_box", None)
    if box is None:
        out["obj"] = zero
        out["obj_geom"] = zero
        out["obj_n"] = zero
    else:
        pred_b = box.to(device=device, dtype=dtype)
        if pred_b.dim() == 2:
            pred_b = pred_b.unsqueeze(0)
        geom, n = geometry_smooth(pred_b, target.get("obj_box"), target.get("obj_valid"))
        cls = zero
        logits = getattr(pred, "obj_class", None)
        goal_cls = _long(target.get("obj_class"), device)
        valid = _target(target.get("obj_valid"), device, dtype)
        if logits is not None and goal_cls is not None and valid is not None:
            logit_b = logits.to(device=device)
            if logit_b.dim() == 2:
                logit_b = logit_b.unsqueeze(0)
            slot = (valid.reshape(pred_b.shape[0], pred_b.shape[1], -1) > 0.5).any(dim=-1)
            if bool(slot.any()):
                cls = functional.cross_entropy(
                    logit_b.reshape(-1, logit_b.shape[-1])[slot.reshape(-1)],
                    goal_cls.reshape(-1)[slot.reshape(-1)],
                )
        conf = zero
        conf_logit = getattr(pred, "obj_conf", None)
        if conf_logit is not None and valid is not None:
            present = (valid.reshape(pred_b.shape[0], pred_b.shape[1], -1) > 0.5).any(dim=-1).to(dtype)
            conf = _conf_bce(conf_logit.reshape_as(present).to(dtype), present)
        out["obj_geom"] = geom + cls
        out["obj_n"] = n
        out["obj"] = out["obj_geom"] + CONF_WEIGHT * conf
    brake = getattr(pred, "brake", None)
    if brake is None:
        out["brakelight"] = zero
        out["brakelight_geom"] = zero
        out["brakelight_n"] = zero
    else:
        logit = brake.to(device=device, dtype=dtype).reshape(-1)
        goal = _target(target.get("brake"), device, dtype)
        valid = _target(target.get("brake_valid"), device, dtype)
        if goal is None or valid is None:
            geom = zero
            n = zero
            conf = zero
        else:
            goal_b = goal.reshape(-1)
            valid_b = valid.reshape(-1)
            if logit.shape[0] == 1 and goal_b.shape[0] != 1:
                goal_b = goal_b[:1]
                valid_b = valid_b[:1]
            on = valid_b > 0.5
            n = on.to(dtype).sum()
            if bool(on.any()):
                geom = functional.binary_cross_entropy_with_logits(logit[on], goal_b[on])
            else:
                geom = zero
            conf_logit = getattr(pred, "brake_conf", None)
            conf = zero if conf_logit is None else _conf_bce(conf_logit.reshape(-1).to(dtype), on.to(dtype))
        out["brakelight_geom"] = geom
        out["brakelight_n"] = n
        out["brakelight"] = geom + CONF_WEIGHT * conf
    return out
