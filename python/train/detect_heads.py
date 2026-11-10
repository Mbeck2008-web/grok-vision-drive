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


def _dead(device: Any, dtype: Any) -> Any:
    """A constant 0 that is not part of the autograd graph."""
    return torch.zeros((), device=device, dtype=dtype).detach()


def _row_flag(value: Any, rows: int, device: Any) -> Any:
    """Broadcast a 0/1 recorder flag to ``(rows,)`` bool. Missing means false."""
    if value is None:
        return torch.zeros(rows, device=device, dtype=torch.bool)
    flag = torch.as_tensor(value, device=device).reshape(-1) > 0.5
    if int(flag.numel()) == 1 and rows != 1:
        flag = flag.expand(rows)
    if int(flag.numel()) != rows:
        flag = flag[:rows]
    return flag


def detect_terms(pred: Any, target: Mapping[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    """Per-head terms. A head with no provided label is a detached 0, conf included.

    Presence BCE and class CE run only on rows or slots the recorder actually
    labeled. Empty object slots are negatives only when that frame's traffic
    or static list was present (``obj_listed``). Geometry stays a mean over
    the valid elements, so a partial mask is not diluted by the zeros.
    """
    dead = _dead(device, dtype)
    out: dict[str, Any] = {}
    edge = getattr(pred, "edge", None)
    if edge is None:
        out["edge"] = dead
        out["edge_geom"] = dead
        out["edge_n"] = dead
    else:
        pred_e = edge.to(device=device, dtype=dtype)
        if pred_e.dim() == 1:
            pred_e = pred_e.unsqueeze(0)
        valid = _target(target.get("edge_valid"), device, dtype)
        labeled = False if valid is None else bool((valid.reshape(pred_e.shape[0], -1) > 0.5).any())
        if not labeled:
            out["edge"] = dead
            out["edge_geom"] = dead
            out["edge_n"] = dead
        else:
            geom, n = geometry_smooth(pred_e, target.get("edge"), valid)
            present = (valid.reshape(pred_e.shape[0], -1) > 0.5).any(dim=1)
            conf_logit = getattr(pred, "edge_conf", None)
            conf = dead
            if conf_logit is not None and bool(present.any()):
                conf = _conf_bce(
                    conf_logit.reshape(-1).to(device=device, dtype=dtype)[present],
                    torch.ones(int(present.sum()), device=device, dtype=dtype),
                )
            out["edge_geom"] = geom
            out["edge_n"] = n
            out["edge"] = geom + float(CONF_WEIGHT) * conf
    box = getattr(pred, "obj_box", None)
    if box is None:
        out["obj"] = dead
        out["obj_geom"] = dead
        out["obj_n"] = dead
    else:
        pred_b = box.to(device=device, dtype=dtype)
        if pred_b.dim() == 2:
            pred_b = pred_b.unsqueeze(0)
        valid = _target(target.get("obj_valid"), device, dtype)
        if valid is None:
            out["obj"] = dead
            out["obj_geom"] = dead
            out["obj_n"] = dead
        else:
            valid_b = valid.reshape(pred_b.shape[0], pred_b.shape[1], -1)
            slot = (valid_b > 0.5).any(dim=-1)
            listed = _row_flag(target.get("obj_listed"), int(pred_b.shape[0]), device)
            if not bool(slot.any()) and not bool(listed.any()):
                out["obj"] = dead
                out["obj_geom"] = dead
                out["obj_n"] = dead
            else:
                geom, n = geometry_smooth(pred_b, target.get("obj_box"), valid)
                if not bool(slot.any()):
                    geom = dead
                cls = dead
                logits = getattr(pred, "obj_class", None)
                goal_cls = _long(target.get("obj_class"), device)
                if logits is not None and goal_cls is not None and bool(slot.any()):
                    logit_b = logits.to(device=device)
                    if logit_b.dim() == 2:
                        logit_b = logit_b.unsqueeze(0)
                    cls = functional.cross_entropy(
                        logit_b.reshape(-1, logit_b.shape[-1])[slot.reshape(-1)],
                        goal_cls.reshape(-1)[slot.reshape(-1)],
                    )
                conf = dead
                conf_logit = getattr(pred, "obj_conf", None)
                if conf_logit is not None:
                    supervise = listed.unsqueeze(1).expand_as(slot) | slot
                    if bool(supervise.any()):
                        conf = _conf_bce(
                            conf_logit.to(device=device, dtype=dtype).reshape_as(slot)[supervise],
                            slot.to(dtype)[supervise],
                        )
                out["obj_geom"] = geom
                out["obj_n"] = n
                out["obj"] = geom + cls + float(CONF_WEIGHT) * conf
    brake = getattr(pred, "brake", None)
    if brake is None:
        out["brakelight"] = dead
        out["brakelight_geom"] = dead
        out["brakelight_n"] = dead
    else:
        logit = brake.to(device=device, dtype=dtype).reshape(-1)
        goal = _target(target.get("brake"), device, dtype)
        valid = _target(target.get("brake_valid"), device, dtype)
        if goal is None or valid is None or not bool((valid.reshape(-1) > 0.5).any()):
            out["brakelight"] = dead
            out["brakelight_geom"] = dead
            out["brakelight_n"] = dead
        else:
            goal_b = goal.reshape(-1)
            valid_b = valid.reshape(-1)
            if logit.shape[0] == 1 and goal_b.shape[0] != 1:
                goal_b = goal_b[:1]
                valid_b = valid_b[:1]
            on = valid_b > 0.5
            geom = functional.binary_cross_entropy_with_logits(logit[on], goal_b[on])
            conf_logit = getattr(pred, "brake_conf", None)
            conf = dead
            if conf_logit is not None:
                conf = _conf_bce(
                    conf_logit.reshape(-1).to(device=device, dtype=dtype)[on],
                    torch.ones(int(on.sum()), device=device, dtype=dtype),
                )
            out["brakelight_geom"] = geom
            out["brakelight_n"] = on.to(dtype).sum()
            out["brakelight"] = geom + float(CONF_WEIGHT) * conf
    return out
