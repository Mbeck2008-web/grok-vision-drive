"""Per-ego-model camera mounts on top of config/cameras.yaml.

cameras.yaml is calibrated on the etk800. A ``vehicle_overrides`` block keyed by
model id holds per-camera keys that replace the base camera's keys for that
model only:

    vehicle_overrides:
      sbr:
        repeatL: {pos_m: [...]}

``rig_for_model`` drops the block in every case, so a model without an entry
(the etk800, or an unknown car) gets exactly the base rig.
"""
from __future__ import annotations

from typing import Any, Mapping

BASE_RIG_MODEL = "etk800"
OVERRIDES_KEY = "vehicle_overrides"


def _overrides(cfg: Mapping[str, Any] | None, model: str | None) -> dict[str, Any]:
    block = (cfg or {}).get(OVERRIDES_KEY)
    if not isinstance(block, Mapping) or not model:
        return {}
    entry = block.get(str(model).strip().lower())
    return dict(entry) if isinstance(entry, Mapping) else {}


def rig_overrides(cfg: Mapping[str, Any] | None, model: str | None) -> dict[str, Any]:
    """The per-camera patches applied for ``model`` ({} for the base rig), for the run record."""
    return {str(cid): dict(p) for cid, p in _overrides(cfg, model).items() if isinstance(p, Mapping)}


def rig_model(cfg: Mapping[str, Any] | None, model: str | None) -> str:
    """The model whose mounts ``rig_for_model(cfg, model)`` uses."""
    return str(model).strip().lower() if _overrides(cfg, model) else BASE_RIG_MODEL


def rig_for_model(cfg: Mapping[str, Any] | None, model: str | None) -> dict[str, Any]:
    """The camera config for ``model``: the base rig with that model's overrides merged per camera id."""
    out = {key: value for key, value in dict(cfg or {}).items() if key != OVERRIDES_KEY}
    over = _overrides(cfg, model)
    if not over:
        return out
    cams: list[Any] = []
    for spec in out.get("cameras") or []:
        patch = over.get(str(spec.get("id") or "")) if isinstance(spec, Mapping) else None
        cams.append({**spec, **patch} if isinstance(patch, Mapping) else spec)
    out["cameras"] = cams
    return out
