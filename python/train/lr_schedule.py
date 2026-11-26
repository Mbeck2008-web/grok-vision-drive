"""Trainer 2.0 learning-rate schedule and parameter groups (docs/specs/trainer_2.md, PR 3).

Per stage: linear warmup over ``max(500, 5%)`` of the stage's steps, then a
cosine down to ``0.05 x peak``. The joint stage peaks at ``0.1 x`` the base
rate. All three figures are spec proposals.

The schedule composes with ``train_scene._pin_optimizer_lr``: every group's
rate is ``schedule(step) * group["lr_mult"]``, the same ``rate * lr_mult``
rule the pin uses, so a resume that re-pins the CLI rate and then re-applies
the schedule lands on the scheduled rate.

``scene_param_groups`` splits a ``SceneNet`` into AdamW groups (stem, trunk,
one per head) with ``lr_mult``. It is a free function, not a ``SceneNet``
method: ``train_scene`` calls ``net.param_groups`` whenever a net has one and
``--backbone-lr-mult != 1`` (train_scene.py ``hasattr(..., "param_groups")``),
so a new method would silently change plain scratch-net runs.

Training-only module. Nothing on a live path imports it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

WARMUP_MIN_STEPS = 500
WARMUP_FRAC = 0.05
FLOOR_FRAC = 0.05
JOINT_PEAK_MULT = 0.1
STEM_MULT_AFTER_SSL = 0.1
TRAINED_HEAD_MULT = 0.3
ENTERING_HEAD_MULT = 1.0
TRUNK_MULT = 1.0

__all__ = [
    "ENTERING_HEAD_MULT",
    "FLOOR_FRAC",
    "JOINT_PEAK_MULT",
    "STEM_MULT_AFTER_SSL",
    "TRAINED_HEAD_MULT",
    "WARMUP_FRAC",
    "WARMUP_MIN_STEPS",
    "WarmupCosine",
    "apply_lr",
    "scene_param_groups",
    "stage_lr_mults",
    "stage_peak",
]


class WarmupCosine:
    """Linear warmup to ``peak``, then cosine to ``floor_frac * peak`` at ``total_steps``."""

    def __init__(
        self,
        peak: float,
        total_steps: int,
        *,
        warmup_steps: int | None = None,
        floor_frac: float = FLOOR_FRAC,
    ) -> None:
        if not (float(peak) > 0.0 and math.isfinite(float(peak))):
            raise ValueError(f"peak lr must be a positive number, got {peak!r}")
        if int(total_steps) < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps!r}")
        if not 0.0 <= float(floor_frac) <= 1.0:
            raise ValueError(f"floor_frac must be in [0, 1], got {floor_frac!r}")
        self.peak = float(peak)
        self.total_steps = int(total_steps)
        default = max(WARMUP_MIN_STEPS, int(math.ceil(WARMUP_FRAC * self.total_steps)))
        warm = default if warmup_steps is None else int(warmup_steps)
        # A stage shorter than the warmup warms up for its whole length.
        self.warmup_steps = max(0, min(warm, self.total_steps))
        self.floor_frac = float(floor_frac)

    def lr_at(self, step: int) -> float:
        """Rate for optimizer step ``step`` (0-based). Past the end it holds the floor."""
        s = max(0, int(step))
        floor = self.floor_frac * self.peak
        if s < self.warmup_steps:
            return self.peak * float(s + 1) / float(self.warmup_steps)
        decay = self.total_steps - self.warmup_steps
        if decay <= 0:
            return self.peak
        frac = min(1.0, float(s - self.warmup_steps) / float(decay))
        return floor + (self.peak - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak": self.peak,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "floor_frac": self.floor_frac,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "WarmupCosine":
        return cls(
            float(state["peak"]),
            int(state["total_steps"]),
            warmup_steps=int(state["warmup_steps"]),
            floor_frac=float(state["floor_frac"]),
        )


def stage_peak(base_lr: float, *, joint: bool) -> float:
    """Peak rate of a stage: the base (CLI) rate, or ``0.1 x`` for the joint stage."""
    return float(base_lr) * (JOINT_PEAK_MULT if joint else 1.0)


def apply_lr(opt: Any, schedule: WarmupCosine, step: int) -> float:
    """Write ``schedule(step) * lr_mult`` onto every group (``_pin_optimizer_lr`` rule). Returns the rate."""
    rate = schedule.lr_at(step)
    for group in getattr(opt, "param_groups", []) or []:
        if isinstance(group, dict):
            group["lr"] = rate * float(group.get("lr_mult", 1.0))
    return rate


def stage_lr_mults(
    *,
    ssl_done: bool,
    trained_heads: Iterable[str] = (),
    entering_heads: Iterable[str] = (),
) -> dict[str, float]:
    """``lr_mult`` per group name: stem 0.1 after stage 0, trained heads 0.3, entering 1.0.

    The trunk (GRU and anything that is neither stem nor a head) keeps 1.0;
    the spec gives it no number. A head both trained and entering is refused.
    """
    trained = set(trained_heads)
    entering = set(entering_heads)
    both = sorted(trained & entering)
    if both:
        raise ValueError(f"head(s) both trained and entering: {both}")
    mults: dict[str, float] = {"stem": STEM_MULT_AFTER_SSL if ssl_done else 1.0, "trunk": TRUNK_MULT}
    mults.update({name: TRAINED_HEAD_MULT for name in trained})
    mults.update({name: ENTERING_HEAD_MULT for name in entering})
    return mults


def _group_of(param_name: str, heads: set[str]) -> str:
    top = param_name.split(".", 1)[0]
    if top == "stem":
        return "stem"
    if top in heads:
        return top
    return "trunk"


def scene_param_groups(net: Any, lr: float, mults: Mapping[str, float]) -> list[dict[str, Any]]:
    """AdamW groups ``{params, lr, lr_mult, name}``: stem, trunk, then each ``*_head`` child.

    Every trainable parameter lands in exactly one group. A group name missing
    from ``mults`` gets 1.0. A name in ``mults`` that the net does not have is
    refused (a typo would otherwise train that head at the wrong rate).
    """
    heads = {name for name, _m in net.named_children() if name.endswith("_head")}
    known = heads | {"stem", "trunk"}
    unknown = sorted(set(mults) - known)
    if unknown:
        raise ValueError(f"lr_mult for unknown group(s) {unknown}; net has {sorted(known)}")
    order = ["stem", "trunk"] + [name for name, _m in net.named_children() if name in heads]
    buckets: dict[str, list[Any]] = {name: [] for name in order}
    for pname, param in net.named_parameters():
        if param.requires_grad:
            buckets[_group_of(pname, heads)].append(param)
    groups = []
    for name in order:
        if not buckets[name]:
            continue
        mult = float(mults.get(name, 1.0))
        groups.append({"params": buckets[name], "lr": float(lr) * mult, "lr_mult": mult, "name": name})
    return groups
