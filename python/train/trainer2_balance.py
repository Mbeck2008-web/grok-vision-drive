"""Trainer 2.0 loss balancing: per-task grad norms, GradNorm, head ramps.

docs/specs/trainer_2.md, "Loss, gradients, LR" (PR 4):

* per-task grad norm ``||dL_i / dW_shared||`` with ``W_shared = stem.fc.weight``
  (cheap), one JSON-ready record per step for the status/trace file;
* GradNorm (Chen et al. 2018): weights ``w_i``, ``alpha = 1.5``, renormalized
  so ``sum(w_i) = n_active`` and clamped to ``[0.1, 10]`` (spec proposals).
  Distill terms are excluded and keep fixed weights;
* gradual activation: a head entering at step ``s0`` gets
  ``ramp(t) = min(1, (t - s0) / R)``, ``R = 2000`` (spec proposal). Its
  effective weight is ``ramp * w_i``. An inactive head computes no loss.

The GradNorm weight step size ``WEIGHT_LR`` = 0.025 is a **proposal** too.
It uses the closed-form gradient of
``L_grad = sum_i |G_i - Gbar * r_i**alpha|`` with ``G_i = w_i * g_i``:
``dL_grad / dw_i = sign(G_i - target_i) * g_i`` (the target is held constant,
as in the paper), so no second backward pass is needed.

Training-only module. Nothing on a live path imports it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

ALPHA = 1.5
W_MIN = 0.1
W_MAX = 10.0
RAMP_STEPS = 2000
# PROPOSAL (not in the spec before this PR; now listed there as a proposal): step size of the GradNorm
# weight update, ``w_i <- w_i - WEIGHT_LR * dL_grad/dw_i``. Untuned; pending a real run.
WEIGHT_LR = 0.025
SHARED_PARAM = "stem.fc.weight"

__all__ = [
    "ALPHA",
    "RAMP_STEPS",
    "SHARED_PARAM",
    "W_MAX",
    "W_MIN",
    "GradNorm",
    "HeadRamp",
    "combine_losses",
    "grad_norm_record",
    "per_task_grad_norms",
    "project_weights",
]


class HeadRamp:
    """Entry step per head and the linear ramp ``min(1, (t - s0) / R)``."""

    def __init__(self, entry_steps: Mapping[str, int], ramp_steps: int = RAMP_STEPS) -> None:
        if int(ramp_steps) < 1:
            raise ValueError(f"ramp_steps must be >= 1, got {ramp_steps!r}")
        self.entry = {str(k): int(v) for k, v in entry_steps.items()}
        self.ramp_steps = int(ramp_steps)

    def active(self, name: str, step: int) -> bool:
        return int(step) >= self.entry[name]

    def factor(self, name: str, step: int) -> float:
        s0 = self.entry[name]
        if int(step) < s0:
            return 0.0
        return min(1.0, float(int(step) - s0) / float(self.ramp_steps))

    def active_names(self, step: int) -> list[str]:
        return [name for name in self.entry if self.active(name, step)]


def project_weights(weights: Mapping[str, float], total: float, lo: float = W_MIN, hi: float = W_MAX) -> dict[str, float]:
    """Scale then clamp so every weight is in ``[lo, hi]`` and they sum to ``total``.

    Finds ``c`` with ``sum(clamp(c * w_i, lo, hi)) == total`` by bisection; the
    order of the weights is kept. ``total`` must be reachable
    (``n * lo <= total <= n * hi``).
    """
    names = list(weights)
    n = len(names)
    if n == 0:
        return {}
    if not (n * lo - 1e-12 <= float(total) <= n * hi + 1e-12):
        raise ValueError(f"sum {total} unreachable for {n} weights in [{lo}, {hi}]")
    raw = [max(float(weights[k]), 1e-12) for k in names]

    def mass(c: float) -> float:
        return sum(min(hi, max(lo, c * w)) for w in raw)

    a, b = 0.0, 1.0
    while mass(b) < total:
        b *= 2.0
        if b > 1e30:
            break
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mass(mid) < total:
            a = mid
        else:
            b = mid
    out = [min(hi, max(lo, b * w)) for w in raw]
    # Spread the bisection residue over the unclamped weights so the sum is exact.
    free = [i for i, v in enumerate(out) if lo < v < hi]
    resid = float(total) - sum(out)
    if free and resid:
        share = resid / len(free)
        for i in free:
            out[i] = min(hi, max(lo, out[i] + share))
    return dict(zip(names, out))


class GradNorm:
    """Learnable task weights for the GradNorm-balanced heads. Distill terms stay fixed."""

    def __init__(
        self,
        names: Iterable[str],
        *,
        alpha: float = ALPHA,
        lr: float = WEIGHT_LR,
        fixed: Mapping[str, float] | None = None,
    ) -> None:
        self.weights: dict[str, float] = {str(n): 1.0 for n in names}
        self.fixed: dict[str, float] = {str(k): float(v) for k, v in (fixed or {}).items()}
        overlap = sorted(set(self.weights) & set(self.fixed))
        if overlap:
            raise ValueError(f"term(s) both balanced and fixed: {overlap}")
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.initial: dict[str, float] = {}

    def weight_of(self, name: str) -> float:
        if name in self.fixed:
            return self.fixed[name]
        return self.weights[name]

    def update(self, losses: Mapping[str, float], grad_norms: Mapping[str, float], *, active: Iterable[str]) -> dict[str, float]:
        """One GradNorm weight step over the ``active`` heads; returns the new weights.

        ``losses`` are this step's (unweighted) task losses, ``grad_norms`` the
        unweighted ``||dL_i / dW_shared||``. A head's first active loss is its
        ``L_i(0)``. Weights of inactive heads are left as they are; the active
        ones are projected to ``sum == n_active`` inside the clamp.
        """
        names = [str(n) for n in active]
        for name in names:
            if name in self.fixed or name not in self.weights:
                raise ValueError(f"{name!r} is not a GradNorm-balanced term")
            for what, src in (("loss", losses), ("grad norm", grad_norms)):
                val = float(src[name])
                if not math.isfinite(val) or val < 0.0:
                    raise ValueError(f"{what} of {name!r} is {val!r}")
        for name in names:
            self.initial.setdefault(name, max(float(losses[name]), 1e-12))
        if len(names) < 2:
            return dict(self.weights)
        g = {n: float(grad_norms[n]) for n in names}
        big_g = {n: self.weights[n] * g[n] for n in names}
        g_bar = sum(big_g.values()) / len(names)
        ratio = {n: float(losses[n]) / self.initial[n] for n in names}
        r_bar = sum(ratio.values()) / len(names)
        new = {}
        for n in names:
            r = ratio[n] / r_bar if r_bar > 0 else 1.0
            target = g_bar * (r ** self.alpha)
            diff = big_g[n] - target
            grad = (1.0 if diff > 0 else -1.0 if diff < 0 else 0.0) * g[n]
            new[n] = self.weights[n] - self.lr * grad
        self.weights.update(project_weights(new, float(len(names))))
        return dict(self.weights)


def per_task_grad_norms(losses: Mapping[str, Any], shared: Any) -> dict[str, float]:
    """``||dL_i / d shared||`` per term, without touching ``shared.grad``."""
    import torch

    out: dict[str, float] = {}
    for name, loss in losses.items():
        (grad,) = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        out[str(name)] = 0.0 if grad is None else float(grad.detach().norm())
    return out


def combine_losses(
    losses: Mapping[str, Any],
    balance: GradNorm,
    ramp: HeadRamp,
    *,
    step: int,
    fixed_losses: Mapping[str, Any] | None = None,
) -> Any:
    """``sum(ramp_i * w_i * L_i)`` over active heads, plus fixed-weight (distill) terms."""
    total: Any = 0.0
    for name, loss in losses.items():
        if not ramp.active(name, step):
            continue
        total = total + ramp.factor(name, step) * balance.weight_of(name) * loss
    for name, loss in (fixed_losses or {}).items():
        if name in balance.weights:
            raise ValueError(f"{name!r} is GradNorm-balanced, not a fixed term")
        weight = balance.fixed.get(name, 1.0)
        total = total + weight * loss
    return total


def grad_norm_record(step: int, grad_norms: Mapping[str, float], balance: GradNorm, ramp: HeadRamp) -> dict[str, Any]:
    """One per-step trace entry: grad norm, weight and ramp per active head."""
    names = sorted(grad_norms)
    return {
        "step": int(step),
        "shared": SHARED_PARAM,
        "grad_norm": {n: float(grad_norms[n]) for n in names},
        "weight": {n: float(balance.weight_of(n)) for n in names},
        "ramp": {n: float(ramp.factor(n, step)) for n in names},
    }
