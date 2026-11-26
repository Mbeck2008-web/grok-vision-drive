#!/usr/bin/env python3
"""Trainer 2.0 PR 4: per-task grad norms, GradNorm, head ramps (docs/specs/trainer_2.md).

Checks: check_per_task_grad_norm_logged, check_gradnorm_weights_sum_and_clamp,
check_head_ramp_linear. Torch parts skip visibly without torch.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402


def check_head_ramp_linear() -> None:
    from python.train.trainer2_balance import RAMP_STEPS, HeadRamp

    assert RAMP_STEPS == 2000
    ramp = HeadRamp({"lane_gt": 0, "sign": 1000})
    assert ramp.factor("lane_gt", 0) == 0.0 and ramp.active("lane_gt", 0)
    assert ramp.factor("lane_gt", 1000) == 0.5
    assert ramp.factor("lane_gt", 2000) == 1.0 and ramp.factor("lane_gt", 10**6) == 1.0
    assert not ramp.active("sign", 999) and ramp.factor("sign", 999) == 0.0
    assert ramp.active("sign", 1000)
    vals = [ramp.factor("sign", s) for s in range(1000, 3101)]
    diffs = {round(b - a, 12) for a, b in zip(vals, vals[1:])}
    assert diffs == {round(1.0 / 2000, 12), 0.0} and vals[-1] == 1.0, diffs  # linear, then flat at 1.0
    assert vals.index(1.0) == 2000
    assert ramp.active_names(500) == ["lane_gt"]
    assert ramp.active_names(1500) == ["lane_gt", "sign"]
    try:
        ramp.factor("curb", 0)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown head not refused")
    print("head ramp: inactive before s0, (t-s0)/2000 linear, 1.0 from s0+2000 OK")


def check_gradnorm_weights_sum_and_clamp() -> None:
    from python.train.trainer2_balance import ALPHA, W_MAX, W_MIN, GradNorm, project_weights

    assert ALPHA == 1.5 and (W_MIN, W_MAX) == (0.1, 10.0)
    # Projection: sum == n, every weight in the clamp, order kept.
    w = project_weights({"a": 1e-6, "b": 1.0, "c": 50.0, "d": 3.0}, 4)
    assert math.isclose(sum(w.values()), 4.0, rel_tol=1e-9), w
    assert all(W_MIN - 1e-12 <= v <= W_MAX + 1e-12 for v in w.values()), w
    assert w["a"] <= w["b"] <= w["d"] <= w["c"], w
    w2 = project_weights({"a": 1.0, "b": 1000.0}, 2)
    assert math.isclose(w2["a"], 0.1, abs_tol=1e-9) and math.isclose(w2["b"], 1.9, abs_tol=1e-9), w2

    gn = GradNorm(["lane_gt", "sign", "aux_depth"], fixed={"kd_feat": 1.0})
    assert gn.weights == {"lane_gt": 1.0, "sign": 1.0, "aux_depth": 1.0}
    # lane_gt's loss falls fast, sign's barely moves: sign trains slower so it gets more weight.
    first = {"lane_gt": 2.0, "sign": 2.0, "aux_depth": 2.0}
    for step in range(400):
        losses = {"lane_gt": 2.0 * 0.99 ** step, "sign": 2.0 * 0.9995 ** step, "aux_depth": 2.0 * 0.997 ** step}
        norms = {"lane_gt": 1.0, "sign": 1.0, "aux_depth": 1.0}
        gn.update(losses if step else first, norms, active=["lane_gt", "sign", "aux_depth"])
        assert math.isclose(sum(gn.weights.values()), 3.0, rel_tol=1e-9), gn.weights
        assert all(W_MIN - 1e-12 <= v <= W_MAX + 1e-12 for v in gn.weights.values()), gn.weights
    assert gn.weights["sign"] > gn.weights["aux_depth"] > gn.weights["lane_gt"], gn.weights
    # Distill terms are excluded from GradNorm: fixed weight, never updated, not in the sum.
    assert gn.weight_of("kd_feat") == 1.0 and "kd_feat" not in gn.weights
    try:
        gn.update({"kd_feat": 1.0}, {"kd_feat": 1.0}, active=["kd_feat"])
    except ValueError:
        pass
    else:
        raise AssertionError("a fixed (distill) term was accepted by GradNorm")
    # A head entering later is renormalized over the active set only (sum == n_active).
    gn2 = GradNorm(["lane_gt", "sign"])
    gn2.update({"lane_gt": 1.0}, {"lane_gt": 5.0}, active=["lane_gt"])
    assert gn2.weights["lane_gt"] == 1.0 and gn2.weights["sign"] == 1.0
    gn2.update({"lane_gt": 0.5, "sign": 3.0}, {"lane_gt": 5.0, "sign": 0.01}, active=["lane_gt", "sign"])
    assert math.isclose(sum(gn2.weights.values()), 2.0, rel_tol=1e-9), gn2.weights
    # Extreme gradients drive weights into the clamp; the sum still holds.
    gn3 = GradNorm(["a", "b", "c"], lr=10.0)
    for _ in range(50):
        gn3.update({"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1e4, "b": 1e-4, "c": 1.0}, active=["a", "b", "c"])
    assert math.isclose(gn3.weights["a"], W_MIN, abs_tol=1e-9), gn3.weights
    assert math.isclose(sum(gn3.weights.values()), 3.0, rel_tol=1e-9), gn3.weights
    for bad in ({"a": float("nan"), "b": 1.0, "c": 1.0}, {"a": -1.0, "b": 1.0, "c": 1.0}):
        try:
            gn3.update(bad, {"a": 1.0, "b": 1.0, "c": 1.0}, active=["a", "b", "c"])
        except ValueError:
            continue
        raise AssertionError(f"bad losses {bad} accepted")
    print(f"gradnorm: sum == n_active, clamp [0.1, 10], slow task up-weighted {gn.weights} OK")


def check_per_task_grad_norm_logged() -> None:
    try:
        import torch
    except ImportError:
        skip_check("check_per_task_grad_norm_logged", "torch not installed")
        return
    from python.train.scene_net import SceneNet
    from python.train.trainer2_balance import GradNorm, HeadRamp, combine_losses, grad_norm_record, per_task_grad_norms

    torch.manual_seed(0)
    net = SceneNet()
    x = torch.rand(1, 3, int(net.canvas_h), int(net.canvas_w))
    feat = net.stem(x)
    losses = {
        "lane_gt": (feat ** 2).mean(),
        "sign": (feat - 1.0).abs().mean(),
        "aux_depth": feat.sum() * 0.0 + (feat[..., :3] ** 3).mean(),
    }
    shared = net.stem.fc.weight
    norms = per_task_grad_norms(losses, shared)
    for name, loss in losses.items():
        (grad,) = torch.autograd.grad(loss, shared, retain_graph=True)
        assert math.isclose(norms[name], float(grad.norm()), rel_tol=1e-6), (name, norms[name], float(grad.norm()))
    assert shared.grad is None, "per_task_grad_norms must not accumulate into .grad"
    gn = GradNorm(list(losses))
    ramp = HeadRamp({"lane_gt": 0, "sign": 0, "aux_depth": 100})
    active = ramp.active_names(50)
    assert active == ["lane_gt", "sign"]
    total = combine_losses(losses, gn, ramp, step=50, fixed_losses={"kd_feat": feat.mean() * 0.0 + 0.25})
    want = (50 / 2000) * float(losses["lane_gt"].detach()) + (50 / 2000) * float(losses["sign"].detach()) + 0.25
    assert math.isclose(float(total), want, rel_tol=1e-6), (float(total), want)
    rec = grad_norm_record(50, {k: norms[k] for k in active}, gn, ramp)
    text = json.dumps(rec, allow_nan=False)
    back = json.loads(text)
    assert back["step"] == 50 and set(back["grad_norm"]) == set(active), back
    assert set(back["weight"]) == set(active) and set(back["ramp"]) == set(active), back
    assert all(math.isfinite(v) and v > 0 for v in back["grad_norm"].values()), back
    assert back["shared"] == "stem.fc.weight"
    print(f"per-task grad norm: ||dL_i/d stem.fc.weight|| matches autograd for {sorted(losses)}; record JSON {text[:80]}... OK")


def main() -> int:
    checks = (check_head_ramp_linear, check_gradnorm_weights_sum_and_clamp, check_per_task_grad_norm_logged)
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
