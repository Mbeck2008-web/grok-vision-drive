#!/usr/bin/env python3
"""Trainer 2.0 PR 4: per-task grad norms, GradNorm, head ramps (docs/specs/trainer_2.md).

Checks: check_per_task_grad_norm_logged, check_gradnorm_weights_sum_and_clamp,
check_head_ramp_linear, check_gradnorm_one_step_hand_computed (hand-computed weights),
check_project_weights_clamp_exact, check_combine_losses_ramp_and_inactive,
check_gradnorm_lone_head_renormalizes_to_active_count,
check_grad_norm_record_weights_after_update, check_combine_refuses_balanced_head_as_fixed_term. Torch parts skip visibly without torch.
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
    assert math.isclose(float(total.detach()), want, rel_tol=1e-6), (float(total.detach()), want)
    rec = grad_norm_record(50, {k: norms[k] for k in active}, gn, ramp)
    text = json.dumps(rec, allow_nan=False)
    back = json.loads(text)
    assert back["step"] == 50 and set(back["grad_norm"]) == set(active), back
    assert set(back["weight"]) == set(active) and set(back["ramp"]) == set(active), back
    assert all(math.isfinite(v) and v > 0 for v in back["grad_norm"].values()), back
    assert back["shared"] == "stem.fc.weight"
    print(f"per-task grad norm: ||dL_i/d stem.fc.weight|| matches autograd for {sorted(losses)}; record JSON {text[:80]}... OK")


def _gn(names, *, lr, **kw):
    from python.train.trainer2_balance import GradNorm

    return GradNorm(names, lr=lr, **kw)


def check_gradnorm_one_step_hand_computed() -> None:
    """GradNorm weight step on two heads, every number worked by hand (no clamp binds, lr 0.1).

    dL_grad/dw_i = sign(G_i - Gbar * r_i**alpha) * g_i, with G_i = w_i * g_i (the *weighted* norm sets
    the sign, the *raw* norm g_i the size), r_i = (L_i/L_i(0)) / mean(L/L(0)); then w <- w - lr * grad,
    projected to sum == 2.
    """
    from python.train.trainer2_balance import ALPHA, WEIGHT_LR

    assert ALPHA == 1.5
    assert WEIGHT_LR == 0.025  # PROPOSAL (spec: "spec proposal"); pinned only so a tuning change is deliberate
    close = lambda got, want: math.isclose(got, want, rel_tol=1e-12)  # noqa: E731

    # Step 1, equal losses so r = (1, 1): G = (3, 1), Gbar = 2, targets (2, 2) -> signs (+, -),
    # grad = (+3, -1), w = (1 - 0.3, 1 + 0.1) = (0.7, 1.1), sum 1.8 -> x 2/1.8 = (7/9, 11/9).
    gn = _gn(["a", "b"], lr=0.1)
    out = gn.update({"a": 2.0, "b": 2.0}, {"a": 3.0, "b": 1.0}, active=["a", "b"])
    assert close(out["a"], 7.0 / 9.0) and close(out["b"], 11.0 / 9.0), out
    assert gn.initial == {"a": 2.0, "b": 2.0}

    # Step 2 with unequal weights (G != g): L = (1, 2) so L/L0 = (0.5, 1), r = (2/3, 4/3); g = (2, 1).
    # G = (14/9, 11/9), Gbar = 25/18. targets = Gbar * r**1.5 = (0.7560, 2.1383) -> signs (+, -);
    # grad = (+2, -1) (g, not G: using G would give 0.6328 / 1.3672), w = (7/9 - 0.2, 11/9 + 0.1) = (0.5778, 1.3222),
    # sum 1.9 -> x 2/1.9.
    out = gn.update({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}, active=["a", "b"])
    new_a, new_b = 7.0 / 9.0 - 0.2, 11.0 / 9.0 + 0.1
    assert close(out["a"], new_a * 2.0 / 1.9) and close(out["b"], new_b * 2.0 / 1.9), out
    assert math.isclose(out["a"], 0.6081871345029238, rel_tol=1e-12)
    assert not math.isclose(out["a"], 0.6327683615819208, rel_tol=1e-6), "step used G_i where g_i belongs"
    assert gn.initial == {"a": 2.0, "b": 2.0}, "L_i(0) must be the first active loss, not refreshed"

    # alpha decides the sign. Same first call (L0 = 2, 2), then L = (1, 2): r = (2/3, 4/3).
    #  A) g = (3, 7): G/Gbar = (0.6, 1.4). alpha 1.5: targets (2.72, 7.70) -> (+, -) -> w = (0.7, 1.7) -> (7/12, 17/12).
    #     alpha 1.0: targets (3.33, 6.67) -> (-, +) -> w = (1.3, 0.3) -> (1.625, 0.375).
    #  B) g = (1, 4): G = (1, 4), Gbar = 2.5. alpha 1.5: targets (1.36, 3.85) -> (-, +) -> w = (1.1, 0.6)
    #     -> (22/17, 12/17). alpha 3.0: targets (0.74, 5.93) -> (+, -) -> w = (0.9, 1.4) -> (18/23, 28/23).
    #     (alpha 3.0 gives the same signs as 1.5 in A, alpha 1.0 the same as 1.5 in B: each case separates one.)
    def two_steps(norms, alpha=None):
        kw = {} if alpha is None else {"alpha": alpha}
        g = _gn(["a", "b"], lr=0.1, **kw)
        g.update({"a": 2.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, active=["a", "b"])  # equal norms: no move, sets L(0)
        assert g.weights == {"a": 1.0, "b": 1.0}, g.weights
        return g.update({"a": 1.0, "b": 2.0}, norms, active=["a", "b"])

    a_default = two_steps({"a": 3.0, "b": 7.0})
    assert close(a_default["a"], 7.0 / 12.0) and close(a_default["b"], 17.0 / 12.0), a_default
    a_lin = two_steps({"a": 3.0, "b": 7.0}, alpha=1.0)
    assert close(a_lin["a"], 1.625) and close(a_lin["b"], 0.375), a_lin
    b_default = two_steps({"a": 1.0, "b": 4.0})
    assert close(b_default["a"], 22.0 / 17.0) and close(b_default["b"], 12.0 / 17.0), b_default
    b_cubic = two_steps({"a": 1.0, "b": 4.0}, alpha=3.0)
    assert close(b_cubic["a"], 18.0 / 23.0) and close(b_cubic["b"], 28.0 / 23.0), b_cubic
    print("gradnorm one step: 7/9, 11/9 then 0.6082, 1.3918; alpha 1.0 / 1.5 / 3.0 give different signs, hand-computed OK")


def check_project_weights_clamp_exact() -> None:
    from python.train.trainer2_balance import project_weights

    # a is clamped up to 0.1; b and c share the rest in proportion 1:3: 4x = 2.9, x = 0.725.
    w = project_weights({"a": 1e-6, "b": 1.0, "c": 3.0}, 3.0)
    assert math.isclose(w["a"], 0.1, rel_tol=1e-9) and math.isclose(w["b"], 0.725, rel_tol=1e-9), w
    assert math.isclose(w["c"], 2.175, rel_tol=1e-9), w
    # c is clamped high at 10; a and b share 12 - 10 = 2 in proportion 1:2 -> 2/3 and 4/3.
    w = project_weights({"a": 1.0, "b": 2.0, "c": 1e6}, 12.0)
    assert math.isclose(w["c"], 10.0, rel_tol=1e-9) and math.isclose(w["a"], 2.0 / 3.0, rel_tol=1e-9), w
    assert math.isclose(w["b"], 4.0 / 3.0, rel_tol=1e-9), w
    print("project_weights: clamped and proportional shares hand-computed (0.1 / 0.725 / 2.175 and 2/3 / 4/3 / 10) OK")


class _Boom:
    """A loss that must never be touched (an inactive head computes no loss)."""

    def __mul__(self, _other):
        raise AssertionError("inactive head's loss was used")

    __rmul__ = __mul__


def check_combine_losses_ramp_and_inactive() -> None:
    from python.train.trainer2_balance import GradNorm, HeadRamp, combine_losses

    gn = GradNorm(["a", "b", "late"], fixed={"kd": 0.5})
    gn.weights.update({"a": 1.5, "b": 0.5, "late": 1.0})
    ramp = HeadRamp({"a": 0, "b": 100, "late": 1000})
    # step 600: a ramp 600/2000 = 0.3, b ramp 500/2000 = 0.25, late inactive (its loss is never read).
    total = combine_losses({"a": 2.0, "b": 4.0, "late": _Boom()}, gn, ramp, step=600, fixed_losses={"kd": 3.0})
    want = 0.3 * 1.5 * 2.0 + 0.25 * 0.5 * 4.0 + 0.5 * 3.0
    assert math.isclose(total, want, rel_tol=1e-12), (total, want)
    assert math.isclose(want, 0.9 + 0.5 + 1.5, rel_tol=1e-12)
    # Before any head enters nothing but the fixed term counts.
    assert combine_losses({"a": _Boom()}, gn, HeadRamp({"a": 5}), step=4, fixed_losses=None) == 0.0
    print("combine_losses: ramp x weight x loss summed over active heads only, fixed term added OK")


def check_gradnorm_lone_head_renormalizes_to_active_count() -> None:
    # Spec (docs/specs/trainer_2.md, GradNorm): renormalized so sum(w_i) == n_active.
    # With one active head that is 1.0, whatever it earned while more heads were active.
    for earned in (1.4, 0.3, 10.0, 0.1):
        gn = _gn(["a", "b"], lr=0.1)
        gn.weights["a"] = earned
        gn.weights["b"] = 0.7  # inactive: must not be touched
        out = gn.update({"a": 1.0}, {"a": 9.0}, active=["a"])
        assert out["a"] == 1.0 and gn.weights["a"] == 1.0, (earned, out)
        assert gn.weights["b"] == 0.7 and out["b"] == 0.7, "an inactive head's weight changed"
        assert gn.initial == {"a": 1.0}, "the lone head's first loss must still be recorded as L(0)"
    # The weight is 1.0 whatever the grad norm: no gradient step is taken with one head.
    for g in (0.0, 1e-6, 1e6):
        gn = _gn(["a"], lr=0.5)
        assert gn.update({"a": 3.0}, {"a": g}, active=["a"]) == {"a": 1.0}
    # No active head: nothing changes.
    gn = _gn(["a", "b"], lr=0.1)
    gn.weights["a"] = 1.4
    assert gn.update({}, {}, active=[]) == {"a": 1.4, "b": 1.0}
    # Two active heads still sum to 2 (the lone-head rule is the n = 1 case of the same projection).
    gn = _gn(["a", "b"], lr=0.1)
    out = gn.update({"a": 1.0, "b": 1.0}, {"a": 4.0, "b": 1.0}, active=["a", "b"])
    assert math.isclose(out["a"] + out["b"], 2.0, abs_tol=1e-9), out
    print("gradnorm: a lone active head is projected to 1.0 (sum == n_active), inactive weights and L(0) kept OK")


def check_grad_norm_record_weights_after_update() -> None:
    from python.train.trainer2_balance import HeadRamp, grad_norm_record

    gn = _gn(["a", "b", "c"], lr=0.1, fixed={"kd": 0.25})
    gn.update({"a": 1.0, "b": 1.0}, {"a": 4.0, "b": 1.0}, active=["a", "b"])
    gn.update({"a": 0.5, "b": 1.5}, {"a": 4.0, "b": 1.0}, active=["a", "b"])
    assert gn.weights["a"] != 1.0 and gn.weights["b"] != 1.0, gn.weights  # the update moved them
    ramp = HeadRamp({"a": 0, "b": 0, "c": 100})
    rec = grad_norm_record(10, {"a": 4.0, "b": 1.0}, gn, ramp)
    assert rec["weight"] == {"a": gn.weights["a"], "b": gn.weights["b"]}, (rec["weight"], gn.weights)
    assert rec["weight"]["a"] != 1.0 and rec["weight"]["b"] != 1.0
    assert rec["ramp"] == {"a": 10 / 2000, "b": 10 / 2000} and rec["grad_norm"] == {"a": 4.0, "b": 1.0}
    print(f"grad_norm_record: weights after two updates {rec['weight']} == GradNorm.weights OK")


def check_combine_refuses_balanced_head_as_fixed_term() -> None:
    try:
        import torch
    except ImportError:
        skip_check("check_combine_refuses_balanced_head_as_fixed_term", "torch not installed")
        return
    from python.train.trainer2_balance import HeadRamp, combine_losses

    gn = _gn(["a", "b"], lr=0.1, fixed={"kd": 0.5})
    ramp = HeadRamp({"a": 0, "b": 0})
    losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    for balanced in ("a", "b"):
        try:
            combine_losses(losses, gn, ramp, step=2000, fixed_losses={balanced: torch.tensor(3.0)})
        except ValueError as exc:
            assert balanced in str(exc) and "GradNorm-balanced" in str(exc), exc
        else:
            raise AssertionError(f"balanced head {balanced!r} offered as a fixed term was accepted (double-counted)")
    ok = combine_losses(losses, gn, ramp, step=2000, fixed_losses={"kd": torch.tensor(4.0)})
    assert math.isclose(float(ok.detach()), 1.0 + 2.0 + 0.5 * 4.0), float(ok.detach())
    print("combine_losses: a GradNorm-balanced head offered as a fixed term is refused; a real fixed term still adds OK")


def main() -> int:
    checks = (
        check_head_ramp_linear,
        check_gradnorm_weights_sum_and_clamp,
        check_gradnorm_one_step_hand_computed,
        check_project_weights_clamp_exact,
        check_combine_losses_ramp_and_inactive,
        check_gradnorm_lone_head_renormalizes_to_active_count,
        check_grad_norm_record_weights_after_update,
        check_combine_refuses_balanced_head_as_fixed_term,
        check_per_task_grad_norm_logged,
    )
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
