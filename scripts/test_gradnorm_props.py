#!/usr/bin/env python3
"""Property + fuzz checks for Trainer 2.0 GradNorm and the warmup+cosine LR schedule.

Modules: python/train/trainer2_balance.py (GradNorm, project_weights, HeadRamp,
combine_losses, per_task_grad_norms), python/train/lr_schedule.py (WarmupCosine,
apply_lr). Invariants are written down in docs/specs/gradnorm_props_notes.md (G1..G7, L1..L7).

G1  update() output: every weight finite, in [W_MIN, W_MAX], strictly positive.
G2  Over the active heads the weights sum to n_active (spec: "renormalized so
    sum(w_i) = n_active"), to 1e-9 per head, for any finite non-negative losses and grad
    norms (zero, subnormal, huge), any finite alpha >= 0 and lr >= 0, over many steps.
G3  Inactive heads (and fixed/distill terms) are left untouched, whatever their garbage
    loss/grad inputs are; a zero-grad head, a zero-loss head or a disabled head never crashes
    and never poisons the others.
G4  Bad active inputs (NaN/inf/negative) raise ValueError and leave state unchanged.
G5  Constructor refuses a non-finite or negative alpha / lr (they would poison or crash update()).
G6  HeadRamp.factor is 0 before entry, in [0, 1], non-decreasing, exactly 1.0 from s0 + R on;
    combine_losses ignores inactive heads even when their loss is NaN.
G7  per_task_grad_norms: a head with no path to the shared weight, a constant loss, or a zero
    gradient reports 0.0, never raises, and shared.grad is untouched (torch; skips visibly).
L1  WarmupCosine.lr_at is finite and 0 <= lr <= peak for every step, incl. negative and
    past-the-end steps, for any finite positive peak.
L2  Non-decreasing through warmup (reaches peak at the last warmup step), non-increasing after.
L3  After warmup lr >= floor_frac * peak; lr_at(total_steps) == floor exactly when there is a
    cosine phase; past the end it holds that value.
L4  Edge configs warmup=0, warmup=total, total=1, warmup>total, warmup<0 never crash/NaN;
    warmup=0 starts at peak; total==warmup holds peak after warmup.
L5  state_dict round-trips to an identical schedule (JSON-safe).
L6  The constructor raises ValueError (nothing else) for an invalid peak / total / floor_frac.
L7  apply_lr writes schedule(step) * lr_mult onto every dict group and returns the rate.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gvd_test_env import skip_check  # noqa: E402

try:
    from hypothesis import HealthCheck, example, given, settings
    from hypothesis import strategies as st
except ImportError:  # dev-only dep (requirements-dev.txt); CI fails instead of skipping
    given = None

from python.train.lr_schedule import WarmupCosine, apply_lr  # noqa: E402
from python.train.trainer2_balance import (  # noqa: E402
    W_MAX,
    W_MIN,
    GradNorm,
    HeadRamp,
    combine_losses,
    grad_norm_record,
    project_weights,
)

F_MAX = sys.float_info.max
SUM_TOL = 1e-9  # per head


# ----------------------------------------------------------------- helpers

def _check_weights(w: dict[str, float], active: list[str], before: dict[str, float]) -> None:
    for name, v in w.items():
        assert math.isfinite(v), (name, v, w)
        assert W_MIN <= v <= W_MAX or name not in active, (name, v)
        assert v > 0.0
    for name in w:
        if name not in active:
            assert w[name] == before[name], (name, w[name], before[name])  # untouched
    if active:
        s = sum(w[n] for n in active)
        assert abs(s - len(active)) <= SUM_TOL * len(active), (s, len(active), w)


def _run_steps(names, alpha, lr, steps) -> None:
    """steps: list of (active_subset, {name: (loss, grad)}) over the same GradNorm."""
    gn = GradNorm(names, alpha=alpha, lr=lr)
    for active, vals in steps:
        before = dict(gn.weights)
        losses = {n: vals[n][0] for n in names}
        grads = {n: vals[n][1] for n in names}
        out = gn.update(losses, grads, active=active)
        assert out == gn.weights and out is not gn.weights
        _check_weights(out, active, before)


# ---------------------------------------------------- deterministic pins
# Each pin is the shrunk counterexample of a bug found by the fuzz (see
# docs/specs/gradnorm_props_notes.md "Counterexamples"); they run without hypothesis.

def check_pin_alpha_overflow() -> None:
    """alpha=1024, losses (0, 1): r = 2.0, 2.0 ** 1024 overflowed (OverflowError)."""
    gn = GradNorm(["a", "b"], alpha=1024.0)
    out = gn.update({"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, active=["a", "b"])
    _check_weights(out, ["a", "b"], {"a": 1.0, "b": 1.0})


def check_pin_huge_grad_norm_keeps_sum() -> None:
    """grad norms ~6.4e62 (finite, accepted): the bisection bracket [0, 1] could not resolve
    the scale and returned (10.0, 0.1), sum 10.1 != n_active 2."""
    gn = GradNorm(["a", "b"], alpha=0.0)
    out = gn.update({"a": 0.0, "b": 0.0}, {"a": 6.42775217703596e62, "b": 6.427752177035961e62}, active=["a", "b"])
    _check_weights(out, ["a", "b"], {"a": 1.0, "b": 1.0})
    w = project_weights({"a": 2.5e306, "b": 1e-12}, 2.0)
    assert abs(sum(w.values()) - 2.0) <= 1e-9 and all(W_MIN <= v <= W_MAX for v in w.values()), w


def check_pin_ctor_refuses_poison_hyperparams() -> None:
    """lr=nan / lr=inf broke the sum (0.1+0.1, 10+0.1); alpha<0 raised ZeroDivisionError on a
    zero-loss head; alpha=nan/inf are meaningless. All must be refused up front."""
    for kw in (
        {"lr": float("nan")}, {"lr": float("inf")}, {"lr": -0.025},
        {"alpha": float("nan")}, {"alpha": float("inf")}, {"alpha": -1.0},
    ):
        try:
            GradNorm(["a", "b"], **kw)
        except ValueError:
            continue
        raise AssertionError(f"GradNorm({kw}) was accepted")
    GradNorm(["a", "b"], alpha=0.0, lr=0.0)  # boundary values stay legal
    gn = GradNorm(["a", "b"], alpha=0.0, lr=0.0)
    out = gn.update({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, active=["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}  # lr=0: no step


def check_pin_warmup_last_step_not_above_peak() -> None:
    """peak=0.7351220134688766, total=118 (warmup = whole stage): peak * 118 / 118 was
    0.7351220134688767, one ulp above peak."""
    w = WarmupCosine(0.7351220134688766, 118)
    assert w.warmup_steps == 118
    assert w.lr_at(117) == w.peak, w.lr_at(117)
    assert max(w.lr_at(s) for s in range(0, 130)) <= w.peak


def check_pin_peak_overflow_in_warmup() -> None:
    """peak=8.98846567431158e+307, total=2: peak * 2 overflowed to inf."""
    w = WarmupCosine(8.98846567431158e307, 2)
    for s in range(-1, 6):
        v = w.lr_at(s)
        assert math.isfinite(v) and 0.0 <= v <= w.peak, (s, v)


def check_pin_constant_loss_has_zero_grad_norm() -> None:
    try:
        import torch
    except ImportError:
        skip_check("pin_constant_loss_has_zero_grad_norm", "torch not installed")
        return
    from python.train.trainer2_balance import per_task_grad_norms

    shared = torch.nn.Parameter(torch.ones(3))
    out = per_task_grad_norms({"const": torch.tensor(0.0), "live": (shared * 2).sum()}, shared)
    assert out["const"] == 0.0 and abs(out["live"] - 2.0 * math.sqrt(3.0)) < 1e-6, out
    assert shared.grad is None


def check_documented_edges() -> None:
    """Edge behavior pinned (not changed by this PR)."""
    # lone / empty active set
    gn = GradNorm(["a", "b"])
    assert gn.update({}, {}, active=[]) == {"a": 1.0, "b": 1.0}
    assert gn.update({"a": 3.0}, {"a": 0.0}, active=["a"]) == {"a": 1.0, "b": 1.0}
    # total == warmup: peak at the last warmup step and held after (no cosine phase)
    w = WarmupCosine(0.01, 10, warmup_steps=10)
    assert w.lr_at(9) == 0.01 and w.lr_at(10) == 0.01 and w.lr_at(10**9) == 0.01
    # warmup = 0: starts at peak, floor at total
    w = WarmupCosine(0.01, 10, warmup_steps=0)
    assert w.lr_at(0) == 0.01 and w.lr_at(10) == 0.01 * w.floor_frac and w.lr_at(10**9) == w.lr_at(10)
    # total = 1
    w = WarmupCosine(0.01, 1)
    assert [w.lr_at(s) for s in (-3, 0, 1, 9)] == [0.01] * 4
    # warmup > total clamps to total; negative warmup -> 0
    assert WarmupCosine(1.0, 5, warmup_steps=99).warmup_steps == 5
    assert WarmupCosine(1.0, 5, warmup_steps=-4).warmup_steps == 0


# ----------------------------------------------------------- hypothesis

def _build_props():
    if given is None:
        return []
    cfg = settings(max_examples=150, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    cfg_torch = settings(max_examples=40, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))

    nonneg = st.floats(min_value=0.0, max_value=F_MAX, allow_nan=False, allow_infinity=False)
    wild = st.one_of(
        st.sampled_from([0.0, 5e-324, 1e-300, 1e-12, 1.0, 1e12, 1e60, 1e300, F_MAX]),
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        nonneg,
    )
    alphas = st.one_of(st.sampled_from([0.0, 1.0, 1.5, 3.0, 1023.0, 1024.0, 1e6, F_MAX]), st.floats(0.0, 64.0), nonneg)
    lrs = st.one_of(st.sampled_from([0.0, 0.025, 1.0, 1e6, 1e300, F_MAX]), st.floats(0.0, 10.0), nonneg)
    names_st = st.integers(1, 8).map(lambda n: [f"h{i}" for i in range(n)])

    @st.composite
    def scenario(draw, max_steps=12):
        names = draw(names_st)
        steps = []
        for _ in range(draw(st.integers(1, max_steps))):
            active = draw(st.lists(st.sampled_from(names), unique=True, max_size=len(names)))
            vals = {n: (draw(wild), draw(wild)) for n in names}
            steps.append((active, vals))
        return names, draw(alphas), draw(lrs), steps

    @cfg
    @given(scenario())
    @example(([ "a", "b"], 1024.0, 0.025, [(["a", "b"], {"a": (0.0, 0.0), "b": (1.0, 0.0)})]))
    @example((["a", "b"], 0.0, 0.025, [(["a", "b"], {"a": (0.0, 6.42775217703596e62), "b": (0.0, 6.427752177035961e62)})]))
    def prop_gradnorm_finite_clamped_renormalized(sc):
        names, alpha, lr, steps = sc
        _run_steps(names, alpha, lr, steps)

    @cfg
    @given(st.lists(st.tuples(wild, wild), min_size=2, max_size=4), st.integers(200, 600), st.integers(0, 2**31 - 1))
    def prop_gradnorm_long_horizon(base, n_steps, seed):
        import random

        rng = random.Random(seed)
        names = [f"h{i}" for i in range(len(base))]
        gn = GradNorm(names)
        for t in range(n_steps):
            active = [n for n in names if rng.random() < 0.9] or names[:1]
            losses = {n: min(F_MAX, base[i][0] * rng.choice([0.0, 0.5, 1.0, 1e-6, 1e6, 3.0])) for i, n in enumerate(names)}
            grads = {n: min(F_MAX, base[i][1] * rng.choice([0.0, 0.5, 1.0, 1e-6, 1e6, 3.0])) for i, n in enumerate(names)}
            before = dict(gn.weights)
            _check_weights(gn.update(losses, grads, active=active), active, before)

    @cfg
    @given(scenario(max_steps=6), st.lists(st.sampled_from([float("nan"), float("inf"), -1.0, -1e-300, 1e30]), min_size=8, max_size=8))
    def prop_gradnorm_inactive_garbage_and_zero_heads(sc, garbage):
        names, alpha, lr, steps = sc
        gn = GradNorm(names, alpha=alpha, lr=lr, fixed={"distill": 0.3})
        for active, vals in steps:
            before = dict(gn.weights)
            losses = {n: vals[n][0] for n in names}
            grads = {n: vals[n][1] for n in names}
            for i, n in enumerate(names):
                if n not in active:
                    losses[n], grads[n] = garbage[i % 8], garbage[(i + 3) % 8]  # disabled head: junk
                elif i % 2 == 0:
                    grads[n] = 0.0  # zero-grad head
            out = gn.update(losses, grads, active=active)
            _check_weights(out, active, before)
            assert gn.fixed == {"distill": 0.3} and gn.weight_of("distill") == 0.3

    bad_vals = st.sampled_from([float("nan"), float("inf"), float("-inf"), -1.0, -1e-300, -F_MAX])

    @cfg
    @given(scenario(max_steps=3), bad_vals, st.booleans(), st.data())
    def prop_gradnorm_bad_active_input_raises_and_keeps_state(sc, bad, in_loss, data):
        names, alpha, lr, steps = sc
        gn = GradNorm(names, alpha=alpha, lr=lr)
        for active, vals in steps:
            gn.update({n: vals[n][0] for n in names}, {n: vals[n][1] for n in names}, active=active)
        w0, init0 = dict(gn.weights), dict(gn.initial)
        active = data.draw(st.lists(st.sampled_from(names), unique=True, min_size=1, max_size=len(names)))
        victim = data.draw(st.sampled_from(active))
        losses = {n: 1.0 for n in names}
        grads = {n: 1.0 for n in names}
        (losses if in_loss else grads)[victim] = bad
        try:
            gn.update(losses, grads, active=active)
        except ValueError:
            pass
        else:
            raise AssertionError("bad input accepted")
        assert gn.weights == w0 and gn.initial == init0

    @cfg
    @given(
        st.lists(nonneg, min_size=1, max_size=12),
        st.floats(min_value=0.0, max_value=1.0),
        st.data(),
    )
    def prop_project_weights_exact_sum_in_clamp(raw, frac, data):
        n = len(raw)
        total = n * W_MIN + frac * n * (W_MAX - W_MIN)
        out = project_weights({f"k{i}": v for i, v in enumerate(raw)}, total)
        assert list(out) == [f"k{i}" for i in range(n)]
        assert all(math.isfinite(v) and W_MIN <= v <= W_MAX for v in out.values()), out
        assert abs(sum(out.values()) - total) <= SUM_TOL * n, (sum(out.values()), total)
        # order of weights is kept (monotone in the raw value)
        pairs = sorted(zip(raw, out.values()))
        assert all(b2 >= b1 - 1e-12 for (_, b1), (_, b2) in zip(pairs, pairs[1:])), pairs

    entries = st.dictionaries(st.sampled_from(["a", "b", "c", "d"]), st.integers(-10**6, 10**6), min_size=1)

    @cfg
    @given(entries, st.integers(1, 5000), st.lists(st.integers(-2 * 10**6, 2 * 10**6), min_size=1, max_size=40))
    def prop_head_ramp_bounds_monotone_and_combine_ignores_inactive(entry, ramp_steps, steps):
        ramp = HeadRamp(entry, ramp_steps)
        for name, s0 in entry.items():
            prev = 0.0
            for s in sorted(steps):
                f = ramp.factor(name, s)
                assert math.isfinite(f) and 0.0 <= f <= 1.0, (name, s, f)
                assert (f == 0.0) if s < s0 else True
                assert f >= prev, (name, s, f, prev)
                if s >= s0 + ramp_steps:
                    assert f == 1.0
                assert ramp.active(name, s) == (s >= s0)
                prev = f
        for s in steps[:5]:
            gn = GradNorm(list(entry))
            losses = {n: (float("nan") if not ramp.active(n, s) else 2.0) for n in entry}
            total = combine_losses(losses, gn, ramp, step=s)
            want = sum(ramp.factor(n, s) * 1.0 * 2.0 for n in entry if ramp.active(n, s))
            assert total == want or (math.isfinite(want) and abs(total - want) <= 1e-12 * max(1.0, abs(want))), (total, want)
            assert not math.isnan(total)
            rec = grad_norm_record(s, {n: 0.0 for n in entry}, gn, ramp)
            assert json.loads(json.dumps(rec, allow_nan=False)) == rec

    coef = st.one_of(st.sampled_from([0.0, -0.0, 1.0, -3.5, 1e-30, 1e30]), st.floats(-1e6, 1e6, allow_nan=False))

    @cfg_torch
    @given(st.lists(coef, min_size=1, max_size=5), st.integers(1, 6))
    def prop_per_task_grad_norms_zero_unused_const_heads(coefs, dim):
        try:
            import torch
        except ImportError:
            skip_check("prop_per_task_grad_norms_zero_unused_const_heads", "torch not installed")
            return
        from python.train.trainer2_balance import per_task_grad_norms

        shared = torch.nn.Parameter(torch.ones(dim, dtype=torch.float64))
        other = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        losses = {}
        for i, c in enumerate(coefs):
            losses[f"live{i}"] = (shared * c).sum()
        losses["unused"] = (other * 2.0).sum()
        losses["const"] = torch.tensor(1.5, dtype=torch.float64)
        losses["zeroed"] = (shared * 0.0).sum()
        out = per_task_grad_norms(losses, shared)
        assert shared.grad is None and other.grad is None
        assert out["unused"] == 0.0 and out["const"] == 0.0 and out["zeroed"] == 0.0, out
        for i, c in enumerate(coefs):
            assert math.isfinite(out[f"live{i}"])
            assert math.isclose(out[f"live{i}"], abs(c) * math.sqrt(dim), rel_tol=1e-9, abs_tol=0.0), (c, out)
        # a zero-grad head does not poison the balancer
        gn = GradNorm(list(out))
        gn.update({n: 1.0 for n in out}, out, active=list(out))
        assert abs(sum(gn.weights.values()) - len(out)) <= SUM_TOL * len(out)

    # -------------------------------------------------------------- LR
    peaks = st.one_of(
        st.sampled_from([5e-324, 1e-300, 1e-6, 3e-4, 1.0, 1e30, 8.98846567431158e307, F_MAX]),
        st.floats(min_value=5e-324, max_value=F_MAX, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-8, max_value=10.0),
    )
    totals = st.one_of(st.integers(1, 12), st.integers(1, 10**6), st.sampled_from([118, 499, 500, 501, 10**9]))
    floors = st.one_of(st.sampled_from([0.0, 0.05, 1.0]), st.floats(0.0, 1.0))
    warms = st.one_of(st.none(), st.integers(-5, 12), st.integers(0, 10**6))

    @st.composite
    def schedule(draw, max_peak=F_MAX):
        peak, total, ff, warm = draw(peaks), draw(totals), draw(floors), draw(warms)
        return WarmupCosine(min(peak, max_peak), total, warmup_steps=warm, floor_frac=ff)

    @cfg
    @given(schedule(), st.integers(-10, 3 * 10**9), st.integers(1, 120))
    @example(WarmupCosine(0.7351220134688766, 118), 100, 50)
    @example(WarmupCosine(8.98846567431158e307, 2), 0, 4)
    def prop_lr_bounds_and_monotone_within_phase(sch, s0, k):
        peak, floor = sch.peak, sch.floor_frac * sch.peak
        prev = None
        for s in range(s0, s0 + k):
            v = sch.lr_at(s)
            assert math.isfinite(v) and 0.0 <= v <= peak, (s, v, peak)
            eff = max(0, s)
            if eff >= sch.warmup_steps and sch.total_steps > sch.warmup_steps:
                assert v >= floor, (s, v, floor)
            if prev is not None and s >= 1:
                if eff < sch.warmup_steps:
                    assert v >= prev, ("warmup decreased", s, prev, v)
                elif eff - 1 >= sch.warmup_steps:
                    assert v <= prev, ("cosine increased", s, prev, v)
                else:  # first cosine step follows the last warmup step
                    assert v <= prev or sch.total_steps == sch.warmup_steps, ("jump up at phase change", s, prev, v)
            prev = v

    @cfg
    @given(schedule())
    def prop_lr_phase_boundaries_and_tail(sch):
        peak, warm, total = sch.peak, sch.warmup_steps, sch.total_steps
        if warm > 0:
            assert sch.lr_at(warm - 1) == peak  # warmup ends exactly at peak
        else:
            assert sch.lr_at(0) == peak
        assert sch.lr_at(-7) == sch.lr_at(0)
        if total > warm:
            end = sch.floor_frac * peak
            assert sch.lr_at(total) == end, (sch.lr_at(total), end)
            assert sch.lr_at(total + 12345) == end and sch.lr_at(10**15) == end
        else:
            assert sch.lr_at(total) == peak and sch.lr_at(10**15) == peak  # whole stage was warmup
        # whole-schedule scan for short stages: no NaN, bounded, unimodal
        if total <= 3000:
            vals = [sch.lr_at(s) for s in range(0, total + 3)]
            assert all(math.isfinite(v) and 0.0 <= v <= peak for v in vals)
            top = vals.index(max(vals))
            assert all(a <= b for a, b in zip(vals[: top + 1], vals[1 : top + 1]))
            assert all(a >= b for a, b in zip(vals[top:], vals[top + 1 :]))

    @cfg
    @given(schedule(), st.lists(st.integers(-5, 3 * 10**9), min_size=1, max_size=30))
    def prop_lr_state_dict_round_trip(sch, steps):
        state = json.loads(json.dumps(sch.state_dict(), allow_nan=False))
        again = WarmupCosine.from_state_dict(state)
        assert again.state_dict() == sch.state_dict()
        assert [again.lr_at(s) for s in steps] == [sch.lr_at(s) for s in steps]

    junk_floats = st.one_of(
        st.sampled_from([float("nan"), float("inf"), float("-inf"), 0.0, -0.0, -1.0, 1.0, 1.0000001, -1e-300]),
        st.floats(allow_nan=True, allow_infinity=True),
    )

    @cfg
    @given(junk_floats, st.integers(-3, 5), junk_floats, st.one_of(st.none(), st.integers(-3, 9)))
    def prop_lr_constructor_validates(peak, total, ff, warm):
        ok = peak > 0.0 and math.isfinite(peak) and total >= 1 and 0.0 <= ff <= 1.0
        try:
            sch = WarmupCosine(peak, total, warmup_steps=warm, floor_frac=ff)
        except ValueError:
            assert not ok, (peak, total, ff)
            return
        assert ok, ("accepted invalid", peak, total, ff)
        for s in range(-1, total + 3):
            v = sch.lr_at(s)
            assert math.isfinite(v) and 0.0 <= v <= sch.peak

    class _Opt:
        def __init__(self, mults):
            self.param_groups = [{"lr": 123.0, "lr_mult": m} for m in mults] + [{"lr": 9.0}]

    @cfg
    @given(schedule(max_peak=1e30), st.integers(-3, 10**6), st.lists(st.floats(0.0, 100.0), max_size=5))
    def prop_apply_lr_writes_rate_times_mult(sch, step, mults):
        opt = _Opt(mults)
        rate = apply_lr(opt, sch, step)
        assert rate == sch.lr_at(step)
        for g, m in zip(opt.param_groups, mults + [1.0]):
            assert g["lr"] == rate * m and math.isfinite(g["lr"])
        assert apply_lr(object(), sch, step) == rate  # no param_groups: nothing to write, no crash

    return [
        prop_gradnorm_finite_clamped_renormalized,
        prop_gradnorm_long_horizon,
        prop_gradnorm_inactive_garbage_and_zero_heads,
        prop_gradnorm_bad_active_input_raises_and_keeps_state,
        prop_project_weights_exact_sum_in_clamp,
        prop_head_ramp_bounds_monotone_and_combine_ignores_inactive,
        prop_per_task_grad_norms_zero_unused_const_heads,
        prop_lr_bounds_and_monotone_within_phase,
        prop_lr_phase_boundaries_and_tail,
        prop_lr_state_dict_round_trip,
        prop_lr_constructor_validates,
        prop_apply_lr_writes_rate_times_mult,
    ]


PROPS = _build_props()


def _ci_requires_hypothesis() -> bool:
    """True when running under CI (GITHUB_ACTIONS or CI=true/1/yes)."""
    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
    return _on("CI") or _on("GITHUB_ACTIONS")


def _run_hypothesis_props_or_skip() -> None:
    """Run property tests; fail hard under CI if hypothesis is missing, else skip locally."""
    if given is not None:
        for prop in PROPS:
            prop()
            print(f"{prop.__name__}: holds OK")
        return
    if _ci_requires_hypothesis():
        raise SystemExit(
            "FAIL hypothesis_properties: hypothesis is not installed under CI "
            f"(CI={os.environ.get('CI')!r}, GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS')!r}). "
            "CI must install requirements-dev.txt (hypothesis==6.170.17)."
        )
    skip_check("hypothesis_properties", "hypothesis not installed (dev-only dep; CI installs it)")


def main() -> None:
    check_documented_edges()
    check_pin_alpha_overflow()
    check_pin_huge_grad_norm_keeps_sum()
    check_pin_ctor_refuses_poison_hyperparams()
    check_pin_warmup_last_step_not_above_peak()
    check_pin_peak_overflow_in_warmup()
    check_pin_constant_loss_has_zero_grad_norm()
    _run_hypothesis_props_or_skip()
    print("test_gradnorm_props: OK")


if __name__ == "__main__":
    main()


# Pytest entrypoints (for mutmut 3.8.1).
def test_documented_edges() -> None:
    check_documented_edges()


def test_pins() -> None:
    check_pin_alpha_overflow()
    check_pin_huge_grad_norm_keeps_sum()
    check_pin_ctor_refuses_poison_hyperparams()
    check_pin_warmup_last_step_not_above_peak()
    check_pin_peak_overflow_in_warmup()
    check_pin_constant_loss_has_zero_grad_norm()


def test_hypothesis_props() -> None:
    _run_hypothesis_props_or_skip()
