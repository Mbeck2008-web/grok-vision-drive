# GradNorm + warmup/cosine: property and fuzz invariants (`scripts/test_gradnorm_props.py`)

Assignment: GVD swarm-4 `gvd/gradnorm-props`, off main `a56f514`. Test and hardening only.
Modules under test: `python/train/trainer2_balance.py` (GradNorm, `project_weights`,
`HeadRamp`, `combine_losses`, `grad_norm_record`, `per_task_grad_norms`) and
`python/train/lr_schedule.py` (`WarmupCosine`, `apply_lr`). The spec text is the
"Loss, gradients, LR" section of `docs/specs/trainer_2.md`; it is not edited here.
Neither module is on a live path, and nothing about heartbeat, AEB or engage is touched.

How it runs: a standalone script like `test_rv_splat_sort.py`. The pins run without
hypothesis. The properties use hypothesis (`requirements-dev.txt`, `hypothesis==6.170.17`),
`derandomize=True` (the same examples every run, so CI is not flaky), and under `CI=true` /
`GITHUB_ACTIONS` a missing hypothesis fails the script instead of skipping. The torch part
(G7) skips visibly without torch. `run_offline_suite.py` globs `scripts/test_*.py`, so no
registration is needed. About 25 s of CPU for the whole file.

## Invariants

| id | invariant | prop / pin |
| --- | --- | --- |
| G1 | After `update`, every weight is finite, strictly positive, and in `[W_MIN, W_MAX]` = `[0.1, 10]` (spec clamp). | `prop_gradnorm_finite_clamped_renormalized`, `prop_gradnorm_long_horizon` (200-600 steps) |
| G2 | Over the active heads the weights sum to `n_active` (spec: "renormalized so sum(w_i) = n_active"), to 1e-9 per head. Inputs are any finite loss and grad norm >= 0 (0, subnormal, 1e-300, 1e300, float max), any alpha in `[0, F_MAX]`, any lr in `[0, F_MAX]`, any active subset (empty, lone, all), up to 8 heads. | same two props, `prop_project_weights_exact_sum_in_clamp` |
| G3 | Inactive heads and fixed (distill) terms are untouched (exactly equal) whatever junk (NaN, inf, negative) is in their loss/grad entries. A zero-grad head and a zero-loss head never crash and the other heads stay finite and renormalized. | `prop_gradnorm_inactive_garbage_and_zero_heads` |
| G4 | A NaN, inf or negative loss or grad norm on an active head raises `ValueError`; weights and the stored `L_i(0)` are unchanged. | `prop_gradnorm_bad_active_input_raises_and_keeps_state` |
| G5 | The constructor refuses non-finite or negative `alpha` and `lr` (they poison or crash `update`). 0 stays legal (lr=0 is no step). | `check_pin_ctor_refuses_poison_hyperparams` |
| G6 | `HeadRamp.factor` is 0 before the entry step, in `[0, 1]`, non-decreasing, exactly 1.0 from `s0 + R`; `combine_losses` ignores inactive heads even with NaN loss; `grad_norm_record` is strict-JSON. | `prop_head_ramp_bounds_monotone_and_combine_ignores_inactive` |
| G7 | `per_task_grad_norms`: no path to shared, a constant (no-graph) loss and a zero gradient all report `0.0`; `shared.grad` is untouched; the result feeds `update` without poisoning it. | `prop_per_task_grad_norms_zero_unused_const_heads`, `check_pin_constant_loss_has_zero_grad_norm` |
| L1 | `lr_at` is finite and `0 <= lr <= peak` for every step (negative, past the end too), for peak in `[1e-300, F_MAX]`. | `prop_lr_bounds_and_monotone_within_phase` |
| L2 | Non-decreasing through warmup (reaches `peak` exactly on the last warmup step), non-increasing after. | same, `prop_lr_phase_boundaries_and_tail` |
| L3 | After warmup `lr >= floor_frac * peak`; `lr_at(total_steps) == floor` exactly when a cosine phase exists; past the end it holds that value. | `prop_lr_phase_boundaries_and_tail` |
| L4 | Edge configs (warmup 0, warmup == total, total 1, warmup > total, warmup < 0) do not crash or go NaN. Pinned behavior below. | `check_documented_edges`, both LR props |
| L5 | `state_dict` is JSON-safe and round-trips to an identical schedule. | `prop_lr_state_dict_round_trip` |
| L6 | An invalid peak / total / floor_frac raises `ValueError` and nothing else; a valid one never yields NaN. | `prop_lr_constructor_validates` |
| L7 | `apply_lr` writes `schedule(step) * lr_mult` onto every dict group and returns the rate. | `prop_apply_lr_writes_rate_times_mult` (peak <= 1e30, so `rate * mult` is finite) |

The spec states only G2, the clamp in G1, the ramp (G6), the warmup length / floor, and the
`rate * lr_mult` rule (L7). Everything else is the code's own documented contract
(docstrings) or a no-crash / no-NaN invariant.

## Edge behavior pinned (not changed)

* One active head: its weight is projected to 1.0 and no step is taken. No active head: no change.
* `total_steps == warmup_steps` (including a stage shorter than the default warmup, since warmup
  is capped at the stage length): peak at the last warmup step, **held at peak** afterwards. There
  is no cosine phase, so the schedule never reaches the floor. The `lr_at` docstring says "past
  the end it holds the floor"; that is only true when a cosine phase exists.
* `warmup_steps=0`: step 0 is `peak`. `warmup_steps < 0` is 0. `warmup_steps > total` is `total`.
* Warmup starts at `peak / warmup`, **below** `floor_frac * peak` when warmup > 20 steps, so
  "lr in `[floor, peak]`" holds only after warmup. In warmup it is `(0, peak]`.
* Domain limits (not bugs): a subnormal `peak` (5e-324) rounds `0.5 * peak` to 0 on the first
  cosine step; `apply_lr` with `peak * lr_mult > F_MAX` overflows to inf (the multiplier, not the
  schedule); torch's float64 `norm` squares the entries, so a coefficient below about 1e-162 reports
  0.0.
* Out of contract, not fixed: duplicate names in `active` (for example `["a", "a"]`) count twice
  in `n_active` while the weights dict has one entry. `HeadRamp.active_names` never produces
  duplicates.

## Counterexamples found (hypothesis-shrunk) and fixes

All were found on `a56f514`, shrunk, pinned as `check_pin_*` and committed fail-first
(`4f691f9`, `29ff44a`), then fixed in one commit.

| id | counterexample | effect | fix |
| --- | --- | --- | --- |
| C1 | `GradNorm(alpha=1024.0)`, losses `(0, 1)`, 2 heads | `OverflowError` (`2.0 ** 1024`) in `update` | `r ** alpha` overflow gives `inf` (target is astronomically large; the weight is pushed up, then clamped) |
| C2 | grad norms `6.4e62`, 2 heads; `project_weights({k: 1.6e61}, 0.1)` | weights `(10.0, 0.1)`, sum 10.1 != 2: the bisection bracket `[0, 1]` x 200 halvings cannot resolve `c < ~6e-61` | bracket starts at `min(1, hi / max(raw))`; raw is capped at float max so `inf` (lr * grad overflow) cannot give a 0 bracket (this hung an earlier draft of the fix; `check_pin_huge_grad_norm_keeps_sum` and the props cover it) |
| C3 | `GradNorm(lr=nan)` / `lr=inf` / `alpha=-1.0` with a zero-loss head | weights `(0.1, 0.1)` / `(10, 0.1)`: renormalization silently broken; `ZeroDivisionError` | constructor raises `ValueError` for non-finite or negative alpha/lr |
| C4 | `WarmupCosine(0.7351220134688766, 118).lr_at(117)` | `0.7351220134688767` > peak by 1 ulp (`peak * 118 / 118`) | warmup is `peak * ((s + 1) / warmup)`, exactly `peak` on the last step |
| C5 | `WarmupCosine(8.98846567431158e307, 2).lr_at(1)` | `inf` (`peak * 2` overflow) | same quotient-first form |
| C4b | `WarmupCosine(7662519389186461.0, 2, warmup_steps=1, floor_frac=0.10546875).lr_at(1)` | cosine start `floor + (peak - floor) * 1.0` is 1 ulp above peak | `min(peak, ...)` on the cosine branch |
| C6 | `per_task_grad_norms({"const": torch.tensor(0.)}, shared)` | `RuntimeError: ... does not require grad` | a loss with `requires_grad=False` reports `0.0` |

Effect on real runs: none expected. Training grad norms are float32 (< 3.4e38), alpha is 1.5,
lr is 0.025, and every fixed path is a 1-ulp or out-of-range case. Fixed-path numerics: the
warmup rate may differ from before by 1 ulp (division order); the cosine branch is bit-identical
except where it would have exceeded peak; `project_weights` is bit-identical whenever
`max(raw) <= W_MAX`. The existing `test_trainer2_balance.py` and `test_trainer2_lr_schedule.py`
pass unchanged.
