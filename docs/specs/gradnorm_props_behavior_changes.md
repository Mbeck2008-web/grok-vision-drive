# gvd/gradnorm-props: production behavior changes (all in e738122)

Each change has a fail-first test that fails on a56f514 (log: failfirst_run_commit2.log, 15 of 20 failing) and
passes on the branch head. Everything else on the branch is test or docs only.

| # | File / function | Old behavior on a56f514 | New behavior | Fail-first test |
|---|---|---|---|---|
| 1 | `trainer2_balance.GradNorm.update` | `r ** alpha` raises `OverflowError` for r > 1 with a huge alpha (C1: alpha=1024) | overflow becomes `target = inf`; the head steps down, the result stays finite and renormalized | `check_pin_alpha_overflow`, `prop_gradnorm_bad_active_input_raises_and_keeps_state` |
| 2 | `trainer2_balance.project_weights` bracket | bracket fixed at [0, 1]; after 200 halvings `c` can't go below ~6e-61, so a raw weight > ~1e61 returned (10.0, 0.1), sum 10.1 != 2 (C2) | bracket starts at `min(1.0, hi / max(raw))`. weights with max <= hi still start at 1.0, so the ordinary path is unchanged | `check_pin_huge_grad_norm_keeps_sum`, `prop_project_weights_exact_sum_in_clamp`, `prop_gradnorm_long_horizon` |
| 3 | `trainer2_balance.project_weights` raw clamp | an `inf` raw weight (lr * grad_norm overflow) would make the new bracket `hi / inf = 0` | raw weights clamped to `sys.float_info.max` (still the largest), so the bracket stays > 0 | covered by `prop_gradnorm_inactive_garbage_and_zero_heads` and `prop_gradnorm_finite_clamped_renormalized` (inf/1e30 grad norms). this goes with #2: on a56f514 the old bracket didn't need it |
| 4 | `trainer2_balance.GradNorm.__init__` | NaN/inf lr accepted (sum .2 or 10.1 instead of n); negative alpha accepted, then `0.0 ** -1` gives `ZeroDivisionError` in update (C3) | `ValueError` naming alpha/lr if it isn't finite and >= 0 | `check_pin_ctor_refuses_poison_hyperparams` |
| 5 | `lr_schedule.WarmupCosine.lr_at` warmup | `peak * (s+1) / w` can be 1 ulp above peak on the last warmup step (C4), and `peak * (s+1)` overflows to inf for a huge peak (C5) | `peak * ((s+1) / w)`: exactly peak on the last step, no overflow. values on other warmup steps can move by <= 1 ulp | `check_pin_warmup_last_step_not_above_peak`, `check_pin_peak_overflow_in_warmup` |
| 6 | `lr_schedule.WarmupCosine.lr_at` cosine | `floor + (peak - floor) * 1.0` can round 1 ulp above peak at the cosine start | `min(peak, ...)`; only changes the value when it would exceed peak | `check_pin_cosine_start_not_above_peak` (pinned in 29ff44a, failing there) |
| 7 | `trainer2_balance.per_task_grad_norms` | a constant loss tensor (no graph) raises `RuntimeError` from autograd (C6) | grad norm 0.0 for it, the same as an unused head | `check_pin_constant_loss_has_zero_grad_norm`, `prop_per_task_grad_norms_zero_unused_const_heads` |

Out of contract and NOT changed: duplicate names in `active` (e.g. `['a','a']`) are not deduplicated.
