# GradNorm / warmup-cosine mutation pass (Breaker, swarm-4)

Tool: mutmut 3.8.1 (dev-only, not added to requirements), run on `python/train/trainer2_balance.py` and
`python/train/lr_schedule.py` (every function in both files) with `scripts/test_gradnorm_props.py` as the test runner.
Logs: `swarm-4/gradnorm-evidence/mutmut_run{1..4}.log`, `mutmut_results_run4.txt`.

## History

- run 3 (f7c8321): 562 mutants, 422 killed, 2 timeout, 20 survived, 118 no tests.
- run 4 (test file as of 4b3f7d7 + residual-spread pin + LR group props): 562 mutants, 541 killed, 2 timeout
  (`project_weights` 49/50, the bisection never ends, so detected), 19 survived, **0 no tests**.
- after run 4: `check_per_task_grad_norms_keeps_shared_graph` kills `per_task_grad_norms` 19/23/25 (retain_graph
  None/omitted/False). Confirmed with the same edits on a temp copy (RuntimeError: backward through freed graph).
  Four more exact-sum inputs in `check_pin_residual_spread_makes_sum_exact` kill `project_weights` 62 (`mass(mid) <= total`)
  and 77 (`lo <= v` in the free set). Both were wrongly claimed equivalent after run 3: on 20k random inputs they break
  the exact sum in 1516 and 141 cases. Confirmed by swapping in each mutant on the pin. mutmut was not re-run for these
  five. Net after run 4: 562 total, 546 killed, 2 timeout, 14 survived, 0 no tests.

The 118 no-tests in run 3 were 85 in `stage_lr_mults`/`scene_param_groups`/`_group_of` (PR 3 code, which only the
existing `test_trainer2_lr_schedule.py` covered, not this runner) and 33 in `per_task_grad_norms` (not mapped by
mutmut). Run 4 maps all of them: `prop_stage_lr_mults_spec_numbers` and `prop_scene_param_groups_partition_and_rates`
were added, and the per-task props are now mapped too.

## Remaining survivors (14)

Evidence for `project_weights`: `gradnorm-evidence/pw_equiv_check.txt` (20k random inputs, original vs mutant).

- `apply_lr` 6: `[]` -> `None` inside `getattr(..., [])` followed by `or []`; same result.
- `HeadRamp.factor` 3: `<` -> `<=` at `step == s0`; both return exactly 0.0 (`(s0 - s0) / ramp_steps`, and
  `ramp_steps >= 1` is enforced in the constructor).
- `_group_of` 5, 8: `split(".", 1)[0]` -> `split(".")[0]` / `split(".", 2)[0]`; element 0 is identical for any string.
- `project_weights` 44, 51, 52, 53, 56: bit-identical to the original on all 20k inputs (0 differ). The start bracket
  `min(2.0, ...)`, growth `*3`, cutoff `>=`/the same literal 1e30, and a 201st halving all reach the same final `b`.
- `project_weights` 48 (`while mass(b) <= total`) and 78 (`v <= hi` in the free set): **not bit-identical**. They differ on
  208 and 4 of the 20k inputs, by at most 7.1e-16 and 1.8e-16 relative, and they never break the exact sum, the bounds or
  the water-filling property. They are a different valid rounding of the same projection. Killing them would mean pinning
  the last bit of each weight, which the spec doesn't define. I'm calling these "equivalent under the stated contract
  (exact sum, bounds, proportional water-filling)", not strictly equivalent.
- `per_task_grad_norms` 5, 8, 11: the `getattr(loss, "requires_grad", True)` default -> None / no default / False. Every
  torch tensor has `requires_grad`, so the default is never read on a tensor. They differ only on a non-tensor loss, which
  is outside the contract (the original then fails inside `torch.autograd.grad` anyway).

## Residual-spread path (was "weakly guarded")

`check_pin_residual_spread_makes_sum_exact` pins three inputs where `project_weights` returns `sum == total`
exactly (float equality, the contract stated in the code comment) and where removing the spread, or getting the share wrong,
lands 4.4e-16 to 1.3e-14 off. Mutants 75, 79, 85, 95 are killed in run 4. On 20k random inputs, the spread step changes the
output in about 6% of cases, always at ULP scale. That's the size of the effect: the step guarantees exactness and
nothing larger.
