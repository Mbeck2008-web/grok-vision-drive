# GradNorm / warmup-cosine mutation pass (Breaker, swarm-4)

Tool: mutmut (dev-only, not added to requirements), run on `python/train/trainer2_balance.py` and
`python/train/lr_schedule.py` with `scripts/test_gradnorm_props.py` as the test runner.

## mutmut run 3 (head f7c8321)

562 mutants: 422 killed, 2 timeout (both `project_weights`, counted as detected), 20 survived, 118 no tests.

## Survivors after run 3

Killed afterwards by `check_on_target_head_beside_moving_heads_takes_no_step` (confirmed by hand-applied
mutants on a temp copy, mutmut not re-run): `GradNorm.update` 71, 74, 77, 78, 79. All change the sign term
only when `diff == 0` exactly; the old equal-heads check could not see them because projection renormalizes
every head the same way. The new check puts one head exactly on target beside two moving heads.

Equivalent (no observable difference):
- `apply_lr` 6: `[]` -> `None` before `or []`.
- `HeadRamp.factor` 3: `<` -> `<=` at `step == s0`; both return 0.0 (`ramp_steps >= 1` is enforced).
- `project_weights` 44, 48, 51, 52, 53, 56, 62: bisection bracket / iteration details that converge to the
  same root (53 rewrites 1e30 to the same literal).
- `project_weights` 77, 78: free-set boundary `<`/`<=`; clamped weights already sit on the bound.
- `project_weights` 75, 79, 85, 95: residual spread after 200 bisection steps. The residue is ULP-level,
  far under the 1e-12 sum tolerance, so they are equivalent at spec tolerance. This path is weakly guarded.

## No tests (118)

mutmut mapped no test to: `lr_schedule.scene_param_groups` (50), `lr_schedule.stage_lr_mults` (18),
`lr_schedule._group_of` (17). They're outside the GradNorm / warmup-cosine scope of this branch.

`trainer2_balance.per_task_grad_norms` (33) shows as no tests because of mutmut's coverage mapping (torch autograd), but
`prop_per_task_grad_norms_zero_unused_const_heads` does exercise it. Hand mutants on a temp copy, full props
script: 5 of 6 killed (const-head 0.0 -> 1.0, None-grad 0.0 -> 1.0, norm -> norm**2, allow_unused False,
const-head branch removed). Survivor: `getattr(loss, "requires_grad", True)` default -> False, which is
equivalent for every torch tensor (the attribute always exists).
