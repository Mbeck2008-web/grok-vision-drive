# Trainer 2.0 — open questions for Michael

Follow-up to `docs/specs/trainer_2.md`. This file does not gate that spec. The six questions below were moved out of the AC so `trainer_2.md` can be gated on its own. Each lists the spec's proposed default. Michael has not decided these.

1. "Promote" = stage 0 inside `--recipe trainer2` (proposed), or default-on for every `train_scene` run?
   **Proposed default:** stage 0 of `--recipe trainer2` (on by default inside the recipe). The plain `--exp-ssl-pretrain` flag stays and stays off by default.

2. Is the baseline for the matched comparison the current default recipe on the same 8-cam set and manifest (proposed), or the last shipped checkpoint re-scored on that holdout?
   **Proposed default:** today's `train_scene` default recipe on the same 8-cam set, the same split manifest, and the same epochs budget.

3. OK to skip the BEV/temporal/3DGS/refine stages (logged) until Phases 5/6/9 land, rather than blocking Trainer 2.0 on them?
   **Proposed default:** yes — skip and log those stages until their heads exist on main. Do not block Trainer 2.0 on them.

4. Teacher = `xl` preset (proposed), or is a larger offline teacher worth an IQU45 VRAM measurement first?
   **Proposed default:** the existing `xl` preset (`python/train/model_size.PRESETS["xl"]`). No larger offline teacher until an IQU45 measurement says otherwise.

5. `--scene-ckpt` live input is main-only (`scene_ckpt_runtime._main_canvas`). Should GVD switch it to the full 8-cam `compose_strip` before any Trainer 2.0 head goes live?
   **Proposed default:** yes — GVD switches `--scene-ckpt` to the full 8-cam `compose_strip` (a runtime change, outside these PRs). Until that lands, Trainer 2.0 heads must not ship on the `--scene-ckpt` path.

6. Stop-event thresholds (40 m, 2 m/s, 0.5 m/s, +1 m, brake 0.3): OK as the pinned definition?
   **Proposed default:** yes — those thresholds, pinned in `run_manifest.json` before training: a holdout stop event is a GT stop sign/stop line on the ego path within 40 m with ego speed ≥ 2 m/s; a missed brake is v not < 0.5 m/s by stop line + 1 m; a false brake is `ctl_brake` > 0.3 with no stop event within 40 m.
