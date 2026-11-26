# Trainer 2.0 (GVD 2.0 Phase 2): spec / AC

Status: **spec only**. No code in this PR. Base `origin/main` `98af813`.
Source: `/workspace/gvd2/spec.md` (master spec, "Training:" line; Phase 2).
Owner of merges: Michael. Gate: Skeptic, per PR.

Tags used below: **exists** = on `98af813` (path cited). **new** = does not exist; where it goes is named.
VRAM numbers are **measured** (repo source cited) or **est.** (math shown). The box has no GPU and no torch, so nothing here is measured on a GPU.

## Goal

One opt-in trainer recipe, `train_scene --recipe trainer2` (**new** flag in `python/train/train_scene.py`; default off, so every existing command is bit-identical). It trains the existing `SceneNet` with the existing `SharedStem` (`python/train/scene_net.py`) through a fixed curriculum, with per-task loss/grad monitoring, GradNorm, head ramps, warmup+cosine and per-group LR. It must beat today's trainer on a pre-registered, matched holdout, with one hashed run each. Optional late add-on: a bigger offline teacher distilled into the small live student.

Inputs: **7-cam set**: `repeatL, pillarL, main, narrow, pillarR, repeatR, rear`. GVD is removing `wide` from the model inputs, the recorder and every wide-derived lane/polyline (`lanes.py:810` `src="wide"`, `path_predictor.py:564-590`, `pipeline.py:338`; see `/workspace/gvd-notes/wide_aeb_check.md`). Trainer 2.0 reads the cam list from `python/data/strip_writer.STRIP_ORDER` after that removal. It never names `wide`, and no input or target comes from wide. With 7 cams the sector token is 7×64 = 448 wide, not `TOKEN_DIM` 512. `SceneNet` already adds `self.skip = Linear(448, 512)` when `token_dim != gru_hidden`, so the arch tag changes and 8-cam heads/GRU checkpoints do not load. The `SharedStem` weights are cam-agnostic (one stem shared over every sector), so `--init-stem` from an 8-cam run is legal.

## Non-goals

- **Safety path untouched**: heartbeat, AEB and engage (`python/control/actuate.py`, `python/control/ego_speed.py`, `python/sensors/live_hold.py`, `python/control/echo_link.py`, `python/perception/cipv.py`, `beamng_mod/.../gvd/main.lua`). No trainer module is imported by them (`check_safety_path_no_trainer_imports`).
- No live change. Live runtime (`scene_ckpt_runtime.py`, `scene_live.py`) and the ONNX export contract stay as they are, apart from the 7-cam arch that GVD's wide removal brings.
- No new heavy backbone live. The deployed model is `tiny`/`small` (`python/train/model_size.PRESETS`), YOLOv8n stays for objects, and no new deps.
- No BEV/LSS, 3DGS or refiner *architecture* here. Those stages train heads that Phases 5/6/9 add. Until a head exists on main, its stage is **skipped and logged**, not faked.
- No checkpoint is ever deleted: old 8-cam `.npz`/ONNX and epoch snapshots are kept. The trainer writes new files only.
- No GT, sim depth or teacher on any live path.

## Teacher/student: three different things (kept separate)

| Name | Exists? | What it is | Trainer 2.0 use |
|---|---|---|---|
| SSL target | exists: `python/train/exp_ssl.py` `masked_token_pretext_loss`, flag `--exp-ssl-pretrain` + `--exp-ssl-out` + `--init-stem` | The **same net**, stop-grad (`token_full.detach()`) vs a masked canvas, L1. No EMA. It can collapse: a constant token gives zero loss. | **Promote = stage 0 of `--recipe trainer2`** (on by default inside the recipe; the plain `--exp-ssl-pretrain` flag stays and stays off by default). Upgrade (**new**, `exp_ssl.py`): EMA target stem (momentum 0.996), plus a collapse guard that aborts if the per-dim token std on a fixed train probe is < 0.01. |
| W2 feature distill | exists: `python/train/exp_distill.py` (`load_teacher`, `run_student_epochs`, `StudentFromFeats`), flags `--exp-distill-feats/-cache/-teacher` | A tiny GRU + lane_gt student on cached frozen teacher tokens, for fast head sweeps. Training-only, never exported. | Unchanged. Not part of the curriculum. |
| Big-teacher → live student KD | **new** (`python/train/trainer2_distill.py`) | A bigger offline teacher distills into the deployed small student. | Optional, last PR (see Distillation). |

## Curriculum

Each stage resumes the previous stage's best checkpoint (written by `python/train/epoch_ckpt.py`). **Entry** = the previous stage exited, plus the stage's data and head exist on main. **Exit** = holdout primary metric has no gain > 1% over 3 evals (`--eval-every`), or a max epoch count. Both are pinned in the run manifest before training. **Non-regression:** a stage may not make any earlier-stage holdout metric > 5% (relative) worse. If it does, the stage's checkpoint is rejected and the run stops (proposed thresholds).

| # | Stage | Active heads (exists names) | Data (train rows only) | Exit metric |
|---|---|---|---|---|
| 0 | SSL | `stem` only | Unlabeled frames via `exp_ssl.ssl_rows_from_split` (holdout and 4 s purge excluded) | pretext loss plateau, collapse guard holds |
| 1 | depth/sem | `aux_depth_head`, `aux_sem_head` (`--aux-depth/--aux-sem`) | Recorded sim depth/sem via `python/train/sensor_aux.py`, train-only. Today `cam="main"`; extending to `narrow` + side cams is **new**. | depth AbsRel, sem mIoU (holdout) |
| 2 | lanes | `lane_head`, `lane_valid_head`, `curb_head`, `curb_valid_head`, `lane_gt_head`, `road_edge` | `lane_gt` targets (`scene_net._lane_gt_loss`) from main/narrow, plus 7-cam context. No wide-derived polylines. | 8 m holdout `r`/`rms` (`scripts/eval_scene_ckpt.py`) |
| 3 | BEV | **new** BEV head on the Phase 5 spatial path (`python/train/bev_lss.py` is absent on main; draft #241) | `python/train/occ_map_labels.py` / `occ_labels.py` from the 7 cams. Unknown/occluded cells are never a "free" target. | BEV IoU, free/unknown accuracy |
| 4 | temporal | **new** future-BEV heads at 0.5/1/2 s | Same rows; targets are the future frame's BEV *within the same recording and split* | future BEV IoU per horizon |
| 5 | 3DGS | **new**, per `docs/specs/reality_view_3dgs.md`. Only after RV step 0 (`check_live_depth_source_on_main`) passes: a depth-consistency / rendered-depth loss | Offline recorded frames + poses. GT only in tests named GT. | rendered-depth AbsRel, δ<1.25 (RV Outputs) |
| 6 | refine | Phase 9 refiner, a separate model. Not in SceneNet. | n/a here | skipped until the Phase 9 spec |
| 7 | joint | All heads that exist, peak LR ×0.1 | All of the above | primary + no head worse than its stage exit by > 2% |

## Loss, gradients, LR

- **Per-task loss**: exists. `train_scene` step trace carries `heads` per `scene_net.LOSS_TERMS` (`points, class, lamps, valid, lane_gt`), plus `_loss_terms` extras (`ctl, road_edge, stop_line, aux_depth, aux_sem`). Keep it; add BEV/temporal terms when those heads land.
- **Per-task grad norm**: **new**. Today only the global `clip_grad_norm_` value is logged (`train_scene.py` ~l.1675). Add ‖∂L_i/∂W_shared‖ per term, with W_shared = `stem.fc.weight` (cheap). Log it per step to the existing status/trace JSON.
- **GradNorm** (Chen et al. 2018), **new** `python/train/trainer2_balance.py`: learnable w_i, α = 1.5, renormalized so Σw_i = n_active. Clamp w_i ∈ [0.1, 10]. Grads are taken on `stem.fc.weight` only. Distill terms are excluded (fixed weights).
- **Gradual activation**: a head entering at step s₀ gets ramp(t) = min(1, (t−s₀)/R), R = 2000 steps (proposed). Its effective weight is ramp·w_i. Inactive heads compute no loss.
- **Warmup + cosine**: **new** `python/train/lr_schedule.py`. Per stage: linear warmup over max(500, 5%) of the stage's steps, then cosine to 0.05×peak. Joint peak = 0.1×. It must compose with `_pin_optimizer_lr` (resume keeps the CLI rate × `lr_mult`).
- **Differential LR**: `param_groups(lr, mult)` exists on `PretrainedSceneNet` only (`pretrained_net.py:168`). Add the same on `SceneNet` (**new**). Groups: `stem` ×0.1 after stage 0, heads already trained ×0.3, the entering head ×1.0. Reuse `lr_mult` and `--backbone-lr-mult` semantics.
- Grad clip: existing `clip`. Precision: **fp32**. `model_size.resolve_precision` already keeps Pascal (sm_61) on fp32, and `occ_sparse.py` notes Pascal fp16 math ≈1/64 of fp32.

## Data split and holdout

- Split by **recording + time block, never by frame**. Use the existing #187 machinery: `continual.pass_plan` / `strip_sources.combined_pass_plan` (`--holdout-blocks`), the `HOLDOUT_GAP_S` = 4 s purge, optional geographic cells (`HOLDOUT_CELL_M` 250 m + 50 m buffer, `--holdout-cells`).
- **new** `python/train/trainer2_split.py`: writes `split_manifest.json`, with sorted (source, i) per split plus sha256, **before** step 0. SSL, every stage, the teacher and the distill cache all read this one manifest. Temporal targets may not cross a split boundary.
- `check_trainer2_split_no_overlap`: no (source, i) in both splits; within a recording, min |t_train − t_hold| ≥ 4 s; cells disjoint when on; temporal targets' source frames are in the same split; the SSL row set ⊆ train.
- **Matched-holdout comparison**: baseline = today's `train_scene` default recipe (same `--cams` 7-cam set, same manifest, same epochs budget). Score both with `scripts/eval_scene_ckpt.py` (`n, r, rms` + per-head lines). New heads without a baseline counterpart are reported "no baseline", not as wins.
- **Pre-registration** (`run_manifest.json`, **new**): git SHA, config sha256, seed, split sha256, decision rule. Written before training. **One run per arm, no seed re-rolls.** Trainer 2.0 wins iff holdout `rms` is ≤ baseline and no shared head is worse by > 2%. Otherwise trainer2 stays opt-in and the result is logged as a loss.

## VRAM budgets (two)

Canvas for 7 cams at scale 1 (est.): 6×256 + 1×340 (`rear`, 16:9) + 6×8 gap = **1924×192** (`strip_writer` constants). Stem = 4 stride-2 3×3 convs. Activations per frame ≈ Σ(C·H·W) × 3 stored tensors (conv/GN/ReLU) × 4 B.

**Train budget: full GTX 1080 Ti 11 GB, game off** (Michael, 6:07 PM). Target ≤ 10 GB allocated, leaving ~1 GB for context and fragmentation (est.).
- Student `small` (32,64,128,128): (96·962·32 + 48·481·64 + 24·241·128 + 12·121·128) = 5.36 M floats ×3 ×4 B ≈ **64 MB/frame**, ×T=16 ≈ **1.03 GB/window** without `--grad-ckpt`. With `--grad-ckpt` (exists, scratch net only), ≈ one tick (~64 MB) + tokens. Weights+grad+AdamW ≈ 2.2 M params (`architecture_parameter_count()` = 2,195,486 for 8-cam) ×16 B ≈ 35 MB. Batch: `--batch 0` auto-fit (`resolve_scene_batch` / `_probe_cuda`, exists) → est. ~8 windows without ckpt.
- AMP/fp16: off on sm_61 (see above). It is a memory option only, and only if a run is VRAM-bound, which it isn't at these sizes. Grad accumulation: **new** `--accum N`, used only if the auto-fit batch < 4 windows.

**Deploy budget: live student ≤ 4 GB inference** (`config/hardware.yaml` `max_inference_vram_gb: 4`), ~640 px long side, ONNX FP32 on Pascal, preset `tiny`/`small` only. `small` paper estimate is 2.2 GB (`docs/research/gpu_tiers.md`, est., not a profile). Live `SceneCkptRuntime` is on CPU, est. 34–170 MB for the 8-cam canvas (same doc). Exported student ≈ 2.2 M × 4 B ≈ 9 MB of weights (est.). Supervisor RSS ≤ 12 GB unchanged. `check_student_export_fits_deploy`: the exported ONNX's param count equals the student's `architecture_parameter_count`, its preset ∈ {tiny, small}, and `PRESETS[preset].est_vram_gb` ≤ 4.0. A measured IQU45 VRAM reading is a GVD-scheduled gate, and it skips on the box.

## Distillation (optional, last PR)

- **Teacher**: the `xl` preset (`model_size.PRESETS["xl"]`: stem (96,160,256,256), GRU 1024). This is existing code with no new deps. Params (est., same formula as `architecture_parameter_count`): **≈7.1 M**, 3.2× student. Weights+grad+AdamW ≈ 114 MB. Activations: (96·962·96 + 48·481·160 + 24·241·256 + 12·121·256) = 14.4 M floats ×3 ×4 B ≈ **173 MB/frame**, ≈ **2.77 GB/window** at T=16. Without ckpt, batch 3 ≈ 8.3 GB (fits 10 GB). With `--grad-ckpt`, batch ≥ 16 on activations. FP32 only: sm_61 has no Tensor Cores, and fp16 gives no speedup. Bigger than `xl` (e.g. a widened stem) is not proposed: it is unmeasured, and it would be a new heavy net. All of these numbers are est.
- **Losses** (student total = GT task losses, GradNorm-weighted, + fixed KD terms):
  - L_feat: MSE on the 64-d sector tokens. `SECTOR_FEAT` = 64 for every preset, so no adapter is needed. λ = 1.0.
  - L_logit: smooth-L1 to the teacher's lane/curb points where teacher valid > 0.5. KL at T = 2 (×T²) on class/lamp heads. λ = 0.5.
  - L_depth: L1 on log `aux_depth` vs the teacher's. The GT depth term stays. λ = 0.5.
  - L_bev: per-cell KL **only on cells observed in GT**. Unknown/occluded cells are never pulled toward the teacher's "free". λ = 0.5.
  - Schedule: KD ramps 0→1 over the first 10% of student steps, then holds, then falls linearly to 0.25 during the joint stage, so GT dominates at the end.
  - Teacher frozen, eval mode. Its outputs are cached per **train** row (same rule as `exp_distill`: never holdout).
- **Evidence bar**: the teacher must first beat the no-KD student on the holdout, or KD is not tried. Then `student_kd` vs `student_nokd`: same arch, manifest, seed and config hash, differing only by `--kd`, one hashed run each, no re-rolls. Keep KD iff `student_kd` holdout `rms` < `student_nokd` and no head is worse by > 2%. **Otherwise distillation is dropped.**
- **Isolation**: `check_teacher_not_on_live_path` (AST + fresh-interpreter walk from the live entry points, in the same style as `scripts/test_occ_runtime_imports.py`): `trainer2_distill` and teacher weights are unreachable. `check_onnx_student_only`: the exported graph's initializers match the student exactly, with no teacher-named or teacher-sized tensors.

## Tests (each written first, failing on `98af813`)

All live in `scripts/test_trainer2_*.py`. Torch parts use `gvd_test_env.skip_check` locally (the box has no torch) and run in CI (`.github/workflows/tests.yml` installs torch CPU).
- `check_recipe_off_is_bit_identical`: no `--recipe` → same argparse defaults and same step trace on a 2-step CPU fixture.
- `check_trainer_no_wide_input`: the default trainer2 cam set has no `wide`; `SceneNet(cams=…)` sectors exclude it; a strip with a `wide` sector is refused.
- `check_no_wide_targets`: every target builder used by the recipe refuses rows/polylines with `src == "wide"` or a wide-cam origin. AST check: no `"wide"` literal in the `trainer2_*` modules.
- `check_safety_path_no_trainer_imports`: an AST walk from the safety modules listed in Non-goals never reaches `python.train.*`.
- `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`.
- `check_curriculum_order_and_gates`: stage order is fixed, a stage with a missing head is skipped and logged, and the non-regression gate rejects a checkpoint.
- `check_head_ramp_linear`, `check_gradnorm_weights_sum_and_clamp`, `check_per_task_grad_norm_logged`.
- `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`.
- `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll` (a second run with the same manifest and a different seed is refused).
- `check_student_export_fits_deploy`, `check_old_checkpoints_untouched` (the trainer never unlinks an existing ckpt; only its own `.partial` files).
- Late PR: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`.

## Open questions for Michael

1. "Promote" = stage 0 inside `--recipe trainer2` (proposed), or default-on for every `train_scene` run?
2. Is the baseline for the matched comparison the current default recipe on the 7-cam set (proposed), or the last shipped 8-cam checkpoint? The 8-cam checkpoint can't run on 7-cam inputs.
3. OK to skip the BEV/temporal/3DGS/refine stages (logged) until Phases 5/6/9 land, rather than blocking Trainer 2.0 on them?
4. Teacher = `xl` preset (proposed), or is a larger offline teacher worth an IQU45 VRAM measurement first?

## Appendix A: file touch list

| File | Change |
|---|---|
| `docs/specs/trainer_2.md` | this spec (PR 0) |
| `python/train/train_scene.py` | **edit**: `--recipe trainer2`, `--stage`, `--accum`, `--kd`; schedule/balance hooks in the step loop |
| `python/train/scene_net.py` | **edit**: `SceneNet.param_groups`; per-term grad-norm helper |
| `python/train/exp_ssl.py` | **edit**: EMA target, collapse guard |
| `python/train/lr_schedule.py` | **new** |
| `python/train/trainer2_balance.py` | **new**: GradNorm, ramps |
| `python/train/trainer2_recipe.py` | **new**: stage table, gates |
| `python/train/trainer2_split.py` | **new**: manifest, hashes |
| `python/train/trainer2_distill.py` | **new** (late PR) |
| `scripts/test_trainer2_*.py` | **new** tests |

## Appendix B: verify steps (offline, box CPU)

```
ps aux | grep -c "[r]un_offline_suite"             # must be 0 before starting
python3 scripts/test_trainer2_split.py             # per-PR file, fail-first then pass
python3 scripts/run_offline_suite.py               # full suite; paste the final line
```
CI (torch CPU) runs the same suite on the draft PR. Skeptic re-runs it in `/workspace/skeptic`.

## Appendix C: PR sequence (each separately gateable)

1. **PR 0**: this spec.
2. **PR 1**: split manifest + pre-registration (`trainer2_split.py`). Tests: `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll`.
3. **PR 2**: 7-cam / no-wide guards + safety-import check. Tests: `check_trainer_no_wide_input`, `check_no_wide_targets`, `check_safety_path_no_trainer_imports`. Lands after GVD's wide removal.
4. **PR 3**: LR schedule + `SceneNet.param_groups`. Tests: `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`, `check_recipe_off_is_bit_identical`.
5. **PR 4**: per-task grad norms, GradNorm, ramps. Tests: `check_per_task_grad_norm_logged`, `check_gradnorm_weights_sum_and_clamp`, `check_head_ramp_linear`.
6. **PR 5**: SSL promotion (EMA + guard) as stage 0. Tests: `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`.
7. **PR 6**: curriculum runner (stages 0–2 live; 3–6 skip-logged) + deploy check + ckpt retention. Tests: `check_curriculum_order_and_gates`, `check_student_export_fits_deploy`, `check_old_checkpoints_untouched`.
8. **PR 7**: matched-holdout run (GVD schedules the GPU): one baseline and one trainer2 run, with manifest hashes posted before training. Evidence only, no code.
9. **PR 8 (optional, after PR 7 lands)**: teacher→student KD. Tests: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`. Then one `student_kd` run and one `student_nokd` run. KD is dropped if it does not win.
10. BEV / temporal / 3DGS stages get enabled in the PR that lands each head (Phases 5/6), each with its own `check_*`.
