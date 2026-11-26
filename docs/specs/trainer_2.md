# Trainer 2.0 (GVD 2.0 Phase 2): spec / AC

Status: **spec only**. No code in this PR. Base `origin/main` `98af813`.
Source: `/workspace/gvd2/spec.md` (master spec, "Training:" line; Phase 2).
Owner of merges: Michael. Gate: Skeptic, per PR.

Tags used below: **exists** = on `98af813` (path cited). **new** = does not exist; where it goes is named.
VRAM numbers are **measured** (repo source cited) or **est.** (math shown). The box has no GPU and no torch, so nothing here is measured on a GPU.

## Goal

One opt-in trainer recipe, `train_scene --recipe trainer2` (**new** flag in `python/train/train_scene.py`; default off, so every existing command is bit-identical). It trains the existing `SceneNet` with the existing `SharedStem` (`python/train/scene_net.py`) through a fixed curriculum, with per-task loss/grad monitoring, GradNorm, head ramps, warmup+cosine and per-group LR. It must beat today's trainer on a pre-registered, matched holdout, with one hashed run each. Optional late add-on: a bigger offline teacher distilled into the small live student.

Inputs: **8 cams, as on `98af813`**: `strip_writer.STRIP_ORDER` = `repeatL, pillarL, wide, main, narrow, pillarR, repeatR, rear` (`WIDE_CAMS` = {wide, rear}; 2272×192 canvas; token 8×64 = `TOKEN_DIM` 512). Wide stays mounted with the same FOV and res (Michael, 6:48 PM). Trainer 2.0 reads the cam list from `STRIP_ORDER` and never hard-codes it.
**Wide near-range gating is a dependency, not designed here.** Wide's near-range-only use (lanes, curbs and stop signs a few meters to each side, possibly a low-speed gate, to close the 22–37.5° gap) is defined on GVD's branch `gvd/wide-near-range`. Trainer 2.0 applies **the same wide gating as live**, through the same function that branch exports, with **no separate train-time rule**. Until that branch merges, wide is used exactly as main uses it today. Any wide-derived target the recipe consumes must pass through that same gate (`check_wide_gate_same_as_live`).

## Train input = live input (Michael, 6:37 PM)

Every head trains on the same stitched 8-cam strip the live model receives, wide included.
- Train: `train_scene._frame_row` → `_load_bgr` → `scene_loader.load_canvas` → `raw_frames.load_training_bgr` → `compose_strip(load_raw_cameras(...))`. Hold-last rows go `held_canvas` → `compose_strip`. Then optional `cam_strip.extract_cams` / `scale_canvas`, and train-only `photometric_jitter` (`--augment`).
- Live: `run_vision.py:1551` `compose_strip(frames)` → `scene_guard.observe` → `scene_lane_runtime._strip_nchw` (uint8 → float32 /255, HWC→NCHW).
- **Gap found:** `scene_ckpt_runtime._main_canvas` (`--scene-ckpt`) builds `compose_strip({"main": frame})`, which is **main-only**. A multi-cam head would get a black canvas outside main there. Trainer 2.0 heads must not ship on that path until it feeds all 8 cams (a GVD runtime change, outside these PRs; Open question 5).
- **Normalization** must be one function (**new** `python/train/strip_tensor.py::strip_to_nchw`, moved verbatim from `_strip_nchw`; `exp_ssl.frame_to_nchw` and the trainer call it).
- `check_train_live_input_identical`: on the same recorded fixture frames (`tests/fixtures`), the train builder (augment off) and the live builder (`compose_strip` + `_strip_nchw`) give **byte-identical** float32 NCHW tensors. A sha256 is pinned per fixture, covering cam order (`STRIP_ORDER`, 8 names incl. `wide`), sector crop/x0/x1, resolution and /255. Any drift fails the pin. `--scale` < 1 and `--augment` are train-only and are refused in a recipe meant for a live export.

## Non-goals

- **Safety path untouched**: heartbeat, AEB and engage (`python/control/actuate.py`, `python/control/ego_speed.py`, `python/sensors/live_hold.py`, `python/control/echo_link.py`, `python/perception/cipv.py`, `beamng_mod/.../gvd/main.lua`). No trainer module is imported by them (`check_safety_path_no_trainer_imports`).
- No live change. Live runtime (`scene_ckpt_runtime.py`, `scene_live.py`) and the ONNX export contract stay as they are.
- No new heavy backbone live. The deployed model is `tiny`/`small` (`python/train/model_size.PRESETS`), YOLOv8n stays for objects, and no new deps.
- No BEV/LSS, 3DGS or refiner *architecture* here. Those stages train heads that Phases 5/6/9 add. Until a head exists on main, its stage is **skipped and logged**, not faked.
- No checkpoint is ever deleted: old 8-cam `.npz`/ONNX and epoch snapshots are kept. The trainer writes new files only.
- No GT, sim depth or teacher on any live path.
- **The hand-coded stop-sign rule is not changed**: `python/planning/path_predictor.py:1202-1207` (a `stop_sign` cut at `station − SIGN_STANDOFF_M`, 3.0 m, l.68), the comfort-decel target and hard brake in `_speed_flags` (l.1295-1349), then `fuse_planner_cipv` → `actuate.plan_command`. It stays as the backstop; see Learned stop braking.

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
| 2 | lanes | `lane_head`, `lane_valid_head`, `curb_head`, `curb_valid_head`, `lane_gt_head`, `road_edge` | `lane_gt` targets (`scene_net._lane_gt_loss`) on the 8-cam strip. Wide-derived lane/curb targets only through `gvd/wide-near-range`'s gate (same as live). | 8 m holdout `r`/`rms` (`scripts/eval_scene_ckpt.py`) |
| 2b | stop (learned stop braking) | `sign_head`/`sign_class_head` (`SIGN_CLASSES` has `stop_sign`), `stop_line` (train-only, `intersection_labels.stop_line_arrays`), `ctl_head` → `ctl_brake` (offline only, `ctl_targets.CTL_FIELDS` `brake_ai`, horizons 0–5 s) | Camera strip in. Labels: recorded `brake_ai` + sim stop-line/sign GT, **training labels only** | missed / false brakes, stop distance (see below) |
| 3 | BEV | **new** BEV head on the Phase 5 spatial path (`python/train/bev_lss.py` is absent on main; draft #241) | `python/train/occ_map_labels.py` / `occ_labels.py` from the 8 cams (wide gated as live). Unknown/occluded cells are never a "free" target. | BEV IoU, free/unknown accuracy |
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
- **Matched-holdout comparison**: baseline = today's `train_scene` default recipe (same `--cams` 8-cam set, same manifest, same epochs budget). Score both with `scripts/eval_scene_ckpt.py` (`n, r, rms` + per-head lines). New heads without a baseline counterpart are reported "no baseline", not as wins.
- **Pre-registration** (`run_manifest.json`, **new**): git SHA, config sha256, seed, split sha256, decision rule. Written before training. **One run per arm, no seed re-rolls.** Trainer 2.0 wins iff holdout `rms` is ≤ baseline and no shared head is worse by > 2%. Otherwise trainer2 stays opt-in and the result is logged as a loss.

## Learned stop braking (Michael, 6:37 PM)

- **Goal:** the model learns stop-sign braking end to end from the camera strip. Planner-path AEB and the hand-coded rule above stay in place, unchanged, as the safety backstop.
- **Inference:** camera-only (the 8-cam strip, wide gated as live). **Sim GT** (stop-line position via `intersection_labels`, sign class/pose, the bot's recorded `brake_ai`) is used **only as a training label**: never an input, never on a live path. GT modules are in the `test_occ_runtime_imports.py`-style isolation set.
- **Events (new, `python/train/stop_events.py`):** a holdout stop event is any approach where a GT stop sign/stop line lies on the ego path within 40 m and ego speed is ≥ 2 m/s. Thresholds are proposed and pinned in `run_manifest.json`.
- **Metrics** (new `scripts/eval_stop_brake.py`, offline, recorded holdout):
  - **missed brakes:** the predicted brake profile does not reach v < 0.5 m/s by stop line + 1 m. **Must be 0.**
  - **false brakes:** `ctl_brake` > 0.3 with no stop event within 40 m, counted per km.
  - **stop distance:** signed m from the stop line at v < 0.5 m/s, median and p95.
- **Known risk: 22–37.5° side-angle gap.** Stop signs 22–37.5° off-axis are covered only by `wide`. Without wide, GVD counted 7 missed stop-sign brakes. Wide stays, but only its near-range data is used (`gvd/wide-near-range`), so the far/side coverage there depends on that gate. A camera-only learned brake cannot beat what its gated inputs see. The hand rule and AEB stay as the backstop.
- **Side-angle holdout slice:** a separate `stop_side_angle` slice covers events whose stop sign bearing from ego is 22–37.5° (the side-angle gap) at first GT visibility. It is reported on its own (missed/false/stop distance), and 0 missed brakes is required on it too. It is never pooled away into the overall number.
- **Comparator:** the hand-coded rule replayed read-only (`predict_path` on the same recorded holdout frames' detections), on the same split manifest, metrics and events. One hashed run each, no re-rolls.
- **Retirement rule:** the hand-coded rule and AEB are **never** removed or weakened in any Trainer 2.0 PR. Retiring the hand-coded stop rule needs all of: 0 missed brakes on the matched holdout, ≤ the rule's false-brake rate, and stop distance p95 no worse. Then **Michael's explicit OK** and a separate PR outside Trainer 2.0. AEB (`cipv.aeb`, `plan.aeb`) is not part of that retirement and stays as the backstop.
- `ctl` and `stop_line` stay offline/not exported in Trainer 2.0. Using a learned brake live is a later, separately gated spec.

## VRAM budgets (two)

Canvas, 8 cams at scale 1 (`strip_writer`: 6×256 + 2×340 (`wide`, `rear`) + 7×8 gap = **2272×192**). Stem = 4 stride-2 3×3 convs → 96×1136, 48×568, 24×284, 12×142. Activations per frame ≈ Σ(C·H·W) × 3 stored tensors (conv/GN/ReLU) × 4 B.

**Train budget: full GTX 1080 Ti 11 GB, game off** (Michael, 6:07 PM). Target ≤ 10 GB allocated, leaving ~1 GB for context and fragmentation (est.).
- Student `small` (32,64,128,128): (96·1136·32 + 48·568·64 + 24·284·128 + 12·142·128) = 6.33 M floats ×3 ×4 B ≈ **76 MB/frame**, ×T=16 ≈ **1.21 GB/window** without `--grad-ckpt`. With `--grad-ckpt` (exists, scratch net only), ≈ one tick (~76 MB) + tokens. Weights+grad+AdamW: 2,195,486 params (`architecture_parameter_count()`, 8-cam) ×16 B ≈ 35 MB. Batch: `--batch 0` auto-fit (`resolve_scene_batch` / `_probe_cuda`, exists) → est. ~7 windows without ckpt.
- AMP/fp16: off on sm_61 (see above). It is a memory option only, and only if a run is VRAM-bound, which it isn't at these sizes. Grad accumulation: **new** `--accum N`, used only if the auto-fit batch < 4 windows.

**Deploy budget: live student ≤ 4 GB inference** (`config/hardware.yaml` `max_inference_vram_gb: 4`), ~640 px long side, ONNX FP32 on Pascal, preset `tiny`/`small` only. `small` paper estimate is 2.2 GB (`docs/research/gpu_tiers.md`, est., not a profile). Live `SceneCkptRuntime` is on CPU, est. 34–170 MB for the 8-cam canvas (same doc). Exported student ≈ 2.2 M × 4 B ≈ 9 MB of weights (est.). Supervisor RSS ≤ 12 GB unchanged. `check_student_export_fits_deploy`: the exported ONNX's param count equals the student's `architecture_parameter_count`, its preset ∈ {tiny, small}, and `PRESETS[preset].est_vram_gb` ≤ 4.0. A measured IQU45 VRAM reading is a GVD-scheduled gate, and it skips on the box.

## Distillation (optional, last PR)

- **Teacher**: the `xl` preset (`model_size.PRESETS["xl"]`: stem (96,160,256,256), GRU 1024). This is existing code with no new deps. Params (est., same formula as `architecture_parameter_count`): **≈7.1 M**, 3.2× student. Weights+grad+AdamW ≈ 114 MB. Activations: (96·1136·96 + 48·568·160 + 24·284·256 + 12·142·256) = 17.0 M floats ×3 ×4 B ≈ **204 MB/frame**, ≈ **3.27 GB/window** at T=16. Without ckpt, batch 2 ≈ 6.5 GB (batch 3 ≈ 9.8 GB is too tight for the 10 GB target). With `--grad-ckpt`, batch ≥ 16 on activations. FP32 only: sm_61 has no Tensor Cores, and fp16 gives no speedup. Bigger than `xl` (e.g. a widened stem) is not proposed: it is unmeasured, and it would be a new heavy net. All of these numbers are est.
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
- `check_wide_gate_same_as_live`: the recipe's wide handling calls the same gate function as live (defined by `gvd/wide-near-range`; identity check on the function object plus an AST check that no `trainer2_*` module defines its own wide range/speed rule). Until that branch merges, it asserts wide is passed through exactly as main does today.
- `check_safety_path_no_trainer_imports`: an AST walk from the safety modules listed in Non-goals never reaches `python.train.*`.
- `check_train_live_input_identical` (hash pin, see Train input = live input), `check_strip_order_matches_live`.
- `check_stop_labels_train_only` (stop GT/`brake_ai` is never an input; GT modules are unreachable from live entry points), `check_stop_events_split_safe` (events are counted in holdout rows only), `check_stop_side_angle_slice` (a 30° fixture event lands in `stop_side_angle` and is reported separately), `check_stop_metrics_on_fixture` (a hand-built clip with 1 missed brake, 1 false brake and a known stop distance gives exactly those numbers), `check_hand_rule_untouched` (sha256 pin of `path_predictor.py`, `pipeline.py`, `actuate.py`, `cipv.py` at the base SHA; any Trainer 2.0 PR that changes them fails).
- `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`.
- `check_curriculum_order_and_gates`: stage order is fixed, a stage with a missing head is skipped and logged, and the non-regression gate rejects a checkpoint.
- `check_head_ramp_linear`, `check_gradnorm_weights_sum_and_clamp`, `check_per_task_grad_norm_logged`.
- `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`.
- `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll` (a second run with the same manifest and a different seed is refused).
- `check_student_export_fits_deploy`, `check_old_checkpoints_untouched` (the trainer never unlinks an existing ckpt; only its own `.partial` files).
- Late PR: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`.

## Open questions for Michael

1. "Promote" = stage 0 inside `--recipe trainer2` (proposed), or default-on for every `train_scene` run?
2. Is the baseline for the matched comparison the current default recipe on the same 8-cam set and manifest (proposed), or the last shipped checkpoint re-scored on that holdout?
3. OK to skip the BEV/temporal/3DGS/refine stages (logged) until Phases 5/6/9 land, rather than blocking Trainer 2.0 on them?
4. Teacher = `xl` preset (proposed), or is a larger offline teacher worth an IQU45 VRAM measurement first?
5. `--scene-ckpt` live input is main-only (`scene_ckpt_runtime._main_canvas`). Should GVD switch it to the full 8-cam `compose_strip` before any Trainer 2.0 head goes live?
6. Stop-event thresholds (40 m, 2 m/s, 0.5 m/s, +1 m, brake 0.3): OK as the pinned definition?

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
| `python/train/strip_tensor.py` | **new**: single `strip_to_nchw` (moved from `scene_lane_runtime._strip_nchw`, behavior identical) |
| `python/train/scene_lane_runtime.py` | **edit**: call `strip_to_nchw` (no behavior change, pinned by hash) |
| `python/train/stop_events.py`, `scripts/eval_stop_brake.py` | **new**: offline stop events + metrics |
| `scripts/test_trainer2_*.py` | **new** tests |
| `python/planning/*`, `python/perception/pipeline.py`, `python/perception/cipv.py`, `python/control/*` | **not touched** (pinned) |

## Appendix B: verify steps (offline, box CPU)

```
ps aux | grep -c "[r]un_offline_suite"             # must be 0 before starting
python3 scripts/test_trainer2_split.py             # per-PR file, fail-first then pass
python3 scripts/test_trainer2_input_parity.py      # train vs live tensor hash pin
python3 scripts/eval_stop_brake.py --ckpt <student.npz> --manifest <split_manifest.json>  # learned vs hand rule, offline
git diff --name-only origin/main...HEAD | grep -E '^python/(planning|control)/|perception/(pipeline|cipv)\.py' && echo FAIL-safety-touched
python3 scripts/run_offline_suite.py               # full suite; paste the final line
```
CI (torch CPU) runs the same suite on the draft PR. Skeptic re-runs it in `/workspace/skeptic`.

## Appendix C: PR sequence (each separately gateable)

1. **PR 0**: this spec.
2. **PR 1**: split manifest + pre-registration (`trainer2_split.py`). Tests: `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll`.
2b. **Dependency (GVD's, not Trainer 2.0): `gvd/wide-near-range`** defines wide's near-range gating. PR 2's gate check binds to its function once it merges. Before then, PR 2 pins today's pass-through.
3. **PR 2**: wide-gate parity + safety-import check. Tests: `check_wide_gate_same_as_live`, `check_safety_path_no_trainer_imports`.
3b. **PR 2b**: one input builder. `strip_to_nchw` plus the train/live hash pin. Tests: `check_train_live_input_identical`, `check_strip_order_matches_live`. No live behavior change.
4. **PR 3**: LR schedule + `SceneNet.param_groups`. Tests: `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`, `check_recipe_off_is_bit_identical`.
5. **PR 4**: per-task grad norms, GradNorm, ramps. Tests: `check_per_task_grad_norm_logged`, `check_gradnorm_weights_sum_and_clamp`, `check_head_ramp_linear`.
6. **PR 5**: SSL promotion (EMA + guard) as stage 0. Tests: `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`.
7. **PR 6**: curriculum runner (stages 0–2 live; 3–6 skip-logged) + deploy check + ckpt retention. Tests: `check_curriculum_order_and_gates`, `check_student_export_fits_deploy`, `check_old_checkpoints_untouched`.
7b. **PR 6b**: learned stop braking, stage 2b. Labels, events, metrics and the replay comparator. Tests: `check_stop_labels_train_only`, `check_stop_events_split_safe`, `check_stop_metrics_on_fixture`, `check_hand_rule_untouched`. The hand rule and AEB are not changed.
8. **PR 7**: matched-holdout run (GVD schedules the GPU): one baseline and one trainer2 run, with manifest hashes posted before training, plus the stop-brake table (learned vs hand rule). Evidence only, no code. Any retirement of the hand rule is a separate post-Trainer-2.0 PR, with Michael's OK only.
9. **PR 8 (optional, after PR 7 lands)**: teacher→student KD. Tests: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`. Then one `student_kd` run and one `student_nokd` run. KD is dropped if it does not win.
10. BEV / temporal / 3DGS stages get enabled in the PR that lands each head (Phases 5/6), each with its own `check_*`.
