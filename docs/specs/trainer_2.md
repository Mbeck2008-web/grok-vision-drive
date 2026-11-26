# Trainer 2.0 (GVD 2.0 Phase 2): spec / AC

Status: **spec only**. No code in this PR. Base `origin/main` `3f1fab1` (this branch merged main at `ac5ab3f` / PR #251, `9be50921` / PR #250, and `3f1fab1` / PR #252).
Source: `/workspace/gvd2/spec.md` (master spec, "Training:" line; Phase 2).
Owner of merges: Michael. Gate: Skeptic, per PR.
Owner questions asked in an earlier draft were answered by Michael on 2026-11-25 (8:25–8:29 PM CT, relayed by GVD). Decisions are inlined below and labeled **decided by Michael 2026-11-25**.

Tags used below: **exists** = on `98af813` (path cited). **new** = does not exist; where it goes is named.
VRAM numbers are **measured** (repo source cited) or **est.** (math shown). The box has no GPU and no torch, so nothing here is measured on a GPU. Figures stay estimates until the IQU45 xl-fit measurement (below) replaces them.

## Goal

A trainer recipe, `train_scene --recipe trainer2` (**new** flag in `python/train/train_scene.py`; **spec proposal:** the recipe flag itself stays default off). It trains the existing `SceneNet` with the existing `SharedStem` (`python/train/scene_net.py`) through a fixed curriculum, with per-task loss/grad monitoring, GradNorm, head ramps, warmup+cosine and per-group LR. Optional late add-on: a bigger offline teacher distilled into the small live student.

**SSL is default ON for every `train_scene` run**, not only inside `--recipe trainer2` (**decided by Michael 2026-11-25**: "It should be on every training run"). This **changes the default behavior of plain `train_scene`** relative to `98af813`. **Spec proposal (not Michael's choice):** keep an explicit opt-out (`--no-ssl-pretrain`, **new**) so A/B baselines can restore the old supervised-only path. The baseline arm of every comparison must state whether SSL was on. See Teacher/student and the decision-rule tiers.

Inputs: **8 cams, as on `98af813`**: `strip_writer.STRIP_ORDER` = `repeatL, pillarL, wide, main, narrow, pillarR, repeatR, rear` (`WIDE_CAMS` = {wide, rear}; 2272×192 canvas; token 8×64 = `TOKEN_DIM` 512). Wide stays mounted with the same FOV and res (Michael, 6:48 PM). Trainer 2.0 reads the cam list from `STRIP_ORDER` and never hard-codes it.

**Wide near-range gating — unblocked.** PR #251 has merged to main (`ac5ab3f`; branch head was `gvd/wide-near-range` `d3546a6`). Title: *lanes: gate wide-camera lane points to the near field (keep wide; 0 missed AEB brakes)*. On current main (`3f1fab1`), `python/perception/lanes.py` defines `WIDE_NEAR_MAX_Y_M = 9.0` (l.69) and `WIDE_NEAR_MAX_X_M = 5.0` (l.70). `_estimate_span_lanes(..., gate_wide: bool = True)` is declared at l.805–809. The point filter is applied **inline** at l.834–838: wide points with `y > 9.0` m or `|x| > 5.0` m are dropped. A wide polyline left with fewer than 2 points is dropped whole at l.839–840. There is still **no separately named gate function**. This spec does not invent one.

**Live call (default on).** The live planner path is `python/perception/pipeline.py:193` `lanes_from_view(main_bgr, stitch_bgr)` → `estimate_lanes(..., x_spans=windshield_x_spans(...))` (`lanes.py:592–596` on a full stitch, or l.608–612) → `_estimate_lanes_impl` (defined l.861; called from `estimate_lanes` at l.94–99) → `_estimate_span_lanes(bgr, far_m, x_spans)` at l.869. That call does **not** pass `gate_wide`, so the default **`gate_wide=True`** applies. The 8-cam *pixels* are still painted in full by `python/data/strip_writer.compose_strip` (live: `run_vision.py:1603`; train: `raw_frames.load_training_bgr` / `held_canvas`); the gate is on wide-derived *lane points*, not on the stitch.

**Trainer 2.0 parity (option a, picked because it is the smallest change):** there is already one implementation with a True default, so this spec does not extract a new named function (option b would be a code change just to name an inline filter). The trainer must call the same `_estimate_span_lanes(..., gate_wide=True)` path (or `estimate_lanes` / `lanes_from_view`, which reach it) and **import** `WIDE_NEAR_MAX_Y_M` and `WIDE_NEAR_MAX_X_M` from `python.perception.lanes`, never redefining them. It adds no train-time wide range or speed rule of its own.

**`check_wide_gate_same_as_live`:** (1) an AST / grep check that no `trainer2_*` module (and no train target builder used by the recipe) assigns `WIDE_NEAR_MAX_Y_M` or `WIDE_NEAR_MAX_X_M`, or hard-codes 9.0 / 5.0 as a wide gate; (2) on a fixture stitch, the trainer's wide-derived lane points are identical to `lanes._estimate_span_lanes(..., gate_wide=True)` (and differ from `gate_wide=False`, which is already used in `scripts/test_wide_near_range.py`).

**#251 evidence, cited by GVD:** 255-case sweep (85 strips × stop sign / closing lead / curve lead) = 216/216 brakes vs main `98af813`, **0 missed**, **0 new** brakes; full-record diffs on `aeb`, `stop_reason`, and `lane_source` = 0. Pinned baseline: `tests/fixtures/aeb_baseline/main_98af813_255.json`.

## Train input = live input (Michael, 6:37 PM; Q5 decided 2026-11-25)

Every head trains on the same stitched 8-cam strip the live model receives, wide included. **The full 8-cam `compose_strip` is the live and test input every time** (**decided by Michael 2026-11-25**). `--scene-ckpt` switches off main-only (`scene_ckpt_runtime._main_canvas`). That runtime change is a Trainer 2.0 PR, and it lands **ahead of any Trainer 2.0 head going live**. After it, the previous main-only `--scene-ckpt` path is gone (resolved by that PR, not left as a residual risk).
- Train: `train_scene._frame_row` → `_load_bgr` → `scene_loader.load_canvas` → `raw_frames.load_training_bgr` → `compose_strip(load_raw_cameras(...))`. Hold-last rows go `held_canvas` → `compose_strip`. Then optional `cam_strip.extract_cams` / `scale_canvas`, and train-only `photometric_jitter` (`--augment`).
- Live (lane path today): `run_vision.py:1603` `compose_strip(frames)` → `scene_guard.observe` → `scene_lane_runtime._strip_nchw` (uint8 → float32 /255, HWC→NCHW).
- Live (`--scene-ckpt`, after the switch): the same `compose_strip` of the live 8-cam frame dict, not `compose_strip({"main": frame})`.
- **Normalization** exists twice today: trainer `scene_net.SceneNet._strip_nchw`, live `scene_lane_runtime._strip_nchw`. They are cross-checked only on synthetic arrays by `scripts/test_scene_strip_scale.py` (#140, `check_torch_cross`). **New** `python/train/strip_tensor.py::strip_to_nchw` makes it one function (moved verbatim); both sites and `exp_ssl.frame_to_nchw` call it. The pin below covers the whole stitch → tensor path on real recorded frames, not just the scaling.
- `check_train_live_input_identical`: on the same recorded fixture frames (`tests/fixtures`), the train builder (augment off), the live lane builder (`compose_strip` + `_strip_nchw`), **and** the `--scene-ckpt` input builder give **byte-identical** float32 NCHW tensors (the `--scene-ckpt` path may be a sibling check if the wiring is cleaner that way; both must pass). A sha256 is pinned per fixture, covering cam order (`STRIP_ORDER`, 8 names incl. `wide`), sector crop/x0/x1, resolution and /255. Any drift fails the pin. `--scale` < 1 and `--augment` are train-only and are refused in a recipe meant for a live export.

## Non-goals

- **Safety path untouched**: heartbeat, AEB and engage (`python/control/actuate.py`, `python/control/ego_speed.py`, `python/sensors/live_hold.py`, `python/control/echo_link.py`, `python/perception/cipv.py`, `beamng_mod/.../gvd/main.lua`). No trainer module is imported by them (`check_safety_path_no_trainer_imports`).
- No live change except the `--scene-ckpt` 8-cam switch named above. Heartbeat / AEB / engage stay as they are. The ONNX export contract stays student-only, `tiny`/`small`.
- No new heavy backbone live. The deployed model is `tiny`/`small` (`python/train/model_size.PRESETS`), YOLOv8n stays for objects, and no new deps.
- **BEV / temporal / 3DGS / refine stages stay skipped and logged until Reality View is built and integrated**, then they get implemented (**decided by Michael 2026-11-25**). Entry is gated on that, not on master-spec Phases 5/6/9.
- No checkpoint is ever deleted: old 8-cam `.npz`/ONNX and epoch snapshots are kept. The trainer writes new files only.
- No GT, sim depth or teacher on any live path. Runtime stays camera-only.
- **Pedestrians are out of scope** (BeamNG has little pedestrian data). No pedestrian class, labels, heads, or 2.0-bar slice. `OBJECT_CLASSES` on `98af813` is already `(vehicle, car, truck, bus, bike)` (`python/train/scene_net.py`).
- **RL is out of scope for v1** (a later option, not this spec). Fail-and-learn is DAgger-style supervised relabel only.
- **The hand-coded stop-sign rule is not changed**: `python/planning/path_predictor.py:1202-1207` (a `stop_sign` cut at `station − SIGN_STANDOFF_M`, 3.0 m, l.68), the comfort-decel target and hard brake in `_speed_flags` (l.1295-1349), then `fuse_planner_cipv` → `actuate.plan_command`. It stays as the backstop; see Learned stop braking. Traffic-light cuts in the same function (`:1215`) are also left untouched.

## Teacher/student: three different things (kept separate)

| Name | Exists? | What it is | Trainer 2.0 use |
|---|---|---|---|
| SSL target | exists: `python/train/exp_ssl.py` `masked_token_pretext_loss`, flag `--exp-ssl-pretrain` + `--exp-ssl-out` + `--init-stem` | The **same net**, stop-grad (`token_full.detach()`) vs a masked canvas, L1. No EMA. It can collapse: a constant token gives zero loss. | **Promote = default ON for every `train_scene` run** (**decided by Michael 2026-11-25**). The existing `--exp-ssl-pretrain` becomes the default (or is implied by the default recipe of `train_scene`). **New** `--no-ssl-pretrain` is the explicit opt-out for A/B baselines (**spec proposal**). This changes default `train_scene` behavior vs `98af813`. Upgrade (**new**, `exp_ssl.py`): EMA target stem (momentum 0.996) plus a collapse guard that aborts if the per-dim token std on a fixed train probe is < 0.01 — both figures are a **spec proposal**. It is also stage 0 of `--recipe trainer2`. |
| W2 feature distill | exists: `python/train/exp_distill.py` (`load_teacher`, `run_student_epochs`, `StudentFromFeats`), flags `--exp-distill-feats/-cache/-teacher` | A tiny GRU + lane_gt student on cached frozen teacher tokens, for fast head sweeps. Training-only, never exported. | Unchanged. Not part of the curriculum. Not the 2.0 teacher. |
| Big-teacher → live student KD | **new** (`python/train/trainer2_distill.py`) | A bigger offline teacher distills into the deployed small student. | Optional, last training PR, and only after the xl-fit measurement (see Distillation). |

## Curriculum

Each stage resumes the previous stage's best checkpoint (written by `python/train/epoch_ckpt.py`). **Entry** = the previous stage exited, plus the stage's data and head exist on main. For stages 3–6, entry also requires Reality View built and integrated (**decided by Michael 2026-11-25**); until then those stages are skipped and logged. **Exit** = holdout primary metric has no gain > 1% over 3 evals (`--eval-every`), or a max epoch count (**spec proposal**). Both are pinned in the run manifest before training. **Non-regression (spec proposal):** a stage may not make any earlier-stage holdout metric > 5% (relative) worse. If it does, the stage's checkpoint is rejected and the run stops.

| # | Stage | Active heads (exists names) | Data (train rows only) | Exit metric |
|---|---|---|---|---|
| 0 | SSL | `stem` only | Unlabeled frames via `exp_ssl.ssl_rows_from_split` (holdout, held-out map, and 4 s purge excluded) | pretext loss plateau, collapse guard holds |
| 1 | depth/sem | `aux_depth_head`, `aux_sem_head` (`--aux-depth/--aux-sem`) | Recorded sim depth/sem via `python/train/sensor_aux.py`, train-only. Today `cam="main"`; extending to `narrow` + side cams is **new**. | depth AbsRel, sem mIoU (holdout) |
| 2 | lanes | `lane_head`, `lane_valid_head`, `curb_head`, `curb_valid_head`, `lane_gt_head`, `road_edge` | `lane_gt` targets (`scene_net._lane_gt_loss`) on the 8-cam strip. Wide-derived lane/curb targets go through the same live gate: `_estimate_span_lanes(..., gate_wide=True)` importing `WIDE_NEAR_MAX_Y_M` / `WIDE_NEAR_MAX_X_M` (see Wide near-range gating). No train-time wide rule. | 8 m holdout `r`/`rms` (`scripts/eval_scene_ckpt.py`) |
| 2b | stop (learned stop + light braking) | `sign_head`/`sign_class_head`/`sign_state_head` (`SIGN_CLASSES` includes `stop_sign`, `yield`, `traffic_light`; `LAMP_STATES` is `unknown, red, yellow, green`), `stop_line` (train-only, `intersection_labels.stop_line_arrays`), `ctl_head` → `ctl_brake` (offline only, `ctl_targets.CTL_FIELDS` `brake_ai`, horizons 0–5 s) | Camera strip in. Labels: recorded `brake_ai` + sim stop-line/sign/lamp GT, **training labels only** | missed / false brakes (signs and lights), stop distance, plus the scenario slices that are labelable (see below) |
| 3 | BEV | **new** BEV head on the spatial path (`python/train/bev_lss.py` is absent on main) | `python/train/occ_map_labels.py` / `occ_labels.py` from the 8 cams (wide as live). Unknown/occluded cells are never a "free" target. | BEV IoU, free/unknown accuracy. **Skipped until Reality View is built and integrated.** |
| 4 | temporal | **new** future-BEV heads at 0.5/1/2 s | Same rows; targets are the future frame's BEV *within the same recording and split* | future BEV IoU per horizon. **Skipped until Reality View is built and integrated.** |
| 5 | 3DGS | **new**, per `docs/specs/reality_view_3dgs.md` | Offline recorded frames + poses. GT only in tests named GT. | rendered-depth AbsRel, δ<1.25. **Skipped until Reality View is built and integrated.** |
| 6 | refine | Phase 9 refiner, a separate model. Not in SceneNet. | n/a here | **Skipped until Reality View is built and integrated.** |
| 7 | joint | All heads that exist, peak LR ×0.1 (**spec proposal**) | All of the above that have run | primary + no head worse than its stage exit by > 2% (**spec proposal**) |

## Loss, gradients, LR

- **Per-task loss**: exists. `train_scene` step trace carries `heads` per `scene_net.LOSS_TERMS` (`points, class, lamps, valid, lane_gt`), plus `_loss_terms` extras (`ctl, road_edge, stop_line, aux_depth, aux_sem`). Keep it; add BEV/temporal terms when those heads land.
- **Per-task grad norm**: **new**. Today only the global `clip_grad_norm_` value is logged (`train_scene.py` ~l.1675). Add ‖∂L_i/∂W_shared‖ per term, with W_shared = `stem.fc.weight` (cheap). Log it per step to the existing status/trace JSON.
- **GradNorm** (Chen et al. 2018), **new** `python/train/trainer2_balance.py`: learnable w_i, α = 1.5, renormalized so Σw_i = n_active, clamp w_i ∈ [0.1, 10] — α, the clamp and the weight step size `WEIGHT_LR` = 0.025 (`w_i ← w_i − WEIGHT_LR · ∂L_grad/∂w_i`, untuned) are a **spec proposal**. Grads are taken on `stem.fc.weight` only. Distill terms are excluded (fixed weights).
- **Gradual activation**: a head entering at step s₀ gets ramp(t) = min(1, (t−s₀)/R), R = 2000 steps (**spec proposal**). Its effective weight is ramp·w_i. Inactive heads compute no loss.
- **Warmup + cosine**: **new** `python/train/lr_schedule.py`. Per stage: linear warmup over max(500, 5%) of the stage's steps, then cosine to 0.05×peak; joint peak = 0.1× (**spec proposal**). It must compose with `_pin_optimizer_lr` (resume keeps the CLI rate × `lr_mult`).
- **Differential LR**: `param_groups(lr, mult)` exists on `PretrainedSceneNet` only (`pretrained_net.py:168`). Add the same on `SceneNet` (**new**). Groups: `stem` ×0.1 after stage 0, heads already trained ×0.3, the entering head ×1.0 (**spec proposal**). Reuse `lr_mult` and `--backbone-lr-mult` semantics.
- Grad clip: existing `clip`. Precision: **fp32**. `model_size.resolve_precision` already keeps Pascal (sm_61) on fp32, and `occ_sparse.py` notes Pascal fp16 math ≈1/64 of fp32.

## Data split and holdout

- Split by **recording + time block, never by frame**. Use the existing #187 machinery: `continual.pass_plan` / `strip_sources.combined_pass_plan` (`--holdout-blocks`), the `HOLDOUT_GAP_S` = 4 s purge, optional geographic cells (`HOLDOUT_CELL_M` 250 m + 50 m buffer, `--holdout-cells`).
- **new** `python/train/trainer2_split.py`: writes `split_manifest.json`, with sorted (source, i) per split plus sha256, **before** step 0. SSL, every stage, the teacher and the distill cache all read this one manifest. Temporal targets may not cross a split boundary. The manifest also names the **held-out map** for the 2.0 bar. That map is barred from every training root (SSL, teacher, and the distill cache included) **and** from the matched holdout used for stage exit, model selection, and the 25% RMS gate — **GVD call 2026-11-25** (interpreting Michael's "held out of all training"). This is not Michael's own words, so it is not labeled decided by Michael. **Open question, PROPOSAL pending Michael (do not read as decided):** whether that bar also covers the matched holdout. It is the named constant `BAR_HELDOUT_MAP_FROM_HOLDOUT` (`build_split_manifest(bar_heldout_from_holdout=...)`, default True), recorded and hashed in the manifest as `heldout_map_bars_holdout`; False bars the map from train only and leaves its rows in the holdout. Both settings are tested, so his ruling is a one-line change.
- `check_trainer2_split_no_overlap`: no (source, i) in both splits; within a recording, min |t_train − t_hold| ≥ 4 s; cells disjoint when on; temporal targets' source frames are in the same split; the SSL row set ⊆ train; no row from the held-out map is in train, SSL, teacher, the distill cache, **or** the matched holdout used for exit, model selection, and the 25% RMS gate.
  - Implementation notes (`trainer2_split.py`): cells-disjoint is enforced when `build_split_manifest(cells=<choose_holdout_cells spec>)` is given (every holdout row in a listed cell, every train row outside the listed cells and their buffer; buffer rows and rows with no position are purged and counted in `purged_buffer_rows`; `check_split_manifest` re-checks the per-row cells). A recording with holdout rows needs a finite `t` on every row, or the build is refused (the 4 s gap cannot be proven). `preregister(..., registry=...)` is required: a second seed for the same arm is refused in any run folder that shares the registry.
- **Baseline** for every comparison is the **previous shipped / best model** (a frozen checkpoint), not a re-train of today's recipe (**decided by Michael 2026-11-25**). Score with `scripts/eval_scene_ckpt.py` (`n, r, rms` + per-head lines) on the matched holdout. New heads without a baseline counterpart are reported "no baseline", not as wins. The baseline arm's run card must state whether that checkpoint was trained with SSL on or off.
- **Pre-registration** (`run_manifest.json`, **new**): git SHA, config sha256, seed, split sha256 (including the held-out map), SSL on/off, decision rule (which tier). Written before training. **One run per arm, no seed re-rolls.**

## Decision rules (two tiers)

Both tiers are fixed and pre-registered in `run_manifest.json` before the run (**spec proposal:** the pre-registration process). Runtime is camera-only. Live inference VRAM stays ≤ 4 GB (**decided by Michael 2026-11-25**). The two-tier split itself (a per-iteration merge bar, and a harder 2.0 naming bar) was locked at 8:29 PM CT, relayed by GVD.

**Per-iteration minimum (merge bar for each Trainer 2.0 iteration).** Anything that merges must beat the previous shipped/best model on the matched holdout. **Decided by Michael 2026-11-25:**
- lane RMS at least **10% better**,
- **0 missed** stop-sign / red-light brakes,
- false-brake rate (`ctl_brake` > 0.3 with no stop event within 40 m) **no worse**.

Also required on this bar, relayed in the 8:29 PM two-tier lock (**decided by Michael 2026-11-25**): the 0-missed cut includes the 22–37.5° `stop_side_angle` slice; green-light false-brake rate (`ctl_brake` > 0.3 on green when there is no other stop event) is **no worse**; every **labelable** scenario slice (below) has **no regression** vs the previous model.

This is the floor, not the 2.0 name. A miss on this bar does not ship in the Trainer 2.0 line; the result is logged as a loss and trainer2 stays opt-in for that iteration.

**2.0 naming bar.** The "2.0" name is earned **only** by **zero interventions on an unseen map**. Anything short of this ships as **1.x**. **Decided by Michael 2026-11-25:**
- a full **30-minute** drive on a map held out of **all** training (teacher, SSL, and the distill cache included; the map is recorded in `split_manifest.json`),
- **random AI traffic**, run in both **day and night**,
- **0 takeovers, 0 collisions, 0 missed stop signs / red lights** (full stop held ≥ 0.5 s), **0 curb strikes**,
- every scenario slice from the addenda also passes,
- lane RMS at least **25% better** than the current (previous shipped/best) model, measured on the **matched holdout** (not on the 30-minute drive),
- runtime camera-only, live VRAM ≤ 4 GB.

The 2.0 map is also barred from that matched holdout (so it cannot be used for stage exit, model selection, or the 25% RMS gate) — **GVD call 2026-11-25** (interpreting Michael's "held out of all training"). See Data split and holdout.

**Intervention (one definition, used everywhere "zero interventions" or a 0-count is scored on the 2.0 drive).** An intervention is any of: a takeover, a collision, a missed stop-sign or red-light stop (full stop held ≥ 0.5 s), a curb strike, **and** any AEB or hand-coded stop-rule activation during the 2.0 drive. "Zero interventions" means the safety backstop never fires. **GVD call 2026-11-25** (interpreting Michael's "zero interventions"). This is not Michael's own words, so it is not labeled decided by Michael.

**Open question, PROPOSAL pending Michael (do not read as decided):** whether an AEB or hand-coded stop-rule activation counts as an intervention. `python/train/trainer2_bar.py` counts it today (`BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS = True`, argument `count_backstop`); with False the activations are still tallied and reported (`backstop`) but not counted in `total`. Both settings are pinned by `check_backstop_config_proposal_pending_michael`, so his ruling is a one-line change.

**Scoring rules in `trainer2_bar.py` (implementation of the rules above, no new thresholds):** the scenario-slice set, each slice's direction (`higher_is_better` / `lower_is_better`) and its 2.0 limit are the **Slice rules table** under Scenario slices below, pinned in code as `SLICE_RULES`; a run card cannot choose them (a slice name, direction or stated limit that differs from the table raises). Iteration bar: every `labelable_today` slice in the table must be on both cards, and there is no regression in the slice's direction; a stop-score slice must include `missed` (not defaulted to 0). 2.0 bar: **every slice in the table must be on the card and pass its table limit** (a missing slice, a `None`/NaN value or a value past the limit fails). The four slices that cannot be labelled yet are required too (`REQUIRE_UNLABELABLE_SLICES_FOR_2_0`, **PROPOSAL pending Michael**: literal reading of "every scenario slice from the addenda also passes"). A `None` or non-finite lane RMS, false-brake rate, slice value, drive length or VRAM fails (never passes); `stop_events.score` returns `None` for a rate with no distance. An unknown drive-event kind raises.

The 30-minute live drive is run **on Desktop-IQU45 by GVD**. Swarm coders do not touch IQU45. It is blocked until the **2.0 bar blockers** in Dependencies to build exist.

## Learned stop braking (Michael, 6:37 PM; Q6 and lights decided 2026-11-25)

- **Goal:** the model learns stop-sign and traffic-light braking end to end from the camera strip. Planner-path AEB and the hand-coded rules above stay in place, unchanged, as the safety backstop.
- **Inference:** camera-only (the 8-cam strip; wide pixels via `compose_strip`, wide-derived lane points gated as live with `gate_wide=True`). **Sim / game GT** is used **only as a training label**: never an input, never on a live path. GT modules stay in the `test_occ_runtime_imports.py`-style isolation set.
- **Label sources that exist on main `98af813`:** the collector writes per-tick lamp/sign payloads to `signals.jsonl` (`scripts/collect_scene_lanes.py:3763` → `python/collect/recorder_v4.append_signal_tick` → `python/collect/signal_labels.read_map_node_signals`, BeamNG `traffic_signals.get_map_node_signals`). Offline, `scripts/export_signal_labels.py` joins that file onto `state.jsonl` by `t` and writes `signs.jsonl` as `{t, signs:[{cls,x,y,state,src}]}` or `{t, err}` (`signal_labels.labels_from_signals`, `LAMP_STATES` = `unknown, red, yellow, green`, `src` = `"traffic_signals"`). `python/train/intersection_labels.attach_sign_labels` copies `signs.jsonl` onto train rows; `attach_stop_lines` joins `stoplines.jsonl`. `docs/specs/labels_v4.md` (row "Traffic-light state") still says "Not captured / trainer does not read signs.jsonl"; that line is stale relative to `98af813`. **Dependency:** a recording is only lamp-labelable if `signals.jsonl` is present and `signs.jsonl` has been exported; older roots that lack those files contribute no light/sign GT (mask, do not invent). `unknown` lamp state is excluded from lamp CE and does not count as a red/yellow/green event.
- **Events (new, `python/train/stop_events.py`).** **Decided by Michael 2026-11-25:** a stop is scored only if the recorded ego comes to a **full stop** (`v < 0.5 m/s`) and **holds it ≥ 0.5 s** before the stop line + **1 m**; keep the **40 m**, **2 m/s**, and false-brake `ctl_brake` > **0.3** cuts; drop the earlier miss rule that required only reaching `v < 0.5 m/s` by the line + 1 m with no hold. Lights are in the learned-brake AC (red-light misses are on the 2.0 bar).
  - A **candidate approach** is a GT stop sign, a GT stop line, or a **red or yellow** light on the ego path within 40 m, with ego speed ≥ 2 m/s at first entry into that window. Treating a **yellow** light as a scored stop event (alongside red) is a **spec proposal**.
- **Metrics** (new `scripts/eval_stop_brake.py`, offline, recorded holdout; the same cuts apply on the 2.0 live drive):
  - **missed stop-sign brake / missed red-light stop:** the predicted (or live) brake profile does not produce a full stop held ≥ 0.5 s before the stop line + 1 m. **Must be 0** on the scored event set, including the `stop_side_angle` slice and red lights.
  - **Open question, PROPOSAL pending Michael (do not read as decided):** a red light that turns green during the approach, after which the car drives on. `python/train/stop_events.py` counts it as a **missed red-light stop** today (the event is typed by the lamp state at first entry; the literal spec). Whether it should not count is Michael's call; `RED_TO_GREEN_PROCEED_IS_MISS` (default `True`, today's behaviour) selects it: `False` drops a red/yellow approach whose lamp turned green before the line and that was not held (green only past the line is still a miss). `check_red_to_green_proceed_current_behaviour` pins the default and `check_red_to_green_flag_false_drops_the_approach` pins the other setting; both change together if he rules.
  - **Scoring contract (`stop_events.score`):** recorded `v` decides which approaches exist (2 m/s entry) and the holdout `km`; `speed_key` (default `v`) names the speed profile judged for the held stop, so a model's predicted profile is scored on the same events as every other. A row missing the GT target is skipped (a dropout over 1 s ends the window). A rate with no distance (`km` = 0) or a quantile with no held stop is `None`, never `NaN` or `0.0`; `trainer2_bar` fails on `None`.
  - **false brakes:** `ctl_brake` > 0.3 with no stop event within 40 m, counted per km. Must be no worse than the previous model.
  - **green-light false brakes:** `ctl_brake` > 0.3 on a GT **green** light when there is no other stop event. Must be no worse than the previous model.
  - **stop distance:** signed m from the stop line at the start of the held full stop, median and p95.
- **Known risk: 22–37.5° side-angle gap.** Stop signs 22–37.5° off-axis are covered only by `wide`. Without wide, GVD counted 7 missed stop-sign brakes. Wide stays. Near-range-only use is now on main via PR #251 (`WIDE_NEAR_MAX_Y_M` 9.0 m, `WIDE_NEAR_MAX_X_M` 5.0 m, 216/216 brakes, 0 missed). Far/side coverage there is whatever that gate admits. A camera-only learned brake cannot beat what its gated inputs see. The `stop_side_angle` slice (22–37.5°) stays, and 0 missed brakes is still required on it. The hand rule and AEB stay as the backstop.
- **Side-angle holdout slice:** a separate `stop_side_angle` slice covers events whose stop sign bearing from ego is 22–37.5° at first GT visibility. It is reported on its own, and 0 missed brakes is required on it too. It is never pooled away into the overall number.
- **Comparator (holdout):** the hand-coded rule replayed read-only (`predict_path` on the same recorded holdout frames' detections), on the same split manifest, metrics and events. One hashed run each, no re-rolls.
- **Retirement rule:** the hand-coded rule and AEB are **never** removed or weakened in any Trainer 2.0 PR. Retiring the hand-coded stop rule needs the per-iteration bar, then the 2.0 naming bar, then **Michael's explicit OK** and a separate PR. AEB (`cipv.aeb`, `plan.aeb`) is not part of that retirement and stays as the backstop.
- `ctl` and `stop_line` stay offline/not exported in Trainer 2.0. Using a learned brake live is a later, separately gated step (required for the 2.0 live drive to score a learned profile; until then the live drive scores the engaged stack that is actually running).

## Scenario slices (Michael, 8:27 PM; no-regression on the iteration bar, must pass on the 2.0 bar)

Each slice is scored separately on the matched holdout. Labels come from BeamNG game data and are used for **training only**. Runtime stays camera-only. Pedestrians stay out of scope. Where a criterion is vague, the measurable proxy is a **spec proposal**, not Michael's wording, and is marked as such.

Collision proxy used below (spec proposal): in the event window, no `tags.impact` / `impact_decel_mps2` on `state.jsonl` rows, no `impacts.jsonl` row with `kind=impact` (`scripts/collect_scene_lanes.py:84-120`; this log only sees a longitudinal deceleration faster than 15 m/s², and is not a complete crash list), and no rise in `ego.damage`.

| # | Slice (Michael) | Labelable today? | How / path | Proxy (if needed) |
|---|---|---|---|---|
| 1 | Lead vehicle braking / stopped ahead: 0 collisions, reacts to brake lights | **Yes**, with gaps | `state.jsonl` `tags.traffic` at 2 Hz: `pos, vel, dir, name, brake_lights` (`scripts/collect_scene_lanes.py:41-42, 3121, 3684`; `python/collect/recorder_v4.with_brake_lights`). `brake_lights` may be `null` when Electrics cannot attach. `ego.speed_mps`, `ego.damage`, `tags.impact`, `impacts.jsonl`. | Event: a traffic object with ego-forward y in (0, 40] m and \|x\| ≤ 2 m, ego ≥ 2 m/s, and (`brake_lights` truthy or horizontal \|vel\| ≈ 0). 0 collisions = the proxy above. **"Reacts to brake lights" (spec proposal):** when in-path lead `brake_lights` is on (non-null, truthy), `ctl_brake` > 0.3 within 1.0 s or ego speed falls ≥ 1 m/s within 2.0 s, and no collision. |
| 2 | Cut-ins: no collision; false-brake rate (`ctl_brake` > 0.3) no worse | **Yes**, clip-mode and derivable | Clip plans: `python/collect/clip_mode.EDGE_CASES` includes `cut_in`; `clip_tags.json` stores `edge_case` / `edge_case_observed` (lateral constants `CUT_IN_FROM_M=2.5`, `CUT_IN_TO_M=1.0`, `CUT_IN_AHEAD_MAX_M=30`). Same traffic tracks as slice 1 on every collect. | Event: `edge_case_observed` on a `cut_in` clip, or a traffic track that crosses \|x\| from > 2.5 m to < 1.0 m ahead within 30 m in a 6 s window (`CUT_IN_WINDOW_S`). 0 collisions = the proxy above. False-brake rate as in stop metrics, no worse than the previous model on this slice. |
| 3 | Cones / work zones: 0 cone contacts | **No** | `OBJECT_CLASSES` is `(vehicle, car, truck, bus, bike)` (`scene_net.py`). No cone/prop class. `docs/ROADMAP.md` (Cones / obstacles): collector does not record cones; YOLO has no cone class. | Cannot score. See Dependencies. |
| 4 | Yield signs: slow down and yield; full stop only when cross traffic is present | **Partial** | Yield class exists in `signal_labels.SIGN_CLASSES` and `signs.jsonl` (`cls=yield`). Cross traffic: `tags.traffic` pos/vel/dir. | Event: a yield on the ego path within 40 m, ego ≥ 2 m/s. **"Slow down" (spec proposal):** min ego speed in the approach ≤ 0.6× the speed at first sighting, before the sign + 1 m. **"Full stop only with cross traffic" (spec proposal):** a held full stop (v < 0.5 m/s for ≥ 0.5 s) is required iff a traffic object is within 20 m of the sign with a conflicting heading (\|dir_ego · dir_other\| < 0.5) during the approach; a held full stop without that traffic is a false stop. |
| 5 | Railroad crossings, on maps that have them | **No** | No railroad / railway / crossing field under `python/collect`, `python/data`, `python/train`, or `docs/specs`. | Cannot score. See Dependencies. |
| 6 | Sharp curves: speed reduced before curve entry | **Yes** (geometry from ego) | `state.jsonl` `ego.yaw_rate` and `ego.speed_mps` (`collect_scene_lanes.py:3709-3713`). Clip-mode observer already uses \|yaw_rate\|/speed ≥ `SHARP_CURVE_PER_M` (1/50 m) at speed ≥ `SHARP_CURVE_MIN_MPS` (5 m/s) (`clip_mode.py:77-79, 298-307`). | Event: a row meeting those two constants. **"Speed reduced before curve entry" (spec proposal):** peak speed in the 2 s before the first such row is ≥ 10% below the speed 4 s before that row. |
| 7 | Night / rain / fog, reported as separate slices | **Partial** | Clip-mode: `clip_tags.json` has `weather`, `time_of_day_h`, `edge_case` (`clip_mode.write_tags`). Night can also be checked from mean main luma ≤ `NIGHT_MAX_LUMA` (60) (`clip_mode.py:80, 314-323`). Rain/fog: the collector sends `core_weather.setWeather` (`python/collect/auto_mode.environment_commands`); `_observe` says rain/fog are **not observable from the recording offline**. Non-clip runs: weather/tod are only in `collect.json` `argv` when `--weather` / `--time-of-day` were passed, not a per-row field in `state.jsonl`. | Slice membership: `clip_tags` when present; else `argv` if those flags were set; night also by luma. Report each of night, rain, fog on its own. Metrics = the iteration/2.0 cuts restricted to that slice. |
| 8 | Safety: heartbeat / AEB / disengage tests pass, and live inference VRAM ≤ 4 GB | **Yes** (tests and caps, not a GT slice) | Existing offline checks around heartbeat / AEB / engage (suite). Vision cap: `config/hardware.yaml` `max_inference_vram_gb: 4`. `check_student_export_fits_deploy` / `check_safety_path_no_trainer_imports`. | The suite and the deploy VRAM check must stay green. Not a holdout-label slice. |

### Slice rules table (pinned in `python/train/trainer2_bar.SLICE_RULES`)

One number per slice on the run card. **Basis `spec`** = the criterion and number are in the table above (0 collisions, 0 cone contacts, false-brake rate no worse than the previous model). **Basis `PROPOSAL`** = the spec states only a measurable proxy and no number; the limit is the strictest reading (1.0 = every event handled correctly) and waits for Michael. The key names are the scorer's own; the spec numbers rows 1-8. Row 8 (safety tests and the 4 GB VRAM cap) is not a holdout slice: the suite and the drive's `live_vram_gb` gate it.

| Slice key | Row | Direction | 2.0 limit | Basis | Labelable today |
|---|---|---|---|---|---|
| `lead_vehicle_collisions` | 1 | lower_is_better | 0 | spec | yes |
| `lead_brake_light_reaction` | 1 | higher_is_better | 1.0 | PROPOSAL | yes |
| `cut_in_collisions` | 2 | lower_is_better | 0 | spec | yes |
| `cut_in_false_brakes_per_km` | 2 | lower_is_better | no_regression | spec | yes |
| `cone_contacts` | 3 | lower_is_better | 0 | spec | no |
| `yield` | 4 | higher_is_better | 1.0 | PROPOSAL | yes |
| `railroad_crossing` | 5 | higher_is_better | 1.0 | PROPOSAL | no |
| `sharp_curve` | 6 | higher_is_better | 1.0 | PROPOSAL | yes |
| `night` | 7 | higher_is_better | 1.0 | PROPOSAL | yes |
| `rain` | 7 | higher_is_better | 1.0 | PROPOSAL | no |
| `fog` | 7 | higher_is_better | 1.0 | PROPOSAL | no |

`lead_brake_light_reaction`, `yield` and `sharp_curve` are the share of that slice's events that meet the row's proxy; `night`, `rain` and `fog` are 1.0 when every iteration/2.0 cut restricted to that slice passes, else 0.0.

## Fail-and-learn closed-loop data (Michael, 8:39 PM)

The model drives in-game through the **existing** heartbeat / AEB / engage path. This mode does not touch the safety path. **Dependency:** it needs a model good enough to drive (a passing per-iteration student, engaged under `--scene-ckpt` after the 8-cam switch). Runtime stays camera-only. RL is out of scope for v1.

**Cameras are already pulled on live ticks.** `python/run_vision.py` grabs the rig every tick (`CapturePump` at l.1255 / `backend.grab()` at l.1299) and stitches it with `stitch_frames(bundle_tick_frames(bundle))` (l.1436; the comment at l.1435 is "eight cameras, one timestamp"). The live failure recorder is `python.data.record.ClipRecorder`, constructed in `run_vision.py:1024` and fed at l.2309 via `recorder.push(main, st, ...)`. On `98af813` (and still on `3f1fab1`) that ring stores **cam_main JPEG only** plus a slim state dict (engaged, disengage_reason, planner, ego, …), not the lossless 8-cam `compose_strip` inputs the trainer uses. Defaults: `DEFAULT_PRE_S = 4.0`, `DEFAULT_POST_S = 2.0`, in-memory `RING_CAP_BYTES = 2 GiB` (`python/data/record.py`). CLI: `--clip-pre-s`, `--clip-post-s` (0 keeps those defaults). The AI-driven collector (`scripts/collect_scene_lanes.py`) already writes lossless 8-cam raw PNGs, but that is not the engaged-model path. **Fail-and-learn must extend the live ring (or hook the collect writer) to save the same 8-cam lossless inputs the trainer consumes**, with a 30 s window, about **20 s before and 10 s after** the trigger (**spec proposal:** the 20+10 s window). That window and the 8-cam lossless flush are **new**; today's 4+2 s cam_main JPEG ring is the starting hook, not the training artifact.

**Triggers** (flush the ring). Today, `ClipRecorder` actually requests clips for:
- **disengage:** `ClipRecorder.note_engage` (l.189) on a falling engage edge, and `python/runtime/shadow.py` sets `clip_trigger="disengage"`; `run_vision.py:2310–2314` calls `recorder.note_engage` / `recorder.request`.
- **AEB / near-miss:** `ClipRecorder.check_aeb` (`aeb_brake`, `near_miss_ttc`; l.196–217), requested at `run_vision.py:2315–2317`.
- **manual.**
A `launch_guard` damage jump (`python/control/launch_guard.py`; `launch_guard.update` at `run_vision.py:1891–1898`, crash disengage at l.1925–1937) disengages (`crash_reason`) and should also fire `note_engage` on that tick, but the clip is tagged `"disengage"`, not a dedicated collision trigger. **Not present as clip triggers on main** (add to Dependencies): collision/damage as its own tag, lane-line crossing (debug-only today: `python/debug/gt_logger.py` `departed` / `off_road`, not wired to `ClipRecorder`), red-light or stop-sign run, hard false brake, curb strike (see dep 7).

**Relabel, DAgger-style.** Expert actions at the same frames come from BeamNG's AI / navgraph and are **training labels only**. Existing collect already stores `wheel.steer_ai`, `pedals.throttle_ai`, `pedals.brake_ai` when BeamNG AI is driving (`python/collect/control_log.py`, consumed by `python/train/ctl_targets.py` `CTL_FIELDS`). While the model is engaged, live actuation **disables** AI (`python/control/actuate.py:1023` `ai.setMode('disabled')`), so those fields are not a live expert. **New hook (dependency):** query the expert command from BeamNG AI / navgraph at the recorded poses **without applying it**, write it onto the clip as `steer_ai` / `throttle_ai` / `brake_ai` (same field names as collect; do not invent new ones), and tag the frames as expert-relabeled. **NEVER train on the model's own bad actions.** AC: `check_fail_learn_no_model_ctl_targets` — no `ctl_*` target is taken from a model-driven frame; only expert-relabeled `steer_ai` / `throttle_ai` / `brake_ai` (or a train-only collector run where BeamNG AI was the driver). A clip that lacks the expert sidecar is masked, not trained.

**Upweight** these clips in the next run. **Spec proposal (not Michael's number):** loss weight 3.0 relative to ordinary train rows, configurable. The weight must respect `split_manifest.json`: no fail-and-learn clip from a holdout recording, a held-out map, or the unseen 2.0 map is admitted to train, SSL, the teacher, or the distill cache (`check_fail_learn_respects_split`).

**Disk cap.** The live ring cap (2 GiB) is not a dataset cap. **Spec proposal (not Michael's number):** keep at most 50 GB of fail-and-learn clips on disk (configurable). Oldest clips evict first, except that a clip must not be deleted if it is named in an already-hashed `run_manifest.json` (old checkpoints / manifests stay; the trainer still never deletes weights).

## Dependencies to build

These are not invented labels. They block scoring until they exist. The **2.0 bar blockers** are Dependencies **5–8** (held-out map, takeover event logger, curb-strike event log, and the IQU45 day/night harness). The 2.0 naming drive is not run until those four exist. Dependencies 1–4 block scoring of the slices that need them, not the drive itself. Dependencies 9–11 block fail-and-learn, not the 2.0 name. Appendix C's 2.0 naming-bar step points at this same list.

1. **`signs.jsonl` export on every training root that should contribute sign/lamp GT.** The recorder already writes `signals.jsonl`. The trainer only sees lamps after `scripts/export_signal_labels.py`. Roots without both files are masked.
2. **Cone / work-zone labels.** No field today. Need a BeamNG-backed, train-only sidecar (class + ego xy, or contacts) before slice 3 can score. Do not invent a field name here.
3. **Railroad-crossing labels.** No field today. Need a train-only sidecar on maps that actually have crossings, before slice 5 can score.
4. **Weather / time-of-day on every collect, not only clip-mode tags.** Needed to slice non-clip recordings for night/rain/fog. Rain/fog still have no offline pixel check.
5. **Held-out map for the 2.0 bar** (a **2.0 bar blocker**). There is **no recording inventory in this repo**. Collector-selectable maps on main: `python/collect/auto_mode.MAPS` / `DEFAULT_MAPS` = `west_coast_usa`, `italy`, `east_coast_usa`, plus `utah`, `jungle_rock_island`; clip-mode `DIRT_MAPS` also lists `small_island`. README / `docs/ROADMAP.md` treat shipped models as mostly WCUSA, with ~4 h recorded (not re-measured here). GVD must name a map that has **zero** hours in every training root, write it into `split_manifest.json`, and keep it out of SSL / teacher / distill **and** out of the matched holdout used for exit, model selection, and the 25% RMS gate (the GVD call in Data split and holdout). Swarm does not invent the name.
6. **Takeover event logger for a scored live drive** (a **2.0 bar blocker**). Live already writes `disengage_reason` via `write_engage_flag` (`python/run_vision.py:1497, 1517, 1737, 1762, 1801, 1932, 1948, 2496`), `should_disengage` (`python/runtime/shadow.py`), and wheel-override detection (`python/control/override.py`). There is **no per-drive takeover counter or event sidecar** to score "0 takeovers" over 30 minutes. One must exist before the 2.0 run; this spec does not name a path or event that does not exist.
7. **Curb-strike event log** (a **2.0 bar blocker**). `ego.damage` and `impacts.jsonl` are not curb-specific. `tech["curbs"]` is unfilled (`docs/specs/labels_v4.md`). `pack_road_edge` is a training label, not a contact event. Need a train-and-live curb-strike logger before "0 curb strikes" can be scored. This spec does not name a path that does not exist.
8. **Random-traffic + day/night 30-minute drive harness** (a **2.0 bar blocker**), run on **Desktop-IQU45 by GVD** (swarm does not touch IQU45). It must load the held-out map, enable random AI traffic, run day and night, and tally every **intervention** as defined under Decision rules (takeovers, collisions, missed stop signs/red lights held ≥ 0.5 s, curb strikes, and AEB / hand-rule activations). No such harness exists on `3f1fab1`.
9. **8-cam lossless live ring for fail-and-learn.** Extend `python.data.record.ClipRecorder` (or hook the collect writer) so a trigger saves the same lossless 8-cam inputs as `compose_strip`, with ~20 s pre and ~10 s post (**spec proposal:** the 20+10 s window; today: cam_main JPEG, 4+2 s).
10. **Missing fail-and-learn clip triggers**, wired to that ring without changing heartbeat / AEB / engage behavior: a dedicated collision/damage tag (today a damage jump only disengages via `launch_guard` and may flush as `"disengage"`); lane-line crossing (the `departed` / `off_road` logic in `python/debug/gt_logger.py` is debug-only); red-light or stop-sign run; hard false brake; curb strike (dep 7).
11. **Expert-relabel hook while the model is engaged.** Query BeamNG AI / navgraph at the clip's poses without applying the command (`ai.setMode('disabled')` stays in effect on the car). Write `steer_ai` / `throttle_ai` / `brake_ai` onto the clip. No such shadow-expert path exists on `98af813`.

## VRAM budgets (two)

Canvas, 8 cams at scale 1 (`strip_writer`: 6×256 + 2×340 (`wide`, `rear`) + 7×8 gap = **2272×192**). Stem = 4 stride-2 3×3 convs → 96×1136, 48×568, 24×284, 12×142. Activations per frame ≈ Σ(C·H·W) × 3 stored tensors (conv/GN/ReLU) × 4 B.

**Train budget: full GTX 1080 Ti 11 GB, game off** (Michael, 6:07 PM). Target ≤ 10 GB allocated, leaving ~1 GB for context and fragmentation (est.).
- Student `small` (32,64,128,128): (96·1136·32 + 48·568·64 + 24·284·128 + 12·142·128) = 6.33 M floats ×3 ×4 B ≈ **76 MB/frame**, ×T=16 ≈ **1.21 GB/window** without `--grad-ckpt`. With `--grad-ckpt` (exists, scratch net only), ≈ one tick (~76 MB) + tokens. Weights+grad+AdamW: 2,195,486 params (`architecture_parameter_count()`, 8-cam) ×16 B ≈ 35 MB. Batch: `--batch 0` auto-fit (`resolve_scene_batch` / `_probe_cuda`, exists) → est. ~7 windows without ckpt.
- AMP/fp16: off on sm_61 (see above). It is a memory option only, and only if a run is VRAM-bound, which it isn't at these sizes. Grad accumulation: **new** `--accum N`, used only if the auto-fit batch < 4 windows (**spec proposal**).

**Deploy budget: live student ≤ 4 GB inference** (`config/hardware.yaml` `max_inference_vram_gb: 4`), ~640 px long side, ONNX FP32 on Pascal, preset `tiny`/`small` only (**decided by Michael 2026-11-25**, student stays ≤ 4 GB live). `small` paper estimate is 2.2 GB (`docs/research/gpu_tiers.md`, est., not a profile). Live `SceneCkptRuntime` is on CPU, est. 34–170 MB for the 8-cam canvas (same doc). Exported student ≈ 2.2 M × 4 B ≈ 9 MB of weights (est.). Supervisor RSS ≤ 12 GB unchanged. `check_student_export_fits_deploy`: the exported ONNX's param count equals the student's `architecture_parameter_count`, its preset ∈ {tiny, small}, and `PRESETS[preset].est_vram_gb` ≤ 4.0. A measured IQU45 VRAM reading is a GVD-scheduled gate, and it skips on the box.

## Distillation (optional, last training PR)

- **Teacher = `xl`, but first enlarge the `xl` preset to the biggest that fits on the 1080 Ti 11 GB for training** (fp32, Pascal, no tensor cores), and **measure it on Desktop-IQU45 before any distill training** (**decided by Michael 2026-11-25**). That enlargement + measurement is its **own early PR/step**. The measurement AC (peak VRAM, batch, window, step time, and headroom margin) is run **by GVD on IQU45**. Swarm coders do not touch IQU45. Until that measurement lands, every VRAM figure in this spec stays an **estimate**. The deployed student stays ≤ 4 GB live.
- **Current `xl` estimate** (`model_size.PRESETS["xl"]` today: stem (96,160,256,256), GRU 1024), est. only: **≈7.1 M** params (same formula as `architecture_parameter_count`), 3.2× student. Weights+grad+AdamW ≈ 114 MB. Activations: (96·1136·96 + 48·568·160 + 24·284·256 + 12·142·256) = 17.0 M floats ×3 ×4 B ≈ **204 MB/frame**, ≈ **3.27 GB/window** at T=16. Without ckpt, batch 2 ≈ 6.5 GB (batch 3 ≈ 9.8 GB is too tight for the 10 GB target). With `--grad-ckpt`, batch ≥ 16 on activations. FP32 only. These numbers do not replace the IQU45 measurement, and the enlarged preset may differ.
- **Losses** (student total = GT task losses, GradNorm-weighted, + fixed KD terms). λ, T, and the KD schedule below are a **spec proposal**:
  - L_feat: MSE on the 64-d sector tokens. `SECTOR_FEAT` = 64 for every preset, so no adapter is needed. λ = 1.0.
  - L_logit: smooth-L1 to the teacher's lane/curb points where teacher valid > 0.5. KL at T = 2 (×T²) on class/lamp heads. λ = 0.5.
  - L_depth: L1 on log `aux_depth` vs the teacher's. The GT depth term stays. λ = 0.5.
  - L_bev: per-cell KL **only on cells observed in GT**. Unknown/occluded cells are never pulled toward the teacher's "free". λ = 0.5.
  - Schedule: KD ramps 0→1 over the first 10% of student steps, then holds, then falls linearly to 0.25 during the joint stage, so GT dominates at the end.
  - Teacher frozen, eval mode. Its outputs are cached per **train** row (same rule as `exp_distill`: never holdout, never the held-out map).
- **Evidence bar**: the teacher must first beat the no-KD student on the holdout, or KD is not tried. Then `student_kd` vs `student_nokd`: same arch, manifest, seed and config hash, differing only by `--kd`, one hashed run each, no re-rolls. Keep KD iff `student_kd` meets the **per-iteration** bar against `student_nokd` (and no head is worse by > 2% — **spec proposal**). **Otherwise distillation is dropped.** The 2.0 naming bar is a later IQU45 drive, not a reason to keep a losing KD.
- **Isolation**: `check_teacher_not_on_live_path` (AST + fresh-interpreter walk from the live entry points, in the same style as `scripts/test_occ_runtime_imports.py`): `trainer2_distill` and teacher weights are unreachable. `check_onnx_student_only`: the exported graph's initializers match the student exactly, with no teacher-named or teacher-sized tensors.

## Tests (each written first, failing on `98af813`)

All live in `scripts/test_trainer2_*.py`. Torch parts use `gvd_test_env.skip_check` locally (the box has no torch) and run in CI (`.github/workflows/tests.yml` installs torch CPU).
- `check_recipe_off_is_bit_identical`: with `--no-ssl-pretrain` and no `--recipe`, argparse defaults and a 2-step CPU supervised trace match `98af813`. (Default `train_scene` is no longer bit-identical, because SSL is default ON.)
- `check_ssl_default_on`: a bare `train_scene` parse has SSL on; `--no-ssl-pretrain` turns it off.
- `check_wide_gate_same_as_live`: the trainer calls `_estimate_span_lanes(..., gate_wide=True)` (or `estimate_lanes` / `lanes_from_view`) and imports `WIDE_NEAR_MAX_Y_M` / `WIDE_NEAR_MAX_X_M`; AST/grep forbids redefining those constants or hard-coding 9.0 / 5.0 as a wide gate; a fixture's wide points match `gate_wide=True` and differ from `gate_wide=False`.
- `check_safety_path_no_trainer_imports`: an AST walk from the safety modules listed in Non-goals never reaches `python.train.*`.
- `check_train_live_input_identical` (hash pin, including the post-switch `--scene-ckpt` builder; a sibling check is allowed), `check_strip_order_matches_live`, `check_scene_ckpt_is_8cam` (fails on `_main_canvas` as it is on `98af813`).
- `check_stop_labels_train_only` (stop / lamp GT and `brake_ai` are never an input; GT modules are unreachable from live entry points), `check_stop_events_split_safe` (events are counted in holdout rows only), `check_stop_event_requires_hold` (an approach that dips below 0.5 m/s but does not hold 0.5 s is not a scored event; a miss is a failure to produce that hold), `check_stop_side_angle_slice` (a 30° fixture event lands in `stop_side_angle` and is reported separately), `check_light_events_red_yellow_only` (green and `unknown` are not stop events; green + `ctl_brake` > 0.3 with no other event is a green-light false brake), `check_stop_metrics_on_fixture` (a hand-built clip with known misses, false brakes, and a known stop distance gives exactly those numbers), `check_hand_rule_untouched` (sha256 pin of `path_predictor.py`, `pipeline.py`, `actuate.py`, `cipv.py` at the base SHA; any Trainer 2.0 PR that changes them fails).
- `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`.
- PR 5 follow-ups after review (Skeptic 2): `check_main_passes_ssl_stem` (main hands `train_directory` the SSL stem when on, `None` or the user's `--init-stem` when off), `check_ssl_loss_target_is_ema_copy` (the loss gets the frozen EMA copy; it follows the 0.996 recurrence), `check_ssl_empty_split_refuses_when_trainer_trains` (SSL has 0 train rows but the trainer plan has some: exit 2 naming `--no-ssl-pretrain`; both 0: the trainer reports), `check_ssl_recorded_in_stats_and_ckpt` (`stats["ssl"]` and every epoch ckpt `meta["ssl"]`: on/why/key/rows/reused), `check_ssl_oom_exits_2_and_releases_cuda`, `check_ssl_stem_key_tracks_recipe_and_content` (stem key includes `SSL_RECIPE` and a per-source frame-content hash; bump `SSL_RECIPE` when the pretext changes).
- `check_curriculum_order_and_gates`: stage order is fixed; stages 3–6 are skipped and logged until Reality View is built and integrated; the non-regression gate rejects a checkpoint.
- `check_head_ramp_linear`, `check_gradnorm_weights_sum_and_clamp`, `check_per_task_grad_norm_logged`.
- `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`.
- `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll` (a second run with the same manifest and a different seed is refused), `check_heldout_map_excluded` (the named held-out map is absent from train, SSL, teacher, the distill cache, **and** the matched holdout used for exit, model selection, and the 25% RMS gate).
- `check_iteration_bar_thresholds`, `check_2_0_bar_not_claimed_without_iqu45_drive`.
- `check_student_export_fits_deploy`, `check_old_checkpoints_untouched` (the trainer never unlinks an existing ckpt; only its own `.partial` files).
- `check_fail_learn_no_model_ctl_targets` (no `ctl_*` target from model-driven frames; only expert-relabeled `*_ai` or AI-driven collect), `check_fail_learn_respects_split` (no clip from holdout / held-out map / 2.0 map), `check_fail_learn_window` (a fixture trigger keeps ~20 s before and ~10 s after; window is a **spec proposal**).
- Late PR: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`.
- Slice checks land only for slices that are labelable; cone and railroad checks wait on their sidecars.

## Appendix A: file touch list

| File | Change |
|---|---|
| `docs/specs/trainer_2.md` | this spec (PR 0) |
| `python/train/train_scene.py` | **edit**: SSL default ON, `--no-ssl-pretrain`, `--recipe trainer2`, `--stage`, `--accum`, `--kd`; schedule/balance hooks in the step loop |
| `python/train/scene_net.py` | **edit**: `SceneNet.param_groups`; per-term grad-norm helper |
| `python/train/exp_ssl.py` | **edit**: EMA target, collapse guard; default-on wiring |
| `python/train/scene_ckpt_runtime.py` | **edit**: `--scene-ckpt` uses full 8-cam `compose_strip` (PR ahead of live heads) |
| `python/train/lr_schedule.py` | **new** |
| `python/train/trainer2_balance.py` | **new**: GradNorm, ramps |
| `python/train/trainer2_recipe.py` | **new**: stage table, gates |
| `python/train/trainer2_split.py` | **new**: manifest, hashes, held-out map |
| `python/train/model_size.py` | **edit** (early GVD PR): enlarge `xl` to the measured 11 GB fit |
| `python/train/trainer2_distill.py` | **new** (late PR, after the xl measurement) |
| `python/train/strip_tensor.py` | **new**: single `strip_to_nchw` (moved from `scene_lane_runtime._strip_nchw`, behavior identical) |
| `python/train/scene_lane_runtime.py`, `python/train/scene_net.py` | **edit**: both `_strip_nchw` call `strip_to_nchw` (no behavior change, pinned by hash + `test_scene_strip_scale.py`) |
| `python/train/stop_events.py`, `scripts/eval_stop_brake.py` | **new**: offline stop / light events + metrics + labelable slices |
| `python/data/record.py`, `python/run_vision.py` | **edit** (fail-and-learn PR): 8-cam lossless ring, 20+10 s window (**spec proposal**), new triggers; do not change heartbeat / AEB / engage |
| `python/train/fail_learn.py`, `scripts/relabel_fail_clips.py` | **new**: expert-relabel join, upweight, disk cap, split-manifest filter |
| `scripts/test_trainer2_*.py` | **new** tests |
| `python/planning/*`, `python/perception/pipeline.py`, `python/perception/cipv.py`, `python/control/*` | **not touched** (pinned) |

## Appendix B: verify steps (offline, box CPU)

```
ps aux | grep -c "[r]un_offline_suite"             # must be 0 before starting
python3 scripts/test_trainer2_split.py             # per-PR file, fail-first then pass
python3 scripts/test_trainer2_input_parity.py      # train vs live vs --scene-ckpt tensor hash pin
python3 scripts/eval_stop_brake.py --ckpt <student.npz> --manifest <split_manifest.json>  # learned vs hand rule, offline, incl. lights and labelable slices
git diff --name-only origin/main...HEAD | grep -E '^python/(planning|control)/|perception/(pipeline|cipv)\.py' && echo FAIL-safety-touched
python3 scripts/run_offline_suite.py               # full suite; paste the final line
```
CI (torch CPU) runs the same suite on the draft PR. Skeptic re-runs it in `/workspace/skeptic`.
The 2.0 naming-bar 30-minute drive and the xl-fit VRAM measurement are IQU45-only, scheduled and run by GVD, not by this suite.

## Appendix C: PR sequence (each separately gateable)

1. **PR 0**: this spec.
2. **PR 1**: split manifest + pre-registration (`trainer2_split.py`), including the held-out-map slot. Tests: `check_trainer2_split_no_overlap`, `check_manifest_hashed_before_train`, `check_no_seed_reroll`, `check_heldout_map_excluded`.
3. **PR #251 has merged** (`ac5ab3f`). No longer a blocker. Wide-gate parity is part of PR 2.
4. **PR 2**: wide-gate parity + safety-import check. Tests: `check_wide_gate_same_as_live`, `check_safety_path_no_trainer_imports`.
5. **PR 2b**: `--scene-ckpt` switches to the full 8-cam `compose_strip`. Tests: `check_scene_ckpt_is_8cam`, and fold that path into `check_train_live_input_identical` (or a sibling). **This PR lands before any Trainer 2.0 head goes live.**
6. **PR 2c**: one input builder. `strip_to_nchw` plus the train/live hash pin. Tests: `check_train_live_input_identical`, `check_strip_order_matches_live`. No other live behavior change.
7. **PR 3**: LR schedule + `SceneNet.param_groups`. Tests: `check_warmup_cosine_shape`, `check_param_groups_lr_mult`, `check_resume_keeps_schedule`, `check_recipe_off_is_bit_identical` (with `--no-ssl-pretrain`).
8. **PR 4**: per-task grad norms, GradNorm, ramps. Tests: `check_per_task_grad_norm_logged`, `check_gradnorm_weights_sum_and_clamp`, `check_head_ramp_linear`.
9. **PR 5**: SSL default ON for every `train_scene` run, plus `--no-ssl-pretrain`, EMA and the collapse guard. Tests: `check_ssl_default_on`, `check_ssl_ema_no_collapse`, `check_ssl_rows_train_only`. This is the behavior-change PR; call it out in the PR body.
10. **PR 5b (early, GVD on IQU45):** enlarge the `xl` preset to the biggest that fits 11 GB fp32 on the 1080 Ti, and post the measurement AC (peak VRAM, batch, window, step time, headroom). Swarm does not touch IQU45. Required before any distill training. Estimates in this spec stay estimates until this lands.
11. **PR 6**: curriculum runner (stages 0–2 / 2b live; 3–6 skip-logged until Reality View is built and integrated) + deploy check + ckpt retention. Tests: `check_curriculum_order_and_gates`, `check_student_export_fits_deploy`, `check_old_checkpoints_untouched`.
12. **PR 6b**: learned stop-sign and traffic-light braking, stage 2b, plus labelable scenario-slice scoring. Tests: `check_stop_labels_train_only`, `check_stop_events_split_safe`, `check_stop_event_requires_hold`, `check_stop_side_angle_slice`, `check_light_events_red_yellow_only`, `check_stop_metrics_on_fixture`, `check_hand_rule_untouched`. The hand rule and AEB are not changed. Cone and railroad slices wait on their sidecars.
13. **PR 7**: matched-holdout **per-iteration** run (GVD schedules the GPU): previous shipped/best checkpoint vs one trainer2 run, manifest hashes posted before training, SSL on/off stated, plus the stop/light table and labelable slices. Evidence only, no code. This is the merge bar, not the 2.0 name.
14. **PR 7b (after PR 6b and PR 7):** fail-and-learn closed-loop data. 8-cam lossless ring at 20+10 s, missing triggers, expert-relabel hook, upweight, disk cap, split-manifest filter. Tests: `check_fail_learn_no_model_ctl_targets`, `check_fail_learn_respects_split`, `check_fail_learn_window`. Safety path is not touched; the model drives through existing heartbeat / AEB / engage. Needs a model good enough to drive. GVD schedules any IQU45 driving.
15. **PR 8 (optional, after PR 5b and PR 7):** teacher→student KD on the measured xl teacher. Tests: `check_teacher_not_on_live_path`, `check_onnx_student_only`, `check_kd_bev_masks_unknown`, `check_kd_cache_train_only`. Then one `student_kd` run and one `student_nokd` run. KD is dropped if it does not win the per-iteration bar.
16. BEV / temporal / 3DGS / refine stages get enabled in the PR that lands each head, **only after Reality View is built and integrated**, each with its own `check_*`.
17. **2.0 naming bar (GVD on IQU45, not a Swarm coding PR):** 30-minute day and night drives on the held-out map with random AI traffic, after the **2.0 bar blockers** (Dependencies 5–8 in Dependencies to build) exist. Zero interventions as defined under Decision rules; every slice passing; lane RMS ≥ 25% better on the **matched holdout** (not on the drive). Anything short ships as 1.x.

