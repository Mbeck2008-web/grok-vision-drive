# GVD roadmap: mostly-works features, then shipped baseline models

Status as of **Monday 9 November 2026, evening America/Chicago**.
Audit base: `origin/main` at `72b58543941b4e3b68406fc9e12fcb0936594f2f` (merge of #194).
This is a plan, not a results report. Live gates stay **UNPROVEN** unless a log or clip on the dev PC shows them.

Grok Vision Drive is a camera-only drive-assist toy for BeamNG.drive / BeamNG.tech: a Python vision supervisor plus a BeamNG mod. It is entertainment software, not a real-car controller, and not Tesla FSD.

## Goal

Get every in-game driving feature to a **mostly works** level. That includes intersections. Pedestrians are out of scope because stock BeamNG traffic on the maps GVD uses is treated as vehicle-only (no pedestrian objects to learn from). After features mostly work, rent a cloud GPU (a suggested option: a RunPod 3090 at about $0.22/hr) and train every model-size preset (tiny / small / medium / large / xl, one per GPU tier). Ship those checkpoints with GVD as baselines so people can try the stock models and then compare models they train on their own data. The visual target is a stage that feels close to Tesla FSD Beta, drawn only from cameras and the model.

## Hard rules

- Deploy is camera-only. BeamNG ground truth and sensors (depth, semantic, radar, LiDAR, map geometry, `traffic_signals`, navgraph) are for **training labels, grading, and debug** only.
- The visualizer draws only what the cameras or the model detected. Never GT, map, or sensor overlays on the live stage, except `--debug-gt`, which stays off by default.
- GPS is allowed only for turn-by-turn route directions (`turn`, `next_jct_m`). It is not localization and not a path.
- Never use `--ctl-cond`. Never use fp16 on Pascal (the GTX 1080 Ti, compute 6.1).
- Host is IQU45: i9-9900K, 1080 Ti 11 GB shared with BeamNG, vision inference ≤ 4 GB VRAM, supervisor RSS < 12 GB. One GPU job at a time. Never kill a live BeamNG session.
- Heartbeat, AEB, and engage safety stay intact. Live drives only with Michael present.
- Road edges are their own capability. Never invent lane lines or a default 3.5 m lane when a side is missing.
- Every merge is gated by Skeptic: score > 8.5, GREEN, empty Fix list (including nits).
- Never invent results.

## How to read status

| Label | Meaning |
| --- | --- |
| live-proven | A log or clip from IQU45 shows it working. |
| offline-tested | Code is on `main` (or a named PR) and the offline suite / unit tests exercise it. Not a live proof. |
| UNPROVEN | Code may exist; nothing on file proves it in a live game. |
| missing | Not built, or built only as a stub / planned head with no labels. |
| in flight | Another agent or machine already owns it. Do not duplicate. |
| out of scope | Explicitly excluded (pedestrians; parking as a mode, per `docs/SPEC_intersections.md`). |

A feature can be offline-tested and still live UNPROVEN. That is the normal state of almost every drive capability today.

README (`What is proven`) is stricter and still correct: the only live-proven path is the command-line collector recording while BeamNG's AI drove (rec2 / rec3, 19 October 2026). Everything about GVD driving the car is UNPROVEN.

## Work already in flight

Do not start a second copy of these. Slot new work behind them.

| Work | Where | Role in this plan |
| --- | --- | --- |
| Detect heads: curbs / road edges, oriented vehicles, lead brake lamp | PR #199 (`cursor/train-detect-heads-dcc6`, head `ce14621` after merge of `origin/main` at `72b5854`) | Perception for edges, other cars, brake lamps. Cone boxes and lead-lamp GT stay masked until the collector records them. Awaits merge with `main` and Skeptic. |
| Live frame lag | PR #201 (`gvd/frame-staleness`, head `b289212`). Shadow run: `frame_age` p90 11.5 s at 3.1 Hz. | Blocks any honest live drive. Merge and prove on IQU45 before engage gates. BeamNGpy may be patched inside this repo only if evidence shows it is part of the lag. |
| GPU-aware model-size presets | Cloud agent on `gvd/gpu-aware-model-size` (branch not on `origin` as of this audit). Research note pending (Cloud research task, not yet in repo). | Size by card speed and AI hardware (INT8 / DP4A on the 1080 Ti; tensor cores on 20-series+; never fp16 on Pascal), not by VRAM first. Reserve game GPU/CPU/RAM, then fit the model in what is left; step down one size if the game's frame rate or free VRAM drops. |
| A/B train suite | IQU45, estimated to run through about 5:00 AM America/Chicago on 10 November 2026 (unverified estimate; no job log cited in this audit) | baseline / W1 `--exp-cuda-graph` / W5 `--exp-active-subset`. Do not start another GPU job. |
| ep11 re-score on the #187 holdout | Queued after the A/B suite | Same holdout (`--holdout-blocks 5 --mask-events`). Do not invent a number before the log exists. |
| Cache copy to C: SSD | Queued after A/B and the re-score | Raw data (an unverified estimate of ~74 GB from project notes, not a dir listing in this audit) does not fit on C:; only the frame cache moves. |
| License | PR #192 | Awaits Michael. Not a feature blocker. |
| Known small bugs | IQU45 environment | `rss_mb` / `viz_ms` log 0; ffmpeg not on PATH; no torch in the `.venv`. Fix when a local pass is free; not a roadmap phase of their own. |

Open but stale or parked PRs that this plan does not restart: #160 (junction map labels, implements step 1 of `docs/SPEC_intersections.md` — rebase and re-gate rather than rewrite), #141, #106, #103, #100, #83, and several October drive-safety PRs (#81, #79, #78, #77, #75, #74). Treat #160 as the existing junction-map implementation to finish, not a greenfield task.

## Feature inventory

Audit sources: `README.md`, `docs/SHADOW_GATE.md`, `docs/SPEC_intersections.md`, `docs/gvd_state_schema.md`, `python/train/scene_net.py`, `python/planning/path_predictor.py`, `python/planning/speed.py`, `python/perception/{cipv,detect,road_model}.py`, `python/collect/`, `scripts/collect_scene_lanes.py`, open issues #154 and the file-bus issues, and the PRs above. BeamNG map *contents* (whether a given level actually places traffic lights or speed-limit signs) were **not** inspected from `content/levels/*.zip`; those rows say unverified.

`architecture_parameter_count()` on this tip returns **2,203,694** for the default exported scene net (shared stem + the heads in `scene_head_sizes()`). There is no model-width or size preset flag on `main`. `--scale` only downscales the canvas.

Default scene heads on `main`: `lanes`, `curbs`, `objects`, `signs`, `lane_gt`. Optional / training-only: `ctl` (offline only; `--ctl-cond` stays refused live), `road_edge` (PR #179, separate from paint), `aux_depth` / `aux_sem` (train-only). Sign classes the net can name: `stop_sign`, `yield`, `traffic_light`, `speed_limit`. Lamp states: `unknown`, `red`, `yellow`, `green`. Object classes include `pedestrian` and `bike`; those stay unused for data and evaluation.

The shipped detector is YOLOv8n (`models/yolov8n.onnx`) with a COCO map: vehicles, `traffic_light` (9), `stop_sign` (11). There is **no lamp-colour classifier**; `docs/gvd_state_schema.md` says light `state` is `unknown`. The hardcoded planner will end a path at a stop sign, a yield, or a **red** light if those signs reach it; an unknown lamp is not treated as red (`path_predictor.py`).

| Feature | Code today | Data / labels | Status | Live gate |
| --- | --- | --- | --- | --- |
| Lane keeping | Scene `lane_gt` + modular controller; `--scene-ckpt` shadow and `--scene-ckpt-drive` (#191, #182). Hough fallback. Confidence vetoes (#186). | Painted DecalRoad lines (`painted_lanes.py`). No paint on one side = invalid slot, never a guessed width. | offline-tested / live UNPROVEN. No scene-net checkpoint has been shown to keep a lane live. | Shadow vs paint GT (`docs/SHADOW_GATE.md`), then a conservative `--scene-ckpt-drive` with Michael present. |
| Road edges / unpainted roads | Viz curbs are a 0.4 m *predicted* offset outside outermost paint (`road_model.road_edges`, always `kind=predicted`). Train-only `road_edge` head (#179). Detect `edge` head in flight (#199). README: unpainted / texture-baked roads get no labels. | `road_width_m` / `road_offset_m` or an explicit polyline when present (#199). Missing side stays masked. | Partial: train plumbing offline-tested; deploy detection in flight / UNPROVEN. Unpainted drive: missing. | After #199: offline edge error on labelled sides; live: model-drawn edges only, no invented paint. |
| Curves | Planner continues visible paint, including a circular-arc / roundabout heuristic. `--balance` upsamples curves. | Same paint labels; `--balance` tags curves / weaves / ramps. | offline-tested / live UNPROVEN | Holdout RMSE on curve-tagged rows; live: stay in lane on marked WCUSA curves at the speed cap. |
| Lead follow / ACC | `python/perception/cipv.py`, `plan_speed`, DRIVE-tab CIPV and speed cap (default 18 m/s). Signed relative speed for TTC (#184). | YOLO / future scene objects + tracker (#174 addressed tracker copy-through; issue #151 still open). | offline-tested / live UNPROVEN | Live: match lead speed without collision or false full-brake on a same-speed car ahead. |
| AEB | TTC < 1.2 s → target 0, glance HOLD; stays on after Engage drops (#183). Lua command dead-man is separate. | Same tracks. | offline-tested / live UNPROVEN | Live: AEB on a closing lead; no AEB on a receding or distant car. Michael present. |
| Cones / obstacles | #199 object / cone head. Collector does not record cones; semantic images are not turned into boxes. YOLO has no cone class. | Missing (detached zero loss if trained now). | missing (head in flight, labels missing) | First record static props; then offline recall; then live dodge / stop. |
| Vehicles with pose | YOLO boxes + tracker; scene `objects` head (x, y, yaw, length, width, speed, …). #199 oriented objects from traffic `pos` / `dir`. Length/width/height stay masked (probe has no size). | Traffic probe position / velocity / direction when present. | Partial, offline-tested / live UNPROVEN | Boxes stay on the lead through a curve; heading error reported only from the model. |
| Brake lights | Stage can mark `BRAKE` on a hazard. #199 `brakelight` head. Ego `brakelights` is not the lead; traffic records have no lamp state. | Missing for other cars. | missing labels; head in flight | After a lead-lamp sidecar exists: offline lamp accuracy; live: slow when the lead's lamp is on. |
| Traffic lights | COCO detector + scene `signs` / `sign_state`. Planner stops only on *red*. No lamp classifier today. beamngpy 1.36 `traffic_signals.*` (including `get_map_node_signals`) is documented in `docs/SPEC_intersections.md` as the **offline** lamp GT source. Whether `west_coast_usa` (or italy / east_coast_usa / utah / jungle_rock_island) actually places signal objects was not verified from the level zip. | Sign GT is not written by `collect_scene_lanes.py` today (no `signs` writer in the collector). | Partial code; labels missing; map contents unverified | After signal GT: offline lamp accuracy; live: stop on red, go on green, unknown ≠ red. |
| Stop signs | COCO `stop_sign` + scene class + planner stop at `SIGN_STANDOFF_M`. Same missing GT pipeline. Presence of stop-sign props per map: unverified from the level zip. | Missing collector sidecar | Partial code / live UNPROVEN | Live: full stop, then proceed when clear. Vision only. |
| Intersections (detect, turn, yield, GPS turns) | README: "Lane keeping only. There is no intersection handling, no turns at junctions and no routing." Issue #154. Crossing paint is dropped (labeler `HEADING_MAX_DEG = 45`; Hough slope gates). Path is extrapolated through the box. Junction labels go blank (need paint on both sides). Weaves abort near junctions, so crossings are scarce. PR1 (road loader + `junction_grade.py`) is on `main`. PR #160 (junction map + `jct.jsonl`) is open, not merged. GPS today is a lat/lon *hint* plus optional pin range/bearing (`nav_hint_from_bundle`); pin is not a route and `drive_to_pin` is false. | Need junction map (#160), crossing recordings, turn-lane and junction-path labels, stop-line labels. | missing (specced in `docs/SPEC_intersections.md` M1–M5) | See Phase C / M4. |
| Lane changes | Collector can plan weaves / lane changes (`excursion.py`). Planner has no lane-change policy; blinker is guessed from path shape. | Weave rows exist; not reviewed as a drive skill. | Data partial; policy missing | Live: signal, gap check (vision), change, settle. Not through a junction. |
| Merges / highway on-ramps | Spawn deck on WCUSA: ~40% city, 35% highway, 25% ramps (`spawn_mix.py`). Viz can tag merge/exit from paint geometry (`lane_roles.py`). No merge/yield planner. | Ramp-tagged rows via `--balance` / spawn mix. How much live ramp data exists: not re-counted here. | Partial data; policy missing | Live: merge into a gap or wait; no invented paint on the gore. |
| Speed limits / speed control | Longitudinal: `V_CAP` 18 m/s, UI speed cap and optional cruise override, curve and TTC caps (`planning/speed.py`). No vision speed-limit reader. Road `speedLimit` from the network must not drive deploy. Whether maps place readable speed-limit signs: unverified. | Scene class `speed_limit` has no collector GT. | Speed *control* offline-tested / UNPROVEN; limit *reading* missing | First verify signs exist on a map; then offline class accuracy; live: cap near the posted limit ± a stated band. |
| Parking / stopping | Stopping: AEB, sign/light path end, veto, Lua dead-man, Tech parking-brake hold so arcade gear 0 is not reverse (#185). Parking as a mode is a **non-goal** in `docs/SPEC_intersections.md`. | n/a | Stopping: offline-tested / UNPROVEN. Parking mode: out of scope. | Stopping is covered by AEB and vision-stop gates. |
| Night / weather / dirt | AUTO tab can cycle weather and time-of-day (offline smoke only). Dirt / unpainted roads: README limitation. | Diversity of night/rain/fog in existing ~4 h: not audited file-by-file. | mostly missing / unverified data | After labelled diverse clips: holdout by condition; live smoke at night and in rain on WCUSA. |
| Multi-cam (8-cam) | Tech rig, same-tick design (open PR #74 still listed). Scene-ckpt loads are main-only (#188). Retail is one window capture. | Collector can record all eight; training so far is typically `--cams main`. | Plumbing offline-tested; live 8-cam drive UNPROVEN | After frame-age is fixed: all eight slots update with main; no stale side cam used for drive. |
| Visualizer | GVD VISION stage (ribbon #197/#198, lanes, boxes, signs/lights glyphs, chase/BEV). Overlay contract on #199 (`docs/VIZ_CONTRACT.md`). Synthetic README media, not live captures. | Must consume model or camera fields only (`is_model_drawable`). | offline-tested / live UNPROVEN | A reviewer can tell what the model saw; no GT leak in `renderable_state`. |
| Recording / data diversity | CLI collector live-proven (rec2/rec3). Combined multi-root training (#180). ~4 h of driving is the current bottleneck (order of magnitude from project notes, not re-measured this audit). Maps in the AUTO list: WCUSA, italy, east_coast_usa, utah, jungle_rock_island. README: treat models as WCUSA-only; spawn deck is WCUSA-only. Highway speeds: spawn mix includes highway; collector speed range is configurable (AUTO default 8–17 m/s, cap 30). | Lossless `raw/` PNGs + `state.jsonl` + `gt.jsonl`. JPEG strips are not training data. | Recording: live-proven. Diversity: insufficient for "mostly works." | New recordings after collector gaps (junctions, signs, cones, night) are listed in Phase C. |
| Eval / holdout | `--holdout-blocks` / cells (#187), `--mask-events`, `scripts/eval_onnx_holdout.py`, `eval_scene_lanes.py`. | Same recordings. ep11 re-score is queued, result UNPROVEN until the log exists. | offline-tested | Compare models only on the same holdout and mask set. |
| Model-size presets | Missing on `main`. In flight on Cloud. | n/a | missing / in flight | Startup probe picks a size from latency, then VRAM; INT8 on Pascal; step-down if the game stutters. |
| Packaging / release | `install.bat`, `play_gvd.bat`, `play_gvd_tech.bat`, BeamNG mod, VERSION 1.7.13 alpha. Shipped weights: `models/yolov8n.onnx` only. No GitHub Release of scene checkpoints or model cards. | n/a | missing (baselines) | Phase F. |

### Status counts (22 rows above)

These are plan labels, not a scoreboard of live drives. Each of the 22 inventory rows has exactly one primary status. Split rows are footnoted, not double-counted. The five buckets partition the table: **1 + 7 + 7 + 6 + 1 = 22**.

- live-proven: **1**
- offline-tested and live UNPROVEN: **7**
- partial (code yes, labels or policy incomplete): **7**
- missing: **6**
- out of scope: **1** (parking mode; pedestrians are also out of scope and are not a row)

| # | Feature | Primary status | Note |
| --- | --- | --- | --- |
| 1 | Lane keeping | offline-tested / live UNPROVEN | |
| 2 | Road edges / unpainted roads | partial | Unpainted drive is still missing; counted once as partial. |
| 3 | Curves | offline-tested / live UNPROVEN | |
| 4 | Lead follow / ACC | offline-tested / live UNPROVEN | |
| 5 | AEB | offline-tested / live UNPROVEN | |
| 6 | Cones / obstacles | missing | Head in flight; labels missing. |
| 7 | Vehicles with pose | partial | Also offline-tested / live UNPROVEN; primary is partial. |
| 8 | Brake lights | missing | Labels missing; head in flight. |
| 9 | Traffic lights | partial | Labels missing; map contents unverified. |
| 10 | Stop signs | partial | |
| 11 | Intersections | missing | Specced M1–M5; #160 open. |
| 12 | Lane changes | partial | Data partial; policy missing. Counted once as partial. |
| 13 | Merges / highway on-ramps | partial | Data partial; policy missing. |
| 14 | Speed limits / speed control | partial | Control is offline-tested / UNPROVEN; limit reading is missing. Counted once as partial. |
| 15 | Parking / stopping | out of scope | Parking mode is out of scope. Stopping is already covered under AEB (row 5) and is not counted again. |
| 16 | Night / weather / dirt | missing | Mostly missing / unverified data. |
| 17 | Multi-cam (8-cam) | offline-tested / live UNPROVEN | Plumbing offline-tested; live 8-cam drive UNPROVEN. |
| 18 | Visualizer | offline-tested / live UNPROVEN | |
| 19 | Recording / data diversity | live-proven | Recording is live-proven (rec2/rec3). Diversity is still insufficient; counted once as live-proven. |
| 20 | Eval / holdout | offline-tested / live UNPROVEN | |
| 21 | Model-size presets | missing | In flight on Cloud; missing on `main`. |
| 22 | Packaging / release | missing | Baseline checkpoints not shipped. |

Several rows are also in flight (#199, presets, frame lag). In-flight is a work-ownership label, not a sixth status bucket, so it does not change the counts.

## Bottlenecks (in order)

1. **Data volume and variety** — about four hours, mostly WCUSA paint-following under BeamNG AI; scarce junction crossings; no sign/lamp/cone/lead-lamp sidecars.
2. **Live frame lag** — `frame_age` p90 11.5 s at 3.1 Hz. In flight (#201).
3. **Training I/O** — cache still on the HDD until the queued C: copy.
4. **Model size / card speed** — presets in flight; default net is a single architecture (~2.20 M weights by the in-repo counter).

Compute is not the first limiter. A mini-PC CPU cluster is not part of this plan.

## Phases

Each item lists goal, why, dependencies, likely files, labels (train-only where noted), acceptance (offline + live), effort (S/M/L), and owner (local / Cursor Cloud / IQU45 run).

### Phase A — Finish what is already running

**A1. Frame-age gate (#201)**
- Goal: drop stale camera frames and refuse to drive on them.
- Why: a 11.5 s p90 age makes every other live gate meaningless.
- Deps: Cloud PR #201; BeamNGpy changes allowed only with evidence and kept in-repo.
- Files: `python/runtime/frame_gate.py`, `python/sensors/cameras.py`, `python/sensors/frame_slot.py`, `python/run_vision.py`, `python/train/scene_ckpt_runtime.py`.
- Labels: none.
- Accept: offline suite + `scripts/test_frame_staleness.py`; live on IQU45 with Michael present: `frame_age` p90 in the same league as the tick period (target loop around 10–15 Hz, p50 ≤ 162 ms / p95 ≤ 215 ms from the intersection spec's dt bar). UNPROVEN until that log exists.
- Effort: M (in flight). Owner: Cursor Cloud, then IQU45 run.

**A2. Merge detect heads (#199) after Skeptic**
- Goal: opt-in `edge` / `obj` / `brakelight` without changing default `SceneNet` weights when those heads are off.
- Why: later perception work should not fork `scene_net.py` again.
- Deps: Skeptic GREEN > 8.5, empty Fix list; ordinary merge of current `main`.
- Files: listed on the PR (`detect_heads.py`, `detect_targets.py`, `export_detect_labels.py`, viz contract).
- Labels: export via `scripts/export_detect_labels.py`; cones and lead lamps remain masked.
- Accept: already reports 156 offline tests on `ce14621` (that figure is the PR's, not re-run here). Live: none until labels exist.
- Effort: S remaining. Owner: local merge after Skeptic.

**A3. IQU45 A/B + ep11 re-score + cache to C:**
- Goal: pick a trainer setting from the A/B suite; score ep11 on the #187 holdout; move the frame cache to C:.
- Why: later feature training should use the faster disk and a known holdout.
- Deps: GPU free after the A/B suite ends (estimated about 5:00 AM America/Chicago on 10 November 2026; unverified estimate, no job log cited in this audit). One job at a time.
- Files: none on the repo required.
- Accept: write the actual numbers from the logs; never invent them. Cache copy leaves raw data on the data drive.
- Effort: S once the jobs finish. Owner: IQU45 run (already queued).

**A4. GPU-aware presets (Cloud)**
- Goal: tiny / small / medium / large / xl (names may match the Cloud design) chosen by a short startup benchmark, then a VRAM check; tensor cores or INT8 as available; never fp16 on Pascal; reserve BeamNG's share first; step down one size if the game lags.
- Why: shipping many sizes in Phase F, and running the biggest net the 1080 Ti can hold at target Hz.
- Deps: research doc pending (Cloud research task, not yet in repo); hardware.yaml budgets (`max_inference_vram_gb: 4`, `live_input_long_side: 640`).
- Files (expected): `python/train/scene_net.py`, `python/train/train_scene.py`, `python/runtime/hw_probe.py`, `config/hardware.yaml`, README.
- Accept: offline — each preset builds, exports, and reports parameter count and a dummy-step latency hook; Pascal path stays FP32 train / INT8 or FP32 infer. Live UNPROVEN until IQU45.
- Effort: L (in flight). Owner: Cursor Cloud, then IQU45 probe.

### Phase B — Safety and honesty before any engage

**B1. Shadow log vs paint GT** (`docs/SHADOW_GATE.md`)
- Goal: main-camera scene net on WCUSA, `--no-engage`, log model `lane_x` next to painted GT, with junction / respawn / settle / near-black masks.
- Why: this is the next proof the project already wrote down. It is not a driving test.
- Deps: A1 so frames are current; #140 and live-tone fixes already merged; #150 (YOLO NMS) and the remainder of #151 still listed as preconditions in that doc.
- Files: `python/debug/gt_logger.py`, a new grader script, collector/supervisor mask flags.
- Labels: painted decals, not road-network edges.
- Accept: a real shadow log on disk; grader prints counts by mask reason. No pass number until a measured run exists.
- Effort: M. Owner: local code, then IQU45 run with Michael present (no engage).

**B2. Remaining #150 detector work and tracker wobble (#151)**
- Goal: finish the #150 work that is still open after class-wise NMS. NMS already landed in #172 (`17c2afa`, 5 November 2026): `_nms_xyxy` at IoU 0.45 after the 0.35 cutoff on the YOLOv8 ONNX path in `python/perception/detect.py`. Remaining #150 work is the score enter/exit band, a check on a recorded frame, and a tick-time measurement. Tracker speed should come from filtered motion, not box-height jitter (#151; #174 already filtered tracker position; the issue is still open).
- Why: CIPV, AEB, and the stage boxes still need the enter/exit band and a recorded-frame check. NMS itself is already on `main`. `docs/SHADOW_GATE.md` line 71 still says the detector has no NMS; update that stale line in a follow-up (out of scope for this docs PR).
- Deps: none. B1's citation that #150 is still listed as a precondition in `docs/SHADOW_GATE.md` is an accurate description of that stale doc, not of `main`.
- Files: `python/perception/detect.py`, `python/perception/track.py`, tests; a recorded-frame check and tick-time measurement script or log.
- Accept: enter/exit band covered by offline tests; a recorded-frame check exists; tick-time is measured and written down. No live claim.
- Effort: S. Owner: local.

**B3. File-bus honesty (#123, #143, #130)**
- Goal: atomic `gvd_ego.json`; `atomic_write_json` must not stall past the heartbeat; measure supervisor tick vs mod echo.
- Why: the shadow-gate doc calls the file bus the real engage risk.
- Deps: none for the code fixes. Live measure needs IQU45 + BeamNG.
- Files: Lua mod writers, `python/runtime/state_io.py`, a measure script.
- Accept: offline tests for torn reads and write duration under a slow-FS fake; live measurement log UNPROVEN until taken.
- Effort: M. Owner: local, then IQU45.

### Phase C — Labels and recordings the features need

Vision heads without labels are detached zeros. This phase is the main lever on bottleneck 1. Map and sensor data stay out of the runtime path (AST isolation tests, same style as `scripts/test_junction_grade.py`).

**C1. Finish junction map labels (rebase PR #160)**
- Goal: per-level junction map and per-row `jct.jsonl` (`in_jct`, `jct_id`, `next_jct_m`, `jct_type`, `minor`, `src`) for training masks and grading only.
- Why: junctions are blank or extrapolated today (#154).
- Deps: PR1 already on main; rebase #160 onto current `main`, Skeptic.
- Files: `python/collect/junction_map.py`, `scripts/build_junction_map.py`, `scripts/label_junctions.py`, collector hook, `scripts/junction_grade.py`.
- Labels: from an offline road dump / Navigraph comparison for grading, never for steer.
- Accept: existing golden freeze (PR body: 638 junctions on the WCUSA fixture, 253 major / 385 minor — those numbers are the PR's fixture counts, not a live map survey). Offline suite green. Isolation AST scan.
- Effort: M. Owner: local (rebase), not a rewrite.

**C2. Stop-line and junction-box labels**
- Goal: treat near-horizontal paint as `src=stop_line` (or mask), not as a discarded lane line and not as invented lane paint. Write a sidecar the trainer can learn from or mask.
- Why: M3a needs this; today's labeler drops paint more than 45° off heading.
- Deps: none to start; pairs with C1 for `in_jct`.
- Files: `python/debug/painted_lanes.py`, `python/debug/lane_targets.py`, `scripts/relabel_painted.py` or `scripts/label_stoplines.py`, tests and a synthetic fixture.
- Labels: DecalRoad paint classified by heading; training-only.
- Accept: synthetic stop bar labelled `stop_line` and excluded from `lane_gt`; a through-road's parallel paint unchanged. Isolation from `run_vision` / `path_predictor`.
- Effort: M. Owner: local. **Next-up spec below.**

**C3. GPS route intent, shadow only (SPEC M1)**
- Goal: `{turn, next_jct_m}` (and `route_src`) toward a user pin, display on the LIVE tab, **not** consumed by the planner yet.
- Why: turn-by-turn is the one allowed GPS use; M4 later feeds only `turn` as intent.
- Deps: beamngpy ≥ 1.36 already pinned (SPEC PR0). Navigraph used off the control path.
- Files: new `python/sensors/route_intent.py` (or similar), `python/run_vision.py` display, `python/viz/nerd.py`, `scripts/test_route_intent.py`.
- Labels: none for the net. Telemetry only.
- Accept: offline tests on a synthetic graph: `turn ∈ {left, right, straight, none}`, `next_jct_m` falls toward a junction; AST scan proves `path_predictor` / actuate do not import it. Live display UNPROVEN until IQU45.
- Effort: M. Owner: local. **Next-up spec below.**

**C4. Train-only sign and lamp GT**
- Goal: a `signs.jsonl` (or equivalent) sidecar from beamngpy `traffic_signals.get_map_node_signals` and whatever static sign catalog the level exposes, joined on time and pose. Never drawn, never sent to the planner.
- Why: stop / yield / light / speed-limit heads cannot learn without it. Collector currently writes no `signs` field.
- Deps: verify per map whether signals and sign props exist (open the level zip or dump once on IQU45). If a map has no signals, document that and skip it.
- Files: new `python/collect/signal_labels.py`, `scripts/export_signal_labels.py`, optional collector hook, tests with a fake `bng`.
- Labels: training-only.
- Accept: fake session produces stable rows; missing API → masked, not invented lamps. Isolation AST scan.
- Effort: M. Owner: local. **Next-up spec below.**

**C5. Junction-crossing and highway recordings**
- Goal: collect clips that actually go through junctions (not aborted by weave-guard) and that include highway speeds, on-ramps, and more than one lighting/weather setting. Keep `--min-paint-cov` honesty: do not invent paint.
- Why: M2b and merges cannot be trained on scarce crossings.
- Deps: C1 helpful; collector flag so junction approach is recorded rather than skipped. IQU45 + BeamNG.tech. Michael present if any engage; AI-driven collection does not need engage.
- Files: `scripts/collect_scene_lanes.py`, `python/collect/excursion.py`.
- Accept: `collect.json` shows a stated count of `in_jct` ticks after C1; no data deletion without Michael's OK on an exact list.
- Effort: L (machine time). Owner: IQU45 run.

**C6. Cones and lead brake-lamp recording**
- Goal: give #199 labels so those heads are not detached zeros.
- Why: obstacles and lead braking are part of "mostly works."
- Deps: #199 merged; a BeamNG way to spawn or find cones (unverified here — confirm with a Tech session before promising a class). Lead lamp from other vehicles' electrics if beamngpy exposes it; otherwise document the gap.
- Files: collector traffic probe, `scripts/export_detect_labels.py`.
- Accept: at least one recording where cone boxes and/or lead `brakelight` are non-masked; otherwise the head stays off.
- Effort: M. Owner: local code + IQU45 run.

**C7. Turn-lane labels and head (SPEC M2a) and junction-path head (M2b)**
- Goal: vision chooses the turn lane and the path through the box from cameras, not from the map.
- Why: owner decision: turn-lane choice ships before turns.
- Deps: C1, C2, C5; overlay check that labels sit on real paint / arrows.
- Files: labeler, `scene_net.py` new heads (opt-in, default net unchanged), `train_scene.py --heads`, viz contract (model-only).
- Accept: offline head losses on labelled crossings; default `SceneNet` weights unchanged when heads are off (same test style as #199). Live later in D/E.
- Effort: L. Owner: local or Cloud after labels exist.

### Phase D — Make the current stack actually drive (mostly works on painted roads)

Only after A1 (frame age) and B1 (shadow log). Live drives with Michael present. Camera-only. No `--ctl-cond`. No fp16 on the 1080 Ti. Vision ≤ 4 GB.

**D1. Conservative scene-ckpt lane keep**
- Goal: `--scene-ckpt-drive` holds a painted lane on WCUSA at a low speed cap (existing default 15 m/s cap flag).
- Why: README: GVD enters beta when a trained model drives the car.
- Deps: A1, A3 numbers, B1, B2, B3 as much as practical.
- Files: already on main (`scene_ckpt_runtime.py`, `run_vision.py`).
- Accept: offline already covered by #191 tests. Live: a clip + log with engage, heartbeat fresh, no veto storm, car stays in the painted lane on a chosen straight and a chosen curve. UNPROVEN until that run.
- Effort: M (run + whatever small fixes appear). Owner: IQU45 run.

**D2. Live CIPV / AEB**
- Goal: follow a lead and full-brake on a real closing TTC.
- Deps: D1 or parallel after A1; B2.
- Files: `cipv.py`, `path_predictor.py`, `planning/speed.py`.
- Accept: scripted Tech traffic; logs show TTC and `aeb` transitions that match the scene in the cam_main PIP (model/detector boxes, not GT).
- Effort: M. Owner: IQU45 run.

**D3. Road-edge following where paint is missing on one side**
- Goal: use the detected edge head (not an invented 3.5 m line) when only one paint line exists, at low weight / confidence.
- Deps: #199, C-level edge labels, overlay < 0.5 m on known curb stretches or mask (standing labeler note).
- Files: planner pairing, viz (model edge only).
- Accept: offline masked regression; live on a chosen WCUSA curb stretch. Unpainted dirt remains a later, separate claim.
- Effort: M. Owner: local + IQU45.

**D4. 8-cam when latency allows**
- Goal: train and run more than `main` without blowing the 4 GB / Hz budget.
- Deps: A1, A4 (size step-down), main-only is still the default on the 1080 Ti.
- Accept: live camera wall updates with main; frame-age gate still green.
- Effort: L. Owner: IQU45, after presets.

### Phase E — Intersections and the rest of "mostly works"

Follow `docs/SPEC_intersections.md` rather than inventing a second intersection design. Map geometry never reaches steer, pedals, stops, or lane choice. At M4 only `turn` may reach the planner as intent.

**E1. Vision stops (SPEC M3a)**
- Goal: stop-line head + sign/lamp gate from vision.
- Deps: C2, C4, planner already ends the path on stop/yield/red if it is given those objects.
- Files: `scene_net.py` stop-line head, `path_predictor.py` wiring from *model* signs (not GT), tests.
- Accept: offline: labelled stop bars; unknown lamp ≠ red. Live (SPEC): 10/10 vision stops on agreed intersections, 0 actuations sourced from the map.
- Effort: L. Owner: local + IQU45.

**E2. Live junction prove (SPEC M4)**
- Goal: correct-lane turns using vision path + GPS `turn` intent only.
- Deps: A1 dt bar, 8-cam checkpoint or an explicit waiver to stay main-only on the 1080 Ti, M1 (C3), M2a/M2b (C7), M3a (E1).
- Accept: SPEC M4 bar (20/20 correct-lane turns across ≥ 3 junctions, budgets held). UNPROVEN until run.
- Effort: L. Owner: IQU45.

**E3. Click-to-goal pin (SPEC M5)**
- Goal: destination source for M1, still GPS-only output.
- Deps: C3.
- Effort: S–M. Owner: local.

**E4. Lane changes**
- Goal: planned change into an adjacent lane with a vision gap check and blinker as a command, not a guess from curvature.
- Deps: D1, reliable adjacent-lane paint or edge; not inside `in_jct`.
- Files: planner corridor, excursion ideas reused only as data not as the live AI line.
- Accept: offline scenarios; live:  several changes each way on a WCUSA multi-lane road with a documented miss/abort rate. UNPROVEN until run.
- Effort: L. Owner: local + IQU45.

**E5. Merges and on-ramps**
- Goal: yield-or-go at a gore using vision tracks and paint/edge, never navgraph.
- Deps: C5 ramp recordings, E4 helpful, D2.
- Accept: live merges on WCUSA ramps with Michael present.
- Effort: L. Owner: IQU45.

**E6. Speed-limit reading (if signs exist)**
- Goal: cap from a *seen* speed-limit sign. If the map has no such props, keep the UI cap and document it.
- Deps: C4 verification step.
- Effort: M if signs exist, otherwise S (document). Owner: local + IQU45.

**E7. Night / weather robustness**
- Goal: "mostly works" still holds at night and in rain/fog on WCUSA.
- Deps: C5 diverse data; photometric `--augment` already exists.
- Accept: holdout split by condition if labels allow; live smoke clips. UNPROVEN until run.
- Effort: M–L. Owner: IQU45.

**E8. Visualizer polish**
- Goal: stage readability closer to an FSD Beta-like cabin view, still model/camera only.
- Deps: #199 viz contract; never draw GT even as a debug default in shipped builds.
- Effort: M. Owner: local.

Pedestrians stay out of scope. Parking-as-a-mode stays out of scope. When this phase's live gates are mostly green — especially intersections and stoplight handling, tested offline and live with Michael present — GVD is ready for Phase Collect, then Phase F.

### Phase Collect — 8–10+ hour diverse recording

Trigger (Michael): intersections and stoplight handling work and have been tested (Phase E, at least E1 vision stops / M3a and E2 live junction prove / M4). Then record one long, diverse run before any rented-GPU baseline training.

**Collect1. One 8–10+ hour mixed session**
- Goal: a single planned collection of **at least 8 hours**, preferably 10 or more, after the intersection and signal stack is proven enough to label what is recorded. Cover highway speeds, junctions, traffic signals (where the map actually has them), night and weather if the AUTO / collector cycles are available, and more than one map from the existing list (west_coast_usa plus at least one of italy, east_coast_usa, utah, jungle_rock_island).
- Why: today's ~4 h, mostly WCUSA paint-following, is too narrow to train the ship baselines. Phase F trains every preset on this run (plus earlier data if it stays in the mix), not on the short development set alone.
- Deps: Phase E intersection and stoplight gates; collector hooks from Phase C (junction crossings, sign/lamp sidecar, stop-line labels) so the hours can actually be labelled. Do not invent paint, lamps, or map contents. Recordings are deleted only with Michael's OK on an exact list. Model and checkpoint files (npz, ONNX, trainer weight snapshots) are never deleted.
- Files: `scripts/collect_scene_lanes.py` flags already on main (maps rotation, weather, time of day, speed range, spawn mix, `--minutes`, `--until-disk`, `--max-file-gib`); status in `collect.json`. No new actuation.
- Labels: painted lanes, stop-lines, junction sidecars, and sign/lamp GT where the APIs and maps provide them. Training-only sensors stay training-only.
- Disk plan: Record to the data drive (`GVD_data` / `GVD_DATA_DIR`), never C: or D: (the collector already refuses those as the data root; README Data disk). Before the first chunk, check free space on that drive and keep at least the 20 GB stop floor plus room for the planned chunk. The collector stops if the data disk drops under 20 GB free (`scripts/collect_scene_lanes.py`). The uncompressed ceiling for the yaml rig at 10 Hz is about 231.7 GB/h (collector docstring; that is the documented ceiling, not a measured rate from this run). An 8–10 h lossless 8-cam session at that ceiling is on the order of 1.9–2.3 TB, on top of the existing raw set (an unverified estimate of ~74 GB from project notes) which stays on disk. Do not use the default `--minutes 45`. Either set `--minutes` ≥ 480 (8 h) or, preferably, run documented resume chunks (the output directory is resumable and a second start appends until `--minutes` is reached) so each piece can finish, be checked, and if needed be moved or rotated off the FAT32 data drive before the next chunk. Do not use `--until-disk` as a substitute for the ≥ 8 h bar: that flag ignores the clock and stops only at the 20 GB floor, so it cannot prove the hours exit. The data drive is FAT32, so no file may exceed 4 GB; `--max-file-gib` (default 3.5) exits 7 (`file_cap`) if a file would pass the cap. Between chunks: confirm free space again, and if the drive cannot hold the next piece, move completed chunks to a larger volume (or compress / record fewer cameras / a lower `--rates` than 10 Hz, with the chosen rate written in the mix report) — then resume. Record a measured GB/h from that run's `collect.json` rate line and use it for later chunks rather than inventing a new number. Existing recordings are deleted only with Michael's OK on an exact list. Model and checkpoint files are never deleted.
- Accept: `collect.json` (or the run log) shows elapsed time ≥ 8 h. A short written mix report lists hours (or frame counts) by map, day vs night, weather, highway vs city, and `in_jct` / signal-row counts — using the real numbers, and marking a category unavailable if a map has no signals or night/weather was not run. Valid-label share and the damage sensor are checked the same way as earlier recordings. UNPROVEN until that log exists.
- Effort: L (machine time; BeamNG AI may drive). Owner: IQU45 run. Live model drive is not part of this recording.

### Phase F — Rent a GPU and ship baseline models

Trigger (Michael): intersections and the other in-game features are at mostly-works (no pedestrians), **and** the Phase Collect 8–10+ hour diverse run is on disk. Then rent a GPU and train the full size set to ship with GVD.

**F1. Cloud trainer bring-up**
- Goal: one RunPod (or equal) 3090 job that reproduces the IQU45 `train_scene` invocation: same `--holdout-blocks`, `--mask-events`, `--holdout-source`, LR `1e-3`, no `--ctl-cond`, mixed precision allowed on the 3090 (compute ≥ 7.0), FP32 export.
- Why: one 1080 Ti cannot train five sizes in a reasonable time; a 3090 at about $0.22/hr is a suggested option.
- Deps: Phase E mostly-works; Phase Collect pack (the 8–10+ hour diverse run, plus earlier recordings if they remain in the mix); A4 presets merged; cache-friendly dataset pack (no deletion of raw IQU45 data).
- Files: a `docs/CLOUD_TRAIN.md` (or a section in the README) with the exact commands, image, and data layout.
- Accept: a smoke epoch on the rented box that matches the local dry-run shapes; dollars and hours logged, not guessed after the fact.
- Effort: M. Owner: IQU45 data export + cloud run. Needs Michael's yes to spend.

**F2. Train every preset on the same data and the same holdout**
- Goal: tiny / small / medium / large / xl (or the preset names A4 ships), one checkpoint family per tier, trained on the Phase Collect diverse run (and any earlier roots kept in that combined set).
- Why: users try a stock model that fits their card, then compare their own training.
- Deps: F1, A4, Phase Collect.
- Accept: each size writes `metrics.json`, the ONNX (FP32 graph), and the epoch npz. Score all of them with `scripts/eval_onnx_holdout.py` on the **same** holdout rows (`rows_sha` must match). Publish the table; do not invent cells.
- Effort: L (mostly GPU hours). Owner: cloud run.

**F3. Model cards and downloadable baselines**
- Goal: one card per size: parameter count, train commit, data mix (maps, hours, cameras), holdout identity, metrics, intended GPU tier, Pascal notes (no fp16), license.
- Files: `models/baselines/<size>/` plus `models/NOTICE` updates; GitHub Release next to the alpha zip.
- Accept: a cold machine can `install.bat`, download one baseline, and run `--scene-onnx` / `--scene-ckpt` without training.
- Effort: M. Owner: local packaging.

**F4. "Train on your own data and compare" guide**
- Goal: README section: collect → cache → `train_scene` with the user's chosen preset → `eval_onnx_holdout.py` against the shipped baseline on *their* holdout or on a published eval split if one is released.
- Why: Michael's ship criterion: people test stock models, then their own.
- Accept: the commands work from a clean clone in the offline suite's dry-run mode.
- Effort: S. Owner: local.

**F5. Release**
- Goal: version bump (feature release, so a main alpha bump per README: `1.x.0`), changelog from git, installers, baselines, guide.
- Deps: Skeptic on the packaging PR; Michael on license #192 if that should ship with it.
- Effort: M. Owner: local + Michael.

## Recommended order (dependency)

```
A1 frame age ─┬─► B1 shadow log ─► D1 live lane keep ─► D2 CIPV/AEB
A2 #199      ─┤                                      ├─► D3 edges
A3 A/B+cache ─┤                                      └─► D4 8-cam
A4 presets ───┘
C1 #160 rebase ─► C5 junction/highway data ─► C7 turn-lane + junction-path
C2 stop-line labels ─────────────────────────► E1 vision stops ─► E2 M4 junction prove
C3 M1 GPS intent ────────────────────────────► E3 M5 pin ───────► E2
C4 sign/lamp GT ─────────────────────────────► E1 / E6
C6 cone + lead lamp ─► #199 heads become real ─► D2/D3 extras
D* + E* mostly works ─► Collect 8–10+ h diverse run ─► F1–F5 cloud baselines
B2 remaining #150 (enter/exit band, recorded-frame check, tick-time), B3 file bus can proceed in parallel as soon as A1 is not conflicting
```

IQU45 GPU stays single-job: A3 first (already running), then no new train until Michael opens the queue.

## What this plan will not do

- Pedestrian detection or yielding as a shipped skill.
- Parking-as-a-mode.
- Driving from depth, semantic, LiDAR, radar, or the road network.
- `--ctl-cond` or treating open-loop ctl error as a driver.
- fp16 on the 1080 Ti.
- Unattended live model drive.
- Deleting recordings without Michael's OK on an exact list.
- Deleting model or checkpoint files (npz, ONNX, or trainer weight snapshots).
- A CPU mini-PC training cluster.
- Inventing holdout scores, live Hz, or "mostly works" claims before the logs exist.

## Sources

- Repo tip `72b5854` on `origin/main`
- `README.md`, `docs/SHADOW_GATE.md`, `docs/SPEC_intersections.md`, `docs/gvd_state_schema.md`
- Issue #154 (intersections), #150, #151, #123, #143, #130
- PRs #199, #201, #192, #191, #187, #179, #172, #160
