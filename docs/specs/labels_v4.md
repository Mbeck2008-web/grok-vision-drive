# Labels v4 — training-label batch (spec only)

Training may use any BeamNG signal. Runtime and the visualizer stay RGB: model outputs only, sim/map overlays only with `--debug-gt` (`python/run_vision.py:456-462`). Live inference is RGB + ego kinematics + a GPS nav hint; LiDAR, radar, and IMU do not feed the planner (`python/run_vision.py:849-851`). This batch does not turn those extras on.

Painted lanes stay paint. Road edges, curbs, and a missing line are never written into `lane_gt`. No head may invent a line, a cone, a lamp colour, or a box the sensors did not support.

Host: GTX 1080 Ti, 11 GB, shared with BeamNG, live cap 4 GB, Pascal fp32 (`config/hardware.yaml:1-16`, `python/train/model_size.py:35`, `python/train/model_size.py:187-197`, `python/train/scene_net.py:6-8`). New supervision is training-only, same pattern as aux / `road_edge` (not in `OUTPUT_FIELDS`, not in `_OnnxScene`, `python/train/scene_net.py:96-112`, `python/train/scene_net.py:978-981`, `python/train/scene_net.py:2041-2054`).

There is no `tags.jsonl`. The collector inlines a `tags` dict into `state.jsonl` (`scripts/collect_scene_lanes.py:3180-3204`). The junction sidecar is `intent.jsonl`.

## Label types

### 1. Intersections

| Piece | Source already in repo | Derivation | File | Check | Mask / weight | Head | Recorder today |
|---|---|---|---|---|---|---|---|
| Junction distance and the exit the car took | Road graph + future pose. Junctions are degree ≥ 3 (`python/sensors/route_intent.py:82-83`). `label_sequence` snaps to one and classifies the driven exit (`python/collect/route_hindsight.py:138-238`). `APPROACH_M` is 60 (`python/sensors/route_intent.py:21`). | Offline `scripts/label_route_intent.py` loads a graph JSON and refuses `gt.jsonl` (`scripts/label_route_intent.py:20-40`). Row is `{t, cmd, next_jct_m, src:"hindsight"}` (`python/collect/route_hindsight.py:68-83`). `cmd` is `follow` outside 60 m, else `left` / `right` / `straight`, or `mask`. | `intent.jsonl` beside `state.jsonl`. | Replay the sidecar against the graph. A masked `uturn` / `roundabout` / `ambiguous` / `no_future` must stay masked. | Loss only when `cmd` is `left`/`right`/`straight`. Do **not** copy `attach_route_cmds`, which turns a missing sidecar into `follow` (`python/train/route_cond.py:126-128`). | Existing `--route-cond` embedding consumes `cmd` and `next_jct_m` (`python/train/route_cond.py:116-135`). No new live input. | Pose and `dir` are in `state.jsonl`. The graph file is offline. `near_junction` is a weave gate inside 12 m and treats a dead end as a junction (`python/debug/gt_geom.py:69`, `python/collect/excursion.py:764-798`). It is not stored and is not this label. |
| Which branch continues | Same function already picks `entry` and `exit_n` neighbor ids, then drops them (`python/collect/route_hindsight.py:218-238`). | Write those two ids onto the sidecar. That is the exit the car drove, not a painted turn-lane. | Same `intent.jsonl`, keys `entry`, `exit`. | Ids must be neighbors of the junction node. | `mask` rows stay out of the loss. | Same route embedding. Do not add a live path head in this batch. | Same pose stream. |
| Per-lane through/turn permission | **Not derivable.** `LANE_LINE_MATERIALS` is paint only (`python/debug/painted_lanes.py:73-85`). The graph has no turn-lane bits. | Do not invent arrows or a `lane_turn` tensor. Leave this masked until a decal material or signal attribute that names the permission shows up in a dump. | — | — | — | None. `HEAD_NAMES` has no turn-lane head (`python/train/scene_net.py:112`). | — |
| Stop line | Painted decals whose stroke is ≥60° off heading, crossing x=0, y in [-1, 80] m (`python/debug/painted_lanes.py:57-58`, `python/debug/painted_lanes.py:304-319`, `python/debug/painted_lanes.py:584-628`). | `scripts/label_stoplines.py` → bars `{src, y_m, x0, x1, valid}`. `src=stop_line` is valid; the 45–60° band is `src=mask`, `valid=false` (`python/debug/painted_lanes.py:321-336`, `python/debug/painted_lanes.py:567-581`). Never written into `lane_gt`. | `stoplines.jsonl`. CLI refuses `gt.jsonl` (`scripts/label_stoplines.py:1-4`, `scripts/label_stoplines.py:47-48`). | Project `y_m` onto the main frame the way `scripts/lane_contact_sheet.py` projects `lane_gt`. Median \|error\| ≤0.5 m (the merger already collapses bars 0.5 m apart, `python/debug/painted_lanes.py:621`). A `mask` bar must not train. | Positive loss only for `valid=true` bars. Crosswalks and stop bars share this heading test; do not relabel one as the other. | No consumer. The signs head is a point, not a bar (`python/train/scene_net.py:61-63`). Add a training-only scalar `(y_m, valid)`, not an `OUTPUT_FIELDS` entry, in the intersection PR. | Decals come from the level zip / `<map>_lines.json.gz` (`python/debug/painted_lanes.py:127-145`), not from the camera. Offline. |
| Traffic-light state | `traffic_signals.get_map_node_signals` when the API exists (`python/collect/signal_labels.py:56-67`). Ego frame via `world_to_ego` (`python/collect/signal_labels.py:80-97`). Class must be in `SIGN_CLASSES`; unknown class or pose is dropped (`python/collect/signal_labels.py:146-175`). Lamp text maps through `_norm_lamp`; a non-light is `unknown` (`python/collect/signal_labels.py:49-54`). | One **per-tick** payload. `scripts/export_signal_labels.py` stamps a single JSON file onto every row (`scripts/export_signal_labels.py:39-54`), so it cannot be the live lamp. | `signals.jsonl` raw per tick, then `signs.jsonl` `{t, signs:[{cls,x,y,state,src}]}` or `{t, err}` (`python/collect/signal_labels.py:180-192`). | Join on `t`. A row with `err` or unrecognized payload trains nothing. Overlay the ego xy on the main frame. | Position loss only for parsed signs. Lamp CE only when `state` is `red`/`yellow`/`green`. `unknown` stays index 0 (`python/train/scene_net.py:64`) and must **not** take lamp loss: today a present sign is `sign_valid=1` even when the lamp is unknown (`python/train/scene_net.py:611-632`). | Existing `signs` / `sign_class` / `sign_state` (`python/train/scene_net.py:57-64`, `python/train/scene_net.py:674-707`). Trainer does not read `signs.jsonl` today (no loader under `python/`). | **Not captured.** The collector never calls `traffic_signals`. |

### 2. Obstacles / drivable space

**Source.** Training-only depth and annotation already decode when a camera has `train_gt: true`: uint16 centimetres in `raw/<cam>_depth`, uint8 class ids in `raw/<cam>_sem`, names in `sem_classes.json` (`python/data/raw_frames.py:538-541`, `python/data/raw_frames.py:584-641`, `python/sensors/cameras.py:753-770`). The collector writes those planes only for cameras that rendered them, and writes the palette once (`scripts/collect_scene_lanes.py:2720-2730`, `scripts/collect_scene_lanes.py:3534-3541`). `config/cameras.yaml:71` leaves `train_gt` commented out, and collect passes `rgb_only=True`, so **current recordings do not contain depth or semantics.**

**Derivation.** Do not reuse `drivable_mask_from_sem` as “everything else is blocked.” That helper is an allowlist of paved names (`ROAD`, `Asphalt`, `ASPHALT`, `Highway`, `HIGHWAY`) and leaves every other id at 0 (`python/train/sensor_aux.py:28-29`, `python/train/sensor_aux.py:239-254`). Dirt is not in that list on purpose. Build a three-way label from the full-resolution PNGs (the 12×16 aux grid is a feature target, not a driving mask, `python/train/sensor_aux.py:21-23`):

- `drivable` — pixel whose class name is on the paved allowlist **and** whose name was present in this recording’s `sem_classes.json`.
- `blocked` — pixel whose class name is on a human allowlist written down from that same file (walls, trees, poles, barriers, guard rails, buildings, and any other static class the dump actually contains). No name is added because it “sounds like” an obstacle.
- `unknown` — class 0, a name on neither list, or depth 0. Depth 0 is already invalid in the aux loader (`python/train/sensor_aux.py:69-84`). The centimetre scale is `depth_01 * far * 100` and is documented as unverified (`python/data/raw_frames.py:589-610`).

**Clearance.** For blocked pixels only, unproject with the recorded intrinsics and the `cameras.yaml` pose (same projection family as `lane_contact_sheet`). Nearest distance is the minimum range whose foot lies within 0.5 m of the painted center polyline (`lane_gt` centers at 5/10/20/30 m, `python/debug/lane_targets.py:19-22`). If the 5 m and 10 m center slots are invalid, clearance is masked. Do not invent a straight corridor.

**Storage.** `obstacles.jsonl`: `{t, cam, grid_h, grid_w, x_min, x_max, y_min, y_max, mask}` where `mask` is row-major `0=unknown, 1=drivable, 2=blocked` on a fixed BEV grid (64×64, x ∈ [-16, 16] m, y ∈ [0, 40] m), plus `clear_m` and `clear_valid`. Full PNGs stay the raw record. Do not put this in `gt.jsonl`.

**Check.** Overlay the BEV class on `raw/main_sem` for a handful of frames: every `blocked` cell must be an allowlisted id, every `unknown` cell must be absent from both lists. Before trusting `clear_m`, one parked prop at a measured distance must agree within 1 m. If it does not, the depth scale is wrong and clearance stays masked.

**Weight.** CE on `drivable`/`blocked` only. `unknown` weight 0. Clearance is masked smooth-L1, weight 0 until the prop check passes.

**Head.** New training-only conv on the main-sector stem, same attachment point as `aux_depth_head` (`python/train/scene_net.py:1573-1584`). Not exported. The live net does not grow.

**LiDAR / radar.** Off (`config/sensors.yaml:12-18`). Not a source for this batch. Point clusters are not obstacles and not cones.

### 3. Curbs and road edges

Painted `lane_gt` stays `tech_from_painted`: a lookahead with paint on only one side is NaN / mask 0, never a guessed width (`python/debug/painted_lanes.py:1-21`, `python/debug/lane_targets.py:178-233`). The collector writes that dict and does not write `curbs` (`scripts/collect_scene_lanes.py:2871-2884`).

**Road edge (mapped boundary, not paint, not a curb).** `get_road_edges` left/right become one median width; the widest node is treated as a junction flare and discarded (`python/debug/gt_geom.py:225-273`). `pack_road_edge` turns `road_offset_m`, `road_width_m`, and heading into left/right x at `LOOKAHEADS_M` and never touches `lane_gt` (`python/train/sensor_aux.py:181-236`). The opt-in `road_edge_head` already exists and is outside the default ONNX (`python/train/scene_net.py:1587-1590`, `python/train/scene_net.py:1304-1313`). Use that vector. Do not retarget it at the flare edges in this batch.

**Curbs.** The curb head wants two 16-point polylines (`python/train/scene_net.py:54-55`, `python/train/scene_net.py:678-682`). Nothing fills `tech["curbs"]`. The live “curb” is a 0.4 m shift of the outer painted lane and is explicitly not a detection (`python/perception/road_model.py:198-204`). A curb label is the depth boundary of an allowlisted curb/guard-rail class, resampled to 16 ego xy points. **If `sem_classes.json` has no such name, `curb_valid` stays 0.** Do not copy `road_edge` into the curb slots or into `lane_gt`.

**Check.** Contact-sheet the road-edge xs. A unit test must fail if any `road_edge` value is written into a `lane_gt` slot. Curb overlay only after the allowlist exists; otherwise the sheet shows no curb.

**Weight.** Masked smooth-L1. Default `--heads lane_gt` already zeros the other heads (`python/train/scene_net.py:1292-1299`). Road-edge and curb losses stay opt-in.

### 4. Cones

**Blocked until a palette proof.** No cone class, no static-prop walk, and no cone file exist. `OBJECT_CLASSES` is `vehicle, car, truck, bus, pedestrian, bike` (`python/train/scene_net.py:62`). Inserting `cone` there would shift indices and break old checkpoints.

If a later `sem_classes.json` contains a name a human marks as a cone, the label is connected components of that id, median depth, ego xy, written to `cones.jsonl` `{t, cones:[{x,y}]}`. Components with invalid depth are dropped, not placed at a default range. The head, when added, is a new training-only slot, not a new `OBJECT_CLASSES` entry, and not in default ONNX. LiDAR blobs are not cones.

### 5. Vehicles, heading, brake lights

**Position and heading, companions only.** `TRAFFIC_CHUNK` reads `pos`, `vel`, `dir` for `gvd_traffic_0` and `gvd_traffic_1` every 0.5 s (`python/collect/control_log.py:68-79`, `python/collect/control_log.py:165-177`). Other rows store `traffic: null` (`scripts/collect_scene_lanes.py:71-72`, `scripts/collect_scene_lanes.py:3204`). Spawned companions are `etk800` (`python/collect/spawn_mix.py:1013-1025`). West-coast default count is 2; other maps default to 0 (`python/collect/spawn_mix.py:966-969`).

**Derivation.** Ego xy from `world_to_ego` (`python/collect/signal_labels.py:80-97`). Yaw is the traffic `dir` in that same frame. Speed is the horizontal `vel` magnitude. Class `car` (they are etk800s). On a null tick, `object_valid=0` for that slot. Do not interpolate a car. `unseen_s` may count time since the last real sample for that name and must not create a new box.

**Do not train size.** `_object_matrix` fills missing length/width with 4.5 m and 1.8 m (`python/train/scene_net.py:574-584`). Those defaults are not labels. Supervise x, y, yaw, speed only.

**Ambient traffic** is not in the probe. The spawn Lua already walks `be:getObjectCount` (`python/collect/spawn_mix.py:992-1000`). Extending the probe to every non-player vehicle’s `pos`/`vel`/`dir` is a recorder change, still without a class guess: model string if the object returns one, else `vehicle`, never `truck`/`bus`/`pedestrian`/`bike` from a missing field.

**Brake lights of other vehicles are not derivable from current data.** Ego `veh.brakelights` is the player electrics whitelist (`python/collect/control_log.py:34`, `python/collect/control_log.py:217-228`). The traffic chunk does not read electrics. Do not copy the ego lamp onto other boxes, and do not threshold red pixels. A later one-tick vehicle-Lua probe may add `brakelights` only after a dump shows the field coming back. Until then the channel stays masked. No head field exists for it today (`OBJECT_FEAT` is x, y, yaw, length, width, speed, partial, unseen_s, `python/train/scene_net.py:60`).

**Check.** Ego-frame xy versus `world_to_ego(pos)` within 0.5 m on rows where `traffic` is non-null. Heading versus `dir` within 5°. A null row must not score.

**Head.** Existing `objects` / `object_class` / `object_valid` (`python/train/scene_net.py:544-588`). Empty slots stay `object_valid=0` so they do not dominate the match loss.

## Non-goals

- No runtime or visualizer change. No new default ONNX output. No LiDAR/radar attach. No `--debug-gt` drawings in this batch.
- No edit to `gt.jsonl` / `lane_gt` packing. Sidecars only, matching stop-line, sign, and intent writers.
- No per-lane turn arrows, no cone class index, no other-vehicle brake-light channel, until the blocked items above have a dump.
- No learning-rate change, no second pass at `1e-4`, no fp16 on this GPU.
- Parking, pedestrians, and U-turns stay masked (`docs/SPEC_route_conditioning.md`).

## Recorder changes before the next collect

1. Set `train_gt: true` on `main` and `wide` only (`wide` is the urban-intersection camera, `config/cameras.yaml:72-80`). Leave the other six RGB-only so depth and annotation do not render eight ways on the shared 11 GB card.
2. Append one `get_map_node_signals` payload per tick to `signals.jsonl` (`python/collect/signal_labels.py:56-67`). On a missing API, write `{t, err}` and continue. Do not invent lamps.
3. Extend the traffic read to every non-player object’s `pos`/`vel`/`dir`, still at 2 Hz, names included. Do not add brake lights in that change.
4. Keep LiDAR and radar false.

Offline labelers (`intent`, `stoplines`, `signs`, `obstacles`, `road_edge`) run after the collect, BeamNG closed.

## Implementation PRs (smallest first)

1. **Recorder only.** Items 1–4 above. No trainer, no head. Acceptance: a short collect writes `raw/main_depth`, `raw/main_sem`, `raw/wide_*`, `sem_classes.json`, and per-tick `signals.jsonl`; RGB folders unchanged; `gt.jsonl` still paint-only.
2. **Obstacles.** `obstacles.jsonl` + training-only head. Palette allowlists checked in. Clearance weight 0 until the prop check. No ONNX change.
3. **Intersections.** Generate `intent.jsonl` (plus `entry`/`exit`), `stoplines.jsonl`, and `signs.jsonl` from the per-tick dump. Join on `t` with mask-on-missing. Lamp loss skips `unknown`. Training-only stop-line scalar. Existing signs head and `--route-cond` consume the rest. No live input added.
4. **Road edge, then curbs.** Turn on the existing `road_edge` head from `pack_road_edge`. Curb polylines only if the allowlist has a curb/guard-rail name; otherwise `curb_valid=0` and a test that proves it.
5. **Vehicles.** Fill `tech.objects` from the traffic rows. Size channels out of the loss. Brake lights untouched.
6. **Cones, conditional.** Only if PR 1’s `sem_classes.json` has a human-confirmed cone name. Otherwise this PR is not opened.

## Offline verify

- `py -3 scripts/lane_contact_sheet.py` on paint, plus the same projection for stop bars, road-edge xs, and vehicle centers.
- Unit tests: missing sidecar → mask, not `follow`; `road_edge` not copied into `lane_gt`; `curb_valid` all 0 without an allowlist; unknown lamp excluded from lamp CE; depth 0 and class 0 excluded from the obstacle loss; null traffic row has `object_valid=0`.
- `scripts/run_offline_suite.py` after each PR. No collect and no train on this machine.

## Risks

- **VRAM.** Depth and annotation on eight cameras, a full-res decoder, or a new default ONNX head will fight BeamNG on 11 GB. Main+wide and training-only heads are the cap. Live infer stays ≤4 GB fp32.
- **Label noise.** Nav median width is not a curb (`python/debug/gt_geom.py:261-265`). Perpendicular paint mixes stop bars and crosswalks. `near_junction` is the wrong junction label. Object size defaults (4.5×1.8) and a missing-intent→`follow` join would both teach a lie. Depth scale is unverified until the prop check. `train_gt` decode errors skip GT and keep the colour frame (`scripts/collect_scene_lanes.py:3654-3668`).
- **Class imbalance.** Most ticks have no stop bar, no cone, no light, and empty object slots. Loss is on valid slots only. Repeat junction approaches, stop-line rows, and blocked-obstacle rows the way `--balance` repeats curves, and keep those repeats inside one split.
- **Holdout leakage.** A time-tail split puts the same intersection in both sides. Use `--holdout-cells` (250 m cells, 50 m drop around them, `python/train/train_scene.py:333-339`). Join sidecars only inside the row’s own recording root. `impacts` / `settle` / `stall` stay masked (`python/train/event_mask.py:1-21`) and must not be the only source of “stopped at a light.”
