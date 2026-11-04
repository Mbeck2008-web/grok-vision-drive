# Shadow gate

Status: **Unproven.** No shadow log exists under `GVD_DATA_DIR` yet. The bar below stays unproven until one does.

## Where things stand

- No trained model has driven the car.
- The only live proof is recording: `scripts/collect_scene_lanes.py` recording while BeamNG's AI drives (rec2, rec3; see the README).
- The ctl head (`--heads lane_gt,ctl`, `--ctl-cond`, `--ctl-loss`) is offline only. Open-loop error on the AI's own controls is not a driver. A Rev 5 or other ctl PASS is not engage evidence.

## The next proof

A **shadow** run of the **main-camera** scene net on **west_coast_usa**, scored against **GT paint labels** (the painted-lane labels, `gt_source` `painted_decals`). The model predicts. It never actuates.

That is the whole gate. It is not a driving test.

## Excluded

- Engage or any actuation from the model, including shadow-applying it.
- Eight cameras. Main camera only.
- Retail BeamNG.drive.
- Junctions. They stay masked, not solved.
- Force feedback.
- Any new loss or ctl head work. ctl stays offline, and a `gvd_ctl_cond` model stays refused live (the scene runtime already refuses it).
- Screenshots. A screenshot is not a log.

## What exists today

Real names on main:

- Supervisor: `python/run_vision.py --lane-source scene --scene-onnx <model>` reads the scene net's lane center every tick.
- Hard engage disable: `--no-engage` (or `GVD_NO_ENGAGE=1`). No key, UI or file can engage.
- Live GT log: `--gt-log` (or `GVD_GT_LOG=1`) writes `logs/gt_<stamp>.jsonl` under the data root (`GVD_data`, or `GVD_DATA_DIR`). Each row has `model_lane_x`, `model_conf`, `lane_src`, `engaged` and pose.
- Painted-lane labels: `python/debug/lane_targets.py` (`tech_from_painted`), used by the collector for `gt.jsonl` and by `scripts/relabel_painted.py` offline for `gt_painted.jsonl` beside `state.jsonl`.

`--policy shadow` is not this gate. It computes the modular and e2e commands and, when engaged, actuates the modular one. The gate runs with engage hard-disabled.

## Not built yet

- Paint labels in the live log. The live GT log's `gt_center_x` / `gt_center_xs` come from the road network (`get_roads` edges), not from paint. Nothing writes the scene-net prediction and the paint label for the same live frame yet.
- A junction mask in the live log.
- A respawn / reset flag in the GT log rows.
- A grader for a shadow log.

## What the log must contain

Per frame:

- time, pose and map (west_coast_usa);
- the scene-net prediction and its confidence;
- the GT paint label for the same frame, in the same lateral frame as the model, with its valid mask (a slot with no paint is invalid, never scored as a centered lane);
- `engaged` false on every tick, with engage hard-disabled;
- mask flags, with the reason, for junction, respawn / reset, settle, impact and near-black frames, and for ticks where the scene input is held or refused (`scene/stop`).

Per run:

- the model file, its hash and its metadata (main camera only, no `gvd_ctl_cond`);
- the code commit;
- whether `GVD_LIVE_TONE` was set (it must not be);
- what drove the car during the run;
- counts of scored and masked frames, by mask reason.

No pass number is set here. A threshold has to come from a measured run, not from this doc.

## Preconditions: train and live must agree first

A shadow log does not count while these are open:

- #140: the scene runtime's `/255` rule (`max <= 1.5`) differs from the trainer for near-black uint8 strips.
- The live tone curve is still opt-in (`GVD_LIVE_TONE=1`) rather than gone. With it on, live frames are not the raw frames the scene net trained on.
- #150: the detector has no NMS after the 0.35 cutoff on the YOLOv8 ONNX path (overlapping boxes, blink near the threshold).
- #151: the tracker copies the detection position each tick, so box-height range wobble becomes metres and fake speed.

## After the gate, not before

The file bus is the real engage risk:

- #123: the mod writes `gvd_ego.json` non-atomically (torn reads).
- #143: the supervisor tick's `atomic_write_json` `open()` can stall 13-81 ms, which can exceed the heartbeat period.
- #130: the live supervisor tick rate vs the mod echo rate is not measured (gap-run disengage risk).
- #126: the Lua command-stream dead-man brakes at speed (brake 1 between 0.35 and 1.0 s).

That work is the PR after a boring shadow log, not before it, and not in a docs diff.
