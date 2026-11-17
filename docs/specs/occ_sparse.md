# Sparse occupancy net (`occ_sparse`), Swarm 3 / Redesigner

Branch `gvd/occ-sparse-s3`, off main `b098384`.

## Goal

Build a camera-only network that answers "what occupies this 3D point, of which class, moving how?" for a sparse set of query points around the ego. Training labels come from recorded depth, semantics and traffic boxes. Its outputs feed the C++ visualizer (PR #232) as perception.

## Design (first principles)

The net only has to answer for points we ask about, so nothing is dense in 3D.

- **Queries, not a volume.** A query is an ego-frame point `(x right, y forward, z up, level)`. The fixed rig (`config/cameras.yaml` via `lanes.camera_poses`) projects it into each camera with the same pinhole as `lanes.py`. Features are bilinearly sampled (`grid_sample`) from a stride-8 per-camera conv map. Cameras that can't see the point are masked out, and the visible ones are mean-pooled. A Fourier encoding of the point is added and an MLP outputs `occ` (1 logit), `sem` (7 logits) and `flow` (2, m/s). Memory is O(cameras × image) + O(queries). The only dense tensor is the 2D image feature map.
- **Grid** (for labels and the query lattice only): x [-25.6, 25.6], y [-12.8, 51.2], z [-1.6, 3.2], 0.4 m voxels. That is 128×160×12 = 245,760 fine cells. Coarse cells are 4×4×2 voxels, giving 32×40×6 = 7,680.
- **Coarse-to-fine inference** (`infer_sparse`): evaluate all 7,680 coarse cells, keep the top cells above `tau`, and evaluate only their children, capped at `max_fine` (8,192 by default). Queried cells are at most 7,680 + 8,192 = 15,872, which is **6.5% of the dense grid**. Stored output is the occupied subset only.
- **Where sparsity begins.** The image side is dense: one shared stride-8 backbone runs on all N cameras (8 × 80×45×48 floats at 640×360, the 1080ti profile). Everything after it is sparse. There's no 3D feature volume. 3D work starts at the query lattice, and the net only evaluates the cells it's asked about: ≤15,872 of 245,760, or 6.5%. The output keeps occupied cells only, typically a few hundred to a few thousand. Fine children of non-occupied coarse cells are never created. If more coarse cells pass `tau` than `max_fine` allows, the nearest ones (xy distance to ego) are kept, so the space behind a wall can't starve the near field.
- **Multi-camera fusion (AI Day style, in query space).** All cameras share one backbone. Each 3D query is projected into every camera, and the samples are fused by a weighted mean. A camera's weight is visibility × `cam_valid` × `exp(-age/0.3 s)`, and it's 0 when the frame is older than 1 s. That gives one fused feature per 3D point in the ego frame, which is the "single 3D/ego feature volume" evaluated sparsely instead of densely.
- **Temporal memory.** `FeatureMemory` keeps the last `mem_len` backbone outputs plus each frame's ego pose (world x, y, heading) and each camera's own capture time. Each query is moved into every past ego frame with `relative_xf(cur, past)` (a 2D rigid transform). It's sampled there with the same fusion, which down-weights stale cams per camera, and pooled with a learned weight per age slot. The MLP sees `[current, temporal, current−temporal, has_memory, n_cams, PE, level]`. A static world point is therefore looked up at the same world point in every past frame (G23 static-scene test, ego driving forward and turning). Motion (flow) is learnable only through memory (G22). Runtime runs the backbone once per tick and caches it, so there's no recompute of past frames.
- **Ego motion is camera-only by default.** The temporal warp and the tracker need ego motion. `SparseOccNet.vo` is a small visual-odometry head. It reads this tick's and the newest cached backbone maps, predicts (dx, dy, dyaw) per camera in that camera's axes, rotates that into the ego frame, and averages over fresh cams. `infer_sparse` integrates it into `FeatureMemory.odom` and warps memory with it. `vo_speed_yaw` turns it into the tracker's speed and yaw rate. The vehicle's own speed/yaw (or any external pose) is accepted only with `ego_from_vehicle=True` (`--ego-from-vehicle`, default off). With the flag off, an external pose raises. The recorded pose is a training label only: `train_occ` supervises the VO head with the pose delta (`vo_loss`). G24 trains VO on a synthetic static world and drives a held-out 30-tick clip with the flag off. It reports end drift in metres and the memory world-point error under the VO pose. G23 checks the warp maths with the exact pose.
- **Heads vs the AC.** The AC lists occ / lanes / objects heads. Here the trained heads are occ (1), sem (7, which includes lane-paint, drivable, curb and vehicle classes) and flow (2, auxiliary), plus the VO head. Lanes, edges and objects are derived from classified occupied cells in `occ_viz`. That's a deliberate deviation. All three outputs share one query decoder, so a cell can't be lane paint and a vehicle at once. Lane/curb labels are still UNLABELED or CONDITIONAL (see the table), so separate lane and object heads would train on nothing today. Once paint labels exist, a polyline lane head can be added on the same fused query features without touching the backbone.
- **Reduced-cam mode.** Cameras are a mask, not a code path. `cam_valid` zeroes a camera's weight. The `1080ti_3cam` (main+wide+narrow) and `1080ti_2cam` (main+wide) profiles build a net with only those cams and load the 8-cam weights. G23 shows that's numerically identical to the masked 8-cam net. If the live rig can't keep 8 cams in the same tick, drop resolution first (Michael's rule), then cams. The live 8-cam rate is UNPROVEN. The recorder runs 10/10/10/2/2/1/1/1 Hz, and per-cam age handles the slow ones.
- **Downstream never sees pixels.** `occ_viz` (and later the planner) take head outputs only: cell indices plus class logits. G16b checks three things: an import trace shows no torch, cv2, PIL or net module; no image parameter exists; and a run with poisoned `images`/`frames`/`feat` keys present still produces a valid v1 message.
- **No new dependencies.** It uses torch/torchvision, numpy and cv2, which are already in the repo. Shards use stdlib `tarfile`.

Trade-off against a sparse-conv or BEV design: there's no spconv/MinkowskiEngine build on Windows/Pascal and no dense BEV volume. The cost is that each query pools only one feature per camera, with no explicit depth reasoning, so fine geometry depends on the conv features and on many queries.

## Non-goals

- No changes to safety, actuation, AEB, heartbeat, engage, the supervisor control loop, or `scene_net` heads and ONNX.
- No live wiring into `run_vision.py` in this branch. The publisher hook is PR #232's. This branch provides `occ_viz.viz_state_from_infer` for it to call.
- No temporal fusion. Flow comes from a single frame: each tick is trained on box velocity.

## Files

| File | Side | What |
|---|---|---|
| `python/train/occ_sparse.py` | runtime | grid, `SparseOccNet`, `infer_sparse`, `export_onnx`, `memory_budget`. Imports no ground-truth module (G16). |
| `python/perception/occ_viz.py` | runtime | occupancy cells → v1 viz message (`occ_to_viz`), polylines, multi-frame `OccTracker`, colours. numpy only. |
| `python/train/occ_labels.py` | training only | depth/sem/boxes → sparse labels; `sample_queries`; `occ_loss` |
| `python/data/occ_shards.py` | training only | tar shards (WebDataset layout), offset index, recovery weights |
| `python/train/train_occ.py` | training only | `build` (recording → shards) and `train` (shards → net, `--export` ONNX) |
| `scripts/test_occ_sparse.py` | test | G1–G21 |
| `docs/specs/occ_sparse.md` | doc | this file |

## Labels (training ground truth only)

- **Sources.** Every camera that recorded `<cam>_depth` (uint16 cm, view-Z), optional `<cam>_sem` plus `sem_classes.json`, and `state.jsonl` traffic via `vehicle_labels._boxes`. Today only `main` and `wide` have `train_gt: true` in `config/cameras.yaml`. The label code takes any camera that has depth, so enabling depth on the other 6 cameras needs no code change.
- **Occupied** cells are voxelized depth hits. **Free** cells are ray samples before the hit, minus one voxel of margin, clipped to the grid. **Unknown** is never written, so it can never train as free. Depth 0 and 65535 are invalid (G2). Occupied wins over free (G4).
- **Semantics.** Names map to classes only through exact lists (`class_for_name`):
  - drivable = `DRIVABLE_CLASS_ALLOWLIST`
  - lane paint = `LANE_LINE_MATERIALS` without `line_reflectors`, yellow when the name says yellow
  - crosswalk, stop line and curb lists are **empty** until a `sem_classes.json` dump proves a name
  - anything else is ignore (255)
  - thin paint wins a voxel at ≥25% of its votes; otherwise the majority wins (G21)
- **Vehicles.** A box with length and width fills its oriented footprint. Without them (the recorder today), only a 0.8 m core disk is filled, never an invented 4.5×1.8 m. z runs 0.2–1.4 m. Flow = speed × heading in the ego frame and is NaN elsewhere (G6).
- **Format.** int16 index lists (`occ_idx`, `free_idx`), uint8 `occ_cls` and float32 `flow`, stored as npz inside the shard.

## Data: shards

`train_occ build` reads a recording once and writes `<rec>-NNNNNN.tar` files. Each sample has `<key>.<cam>.png` (the recorded bytes, not re-encoded), `<key>.occ.npz` and `<key>.json` (`rec`, `i`, `recovery`, `cams`). Shards roll over below `--max-shard-gb` (1 GB default, under the FAT32 4 GB limit). `ShardIndex` keeps tar byte offsets for random access. `OccShardDataset` opens only `.tar` files, and a test spies on `open()` with the recording deleted (G17).

## Recoveries (20–30%)

`is_recovery_row` is true when `tags.excursion.phase == "recover"` or `tags.recover is True`. `recovery_mix_weights` gives recoveries exactly `target` (0.20–0.30, default 0.25) of the expected draws. `make_sampler` wraps it in `WeightedRandomSampler` and prints the input and drawn fractions. G9 checks the weights, and G18 checks 8,000 real sampler draws land in 0.20–0.30 from a set with 1/12 recoveries. With zero recoveries the weights stay uniform and the print shows `0.000`. Nothing is faked.

## Viz outputs (Michael 11:24 PM AC; schema frozen at viz_cpp.md @ f3555bf)

`occ_to_viz(outputs, *, grid, t, seq, ego_speed_mps, yaw_rate, tracker) -> dict` in `python/perception/occ_viz.py` returns a `gvd_scene` v1 message as defined in `git show f3555bf:docs/specs/viz_cpp.md` (`v:1`, `type:"gvd_scene"`, `seq`, `t`, `ego`, `aeb`, `lanes`, `edges`, `tracks`, `signs`, `path`, `path_conf`; caps 64 tracks, 16 lanes, 8 edges, 120 pts per polyline). `validate_v1` checks those required fields and allows unknown keys, which is what the C++ parser does. `ego`, `aeb`, `signs` and `path` are neutral placeholders: the supervisor owns them, the occupancy net never does.

v1 fields filled from occupancy:

| field | content | derived how |
|---|---|---|
| `lanes[]` | `{pts, paint: white\|yellow, kind: "detected"}` | `lane_white` / `lane_yellow` cells → 8-connected BEV components → PCA axis → mean point per 0.4 m bin along the axis |
| `edges[]` | `{pts, kind: "detected"}` | per 0.4 m row, outermost `drivable` cell left and right of x=0 (row gaps ≥2 split polylines); also `curb` components → polyline |
| `tracks[]` | `{id, cls, x, y, yaw, l, w, v?}` | `vehicle` components → centroid + oriented extent per frame; `OccTracker` gives `id`, `v` and `yaw` (see below) |

Optional extras (not in v1, ignored by the parser, documented here): `lanes[].src="occ"`, `edges[].src = drivable|curb`, `tracks[].motion`, and `occ = {cells: [[x,y,z,cls]...] (≤4096), crosswalks: [{pts}] (closed rectangle of a crosswalk component), stop_lines: [{pts}] (polyline of a stop_line component)}`.

**Motion state comes only from tracks over time.** `OccTracker` associates per-frame vehicle detections by greedy nearest neighbour (3 m gate) against each track's predicted position, in an odometry frame integrated from ego speed and yaw rate, so a parked car stays still while the ego drives. Velocity = least-squares slope of the last ≤10 positions. A track gets `v` and `motion` only after ≥3 detections spanning ≥0.2 s; before that both are absent. `motion` is `parked` (<0.5 m/s), `same_way` (heading within ~45° of ego), `alongside` (same way, |x|>1.5 m, |y|<6 m), `oncoming`, or `crossing`. The net's `flow` head is a training auxiliary and is **not** used for motion.

**Colours** (`VIZ_COLORS`, `color_of`): white paint white, yellow paint yellow, red only for edges and curbs. Objects follow the City Streets legend (same_way green, alongside blue, oncoming amber, crossing purple) except parked, which is orange because red is reserved.

### Ground-truth source per output

| output | label source in recorder data | status |
|---|---|---|
| occupied / free | `<cam>_depth` PNG (cm, 0 and 65535 invalid) ray-cast through the rig | LABELED |
| `drivable` | `<cam>_sem` + `sem_classes.json` names in `sensor_aux.DRIVABLE_CLASS_ALLOWLIST` | LABELED |
| `vehicle` cells, flow | scene-row traffic boxes (`vehicle_labels._boxes`): pose, size, yaw, `speed_mps` | LABELED |
| `lane_white` / `lane_yellow` | sem palette names in `painted_lanes.LANE_LINE_MATERIALS` (minus `line_reflectors`) | CONDITIONAL: no palette dump yet shows those DecalRoad names in `sem_classes.json`; until one does, treat as UNLABELED |
| `crosswalk`, `stop_line`, `curb` | none found (name lists empty) | **UNLABELED**: the code paths exist and are tested on synthetic cells, but no claim is made |
| track `motion` | derived at runtime from tracks; evaluable against box `speed_mps`/yaw over frames | no trained head; tested on a synthetic multi-frame scene only |

## Hardware profiles (one architecture, config only)

`PROFILES` in `occ_sparse.py`, built with `build_profile(name)`:

| profile | width/hidden/depth | input | cams | voxel | mem_len | max_fine | precision |
|---|---|---|---|---|---|---|---|
| `1080ti` | 48/128/1 | 640 long side | 8 | 0.4 m | 2 | 8,192 | fp32 |
| `1080ti_3cam` | 48/128/1 | 640 | main, wide, narrow | 0.4 m | 2 | 8,192 | fp32 |
| `1080ti_2cam` | 48/128/1 | 640 | main, wide | 0.4 m | 2 | 8,192 | fp32 |
| `1080ti_mainwide_hi` | 48/128/1 | 960 | main, wide | 0.4 m | 2 | 8,192 | fp32 |
| `high` | 128/256/3 | 960 | 8 | 0.2 m | 4 | 32,768 | fp16 |

Knobs that keep weights loadable: input res, cams, voxel size and range, `mem_len` (up to 8), `max_fine` and precision. Rig, grid and position-scale buffers are rebuilt from config and not saved, and `age_w` is always 8 wide. A changed range rescales the position encoding, so it loads but wants a fine-tune. Knobs that force a retrain: width, hidden and depth. `load_compatible` raises `ValueError(... retrain ...)` naming the mismatched tensors. G23 tests both directions.

VRAM budget: BeamNG's measured free VRAM minus 1 GB. At ~4 GB free that's 3 GB, and the `1080ti` profile must fit it. G23 asserts the CPU-hook estimate is under 3 GB. The real number must come from IQU45 with BeamNG running.

Pascal (GTX 1080 Ti) runs fp16 math at about 1/64 rate, so fp16 there saves memory, not time. Any fps claim must sit next to its VRAM figure and come from IQU45 with BeamNG live. The real speedup path is int8 via TensorRT, which needs a calibration set (about 500 recorded frames across lighting and scenes) and a reported accuracy delta (IoU and recall vs fp32). That's NOT implemented here: `precision: int8` is a declared knob only and UNPROVEN.

## Budget

`memory_budget()` is a **CPU estimate** (all CPU memory numbers here are estimates), including the temporal branch and cached memory frames: forward hooks sum activation bytes plus parameters and Adam state, at fp32 and halved for fp16. It is not a 1080 Ti measurement. The real numbers come from running `train_occ train` on Desktop-IQU45 (`peak_vram_gb` is printed per epoch) when the GPU is free.

AMP: `--amp auto` uses fp16 autocast plus GradScaler on CUDA. AGENTS.md keeps Pascal at fp32 for `scene_net`. On a 1080 Ti, fp16 saves memory but its math is slow, so `--amp off` is the fallback. Measure both.

## Verify

```
PATH=<venv>/bin:$PATH python scripts/test_occ_sparse.py
PATH=<venv>/bin:$PATH xvfb-run -a python scripts/run_offline_suite.py
py -3 -m python.train.train_occ build --recording <rec> --out <nvme>\occ_shards
py -3 -m python.train.train_occ train --shards <nvme>\occ_shards --epochs 1 --export <dir>\occ.onnx
```
