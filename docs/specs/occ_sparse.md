# Sparse occupancy (attempt s2, "detective": measure first)

Branch `gvd/occ-sparse-s2`, base `main` b098384. Runtime is camera-only; depth,
semantics and vehicle boxes are training ground truth only.

## Goal

A sparse 3D occupancy net for GVD that fits what the recorder, the loader and the
GTX 1080 Ti (shared with BeamNG) can actually carry, and that feeds the C++
visualizer (`gvd_scene` v1, `docs/specs/viz_cpp.md` @ f3555bf) through
`occ_to_viz(outputs) -> dict`.

## Non-goals

- No change to safety, actuation, AEB, `drive_*`, the live control net, ONNX
  exports, `gt.jsonl` / `lane_gt` packing or `sensor_aux`.
- No dense voxel head (not even masked). No per-cell flow head. No single-frame
  motion guess.
- No lane-paint colour, crosswalk or stop-line head in this branch (see
  "Outputs and their labels": UNLABELED / not exported).
- No recorder change in this branch (recorder findings are listed, not applied).
- Not wired into `run_vision` yet: nothing on the live path imports it.
- No dense fused grid anywhere, not even as a baseline (see "Where sparsity begins").

## Measurements that drove the design

All taken on the shared Linux box `grok-bot-vm-419429730` (8 vCPU Xeon, **no GPU**,
torch 2.14.1+cpu) on 2026-11-16 23:35-23:37 CST while ~7 other suites were running
(loadavg 28.9-30.5 on 8 cores). Each run is logged with `date` and
`ps -eo pid,pcpu,rss,etime,cmd --sort=-pcpu | head -15` before and after
(`probes/measure.sh`, logs in the report). CPU timings are inflated by that load;
**every VRAM figure below is an estimate** (activation-byte count), the real numbers
come from `scripts/bench_occ_sparse.py` on Desktop-IQU45.

| What | Number | Source |
|---|---|---|
| Rig rates (`config/cameras.yaml`) | narrow/main/wide 10 Hz, pillars 2 Hz, repeaters/rear 1 Hz = 37 frames/s | `probe_recorder.py` |
| RGB loose PNG (level 1), 640x480 | 401 KB/frame, encode 29 ms, decode 14 ms | 20 real frames |
| Depth u16 PNG / sem u8 PNG | 98 KB enc 11 ms / 18 KB enc 3 ms | real fixture 037126 |
| Recording volume at rig rates | RGB 46.0 GB/h; GT main+wide +6.8 GB/h; GT all 8 cams +13.4 GB/h | computed from the above |
| Recorder PNG CPU at rig rates | RGB 1.0 core-s/s; GT all 8 +0.48 core-s/s | same |
| Loose files/h | 277k (RGB + GT 2 cams), 400k (RGB + GT 8 cams) | same |
| Label build, real main frame | stride 1: 4680 cells, 293 ms; **stride 2: 3201 cells (68%), 38 ms**; stride 4: 1702, 7 ms | `probe_labels.py` |
| Stored label sparsity | 1 cam 0.98% of the dense grid (stride 2); 8-cam proxy 3.5% | same |
| Dense grid | 128 x 160 x 16 = 327,680 cells (0.4 m, x +-25.6, y -12.8..51.2, z -2..4.4) | `OccGrid` |
| Coarse-to-fine oracle, 4x then 2x | 10% (1 cam) / 33% (8-cam proxy) of dense queried | `probe_c2f.py` |
| Encoder cost 8 cams @640 | 7.28 GMAC, feature maps 8.8 MiB fp32 | `probe_flops.py` |
| Head cost | 30.5 kMAC/query; 111k queries = 3.4 GMAC | same |
| Net size | 165,542 params | `bench_occ_sparse.py` |
| CPU inference 8 cams @640 (untrained, caps hit) | 21.3 s median, 4 threads, 23:35:51-23:40:08 CST, load 30-37 (meaningless for the GPU, recorded only) | bench |
| CPU train step b4 8 cams, workers 0 | loader wait 0.38 s vs compute 22.7 s median; peak RSS 1.85 GB | bench, same run |
| Shard build (labels from 8 cams) | 3.4 s/sample on the loaded box | bench, same run |
| Activations at 8192 queries, fp32 (estimate) | 427 MB | bench |
| Shard vs loose bytes/sample (bench, synthetic) | 233 KB vs 143 KB | bench |

What the numbers say:

1. **Labels must be offline, not in the recorder.** Back-projection at stride 1 is
   ~0.3 s/frame; stride 2 keeps 68% of the cells at 13% of the time. The recorder
   is already at ~1.5 core-s/s of PNG encode at rig rates, so labels are built once,
   offline, into shards (`python -m python.data.occ_shards`).
2. **Loose PNGs are the loader cost.** A training sample is 8 RGB + labels = 8
   PNG decodes (~14 ms each) plus 8 depth + 8 sem decodes if labels were rebuilt.
   Shards hold the RGB PNG bytes and the already-built sparse labels (`.npy`: keys int64,
   class u8, free points f16), so a sample is 8 RGB decodes and three small `.npy`
   loads, read by byte offset from one tar. One file per ~1 GB instead of ~400k
   files/h. Cost: the bench sample is 233 KB in a shard vs 143 KB of loose
   RGB+depth+sem PNGs (labels + tar headers; synthetic images, measured once).
3. **The head is the VRAM driver, not the encoder.** 8 feature maps are 8.8 MiB;
   activations scale with query count. So the head is capped per stage
   (`max_points`) and coarse-to-fine (4x, then 2x, then fine), and training samples
   a fixed query budget (1024 occupied + 1024 free + 512 coarse) instead of the grid.
4. **Real class names.** The real main frame is 62% `STREET`, which the shared
   `sensor_aux.DRIVABLE_CLASS_ALLOWLIST` does not contain: with it alone a real
   recording has **zero drivable cells and no road-edge label**. Occupancy uses its
   own allowlist (`OCC_DRIVABLE_ALLOWLIST` = that list + `STREET`,
   `DRIVING_INSTRUCTIONS`); `sensor_aux` is untouched. On the real frame this gives
   1601 drivable / 1600 other cells, and the derived right road edge sits at
   x = 6.0 m (p10-p90 4.4-6.8 m) over y 19-42 m.

## Design

- **Grid** `OccGrid` (`python/perception/occ_geometry.py`): index space only, cell
  key = flat index; nothing allocates a dense array.
- **Labels** `build_occ_labels` (`python/train/occ_labels.py`, training only): every
  camera's recorded depth + semantic back-projected with the `cameras.yaml` pose
  (stride 2), far-plane hits are free rays, depth 0 is invalid, ego clear box
  dropped, rays clipped to the grid for free samples. Vehicle boxes from the traffic
  rows: a sized row fills its box, an unsized row only relabels observed cells
  within 2 m. Classes: `other`, `drivable`, `vehicle`.
- **Shards** `python/data/occ_shards.py`: WebDataset-style `.tar` with a sidecar
  `.idx.json` of byte offsets; `OccShardDataset` reads only shards and refuses a
  folder without them (`ShardOnlyError`).
- **Sampler** `RecoveryMixSampler` (`python/train/occ_sampler.py`): exactly
  `k = recovery_count(batch, ratio)` recovery samples per batch, ratio must be in
  [0.20, 0.30], refuses an empty recovery pool. Recovery = excursion phase
  `recover`/`abort` or the collector's `recover` tag.
- **Net** `SparseOccNet` (`python/perception/occ_sparse.py`, runtime): shared small
  CNN per camera (32/64/96 ch, stride 8, 64-ch features), each 3D query projected
  into every camera, bilinear features averaged over the cameras that see it, plus
  a positional encoding, through a 128-wide MLP to occupancy + class + "any occupied
  in this coarse cell". `predict` runs 4x coarse, 2x coarse, then fine only under
  hot parents, with a per-stage cap.
- **Trainer** `python/train/occ_train.py`: BCE occupancy, BCE coarse-any, CE class on
  occupied queries; AMP fp16 with GradScaler (`--amp fp16` forces it on Pascal).

### Where sparsity begins

Sparse end to end. The only dense tensors are the per-camera **2-D** image
feature maps (64 ch at 1/8 of the input; 8.8 MiB fp32 for 8 cams at 640). There
is no BEV plane and no 3-D volume. Fusion happens **at the sparse query points**:
each 3-D point is projected into every camera and the bilinear features are
averaged there (`SparseOccNet.query`). `predict` queries the visible 1.6 m
coarse cells (5,044 of 327,680 for 8 cams), descends only under hot parents, and
caps each stage (absolute caps below). Stored labels are a sorted key vector (8-cam proxy 3.5% of the
grid, one real camera 1.0%). `test_occ_cams.check_sparsity_begins` hooks every
head layer and asserts no 3-D batch has more than `max_points` rows (worst case
all-hot: largest batch 16,384 rows vs 327,680 dense; 37,812 points queried in
total = 11.5% of dense); `test_occ_labels` prints stored vs grid cells. There is
no dense baseline in this branch, so no crossover is claimed: a dense head would
query 327,680 points per frame against 5,044 (all cold) to 37,812 (all hot,
capped) here.

Every stage has an absolute cap (`1080ti` profile: `max_coarse` 2,048 on the top
stage, `max_points` 16,384 top-k per later stage). The top stage queries last
frame's hot top cells first (sparse temporal carry, <= half the cap) and then a
rotating stride subset of the visible coarse cells, so each one is revisited
within ceil(visible / cap) frames (tested: cap 1,000, 5,044 cells swept in 6
frames). There is no ray x depth-bin lifting at all: 3-D points are queried, and
each later stage keeps only the top-k children by probability above the
threshold.

Per stage at 8 cameras (fine frustum = 322,331 cells visible to some camera at
the `1080ti` rig; full grid 327,680), `test_occ_cams.check_stage_counts`,
untrained head forced cold / random / all-hot, first frame:

| Stage | cap | cold | random head | all-hot |
|---|---|---|---|---|
| fusion + query, 4x coarse (1.6 m) | 2,048 | 1,682 | 1,682 | 1,682 |
| fusion + query, 2x coarse | 16,384 | 0 | 13,456 | 13,456 |
| fusion + query, fine (0.4 m) | 16,384 | 0 | 16,384 | 16,384 |
| head output stored (cells) | <= fine | 0 | 14,567 | 16,384 |
| temporal memory: VO static BEV keys | <= stored | 0 | 5,244 | 3,072 |
| temporal memory: carried hot top cells | 1,024 | 0 | <= 1,024 | <= 1,024 |
| temporal memory: tracker | 64 x 10 | 0 | 0 tracks | 0 tracks |

Every post-backbone tensor is counted with a `TorchDispatchMode` over all aten
ops after the encoder (1,749 op outputs in that run): largest dimension 16,384
(the cap), largest numel 2,097,152 (16,384 rows x 128 hidden); no tensor has a
grid- or frustum-sized dimension. The candidate list of visible top cells is
built once per camera set (enumerating the 5,120 coarse indices), cached, and is
an index list, not a feature volume. Temporal memory is the carried hot top-cell
indices, the previous frame's sorted static BEV keys for VO (no raster:
`np.searchsorted`), and the tracker's track list. No array the size of the grid
(3-D or BEV) is allocated at runtime.

### Cameras: presets, per-camera age, missing cameras

GVD reports the recorder at 8 cams 10/10/10/2/2/1/1/1 Hz, 9.89 Hz overall, 0
drops over 2 h (GVD-reported, used as the design input); the live 8-cam rate at
the planner tick is unproven. So:

- `CAM_PRESETS`: `min` = main+wide (minimum), `front3` = main+wide+narrow,
  `all8`. One set of weights serves every preset (the encoder is shared).
- `cam_weights(cam_ids, preset=, ages_s=)`: each camera carries its own frame
  age (now minus its own timestamp; the live supervisor already has
  `python.data.cam_hold.ages_vector`). Weight `exp(-age/0.5 s)`, 0 past 1.5 s, 0
  for a missing/NaN/negative age. Fusion is the weighted mean over the cameras
  that see a point; a zero-weight or `None` camera is not even encoded.
- Training draws the same: main+wide always weight 1, each other camera dropped
  with p = 0.3 or aged uniformly in [0, 1.5 s] (`train_cam_weights`, `--cam-drop`).
- Tests (`test_occ_cams`): both presets give the same outputs as the same
  weights built on only those cameras; at t = 0.95 s of the rig pattern main
  weighs 0.90, pillarL 0.41, rear 0.15; a fully stale camera's pixels change
  nothing; a partly stale camera counts less than a fresh one.

### Profiles (GTX 1080 Ti shared with BeamNG)

Pascal fp16 math is ~1/64 of fp32: **fp16 saves memory, not time**; no fp16
speed claim is made. All profiles compute in fp32 (`--fp16` in the bench is a
memory experiment only). int8 via TensorRT (DP4A) is the only precision speed-up
on this card; it is **not implemented** (needs a calibration set and an accuracy
delta vs fp32, both UNMEASURED; would add TensorRT as a dependency).

| Profile | Net | Cameras @ input | Params | Memory ESTIMATE fp32 (MB): weights + feature cache (K=1) + encoder peak + queries = total |
|---|---|---|---|---|
| `1080ti` | default (32/64/96, feat 64, hidden 128) | all8 @ 640 | 165,542 | 0.6 + 8.8 + 14.1 + 276.8 = 300.2 |
| `front3` | default | narrow, main, wide @ 640 | 165,542 | 0.6 + 3.2 + 14.1 + 276.8 = 294.7 |
| `front_cluster` | default | main @ 800 + wide @ 640 | 165,542 | 0.6 + 2.7 + 22.0 + 276.8 = 302.1 |
| `1080ti_cheap` | cheap (16/32/48, feat 32, hidden 64) | front3 @ 480 | 43,350 | 0.2 + 0.9 + 4.6 + 156.8 = 162.4 |
| `1080ti_min` | cheap | main+wide @ 320 | 43,350 | 0.2 + 0.3 + 2.1 + 156.8 = 159.2 |

These are arithmetic (`memory_estimate`), not measurements, and exclude the
CUDA context and allocator slack. Temporal cache term is
K x cams x C x H/8 x W/8 x bytes with K = 1: the net is single-frame; temporal
state is the tracker (<= 64 tracks x 10 positions). A profile only "fits" after
GVD posts `nvidia-smi` from Desktop-IQU45 with BeamNG running, minus 1 GB. The
query term dominates and scales with `max_points` (65,536 default), so the
cheapest memory knob is `max_points`, then the camera count.

Michael's rule: drop input resolution before giving up same-tick camera sync.
Input resolution is a bench knob (`--long-side`, `--sweep-long 640,480,320`).

#### front_cluster vs front3 (measured, `scripts/compare_front_cluster.py`)

Run 2026-11-17 00:09:31-00:24:08 CST on the loaded CPU box (loadavg 23-35 on 8
cores; `ps` logged before/after). Same 16 synthetic train clips, same 8 held-out
clips (one box 10-48 m ahead, |x| <= 4 m), seed 0, 150 steps, batch 4. Trained
at 1/8 of the profile sizes (the 800/640 ratio kept). One seed, few far cells:
**the differences are within noise; neither rig is shown better.**

| Rig | Input (trained / full) | Params | Mem ESTIMATE full size | Held-out query IoU | Vehicle recall < 30 m (159 cells) | >= 30 m (86 cells) |
|---|---|---|---|---|---|---|
| front_cluster | main 100x75 + wide 80x45 / main 800x600 + wide 640x360 | 165,542 | 302.1 MB | 0.861 | 0.447 | 0.547 |
| front3 | narrow, main 80x60 + wide 80x45 / 3 cams @640 | 165,542 | 294.7 MB | 0.855 | 0.409 | 0.593 |

Train loss 1.667 -> 0.102 (front_cluster) and 1.667 -> 0.106 (front3). The real
question (narrow's 800 m far plane vs a sharper main at range) needs recorded
held-out clips at full size on IQU45; the script is ready for it once
shards of real clips exist.

### Checkpoints: what stays loadable

`save_checkpoint` stores the weights plus `net_spec` (channels, feat, hidden,
levels, classes, PE). `load_checkpoint` is strict.

- **Loadable** (no retrain; tested, forward runs): input resolution, camera
  count/preset/ages (shared encoder; camera poses and intrinsics are
  non-persistent buffers), grid range and voxel size (queries are continuous
  points; the head is resolution-agnostic, though the PE normalises by the grid
  box so a different box shifts the encoding and wants a fine-tune), tracker
  frames (`MIN_FRAMES`, not weights).
- **Retrain** (refused with `CheckpointMismatch` naming the knob, nothing
  loaded): encoder width (channels), depth (number of stages), `feat`,
  `hidden`, the `cheap` config, levels, class count.

### Ego motion (temporal warp)

Runtime default is **camera-only VO** (`python/perception/occ_ego.py`): the
static non-drivable, non-vehicle BEV cells of the previous prediction are
registered onto the current one by a coarse-to-fine search over forward travel
(-1..4 m/frame) and yaw (+-0.15 rad/frame). `--ego-from-vehicle` (default
**OFF**, GVD ruling) switches to the car's own wheel speed + IMU yaw rate, read
only through `VEHICLE_SIGNAL_KEYS = (wheelspeed_mps, imu_yaw_rate)`; the BeamNG
mapping (`electrics.wheelspeed`, an IMU gyro z) is **for GVD to confirm**.
BeamNG GT pose, scenetree, `state.jsonl` ego rows: training labels only.

Measured on the synthetic static world (ego 15 m/s, 0.2 rad/s, 10 Hz, flag
off): per-frame forward error <= 0.10 m, yaw error < 0.0025 rad; pose drift
0.10, 0.10, 0.20, 0.15, 0.15, 0.20, 0.25, 0.25, 0.30 m over 13.5 m driven
(2.2%); 99% of static cells land on their true world cell (+-1) every frame vs
1% with no warp. Limitation: a featureless straight road has no forward
constraint; the estimate is held and flagged `motion_ok = False`.

`occ_odometry` votes only on *observed* static cells (`prev_obs` / `cur_obs`;
default all observed). Unobserved `decode_unseen` clutter cannot vote, so
there is no residual-VO path. The score's denominator is the previous
observed cells that, after the hypothesis warp, fall in the current
observed region (this tick's live/fresh cameras, with this tick's
occluders), so a side or rear camera going stale does not collapse the
overlap and trigger a hold. If fewer than `VO_MIN_CELLS` of those are
visible, or the score is below `VO_MIN_SCORE` (0.4), the last estimate is
held, `ok` is False, and `OccMemory` is cleared (no blind carry).
`check_ego` requires 11/11 ticks `ok` at the recorder's camera rates, with
all cameras fresh, and with only the front three fresh. A parked ego with
1-3 voxel smears, whole-wall depth jumps, moving objects, or lateral smear
keeps `(0, 0)` (`ok=True`); slow real motion on observed cells stays within
one search step (0.2 m). `OccRuntime` computes `observed_mask` from this
tick's live cameras and occluders before `ego.update` and passes the mask,
those cameras, and the occluders in so the score uses the same observed
region. Tests: `check_vo_clutter`,
`check_vo_parked_flicker`, `check_vo_parked_range_flicker`,
`check_vo_parked_structured`, `check_vo_slow_motion`,
`check_vo_runtime_obs`. The older residual share-gate tests and mutants
N3/N4 no longer apply (the residual mutation sites are gone).

Limitation (same as base, not fixed here): *observed* ego-fixed clutter
still votes. A dense self-occluded sheet is hidden by `observed_mask` and
does not lock VO, but sparse observed floaters can (e.g. 800 cells at
z 0.6-3 m: obs-VO 0.00 m at true 1.5 m, `ok=True`, score 0.67). The
claim is only that unobserved `decode_unseen` clutter cannot vote.

Tests: `test_occ_ego` (VO, observed-only clutter, parked structured probes, slow motion, world-fixed cells with the flag off, non-identity
warp, drift printout), `test_occ_runtime_imports` (no GT-pose module reachable
from the occupancy runtime, no pose-reading identifier in its source, beamngpy
not loaded, flag defaults off, a spy mapping proves the default path reads no
vehicle key and the flag path only the two signals, and a guard that raises on
any camera-array access runs through `runtime_frame -> occ_to_viz -> encode ->
planner stub`).

### Heads vs the AC (deviation)

The net has explicit **occupancy** (+ coarse "any") and **semantic class**
heads. Road edges and objects are **derived** from occupancy (edges: drivable
-> other transitions; objects: vehicle clusters) and motion comes from the
tracker. There is no lane / paint / crosswalk / stop-line head. Reason: the
per-frame labels for them are not packed in this branch (paint material and
`stoplines.jsonl` exist offline only), and an untrained head would be a head
trained on nothing. `lanes` pass through from the existing lane head when the
caller supplies them; paint colour stays UNLABELED.

## Outputs and their labels (viz contract)

`occ_ego.runtime_frame(pred, tracker, ego, t)` (= `EgoMotion` -> `occ_outputs`) -> `occ_viz.occ_to_viz` -> `gvd_scene` v1. Every v1
required key is always present (`v, type, seq, t, ego, aeb, lanes, edges, tracks,
signs, path, path_conf`); the C++ parser ignores the optional keys below.

| Output | v1 field | Runtime source (camera-only) | Ground-truth source (recorder) | Status |
|---|---|---|---|---|
| Sparse occupied cells + class | optional `occ: {voxel, origin, classes, cells:[[ix,iy,iz,cls]]}`, nearest first, <= 2500, halved until the datagram fits 60 kB | `SparseOccNet.predict` | `raw/<cam>_depth` + `raw/<cam>_sem` all 8 cams + `sem_classes.json` + traffic rows (`build_occ_labels`) | trained head |
| Road edges | `edges[]` `{pts, kind:"predicted", class:"road_edge", color:"red", side, source:"occ"}` | `road_edges`: per 0.4 m row, the drivable run nearest x=0 (gaps <= 6 cells bridged), boundary only where the next cell out is an observed non-drivable cell, 5-row median | drivable/other cells from the same depth+sem labels; synthetic-scene test checks positions | derived from occupancy |
| Curbs | `edges[]` `class:"curb"` | adapter accepts `curbs` | none: no curb class proven in the palette (`labels_v4.md`, curbs) | **UNLABELED, not emitted** |
| Lane lines + paint | `lanes[]` `{pts, paint, color}` | pass-through of an existing lane head when the caller supplies one; paint is `yellow` only if the record says yellow, else `white` | paint materials `line_white`/`line_yellow`... in `painted_lanes.LANE_LINE_MATERIALS` exist, but no head in this branch predicts colour | **paint colour UNLABELED** (no runtime head) |
| Crosswalks / stop lines | optional `crosswalks[]`, `stop_lines[]` `{pts, color:"white"}` | adapter accepts them if a caller supplies them | `stoplines.jsonl` (`scripts/label_stoplines.py`) exists offline | **not exported**: no runtime head |
| Objects | `tracks[]` `{id, cls, x, y, yaw, l, w, h, v, motion, color}` | `objects`: 8-connected BEV clusters of vehicle cells, PCA footprint/heading (`yaw_src: pca`), velocity heading once tracked (`yaw_src: track`) | traffic rows (`tags.traffic`) + vehicle sem cells | derived from occupancy |
| Motion state | optional `tracks[].motion` = `same_way|oncoming|crossing|alongside|parked|unknown` | `OccTracker`: nearest-neighbour association (3 m gate, 2 misses), ego-motion compensated (camera VO by default), least-squares velocity over >= 3 frames; `unknown` before 3 frames | per-tick traffic rows positions over time (offline, for evaluation) | multi-frame tracker, never one frame |

Motion rule (`motion_state`, ego frame, world velocity after ego-motion
compensation; angle 0 = ego heading): speed < 0.7 m/s -> `parked`; angle <= 35 deg
-> `alongside` when |x| > 1.8 m and |y| < 6 m, else `same_way`; >= 145 deg ->
`oncoming`; 55-125 deg -> `crossing`; the bands between go to the nearer of
same_way / oncoming.

Colours (Michael's rule wins over the legend): white paint white, yellow paint
yellow, **red only on `edges[]` (road edges and curbs)**. Tracks: same_way green,
oncoming yellow, crossing purple, alongside blue, parked orange, unknown blue.
Crosswalks and stop lines white.

## `occ_out` v1 (runtime output contract, `python/perception/occ_out.py`)

`OccRuntime.step(images, t, ages_s=..., vehicle=None)` runs net -> camera VO ->
tracker -> outputs and returns one **validated** `occ_out` dict per tick.
`validate_occ_out` **raises** `OccOutError` on any unknown field at any level
(new fields => `v` bump), a missing field, NaN/inf, wrong type/shape, a
probability or confidence outside [0, 1], an unknown class / paint / motion /
camera status / ego source, `stored <= queried <= full` broken, a count that
disagrees with its array, a non-unit heading, or any array over its cap. Only
`occ_viz.occ_to_viz` (viz v1 side) tolerates extra input keys, and it **drops
and counts** them (`occ_viz.DROPPED`, one log warning per key) instead of
passing them through. `occ_out_to_viz_inputs(msg)` feeds `occ_to_viz` with zero
dropped keys.

| Field | Content | Cap | Debug panel |
|---|---|---|---|
| `v, type, seq, t` | 1, `"occ_out"`, tick counter, frame time (s) | - | graphs (x axis) |
| `grid` | `voxel_m`, `origin[3]`, `shape[3]` | - | 3-D grid |
| `counts` | `full` (grid cells), `frustum` (fine cells seen by the active cams, counted once per camera set in 64k chunks, cached), `queried`, `stored`, `stage_queries` (4x, 2x, fine), `cells_out`, `objects`, `edges` | - | HUD |
| `cells` | `ijk[[i,j,k]]`, `cls` (0 other, 1 drivable, 2 vehicle), `prob`; nearest-first | 4096 | 3-D grid |
| `edges[]` | `side` left/right, `kind` road_edge (curb UNLABELED, never emitted), `pts[[x,y]]`, `conf[]` per point = occupancy prob of the drivable boundary cell | 8 x 120 pts | 3-D grid red dots, top-down |
| `lanes[]` | `paint` white/yellow, `pts`, `conf` - **always empty**: paint UNLABELED, no runtime head | 16 x 120 | 3-D grid lane-dot chains (none drawn) |
| `corridor` | `pts` drivable polygon, ego frame, implicitly closed (first vertex not repeated), non-self-intersecting (validator checks), `conf` | 64 verts | top-down |
| `objects[]` | `id` (track id), `cls` vehicle, `x, y, z0, l, w, h`, `heading [cos, sin]`, `heading_src` box/velocity, `motion`, `speed_mps`, `frames` | 64 | top-down numbered arrows |
| `boxes2d{cam}[]` | `id`, `cls`, `box [u0,v0,u1,v1]` px | 32 / cam | camera panels |
| `cams{8}` | `status` LIVE/STALE/ABSENT, `age_s`, `weight` (fusion weight), `res [w,h]`; ABSENT = not in preset / no frame / no age, with `age_s = res = None`, `weight = 0` (never zeros-as-data) | 8 | camera panels |
| `ego` | `speed_mps`, `yaw_rate_rps` (left/CCW +), `source` vo/vehicle, `ok`, `dt_s` | - | HUD, speed + yaw-rate graphs |
| `timing_ms` | `predict`, `outputs`, `total` from in-code `perf_counter` | - | HUD |
| `peak_vram_mb`, `vram_status` | `None` / `"UNPROVEN"` until GVD posts nvidia-smi from IQU45; a number is only valid with `vram_status = "MEASURED"` | - | - |
| `vram_est_mb_cpu` | arithmetic CPU-box estimate (labelled), separate field, never copied into `peak_vram_mb` | - | - |

`PANEL_FIELDS` maps each of the 4 panels of `debug_view_multipanel.png` to the
fields above; `test_occ_out.check_panels` asserts every path resolves. The
renderer itself is the C++ viz (PR #232); this round scores the contract.

Decisions the Skeptic asked to state:

- **Ego speed / yaw rate**: `source = "vo"` (camera VO) unless
  `--ego-from-vehicle` (`"vehicle"`, wheel speed + IMU yaw rate). Rates are
  per-frame VO motion divided by `dt` from the frame timestamps (no fixed
  0.1 s). VO now ignores static cells at road-surface height (>= 1 voxel above
  the median drivable z): flat grass/pavement is identical at every
  along-track shift and pulled the forward estimate short (measured 0.35 m on a
  0.6-1.2 m frame before; 0.25 m worst after; `test_occ_ego` forward error 0.10 m worst before and after).
- **Heading** is one field, `heading = [cos, sin]`: the box (PCA) axis
  (`heading_src = "box"`, +-pi ambiguous) while the track is stationary or
  younger than 3 frames, the track velocity direction once speed >= 0.7 m/s
  (`"velocity"`). No separate `yaw`.
- **Per-camera 2-D boxes** are **projections** of the 3-D object boxes (8
  corners, points in front of the near plane), clipped to the image, same id
  and class as the 3-D object, only for LIVE cameras. No separate 2-D head, so
  no extra VRAM.
- **Corridor**: rows where both road edges exist, resampled to <= 32 rows; up
  the left edge, back down the right. `conf` = edge support over the shared
  span x share of rows with left < right - 0.4 m. Below 0.5, one side only, or
  < 1 m shared: `pts = []` (no default polygon).

Measured (`test_occ_out`, CPU box, synthetic world, ego 12 m/s turning left
0.1 rad/s at irregular dt 0.05-0.2 s): |speed*dt - |(dx,dy)|| <= 0.25 m per
frame, |yaw_rate*dt - dyaw| <= 0.005 rad; mean 11.52 m/s (true 12.0), 0.115
rad/s (true 0.1). Corridor vertices within 0.20 m of the true road edge, 64
verts, conf 0.98. Message 71 kB; 30k stored cells on the default grid and on a
64x finer grid both cap at 4096 cells out (88 kB vs 84 kB, cap 200 kB).

## Unseen road structure (occluded / out-of-view), `occ_out` v2

Goal: predict road structure no camera ray reaches this tick (cross legs
behind corner buildings, the back of a roundabout, corners hidden by cars).

**Labels** (`python/train/occ_map_labels.py`, training only): the ground layer
of the WHOLE grid is labelled from map GT, not masked to camera-visible
pixels: `drivable` = cell centre within width/2 of a road centerline, `other`
elsewhere. Map source: the recorder road dump (`roads/<map>.json.gz` ->
`road_dump.load_road_dump` -> `RoadPoly` centerline/width/z) + the tick's ego
pose row, both sim truth (training labels only; `occ_map_labels` is on the
GT-module list of `test_occ_runtime_imports`). `merge_targets` = camera hits +
map ground layer (map wins on the ground layer; a vehicle cell stays vehicle).
Road outlines (centerline +-width/2, junction mouths removed) are the
edge-chamfer GT. `visible_keys` ray-casts the 8 cameras' depth (pixel stride
2-4) and keeps every cell a ray crosses up to its hit; it is used **only** to
split metrics visible / occluded, never to mask the loss. Lanes: no map lane
label is packed (paint UNLABELED), so no lane target and no lane chamfer.

**Loss**: every map cell is a target (`occ_w = 1`, class set) whether a ray
reached it or not. `sample_queries(..., visible_only=True)` reproduces the old
camera-masked loss and exists for the trace test only.

**Model**: (1) `ctx`: each query also gets a per-tick scene vector (cam_w
weighted mean of the cameras' pooled feature maps, B x feat), so a query no
ray reaches decodes from the scene, not from its position alone; (2)
`decode_unseen` (default on): the top coarse stage sweeps the whole top grid,
not only frustum cells, same `max_coarse` cap. Both change the head input
width (checkpoint `spec.ctx`), i.e. retrain. (3) **Temporal memory**
(`occ_out.OccMemory`, causal): last tick's static cells (vehicles excluded)
warped into the current frame by the camera-VO motion, prob x exp(-dt/2 s),
dropped when a camera sees the cell now (current prediction wins) or prob <
0.35, cleared when VO is not ok; capped at 16,384 cells (plain ground last).

**occ_out v2**: v1 + exactly one field per element type, `observed: bool` per
cell (`cells.observed`) and per polyline point (`edges[].observed`,
`lanes[].observed`): true when a LIVE camera's ray reaches the point with no
predicted occluder (non-drivable cell >= 1 voxel above the road surface) in
front. Memory-carried and unseen-decoded cells are `observed = false` and
still carry finite `prob` / valid `cls`. Validator is version-gated (v1 may
not carry `observed`, v2 must; unknown keys still raise). v2 also allows
`stored <= queried + 16,384` (memory cells). Cell cap order: plain ground
(class other at road height) last, then nearest first. occ -> viz v1 mapping
is `occ_out.VIZ_DELTA` (`observed`, `prob`, `conf` are not in viz v1 and are
dropped there); `python/viz/occ_viz.py` and the C++ viz are not changed by
this part.

**Window**: unseen proposals are the occupancy grid itself (x +-25.6 m,
y -12.8..51.2 m, ego frame); occluded metrics are inside it; the eval reports
per clip the share of GT centerline length within 80 m that lies beyond it.

**Causality**: `OccRuntime.step(t)` reads only tick t's images and state from
earlier ticks (VO, tracker, memory); nothing is smoothed backwards or revised.
`test_occ_unseen.check_causality`: frames after t perturbed (junk cells /
grey images) or deleted -> every occ_out at <= t byte-identical (timer
injected so `timing_ms` is reproducible), stub net and a real SparseOccNet.

**Safety isolation**: `test_occ_runtime_imports.check_safety_isolation`: the
12 actuation / AEB / engage / heartbeat modules (`python.run_vision`,
`python.control.*`, `python.runtime.{conf_gate,frame_gate,state_io,shadow}`)
reach 0 occupancy modules, by static AST walk and by import-time
`sys.modules`. occ_out feeds the debug view / C++ viz only.

**VRAM (CPU-box arithmetic estimate, `vram_est_mb_cpu`)**: `memory_estimate`
takes `batch`; at the `1080ti` profile (8 cams @640, default net, ctx on,
max_points 16,384): batch 1 (live and the offline runner `OccRuntime`, which is
batch 1) = 104.7 MB; batch 4 (training batch) = 416.8 MB, both under the
4,096 MB cap. Memory module = 16,384 x 13 B on the CPU (no VRAM).
`peak_vram_mb` stays UNPROVEN until IQU45 nvidia-smi.

Tests: `test_occ_unseen` (TRACER traces, labels, unseen decode, observed,
memory, causality), `test_occ_memory_gap` (hidden-leg proof through
`OccRuntime`: predict + cap + vehicle-signal ego + observed + memory; lift retrain,
loss ablation, zero-warp, memory-off mutants; camera VO is not in this proof), `test_occ_net` (xy-spread
cap, below-ground-hot cap, per-column ground band at 0.8/1.6 m including
3% downhill, -0.7 m offset, flat, and exact band contents, class-balanced ground positives, below-ground
free samples, deterministic overfit/predict), `test_occ_out` (v2 fields, version gate),
`test_occ_runtime_imports` (safety isolation), `test_eval_occ_unseen` (eval
smoke). Metrics: `scripts/eval_occ_unseen.py` (results below).

Later-stage `predict` caps (`_capped_hot`, at production `LEVELS` (4, 2):
cell 1.6 m and 0.8 m): the ground band is chosen from the bottom of each
xy column, not the top. The seed is the highest near-max layer (within a
small margin of the column max) in a three-layer window starting at the
lowest near-max layer — the surface of a fill+ground stack. Seeds that
sit above the neighbourhood lower envelope (a wall or car roof, or a
building with no ground under it) inherit the nearest ground seed, so
structure stays in the above-ground pool. A below-ground fill that ties
or is slightly hotter does not win. Below and above are relative to that column's ground, so a downhill
or height-offset road stays the band instead of a fixed z=0 slab. At most
half the cap goes to the band; above-ground structure is filled next, then
below-ground, each pool spread in xy so a high-p off-road region cannot take
the whole ground budget. A holed or partial ground layer therefore cannot
lose its own columns to a more populous below-ground fill. `sample_queries` class-balances ground positives and spends half of
`n_free` on unoccupied *below-ground* cell centres (never on above-ground
cells missing from `keys`, which may be occluded structure).

#### Unseen-structure results (SYNTHETIC, `eval_occ_unseen.py`)

This table is from a run on 2026-11-17 01:55-04:13 CST, before the residual-VO,
three-pool cap, and below-ground free-sample commits (`490fc34`..`b5279d2`,
about 12:00-1:00 PM CST the same day). It has not been re-run at later heads.
Run was on the loaded CPU box (load 26-73),
`--steps 300 --train-seeds 10 --eval-seeds 3 --frames 5 --baseline --threads 4`,
cheap net (45,398 params) at 64 px, 100 train samples; held-out seeds 900-902
per clip kind; last frame of each clip, mean over the 3 clips. Visible /
occluded split by the 8-camera ray-cast mask (metric only). All metrics are
inside the ego grid window; the share of GT road length beyond the window is
listed per clip. One seed, tiny model: **these are not evidence that unseen
decoding works yet.**

| Kind | Model | IoU visible | IoU occluded | edge chamfer vis (m) | edge chamfer occ (m) |
|---|---|---|---|---|---|
| intersection | map_unmasked (+memory) | 0.475 | 0.109 | 1.22 | 3.05 |
| intersection | map_unmasked, no memory | 0.474 | 0.111 | 1.22 | 3.17 |
| intersection | camera_masked baseline | 0.345 | 0.087 | 1.62 | 3.64 |
| roundabout | map_unmasked (+memory) | 0.203 | 0.095 | 1.87 | NaN (no occluded edge predicted) |
| roundabout | map_unmasked, no memory | 0.203 | 0.095 | 1.87 | NaN |
| roundabout | camera_masked baseline | 0.121 | 0.207 | 2.07 | 2.99 |

GT cells per clip: intersection 2,882 visible / 2,110 occluded; roundabout
1,885 / 3,312. Road beyond the window: intersection 0.358 / 0.358 / 0.358,
roundabout 0.318 / 0.294 / 0.327 (seeds 900/901/902).
Reading: the unmasked model beats the baseline on visible IoU in both kinds
and on occluded IoU/chamfer at the intersection, but **loses on roundabout
occluded IoU (0.095 vs 0.207)**. On this synthetic eval, adding memory
changes occluded IoU by <= 0.002 — it is not a proof that memory works,
and it is not a proof that it does not. The memory contract is tested by
`test_occ_unseen` (unit carry 79% vs 0% on a controlled clip) and by
`test_occ_memory_gap` (OccRuntime hidden-leg Youden J, held-out starts,
mutants must fail the gate). VRAM estimate: batch 1 104.7 MB, batch 4
416.8 MB (CPU arithmetic); runner batch 1 (eval and live), 4 (training).

#### Grid window (eval radius)

The occupancy metric and the hidden-leg proof are inside `DEFAULT_GRID`
(0.4 m voxels). Anything outside this window is reported as a share of GT
road length, not as occupancy error.

| axis | min (m) | max (m) | cells | half-extent / radius from ego (m) |
| --- | ---: | ---: | ---: | ---: |
| x | -25.6 | 25.6 | 128 | 25.6 |
| y | -12.8 | 51.2 | 160 | 12.8 behind / 51.2 ahead |
| z | -2.0 | 4.4 | 16 | 2.0 below / 4.4 above |

Eval (synthetic, CPU; same flags as the table above):

```
PATH=<venv>/bin:$PATH python scripts/eval_occ_unseen.py --steps 300 --train-seeds 10 --eval-seeds 3 --frames 5 --baseline --threads 4
```

## Files

New: `python/perception/occ_geometry.py`, `python/perception/occ_sparse.py`,
`python/perception/occ_outputs.py`, `python/perception/occ_ego.py`, `python/perception/occ_out.py`, `python/viz/occ_viz.py`, `python/train/occ_map_labels.py`, `tests/synthetic_scenes.py`, `scripts/eval_occ_unseen.py`,
`python/train/occ_labels.py`, `python/train/occ_sampler.py`,
`python/train/occ_train.py`, `python/data/occ_shards.py`,
`scripts/bench_occ_sparse.py`, `scripts/compare_front_cluster.py`, `tests/synthetic_occ.py`, tests
`scripts/test_occ_{labels,shards,sampler,net,cams,ego,runtime_imports,viz,out,unseen,memory_gap}.py`, `scripts/test_eval_occ_unseen.py`,
`scripts/test_bench_occ_sparse.py`, this spec. The offline suite picks up
`scripts/test_*.py` itself.

## Verify

```
PATH=<venv>/bin:$PATH xvfb-run -a python scripts/run_offline_suite.py
python scripts/test_occ_labels.py      # geometry, 8 cams, boxes, invalid depth, class names
python scripts/test_occ_out.py         # occ_out v1: 4-panel fields, ego dt/units/source, heading, 2-D boxes, corridor, rejects, VRAM fields, flat size
python scripts/test_occ_shards.py      # roundtrip, no loose-PNG fallback, rollover, CLI
python scripts/test_occ_sampler.py     # 20-30% recovery per batch, enforced
python scripts/test_occ_net.py         # projection, camera-only, sparse predict, overfit, cap, class balance, free samples
python scripts/test_occ_cams.py        # presets, per-cam ages, sparsity begins, checkpoints, profiles
python scripts/test_occ_ego.py         # camera VO, observed-only clutter, parked structured probes, slow motion, static world stays fixed with the flag off
python scripts/test_occ_runtime_imports.py  # GT label/pose modules unreachable, flag off, spy, image guard
python scripts/test_occ_viz.py         # edges vs synthetic road, tracker cases, v1 message, red rule
python scripts/test_occ_unseen.py      # unmasked occluded loss, unseen decode, memory rules, causality
python scripts/test_occ_memory_gap.py  # OccRuntime hidden-leg proof + mutants (CPU, ~few minutes)
python scripts/eval_occ_unseen.py --steps 300 --train-seeds 10 --eval-seeds 3 --frames 5 --baseline --threads 4
```

Desktop-IQU45 (real VRAM and FPS, GTX 1080 Ti, BeamNG running; note what else holds the GPU):

```
py -3 scripts/bench_occ_sparse.py --device cuda --profile 1080ti --iters 20 --warmup 3 --samples 16 --steps 20 --batch 4 --amp fp16 --workers 4
py -3 scripts/bench_occ_sparse.py --device cuda --sweep --iters 20 --warmup 3
py -3 scripts/bench_occ_sparse.py --device cuda --profile front_cluster --infer-only --iters 20 --warmup 3
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# then record the reading (only this fills peak_vram_mb; the arithmetic number stays vram_est_mb_cpu):
py -3 scripts/bench_occ_sparse.py --device cuda --profile 1080ti --infer-only --iters 20 --warmup 3 --nvidia-smi-mb <used MB>
```

Each run ends with a `SUMMARY` line: `fps=` next to `infer_vram_mb=` (8 cams @640
for `--profile 1080ti`), `vram_est_mb_cpu=` (arithmetic estimate), `peak_vram_mb=UNPROVEN`
(until `--nvidia-smi-mb` is given; then `MEASURED`), and for a training
run `train_vram_mb=` (AMP: memory saving only on Pascal). The sweep covers net
(default/cheap) x cameras (all8/front3/min) x long side (640/480/320) x fp32/fp16.
Acceptance: inference <= 4096 MB and (VRAM free with BeamNG running - 1 GB);
training <= 11264 MB. If `train_wait_ms` is a large share of `train_compute_ms`,
raise `--workers`.

Real data: `python -m python.data.occ_shards <recording> <nvme>/occ_shards/<name>`
then `python -m python.train.occ_train <nvme>/occ_shards --out <ckpt> --amp fp16`.

## Risks

- Never trained on real data; edges/objects are only as good as the occupancy.
- 8-cam labels need `train_gt` on all 8 cams in the recorder (today main + wide
  only: +6.6 GB/h more if all 8 are turned on at rig rates).
- PCA heading is +-pi ambiguous for a parked car until it moves.
- Camera VO needs static non-road structure; on a bare straight road it holds
  the last estimate (flagged). Not yet measured on recorded clips.
- Rig geometry is read through `raw_frames.camera_sidecar`, which loads the live
  camera client module (it references beamngpy lazily); the occupancy modules
  themselves never import beamngpy or a pose source (tested).
- Untrained `predict` hits the per-stage caps (33.9% of dense queried); a trained
  net queries fewer, the oracle says 10-33%.
