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
- Not wired into `scripts/live_*` yet: nothing on the live path imports it.

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

## Outputs and their labels (viz contract)

`occ_outputs.occ_outputs(pred, tracker, t, ...)` -> `occ_viz.occ_to_viz` -> `gvd_scene` v1. Every v1
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
| Motion state | optional `tracks[].motion` = `same_way|oncoming|crossing|alongside|parked|unknown` | `OccTracker`: nearest-neighbour association (3 m gate, 2 misses), ego-motion compensated (speed, yaw rate), least-squares velocity over >= 3 frames; `unknown` before 3 frames | per-tick traffic rows positions over time (offline, for evaluation) | multi-frame tracker, never one frame |

Motion rule (`motion_state`, ego frame, world velocity after ego-motion
compensation; angle 0 = ego heading): speed < 0.7 m/s -> `parked`; angle <= 35 deg
-> `alongside` when |x| > 1.8 m and |y| < 6 m, else `same_way`; >= 145 deg ->
`oncoming`; 55-125 deg -> `crossing`; the bands between go to the nearer of
same_way / oncoming.

Colours (Michael's rule wins over the legend): white paint white, yellow paint
yellow, **red only on `edges[]` (road edges and curbs)**. Tracks: same_way green,
oncoming yellow, crossing purple, alongside blue, parked orange, unknown blue.
Crosswalks and stop lines white.

## Files

New: `python/perception/occ_geometry.py`, `python/perception/occ_sparse.py`,
`python/perception/occ_outputs.py`, `python/viz/occ_viz.py`,
`python/train/occ_labels.py`, `python/train/occ_sampler.py`,
`python/train/occ_train.py`, `python/data/occ_shards.py`,
`scripts/bench_occ_sparse.py`, `tests/synthetic_occ.py`, tests
`scripts/test_occ_{labels,shards,sampler,net,runtime_imports,viz}.py`,
`scripts/test_bench_occ_sparse.py`, this spec. No existing file is modified (the
offline suite picks up `scripts/test_*.py` itself).

## Verify

```
PATH=<venv>/bin:$PATH xvfb-run -a python scripts/run_offline_suite.py
python scripts/test_occ_labels.py      # geometry, 8 cams, boxes, invalid depth, class names
python scripts/test_occ_shards.py      # roundtrip, no loose-PNG fallback, rollover, CLI
python scripts/test_occ_sampler.py     # 20-30% recovery per batch, enforced
python scripts/test_occ_net.py         # projection, camera-only, sparse predict, overfit
python scripts/test_occ_runtime_imports.py  # GT modules not reachable from live entry points
python scripts/test_occ_viz.py         # edges vs synthetic road, tracker cases, v1 message, red rule
```

Desktop-IQU45 (real VRAM, GTX 1080 Ti; note what else holds the GPU):

```
py -3 scripts/bench_occ_sparse.py --device cuda --long-side 640 --fp16 --iters 20 --warmup 3 --samples 16 --steps 20 --batch 4 --amp fp16 --workers 4
```

`infer_cuda_max_allocated_mb` = fp16 inference peak with all 8 cams at 640 (untrained
net, caps hit: worst case); `train_cuda_max_allocated_mb` = AMP training peak.
Acceptance: inference <= 4096 MB, training <= 11264 MB. If `train_wait_ms_median`
is a large share of `train_compute_ms_median`, raise `--workers`.

Real data: `python -m python.data.occ_shards <recording> <nvme>/occ_shards/<name>`
then `python -m python.train.occ_train <nvme>/occ_shards --out <ckpt> --amp fp16`.

## Risks

- Never trained on real data; edges/objects are only as good as the occupancy.
- 8-cam labels need `train_gt` on all 8 cams in the recorder (today main + wide
  only: +6.6 GB/h more if all 8 are turned on at rig rates).
- PCA heading is +-pi ambiguous for a parked car until it moves.
- Untrained `predict` hits the per-stage caps (33.9% of dense queried); a trained
  net queries fewer, the oracle says 10-33%.
