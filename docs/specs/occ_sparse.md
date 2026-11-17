# Sparse occupancy head (training-only), v1

## Goal
Teach SceneNet 3D occupancy around ego from the labels v4 ground truth we already record,
without a dense 3D grid and without changing what runs live.

- Labels: `raw/<cam>_depth` (uint16 cm) + `raw/<cam>_sem` (uint8 ids, `sem_classes.json`)
  for every cam with `train_gt`, plus traffic vehicle centres from `state.jsonl`.
- Grid (ego frame, x right / y forward / z up): 0.5 m voxels, x [-16,16), y [-16,40), z [-1,3)
  = 64x112x8 = 57,344 cells. Same x span and forward reach as the labels v4 obstacles BEV.
- Classes: `unknown`, `drivable` (`sensor_aux.DRIVABLE_CLASS_ALLOWLIST`), `vehicle`.
  Priority vehicle > unknown > drivable. Free = ray samples at 25/50/75 % of each hit depth,
  minus occupied cells.
- Stored per tick: occupied voxel ids + class, free voxel ids. Typical synthetic 8-cam tick:
  704 occupied + 912 free of 57,344 (2.8 %). Nothing dense is ever stored or predicted.
- Head: `SparseOccHead` = Fourier PE of the voxel centre (6 freqs) concatenated with the
  pooled SceneNet feature, MLP width 128 -> 1 occ logit + 3 class logits. Cost is O(queries),
  ~88k params. Loss = masked BCE on occ + CE on occupied cells; empty labels give zero loss.
- Storage: WebDataset tar shards (`occ-<rec>-NNNNNN.tar`, members `<key>.occ_idx.npy`,
  `.occ_cls.npy`, `.free_idx.npy`, `.json`), rotated below 3.5 GB (FAT32).
- Recovery mix: `recovery_weights(rows, target=0.25)` gives sampler weights so
  `excursion.phase == "recover"` rows are 25 % of draws; targets outside 20–30 % raise.

## Viz schema (C++ visualizer, frozen v1 = `docs/specs/viz_cpp.md` at `f3555bf`)
`python/viz/occ_scene.py` (numpy only; imports no torch, labelers or raw frames). Entry point
`occ_to_viz(outputs, tracker=, t=, ego_speed_mps=) -> dict`. Its output has every v1 key
(`v type seq t ego aeb lanes edges tracks signs path path_conf`), and v1 builders ignore the optional
extras below. When `python.viz.cpp_publisher` is importable (PR #232), the adapter uses its
`build_scene_message` and only adds fields. Red is the renderer's colour for `edges` only. Lanes are
`white` or `yellow`, and edges carry no paint.

| field | shape | from | GT source in recorder data |
|---|---|---|---|
| `occ` (optional) | `{x, y, z, cls}`, cls unknown/drivable/vehicle, at most 800 (about 41 kB) | occ head on queried voxels, sigmoid > 0.5 | labels v4 `raw/<cam>_depth` + `_sem` + traffic (this spec) |
| `edges[]` src `occ` | `{pts, kind: detected, src: occ}` | outermost drivable cell per 1 m forward bin, left and right, ±0.25 m | same as `occ` (derived) |
| `edges[]` src `curb` | `{pts, kind: detected, src: curb}` | `curbs` head | existing curbs labels |
| `lanes[].paint` | `white` / `yellow` | `lane_paint` = P(yellow) per lane, white when absent | **labelable, no head yet**: decal materials `line_yellow*` vs `line_white*` (`python/debug/painted_lanes.py`). UNLABELED in the net; live is white. |
| `stop_lines` (optional) | `{pts: [[-1.8, y], [1.8, y]]}` | `stop_line` = (y_m, valid logit) | `intersection_labels.stop_line_arrays`. The head is training-only, so live has none. |
| `crosswalks` (optional) | `{pts: [[x0, y0], [x1, y1]]}` | `crosswalks` (M, 4) if a head provides it | **UNLABELED**: no crosswalk label source on main. Not claimed. |
| `tracks[].motion` (optional) | unknown/same_way/oncoming/crossing/alongside/parked | `MotionTracker` over ≥3 frames spanning ≥0.2 s | traffic positions per tick (`vehicle_labels`). Deterministic, not learned |

Motion is never a single-frame guess. Tracks are matched to the previous tick by nearest position
(≤3 m). Ground velocity is displacement/dt plus ego forward speed. The state comes from that velocity
heading against +y: under 0.5 m/s is parked, ≤30° is same_way (alongside when |y|<6 and |x|>1.5),
≥150° is oncoming, and 60–120° is crossing. A track reports `unknown` until it has enough history.
v1 `yaw` is radians from +x (π/2 = ego heading), the same convention as `vehicle_labels._yaw`.

Tests: `scripts/test_occ_viz.py`. It round-trips the JSON, checks edge positions against tolerances on
a synthetic road band, checks the five motion states over four frames, checks a single frame stays
`unknown`, checks red only on edges and curbs, checks the v1 required keys, and checks that unknown
keys are ignored. Checked locally against the `f3555bf` builder (not in CI, since #232 is not on
main). Extras were `crosswalks`, `occ` and `stop_lines`, the encoded size was 28.8 kB, and there were
no problems.

## Non-goals
- No runtime / ONNX change: `occ_head` is outside `OUTPUT_FIELDS` and the export, like
  `aux_depth_head` and `stop_line_head`. `run_vision` never imports `occ_sparse`.
- No safety, actuation, heartbeat, AEB or engage change.
- No recorder or `config/cameras.yaml` change (train_gt stays main+wide, see notes).
- No new deps, no heavy models. Stdlib `tarfile` + numpy only.
- No `train_scene` CLI flag yet (follow-up: `--occ-sparse` + shard path + weight).

## File touch list
- `python/train/occ_sparse.py` (new): labeler, shards, sampler weights, queries, loss, head.
- `python/train/scene_net.py`: opt-in `occ_sparse=False` kwarg, `occ_feat` on the prediction,
  `:occ_sparse=128` arch suffix. Default net is byte-for-byte the same; plain checkpoints load.
- `python/viz/occ_scene.py` (new): `occ_to_viz` adapter, `MotionTracker`, v1 validator.
- `scripts/test_occ_sparse.py`, `scripts/test_occ_viz.py`, `docs/specs/occ_sparse.md` (new).

## Verify
1. `python scripts/test_occ_sparse.py` and `python scripts/test_occ_viz.py` -> `test_occ_sparse: OK` (overfit loss falls >10x, acc >0.97).
2. `PATH=venv xvfb-run -a python scripts/run_offline_suite.py` green.
3. CI attempt 1 green on the draft PR.

## Notes / conflicts found
- The brief says depth/sem is recorded for all 8 cams. It is not: commit b0ac072 and
  labels_v4.md keep `train_gt` on main+wide only to protect the 11 GB card. The labeler takes
  any cam set (tested with 8 synthetic cams); turning on the other six is a separate VRAM call.
- The brief asks for AMP fp16 on Pascal. AGENTS.md says Pascal (6.1) stays FP32. This head
  adds ~88k params and no activations beyond `(B, k, 128)`, so FP32 is the default here.
- VRAM was not measured on the 1080 Ti (no GPU on the build box). Numbers here are CPU only.
