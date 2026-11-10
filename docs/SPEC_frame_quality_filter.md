# SPEC: training-time frame quality filter

Rev 1. Offline labels and training only. Deploy stays camera-only.

**Goal.** Stop the scene net from treating two kinds of bad recorded frames
as normal driving:

1. **Wrong-way** frames, where the recording car (BeamNG AI) is travelling
   against the legal direction of a one-way / divided carriageway, or in the
   opposing lane of a two-way road.
2. **Overexposed road** frames, where the lower half of the training image
   (the road) is clipped so lane paint, edges, and lights are gone.

Both checks run at labeling / cache / train-index time from recorded
`state.jsonl` poses plus a saved road dump, and from the training image
pixels. Nothing here is a model input. Runtime perception does not load the
map, does not call this detector, and stays camera-only.

## Non-goals

- Changing BeamNG AI, spawn heading, or capture exposure (those belong in
  the later 8–10 h collection run).
- Feeding map / pose / clip fraction into the net or the visualizer.
- Deleting or rewriting recordings or `gt.jsonl`.
- Down-weighting (this revision drops frames; it does not scale the loss).
- Classifying junctions, U-turns, or off-network poses as wrong-way
  (those stay `unknown` and are kept).
- Using `gt.jsonl` `heading_err_rad`: the live labeler flips the road
  tangent to match the car, so a 180° wrong-way still looks aligned.

## Method (wrong-way)

Load `E:\GVD_data\roads\<map>.json.gz` (or `--road-dump`): each drivable
road keeps its **stored** node order, `oneWay`, `lanesLeft` / `lanesRight`,
and width. Project the ego `pos` onto the nearest on-road centerline
**without** flipping the tangent.

- `align` is the dot product of vehicle heading and the stored tangent.
- `offset_m` is positive to the right of the stored tangent (US RHT).
- A road is treated as one-way when `oneWay` is true, or when
  `lanesLeft == 0` and `lanesRight >= 1`.
- **One-way:** `align < -cos(60°)` → `wrong_way` / `oneway_against`;
  `align > cos(60°)` → `ok`; otherwise `unknown` / `oneway_oblique`.
- **Two-way:** if `|align| < cos(60°)` → `unknown` / `twoway_oblique`;
  if `|offset_m| < 0.75` → `unknown` / `twoway_centerline`;
  if `offset_m * align < 0` → `wrong_way` / `twoway_opposing_lane`;
  else `ok`.
- Off the network (farther than `width/2 + 1.5 m` from every road) →
  `unknown` / `off_network`.

`--drop-wrong-way` removes only `verdict == wrong_way`. `ok` and `unknown`
are kept.

Validated on IQU45 `sgt_2h_20261105` frame `i=28392` (turn arrow toward the
camera, yellow line on the right): classified `wrong_way` / `oneway_against`
on dump road `893958.0`, heading 172.5° from the stored tangent. Nearby
one-way carriageway `892718.0` is also against traffic; the matching-direction
carriageway `893963.0` is ~8.8 m across the median.

## Method (overexposure)

`road_clip_fraction` is the share of **lower-half** pixels whose any RGB
channel is `>= 250`. `--max-road-clip T` (suggested `0.20`) drops a frame
when that fraction **exceeds** `T`. Default `T = -1` disables the check.
Computed from the training image at index time (or a test-injected loader),
never at deploy.

Prior sample on these recordings (150 main frames each, not a full scan):
median lower-half clip 3.0% / 3.3%; 25.3% (sgt) and 22.7% (r2) exceeded 20%;
worst sampled frame 92.1% (sgt).

## Flags (`train_scene.py`)

| Flag | Default | Effect |
| --- | --- | --- |
| `--drop-wrong-way` | off | Drop `wrong_way` rows before the holdout split |
| `--road-dump PATH` | empty → `roads_dir() / <map>.json.gz` | Road network used by the detector |
| `--max-road-clip T` | `-1` (disabled) | Drop frames with lower-half clip `> T` |

Old behavior is unchanged when both filters are left off.

## Files

- `python/train/frame_quality.py` — detector, clip fraction, row drop
- `python/train/train_scene.py` — flags and index-time application
- `docs/SPEC_frame_quality_filter.md` — this spec
- `scripts/test_frame_quality_filter.py` — offline tests

## Verify

- `python scripts/test_frame_quality_filter.py` (also collected by
  `scripts/run_offline_suite.py`).
- Parser defaults keep both filters off.
- Synthetic one-way against-traffic pose is `wrong_way`; with-traffic is `ok`.
- Synthetic two-way opposing lane is `wrong_way`; correct side is `ok`;
  centerline and off-network are `unknown` and are not dropped.
- A lower-half white image exceeds `0.20`; an upper-only clip does not.
- `apply_frame_quality` with both flags off returns the same rows.
