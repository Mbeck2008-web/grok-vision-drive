# SPEC: training-time frame quality filter

Rev 4. Offline labels and training only. Deploy stays camera-only.

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
road keeps its stored node order, `oneWay`, `flipDirection`,
`lanesLeft` / `lanesRight`, width, and point heights. Project the ego
`pos` onto the nearest on-road centerline. **Do not flip the tangent to
match the car** (that is the live labeler's `heading_err_rad` trick and
hides a 180° reverse). **Do reverse the tangent when `flipDirection` is
true**: BeamNG's legal driving direction follows node order by default
and `flipDirection` reverses that logical direction without rewriting
the stored edges. "Unflipped" in this spec means only the first of those
two rules.

- The legal tangent is the stored node-order tangent, negated when
  `flipDirection` is true. `align` is the dot product of vehicle heading
  and that legal tangent. `offset_m` is positive to the right of it (US RHT).
- A snap whose closest foot has `|dz| > 3 m` (when both the pose and the
  road carry height) is discarded so a bridge/overpass does not steal
  the car from the deck it is on. Remaining candidates are ranked by 3D
  distance.
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

The default wrong-way drop (opt out with `--keep-wrong-way`) removes only
`verdict == wrong_way`. `ok` and `unknown` are kept. `--drop-wrong-way` is
kept as a compatibility alias for the default drop; it also makes a missing
road dump a hard error (see Flags).

Validated on IQU45 `sgt_2h_20261105` frame `i=28392` (turn arrow toward the
camera, yellow line on the right): classified `wrong_way` / `oneway_against`
on dump road `893958.0`, heading 172.5° from the stored tangent. Nearby
one-way carriageway `892718.0` is also against traffic; the matching-direction
carriageway `893963.0` is ~8.8 m across the median.

## Method (overexposure)

`road_clip_fraction` is the share of **lower-half** pixels whose any RGB
channel is `>= 250`, computed **per selected training camera** (the `--cams`
set, `STRIP_ORDER` only — never `main_depth` / `main_sem`, never the full
2272×192 `compose_strip` canvas with black pads for missing cameras). A
frame is dropped when the **maximum** of those present per-cam fractions
**exceeds** T. Absent cameras are omitted, so a fully washed main sector
trips default T=0.20 even when the other cameras are black or missing
(on the old full canvas a 100% clipped main is only 256/2272 ≈ 0.113 and
could not trip T). Default `T = 0.20`. Pass `T = -1` to disable. Never at
deploy.

`--cams main` (the documented training command and the live A/B trainer)
therefore uses the 256×192 main sector alone. A multi-cam selection such
as `--cams main,wide` drops if **any** of those present cameras exceeds T,
not only main.

Per-cam fractions are persisted in a `road_clip_v1.jsonl` sidecar keyed by
source + frame id + metric version. When `--cache` is set, the sidecar
lives in the cache directory and sectors are read from the frame cache
instead of re-decoding raw PNGs. A later dataset build with the same
sidecar does not re-decode.

### Measured default-on cost (IQU45, 2026-11-10)

Evenly spaced sample **n=2501 per recording** (stride ≈28, read-only) of
`sgt_2h_20261105` (70704 rows) and `sgt_2h_20261105_r2` (71206 rows).
Wrong-way is exact on every row (sgt **2445 / 3.458%**, r2 **2229 / 3.130%**).
The 150-frame main sample quoted in Rev 3 (25.3% / 22.7%) is superseded.

| Recording | Official full-canvas >0.20 | Main sector >0.20 | Combined official (WW + canvas) | Combined if main-filtered (WW + main clip) |
| --- | ---: | ---: | ---: | ---: |
| sgt | 14.87% (372/2501, 13.53–16.32) | 26.67% (667/2501, 24.97–28.44) | 18.11% kept 81.89% (2048/2501) | **29.47%** kept 70.53% (1764/2501) |
| r2 | 14.19% (355/2501, 12.88–15.62) | 23.79% (595/2501, 22.16–25.50) | 16.87% kept 83.13% (2079/2501) | **26.19%** kept 73.81% (1846/2501) |

Default-on cost for `--cams main` (main-sector clip + wrong-way) is about
**29.5% sgt / 26.2% r2**. Load is uneven: sgt `west_coast_usa#7` is 119/252
main-over (**47.2%** of that segment’s sample); r2 `#5` is 129/250 (**51.6%**).
Main-over frames come in long runs (sgt: 127 runs, median 3 stride-steps,
mean 5.3, p90=12, max 34 steps ≈ 94 s / 933 indices, 39 singletons). The
old canvas cut was a stricter multi-camera average that kept roughly
two-fifths of washed-main frames (296/667 sgt, 241/595 r2). Wide alone
exceeds 0.20 on about 40–46% of sampled frames, so adding `wide` to
`--cams` drops more than the main-only figures above.

Source: Skeptic measurement on Desktop-IQU45 for PR #211 (2:18–2:30 AM
America/Chicago, Nov 10, 2026; evenly spaced n=2501/recording, BelowNormal,
STRIP_ORDER cameras only).

## Flags (`train_scene.py`)

| Flag | Default | Effect |
| --- | --- | --- |
| (wrong-way drop) | on | Drop `wrong_way` rows before the holdout split |
| `--keep-wrong-way` | off | Opt out of the default wrong-way drop |
| `--drop-wrong-way` | off (compat alias) | Same drop as the default; fail if the road dump is missing |
| `--road-dump PATH` | empty → `roads_dir() / <map>.json.gz` | Road network used by the detector |
| `--max-road-clip T` | `0.20` | Drop when the max per-cam lower-half clip of the selected `--cams` exceeds T. `-1` disables |

Michael asked to "just drop the wrong way and clipping", so both filters are
on by default. Opt out with `--keep-wrong-way` and `--max-road-clip -1` to
recover the earlier unfiltered row set.

If no road dump exists for the map (and `--road-dump` was not given), the
wrong-way filter **does not crash training**: it logs one warning and keeps
wrong-way / unknown frames. The road-clip filter still runs at the default
T when clip is on; the warning must not say that every frame was kept. If
the user explicitly passed `--drop-wrong-way`, a missing or unreadable dump
fails the run.

At dataset build the trainer prints one line with the dropped count for
each filter (`wrong_way=… overexposed=… kept=…`).

The filter runs before the holdout split, so train and holdout both lose
the dropped rows. Metrics from a filtered run are **not comparable** to
earlier unfiltered runs (different row set, and a newly chosen
`holdout_cells.json` can pick different cells). Compare filtered runs
only to other runs that used the same flags.

Default `resume=True` reloads weights, step, optimizer, and a cursor into
the pass plan. The pass is rebuilt from the new (filtered, shorter) row
list. Resuming an unfiltered `epoch_ckpt` on a now-filtered row set is a
distribution shift plus a possible cursor/length mismatch, not a crash.
The live A/B recipe uses `--no-resume`.

## Files

- `python/train/frame_quality.py` — detector, clip fraction, row drop
- `python/train/train_scene.py` — flags and index-time application
- `docs/SPEC_frame_quality_filter.md` — this spec
- `scripts/test_frame_quality_filter.py` — offline tests

## Verify

- `python scripts/test_frame_quality_filter.py` (also collected by
  `scripts/run_offline_suite.py`).
- Parser defaults turn both filters on (`--max-road-clip` is `0.20`;
  wrong-way drop is on unless `--keep-wrong-way`). `--drop-wrong-way` remains
  accepted. `--keep-wrong-way` and `--max-road-clip -1` opt out.
- `apply_frame_quality` function defaults stay off so library callers must
  opt in. The trainer maps parser flags through `frame_quality_from_args`.
- A missing road dump with the default drop on logs one warning and keeps
  wrong-way frames as unknown; default clip still runs and the warning does
  not say every frame was kept. Explicit `--drop-wrong-way`
  (`require_road_dump`) raises `FrameQualityError`.
- A fully washed main sector on an otherwise black 2272×192 canvas is
  dropped at T=0.20 (`check_washed_main_drops_when_other_cams_black`).
  Multi-cam drop uses the max (`check_multicam_drop_uses_max`).
- A second dataset build with a `road_clip_v1` sidecar does not re-decode
  (`check_clip_sidecar_skips_second_decode`). Clip uses `--cache` sectors
  when present and never walks `main_depth` / `main_sem`.
- A sidecar line tagged `road_clip_v0` or `road_clip_v2` is ignored and
  the frame is re-measured (`check_clip_sidecar_ignores_other_metric_versions`).
- Synthetic one-way against-traffic pose is `wrong_way`; with-traffic is `ok`.
- One-way edges along +X with `flipDirection: true`: heading `0` is
  `wrong_way` / `oneway_against`; heading `π` is `ok` / `oneway_with`.
- Overpass fixture: a higher opposite one-way 8 m above must not snap a
  car that is travelling with the lower road.
- Synthetic two-way opposing lane is `wrong_way`; correct side is `ok`;
  centerline and off-network are `unknown` and are not dropped.
- A lower-half white image exceeds `0.20`; an upper-only clip does not.
- `apply_frame_quality` with both flags off returns the same rows.
