# Debugging from a shell

All of these files land on the data disk, under `<data dir>`.
`GVD_DATA_DIR` moves the whole tree. C: and D: are refused.

One command shows whatever is running, or the last run:

```
py -3 scripts\gvd_status.py
```

It prints free disk, collection, the training heartbeat (step and ETA),
the last metrics line, a crash file if training died, the live lane
line, the validator, and the eval summary.

## Automatic Mode

BeamNG AI collection. The GVD VISION **AUTO** tab (`U`, then Start / Stop) and this command are the same recorder. Every tick is `driver=beamng_ai`. Files stay under `<data dir>`. The 20 GB free stop still applies. Status is `<data dir>\status\collect.json`.

```
py -3 scripts\collect_scene_lanes.py --minutes 45
py -3 scripts\collect_scene_lanes.py --map west_coast_usa --route random --waypoints a,b --speed-min 8 --speed-max 16 --minutes 30
py -3 scripts\collect_scene_lanes.py --until-disk --traffic --cameras main,narrow,wide --time-of-day 0.5 --weather clear
```

`--until-disk` ignores the clock and stops only under 20 GB free. `--traffic` uses the traffic AI. Off is span, traffic-free. Time of day and weather are sent when set. Empty leaves the sky alone.

## Collect

```
py -3 scripts\collect_scene_lanes.py --minutes 45
```

Each camera is saved on its own, lossless, as BeamNG returned it:
`<data dir>\scene_lanes\raw\<cam>\NNNNNN.png` (uint8) or `.npy` (any
other dtype). No JPEG, resize, crop, colour change, overlay, or stitch
is written into those files. `state.jsonl` has the timestamps, `dt_s`,
and the camera pose and intrinsics. Lane targets are `gt.jsonl`. The
trainer stitches and resizes when it loads.

PNG versus uint8 was measured on the 85 real BeamNG frames in
`tests/fixtures/gt_run*.jpg` (1336×96 strips): mean ratio 0.507, worst
0.619. Saving every camera at 10 Hz would be 231.7 GB/h uncompressed.
The recorder does not do that.

| camera | size | rate | PNG at the worst measured ratio |
| --- | --- | --- | --- |
| narrow, main | 640×480×3 | 10 Hz | 19.13 GB/h each |
| wide | 640×360×3 | 10 Hz | 14.34 GB/h |
| pillarL, pillarR | 640×480×3 | 2 Hz | 3.83 GB/h each |
| repeatL, repeatR | 640×480×3 | 1 Hz | 1.91 GB/h each |
| rear | 640×360×3 | 1 Hz | 1.43 GB/h |

Lane cameras (narrow, main, wide) stay at the 10 Hz loop. Pillars are
2 Hz. Repeaters and the rear camera are 1 Hz. That plan is **65.5 GB/h**
of PNG at the worst measured ratio, or **105.8 GB/h** uncompressed. A
45 minute collect is about 49 GB of PNG at that worst ratio. The minute
line prints measured GB/h, queue depth, and dropped frames. Disk writes
run on a side thread with a queue of 8 ticks. A full queue drops the
tick instead of stalling capture. Free space is checked every 10
seconds. The run stops under 20 GB free.

Each tick's `state.jsonl` line has the capture timestamp, `dt_s`, the
steer/throttle/brake that were applied, speed, yaw rate, pose, and who
was driving (`beamng_ai`, `gvd`, or `human`). Lane targets for that same
index are in `gt.jsonl`.

Every minute, and once at start and stop, this rewrites
`<data dir>\status\collect.json`: frames, Hz, dropped cameras, cameras
that have been missing for several ticks, the fraction of ticks with a
valid lane target, free disk, the AI speed and mode, and the raw GB/h
ceiling.

## Check a recording

```
py -3 scripts\validate_scene_lanes.py --strips <data dir>\scene_lanes
```

Writes `<data dir>\status\validate\report.json` and a few
`montage_NNNNNN.png` files with the ground-truth lane drawn. The report
counts rows against the raw camera files, timestamp gaps, NaNs, target
ranges, and blank frames per camera. Montage PNGs are written beside the
report, not onto the recorded frames. Exit 1 means the report has issues.

## Train

```
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --epochs 30 --export <data dir>\models\e2e_scene.onnx
```

`<data dir>\status\metrics.jsonl` gains a line every 10 steps: loss per
head, grad norm, learning rate, step time, GPU memory, GPU utilization
when `nvidia-smi` answers, and NaN/Inf counts. A validation line with
r, rms, and bias is added at the end of each pass.

`<data dir>\status\train_heartbeat.json` is rewritten about every 30
seconds with the step and the ETA.

A checkpoint is written about every 3 minutes next to the strip folder
(`scene_lanes.gvd-pause.npz`). The next start loads it. Pass `--resume`
to say that explicitly. `--no-resume` starts at step 0 and leaves the
file until a new checkpoint replaces it.

If the step throws, the traceback and the last batch's row ids are in
`<data dir>\status\train_crash.txt`.

Shapes only, two steps, then stop:

```
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --dry-run
```

Overfit the first 8 strips. Exit 1 unless the lane-head loss falls
below 0.05. This does not export and does not replace a checkpoint.

```
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --sanity
```

## Evaluate

```
py -3 scripts\eval_scene_lanes.py --strips <data dir>\scene_lanes
```

Writes `<data dir>\eval\predictions.csv` (one row per holdout sample),
`<data dir>\eval\worst\*.png` (predicted lane in blue, ground truth in
green), `<data dir>\eval\lookahead.png`, and `summary.json` with r,
rms, and bias at 5, 10, 20, and 30 m plus heading. The 8 m number is
the one the live log uses.

`--check-targets` only checks that the file has targets. It does not
need PyTorch.

## Controller with perfect lanes

```
py -3 scripts\live_lane_test.py --lane-source gt --seconds 120 --min-in-lane 80
```

`GVD_LANE_SOURCE=gt` does the same thing. The supervisor uses the road-network center at 8 m as `lane_x` and stops when that hit is missing. Scene-net lanes stay off this path.

On the saved logs, engaged ticks with `|steer|>=0.95` are 0 of 1186 across runs 5, 6, 7, and 9. Run 8 (the reverse) is 6 of 28 and is not in that share. Run 10's log is not in the tree. An offline bicycle with that same 8 m center holds a straight road and a 200 m curve for 120 s and more than 300 m.

## Live

```
py -3 -m python.run_vision --backend beamngpy --lane-source scene --scene-onnx <data dir>\models\e2e_scene.onnx --gt-log --hz 10
```

The segment line prints `model_lane_x` and `model_conf`. The ground-truth
JSONL under `<data dir>\logs` stores those next to `gt_center_x`, and
the debug JPEG overlay does too. `<data dir>\status\live.json` is the
latest sample. `gt_report` is unchanged:

```
py -3 scripts\gt_report.py <data dir>\logs\gt_YYYYMMDD_HHMMSS.jsonl
```
