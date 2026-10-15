# Debugging from a shell

All of these files land on the data disk. On the driving PC that is
`E:\GVD_data`. `GVD_DATA_DIR` moves the whole tree. C: and D: are refused.

One command shows whatever is running, or the last run:

```
py -3 scripts\gvd_status.py
```

It prints free disk, collection, the training heartbeat (step and ETA),
the last metrics line, a crash file if training died, the live lane
line, the validator, and the eval summary.

## Collect

```
py -3 scripts\collect_scene_lanes.py --minutes 45
```

Every minute, and once at start and stop, this rewrites
`E:\GVD_data\status\collect.json`: frames, Hz, dropped cameras, cameras
that have been missing for several ticks, the fraction of ticks with a
valid lane target, free disk, and the AI speed and mode. The run stops
itself under 20 GB free.

## Check a recording

```
py -3 scripts\validate_scene_lanes.py --strips E:\GVD_data\scene_lanes
```

Writes `E:\GVD_data\status\validate\report.json` and a few
`montage_NNNNNN.png` files with the ground-truth lane drawn. The report
counts rows against JPEGs, timestamp gaps, NaNs, target ranges, and
blank sectors per camera. Exit 1 means the report has issues.

## Train

```
py -3 -m python.train.train_scene --strips E:\GVD_data\scene_lanes --epochs 30 --export E:\GVD_data\models\e2e_scene.onnx
```

`E:\GVD_data\status\metrics.jsonl` gains a line every 10 steps: loss per
head, grad norm, learning rate, step time, GPU memory, GPU utilization
when `nvidia-smi` answers, and NaN/Inf counts. A validation line with
r, rms, and bias is added at the end of each pass.

`E:\GVD_data\status\train_heartbeat.json` is rewritten about every 30
seconds with the step and the ETA.

A checkpoint is written about every 3 minutes next to the strip folder
(`scene_lanes.gvd-pause.npz`). The next start loads it. Pass `--resume`
to say that explicitly. `--no-resume` starts at step 0 and leaves the
file until a new checkpoint replaces it.

If the step throws, the traceback and the last batch's row ids are in
`E:\GVD_data\status\train_crash.txt`.

Shapes only, two steps, then stop:

```
py -3 -m python.train.train_scene --strips E:\GVD_data\scene_lanes --dry-run
```

Overfit the first 8 strips. Exit 1 unless the lane-head loss falls
below 0.05. This does not export and does not replace a checkpoint.

```
py -3 -m python.train.train_scene --strips E:\GVD_data\scene_lanes --sanity
```

## Evaluate

```
py -3 scripts\eval_scene_lanes.py --strips E:\GVD_data\scene_lanes
```

Writes `E:\GVD_data\eval\predictions.csv` (one row per holdout sample),
`E:\GVD_data\eval\worst\*.png` (predicted lane in blue, ground truth in
green), `E:\GVD_data\eval\lookahead.png`, and `summary.json` with r,
rms, and bias at 5, 10, 20, and 30 m plus heading. The 8 m number is
the one the live log uses.

`--check-targets` only checks that the file has targets. It does not
need PyTorch.

## Live

```
py -3 -m python.run_vision --backend beamngpy --lane-source scene --scene-onnx E:\GVD_data\models\e2e_scene.onnx --gt-log --hz 10
```

The segment line prints `model_lane_x` and `model_conf`. The ground-truth
JSONL under `E:\GVD_data\logs` stores those next to `gt_center_x`, and
the debug JPEG overlay does too. `E:\GVD_data\status\live.json` is the
latest sample. `gt_report` is unchanged:

```
py -3 scripts\gt_report.py E:\GVD_data\logs\gt_YYYYMMDD_HHMMSS.jsonl
```
