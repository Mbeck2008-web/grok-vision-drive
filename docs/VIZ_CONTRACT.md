# Visualizer contract

Everything the visualizer draws comes from the cameras and the live model that is currently driving the car. That means the outputs of the deployed model at runtime.

BeamNG ground truth, game poses, sensors, the map, the road data, and the AI path are not drawn at runtime, including as a debug fallback. BeamNG ground truth is only for training labels.

If a model output is missing, or its confidence is under 0.40, draw nothing. Do not guess a line, a box, or a lamp.

`python/train/viz_contract.py` is the check. `renderable_state` drops every field that is not a high-confidence output of one named head on one checkpoint. `require_model_drawable` raises on anything else. The ribbon in #197 already follows the same rule for lane centers: finite model samples only, hidden under 0.40.

## Drawable record

Each drawable field carries:

| key | meaning |
| --- | --- |
| `src` | `model` |
| `head` | `lane_gt`, `edge`, `obj`, or `brakelight` |
| `ckpt` | checkpoint file name |
| `confidence` | model confidence, in `[0, 1]` |

`label_src=beamng_gt` is a training label. It is never drawable, even if someone also sets `src=model`.

A scene-ckpt tick stores the lane tag on `scene_ckpt.lane` and, when those heads were trained, `scene_ckpt.detect`. Heads that were not in the checkpoint are omitted. Low-confidence edge points are an empty list. Low-confidence boxes and brake lamps are omitted. The old `lane_x` number stays for the existing shadow path. New drawing goes through `renderable_state`.

`edge` is a road-edge head. It is not paint and it is not copied into `lane_gt`.

## Training labels

`scripts/export_detect_labels.py` writes `<strip>/detect.jsonl` from a recorded `state.jsonl`. Those rows are `label_src=beamng_gt` and `train_only`.

What today's recordings can actually label:

* Road edge, when `road_width_m` and `road_offset_m` are present, or when an ego-frame polyline is in `edge_left` / `edge_right`. `lane_width_m` and painted `lane_gt` are not read. A polyline that does not cover a lookahead leaves that slot masked. One side does not invent the other.
* Vehicle position and heading, when the row has ego pose and traffic `pos` / `dir`. Length, width, and height stay masked unless the row already has them. The traffic probe records pos, vel, and dir only.
* Cones, only from `static_objects` or `cones` already on the row. The annotation camera is not turned into a box.
* Lead brake lamps, only when a traffic vehicle ahead has `brakelights`. The ego car's own lamp is not the lead. Today's collector does not record other vehicles' lamps, so this label is masked on current strips.

The heads are off unless named: `--heads lane_gt,edge,obj,brakelight`. An empty `--heads` does not build them, and a default `SceneNet` checkpoint stays the same. They are not default ONNX outputs. Live fields show up on `--scene-ckpt` when the checkpoint contains the heads. They do not steer.

Holdout lane RMS still decides the run. When the heads are on, the val log adds `edge_mae`, `obj_mae`, and `brake_mae` with a labeled count. A head with no labels prints `masked`.

## IQU45

Quit BeamNG before training. Do not run these from a review machine.

```
py -3 scripts\export_detect_labels.py E:\GVD_data\scene_lanes
py -3 -m python.train.train_scene --strips E:\GVD_data\scene_lanes --cams main --heads lane_gt,edge,obj,brakelight --augment --balance --cache C:\GVD_cache --epochs 30 --clip 1.0 --lane-loss huber --min-valid 0.3 --lr 1e-3
```

Learning rate stays `1e-3`. The three new heads are linear maps on the existing 512-d feature. They do not change the fp32 training path and they are not a second detector.
