# Drive log

Opt-in supervisor log for a live scene-ckpt drive. It does not change steering, pedals, gains, AEB, heartbeat, or engage. Ground truth is written only when `--drive-log-gt` is set, and only into this file. The visualizer still draws sim ground truth only under `--debug-gt`.

## Record (Desktop-IQU45)

BeamNG.tech 0.39 is already up (`BeamNG.tech.exe -tcom -console -gfx vk`, port 25252). One process. The Grok Vision Drive mod is enabled. Spawn the car, then from the repo:

```
py -3 -m python.run_vision --backend beamngpy --scene-ckpt E:\GVD_data\models\<run>.ckpt_ep11_step<S>.npz --scene-ckpt-drive --hz 10 --drive-log E:\GVD_data\logs\drive.jsonl --drive-log-gt
```

`<run>.ckpt_ep11_step<S>.npz` is the aux_ds epoch-11 checkpoint (`train_scene` names it `<strips>.run<UTC>.ckpt_ep<N>_step<S>.npz`). `play_gvd_tech.bat` attaches and forwards extra arguments, so the same flags can be passed after the bat's own `--backend beamngpy --viz`.

Engage from the GVD VISION window. `q` quits the supervisor and leaves BeamNG open. Each tick is flushed, so a crash still leaves the rows.

`--drive-log-gt` needs the map's painted lines (`<data>\roads\<map>_lines.json.gz`, or `BNG_HOME\content\levels\<map>.zip`). The 8 m value is `lane_at` → `tech_from_painted` → `lane_x_from_vector`, the training-label path. A missing side is null, not a guessed center. The default frame is the ref node (`state_pos_ref_node`). A checkpoint whose meta `gvd_gt_origin` is `body_centreline` uses that frame when the vehicle jbeam is readable; otherwise `gt_lane_x` stays null.

## Score

```
py -3 -m python.debug.drive_log_report E:\GVD_data\logs\drive.jsonl
```

Prints mean and RMS of `model_lane_x - gt_lane_x`, the sign-agreement percent, and every tick where `sign(steer_cmd) != sign(lane_x)`. `post_law` means the lane law (`steer_law`) still agreed with `lane_x` and a later step replaced the command.
