# Scene-checkpoint shadow overlay — Spec / AC

**Status:** design for implementation on branch `gvd/scene-ckpt-shadow`  
**Repo:** `Mbeck2008-web/grok-vision-drive`  
**Date:** 2026-11-09 (America/Chicago)  
**Scope gate:** smallest coherent change so a `train_scene` epoch `.npz` can be loaded live, run, and drawn. No CloudAgent. No merge. No PC run in this change.

## Intent

`python.run_vision --scene-ckpt <npz>` loads a `train_scene` epoch checkpoint (`gvd-epoch-ckpt-1`), runs `SceneNet` on the **main** camera that live capture already resized to `live_input_long_side` (~640), and draws the `lane_gt` prediction (lateral lane-center / path at the trained lookaheads, including the live 8 m sample) as a **distinct** overlay next to the existing viz.

This is **SHADOW ONLY**. The prediction never reaches actuation. While the flag is set, Engage is hard-blocked (same rule as `--no-engage`): no key, UI, bot file, or `--force-engage` can drive from this model, or from anything else.

Inference stays on **CPU** (numpy/torch CPU) so VRAM stays with BeamNG (budget ≤ 4 GB extra VRAM; supervisor RSS well under 12 GB). The forward pass must match `train_scene` / `SceneNet.forward` on the same canvas.

## Non-goals

- Feeding this prediction into `--lane-source scene`, `path_ego` used by the wheel, `follow_lane_steer`, pedals, or any `DriveCommand`.
- Replacing Hough / the existing ONNX `SceneLaneRuntime` path (that path still actuates when `--lane-source scene` is set; `--scene-ckpt` does not turn it on).
- Changing `train_scene --export` (it still traces a fresh net when no checkpoint is given). Exporting a given checkpoint already exists: `scripts/export_epoch_onnx.py`.
- CUDA / GPU inference, eight-camera live eval, ctl-head actuation, cam-age models (refuse a `cam_age` / `gvd_cam_age` checkpoint the same way the ONNX runtime does).
- Running BeamNG or recording a clip as part of this PR. The Windows command is documented only.
- Merging to main.

## Existing contracts (inspected)

| Contract | Location | Rule kept |
| --- | --- | --- |
| Epoch `.npz` | `python/train/epoch_ckpt.py` `FORMAT=gvd-epoch-ckpt-1` | `load_epoch_ckpt` + `load_into_net` (no silent partial load) |
| Architecture | `epoch_ckpt.build_net` | cams / scale / backbone / cam_age / ctl / cond_dim from meta |
| Aux heads | checkpoint weights (`aux_*_head`, `pause_arch` `:aux=`) | Production aux-run files may omit `aux_sensors` in meta; infer from weights/arch so those files load |
| Live canvas | `compose_strip` + `extract_cams` / `prepare_model_strip` | Main sector 256×192 after the same resize the collector uses; live frames arrive at `live_input_long_side` 640 |
| `lane_gt` | `python/debug/lane_targets.py` | 14-vector; `lane_x_from_vector` at 8 m; `confidence_from_vector` |
| Engage disable | `run_vision --no-engage` | Reused: `--scene-ckpt` forces this on |
| Clips | `--encode` / `ClipRecorder` / key C | Unchanged encoder path; when the flag is set, the pushed main frame carries the overlay so a QSV clip shows the prediction |
| Shadow gate doc | `docs/SHADOW_GATE.md` | `--policy shadow` is not this gate (it still actuates modular). This flag is the shadow path for a trained scene ckpt. |

## Design

1. **CLI:** `python.run_vision --scene-ckpt PATH` (kebab-case, matches `--scene-onnx`). Missing / unreadable / wrong-format files exit before the loop with a clear error.
2. **Loader:** new `python/train/scene_ckpt_runtime.py`. Load via `load_epoch_ckpt`, `build_net` (extended to honor / infer `aux_sensors` and `road_edge`), `load_into_net`, `net.cpu().eval()`. No `.cuda()`. A `cam_age` checkpoint is refused. `ctl` weights may load; their outputs are ignored.
3. **Tick:** take `bundle.main_bgr()` (already ~640 long side). Reject a frame that is not HxWx3. Compose with `compose_strip({"main": frame})` and pack to the checkpoint's cams (main-only extracts the 256×192 sector). Run `SceneNet.forward` with the real `dt` and persistent GRU state. Read `lane_gt` only.
4. **Isolation:** store the result under `state["scene_ckpt"]` for HUD/overlay. Do **not** assign it to `pout.path_ego`, `lane_x`, or the command. `--scene-ckpt` sets `args.no_engage = True` at parse time.
5. **Overlay:** `draw_scene_ckpt_overlay(bgr, pred)` paints a distinct color (not the ice-blue corridor) of the predicted center path plus a `SHADOW` caption (`lane_x`, confidence). Applied to the main frame used for M4 clips and the VISION PIP.
6. **Import cost:** `run_vision` imports the runtime only when the flag is set (same reason `scene_live` is torch-free).

## File touch list

- `docs/superpowers/specs/2026-11-09-scene-ckpt-shadow-design.md` (this file)
- `python/train/scene_ckpt_runtime.py` (new: load, infer, overlay, isolation helpers)
- `python/train/epoch_ckpt.py` (`build_net` infers aux / road_edge)
- `python/run_vision.py` (`--scene-ckpt`, force no-engage, tick + clip frame)
- `scripts/test_scene_ckpt_shadow.py` (new)

## Acceptance criteria

1. **Spec committed** on `gvd/scene-ckpt-shadow` with this document.
2. **Loader round-trip:** write a tiny main-only CPU checkpoint from a seeded `SceneNet`, load it, run the same fixture frame (480×640×3, long side 640) through the runtime and through `SceneNet.forward` on the composed canvas; `lane_gt` matches within 1e-5.
3. **Aux file without meta flag:** a net trained with `aux_sensors=True` written without `aux_sensors` in meta still loads (weights present); forward matches.
4. **Shape / channel:** HxWx3 uint8 main is accepted and packed to the net canvas (192×256×3 for `--cams main`); grayscale or 4-channel raises a clear error; a missing frame returns an empty prediction (no crash).
5. **Shadow cannot actuate:** `--scene-ckpt` + `--force-engage` (no `--no-engage` on the CLI) keeps every stub tick `engaged=0` and never copies the prediction into the actuated command. A helper proves a command is unchanged by a prediction.
6. **Overlay:** drawing on a stub camera frame changes pixels and writes a visible `SHADOW` mark in a color distinct from the existing ice-blue viz.
7. **Offline suite** (`scripts/run_offline_suite.py`) is green on this branch.
8. **PR to main** with spec, diff summary, and evidence. Do not merge.

## Verify steps

```
# from repo root, CPU torch interpreter
python scripts/test_scene_ckpt_shadow.py
python scripts/run_offline_suite.py
```

## Windows command (Desktop-IQU45; not run in this change)

Documented in the PR after the flag exists. Uses `--encode qsv` (or `auto`, which prefers QSV) and the epoch-11 checkpoint path. Press **C** to flush a clip (existing M4 path). `--scene-ckpt` blocks Engage.

## Out of scope follow-ups

- Optional `train_scene --export` from a given checkpoint (use `scripts/export_epoch_onnx.py` today).
- In-game recording and the video itself (separate step on the PC, after this PR).
