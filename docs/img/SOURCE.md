# docs/img sources

These stills are offline previews, not live captures. The JPEGs in this folder were downscaled and compressed (each under 400 KB) from this project's own render outputs. They are not simulator ground-truth debug views.

`viz-ribbon.png` was already in the repo: an offline render of the **classic** GVD VISION stage (now a disabled option; unlock with `--view classic --enable-classic-view`).

## Shared clip and depth checkpoint

All three Reality View JPEGs come from the same recorded rec_v2 clip and the same scene-net checkpoint:

- Clip: `rec_v2_8h_20261124/clips_20261124_223309/clip_015_heavy_traffic_west_coast_usa` (206 frames; recorded on the dev PC, 2026-11-24).
- Checkpoint: `ctl_20261105_p01.run20261107T011144Z.ckpt_ep11_step86808.npz` under `sgt_combined_aux_ds_20261106/L1/ckpt/` (aux_depth + aux_sem, epoch 11, 12×16 grid, sky class 21 masked). Built with the trainer code at `4e41994` (PR #180).
- Current `main` (and `8ce34d5`, the C-prototype renderer pin) **refuses** this checkpoint: `object_class_head` is 96 outputs (pre-v2, six classes) vs 80 (`N_OBJECTS` 16 × 5 classes). See `python/train/scene_net.py` `check_object_class_head`. `scripts/migrate_ckpt_classes.py` is opt-in; refusal stays the default. Depth for these stills was therefore built from a checkout that still matched the checkpoint, not from `main`.

## C prototype still (`rv_offline_c_prototype_frame110.jpg`)

- Source render: `rv_still_frame110.png` (frame 110 of the clip above). In-image footer: "OFFLINE PREVIEW, not live".
- Renderer: `python/viz/reality_view_c.py` (option C prototype, on `main`) at `8ce34d5` (`8ce34d5c5b4a480c0188b3f5d2610b72045dca4b`, ancestor of current `main`). CPU only. Chase pose used: y=-4, z=2.6, pitch=-12 (written on the still). No simulator ground truth is drawn.
- Regenerating the PNG (needs the clip, the checkpoint, and a model-build checkout at `4e41994`; `--repo` is a checkout of the C renderer at `8ce34d5` or later `main`):

```
py -3 tools/rv_preview_from_clip.py --repo <renderer-checkout> --model-repo <ckpt-code-checkout@4e41994> --clip <clip_015_heavy_traffic_west_coast_usa> --ckpt <ctl_20261105_p01.run20261107T011144Z.ckpt_ep11_step86808.npz> --out <out-dir> --main-sha 8ce34d5 --chase-y -4 --chase-z 2.6 --chase-pitch -12 --unproj-w 320 --unproj-h 240
```

The driver used for the original still is `tools/rv_preview_from_clip.py` from the preview work (same flags). Then JPEG-compress the frame-110 PNG, long side 1400, quality 85.

## 8-camera fusion stills (`rv_offline_preview_frame180.jpg`, `rv_offline_preview_frame040.jpg`)

- Source renders: `rv_v3_still_frame180.png` and `rv_v3_still_frame040.png` (frames 180 and 40, chase pose A). In-image footer: "OFFLINE PREVIEW, not live". The right-hand inset is a main-camera-only comparison, not simulator ground truth.
- These use code that is **not on `main`**:
  - Multi-camera fusion library: branch `gvd/rv-multicam` (PR #264, draft, unmerged) at `74a6a72` (`74a6a72a1ad8aeac41264b0a3c4ddb9327301862`).
  - Ego overlay: branch `gvd/rv-ego-model3` (PR #273, draft, unmerged). The stills were rendered from that in-progress mesh; the first commit on the branch is `d8e7a77` (after the stills were written). Later commits on that branch (`5f2fa45`, `7c8ab19`) strip the rear T badge and were **not** used for these stills. Pose A shows the roof and hood; the hood badge was already removed. The mesh is CC BY 4.0, David_Holiday.
- Regenerating the PNGs (needs the clip, a `preds.npz` of per-camera aux depth from the checkpoint above, a checkout of `gvd/rv-multicam` @ `74a6a72`, and `python/viz/rv_ego_model.py` from `gvd/rv-ego-model3`):

```
set RV_EGO_MODEL_PY=<ego-checkout>/python/viz/rv_ego_model.py
py -3 tools/rv_multicam_preview_v3.py --repo <rv-multicam-checkout@74a6a72> --clip <clip_015_heavy_traffic_west_coast_usa> --preds <preds.npz> --out <out-dir> --pose A --frames 40,180
```

The driver used for the original stills is `tools/rv_multicam_preview_v3.py` from the preview work (same flags). Then JPEG-compress, long side 1400, quality 85.

The original PNGs and `preds.npz` are preview artifacts, not files in this repo. The clip and checkpoint live on the dev PC data drive, not in git.
