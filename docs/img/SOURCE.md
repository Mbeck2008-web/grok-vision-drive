# docs/img sources

`viz-ribbon.png` was already in the repo (offline render of the classic GVD VISION stage).

The Reality View JPEGs were exported from this project's own offline preview stills on the shared box, then downscaled and JPEG-compressed for the README (each under 400 KB). They are not live captures.

- `rv_offline_c_prototype_frame110.jpg` from `/workspace/rv-preview/rv_still_frame110.png`. Offline CPU render of `python/viz/reality_view_c.py` (option C prototype, on main, flag default off). Clip: `rec_v2_8h_20261124/clips_20261124_223309/clip_015_heavy_traffic_west_coast_usa`, frame 110. Depth is the scene-net `aux_depth` head (12x16, upsampled). No simulator ground truth is drawn. The still itself is labelled "OFFLINE PREVIEW, not live".

- `rv_offline_preview_frame180.jpg` and `rv_offline_preview_frame040.jpg` from `/workspace/rv-preview/v3/rv_v3_still_frame180.png` and `rv_v3_still_frame040.png`. Offline 8-camera fusion preview of the same clip (frames 180 and 40, chase pose A). Multi-camera fusion and the Model 3 ego overlay are not the live GVD VISION path on main (those pieces are still in progress). No simulator ground truth is drawn. Pose A shows the roof and hood; the hood badge was already removed from the mesh in `cpp/viz/assets`. The stills themselves are labelled "OFFLINE PREVIEW, not live".
