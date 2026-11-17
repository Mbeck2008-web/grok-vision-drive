# Vector stage

## Goal

The live GVD VISION stage defaults to a black chase view: dotted lanes, a dotted path from the planner or scene ribbon, wireframe boxes, signal icons, and a small speed readout. An optional overlay draws the same elements on the main camera and adds a speed bar. The previous ice-ribbon cabin stays available.

## Non-goals

- No change to actuation, heartbeat, AEB, engage, collection, training, or the Lua mod.
- No BeamNG ground truth in the live picture. `path_world` and `debug_gt` do not move pixels.
- No new dependency. OpenCV and numpy only. The stage stays on the CPU.
- No second product name on the picture. The mark is GVD.
- The multi-panel board, the offline label film, and the showcase presets are separate views. They reuse these drawers.

## Files

| File | Role |
| --- | --- |
| `python/viz/early_vector.py` | Dots, wires, ego, hud, overlay |
| `python/viz/stage.py` | `VizUI.style` / `overlay`. Legacy remains the function default so older cabin tests keep their pixels. The supervisor sets `early`. |
| `python/run_vision.py` | `--viz-style early\|legacy` (default early), `--viz-overlay` |
| `scripts/test_viz_early_fsd.py` | Pixel checks and the 1280x720 timing print |
| `scripts/render_viz_gallery.py` | Offline stills |
| `docs/images/viz_early_fsd/` | intersection, multilane, parking, overlay |

`render_stage` without a style still paints the ice ribbon. `python run_vision.py --viz` paints the vector stage.

## Verify

```
python scripts/test_viz_early_fsd.py
python scripts/test_gvd_viz_stage.py
python scripts/run_offline_suite.py
```

The vector test prints a median millisecond time at 1280x720. The budget is under 8 ms.

## Debug board

Off unless `--viz-debug` or key `b`. Key `d` still opens the DRIVE tab. While the board is off, `render_debug_board` is not called.

Panels, left to right: a perspective grid with the vector scene, up to three camera tiles with a raw readout (a flag above 0.5 is green), a top-down map with track ids, and speed / yaw / pedal graphs. Camera tiles use the frames the supervisor already has. Empty state still returns a frame.

`python/viz/debug_board.py`, `scripts/test_viz_debug_board.py`.

## Label film

`scripts/render_label_video.py <recording_dir> --out clip.mp4` reads `frames/` or `raw/main/` and `labels.jsonl` (else `gt.jsonl` or `state.jsonl`). It grays the camera, then draws whatever fields exist: yellow wires, red when `in_path` or the lead id matches, green lane polylines, a thick red stop bar, blue curb squares, sign wires, orange cone wires, a stop mark when a stop is present, a right-edge ruler, and a plain readout. A short label file leaves later frames empty. Nothing under `python/runtime`, `python/viz`, or `run_vision.py` imports it.

`python/offline/label_review.py`, `scripts/test_viz_label_review.py`.

## Showcase

`--viz-showcase` with `--viz-showcase-layout full|split|pip` and `--viz-preset square|story|wide` (1080x1080, 1080x1920, 1920x1080). The frame is the vector stage plus a GVD mark. Split puts the camera on the left. Pip insets it. The nerd panel is not attached. `lerp_state` is for the offline clip only.

`python/viz/showcase.py`, `scripts/render_showcase.py`, `scripts/test_viz_showcase.py`, `docs/images/viz_showcase/`.
