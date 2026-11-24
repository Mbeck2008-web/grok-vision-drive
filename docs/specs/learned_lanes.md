# Learned lanes in the live lane path (Phase 3 groundwork)

Status: **flag default OFF**. No trained SceneNet lane weights and no held-out
learned-vs-Hough eval exist, so learned-lane accuracy is **UNPROVEN**. The flag
flips to ON only when `scripts/eval_learned_lanes.py` reports
`learned_beats_hough: true` on real held-out data.

## Switch

- `--learned-lanes` / `--no-learned-lanes` (CLI wins), else `GVD_LEARNED_LANES=1`, else OFF.
- Only with `--lane-source hough` (the default). `scene` and `gt` keep their own paths
  (`scene` still stops on low confidence and never falls back to Hough).
- OFF: Hough exactly as before; `lane_head = {"mode": "hough", "reason": "flag_off"}`.

## Per tick with the flag ON (`python/perception/lane_select.py`)

1. Learned: `SceneLaneRuntime` (ORT CPU EP, as today) on the 8-camera strip; read
   `lanes`, `lane_valid`, `curbs`, `curb_valid` (+ `lane_gt` conf). Slots with
   `lane_valid >= 0.5` become painted lanes; curbs go to `curbs_bev`, never `lanes_bev`.
2. Fallback to Hough (`lanes_from_view`) with an explicit reason when the learned
   output is not usable: `model_missing`, `model_refused`, `no_frame`, `runtime_error`,
   `lanes_untrained`, `bad_shape`, `nonfinite_output`, `out_of_range`, `low_conf`
   (no slot >= 0.5, or conf < 0.40), `learned_disabled` (shape error, latched).
3. Sanitize: drop non-finite / out-of-box points (|x| <= 15 m, -10 <= y <= 120 m), clamp conf and curvature.
4. Paint gate: no painted-stripe evidence in the forward image -> zero lanes, from either source.
5. Telemetry: `PerceptionOut.lane_head` / `gvd_state.json` `lane_head`; a `[GVD] lane head:` line on every change.

Heartbeat / AEB / engage are untouched: lanes feed the planner exactly where Hough did.

## Invariants (tested in `scripts/test_learned_lanes_props.py`)

- I1 no paint, no lines (learned or fallback)
- I2 edges are not paint (curbs never in `lanes_bev`; edge-only scene -> no lanes)
- I3 never silent (every fallback has a reason)
- I4 finite and bounded (any frame size, any head output)
- I5 deterministic

## Counterexamples on main 3e4a577 (hypothesis, shrunk)

- Hough on noise: `noise(h=96, w=128, seed=0, lo=0)` -> 2 lanes, conf 0.95.
- Hough on an edge-only road: `edge_only(96, 128, side='grass', curb_v=185, curb_px=12, horizon=0.5,
  half_bottom=0.5, half_top=0.125, shift=0.125)` -> 1 lane, conf 0.4.
- Naive learned wiring (`scene_to_predict_kwargs`): `curb_valid=[0.0, 1.0]` -> a curb in `lanes_bev`;
  `lanes[0,3,0]=nan` -> `{'x': nan}` in `lanes_bev`.

## Limits

- The paint gate is a pixel heuristic. A bright band narrower than `PAINT_MAX_WIDTH_PX` (24 px)
  looks like paint and counts; a worn or very hazy stripe it misses gives no lanes (a stop).
  It is frame-level: with real paint present, a Hough curb line is not removed.
- The row-continuity check assumes a camera-like aspect (stripe moves <= ~4 px per row).
- All 87 real BeamNG fixture frames in `tests/fixtures` pass the gate; that is not a
  held-out accuracy result.
