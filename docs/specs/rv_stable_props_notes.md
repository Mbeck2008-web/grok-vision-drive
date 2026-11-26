# rv-stable-fill property tests (scripts/test_rv_stable_props.py)

Breaker (swarm-4), test-only. Thresholds committed in 723ccce before any run on 783fedd.
Runs the real `cam_frame` from `tools/rv_stable_preview.py` (parsed + executed) on fake preds;
only the 2-line CamDepthFilter call from the preview main loop is mirrored.

## Known failing on 783fedd (fail -> pass tests for the fix)

| check | invariant | counterexample | code |
|---|---|---|---|
| check_pin_label_edge_dip | A | S9 tunnel fixture, wide: label-edge px z=-0.2376 m at ground range 13.97 m | preview depth-edge cut (after clamp) |
| check_pin_object_base_dip | A | exact ground + 3x12 px face at ~8 m, wide: road under it z=-0.1276 m at 9.53 m | edge cut |
| check_pin_horizon_band_dip | A | exact flat ground, no road label, wide: dips from 17.76 m, z=-1.02 m at 29.9 m | edge cut |
| prop_near_rows_no_dip | A | narrow, bend (1, 1, 0.5), road rows >= 50%: z=-0.032 m at 9.28 m | edge cut |
| check_pin_filter_unsnaps_road / prop_filter_keeps_ground | A | perfect pose, exact ground: wide z=-0.053 m at 14.96 m; narrow 1 m step z=-0.0063 m | CamDepthFilter.update |
| check_pin_filter_far_past_cap / prop_filter_keeps_cap | D | static wall past cap, yaw 3 deg/frame: wide [239.98, 260.43] m; narrow 1 deg: [249.49, 250.51] | CamDepthFilter.update |
| check_pin_clamp_inf_then_cap | A | +inf on a downward ray survives ground_clamp, far_to_cap puts it at z=-274 m | ground_clamp `np.isfinite(d)` (unit; not reachable from preds) |
| check_pin_objects_snapped_flat / prop_objects_keep_height | O | 12x16 class grid: main curb@20 m 50/100 px to z=0; pole@4 m GT 1.11 m -> 0.00 | road_snap on coarse labels |
| check_clip015_near_road_bias | H | needs clip015 (not on the box); belongs to gvd/rv-ground-plane | - |

Evidence: /workspace/swarm/swarm-4/rv-evidence/ (props_report_<sha>.log, measure_<sha>.log).
