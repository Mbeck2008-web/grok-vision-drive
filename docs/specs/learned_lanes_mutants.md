# lane_select.py mutants: every eccf5c4 survivor, classified

- r3 (eccf5c4): 907 mutants, 741 killed, 164 survived, 2 segfault.
- r4 (this branch, new checks in scripts/test_learned_lanes_props.py): {'killed': 862, 'survived': 43, 'segfault': 2}.
  mutmut 3.8.1, CPU, `mut-lanes/` (paths: python/perception/lane_select.py; tests: tests_lanes/ pytest wrapper over the props script). Log: mutmut_run4.log, results: mutmut_results_r4.txt.
- The 2 segfaults (paint_evidence_px 133 `mask = None`, 144 `connectedComponentsWithStats(None)`) crash cv2 and the test process; the mutant cannot pass. Not survivors.
- Of the 164 r3 survivors: killed 121, equivalent 43, unclassified 0.
- "KILLED by" names the check written to kill that mutant (by design; mutmut stops at the first failing test and does not record which). All 121 are reported killed by mutmut r4.
- Equivalents cross-checked empirically where they are pure functions of an image (equiv_crosscheck.py / .log).
- No tolerance, threshold or invariant was changed. lane_select.py changed only in docstrings/comments.

| mutant | function | mutation | verdict |
|---|---|---|---|
| x_paint_evidence_px__mutmut_2 | paint_evidence_px | `return 1` | KILLED by check_paint_gate_input_forms (None -> 0) |
| x_paint_evidence_px__mutmut_5 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 4 and arr.shape[1] < 4:` | EQUIVALENT: height < 4 is caught by the ROI check (rows < 4 -> 0); width < 5 gives ridge <= 0 (needs x-tau and x+tau in frame, tau >= 2) -> 0 |
| x_paint_evidence_px__mutmut_6 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 and arr.shape[0] < 4 or arr.shape[1] < 4:` | KILLED by check_paint_gate_input_forms (2-channel frame -> 0; mutant raises in cvtColor) |
| x_paint_evidence_px__mutmut_13 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[1] < 4 or arr.shape[1] < 4:` | EQUIVALENT: height check dropped: h < 4 always leaves < 4 ROI rows -> same 0 |
| x_paint_evidence_px__mutmut_14 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] <= 4 or arr.shape[1] < 4:` | EQUIVALENT: h == 4 leaves 3 ROI rows -> ROI check returns 0 anyway |
| x_paint_evidence_px__mutmut_15 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 5 or arr.shape[1] < 4:` | EQUIVALENT: h == 4 leaves 3 ROI rows -> ROI check returns 0 anyway |
| x_paint_evidence_px__mutmut_17 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 4 or arr.shape[1] <= 4:` | EQUIVALENT: w == 4: no x has both x-2 and x+2 in frame, ridge <= 0, mask empty -> 0 |
| x_paint_evidence_px__mutmut_18 | paint_evidence_px | `if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 4 or arr.shape[1] < 5:` | EQUIVALENT: w == 4: ridge <= 0 everywhere -> 0 |
| x_paint_evidence_px__mutmut_22 | paint_evidence_px | `img = np.ascontiguousarray(arr[:, :, :4])` | EQUIVALENT: cv2 COLOR_BGR2HSV accepts 3 or 4 channels and ignores the 4th; checked: BGRA == BGR in paint_gate_input_forms |
| x_paint_evidence_px__mutmut_23 | paint_evidence_px | `if img.dtype == np.uint8:` | KILLED by check_paint_gate_colour_boundaries / count_definition via paint_both (float64 frame == uint8 frame) |
| x_paint_evidence_px__mutmut_27 | paint_evidence_px | `img = np.clip(np.nan_to_num(img.astype(np.float64)), None, 255).astype(np.uint8)` | KILLED by check_paint_gate_input_forms (negative float stripe is black, not wrapped bright) |
| x_paint_evidence_px__mutmut_28 | paint_evidence_px | `img = np.clip(np.nan_to_num(img.astype(np.float64)), 0, None).astype(np.uint8)` | KILLED by check_paint_gate_input_forms (overexposed 300 float stripe counts like 255) |
| x_paint_evidence_px__mutmut_33 | paint_evidence_px | `img = np.clip(np.nan_to_num(img.astype(None)), 0, 255).astype(np.uint8)` | EQUIVALENT: ndarray.astype(None) is float64 |
| x_paint_evidence_px__mutmut_34 | paint_evidence_px | `img = np.clip(np.nan_to_num(img.astype(np.float64)), 1, 255).astype(np.uint8)` | KILLED by check_paint_gate_colour_boundaries (float copy of sat-ridge-12 frame: 0 -> 1 drops ridge to 11) |
| x_paint_evidence_px__mutmut_35 | paint_evidence_px | `img = np.clip(np.nan_to_num(img.astype(np.float64)), 0, 256).astype(np.uint8)` | KILLED by check_paint_gate_input_forms (300 -> clip 256 -> wraps to 0) |
| x_paint_evidence_px__mutmut_42 | paint_evidence_px | `if img.shape[1] < 4:` | KILLED by check_paint_gate_input_forms (ROI under 4 rows -> 0) |
| x_paint_evidence_px__mutmut_43 | paint_evidence_px | `if img.shape[0] <= 4:` | KILLED by check_paint_gate_input_forms (ROI exactly 4 rows is read: 14*4) |
| x_paint_evidence_px__mutmut_44 | paint_evidence_px | `if img.shape[0] < 5:` | KILLED by check_paint_gate_input_forms (ROI exactly 4 rows is read: 14*4) |
| x_paint_evidence_px__mutmut_45 | paint_evidence_px | `return 1` | KILLED by check_paint_gate_input_forms (ROI under 4 rows -> 0, not 1) |
| x_paint_evidence_px__mutmut_52 | paint_evidence_px | `sat = hsv[:, :, 1].astype(None)` | EQUIVALENT: float64 instead of int16 S: differences of 0..255 values are exact in both |
| x_paint_evidence_px__mutmut_53 | paint_evidence_px | `sat = hsv[:, :, 2].astype(np.int16)` | KILLED by check_paint_gate_colour_boundaries (yellow on grey of the same V: only the S ridge sees it) |
| x_paint_evidence_px__mutmut_55 | paint_evidence_px | `val = hsv[:, :, 2].astype(None)` | EQUIVALENT: float64 instead of int16 V: same values |
| x_paint_evidence_px__mutmut_57 | paint_evidence_px | `run3 = None` | KILLED by check_paint_gate_stripe_shape_rules (slanted 4 px stripe, 2 px/row; 3x3 open erases it) |
| x_paint_evidence_px__mutmut_59 | paint_evidence_px | `run3 = np.ones((1, 3), None)` | EQUIVALENT: cv2 morphology uses kernel != 0; float ones kernel == uint8 ones kernel |
| x_paint_evidence_px__mutmut_61 | paint_evidence_px | `run3 = np.ones((1, 3), )` | EQUIVALENT: np.ones default dtype float64; same structuring element |
| x_paint_evidence_px__mutmut_62 | paint_evidence_px | `run3 = np.ones((2, 3), np.uint8)` | KILLED by check_paint_gate_stripe_shape_rules (slanted stripe; 2x3 open erases it) |
| x_paint_evidence_px__mutmut_63 | paint_evidence_px | `run3 = np.ones((1, 4), np.uint8)` | KILLED by check_paint_gate_stripe_shape_rules (3 px run counts; 1x4 open erases it) |
| x_paint_evidence_px__mutmut_66 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 255, 255)), None, run3) > 0` | KILLED by check_paint_gate_count_definition (yellow: (W+8)*(R-4); erode instead of open shrinks W) |
| x_paint_evidence_px__mutmut_67 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 255, 255)), cv2.MORPH_OPEN, None) > 0` | KILLED by check_paint_gate_stripe_shape_rules (slanted yellow stripe; default 3x3 kernel erases it) |
| x_paint_evidence_px__mutmut_77 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (9, 28, 80), (42, 255, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (yellow H 8 counts) |
| x_paint_evidence_px__mutmut_78 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 29, 80), (42, 255, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (yellow S 28 counts) |
| x_paint_evidence_px__mutmut_79 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 81), (42, 255, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (yellow V 80 counts) |
| x_paint_evidence_px__mutmut_80 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (43, 255, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (yellow H 43 does not count) |
| x_paint_evidence_px__mutmut_81 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 256, 255)), cv2.MORPH_OPEN, run3) > 0` | EQUIVALENT: S upper 255 -> 256: HSV of uint8 BGR is H 0..179, S/V 0..255 |
| x_paint_evidence_px__mutmut_82 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 255, 256)), cv2.MORPH_OPEN, run3) > 0` | EQUIVALENT: V upper 255 -> 256: HSV of uint8 BGR is H 0..179, S/V 0..255 |
| x_paint_evidence_px__mutmut_84 | paint_evidence_px | `yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 255, 255)), cv2.MORPH_OPEN, run3) > 1` | EQUIVALENT: morphologyEx(inRange) output is 0 or 255: > 1 == > 0 |
| x_paint_evidence_px__mutmut_87 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 18, 255)), None, run3) > 0` | KILLED by check_paint_gate_count_definition (white: erode instead of open shrinks W) |
| x_paint_evidence_px__mutmut_88 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 18, 255)), cv2.MORPH_OPEN, None) > 0` | KILLED by check_paint_gate_stripe_shape_rules (slanted white stripe; default 3x3 kernel erases it) |
| x_paint_evidence_px__mutmut_100 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 186), (180, 18, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (white V 185 counts) |
| x_paint_evidence_px__mutmut_101 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (181, 18, 255)), cv2.MORPH_OPEN, run3) > 0` | EQUIVALENT: H upper 180 -> 181: HSV of uint8 BGR is H 0..179, S/V 0..255 |
| x_paint_evidence_px__mutmut_102 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 19, 255)), cv2.MORPH_OPEN, run3) > 0` | KILLED by check_paint_gate_colour_boundaries (white S 19 does not count) |
| x_paint_evidence_px__mutmut_103 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 18, 256)), cv2.MORPH_OPEN, run3) > 0` | EQUIVALENT: V upper 255 -> 256: HSV of uint8 BGR is H 0..179, S/V 0..255 |
| x_paint_evidence_px__mutmut_105 | paint_evidence_px | `white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 18, 255)), cv2.MORPH_OPEN, run3) > 1` | EQUIVALENT: morphologyEx(inRange) output is 0 or 255: > 1 == > 0 |
| x_paint_evidence_px__mutmut_115 | paint_evidence_px | `mask[(yellow & (_ridge(sat) > 12)) ¦ (white & (_ridge(val) >= 18))] = 255` | KILLED by check_paint_gate_colour_boundaries (sat ridge exactly 12 counts) |
| x_paint_evidence_px__mutmut_116 | paint_evidence_px | `mask[(yellow & (_ridge(sat) >= 13)) ¦ (white & (_ridge(val) >= 18))] = 255` | KILLED by check_paint_gate_colour_boundaries (sat ridge exactly 12 counts) |
| x_paint_evidence_px__mutmut_119 | paint_evidence_px | `mask[(yellow & (_ridge(sat) >= 12)) ¦ (white & (_ridge(val) > 18))] = 255` | KILLED by check_paint_gate_colour_boundaries (value ridge exactly 18 counts) |
| x_paint_evidence_px__mutmut_120 | paint_evidence_px | `mask[(yellow & (_ridge(sat) >= 12)) ¦ (white & (_ridge(val) >= 19))] = 255` | KILLED by check_paint_gate_colour_boundaries (value ridge exactly 18 counts) |
| x_paint_evidence_px__mutmut_124 | paint_evidence_px | `joined = cv2.dilate(mask, None)` | KILLED by check_paint_gate_count_definition (3x3 join gives (W+2)*(R-2), not (W+8)*(R-4)) |
| x_paint_evidence_px__mutmut_128 | paint_evidence_px | `joined = cv2.dilate(mask, np.ones((1, 9), None))` | EQUIVALENT: float kernel for dilate: same structuring element |
| x_paint_evidence_px__mutmut_130 | paint_evidence_px | `joined = cv2.dilate(mask, np.ones((1, 9), ))` | EQUIVALENT: np.ones float64 kernel for dilate: same element |
| x_paint_evidence_px__mutmut_131 | paint_evidence_px | `joined = cv2.dilate(mask, np.ones((2, 9), np.uint8))` | KILLED by check_paint_gate_count_definition (2x9 join adds a row) |
| x_paint_evidence_px__mutmut_132 | paint_evidence_px | `joined = cv2.dilate(mask, np.ones((1, 10), np.uint8))` | KILLED by check_paint_gate_count_definition (1x10 join gives W+9) |
| x_paint_evidence_px__mutmut_135 | paint_evidence_px | `mask = cv2.erode(joined, None)` | KILLED by check_paint_gate_count_definition (3x3 erode instead of 5x1) |
| x_paint_evidence_px__mutmut_139 | paint_evidence_px | `mask = cv2.erode(joined, np.ones((PAINT_MIN_ROWS, 1), None))` | EQUIVALENT: float kernel for erode: same element |
| x_paint_evidence_px__mutmut_141 | paint_evidence_px | `mask = cv2.erode(joined, np.ones((PAINT_MIN_ROWS, 1), ))` | EQUIVALENT: np.ones float64 kernel for erode: same element |
| x_paint_evidence_px__mutmut_142 | paint_evidence_px | `mask = cv2.erode(joined, np.ones((PAINT_MIN_ROWS, 2), np.uint8))` | KILLED by check_paint_gate_count_definition (5x2 erode gives W+7) |
| x_paint_evidence_px__mutmut_145 | paint_evidence_px | `n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=None)` | EQUIVALENT: OpenCV binding maps connectivity=None to the default 8 (checked: same n as 8, differs from 4) |
| x_paint_evidence_px__mutmut_147 | paint_evidence_px | `n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, )` | EQUIVALENT: connectivity default is 8 |
| x_paint_evidence_px__mutmut_165 | paint_evidence_px | `if max(bw, bh) > PAINT_MIN_RUN_PX:` | KILLED by check_paint_gate_stripe_shape_rules (component exactly PAINT_MIN_RUN_PX=12 long counts) |
| x_paint_evidence_px__mutmut_166 | paint_evidence_px | `total = int(stats[i, cv2.CC_STAT_AREA])` | KILLED by check_paint_gate_count_definition (two stripes add up) |
| x__curvature__mutmut_6 | _curvature | `if float(np.ptp(ys)) <= 1.0:` | KILLED by check_learned_parse_contract (lane spanning exactly 1 m counts) |
| x__curvature__mutmut_7 | _curvature | `if float(np.ptp(ys)) < 2.0:` | KILLED by check_learned_parse_contract (1 m span counts) |
| x__curvature__mutmut_8 | _curvature | `break` | KILLED by check_learned_parse_contract (short lane first does not stop the others) |
| x__default_runtime_factory__mutmut_1 | _default_runtime_factory | `return SceneLaneRuntime(None)` | KILLED by check_wiring_contract (--scene-onnx path reaches SceneLaneRuntime; load_reason names it) |
| x__poly_dicts__mutmut_7 | _poly_dicts | `return [{"x": float(x), "y": float(y), "XXzXX": 0.0} for x, y in points]` | KILLED by check_learned_parse_contract (points are Hough's {x,y,z=0.0}) |
| x__poly_dicts__mutmut_8 | _poly_dicts | `return [{"x": float(x), "y": float(y), "Z": 0.0} for x, y in points]` | KILLED by check_learned_parse_contract (point schema) |
| x__poly_dicts__mutmut_9 | _poly_dicts | `return [{"x": float(x), "y": float(y), "z": 1.0} for x, y in points]` | KILLED by check_learned_parse_contract (z == 0.0) |
| x__poly_ok__mutmut_11 | _poly_ok | `x = float(p.get("x", float("NAN")))` | EQUIVALENT: float("NAN") is nan |
| x__poly_ok__mutmut_22 | _poly_ok | `y = float(p.get("y", float("NAN")))` | EQUIVALENT: float("NAN") is nan |
| x__ridge__mutmut_4 | _ridge | `best = np.full(chan.shape, -32768, dtype=None)` | EQUIVALENT: np.full(..., -32768) without dtype is int64: same values, wider type |
| x__ridge__mutmut_7 | _ridge | `best = np.full(chan.shape, -32768, )` | EQUIVALENT: same as 4 (int64) |
| x__ridge__mutmut_12 | _ridge | `if tau > w:` | EQUIVALENT: tau == w adds a 0 ridge (left = right = chan): only lifts negative values to 0, never across >= 12 / >= 18 (sole caller) |
| x__ridge__mutmut_13 | _ridge | `break` | EQUIVALENT: PAINT_TAUS ascending: once tau >= w every later tau is too; break == continue |
| x_build_lane_selector__mutmut_7 | build_lane_selector | `on = learned_lanes_enabled(getattr(args, "learned_lanes", ))` | KILLED by check_wiring_contract (no learned_lanes attribute and no env -> OFF, no exception) |
| x_build_lane_selector__mutmut_15 | build_lane_selector | `source = str(getattr(args, "lane_source", None) or "hough")` | EQUIVALENT: missing lane_source: None or 'hough' == 'hough' or 'hough' |
| x_build_lane_selector__mutmut_18 | build_lane_selector | `source = str(getattr(args, "lane_source", ) or "hough")` | KILLED by check_wiring_contract (missing / empty lane_source means hough) |
| x_build_lane_selector__mutmut_21 | build_lane_selector | `source = str(getattr(args, "lane_source", "XXhoughXX") or "hough")` | KILLED by check_wiring_contract (missing / empty lane_source means hough) |
| x_build_lane_selector__mutmut_22 | build_lane_selector | `source = str(getattr(args, "lane_source", "HOUGH") or "hough")` | KILLED by check_wiring_contract (missing / empty lane_source means hough) |
| x_build_lane_selector__mutmut_23 | build_lane_selector | `source = str(getattr(args, "lane_source", "hough") or "XXhoughXX")` | KILLED by check_wiring_contract (missing / empty lane_source means hough) |
| x_build_lane_selector__mutmut_24 | build_lane_selector | `source = str(getattr(args, "lane_source", "hough") or "HOUGH")` | KILLED by check_wiring_contract (missing / empty lane_source means hough) |
| x_build_lane_selector__mutmut_27 | build_lane_selector | `print_fn("[GVD] lane head: hough (learned lanes OFF; --learned-lanes or GVD_LEARNED_LANES=1)", flush=None)` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_29 | build_lane_selector | `print_fn("[GVD] lane head: hough (learned lanes OFF; --learned-lanes or GVD_LEARNED_LANES=1)", )` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_30 | build_lane_selector | `print_fn("XX[GVD] lane head: hough (learned lanes OFF; --learned-lanes or GVD_LEARNED_LANES=1)XX", flush=True)` | KILLED by check_wiring_contract (OFF start line text) |
| x_build_lane_selector__mutmut_33 | build_lane_selector | `print_fn("[GVD] lane head: hough (learned lanes OFF; --learned-lanes or GVD_LEARNED_LANES=1)", flush=False)` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_38 | build_lane_selector | `print_fn(f"[GVD] lane head: --learned-lanes ignored, --lane-source {source} drives lanes", flush=None)` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_40 | build_lane_selector | `print_fn(f"[GVD] lane head: --learned-lanes ignored, --lane-source {source} drives lanes", )` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_41 | build_lane_selector | `print_fn(f"[GVD] lane head: --learned-lanes ignored, --lane-source {source} drives lanes", flush=False)` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_42 | build_lane_selector | `path = None` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_43 | build_lane_selector | `path = getattr(args, "scene_onnx", "") and None` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_44 | build_lane_selector | `path = getattr(None, "scene_onnx", "") or None` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_46 | build_lane_selector | `path = getattr(args, "scene_onnx", None) or None` | EQUIVALENT: missing scene_onnx: None or None == '' or None |
| x_build_lane_selector__mutmut_49 | build_lane_selector | `path = getattr(args, "scene_onnx", ) or None` | KILLED by check_wiring_contract (no scene_onnx attribute -> path None) |
| x_build_lane_selector__mutmut_50 | build_lane_selector | `path = getattr(args, "XXscene_onnxXX", "") or None` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_51 | build_lane_selector | `path = getattr(args, "SCENE_ONNX", "") or None` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_52 | build_lane_selector | `path = getattr(args, "scene_onnx", "XXXX") or None` | KILLED by check_wiring_contract (no scene_onnx attribute -> path None) |
| x_build_lane_selector__mutmut_54 | build_lane_selector | `selector = LearnedLaneSelector(None, runtime_factory=factory)` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_55 | build_lane_selector | `selector = LearnedLaneSelector(path, runtime_factory=None)` | KILLED by check_wiring_contract (the given factory is the one called) |
| x_build_lane_selector__mutmut_56 | build_lane_selector | `selector = LearnedLaneSelector(runtime_factory=factory)` | KILLED by check_wiring_contract (--scene-onnx path reaches the factory) |
| x_build_lane_selector__mutmut_57 | build_lane_selector | `selector = LearnedLaneSelector(path, )` | KILLED by check_wiring_contract (the given factory is the one called) |
| x_build_lane_selector__mutmut_59 | build_lane_selector | `print_fn(f"[GVD] lane head: {selector.describe()}", flush=None)` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_61 | build_lane_selector | `print_fn(f"[GVD] lane head: {selector.describe()}", )` | KILLED by check_wiring_contract (start lines flushed) |
| x_build_lane_selector__mutmut_62 | build_lane_selector | `print_fn(f"[GVD] lane head: {selector.describe()}", flush=False)` | KILLED by check_wiring_contract (start lines flushed) |
| x_forward_image__mutmut_8 | forward_image | `if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2 and arr.shape[2] < 3:` | KILLED by check_forward_image_contract (2-channel -> None) |
| x_forward_image__mutmut_14 | forward_image | `if arr.ndim != 3 or arr.shape[0] <= 2 or arr.shape[1] < 2 or arr.shape[2] < 3:` | KILLED by check_forward_image_contract (2 rows is read) |
| x_forward_image__mutmut_15 | forward_image | `if arr.ndim != 3 or arr.shape[0] < 3 or arr.shape[1] < 2 or arr.shape[2] < 3:` | KILLED by check_forward_image_contract (2 rows is read) |
| x_forward_image__mutmut_16 | forward_image | `if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[2] < 2 or arr.shape[2] < 3:` | KILLED by check_forward_image_contract (1 px wide -> None) |
| x_forward_image__mutmut_17 | forward_image | `if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] <= 2 or arr.shape[2] < 3:` | KILLED by check_forward_image_contract (2 px wide is read) |
| x_forward_image__mutmut_18 | forward_image | `if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 3 or arr.shape[2] < 3:` | KILLED by check_forward_image_contract (2 px wide is read) |
| x_forward_image__mutmut_22 | forward_image | `if stitch_bgr is not None or _full_stitch(arr):` | KILLED by check_forward_image_contract (stitch crop only for stitch_bgr) |
| x_forward_image__mutmut_32 | forward_image | `arr = np.clip(np.nan_to_num(arr.astype(np.float64)), None, 255).astype(np.uint8)` | KILLED by check_forward_image_contract (negative float -> 0) |
| x_forward_image__mutmut_38 | forward_image | `arr = np.clip(np.nan_to_num(arr.astype(None)), 0, 255).astype(np.uint8)` | EQUIVALENT: ndarray.astype(None) is float64 |
| x_forward_image__mutmut_39 | forward_image | `arr = np.clip(np.nan_to_num(arr.astype(np.float64)), 1, 255).astype(np.uint8)` | KILLED by check_forward_image_contract (float in-range frame == uint8 frame; 0 stays 0) |
| x_lane_strip_for__mutmut_6 | lane_strip_for | `selector = getattr(perc, "lane_selector", )` | KILLED by check_wiring_contract (perc without lane_selector -> None) |
| x_lane_strip_for__mutmut_14 | lane_strip_for | `if selector is None or not getattr(selector, "ready", None):` | EQUIVALENT: missing ready: not None == not False |
| x_lane_strip_for__mutmut_17 | lane_strip_for | `if selector is None or not getattr(selector, "ready", ):` | KILLED by check_wiring_contract (selector without ready -> None) |
| x_lane_strip_for__mutmut_20 | lane_strip_for | `if selector is None or not getattr(selector, "ready", True):` | KILLED by check_wiring_contract (selector without ready -> None) |
| x_lane_strip_for__mutmut_26 | lane_strip_for | `frames = getattr(bundle, "frames", )` | KILLED by check_wiring_contract (bundle without frames -> None) |
| x_lane_strip_for__mutmut_29 | lane_strip_for | `if not isinstance(frames, dict) and not frames:` | KILLED by check_wiring_contract (frames {} / list -> None) |
| x_lane_strip_for__mutmut_32 | lane_strip_for | `return compose_strip(None)` | KILLED by check_wiring_contract (strip == compose_strip(frames)) |
| x_learned_lanes_enabled__mutmut_8 | learned_lanes_enabled | `if raw is None and str(raw).strip() == "":` | EQUIVALENT: raw None: 'none' is not in the truthy list -> False == LEARNED_LANES_DEFAULT (pinned False) |
| x_learned_lanes_enabled__mutmut_10 | learned_lanes_enabled | `if raw is None or str(None).strip() == "":` | EQUIVALENT: empty raw falls to the truthy-list test -> False == LEARNED_LANES_DEFAULT (pinned False) |
| x_learned_lanes_enabled__mutmut_12 | learned_lanes_enabled | `if raw is None or str(raw).strip() == "XXXX":` | EQUIVALENT: same as 10 |
| x_log_lane_head_change__mutmut_3 | log_lane_head_change | `key = (head.get(None), head.get("reason"), bool(head.get("paint_gated", False)))` | KILLED by check_lane_head_log_contract (a source-only change logs a line) |
| x_log_lane_head_change__mutmut_4 | log_lane_head_change | `key = (head.get("XXsourceXX"), head.get("reason"), bool(head.get("paint_gated", False)))` | KILLED by check_lane_head_log_contract (a source-only change logs a line) |
| x_log_lane_head_change__mutmut_5 | log_lane_head_change | `key = (head.get("SOURCE"), head.get("reason"), bool(head.get("paint_gated", False)))` | KILLED by check_lane_head_log_contract (a source-only change logs a line) |
| x_log_lane_head_change__mutmut_9 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(None))` | KILLED by check_lane_head_log_contract (a paint_gated-only change logs a line) |
| x_log_lane_head_change__mutmut_10 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get(None, False)))` | KILLED by check_lane_head_log_contract (a paint_gated-only change logs a line) |
| x_log_lane_head_change__mutmut_11 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get("paint_gated", None)))` | EQUIVALENT: missing paint_gated: bool(None) == bool(False) |
| x_log_lane_head_change__mutmut_12 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get(False)))` | KILLED by check_lane_head_log_contract (a paint_gated-only change logs a line) |
| x_log_lane_head_change__mutmut_13 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get("paint_gated", )))` | EQUIVALENT: head.get('paint_gated') is None when missing: bool(None) == bool(False) |
| x_log_lane_head_change__mutmut_14 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get("XXpaint_gatedXX", False)))` | KILLED by check_lane_head_log_contract (a paint_gated-only change logs a line) |
| x_log_lane_head_change__mutmut_15 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get("PAINT_GATED", False)))` | KILLED by check_lane_head_log_contract (a paint_gated-only change logs a line) |
| x_log_lane_head_change__mutmut_16 | log_lane_head_change | `key = (head.get("source"), head.get("reason"), bool(head.get("paint_gated", True)))` | KILLED by check_lane_head_log_contract (flag-off head logs paint_gated=0) |
| x_log_lane_head_change__mutmut_19 | log_lane_head_change | `flush=None,` | KILLED by check_lane_head_log_contract (lines flushed) |
| x_log_lane_head_change__mutmut_21 | log_lane_head_change | `)` | KILLED by check_lane_head_log_contract (lines flushed) |
| x_log_lane_head_change__mutmut_22 | log_lane_head_change | `f"[GVD] lane head: source={key[1]} reason={key[1]} paint_gated={int(key[2])} "` | KILLED by check_lane_head_log_contract (line names source=) |
| x_log_lane_head_change__mutmut_26 | log_lane_head_change | `f"learned_conf={head.get(None, 'na')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_27 | log_lane_head_change | `f"learned_conf={head.get('learned_conf', None)}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_28 | log_lane_head_change | `f"learned_conf={head.get('na')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_29 | log_lane_head_change | `f"learned_conf={head.get('learned_conf', )}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_30 | log_lane_head_change | `f"learned_conf={head.get('XXlearned_confXX', 'na')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_31 | log_lane_head_change | `f"learned_conf={head.get('LEARNED_CONF', 'na')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_32 | log_lane_head_change | `f"learned_conf={head.get('learned_conf', 'XXnaXX')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_33 | log_lane_head_change | `f"learned_conf={head.get('learned_conf', 'NA')}",` | KILLED by check_lane_head_log_contract (learned_conf value / 'na') |
| x_log_lane_head_change__mutmut_34 | log_lane_head_change | `flush=False,` | KILLED by check_lane_head_log_contract (lines flushed) |
| x_parse_learned_lanes__mutmut_3 | parse_learned_lanes | `lanes = np.asarray(outputs.get("lanes"), dtype=None)` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_5 | parse_learned_lanes | `lanes = np.asarray(outputs.get("lanes"), )` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_11 | parse_learned_lanes | `lane_valid = np.asarray(outputs.get("lane_valid"), dtype=None)` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_13 | parse_learned_lanes | `lane_valid = np.asarray(outputs.get("lane_valid"), )` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_19 | parse_learned_lanes | `curbs = np.asarray(outputs.get("curbs"), dtype=None)` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_21 | parse_learned_lanes | `curbs = np.asarray(outputs.get("curbs"), )` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_27 | parse_learned_lanes | `curb_valid = np.asarray(outputs.get("curb_valid"), dtype=None)` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_29 | parse_learned_lanes | `curb_valid = np.asarray(outputs.get("curb_valid"), )` | KILLED by check_learned_parse_contract (non-numeric output -> bad_shape, no exception) |
| x_parse_learned_lanes__mutmut_61 | parse_learned_lanes | `lane_valid = lane_valid.reshape(None)` | KILLED by check_learned_parse_contract (batched (1,6) lane_valid parses like flat) |
| x_parse_learned_lanes__mutmut_71 | parse_learned_lanes | `curb_valid = curb_valid.reshape(None)` | KILLED by check_learned_parse_contract (batched (1,2) curb_valid) |
| x_parse_learned_lanes__mutmut_154 | parse_learned_lanes | `conf = min(conf, float(min(2.0, max(0.0, hc))))` | EQUIVALENT: conf <= 1 already (lane_valid range-checked <= 1), so min(conf, min(2, hc)) == min(conf, min(1, hc)) |
| x_parse_learned_lanes__mutmut_187 | parse_learned_lanes | `return LearnedParse(res, curbs_bev=curbs_bev, conf=conf, )` | EQUIVALENT: LearnedParse.reason defaults to '' |
| xǁLaneSelectionǁtelemetry__mutmut_21 | ǁLaneSelectionǁtelemetry | `"learned_conf": round(float(self.learned_conf), 4),` | KILLED by check_wiring_contract (learned_conf 3 decimals) |
| xǁLaneSelectionǁtelemetry__mutmut_27 | ǁLaneSelectionǁtelemetry | `"paint_gated": bool(None),` | KILLED by check_wiring_contract (telemetry paint_gated True when gated) |
| xǁLearnedLaneSelectorǁ__init____mutmut_19 | ǁLearnedLaneSelectorǁ__init__ | `if rt is None or not bool(getattr(rt, "ready", None)):` | EQUIVALENT: missing ready: bool(None) == bool(False) |
| xǁLearnedLaneSelectorǁ__init____mutmut_31 | ǁLearnedLaneSelectorǁ__init__ | `why = str(getattr(rt, "reason", None) or "not ready")` | EQUIVALENT: missing reason: None or 'not ready' == '' or 'not ready' |
| xǁLearnedLaneSelectorǁdescribe__mutmut_4 | ǁLearnedLaneSelectorǁdescribe | `return f"learned lanes ON, model ready ({getattr(self.runtime, 'path', None)})"` | KILLED by check_wiring_contract (runtime without path -> 'model ready (?)') |
| xǁLearnedLaneSelectorǁdescribe__mutmut_7 | ǁLearnedLaneSelectorǁdescribe | `return f"learned lanes ON, model ready ({getattr(self.runtime, 'path', )})"` | KILLED by check_wiring_contract (runtime without path -> 'model ready (?)') |
| xǁLearnedLaneSelectorǁdescribe__mutmut_10 | ǁLearnedLaneSelectorǁdescribe | `return f"learned lanes ON, model ready ({getattr(self.runtime, 'path', 'XX?XX')})"` | KILLED by check_wiring_contract (runtime without path -> 'model ready (?)') |
