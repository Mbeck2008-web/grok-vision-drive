# Reality View C prototype mutmut survivors (`python/viz/reality_view_c.py`)

Source: mutmut 3.8.1, results in `/workspace/swarm/swarm-4/rv-c-proto-evidence/mutmut_final2.log`.

## Totals

- Generated: 937
- Killed: 765
- Survived: 172 (0 timeouts, 0 segfaults, 0 suspicious)
- Of the 172 survivors: all are equivalent (109 error-message/label, 42 dtype/copy, 18 redundant guard/symmetry, 2 struct signedness, and `x__unproject_view__mutmut_33` intermediate-dtype).

## The 7 Skeptic pre-gate items

x__unit_vec__mutmut_16: KILLED by check_mutation_and_edge_cases (direct `_unit_vec((1e-12, 0.0, 0.0), "v")` must raise; mutant uses `norm < 1e-12` and accepts this input).
x__unit_vec__mutmut_20: KILLED by check_mutation_and_edge_cases (direct `_unit_vec((3.0, 0.0, 0.0), "v") == (1.0, 0.0, 0.0)`; mutant multiplies by the norm and returns (9, 0, 0)).
x__unproject_view__mutmut_33: EQUIVALENT — `.astype(np.uint8)` -> `.astype(None)` keeps the clipped RGB as float64 in [0, 255]; the only later use is `cols = rgb[ys, xs].astype(np.uint8, copy=False)`, and NumPy float-to-uint8 truncation of those same clipped values is identical, so no input can distinguish the intermediate dtype.
x__unproject_view__mutmut_90: KILLED by check_mutation_and_edge_cases (direct `_unproject_view` on an all-NaN view asserts `pts.shape == (0, 3)`; mutant returns `(0, 4)`).
x__unproject_view__mutmut_95: KILLED by check_mutation_and_edge_cases (direct `_unproject_view` on an all-NaN view asserts `cols.shape == (0, 3)`; mutant returns `(1, 3)`).
x__unproject_view__mutmut_96: KILLED by check_mutation_and_edge_cases (direct `_unproject_view` on an all-NaN view asserts `cols.shape == (0, 3)`; mutant returns `(0, 4)`).
x_render_chase_depth_billboards__mutmut_162: KILLED by check_mutation_and_edge_cases (16-wide source / 8-wide chase fixture places a valid point at `floor(uc)=8 == out_w`; original culls it with `point_count == 0`, mutant uses `px <= out_w` and indexes out of range or paints the edge).

## Remaining survivors by category

### Error-message / diagnostic-name strings (109)

- `x__check_cam_id__mutmut_4`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_cam_id__mutmut_5`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_cam_id__mutmut_6`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_cam_id__mutmut_22`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_int_dim__mutmut_3`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_int_dim__mutmut_4`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_int_dim__mutmut_10`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_finite_float__mutmut_3`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_finite_float__mutmut_4`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__check_finite_float__mutmut_9`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unit_vec__mutmut_3`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unit_vec__mutmut_9`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unit_vec__mutmut_18`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_6`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_14`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_15`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_18`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_26`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_27`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_30`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_34`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_35`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_41`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_43`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_47`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_48`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_50`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_54`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_55`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_60`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_63`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_64`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_65`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_68`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_72`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_73`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_76`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_80`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_81`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_84`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_88`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_89`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_125`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_128`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_129`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_132`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_135`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_136`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_152`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_155`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_camera_from_angles__mutmut_156`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_2`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_10`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_11`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_13`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_21`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_22`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_25`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_29`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_30`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_32`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_36`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_37`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_38`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_42`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_43`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_44`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_48`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_49`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_55`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_57`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_61`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_62`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_64`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_68`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_69`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_74`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_75`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_79`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_80`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_82`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_85`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_86`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_88`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_91`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_92`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_94`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_97`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__validate_pose__mutmut_98`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_12`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_15`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_22`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_25`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_26`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_29`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_30`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_48`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_73`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_103`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x__unproject_view__mutmut_111`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_4`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_12`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_13`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_16`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_24`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_25`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_render_chase_depth_billboards__mutmut_271`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_encode_png_bytes__mutmut_11`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.
- `x_encode_png_bytes__mutmut_22`: EQUIVALENT: only the ValueError message text or a diagnostic name string passed into a checker changes; exception type and control flow are unchanged.

### dtype / copy keyword elision (42)

- `xǁCameraPoseǁrotation_matrix__mutmut_2`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `xǁCameraPoseǁrotation_matrix__mutmut_4`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unit_vec__mutmut_6`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unit_vec__mutmut_8`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_110`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_112`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_115`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_117`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_143`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_145`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_147`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_camera_from_angles__mutmut_149`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_47`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_55`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_56`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_58`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_59`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_77`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_79`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_80`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_86`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_88`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_125`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_127`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_129`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_130`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_132`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x__unproject_view__mutmut_133`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_39`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_41`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_85`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_91`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_113`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_115`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_187`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_189`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_193`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_196`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_206`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_246`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_render_chase_depth_billboards__mutmut_248`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.
- `x_encode_png_bytes__mutmut_46`: EQUIVALENT: dtype=None / omitted dtype / copy=None / .astype(None) on an array already of the intended dtype (or a later explicit cast), so values and observable dtypes match.

### Redundant guard or symmetric offset (18)

- `x_chase_pose_default__mutmut_19`: EQUIVALENT: omitting near_m=/far_m= uses camera_from_angles defaults that are exactly DEFAULT_CHASE_NEAR_M / DEFAULT_CHASE_FAR_M.
- `x_chase_pose_default__mutmut_20`: EQUIVALENT: omitting near_m=/far_m= uses camera_from_angles defaults that are exactly DEFAULT_CHASE_NEAR_M / DEFAULT_CHASE_FAR_M.
- `x__unproject_view__mutmut_42`: EQUIVALENT: nan_to_num fill argument is followed by np.clip(..., 0, 255), so the mutated fill is clipped to the same uint8 output.
- `x__unproject_view__mutmut_43`: EQUIVALENT: nan_to_num fill argument is followed by np.clip(..., 0, 255), so the mutated fill is clipped to the same uint8 output.
- `x__unproject_view__mutmut_45`: EQUIVALENT: nan_to_num fill argument is followed by np.clip(..., 0, 255), so the mutated fill is clipped to the same uint8 output.
- `x__unproject_view__mutmut_46`: EQUIVALENT: nan_to_num fill argument is followed by np.clip(..., 0, 255), so the mutated fill is clipped to the same uint8 output.
- `x__unproject_view__mutmut_50`: EQUIVALENT: nan_to_num fill argument is followed by np.clip(..., 0, 255), so the mutated fill is clipped to the same uint8 output.
- `x__unproject_view__mutmut_65`: EQUIVALENT: d64 >= 0.0 is implied by d64 >= near_m and near_m > 0, which is already in the same conjunction.
- `x__unproject_view__mutmut_120`: EQUIVALENT: np.stack(..., axis=-1) and axis=+1 are the same on a 1-D stack of equal-length vectors.
- `x_render_chase_depth_billboards__mutmut_66`: EQUIVALENT: empty unproject results are shape (0, 3) so shape[1] > 0 still appends an empty (0, 3) array; np.concatenate of those empties with/without other views does not change points or the rendered buffers.
- `x_render_chase_depth_billboards__mutmut_67`: EQUIVALENT: `if pts.shape[0] >= 0` is always true; appending a (0, 3) empty array and concatenating it with other views (or concatenating only empties, then taking the empty in_z early return) leaves the render unchanged.
- `x_render_chase_depth_billboards__mutmut_94`: EQUIVALENT: subsample uses linspace(0, n-1, cap), which is a no-op when n == cap, so `>` vs `>=` does not change the kept points.
- `x_render_chase_depth_billboards__mutmut_212`: EQUIVALENT: ndarray.reshape(-2) flattens the same way as reshape(-1) for these 2-D offset grids (NumPy treats a single negative dim as 'infer').
- `x_render_chase_depth_billboards__mutmut_216`: EQUIVALENT: ndarray.reshape(-2) flattens the same way as reshape(-1) for these 2-D offset grids (NumPy treats a single negative dim as 'infer').
- `x_render_chase_depth_billboards__mutmut_219`: EQUIVALENT: billboard offset grid is symmetric around 0, so adding vs subtracting the offset set yields the same splat coverage.
- `x_render_chase_depth_billboards__mutmut_221`: EQUIVALENT: ndarray.reshape(-2) flattens the same way as reshape(-1) for these 2-D offset grids (NumPy treats a single negative dim as 'infer').
- `x_render_chase_depth_billboards__mutmut_224`: EQUIVALENT: billboard offset grid is symmetric around 0, so adding vs subtracting the offset set yields the same splat coverage.
- `x_render_chase_depth_billboards__mutmut_226`: EQUIVALENT: ndarray.reshape(-2) flattens the same way as reshape(-1) for these 2-D offset grids (NumPy treats a single negative dim as 'infer').

### struct.pack signed vs unsigned (2)

- `x__png_chunk__mutmut_18`: EQUIVALENT: struct.pack signed vs unsigned format character on PNG length/IHDR fields; packed integers are in [1, 1920] so the bit pattern is identical.
- `x_encode_png_bytes__mutmut_66`: EQUIVALENT: struct.pack signed vs unsigned format character on PNG length/IHDR fields; packed integers are in [1, 1920] so the bit pattern is identical.

### Other (1) — also equivalent, named in the 7 above

- `x__unproject_view__mutmut_33`: EQUIVALENT: `.astype(np.uint8)` -> `.astype(None)` on the clipped float RGB keeps float64 in [0, 255]; the only later use is `cols = rgb[ys, xs].astype(np.uint8, copy=False)`, and NumPy float-to-uint8 truncation of those same clipped values is identical, so no input can distinguish the intermediate dtype.

