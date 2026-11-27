# rv-side-rear PR notes (gvd/rv-side-rear)

Scope: road snap, depth-edge cut and splat changes for the Reality View stable-fill preview, plus Breaker's props file.
Sky/backdrop work is NOT in this branch (parked on `gvd/rv-sky-scratch` at 0ed52be; reverted here by 28dbf80 and e857cc3).

## What changed in python/ and tools/ (truthful list; `git diff 783fedd..HEAD -- python/ tools/` = 2 files, +138/-20)

Against `main` (a56f514) `python/viz/rv_stable_fill.py` and `tools/rv_stable_preview.py` do not exist at all (the stable-fill feature, default off: `STABLE_FILL_DEFAULT_ON = False`, is already on `gvd/rv-stable-fill` at 783fedd). Production rendering and `ROAD_SNAP_CAMS` are CHANGED relative to 783fedd:

- `ROAD_SNAP_CAMS`: was road snap for every cam (783fedd) -> now `("main", "pillarL", "pillarR", "repeatL", "repeatR", "rear")`. Wide snap is OFF by default (Michael 2026-11-26), narrow is off; both stay switchable. `ROAD_SNAP_RANGE_M = {"wide": 15.0}` caps wide at 15 m if switched on.
- New `ROAD_SNAP_MAX_ABOVE_M = 0.30`: `road_snap` leaves a road-labelled px at its raw depth when the net puts it more than 0.30 m above the plane (one-sided rejection).
- New `snap_road_for_cam(cam_id, ...)`; `road_snap(..., exclude=)` so ego-hood px are not snapped.
- `ground_clamp` now also clamps +inf on a downward ray.
- `CamDepthFilter`: a blended depth never pushes an on/above-road input below the road plane.
- New `edge_cut_far_side` replaces the preview's inline `cv2.dilate` cut: no snapped exemption, and a px keeps its own depth where the far side would be past its ground hit (never under the road). The cut now runs before `ground_clamp` (clamp after the cut).
- `point_footprint`: on-plane points' splat footprint grows to the ground row spacing (rear/side holes).
- `splat_max_radius_px(width) = max(2, W//40)` (was `max(2, W//120)`).
- `tools/rv_stable_preview.py` calls the above (`snap_road_for_cam` with an ego mask, `edge_cut_far_side`, `point_footprint`, `splat_max_radius_px`).

Part B commits (side/rear snap work on top of ca70e02): 434a654 (ground splats reach the next camera row, rear snapped, snapped px go through the edge cut), 64d4c03 (ego-hood px not snapped, wide capped at 15 m, shared edge cut with no snapped exemption, splat cap W//40), 68281e7 (wide road snap OFF by default; main/pillars/repeaters/rear snap with the 0.30 m rejection; all cams switchable).

## The 0.30 m rejection lines vs origin/gvd/swarm3-ca70e02 (ca70e02)

Three lines carry the rejection. Two are byte-identical: `ROAD_SNAP_MAX_ABOVE_M: float = 0.30` and the `above = np.isfinite(d) & (h + d * dz > float(max_above_m))` line, plus the `snap = ... & ~above` mask line. The `road_snap` signature line (`max_above_m: float = ROAD_SNAP_MAX_ABOVE_M`) gained a trailing comma because of the new `exclude` parameter. The 0.30 value is unchanged.

## M04b and M07

- M04b is pinned by `check_edge_cut_drops_px_next_to_sky` (S22), commit 98a7bc6, with no render change (`<=` kept in `edge_cut_far_side`). With GT depth, the non-sky px next to +inf sky is cut to +inf and is not drawn; the test asserts the 12 border px and fails on a `<` mutant.
- M07: `ground_clamp` `d > zmax` mutated to `>=` (`python/viz/rv_stable_fill.py:357`). Depth output is identical; only the clamp mask differs on exact ties (px exactly at the ground hit), and that mask feeds `clamp_frac` only. Equivalent mutant on depth, not pinned.

## Known issues until Trainer 2.0 (do not regress in this PR)

- Wide rings remain. With wide snap off (Michael 2026-11-26) the wide cam's net depth sits too high on the road; this branch does not fix the ripple (see `check_clip015_ripple_wide_dz`, `check_wide_snap_off_no_plane_pull_0_5m`, KNOWN_FAILING).
- pillarR / repeatR gate (b) is a known failure: pillar 0-15 m band, clip015 f40/f160, `check_clip015_snap_misplace`: pillarR 9.48% (3055/32213) and repeatR 2.56% (985/38413) drawn snapped px with GT z > 0.3 m (max 1%). `ROAD_SNAP_CAMS` and the 0.30 m rejection are as described above.
- Fixed 0.3 m splat sites in `scripts/test_rv_side_rear_props.py` (Breaker's file; not the renderer's own `point_footprint` and cap), at f59b7b2: line 579 (`hole_fraction` paint), 2070, 2165, 2366 (all `np.full(..., 0.3)` footprints); line 771 is the check that counts them and 2984 is a fallback when no footprint exists. (The brief listed :579, :1773, :1974, :2592. :1773/:1974/:2592 are the same footprint lines as 2165/2366/2984 on this head, shifted down by the props insert above them; the site lines are byte-identical to the aeed334 parent.)

## rear_holes evidence (Skeptic ruling b)

`check_synthetic_rear_holes` read 5.35% at +-135 deg (15-30 m) under the props fixed 0.3 m splat and 4 px cap. Through the real preview path (`point_footprint` + `GroundGrid` + `splat_max_radius_px`), same synthetic scene, GT road mask and 15-30 m band, there are 0 holes. Command (repo root; evidence only, not collected by `run_offline_suite`, which runs `scripts/test_*.py`):

```
/workspace/swarm/swarm-3/venv-bev/bin/python scripts/evidence_rear_holes_real_render.py 2>&1 | tail -18
```

Output (log: scratch/ring/evidence_f59b7b2.log):

```
chase 480x270; real splat cap 12px (props default cap 4px); rendered pts 234116; grid cells 18757; GT road pts 164441
    180 deg  15-30 m  holes   36 /  1112 =   3.24%   A props splat (0.3 m fixed, cap 4, fill)
    180 deg  15-30 m  holes    0 /  1112 =   0.00%   B real footprint, props cap 4, no grid, fill
    180 deg  15-30 m  holes    0 /  1112 =   0.00%   C real footprint, real cap, no grid, fill
    180 deg  15-30 m  holes    0 /  1112 =   0.00%   D real footprint + grid, real cap, fill (LIVE PATH)
    180 deg  15-30 m  holes    0 /  1112 =   0.00%   E same as D, observed only (no gap-fill)
    135 deg  15-30 m  holes  124 /  2319 =   5.35%   A props splat (0.3 m fixed, cap 4, fill)
    135 deg  15-30 m  holes   26 /  2319 =   1.12%   B real footprint, props cap 4, no grid, fill
    135 deg  15-30 m  holes    0 /  2319 =   0.00%   C real footprint, real cap, no grid, fill
    135 deg  15-30 m  holes    0 /  2319 =   0.00%   D real footprint + grid, real cap, fill (LIVE PATH)
    135 deg  15-30 m  holes    0 /  2319 =   0.00%   E same as D, observed only (no gap-fill)
   -135 deg  15-30 m  holes  124 /  2319 =   5.35%   A props splat (0.3 m fixed, cap 4, fill)
   -135 deg  15-30 m  holes   26 /  2319 =   1.12%   B real footprint, props cap 4, no grid, fill
   -135 deg  15-30 m  holes    0 /  2319 =   0.00%   C real footprint, real cap, no grid, fill
   -135 deg  15-30 m  holes    0 /  2319 =   0.00%   D real footprint + grid, real cap, fill (LIVE PATH)
   -135 deg  15-30 m  holes    0 /  2319 =   0.00%   E same as D, observed only (no gap-fill)
grid cells by range: <15 m 7606, 15-30 m 6079, >=30 m 5072; rendered pts 15-30 m (hypot) 18830
```

The rear-holes check itself was fixed in Breaker's props commits (ddb6ee4, 536724a); the evidence commit cb201c5 changed no production code.
