# rv-side-rear PR notes

## Known issues until Trainer 2.0 (do not regress in this PR)

- **Wide rings remain.** With wide snap off (Michael 2026-11-26), the wide cam's net depth sits too high on the road (allowed-dev median |dz| 0.247 m at 5-10 m, 0.547 m at 10-15 m). This is a known issue until Trainer 2.0. Breaker adds the KNOWN_FAILING line; this PR does not edit KNOWN_FAILING.
- **pillarR / repeatR gate (b)** is a known failure (Tracer's rv_snapb trace). Breaker adds that line. `ROAD_SNAP_CAMS` and the 0.30 m rejection are unchanged.

Wide snap stays OFF by default and remains switchable.

## rear_holes evidence (Skeptic ruling b)

`check_synthetic_rear_holes` reads 5.35% at +-135 deg (15-30 m) because the props metric draws every point with a fixed 0.3 m splat and a 4 px cap, not the renderer's own `point_footprint` and splat cap. Measured through the real preview path (`point_footprint` + `GroundGrid` + `splat_max_radius_px`), same synthetic scene, same GT road mask, same 15-30 m band, on a5978d5. Command (repo root, evidence only, not collected by `run_offline_suite`, which runs `scripts/test_*.py`):

```
python scripts/evidence_rear_holes_real_render.py
```

Output:

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

Production rendering is unchanged by this commit. The props metric gets fixed in Breaker's props commit (hole_fraction using the renderer's own footprint and cap).
