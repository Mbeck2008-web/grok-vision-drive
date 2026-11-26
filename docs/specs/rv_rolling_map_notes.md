# Reality View step 6: rolling map (`python/viz/rv_rolling_map.py`) notes

Spec: `docs/specs/reality_view_3dgs.md` step 6 and `check_no_ghost_rolling_map`.
Tests: `scripts/test_rv_rolling_map.py` (auto-run by `scripts/run_offline_suite.py`).
Flag: this module is not wired to anything. `viz.reality_view_enabled` stays default OFF (`rv_contract.reality_view_enabled`).
CPU-only numpy: no GL, no CUDA, no device selection or config (asserted by `check_spec_constants_and_cpu_only`).

## Design (one paragraph)

One record per occupied 0.25 m voxel, sorted by packed cell key. A frame (or one camera of a frame) of world-frame points is binned;
each hit cell moves centre and colour toward the frame mean by `alpha` (0.3) and its opacity by `w += beta*o*(1-w)` (beta 0.35,
bounded by 1, never falls on re-observation). A new cell starts at `0.05*o`, so it is only visible (opacity >= 0.1) after a second hit.
Points inside a dynamic box (+0.5 m margin) never enter; cells already inside one are purged (also after the EMA step). Cells older than
`max_age_s` (30 s) or outside the +/-64 m window around the ego (per axis, inclusive) are evicted; over `n_max` (300k) the oldest go first,
ties farthest first, ties lowest key. Time is an input (`t_s`); no wall clock, no RNG.

## Where the spec was loose and which reading the tests take

| Spec text | Reading | Why |
| --- | --- | --- |
| (a) "two Gaussians in one cell whose centres differ by > 0.5 m along the same camera ray" | A Gaussian more than one cell edge (0.25 m) off every GT surface marks its 0.25 m cell as a duplicate-surface cell; rate = such cells / occupied cells (opacity >= 0.1). Gated <= 5%. | Two centres in one 0.25 m cell cannot be 0.5 m apart. The literal ray-bundle reading (a 0.25 m tube around the camera ray, partner > 0.5 m along it) was implemented (`_duplicate_surface_rate`, checked against brute force) and flags **62%** of cells on a perfectly clean, noise-free static scene, because a ground plane seen at 1.8 deg incidence lies inside any 0.25 m tube for metres along the ray. It cannot gate anything. A 0.25 m normal offset is <= 0.5 m along any ray at <= 60 deg incidence, so the gated reading is at least as strict for every non-grazing view. |
| (b) "pose noise sigma = 0.15 m, 1 deg" | Per-camera, per-frame iid noise: translation sigma 0.15 m per axis, rotation sigma 1 deg per axis. | Larger than a 1 deg total rotation, so stricter. |
| (b) "80 ms inter-cam skew" | The `main` camera captures 80 ms after the pose stamp (stale pose, 10 m/s, 0.8 m). | Covers the Tracer max 79.8 ms. |
| (b) "median |z_render - z_gt|" | First-hit Z-buffer of every splat with opacity >= 0.1 at the true final pose; valid GT pixels with no splat count as infinite error. | Stricter than alpha-composited expected depth and than skipping holes. |
| (c) "old cell", "gone >= 1.0 s" | Splat with opacity > 0.1 inside a vehicle's GT volume (+ half cell) from >= 1.0 s earlier, above the road (z >= 0.175 m). With pose noise, splats that a vehicle-free control run of the same clip also has are not counted. | Noise smears the road upward into the vehicle volume. |

## Measured result of the proposed thresholds (unmeasured by the spec; numbers are from this branch)

* (a) 0.0000 on the clean static scene (limit 0.05).
* (c) 0 leftover splats: vehicle parked then gone, vehicle crossing then gone, with and without pose noise + skew, and with the GT box unknown for the first 0.3 s. An append-only baseline leaves 72 834.
* (b) **misses.** Seed 11 (the canned clip's default, fixed before any run): median 0.681 m vs 0.50 m. Seeds 1..12: z-buffer 2/12 pass (median 0.53 m, range 0.46..0.75 m); alpha-composited expected depth 10/12 pass (median 0.37 m, range 0.27..0.53 m); an append-only baseline 0/12 pass on both (median 0.84 m). Without noise the same instrument gives 0.19 m, trans-only 0.37 m, rot-only 0.44 m, both 0.57 m, both + skew 0.68 m. The spec's No-go remedies were tried: shrinking `max_age_s` to 1.0 s / 0.5 s and the radius to 32 / 16 m does not reach 0.50 m on the z-buffer instrument (0.49..0.84 m, holes rise to 24..34%); a coarser voxel is worse (0.5 m: 0.81, 1.0 m: 0.94).
  Cause: independent per-frame pose noise turns a ground surface into a vertical slab of cells; the first hit of that slab sits above the true surface and, at grazing incidence, 0.15 m of height is 0.5 m of depth. The voxel EMA only averages inside a cell, so it does not thin the slab.
  The test is therefore red on (b) by design (spec: "fail the test if missed"). It was not loosened.
