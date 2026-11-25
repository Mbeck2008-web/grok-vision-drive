# BEV LSS training-label visibility gate

Status: **flag default OFF**. This document covers the training-label
visibility / occlusion gate in `python/train/bev_lss.py` only. The
camera-only runtime path is untouched.

## Observable vs unobservable leaks

A hidden ground cell (every in-view camera's continuous sightline to the
ground centre hits the inset wall) that the gate still marks free is a
**leak**. Leaks split by ground-truth ray geometry, not by what the gate
saw:

- **Observable.** At least one 24×32 feature-centre or upper-row-edge ray
  from any camera hits the wall face at any height, including a probe
  point at height 0.45 m along the ray. Marking a hidden cell free is a
  leak. The bar is **zero observable leaks**.
- **Unobservable.** No such ray hits the wall face at any height. The
  cameras never measured that wall, so the gate cannot close the shadow
  without inventing occupancy. These rows, if any, must be named here
  and pinned so the count cannot grow.

`scripts/test_bev_lss.py::check_unobservable_leak_count_pinned` asserts
`UNOBS_LEAK_COUNT_PIN` (currently 0) on a fixed case set that includes
axis-aligned walls, a diagonal stair, an offset wall, and a one-cell
pole. The same geometry check classifies each leak as observable or
unobservable.

## Limitation: Unobservable 24×32 umbra

Named limitation. A wall (or the part of a wall) that no ground-truth
24×32 feature-centre or upper-row-edge ray hits — at any height, including
an upper-ray point at 0.45 m — leaves a geometric umbra the gate cannot
honestly close. Those free-in-hidden cells are **unobservable**, not a
failure of the zero-observable bar. Raising `UNOBS_LEAK_COUNT_PIN` requires
naming the new rows in this section.

Current pinned unobservable rows: **none** (count 0).

## Declared residual observable leaks

The zero-observable bar is **not met** on this head. These rows stay in
`check_unobservable_leak_count_pinned` (which must fail until they are 0)
and in the off-branch leak repro. They are not unobservable: GT 24×32
rays hit the wall face (including a 0.45 m upper-row probe).

- `diag_0.6_8cam`: 2 free-in-hidden cells at (-7.25, 8.75) and
  (-6.75, 8.25). The 24×32 splat lands ~1 m right of the stair's left
  end; 12-step nearest sampling plus `SIGHTLINE_MARGIN_M = 1.2` m miss
  the corner. Face hits on the wall as a whole are 485.

Do not drop these cases from the pin test to go green.

## Over-hide and far coverage

- Visible-road recall is (geometrically visible ground cells marked
  observed) / (total truly visible). Report it next to leak totals.
- A one-cell pole / sign (0.2 m × 3 m, boxed to one 0.5 m cell) must not
  hide visible road beside it beyond geometric truth. The official test
  pins same-range side cells (y 9.75–10.75 m, |x| 1.5–3.0 m) that are
  seen on empty road.
- `check_visibility_by_construction` requires `seen_far` (mean of
  `clear_mass > 0` on empty 8-camera road) **> 0.2**. Do not loosen this
  floor.

## Sightline construction (training labels only)

- Per-cell occluder height is a graze-bounded column top (LSE / max), not
  a mean of front-face hits. Thin (1–2 cell) fronts use `BEV_Z_MAX_M` and
  are not dilated on that row (adjacent empty rows may still receive the 3-cell window).
- Wide fronts (≥ `OCC_MIN_FRONT_CELLS`) extend laterally up to
  `OCC_FRONT_EXTEND_CELLS` through cells without a near-ground surface,
  after a one-cell far-row seed.
- `sightline_blocked` uses 12 steps and `SIGHTLINE_MARGIN_M = 1.2` m.
  Height-aware `ray_crossed_raised` lets a ray go over a low close wall.
  A 24x32 splat plus 12-step sampling can still miss a 0.5 m diagonal
  corner; those residual cells are called out as observable leaks if a
  GT ray hits the wall face.
