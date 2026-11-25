# BEV LSS visibility labels: named limitation (G5)

Scope: `python/train/bev_lss.py` training-label visibility gate (default off; nothing in the driving path imports it).

## Observable vs unobservable barriers (referee ruling, 2026-11-25)

- **Observable**: some camera ray at full input resolution (every content pixel, not the 24 x 32 feature
  grid) has its first hit on the barrier, at any height. Ground hidden behind an observable barrier must never
  be labelled seen or free. The referee (Skeptic, `swarm241_probe/face_hits.py`) decides this per row from
  ray geometry only; 324 of its 336 rows are observable.
- **Unobservable**: no ray of any camera touches the barrier: out of every camera's view (rear / side /
  under-nose blind spots, lateral offsets outside the field of view) or fully behind a taller occluder. The rig
  has no evidence of it. This is the only named limitation.

Pinned: `check_unobservable_leak_pinned` (scripts/test_bev_lss.py) rebuilds two of the referee's
unobservable rows (a 0.6 m wall behind a 1.8 m wall; a rear wall with no rear camera), checks no
full-resolution ray first-hits the target, and pins their leak at 0 cells.

## Known gaps still open (observable)

Not a limitation: these are open leaks against the zero-leak bar. See
`/workspace/gvd2/audit/redesigner_bev_barrier.md` for the per-row table. Low barriers and curbs (0.15 - 0.45 m)
whose face hits sit inside the 0.5 m ground band and do not stand out from the neighbouring rows, and wall
sections that only a coarse camera (wide) sees and whose rows straddle the face (y = 8 m class), still let
mid-air band points behind them count as seen.
