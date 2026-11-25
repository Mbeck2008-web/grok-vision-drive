# BEV LSS visibility labels: named limitation (G5)

Scope: `python/train/bev_lss.py` training-label visibility gate (default off; nothing in the driving path imports it).

## Observable vs unobservable barriers

- **Observable**: some valid feature ray (24 x 32 grid, stride 8, letterbox-valid) of some camera has its
  first hit on the barrier, at any height. Ground hidden behind an observable barrier must never be labelled
  seen or free. Measured by `probes_barrier/probe_barrier3.py` (Swarm 3), decided from ray geometry only.
- **Unobservable**: no valid feature ray of any camera touches the barrier (it falls between two rows /
  columns). The rig has no evidence of it; ground behind it can read as seen. This is a named limitation of
  any label built from these rays, not a gate bug.

Pinned: `check_unobservable_leak_pinned` (scripts/test_bev_lss.py): 0.3 m wall 15 m ahead, main only, has
no ray on it and at most 102 leaked cells. The count may only shrink.

## Known gaps still open (observable)

See `/workspace/gvd2/audit/redesigner_bev_barrier.md` for the current probe table. Barriers whose only hits
sit inside one depth bin with the ground hit of the row below (e.g. 0.45 m at 12 m, main only) cannot be
told from a 15 % grade by the column riser test, and walls lower than the 0.3 m occluder cutoff are passed
over (G5 "objects lower than the band can be passed over").
