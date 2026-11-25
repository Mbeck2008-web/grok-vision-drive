# PR #237 mutant diffs

Each `*.diff` is against `python/perception/occ_sparse.py` at the parent of
this commit (apply with `patch -p1`). The kill site is the first
`scripts/test_occ_net.py` assertion that fails after the patch.

| Mutant | First kill |
| --- | --- |
| M1_no_minres_clamp | `scripts/test_occ_net.py:1256` (`check_capped_hot_low_p_grade_and_saturated_fill`, Major 3 +10% saturated walls `922 < 1000`) |
| M2_no_098_scale | `scripts/test_occ_net.py:1256` (Major 3 +10% saturated walls `994 < 1000`) |
| M3_no_056_floor | `scripts/test_occ_net.py:1256` (Major 3 +10% saturated walls `940 < 1000`) |
| M3_no_onecell | `scripts/test_occ_net.py:1348` (`check_capped_hot_onecell_and_gceil`: `24/48` car columns lose the buried road) |
| M4_depth_unclamped | `scripts/test_occ_net.py:994` (`check_capped_hot_exposed_ground` per-column +10% σ=0.02) |
| M5_snap_low_no_vi | `scripts/test_occ_net.py:594` (`run_parts`, `f=2 uphill +15%` `below 165 > 140`) |
| M6_no_rel_vote | `scripts/test_occ_net.py:1223` (Major 2 road p.63 + cars p.66, `4096` at car layer) |
| M7_no_snap_own | `scripts/test_occ_net.py:594` (`f=2 uphill +15%` `below 146 > 140`) |
| M8_no_mine | `scripts/test_occ_net.py:1204` (Major 2 corner +1 step p.60, `0/3655` at own seed) |
| M8_restore_gceil | `scripts/test_occ_net.py:1393` (`check_capped_hot_onecell_and_gceil`: `14/728` building columns one layer low) |
| M9_no_line_fit | `scripts/test_occ_net.py:1214` (Major 2 +10% grade half p.62, `41` wrong) |
| X4_ref_seeds_any_p | `scripts/test_occ_net.py:1168` (`check_capped_hot_inherit_and_vote_edges`, X4 non-voter seed) |
| X8_nearest_x_only | `scripts/test_occ_net.py:1117` (isolated p=0.63 hole inherit; also killed later by the X8/X10 +y pin) |
| X9_inherit_round_half | `scripts/test_occ_net.py:1150` (X9 leftover inherit `+0.5` jump) |
| X10_inherit_off_raw | `scripts/test_occ_net.py:1117` (isolated p=0.63 hole inherit; also killed later by the X8/X10 +y pin) |
| X13_vote_pmin_0.62 | `scripts/test_occ_net.py:1117` (isolated p=0.63 hole voted) |
| HAS_ENV_MARGIN_0.049 | `scripts/test_occ_net.py:973` (`check_capped_hot_exposed_ground`, f=2 crest σ=0.01 walls) |

## M3_no_onecell and M8_restore_gceil

On every pre-existing `check_capped_hot_*` fixture in `test_occ_net.py`
(169 unique `_column_ground_lvl` / `_capped_hot` calls, including the
f=4 fill 0.92 site) M3_no_onecell was byte-identical to good code.
M8_restore_gceil already differed on four of those fixtures (f=2 +10%
noise-free, f=2 σ=0.02 +10%/+15%, crest σ=0.01) but still passed the
old pins (e.g. per-column `low <= 24` allowed the 14-column drop).

New fail-first pins in `check_capped_hot_onecell_and_gceil`:

- M3: 8 duplicate p=0.90 cells at one voxel must not pool a 0.05 gap
  into a 7×7 car lot's buried road. Good: `48/48` car columns stay on
  level 4. M3: `24/48` jump to the car layer. Assert `:1348`.
- M8: f=2 +10% noise-free building columns must not sit below true
  ground. Good: `0/728` low. M8: `14/728` low (all at y=26). Assert `:1393`.
