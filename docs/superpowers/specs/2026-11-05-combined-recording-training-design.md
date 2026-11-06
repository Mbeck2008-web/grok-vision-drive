# Combined recording training + E: cache — Spec / AC

**Status:** design for implementation on branch `gvd/combined-recording-training`  
**Repo:** `Mbeck2008-web/grok-vision-drive`  
**Base:** `main` @ `145e073` (PR #179 merged); do **not** edit `gvd/sensor-aux-supervision`  
**Date:** 2026-11-05 (America/Chicago)  
**Scope gate:** smallest coherent change; local only (no CloudAgent); no collector / live ONNX / PC runtime edits; **no training launch** in this PR.

## Intent

Train on **two completed immutable recordings** without a physical merged folder:

| Role | Path (PC) | Frames | Holdout |
| --- | --- | --- | --- |
| **Prior** (validation source) | `E:\GVD_data\sgt_2h_20261105\scene_lanes\ctl_20261105_p01` | 69521 | **Fixed** time-block holdout (`--holdout-blocks 5`) |
| **New** (train-only) | `E:\GVD_data\sgt_2h_20261105_r2\scene_lanes\ctl_20261105_p01` | 69914 | **Never** contributes holdout rows |

Both folders are named `ctl_20261105_p01`, so bare frame `i` and session ids **collide**. Training must namespace by **source**. Cache goes on **E:** (user-approved despite speed). Combined main-only cache estimate ≈ **19.15 GiB** (prior ≈ 9.55 GiB + new ≈ 9.6 GiB). C: is tight (~39 GiB free); no C: cache required; **no deletions**.

This PR ships **CLI + loader + cache + holdout plumbing** and offline fixtures. The authorized next run (12 epochs, lane_gt, seed 73, no-augment, aux off) is **out of scope to launch** here; the PR must leave a ready-to-use command template that preserves the prior/new split.

## Non-goals

- Editing the old PR #179 branch or re-opening sensor-aux work.
- Physical copy/merge of raw/GT into one folder; mutating recorded labels.
- Broadening architecture, losses, aux/road_edge (default **off**; lane-only for the planned run).
- Collector / live ONNX / queued-chain / PC train launch.
- Claiming measured quality gain from combining recordings (UNPROVEN until the authorized run finishes under Skeptic).
- Requiring C: cache or deleting any existing data.

## Failure that motivates this

The post-recording chain passed **two** `--strips` argv tokens. `ArgumentParser` has `--strips type=str` (one value) → `unrecognized arguments` → cache build exit 2; no train started. There is **no** official multi-root path today: `build_frame_cache` / `train_directory` / `load_scene_rows` are single-root.

## Existing contracts (inspected)

| Contract | Location | Rule kept |
| --- | --- | --- |
| `--strips` | `train_scene.build_parser` | Single string today; single-root behavior must remain bit-compatible |
| Holdout blocks | `continual.pass_plan` / `--holdout-blocks` | Time-split; default `GVD_HOLDOUT_BLOCKS` / CLI; gap `HOLDOUT_GAP_S` |
| Session / GRU reset | `continual.recording_spans`, `scene_loader.recompute_dt` | Clear state on session change / long gap |
| Frame cache | `frame_cache.py` format `gvd-frame-cache-2` | Chunks capped `CHUNK_BYTES = 3_500_000_000` (< FAT32 4 GB); create-only; stale on format / manifest sha mismatch |
| FAT32 | `AGENTS.md` / README | No file may exceed 4 GB |
| Aux / road_edge | PR #179 | Opt-in; default off for this run |
| Data drive | `check_data_path` | E: approved for cache |

### Cache size clarification (not a format bug)

Prior main-only cache ≈ `10_251_288_960` bytes (**9.55 GiB**) matches `69521 × 192 × 256 × 3` pixels — that is the **sum of shards**, not one file. Official build already splits at ≤ 3.5 GB per `.npy`. Combined ≈ 19.15 GiB remains many shards, each ≤ 3.5 GB. Spec requires an explicit post-write guard that every cache file is `< 4 GiB` (fail hard if not). Do **not** silently emit oversized files.

## Design

### A. Strip list parsing (backward compatible)

1. Change `--strips` to **`action="append"`** (repeatable). Zero occurrences → default `scene_lanes_dir()` as today. One occurrence → **single-root mode** (identical call graph to today). Two or more → **multi-root mode**.
2. Also accept a **single** token that contains commas as a list (`PRIOR,NEW`) so scripts that join paths still work. Empty segments rejected. Paths may be quoted by the shell; commas inside a path are unsupported (document; Windows GVD paths do not use commas).
3. Normalize via `parse_strips_arg(values: list[str] | None) -> tuple[Path, ...]` (absolute/resolved for identity; existence checked like today’s `check_data_path`).
4. New optional `--holdout-source PATH` (or index `0`): which listed root is the validation source. **Default: first `--strips` entry** (the prior). Every other root is **train-only**.
5. Reject unknown `--holdout-source` not in the strips list. Reject duplicate resolved paths.

This fixes the two-token chain failure without a physical merge.

### B. Namespaced row identity

When loading multi-root:

1. Each source gets a stable `source_key` = short hash of the resolved path (hex, e.g. 12 chars) **plus** a human `source_name` = final path component (may collide; key must not).
2. Every row dict gains:
   - `source`: `source_key`
   - `source_root`: resolved path string (for IO)
   - `i`: **unchanged** within-source frame index
   - `session`: `f"{source_key}:{original_session_or_empty}"` so GRU never bridges two recordings that share a folder basename
3. No row from source B may be addressed by source A’s root. RGB / GT / paint / sensor / mask sidecars resolve **only** through that row’s `source_root`.
4. Single-root mode does **not** rewrite `session` (preserve exact old holdout selection and metrics).

### C. Holdout / no leakage

1. Build the pass plan as:
   - `pass_plan(prior_rows, …, blocks=holdout_blocks)` for the holdout source (unchanged algorithm).
   - `_train_only_plan(new_rows, …)` (existing helper: `fraction=0`) for every train-only source.
   - Concatenate: all train pieces (prior train + new train) then prior holdout item(s), in a **deterministic** order documented in code (prior train windows first in time order, then each train-only source in CLI order, then prior holdout).
2. With **only** the prior path and `--holdout-blocks 5`, holdout row **count** and the set of `(i)` values must equal today’s single-root run on that path (fixture-proven).
3. With prior+new: holdout rows’ `source` must all equal the holdout-source key; **zero** new-source rows in any `kind=="holdout"` piece.
4. Train windows must not include holdout `(source,i)` pairs. Gap rows remain neither train nor holdout as today.
5. Mask files remain relative to **each** source root; no cross-root id application.

### D. Frame cache multi-source

1. **Single-root** `build_frame_cache(root, dest, cams, …)` behavior and format `gvd-frame-cache-2` unchanged (existing caches keep working).
2. **Multi-root** build writes format **`gvd-frame-cache-3`**:
   - `sources`: `[{key, root, name, frames}, …]`
   - Each chunk entry includes `source` key; `ids` remain within-source indexes.
   - Index map in `FrameCache`: `(source_key, i) → chunk slot` (string key `"source:i"`).
3. Reader: format-2 loads as today (implicit single source). Format-3 requires callers to pass `source` when reading; trainer uses `row["source"]`.
4. Stale detection: wrong format for the mode, manifest sha mismatch, missing source key, or listed `root` path no longer matching the build-time resolved path → `stale=True` / refuse (create-only rebuild into a **new** folder; never rewrite in place).
5. Every written `.npy` / manifest file size must be `< 4 * 1024**3` or raise before finishing. Keep `CHUNK_BYTES = 3_500_000_000`.
6. Streaming / bounded RAM: keep current per-chunk flush pattern; do not materialize both recordings’ pixels at once.

### E. Trainer / CLI wiring

1. `train_directory` accepts `roots: Path | Sequence[Path]` (or keep `root` + optional `extra_roots`). Single Path path ≡ today.
2. `--build-cache` with multiple strips builds one multi-source cache into `--cache DIR` (create-only on E:).
3. Status / ckpt / holdout_cells paths: default under the **holdout-source** root (or explicit `--ckpt-dir` / `--pause-dir` as today). Document that combined runs should pass explicit `--ckpt-dir` / `--pause-dir` on E: run folders.
4. Aux / road_edge flags untouched; defaults remain off.

### F. README / command template

Document the authorized next run (do not execute in this PR):

```text
py -3 -m python.train.train_scene ^
  --strips E:\GVD_data\sgt_2h_20261105\scene_lanes\ctl_20261105_p01 ^
  --strips E:\GVD_data\sgt_2h_20261105_r2\scene_lanes\ctl_20261105_p01 ^
  --holdout-source E:\GVD_data\sgt_2h_20261105\scene_lanes\ctl_20261105_p01 ^
  --cams main --heads lane_gt --epochs 12 --exact-epochs 12 ^
  --lr 1e-3 --backbone-lr-mult 1 --seed 73 --no-augment ^
  --holdout-blocks 5 --cache E:\GVD_cache\sgt_combined_main_20261105 ^
  --ckpt-dir E:\GVD_runs\sgt_combined_lane_20261105
```

Build cache first with `--build-cache` into the same `--cache` path when empty. No `--aux-*` / `road_edge` on this run.

## Acceptance criteria

1. **Spec committed** on `gvd/combined-recording-training`.
2. **Single-root regression:** one `--strips PATH` (and default empty→scene_lanes_dir) behaves as before; existing format-2 cache still loads; focused + full suite green.
3. **Argparse:** two `--strips` tokens accepted; comma-joined single token accepted; unrecognized-arg failure for the old two-token case is gone.
4. **Namespaced identity:** fixture with two roots sharing basename + overlapping `i` + identical `session` strings trains/holds without collision; sidecars resolve per `source_root`.
5. **Holdout baseline:** prior-only + `--holdout-blocks 5` → identical holdout `i` set vs today’s API on that fixture.
6. **No leakage:** in prior+new mode, holdout pieces contain only prior `source`; new frames never appear in holdout; holdout `(source,i)` absent from train pieces.
7. **Cache:** multi-root build produces format-3 shards each `< 4 GiB`; read-back returns the correct sector for `(source,i)`; stale when a source root path changes or sha mismatches; single-root still format-2.
8. **No merge hack:** tests do not create a combined raw folder; roots stay separate on disk.
9. **Aux default off** / lane-only path unchanged in tests.
10. **Full** `scripts/run_offline_suite.py` on committed head: **≥ 137 passed, 0 failed** (lua available on box/CI).
11. **PR opened** (no merge); **final-head CI attempt1 success** (no reruns); Skeptic handoff for sole critic `edba48ea-7782-455f-9086-5ca17f5deef9` targeting **> 8.5 GREEN, empty Fix list**.
12. Ready-to-use cache/train command template in README (or PR body) preserving prior=holdout / new=train-only; disk note: E: cache ~19.15 GiB aggregate, C: not required.

## Budget / constraints (honest)

| Item | Value |
| --- | --- |
| Prior main cache aggregate | ≈ 9.55 GiB (many ≤ 3.5 GB shards) |
| Combined main estimate | ≈ 19.15 GiB on E: |
| E: free (at authorization) | ≈ 191.7 GiB |
| C: free | ≈ 39.38 GiB — **not** required for this cache |
| Per-file limit | `< 4 GiB` (FAT32); `CHUNK_BYTES=3.5e9` |
| RAM | Stream per chunk; no full-recording pixel materialization |
| VRAM / train wall | **UNPROVEN** in this PR (no train launch) |

## Explicitly UNPROVEN / not claimed

- Quality of the 12-epoch combined run vs prior-only.
- That E: cache I/O rate is acceptable for wall-clock (user accepted the trade).
- Aux/road_edge benefit on these recordings (separate real-data prove).

## Suggested file touch list (smallest)

| File | Change |
| --- | --- |
| `docs/superpowers/specs/2026-11-05-combined-recording-training-design.md` | This spec |
| `python/train/strip_sources.py` (new) | `parse_strips_arg`, `source_key`, `load_multi_source_rows`, holdout/train split helpers |
| `python/train/frame_cache.py` | format-3 multi-source build/read; size guard |
| `python/train/train_scene.py` | `--strips` append; `--holdout-source`; wire multi-root into cache + `train_directory` |
| `python/train/continual.py` | Only if needed for plan concat helper (prefer keep logic in `strip_sources`) |
| `scripts/test_combined_recording_training.py` (new) | Fixtures + AC tests |
| `README.md` | Short blurb + command template |
