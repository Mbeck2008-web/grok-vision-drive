# Sensor aux + road-edge supervision (training-only) — Spec / AC

**Status:** design for implementation on branch `gvd/sensor-aux-supervision`  
**Repo:** `Mbeck2008-web/grok-vision-drive`  
**Date:** 2026-11-05 (America/Chicago)  
**Scope gate:** smallest coherent change; no collector / live ONNX / queued train edits; no CloudAgent.

## Intent

Use recorded **depth** and **semantic** maps as **TRAINING-ONLY** supervision so the camera backbone can learn useful road features, and add **road-edge** as a **SEPARATE** model capability for weak/faint/absent painted lane lines (including dirt roads later). Deployed perception stays **vision-only (RGB)**. GPS remains **route / turn-by-turn only**, never lane geometry or perception (GPS code out of scope).

This PR authorizes **offline learning** of those signals. It does **not** deploy or claim safe driving on unpainted / dirt roads, and it does **not** remove existing live paint / weave restrictions.

## Non-goals

- Modifying the collector, live ONNX path, queued chain, or launching training.
- Reading any live PC recording / GT / raw state.
- Substituting road edges into paint `lane_gt` slots.
- Claiming measured improvement from aux loss weights.
- Claiming dirt-road generalization (UNPROVEN; current west_coast_usa sensor recording may contain none).
- Unbounded redesign of SceneNet / strip / GRU.

## Existing contracts (inspected)

| Contract | Location | Rule kept |
| --- | --- | --- |
| Depth PNG | `raw/<cam>_depth/*.png` uint16 cm | `train_only` in state; not in driving cam lists |
| Semantic PNG | `raw/<cam>_sem/*.png` uint8 class id | + `sem_classes.json` (`kind=gvd_sem_classes`, `train_only`) |
| Scene index | `training_frame_bytes` / listings | Must **not** count or open train-only folders (already tested) |
| Paint labels | `tech_from_painted` / DecalRoad materials | Authoritative for `lane_gt`; missing paint → masked |
| Masks / junctions | event masks, weave/junction guards | Remain authoritative exclusions |
| Default train | `--lr 1e-3 --backbone-lr-mult 1 --heads lane_gt --seed 73` no-augment | Behavior preserved unless explicit aux opt-in |
| ONNX | single RGB strip in; `OUTPUT_FIELDS` + state | Default export unchanged |

Semantic taxonomy comes from BeamNG `get_annotations()` → `sem_classes.json`. Class names are **recording-specific**. Do **not** assume a class named `ROAD` includes dirt, gravel, or unannotated terrain. Unknown / non-allowlisted classes are masked for any road/drivable-derived term.

Paint labeler remains the only source of **painted lane** targets. Road-network edges (`RoadPoly` width / offset) are a **different** signal (nav corridor / mapped road boundary), historically rejected as paint substitutes.

## Design (recommended, tight)

### A. Train-only dense aux (depth + semantic) — feature supervision

1. **Loader** (new helpers under `python/train/`): given a recording root + row index + base cam (default `main`), load aligned `*_depth` / `*_sem` PNGs when present and tagged `train_only` (or absent → empty / all-masked targets, not invented labels).
2. **Schema / alignment:** depth must be uint16 2-D; sem uint8 2-D; shapes must match each other; when RGB main frame size is known, depth/sem must match that HxW. Mismatch or bad `sem_classes.json` schema → **hard fail** with a clear error (no silent resize inventing labels).
3. **Targets:** downsampled fixed grid (e.g. `AUX_H × AUX_W`, small and bounded) for depth (meters or cm→m float) and sem class ids. Invalid / missing pixels masked. Depth 0 may be treated as invalid if the writer documents that; otherwise only explicit missing files/mask.
4. **Heads:** optional training-only heads on the **main-sector stem feature map** (before horizontal pool collapse), predicting the downsampled depth + class logits. **Never** attached to the ONNX wrapper. Detached / unused at export and eval-export.
5. **Losses:** masked L1/Huber for depth; masked CE for sem. Weights CLI-configurable (`--aux-depth-weight`, `--aux-sem-weight`); default **0 / off**. Experimental only — no quality claim.
6. **Opt-in:** `--aux-depth` / `--aux-sem` (or a single `--aux-sensors` that enables both). Default training path identical to today.

### B. Road-edge capability — separate from paint

1. **Name:** `road_edge` (targets `road_edge`, mask `road_edge_valid`). **Not** a `lane_gt` alias. **Not** written into paint slots.
2. **Semantics:** ego-frame **left/right mapped-road boundary** xs at the same lookahead ys as paint (`LOOKAHEADS_M`), plus validity. Absent / off-network / unknown → mask 0. Validity must **not** be interpreted as “assumed drivable area” for planning in this PR (learning signal only).
3. **Sources (priority):**
   - Explicit row fields `road_edge` + `road_edge_valid` when present (tests / future offline labeler).
   - Else optional offline projection from pose + road network dump (`RoadPoly` center ± half width), when both are available to the trainer helper — **never** from DecalRoad paint.
   - Semantic class boundary is **not** used as a paint substitute and is **not** used to invent road_edge xs in v1 (taxonomy / dirt uncertainty). Semantic aux (A) may still train class recognition separately.
4. **Head:** optional linear head on GRU feature, shaped like a dedicated road-edge vector (`ROAD_EDGE_DIM`), parallel to `lane_gt_head`. Enabled only with `--heads …,road_edge` or `--aux-road-edge` (exact flag in plan). Default `--heads lane_gt` unchanged: no road_edge params in the default module path used for today’s export, **or** road_edge module exists but is unused and excluded from export (prefer: not constructed unless opted in, mirroring `ctl`).
5. **Loss:** masked regression; configurable weight; off by default. Does not replace or alter `_lane_gt_loss`.
6. **Unpainted / faint roads:** when paint is missing, `lane_gt_valid` stays clear; `road_edge_valid` may still be set if network projection is valid. Tests must cover: unpainted row → paint masked; road_edge independently valid or masked; **no** copy of road_edge into lane_gt.
7. **Dirt roads:** generalization **UNPROVEN**. No allowlist entry invents dirt as paved `ROAD`. Without held-out dirt data, do not claim capability.

### C. Export / inference contract

1. **Default ONNX:** identical inputs (`strip`, `dt`, `h`, `tokens`, `mask` [+ optional ages/cond]) and `OUTPUT_FIELDS` (+ ctl if that opt-in). **No** depth/sem/road_edge tensors as inputs. **No** aux dense outputs.
2. **Optional additive road_edge export:** only behind an explicit flag (e.g. `--export-road-edge`). Must stamp metadata (`gvd_road_edge=1` / version) and keep legacy output names/order unchanged (append only). Tests: old consumer that reads pre-flag outputs still matches. If additive export cannot stay compatible without mutating legacy outputs, **omit export in this PR** and keep road_edge training-only; document the tradeoff.
3. **RGB-only deployed perception:** documented in README blurb; live path untouched.

### D. Holdout / leakage / memory

1. Aux targets load only for the same row indices already in a train or holdout window; no cross-split file peek beyond that row’s own assets.
2. Tests prove target tensors are **not** concatenated into the network input strip / ages / cond.
3. Aux feature maps and heads must keep a **finite, bounded** activation footprint (fixed `AUX_H/W`, no full-res decoder). Memory probe / unit assertion on tensor sizes.

## Acceptance criteria

1. **Spec committed** on `gvd/sensor-aux-supervision` with this document.
2. **Default compatibility:** with flags LR `1e-3`, backbone-lr-mult `1`, `--heads lane_gt`, seed `73`, no-augment, and aux flags off: loss terms, ONNX I/O names, and lane_gt behavior match pre-change tests.
3. **Aux load:** synthetic fixture with uint16 depth + uint8 sem + valid `sem_classes.json` yields masked-safe aux targets; missing files → masked / skipped term; shape/dtype/schema errors raise clearly.
4. **No paint substitution:** tests show road_edge ≠ lane_gt; unpainted paint slots stay invalid even when road_edge is valid; no code path copies road edges into `lane_gt`.
5. **Semantic caution:** unknown / non-allowlisted class ids do not become “drivable” for any road-edge or drivable loss; allowlist is explicit and tested.
6. **No target leakage into inference inputs:** test builds a batch with distinctive aux target values and asserts forward inputs are RGB strip (+dt/state only).
7. **ONNX parity:** default export input/output name lists equal baseline; optional road_edge export (if implemented) is append-only + metadata.
8. **Opt-in only:** aux losses absent from default `LOSS_TERMS` accounting unless flags on; lane_gt still required for the existing train refusal paths that expect it.
9. **Offline suite:** focused new tests fail-first then pass; full `scripts/run_offline_suite.py` on committed head via `/workspace/venv-a1`.
10. **PR opened** (no merge); CI final-head attempt1 result disclosed honestly; Skeptic handoff prepared for parent → sole critic `edba48ea-7782-455f-9086-5ca17f5deef9`.
11. **Unproven gates called out:** real completed depth/sem train proof; dirt-road generalization; measured aux weight benefit; live memory on full strip+aux on GPU — all UNPROVEN here.

## Spec tradeoffs (report before widening)

| Need | Fit in constraints? | If not |
| --- | --- | --- |
| Dense full-res depth/sem decoder | No — memory unbounded vs strip trainer | Fixed small aux grid only |
| Dirt-road labels from current WC recording | Unknown / likely absent | Leave UNPROVEN; no invented dirt class |
| Road-edge from semantic boundary alone | Unsafe without taxonomy proof | Network projection + explicit fields only in v1 |
| Deploy road_edge in default ONNX | Conflicts with “don’t mutate legacy” + “not deploying” | Training-only head unless explicit additive export flag |
| Change collector to write road_edge into state | Out of scope (collector frozen) | Offline helper + test-injected fields |

## Out of scope follow-ups (parent-gated)

- Train proof on completed real sensor labels after Skeptic GREEN.
- Additive live consumer for road_edge.
- Dirt-map recording + held-out eval.
- Any change to the ~7:25 PM CT queued lane-only train.

