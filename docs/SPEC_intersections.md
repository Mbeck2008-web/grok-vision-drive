# SPEC: intersections, junction turns, routing (rev 5, no code)

Base: `main` @ `dd6598d`. APIs checked against the beamngpy 1.36 and 1.35.1 wheels.

**Project rule:** strictly vision-only for perception and control. The single runtime map exposure is the route intent {route_src, next_jct_m, turn}, like GPS. Only `turn` may reach the planner, as intent, at M4. All other map data (`get_road_network`, `NavigraphData` geometry, `traffic_signals`, labeler geometry) appears only as **offline labels, ground truth and grading**. It never feeds steer, pedals, stops, lane choice or path.

**Owner decisions folded in:**

| Topic | Decision |
|---|---|
| Stops | Lamp and sign state come from vision only. |
| Turn lanes | Turn-lane choice comes before turns ship. |
| Live proves | Wait for the loop dt bar (p50 ≤162 / p95 ≤215 ms) and an 8-cam checkpoint. |
| Pin | beamngpy **>=1.36**; 1.35 dropped. |
| Unknown lamp | Keep today's rule: unknown is not red (`path_predictor.py:18` unchanged). |
| Junction box | The path through the box comes **from vision**. |
| Route purpose | Reach a user-set map pin (M5). |
| Route source | **Option B:** the map route is GPS navigation only and gives just {turn, next_jct_m}. Everything else is vision. |
| **Non-goal: parking** | No park trigger or mode. Parking may only emerge later from training data. |

## 1. Current state

- **README.** `README.md:443` says lane keeping only and no routing. `README.md:343` says the labeler ignores paint more than 45° off heading (stop lines).
- **The live scene path is a straight line.** `run_vision.py:929-963` uses one 8 m `lane_hat`, through `straight_lane_path` (`scene_lane_runtime.py:85`). `gt` mode does the same (`:904-927`).
- **The planner** (`path_predictor.py:1-22,290`) runs on the hough path only (`pipeline.py:316-330`).
  - Stops come from scene signs (`:1145`).
  - The blinker is guessed from path shape (`:463`).
- **Scene heads** (`scene_net.py:121-135`): lanes, curbs, objects, signs (class and lamp, `:63-64`), `lane_gt`.
  - There is no turn-lane, junction-path or stop-line head.
- **Pin surfaces:**
  - `requirements-beamng.txt:2-5`
  - `config/tech.yaml:11,13,23`
  - `python/sensors/tech.py:428-429`
  - `scripts/test_tech_session.py:2139-2140,2208` asserts `>=1.35,<1.37`.
- **beamngpy:** `get_road_network`, `get_road_edges` and `NavigraphData`/`GetRoadGraph` exist in 1.35 and 1.36. `traffic_signals.*` is 1.36 only. There is no `ai.drive_to`.

**Road-network reads today. Every one is a full `get_road_network` socket read.**

| # | Call site | Reached from | When it fires | Re-read on respawn? | Level change? |
|---|---|---|---|---|---|
| R1 | `gt_logger.py:446` `_ensure_roads` → `load_road_polys` (`:187`, read `:203`) | `write` `:460`, `curve_preview` `:488`, `:568` | Lazily on the first GT tick of a supervisor run (`--gt-log` or `--lane-source gt`). Retries every 2 s while empty. | No (cached forever) | **Never invalidated (stale-cache bug)** |
| R2 | `level_session.py:446` `_roads` → `load_road_polys` | `begin_level` `:773,800,802(poll),807` ← `enter_map` (`collect_scene_lanes.py:973`) at `:1055` (map checks), `:1091` (road dumps), `:1609` (`--reload-level`), `:2252` (segment/map change) | Each level or segment load. `:802` polls repeatedly until roads appear. | No | Yes (reads again) |
| R3 | `level_session.py:705` `snap_to_road` → `_network_spec` (`:647`, read `:652`) | `begin_level` `:812`; **`collect_scene_lanes.py:1891` in `_abort` recovery** | Each level load, plus **every recovery respawn, once per `recover_points` candidate** | **Yes (up to N reads per recovery)** | Yes |
| R4 | `gt_logger.py:146` `raw_road_network` (read `:162`, `get_roads` `:177`) | `collect_scene_lanes.py:1093` `run_road_dumps` | Once per level in road-dump mode | No | Yes |

## 2. Budgets (all milestones)

- **No per-tick socket reads.**
- Runtime intent and routing logic adds ≤1.0 ms p50 and ≤3.0 ms p99 to `heartbeat_ms` (`run_vision.py:1406`).
- All new scene heads together add ≤2 ms to CPU onnxruntime inference.

## 3. Milestones

| # | What | Map at runtime? | Actuates? | Can start |
|---|---|---|---|---|
| PR0 | Pin >=1.36 (small PR) | no | no | now |
| PR1 | M0 loader subtract + offline junction grader | no | no | now |
| M1 | GPS route intent {turn, next_jct_m}, shadow only | GPS-only | no | now (offline) |
| M2a | Turn-lane labels + head | no | no | after labeler |
| M2b | Vision junction-path head | no | no | after labeler + junction recordings |
| M3a | Vision stops: stop-line head + sign/lamp gate | no | no | after labeler, PR0 |
| M4 | Live junction prove | GPS-only (`turn` as intent) | yes | dt bar + 8-cam checkpoint + M1 + M2a/M2b/M3a gates |
| M5 | Click-to-goal map pin (destination source) | GPS-only | no | after M1 |

### PR0: pin >=1.36

- **Change:**
  - `requirements-beamng.txt:5` → `beamngpy>=1.36,<1.37`.
  - Drop the 1.35/0.38 lines from `config/tech.yaml` and `tech.py:428-429`.
  - Update `test_tech_session.py:2139-2140,2208`.
  - Add a README note.
- **Accept:** `rg -n "1\.35|0\.38" requirements-beamng.txt config python/sensors/tech.py` returns nothing, and the offline suite is green. The 1.35.1 socket shims (`tech.py:639,760`) may be removed only if tests prove they're dead. Otherwise leave them and note it.

### PR1: M0 loader subtract + offline-only junction grader (no runtime navgraph)

**Goal:**
- One road loader, `python/collect/road_loader.py`, with a cache keyed by level name and invalidated on level change. R1–R4 all use it.
- That fixes R1's stale cache and R3's re-read on recovery.
- An **offline-only** `scripts/junction_grade.py` labels each junction crossing in a GT log as left, through or right, from the logged heading change. It uses synthetic or offline road data.

Nothing in PR1 is imported by the runtime control path.

**Non-goals:** routing, intent, any `DriveCommand`/`path_ego`/blinker change, parking.

**Touch:**
- `python/collect/road_loader.py` (new)
- `python/debug/gt_logger.py` (R1, R4)
- `python/collect/level_session.py` (R2, R3)
- `scripts/collect_scene_lanes.py:1093` (R4 caller)
- `scripts/junction_grade.py` (new)
- `scripts/test_road_loader_cache.py`, `scripts/test_junction_grade.py`, `scripts/test_lane_gt_frozen.py` (new)
- `tests/fixtures/junctions_synthetic.json`: **hand-built**, no BeamNG map data. It holds a T, a 4-way, a Y, a one-way and an unreachable node.
- `tests/fixtures/lane_gt_pre_pr1.jsonl.gz`

**Acceptance:**
1. **Single read site.** `rg -n "get_road_network\(" python scripts -g '!scripts/test_*'` finds only `road_loader.py`.
2. **Cache lifetime.** A fake `bng` counts reads:
   - session start: 1;
   - 1,000 GT ticks: 0 extra;
   - recovery with 5 `recover_points`: **0 extra**;
   - `begin_level` with the same level: 0;
   - level change: exactly 1;
   - an empty network keeps R1's 2 s retry.
3. **`lane_gt` frozen.**
   - Commit 1 of the PR adds `lane_gt_pre_pr1.jsonl.gz`, **generated on unchanged `main` @ `dd6598d`** by driving `GroundTruthLogger` over the synthetic roads with a scripted pose track. It is not regenerated afterwards.
   - The test compares post-change output to it byte for byte. The fixture sha256 goes in the PR body.
   - **Scope: steady state only.** One level, no level change, no recovery respawn. Those cases are covered by tests 3a and 3b, not by this fixture.
3a. **R1 level-change test (an intended behaviour change).**
   - The fake `bng` serves level A's synthetic roads, then reports a level change and serves level B's.
   - After the change, `GroundTruthLogger` output (`lane_x`, `gt_center_x`) must come from **B's** roads.
   - Old `main` @ `dd6598d` must FAIL this test, because it keeps A's roads forever (`gt_logger.py:440`). The test is committed red on old main first, then turned green, and the PR body declares the change.
3b. **R3 recovery-equivalence test.**
   - The cached network is deep-equal to a fresh `get_road_network(include_edges=True, drivable_only=True)` read from the same fake.
   - For each `recover_points` candidate, `snap_to_road` after a recovery respawn gives the same pose as today's code path, which re-reads on every call (`level_session.py:705`).
4. **Grader.** It classifies the synthetic T, 4-way and Y crossings correctly. It honours the one-way. The unreachable node gives `no_route` with no exception.
5. **Isolation.**
   - An AST scan shows `run_vision.py`, `actuate.py`, `path_predictor.py`, `scene_lane_runtime.py` and the `perception/*` modules do not import `junction_grade` or any navgraph code.
   - `road_loader` is imported only via `gt_logger` and `level_session`.
6. The offline suite is green with no new dependencies.

**Verify:**
```
py -3 scripts/run_offline_suite.py
py -3 scripts/test_road_loader_cache.py
py -3 scripts/test_lane_gt_frozen.py
py -3 scripts/test_junction_grade.py
```

**Optional dev-PC run (grader proof):**
- Before the run, write the bar file in `logs/`: "turn match ≥95% over N ≥10 junctions; with N=10 this is 10/10". Record its sha256 in the PR. The grader prints that hash and refuses a bar file whose hash differs.
- Drive manually with `--gt-log --no-engage`. `junction_grade.py` compares its heading-change class to the branch taken in the offline road dump, automatically for every junction.

**Merge precondition:** the R1 recording audit is done and attached to the PR. Done in rev 5; see "R1 audit" below.

**Stop if:**
- `lane_gt` bytes differ.
- Any read site needs a per-tick read.
- A level-change signal isn't observable at R1. In that case, document it and key the cache on the level the vehicle is in, not the session.
- The synthetic fixture can't drive the real `GroundTruthLogger` path.

**R1 audit (code proof at `dd6598d`, no data read). No recording is affected.**
- **Recorded labels never go through R1.**
  - The collector's `gt.jsonl` labels come from `tech_from_painted(painted.lane_at(...))` at `collect_scene_lanes.py:2568`.
  - `painted` is the level's painted line decals, rebuilt on every level change: `painted_for` at `:882`, called at `:1611` (`--reload-level`) and `:2254` (segment or map change).
  - The road network (R2, refreshed per level by `enter_map`) only drives the excursion controller (`follow_label` `:2433`). Per the comment at `:2564-2566`, it is not a label.
  - `load_road_polys` is imported at `:1451` but never called; this is its only occurrence in the file.
  - `GroundTruthLogger` is instantiated only by the supervisor (`run_vision.py:619`). It writes debug logs `logs/gt_<stamp>.jsonl` (`gt_logger.py:428-437`), not recordings. Training reads each recording's `gt.jsonl` (`raw_frames.py:272,595`). The offline relabelers (`scripts/relabel_v103.py`, `scripts/relabel_painted.py`) do not import `gt_logger`.
- **The supervisor never changes level itself.** `run_vision.py`, `python/sensors/*` and `python/control/*` contain no `load_scenario`, `start_scenario`, `enter_map`, `begin_level` or `spawn_scenario` call.
- **R1 staleness can therefore only hit** a supervisor `--gt-log` or `--lane-source gt` session where the level is changed by hand in-game. That affects GT debug logs and `gt`-mode driving, never recordings or training caches.
- Dev-PC data was not inspected, because the code rules it out.

### M1: GPS route intent (shadow only)

- **What it does:** `NavigraphData` gives a route from ego to the M5 pin. The only output is {turn, next_jct_m}, from which `route_src` is derived in-process. There is no geometry, stop, lane or speed output.
- **Where it runs:** a separate route process (`multiprocessing.get_context("spawn")`, Windows-safe) holds the graph and the route cursor. Each tick, the supervisor does one non-blocking pose put and one result get.
- **Accept:**
  - **Per-tick IPC:** the put and get together cost ≤1.0 ms p50 / ≤3.0 ms p99.
  - **Empty get:** when the non-blocking get returns nothing, the tick reuses the last result and logs `route_age_ticks`, the number of ticks since the last fresh result. A test asserts the reuse and the age count.
  - **Heartbeat:** `heartbeat_ms` p50/p99 stays within +1.0/+3.0 ms of baseline (route off), both during an active solve and with none.
    - Dev-PC measurement: offline on i9-9900K / Windows via `scripts/route_replay_bench.py`, which replays a recorded tick stream at real rate with no BeamNG.
    - **The dev-PC replay bench is the real gate.**
    - CI: the offline suite runs the same harness with a looser +5/+15 ms bound. That is a **fail-only sanity check**: passing it proves nothing about the budget, and runner hardware is not representative.
  - **Shadow proof:** a replay fixture (synthetic roads plus a scripted track) crosses ≥2 junctions with an active route, with `turn != straight` on ≥1 of them.
    - Steer, throttle, brake and blinker must be byte-identical with and without `--route-goal`.
    - The telemetry must actually be populated: `turn ∈ {left, right}` seen, and `next_jct_m` falling to ≤5 m at each junction.
    - An AST import scan is an extra check.
  - **Queue schema check (AST plus runtime):**
    - Every message the route process puts on its result Queue must have exactly the keys {turn, next_jct_m}, with `turn` ∈ {left, right, straight, none} and `next_jct_m` a float or None.
    - An AST scan of the route-process module must show its only `put` call sends that schema.
    - A runtime test wraps the Queue and fails if any message carries another key, or any list, dict, node id or polyline value.
    - Net effect: no geometry, nodes or polylines cross the Queue.

### M2a: turn-lane head (offline)

- **Label:** from the queued labeler's turning lanes, `lane_turn[6,3]` gives the allowed {left, through, right} per lane slot within 60 m of a junction, plus a mask.
- **Head:** `lane_turn_head`, a linear layer to 6×3 with sigmoid, plus `HEAD_NAMES` `lane_turn`. Fine-tune on a frozen trunk first, with a full retrain only if it misses the gate.
- **Use (M4):** the intent picks a lane slot whose bit is ≥τ. If none qualifies, the car does not turn: it keeps lane and the miss is logged. The rightmost-lane assumption is shadow-only.
- **Gate (holdout):**
  - per-bit F1 ≥0.95;
  - ego-lane allowed-turn correct ≥98% at 30 m;
  - false "allowed" ≤0.5%.

### M2b: vision junction-path head (offline)

- **Label:**
  - The future ego trajectory actually driven, in the ego frame. It covers from ≤40 m before the box through +15 m after, resampled to 16 points over 0–40 m of arc.
  - It is filed under the intent taken, which the PR1 grader classifies offline.
  - Only crossings with no lane departure or impact count.
  - The labeler's junction geometry is a corridor for grading only.
  - **The map is never used at runtime.**
- **Head:** `junction_path_head`, outputting 3×16×2 (one path per intent) plus 3 valid flags. Runtime picks the path by intent, so there's no new ONNX input.
- **Gate (holdout crossings):**
  - lateral error median ≤0.3 m and p95 ≤0.8 m over 0–20 m;
  - through-box success (the predicted path stays inside the GT corridor) ≥98%;
  - false-valid for an impossible intent ≤1%.
- **Budget:** counted in the ≤2 ms total for new heads.
- **Risk:** the recordings are west_coast_usa only with 2 traffic cars and few turns, so the collector needs junction-crossing coverage.

### M3a: vision stops (offline)

- **Labels:**
  - the labeler's stop line `y_m` plus control class;
  - lamp ground truth from `traffic_signals.get_map_node_signals` (1.36, via PR0), **recorded offline only**.
- **Head:** `stop_line_head` = (y_m, valid). The existing sign and lamp heads are reused.
- **Gate (approaches ≤40 m):**
  - sign/light recall ≥0.98, precision ≥0.95;
  - red/yellow recall ≥0.99;
  - green-when-red ≤0.1%;
  - stop-line error median ≤0.5 m and p95 ≤1.5 m at ≤20 m;
  - state stable for ≥3 ticks.
- **The gate report also tracks the unknown-lamp share.** The unknown rule is unchanged.
- **Shadow:** then ≥200 live approaches at ≥98% agreement.
- **Actuation:** vision via `_apply_scene` only.

### M4: live junction prove

- **Path:** vision lane, then the M2b vision junction path for the intent, then vision lane. **No map geometry and no `map/junction` tag.** The blinker comes from intent, and `blinker_for_path` is retired.
- **Accept:**
  - 20/20 turns in the correct lane across ≥3 junctions;
  - 10/10 vision stops;
  - 0 actuations sourced from map;
  - budgets held.
- **Verify:** `live_lane_test.py` with 2 traffic cars.

### M5: click-to-goal

- Click on the VIZ BEV (`stage.py:1,92`) to set the destination pin, sent over the prefs bus (`run_vision.py:706`).
- The route is solved in the M1 route process.

## 4. Owner decisions (no open questions)

- Pin is >=1.36.
- Unknown lamp keeps today's rule.
- Junction-box path comes from vision.
- Stops come from vision.
- Turn-lane choice comes before turns ship.
- Live proves wait for the dt bar and an 8-cam checkpoint.
- Parking is a non-goal.
- **Route source is Option B:** GPS-only {turn, next_jct_m} toward a user-set pin.
