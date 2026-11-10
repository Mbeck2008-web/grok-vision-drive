# SPEC: route conditioning (navigation command as model input)

Rev 1. Based on `gvd/route-intent` (#203). No code claims below are live-proven.

**Goal.** The scene net must turn left, right, or go straight at a junction
when the route says to, the way a navigation-conditioned stack does. Deploy
perception stays camera-only. The only non-camera runtime input is the GPS
turn-by-turn command from `route_intent` (`{turn, next_jct_m}`). This is
**not** `--ctl-cond`: ego speed and control GT never enter the net.

**Why hindsight labels.** BeamNG's AI cannot be told to take a named exit.
For each recorded frame that approaches a junction, the training command is
the turn the car *actually took afterwards*, read from its future trajectory
(heading change and exit through the box) plus the offline road graph /
`state.jsonl` poses. Those poses are training-only GT. At run time the same
command slot is filled from GPS `route_intent`, never from the graph.

## Non-goals

- Feeding `--ctl-cond` fields (speed, steer, throttle, brake, yaw rate).
- Using the map, Navigraph, or trajectory as a live path or localizer.
- Drawing GT or the route polyline on the visualizer (it still draws model
  output only).
- Commanding BeamNG AI; parking; pedestrians; U-turns and roundabouts as
  supported maneuvers (they are masked).

## 1. Hindsight labeler

Sidecar `intent.jsonl`, one object per `state.jsonl` row:

`{t, cmd, next_jct_m, src}` where `cmd ∈ {follow, left, right, straight, mask}`
and `src` is `hindsight`. A masked row may also carry `reason ∈ {ambiguous,
uturn, roundabout, no_future}`. The writer refuses to overwrite `gt.jsonl`.

Constants (fixed, not learned):

| Name | Value | Role |
|---|---|---|
| `LOOKAHEAD_M` | 80 | Future path searched for the next junction |
| `APPROACH_M` | 60 | Inside this range, emit a turn command (else `follow`) |
| `EXIT_M` | 15 | Must observe this far past the node to label the exit |
| `THROUGH_DEG` | 20 | Same as `route_intent.classify_turn` |
| `AMBIGUOUS_LO/HI` | 16 / 24 | Heading change in this band is `mask` / `ambiguous` |
| `UTURN_DEG` | 135 | `|delta|` at or above this is `mask` / `uturn` |
| `ROUNDABOUT_DEG` | 270 | Cumulative `\|dheading\|` within 25 m of the node |

`next_jct_m` is along-track distance to the next degree≥3 node on the future
path (first future sample within 16 m of the node), or `null` when none lies
inside `LOOKAHEAD_M`. Far from a junction (`next_jct_m` missing or
`> APPROACH_M`) the command is `follow`. The traveled **entry and exit edges**
are chosen from the pose `EXIT_M` before and after the node; the turn is
`left` / `right` / `straight` from those edge headings (CCW positive, matching
#203). The same road back is a U-turn. If the recording ends before that exit
sample, `cmd=mask` and `reason=no_future`. A heading change whose graph angle
falls in the 16–24° band is `ambiguous`.

## 2. Model input

**Choice: additive command embedding, not FiLM and not branch-per-command
heads.** A 5-vector (one-hot `follow/left/right/straight` plus
`clip(next_jct_m / 150, 0, 2)`, with missing distance encoded as `1.0`) is
projected by `Linear(5, hidden)` and **added** to the GRU feature before the
existing heads. Reasons: one small layer, default-off architecture is
byte-identical to today's net so old checkpoints load unchanged (`strict=True`
on a default net), and flipping the command changes `lane_gt` without
duplicating every head (4× parameters) or adding FiLM scale/shift that we
have no evidence we need yet.

Flag: `--route-cond`, default off. `SceneNet(..., route_cond=True)` sets
`pause_arch` `...:route_cond=5` and creates `route_proj`. A net without the
flag has no `route_proj` and rejects a `route` tensor. Missing intent at
run time or train time becomes `cmd=follow`.

## 3. Training

`intent.jsonl` is attached onto loaded rows (`route_cmd`, `next_jct_m`).
Masked commands are encoded as `follow` for the tensor (no fifth class) and
are **not** treated as rare. Rows with `route_cmd ∈ {left, right, straight}`
count as rare so `--balance` oversamples junction approaches (class balance
against the much larger `follow` set). Per-command holdout metrics are
`n` and mean error grouped by `cmd`.

## 4. Runtime

Only when the loaded net or ONNX graph has the route input: merge the #203
`route_intent` result into `nav` and encode `{turn, next_jct_m}`. `turn=none`
or a missing result is `follow`. The planner still does not import
`route_intent` or the hindsight labeler. Visualizer unchanged: it draws the
conditioned model output, not the command or the graph.

## 5. Eval / acceptance

- Offline: synthetic T / 4-way / Y fixtures label `left`/`right`/`straight`
  on approach and `follow` when far; U-turn and roundabout paths mask;
  ambiguous 16–24° band masks; CLI writes `intent.jsonl` and refuses
  `gt.jsonl`. Isolation AST: planner / actuation do not import the labeler.
- Model: default `SceneNet` parameter set unchanged; old state_dict loads
  into a default net; `--route-cond` shape matches `lane_gt`; flipping
  `left` vs `right` on the same strip changes `lane_gt`. Trainer exposes
  `--route-cond` and threads it into `SceneNet` and the batch `route` feed.
- Live IQU45 gate: UNPROVEN until a `--route-cond` checkpoint is driven
  with Michael present and GPS intent actually changes the taken exit.

## Files

- `python/collect/route_hindsight.py`, `scripts/label_route_intent.py`
- `python/train/route_cond.py`, `python/train/scene_net.py`, `python/train/train_scene.py`
- `python/train/epoch_ckpt.py`, `python/train/scene_ckpt_runtime.py`, `python/train/scene_lane_runtime.py`
- `python/run_vision.py` (feed only under the flag)
- `scripts/test_route_hindsight.py`, `scripts/test_route_cond.py`
