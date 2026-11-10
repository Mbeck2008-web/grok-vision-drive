# SPEC: `--no-conf-gate` (opt-in confidence-veto bypass)

Base: `origin/main` @ `73790a1` (PR #203 merged). Owner request 2026-11-09: sim toy; crashes are acceptable; watch what the model does at low confidence.

## Goal

Add `--no-conf-gate` to `python.run_vision`. When set, low perception / lane / path confidence no longer blocks Engage or forces hold / stop / no-steer. The model keeps steering. Default OFF is bit-identical to today's behavior.

## Non-goals

- Do not change visualizer ribbon styling, dimming, or scene-ckpt overlay captions (another branch owns that).
- Do not disable AEB, heartbeat / watchdog, missing-frame / no-main stops, user disengage, or Soft Esc.
- Do not add a yaml/env silent on-switch. The only enable path is the CLI flag (default `store_true` off).
- Frame-staleness (`--max-frame-age-ms` / `gate_drive_command` from PR #201, now on `main` @ `7ad9457`) must stay ON regardless of this flag. `--no-conf-gate` must not call, wrap, or parameterize `gate_drive_command`.

## Helper

Every confidence veto routes through one helper, `python.runtime.conf_gate.conf_veto_blocks(*, no_conf_gate, triggered) -> bool`:

- `no_conf_gate=False` (default): return `bool(triggered)` — today's comparison at each call site.
- `no_conf_gate=True`: return `False` — never block on confidence.

Call sites keep their existing trigger expressions so default behavior does not change (including NaN comparisons in `evaluate_veto`).

## Veto paths that must use the helper

1. `evaluate_veto` lane-conf → `low_lane_conf` (Hough / modular / e2e / shadow supervisor).
2. `evaluate_veto` path-conf → `low_path_conf`.
3. `shadow_tick` modular branch: `low_lane_conf` stop and `low_path_conf` stop.
4. `scene_lane_decision` confidence floor (used by `--lane-source scene` and by `apply_scene_ckpt_lane`). Missing or non-finite `lane_x` still stops; that is not a confidence veto.
5. `run_vision` `--scene-ckpt-drive` path that turns `apply_scene_ckpt_lane(...) == "stop"` into `scene_ckpt_low`, and the `--lane-source scene` path that turns a non-`lane` decision into `scene_lane_low`. Those stops follow (4) once the helper is wired.

Bypassing the helper reasons also means `should_disengage` is not set for `low_lane_conf` / `low_path_conf`, so Engage is not dropped for low confidence.

## Must stay ON with the flag

- AEB (`aeb_brake` / `aeb_warn` and `plan_command` brakes).
- Heartbeat / watchdog (`heartbeat_stale` release + disengage).
- Missing / non-finite lane geometry (`scene_lane_decision` on bad `lane_x`; `gt_lane_missing`).
- Scene input-shape HOLD (`scene_input_shape`).
- User disengage, `--no-engage`, Soft Esc / player override.
- Preview-blocked (empty Hough path without `--allow-preview-drive`) — not a confidence veto.

## CLI / UX

- `--no-conf-gate` on `python.run_vision`, `action="store_true"`, default off.
- Startup prints `python.runtime.conf_gate.STARTUP_WARNING` when the flag is on (must include `WARNING` and say confidence vetoes are off while AEB/heartbeat stay on).
- When active, `gvd_state.json` sets `no_conf_gate: true` and the cabin HUD draws `CONF GATE OFF` (`python.runtime.conf_gate.HUD_TAG`) in the top bar. Do not restyle the ribbon.

## Tests (`scripts/test_no_conf_gate.py`)

- Default: each conf veto above still fires.
- Flag: each conf veto is bypassed (steer / `reason==ok` / decision `lane` as appropriate; Engage not dropped).
- Flag: AEB and heartbeat still stop / release.
- Flag: missing `lane_x` still stops.
- Flag: a stale or missing `captured_at` still becomes `frame_stale` with brake 1 and no steer.
- AST: `gate_drive_command` and `python/runtime/frame_gate.py` do not mention `no_conf_gate`.
- `build_parser().parse_args([])` has `no_conf_gate is False`; `parse_args(["--no-conf-gate"])` is True; neither raises.
- AST: `evaluate_veto`, the modular `shadow_tick` branch, and `scene_lane_decision` call `conf_veto_blocks`.

## Verify

```
python scripts/test_no_conf_gate.py
python scripts/run_offline_suite.py
```
