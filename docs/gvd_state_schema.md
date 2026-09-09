# `gvd_state.json` fields (BeamNG path ribbon + nerd panel)

Path: `%USERPROFILE%\Documents\GVD\gvd_state.json` (written by `python/run_vision.py`).

| Field | Type | Notes |
| --- | --- | --- |
| `path_ego` | `[{x,y,z}, ...]` | Vehicle frame (+X right, +Y forward, +Z up), ~1 m spacing, 30–40 m |
| `path_world` | optional same | If set, GELua draws this directly |
| `path_width` | m | Default ~2.0 (1.8–2.2 aesthetic) |
| `path_conf` | 0–1 | Low → thinner/shorter/darker ribbon |
| `path_debug_preview` | bool | `false` only when corridor path came from lanes; geometric/steer fallback stays `true` |
| `gvd_show_path` | bool | Default true |
| `show_agent_ghosts` | bool | Default false; dim agent mode-0 only |
| `agents[]` | optional | `id`, `path_ego` for forecast mode 0 |
| `heartbeat_unix` | float | Age >0.35 s → ribbon fades 0.4 s then hides |
| `policy` | string | `map-ai` draws amber (dimmer) instead of ice-blue |
| `ego.steer_deg` | float | Used for debug preview if path missing |

Lua: `gvd_main.drawPath` on `onPreRender` / `onDebugDraw`. Engaged-only (Alt+A). Caps: 40 ego segs, 8 agents × 10 segs. Label in console: GVD PATH. No DecalRoad / map edit.


| `capture_backend` | string | `beamngpy` / `window` / `stub` |
| `capture_note` | string | e.g. `retail: 1 window` |
| `rss_mb` | float | supervisor RSS; >12 GB is a bug |
| `cam_health.narrow` | enum | added in M1 8-cam set |


## M2 fields

| `detector` | string | `empty` / `synthetic` / `yolov8n-onnx` / `yolov8n-ultra` |
| `tracks[]` | list | id, class, x,y,yaw,speed_mps (weak Δy hint),… ego frame |
| `planner.cipv_id` | int? | lead track id in path tube |
| `planner.ttc_lead` | float? | seconds; null if ego_speed unknown |
| `planner.aeb` | off/warn/brake | state flag only (M2); needs ego_speed > 0 |


## M3 fields

| `engaged` | bool | Mirrored from Alt+A via `gvd_engage.json` (Lua writes; Python reads) |
| `disengage_reason` | string | `none` / `not_engaged` / `preview_blocked` / `heartbeat_stale` / `driver_override` / … |
| `actuator` | string | `beamngpy` / `cmd_json` / `null` |
| `cmd_seq` | int | Monotonic command sequence |
| `cmd_applied` | bool | True only when BeamNGpy `vehicle.control` ran; `cmd_json` sink stays false |
| `cmd_reason` | string | Gate / plan reason; `cmd_json_sink` = file written, car not moved |
| `ego.speed_mps` | float | From Electrics `wheelspeed`/`airspeed` when available; else last known (not invented 10) |
| `ego.throttle` / `ego.brake` | float | Last commanded values |

Also: `Documents/GVD/gvd_cmd.json` = `{steer,throttle,brake,seq,heartbeat_mtime}` fallback sink.


## M4 fields

| `last_clip_trigger` | string | `none` / `disengage` / `aeb_brake` / `near_miss_ttc` / `manual` / `smoke` |
| `last_clip_path` | string? | Last flushed clip directory under Documents/GVD/clips |
| `encode_backend` | string | `h264_qsv` / `libx264` / `h264_nvenc` / `none` |


## Dual-viz

| `path_world` | optional `[{x,y,z}]` | World path from BeamNGpy pose × path_ego (kinematics, not a map). Lua prefers this for 1:1 ribbon. |
| `show_agent_ghosts` | bool | Default true when `tracks_n>0`; in-game track hulls + OpenCV ghosts |
| `planner.cipv_id` | int? | Brighter ice / LEAD on that track |


## Path note

Lua and Python share `%USERPROFILE%\\Documents\\GVD` (or HOME). GELua often has empty `USERPROFILE`; `gvd/main.lua` derives the profile from `LOCALAPPDATA` / `FS:getUserPath` / `FS:virtual2Native` instead of writing bare files under the BeamNG userfolder.

## UI prefs (`Documents/GVD/gvd_ui_prefs.json`)

Written by the in-game **GVD** app (GELua is the only writer). **Wins over** `gvd_state` for path/ghosts while the file exists. Setters also mirror path/ghosts into `gvd_state` when JSON encode is available so OpenCV follows.

| `show_path` | bool | Ribbon on/off |
| `show_agent_ghosts` | bool | Track hulls on/off |
| `show_scene` | bool | In-app VISION canvas on/off (does not change the world ribbon) |
| `policy` | string? | `modular` / `e2e` / `shadow` request; only present while the player picks one this session |
| `viz_screen` | string? | `auto` / `1` / `2` / `3` — where the OpenCV `GVD VISION` window should sit |
| `mtime` | int | `os.time()` at write |

`policy` / `viz_screen` are **session requests**: `run_vision.py` applies them only when `mtime` is at or after supervisor start, so a pref from a past session never overrides `--policy` at launch. Modular veto and dead-man are unchanged by a policy request.

## In-game app bus

`gvd/main.lua` pushes `guihooks.trigger('gvdUi', …)` every 250 ms (100 ms while the app's scene is on) with the state above plus capped scene geometry: `path` ≤28 points, `tracks` ≤12 (`id,cls,x,y,yaw,v,lead`), `lanes` ≤3×12 points, all ego frame and rounded to 2 dp. `link` is `live` / `stale` / `none` from the heartbeat age, so the app can show the dead-man without a second file bus. `gvdStrip` keeps its old shape.

| `viz_window` | bool | Python `--viz` OpenCV window exists |
| `viz_screen` | string | Monitor the window was last placed on |
| `viz_note` | string | e.g. `screen 2 (non-primary)` / `auto→primary (second screen not found …)` |


## M5 fields

| `policy` | string | `modular` (default; safety supervisor) / `e2e` / `shadow` |
| `shadow.steer` | float | E2E proposed steer [-1,1]; written every tick even when disengaged |
| `shadow.throttle` | float | E2E proposed throttle [0,1] |
| `shadow.brake` | float | E2E proposed brake [0,1] |
| `e2e_ok` | bool | False when modular vetoes E2E (low lane_conf / heartbeat / disagreement / forward fail) |
| `veto_reason` | string | `none` / `aeb_brake` / `aeb_warn` / `low_lane_conf` / `low_path_conf` / `heartbeat_stale` / `disagreement` / `e2e_forward_fail` / `preview_blocked` |
| `e2e_backend` | string | `stub` / `onnx` |

Perception always runs. Actuators only when engaged **and** modular OK. Shadow mode computes both intents; default apply path stays modular. Dead-man / heartbeat unchanged. No weight blobs in git (`models/e2e_current.onnx` gitignored).

