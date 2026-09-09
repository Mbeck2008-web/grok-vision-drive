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
