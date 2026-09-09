# `gvd_state.json` fields (BeamNG path ribbon + nerd panel)

Path: `%USERPROFILE%\Documents\GVD\gvd_state.json` (written by `python/run_vision.py`).

| Field | Type | Notes |
| --- | --- | --- |
| `path_ego` | `[{x,y,z}, ...]` | Vehicle frame (+X right, +Y forward, +Z up), ~1 m spacing, 30–40 m |
| `path_world` | optional same | If set, GELua draws this directly |
| `path_width` | m | Default ~2.0 (1.8–2.2 aesthetic) |
| `path_conf` | 0–1 | Low → thinner/shorter/darker ribbon |
| `path_debug_preview` | bool | True when path is steer-preview stub |
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
