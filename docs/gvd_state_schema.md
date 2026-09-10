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

Lua: `gvd_main.drawPath` on `onPreRender` / `onDebugDraw`. Engaged-only (Alt+G). Caps: 40 ego segs, 8 agents × 10 segs. Label in console: GVD PATH. No DecalRoad / map edit.


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

| `engaged` | bool | Mirrored from Alt+G via `gvd_engage.json` (Lua writes; Python reads) |
| `disengage_reason` | string | `none` / `not_engaged` / `preview_blocked` / `heartbeat_stale` / `player_steer` / `player_brake` / `player_throttle` / … (sticky reasons live in `gvd_engage.json`) |
| `actuator` | string | `beamngpy` (Tech) / `cmd_json` (retail: GELua applies) / `null` |
| `cmd_seq` | int | Monotonic command sequence |
| `cmd_applied` | bool | True when BeamNGpy `vehicle.control` ran, **or** the mod acked the seq via `gvd_ego.json` (fresh, `applying`) |
| `cmd_reason` | string | Gate / plan reason; `cmd_json_applied` (acked) / `cmd_json_pending` (written, no ack) / `cmd_json_idle` (not engaged) |
| `ego.speed_mps` | float | From Electrics `wheelspeed`/`airspeed` (BeamNGpy) or the mod's `gvd_ego.json` echo; else last known (not invented 10) |
| `ego.throttle` / `ego.brake` | float | Last commanded values |
| `ego.yaw_rate` / `ego.accel` | float | Tech: yaw from consecutive pose dirs; accel from GForces gx/gy. Else 0 |
| `ego_source` | string | `beamngpy` / `lua` (gvd_ego.json fresh) / `none` — M6 |
| `cmd_ack_seq` | int | Last `applied_seq` echoed by the mod (-1 none) — M6 |
| `lua_applying` | bool | Mod currently holds the player vehicle's inputs — M6 |
| `vehicle.vid` / `model` | string? | Tech player vehicle when connected |
| `vehicle.connected` | bool | BeamNGpy vehicle handle is live |
| `vehicle.damage` | float? | Ground-truth Damage sensor; null when missing |
| `vehicle.gear` / `rpm` | | Electrics extras |
| `vehicle.pose_ok` | bool | `pos` + `dir` present for `path_world` |
| `vehicle.sensors` | dict | `electrics`/`damage`/`gforces`/`gps` → `ok`/`missing` |
| `nav.mode` | string | `missing` (retail / no GPS fix) / `hint` (Tech lat/lon). Never `route` until a planner exists |
| `nav.drive_to_pin` | bool | Always `false` today — pin is a hint, not a route |
| `nav.gps` | `{lat,lon,x,y,ok}` or null | BeamNGpy GPS; BeamNG maps have no real-world lat/lon (`config/tech.yaml` `gps.ref_*` is world origin) |
| `nav.pin` | `{lat,lon,name}` or null | Destination from `nav.pin_*` / `GVD_NAV_PIN_*`; empty until you drop a pin |
| `nav.range_m` / `bearing_deg` | float? | Haversine range; bearing clockwise from north |
| `nav.bearing_rel_deg` | float? | Pin vs heading (−180..180, + = right). Heading from GPS motion or world `dir` |

Also: `Documents/GVD/gvd_cmd.json` = `{steer,throttle,brake,seq,engaged,heartbeat_mtime,reason}` (see M6 below).
`config/tech.yaml` owns host/port/home, wait-for-vehicle, vehicle-data sensors, GPS origin, and the optional nav pin (RGB cameras stay in `cameras.yaml`).


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

## Viz road model

Drawn by the OpenCV **GVD VISION** window (`python/viz/stage.py`). The in-game **GVD** app is Engage + settings only. None of this feeds the planner — corridor, CIPV and AEB still read `lanes_bev` / `tracks` exactly as before.

| `lanes_ext[]` | `{points:[{x,y}], kind, side, style, index}` | `kind`: `detected` (Hough saw the paint) / `predicted` (a detected boundary offset sideways by the measured lane width) / `stub` (`--smoke` only). `index` counts boundaries out from the ego lane (`±1` = its own edges, out to `±3` for the two-lane-per-side fan). `style` stays `unknown` — nothing classifies solid vs dashed yet |
| `road_edges[]` | `{points:[{x,y}], kind, side}` | Kerb line just outside the outermost boundary. **Always `predicted`**: no kerb detector exists, this is the road edge implied by the lanes we can see |
| `signs[]` | `{cls,x,y,conf,state}` | Road furniture straight from the detector: `stop_sign` (COCO 11), `traffic_light` (9), `pole` (10 / 12 — hydrants and parking meters, drawn as short grey sticks). Never tracked and never offered to CIPV or AEB. `state` is `unknown` for lights — no lamp-colour classifier, so the OpenCV stage draws all three lamps as empty rings |
| `agents[]` | `{id, path_ego:[{x,y}]}` | Mode-0 constant-yaw-rate forecast fan per moving track, same toy math as the OpenCV view |

### Scene cues derived in the OpenCV stage

These come from state the stage already has — no extra fields, and each one needs its own signal:

| Cue | Condition |
| --- | --- |
| Road user lit ice-blue | inside the planner's corridor: `\|x\|` within `path_width/2 + 0.45` of a `path_ego` point at that range. No corridor → nothing highlighted |
| Road user red + `BRAKE` tag | it is the CIPV **and** (`planner.aeb == brake` or `planner.ttc_lead < 1.5`) |
| Slow-down chevrons on the ribbon | `planner.aeb` not `off`, `ego.brake` above 0.05, or `planner.target_v` below `ego.speed_mps` |
| Ribbon shade | accelerating (`target_v` above speed) brightest, coasting normal, slowing dimmer, planned stop faintest |
| Hard stop bar across the ribbon | planner halted (`target_v <= 0.2` or AEB brake) **and** a CIPV exists — the bar sits at the lead, because that is the constraint being stopped for. No CIPV means no stopping point we can honestly claim, so no bar |
| Traffic light tinted ice-blue | `signs[].relevant == true`. **Nothing sets it today** — the stack has no route-relevance signal, so every light renders muted |

The only filled surface in the scene is the ego corridor; lane paint, kerbs and the lane fan are thin vector polylines, and the sky is void — there is no backdrop.

Predictions need an anchor: with no detected lane there are no predicted lanes and no road edges, and with `lane_conf` under 0.25 only the detected boundaries ship. Sign positions inherit `project_box_to_ego`'s crude pinhole estimate, and sign/light heights in the scene are a drawing convention, not a measurement. The stage draws detected geometry solid and everything predicted dim + dashed, skips `kind=stub` except `--smoke` (`viz_smoke`), and prints e.g. `lanes 2 seen+2 pred · edges pred · 2 signs` under the window. Loop under 8 Hz drops forecast fans, signs and the `cam_main` PIP.

## In-game app bus

`gvd/main.lua` pushes `guihooks.trigger('gvdUi', …)` every 250 ms (100 ms while the app's scene is on) with the state above plus capped scene geometry: `path` ≤28 points, `tracks` ≤12 (`id,cls,x,y,yaw,v,lead`), `lanes` ≤6×12 (`pts,kind,side,style,idx`), `edges` ≤2×12, `signs` ≤8, `fans` ≤6×8, all ego frame and rounded to 2 dp. `link` is `live` / `stale` / `none` from the heartbeat age, so the app can show the dead-man without a second file bus. `applying` rides along, so the app can show `DRIVE` while the mod holds the wheel (M6 retail); on Tech, `actuator=beamngpy` + `cmd_applied` means the same thing. `gvdStrip` keeps its old shape plus `applying`.

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


## M6 — `gvd_engage.json` contract (retail package)

Path: `Documents/GVD/gvd_engage.json`, shared by GELua and Python.

| Writer | Payload | When |
| --- | --- | --- |
| Lua (`gvd_main.writeEngageFile`) | `{"engaged":true\|false,"mtime":<os.time() int>,"disengage_reason":"<why>"}` | Alt+G / GVD app button, `player_steer` / `player_brake` / `player_throttle`, `command_stream_dead`, `extension_unloaded` |
| Python (`write_engage_flag`) | `{"engaged": false, "mtime": <time.time() float>, "disengage_reason": "<why>"}` | `player_steer` / `player_brake` / `player_throttle` / modular veto / stale heartbeat / `finally` on exit |

Python reads the file every tick and mirrors it into `engaged` (never invents engage). Lua polls it every 0.1 s **only while engaged** and adopts `engaged=false` when the file says so and `mtime` ≥ Lua's own last toggle stamp; it logs `[GVD] DISENGAGED by supervisor (<disengage_reason>)`, releases the vehicle inputs and refreshes the HUD/UI app. A file saying `true` never engages Lua — engage always starts in-game. Both sides write `false` on `player_steer` / `player_brake` / `player_throttle` (sticky, whichever sees it first) and Lua writes `false` when its dead-man fires. The file is the durable record of *why*: later supervisor ticks only see `engaged=false` and write the generic `not_engaged` into `disengage_reason`, so the mod keeps the reason it adopted for the HUD.

## Player override (force-feedback residual)

`config/control.yaml` `override:` owns the research-pin thresholds; the supervisor mirrors them into `override_cfg` every tick so `gvd_main` runs the same numbers without parsing YAML. The signal is `|steering_input − cmd.steer|` against the command in force when the echo was sampled, never an absolute angle. Live FFB is **UNPROVEN**. `CMD_DEAD_S` is not part of this.

| Field | Type | Notes |
| --- | --- | --- |
| `override_cfg.steer_enter` / `steer_exit` | float | 0.08 / 0.04. Dwell charges above enter, discharges below exit, frozen between |
| `override_cfg.steer_hold_ms` | float | 200. How long the player has to keep pushing one way |
| `override_cfg.steer_spike` | float | 0.20. Sample-to-sample jump past this is mechanical: the EMA holds |
| `override_cfg.lpf_tau_ms` | float | 80. EMA time constant, on the residual only |
| `override_cfg.brake_enter` / `throttle_enter` | float | 0.06 / 0.10. Pedals: no filter, no dwell, one-sided, brake tighter |
| `override.channel` | string | `none` / `steer` / `brake` / `throttle` |
| `override.reason` | string | `none` / `player_steer` / `player_brake` / `player_throttle` |
| `override.steer_raw` / `steer_filt` / `steer_eff` | float | Residual, after EMA, after opposition bias |
| `override.pedal_residual` | float | Pedal press beyond the aligned command |
| `override.steer_held_ms` | float | Current dwell |
| `override.spike` | bool | This sample was spike-rejected |
| `override.opposition` | float | 0..1, how much the residual fights GVD's steer |
| `override.ref_seq` | int | Command seq the residual was measured against |
| `override.armed` | bool | False during the ~3 τ warm-up after engage |

On a trip both sides write `gvd_engage.json` `engaged=false` with the `player_*` reason and stay off until Alt+G; the override tick also puts that reason in `gvd_cmd.json`.

## M6 — retail drive bus (`gvd_cmd.json` → vehicle, `gvd_ego.json` ← vehicle)

`Documents/GVD/gvd_cmd.json` — written by Python every tick (`CmdJsonActuator`, atomic tmp+replace with retry):

| Field | Type | Notes |
| --- | --- | --- |
| `steer` | -1..1 | `+` = right (BeamNG `input.event('steering')` / `kbdSteer` convention) |
| `throttle` / `brake` | 0..1 | Brake > 0 ⇒ Lua forces throttle 0 |
| `seq` | int | Monotonic; Lua treats a non-advancing seq as a stalled supervisor |
| `engaged` | bool | Supervisor-side engaged after gates. Lua applies **only** when this is true and it is engaged itself |
| `heartbeat_mtime` | float | `time.time()`; Lua ignores files whose stamp is > `CMD_DEAD_S` (1.0 s) behind `os.time()` (old session) |
| `reason` | string | `ok` / `preview_blocked` / `not_engaged` / `veto:*` / … (diagnostic) |

Lua (`gvd_main.applyCmdJson`, 20 Hz): `input.event('steering', s, 1)`; `input.event('throttle', t, 2)`; `input.event('brake', b, 2)` on `be:getPlayerVehicle(0)` via `queueLuaCommand`; `drivetrain.setShifterMode('arcade')` once. No new seq for `CMD_STALE_S` (0.35 s) → steer 0 / throttle 0 / brake 1 hold; after `CMD_DEAD_S` (1.0 s) → release (all 0), `engaged=false`, `gvd_engage.json` false. Any disengage (Alt+G, supervisor false, unload) sends one release and stops applying. `cmd.engaged=false` → release immediately (no brake tap on the player).

`Documents/GVD/gvd_ego.json` — written by Lua at ~10 Hz while the supervisor's state heartbeat is alive (vehicle Lua `electrics.values` → `obj:queueGameEngineLua` → `gvd_main.onEgoFeedback`):

| Field | Type | Notes |
| --- | --- | --- |
| `speed_mps` | float | `wheelspeed` (fallback `airspeed`) — feeds `ego.speed_mps`, TTC, speed plan |
| `steering_input` / `throttle_input` / `brake_input` | float | All three feed player-override detection (steer residual vs aligned `cmd.steer`, pedals one-sided) |
| `applied_seq` | int | Last cmd seq Lua applied |
| `applying` | bool | Lua currently holds the inputs |
| `mtime` | int | `os.time()`; Python uses the file mtime, fresh ≤ 1 s |

Retail (`capture_backend=window`): `actuator=cmd_json`; `cmd_applied` follows the ack. Boot line: `backend=window cams=1/8 path=retail (1 window capture; not 8; drive=gvd_cmd.json->mod Lua)`.

