# `gvd_state.json` fields (BeamNG path ribbon + nerd panel)

Path: product sandbox `gvd_state.json` (written by `python/run_vision.py`). **Not** `%USERPROFILE%\Documents\GVD`. **Not** OneDrive.

- Tech: `%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD\gvd_state.json`
- Drive / retail: `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD\gvd_state.json`
- `GVD_DOCS_DIR` override wins (full GVD root) for **Python writers**. Steam GELua does not inherit it — live Lua reads relative `Documents/GVD` via VFS / `FS:readFile` (the userfolder mapping of that sandbox). No absolute `io.open`.

| Field | Type | Notes |
| --- | --- | --- |
| `path_ego` | `[{x,y,z}, ...]` | Modular planner path, vehicle frame (+X right, +Y forward). The VISION ribbon ends at the last point. It is not repeated out to a draw distance. |
| `path_world` | optional same | If set, GELua draws this directly |
| `path_width` | m | Planner corridor width. VISION half-width is `path_width / 2` in meters at every station. |
| `path_e2e` | optional `[{x,y,z}]` | E2E polyline when the onnx exports one. Empty when the net only outputs steer/throttle/brake. |
| `path_conf` | 0–1 | Low → thinner/shorter/darker ribbon |
| `path_debug_preview` | bool | `false` only when corridor path came from lanes; geometric/steer fallback stays `true` |
| `gvd_show_path` | bool | Default true |
| `show_agent_ghosts` | bool | Default false; dim agent mode-0 only |
| `agents[]` | optional | `id`, `path_ego` for forecast mode 0 |
| `heartbeat_unix` | float | Age >1.5 s → ribbon fades 0.4 s then hides. A shorter hitch does not mark the link stale. |
| `policy` | string | `map-ai` draws amber (dimmer) instead of ice-blue |
| `ego.steer_deg` | float | Used for debug preview if path missing |

Lua: `gvd_main.drawPath` on `onPreRender` / `onDebugDraw`. Runs whenever the supervisor heartbeat is live (dimmer while disengaged). Ice underglow is engage-only. Caps: 40 ego segs, 8 agents × 10 segs. Label in console: GVD PATH. No DecalRoad / map edit.


| `capture_backend` | string | `beamngpy` / `window` / `stub` |
| `capture_note` | string | e.g. `retail: 1 window` |
| `rss_mb` | float | supervisor RSS; >12 GB is a bug |
| `cam_health.narrow` | enum | added in M1 8-cam set |
| `loop_hz` | float | honest supervisor loop EMA (never clamped to a fake ≥10) |
| `camera_hz` | float | unique GPU-frame EMA — new frames only, not cache re-shows / `main is not None` |
| `grab_ms` | float | camera grab wall-ms this tick |
| `grab_phase` | int | Hitch-wheel slot (`grab_i % wheel`, locked wheel 16). `-1` on retail window / stub (no wheel). |
| `grab_poll_free` | bool | This grab did not read a companion. Locked schedule: poll-free on slots 4 and 8–15 (main `stream_raw` only). Hitch slots 0, 1, 2, 3, 5, 6, 7 each read one companion with `stream_raw`. Yaml `update_s: -1` is that hitch. The sensor `requested_update_time` is 1 s, offscreen. Grab does not send `SendAdHocRequestCamera`. |
| `companion_inflight` | int | Ad-hoc companion renders still in flight. The live grab does not start those renders, so this stays 0. |
| `grab_read_blocked` | bool | This tick did not wait out a `stream_raw` or legacy `poll` that was still running after 50 ms. The last real frame stays painted. |
| `sensors_poll_ms` | float | Wall-ms of this tick's first `vehicle.sensors.poll`. 0 when no vehicle poll ran. |
| `poll_gps_ms` | float | Wall-ms of `GPS.poll` (PollGPSGE) when this tick sent it. 0 when GPS is absent or the sample was coalesced. |
| `poll_gps_sent` | bool | True only when this tick called `GPS.poll`. |
| `electrics_ms` | float | Wall-ms of the grab-loop `read_electrics` (snapshot reuse, or a miss-path poll). |
| `infer_ms` | float | perception tick ms |
| `viz_ms` | float | OpenCV stage ms |

Soft Esc reads the same tick from `gvd_state.json` and the `[GVD] seg` line. Poll-free `grab_ms` is the main `stream_raw` cost. The gap versus a hitch-phase `grab_ms` is the companion `stream_raw`. `companion_inflight` stays 0. `grab_read_blocked=1` means the tick moved on instead of waiting out a blocked read. `heartbeat_ms - grab_ms - infer_ms` is the rest of the tick before the heartbeat stamp, including the ego-poll tail (`sensors_poll_ms`, `poll_gps_ms`, `electrics_ms`). `camera_hz` stays the unique GPU-frame EMA. `rss_mb` is the supervisor RSS sample for a long-run sag check.


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
| `cmd_applied` | bool | True when BeamNGpy `vehicle.control` ran and the Tech steer-hold queue landed, **or** the mod acked this seq via `gvd_ego.json` (fresh, `applying`, `applied_seq` equal to `cmd_seq` or up to 5 behind — a higher seq from a previous supervisor is not an ack). A missing hold queue, a failed flag write, or a failed hold queue before the lock lands leaves this false and sends no `vehicle.control`. After the lock is up, a failed hold refresh still sends throttle and brake with steering omitted, and `cmd_applied` stays false. The override residual keeps the last steer that was actually queued. On Tech, `vehicle.control` returning does not mean the wheel left that steer in place; `tech_steer_hold` is the lock that does |
| `tech_steer_hold` | bool | Tech only. Written true immediately before every hold queue until the lock is up. The steady snapshot keeps it true while a hold is pending or the whitelist is up, and Python writes it false after a release queue is accepted. The mod leaves it set when a release queue is accepted during play. A later load with a live heartbeat arms the watch. A later load whose heartbeat is already stale replays the release once and then clears this flag so the latch does not stay owed. Unload retries the release and writes this true when every try fails and the state file can be read. `tech_steer_hold_vid` is the beamngpy vehicle name the hold was queued on. Lua resolves that name with `scenetree.findObject`, then `be:getObjectByID` when it is numeric, then the player vehicle. A player-only fallback still queues the release and keeps the latch until the named vehicle is released. Python writes this false after its own release is accepted, including quit. The mod clears the steering whitelist when this flag is false or the state heartbeat is stale, and on unload, and drops its latch after the named vehicle's release queue is accepted |
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
| `vehicle.sensors` | dict | `electrics`/`damage`/`gforces`/`gps` plus optional `lidar`/`radar`/`advanced_imu` → `ok`/`missing` |
| `sensors` | dict | Extra bus health: `imu` / `gps` / `lidar` / `radar` sources, `foxglove`, `drive_uses=vision`, `lidar_lua` bool. Never fed to the planner |
| `nav.mode` | string | `missing` (no GPS fix) / `hint` (Tech GPS or retail pose-derived lat/lon). Never `route` until a planner exists |
| `nav.drive_to_pin` | bool | Always `false` today — pin is a hint, not a route |
| `nav.gps` | `{lat,lon,x,y,ok}` or null | BeamNGpy GPS or retail `lua_pose` (world xy × `gps.ref_*`). Maps have no real-world lat/lon |
| `nav.pin` | `{lat,lon,name}` or null | Destination from `nav.pin_*` / `GVD_NAV_PIN_*`; empty until you drop a pin |
| `nav.range_m` / `bearing_deg` | float? | Haversine range; bearing clockwise from north |
| `nav.bearing_rel_deg` | float? | Pin vs heading (−180..180, + = right). Heading from GPS motion or world `dir` |

Also: `Documents/GVD/gvd_cmd.json` = `{steer,throttle,brake,seq,engaged,heartbeat_mtime,reason}` (see M6 below).
`config/tech.yaml` owns host/port/home, wait-for-vehicle, vehicle-data sensors, GPS origin, and the optional nav pin (RGB cameras stay in `cameras.yaml`). Optional LiDAR / radar / Foxglove live in `config/sensors.yaml` — Foxglove / future fusion only; the corridor planner ignores them. Snapshot: `Documents/GVD/gvd_sensors.json`. Optional retail ray sweep: `Documents/GVD/gvd_scan.json` when `sensors.lidar_lua` is true.


## M4 fields

| `last_clip_trigger` | string | `none` / `disengage` / `aeb_brake` / `near_miss_ttc` / `manual` / `smoke` |
| `last_clip_path` | string? | Last flushed clip directory under Documents/GVD/clips |
| `encode_backend` | string | `h264_qsv` / `libx264` / `h264_nvenc` / `none` |


## Dual-viz

| `path_world` | optional `[{x,y,z}]` | World path from BeamNGpy pose × path_ego (kinematics, not a map). Lua prefers this for 1:1 ribbon. |
| `show_agent_ghosts` | bool | Default true when `tracks_n>0`; in-game track hulls + OpenCV ghosts |
| `planner.cipv_id` | int? | Brighter ice / LEAD on that track |


## Path note

Python **writes** the **running product** sandbox (Tech `BeamNG.tech\current\Documents\GVD` vs Drive `BeamNG.drive\current\Documents\GVD` under `%LOCALAPPDATA%\BeamNG\`). `GVD_DOCS_DIR` is Python-only — Steam GELua does not inherit it. GELua **reads** relative `Documents/GVD` via VFS / `FS:readFile`. Never `%USERPROFILE%\Documents\GVD`. Never OneDrive. Never a bare `gvd_*.json` under userfolder `current\`. Never absolute `io.open` on the bus.

Both sides print every second: `python_bus=` (absolute) `lua_bus=` (resolved `Documents/GVD`) `product=drive|tech` `state_mtime` `engage` `seq`. If those folders are not the same, `link=MISMATCH` and actuation is refused. VISION paints only from that shared `gvd_state.json`; missing or the other product tree is a loud mismatch, not a fake corridor.

Tech vision does not start camera Hz until the wait gate passes: research port `:25252` LISTENING, the GVD mod under Tech `current\mods\unpacked\gvd`, a spawned vehicle, and a fresh `lua_bus` (handshake age < 1 s) with `buses_same(python_bus, lua_bus)` so `link=ok`. A missing or stale `gvd_link.json` / `gvd_ego.json` stays `link=MISMATCH`. Live `lua_bus` wins over a `GVD_DOCS_DIR` guess. The human one-starter is install-root `BeamNG.tech.exe -tcom -console -gfx dx11`. BeamNGpy launch uses that same root exe and `-gfx dx11`. A missing `tech.key` or user path does not open or close this gate. Attach bounds Hello with `socket_timeout` even when BeamNGpy has no such argument. A Hello timeout refuses once and is not reported as a missing vehicle. A listening `:25252` keeps `launch` false. The wait-gate line includes the lua folder path. The supervisor reuses that Hello session for cameras. A connect failure after the hold exits the supervisor. Esc or `q` in GVD VISION disconnects that socket only (`quit_on_close=false`); it does not quit BeamNG.tech.

| `python_bus` | string | Absolute folder Python wrote |
| `lua_bus` | string | Resolved `Documents/GVD` (product sandbox) |
| `product` | string | `drive` / `tech` |
| `bus_link` | string | `ok` / `MISMATCH` |

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

| `lanes_ext[]` | `{points:[{x,y}], kind, side, style, index}` | `kind`: `detected` (Hough saw the paint) / `predicted` (a sideways offset of a seen boundary; the live writer does not add one) / `stub` (`--smoke` only). `index` is `±1` for the ego pair. Every extra boundary the fit returned on that side shares `±2`. `style` stays `unknown` — nothing classifies solid vs dashed yet |
| `road_edges[]` | `{points:[{x,y}], kind, side}` | Kerb line just outside the outermost boundary. **Always `predicted`**: no kerb detector exists, this is the road edge implied by the lanes we can see |
| `signs[]` | `{cls,x,y,conf,state}` | Road furniture straight from the detector: `stop_sign` (COCO 11), `traffic_light` (9), `pole` (10 / 12 — hydrants and parking meters, drawn as short grey sticks). Never tracked and never offered to CIPV or AEB. `state` is `unknown` for lights — no lamp-colour classifier, so the OpenCV stage draws all three lamps as empty rings |
| `agents[]` | `{id, path_ego:[{x,y}]}` | Mode-0 constant-yaw-rate forecast fan per moving track, same toy math as the OpenCV view |

### Scene cues derived in the OpenCV stage

These come from state the stage already has — no extra fields, and each one needs its own signal:

| Cue | Condition |
| --- | --- |
| Road user lit ice-blue | inside the planner's corridor: `\|x\|` within `path_width/2 + 0.45` of a `path_ego` point at that range. No corridor → nothing highlighted |
| Road user red + `BRAKE` tag | it is the CIPV **and** (`planner.aeb == brake` or `planner.ttc_lead < 1.5`) |
| Ribbon shade | accelerating (`target_v` above speed) brightest, coasting normal, slowing dimmer, planned stop faintest |
| Hard stop bar across the ribbon | planner halted (`target_v <= 0.2` or AEB brake) **and** a CIPV exists — the bar sits at the lead, because that is the constraint being stopped for. No CIPV means no stopping point we can honestly claim, so no bar |
| Traffic light tinted ice-blue | `signs[].relevant == true`. **Nothing sets it today** — the stack has no route-relevance signal, so every light renders muted |

The only filled surface in the scene is the ego corridor; lane paint and kerbs are thin vector polylines, and the sky is void — there is no backdrop.

The live writer keeps every boundary the fit returned and tags them `detected`. A wider road is those extra lines, not a sideways copy of the ego pair. With no fit there is no lane paint and no road edge. The kerb is predicted, 0.4 m outside the outermost real line. Sign positions inherit `project_box_to_ego`'s crude pinhole estimate, and sign/light heights in the scene are a drawing convention, not a measurement. The stage draws detected geometry solid and predicted geometry dim + dashed, skips `kind=stub` except `--smoke` (`viz_smoke`), and a live two-line road prints e.g. `lanes 2 seen · edges pred · 2 signs` under the window. Loop under 8 Hz drops forecast fans, signs and the `cam_main` PIP.

## In-game app bus

`gvd/main.lua` pushes `guihooks.trigger('gvdUi', ...)` on a single 10 Hz `tickPush` path (same period as the `egoFb` poll). Wheel/pedal bars ride that path from `egoFb`; `onEgoFeedback` does not push a second HUD stream. Payload still includes capped scene geometry: `path` ≤28 points, `tracks` ≤12 (`id,cls,x,y,yaw,v,lead`), `lanes` ≤6×12 (`pts,kind,side,style,idx`), `edges` ≤2×12, `signs` ≤8, `fans` ≤6×8, all ego frame and rounded to 2 dp. `link` is `live` / `stale` / `none` from the heartbeat age, or `mismatch` when `python_bus` and `lua_bus` are not the same folder (refuse actuation; do not guess). The app can show the dead-man without a second file bus. `tag` is the shared glance word: `MISMATCH` when the buses differ; `HOLD` while engaged and the link is not live, or `cmd_reason` is `preview_blocked` / `heartbeat_stale` / `veto:*`, or AEB is `brake`/`warn`; `DRIVE` only while engaged and not holding and (Lua `applying` or Tech `actuator=beamngpy` with `cmd_applied`); a retail `cmd_json_pending` write is `ON`, not `DRIVE`; `ON` while engaged and waiting; `OFF` when disengaged. `veto_reason=e2e_stub` is a HOLD. The full app maps that to DISENGAGED / ENGAGED / HOLD / DRIVE / MISMATCH. `gvdStrip` gets the same `tag` plus `link` so it cannot say DRIVE during a brake hold. The OpenCV title bar uses the same word.

Live wheel / pedal HUD fields are the same retail Direct Drive echo already written to `gvd_ego.json` (`M.onEgoFeedback` / `egoFb`). No parallel input stack.

| `steerInput` / `throttleInput` / `brakeInput` | float? | Applied electrics (`steering_input` / `throttle_input` / `brake_input`) |
| `playerDevice` | bool? | True when vehicle Lua saw a non-`gvd` `input.lastInputs` source |
| `playerSteer` / `playerThrottle` / `playerBrake` | float? | Strongest non-`gvd` lastInputs axis |

| `viz_window` | bool | Python `--viz` OpenCV window exists |
| `viz_screen` | string | Monitor the window was last placed on |
| `viz_note` | string | e.g. `screen 2 (non-primary)` / `auto→primary (second screen not found …)` |


## M5 fields

| `policy` | string | `modular` (default; safety supervisor) / `e2e` / `shadow` |
| `shadow.steer` | float | E2E proposed steer [-1,1]; written every tick even when disengaged |
| `shadow.throttle` | float | E2E proposed throttle [0,1] |
| `shadow.brake` | float | E2E proposed brake [0,1] |
| `e2e_ok` | bool | False when modular vetoes E2E (low lane_conf / heartbeat / disagreement / forward fail) |
| `veto_reason` | string | `none` / `aeb_brake` / `aeb_warn` / `low_lane_conf` / `low_path_conf` / `heartbeat_stale` / `disagreement` / `e2e_forward_fail` / `e2e_stub` / `preview_blocked` |
| `e2e_backend` | string | `stub` / `onnx` |

A measured `loop_hz` or `camera_hz` under `min_accept_hz` is not an Engage drop and does not replace the plan with a brake hold. A hitch that used to mark the link stale for a split second was the HOLD flash. CAMS blit dropping remains 8 Hz. The narrow-far hitch remains 10 Hz. The command dead-man is still `CMD_STALE_S` 0.35 s.

Perception always runs (loaded detector, lanes, corridor planner, other-vehicle tracks, E2E shadow). Path ribbon / GVD VISION overlays stay up. The VISION ice ribbon is `path_width / 2` meters each side and only as long as the path being painted. Modular, or e2e/shadow held by a veto (including `e2e_stub`), paints `path_ego`. `policy=e2e` with `e2e_backend=onnx` and `veto_reason=none` paints `path_e2e` or a steer integral of that same length. Shadow keeps the modular ribbon and may add a thinner ghost of the other path when the clean cabin (key 0) is off. Actuators only when engaged **and** modular OK. Shadow mode computes both intents; default apply path stays modular. `e2e_stub` (no trained onnx, `e2e_backend=stub`) holds the brake and stays engaged on `--policy e2e`. Shadow does not treat that stub as a disagreement. The command dead-man is unchanged (`CMD_STALE_S` 0.35 s, `CMD_DEAD_S` 1.0 s). The link/ribbon stale window is 1.5 s, not 0.35 s, so one grab hitch does not mark the link stale. Detector default is shipped `models/yolov8n.onnx`. No E2E checkpoint in git (`models/e2e_current.onnx` still gitignored).


## M6 — `gvd_engage.json` contract (retail package)

Path: `Documents/GVD/gvd_engage.json`, shared by GELua and Python.

| Writer | Payload | When |
| --- | --- | --- |
| Lua (`gvd_main.writeEngageFile`) | `{"engaged":true\|false,"mtime":<os.time() int>,"disengage_reason":"<why>"}` | Alt+G / GVD app button, `player_steer` / `player_brake` / `player_throttle`, `command_stream_dead`, `extension_unloaded`. This is the only writer of `engaged:true`. |
| Lua (`M.onUpdate`, every 0.5 s) | same shape, `engaged:true`, fresh `mtime` | While the in-memory latch is on. Does **not** bump `lastEngageWriteUnix`. Refuses to write `true` over a supervisor `false` whose `mtime` ≥ that stamp. |
| Python (`write_engage_flag`) | `{"engaged": false, "mtime": <time.time() float>, "disengage_reason": "<why>"}` | `player_steer` / `player_brake` / `player_throttle` / modular veto / stale heartbeat / `finally` on exit. Python does not write `engaged:true`. |
| Ship bot (`scripts/bot_engage.py`) | `gvd_bot_engage.json` `{"engaged": true\|false, "mtime": <time.time()>}` | `on` engages the Python supervisor without a keypress or a focused window. `off` clears it. A new `true` counts only while `mtime` is about 2.5 s fresh, then the latch stays until `off`, a driver override, or a veto that drops Engage. A stale leftover `true` does not engage. |

Python reads the file every tick. `engaged:false` is always off. `engaged:true` counts only when JSON `mtime` age is in `[-1.0, 2.5]` seconds (`ENGAGE_FRESH_S`). Missing `mtime`, or a stamp left from a crashed session, is not engage — Tech must not call `vehicle.control` from it. A missing file returns the caller's default (the supervisor default is false). Lua polls the file every 0.1 s **only while engaged** and adopts `engaged=false` when the file says so and `mtime` ≥ Lua's own last toggle stamp; it logs `[GVD] DISENGAGED by supervisor (<disengage_reason>)`, releases the vehicle inputs and refreshes the HUD/UI app. A file saying `true` never engages Lua — engage always starts in-game (Alt+G or the app button). Both sides write `false` on `player_steer` / `player_brake` / `player_throttle` (sticky, whichever sees it first) and Lua writes `false` when its dead-man fires. The file is the durable record of *why*: later supervisor ticks only see `engaged=false` and write the generic `not_engaged` into `disengage_reason`, so the mod keeps the reason it adopted for the HUD.

## Player override (force-feedback residual)

`config/control.yaml` `override:` owns the research-pin thresholds; the supervisor mirrors them into `override_cfg` every tick so `gvd_main` runs the same numbers without parsing YAML. Without a player device the signal is `|steering_input − cmd.steer|` against the command in force when the echo was sampled. On retail while the secondary Direct Drive source is locked, electrics are GVD's command and override reads `player_*` lastInputs as an absolute axis. On Tech while `tech_steer_hold` is set, electrics are the gvd steer and the 1.7.4 residual reads `player_steering` (the physical wheel). A missing player echo while the hold is on is not treated as the command. Once the detector is armed and the hold is on, rest is seeded at 0 so the next wheel sample, including a grab already in hand, is not stored as the resting angle. With the lock on, the residual uses `steer_center_deadband` 0.15. A wheel between 0 and the command is residual 0. Past the command on that side the residual is `(w − c)`. On the opposite side of 0, and when `|c| < 0.02`, the residual is `sign(w) * max(0, |w| − 0.15)`. A wheel at 0 against command 0.35 stays engaged. A wheel resting at 0.10 against a straight command stays engaged. A held 0.25 in any direction drops Engage after the dwell. A fresh numeric `player_steering` counts as that wheel even when `player_device` is false. The field absent, or `gvd_ego.json` stale, for more than 2 seconds while the hold is on drops Engage with `tech_wheel_blind`. Live FFB is **UNPROVEN**. `CMD_DEAD_S` is not part of this.

| Field | Type | Notes |
| --- | --- | --- |
| `override_cfg.steer_enter` / `steer_exit` | float | 0.08 / 0.04. Dwell charges above enter, discharges below exit, frozen between |
| `override_cfg.steer_hold_ms` | float | 200. How long the player has to keep pushing one way |
| `override_cfg.steer_spike` | float | 0.20. Sample-to-sample jump past this is mechanical: the EMA holds |
| `override_cfg.lpf_tau_ms` | float | 80. EMA time constant, on the residual only |
| `override_cfg.steer_center_deadband` | float | 0.15. Tech steer hold only. Straight/centre deadband: residual `sign(w) * max(0, |w| − D)` when the command is straight (`|c| < 0.02`) or the wheel is on the opposite side of 0. Past the command the residual is `(w − c)`. Retail Lua stores the pin and does not read it |
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

Lua (`gvd_main.applyCmdJson`, 20 Hz): `input.event('steering', s, 2, 900, 0, nil, 'gvd')`; `input.event('throttle', t, 2, 0, 0, nil, 'gvd')`; `input.event('brake', b, 2, 0, 0, nil, 'gvd')`; `input.event('parkingbrake', 0, 2, 0, 0, nil, 'gvd')`; `input.event('clutch', 0, 2, 0, 0, nil, 'gvd')` on `be:getPlayerVehicle(0)` via `queueLuaCommand` (FILTER_DIRECT; steering angle 900 marks Direct Drive, lockType 0 so -1..1 is already fraction of vehicle lock; pedal angle 0 is unused); `input.setAllowedInputSource(..., 'gvd', true)` and `('local', false)` on steering/throttle/brake/parkingbrake/clutch while applying so a connected device cannot overwrite; `drivetrain.setShifterMode(2)` once (mode 2 is arcade). On release: zeros on source `gvd`, then `setAllowedInputSource(..., nil)` to give the player device back. No new seq for `CMD_STALE_S` (0.35 s) → steer 0 / throttle 0 / brake 1 hold; after `CMD_DEAD_S` (1.0 s) → release (all 0), `engaged=false`, `gvd_engage.json` false. Any disengage (Alt+G, supervisor false, unload) sends one release and stops applying. `cmd.engaged=false` → release immediately (no brake tap on the player).

`Documents/GVD/gvd_ego.json` — written by Lua at ~10 Hz while the supervisor's state heartbeat is alive (vehicle Lua `electrics.values` → `obj:queueGameEngineLua` → `gvd_main.onEgoFeedback`):

| Field | Type | Notes |
| --- | --- | --- |
| `speed_mps` | float | `wheelspeed` (fallback `airspeed`) — feeds `ego.speed_mps`, TTC, speed plan |
| `steering_input` / `throttle_input` / `brake_input` | float | Applied electrics (GVD's command while the Direct Drive source is locked). Fallback override signal when `player_device` is false |
| `player_device` | bool | True when vehicle Lua saw a non-`gvd` `input.lastInputs` source (physical wheel/pad/keys). Sources `adas` / `beamngpy` / `tech` are not a player device |
| `player_steering` / `player_throttle` / `player_brake` | float | Strongest non-`gvd` / `adas` / `beamngpy` / `tech` lastInputs axis. Recorded even when that source is blocked. Retail override uses these as an absolute axis when `player_device` is true (centered wheel is 0, not residual vs `cmd.steer`). Tech steer hold uses a fresh numeric `player_steering` as the wheel even when `player_device` is false (centre deadband 0.15; a held rotation of 0.25 drops Engage). The blind timer starts when that field is absent, JSON null (`nPlayer == 0`), or the file is stale. A numeric 0 is a centered player device. Pedals on Tech stay the electrics residual; they are not in the steering whitelist |
| `applied_seq` | int | Last cmd seq Lua applied. Python claims `cmd_applied` only when `0 <= cmd_seq - applied_seq <= 5` |
| `applying` | bool | Lua currently holds the inputs |
| `mtime` | int | `os.time()`; Python uses the file mtime, fresh ≤ 1 s |
| `gx` / `gy` / `gz` / `yaw_rate` | float? | Vehicle `sensors.gx2` + `obj:getYawAngularVelocity` (IMU extras; planner still vision-only) |
| `pos` / `dir` | `{x,y,z}`? | Player vehicle world pose from GE (`getPosition` / `getDirectionVector`); GPS lat/lon is derived from this × `gps.ref_*` |

Retail (`capture_backend=window`): `actuator=cmd_json`; `cmd_applied` follows the ack. Boot line: `backend=window cams=1/8 path=retail (1 window capture; not 8; drive=gvd_cmd.json->mod Lua secondary Direct Drive wheel+pedals)`.

Tech (`actuator=beamngpy`): throttle, brake, parking brake, and clutch stay on `vehicle.control` (arcade, no gear field). `steering` is omitted from that message. A steering field is pad filter 1 on source `local` and overwrites `lastInputs.local.steering` even when the whitelist keeps it off the hydros, so a held wheel grab would read back as the command. While engaged, Python queues `tech_steer_hold_lua`: `setAllowedInputSource('steering','gvd',true)`, `('steering','local',false)`, then `input.event('steering', s, 2, 900, 0, nil, 'gvd')`. The hold is queued before `vehicle.control`, and `tech_steer_hold` is written true immediately before every such queue until the lock is up. The wheel angle remains in `lastInputs` / `player_steering` because the wheel is onChange and Control no longer writes the slot. A held 0.25 past the command, opposite the command, or against a straight command is `player_steer` and drops Engage. A wheel at 0 against command 0.35, and a wheel at 0.10 against a straight command, stay engaged. Disengage or shutdown queues `TECH_STEER_RELEASE_LUA`: `setAllowedInputSource('steering',nil)`, then `input.event` of `lastInputs['local'].steering` on source `local`, so a wheel already held drives without waiting for the next onChange. `lastInputs` normally stores only the axis value. A table value may carry `value`, `angle`, `lockType`, and `filter`; a number may have sibling `steeringAngle` or `angle`, `steeringLockType` or `lockType`, and `steeringFilter` or `filter`. Otherwise the replay is Direct Drive filter 2, angle 900, lock type 0, which does not reproduce a non-direct player binding. The mod queues that same release when `tech_steer_hold` is false, when the state heartbeat is stale past 2 s, or on extension unload (a few tries). The target is `tech_steer_hold_vid` when that name was published: `scenetree.findObject`, then `be:getObjectByID` when the id is numeric, then the player vehicle. A player-only fallback still releases, and the latch stays until the named vehicle is released. With no id, it uses the last applied vehicle or the player vehicle. It clears its latch only after the queue is accepted. During play that accept leaves the disk flag set. When unload's release was accepted and a later load finds the heartbeat already stale, that load replays the release once and then clears `tech_steer_hold`. A live heartbeat on load only arms the watch. A failed unload leaves `tech_steer_hold` true on disk when the state file can be read. A failed hold queue or a failed flag write before the lock lands leaves `cmd_applied` false and sends no `vehicle.control`. After the lock is up, a failed hold refresh still sends throttle and brake with steering omitted, and `cmd_applied` stays false. The override residual keeps the last steer that was actually queued. Log lines: `[GVD] tech steer hold <steer> (player wheel locked out; source=gvd)`, `[GVD] tech steer release (player wheel restored)` (Python), `[GVD] tech steer grab blind: player_steering missing while hold is on` (once per lock, while the hold is on and the wheel echo is missing or the ego file is stale), `[GVD] DISENGAGED: tech_wheel_blind`, and `[GVD] tech steer release (supervisor heartbeat stale|hold flag clear|extension unloaded|load replay)` (Lua).

## Debug knobs (`debug`)

Written every tick from the GVD VISION nerd **DRIVE** / **VIZ** tabs (`python/runtime/debug_opts.py`). Defaults match stock supervisor behaviour (preview blocked, Alt+G engage, product lexicon). These mutate the command **this tick**; they do not invent cameras or feed LiDAR into the planner.

| Field | Notes |
| --- | --- |
| `allow_preview` | Same gate as `--allow-preview-drive` |
| `force_engage` | Debug-only; Python treats the session as engaged without Alt+G. Does not write `gvd_engage.json` true (Lua still only engages in-game) |
| `ignore_override` | Wheel/pedals still measured; they do not disengage |
| `ignore_veto_disengage` | Modular veto still holds/brakes; engage stays. Heartbeat dead-man still disengages |
| `policy` | `session` (use launch/in-game policy) or `modular` / `e2e` / `shadow` |
| `steer_on` / `throttle_on` / `brake_on` | Mute an actuator channel |
| `steer_gain` / `max_steer` / `invert_steer` | Post-plan steer rewrite |
| `hold_brake` / `freeze_cmd` | Hold `brake=1` or repeat last command |
| `aeb_on` / `aeb_ttc` / `cipv_on` | Rewrite planner AEB/CIPV before `shadow_tick` |
| `speed_cap` / `cruise_mps` / `corridor_width` / `lane_conf_min` | Speed and veto knobs |
| `lanes_on` / `detector_on` | Drop lane paint (preview path) or YOLO tracks |
| `detector_id` / `e2e_id` | Nerd **MODEL** tab. `auto` uses shipped `yolov8n.onnx` when present. `synthetic` / `empty` plus any extra `models/yolov8*.onnx\|pt` or `e2e*.onnx`. Unknown ids are not listed. Lanes are Hough (no net). |
| `viz_*` | OpenCV overlay layers. `viz_occ` occupancy is **from tracks**, not a learned grid. Keys `1–5` map to occ / detector boxes / lane polynomials / camera FOV / planner samples |

`occupancy` in state stays `null` (no occupancy net). The VIZ overlay is drawn in OpenCV only.

