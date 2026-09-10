# Grok Vision Drive (GVD)

Public **MIT** BeamNG.drive / BeamNG.tech **camera-only** self-driving **toy** inspired by Tesla FSD *software strategies* — not a clone, not a Tesla product, and **not** real-vehicle control.

Entertainment only. Never use this stack to control a physical car.

## Honesty (M0)

- **Vision only** at inference: RGB cameras + ego kinematics + GPS as a **nav hint**. Optional pin (`nav.pin_lat` / `nav.pin_lon` in `config/tech.yaml`, or `GVD_NAV_PIN_LAT` / `GVD_NAV_PIN_LON`) is **not a route** — `nav.mode=hint`, `drive_to_pin=false`, and the corridor planner ignores the pin.
- **Optional extras** (`config/sensors.yaml`): IMU + GPS on by default; LiDAR / radar / lidar_lua / Foxglove **off**. They are a **sensor bus** (Foxglove / future fusion only) — they never feed `ModularPerception` or the planner. Ultrasonic / IdealRadar stay refused. See [Extra sensors / Foxglove](#extra-sensors--foxglove).
- **Tech path** (preferred): BeamNGpy Camera sensors, 8-cam rig, BeamNGpy `vehicle.control`. **Retail path**: window-capture fallback for a **single** main view (`cams=1/8`) — do not pretend that is 8 cameras. Retail **drives** through the mod: Python writes `Documents/GVD/gvd_cmd.json`, GELua applies it with the same vehicle-Lua `input.event` calls BeamNG's own AI uses (no DLL, no hooks), and echoes speed/inputs back via `gvd_ego.json`.
- Default GPU profile: **NVIDIA GTX 1080 Ti (11 GB)**. Live stack must share the card with BeamNG.
- Not Tesla FSD. No Tesla logos. Repository title stays free of “FSD”. “Vision Drive” / “camera autopilot toy” are fine.
- Strategy analogies in docs mean: Tesla idea → **our toy version does X**.

Credits: [VisionPilot](https://github.com/visionpilot-project/VisionPilot), BeamNG / BeamNGpy, Udacity/Aly lane pipelines, Ultralytics if YOLO is used later.

## Install

Get `gvd-retail-<version>.zip` from GitHub Releases (or build it — see [Release zip](#release-zip-m6)), extract it anywhere, then:

1. Double-click `install.bat` (prefers `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods` on 0.38+; else newest versioned folder)
2. Fully quit and restart BeamNG
3. Enable Grok Vision Drive in Mod Manager if needed (Alt+G engage)

Optional: double-click `play_gvd.bat` to start the Python supervisor (retail `window` backend) + Steam BeamNG (`284160`). Uninstall: `uninstall.bat` (removes only `mods\unpacked\gvd` and `mods\gvd.zip`).

Windows first. No admin. Never writes into `Steam\steamapps\common\BeamNG.drive`. “Inject” here means copy into the correct user mods folder so BeamNG loads the mod itself — no DLLs, hooks, or process injection.

## Player guide (retail, M6)

What you get on **retail BeamNG.drive**: the in-game **GVD** app (Engage / Disengage, show path, show ghosts, policy, screen pick), Alt+G engage, the ice-blue path ribbon on the road, the **GVD VISION** second window (void-stage lexicon: multi-lane fan, ice corridor, CIPV boxes, curbs, signs), clips in `Documents\GVD\clips`, and **driving**: while engaged, GVD steers, throttles and brakes the sim car along the lane corridor (HUD reads `modular|DRIVE`).

What retail is **not**: eight cameras. It captures **one** window (the main view); the other seven stay `missing` in the nerd panel and the boot line says `cams=1/8 path=retail`. It drives through the **mod's Lua** (`gvd_cmd.json` → vehicle `input.event`), not through BeamNGpy. Eight cameras and BeamNGpy direct control need BeamNG.**tech** (`GVD_BACKEND=beamngpy`, `GVD_BEAMNG=1`) — the preferred path.

1. Extract the zip, double-click `install.bat`, restart BeamNG, enable **Grok Vision Drive** in Mod Manager.
2. In BeamNG open **Apps** → add **GVD** (and optionally **GVD Strip**).
3. Install Python 3 (python.org, tick *Add to PATH*), then `pip install -r requirements-retail.txt` — or let `play_gvd.bat` offer to do it when packages are missing.
4. Double-click `play_gvd.bat`. It opens the supervisor console (`cmd /k`, so errors stay readable), the **GVD VISION** window, and BeamNG via Steam. Window capture locks onto a visible window titled **BeamNG** as soon as it appears; if none is found it grabs the primary monitor, so run BeamNG fullscreen in that case (the nerd panel `capture` line tells you which).
5. Drive onto a road with visible lane lines, then press **Alt+G** (fallback **Ctrl+Alt+G**) or the app's **Engage**. **Alt+A** stays BeamNG's stock range display (`toggleRangeStatus`); GVD does not steal it. The ribbon appears and the HUD strip reads `modular|ON`; as soon as perception sees both lane lines (`path_debug_preview=false`) GVD takes the wheel and the strip switches to `modular|DRIVE`. Until then it holds the brake (`preview_blocked`). Take over any time by steering against it (`player_steer`), touching a pedal (`player_brake` / `player_throttle`), or pressing Alt+G — sticky until the next Alt+G.

**Force-feedback wheels.** A wheel is welcome. Override detection looks at `|steering_input − cmd.steer|`, not the absolute angle, then spike-rejects and EMA-filters that residual so self-aligning torque, spring centering and kicks over bumps no longer disengage GVD. A real pull held ~200 ms still wins, and so does any real brake or throttle press. Tune it in `config\control.yaml` under `override:`. Live FFB behaviour is **UNPROVEN**.

Keys in **GVD VISION**: `V` nerd panel, `D` DRIVE tab (gates / actuators / AEB), `G` VIZ tab (overlay layers), `[` `]` / `?` cycle tabs, `0` clean cabin, `1–5` debug layers (occupancy / detector boxes / lane polynomials / camera FOV / planner samples), `T` chase↔BEV, `C` manual clip, `q` quit. Click the nerd tabs and `+`/`−` to change knobs; they write the command this tick.

**How retail drives.** Every tick the supervisor writes `Documents\GVD\gvd_cmd.json` (`steer, throttle, brake, seq, engaged, heartbeat_mtime`). The mod polls it at 20 Hz and, while **both** sides are engaged, feeds the player vehicle with `input.event('steering', s, 1)` / `input.event('throttle'|'brake', v, 2)` — the calls BeamNG's own AI and BeamNGpy use (`+steer` = right, pad-smoothed steering, direct pedals) — and switches the gearbox to arcade once. Vehicle Lua echoes `wheelspeed`, `steering_input`, `throttle_input`, `brake_input` and the applied `seq` back through `gvd_ego.json`, which gives the planner real ego speed (closed-loop throttle, TTC/AEB) and lets `cmd_applied` be `true` only on a fresh Lua ack (`cmd_reason`: `cmd_json_applied` / `cmd_json_pending` / `cmd_json_idle`).

**Heartbeat dead-man and disengage.** The supervisor stamps `heartbeat_mtime` into `gvd_state.json` every tick; if it stalls for more than 0.35 s the ribbon fades over 0.4 s and hides, returning when the beat returns. On the drive side the mod only applies commands while `seq` keeps advancing: `CMD_STALE_S` 0.35 s without a new command → steering straight, throttle 0, **brake hold**; `CMD_DEAD_S` 1.0 s → inputs released, auto-disengage, `gvd_engage.json` set to `false`. You are disengaged when:

- you press Alt+G / **Disengage** — the mod sends steering/throttle/brake 0 once and stops touching the car;
- you steer against it (`player_steer`) or use a pedal (`player_brake` / `player_throttle`) — sticky until the next Alt+G, judged by the mod **and** the supervisor, whichever sees it first;
- the supervisor quits (`q`, window closed, Ctrl+C): its `finally` block writes a stop and `engaged=false`, the mod releases the car;
- the modular supervisor vetoes an E2E/shadow policy (`--policy e2e|shadow`: low `lane_conf` / `path_conf`, disagreement, AEB). The default `modular` policy disengages only via Alt+G, override, supervisor exit or the dead-man.

The mod reads `engaged=false` back and flips the HUD to **OFF** within ~0.1 s (it never turns engage *on* from a file — engage always starts in-game). Re-engage with Alt+G. A clip is flushed on every disengage, AEB brake and near-miss (`Documents\GVD\clips\clip_<time>_<trigger>`).

Files: `Documents\GVD\{gvd_state.json, gvd_engage.json, gvd_cmd.json, gvd_ego.json, gvd_ui_prefs.json, clips\, python\, config\}`. `uninstall.bat` removes the mod only; delete `Documents\GVD` yourself if you want a clean slate. Nothing here talks to a real car and nothing carries Tesla / “FSD” branding.

Troubleshooting: mod missing in Mod Manager → run `install.bat` again and check the path it prints. “Python not found” → install Python 3 with *Add to PATH*. “Missing Python packages” → answer `Y` or run `pip install -r requirements-retail.txt`. No `cam_main` PIP / `capture_note` says *fullscreen/monitor* → make the BeamNG window visible with **BeamNG** in its title, or run fullscreen; `window via unavailable` → `pip install mss`. HUD stays `ON`, never `DRIVE` → GVD only drives once it sees both lane lines (nerd panel `lane` conf, `preview=`); `--allow-preview-drive` follows the steer-preview path instead (debug only). `DRIVE` but the car does nothing → nerd panel `cmd` line: `cmd_json_pending` means the mod is not acking (mod not enabled, no player vehicle, or the supervisor is engaged while the game is not). Boot refuses (`REFUSE: dGPU VRAM … < 10 GB while BeamNG is up`) → `play_gvd.bat --vision-only` (drops the GPU guard, nothing else).

## Tech path (when you have BeamNG.tech)

There is still no retail mod that creates those eight cameras. Once `tech.key` is in the **Tech install directory** (not the user folder), GVD already knows how to attach them and pull vehicle data.

1. `install.bat` copies the Lua mod into Drive `current\mods` and, if present, `%LOCALAPPDATA%\BeamNG.tech\current\mods`.
2. `pip install -r requirements-beamng.txt` (BeamNGpy version must match Tech: **0.38 → 1.35.x**, **0.39 → 1.36**).
3. Set `BNG_HOME` to the Tech install folder. Edit `config\tech.yaml` for host/port/`wait_vehicle_s` if needed.
4. Start BeamNG.tech, spawn a vehicle (ETK800 is the camera-draft car), enable GVD in Mod Manager.
5. Double-click `play_gvd_tech.bat` (or `python python/run_vision.py --backend beamngpy --viz`). That sets `GVD_BEAMNG=1`, attaches the 8 RGB cameras from `cameras.yaml` (GVD frame converted to BeamNG vehicle space), **Electrics / Damage / GForces** plus pose, and a **GPS nav hint** (lat/lon). Optional destination: set `nav.pin_lat` / `nav.pin_lon` in `config/tech.yaml` (or `GVD_NAV_PIN_LAT` / `GVD_NAV_PIN_LON`) for range and bearing. Pin is **not a route** — the corridor planner ignores the pin. Optional LiDAR / radar / AdvancedIMU: opt-in in `config/sensors.yaml` (see [Extra sensors / Foxglove](#extra-sensors--foxglove)).
6. Probe without driving: `python python/run_vision.py --tech-probe`.
7. Engage is still Alt+G. Drive is BeamNGpy `vehicle.control` on the player vehicle (`actuator=beamngpy`). Ego speed/steer/pedals come from Electrics; the world ribbon uses pose × `path_ego`.

Live Tech attach is **UNPROVEN** until a Windows smoke. `--backend auto` does **not** pick Tech just because beamngpy is installed — only `GVD_BEAMNG=1` or `--backend beamngpy`.

## Extra sensors / Foxglove

Optional bus in `config/sensors.yaml`. Defaults: IMU + GPS **on**; `lidar` / `radar` / `lidar_lua` / `foxglove.enabled` **off**. Env: `GVD_LIDAR=1`, `GVD_RADAR=1`, `GVD_LIDAR_LUA=1`, `GVD_FOXGLOVE=1`. Extras are Foxglove / future fusion only — they never feed `ModularPerception` or the planner. Ultrasonic / IdealRadar stay refused.

- **Retail** (no BeamNGpy): IMU + world pose from Lua `gvd_ego.json`. GPS derived from world xy × `gps.ref_*` (same sphere as Tech). Optional coarse Lua ray sweep `Documents/GVD/gvd_scan.json` if `lidar_lua`. Radar has no retail source.
- **Tech**: optional BeamNGpy Lidar / Radar attach. GPS already exists. AdvancedIMU only if `advanced_imu: true` (default IMU is GForces / Lua).
- **Foxglove**: `pip install -r requirements-foxglove.txt`, then `foxglove.enabled: true` / `GVD_FOXGLOVE=1` / `--foxglove`. Connect the Foxglove app to `ws://127.0.0.1:8765`. Missing SDK → skip. Snapshot always at `Documents/GVD/gvd_sensors.json`. **Not in the retail zip.**

## Release zip (M6)

Windows: double-click `scripts\make_release_zip.bat`. Anywhere: `python scripts/make_release_zip.py`. Output: `dist\gvd-retail-<version>.zip` (gitignored), `<version>` = exact git tag if any, else `m6-<sha>[-dirty]`; override with `--version v0.6.0`, `--out path`, `--flat` (no top-level folder), `--list` (manifest only).

Packs: `install.bat`, `uninstall.bat`, `play_gvd.bat`, `beamng_mod/`, `python/` (retail runtime), `config/` (including `sensors.yaml`), `requirements.txt` + `requirements-retail.txt`, `LICENSE`, `README.md`, `docs/*.md`, `models/.gitkeep`, plus a generated `VERSION.txt` (version, build time, git sha, the retail honesty lines). Excludes `data/clips/`, weights (`*.onnx *.pt *.pth *.bin *.safetensors`), `.git`, `scripts/` (tests + this tool), `__pycache__`, `dist/`, and `requirements-foxglove.txt`. `*.bat` are written CRLF. After writing, the script re-opens the zip, refuses forbidden members and missing must-haves (mod entry point, `run_vision.py`, launchers), and exits non-zero on any problem. Attach the zip to a GitHub Release. Offline check: `PYTHONPATH=. python scripts/test_m6_retail.py`.

## Layout

```
install.bat / uninstall.bat / play_gvd.bat / play_gvd_tech.bat
beamng_mod/          → copied to %LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd
python/              → also copied to %USERPROFILE%\Documents\GVD\python
config/              → also copied to %USERPROFILE%\Documents\GVD\config
config/sensors.yaml  → optional IMU / GPS / LiDAR / radar / Foxglove bus (planner ignores extras)
requirements-retail.txt   → 1-cam window runtime (numpy, OpenCV, PyYAML, mss; no beamngpy)
requirements-foxglove.txt  → optional Foxglove SDK (not in the retail zip)
scripts/make_release_zip.py / .bat → dist/gvd-retail-<version>.zip (not shipped)
```

After install, unpacked tree:

```
mods\unpacked\gvd\
  scripts\gvd\modScript.lua
  lua\ge\extensions\gvd\main.lua
  ui\modules\apps\GVD\              # Engage HUD app
  lua\ge\extensions\core\input\actions\gvd.json
  settings\inputmaps\keyboardGvd.json
```

## In-game UI app

After install, start a level, open the **Apps** editor (Esc → *UI Apps*, or the app-layout button on the HUD), drag **GVD** in from the app list, size it, then **Save layout**. Source: `beamng_mod/ui/modules/apps/GVD/`.

The in-game app is **Engage / Disengage + settings only**. The rich VISION lexicon (void stage, multi-lane fan, ice corridor, CIPV boxes, warm curbs, sign/light glyphs) is drawn on the **GVD VISION** OpenCV second screen from the same `Documents/GVD/gvd_state.json`. See [`docs/gvd_state_schema.md`](docs/gvd_state_schema.md#viz-road-model) for what is measured vs inferred. Controls:

- **Engage / Disengage** — the same toggle as Alt+G. The state strip reads `DRIVE` (the mod is feeding the player vehicle from `gvd_cmd.json`, or BeamNGpy is applying on Tech — steer to take over), `ENGAGED` (armed, nothing applied yet), `HOLD` (heartbeat stale → dead-man, inputs released) or `DISENGAGED` with the last reason.
- **Path / Ghosts** — writes `gvd_ui_prefs.json`, so the world ribbon and the OpenCV window follow.
- **Policy** `modular | e2e | shadow` — shows what the supervisor is actually running and requests a change for the running session; the modular veto is unchanged and `--policy` still wins at launch.
- **Sensing** — capture backend, `n/8` healthy feeds, the retail `cam_main only` note, and buttons that move the **GVD VISION** OpenCV window to screen 1 / 2.
- **+ nerd** — loop/camera Hz, infer ms, VRAM, detector, actuator, clip encoder, heartbeat age.

Titles stay **GVD** / **VISION**; Alt+G works with or without the app open. Offline check: `PYTHONPATH=. python scripts/test_gvd_ui_app.py`.

## Two views

Visualization **toy** (not a scientific claim that forecasts match Waymo). On-screen title: **GVD** / **VISION**. No Tesla logos, no “Full Self-Driving”, no “FSD” product label.

**In-game (BeamNG world):** ice-blue ribbon drawn on the pavement via GELua `debugDrawer` (`drawSquarePrism`, 3-line fallback) — **1:1** with `path_ego` / `path_world` (x right, y forward, z up). Track ghosts sit on the road at the same transform; CIPV is brighter. Engage with Alt+G. GELua resolves `Documents/GVD` via USERPROFILE/LOCALAPPDATA/`FS:getUserPath` (USERPROFILE is often empty in-process). This is **not** a 2D camera overlay. In-game strip app: **GVD Strip** (mode · Hz · TTC · N).

**Second screen (`GVD VISION`):** OpenCV cabin on monitor 2 when available (`--viz-screen auto|1|2`, `--viz-fullscreen`, or `GVD_VIZ_MONITOR=2`). This is the VISION lexicon: near-black void stage, thin vector lane paint (`lanes_ext`: solid detected / dim-dashed predicted; smoke stubs only), warm-grey kerbs (`road_edges`), ice-blue ego corridor (intent shade, chevrons while slowing, stop bar when halted behind a CIPV), agent boxes (ice-blue in-path, CIPV **LEAD**, red **BRAKE**), forecast fans, and stop-sign / traffic-light / pole glyphs when `signs[]` is present. Optional `cam_main` PIP. The nerd **VIZ** tab / keys `1–5` add a denser debug stack (occupancy from tracks, detector boxes, camera FOV, planner samples) on top of that lexicon — overlay only, still vision-only at inference. Loop under 8 Hz drops fans, signs and PIP first. Caps: 32 agents / 16 forecast fans / 3 modes. One monitor → window stays put; drag it, or use motherboard HDMI for UHD 630 as display 2. Live dual-monitor + Alt+G still **UNPROVEN** on Linux / until Windows gate.

```bash
pip install -r requirements-viz.txt
PYTHONPATH=. python python/run_vision.py --smoke   # writes docs/gvd_viz_smoke.png
PYTHONPATH=. python python/run_vision.py --viz      # live window + state file for BeamNG
PYTHONPATH=. python scripts/test_gvd_viz_stage.py
```

Keys in `--viz`: `V` nerd, `D` DRIVE (live actuators), `G` VIZ (occupancy / boxes / FOV / cost), `[` `]` tabs, `j/k` `h/l` / click to edit, `0` clean cabin, `1–5` overlay layers, `T` chase↔BEV, `C` clip, `q` quit. Occupancy is derived from tracks, not a learned grid.

## Status

**Force-feedback player override:** wheel chatter no longer disengages GVD. Signal is `|steering_input − aligned cmd.steer|` (never an absolute angle), then spike reject → EMA → hysteresis → dwell. Pedals are asymmetric and tight. Live FFB is **UNPROVEN**. Details under [Actuation](#actuation-m3).

**M6 (retail package):** player-ready **retail** slice. `scripts/make_release_zip.py` (+ `.bat`) builds `dist/gvd-retail-<version>.zip`. `play_gvd.bat` runs the **`window` backend** (`cams=1/8 path=retail`), offers `pip install -r requirements-retail.txt`, keeps the console open on errors. Drive bus: `gvd_cmd.json` → Lua `input.event`; echo `gvd_ego.json`. 8 cams + BeamNGpy stay Tech. Live Alt+G / drive still **UNPROVEN**. Player guide above.

**M5 (shadow + tiny E2E):** Perception always runs; actuators only when engaged. `--policy modular|e2e|shadow` (default **modular**). `policy_e2e` = PilotNet-scale stub. Shadow fields every tick; modular veto on low `lane_conf` / heartbeat / disagreement. Toy VRAM ~0.15–0.4 GB. No transformers/ViT/BEV.

**M4 (clips):** ring-buffer + QSV/libx264 flush on disengage / AEB / near-miss / key C.

**M3 (actuation):** sim-only BeamNGpy `vehicle.control` / `gvd_cmd.json`. Engage Alt+G. Dead-man unchanged.

**M2 (perception):** detect→track→CIPV→corridor; live default no synthetic cars.

**M1 (cameras + hw probe):** `beamngpy | window | stub`, 8-cam yaml, hw_probe. Live still **UNPROVEN on Linux**.

Host profile (target): Intel **i9-9900K** + **UHD 630** (QSV encode) + **GTX 1080 Ti 11 GB** (infer ≤4 GB) + **32 GB DDR4**. See `config/hardware.yaml`.

## Cameras (M1)

```bash
pip install -r requirements.txt          # core
pip install -r requirements-beamng.txt   # optional: beamngpy + bettercam/mss
PYTHONPATH=. python python/run_vision.py --smoke
PYTHONPATH=. python python/run_vision.py --backend stub
PYTHONPATH=. python python/run_vision.py --backend window --viz   # retail: 1 window only
PYTHONPATH=. python python/run_vision.py --backend beamngpy       # needs Tech; else honest missing
```

Default `--backend auto`: beamngpy if importable → else window → else stub. `play_gvd.bat` (retail launcher) forces `--backend window` (`set GVD_BACKEND=beamngpy` for Tech). Window backend fills **main / cam_main only**; other `cam_health` stay `missing`. Boot line ends with `cams=1/8 path=retail (1 window capture; not 8)`; nerd panel shows `retail: 1 window`. Never synthesizes 8 frames from one grab.

Window capture prefers a visible window whose title contains **BeamNG** (Win32 / wmctrl). If none is found, it falls back to the primary monitor — use **fullscreen BeamNG** in that case (`capture_note` says so).

Tech path: `GVD_BEAMNG=1` (or `--backend beamngpy` / `play_gvd_tech.bat`) attaches color-only BeamNGpy `Camera` sensors from `config/cameras.yaml` to the player vehicle, converting GVD frame (+X right, +Y forward) into BeamNG Camera vehicle space. Depth/semantic stay OFF. The same session attaches Electrics, Damage, GForces, and GPS (`config/tech.yaml`) — GPS is a nav hint (lat/lon + optional pin), **not a route**. Optional LiDAR / radar: `config/sensors.yaml` (off by default; Foxglove / future fusion only). Live smoke still **UNPROVEN on Linux**.

Live start **refuses** (exit 1) if probed dGPU VRAM is under 10 GB while BeamNG is running, unless `--vision-only`.

BIOS (Windows): enable **iGPU Multi-Monitor** so UHD 630 QSV exists while 1080 Ti drives the display (M4 encode). Do not set DVMT to 2 GB.

## Perception (M2)

Vision-only **inference**: RGB + ego kinematics + GPS as a nav hint. Optional extras are Foxglove / future fusion only — they never feed `ModularPerception` or the planner. The corridor planner ignores the pin. Inspired by VisionPilot / Apollo camera-pipeline *names* — reimplemented tiny in-repo (not a vendor fork). Forecasts stay CV/CYR toys.

```bash
pip install -r requirements.txt
pip install -r requirements-perception.txt   # optional
python scripts/download_yolov8n.py --onnx
# or: yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True && mv yolov8n.onnx models/
PYTHONPATH=. python python/run_vision.py --smoke
```

## Actuation (M3)

Sim-only. Preferred (Tech): BeamNGpy `set_shift_mode("arcade")` + `control(steering, throttle, brake)` on the player vehicle (`get_current` / `get_player_vehicle_id`). Retail: write `Documents/GVD/gvd_cmd.json`; GELua polls it at 20 Hz and applies it with vehicle-Lua `input.event` — **no DLL**.

```bash
PYTHONPATH=. python python/run_vision.py --backend beamngpy --viz   # Tech path
# --allow-preview-drive   # opt-in only; default blocks preview paths
```

Safety: Alt+G engage; heartbeat dead-man; AEB `brake=1`/`throttle=0`; driver override (below) disengages and stays off until Alt+G; kill Python → `finally` stop + `engaged=false`, Lua releases the car and fades the ribbon; Lua-side dead-man holds the brake after `CMD_STALE_S` (0.35 s) without a new `seq` and releases + disengages after `CMD_DEAD_S` (1.0 s). `--force-engage` is **debug-only** (never default). Live drive on Windows = Michael smoke / still UNPROVEN here. Live Tech / FFB / Alt+G still **UNPROVEN**.

**Player override (force-feedback residual).** `python/control/override.py` and `gvd_main` run the same maths on both sides of the bus. The signal is `|steering_input − cmd.steer|` against the command that was in force when the echo was sampled (`applied_seq` / last applied), never an absolute angle — so GVD's own steer coming back is residual 0. Then: spike reject (`steer_spike` 0.20, a sample-to-sample jump is mechanical, the EMA holds) → EMA on the residual only (`lpf_tau_ms` 80) → soft opposition bias (a residual fighting GVD's steer counts a little more) → hysteresis (`steer_enter` 0.08 / `steer_exit` 0.04) → dwell (`steer_hold_ms` 200). Pedals are asymmetric and tight: no filter, no dwell, one-sided (only a press beyond what GVD asked for), `brake_enter` 0.06 / `throttle_enter` 0.10. Reasons: `player_steer` / `player_brake` / `player_throttle`. Thresholds: `config/control.yaml` `override:`, mirrored into `gvd_state.json` as `override_cfg`. This does **not** change `CMD_DEAD_S`. Live FFB is **UNPROVEN**. Offline check: `PYTHONPATH=. python scripts/test_ffb_override.py`.

`cmd_json` (retail, `window` backend — no BeamNGpy vehicle) writes `gvd_cmd.json` = `{steer, throttle, brake, seq, engaged, heartbeat_mtime, reason}` every tick. `gvd_main.applyCmdJson` applies it only while the mod is engaged **and** the payload says `engaged` **and** `seq` keeps advancing; it echoes `applied_seq` plus electrics in `gvd_ego.json`. `cmd_applied=true` / `cmd_reason=cmd_json_applied` only on a fresh ack; `cmd_json_pending` = written but not acked (mod off, no vehicle); `cmd_json_idle` = not engaged, Lua hands off. Gate reasons (`not_engaged`, `preview_blocked`, `veto:*`, `player_steer`, `player_brake`, `player_throttle`) pass through unchanged. Engaged gate holds (e.g. `preview_blocked`) ride along as `brake=1` and are applied.

## Clips (M4)

```bash
PYTHONPATH=. python python/run_vision.py --smoke          # dry-run clip + viz smoke
PYTHONPATH=. python scripts/test_m4_clips.py
PYTHONPATH=. python python/run_vision.py --viz            # key C = manual clip
```

Clips land in `Documents/GVD/clips/` (repo `data/clips/` gitignored). Encode: `ffmpeg` `h264_qsv` if `hw_probe` qsv=yes, else `libx264` veryfast CRF~23. `--encode nvenc` only when you explicitly want Pascal encode (not default).

## Shadow / E2E (M5)

```bash
PYTHONPATH=. python python/run_vision.py --smoke
PYTHONPATH=. python python/run_vision.py --policy shadow --backend stub
PYTHONPATH=. python python/run_vision.py --policy e2e --backend stub   # stub if no models/e2e_current.onnx
PYTHONPATH=. python scripts/test_m5_shadow.py
PYTHONPATH=. python -m python.train.train_e2e --smoke
```

Modular veto thresholds: `config/control.yaml`. E2E input 320×180: `config/perception.yaml` / `control.yaml`. Weights stay out of git.
