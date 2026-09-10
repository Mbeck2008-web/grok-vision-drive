# Grok Vision Drive (GVD)

Public **MIT** BeamNG.drive / BeamNG.tech **camera-only** self-driving **toy** inspired by Tesla FSD *software strategies* — not a clone, not a Tesla product, and **not** real-vehicle control.

Entertainment only. Never use this stack to control a physical car.

## Honesty (M0)

- **Vision only** at inference (RGB cams + ego kinematics + coarse nav hint). No LiDAR / radar / ultrasonic / GPS-as-localization at runtime.
- **Tech path** (preferred): BeamNGpy Camera sensors, 8-cam rig, BeamNGpy `vehicle.control`. **Retail path**: window-capture fallback for a **single** main view — do not pretend that is 8 cameras. Retail **drives** through the mod: Python writes `Documents/GVD/gvd_cmd.json`, GELua applies it to the player vehicle with the same vehicle-Lua `input.event` calls BeamNG's own AI uses (no DLL, no hooks), and echoes speed/inputs back via `gvd_ego.json`.
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

Keys in **GVD VISION**: `V` nerd panel, `?` help, `0` clean cabin, `1–5` debug layers, `T` chase↔BEV, `C` manual clip, `q` quit.

**How retail drives.** Every tick the supervisor writes `Documents\GVD\gvd_cmd.json` (`steer, throttle, brake, seq, engaged, heartbeat_mtime`). The mod polls it at 20 Hz and, while **both** sides are engaged, feeds the player vehicle with `input.event('steering', s, 1)` / `input.event('throttle'|'brake', v, 2)` — the calls BeamNG's own AI and BeamNGpy use (`+steer` = right, pad-smoothed steering, direct pedals) — and switches the gearbox to arcade once. Vehicle Lua echoes `wheelspeed`, `steering_input`, `throttle_input`, `brake_input` and the applied `seq` back through `gvd_ego.json`, which gives the planner real ego speed (closed-loop throttle, TTC/AEB) and lets `cmd_applied` be `true` only on a fresh Lua ack (`cmd_reason`: `cmd_json_applied` / `cmd_json_pending` / `cmd_json_idle`).

**Heartbeat dead-man and disengage.** The supervisor stamps `heartbeat_mtime` into `gvd_state.json` every tick; if it stalls for more than 0.35 s the ribbon fades over 0.4 s and hides, returning when the beat returns. On the drive side the mod only applies commands while `seq` keeps advancing: `CMD_STALE_S` 0.35 s without a new command → steering straight, throttle 0, **brake hold**; `CMD_DEAD_S` 1.0 s → inputs released, auto-disengage, `gvd_engage.json` set to `false`. You are disengaged when:

- you press Alt+G / **Disengage** — the mod sends steering/throttle/brake 0 once and stops touching the car;
- you steer against it (`player_steer`) or use a pedal (`player_brake` / `player_throttle`) — sticky until the next Alt+G, judged by the mod **and** the supervisor, whichever sees it first;
- the supervisor quits (`q`, window closed, Ctrl+C): its `finally` block writes a stop and `engaged=false`, the mod releases the car;
- the modular supervisor vetoes an E2E/shadow policy (`--policy e2e|shadow`: low `lane_conf` / `path_conf`, disagreement, AEB). The default `modular` policy disengages only via Alt+G, override, supervisor exit or the dead-man.

The mod reads `engaged=false` back and flips the HUD to **OFF** within ~0.1 s (it never turns engage *on* from a file — engage always starts in-game). Re-engage with Alt+G. A clip is flushed on every disengage, AEB brake and near-miss (`Documents\GVD\clips\clip_<time>_<trigger>`).

Files: `Documents\GVD\{gvd_state.json, gvd_engage.json, gvd_cmd.json, gvd_ego.json, gvd_ui_prefs.json, clips\, python\, config\}`. `uninstall.bat` removes the mod only; delete `Documents\GVD` yourself if you want a clean slate. Nothing here talks to a real car and nothing carries Tesla / “FSD” branding.

Troubleshooting: mod missing in Mod Manager → run `install.bat` again and check the path it prints. “Python not found” → install Python 3 with *Add to PATH*. “Missing Python packages” → answer `Y` or run `pip install -r requirements-retail.txt`. No `cam_main` PIP / `capture_note` says *fullscreen/monitor* → make the BeamNG window visible with **BeamNG** in its title, or run fullscreen; `window via unavailable` → `pip install mss`. HUD stays `ON`, never `DRIVE` → GVD only drives once it sees both lane lines (nerd panel `lane` conf, `preview=`); `--allow-preview-drive` follows the steer-preview path instead (debug only). `DRIVE` but the car does nothing → nerd panel `cmd` line: `cmd_json_pending` means the mod is not acking (mod not enabled, no player vehicle, or the supervisor is engaged while the game is not). Boot refuses (`REFUSE: dGPU VRAM … < 10 GB while BeamNG is up`) → `play_gvd.bat --vision-only` (drops the GPU guard, nothing else).

## Release zip (M6)

Windows: double-click `scripts\make_release_zip.bat`. Anywhere: `python scripts/make_release_zip.py`. Output: `dist\gvd-retail-<version>.zip` (gitignored), `<version>` = exact git tag if any, else `m6-<sha>[-dirty]`; override with `--version v0.6.0`, `--out path`, `--flat` (no top-level folder), `--list` (manifest only).

Packs: `install.bat`, `uninstall.bat`, `play_gvd.bat`, `beamng_mod/`, `python/` (retail runtime), `config/`, `requirements.txt` + `requirements-retail.txt`, `LICENSE`, `README.md`, `docs/*.md`, `models/.gitkeep`, plus a generated `VERSION.txt` (version, build time, git sha, the retail honesty lines). Excludes `data/clips/`, weights (`*.onnx *.pt *.pth *.bin *.safetensors`), `.git`, `scripts/` (tests + this tool), `__pycache__`, `dist/`. `*.bat` are written CRLF. After writing, the script re-opens the zip, refuses forbidden members and missing must-haves (mod entry point, `run_vision.py`, launchers), and exits non-zero on any problem. Attach the zip to a GitHub Release. Offline check: `PYTHONPATH=. python scripts/test_m6_retail.py`.

## Layout

```
install.bat / uninstall.bat / play_gvd.bat
beamng_mod/          → copied to %LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd
python/              → also copied to %USERPROFILE%\Documents\GVD\python
config/              → also copied to %USERPROFILE%\Documents\GVD\config
requirements-retail.txt   → 1-cam window runtime (numpy, OpenCV, PyYAML, mss; no beamngpy)
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

**In-game (BeamNG world):** ice-blue ribbon drawn on the pavement via GELua `debugDrawer` (`drawSquarePrism`, 3-line fallback) — **1:1** with `path_ego` / `path_world` (x right, y forward, z up). Track ghosts sit on the road at the same transform; CIPV is brighter. Engage with Alt+G. GELua resolves `Documents/GVD` via USERPROFILE/LOCALAPPDATA/`FS:getUserPath` (USERPROFILE is often empty in-process). This is **not** a 2D camera overlay.

**Second screen (`GVD VISION`):** OpenCV cabin on monitor 2 when available (`--viz-screen auto|1|2`, `--viz-fullscreen`, or `GVD_VIZ_MONITOR=2`). This is the VISION lexicon: near-black void stage, thin vector lane paint (`lanes_ext`: solid detected / dim-dashed predicted; smoke stubs only), warm-grey kerbs (`road_edges`), ice-blue ego corridor (intent shade, chevrons while slowing, stop bar when halted behind a CIPV), agent boxes (ice-blue in-path, CIPV **LEAD**, red **BRAKE**), forecast fans, and stop-sign / traffic-light / pole glyphs when `signs[]` is present. Optional `cam_main` PIP. Loop under 8 Hz drops fans, signs and PIP first. One monitor → window stays put; drag it, or use motherboard HDMI for UHD 630 as display 2. Live dual-monitor + Alt+G still **UNPROVEN** on Linux / until Windows gate.

## Status

**Force-feedback player override (this PR):** wheel chatter no longer disengages GVD. The signal is `|steering_input − aligned cmd.steer|` (never an absolute angle), then spike reject → EMA (`lpf_tau_ms` 80) → soft opposition bias → hysteresis (`steer_enter` 0.08 / `steer_exit` 0.04) → dwell (`steer_hold_ms` 200). Pedals are asymmetric and tight (`brake_enter` 0.06 / `throttle_enter` 0.10), no filter. Reasons: `player_steer` / `player_brake` / `player_throttle`. Thresholds live in `config/control.yaml` and are mirrored to the mod through `gvd_state.json`. `CMD_DEAD_S` is unchanged. Live FFB is **UNPROVEN**. Details under [Actuation](#actuation-m3).

**M6 (retail package):** player-ready **retail** slice. `scripts/make_release_zip.py` (+ `.bat`) builds `dist/gvd-retail-<version>.zip` — launchers, `beamng_mod/`, `python/`, `config/`, requirements, LICENSE, README, `VERSION.txt`; never clips, weights, `.git`, tests. `play_gvd.bat` runs the supervisor on the **`window` backend** (1 capture, `cams=1/8 path=retail` in the hw_probe boot line), checks the runtime and offers `pip install -r requirements-retail.txt`, keeps its console open on errors. Retail **drives** over the M3 file bus: `gvd_cmd.json` (`steer/throttle/brake/seq/engaged`) → `gvd_main.applyCmdJson` → vehicle-Lua `input.event` (the stock AI / BeamNGpy calls, arcade shifter once); vehicle electrics echo back through `gvd_ego.json` (wheelspeed, inputs, applied seq) so the planner has real ego speed, TTC works, player override works and `cmd_applied` is claimed only on a fresh Lua ack. Lua dead-man: `CMD_STALE_S` 0.35 s without a new seq → brake hold, `CMD_DEAD_S` 1.0 s → release + auto-disengage; every disengage releases the inputs. Driver override is now sticky (writes `engaged=false`). JSON writers retry on Windows sharing violations. 8 cams + BeamNGpy direct control stay Tech (preferred). Mod adopts `engaged=false` from `gvd_engage.json` when the supervisor vetoes/exits so the HUD never stays ON without a heartbeat. Player guide above. Live Alt+G / drive still **UNPROVEN** here. No Tesla / FSD chrome; no real-car.

**M5 (shadow + tiny E2E):** Perception always runs; actuators only when engaged. `--policy modular|e2e|shadow` (default **modular** = safety supervisor, vetoes E2E). `policy_e2e` = PilotNet-scale tiny CNN/MLP stub: 2×320×180 (main+wide) + speed/steer → `{steer, accel}`; loads `models/e2e_current.onnx` if present else numpy stub. Shadow fields `shadow.{steer,throttle,brake}` written every tick; modular veto on low `lane_conf` / heartbeat / disagreement → hold/disengage (+ clip if recorder). Toy VRAM ~0.15–0.4 GB. No transformers/ViT/BEV/AutoSteer-HD; no Tesla/FSD chrome; no real-car.

**M4 (clips):** ring-buffer + QSV/libx264 flush on disengage / AEB / near-miss / key C.

**M3 (actuation):** sim-only BeamNGpy `vehicle.control` / `gvd_cmd.json`. Engage Alt+G. Dead-man unchanged.

**M2 (perception):** detect→track→CIPV→corridor; live default no synthetic cars.

**M1 (cameras + hw probe):** `beamngpy | window | stub`, 8-cam yaml, hw_probe. Live still **UNPROVEN on Linux**.

Host profile (target): Intel **i9-9900K** + **UHD 630** (QSV encode) + **GTX 1080 Ti 11 GB** (infer ≤4 GB) + **32 GB DDR4**. See `config/hardware.yaml`.



## GVD Viz

Visualization **toy** (not a scientific claim that forecasts match Waymo). On-screen title: **GVD** / **VISION**. No Tesla logos, no “Full Self-Driving”, no “FSD” product label.

### In-game (required)
Ice-blue ego ribbon on the asphalt via GELua `debugDrawer` (`drawSquarePrism` → `drawLine` fallback), data from `Documents/GVD/gvd_state.json`. Engage with Alt+G. See `docs/gvd_state_schema.md`.

### Python window (extra)
OpenCV **GVD VISION** second screen owns the cabin lexicon (void stage, multi-lane fan, ice corridor, CIPV boxes, warm curbs, sign/light glyphs) plus nerd panel + forecast fans:

```bash
pip install -r requirements-viz.txt
PYTHONPATH=. python python/run_vision.py --smoke   # writes docs/gvd_viz_smoke.png
PYTHONPATH=. python python/run_vision.py --viz      # live window + state file for BeamNG
PYTHONPATH=. python scripts/test_gvd_viz_stage.py
```

Keys in `--viz`: `V` nerd, `?` help, `0` clean cabin, `1–5` debug layers, `T` chase↔BEV, `q` quit. Caps: 32 agents / 16 forecast fans / 3 modes. If policy &lt; 8 Hz, drop fans, signs and PIP first. In-game strip app: **GVD Strip** (mode · Hz · TTC · N).


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

Tech path: `GVD_BEAMNG=1` attaches color-only BeamNGpy `Camera` sensors from `config/cameras.yaml` to the current vehicle (`GVD_BEAMNG_HOST`/`PORT`, optional `BNG_HOME`). Depth/semantic stay OFF. Live smoke still **UNPROVEN on Linux**.

Live start **refuses** (exit 1) if probed dGPU VRAM &lt; 10 GB while BeamNG is running, unless `--vision-only`.

BIOS (Windows): enable **iGPU Multi-Monitor** so UHD 630 QSV exists while 1080 Ti drives the display (M4 encode). Do not set DVMT to 2 GB.


## Perception (M2)

Vision-only: no LiDAR/radar/GPS-loc/HD-map in the live loop. Inspired by VisionPilot / Apollo camera-pipeline *names* — reimplemented tiny in-repo (not a vendor fork). Forecasts stay CV/CYR toys.

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

Safety: Alt+G engage; heartbeat dead-man; AEB `brake=1`/`throttle=0`; driver override (below) disengages and stays off until Alt+G; kill Python → `finally` stop + `engaged=false`, Lua releases the car and fades the ribbon; Lua-side dead-man holds the brake after `CMD_STALE_S` (0.35 s) without a new `seq` and releases + disengages after `CMD_DEAD_S` (1.0 s). `--force-engage` is **debug-only** (never default). Live drive on Windows = Michael smoke / still UNPROVEN here.

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

