# Grok Vision Drive (GVD)

Public **MIT** BeamNG.drive / BeamNG.tech **camera-only** self-driving **toy** inspired by Tesla FSD *software strategies* — not a clone, not a Tesla product, and **not** real-vehicle control.

Entertainment only. Never use this stack to control a physical car.

## Honesty (M0)

- **Vision only** at inference (RGB cams + ego kinematics + coarse nav hint). No LiDAR / radar / ultrasonic / GPS-as-localization at runtime.
- **Tech path** (preferred): BeamNGpy Camera sensors, 8-cam rig. **Retail path**: window-capture fallback for a **single** main view — do not pretend that is 8 cameras. Retail also has **no vehicle handle**: GVD never steers the car there (Engage = ribbon + HUD + clips; `gvd_cmd.json` is a no-op sink).
- Default GPU profile: **NVIDIA GTX 1080 Ti (11 GB)**. Live stack must share the card with BeamNG.
- Not Tesla FSD. No Tesla logos. Repository title stays free of “FSD”. “Vision Drive” / “camera autopilot toy” are fine.
- Strategy analogies in docs mean: Tesla idea → **our toy version does X**.

Credits: [VisionPilot](https://github.com/visionpilot-project/VisionPilot), BeamNG / BeamNGpy, Udacity/Aly lane pipelines, Ultralytics if YOLO is used later.

## Install

Get `gvd-retail-<version>.zip` from GitHub Releases (or build it — see [Release zip](#release-zip-m6)), extract it anywhere, then:

1. Double-click `install.bat` (prefers `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods` on 0.38+; else newest versioned folder)
2. Fully quit and restart BeamNG
3. Enable Grok Vision Drive in Mod Manager if needed (Alt+A engage)

Optional: double-click `play_gvd.bat` to start the Python supervisor (retail `window` backend) + Steam BeamNG (`284160`). Uninstall: `uninstall.bat` (removes only `mods\unpacked\gvd` and `mods\gvd.zip`).

Windows first. No admin. Never writes into `Steam\steamapps\common\BeamNG.drive`. “Inject” here means copy into the correct user mods folder so BeamNG loads the mod itself — no DLLs, hooks, or process injection.

## Player guide (retail, M6)

What you get on **retail BeamNG.drive**: the in-game **GVD** app (Engage / Disengage, show path, show ghosts), Alt+A engage, the ice-blue path ribbon on the road, the **GVD VISION** second window (cabin stage, tracks, nerd panel), and clips in `Documents\GVD\clips`.

What retail is **not**: it captures **one** window (the main view). The other seven cameras stay `missing` in the nerd panel and the boot line says `cams=1/8 path=retail`. It has **no vehicle handle**, so GVD **does not steer the car** — `actuator=cmd_json` is a no-op sink and `cmd_applied` stays `false`. Eight cameras and real actuation need BeamNG.**tech** + BeamNGpy (`GVD_BACKEND=beamngpy`, `GVD_BEAMNG=1`).

1. Extract the zip, double-click `install.bat`, restart BeamNG, enable **Grok Vision Drive** in Mod Manager.
2. In BeamNG open **Apps** → add **GVD** (and optionally **GVD Strip**).
3. Install Python 3 (python.org, tick *Add to PATH*), then `pip install -r requirements-retail.txt` — or let `play_gvd.bat` offer to do it when packages are missing.
4. Double-click `play_gvd.bat`. It opens the supervisor console (`cmd /k`, so errors stay readable), the **GVD VISION** window, and BeamNG via Steam. Window capture locks onto a visible window titled **BeamNG** as soon as it appears; if none is found it grabs the primary monitor, so run BeamNG fullscreen in that case (the nerd panel `capture` line tells you which).
5. Drive. Press **Alt+A** or the app's **Engage** to show the ribbon; the HUD strip reads `modular|ON`. Alt+A again = Disengage.

Keys in **GVD VISION**: `V` nerd panel, `?` help, `0` clean cabin, `1–5` debug layers, `T` chase↔BEV, `C` manual clip, `q` quit.

**Heartbeat dead-man and disengage.** The supervisor stamps `heartbeat_mtime` into `Documents\GVD\gvd_state.json` every tick. If it stalls for more than 0.35 s the in-game ribbon fades over 0.4 s and hides; it returns when the beat returns. You are disengaged when:

- you press Alt+A / **Disengage**;
- the supervisor quits (`q`, window closed, Ctrl+C): its `finally` block sends stop (brake 1 / throttle 0 on Tech) and writes `engaged=false` to `gvd_engage.json`;
- the modular supervisor vetoes an E2E/shadow policy (`--policy e2e|shadow`: low `lane_conf` / `path_conf`, disagreement, AEB) or you steer hard against it on Tech (`driver_override`). The default `modular` policy only disengages via Alt+A or supervisor exit.

Since M6 the mod reads that `engaged=false` back and flips the HUD to **OFF** within ~0.1 s (it never turns engage *on* from the file — engage always starts in-game). Re-engage with Alt+A. A clip is flushed on every disengage, AEB brake and near-miss (`Documents\GVD\clips\clip_<time>_<trigger>`).

Files: `Documents\GVD\{gvd_state.json, gvd_engage.json, gvd_cmd.json, gvd_ui_prefs.json, clips\, python\, config\}`. `uninstall.bat` removes the mod only; delete `Documents\GVD` yourself if you want a clean slate. Nothing here talks to a real car and nothing carries Tesla / “FSD” branding.

Troubleshooting: mod missing in Mod Manager → run `install.bat` again and check the path it prints. “Python not found” → install Python 3 with *Add to PATH*. “Missing Python packages” → answer `Y` or run `pip install -r requirements-retail.txt`. No `cam_main` PIP / `capture_note` says *fullscreen/monitor* → make the BeamNG window visible with **BeamNG** in its title, or run fullscreen; `window via unavailable` → `pip install mss`. Boot refuses (`REFUSE: dGPU VRAM … < 10 GB while BeamNG is up`) → `play_gvd.bat --vision-only` (retail never drives anyway).

## Release zip (M6)

Windows: double-click `scripts\make_release_zip.bat`. Anywhere: `python scripts/make_release_zip.py`. Output: `dist\gvd-retail-<version>.zip` (gitignored), `<version>` = exact git tag if any, else `m6-<sha>[-dirty]`; override with `--version v0.6.0`, `--out path`, `--flat` (no top-level folder), `--list` (manifest only).

Packs: `install.bat`, `uninstall.bat`, `play_gvd.bat`, `beamng_mod/`, `python/` (retail runtime), `config/`, `requirements*.txt`, `LICENSE`, `README.md`, `docs/*.md`, `models/.gitkeep`, plus a generated `VERSION.txt` (version, build time, git sha, the retail honesty lines). Excludes `data/clips/`, weights (`*.onnx *.pt *.pth *.bin *.safetensors`), `.git`, `scripts/` (tests + this tool), `__pycache__`, `dist/`. `*.bat` are written CRLF. After writing, the script re-opens the zip, refuses forbidden members and missing must-haves (mod entry point, `run_vision.py`, launchers), and exits non-zero on any problem. Attach the zip to a GitHub Release. Offline check: `PYTHONPATH=. python scripts/test_m6_retail.py`.

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

After install, open BeamNG **Apps** and add **GVD** (`beamng_mod/ui/modules/apps/GVD/`) (Engage / Disengage, show path, show ghosts). Titles stay **GVD** / **VISION**. Alt+A still works. No FSD chrome.

## Two views

**In-game (BeamNG world):** ice-blue ribbon drawn on the pavement via GELua `debugDrawer` (`drawSquarePrism`, 3-line fallback) — **1:1** with `path_ego` / `path_world` (x right, y forward, z up). Track ghosts sit on the road at the same transform; CIPV is brighter. Engage with Alt+A. GELua resolves `Documents/GVD` via USERPROFILE/LOCALAPPDATA/`FS:getUserPath` (USERPROFILE is often empty in-process). This is **not** a 2D camera overlay.

**Second screen (`GVD VISION`):** OpenCV cabin on monitor 2 when available (`--viz-screen auto|1|2`, `--viz-fullscreen`, or `GVD_VIZ_MONITOR=2`). Same corridor, tracks, CIPV **LEAD** mark, optional `cam_main` PIP. One monitor → window stays put; drag it, or use motherboard HDMI for UHD 630 as display 2. Live dual-monitor + Alt+A still **UNPROVEN** on Linux / until Windows gate.

## Status

**M6 (retail package, this PR):** player-ready **retail** slice. `scripts/make_release_zip.py` (+ `.bat`) builds `dist/gvd-retail-<version>.zip` — launchers, `beamng_mod/`, `python/`, `config/`, requirements, LICENSE, README, `VERSION.txt`; never clips, weights, `.git`, tests. `play_gvd.bat` runs the supervisor on the **`window` backend** (1 capture, `cams=1/8 path=retail` in the hw_probe boot line), checks the runtime and offers `pip install -r requirements-retail.txt`, keeps its console open on errors. Retail **cannot drive**: `actuator=cmd_json` is a no-op sink; 8 cams + actuation stay Tech/BeamNGpy. Mod now adopts `engaged=false` from `gvd_engage.json` when the supervisor vetoes/exits so the HUD never stays ON without a heartbeat. Player guide above. Live Alt+A still **UNPROVEN** here. No Tesla / FSD chrome; no real-car.

**M5 (shadow + tiny E2E):** Perception always runs; actuators only when engaged. `--policy modular|e2e|shadow` (default **modular** = safety supervisor, vetoes E2E). `policy_e2e` = PilotNet-scale tiny CNN/MLP stub: 2×320×180 (main+wide) + speed/steer → `{steer, accel}`; loads `models/e2e_current.onnx` if present else numpy stub. Shadow fields `shadow.{steer,throttle,brake}` written every tick; modular veto on low `lane_conf` / heartbeat / disagreement → hold/disengage (+ clip if recorder). Toy VRAM ~0.15–0.4 GB. No transformers/ViT/BEV/AutoSteer-HD; no Tesla/FSD chrome; no real-car.

**M4 (clips):** ring-buffer + QSV/libx264 flush on disengage / AEB / near-miss / key C.

**M3 (actuation):** sim-only BeamNGpy `vehicle.control` / `gvd_cmd.json`. Engage Alt+A. Dead-man unchanged.

**M2 (perception):** detect→track→CIPV→corridor; live default no synthetic cars.

**M1 (cameras + hw probe):** `beamngpy | window | stub`, 8-cam yaml, hw_probe. Live still **UNPROVEN on Linux**.

Host profile (target): Intel **i9-9900K** + **UHD 630** (QSV encode) + **GTX 1080 Ti 11 GB** (infer ≤4 GB) + **32 GB DDR4**. See `config/hardware.yaml`.



## GVD Viz

Visualization **toy** (not a scientific claim that forecasts match Waymo). On-screen title: **GVD** / **VISION**. No Tesla logos, no “Full Self-Driving”, no “FSD” product label.

### In-game (required)
Ice-blue ego ribbon on the asphalt via GELua `debugDrawer` (`drawSquarePrism` → `drawLine` fallback), data from `Documents/GVD/gvd_state.json`. Engage with Alt+A. See `docs/gvd_state_schema.md`.

### Python window (extra)
OpenCV cabin stage + nerd panel + forecast fans:

```bash
pip install -r requirements-viz.txt
PYTHONPATH=. python python/run_vision.py --smoke   # writes docs/gvd_viz_smoke.png
PYTHONPATH=. python python/run_vision.py --viz      # live window + state file for BeamNG
```

Keys in `--viz`: `V` nerd, `?` help, `0` clean cabin, `1–5` debug layers, `T` chase↔BEV, `q` quit. Caps: 32 agents / 16 forecast fans / 3 modes. If policy &lt; 8 Hz, drop fans + PIP first. In-game strip app: **GVD Strip** (mode · Hz · TTC · N).


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

Sim-only. Preferred: BeamNGpy `set_shift_mode("arcade")` + `control(steering, throttle, brake)` on the player vehicle (`get_current` / `get_player_vehicle_id`). Fallback: write `Documents/GVD/gvd_cmd.json` for GELua to poll — **no DLL**.

```bash
PYTHONPATH=. python python/run_vision.py --backend beamngpy --viz   # Tech path
# --allow-preview-drive   # opt-in only; default blocks preview paths
```

Safety: Alt+A engage; heartbeat dead-man; AEB `brake=1`/`throttle=0`; kill Python → `finally` stop + Lua fade (loop itself sets `heartbeat_ok=True` while alive). `--force-engage` is **debug-only** (never default). Live drive on Windows = Michael smoke / still UNPROVEN here.

`cmd_json` fallback writes `gvd_cmd.json` but reports `cmd_applied=false` / `cmd_json_sink` until a real GE apply exists (Lua poll is a no-op sink today). That is the **retail** case: the `window` backend has no vehicle, so the supervisor prints `actuator=cmd_json is a no-op sink` and the car is never driven.


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

