# Grok Vision Drive (GVD)

Public **MIT** BeamNG.drive / BeamNG.tech **camera-only** self-driving **toy** inspired by Tesla FSD *software strategies* — not a clone, not a Tesla product, and **not** real-vehicle control.

Entertainment only. Never use this stack to control a physical car.

## Honesty (M0)

- **Vision only** at inference (RGB cams + ego kinematics + coarse nav hint). No LiDAR / radar / ultrasonic / GPS-as-localization at runtime.
- **Tech path** (preferred): BeamNGpy Camera sensors, 8-cam rig. **Retail path**: window-capture fallback for a **single** main view — do not pretend that is 8 cameras.
- Default GPU profile: **NVIDIA GTX 1080 Ti (11 GB)**. Live stack must share the card with BeamNG.
- Not Tesla FSD. No Tesla logos. Repository title stays free of “FSD”. “Vision Drive” / “camera autopilot toy” are fine.
- Strategy analogies in docs mean: Tesla idea → **our toy version does X**.

Credits: [VisionPilot](https://github.com/visionpilot-project/VisionPilot), BeamNG / BeamNGpy, Udacity/Aly lane pipelines, Ultralytics if YOLO is used later.

## Install

1. Double-click `install.bat`
2. Restart BeamNG
3. Enable Grok Vision Drive in Mod Manager if needed (Alt+A engage)

Optional: double-click `play_gvd.bat` to start the Python stub + Steam BeamNG (`284160`). Uninstall: `uninstall.bat` (removes only `mods\unpacked\gvd` and `mods\gvd.zip`).

Windows first. No admin. Never writes into `Steam\steamapps\common\BeamNG.drive`. “Inject” here means copy into the correct user mods folder so BeamNG loads the mod itself — no DLLs, hooks, or process injection.

## Layout

```
install.bat / uninstall.bat / play_gvd.bat
beamng_mod/          → copied to %LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd
python/              → also copied to %USERPROFILE%\Documents\GVD\python
config/              → also copied to %USERPROFILE%\Documents\GVD\config
```

After install, unpacked tree:

```
mods\unpacked\gvd\
  scripts\gvd\modScript.lua
  lua\ge\extensions\gvd_main.lua
  lua\ge\extensions\core\input\actions\gvd.json
  settings\inputmaps\keyboardGvd.json
```

## Status

**M3 (actuation, this PR):** sim-only. Preferred BeamNGpy `vehicle.control` (arcade); fallback atomic `Documents/GVD/gvd_cmd.json` (no DLL). Engage = Alt+A → `gvd_engage.json`. No drive when `path_debug_preview=true` unless `--allow-preview-drive`. Ego speed from Electrics when present (never invent 10 m/s). Stale heartbeat / disengage / shutdown → throttle 0 + brake. **Arcade + hold brake can auto-shift reverse** — AEB keeps `throttle=0`. Live BeamNG.tech still **UNPROVEN on Linux**.

**M2 (perception):** detect→track→CIPV→corridor; live default no synthetic cars; `path_debug_preview=false` only for lane-derived corridor.

**M1 (cameras + hw probe):** `beamngpy | window | stub`, 8-cam yaml, hw_probe. Live Camera attach / Alt+A still **UNPROVEN on Linux**.

M4 clips next.

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

Default `--backend auto`: beamngpy if importable → else window → else stub. Window backend fills **main / cam_main only**; other `cam_health` stay `missing`. Nerd panel shows `retail: 1 window` when that backend is active. Never synthesizes 8 frames from one grab.

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

`cmd_json` fallback writes `gvd_cmd.json` but reports `cmd_applied=false` / `cmd_json_sink` until a real GE apply exists (Lua poll is a no-op sink today).
