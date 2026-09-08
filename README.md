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
3. Enable Grok Vision Drive in Mod Manager if needed, add the GVD UI app

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
  ui\modModules.json
  settings\inputmaps\
```

## Status

Installer + mod auto-load scaffolding. Vision / data-engine milestones (M1+) come next.
