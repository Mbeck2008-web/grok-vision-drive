# GVD handoff (cold resume)

Durable parked-state note for later Cursor Cloud Grok sessions.
Do not claim live dual-monitor, FFB, QSV, or Tech 8-cam proven from this host.

Repo: [Mbeck2008-web/grok-vision-drive](https://github.com/Mbeck2008-web/grok-vision-drive)

Schema / UI contract: [`docs/gvd_state_schema.md`](gvd_state_schema.md)


## Snapshot (2026-09-10, after #22)

Michael asked to land remaining open PRs onto `main` (same session as nerd debug #26). [#22](https://github.com/Mbeck2008-web/grok-vision-drive/pull/22) was merged **at that request** even though the Windows 0.39.4 live prove is still **UNPROVEN**. Offline tests here are not that prove.

| Item | Value |
| --- | --- |
| `main` before this merge | `6b80d89` -- [#21](https://github.com/Mbeck2008-web/grok-vision-drive/pull/21) lexicon, [#24](https://github.com/Mbeck2008-web/grok-vision-drive/pull/24) Tech vehicle + GPS hint, [#25](https://github.com/Mbeck2008-web/grok-vision-drive/pull/25) extras/Foxglove bus, [#26](https://github.com/Mbeck2008-web/grok-vision-drive/pull/26) nerd DRIVE/VIZ |
| #22 branch | `cursor/hotfix-cef-ascii-actionmap-711b` @ `4c9f1d9` (CEF/ActionMap/cp1252 @ `47f2ceb`) |
| Live prove | **UNPROVEN** -- Desktop-IQU45, BeamNG.drive **0.39.4**, DX (not Vulkan) |
| Hardware profile | i9-9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB |

Live dual-monitor + Alt+G remains **UNPROVEN**. Live FFB remains **UNPROVEN**. Live QSV remains **UNPROVEN**. Live Tech remains **UNPROVEN**.


## Product rules (non-negotiable)

- Vision-only BeamNG **toy**. No Tesla logos. No "FSD" / "Full Self-Driving" chrome or names. Repo title stays free of "FSD".
- **Retail** = 1-cam OpenCV / window capture + Lua cmd bus (`Documents/GVD/gvd_cmd.json` -> GELua `input.event`). Do not pretend that is 8 cameras.
- **Tech** = BeamNGpy 8-cam + `vehicle.control` (preferred path; needs BeamNG.tech). Optional extras are a sensor bus the planner ignores. GPS is a nav hint, not a route.
- In-game Apps **GVD** = slim Engage / Disengage + settings only.
- Rich VISION lexicon lives on the Python `python/viz/` second screen (`stage.py`). Nerd DRIVE/VIZ tabs are OpenCV, not CEF.
- Engage key: **Alt+G** (and **Ctrl+Alt+G**). **Alt+A** is stock BeamNG `toggleRangeStatus`. Do not steal it.
- Hardware: 9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB. Live start refuses if probed dGPU VRAM < 10 GB while BeamNG is up, unless `--vision-only`.
- Entertainment only. Never control a physical car.


## What #22 fixed (live blockers on old `5824259`)

Surgical ASCII-only strings. No drive-logic rewrite. In-game app stays slim Engage + settings.

1. **CEF blank Apps tile.** Non-ASCII in `beamng_mod` UI (`app.js` / `app.html`: em-dash / middot / ellipsis) broke BeamNG CEF. The `gvd-app` directive never registered. Tile was blank (Hide only). Repo uses `--` / `-` / `...`.
2. **ActionMap description.** BeamNG could not create a description for `keyboard0::alt+g` / `ctrl+alt+g`. Short ASCII title/desc (`GVD Engage` / `Toggle GVD engage`) in `actions/gvd.json`. `onExtensionLoaded` reloads **actions then bindings** so Alt+G gets a description after `gvd.json` is visible.
3. **Windows cp1252 crash.** `UnicodeEncodeError` on U+2192 in `python/viz/monitors.py` `place_opencv_window`. Prints use ASCII `->`. Same for `hw_probe` refuse/note **prints**. `scripts/test_gvd_ui_app.py` fails the build on BOM / non-ASCII / non-cp1252 in CEF / JSON / ActionMap / `monitors.py`.

Writing only `Documents/GVD/gvd_engage.json` is **not** Engage. Python never invents engage. Lua only adopts `engaged=false` from that file. Engage always starts in-game (Alt+G or the app button). A file saying `true` never engages Lua.

`.cursor/environment.json` + `install.sh` came along on the hotfix tip (Cloud Agent Linux slice). They are repo tooling; they must not ship in the retail zip.


## Verify reinstall checklist (Windows 0.39.4, when Desktop-IQU45 is online)

A dirty leftover `app.js` from a partial wipe will look like a CEF regression. Run this on **current `main`**, not the old `47f2ceb` tip (main also has #24/#25/#26).

1. Fully quit BeamNG.
2. Wipe unpacked `gvd`:
   - `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods\unpacked\gvd`
   - and legacy `%LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd` if present
3. Copy this tip's `beamng_mod\*` into that path -- or run `install.bat`. **No hand-sanitized leftover `app.js`.**
4. Relaunch BeamNG **DX** (not Vulkan), enable **Grok Vision Drive** in Mod Manager, sit in a vehicle.
5. Remove the GVD app from the cockpit if it is already placed, then re-add it from Apps (stale layout can keep a dead tile).
6. Prove all of:
   - Tile shows Engage / settings (not blank). Console registers `gvd-app` with no CEF parse error.
   - Alt+G and Ctrl+Alt+G log `[GVD] ENGAGED` / `DISENGAGED` with **no** ActionMap `Could not create a description for binding keyboard0::alt+g` (or `ctrl+alt+g`).
   - Writing only `gvd_engage.json` is **not** enough -- need the Lua toggle (key or app).
   - `play_gvd.bat` / supervisor survives **without** `PYTHONUTF8=1` (no `UnicodeEncodeError` from `place_opencv_window`).
   - OpenCV second-screen lexicon is present when `run_vision --viz` / `play_gvd.bat` runs (void stage, lane fan, ice corridor -- not a blank window).

If verify fails, stay on a hotfix branch. Do not call live dual proven from offline tests.


## Soft backlog (out of scope unless Michael asks)

- Cones / crosswalks on the OpenCV lexicon
- Boot nit: `Couldn't find action gvd_toggle_engage` race (reload-on-load is the mitigation; the first bind can still log)
- Critic soft: in-game app height 400
- Live dual-monitor proof with Michael in-car (still **UNPROVEN**)


## How to resume (Cloud Grok)

1. Work from current `main`. Do not open a parallel "fix CEF" branch unless live prove names a new bug.
2. Do not claim live dual proven. Do not burn Grok Bot Lab for this until Michael says so.
3. Offline checks on this host (Linux Cloud) are not a live gate:

```
PYTHONPATH=. python scripts/test_gvd_ui_app.py
PYTHONPATH=. python scripts/test_gvd_viz_stage.py
PYTHONPATH=. python scripts/test_ffb_override.py
PYTHONPATH=. python scripts/test_m6_retail.py
PYTHONPATH=. python scripts/test_debug_opts.py
```

4. When Desktop-IQU45 is online, run the reinstall checklist above with Michael in-car.
5. Product files that matter for the blank-tile / bind / cp1252 class of bugs:
   - CEF app: `beamng_mod/ui/modules/apps/GVD/`
   - ActionMap: `beamng_mod/lua/ge/extensions/core/input/actions/gvd.json`
   - Binds: `beamng_mod/settings/inputmaps/keyboardGvd.json`
   - Reload: `M.onExtensionLoaded` in `beamng_mod/lua/ge/extensions/gvd/main.lua`
   - Console prints: `python/viz/monitors.py`, `python/runtime/hw_probe.py`
   - Lexicon: `python/viz/stage.py` (second screen), not the in-game canvas
6. Retail drive bus: `gvd_cmd.json` -> Lua, `gvd_ego.json` <- vehicle, `gvd_engage.json` = sticky engage/disengage record. See [`gvd_state_schema.md`](gvd_state_schema.md).


## Honesty flags that stay UNPROVEN

- Live dual-monitor + Alt+G on Desktop-IQU45
- Live FFB player-override on a real wheel
- Live QSV encode
- Live 8-cam Tech path on this Cloud host (Linux)

Do not flip any of those to proven from this document or from offline tests.
