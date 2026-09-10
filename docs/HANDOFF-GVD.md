# GVD handoff (cold resume)

Durable parked-state note for any later Cursor Cloud Grok session.
Do not invent a newer main or hotfix tip than the SHAs below.
Do not merge [#22](https://github.com/Mbeck2008-web/grok-vision-drive/pull/22) until the Windows live prove passes.
Do not claim live dual-monitor proven.

Repo: [Mbeck2008-web/grok-vision-drive](https://github.com/Mbeck2008-web/grok-vision-drive)

Schema / UI contract (not a handoff): [`docs/gvd_state_schema.md`](gvd_state_schema.md)


## Snapshot (parked 2026-09-10)

Michael handed GVD off from **Grok Bot Code Lab** to **normal Cursor Cloud Grok** until Grok Bot usage resets. He was away from the computer until the next day and asked to freeze Grok Bot Lab burns.

| Item | Value |
| --- | --- |
| Main tip before hotfix work | `5824259` -- includes [#21](https://github.com/Mbeck2008-web/grok-vision-drive/pull/21) OpenCV VISION lexicon on the second screen, slim in-game Engage, Alt+G |
| Open PR | [#22](https://github.com/Mbeck2008-web/grok-vision-drive/pull/22) (draft). **Do not merge.** |
| Branch | `cursor/hotfix-cef-ascii-actionmap-711b` |
| Hotfix tip (CEF / ActionMap / cp1252) | `47f2ceb646143a5a24b281aebddd8cd51324f5af` |
| Critic | 8.8/10 PASS (CEF ASCII `app.js`, Alt+G ActionMap reload + ASCII titles, cp1252-safe prints with ASCII `->`). Soft only: live dual **UNPROVEN**; `hw_probe.py` comments still had unicode (optional; this follow-up ASCII-sanitized those comments only) |
| Verify | **BLOCKED** -- Desktop-IQU45 (Windows gaming PC) was offline / Grok Bot desktop app disconnected. Cannot wipe/reinstall or prove live. A partial wipe may have left unpacked `gvd` dirty. **Do not merge-green and do not merge #22 until live prove passes.** |
| Host for live prove | Desktop-IQU45, BeamNG.drive **0.39.4**, DX (not Vulkan) |
| Hardware profile | i9-9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB |

This follow-up (handoff doc + optional `hw_probe` comment ASCII) may sit on a newer commit on the same branch. That commit is **not** a new product tip. Reinstall from `47f2ceb` **or** this branch tip (it contains `47f2ceb`). Do not treat a later SHA as a new hotfix unless it changes drive / CEF / ActionMap / print paths.

Live dual-monitor + Alt+G remains **UNPROVEN**. Live FFB remains **UNPROVEN**. Live QSV remains **UNPROVEN** until Windows smoke.


## Product rules (non-negotiable)

- Vision-only BeamNG **toy**. No Tesla logos. No "FSD" / "Full Self-Driving" chrome or names. Repo title stays free of "FSD".
- **Retail** = 1-cam OpenCV / window capture + Lua cmd bus (`Documents/GVD/gvd_cmd.json` -> GELua `input.event`). Do not pretend that is 8 cameras.
- **Tech** = BeamNGpy 8-cam + `vehicle.control` (preferred path; needs BeamNG.tech).
- In-game Apps **GVD** = slim Engage / Disengage + settings only.
- Rich VISION lexicon lives on the Python `python/viz/` second screen (`stage.py`) from #21. Do not put the cabin lexicon back into the CEF app.
- Engage key: **Alt+G** (and **Ctrl+Alt+G**). **Alt+A** is stock BeamNG `toggleRangeStatus`. Do not steal it.
- Hardware: 9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB. Live start refuses if probed dGPU VRAM < 10 GB while BeamNG is up, unless `--vision-only`.
- Entertainment only. Never control a physical car.


## What #22 already fixed (live blockers on `5824259`)

Surgical ASCII-only strings. No drive-logic rewrite. In-game app stays the slim Engage + settings from #21.

1. **CEF blank Apps tile.** Non-ASCII in `beamng_mod` UI (`app.js` / `app.html`: em-dash / middot / ellipsis) broke BeamNG CEF. The `gvd-app` directive never registered. Tile was blank (Hide only). Repo now uses `--` / `-` / `...`.
2. **ActionMap description.** BeamNG could not create a description for `keyboard0::alt+g` / `ctrl+alt+g`. Fixed with short ASCII title/desc (`GVD Engage` / `Toggle GVD engage`) in `actions/gvd.json`. `onExtensionLoaded` reloads **actions then bindings** so Alt+G gets a description after `gvd.json` is visible.
3. **Windows cp1252 crash.** `UnicodeEncodeError` on U+2192 in `python/viz/monitors.py` `place_opencv_window`. Prints now use ASCII `->`. Same for `hw_probe` refuse/note **prints**. `scripts/test_gvd_ui_app.py` fails the build on BOM / non-ASCII / non-cp1252 in CEF / JSON / ActionMap / `monitors.py`.

Writing only `Documents/GVD/gvd_engage.json` is **not** Engage. Python never invents engage. Lua only adopts `engaged=false` from that file. Engage always starts in-game (Alt+G or the app button). A file saying `true` never engages Lua.

Touched by `47f2ceb` (do not re-litigate unless live prove fails):

- `beamng_mod/ui/modules/apps/GVD/app.js`, `app.html`, `app.json`
- `beamng_mod/ui/modules/apps/gvd_strip/app.html`, `app.json`
- `beamng_mod/lua/ge/extensions/core/input/actions/gvd.json`
- `beamng_mod/lua/ge/extensions/gvd/main.lua` (ActionMap reload)
- `python/viz/monitors.py`
- `python/runtime/hw_probe.py` (refuse/note prints)
- `scripts/test_gvd_ui_app.py` (ASCII / BOM / cp1252 guard)


## Verify reinstall checklist (Windows 0.39.4, when Desktop-IQU45 is online)

Do this before anyone merges #22 to `main`. A dirty leftover `app.js` from a partial wipe will look like a CEF regression.

1. Fully quit BeamNG.
2. Wipe unpacked `gvd`:
   - `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods\unpacked\gvd`
   - and legacy `%LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd` if present
3. Copy tip `47f2ceb` (or this branch tip, or merged `main` if later merged) `beamng_mod\*` into that path -- or run `install.bat` from the branch. **No hand-sanitized leftover `app.js`.**
4. Relaunch BeamNG **DX** (not Vulkan), enable **Grok Vision Drive** in Mod Manager, sit in a vehicle.
5. Remove the GVD app from the cockpit if it is already placed, then re-add it from Apps (stale layout can keep a dead tile).
6. Prove all of:
   - Tile shows Engage / settings (not blank). Console registers `gvd-app` with no CEF parse error.
   - Alt+G and Ctrl+Alt+G log `[GVD] ENGAGED` / `DISENGAGED` with **no** ActionMap `Could not create a description for binding keyboard0::alt+g` (or `ctrl+alt+g`).
   - Writing only `gvd_engage.json` is **not** enough -- need the Lua toggle (key or app).
   - `play_gvd.bat` / supervisor survives **without** `PYTHONUTF8=1` (no `UnicodeEncodeError` from `place_opencv_window`).
   - OpenCV second-screen lexicon is present when `run_vision --viz` / `play_gvd.bat` runs (void stage, lane fan, ice corridor -- not a blank window).
7. **Only then** merge #22 to `main`.

If verify fails, stay on this branch. Do not merge-green from offline tests alone.


## Soft backlog (out of scope unless Michael asks)

- Cones / crosswalks on the OpenCV lexicon
- Boot nit: `Couldn't find action gvd_toggle_engage` race (reload-on-load is the mitigation; the first bind can still log)
- Critic soft: in-game app height 400
- Live dual-monitor proof with Michael in-car (still **UNPROVEN**)


## How to resume (Cloud Grok)

1. Checkout `cursor/hotfix-cef-ascii-actionmap-711b`. Do not open a parallel "fix CEF" branch unless live prove names a new bug.
2. Do not merge #22. Do not claim live dual proven. Do not burn Grok Bot Lab for this until Michael says so.
3. Offline checks on this host (Linux Cloud) are not a live gate:

```
PYTHONPATH=. python scripts/test_gvd_ui_app.py
PYTHONPATH=. python scripts/test_gvd_viz_stage.py
PYTHONPATH=. python scripts/test_ffb_override.py
PYTHONPATH=. python scripts/test_m6_retail.py
```

4. When Desktop-IQU45 is online, run the reinstall checklist above with Michael in-car. Then merge.
5. Product files that matter for the blank-tile / bind / cp1252 class of bugs:
   - CEF app: `beamng_mod/ui/modules/apps/GVD/`
   - ActionMap: `beamng_mod/lua/ge/extensions/core/input/actions/gvd.json`
   - Binds: `beamng_mod/settings/inputmaps/keyboardGvd.json`
   - Reload: `M.onExtensionLoaded` in `beamng_mod/lua/ge/extensions/gvd/main.lua`
   - Console prints: `python/viz/monitors.py`, `python/runtime/hw_probe.py`
   - Lexicon: `python/viz/stage.py` (second screen), not the in-game canvas
6. Retail drive bus (unchanged by #22): `gvd_cmd.json` -> Lua, `gvd_ego.json` <- vehicle, `gvd_engage.json` = sticky engage/disengage record. See [`gvd_state_schema.md`](gvd_state_schema.md).


## Honesty flags that stay UNPROVEN

- Live dual-monitor + Alt+G on Desktop-IQU45
- Live FFB player-override on a real wheel
- Live QSV encode
- Live 8-cam Tech path on this Cloud host (Linux)

Do not flip any of those to proven from this document or from offline tests.
