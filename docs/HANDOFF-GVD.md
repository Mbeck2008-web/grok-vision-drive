# GVD handoff (cold resume)

Durable parked-state note for later Cursor Cloud Grok sessions.
Do not claim live dual-monitor, FFB, QSV, or Tech 8-cam proven from this host.

Repo: [Mbeck2008-web/grok-vision-drive](https://github.com/Mbeck2008-web/grok-vision-drive)

Schema / UI contract: [`docs/gvd_state_schema.md`](gvd_state_schema.md)


## Snapshot (2026-09-16, Critic hotfix after FAIL 8.2/10 on merged #32 @ 160812e)

#32 landed camelCase `gvdApp` + wheel HUD + action-cache bust, but Critic still failed Alt+G ready and HUD cadence:

1. **EGO_POLL_S comment.** pollEgo is ungated; the comment no longer says supervisor-only.
2. **4 Hz wheel HUD.** `tickPush` used 0.25 s unless `showScene`. HUD push is 10 Hz (`UI_PUSH_S = EGO_POLL_S`) and `onEgoFeedback` pushes the same `egoFb` axes.
3. **Swallowed pcalls.** `tryCall` logs `label failed: err` instead of silent `pcall`.
4. **Tests.** `test_gvd_ui_app.py` requires `core_input_actions.load` / `loadActions` then `core_input_bindings.reloadBindings()`, and forbids `bindings.loadActions`.
5. **onFileChanged arity.** Calls are `(path, 'added')`.
6. **bindReady too early.** `getActiveActions().gvd_toggle_engage` title/desc is not a live Alt+G bind. Ready means `reloadBindings()` actually ran (or the delayed `forceRefresh(0.1)` wait elapsed).
7. **onFileChanged skipped reload.** Always `load`/`loadActions` then `reloadBindings()`. `onFileChanged` is not an `elseif` substitute.

| Item | Value |
| --- | --- |
| `main` tip this branch is off | `160812e` / `2f5cb71` -- merged #32 |
| Live prove | **UNPROVEN** -- Desktop-IQU45, BeamNG.drive **0.39.4**, DX (not Vulkan) |
| Hardware profile | i9-9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB |

Keep the camelCase `gvdApp` / `gvdStrip` contract and object `css`. Do not regress to `gvd-app`. Live dual-monitor + Alt+G remains **UNPROVEN**.

| Item | Value |
| --- | --- |
| `main` tip this branch is off | `160812e` / `2f5cb71` -- merged #32 |
| Live prove | **UNPROVEN** -- Desktop-IQU45, BeamNG.drive **0.39.4**, DX (not Vulkan) |
| Hardware profile | i9-9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB |

Keep the camelCase `gvdApp` contract. Do not regress to `gvd-app`. Live dual-monitor + Alt+G remains **UNPROVEN**.


## Snapshot (2026-09-13, CEF + Alt+G hotfix after live FAIL @ main 1543777)

Live Desktop-IQU45 on main@1543777 (includes #22 ASCII): Apps tile still blank, Alt+G still `Could not create a description`, early `Couldn't find action gvd_toggle_engage` on `keyboard.diff` ~1s before mod mount. ASCII was not the remaining CEF bug.

Root causes (repo + BeamNG 0.39 UiApps / input docs):

1. **CEF directive.** `app.json` had `"directive": "gvd-app"` (kebab). UiAppsService does `$injector.has(directive + 'Directive')` after loading `app.js`. Angular registers `gvdAppDirective`. Lookup of `gvd-appDirective` fails -- tile blank, Hide only. Official app contract is camelCase `"directive": "gvdApp"` matching `.directive('gvdApp', ...)`, plus `css` as an object (not a JSON string).
2. **ActionMap race.** `core_input_actions` caches the action table on first read. `keyboard.diff` binds `gvd_toggle_engage` before unpacked `gvd.json` is mounted. #22 called `core_input_bindings.loadActions()` -- that function is not on that module (`load`/`getActiveActions` live on `core_input_actions`). Reload never saw `gvd.json`, so `am:bind` still had no title/desc.

This hotfix rewrites the slim in-game app to the official contract, ships live wheel/pedal readouts from the existing `egoFb` / `gvd_ego.json` Direct Drive echo (no second input stack), busts the action cache, and defer/retries `reloadBindings`.

| Item | Value |
| --- | --- |
| `main` tip this branch is off | `1543777` -- includes #22 ASCII CEF / Alt+G / cp1252 and #31 Direct Drive wheel+pedals |
| Live prove | **UNPROVEN** -- Desktop-IQU45, BeamNG.drive **0.39.4**, DX (not Vulkan) |
| Hardware profile | i9-9900K + UHD 630 QSV + GTX 1080 Ti, infer <= 4 GB |

Live dual-monitor + Alt+G remains **UNPROVEN**. Live FFB remains **UNPROVEN**. Live QSV remains **UNPROVEN**. Live Tech remains **UNPROVEN**.


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
- **Retail** = 1-cam OpenCV / window capture + Lua cmd bus (`Documents/GVD/gvd_cmd.json` -> GELua secondary Direct Drive wheel+pedals). Do not pretend that is 8 cameras.
- **Tech** = BeamNGpy 8-cam + `vehicle.control` (preferred path; needs BeamNG.tech). Optional extras are a sensor bus the planner ignores. GPS is a nav hint, not a route.
- In-game Apps **GVD** = slim Engage / Disengage + settings + live wheel/pedal echo only.
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


## Verify reinstall checklist (Windows 0.39.4, Desktop-IQU45, Michael in-car)

A dirty leftover `app.js` / stale HUD layout will look like a CEF regression. Install **this hotfix tip**, not main@160812e (#32).

1. Fully quit BeamNG.
2. Wipe unpacked `gvd`:
   - `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods\unpacked\gvd`
   - and legacy `%LOCALAPPDATA%\BeamNG.drive\<ver>\mods\unpacked\gvd` if present
3. Copy this tip's `beamng_mod\*` into that path -- or run `install.bat`. **No leftover `app.js` / `app.json`.** Confirm installed `app.json` has `"directive": "gvdApp"` (camelCase) and `css` is an object.
4. Relaunch BeamNG **DX** (not Vulkan), enable **Grok Vision Drive** in Mod Manager, sit in a vehicle.
5. Remove the GVD app from the cockpit if it is already placed, then re-add it from HUD Apps (stale layout can keep a dead tile).
6. Prove all of:
   - Tile shows Engage / settings / **WHEEL / PEDALS** (not blank, not only Hide). Console has **no** `[UiAppsService] failed to load directive: gvd-app`.
   - Steering the wheel and pressing gas/brake moves the in-game bars from the existing `gvd_ego.json` / `egoFb` echo (applied electrics + player lastInputs). Bars should feel live (~10 Hz, same as the ego poll), not stepped at 4 Hz. No second input stack.
   - Alt+G and Ctrl+Alt+G log `[GVD] ENGAGED` / `DISENGAGED` with **no** leftover ActionMap `Could not create a description for binding keyboard0::alt+g` (or `ctrl+alt+g`) after the extension is loaded. An early boot `Couldn't find action gvd_toggle_engage` on `keyboard.diff` is OK only if the later retry logs `[GVD] Alt+G action ready` *after* bindings reload (not merely after the action cache drop) and the key then works.
   - Writing only `gvd_engage.json` is **not** Engage -- need the Lua toggle (key or app). Ignore a stale Sep-9 `engaged:true` file.
   - `play_gvd.bat` / supervisor survives **without** `PYTHONUTF8=1`.
   - OpenCV second-screen lexicon is present when `run_vision --viz` / `play_gvd.bat` runs. In-game app stays slim (no VISION canvas).

If verify fails, stay on this hotfix branch. Do not call live dual proven from offline tests.


## Soft backlog (out of scope unless Michael asks)

- Cones / crosswalks on the OpenCV lexicon
- Boot nit: first `Couldn't find action gvd_toggle_engage` on `keyboard.diff` can still log before mount (retry is the mitigation)
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
lua5.1 scripts/test_gvd_bind_reload.lua   # bundled via test_m6_retail when lua5.1/luajit exists
```

4. When Desktop-IQU45 is online, run the reinstall checklist above with Michael in-car.
5. Product files that matter for the blank-tile / bind / cp1252 class of bugs:
   - CEF app: `beamng_mod/ui/modules/apps/GVD/` -- `app.json` directive must be camelCase `gvdApp`
   - ActionMap: `beamng_mod/lua/ge/extensions/core/input/actions/gvd.json`
   - Binds: `beamng_mod/settings/inputmaps/keyboardGvd.json`
   - Reload: `refreshEngageBindings` in `gvd/main.lua` (`core_input_actions.load` / `loadActions`, then `core_input_bindings.reloadBindings`; `onFileChanged(path, 'added')` must not skip that reload)
   - Console prints: `python/viz/monitors.py`, `python/runtime/hw_probe.py`
   - Lexicon: `python/viz/stage.py` (second screen), not the in-game canvas
6. Retail drive bus: `gvd_cmd.json` -> Lua, `gvd_ego.json` <- vehicle, `gvd_engage.json` = sticky engage/disengage record. See [`gvd_state_schema.md`](gvd_state_schema.md).


## Honesty flags that stay UNPROVEN

- Live dual-monitor + Alt+G on Desktop-IQU45
- Live FFB player-override on a real wheel
- Live QSV encode
- Live 8-cam Tech path on this Cloud host (Linux)

Do not flip any of those to proven from this document or from offline tests.
