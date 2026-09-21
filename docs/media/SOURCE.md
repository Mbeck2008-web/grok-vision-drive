# README media provenance

These files are **synthetic**. They are not live photos from the Windows live machine / BeamNG.

The Windows live host had no recent GVD viz or in-game UI shots (only unrelated old BeamNG clips). Repo search found `docs/gvd_viz_smoke.png` (OpenCV smoke) and the 64px `beamng_mod/ui/modules/apps/GVD/app.png` icon — neither is a live cabin or HUD photo.

| File | What | How |
| --- | --- | --- |
| `gvd_cabin_synthetic.png` | GVD VISION clean cabin, chase 3/4, parked (`engaged=false`, title **OFF**, no ice underglow). Agent boxes are empty solids (LEAD / BRAKE, no center disc). Ground, stub lanes, and curbs span the `cameras.yaml` viz range. The ice ribbon is the authored smoke path (~60 m), not a live planner and not stretched to the horizon | `python.viz.stage.smoke` (`scripts/render_readme_media.py`) |
| `gvd_ingame_ui_synthetic.png` | GVD Apps HUD in **DISENGAGED** / standby (`ENGAGE` + Alt+G, nerd closed). CSS from `app.html`, footer text matches the app. Corner stamp `synthetic`. | Chrome raster of that CSS (OpenCV fallback). Unclipped width 330px; the in-game tile stays 330×440 |
| `gvd_ingame_ui_drive_synthetic.png` | Same HUD with the **DRIVE** glance word (`is-drive`, `DISENGAGE`, "steer to take over"). Wheel and gas numbers are placeholders | Same renderer, `drive=True`. Not a live Engage |

Soft Esc parked. Esc is BeamNG **UI Apps**, not GVD Engage. Do not retitle these as live.

To replace with real crops later: drop live GVD cabin / HUD photos here, keep the filenames (or update README paths), and change the README caption from **synthetic** to **real**.
