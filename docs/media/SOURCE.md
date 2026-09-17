# README media provenance

These files are **synthetic**. They are not live photos from the Windows live machine / BeamNG.

The Windows live host had no recent GVD viz or in-game UI shots (only unrelated old BeamNG clips). Repo search found `docs/gvd_viz_smoke.png` (OpenCV smoke) and the 64px `beamng_mod/ui/modules/apps/GVD/app.png` icon — neither is a live cabin or HUD photo.

| File | What | How |
| --- | --- | --- |
| `gvd_cabin_synthetic.png` | GVD VISION clean cabin, chase 3/4, parked (`engaged=false`, no ice underglow) | `python.viz.stage.smoke` (`scripts/render_readme_media.py`) |
| `gvd_ingame_ui_synthetic.png` | GVD Apps HUD in **DISENGAGED** / standby (`ENGAGE` + Alt+G) | Raster of `beamng_mod/ui/modules/apps/GVD/app.html` CSS (Chrome; OpenCV fallback) |

Soft Esc parked. Esc is BeamNG **UI Apps**, not GVD Engage. Do not retitle these as live.

To replace with real crops later: drop live GVD cabin / HUD photos here, keep the filenames (or update README paths), and change the README caption from **synthetic** to **real**.
