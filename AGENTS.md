# Agent brief — install GVD on this PC

Paste this file into [Grok Build](https://x.ai/cli), Grok Bot, or any computer-use agent. Use **extra-high** reasoning. If the repo is not already open, clone https://github.com/Mbeck2008-web/grok-vision-drive and work in the tree.

## Product

Grok Vision Drive (GVD) is an MIT BeamNG.drive / BeamNG.tech **camera-only** autopilot **toy**. Entertainment only. Never control a physical car. Not Tesla. No Tesla logos. No vendor autopilot product names.

## Job

Install GVD on **this** computer and size it to **this** hardware. Read the repo (`README.md`, `install.bat`, `play_gvd.bat`, `play_gvd_tech.bat`, `config/`) instead of asking for a flag list or a hardware table.

Probe or accept this PC: OS, CPU, system RAM, dGPU name and VRAM, iGPU if any. Patch `config/hardware.yaml` to that GPU / VRAM / RAM. Keep inference honest — `max_inference_vram_gb` is a cap, not a brag. Do not invent a CLI `--profile unlimited`. Live start refuses dGPU VRAM under 10 GB while BeamNG is up unless `--vision-only`.

Pick the product from what is **actually installed**. Steam BeamNG.drive → retail: one window capture (`cams=1/8`), `install.bat`, `requirements-retail.txt`, `play_gvd.bat`. BeamNG.tech with `tech.key` next to the exe → 8-cam Tech: `requirements-beamng.txt`, `play_gvd_tech.bat`. Do **not** pick Tech because `beamngpy` is pip-installed. Do not mix Drive and Tech userfolders. Never write into the Steam or Tech **game** folder.

Install Python 3 on PATH if missing, the matching requirements file, and the Lua mod via `install.bat` (Windows, no admin). Enable **Grok Vision Drive** in Mod Manager. Leave `config/tech.yaml` host/port alone unless Tech is the chosen path and the live listen address is wrong.

Do not claim live dual-monitor, Alt+G, FFB, QSV, or Tech 8-cam proven unless you ran it on this box. Retail is not eight cameras. If BeamNG is not installed, stop and say so.
