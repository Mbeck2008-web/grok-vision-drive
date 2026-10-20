# Agent brief — set up GVD on this PC

Paste this file into an agent when someone asks to install or run Grok Vision Drive. Work in the repo. Verify each step before the next one. Ask before any download larger than a few GB (the CUDA wheel is one of those). Do not skip `--check-maps`.

Grok Vision Drive (GVD) is an MIT **camera-only** toy for **BeamNG.tech**. Entertainment only. Never control a physical car. Not Tesla. No Tesla logos. No vendor autopilot product names. Never write into the Steam or Tech **game** folder. Do not call `BeamNGpy.close()`.

## 1. See the machine

Record OS, CPU, RAM, dGPU name, VRAM, and whether an iGPU is present. Patch `config/hardware.yaml` to that GPU, VRAM, and RAM. `max_inference_vram_gb` is a cap, not a brag. Do not invent `--profile unlimited`. Live start refuses a dGPU under 10 GB while BeamNG is up unless `--vision-only`.

This path is **BeamNG.tech 0.39 on Windows**, beamngpy backend. Retail window capture is a different product. Do not pick Tech only because `beamngpy` imports. Do not mix Drive and Tech userfolders.

## 2. Python, then PyTorch

Python 3 on PATH. Confirm with `py -3 --version`.

Ask before installing PyTorch. The wheel is several GB. The build to install is **PyTorch 2.14, CUDA 12.6 (cu126)**:

```
py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then `onnx` and `onnxruntime`. Confirm `torch.cuda.get_device_capability()`. **Pascal (6.1) stays FP32. Compute 7.0 and newer use mixed precision.** The exported ONNX graph is FP32 either way. Do not lower the trainer learning rate. The default is `1e-3`.

Match BeamNGpy to Tech: 0.39 uses beamngpy 1.36 (`config/tech.yaml` `beamngpy_pin`). Install `requirements-beamng.txt` for the Tech extras. Do not put beamngpy into the retail requirements file.

## 3. Data disk

Set `GVD_DATA_DIR` to `<data dir>` on a large drive. C: and D: are refused. The collector and the trainer stop when that drive has under **20 GB free**.

**FAT32 stores at most 4 GB in one file.** Cache chunks are kept under 3.5 GB for that reason. If the data disk is FAT32, say so before a long collect.

## 4. Start Tech, then the supervisor

Launch **one** BeamNG.tech 0.39 process:

```
BeamNG.tech.exe -tcom -console -gfx dx11
```

The research port is **25252**. Env overrides, when set: `GVD_BEAMNG_HOST`, `GVD_BEAMNG_PORT`, `BNG_HOME` / `BEAMNG_HOME`, `GVD_BEAMNG_USER`, `GVD_TECH_LAUNCH` (`0` attach, `1` only if nothing is already listening), `GVD_TECH_SOCKET_TIMEOUT`, `GVD_BEAMNGPY_PIN`. `GVD_BACKEND=beamngpy`, `GVD_PRODUCT=tech`, `GVD_DOCS_DIR` points at the Tech userfolder `Documents\GVD`, not the Drive folder.

If port 25252 is already listening, attach. Do not start a second Tech. `play_gvd_tech.bat` sets `GVD_TECH_LAUNCH=0` and does not launch the game.

Enable **Grok Vision Drive** in the Mod Manager. The supervisor is the GVD VISION window (`play_gvd_tech.bat`). Engage from that window. `q` quits the supervisor and does not close BeamNG. Do not claim Alt+G, force feedback, or a live 8-camera drive unless you saw them on this PC.

## 5. Check the maps before collecting

```
py -3 scripts\collect_scene_lanes.py --check-maps --maps west_coast_usa,italy,east_coast_usa,hirochi_raceway,gridmap_v2,small_island
```

Read the spawn, edge count, and camera count. A level that fails this check does not go in the collect. `--dump-roads` writes `<data dir>\roads\<map>.json.gz` and then exits. Do not collect until `--check-maps` has been run on this install.

Automatic Mode records with `scripts\collect_scene_lanes.py`. Output is `<data dir>\scene_lanes`. West coast at 60% is 12-minute chunks rotated with the other maps, not one long block. `--respawn-min 3` teleports west coast to a new city, highway, or ramp spawn about four times per chunk. `--traffic 2` is the west-coast default: two companion cars, ahead and behind, replaced on each respawn. The ego stays in span mode.

```
py -3 scripts\collect_scene_lanes.py --minutes 120 --maps west_coast_usa,italy,east_coast_usa,hirochi_raceway,gridmap_v2,small_island --map-weights west_coast_usa=0.6 --segment-min 12 --respawn-min 3 --traffic 2 --smoke-s 60
```

City speed is 8–14 m/s, highway 20–30, ramps 10–18, when the spawn type is known. The log line is `collect spawn <segment> <city|highway|ramp>`.

## 6. Validate, then train with BeamNG closed

```
py -3 scripts\validate_scene_lanes.py --strips <data dir>\scene_lanes
```

Quit BeamNG before training so the GPU is not shared with the game. Order is dry-run, then sanity, then the full run. Learning rate stays **1e-3** for the whole run. Holdout stop ends the run. It does not start a second pass at `1e-4`. `--batch` does not change the learning rate. A pause file cannot resume a lower rate. After a failed 1e-4 pass, start the next run with `--no-resume` once that process has exited.

```
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --cams main --build-cache <cache dir>
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --cams main --heads lane_gt --augment --cache <cache dir> --dry-run
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --cams main --heads lane_gt --augment --cache <cache dir> --sanity
py -3 -m python.train.train_scene --strips <data dir>\scene_lanes --cams main --heads lane_gt --augment --balance --cache <cache dir> --epochs 30 --clip 1.0 --lane-loss huber --min-valid 0.3 --export <data dir>\models\e2e_scene.onnx
```

`--cams` is the camera subset (empty means all eight). The ONNX metadata key `gvd_cams` must match, and the live runtime reads it. `--heads lane_gt` zeros the points, valid, class, and lamps losses. The export still has every head. `--augment` is on by default for training and off for validation. It is photometric only (brightness, contrast, gamma). A sideways crop is not applied, because `lane_gt` is in metres. `--cache` reads the decoded chunks. `--scale 0.5` downscales before the cache and the net. `--workers` defaults to 4 on Windows (persistent workers, pinned memory on CUDA) and 0 elsewhere. Each epoch shuffles training windows and starts each one from a fresh GRU state. `--balance` repeats curves, weaves, and ramps. `--min-valid 0.3` drops windows whose lane centers are mostly missing. `--clip 1.0` is `clip_grad_norm`. `--lane-loss huber` is smooth-L1.

## 7. Live scene-net drive

After export:

```
py -3 -m python.run_vision --backend beamngpy --lane-source scene --scene-onnx <data dir>\models\e2e_scene.onnx --gt-log --hz 10
```

The runtime refuses a graph with no `lane_gt` output. A low confidence stops. It does not fall back to the pixel lane finder.

## 8. Status files

| File | What it is |
| --- | --- |
| `<data dir>\status\collect.json` | Collect heartbeat: frames, Hz, disk, speed |
| `<data dir>\status\collect_crash.txt` | Last collect failure |
| `<data dir>\status\validate\report.json` | Validator report |
| `<data dir>\status\metrics.jsonl` | Train loss, grad norm, val r / rms / bias |
| `<data dir>\status\train_heartbeat.json` | Train step and ETA, about every 30 s |
| `<data dir>\status\train_crash.txt` | Train exception and the last batch's row ids |
| `<data dir>\scene_lanes.gvd-pause.npz` | Pause checkpoint. A camera-subset run will not load an 8-camera pause |
| `<data dir>\status\live.json` | Live supervisor status |

## 9. Failures

- **Port 25252 closed:** Tech was started without `-tcom`, or a second copy is fighting the port. One process only.
- **Hello timeout:** Not a missing vehicle. Widen `GVD_TECH_SOCKET_TIMEOUT`. Do not close the socket and retry in a loop.
- **C: or D: refused / under 20 GB:** Point `GVD_DATA_DIR` at the data disk.
- **Cache file over 4 GB:** The data drive is FAT32. Rebuild with `--cams` limited to the cameras you train. Chunks are capped at 3.5 GB.
- **Train loss sits near zero then spikes:** The old loader walked the drive in order. Shuffle, `--balance`, and `--clip 1.0` are the fix. `--heads lane_gt` must show points, valid, class, and lamps at 0.
- **Val rms worse than `rms_mean` after one pass:** The head is still a flat guess. Train and holdout use the same metre labels. There is no per-target normalization to turn on. Keep going with main-only, lane_gt-only, and the cache. Do not drop the learning rate below `1e-3`. Holdout stop ends the run.
- **UnboundLocalError in `_lane_gt_loss`:** An inner `import torch` shadowed the module. That import is gone. Update to a commit that has the fix.
- **Do not claim** a live Tech drive, Alt+G, or eight-camera collect unless this PC just did it.
