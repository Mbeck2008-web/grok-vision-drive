# C++ visualizer `gvd_viz` (early FSD beta look)

Status: **M1** (this PR). Replaces the Python/OpenCV early-vector stage of PR #231, which stays open and unmerged as a reference only.

## Goal

A separate C++ / OpenGL 3.3 core app that draws what GVD's perception sees in the look of the earliest Tesla FSD beta. The primary reference is Michael's screenshots in `gvd-viz-refs/fsd-early-beta/ref1-4`, not #231:

- black background
- lane lines as glowing beads/short dashes in the real paint colour: white, and yellow only where the lane record says the paint is yellow. No purple and no colour by index.
- red dotted road edges and curbs (red means edges/curbs and nothing else)
- neon wireframe 3D boxes with a centre dot and glow: blue cars, green lead (CIPV), yellow cross traffic and bikes, magenta trucks, orange parked/obstacles, pink pedestrians
- traffic-light icons with the lit lamp glowing, stop sign plus stop bar and zebra crosswalk
- dotted blue planned path, a white ego car with a ground glow, a speed bar with a set-speed tick
- high chase camera

It has to stay light. It shares the GTX 1080 Ti with BeamNG and the vision model (vision ≤ 4 GB). Targets: < 200 MB VRAM and < 2 ms GPU per frame at 1280x720.

## Non-goals

- No ground truth. The renderer only gets the perception output the supervisor already has. `path_world`, `debug_gt` overlays, `*_gt`/`gt_*` keys and labeler output are never published.
- No changes to safety, actuation or the recorder. The viz cannot block, slow or crash the control loop.
- No 3D car models for traffic (boxes only, as in the early beta). No later v9+ customer UI.
- Live lanes have no paint field yet (`lanes_ext` carries index/kind/points/side/style only), so in a live drive every lane is white, including real yellow centre lines. Yellow shows only when a record carries `paint`/`color`/`material` with "yellow". Detecting paint colour is a separate perception PR.

## Architecture

```
run_vision.py tick ──st──> CppVizPublisher.publish(st)          (Python, same thread)
   whitelist build (perception keys only) -> compact JSON -> one UDP datagram
   non-blocking socket to 127.0.0.1:47800, ≤ 60 Hz, drops on EAGAIN/refused
                                │
                                ▼
gvd_viz --listen 47800  (separate process, GLFW window, vsync)
   drain socket each frame, keep newest valid v1 message, render
gvd_viz --offline scenes.jsonl --png out.png   (headless EGL, tests/gallery)
```

- `cpp/viz/`: CMake ≥ 3.20, C++17. glad 2 (GL 3.3 core + EGL 1.5 runtime loader) and stb are vendored in `third_party/`. glm 1.0.1 and nlohmann_json 3.11.3 come from the system package if present, else pinned FetchContent tarballs (SHA-256 checked). GLFW 3.4 (FetchContent) is used for the live window. It is on by default on Windows and on Linux when X11 dev headers are present.
- Headless: EGL surfaceless (Mesa) on Linux, so CI and the box render on llvmpipe with no X server. On Windows, headless uses a hidden GLFW window.
- Presets: `linux-release` (Ninja, EGL) and `windows-msvc` (VS 2022 x64, GLFW).
- Rendering: one instanced "capsule" primitive (screen-space segment SDF with an analytic glow halo, additive) for lane beads, curb dots, path beads and box edges. Plus a flat-shaded low-poly ego mesh (depth-tested first so lines under it hide), ground quads (stop bar, zebra, ego glow), SDF sprites (traffic light, stop sign) and an SDF HUD. 4x MSAA FBO, resolved, then read back (offline) or blitted (live).
- Fail-safe: `--viz-cpp` is off by default. A missing `gvd_viz` prints one note and the publisher keeps sending, which is harmless. `publish` catches everything and only counts errors. The app is started with `Popen` (no wait) and terminated at exit (`finally` + `atexit`).

## Message schema v1

One JSON object per datagram / per line (offline file), ≤ 60 000 bytes. Points are thinned to fit. Ego frame: x right, y forward, metres. `yaw` is radians from +x (π/2 = same heading as ego).

```json
{"v":1,"type":"gvd_scene","seq":12,"t":1234.5678,
 "ego":{"engaged":true,"speed_mps":13.4,"set_mps":15.0},
 "lead_id":4,"aeb":"off","path_conf":0.9,
 "lanes":[{"pts":[[-1.8,-10],[-1.8,80]],"paint":"yellow|white","kind":"detected|predicted"}],
 "edges":[{"pts":[[5.5,-10],[5.5,80]],"kind":"predicted"}],
 "tracks":[{"id":4,"cls":"vehicle","x":0,"y":15,"yaw":1.57,"l":4.5,"w":1.85,"h":1.5,"v":10,"lead":true}],
 "signs":[{"cls":"traffic_light|stop_sign|pole","x":0,"y":45,"state":"red|yellow|green|unknown"}],
 "path":{"pts":[[0,2],[0,30]]}}
```

Sources (the only state keys read): `engaged`, `ego.speed_mps`, `planner.{target_v,cipv_id,aeb}`, `path_ego`, `path_conf`, `lanes_ext`, `road_edges`, `tracks`, `signs`. The C++ parser rejects `v != 1` or `type != gvd_scene`, and reads only the fields above. Track `l/w/h` are optional, with class priors filling them in. Caps: 64 tracks, 16 signs, 16 lanes, 8 edges, 120 points per polyline.

## File touch list

- New: `cpp/viz/**` (CMakeLists, presets, `src/{scene,renderer,context,udp,image_io,main}`, `tests/test_render.cpp`, `tests/run_cli.cmake`, `tests/scenes/basic.jsonl`, vendored `third_party/{glad,stb}`), `python/viz/cpp_publisher.py`, `scripts/test_viz_cpp_publisher.py`, `scripts/test_viz_cpp_render.py`, `scripts/render_viz_cpp_gallery.py`, this spec.
- Changed: `python/run_vision.py` (flags `--viz-cpp`, `--viz-cpp-port`, `--viz-cpp-exe`, `--viz-cpp-no-launch`; `start_cpp_viz`; one `publish(st)` per tick; `close()` in `finally`), `.github/workflows/tests.yml` (apt cmake/ninja/g++/Mesa EGL, build + ctest step, `GVD_VIZ_BUILD_DIR` for the suite), README.

## Verify

1. `cmake --preset linux-release && cmake --build --preset linux-release && ctest --preset linux-release` (Linux), or `cmake --preset windows-msvc && cmake --build --preset windows-msvc && ctest --preset windows-msvc` (Desktop-IQU45).
2. `python scripts/test_viz_cpp_publisher.py`: schema, whitelist (a recording dict proves no other key is read), GT keys ignored, paint rule, non-blocking with no listener and with a receiver that never reads, oversize, fail-safe, flag off by default.
3. `python scripts/test_viz_cpp_render.py`: builds (or reuses `GVD_VIZ_BUILD_DIR`), runs ctest, then publisher → file → `gvd_viz` PNG → pixel checks. It skips visibly when the host has no CMake/compiler.
4. `gvd_viz --offline f.jsonl --bench 300 --size 1280x720` prints wall/GPU ms (GL timer query) and the GL bytes the renderer allocated.
5. Live (IQU45): `python -m python.run_vision --viz-cpp ...`. Check that the loop Hz is unchanged with and without `gvd_viz` running, and that killing `gvd_viz` mid-drive changes nothing in the supervisor.

## Milestones

- **M1 (this PR):** skeleton, schema v1, publisher + flag, lanes/edges/path beads, wireframe boxes + centre dot + glow halo, ego mesh + ground glow, traffic lights, stop sign + stop bar + zebra, speed digits + bar, 4x MSAA, headless EGL, ctest pixel tests, ref compare sheets.
- **M2:** separable bloom pass (half-res), 60 fps interpolation between perception updates (track id matching + lane/path blending with message `t`), dashed vs solid from lane `style`, a better ego model (wheels, mirrors), measured VRAM/GPU on the 1080 Ti (`GL_NVX_gpu_memory_info`), live smoke on IQU45.
- **M3:** overlay on the main camera image (shared-memory frame or JPEG side channel, with main camera intrinsics), showcase exports (square/vertical/wide/split/PiP) and the label-review video. These replace #231's Python debug board, showcase and label video.
