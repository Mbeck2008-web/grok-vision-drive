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
- high chase camera (default / `--lens chase`: eye 0,-32,10 m, look 0,40,0 m, vfov 40°; ego in the lower third, ~3 lanes and a long look-ahead at 16:9)

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

- `cpp/viz/`: CMake ≥ 3.20, C++17. glad 2 (GL 3.3 core + EGL 1.5 runtime loader) and stb are vendored in `third_party/`. glm 1.0.1 and nlohmann_json 3.11.3 **always** come from pinned FetchContent tarballs (SHA-256 checked); system packages are not used. GLFW uses a system 3.3+ package if CMake finds one, otherwise the pinned 3.4 tarball. GLFW is on by default on Windows and on Linux when X11 dev headers are present.
- Headless: EGL surfaceless (Mesa) on Linux, so CI and the box render on llvmpipe with no X server. On Windows, headless uses a hidden GLFW window. `GLFW_VISIBLE` is set **after** `glfwInit` (GLFW 3.4 drops pre-init hints, and `glfwInit` resets them).
- Presets: `linux-release` (Ninja, EGL) and `windows-msvc` (VS 2022 x64, GLFW).
- Rendering: one instanced "capsule" primitive (screen-space segment SDF with an analytic glow halo, additive) for lane beads, curb dots, path beads and box edges. Plus a flat-shaded low-poly ego mesh (depth-tested first so lines under it hide), ground quads (stop bar, zebra, ego glow), SDF sprites (traffic light, stop sign) and an SDF HUD. 4x MSAA FBO, resolved, then read back (offline) or blitted (live). Internal render-target size and MSAA are capped so allocated GL memory stays under 200 MB (MSAA is reduced first, then resolution). A 4K window is blitted from the capped internal target.
- Fail-safe: `--viz-cpp` is off by default. A missing `gvd_viz` prints one note and the publisher keeps sending, which is harmless. `publish` catches everything and only counts errors. The app is started with `Popen` (no wait) and on `close()` is `terminate()`d, waited, and reaped (with a `kill` fallback).
- Live mode drains the UDP socket every frame and keeps only the newest datagram. If no valid message has arrived for more than 1 second (`scene_is_stale`), the view blanks to an empty road plus ego.

## Message schema v1

One JSON object per datagram / per line (offline file), ≤ 60 000 bytes (hard UDP payload limit 65 507). Points are thinned to fit. Ego frame: x right, y forward, metres. `yaw` is radians from +x (π/2 = same heading as ego).

The on-wire schema is **strict**: exact keys, strict enums, JSON integers only for ids. Unknown keys are ignored (so a stray GT field can never be drawn). Invalid enums on a lane, edge, track or sign drop that element, not the whole message, except `aeb` (an invalid `aeb` rejects the message). Caps are enforced on **both** the Python publisher and the C++ parser.

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

### Fields

| Field | Rules |
| --- | --- |
| `v` | Integer, must be `1`. Missing or any other value: message rejected. |
| `type` | Must be exactly `"gvd_scene"`. |
| `seq` | JSON integer (publisher: non-negative, monotonic per publisher instance, increments only when a datagram is actually sent). Wrong type is ignored (stays 0). |
| `t` | Number of seconds, finite. Live publisher uses `time.monotonic()`, rounded to 4 decimal places. Offline `write_scene_file` uses `frame_index / 30`. Wrong type / non-finite is ignored (stays 0). |
| `ego.engaged` | Boolean, default `false`. |
| `ego.speed_mps` | Number, default 0. |
| `ego.set_mps` | Number, optional; omitted when the planner has no target. `< 0` means none. |
| `lead_id` | JSON integer only (no floats, no strings). Identifies the CIPV. Optional; missing or invalid is `-1` (no lead). |
| `aeb` | Enum: `"off"` \| `"warn"` \| `"brake"`. Missing defaults to `"off"`. Any other value **rejects the message** (C++) or is emitted as `"off"` (publisher, so it never sends an invalid value). |
| `path_conf` | Number in `[0, 1]`, **default `1.0`** when missing. Both sides clamp into `[0, 1]`. The renderer further floors the path glow at 0.45 so a very low conf is still visible. |
| `lanes[].paint` | Enum: `"white"` \| `"yellow"`. Exact key `paint` and exact value only. Missing → `"white"`. Any other value (including `"Yellow"`, `"not_yellow"`, aliases like `paint_color` / `colour` / `line_color` / `color` / `material`, or `yellow: true`) is **rejected** (the lane is dropped; it is not treated as yellow). |
| `lanes[].kind`, `edges[].kind` | Enum: `"detected"` \| `"predicted"`. Missing → `"detected"`. Any other value drops that lane/edge. Non-`"predicted"` is no longer silently treated as detected. |
| `lanes[].pts`, `edges[].pts`, `path.pts` | Array of `[x, y]` number pairs, key **`pts` only** (not `points`). Non-finite points skipped. |
| `tracks[].id` | JSON integer only. Floats and strings are rejected (the track is dropped if `id` is present and not an integer; a missing `id` is allowed and stays `-1`). |
| `tracks[].cls` | Enum: `vehicle`, `car`, `pedestrian`, `ped`, `person`, `bicycle`, `bike`, `cyclist`, `motorcycle`, `truck`, `bus`, `trailer`, `obstacle`, `parked`, `debris`, `cone`. Missing → `"vehicle"`. Any other value drops the track. On-wire key is `cls` only. |
| `tracks[].l/w/h/v` | Optional numbers (class priors fill in `l/w/h` in the renderer). On-wire keys are `l`, `w`, `h`, `v` only. |
| `tracks[].lead` | Optional boolean. C++ treats a track as lead if `lead` is true **or** `id == lead_id`. The live publisher does **not** emit `lead` (lead is indicated by `lead_id` from `planner.cipv_id` only). Offline / `occ_to_viz` may set `lead` if they want. |
| `signs[].cls` | Enum: `traffic_light` \| `stop_sign` \| `pole`. Anything else drops the sign. |
| `signs[].state` | Enum: `red` \| `yellow` \| `green` \| `unknown`. Missing → `"unknown"`. Anything else drops the sign. |

Caps (extra elements / points dropped, both sides): **64** tracks, **16** signs, **16** lanes, **8** edges, **120** points per polyline.

### Supervisor-state input (publisher only, not on the wire)

The publisher still reads supervisor state keys: `engaged`, `ego.speed_mps`, `planner.{target_v,cipv_id,aeb}`, `path_ego`, `path_conf`, `lanes_ext`, `road_edges`, `tracks`, `signs`. Ground truth (`path_world`, `debug_gt`, `*_gt` / `gt_*`) is never read.

These **input** aliases are mapped onto the strict on-wire keys and then discarded. They must not appear in the datagram:

- track `class` → `cls`; `length`/`width`/`height`/`speed_mps` → `l`/`w`/`h`/`v`
- lane / edge / path `points` or `pts`, list or `{points|pts: ...}` dict → on-wire `{pts: [[x,y], ...]}`

`occ_to_viz` (and any other adapter writing schema v1) should emit the **on-wire** keys and enums above, not the supervisor aliases.

### Contract changes in this revision (for `occ_to_viz`)

Strict where the first draft was loose. Adapters must:

1. Send `paint` as exactly `"white"` or `"yellow"` (no aliases, no substring match, no `yellow: true`). Missing paint is white.
2. Send `kind` as exactly `"detected"` or `"predicted"`.
3. Send `aeb` as exactly `"off"`, `"warn"` or `"brake"`.
4. Send `tracks[].cls` and `signs[].cls` / `state` from the enums above.
5. Send integer `id` / `lead_id` / `seq` (no floats, no numeric strings).
6. Send `path_conf` in `[0, 1]` or omit it (default 1.0).
7. Use on-wire keys `cls`, `l`, `w`, `h`, `v`, `pts` (not `class`, `length`, `points`).
8. Honour the caps; the C++ parser now truncates the same way the publisher does.
9. Not rely on `tracks[].lead` being present in a live publisher datagram; set `lead_id` (and optionally `lead` if you want).
10. Not include `path_world` or other GT keys for drawing; they are ignored if present.

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
