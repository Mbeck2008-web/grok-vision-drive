# Reality View (3D Gaussian Splatting) spike

Status: **spec only**. Flag default **OFF**. No implementation on this
branch. Send to Skeptic before any code.

Base: `origin/main` `8f4a163` (2026-11-25). Host: Desktop-IQU45, GTX 1080 Ti
11 GB (Pascal, sm_61, no tensor / RT cores) shared with BeamNG; vision
inference cap 4 GB (`config/hardware.yaml`); supervisor RSS > 12 GB is a bug.
The box has no GPU; GPU figures in this note are cited or marked unmeasured.
IQU45 proves are scheduled by GVD, not by this spike.

This narrows GVD 2.0 Phase 6 (offline static 3DGS prototype) from
`/workspace/gvd2/spec.md`. It does not implement Phase 7 (dynamic splats),
Phase 8 (full C++ integration beyond a contract), or Phase 9 (refinement net).

## Goal

The left pane of **GVD VISION** (already named Reality View, PR #242) shows a
camera-only 3D reconstruction of the surroundings: a chase / drone-style view
built from the model's own cameras, intrinsics / extrinsics, ego pose, and
optional *predicted* depth, not from boxes or lane scribbles alone. Until the
renderer exists, the pane stays the current placeholder
(`python/viz/view_select.py`). The classic ice-blue ribbon stage remains a
disabled option (`viz.classic_view_enabled`).

## Non-goals

- No control use: Reality View is viz / perception-side only. It must not
  feed AEB, engage, heartbeat, or the planner.
- No live lidar or sim depth / semantic / future-state GT. Recorded sim GT
  may be used **offline** for training and eval only, and any GT render must
  be labeled GT.
- No live heavy nets (no pixelSplat / MVSplat / DepthSplat / DrivingForward
  / nerfstudio / YOLOv8s+ as a generator). YOLOv8n-class and the existing
  SharedStem only.
- No training of a big reconstruction or diffusion / mean-flow refiner in
  this spike. The optional later residual refiner stays Phase 9.
- No raising `max_inference_vram_gb`. No new CUDA context in the viz
  process. No Unreal. Do not overload the gvd_viz UDP scene datagram
  (`docs/specs/viz_cpp.md`: 60 kB cap).

## What it must guarantee

**Budgets** (proposed gates; measure on IQU45 before calling them proven):

| Budget | Proposed gate | Cited vs unmeasured |
| --- | --- | --- |
| Vision inference | Stay ≤ **4.0 GB**; do not raise the cap | **Cited:** `config/hardware.yaml` |
| RV *generation* (any new GPU net or splat-parameter head) | ≤ **1.0 GB** of that 4 GB, and (existing vision + RV) ≤ 4 GB | **Unmeasured.** Justification: SceneNet `small` paper estimate is 2.2 GB (`docs/research/gpu_tiers.md`, not a profile). A 1 GB slice leaves headroom if inference later moves off CPU. Approaches C and A-without-a-new-net should be ~0 GB generation. |
| RV *render* (gvd_viz / GL) | ≤ **300 MB** measured NVIDIA process memory | **Unmeasured** on IQU45. Arithmetic (Breaker Phase 0): ~40–45 MiB GL at 300k SH0 + 480², plus a commonly cited 100–300 MB GL context. Existing viz formula cap is 200 MB (`docs/specs/viz_cpp.md`), not measured GPU VRAM. |
| Supervisor RSS | Stay < **12 GB**. Flag-on delta in the supervisor ≤ **256 MB** vs flag-off | 12 GB is an existing bug threshold (`docs/gvd_state_schema.md`). The 256 MB delta is **proposed / unmeasured**. |
| Separate viz process RSS | ≤ **2 GB** | **Proposed / unmeasured.** |
| Viz rate / latency | Drop frames rather than stall. Aim to refresh at 10–15 Hz (`target_hz_min` / `target_hz_happy`). p95 render < 33 ms; mark the view stale if photon age > 200 ms | **Unmeasured.** 200 ms age is Breaker's proposed indicator, not a measurement. CPU sort of 300k keys was 2.1 ms on the box Xeon (AVX-512), not on the i9-9900K. |
| Safety loop | loop Hz and `infer_ms` p95 change < **5%** with RV on vs off (synthetic 300k load is enough) | **Proposed** (Breaker Q5), **unmeasured.** |

**Isolation.** Heartbeat / AEB / engage are sacred. The renderer is a
separate process (preferred: extend `gvd_viz`) or a lower-priority thread.
It is read-only to control: no back-channel. Bounded shared memory or
binary IPC, non-blocking, stale-frame tagged; the 60 kB UDP scene message
stays for lanes / tracks / HUD only. Work on the supervisor tick, if any,
must sit *after* `actuator.apply` and must drop on backpressure. Never
block a camera grab or the heartbeat stamp.

**Degrade / disable.** Flag default OFF: `viz.reality_view_enabled: false`
(literal true only) or `--reality-view`. Flag off = today's placeholder,
bit-identical. On failure, missing predicted depth, over-budget, or a
crashed viz process: fall back to the placeholder, print one note, keep
driving. It is never an error and never a disengage.

**Determinism (offline).** Same stub / recorded frames, K, pose, and
(optional) predicted depth → bit-identical RGB (and optional depth)
on CPU. No wall clock, no RNG, no sim GT unless the test is named GT.

**Honesty.** Unobserved or unknown space is never painted as a verified
surface (hatch / fog / empty). Dynamic vehicles must not ghost into a
static map. Geometry is authoritative if a later refiner appears
(out of scope here).

## Candidate approaches

BEV LSS (`python/train/bev_lss.py`) is **not on main** (draft PR #241).
"Existing depth" means a predicted depth / LSS head when it lands, or a
stub / recorded *predicted* tensor in tests. Sim depth is eval-only.

| | A. Feed-forward SH0 splats + Pascal-safe rasterizer | B. Incremental online GS map (MonoGS / SplaTAM-style) | C. Pseudo-splat (depth billboards / point cloud) |
| --- | --- | --- | --- |
| Idea | Unproject predicted depth × RGB (and later LSS / spatial-path features) to SH0 Gaussians; optional EMA merge into a bounded rolling map; render in `gvd_viz` as GL 3.3 CPU-sorted instanced quads (antimatter15-style). No live SGD. | Keep and optimize a persistent Gaussian map from multi-cam + ego pose (photometric residuals, densify / prune). | Unproject depth to points or camera-colored billboards; OpenCV / numpy (or the existing GL path) only. |
| Pros | Matches the GVD 2.0 "model-generated splats" path; SH0 map is small; extends #232; sort can stay on CPU (off the shared GPU); mistakes stay visible. | Best temporal consistency *if* it fits; fills unseen sides over time. | Zero new deps; runs on the box CPU; proves the IPC / flag / isolation contract; honest degrade path. |
| Cons | Raw drone-view quality from a small stem may look like sparse confetti (Phase 0 risk). Rolling merge needs a pose-drift gate. LSS is not on main yet. | Published systems were timed on 3080 Ti / 4090 / A100, not a 1080 Ti next to BeamNG. Live optimization hides model error and fights the 4 GB cap. | Holes, no soft splat fill, weaker novel-view look. |
| Live deps | None required. Optional later: none in the viz process. | Typically a CUDA rasterizer + a training loop (gsplat / INRIA). | numpy, OpenCV (already in `requirements-viz.txt`). |
| Pascal / Windows | GL 3.3 path: `gvd_viz` already has a `windows-msvc` preset. **gsplat ≥1.0 is out on sm_61** (maintainer: needs CC ≥7.0, `cg::labeled_partition`). | Same rasterizer block. Indoor SLAM papers are not a surround-driving prove on Pascal. | No GPU ISA risk. |
| VRAM | Generation: unmeasured (0 if no new net). Render arithmetic ~40–45 MiB + context (unmeasured). | SplaTAM peak **14 GB** GPU on Replica (survey on A100). Photo-SLAM **4–6 GB** on Jetson / 3080 Ti laptop / 4090 — already the whole vision cap, indoor, not shared with BeamNG. MonoGS ~3 FPS on a 4090 (author). | Unmeasured; should be small (point buffers + one RGB). |

Heavy feed-forward papers are **not** a fourth live candidate: DrivingForward
reports **7.58 GB** inference (MF; pixelSplat 11.3 GB, MVSplat 5.45 GB);
DepthSplat quotes 12×512×960 in **0.6 s** (no VRAM figure here). Those nets
do not fit the 4 GB / no-heavy-net rule. Use them only as citations.

## Recommendation

**A**, with **C as the first spike and the degrade path.** Kill **B** as a
live approach.

Trade-off: A is the product shape (model-generated SH0 Gaussians, optional
bounded rolling merge, hybrid with road mesh / vector lanes / metric
objects so splats are never the only truth) but it will not look
photoreal on day one and its latency / VRAM are unmeasured on the 1080 Ti.
C is the only path that can be proven on this box this week, with no new
deps, and it stays as the fallback if A misses a go/no-go gate. B's
published memory and hardware are incompatible with a 4 GB Pascal slice
shared with BeamNG, and a live optimizer would hide the model's mistakes.

Live rasterizer: our own GL 3.3 CPU-sorted instanced-quad pass inside
`gvd_viz` (Phase 0 Breaker verdict). Do not ship nerfstudio or gsplat ≥1.0.
gsplat 0.1.12 or INRIA `diff-gaussian-rasterization` are **offline teacher
fits only**, and only after an IQU45 build prove (both unofficial on
sm_61). INRIA is a **research / non-commercial** license (see appendix);
do not add it to the runtime tree.

## Interface

**Inputs** (from the supervisor, copies only): camera frames for the live
rig (8-cam: `narrow`, `main`, `wide`, `pillarL`, `pillarR`, `repeatL`,
`repeatR`, `rear`; a 3-cam subset is the windshield cluster `main` /
`narrow` / `wide`). Long side ≤ 640. Intrinsics / extrinsics from
`config/cameras.yaml` (pinhole K from `fov_v` + resolution until a
calibrated K exists). Ego pose: the live default (VO); do not add wheel
odometry or sim pose as a live input. Optional predicted depth / splat
parameters from the existing spatial path or LSS when present. Capture
backends stay `beamngpy` | `window` | `stub`.

**Outputs:** an RGB image for the left GVD VISION pane (and optional
rendered depth). Chase / drone camera is the product viewpoint; a
recording-only `chase_camera` already exists in `cameras.yaml` for
offline supervision, and is not a live sensor.

**Plug-in:** `viz.reality_view_enabled` / `--reality-view`, default **OFF**.
`viz.main_view` stays `reality` (placeholder) regardless. Preferred
process boundary: vision process writes a double-buffered splat or
billboard block to shared memory; `gvd_viz` maps the newest complete
slot and composites over the existing vector / ego layers. An in-process
OpenCV C path is allowed only for the CPU spike and must still drop
frames and stay off the safety path.

## Feasibility plan

Each step has a go / no-go gate. Steps 1–3 and 5 run offline on the box
CPU (stub or recorded frames). Steps 4 and 7 are IQU45 proves that GVD
schedules; this swarm does not touch Desktop-IQU45.

1. **Contract and flag pin.** Schema for inputs / outputs / SHM slot;
   flag default OFF; placeholder bit-identical to `8f4a163`; no safety-path
   imports. **Go:** fail-first offline tests pass. **No-go:** flag-off
   pixels or tick path change.
2. **C prototype (CPU).** Unproject depth to points / billboards from a
   chase pose; GT depth only in tests named GT. **Go:** deterministic PNG
   pin; unknown cells not filled as surface. **No-go:** GT used on a live
   path, or a new dependency.
3. **Isolation harness.** A deliberately slow renderer cannot move a stub
   control-loop period outside the 5% gate; killing the viz process leaves
   heartbeat / AEB / engage unchanged. **Go:** the test fails if the
   renderer is on the control thread. **No-go:** any stall of the loop.
4. **Rasterizer prove (IQU45, GVD-scheduled).** Build/smoke gsplat 0.1.12
   and INRIA with `TORCH_CUDA_ARCH_LIST=6.1` on Windows (offline only);
   record GL version via `gvd_viz --info`. **Go:** at least the GL 3.3
   path is accepted for live A. CUDA rasterizers may fail; that is not a
   live blocker. **No-go:** if even GL 3.3 cannot composite a 50k splat
   or billboard pass under the 300 MB / contention gates, stay on C.
5. **A generator (CPU first).** SH0 Gaussians from predicted depth + RGB
   + K, cap N, no new heavy net. **Go:** offline render from a canned
   pose; honesty pin (unknown not solid); N cap enforced. **No-go:** a
   new live net or a raised VRAM cap.
6. **Optional rolling merge.** Voxel-hash EMA in a bounded region (e.g.
   ±64 m, N_max 300k as a working cap, not a measured overdraw). Evict by
   age / distance. Dynamic boxes never enter the static map. **Go:** a
   canned moving-vehicle clip (GT eval only) does not ghost. **No-go:**
   if VO drift ghosts the map, shrink radius / age or skip merge and
   keep per-frame A.
7. **IQU45 system prove (GVD-scheduled).** VRAM, RSS, `infer_ms`, loop
   Hz, BeamNG FPS, RV on vs off, with BeamNG running. **Go:** all budget
   and isolation gates hold. **No-go:** any missed gate ships C only, or
   keeps the placeholder.
8. **Kill or graduate.** If C from GT depth is not a useful 3D view at
   the chase pose, drop the photoreal claim and keep the placeholder +
   classic. If A misses budgets, C stays the live path. If both fail
   isolation, Reality View stays a placeholder.

## Acceptance criteria (eventual implementation)

Fail-first offline tests (box CPU) before any IQU45 run:

- `check_flag_off_is_placeholder`: flag off matches `8f4a163` placeholder
  pixels on a pinned stub state.
- `check_no_sim_gt_on_live_path`: the live renderer API rejects or ignores
  sim-depth keys; a GT fixture is only consumed by a test named GT.
- `check_unknown_not_free`: cells marked unknown / unobserved are not
  opaque surface.
- `check_deterministic_cpu`: two runs on the same stub frames are equal.
- `check_n_cap`: a generator or map cannot exceed N_max.
- `check_safety_loop_latency_unchanged`: stub loop Hz and a stand-in
  `infer_ms` p95 move < 5% with a slow or 300k-load renderer; a blocked
  renderer is dropped, not waited on.
- `check_renderer_crash_degrades`: killing the viz process leaves
  heartbeat / AEB / engage and the placeholder path intact.
- `check_vram_rss_hooks`: the same counters as `scripts/test_vram_cap.py`
  and `rss_mb` still apply; flag-on does not raise the 4 GB cap. Numeric
  VRAM / RSS gates are asserted on the IQU45 prove (step 7), not faked
  on the box.

## Risks, open questions, kill criteria

Top risks: (1) WDDM time-slicing on Pascal steals SMs from BeamNG or
vision (memory is not the likely failure; this is unmeasured). (2)
Predicted depth / LSS is not on main, and a small stem may not support a
useful drone view. (3) Ego-pose drift and inter-cam skew (Tracer Phase 0:
median 34.4 ms, max 79.8 ms) ghost a rolling map.

Open questions (from Phase 0, still open): Q3 sm_61 CUDA rasterizer
build on Windows; Q5 contention vs BeamNG; Q6 whether WDDM / priority
can deprioritize `gvd_viz`; live VO drift vs a ±64 m map.

**Kill** the live A/B path if: a new dep needs CC ≥7.0; generation +
existing vision exceeds 4 GB; the safety-loop test fails; INRIA / a
non-commercial rasterizer is required at runtime; or a heavy feed-forward
net is the only way to get a useful image. **Kill** the photoreal claim
(keep placeholder) if step 8 fails.

---

## Appendix A — Prior art and citations

Feed-forward / driving: pixelSplat (CVPR 2024);
[MVSplat](https://arxiv.org/abs/2403.14627);
[DepthSplat](https://arxiv.org/abs/2410.13862) (CVPR 2025; 12 views
512×960 in 0.6 s);
[DrivingForward](https://arxiv.org/abs/2409.12753) (inference 0.63 s,
**7.58 GB** MF; pixelSplat 11.3 GB; MVSplat 5.45 GB);
[DrivingScene](https://arxiv.org/abs/2510.24734);
Street Gaussians / DrivingGaussian / OmniRe / EmerNeRF-class (per-scene
optimized, not a 4 GB live recipe).

Online / SLAM GS: [MonoGS](https://github.com/muskie82/MonoGS) (~3 FPS
on a 4090, author); SplaTAM (survey peak **14 GB** GPU on Replica / A100,
[Pangaliman et al., CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/WiCV/papers/Pangaliman_A_Survey_of_Spatial_Memory_Representations_for_Efficient_Robot_Navigation_CVPRW_2026_paper.pdf));
[Photo-SLAM](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Photo-SLAM_Real-time_Simultaneous_Localization_and_Photorealistic_Mapping_for_Monocular_Stereo_CVPR_2024_paper.pdf)
(**4 GB** Jetson / laptop, **5–6 GB** desktop 4090, Replica, not shared
with a game).

Rasterizers: [gsplat #401](https://github.com/nerfstudio-project/gsplat/issues/401),
[#226](https://github.com/nerfstudio-project/gsplat/issues/226) (maintainer:
gsplat ≥1.0 needs CC ≥7.0); gsplat 0.1.12 reported on a 1080 Ti (user,
#401, unofficial);
[INRIA gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(README: CC 7.0+, OpenGL 4.5 viewer, 4 GB VRAM recommended; one user still
runs the rasterizer on a 1080 Ti);
[antimatter15/splat](https://github.com/antimatter15/splat) (WebGL 1, CPU
sort in a worker; ~4 fps sort vs 60 fps render; ~1M splats ~150 ms sort).

GVD grounding (read-only, not in this repo tree): `/workspace/gvd2/spec.md`,
`/workspace/gvd2/audit/redesigner.md` §13, `/workspace/gvd2/audit/breaker.md`
§14–15, 19–20, `/workspace/gvd2/audit/tracer.md` (sync / depth / pose).
In-repo: PR #242 (`python/viz/view_select.py`, README UI guide),
`docs/specs/viz_cpp.md`, `config/cameras.yaml`, `config/hardware.yaml`,
`docs/research/gpu_tiers.md`. `gh issue list --search "Reality View"` was
empty; the only merged Reality View change found is PR #242. PR #241
(BEV LSS) is still a draft and is not on `8f4a163`.

## Appendix B — Candidate dependencies (none required for C or live A)

| Dep | License | sm_61 | Windows | Size / note |
| --- | --- | --- | --- | --- |
| *(none)* for C and live A | — | n/a | `gvd_viz` already builds | Prefer this. |
| antimatter15/splat (reference only) | MIT | n/a (GL) | n/a | Do not import; copy the *idea* (CPU sort + instanced quads). |
| gsplat ≥1.0 / nerfstudio | Apache-2.0 | **No** | Wheels exist for gsplat 1.x (e.g. pt2.4/cu12.4) but not for sm_61 | **Heavy. Forbidden live.** |
| gsplat 0.1.12 | Apache-2.0 | User report only (unofficial) | **Unmeasured** (current wheels are 1.x) | Offline teacher only, after IQU45 smoke. |
| INRIA `diff-gaussian-rasterization` / SIBR | Gaussian-Splatting License: research / non-commercial; commercial use needs Inria consent (`LICENSE.md`) | Official CC 7.0+; unofficial 1080 Ti report | Unmeasured | **Do not add to the runtime.** Offline teacher only if Q3 passes *and* Michael accepts the license. |
| Community gsplat CC 6.x fork (e.g. abhigyan7) | Check fork license | Claimed (GTX 1050 Ti) | Unmeasured | Not a runtime dep; do not vendor without a license/review pass. |

No new runtime dependency is proposed for the recommended path. Any later
offline teacher dep must be called out as dev-only and must not be imported
by `run_vision.py`.
