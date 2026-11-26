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
*predicted* depth, not from boxes or lane scribbles alone. Until a live
predicted-depth source exists on main *and* a renderer exists, the live pane
stays the current PR #242 placeholder (`python/viz/view_select.py`). The
classic ice-blue ribbon stage remains a disabled option
(`viz.classic_view_enabled`).

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
- This spike does **not** add a live depth head, enable train-only
  `aux_depth` at runtime, or treat draft PR #241 as if it had landed.

## Live depth status (main `8f4a163`)

There is **no live predicted-depth tensor on main `8f4a163`.**
`python/train/bev_lss.py` is absent (draft PR #241, not merged). SceneNet
`aux_depth` / `aux_sem` are **train-only** (`docs/ROADMAP.md`). There is no
other "existing spatial path" depth export on the live net. Sim depth is
recording / eval only (`config/cameras.yaml`: live depth OFF).

Until a live predicted-depth source exists on main, **A and C are
stub / recording-only** (fixtures, or sim GT in tests named GT). The **live
fallback is the PR #242 placeholder**, not C. C is not a live degrade path
while it depends on the same missing depth.

**If #241 never lands:** this spike does not invent a live depth source and
does not graduate a live A/C view. Reality View stays the placeholder
(classic remains a disabled option). A later, separately gated live depth
head on the existing SharedStem (YOLOv8n-class, default OFF, counted in the
generation budget below) is GVD's call and a new spec, not work in this
spike. Step 0 below is the gate: no live depth on main → offline-only +
placeholder.

## What it must guarantee

**Budgets** (proposed gates; measure on IQU45 before calling them proven).
Each row has a go / no-go fallback. Sort and render run in the **viz
process**, never in the supervisor.

| Budget | Proposed gate | Fallback if missed | Cited vs unmeasured |
| --- | --- | --- | --- |
| Vision inference | Stay ≤ **4.0 GB**; do not raise the cap | Kill any new generation net and the live A/C path; keep the placeholder. Never raise the cap. | **Cited:** `config/hardware.yaml` |
| RV *generation* (any new GPU net or splat-parameter head) | Soft cap ≤ **1.0 GB** of that 4 GB; hard cap (existing vision + RV) ≤ 4 GB | Over 1 GB but still ≤ 4 GB total: not a kill; shrink the head, drop the new net, or stay on no-new-net A/C (offline until live depth exists). Over 4 GB total: kill the generation net; if A/C cannot run without it, placeholder. | **Unmeasured.** Justification: SceneNet `small` paper estimate is 2.2 GB (`docs/research/gpu_tiers.md`, not a profile). A 1 GB slice leaves headroom if inference later moves off CPU. A/C with no new net should be ~0 GB generation. |
| RV *render* (whole `gvd_viz` process) | ≤ **300 MB** measured NVIDIA process memory for the **whole** `gvd_viz` process, not an increment on the 200 MB formula cap | Reduce N, then target size / MSAA; if still over, do not graduate live A. If C also misses, placeholder. | **Unmeasured** on IQU45. Arithmetic (Breaker Phase 0): ~40–45 MiB GL at 300k SH0 + 480², plus a commonly cited 100–300 MB GL context. See "Render cap vs `viz_cpp.md`" below. |
| Supervisor RSS | Stay < **12 GB**. Flag-on delta in the supervisor ≤ **256 MB** vs flag-off | Over 12 GB: kill live RV, placeholder. Over 256 MB delta: shrink SHM / copies; if still over, placeholder. | 12 GB is an existing bug threshold (`docs/gvd_state_schema.md`). The 256 MB delta is **proposed / unmeasured**. |
| Separate viz process RSS | ≤ **2 GB** | Reduce N / buffers; if still over, placeholder. | **Proposed / unmeasured.** |
| Viz rate / latency | Drop frames rather than stall. **Go:** update ≥ **10 Hz** when frames are available, and **p95 sort+render < 33 ms in the viz process** (sort is included; see sort note). **Aim** 15 Hz (`target_hz_happy`). If photon age > **200 ms**, the view **must** show a stale indicator. | First lower N until the 33 ms / 10 Hz gates hold. Then allow update down to 5 Hz with the stale indicator still required. Then, only if a live depth source exists, try C at the same gates. Then placeholder. Never stall and never move sort/render onto the supervisor. Persistent age p95 > 200 ms after those cuts is a no-go for that configuration. | **Unmeasured** on the i9-9900K. 200 ms age is Breaker's proposed indicator, not a measurement. |
| Safety loop | loop Hz and `infer_ms` p95 change < **5%** with RV on vs off (synthetic 300k load is enough) | Kill the live RV path (placeholder). Isolation failure is not retried at a lower Hz. | **Proposed** (Breaker Q5), **unmeasured.** |
| BeamNG / WDDM | BeamNG FPS drop ≤ **5%** RV on vs off (same relative gate as safety Hz). If BeamNG was ≥ 30 fps with RV off, RV must not drive it under 30 (the existing downsize trigger in `docs/research/gpu_tiers.md`). If it was already under 30 with RV off, only the 5% relative gate applies. | Lower N / viz GPU work, then lower RV Hz as in the rate row; if still over, disable live RV (placeholder). | **Proposed / unmeasured.** |

**Sort note (not a host number).** A 16-bit radix sort of 300k keys took
2.1 ms on the **box Xeon (AVX-512)**. That is not a measurement on the
i9-9900K and must not be used as the host budget. Whether sort+render
fits inside p95 < 33 ms on the 9900K is **unmeasured** and is an IQU45
gate (`check_iqu45_sort_render_p95`). The antimatter15 README figure of
~150 ms to sort ~1M splats in JS is not a GVD budget; live N is capped
(working cap 300k, unmeasured overdraw) and the sort lives only in the
viz process.

**Render cap vs `viz_cpp.md`.** `docs/specs/viz_cpp.md` caps allocated GL
memory at **200 MB** via a CPU-side `gl_bytes()` formula (FBO / MSAA /
targets), "not a measured GPU VRAM figure," and it does not yet include
splat or billboard buffers. The **300 MB** gate here is measured NVIDIA
memory for the **whole `gvd_viz` process** on IQU45 (existing targets +
splat/billboard buffers + driver/context), not 200 MB plus a 100 MB
increment. The 200 MB formula cap stays in force for the existing
MSAA/FBO path; this spike does not silently raise it. Splat/billboard
allocations get their own formula line, and the **sum** of formula
allocations (existing + RV) is proposed to stay ≤ 200 MB unless a later,
explicit edit to `viz_cpp.md` raises that formula after an IQU45
measurement. Missing the measured 300 MB process gate is a no-go even if
the formula sum is under 200 MB. The 100 MB gap is proposed headroom for
unmeasured context/driver cost (Breaker: commonly cited 100–300 MB,
unmeasured).

**Isolation.** Heartbeat / AEB / engage are sacred. The **live** renderer
is a **separate process only** (preferred: extend `gvd_viz`). There is no
live "lower-priority thread" and no in-process OpenCV / numpy renderer on
a supervisor thread (a crash there would take heartbeat / AEB / engage
with it). Offline C prototypes and unit tests may use OpenCV in a **test
process**, not inside `run_vision.py`. The renderer is read-only to
control: no back-channel. Bounded shared memory or binary IPC,
non-blocking, stale-frame tagged; the 60 kB UDP scene message stays for
lanes / tracks / HUD only. After `actuator.apply`, supervisor work
is a non-blocking **memcpy / handoff only** (already-packed splat or
billboard block, or a pointer to a slot the infer process already wrote)
and must drop on backpressure. Unproject and splat / billboard packing
happen in the **infer process or the viz process**, never in the
supervisor. Never block a camera grab or the heartbeat stamp. Sort and
rasterize only in the viz process.

**Degrade / disable.** Flag default OFF: `viz.reality_view_enabled: false`
(literal true only) or `--reality-view`. Flag off = today's placeholder,
bit-identical. On failure, over-budget, a crashed viz process, **or
missing live predicted depth**: fall back to the **PR #242 placeholder**,
print one note, keep driving. It is never an error and never a disengage.
C becomes a live degrade only after a live depth source exists on main
and C itself has passed the isolation and budget gates.

**Determinism (offline).** Same stub / recorded frames, K, pose, and
depth tensor → bit-identical RGB and rendered depth on CPU. No wall
clock, no RNG, no sim GT unless the test is named GT.

**Honesty.** Unobserved or unknown space is never painted as a verified
surface (hatch / fog / empty). Dynamic vehicles must not ghost into a
static map. Geometry is authoritative if a later refiner appears
(out of scope here).

## Candidate approaches

On `8f4a163`, A and C have **no live depth source**. Their "predicted
depth" in the table is a stub / recorded tensor, or sim GT in tests named
GT, until step 0 passes. Sim depth is eval-only. There is no live
"spatial path" depth to call on.

| | A. Feed-forward SH0 splats + Pascal-safe rasterizer | B. Incremental online GS map (MonoGS / SplaTAM-style) | C. Pseudo-splat (depth billboards / point cloud) |
| --- | --- | --- | --- |
| Idea | Unproject predicted depth × RGB (and later LSS features, if #241 lands) to SH0 Gaussians; optional EMA merge into a bounded rolling map; render in `gvd_viz` as GL 3.3 CPU-sorted instanced quads (antimatter15-style). No live SGD. | Keep and optimize a persistent Gaussian map from multi-cam + ego pose (photometric residuals, densify / prune). | Unproject depth to points or camera-colored billboards; OpenCV / numpy in a **test process**, or the same `gvd_viz` path for a live-shaped build. |
| Pros | Matches the GVD 2.0 "model-generated splats" path; SH0 map is small; extends #232; sort stays on the viz-process CPU (off the shared GPU and off the supervisor); mistakes stay visible. | Best temporal consistency *if* it fits; fills unseen sides over time. | Zero new deps; runs on the box CPU; proves the IPC / flag / isolation contract. |
| Cons | Raw drone-view quality from a small stem may look like sparse confetti (Phase 0 risk). Rolling merge needs a pose-drift gate. No live depth on main today. | Published systems were timed on 3080 Ti / 4090 / A100, not a 1080 Ti next to BeamNG. Live optimization hides model error and fights the 4 GB cap. | Holes, no soft splat fill, weaker novel-view look. Same depth dependency as A. **Not** the live degrade path until a live depth source exists. |
| Live deps | None required. None in the viz process. | Typically a CUDA rasterizer + a training loop (gsplat / INRIA). | numpy, OpenCV (already in `requirements-viz.txt`). |
| Pascal / Windows | GL 3.3 path: `gvd_viz` already has a `windows-msvc` preset. **gsplat ≥1.0 is out on sm_61** (maintainer: needs CC ≥7.0, `cg::labeled_partition`). | Same rasterizer block. Indoor SLAM papers are not a surround-driving prove on Pascal. | No GPU ISA risk. |
| VRAM | Generation: unmeasured (0 if no new net). Render arithmetic ~40–45 MiB + context (unmeasured). | SplaTAM peak **14 GB** GPU on Replica (survey on A100). Photo-SLAM **4–6 GB** on Jetson / 3080 Ti laptop / 4090 — already the whole vision cap, indoor, not shared with BeamNG. MonoGS ~3 FPS on a 4090 (author). | Unmeasured; should be small (point buffers + one RGB). |

Heavy feed-forward papers are **not** a fourth live candidate: DrivingForward
reports **7.58 GB** inference (MF; pixelSplat 11.3 GB, MVSplat 5.45 GB);
DepthSplat quotes 12×512×960 in **0.6 s** (no VRAM figure here). Those nets
do not fit the 4 GB / no-heavy-net rule. Use them only as citations.

## Recommendation

**A** as the product shape, with **C as the first *offline* spike.** Kill
**B** as a live approach. The **live degrade is the PR #242 placeholder**
until a live predicted-depth source exists on main (step 0). After that
gate, C may become the live degrade if A misses a budget or quality gate.

Trade-off: A is the product shape (model-generated SH0 Gaussians, optional
bounded rolling merge, hybrid with road mesh / vector lanes / metric
objects so splats are never the only truth) but it will not look
photoreal on day one, its latency / VRAM are unmeasured on the 1080 Ti,
and it cannot go live on `8f4a163` because there is no live depth. C is
the only path that can be proven on this box this week, with no new deps,
but it has the same depth dependency and is not a live fallback today.
B's published memory and hardware are incompatible with a 4 GB Pascal
slice shared with BeamNG, and a live optimizer would hide the model's
mistakes.

Live rasterizer: our own GL 3.3 CPU-sorted instanced-quad pass inside
`gvd_viz` (Phase 0 Breaker verdict). Sort+render stay in that process and
must meet p95 < 33 ms on the i9-9900K (unmeasured until IQU45). Do not
ship nerfstudio or gsplat ≥1.0. gsplat 0.1.12 or INRIA
`diff-gaussian-rasterization` are **offline teacher fits only**, and only
after an IQU45 build prove (both unofficial on sm_61). INRIA is a
**research / non-commercial** license (see appendix); do not add it to
the runtime tree.

## Interface

**Inputs** (from the supervisor, copies only): camera frames for the live
rig (8-cam: `narrow`, `main`, `wide`, `pillarL`, `pillarR`, `repeatL`,
`repeatR`, `rear`; a 3-cam subset is the windshield cluster `main` /
`narrow` / `wide`). Long side ≤ 640. Intrinsics / extrinsics from
`config/cameras.yaml` (pinhole K from `fov_v` + resolution until a
calibrated K exists). Ego pose: the live default (VO); do not add wheel
odometry or sim pose as a live input. Predicted depth / splat parameters
**only when a live predicted-depth source exists on main** (not
`aux_depth`, not sim GT, not a draft branch). On `8f4a163` that input is
absent, so the live path does not start a renderer. Capture backends stay
`beamngpy` | `window` | `stub`.

**Outputs:** an RGB image for the left GVD VISION pane, **and a rendered
depth buffer** at the same pose and resolution as that RGB. The on-screen
live view may show RGB only (displaying depth is optional). The depth
buffer is **required** whenever the renderer runs, including step 8 and
graduation: AbsRel and δ < 1.25 are scored from this buffer against sim
GT (offline, recorded frames). Chase / drone camera is the product
viewpoint; a recording-only `chase_camera` already exists in
`cameras.yaml` for offline supervision, and is not a live sensor.

**Plug-in:** `viz.reality_view_enabled` / `--reality-view`, default **OFF**.
`viz.main_view` stays `reality` (placeholder) regardless. Live process
boundary: the infer process or the viz process unprojects and packs the
splat or billboard block into a double-buffered shared-memory slot. After
`actuator.apply`, the supervisor only memcpy / handoffs that slot
(non-blocking, drop on backpressure). A **separate** `gvd_viz` process
maps the newest complete slot and composites over the existing vector /
ego layers. No in-process live renderer, and no unproject or packing on
the supervisor.

## Feasibility plan

Each step has a go / no-go gate and a test to write first. Steps 1–3, 5,
6, and 8 run offline on the box CPU (stub or recorded frames). Steps 4
and 7 are IQU45 proves that GVD schedules; this swarm does not touch
Desktop-IQU45. Tests for 4 and 7 are written first and skip (or assert
"unmeasured") on the box, then run on IQU45.

0. **Live depth source gate.** A live predicted-depth tensor must exist
   on main: named, exported on the live net, not train-only `aux_depth`,
   not sim GT. **Test first:** `check_live_depth_source_on_main` — a
   **graduation gate**, not a default-suite check. It is expected to
   fail on `8f4a163` (`bev_lss.py` missing, `aux_depth` not a live
   export) and passes only when a documented live tensor is present.
   It is **not run in the default offline suite on main** (so that
   expected fail cannot show up as a suite red). It runs only in the
   Reality View graduation-gate job (step 0 / step 8 graduate). **Go:**
   the gate-job check passes. **No-go (current):** live RV stays the
   #242 placeholder; A and C stay stub / recording-only; no live
   graduate of A or C. If #241 never lands and no later depth-head spec
   is accepted, this no-go stands and the live A/C path is killed.
1. **Contract and flag pin.** Schema for inputs / outputs / SHM slot;
   flag default OFF; placeholder bit-identical to `8f4a163`; no
   safety-path imports; live path is a separate process only.
   **Test first:** `check_flag_off_is_placeholder`,
   `check_live_path_is_separate_process` (live flag starts or attaches
   only a child process; the supervisor module does not import a
   renderer that draws on its own thread). **Go:** those tests pass.
   **No-go:** flag-off pixels or tick path change, or an in-process live
   renderer.
2. **C prototype (CPU, offline).** Unproject depth to points / billboards
   from a chase pose in a **test process**; GT depth only in tests named
   GT. **Test first:** `check_deterministic_cpu`,
   `check_unknown_not_free`, `check_no_sim_gt_on_live_path`. **Go:** those
   pass, plus a deterministic PNG pin. **No-go:** GT used on a live path,
   a new dependency, or the prototype linked into the supervisor.
3. **Isolation harness.** A deliberately slow renderer **or generator**
   (unproject / splat packing) in a **child** infer or viz process cannot
   move a stub control-loop period outside the 5% gate; killing or
   crashing that child leaves heartbeat / AEB / engage and the
   supervisor itself running. **Test first:**
   `check_safety_loop_latency_unchanged` (slow child generator and slow
   child renderer),
   `check_renderer_crash_degrades` (SIGKILL / abort the viz or infer
   child; assert the supervisor process is still alive, heartbeat stamps
   continue, AEB / engage state is unchanged, and the main view degrades
   to the placeholder),
   `check_supervisor_after_apply_is_copy_only` (after `actuator.apply`,
   the supervisor path is memcpy / handoff only: unproject and splat
   packing are not called in the supervisor; the post-apply cost is the
   copy, not generation). **Go:** all three pass; the tests fail if any
   renderer or generator is on the supervisor thread. **No-go:** any
   stall, supervisor death, or packing on the supervisor.
4. **Rasterizer prove (IQU45, GVD-scheduled).** Build/smoke gsplat 0.1.12
   and INRIA with `TORCH_CUDA_ARCH_LIST=6.1` on Windows (offline only);
   record GL version via `gvd_viz --info`. **Test first:**
   `check_gl33_splat_50k_under_cap` — a 50k splat or billboard pass
   composites in `gvd_viz`; measured whole-process NVIDIA memory ≤ 300 MB
   and the formula sum stays under the `viz_cpp.md` rule above;
   `check_offline_cuda_rasterizer_smoke` records pass/fail of the two
   CUDA builds (failure is not a live blocker). **Go:** the GL 3.3 cap
   check passes (on IQU45). **No-go:** if even GL 3.3 cannot composite
   that 50k pass under the 300 MB / BeamNG-FPS gates, do not graduate
   live A; C stays offline-only until step 0, then is retried only if it
   meets the same caps.
5. **A generator (CPU first, offline).** SH0 Gaussians from a fixture
   predicted-depth tensor + RGB + K, cap N, no new heavy net. **Test
   first:** `check_n_cap`, `check_unknown_not_free` (reused). **Go:**
   offline render from a canned pose; honesty pin; N cap enforced.
   **No-go:** a new live net or a raised VRAM cap.
6. **Optional rolling merge (offline).** Voxel-hash EMA in a bounded
   region (e.g. ±64 m, N_max 300k as a working cap, not a measured
   overdraw). Evict by age / distance. Dynamic boxes never enter the
   static map. **Test first:** `check_no_ghost_rolling_map` (GT-named
   canned clips). **Proposed thresholds (unmeasured, fail the test if
   missed):** (a) static scene: duplicate-surface rate ≤ **5%** of
   occupied 0.25 m cells (two Gaussians in one cell whose centres differ
   by > 0.5 m along the same camera ray, both with opacity ≥ 0.1);
   (b) the same static scene with injected pose noise (σ = 0.15 m
   translation, 1° rotation) and **80 ms** inter-cam skew (covers the
   Tracer max 79.8 ms): median |z_render − z_gt| ≤ **0.50 m** on valid GT
   depth in 0.5–40 m; (c) a moving GT vehicle box must not leave a
   persistent static splat in its old cell after it has been gone ≥ 1.0 s
   (leftover opacity ≤ 0.1, or the cell evicted). **Go:** the test
   passes. **No-go:** shrink radius / age, or skip merge and keep
   per-frame A.
7. **IQU45 system prove (GVD-scheduled).** VRAM, RSS, sort+render p95,
   `infer_ms`, loop Hz, BeamNG FPS, RV on vs off, with BeamNG running.
   **Test first (written now, run on IQU45):** `check_iqu45_budgets`
   asserts every numeric row in the budget table, including the 4 GB /
   1 GB / 300 MB / RSS / 2 GB process RSS gates and their fallbacks;
   `check_iqu45_sort_render_p95` asserts p95 **sort+render** < 33 ms in
   the viz process on the i9-9900K (sort included; the 2.1 ms Xeon figure
   is not used); `check_iqu45_safety_and_beamng` asserts loop Hz and
   `infer_ms` p95 change < 5% and BeamNG FPS drop ≤ 5% (and the 30 fps
   floor when it applied). **Go:** all three pass. **No-go:** apply the
   row-by-row fallbacks in the budget table; a safety-loop miss kills
   live RV (placeholder).
8. **Kill or graduate (quality rubric).** Two offline runs on canned
   recorded / stub frames, chase or held-out camera, scored against sim
   GT (these tests are named GT). Both use the same thresholds below
   and require the renderer's depth buffer (see Outputs).
   (1) **Precondition / kill:** `check_useful_view_metrics` and
   `check_useful_view_checklist` on **C-from-GT** (sim GT depth as the
   unprojection source). If this fails, kill the photoreal claim and
   keep the placeholder + classic.
   (2) **Graduation gate (separate run):**
   `check_useful_view_metrics_live_depth` and
   `check_useful_view_checklist_live_depth` feed the **live
   predicted-depth source** (not sim GT, not `aux_depth`) to C and/or A
   and score the render against sim GT. Passing C-from-GT alone does
   **not** graduate a live path.
   **Go:** (1) and (2) both pass for the candidate path, step 0 has
   passed, and step 7 has passed → graduate that path (A if A-from-live-
   depth passed; else C if C-from-live-depth passed).
   **No-go:** (1) fails → kill the photoreal claim. (1) passes but (2)
   fails for A → do not graduate A; try C-from-live-depth at the same
   thresholds. (2) fails for both → stay on the placeholder even if
   C-from-GT passed. If step 0 is still a no-go, nothing graduates live.
   If isolation (step 3 or 7) failed, Reality View stays a placeholder.

## Step 8 quality rubric (proposed, unmeasured)

Numeric gates in `check_useful_view_metrics` (C-from-GT) and
`check_useful_view_metrics_live_depth` (graduation; live predicted
depth as the unprojection source). The same thresholds apply to both.
Computed only on valid GT pixels (finite sim-GT depth, 0.5–80 m unless
noted), canned clip, offline. AbsRel and δ < 1.25 compare the
**required rendered-depth buffer** (Outputs) to sim GT at the chase or
held-out pose. A run that does not emit that buffer fails these two
metrics (it does not skip them).

| Metric | Proposed pass | Notes |
| --- | --- | --- |
| AbsRel on the required rendered-depth buffer vs sim GT, 0.5–80 m | ≤ **0.20** | Standard relative depth error; **proposed**, not a measured baseline. Same gate for C-from-GT and the live-depth graduation run. |
| δ < 1.25 (fraction of those pixels with max(d/d\*, d\*/d) < 1.25) | ≥ **0.70** | **Proposed.** Same source: rendered-depth buffer vs sim GT. |
| Coverage: fraction of chase-view pixels with valid GT depth in 0.5–40 m that have a splat/billboard with opacity ≥ 0.1 | ≥ **50%** | Holes are allowed; a spray that misses half the near scene fails. **Proposed.** |
| Held-out view PSNR / SSIM vs that camera's GT RGB (chase or a held-out rig cam on the same tick) | PSNR ≥ **16 dB**, SSIM ≥ **0.45** | Modest on purpose for a first small-stem / billboard render; not photoreal. **Proposed.** |
| Rate / age (only required to graduate *live*, via step 7) | update ≥ 10 Hz or a documented fallback in the rate row; p95 sort+render < 33 ms; stale indicator if age > 200 ms | Offline CPU fps is not this gate. |

Written yes/no checklist in `check_useful_view_checklist` (C-from-GT)
and `check_useful_view_checklist_live_depth` (graduation). **All must be
yes** on the same canned clip for that run (human or scripted asserts
where noted):

- Road surface is recognizable as a surface in the chase view, not only
  an unstructured point spray.
- Ego pose lines up with the existing ego mesh / ribbon: no ≥ 5 m
  lateral jump (scripted: ego footprint overlap with the mesh projection).
- A known vehicle in the clip is near the right place (scripted: centre
  error ≤ **3 m** in the ego frame, **proposed**) and is not drawn twice.
- Unobserved regions (no camera coverage) are hatched, fog, or empty,
  not a fake wall (`check_unknown_not_free` must still pass).
- No sim GT is consumed on a non-GT-named path
  (`check_no_sim_gt_on_live_path`).
- The stale indicator is present when the harness sets age > 200 ms.

**Step 8(1) scoring rules — spec proposal (Redesigner, not Michael's).**
These pins say how the published table and checklist above are applied.
They do not change any number already in that table (AbsRel ≤ 0.20,
δ < 1.25 fraction ≥ 0.70, coverage ≥ 50% with opacity ≥ 0.1, PSNR ≥ 16 dB,
SSIM ≥ 0.45, vehicle centre error ≤ 3 m, stale age 200 ms, GT ranges
0.5–80 m and 0.5–40 m).

| Topic | Proposed rule |
| --- | --- |
| AbsRel / δ pixel set | Scored only on **covered** pixels: opacity ≥ 0.1 AND valid rendered depth (finite, > 0) AND finite GT depth in 0.5–80 m. Holes are left to the coverage gate. (`ABS_REL_MAX`, `DELTA_125_MIN`, `COVERAGE_OPACITY_MIN`, `GT_DEPTH_RANGE_M`) |
| Vacuous scores | Zero scored pixels **fails** AbsRel and δ (never a pass). A missing rendered-depth buffer also fails those two metrics and does not skip them (already in the table notes). |
| Range edges | Inclusive: 0.5 m and 80 m count for AbsRel / δ (`GT_DEPTH_RANGE_M`); 0.5 m and 40 m count for coverage (`GT_COVERAGE_RANGE_M`). |
| δ < 1.25 | Strict `< 1.25`. A ratio of exactly 1.25 is not a hit. The 0.70 fraction gate itself is `≥` (`DELTA_125_MIN`). |
| Other inclusive gates | AbsRel ≤ 0.20, coverage ≥ 50%, opacity ≥ 0.1, PSNR ≥ 16 dB, SSIM ≥ 0.45, vehicle centre error ≤ 3 m, as written (`ABS_REL_MAX`, `COVERAGE_MIN`, `COVERAGE_OPACITY_MIN`, `PSNR_MIN_DB`, `SSIM_MIN`, `VEHICLE_CENTER_ERR_MAX_M`). |
| Stale indicator | Required when photon age **> 200 ms**. Age of exactly 200 ms is not stale. Matches `rv_contract.pack_output_slot` (`age_ms > RV_STALE_AGE_MS`). (`STALE_AGE_MS`) |
| Road surface (checklist) | Scripted: fraction of GT-road pixels (valid GT depth in 0.5–40 m) that are covered ≥ 50%, **and** AbsRel on those covered road pixels ≤ 0.20. A dense but wrong-depth road fails. Same numbers as the table, applied to the road mask (`ROAD_MIN_COVERAGE`, `ROAD_ABS_REL_MAX`). |
| Ego pose (checklist) | The scripted test is ego-footprint overlap with the mesh / ribbon projection. A tick with no overlap fails. A 4.9 m lateral offset of a 1.9 m-wide footprint has no overlap and fails; overlapping footprints of width *w* cannot be 5 m apart laterally, so no separate 5 m numeric gate is applied. |
| Vehicle centre | Median of the covered vehicle-mask pixels, ego-frame xy. Error ≤ 3 m inclusive (`VEHICLE_CENTER_ERR_MAX_M`). |
| Drawn twice | Fails if an elevated off-surface blob exists: height ≥ 0.5 m above the GT road plane (`GHOST_HEIGHT_MIN_M`), \|d − d\*\| > max(1 m, 20% of d\*) (`GHOST_DEPTH_TOL_M`), after 2 px dilation of the vehicle mask (`GHOST_EDGE_DILATE_PX`) and a 1 px morphological opening (`GHOST_OPEN_PX`), whose pixel count is ≥ 25% of the vehicle's pixels (`GHOST_MIN_FRAC_OF_VEHICLE`). Thin silhouette bleed is ignored. |
| Fake wall (unobserved) | Fails if more than 1% of the **2 px-eroded** unobserved mask is opaque (opacity ≥ 0.1 and valid depth) (`UNOBSERVED_OPAQUE_MAX_FRAC`, `UNOBSERVED_EDGE_ERODE_PX`). Hatch / fog / empty (RGB paint with opacity 0) is allowed. The 2 px erosion ignores billboard bleed at the region border. |
| Missing opacity | `RvOutputSpec` has no opacity channel. `opacity=None` means every valid-depth pixel is treated as opaque. |
| Camera / ego conventions | OpenCV camera frame (x right, y down, z forward). Depth is z-depth in metres. `t_ego_cam` is camera → ego. Ego is x forward, y left, z up. A rendered-depth hole is non-finite or ≤ 0. |
| SSIM definition | Wang *et al.* mean SSIM, uniform **box 7×7** window (`SSIM_WINDOW`; valid region, no padding, **not a Gaussian** — this window choice is a spec proposal, Redesigner, not Michael's), per-channel then averaged, K1 = 0.01, K2 = 0.03, L = 255, population moments (E[x²] − μ²). No new dependency. A library cross-check would call `skimage.metrics.structural_similarity(..., gaussian_weights=False, win_size=7, use_sample_covariance=True)`; scikit-image was not installed on the box that implemented this, so there is no library number in the tests. Note: that skimage flag uses sample covariance (/N−1); this scorer uses population moments, which agree on constant images (zero variance) and differ on textured ones. |

## Acceptance criteria (eventual implementation)

Fail-first offline tests (box CPU) before any IQU45 run:

- `check_live_depth_source_on_main`: graduation gate (step 0), **not**
  part of the default offline suite on main. It runs only in the Reality
  View graduation-gate job. On `8f4a163` that job is expected to fail
  (no live predicted depth); that expected no-go is not a default-suite
  red and does not block main CI. Live graduate stays blocked until the
  gate job passes.
- `check_flag_off_is_placeholder`: flag off matches `8f4a163` placeholder
  pixels on a pinned stub state.
- `check_live_path_is_separate_process`: the live flag path only starts
  or attaches a child process; no in-process live renderer in the
  supervisor.
- `check_no_sim_gt_on_live_path`: the live renderer API rejects or ignores
  sim-depth keys; a GT fixture is only consumed by a test named GT.
- `check_unknown_not_free`: cells marked unknown / unobserved are not
  opaque surface.
- `check_deterministic_cpu`: two runs on the same stub frames are equal.
- `check_n_cap`: a generator or map cannot exceed N_max.
- `check_safety_loop_latency_unchanged`: stub loop Hz and a stand-in
  `infer_ms` p95 move < 5% with a slow or 300k-load renderer **or
  generator** (unproject / packing) in a child infer or viz process; a
  blocked child is dropped, not waited on.
- `check_renderer_crash_degrades`: SIGKILL or abort of the viz or infer
  **child** leaves the supervisor process alive, heartbeat stamps
  continuing, AEB / engage unchanged, and the main view on the
  placeholder.
- `check_supervisor_after_apply_is_copy_only`: after `actuator.apply`,
  the supervisor only memcpy / handoffs; unproject and splat packing are
  not on the supervisor path.
- `check_no_ghost_rolling_map`: the step 6 thresholds (duplicate-surface
  ≤ 5%, median depth residual ≤ 0.50 m under pose noise + 80 ms skew,
  no leftover dynamic splat after 1.0 s).
- `check_useful_view_metrics` / `check_useful_view_checklist`: step 8
  precondition / kill, C-from-GT, same numeric thresholds and checklist.
- `check_useful_view_metrics_live_depth` /
  `check_useful_view_checklist_live_depth`: step 8 graduation gate; the
  same thresholds, but the unprojection source is the live predicted-
  depth tensor, scored against sim GT on recorded frames. Required to
  graduate live C or A; C-from-GT alone is not enough.
- `check_vram_rss_hooks`: the same counters as `scripts/test_vram_cap.py`
  and `rss_mb` still apply; flag-on does not raise the 4 GB cap. Numeric
  VRAM / RSS / BeamNG / sort+render gates are asserted by
  `check_iqu45_budgets`, `check_iqu45_sort_render_p95`, and
  `check_iqu45_safety_and_beamng` on the IQU45 prove (step 7), not faked
  on the box.
- `check_gl33_splat_50k_under_cap` / `check_offline_cuda_rasterizer_smoke`:
  written first for step 4; numeric memory asserts run on IQU45.

## Risks, open questions, kill criteria

Top risks: (1) WDDM time-slicing on Pascal steals SMs from BeamNG or
vision (memory is not the likely failure; this is unmeasured). Gated by
the BeamNG FPS ≤ 5% drop (and the 30 fps floor when it applies) plus the
safety-loop 5% gate. (2) There is no live predicted depth on main; if
#241 never lands and no later depth head is accepted, the live A/C path
stays killed and a small stem may still fail the step 8 rubric offline.
(3) Ego-pose drift and inter-cam skew (Tracer Phase 0: median 34.4 ms,
max 79.8 ms) ghost a rolling map (gated by `check_no_ghost_rolling_map`).

Open questions (from Phase 0, still open): Q3 sm_61 CUDA rasterizer
build on Windows; Q5 contention vs BeamNG; Q6 whether WDDM / priority
can deprioritize `gvd_viz`; live VO drift vs a ±64 m map.

**Kill** the live A/B/C graduate path if: step 0 never passes; a new dep
needs CC ≥7.0; (vision + RV) exceeds 4 GB after the generation fallback;
the safety-loop or renderer-crash test fails; INRIA / a non-commercial
rasterizer is required at runtime; or a heavy feed-forward net is the
only way to pass step 8. Missing the soft 1 GB generation slice alone is
**not** a kill if the 4 GB hard cap still holds. **Kill** the photoreal
claim (keep placeholder + classic) if step 8's C-from-GT rubric fails.
Failing the live-depth graduation run does not by itself kill the
photoreal claim, but it blocks live graduate of that path.

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
sort in a worker; ~4 fps sort vs 60 fps render; ~1M splats ~150 ms sort;
not a 9900K budget).

GVD grounding (read-only, not in this repo tree): `/workspace/gvd2/spec.md`,
`/workspace/gvd2/audit/redesigner.md` §13, `/workspace/gvd2/audit/breaker.md`
§14–15, 19–20, `/workspace/gvd2/audit/tracer.md` (sync / depth / pose).
In-repo: PR #242 (`python/viz/view_select.py`, README UI guide),
`docs/specs/viz_cpp.md`, `config/cameras.yaml`, `config/hardware.yaml`,
`docs/research/gpu_tiers.md`, `docs/ROADMAP.md` (`aux_depth` train-only).
`gh issue list --search "Reality View"` was empty; the only merged
Reality View change found is PR #242. PR #241 (BEV LSS) is still a draft
and is not on `8f4a163`.

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
