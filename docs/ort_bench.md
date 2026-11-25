# SceneNet: PyTorch vs ONNX Runtime FP32 (export, parity, EP report, bench)

Phase 0 decision D4: PyTorch trains, **ORT CUDA EP FP32** deploys. This harness
measures that path for the **current** SceneNet (small preset, 8 cams,
2,203,694 params). Offline only: nothing live imports it, it feeds nothing
live (no odometry, no controls), and it never touches heartbeat / AEB / engage.

| piece | file |
| --- | --- |
| library | `python/train/ort_bench.py` |
| CLI (JSON out) | `scripts/ort_bench.py` |
| offline test (CPU EP) | `scripts/test_ort_bench.py` |
| scripted mutants (dev tool) | `scripts/ort_bench_mutants.py` |
| optional pins | `requirements-ort.txt` |

## What it does

1. **Export**: reuses `scene_net.export_scene_onnx` (no second exporter). Opset
   **17** (fixed in `_trace_scene_onnx`, asserted). Inputs are the live ones
   (`scene_lane_runtime.observe`): `strip` 1x3x192x2272 float in [0,1] (8
   sectors, NCHW), `dt` 1, `h` 1x1x512, `tokens` 1x90x512, `mask` 1x90. The
   tracer renames state inputs (`h.1`, ...); names are matched with the
   runtime's `_base_name`.
2. **Parity**: seeded strips, 6-tick recurrent rollout (dt 0, 0.1, 0.1, 0.05,
   0.2, 0.1). Each side feeds its own `h`/`tokens`/`mask` outputs back, as the
   live loop does. Every output (12 heads + 3 state tensors) on every tick is
   compared.
   **Tolerance (stated 2026-11-23 before measuring, never widened):**
   `|ort - torch| <= 1e-4 + 1e-3 * |torch|` per element (`PARITY_ATOL`,
   `PARITY_RTOL`). Shape mismatch, a missing output, or a NaN/Inf mismatch
   fails. CPU and CUDA EP use the same numbers. Parity failure is exit code 1.
3. **Op/EP report**: ORT profiling gives each executed node's provider;
   `fallback_ops` = nodes not on the requested EP. The partitioned, optimized
   graph (`optimized_model_filepath`) is scanned for `MemcpyToHost` /
   `MemcpyFromHost` nodes the partitioner inserted. Per-tick host<->device bytes
   for numpy feeds are reported too (`h2d_bytes_per_tick` 5,421,420,
   `d2h_bytes_per_tick` 189,216).
4. **Bench**: warmup + N timed live ticks (state fed back), p50/p90/p99/max,
   PyTorch eager CPU on the same tick for reference, peak RSS (ru_maxrss on
   POSIX, psutil peak working set on Windows if installed). On CUDA, VRAM via
   an `nvidia-smi` sampler: device baseline/peak/delta and this pid's
   `used_memory` (WDDM often reports `[N/A]` per process; then use the delta
   with BeamNG closed). CUDA EP is created with `gpu_mem_limit` 4 GiB
   (the live YOLO cap) and `arena_extend_strategy=kSameAsRequested`.
   `--ort-only --onnx X` blocks the torch import so RSS/VRAM are the deploy
   path's own.

## Box results (CPU EP, no GPU; NOT IQU45 numbers)

See the branch report. The box is a shared, loaded 8-vCPU VM; latency there is
indicative only.

## Pins (requirements-ort.txt), all UNVERIFIED until Q1-Q3

| item | pin | why / source |
| --- | --- | --- |
| onnxruntime-gpu | 1.26.0 | Last PyPI onnxruntime-gpu built with CUDA 12.8; from 1.27 PyPI wheels are CUDA 13.0 ([ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)). CUDA 13 removed Pascal ([CUDA 13.0 release notes, quoted](https://btw.media/en/nvidia-cuda-support-date-hangs-on-driver-branch-not-card); a GTX 1080 user hit "architectural feature absent" on ORT 1.27/CUDA 13 and fixed it with 1.26.0, [rembg#849](https://github.com/danielgatis/rembg/issues/849)). ORT's packaging matrix lists Windows x64 CUDA 12.8 archs `61;75;86;89;120a` ([onnxruntime#31992](https://github.com/microsoft/onnxruntime/pull/31992), merged 2026-08-11, i.e. after 1.26.0: whether the 1.26.0 Windows wheel already had sm_61 SASS is **UNVERIFIED**; Q1 settles it). The Linux CUDA 12.8 build was measured with sm_60 SASS (runs on sm_61) ([ollaya ADR 0004](https://git.xszz.org/ollaya-dev/ollaya/blob/main/docs/decisions/0004-cuda-onnxruntime-builds.md)). |
| CUDA runtime libs | 12.8.x wheels | ORT 1.21-1.26 "built with CUDA 12.8, requires CUDA 12.8 or newer" (ORT docs). Driver R570+ advised. |
| cuDNN | nvidia-cudnn-cu12==9.10.2.21 | Pre-Turing support ends after cuDNN 9.10.x (9.10.1 matrix lists CC 6.1, [cuDNN 9.10.1 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.1/reference/support-matrix.html); latest 9.28.1 starts at 7.5, [latest matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html); [ultralytics#22002](https://github.com/ultralytics/ultralytics/issues/22002): 1080 Ti breaks above 9.10.2). onnxruntime-gpu's `[cudnn]` extra only asks `~=9.0`, so it must be pinned. Same version torch 2.14.1+cu126 requires (wheel metadata `nvidia-cudnn-cu12==9.10.2.21`). |
| torch (training) | 2.14.1 cu126 (unchanged) | cu126 is the only torch CUDA build with Pascal SASS (sm_50/60/70..90; Windows adds 6.1); cu128+ start at sm_70/75 ([PyTorch RELEASE.md matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [dev-discuss 3128](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)). 2.14 is the last release with cu126 wheels ([dev-discuss 3432](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). Do not upgrade past 2.14 on IQU45. |

onnxruntime and onnxruntime-gpu both provide the `onnxruntime` module: use a
separate venv for the GPU bench (Q1) so the live retail env is not changed.

## IQU45 commands (for GVD to run later; ask first, one GPU job at a time, after the recorder smoke run)

Nothing here has been run on Desktop-IQU45. BeamNG closed for Q2/Q3 so the VRAM
delta is clean.

- **Q1 install + EP load (no model run):**
  ```bat
  cd <repo> && git fetch && git checkout gvd/swarm2-ort-bench
  py -3.12 -m venv %USERPROFILE%\venv-ortgpu
  %USERPROFILE%\venv-ortgpu\Scripts\python -m pip install -r requirements-ort.txt numpy PyYAML opencv-python
  %USERPROFILE%\venv-ortgpu\Scripts\python -c "import onnxruntime as o; o.preload_dlls(); print(o.__version__, o.get_device(), o.get_available_providers())"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.used,memory.total --format=csv
  ```
  Pass: `CUDAExecutionProvider` listed, compute_cap 6.1, driver >= 570.
- **Q2 ORT-only CUDA bench (no torch in the process):** export once on the CPU
  with the normal env, then bench in the GPU venv:
  ```bat
  py -3 scripts\ort_bench.py --ep cpu --workdir %TEMP%\gvd_ort --out %TEMP%\gvd_ort\bench_cpu.json
  %USERPROFILE%\venv-ortgpu\Scripts\python scripts\ort_bench.py --ep cuda --ort-only --onnx %TEMP%\gvd_ort\scene_net_bench.onnx --warmup 20 --iters 300 --workdir %TEMP%\gvd_ort --out %TEMP%\gvd_ort\ort_only_cuda.json
  ```
  Read: `ops.fallback_ops` (nodes on CPU), `ops.memcpy_total`, `bench.ort.p50_ms/p99_ms`,
  `bench.vram.device_delta_mb` / `pid_peak_mb` (must stay < 4096), `bench.peak_rss_mb_after_ort_runs`.
  If session creation raises "requested CUDAExecutionProvider, session has [...]", the EP did not load: capture the console log.
- **Q3 CUDA parity (needs torch CPU + onnxruntime-gpu in one venv):**
  ```bat
  %USERPROFILE%\venv-ortgpu\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  %USERPROFILE%\venv-ortgpu\Scripts\python scripts\ort_bench.py --ep cuda --warmup 20 --iters 300 --out %TEMP%\gvd_ort\full_cuda.json
  ```
  Exit 1 = parity failed at the stated tolerance: report it, do not widen it.
  (The CPU torch wheel keeps the venv's cuDNN 9.10.2 the only cuDNN loaded.)
- **Q4 with BeamNG running (driving priority check):** repeat Q2 with BeamNG at the
  usual settings; compare p99 and device delta. Not before GVD schedules a GPU slot.

## Findings worth knowing

- The export is a trace. `scene_net.py:2222` (`if float(strip.max()) > 1.5: strip/255`)
  is frozen at trace time with the zero example, so the graph **never** rescales:
  feeding 0..255 floats would silently be wrong. The live runtime already scales in
  `_strip_nchw`; any new caller must feed [0,1].
- The tracer also freezes batch=1 and the canvas size (fixed dims), which the live
  runtime already enforces (`SceneInputShapeError`).
