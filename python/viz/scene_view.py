"""View-only scene-net lane viewer: a view-only same-map experiment.

Shows what a trained scene net (``lane_gt[14]``) outputs on the live cameras, drawn on the stage
in amber. It is not a lane source and it never drives: nothing here imports control, engage or
the actuator, and run_vision only stores ``SceneView.to_state()`` in ``st["scene_view"]``.

Main thread (``submit``): compose the strip with the training code (``compose_strip`` ->
``prepare_model_strip``), copy it, hand it to one worker thread. The worker owns the
``SceneLaneRuntime`` (ORT CPU, 2 intra-op threads) and its GRU state, and publishes an immutable
``SceneView``. The main thread never waits on the worker: a busy worker drops the step.

Steps: a new main frame, >= 90 ms since the last step, worker idle. GRU reset every 16 steps
(``CHUNK_MAX``, the training window), on model switch, a respawn edge, a gap > ``MEMORY_S``.
Budget (P = 1/hz): skip when the previous heartbeat > 0.75 P or the loop gap > 1.5 P; main
slice EMA > 0.10 P pauses 2 s ("throttled"); forward EMA > 0.75 P pauses 2 s ("slow"); three
consecutive errors latch the viewer off until the next switch.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from python.data.strip_writer import compose_strip
from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M, confidence_from_vector
from python.sensors.live_hold import RespawnWatch, bundle_ages, scrub_after_respawn
from python.train.cam_strip import prepare_model_strip

VIEW_AMBER_BGR = (32, 176, 255)  # #FFB020
VIEW_BG_BGR = (16, 13, 12)
VIEW_ERROR_BGR = (60, 60, 230)
VIEW_INTRA_OP_THREADS = 2
STEP_MIN_S = 0.090
# continual.CHUNK_MAX and scene_net.MEMORY_S (asserted equal in scripts/test_scene_view.py).
# Not imported: scene_net imports torch at load and run_vision imports this module.
CHUNK = 16
GAP_RESET_S = 15.0
PAUSE_S = 2.0
ERR_LATCH = 3
EMA_ALPHA = 0.2
MIN_CONF = 0.40  # scene_lane_runtime.SCENE_LANE_MIN_CONF
LANE71_BADGE = "lane71 needs 8 cams (BeamNG.tech)"
FRAME = "body_centreline"


def _default_runtime(path: Path, **kw: Any) -> Any:
    from python.train.scene_lane_runtime import SceneLaneRuntime

    return SceneLaneRuntime(path, **kw)


# Test hook: scripts/gvd_slow_view_child.py swaps in a slow fake runtime.
DEFAULT_RUNTIME_FACTORY: Callable[..., Any] = _default_runtime


@dataclass(frozen=True)
class SceneView:
    """One published viewer result. Display only: no control fields."""

    stem: str = "off"
    status: str = "off"  # off | waiting | ok | slow | throttled | error | refused
    badge: str = ""
    lane_gt: tuple[float, ...] = ()
    conf: float = 0.0
    lane_x: float | None = None
    dt_ms: float = 0.0
    k: int = 0
    steps: int = 0
    dropped: int = 0
    skipped: int = 0
    forward_ms: float = 0.0
    frame: str = FRAME

    def to_state(self) -> dict[str, Any]:
        out = asdict(self)
        out["lane_gt"] = [float(v) if math.isfinite(float(v)) else None for v in self.lane_gt]
        return out


def decode_lane_gt(vec: Any) -> dict[str, Any] | None:
    """``lane_gt[14]`` -> centre/left/right ``(x lateral m, y ahead m)`` points, heading, conf."""
    try:
        arr = np.asarray([np.nan if v is None else v for v in list(vec)], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < LANE_GT_DIM:
        return None
    lines: dict[str, list[tuple[float, float]]] = {"centre": [], "left": [], "right": []}
    for k, y in enumerate(LOOKAHEADS_M):
        for j, name in enumerate(("centre", "left", "right")):
            x = float(arr[3 * k + j])
            if math.isfinite(x):
                lines[name].append((x, float(y)))
    heading = float(arr[3 * len(LOOKAHEADS_M)])
    return {**lines, "heading_rad": heading if math.isfinite(heading) else None,
            "conf": confidence_from_vector(arr[:LANE_GT_DIM])}


def resolve_view_path(raw: str) -> Path:
    """A relative ``--scene-view`` path is relative to the data root."""
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    try:
        from python.data.data_root import data_root

        return data_root() / path
    except Exception:
        return path


def parse_scene_view_arg(text: str | None) -> dict[str, Path]:
    """``PATH[,PATH]`` -> {stem: path}, in order."""
    out: dict[str, Path] = {}
    for part in str(text or "").split(","):
        if part.strip():
            path = resolve_view_path(part)
            out[path.stem] = path
    return out


def view_strip(frames: Mapping[str, Any], cams: tuple[str, ...], canvas_w: int) -> np.ndarray | None:
    """The model's canvas built by the training code; an owned copy."""
    strip = compose_strip({cam: frames[cam] for cam in cams if cam in frames})
    out = prepare_model_strip(strip, tuple(cams), int(canvas_w))
    return None if out is None else np.array(out, dtype=np.uint8, copy=True)


class SceneNetViewer:
    """One worker thread + one scene-net runtime; display only."""

    def __init__(
        self,
        models: Mapping[str, Path],
        *,
        hz: float,
        can_feed_cam_age: bool,
        runtime_factory: Callable[..., Any] | None = None,
        threaded: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.models = {str(k): Path(v) for k, v in dict(models).items()}
        self.period_s = 1.0 / max(1e-3, float(hz))
        self.can_feed_cam_age = bool(can_feed_cam_age)
        self.factory = runtime_factory
        self.threaded = bool(threaded)
        self.clock = clock
        self.stem = "off"
        self.dropped = 0
        self.skipped = 0
        self.steps = 0
        self._gen = 0
        self._view = SceneView()
        self._info: dict[str, Any] | None = None
        self._rt: Any = None
        self._busy = False
        self._last_key: Any = None
        self._last_step_t: float | None = None
        self._reset_pending = True
        self._slice_ema: float | None = None
        self._fwd_ema: float | None = None
        self._pause_until = 0.0
        self._pause_status = ""
        self._hold_on = False
        self._respawn = RespawnWatch()
        self._errors = 0
        self._latched = False
        self._k = 0
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        if self.threaded:
            self._thread = threading.Thread(target=self._run, name="gvd-scene-view", daemon=True)
            self._thread.start()

    # ---- main thread ------------------------------------------------------------------

    @property
    def runtime(self) -> Any:
        return self._rt

    def select(self, stem: str | None) -> None:
        stem = str(stem or "off")
        self._gen += 1
        self.stem = stem
        self._info = None
        self._last_key = None
        self._last_step_t = None
        self._reset_pending = True
        self._slice_ema = None
        self._fwd_ema = None
        self._pause_until = 0.0
        self._errors = 0
        self._latched = False
        self._view = SceneView(stem=stem, status="off" if stem == "off" else "waiting")
        if stem != "off":
            # Import torch / ORT here, on the caller's thread (startup, before the loop), not in the
            # worker: a first import in a background thread holds the GIL for ~1 s and starves the tick.
            import python.train.scene_lane_runtime  # noqa: F401
        self._post(("load", self._gen, stem))

    def observe_respawn(self, vdata: Any, tick_dt: float | None, bundle: Any, backend: Any) -> None:
        if self._respawn.observe(vdata, tick_dt, mono=time.monotonic()) is None:
            return
        if self._hold_on and backend is not None:
            backend.reset_live_hold()
            scrub_after_respawn(bundle)
        self._reset_pending = True

    def submit(self, bundle: Any, *, backend: Any, prev_heartbeat_ms: float, loop_gap_s: float) -> float:
        """Queue one step if the gates allow; returns this call's main-thread ms. Never raises."""
        t0 = self.clock()
        try:
            self._submit(bundle, backend, float(prev_heartbeat_ms or 0.0), float(loop_gap_s or 0.0), t0)
        except Exception as exc:
            self._error(exc)
        ms = max(0.0, (self.clock() - t0) * 1000.0)
        if self._info is not None:
            self._slice_ema = ms if self._slice_ema is None else (1 - EMA_ALPHA) * self._slice_ema + EMA_ALPHA * ms
            if self._slice_ema > 0.10 * self.period_s * 1000.0:
                self._pause(t0, "throttled")
                self._slice_ema = None
        return ms

    def _pause(self, now: float, status: str) -> None:
        self._pause_until = now + PAUSE_S
        self._pause_status = status

    def _submit(self, bundle: Any, backend: Any, prev_hb_ms: float, gap_s: float, now: float) -> None:
        info = self._info
        if self.stem == "off" or info is None or self._latched or bundle is None:
            return
        if now < self._pause_until:
            return
        fwd = self._fwd_ema
        if fwd is not None and fwd > 0.75 * self.period_s * 1000.0:
            self._fwd_ema = None
            self._pause(now, "slow")
            return
        if prev_hb_ms > 0.75 * self.period_s * 1000.0 or gap_s > 1.5 * self.period_s:
            self.skipped += 1
            return
        frames = getattr(bundle, "frames", None) or {}
        stamps = getattr(bundle, "timestamps", None) or {}
        key = stamps.get("main")
        if key is None:
            key = id(frames.get("main"))
        if key == self._last_key:
            return
        if self._last_step_t is not None and now - self._last_step_t < STEP_MIN_S:
            return
        if self._busy:
            self.dropped += 1
            return
        ages = None
        if info["cam_age"]:
            if not self._hold_on and backend is not None and hasattr(backend, "enable_live_hold"):
                backend.enable_live_hold()
                self._hold_on = True
            frames, ages = bundle_ages(bundle)
        strip = view_strip(frames, info["cams"], info["canvas_w"])
        if strip is None:
            raise ValueError("no strip at the model canvas")
        dt = None if self._last_step_t is None else now - self._last_step_t
        reset = self._reset_pending or (dt is not None and dt > GAP_RESET_S)
        self._reset_pending = False
        self._last_key = key
        self._last_step_t = now
        self._busy = True
        self.steps += 1
        self._post(("step", self._gen, strip, ages, None if reset else dt, reset))

    def _error(self, exc: BaseException) -> None:
        self._errors += 1
        badge = f"error: {type(exc).__name__}"
        if self._errors >= ERR_LATCH:
            self._latched = True
            badge += " (latched off)"
        self._view = SceneView(stem=self.stem, status="error", badge=badge)

    def latest(self) -> SceneView:
        """The last result with this side's counters. Never raises."""
        try:
            view = self._view
            status, badge = view.status, view.badge
            if status in ("ok", "waiting") and self.clock() < self._pause_until:
                status = badge = self._pause_status
            return SceneView(**{**asdict(view), "lane_gt": view.lane_gt, "status": status, "badge": badge,
                                "steps": self.steps, "dropped": self.dropped, "skipped": self.skipped})
        except Exception as exc:
            return SceneView(stem=str(getattr(self, "stem", "off")), status="error", badge=f"error: {type(exc).__name__}")

    def close(self) -> None:
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _post(self, job: Any) -> None:
        if self.threaded:
            self._q.put(job)
        else:
            self._do(job)

    # ---- worker -------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            self._do(job)

    def _do(self, job: tuple) -> None:
        kind, gen = job[0], job[1]
        try:
            if kind == "load":
                self._load(gen, job[2])
            elif gen == self._gen:
                self._step(gen, *job[2:])
        except Exception as exc:
            if gen == self._gen:
                self._error(exc)
        finally:
            if kind == "step":
                self._busy = False

    def _load(self, gen: int, stem: str) -> None:
        self._rt = None
        if stem == "off" or gen != self._gen:
            return
        from python.train.scene_lane_runtime import SceneCamAgeError, SceneInputShapeError

        path = self.models.get(stem)
        if path is None:
            self._view = SceneView(stem=stem, status="refused", badge=f"unknown model {stem}")
            return
        factory = self.factory or DEFAULT_RUNTIME_FACTORY
        try:
            rt = factory(path, feeds_cam_age=self.can_feed_cam_age, intra_op_threads=VIEW_INTRA_OP_THREADS)
        except SceneCamAgeError as exc:
            badge = LANE71_BADGE if "8 per-camera" in str(exc) else "refused: cam_age contract"
            self._view = SceneView(stem=stem, status="refused", badge=badge)
            return
        except SceneInputShapeError:
            self._view = SceneView(stem=stem, status="refused", badge="refused: wrong strip width")
            return
        if not getattr(rt, "ready", False):
            self._view = SceneView(stem=stem, status="refused", badge=str(getattr(rt, "reason", "not ready")))
            return
        if gen != self._gen:
            return
        self._rt = rt
        self._k = 0
        self._info = {"cams": tuple(rt.cams), "canvas_w": int(rt.canvas_w), "cam_age": bool(rt.cam_age)}

    def _step(self, gen: int, strip: np.ndarray, ages: Any, dt_s: float | None, reset: bool) -> None:
        rt = self._rt
        if rt is None:
            return
        if reset or self._k >= CHUNK:
            rt.reset_state()
            self._k = 0
            dt_s = None
        before = rt.last_timers
        t0 = time.perf_counter()
        lane_x, conf = rt.observe(strip, dt_s, ages)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        if rt.last_timers is before:
            raise RuntimeError("scene forward failed")
        vec = np.asarray(rt.output("lane_gt"), dtype=np.float64).reshape(-1)[:LANE_GT_DIM]
        if vec.size < LANE_GT_DIM:
            raise ValueError(f"lane_gt has {vec.size} values")
        self._k += 1
        if self._k >= CHUNK:
            rt.reset_state()  # the window ends here; the state is fresh again
        self._fwd_ema = fwd_ms if self._fwd_ema is None else (1 - EMA_ALPHA) * self._fwd_ema + EMA_ALPHA * fwd_ms
        self._errors = 0
        if gen != self._gen:
            return
        self._view = SceneView(
            stem=self.stem, status="ok", badge="low conf" if float(conf) < MIN_CONF else "",
            lane_gt=tuple(float(v) for v in vec), conf=float(conf),
            lane_x=None if lane_x is None else float(lane_x),
            dt_ms=0.0 if dt_s is None else float(dt_s) * 1000.0, k=self._k, forward_ms=fwd_ms,
        )


# ---- run_vision helpers: every call guarded, nothing returned but display data ----------


def view_tick(viewer: Any, bundle: Any, *, backend: Any, prev_heartbeat_ms: float, loop_gap_s: float) -> float:
    if viewer is None:
        return 0.0
    try:
        return float(viewer.submit(bundle, backend=backend, prev_heartbeat_ms=prev_heartbeat_ms, loop_gap_s=loop_gap_s))
    except Exception:
        return 0.0


def view_respawn(viewer: Any, vdata: Any, tick_dt: float | None, bundle: Any, backend: Any) -> None:
    if viewer is None:
        return
    try:
        viewer.observe_respawn(vdata, tick_dt, bundle, backend)
    except Exception:
        pass


def view_state(viewer: Any) -> dict[str, Any] | None:
    if viewer is None:
        return None
    try:
        return viewer.latest().to_state()
    except Exception as exc:
        return {"stem": "?", "status": "error", "badge": f"error: {type(exc).__name__}"}


# ---- stage overlay ----------------------------------------------------------------------


def _mix(conf: float) -> tuple[int, int, int]:
    a = max(0.15, min(1.0, float(conf)))
    return tuple(int(round(VIEW_BG_BGR[i] + (VIEW_AMBER_BGR[i] - VIEW_BG_BGR[i]) * a)) for i in range(3))


def _dashed(img: np.ndarray, pts: list[tuple[int, int]], colour: tuple[int, int, int], dash: int = 10) -> None:
    import cv2

    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) // dash))
        for i in range(0, n, 2):
            a, b = i / n, min(1.0, (i + 1) / n)
            cv2.line(img, (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a)),
                     (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b)), colour, 2, cv2.LINE_AA)


def draw_scene_view(img: np.ndarray, cam: Any, view: Mapping[str, Any] | None, *, origin: tuple[int, int] = (16, 120)) -> None:
    """Amber lane_gt lines in the stage ego frame + the view-only label. Display only."""
    import cv2

    if not isinstance(view, Mapping) or view.get("status") in (None, "off"):
        return
    status = str(view.get("status"))
    stem = str(view.get("stem", "?"))
    conf = float(view.get("conf") or 0.0)
    label = (f"SCENE VIEW (view-only) {stem} centreline conf {conf:.2f} "
             f"dt {float(view.get('dt_ms') or 0.0):.0f} ms {int(view.get('k') or 0)}/{CHUNK}")
    colour = _mix(conf)
    lines = decode_lane_gt(view.get("lane_gt") or []) if status in ("ok", "slow", "throttled") else None
    if lines is not None:
        for name in ("left", "right", "centre"):
            pts = [cam.project(x, y, 0.0) for x, y in lines[name]]
            if len(pts) < 2:
                continue
            if name == "centre":
                _dashed(img, pts, colour)
            else:
                cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, colour, 2, cv2.LINE_AA)
    x, y = origin
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, VIEW_AMBER_BGR, 1, cv2.LINE_AA)
    badge = str(view.get("badge") or "")
    if status == "ok" and conf < MIN_CONF:
        badge = "low conf"
    if badge:
        red = status in ("error", "refused")
        cv2.putText(img, badge, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    VIEW_ERROR_BGR if red else colour, 1, cv2.LINE_AA)
