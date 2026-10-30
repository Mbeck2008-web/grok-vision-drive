"""View-only scene-net lane viewer: a view-only same-map experiment.

Shows what a trained scene net (``lane_gt[14]``) outputs on the live cameras, drawn on the stage
in amber. It is not a lane source and it never drives: nothing here imports control, engage or
the actuator, and run_vision only stores ``SceneView.to_state()`` in ``st["scene_view"]``.

Rev 4: the forward runs in a child process (multiprocessing, spawn). The main thread
(``submit``) composes the strip with the training code (``compose_strip`` ->
``prepare_model_strip``), writes it into one shared-memory block and sends a small step message
over a pipe; at most one step is in flight (a busy child drops the step). The child owns the
``SceneLaneRuntime`` (ORT CPU, 2 intra-op threads), the GRU state, the 16-step window and the
resets, and sends back an immutable ``SceneView``. The main process never imports torch or
onnxruntime for the viewer, and a forward that holds a GIL holds the child's, not run_vision's.

The child dies with the parent: daemon process, ``PR_SET_PDEATHSIG`` on Linux, and a watchdog
thread on the parent's process sentinel (all platforms, the Windows path). It reports its RSS,
which counts toward the supervisor's memory budget. A switch shows badge "loading" (no lane
lines) until the new model's first result.

Steps: a new main frame, >= 90 ms since the last step, child idle. GRU reset every 16 steps
(``CHUNK_MAX``, the training window), on model switch, a respawn edge, a gap > ``MEMORY_S``.
Budget (P = 1/hz): skip when the previous heartbeat > 0.75 P or the loop gap > 1.5 P; main
slice EMA > 0.10 P pauses 2 s ("throttled"); forward EMA > 0.75 P pauses 2 s ("slow"); three
consecutive errors latch the viewer off until the next switch.
"""
from __future__ import annotations

import contextlib
import importlib
import math
import os
import sys
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
LOADING = "loading"
FRAME = "body_centreline"
CLOSE_JOIN_S = 0.3
WATCHDOG_POLL_S = 0.5


def _default_runtime(path: Path, **kw: Any) -> Any:
    from python.train.scene_lane_runtime import SceneLaneRuntime

    return SceneLaneRuntime(path, **kw)


# "module:attr" (or a callable, inline mode only). The child process imports it by name.
# Test hook: scripts/gvd_slow_view_child.py swaps in a slow fake runtime.
DEFAULT_RUNTIME_FACTORY: Any = "python.viz.scene_view:_default_runtime"


def _resolve_factory(spec: Any) -> Callable[..., Any]:
    """``"module:attr"`` | ``("module:attr", {kw})`` | callable -> a callable."""
    kwargs: dict[str, Any] = {}
    if isinstance(spec, (tuple, list)):
        spec, kwargs = spec[0], dict(spec[1] or {})
    if isinstance(spec, str):
        mod, _, attr = spec.partition(":")
        fn = getattr(importlib.import_module(mod), attr)
    else:
        fn = spec
    if not kwargs:
        return fn
    return lambda *a, **kw: fn(*a, **{**kwargs, **kw})


def _rss_mb() -> float:
    """This process's resident set, MB (0.0 when unknown)."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            proc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(kb) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class SceneView:
    """One published viewer result. Display only: no control fields."""

    stem: str = "off"
    status: str = "off"  # off | loading | ok | slow | throttled | error | refused
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
    child_rss_mb: float = 0.0
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


def _model_strip(frames: Mapping[str, Any], cams: tuple[str, ...], canvas_w: int) -> Any:
    strip = compose_strip({cam: frames[cam] for cam in cams if cam in frames})
    return prepare_model_strip(strip, tuple(cams), int(canvas_w))


def view_strip(frames: Mapping[str, Any], cams: tuple[str, ...], canvas_w: int) -> np.ndarray | None:
    """The model's canvas built by the training code; an owned copy."""
    out = _model_strip(frames, cams, canvas_w)
    return None if out is None else np.array(out, dtype=np.uint8, copy=True)


# ---- child side: owns the runtime, the GRU state and the resets ----------------------------


class _Engine:
    """The forward side. Runs in the viewer's child process (or inline, for tests).

    Messages in: ``("load", gen, stem)``, ``("step", gen, strip, ages, dt_s, reset)``.
    Replies: dicts with ``kind`` in loaded | refused | step | error | stale.
    """

    def __init__(self, models: Mapping[str, Path], can_feed_cam_age: bool, factory: Callable[..., Any]) -> None:
        self.models = {str(k): Path(v) for k, v in dict(models).items()}
        self.can_feed_cam_age = bool(can_feed_cam_age)
        self.factory = factory
        self.rt: Any = None
        self.gen = 0
        self.stem = "off"
        self.k = 0

    def load(self, gen: int, stem: str) -> dict[str, Any]:
        self.rt, self.gen, self.stem, self.k = None, int(gen), str(stem), 0
        if stem == "off":
            return {"kind": "loaded", "gen": gen, "info": None}
        try:
            from python.train.scene_lane_runtime import SceneCamAgeError, SceneInputShapeError

            path = self.models.get(stem)
            if path is None:
                return {"kind": "refused", "gen": gen, "badge": f"unknown model {stem}"}
            try:
                rt = self.factory(path, feeds_cam_age=self.can_feed_cam_age, intra_op_threads=VIEW_INTRA_OP_THREADS)
            except SceneCamAgeError as exc:
                badge = LANE71_BADGE if "8 per-camera" in str(exc) else "refused: cam_age contract"
                return {"kind": "refused", "gen": gen, "badge": badge}
            except SceneInputShapeError:
                return {"kind": "refused", "gen": gen, "badge": "refused: wrong strip width"}
            if not getattr(rt, "ready", False):
                return {"kind": "refused", "gen": gen, "badge": str(getattr(rt, "reason", "not ready"))}
        except Exception as exc:
            return {"kind": "error", "gen": gen, "error": type(exc).__name__}
        self.rt = rt
        info = {"cams": tuple(rt.cams), "canvas_w": int(rt.canvas_w), "cam_age": bool(rt.cam_age)}
        return {"kind": "loaded", "gen": gen, "info": info, "rss_mb": _rss_mb(), "pid": os.getpid()}

    def step(self, gen: int, strip: np.ndarray, ages: Any, dt_s: float | None, reset: bool) -> dict[str, Any]:
        rt = self.rt
        if rt is None or gen != self.gen:
            return {"kind": "stale", "gen": gen}
        try:
            if reset or self.k >= CHUNK:
                rt.reset_state()
                self.k = 0
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
            self.k += 1
            if self.k >= CHUNK:
                rt.reset_state()  # the window ends here; the state is fresh again
        except Exception as exc:
            return {"kind": "error", "gen": gen, "error": type(exc).__name__}
        rss = _rss_mb()
        view = SceneView(
            stem=self.stem, status="ok", badge="low conf" if float(conf) < MIN_CONF else "",
            lane_gt=tuple(float(v) for v in vec), conf=float(conf),
            lane_x=None if lane_x is None else float(lane_x),
            dt_ms=0.0 if dt_s is None else float(dt_s) * 1000.0, k=self.k, forward_ms=fwd_ms, child_rss_mb=rss,
        )
        return {"kind": "step", "gen": gen, "view": view, "forward_ms": fwd_ms, "rss_mb": rss}


def _die_with_parent(parent_pid: int) -> None:
    """Exit when the parent dies, however it dies (SIGKILL included)."""
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            import ctypes
            import signal

            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL))  # PR_SET_PDEATHSIG
    if os.name == "posix" and os.getppid() != parent_pid:
        os._exit(0)  # the parent died before the line above
    import multiprocessing
    from multiprocessing.connection import wait

    parent = multiprocessing.parent_process()

    def watch() -> None:
        sentinel = getattr(parent, "sentinel", None)
        if sentinel is not None:
            with contextlib.suppress(Exception):
                wait([sentinel])
                os._exit(0)
        while True:  # no sentinel: poll the parent pid
            time.sleep(WATCHDOG_POLL_S)
            if os.name == "posix" and os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch, name="gvd-scene-view-watchdog", daemon=True).start()


def _attach(name: str) -> Any:
    from multiprocessing import shared_memory

    # A spawn child shares the parent's resource tracker, so attaching here registers the name
    # twice in one set (a no-op); the parent's unlink() unregisters it.
    return shared_memory.SharedMemory(name=name)


def _child_main(conn: Any, models: dict, can_feed_cam_age: bool, factory_spec: Any, parent_pid: int) -> None:
    _die_with_parent(parent_pid)
    engine = _Engine(models, can_feed_cam_age, _resolve_factory(factory_spec))
    shm: Any = None
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            if msg is None:
                return
            kind = msg[0]
            if kind == "load":
                reply = engine.load(msg[1], msg[2])
            elif kind == "step":
                _, gen, name, shape, dtype, ages, dt_s, reset = msg
                if shm is None or shm.name != name:
                    if shm is not None:
                        shm.close()
                    shm = _attach(name)
                view = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                strip = np.array(view, copy=True)
                del view
                reply = engine.step(gen, strip, ages, dt_s, reset)
            elif kind == "modules":
                reply = {"kind": "modules", "names": sorted({m.split(".")[0] for m in sys.modules} & set(msg[1]))}
            else:
                continue
            conn.send(reply)
    finally:
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()


@contextlib.contextmanager
def _no_main_reimport():
    """Keep spawn from re-importing run_vision (or a test script) as ``__mp_main__`` in the child."""
    main = sys.modules.get("__main__")
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if main is not None and k in main.__dict__}
    try:
        if main is not None:
            main.__dict__.pop("__file__", None)
            main.__dict__["__spec__"] = None
        yield
    finally:
        if main is not None:
            main.__dict__.pop("__spec__", None)
            main.__dict__.update(saved)


# ---- main side --------------------------------------------------------------------------


class SceneNetViewer:
    """A scene-net runtime in one child process; display only.

    ``inline=True`` (tests only) runs the same forward code synchronously in this process.
    """

    def __init__(
        self,
        models: Mapping[str, Path],
        *,
        hz: float,
        can_feed_cam_age: bool,
        runtime_factory: Any = None,
        inline: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        log: Callable[[str], None] | None = print,
    ) -> None:
        self.models = {str(k): Path(v) for k, v in dict(models).items()}
        self.period_s = 1.0 / max(1e-3, float(hz))
        self.can_feed_cam_age = bool(can_feed_cam_age)
        self.factory = runtime_factory if runtime_factory is not None else DEFAULT_RUNTIME_FACTORY
        self.inline = bool(inline)
        if not self.inline and callable(self.factory):
            raise TypeError("runtime_factory must be 'module:attr' (it is built in the child process)")
        self.clock = clock
        self.log = log
        self.stem = "off"
        self.dropped = 0
        self.skipped = 0
        self.steps = 0
        self.child: Any = None
        self.child_rss_mb = 0.0
        self._conn: Any = None
        self._shm: Any = None
        self._gen = 0
        self._view = SceneView()
        self._info: dict[str, Any] | None = None
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
        self._closed = False
        self._engine = _Engine(self.models, self.can_feed_cam_age, _resolve_factory(self.factory)) if self.inline else None
        self._inbox: list[dict[str, Any]] = []

    # ---- public -------------------------------------------------------------------------

    @property
    def runtime(self) -> Any:
        """The runtime object (inline mode only; in the child process otherwise)."""
        return None if self._engine is None else self._engine.rt

    @property
    def shm_name(self) -> str | None:
        return None if self._shm is None else self._shm.name

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
        self._view = SceneView(stem=stem, status="off") if stem == "off" else SceneView(stem=stem, status=LOADING, badge=LOADING)
        if stem == "off" and self.child is None and self._engine is None:
            return
        try:
            self._send(("load", self._gen, stem))
        except Exception as exc:
            self._error_name(type(exc).__name__)

    def observe_respawn(self, vdata: Any, tick_dt: float | None, bundle: Any, backend: Any) -> None:
        if self._respawn.observe(vdata, tick_dt, mono=time.monotonic()) is None:
            return
        if self._hold_on and backend is not None:
            backend.reset_live_hold()
            scrub_after_respawn(bundle)
        self._reset_pending = True

    def submit(self, bundle: Any, *, backend: Any, prev_heartbeat_ms: float, loop_gap_s: float) -> float:
        """Send one step if the gates allow; returns this call's main-thread ms. Never raises."""
        t0 = self.clock()
        try:
            self._drain()
            self._submit(bundle, backend, float(prev_heartbeat_ms or 0.0), float(loop_gap_s or 0.0), t0)
            self._drain()  # inline mode: the reply is already here
        except Exception as exc:
            self._error_name(type(exc).__name__)
        ms = max(0.0, (self.clock() - t0) * 1000.0)
        if self._info is not None:
            self._slice_ema = ms if self._slice_ema is None else (1 - EMA_ALPHA) * self._slice_ema + EMA_ALPHA * ms
            if self._slice_ema > 0.10 * self.period_s * 1000.0:
                self._pause(t0, "throttled")
                self._slice_ema = None
        return ms

    def latest(self) -> SceneView:
        """The last result with this side's counters. Never raises."""
        try:
            self._drain()
            view = self._view
            status, badge = view.status, view.badge
            if status == "ok" and self.clock() < self._pause_until:
                status = badge = self._pause_status
            return SceneView(**{**asdict(view), "lane_gt": view.lane_gt, "status": status, "badge": badge,
                                "steps": self.steps, "dropped": self.dropped, "skipped": self.skipped,
                                "child_rss_mb": self.child_rss_mb})
        except Exception as exc:
            return SceneView(stem=str(getattr(self, "stem", "off")), status="error", badge=f"error: {type(exc).__name__}")

    def child_modules(self, names: tuple[str, ...], timeout_s: float = 10.0) -> list[str]:
        """Which of ``names`` the child has imported (diagnostics and tests; blocks)."""
        if self._conn is None:
            return []
        self._conn.send(("modules", tuple(names)))
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            if self._conn.poll(0.05):
                msg = self._conn.recv()
                if msg.get("kind") == "modules":
                    return list(msg["names"])
                self._handle(msg)
        raise TimeoutError("child did not answer")

    def close(self) -> None:
        """Stop the child (<= ~0.5 s) and free the shared memory. Never raises."""
        self._closed = True
        proc, conn = self.child, self._conn
        if proc is not None:
            with contextlib.suppress(Exception):
                conn.send(None)
            with contextlib.suppress(Exception):
                proc.join(CLOSE_JOIN_S)
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.join(0.2)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        self._conn = None
        self._free_shm()

    # ---- main thread internals ----------------------------------------------------------

    def _pause(self, now: float, status: str) -> None:
        self._pause_until = now + PAUSE_S
        self._pause_status = status

    def _start_child(self) -> None:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_child_main, name="gvd-scene-view", daemon=True,
            args=(child_conn, {k: str(v) for k, v in self.models.items()}, self.can_feed_cam_age,
                  self.factory, os.getpid()),
        )
        with _no_main_reimport():
            proc.start()
        child_conn.close()
        self.child, self._conn = proc, parent_conn

    def _send(self, msg: tuple) -> None:
        if self._engine is not None:
            if msg[0] == "load":
                self._inbox.append(self._engine.load(msg[1], msg[2]))
            elif msg[0] == "step":
                _, gen, strip, ages, dt_s, reset = msg
                self._inbox.append(self._engine.step(gen, strip, ages, dt_s, reset))
            return
        if self._closed:
            raise RuntimeError("viewer closed")
        if self.child is None:
            self._start_child()
        self._conn.send(msg)

    def _drain(self) -> None:
        while self._inbox:
            self._handle(self._inbox.pop(0))
        conn = self._conn
        if conn is None:
            return
        try:
            while conn.poll(0):
                self._handle(conn.recv())
        except (EOFError, OSError):
            self._child_died()

    def _child_died(self) -> None:
        code = None if self.child is None else self.child.exitcode
        with contextlib.suppress(Exception):
            self._conn.close()
        self._conn = None
        self.child = None
        self._busy = False
        self._info = None
        if not self._closed:
            self._errors = ERR_LATCH - 1
            self._error_name(f"child exited ({code})")

    def _handle(self, msg: dict[str, Any]) -> None:
        kind, gen = msg.get("kind"), msg.get("gen")
        if kind in ("step", "error", "stale"):
            self._busy = False
        if gen != self._gen:
            return
        if kind == "loaded":
            self._info = msg.get("info")
            if self._info is not None:
                self.child_rss_mb = float(msg.get("rss_mb") or 0.0)
                if self.log is not None and msg.get("pid") is not None:
                    self.log(f"[GVD] scene view child pid={msg['pid']} rss={self.child_rss_mb:.0f} MB "
                             f"model={self.stem} (view-only; counted in the 12 GB supervisor budget)")
        elif kind == "refused":
            self._info = None
            self._view = SceneView(stem=self.stem, status="refused", badge=str(msg.get("badge", "")))
        elif kind == "step":
            fwd = float(msg.get("forward_ms") or 0.0)
            self._fwd_ema = fwd if self._fwd_ema is None else (1 - EMA_ALPHA) * self._fwd_ema + EMA_ALPHA * fwd
            self.child_rss_mb = float(msg.get("rss_mb") or self.child_rss_mb)
            self._errors = 0
            self._view = msg["view"]
        elif kind == "error":
            self._error_name(str(msg.get("error", "Error")))

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
        strip = _model_strip(frames, info["cams"], info["canvas_w"])
        if strip is None:
            raise ValueError("no strip at the model canvas")
        dt = None if self._last_step_t is None else now - self._last_step_t
        reset = self._reset_pending or (dt is not None and dt > GAP_RESET_S)
        self._reset_pending = False
        self._last_key = key
        self._last_step_t = now
        self._busy = True
        self.steps += 1
        dt_s = None if reset else dt
        ages = None if ages is None else np.array(ages, copy=True)
        if self._engine is not None:
            self._send(("step", self._gen, np.array(strip, dtype=np.uint8, copy=True), ages, dt_s, reset))
            return
        arr = np.asarray(strip, dtype=np.uint8)
        shm = self._shm_for(arr.nbytes)
        dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        dst[...] = arr
        del dst
        try:
            self._send(("step", self._gen, shm.name, arr.shape, arr.dtype.str, ages, dt_s, reset))
        except Exception:
            self._busy = False
            raise

    def _shm_for(self, nbytes: int) -> Any:
        if self._shm is not None and self._shm.size >= nbytes:
            return self._shm
        from multiprocessing import shared_memory

        self._free_shm()
        self._shm = shared_memory.SharedMemory(create=True, size=max(1, int(nbytes)))
        return self._shm

    def _free_shm(self) -> None:
        shm, self._shm = self._shm, None
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                shm.unlink()

    def _error_name(self, name: str) -> None:
        self._errors += 1
        badge = f"error: {name}"
        if self._errors >= ERR_LATCH:
            self._latched = True
            badge += " (latched off)"
        self._view = SceneView(stem=self.stem, status="error", badge=badge)


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
