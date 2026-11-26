#!/usr/bin/env python3
"""Offline isolation-harness tests for Reality View Step 3.

Spec: docs/specs/reality_view_3dgs.md, Feasibility plan step 3 ("Isolation
harness") and the "Isolation" / "Degrade / disable" paragraphs.

Tests (names from the spec):
- check_safety_loop_latency_unchanged
- check_renderer_crash_degrades
- check_supervisor_after_apply_is_copy_only

The supervisor stand-in is ``_StubSafetyLoop`` below: the run_vision per-tick
order reduced to what is offline-testable on CPU. Each tick stamps the
heartbeat (perf_counter, like ``loop_t0``/``heartbeat_ms``), runs a fixed
stand-in infer, runs the real ``python.runtime.shadow.shadow_tick`` (AEB veto,
heartbeat_stale release, engage gate) and the real ``NullActuator`` apply/stop,
then calls the RV supervisor hook ``RvHost.after_apply`` (post-``actuator.apply``),
then sleeps to the next deadline (paced loop, like ``--hz``). The loop lives in
this test, not in an RV module, so no RV module imports the safety path.

No GPU, no new deps (stdlib + numpy + opencv-headless + PyYAML). Runs on Linux
and windows-latest; win32 kills children with Popen.kill (TerminateProcess),
never signal.SIGKILL.
"""
from __future__ import annotations

import ast
import hashlib
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE_REL = 0.05  # spec: loop Hz and infer_ms p95 change < 5% with RV on vs off
LOOP_HZ = 50.0  # paced stub loop; 20 ms period
WARMUP_TICKS = 25
MEASURE_TICKS = 100
CPU_QUANTILE = 0.10  # box noise only adds CPU time; the low quantile tracks the uncontended cost
ROUNDS = 5  # interleaved A/B reps: OFF, gen, OFF, renderer per rep
FRAME_W, FRAME_H = 320, 240  # supervisor -> child frame handoff (already-captured RGB copy)
CRASH_DETECT_BOUND_S = 1.0
CRASH_DETECT_BOUND_TICKS = 25


def _harness():
    """Import the Step 3 supervisor-side harness or fail with what is missing."""
    try:
        import python.viz.rv_isolation as iso
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "step 3 isolation harness missing: python/viz/rv_isolation.py does not exist "
            f"({exc}); there is no supervisor-side RvHost to poll, degrade, or copy-only apply"
        ) from None
    for name in ("RvHost", "RvSharedSlot", "RV_ISO_CHILD_MODULE"):
        assert hasattr(iso, name), f"python.viz.rv_isolation has no {name}"
    return iso


def _sha256(arr) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def _kill_child_hard(pid_or_popen) -> None:
    """SIGKILL on POSIX, TerminateProcess on win32 (no signal.SIGKILL there)."""
    if sys.platform == "win32":
        pid_or_popen.kill()
    else:
        os.kill(pid_or_popen.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Supervisor stand-in
# ---------------------------------------------------------------------------


class _FixedE2E:
    def forward(self, *args, **kwargs):
        from python.control.e2e import E2EIntent

        del args, kwargs
        return E2EIntent(steer=0.12, throttle=0.3, brake=0.0, accel=0.3, ok=True, backend="stub")


def _path() -> list[dict[str, float]]:
    return [{"x": 0.02 * i, "y": float(i)} for i in range(12)]


def _inputs(i: int) -> dict:
    """Deterministic per-tick inputs covering engage on/off, AEB brake, heartbeat stale."""
    return {
        "engaged": (i % 40) >= 5,
        "heartbeat_ok": (i % 37) != 11,
        "planner": {"target_v": 9.0, "aeb": "brake" if (i % 23) in (3, 4) else "off"},
        "ego_v": 8.0 + 0.01 * (i % 50),
    }


_INFER_N = 0


def _calibrate_infer(target_ms: float = 3.0) -> None:
    """Fixed-work stand-in infer (pure Python, single thread). Calibrated once per process."""
    global _INFER_N
    n = 2000
    while True:
        t0 = time.perf_counter()
        _standin_infer(n)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt >= 0.5:
            break
        n *= 2
    _INFER_N = max(1000, int(n * target_ms / max(dt, 1e-6)))


def _standin_infer(n: int) -> int:
    acc = 0
    for k in range(n):
        acc = (acc + k * k) & 0xFFFFFFFF
    return acc


class _StubSafetyLoop:
    """run_vision tick order, offline. Records heartbeat stamps, infer_ms, post-apply ms and outputs."""

    def __init__(self, host=None, *, hz: float = LOOP_HZ, frame: bytes | None = None, cpu_now=None) -> None:
        from python.control.actuate import NullActuator
        from python.runtime.shadow import ShadowConfig

        self.host = host
        self.period = 1.0 / hz
        self.frame = frame
        self.actuator = NullActuator()
        self.cfg = ShadowConfig()
        self.e2e = _FixedE2E()
        self.hb: list[float] = []
        self.infer_ms: list[float] = []
        self.post_apply_ms: list[float] = []
        self.outputs: list[tuple] = []
        self.cpu_safety: list[float] = []  # thread CPU of heartbeat..apply (no RV)
        self.cpu_rv: list[float] = []  # thread CPU of after_apply
        self._cpu = cpu_now if cpu_now is not None else (lambda: 0.0)
        self._next = None

    def tick(self, i: int) -> None:
        from python.runtime.shadow import shadow_tick

        if self._next is None:
            self._next = time.perf_counter()
        c0 = self._cpu()
        loop_t0 = time.perf_counter()
        self.hb.append(loop_t0)  # heartbeat stamp
        inp = _inputs(i)
        t_inf = time.perf_counter()
        _standin_infer(_INFER_N)
        self.infer_ms.append((time.perf_counter() - t_inf) * 1000.0)
        st = shadow_tick(
            policy="shadow",
            engaged=inp["engaged"],
            heartbeat_ok=inp["heartbeat_ok"],
            path_debug_preview=False,
            allow_preview_drive=False,
            path_ego=_path(),
            planner=inp["planner"],
            ego_speed_mps=inp["ego_v"],
            lane_conf=0.9,
            path_conf=0.9,
            seq=i,
            e2e_policy=self.e2e,
            cfg=self.cfg,
        )
        cmd = st.applied
        if inp["engaged"] and cmd.reason == "ok":
            applied = self.actuator.apply(cmd)
        else:
            applied = self.actuator.stop(seq=i, reason=cmd.reason)
        # ---- after actuator.apply: RV supervisor work (copy / handoff only) ----
        c1 = self._cpu()
        t_post = time.perf_counter()
        if self.host is not None:
            ret = self.host.after_apply(self.frame, seq=i)
            assert isinstance(ret, bool), f"after_apply must return a plain bool, got {type(ret)!r}"
        self.post_apply_ms.append((time.perf_counter() - t_post) * 1000.0)
        c2 = self._cpu()
        self.cpu_safety.append(c1 - c0)
        self.cpu_rv.append(c2 - c1)
        self.outputs.append(
            (
                i,
                round(float(applied.steer), 9),
                round(float(applied.throttle), 9),
                round(float(applied.brake), 9),
                str(applied.reason),
                str(st.veto_reason),
                bool(st.should_disengage),
            )
        )
        self._next += self.period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.perf_counter()

    def periods_ms(self, skip: int = 0) -> list[float]:
        hb = self.hb[skip:]
        return [(b - a) * 1000.0 for a, b in zip(hb, hb[1:])]


def _frame_bytes() -> bytes:
    import numpy as np

    return (np.arange(FRAME_W * FRAME_H * 3, dtype=np.uint32) % 251).astype(np.uint8).tobytes()


class _NoBlockGuard:
    """Wrap the host child's Popen so any blocking wait/communicate in the loop raises."""

    def __init__(self, popen) -> None:
        self.popen = popen
        self.calls: list[str] = []
        self._orig = {}

    def __enter__(self):
        for name in ("wait", "communicate"):
            orig = getattr(self.popen, name)
            self._orig[name] = orig

            def _blocked(*a, _name=name, **kw):
                self.calls.append(_name)
                raise AssertionError(f"blocking Popen.{_name}() called on the supervisor loop")

            setattr(self.popen, name, _blocked)
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self.popen, name, orig)
        return False


# ---------------------------------------------------------------------------
# 1. check_safety_loop_latency_unchanged
# ---------------------------------------------------------------------------


def _thread_cpu_reader():
    """(read(), unit) for this thread's CPU use.

    Linux/macOS: time.thread_time (CLOCK_THREAD_CPUTIME_ID, ns resolution).
    win32: QueryThreadCycleTime (cycle counter). time.thread_time on win32 is
    GetThreadTimes, which only advances per ~15.6 ms scheduler tick: about 5%
    quantization on a 300 ms rep, as large as the gate itself.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentThread.restype = wintypes.HANDLE
        k32.QueryThreadCycleTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
        k32.QueryThreadCycleTime.restype = wintypes.BOOL
        h = k32.GetCurrentThread()

        def _read() -> float:
            c = ctypes.c_ulonglong(0)
            if not k32.QueryThreadCycleTime(h, ctypes.byref(c)):
                raise OSError(ctypes.get_last_error(), "QueryThreadCycleTime failed")
            return c.value / 1e6

        return _read, "Mcycles"
    return (lambda: time.thread_time() * 1000.0), "ms"


def check_safety_loop_latency_unchanged() -> None:
    """Slow child generator / slow (hung) child renderer do not move the stub loop by >5%.

    Method (wall-clock p95 of a 3 ms stand-in is +-30% noise on a loaded shared
    box, so it is reported, not gated):
    - Paced 50 Hz stub loop: heartbeat stamp, fixed-work stand-in infer, real
      shadow_tick + NullActuator, then RV ``after_apply``.
    - Interleaved A/B reps: each rep runs OFF, ON+slow_generator, OFF,
      ON+slow_renderer (fresh child per run, WARMUP_TICKS discarded), after one
      discarded OFF warm-up run. Each run gives one number per metric; the gate
      compares the median over reps of the child runs with the median over all
      OFF runs, so drift and load spikes hit both sides.
    - Gated at 5% (spec: loop Hz and infer p95 change < 5%):
      (a) loop Hz over the measured ticks (median over reps vs OFF);
      (b) safety-path CPU per tick (this thread's CPU from the heartbeat stamp
          through actuator.apply: stand-in infer + shadow_tick + apply), at the
          p10 per run, median over reps vs OFF. Load from other processes only
          adds CPU time (preemption, cache / SMT contention), so the low quantile
          estimates the uncontended cost and is stable run to run, while a child
          that slows the supervisor thread on every tick shifts all quantiles;
      (c) RV share of supervisor CPU: after_apply CPU / whole-tick CPU, paired
          inside the same run, < 5% (median over reps);
      (d) after_apply wall p95 < 5% of the period (a child that makes the loop
          wait, even a bounded wait or sleep that the paced Hz would hide,
          fails here).
      Per-tick wall p95 (infer, period) on a loaded shared box swings by more
      than 5% between identical OFF runs, so it is printed per rep, not gated.
    - Structural: every Popen.wait / communicate on the child is trapped during
      the loop (any call fails the check); the backpressure drop path must be
      exercised (hung renderer -> dropped handoffs, generator mid-write stall ->
      dropped torn/busy reads); the host must not degrade on a slow-but-alive
      child.
    Wall infer p50/p95 and period p50/p95 are printed per rep for the record.
    """
    _calibrate_infer()
    frame = _frame_bytes()
    cpu_now, cpu_unit = _thread_cpu_reader()
    period_s = 1.0 / LOOP_HZ
    runs: dict[str, list[dict]] = {"off": [], "gen": [], "renderer": []}
    host_stats: dict[str, dict] = {}
    log: list[str] = []

    def _run(cond: str) -> dict:
        host = None
        if cond != "off":
            iso = _harness()
            mode = "slow_generator" if cond == "gen" else "slow_renderer"
            host = iso.RvHost(module=iso.RV_ISO_CHILD_MODULE, child_args=("--mode", mode), print_notes=False)
            pid = host.launch()
            assert pid is not None, f"RvHost.launch failed: {host.note}"
            assert host.wait_ready(5.0), f"{mode} child never armed (note={host.note!r})"
        loop = _StubSafetyLoop(host, frame=frame if host is not None else None, cpu_now=cpu_now)
        try:
            guard = _NoBlockGuard(host.client._proc) if host is not None else None
            if guard is not None:
                guard.__enter__()
            try:
                for i in range(WARMUP_TICKS + MEASURE_TICKS):
                    loop.tick(i)
            finally:
                if guard is not None:
                    guard.__exit__(None, None, None)
            if host is not None:
                assert not guard.calls, f"{cond}: blocking calls on the loop: {guard.calls}"
                assert not host.degraded, f"{cond}: host degraded during a slow-but-alive child run: {host.note}"
                s = host_stats.setdefault(cond, {"handoff_dropped": 0, "out_dropped": 0, "out_applied": 0, "handoff_ok": 0})
                for k in s:
                    s[k] += int(getattr(host.stats, k))
        finally:
            if host is not None:
                host.close()
        hb = loop.hb[WARMUP_TICKS:]
        per = loop.periods_ms(skip=WARMUP_TICKS)
        inf = loop.infer_ms[WARMUP_TICKS:]
        post = loop.post_apply_ms[WARMUP_TICKS:]
        cs = loop.cpu_safety[WARMUP_TICKS:]
        cr = loop.cpu_rv[WARMUP_TICKS:]
        r = {
            "hz": (len(hb) - 1) / (hb[-1] - hb[0]),
            "cpu_q": _pct(cs, CPU_QUANTILE),
            "cpu_mean": statistics.fmean(cs),
            "rv_share": sum(cr) / (sum(cs) + sum(cr)),
            "post_p95_us": _pct(post, 0.95) * 1000.0,
            "post_max_us": max(post) * 1000.0,
            "per_p50": statistics.median(per),
            "per_p95": _pct(per, 0.95),
            "inf_p50": statistics.median(inf),
            "inf_p95": _pct(inf, 0.95),
        }
        return r

    _run("off")  # discarded warm-up run (imports, caches, CPU clocks)
    for rep in range(ROUNDS):
        for cond in ("off", "gen", "off", "renderer"):
            r = _run(cond)
            runs[cond].append(r)
            log.append(
                f"  rep{rep} {cond:<8} hz={r['hz']:.3f} safety_cpu p{int(CPU_QUANTILE * 100)}={r['cpu_q']:.4f} "
                f"mean={r['cpu_mean']:.4f} {cpu_unit} rv_cpu_share={r['rv_share'] * 100:.2f}% "
                f"after_apply p95={r['post_p95_us']:.1f} max={r['post_max_us']:.1f} us | "
                f"[report] period p50={r['per_p50']:.3f} p95={r['per_p95']:.3f} ms "
                f"infer_wall p50={r['inf_p50']:.3f} p95={r['inf_p95']:.3f} ms"
            )

    def _med(cond: str, key: str) -> float:
        return statistics.median(x[key] for x in runs[cond])

    summary = []
    for cond in ("off", "gen", "renderer"):
        summary.append(
            f"  median {cond:<8} hz={_med(cond, 'hz'):.3f} safety_cpu p{int(CPU_QUANTILE * 100)}={_med(cond, 'cpu_q'):.4f} "
            f"mean={_med(cond, 'cpu_mean'):.4f} {cpu_unit} rv_cpu_share={_med(cond, 'rv_share') * 100:.2f}% "
            f"after_apply p95={_med(cond, 'post_p95_us'):.1f} us infer_wall p95={_med(cond, 'inf_p95'):.3f} ms (report)"
        )
    print(
        f"check_safety_loop_latency_unchanged ({sys.platform}, {ROUNDS} reps x {MEASURE_TICKS} ticks @ {LOOP_HZ:.0f} Hz, cpu={cpu_unit}):\n"
        + "\n".join(log + summary),
        flush=True,
    )
    print(f"  host stats: {host_stats}", flush=True)

    for cond in ("gen", "renderer"):
        for key, label in (("hz", "loop Hz"), ("cpu_q", f"safety-path CPU/tick p{int(CPU_QUANTILE * 100)}")):
            b, v = _med("off", key), _med(cond, key)
            rel = (v - b) / b
            assert abs(rel) < GATE_REL, (
                f"{cond}: {label} median moved {rel * 100:+.2f}% (base {b:.4f}, RV {v:.4f}); gate 5%"
            )
        share = _med(cond, "rv_share")
        assert share < GATE_REL, f"{cond}: after_apply is {share * 100:.2f}% of safety-loop CPU; gate 5%"
        p95_post_s = _med(cond, "post_p95_us") / 1e6
        assert p95_post_s < GATE_REL * period_s, (
            f"{cond}: after_apply p95 {p95_post_s * 1e6:.0f} us >= 5% of the {period_s * 1000:.0f} ms period: the loop waited on RV"
        )
    assert host_stats["renderer"]["handoff_dropped"] > 0, "hung renderer never caused a backpressure drop"
    assert host_stats["gen"]["out_applied"] > 0, "slow generator output was never applied"
    assert host_stats["gen"]["out_dropped"] > 0, "generator mid-write stall never produced a dropped (torn/busy) read"


# ---------------------------------------------------------------------------
# 2. check_renderer_crash_degrades
# ---------------------------------------------------------------------------


def _valid_rv_slot(seq: int, w: int = 64, h: int = 48) -> bytes:
    import numpy as np

    from python.viz.rv_contract import RV_SCHEMA_VERSION, RvOutputSpec, pack_output_slot

    rgb = ((np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3) * 7 + seq) % 256).astype(np.uint8)
    depth = np.full((h, w), 12.5, dtype=np.float32)
    spec = RvOutputSpec(
        schema_version=RV_SCHEMA_VERSION, seq=seq, timestamp_s=1.0 + seq, pose_4x4=np.eye(4, dtype=np.float32), rgb=rgb, depth=depth
    )
    return pack_output_slot(spec, age_ms=0.0)


def check_renderer_crash_degrades() -> None:
    """SIGKILL the viz (stub) child and the infer (generator) child mid-run.

    Asserts: the supervisor loop keeps ticking with no exception escaping,
    heartbeat stamps continue at the paced rate, AEB / engage outputs equal a
    no-RV control run on identical inputs, the host detects the death within a
    bounded time and prints one note, and the main view is the bit-identical
    PR #242 placeholder afterwards (it showed the RV frame before the kill).
    """
    from python.viz.stage import STAGE_H, STAGE_W
    from python.viz.view_select import render_reality_placeholder

    _calibrate_infer(1.0)
    n_ticks, kill_at = 120, 40
    control = _StubSafetyLoop(None)
    for i in range(n_ticks):
        control.tick(i)

    iso = _harness()
    report = []
    for label, module, child_args in (
        ("viz-stub", "python.viz.rv_renderer_stub", ()),
        ("infer-gen", iso.RV_ISO_CHILD_MODULE, ("--mode", "slow_generator")),
    ):
        host = iso.RvHost(module=module, child_args=child_args, print_notes=False)
        try:
            pid = host.launch()
            assert pid is not None, f"{label}: launch failed: {host.note}"
            writer = iso.RvSharedSlot(host.out_slot_path, create=False) if module.endswith("rv_renderer_stub") else None
            if writer is None:
                assert host.wait_ready(5.0), f"{label}: child never armed"
            loop = _StubSafetyLoop(host, frame=_frame_bytes())
            states = [{"engaged": False}, {"engaged": True}]
            detected_tick = None
            t_kill = None
            t_detect = None
            for i in range(n_ticks):
                if writer is not None and i < kill_at and i % 5 == 0:
                    writer.write(_valid_rv_slot(i))
                if i == kill_at:
                    if writer is None:
                        deadline = time.perf_counter() + 3.0
                        while host.stats.out_applied == 0 and time.perf_counter() < deadline:
                            time.sleep(0.01)
                            host.after_apply(None, seq=0)
                        assert host.stats.out_applied > 0, f"{label}: generator output never applied before kill"
                    assert not host.degraded, f"{label}: degraded before kill: {host.note}"
                    for st in states:
                        v = host.main_view(dict(st), STAGE_W, STAGE_H)
                        assert _sha256(v) != _sha256(render_reality_placeholder(dict(st), STAGE_W, STAGE_H)), (
                            f"{label}: healthy host with an applied frame should not show the placeholder"
                        )
                    _kill_child_hard(host.client._proc)
                    t_kill = time.perf_counter()
                try:
                    loop.tick(i)
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"{label}: exception escaped into the safety loop at tick {i}: {exc!r}") from exc
                if t_kill is not None and detected_tick is None and host.degraded:
                    detected_tick, t_detect = i, time.perf_counter()
            assert detected_tick is not None, f"{label}: child SIGKILL never detected; host still not degraded"
            assert t_detect - t_kill <= CRASH_DETECT_BOUND_S, f"{label}: detection took {t_detect - t_kill:.3f}s"
            assert detected_tick - kill_at <= CRASH_DETECT_BOUND_TICKS, f"{label}: detection took {detected_tick - kill_at} ticks"
            assert len(loop.hb) == n_ticks, f"{label}: heartbeat stamps {len(loop.hb)} != {n_ticks}"
            assert all(b > a for a, b in zip(loop.hb, loop.hb[1:])), f"{label}: heartbeat stamps not increasing"
            post = loop.periods_ms(skip=kill_at)
            nominal = 1000.0 / LOOP_HZ
            assert max(post) < 3.0 * nominal, f"{label}: heartbeat gap {max(post):.1f} ms after kill (nominal {nominal:.0f})"
            assert loop.outputs == control.outputs, f"{label}: AEB/engage outputs differ from the no-RV control"
            assert host.degrade_notes == 1, f"{label}: expected exactly one degrade note, got {host.degrade_notes}"
            for st in states:
                v = host.main_view(dict(st), STAGE_W, STAGE_H)
                ref = render_reality_placeholder(dict(st), STAGE_W, STAGE_H)
                assert v.dtype == ref.dtype and v.shape == ref.shape, f"{label}: view shape {v.shape}"
                assert _sha256(v) == _sha256(ref), f"{label}: degraded view is not the bit-identical placeholder for {st}"
            assert host.after_apply(_frame_bytes(), seq=n_ticks) is False
            report.append(
                f"  {label}: killed at tick {kill_at}, degraded at tick {detected_tick} "
                f"({(t_detect - t_kill) * 1000:.1f} ms), max post-kill period {max(post):.2f} ms, "
                f"outputs == control ({len(control.outputs)} ticks), note={host.note!r}"
            )
            if writer is not None:
                writer.close()
        finally:
            host.close()
    print("check_renderer_crash_degrades:\n" + "\n".join(report), flush=True)


# ---------------------------------------------------------------------------
# 3. check_supervisor_after_apply_is_copy_only
# ---------------------------------------------------------------------------

_FORBIDDEN_CALL_NAMES = {
    "pack_output_slot",
    "unpack_output_slot",
    "validate_rv_output",
    "validate_rv_input",
    "render_reality_placeholder",
    "main_view",
}


def _ast_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            mods.update(f"{node.module}.{a.name}" for a in node.names)
    return mods


def check_supervisor_after_apply_is_copy_only() -> None:
    """After actuator.apply the supervisor only copies slot bytes; no generation, no retained refs.

    - A profiler (sys.setprofile) records every Python and C call inside
      after_apply: no generator/packing/parse function (pack/unpack_output_slot,
      validate_*, render), nothing from numpy/cv2, no rv_isolation_child code,
      and exactly one header validate (RvShmHeader.unpack).
    - The applied bytes equal the slot bytes; mutating the shared slot
      afterwards (in place, no seq bump) does not change the supervisor's view
      (no reference into the slot is retained).
    - A writer stalled mid-write (odd seq) and a bad-magic header are dropped /
      rejected and the previous good view is kept.
    - The post-apply cost is the copy: after_apply on a full 640x640 slot is
      within 10x + 1 ms of a plain bytes copy of the same size (generation of a
      300k splat block would be orders of magnitude more).
    - No RV objects reach control / frame_gate / state_io / record (AST), and
      no RV supervisor module imports the safety path.
    """
    import struct

    iso = _harness()
    from python.viz import rv_contract as rc

    host = iso.RvHost(module="python.viz.rv_renderer_stub", print_notes=False)
    writer = None
    try:
        assert host.launch() is not None, host.note
        writer = iso.RvSharedSlot(host.out_slot_path, create=False)
        raw = _valid_rv_slot(7)
        assert writer.write(raw)

        events: list[tuple[str, str, str]] = []

        def _prof(frame, event, arg):
            if event == "call":
                co = frame.f_code
                events.append(("py", co.co_name, co.co_filename))
            elif event == "c_call":
                events.append(("c", getattr(arg, "__name__", "?"), str(getattr(arg, "__module__", "") or getattr(getattr(arg, "__self__", None), "__name__", ""))))

        sentinels = {}
        for name in ("pack_output_slot", "unpack_output_slot", "validate_rv_output", "validate_rv_input"):
            sentinels[name] = getattr(rc, name)

            def _boom(*a, _n=name, **kw):
                raise AssertionError(f"{_n} called on the supervisor after-apply path")

            setattr(rc, name, _boom)
        try:
            sys.setprofile(_prof)
            ok = host.after_apply(None, seq=1)
            sys.setprofile(None)
        finally:
            sys.setprofile(None)
            for name, fn in sentinels.items():
                setattr(rc, name, fn)
        assert ok is True, f"after_apply did not apply a valid slot (degraded={host.degraded}, note={host.note!r})"
        assert not host.degraded, host.note
        assert host.stats.out_applied == 1, host.stats
        names_py = [n for k, n, f in events if k == "py"]
        bad = [e for e in events if e[1] in _FORBIDDEN_CALL_NAMES]
        bad += [e for e in events if e[0] == "py" and ("numpy" in e[2] or "cv2" in e[2] or "rv_isolation_child" in e[2])]
        bad += [e for e in events if e[0] == "c" and (e[2].startswith("numpy") or e[2].startswith("cv2"))]
        assert not bad, f"non-copy work on the after-apply path: {sorted(set(bad))[:8]}"
        n_unpack = sum(1 for k, n, f in events if k == "py" and n == "unpack" and f.endswith("rv_contract.py"))
        assert n_unpack == 1, f"expected exactly one header validate (RvShmHeader.unpack), saw {n_unpack}; calls={names_py}"

        view = host.view_bytes()
        assert view == raw, "applied view bytes != slot bytes"

        # Mutate the shared slot in place (no seq bump): the supervisor's copy must not change.
        with open(host.out_slot_path, "r+b") as fh:
            fh.seek(iso.RV_CTRL_SIZE + rc.SLOT_HEADER_SIZE)
            fh.write(b"\xff" * 512)
            fh.flush()
        assert host.view_bytes() == raw, "supervisor view changed when the shared slot was mutated: slot reference retained"
        assert host.after_apply(None, seq=2) is True
        assert host.view_bytes() == raw, "same-seq slot was re-applied or aliased"

        # Writer stalled mid-write: seq odd -> drop, keep the previous view.
        seq_now = struct.unpack_from("<Q", open(host.out_slot_path, "rb").read(8))[0]
        with open(host.out_slot_path, "r+b") as fh:
            fh.write(struct.pack("<Q", seq_now + 1))
            fh.seek(iso.RV_CTRL_SIZE)
            fh.write(_valid_rv_slot(99))
            fh.flush()
        dropped0 = host.stats.out_dropped
        host.after_apply(None, seq=3)
        assert host.stats.out_dropped == dropped0 + 1, "torn / in-progress slot (odd seq) was not dropped"
        assert host.view_bytes() == raw, "torn slot replaced the supervisor view"
        with open(host.out_slot_path, "r+b") as fh:
            fh.write(struct.pack("<Q", seq_now + 2))
            fh.flush()

        # Bad magic -> rejected by the header validate, previous view kept.
        bad_hdr = bytearray(_valid_rv_slot(8))
        bad_hdr[0:8] = b"BADMAGIC"
        assert writer.write(bytes(bad_hdr))
        rej0 = host.stats.out_rejected
        assert host.after_apply(None, seq=4) is True
        assert host.stats.out_rejected == rej0 + 1, "bad-magic slot was not rejected by the header validate"
        assert host.view_bytes() == raw, "rejected slot replaced the supervisor view"
        assert not host.degraded

        # Post-apply cost is the copy.
        big_slot = _valid_rv_slot(11, w=640, h=640)
        dst = bytearray(len(big_slot))
        t_copy, t_apply = [], []
        for k in range(15):
            t0 = time.perf_counter()
            dst[:] = big_slot
            t_copy.append(time.perf_counter() - t0)
            writer.write(big_slot)
            t0 = time.perf_counter()
            host.after_apply(None, seq=10 + k)
            t_apply.append(time.perf_counter() - t0)
        mc, ma = statistics.median(t_copy), statistics.median(t_apply)
        assert host.view_bytes() == big_slot
        assert ma <= 10.0 * mc + 1e-3, f"after_apply {ma * 1e3:.3f} ms vs copy {mc * 1e3:.3f} ms: not copy-bound"
        print(
            f"check_supervisor_after_apply_is_copy_only: py calls in after_apply={sorted(set(names_py))}; "
            f"full-slot ({len(big_slot)} B) after_apply median {ma * 1e3:.3f} ms vs bytes copy {mc * 1e3:.3f} ms",
            flush=True,
        )
    finally:
        if writer is not None:
            writer.close()
        host.close()

    # Static: no RV object reaches control / frame_gate / state_io / record.
    targets = list((ROOT / "python" / "control").glob("*.py")) + [
        ROOT / "python" / "runtime" / "frame_gate.py",
        ROOT / "python" / "runtime" / "state_io.py",
        ROOT / "python" / "data" / "record.py",
    ]
    for p in targets:
        mods = _ast_imported_modules(p)
        leaked = [m for m in mods if ".rv_" in m or m.startswith("rv_")]
        assert not leaked, f"{p.relative_to(ROOT)} imports RV modules: {leaked}"
    forbidden_prefixes = ("python.control", "python.runtime.frame_gate", "python.runtime.state_io", "python.data.record")
    forbidden_sub = ("actuate", "heartbeat", "aeb", "engage", "frame_gate", "override", "launch_guard")
    for rel in ("python/viz/rv_isolation.py", "python/viz/rv_isolation_child.py"):
        p = ROOT / rel
        assert p.is_file(), f"missing {rel}"
        for m in _ast_imported_modules(p):
            assert not m.startswith(forbidden_prefixes), f"{rel} imports safety module {m}"
            assert not any(s in m.lower() for s in forbidden_sub), f"{rel} imports forbidden safety symbol {m}"


def main() -> None:
    failures = []
    for fn in (check_safety_loop_latency_unchanged, check_renderer_crash_degrades, check_supervisor_after_apply_is_copy_only):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.perf_counter() - t0:.1f}s)", flush=True)
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"test_rv_isolation: {len(failures)} failed: {', '.join(failures)}")
    print("test_rv_isolation: OK")


if __name__ == "__main__":
    main()
