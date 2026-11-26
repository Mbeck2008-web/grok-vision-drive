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
MEASURE_TICKS = 150
ROUNDS = 2  # conditions interleaved per round (OFF, gen, renderer, OFF, gen, renderer)
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

    def __init__(self, host=None, *, hz: float = LOOP_HZ, frame: bytes | None = None) -> None:
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
        self._next = None

    def tick(self, i: int) -> None:
        from python.runtime.shadow import shadow_tick

        if self._next is None:
            self._next = time.perf_counter()
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
        t_post = time.perf_counter()
        if self.host is not None:
            ret = self.host.after_apply(self.frame, seq=i)
            assert isinstance(ret, bool), f"after_apply must return a plain bool, got {type(ret)!r}"
        self.post_apply_ms.append((time.perf_counter() - t_post) * 1000.0)
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


def check_safety_loop_latency_unchanged() -> None:
    """Slow child generator / slow (hung) child renderer do not move the stub loop by >5%.

    Method: paced 50 Hz stub loop (heartbeat stamp, fixed-work stand-in infer,
    real shadow_tick + NullActuator, RV after_apply). Conditions OFF (no RV),
    ON+slow_generator child (80 ms "packing" per output, stalls 30 ms mid-write
    of the shared output slot), ON+slow_renderer child (holds the input slot
    reader-busy flag for 250 ms per read, then hangs with it set). Each run has
    WARMUP_TICKS discarded; conditions are interleaved over ROUNDS and pooled,
    so drift on a noisy box hits all conditions alike. Gates vs the pooled OFF
    baseline: loop-period p50 and p95 within 5%, stand-in infer_ms p50 and p95
    within 5%. Structural (wall time can hide a short block): every Popen.wait /
    communicate on the child is trapped during the loop; the backpressure drop
    path must be exercised (renderer case drops handoffs, generator case drops
    torn/busy output reads); after_apply p95 must stay under 5% of the period.
    """
    _calibrate_infer()
    frame = _frame_bytes()
    pooled: dict[str, dict[str, list[float]]] = {}
    host_stats: dict[str, dict] = {}

    # Baseline first so the fail-first log shows the OFF numbers even without a harness.
    def _run(cond: str) -> None:
        host = None
        if cond != "off":
            iso = _harness()
            mode = "slow_generator" if cond == "gen" else "slow_renderer"
            host = iso.RvHost(module=iso.RV_ISO_CHILD_MODULE, child_args=("--mode", mode), print_notes=False)
            pid = host.launch()
            assert pid is not None, f"RvHost.launch failed: {host.note}"
            assert host.wait_ready(5.0), f"{mode} child never armed (note={host.note!r})"
        loop = _StubSafetyLoop(host, frame=frame if host is not None else None)
        try:
            if host is not None:
                with _NoBlockGuard(host.client._proc) as guard:
                    for i in range(WARMUP_TICKS + MEASURE_TICKS):
                        loop.tick(i)
                assert not guard.calls, f"{cond}: blocking calls on the loop: {guard.calls}"
                assert not host.degraded, f"{cond}: host degraded during a slow-but-alive child run: {host.note}"
                s = host_stats.setdefault(cond, {"handoff_dropped": 0, "out_dropped": 0, "out_applied": 0, "handoff_ok": 0})
                for k in s:
                    s[k] += int(getattr(host.stats, k))
            else:
                for i in range(WARMUP_TICKS + MEASURE_TICKS):
                    loop.tick(i)
        finally:
            if host is not None:
                host.close()
        d = pooled.setdefault(cond, {"period": [], "infer": [], "post": []})
        d["period"] += loop.periods_ms(skip=WARMUP_TICKS)
        d["infer"] += loop.infer_ms[WARMUP_TICKS:]
        d["post"] += loop.post_apply_ms[WARMUP_TICKS:]

    for _ in range(ROUNDS):
        for cond in ("off", "gen", "renderer"):
            _run(cond)

    base = pooled["off"]
    lines = []
    for cond in ("off", "gen", "renderer"):
        d = pooled[cond]
        lines.append(
            f"  {cond:<9} period p50={statistics.median(d['period']):.3f} p95={_pct(d['period'], 0.95):.3f} ms | "
            f"infer p50={statistics.median(d['infer']):.3f} p95={_pct(d['infer'], 0.95):.3f} ms | "
            f"after_apply p50={statistics.median(d['post']) * 1000:.1f} p95={_pct(d['post'], 0.95) * 1000:.1f} "
            f"max={max(d['post']) * 1000:.1f} us | n={len(d['period'])}"
        )
    print("check_safety_loop_latency_unchanged:\n" + "\n".join(lines), flush=True)
    print(f"  host stats: {host_stats}", flush=True)

    period_s = 1.0 / LOOP_HZ
    for cond in ("gen", "renderer"):
        d = pooled[cond]
        for metric, key in (("loop period", "period"), ("infer_ms", "infer")):
            for qname, fn in (("p50", statistics.median), ("p95", lambda xs: _pct(xs, 0.95))):
                b, v = fn(base[key]), fn(d[key])
                rel = (v - b) / b
                assert abs(rel) < GATE_REL, (
                    f"{cond}: {metric} {qname} moved {rel * 100:+.2f}% (base {b:.3f} ms, RV {v:.3f} ms); gate 5%"
                )
        p95_post = _pct(d["post"], 0.95) / 1000.0
        assert p95_post < GATE_REL * period_s, (
            f"{cond}: after_apply p95 {p95_post * 1e6:.0f} us >= 5% of the {period_s * 1000:.0f} ms period"
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
