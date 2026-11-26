#!/usr/bin/env python3
"""Reality View step 3: isolation harness (stub children, CPU, offline).

docs/specs/reality_view_3dgs.md, Feasibility plan step 3. A stub control
loop (heartbeat stamp, AEB, engage, ``NullActuator.apply``, then the RV
handoff) runs here, in the supervisor's place. The generator (unproject +
splat packing) and the renderer run in **child processes**
(``scripts/rv_stub_child.py``) behind ``python/runtime/rv_link.py``.

* ``check_safety_loop_latency_unchanged``: a slow child generator and a slow
  child renderer move neither loop Hz nor the stub ``infer_ms`` p95 by 5% or
  more (A/B interleaved, median of rounds).
* ``check_renderer_crash_degrades``: SIGKILL the renderer, SIGABRT the
  generator; the loop finishes every tick, heartbeat stamps stay fresh,
  AEB / engage match the no-RV run tick for tick, and the main view is the
  placeholder, bit-identical.
* ``check_supervisor_after_apply_is_copy_only``: a profiler traces every
  call between ``actuator.apply`` returning and the tick end. Only
  ``rv_link`` code runs there, never ``unproject`` / ``pack_splats``, and
  the post-apply cost stays near a plain memcpy of the same bytes.
* ``check_gates_catch_inline_generator``: the same gates fail when the
  generator runs on the loop thread, so the three checks above can fail.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import signal  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.control.actuate import (  # noqa: E402
    HEARTBEAT_STALE_S,
    DriveCommand,
    NullActuator,
    heartbeat_fresh,
    may_drive,
)
from python.runtime import rv_link  # noqa: E402
from python.runtime.rv_link import RvLink  # noqa: E402
from python.viz.view_select import render_reality_placeholder  # noqa: E402

CHILD = str(ROOT / "scripts" / "rv_stub_child.py")
SLOT_BYTES = 160 * 120 * 16
PERIOD_S = 0.025
INFER_MS = 5.0
GATE = 0.05
W, H = 320, 180

_rng = np.random.default_rng(7)
_A = _rng.standard_normal((96, 96)).astype(np.float32)


def _calibrate(ms: float) -> int:
    x, n, t0 = _A, 0, time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < ms * 20:
        x = np.tanh(x @ _A * 0.01)
        n += 1
    return max(n // 20, 1)


_INFER_ITERS: int | None = None


def stub_infer() -> float:
    """Fixed CPU work (a fixed iteration count, ~INFER_MS) standing in for vision infer."""
    global _INFER_ITERS
    if _INFER_ITERS is None:
        _INFER_ITERS = _calibrate(INFER_MS)
    x = _A
    for _ in range(_INFER_ITERS):
        x = np.tanh(x @ _A * 0.01)
    return float(x[0, 0])


def aeb_stub(tick: int) -> bool:
    """Deterministic AEB: obstacle closes at 0.4 m/tick, 12 m/s ego, TTC < 1.5 s."""
    dist = max(60.0 - 0.4 * tick, 0.1)
    return dist / 12.0 < 1.5


def engage_stub(tick: int) -> bool:
    return 5 <= tick < 170


def run_loop(link: RvLink | None, ticks: int, *, inline_gen=None, on_tick=None, trace=None) -> dict:
    act = NullActuator()
    hb: list[float] = []
    safety: list[tuple] = []
    period: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    views: list[str] = []
    t_next = time.perf_counter()
    t_prev = None
    for tick in range(ticks):
        now = time.perf_counter()
        if t_prev is not None:
            period.append(now - t_prev)
        t_prev = now
        hb.append(time.time())  # heartbeat stamp
        t0 = time.perf_counter()
        stub_infer()
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        engaged = engage_stub(tick)
        ok, reason = may_drive(engaged=engaged, heartbeat_ok=heartbeat_fresh(hb[-1]), path_debug_preview=False, allow_preview_drive=False)
        aeb = aeb_stub(tick)
        cmd = DriveCommand(throttle=0.0 if aeb else 0.3, brake=1.0 if aeb else 0.0, seq=tick, reason="aeb" if aeb else reason)
        applied = act.apply(cmd)
        safety.append((tick, engaged, ok, reason, aeb, applied.brake, applied.reason))
        # --- post-apply: copy / handoff only ---
        tp = time.perf_counter()
        if trace is not None:
            trace.start()
        if link is not None:
            link.handoff()
        if inline_gen is not None:
            inline_gen()
        if trace is not None:
            trace.stop()
        post_ms.append((time.perf_counter() - tp) * 1000.0)
        if link is not None:
            views.append("placeholder" if link.main_view({}, W, H) is not None else "rv")
        if on_tick is not None:
            on_tick(tick)
        t_next += PERIOD_S
        delay = t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            t_next = time.perf_counter()
    return {"hb": hb, "safety": safety, "period": period, "infer_ms": infer_ms, "post_ms": post_ms, "views": views}


def p95(xs: list[float]) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), 95))


def slow_link() -> RvLink:
    return RvLink(
        gen_argv=[CHILD, "gen", "--slow-ms", "40", "--burn-ms", "15"],
        viz_argv=[CHILD, "viz", "--slow-ms", "90", "--burn-ms", "30"],
        slot_bytes=SLOT_BYTES,
        stale_s=0.5,
    )


def wait_first_frame(link: RvLink, timeout_s: float = 10.0) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        if link.handoff():
            return
        time.sleep(0.01)
    raise AssertionError("generator child never published a frame")


def _inline_generator():
    import rv_stub_child as c

    depth = np.linspace(2.0, 40.0, c.H * c.W, dtype=np.float32).reshape(c.H, c.W)
    pose = np.eye(4, dtype=np.float32)

    def gen() -> None:
        c.pack_splats(c.unproject(depth, 140.0, 140.0, c.W / 2, c.H / 2, pose), 1)
        time.sleep(0.04)  # same slowness as the child generator

    return gen


def latency_rounds(rounds: int, ticks: int, inline: bool) -> tuple[float, float, float, float]:
    """Interleave off/on rounds; return median (hz_off, hz_on, p95_off, p95_on)."""
    hz_off, hz_on, p_off, p_on = [], [], [], []
    for _ in range(rounds):
        off = run_loop(None, ticks)
        if inline:
            on = run_loop(None, ticks, inline_gen=_inline_generator())
        else:
            link = slow_link()
            link.start()
            try:
                wait_first_frame(link)
                on = run_loop(link, ticks)
                assert link.alive_gen() and link.alive_viz(), "a child died during the latency run"
                assert link.frames_handed > 0, "no frame was handed off: the RV path did not run"
            finally:
                link.close()
        hz_off.append(1.0 / statistics.mean(off["period"]))
        hz_on.append(1.0 / statistics.mean(on["period"]))
        p_off.append(p95(off["infer_ms"]))
        p_on.append(p95(on["infer_ms"]))
    return statistics.median(hz_off), statistics.median(hz_on), statistics.median(p_off), statistics.median(p_on)


def latency_ok(hz_off: float, hz_on: float, p_off: float, p_on: float) -> bool:
    return abs(hz_on - hz_off) / hz_off < GATE and abs(p_on - p_off) / p_off < GATE


def check_safety_loop_latency_unchanged() -> None:
    r = latency_rounds(rounds=3, ticks=120, inline=False)
    print(f"  latency off/on: hz {r[0]:.2f}/{r[1]:.2f}  infer_ms p95 {r[2]:.3f}/{r[3]:.3f}", flush=True)
    assert latency_ok(*r), f"RV children moved the safety loop by >= 5%: {r}"


def check_renderer_crash_degrades() -> None:
    ticks = 180
    ref = run_loop(None, ticks)
    link = slow_link()
    link.start()
    killed: dict[str, int] = {}

    def on_tick(tick: int) -> None:
        if tick == 50:
            os.kill(link.viz_pid, signal.SIGKILL)
            killed["viz"] = tick
        if tick == 100:
            os.kill(link.gen_pid, signal.SIGABRT)
            killed["gen"] = tick

    try:
        wait_first_frame(link)
        got = run_loop(link, ticks, on_tick=on_tick)
        assert not link.alive_viz() and not link.alive_gen(), "children were not killed"
        state = {"engaged": False}
        degraded = link.main_view(state, W, H)
        assert degraded is not None, "main view did not degrade after the children died"
        assert np.array_equal(degraded, render_reality_placeholder(state, W, H)), "degraded view is not the placeholder"
        handed_after_kill = link.handoff()
    finally:
        link.close()
    assert killed == {"viz": 50, "gen": 100}, killed
    assert len(got["hb"]) == ticks, f"loop stopped at tick {len(got['hb'])}"
    gaps = np.diff(np.asarray(got["hb"]))
    assert float(gaps.max()) < HEARTBEAT_STALE_S / 5, f"heartbeat gap {gaps.max():.3f}s"
    assert got["safety"] == ref["safety"], "AEB / engage / command changed vs the no-RV run"
    assert got["views"][10] == "rv", got["views"][:12]
    assert got["views"][55:].count("rv") == 0, "view still claims rv after the renderer died"
    assert handed_after_kill is False


def make_trace():
    calls: list[tuple[str, str]] = []

    class Trace:
        def _prof(self, frame, event, arg):
            if event == "call":
                calls.append((frame.f_code.co_filename, frame.f_code.co_name))
            elif event == "c_call":
                calls.append(("<c>", getattr(arg, "__qualname__", repr(arg))))

        def start(self) -> None:
            sys.setprofile(self._prof)

        def stop(self) -> None:
            sys.setprofile(None)

    return Trace(), calls


def copy_only_report(calls: list[tuple[str, str]]) -> tuple[set, set]:
    link_file = str(Path(rv_link.__file__).resolve())
    py = {(f, n) for f, n in calls if f != "<c>"}
    foreign = {(f, n) for f, n in py if str(Path(f).resolve()) != link_file and n != "stop"}
    gen_names = {n for _, n in calls if n in ("unproject", "pack_splats") or n.endswith(".unproject")}
    return foreign, gen_names


def memcpy_p95_ms(nbytes: int) -> float:
    src = bytearray(os.urandom(nbytes))
    dst = bytearray(nbytes)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        dst[:] = src
        ts.append((time.perf_counter() - t0) * 1000.0)
    return p95(ts)


def check_supervisor_after_apply_is_copy_only() -> None:
    link = slow_link()
    link.start()
    trace, calls = make_trace()
    try:
        wait_first_frame(link)
        base = link.frames_handed
        got = run_loop(link, 80, trace=trace)
        handed = link.frames_handed - base
        link.handoff()
        fresh = time.monotonic()
        assert link.main_view({}, W, H, now=fresh) is None, "healthy link degraded"
        assert link.main_view({}, W, H, now=fresh + 1.0) is not None, "a frozen generator (no new frame for 1 s) did not degrade"
    finally:
        link.close()
    foreign, gen = copy_only_report(calls)
    assert handed >= 3, f"only {handed} frames handed off; trace proves nothing"
    assert not gen, f"generation ran on the supervisor: {gen}"
    assert not foreign, f"non-rv_link code after apply: {sorted(foreign)[:8]}"
    untraced = run_loop_post_ms()
    mc = memcpy_p95_ms(SLOT_BYTES)
    print(f"  post-apply p95 {untraced:.3f} ms vs memcpy p95 {mc:.3f} ms ({SLOT_BYTES} B)", flush=True)
    assert untraced <= 4.0 * mc + 0.5, f"post-apply {untraced:.3f} ms is not copy-sized (memcpy {mc:.3f} ms)"
    assert got["safety"]


def run_loop_post_ms() -> float:
    link = slow_link()
    link.start()
    try:
        wait_first_frame(link)
        return p95(run_loop(link, 80)["post_ms"])
    finally:
        link.close()


def check_gates_catch_inline_generator() -> None:
    r = latency_rounds(rounds=1, ticks=60, inline=True)
    assert not latency_ok(*r), f"latency gate passed with the generator on the loop thread: {r}"
    trace, calls = make_trace()
    run_loop(None, 5, inline_gen=_inline_generator(), trace=trace)
    foreign, gen = copy_only_report(calls)
    assert gen and foreign, "copy-only trace missed an inline generator"


def check_rv_link_is_copy_only_source() -> None:
    src = Path(rv_link.__file__).read_text(encoding="utf-8")
    for bad in ("threading", "rv_stub_child", "unproject", "pack_splats", "import cv2", "torch"):
        assert bad not in src.split('"""', 2)[-1], f"rv_link.py references {bad!r}"


def check_torn_or_stale_frame_dropped() -> None:
    """Handoff drops a frame mid-write and an already-handed frame; never waits."""
    link = RvLink(gen_argv=[], viz_argv=[], slot_bytes=64)
    link.gen_slot, link.viz_slot = rv_link.RvSlot.create(64), rv_link.RvSlot.create(64)
    try:
        assert link.handoff() is False  # nothing published yet
        link.gen_slot.publish(b"a" * 64)
        assert link.handoff() is True
        assert link.handoff() is False  # same frame again
        i, v = link.gen_slot.begin_write()
        v.release()
        link.gen_slot.commit(i, 64, 2)
        _, v = link.gen_slot.begin_write()  # writer now mid-write on the other buffer...
        v.release()
        latest = rv_link._HDR.unpack_from(link.gen_slot._mv, 0)[3]
        seq = rv_link._BUF.unpack_from(link.gen_slot._mv, rv_link._BUF_AT[latest])[0]
        assert seq % 2 == 0  # ...the newest complete frame stays readable
        assert link.handoff() is True
        out = bytearray(64)
        assert link.viz_slot.read_into(out, 0) == (2, 64)
        # A torn newest buffer (odd seq) is dropped, not waited on.
        mv = link.gen_slot._mv
        s0, f0, n0 = rv_link._BUF.unpack_from(mv, rv_link._BUF_AT[latest])
        rv_link._BUF.pack_into(mv, rv_link._BUF_AT[latest], s0 | 1, f0 + 5, n0)
        t0 = time.perf_counter()
        assert link.handoff() is False
        assert time.perf_counter() - t0 < 0.01
        del mv
    finally:
        link.gen_slot.close()
        link.viz_slot.close()


def check_writer_lap_during_copy_is_dropped() -> None:
    """If the writer laps the buffer during the memcpy, the copy is dropped, not published."""
    src, dst = rv_link.RvSlot.create(64), rv_link.RvSlot.create(64)
    real = rv_link._BUF
    calls = {"n": 0}

    class LapOnRecheck:
        size = real.size

        def pack_into(self, *a):
            return real.pack_into(*a)

        def unpack_from(self, buf, off=0):
            calls["n"] += 1
            seq, frame, n = real.unpack_from(buf, off)
            if calls["n"] == 3:  # src newest, dst begin_write, then the src re-check
                return seq + 2, frame, n
            return seq, frame, n

    try:
        src.publish(b"x" * 64)
        rv_link._BUF = LapOnRecheck()
        try:
            assert src.copy_newest_to(dst, 0) is None, "torn copy was published"
        finally:
            rv_link._BUF = real
        assert calls["n"] >= 3
        assert dst.read_into(bytearray(64), 0) is None, "dst exposed a torn frame"
        assert src.copy_newest_to(dst, 0) == 1  # clean retry next tick works
    finally:
        src.close()
        dst.close()


def check_children_run_below_supervisor_priority() -> None:
    link = slow_link()
    link.start()
    try:
        if sys.platform != "win32":
            me = os.getpriority(os.PRIO_PROCESS, 0)
            for pid in (link.gen_pid, link.viz_pid):
                assert os.getpriority(os.PRIO_PROCESS, pid) >= me + rv_link.CHILD_NICE, pid
    finally:
        link.close()


def main() -> None:
    check_rv_link_is_copy_only_source()
    check_writer_lap_during_copy_is_dropped()
    check_children_run_below_supervisor_priority()
    check_torn_or_stale_frame_dropped()
    check_gates_catch_inline_generator()
    check_supervisor_after_apply_is_copy_only()
    check_renderer_crash_degrades()
    check_safety_loop_latency_unchanged()
    print("test_rv_isolation: OK")


def test_rv_isolation() -> None:
    main()


if __name__ == "__main__":
    main()
