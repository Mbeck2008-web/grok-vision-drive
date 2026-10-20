#!/usr/bin/env python3
"""The override consumes fresh echoes only (python/control/echo_link.py). Offline, no BeamNG.

The mod echoes the wheel at 10 Hz and the supervisor reads it at 15-20 Hz, with jitter and the
odd stall. These checks replay that timing deterministically (seeded):

* repeat/skip stress: force-feedback chatter flipping sign on every echo never disengages, and
  the old consumption (every read taken as a new sample) still reproduces the false takeover;
* takeover time: a real pull against the command is timed before (every read) and after (fresh
  echoes only), in milliseconds; after must be no slower than before + 100 ms;
* a frozen echo while engaged is a link fault (`echo_stale`), never "no override";
* an echo without `echo_seq` (older mod) is refused (`lua_echo_outdated`), not judged.

"before" here is the same detector fed `fresh=True, gap=False` on every read, which is exactly
what the supervisor did before the gate (the old `update` had no other path).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import ego_path, read_ego_feedback  # noqa: E402
from python.control.echo_link import (  # noqa: E402
    ECHO_STALE_MS,
    GAP_RUN_MAX,
    REASON_ECHO_OUTDATED,
    REASON_ECHO_STALE,
    EchoGate,
)
from python.control.override import OverrideConfig, OverrideDetector  # noqa: E402

ECHO_HZ = 10.0          # gvd_main EGO_POLL_S
TAKEOVER_BUDGET_MS = 100.0


def _echo_times(rng: random.Random, seconds: float, *, jitter: float = 0.2) -> list[float]:
    t = rng.random() / ECHO_HZ
    out = []
    while t < seconds + 1.0:
        out.append(t)
        t += (1.0 / ECHO_HZ) * (1.0 + rng.uniform(-jitter, jitter))
    return out


def drive(
    rng: random.Random,
    *,
    wheel,
    seconds: float,
    sup_hz: float,
    gated: bool,
    cmd_steer: float = 0.0,
    stall_p: float = 0.0,
    freeze_at: float | None = None,
    stamp_echo: bool = True,
) -> tuple[str, float, list[str]]:
    """Replay the supervisor loop. Returns (reason, t, statuses). reason 'none' = still engaged.

    `wheel(t, k)` is the steering_input of echo k sampled at t. `freeze_at` stops new echoes
    (the file stays fresh, echo_seq stops). `stamp_echo=False` drops echo_seq (older mod).
    """
    det = OverrideDetector(OverrideConfig())
    gate = EchoGate()
    times = _echo_times(rng, seconds)
    echoes = [(t, k, wheel(t, k)) for k, t in enumerate(times)]
    t, seq, ei = 0.0, 0, -1
    seen: list[str] = []
    while t < seconds:
        while ei + 1 < len(echoes) and echoes[ei + 1][0] <= t and (freeze_at is None or echoes[ei + 1][0] < freeze_at):
            ei += 1
        fb = None
        if ei >= 0:
            _, k, v = echoes[ei]
            fb = SimpleNamespace(fresh=True, echo_seq=k if stamp_echo else None, steering_input=v)
        if gated:
            r = gate.observe(engaged=True, ego_fb=fb, now=t)
            seen.append(r.status)
            if r.fault:
                return r.fault, t, seen
            fresh, gap = r.fresh, r.gap
        else:
            fresh, gap = True, False
        verdict = det.update(
            engaged=True,
            steering_input=None if fb is None else fb.steering_input,
            throttle_input=0.0,
            brake_input=0.0,
            applied_seq=None if fb is None else seq,
            now=t,
            fresh=fresh,
            gap=gap,
        )
        if verdict.active:
            return verdict.reason, t, seen
        seq += 1
        det.note_command(seq=seq, steer=cmd_steer, throttle=0.0, brake=0.0, now=t)
        dt = (1.0 / sup_hz) * (1.0 + rng.uniform(-0.3, 0.3))
        if rng.random() < stall_p:
            dt += rng.uniform(0.05, 0.25)
        t += dt
    return "none", t, seen


def _chatter(_t: float, k: int) -> float:
    return 0.9 if k % 2 else -0.9


def check_gate_states() -> None:
    g = EchoGate()
    fb = lambda k, fresh=True: SimpleNamespace(fresh=fresh, echo_seq=k)  # noqa: E731
    assert g.observe(engaged=True, ego_fb=None, now=0.0).status == "waiting"
    r = g.observe(engaged=True, ego_fb=fb(5), now=0.05)
    assert r.fresh and not r.gap and r.seq == 5, r
    r = g.observe(engaged=True, ego_fb=fb(5), now=0.10)
    assert r.status == "repeat" and not r.fresh and r.fault is None, r
    r = g.observe(engaged=True, ego_fb=fb(6), now=0.15)
    assert r.fresh and not r.gap, r
    r = g.observe(engaged=True, ego_fb=fb(8), now=0.30)
    assert r.fresh and r.gap and r.skipped == 1, r
    # A reloaded mod restarts its counter: a step back is new, and a gap (not adjacent).
    r = g.observe(engaged=True, ego_fb=fb(1), now=0.35)
    assert r.fresh and r.gap, r
    # The link was live; then nothing new for ~300 ms -> stale, by name.
    r = g.observe(engaged=True, ego_fb=fb(1), now=0.35 + ECHO_STALE_MS / 1000.0 - 0.01)
    assert r.status == "repeat", r
    r = g.observe(engaged=True, ego_fb=fb(1), now=0.35 + ECHO_STALE_MS / 1000.0 + 0.001)
    assert r.status == "stale" and r.fault == REASON_ECHO_STALE, r
    # The file going away (or past EGO_FRESH_S) after the link was up is stale too, not "waiting".
    g2 = EchoGate()
    g2.observe(engaged=True, ego_fb=fb(1), now=0.0)
    assert g2.observe(engaged=True, ego_fb=None, now=0.31).fault == REASON_ECHO_STALE
    g3 = EchoGate()
    g3.observe(engaged=True, ego_fb=fb(1), now=0.0)
    assert g3.observe(engaged=True, ego_fb=fb(2, fresh=False), now=0.31).fault == REASON_ECHO_STALE
    # Every fresh echo arriving with a gap leaves the filter blind: a link fault as well.
    g4 = EchoGate()
    g4.observe(engaged=True, ego_fb=fb(0), now=0.0)
    for i in range(1, GAP_RUN_MAX):
        assert g4.observe(engaged=True, ego_fb=fb(2 * i), now=0.1 * i).fault is None
    assert g4.observe(engaged=True, ego_fb=fb(2 * GAP_RUN_MAX), now=0.1 * GAP_RUN_MAX).fault == REASON_ECHO_STALE
    # Disengaged forgets everything; re-engage starts at waiting again.
    assert g.observe(engaged=False, ego_fb=fb(9), now=1.0).status == "off"
    assert g.observe(engaged=True, ego_fb=None, now=1.1).status == "waiting"
    # Older mod: a live echo without echo_seq is refused, never judged.
    r = EchoGate().observe(engaged=True, ego_fb=SimpleNamespace(fresh=True, echo_seq=None), now=0.0)
    assert r.status == "outdated" and r.fault == REASON_ECHO_OUTDATED, r


def check_ego_parse() -> None:
    p = ego_path()
    keep = p.read_bytes() if p.is_file() else None
    try:
        base = {"speed_mps": 1.0, "steering_input": 0.1, "applied_seq": 3, "applying": True, "mtime": 0}
        for raw, want in ((7, 7), (7.0, 7), (None, None), (True, None), ("7", None), (7.5, None)):
            data = dict(base)
            if raw is not None:
                data["echo_seq"] = raw
            p.write_text(json.dumps(data), encoding="utf-8")
            fb = read_ego_feedback()
            assert fb is not None and fb.echo_seq == want, (raw, fb and fb.echo_seq)
    finally:
        if keep is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(keep)


def check_repeat_skip_stress() -> None:
    """Chatter flipping per echo, read at 15/20 Hz with jitter, repeats and skips (stalls)."""
    runs = 400
    print("  repeat/skip stress (3 s runs, chatter +/-0.9 per echo):")
    for hz in (15.0, 20.0):
        for stall in (0.0, 0.02):
            rng_b = random.Random(1000 + int(hz) + int(stall * 100))
            rng_a = random.Random(1000 + int(hz) + int(stall * 100))
            before = after = stale = 0
            for _ in range(runs):
                rb, _, _ = drive(rng_b, wheel=_chatter, seconds=3.0, sup_hz=hz, gated=False, stall_p=stall)
                ra, _, seen = drive(rng_a, wheel=_chatter, seconds=3.0, sup_hz=hz, gated=True, stall_p=stall)
                before += rb == "player_steer"
                after += ra == "player_steer"
                stale += ra == REASON_ECHO_STALE
                assert "repeat" in seen, "the replay never re-read an echo; the stress is not stressing"
            print(f"    {hz:.0f} Hz stall_p={stall:.2f}: false player_steer before {before}/{runs}, after {after}/{runs} (echo_stale {stale})")
            assert after == 0, f"chatter read as player_steer {after}/{runs} at {hz} Hz"
            assert stale <= runs // 50, f"echo_stale on a live link {stale}/{runs}"
            assert before > 0, "the old every-read consumption no longer reproduces the false takeover"


def _takeover_ms(gated: bool, hz: float, seed: int, runs: int = 300) -> list[float]:
    """ms from the wheel pull (0.25 against a 0.2 command) to player_steer."""
    rng = random.Random(seed)
    t_pull, cmd = 1.0, 0.2
    out = []
    for _ in range(runs):
        reason, t, _ = drive(
            rng,
            wheel=lambda t, _k: cmd if t < t_pull else cmd - 0.25,
            seconds=4.0,
            sup_hz=hz,
            gated=gated,
            cmd_steer=cmd,
        )
        assert reason == "player_steer", (gated, hz, reason)
        out.append((t - t_pull) * 1000.0)
    return out


def check_takeover_time() -> None:
    print("  takeover (0.25 pull against a 0.2 command), ms from pull to player_steer:")
    for hz in (15.0, 20.0):
        b = _takeover_ms(False, hz, seed=77)
        a = _takeover_ms(True, hz, seed=77)
        bm, am = statistics.mean(b), statistics.mean(a)
        b95, a95 = sorted(b)[int(0.95 * len(b))], sorted(a)[int(0.95 * len(a))]
        print(f"    {hz:.0f} Hz loop / 10 Hz echo: before mean {bm:.0f} p95 {b95:.0f} | after mean {am:.0f} p95 {a95:.0f}")
        assert am <= bm + TAKEOVER_BUDGET_MS, (bm, am)
        assert a95 <= b95 + TAKEOVER_BUDGET_MS, (b95, a95)


def check_stale_fails_safe() -> None:
    """Echo frozen while engaged (and while the wheel chatters): echo_stale, within ~300 ms."""
    for hz in (15.0, 20.0):
        rng = random.Random(5)
        for _ in range(50):
            reason, t, seen = drive(rng, wheel=_chatter, seconds=4.0, sup_hz=hz, gated=True, freeze_at=1.5)
            assert reason == REASON_ECHO_STALE, (reason, t)
            late = (t - 1.5) * 1000.0
            # The last echo before the freeze can be up to one echo period old.
            assert late <= ECHO_STALE_MS + 1000.0 / ECHO_HZ * 1.2 + 1000.0 / hz * 1.3 + 1.0, late
            assert "none" not in seen


def check_outdated_refused() -> None:
    rng = random.Random(3)
    reason, t, seen = drive(rng, wheel=lambda _t, _k: 0.0, seconds=2.0, sup_hz=20.0, gated=True, stamp_echo=False)
    assert reason == REASON_ECHO_OUTDATED, reason
    assert "fresh" not in seen, seen


def main() -> None:
    check_gate_states()
    check_ego_parse()
    check_repeat_skip_stress()
    check_takeover_time()
    check_stale_fails_safe()
    check_outdated_refused()
    print("test_ffb_echo_fresh: OK")


if __name__ == "__main__":
    main()
