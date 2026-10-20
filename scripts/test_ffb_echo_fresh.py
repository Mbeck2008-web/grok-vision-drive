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
import re
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
    MIN_MOD_VERSION,
    REASON_ECHO_OUTDATED,
    REASON_ECHO_STALE,
    EchoGate,
    outdated_message,
    parse_version,
)
from python.control.override import GAP_CREDIT_MAX_S, OverrideConfig, OverrideDetector  # noqa: E402

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
    edit_echoes=None,
    cmd_stop: tuple[float, float] | None = None,
    seq_meaning: str = "lua",
) -> tuple[str, float, list[str]]:
    """Replay the supervisor loop. Returns (reason, t, statuses). reason 'none' = still engaged.

    `wheel(t, k)` is the steering_input of echo k sampled at t. `freeze_at` stops new echoes
    (the file stays fresh, echo_seq stops). `stamp_echo=False` drops echo_seq (older mod).

    Injections: `edit_echoes(echoes)` rewrites the mod's echo list of (t, echo_seq, wheel):
    drop one and the supervisor never sees it, so the next arrives with a skipped echo_seq;
    renumber and echo_seq repeats values it already used (a mod reload). `cmd_stop=(t0, t1)`: commands stop reaching the
    mod; its `applied_seq` freezes while its echoes keep coming. `seq_meaning` is what the fake
    mod stamps as echo_seq: "lua" (gvd_main: its own counter, +1 per echo) or "cmd" (the wrong
    meaning: the last command seq it received), kept only to prove the difference.
    """
    det = OverrideDetector(OverrideConfig())
    gate = EchoGate()
    times = _echo_times(rng, seconds)
    echoes = [(t, k, wheel(t, k)) for k, t in enumerate(times)]
    if edit_echoes is not None:
        echoes = edit_echoes(echoes)
    stamped: dict[int, int] = {}  # echo index -> echo_seq the fake mod wrote
    t, seq, ei = 0.0, 0, -1
    lua_seq = 0  # the newest command the mod received (its applied_seq)
    seen: list[str] = []
    while t < seconds:
        if cmd_stop is None or not (cmd_stop[0] <= t < cmd_stop[1]):
            lua_seq = seq
        while ei + 1 < len(echoes) and echoes[ei + 1][0] <= t and (freeze_at is None or echoes[ei + 1][0] < freeze_at):
            ei += 1
            # Stamped when the mod writes it. The wheel value never depends on this.
            stamped[ei] = echoes[ei][1] if seq_meaning == "lua" else lua_seq
        fb = None
        if ei >= 0:
            _, _k, v = echoes[ei]
            fb = SimpleNamespace(
                fresh=True,
                echo_seq=stamped[ei] if stamp_echo else None,
                mod_version=MIN_MOD_VERSION if stamp_echo else None,
                steering_input=v,
            )
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
            applied_seq=None if fb is None else lua_seq,
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
    fb = lambda k, fresh=True: SimpleNamespace(fresh=fresh, echo_seq=k, mod_version=MIN_MOD_VERSION)  # noqa: E731
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


def check_mod_version_gate() -> None:
    """Below MIN_MOD_VERSION is refused, and the refusal names the minimum version required."""
    assert MIN_MOD_VERSION == "1.7.13", MIN_MOD_VERSION
    assert parse_version("1.7.13") == (1, 7, 13) and parse_version("1.7.13-alpha-abc123") == (1, 7, 13)
    assert parse_version(None) is None and parse_version("dev") is None
    main_lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    m = re.search(r"M\.MOD_VERSION\s*=\s*'([^']+)'", main_lua)
    assert m, "gvd_main has no M.MOD_VERSION"
    assert parse_version(m.group(1)) >= parse_version(MIN_MOD_VERSION), m.group(1)
    assert '"mod_version":"%s"' in main_lua, "gvd_ego.json does not carry mod_version"
    cases = (
        (dict(echo_seq=1, mod_version=MIN_MOD_VERSION), "fresh"),
        (dict(echo_seq=1, mod_version="1.8.0"), "fresh"),
        (dict(echo_seq=1, mod_version=MIN_MOD_VERSION + "-alpha-abc123"), "fresh"),
        (dict(echo_seq=1, mod_version="1.7.12"), "outdated"),
        (dict(echo_seq=1, mod_version=None), "outdated"),
        (dict(echo_seq=None, mod_version=MIN_MOD_VERSION), "outdated"),
    )
    for kw, want in cases:
        r = EchoGate().observe(engaged=True, ego_fb=SimpleNamespace(fresh=True, **kw), now=0.0)
        assert r.status == want, (kw, r)
    old = outdated_message(SimpleNamespace(echo_seq=1, mod_version="1.7.12"))
    assert old.startswith(REASON_ECHO_OUTDATED + ":") and "mod 1.7.12" in old, old
    assert f"mod {MIN_MOD_VERSION} or newer is required" in old, old
    none = outdated_message(SimpleNamespace(echo_seq=None, mod_version=None))
    assert "without mod_version and no echo_seq" in none and MIN_MOD_VERSION in none, none


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


def check_chatter_rule_exact() -> None:
    """The chatter rule, pinned: only the first fresh echo after a skipped or repeated echo_seq
    is discounted. Spike after spike on contiguous echoes still charges the dwell."""
    cfg = OverrideConfig()
    det = OverrideDetector(cfg)
    t = 0.0

    def step(steer: float, *, gap: bool = False, dt: float = 0.1):
        nonlocal t
        t += dt
        v = det.update(engaged=True, steering_input=steer, throttle_input=0.0, brake_input=0.0,
                       applied_seq=None, now=t, gap=gap)
        det.note_command(seq=int(t * 100), steer=0.0, throttle=0.0, brake=0.0, now=t)
        return v

    det.note_command(seq=0, steer=0.0, throttle=0.0, brake=0.0, now=0.0)
    for _ in range(4):
        step(0.0)  # warm-up
    step(0.07)
    v = step(0.10)  # filter climbs past steer_enter on contiguous, non-spike echoes
    assert not v.spike and v.steer_filt >= cfg.steer_enter * 0.5, v
    v = step(0.13, dt=0.02)
    held0 = v.steer_held_ms
    # Two same-sign spikes in a row on contiguous echoes: both are spikes, the filter holds,
    # and the dwell charges on each (this was suppressed before as "spike after spike").
    v1 = step(0.13 + cfg.steer_spike + 0.15, dt=0.02)
    v2 = step(0.13 + 2 * (cfg.steer_spike + 0.15), dt=0.02)
    assert v1.spike and v2.spike, (v1, v2)
    if abs(v2.steer_eff) >= cfg.steer_enter:
        assert v1.steer_held_ms > held0 and v2.steer_held_ms > v1.steer_held_ms, (held0, v1, v2)
    # The sample right after a skipped / repeated echo_seq: no spike test, no filter, no dwell.
    v3 = step(-0.9, gap=True, dt=0.02)
    assert v3.gap_hold and not v3.spike, v3
    assert v3.steer_filt == v2.steer_filt and v3.steer_held_ms == v2.steer_held_ms, (v2, v3)
    # It re-seeded the spike test: the next contiguous echo is judged against it.
    v4 = step(-0.9, dt=0.02)
    assert not v4.spike and not v4.gap_hold, v4


def _scripted(echoes: list[tuple[float, int, float]], *, cmd: float, t_end: float, read_hz: float = 20.0):
    """Gate + detector over a fixed echo timeline [(t, echo_seq, steering_input)], read at
    `read_hz`. Returns every non-repeat (t, EchoRead, verdict)."""
    det = OverrideDetector(OverrideConfig())
    gate = EchoGate()
    out = []
    t, i, n = 0.0, -1, 0
    while t < t_end:
        while i + 1 < len(echoes) and echoes[i + 1][0] <= t:
            i += 1
        fb = None
        if i >= 0:
            fb = SimpleNamespace(fresh=True, echo_seq=echoes[i][1], mod_version=MIN_MOD_VERSION, steering_input=echoes[i][2])
        r = gate.observe(engaged=True, ego_fb=fb, now=t)
        assert r.fault is None, (t, r)
        v = det.update(engaged=True, steering_input=None if fb is None else fb.steering_input,
                       throttle_input=0.0, brake_input=0.0, applied_seq=None if fb is None else n,
                       now=t, fresh=r.fresh, gap=r.gap)
        if r.fresh:
            out.append((round(t, 3), r, v))
        n += 1
        det.note_command(seq=n, steer=cmd, throttle=0.0, brake=0.0, now=t)
        t = round(t + 1.0 / read_hz, 6)
    return out


def check_gap_credit_capped() -> None:
    """A long gap is unobserved time, not a held wheel.

    The interval a gap sample held back goes to the next agreeing echo, capped at one nominal
    echo interval (GAP_CREDIT_MAX_S = 0.10 s, gvd_main EGO_POLL_S), and that confirming echo
    cannot fire the takeover itself.
    """
    import python.control.override as ov

    assert GAP_CREDIT_MAX_S == 0.10, GAP_CREDIT_MAX_S
    cfg = OverrideConfig()
    cmd = 0.2
    base = [(0.1 * j, j, cmd) for j in range(1, 11)]  # 1 s on the command: armed, residual 0
    t0 = base[-1][0]
    pulled = cmd - 0.25
    # Reviewer case: a 290 ms gap (under the 300 ms stale limit; echoes 11 and 12 missed),
    # the echo after it a spike (a 0.25 jump), then one agreeing echo 100 ms later. No
    # takeover through that echo and the re-reads of it.
    tl = base + [(t0 + 0.29, 13, pulled), (t0 + 0.39, 14, pulled)]

    def run(timeline):
        return _scripted(timeline, cmd=cmd, t_end=timeline[-1][0] + 0.08)

    seen = run(tl)
    assert all(not v.active for (_t, _r, v) in seen), [(t, v.steer_held_ms, v.active) for (t, _r, v) in seen]
    after = [(t, r, v) for (t, r, v) in seen if t > t0 + 0.2]
    (tg, rg, vg), (ta, _ra, va) = after[:2]
    assert rg.gap and vg.gap_hold and not va.gap_hold and not va.spike, (rg, vg, va)
    own = (ta - tg) * 1000.0
    # The dwell after the confirming echo: its own interval plus at most one echo interval.
    assert va.steer_held_ms <= own + GAP_CREDIT_MAX_S * 1000.0 + 1.0, (va.steer_held_ms, own)
    print(f"  gap credit cap {GAP_CREDIT_MAX_S * 1000:.0f} ms: 290 ms gap + spike + 1 agreeing echo -> dwell {va.steer_held_ms:.0f} ms, no takeover")
    # Uncapped (the b7b3213 behaviour) the same echo charged the whole gap.
    keep = ov.GAP_CREDIT_MAX_S
    try:
        ov.GAP_CREDIT_MAX_S = 10.0
        unc = [v for (t, _r, v) in run(tl) if t > t0 + 0.2][1]
    finally:
        ov.GAP_CREDIT_MAX_S = keep
    assert unc.steer_held_ms >= own + 280.0, (unc.steer_held_ms, own)
    # The confirming echo charges but never fires: held on, the next echo is the takeover,
    # the same three fresh echoes a contiguous pull needs (spike, agree, agree).
    tl3 = base + [(t0 + 0.29, 13, pulled), (t0 + 0.39, 14, pulled), (t0 + 0.49, 15, pulled)]
    seen3 = [(t, v) for (t, _r, v) in run(tl3) if t > t0 + 0.2]
    assert not seen3[0][1].active and not seen3[1][1].active and seen3[2][1].active, seen3
    assert abs(cfg.steer_hold_ms - 200.0) < 1e-9


def _inject_around_pull(t_pull: float):
    """Skipped and repeated echo_seq around a pull, two contiguous echoes between injections
    (three gaps in a row is echo_stale by design, not what this measures):

    * every 5th echo before the pull is missed;
    * the first echo carrying the pull is missed: the pull lands right after a skipped seq;
    * three echoes later the mod's counter restarts (a reload): echo_seq steps back and then
      repeats values already used, in the middle of the dwell.
    """

    def edit(echoes):
        i_pull = next(i for i, (t, _k, _v) in enumerate(echoes) if t >= t_pull)
        out = []
        for i, (t, k, v) in enumerate(echoes):
            if i == i_pull or (i < i_pull - 2 and k % 5 == 0):
                continue
            out.append((t, k, v))
        # Index in `out` of the third kept echo after the missed one.
        j = next(n for n, (t, _k, _v) in enumerate(out) if t >= t_pull) + 3
        k0 = out[j][1]
        return out[:j] + [(t, k - k0, v) for (t, k, v) in out[j:]]

    return edit


def _injected_takeover_ms(gated: bool, hz: float, seed: int, runs: int = 300) -> list[float]:
    """0.25 pull against a 0.2 command through `_inject_around_pull`, read at `hz` with jitter
    (so most reads also repeat the echo already judged)."""
    rng = random.Random(seed)
    t_pull, cmd = 1.0, 0.2
    out = []
    for _ in range(runs):
        reason, t, seen = drive(
            rng,
            wheel=lambda t, _k: cmd if t < t_pull else cmd - 0.25,
            seconds=4.0,
            sup_hz=hz,
            gated=gated,
            cmd_steer=cmd,
            edit_echoes=_inject_around_pull(t_pull),
        )
        assert reason == "player_steer", (gated, hz, reason)
        if gated:
            assert "repeat" in seen, "no repeated echo_seq was read"
        out.append((t - t_pull) * 1000.0)
    return out


def check_takeover_through_injected_echoes() -> None:
    """A real pull through repeated and skipped echoes still takes over in time.

    "before" is c5ed013's consumption (every read a sample, no gate) on the same injected
    timeline; the gated path must be within before + 100 ms, mean and p95.
    """
    print("  takeover through skipped/repeated echo_seq (0.25 pull vs 0.2 command):")
    for hz in (15.0, 20.0):
        b = _injected_takeover_ms(False, hz, seed=91)
        a = _injected_takeover_ms(True, hz, seed=91)
        bm, am = statistics.mean(b), statistics.mean(a)
        b95, a95 = sorted(b)[int(0.95 * len(b))], sorted(a)[int(0.95 * len(a))]
        print(f"    {hz:.0f} Hz loop: before (c5ed013) mean {bm:.0f} p95 {b95:.0f} | after mean {am:.0f} p95 {a95:.0f}")
        assert am <= bm + TAKEOVER_BUDGET_MS, (hz, bm, am)
        assert a95 <= b95 + TAKEOVER_BUDGET_MS, (hz, b95, a95)


def check_cmd_stop_not_stale() -> None:
    """Commands stop reaching the mod for 1 s while it keeps echoing: never echo_stale.

    gvd_main stamps echo_seq from its own counter, so the echoes keep moving on. The wrong
    meaning (echo_seq = last command seq) freezes with the commands and trips echo_stale on a
    live wheel; that run is here only to prove this check can tell the two apart.
    """
    for hz in (15.0, 20.0):
        for seed in range(20):
            reason, _t, seen = drive(
                random.Random(seed), wheel=lambda _t, _k: 0.0, seconds=3.0, sup_hz=hz,
                gated=True, cmd_stop=(1.0, 2.0), seq_meaning="lua",
            )
            assert reason == "none", (hz, seed, reason)
            assert REASON_ECHO_STALE not in seen and "stale" not in seen, (hz, seed)
        wrong, _t, _seen = drive(
            random.Random(0), wheel=lambda _t, _k: 0.0, seconds=3.0, sup_hz=hz,
            gated=True, cmd_stop=(1.0, 2.0), seq_meaning="cmd",
        )
        assert wrong == REASON_ECHO_STALE, f"command-seq echo_seq should freeze and trip stale, got {wrong}"


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
    check_mod_version_gate()
    check_ego_parse()
    check_chatter_rule_exact()
    check_gap_credit_capped()
    check_cmd_stop_not_stale()
    check_repeat_skip_stress()
    check_takeover_time()
    check_takeover_through_injected_echoes()
    check_stale_fails_safe()
    check_outdated_refused()
    print("test_ffb_echo_fresh: OK")


if __name__ == "__main__":
    main()
