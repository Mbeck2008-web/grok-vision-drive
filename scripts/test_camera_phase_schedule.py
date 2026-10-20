#!/usr/bin/env python3
"""Camera capture is phase-scheduled: the delivered rate matches the config.

The elapsed-time check (``now - last + 1e-4 >= 1/hz``, main up to 1addc93
and 6b59eef) restarted each camera's period at its last frame. The
collector's 0.1 s tick jitters and 44% of rec5c's ticks came in under
0.1 s, so a 10 Hz camera skipped them: 5.85 Hz delivered at 8.94 Hz of
state rows. ``CameraSchedule`` keeps a phase grid per camera (``next_due
+= period``), serves each phase on the tick nearest to it, and skips
ahead past missed phases instead of bursting.

Checks, each on jittered synthetic ticks and on the real rec5c tick
deltas (tests/fixtures/rec5c_tick_deltas.txt.gz, microseconds, one block
per segment; no absolute times):

* delivered rate per camera >= 0.95 x min(configured hz, measured tick rate)
  at 10, 2 and 1 Hz;
* never two frames of one camera on one tick, and no catch-up burst after
  a long gap;
* the slow cameras are staggered onto different ticks;
* the old elapsed-time check fails the rate check on the same ticks.

``--impl camera_due`` runs the rate checks against ``camera_due`` from the
checked-out ``python.data.raw_frames``. On main that is the old check, so
the test can be run on main's own tree to show it failing.
"""
from __future__ import annotations

import argparse
import gzip
import math
import random
import sys
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "rec5c_tick_deltas.txt.gz"
RATE_FLOOR = 0.95
GROUPS = (("main", "narrow", "wide"), ("pillarL", "pillarR"), ("repeatL", "repeatR", "rear"))
SLOW = ("pillarL", "pillarR", "repeatL", "repeatR", "rear")

# Recorded in rec5c (state.jsonl cams per row): main frames 57,743 with
# 27,209 one-tick steps, 30,522 two-tick steps and 11 three-tick steps.
REC5C_MAIN_FRAMES = 57743

def legacy_due(cam_id: str, now_s: float, last_s: Mapping[str, float], rates: Mapping[str, float] | None = None) -> bool:
    """The due-check from main (1addc93 .. 6b59eef), verbatim apart from the name."""
    from python.data.raw_frames import CAMERA_HZ

    table = CAMERA_HZ if rates is None else rates
    hz = float(table.get(str(cam_id), 0.0))
    if hz <= 0.0:
        return False
    prev = last_s.get(str(cam_id))
    if prev is None:
        return True
    return (float(now_s) - float(prev)) + 1e-4 >= (1.0 / hz)


class ElapsedSchedule:
    """An elapsed-time due-check driven the way the old collector loop drove it."""

    def __init__(self, table: Mapping[str, float], due: Callable[..., bool] = legacy_due) -> None:
        self.table = dict(table)
        self.due = due
        self.last: dict[str, float] = {}

    def tick(self, now: float, cams) -> list[str]:
        taken = []
        for cid in cams:
            if self.due(cid, now, self.last, self.table):
                self.last[cid] = now
                taken.append(cid)
        return taken


class SkipOnlySchedule:
    """Phase grid with skip-ahead but no nearest-tick rounding (a tick takes only due phases)."""

    def __init__(self, table: Mapping[str, float]) -> None:
        from python.data.raw_frames import CameraSchedule

        self.inner = CameraSchedule(table)

    def tick(self, now: float, cams) -> list[str]:
        self.inner.prev_tick = None  # no lead: only phases already due
        return self.inner.tick(now, cams)


def phase_schedule(phases: Mapping[str, float] | None = None) -> Callable[[Mapping[str, float]], object]:
    from python.data.raw_frames import CameraSchedule

    return lambda table: CameraSchedule(table, phases)


def rates() -> dict[str, float]:
    from python.data.raw_frames import CAMERA_HZ

    return {str(k): float(v) for k, v in CAMERA_HZ.items()}


def load_rec5c() -> list[list[float]]:
    """Tick times per segment, seconds from the segment's first row."""
    segments: list[list[float]] = []
    current = [0.0]
    with gzip.open(FIXTURE, "rt", encoding="ascii") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("#"):
                continue
            if not line:
                segments.append(current)
                current = [0.0]
                continue
            current.append(current[-1] + int(line) / 1e6)
    if len(current) > 1:
        segments.append(current)
    return segments


def jittered(n: int, mean_s: float, sd_s: float, seed: int, gap_every: int = 0, gap_s: float = 0.0) -> list[float]:
    rng = random.Random(seed)
    times = [0.0]
    for k in range(1, n):
        step = max(0.01, rng.gauss(mean_s, sd_s))
        if gap_every and k % gap_every == 0:
            step += gap_s
        times.append(times[-1] + step)
    return times


def run(segments: list[list[float]], make: Callable[[Mapping[str, float]], object], table: Mapping[str, float], base: float = 1000.0) -> dict:
    """Replay tick times through a scheduler. A fresh schedule per segment, as a respawn does."""
    frames = {cid: [] for cid in table}  # (segment, tick index, time)
    per_tick: list[list[str]] = []
    span = 0.0
    ticks = 0
    for seg_n, seg in enumerate(segments):
        sched = make(table)
        for k, t in enumerate(seg):
            now = base + seg_n * 1e5 + t
            got = list(sched.tick(now, list(table)))
            for cid in got:
                frames[cid].append((seg_n, k, now))
            per_tick.append(got)
        span += seg[-1] - seg[0]
        ticks += len(seg) - 1
    return {"frames": frames, "per_tick": per_tick, "span_s": span, "tick_hz": ticks / span}


def delivered(result: dict) -> dict[str, float]:
    span = result["span_s"]
    out = {}
    for cid, items in result["frames"].items():
        # Count intervals, as the tick rate does (first frame of each segment is the anchor).
        segs = {s for s, _k, _t in items}
        out[cid] = (len(items) - len(segs)) / span
    return out


def rate_failures(result: dict, table: Mapping[str, float]) -> list[str]:
    got = delivered(result)
    bad = []
    for cid, hz in table.items():
        want = RATE_FLOOR * min(float(hz), result["tick_hz"])
        if got[cid] < want:
            bad.append(f"{cid} {got[cid]:.3f} Hz < {want:.3f} (0.95 x min({hz:g}, {result['tick_hz']:.3f}))")
    return bad


def rate_table(name: str, result: dict, table: Mapping[str, float]) -> str:
    got = delivered(result)
    rows = [f"{name}: ticks {result['tick_hz']:.3f} Hz over {result['span_s']:.0f} s"]
    for group in GROUPS:
        for cid in group:
            want = min(float(table[cid]), result["tick_hz"])
            rows.append(
                f"  {cid:8s} config {table[cid]:4.1f} Hz delivered {got[cid]:6.3f} Hz "
                f"({got[cid] / want:5.1%} of min(config, ticks))"
            )
    return "\n".join(rows)


def synthetic_cases() -> dict[str, list[list[float]]]:
    return {
        # 0.1 s plan, jitter like rec5c's core (44% under 0.1 s).
        "jitter 0.104+-0.02 s": [jittered(20000, 0.104, 0.02, seed=5)],
        # Ticks faster than the plan on average: 10 Hz must hold at 10, not tick rate.
        "jitter 0.092+-0.02 s": [jittered(20000, 0.092, 0.02, seed=6)],
        # Slower ticks: the 10 Hz cameras follow the tick rate.
        "jitter 0.125+-0.03 s": [jittered(20000, 0.125, 0.03, seed=7)],
    }


def gap_cases() -> dict[str, list[list[float]]]:
    # Long gaps every 300 ticks. No ticks, no frames: only the burst checks run here.
    return {"jitter + 3 s gaps": [jittered(20000, 0.102, 0.02, seed=8, gap_every=300, gap_s=3.0)]}


def check_rates(make: Callable[[Mapping[str, float]], object], label: str) -> list[str]:
    table = rates()
    problems = []
    for name, segs in synthetic_cases().items():
        result = run(segs, make, table)
        print(rate_table(f"{label} {name}", result, table))
        problems += [f"{name}: {p}" for p in rate_failures(result, table)]
    result = run(load_rec5c(), make, table)
    print(rate_table(f"{label} rec5c ticks", result, table))
    problems += [f"rec5c: {p}" for p in rate_failures(result, table)]
    return problems


def check_new_rates() -> None:
    problems = check_rates(phase_schedule(), "phase")
    assert not problems, problems


def check_old_check_fails() -> None:
    """The same rate check fails on the elapsed-time due-check."""
    table = rates()
    problems = check_rates(ElapsedSchedule, "old due-check")
    assert any(" main " in f" {p.split(': ', 1)[1]}" for p in problems), problems
    assert any(p.startswith("rec5c: main ") for p in problems), problems
    result = run(load_rec5c(), ElapsedSchedule, table)
    # The replay reproduces the recording: main within 5% of 57,743 frames,
    # and about as many two-tick steps as one-tick steps (recorded 30,522 vs
    # 27,209). (Not exact: the collector
    # checked against time.monotonic() after the camera poll, the fixture is
    # the row clock ``t`` read after the vehicle poll.)
    frames = result["frames"]["main"]
    print(f"old due-check replay main frames {len(frames)} (recorded {REC5C_MAIN_FRAMES})")
    assert abs(len(frames) - REC5C_MAIN_FRAMES) / REC5C_MAIN_FRAMES < 0.05, len(frames)
    steps: dict[int, int] = {}
    for (s0, k0, _a), (s1, k1, _b) in zip(frames, frames[1:]):
        if s0 == s1:
            steps[k1 - k0] = steps.get(k1 - k0, 0) + 1
    print(f"old due-check replay main tick steps {dict(sorted(steps.items()))} "
          "(recorded {1: 27209, 2: 30522, 3: 11})")
    assert steps.get(2, 0) > 0.8 * steps.get(1, 0), steps
    print(f"old due-check fails {len(problems)} rate checks (expected)")


def check_skip_only_falls_short() -> None:
    """Why the schedule rounds to the nearest tick: due-only takes miss 10 Hz on rec5c."""
    table = rates()
    result = run(load_rec5c(), SkipOnlySchedule, table)
    print(rate_table("skip-only phase (no nearest-tick rounding) rec5c ticks", result, table))
    got = delivered(result)["main"] / min(10.0, result["tick_hz"])
    assert got < RATE_FLOOR, got


def check_one_per_tick_and_no_burst() -> None:
    from python.data.raw_frames import CameraSchedule

    table = rates()
    # A second tick at the same instant takes nothing.
    sched = CameraSchedule(table)
    for now in (0.0, 0.1, 0.35, 2.0, 9.0):
        sched.tick(now, list(table))
        assert sched.tick(now, list(table)) == [], now
    # After a long gap: one frame on the first tick back, then the period.
    for gap in (0.25, 0.45, 1.3, 5.0, 60.0):
        sched = CameraSchedule(table)
        t = 0.0
        while t < 3.0:
            sched.tick(t, list(table))
            t += 0.1
        back = t + gap
        taken = {cid: [] for cid in table}
        for k in range(40):
            now = back + 0.1 * k
            for cid in sched.tick(now, list(table)):
                taken[cid].append(now)
        for cid, hz in table.items():
            period = 1.0 / hz
            # The phases in the 3.9 s after the gap, plus the one that came
            # due during the gap: no missed phase is made up.
            most = math.floor((4.0 - 0.1) / period + 1e-9) + 2
            assert len(taken[cid]) <= most, (cid, gap, len(taken[cid]), most)
            if hz >= 10.0:
                assert len(taken[cid]) == 40, (cid, gap, len(taken[cid]))
    # Full replays: one frame per camera per tick, at most hz + 1 frames in any
    # 1 s, and no frame of a slow camera on the tick right after a gap's first frame.
    cases = list(synthetic_cases().items()) + list(gap_cases().items()) + [("rec5c", load_rec5c())]
    pairs: dict[str, int] = {}
    for name, segs in cases:
        result = run(segs, phase_schedule(), table)
        for got in result["per_tick"]:
            assert len(got) == len(set(got)), (name, got)
        for cid, items in result["frames"].items():
            times = [t for _s, _k, t in items]
            hz = table[cid]
            # Never ahead of the grid: in any window of W seconds, at most the
            # W * hz phases plus one served late (it came due before the window)
            # and one served up to half a tick early (nearest-tick rounding).
            for window in (1.0, 10.0, 60.0):
                cap = math.floor(window * hz + 1e-9) + 2
                lo = 0
                for hi in range(len(times)):
                    while times[hi] - times[lo] >= window:
                        lo += 1
                    assert hi - lo + 1 <= cap, (name, cid, window, hi - lo + 1, cap)
            if hz < 10.0:
                # A slow camera on two consecutive ticks only as a late frame then
                # its next grid phase; three frames never inside half a period
                # (a catch-up run would put them on three ticks in a row).
                for a, b, c in zip(items, items[1:], items[2:]):
                    if a[0] == c[0]:
                        assert c[2] - a[2] >= 0.5 / hz, (name, cid, a[1], c[2] - a[2])
                pairs[cid] = pairs.get(cid, 0) + sum(
                    1 for a, b in zip(items, items[1:]) if a[0] == b[0] and b[1] == a[1] + 1
                )
    print("no burst: one frame per camera per tick; <= W*hz+2 frames in any W = 1, 10, 60 s; "
          "no slow-camera frame on 3 ticks in a row; gaps 0.25-60 s")
    print(f"slow-camera frames on 2 consecutive ticks (late frame + next phase), all replays: {pairs}")


def check_undo() -> None:
    from python.data.raw_frames import CameraSchedule

    table = rates()
    sched = CameraSchedule(table)
    first = sched.tick(0.0, list(table))
    sched.undo()
    assert sched.tick(0.05, list(table)) == first  # nothing was written: the anchor tick's frames are owed
    sched = CameraSchedule(table)
    assert "main" in sched.tick(0.0, ["main"])
    assert "main" in sched.tick(0.1, ["main"])
    sched.undo()  # the 0.1 s tick was not written
    assert "main" in sched.tick(0.13, ["main"])  # owed phase taken on the next tick
    assert sched.tick(0.15, ["main"]) == []
    print("undo: an unwritten tick's phase is taken on the next tick")


def check_stagger() -> None:
    from python.data.raw_frames import CAMERA_PHASE_S

    table = rates()
    # On an exact 0.1 s grid no tick carries more than one slow camera.
    grid = [[0.1 * k for k in range(1000)]]
    result = run(grid, phase_schedule(), table)
    worst = max(sum(1 for c in got if c in SLOW) for got in result["per_tick"])
    assert worst == 1, worst
    flat = run(grid, phase_schedule({}), table)
    assert max(sum(1 for c in got if c in SLOW) for got in flat["per_tick"]) == 5
    for name, segs in [("jitter", synthetic_cases()["jitter 0.104+-0.02 s"]), ("rec5c", load_rec5c())]:
        staggered = run(segs, phase_schedule(), table)
        unstaggered = run(segs, phase_schedule({}), table)

        def multi(res: dict) -> float:
            ticks = res["per_tick"]
            return sum(1 for got in ticks if sum(1 for c in got if c in SLOW) >= 2) / len(ticks)

        a, b = multi(staggered), multi(unstaggered)
        print(f"stagger {name}: ticks with 2+ slow-camera frames {a:.2%} staggered vs {b:.2%} same phase")
        assert a < 0.5 * b, (name, a, b)
        assert not rate_failures(staggered, table)
    assert set(CAMERA_PHASE_S) <= set(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("schedule", "camera_due"), default="schedule")
    args = parser.parse_args()
    if args.impl == "camera_due":
        # The rate checks against camera_due from this tree's raw_frames (main's API).
        from python.data import raw_frames

        problems = check_rates(lambda table: ElapsedSchedule(table, raw_frames.camera_due), "camera_due")
        for line in problems:
            print(f"FAIL {line}")
        assert not problems, f"{len(problems)} rate checks failed on camera_due"
        return
    check_new_rates()
    check_one_per_tick_and_no_burst()
    check_undo()
    check_stagger()
    check_skip_only_falls_short()
    check_old_check_fails()
    print("test_camera_phase_schedule OK")


if __name__ == "__main__":
    main()
