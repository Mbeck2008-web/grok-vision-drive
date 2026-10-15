#!/usr/bin/env python3
"""Replay the PR 75 lane-hold log through the steer law.

The live commands are ``clip(lookahead_x / 2.5, -1, 1)`` with ``x`` rounded
to 4 decimals (``path_steer_from_ego`` on that branch, the return after
``steer_sample_point``). 43% of 1,910 commands sat at ±1, and the first 741
reversed sign 102 times. Logged curvature at the same seconds is ~0, which
is about 0.02 m of arc sagitta at 8 m — not the lateral the wheel read.

The capture has no ``path_ego`` and no lookahead ``(x, y)``. Unsaturated
commands identify that ``x``. Saturated commands only say ``|x| >= 2.5``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    STEER_LOCK_RAD,
    STEER_RATE_PER_S,
    WHEELBASE_M,
    LaneSteerState,
    follow_lane_steer,
    path_steer_from_ego,
)

FIXTURE = ROOT / "tests" / "fixtures" / "pr75_lane_hold"
# Sample clock minus command clock for the third live run (sup3 / cmdlog).
SAMPLE_OFFSET_S = 556.11


def _legacy_steer(lateral_m: float) -> float:
    """PR 75 ``path_steer_from_ego``: meters / 2.5, clipped. Not the fix."""
    return max(-1.0, min(1.0, float(lateral_m) / 2.5))


def _path_at(lateral_m: float, y_m: float = 8.0) -> list[dict[str, float]]:
    """Straight ego polyline. The y>=8 sample is ``lateral_m``."""
    return [{"x": float(lateral_m), "y": float(i), "z": 0.0} for i in range(0, 16)]


def _arc_from_curvature(curvature: float, horizon_m: float = 36.0) -> list[dict[str, float]]:
    """Arc from the bumper, heading 0. This is what curvature alone can build."""
    k = float(curvature)
    pts: list[dict[str, float]] = []
    x = y = heading = 0.0
    n = int(horizon_m)
    for _ in range(n + 1):
        pts.append({"x": x, "y": y, "z": 0.0})
        heading += k
        x += math.sin(heading)
        y += math.cos(heading)
    return pts


def _load() -> tuple[list[dict], list[dict]]:
    cmds = [json.loads(line) for line in (FIXTURE / "cmdlog_head.jsonl").read_text().splitlines() if line]
    samples = [json.loads(line) for line in (FIXTURE / "samples_aligned.jsonl").read_text().splitlines() if line]
    return cmds, samples


def _nearest_sample(samples: list[dict], cmd_t: float, t0: float) -> dict | None:
    target = (cmd_t - t0) + SAMPLE_OFFSET_S
    best = None
    best_dt = 1e9
    for sample in samples:
        dt = abs(float(sample["t"]) - target)
        if dt < best_dt:
            best, best_dt = sample, dt
    if best is None or best_dt > 0.7:
        return None
    return best


def check_log_is_legacy_law() -> None:
    cmds, samples = _load()
    assert cmds and samples
    assert "path_ego" not in samples[0]
    assert "lateral_m" not in samples[0]["planner"]
    ok = [c for c in cmds if c["reason"] == "ok"]
    unsaturated = [c for c in ok if abs(float(c["out_steer"])) < 0.999]
    assert unsaturated, "fixture lost the unsaturated commands"
    for cmd in unsaturated:
        steer = float(cmd["out_steer"])
        # Path points are round(x, 4). The command is that x / 2.5.
        lateral = round(steer * 2.5, 4)
        assert abs(_legacy_steer(lateral) - steer) < 1e-6, (steer, lateral)
        replay = path_steer_from_ego(_path_at(lateral))
        # 0.02 m must not come back out near the logged command's scale
        # when the logged command was already the old gain. The new law is
        # a wheel angle over 30 deg, so it is much smaller than meters/2.5.
        assert abs(replay) < abs(steer) * 0.5 + 0.02, (steer, lateral, replay)
        assert abs(replay) < 0.6, (steer, replay)

    saturated = [c for c in ok if abs(float(c["out_steer"])) >= 0.999]
    assert len(saturated) >= 10, len(saturated)
    for sign in (1.0, -1.0):
        legacy = _legacy_steer(sign * 2.5)
        assert legacy == sign
        pursuit = path_steer_from_ego(_path_at(sign * 2.5))
        assert pursuit * sign > 0.0
        assert abs(pursuit) < 0.5, pursuit


def check_curvature_is_not_the_steer_input() -> None:
    """An arc from the logged curvature does not reproduce a saturated command.

    A real bend can saturate ``x / 2.5`` the same way a large offset does, so
    some commands do match the arc. A path with ``|curvature| < 0.002`` has
    about 0.02 m of sagitta at 8 m. That arc cannot be the ±1 command.
    """
    cmds, samples = _load()
    t0 = float(cmds[0]["t"])
    explained = 0
    straight_but_saturated = 0
    straight_explained = 0
    for cmd in cmds:
        if cmd["reason"] != "ok":
            continue
        sample = _nearest_sample(samples, float(cmd["t"]), t0)
        if sample is None:
            continue
        curvature = float(sample["planner"]["curvature"])
        arc = _arc_from_curvature(curvature)
        arc_x = next(p["x"] for p in arc if p["y"] >= 8.0)
        arc_steer = _legacy_steer(arc_x)
        logged = float(cmd["out_steer"])
        straight_sat = abs(logged) >= 0.999 and abs(curvature) < 0.002
        if straight_sat:
            straight_but_saturated += 1
            if abs(arc_steer - logged) < 0.05:
                straight_explained += 1
            assert abs(arc_steer) < 0.05, (curvature, arc_steer, logged)
        if abs(arc_steer - logged) < 0.05:
            explained += 1
    assert straight_but_saturated >= 5, straight_but_saturated
    assert straight_explained == 0, straight_explained
    # Bends that really are curved still match. They are not the straight ones.
    assert explained < straight_but_saturated, (explained, straight_but_saturated)


def _flips(values: list[float]) -> int:
    return sum(1 for a, b in zip(values, values[1:]) if a * b < 0.0)


def check_rate_limit_blocks_log_flips() -> None:
    cmds, samples = _load()
    t0 = float(cmds[0]["t"])
    state = LaneSteerState()
    prev_t = None
    out: list[float] = []
    logged_ok: list[float] = []
    printed = 0
    # sample_x is the logged lookahead (recovered). old_steer is x/2.5.
    # pursuit is the wheel-angle law. limited is that law after the PD slew.
    print("replay  sample_x  curvature  old_steer  pursuit  limited")
    for cmd in cmds:
        t = float(cmd["t"])
        dt = None if prev_t is None else t - prev_t
        prev_t = t
        # Live supervisor resets the follower on any non-ok command.
        if cmd["reason"] != "ok":
            state.reset()
            continue
        steer = float(cmd["out_steer"])
        logged_ok.append(steer)
        if abs(steer) >= 0.999:
            lateral = math.copysign(2.5, steer)
            x_txt = ">=2.5"
        else:
            lateral = round(steer * 2.5, 4)
            x_txt = f"{lateral:+.4f}"
        armed_before = state.armed
        origin = state.steer if armed_before else 0.0
        geometric = path_steer_from_ego(_path_at(lateral))
        followed = follow_lane_steer(geometric, lateral, state, dt)
        out.append(followed)
        if dt is not None and dt > 1e-4:
            assert abs(followed - origin) <= STEER_RATE_PER_S * dt + 1e-9, (
                followed, origin, dt,
            )
        if printed < 12:
            sample = _nearest_sample(samples, t, t0)
            if sample is None:
                continue
            curvature = float(sample["planner"]["curvature"])
            print(
                f"  {x_txt:>8}  {curvature:+.5f}  {steer:+.5f}  {geometric:+.5f}  {followed:+.5f}"
            )
            printed += 1
    assert printed == 12
    logged_flips = _flips(logged_ok)
    out_flips = _flips(out)
    print(f"sign_flips old={logged_flips} new={out_flips}")
    assert logged_flips >= 8, logged_flips
    assert out_flips < logged_flips, (out_flips, logged_flips)
    assert max(abs(v) for v in out) < 0.6
    # One tick at the live median spacing cannot reverse a full-lock request.
    jump = LaneSteerState()
    held = follow_lane_steer(path_steer_from_ego(_path_at(2.5)), 2.5, jump, None)
    assert held > 0.2
    reversed_cmd = follow_lane_steer(path_steer_from_ego(_path_at(-2.5)), -2.5, jump, 0.31)
    assert reversed_cmd > 0.0, reversed_cmd
    # A 2 cm lookahead, the curvature sagitta the log actually supports,
    # stays a small wheel angle even with no rate limit.
    tiny = path_steer_from_ego(_path_at(0.02))
    assert abs(tiny) < 0.02, tiny
    # And the pursuit angle is the wheel angle over the 30 deg lock.
    delta = math.atan2(2.0 * WHEELBASE_M * 0.02, 0.02 * 0.02 + 64.0)
    assert abs(tiny - delta / STEER_LOCK_RAD) < 1e-12


def main() -> None:
    check_log_is_legacy_law()
    check_curvature_is_not_the_steer_input()
    check_rate_limit_blocks_log_flips()
    print("test_lane_hold_replay: OK")


if __name__ == "__main__":
    main()
