#!/usr/bin/env python3
"""Replay the PR 82 lane-hold log through the steer law.

The fixture is the engaged ``[GVD] seg`` lines from a BeamNG.tech run of
de9de9e (west_coast_usa, etk800, about 90 s). Logged ``steer`` is what that
build sent. 105 of 114 ``lane_x=na`` ticks are exactly 0. A derivative cap of
0.20 on a 120 ms late sample also cut the command while ``lane_x`` grew.

Replaying those lines with the old law goes to 0 on a dropout and can flip
sign on a physical rise in ``lane_x``. The current law holds and keeps the
sign. The assertions fail if ``follow_lane_steer`` goes back to that.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    LANE_HOLD_DECAY_PER_S,
    STEER_D_CAP,
    STEER_RATE_PER_S,
    LaneSteerState,
    follow_lane_steer,
    lane_sample_hold_reason,
    path_steer_from_ego,
)
from python.perception.pipeline import drive_plan_from_lanes  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "pr82_lane_hold" / "segs.jsonl"


def _legacy_follow(geometric: float, lateral_m: float | None, state: dict, dt_s: float | None) -> float:
    """de9de9e ``follow_lane_steer``. D cap 0.20. A missing lateral slews to 0."""
    raw = float(geometric)
    lat = None if lateral_m is None else float(lateral_m)
    d_term = 0.0
    if lat is not None and state["have"] and dt_s is not None and dt_s > 1e-4:
        d_lat = (lat - state["lat"]) / float(dt_s)
        d_term = max(-0.20, min(0.20, -0.05 * d_lat))
    raw = max(-1.0, min(1.0, raw + d_term))
    if dt_s is not None and dt_s > 1e-4:
        step = 0.75 * float(dt_s)
        origin = state["steer"] if state["armed"] else 0.0
        raw = max(origin - step, min(origin + step, raw))
    state["steer"] = raw
    state["armed"] = True
    if lat is not None:
        state["lat"] = lat
        state["have"] = True
    return raw


def _path(x: float, y: float) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)}, {"x": float(x), "y": float(y) + 1.0}]


def _load() -> list[dict]:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]
    assert len(rows) > 400, len(rows)
    return rows


def _replay(rows: list[dict]) -> list[dict]:
    state = LaneSteerState()
    out: list[dict] = []
    for row in rows:
        dt = float(row["dt"])
        x = row["lane_x"]
        y = row["lane_y"]
        if x is None or y is None:
            steer = follow_lane_steer(0.0, None, state, dt, miss_reason="lane_hold")
        else:
            miss = lane_sample_hold_reason(float(x), float(y))
            if miss:
                steer = follow_lane_steer(0.0, None, state, dt, miss_reason=miss)
            else:
                geom = path_steer_from_ego(_path(float(x), float(y)))
                steer = follow_lane_steer(geom, float(x), state, dt)
        out.append({
            "lane_x": x,
            "lane_y": y,
            "old": float(row["steer"]),
            "new": float(steer),
            "why": state.reason,
            "dt": dt,
        })
    return out


def check_old_law_zeros_and_new_holds() -> None:
    """The synthetic the log is full of: a command, then the lane vanishes."""
    geom = path_steer_from_ego(_path(2.0, 8.0))
    new_state = LaneSteerState()
    old_state = {"steer": 0.0, "lat": 0.0, "have": False, "armed": False}
    primed = follow_lane_steer(geom, 2.0, new_state, None)
    old_primed = _legacy_follow(geom, 2.0, old_state, None)
    assert abs(primed - old_primed) < 1e-9
    assert primed > 0.2, primed
    for _ in range(8):
        new = follow_lane_steer(0.0, None, new_state, 0.14, miss_reason="lane_hold")
        old = _legacy_follow(0.0, None, old_state, 0.14)
    assert old == 0.0, old
    assert new > 0.10, new
    assert new_state.reason == "lane_hold"
    print(f"dropout  lane_x=na       old_steer={old:+.4f}  new_steer={new:+.4f}  lane_hold")
    # Eight ticks at the live spacing cannot dump the held command.
    assert abs(new - primed) <= LANE_HOLD_DECAY_PER_S * 0.14 * 8 + 1e-9


def check_growing_lane_is_not_cut() -> None:
    """0.5 m to 0.9 m in 140 ms is a real drift. The old D term flipped the sign."""
    a = path_steer_from_ego(_path(0.5, 8.0))
    b = path_steer_from_ego(_path(0.9, 8.0))
    new_state = LaneSteerState()
    old_state = {"steer": 0.0, "lat": 0.0, "have": False, "armed": False}
    follow_lane_steer(a, 0.5, new_state, None)
    _legacy_follow(a, 0.5, old_state, None)
    new = follow_lane_steer(b, 0.9, new_state, 0.14)
    old = _legacy_follow(b, 0.9, old_state, 0.14)
    assert old < 0.0, old
    assert new > 0.05, new
    assert new > old
    assert STEER_D_CAP <= 0.05
    print(f"grow    lane_x=+0.9000  old_steer={old:+.4f}  new_steer={new:+.4f}  pursuit")


def check_log_dropout_stays_up() -> list[dict]:
    rows = _load()
    played = _replay(rows)
    na_logged_zero = [
        i for i, row in enumerate(played)
        if row["lane_x"] is None and abs(row["old"]) < 1e-9
    ]
    assert len(na_logged_zero) >= 80, len(na_logged_zero)
    # The live wheel was centered. The replay of those same ticks is not,
    # except where the hold has been decaying for a long gap.
    still_up = [i for i in na_logged_zero if abs(played[i]["new"]) > 0.05]
    assert len(still_up) >= 40, len(still_up)
    # First tick of a dropout does not step toward 0 at the steer rate limit.
    # It only creeps by the hold decay.
    checked = 0
    for i in range(1, len(played)):
        prev, row = played[i - 1], played[i]
        if row["why"] not in ("lane_hold", "lane_near", "lane_far"):
            continue
        if prev["why"] == "pursuit" and abs(prev["new"]) > 0.15:
            creep = LANE_HOLD_DECAY_PER_S * row["dt"] + 1e-9
            assert abs(row["new"] - prev["new"]) <= creep, (row, prev)
            checked += 1
    assert checked >= 8, checked
    return played


def check_rate_limit(played: list[dict]) -> None:
    for prev, row in zip(played, played[1:]):
        step = STEER_RATE_PER_S * row["dt"] + 1e-9
        assert abs(row["new"] - prev["new"]) <= step, (prev, row)


def check_rejected_samples() -> None:
    assert lane_sample_hold_reason(0.4, 8.0) is None
    assert lane_sample_hold_reason(1.2, 0.4) == "lane_near"
    assert lane_sample_hold_reason(12.0, 8.3) == "lane_far"
    assert lane_sample_hold_reason(0.7, 8.0) is None


def check_prior_fills_a_bad_frame() -> None:
    good = [
        [{"x": -1.75, "y": float(y)} for y in range(0, 20, 2)],
        [{"x": 1.75, "y": float(y)} for y in range(0, 20, 2)],
    ]
    plan, stored = drive_plan_from_lanes(
        good, [], lane_conf=0.8, curvature=0.0, tracks=[], signs=[], ego_speed_mps=5.0,
    )
    assert plan.drivable and stored
    # One point is not a route. The last good lines stay, and we still drive.
    bad = [[{"x": 9.0, "y": 1.0}]]
    held, stored_after = drive_plan_from_lanes(
        bad, stored, lane_conf=0.0, curvature=0.0, tracks=[], signs=[], ego_speed_mps=5.0,
    )
    assert held.drivable and held.prediction == "context", held.prediction
    assert stored_after == stored
    empty, still = drive_plan_from_lanes(
        [], stored, lane_conf=0.0, curvature=0.0, tracks=[], signs=[], ego_speed_mps=5.0,
    )
    assert empty.drivable and empty.prediction == "context"
    assert still == stored


def _print_table(played: list[dict]) -> None:
    want: list[int] = list(range(8))
    for i, row in enumerate(played):
        if row["lane_x"] is None:
            want.extend(range(max(0, i - 1), min(len(played), i + 6)))
            break
    for i, row in enumerate(played):
        x = row["lane_x"]
        if x is not None and abs(float(x)) > 8.0:
            want.extend(range(max(0, i - 1), min(len(played), i + 3)))
            break
    # A quiet stretch: small lane_x, pursuit, after the wheel has settled.
    for i, row in enumerate(played):
        x = row["lane_x"]
        if row["why"] == "pursuit" and x is not None and abs(float(x)) < 0.5 and i > 40:
            want.append(i)
            break
    seen: list[int] = []
    for i in want:
        if i not in seen:
            seen.append(i)
    print("replay  lane_x     old_steer  new_steer  why")
    for i in seen:
        row = played[i]
        x = "na" if row["lane_x"] is None else f"{float(row['lane_x']):+.4f}"
        print(f"  {i:4d}  {x:>8}  {row['old']:+.4f}  {row['new']:+.4f}  {row['why']}")
    na = [r for r in played if r["lane_x"] is None]
    zero = [r for r in na if abs(r["old"]) < 1e-9]
    up = [r for r in zero if abs(r["new"]) > 0.05]
    print(
        f"na={len(na)} logged_zero={len(zero)} new_still_up={len(up)} "
        f"max_new={max(abs(r['new']) for r in played):.3f}"
    )


def main() -> None:
    check_rejected_samples()
    check_old_law_zeros_and_new_holds()
    check_growing_lane_is_not_cut()
    check_prior_fills_a_bad_frame()
    played = check_log_dropout_stays_up()
    check_rate_limit(played)
    _print_table(played)
    print("test_lane_hold_pr82: OK")


if __name__ == "__main__":
    main()
