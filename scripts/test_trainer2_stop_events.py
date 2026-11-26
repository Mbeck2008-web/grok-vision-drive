#!/usr/bin/env python3
"""Trainer 2.0 PR 6b (scoring part): offline stop-sign / traffic-light events and metrics.

docs/specs/trainer_2.md, "Learned stop braking" (cuts decided by Michael 2026-11-25):
a stop counts only as a full stop (v < 0.5 m/s) held >= 0.5 s before the stop
line + 1 m; 40 m approach window; 2 m/s entry speed; false brake = ctl_brake
> 0.3 with no stop event within 40 m. Yellow counts as a stop event (spec
proposal). Green and unknown are never stop events.

Checks: check_stop_event_requires_hold, check_stop_side_angle_slice,
check_light_events_red_yellow_only, check_stop_metrics_on_fixture,
check_stop_events_split_safe, check_stop_labels_train_only,
check_hand_rule_untouched, check_missing_gt_row_does_not_end_the_window,
check_speed_key_scores_the_predicted_profile, check_side_angle_uses_first_visibility_not_entry,
check_km_counts_holdout_rows_only, check_no_nan_in_scores,
check_red_to_green_proceed_current_behaviour and
check_red_to_green_flag_false_drops_the_approach (PROPOSAL pending Michael).
Pure Python (no torch).
"""
from __future__ import annotations

import ast
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

DT = 0.1
# sha256 at main 8ce34d5 (spec: hand rule, planner fuse and AEB are never changed by Trainer 2.0).
HAND_RULE_PINS = {
    "python/planning/path_predictor.py": "22ca3f1158ec11b25396876e063ce8b1afdc2833d65660e6377becd09648db87",
    "python/perception/pipeline.py": "b49659341ef1a2d80be3d4f736381c02406e39d91bc9f6c2d333636d32f48d14",
    "python/control/actuate.py": "3d5edfcdb4d7d98bacb98c1f6b75cf0ce9f2d79caf08c75c2eb307aa5d446126",
    "python/perception/cipv.py": "ca743f45f4e385550255c0d3d0d1d522ba86b8efaeefb038c32cc42bf34fb9b3",
}


def approach(
    *,
    sid: str = "s1",
    cls: str = "stop_sign",
    state: str | None = None,
    y0: float = 50.0,
    v0: float = 10.0,
    x: float = 0.0,
    stop_at_y: float | None = 2.0,
    hold_s: float = 1.0,
    t0: float = 0.0,
    split: str = "holdout",
    source: str = "rec",
    i0: int = 0,
    brake: float = 0.6,
):
    """Rows of one approach (y = metres to the stop line, 10 Hz).

    Cruise at ``v0``, brake from ``y0 - 15`` m so the car reaches 0 m/s at
    ``stop_at_y``, sit at exactly 0 m/s for ``hold_s`` (the only rows below
    0.5 m/s), then pull away at 1 m/s and accelerate through the line.
    ``stop_at_y=None`` drives through at ``v0``.
    """
    rows = []
    t, y, v = t0, y0, v0
    decel = None if stop_at_y is None else v0 * v0 / (2.0 * max(0.1, (y0 - 15.0) - stop_at_y))
    hold_rows = int(round(hold_s / DT))
    held = 0
    i = i0
    while y > -10.0 and t < t0 + 120:
        braking = 0.0
        if decel is not None and y <= y0 - 15.0 and held < hold_rows:
            braking = brake
            if v > 0.0:
                nxt = v - decel * DT
                # land exactly on the stop: no partial rows below 0.5 m/s before the hold
                v = 0.0 if nxt < 0.5 or y - v * DT <= stop_at_y else nxt
                if v == 0.0:
                    y = stop_at_y
            if v == 0.0:
                held += 1
        elif decel is not None and held >= hold_rows:
            v = 1.0 if v < 1.0 else min(v0, v + 2.0 * DT)
        target = {"id": sid, "cls": cls, "x": x, "y": round(y, 4)}
        if state is not None:
            target["state"] = state
        rows.append({"t": round(t, 6), "v": round(v, 6), "ctl_brake": braking, "split": split,
                     "source": source, "i": i, "stops": [target]})
        y -= v * DT
        t += DT
        i += 1
    return rows


def check_stop_event_requires_hold() -> None:
    from python.train import stop_events as se

    assert (se.STOP_V_MPS, se.HOLD_S, se.LINE_TOL_M, se.WINDOW_M, se.ENTRY_V_MPS, se.BRAKE_ON) == (0.5, 0.5, 1.0, 40.0, 2.0, 0.3)
    good = se.score(approach(hold_s=1.0))
    assert good["events"] == 1 and good["missed"] == 0, good
    dip = se.score(approach(hold_s=0.3))  # dips below 0.5 m/s but holds only 0.3 s
    assert dip["events"] == 1 and dip["missed"] == 1, dip
    run = se.score(approach(stop_at_y=None))
    assert run["events"] == 1 and run["missed"] == 1, run
    late = se.score(approach(stop_at_y=-2.0, hold_s=1.0))  # stops 2 m past the line (> line + 1 m)
    assert late["missed"] == 1, late
    slow = se.score(approach(v0=1.5, hold_s=1.0))  # entry speed < 2 m/s: not a scored approach
    assert slow["events"] == 0, slow
    exact = se.score(approach(hold_s=0.5))  # exactly 0.5 s at 0 m/s: held
    assert exact["missed"] == 0, exact
    short = se.score(approach(hold_s=0.4))
    assert short["missed"] == 1, short
    print("stop hold: held 1.0 s ok; 0.3 s dip, run-through and stop past line+1 m are misses; <2 m/s entry not scored OK")


def check_stop_side_angle_slice() -> None:
    from python.train import stop_events as se

    y0 = 30.0
    x30 = y0 * math.tan(math.radians(30.0))  # 30 deg off-axis at first visibility
    rows = approach(sid="side", y0=y0, x=x30, stop_at_y=None)
    rows += approach(sid="ahead", y0=45.0, x=0.0, t0=200.0, i0=10_000, hold_s=1.0)
    res = se.score(rows)
    assert res["events"] == 2 and res["missed"] == 1, res
    side = res["slices"]["stop_side_angle"]
    assert side == {"events": 1, "missed": 1}, side
    assert math.isclose(se.bearing_deg({"x": x30, "y": y0}), 30.0, abs_tol=1e-9)
    assert math.isclose(se.bearing_deg({"x": -x30, "y": y0}), 30.0, abs_tol=1e-9)
    edge = se.score(approach(y0=30.0, x=30.0 * math.tan(math.radians(40.0)), stop_at_y=None))
    assert edge["slices"]["stop_side_angle"]["events"] == 0, edge
    print("side angle: 30 deg event in stop_side_angle (1 event, 1 missed), reported apart from the 0 deg event OK")


def check_light_events_red_yellow_only() -> None:
    from python.train import stop_events as se

    red = se.score(approach(cls="traffic_light", state="red", hold_s=1.0))
    yellow = se.score(approach(cls="traffic_light", state="yellow", stop_at_y=None))
    green = se.score(approach(cls="traffic_light", state="green", stop_at_y=None, brake=0.0))
    unknown = se.score(approach(cls="traffic_light", state="unknown", stop_at_y=None, brake=0.0))
    assert red["events"] == 1 and red["missed"] == 0 and red["slices"]["red_light"] == {"events": 1, "missed": 0}, red
    assert yellow["events"] == 1 and yellow["missed"] == 1, yellow
    assert green["events"] == 0 and unknown["events"] == 0, (green, unknown)
    # Green light, model brakes hard with no other stop event in 40 m: a green-light false brake.
    gfb = se.score(approach(cls="traffic_light", state="green", hold_s=1.0, brake=0.6))
    assert gfb["events"] == 0 and gfb["green_false_brakes"] == 1 and gfb["false_brakes"] == 1, gfb
    # Green light but a stop sign in the same 40 m: braking is for the sign, not a green false brake.
    rows = approach(cls="traffic_light", state="green", hold_s=1.0)
    for r in rows:
        r["stops"].append({"id": "s2", "cls": "stop_sign", "x": 0.0, "y": r["stops"][0]["y"] + 1.0})
    both = se.score(rows)
    assert both["green_false_brakes"] == 0 and both["false_brakes"] == 0 and both["events"] == 1, both
    print("lights: red/yellow scored, green/unknown never events; green + ctl_brake>0.3 alone = green false brake OK")


def check_stop_metrics_on_fixture() -> None:
    from python.train import stop_events as se

    rows = []
    rows += approach(sid="a", stop_at_y=2.0, hold_s=1.0)  # held, stop distance ~2 m
    rows += approach(sid="b", stop_at_y=None, t0=100.0, i0=2000)  # miss
    rows += approach(sid="c", stop_at_y=0.5, hold_s=0.8, t0=200.0, i0=4000)  # held, ~0.5 m
    # 1 km of open road at 20 m/s with two separate hard brakes (no stop target): 2 false brakes.
    t, i = 400.0, 8000
    for k in range(500):
        brake = 0.5 if (100 <= k < 110 or 300 <= k < 305) else 0.0
        rows.append({"t": round(t, 6), "v": 20.0, "ctl_brake": brake, "split": "holdout", "source": "rec", "i": i, "stops": []})
        t += DT
        i += 1
    res = se.score(rows)
    assert res["events"] == 3 and res["missed"] == 1, res
    assert res["false_brakes"] == 2, res
    km = res["km"]
    assert 1.0 < km < 1.3, km
    assert math.isclose(res["false_brakes_per_km"], 2 / km, rel_tol=1e-9), res
    dist = res["stop_distance_m"]
    assert dist["n"] == 2, dist
    assert 0.0 < dist["median"] < 2.5 and dist["p95"] >= dist["median"], dist
    sd = sorted(e["stop_distance_m"] for e in res["detail"] if e["held"])
    assert abs(sd[0] - 0.5) < 0.6 and abs(sd[1] - 2.0) < 0.6, sd
    print(f"metrics: 3 events, 1 missed, 2 false brakes over {km:.3f} km, stop distance median {dist['median']:.2f} m p95 {dist['p95']:.2f} m OK")


def check_stop_events_split_safe() -> None:
    from python.train import stop_events as se

    rows = approach(split="train", stop_at_y=None) + approach(sid="h", split="holdout", t0=100.0, i0=5000, hold_s=1.0)
    res = se.score(rows)
    assert res["events"] == 1 and res["missed"] == 0, res
    # An approach that straddles the split boundary is not scored at all.
    mixed = approach(sid="m", stop_at_y=None)
    for r in mixed[: len(mixed) // 2]:
        r["split"] = "train"
    assert se.score(mixed)["events"] == 0
    try:
        se.score([{"t": 0.0, "v": 1.0, "stops": []}])
    except se.StopEventError:
        pass
    else:
        raise AssertionError("row without split accepted")
    print("split: train-row approaches and boundary-straddling approaches are not scored; untagged rows refused OK")


def check_stop_labels_train_only() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from test_occ_runtime_imports import LIVE_ENTRY_POINTS, chain, reachable

    seen = reachable(LIVE_ENTRY_POINTS)
    bad = [chain(seen, m) for m in ("python.train.stop_events",) if m in seen]
    assert not bad, bad
    tree = ast.parse((ROOT / "python" / "train" / "stop_events.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not {m for m in mods if m and (m.startswith("python.") or m in {"torch", "beamngpy"})}, mods
    print("train-only: stop_events unreachable from the live entry points and imports no python.* / torch / beamngpy OK")


def check_hand_rule_untouched() -> None:
    for rel, want in HAND_RULE_PINS.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == want, f"{rel} changed (sha256 {got}); Trainer 2.0 must not touch the hand rule / AEB"
    print(f"hand rule: {len(HAND_RULE_PINS)} files match their main 8ce34d5 sha256 OK")


def _drop_target(rows, *, y_lo: float, y_hi: float) -> int:
    """Blank the GT target on rows with y_lo <= y < y_hi (a dropped label). Returns how many rows."""
    n = 0
    for r in rows:
        if r["stops"] and y_lo <= r["stops"][0]["y"] < y_hi:
            r["stops"] = []
            n += 1
    return n


def check_missing_gt_row_does_not_end_the_window() -> None:
    """A dropped label is skipped, not read as the end of the approach (probe P2b: a held stop scored a miss)."""
    from python.train import stop_events as se

    base = approach(hold_s=1.5)
    assert se.score(base)["missed"] == 0
    # One missing GT row mid-approach (y ~ 30 m), as in the probe.
    one = approach(hold_s=1.5)
    assert _drop_target(one, y_lo=29.0, y_hi=30.0) == 1
    res = se.score(one)
    assert res["events"] == 1 and res["missed"] == 0, res
    # Missing rows inside the hold itself (the car sits at y = 2 m, 0 m/s).
    inside = approach(hold_s=1.5)
    n = sum(1 for r in inside if r["v"] == 0.0)
    for r in [r for r in inside if r["v"] == 0.0][2:5]:
        r["stops"] = []
    assert n >= 10
    res = se.score(inside)
    assert res["events"] == 1 and res["missed"] == 0, res
    assert math.isclose(res["detail"][0]["stop_distance_m"], 2.0, abs_tol=0.01), res["detail"]
    # A skipped row that is not a holdout row still keeps the approach out of the score.
    mixed = approach(hold_s=1.5)
    _drop_target(mixed, y_lo=29.0, y_hi=30.0)
    for r in mixed:
        if not r["stops"]:
            r["split"] = "train"
    dropped = [r for r in mixed if not r["stops"]]
    assert len(dropped) == 1 and dropped[0]["split"] == "train"
    assert se.score(mixed)["events"] == 0, "a train row inside the window must keep the approach unscored"
    # A dropout longer than MAX_DT_S is a lost target: a later reappearance does not rescue a run-through.
    lost = approach(stop_at_y=None)  # drives through at 10 m/s, never stops
    last_t = lost[-1]["t"]
    for r in lost:
        if r["stops"][0]["y"] < 8.0:
            r["stops"] = []
    tail = approach(stop_at_y=2.0, hold_s=1.0, t0=last_t + 5.0, i0=9000)  # same id "s1", stopped properly 5 s later
    res = se.score(lost + tail)
    assert res["events"] == 1 and res["missed"] == 1, res
    print("missing GT rows: one-row gap, gap inside the hold, train row in a gap, >1 s dropout all handled; held stop stays held OK")


def check_speed_key_scores_the_predicted_profile() -> None:
    """Contract: ``v`` fixes which approaches exist and km; ``speed_key`` decides whether the stop was held."""
    from python.train import stop_events as se

    # Expert stops, prediction never slows: default scores the expert, speed_key scores the prediction.
    rows = approach(hold_s=1.0)
    for r in rows:
        r["pred_v"] = 10.0
    assert se.score(rows)["missed"] == 0
    assert se.score(rows, speed_key="v")["missed"] == 0
    pred = se.score(rows, speed_key="pred_v")
    assert pred["events"] == 1 and pred["missed"] == 1, pred
    # Expert drives through at 10 m/s; the prediction is below 0.5 m/s for 1.2 s from y ~ 20 m. It is judged
    # open-loop at the recorded positions, so the stop distance is the recorded y where the hold starts.
    rows = approach(stop_at_y=None)
    for r in rows:
        r["pred_v"] = r["v"]
    held_idx = [k for k, r in enumerate(rows) if r["stops"][0]["y"] <= 20.0][:12]
    assert len(held_idx) == 12
    for k in held_idx:
        rows[k]["pred_v"] = 0.0
    assert se.score(rows)["missed"] == 1
    pred = se.score(rows, speed_key="pred_v")
    assert pred["events"] == 1 and pred["missed"] == 0, pred
    assert 18.5 <= pred["detail"][0]["stop_distance_m"] <= 20.0, pred["detail"]
    # The entry gate and km come from v: a slow recorded entry is never an event, whatever pred_v says.
    slow = approach(v0=1.5, hold_s=1.0)
    for r in slow:
        r["pred_v"] = 0.0
    assert se.score(slow, speed_key="pred_v")["events"] == 0
    # ... and a recorded entry >= 2 m/s is an event even when the predicted profile is already stopped there.
    parked = approach(hold_s=1.0)
    for r in parked:
        r["pred_v"] = 0.0
    res = se.score(parked, speed_key="pred_v")
    assert res["events"] == 1 and res["missed"] == 0, res
    assert se.score(rows)["km"] == se.score(rows, speed_key="pred_v")["km"] > 0.0
    # km is the recorded v integrated over the holdout rows (the 12 pred_v = 0 rows must still count at v).
    want_km = sum(max(0.0, float(b["v"])) * (float(b["t"]) - float(a["t"])) for a, b in zip(rows, rows[1:])) / 1000.0
    assert math.isclose(se.score(rows)["km"], want_km, rel_tol=1e-9), (se.score(rows)["km"], want_km)
    assert math.isclose(se.score(rows, speed_key="pred_v")["km"], want_km, rel_tol=1e-9)
    # No fallback to v: a missing, non-numeric, NaN or negative speed_key value is refused.
    for bad in (None, "fast", float("nan"), float("inf"), -1.0, "missing"):
        rows2 = approach(hold_s=1.0)
        for r in rows2:
            r["pred_v"] = 0.0
        if bad == "missing":
            del rows2[3]["pred_v"]
        else:
            rows2[3]["pred_v"] = bad
        try:
            se.score(rows2, speed_key="pred_v")
        except se.StopEventError:
            pass
        else:
            raise AssertionError(f"speed_key value {bad!r} accepted")
    try:
        se.score(approach(hold_s=1.0), speed_key="pred_v")
    except se.StopEventError:
        pass
    else:
        raise AssertionError("rows without the speed_key field fell back to v")
    for key, bad in (("v", float("nan")), ("t", float("nan")), ("v", -0.1)):
        rows3 = approach(hold_s=1.0)
        rows3[4][key] = bad
        try:
            se.score(rows3)
        except se.StopEventError:
            continue
        raise AssertionError(f"row {key}={bad!r} accepted")
    print("speed_key: predicted profile decides the hold, v decides events and km; missing / NaN / negative refused OK")


def _set_x(rows, fn) -> None:
    for r in rows:
        r["stops"][0]["x"] = fn(r["stops"][0]["y"])


def check_side_angle_uses_first_visibility_not_entry() -> None:
    from python.train import stop_events as se

    t30 = math.tan(math.radians(30.0))
    # First visible at y ~ 60 m, 30 deg off axis (in 22-37.5); by the 40 m entry it is ~7 deg: still the side slice.
    far_side = approach(y0=60.0, stop_at_y=None)
    _set_x(far_side, lambda y: y * t30 if y > 52.0 else 5.0)
    res = se.score(far_side)
    assert res["events"] == 1, res
    assert 29.0 < res["detail"][0]["first_bearing_deg"] < 31.0, res["detail"]
    assert res["slices"]["stop_side_angle"] == {"events": 1, "missed": 1}, res["slices"]
    # First visible dead ahead, 30 deg off axis only at entry: not the side slice.
    late_side = approach(y0=60.0, stop_at_y=None)
    _set_x(late_side, lambda y: 0.5 if y > 45.0 else y * t30)
    res = se.score(late_side)
    assert res["events"] == 1 and res["detail"][0]["first_bearing_deg"] < 5.0, res["detail"]
    assert res["slices"]["stop_side_angle"] == {"events": 0, "missed": 0}, res["slices"]
    print("side angle: bearing is read at first GT visibility (60 m), not at the 40 m window entry OK")


def check_km_counts_holdout_rows_only() -> None:
    from python.train import stop_events as se

    rows = []
    for k in range(1000):  # 100 s of train driving at 20 m/s = 2 km that must not count
        rows.append({"t": round(k * DT, 6), "v": 20.0, "ctl_brake": 0.0, "split": "train", "source": "rec", "i": k, "stops": []})
    for k in range(100):  # 10 s of holdout at 10 m/s = 100 m
        rows.append({"t": round(100.0 + k * DT, 6), "v": 10.0, "ctl_brake": 0.0, "split": "holdout", "source": "rec", "i": 1000 + k, "stops": []})
    res = se.score(rows)
    assert math.isclose(res["km"], 0.1, rel_tol=1e-9), res["km"]
    train_only = se.score(rows[:1000])
    assert train_only["km"] == 0.0, train_only["km"]
    print(f"km: {res['km']:.3f} km from 100 holdout rows; 2 km of train rows not counted OK")


def check_no_nan_in_scores() -> None:
    """Contract with trainer2_bar: a rate with nothing to divide by is None, never NaN or 0.0."""
    import json

    from python.train import stop_events as se

    empty = se.score(approach(split="train", stop_at_y=None))
    assert empty["km"] == 0.0
    assert empty["false_brakes_per_km"] is None and empty["green_false_brakes_per_km"] is None, empty
    assert empty["stop_distance_m"] == {"n": 0, "median": None, "p95": None}, empty["stop_distance_m"]
    json.dumps(empty, allow_nan=False)
    assert se.score([])["false_brakes_per_km"] is None
    run = se.score(approach(stop_at_y=None))  # holdout km > 0 but no held stop
    assert run["km"] > 0.0 and isinstance(run["false_brakes_per_km"], float)
    assert run["stop_distance_m"]["median"] is None and run["stop_distance_m"]["p95"] is None
    json.dumps(run, allow_nan=False)
    ok = se.score(approach(hold_s=1.0))
    assert isinstance(ok["stop_distance_m"]["median"], float) and isinstance(ok["stop_distance_m"]["p95"], float)
    json.dumps(ok, allow_nan=False)
    print("no NaN: km == 0 and no held stop give None (JSON-safe), real values stay floats OK")


def check_red_to_green_proceed_current_behaviour() -> None:
    """PROPOSAL pending Michael: this pins today's behaviour, it does not decide the question.

    Red at first entry, green before the car must stop, car drives on through the line:
    scored as a MISSED red-light stop (the event is typed by the lamp at entry; the literal spec).
    If Michael rules this should not count, this test and ``RED_TO_GREEN_PROCEED_IS_MISS`` change together.
    """
    from python.train import stop_events as se

    assert se.RED_TO_GREEN_PROCEED_IS_MISS is True  # PROPOSAL pending Michael
    rows = approach(cls="traffic_light", state="red", stop_at_y=None, v0=6.0, y0=45.0, brake=0.0)
    for r in rows:
        if r["stops"][0]["y"] <= 15.0:
            r["stops"][0]["state"] = "green"
    assert rows[0]["stops"][0]["state"] == "red" and any(r["stops"][0]["state"] == "green" for r in rows)
    res = se.score(rows)
    assert res["events"] == 1 and res["missed"] == 1, res
    assert res["slices"]["red_light"] == {"events": 1, "missed": 1}, res["slices"]
    assert res["detail"][0]["state"] == "red"
    print("red->green then proceed: counted as a missed red light today (PROPOSAL pending Michael; pinned) OK")


def _lamp_turns(rows, *, at_or_below):
    """Switch the lamp to green on rows with y <= ``at_or_below`` (copy of the rows' targets)."""
    for r in rows:
        if r["stops"][0]["y"] <= at_or_below:
            r["stops"][0]["state"] = "green"
    return rows


def check_red_to_green_flag_false_drops_the_approach() -> None:
    """PROPOSAL pending Michael: the other setting of RED_TO_GREEN_PROCEED_IS_MISS (default stays True)."""
    from python.train import stop_events as se

    def light(state, **kw):
        return approach(cls="traffic_light", state=state, stop_at_y=kw.pop("stop_at_y", None), v0=6.0, y0=45.0, brake=0.0, **kw)

    def run(rows, flag):
        saved = se.RED_TO_GREEN_PROCEED_IS_MISS
        se.RED_TO_GREEN_PROCEED_IS_MISS = flag
        try:
            return se.score(rows)
        finally:
            se.RED_TO_GREEN_PROCEED_IS_MISS = saved

    assert se.RED_TO_GREEN_PROCEED_IS_MISS is True, "default must stay True while the question is open"
    for lamp in ("red", "yellow"):
        rows = _lamp_turns(light(lamp), at_or_below=15.0)
        assert rows[0]["stops"][0]["state"] == lamp
        on, off = run(rows, True), run(rows, False)
        assert on["events"] == 1 and on["missed"] == 1, (lamp, on)
        assert off["events"] == 0 and off["missed"] == 0 and off["detail"] == [], (lamp, off)
        assert se.RED_TO_GREEN_PROCEED_IS_MISS is True, "the flag was not restored"
    # Still a miss with the flag False: the lamp never changes, or turns green only past the line (y <= 0).
    steady = run(light("red"), False)
    assert steady["events"] == 1 and steady["missed"] == 1, steady
    late = run(_lamp_turns(light("red"), at_or_below=0.0), False)
    assert late["events"] == 1 and late["missed"] == 1, late
    # A held stop is a held stop, whether or not the lamp turned green meanwhile.
    held = _lamp_turns(light("red", stop_at_y=2.0, hold_s=1.0), at_or_below=5.0)
    for flag in (True, False):
        res = run(held, flag)
        assert res["events"] == 1 and res["missed"] == 0, (flag, res)
    # A stop sign has no lamp: the flag changes nothing.
    sign = approach(stop_at_y=None)
    assert run(sign, False)["missed"] == run(sign, True)["missed"] == 1
    print("red->green flag: False drops a red/yellow approach released before the line; late green, steady red, held stop and signs unchanged OK")


def main() -> int:
    checks = (
        check_stop_event_requires_hold,
        check_stop_side_angle_slice,
        check_light_events_red_yellow_only,
        check_stop_metrics_on_fixture,
        check_stop_events_split_safe,
        check_stop_labels_train_only,
        check_hand_rule_untouched,
        check_missing_gt_row_does_not_end_the_window,
        check_speed_key_scores_the_predicted_profile,
        check_side_angle_uses_first_visibility_not_entry,
        check_km_counts_holdout_rows_only,
        check_no_nan_in_scores,
        check_red_to_green_proceed_current_behaviour,
        check_red_to_green_flag_false_drops_the_approach,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
