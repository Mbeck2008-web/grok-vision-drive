#!/usr/bin/env python3
"""Trainer 2.0 stage 2b (docs/specs/trainer_2.md, 2b / PR 6b): stop ids, the predicted speed profile, the 2b metrics.

Torch-free checks of ``python/train/stop_pred.py`` and the sign ids; the recipe wiring and the real training run
are in ``test_trainer2_recipe.py`` / ``test_trainer2_recipe_cli.py``. Every number below is worked out by hand in
the comments (nothing is read back from the code under test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

DT = 0.1


def _ctl(thr: float, brk: float):
    """ctl (3, K=6): steer, throttle, brake; only column 0 (h = 0) is read."""
    out = np.zeros((3, 6))
    out[1, :], out[2, :] = thr, brk
    return out


def _approach(source, *, n, v, brake, thr=0.0, target=None, y0=40.0, state=None, sid="stop_sign@1.0,2.0"):
    """``n`` rows at 10 Hz. ``v(k)``: recorded speed; ``brake(k)``: predicted brake; the target is ``y0`` m ahead at k=0
    and closes at the recorded speed."""
    pairs, y = [], y0
    for k in range(n):
        signs = []
        if target is not None:
            signs = [{"id": sid, "cls": target, "x": 0.5, "y": y, "state": state}]
        row = {"i": k, "ego": {"sim_t": 100.0 + k * DT, "speed_mps": v(k)}, "signs": signs, "source": source}
        pairs.append((row, _ctl(thr if not callable(thr) else thr(k), brake(k))))
        y -= v(k) * DT
    return pairs


def check_sign_ids_are_stable() -> None:
    from python.collect.signal_labels import labels_from_signals, sign_id

    raw = [{"cls": "stop_sign", "pos": [10.0, 50.0, 0.0]}, {"cls": "traffic_light", "pos": [10.04, 50.0, 0.0], "state": "red"}]
    a = labels_from_signals(raw, [0.0, 0.0], [0.0, 1.0])
    b = labels_from_signals(raw, [0.0, 20.0], [0.0, 1.0])  # the car moved 20 m: ego-frame y changes, the id must not
    ids_a = {s["cls"]: s["id"] for s in a}
    ids_b = {s["cls"]: s["id"] for s in b}
    assert ids_a == ids_b == {"stop_sign": "stop_sign@10.0,50.0", "traffic_light": "traffic_light@10.0,50.0"}, (ids_a, ids_b)
    assert [s["y"] for s in a] != [s["y"] for s in b]
    assert sign_id("yield", (-3.04, 7.96)) == "yield@-3.0,8.0"
    other = labels_from_signals([{"cls": "stop_sign", "pos": [10.0, 80.0, 0.0]}], [0.0, 0.0], [0.0, 1.0])
    assert other[0]["id"] != ids_a["stop_sign"]  # a different sign is a different event
    print("sign ids: world-position based, identical across ticks, distinct per sign OK")


def check_stop_rows_inputs() -> None:
    from python.train import stop_pred

    pairs = _approach("A", n=3, v=lambda k: 5.0, brake=lambda k: 0.2, target="stop_sign")
    pairs[1][0]["signs"].append({"cls": "stop_sign", "x": 0.0, "y": 9.0})  # no id: not a trackable event
    pairs[2][0]["signs"].append({"id": "q", "cls": "stop_sign", "x": float("nan"), "y": 9.0})  # no pose
    rows, skipped = stop_pred.stop_rows(reversed(pairs))
    assert [r["t"] for r in rows] == [100.0, 100.1, 100.2], rows  # time order, whatever order the pass came in
    assert skipped == {"no_id": 1, "no_pose": 1, "rows": 3}, skipped
    assert all(r["split"] == "holdout" and r["v"] == 5.0 and r["ctl_brake"] == 0.2 for r in rows)
    assert [len(r["stops"]) for r in rows] == [1, 1, 1]
    # a row that cannot be placed in time is refused, never dropped or given a default
    for bad in ({"i": 0, "ego": {"speed_mps": 3.0}}, {"i": 0, "ego": {"sim_t": 1.0}}, {"i": 0}, {"i": 0, "ego": {"sim_t": 1.0, "speed_mps": -1.0}}):
        try:
            stop_pred.stop_rows([(bad, _ctl(0, 0))])
        except stop_pred.StopNotMeasured:
            pass
        else:
            raise AssertionError(f"row {bad} must be refused")
    try:
        stop_pred.stop_rows([(pairs[0][0], np.full((3, 6), np.nan))])
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("a non-finite ctl prediction must be refused")
    # ctl is clamped to its range; the sim time and speed come from `ego` (not the wall-clock t)
    r, _ = stop_pred.stop_rows([({"i": 0, "t": 5.0, "ego": {"sim_t": 7.0, "speed_mps": 3.0}}, _ctl(1.5, -0.2))])
    assert r[0]["t"] == 7.0 and r[0]["ctl_throttle"] == 1.0 and r[0]["ctl_brake"] == 0.0 and r[0]["source"] == "", r
    # signs in tech (the sidecar join writes there when the row has a tech dict)
    r, _ = stop_pred.stop_rows([({"i": 0, "ego": {"sim_t": 1.0, "speed_mps": 3.0},
                                  "tech": {"signs": [{"id": "z", "cls": "traffic_light", "x": 1.0, "y": 2.0, "state": "red"}]}}, _ctl(0, 0))])
    assert r[0]["stops"] == [{"id": "z", "cls": "traffic_light", "x": 1.0, "y": 2.0, "state": "red"}], r
    print("stop rows: sim time + recorded speed + ctl h=0, ids required, time ordered, bad rows refused OK")


def check_predicted_speed_profile() -> None:
    from python.train import stop_pred

    # recorded 10 m/s, sign 40 m ahead. The model brakes 0.5 (decel 4.5 * 0.5 = 2.25 m/s^2): v_k = 10 - 0.225 k
    pairs = _approach("A", n=20, v=lambda k: 10.0, brake=lambda k: 0.5, target="stop_sign")
    rows, _ = stop_pred.stop_rows(pairs)
    stop_pred.add_predicted_speed(rows)
    for k in (0, 1, 5, 19):
        assert abs(rows[k]["pred_v"] - (10.0 - 0.225 * k)) < 1e-9, (k, rows[k]["pred_v"])
    # throttle adds 2.0 m/s^2 at full throttle: brake .5 + throttle .5 -> (1.0 - 2.25) = -1.25 m/s^2 -> 10 - 0.125 k
    rows, _ = stop_pred.stop_rows(_approach("A", n=20, v=lambda k: 10.0, brake=lambda k: 0.5, thr=0.5, target="stop_sign"))
    stop_pred.add_predicted_speed(rows)
    assert abs(rows[8]["pred_v"] - (10.0 - 0.125 * 8)) < 1e-9, rows[8]["pred_v"]
    # never below 0: brake 1 from 10 m/s stops in 10/4.5 = 2.22 s and stays at 0
    rows, _ = stop_pred.stop_rows(_approach("A", n=40, v=lambda k: 10.0, brake=lambda k: 1.0, target="stop_sign"))
    stop_pred.add_predicted_speed(rows)
    assert rows[23]["pred_v"] == 0.0 and rows[39]["pred_v"] == 0.0 and rows[22]["pred_v"] > 0.0, [r["pred_v"] for r in rows[20:25]]
    # the recorded stop never credits the model: the car stops at k=10 (recorded v 0 from there) and the model does not brake
    pairs = _approach("A", n=30, v=lambda k: 10.0 if k < 10 else 0.0, brake=lambda k: 0.0, target="stop_sign")
    rows, _ = stop_pred.stop_rows(pairs)
    stop_pred.add_predicted_speed(rows)
    assert rows[0]["pred_v"] == 10.0 and rows[9]["pred_v"] == 10.0 and rows[12]["pred_v"] == 10.0, [r["pred_v"] for r in rows[:14]]
    # outside any approach pred_v is the recorded speed
    pairs = _approach("A", n=10, v=lambda k: 4.0 + k, brake=lambda k: 1.0, target=None)
    rows, _ = stop_pred.stop_rows(pairs)
    stop_pred.add_predicted_speed(rows)
    assert [r["pred_v"] for r in rows] == [r["v"] for r in rows]
    print("predicted speed: dv/dt = 2.0 throttle - 4.5 brake from the entry speed, floored at 0, not re-anchored to the recorded stop OK")


def _metrics(sources):
    from python.train import stop_pred

    pairs = []
    for p in sources:
        pairs.extend(p)
    return stop_pred.measure(pairs)


def check_measure_known_counts() -> None:
    """Hand-built holdout: 2 stop-sign events (1 held, 1 missed), 2 lights (red missed, yellow held), a green light,
    and a long cruise with two false-brake onsets. The numbers are the ones the spec's definitions give."""
    cruise = lambda k: 10.0  # noqa: E731
    sources = [
        # A: stop sign, the model brakes fully from the start -> stops at k=23 (pred_v 0 from 2.3 s), held to the end
        _approach("A", n=45, v=cruise, brake=lambda k: 1.0, target="stop_sign", sid="stop_sign@A"),
        # B: stop sign, the model never brakes -> missed
        _approach("B", n=45, v=cruise, brake=lambda k: 0.0, target="stop_sign", sid="stop_sign@B"),
        # C: red light, the model brakes 0.1 (0.45 m/s^2 -> -2 m/s in 4.5 s): never stops -> missed light
        _approach("C", n=45, v=cruise, brake=lambda k: 0.1, target="traffic_light", state="red", sid="light@C"),
        # D: yellow light, full brake -> held
        _approach("D", n=45, v=cruise, brake=lambda k: 1.0, target="traffic_light", state="yellow", sid="light@D"),
        # E: a green light is not a stop event; braking on it with nothing else is a green-light false brake
        _approach("E", n=45, v=cruise, brake=lambda k: 1.0 if 10 <= k < 14 else 0.0, target="traffic_light", state="green", sid="light@E"),
        # F: no signs at all, 200 rows at 10 m/s (200 m) with two brake onsets (> 0.3)
        _approach("F", n=200, v=cruise, brake=lambda k: 0.8 if (30 <= k < 35 or 120 <= k < 125) else 0.0, target=None),
    ]
    m = _metrics(sources)
    assert (m["stop_events"], m["light_events"]) == (2, 2), m
    assert m["missed_stop"] == 1.0 and m["missed_light"] == 1.0, m
    # km: 5 sources x 44 intervals x 1 m + F 199 m. Rows are 10 Hz at 10 m/s -> 1 m per interval
    assert abs(m["stop_km"] - (4 * 44 + 44 + 199) / 1000.0) < 1e-9, m["stop_km"]
    # false brakes: the 2 onsets in F, 1 on the green light in E (green + no other event: counted as green too)
    assert abs(m["false_brake"] - 3 / m["stop_km"]) < 1e-9, m
    assert abs(m["green_false_brake"] - 1 / m["stop_km"]) < 1e-9, m
    assert m["stop_distance_m"]["n"] == 2, m["stop_distance_m"]  # A and D held
    assert m["stop_side_angle_events"] == 0
    # braking late enough is not a stop: the model brakes 1.0 only after 3 s (k >= 30): v = 10 - 4.5*(k-30)*.1 -> 0 at k=52; the
    # recorded car passes the sign line+1 m (y < -1) at k = 41 -> the window has ended before the stop: missed
    late = _approach("A", n=70, v=cruise, brake=lambda k: 1.0 if k >= 30 else 0.0, target="stop_sign", sid="stop_sign@A")
    m = _metrics([late, _approach("F", n=50, v=cruise, brake=lambda k: 0.0, target=None)])
    assert m["stop_events"] == 1 and m["missed_stop"] == 1.0, m
    # the hold: full brake until the car is at 0, then the model releases and gives throttle -> no 0.5 s hold -> missed
    rel = _approach("A", n=45, v=cruise, brake=lambda k: 1.0 if k < 23 else 0.0, thr=lambda k: 0.0 if k < 23 else 1.0, target="stop_sign")
    m = _metrics([rel, _approach("F", n=50, v=cruise, brake=lambda k: 0.0, target=None)])
    assert m["missed_stop"] == 1.0, m
    print("measure: 2 stop + 2 light events -> 1 + 1 missed; false brakes / km, green false brakes, hold and late brake OK")


def check_side_angle_slice_and_empty_metrics() -> None:
    from python.train import stop_pred

    # a 30 deg sign: x = y * tan(30 deg) at the first visible tick -> in the 22-37.5 deg slice
    pairs = _approach("A", n=45, v=lambda k: 10.0, brake=lambda k: 1.0, target="stop_sign", y0=30.0)
    for row, _ctl_ in pairs:
        for s in row["signs"]:
            s["x"] = s["y"] * np.tan(np.radians(30.0))
    pairs += _approach("F", n=50, v=lambda k: 10.0, brake=lambda k: 0.0, target=None)
    m = stop_pred.measure(pairs)
    assert m["stop_side_angle_events"] == 1 and m["stop_side_angle_missed"] == 0 and m["missed_stop"] == 0.0, m
    # no light events -> missed_light is None (not 0): the stage cannot claim a pass it did not measure
    assert m["light_events"] == 0 and m["missed_light"] is None, m
    # nothing at all
    m = stop_pred.measure(_approach("F", n=50, v=lambda k: 10.0, brake=lambda k: 0.0, target=None))
    assert m["missed_stop"] is None and m["missed_light"] is None and m["false_brake"] == 0.0, m
    # a recording that never moves has no km: the false-brake rate is None, never 0 or NaN
    m = stop_pred.measure(_approach("F", n=10, v=lambda k: 0.0, brake=lambda k: 0.0, target=None))
    assert m["false_brake"] is None, m
    try:
        stop_pred.measure([])
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("no rows must be refused")
    ev = stop_pred.recorded_events(_approach("A", n=45, v=lambda k: 10.0, brake=lambda k: 0.0, target="traffic_light", state="red") + _approach("B", n=45, v=lambda k: 10.0, brake=lambda k: 0.0, target="stop_sign", sid="s@B"))
    assert ev == {"stop": 1, "light": 1, "rows": 90}, ev
    print("measure: side-angle slice reported on its own; a class with no events is None; no distance -> None OK")


def main() -> int:
    checks = (check_sign_ids_are_stable, check_stop_rows_inputs, check_predicted_speed_profile, check_measure_known_counts,
              check_side_angle_slice_and_empty_metrics)
    failed = 0
    for check in checks:
        try:
            check()
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"FAIL {check.__name__}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
