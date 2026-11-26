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
K_THR, K_BRK, K_DRAG = 2.0, 4.5, 0.0


def _motion(k_thr=K_THR, k_brk=K_BRK, k_drag=K_DRAG):
    from python.train.stop_pred import Motion

    return Motion(k_thr, k_brk, k_drag)


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
    stop_pred.add_predicted_speed(rows, _motion())
    for k in (0, 1, 5, 19):
        assert abs(rows[k]["pred_v"] - (10.0 - 0.225 * k)) < 1e-9, (k, rows[k]["pred_v"])
    # throttle adds 2.0 m/s^2 at full throttle: brake .5 + throttle .5 -> (1.0 - 2.25) = -1.25 m/s^2 -> 10 - 0.125 k
    rows, _ = stop_pred.stop_rows(_approach("A", n=20, v=lambda k: 10.0, brake=lambda k: 0.5, thr=0.5, target="stop_sign"))
    stop_pred.add_predicted_speed(rows, _motion())
    assert abs(rows[8]["pred_v"] - (10.0 - 0.125 * 8)) < 1e-9, rows[8]["pred_v"]
    # never below 0: brake 1 from 10 m/s stops in 10/4.5 = 2.22 s and stays at 0
    rows, _ = stop_pred.stop_rows(_approach("A", n=40, v=lambda k: 10.0, brake=lambda k: 1.0, target="stop_sign"))
    stop_pred.add_predicted_speed(rows, _motion())
    assert rows[23]["pred_v"] == 0.0 and rows[39]["pred_v"] == 0.0 and rows[22]["pred_v"] > 0.0, [r["pred_v"] for r in rows[20:25]]
    # the recorded stop never credits the model: the car stops at k=10 (recorded v 0 from there) and the model does not brake
    pairs = _approach("A", n=30, v=lambda k: 10.0 if k < 10 else 0.0, brake=lambda k: 0.0, target="stop_sign")
    rows, _ = stop_pred.stop_rows(pairs)
    stop_pred.add_predicted_speed(rows, _motion())
    assert rows[0]["pred_v"] == 10.0 and rows[9]["pred_v"] == 10.0 and rows[12]["pred_v"] == 10.0, [r["pred_v"] for r in rows[:14]]
    # outside any approach there is no profile: a speed that is not a stop, never the recorded speed
    pairs = _approach("A", n=10, v=lambda k: 4.0 + k, brake=lambda k: 1.0, target=None)
    rows, _ = stop_pred.stop_rows(pairs)
    stop_pred.add_predicted_speed(rows, _motion())
    assert [r["pred_v"] for r in rows] == [stop_pred.NOT_SCORED_V] * 10, [r["pred_v"] for r in rows]  # never the recorded speed
    print("predicted speed: dv/dt = 2.0 throttle - 4.5 brake from the entry speed, floored at 0, not re-anchored to the recorded stop OK")


def _metrics(sources):
    from python.train import stop_pred

    pairs = []
    for p in sources:
        pairs.extend(p)
    return stop_pred.measure(pairs, _motion())


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
    m = stop_pred.measure(pairs, _motion())
    assert m["stop_side_angle_events"] == 1 and m["stop_side_angle_missed"] == 0 and m["missed_stop"] == 0.0, m
    # no light events -> missed_light is None (not 0): the stage cannot claim a pass it did not measure
    assert m["light_events"] == 0 and m["missed_light"] is None, m
    # nothing at all
    m = stop_pred.measure(_approach("F", n=50, v=lambda k: 10.0, brake=lambda k: 0.0, target=None), _motion())
    assert m["missed_stop"] is None and m["missed_light"] is None and m["false_brake"] == 0.0, m
    # a recording that never moves has no km: the false-brake rate is None, never 0 or NaN
    m = stop_pred.measure(_approach("F", n=10, v=lambda k: 0.0, brake=lambda k: 0.0, target=None), _motion())
    assert m["false_brake"] is None, m
    try:
        stop_pred.measure([], _motion())
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("no rows must be refused")
    ev = stop_pred.recorded_events(_approach("A", n=45, v=lambda k: 10.0, brake=lambda k: 0.0, target="traffic_light", state="red") + _approach("B", n=45, v=lambda k: 10.0, brake=lambda k: 0.0, target="stop_sign", sid="s@B"))
    assert ev == {"stop": 1, "light": 1, "rows": 90}, ev
    print("measure: side-angle slice reported on its own; a class with no events is None; no distance -> None OK")


def _train_rows(k_thr=1.5, k_brk=5.0, k_drag=0.05, n=400, seg="s", start_i=0, v0=14.0):
    """Recorded train rows from a known car: dv/dt = k_thr*thr - k_brk*brk - k_drag*v, controls vary, v stays > 0."""
    rows, v = [], v0
    for k in range(n):
        thr = 0.5 + 0.5 * np.sin(k / 7.0) if (k % 40) >= 8 else 0.0
        brk = 0.6 if (k % 40) < 8 and k % 40 >= 3 else 0.0
        rows.append({"i": start_i + k, "segment": seg, "ego": {"sim_t": 50.0 + k * DT, "speed_mps": v},
                     "pedals": {"throttle_ai": float(thr), "brake_ai": float(brk), "parkingbrake_ai": 0.0}, "veh": {"reverse": 0}})
        v = max(0.0, v + (k_thr * thr - k_brk * brk - k_drag * v) * DT)
    return rows


def check_fit_motion_train_rows_only() -> None:
    from python.train import stop_pred

    rows = _train_rows()
    assert min(r["ego"]["speed_mps"] for r in rows) > 0.5, "the fixture must not floor"
    m = stop_pred.fit_motion(rows, holdout_ids={9999, 10000})
    assert abs(m.k_thr - 1.5) < 1e-6 and abs(m.k_brk - 5.0) < 1e-6 and abs(m.k_drag - 0.05) < 1e-6, m.as_dict()
    assert m.n_rows == len(rows) - 1 and m.n_brake_rows > 10, m.as_dict()
    assert set(m.ids) == {str(r["i"]) for r in rows}, "the fit records which rows it saw"
    # a holdout id in the fit set is refused, naming it (not dropped): the fit never sees holdout rows
    try:
        stop_pred.fit_motion(rows, holdout_ids={17})
    except stop_pred.StopNotMeasured as exc:
        assert "holdout row id" in str(exc) and "17" in str(exc), exc
    else:
        raise AssertionError("a holdout id in the fit set must be refused")
    # ids compare as text: 17 and "17" are the same row
    try:
        stop_pred.fit_motion(rows, holdout_ids={"17"})
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("string holdout ids must match int row ids")
    # the fit uses only the rows it is given: dropping the rows after i=200 changes nothing about the rows before
    half = stop_pred.fit_motion(rows[:200], holdout_ids=set())
    assert abs(half.k_brk - 5.0) < 1e-6
    # no braking in the train rows -> no brake map: refused, not defaulted
    flat = [dict(r, pedals=dict(r["pedals"], brake_ai=0.0)) for r in rows]
    for thing, why in ((flat, "braking"), (rows[:10], "needs")):
        try:
            stop_pred.fit_motion(thing, holdout_ids=set())
        except stop_pred.StopNotMeasured as exc:
            assert why in str(exc), exc
        else:
            raise AssertionError("an unfittable train set must be refused")
    # braking that does not slow the car (brake_ai pressed, speed constant) -> k_brk 0 -> refused
    stuck = [dict(r, ego={"sim_t": r["ego"]["sim_t"], "speed_mps": 10.0}) for r in rows]
    try:
        stop_pred.fit_motion(stuck, holdout_ids=set())
    except stop_pred.StopNotMeasured as exc:
        assert "no deceleration" in str(exc), exc
    else:
        raise AssertionError("no deceleration under braking must be refused")
    # a hole, a segment change, reverse and the parking brake do not make accelerations
    holed = [dict(r) for r in rows]
    for k in range(60, 120):
        holed[k] = dict(holed[k], segment=f"seg{k}")  # every pair in 60..120 crosses a segment
    assert stop_pred.fit_motion(holed, holdout_ids=set()).n_rows < len(rows) - 60
    rev = [dict(r, veh={"reverse": 1}) for r in rows]
    try:
        stop_pred.fit_motion(rev, holdout_ids=set())
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("reverse rows are not fit rows")
    pb = [dict(r, pedals=dict(r["pedals"], parkingbrake_ai=0.9)) for r in rows]
    try:
        stop_pred.fit_motion(pb, holdout_ids=set())
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("parking-brake rows are not fit rows")
    # non-negative: a car whose throttle seems to slow it (k_thr = -0.5 in the data) gets k_thr = 0, never a negative gain
    odd = _train_rows(k_thr=-0.5, k_brk=5.0, k_drag=0.0, n=150, v0=80.0)
    assert min(r["ego"]["speed_mps"] for r in odd) > 1.0
    m = stop_pred.fit_motion(odd, holdout_ids=set())
    assert m.k_thr == 0.0 and m.k_brk > 0.0 and m.k_drag >= 0.0, m.as_dict()
    # the remaining coefficients are REFITTED without the clamped column (not just the negative one set to 0)
    xs = [(r["pedals"]["brake_ai"], a["ego"]["speed_mps"], (b["ego"]["speed_mps"] - a["ego"]["speed_mps"]) / DT)
          for a, b in zip(odd, odd[1:]) for r in (a,) if b["ego"]["speed_mps"] > 0.0]
    design = np.asarray([[-b, -v] for b, v, _ in xs])
    ref, *_ = np.linalg.lstsq(design, np.asarray([y for _, _, y in xs]), rcond=None)
    assert ref[0] > 0.0 and ref[1] >= 0.0 and abs(m.k_brk - ref[0]) < 1e-6 and abs(m.k_drag - ref[1]) < 1e-6, (m.as_dict(), ref)
    print("motion fit: recovers k_thr / k_brk / k_drag from train rows, refuses holdout ids, unfittable sets and k_brk <= 0 OK")


def _brake_rows(n, n_brake, *, brake=0.6, dt=DT, v0=60.0, start=0):
    """``n`` rows of a car braking for its first ``n_brake`` rows (map k_brk 5, k_thr 1.5, drag .05), then on the throttle."""
    rows, v = [], v0
    for k in range(n):
        thr, brk = (0.0, brake) if k < n_brake else (0.5 + 0.5 * np.sin(k / 3.0), 0.0)
        rows.append({"i": start + k, "segment": "s", "ego": {"sim_t": 10.0 + k * dt, "speed_mps": v},
                     "pedals": {"throttle_ai": float(thr), "brake_ai": float(brk), "parkingbrake_ai": 0.0}, "veh": {"reverse": 0}})
        v = max(0.0, v + (1.5 * thr - 5.0 * brk - 0.05 * v) * dt)
    return rows


def check_fit_motion_edges() -> None:
    """The fit's gates one by one: pair count, braking-row count / threshold, time step, glitches, motion constructor."""
    from python.train import stop_pred

    # 30 usable pairs and 10 braking rows is the least that fits (31 rows -> 30 pairs); one fewer of either is refused
    ok = stop_pred.fit_motion(_brake_rows(31, 10), holdout_ids=set())
    assert ok.n_rows == 30 and ok.n_brake_rows == 10 and abs(ok.k_brk - 5.0) < 1e-6, ok.as_dict()
    for rows, why in ((_brake_rows(30, 10), "29 / 10"), (_brake_rows(60, 9), "59 / 9")):
        try:
            stop_pred.fit_motion(rows, holdout_ids=set())
        except stop_pred.StopNotMeasured as exc:
            assert why in str(exc), (why, exc)
        else:
            raise AssertionError(f"{why} must be refused")
    # brake_ai at the 0.1 gate: 0.6 counts as braking, a 0.05 touch does not
    assert stop_pred.fit_motion(_brake_rows(60, 10, brake=0.6), holdout_ids=set()).n_brake_rows == 10
    try:
        stop_pred.fit_motion(_brake_rows(60, 40, brake=0.05), holdout_ids=set())
    except stop_pred.StopNotMeasured as exc:
        assert "0 braking" in str(exc) or "/ 0" in str(exc), exc
    else:
        raise AssertionError("a brake touch below the 0.1 gate is not a braking row")
    good = _brake_rows(120, 30)
    base = stop_pred.fit_motion(good, holdout_ids=set())
    # a stretch recorded with the wrong clock (speeds stepped at 0.1 s but stamped 0.5 s or 0.015 s apart) is not a fit row
    for bad_dt in (0.5, 0.015):
        extra = _brake_rows(60, 60, brake=0.2, v0=15.0, start=1000)  # |accel| stays under the glitch cap even at 0.015 s
        for k, r in enumerate(extra):
            r["segment"] = "other"
            r["ego"] = {"sim_t": 500.0 + k * bad_dt, "speed_mps": r["ego"]["speed_mps"]}
        got = stop_pred.fit_motion(good + extra, holdout_ids=set())
        assert got.n_rows == base.n_rows and abs(got.k_brk - base.k_brk) < 1e-9 and abs(got.k_drag - base.k_drag) < 1e-9, (bad_dt, got.as_dict())
    # a speed glitch (a respawn: 8 m/s -> 300 m/s -> 8) is not an acceleration
    glitch = [dict(r) for r in good]
    glitch[50] = dict(glitch[50], ego={"sim_t": glitch[50]["ego"]["sim_t"], "speed_mps": 300.0})
    got = stop_pred.fit_motion(glitch, holdout_ids=set())
    assert abs(got.k_brk - 5.0) < 0.05 and got.n_rows == base.n_rows - 2, got.as_dict()
    # the motion constructor: finite, non-negative, and braking must slow the car
    for args in ((-0.1, 1.0, 0.1), (1.0, 1.0, -0.1), (1.0, 0.0, 0.1), (1.0, -1.0, 0.1), (float("nan"), 1.0, 0.1), (1.0, 1.0, float("inf"))):
        try:
            stop_pred.Motion(*args)
        except stop_pred.StopNotMeasured:
            pass
        else:
            raise AssertionError(f"Motion{args} must be refused")
    assert stop_pred.Motion(0.0, 0.5, 0.0).accel(1.0, 1.0, 10.0) == -0.5
    print("motion fit gates: 30 pairs / 10 braking rows, 0.1 brake gate, dt 0.02-0.25 s, glitch cap, constructor refusals OK")


def _srow(k, t, v, y=None, *, sid="A", cls="stop_sign", state=None, brake=0.0, thr=0.0):
    """One ``stop_events.score`` row."""
    stops = [] if y is None else [{"id": sid, "cls": cls, "x": 0.5, "y": y, "state": state}]
    return {"i": k, "t": t, "v": v, "split": "holdout", "source": "S", "ctl_brake": brake, "ctl_throttle": thr, "stops": stops}


def check_spans_and_profile_edges() -> None:
    """Approach spans (what the profile covers) and the profile's own edges: t0 row, time holes, rows outside a span."""
    from python.train import stop_events as se
    from python.train import stop_pred

    # a target that goes away across a time hole ends the span there; one that flickers (same clock) does not
    rows = [_srow(k, k * 0.1, 8.0, 30 - 0.8 * k) for k in range(11)]
    rows += [_srow(k, 3.0 + (k - 11) * 0.1, 8.0) for k in range(11, 15)]
    rows += [_srow(k, 3.5 + (k - 15) * 0.1, 8.0, 20 - 0.8 * (k - 15)) for k in range(15, 21)]
    assert stop_pred._approach_spans(rows) == [(0, 10)], stop_pred._approach_spans(rows)
    flick = [_srow(k, k * 0.1, 8.0, None if 11 <= k <= 14 else 30 - 0.8 * k) for k in range(21)]
    assert stop_pred._approach_spans(flick) == [(0, 20)], stop_pred._approach_spans(flick)
    # the span ends at the line + 1 m: the target is at y = 2.0, 0.5, -0.8 (still in), -1.2 (out), -3 (out)
    ys = [30.0, 20.0, 10.0, 2.0, 0.5, -0.8, -1.2, -3.0]
    line = [_srow(k, k * 0.1, 8.0, y) for k, y in enumerate(ys)]
    assert stop_pred._approach_spans(line) == [(0, 5)], stop_pred._approach_spans(line)
    # two overlapping approaches are one span (one car, one profile), the longer end wins when one is nested in the other
    nest = []
    for k in range(60):
        r = _srow(k, k * 0.1 + (2.0 if k > 30 else 0.0), 3.0, 39.0 - 0.3 * k)
        if 10 <= k <= 30:
            r["stops"].append({"id": "B", "cls": "stop_sign", "x": 0.5, "y": 35.0 - 0.3 * (k - 10), "state": None})
        nest.append(r)
    assert stop_pred._approach_spans(nest) == [(0, 59)], stop_pred._approach_spans(nest)
    apart = [_srow(k, k * 0.1, 8.0, 30 - 0.8 * k) for k in range(40)]
    apart += [_srow(k, 20.0 + k * 0.1, 8.0, 30 - 0.8 * (k - 40), sid="C") for k in range(40, 80)]
    spans = stop_pred._approach_spans(apart)
    assert len(spans) == 2 and spans[0][0] == 0 and spans[1][0] == 40, spans

    motion = stop_pred.Motion(2.0, 4.0, 0.0)
    # t0 row keeps the recorded speed even when earlier rows (outside the span) are braking; rows outside a span are no stop
    pre = [_srow(k, k * 0.1, 8.0, None, brake=0.9) for k in range(10)]
    app = [_srow(10 + k, 1.0 + k * 0.1, 8.0, 30 - 0.8 * k, brake=0.9) for k in range(11)]
    post = [_srow(21 + k, 2.1 + k * 0.1, 8.0, None, brake=0.9) for k in range(5)]
    allrows = pre + app + post
    stop_pred.add_predicted_speed(allrows, motion)
    assert all(r["pred_v"] == stop_pred.NOT_SCORED_V for r in pre + post) and stop_pred.NOT_SCORED_V >= 10 * se.STOP_V_MPS
    assert app[0]["pred_v"] == 8.0, app[0]["pred_v"]  # t0: the recorded speed, not one step of the 0.9 brake
    assert abs(app[1]["pred_v"] - (8.0 - 4.0 * 0.9 * 0.1)) < 1e-9 and app[-1]["pred_v"] < app[1]["pred_v"]
    # a hole in the clock inside the span (3 s between two rows with the target in view) is not integrated across
    hole = [_srow(k, k * 0.1 + (3.0 if k >= 6 else 0.0), 8.0, 30 - 0.8 * k, brake=0.9) for k in range(12)]
    stop_pred.add_predicted_speed(hole, motion)
    assert hole[6]["pred_v"] == hole[5]["pred_v"] and hole[7]["pred_v"] < hole[6]["pred_v"], [r["pred_v"] for r in hole]
    # a row whose clock does not advance is not integrated either
    same = [_srow(k, 0.0 if k == 3 else k * 0.1, 8.0, 30 - 0.8 * k, brake=0.9) for k in range(8)]
    same[3]["t"] = same[2]["t"]
    stop_pred.add_predicted_speed(same, motion)
    assert same[3]["pred_v"] == same[2]["pred_v"], [r["pred_v"] for r in same]

    # ctl outputs outside [0, 1] are clamped before they drive the profile; a non-finite one is refused
    pairs = [({"i": 0, "ego": {"sim_t": 1.0, "speed_mps": 5.0}, "signs": []}, _ctl(2.5, 1.7)),
             ({"i": 1, "ego": {"sim_t": 1.1, "speed_mps": 5.0}, "signs": []}, _ctl(-0.4, -0.2))]
    rows, _ = stop_pred.stop_rows(pairs)
    assert (rows[0]["ctl_throttle"], rows[0]["ctl_brake"], rows[1]["ctl_throttle"], rows[1]["ctl_brake"]) == (1.0, 1.0, 0.0, 0.0), rows
    try:
        stop_pred.stop_rows([({"i": 0, "ego": {"sim_t": 1.0, "speed_mps": 5.0}}, _ctl(float("nan"), 0.0))])
    except stop_pred.StopNotMeasured:
        pass
    else:
        raise AssertionError("a non-finite ctl prediction must be refused")
    # an id'd stop line is a stop-sign-class event (counted in missed_stop); a sign with no id is skipped, not scored
    ev_rows = [_srow(k, k * 0.1, 8.0, 30 - 0.8 * k, cls="stop_line", brake=0.0) for k in range(40)]
    stop_pred.add_predicted_speed(ev_rows, stop_pred.Motion(2.0, 4.0, 0.0))
    res = se.score(ev_rows, speed_key="pred_v")
    assert res["slices"]["stop_line"]["events"] == 1 and res["slices"]["stop_line"]["missed"] == 1, res["slices"]
    line_pairs = [({"i": r["i"], "ego": {"sim_t": 100.0 + r["t"], "speed_mps": r["v"]}, "source": "S", "signs": list(r["stops"])},
                   _ctl(0.0, 0.0)) for r in ev_rows]
    got = stop_pred.measure(line_pairs, stop_pred.Motion(2.0, 4.0, 0.0))
    assert got["stop_events"] == 1 and got["missed_stop"] == 1.0 and got["light_events"] == 0 and got["missed_light"] is None, got

    # the red -> green rule (a proposal, off by default) must never see the RECORDED speed: a lamp that turns green
    # while the recorded car drives on is still an event the MODEL is scored on
    saved = se.RED_TO_GREEN_PROCEED_IS_MISS
    try:
        se.RED_TO_GREEN_PROCEED_IS_MISS = False
        lamp = [_srow(k, k * 0.1, 8.0, 40 - 0.8 * k, cls="traffic_light", state="red" if k < 10 else "green", brake=0.9)
                for k in range(50)]
        pairs = [({"i": r["i"], "ego": {"sim_t": 100.0 + r["t"], "speed_mps": r["v"]}, "source": "S",
                   "signs": [dict(s, id=s["id"]) for s in r["stops"]]}, _ctl(0.0, r["ctl_brake"])) for r in lamp]
        assert stop_pred.recorded_events(pairs) == {"stop": 0, "light": 1, "rows": 50}, stop_pred.recorded_events(pairs)
        got = stop_pred.measure(pairs, motion)
        assert got["light_events"] == 1 and got["missed_light"] == 0.0, got
    finally:
        se.RED_TO_GREEN_PROCEED_IS_MISS = saved
    print("approach spans (time holes, line + 1 m, merging), t0 row, clock holes, clamps, stop_line slice, red->green never reads recorded v OK")


def check_profile_never_reads_recorded_speed_after_t0() -> None:
    """Poison the recorded speed, a wheel speed and an acceleration after t0 (NaN, 0, permuted): same profile, same score."""
    from python.train import stop_pred

    def build():
        # the model brakes fully from k=2 (4.5 m/s^2 from 12 m/s: stopped by k=29), a sign 40 m ahead, window to k=34
        pairs = _approach("A", n=50, v=lambda k: 12.0, brake=lambda k: 1.0 if 2 <= k < 30 else 0.0, target="stop_sign")
        for row, _c in pairs:
            row["wheel_speed"] = row["ego"]["speed_mps"]
            row["ego"]["accel_mps2"] = 0.3
            row["tech"] = {"speed_mps": row["ego"]["speed_mps"], "accel": 0.3}
        pairs += _approach("F", n=60, v=lambda k: 12.0, brake=lambda k: 0.0, target=None)
        return pairs

    def profile(pairs):
        rows, _ = stop_pred.stop_rows(pairs)
        stop_pred.add_predicted_speed(rows, _motion())
        return [(r["source"], r["t"], r["pred_v"]) for r in rows]

    base_pairs = build()
    base = profile(base_pairs)
    t0 = 100.0  # the first row of source A is the approach start: 12 m/s, target at 40 m
    assert abs(base[0][2] - 12.0) < 1e-12 and base[10][2] < 12.0 and base[31][2] == 0.0 and base[34][2] == 0.0, [b[2] for b in base[:36]]

    def poison(kind):
        pairs = build()
        rng = np.random.RandomState(1)
        for row, _c in pairs:
            if row["source"] == "A" and row["ego"]["sim_t"] > t0 + 1e-9:
                vals = {"nan": float("nan"), "zero": 0.0, "perm": float(rng.uniform(0, 30))}[kind]
                row["ego"]["speed_mps"] = vals
                row["wheel_speed"] = vals
                row["ego"]["accel_mps2"] = vals
                row["tech"] = {"speed_mps": vals, "accel": vals}
        return pairs

    import struct

    def raw(x):
        return [struct.pack("<d", v[2]) for v in x]

    for kind in ("nan", "zero", "perm"):
        pairs = poison(kind)
        if kind == "nan":
            # the stop events need a finite recorded speed to exist, so NaN is poisoned straight into the profile step
            rows, _ = stop_pred.stop_rows(base_pairs)
            for r in rows:
                if r["source"] == "A" and r["t"] > t0 + 1e-9:
                    r["v"] = float("nan")
            stop_pred.add_predicted_speed(rows, _motion())
            got = [(r["source"], r["t"], r["pred_v"]) for r in rows]
        else:
            got = profile(pairs)
        assert raw(got) == raw(base), (kind, [g[2] for g in got[:12]], [b[2] for b in base[:12]])
    # the same score with speed_key = pred_v: misses, held stops and the stop distance do not move with the poison
    ref = stop_pred.measure(base_pairs, _motion())
    for kind in ("zero", "perm"):
        got = stop_pred.measure(poison(kind), _motion())
        for key in ("missed_stop", "missed_light", "stop_events", "light_events", "stop_distance_m", "stop_side_angle_missed"):
            assert got[key] == ref[key], (kind, key, got[key], ref[key])
    assert ref["missed_stop"] == 0.0 and ref["stop_events"] == 1, ref
    # entry (t0) does read the recorded speed: a car below 2 m/s at entry makes no event at all
    slow = _approach("A", n=50, v=lambda k: 1.0, brake=lambda k: 1.0, target="stop_sign") + _approach("F", n=60, v=lambda k: 12.0, brake=lambda k: 0.0, target=None)
    assert stop_pred.measure(slow, _motion())["stop_events"] == 0
    print("profile: byte-identical when the recorded speed / wheel speed / accel after t0 are NaN, 0 or random; same score OK")


def check_brake_zero_behaviour() -> None:
    """brake = 0: the car coasts on the fitted drag, a stopped car stays stopped, throttle ends the hold."""
    from python.train import stop_pred

    drag = _motion(2.0, 4.5, 0.2)

    def run(v0, thr, brk, n=30):
        rows, _ = stop_pred.stop_rows(_approach("A", n=n, v=lambda k: v0, brake=lambda k: brk, thr=thr, target="stop_sign"))
        stop_pred.add_predicted_speed(rows, drag)
        return [r["pred_v"] for r in rows]

    coast = run(10.0, 0.0, 0.0)
    # dv/dt = -0.2 v, stepped at 0.1 s: v_k = 10 * (1 - 0.02)^k
    assert all(abs(coast[k] - 10.0 * 0.98 ** k) < 1e-9 for k in (0, 1, 7, 29)), coast[:5]
    assert coast[-1] > 5.0  # a coast is not a stop
    # a stopped car: brake it to 0 (stay at brake 1), then release with throttle 0 -> stays 0 (hold needs no brake)
    # (sign 40 m ahead at 10 m/s recorded: the approach window is rows 0..40)
    held = run(10.0, 0.0, 1.0, n=40)
    assert held[-1] == 0.0
    rows, _ = stop_pred.stop_rows(_approach("A", n=40, v=lambda k: 10.0, brake=lambda k: 1.0 if k < 30 else 0.0, thr=0.0, target="stop_sign"))
    stop_pred.add_predicted_speed(rows, drag)
    assert rows[29]["pred_v"] == 0.0 and rows[39]["pred_v"] == 0.0 and rows[34]["pred_v"] == 0.0, [r["pred_v"] for r in rows[26:40]]
    # throttle after the stop: the car moves again (k_thr 2.0 * 0.5 = 1 m/s^2) -> the hold ends
    rows, _ = stop_pred.stop_rows(_approach("A", n=40, v=lambda k: 10.0, brake=lambda k: 1.0 if k < 30 else 0.0,
                                            thr=lambda k: 0.5 if k >= 30 else 0.0, target="stop_sign"))
    stop_pred.add_predicted_speed(rows, drag)
    assert rows[29]["pred_v"] == 0.0 and rows[31]["pred_v"] > 0.0 and rows[38]["pred_v"] > rows[31]["pred_v"], [r["pred_v"] for r in rows[28:40]]
    # and that decides the score: braked-then-released stays a held stop; braked-then-throttle is a miss
    still = _approach("A", n=45, v=lambda k: 10.0, brake=lambda k: 1.0 if k < 22 else 0.0, thr=0.0, target="stop_sign") + \
        _approach("F", n=60, v=lambda k: 10.0, brake=lambda k: 0.0, target=None)
    # (with the drag the car is at 0.09 m/s at k=18; throttle from k=19 lifts it past 0.5 m/s at k=21: 0.3 s below 0.5, no hold)
    go = _approach("A", n=45, v=lambda k: 10.0, brake=lambda k: 1.0 if k < 19 else 0.0, thr=lambda k: 1.0 if k >= 19 else 0.0, target="stop_sign") + \
        _approach("F", n=60, v=lambda k: 10.0, brake=lambda k: 0.0, target=None)
    assert stop_pred.measure(still, drag)["missed_stop"] == 0.0
    assert stop_pred.measure(go, drag)["missed_stop"] == 1.0
    # the spec says so
    spec = (ROOT / "docs" / "specs" / "trainer_2.md").read_text(encoding="utf-8")
    for phrase in ("dv/dt = K_THR * throttle - K_BRK * brake - K_DRAG * v", "coasts", "stays stopped", "train rows only"):
        assert phrase in spec, phrase
    print("brake = 0: coasts on the fitted drag, a stopped car stays stopped, throttle ends the hold; written in the spec OK")


def check_preflight_on_recording() -> None:
    """``stop_pred.preflight``: the manifest split of a recording with sign ids answers 'can 2b be scored' before training."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("SKIP check_preflight_on_recording: cv2 not installed")
        return
    import tempfile

    import stage2b_fixture as fx
    from python.train import stop_pred
    from python.train.trainer2_recipe import legacy_split_rows

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        good = fx.build(base / "good", fx.N_ROWS, fx.intersections(fx.N_ROWS))
        train, hold = legacy_split_rows((good,), good, holdout_blocks=1)
        out = stop_pred.preflight(train, hold, default_root=good)
        assert out["events"] == {"stop": 1, "light": 1, "rows": len(hold)}, out
        m = out["motion"]
        assert abs(m.k_thr - fx.K_THR) < 0.05 and abs(m.k_brk - fx.K_BRK) < 0.05 and abs(m.k_drag - fx.K_DRAG) < 0.01, m.as_dict()
        hold_ids = {r["i"] for r in hold}
        assert not (set(m.ids) & {str(i) for i in hold_ids}), "the fit saw a holdout row"
        assert m.n_rows < fx.SPLIT  # only the train segment's pairs
        # a holdout row that is also in the train rows is still kept out of the fit
        leaky = train + [dict(hold[0])]
        assert str(hold[0]["i"]) not in stop_pred.preflight(leaky, hold, default_root=good)["motion"].ids
        # without the sidecar there are no events: refused, saying so (never a vacuous 0)
        bare = fx.build(base / "bare", fx.N_ROWS, fx.intersections(fx.N_ROWS), with_signs=False)
        tr2, ho2 = legacy_split_rows((bare,), bare, holdout_blocks=1)
        try:
            stop_pred.preflight(tr2, ho2, default_root=bare)
        except stop_pred.StopNotMeasured as exc:
            assert "stage 2b" in str(exc) and "0 stop-sign/line" in str(exc) and "0 red/yellow-light" in str(exc), exc
            assert str(exc).count("stage 2b cannot be scored") == 1, exc
        else:
            raise AssertionError("no sign ids must refuse")
        # without the join root nothing is found either (single-root rows carry no source_root)
        try:
            stop_pred.preflight(train, hold)
        except stop_pred.StopNotMeasured as exc:
            assert "0 stop-sign/line" in str(exc), exc
        else:
            raise AssertionError("no root, no sidecar join")
        # a holdout with no stop but a train segment that never brakes: nothing to fit
        flat = fx.build(base / "flat", fx.N_ROWS, [], with_signs=True)
        tr3, ho3 = legacy_split_rows((flat,), flat, holdout_blocks=1)
        try:
            stop_pred.preflight(tr3, ho3, default_root=flat)
        except stop_pred.StopNotMeasured as exc:
            assert str(exc).startswith("stage 2b cannot be scored"), exc
        else:
            raise AssertionError("a recording with no intersections must refuse")
        # only one kind of event in the holdout is not enough: each is refused, saying which is missing
        ints = fx.intersections(fx.N_ROWS)
        for label, picked, want in (("sign only", ints[:2], "1 stop-sign/line and 0 red/yellow-light"),
                                    ("light only", [ints[0], ints[2]], "0 stop-sign/line and 1 red/yellow-light")):
            if label == "light only":  # the light alone, placed where the sign's approach starts
                picked = [ints[0], ("traffic_light", ints[1][1], "red")]
            rec = fx.build(base / label.replace(" ", "_"), fx.N_ROWS, picked)
            trx, hox = legacy_split_rows((rec,), rec, holdout_blocks=1)
            try:
                stop_pred.preflight(trx, hox, default_root=rec)
            except stop_pred.StopNotMeasured as exc:
                assert want in str(exc), (label, str(exc)[:200])
            else:
                raise AssertionError(f"{label}: must refuse")
        # events are there but the train segment never brakes: the fit's own reason, still named as stage 2b
        head = fx.script(fx.N_ROWS, [])
        sign_y = head[fx.SPLIT + 4][0] + 45.0
        nobrake = fx.build(base / "nobrake", fx.N_ROWS, [("stop_sign", sign_y, None), ("traffic_light", sign_y + 90.0, "red")])
        trn, hon = legacy_split_rows((nobrake,), nobrake, holdout_blocks=1)
        try:
            stop_pred.preflight(trn, hon, default_root=nobrake)
        except stop_pred.StopNotMeasured as exc:
            assert str(exc).startswith("stage 2b cannot be scored: the motion fit needs"), exc
            assert str(exc).count("stage 2b cannot be scored") == 1, exc  # a message that already says it is not prefixed twice
        else:
            raise AssertionError("no braking rows in train must refuse")
    print("preflight: events + train-only fit answered from the manifest split; no sign ids / no join root / no stops refuse with the reason OK")


def main() -> int:
    checks = (check_sign_ids_are_stable, check_stop_rows_inputs, check_predicted_speed_profile, check_measure_known_counts,
              check_side_angle_slice_and_empty_metrics, check_fit_motion_train_rows_only,
              check_profile_never_reads_recorded_speed_after_t0, check_brake_zero_behaviour,
              check_preflight_on_recording, check_fit_motion_edges, check_spans_and_profile_edges)
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
