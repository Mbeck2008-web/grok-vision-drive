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
    print("motion fit: recovers k_thr / k_brk / k_drag from train rows, refuses holdout ids, unfittable sets and k_brk <= 0 OK")


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



def main() -> int:
    checks = (check_sign_ids_are_stable, check_stop_rows_inputs, check_predicted_speed_profile, check_measure_known_counts,
              check_side_angle_slice_and_empty_metrics, check_fit_motion_train_rows_only,
              check_profile_never_reads_recorded_speed_after_t0, check_brake_zero_behaviour)
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
