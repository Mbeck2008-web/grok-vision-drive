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
check_hand_rule_untouched. Pure Python (no torch).
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


def main() -> int:
    checks = (
        check_stop_event_requires_hold,
        check_stop_side_angle_slice,
        check_light_events_red_yellow_only,
        check_stop_metrics_on_fixture,
        check_stop_events_split_safe,
        check_stop_labels_train_only,
        check_hand_rule_untouched,
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
