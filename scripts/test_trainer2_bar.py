#!/usr/bin/env python3
"""Trainer 2.0: per-iteration merge bar and the 2.0 naming bar (docs/specs/trainer_2.md, Decision rules).

Checks: check_iteration_bar_thresholds, check_2_0_bar_not_claimed_without_iqu45_drive,
check_intervention_tally (AEB and hand-rule activations count as interventions:
GVD call 2026-11-25, a proposal pending Michael). Pure Python.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _card(rms: float = 0.20, *, missed: int = 0, side_missed: int = 0, red_missed: int = 0,
          fb_km: float = 0.5, green_fb_km: float = 0.1, slices=None, ssl: bool = True):
    return {
        "lane_rms": rms,
        "ssl_pretrain": ssl,
        "stop": {
            "missed": missed,
            "slices": {"stop_side_angle": {"events": 4, "missed": side_missed},
                       "red_light": {"events": 6, "missed": red_missed}},
            "false_brakes_per_km": fb_km,
            "green_false_brakes_per_km": green_fb_km,
        },
        "slices": dict(slices or {"lead_brake": 1.0, "cut_in": 1.0, "sharp_curve": 0.9}),
    }


def _drive(tod: str, *, minutes: float = 30.0, mapname: str = "utah", traffic: bool = True, events=()):
    return {"source": "iqu45", "map": mapname, "time_of_day": tod, "minutes": minutes,
            "random_traffic": traffic, "camera_only": True, "live_vram_gb": 2.1, "events": list(events)}


def check_iteration_bar_thresholds() -> None:
    from python.train.trainer2_bar import iteration_bar

    prev = _card(0.20)
    ok = iteration_bar(_card(0.18), prev)  # exactly 10% better
    assert ok["pass"], ok
    near = iteration_bar(_card(0.1801), prev)  # 9.95% better
    assert not near["pass"] and "lane_rms" in near["fail"], near
    for kw, key in (
        ({"missed": 1}, "missed_stop"),
        ({"side_missed": 1}, "stop_side_angle"),
        ({"red_missed": 1}, "red_light"),
        ({"fb_km": 0.51}, "false_brake_rate"),
        ({"green_fb_km": 0.11}, "green_false_brake_rate"),
        ({"slices": {"lead_brake": 1.0, "cut_in": 0.99, "sharp_curve": 0.9}}, "slice:cut_in"),
        ({"slices": {"lead_brake": 1.0, "sharp_curve": 0.9}}, "slice:cut_in"),
    ):
        res = iteration_bar(_card(0.15, **kw), prev)
        assert not res["pass"] and key in res["fail"], (kw, res)
    same_fb = iteration_bar(_card(0.15, fb_km=0.5, green_fb_km=0.1), prev)  # "no worse" includes equal
    assert same_fb["pass"], same_fb
    new_slice = iteration_bar(_card(0.15, slices={"lead_brake": 1.0, "cut_in": 1.0, "sharp_curve": 0.9, "yield": 0.5}), prev)
    assert new_slice["pass"] and new_slice["no_baseline"] == ["yield"], new_slice
    bad = copy.deepcopy(prev)
    del bad["ssl_pretrain"]
    try:
        iteration_bar(_card(0.15), bad)
    except ValueError:
        pass
    else:
        raise AssertionError("baseline card without ssl_pretrain accepted")
    print("iteration bar: RMS >= 10% better, 0 missed (side angle + red), false/green-false no worse, slices no regression OK")


def check_intervention_tally() -> None:
    from python.train.trainer2_bar import INTERVENTION_KINDS, tally_interventions

    assert set(INTERVENTION_KINDS) == {
        "takeover", "collision", "missed_stop_sign", "missed_red_light", "curb_strike", "aeb", "hand_rule"}
    t = tally_interventions([{"kind": "aeb", "t": 3.0}, {"kind": "hand_rule", "t": 9.0}, {"kind": "takeover", "t": 12.0},
                             {"kind": "lane_change", "t": 14.0}])
    assert t == {"takeover": 1, "collision": 0, "missed_stop_sign": 0, "missed_red_light": 0, "curb_strike": 0,
                 "aeb": 1, "hand_rule": 1, "total": 3}, t
    try:
        tally_interventions([{"t": 1.0}])
    except ValueError:
        pass
    else:
        raise AssertionError("event without kind accepted")
    print("interventions: takeover/collision/missed stop/red/curb strike + AEB + hand rule tallied; other events ignored OK")


def check_2_0_bar_not_claimed_without_iqu45_drive() -> None:
    from python.train.trainer2_bar import naming_bar

    new, prev = _card(0.15), _card(0.20)  # 25% better
    split = {"heldout_map": "utah"}
    full = naming_bar(new, prev, split_manifest=split, drives=[_drive("day"), _drive("night")])
    assert full["name"] == "2.0" and full["pass"], full
    assert naming_bar(new, prev, split_manifest=split, drives=[])["name"] == "1.x"
    assert naming_bar(new, prev, split_manifest=split, drives=None)["name"] == "1.x"
    cases = {
        "night drive": [_drive("day")],
        "drive_minutes": [_drive("day"), _drive("night", minutes=29.9)],
        "heldout_map": [_drive("day"), _drive("night", mapname="italy")],
        "random_traffic": [_drive("day"), _drive("night", traffic=False)],
        "interventions": [_drive("day"), _drive("night", events=[{"kind": "aeb", "t": 100.0}])],
    }
    for why, drives in cases.items():
        res = naming_bar(new, prev, split_manifest=split, drives=drives)
        assert res["name"] == "1.x" and any(why.split()[0] in f for f in res["fail"]), (why, res)
    rms = naming_bar(_card(0.1501), prev, split_manifest=split, drives=[_drive("day"), _drive("night")])
    assert rms["name"] == "1.x" and "lane_rms_25" in rms["fail"], rms
    nomap = naming_bar(new, prev, split_manifest={"heldout_map": None}, drives=[_drive("day"), _drive("night")])
    assert nomap["name"] == "1.x" and any(f.startswith("heldout_map") for f in nomap["fail"]), nomap
    box = naming_bar(new, prev, split_manifest=split, drives=[dict(_drive("day"), source="box"), _drive("night")])
    assert box["name"] == "1.x" and any(f.startswith("iqu45") for f in box["fail"]), box
    vram = naming_bar(new, prev, split_manifest=split, drives=[_drive("day"), dict(_drive("night"), live_vram_gb=4.2)])
    assert vram["name"] == "1.x" and any(f.startswith("live_vram") for f in vram["fail"]), vram
    iter_fail = naming_bar(_card(0.15, missed=1), prev, split_manifest=split, drives=[_drive("day"), _drive("night")])
    assert iter_fail["name"] == "1.x" and "iteration_bar" in iter_fail["fail"], iter_fail
    print("2.0 bar: named only with a 30-min day AND night IQU45 drive on the held-out map, random traffic, 0 interventions, RMS >= 25% OK")


def main() -> int:
    checks = (check_iteration_bar_thresholds, check_intervention_tally, check_2_0_bar_not_claimed_without_iqu45_drive)
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
