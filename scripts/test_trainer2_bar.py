#!/usr/bin/env python3
"""Trainer 2.0: per-iteration merge bar and the 2.0 naming bar (docs/specs/trainer_2.md, Decision rules).

Checks: check_iteration_bar_thresholds, check_2_0_bar_not_claimed_without_iqu45_drive,
check_intervention_tally, check_slice_direction, check_2_0_requires_every_slice_to_pass,
check_slice_rules_table_pins_what_a_card_may_claim, check_slice_rules_table_matches_spec,
check_unlabelable_slices_config_proposal_pending_michael,
check_iteration_bar_requires_labelable_slices_and_missed, check_slice_rules_are_frozen,
check_unmeasured_is_not_passing, check_unknown_intervention_kind_raises,
check_backstop_config_proposal_pending_michael (AEB and hand-rule activations count as
interventions: GVD call 2026-11-25, a PROPOSAL pending Michael), check_drive_record_gates.
Pure Python.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HI, LO = "higher_is_better", "lower_is_better"


def _s(value, direction=HI, limit=None):
    rec = {"value": value, "direction": direction}
    if limit is not None:
        rec["limit"] = limit
    return rec


def _slices(**over):
    """Every slice in SLICE_RULES at a passing value; ``over`` replaces (or, with None, drops) entries."""
    base = {
        "lead_vehicle_collisions": _s(0, LO, 0),
        "lead_brake_light_reaction": _s(1.0, HI, 1.0),
        "cut_in_collisions": _s(0, LO, 0),
        "cut_in_false_brakes_per_km": _s(0.5, LO),
        "cone_contacts": _s(0, LO, 0),
        "yield": _s(1.0, HI, 1.0),
        "railroad_crossing": _s(1.0, HI, 1.0),
        "sharp_curve": _s(1.0, HI, 1.0),
        "night": _s(1.0, HI, 1.0),
        "rain": _s(1.0, HI, 1.0),
        "fog": _s(1.0, HI, 1.0),
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


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
        "slices": dict(_slices() if slices is None else slices),
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
        ({"slices": _slices(**{"yield": _s(0.99, HI, 1.0)})}, "slice:yield"),
        ({"slices": _slices(**{"yield": None})}, "slice:yield"),
    ):
        res = iteration_bar(_card(0.15, **kw), prev)
        assert not res["pass"] and key in res["fail"], (kw, res)
    same_fb = iteration_bar(_card(0.15, fb_km=0.5, green_fb_km=0.1), prev)  # "no worse" includes equal
    assert same_fb["pass"], same_fb
    # An unlabelable slice may appear on the new card only: listed under no_baseline, not a fail.
    thin_prev = _card(0.20, slices=_slices(**{"fog": None}))
    new_slice = iteration_bar(_card(0.15, slices=_slices()), thin_prev)
    assert new_slice["pass"] and new_slice["no_baseline"] == ["fog"], new_slice
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
    t = tally_interventions([{"kind": "aeb", "t": 3.0}, {"kind": "hand_rule", "t": 9.0}, {"kind": "takeover", "t": 12.0}])
    assert t == {"takeover": 1, "collision": 0, "missed_stop_sign": 0, "missed_red_light": 0, "curb_strike": 0,
                 "aeb": 1, "hand_rule": 1, "backstop": 2, "total": 3}, t
    try:
        tally_interventions([{"t": 1.0}])
    except ValueError:
        pass
    else:
        raise AssertionError("event without kind accepted")
    print("interventions: takeover/collision/missed stop/red/curb strike + AEB + hand rule tallied OK")


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
    assert nomap["name"] == "1.x" and "heldout_map: split manifest names none" in nomap["fail"], nomap
    nomap_no_drive = naming_bar(new, prev, split_manifest={"heldout_map": "  "}, drives=[])
    assert "heldout_map: split manifest names none" in nomap_no_drive["fail"], nomap_no_drive
    box = naming_bar(new, prev, split_manifest=split, drives=[dict(_drive("day"), source="box"), _drive("night")])
    assert box["name"] == "1.x" and any(f.startswith("iqu45") for f in box["fail"]), box
    vram = naming_bar(new, prev, split_manifest=split, drives=[_drive("day"), dict(_drive("night"), live_vram_gb=4.2)])
    assert vram["name"] == "1.x" and any(f.startswith("live_vram") for f in vram["fail"]), vram
    iter_fail = naming_bar(_card(0.15, missed=1), prev, split_manifest=split, drives=[_drive("day"), _drive("night")])
    assert iter_fail["name"] == "1.x" and "iteration_bar" in iter_fail["fail"], iter_fail
    print("2.0 bar: named only with a 30-min day AND night IQU45 drive on the held-out map, random traffic, 0 interventions, RMS >= 25% OK")


def check_slice_direction() -> None:
    """Slices carry a direction: a rising lower-is-better slice is a regression (it used to pass)."""
    import math

    from python.train.trainer2_bar import iteration_bar

    prev = _card(0.20)
    for over, key in (
        ({"cut_in_false_brakes_per_km": _s(4.0, LO)}, "slice:cut_in_false_brakes_per_km"),  # 0.5 -> 4.0 /km
        ({"cut_in_collisions": _s(3, LO, 0)}, "slice:cut_in_collisions"),  # 0 -> 3
        ({"cut_in_false_brakes_per_km": _s(0.5000001, LO)}, "slice:cut_in_false_brakes_per_km"),
        ({"lead_brake_light_reaction": _s(0.99, HI, 1.0)}, "slice:lead_brake_light_reaction"),
        ({"lead_brake_light_reaction": _s(float("nan"), HI, 1.0)}, "slice:lead_brake_light_reaction"),
        ({"cut_in_collisions": _s(float("nan"), LO, 0)}, "slice:cut_in_collisions"),
        ({"cut_in_collisions": _s(None, LO, 0)}, "slice:cut_in_collisions"),
        ({"cut_in_collisions": _s(float("inf"), LO, 0)}, "slice:cut_in_collisions"),
        ({"lead_brake_light_reaction": _s(True, HI, 1.0)}, "slice:lead_brake_light_reaction"),  # a bool is not a measurement
    ):
        res = iteration_bar(_card(0.15, slices=_slices(**over)), prev)
        assert not res["pass"] and key in res["fail"], (over, res)
    better = iteration_bar(_card(0.15, slices=_slices(
        cut_in_false_brakes_per_km=_s(0.2, LO), lead_brake_light_reaction=_s(1.0, HI, 1.0))), prev)
    assert better["pass"], better  # lower-is-better improving, higher-is-better equal
    equal = iteration_bar(_card(0.15, slices=_slices(cut_in_false_brakes_per_km=_s(0.5, LO))), prev)
    assert equal["pass"], equal
    assert math.isfinite(better["lane_rms_gain"])
    for bad in ({"lead_brake_light_reaction": 1.0}, {"lead_brake_light_reaction": {"value": 1.0}}, {"lead_brake_light_reaction": {"value": 1.0, "direction": "up"}}):
        for which in ("new", "baseline"):
            cards = (_card(0.15, slices=_slices(**bad)), prev) if which == "new" else (_card(0.15), _card(0.20, slices=_slices(**bad)))
            try:
                iteration_bar(*cards)
            except ValueError:
                continue
            raise AssertionError(f"{which} slice {bad!r} without a direction accepted")
    print("slice direction: cut-in false brakes 0.5->4.0/km and collisions 0->3 fail; lower-is-better improvement passes OK")


def _name(new_slices, prev_slices=None, **kw):
    from python.train.trainer2_bar import naming_bar

    prev_slices = new_slices if prev_slices is None else prev_slices
    return naming_bar(_card(0.15, slices=new_slices), _card(0.20, slices=prev_slices),
                      split_manifest={"heldout_map": "utah"}, drives=[_drive("day"), _drive("night")], **kw)


def check_2_0_requires_every_slice_to_pass() -> None:
    """Spec: "every scenario slice from the addenda also passes". No regression is not the same as passing."""
    assert _name(_slices())["name"] == "2.0"
    # Same value on both cards: no regression, still not passing (yield 0.2 vs the table's 1.0).
    weak = _slices(**{"yield": _s(0.2, HI, 1.0)})
    res = _name(weak)
    assert res["name"] == "1.x" and res["iteration"]["pass"], res
    assert any(f.startswith("slices_pass: yield 0.2") for f in res["fail"]), res
    # The limit may be left off the card: the table's applies (0.2 fails, 1.0 passes).
    res = _name(_slices(**{"yield": _s(0.2, HI)}))
    assert res["name"] == "1.x" and any(f.startswith("slices_pass: yield") for f in res["fail"]), res
    assert _name(_slices(**{"yield": _s(1.0, HI)}))["name"] == "2.0"
    # A lower-is-better slice that did not regress but is over its limit (1 collision vs limit 0).
    hot = _slices(cut_in_collisions=_s(1, LO, 0))
    res = _name(hot)
    assert res["name"] == "1.x" and any(f.startswith("slices_pass: cut_in_collisions") for f in res["fail"]), res
    # Exactly at the limit passes (higher and lower, see _slices); one slice under it is enough to fail.
    one = _slices(**{"sharp_curve": _s(0.99, HI, 1.0)})
    res = _name(one)
    assert res["name"] == "1.x" and sum(f.startswith("slices_pass") for f in res["fail"]) == 1, res
    # NaN / None slice values fail their limit; the no-regression slice must still be measured.
    for bad in (float("nan"), None, float("inf")):
        res = _name(_slices(cut_in_collisions=_s(bad, LO, 0)))
        assert res["name"] == "1.x" and any(f.startswith("slices_pass: cut_in_collisions") for f in res["fail"]), (bad, res)
    res = _name(_slices(cut_in_false_brakes_per_km=_s(None, LO)))
    assert res["name"] == "1.x" and "slices_pass: cut_in_false_brakes_per_km not measured" in res["fail"], res
    # The no-regression slice passes the 2.0 bar only while it has not regressed (the iteration bar says so).
    res = _name(_slices(cut_in_false_brakes_per_km=_s(0.9, LO)), _slices(cut_in_false_brakes_per_km=_s(0.5, LO)))
    assert res["name"] == "1.x" and "iteration_bar" in res["fail"], res
    print("2.0 bar: every slice must meet the table's limit (yield 0.2 with no regression is 1.x), at-limit passes OK")


def check_slice_rules_table_pins_what_a_card_may_claim() -> None:
    """Skeptic 2's three probes: a card picks neither its slices, its directions nor its limits."""
    from python.train.trainer2_bar import HIGHER, LOWER, NO_REGRESSION, SLICE_RULES, iteration_bar

    # P5: yield 0.2 with a self-set lax limit of 0.1 (table: 1.0) is refused, not scored against 0.1.
    try:
        _name(_slices(**{"yield": _s(0.2, HI, 0.1)}))
    except ValueError as exc:
        assert "'yield'" in str(exc) and "limit" in str(exc), exc
    else:
        raise AssertionError("a card's self-set yield limit 0.1 reached the bar")
    # P6: 3 collisions labelled higher_is_better (table: lower_is_better) on both cards.
    flipped = _slices(lead_vehicle_collisions=_s(3, HI, 0), cut_in_collisions=_s(3, HI, 0))
    for prev_slices in (_slices(), flipped):  # flipped on the new card only, then on both cards
        try:
            _name(flipped, prev_slices)
        except ValueError as exc:
            assert "direction" in str(exc) and "disagrees with the table" in str(exc), exc
        else:
            raise AssertionError("collisions labelled higher_is_better reached the bar")
    try:
        iteration_bar(_card(0.15, slices=flipped), _card(0.20, slices=flipped))
    except ValueError:
        pass
    else:
        raise AssertionError("the iteration bar accepted a flipped collision direction")
    # P7: a card with only one slice is 1.x, and every missing required slice is listed.
    only = {"sharp_curve": _s(1.0, HI, 1.0)}
    res = _name(only)
    missing = [f for f in res["fail"] if f.endswith("required slice missing")]
    assert res["name"] == "1.x" and len(missing) == len(SLICE_RULES) - 1, res
    assert not any("sharp_curve" in f for f in missing)
    # Nothing at all.
    res = _name({})
    assert res["name"] == "1.x" and sum(f.endswith("required slice missing") for f in res["fail"]) == len(SLICE_RULES), res
    # A stated limit that equals the table's is fine; a different one, or one on a no-regression slice, is not.
    assert _name(_slices(**{"yield": _s(1.0, HI, 1.0)}))["name"] == "2.0"
    for bad in (_slices(**{"yield": _s(1.0, HI, 0.9)}), _slices(cut_in_false_brakes_per_km=_s(0.5, LO, 1.0)),
                _slices(cone_contacts=_s(0, LO, 1)), _slices(**{"yield": {"value": 1.0, "direction": HI, "limit": float("nan")}})):
        try:
            _name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a card limit that disagrees with the table was accepted: {bad!r}")
    # An unknown slice name raises (a typo must not drop a slice silently), on either card.
    for typo in ("yeild", "collisions", "Yield"):
        for which in ("new", "baseline"):
            args = (_slices(**{typo: _s(1.0, HI, 1.0)}), None) if which == "new" else (_slices(), _slices(**{typo: _s(1.0, HI, 1.0)}))
            try:
                _name(*args)
            except ValueError as exc:
                assert "not in the slice rules table" in str(exc), exc
            else:
                raise AssertionError(f"unknown slice {typo!r} accepted on the {which} card")
    # The table itself: every slice has a valid direction, a numeric limit or no-regression, and a stated basis.
    assert set(SLICE_RULES) == {
        "lead_vehicle_collisions", "lead_brake_light_reaction", "cut_in_collisions", "cut_in_false_brakes_per_km",
        "cone_contacts", "yield", "railroad_crossing", "sharp_curve", "night", "rain", "fog"}
    for name, rule in SLICE_RULES.items():
        assert rule["direction"] in (HIGHER, LOWER) and rule["basis"] in ("spec", "PROPOSAL"), (name, dict(rule))
        assert rule["limit"] == NO_REGRESSION or isinstance(rule["limit"], (int, float)), (name, dict(rule))
    assert SLICE_RULES["lead_vehicle_collisions"]["direction"] == LOWER and SLICE_RULES["lead_vehicle_collisions"]["limit"] == 0
    assert SLICE_RULES["cut_in_false_brakes_per_km"]["limit"] == NO_REGRESSION
    print("slice rules table: yield limit 0.1, collisions as higher_is_better, one-slice card, typos all refused or 1.x OK")


def check_slice_rules_table_matches_spec() -> None:
    """The table in code equals the "Slice rules table" in docs/specs/trainer_2.md (no silent drift either way)."""
    from python.train.trainer2_bar import HIGHER, LOWER, NO_REGRESSION, SLICE_RULES

    text = (ROOT / "docs" / "specs" / "trainer_2.md").read_text(encoding="utf-8")
    head = "### Slice rules table"
    assert head in text, "the spec has no 'Slice rules table' section"
    block = text[text.index(head):].split("\n## ", 1)[0]
    rows = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("`") and cells[0].endswith("`"):
            rows[cells[0].strip("`")] = cells
    assert set(rows) == set(SLICE_RULES), (sorted(set(rows) ^ set(SLICE_RULES)))
    for name, (key, row, direction, limit, basis, labelable) in rows.items():
        rule = SLICE_RULES[name]
        assert direction == rule["direction"] and direction in (HIGHER, LOWER), (name, direction)
        want = rule["limit"]
        assert (limit == NO_REGRESSION) if want == NO_REGRESSION else (float(limit) == float(want)), (name, limit, want)
        assert basis == rule["basis"], (name, basis, rule["basis"])
        assert (labelable == "yes") == rule["labelable_today"], (name, labelable)
        assert int(row) == rule["spec_row"], (name, row)
    assert "`REQUIRE_UNLABELABLE_SLICES_FOR_2_0`" in text and "PROPOSAL pending Michael" in text
    print(f"slice rules table: {len(rows)} spec rows == SLICE_RULES (name, direction, limit, basis, labelable, row) OK")


def check_unlabelable_slices_config_proposal_pending_michael() -> None:
    """PROPOSAL pending Michael: are cones / railroad / rain / fog required for the 2.0 name before they can be labelled?"""
    from python.train import trainer2_bar as bar

    assert bar.REQUIRE_UNLABELABLE_SLICES_FOR_2_0 is True  # PROPOSAL pending Michael
    unlabelable = [n for n, r in bar.SLICE_RULES.items() if not r["labelable_today"]]
    assert sorted(unlabelable) == ["cone_contacts", "fog", "railroad_crossing", "rain"]
    card = _slices(**{n: None for n in unlabelable})
    saved = bar.REQUIRE_UNLABELABLE_SLICES_FOR_2_0
    try:
        strict = _name(card)
        bar.REQUIRE_UNLABELABLE_SLICES_FOR_2_0 = False
        lax = _name(card)
        # Even when not required, a reported unlabelable slice is still checked against its limit.
        lax_fail = _name(_slices(cone_contacts=_s(1, LO, 0), rain=None, fog=None, railroad_crossing=None))
        lax_no_labelable = _name(_slices(**{"yield": None}))
    finally:
        bar.REQUIRE_UNLABELABLE_SLICES_FOR_2_0 = saved
    assert strict["name"] == "1.x" and sum(f.endswith("required slice missing") for f in strict["fail"]) == 4, strict
    assert lax["name"] == "2.0", lax
    assert lax_fail["name"] == "1.x" and any(f.startswith("slices_pass: cone_contacts") for f in lax_fail["fail"]), lax_fail
    assert lax_no_labelable["name"] == "1.x" and "slices_pass: yield required slice missing" in lax_no_labelable["fail"]
    print("unlabelable slices: required by default (1.x without them); False lets them be absent but a reported one must pass (PROPOSAL pending Michael) OK")


def check_unmeasured_is_not_passing() -> None:
    """None / NaN / inf never pass: lane RMS, false-brake rates, drive minutes, VRAM. (stop_events.score gives None at km = 0.)"""
    from python.train.trainer2_bar import iteration_bar, naming_bar

    prev = _card(0.20)
    nan, inf = float("nan"), float("inf")
    for bad in (nan, None, inf, "n/a"):
        res = iteration_bar(_card(0.15, fb_km=bad), prev)
        assert not res["pass"] and any(f.startswith("false_brake_rate: not measured") for f in res["fail"]), (bad, res)
        res = iteration_bar(_card(0.15, green_fb_km=bad), prev)
        assert not res["pass"] and any(f.startswith("green_false_brake_rate: not measured") for f in res["fail"]), (bad, res)
        res = iteration_bar(_card(bad), prev)
        assert not res["pass"] and "lane_rms" in res["fail"] and res["lane_rms_gain"] is None, (bad, res)
        res = iteration_bar(_card(0.15), _card(0.20, fb_km=bad))
        assert not res["pass"] and any("baseline not measured" in f for f in res["fail"]), (bad, res)
    split = {"heldout_map": "utah"}
    day_night = [_drive("day"), _drive("night")]
    res = naming_bar(_card(nan), prev, split_manifest=split, drives=day_night)
    assert res["name"] == "1.x" and "lane_rms_25" in res["fail"], res
    for bad in (nan, None, inf, "x"):
        res = naming_bar(_card(0.15), prev, split_manifest=split, drives=[_drive("day"), dict(_drive("night"), minutes=bad)])
        assert res["name"] == "1.x" and any(f.startswith("drive_minutes") for f in res["fail"]), (bad, res)
        res = naming_bar(_card(0.15), prev, split_manifest=split, drives=[_drive("day"), dict(_drive("night"), live_vram_gb=bad)])
        assert res["name"] == "1.x" and any(f.startswith("live_vram") for f in res["fail"]), (bad, res)
    try:
        iteration_bar(_card(0.15), _card(nan))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN baseline lane RMS accepted")
    # A stop card without the 22-37.5 deg or red-light slice has not shown 0 missed there.
    for gone in ("stop_side_angle", "red_light"):
        card = _card(0.15)
        del card["stop"]["slices"][gone]
        res = iteration_bar(card, prev)
        assert not res["pass"] and any(f.startswith(f"{gone}: slice missing") for f in res["fail"]), res
    print("unmeasured: None/NaN/inf RMS, false-brake rates, minutes and VRAM fail instead of passing OK")


def check_unknown_intervention_kind_raises() -> None:
    from python.train.trainer2_bar import tally_interventions

    for kind in ("Takeover", "disengage", "AEB", "takeover ", "lane_change", 5):
        try:
            tally_interventions([{"kind": kind, "t": 1.0}])
        except ValueError as exc:
            assert "unknown drive event kind" in str(exc), exc
        else:
            raise AssertionError(f"kind {kind!r} silently ignored")
    for no_kind in ({"t": 1.0}, {"kind": None}, {"kind": "  "}):
        try:
            tally_interventions([no_kind])
        except ValueError as exc:
            assert "without kind" in str(exc), exc
        else:
            raise AssertionError(f"event {no_kind!r} without a kind accepted")
    try:
        tally_interventions([{"kind": "aeb"}, {"kind": "Takeover"}])
    except ValueError:
        pass
    else:
        raise AssertionError("one unknown kind among valid ones was ignored")
    print("interventions: unknown or mis-cased event kinds raise instead of reading as a clean drive OK")


def check_backstop_config_proposal_pending_michael() -> None:
    """OPEN QUESTION, PROPOSAL pending Michael: do AEB / hand-rule activations count as interventions?

    Default True is GVD's 2026-11-25 call, not his ruling. Both settings are pinned so his answer is a
    one-line change of BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS.
    """
    from python.train import trainer2_bar as bar

    assert bar.BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS is True  # PROPOSAL pending Michael
    assert bar.BACKSTOP_KINDS == ("aeb", "hand_rule")
    events = [{"kind": "aeb", "t": 1.0}, {"kind": "hand_rule", "t": 2.0}]
    on = bar.tally_interventions(events)
    off = bar.tally_interventions(events, count_backstop=False)
    assert on["total"] == 2 and on["backstop"] == 2, on
    assert off["total"] == 0 and off["backstop"] == 2 and off["aeb"] == 1 and off["hand_rule"] == 1, off
    assert bar.tally_interventions([{"kind": "collision"}], count_backstop=False)["total"] == 1
    new, prev = _card(0.15), _card(0.20)
    split = {"heldout_map": "utah"}
    drives = [_drive("day"), _drive("night", events=events)]
    counted = bar.naming_bar(new, prev, split_manifest=split, drives=drives)
    assert counted["name"] == "1.x" and any(f.startswith("interventions:") for f in counted["fail"]), counted
    reported = bar.naming_bar(new, prev, split_manifest=split, drives=drives, count_backstop=False)
    assert reported["name"] == "2.0", reported
    assert reported["interventions"][1]["backstop"] == 2, reported["interventions"]  # still reported
    core = [_drive("day"), _drive("night", events=[{"kind": "curb_strike", "t": 5.0}])]
    assert bar.naming_bar(new, prev, split_manifest=split, drives=core, count_backstop=False)["name"] == "1.x"
    print("backstop: AEB / hand-rule count as interventions by default (PROPOSAL pending Michael); both settings pinned OK")


def check_drive_record_gates() -> None:
    from python.train.trainer2_bar import MAX_LIVE_VRAM_GB, naming_bar

    new, prev = _card(0.15), _card(0.20)
    split = {"heldout_map": "utah"}
    ok = [_drive("day"), _drive("night")]
    assert naming_bar(new, prev, split_manifest=split, drives=ok)["name"] == "2.0"
    gone = _drive("night")
    del gone["camera_only"]
    cases = {
        "camera_only": [dict(_drive("day"), camera_only=False), _drive("night")],
        "camera_only missing": [_drive("day"), gone],
        "camera_only truthy": [_drive("day"), dict(_drive("night"), camera_only="yes")],
    }
    for why, drives in cases.items():
        res = naming_bar(new, prev, split_manifest=split, drives=drives)
        assert res["name"] == "1.x" and any(f.startswith("camera_only") for f in res["fail"]), (why, res)
    novram = _drive("night")
    del novram["live_vram_gb"]
    for why, night in (("None", dict(_drive("night"), live_vram_gb=None)), ("missing", novram)):
        res = naming_bar(new, prev, split_manifest=split, drives=[_drive("day"), night])
        assert res["name"] == "1.x" and any(f.startswith("live_vram") for f in res["fail"]), (why, res)
    at_cap = naming_bar(new, prev, split_manifest=split,
                        drives=[_drive("day"), dict(_drive("night"), live_vram_gb=MAX_LIVE_VRAM_GB)])
    assert at_cap["name"] == "2.0", at_cap  # exactly 4.0 GB is allowed
    print("drive records: camera_only must be True, live_vram_gb must be present and <= 4.0 GB OK")



def check_iteration_bar_requires_labelable_slices_and_missed() -> None:
    """P8: empty slice maps on both cards fail. P10: a stop slice without 'missed' fails (not default 0)."""
    from python.train.trainer2_bar import SLICE_RULES, iteration_bar

    assert iteration_bar(_card(0.15), _card(0.20))["pass"]
    # P8
    res = iteration_bar(_card(0.15, slices={}), _card(0.20, slices={}))
    labelable = [n for n, r in SLICE_RULES.items() if r["labelable_today"]]
    unlabelable = [n for n, r in SLICE_RULES.items() if not r["labelable_today"]]
    assert not res["pass"], res
    for name in labelable:
        assert f"slice:{name}" in res["fail"], (name, res)
    for name in unlabelable:
        assert f"slice:{name}" not in res["fail"], (name, res)
    # Unlabelable slices may be absent from both cards; the rest of the bar still passes.
    dropped = {n: None for n in unlabelable}
    assert iteration_bar(_card(0.15, slices=_slices(**dropped)), _card(0.20, slices=_slices(**dropped)))["pass"]
    # Missing from the baseline only (yield is labelable).
    res = iteration_bar(_card(0.15), _card(0.20, slices=_slices(**{"yield": None})))
    assert not res["pass"] and "slice:yield" in res["fail"], res
    # P10: stop slice present but 'missed' omitted is not a 0-miss pass.
    for gone in ("stop_side_angle", "red_light"):
        card = _card(0.15)
        rec = dict(card["stop"]["slices"][gone])
        rec.pop("missed")
        card["stop"]["slices"][gone] = rec
        res = iteration_bar(card, _card(0.20))
        assert not res["pass"] and f"{gone}: missed not reported" in res["fail"], (gone, res)
    # An explicit missed=0 still passes; missed=None does not.
    card = _card(0.15)
    card["stop"]["slices"]["stop_side_angle"] = {"events": 4, "missed": 0}
    assert iteration_bar(card, _card(0.20))["pass"]
    card = _card(0.15)
    card["stop"]["slices"]["red_light"] = {"events": 6, "missed": None}
    res = iteration_bar(card, _card(0.20))
    assert not res["pass"] and "red_light: missed not reported" in res["fail"], res
    # P1: a non-mapping rec (0, [], "missed") is a clean "missed not reported", never a TypeError.
    for bad in (0, [], "missed"):
        card = _card(0.15)
        card["stop"]["slices"]["red_light"] = bad
        try:
            res = iteration_bar(card, _card(0.20))
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"non-mapping red_light rec {bad!r} raised {type(exc).__name__}: {exc}") from exc
        assert not res["pass"] and res["fail"].count("red_light: missed not reported") == 1, (bad, res)
        assert "red_light" not in res["fail"], (bad, res)  # not scored as a counted miss
    # missed must be a real int: reject 0.5, "0", bool (int(True)==1 / int("0")==0 would mis-score).
    for bad in (0.5, 0.0, "0", "1", True, False):
        card = _card(0.15)
        card["stop"]["slices"]["red_light"] = {"events": 6, "missed": bad}
        try:
            res = iteration_bar(card, _card(0.20))
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"non-int missed {bad!r} raised {type(exc).__name__}: {exc}") from exc
        assert not res["pass"] and "red_light: missed not reported" in res["fail"], (bad, res)
    card = _card(0.15)
    card["stop"]["slices"]["red_light"] = {"events": 6, "missed": 1}
    res = iteration_bar(card, _card(0.20))
    assert not res["pass"] and "red_light" in res["fail"] and "red_light: missed not reported" not in res["fail"], res
    # P6b: missing from new but present in baseline is listed once, not twice (required-loop + prev-loop).
    res = iteration_bar(_card(0.15, slices=_slices(**{"yield": None})), _card(0.20))
    assert not res["pass"] and res["fail"].count("slice:yield") == 1, res
    print("iteration bar: every labelable slice required on both cards (empty maps fail); stop 'missed' is a real int; non-mapping rec fails cleanly OK")


def check_slice_rules_are_frozen() -> None:
    """X23: SLICE_RULES is immutable, including each nested rule (MappingProxyType or equivalent)."""
    from python.train.trainer2_bar import SLICE_RULES

    rule = SLICE_RULES["yield"]
    try:
        SLICE_RULES["yield"] = dict(rule)
    except TypeError:
        pass
    else:
        raise AssertionError("SLICE_RULES accepted an item assignment (not frozen)")
    try:
        del SLICE_RULES["yield"]
    except TypeError:
        pass
    else:
        raise AssertionError("SLICE_RULES accepted a deletion (not frozen)")
    try:
        SLICE_RULES.update({})  # type: ignore[attr-defined]
    except (TypeError, AttributeError):
        pass
    else:
        raise AssertionError("SLICE_RULES.update succeeded")
    try:
        SLICE_RULES["yield"]["limit"] = 0.1
    except TypeError:
        pass
    else:
        raise AssertionError("a nested SLICE_RULES entry accepted an item assignment (not frozen)")
    try:
        SLICE_RULES["yield"]["direction"] = "lower_is_better"
    except TypeError:
        pass
    else:
        raise AssertionError("a nested SLICE_RULES entry accepted a direction change")
    assert SLICE_RULES["yield"]["limit"] == 1.0 and SLICE_RULES["yield"] is rule
    print("slice rules table: SLICE_RULES and each nested rule are frozen OK")


def main() -> int:
    checks = (
        check_iteration_bar_thresholds,
        check_iteration_bar_requires_labelable_slices_and_missed,
        check_slice_rules_are_frozen,
        check_intervention_tally,
        check_2_0_bar_not_claimed_without_iqu45_drive,
        check_slice_direction,
        check_2_0_requires_every_slice_to_pass,
        check_slice_rules_table_pins_what_a_card_may_claim,
        check_slice_rules_table_matches_spec,
        check_unlabelable_slices_config_proposal_pending_michael,
        check_unmeasured_is_not_passing,
        check_unknown_intervention_kind_raises,
        check_backstop_config_proposal_pending_michael,
        check_drive_record_gates,
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
