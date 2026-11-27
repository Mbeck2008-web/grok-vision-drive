#!/usr/bin/env python3
"""collect_scene_lanes.py --route-plan: the flag and its refusals. Offline; BeamNG is never started.

The route driving itself (rolling ai.setLine windows, first spawn on the route) is tested through
python/collect/long_route.py with a fake vehicle in test_long_route.py. The collector's loop that calls
it needs a live BeamNG and has not been run live.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _collector():
    spec = importlib.util.spec_from_file_location("collect_scene_lanes_route", ROOT / "scripts" / "collect_scene_lanes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def check_flag_parses_and_defaults_off() -> None:
    mod = _collector()
    parser = mod.build_parser()
    assert parser.parse_args([]).route_plan == ""
    args = parser.parse_args(["--route-plan", "/tmp/r.json", "--no-excursions", "--maps", "west_coast_usa"])
    assert args.route_plan == "/tmp/r.json" and args.no_excursions is True
    print("collector: --route-plan parses, default off")


def check_refuses_a_bad_plan_before_any_game() -> None:
    from scripts.gvd_test_env import collect_root
    from test_long_route import loop_network, route_of

    mod = _collector()
    _g, route = route_of(loop_network(dirt=True), seed=2)
    with collect_root() as root:
        good = root / "route.json"
        good.write_text(json.dumps(route.to_json()), encoding="utf-8")
        cases = {
            "missing": (root / "nope.json", "synthetic", "unreadable"),
            "other map": (good, "west_coast_usa", "route plan is for map synthetic"),
        }
        for label, (plan, maps, want) in cases.items():
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = mod.main(["--out", str(root / "out"), "--maps", maps, "--route-plan", str(plan), "--minutes", "1", "--no-excursions"])
            assert code == 2, (label, code, err.getvalue())
            assert want in err.getvalue(), (label, err.getvalue())
    print("collector: an unreadable plan or a plan for another map exits 2 before BeamNG starts")


def main() -> None:
    check_flag_parses_and_defaults_off()
    check_refuses_a_bad_plan_before_any_game()
    print("test_collect_route_plan: OK")


if __name__ == "__main__":
    main()
