#!/usr/bin/env python3
"""The collector's route wiring (--route-plan) end to end, offline, against a fake BeamNG.

``collect.main`` runs on the fake-BeamNG harness of test_collect_spawn_fix (fake clock, fake vehicle,
no game, never a live BeamNG). The fake car obeys the AI commands it is sent: it drives the planned route
at 25 m/s once it has been told ``ai.setMode("manual")`` and given an ``ai.setLine`` window, and stops
when it is handed back to ``span``. This tests what the unit tests of long_route.py cannot reach: the
collector loop's use of the driver.

Wiring pinned here: route commands reach the vehicle; the plain span send stays off while the route is
active; the AI state is re-sent after a spawn/segment open; the first spawn is the route's start, not a
deck draw; no deck respawns (impact, stall) and no excursions on a route clip; the hand-off back to
span at the route's end and when the car leaves the route; ``route_driver.json`` and collect.json carry
the driver summary; and the route starts with the car at rest.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_collect_spawn_fix as sf  # noqa: E402
from python.collect import long_route as lr  # noqa: E402
from python.collect.spawn_mix import CITY, HIGHWAY  # noqa: E402

SPEED = 25.0
MAP = "west_coast_usa"


def make_route() -> lr.Route:
    """1 km of highway north, then 1.5 km of city road east, south and west back to the start."""
    legs = [((0.0, 0.0), (0.0, 1000.0), HIGHWAY), ((0.0, 1000.0), (250.0, 1000.0), CITY),
            ((250.0, 1000.0), (250.0, 0.0), CITY), ((250.0, 0.0), (0.0, 0.0), CITY)]
    pts: list[tuple[float, float, float]] = []
    kinds: list[str] = []
    for (ax, ay), (bx, by), kind in legs:
        n = max(1, int(math.hypot(bx - ax, by - ay) // 25))
        for k in range(n):
            pts.append((ax + (bx - ax) * k / n, ay + (by - ay) * k / n, 0.4))
            kinds.append(kind)
    pts.append((0.0, 0.0, 0.4))
    kinds.append(CITY)
    cum = tuple(lr._cumulative(pts))
    far_i = next(i for i, p in enumerate(pts) if p[:2] == (0.0, 1000.0))
    return lr.Route(
        map=MAP, seed=1, direction=0, requested_direction=0, points=tuple(pts), kinds=tuple(kinds),
        far_end_index=far_i, highway_m=cum[far_i], return_m=cum[-1] - cum[far_i], dirt_m=0.0, city_m=cum[-1] - cum[far_i],
        ramp_m=0.0, legs=("highway", "city", "highway"), start_road="hw0", far_road="hw0", via=(250.0, 500.0), cum=cum,
    )


class RouteCar(sf._CrashVehicle):
    """Starts at rest. Drives the route at SPEED once manual mode and a line were sent; span stops it."""

    def __init__(self, stop: Path, route: lr.Route | None = None, rest_polls: int = 5, leave_at_m: float | None = None,
                 tail_polls: int = 125, max_polls: int = 3000, **_kw: object) -> None:
        super().__init__(stop, stop_at=10**9)
        self.route = route
        self.rest_polls, self.leave_at_m, self.tail_polls, self.max_polls = rest_polls, leave_at_m, tail_polls, max_polls
        self.mode = None
        self.has_line = False
        self.s = 0.0
        self.placed: list[tuple[tuple, tuple]] = []
        self.cmd_poll: list[int] = []
        self.cmd_s: list[float] = []
        self.parked = False
        self.v = 0.0
        self.span_poll: int | None = None
        self.max_s = 0.0

    def teleport(self, pos, rot_quat=None, reset=False) -> None:
        self.placed.append((tuple(pos), tuple(rot_quat or ())))
        if len(self.placed) == 1:  # the car is on the first pose; later teleports do not move the fake
            self.s = 0.0

    def queue_lua_command(self, command: str, *args: object) -> None:
        super().queue_lua_command(command, *args)
        self.cmd_poll.append(self.polls)
        self.cmd_s.append(self.s)
        if 'ai.setMode("manual")' in command:
            self.mode = "manual"
        elif "ai.setMode(" in command and '"span"' in command:
            if self.mode == "manual":  # a hand-back after the route; the first span is the collector's own start-up send
                self.span_poll = self.span_poll or self.polls
            self.mode = "span"
        if "setLine" in command:
            self.has_line = True

    def _at(self, s: float) -> tuple[float, float, float, float]:
        r, cum = self.route, self.route.cum
        i = max(0, min(len(r.points) - 2, next((k for k in range(len(cum) - 1) if cum[k + 1] >= s), len(cum) - 2)))
        t = min(1.0, max(0.0, (s - cum[i]) / max(1e-9, cum[i + 1] - cum[i])))
        (ax, ay, _), (bx, by, _) = r.points[i], r.points[i + 1]
        n = max(1e-9, math.hypot(bx - ax, by - ay))
        return ax + (bx - ax) * t, ay + (by - ay) * t, (bx - ax) / n, (by - ay) / n

    def poll_sensors(self) -> None:
        self.polls += 1
        driven = self.polls > self.rest_polls and self.mode == "manual" and self.has_line
        if self.leave_at_m is not None and self.s >= self.leave_at_m:
            self.parked = True  # off the route, and stopped there (a stuck car is a car at rest)
        if driven and not self.parked:
            self.v = SPEED
            self.s = min(self.route.total_m, self.s + self.v * 0.1)
        else:
            self.v = max(0.0, self.v - 0.5)  # 5 m/s2: a stop, not an impact
        self.max_s = max(self.max_s, self.s)
        x, y, dx, dy = self._at(self.s)
        if self.parked:
            x += 120.0
        self.state = {
            "pos": [x, y, 0.5], "dir": [dx, dy, 0.0], "up": [0.0, 0.0, 1.0],
            "electrics": {"wheelspeed": self.v, "steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0},
        }
        done_wait = self.span_poll is not None and self.polls >= self.span_poll + self.tail_polls
        if done_wait or self.polls >= self.max_polls:
            self.stop.write_text("stop", encoding="utf-8")


def run(route: lr.Route, extra: list[str] | None = None, **car_kw: object):
    from scripts.gvd_test_env import collect_root

    cars: list[RouteCar] = []

    def factory(stop: Path, **_kw: object) -> RouteCar:
        cars.append(RouteCar(stop, route, **car_kw))
        return cars[-1]

    root = sf._COLLECT_ROOTS.enter_context(collect_root())
    plan = root / "route.json"
    plan.write_text(json.dumps(route.to_json()), encoding="utf-8")
    previous = sf._CrashVehicle
    sf._CrashVehicle = factory  # type: ignore[assignment]
    try:
        # No --no-excursions, --no-impact-respawn or stall setting of its own: the route clip must turn those off itself.
        code, text, out = sf._run_west_coast(
            sf._Paint(lambda x, y: True), sf._west_pools(),
            ["--route-plan", str(plan), "--min-paint-cov", "0", "--stall-respawn-s", "5"] + list(extra or []), 10**9, root=root,
        )
    finally:
        sf._CrashVehicle = previous  # type: ignore[assignment]
    return code, text, out, cars[0]


def _no_excursions_flag_run(route: lr.Route, **kw: object):
    # _run_west_coast always adds --no-excursions; the route test needs excursions left at their default.
    import scripts.collect_scene_lanes as collect

    orig = collect.main

    def main(argv):
        return orig([a for a in argv if a != "--no-excursions"])

    collect.main = main  # type: ignore[assignment]
    try:
        return run(route, **kw)
    finally:
        collect.main = orig  # type: ignore[assignment]


def check_route_runs_through_the_collector() -> None:
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route)
    assert code == 0, (code, text[-2500:])
    total = route.total_m
    # first spawn: the route's own start pose, announced, not a deck draw
    assert "collect route start" in text and "road=hw0" in text, text[:2500]
    start, _rot = route.start_pose()
    assert car.placed and all(abs(a - b) < 1e-6 for a, b in zip(car.placed[0][0], start)), (car.placed[:1], start)
    # no deck respawn of any kind (impact, stall) during the clip
    assert len(car.placed) == 1, car.placed
    assert "stall_respawn" not in text and "impact_respawn" not in text, text[-1500:]
    # the route commands reached the car, while the car was at rest at the start
    lines = [i for i, c in enumerate(car.commands) if "setLine" in c]
    assert len(lines) >= 20, len(lines)
    assert car.max_s >= total - lr.END_TOL_M - 3.0, (car.max_s, total)
    assert car.cmd_s[lines[0]] == 0.0, car.cmd_s[lines[0]]  # the first window went out while the car was still at rest
    # the AI mode is re-sent once the car is driving (the segment opens and clears the AI state), well before 80 m
    manual = [i for i, c in enumerate(car.commands) if 'ai.setMode("manual")' in c]
    assert len(manual) == 2, [car.commands[i] for i in manual]
    assert car.cmd_poll[manual[1]] - car.cmd_poll[manual[0]] <= 12, [car.cmd_poll[i] for i in manual]
    # the plain span send stays off while the route is active, and takes over once it ends
    spans = [i for i, c in enumerate(car.commands) if 'ai.setMode("span")' in c and i > lines[0]]
    assert spans and min(spans) > max(lines), (spans[:3], lines[-3:])
    assert len(spans) >= 2, "the collector's own span send resumes after the driver's hand-off, even with the car at rest"
    assert not any("setLine" in c for c in car.commands[spans[0]:]), "no route line after the hand-off"
    # no excursions on a route clip (its lane/weave logic would fight the route)
    assert car.bbox_calls == 0, car.bbox_calls
    assert "collect excursion" not in text and "excursion=weave" not in text
    # the driver summary is written to the clip and to collect.json
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["far_end_reached"] is True, summary
    assert summary["progress_m"] >= total - lr.END_TOL_M and summary["lines_sent"] == len(lines), (summary, len(lines))
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["route_driver"] == summary, (status["route_driver"], summary)
    assert status["impact_respawn"]["on"] is False, status["impact_respawn"]
    assert "collect route west_coast_usa#0 driver" in text and "'state': 'done'" in text, text[-1500:]
    print(f"collector route run: {len(lines)} route lines over {total:.0f} m, start pose from the route, one respawn-free segment, hand-off to span, route_driver.json written")


def check_leaving_the_route_hands_back_to_span() -> None:
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, leave_at_m=600.0)
    assert code == 0, (code, text[-2500:])
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "off_route" and summary["far_end_reached"] is False, summary
    assert summary["max_off_route_m"] > 100.0, summary
    lines = [i for i, c in enumerate(car.commands) if "setLine" in c]
    spans = [i for i, c in enumerate(car.commands) if 'ai.setMode("span")' in c and i > lines[0]]
    assert spans and min(spans) > max(lines), "span only after the last route line"
    assert "'state': 'off_route'" in text and len(car.placed) == 1, text[-1500:]
    # the car is parked off the route (at rest): the hand-back comes from the route driver within a second or two,
    # not from the 10 s stationary abort and recover
    assert "collect abort" not in text and "collect recover" not in text, text[-1500:]
    assert car.span_poll is not None
    print(f"collector route run: leaving the route (max {summary['max_off_route_m']} m off) hands the car back to span")


def main() -> None:
    check_route_runs_through_the_collector()
    check_leaving_the_route_hands_back_to_span()
    print("test_collect_route_drive: OK")


if __name__ == "__main__":
    main()
