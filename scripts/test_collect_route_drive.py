#!/usr/bin/env python3
"""The collector's route wiring (--route-plan) end to end, offline, against a fake BeamNG.

``collect.main`` runs on the fake-BeamNG harness of test_collect_spawn_fix (fake clock, fake vehicle,
no game, never a live BeamNG). The fake car obeys the AI commands it is sent: it drives the planned route
at 25 m/s once it has been told ``ai.setMode("manual")`` and given a set_line window, and stops
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
from python.collect.excursion import is_set_line as _is_set_line  # noqa: E402

SPEED = 25.0
CRAWL_M, CRAWL_V = 4.0, 0.95
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
                 tail_polls: int = 40, max_polls: int = 3000, stall_at_m: float | None = None,
                 recover_to: str | None = None, stuck_start: str | None = None,
                 restall_after_m: float | None = None, wrong_deck_dz: float = 0.0,
                 wrong_deck_on_place: int | None = None, follow_deck_after_m: float | None = None,
                 follow_deck_dz: float = 7.8, **_kw: object) -> None:
        super().__init__(stop, stop_at=10**9)
        self.wrong_deck_dz = float(wrong_deck_dz or 0.0)
        self.wrong_deck_on_place = int(wrong_deck_on_place) if wrong_deck_on_place else None
        self.follow_deck_after_m = None if follow_deck_after_m is None else float(follow_deck_after_m)
        self.follow_deck_dz = float(follow_deck_dz)
        self.route = route
        self.rest_polls, self.leave_at_m, self.tail_polls, self.max_polls = rest_polls, leave_at_m, tail_polls, max_polls
        self.mode = None
        self.has_line = False
        self.set_line_polls: list[int] = []
        self.s = 0.0
        self.placed: list[tuple[tuple, tuple]] = []
        self.s_at_teleport: list[float] = []  # where on the route the car was when each teleport came
        self.place_polls: list[int] = []
        self.cmd_poll: list[int] = []
        self.cmd_s: list[float] = []
        self.parked = False
        self.v = 0.0
        self.stall_at_m, self.recover_to = stall_at_m, recover_to
        self.stall = None  # None -> 'stalled' (stops at stall_at_m) -> 'resumed' (a fresh manual mode arrived)
        self.stall_poll: int | None = None
        self.span_poll: int | None = None
        self.max_s = 0.0
        # The smoke failure: the car sits at the route's start and never moves (stationary abort, recover, abort
        # again). "until_moved": it drives once a teleport puts it on the route 10+ m past the start; "always": never.
        self.stuck_start = stuck_start
        self.stuck = stuck_start is not None
        self.restall_after_m = restall_after_m
        self.teleports_on_route: list[tuple[float, float]] = []  # (arc length, distance off the route) of each later teleport

    def teleport(self, pos, rot_quat=None, reset=False) -> None:
        self.placed.append((tuple(pos), tuple(rot_quat or ())))
        self.s_at_teleport.append(self.s)
        self.place_polls.append(self.polls)
        if len(self.placed) == 1:  # the car is on the first pose; later teleports do not move the fake ...
            self.s = 0.0
        elif self.stuck_start is not None or self.restall_after_m is not None or self.follow_deck_after_m is not None:
            # Follow a later on-route teleport (stuck-start, mid-drive wrong_deck resume, or a mid-route resume).
            # A recover in place (same arc, restall) leaves the car stalled so the second abort can end the segment.
            _i, s, d = lr.project_forward(self.route, pos[0], pos[1], 0, back=0, ahead_m=1e9)
            self.teleports_on_route.append((s, d))
            if d > lr.OFF_ROUTE_M:
                self.parked = True  # a deck pose km away: the car is off the route and at rest
            elif self.restall_after_m is not None and s <= self.s + 5.0:
                pass  # recover in place: stay stalled at the same arc
            else:
                self.s = s
                if self.stuck_start == "until_moved" and s >= 10.0:
                    self.stuck = False
                if self.restall_after_m is not None:
                    self.stall = None
                    self.has_line = False
                    self.stall_at_m = s + self.restall_after_m
        elif self.recover_to == "end":  # ... unless the recover is asked to land it at the route's end
            self.s = self.route.total_m - 10.0
        elif self.recover_to == "off":  # or off the route
            self.parked = True

    def _note_command(self, command: object) -> None:
        super()._note_command(command)
        self.cmd_poll.append(self.polls)
        self.cmd_s.append(self.s)
        if _is_set_line(command):
            self.has_line = True
            self.set_line_polls.append(self.polls)
            return
        if not isinstance(command, str):
            return
        if 'ai.setMode("manual")' in command:
            self.mode = "manual"
            if self.stall == "stalled":
                self.stall = "resumed"  # the AI was told to drive again after the recover
        elif "ai.setMode(" in command and '"span"' in command:
            if self.mode == "manual":  # a hand-back after the route; the first span is the collector's own start-up send
                self.span_poll = self.span_poll or self.polls
            self.mode = "span"

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
        if self.stall_at_m is not None and self.s >= self.stall_at_m:
            if self.restall_after_m is not None:
                # Stay stalled through a recover in place; only a resume teleport re-arms stall_at_m.
                if self.stall != "stalled":
                    self.stall, self.stall_poll = "stalled", self.polls
            elif self.stall is None:
                self.stall, self.stall_poll = "stalled", self.polls  # the AI is reset: only a fresh manual mode restarts it
        if self.stall == "stalled" or self.stuck:
            driven = False
        if self.leave_at_m is not None and self.s >= self.leave_at_m:
            self.parked = True  # off the route, and stopped there (a stuck car is a car at rest)
        if driven and not self.parked:
            # The car brakes (5 m/s2) into the route's end and crawls the last metres before the end tolerance (< 1 m/s):
            # it is not "driving" when it is handed back, which is when the collector's own span send must still take over.
            left = max(0.0, self.route.total_m - lr.END_TOL_M - CRAWL_M - self.s)
            self.v = min(SPEED, max(CRAWL_V, math.sqrt(2.0 * 5.0 * left)))
            self.s = min(self.route.total_m, self.s + self.v * 0.1)
        else:
            self.v = max(0.0, self.v - 0.5)  # 5 m/s2: a stop, not an impact
        self.max_s = max(self.max_s, self.s)
        x, y, dx, dy = self._at(self.s)
        if self.parked:
            x += 120.0
        z = 0.5
        if self.placed:
            # After a teleport, report the placed z so a recover/resume settle can be checked.
            z = float(self.placed[-1][0][2])
        if self.wrong_deck_dz and self.placed:
            # Same xy as the teleport, z = requested z + dz (smoke-c resume: +17 m on an overpass deck).
            x, y, z = self.placed[-1][0][0], self.placed[-1][0][1], self.placed[-1][0][2] + self.wrong_deck_dz
        elif self.wrong_deck_on_place and self.placed and len(self.placed) == self.wrong_deck_on_place:
            z = float(self.placed[-1][0][2]) + 17.0
        if self.follow_deck_after_m is not None and self.s >= self.follow_deck_after_m and len(self.placed) == 1:
            # Mid-drive only (still the start place). Resume teleports clear the lift.
            plan_z = float(self.route.points[0][2]) if self.route is not None else 0.4
            if self.route is not None:
                _i, _s, _d = lr.project_forward(self.route, x, y, 0, back=0, ahead_m=1e9)
                pts, cum = self.route.points, self.route.cum
                if _i < len(pts) - 1:
                    seg = max(1e-6, cum[_i + 1] - cum[_i])
                    t = min(1.0, max(0.0, (_s - cum[_i]) / seg))
                    plan_z = float(pts[_i][2] + (pts[_i + 1][2] - pts[_i][2]) * t)
                else:
                    plan_z = float(pts[-1][2])
            z = plan_z + self.follow_deck_dz
        self.state = {
            "pos": [x, y, z], "dir": [dx, dy, 0.0], "up": [0.0, 0.0, 1.0],
            "electrics": {"wheelspeed": self.v, "steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0},
        }
        done_wait = self.span_poll is not None and self.polls >= self.span_poll + self.tail_polls
        if done_wait or self.polls >= self.max_polls:
            self.stop.write_text("stop", encoding="utf-8")


def run(route: lr.Route, extra: list[str] | None = None, paint=None, **car_kw: object):
    from scripts.gvd_test_env import collect_root

    cars: list[RouteCar] = []
    snap_dz = float(car_kw.pop("snap_dz", 0.0) or 0.0)

    def factory(stop: Path, **_kw: object) -> RouteCar:
        cars.append(RouteCar(stop, route, **car_kw))
        return cars[-1]

    root = sf._COLLECT_ROOTS.enter_context(collect_root())
    plan = root / "route.json"
    plan.write_text(json.dumps(route.to_json()), encoding="utf-8")
    previous = sf._CrashVehicle
    sf._CrashVehicle = factory  # type: ignore[assignment]
    import python.collect.level_session as level_session
    prev_snap = level_session.snap_to_road

    def _snap(bng, near):
        # The harness BeamNG road only runs to y=80; a real recover stays at the car.
        del bng
        if near is None or len(near) < 3:
            return None
        return (float(near[0]), float(near[1]), float(near[2]) + snap_dz), (0.0, 0.0, 0.0, 1.0)

    level_session.snap_to_road = _snap  # type: ignore[assignment]
    try:
        # No --no-excursions, --no-impact-respawn or stall setting of its own: the route clip must turn those off itself.
        code, text, out = sf._run_west_coast(
            paint if paint is not None else sf._Paint(lambda x, y: True), sf._west_pools(),
            ["--route-plan", str(plan), "--min-paint-cov", "0", "--stall-respawn-s", "5"] + list(extra or []), 10**9, root=root,
        )
    finally:
        sf._CrashVehicle = previous  # type: ignore[assignment]
        level_session.snap_to_road = prev_snap  # type: ignore[assignment]
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



def _assert_set_line_dispatched(car, *, z: float) -> None:
    """The real _queue called vehicle.ai.set_line with beamngpy 1.36's signature."""
    assert car.set_line_calls, "vehicle.ai.set_line was not called"
    for line, cling in car.set_line_calls:
        assert cling is True, cling
        assert line, "set_line window is not empty"
        for node in line:
            pos = node["pos"]
            assert isinstance(pos, tuple) and len(pos) == 3, pos
            assert pos[2] == z, f"pos z must be the route height, not hardcoded 0: {pos}"
            assert "speed" in node, node
    assert car.lua_commands, "expected Lua setMode/setSpeed strings"
    assert all(isinstance(c, str) for c in car.lua_commands)
    assert not any(_is_set_line(c) for c in car.lua_commands)


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
    assert "collect abort" not in text and "collect recover" not in text, "the route ends by hand-back, not by a stall abort"
    # the route commands reached the car, while the car was at rest at the start
    lines = [i for i, c in enumerate(car.commands) if _is_set_line(c)]
    assert len(lines) >= 20, len(lines)
    assert car.max_s >= total - lr.END_TOL_M - 3.0, (car.max_s, total)
    assert car.cmd_s[lines[0]] == 0.0, car.cmd_s[lines[0]]  # the first window went out while the car was still at rest
    # the AI mode is re-sent once the car is driving (the segment opens and clears the AI state), well before 80 m
    manual = [i for i, c in enumerate(car.commands) if isinstance(c, str) and 'ai.setMode("manual")' in c]
    assert len(manual) == 2, [car.commands[i] for i in manual]
    assert car.cmd_poll[manual[1]] - car.cmd_poll[manual[0]] <= 12, [car.cmd_poll[i] for i in manual]
    # the plain span send stays off while the route is active, and takes over once it ends
    spans = [i for i, c in enumerate(car.commands) if isinstance(c, str) and 'ai.setMode("span")' in c and i > lines[0]]
    assert spans and min(spans) > max(lines), (spans[:3], lines[-3:])
    assert len(spans) >= 2, "the collector's own span send resumes after the driver's hand-off, even with the car at rest"
    assert not any(_is_set_line(c) for c in car.commands[spans[0]:]), "no route line after the hand-off"
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
    _assert_set_line_dispatched(car, z=0.4)
    print(f"collector route run: {len(lines)} route lines over {total:.0f} m, start pose from the route, one respawn-free segment, hand-off to span, route_driver.json written")


def check_leaving_the_route_hands_back_to_span() -> None:
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, leave_at_m=600.0)
    assert code == 0, (code, text[-2500:])
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "off_route" and summary["far_end_reached"] is False, summary
    assert summary["max_off_route_m"] > 100.0, summary
    lines = [i for i, c in enumerate(car.commands) if _is_set_line(c)]
    spans = [i for i, c in enumerate(car.commands) if isinstance(c, str) and 'ai.setMode("span")' in c and i > lines[0]]
    assert spans and min(spans) > max(lines), "span only after the last route line"
    assert "'state': 'off_route'" in text and len(car.placed) == 1, text[-1500:]
    # the car is parked off the route (at rest): the hand-back comes from the route driver within a second or two,
    # not from the 10 s stationary abort and recover
    assert "collect abort" not in text and "collect recover" not in text, text[-1500:]
    assert car.span_poll is not None
    print(f"collector route run: leaving the route (max {summary['max_off_route_m']} m off) hands the car back to span")


def _spans_after(car: RouteCar) -> list[int]:
    return [i for i, c in enumerate(car.commands) if isinstance(c, str) and 'ai.setMode("span")' in c and car.cmd_poll[i] > car.stall_poll]


def check_recover_to_rest_resends_the_window() -> None:
    """The AI stalls mid-route and the collector recovers the car: at rest, the route window must go out again."""
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, stall_at_m=500.0)
    assert code == 0, (code, text[-1500:])
    assert "collect abort stationary" in text and "collect recover" in text, text[-2500:]
    assert len(car.placed) == 2, car.placed  # the route start, then the recover
    # mode + window re-sent from where the car stopped, after the recover
    again = [i for i, c in enumerate(car.commands) if isinstance(c, str) and 'ai.setMode("manual")' in c and car.cmd_s[i] >= 499.0]
    assert again, [(c[:30], s_) for c, s_ in zip(car.commands, car.cmd_s) if isinstance(c, str) and "setMode" in c]
    assert car.stall == "resumed" and car.max_s >= route.total_m - lr.END_TOL_M - 3.0, (car.stall, car.max_s)
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["far_end_reached"] is True, summary
    print("collector route run: a recover to rest mid-route re-sends mode + window and the route finishes")


def check_finish_or_abandon_at_rest_hands_back_to_span() -> None:
    """The recover leaves the car at rest where the route is already finished (or abandoned): span must be sent."""
    for where, state, stall_m in (("end", "done", 2300.0), ("off", "off_route", 500.0)):
        route = make_route()
        code, text, out, car = _no_excursions_flag_run(route, stall_at_m=stall_m, recover_to=where)
        assert code == 0, (where, code, text[-1500:])
        assert "collect recover" in text and f"'state': '{state}'" in text, (where, text[-2500:])
        summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
        assert summary["state"] == state, (where, summary)
        # the driver's own span commands plus the collector's plain send, which only runs if last_ai was cleared
        spans = _spans_after(car)
        assert len(spans) >= 2, (where, spans)
        assert "collect ai speed=" in text.split("collect recover", 1)[1], (where, text[-1500:])
    print("collector route run: a route finished or abandoned while the car is at rest after a recover gets span, then the plain AI send")


def check_stuck_start_respawns_on_the_route() -> None:
    """The smoke (2026-11-27, dc43ef3): the car sat at the route's start through a stationary abort, a recover and a
    second abort. The segment ended, and the respawn came from the paint-coverage deck: 2 km off the route, so
    the driver went off_route with driven_m = 0 and the clip recorded 10 minutes off the plan. The respawn must be on the route."""
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, stuck_start="until_moved")
    assert code == 0, (code, text[-2500:])
    assert text.count("collect abort stationary") == 2 and "collect segment end west_coast_usa#0 stationary" in text, text[-3000:]
    assert len(car.placed) == 3, car.placed  # the route start, the recover, then the respawn that opens the next segment
    resume = car.teleports_on_route[-1]
    assert resume[1] < 1.0, f"the respawn is {resume[1]:.0f} m off the planned route (arc {resume[0]:.0f} m): a deck pick, not a route pick"
    assert abs(resume[0] - lr.RESUME_NUDGE_M) < 2.0, f"respawn at arc {resume[0]:.1f} m: the nudge past the stall point is {lr.RESUME_NUDGE_M} m"
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["far_end_reached"] is True, summary
    assert summary["resumes"] == 1 and abs(summary["skipped_m"] - lr.RESUME_NUDGE_M) < 1e-6, summary
    assert "collect route resume" in text and "'state': 'off_route'" not in text, text[-2500:]
    print(f"collector route run: stuck at the start twice -> respawn on the route at {resume[0]:.0f} m ({resume[1]:.1f} m off the line), the route finishes")


def check_stuck_route_ends_the_clip_honestly() -> None:
    """The car never moves wherever it is put: every respawn is on the route, and the collector's own segment-fail
    guard ends the clip (3 segments with no recorded row) instead of recording minutes off the plan."""
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, stuck_start="always")
    assert code == 2, (code, text[-2500:])
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["resumes"] == 2 and summary["state"] == "following", summary  # never off_route: the deck was not used
    assert all(d < 1.0 for _s, d in car.teleports_on_route), f"a respawn left the route: {car.teleports_on_route}"
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "segment_fail" and status["stop_code"] == 2, status
    print("collector route run: a car that never moves is put back on the route twice, then the clip stops (exit 2, segment_fail), never off-route")


def check_unresumable_route_ends_the_clip_honestly() -> None:
    """Nothing left to resume on (here: the route-end margin is raised past the route): the clip stops with a named reason."""
    route = make_route()
    saved = lr.RESUME_MIN_LEFT_M
    lr.RESUME_MIN_LEFT_M = 10.0 ** 6
    try:
        code, text, out, car = _no_excursions_flag_run(route, stuck_start="until_moved")
    finally:
        lr.RESUME_MIN_LEFT_M = saved
    assert code == 2, (code, text[-2500:])
    assert len(car.placed) == 2, f"no deck pick after the failed resume: {car.placed}"  # the route start and the recover
    assert "route respawn impossible" in text and "of the route are left" in text, text[-2000:]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "spawn_failed" and "route respawn impossible" in (status.get("stop_detail") or ""), status
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["resumes"] == 0 and summary["state"] != "off_route", summary
    print("collector route run: a route that cannot be resumed stops the clip (exit 2, spawn_failed, 'route respawn impossible'), no deck respawn")


def check_timed_respawn_stays_off_a_route_clip() -> None:
    """--respawn-min teleports to a new deck spawn on its own clock. While the route is being driven that would
    leave it mid-drive; after the route is finished the clip roams and the clock applies again, as before."""
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, extra=["--respawn-min", "0.2"])
    assert code == 0, (code, text[-2500:])
    assert len(car.placed) >= 2, "the clock never fired after the route: this check would not see a mid-route teleport"
    early = [round(s_) for s_ in car.s_at_teleport[1:] if s_ < route.total_m - lr.END_TOL_M - 10.0]
    assert not early, f"a timed respawn moved the car off the route mid-drive (at route metres {early}): {car.placed}"
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["resumes"] == 0 and summary["max_off_route_m"] < 1.0, summary
    print("collector route run: a --respawn-min clock does not teleport a route clip while the route is driven")


def check_low_yield_stays_off_a_route_clip() -> None:
    """A low-yield cut uses force=True and would have called _apply_spawn with source=="spawn", which the
    old guard treated as a route resume: a moving car would be teleported 40 m ahead. The cut must keep
    the car on the route (plan=keep), with no resume."""
    route = make_route()
    # Unpainted labels are invalid, so YieldGuard fires after its 60 s moving window.
    paint = sf._Paint(lambda x, y: False)
    code, text, out, car = _no_excursions_flag_run(route, extra=["--min-live-yield", "0.99"], paint=paint)
    assert code == 0, (code, text[-2500:])
    assert "collect low_yield" in text, text[-3000:]
    mid_route_resumes = [line for line in text.splitlines() if "collect route resume" in line]
    assert not mid_route_resumes, f"low-yield teleported the car on the route: {mid_route_resumes}\n{text[-2500:]}"
    assert "-> keep" in text, text[-2500:]
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["resumes"] == 0 and summary["max_off_route_m"] < 1.0, summary
    assert all(d < 1.0 for _s, d in car.teleports_on_route), car.teleports_on_route
    print("collector route run: a low-yield cut does not teleport a route clip; the route finishes")


def check_recheck_stays_off_a_route_clip() -> None:
    """--recheck-cov-min would teleport to a painted deck road. While the route is driven the recheck
    must keep the car on the (unpainted) route, with no resume."""
    route = make_route()
    # The route is at x=0..250; east deck roads (x>1000) are painted, so a rogue deck pick would leave the route.
    paint = sf._Paint(lambda x, y: x > 1000.0)
    code, text, out, car = _no_excursions_flag_run(
        route, extra=["--min-paint-cov", "0.5", "--recheck-cov-min", "0.2"], paint=paint,
    )
    assert code == 0, (code, text[-2500:])
    assert "collect recheck_cov" in text, text[-3000:]
    mid_route_resumes = [line for line in text.splitlines() if "collect route resume" in line]
    assert not mid_route_resumes, f"recheck teleported the car on the route: {mid_route_resumes}\n{text[-2500:]}"
    assert "-> keep" in text, text[-2500:]
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] == "done" and summary["resumes"] == 0 and summary["max_off_route_m"] < 1.0, summary
    assert all(d < 1.0 for _s, d in car.teleports_on_route), car.teleports_on_route
    early = [round(s_) for s_ in car.s_at_teleport[1:] if s_ < route.total_m - lr.END_TOL_M - 10.0]
    assert not early, f"a recheck teleport moved the car mid-drive (at route metres {early}): {car.placed}"
    print("collector route run: a paint-coverage recheck does not teleport a route clip; the route finishes")


def check_three_resumes_then_the_clip_stops() -> None:
    """MAX_RESUMES is 3 and must be reachable through the collector: the car records rows (so segment_fail
    does not fire after two empty segments), stalls three times, is put back on the route each time, and
    the fourth resume is refused with spawn_failed."""
    route = make_route()
    # A longer plan so four ~20 s stall losses stay under the 20% recover-lost exit.
    code, text, out, car = _no_excursions_flag_run(
        route, extra=["--minutes", "20"], stall_at_m=50.0, restall_after_m=50.0, max_polls=20000,
    )
    assert code == 2, (code, text[-3000:])
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["resumes"] == 3, summary
    assert summary["state"] != "off_route", summary
    assert "route respawn impossible" in text and "already put back on the route" in text, text[-2500:]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "spawn_failed" and "route respawn impossible" in (status.get("stop_detail") or ""), status
    assert all(d < 1.0 for _s, d in car.teleports_on_route), f"a respawn left the route: {car.teleports_on_route}"
    print("collector route run: three on-route resumes, then the fourth is refused (exit 2, spawn_failed), never off-route")


def check_failed_route_resume_placement_stops_the_clip() -> None:
    """The 3rd place_vehicle is the first route resume (start, recover, then resume). If it
    fails, the clip must stop (spawn_failed): resume() already counted that resume, and
    swallowing the error used to let the clip continue and burn another. The first open
    must still settle as source=spawn (not segment)."""
    import python.collect.level_session as level_session

    orig = level_session.place_vehicle
    n = [0]

    def pv(bng, vehicle, pos, rot):
        n[0] += 1
        if n[0] == 3:
            return "session lost: simulated placement failure"
        return orig(bng, vehicle, pos, rot)

    level_session.place_vehicle = pv  # type: ignore[assignment]
    try:
        route = make_route()
        code, text, out, car = _no_excursions_flag_run(route, stuck_start="until_moved")
    finally:
        level_session.place_vehicle = orig  # type: ignore[assignment]
    assert code == 2, (code, text[-3000:])
    assert "legal spawn pose could not be placed" in text and "session lost" in text, text[-2500:]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "spawn_failed" and "session lost" in (status.get("stop_detail") or ""), status
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["resumes"] == 1, summary  # the failed resume was counted; no second resume
    assert "resumes=2" not in text, text[-2500:]
    # First open is source=spawn: N6 (drop spawns>0 from spawn_src) would settle #0 as segment.
    first_settle = [line for line in text.splitlines() if "teleport_settle west_coast_usa#0" in line]
    assert first_settle and any("source=spawn" in line for line in first_settle), first_settle
    assert not any("source=segment" in line for line in first_settle), first_settle
    print("collector route run: a failed resume placement stops the clip (exit 2, spawn_failed), resumes=1, first settle is source=spawn")


def check_wrong_deck_start_does_not_send_set_line() -> None:
    """A route start that settles +17 m above the requested z must not follow or send set_line.

    The existing spawn_failed path handles the error (the clip stops; no retry loop).
    """
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, wrong_deck_dz=17.0)
    assert code == 2, (code, text[-3000:])
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "spawn_failed", status
    detail = (status.get("stop_detail") or "") + text
    assert "could not be placed" in detail or "off the planned z" in detail, detail[-2500:]
    assert not car.set_line_calls, f"set_line must not go out after a wrong-deck settle: {car.set_line_calls}"
    assert car.has_line is False
    print("collector route run: a +17 m wrong-deck start is spawn_failed, no set_line")

def check_wrong_deck_resume_does_not_send_set_line() -> None:
    """The 3rd place_vehicle is the first route resume (source=segment). A +17 m
    settle there must be spawn_failed. A mutant that drops "segment" from the
    reject tuple swallows the error and keeps going (a later resume or set_line).
    """
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(route, stuck_start="until_moved", wrong_deck_on_place=3)
    assert code == 2, (code, text[-3000:])
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "spawn_failed", status
    detail = (status.get("stop_detail") or "") + text
    assert "could not be placed" in detail and "off the planned z" in detail, detail[-2500:]
    assert "source=segment" in text, text[-2500:]
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["resumes"] == 1, summary  # the failed resume was counted; no second resume
    assert len(car.place_polls) >= 3, car.place_polls
    resume_at = car.place_polls[2]
    line_after = [p for p in car.set_line_polls if p >= resume_at]
    assert not line_after, f"set_line went out after a wrong-deck resume: polls {line_after}"
    print("collector route run: a +17 m wrong-deck resume (source=segment) is spawn_failed, resumes=1, no set_line after the place")


def check_recover_wrong_deck_compares_to_plan_z() -> None:
    """Recover snap and settle both +17 m above plan z must fail the recover z-check.

    Comparing settled z to the snap pose (moved) misses this: both are high, |dz|~0.
    The check must use the route's plan z. End the segment (resume path), do not
    follow on the wrong deck, and do not stop as recover_failed.
    """
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(
        route, stall_at_m=50.0, snap_dz=17.0, extra=["--minutes", "20"], max_polls=8000,
    )
    # Recover must log the z failure against the plan (not silently follow).
    rec = [line for line in text.splitlines() if "collect recover" in line and "failed" in line and "off the planned z" in line]
    assert rec, f"recover z-check must fire when snap and settle are both high:\n{text[-3000:]}"
    assert "recover_failed" not in (json.loads((out / "collect.json").read_text(encoding="utf-8")).get("stop_reason") or ""), text[-2000:]
    # After recover reject the car must not be followed on the +17 m deck: no set_line
    # between the recover place (2nd) and a later resume place (3rd), if any.
    assert len(car.place_polls) >= 2, car.place_polls
    rec_at = car.place_polls[1]
    resume_at = car.place_polls[2] if len(car.place_polls) >= 3 else 10**9
    line_between = [p for p in getattr(car, "set_line_polls", []) if rec_at <= p < resume_at]
    assert not line_between, f"set_line on the wrong recover deck: {line_between}"

    # The recover reject itself must end the segment under the abort that was in
    # progress (typically stationary). A swallowed reject would only be caught
    # later as mid-drive wrong_deck.
    lines = text.splitlines()
    failed_i = next(
        (i for i, line in enumerate(lines) if "collect recover" in line and "failed" in line and "off the planned z" in line),
        None,
    )
    assert failed_i is not None
    later_ends = [line for line in lines[failed_i:] if line.startswith("collect segment end")]
    assert later_ends, "recover z-fail must end the segment"
    assert "wrong_deck" not in later_ends[0], later_ends[0]
    print("collector route run: recover snap+settle both +17 m vs plan z ends the segment, no set_line on that deck")


def check_recover_wrong_deck_uses_post_settle_z() -> None:
    """A recover whose snap (pre-settle) is on the plan but the settled z is +17 m must fail.

    A mutant that compares to the pre-settle / moved z misses this and follows.
    """
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(
        route, stall_at_m=50.0, wrong_deck_on_place=2, extra=["--minutes", "20"], max_polls=8000,
    )
    rec = [line for line in text.splitlines() if "collect recover" in line and "failed" in line and "off the planned z" in line]
    assert rec, f"recover must use post-settle z:\n{text[-3000:]}"
    assert "recover_failed" not in (json.loads((out / "collect.json").read_text(encoding="utf-8")).get("stop_reason") or "")

    # The recover reject itself must end the segment under the abort that was in
    # progress (typically stationary). A swallowed reject would only be caught
    # later as mid-drive wrong_deck.
    lines = text.splitlines()
    failed_i = next(
        (i for i, line in enumerate(lines) if "collect recover" in line and "failed" in line and "off the planned z" in line),
        None,
    )
    assert failed_i is not None
    later_ends = [line for line in lines[failed_i:] if line.startswith("collect segment end")]
    assert later_ends, "recover z-fail must end the segment"
    assert "wrong_deck" not in later_ends[0], later_ends[0]
    print("collector route run: recover on-plan snap + +17 m settle is a failed recover (post-settle z)")


def check_mid_drive_wrong_deck_ends_the_segment() -> None:
    """A mid-drive |car z − plan z| jump (smoke-c +7.8 m) must stop set_line and end the
    segment so the next open is a route resume, not off_route or recover_failed.

    Exactly one wrong_deck segment end, then one resume, then the route continues
    past the jump with no spawn_failed (kills G6: wrong_deck does not end the
    segment; G5: deck_error left set, so every resume re-aborts to spawn_failed).
    """
    route = make_route()
    code, text, out, car = _no_excursions_flag_run(
        route, follow_deck_after_m=80.0, follow_deck_dz=7.8, extra=["--minutes", "20"], max_polls=8000,
    )
    assert "off the planned z" in text, text[-3000:]
    follow_lines = [line for line in text.splitlines() if "following" in line and "off the planned z" in line]
    assert follow_lines or "wrong_deck" in text, text[-3000:]
    lines = text.splitlines()
    ends = [line for line in lines if line.startswith("collect segment end") and "wrong_deck" in line]
    assert len(ends) == 1, ends
    # G6 drops wrong_deck as an immediate segment end, so the first abort falls
    # into recover; the recover then fails z and still ends as wrong_deck. The
    # first abort itself must end the segment, with no recover in between.
    follow_i = next(i for i, line in enumerate(lines) if "following" in line and "off the planned z" in line)
    end_i = next(i for i, line in enumerate(lines) if line.startswith("collect segment end") and "wrong_deck" in line)
    between = lines[follow_i:end_i + 1]
    assert any("collect abort wrong_deck" in line for line in between), between
    assert not any("collect recover" in line for line in between), between
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status.get("stop_reason") not in {"spawn_failed", "recover_failed"}, status
    summary = json.loads((out / "route_driver.json").read_text(encoding="utf-8"))
    assert summary["state"] != "off_route", summary
    assert summary["resumes"] == 1, summary
    assert summary["progress_m"] >= 80.0 + lr.RESUME_NUDGE_M - 1.0, summary
    resume_lines = [line for line in text.splitlines() if "collect route resume" in line]
    assert len(resume_lines) == 1, resume_lines
    assert len(car.place_polls) >= 2, car.place_polls
    resume_at = car.place_polls[1]
    line_after = [p for p in car.set_line_polls if p >= resume_at]
    assert line_after, f"set_line must go out after the resume: places={car.place_polls} lines={car.set_line_polls}"
    print("collector route run: mid-drive +7.8 m ends the segment once, resumes once, and continues past the jump")


def check_failed_route_place_message_surfaces_low_yield_and_segment() -> None:
    """A wrong-deck reject for source=low_yield or source=segment is an error, not None.

    The collector must not swallow it while a route is active (caller would keep driving).
    low_yield still must not teleport an on-plan car (check_low_yield_stays_off_a_route_clip).
    """
    import scripts.collect_scene_lanes as collect

    assert "segment" in collect.ROUTE_PLACE_FAIL_SOURCES
    assert "low_yield" in collect.ROUTE_PLACE_FAIL_SOURCES
    err = "route place landed +17.0 m off the planned z (limit 5 m)"
    for source in ("spawn", "segment", "recheck", "low_yield"):
        msg = collect.failed_route_place_message(source, "west_coast_usa", err)
        assert msg, f"{source} must surface the reject"
        assert "could not be placed" in msg and "17" in msg, (source, msg)
    # stall/impact mid-run still keep the current pose when they somehow fail a place
    assert collect.failed_route_place_message("stall", "west_coast_usa", err) is None
    assert collect.failed_route_place_message("impact", "west_coast_usa", err) is None
    print("failed_route_place_message: spawn/segment/recheck/low_yield surface the error; stall/impact do not")



def main() -> None:
    check_route_runs_through_the_collector()
    check_leaving_the_route_hands_back_to_span()
    check_recover_to_rest_resends_the_window()
    check_finish_or_abandon_at_rest_hands_back_to_span()
    check_stuck_start_respawns_on_the_route()
    check_stuck_route_ends_the_clip_honestly()
    check_unresumable_route_ends_the_clip_honestly()
    check_timed_respawn_stays_off_a_route_clip()
    check_low_yield_stays_off_a_route_clip()
    check_recheck_stays_off_a_route_clip()
    check_three_resumes_then_the_clip_stops()
    check_failed_route_resume_placement_stops_the_clip()
    check_wrong_deck_start_does_not_send_set_line()
    check_wrong_deck_resume_does_not_send_set_line()
    check_recover_wrong_deck_compares_to_plan_z()
    check_recover_wrong_deck_uses_post_settle_z()
    check_mid_drive_wrong_deck_ends_the_segment()
    check_failed_route_place_message_surfaces_low_yield_and_segment()
    print("test_collect_route_drive: OK")


if __name__ == "__main__":
    main()
