#!/usr/bin/env python3
"""Long-clip route planner, driver and measure (python/collect/long_route.py), offline.

Synthetic networks check the planner exactly (one-way, junction links, dirt,
direction); the bundled west_coast_usa / italy dumps check it on real roads
(legal spawn pose, direction mix, different loops per seed). The driver runs
against a fake vehicle that records the Lua it is sent and moves along it.

    py -3 scripts/test_long_route.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.collect import long_route as lr  # noqa: E402
from python.collect.spawn_mix import CITY, HIGHWAY, classify_roads, load_nav_roads, spawn_roads_path  # noqa: E402


def road(rid, pts, *, oneway=False, flip=False, lanes=2, width=8.0, material=""):
    pts = [(float(x), float(y), 0.0) for x, y in pts]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    left, right = (0, lanes) if oneway else (lanes // 2 or 1, lanes // 2 or 1)
    return {
        "id": str(rid), "pts": pts, "widths": [width] * len(pts), "lanes": left + right,
        "lanes_left": left, "lanes_right": right, "length": length, "oneway": oneway, "flip": flip,
        "auto_lanes": False, "width": width, "material": material,
    }


def line(x0, y0, x1, y1, step=100.0):
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) // step))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]


def loop_network(*, dirt: bool, mirror: bool = False, ramp_far: bool = False):
    """Highway y=0 from x=0 to x=3000 (one-way east), a city loop north and, if ``dirt``, a dirt loop south."""
    sx = -1.0 if mirror else 1.0
    def P(pts):
        return [(sx * x, y) for x, y in pts]
    roads = [road("H1", P(line(0, 0, 3000, 0)), oneway=True, lanes=2, width=8.0)]
    roads += [
        road("C1", P(line(3000, 10, 3000, 1500)), width=7.0),
        road("C2", P(line(3000, 1500, 0, 1500)), width=7.0),
        road("C3", P(line(0, 1500, 0, 10)), width=7.0),
    ]
    if dirt:
        mat = "DirtRoad_variation_01"
        roads += [
            road("D1", P(line(3000, -10, 3000, -1500)), width=6.0, material=mat),
            road("D2", P(line(3000, -1500, 0, -1500)), width=6.0, material=mat),
            road("D3", P(line(0, -1500, 0, -10)), width=6.0, material=mat),
        ]
    return roads


def route_of(roads, *, seed=1, direction=0, check_start=True):
    g = lr.build_graph(roads)
    if not check_start:
        g.pose_legal = lambda *_a, **_k: True  # type: ignore[method-assign]
    return g, lr.plan_highway_route(g, map_name="synthetic", seed=seed, direction=direction)


def check_dirt_material_words() -> None:
    for m in ("DirtRoad_variation_01", "mud_tiretracks", "road_gravel", "Dirt_road_variation_03", "dirt_tiretracks", "m_dirt_variation_04"):
        assert lr.is_dirt_material(m), m
    for m in ("road_invisible", "AsphaltRoad_variation_two_way", "road_asphalt_light", "DefaultDecalRoadMaterial", "", None):
        assert not lr.is_dirt_material(m), m
    print("dirt material: dirt / mud / gravel / tire tracks are unpaved; asphalt and road_invisible are not")


def check_graph_directions_and_links() -> None:
    g = lr.build_graph(loop_network(dirt=False))
    h = [i for i, r in enumerate(g.roads) if r["id"] == "H1"][0]
    a, b = g.node_of[(h, 0)], g.node_of[(h, 1)]
    assert any(to == b for to, _l, _r in g.adj[a]) and not any(to == a for to, _l, _r in g.adj[b]), "one-way east only"
    flipped = loop_network(dirt=False)
    flipped[0]["flip"] = True
    gf = lr.build_graph(flipped)
    hf = [i for i, r in enumerate(gf.roads) if r["id"] == "H1"][0]
    a, b = gf.node_of[(hf, 0)], gf.node_of[(hf, 1)]
    assert any(to == a for to, _l, _r in gf.adj[b]) and not any(to == b for to, _l, _r in gf.adj[a]), "flipDirection reverses a one-way"
    c1 = [i for i, r in enumerate(g.roads) if r["id"] == "C1"][0]
    far = g.node_of[(h, len(g.roads[h]["pts"]) - 1)]
    c1_start = g.node_of[(c1, 0)]
    assert any(to == c1_start and ri == -1 for to, _l, ri in g.adj[far]), "road ends 10 m apart join"
    # 30 m away does not join; a 6 m bridge (dz) does not join.
    far_roads = loop_network(dirt=False)
    far_roads[1] = road("C1", line(3000, 30, 3000, 1500), width=7.0)
    g2 = lr.build_graph(far_roads)
    far2 = g2.node_of[(0, len(g2.roads[0]["pts"]) - 1)]
    assert not any(ri == -1 for _t, _l, ri in g2.adj[far2]), "30 m apart: no join"
    hi = loop_network(dirt=False)
    hi[1]["pts"][0] = (3000.0, 10.0, 6.0)
    g3 = lr.build_graph(hi)
    assert not any(ri == -1 for _t, _l, ri in g3.adj[g3.node_of[(0, len(g3.roads[0]["pts"]) - 1)]]), "6 m above: no join"
    print("graph: one-way, flipDirection, 15 m junction join, 30 m gap and 6 m bridge do not join")


def check_route_shape_synthetic() -> None:
    _g, r = route_of(loop_network(dirt=False))
    assert r is not None
    # Starts at the west end of the highway, in the right lane, facing east.
    assert abs(r.points[0][0] - 0.0) < 2.0 and r.points[0][1] < 0.0 and abs(r.points[0][1]) < 6.0, r.points[0]
    (px, py, _pz), rot = r.start_pose()
    yaw = math.atan2(2 * rot[3] * rot[2], 1 - 2 * rot[2] ** 2)
    assert abs(yaw) < 0.2, yaw
    fx, fy, _ = r.points[r.far_end_index]
    assert abs(fx - 3000.0) < 2.0, (fx, fy)
    assert 2900.0 < r.highway_m < 3100.0, r.highway_m
    assert set(r.kinds[: r.far_end_index + 1]) == {HIGHWAY}, "highway leg is highway only"
    ret = r.kinds[r.far_end_index + 1 :]
    assert ret, ret
    assert HIGHWAY not in ret[:-2], ("return leg avoids highway", ret[-4:])
    assert r.return_m > 1500.0 and r.dirt_m == 0.0 and r.city_m > 3000.0, (r.return_m, r.dirt_m, r.city_m)
    # The return loops back to the start end of the highway.
    ex, ey, _ = r.points[-1]
    assert math.hypot(ex - 0.0, ey) < 40.0, (ex, ey)
    assert r.legs == ("highway", "city", "highway"), r.legs
    assert r.direction == 0 and r.bearing_deg() > 80.0 and r.bearing_deg() < 100.0
    print("synthetic route: starts at the west highway end facing east, drives to the east end, returns by the city loop to the start")


def check_dirt_is_used_when_present_and_omitted_when_not() -> None:
    for seed in range(6):
        _g, r = route_of(loop_network(dirt=True), seed=seed)
        assert r is not None and r.dirt_m > 2500.0, (seed, r and r.dirt_m)
        assert r.legs == ("highway", "dirt", "highway"), r.legs
        assert "dirt" in r.kinds
    _g, r = route_of(loop_network(dirt=False), seed=1)
    assert r.dirt_m == 0.0 and "dirt" not in r.kinds and "dirt" not in r.legs
    print("dirt: the return leg takes the unpaved loop when the road material says so; no dirt leg is claimed otherwise")


def check_direction_follows_bearing() -> None:
    _g, east = route_of(loop_network(dirt=False))
    _g, west = route_of(loop_network(dirt=False, mirror=True), direction=1)
    assert east.direction == 0 and west.direction == 1, (east.direction, west.direction)
    assert abs(((west.bearing_deg() - east.bearing_deg()) % 360.0) - 180.0) < 10.0
    _g, forced = route_of(loop_network(dirt=False, mirror=True), direction=0)
    assert forced.direction == 1 and forced.requested_direction == 0, "a map that only runs one way reports what it got"
    print("direction: 0 = eastbound half of the compass, 1 = westbound; a one-way-only map reports the direction it got")


def check_no_highway_gives_no_route() -> None:
    roads = [r for r in loop_network(dirt=True) if r["id"] != "H1"]
    g = lr.build_graph(roads)
    assert lr.plan_highway_route(g, map_name="x", seed=1, direction=0) is None
    print("no highway: no route (the clip falls back to span roaming)")


def check_json_roundtrip_and_determinism() -> None:
    _g, a = route_of(loop_network(dirt=True), seed=4)
    _g, b = route_of(loop_network(dirt=True), seed=4)
    assert a.points == b.points and a.kinds == b.kinds
    back = lr.Route.from_json(__import__("json").loads(__import__("json").dumps(a.to_json())))
    assert len(back.points) == len(a.points) and abs(back.total_m - a.total_m) < 5.0
    assert back.far_end_index == a.far_end_index and back.legs == a.legs and back.direction == a.direction
    print("route: seeded (same inputs, same route) and survives a JSON round trip")


# ------------------------------------------------------------------ real maps


def _real(map_name):
    roads = load_nav_roads(spawn_roads_path(map_name))
    return lr.build_graph(roads)


def check_real_maps_routes() -> None:
    for map_name in ("west_coast_usa", "italy"):
        g = _real(map_name)
        routes = {}
        for seed in range(6):
            for d in (0, 1):
                r = lr.plan_highway_route(g, map_name=map_name, seed=seed, direction=d)
                assert r is not None, (map_name, seed, d)
                routes[(seed, d)] = r
                assert r.highway_m >= lr.MIN_HIGHWAY_M, r.highway_m
                assert r.return_m >= 1500.0, (map_name, seed, d, r.return_m)
                assert g.pose_legal(*r.start_pose()), "spawn pose is on the right side of the road"
                assert HIGHWAY not in r.kinds[r.far_end_index + 1 : -2], "return avoids highway roads"
                assert r.direction == d, (map_name, seed, d, r.direction)
        shapes = {tuple(round(v) for p in r.points[::40] for v in p[:2]) for r in routes.values()}
        assert len(shapes) >= 10, (map_name, len(shapes))
        returns = {tuple(round(v) for p in r.points[r.far_end_index :: 30] for v in p[:2]) for r in routes.values()}
        assert len(returns) >= 8, ("different loop back each time", map_name, len(returns))
        e = [routes[(s, 0)].bearing_deg() for s in range(6)]
        w = [routes[(s, 1)].bearing_deg() for s in range(6)]
        assert all(b < 180.0 for b in e) and all(b >= 180.0 for b in w), (e, w)
    print("real maps: west_coast_usa and italy plan a legal highway end-to-end route and a different return loop per seed, direction honoured")


def check_west_coast_uses_gravel() -> None:
    g = _real("west_coast_usa")
    dirt = sum(
        lr.plan_highway_route(g, map_name="west_coast_usa", seed=s, direction=s % 2).dirt_m for s in range(8)
    )
    assert dirt > 1000.0, dirt
    gi = _real("italy")
    assert sum(lr.plan_highway_route(gi, map_name="italy", seed=s, direction=s % 2).dirt_m for s in range(4)) == 0.0, \
        "italy's dump has no unpaved material, so no dirt leg is claimed"
    print(f"real dirt: west_coast_usa gravel is used on the return leg ({dirt:.0f} m over 8 routes); italy's dump has none and claims none")


# ------------------------------------------------------------------ driver / fake vehicle

def is_set_line_cmd(cmd: object) -> bool:
    """The structured command the collector dispatches to ``vehicle.ai.set_line``."""
    return (
        isinstance(cmd, tuple)
        and len(cmd) == 3
        and cmd[0] == "set_line"
        and isinstance(cmd[1], list)
        and isinstance(cmd[2], bool)
    )


def set_line_payload(cmd: object) -> tuple[list[dict], bool]:
    assert is_set_line_cmd(cmd), cmd
    line, cling = cmd[1], cmd[2]
    return line, cling


def set_line_points(cmd: object) -> list[tuple[float, float, float]]:
    line, cling = set_line_payload(cmd)
    assert cling is True, cling
    out: list[tuple[float, float, float]] = []
    for node in line:
        pos = node["pos"]
        assert isinstance(pos, tuple) and len(pos) == 3, pos
        out.append((float(pos[0]), float(pos[1]), float(pos[2])))
        assert "speed" in node
    return out


def assert_no_ai_setline(cmds: list[object]) -> None:
    for cmd in cmds:
        text = cmd if isinstance(cmd, str) else repr(cmd)
        assert "ai.setLine(" not in text, cmd


class FakeVehicle:
    """Records the commands the driver sends, like the collector's ``_queue(vehicle, cmd)``."""

    def __init__(self):
        self.sent: list[object] = []
        self.mode = "span"
        self.line: list[tuple[float, float, float]] = []

    def queue(self, cmd: object) -> None:
        self.sent.append(cmd)
        if isinstance(cmd, str) and cmd.startswith("ai.setMode("):
            self.mode = cmd.split('"')[1]
        if is_set_line_cmd(cmd):
            self.line = set_line_points(cmd)


def drive(route, vehicle, driver, *, step_m=6.0, speed=14.0, start_s=0.0, stop_s=None, jitter=0.0):
    """Move a car along ``route`` and call the driver each tick."""
    s = start_s
    end = route.total_m if stop_s is None else stop_s
    ticks = 0
    while s <= end:
        i = max(0, min(len(route.points) - 2, next((k for k in range(len(route.cum) - 1) if route.cum[k + 1] >= s), len(route.points) - 2)))
        t = (s - route.cum[i]) / max(1e-6, route.cum[i + 1] - route.cum[i])
        x = route.points[i][0] + t * (route.points[i + 1][0] - route.points[i][0]) + jitter
        y = route.points[i][1] + t * (route.points[i + 1][1] - route.points[i][1])
        for cmd in driver.commands(x, y, speed):
            vehicle.queue(cmd)
        s += step_m
        ticks += 1
    return ticks



def check_driver_uses_set_line_api() -> None:
    """No ``ai.setLine`` Lua (it does not exist on Tech 0.39). The window goes out through set_line."""
    from dataclasses import replace

    _g, route = route_of(loop_network(dirt=False), seed=2)
    # Synthetic roads sit at z=0; lift the planned line so a helper that still hardcodes z=0 fails.
    pts = tuple((p[0], p[1], 78.74 + i * 0.05) for i, p in enumerate(route.points))
    route = replace(route, points=pts, cum=route.cum)
    drv = lr.RouteDriver(route)
    x, y, z0 = route.points[0]
    cmds = drv.commands(x, y, 20.0)
    assert_no_ai_setline(cmds)
    assert cmds[0] == 'ai.setMode("manual")'
    assert cmds[1] == 'ai.driveInLane("off")'
    assert cmds[2] == 'ai.setSpeedMode("limit")'
    assert cmds[3] == "ai.setSpeed(20.0)"
    assert is_set_line_cmd(cmds[4]) and len(cmds) == 5, cmds
    line, cling = set_line_payload(cmds[4])
    assert cling is True
    assert line, "set_line window is not empty"
    for node in line:
        pos = node["pos"]
        assert isinstance(pos, tuple) and len(pos) == 3, pos
        assert pos[2] != 0.0, f"pos z must be the route height, not hardcoded 0: {pos}"
        assert node["speed"] == 20.0, node
    assert abs(line[0]["pos"][0] - x) < 1e-6 and abs(line[0]["pos"][1] - y) < 1e-6
    assert abs(line[0]["pos"][2] - z0) < 1e-6, (line[0]["pos"], z0)
    print("driver: set_line uses 3-tuple pos with real z, speed, cling=True; no ai.setLine Lua")


def check_driver_sends_the_route_in_order() -> None:
    _g, route = route_of(loop_network(dirt=True), seed=2)
    veh, drv = FakeVehicle(), lr.RouteDriver(route)
    drive(route, veh, drv)
    assert_no_ai_setline(veh.sent)
    lines = [c for c in veh.sent if is_set_line_cmd(c)]
    assert lines and veh.sent[0] == 'ai.setMode("manual")' and veh.sent[1] == 'ai.driveInLane("off")', veh.sent[:3]
    assert veh.sent[2] == 'ai.setSpeedMode("limit")' and veh.sent[3].startswith("ai.setSpeed(14.0)"), veh.sent[:4]
    assert sum(1 for c in veh.sent if c == 'ai.setMode("manual")') == 1, "manual mode is set once; later sends only replace the line"
    # Every line starts at the car's place on the route, runs forward along it, and is at most the window long.
    prev_first = -1.0
    for cmd in lines:
        pts = set_line_points(cmd)
        i0, s0, d0 = lr.project_forward(route, pts[0][0], pts[0][1], 0, ahead_m=1e9)
        assert d0 < 3.0, "line starts on the route"
        assert s0 >= prev_first - 1e-6, "lines move forward, never back"
        prev_first = s0
        i1, s1, d1 = lr.project_forward(route, pts[-1][0], pts[-1][1], i0, ahead_m=1e9)
        assert d1 < 3.0 and s1 > s0 and s1 - s0 <= 420.0, (s0, s1)
        if s0 < route.total_m - 700.0:
            assert s1 - s0 >= 300.0, (s0, s1)  # a full 300 m window (plus the point past it) until the route's end
        assert all(a[0] != b[0] or a[1] != b[1] for a, b in zip(pts, pts[1:]))
    # The first line is the highway, a later one is the return leg (off the highway).
    first_pts = set_line_points(lines[0])
    assert abs(first_pts[0][0]) < 5.0 and first_pts[0][1] < 0.0 and abs(first_pts[-1][1]) < 6.0, first_pts[:2]
    assert any(max(abs(p[1]) for p in set_line_points(c)) > 500.0 for c in lines), "return leg is sent"
    assert drv.far_end_reached and drv.state == "done", drv.summary()
    assert veh.sent[-4:] == lr.span_commands(14.0) and veh.mode == "span", "end of the route hands back to span"
    assert drv.commands(0.0, 0.0, 14.0) == [], "nothing more after done"
    assert 4 <= drv.lines_sent <= 120, drv.lines_sent
    print(f"driver (fake vehicle): {drv.lines_sent} set_line windows, in order, highway then return, span at the end")


def check_driver_resend_rules() -> None:
    _g, route = route_of(loop_network(dirt=False), seed=2)
    veh, drv = FakeVehicle(), lr.RouteDriver(route)
    x, y = route.points[0][0], route.points[0][1]
    first = drv.commands(x, y, 14.0)
    assert_no_ai_setline(first)
    assert any(is_set_line_cmd(c) for c in first)
    assert drv.commands(x, y, 14.0) == [], "same place and speed: nothing re-sent"
    assert drv.commands(x, y, 14.2) == [], "speed change under 0.5 m/s: nothing"
    again = drv.commands(x, y, 15.0)
    assert_no_ai_setline(again)
    assert again and again[0] == "ai.setSpeed(15.0)" and any(is_set_line_cmd(c) for c in again), again[:1]
    assert 'ai.setMode("manual")' not in again, "the mode is not re-sent"
    # Far enough along: a fresh window.
    px, py, _ = route.points[next(i for i, c in enumerate(route.cum) if c > lr.RESEND_M + 30)]
    later = drv.commands(px, py, 15.0)
    assert_no_ai_setline(later)
    assert any(is_set_line_cmd(c) for c in later)
    print("driver: re-sends on 80 m of progress or a 0.5 m/s speed change, never re-sets the mode")


def check_driver_speed_caps() -> None:
    _g, route = route_of(loop_network(dirt=True), seed=2)
    # Dirt and city take less than the highway's 25 m/s; the highway keeps it.
    drv = lr.RouteDriver(route)
    veh = FakeVehicle()
    drive(route, veh, drv, speed=25.0)
    speeds = [float(c[len("ai.setSpeed("):-1]) for c in veh.sent if isinstance(c, str) and c.startswith("ai.setSpeed(")]
    assert speeds[0] == 25.0 and max(speeds) == 25.0, speeds
    assert 11.0 in speeds and min(speeds) == 11.0, speeds
    assert all(sp in (25.0, 14.0, 11.0) for sp in speeds), sorted(set(speeds))
    # The set_line nodes carry the capped speed too.
    node_speeds = {float(n["speed"]) for c in veh.sent if is_set_line_cmd(c) for n in set_line_payload(c)[0]}
    assert node_speeds == {25.0, 11.0}, node_speeds  # this route is highway then dirt

    # A city return (no dirt): city road is capped at 14 m/s, written as a literal here, not the constant.
    _g, city_route = route_of(loop_network(dirt=False), seed=2)
    assert CITY in city_route.kinds and "dirt" not in city_route.kinds, set(city_route.kinds)
    veh = FakeVehicle()
    drive(city_route, veh, lr.RouteDriver(city_route), speed=25.0)
    speeds = [float(c[len("ai.setSpeed("):-1]) for c in veh.sent if isinstance(c, str) and c.startswith("ai.setSpeed(")]
    assert max(speeds) == 25.0 and 14.0 in speeds and set(speeds) == {25.0, 14.0}, sorted(set(speeds))
    nodes = {float(n["speed"]) for c in veh.sent if is_set_line_cmd(c) for n in set_line_payload(c)[0]}
    assert nodes == {25.0, 14.0}, nodes

    print("driver: speeds are capped by road kind (dirt 11, city 14 m/s); the highway keeps the clip speed")


def check_collector_hooks() -> None:
    import json
    import tempfile

    _g, route = route_of(loop_network(dirt=True), seed=2)
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "route.json"
        good.write_text(json.dumps(route.to_json()), encoding="utf-8")
        again = lr.load_route_plan(good, ["synthetic"])
        assert len(again.points) == len(route.points) and max(abs(a - b) for p, q in zip(again.points, route.points) for a, b in zip(p, q)) < 0.01 and again.far_end_index == route.far_end_index and again.map == "synthetic"
        for maps, why in ((["west_coast_usa"], "other map"), (["synthetic", "italy"], "two maps"), ([], "no map")):
            try:
                lr.load_route_plan(good, maps)
            except ValueError as exc:
                assert "route plan is for map synthetic" in str(exc), exc
            else:
                raise AssertionError(f"accepted {why}")
        for name, text in (("missing.json", None), ("junk.json", "{not json"), ("empty.json", "{}")):
            path = Path(td) / name
            if text is not None:
                path.write_text(text, encoding="utf-8")
            try:
                lr.load_route_plan(path, ["synthetic"])
            except ValueError as exc:
                assert "unreadable" in str(exc), exc
            else:
                raise AssertionError(f"accepted {name}")
        # The driver summary file is replaced in place and parses.
        drv = lr.RouteDriver(route)
        drv.commands(route.points[0][0], route.points[0][1], 20.0)
        f = lr.write_driver_summary(td, drv)
        assert f.name == "route_driver.json" and json.loads(f.read_text(encoding="utf-8")) == drv.summary()
        drv.commands(route.points[0][0], route.points[0][1], 25.0)
        lr.write_driver_summary(td, drv)
        assert json.loads(f.read_text(encoding="utf-8"))["lines_sent"] == 2 and not list(Path(td).glob("*.tmp"))

    # First spawn: the route's own start pose on its highway road, a highway pick.
    pick = lr.route_spawn_pick(route)
    pos, rot = route.start_pose()
    assert pick.kind == HIGHWAY and pick.pos == pos and pick.rot == rot and pick.road_id == route.start_road
    assert pick.speed == (20.0, 30.0) and lr.route_spawn_pick(route, (8.0, 14.0)).speed == (8.0, 14.0)

    # One tick of driving: the driver owns the AI while active, the route's end hands back, a spawn reset re-sends.
    drv = lr.RouteDriver(route)
    sent: list[object] = []
    x, y = route.points[0][0], route.points[0][1]
    assert lr.drive_tick(drv, [x, y, 0.0], 20.0, sent.append) is True
    assert_no_ai_setline(sent)
    assert sent[0] == 'ai.setMode("manual")' and any(is_set_line_cmd(c) for c in sent), sent[:2]
    n = len(sent)
    assert lr.drive_tick(drv, [x, y, 0.0], 20.0, sent.append) is True and len(sent) == n, "nothing re-sent for the same place"
    assert lr.drive_tick(drv, None, 20.0, sent.append) is True and len(sent) == n, "no position: wait"
    drv.reset_send()
    assert lr.drive_tick(drv, [x, y, 0.0], 20.0, sent.append) is True and sent[n] == 'ai.setMode("manual")', "after a reset the mode and window are sent again"
    ex, ey = route.points[-1][0], route.points[-1][1]
    drv2 = lr.RouteDriver(route)
    drv2.progress_m = route.total_m
    drv2.index = len(route.points) - 2
    tail: list[str] = []
    assert lr.drive_tick(drv2, [ex, ey, 0.0], 20.0, tail.append) is True and tail == lr.span_commands(20.0)
    assert not drv2.active and lr.drive_tick(drv2, [ex, ey, 0.0], 20.0, tail.append) is False and tail == lr.span_commands(20.0)
    print("collector hooks: plan loads only for its one map, first spawn is the route start, tick sends/hands back/re-sends")


def check_route_length_fits_the_clip() -> None:
    """total_m is bounded by the clip's driving budget (duration x 11 m/s): at most 1.25x (hard), and 0.4x or more (it prefers 0.6x, but honouring the requested direction comes first)."""
    fills = cases = 0
    for map_name in ("west_coast_usa", "italy"):
        g = _real(map_name)
        for dur in (300.0, 600.0, 900.0):
            for seed in range(4):
                for d in (0, 1):
                    r = lr.plan_highway_route(g, map_name=map_name, seed=seed, direction=d, duration_s=dur)
                    assert r is not None, (map_name, dur, seed, d)
                    ratio = r.total_m / (dur * 11.0)
                    assert 0.4 <= ratio <= 1.25 + 1e-9, (map_name, dur, seed, d, round(ratio, 2))
                    fills += ratio >= 0.6
                    cases += 1
    # The planner prefers a route that fills the clip (>= 0.6x): 45 of these 48 do; the lower window is what does it.
    assert cases == 48 and fills >= 44, (fills, cases)
    # A clip too short to hold the only highway is not given an overlong route.
    _g, r = route_of(loop_network(dirt=True), seed=1)
    assert r is not None and r.total_m <= 1.25 * 750.0 * 11.0
    g = lr.build_graph(loop_network(dirt=True))
    for dur in (60.0, 200.0):
        assert lr.plan_highway_route(g, map_name="synthetic", seed=1, direction=0, duration_s=dur) is None, dur
    print("route length: 0.4-1.25x (never over) the clip budget on both real maps; a clip too short for the highway gets no route")


def check_length_cap_and_window_pinned() -> None:
    """The cap (1.25x the budget) and the preferred window (>= 0.6x) as literals, and as behaviour at their edges.

    The synthetic network's planner cannot reach 1.25x on its own, so ``_assemble`` is wrapped to hand the planner
    candidates of an exact length ratio to the budget (duration x 11 m/s).
    """
    import dataclasses

    assert (lr.BUDGET_MAX_FRAC, lr.BUDGET_MIN_FRAC, lr.AVG_SPEED_MPS) == (1.25, 0.6, 11.0)
    g = lr.build_graph(loop_network(dirt=False))
    dur = 750.0
    budget = dur * 11.0
    real = lr._assemble

    def plan(ratios):
        calls = []

        def fake(*a, **k):
            r = real(*a, **k)
            ratio = ratios[min(len(calls), len(ratios) - 1)]
            calls.append(ratio)
            kk = ratio * budget / r.total_m
            return dataclasses.replace(r, cum=tuple(c * kk for c in r.cum))

        lr._assemble = fake  # type: ignore[assignment]
        try:
            out = lr.plan_highway_route(g, map_name="synthetic", seed=1, direction=0, duration_s=dur)
        finally:
            lr._assemble = real  # type: ignore[assignment]
        return out, calls

    r, _ = plan([1.2])
    assert r is not None and abs(r.total_m / budget - 1.2) < 1e-6
    r, _ = plan([1.245])
    assert r is not None and abs(r.total_m / budget - 1.245) < 1e-6
    r, calls = plan([1.255])  # just over the cap: refused in every tier, however many candidates are tried
    assert r is None and len(calls) >= 3, (r, calls)
    r, _ = plan([1.27])
    assert r is None
    # The window: a candidate at 0.7x is taken in the first tier, ahead of a later 1.0x one.
    r, _ = plan([0.7, 1.0])
    assert r is not None and abs(r.total_m / budget - 0.7) < 1e-6, r and r.total_m / budget
    # Just above 0.6x (0.62x) must also take the first tier: pins the window against a 0.65x compare.
    r, _ = plan([0.62, 1.0])
    assert r is not None and abs(r.total_m / budget - 0.62) < 1e-6, r and r.total_m / budget
    # Under 0.6x the window refuses it, the second tier takes it (a route shorter than the clip is allowed).
    r, calls = plan([0.55])
    assert r is not None and abs(r.total_m / budget - 0.55) < 1e-6 and len(calls) >= 2
    # Just under 0.6x (0.58x) must also miss the first tier.
    r, calls = plan([0.58])
    assert r is not None and abs(r.total_m / budget - 0.58) < 1e-6 and len(calls) >= 2
    r, _ = plan([0.55, 0.65])
    assert r is not None and abs(r.total_m / budget - 0.65) < 1e-6, r and r.total_m / budget  # the window's pick wins
    # The last tier: a map whose only loop back is short (< 1500 m) still gets a route (honestly short), not None.
    short = lr.build_graph([road("H1", line(0, 0, 900, 0), oneway=True, lanes=2, width=8.0),
                            road("C1", line(900, 10, 900, 200), width=7.0), road("C2", line(900, 200, 0, 200), width=7.0),
                            road("C3", line(0, 200, 0, 10), width=7.0)])
    r = lr.plan_highway_route(short, map_name="synthetic", seed=1, direction=0)
    assert r is not None and 1000.0 < r.return_m < lr.MIN_RETURN_M and r.highway_m >= 800.0, r
    print("length cap 1.25x and window 0.6x pinned (literals, edge candidates, last tier for short loops)")


def check_short_highway_gets_no_route() -> None:
    """A highway under 800 m is not a long highway: no route, however good the loop back is."""
    def net(hw_m):
        return [road("H1", line(0, 0, hw_m, 0), oneway=True, lanes=2, width=8.0),
                road("C1", line(hw_m, 10, hw_m, 800), width=7.0), road("C2", line(hw_m, 800, 0, 800), width=7.0),
                road("C3", line(0, 800, 0, 10), width=7.0)]
    assert lr.plan_highway_route(lr.build_graph(net(700)), map_name="synthetic", seed=1, direction=0) is None
    ok = lr.plan_highway_route(lr.build_graph(net(900)), map_name="synthetic", seed=1, direction=0)
    assert ok is not None and 800.0 <= ok.highway_m <= 1000.0, ok
    print("short highway: 700 m gets no route, 900 m does")


def check_end_tolerance_and_constants() -> None:
    _g, route = route_of(loop_network(dirt=False), seed=2)
    # 25 m from the end is the end; 40 m is not.
    for back, done in ((40.0, False), (10.0, True)):
        drv = lr.RouteDriver(route)
        pt = _samples(route, route.total_m - back, route.total_m - back, step=1.0)[0]
        drv.index = len(route.points) - 3
        drv.progress_m = route.total_m - back - 1.0
        drv.commands(pt[0], pt[1], 14.0)
        assert (drv.state == "done") == done, (back, drv.state, drv.progress_m)
    m = lr.measure_route(route, _samples(route, 0.0, route.total_m - 40.0))
    assert not m["loop_closed"], m
    m = lr.measure_route(route, _samples(route, 0.0, route.total_m - 10.0))
    assert m["loop_closed"], m
    print("end: 25 m from the end of the route counts as the end, 40 m does not")


def check_return_cost_jitter_and_avoid() -> None:
    g, route = route_of(loop_network(dirt=True), seed=2)
    hs = [lr._hash01(s, "out", a, b) for s in range(5) for a in range(5) for b in range(5)]
    assert all(0.0 <= h < 1.0 for h in hs) and len(set(hs)) > 100, "spread over 0..1"
    assert lr._hash01(1, "out", 3, 4) == lr._hash01(1, "out", 3, 4)
    assert len({lr._hash01(s, "out", 3, 4) for s in range(20)}) > 15, "the seed matters"
    assert lr._hash01(1, "out", 3, 4) != lr._hash01(1, "back", 3, 4), "the tag matters"
    # A city edge of 100 m: cost is length x a jitter in [0.85, 1.7], different per seed, 4x with avoid.
    ri = next(i for i, r in enumerate(g.roads) if r["id"] == "C2")
    n0 = g.node_of[(ri, 0)]
    n1 = g.node_of[(ri, 1)]
    costs = {s: g.return_cost(s, "out")(n0, n1, 100.0, ri) for s in range(12)}
    assert all(85.0 - 1e-6 <= c <= 170.0 + 1e-6 for c in costs.values()), costs
    assert len(set(costs.values())) > 6, "seeded jitter varies the cost"
    plain = g.return_cost(3, "out")(n0, n1, 100.0, ri)
    avoided = g.return_cost(3, "out", avoid={n1})(n0, n1, 100.0, ri)
    assert abs(avoided - 4.0 * plain) < 1e-9, (plain, avoided)
    dri = next(i for i, r in enumerate(g.roads) if r["id"] == "D2")
    d0, d1 = g.node_of[(dri, 0)], g.node_of[(dri, 1)]
    unjittered = lambda a, b: lr.JITTER_LO + (lr.JITTER_HI - lr.JITTER_LO) * lr._hash01(3, "out", a, b)
    assert abs(g.return_cost(3, "out")(d0, d1, 100.0, dri) - 100.0 * 0.5 * unjittered(d0, d1)) < 1e-9, "unpaved costs half"
    hri = next(i for i, r in enumerate(g.roads) if r["id"] == "H1")
    assert g.return_cost(3, "out")(g.node_of[(hri, 0)], g.node_of[(hri, 1)], 100.0, hri) is None, "the return leg never takes the highway"
    assert g.highway_cost(n0, n1, 100.0, ri) is None and g.highway_cost(g.node_of[(hri, 0)], g.node_of[(hri, 1)], 100.0, hri) == 100.0
    # A junction link between a city node and a highway node is not a highway-leg edge.
    assert g.highway_cost(n0, g.node_of[(hri, 0)], 8.0, -1) is None
    print("return cost: seeded jitter 0.85-1.7, unpaved half, avoid x4, no highway; highway leg highway only")


def check_dirt_preferred_over_equal_city_loop() -> None:
    dirt = 0
    for seed in range(8):
        _g, r = route_of(loop_network(dirt=True), seed=seed)
        # The synthetic city loop and dirt loop are the same length; both are available.
        assert r.dirt_m > 0.0 or r.city_m > 0.0
        dirt += 1 if r.dirt_m > r.city_m else 0
    assert dirt >= 7, dirt
    print(f"dirt: the return loop takes the dirt road when an equal city road is available ({dirt}/8)")


def check_unusable_plan_refused() -> None:
    import json
    import tempfile

    _g, route = route_of(loop_network(dirt=True), seed=2)
    good = route.to_json()
    with tempfile.TemporaryDirectory() as td:
        for label, edit in (("far end at the start", {"far_end_index": 0}), ("far end past the end", {"far_end_index": len(route.points)}),
                            ("one point", {"points": good["points"][:1], "kinds": good["kinds"][:1], "far_end_index": 0})):
            path = Path(td) / "p.json"
            path.write_text(json.dumps({**good, **edit}), encoding="utf-8")
            try:
                lr.load_route_plan(path, ["synthetic"])
            except ValueError as exc:
                assert "no usable route" in str(exc), (label, exc)
            else:
                raise AssertionError(f"accepted a plan with {label}")
    print("collector: a plan with no usable route is refused")


def check_driver_off_route_and_far_end() -> None:
    _g, route = route_of(loop_network(dirt=False), seed=2)
    drv = lr.RouteDriver(route)
    x, y, _ = route.points[3]
    assert drv.commands(x, y, 14.0)
    out: list[str] = []
    for _ in range(7):
        out += drv.commands(x + 500.0, y + 500.0, 14.0)
    assert out == [] and drv.state == "following", "a few off-route ticks are tolerated"
    cmds = drv.commands(x + 500.0, y + 500.0, 14.0)
    assert cmds == lr.span_commands(14.0) and drv.state == "off_route", cmds
    assert drv.commands(x, y, 14.0) == [] and not drv.active
    assert drv.max_off_route_m > 400.0 and not drv.far_end_reached
    # An off-route blip that recovers resets the count.
    drv2 = lr.RouteDriver(route)
    drv2.commands(x, y, 14.0)
    for _ in range(7):
        drv2.commands(x + 500.0, y, 14.0)
    drv2.commands(x, y, 14.0)
    for _ in range(7):
        drv2.commands(x + 500.0, y, 14.0)
    assert drv2.state == "following"
    # Far end flag: not before it, set once the highway leg is driven.
    veh, drv3 = FakeVehicle(), lr.RouteDriver(route)
    drive(route, veh, drv3, stop_s=route.far_end_m - 200.0)
    assert not drv3.far_end_reached and drv3.progress_m > 0.5 * route.far_end_m
    drive(route, veh, drv3, start_s=route.far_end_m - 200.0, stop_s=route.far_end_m)
    assert drv3.far_end_reached and drv3.state == "following"
    print("driver: off-route for 8 ticks hands back to span; the far-end flag is set only after the highway leg")


def check_driver_resume_on_the_route() -> None:
    """A respawn on the route itself: past the farthest point reached, on the line, facing along it; refused when it cannot be."""
    # Pin the literals, not just the names: a mutant that changes 40, 3 or 150 must fail here.
    assert lr.RESUME_NUDGE_M == 40.0
    assert lr.MAX_RESUMES == 3
    assert lr.RESUME_MIN_LEFT_M == 150.0
    _g, route = route_of(loop_network(dirt=True), seed=3)
    drv = lr.RouteDriver(route)
    drv.note_start()
    assert drv.spawns == 1 and drv.resumes == 0 and drv.skipped_m == 0.0
    # At the start, nothing driven: the pick is the nudge along the route, on its line (not a deck pose), facing along it.
    pick, why = drv.resume()
    assert why == "" and pick is not None and drv.resumes == 1 and drv.spawns == 2, (pick, why)
    i, s_, d = lr.project_forward(route, pick.pos[0], pick.pos[1], 0, back=0, ahead_m=1e9)
    assert d < 0.5 and abs(s_ - lr.RESUME_NUDGE_M) < 0.5, (s_, d)
    assert abs(pick.pos[2] - (route.points[i][2] + 0.5)) < 2.0, "z is the route's, lifted for the drop"
    qx, qy, qz, qw = pick.rot
    yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    ahead = route.points[i + 1]
    assert math.cos(yaw) * (ahead[0] - route.points[i][0]) + math.sin(yaw) * (ahead[1] - route.points[i][1]) > 0.0, "faces along the route"
    assert pick.kind in ("highway", "city", "ramp") and pick.speed[0] > 0.0 and pick.road_id.startswith("route@")
    assert abs(drv.progress_m - lr.RESUME_NUDGE_M) < 1e-6
    assert drv.skipped_m == lr.RESUME_NUDGE_M and drv.summary()["resumes"] == 1 and drv.summary()["skipped_m"] == lr.RESUME_NUDGE_M
    # The next tick at the new place re-sends mode + window (the AI was reset by the teleport), and tracks from there.
    cmds = drv.commands(pick.pos[0], pick.pos[1], 14.0)
    assert_no_ai_setline(cmds)
    assert 'ai.setMode("manual")' in cmds and any(is_set_line_cmd(c) for c in cmds), cmds[:3]
    assert drv.state == "following" and drv.max_off_route_m < 1.0
    # Driven on to 700 m, stalled again: the next resume is past 700 m, not back at the start.
    for k, c in enumerate(route.cum):
        if lr.RESUME_NUDGE_M < c <= 700.0:
            drv.commands(route.points[k][0], route.points[k][1], 14.0)
    assert 600.0 <= drv.progress_m <= 700.0 and drv.state == "following", drv.summary()
    reached = drv.progress_m
    pick2, why2 = drv.resume()
    assert pick2 is not None, why2
    _i, s2, d2 = lr.project_forward(route, pick2.pos[0], pick2.pos[1], 0, back=0, ahead_m=1e9)
    assert d2 < 0.5 and abs(s2 - (reached + 40.0)) < 0.5, (s2, reached)
    # Progress can go backwards: project_forward searches 3 segments back. The resume must still
    # land 40 m past the farthest point reached, not past the later (smaller) projection.
    drv3 = lr.RouteDriver(route)
    drv3.note_start()
    for k, c in enumerate(route.cum):
        if lr.RESUME_NUDGE_M < c <= 700.0:
            drv3.commands(route.points[k][0], route.points[k][1], 14.0)
    farthest = drv3.progress_m
    assert 600.0 <= farthest <= 700.0 and drv3.state == "following", drv3.summary()
    behind_i = max(0, drv3.index - 2)
    bx, by = route.points[behind_i][0], route.points[behind_i][1]
    _bi, back_s, back_d = lr.project_forward(route, bx, by, drv3.index)
    assert back_d < 1.0 and back_s < farthest - 10.0, (back_s, back_d, farthest, behind_i, drv3.index)
    drv3.commands(bx, by, 14.0)
    assert drv3.state == "following" and abs(drv3.progress_m - farthest) < 1e-9, (
        f"progress rolled back from {farthest} to {drv3.progress_m} (projected {back_s})"
    )
    pick_b, why_b = drv3.resume()
    assert pick_b is not None, why_b
    _i, s_b, d_b = lr.project_forward(route, pick_b.pos[0], pick_b.pos[1], 0, back=0, ahead_m=1e9)
    assert d_b < 0.5 and abs(s_b - (farthest + 40.0)) < 0.5, (s_b, farthest, back_s)
    # The cap: MAX_RESUMES then None, with the reason, and the driver untouched by the refusal.
    while drv.resumes < lr.MAX_RESUMES:
        assert drv.resume()[0] is not None
    before = drv.summary()
    none, why3 = drv.resume()
    assert none is None and "already" in why3 and drv.summary() == before, (why3, before, drv.summary())
    # Too little route left: None, with the reason.
    late = lr.RouteDriver(route)
    late.note_start()
    late.progress_m = route.total_m - 150.0 - 40.0 + 1.0
    nope, why4 = late.resume()
    assert nope is None and "of the route are left" in why4 and late.resumes == 0, why4
    edge = lr.RouteDriver(route)
    edge.note_start()
    edge.progress_m = route.total_m - 150.0 - 40.0 - 1.0
    assert edge.resume()[0] is not None, "just inside the margin still resumes"
    # Exact boundary: left == 150 m after the nudge must still resume (`<`, not `<=`).
    exact = lr.RouteDriver(route)
    exact.note_start()
    exact.progress_m = route.total_m - 150.0 - 40.0
    got_exact, why_exact = exact.resume()
    assert got_exact is not None and exact.resumes == 1, (why_exact, route.total_m - (exact.progress_m if got_exact is None else 0.0))
    # A finished or abandoned driver is never resumed (the collector's deck takes over, as before).
    for state in ("done", "off_route"):
        dead = lr.RouteDriver(route)
        dead.note_start()
        dead.state = state
        got, why5 = dead.resume()
        assert got is None and state in why5 and dead.resumes == 0, (state, why5)
    # The kind and speed come from the route: a dirt stretch takes the city speed range.
    dirt_s = next((route.cum[k] for k, kd in enumerate(route.kinds) if kd == "dirt" and 0 < k < len(route.points) - 2), None)
    assert dirt_s is not None
    _i, dpick = lr.route_resume_pick(route, dirt_s + 1.0)
    assert dpick.kind == "city" and dpick.speed == lr.SPEED["city"], dpick
    # At a kind-change segment, the pick must use kinds[i+1], not kinds[i].
    change = next((k for k in range(len(route.kinds) - 2)
                   if route.kinds[k] != route.kinds[k + 1] and (route.cum[k + 1] - route.cum[k]) > 1.0), None)
    assert change is not None, "loop_network(dirt=True) has no kind-change segment"
    s_change = route.cum[change] + min(1.0, 0.5 * (route.cum[change + 1] - route.cum[change]))
    i_change, cpick = lr.route_resume_pick(route, s_change)
    assert i_change == change, (i_change, change, s_change, route.cum[change], route.cum[change + 1])
    assert route.kinds[i_change] != route.kinds[i_change + 1], (route.kinds[i_change], route.kinds[i_change + 1])
    upcoming = route.kinds[i_change + 1]
    expected = "city" if upcoming == "dirt" else upcoming
    assert cpick.kind == expected and (upcoming != "dirt" or cpick.speed == lr.SPEED["city"]), (
        cpick.kind, upcoming, route.kinds[i_change], cpick.speed
    )
    print("driver resume: on the route line, facing along it, past the farthest point, capped, refused near the end / when done")


# ------------------------------------------------------------------ measure


def _samples(route, a, b, step=8.0, jitter=0.0):
    out = []
    s = a
    while s <= b:
        i = next((k for k in range(len(route.cum) - 1) if route.cum[k + 1] >= s), len(route.points) - 2)
        t = (s - route.cum[i]) / max(1e-6, route.cum[i + 1] - route.cum[i])
        out.append((route.points[i][0] + t * (route.points[i + 1][0] - route.points[i][0]) + jitter,
                    route.points[i][1] + t * (route.points[i + 1][1] - route.points[i][1])))
        s += step
    return out


def check_measure_route() -> None:
    _g, route = route_of(loop_network(dirt=True), seed=3)
    full = lr.measure_route(route, _samples(route, 0.0, route.total_m))
    assert full["far_end_reached"] and full["loop_closed"], full
    assert full["highway_fraction"] == 1.0 and full["return_fraction"] > 0.98, full
    assert full["dirt_driven_m"] > 2500.0 and full["off_route_samples"] == 0, full
    assert abs(full["driven_m"] - route.total_m) < 20.0 and full["planned_m"] == round(route.total_m, 1)
    half = lr.measure_route(route, _samples(route, 0.0, 0.5 * route.highway_m))
    assert not half["far_end_reached"] and not half["loop_closed"], half
    assert 0.45 < half["highway_fraction"] < 0.55 and half["return_fraction"] == 0.0 and half["dirt_driven_m"] == 0.0, half
    hw_only = lr.measure_route(route, _samples(route, 0.0, route.far_end_m + 10.0))
    assert hw_only["far_end_reached"] and not hw_only["loop_closed"] and hw_only["return_fraction"] < 0.01, hw_only
    # A car that is somewhere else earns nothing; a teleport ahead earns nothing.
    away = lr.measure_route(route, [(x + 400.0, y + 400.0) for x, y in _samples(route, 0.0, 600.0)])
    assert away["driven_m"] == 0.0 and away["off_route_samples"] == away["samples"] > 10, away
    jump = lr.measure_route(route, _samples(route, 0.0, 100.0) + _samples(route, route.far_end_m, route.far_end_m + 50.0))
    assert not jump["far_end_reached"] and jump["driven_m"] < 600.0, jump
    # Off the route by 80 m for a stretch, then back: those samples are counted off, the rest progress.
    mixed = _samples(route, 0.0, 800.0) + [(x, y + 80.0) for x, y in _samples(route, 800.0, 1000.0)] + _samples(route, 1000.0, 1400.0)
    m = lr.measure_route(route, mixed)
    assert m["off_route_samples"] > 20 and m["max_off_route_m"] >= 79.0 and m["driven_m"] > 1300.0, m
    assert lr.measure_route(route, [])["driven_m"] == 0.0
    print("measure: full drive 1.0 / far end / loop closed; half drive 0.5; elsewhere or teleport earn nothing; off-route counted")


def main() -> None:
    check_dirt_material_words()
    check_graph_directions_and_links()
    check_route_shape_synthetic()
    check_dirt_is_used_when_present_and_omitted_when_not()
    check_direction_follows_bearing()
    check_no_highway_gives_no_route()
    check_json_roundtrip_and_determinism()
    check_driver_uses_set_line_api()
    check_driver_sends_the_route_in_order()
    check_driver_resend_rules()
    check_driver_speed_caps()
    check_collector_hooks()
    check_route_length_fits_the_clip()
    check_length_cap_and_window_pinned()
    check_short_highway_gets_no_route()
    check_end_tolerance_and_constants()
    check_return_cost_jitter_and_avoid()
    check_dirt_preferred_over_equal_city_loop()
    check_unusable_plan_refused()
    check_driver_off_route_and_far_end()
    check_driver_resume_on_the_route()
    check_measure_route()
    check_real_maps_routes()
    check_west_coast_uses_gravel()
    print("test_long_route: OK")


if __name__ == "__main__":
    main()
