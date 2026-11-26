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
    assert r.return_m > lr.MIN_RETURN_M and r.dirt_m == 0.0 and r.city_m > 3000.0, (r.return_m, r.dirt_m, r.city_m)
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
                assert r.return_m >= lr.MIN_RETURN_M, (map_name, seed, d, r.return_m)
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

SET_LINE = re.compile(r"ai\.setLine\(\{(.*)\}, true\)")
POS = re.compile(r"\{pos=\{([-0-9.]+),([-0-9.]+),0\},speed=([-0-9.]+)\}")


class FakeVehicle:
    """Records the Lua the driver sends, like the collector's ``_queue(vehicle, cmd)``."""

    def __init__(self):
        self.sent: list[str] = []
        self.mode = "span"
        self.line: list[tuple[float, float]] = []

    def queue(self, cmd: str) -> None:
        self.sent.append(cmd)
        if cmd.startswith("ai.setMode("):
            self.mode = cmd.split('"')[1]
        m = SET_LINE.match(cmd)
        if m:
            self.line = [(float(a), float(b)) for a, b, _v in POS.findall(m.group(1))]


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


def check_driver_sends_the_route_in_order() -> None:
    _g, route = route_of(loop_network(dirt=True), seed=2)
    veh, drv = FakeVehicle(), lr.RouteDriver(route)
    drive(route, veh, drv)
    lines = [c for c in veh.sent if c.startswith("ai.setLine(")]
    assert lines and veh.sent[0] == 'ai.setMode("manual")' and veh.sent[1] == 'ai.driveInLane("off")', veh.sent[:3]
    assert veh.sent[2] == 'ai.setSpeedMode("limit")' and veh.sent[3].startswith("ai.setSpeed(14.0)"), veh.sent[:4]
    assert sum(1 for c in veh.sent if c == 'ai.setMode("manual")') == 1, "manual mode is set once; later sends only replace the line"
    # Every line starts at the car's place on the route, runs forward along it, and is at most the window long.
    prev_first = -1.0
    for cmd in lines:
        pts = [(float(a), float(b)) for a, b, _v in POS.findall(SET_LINE.match(cmd).group(1))]
        i0, s0, d0 = lr.project_forward(route, pts[0][0], pts[0][1], 0, ahead_m=1e9)
        assert d0 < 3.0, "line starts on the route"
        assert s0 >= prev_first - 1e-6, "lines move forward, never back"
        prev_first = s0
        i1, s1, d1 = lr.project_forward(route, pts[-1][0], pts[-1][1], i0, ahead_m=1e9)
        assert d1 < 3.0 and s1 > s0 and s1 - s0 <= lr.WINDOW_M + 120.0, (s0, s1)
        assert all(a[0] != b[0] or a[1] != b[1] for a, b in zip(pts, pts[1:]))
    # The first line is the highway, a later one is the return leg (off the highway).
    first_pts = [(float(a), float(b)) for a, b, _v in POS.findall(SET_LINE.match(lines[0]).group(1))]
    assert abs(first_pts[0][0]) < 5.0 and first_pts[0][1] < 0.0 and abs(first_pts[-1][1]) < 6.0, first_pts[:2]
    assert any(max(abs(p[1]) for p in [(float(a), float(b)) for a, b, _v in POS.findall(SET_LINE.match(c).group(1))]) > 500.0 for c in lines), "return leg is sent"
    assert drv.far_end_reached and drv.state == "done", drv.summary()
    assert veh.sent[-4:] == lr.span_commands(14.0) and veh.mode == "span", "end of the route hands back to span"
    assert drv.commands(0.0, 0.0, 14.0) == [], "nothing more after done"
    assert 4 <= drv.lines_sent <= 120, drv.lines_sent
    print(f"driver (fake vehicle): {drv.lines_sent} ai.setLine windows, in order, highway then return, span at the end")


def check_driver_resend_rules() -> None:
    _g, route = route_of(loop_network(dirt=False), seed=2)
    veh, drv = FakeVehicle(), lr.RouteDriver(route)
    x, y = route.points[0][0], route.points[0][1]
    first = drv.commands(x, y, 14.0)
    assert any(c.startswith("ai.setLine(") for c in first)
    assert drv.commands(x, y, 14.0) == [], "same place and speed: nothing re-sent"
    assert drv.commands(x, y, 14.2) == [], "speed change under 0.5 m/s: nothing"
    again = drv.commands(x, y, 15.0)
    assert again and again[0] == "ai.setSpeed(15.0)" and any(c.startswith("ai.setLine(") for c in again), again[:1]
    assert 'ai.setMode("manual")' not in again, "the mode is not re-sent"
    # Far enough along: a fresh window.
    px, py, _ = route.points[next(i for i, c in enumerate(route.cum) if c > lr.RESEND_M + 30)]
    assert any(c.startswith("ai.setLine(") for c in drv.commands(px, py, 15.0))
    print("driver: re-sends on 80 m of progress or a 0.5 m/s speed change, never re-sets the mode")


def check_driver_off_route_and_far_end() -> None:
    _g, route = route_of(loop_network(dirt=False), seed=2)
    drv = lr.RouteDriver(route)
    x, y, _ = route.points[3]
    assert drv.commands(x, y, 14.0)
    out: list[str] = []
    for _ in range(lr.OFF_TICKS - 1):
        out += drv.commands(x + 500.0, y + 500.0, 14.0)
    assert out == [] and drv.state == "following", "a few off-route ticks are tolerated"
    cmds = drv.commands(x + 500.0, y + 500.0, 14.0)
    assert cmds == lr.span_commands(14.0) and drv.state == "off_route", cmds
    assert drv.commands(x, y, 14.0) == [] and not drv.active
    assert drv.max_off_route_m > 400.0 and not drv.far_end_reached
    # An off-route blip that recovers resets the count.
    drv2 = lr.RouteDriver(route)
    drv2.commands(x, y, 14.0)
    for _ in range(lr.OFF_TICKS - 1):
        drv2.commands(x + 500.0, y, 14.0)
    drv2.commands(x, y, 14.0)
    for _ in range(lr.OFF_TICKS - 1):
        drv2.commands(x + 500.0, y, 14.0)
    assert drv2.state == "following"
    # Far end flag: not before it, set once the highway leg is driven.
    veh, drv3 = FakeVehicle(), lr.RouteDriver(route)
    drive(route, veh, drv3, stop_s=route.far_end_m - 200.0)
    assert not drv3.far_end_reached and drv3.progress_m > 0.5 * route.far_end_m
    drive(route, veh, drv3, start_s=route.far_end_m - 200.0, stop_s=route.far_end_m)
    assert drv3.far_end_reached and drv3.state == "following"
    print("driver: off-route for 8 ticks hands back to span; the far-end flag is set only after the highway leg")


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
    check_driver_sends_the_route_in_order()
    check_driver_resend_rules()
    check_driver_off_route_and_far_end()
    check_measure_route()
    check_real_maps_routes()
    check_west_coast_uses_gravel()
    print("test_long_route: OK")


if __name__ == "__main__":
    main()
