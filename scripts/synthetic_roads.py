"""Fake BeamNG session over the hand-built roads in ``tests/fixtures/junctions_synthetic.json``.

Shared by ``test_lane_gt_frozen.py``, ``test_gt_level_change.py``,
``test_road_loader_cache.py`` and ``test_junction_grade.py``. Not a test
itself: the offline suite only runs ``test_*`` files.

Everything here is synthetic. No BeamNG map data is read or stored.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GRAPH_PATH = FIXTURES / "junctions_synthetic.json"
FROZEN_PATH = FIXTURES / "lane_gt_pre_pr1.jsonl.gz"
# Fixed start of the fake clocks. Any value works; it only has to be the same
# on every run so the frozen log is reproducible.
WALL0 = 1_800_000_000.0
SIM0 = 100.0


def load_graph(path: Path = GRAPH_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return dx / n, dy / n


def network_spec(graph: dict[str, Any], level: str) -> dict[str, Any]:
    """``get_road_network(include_edges=True)`` shaped roads for one synthetic level.

    One straight road per graph edge, sampled every ``sample_m``, two lanes
    (one each way), ``left`` / ``middle`` / ``right`` edge points.
    """
    lane = float(graph["lane_width_m"])
    step = float(graph["sample_m"])
    lvl = graph["levels"][level]
    nodes = lvl["nodes"]
    roads: dict[str, Any] = {}
    for edge in lvl["edges"]:
        ax, ay = nodes[edge["a"]]
        bx, by = nodes[edge["b"]]
        ux, uy = _unit(bx - ax, by - ay)
        lx, ly = -uy, ux  # left of a->b
        length = math.hypot(bx - ax, by - ay)
        count = max(2, int(round(length / step)) + 1)
        points = []
        for i in range(count):
            s = length * i / (count - 1)
            mx, my = ax + ux * s, ay + uy * s
            points.append(
                {
                    "left": [mx + lx * lane, my + ly * lane, 0.0],
                    "middle": [mx, my, 0.0],
                    "right": [mx - lx * lane, my - ly * lane, 0.0],
                }
            )
        roads[f"{edge['a']}-{edge['b']}"] = {
            "edges": points,
            "lanesLeft": 1,
            "lanesRight": 1,
            "drivability": 1.0,
            "oneWay": bool(edge.get("oneway", False)),
        }
    return roads


def track(
    graph: dict[str, Any],
    level: str,
    path: Iterable[str],
    speed_mps: float = 10.0,
    dt_s: float = 0.1,
    lane_side: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Scripted ``(x, y, heading_rad)`` poses along ``path`` in the right-hand lane.

    The node polyline is cut into ``speed * dt`` steps, corners are rounded
    with a moving average, and each point moves half a lane to the right.
    Values are rounded to 0.1 mm so the poses do not depend on libm.
    """
    nodes = graph["levels"][level]["nodes"]
    pts = [tuple(map(float, nodes[name])) for name in path]
    step = float(speed_mps) * float(dt_s)
    dense: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(round(length / step)))
        for i in range(n):
            f = i / n
            dense.append((ax + (bx - ax) * f, ay + (by - ay) * f))
    dense.append(pts[-1])
    half = 12
    for _ in range(3):
        smooth = []
        for i in range(len(dense)):
            lo, hi = max(0, i - half), min(len(dense), i + half + 1)
            if i - lo != hi - 1 - i:  # keep the ends straight: symmetric window only
                k = min(i - lo, hi - 1 - i)
                lo, hi = i - k, i + k + 1
            xs = [p[0] for p in dense[lo:hi]]
            ys = [p[1] for p in dense[lo:hi]]
            smooth.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        dense = smooth
    lane = float(graph["lane_width_m"]) * 0.5 * float(lane_side)
    poses: list[tuple[float, float, float]] = []
    for i in range(len(dense)):
        a = dense[max(0, i - 1)]
        b = dense[min(len(dense) - 1, i + 1)]
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        x = dense[i][0] + math.sin(heading) * lane
        y = dense[i][1] - math.cos(heading) * lane
        poses.append((round(x, 4), round(y, 4), round(heading, 6)))
    return poses


class FakeVehicleData:
    """What ``gt_logger._pose`` and ``write`` read from a live vehicle."""

    def __init__(self, x: float, y: float, heading: float, speed_mps: float) -> None:
        self.pos = (x, y, 0.5)
        self.dir = (round(math.cos(heading), 9), round(math.sin(heading), 9), 0.0)
        self.speed_mps = speed_mps
        self.damage = 0.0


class _Current:
    def __init__(self, level: str) -> None:
        self.level = level
        self.name = level


class _Scenario:
    def __init__(self, owner: "FakeBng") -> None:
        self._owner = owner

    def get_road_network(self, include_edges: bool = False, drivable_only: bool = False) -> Any:
        self._owner.reads += 1
        self._owner.read_args.append((include_edges, drivable_only))
        return self._owner.network()

    def get_current(self, connect: bool = True) -> _Current:
        return _Current(self._owner.level)


class _Control:
    def __init__(self, owner: "FakeBng") -> None:
        self._owner = owner

    def queue_lua_command(self, chunk: str, response: bool = False) -> Any:
        self._owner.lua_calls += 1
        if "getCurrentLevelIdentifier" in chunk:
            return self._owner.level
        return None


class FakeBng:
    """A BeamNGpy session stand-in that counts ``get_road_network`` reads."""

    def __init__(self, graph: dict[str, Any], level: str, *, empty: bool = False) -> None:
        self.graph = graph
        self.level = level
        self.empty = empty
        self.reads = 0
        self.read_args: list[tuple[bool, bool]] = []
        self.lua_calls = 0
        self.sim_t = SIM0
        self.scenario = _Scenario(self)
        self.control = _Control(self)

    def network(self) -> dict[str, Any]:
        if self.empty:
            return {}
        return network_spec(self.graph, self.level)

    def switch_level(self, level: str) -> None:
        self.level = level

    def get_gamestate(self) -> dict[str, Any]:
        return {"state": "scenario", "time": self.sim_t}


class FakeClock:
    """Stands in for ``time.time`` while a test drives the logger."""

    def __init__(self, start: float = WALL0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


def drive(logger: Any, bng: FakeBng, poses: list[tuple[float, float, float]], clock: FakeClock,
          *, speed_mps: float = 10.0, dt_s: float = 0.1) -> list[dict[str, Any]]:
    """Feed ``poses`` through ``curve_preview`` and ``write`` like ``run_vision`` does."""
    rows: list[dict[str, Any]] = []
    for x, y, heading in poses:
        clock.now = round(clock.now + dt_s, 6)
        bng.sim_t = round(bng.sim_t + dt_s, 6)
        vdata = FakeVehicleData(x, y, heading, speed_mps)
        preview = logger.curve_preview(vdata, bng, socket_busy=False, dt_s=dt_s)
        lane_x = None if preview is None else preview.lane_x
        rows.append(
            logger.write(
                bng=bng,
                vdata=vdata,
                engaged=True,
                lane_x=lane_x,
                lane_src="gt" if lane_x is not None else "none",
                steer=0.0,
                steer_why="synthetic",
            )
        )
    return rows


def _canon(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        out = round(value, 9)
        return 0.0 if out == 0.0 else out
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def canonical_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    """Rows as JSONL with floats rounded to 1e-9 (and -0.0 as 0.0).

    The rounding keeps last-bit libm differences between Windows and Linux
    out of the comparison; anything a real change would move is far larger.
    """
    lines = [json.dumps(_canon(row), separators=(",", ":")) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def gzip_bytes(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


class patched_logger_env:
    """Patch ``time.time`` and the live-status side channel while driving a logger."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def __enter__(self) -> "patched_logger_env":
        import time as _time

        from python.debug import remote_status

        self._time = _time
        self._real_time = _time.time
        self._remote = remote_status
        self._real_live = remote_status.note_live
        _time.time = self.clock  # type: ignore[assignment]
        remote_status.note_live = lambda *a, **k: None  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        self._time.time = self._real_time  # type: ignore[assignment]
        self._remote.note_live = self._real_live  # type: ignore[assignment]
