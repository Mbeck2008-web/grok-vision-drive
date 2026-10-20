#!/usr/bin/env python3
"""The excursion line goes out with beamngpy's set_line and is acked by readback.

0.39.4.0's vehicle ``ai`` has no ``setLine``. The collector used to queue
``ai.setLine(...)`` as raw Lua after ``ai.setMode("manual")``: the call
failed in the vehicle VM, the span route was already gone, and every weave
ran on held inputs. These checks use a fake vehicle:

* the send order is ``setMode("manual")``, ``driveInLane("off")``, then
  ``ai.set_line`` (beamngpy), and nothing but readbacks after it;
* no readback that shows the line within LINE_ACK_TIMEOUT_S aborts the move
  ``line_not_active`` (timeout, set_line error, no ``ai.set_line``, a stale
  line, no Lua reply);
* the collector loop with a vehicle whose line never takes aborts the weave
  ``line_not_active``, puts span lane-follow back at once, and never tags a
  weave row without an ack.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import ai_line  # noqa: E402
from python.collect.ai_line import (  # noqa: E402
    LINE_ACK_POLL_S,
    LINE_ACK_TIMEOUT_S,
    LINE_LOST,
    LINE_NOT_ACTIVE,
    line_nodes,
    line_state_lua,
    recheck_line,
    send_and_ack,
)
from python.collect.excursion import excursion_commands  # noqa: E402

_REAL_SLEEP = time.sleep
POINTS = [(10.0, 0.0), (10.3, 4.0), (10.6, 8.0), (10.3, 12.0), (10.0, 16.0)]


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


class _Ai:
    def __init__(self, log: list, fail: Exception | None = None) -> None:
        self.log = log
        self.fail = fail

    def set_line(self, line: list, cling: bool = True) -> None:
        self.log.append(("set_line", line, cling))
        if self.fail is not None:
            raise self.fail


class _Vehicle:
    """``replies`` is what each readback returns, in order; the last one repeats."""

    def __init__(self, replies: list[str], *, ai: bool = True, fail: Exception | None = None) -> None:
        self.log: list = []
        self.replies = list(replies)
        if ai:
            self.ai = _Ai(self.log, fail)

    def queue_lua_command(self, chunk: str, response: bool = False) -> str | None:
        self.log.append(("lua", chunk, response))
        if not response:
            return None
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0] if self.replies else None


def _queue(vehicle: _Vehicle, command: str) -> None:
    vehicle.queue_lua_command(command)


def _active(n: int = len(POINTS)) -> str:
    x, y = POINTS[n - 1]
    return f"manual|{x:.2f},{y:.2f}|true"


def check_send_order_and_ack() -> None:
    clock = _Clock()
    veh = _Vehicle(["manual|none|true", "manual|none|true", _active()])
    ack = send_and_ack(veh, POINTS, 12.0, queue=_queue, clock=clock, sleep=clock.sleep)
    assert ack.ok, ack
    assert ack.polls == 3 and ack.mode == "manual"
    assert abs(ack.wait_s - 2 * LINE_ACK_POLL_S) < 1e-9, ack.wait_s
    kinds = [entry[0] if entry[0] == "set_line" else entry[1] for entry in veh.log]
    assert kinds[:3] == ['ai.setMode("manual")', 'ai.driveInLane("off")', "set_line"], kinds
    after = veh.log[3:]
    # Only readbacks (response=True) after the line: nothing that could reset the route.
    assert after and all(e[0] == "lua" and e[2] is True and e[1] == line_state_lua(len(POINTS)) for e in after), after
    _name, line, cling = veh.log[2]
    assert cling is True
    assert line == line_nodes(POINTS, 12.0)
    assert line[0] == {"pos": (10.0, 0.0, 0.0), "speed": 12.0}
    assert ack.as_tag() == {"ok": True, "mode": "manual", "wait_s": 0.1, "polls": 3}
    # The excursion commands for a line are the pre-commands only: no Lua line, no setSpeed after it.
    pre = excursion_commands(12.0, "weave", 1, path=POINTS)
    assert pre == ['ai.setMode("manual")', 'ai.driveInLane("off")'], pre
    assert not any("setLine" in c for c in pre)
    print(f"line sent with set_line, acked after {ack.polls} readbacks")


def check_no_ack_aborts() -> None:
    clock = _Clock()
    veh = _Vehicle(["manual|none|true"])
    ack = send_and_ack(veh, POINTS, 12.0, queue=_queue, clock=clock, sleep=clock.sleep)
    assert not ack.ok and ack.reason == LINE_NOT_ACTIVE, ack
    assert ack.wait_s >= LINE_ACK_TIMEOUT_S - 1e-9 and ack.wait_s < LINE_ACK_TIMEOUT_S + LINE_ACK_POLL_S + 1e-9, ack.wait_s
    assert ack.polls == round(LINE_ACK_TIMEOUT_S / LINE_ACK_POLL_S) + 1, ack.polls
    assert "timeout" in ack.detail
    tag = ack.as_tag()
    assert tag["ok"] is False and tag["reason"] == LINE_NOT_ACTIVE

    # Span still running (the manual switch did not land) is not an ack either.
    clock = _Clock()
    ack = send_and_ack(_Vehicle(["span|none|true"]), POINTS, 12.0, queue=_queue, clock=clock, sleep=clock.sleep)
    assert not ack.ok and ack.mode == "span"

    # A line node from an older, different line is not this line.
    clock = _Clock()
    ack = send_and_ack(_Vehicle(["manual|55.00,90.00|true"]), POINTS, 12.0, queue=_queue, clock=clock, sleep=clock.sleep)
    assert not ack.ok and ack.reason == LINE_NOT_ACTIVE

    # set_line raised: no readback, aborted.
    veh = _Vehicle([_active()], fail=RuntimeError("socket closed"))
    ack = send_and_ack(veh, POINTS, 12.0, queue=_queue, clock=_Clock(), sleep=lambda s: None)
    assert not ack.ok and ack.reason == LINE_NOT_ACTIVE and "socket closed" in ack.detail
    assert not any(e[0] == "lua" and e[2] for e in veh.log)

    # No beamngpy handle: no hand-built Lua line instead.
    veh = _Vehicle([_active()], ai=False)
    ack = send_and_ack(veh, POINTS, 12.0, queue=_queue, clock=_Clock(), sleep=lambda s: None)
    assert not ack.ok and "no beamngpy AI set_line" in ack.detail
    assert not any("setLine" in str(e[1]) or "driveUsingPath" in str(e[1]) for e in veh.log)

    # A Lua chunk that cannot reply (None) never acks.
    clock = _Clock()
    ack = send_and_ack(_Vehicle([None]), POINTS, 12.0, queue=_queue, clock=clock, sleep=clock.sleep)  # type: ignore[list-item]
    assert not ack.ok
    print("no readback ack within the timeout aborts line_not_active")


def check_recheck() -> None:
    assert recheck_line(_Vehicle([_active()]), POINTS).ok
    lost = recheck_line(_Vehicle(["span|none|true"]), POINTS)
    assert not lost.ok and lost.reason == LINE_LOST
    print("a route that is gone mid-line is line_lost")


# ---------------------------------------------------------------- collector loop
def _run_collector(behaviour: str) -> tuple[str, list, list[dict]]:
    """Drive the real collect loop on a straight painted road with a fake vehicle.

    ``behaviour``: ``never`` (the line never shows in the readback), ``ok``
    (the readback shows the sent line), ``lose`` (it shows, then is gone on
    the second recheck).
    """
    import io
    import json
    import tempfile

    import numpy as np

    import scripts.collect_scene_lanes as collect
    import python.collect.excursion as excursion
    from python.collect.auto_mode import stop_file
    from python.debug.gt_geom import RoadPoly

    root = Path(tempfile.mkdtemp())
    out = root / "scene"
    stop = stop_file(out)
    clock = {"t": 1000.0}

    class Vehicle(_Vehicle):
        model = "etk800"

        def __init__(self) -> None:
            super().__init__(["manual|none|true"])
            self.sent: tuple[float, float] | None = None
            self.reads = 0
            self.y = 5.0
            self.polls = 0
            self.state = {}

        def poll_sensors(self) -> None:
            self.polls += 1
            _REAL_SLEEP(0.004)  # lets the row writer thread keep up with the fake clock
            self.y += 1.2
            self.state = {
                "pos": [1.75, self.y, 0.5], "dir": [0.0, 1.0, 0.0], "vel": [0.0, 12.0, 0.0],
                "electrics": {"steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0, "wheelspeed": 12.0},
            }
            if self.polls >= 400:
                stop.write_text("stop", encoding="utf-8")

        def connect(self, bng: object) -> None:
            del bng

        def is_connected(self) -> bool:
            return True

        def queue_lua_command(self, chunk: str, response: bool = False) -> str | None:
            reply = super().queue_lua_command(chunk, response)
            if not response or behaviour == "never" or self.sent is None:
                return reply
            self.reads += 1
            if behaviour == "lose" and self.reads >= 3:
                return "span|none|true"
            return f"manual|{self.sent[0]:.2f},{self.sent[1]:.2f}|true"

    vehicle = Vehicle()
    real_set_line = vehicle.ai.set_line

    def set_line(line: list, cling: bool = True) -> None:
        real_set_line(line, cling)
        vehicle.sent = (float(line[-1]["pos"][0]), float(line[-1]["pos"][1]))
        vehicle.reads = 0

    vehicle.ai.set_line = set_line  # type: ignore[method-assign]

    class Api:
        def get_current(self) -> dict:
            return {"ego": vehicle}

        def get_player_vehicle_id(self) -> dict:
            return {"vid": "ego"}

    class Bng:
        vehicles = Api()

        def queue_lua_command(self, *a: object, **k: object) -> None:
            return None

    bng = Bng()

    class Cam:
        def poll(self) -> dict:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[0, 0] = (8, 16, 24)
            return {"colour": image}

    import python.debug.gt_logger as gt_logger
    import python.sensors.tech as tech
    from python.debug.painted_lanes import PaintedLanes

    ys = [float(y) for y in range(-200, 2001, 5)]
    paint = PaintedLanes([{"material": "line_white", "nodes": [[x, y, 0.4, 0.15] for y in ys]} for x in (0.0, 3.5)])
    road = [RoadPoly("r", [(0.0, -200.0), (0.0, 2000.0)], 7.0)]
    saved = (
        collect.time.time, collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s, collect.painted_for, collect._attach_cameras,
        tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy, gt_logger.load_road_polys,
    )

    def sleep(seconds: float) -> None:
        clock["t"] += float(seconds or 0.0)

    collect.time.sleep = sleep  # type: ignore[assignment]
    collect.time.monotonic = lambda: clock["t"]  # type: ignore[assignment]
    # Rows are stamped and paced by wall time: the same fake clock, so the run is deterministic.
    collect.time.time = lambda: 1.7e9 + clock["t"]  # type: ignore[assignment]
    collect.tick_sleep_s = lambda _elapsed, _period=0.1: 0.1  # type: ignore[assignment]
    collect.painted_for = lambda level: paint  # type: ignore[assignment]
    collect._attach_cameras = lambda session, veh, cfg: {"main": Cam()}  # type: ignore[assignment]
    tech.open_tech_beamngpy = lambda *a, **k: bng  # type: ignore[assignment]
    tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
    tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
    gt_logger.load_road_polys = lambda session: (road, "")  # type: ignore[assignment]
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        collect.main(["--minutes", "2", "--smoke-s", "0", "--data-dir", str(root), "--out", str(out),
                      "--host", "127.0.0.1", "--port", "1", "--speed-min", "12", "--speed-max", "12"])
    finally:
        sys.stdout = old
        (
            collect.time.time, collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s, collect.painted_for, collect._attach_cameras,
            tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy, gt_logger.load_road_polys,
        ) = saved
    text = buf.getvalue()
    rows = [json.loads(ln) for ln in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    return text, list(vehicle.log), rows


def _line_sends(log: list) -> list[int]:
    return [i for i, e in enumerate(log) if e[0] == "set_line"]


def check_collector_aborts_weave_without_ack() -> None:
    text, log, rows = _run_collector("never")
    sends = _line_sends(log)
    assert sends, "no line was sent:\n" + "\n".join(ln for ln in text.splitlines() if "gate" in ln)[:3000]
    assert "abort line_not_active" in text, text[-3000:]
    for i in sends:
        # After each line: only readbacks, then the abort's lane-follow restore.
        rest = [e for e in log[i + 1:] if e[0] == "lua" and not e[2]]
        assert rest and rest[0][1] == 'ai.setMode("span")', rest[:3]
        assert any(e[1] == 'ai.driveInLane("on")' for e in rest[:4])
    assert not any(e[0] == "lua" and "setLine" in e[1] for e in log)
    phases = [r["excursion"]["phase"] for r in rows]
    assert "weave" not in phases and "lane_change" not in phases, sorted(set(phases))
    aborts = [r for r in rows if r["excursion"]["phase"] == "abort"]
    assert aborts and all(r["excursion"]["reason"] == LINE_NOT_ACTIVE for r in aborts), {r["excursion"]["reason"] for r in aborts}
    print(f"collector: {len(sends)} lines without an ack, each aborted line_not_active, span restored")


def check_collector_tags_acked_weave() -> None:
    text, log, rows = _run_collector("ok")
    sends = _line_sends(log)
    assert sends and "line_not_active" not in text, text[-3000:]
    weave = [r for r in rows if r["excursion"]["phase"] == "weave"]
    assert len(weave) >= 100, (len(weave), sorted({r["excursion"]["phase"] for r in rows}))
    assert all((r["excursion"].get("line_ack") or {}).get("ok") is True for r in weave)
    assert not any("line_ack" in r["excursion"] for r in rows if r["excursion"]["phase"] in ("cruise", "abort"))
    for i in sends:
        later = [e for e in log[i + 1:] if e[0] == "lua" and not e[2]]
        # Nothing between the line and the end of the move: the first plain command restores lane-follow.
        assert not later or later[0][1].startswith('ai.setMode("span")'), later[:2]
    print(f"collector: {len(weave)} weave rows tagged with an ok line_ack")


def check_collector_line_lost() -> None:
    text, log, rows = _run_collector("lose")
    assert "abort line_lost" in text, text[-3000:]
    aborts = {r["excursion"]["reason"] for r in rows if r["excursion"]["phase"] == "abort"}
    assert LINE_LOST in aborts, aborts
    weave = [r for r in rows if r["excursion"]["phase"] == "weave"]
    # Two rechecks at LINE_RECHECK_S: the weave ran ~2 s (about 20 ticks), then aborted.
    assert 10 <= len(weave) <= 30, len(weave)
    print("collector: a route lost mid-line aborts line_lost")


def main() -> None:
    check_send_order_and_ack()
    check_no_ack_aborts()
    check_recheck()
    check_collector_aborts_weave_without_ack()
    check_collector_tags_acked_weave()
    check_collector_line_lost()
    print("weave line ack ok")


if __name__ == "__main__":
    main()
