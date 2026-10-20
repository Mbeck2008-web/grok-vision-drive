#!/usr/bin/env python3
"""The excursion line goes out with beamngpy's set_line and is acked by readback, without blocking.

0.39.4.0's vehicle ``ai`` has no ``setLine``. The collector used to queue
``ai.setLine(...)`` as raw Lua after ``ai.setMode("manual")``: the call
failed in the vehicle VM, the span route was already gone, and every weave
ran on held inputs. These checks use fake vehicles:

* send order: ``setMode("manual")``, ``driveInLane("off")``, ``ai.set_line``
  (beamngpy), then only readback requests;
* the ack needs ALL of: the AI route map's last ``wp_`` node on our last
  point, ``ai.isDriving()``, and the AI's current target a ``wp_`` node.
  Negative controls: manual mode with the node but not driving; a target
  that is not a ``wp_`` node; a stale line; no route / no debug library
  (the readback chunk run in Lua 5.1 against a stand-in ``ai`` module);
* ``LineTracker`` never waits: every call returns at once, the reply is
  picked up on a later tick, LINE_ACK_TIMEOUT_S (from the send) is the only
  budget, and a reply seen after it does not count;
* the beamngpy transport sends ``QueueLuaCommandVE`` and reads the reply only
  when the socket has one (a socketpair stand-in for beamngpy 1.36's
  connection);
* the real collect loop: rows before the ack are ``weave_pending`` (never
  ``weave``) and the loop keeps its 0.1 s ticks during a pending ack; a line
  that never acks aborts ``line_not_active``, a route lost mid-line
  ``line_lost``; ``weave_stats.line_smoke`` and the ``collect smoke`` log
  lines carry the smoke-run instrumentation, and a teleport at the weave
  start shows as a snap.
"""
from __future__ import annotations

import json
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
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
    LINE_RECHECK_S,
    LineTracker,
    line_is_active,
    line_nodes,
    line_state_lua,
    parse_line_state,
    pending_phase,
)
from python.collect.excursion import ACTIVE, EXCURSION_MASK_PHASES, PENDING, excursion_commands  # noqa: E402

_REAL_SLEEP = time.sleep
POINTS = [(10.0, 0.0), (10.3, 4.0), (10.6, 8.0), (10.3, 12.0), (10.0, 16.0)]
TICK = 0.1


def _reply(*, xy: tuple[float, float] | None = POINTS[-1], driving: bool = True, target: str = "wp_2",
           mode: str = "manual", src: str = "upv", same: bool = True) -> str:
    where = "none" if xy is None else f"{xy[0]:.2f},{xy[1]:.2f}"
    return f"{mode}|{where}|{'true' if driving else 'false'}|{target or 'none'}|{src}|{'true' if same else 'false'}"


GOOD = _reply()


class _Handle:
    """A readback in flight: done after ``after`` polls (None: never)."""

    def __init__(self, reply: object, after: int | None, log: list) -> None:
        self.reply, self.after, self.polls, self.log = reply, after, 0, log

    def poll(self) -> tuple[bool, object, str]:
        self.polls += 1
        self.log.append(("poll",))
        if self.after is None or self.polls < self.after:
            return False, None, ""
        return True, self.reply, ""


class _Ai:
    def __init__(self, log: list, fail: Exception | None = None) -> None:
        self.log = log
        self.fail = fail

    def set_line(self, line: list, cling: bool = True) -> None:
        self.log.append(("set_line", line, cling))
        if self.fail is not None:
            raise self.fail


class _Vehicle:
    """``replies``: what each readback returns, in order (the last repeats); each is ready after ``after`` polls."""

    def __init__(self, replies: list, *, after: int | None = 1, ai: bool = True, fail: Exception | None = None,
                 readback: bool = True) -> None:
        self.log: list = []
        self.replies = list(replies)
        self.after = after
        if ai:
            self.ai = _Ai(self.log, fail)
        if readback:
            self.send_lua_readback = self._send_lua_readback

    def queue_lua_command(self, chunk: str, response: bool = False) -> None:
        assert response is False, "nothing may wait for a Lua reply"
        self.log.append(("lua", chunk, response))

    def _send_lua_readback(self, chunk: str) -> _Handle:
        self.log.append(("readback", chunk))
        reply = self.replies.pop(0) if len(self.replies) > 1 else (self.replies[0] if self.replies else None)
        return _Handle(reply, self.after, self.log)


def _queue(vehicle: _Vehicle, command: str) -> None:
    vehicle.queue_lua_command(command)


def _run(tracker: LineTracker, veh: _Vehicle, t0: float = 100.0, ticks: int = 20) -> tuple[list, float]:
    """Tick every TICK s until an event (or ``ticks``). Returns (events, wall seconds spent inside tick())."""
    events, spent = [], 0.0
    for k in range(1, ticks + 1):
        w = time.perf_counter()
        ev = tracker.tick(veh, t0 + k * TICK)
        spent += time.perf_counter() - w
        if ev.kind:
            events.append((round(k * TICK, 3), ev))
            if ev.kind in ("acked", LINE_NOT_ACTIVE, LINE_LOST):
                break
    return events, spent


# ---------------------------------------------------------------- ack conditions
def check_ack_conditions() -> None:
    ok, why = line_is_active(GOOD, POINTS)
    assert ok, why
    # Negative control 1: manual mode, the wp_ node on our last point, a wp_ target, but not driving.
    ok, why = line_is_active(_reply(driving=False), POINTS)
    assert not ok and why == "not_driving", why
    # Negative control 2: driving, the node in place, but the target is not a wp_ node
    # (a navgraph node: the span route; or no route at all).
    for target in ("1234", "none", "", "wp", "awp_2"):
        ok, why = line_is_active(_reply(target=target), POINTS)
        assert not ok and why.startswith("target="), (target, why)
    # A stale line: the node is from an older, different line.
    ok, why = line_is_active(_reply(xy=(55.0, 90.0)), POINTS)
    assert not ok and why.startswith("last_node_off="), why
    ok, why = line_is_active(_reply(xy=None), POINTS)
    assert not ok and why == "no_last_node", why
    # The mode echo is not a condition: span with everything else true is still the line's route...
    assert line_is_active(_reply(mode="span"), POINTS)[0]
    # ...and manual alone (the old ack) is not enough.
    assert not line_is_active("manual|10.00,16.00|true", POINTS)[0]
    assert not line_is_active(_reply(src="nodebug", target="none"), POINTS)[0]
    assert "src=nodebug" in line_is_active(_reply(src="nodebug", target="none"), POINTS)[1]
    # parse keeps isDriving() and the target (the old parser dropped isDriving).
    st = parse_line_state(_reply(driving=False, target="wp_7", same=False))
    assert st.driving is False and st.target == "wp_7" and st.src == "upv" and st.same_map is False
    assert st.last_xy == (10.0, 16.0) and st.mode == "manual"
    assert parse_line_state(None).driving is False and parse_line_state(None).target == ""
    print("ack: last node + isDriving + wp_ target, all required; manual-only, not driving, non-wp_ target fail")


def _lua() -> str | None:
    for exe in ("lua5.1", "lua"):
        found = shutil.which(exe)
        if found:
            return found
    return None


LUA_HARNESS = r"""
local chunk = io.read("*a")
local function nop() end
local case = arg[1]
local mapData = {positions = {wp_5 = {x = 10.0, y = 16.0}, wp_1 = {x = 10.0, y = 0.0}}}
local currentRoute = {path = {"wp_1", "wp_2", "wp_3", "wp_4", "wp_5"}, plan = {{pathidx = 1}, {pathidx = 2}}}
if case == "empty_plan" then currentRoute.plan = {} end
if case == "no_route" then currentRoute = nil end
if case == "navgraph" then currentRoute = {path = {"7701", "7702"}, plan = {{pathidx = 1}, {pathidx = 2}}} end
local function updateGFX(dt)
  if mapData ~= mapmgr.mapData then currentRoute = nil end
  return currentRoute
end
ai = {}
ai.getState = function() return {mode = "manual"} end
ai.updateGFX = updateGFX
ai.isDriving = function() return ai.updateGFX == updateGFX end
mapmgr = {mapData = mapData}
if case == "not_driving" then ai.updateGFX = nop end
if case == "other_map" then mapmgr = {mapData = {positions = {wp_5 = {x = 10.0, y = 16.0}}}} end
if case == "stale_mapmgr" then mapData = nil; currentRoute = nil end
if case == "nodebug" then debug = nil end
local f = assert((loadstring or load)(chunk))
io.write(tostring(f()))
"""


def check_lua_chunk() -> None:
    """Run line_state_lua in Lua 5.1 against a stand-in ai whose route lives in updateGFX upvalues."""
    chunk = line_state_lua(len(POINTS))
    assert "debug.getupvalue" in chunk and "ai.updateGFX" in chunk and "'currentRoute'" in chunk
    assert "'mapData'" in chunk and "r.plan[2]" in chunk and "ai.isDriving()" in chunk
    lua = _lua()
    assert lua, "lua5.1 is not on PATH"
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "h.lua"
        harness.write_text(LUA_HARNESS, encoding="utf-8")

        def run(case: str) -> str:
            res = subprocess.run([lua, str(harness), case], input=chunk, capture_output=True, text=True, timeout=30)
            assert res.returncode == 0, (case, res.stderr)
            return res.stdout

        got = {case: run(case) for case in
               ("ok", "not_driving", "empty_plan", "no_route", "navgraph", "other_map", "stale_mapmgr", "nodebug")}
    assert got["ok"] == "manual|10.00,16.00|true|wp_2|upv|true", got
    assert line_is_active(got["ok"], POINTS)[0]
    # updateGFX is nop: not driving, and the route upvalues are out of reach.
    assert got["not_driving"] == "manual|none|false|none|noupv|false", got
    assert got["empty_plan"].split("|")[3] == "none", got
    assert got["no_route"].split("|")[3] == "none", got
    assert got["navgraph"].split("|")[3] == "7701" or got["navgraph"].split("|")[3] == "7702", got
    assert got["other_map"].endswith("|true|wp_2|upv|false"), got  # logged; the route map is the AI's own
    # wp_ nodes left in mapmgr.mapData after resetMapAndRoute (ai-local map and route nil): no ack.
    assert got["stale_mapmgr"] == "manual|none|true|none|upv|false", got
    assert got["nodebug"] == "manual|none|true|none|nodebug|false", got
    for case in ("not_driving", "empty_plan", "no_route", "navgraph", "stale_mapmgr", "nodebug"):
        assert not line_is_active(got[case], POINTS)[0], (case, got[case])
    print("readback chunk in Lua 5.1: route and map from updateGFX upvalues; stale mapmgr nodes are no ack")


# ---------------------------------------------------------------- tracker
def check_send_order_and_nonblocking_ack() -> None:
    tracker = LineTracker()
    veh = _Vehicle([_reply(target="none"), _reply(driving=False), GOOD], after=1)
    w = time.perf_counter()
    started = tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    assert time.perf_counter() - w < 0.05
    assert started is None and tracker.pending and tracker.phase == "weave"
    kinds = [e[0] if e[0] != "lua" else e[1] for e in veh.log]
    assert kinds == ['ai.setMode("manual")', 'ai.driveInLane("off")', "set_line", "readback"], kinds
    _name, line, cling = veh.log[2]
    assert cling is True and line == line_nodes(POINTS, 12.0)
    assert veh.log[3][1] == line_state_lua(len(POINTS))
    tag = tracker.tag()
    assert tag == {"ok": False, "mode": "", "wait_s": 0.0, "polls": 0, "pending": True}, tag
    events, spent = _run(tracker, veh)
    assert spent < 0.05, spent  # no waiting inside tick()
    assert [e[1].kind for e in events] == ["acked"], events
    t_ack, ev = events[0]
    # Reply 1 (no target) on tick 1, reply 2 (not driving) on tick 2, the good one on tick 3.
    assert abs(t_ack - 0.3) < 1e-9 and ev.ack.polls == 3, (t_ack, ev.ack)
    assert tracker.active and tracker.tag() == {"ok": True, "mode": "manual", "wait_s": 0.3, "polls": 3}
    after = veh.log[3:]
    assert all(e[0] in ("readback", "poll") for e in after), after  # nothing that could reset the route
    # The excursion commands for a line are the pre-commands only.
    pre = excursion_commands(12.0, "weave", 1, path=POINTS)
    assert pre == ['ai.setMode("manual")', 'ai.driveInLane("off")'], pre
    print(f"line sent with set_line; acked on tick {round(t_ack / TICK)} without waiting ({spent * 1e3:.2f} ms in tick)")


def check_no_ack_aborts() -> None:
    # Never a reply: the budget runs out at LINE_ACK_TIMEOUT_S after set_line.
    tracker = LineTracker()
    veh = _Vehicle([GOOD], after=None)
    assert tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave") is None
    events, spent = _run(tracker, veh)
    assert spent < 0.05, spent
    assert [e[1].kind for e in events] == [LINE_NOT_ACTIVE], events
    t_fail, ev = events[0]
    assert abs(t_fail - LINE_ACK_TIMEOUT_S) < 1e-9, t_fail
    assert ev.ack.reason == LINE_NOT_ACTIVE and "timeout" in ev.ack.detail and ev.ack.as_tag()["ok"] is False
    assert tracker.state == "idle"
    # One readback in flight at a time: the unanswered one is not re-sent.
    assert sum(1 for e in veh.log if e[0] == "readback") == 1

    # Every reply fails one condition: polled once per tick (>= LINE_ACK_POLL_S apart), aborted at the budget.
    for bad in (_reply(driving=False), _reply(target="1234"), _reply(xy=(55.0, 90.0)), "manual|10.00,16.00|true", None):
        tracker = LineTracker()
        veh = _Vehicle([bad], after=1)
        tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
        events, _ = _run(tracker, veh)
        assert [e[1].kind for e in events] == [LINE_NOT_ACTIVE], (bad, events)
        assert abs(events[0][0] - LINE_ACK_TIMEOUT_S) < 1e-9
        assert events[0][1].ack.polls == round(LINE_ACK_TIMEOUT_S / TICK), events[0][1].ack
        assert TICK >= LINE_ACK_POLL_S

    # A good reply seen after the budget does not count (no new tolerance).
    tracker = LineTracker()
    veh = _Vehicle([GOOD], after=6)
    tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    events, _ = _run(tracker, veh)
    assert [e[1].kind for e in events] == [LINE_NOT_ACTIVE], events
    # ...and one seen exactly at the budget does.
    tracker = LineTracker()
    veh = _Vehicle([GOOD], after=5)
    tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    events, _ = _run(tracker, veh)
    assert [e[1].kind for e in events] == ["acked"] and abs(events[0][1].ack.wait_s - 0.5) < 1e-9, events

    # set_line raised: no readback, failed at once.
    veh = _Vehicle([GOOD], fail=RuntimeError("socket closed"))
    failed = LineTracker().start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    assert failed is not None and failed.reason == LINE_NOT_ACTIVE and "socket closed" in failed.detail
    assert not any(e[0] == "readback" for e in veh.log)
    # No beamngpy handle: no hand-built Lua line instead.
    veh = _Vehicle([GOOD], ai=False)
    failed = LineTracker().start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    assert failed is not None and "no beamngpy AI set_line" in failed.detail
    assert not any("setLine" in str(e) or "driveUsingPath" in str(e) for e in veh.log)
    # A vehicle that can only answer by blocking: refused, not waited on.
    veh = _Vehicle([GOOD], readback=False)
    failed = LineTracker().start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    assert failed is not None and "without blocking" in failed.detail, failed
    print("no ack within LINE_ACK_TIMEOUT_S of the send aborts line_not_active; a late reply does not count")


def check_recheck() -> None:
    for replies, after, want in (
        ([GOOD], 1, None),
        ([GOOD, _reply(driving=False)], 1, LINE_LOST),
        ([GOOD, _reply(target="1234")], 1, LINE_LOST),
        ([GOOD, GOOD, _reply(xy=None, target="none", src="noupv", driving=False)], 1, LINE_LOST),
    ):
        tracker = LineTracker()
        veh = _Vehicle(replies, after=after)
        tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
        events, spent = _run(tracker, veh, ticks=60)
        kinds = [e[1].kind for e in events]
        assert kinds[0] == "acked", kinds
        assert spent < 0.05
        rest, _ = _run(tracker, veh, t0=100.0 + events[0][0], ticks=60)
        kinds = [e[1].kind for e in rest]
        if want is None:
            sent = [t for t, e in rest if e.kind == "recheck_sent"]
            assert len(sent) >= 5 and all(abs((b - a) - LINE_RECHECK_S) < 1e-6 for a, b in zip(sent, sent[1:])), sent
            assert "recheck_ok" in kinds and LINE_LOST not in kinds, kinds
        else:
            assert kinds[-1] == LINE_LOST and rest[-1][1].ack.reason == LINE_LOST, kinds
            assert tracker.state == "idle"
    # A recheck with no reply within LINE_ACK_TIMEOUT_S is line_lost too.
    tracker = LineTracker()
    veh = _Vehicle([GOOD], after=1)
    tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    _run(tracker, veh)
    veh.after = None
    rest, _ = _run(tracker, veh, t0=100.1, ticks=40)
    sent = [t for t, e in rest if e.kind == "recheck_sent"]
    assert rest[-1][1].kind == LINE_LOST and "no reply" in rest[-1][1].detail, rest
    assert abs(rest[-1][0] - sent[0] - LINE_ACK_TIMEOUT_S) < 1e-6, (sent, rest[-1][0])
    # drop() leaves the reply in flight to be drained later, never re-used.
    tracker = LineTracker()
    veh = _Vehicle([GOOD], after=2)
    tracker.start(veh, POINTS, 12.0, queue=_queue, now=100.0, phase="weave")
    tracker.drop()
    assert tracker.state == "idle" and tracker.tag() is None
    assert tracker.tick(veh, 100.1).kind == "" and tracker.tick(veh, 100.2).kind == ""
    assert tracker._orphans == []
    print("rechecks every LINE_RECHECK_S, non-blocking; a gone route or a silent recheck is line_lost")


# ---------------------------------------------------------------- beamngpy transport
class _FakeBngSocket:
    """beamngpy 1.36 PrefixedLengthSocket stand-in: ``skt`` is a real socket, ``recv()`` one length-prefixed message."""

    def __init__(self, skt: socket.socket) -> None:
        self.skt = skt

    def _exactly(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            got = self.skt.recv(n - len(out))
            assert got
            out += got
        return out

    def recv(self) -> bytes:
        (length,) = struct.unpack("!I", self._exactly(4))
        return self._exactly(length)


class _FakeBngConnection:
    """The beamngpy 1.36 ``Connection`` parts the transport uses (connection.py 97, 211-259)."""

    def __init__(self, skt: socket.socket) -> None:
        self.skt = _FakeBngSocket(skt)
        self.received_messages: dict = {}
        self.req_id = 0
        self.sent: list = []

    def _unpack_data(self, data: bytes) -> dict:
        return json.loads(data.decode("utf-8"))

    def recv(self, req_id: int) -> dict:  # what a blocking call (sensor poll, queued command) does
        if req_id in self.received_messages:
            return self.received_messages.pop(req_id)
        while True:
            message = self._unpack_data(self.skt.recv())
            _id = int(message.pop("_id"))
            if _id == req_id:
                return message
            self.received_messages[_id] = message


class _Resp:
    def __init__(self, req_id: int) -> None:
        self.req_id = req_id


class _BngVehicle:
    def __init__(self, skt: socket.socket) -> None:
        self.connection = _FakeBngConnection(skt)

    def _send(self, data: dict) -> _Resp:
        conn = self.connection
        rid = conn.req_id
        conn.req_id += 1
        conn.sent.append((rid, dict(data)))
        return _Resp(rid)


def _frame(obj: dict) -> bytes:
    body = json.dumps(obj).encode("utf-8")
    return struct.pack("!I", len(body)) + body


def check_beamngpy_transport() -> None:
    a, b = socket.socketpair()
    try:
        veh = _BngVehicle(a)
        rb = ai_line.start_readback(veh, "return 1")
        rid, data = veh.connection.sent[0]
        assert data == {"type": "QueueLuaCommandVE", "chunk": "return 1", "resp": True}, data
        w = time.perf_counter()
        assert rb.poll() == (False, None, "")  # nothing on the socket: no wait
        assert time.perf_counter() - w < 0.05
        b.sendall(_frame({"_id": rid, "type": "ExecutedLuaChunkVE", "resp": GOOD}))
        deadline = time.time() + 2.0
        got = rb.poll()
        while not got[0] and time.time() < deadline:
            _REAL_SLEEP(0.005)
            got = rb.poll()
        assert got == (True, GOOD, ""), got
        # A blocking call made in between files the reply by id; the poll takes it from there.
        rb2 = ai_line.start_readback(veh, "return 2")
        rid2 = veh.connection.sent[-1][0]
        other = 99
        b.sendall(_frame({"_id": rid2, "type": "ExecutedLuaChunkVE", "resp": "two"}))
        b.sendall(_frame({"_id": other, "type": "Ack"}))
        assert veh.connection.recv(other) == {"type": "Ack"}
        assert rid2 in veh.connection.received_messages
        assert rb2.poll() == (True, "two", "") and rid2 not in veh.connection.received_messages
        # A Lua error reply is done, with the error.
        rb3 = ai_line.start_readback(veh, "error()")
        rid3 = veh.connection.sent[-1][0]
        b.sendall(_frame({"_id": rid3, "bngError": "boom"}))
        deadline = time.time() + 2.0
        got = rb3.poll()
        while not got[0] and time.time() < deadline:
            _REAL_SLEEP(0.005)
            got = rb3.poll()
        assert got[0] and got[1] is None and "boom" in got[2], got
    finally:
        a.close()
        b.close()
    print("beamngpy transport: QueueLuaCommandVE resp=True sent, reply read only when the socket has it")


# ---------------------------------------------------------------- collector loop
def _run_collector(behaviour: str, *, ready_after: int = 1) -> tuple[str, list, list[dict], dict]:
    """Drive the real collect loop on a straight painted road with a fake vehicle.

    ``behaviour``: ``never`` (the readback never shows the sent line), ``ok``
    (it does, each reply ready after ``ready_after`` polls, one per tick),
    ``lose`` (ok, then the second recheck finds the route gone), ``snap``
    (ok, and set_line moves the car 3 m: the teleport the smoke run must catch).
    """
    import io

    import numpy as np

    import scripts.collect_scene_lanes as collect
    from python.collect.auto_mode import stop_file
    from python.debug.gt_geom import RoadPoly

    root = Path(tempfile.mkdtemp())
    out = root / "scene"
    stop = stop_file(out)
    clock = {"t": 1000.0}

    class Vehicle(_Vehicle):
        model = "etk800"

        def __init__(self) -> None:
            super().__init__([_reply(target="none", driving=False, src="noupv", xy=None)], after=ready_after)
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

        def _send_lua_readback(self, chunk: str) -> _Handle:
            handle = super()._send_lua_readback(chunk)
            if behaviour == "never" or self.sent is None:
                return handle
            self.reads += 1
            if behaviour == "lose" and self.reads >= 3 and self.reads > ready_after:
                handle.reply = _reply(xy=None, target="none", driving=True, mode="manual")
            else:
                handle.reply = _reply(xy=self.sent)
            return handle

    vehicle = Vehicle()
    real_set_line = vehicle.ai.set_line

    def set_line(line: list, cling: bool = True) -> None:
        real_set_line(line, cling)
        vehicle.sent = (float(line[-1]["pos"][0]), float(line[-1]["pos"][1]))
        vehicle.reads = 0
        if behaviour == "snap":
            vehicle.y -= 3.0  # driveUsingPath teleports to the first node

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
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds or 0.0))
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
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    status["_sleeps"] = sleeps
    return text, list(vehicle.log), rows, status


def _line_sends(log: list) -> list[int]:
    return [i for i, e in enumerate(log) if e[0] == "set_line"]


def _phase(r: dict) -> str:
    return (r.get("excursion") or {}).get("phase")


def check_collector_aborts_weave_without_ack() -> None:
    text, log, rows, status = _run_collector("never")
    sends = _line_sends(log)
    assert sends, "no line was sent:\n" + "\n".join(ln for ln in text.splitlines() if "gate" in ln)[:3000]
    assert "abort line_not_active" in text, text[-3000:]
    for i in sends:
        # After each line: only readbacks, then the abort's lane-follow restore.
        rest = [e for e in log[i + 1:] if e[0] == "lua"]
        assert rest and rest[0][1] == 'ai.setMode("span")', rest[:3]
        assert any(e[1] == 'ai.driveInLane("on")' for e in rest[:4])
    assert not any(e[0] == "lua" and "setLine" in e[1] for e in log)
    phases = [_phase(r) for r in rows]
    assert "weave" not in phases and "lane_change" not in phases, sorted(set(phases))
    pending = [r for r in rows if _phase(r) == "weave_pending"]
    # Sent tick + ticks until the 0.5 s budget ran out (0.1 s ticks): ~5 rows per line, all pending, none ok.
    assert pending and len(pending) <= len(sends) * (round(LINE_ACK_TIMEOUT_S / TICK) + 1), (len(pending), len(sends))
    assert all(r["excursion"]["line_ack"] == {**r["excursion"]["line_ack"], "ok": False, "pending": True} for r in pending)
    aborts = [r for r in rows if _phase(r) == "abort"]
    assert aborts and all(r["excursion"]["reason"] == LINE_NOT_ACTIVE for r in aborts), {r["excursion"]["reason"] for r in aborts}
    assert status["weave_stats"]["aborted"]["weave"][LINE_NOT_ACTIVE] == len(sends), status["weave_stats"]["aborted"]
    bar = status["weave_stats"]["line_smoke"]["pass_bar"]
    assert bar["line_not_active"] == len(sends) and bar["pass"] is False, bar
    print(f"collector: {len(sends)} lines without an ack -> {len(pending)} weave_pending rows, aborted line_not_active")


def check_collector_pending_then_weave() -> None:
    """The ack arrives on the 3rd tick after the send: the loop ticks on, and rows until then are weave_pending."""
    text, log, rows, status = _run_collector("ok", ready_after=3)
    sends = _line_sends(log)
    assert sends and "line_not_active" not in text, text[-3000:]
    assert all(s == 0.1 for s in status["_sleeps"]), sorted(set(status["_sleeps"]))  # only tick sleeps
    first = next(k for k, r in enumerate(rows) if _phase(r) in ("weave", "weave_pending"))
    run = rows[first:first + 8]
    phases = [_phase(r) for r in run]
    # Sent on row 0; reply ready on the 3rd poll (rows 1, 2, 3 poll): rows 0-2 pending, row 3 is the first weave row.
    assert phases[:3] == ["weave_pending"] * 3 and phases[3:6] == ["weave"] * 3, phases
    # The loop kept its 0.1 s tick while the ack was pending: no gap, one row per tick.
    gaps = [round(b["t"] - a["t"], 6) for a, b in zip(rows[first - 3:first + 6], rows[first - 2:first + 7])]
    assert all(abs(g - 0.1) < 1e-6 for g in gaps), gaps
    for r in run[:3]:
        ack = r["excursion"]["line_ack"]
        assert ack["ok"] is False and ack["pending"] is True, ack
    acked = run[3]["excursion"]["line_ack"]
    assert acked["ok"] is True and acked["polls"] == 1 and abs(acked["wait_s"] - 0.3) < 1e-6, acked
    weave = [r for r in rows if _phase(r) == "weave"]
    assert len(weave) >= 100, len(weave)
    assert all((r["excursion"].get("line_ack") or {}).get("ok") is True for r in weave)
    assert not any("line_ack" in r["excursion"] for r in rows if _phase(r) in ("cruise", "abort"))
    # Pending rows are masked with the excursion rows.
    assert all(_phase(r) in EXCURSION_MASK_PHASES for r in rows if _phase(r) != "cruise")
    for i in sends:
        later = [e for e in log[i + 1:] if e[0] == "lua"]
        assert not later or later[0][1].startswith('ai.setMode("span")'), later[:2]
    # Smoke instrumentation: weave start window, ack and recheck tick spacing, yaw vs command.
    sm = status["weave_stats"]["line_smoke"]
    assert sm["weave_starts"] == len(sends) and sm["starts"], sm
    st = sm["starts"][0]
    assert st["dist_first_m"] is not None and st["dist_first_m"] < 3.0 and st["snap"] is False, st
    dts = [t["dt_s"] for t in st["ticks"]]
    assert min(dts) <= -0.9 and max(dts) >= 0.9, dts
    assert all(t["v_mps"] == 12.0 for t in st["ticks"]) and all(t["dpos_m"] in (None, 1.2) for t in st["ticks"]), st["ticks"]
    assert sm["events"].get("ack") == len(sends) and sm["events"].get("recheck_sent", 0) >= 5, sm["events"]
    acks = [e for e in sm["event_windows"] if e["kind"] == "ack"]
    assert acks and all(abs(s - 0.1) < 1e-6 for s in acks[0]["tick_s"]) and acks[0]["max_tick_s"] <= 0.1 + 1e-6, acks[0]
    assert sm["pass_bar"]["max_tick_jump_m"] is not None and sm["pass_bar"]["max_tick_jump_m"] < 0.05, sm["pass_bar"]
    assert sm["pass_bar"]["line_not_active"] == 0 and sm["pass_bar"]["line_lost"] == 0
    assert "collect smoke weave_start" in text and "collect smoke ack" in text and "collect smoke recheck_sent" in text
    assert "collect excursion current#0 weave line acked wait=0.300s polls=1" in text, text[-2000:]
    assert status["notes"]["weave_stats"]["line_smoke"].startswith("smoke-run instrumentation")
    print(f"collector: 3 weave_pending rows, then {len(weave)} weave rows; ticks 0.1 s throughout; smoke logged")


def check_collector_line_lost() -> None:
    text, log, rows, status = _run_collector("lose")
    assert "abort line_lost" in text, text[-3000:]
    aborts = {r["excursion"]["reason"] for r in rows if _phase(r) == "abort"}
    assert LINE_LOST in aborts, aborts
    weave = [r for r in rows if _phase(r) == "weave"]
    # Acked on the tick after the send, two rechecks at LINE_RECHECK_S: ~2 s of weave, then aborted.
    assert 10 <= len(weave) <= 30, len(weave)
    assert status["weave_stats"]["line_smoke"]["events"].get("line_lost", 0) >= 1
    print("collector: a route lost mid-line aborts line_lost")


def check_collector_snap_is_flagged() -> None:
    text, _log, _rows, status = _run_collector("snap")
    sm = status["weave_stats"]["line_smoke"]
    assert sm["starts"] and sm["starts"][0]["snap"] is True and sm["pass_bar"]["snaps_at_weave_start"] >= 1, sm["starts"][:1]
    assert sm["pass_bar"]["max_tick_jump_m"] > 2.5 and sm["pass_bar"]["pass"] is False, sm["pass_bar"]
    assert "snap=KILL" in text
    print("collector: a 3 m teleport at the weave start is a snap (KILL) in line_smoke")


def check_mask_phases() -> None:
    assert PENDING == ("weave_pending", "lane_change_pending", "recover_pending")
    assert all(pending_phase(p) in EXCURSION_MASK_PHASES for p in ACTIVE)
    assert set(ACTIVE) <= set(EXCURSION_MASK_PHASES) and "abort" in EXCURSION_MASK_PHASES
    assert "cruise" not in EXCURSION_MASK_PHASES
    print("excursion mask phases include every *_pending phase")


def main() -> None:
    check_ack_conditions()
    check_lua_chunk()
    check_send_order_and_nonblocking_ack()
    check_no_ack_aborts()
    check_recheck()
    check_beamngpy_transport()
    check_mask_phases()
    check_collector_aborts_weave_without_ack()
    check_collector_pending_then_weave()
    check_collector_line_lost()
    check_collector_snap_is_flagged()
    print("weave line ack ok")


if __name__ == "__main__":
    main()
