"""Excursion lines for the BeamNG.tech 0.39.4.0 AI: send with beamngpy, then read back.

The 0.39.4.0 vehicle ``ai`` module has no ``setLine``. Up to the fix the
collector queued ``ai.setLine({...}, true)`` as raw Lua: every call failed
in the vehicle VM ("attempt to call field 'setLine' (a nil value)"), and the
``ai.setMode("manual")`` sent just before it had already dropped the span
route, so the car ran the whole weave on held inputs.

The supported call is beamngpy's ``vehicle.ai.set_line(line, cling=True)``
(beamngpy 1.36). It is not a Lua string: it sends a ``SetAiLine`` request,
and Tech's vehicle-side ``tech/techCore.lua`` ``handleSetAiLine`` turns it
into ``ai.driveUsingPath({script = nodes, wpSpeeds = speeds})``.

Where the line lives (0.39.4.0 ``lua/vehicle/ai.lua``, ``mapmgr.lua``):

* ``driveUsingPath{script}`` (ai.lua 6010-6048) nils ``mapmgr.mapData``
  (``mapmgr.setCustomMap()``), builds the ``wp_1 .. wp_n`` nodes in a local
  ``graphpath`` and keeps it in the ai-local ``scriptData.mapData``. Nothing
  is in ``mapmgr.mapData`` yet.
* The next manual-mode ``updateGFX`` that has a map (5137-5149, 5309-5311)
  runs ``setScriptedPath`` (3901-3934): ``mapmgr.setCustomMap(arg.mapData)``
  (mapmgr.lua 72-74 sets ``M.mapData``), ai-local ``mapData =
  mapmgr.mapData``, ai-local ``currentRoute = createNewRoute(path)``.
* ``resetMapAndRoute`` (3779-3790: every ``setMode`` to a driving mode)
  clears the ai-local ``mapData`` and ``currentRoute`` but leaves
  ``mapmgr.mapData``: the ``wp_`` nodes of an old line can sit in
  ``mapmgr.mapData`` while the AI has no route at all.

So a ``wp_n`` node in ``mapmgr.mapData`` is not an ack, and ``mode ==
"manual"`` is only the echo of our own ``setMode``. The route and its map
are ai-local upvalues (ai.lua 135 ``mapData``, 137 ``currentRoute``);
``ai.getState()`` returns the module table, not the route. ``line_state_lua``
reads them with ``debug.getupvalue(ai.updateGFX, ...)``: ``ai.updateGFX`` is
the ai-local ``updateGFX`` while the AI drives (5783, ``isDriving`` 6167-6168),
and that function names both upvalues (5137-5146). ai.lua itself (166-169)
and powertrain.lua (1207) call ``debug.getupvalue`` in the vehicle VM.

The line is acked only when ALL of these hold in one readback:

* the last node ``wp_n`` of the AI's own route map is within LINE_MATCH_M of
  the last point we sent (a stale earlier line does not count);
* ``ai.isDriving()`` is true;
* the AI's current target node, ``currentRoute.path[currentRoute.plan[2].pathidx]``
  (the plan node after the car's own, ai.lua 1104, 1543, 2630-2632), is a
  ``wp_`` node.

The ack does not block the collector's tick loop. ``LineTracker.start``
sends the pre-commands, ``set_line`` and one readback request, and returns.
``LineTracker.tick`` (once per collector tick) looks for the reply without
waiting, sends the next readback at most every LINE_ACK_POLL_S, and fails the
line ``line_not_active`` once LINE_ACK_TIMEOUT_S has passed since
``set_line`` without an ack. Once acked it sends a recheck every
LINE_RECHECK_S; a recheck that fails, or has no reply within
LINE_ACK_TIMEOUT_S, ends the move ``line_lost``.
"""
from __future__ import annotations

import math
import select
from dataclasses import dataclass, field
from typing import Any, Callable

# 0.5 s from set_line to an ok readback: the route is built on the first
# ``ai`` graphics frame after the request that has a map (one frame, ~17 ms
# at 60 fps). 0.5 s is 10 frames at a degraded 20 fps, and at 30 m/s it is
# 15 m of a line that is at least MIN_PATH_M (25 m) long.
LINE_ACK_TIMEOUT_S = 0.5
# At most one readback in flight; the next one goes out no sooner than this.
LINE_ACK_POLL_S = 0.05
# While a line runs it is read back once a second; a route dropped later (a
# map reload replaces the AI's custom map and clears its route) ends the
# excursion as ``line_lost`` instead of drifting. A recheck reply has the
# same LINE_ACK_TIMEOUT_S budget.
LINE_RECHECK_S = 1.0
# The last node is placed at our last point (x, y); z is the ground height.
LINE_MATCH_M = 0.5
LINE_NODE_PREFIX = "wp_"

LINE_NOT_ACTIVE = "line_not_active"
LINE_LOST = "line_lost"

# Rows of a move whose line is sent but not acked yet carry this suffix on
# their phase (``weave_pending``): never ``weave``, and masked like one.
PENDING_SUFFIX = "_pending"


def pending_phase(phase: str) -> str:
    return f"{phase}{PENDING_SUFFIX}"


def line_nodes(points: list[tuple[float, float]], speed_mps: float) -> list[dict[str, Any]]:
    """``vehicle.ai.set_line`` input: ``pos`` (x, y, z) and ``speed`` per point. z is clung."""
    speed = float(speed_mps)
    return [{"pos": (round(float(x), 2), round(float(y), 2), 0.0), "speed": round(speed, 2)} for x, y in points]


def pre_line_commands() -> list[str]:
    """Vehicle-VM Lua sent before ``set_line``. Nothing is sent after it.

    ``setMode("manual")`` leaves span (it resets the span route and plan).
    ``driveInLane("off")`` only sets a flag the line's planner reads.
    """
    return ['ai.setMode("manual")', 'ai.driveInLane("off")']


def line_state_lua(n_points: int) -> str:
    """Vehicle-VM chunk: ``mode|x,y|driving|target|src|same_map``.

    * ``x,y``: node ``wp_<n_points>`` in the AI's own route map (the
      ai-local ``mapData`` upvalue), ``none`` when absent;
    * ``driving``: ``ai.isDriving()``;
    * ``target``: ``currentRoute.path[currentRoute.plan[2].pathidx]``, ``none``
      without a route or plan;
    * ``src``: ``upv`` when ``updateGFX`` has the ``currentRoute`` upvalue,
      ``noupv`` when it does not (``nop``: the AI is not driving), ``nodebug``
      without the Lua debug library;
    * ``same_map``: the route map is ``mapmgr.mapData`` (logged only).
    """
    node = f"{LINE_NODE_PREFIX}{int(n_points)}"
    return (
        "local function uv(f, w) "
        "if type(debug) ~= 'table' or type(debug.getupvalue) ~= 'function' then return nil, 'nodebug' end "
        "if type(f) ~= 'function' then return nil, 'noupv' end "
        "local i = 1 "
        "while true do "
        "local n, v = debug.getupvalue(f, i) "
        "if n == nil then return nil, 'noupv' end "
        "if n == w then return v, 'upv' end "
        "i = i + 1 "
        "end "
        "end "
        "local s = ai.getState() "
        "local g = ai.updateGFX "
        "local r, src = uv(g, 'currentRoute') "
        "local m = uv(g, 'mapData') "
        f"local p = m and m.positions and m.positions['{node}'] "
        "local t = 'none' "
        "if type(r) == 'table' and type(r.path) == 'table' and type(r.plan) == 'table' then "
        "local q = r.plan[2] "
        "local k = q and q.pathidx "
        "if k and r.path[k] then t = tostring(r.path[k]) end "
        "end "
        "return tostring(s and s.mode) .. '|' .. "
        "(p and string.format('%.2f,%.2f', p.x, p.y) or 'none') .. '|' .. "
        "tostring(ai.isDriving()) .. '|' .. t .. '|' .. tostring(src) .. '|' .. "
        "tostring(m ~= nil and m == mapmgr.mapData)"
    )


@dataclass(frozen=True)
class LineState:
    mode: str = ""
    last_xy: tuple[float, float] | None = None
    driving: bool = False
    target: str = ""
    src: str = ""
    same_map: bool = False
    raw: str = ""


def parse_line_state(reply: Any) -> LineState:
    """Every field of the readback. A short or empty reply leaves the missing fields false/empty."""
    text = "" if reply is None else str(reply)
    parts = text.split("|")

    def part(i: int) -> str:
        return parts[i] if len(parts) > i else ""

    xy: tuple[float, float] | None = None
    if part(1) not in ("", "none"):
        try:
            a, b = part(1).split(",")
            xy = (float(a), float(b))
        except ValueError:
            xy = None
    target = part(3)
    return LineState(
        mode=part(0),
        last_xy=xy,
        driving=part(2) == "true",
        target="" if target == "none" else target,
        src=part(4),
        same_map=part(5) == "true",
        raw=text,
    )


def line_is_active(reply: Any, points: list[tuple[float, float]]) -> tuple[bool, str]:
    """(ok, why not). ok needs the last point match, ``isDriving()`` and a ``wp_`` target."""
    st = reply if isinstance(reply, LineState) else parse_line_state(reply)
    why: list[str] = []
    if not points or st.last_xy is None:
        why.append("no_last_node")
    else:
        lx, ly = points[-1]
        d = math.hypot(st.last_xy[0] - float(lx), st.last_xy[1] - float(ly))
        if d > LINE_MATCH_M:
            why.append(f"last_node_off={d:.2f}m")
    if not st.driving:
        why.append("not_driving")
    if not st.target.startswith(LINE_NODE_PREFIX):
        # ``src`` says why there is no target when the upvalues were not readable.
        why.append(f"target={st.target or 'none'}" + (f" src={st.src}" if st.src and st.src != "upv" else ""))
    return not why, ",".join(why)


# ---------------------------------------------------------------- readback transport
class _Readback:
    """One readback in flight. ``poll()`` -> (done, reply, error); never waits for the simulator."""

    def poll(self) -> tuple[bool, Any, str]:  # pragma: no cover - interface
        raise NotImplementedError


class _BeamngpyReadback(_Readback):
    """``QueueLuaCommandVE`` with ``resp=True`` on a beamngpy 1.36 vehicle connection, without the wait.

    ``Vehicle.queue_lua_command`` is ``self._send(data).recv(...)``: the
    send, then a blocking read. Here the request goes out the same way
    (``vehicle._send``) and the reply is picked up later:

    * a blocking call made since (a sensor poll, a queued command) reads
      replies until its own and keeps the others in
      ``connection.received_messages`` by request id;
    * otherwise, when the socket has data (``select`` with a zero timeout),
      one message is read and filed by id, as ``Connection.recv`` does.
    """

    def __init__(self, vehicle: Any, chunk: str) -> None:
        self.conn = vehicle.connection
        self.req_id = int(vehicle._send({"type": "QueueLuaCommandVE", "chunk": chunk, "resp": True}).req_id)

    def _raw_socket(self) -> Any:
        skt = getattr(self.conn, "skt", None)
        return getattr(skt, "skt", None)

    def _take(self) -> tuple[bool, Any, str]:
        msg = self.conn.received_messages.pop(self.req_id)
        if isinstance(msg, Exception):
            return True, None, f"{type(msg).__name__}: {msg}"
        if not isinstance(msg, dict):
            return True, None, f"reply {type(msg).__name__}"
        return True, msg.get("resp"), ""

    def poll(self) -> tuple[bool, Any, str]:
        if self.req_id in self.conn.received_messages:
            return self._take()
        raw = self._raw_socket()
        if raw is None:
            return True, None, "vehicle connection has no socket"
        try:
            ready, _w, _x = select.select([raw], [], [], 0)
        except (OSError, ValueError) as exc:
            return True, None, f"select: {type(exc).__name__}: {exc}"
        if not ready:
            return False, None, ""
        try:
            message = self.conn._unpack_data(self.conn.skt.recv())
        except Exception as exc:  # BNGDisconnectedError, socket errors
            return True, None, f"recv: {type(exc).__name__}: {exc}"
        _id = message.pop("_id", None) if isinstance(message, dict) else None
        if _id is None:
            return True, None, "reply without _id"
        if "bngError" in message or "bngValueError" in message:
            message = RuntimeError(str(message.get("bngError") or message.get("bngValueError")))
        self.conn.received_messages[int(_id)] = message
        if self.req_id in self.conn.received_messages:
            return self._take()
        return False, None, ""


def start_readback(vehicle: Any, chunk: str) -> _Readback:
    """Send one readback without waiting for it. Raises when this vehicle cannot do that.

    ``vehicle.send_lua_readback(chunk)`` (an adapter or a test vehicle) wins;
    else a beamngpy vehicle (``_send`` and a ``connection`` with
    ``received_messages``). There is no blocking fallback.
    """
    own = getattr(vehicle, "send_lua_readback", None)
    if callable(own):
        return own(chunk)
    conn = getattr(vehicle, "connection", None)
    if callable(getattr(vehicle, "_send", None)) and isinstance(getattr(conn, "received_messages", None), dict):
        return _BeamngpyReadback(vehicle, chunk)
    raise RuntimeError("vehicle cannot read Lua back without blocking")


# ---------------------------------------------------------------- the ack
@dataclass
class LineAck:
    ok: bool
    reason: str = ""
    detail: str = ""
    mode: str = ""
    wait_s: float = 0.0
    polls: int = 0
    replies: list[str] = field(default_factory=list)
    pending: bool = False

    def as_tag(self) -> dict[str, Any]:
        tag: dict[str, Any] = {"ok": bool(self.ok), "mode": self.mode, "wait_s": round(float(self.wait_s), 3), "polls": int(self.polls)}
        if self.pending:
            tag["pending"] = True
        elif not self.ok:
            tag["reason"] = self.reason
            tag["detail"] = self.detail
        return tag


def send_line(vehicle: Any, points: list[tuple[float, float]], speed_mps: float) -> str:
    """``vehicle.ai.set_line``. Empty string when sent; otherwise why not (no Lua fallback)."""
    ai = getattr(vehicle, "ai", None)
    fn = getattr(ai, "set_line", None) if ai is not None else None
    if not callable(fn):
        return "vehicle has no beamngpy AI set_line"
    try:
        fn(line_nodes(points, speed_mps), cling=True)
    except Exception as exc:  # BNGError, socket errors: the line did not go out
        return f"set_line: {type(exc).__name__}: {exc}"
    return ""


@dataclass(frozen=True)
class LineEvent:
    """What ``LineTracker.tick`` saw this tick. ``kind``: '', 'acked', 'recheck_sent', 'recheck_ok', or a failure."""

    kind: str = ""
    detail: str = ""
    ack: LineAck | None = None


IDLE, PENDING, ACTIVE_LINE = "idle", "pending", "active"


class LineTracker:
    """The line in flight: pending until acked, then rechecked. All calls return at once."""

    def __init__(
        self,
        *,
        timeout_s: float = LINE_ACK_TIMEOUT_S,
        poll_s: float = LINE_ACK_POLL_S,
        recheck_s: float = LINE_RECHECK_S,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.recheck_s = float(recheck_s)
        self.state = IDLE
        self.phase = ""
        self.points: list[tuple[float, float]] = []
        self.ack = LineAck(ok=False)
        self.sent_t = 0.0
        self.checked_t = 0.0
        self._rb: _Readback | None = None
        self._rb_t = 0.0
        self._orphans: list[tuple[_Readback, float]] = []

    # -- state
    @property
    def pending(self) -> bool:
        return self.state == PENDING

    @property
    def active(self) -> bool:
        return self.state == ACTIVE_LINE

    def tag(self) -> dict[str, Any] | None:
        return None if self.state == IDLE else self.ack.as_tag()

    def drop(self) -> None:
        """The move ended some other way. A readback still in flight is drained later and ignored."""
        if self._rb is not None:
            self._orphans.append((self._rb, self._rb_t))
        self._rb = None
        self.state = IDLE
        self.phase = ""
        self.points = []

    # -- sending
    def _send_readback(self, vehicle: Any, now: float) -> str:
        try:
            self._rb = start_readback(vehicle, line_state_lua(len(self.points)))
        except Exception as exc:
            self._rb = None
            return f"readback: {type(exc).__name__}: {exc}"
        self._rb_t = float(now)
        return ""

    def start(
        self,
        vehicle: Any,
        points: list[tuple[float, float]],
        speed_mps: float,
        *,
        queue: Callable[[Any, str], None],
        now: float,
        phase: str = "",
    ) -> LineAck | None:
        """Pre-commands, ``set_line``, one readback request. None: pending. A LineAck: failed now."""
        self.drop()
        for command in pre_line_commands():
            try:
                queue(vehicle, command)
            except Exception as exc:
                return LineAck(ok=False, reason=LINE_NOT_ACTIVE, detail=f"pre: {type(exc).__name__}: {exc}")
        err = send_line(vehicle, points, speed_mps)
        if err:
            return LineAck(ok=False, reason=LINE_NOT_ACTIVE, detail=err)
        self.points = [(float(x), float(y)) for x, y in points]
        self.phase = str(phase)
        self.sent_t = float(now)
        self.ack = LineAck(ok=False, reason=LINE_NOT_ACTIVE, pending=True)
        err = self._send_readback(vehicle, now)
        if err:
            failed = LineAck(ok=False, reason=LINE_NOT_ACTIVE, detail=err)
            self.drop()
            return failed
        self.state = PENDING
        return None

    # -- per tick
    def _drain_orphans(self, now: float) -> None:
        keep = []
        for rb, t in self._orphans:
            try:
                done, _reply, _err = rb.poll()
            except Exception:
                done = True
            if not done and now - t < 10.0 * self.timeout_s:
                keep.append((rb, t))
        self._orphans = keep

    def _poll(self) -> tuple[bool, Any, str]:
        if self._rb is None:
            return False, None, ""
        try:
            done, reply, err = self._rb.poll()
        except Exception as exc:
            done, reply, err = True, None, f"{type(exc).__name__}: {exc}"
        if done:
            self._rb = None
        return done, reply, err

    def tick(self, vehicle: Any, now: float) -> LineEvent:
        now = float(now)
        self._drain_orphans(now)
        if self.state == PENDING:
            return self._tick_pending(vehicle, now)
        if self.state == ACTIVE_LINE:
            return self._tick_active(vehicle, now)
        return LineEvent()

    def _tick_pending(self, vehicle: Any, now: float) -> LineEvent:
        ack = self.ack
        elapsed = now - self.sent_t
        done, reply, err = self._poll()
        if done:
            ack.polls += 1
            ack.replies.append(str(reply) if not err else f"error {err}")
            st = parse_line_state(reply)
            ack.mode = st.mode
            ok, why = (False, err) if err else line_is_active(st, self.points)
            if ok and elapsed <= self.timeout_s + 1e-9:
                ack.ok, ack.reason, ack.detail, ack.pending = True, "", "", False
                ack.wait_s = elapsed
                self.state = ACTIVE_LINE
                self.checked_t = now
                return LineEvent("acked", f"wait={elapsed:.3f}s polls={ack.polls}", ack)
            ack.detail = why
        if elapsed >= self.timeout_s - 1e-9:
            ack.pending = False
            ack.wait_s = elapsed
            last = ack.replies[-1] if ack.replies else "no reply"
            ack.detail = f"timeout {self.timeout_s:.2f}s polls={ack.polls} last={last} ({ack.detail or 'no reply'})"
            failed = ack
            self.drop()
            return LineEvent(LINE_NOT_ACTIVE, failed.detail, failed)
        if self._rb is None and now - self._rb_t >= self.poll_s - 1e-9:
            err = self._send_readback(vehicle, now)
            if err:
                ack.pending = False
                ack.detail = err
                ack.wait_s = elapsed
                failed = ack
                self.drop()
                return LineEvent(LINE_NOT_ACTIVE, err, failed)
        return LineEvent()

    def _tick_active(self, vehicle: Any, now: float) -> LineEvent:
        if self._rb is not None:
            done, reply, err = self._poll()
            if done:
                ok, why = (False, err) if err else line_is_active(reply, self.points)
                if ok:
                    return LineEvent("recheck_ok", f"rtt={now - self._rb_t:.3f}s", self.ack)
                detail = f"recheck last={reply!s} ({why})"
                lost = LineAck(ok=False, reason=LINE_LOST, detail=detail, mode=parse_line_state(reply).mode, polls=1, replies=[str(reply)])
                self.drop()
                return LineEvent(LINE_LOST, detail, lost)
            if now - self._rb_t >= self.timeout_s - 1e-9:
                detail = f"recheck no reply in {self.timeout_s:.2f}s"
                lost = LineAck(ok=False, reason=LINE_LOST, detail=detail, polls=1)
                self.drop()
                return LineEvent(LINE_LOST, detail, lost)
            return LineEvent()
        if now - self.checked_t >= self.recheck_s - 1e-9:
            self.checked_t = now
            err = self._send_readback(vehicle, now)
            if err:
                lost = LineAck(ok=False, reason=LINE_LOST, detail=err, polls=0)
                self.drop()
                return LineEvent(LINE_LOST, err, lost)
            return LineEvent("recheck_sent", "", self.ack)
        return LineEvent()
