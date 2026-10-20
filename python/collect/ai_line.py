"""Excursion lines for the BeamNG.tech 0.39.4.0 AI: send with beamngpy, then read back.

The 0.39.4.0 vehicle ``ai`` module has no ``setLine``. Up to the fix the
collector queued ``ai.setLine({...}, true)`` as raw Lua: every call failed
in the vehicle VM ("attempt to call field 'setLine' (a nil value)"), and the
``ai.setMode("manual")`` sent just before it had already dropped the span
route, so the car ran the whole weave on held inputs.

The supported call is beamngpy's ``vehicle.ai.set_line(line, cling=True)``
(beamngpy 1.36). It is not a Lua string: it sends a ``SetAiLine`` request,
and Tech's vehicle-side ``tech/techCore.lua`` ``handleSetAiLine`` turns it
into ``ai.driveUsingPath({script = nodes, wpSpeeds = speeds})`` with
``nodes[i] = {x, y, z = ground height when cling, v = speed}``. In
``ai.lua`` that stores the path as ``scriptData`` and sets
``M.mode = 'manual'``; the next ``ai`` frame that has a map turns it into
the route (``setScriptedPath``: a custom map with nodes ``wp_1 .. wp_n``).

Anything that runs ``ai.setMode``/``ai.setState``/``ai.spanMap``/
``ai.reset``/``ai.driveUsingPath`` after that drops the route, so nothing is
sent after ``set_line``. ``setSpeed``/``setSpeedMode`` would be reset by
``setScriptedPath`` anyway (the per-node speed drives the line), so they are
not sent with a line either.

The ack is a readback, not the ``AiLineSet`` reply (that only says the
request was handled): ``line_state_lua`` returns the AI mode and the
position of the line's last node in the AI's map. The line is active when
the mode is ``manual`` and that node sits on the last point we sent.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# 0.5 s: the route is built on the first ``ai`` graphics frame after the
# request has a map (one frame, ~17 ms at 60 fps). 0.5 s is 10 frames at a
# degraded 20 fps, and at 30 m/s it is 15 m of a line that is at least
# MIN_PATH_M (25 m) long, so a line that did not take is aborted well before
# the car has run a meaningful part of it on held inputs.
LINE_ACK_TIMEOUT_S = 0.5
LINE_ACK_POLL_S = 0.05
# While a line runs it is read back once a second; a route dropped later (a
# map reload replaces the AI's custom map and clears its route) ends the
# excursion as ``line_lost`` instead of drifting.
LINE_RECHECK_S = 1.0
# The last node is placed at our last point (x, y); z is the ground height.
LINE_MATCH_M = 0.5
LINE_NODE_PREFIX = "wp_"

LINE_NOT_ACTIVE = "line_not_active"
LINE_LOST = "line_lost"


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
    """Vehicle-VM chunk that returns ``mode|x,y|driving`` (``none`` when no line node)."""
    node = f"{LINE_NODE_PREFIX}{int(n_points)}"
    return (
        "local s = ai.getState() "
        "local m = mapmgr.mapData "
        f"local p = m and m.positions and m.positions['{node}'] "
        "return tostring(s and s.mode) .. '|' .. "
        "(p and string.format('%.2f,%.2f', p.x, p.y) or 'none') .. '|' .. "
        "tostring(ai.isDriving())"
    )


def parse_line_state(reply: Any) -> tuple[str, tuple[float, float] | None, bool]:
    text = "" if reply is None else str(reply)
    parts = text.split("|")
    mode = parts[0] if parts else ""
    xy: tuple[float, float] | None = None
    if len(parts) > 1 and parts[1] not in ("", "none"):
        try:
            a, b = parts[1].split(",")
            xy = (float(a), float(b))
        except ValueError:
            xy = None
    driving = len(parts) > 2 and parts[2] == "true"
    return mode, xy, driving


@dataclass
class LineAck:
    ok: bool
    reason: str = ""
    detail: str = ""
    mode: str = ""
    wait_s: float = 0.0
    polls: int = 0
    replies: list[str] = field(default_factory=list)

    def as_tag(self) -> dict[str, Any]:
        tag: dict[str, Any] = {"ok": bool(self.ok), "mode": self.mode, "wait_s": round(float(self.wait_s), 3), "polls": int(self.polls)}
        if not self.ok:
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


def read_line_state(vehicle: Any, n_points: int) -> Any:
    """One readback. Raises when this vehicle cannot return a Lua value."""
    fn = getattr(vehicle, "queue_lua_command", None)
    if not callable(fn):
        raise RuntimeError("vehicle has no queue_lua_command")
    return fn(line_state_lua(n_points), True)


def line_is_active(reply: Any, points: list[tuple[float, float]]) -> tuple[bool, str]:
    mode, xy, _driving = parse_line_state(reply)
    if mode != "manual":
        return False, mode
    if xy is None or not points:
        return False, mode
    lx, ly = points[-1]
    return math.hypot(xy[0] - float(lx), xy[1] - float(ly)) <= LINE_MATCH_M, mode


def wait_line_active(
    vehicle: Any,
    points: list[tuple[float, float]],
    *,
    timeout_s: float = LINE_ACK_TIMEOUT_S,
    poll_s: float = LINE_ACK_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LineAck:
    """Poll the readback until the line is the AI's route, or ``timeout_s`` passes."""
    t0 = clock()
    ack = LineAck(ok=False, reason=LINE_NOT_ACTIVE)
    while True:
        try:
            reply = read_line_state(vehicle, len(points))
        except Exception as exc:
            ack.detail = f"readback: {type(exc).__name__}: {exc}"
            ack.wait_s = clock() - t0
            return ack
        ack.polls += 1
        ack.replies.append(str(reply))
        ok, mode = line_is_active(reply, points)
        ack.mode = mode
        ack.wait_s = clock() - t0
        if ok:
            ack.ok = True
            ack.reason = ""
            return ack
        if ack.wait_s >= float(timeout_s) - 1e-9:
            ack.detail = f"timeout {float(timeout_s):.2f}s last={reply!s}"
            return ack
        sleep(float(poll_s))


def send_and_ack(
    vehicle: Any,
    points: list[tuple[float, float]],
    speed_mps: float,
    *,
    queue: Callable[[Any, str], None],
    timeout_s: float = LINE_ACK_TIMEOUT_S,
    poll_s: float = LINE_ACK_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LineAck:
    """Pre-commands, ``set_line``, then the readback ack. Nothing is sent after the line."""
    for command in pre_line_commands():
        try:
            queue(vehicle, command)
        except Exception as exc:
            return LineAck(ok=False, reason=LINE_NOT_ACTIVE, detail=f"pre: {type(exc).__name__}: {exc}")
    err = send_line(vehicle, points, speed_mps)
    if err:
        return LineAck(ok=False, reason=LINE_NOT_ACTIVE, detail=err)
    return wait_line_active(vehicle, points, timeout_s=timeout_s, poll_s=poll_s, clock=clock, sleep=sleep)


def recheck_line(vehicle: Any, points: list[tuple[float, float]]) -> LineAck:
    """One readback while the line runs. ``ok`` False means the route is gone."""
    try:
        reply = read_line_state(vehicle, len(points))
    except Exception as exc:
        return LineAck(ok=False, reason=LINE_LOST, detail=f"readback: {type(exc).__name__}: {exc}", polls=1)
    ok, mode = line_is_active(reply, points)
    return LineAck(ok=ok, reason="" if ok else LINE_LOST, detail="" if ok else f"last={reply!s}", mode=mode, polls=1, replies=[str(reply)])
