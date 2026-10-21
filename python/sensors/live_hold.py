"""Live hold-last side cameras: scheduled ad-hoc polls and per-camera ages.

The trainer's ``--cam-age`` rows hold each side camera's last recorded frame
and carry its age (``python.data.cam_hold``). Live feeds the same thing:

* **Which camera renders when** is the recorder's own ``CameraSchedule``
  (``python.data.raw_frames``), built from the same ``rates_table`` the
  recorder uses (``scripts/collect_scene_lanes.py``), i.e. ``CAMERA_HZ`` with
  ``CAMERA_PHASE_S``. It is imported, not copied.
* **How**: each due companion gets one ad-hoc poll request
  (``send_ad_hoc_poll_request``). Later ticks ask whether that request id is
  ready and collect it. Every GE call goes through the backend's bounded IO
  call, so the tick never waits on a poll.
* **Age**: a frame is keyed by its request id and stamped with wall time
  (``time.time()``) right after the collect returns, the point where the
  recorder stamps ``cams.<cam>.t`` (wall time after the read). ``now`` is
  main's wall stamp after its read (the recorder's ``cams.main.t``). The
  shared ``cam_hold.held_sector`` then applies the 1/15 s quantisation and
  the 1.5 s cap. A result for any other request id is dropped.
* **Timeout**: a request not collected within ``ADHOC_POLL_TIMEOUT_S`` is
  dropped and that camera is black with age -1 until its next frame.
* main is never held and never ad-hoc: it is read every tick.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from python.data.cam_hold import CAM_AGE_ORDER, NEVER_HELD, NO_FRAME_AGE, CamHold, ages_vector, no_ages
from python.data.raw_frames import CameraSchedule

# One ad-hoc request may take this long from send to collect; after that the
# camera is black / -1 (never an older frame passed off as this request).
ADHOC_POLL_TIMEOUT_S = 0.25
# Each GE call (send / ready / collect) waits at most this long on the tick;
# a slower call keeps running on the IO thread and its result lands later.
ADHOC_CALL_BUDGET_S = 0.008
# Total companion IO a tick may spend before it stops issuing calls.
ADHOC_TICK_BUDGET_S = 0.025


def recorder_schedule() -> CameraSchedule:
    """The recorder's schedule: ``rates_table`` with no overrides, as collect_scene_lanes builds it."""
    from python.collect.auto_mode import rates_table

    return CameraSchedule(rates_table("", ""))


class ScheduledAdHoc:
    """Per-camera ad-hoc requests on the recorder schedule, and the hold-last ages.

    ``io(cid, kind, fn, budget)`` runs ``fn`` (a GE call) with a bounded wait and
    returns its value, or a busy/timeout marker (``is_mark``) when it did not
    finish in ``budget`` or another call holds the socket. A call that finishes
    late is handed back through ``late(cid, kind, value)``.
    """

    def __init__(
        self,
        cams: Iterable[str],
        *,
        schedule: CameraSchedule | None = None,
        timeout_s: float = ADHOC_POLL_TIMEOUT_S,
        call_budget_s: float = ADHOC_CALL_BUDGET_S,
        tick_budget_s: float = ADHOC_TICK_BUDGET_S,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.cams = tuple(str(c) for c in cams if str(c) not in NEVER_HELD)
        self.schedule = schedule if schedule is not None else recorder_schedule()
        self.timeout_s = float(timeout_s)
        self.call_budget_s = float(call_budget_s)
        self.tick_budget_s = float(tick_budget_s)
        self.clock = clock
        self.wall = wall
        self.hold = CamHold(CAM_AGE_ORDER)
        # cid -> {"rid": int, "sent": monotonic, "ready": bool}
        self.pending: dict[str, dict[str, Any]] = {}
        self.timeouts = 0
        self.timeouts_by_cam: dict[str, int] = {}
        self.sent = 0
        self.collected = 0
        self.dropped_ids = 0

    # ---- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Respawn / level load: new schedule anchor, no held frame, no request owed."""
        self.schedule.reset()
        self.hold.reset()
        self.pending.clear()

    def inflight(self) -> int:
        return len(self.pending)

    # ---- results ---------------------------------------------------------

    def deliver(self, cid: str, rid: int, frame: Any, wall_t: float) -> bool:
        """A collected frame for request ``rid``. Only the camera's pending id counts."""
        got = self.pending.get(str(cid))
        if got is None or int(got["rid"]) != int(rid):
            self.dropped_ids += 1
            return False
        self.pending.pop(str(cid), None)
        if frame is None:
            return False
        self.hold.note(str(cid), float(wall_t), frame)
        self.collected += 1
        return True

    def on_sent(self, cid: str, rid: Any, sent_mono: float) -> None:
        if isinstance(rid, bool) or not isinstance(rid, (int, np.integer)):
            return
        if str(cid) in self.pending:
            return
        self.pending[str(cid)] = {"rid": int(rid), "sent": float(sent_mono), "ready": False}
        self.sent += 1

    def on_ready(self, cid: str, rid: Any, flag: Any) -> None:
        got = self.pending.get(str(cid))
        if got is not None and int(got["rid"]) == int(rid) and flag is True:
            got["ready"] = True

    # ---- one tick ----------------------------------------------------------

    def _expire(self, now: float) -> list[str]:
        gone = []
        for cid, got in list(self.pending.items()):
            if now - float(got["sent"]) > self.timeout_s:
                self.pending.pop(cid, None)
                self.hold.forget(cid)
                self.timeouts += 1
                self.timeouts_by_cam[cid] = self.timeouts_by_cam.get(cid, 0) + 1
                gone.append(cid)
        return gone

    def step(
        self,
        sensors: Mapping[str, Any],
        io: Callable[[str, str, Callable[[], Any], float], Any],
        is_mark: Callable[[Any], bool],
        prepare: Callable[[str, Any], Any],
    ) -> dict[str, Any]:
        """Expire, collect what is ready, send what the schedule says is due.

        Returns ``{cid: frame}`` collected on this tick (already in the hold).
        ``prepare(cid, images)`` turns a collect reply into the stored frame
        (or None: unrendered or flash, nothing stored).
        """
        start = self.clock()
        self._expire(start)
        fresh: dict[str, Any] = {}

        def spent() -> bool:
            return self.clock() - start >= self.tick_budget_s

        for cid, got in list(self.pending.items()):
            if spent():
                break
            cam = sensors.get(cid)
            if cam is None:
                self.pending.pop(cid, None)
                continue
            rid = int(got["rid"])
            if not got["ready"]:
                flag = io(cid, "sched_ready", lambda cam=cam, rid=rid: (rid, bool(cam.is_ad_hoc_poll_request_ready(rid))), self.call_budget_s)
                if is_mark(flag) or not isinstance(flag, tuple):
                    continue
                self.on_ready(cid, flag[0], flag[1])
                if not self.pending.get(cid, {}).get("ready"):
                    continue
            if spent():
                break

            def _collect(cam: Any = cam, rid: int = rid) -> tuple[int, Any, float]:
                images = cam.collect_ad_hoc_poll_request(rid)
                # The recorder's stamp: wall time right after the read returns.
                return rid, images, self.wall()

            reply = io(cid, "sched_collect", _collect, self.call_budget_s)
            if is_mark(reply) or not isinstance(reply, tuple):
                continue
            frame = self.land(cid, reply, prepare)
            if frame is not None:
                fresh[cid] = frame
        due = self.schedule.tick(start, [c for c in self.cams if c in sensors])
        for cid in due:
            if cid in self.pending:
                continue  # one request per camera; this phase is served by the one in flight
            if spent():
                break
            cam = sensors[cid]

            def _send(cam: Any = cam) -> tuple[int, float]:
                return int(cam.send_ad_hoc_poll_request()), self.clock()

            reply = io(cid, "sched_send", _send, self.call_budget_s)
            if is_mark(reply) or not isinstance(reply, tuple):
                continue
            self.on_sent(cid, reply[0], reply[1])
        return fresh

    def land(self, cid: str, reply: tuple[int, Any, float], prepare: Callable[[str, Any], Any]) -> Any:
        rid, images, wall_t = reply
        frame = prepare(cid, images)
        if self.deliver(cid, int(rid), frame, float(wall_t)) and frame is not None:
            return frame
        return None

    def late(self, cid: str, kind: str, value: Any, prepare: Callable[[str, Any], Any]) -> None:
        """A GE call that ran past its budget finished; apply its result now."""
        if not isinstance(value, tuple):
            return
        if kind == "sched_send":
            self.on_sent(cid, value[0], value[1])
        elif kind == "sched_ready":
            self.on_ready(cid, value[0], value[1])
        elif kind == "sched_collect":
            self.land(cid, value, prepare)

    # ---- what the net sees -------------------------------------------------

    def compose(self, now_t: float | None, main: Any) -> tuple[dict[str, Any], np.ndarray]:
        """Frames for ``compose_strip`` and the 8 ages, through ``held_sector``.

        main is this tick's frame at age 0 (absent: black, -1). ``now_t`` is
        main's wall stamp; without main there is no ``now`` and every camera is
        black / -1, as a recorded row without main is never trained.
        """
        if now_t is None or main is None:
            return {}, no_ages()
        picks = self.hold.pick(float(now_t))
        picks["main"] = (main, 0.0)
        frames = {cam: frame for cam, (frame, _age) in picks.items() if frame is not None}
        return frames, ages_vector(picks)


def main_only_ages(main_present: bool) -> np.ndarray:
    """The window path and an unscheduled backend: main at age 0, every other camera -1."""
    out = no_ages()
    if main_present:
        out[CAM_AGE_ORDER.index("main")] = 0.0
    return out


def bundle_ages(bundle: Any) -> tuple[dict[str, Any], np.ndarray]:
    """Hold-last frames and ages for one grab, whatever the backend.

    A backend with the scheduled hold fills ``held_frames`` / ``cam_ages``.
    Anything else (window capture, the default BeamNG grab) is main only.
    """
    ages = getattr(bundle, "cam_ages", None)
    if ages is not None:
        held = dict(getattr(bundle, "held_frames", None) or {})
        return held, np.asarray(ages, dtype=np.float32).reshape(-1)
    main = bundle.main_bgr() if hasattr(bundle, "main_bgr") else None
    return ({"main": main} if main is not None else {}), main_only_ages(main is not None)


def ages_text(ages: Any) -> str:
    """``repeatL:-1,pillarL:0.067,...`` in ``CAM_AGE_ORDER`` (the telemetry form)."""
    arr = np.asarray(ages, dtype=np.float32).reshape(-1)
    return ",".join(f"{cam}:{float(arr[k]):.3f}" if float(arr[k]) >= 0 else f"{cam}:-1" for k, cam in enumerate(CAM_AGE_ORDER))


def parse_ages_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        if ":" not in part:
            continue
        cam, raw = part.split(":", 1)
        try:
            out[cam.strip()] = float(raw)
        except ValueError:
            continue
    return out


__all__ = [
    "ADHOC_CALL_BUDGET_S",
    "ADHOC_POLL_TIMEOUT_S",
    "ADHOC_TICK_BUDGET_S",
    "NO_FRAME_AGE",
    "ScheduledAdHoc",
    "ages_text",
    "bundle_ages",
    "main_only_ages",
    "parse_ages_text",
    "recorder_schedule",
]
