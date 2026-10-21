"""Live hold-last side cameras: the recorder's storage filter on streamed frames.

The trainer's ``--cam-age`` rows hold each side camera's last STORED frame and
carry its age (``python.data.cam_hold``). Live builds the same thing the way
the recorder made it:

* **Reads**: every tick reads all 8 cameras with ``stream_raw`` (the default
  BeamNG grab, ``cameras.camera_grab_due``), as the recorder's
  ``_poll_cameras`` does (``scripts/collect_scene_lanes.py``). No ad-hoc
  request, no extra GE call.
* **Which frames count**: the recorder's own ``CameraSchedule``
  (``python.data.raw_frames``, built from ``rates_table`` like the recorder)
  picks which of this tick's frames would be STORED: ``tick(time.monotonic(),
  cams_with_a_frame)``, the call the recorder makes before writing a row.
  Only a stored side frame enters the hold.
* **Age**: a stored frame carries its wall stamp (``time.time()`` right after
  its read, the recorder's ``cams.<cam>.t``). ``now`` is main's stamp (the
  recorder's ``cams.main.t``, ``cam_hold.row_now``). ``cam_hold.held_sector``
  quantises to 1/15 s and caps at 1.5 s. Both clocks are wall time, never a
  tick count, so the ages do not depend on the loop rate.
* **Fronts** (main, narrow, wide) are this tick's read at age 0, or black / -1
  when the read is missing. main is never held.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from python.data.cam_hold import CAM_AGE_ORDER, NO_FRAME_AGE, CamHold, ages_vector, no_ages
from python.data.raw_frames import CameraSchedule

# Read every tick and shown at age 0 (never held).
FRONT_CAMS = frozenset(("main", "narrow", "wide"))
SIDE_CAMS: tuple[str, ...] = tuple(cam for cam in CAM_AGE_ORDER if cam not in FRONT_CAMS)


def recorder_schedule() -> CameraSchedule:
    """The recorder's schedule: ``rates_table`` with no overrides, as collect_scene_lanes builds it."""
    from python.collect.auto_mode import rates_table

    return CameraSchedule(rates_table("", ""))


class LiveStreamHold:
    """This tick's streamed frames -> the recorder's stored set -> hold-last frames and ages."""

    def __init__(self, schedule: CameraSchedule | None = None) -> None:
        self.schedule = schedule if schedule is not None else recorder_schedule()
        self.hold = CamHold(SIDE_CAMS)
        self.stored_by_cam: dict[str, int] = {}
        self.last_taken: tuple[str, ...] = ()

    def reset(self) -> None:
        """Respawn / reset / level load: new schedule anchor, no held frame (the recorder's ``cam_sched.reset``)."""
        self.schedule.reset()
        self.hold.reset()

    def step(
        self, frames: Mapping[str, Any], walls: Mapping[str, float], mono_now: float
    ) -> tuple[dict[str, Any], np.ndarray]:
        """One tick. ``frames``: this tick's reads by camera; ``walls``: wall time after each read.

        Returns the frames for ``compose_strip`` and the 8 ages in ``CAM_AGE_ORDER``.
        """
        present = [cid for cid in frames if cid in CAM_AGE_ORDER and frames[cid] is not None]
        taken = self.schedule.tick(float(mono_now), present)
        self.last_taken = tuple(taken)
        for cid in taken:
            if cid in FRONT_CAMS:
                continue
            self.hold.note(cid, walls.get(cid), frames[cid])
            self.stored_by_cam[cid] = self.stored_by_cam.get(cid, 0) + 1
        now_t = walls.get("main") if frames.get("main") is not None else None
        if now_t is None:
            # No main read: no ``now``; a recorded row without main is never trained.
            return {}, no_ages()
        picks: dict[str, tuple[Any, float]] = dict(self.hold.pick(float(now_t)))
        for cam in FRONT_CAMS:
            frame = frames.get(cam)
            picks[cam] = (frame, 0.0) if frame is not None else (None, NO_FRAME_AGE)
        held = {cam: frame for cam, (frame, _age) in picks.items() if frame is not None}
        return held, ages_vector(picks)


# ---- stale ego ----------------------------------------------------------------

# poll_vehicle markers for a result that is not a new sensors.poll.
STALE_EGO_NOTES = ("sensors.poll coalesced", "sensors.poll skipped")
# Consecutive stale ticks allowed while engaged; the next one takes the heartbeat_stale stop.
STALE_EGO_MAX_TICKS = 1


def vehicle_data_stale(vdata: Any) -> bool:
    """True when ``poll_vehicle`` handed back last-good (coalesced) or empty data instead of a poll."""
    if vdata is None:
        return False
    note = str(getattr(vdata, "note", "") or "")
    return any(tag in note for tag in STALE_EGO_NOTES)


class StaleEgo:
    """Consecutive and total stale ``poll_vehicle`` ticks."""

    def __init__(self) -> None:
        self.run = 0
        self.total = 0

    def observe(self, vdata: Any) -> bool:
        stale = vehicle_data_stale(vdata)
        if stale:
            self.run += 1
            self.total += 1
        else:
            self.run = 0
        return stale

    def blocks_drive(self, engaged: bool) -> bool:
        """Engaged and stale for more than ``STALE_EGO_MAX_TICKS`` ticks in a row."""
        return bool(engaged) and self.run > STALE_EGO_MAX_TICKS


# ---- respawn / reset ------------------------------------------------------------

# A move this far in one tick, beyond what the car could drive, is a respawn.
RESPAWN_JUMP_M = 10.0
RESPAWN_SPEED_MARGIN = 3.0
# In-place reset: beamngpy 1.36 has no reset event or counter (vehicle state:
# pos/dir/up/vel/rotation; electrics has no reset field), so a reset is a
# change the car cannot make: |d speed| > RESET_MAX_G * g * dt + margin, or a
# move further than max(speed) * dt + margin.
GRAVITY = 9.81
RESET_MAX_G = 1.5
RESET_SPEED_MARGIN_MPS = 1.0
RESET_POS_MARGIN_M = 2.0


def _body_speed(vdata: Any) -> float | None:
    """|vel| from the state sensor (body velocity); wheelspeed (``speed_mps``) only when there is none.

    Wheelspeed can drop to 0 in one tick when a wheel locks; the body cannot.
    """
    vel = getattr(vdata, "vel", None)
    if vel is not None:
        try:
            return float(math.sqrt(sum(float(v) * float(v) for v in vel[:3])))
        except (TypeError, ValueError, IndexError):
            pass
    spd = getattr(vdata, "speed_mps", None)
    try:
        return None if spd is None else abs(float(spd))
    except (TypeError, ValueError):
        return None


class RespawnWatch:
    """Live respawn / reset edge for the hold: no frame from before it may survive.

    The supervisor has no respawn event of its own (the recorder marks
    ``recover`` because it issues the respawn itself), and beamngpy exposes no
    reset event. Live sees one of:

    * the vehicle id changes (a new or replaced vehicle);
    * the vehicle reconnects after a disconnect (a level load or re-attach);
    * the position jumps further in one sample than the car could drive:
      more than ``max(RESPAWN_JUMP_M, RESPAWN_SPEED_MARGIN * speed * dt + RESPAWN_JUMP_M / 2)``;
    * an in-place reset (heuristic): body speed changes by more than
      ``RESET_MAX_G * g * dt + RESET_SPEED_MARGIN_MPS``, or the car moved more
      than ``max(speed before, after) * dt + RESET_POS_MARGIN_M``.

    ``dt`` is the time between the two fresh vehicle samples (``mono``); a
    stale (coalesced) sample is skipped, not compared. ``observe`` returns
    the reason on the tick of the edge, else None.
    """

    def __init__(self) -> None:
        self.vid: Any = None
        self.connected: bool | None = None
        self.pos: tuple[float, float, float] | None = None
        self.speed: float | None = None
        self.mono: float | None = None
        self.count = 0

    def observe(self, vdata: Any, dt_s: float | None, mono: float | None = None) -> str | None:
        if vdata is None or vehicle_data_stale(vdata):
            return None
        reason = None
        connected = bool(getattr(vdata, "connected", False))
        vid = getattr(vdata, "vid", None)
        pos = getattr(vdata, "pos", None)
        speed = _body_speed(vdata)
        if mono is not None and self.mono is not None:
            dt = max(0.0, float(mono) - self.mono)
        else:
            dt = 0.0 if dt_s is None else max(0.0, float(dt_s))
        if self.connected is False and connected:
            reason = "reconnect"
        elif vid is not None and self.vid is not None and vid != self.vid:
            reason = "vehicle"
        elif pos is not None and self.pos is not None:
            try:
                jump = float(np.linalg.norm(np.asarray(pos[:3], dtype=np.float64) - np.asarray(self.pos, dtype=np.float64)))
                v_ref = abs(float(getattr(vdata, "speed_mps", 0.0) or 0.0))
                if jump > max(RESPAWN_JUMP_M, RESPAWN_SPEED_MARGIN * v_ref * dt + RESPAWN_JUMP_M / 2.0):
                    reason = f"jump {jump:.1f} m"
                elif speed is not None and self.speed is not None and dt > 0.0:
                    dv = abs(speed - self.speed)
                    if dv > RESET_MAX_G * GRAVITY * dt + RESET_SPEED_MARGIN_MPS:
                        reason = f"reset dv {self.speed:.2f}->{speed:.2f} m/s in {dt * 1000.0:.0f} ms moved {jump:.2f} m"
                    elif jump > max(speed, self.speed) * dt + RESET_POS_MARGIN_M:
                        reason = f"reset moved {jump:.2f} m at {max(speed, self.speed):.2f} m/s in {dt * 1000.0:.0f} ms"
            except (TypeError, ValueError):
                pass
        self.connected = connected
        if vid is not None:
            self.vid = vid
        if pos is not None:
            try:
                self.pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            except (TypeError, ValueError, IndexError):
                pass
        if speed is not None:
            self.speed = speed
        if mono is not None:
            self.mono = float(mono)
        if reason is not None:
            self.count += 1
        return reason


# ---- main exposure -------------------------------------------------------------


def main_exposure(main: Any, prev_main: Any) -> tuple[float | None, bool]:
    """Main's mean luminance (0..1, BGR luma weights) and whether it is a flash frame.

    The flash test is ``cameras.companion_frame_is_flash`` (brighter and bluer
    than the previous frame together), applied to main for the capture only.
    """
    if main is None:
        return None, False
    arr = np.asarray(main)
    if arr.ndim != 3 or arr.shape[-1] < 3 or arr.size == 0:
        return None, False
    luma = float((0.114 * arr[..., 0].mean() + 0.587 * arr[..., 1].mean() + 0.299 * arr[..., 2].mean()) / 255.0)
    flash = False
    if prev_main is not None:
        from python.sensors.cameras import companion_frame_is_flash

        try:
            flash = bool(companion_frame_is_flash(arr, np.asarray(prev_main)))
        except Exception:
            flash = False
    return luma, flash


def scrub_after_respawn(bundle: Any) -> None:
    """This tick's grab ran before the respawn edge was seen: keep main only, every other camera -1."""
    if getattr(bundle, "cam_ages", None) is None:
        return
    main = bundle.main_bgr() if hasattr(bundle, "main_bgr") else None
    bundle.held_frames = {"main": main} if main is not None else {}
    bundle.cam_ages = tuple(float(a) for a in main_only_ages(main is not None))


def main_only_ages(main_present: bool) -> np.ndarray:
    """The window path and an unscheduled backend: main at age 0, every other camera -1."""
    out = no_ages()
    if main_present:
        out[CAM_AGE_ORDER.index("main")] = 0.0
    return out


def bundle_ages(bundle: Any) -> tuple[dict[str, Any], np.ndarray]:
    """Hold-last frames and ages for one grab, whatever the backend.

    A backend with the live hold on fills ``held_frames`` / ``cam_ages``.
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
    "FRONT_CAMS",
    "LiveStreamHold",
    "NO_FRAME_AGE",
    "RespawnWatch",
    "SIDE_CAMS",
    "STALE_EGO_MAX_TICKS",
    "StaleEgo",
    "ages_text",
    "bundle_ages",
    "main_exposure",
    "main_only_ages",
    "parse_ages_text",
    "recorder_schedule",
    "scrub_after_respawn",
    "vehicle_data_stale",
]
