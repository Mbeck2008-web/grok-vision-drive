"""Camera backends for GVD — vision-only RGB. Never fake 8 cams from one window."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

CAM_IDS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
MAIN_ALIASES = ("main", "cam_main")
SIDE_CAM_IDS = frozenset(("pillarL", "pillarR", "repeatL", "repeatR"))
PILLAR_CAM_IDS = frozenset(("pillarL", "pillarR"))
REPEAT_CAM_IDS = frozenset(("repeatL", "repeatR"))
REAR_CAM_IDS = frozenset(("rear",))
FORWARD_CAM_IDS = frozenset(("narrow", "main", "wide"))

# BeamNGpy Camera() defaults near_far_planes=(0.05, 100) — BeamNGpy#199.
# GVD always passes an explicit pair from cameras.yaml so Tech attach is not a silent omit
# of near_far_planes (sides/rear lock at 100 m still go through Camera(..., near_far_planes=)).
DEFAULT_NEAR_M = 0.05
BNGPY_DEFAULT_FAR_M = 100.0
FORWARD_UPDATE_S = 0.067  # ~15 Hz suggestion to the Tech sensor manager
ON_DEMAND_UPDATE_S = -1.0  # yaml: sample on the hitch, not every tick
# Offscreen sensor period when yaml says on-demand. Not an ad-hoc viewport render.
# SendAdHocRequestCamera steps the game view's exposure for one frame (blue shadows).
COMPANION_OFFSCREEN_UPDATE_S = 1.0
SIDE_UPDATE_S = ON_DEMAND_UPDATE_S
REAR_UPDATE_S = ON_DEMAND_UPDATE_S
MAIN_GRAB_DIV = 1
# One supervisor tick reads all eight cameras before the stitch is built.
# A camera that was not read on this tick is missing. An older buffer is not
# painted into the bundle. Soft Esc and Engage use that same grab. A zero
# colour buffer is unrendered, not a picture, and it does not keep a previous
# frame. Yaml -1 is still the offscreen attach period, not a viewport render.
# The sensor is attached at COMPANION_OFFSCREEN_UPDATE_S and read with
# stream_raw. SendAdHocRequestCamera is not used: that render steps the game
# view's exposure for one frame. A read that exceeds CAM_READ_BUDGET_S does
# not stall the rest of the grab and does not fill the slot from an older tick.
CAM_READ_BUDGET_S = 0.05
_OWED_CAP = 7  # one slot per companion; the owed queue cannot grow past the rig
WIDE_GRAB_DIV = 16
NARROW_GRAB_DIV = 16
NARROW_GRAB_PHASE = 1  # odd ticks; wide stays on even ticks
WIDE_GRAB_PHASE = 0
SIDE_GRAB_DIV = 16  # each pillar once per 16 ticks
SIDE_GRAB_PHASE = 2  # pillarL; pillarR steps +1 onto phase 3
REPEAT_GRAB_DIV = 16
REPEAT_GRAB_PHASE = 5  # repeatL; repeatR steps +2 onto phase 7
REAR_GRAB_DIV = 16  # poll rear once per 16 ticks; never drop resolution
REAR_GRAB_PHASE = 6  # main+rear only; no side on this slot
REPEAT_SPREAD_ORDER = ("repeatL", "repeatR")
PILLAR_SPREAD_ORDER = ("pillarL", "pillarR")
PILLAR_SPREAD_STEP = 1
NARROW_FAR_LIVE_HITCH_M = 400.0  # live unique-frame Hz hitch (800→400 if still <10)
CAMERA_HZ_TARGET = 10.0
LIVE_NARROW_HITCH_AFTER_S = 2.0
LIVE_NARROW_HITCH_MIN_UNIQUE = 4
DEFAULT_FAR_M = {
    "narrow": 800.0,
    "main": 300.0,
    "wide": 300.0,
    "pillarL": 100.0,
    "pillarR": 100.0,
    "repeatL": 100.0,
    "repeatR": 100.0,
    "rear": 100.0,
}
DEFAULT_UPDATE_S = {
    "narrow": FORWARD_UPDATE_S,
    "main": FORWARD_UPDATE_S,
    "wide": FORWARD_UPDATE_S,
    "pillarL": ON_DEMAND_UPDATE_S,
    "pillarR": ON_DEMAND_UPDATE_S,
    "repeatL": ON_DEMAND_UPDATE_S,
    "repeatR": ON_DEMAND_UPDATE_S,
    "rear": ON_DEMAND_UPDATE_S,
}
# BeamNGpy update_priority: 0 highest → 1 lowest (getter contract). Starve non-main.
DEFAULT_UPDATE_PRIORITY = {
    "main": 0.0,
    "narrow": 0.5,
    "wide": 0.5,
    "pillarL": 1.0,
    "pillarR": 1.0,
    "repeatL": 1.0,
    "repeatR": 1.0,
    "rear": 1.0,
}
# Role bands pin the lock (narrow 800 > main 300; never both-800).
# Narrow lo 400 is the live hitch floor (still ≥ main 300); attach stays 800.
# Forward 100 m clamps up to the role far. Sides/rear lock at 100 m (explicit planes).
NARROW_FAR_BAND = (400.0, 800.0)
MAIN_FAR_BAND = (300.0, 300.0)
WIDE_FAR_BAND = (300.0, 300.0)
SIDE_FAR_BAND = (100.0, 100.0)
REAR_FAR_BAND = (100.0, 100.0)
NARROW_FAR_HITCH = (800.0,)  # attach starts 800; 400 is live unique-Hz only
MAIN_FAR_HITCH = (300.0,)
WIDE_FAR_HITCH = (300.0,)
SIDE_FAR_HITCH = (100.0,)
REAR_FAR_HITCH = (100.0,)
SIDE_CAM_HITCH_UPDATE_S = ON_DEMAND_UPDATE_S
REAR_CAM_HITCH_UPDATE_S = ON_DEMAND_UPDATE_S


class CamHealth(str, Enum):
    OK = "ok"
    STALE = "stale"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class CameraFrameBundle:
    frames: dict[str, np.ndarray] = field(default_factory=dict)  # BGR uint8
    timestamps: dict[str, float] = field(default_factory=dict)
    health: dict[str, CamHealth] = field(default_factory=dict)
    backend: str = "stub"
    note: str = ""
    grab_ms: float = 0.0
    # Hitch-wheel slot (grab_i % wheel). -1 when this backend has no wheel.
    # Locked wheel: hitch slots 0,1,2,3,5,6,7; poll-free slots 4 and 8–15.
    grab_phase: int = -1
    # True when this grab did not issue a companion PollCamera (main stream_raw only).
    grab_poll_free: bool = True
    unique_gpu_n: int = 0  # new GPU frames this tick (not cache / stream_raw re-shows)
    unique_gpu_ids: tuple[str, ...] = ()
    # Ad-hoc companion requests not yet collected. Capped at 1.
    companion_inflight: int = 0
    # True when a stream_raw / poll / ad-hoc call exceeded CAM_READ_BUDGET_S
    # or the previous read was still running, so this tick did not wait on it.
    grab_read_blocked: bool = False

    def tick_frames(self) -> dict[str, np.ndarray]:
        """Pictures whose timestamps are this grab. An older stamp is left out."""
        return bundle_tick_frames(self)

    def health_str(self) -> dict[str, str]:
        out = {cid: CamHealth.MISSING.value for cid in CAM_IDS}
        for k, v in self.health.items():
            key = "main" if k in MAIN_ALIASES else k
            out[key] = v.value if isinstance(v, CamHealth) else str(v)
        return out

    def main_bgr(self) -> np.ndarray | None:
        for k in MAIN_ALIASES:
            if k in self.frames:
                return self.frames[k]
        return None


def bundle_tick_frames(bundle: CameraFrameBundle | None) -> dict[str, np.ndarray]:
    """Frames from one grab. A camera stamped on another tick is left out.

    ``cam_main`` is the main alias when main is on this tick. It is not a
    ninth camera. A bundle with no timestamps contributes nothing, so a
    cached picture cannot sneak into the stitch.
    """
    if bundle is None:
        return {}
    frames = getattr(bundle, "frames", None) or {}
    stamps = getattr(bundle, "timestamps", None) or {}
    present = [cid for cid in CAM_IDS if cid in frames and cid in stamps]
    if not present:
        return {}
    tick = float(stamps[present[0]])
    out: dict[str, np.ndarray] = {}
    for cid in present:
        if abs(float(stamps[cid]) - tick) > 1e-4:
            continue
        out[cid] = frames[cid]
    if "main" in out:
        out["cam_main"] = out["main"]
    return out


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    import cv2

    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def load_camera_config(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = path or (root / "config" / "cameras.yaml")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"cameras": [{"id": c} for c in CAM_IDS]}


def load_live_long_side(default: int = 640) -> int:
    root = Path(__file__).resolve().parents[2]
    hw = root / "config" / "hardware.yaml"
    try:
        for line in hw.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("live_input_long_side:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return default


def yaw_pitch_to_dir_up(yaw_deg: float, pitch_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """GVD vehicle frame +X right +Y forward +Z up → BeamNGpy dir/up (toy; calibrate on ETK800)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    # look direction
    dx = math.sin(yaw) * math.cos(pitch)
    dy = math.cos(yaw) * math.cos(pitch)
    dz = math.sin(pitch)
    # approximate up (roll=0)
    ux = -math.sin(yaw) * math.sin(pitch)
    uy = -math.cos(yaw) * math.sin(pitch)
    uz = math.cos(pitch)
    return (dx, dy, dz), (ux, uy, uz)


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _pair2(v: Any) -> tuple[float, float] | None:
    """Parse near_far_planes as [near, far] or {near/near_m, far/far_m}."""
    if v is None:
        return None
    if isinstance(v, dict):
        near = _as_float(v.get("near") if v.get("near") is not None else v.get("near_m"))
        far = _as_float(v.get("far") if v.get("far") is not None else v.get("far_m"))
        if near is None or far is None:
            return None
        return (near, far)
    try:
        seq = list(v)
    except TypeError:
        return None
    if len(seq) < 2:
        return None
    near, far = _as_float(seq[0]), _as_float(seq[1])
    if near is None or far is None:
        return None
    return (near, far)


def default_far_m(cid: str) -> float:
    return float(DEFAULT_FAR_M.get(cid, DEFAULT_FAR_M["rear"]))


def far_band(cid: str) -> tuple[float, float]:
    if cid == "narrow":
        return NARROW_FAR_BAND
    if cid == "main":
        return MAIN_FAR_BAND
    if cid == "wide":
        return WIDE_FAR_BAND
    if cid in REAR_CAM_IDS:
        return REAR_FAR_BAND
    return SIDE_FAR_BAND


def default_update_s(cid: str) -> float:
    return float(DEFAULT_UPDATE_S.get(cid, FORWARD_UPDATE_S))


def camera_update_s(
    spec: dict[str, Any] | None,
    *,
    cid: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> float:
    """Per-cam BeamNGpy requested_update_time. Spec yaml wins; else role default.

    Negative means on-demand (no auto GPU update). File-level tech.yaml
    cameras.update_s is not applied: it would clobber sides/rear -1 with 0.067.
    """
    spec = spec or {}
    defaults = defaults or {}
    cam_id = str(cid or spec.get("id") or "")
    for src in (spec, defaults):
        v = _as_float(src.get("requested_update_time"))
        if v is None:
            v = _as_float(src.get("update_s"))
        if v is None:
            continue
        if v < 0:
            return ON_DEMAND_UPDATE_S
        if v > 0:
            return float(v)
    return default_update_s(cam_id)


def camera_update_priority(
    spec: dict[str, Any] | None,
    *,
    cid: str | None = None,
) -> float:
    """BeamNGpy update_priority in [0, 1], 0 = highest. Starve non-main."""
    spec = spec or {}
    cam_id = str(cid or spec.get("id") or "")
    v = _as_float(spec.get("update_priority"))
    if v is None:
        v = DEFAULT_UPDATE_PRIORITY.get(cam_id, 0.5)
    return min(1.0, max(0.0, float(v)))


def read_camera_update_priority(cam: Any) -> float | None:
    """Read BeamNGpy Camera GPU priority. Getter contract: 0 = highest, 1 = lowest."""
    for name in ("get_update_priority", "getUpdatePriority"):
        fn = getattr(cam, name, None)
        if not callable(fn):
            continue
        try:
            v = _as_float(fn())
        except Exception:
            v = None
        if v is not None:
            return min(1.0, max(0.0, float(v)))
    return _as_float(getattr(cam, "update_priority", None))


def priority_highest_is_zero(cam: Any, requested: float) -> bool:
    """True when getter agrees 0 is highest (main requested ~0 must not read ~1)."""
    got = read_camera_update_priority(cam)
    if got is None:
        return True
    req = min(1.0, max(0.0, float(requested)))
    if req <= 0.05 and got >= 0.95:
        return False
    if req >= 0.95 and got <= 0.05:
        return False
    return True


def invert_update_priority(p: float) -> float:
    return min(1.0, max(0.0, 1.0 - float(p)))


def _positive_div(v: Any, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return int(default)
    return n if n >= 1 else int(default)


def _nonneg_int(v: Any, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return int(default)
    return n if n >= 0 else int(default)


def camera_grab_div(cid: str, hitch: dict[str, Any] | None = None) -> int:
    """Configured divisor. The live grab does not skip on it.

    ``camera_grab_due`` reads every rig camera every tick. These numbers stay
    so a hitch file can still name the old slots. They do not drop a camera.
    """
    hitch = hitch if isinstance(hitch, dict) else {}
    if cid == "main":
        return MAIN_GRAB_DIV  # every-tick; hitch yaml cannot raise this
    if cid == "wide":
        return _positive_div(hitch.get("wide_grab_div"), WIDE_GRAB_DIV)
    if cid == "narrow":
        n = _positive_div(hitch.get("narrow_grab_div"), NARROW_GRAB_DIV)
        return max(2, n)  # floor ÷2 so narrow can still miss a wide tick
    if cid in REAR_CAM_IDS:
        return _positive_div(hitch.get("rear_grab_div"), REAR_GRAB_DIV)
    if cid in REPEAT_CAM_IDS:
        return _positive_div(hitch.get("repeat_grab_div"), REPEAT_GRAB_DIV)
    if cid in PILLAR_CAM_IDS or cid in SIDE_CAM_IDS:
        return _positive_div(hitch.get("side_grab_div"), SIDE_GRAB_DIV)
    return 1


def camera_grab_phase(cid: str, hitch: dict[str, Any] | None = None) -> int:
    """Slot on the 16-tick wheel. Wide (even) and narrow (odd) never share a tick.

    pillarL is side phase; pillarR steps +1. repeatL is repeat phase; repeatR
    steps +2. Rear sits on its own slot with no side companion.
    """
    hitch = hitch if isinstance(hitch, dict) else {}
    if cid == "narrow":
        return _nonneg_int(hitch.get("narrow_grab_phase"), NARROW_GRAB_PHASE)
    if cid == "wide":
        return _nonneg_int(hitch.get("wide_grab_phase"), WIDE_GRAB_PHASE)
    if cid == "main":
        return _nonneg_int(hitch.get("main_grab_phase"), 0)
    if cid in REAR_CAM_IDS:
        return _nonneg_int(hitch.get("rear_grab_phase"), REAR_GRAB_PHASE)
    if cid in REPEAT_CAM_IDS:
        base = _nonneg_int(hitch.get("repeat_grab_phase"), REPEAT_GRAB_PHASE)
        div = max(1, camera_grab_div(cid, hitch))
        try:
            slot = REPEAT_SPREAD_ORDER.index(cid)
        except ValueError:
            slot = 0
        return (base + slot * 2) % div
    if cid in PILLAR_CAM_IDS:
        base = _nonneg_int(hitch.get("side_grab_phase"), SIDE_GRAB_PHASE)
        div = max(1, camera_grab_div(cid, hitch))
        try:
            slot = PILLAR_SPREAD_ORDER.index(cid)
        except ValueError:
            slot = 0
        return (base + slot * PILLAR_SPREAD_STEP) % div
    if cid in SIDE_CAM_IDS:
        return _nonneg_int(hitch.get("side_grab_phase"), SIDE_GRAB_PHASE)
    return 0


def grab_due(grab_i: int, div: int, phase: int = 0) -> bool:
    d = max(1, int(div))
    p = int(phase) % d
    return (int(grab_i) % d) == p


def camera_grab_due(cid: str, grab_i: int, hitch: dict[str, Any] | None = None) -> bool:
    """True for every rig camera on this supervisor tick.

    Companions are not updated one per tick. Wide and narrow share the tick.
    ``grab_i`` and the hitch divisors do not skip a camera. A stitch built
    from the bundle is this tick only.
    """
    del grab_i, hitch
    return cid in CAM_IDS


def grab_wheel(hitch: dict[str, Any] | None = None) -> int:
    """Hitch-wheel period. The locked schedule is 16. This does not define camera_hz."""
    return max(camera_grab_div(cid, hitch) for cid in CAM_IDS)


def grab_phase_of(grab_i: int, hitch: dict[str, Any] | None = None) -> int:
    """Wheel slot for this grab index. Every slot reads the whole rig."""
    return int(grab_i) % grab_wheel(hitch)


def grab_is_poll_free(grab_i: int, hitch: dict[str, Any] | None = None) -> bool:
    """True when this tick reads main only.

    The live schedule reads every camera, so a rig tick is not poll-free.
    """
    for cid in CAM_IDS:
        if cid == "main":
            continue
        if camera_grab_due(cid, grab_i, hitch):
            return False
    return True


def _debug_force_engage() -> bool:
    """True when debug engage is already requested on this tick.

    ``--force-engage`` and the nerd-panel bit do not write ``gvd_engage.json``.
    Grab runs before ``note_engaged``, so the latch is still false. Alt+G is
    the live file. Debug engage is argv, or ``args.force_engage`` /
    ``ui.debug.force_engage`` already set on the grab caller. Seeing it
    colours the hitch schedule on this same grab. This does not write the file.
    """
    if any(arg == "--force-engage" for arg in sys.argv):
        return True
    getframe = getattr(sys, "_getframe", None)
    if getframe is None:
        return False
    frame = getframe(1)
    try:
        for _ in range(32):
            if frame is None:
                return False
            try:
                locs = frame.f_locals
                args = locs.get("args") if locs else None
                if getattr(args, "force_engage", None) is True:
                    return True
                ui = locs.get("ui") if locs else None
                debug = getattr(ui, "debug", None) if ui is not None else None
                if getattr(debug, "force_engage", None) is True:
                    return True
            except Exception:
                pass
            frame = frame.f_back
    finally:
        del frame
    return False


def soft_esc_colour_main_only() -> bool:
    """True when this grab is Soft Esc (not engaged).

    Soft Esc is the latch false, ``gvd_engage.json`` not live, and debug
    force-engage off. Grab runs before ``note_engaged``. A live engage file
    or debug force-engage is the rising edge, same as ``sensors.poll``.

    The name is historical. Soft Esc and Engage both read all eight cameras
    on the tick. The flag only chooses the grab note. It does not skip a
    camera and it does not keep an older frame.
    """
    from python.control.actuate import read_engage_flag, soft_esc_sensors_every_tick

    if soft_esc_sensors_every_tick():
        return False
    if bool(read_engage_flag(default=False)):
        return False
    if _debug_force_engage():
        return False
    return True


def clamp_far_m(cid: str, far_m: float) -> float:
    lo, hi = far_band(cid)
    f = float(far_m)
    # Silent BeamNGpy 100 m is not a hitch cut. Narrow live floor is 400, attach stays 800.
    if cid == "narrow" and f <= BNGPY_DEFAULT_FAR_M + 1e-9:
        return default_far_m("narrow")
    return min(hi, max(lo, f))


def camera_clip_planes(
    spec: dict[str, Any] | None,
    *,
    cid: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Per-cam (near_m, far_m) for BeamNGpy Camera.near_far_planes.

    Precedence: spec.near_far_planes > spec.near_m/far_m > defaults.near_m + role far.
    Role far: narrow 800, main 300, wide 300, pillar/repeat 100, rear 100. Clamped to
    the role band so a missing yaml far cannot silently omit near_far_planes, main
    cannot sit at 800 next to narrow, and sides/rear cannot sit at 150. Narrow 400
    is allowed (live hitch if camera_hz still <10); attach yaml stays 800.
    """
    spec = spec or {}
    defaults = defaults or {}
    cam_id = str(cid or spec.get("id") or "")
    near = DEFAULT_NEAR_M
    far = default_far_m(cam_id)
    dn = _as_float(defaults.get("near_m"))
    if dn is not None:
        near = dn
    # File-level far_m is not applied: Spec locks per-id far_m (narrow 800 > main 300).
    sn = _as_float(spec.get("near_m"))
    if sn is not None:
        near = sn
    sf = _as_float(spec.get("far_m"))
    if sf is not None:
        far = sf
    nfp = _pair2(spec.get("near_far_planes"))
    if nfp is None:
        nfp = _pair2(defaults.get("near_far_planes"))
    if nfp is not None:
        near, far = nfp
    if near <= 0 or near >= far:
        near = DEFAULT_NEAR_M
    return (float(near), clamp_far_m(cam_id, far))


def live_narrow_far_m(
    unique_hz: float,
    current_far: float,
    *,
    elapsed_s: float = 0.0,
    unique_n: int = 0,
) -> float:
    """800→400 when unique-frame Hz stays <10 after stagger. Still ≥ main 300."""
    cur = float(current_far)
    floor = float(NARROW_FAR_LIVE_HITCH_M)
    if cur <= floor + 1e-9:
        return floor if cur <= 0 else cur
    if elapsed_s < LIVE_NARROW_HITCH_AFTER_S or int(unique_n) < LIVE_NARROW_HITCH_MIN_UNIQUE:
        return cur
    if float(unique_hz) < CAMERA_HZ_TARGET:
        return floor
    return cur


def far_hitch_ladder(cid: str, far_m: float, *, unique_hz: float | None = None) -> tuple[float, ...]:
    """Requested far, then lower rungs still inside the role band.

    Locked attach is a single rung: narrow 800, main/wide 300, sides/rear 100.
    When unique_hz is set and still <10, the ladder includes live narrow 400.
    Values are clamped first so a both-800 input cannot stay on the ladder.
    """
    far_m = clamp_far_m(cid, float(far_m))
    if cid == "narrow":
        rungs = NARROW_FAR_HITCH
        if unique_hz is not None and float(unique_hz) < CAMERA_HZ_TARGET:
            rungs = (float(far_m), NARROW_FAR_LIVE_HITCH_M)
    elif cid == "main":
        rungs = MAIN_FAR_HITCH
    elif cid == "wide":
        rungs = WIDE_FAR_HITCH
    elif cid in REAR_CAM_IDS:
        rungs = REAR_FAR_HITCH
    else:
        rungs = SIDE_FAR_HITCH
    lo, _hi = far_band(cid)
    out: list[float] = []
    seen: set[float] = set()
    for f in (far_m,) + tuple(float(x) for x in rungs):
        if f > far_m + 1e-9 or f < lo - 1e-9:
            continue
        key = round(f, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(f))
    return tuple(out) or (far_m,)


def iter_clip_attach_attempts(
    spec: dict[str, Any] | None,
    *,
    cid: str | None = None,
    update_s: float | None = None,
    defaults: dict[str, Any] | None = None,
) -> list[tuple[float, float, float]]:
    """(near_m, far_m, update_s) tries. Per-id rate from yaml; never drop resolution.

    `update_s` is ignored when the spec already has requested_update_time / update_s,
    and is ignored for sides/rear so a global 0.067 cannot clobber on-demand -1.
    """
    spec = spec or {}
    cam_id = str(cid or spec.get("id") or "")
    near_m, far_m = camera_clip_planes(spec, cid=cam_id, defaults=defaults)
    rate = camera_update_s(spec, cid=cam_id)
    if update_s is not None and cam_id in FORWARD_CAM_IDS:
        has_own = _as_float(spec.get("requested_update_time")) is not None or _as_float(spec.get("update_s")) is not None
        if not has_own:
            rate = float(update_s)
    return [(near_m, f, rate) for f in far_hitch_ladder(cam_id, far_m)]


def sensor_requested_update_s(update_s: float) -> float:
    """Camera ``requested_update_time`` for a yaml rate.

    Yaml ``-1`` is the hitch. The sensor still renders offscreen at
    ``COMPANION_OFFSCREEN_UPDATE_S``. A negative rate is not sent: that would
    need SendAdHocRequestCamera, which flashes the game view.
    """
    rate = float(update_s)
    if rate < 0:
        return COMPANION_OFFSCREEN_UPDATE_S
    return rate


def beamng_camera_sensor_kwargs(
    *,
    pos: tuple[float, float, float],
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    fov_v: float,
    resolution: tuple[int, int],
    update_s: float,
    near_m: float,
    far_m: float,
    shmem: bool,
    streaming: bool,
    rgb_only: bool,
    update_priority: float = 0.0,
) -> dict[str, Any]:
    """Kwargs for beamngpy.sensors.Camera (name, bng, vehicle stay positional).

    `is_streaming` is always True (stream_raw). The streaming arg is accepted
    for callers but never written False.
    """
    _ = streaming  # kept so call sites stay stable; never set is_streaming False
    # A negative rate would need SendAdHocRequestCamera to get a picture.
    # That request renders through the game view and flashes exposure for one
    # frame. The sensor still updates offscreen; GVD only reads it on the hitch.
    rate = sensor_requested_update_s(update_s)
    return {
        "requested_update_time": rate,
        "update_priority": float(min(1.0, max(0.0, update_priority))),
        "pos": pos,
        "dir": direction,
        "up": up,
        "field_of_view_y": float(fov_v),
        "resolution": (int(resolution[0]), int(resolution[1])),
        "near_far_planes": (float(near_m), float(far_m)),
        "is_using_shared_memory": bool(shmem),
        "is_streaming": True,
        "is_render_colours": True,
        "is_render_annotations": not bool(rgb_only),
        "is_render_instance": False,
        "is_render_depth": not bool(rgb_only),
        "is_snapping_desired": False,
        "is_visualised": False,
    }


def _cam_resolution(cam: Any) -> tuple[int, int] | None:
    res = getattr(cam, "resolution", None)
    if res is None:
        return None
    try:
        w, h = int(res[0]), int(res[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w < 1 or h < 1:
        return None
    return (w, h)


# A daylight colour buffer can sit in the top of the 8-bit range. Pavement
# and a thin lane stripe are then both white paint, so the stripe is not a
# separate region. The viewport picture keeps that road near mid gray.
# Correct it once, here, on the frame that is stored. The lane fit, the
# model, the CAMS tiles, and the PIP all read that frame. A display dim
# never reaches the fit. Mid-gray, dark, and all-zero frames stay put.
_TONE_P40_MIN = 188.0
_TONE_SPREAD_MAX = 70.0
_TONE_ANCHOR_PCT = 78.0
_TONE_TARGET = 136.0
_TONE_GAMMA_CAP = 9.0
_TONE_SAT_MAX = 22.0
# Highway geometry paints the hood from here to the bottom. That panel is
# low-saturation and can be near white while the asphalt is already mid gray.
# It is not the road sample.
_TONE_HOOD_TOP = 0.93
# The rectangle from 42% to the hood is mostly sky. The road is a triangle
# under the vanishing point, so the 40th percentile of that rectangle is the
# sky (186.85 on the test sky, just under the gate). A sky a few levels
# brighter then sets the curve and crushes a mid-gray road. The sample is
# this trapezoid: below the vanishing point, inside the sky corners, above
# the hood.
_TONE_ROAD_TOP = 0.75
_TONE_ROAD_TOP_X0 = 0.25
_TONE_ROAD_TOP_X1 = 0.75
_TONE_ROAD_BOT_X0 = 0.08
_TONE_ROAD_BOT_X1 = 0.92


def _shadow_luma_blue(bgr: np.ndarray) -> tuple[float, float] | None:
    """Darker low-saturation pixels: mean luma, and mean (B - R).

    A settled gray road is a small blue excess. The one-frame companion
    flash is that same shadow, brighter and much bluer. A red/blue channel
    swap does not do this: a gray pixel stays gray.
    """
    if bgr.ndim != 3 or bgr.shape[2] < 3 or bgr.size == 0:
        return None
    img = bgr[:, :, :3]
    if int(img.shape[0]) >= 8:
        img = img[int(img.shape[0]) * 45 // 100 :]
    sample = img.astype(np.float32, copy=False)
    luma = 0.114 * sample[:, :, 0] + 0.587 * sample[:, :, 1] + 0.299 * sample[:, :, 2]
    blue = sample[:, :, 0] - sample[:, :, 2]
    if int(luma.size) < 8:
        return None
    # Darker pixels are the road shadow. A white lane is brighter and stays out.
    cut = float(np.percentile(luma, 45))
    dark = luma <= cut
    if int(np.count_nonzero(dark)) < 8:
        dark = np.ones(luma.shape, dtype=bool)
    return float(luma[dark].mean()), float(blue[dark].mean())


def companion_frame_is_flash(new_bgr: np.ndarray, held_bgr: np.ndarray | None) -> bool:
    """True when this companion buffer is the one bright-blue frame.

    Brightness and the blue shadow are one frame, not two faults. A channel
    swap is not it. Main is not passed here. No settled frame yet → False.
    """
    if held_bgr is None or frame_is_unrendered(held_bgr) or frame_is_unrendered(new_bgr):
        return False
    if new_bgr.shape[:2] != held_bgr.shape[:2]:
        return False
    new_s = _shadow_luma_blue(new_bgr)
    held_s = _shadow_luma_blue(held_bgr)
    if new_s is None or held_s is None:
        return False
    new_luma, new_blue = new_s
    held_luma, held_blue = held_s
    return (new_luma - held_luma) >= 18.0 and (new_blue - held_blue) >= 18.0


def _tone_road_pixels(bgr: np.ndarray) -> np.ndarray:
    """Road-body pixels above the hood, shape ``(N, 3)``.

    A full-width band in that range counts the sky beside the road. This
    trapezoid stays on the asphalt.
    """
    height = int(bgr.shape[0])
    width = int(bgr.shape[1])
    y0 = int(height * _TONE_ROAD_TOP)
    y1 = int(height * _TONE_HOOD_TOP)
    if y1 <= y0 + 4 or width < 8:
        return bgr.reshape(-1, 3)
    rows = bgr[y0:y1]
    span = float(max(int(rows.shape[0]) - 1, 1))
    t = np.arange(int(rows.shape[0]), dtype=np.float32) / span
    left = width * (_TONE_ROAD_TOP_X0 + t * (_TONE_ROAD_BOT_X0 - _TONE_ROAD_TOP_X0))
    right = width * (_TONE_ROAD_TOP_X1 + t * (_TONE_ROAD_BOT_X1 - _TONE_ROAD_TOP_X1))
    xs = np.arange(width, dtype=np.float32)
    mask = (xs[None, :] >= left[:, None]) & (xs[None, :] < right[:, None])
    picked = rows[mask]
    if int(picked.shape[0]) < 32:
        return rows.reshape(-1, 3)
    return np.ascontiguousarray(picked)


def recover_viewport_tone(bgr: np.ndarray) -> np.ndarray:
    """Pull a sun-blown colour buffer toward the viewport picture.

    A highlight knee that keeps a fraction of the levels above 140 pulls the
    road and the stripe together, and the stripe drops through the white-paint
    cut. A power curve on a near-white, low-contrast road does the opposite:
    the road lands near mid gray and the few brighter stripe levels stay above
    that cut. The sample is the road body above the hood, inside the sky
    corners. A frame whose road is already mid gray is returned as it was.
    An all-zero buffer is returned as it was.
    """
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    # Nothing in the white-paint range. Mid gray, dark, and all-zero stay put.
    if bgr.dtype != np.uint8 or bgr.size == 0 or int(bgr.max()) < int(_TONE_P40_MIN):
        return bgr
    pixels = _tone_road_pixels(bgr)
    if pixels.size == 0:
        return bgr
    sample = pixels.astype(np.float32, copy=False)
    luma = 0.114 * sample[:, 0] + 0.587 * sample[:, 1] + 0.299 * sample[:, 2]
    # Road body. The 42%–93% rectangle's 40th percentile is the sky, and warm
    # asphalt sits above the low-sat cut, so either of those samples
    # gamma-crushes a mid-gray road.
    if float(np.percentile(luma, 40)) < _TONE_P40_MIN:
        return bgr
    hi = sample.max(axis=1)
    lo = sample.min(axis=1)
    sat = np.where(hi > 0.0, (hi - lo) * 255.0 / np.maximum(hi, 1.0), 0.0)
    chosen = luma[sat <= _TONE_SAT_MAX]
    if int(chosen.size) < max(32, int(luma.size) // 20):
        chosen = luma.reshape(-1)
    p40 = float(np.percentile(chosen, 40))
    anchor = float(np.percentile(chosen, _TONE_ANCHOR_PCT))
    spread = float(np.percentile(chosen, 90) - p40)
    if p40 < _TONE_P40_MIN or spread > _TONE_SPREAD_MAX:
        return bgr
    x = min(anchor, 250.0) / 255.0
    y = _TONE_TARGET / 255.0
    if x <= y or x >= 0.999:
        return bgr
    gamma = math.log(y) / math.log(x)
    if not math.isfinite(gamma) or gamma < 1.05:
        return bgr
    gamma = min(_TONE_GAMMA_CAP, gamma)
    levels = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.rint(np.power(levels, gamma) * 255.0), 0, 255).astype(np.uint8)
    return lut[bgr]


def colour_to_bgr(
    colour: Any,
    resolution: tuple[int, int] | None = None,
    *,
    tone: bool = True,
) -> np.ndarray | None:
    """RGB(A) / BGRA bytes / array / PIL → BGR uint8. Empty → None.

    ``tone=True`` matches a sun-blown buffer to the viewport picture on the
    returned frame. The live store path passes ``tone=False`` so the companion
    flash check sees the buffer before that curve, then tones only the frame
    it keeps.
    """
    if colour is None:
        return None
    arr: np.ndarray | None = None
    if isinstance(colour, np.ndarray):
        arr = colour
    elif hasattr(colour, "mode"):
        arr = np.asarray(colour)
    elif isinstance(colour, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(memoryview(colour), dtype=np.uint8)
        n = int(buf.size)
        if resolution is not None:
            w, h = int(resolution[0]), int(resolution[1])
            need4, need3 = w * h * 4, w * h * 3
            if n >= need4:
                arr = buf[:need4].reshape(h, w, 4)
            elif n >= need3:
                arr = buf[:need3].reshape(h, w, 3)
        if arr is None and n >= 12:
            if n % 4 == 0:
                px = n // 4
                side = int(px**0.5)
                if side * side == px:
                    arr = buf.reshape(side, side, 4)
            if arr is None and n % 3 == 0:
                px = n // 3
                side = int(px**0.5)
                if side * side == px:
                    arr = buf.reshape(side, side, 3)
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
        return None
    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if not tone:
        return bgr
    return recover_viewport_tone(bgr)


class _IoMark:
    """Identity sentinel for a camera read that did not finish on this tick."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


_IO_BUSY = _IoMark("busy")
_IO_TIMEOUT = _IoMark("timeout")
_camera_backends: weakref.WeakSet = weakref.WeakSet()


def camera_ge_socket_busy() -> bool:
    """True when some live backend still has a camera read on the GE socket.

    ``stream_raw`` is shared memory and does not count. Callers that would
    send ``sensors.poll`` or ``vehicle.control`` check this and wait a tick.
    """
    for backend in list(_camera_backends):
        try:
            if backend._socket_io_busy():
                return True
        except Exception:
            continue
    return False


def _reading_colour(images: Any, resolution: tuple[int, int] | None) -> np.ndarray | None:
    if not isinstance(images, dict):
        return None
    colour = images.get("colour") if images.get("colour") is not None else images.get("color")
    # Flash is judged on this buffer. Tone runs later, on the frame that is stored.
    return colour_to_bgr(colour, resolution, tone=False)


def camera_uses_adhoc(cam: Any) -> bool:
    """True when this sensor can render an on-demand frame without poll().

    BeamNGpy Camera.poll / stream_raw on ``requested_update_time < 0`` return
    the last buffer, which stays zeros until an ad-hoc request is rendered.
    """
    return all(
        callable(getattr(cam, name, None))
        for name in (
            "send_ad_hoc_poll_request",
            "is_ad_hoc_poll_request_ready",
            "collect_ad_hoc_poll_request",
        )
    )


def cap_companion_pending(cam: Any) -> None:
    """Ask the sim to keep a single GPU request queued for this companion.

    poll() on a negative update rate can enqueue renders that never complete.
    One pending slot stops that queue from growing for the life of the session.
    """
    setter = getattr(cam, "set_max_pending_requests", None)
    if not callable(setter):
        return
    try:
        setter(1)
    except Exception:
        pass


def read_camera_colour(
    cam: Any,
    *,
    cid: str,
    resolution: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Main: stream_raw. Wide, narrow, sides, and rear: poll.

    One stream_raw per grab (main, every tick). Wide and narrow alternate
    across ticks and never share one. is_streaming stays true; poll is not a
    second stream_raw. On-demand BeamNGpy cameras take the ad-hoc path in
    ``BeamNGPyBackend.grab`` instead of this poll — poll does not render them.
    """
    res = resolution or _cam_resolution(cam)
    if cid == "main":
        if not hasattr(cam, "stream_raw"):
            return None
        try:
            raw = cam.stream_raw()
        except Exception:
            return None
        return _reading_colour(raw, res)
    # Wide/narrow and sides/rear: poll. Never set streaming false.
    # Cameras that implement ad-hoc are not read here; grab() owns that path.
    if camera_uses_adhoc(cam):
        return None
    if not hasattr(cam, "poll"):
        return None
    try:
        images = cam.poll()
    except Exception:
        return None
    return _reading_colour(images, res)


def frame_is_unrendered(bgr: Any) -> bool:
    """True when there is no picture.

    An on-demand BeamNG colour buffer is often all zeros until the first
    render. That must not be cached as a healthy CAMS tile. One non-zero
    pixel counts, so a dark frame is kept.
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or getattr(bgr, "size", 0) == 0:
        return True
    try:
        return int(bgr.max()) == 0
    except (TypeError, ValueError):
        return True


def frame_signature(bgr: np.ndarray) -> tuple[Any, ...]:
    """Cheap identity for unique-GPU-frame detection (stream_raw re-shows share bytes)."""
    arr = np.ascontiguousarray(bgr)
    return (arr.shape, arr.dtype.str, hash(arr.tobytes()))


@runtime_checkable
class CameraBackend(Protocol):
    name: str

    def open(self) -> None: ...
    def close(self) -> None: ...
    def grab(self) -> CameraFrameBundle: ...


class StubBackend:
    name = "stub"

    def __init__(self, long_side: int | None = None) -> None:
        self.long_side = long_side or load_live_long_side()

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def grab(self) -> CameraFrameBundle:
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        return CameraFrameBundle(
            frames={},
            timestamps={},
            health=health,
            backend=self.name,
            note="stub: no cameras",
            unique_gpu_n=0,
        )


_SKIP_BEAMNG_TITLE = ("crash", "dump", "werfault", "error report", "crashreporter")


def _skip_beamng_title(title: str) -> bool:
    """Drop crash/dump dialogs; they also contain 'BeamNG' and are the wrong capture."""
    t = (title or "").lower()
    if "beamng" not in t:
        return True
    return any(bad in t for bad in _SKIP_BEAMNG_TITLE)


def _beamng_window_rank(title: str, w: int, h: int) -> tuple[int, int]:
    """Retail prefers BeamNG.drive, then a generic BeamNG title, then Tech. Largest area wins ties."""
    t = (title or "").lower()
    if "beamng.drive" in t:
        kind = 3
    elif "beamng.tech" in t:
        kind = 1
    else:
        kind = 2
    return (kind, int(w) * int(h))


def _pick_best_beamng_rect(
    candidates: list[tuple[str, int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """candidates: (title, left, top, width, height)."""
    best: tuple[tuple[int, int], tuple[int, int, int, int]] | None = None
    for title, left, top, w, h in candidates:
        if _skip_beamng_title(title) or w <= 200 or h <= 200:
            continue
        rank = _beamng_window_rank(title, w, h)
        rect = (left, top, w, h)
        if best is None or rank > best[0]:
            best = (rank, rect)
    return None if best is None else best[1]


def _region_moved(a: dict[str, int] | None, b: dict[str, int] | None, slop: int = 8) -> bool:
    if a is None or b is None:
        return a is not b
    return any(abs(int(a[k]) - int(b[k])) > slop for k in ("left", "top", "width", "height"))


def _find_beamng_window_rect() -> tuple[int, int, int, int] | None:
    """Return (left, top, width, height) for the retail BeamNG.drive window, else None."""
    # Windows: win32gui
    try:
        import win32gui  # type: ignore

        found: list[tuple[str, int, int, int, int]] = []

        def _enum(hwnd: int, _: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            try:
                if win32gui.IsIconic(hwnd):
                    return
            except Exception:
                pass
            title = win32gui.GetWindowText(hwnd) or ""
            if _skip_beamng_title(title):
                return
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            found.append((title, left, top, w, h))

        win32gui.EnumWindows(_enum, None)
        picked = _pick_best_beamng_rect(found)
        if picked:
            return picked
    except Exception:
        pass
    # Linux: wmctrl (best-effort)
    try:
        import subprocess

        out = subprocess.check_output(["wmctrl", "-lG"], text=True, timeout=2)
        found_l: list[tuple[str, int, int, int, int]] = []
        for line in out.splitlines():
            if "BeamNG" not in line:
                continue
            parts = line.split(None, 7)
            if len(parts) < 7:
                continue
            x, y, w, h = map(int, parts[2:6])
            title = parts[7] if len(parts) > 7 else line
            found_l.append((title, x, y, w, h))
        picked = _pick_best_beamng_rect(found_l)
        if picked:
            return picked
    except Exception:
        pass
    return None


class WindowBackend:
    """Retail: capture BeamNG window (title match) or full monitor if fullscreen-only."""

    name = "window"

    def __init__(self, long_side: int | None = None) -> None:
        self.long_side = long_side or load_live_long_side()
        self._cam = None
        self._mss = None
        self._impl = "none"
        self._logged = False
        self._region: dict[str, int] | None = None
        self._note = "retail: 1 window"
        self._title_match = False

    def _apply_window_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        """Lock onto the Drive window when it exists; keep last region if the title flickers."""
        if not rect:
            if not self._title_match:
                self._note = "retail: 1 window (fullscreen/monitor — no BeamNG title match)"
            return
        l, t, w, h = rect
        new_region = {"left": l, "top": t, "width": w, "height": h}
        moved = _region_moved(self._region, new_region)
        self._region = new_region
        self._title_match = True
        self._note = "retail: 1 window (BeamNG title match)"
        if moved and str(self._impl).startswith("bettercam"):
            self._recreate_capture()

    def open(self) -> None:
        self._apply_window_rect(_find_beamng_window_rect())

        try:
            import bettercam  # type: ignore

            # numpy path — do not steal 1080 Ti via nvidia_gpu capture
            kwargs: dict[str, Any] = {"output_color": "BGR", "nvidia_gpu": False}
            if self._region:
                r = self._region
                kwargs["region"] = (r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"])
            self._cam = bettercam.create(**kwargs)
            self._impl = "bettercam"
            return
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            self._cam = None
        try:
            import mss  # type: ignore

            self._mss = mss.mss()
            self._impl = "mss"
        except Exception as e:
            self._impl = f"unavailable:{e}"

    def close(self) -> None:
        self._cam = None
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None

    def _recreate_capture(self) -> None:
        """Rebuild bettercam/mss against current region after a late BeamNG title match."""
        self._cam = None
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
        try:
            import bettercam  # type: ignore

            kwargs: dict[str, Any] = {"output_color": "BGR", "nvidia_gpu": False}
            if self._region:
                r = self._region
                kwargs["region"] = (r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"])
            self._cam = bettercam.create(**kwargs)
            self._impl = "bettercam"
            return
        except Exception:
            self._cam = None
        try:
            import mss  # type: ignore

            self._mss = mss.mss()
            self._impl = "mss"
        except Exception as e:
            self._impl = f"unavailable:{e}"

    def grab(self) -> CameraFrameBundle:
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        img = None
        # Re-resolve every grab: Steam starts after the supervisor, and a windowed
        # player can move/resize BeamNG.drive. Prefer Drive over Tech/crash dialogs.
        self._apply_window_rect(_find_beamng_window_rect())

        try:
            if self._cam is not None:
                img = self._cam.grab()
                if img is not None and not isinstance(img, np.ndarray):
                    img = np.asarray(img)
            elif self._mss is not None:
                if self._region:
                    mon = {
                        "left": self._region["left"],
                        "top": self._region["top"],
                        "width": self._region["width"],
                        "height": self._region["height"],
                    }
                else:
                    mon = self._mss.monitors[1] if len(self._mss.monitors) > 1 else self._mss.monitors[0]
                shot = self._mss.grab(mon)
                img = np.asarray(shot)[:, :, :3]
        except BaseException as e:
            # bettercam/comtypes can AV or raise COMError — never crash the supervisor
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            if not self._logged:
                print(f"[GVD] window capture error ({type(e).__name__}): {e}")
                self._logged = True
            # Disable bettercam for this session; fall back to mss if possible
            if self._cam is not None:
                self._cam = None
                try:
                    import mss  # type: ignore

                    if self._mss is None:
                        self._mss = mss.mss()
                    self._impl = "mss(after-bettercam-fail)"
                    self._note = f"{self._note}; bettercam disabled after error"
                except Exception:
                    self._impl = "unavailable:bettercam-fail"
            health["main"] = CamHealth.ERROR
            return CameraFrameBundle(
                frames={},
                timestamps={},
                health=health,
                backend=self.name,
                note=self._note,
                unique_gpu_n=0,
            )

        if img is None:
            health["main"] = CamHealth.MISSING
            if not self._logged:
                print(f"[GVD] window via {self._impl}; {self._note}. Sides stay missing.")
                self._logged = True
            return CameraFrameBundle(
                frames={},
                timestamps={},
                health=health,
                backend=self.name,
                note=self._note,
                unique_gpu_n=0,
            )

        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = resize_long_side(img, self.long_side)
        if getattr(img, "ndim", 0) == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            img = recover_viewport_tone(img)
        ts = time.time()
        frames["main"] = img
        frames["cam_main"] = img
        timestamps["main"] = ts
        health["main"] = CamHealth.OK
        # refuse fake 8-cam
        return CameraFrameBundle(
            frames=frames,
            timestamps=timestamps,
            health=health,
            backend=self.name,
            note=self._note,
            unique_gpu_n=1,
            unique_gpu_ids=("main",),
        )


class BeamNGPyBackend:
    """Tech path: attach color-only BeamNGpy Camera sensors from cameras.yaml."""

    name = "beamngpy"

    def __init__(
        self,
        long_side: int | None = None,
        config: dict[str, Any] | None = None,
        tech_config: dict[str, Any] | None = None,
    ) -> None:
        self.long_side = long_side or load_live_long_side()
        self.config = config or load_camera_config()
        from python.sensors.tech import TechSession, load_tech_config

        self.session = TechSession(tech_config if tech_config is not None else load_tech_config())
        self._sensors: dict[str, Any] = {}
        self._clip_planes: dict[str, tuple[float, float]] = {}
        self._update_s: dict[str, float] = {}
        self._update_priority: dict[str, float] = {}
        self._resolution: dict[str, tuple[int, int]] = {}
        self._logged = False
        self._ok = False
        self.connect_failed = False
        self._grab_i = 0
        self._cache_frames: dict[str, np.ndarray] = {}
        self._cache_ts: dict[str, float] = {}
        self._hitch_steps: list[tuple[str, float, float, float]] = []  # cid, near, far, update_s
        hitch = self.config.get("hitch") if isinstance(self.config.get("hitch"), dict) else {}
        self._hitch = hitch
        self._grab_div = {cid: camera_grab_div(cid, hitch) for cid in CAM_IDS}
        self._grab_phase = {cid: camera_grab_phase(cid, hitch) for cid in CAM_IDS}
        self._side_grab_div = self._grab_div["pillarL"]
        self._rear_grab_div = self._grab_div["rear"]
        self._Camera: Any = None
        self._sensor_kwargs: dict[str, dict[str, Any]] = {}
        self._frame_sig: dict[str, tuple[Any, ...]] = {}
        self._unique_hz_ema = 0.0
        self._unique_n = 0
        self._open_mono = 0.0
        self._narrow_live_hitched = False
        self._last_unique_tick_t = 0.0
        # One ad-hoc companion render in flight: (cid, request_id).
        self._adhoc: tuple[str, int] | None = None
        self._adhoc_ready = False
        self._owed: list[str] = []
        self._io_thread: threading.Thread | None = None
        self._io_box: dict[str, Any] = {}
        self._io_job: tuple[str, str] | None = None
        self._read_blocked = False
        self._late_unique: list[str] = []
        _camera_backends.add(self)

    def open(self) -> None:
        self.connect_failed = False
        try:
            from beamngpy.sensors import Camera  # type: ignore
        except Exception as e:
            if not self._logged:
                print(f"[GVD] beamngpy not available ({e}); cam_health=missing.")
                self._logged = True
            self._ok = False
            self.connect_failed = True
            return
        self._Camera = Camera

        if not self.session.connect(explicit=True):
            self._ok = False
            self._logged = True
            self.connect_failed = True
            return

        # Cameras + vehicle sensors attach here, independent of Alt+G / engaged.
        # Soft Esc keeps all 8 attached. Colour while engaged=false is the same ÷16 hitch.
        try:
            self.session.attach_vehicle_sensors()
        except Exception as e:
            print(f"[GVD] tech vehicle sensors: {e}")

        vehicle = self.session.vehicle
        bng = self.session.bng
        cam_cfg = self.session.config.get("cameras") if isinstance(self.session.config.get("cameras"), dict) else {}
        if cam_cfg.get("attach", True) is False:
            self._ok = False
            print("[GVD] tech.yaml cameras.attach=false; cam_health=missing.")
            self._logged = True
            return

        cams = list(self.config.get("cameras") or [])
        origin = self.config.get("origin_offset") or [0.0, 0.0, 0.0]
        attached = 0
        shmem = bool(cam_cfg.get("shared_memory", True))
        if cam_cfg.get("streaming") is False:
            print("[GVD] cameras.streaming=false ignored; stream_raw needs is_streaming=True")
        rgb_only = bool(cam_cfg.get("rgb_only", True))
        clip_defaults = {"near_m": self.config.get("near_m", DEFAULT_NEAR_M)}
        self._clip_planes = {}
        self._update_s = {}
        self._update_priority = {}
        self._resolution = {}
        self._hitch_steps = []
        self._sensor_kwargs = {}
        self._frame_sig = {}
        self._unique_hz_ema = 0.0
        self._unique_n = 0
        self._open_mono = time.monotonic()
        self._narrow_live_hitched = False
        for spec in cams:
            cid = str(spec.get("id") or "")
            if not cid or cid not in CAM_IDS:
                continue
            try:
                pos = list(spec.get("pos_m") or [0.0, 1.2, 1.2])
                pos = [pos[0] + float(origin[0]), pos[1] + float(origin[1]), pos[2] + float(origin[2])]
                yaw = float(spec.get("yaw_deg") or 0.0)
                pitch = float(spec.get("pitch_deg") or 0.0)
                direction, up = yaw_pitch_to_dir_up(yaw, pitch)
                pos_bng, dir_bng, up_bng = self.session.camera_mount(pos, direction, up)
                res = list(spec.get("live_res") or [640, 480])
                rw, rh = int(res[0]), int(res[1])
                if max(rw, rh) > self.long_side:
                    scale = self.long_side / float(max(rw, rh))
                    rw, rh = max(1, int(rw * scale)), max(1, int(rh * scale))
                fov_v = float(spec.get("fov_v") or 70.0)
                want_near, want_far = camera_clip_planes(spec, cid=cid, defaults=clip_defaults)
                want_rate = camera_update_s(spec, cid=cid)
                want_prio = camera_update_priority(spec, cid=cid)
                last_err: Exception | None = None
                cam = None
                used_near, used_far, used_rate = want_near, want_far, want_rate
                used_prio = want_prio
                attempts = iter_clip_attach_attempts(spec, cid=cid, defaults=clip_defaults)
                step_s = " ".join(
                    f"far_m={try_far:g}@requested_update_time={sensor_requested_update_s(try_rate):g}"
                    for _n, try_far, try_rate in attempts
                )
                print(f"[GVD] beamngpy Camera hitch steps {cid}: {step_s} (not resolution)")
                for try_near, try_far, try_rate in attempts:
                    kwargs = beamng_camera_sensor_kwargs(
                        pos=pos_bng,
                        direction=dir_bng,
                        up=up_bng,
                        fov_v=fov_v,
                        resolution=(rw, rh),
                        update_s=try_rate,
                        near_m=try_near,
                        far_m=try_far,
                        shmem=shmem,
                        streaming=True,
                        rgb_only=rgb_only,
                        update_priority=want_prio,
                    )
                    try:
                        cam = Camera(f"gvd_{cid}", bng, vehicle, **kwargs)
                        used_near, used_far, used_rate = try_near, try_far, try_rate
                        used_prio = float(kwargs.get("update_priority", want_prio))
                        break
                    except TypeError as e:
                        last_err = e
                        msg = str(e)
                        if "update_priority" in msg and "update_priority" in kwargs:
                            kwargs = dict(kwargs)
                            kwargs.pop("update_priority", None)
                            try:
                                cam = Camera(f"gvd_{cid}", bng, vehicle, **kwargs)
                                used_near, used_far, used_rate = try_near, try_far, try_rate
                                used_prio = want_prio
                                break
                            except TypeError as e2:
                                last_err = e2
                                msg = str(e2)
                                cam = None
                            except Exception as e2:
                                last_err = e2
                                cam = None
                                continue
                        if cam is not None:
                            break
                        if "near_far_planes" in msg:
                            print(
                                f"[GVD] beamngpy Camera rejected near_far_planes for {cid} "
                                f"({e}); not attaching with silent {BNGPY_DEFAULT_FAR_M:g} m far."
                            )
                            cam = None
                            break
                    except Exception as e:
                        last_err = e
                        cam = None
                if cam is None:
                    print(f"[GVD] beamngpy Camera attach failed for {cid}: {last_err}")
                    continue
                if abs(used_far - want_far) > 1e-9 or abs(used_rate - want_rate) > 1e-9:
                    print(
                        f"[GVD] beamngpy Camera hitch {cid}: far_m {want_far:g}->{used_far:g} "
                        f"update_s {want_rate:g}->{used_rate:g} (not resolution)"
                    )
                self._sensors[cid] = cam
                if cid != "main":
                    cap_companion_pending(cam)
                self._clip_planes[cid] = (used_near, used_far)
                self._update_s[cid] = float(used_rate)
                self._update_priority[cid] = float(used_prio)
                self._resolution[cid] = (rw, rh)
                self._sensor_kwargs[cid] = dict(kwargs)
                self._hitch_steps.append((cid, float(used_near), float(used_far), float(used_rate)))
                attached += 1
            except Exception as e:
                print(f"[GVD] beamngpy Camera attach failed for {cid}: {e}")

        self._confirm_priority_scale()
        self._ok = attached > 0
        if not self._logged:
            if self._ok:
                clips = " ".join(f"{k}={v[1]:g}" for k, v in self._clip_planes.items())
                rates = " ".join(
                    f"{k}={float(self._sensor_kwargs.get(k, {}).get('requested_update_time', v)):g}"
                    for k, v in self._update_s.items()
                )
                prios = " ".join(f"{k}={v:g}" for k, v in self._update_priority.items())
                print(
                    f"[GVD] beamngpy attached {attached} color Camera(s) from cameras.yaml "
                    f"(near_far_planes far_m {clips}; requested_update_time {rates}; "
                    f"update_priority {prios}; "
                    f"grab_div main={self._grab_div['main']} wide={self._grab_div['wide']} "
                    f"narrow={self._grab_div['narrow']} side={self._side_grab_div} "
                    f"rear={self._rear_grab_div}; "
                    f"stream_raw offscreen requested_update_time 1; "
                    f"GVD→BeamNG vehicle-space convert; depth/semantic OFF)."
                )
            else:
                print("[GVD] beamngpy: zero Cameras attached; cam_health=missing.")
            print("[GVD] note: live BeamNG frame grab still needs Windows Tech smoke if this host cannot render.")
            self._logged = True

    @property
    def vehicle(self):
        """Player vehicle handle when attached (M3 actuation / Electrics)."""
        return self.session.vehicle

    @property
    def bng(self):
        return self.session.bng

    def _socket_io_busy(self) -> bool:
        """True when a timed-out camera read is still inside a GE round-trip.

        ``stream_raw`` only reads shared memory, so it may overlap ``sensors.poll``.
        ``poll`` and ad-hoc calls use the one BeamNGpy socket. The rest of the
        tick must not use that socket until the read returns.
        """
        if not self._io_busy():
            return False
        job = self._io_job
        if job is None:
            return False
        return job[1] != "stream"

    def poll_vehicle(self):
        if self._socket_io_busy():
            vehicle = self.session.vehicle
            last = getattr(self.session, "_last_vehicle_data", None)
            last_map = getattr(self.session, "_last_sensor_map", None)
            if vehicle is not None and last is not None and last_map is not None:
                # Same shape as Soft Esc coalesce: timers 0, snap republished.
                return self.session._coalesced_vehicle_data(vehicle)
            from python.sensors.tech import VehicleData

            return VehicleData(
                connected=vehicle is not None,
                note="camera io in flight; sensors.poll skipped",
            )
        return self.session.poll()

    def close(self) -> None:
        # A timed-out GE read is still on the one BeamNGpy socket. Wait for it
        # before remove() and disconnect so close does not race that call.
        thread = self._io_thread
        if thread is not None and thread.is_alive():
            thread.join()
        self._reap_if_done()
        for cam in list(self._sensors.values()):
            try:
                cam.remove()
            except Exception:
                try:
                    cam.detach()
                except Exception:
                    pass
        self._sensors.clear()
        self._clip_planes.clear()
        self._update_s.clear()
        self._update_priority.clear()
        self._resolution.clear()
        self._hitch_steps.clear()
        self._sensor_kwargs.clear()
        self._frame_sig.clear()
        self._cache_frames.clear()
        self._cache_ts.clear()
        self._adhoc = None
        self._adhoc_ready = False
        self._owed.clear()
        self._late_unique.clear()
        self._grab_i = 0
        self._unique_hz_ema = 0.0
        self._unique_n = 0
        self._narrow_live_hitched = False
        self.session.close()
        self._ok = False

    def _grab_this_tick(self, cid: str, grab_i: int) -> bool:
        return camera_grab_due(cid, grab_i, self._hitch)

    def _store_frame(self, cid: str, bgr: np.ndarray, ts: float, frames: dict, timestamps: dict, health: dict) -> None:
        frames[cid] = bgr
        timestamps[cid] = ts
        health[cid] = CamHealth.OK
        self._cache_frames[cid] = bgr
        self._cache_ts[cid] = ts
        if cid == "main":
            frames["cam_main"] = bgr

    def _reuse_cached(
        self, cid: str, frames: dict, timestamps: dict, health: dict, *, failed: bool
    ) -> None:
        """Keep the last real picture.

        A scheduled skip stays OK when a non-blank frame is cached. No cache
        leaves MISSING (the slot stays labelled). A read that only missed the
        time budget is the same skip: the last real frame stays OK so CAMS
        still counts the tile. A completed read that returns nothing is STALE
        and the pixels stay, so the tile does not go black. An all-zero buffer
        is not a picture and is not stored here.
        """
        bgr = self._cache_frames.get(cid)
        if frame_is_unrendered(bgr):
            bgr = None
        if not failed:
            if bgr is None:
                return
            frames[cid] = bgr
            timestamps[cid] = self._cache_ts.get(cid, time.time())
            health[cid] = CamHealth.OK
            if cid == "main":
                frames["cam_main"] = bgr
            return
        if bgr is None:
            health[cid] = CamHealth.MISSING
            return
        frames[cid] = bgr
        timestamps[cid] = self._cache_ts.get(cid, time.time())
        health[cid] = CamHealth.STALE
        if cid == "main":
            frames["cam_main"] = bgr

    def _confirm_priority_scale(self) -> None:
        """BeamNGpy getter: 0=highest. If main reads ~1, invert so main is not starved."""
        main = self._sensors.get("main")
        if main is None:
            return
        req = float(self._update_priority.get("main", 0.0))
        if priority_highest_is_zero(main, req):
            return
        print(
            "[GVD] BeamNGpy update_priority getter scale inverted "
            "(0 was not highest); flipping so main is not starved"
        )
        for cid, cam in list(self._sensors.items()):
            p = invert_update_priority(self._update_priority.get(cid, 0.5))
            self._update_priority[cid] = p
            if cid in self._sensor_kwargs:
                self._sensor_kwargs[cid]["update_priority"] = p
            setter = getattr(cam, "set_update_priority", None) or getattr(cam, "setUpdatePriority", None)
            if callable(setter):
                try:
                    setter(p)
                except Exception:
                    pass

    def _maybe_live_narrow_hitch(self) -> None:
        """Runtime 800→400 when unique-frame Hz stays <10. Attach yaml stays 800."""
        if self._narrow_live_hitched:
            return
        planes = self._clip_planes.get("narrow")
        if not planes:
            return
        elapsed = time.monotonic() - self._open_mono if self._open_mono else 0.0
        want = live_narrow_far_m(
            self._unique_hz_ema,
            float(planes[1]),
            elapsed_s=elapsed,
            unique_n=self._unique_n,
        )
        if want >= float(planes[1]) - 1e-9:
            return
        ladder = far_hitch_ladder("narrow", float(planes[1]), unique_hz=self._unique_hz_ema)
        nxt = next((f for f in ladder if f < float(planes[1]) - 1e-9), want)
        if abs(nxt - NARROW_FAR_LIVE_HITCH_M) > 1e-6 and abs(want - NARROW_FAR_LIVE_HITCH_M) > 1e-6:
            return
        # Reattach builds a new Camera on the GE socket and drops the old
        # request id. Both are refused while a timed-out ad-hoc call is in
        # that socket; the hitch retries on a later grab.
        if not self._reattach_cam("narrow", far_m=NARROW_FAR_LIVE_HITCH_M):
            return
        self._narrow_live_hitched = True
        print(
            f"[GVD] live unique-frame Hz={self._unique_hz_ema:.2f} < {CAMERA_HZ_TARGET:g}; "
            f"hitch narrow far_m {planes[1]:g}->{NARROW_FAR_LIVE_HITCH_M:g}"
        )

    def _drop_adhoc(self, cid: str) -> None:
        """Forget an in-flight request that belonged to a sensor we are replacing.

        Harvest would otherwise keep asking the new camera for the old id.
        A False or an exception there never clears the slot, so inflight stays
        1 and no other companion can send.
        """
        if self._adhoc is None or self._adhoc[0] != cid:
            return
        self._adhoc = None
        self._adhoc_ready = False
        self._owe(cid)

    def _reattach_cam(self, cid: str, *, far_m: float) -> bool:
        Camera = self._Camera
        kwargs = self._sensor_kwargs.get(cid)
        cam = self._sensors.get(cid)
        if Camera is None or kwargs is None or self.session.vehicle is None or self.session.bng is None:
            return False
        # OpenCamera is a GE round-trip. Do not start it while a timed-out
        # ad-hoc send/ready/collect still holds that socket.
        if self._socket_io_busy():
            return False
        self._reap_if_done()
        self._drop_adhoc(cid)
        near = float(kwargs.get("near_far_planes", (DEFAULT_NEAR_M, far_m))[0])
        new_kwargs = dict(kwargs)
        new_kwargs["near_far_planes"] = (near, float(far_m))
        if cam is not None:
            try:
                cam.remove()
            except Exception:
                try:
                    cam.detach()
                except Exception:
                    pass
        try:
            fresh = Camera(f"gvd_{cid}", self.session.bng, self.session.vehicle, **new_kwargs)
        except TypeError:
            dropped = dict(new_kwargs)
            dropped.pop("update_priority", None)
            try:
                fresh = Camera(f"gvd_{cid}", self.session.bng, self.session.vehicle, **dropped)
                new_kwargs = dropped
            except Exception:
                return False
        except Exception:
            return False
        self._sensors[cid] = fresh
        if cid != "main":
            cap_companion_pending(fresh)
        self._sensor_kwargs[cid] = new_kwargs
        rate = float(self._update_s.get(cid, FORWARD_UPDATE_S))
        self._clip_planes[cid] = (near, float(far_m))
        self._hitch_steps.append((cid, near, float(far_m), rate))
        self._frame_sig.pop(cid, None)
        return True

    def _io_busy(self) -> bool:
        thread = self._io_thread
        return thread is not None and thread.is_alive()

    def _inflight_n(self) -> int:
        if self._adhoc is not None:
            return 1
        job = self._io_job
        if job is not None and job[1] == "adhoc_send" and self._io_busy():
            return 1
        return 0

    def _drop_owed(self, cid: str) -> None:
        self._owed = [c for c in self._owed if c != cid]

    def _owe(self, cid: str) -> None:
        """Remember a companion whose hitch slot could not send.

        One entry per camera. The cap is the rig size so a stuck render cannot
        grow this list for the life of the session.
        """
        if cid in self._owed:
            return
        if self._adhoc is not None and self._adhoc[0] == cid:
            return
        if len(self._owed) >= _OWED_CAP:
            return
        self._owed.append(cid)

    def _take_finished_io(self) -> tuple[str, str, Any] | None:
        thread = self._io_thread
        if thread is None or thread.is_alive():
            return None
        self._io_thread = None
        cid, kind = self._io_job or ("", "")
        self._io_job = None
        box = self._io_box
        self._io_box = {}
        if "e" in box:
            return (cid, kind, None)
        return (cid, kind, box.get("v"))

    def _cache_late_colour(self, cid: str, bgr: np.ndarray | None) -> None:
        if not cid or bgr is None or frame_is_unrendered(bgr):
            return
        bgr = resize_long_side(bgr, self.long_side)
        if cid != "main" and companion_frame_is_flash(bgr, self._cache_frames.get(cid)):
            return
        bgr = recover_viewport_tone(bgr)
        self._cache_frames[cid] = bgr
        self._cache_ts[cid] = time.time()
        sig = frame_signature(bgr)
        if sig == self._frame_sig.get(cid):
            return
        self._frame_sig[cid] = sig
        if cid not in self._late_unique:
            self._late_unique.append(cid)

    def _take_late_unique(self, unique_ids: list[str]) -> None:
        for cid in self._late_unique:
            if cid not in unique_ids:
                unique_ids.append(cid)
        self._late_unique.clear()

    def _apply_late(self, cid: str, kind: str, value: Any) -> None:
        if kind == "adhoc_send" and isinstance(value, int) and self._adhoc is None:
            self._adhoc = (cid, value)
            self._adhoc_ready = False
            self._drop_owed(cid)
            return
        if kind == "adhoc_ready":
            self._adhoc_ready = bool(value)
            return
        if kind == "adhoc_collect":
            self._adhoc = None
            self._adhoc_ready = False
            bgr = _reading_colour(value, self._resolution.get(cid)) if isinstance(value, dict) else None
            self._cache_late_colour(cid, bgr)
            return
        if kind == "poll" and isinstance(value, np.ndarray):
            self._cache_late_colour(cid, value)
            return
        if kind == "stream":
            bgr = _reading_colour(value, self._resolution.get(cid)) if isinstance(value, dict) else None
            self._cache_late_colour(cid, bgr)

    def _reap_if_done(self) -> None:
        late = self._take_finished_io()
        if late is not None:
            self._apply_late(*late)

    def _io_call(self, cid: str, kind: str, fn: Any) -> Any:
        """Run ``fn`` and return its value.

        A call that is still going after ``CAM_READ_BUDGET_S`` returns
        ``_IO_TIMEOUT`` and leaves that one thread running. The next call
        does not start another read until it finishes, so a stuck
        ``stream_raw`` / ``poll`` / ad-hoc round-trip cannot stack or freeze
        the grab for the 0.5–1 s the sim was holding the socket.
        """
        if self._io_busy():
            self._read_blocked = True
            return _IO_BUSY
        self._reap_if_done()
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["v"] = fn()
            except Exception as exc:
                box["e"] = exc

        thread = threading.Thread(target=_run, daemon=True, name="gvd-cam-io")
        self._io_thread = thread
        self._io_box = box
        self._io_job = (cid, kind)
        thread.start()
        thread.join(CAM_READ_BUDGET_S)
        if thread.is_alive():
            self._read_blocked = True
            return _IO_TIMEOUT
        finished = self._take_finished_io()
        if finished is None:
            return None
        return finished[2]

    def _omit_unread(self, cid: str, frames: dict, timestamps: dict, health: dict) -> None:
        """This tick has no picture for ``cid``. Do not copy an older one."""
        frames.pop(cid, None)
        timestamps.pop(cid, None)
        if cid == "main":
            frames.pop("cam_main", None)
        health[cid] = CamHealth.MISSING

    def _publish_read(
        self,
        cid: str,
        bgr: np.ndarray | None,
        ts: float,
        frames: dict,
        timestamps: dict,
        health: dict,
        unique_ids: list[str],
    ) -> None:
        if bgr is None or frame_is_unrendered(bgr):
            self._omit_unread(cid, frames, timestamps, health)
            return
        bgr = resize_long_side(bgr, self.long_side)
        if cid != "main" and companion_frame_is_flash(bgr, self._cache_frames.get(cid)):
            # Brighter and blue in the shadows. Not stored, and not replaced
            # by the previous tick. Judged before the tone curve.
            self._omit_unread(cid, frames, timestamps, health)
            return
        bgr = recover_viewport_tone(bgr)
        sig = frame_signature(bgr)
        unique = sig != self._frame_sig.get(cid)
        self._store_frame(cid, bgr, ts, frames, timestamps, health)
        if unique:
            self._frame_sig[cid] = sig
            unique_ids.append(cid)

    def _kick_adhoc(self, cid: str, cam: Any) -> bool:
        """Start one ad-hoc render. False when a request is already in flight."""
        if self._adhoc is not None or self._io_busy():
            return False

        def _send() -> int:
            return int(cam.send_ad_hoc_poll_request())

        got = self._io_call(cid, "adhoc_send", _send)
        if isinstance(got, int):
            self._adhoc = (cid, got)
            self._adhoc_ready = False
            self._drop_owed(cid)
            return True
        if got is _IO_TIMEOUT or got is _IO_BUSY:
            self._drop_owed(cid)
            return True
        return False

    def _harvest_adhoc(
        self,
        ts: float,
        frames: dict,
        timestamps: dict,
        health: dict,
        unique_ids: list[str],
    ) -> None:
        """Collect a finished ad-hoc render without sending another."""
        if self._adhoc is None:
            return
        cid, rid = self._adhoc
        cam = self._sensors.get(cid)
        if cam is None or not camera_uses_adhoc(cam):
            self._adhoc = None
            self._adhoc_ready = False
            return
        if not self._adhoc_ready:
            flag = self._io_call(
                cid, "adhoc_ready", lambda: bool(cam.is_ad_hoc_poll_request_ready(rid))
            )
            if flag is _IO_BUSY or flag is _IO_TIMEOUT or flag is not True:
                return
        images = self._io_call(cid, "adhoc_collect", lambda: cam.collect_ad_hoc_poll_request(rid))
        if images is _IO_BUSY or images is _IO_TIMEOUT:
            self._adhoc_ready = True
            return
        self._adhoc = None
        self._adhoc_ready = False
        bgr = _reading_colour(images, self._resolution.get(cid)) if isinstance(images, dict) else None
        self._publish_read(cid, bgr, ts, frames, timestamps, health, unique_ids)

    def _reads_shared_memory(self, cid: str, cam: Any) -> bool:
        """Main, and any live camera that also offers ad-hoc.

        Ad-hoc is a viewport render. stream_raw only reads the offscreen buffer.
        A legacy companion with poll and no shared memory stays on poll.
        """
        if not hasattr(cam, "stream_raw"):
            return False
        return cid == "main" or camera_uses_adhoc(cam)

    def _bounded_sensor_colour(self, cid: str, cam: Any) -> np.ndarray | None:
        """stream_raw when the buffer is shared. Legacy poll has no ad-hoc API.

        SendAdHocRequestCamera is not used. It renders on the game view and
        the exposure flashes blue for one frame.
        """
        res = self._resolution.get(cid)

        if self._reads_shared_memory(cid, cam):
            def _stream() -> Any:
                try:
                    return cam.stream_raw()
                except Exception:
                    return None

            raw = self._io_call(cid, "stream", _stream)
            if raw is _IO_BUSY or raw is _IO_TIMEOUT:
                return raw
            return _reading_colour(raw, res)

        def _poll() -> np.ndarray | None:
            return read_camera_colour(cam, cid=cid, resolution=res)

        got = self._io_call(cid, "poll", _poll)
        if got is _IO_BUSY or got is _IO_TIMEOUT:
            return got
        if not isinstance(got, np.ndarray):
            return None
        return got

    def grab(self) -> CameraFrameBundle:
        from python.runtime.hw_probe import ema_hz, unique_frame_hz_inst

        t0 = time.perf_counter()
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        if not self._ok or not self._sensors:
            return CameraFrameBundle(
                frames={},
                timestamps={},
                health=health,
                backend=self.name,
                note="beamngpy: no live Camera session",
                grab_ms=(time.perf_counter() - t0) * 1000.0,
                unique_gpu_n=0,
            )
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        ts = time.time()
        grab_i = self._grab_i
        self._grab_i = grab_i + 1
        phase = grab_phase_of(grab_i, self._hitch)
        companion_polled = False
        unique_ids: list[str] = []
        self._read_blocked = False
        # Soft Esc and Engage both read all eight cameras on this tick.
        # A skipped, timed-out, or empty read stays missing. The previous
        # picture is not the stitch. Companions are not SendAdHocRequestCamera.
        # That render steps the game view's exposure for one frame. They
        # update offscreen; this grab reads shared memory.
        soft_esc_main = soft_esc_colour_main_only()
        self._reap_if_done()
        for cid, cam in self._sensors.items():
            if not self._grab_this_tick(cid, grab_i):
                self._omit_unread(cid, frames, timestamps, health)
                continue
            if cid != "main":
                companion_polled = True
            try:
                bgr = self._bounded_sensor_colour(cid, cam)
                if bgr is _IO_BUSY or bgr is _IO_TIMEOUT:
                    self._omit_unread(cid, frames, timestamps, health)
                else:
                    self._publish_read(cid, bgr, ts, frames, timestamps, health, unique_ids)
            except Exception:
                health[cid] = CamHealth.ERROR
                frames.pop(cid, None)
                timestamps.pop(cid, None)
                if cid == "main":
                    frames.pop("cam_main", None)
        self._maybe_live_narrow_hitch()
        self._take_late_unique(unique_ids)
        unique_n = len(unique_ids)
        now = time.perf_counter()
        if grab_i > 0 and self._last_unique_tick_t > 0:
            gap = max(1e-6, now - self._last_unique_tick_t)
            self._unique_hz_ema = ema_hz(self._unique_hz_ema, unique_frame_hz_inst(unique_n, gap))
        self._last_unique_tick_t = now
        if unique_n > 0:
            self._unique_n += 1
        n_ok = sum(1 for c in CAM_IDS if health.get(c) == CamHealth.OK)
        grab_ms = (time.perf_counter() - t0) * 1000.0
        note = (
            f"beamngpy: {n_ok} colour frame(s) grab={grab_i} unique={unique_n} "
            f"main_div={self._grab_div.get('main', MAIN_GRAB_DIV)} "
            f"wide_div={self._grab_div.get('wide', WIDE_GRAB_DIV)} "
            f"narrow_div={self._grab_div.get('narrow', NARROW_GRAB_DIV)} "
            f"side_div={self._side_grab_div} "
            f"repeat_div={self._grab_div.get('repeatL', REPEAT_GRAB_DIV)} "
            f"rear_div={self._rear_grab_div}"
        )
        # All eight cameras, this tick. Not one companion and not a cached frame.
        note += " same_tick=8"
        if soft_esc_main:
            note += " soft_esc_colour=hitch"
        return CameraFrameBundle(
            frames=frames,
            timestamps=timestamps,
            health=health,
            backend=self.name,
            note=note,
            grab_ms=grab_ms,
            grab_phase=phase,
            grab_poll_free=not companion_polled,
            unique_gpu_n=unique_n,
            unique_gpu_ids=tuple(unique_ids),
            companion_inflight=self._inflight_n(),
            grab_read_blocked=bool(self._read_blocked),
        )


def resolve_backend_name(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    # Do not pick Tech just because beamngpy is importable — that needs a live Tech session.
    if os.environ.get("GVD_BEAMNG", "").strip().lower() in ("1", "true", "yes"):
        return "beamngpy"
    try:
        import bettercam  # noqa: F401

        return "window"
    except Exception:
        pass
    try:
        import mss  # noqa: F401

        return "window"
    except Exception:
        pass
    return "stub"


def make_backend(name: str | None = None) -> CameraBackend:
    resolved = resolve_backend_name(name)
    if resolved == "beamngpy":
        return BeamNGPyBackend()
    if resolved == "window":
        return WindowBackend()
    return StubBackend()
