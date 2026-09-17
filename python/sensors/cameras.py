"""Camera backends for GVD — vision-only RGB. Never fake 8 cams from one window."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

CAM_IDS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
MAIN_ALIASES = ("main", "cam_main")
SIDE_CAM_IDS = frozenset(("pillarL", "pillarR", "repeatL", "repeatR"))
REAR_CAM_IDS = frozenset(("rear",))
FORWARD_CAM_IDS = frozenset(("narrow", "main", "wide"))

# BeamNGpy Camera() defaults near_far_planes=(0.05, 100) — BeamNGpy#199.
# GVD always passes an explicit pair from cameras.yaml so Tech attach is not a silent omit
# of near_far_planes (sides/rear lock at 100 m still go through Camera(..., near_far_planes=)).
DEFAULT_NEAR_M = 0.05
BNGPY_DEFAULT_FAR_M = 100.0
FORWARD_UPDATE_S = 0.067  # ~15 Hz suggestion to the Tech sensor manager
ON_DEMAND_UPDATE_S = -1.0  # no auto GPU update; ad-hoc poll only (sides/rear)
SIDE_UPDATE_S = ON_DEMAND_UPDATE_S
REAR_UPDATE_S = ON_DEMAND_UPDATE_S
MAIN_GRAB_DIV = 1
WIDE_GRAB_DIV = 2
NARROW_GRAB_DIV = 2  # start ÷2 (allowed 2–3; never ÷4)
NARROW_GRAB_PHASE = 1  # offset vs wide so they do not share a tick
WIDE_GRAB_PHASE = 0
SIDE_GRAB_DIV = 2  # poll pillar/repeat every Nth grab; never drop resolution
REAR_GRAB_DIV = 4  # poll rear every Nth grab; never drop resolution
NARROW_FAR_LIVE_HITCH_M = 400.0  # only if live camera_hz still <10 after stagger
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
NARROW_FAR_HITCH = (800.0,)  # do not attach-cut to 400; that is a live Hz hitch only
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
    """Python grab divisor. main÷1, wide÷2, narrow÷2 (2–3), sides÷2, rear÷4."""
    hitch = hitch if isinstance(hitch, dict) else {}
    if cid == "main":
        return _positive_div(hitch.get("main_grab_div"), MAIN_GRAB_DIV)
    if cid == "wide":
        return _positive_div(hitch.get("wide_grab_div"), WIDE_GRAB_DIV)
    if cid == "narrow":
        n = _positive_div(hitch.get("narrow_grab_div"), NARROW_GRAB_DIV)
        return min(3, max(2, n))  # ÷2–3 start; never ÷4
    if cid in REAR_CAM_IDS:
        return _positive_div(hitch.get("rear_grab_div"), REAR_GRAB_DIV)
    if cid in SIDE_CAM_IDS:
        return _positive_div(hitch.get("side_grab_div"), SIDE_GRAB_DIV)
    return 1


def camera_grab_phase(cid: str, hitch: dict[str, Any] | None = None) -> int:
    """Phase offset so wide (0) and narrow (1) do not grab the same tick."""
    hitch = hitch if isinstance(hitch, dict) else {}
    if cid == "narrow":
        return _nonneg_int(hitch.get("narrow_grab_phase"), NARROW_GRAB_PHASE)
    if cid == "wide":
        return _nonneg_int(hitch.get("wide_grab_phase"), WIDE_GRAB_PHASE)
    if cid == "main":
        return _nonneg_int(hitch.get("main_grab_phase"), 0)
    return 0


def grab_due(grab_i: int, div: int, phase: int = 0) -> bool:
    d = max(1, int(div))
    p = int(phase) % d
    return (int(grab_i) % d) == p


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


def far_hitch_ladder(cid: str, far_m: float) -> tuple[float, ...]:
    """Requested far, then lower rungs still inside the role band.

    Locked attach is a single rung: narrow 800, main/wide 300, sides/rear 100.
    Narrow 400 is not on the attach ladder (live camera_hz hitch only).
    Values are clamped first so a both-800 input cannot stay on the ladder.
    """
    far_m = clamp_far_m(cid, float(far_m))
    if cid == "narrow":
        rungs = NARROW_FAR_HITCH
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
    return {
        "requested_update_time": float(update_s),
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


def colour_to_bgr(colour: Any, resolution: tuple[int, int] | None = None) -> np.ndarray | None:
    """RGB(A) / BGRA bytes / array / PIL → BGR uint8. Empty → None."""
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
    return np.ascontiguousarray(rgb[:, :, ::-1])


def read_camera_colour(
    cam: Any,
    *,
    cid: str,
    resolution: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Forwards: stream_raw only (no poll). Sides/rear: poll (on-demand -1)."""
    res = resolution or _cam_resolution(cam)
    if cid in FORWARD_CAM_IDS:
        if not hasattr(cam, "stream_raw"):
            return None
        try:
            raw = cam.stream_raw()
        except Exception:
            return None
        colour = None
        if isinstance(raw, dict):
            colour = raw.get("colour") if raw.get("colour") is not None else raw.get("color")
        return colour_to_bgr(colour, res)
    # sides/rear requested_update_time=-1: need an ad-hoc poll. Never set streaming false.
    if not hasattr(cam, "poll"):
        return None
    try:
        images = cam.poll()
    except Exception:
        return None
    if not isinstance(images, dict):
        return None
    colour = images.get("colour") if images.get("colour") is not None else images.get("color")
    return colour_to_bgr(colour, res)


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
        return CameraFrameBundle(frames={}, timestamps={}, health=health, backend=self.name, note="stub: no cameras")


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
            )

        if img is None:
            health["main"] = CamHealth.MISSING
            if not self._logged:
                print(f"[GVD] window via {self._impl}; {self._note}. Sides stay missing.")
                self._logged = True
            return CameraFrameBundle(frames={}, timestamps={}, health=health, backend=self.name, note=self._note)

        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = resize_long_side(img, self.long_side)
        ts = time.time()
        frames["main"] = img
        frames["cam_main"] = img
        timestamps["main"] = ts
        health["main"] = CamHealth.OK
        # refuse fake 8-cam
        return CameraFrameBundle(frames=frames, timestamps=timestamps, health=health, backend=self.name, note=self._note)


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
        self._grab_i = 0
        self._cache_frames: dict[str, np.ndarray] = {}
        self._cache_ts: dict[str, float] = {}
        self._hitch_steps: list[tuple[str, float, float, float]] = []  # cid, near, far, update_s
        hitch = self.config.get("hitch") if isinstance(self.config.get("hitch"), dict) else {}
        self._grab_div = {cid: camera_grab_div(cid, hitch) for cid in CAM_IDS}
        self._grab_phase = {cid: camera_grab_phase(cid, hitch) for cid in CAM_IDS}
        self._side_grab_div = self._grab_div["pillarL"]
        self._rear_grab_div = self._grab_div["rear"]

    def open(self) -> None:
        try:
            from beamngpy.sensors import Camera  # type: ignore
        except Exception as e:
            if not self._logged:
                print(f"[GVD] beamngpy not available ({e}); cam_health=missing.")
                self._logged = True
            self._ok = False
            return

        if not self.session.connect(explicit=True):
            self._ok = False
            self._logged = True
            return

        # Cameras + vehicle sensors attach here, independent of Alt+G / engaged.
        # Vision LINK and cam_health ok×8 must work with engaged=false.
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
                step_s = " ".join(f"far_m={try_far:g}@update_s={try_rate:g}" for _n, try_far, try_rate in attempts)
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
                self._clip_planes[cid] = (used_near, used_far)
                self._update_s[cid] = float(used_rate)
                self._update_priority[cid] = float(used_prio)
                self._resolution[cid] = (rw, rh)
                self._hitch_steps.append((cid, float(used_near), float(used_far), float(used_rate)))
                attached += 1
            except Exception as e:
                print(f"[GVD] beamngpy Camera attach failed for {cid}: {e}")

        self._ok = attached > 0
        if not self._logged:
            if self._ok:
                clips = " ".join(f"{k}={v[1]:g}" for k, v in self._clip_planes.items())
                rates = " ".join(f"{k}={v:g}" for k, v in self._update_s.items())
                prios = " ".join(f"{k}={v:g}" for k, v in self._update_priority.items())
                print(
                    f"[GVD] beamngpy attached {attached} color Camera(s) from cameras.yaml "
                    f"(near_far_planes far_m {clips}; requested_update_time {rates}; "
                    f"update_priority {prios}; "
                    f"grab_div main={self._grab_div['main']} wide={self._grab_div['wide']} "
                    f"narrow={self._grab_div['narrow']} side={self._side_grab_div} "
                    f"rear={self._rear_grab_div}; stream_raw forwards; "
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

    def poll_vehicle(self):
        return self.session.poll()

    def close(self) -> None:
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
        self._cache_frames.clear()
        self._cache_ts.clear()
        self._grab_i = 0
        self.session.close()
        self._ok = False

    def _grab_this_tick(self, cid: str, grab_i: int) -> bool:
        div = max(1, int(self._grab_div.get(cid, 1)))
        phase = int(self._grab_phase.get(cid, 0))
        return grab_due(grab_i, div, phase)

    def _store_frame(self, cid: str, bgr: np.ndarray, ts: float, frames: dict, timestamps: dict, health: dict) -> None:
        frames[cid] = bgr
        timestamps[cid] = ts
        health[cid] = CamHealth.OK
        self._cache_frames[cid] = bgr
        self._cache_ts[cid] = ts
        if cid == "main":
            frames["cam_main"] = bgr

    def _reuse_cached(self, cid: str, frames: dict, timestamps: dict, health: dict) -> None:
        bgr = self._cache_frames.get(cid)
        if bgr is None:
            health[cid] = CamHealth.STALE
            return
        frames[cid] = bgr
        timestamps[cid] = self._cache_ts.get(cid, time.time())
        health[cid] = CamHealth.OK
        if cid == "main":
            frames["cam_main"] = bgr

    def grab(self) -> CameraFrameBundle:
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
            )
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        ts = time.time()
        grab_i = self._grab_i
        self._grab_i = grab_i + 1
        for cid, cam in self._sensors.items():
            if not self._grab_this_tick(cid, grab_i):
                self._reuse_cached(cid, frames, timestamps, health)
                continue
            try:
                bgr = read_camera_colour(cam, cid=cid, resolution=self._resolution.get(cid))
                if bgr is None:
                    if cid in self._cache_frames:
                        self._reuse_cached(cid, frames, timestamps, health)
                    else:
                        health[cid] = CamHealth.STALE
                    continue
                bgr = resize_long_side(bgr, self.long_side)
                self._store_frame(cid, bgr, ts, frames, timestamps, health)
            except Exception:
                health[cid] = CamHealth.ERROR
        n_ok = sum(1 for c in CAM_IDS if health.get(c) == CamHealth.OK)
        grab_ms = (time.perf_counter() - t0) * 1000.0
        return CameraFrameBundle(
            frames=frames,
            timestamps=timestamps,
            health=health,
            backend=self.name,
            note=(
                f"beamngpy: {n_ok} colour frame(s) grab={grab_i} "
                f"main_div={self._grab_div.get('main', 1)} "
                f"wide_div={self._grab_div.get('wide', 2)} "
                f"narrow_div={self._grab_div.get('narrow', 2)} "
                f"side_div={self._side_grab_div} rear_div={self._rear_grab_div}"
            ),
            grab_ms=grab_ms,
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
