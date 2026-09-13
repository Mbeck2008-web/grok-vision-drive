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
        self._logged = False
        self._ok = False

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
        update_s = float(cam_cfg.get("update_s") or 0.067)
        shmem = bool(cam_cfg.get("shared_memory", True))
        streaming = bool(cam_cfg.get("streaming", True))
        rgb_only = bool(cam_cfg.get("rgb_only", True))
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
                cam = Camera(
                    f"gvd_{cid}",
                    bng,
                    vehicle,
                    requested_update_time=update_s,
                    pos=pos_bng,
                    dir=dir_bng,
                    up=up_bng,
                    field_of_view_y=fov_v,
                    resolution=(rw, rh),
                    is_using_shared_memory=shmem,
                    is_streaming=streaming,
                    is_render_colours=True,
                    is_render_annotations=not rgb_only,
                    is_render_instance=False,
                    is_render_depth=not rgb_only,
                    is_snapping_desired=False,
                    is_visualised=False,
                )
                self._sensors[cid] = cam
                attached += 1
            except Exception as e:
                print(f"[GVD] beamngpy Camera attach failed for {cid}: {e}")

        self._ok = attached > 0
        if not self._logged:
            if self._ok:
                print(
                    f"[GVD] beamngpy attached {attached} color Camera(s) from cameras.yaml "
                    f"(GVD→BeamNG vehicle-space convert; depth/semantic OFF)."
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
        self.session.close()
        self._ok = False

    def grab(self) -> CameraFrameBundle:
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        if not self._ok or not self._sensors:
            return CameraFrameBundle(
                frames={},
                timestamps={},
                health=health,
                backend=self.name,
                note="beamngpy: no live Camera session",
            )
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        ts = time.time()
        for cid, cam in self._sensors.items():
            try:
                images = None
                if hasattr(cam, "stream") and getattr(cam, "is_streaming", False):
                    try:
                        images = cam.stream()
                    except Exception:
                        images = cam.poll() if hasattr(cam, "poll") else None
                elif hasattr(cam, "poll"):
                    images = cam.poll()
                colour = None
                if isinstance(images, dict):
                    colour = images.get("colour") or images.get("color")
                if colour is None:
                    health[cid] = CamHealth.STALE
                    continue
                # PIL Image → BGR
                arr = np.asarray(colour)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    # RGB(A) → BGR
                    bgr = arr[:, :, :3][:, :, ::-1].copy()
                else:
                    health[cid] = CamHealth.ERROR
                    continue
                bgr = resize_long_side(bgr, self.long_side)
                frames[cid] = bgr
                timestamps[cid] = ts
                health[cid] = CamHealth.OK
                if cid == "main":
                    frames["cam_main"] = bgr
            except Exception:
                health[cid] = CamHealth.ERROR
        return CameraFrameBundle(
            frames=frames,
            timestamps=timestamps,
            health=health,
            backend=self.name,
            note=f"beamngpy: {len(frames)} colour frame(s)",
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
