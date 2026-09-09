"""Camera backends for GVD — vision-only RGB. Never fake 8 cams from one window."""

from __future__ import annotations

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
    import json

    root = Path(__file__).resolve().parents[2]
    cfg_path = path or (root / "config" / "cameras.yaml")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        # Minimal fallback without PyYAML: only need ids for health keys
        return {"cameras": [{"id": c} for c in CAM_IDS]}


def load_live_long_side(default: int = 640) -> int:
    root = Path(__file__).resolve().parents[2]
    hw = root / "config" / "hardware.yaml"
    try:
        text = hw.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("live_input_long_side:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return default


@runtime_checkable
class CameraBackend(Protocol):
    name: str

    def open(self) -> None: ...
    def close(self) -> None: ...
    def grab(self) -> CameraFrameBundle: ...


class StubBackend:
    """Offline / CI: no frames. Health stays missing. State/ribbon still work."""

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


class WindowBackend:
    """Retail: single main BeamNG window capture. Never clones into 8 cams."""

    name = "window"

    def __init__(self, long_side: int | None = None) -> None:
        self.long_side = long_side or load_live_long_side()
        self._cam = None
        self._mss = None
        self._impl = "none"
        self._logged = False

    def open(self) -> None:
        # Prefer bettercam with numpy path (no NVIDIA capture path stealing 1080 Ti)
        try:
            import bettercam  # type: ignore

            self._cam = bettercam.create(output_color="BGR", nvidia_gpu=False)
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

    def close(self) -> None:
        self._cam = None
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None

    def grab(self) -> CameraFrameBundle:
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        note = "retail: 1 window"
        img = None
        try:
            if self._cam is not None:
                img = self._cam.grab()
                if img is not None and not isinstance(img, np.ndarray):
                    img = np.asarray(img)
            elif self._mss is not None:
                mon = self._mss.monitors[1] if len(self._mss.monitors) > 1 else self._mss.monitors[0]
                shot = self._mss.grab(mon)
                img = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        except Exception as e:
            if not self._logged:
                print(f"[GVD] window capture error: {e}")
                self._logged = True
            health["main"] = CamHealth.ERROR
            return CameraFrameBundle(frames={}, timestamps={}, health=health, backend=self.name, note=note)

        if img is None:
            health["main"] = CamHealth.MISSING
            if not self._logged:
                print(f"[GVD] window backend open via {self._impl}; no frame yet (retail: 1 window only).")
                self._logged = True
            return CameraFrameBundle(frames={}, timestamps={}, health=health, backend=self.name, note=note)

        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = resize_long_side(img, self.long_side)
        ts = time.time()
        frames["main"] = img
        frames["cam_main"] = img
        timestamps["main"] = ts
        health["main"] = CamHealth.OK
        # all others stay missing — refuse fake 8-cam
        return CameraFrameBundle(frames=frames, timestamps=timestamps, health=health, backend=self.name, note=note)


class BeamNGPyBackend:
    """Tech path: BeamNGpy Camera sensors, color only. No depth/semantic in default."""

    name = "beamngpy"

    def __init__(self, long_side: int | None = None, config: dict[str, Any] | None = None) -> None:
        self.long_side = long_side or load_live_long_side()
        self.config = config or load_camera_config()
        self._sensors: dict[str, Any] = {}
        self._vehicle = None
        self._bng = None
        self._logged = False
        self._ok = False

    def open(self) -> None:
        try:
            from beamngpy import BeamNGpy, Vehicle  # type: ignore
            from beamngpy.sensors import Camera  # type: ignore
        except Exception as e:
            if not self._logged:
                print(f"[GVD] beamngpy not available ({e}); cam_health=missing. Install BeamNG.tech + beamngpy for 8-cam.")
                self._logged = True
            self._ok = False
            return

        # Connection is environment-specific; without a live Tech server we stay missing.
        # Do not invent frames.
        try:
            # Attempt attach only if BEAMNG_HOME / existing research connection conventions exist.
            # Conservative: require GVD_BEAMNG=1 to even try connecting.
            import os

            if os.environ.get("GVD_BEAMNG", "").strip() not in ("1", "true", "yes"):
                if not self._logged:
                    print(
                        "[GVD] beamngpy importable but GVD_BEAMNG not set; "
                        "cam_health=missing (no fake 8-cam). Set GVD_BEAMNG=1 when Tech is running."
                    )
                    self._logged = True
                self._ok = False
                return
            # Live attach is Windows/Tech-specific — leave sensors empty until a session is wired in M1.1+.
            # Document UNPROVEN on Linux.
            if not self._logged:
                print(
                    "[GVD] beamngpy path selected; live Camera attach is UNPROVEN on Linux. "
                    "Color-only mounts are defined in config/cameras.yaml."
                )
                self._logged = True
            self._ok = False
        except Exception as e:
            if not self._logged:
                print(f"[GVD] beamngpy Camera attach failed: {e}; cam_health=missing.")
                self._logged = True
            self._ok = False

    def close(self) -> None:
        self._sensors.clear()
        self._vehicle = None
        self._bng = None

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
        # Placeholder for when sensors are attached in a live Tech session
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        return CameraFrameBundle(frames=frames, timestamps=timestamps, health=health, backend=self.name, note="beamngpy")


def resolve_backend_name(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import beamngpy  # noqa: F401

        return "beamngpy"
    except Exception:
        pass
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
