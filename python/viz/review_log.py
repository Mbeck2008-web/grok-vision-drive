"""Supervisor review capture. Key R, or the VIZ-tab "review capture" row.

While it is on, each tick writes one folder under the GVD documents bus:

    <gvd_docs_dir>/review/<UTC stamp>/
        tick_000001.json
        tick_000001_<camera>.jpg

The json names the frames for that tick. Writing the folder does not close
or stop BeamNG.tech. Turning the capture off only stops new files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from python.runtime.state_io import gvd_docs_dir


def review_root() -> Path:
    """Parent of each capture session: ``Documents/GVD/review``."""
    d = gvd_docs_dir() / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ReviewLog:
    """One capture session. ``root`` is the session folder itself."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tick = 0

    @classmethod
    def open(cls, parent: Path | None = None) -> ReviewLog:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        base = Path(parent) if parent is not None else review_root()
        folder = base / stamp
        n = 2
        while folder.exists():
            folder = base / f"{stamp}_{n}"
            n += 1
        return cls(folder)

    def write_tick(
        self,
        *,
        lanes: list[dict[str, Any]],
        path: list[dict[str, Any]] | None,
        steer: float,
        throttle: float,
        glance: str,
        disengage_reason: str,
        frames: dict[str, Any] | None = None,
        brake: float = 0.0,
        engaged: bool = False,
        cmd_reason: str = "",
    ) -> Path:
        """Write this tick's json and any camera frames. Returns the json path."""
        self.tick += 1
        stem = f"tick_{self.tick:06d}"
        frame_names: list[str] = []
        for cid, img in (frames or {}).items():
            name = self._write_frame(stem, str(cid), img)
            if name:
                frame_names.append(name)
        payload = {
            "tick": self.tick,
            "stem": stem,
            "glance": str(glance or ""),
            "engaged": bool(engaged),
            "disengage_reason": str(disengage_reason or "none"),
            "cmd_reason": str(cmd_reason or ""),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "lanes": [_lane_record(ln) for ln in lanes or []],
            "path": [_point(p) for p in path or []],
            "frames": frame_names,
        }
        dest = self.root / f"{stem}.json"
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest

    def _write_frame(self, stem: str, cid: str, img: Any) -> str | None:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return None
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in cid) or "cam"
        name = f"{stem}_{safe}.jpg"
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            (self.root / name).write_bytes(bytes(buf))
        except Exception:
            return None
        return name


def _point(p: Any) -> dict[str, float]:
    if isinstance(p, dict):
        return {
            "x": float(p.get("x") or 0.0),
            "y": float(p.get("y") or 0.0),
            "z": float(p.get("z") or 0.0),
        }
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}


def _lane_record(ln: dict[str, Any]) -> dict[str, Any]:
    pts = []
    for p in ln.get("points") or []:
        if isinstance(p, dict):
            pts.append({"x": float(p.get("x") or 0.0), "y": float(p.get("y") or 0.0)})
    return {
        "role": str(ln.get("role") or "through"),
        "kind": ln.get("kind"),
        "side": ln.get("side"),
        "index": ln.get("index"),
        "points": pts,
    }


class ReviewCapture:
    """Owns the open session for the supervisor loop. Does not touch BeamNG."""

    def __init__(self) -> None:
        self.log: ReviewLog | None = None
        self._warned = False

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.log = None

    def write(self, enabled: bool, **payload: Any) -> Path | None:
        if not enabled:
            self.log = None
            return None
        try:
            if self.log is None:
                self.log = ReviewLog.open()
                print(f"[GVD] review capture -> {self.log.root}", flush=True)
            return self.log.write_tick(**payload)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"[GVD] review capture failed: {type(exc).__name__}: {exc}", flush=True)
            return None
