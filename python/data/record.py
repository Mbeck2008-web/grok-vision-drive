"""M4 clips: ring-buffer cam_main + state jsonl, flush on trigger.

Encode preference: h264_qsv (UHD 630) → libx264 → (only if --encode nvenc) h264_nvenc.
Ring cap ≤2 GB. Never invent frames. Live QSV still UNPROVEN until Windows smoke.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from python.runtime.state_io import gvd_docs_dir

RING_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
DEFAULT_PRE_S = 4.0
DEFAULT_POST_S = 2.0


def clips_dir() -> Path:
    d = gvd_docs_dir() / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe_encoders(ffmpeg: str | None = None) -> set[str]:
    ff = ffmpeg or shutil.which("ffmpeg")
    if not ff:
        return set()
    try:
        out = subprocess.check_output(
            [ff, "-hide_banner", "-encoders"],
            text=True,
            timeout=5,
            stderr=subprocess.STDOUT,
        )
        found = set()
        for name in ("h264_qsv", "hevc_qsv", "libx264", "h264_nvenc"):
            if name in out:
                found.add(name)
        return found
    except Exception:
        return set()


def choose_encoder(
    *,
    prefer: str = "qsv",
    allow_nvenc: bool = False,
    available: set[str] | None = None,
) -> str:
    """Return ffmpeg encoder name. Never default nvenc."""
    avail = available if available is not None else probe_encoders()
    prefer = (prefer or "qsv").lower()
    if prefer in ("qsv", "h264_qsv") and "h264_qsv" in avail:
        return "h264_qsv"
    if prefer in ("cpu", "libx264", "x264") or "h264_qsv" not in avail:
        if "libx264" in avail:
            return "libx264"
    if allow_nvenc and "h264_nvenc" in avail:
        return "h264_nvenc"
    if "libx264" in avail:
        return "libx264"
    # last resort label for dry-run / missing ffmpeg
    return "none"


@dataclass
class RingFrame:
    t: float
    jpeg: bytes
    state: dict[str, Any]
    nbytes: int = 0

    def __post_init__(self) -> None:
        self.nbytes = len(self.jpeg) + 256


@dataclass
class ClipRecorder:
    """Rolling cam_main ring; flush to Documents/GVD/clips/ on trigger."""

    max_bytes: int = RING_CAP_BYTES
    pre_s: float = DEFAULT_PRE_S
    post_s: float = DEFAULT_POST_S
    encode_prefer: str = "qsv"
    allow_nvenc: bool = False
    hz: float = 15.0
    dry_run: bool = False
    _buf: deque[RingFrame] = field(default_factory=deque)
    _bytes: int = 0
    _last_engaged: bool = False
    _flushing_until: float | None = None
    _pending_trigger: str | None = None
    last_clip_trigger: str = "none"
    last_clip_path: str | None = None
    encoder: str = "none"

    def __post_init__(self) -> None:
        self.encoder = choose_encoder(prefer=self.encode_prefer, allow_nvenc=self.allow_nvenc)

    def _encode_jpeg(self, bgr: np.ndarray) -> bytes:
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                return bytes(buf)
        except Exception:
            pass
        # minimal PPM-ish fallback so ring still works without opencv encode
        return bgr.tobytes()[: min(200_000, bgr.nbytes)]

    def push(self, bgr: np.ndarray | None, state: dict[str, Any], now: float | None = None) -> None:
        if bgr is None:
            return
        t = now if now is not None else time.time()
        jpeg = self._encode_jpeg(bgr)
        # thin state for jsonl
        slim = {
            "t": t,
            "engaged": bool(state.get("engaged")),
            "disengage_reason": state.get("disengage_reason"),
            "path_debug_preview": state.get("path_debug_preview"),
            "planner": state.get("planner"),
            "ego": state.get("ego"),
            "detector": state.get("detector"),
            "cmd_reason": state.get("cmd_reason"),
        }
        fr = RingFrame(t=t, jpeg=jpeg, state=slim)
        self._buf.append(fr)
        self._bytes += fr.nbytes
        while self._bytes > self.max_bytes and self._buf:
            old = self._buf.popleft()
            self._bytes -= old.nbytes
        # also drop by time window ~ pre+post+1
        cutoff = t - (self.pre_s + self.post_s + 1.0)
        while self._buf and self._buf[0].t < cutoff and self._bytes > self.max_bytes // 4:
            old = self._buf.popleft()
            self._bytes -= old.nbytes

    def note_engage(self, engaged: bool) -> str | None:
        """Return trigger name if disengage edge."""
        trig = None
        if self._last_engaged and not engaged:
            trig = "disengage"
        self._last_engaged = bool(engaged)
        return trig

    def check_aeb(self, planner: dict[str, Any] | None) -> str | None:
        if not planner:
            return None
        aeb = str(planner.get("aeb") or "off")
        ttc = planner.get("ttc_lead")
        if aeb == "brake":
            return "aeb_brake"
        if ttc is not None and float(ttc) < 1.5:
            return "near_miss_ttc"
        return None

    def request(self, trigger: str, now: float | None = None) -> None:
        if self._pending_trigger:
            return  # already capturing post
        t = now if now is not None else time.time()
        self._pending_trigger = trigger
        self._flushing_until = t + self.post_s

    def maybe_flush(self, now: float | None = None) -> Path | None:
        if not self._pending_trigger or self._flushing_until is None:
            return None
        t = now if now is not None else time.time()
        if t < self._flushing_until:
            return None
        trigger = self._pending_trigger
        self._pending_trigger = None
        self._flushing_until = None
        return self.flush(trigger, now=t)

    def flush(self, trigger: str, now: float | None = None) -> Path | None:
        t = now if now is not None else time.time()
        start = t - self.pre_s - self.post_s
        frames = [f for f in self._buf if f.t >= start]
        if not frames:
            self.last_clip_trigger = trigger
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
        out = clips_dir() / f"clip_{stamp}_{trigger}"
        out.mkdir(parents=True, exist_ok=True)
        # jsonl state
        jsonl = out / "state.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for fr in frames:
                fh.write(json.dumps(fr.state) + "\n")
        # frames as jpg sequence
        frames_dir = out / "frames"
        frames_dir.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            (frames_dir / f"{i:06d}.jpg").write_bytes(fr.jpeg)
        meta = {
            "trigger": trigger,
            "n_frames": len(frames),
            "encoder": self.encoder,
            "dry_run": self.dry_run,
            "t0": frames[0].t,
            "t1": frames[-1].t,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # optional mp4 via ffmpeg
        mp4 = out / "cam_main.mp4"
        if not self.dry_run and self.encoder != "none" and shutil.which("ffmpeg"):
            self._run_ffmpeg(frames_dir, mp4, len(frames))
        else:
            meta["encode"] = "skipped"
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.last_clip_trigger = trigger
        self.last_clip_path = str(out)
        return out

    def _run_ffmpeg(self, frames_dir: Path, mp4: Path, n: int) -> None:
        ff = shutil.which("ffmpeg")
        if not ff or n <= 0:
            return
        pattern = str(frames_dir / "%06d.jpg")
        fps = max(5.0, min(30.0, self.hz))
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", pattern]
        enc = self.encoder
        if enc == "h264_qsv":
            cmd += ["-c:v", "h264_qsv", "-global_quality", "23"]
        elif enc == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-cq", "23"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        cmd += ["-pix_fmt", "yuv420p", str(mp4)]
        try:
            subprocess.run(cmd, check=False, timeout=120)
        except Exception:
            pass

    @property
    def ring_bytes(self) -> int:
        return self._bytes
