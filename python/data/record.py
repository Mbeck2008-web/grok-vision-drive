"""M4 clips: ring-buffer cam_main + state jsonl, flush on trigger.

Encode preference: h264_qsv (UHD 630) → libx264 → (only if --encode nvenc) h264_nvenc.
Ring cap ≤2 GB. Atomic flush via temp dir + rename. Never invent frames.
Live QSV still UNPROVEN until Windows smoke.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
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
TRIGGER_COOLDOWN_S = 8.0


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
    """Return ffmpeg encoder name. Never default nvenc unless allow_nvenc=True.

    Priority:
      - prefer qsv/auto: h264_qsv → (if allow_nvenc) h264_nvenc → libx264
      - prefer cpu: libx264 → (if allow_nvenc) h264_nvenc
      - prefer nvenc: h264_nvenc (requires allow_nvenc) → libx264
    """
    avail = available if available is not None else probe_encoders()
    prefer = (prefer or "qsv").lower()

    if prefer in ("nvenc", "h264_nvenc"):
        # Explicit --encode nvenc: still prefer QSV when present (don't steal Pascal if QSV works).
        # When QSV missing, h264_nvenc MUST win over silent libx264.
        if "h264_qsv" in avail:
            return "h264_qsv"
        if allow_nvenc and "h264_nvenc" in avail:
            return "h264_nvenc"
        if "libx264" in avail:
            return "libx264"
        return "none"

    if prefer in ("cpu", "libx264", "x264"):
        if "libx264" in avail:
            return "libx264"
        if allow_nvenc and "h264_nvenc" in avail:
            return "h264_nvenc"
        return "none"

    # qsv / auto
    if "h264_qsv" in avail:
        return "h264_qsv"
    if allow_nvenc and "h264_nvenc" in avail:
        return "h264_nvenc"
    if "libx264" in avail:
        return "libx264"
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
    _last_flush_t: float = 0.0
    _aeb_latched: bool = False
    _ttc_latched: bool = False
    last_clip_trigger: str = "none"
    last_clip_path: str | None = None
    last_encode_status: str = "none"
    encoder: str = "none"

    def __post_init__(self) -> None:
        self.encoder = choose_encoder(prefer=self.encode_prefer, allow_nvenc=self.allow_nvenc)

    def _encode_jpeg(self, bgr: np.ndarray) -> bytes | None:
        """Honest JPEG only — never write raw BGR bytes as .jpg."""
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                return bytes(buf)
        except Exception:
            pass
        return None

    def _trim(self) -> None:
        while self._bytes > self.max_bytes and self._buf:
            old = self._buf.popleft()
            self._bytes -= old.nbytes
        # one-frame slack only: if still over after drop, drop more
        while self._buf and self._bytes > self.max_bytes:
            old = self._buf.popleft()
            self._bytes -= old.nbytes

    def push(self, bgr: np.ndarray | None, state: dict[str, Any], now: float | None = None) -> bool:
        """Push frame if JPEG encode succeeds. Returns False if skipped (honesty)."""
        if bgr is None:
            return False
        t = now if now is not None else time.time()
        jpeg = self._encode_jpeg(bgr)
        if jpeg is None:
            return False
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
        self._trim()
        cutoff = t - (self.pre_s + self.post_s + 1.0)
        while self._buf and self._buf[0].t < cutoff and self._bytes > self.max_bytes // 2:
            old = self._buf.popleft()
            self._bytes -= old.nbytes
        return True

    def note_engage(self, engaged: bool) -> str | None:
        trig = None
        if self._last_engaged and not engaged:
            trig = "disengage"
        self._last_engaged = bool(engaged)
        return trig

    def check_aeb(self, planner: dict[str, Any] | None) -> str | None:
        """Edge-trigger AEB / near-miss (cooldown + latch — no clip flood)."""
        if not planner:
            self._aeb_latched = False
            self._ttc_latched = False
            return None
        aeb = str(planner.get("aeb") or "off")
        ttc = planner.get("ttc_lead")
        trig = None
        if aeb == "brake":
            if not self._aeb_latched:
                trig = "aeb_brake"
                self._aeb_latched = True
        else:
            self._aeb_latched = False
        if ttc is not None and float(ttc) < 1.5:
            if not self._ttc_latched and trig is None:
                trig = "near_miss_ttc"
                self._ttc_latched = True
        else:
            self._ttc_latched = False
        return trig

    def request(self, trigger: str, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        if self._pending_trigger:
            return
        if (t - self._last_flush_t) < TRIGGER_COOLDOWN_S and self._last_flush_t > 0:
            return
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
        final = clips_dir() / f"clip_{stamp}_{trigger}"
        # Atomic: build in temp, then rename into clips/
        tmp_root = Path(tempfile.mkdtemp(prefix="gvd_clip_", dir=str(clips_dir())))
        try:
            jsonl = tmp_root / "state.jsonl"
            with jsonl.open("w", encoding="utf-8") as fh:
                for fr in frames:
                    fh.write(json.dumps(fr.state) + "\n")
            frames_dir = tmp_root / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, fr in enumerate(frames):
                (frames_dir / f"{i:06d}.jpg").write_bytes(fr.jpeg)
            meta: dict[str, Any] = {
                "trigger": trigger,
                "n_frames": len(frames),
                "encoder": self.encoder,
                "dry_run": self.dry_run,
                "t0": frames[0].t,
                "t1": frames[-1].t,
                "encode": "pending",
            }
            (tmp_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mp4 = tmp_root / "cam_main.mp4"
            if not self.dry_run and self.encoder != "none" and shutil.which("ffmpeg"):
                ok = self._run_ffmpeg(frames_dir, mp4, len(frames))
                if ok and mp4.is_file() and mp4.stat().st_size > 0:
                    meta["encode"] = "ok"
                else:
                    meta["encode"] = "failed"
                    print(f"[GVD] clip encode FAILED encoder={self.encoder} trigger={trigger}")
            else:
                meta["encode"] = "skipped"
            (tmp_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            tmp_root.rename(final)
        except Exception:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise
        self.last_encode_status = str(meta.get("encode"))
        self.last_clip_trigger = trigger
        self.last_clip_path = str(final)
        self._last_flush_t = t
        return final

    def _run_ffmpeg(self, frames_dir: Path, mp4: Path, n: int) -> bool:
        ff = shutil.which("ffmpeg")
        if not ff or n <= 0:
            return False
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
            r = subprocess.run(cmd, check=False, timeout=120, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[:300]
                print(f"[GVD] ffmpeg rc={r.returncode}: {err}")
                return False
            return mp4.is_file() and mp4.stat().st_size > 0
        except Exception as e:
            print(f"[GVD] ffmpeg exception: {type(e).__name__}: {e}")
            return False

    @property
    def ring_bytes(self) -> int:
        return self._bytes
