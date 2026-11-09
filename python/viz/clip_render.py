"""Hold each real clip tick for its logged duration; do not invent poses."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def held_repeats(
    timestamps: list[float],
    *,
    fps: float = 30.0,
    max_gap_s: float = 0.5,
) -> list[int]:
    """How many output frames to hold each tick.

    Frame i lasts until the next timestamp. A gap longer than ``max_gap_s``
    is a cut: that interval is not filled. The last tick, and a tick before
    a cut, keep the previous in-shot period so the video does not invent
    motion between real samples.
    """
    if not timestamps:
        return []
    rate = float(fps)
    if rate <= 0.0:
        raise ValueError("fps must be positive")
    ts = [float(t) for t in timestamps]
    out: list[int] = []
    prev_dt = 1.0 / rate
    for i, t in enumerate(ts):
        if i + 1 < len(ts):
            dt = ts[i + 1] - t
            if not math.isfinite(dt) or dt < 0.0:
                dt = prev_dt
            elif dt > float(max_gap_s):
                dt = prev_dt
            elif dt > 1e-9:
                prev_dt = dt
        else:
            dt = prev_dt
        out.append(max(1, int(round(dt * rate))))
    return out


def load_clip_ticks(folder: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Pair ``state.jsonl`` rows with ``frames/NNNNNN.jpg`` (or matching names)."""
    root = Path(folder)
    jsonl = root / "state.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"missing {jsonl}")
    rows: list[dict[str, Any]] = []
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    frames_dir = root / "frames"
    files: list[Path] = []
    if frames_dir.is_dir():
        files = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    if len(files) < len(rows):
        extras = sorted(p for p in root.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(extras) >= len(rows):
            files = extras
    if len(files) != len(rows):
        raise ValueError(f"{root}: {len(rows)} state rows vs {len(files)} frames")
    return rows, files


def timestamps_of(rows: list[dict[str, Any]]) -> list[float]:
    ts: list[float] = []
    for i, row in enumerate(rows):
        raw = row.get("t")
        if raw is None:
            raw = row.get("time")
        if raw is None:
            ts.append(float(i) * 0.1)
            continue
        ts.append(float(raw))
    return ts


def ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def encode_held_mp4(
    frame_paths: list[Path],
    repeats: list[int],
    dest: Path,
    *,
    fps: float = 30.0,
) -> Path:
    """Write a libx264 yuv420p mp4 by repeating each source frame ``repeats[i]`` times."""
    if len(frame_paths) != len(repeats):
        raise ValueError("frame_paths and repeats must align")
    if not frame_paths:
        raise ValueError("no frames to encode")
    exe = ffmpeg_bin()
    if not exe:
        raise RuntimeError("ffmpeg not found (install it or imageio-ffmpeg)")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(frame_paths[0]).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="gvd_clip_seq_") as tmp:
        seq = Path(tmp)
        k = 0
        for src, n in zip(frame_paths, repeats):
            src = Path(src)
            for _ in range(int(n)):
                shutil.copyfile(src, seq / f"frame_{k:05d}{suffix}")
                k += 1
        cmd = [
            exe,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(seq / f"frame_%05d{suffix}"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(dest),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "")[-2000:]
            raise RuntimeError(f"ffmpeg failed: {err}") from exc
    if not dest.is_file() or dest.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg wrote no file at {dest}")
    return dest


def count_mp4_frames(path: Path) -> int:
    """Decode frame count with OpenCV (no ffprobe required)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def render_clip_folder(folder: Path, dest: Path, *, fps: float = 30.0) -> dict[str, Any]:
    """Hold each real tick of a clip folder and write ``dest`` (libx264 mp4)."""
    rows, files = load_clip_ticks(Path(folder))
    repeats = held_repeats(timestamps_of(rows), fps=fps)
    encode_held_mp4(files, repeats, Path(dest), fps=fps)
    return {"n_src": len(files), "n_out": int(sum(repeats)), "repeats": repeats, "path": str(dest)}
