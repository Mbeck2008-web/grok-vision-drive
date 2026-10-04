"""Continuous 360° strip recorder for scene-net training.

One tick is one JPEG. The canvas is 2272×192: six 256×192 sectors
(repeatL, pillarL, main, narrow, pillarR, repeatR), two 340×192 sectors
(wide, rear), and seven 8 px gaps. A camera that missed the tick is an
empty sector. The previous picture is not reused.

Every same-tick strip is written as it arrives. ``state.jsonl`` stores the
real timestamp and the seconds since the previous strip. Nothing in this
file resamples, drops, or duplicates a strip to hold a fixed rate.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Left to right, same order as the live strip. Widths are the training canvas,
# not the live 160×96 preview.
STRIP_ORDER: tuple[str, ...] = (
    "repeatL",
    "pillarL",
    "wide",
    "main",
    "narrow",
    "pillarR",
    "repeatR",
    "rear",
)
WIDE_CAMS = frozenset({"wide", "rear"})
CANVAS_H = 192
GAP_PX = 8
SECTOR_W_43 = 256
SECTOR_W_169 = 340
# Inboard repeater columns are the ego body, same fraction as the live strip.
EGO_EDGE_FRAC = 0.25
EGO_EDGE = {"repeatL": "left", "repeatR": "right"}
JPEG_QUALITY = 95


class StripTimestampError(ValueError):
    """Pictures in one strip do not share the strip timestamp."""


@dataclass(frozen=True)
class SectorGeom:
    cam_id: str
    x0: int
    x1: int
    width: int
    aspect: str

    @property
    def wide(self) -> bool:
        return self.aspect == "16:9"


def sector_width(cam_id: str) -> int:
    return SECTOR_W_169 if cam_id in WIDE_CAMS else SECTOR_W_43


def build_sectors() -> tuple[SectorGeom, ...]:
    x = 0
    out: list[SectorGeom] = []
    for i, cam_id in enumerate(STRIP_ORDER):
        width = sector_width(cam_id)
        aspect = "16:9" if cam_id in WIDE_CAMS else "4:3"
        out.append(SectorGeom(cam_id, x, x + width, width, aspect))
        x += width
        if i + 1 != len(STRIP_ORDER):
            x += GAP_PX
    expect = len(WIDE_CAMS) * SECTOR_W_169
    expect += (len(STRIP_ORDER) - len(WIDE_CAMS)) * SECTOR_W_43
    expect += (len(STRIP_ORDER) - 1) * GAP_PX
    if x != expect:
        raise RuntimeError(f"strip canvas width is {x}, expected {expect}")
    return tuple(out)


SECTORS: tuple[SectorGeom, ...] = build_sectors()
CANVAS_W = SECTORS[-1].x1


@dataclass(frozen=True)
class StripRecord:
    index: int
    path: Path
    t: float
    dt_s: float | None


def _picture(img: Any) -> np.ndarray | None:
    if not isinstance(img, np.ndarray) or img.ndim < 2 or img.size == 0:
        return None
    return img


def _fit(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """BGR uint8 sector. A copy, so a caller's buffer stays intact."""
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[-1] > 3:
        arr = np.ascontiguousarray(arr[:, :, :3])
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] == height and arr.shape[1] == width:
        return np.array(arr, copy=True)
    import cv2

    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)


def _clear_ego(cam_id: str, placed: np.ndarray) -> np.ndarray:
    edge = EGO_EDGE.get(cam_id)
    if edge is None:
        return placed
    width = int(placed.shape[1])
    n = min(width, max(1, int(round(width * EGO_EDGE_FRAC))))
    if edge == "left":
        placed[:, :n] = 0
    else:
        placed[:, width - n :] = 0
    return placed


def compose_strip(frames: Mapping[str, Any] | None) -> np.ndarray:
    """Paint this call's pictures onto a fresh canvas.

    A missing camera stays black. No sector is filled from a previous call.
    """
    src = frames or {}
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    for sec in SECTORS:
        img = _picture(src.get(sec.cam_id))
        if img is None and sec.cam_id == "main":
            img = _picture(src.get("cam_main"))
        if img is None:
            continue
        placed = _clear_ego(sec.cam_id, _fit(img, sec.width, CANVAS_H))
        canvas[:, sec.x0 : sec.x1] = placed
    return canvas


def encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _require_same_tick(
    frames: Mapping[str, Any],
    t: float,
    timestamps: Mapping[str, float] | None,
) -> None:
    """Refuse the strip when any present picture has a different timestamp."""
    if timestamps is None:
        if not math.isfinite(float(t)):
            raise StripTimestampError("strip time is not finite")
        return
    tick = float(t)
    if not math.isfinite(tick):
        raise StripTimestampError("strip time is not finite")
    for cam_id in STRIP_ORDER:
        img = _picture(frames.get(cam_id))
        if img is None and cam_id == "main":
            img = _picture(frames.get("cam_main"))
        if img is None:
            continue
        if cam_id not in timestamps and not (cam_id == "main" and "cam_main" in timestamps):
            raise StripTimestampError(f"{cam_id} has a picture and no timestamp")
        raw = timestamps[cam_id] if cam_id in timestamps else timestamps["cam_main"]
        stamp = float(raw)
        if not math.isfinite(stamp) or stamp != tick:
            raise StripTimestampError(
                f"{cam_id} timestamp {stamp} differs from strip time {tick}"
            )


def read_state_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


_SESSION_SEQ = 0


def _new_session_id() -> str:
    global _SESSION_SEQ
    _SESSION_SEQ += 1
    return f"s{_SESSION_SEQ:04d}-{time.time_ns()}"


def _index_in_name(name: str) -> int | None:
    head = name[1:] if name.startswith(".") else name
    head = head.split(".", 1)[0]
    if head.isdigit():
        return int(head)
    return None


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("strip write made no progress")
        view = view[wrote:]


class StripWriter:
    """One session in a directory: ``meta.json``, ``strips/NNNNNN.jpg``, ``state.jsonl``.

    A new session takes the next free index, so it never reuses a filename.
    ``state.jsonl`` is opened append-only. This class does not delete a strip,
    a state line, or a sibling file.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.strips = self.root / "strips"
        self.strips.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.jsonl"
        self._session = _new_session_id()
        self._n = self._next_index()
        # A new session has no previous strip. dt_s on its first line is null.
        self._prev_t: float | None = None
        self._write_meta()

    def _next_index(self) -> int:
        used: set[int] = set()
        for row in read_state_jsonl(self.state_path):
            raw = row.get("i")
            if isinstance(raw, bool):
                continue
            try:
                used.add(int(raw))
            except (TypeError, ValueError):
                continue
        if self.strips.is_dir():
            for path in self.strips.iterdir():
                if not path.is_file():
                    continue
                idx = _index_in_name(path.name)
                if idx is not None:
                    used.add(idx)
        n = 0
        while n in used:
            n += 1
        return n

    def _write_meta(self) -> None:
        sectors = [
            {
                "id": sec.cam_id,
                "x0": sec.x0,
                "x1": sec.x1,
                "w": sec.width,
                "h": CANVAS_H,
                "aspect": sec.aspect,
            }
            for sec in SECTORS
        ]
        meta = {
            "kind": "gvd_scene_strips",
            "canvas": [CANVAS_W, CANVAS_H],
            "canvas_w": CANVAS_W,
            "canvas_h": CANVAS_H,
            "gap_px": GAP_PX,
            "order": list(STRIP_ORDER),
            "sectors": sectors,
            "n_strips": len(read_state_jsonl(self.state_path)),
            "fixed_hz": None,
            "note": (
                "One JPEG per tick, written as it arrives. "
                "dt_s in state.jsonl is the seconds since the previous strip. "
                "A missing camera is an empty sector."
            ),
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def write(
        self,
        frames: Mapping[str, Any] | None,
        *,
        t: float,
        timestamps: Mapping[str, float] | None = None,
        ego: Mapping[str, Any] | None = None,
        wheel: Mapping[str, Any] | None = None,
        pedals: Mapping[str, Any] | None = None,
        tech: Mapping[str, Any] | None = None,
    ) -> StripRecord:
        """Write one strip. Raises ``StripTimestampError`` before any file change."""
        src: Mapping[str, Any] = frames or {}
        _require_same_tick(src, t, timestamps)
        t_store = round(float(t), 6)
        dt_s = None if self._prev_t is None else round(t_store - self._prev_t, 6)
        canvas = compose_strip(src)
        idx, final = self._create_exclusive(encode_jpeg(canvas))
        line = {
            "i": idx,
            "t": t_store,
            "dt_s": dt_s,
            "session": self._session,
            "ego": _jsonable(dict(ego or {})),
            "wheel": _jsonable(dict(wheel or {})),
            "pedals": _jsonable(dict(pedals or {})),
        }
        if tech is not None:
            line["tech"] = _jsonable(dict(tech))
        try:
            self._append_state(line)
        except Exception:
            # The jpeg stays. Crash recovery and a failed state line do not delete it.
            self._n = idx + 1
            raise
        self._n = idx + 1
        self._prev_t = t_store
        self._write_meta()
        return StripRecord(index=idx, path=final, t=t_store, dt_s=dt_s)

    def _create_exclusive(self, payload: bytes) -> tuple[int, Path]:
        """Create a new JPEG. An existing name is left untouched."""
        for _ in range(1_000_000):
            idx = self._n
            final = self.strips / f"{idx:06d}.jpg"
            if final.exists():
                self._n = idx + 1
                continue
            try:
                fd = os.open(str(final), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self._n = idx + 1
                continue
            try:
                _write_all(fd, payload)
            except Exception:
                os.close(fd)
                self._n = idx + 1
                raise
            os.close(fd)
            return idx, final
        raise RuntimeError("no free strip filename")

    def _append_state(self, line: dict[str, Any]) -> None:
        """Append one JSON line. ``O_APPEND`` does not truncate the file."""
        blob = (json.dumps(line) + "\n").encode("utf-8")
        fd = os.open(str(self.state_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        try:
            _write_all(fd, blob)
            os.fsync(fd)
        finally:
            os.close(fd)


def _finite_stamp(value: Any) -> float | None:
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp):
        return None
    return stamp


def same_tick_frames(
    frames: Mapping[str, Any] | None,
    timestamps: Mapping[str, float] | None,
    t: float,
) -> tuple[dict[str, Any], dict[str, float] | None, float]:
    """Keep pictures that share one tick. A different stamp is left out.

    The writer still refuses a mixed set. Callers that have a live bundle
    use this so a stale camera becomes an empty sector instead of a reused frame.
    """
    src = dict(frames or {})
    if not timestamps:
        tick = float(t)
        return src, None, tick
    stamps: dict[str, float] = {}
    for key, raw in timestamps.items():
        stamp = _finite_stamp(raw)
        if stamp is not None:
            stamps[str(key)] = stamp
    tick: float | None = None
    for key in ("main", "cam_main"):
        if key in stamps and _picture(src.get(key)) is not None:
            tick = stamps[key]
            break
    if tick is None and stamps:
        tick = max(stamps.values())
    if tick is None:
        tick = float(t)
    kept: dict[str, Any] = {}
    kept_stamps: dict[str, float] = {}
    for cam_id, img in src.items():
        if _picture(img) is None:
            continue
        keys = [str(cam_id)]
        if cam_id == "main":
            keys.append("cam_main")
        elif cam_id == "cam_main":
            keys.append("main")
        stamp = None
        for key in keys:
            if key in stamps:
                stamp = stamps[key]
                break
        if stamp is None or stamp != tick:
            continue
        kept[str(cam_id)] = img
        kept_stamps[str(cam_id)] = tick
    return kept, kept_stamps, tick


def _training_files(root: Path) -> list[Path]:
    """Strips, state lines, and the strip meta. Nothing else in the folder."""
    found: list[Path] = []
    for name in ("state.jsonl", "meta.json"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            found.append(path)
    strips = root / "strips"
    if strips.is_dir() and not strips.is_symlink():
        for path in strips.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            if name.endswith(".jpg") or name.endswith(".jpg.tmp") or name.endswith(".partial"):
                found.append(path)
    return found


def wipe_training(root: Path | str, *, confirm: bool = False) -> list[Path]:
    """Delete training files in ``root`` only when ``confirm`` is True.

    Leaves notes, other folders, and files that are not strip training data.
    """
    if confirm is not True:
        return []
    base = Path(root)
    if not base.is_dir():
        return []
    deleted: list[Path] = []
    try:
        base_resolved = base.resolve()
    except OSError:
        return []
    for path in _training_files(base):
        try:
            path.resolve().relative_to(base_resolved)
        except (OSError, ValueError):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        path.unlink()
        deleted.append(path)
    return deleted


def default_strip_prefs_path() -> Path:
    from python.runtime.paths import gvd_docs_dir

    return gvd_docs_dir() / "gvd_strip_record.json"


def default_strip_destination() -> str:
    from python.runtime.paths import gvd_docs_dir

    return str(gvd_docs_dir() / "scene_strips")


def _read_destination(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("destination") or "").strip()


def _save_destination(path: Path, destination: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"destination": destination}) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


class StripRecordControl:
    """Start and stop strip recording. Engage does not change that flag.

    The destination is remembered outside the training folder. Stop, restart,
    and a destination change do not delete training files. ``arm_free_space``
    only arms a wipe. ``confirm_free_space`` deletes after that arm.
    """

    def __init__(self, prefs_path: Path | str | None = None, destination: str | None = None) -> None:
        self.prefs_path = Path(prefs_path) if prefs_path is not None else default_strip_prefs_path()
        self.recording = False
        self.engaged = False
        self.free_space_armed = False
        self._armed_destination = ""
        self._writer: StripWriter | None = None
        saved = _read_destination(self.prefs_path)
        if destination is not None:
            self.destination = str(destination).strip()
        elif saved:
            self.destination = saved
        else:
            self.destination = default_strip_destination()

    def status_text(self) -> str:
        if self.recording:
            return "recording on"
        return "recording off"

    def note_engaged(self, engaged: bool) -> None:
        """Remember the car flag. Does not start or stop recording."""
        self.engaged = bool(engaged)

    def start(self) -> None:
        if self.recording:
            return
        if not str(self.destination).strip():
            return
        self._save()
        self._writer = StripWriter(self.destination)
        self.recording = True

    def stop(self) -> None:
        self.recording = False
        self._writer = None

    def set_destination(self, path: str) -> None:
        """Point new files at ``path``. The previous folder is left in place."""
        text = str(path).strip()
        if not text:
            return
        if text == self.destination:
            self._save()
            return
        self.destination = text
        self._writer = None
        if self.free_space_armed:
            self.free_space_armed = False
            self._armed_destination = ""
        self._save()
        if self.recording:
            self._writer = StripWriter(self.destination)

    def offer(
        self,
        frames: Mapping[str, Any] | None,
        *,
        t: float,
        timestamps: Mapping[str, float] | None = None,
        ego: Mapping[str, Any] | None = None,
        wheel: Mapping[str, Any] | None = None,
        pedals: Mapping[str, Any] | None = None,
        tech: Mapping[str, Any] | None = None,
    ) -> StripRecord | None:
        if not self.recording:
            return None
        writer = self._writer
        if writer is None or writer.root != Path(self.destination):
            writer = StripWriter(self.destination)
            self._writer = writer
        kept, stamps, tick = same_tick_frames(frames, timestamps, t)
        try:
            return writer.write(
                kept,
                t=tick,
                timestamps=stamps,
                ego=ego,
                wheel=wheel,
                pedals=pedals,
                tech=tech,
            )
        except StripTimestampError:
            return None

    def arm_free_space(self) -> None:
        """Arm the wipe. Deletes nothing."""
        if not str(self.destination).strip():
            return
        self.free_space_armed = True
        self._armed_destination = self.destination

    def confirm_free_space(self) -> list[Path]:
        """Delete this folder's training files only when Free up space is armed."""
        if not self.free_space_armed:
            return []
        root = self._armed_destination
        self.free_space_armed = False
        self._armed_destination = ""
        if not root or root != self.destination:
            return []
        self._writer = None
        return wipe_training(root, confirm=True)

    def _save(self) -> None:
        try:
            _save_destination(self.prefs_path, self.destination)
        except OSError:
            return
