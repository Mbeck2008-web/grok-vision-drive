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


class StripWriter:
    """Append-only directory: ``meta.json``, ``strips/000000.jpg``, ``state.jsonl``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.strips = self.root / "strips"
        self.strips.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.jsonl"
        self._n, self._prev_t = self._resume()
        self._write_meta()

    def _resume(self) -> tuple[int, float | None]:
        rows = read_state_jsonl(self.state_path)
        if not rows:
            return 0, None
        last = rows[-1].get("t")
        prev = None if last is None else float(last)
        return len(rows), prev

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
            "n_strips": self._n,
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
        idx = self._n
        final = self.strips / f"{idx:06d}.jpg"
        tmp = self.strips / f"{idx:06d}.jpg.tmp"
        tmp.write_bytes(encode_jpeg(canvas))
        tmp.replace(final)
        line = {
            "i": idx,
            "t": t_store,
            "dt_s": dt_s,
            "ego": _jsonable(dict(ego or {})),
            "wheel": _jsonable(dict(wheel or {})),
            "pedals": _jsonable(dict(pedals or {})),
        }
        if tech is not None:
            line["tech"] = _jsonable(dict(tech))
        try:
            with self.state_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except Exception:
            final.unlink(missing_ok=True)
            raise
        self._n = idx + 1
        self._prev_t = t_store
        self._write_meta()
        return StripRecord(index=idx, path=final, t=t_store, dt_s=dt_s)
