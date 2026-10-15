"""Lossless per-camera frames, saved as BeamNG returned them.

Nothing in this file resizes, crops, recolours, denoises, or stitches.
Those steps run later, in the trainer loader, on a copy. Ground-truth
labels are a separate ``gt.jsonl``. Debug drawings never touch these files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from python.data.strip_writer import compose_strip, read_state_jsonl

# The collector sleeps 0.1 s, so 10 Hz is the planning rate. A slower poll
# shows up in the measured minute line, not in this ceiling.
COLLECT_HZ = 10.0


def uncompressed_tick_bytes(sizes: list[tuple[int, int, int]]) -> int:
    """Width, height, channels. This is the uint8 size before PNG."""
    total = 0
    for width, height, channels in sizes:
        total += int(width) * int(height) * int(channels)
    return int(total)


def gb_per_hour(tick_bytes: int, hz: float) -> float:
    """GiB per hour, same 1024-based GB the disk lines already print."""
    return float(tick_bytes) * float(hz) * 3600.0 / (1024.0 ** 3)


def describe_raw_disk(tick_bytes: int, hz: float) -> str:
    per_min_mib = float(tick_bytes) * float(hz) * 60.0 / (1024.0 ** 2)
    per_hour = gb_per_hour(tick_bytes, hz)
    mib = float(tick_bytes) / (1024.0 ** 2)
    return (
        f"raw uncompressed {mib:.2f} MiB/tick at {float(hz):.1f} Hz "
        f"= {per_min_mib:.0f} MiB/min ({per_hour:.1f} GB/h)"
    )


def camera_sidecar(cfg: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Pose and intrinsics from the camera yaml. Not read off the pixels."""
    from python.sensors.cameras import CAM_IDS, camera_clip_planes, yaw_pitch_to_dir_up

    cfg = dict(cfg or {})
    origin = cfg.get("origin_offset") or [0.0, 0.0, 0.0]
    records: list[dict[str, Any]] = []
    for spec in cfg.get("cameras") or []:
        if not isinstance(spec, dict):
            continue
        cid = str(spec.get("id") or "")
        if cid not in CAM_IDS:
            continue
        pos = list(spec.get("pos_m") or [0.0, 1.2, 1.2])
        pos = [
            float(pos[0]) + float(origin[0]),
            float(pos[1]) + float(origin[1]),
            float(pos[2]) + float(origin[2]),
        ]
        yaw = float(spec.get("yaw_deg") or 0.0)
        pitch = float(spec.get("pitch_deg") or 0.0)
        direction, up = yaw_pitch_to_dir_up(yaw, pitch)
        res = spec.get("live_res") or [640, 480]
        near_m, far_m = camera_clip_planes(
            spec, cid=cid, defaults={"near_m": cfg.get("near_m", 0.05)},
        )
        records.append(
            {
                "id": cid,
                "pos_m": [float(v) for v in pos],
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "fov_v": float(spec.get("fov_v") or 70.0),
                "near_m": float(near_m),
                "far_m": float(far_m),
                "resolution": [int(res[0]), int(res[1])],
                "direction": [float(v) for v in direction],
                "up": [float(v) for v in up],
            }
        )
    return records


def rig_tick_bytes(cfg: Mapping[str, Any] | None, channels: int = 3) -> int:
    sizes = []
    for cam in camera_sidecar(cfg):
        width, height = cam["resolution"]
        sizes.append((int(width), int(height), int(channels)))
    return uncompressed_tick_bytes(sizes)


def frame_nbytes(frames: Mapping[str, Any] | None) -> int:
    total = 0
    for image in (frames or {}).values():
        if isinstance(image, np.ndarray):
            total += int(image.nbytes)
    return int(total)


def load_scene_rows(root: Path | str) -> list[dict[str, Any]]:
    """State lines with ground truth joined from ``gt.jsonl`` when that file exists.

    The join is in memory. Neither file is rewritten.
    """
    root = Path(root)
    rows = read_state_jsonl(root / "state.jsonl")
    labels = read_state_jsonl(root / "gt.jsonl")
    by_index: dict[int, dict[str, Any]] = {}
    for label in labels:
        try:
            by_index[int(label.get("i"))] = label
        except (TypeError, ValueError):
            continue
    if not by_index:
        return rows
    merged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            label = by_index.get(int(row.get("i")))
        except (TypeError, ValueError):
            label = None
        if isinstance(label, dict) and isinstance(label.get("tech"), dict):
            item["tech"] = label["tech"]
        merged.append(item)
    return merged


def _used_indices(root: Path) -> set[int]:
    used: set[int] = set()
    for row in read_state_jsonl(root / "state.jsonl"):
        try:
            used.add(int(row.get("i")))
        except (TypeError, ValueError):
            continue
    raw = root / "raw"
    if raw.is_dir():
        for path in raw.rglob("*"):
            if path.is_file() and path.stem.isdigit() and path.suffix in {".png", ".npy"}:
                used.add(int(path.stem))
    return used


def _next_index(used: set[int]) -> int:
    n = 0
    while n in used:
        n += 1
    return n


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(payload)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise OSError(f"raw write made no progress: {path}")
            view = view[wrote:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_line(path: Path, line: Mapping[str, Any]) -> None:
    blob = (json.dumps(line, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        view = memoryview(blob)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise OSError(f"raw state write made no progress: {path}")
            view = view[wrote:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _png_bytes(arr: np.ndarray) -> bytes | None:
    """Lossless PNG of this uint8 array. None when PNG cannot hold it."""
    if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
        return None
    if arr.ndim == 3 and int(arr.shape[2]) not in (1, 3, 4):
        return None
    import cv2

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr))
    if not ok:
        return None
    return buf.tobytes()


def _stamp_of(raw: Any) -> tuple[float | None, Any]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), None
    if isinstance(raw, dict):
        wall = raw.get("wall", raw.get("t"))
        try:
            wall_f = None if wall is None else float(wall)
        except (TypeError, ValueError):
            wall_f = None
        return wall_f, raw.get("sensor_t")
    return None, None


class RawFrameWriter:
    """One directory of per-camera PNG or ``.npy`` files plus two JSONL logs.

    ``state.jsonl`` is time, dt, and the camera pose that was current.
    ``gt.jsonl`` is the lane target. Pixels are not in either file.
    """

    def __init__(self, root: Path | str, cameras: list[dict[str, Any]] | None = None) -> None:
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.jsonl"
        self.gt_path = self.root / "gt.jsonl"
        self.cameras = list(cameras or [])
        self._by_id = {str(cam["id"]): cam for cam in self.cameras if cam.get("id")}
        self._prev_t: float | None = None
        self._prev_cam_t: dict[str, float] = {}
        self._n = _next_index(_used_indices(self.root))
        self._write_cameras()

    def _write_cameras(self) -> None:
        payload = {
            "kind": "gvd_raw_cameras",
            "edited": False,
            "pixels": "png when the array is uint8, otherwise npy",
            "note": (
                "Each file is the array BeamNG returned. "
                "Resize, stitch, and the ego-edge mask run in the trainer loader."
            ),
            "cameras": self.cameras,
        }
        path = self.root / "cameras.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write(
        self,
        frames: Mapping[str, Any] | None,
        *,
        t: float,
        timestamps: Mapping[str, Any] | None = None,
        ego: Mapping[str, Any] | None = None,
        wheel: Mapping[str, Any] | None = None,
        pedals: Mapping[str, Any] | None = None,
        tech: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        src = frames or {}
        idx = self._n
        t_store = round(float(t), 6)
        dt_s = None if self._prev_t is None else round(t_store - self._prev_t, 6)
        cams: dict[str, Any] = {}
        for cam_id, image in src.items():
            if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
                continue
            # Contiguous only. Values, shape, and channel count stay as returned.
            arr = np.ascontiguousarray(image)
            folder = self.raw / str(cam_id)
            png = _png_bytes(arr)
            if png is not None:
                final = folder / f"{idx:06d}.png"
                _write_bytes(final, png)
                stored = final
            else:
                import io

                final = folder / f"{idx:06d}.npy"
                buf = io.BytesIO()
                np.save(buf, arr)
                _write_bytes(final, buf.getvalue())
                stored = final
            wall, sensor_t = _stamp_of(None if timestamps is None else timestamps.get(cam_id))
            cam_dt = None
            prev = self._prev_cam_t.get(str(cam_id))
            if wall is not None and prev is not None:
                cam_dt = round(float(wall) - prev, 6)
            if wall is not None:
                self._prev_cam_t[str(cam_id)] = float(wall)
            pose = dict(self._by_id.get(str(cam_id)) or {})
            cams[str(cam_id)] = {
                "file": str(stored.relative_to(self.root)).replace("\\", "/"),
                "t": wall,
                "sensor_t": sensor_t,
                "dt_s": cam_dt,
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
                "pos_m": pose.get("pos_m"),
                "yaw_deg": pose.get("yaw_deg"),
                "pitch_deg": pose.get("pitch_deg"),
                "fov_v": pose.get("fov_v"),
                "near_m": pose.get("near_m"),
                "far_m": pose.get("far_m"),
                "resolution": pose.get("resolution"),
            }
        line = {
            "i": idx,
            "t": t_store,
            "dt_s": dt_s,
            "ego": {str(k): v for k, v in dict(ego or {}).items()},
            "wheel": {str(k): v for k, v in dict(wheel or {}).items()},
            "pedals": {str(k): v for k, v in dict(pedals or {}).items()},
            "cams": cams,
        }
        _append_line(self.state_path, line)
        if tech is not None:
            _append_line(
                self.gt_path,
                {"i": idx, "t": t_store, "tech": _jsonable(dict(tech))},
            )
        self._n = idx + 1
        self._prev_t = t_store
        return line


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


def load_raw_cameras(root: Path | str, index: int) -> dict[str, np.ndarray]:
    """Read this tick's camera files. No resize and no channel drop."""
    root = Path(root)
    raw = root / "raw"
    frames: dict[str, np.ndarray] = {}
    if not raw.is_dir():
        return frames
    import cv2

    for folder in raw.iterdir():
        if not folder.is_dir():
            continue
        png = folder / f"{int(index):06d}.png"
        npy = folder / f"{int(index):06d}.npy"
        if png.is_file():
            image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if image is not None:
                frames[folder.name] = image
        elif npy.is_file():
            frames[folder.name] = np.load(npy)
    return frames


def load_training_bgr(root: Path | str, index: int) -> np.ndarray:
    """Stitch and resize at load time. The files on disk stay as saved.

    A legacy ``strips/NNNNNN.jpg`` is used only when this tick has no raw cameras.
    """
    frames = load_raw_cameras(root, index)
    if frames:
        return compose_strip(frames)
    import cv2

    from python.data.strip_writer import CANVAS_H, CANVAS_W

    path = Path(root) / "strips" / f"{int(index):06d}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if int(bgr.shape[0]) != CANVAS_H or int(bgr.shape[1]) != CANVAS_W:
        raise ValueError(f"{path} is {bgr.shape[1]}x{bgr.shape[0]}, expected {CANVAS_W}x{CANVAS_H}")
    return bgr


def list_raw_images(root: Path | str) -> list[Path]:
    raw = Path(root) / "raw"
    if not raw.is_dir():
        return []
    found = []
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix in {".png", ".npy"} and path.stem.isdigit():
            found.append(path)
    return sorted(found)
