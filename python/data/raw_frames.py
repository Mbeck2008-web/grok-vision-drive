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

from python.data.binary_io import PNG_SIG, os_binary, write_new_file
from python.data.strip_writer import STRIP_ORDER, compose_strip, read_state_jsonl

# The collector sleeps 0.1 s, so 10 Hz is the planning rate. A slower poll
# shows up in the measured minute line, not in this ceiling.
COLLECT_HZ = 10.0
# PNG size / uint8 size, measured on the 85 real BeamNG frames in
# tests/fixtures/gt_run*.jpg (1336x96 strips). Mean 0.507, worst 0.619.
PNG_RATIO_MEAN = 0.507
PNG_RATIO_MAX = 0.619
# Lane cameras stay at the collect rate. Pillars are slower. Repeaters and
# the rear camera are context, so they are slower still.
CAMERA_HZ = {
    "narrow": 10.0,
    "main": 10.0,
    "wide": 10.0,
    "pillarL": 2.0,
    "pillarR": 2.0,
    "repeatL": 1.0,
    "repeatR": 1.0,
    "rear": 1.0,
}
DRIVERS = ("beamng_ai", "gvd", "human")


def uncompressed_tick_bytes(sizes: list[tuple[int, int, int]]) -> int:
    """Width, height, channels. This is the uint8 size before PNG."""
    total = 0
    for width, height, channels in sizes:
        total += int(width) * int(height) * int(channels)
    return int(total)


def gb_per_hour(tick_bytes: int, hz: float) -> float:
    """GiB per hour, same 1024-based GB the disk lines already print."""
    return float(tick_bytes) * float(hz) * 3600.0 / (1024.0 ** 3)


def camera_due(cam_id: str, now_s: float, last_s: Mapping[str, float], rates: Mapping[str, float] | None = None) -> bool:
    """True when this camera's period has elapsed. ``now_s`` is monotonic."""
    table = CAMERA_HZ if rates is None else rates
    hz = float(table.get(str(cam_id), 0.0))
    if hz <= 0.0:
        return False
    prev = last_s.get(str(cam_id))
    if prev is None:
        return True
    return (float(now_s) - float(prev)) + 1e-4 >= (1.0 / hz)


def capture_plan(cfg: Mapping[str, Any] | None, ratio: float = PNG_RATIO_MAX) -> list[dict[str, Any]]:
    """Per-camera bytes and GB/h at ``CAMERA_HZ`` and a PNG/raw ratio."""
    rows = []
    for cam in camera_sidecar(cfg):
        cid = str(cam["id"])
        width, height = cam["resolution"]
        raw = int(width) * int(height) * 3
        hz = float(CAMERA_HZ.get(cid, 0.0))
        rows.append(
            {
                "id": cid,
                "width": int(width),
                "height": int(height),
                "channels": 3,
                "hz": hz,
                "uncompressed_bytes": raw,
                "uncompressed_gb_per_hour": gb_per_hour(raw, hz),
                "png_gb_per_hour": gb_per_hour(raw, hz) * float(ratio),
            }
        )
    return rows


def plan_totals(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "uncompressed_gb_per_hour": sum(float(row["uncompressed_gb_per_hour"]) for row in rows),
        "png_gb_per_hour": sum(float(row["png_gb_per_hour"]) for row in rows),
    }


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
    # GVD_GT_FILE=gt_painted.jsonl trains on the offline painted-lane relabel
    # without touching the recorded gt.jsonl.
    gt_name = os.environ.get("GVD_GT_FILE") or "gt.jsonl"
    labels = read_state_jsonl(root / gt_name)
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
    """Exclusive binary create, then a signature and size check.

    ``O_BINARY`` is required. Without it Windows rewrites every ``\\n`` as
    ``\\r\\n`` and the PNG signature becomes ``\\x89PNG\\r\\r\\n\\x1a``.
    """
    write_new_file(path, payload)


def _append_line(path: Path, line: Mapping[str, Any]) -> None:
    blob = (json.dumps(line, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(path),
        os.O_CREAT | os.O_APPEND | os.O_WRONLY | os_binary(),
        0o644,
    )
    try:
        view = memoryview(blob)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise OSError(f"raw state write made no progress: {path}")
            view = view[wrote:]
    finally:
        os.close(fd)


def encode_cameras(
    frames: Mapping[str, Any] | None,
) -> list[tuple[str, bytes, str, tuple[int, ...], str]]:
    """Lossless bytes for one tick. PNG level 1 for uint8, otherwise ``.npy``."""
    encoded: list[tuple[str, bytes, str, tuple[int, ...], str]] = []
    for cam_id, image in dict(frames or {}).items():
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
            continue
        arr = np.ascontiguousarray(image)
        png = _png_bytes(arr)
        shape = tuple(int(v) for v in arr.shape)
        if png is not None:
            encoded.append((str(cam_id), png, ".png", shape, str(arr.dtype)))
            continue
        import io

        buf = io.BytesIO()
        np.save(buf, arr)
        encoded.append((str(cam_id), buf.getvalue(), ".npy", shape, str(arr.dtype)))
    return encoded


def _png_bytes(arr: np.ndarray) -> bytes | None:
    """Lossless PNG of this uint8 array. None when PNG cannot hold it."""
    if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
        return None
    if arr.ndim == 3 and int(arr.shape[2]) not in (1, 3, 4):
        return None
    import cv2

    # Level 1 is still lossless. The default (3–6) was too slow to hold 10 Hz.
    ok, buf = cv2.imencode(
        ".png",
        np.ascontiguousarray(arr),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
    )
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
        self.stored_bytes = 0
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
            "hz": dict(CAMERA_HZ),
            "png_ratio_measured": {
                "mean": PNG_RATIO_MEAN,
                "max": PNG_RATIO_MAX,
                "frames": 85,
                "source": "tests/fixtures/gt_run*.jpg",
            },
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
        driver: str | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.write_encoded(
            encode_cameras(frames),
            t=t,
            timestamps=timestamps,
            ego=ego,
            wheel=wheel,
            pedals=pedals,
            tech=tech,
            driver=driver,
            tags=tags,
        )

    def write_encoded(
        self,
        encoded: list[tuple[str, bytes, str, tuple[int, ...], str]],
        *,
        t: float,
        timestamps: Mapping[str, Any] | None = None,
        ego: Mapping[str, Any] | None = None,
        wheel: Mapping[str, Any] | None = None,
        pedals: Mapping[str, Any] | None = None,
        tech: Mapping[str, Any] | None = None,
        driver: str | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write already-encoded camera bytes and one state line.

        Callers that encode on several threads still commit here one tick
        at a time, in capture order, so ``dt_s`` stays the capture gap.
        """
        idx = self._n
        t_store = round(float(t), 6)
        dt_s = None if self._prev_t is None else round(t_store - self._prev_t, 6)
        cams: dict[str, Any] = {}
        for cam_id, payload, suffix, shape, dtype in encoded:
            folder = self.raw / str(cam_id)
            final = folder / f"{idx:06d}{suffix}"
            _write_bytes(final, payload)
            self.stored_bytes += len(payload)
            wall, sensor_t = _stamp_of(None if timestamps is None else timestamps.get(cam_id))
            cam_dt = None
            prev = self._prev_cam_t.get(str(cam_id))
            if wall is not None and prev is not None:
                cam_dt = round(float(wall) - prev, 6)
            if wall is not None:
                self._prev_cam_t[str(cam_id)] = float(wall)
            pose = dict(self._by_id.get(str(cam_id)) or {})
            cams[str(cam_id)] = {
                "file": str(final.relative_to(self.root)).replace("\\", "/"),
                "t": wall,
                "sensor_t": sensor_t,
                "dt_s": cam_dt,
                "shape": [int(v) for v in shape],
                "dtype": str(dtype),
                "pos_m": pose.get("pos_m"),
                "yaw_deg": pose.get("yaw_deg"),
                "pitch_deg": pose.get("pitch_deg"),
                "fov_v": pose.get("fov_v"),
                "near_m": pose.get("near_m"),
                "far_m": pose.get("far_m"),
                "resolution": pose.get("resolution"),
            }
        ego_d = {str(k): _jsonable(v) for k, v in dict(ego or {}).items()}
        wheel_d = {str(k): _jsonable(v) for k, v in dict(wheel or {}).items()}
        pedals_d = {str(k): _jsonable(v) for k, v in dict(pedals or {}).items()}
        who = str(driver or "")
        if who and who not in DRIVERS:
            who = "unknown"
        line = {
            "i": idx,
            "t": t_store,
            "dt_s": dt_s,
            "driver": who or None,
            "ego": ego_d,
            "wheel": wheel_d,
            "pedals": pedals_d,
            "capture": {
                "t": t_store,
                "dt_s": dt_s,
                "steer": wheel_d.get("steer", wheel_d.get("steering_input")),
                "throttle": pedals_d.get("throttle"),
                "brake": pedals_d.get("brake"),
                "speed_mps": ego_d.get("speed_mps"),
                "yaw_rate": ego_d.get("yaw_rate"),
                "pos": ego_d.get("pos"),
                "dir": ego_d.get("dir"),
                "driver": who or None,
                "gt": "gt.jsonl",
            },
            "cams": cams,
        }
        for key, value in dict(tags or {}).items():
            line[str(key)] = _jsonable(value)
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


def read_image_file(path: Path | str) -> np.ndarray | None:
    """Decode a PNG (or other OpenCV image) without ``imread``.

    ``cv2.imread`` returns None for some Windows paths even when the file
    is a normal PNG. Reading the bytes and decoding them is the same
    pixels, and it is what the validator and the trainer both use.
    """
    return _decode_image_file(Path(path))


def repaired_copies(path: Path) -> list[Path]:
    """Repaired siblings of one capture. The original path is not included."""
    file = Path(path)
    found: list[Path] = []
    alongside = file.with_name(file.stem + ".repaired.png")
    if alongside != file:
        found.append(alongside)
    if file.parent.parent.name == "raw":
        root = file.parent.parent.parent
        found.append(root / "raw_repaired" / file.parent.name / file.name)
    return found


def png_signature_ok(path: Path | str) -> bool:
    """True when the file starts with the real PNG signature, not a CRLF rewrite."""
    try:
        with Path(path).open("rb") as handle:
            return handle.read(8) == PNG_SIG
    except OSError:
        return False


def image_file_decodable(path: Path | str) -> bool:
    """True when this path, or its repaired PNG sibling, can be opened as an image.

    A PNG is accepted on its signature. Full decode stays in the loader.
    ``raw`` and ``raw_repaired`` are not both required.
    """
    file = Path(path)
    if file.suffix.lower() == ".npy" and file.is_file():
        return True
    if file.suffix.lower() == ".png":
        if file.is_file() and png_signature_ok(file):
            return True
        for alt in repaired_copies(file):
            if alt.is_file() and png_signature_ok(alt):
                return True
    return False


def training_image_path(path: Path | str) -> Path | None:
    """The one file the loader will read for this capture, or None."""
    file = Path(path)
    if file.suffix.lower() == ".npy":
        return file if file.is_file() else None
    if file.suffix.lower() != ".png":
        return None
    if file.is_file() and png_signature_ok(file):
        return file
    for alt in repaired_copies(file):
        if alt.is_file() and png_signature_ok(alt):
            return alt
    return None


def cameras_for_row(row: Mapping[str, Any] | None) -> list[str]:
    """Strip cameras this tick recorded. The net paints any missing sector black."""
    cams = None if row is None else row.get("cams")
    if isinstance(cams, dict) and cams:
        names = [str(key) for key in cams if str(key) in STRIP_ORDER]
        if names:
            return names
    return list(STRIP_ORDER)


def row_has_decodable_frame(root: Path | str, row: Mapping[str, Any]) -> bool:
    """True when at least one camera the strip uses has a decodable file."""
    try:
        index = int(row.get("i"))
    except (TypeError, ValueError):
        return False
    raw = Path(root) / "raw"
    stem = f"{index:06d}"
    for cam in cameras_for_row(row):
        if image_file_decodable(raw / cam / f"{stem}.png"):
            return True
        if image_file_decodable(raw / cam / f"{stem}.npy"):
            return True
    return False


def training_frame_bytes(
    root: Path | str,
    *,
    progress: Any = None,
) -> tuple[int, int]:
    """Bytes and frame count the trainer reads. Each capture is counted once.

    A repaired PNG replaces the original in the total. The two copies are
    not added together. State and camera sidecars are included once.
    """
    root = Path(root)
    if not root.exists():
        return 0, 0
    total = 0
    frames = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if "raw_repaired" in path.parts:
            continue
        if path.parent.parent.name == "raw" and path.suffix.lower() in {".png", ".npy"}:
            chosen = training_image_path(path)
            if chosen is None:
                continue
            try:
                total += int(chosen.stat().st_size)
            except OSError:
                continue
            frames += 1
            if progress is not None and frames % 5000 == 0:
                progress(frames, total)
            continue
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    if progress is not None:
        progress(frames, total)
    return total, frames


def read_image_with_repair(path: Path | str) -> np.ndarray | None:
    """Decode ``path``, then a repaired copy if the original does not decode.

    The original file is only read. Nothing here deletes or rewrites it.
    """
    file = Path(path)
    image = _decode_image_file(file)
    if image is not None:
        return image
    for alt in repaired_copies(file):
        if alt.is_file():
            image = _decode_image_file(alt)
            if image is not None:
                return image
    return None


def _decode_image_file(file: Path) -> np.ndarray | None:
    import cv2

    try:
        data = np.fromfile(file, dtype=np.uint8)
    except OSError:
        return None
    if int(data.size) == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return image


def load_raw_cameras(root: Path | str, index: int) -> dict[str, np.ndarray]:
    """Read this tick's camera files. No resize and no channel drop.

    A text-mode PNG that does not decode is replaced, for this read only, by
    ``raw_repaired/<cam>/NNNNNN.png`` when that copy exists.
    """
    root = Path(root)
    raw = root / "raw"
    frames: dict[str, np.ndarray] = {}
    if not raw.is_dir():
        return frames
    for folder in raw.iterdir():
        if not folder.is_dir():
            continue
        png = folder / f"{int(index):06d}.png"
        npy = folder / f"{int(index):06d}.npy"
        if png.is_file():
            image = read_image_with_repair(png)
            if image is not None:
                frames[folder.name] = image
        elif npy.is_file():
            frames[folder.name] = np.load(npy)
    return frames


def load_training_bgr(root: Path | str, index: int) -> np.ndarray:
    """Stitch and resize at load time. The files on disk stay as saved.

    A recording that has ``raw/`` never falls through to ``strips/*.jpg``.
    Legacy strips are only for a folder that has no raw tree.
    """
    root = Path(root)
    raw = root / "raw"
    if raw.is_dir():
        frames = load_raw_cameras(root, index)
        if not frames:
            raise FileNotFoundError(
                f"no decodable raw frame for index {int(index):06d} under {raw}"
            )
        return compose_strip(frames)
    strips = root / "strips"
    if not strips.is_dir():
        raise FileNotFoundError(f"no raw frames and no strips under {root}")
    import cv2

    from python.data.strip_writer import CANVAS_H, CANVAS_W

    path = strips / f"{int(index):06d}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if int(bgr.shape[0]) != CANVAS_H or int(bgr.shape[1]) != CANVAS_W:
        raise ValueError(f"{path} is {bgr.shape[1]}x{bgr.shape[0]}, expected {CANVAS_W}x{CANVAS_H}")
    return bgr


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _pick(sources: tuple[Mapping[str, Any], ...], keys: tuple[str, ...]) -> float | None:
    for src in sources:
        for key in keys:
            if key in src and src.get(key) is not None:
                got = _finite(src.get(key))
                if got is not None:
                    return got
    return None


def _vec3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    out = []
    for item in value[:3]:
        got = _finite(item)
        if got is None:
            return None
        out.append(got)
    return out


def controls_from_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Steer, pedals, speed, yaw, and pose from a BeamNG vehicle state.

    Missing fields stay None. This does not invent a driver.
    """
    state = dict(state or {})
    electrics = state.get("electrics") if isinstance(state.get("electrics"), dict) else {}
    sources = (electrics, state)
    pos = _vec3(state.get("pos"))
    direction = _vec3(state.get("dir") or state.get("forward"))
    vel = _vec3(state.get("vel") or state.get("velocity"))
    speed = _pick(sources, ("wheelspeed", "airspeed", "speed_mps"))
    if speed is None and vel is not None:
        speed = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5
    return {
        "steer": _pick(sources, ("steering_input", "steering", "steer")),
        "throttle": _pick(sources, ("throttle_input", "throttle")),
        "brake": _pick(sources, ("brake_input", "brake")),
        "speed_mps": speed,
        "yaw_rate": _pick(sources, ("yaw_rate", "yawRate")),
        "pos": pos,
        "dir": direction,
        "damage": _pick(sources, ("damage",)),
    }


def yaw_rate_from_dirs(prev: Any, curr: Any, dt_s: float) -> float | None:
    """Yaw rate about Z from two forward vectors. Radians per second, CCW positive."""
    import math

    a = _vec3(prev)
    b = _vec3(curr)
    dt = _finite(dt_s)
    if a is None or b is None or dt is None or dt <= 0.0:
        return None

    def norm(v: list[float]) -> tuple[float, float]:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if length < 1e-9:
            return 0.0, 0.0
        return v[0] / length, v[1] / length

    ax, ay = norm(a)
    bx, by = norm(b)
    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    return math.atan2(ax * by - ay * bx, dot) / dt


def list_raw_images(root: Path | str) -> list[Path]:
    raw = Path(root) / "raw"
    if not raw.is_dir():
        return []
    found = []
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix in {".png", ".npy"} and path.stem.isdigit():
            found.append(path)
    return sorted(found)


def raw_image_counts(paths: list[Path]) -> dict[str, Any]:
    """Tick count is one camera, not the sum of every camera file.

    ``raw_frames`` is the main-camera count when that camera was recorded,
    otherwise the smallest camera count. ``raw_files`` is the sum.
    """
    per_cam: dict[str, int] = {}
    for path in paths:
        cam = path.parent.name
        per_cam[cam] = per_cam.get(cam, 0) + 1
    if "main" in per_cam:
        ticks = int(per_cam["main"])
    elif per_cam:
        ticks = int(min(per_cam.values()))
    else:
        ticks = 0
    return {"raw_frames": ticks, "raw_files": int(sum(per_cam.values())), "per_cam": per_cam}


def raw_count_issues(per_cam: Mapping[str, int]) -> list[str]:
    """Cameras that share a planned rate should have the same file count.

    Pillars and the rear camera are slower on purpose, so they are not
    compared with the 10 Hz cameras.
    """
    groups: dict[float, list[str]] = {}
    for cam in per_cam:
        if cam not in CAMERA_HZ:
            continue
        groups.setdefault(float(CAMERA_HZ[cam]), []).append(str(cam))
    issues: list[str] = []
    for hz, cams in sorted(groups.items()):
        if len(cams) < 2:
            continue
        counts = {int(per_cam[cam]) for cam in cams}
        if len(counts) == 1:
            continue
        detail = ", ".join(f"{cam}={int(per_cam[cam])}" for cam in sorted(cams))
        issues.append(f"raw count mismatch at {hz:.0f} Hz ({detail})")
    return issues
