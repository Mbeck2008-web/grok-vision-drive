"""Reality View v1 contract, versioned SHM slot schema, and process boundary.

Spec: docs/specs/reality_view_3dgs.md (Step 1).
This module defines only the data/IPC contract, default-OFF flag resolution,
and a separate-process stub launcher. It contains NO renderer, NO unprojection,
NO 3DGS math, and imports NO safety-path modules (heartbeat/AEB/engage/frame_gate).
"""
from __future__ import annotations

import ctypes
import math
import os
import signal
import subprocess
import struct
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

RV_SCHEMA_VERSION = 1
RV_SHM_MAGIC = b"GVDRV001"
RV_MAX_SPLATS = 300_000
RV_MAX_LONG_SIDE = 640
RV_STALE_AGE_MS = 200.0
RV_RENDERER_MODULE = "python.viz.rv_renderer_stub"

RV_RIG_CAMERAS: tuple[str, ...] = (
    "narrow",
    "main",
    "wide",
    "pillarL",
    "pillarR",
    "repeatL",
    "repeatR",
    "rear",
)
RV_WINDSHIELD_CAMERAS: tuple[str, ...] = ("narrow", "main", "wide")
RV_CAPTURE_BACKENDS: frozenset[str] = frozenset({"beamngpy", "window", "stub"})

RV_FLAG_HAS_DEPTH = 1 << 0
RV_FLAG_STALE = 1 << 1
RV_FLAG_VALID = 1 << 2

# 128-byte aligned little-endian header:
# magic(8s), schema_version(I), flags(I), seq(Q), timestamp_s(d), age_ms(d),
# width(I), height(I), splat_count(I), payload_bytes(I), generation(I),
# reserved(I), pose_4x4(16f)
SLOT_HEADER_STRUCT = struct.Struct("<8sIIQddIIIIII16f")
SLOT_HEADER_SIZE = SLOT_HEADER_STRUCT.size

# Max legal output from schema: long side <= RV_MAX_LONG_SIDE (640),
# RGB uint8 (3 B/px) + float32 depth (4 B/px) + 128-B header.
RV_DEFAULT_SLOT_BYTES = SLOT_HEADER_SIZE + RV_MAX_LONG_SIDE * RV_MAX_LONG_SIDE * (3 + 4)

_FORBIDDEN_DEPTH_KEYS: frozenset[str] = frozenset(
    {
        "depth_gt",
        "gt_depth",
        "sim_depth",
        "aux_depth",
        "sem_gt",
        "gt_sem",
        "path_world",
        "debug_gt",
        "labels",
    }
)


class RvSchemaError(ValueError):
    """Raised loudly when an RV input, output, or SHM slot violates v1 schema."""


def _require_version(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v != RV_SCHEMA_VERSION:
        raise RvSchemaError(f"unsupported RV schema_version {v!r}; expected {RV_SCHEMA_VERSION}")
    return v


def _require_non_bool_int(v: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise RvSchemaError(f"{name} must be an int >= {minimum}, got {v!r}")
    return int(v)


def _require_finite_float(v: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(v, bool):
        raise RvSchemaError(f"{name} must be a finite float, got {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise RvSchemaError(f"{name} must be a finite float, got {v!r}") from exc
    if not math.isfinite(x) or (minimum is not None and x < minimum):
        raise RvSchemaError(f"{name} out of range or non-finite: {v!r}")
    return x


def _check_matrix(arr: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        mat = np.asarray(arr, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RvSchemaError(f"{name} could not convert to float32 array") from exc
    if mat.shape != shape or not np.all(np.isfinite(mat)):
        raise RvSchemaError(f"{name} must be finite shape {shape}, got {mat.shape}")
    return mat


def _is_forbidden_gt_key(key: str) -> bool:
    low = str(key).strip().lower()
    return (
        low in _FORBIDDEN_DEPTH_KEYS
        or low.startswith("gt_")
        or low.endswith("_gt")
        or "sim_depth" in low
    )


def reality_view_enabled(
    cameras: Mapping[str, Any] | None = None,
    *,
    cli_reality_view: bool = False,
) -> bool:
    """Resolve the RV live-boundary flag (default OFF, literal True only, no env path)."""
    if cli_reality_view is True:
        return True
    if not isinstance(cameras, Mapping):
        return False
    viz = cameras.get("viz")
    return isinstance(viz, Mapping) and viz.get("reality_view_enabled") is True


@dataclass(frozen=True)
class RvCameraInput:
    cam_id: str
    rgb: np.ndarray
    intrinsics_3x3: np.ndarray
    extrinsics_4x4: np.ndarray


@dataclass(frozen=True)
class RvInputSpec:
    schema_version: int
    seq: int
    timestamp_s: float
    ego_pose_4x4: np.ndarray
    camera_frames: Mapping[str, RvCameraInput]
    predicted_depth: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class RvOutputSpec:
    schema_version: int
    seq: int
    timestamp_s: float
    pose_4x4: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    stale: bool = False


@dataclass(frozen=True)
class RvShmHeader:
    magic: bytes
    schema_version: int
    flags: int
    seq: int
    timestamp_s: float
    age_ms: float
    width: int
    height: int
    splat_count: int
    payload_bytes: int
    generation: int
    pose_4x4: tuple[float, ...]
    reserved: int = 0

    def pack(self) -> bytes:
        if self.magic != RV_SHM_MAGIC:
            raise RvSchemaError(f"bad RV SHM magic {self.magic!r}")
        _require_version(self.schema_version)
        _require_non_bool_int(self.flags, "flags", minimum=0)
        _require_non_bool_int(self.seq, "seq", minimum=0)
        ts = _require_finite_float(self.timestamp_s, "timestamp_s", minimum=0.0)
        age = _require_finite_float(self.age_ms, "age_ms", minimum=0.0)
        w = _require_non_bool_int(self.width, "width", minimum=0)
        h = _require_non_bool_int(self.height, "height", minimum=0)
        splats = _require_non_bool_int(self.splat_count, "splat_count", minimum=0)
        if splats > RV_MAX_SPLATS:
            raise RvSchemaError(f"splat_count {splats} exceeds RV_MAX_SPLATS={RV_MAX_SPLATS}")
        pbytes = _require_non_bool_int(self.payload_bytes, "payload_bytes", minimum=0)
        gen = _require_non_bool_int(self.generation, "generation", minimum=0)
        if len(self.pose_4x4) != 16:
            raise RvSchemaError(f"pose_4x4 must have 16 floats, got {len(self.pose_4x4)}")
        pose_vals = tuple(_require_finite_float(x, "pose_4x4") for x in self.pose_4x4)
        return SLOT_HEADER_STRUCT.pack(
            self.magic,
            self.schema_version,
            self.flags,
            self.seq,
            ts,
            age,
            w,
            h,
            splats,
            pbytes,
            gen,
            self.reserved,
            *pose_vals,
        )

    @classmethod
    def unpack(cls, buf: bytes | bytearray | memoryview) -> "RvShmHeader":
        mv = memoryview(buf)
        if len(mv) < SLOT_HEADER_SIZE:
            raise RvSchemaError(f"SHM slot buffer too short: {len(mv)} < {SLOT_HEADER_SIZE}")
        unpacked = SLOT_HEADER_STRUCT.unpack_from(mv, 0)
        magic = unpacked[0]
        if magic != RV_SHM_MAGIC:
            raise RvSchemaError(f"bad RV SHM magic {magic!r}; expected {RV_SHM_MAGIC!r}")
        version = _require_version(unpacked[1])
        flags = int(unpacked[2])
        seq = int(unpacked[3])
        ts = _require_finite_float(unpacked[4], "timestamp_s", minimum=0.0)
        age = _require_finite_float(unpacked[5], "age_ms", minimum=0.0)
        w, h, splats, pbytes, gen, reserved = (int(x) for x in unpacked[6:12])
        if splats > RV_MAX_SPLATS:
            raise RvSchemaError(f"splat_count {splats} exceeds RV_MAX_SPLATS={RV_MAX_SPLATS}")
        pose_vals = tuple(float(x) for x in unpacked[12:28])
        if not all(math.isfinite(x) for x in pose_vals):
            raise RvSchemaError("non-finite pose_4x4 in SHM header")
        return cls(
            magic=magic,
            schema_version=version,
            flags=flags,
            seq=seq,
            timestamp_s=ts,
            age_ms=age,
            width=w,
            height=h,
            splat_count=splats,
            payload_bytes=pbytes,
            generation=gen,
            pose_4x4=pose_vals,
            reserved=reserved,
        )


def validate_rv_input(spec: RvInputSpec) -> None:
    """Validate versioned RV input contract; reject sim GT and oversize images."""
    _require_version(spec.schema_version)
    _require_non_bool_int(spec.seq, "seq", minimum=0)
    _require_finite_float(spec.timestamp_s, "timestamp_s", minimum=0.0)
    _check_matrix(spec.ego_pose_4x4, (4, 4), "ego_pose_4x4")
    if not isinstance(spec.camera_frames, Mapping) or not spec.camera_frames:
        raise RvSchemaError("camera_frames must be a non-empty mapping")
    for key, cam in spec.camera_frames.items():
        if _is_forbidden_gt_key(str(key)):
            raise RvSchemaError(f"forbidden GT key in camera_frames: {key!r}")
        if not isinstance(cam, RvCameraInput):
            raise RvSchemaError(f"camera_frames[{key!r}] must be RvCameraInput")
        if cam.cam_id not in RV_RIG_CAMERAS:
            raise RvSchemaError(f"unknown rig camera {cam.cam_id!r}")
        rgb = np.asarray(cam.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise RvSchemaError(f"camera {cam.cam_id!r} rgb must be uint8 (H, W, 3), got {rgb.shape} {rgb.dtype}")
        if max(rgb.shape[0], rgb.shape[1]) > RV_MAX_LONG_SIDE or min(rgb.shape[0], rgb.shape[1]) <= 0:
            raise RvSchemaError(
                f"camera {cam.cam_id!r} resolution {rgb.shape[:2]} violates long side <= {RV_MAX_LONG_SIDE}"
            )
        _check_matrix(cam.intrinsics_3x3, (3, 3), f"{cam.cam_id}.intrinsics_3x3")
        _check_matrix(cam.extrinsics_4x4, (4, 4), f"{cam.cam_id}.extrinsics_4x4")

    if spec.predicted_depth is not None:
        if not isinstance(spec.predicted_depth, Mapping):
            raise RvSchemaError("predicted_depth must be a mapping or None")
        for key, depth in spec.predicted_depth.items():
            if _is_forbidden_gt_key(str(key)):
                raise RvSchemaError(f"forbidden GT/aux depth key {key!r} on live input contract")
            if str(key) not in spec.camera_frames:
                raise RvSchemaError(f"predicted_depth key {key!r} has no matching camera frame")
            d_arr = np.asarray(depth)
            h, w = np.asarray(spec.camera_frames[str(key)].rgb).shape[:2]
            if d_arr.dtype != np.float32 or d_arr.shape != (h, w):
                raise RvSchemaError(
                    f"predicted_depth[{key!r}] must be float32 {(h, w)}, got {d_arr.shape} {d_arr.dtype}"
                )


def validate_rv_output(out: RvOutputSpec) -> None:
    """Validate RV renderer output: versioned, finite pose, RGB + required matching depth."""
    _require_version(out.schema_version)
    _require_non_bool_int(out.seq, "seq", minimum=0)
    _require_finite_float(out.timestamp_s, "timestamp_s", minimum=0.0)
    _check_matrix(out.pose_4x4, (4, 4), "pose_4x4")
    if out.rgb is None:
        raise RvSchemaError("rgb is required")
    rgb = np.asarray(out.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RvSchemaError(f"rgb must be uint8 (H, W, 3), got {rgb.shape} {rgb.dtype}")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h <= 0 or w <= 0 or max(h, w) > RV_MAX_LONG_SIDE:
        raise RvSchemaError(
            f"rgb dimensions {(h, w)} must be positive with long side <= {RV_MAX_LONG_SIDE}"
        )
    if out.depth is None:
        raise RvSchemaError("depth buffer is required alongside RGB output")
    depth = np.asarray(out.depth)
    if depth.ndim != 2 or depth.dtype != np.float32 or depth.shape != (h, w):
        raise RvSchemaError(
            f"depth buffer must be float32 with shape {(h, w)} matching RGB, got {depth.shape} {depth.dtype}"
        )


def pack_output_slot(
    out: RvOutputSpec,
    *,
    age_ms: float = 0.0,
    generation: int = 1,
    splat_count: int = 0,
) -> bytes:
    """Serialize validated RV output (RGB + required depth buffer) into a versioned SHM slot."""
    validate_rv_output(out)
    rgb = np.ascontiguousarray(out.rgb, dtype=np.uint8)
    depth = np.ascontiguousarray(out.depth, dtype=np.float32)
    pose = np.ascontiguousarray(out.pose_4x4, dtype=np.float32).reshape(-1)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    rgb_bytes = rgb.tobytes()
    depth_bytes = depth.tobytes()
    flags = RV_FLAG_VALID | RV_FLAG_HAS_DEPTH
    if out.stale or float(age_ms) > RV_STALE_AGE_MS:
        flags |= RV_FLAG_STALE
    hdr = RvShmHeader(
        magic=RV_SHM_MAGIC,
        schema_version=out.schema_version,
        flags=flags,
        seq=out.seq,
        timestamp_s=out.timestamp_s,
        age_ms=age_ms,
        width=w,
        height=h,
        splat_count=splat_count,
        payload_bytes=len(rgb_bytes) + len(depth_bytes),
        generation=generation,
        pose_4x4=tuple(float(x) for x in pose),
    )
    return hdr.pack() + rgb_bytes + depth_bytes


def unpack_output_slot(buf: bytes | bytearray | memoryview) -> tuple[RvShmHeader, RvOutputSpec]:
    """Deserialize a versioned SHM output slot; fails loudly on version/magic/depth errors."""
    hdr = RvShmHeader.unpack(buf)
    if (hdr.flags & RV_FLAG_HAS_DEPTH) == 0:
        raise RvSchemaError("SHM output slot missing required RV_FLAG_HAS_DEPTH")
    if hdr.width <= 0 or hdr.height <= 0 or max(hdr.width, hdr.height) > RV_MAX_LONG_SIDE:
        raise RvSchemaError(f"invalid output dimensions {(hdr.width, hdr.height)}")
    rgb_nbytes = hdr.width * hdr.height * 3
    depth_nbytes = hdr.width * hdr.height * 4
    expected_payload = rgb_nbytes + depth_nbytes
    if hdr.payload_bytes != expected_payload:
        raise RvSchemaError(
            f"SHM payload_bytes={hdr.payload_bytes} does not match RGB+depth size={expected_payload}"
        )
    mv = memoryview(buf)
    total_needed = SLOT_HEADER_SIZE + expected_payload
    if len(mv) < total_needed:
        raise RvSchemaError(f"SHM slot buffer truncated: {len(mv)} < {total_needed}")
    payload = mv[SLOT_HEADER_SIZE:total_needed]
    rgb = np.frombuffer(payload[:rgb_nbytes], dtype=np.uint8).reshape((hdr.height, hdr.width, 3)).copy()
    depth = np.frombuffer(payload[rgb_nbytes:], dtype=np.float32).reshape((hdr.height, hdr.width)).copy()
    pose = np.asarray(hdr.pose_4x4, dtype=np.float32).reshape((4, 4))
    spec = RvOutputSpec(
        schema_version=hdr.schema_version,
        seq=hdr.seq,
        timestamp_s=hdr.timestamp_s,
        pose_4x4=pose,
        rgb=rgb,
        depth=depth,
        stale=bool(hdr.flags & RV_FLAG_STALE),
    )
    validate_rv_output(spec)
    return hdr, spec


def copy_slot_bytes(src: bytes | bytearray | memoryview, dst: bytearray | memoryview) -> int:
    """Copy-only non-blocking slot handoff (no unproject/packing). Returns bytes copied."""
    src_mv = memoryview(src)
    dst_mv = memoryview(dst)
    n = len(src_mv)
    if n > len(dst_mv):
        raise RvSchemaError(f"destination slot buffer too small: {len(dst_mv)} < {n}")
    dst_mv[:n] = src_mv
    return n


class DoubleBufferSlotLayout:
    """Bounded 2-slot double buffer with non-blocking copy-only write and drop on backpressure."""

    num_slots: int = 2

    def __init__(self, slot_bytes: int) -> None:
        cap = _require_non_bool_int(slot_bytes, "slot_bytes", minimum=SLOT_HEADER_SIZE)
        self.slot_bytes = cap
        self._slots: list[bytearray] = [bytearray(cap), bytearray(cap)]
        self._lengths: list[int] = [0, 0]
        self._seqs: list[int] = [-1, -1]
        self._latest_idx: int | None = None
        self._consumer_busy = False
        self._lock = threading.Lock()
        self.dropped: int = 0
        self.written: int = 0

    def set_consumer_busy(self, busy: bool) -> None:
        self._consumer_busy = bool(busy)

    def write_slot(self, data: bytes | bytearray | memoryview, *, seq: int) -> bool:
        """Non-blocking copy into the inactive slot. Drops and counts on backpressure."""
        if len(data) > self.slot_bytes:
            raise RvSchemaError(f"slot payload {len(data)} exceeds slot capacity {self.slot_bytes}")
        if not self._lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            if self._consumer_busy:
                self.dropped += 1
                return False
            write_idx = 0 if self._latest_idx is None else 1 - self._latest_idx
            copied = copy_slot_bytes(data, self._slots[write_idx])
            self._lengths[write_idx] = copied
            self._seqs[write_idx] = int(seq)
            self._latest_idx = write_idx
            self.written += 1
            return True
        finally:
            self._lock.release()

    def read_latest(self) -> bytes | None:
        with self._lock:
            if self._latest_idx is None:
                return None
            idx = self._latest_idx
            n = self._lengths[idx]
            return bytes(self._slots[idx][:n])


_PR_SET_PDEATHSIG = 1


def _load_linux_prctl() -> Any:
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.prctl
        fn.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        fn.restype = ctypes.c_int
        return fn
    except Exception:  # noqa: BLE001
        return None


_LINUX_PRCTL = _load_linux_prctl()


# Test-only hook. Never read from env or config. Production launch leaves this False.
_TEST_DISABLE_PDEATHSIG = False


def pdeathsig_preexec(parent_pid: int | None = None) -> Callable[[], None] | None:
    """Return a Linux preexec_fn setting PR_SET_PDEATHSIG(SIGTERM); None off Linux."""
    if _TEST_DISABLE_PDEATHSIG or not sys.platform.startswith("linux") or _LINUX_PRCTL is None:
        return None
    expected_ppid = os.getpid() if parent_pid is None else int(parent_pid)
    prctl_fn = _LINUX_PRCTL
    sig = int(signal.SIGTERM)

    def _preexec() -> None:
        prctl_fn(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
        if os.getppid() != expected_ppid:
            os._exit(0)

    return _preexec


class RvClient:
    """Supervisor-side handle to the separate Reality View child process.

    Spawns ``python -m python.viz.rv_renderer_stub`` in a separate OS process
    and performs only non-blocking copy-only slot handoff (no in-process render).
    """

    def __init__(
        self,
        slot_bytes: int = RV_DEFAULT_SLOT_BYTES,
        *,
        module: str = RV_RENDERER_MODULE,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.module = str(module)
        self.extra_args = tuple(str(a) for a in extra_args)
        self._ring = DoubleBufferSlotLayout(slot_bytes=slot_bytes)
        self._proc: subprocess.Popen[bytes] | None = None
        self.note: str = ""

    @property
    def child_pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def launch(self) -> int | None:
        if self.is_alive():
            return self.child_pid
        try:
            popen_kw: dict[str, Any] = {"close_fds": True}
            preexec = pdeathsig_preexec(os.getpid())
            if preexec is not None:
                popen_kw["preexec_fn"] = preexec
            self._proc = subprocess.Popen(
                [sys.executable, "-m", self.module, "--serve", "--parent-pid", str(os.getpid()), *self.extra_args],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **popen_kw,
            )
            self.note = f"reality-view: separate stub process pid={self._proc.pid} (placeholder active until live depth)"
            return self._proc.pid
        except OSError as exc:
            self._proc = None
            self.note = f"reality-view: stub launch failed ({exc!r}); using placeholder"
            return None

    def handoff_slot(self, slot_bytes: bytes | bytearray | memoryview, *, seq: int) -> bool:
        """Non-blocking copy-only handoff after actuator.apply; drops on backpressure."""
        if not self.is_alive():
            return False
        try:
            return self._ring.write_slot(slot_bytes, seq=seq)
        except Exception:  # noqa: BLE001 - viz never breaks supervisor
            return False

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            except OSError:
                pass
