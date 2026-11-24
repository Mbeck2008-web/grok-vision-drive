#!/usr/bin/env python3
"""Collector speed work (Michael, 2026-11-24): profile, shared memory, parallel fetch, background writer.

* item 0: scripts/profile_collect_tick.py times one stub tick per stage.
* item 1: cameras stream through BeamNG.tech shared memory; --cam-transport socket is the fallback.
* item 2: --cam-fetch parallel reads/decodes shared-memory cameras on a thread pool; the tick
  still holds one reading per camera with its own capture window, bit-identical to serial.
  Socket cameras never run in parallel (one BeamNGpy socket).
* item 4: the disk writer is a bounded background queue; a write or close error fails the run,
  every queued tick is flushed on exit, and the files are byte-identical to a synchronous write.
No Hz gain is claimed here: that needs the IQU45 live check.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


# ----------------------------------------------------------------- fakes

class _ShmCam:
    """stream_raw like BeamNGpy 1.36: a shared-memory copy, no GE request."""

    def __init__(self, cid: str, w: int = 64, h: int = 48, delay_s: float = 0.0, gt: bool = True, seed: int = 0) -> None:
        self.name = f"gvd_collect_{cid}"
        self.resolution = (w, h)
        self.is_render_depth = gt
        self.is_render_annotations = gt
        self.near_far_planes = (0.05, 300.0)
        self.gvd_sem_palette = {"road": (128, 64, 128), "car": (0, 0, 142)}
        self.delay_s = delay_s
        rng = np.random.default_rng(seed)
        self._colour = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        self._depth = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
        ann = np.zeros((h, w, 4), dtype=np.uint8)
        ann[..., :3] = (128, 64, 128)
        ann[: h // 3, :, :3] = (0, 0, 142)
        self._ann = ann
        self.captured: list[float] = []
        self.threads: set[str] = set()

    def stream_raw(self):
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay_s)
        self.captured.append(time.monotonic())
        raw = {"colour": self._colour.copy()}
        if self.is_render_depth:
            raw["depth"] = self._depth.copy()
        if self.is_render_annotations:
            raw["annotation"] = self._ann.copy()
        return raw


class _SocketCam(_ShmCam):
    """Socket fallback: poll_raw is a GE round trip on the one shared socket."""

    live = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.gvd_transport = "socket"
        self.stream_calls = 0

    def stream_raw(self):  # must not be used on the socket transport
        self.stream_calls += 1
        return {}

    def poll_raw(self):
        with _SocketCam.lock:
            _SocketCam.live += 1
            _SocketCam.peak = max(_SocketCam.peak, _SocketCam.live)
        try:
            return _ShmCam.stream_raw(self)
        finally:
            with _SocketCam.lock:
                _SocketCam.live -= 1


CIDS = ("main", "narrow", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")


def _rig(cls=_ShmCam, delay_s: float = 0.0) -> dict:
    return {cid: cls(cid, delay_s=delay_s, seed=i, gt=(cid != "rear" or True)) for i, cid in enumerate(CIDS)}


# ----------------------------------------------------------------- item 1

def check_shared_memory_default_and_socket_fallback() -> None:
    import scripts.collect_scene_lanes as collect
    import test_labels_v4_recorder as v4
    from python.sensors.cameras import load_camera_config

    base = {"is_using_shared_memory": False, "is_streaming": True, "resolution": (8, 8)}
    shm = collect._camera_kwargs_for_transport(base, "shmem")
    assert shm["is_using_shared_memory"] is True and shm["is_streaming"] is True, shm
    sock = collect._camera_kwargs_for_transport(base, "socket")
    assert sock["is_using_shared_memory"] is False and sock["is_streaming"] is False, sock
    assert collect.build_parser().parse_args([]).cam_transport == "shmem"
    assert collect.build_parser().parse_args(["--cam-transport", "socket"]).cam_transport == "socket"

    beamngpy, sensors, made = v4._fake_modules()
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"], sys.modules["beamngpy.sensors"] = beamngpy, sensors
    prev = collect._CAM_TRANSPORT["mode"]
    try:
        cfg = load_camera_config()

        class _B:
            class camera:  # noqa: N801
                @staticmethod
                def get_annotations():
                    return {}

        for mode, want in (("shmem", True), ("socket", False)):
            made.clear()
            collect._CAM_TRANSPORT["mode"] = mode
            cams = collect._attach_cameras(_B(), object(), cfg)
            assert set(CIDS) <= set(cams), sorted(cams)
            for cid in CIDS:
                kw = made[f"gvd_collect_{cid}"]
                assert kw["is_using_shared_memory"] is want and kw["is_streaming"] is want, (mode, cid, kw)
                assert getattr(cams[cid], "gvd_transport") == mode
            # resolution and depth/sem untouched by the transport
            for spec in cfg["cameras"]:
                if spec.get("id") in CIDS:
                    kw = made[f"gvd_collect_{spec['id']}"]
                    assert tuple(kw["resolution"]) == tuple(int(v) for v in spec.get("live_res") or [640, 480])
                    assert kw["is_render_depth"] is bool(spec.get("train_gt"))
    finally:
        collect._CAM_TRANSPORT["mode"] = prev
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # socket cameras are read with poll_raw, never stream_raw
    cam = _SocketCam("main")
    image, reason, raw = collect._read_camera("main", cam, {})
    assert image is not None and reason == "" and cam.stream_calls == 0, (reason, cam.stream_calls)
    print("cameras: shared memory by default, --cam-transport socket polls (no shmem, no stream)")


# ----------------------------------------------------------------- item 2

def _poll(collect, cams, parallel: bool):
    t0 = time.perf_counter()
    frames, stamps, reasons = collect._poll_cameras(cams, parallel=parallel)
    spent = time.perf_counter() - t0
    gt = collect.take_train_gt()
    return frames, stamps, reasons, gt, spent


def check_parallel_fetch_same_tick_and_identical() -> None:
    import scripts.collect_scene_lanes as collect

    assert collect.build_parser().parse_args([]).cam_fetch == "parallel"
    serial = _poll(collect, _rig(delay_s=0.04), parallel=False)
    cams = _rig(delay_s=0.04)
    par = _poll(collect, cams, parallel=True)
    # bit-identical frames and GT, same cameras, same order
    assert list(serial[0]) == list(par[0]) == list(CIDS), (list(serial[0]), list(par[0]))
    for cid in CIDS:
        assert serial[0][cid].dtype == par[0][cid].dtype and np.array_equal(serial[0][cid], par[0][cid]), cid
    assert sorted(serial[3]) == sorted(par[3]) and serial[3], sorted(par[3])
    for key in serial[3]:
        a, b = serial[3][key], par[3][key]
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b), key
        else:
            assert a == b, key
    assert serial[2] == par[2] == {}
    # really concurrent: worker threads, overlapping windows, faster than serial
    assert all(any(n.startswith("gvd-cam-fetch") for n in c.threads) for c in cams.values()), [c.threads for c in cams.values()]
    windows = [(par[1][c]["mono_before"], par[1][c]["mono_after"]) for c in CIDS]
    assert max(w[0] for w in windows) < min(w[1] for w in windows), "camera windows do not overlap"
    assert par[4] < 0.5 * serial[4], (par[4], serial[4])
    # exact timestamps: each camera's capture sits inside its own window; one reading per camera
    for cid, cam in cams.items():
        assert len(cam.captured) == 1, (cid, cam.captured)
        st = par[1][cid]
        assert st["mono_before"] <= cam.captured[0] <= st["mono_after"], (cid, st, cam.captured)
        assert st["call_s"] is not None and st["call_s"] >= 0.035, (cid, st)
    # same tick: every reading lands inside the one _poll_cameras call
    span = max(w[1] for w in windows) - min(w[0] for w in windows)
    assert span < par[4] + 1e-3, (span, par[4])
    print(f"parallel fetch: 8 cams bit-identical to serial, {par[4]*1000:.0f} ms vs {serial[4]*1000:.0f} ms "
          f"(stub, 40 ms per read), windows overlap, stamps exact")


def check_socket_cameras_stay_serial() -> None:
    import scripts.collect_scene_lanes as collect

    _SocketCam.live = _SocketCam.peak = 0
    cams = _rig(cls=_SocketCam, delay_s=0.01)
    assert collect.parallel_safe(cams) is False
    frames, stamps, reasons, _gt, _s = _poll(collect, cams, parallel=True)
    assert list(frames) == list(CIDS) and not reasons, reasons
    assert _SocketCam.peak == 1, _SocketCam.peak
    assert all(not any(n.startswith("gvd-cam-fetch") for n in c.threads) for c in cams.values())
    # one socket camera in the rig is enough to keep the whole tick serial
    mixed = _rig(delay_s=0.0)
    mixed["rear"] = _SocketCam("rear")
    assert collect.parallel_safe(mixed) is False
    print("socket transport: never polled in parallel (one BeamNGpy socket)")


def check_parallel_gt_failure_is_counted_once_per_camera() -> None:
    import python.data.raw_frames as raw_frames
    import scripts.collect_scene_lanes as collect

    real = raw_frames.train_gt_from_raw

    def boom(cid, cam, raw):
        if cid == "wide":
            raise ValueError("bad depth")
        return real(cid, cam, raw)

    raw_frames.train_gt_from_raw = boom
    before = collect.train_gt_failures()
    try:
        frames, _st, reasons, gt, _s = _poll(collect, _rig(), parallel=True)
    finally:
        raw_frames.train_gt_from_raw = real
    assert collect.train_gt_failures() == before + 1
    assert "wide" in frames and not reasons  # colour kept
    assert not any(k.startswith("wide_") for k in gt) and any(k.startswith("main_") for k in gt), sorted(gt)
    print("parallel fetch: a GT decode error drops that camera's GT only, colour kept, counted")


# ----------------------------------------------------------------- item 4

def _ticks(n: int = 12):
    from python.debug.lane_targets import tech_from_hit

    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        frames = {
            "main": rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8),
            "rear": rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8),
            "main_depth": rng.integers(0, 60000, size=(24, 32), dtype=np.uint16),
            "main_sem": rng.integers(0, 20, size=(24, 32), dtype=np.uint8),
        }
        kwargs = {
            "t": 0.1 * i,
            "driver": "gvd",
            "ego": {"speed_mps": 4.0 + i, "yaw_rate": 0.01, "pos": [1.0, 2.0, 3.0], "dir": [0.0, 1.0, 0.0]},
            "wheel": {"steer": -0.2},
            "pedals": {"throttle": 0.15, "brake": 0.0},
            "tech": tech_from_hit(0.1, 0.0, 3.5, False),
            "tags": {"gt_stride": 1, "i": i},
        }
        out.append((frames, kwargs))
    return out


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _cams():
    return [{"id": "main"}, {"id": "rear"}]


def check_background_writer_byte_identical_to_sync() -> None:
    from python.data.raw_frames import RawFrameWriter, encode_cameras
    from python.data.raw_queue import RawWriteQueue

    ticks = _ticks()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sync = RawFrameWriter(Path(a), _cams())
        for frames, kwargs in ticks:
            sync.write_encoded(encode_cameras(frames), **kwargs)
        sync.close()
        queue = RawWriteQueue(RawFrameWriter(Path(b), _cams()), max_depth=64, workers=4)
        for frames, kwargs in ticks:
            assert queue.submit(frames, **kwargs) is True
        report = queue.close()
        assert report["ok"] and report["written"] == len(ticks) and report["unflushed"] == 0, report
        ta, tb = _tree(Path(a)), _tree(Path(b))
        assert sorted(ta) == sorted(tb), (sorted(set(ta) ^ set(tb)))
        diff = [k for k in ta if ta[k] != tb[k]]
        assert not diff, diff
        assert any(k.endswith(".png") for k in ta) and any("state" in k for k in ta), sorted(ta)[:8]
    print(f"background writer: {len(ta)} files byte-identical to the synchronous writer")


def check_background_writer_bounded_and_never_blocks() -> None:
    from python.data.raw_queue import RawWriteQueue

    class _Hold:
        def __init__(self) -> None:
            self.release = threading.Event()
            self.n = 0

        def write(self, frames, **kwargs) -> None:
            self.release.wait(5.0)
            self.n += 1

    hold = _Hold()
    queue = RawWriteQueue(hold, max_depth=4, workers=2)
    try:
        img = np.ones((8, 8, 3), dtype=np.uint8)
        spent = []
        results = []
        for i in range(20):
            t0 = time.perf_counter()
            results.append(queue.submit({"main": img}, t=float(i)))
            spent.append(time.perf_counter() - t0)
        stats = queue.stats()
        assert stats["queue_depth"] <= 4 and stats["queue_peak"] <= 4, stats
        assert results.count(True) == 4 and stats["dropped_frames"] == 16, (results, stats)
        assert max(spent) < 0.05, max(spent)  # the capture thread never waits on disk
        # byte bound too
        small = RawWriteQueue(hold, max_depth=100, max_bytes=img.nbytes * 2, workers=1)
        try:
            ok = [small.submit({"main": img}, t=float(i)) for i in range(5)]
            assert ok.count(True) <= 2, ok
        finally:
            hold.release.set()
            small.close()
    finally:
        hold.release.set()
        report = queue.close()
    assert report["written"] == 4 and report["dropped"] == 16 and report["unflushed"] == 0, report
    print("background writer: bounded by count and bytes; a full queue drops and counts, submit never blocks")


def check_background_writer_errors_are_loud_and_exit_flushes() -> None:
    from python.data.raw_queue import RawWriteQueue

    class _Slow:
        def __init__(self) -> None:
            self.n = 0

        def write(self, frames, **kwargs) -> None:
            time.sleep(0.02)
            self.n += 1

    slow = _Slow()
    queue = RawWriteQueue(slow, max_depth=64, workers=2)
    img = np.ones((8, 8, 3), dtype=np.uint8)
    for i in range(25):
        assert queue.submit({"main": img}, t=float(i))
    report = queue.close()  # no wait_empty first: close itself flushes
    assert slow.n == 25 and report == {**report, "ok": True, "written": 25, "unflushed": 0}, (slow.n, report)
    assert queue.close() == report  # idempotent

    class _Fail:
        def write(self, frames, **kwargs) -> None:
            if kwargs.get("t") == 3.0:
                raise OSError("disk full")

    queue = RawWriteQueue(_Fail(), max_depth=64, workers=1)
    for i in range(6):
        queue.submit({"main": img}, t=float(i))
    report = queue.close()
    assert report["ok"] is False and report["failed"] == 1 and "disk full" in str(report["error"]), report

    class _CloseFails:
        def write(self, frames, **kwargs) -> None:
            return None

        def close(self) -> None:
            raise OSError("flush failed")

    queue = RawWriteQueue(_CloseFails(), max_depth=8, workers=1)
    queue.submit({"main": img}, t=0.0)
    report = queue.close()
    assert report["ok"] is False and "flush failed" in str(report["error"]), report

    class _Stuck:
        def write(self, frames, **kwargs) -> None:
            time.sleep(2.0)

    queue = RawWriteQueue(_Stuck(), max_depth=8, workers=1)
    for i in range(3):
        queue.submit({"main": img}, t=float(i))
    report = queue.close(timeout_s=0.3)
    assert report["ok"] is False and report["unflushed"] >= 1, report
    print("background writer: close flushes every tick; write, close and timeout failures report ok=False")


def check_collector_fails_loudly_when_exit_flush_fails() -> None:
    """A writer that fails only at close: the run used to exit 0; now writer_failed, exit 2."""
    import python.data.raw_frames as raw_frames
    import test_recorder_v2 as rv2

    real_close = raw_frames.RawFrameWriter.close

    def bad_close(self):
        real_close(self)
        raise OSError("final fsync failed")

    raw_frames.RawFrameWriter.close = bad_close
    try:
        code, text, _out, _made, root_cm = rv2._run_fake_collect([], stop_at=12)
    finally:
        raw_frames.RawFrameWriter.close = real_close
    try:
        assert code == 2, (code, text[-1500:])
        assert "collect stop writer_failed flush on exit" in text and "final fsync failed" in text, text[-1500:]
    finally:
        root_cm.__exit__(None, None, None)
    import scripts.collect_scene_lanes as collect

    collect._FETCH_POOL.update({"pool": None, "n": 0})
    code, text, out, _made, root_cm = rv2._run_fake_collect([], stop_at=12)
    try:
        assert code == 0 and "writer_failed" not in text, text[-1500:]
        assert "collect cameras transport=shmem fetch=parallel" in text, text[:1500]
        # the run's own ticks went through the fetch pool (8 shared-memory cameras)
        assert collect._FETCH_POOL["pool"] is not None and collect._FETCH_POOL["n"] >= 8, collect._FETCH_POOL
    finally:
        root_cm.__exit__(None, None, None)
    print("collector: an exit flush failure stops the run with writer_failed (exit 2); a clean run exits 0")


# ----------------------------------------------------------------- item 0

def check_profile_script_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "profile.json"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "profile_collect_tick.py"), "--scale", "0.125", "--ticks", "3",
             "--read-ms", "5", "--json", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        data = json.loads(out.read_text(encoding="utf-8"))
    for mode in ("serial", "parallel"):
        stages = data[mode]["stages_ms"]
        for key in ("fetch_total", "submit", "tick_total"):
            assert key in stages and stages[key] >= 0.0, (mode, key, stages)
        per = data[mode]["per_cam_ms"]
        assert set(CIDS) <= set(per), sorted(per)
        for cid in CIDS:
            assert per[cid]["read"] >= 4.0 and per[cid]["colour_decode"] >= 0.0 and "gt_decode" in per[cid], (cid, per[cid])
    assert data["writer"]["encode_ms"] > 0.0 and data["writer"]["commit_ms"] >= 0.0, data["writer"]
    assert data["note"].startswith("stub"), data["note"]
    print("profile_collect_tick: per-stage and per-camera times logged (stub, not a live Hz claim)")


def main() -> None:
    check_shared_memory_default_and_socket_fallback()
    check_parallel_fetch_same_tick_and_identical()
    check_socket_cameras_stay_serial()
    check_parallel_gt_failure_is_counted_once_per_camera()
    check_background_writer_byte_identical_to_sync()
    check_background_writer_bounded_and_never_blocks()
    check_background_writer_errors_are_loud_and_exit_flushes()
    check_collector_fails_loudly_when_exit_flush_fails()
    check_profile_script_runs()
    print("test_collect_perf: OK")


if __name__ == "__main__":
    main()
