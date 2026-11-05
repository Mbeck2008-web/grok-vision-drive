#!/usr/bin/env python3
"""Training-only depth / semantic ground truth from the main camera.

Opt-in per camera (``train_gt: true`` in cameras.yaml). Off by default, so a
recording without it is unchanged. The files land in ``raw/main_depth``
(uint16 cm PNG) and ``raw/main_sem`` (uint8 class-id PNG), tagged
``train_only`` in state.jsonl, and no driving / frame-cache camera list
ever returns them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

import scripts.collect_scene_lanes as collect  # noqa: E402
from python.data.raw_frames import (  # noqa: E402
    RawFrameWriter,
    cameras_for_row,
    is_train_only_cam,
    list_raw_images,
    train_gt_from_raw,
)
from python.sensors.cameras import beamng_camera_sensor_kwargs  # noqa: E402
from python.train.cam_strip import parse_cams  # noqa: E402

W, H = 8, 6
FAR_M = 50.0
PALETTE = {"CAR": (0, 0, 142), "ROAD": (128, 64, 128), "SKY": (70, 130, 180)}


def _kwargs(**extra: object) -> dict:
    return beamng_camera_sensor_kwargs(
        pos=(0.0, 0.0, 1.0), direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0), fov_v=50.0,
        resolution=(W, H), update_s=0.1, near_m=0.05, far_m=FAR_M, shmem=True,
        streaming=True, rgb_only=True, cid="main", **extra,
    )


def check_kwargs_default_off() -> None:
    base = _kwargs()
    assert base == _kwargs(train_gt=False)
    assert base["is_render_depth"] is False and base["is_render_annotations"] is False
    assert "integer_depth" not in base
    on = _kwargs(train_gt=True)
    assert on["is_render_depth"] is True and on["is_render_annotations"] is True
    assert on["is_render_instance"] is False and on["integer_depth"] is False
    print("kwargs default off, train_gt on")


class _StubCam:
    is_render_depth = True
    is_render_annotations = True
    resolution = (W, H)
    near_far_planes = (0.05, FAR_M)
    gvd_sem_palette = PALETTE

    def __init__(self, raw: dict) -> None:
        self.raw = raw

    def stream_raw(self) -> dict:
        return self.raw


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((H, W), 0.2, dtype=np.float32)  # 0.2 * 50 m = 10 m
    depth[0, 0] = 1.0  # 50 m
    depth[1, 1] = 14.0  # 700 m clamps to 65535 cm
    ann = np.zeros((H, W, 4), dtype=np.uint8)
    ann[..., :3] = PALETTE["ROAD"]
    ann[0, :, :3] = PALETTE["SKY"]
    ann[2, 2, :3] = (1, 2, 3)  # unknown colour -> 0
    colour = np.zeros((H, W, 3), dtype=np.uint8)
    colour[..., 1] = 90
    return depth, ann, colour


def _expect(gt: dict) -> None:
    assert set(gt) == {"main_depth", "main_sem"}, sorted(gt)
    d = gt["main_depth"]
    assert d.dtype == np.uint16 and d.shape == (H, W)
    assert int(d[3, 3]) == 1000 and int(d[0, 0]) == 5000 and int(d[1, 1]) == 65535
    s = gt["main_sem"]
    assert s.dtype == np.uint8 and s.shape == (H, W)
    ids = {name: i + 1 for i, name in enumerate(sorted(PALETTE))}
    assert int(s[3, 3]) == ids["ROAD"] and int(s[0, 5]) == ids["SKY"] and int(s[2, 2]) == 0


def check_gt_from_arrays_and_bytes() -> None:
    depth, ann, colour = _arrays()
    _expect(train_gt_from_raw("main", _StubCam({}), {"colour": colour, "depth": depth, "annotation": ann}))
    _expect(train_gt_from_raw("main", _StubCam({}), {"depth": depth.tobytes(), "annotation": ann.tobytes()}))

    class _Off(_StubCam):
        is_render_depth = False
        is_render_annotations = False

    assert train_gt_from_raw("main", _Off({}), {"depth": depth, "annotation": ann}) == {}
    assert train_gt_from_raw("main", _StubCam({}), None) == {}
    print("gt arrays and raw bytes decode")


def check_poll_keeps_gt_out_of_frames() -> None:
    depth, ann, colour = _arrays()
    cam = _StubCam({"colour": colour, "depth": depth, "annotation": ann})
    frames, _stamps, reasons = collect._poll_cameras({"main": cam})
    assert set(frames) == {"main"} and reasons == {}, (frames.keys(), reasons)
    gt = collect.take_train_gt()
    _expect(gt)
    assert collect.take_train_gt() == {}
    print("poll frames camera-only, gt taken separately")


def _write(root: Path, gt: dict | None, palette: dict | None = None) -> list[dict]:
    _d, _a, colour = _arrays()
    writer = RawFrameWriter(root, [])
    for i in range(2):
        frames = {"main": colour}
        if gt:
            frames.update(gt)
        writer.write(frames, t=float(i), timestamps={"main": 1.0 + i}, tech={"lane": i})
    if palette is not None:
        from python.data.raw_frames import write_sem_classes

        write_sem_classes(root, palette)
    return [json.loads(x) for x in (root / "state.jsonl").read_text().splitlines()]


def check_writer_and_exclusions() -> None:
    depth, ann, _c = _arrays()
    gt = train_gt_from_raw("main", _StubCam({}), {"depth": depth, "annotation": ann})
    with tempfile.TemporaryDirectory() as tmp:
        plain = _write(Path(tmp) / "plain", None)
        for row in plain:
            assert "train_only" not in json.dumps(row), row
        root = Path(tmp) / "gt"
        rows = _write(root, gt, PALETTE)
        for row in rows:
            assert row["cams"]["main_depth"]["train_only"] is True
            assert row["cams"]["main_sem"]["train_only"] is True
            assert "train_only" not in row["cams"]["main"]
            assert cameras_for_row(row) == ["main"]
        d = cv2.imread(str(root / rows[0]["cams"]["main_depth"]["file"]), cv2.IMREAD_UNCHANGED)
        assert d.dtype == np.uint16 and np.array_equal(d, gt["main_depth"])
        s = cv2.imread(str(root / rows[0]["cams"]["main_sem"]["file"]), cv2.IMREAD_UNCHANGED)
        assert s.dtype == np.uint8 and np.array_equal(s, gt["main_sem"])
        classes = json.loads((root / "sem_classes.json").read_text())
        assert classes["classes"]["1"] == "CAR" and classes["classes"]["0"] == "unknown"
        listed = list_raw_images(root)
        assert listed and all(p.parent.name == "main" or p.parent.parent.name == "main" for p in listed), listed
        # Plain recording bytes do not depend on the feature existing.
        assert (Path(tmp) / "plain" / "state.jsonl").read_bytes().count(b"train_only") == 0
    for name in ("main_depth", "main_sem"):
        assert is_train_only_cam(name)
        try:
            parse_cams(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_cams accepted {name}")
    assert not is_train_only_cam("main")
    print("writer tags train_only; listings exclude gt folders")


def check_collect_attach_flips_with_yaml() -> None:
    """collect's own ``_attach_cameras`` honours ``train_gt`` and attaches the palette."""
    import types

    made: dict[str, dict] = {}

    class _Cam:
        def __init__(self, name, bng, vehicle, **kwargs):
            made[name] = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake = types.ModuleType("beamngpy.sensors")
    fake.Camera = _Cam
    saved = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules.setdefault("beamngpy", types.ModuleType("beamngpy"))
    sys.modules["beamngpy.sensors"] = fake

    class _Api:
        def get_annotations(self):
            return dict(PALETTE)

    bng = types.SimpleNamespace(camera=_Api())
    spec = {"id": "main", "pos_m": [0, 1.2, 1.26], "fov_v": 34.0, "live_res": [W, H], "far_m": 300}
    try:
        off = collect._attach_cameras(bng, object(), {"cameras": [dict(spec)]})
        on = collect._attach_cameras(bng, object(), {"cameras": [dict(spec, train_gt=True)]})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert off["main"].is_render_depth is False and off["main"].is_render_annotations is False
    assert not getattr(off["main"], "gvd_sem_palette", None)
    assert on["main"].is_render_depth is True and on["main"].is_render_annotations is True
    assert on["main"].integer_depth is False
    assert on["main"].gvd_sem_palette == PALETTE
    print("collect attach flips with train_gt yaml")


def check_gt_failure_is_soft() -> None:
    """A broken GT decode logs once with a count; the colour frame still comes back."""
    import contextlib
    import io

    _d, _a, colour = _arrays()
    cam = _StubCam({"colour": colour, "depth": "garbage", "annotation": object()})
    cam.resolution = "bad"  # makes train_gt_from_raw raise inside
    real = collect.train_gt_from_raw if hasattr(collect, "train_gt_from_raw") else None
    import python.data.raw_frames as rf

    def boom(*a, **k):
        raise ValueError("decode broke")

    orig = rf.train_gt_from_raw
    rf.train_gt_from_raw = boom
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for _ in range(3):
                frames, _s, reasons = collect._poll_cameras({"main": cam})
                assert set(frames) == {"main"} and reasons == {}
                assert collect.take_train_gt() == {}
    finally:
        rf.train_gt_from_raw = orig
    text = buf.getvalue()
    assert text.count("train_gt decode failed") == 1, text
    assert collect.train_gt_failures() >= 3
    _ = real
    print("gt decode failure is soft, one warning, counted")


def main() -> None:
    check_kwargs_default_off()
    check_gt_from_arrays_and_bytes()
    check_poll_keeps_gt_out_of_frames()
    check_writer_and_exclusions()
    check_collect_attach_flips_with_yaml()
    check_gt_failure_is_soft()
    print("sensor gt ok")


if __name__ == "__main__":
    main()
