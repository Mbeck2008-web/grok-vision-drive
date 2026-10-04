#!/usr/bin/env python3
"""Scene strip recorder and scene net. No BeamNG and no GPU.

The net check is skipped when PyTorch is not installed. Image size, the state
line, the variable rate, and mixed-precision selection still run.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.data.strip_writer import (  # noqa: E402
    CANVAS_H,
    CANVAS_W,
    GAP_PX,
    SECTORS,
    STRIP_ORDER,
    StripTimestampError,
    StripWriter,
    compose_strip,
    read_state_jsonl,
)
from python.planning.path_predictor import predict_path  # noqa: E402
from python.train.fit import (  # noqa: E402
    TrainFitError,
    USABLE_DEN,
    USABLE_NUM,
    canvas_bytes,
    fit_with_probe,
    measure_driving_file,
    plan_fit,
    read_free_ram_bytes,
    read_free_vram_bytes,
    shrink_batch,
    step_bytes,
    usable_bytes,
)
from python.train.scene_net import (  # noqa: E402
    LOSS_TERMS,
    MEMORY_S,
    OUTPUT_FIELDS,
    PREDICT_PATH_ARGS,
    ScenePrediction,
    amp_enabled,
    assign_min_cost,
    labels_from_tech,
    letterbox_rect,
    loss_mask,
    memory_windows,
    moment_ok,
    precision_for_capability,
    scene_loss,
    scene_to_predict_kwargs,
    step_dt,
)
from python.train.status_window import (  # noqa: E402
    TrainStatus,
    TrainWindow,
    format_status,
    predict_eta_s,
    steps_per_second,
)
from python.train.train_scene import train_directory  # noqa: E402


def _colors() -> dict[str, tuple[int, int, int]]:
    out = {}
    for i, cam_id in enumerate(STRIP_ORDER):
        out[cam_id] = (20 + i * 30, 40 + i * 20, 15 + i * 28)
    return out


def _frame(color: tuple[int, int, int], cam_id: str) -> np.ndarray:
    width = 80 if cam_id in ("wide", "rear") else 60
    img = np.zeros((36, width, 3), dtype=np.uint8)
    img[:] = color
    return img


def _bundle(colors: dict[str, tuple[int, int, int]], *, skip: set[str] | None = None) -> dict[str, np.ndarray]:
    skip = skip or set()
    return {cam_id: _frame(color, cam_id) for cam_id, color in colors.items() if cam_id not in skip}


def _stamps(frames: dict, t: float, **extra: float) -> dict[str, float]:
    stamps = {cam_id: t for cam_id in frames}
    stamps.update(extra)
    return stamps


def _controls() -> dict:
    return {
        "ego": {"speed_mps": 8.0},
        "wheel": {"steer": 0.05},
        "pedals": {"throttle": 0.2, "brake": 0.0},
    }


def _read_jpeg(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, path
    return bgr


def _close(pixel: np.ndarray, color: tuple[int, int, int], tol: int = 12) -> bool:
    return all(abs(int(pixel[i]) - int(color[i])) <= tol for i in range(3))


def check_canvas_geometry() -> None:
    assert CANVAS_H == 192
    assert CANVAS_W == 2272
    assert CANVAS_W == 6 * 256 + 2 * 340 + 7 * GAP_PX
    assert [sec.cam_id for sec in SECTORS] == list(STRIP_ORDER)
    assert GAP_PX == 8
    wide = {sec.cam_id for sec in SECTORS if sec.wide}
    assert wide == {"wide", "rear"}
    for sec in SECTORS:
        assert sec.x1 - sec.x0 == sec.width
        assert sec.width == (340 if sec.wide else 256)
    for left, right in zip(SECTORS, SECTORS[1:]):
        assert right.x0 - left.x1 == GAP_PX
    nw, nh, x0, y0 = letterbox_rect(340, 192, 256, 192)
    assert (nw, nh, x0, y0) == (256, 145, 0, 23), (nw, nh, x0, y0)


def check_variable_rate_and_one_strip() -> None:
    colors = _colors()
    times = [10.0, 10.01, 10.08, 10.5, 12.5]
    with tempfile.TemporaryDirectory() as tmp:
        writer = StripWriter(tmp)
        root = Path(tmp)
        for i, t in enumerate(times):
            frames = _bundle(colors)
            tech = {"note": "labels"} if i == 0 else None
            rec = writer.write(
                frames,
                t=t,
                timestamps=_stamps(frames, t),
                ego=_controls()["ego"],
                wheel=_controls()["wheel"],
                pedals=_controls()["pedals"],
                tech=tech,
            )
            assert rec.index == i
            bgr = _read_jpeg(rec.path)
            assert bgr.shape == (192, 2272, 3), bgr.shape
        jpegs = sorted((root / "strips").glob("*.jpg"))
        assert len(jpegs) == len(times)
        assert not list((root / "strips").glob("*.tmp"))
        rows = read_state_jsonl(root / "state.jsonl")
        assert len(rows) == len(times)
        assert rows[0]["dt_s"] is None
        assert rows[0]["t"] == 10.0
        assert abs(rows[1]["dt_s"] - 0.01) < 1e-6
        assert abs(rows[2]["dt_s"] - 0.07) < 1e-6
        assert abs(rows[3]["dt_s"] - 0.42) < 1e-6
        assert abs(rows[4]["dt_s"] - 2.0) < 1e-6
        assert "tech" in rows[0] and rows[0]["tech"]["note"] == "labels"
        assert "tech" not in rows[1]
        for row in rows:
            assert set(row) >= {"i", "t", "dt_s", "ego", "wheel", "pedals"}
            assert "steer" not in row
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        assert meta["canvas"] == [2272, 192]
        assert meta["fixed_hz"] is None
        assert meta["n_strips"] == len(times)
        assert meta["gap_px"] == 8
        # A fast burst is kept. Nothing is invented to fill a slower rate.
        assert len(rows) == 5


def check_timestamp_refusal_and_empty_sector() -> None:
    colors = _colors()
    with tempfile.TemporaryDirectory() as tmp:
        writer = StripWriter(tmp)
        root = Path(tmp)
        first = _bundle(colors)
        writer.write(first, t=1.0, timestamps=_stamps(first, 1.0), **_controls())
        mixed = _bundle(colors)
        stamps = _stamps(mixed, 2.0)
        stamps["wide"] = 2.5
        before = read_state_jsonl(root / "state.jsonl")
        try:
            writer.write(mixed, t=2.0, timestamps=stamps, **_controls())
        except StripTimestampError:
            pass
        else:
            raise AssertionError("mixed timestamps were accepted")
        after = read_state_jsonl(root / "state.jsonl")
        assert after == before
        assert not (root / "strips" / "000001.jpg").exists()

        # A missed camera is an empty sector even if a stale timestamp is still listed.
        nxt = _bundle(colors, skip={"wide"})
        stale = _stamps(nxt, 1.2)
        stale["wide"] = 0.2
        for cam_id in nxt:
            nxt[cam_id] = _frame((255, 255, 255), cam_id)
        rec = writer.write(nxt, t=1.2, timestamps=stale, **_controls())
        assert abs((rec.dt_s or 0.0) - 0.2) < 1e-6
        bgr = _read_jpeg(rec.path)
        wide = next(sec for sec in SECTORS if sec.cam_id == "wide")
        inset = bgr[16:-16, wide.x0 + 16 : wide.x1 - 16]
        assert int(inset.max()) <= 3, int(inset.max())
        main = next(sec for sec in SECTORS if sec.cam_id == "main")
        center = bgr[CANVAS_H // 2, (main.x0 + main.x1) // 2]
        assert _close(center, (255, 255, 255))
        # The first strip's wide colour is not still sitting in this sector.
        first_bgr = _read_jpeg(root / "strips" / "000000.jpg")
        old = first_bgr[CANVAS_H // 2, (wide.x0 + wide.x1) // 2]
        assert not _close(old, (0, 0, 0), tol=3)
        assert int(inset.mean()) < 2


def check_sectors_and_ego_edge() -> None:
    colors = _colors()
    with tempfile.TemporaryDirectory() as tmp:
        writer = StripWriter(tmp)
        frames = _bundle(colors)
        rec = writer.write(frames, t=3.0, timestamps=_stamps(frames, 3.0), **_controls())
        raw = compose_strip(frames)
        assert raw.shape == (CANVAS_H, CANVAS_W, 3)
        for left, right in zip(SECTORS, SECTORS[1:]):
            gap = raw[:, left.x1 : right.x0]
            assert gap.shape[1] == GAP_PX
            assert int(gap.max()) == 0
        repeat_l = SECTORS[0]
        assert int(raw[:, repeat_l.x0 : repeat_l.x0 + 32].max()) == 0
        repeat_r = next(sec for sec in SECTORS if sec.cam_id == "repeatR")
        assert int(raw[:, repeat_r.x1 - 32 : repeat_r.x1].max()) == 0
        bgr = _read_jpeg(rec.path)
        assert bgr.shape == (CANVAS_H, CANVAS_W, 3)
        for sec in SECTORS:
            x = (sec.x0 + sec.x1) // 2
            assert _close(bgr[CANVAS_H // 2, x], colors[sec.cam_id]), (sec.cam_id, bgr[CANVAS_H // 2, x])


def check_rate_helpers_and_precision() -> None:
    assert precision_for_capability(6.1) == "fp32"
    assert precision_for_capability((6, 1)) == "fp32"
    assert amp_enabled(6.1) is False
    assert precision_for_capability(7.0) == "amp"
    assert precision_for_capability((7, 0)) == "amp"
    assert precision_for_capability(7.5) == "amp"
    assert precision_for_capability((8, 6)) == "amp"
    assert precision_for_capability(None) == "fp32"
    assert precision_for_capability(5.0) == "fp32"
    assert step_dt(None) == 0.0
    assert step_dt(0.07) == 0.07

    fast = [{"t": i * 0.05} for i in range(100)]
    windows = memory_windows(fast, MEMORY_S)
    assert sum(len(w) for w in windows) == 100
    assert len(windows) == 1
    slow = [{"t": t} for t in (0.0, 1.5, 4.0)]
    assert [len(w) for w in memory_windows(slow)] == [3]
    long = [{"t": i * 0.5} for i in range(80)]  # 0 .. 39.5 s
    grouped = memory_windows(long, 15.0)
    assert sum(len(w) for w in grouped) == 80
    for window in grouped:
        assert float(window[-1]["t"]) - float(window[0]["t"]) <= 15.0 + 1e-9

    assert moment_ok({"steer": 0.2}, {"throttle": 0.3, "brake": 0.0})
    assert not moment_ok({"steer": 0.2}, {"throttle": 0.3, "brake": 1.5})
    assert not moment_ok({"steer": float("nan")}, {"brake": 0.0})
    assert not moment_ok({"override": "player_steer"}, {"brake": 0.0})
    assert moment_ok({}, {})
    good = {"wheel": {"steer": 0.0}, "pedals": {"throttle": 0.1, "brake": 0.0}, "tech": {"lanes": []}}
    bad = {"wheel": {"steer": 0.0}, "pedals": {"brake": 2.0}, "tech": {"lanes": []}}
    assert loss_mask(good)
    assert not loss_mask(bad)
    assert not loss_mask({"wheel": {}, "pedals": {}})
    assert "steer" not in inspect.signature(scene_loss).parameters
    assert "wheel" not in inspect.signature(scene_loss).parameters
    assert "pedals" not in inspect.signature(scene_loss).parameters
    assert "steer" not in LOSS_TERMS
    assert "steer" not in OUTPUT_FIELDS
    assert "steer" not in ScenePrediction.__dataclass_fields__


def check_assignment_and_planner() -> None:
    cost = np.array([[9.0, 2.0, 7.0], [3.0, 4.0, 1.0], [6.0, 8.0, 5.0]], dtype=np.float64)
    pairs = dict(assign_min_cost(cost))
    assert pairs == {0: 1, 1: 2, 2: 0}, pairs
    # More predictions than labels: only the labels are matched.
    wide = np.array([[0.0, 5.0], [4.0, 0.0], [3.0, 3.0]], dtype=np.float64)
    matched = assign_min_cost(wide)
    assert len(matched) == 2
    assert dict(matched)[0] == 0
    assert dict(matched)[1] == 1

    params = tuple(inspect.signature(predict_path).parameters)
    assert params == PREDICT_PATH_ARGS, params
    assert not any("steer" in name for name in params)

    left = [{"x": -1.75, "y": float(y)} for y in range(0, 41, 5)]
    right = [{"x": 1.75, "y": float(y)} for y in range(0, 41, 5)]
    tech = {
        "lanes": [left, right],
        "curbs": [
            [{"x": -2.3, "y": float(y)} for y in range(0, 41, 5)],
            [{"x": 2.3, "y": float(y)} for y in range(0, 41, 5)],
        ],
        "tracks": [
            {
                "cls": "vehicle",
                "x": 0.1,
                "y": 18.0,
                "yaw": math.pi / 2.0,
                "length": 4.5,
                "width": 1.8,
                "speed_mps": 0.0,
                "partial": 0.0,
                "unseen_s": 0.0,
            }
        ],
        "signs": [
            {"cls": "stop_sign", "x": 2.2, "y": 24.0, "misses": 0, "seen_fraction": 1.0, "state": "unknown"}
        ],
    }
    labels = labels_from_tech(tech)
    assert labels["lanes"].shape == (6, 16, 2)
    assert labels["lane_valid"].shape == (6,)
    assert labels["curbs"].shape == (2, 16, 2)
    assert labels["objects"].shape == (16, 8)
    assert labels["signs"].shape == (8, 4)
    assert int(labels["lane_valid"].sum()) == 2
    assert int(labels["object_valid"].sum()) == 1
    assert int(labels["sign_valid"].sum()) == 1
    labels["steer"] = 0.8  # type: ignore[index]
    kwargs = scene_to_predict_kwargs(labels, ego_speed_mps=8.0)
    assert "steer" not in kwargs
    assert set(kwargs) <= set(PREDICT_PATH_ARGS)
    stopped = predict_path(**kwargs)
    assert stopped.stop_reason in ("stop_sign", "vehicle")
    open_labels = labels_from_tech({"lanes": [left, right]})
    free = predict_path(**scene_to_predict_kwargs(open_labels, ego_speed_mps=8.0))
    assert free.drivable and free.stop_reason == "none"
    assert free.path_length_m > stopped.path_length_m

    curb_only = labels_from_tech(
        {
            "curbs": [
                [{"x": -2.2, "y": float(y)} for y in range(0, 41, 5)],
                [{"x": 2.2, "y": float(y)} for y in range(0, 41, 5)],
            ]
        }
    )
    curb_kwargs = scene_to_predict_kwargs(curb_only, ego_speed_mps=5.0)
    assert "curbs" not in curb_kwargs
    along = predict_path(**curb_kwargs)
    assert along.drivable, along.prediction
    print("predict_path", list(params))


def check_net_cpu() -> None:
    try:
        import torch
    except ImportError:
        print("scene net: skipped (torch missing)")
        return
    from python.train.scene_net import SceneNet

    net = SceneNet().eval()
    state = net.initial_state()
    strip = torch.zeros(1, 3, CANVAS_H, CANVAS_W)
    pred, state = net(strip, 0.07, state)
    _assert_prediction(pred, state)
    pred, state = net(strip, 1.3, state)
    _assert_prediction(pred, state)
    names = [name for name, _param in net.named_parameters()]
    assert not any("steer" in name for name in names), names
    assert "steer" not in pred.as_dict()

    net.train()
    target = {
        key: torch.as_tensor(value)
        for key, value in labels_from_tech(
            {
                "lanes": [
                    [{"x": -1.75, "y": float(y)} for y in (0, 10, 20)],
                    [{"x": 1.75, "y": float(y)} for y in (0, 10, 20)],
                ],
                "signs": [{"cls": "traffic_light", "x": 0.4, "y": 30.0, "state": "red"}],
            }
        ).items()
    }
    fresh = net.initial_state()
    pred, _state = net(strip, 0.2, fresh)
    loss = scene_loss(pred, target)
    assert set(LOSS_TERMS) <= set(loss)
    assert "steer" not in loss
    loss["total"].backward()
    print("scene net: OK")


def _assert_prediction(pred: ScenePrediction, state: object) -> None:
    assert tuple(pred.lanes.shape) == (6, 16, 2)
    assert tuple(pred.lane_valid.shape) == (6,)
    assert tuple(pred.curbs.shape) == (2, 16, 2)
    assert tuple(pred.curb_valid.shape) == (2,)
    assert tuple(pred.objects.shape) == (16, 8)
    assert tuple(pred.object_class.shape) == (16, 6)
    assert tuple(pred.object_valid.shape) == (16,)
    assert tuple(pred.signs.shape) == (8, 4)
    assert tuple(pred.sign_class.shape) == (8, 4)
    assert tuple(pred.sign_state.shape) == (8, 4)
    assert tuple(pred.sign_valid.shape) == (8,)
    assert tuple(state.h.shape) == (1, 1, 512)  # type: ignore[attr-defined]
    assert tuple(state.tokens.shape) == (1, 8, 512)  # type: ignore[attr-defined]
    assert tuple(state.mask.shape) == (1, 8)  # type: ignore[attr-defined]
    assert torch_isfinite(pred)


def torch_isfinite(pred: ScenePrediction) -> bool:
    import torch

    for value in pred.as_dict().values():
        assert torch.isfinite(value).all()
    return True


def _free_covering(need: int) -> int:
    return (int(need) * USABLE_DEN + USABLE_NUM - 1) // USABLE_NUM


def check_fit_and_window() -> None:
    n_strips = 4
    window_strips = 4
    n_windows = 16
    small_file = canvas_bytes() * n_strips
    free = _free_covering(step_bytes(4, file_bytes=small_file, n_strips=n_strips, window_strips=window_strips))
    assert usable_bytes(free) >= step_bytes(4, file_bytes=small_file, n_strips=n_strips, window_strips=window_strips)
    small = plan_fit(
        file_bytes=small_file,
        n_strips=n_strips,
        free_vram_bytes=free,
        free_ram_bytes=0,
        n_windows=n_windows,
        window_strips=window_strips,
    )
    assert small.device == "cuda" and small.recommended and small.batch == 4, small
    assert small.file_bytes == small_file

    dropped = None
    for factor in (2, 3, 4, 5, 6, 8):
        big_file = canvas_bytes() * n_strips * factor
        try:
            big = plan_fit(
                file_bytes=big_file,
                n_strips=n_strips,
                free_vram_bytes=free,
                free_ram_bytes=0,
                n_windows=n_windows,
                window_strips=window_strips,
            )
        except TrainFitError:
            continue
        if big.device == "cuda" and big.recommended and big.batch < small.batch:
            dropped = big
            break
    assert dropped is not None, "a larger driving file should shrink the GPU batch"

    one = step_bytes(1, file_bytes=small_file, n_strips=n_strips, window_strips=window_strips)
    two = step_bytes(2, file_bytes=small_file, n_strips=n_strips, window_strips=window_strips)
    free_one = _free_covering(one)
    assert usable_bytes(free_one) < two
    tight = plan_fit(
        file_bytes=small_file,
        n_strips=n_strips,
        free_vram_bytes=free_one,
        free_ram_bytes=free_one * 30,
        n_windows=n_windows,
        window_strips=window_strips,
    )
    assert tight.device == "cuda" and tight.recommended and tight.batch == 1, tight

    free_vram = max(1, one // 8)
    free_ram = _free_covering(one)
    assert free_ram > free_vram
    cpu = plan_fit(
        file_bytes=small_file,
        n_strips=n_strips,
        free_vram_bytes=free_vram,
        free_ram_bytes=free_ram,
        n_windows=n_windows,
        window_strips=window_strips,
    )
    assert cpu.device == "cpu" and cpu.recommended is False and cpu.batch >= 1, cpu

    for free_vram_bytes, free_ram_bytes in ((1024, 2048), (4096, 1024)):
        try:
            plan_fit(
                file_bytes=small_file,
                n_strips=n_strips,
                free_vram_bytes=free_vram_bytes,
                free_ram_bytes=free_ram_bytes,
                n_windows=n_windows,
                window_strips=window_strips,
            )
        except TrainFitError:
            pass
        else:
            raise AssertionError("step should not fit")

    huge = _free_covering(step_bytes(32, file_bytes=small_file, n_strips=n_strips, window_strips=window_strips))
    capped = plan_fit(
        file_bytes=small_file,
        n_strips=n_strips,
        free_vram_bytes=huge,
        free_ram_bytes=1,
        n_windows=2,
        window_strips=window_strips,
    )
    assert capped.batch == 2 and capped.device == "cuda" and capped.recommended

    calls: list[tuple[str, int]] = []

    def accept_gpu(device: str, batch: int) -> bool:
        calls.append((device, batch))
        return True

    probed = fit_with_probe(small, accept_gpu, n_windows=n_windows)
    assert probed.device == "cuda" and probed.recommended and probed.batch == small.batch
    assert calls == [("cuda", small.batch)]

    calls.clear()

    def shrink_gpu(device: str, batch: int) -> bool:
        calls.append((device, batch))
        return device == "cuda" and batch <= 1

    shrunk = fit_with_probe(small, shrink_gpu, n_windows=n_windows)
    assert shrunk.device == "cuda" and shrunk.recommended and shrunk.batch == 1, shrunk
    assert calls[0] == ("cuda", small.batch)
    assert calls[-1] == ("cuda", 1)
    assert all(device == "cuda" for device, _batch in calls)

    rich = plan_fit(
        file_bytes=small_file,
        n_strips=n_strips,
        free_vram_bytes=free,
        free_ram_bytes=free * 4,
        n_windows=4,
        window_strips=window_strips,
    )
    assert rich.device == "cuda" and rich.recommended
    calls.clear()

    def gpu_miss(device: str, batch: int) -> bool:
        calls.append((device, batch))
        return device == "cpu" and batch <= 2

    fallen = fit_with_probe(rich, gpu_miss, n_windows=4)
    assert fallen.device == "cpu" and fallen.recommended is False and fallen.batch == 2, fallen
    assert any(device == "cuda" for device, _batch in calls)
    assert ("cpu", 2) in calls

    calls.clear()

    def accept_cpu(device: str, batch: int) -> bool:
        calls.append((device, batch))
        return True

    stayed = fit_with_probe(cpu, accept_cpu, n_windows=n_windows)
    assert stayed.device == "cpu" and stayed.recommended is False
    assert calls and all(device == "cpu" for device, _batch in calls)

    assert shrink_batch(8) == 4
    assert shrink_batch(3) == 1
    assert shrink_batch(1) == 0

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "state.jsonl").write_text("abc", encoding="utf-8")
        (folder / "strips").mkdir()
        (folder / "strips" / "000000.jpg").write_bytes(b"12345")
        assert measure_driving_file(folder) == 8
    assert read_free_ram_bytes() > 0
    assert read_free_vram_bytes() >= 0

    gpu_text = format_status(
        TrainStatus(
            eta_s=90,
            loss=0.25,
            lr=1e-3,
            steps_per_sec=2.5,
            memory_bytes=1610612736,
            memory_kind="vram",
            device="cuda",
            batch=4,
            recommended=True,
            step=3,
            steps=10,
        )
    )
    assert "time left 1m 30s" in gpu_text
    assert "loss 0.2500" in gpu_text
    assert "lr 0.001" in gpu_text
    assert "steps/s 2.50" in gpu_text
    assert "VRAM 1.50 GB" in gpu_text
    assert "not recommended" not in gpu_text
    assert "cuda  batch 4  recommended" in gpu_text

    cpu_text = format_status(
        TrainStatus(
            eta_s=None,
            loss=None,
            lr=1e-3,
            steps_per_sec=0.0,
            memory_bytes=2 * 1024 * 1024,
            memory_kind="ram",
            device="cpu",
            batch=1,
            recommended=False,
            step=0,
            steps=4,
        )
    )
    assert "time left --" in cpu_text
    assert "loss --" in cpu_text
    assert "steps/s 0.00" in cpu_text
    assert "RAM 2 MB" in cpu_text
    assert "cpu  batch 1  not recommended" in cpu_text
    assert "VRAM" not in cpu_text
    assert predict_eta_s(0, 10.0, 5) is None
    assert predict_eta_s(4, 2.0, 4) == 2.0
    assert steps_per_second(4, 2.0) == 2.0

    window = TrainWindow()
    window.update(
        TrainStatus(
            eta_s=12,
            loss=1.5,
            lr=1e-3,
            steps_per_sec=1.0,
            memory_bytes=1024 * 1024,
            memory_kind="ram",
            device="cpu",
            batch=1,
            recommended=False,
            step=1,
            steps=3,
        )
    )
    assert "not recommended" in window.lines
    assert window.available is False


def check_trainer_is_offline() -> None:
    files = (
        ROOT / "python" / "train" / "train_scene.py",
        ROOT / "python" / "train" / "fit.py",
        ROOT / "python" / "train" / "status_window.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    for bad in (
        "taskkill",
        "os.kill",
        "close_beamng",
        "quit_beamng",
        "beamng.disconnect",
        "flask",
        "http.server",
        "socketserver",
        "fastapi",
    ):
        assert bad not in text, bad
    body = inspect.getsource(train_directory)
    assert body.index("measure_driving_file") < body.index("plan_fit")
    assert body.index("read_free_vram_bytes") < body.index("plan_fit")
    assert body.index("plan_fit") < body.index("_optimizer_step")
    assert "step_dt" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Soft Esc parked" in readme
    assert "## Alpha 1.7.8" in readme
    assert "1.7.9" not in readme
    assert "2272" in readme
    assert "free VRAM" in readme
    assert "not recommended" in readme


def main() -> None:
    check_canvas_geometry()
    check_variable_rate_and_one_strip()
    check_timestamp_refusal_and_empty_sector()
    check_sectors_and_ego_edge()
    check_rate_helpers_and_precision()
    check_assignment_and_planner()
    check_fit_and_window()
    check_trainer_is_offline()
    check_net_cpu()
    print("test_scene_strip: OK")


if __name__ == "__main__":
    main()
