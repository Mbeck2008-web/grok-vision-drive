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
    StripRecordControl,
    StripTimestampError,
    StripWriter,
    compose_strip,
    read_state_jsonl,
    wipe_training,
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
    export_scene_onnx,
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
    BG,
    FG,
    FONT,
    FS_BODY,
    FS_DIM,
    FS_TITLE,
    ICE,
    PAD_X,
    GRAPH_FILL,
    GRAPH_H,
    PANEL_W,
    ROW_H,
    TH,
    TrainStatus,
    TrainWindow,
    format_parameter_count,
    format_recorded_hm,
    format_status,
    parameter_fact,
    loss_graph_box,
    loss_plot_box,
    loss_polyline,
    predict_eta_s,
    read_recorded_seconds,
    render_train_panel,
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


def _state_line(index: int, t: float, dt_s: float | None, session: str) -> str:
    return json.dumps({"i": index, "t": t, "dt_s": dt_s, "session": session}) + "\n"


def check_recorded_hours_minutes() -> None:
    """Two sessions, one gap that is not 0.2 s, shown as hours and minutes."""
    # 0.5 + 3.5 + 3716 = 3720 s = 1 hour 2 minutes.
    # The parked jump from t=1004 to t=50000 is a null dt_s and is not driving.
    # Five frames at 5 Hz would be 1 second, which is 0 hours 0 minutes.
    session_a = (
        _state_line(0, 1000.0, None, "s1")
        + _state_line(1, 1000.5, 0.5, "s1")
        + _state_line(2, 1004.0, 3.5, "s1")
    )
    session_b = _state_line(3, 50000.0, None, "s2") + _state_line(4, 53716.0, 3716.0, "s2")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty = root / "empty"
        empty.mkdir()
        (empty / "notes.txt").write_bytes(b"leave-me")
        window = TrainWindow()
        window.load_folder(empty)
        assert window.lines.splitlines()[0] == "recorded 0 hours 0 minutes"
        assert "idle" in window.lines
        assert window.losses == []
        assert read_recorded_seconds(empty) == 0.0
        assert format_recorded_hm(0) == "0 hours 0 minutes"
        assert (empty / "notes.txt").read_bytes() == b"leave-me"
        assert not (empty / "state.jsonl").exists()
        assert not (empty / "meta.json").exists()

        folder = root / "strips_out"
        folder.mkdir()
        (folder / "strips").mkdir()
        planted = folder / "strips" / "000000.jpg"
        planted.write_bytes(b"keep-jpeg")
        state = folder / "state.jsonl"
        state.write_text(session_a + session_b, encoding="utf-8")
        before = state.read_bytes()
        assert abs(read_recorded_seconds(folder) - 3720.0) < 1e-6
        window.load_folder(folder)
        assert "1 hours 2 minutes" in window.lines
        assert window.recorded_s == read_recorded_seconds(folder)
        shown = format_status(
            TrainStatus(
                eta_s=None,
                loss=None,
                lr=1e-3,
                steps_per_sec=0.0,
                memory_bytes=0,
                memory_kind="ram",
                device="cpu",
                batch=1,
                recommended=False,
                step=0,
                steps=1,
                recorded_s=window.recorded_s,
            )
        )
        assert shown.splitlines()[0] == "recorded 1 hours 2 minutes"
        window.update(
            TrainStatus(
                eta_s=10,
                loss=0.1,
                lr=1e-3,
                steps_per_sec=1.0,
                memory_bytes=1024,
                memory_kind="ram",
                device="cpu",
                batch=1,
                recommended=False,
                step=1,
                steps=2,
            )
        )
        assert window.lines.splitlines()[0] == "recorded 1 hours 2 minutes"
        assert state.read_bytes() == before
        assert planted.read_bytes() == b"keep-jpeg"
        assert not (folder / "meta.json").exists()

        extra = _state_line(5, 53776.0, 60.0, "s2")
        with state.open("a", encoding="utf-8") as fh:
            fh.write(extra)
        window.refresh_recorded()
        assert "1 hours 3 minutes" in window.lines
        assert abs(window.recorded_s - 3780.0) < 1e-6
        assert state.read_bytes().startswith(before)
        assert planted.read_bytes() == b"keep-jpeg"


def check_nan_cost_and_cpu_export() -> None:
    """All-NaN assignment returns, and export builds CPU inputs for a CUDA module."""
    blank = np.full((3, 2), np.nan, dtype=np.float64)
    assert assign_min_cost(blank) == []
    assert assign_min_cost(np.full((2, 2), np.inf)) == []
    mixed = np.array([[np.nan, 0.5], [0.25, np.nan]], dtype=np.float64)
    assert dict(assign_min_cost(mixed)) == {0: 1, 1: 0}
    assert "cpu_export_inputs" in inspect.getsource(export_scene_onnx)

    class _Device:
        def __init__(self, name: str) -> None:
            self.type = name

        def __str__(self) -> str:
            return self.type

    class _Tensor:
        def __init__(self, device: _Device) -> None:
            self.device = device

    class _State:
        def __init__(self, device: _Device) -> None:
            self.h = _Tensor(device)
            self.tokens = _Tensor(device)
            self.mask = _Tensor(device)

    class _CudaNet:
        def __init__(self) -> None:
            self.device = _Device("cuda")

        def to(self, device: object) -> "_CudaNet":
            name = getattr(device, "type", None) or str(device)
            self.device = _Device(str(name))
            return self

        def eval(self) -> "_CudaNet":
            return self

        def initial_state(self, device: object = None, dtype: object = None, batch: int = 1) -> _State:
            del dtype, batch
            name = "cpu" if device is None else (getattr(device, "type", None) or str(device))
            return _State(_Device(str(name)))

    def zeros(*_shape: int, device: object = None, **_kwargs: object) -> _Tensor:
        name = "cpu" if device is None else (getattr(device, "type", None) or str(device))
        return _Tensor(_Device(str(name)))

    seen: dict[str, object] = {}

    def trace(model: _CudaNet, args: tuple[_Tensor, ...], path_str: str) -> None:
        seen["model"] = model
        seen["args"] = args
        Path(path_str).write_bytes(b"onnx")

    net = _CudaNet()
    assert net.device.type != "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        out = export_scene_onnx(Path(tmp) / "e2e_scene.onnx", net, zeros=zeros, trace=trace)
        assert out.is_file()
    traced = seen["model"]
    assert isinstance(traced, _CudaNet)
    assert traced.device.type == "cpu"
    args = seen["args"]
    assert isinstance(args, tuple) and len(args) == 5
    assert all(tensor.device.type == "cpu" for tensor in args)


def _jpeg_bytes(folder: Path) -> dict[str, bytes]:
    strips = folder / "strips"
    if not strips.is_dir():
        return {}
    return {path.name: path.read_bytes() for path in strips.glob("*.jpg")}


def check_append_and_confirm_wipe() -> None:
    """A second start only adds files. Wipe deletes only after confirm, and only in that folder."""
    colors = _colors()
    frames = _bundle(colors)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "disk_a"
        other = root / "disk_b"
        prefs = root / "remember" / "gvd_strip_record.json"
        decoy = root / "elsewhere" / "strips" / "000000.jpg"
        decoy.parent.mkdir(parents=True)
        decoy.write_bytes(b"not-this-folder")

        orphan = root / "orphan"
        (orphan / "strips").mkdir(parents=True)
        planted = orphan / "strips" / "000000.jpg"
        planted.write_bytes(b"keep-orphan")
        old_state = '{"i": 0, "t": 1.0, "dt_s": null, "kept": true}\n'
        (orphan / "state.jsonl").write_text(old_state, encoding="utf-8")
        writer = StripWriter(orphan)
        rec = writer.write(frames, t=2.0, timestamps=_stamps(frames, 2.0), **_controls())
        assert planted.read_bytes() == b"keep-orphan"
        assert rec.path.name == "000001.jpg"
        assert (orphan / "state.jsonl").read_text(encoding="utf-8").startswith(old_state)
        real_append = writer._append_state

        def fail_append(_line: dict) -> None:
            raise OSError("full")

        writer._append_state = fail_append
        try:
            writer.write(frames, t=2.5, timestamps=_stamps(frames, 2.5), **_controls())
        except OSError:
            pass
        else:
            raise AssertionError("a failed state line was swallowed")
        assert planted.read_bytes() == b"keep-orphan"
        assert (orphan / "strips" / "000001.jpg").read_bytes() == rec.path.read_bytes()
        orphan_new = orphan / "strips" / "000002.jpg"
        assert orphan_new.is_file()
        assert len(read_state_jsonl(orphan / "state.jsonl")) == 2
        writer._append_state = real_append
        assert wipe_training(orphan, confirm=False) == []
        assert planted.read_bytes() == b"keep-orphan"
        assert "unlink" not in inspect.getsource(StripWriter.write)
        assert "unlink" not in inspect.getsource(StripWriter._create_exclusive)
        assert "wipe_training" not in inspect.getsource(StripRecordControl.stop)
        assert "wipe_training" not in inspect.getsource(StripRecordControl.start)
        assert "wipe_training" not in inspect.getsource(StripRecordControl.set_destination)
        assert "rmtree" not in (ROOT / "python" / "data" / "strip_writer.py").read_text(encoding="utf-8")

        ctl = StripRecordControl(prefs_path=prefs, destination=str(folder))
        assert ctl.status_text() == "recording off"
        assert ctl.offer(frames, t=1.0, timestamps=_stamps(frames, 1.0), **_controls()) is None
        ctl.note_engaged(True)
        assert ctl.recording is False
        ctl.start()
        assert ctl.status_text() == "recording on"
        assert ctl.recording is True
        first = ctl.offer(frames, t=10.0, timestamps=_stamps(frames, 10.0), **_controls())
        second = ctl.offer(frames, t=10.2, timestamps=_stamps(frames, 10.2), **_controls())
        assert first is not None and second is not None
        assert first.path.name == "000000.jpg"
        assert second.path.name == "000001.jpg"
        assert abs((second.dt_s or 0.0) - 0.2) < 1e-6
        ctl.stop()
        assert ctl.recording is False
        assert ctl.status_text() == "recording off"
        ctl.note_engaged(True)
        assert ctl.offer(frames, t=10.4, timestamps=_stamps(frames, 10.4), **_controls()) is None
        session_one = _jpeg_bytes(folder)
        state_one = (folder / "state.jsonl").read_text(encoding="utf-8")
        assert set(session_one) == {"000000.jpg", "000001.jpg"}

        remembered = StripRecordControl(prefs_path=prefs)
        assert remembered.destination == str(folder)
        assert remembered.recording is False
        assert _jpeg_bytes(folder) == session_one
        assert (folder / "state.jsonl").read_text(encoding="utf-8") == state_one

        ctl.start()
        assert _jpeg_bytes(folder) == session_one
        assert (folder / "state.jsonl").read_text(encoding="utf-8") == state_one
        added = ctl.offer(frames, t=80.0, timestamps=_stamps(frames, 80.0), **_controls())
        assert added is not None
        assert added.path.name not in session_one
        assert added.dt_s is None
        kept = {name: data for name, data in _jpeg_bytes(folder).items() if name in session_one}
        assert kept == session_one
        state_two = (folder / "state.jsonl").read_text(encoding="utf-8")
        assert state_two.startswith(state_one)
        rows = read_state_jsonl(folder / "state.jsonl")
        assert rows[0]["session"] == rows[1]["session"]
        assert rows[-1]["session"] != rows[0]["session"]
        ctl.note_engaged(True)
        assert ctl.recording is True
        followed = ctl.offer(frames, t=80.4, timestamps=_stamps(frames, 80.4), **_controls())
        assert followed is not None
        assert abs((followed.dt_s or 0.0) - 0.4) < 1e-6
        assert followed.path.name not in session_one
        ctl.note_engaged(False)
        assert ctl.recording is True
        stale = _bundle(colors)
        stamps = _stamps(stale, 80.9)
        stamps["wide"] = 1.0
        for cam_id in stale:
            stale[cam_id] = _frame((255, 255, 255), cam_id)
        missed = ctl.offer(stale, t=80.9, timestamps=stamps, **_controls())
        assert missed is not None
        missed_bgr = _read_jpeg(missed.path)
        wide = next(sec for sec in SECTORS if sec.cam_id == "wide")
        inset = missed_bgr[16:-16, wide.x0 + 16 : wide.x1 - 16]
        assert int(inset.max()) <= 3
        ctl.stop()
        assert ctl.recording is False
        stopped = _jpeg_bytes(folder)
        state_stopped = (folder / "state.jsonl").read_text(encoding="utf-8")
        assert session_one.keys() <= stopped.keys()
        for name, data in session_one.items():
            assert stopped[name] == data

        ctl.set_destination("D:/gvd_strips")
        assert ctl.destination == "D:/gvd_strips"
        assert json.loads(prefs.read_text(encoding="utf-8"))["destination"] == "D:/gvd_strips"
        assert not Path("D:/gvd_strips").exists()
        reloaded = StripRecordControl(prefs_path=prefs)
        assert reloaded.destination == "D:/gvd_strips"
        assert reloaded.recording is False
        assert _jpeg_bytes(folder) == stopped
        ctl.set_destination(str(other))
        assert ctl.recording is False
        ctl.start()
        moved = ctl.offer(frames, t=3.0, timestamps=_stamps(frames, 3.0), **_controls())
        assert moved is not None
        assert moved.path.parent.parent == other
        assert _jpeg_bytes(folder) == stopped
        assert (folder / "state.jsonl").read_text(encoding="utf-8") == state_stopped
        ctl.arm_free_space()
        ctl.set_destination(str(folder))
        assert ctl.free_space_armed is False
        assert ctl.confirm_free_space() == []
        assert _jpeg_bytes(folder) == stopped
        assert _jpeg_bytes(other) != {}

        (folder / "notes.txt").write_bytes(b"keep-me")
        (folder / "strips" / "side.txt").write_bytes(b"keep-side")
        (folder / "strips" / "000099.jpg.tmp").write_bytes(b"partial")
        (folder / "extra").mkdir()
        (folder / "extra" / "000000.jpg").write_bytes(b"not-a-strip")
        (folder / "extra" / "state.jsonl").write_text("leave\n", encoding="utf-8")
        assert wipe_training(folder, confirm=False) == []
        assert _jpeg_bytes(folder) == stopped
        ctl.arm_free_space()
        assert ctl.free_space_armed is True
        assert _jpeg_bytes(folder) == stopped
        stranger = StripRecordControl(prefs_path=prefs)
        assert stranger.confirm_free_space() == []
        assert _jpeg_bytes(folder) == stopped
        removed = ctl.confirm_free_space()
        assert removed
        assert _jpeg_bytes(folder) == {}
        assert not (folder / "state.jsonl").exists()
        assert not (folder / "meta.json").exists()
        assert not (folder / "strips" / "000099.jpg.tmp").exists()
        assert (folder / "notes.txt").read_bytes() == b"keep-me"
        assert (folder / "strips" / "side.txt").read_bytes() == b"keep-side"
        assert (folder / "extra" / "000000.jpg").read_bytes() == b"not-a-strip"
        assert (folder / "extra" / "state.jsonl").read_text(encoding="utf-8") == "leave\n"
        assert decoy.read_bytes() == b"not-this-folder"
        assert _jpeg_bytes(other) != {}
        assert prefs.is_file()
        assert ctl.confirm_free_space() == []
        assert (folder / "notes.txt").read_bytes() == b"keep-me"

        from python.viz.nerd import render_panel
        from python.viz.stage import STAGE_W, VizUI

        ui_prefs = root / "ui_prefs.json"
        (other / "keep.txt").write_bytes(b"keep-other")
        (other / "strips" / "side.txt").write_bytes(b"keep-other-side")
        other_before = _jpeg_bytes(other)
        ui = VizUI()
        ui.show_nerd = True
        ui.nerd_tab = "live"
        ui.record = StripRecordControl(prefs_path=ui_prefs, destination=str(other))
        panel = render_panel({"engaged": True}, h=800, w=560, ui=ui)
        assert panel.shape[0] == 800
        ids = {h.get("id") for h in ui.nerd_hits if h.get("kind") == "record"}
        assert {"start", "stop", "free", "confirm", "dest", "status"} <= ids

        def click(ident: str) -> None:
            hit = next(h for h in ui.nerd_hits if h.get("kind") == "record" and h.get("id") == ident)
            rect = hit["rect"]
            x = STAGE_W + (int(rect[0]) + int(rect[2])) // 2
            y = (int(rect[1]) + int(rect[3])) // 2
            assert ui.handle_click(x, y, stage_w=STAGE_W)

        ui.feed_strip(frames, timestamps=_stamps(frames, 4.0), t=4.0, engaged=True)
        assert ui.record.recording is False
        assert ui.record.engaged is True
        click("start")
        assert ui.record.status_text() == "recording on"
        ui.feed_strip(frames, timestamps=_stamps(frames, 4.2), t=4.2, engaged=True)
        assert ui.record.recording is True
        panel = render_panel({"engaged": True}, h=800, w=560, ui=ui)
        mark = next(h for h in ui.nerd_hits if h.get("kind") == "record" and h.get("id") == "status")
        mx = (int(mark["rect"][0]) + int(mark["rect"][2])) // 2
        my = (int(mark["rect"][1]) + int(mark["rect"][3])) // 2
        on_px = panel[my, mx]
        assert int(on_px[1]) > int(on_px[2])
        grown = _jpeg_bytes(other)
        assert other_before.keys() <= grown.keys()
        for name, data in other_before.items():
            assert grown[name] == data
        assert len(grown) > len(other_before)
        click("stop")
        assert ui.record.status_text() == "recording off"
        assert ui.record.recording is False
        held = _jpeg_bytes(other)
        ui.feed_strip(frames, timestamps=_stamps(frames, 4.8), t=4.8, engaged=True)
        assert ui.record.recording is False
        assert _jpeg_bytes(other) == held
        panel = render_panel({"engaged": True}, h=800, w=560, ui=ui)
        mark = next(h for h in ui.nerd_hits if h.get("kind") == "record" and h.get("id") == "status")
        mx = (int(mark["rect"][0]) + int(mark["rect"][2])) // 2
        my = (int(mark["rect"][1]) + int(mark["rect"][3])) // 2
        off_px = panel[my, mx]
        assert int(off_px[2]) > int(off_px[1])
        click("confirm")
        assert _jpeg_bytes(other) == held
        click("free")
        assert ui.record.free_space_armed is True
        assert _jpeg_bytes(other) == held
        click("confirm")
        assert _jpeg_bytes(other) == {}
        assert not (other / "state.jsonl").exists()
        assert not (other / "meta.json").exists()
        assert (other / "keep.txt").read_bytes() == b"keep-other"
        assert (other / "strips" / "side.txt").read_bytes() == b"keep-other-side"
        assert decoy.read_bytes() == b"not-this-folder"
        assert (folder / "notes.txt").read_bytes() == b"keep-me"
        assert ui_prefs.is_file()


def _status_for_graph(**overrides: object) -> TrainStatus:
    fields: dict[str, object] = {
        "eta_s": 90,
        "loss": None,
        "lr": 1e-3,
        "steps_per_sec": 2.5,
        "memory_bytes": 1610612736,
        "memory_kind": "vram",
        "device": "cuda",
        "batch": 4,
        "recommended": True,
        "step": 1,
        "steps": 10,
        "recorded_s": 3720.0,
    }
    fields.update(overrides)
    return TrainStatus(**fields)  # type: ignore[arg-type]


def check_train_panel_loss_graph() -> None:
    """The panel is the supervisor's dark type, and the loss curve is the given series."""
    from python.viz import nerd

    assert BG == nerd.BG
    assert FG == nerd.FG
    assert ICE == nerd.ICE
    assert FONT == nerd.FONT
    assert FS_TITLE == nerd.FS_TITLE
    assert FS_BODY == nerd.FS_BODY
    assert FS_DIM == nerd.FS_DIM
    assert TH == nerd.TH
    assert ROW_H == nerd.ROW_H
    assert PAD_X == nerd.PAD_X
    assert PANEL_W == nerd.NERD_WIDTH

    idle_window = TrainWindow()
    idle_window.load_folder(Path("/no/such/gvd-strips"))
    assert idle_window.available is False
    assert idle_window.losses == []
    assert idle_window.panel is not None
    assert idle_window.panel.shape[1] == PANEL_W
    assert idle_window.panel.shape[0] > GRAPH_H
    assert tuple(int(channel) for channel in idle_window.panel[0, 0]) == BG
    assert float(idle_window.panel.mean()) < 80
    assert "0 hours 0 minutes" in idle_window.lines
    assert "idle" in idle_window.lines
    plot = loss_plot_box(loss_graph_box(idle_window.panel.shape[0], idle_window.panel.shape[1]))
    assert loss_polyline([], plot) == []
    idle_crop = idle_window.panel[plot[1] : plot[3], plot[0] : plot[2]]
    assert np.all(idle_crop == np.array(GRAPH_FILL, dtype=np.uint8))

    window = TrainWindow()
    series = [1.0, 0.25, 0.8]
    panels = []
    for index, loss in enumerate(series, start=1):
        window.update(_status_for_graph(loss=loss, step=index))
        assert window.panel is not None
        panels.append(window.panel.copy())
    assert window.losses == series
    assert not np.array_equal(panels[0], panels[-1])
    plot = loss_plot_box(loss_graph_box(window.panel.shape[0], window.panel.shape[1]))
    points = loss_polyline(window.losses, plot)
    assert len(points) == 3
    assert points[0][0] < points[1][0] < points[2][0]
    assert points[0][1] < points[2][1] < points[1][1]
    live = window.panel
    fill = float(np.array(GRAPH_FILL).mean())
    for x, y in points:
        live_patch = live[y - 2 : y + 3, x - 2 : x + 3]
        assert float(live_patch.mean()) > fill + 5

    text = window.lines
    assert "1 hours 2 minutes" in text
    assert "time left 1m 30s" in text
    assert "loss 0.8000" in text
    assert "lr 0.001" in text
    assert "steps/s 2.50" in text
    assert "VRAM 1.50 GB" in text
    assert "not recommended" not in text
    window.update(_status_for_graph(loss=None, step=4))
    assert window.losses == series
    assert len(loss_polyline(window.losses, plot)) == 3

    cpu = format_status(
        _status_for_graph(device="cpu", recommended=False, memory_kind="ram", batch=1, loss=0.8)
    )
    assert "cpu  batch 1  not recommended" in cpu
    cpu_panel = render_train_panel(cpu.splitlines(), [0.8])
    assert tuple(int(channel) for channel in cpu_panel[0, 0]) == BG
    assert cpu_panel.shape[1] == PANEL_W
    assert cpu_panel.shape[0] > GRAPH_H
    window.open()
    window.close()
    assert window.panel is not None


def check_parameter_count_hand_sum() -> None:
    """A tiny built net's trainable count matches a hand sum of its layers."""
    from python.train.scene_net import (
        architecture_parameter_count,
        conv2d_parameter_count,
        count_module_parameters,
        gru_parameter_count,
        linear_parameter_count,
        scene_parameter_count,
        torch_ready,
    )

    # Conv2d(2, 3, kernel 3, bias): weight 3*2*3*3 = 54, bias 3, total 57.
    # Linear(3, 4, bias): weight 4*3 = 12, bias 4, total 16.
    # GRU(input 4, hidden 2, one layer, bias): ih 3*2*4 = 24, hh 3*2*2 = 12,
    # bias_ih 6, bias_hh 6, total 48.
    hand = 57 + 16 + 48
    counted = (
        conv2d_parameter_count(2, 3, 3, bias=True)
        + linear_parameter_count(3, 4, bias=True)
        + gru_parameter_count(4, 2, num_layers=1, bias=True)
    )
    assert counted == hand == 121
    if torch_ready():
        from torch import nn

        from python.train.scene_net import SceneNet

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, bias=True)
                self.fc = nn.Linear(3, 4, bias=True)
                self.gru = nn.GRU(4, 2, num_layers=1, batch_first=True)

        tiny = Tiny()
        assert count_module_parameters(tiny) == hand
        summed, exact = scene_parameter_count(tiny)
        assert summed == hand and exact is False
        frozen = tiny.fc.bias
        frozen.requires_grad_(False)
        assert count_module_parameters(tiny) == hand - 4
        real = count_module_parameters(SceneNet())
        assert real == architecture_parameter_count()
        shown, approximate = scene_parameter_count()
        assert shown == real and approximate is False
    else:
        # No tensors. The real net is the same layer rules as the tiny net.
        shown, approximate = scene_parameter_count()
        assert approximate is True
        assert shown == architecture_parameter_count()
        assert shown > hand
    fact = parameter_fact()
    assert format_parameter_count(shown) in fact
    if approximate:
        assert fact.startswith("params approximate ")
    else:
        assert fact.startswith("params ")
        assert "approximate" not in fact
    assert fact in format_status(_status_for_graph())
    idle = TrainWindow()
    idle.load_folder(Path("/no/such/gvd-strips-params"))
    assert idle.lines.splitlines()[0] == "recorded 0 hours 0 minutes"
    assert fact in idle.lines
    assert "idle" in idle.lines


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
    check_append_and_confirm_wipe()
    check_sectors_and_ego_edge()
    check_rate_helpers_and_precision()
    check_assignment_and_planner()
    check_nan_cost_and_cpu_export()
    check_fit_and_window()
    check_recorded_hours_minutes()
    check_train_panel_loss_graph()
    check_parameter_count_hand_sum()
    check_trainer_is_offline()
    check_net_cpu()
    print("test_scene_strip: OK")


if __name__ == "__main__":
    main()
