"""Train the scene net from a strip directory.

Reads ``meta.json``, ``strips/*.jpg``, and ``state.jsonl``. Each step uses the
recorded seconds since the previous strip. Windows cover 15 seconds of real
time. Before the first step the trainer measures that directory and the free
VRAM, then shrinks the batch until the step fits. CPU is used only when the
card cannot hold one step and the machine has more CPU RAM, and that path is
not recommended. The run is drawn on the same dark panel as the supervisor. This process does not start or
close BeamNG.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.strip_writer import CANVAS_H, CANVAS_W, read_state_jsonl  # noqa: E402
from python.train.fit import (  # noqa: E402
    TrainFitError,
    fit_with_probe,
    largest_batch,
    measure_driving_file,
    plan_fit,
    read_free_ram_bytes,
    read_free_vram_bytes,
    read_used_ram_bytes,
    read_used_vram_bytes,
    shrink_batch,
    usable_bytes,
)
from python.train.scene_net import (  # noqa: E402
    MEMORY_S,
    OUTPUT_FIELDS,
    ScenePrediction,
    default_onnx_path,
    export_scene_onnx,
    labels_from_tech,
    loss_mask,
    memory_windows,
    precision_for_capability,
    scene_loss,
    step_dt,
    torch_ready,
)
from python.train.status_window import (  # noqa: E402
    TrainStatus,
    TrainWindow,
    predict_eta_s,
    steps_per_second,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the GVD scene net from recorded strips")
    parser.add_argument("--strips", type=str, default="", help="Directory with meta.json, strips/, state.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--capability",
        type=float,
        default=None,
        help="Compute capability override. 6.1 stays FP32. 7.0 and newer select mixed precision.",
    )
    parser.add_argument(
        "--export",
        nargs="?",
        const=str(default_onnx_path()),
        default=None,
        help="Write models/e2e_scene.onnx (FP32 graph). Optional path.",
    )
    return parser


def _load_bgr(root: Path, index: int) -> Any:
    import cv2

    path = root / "strips" / f"{index:06d}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if int(bgr.shape[0]) != CANVAS_H or int(bgr.shape[1]) != CANVAS_W:
        raise ValueError(f"{path} is {bgr.shape[1]}x{bgr.shape[0]}, expected {CANVAS_W}x{CANVAS_H}")
    return bgr


def _autocast(enabled: bool) -> Any:
    import torch

    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _scaler(enabled: bool) -> Any:
    if not enabled:
        return None
    import torch

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _capability(device: Any, capability: float | None) -> Any:
    if capability is not None:
        return capability
    if device.type == "cuda":
        import torch

        return torch.cuda.get_device_capability(device)
    return None


def _as_target(labels: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    out = {}
    long_keys = {"object_class", "sign_class", "sign_state"}
    for key, value in labels.items():
        dtype = torch.long if key in long_keys else torch.float32
        out[key] = torch.as_tensor(value, device=device, dtype=dtype)
    return out


def _prediction_at(pred: ScenePrediction, index: int) -> ScenePrediction:
    return ScenePrediction(**{name: getattr(pred, name)[index] for name in OUTPUT_FIELDS})


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _memory_reading(device_name: str) -> tuple[int, str]:
    if device_name == "cuda":
        used = read_used_vram_bytes()
        if used <= 0:
            try:
                import torch

                if torch.cuda.is_available():
                    used = int(torch.cuda.memory_reserved())
            except Exception:
                used = 0
        return used, "vram"
    return read_used_ram_bytes(), "ram"


def _probe_cuda(net: Any, batch: int, window_strips: int, amp: bool, device: Any) -> bool:
    """One dummy window of this batch. Freed before the first real step."""
    import torch

    was_training = bool(net.training)
    net.train()
    ok = False
    try:
        state = net.initial_state(device=device, batch=batch)
        total = None
        for _ in range(max(1, int(window_strips))):
            strip = torch.zeros(int(batch), 3, CANVAS_H, CANVAS_W, device=device)
            dt = torch.zeros(int(batch), device=device)
            with _autocast(amp):
                pred, state = net.forward_batch(strip, dt, state)
            term = pred.lanes.float().sum()
            total = term if total is None else total + term
        if total is None:
            return False
        total.backward()
        ok = True
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        ok = False
    finally:
        net.zero_grad(set_to_none=True)
        _release_cuda()
        if not was_training:
            net.eval()
    return ok


def _blank() -> Any:
    return np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)


def _optimizer_step(
    net: Any,
    opt: Any,
    scaler: Any,
    root: Path,
    chunk: list[list[Mapping[str, Any]]],
    *,
    device: Any,
    amp: bool,
    view: TrainWindow | None,
) -> tuple[float | None, int]:
    import torch

    length = max(len(window) for window in chunk)
    width = len(chunk)
    blank = _blank()
    net.train()
    opt.zero_grad(set_to_none=True)
    state = net.initial_state(device=device, batch=width)
    total = None
    n_loss = 0
    for tick in range(length):
        frames = []
        dts = []
        for window in chunk:
            if tick < len(window):
                row = window[tick]
                frames.append(_load_bgr(root, int(row["i"])))
                dts.append(step_dt(row.get("dt_s")))
            else:
                frames.append(blank)
                dts.append(0.0)
        image = torch.from_numpy(np.stack(frames)).to(device)
        dt = torch.tensor(dts, dtype=torch.float32, device=device)
        with _autocast(amp):
            pred, state = net.forward_batch(image, dt, state)
        for index, window in enumerate(chunk):
            if tick >= len(window):
                continue
            row = window[tick]
            if not loss_mask(row):
                continue
            step = scene_loss(_prediction_at(pred, index), _as_target(labels_from_tech(row.get("tech")), device))
            value = step["total"]
            total = value if total is None else total + value
            n_loss += 1
        if view is not None:
            view.pump()
    if total is None or n_loss == 0:
        return None, 0
    loss = total / n_loss
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    return float(loss.detach().item()), n_loss


def _publish(view: TrainWindow, status: TrainStatus) -> None:
    view.update(status)


def train_directory(
    root: Path,
    *,
    epochs: int,
    lr: float,
    capability: float | None,
    export_path: Path | None,
    free_vram_bytes: int | None = None,
    free_ram_bytes: int | None = None,
    view: TrainWindow | None = None,
) -> dict[str, Any]:
    import torch

    from python.train.scene_net import SceneNet

    root = Path(root)
    rows = read_state_jsonl(root / "state.jsonl")
    file_bytes = measure_driving_file(root)
    if free_vram_bytes is None:
        free_vram_bytes = read_free_vram_bytes()
    if free_ram_bytes is None:
        free_ram_bytes = read_free_ram_bytes()
    windows = memory_windows(rows, MEMORY_S)
    own_view = view is None
    if view is None:
        view = TrainWindow()
    view.load_folder(root)

    def _empty(precision: str = "fp32") -> dict[str, Any]:
        if export_path is not None:
            export_scene_onnx(export_path)
        return {
            "strips": len(rows),
            "windows": 0,
            "supervised": 0,
            "precision": precision,
            "capability": capability,
            "export": None if export_path is None else str(export_path),
            "device": "cpu",
            "batch": 0,
            "recommended": False,
            "file_bytes": file_bytes,
            "free_vram_bytes": int(free_vram_bytes),
            "free_ram_bytes": int(free_ram_bytes),
        }

    if not windows:
        return _empty()

    window_strips = max(len(window) for window in windows)
    plan = plan_fit(
        file_bytes=file_bytes,
        n_strips=len(rows),
        free_vram_bytes=int(free_vram_bytes),
        free_ram_bytes=int(free_ram_bytes),
        n_windows=len(windows),
        window_strips=window_strips,
    )
    torch.manual_seed(0)
    holder: dict[str, Any] = {}

    def _arm(device_name: str) -> None:
        device = torch.device(device_name)
        if "net" not in holder:
            holder["net"] = SceneNet().to(device)
        elif holder["device"].type != device_name:
            holder["net"].to(device)
            _release_cuda()
        cap = _capability(device, capability)
        amp = precision_for_capability(cap) == "amp" and device.type == "cuda"
        holder["device"] = device
        holder["cap"] = cap
        holder["amp"] = amp
        holder["precision"] = "amp" if amp else "fp32"
        holder["opt"] = torch.optim.Adam(holder["net"].parameters(), lr=float(lr))
        holder["scaler"] = _scaler(amp)

    def probe(device_name: str, batch: int) -> bool:
        if device_name != "cuda":
            return True
        if not torch.cuda.is_available():
            return False
        _arm("cuda")
        return _probe_cuda(holder["net"], int(batch), window_strips, bool(holder["amp"]), holder["device"])

    if plan.device == "cuda":
        plan = fit_with_probe(plan, probe, n_windows=len(windows))
    _arm(plan.device)

    device = holder["device"]
    net = holder["net"]
    batch = int(plan.batch)
    recommended = bool(plan.recommended) and device.type == "cuda"
    note = "recommended" if recommended else "not recommended"
    print(
        f"[GVD] scene fit file={file_bytes} free_vram={int(free_vram_bytes)} "
        f"free_ram={int(free_ram_bytes)} device={device.type} batch={batch} {note}",
        flush=True,
    )
    view.open()
    epochs_n = max(1, int(epochs))
    supervised = 0
    try:
        remaining = ((len(windows) + batch - 1) // batch) * epochs_n
        used, kind = _memory_reading(device.type)
        _publish(
            view,
            TrainStatus(
                eta_s=None,
                loss=None,
                lr=float(lr),
                steps_per_sec=0.0,
                memory_bytes=used,
                memory_kind=kind,
                device=device.type,
                batch=batch,
                recommended=recommended,
                step=0,
                steps=remaining,
            ),
        )
        epoch = 0
        cursor = 0
        done = 0
        last_loss: float | None = None
        started = time.perf_counter()
        while epoch < epochs_n:
            if cursor >= len(windows):
                epoch += 1
                cursor = 0
                continue
            chunk = windows[cursor : cursor + batch]
            try:
                step_loss, n_sup = _optimizer_step(
                    net,
                    holder["opt"],
                    holder["scaler"],
                    root,
                    chunk,
                    device=device,
                    amp=bool(holder["amp"]),
                    view=view,
                )
                if step_loss is not None:
                    last_loss = step_loss
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                net.zero_grad(set_to_none=True)
                _release_cuda()
                smaller = shrink_batch(batch) if device.type == "cuda" else 0
                if device.type == "cuda" and smaller >= 1:
                    batch = smaller
                    recommended = True
                    continue
                if device.type == "cuda" and int(free_ram_bytes) > int(free_vram_bytes):
                    cpu_batch = largest_batch(
                        usable_bytes(int(free_ram_bytes)),
                        file_bytes=file_bytes,
                        n_strips=len(rows),
                        window_strips=window_strips,
                        limit=len(windows),
                    )
                    if cpu_batch < 1:
                        raise TrainFitError(
                            "the training step fits neither free VRAM nor a larger pool of CPU RAM"
                        ) from exc
                    _arm("cpu")
                    device = holder["device"]
                    net = holder["net"]
                    batch = cpu_batch
                    recommended = False
                    continue
                raise TrainFitError(
                    "the training step fits neither free VRAM nor a larger pool of CPU RAM"
                ) from exc
            supervised += n_sup
            cursor += len(chunk)
            done += 1
            elapsed = time.perf_counter() - started
            remaining_windows = (epochs_n - epoch - 1) * len(windows) + (len(windows) - cursor)
            remaining_steps = (remaining_windows + batch - 1) // batch if remaining_windows else 0
            used, kind = _memory_reading(device.type)
            _publish(
                view,
                TrainStatus(
                    eta_s=predict_eta_s(done, elapsed, remaining_steps),
                    loss=last_loss,
                    lr=float(holder["opt"].param_groups[0]["lr"]),
                    steps_per_sec=steps_per_second(done, elapsed),
                    memory_bytes=used,
                    memory_kind=kind,
                    device=device.type,
                    batch=batch,
                    recommended=recommended,
                    step=done,
                    steps=done + remaining_steps,
                ),
            )
        if export_path is not None:
            export_scene_onnx(export_path, net)
    finally:
        if own_view:
            view.close()
    return {
        "strips": len(rows),
        "windows": len(windows),
        "supervised": supervised,
        "precision": holder["precision"],
        "capability": holder["cap"],
        "export": None if export_path is None else str(export_path),
        "device": device.type,
        "batch": batch,
        "recommended": recommended,
        "file_bytes": file_bytes,
        "free_vram_bytes": int(free_vram_bytes),
        "free_ram_bytes": int(free_ram_bytes),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    export_path = None if args.export is None else Path(args.export)
    if export_path is not None and not args.strips:
        if not torch_ready():
            print("[GVD] scene export needs PyTorch")
            sys.exit(2)
        path = export_scene_onnx(export_path)
        print(f"[GVD] exported {path}")
        return
    if not args.strips:
        print("[GVD] scene train: pass --strips, or --export to write the ONNX graph")
        return
    if not torch_ready():
        print("[GVD] scene train needs PyTorch")
        sys.exit(2)
    try:
        stats = train_directory(
            Path(args.strips),
            epochs=args.epochs,
            lr=args.lr,
            capability=args.capability,
            export_path=export_path,
        )
    except TrainFitError as exc:
        print(f"[GVD] scene train: {exc}")
        sys.exit(2)
    note = "recommended" if stats["recommended"] else "not recommended"
    print(
        f"[GVD] scene train strips={stats['strips']} windows={stats['windows']} "
        f"supervised={stats['supervised']} precision={stats['precision']} "
        f"device={stats['device']} batch={stats['batch']} {note}"
    )
    if stats["export"]:
        print(f"[GVD] exported {stats['export']}")


if __name__ == "__main__":
    main()
