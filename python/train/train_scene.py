"""Train the scene net from a strip directory.

Reads ``meta.json``, ``strips/*.jpg``, and ``state.jsonl``. Each step uses the
recorded seconds since the previous strip. The GRU state carries across
overlapping chunks of one recording and clears on a gap or a new recording.
Backprop stops at the chunk. It does not run through the 15 second carry.
Before the first step the trainer measures that directory and the free
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
from python.train.continual import (  # noqa: E402
    GRAD_CLIP,
    WEIGHT_DECAY,
    chunk_stride,
    chunk_vram_budget,
    choose_chunk_length,
    holdout_stopped,
    offcenter_view,
    ordered_learning_rates,
    pass_plan,
    pick_learning_rate,
)
from python.train.fit import (  # noqa: E402
    TrainFitError,
    bytes_per_strip,
    fit_with_probe,
    largest_batch,
    measure_driving_file,
    plan_fit,
    read_free_ram_bytes,
    read_free_vram_bytes,
    read_used_ram_bytes,
    read_used_vram_bytes,
    usable_bytes,
)
from python.train.scene_net import (  # noqa: E402
    OUTPUT_FIELDS,
    ScenePrediction,
    SceneState,
    checkpoint_arch,
    default_onnx_path,
    export_scene_onnx,
    labels_from_tech,
    loss_mask,
    precision_for_capability,
    scene_loss,
    step_dt,
    torch_ready,
)
from python.train.checkpoint import (  # noqa: E402
    clear_checkpoint,
    load_checkpoint,
    resume_fields,
    save_checkpoint,
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


def _detach_state(state: SceneState) -> SceneState:
    """Copy the hidden state off the graph so the next chunk cannot backprop into this one."""
    return SceneState(
        h=state.h.detach().clone(),
        tokens=state.tokens.detach().clone(),
        mask=state.mask.detach().clone(),
    )


def overlap_gradient_cut() -> dict[str, bool] | None:
    """Backward two overlapping chunks. ``None`` when PyTorch is not installed.

    The second chunk starts from ``_detach_state`` at the stride, the same cut
    ``_optimizer_step`` uses between chunks. Its gradient has to land on its
    own frames and must not land on the first chunk's frames.
    """
    if not torch_ready():
        return None
    import torch
    from torch import nn

    torch.manual_seed(0)
    hidden = 8
    length = 4
    stride = 2
    step = nn.Linear(hidden, hidden)

    def cell(frame: Any, state: SceneState) -> tuple[Any, SceneState]:
        mixed = torch.tanh(step(state.h.reshape(1, hidden) + frame.reshape(1, hidden)))
        h_new = mixed.reshape(1, 1, hidden)
        tokens = state.tokens + frame.reshape(1, 1, hidden)
        mask = state.mask + 0.2
        return mixed.sum(), SceneState(h=h_new, tokens=tokens, mask=mask)

    def run(frames: list[Any], state: SceneState) -> tuple[Any, SceneState]:
        state = _detach_state(state)
        total = None
        mark: SceneState | None = None
        for tick, frame in enumerate(frames):
            value, state = cell(frame, state)
            total = value if total is None else total + value
            if tick + 1 == stride:
                mark = state
        end = _detach_state(state)
        carry = end if mark is None else _detach_state(mark)
        return total, carry

    first = [torch.randn(hidden, requires_grad=True) for _ in range(length)]
    second = [torch.randn(hidden, requires_grad=True) for _ in range(length)]
    state = SceneState(
        h=torch.zeros(1, 1, hidden),
        tokens=torch.zeros(1, 1, hidden),
        mask=torch.zeros(1, 1),
    )
    _loss1, carry = run(first, state)
    loss2, _carry2 = run(second, carry)
    loss2.backward()

    def reached(frames: list[Any]) -> bool:
        for frame in frames:
            grad = frame.grad
            if grad is not None and float(grad.detach().abs().sum()) > 0.0:
                return True
        return False

    return {
        "overlap": stride < length,
        "earlier_reached": reached(first),
        "own_reached": reached(second),
    }


def _frame_row(root: Path, row: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    frame = np.ascontiguousarray(_load_bgr(root, int(row["i"])))
    # The stitch is not a yaw roll, so this returns the frame and the row.
    frame, row = offcenter_view(frame, row)
    return frame, row


def _optimizer_step(
    net: Any,
    opt: Any,
    scaler: Any,
    root: Path,
    rows: list[Mapping[str, Any]],
    state: SceneState,
    *,
    device: Any,
    amp: bool,
    view: TrainWindow | None,
    stride: int,
    clip: float,
) -> tuple[float | None, int, SceneState, SceneState]:
    """Backprop through this chunk only.

    ``state`` is already detached from the previous chunk. The returned carry
    is the state ``stride`` frames in, also detached, so the next backward
    pass starts before this chunk ends and does not grow the graph.
    """
    import torch

    net.train()
    opt.zero_grad(set_to_none=True)
    state = _detach_state(state)
    total = None
    n_loss = 0
    stride_mark: SceneState | None = None
    step_at = max(1, int(stride))
    for tick, row in enumerate(rows):
        frame, row = _frame_row(root, row)
        image = torch.from_numpy(frame).to(device)
        dt = torch.tensor([step_dt(row.get("dt_s"))], dtype=torch.float32, device=device)
        with _autocast(amp):
            pred, state = net.forward_batch(image, dt, state)
        if loss_mask(row):
            step = scene_loss(_prediction_at(pred, 0), _as_target(labels_from_tech(row.get("tech")), device))
            value = step["total"]
            total = value if total is None else total + value
            n_loss += 1
        if tick + 1 == step_at:
            stride_mark = state
        if view is not None:
            view.pump()
    end_state = _detach_state(state)
    carry = end_state if stride_mark is None else _detach_state(stride_mark)
    if total is None or n_loss == 0:
        return None, 0, carry, end_state
    loss = total / n_loss
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip))
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip))
        opt.step()
    net.zero_grad(set_to_none=True)
    return float(loss.detach().item()), n_loss, carry, end_state


def _eval_sequence(
    net: Any,
    root: Path,
    rows: list[Mapping[str, Any]],
    state: SceneState,
    *,
    device: Any,
    amp: bool,
    view: TrainWindow | None,
) -> tuple[float | None, int]:
    """Holdout forward. No optimizer step and no gradient through these frames."""
    import torch

    net.eval()
    total = None
    n_loss = 0
    try:
        with torch.no_grad():
            for row in rows:
                frame, row = _frame_row(root, row)
                image = torch.from_numpy(frame).to(device)
                dt = torch.tensor([step_dt(row.get("dt_s"))], dtype=torch.float32, device=device)
                with _autocast(amp):
                    pred, state = net.forward_batch(image, dt, state)
                if not loss_mask(row):
                    continue
                value = scene_loss(_prediction_at(pred, 0), _as_target(labels_from_tech(row.get("tech")), device))["total"]
                total = value if total is None else total + value
                n_loss += 1
                if view is not None:
                    view.pump()
    finally:
        net.train()
    if total is None or n_loss == 0:
        return None, 0
    return float((total / n_loss).detach().item()), n_loss


def _publish(view: TrainWindow, status: TrainStatus, *, record_loss: bool = True) -> None:
    view.update(status, record_loss=record_loss)


def _wait_for_start(view: TrainWindow) -> bool:
    """Wait until Start. With no window, the run begins instead of blocking."""
    if view.consume_start():
        return True
    if not view.available:
        return True
    while view.available and view.window_visible():
        view.pump()
        if view.consume_start():
            return True
        time.sleep(0.05)
    return False


def _to_numpy_tree(obj: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {str(key): _to_numpy_tree(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy_tree(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return np.array(obj, copy=True)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"cannot pause {type(obj).__name__}")


def _from_numpy_tree(obj: Any) -> Any:
    import torch

    if isinstance(obj, np.ndarray):
        return torch.tensor(np.array(obj, copy=True))
    if isinstance(obj, dict):
        return {str(key): _from_numpy_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_from_numpy_tree(value) for value in obj]
    return obj


def pause_matches_stem(fields: Mapping[str, Any]) -> bool:
    """True when this pause file was written for the current GroupNorm stem."""
    return fields.get("arch") == checkpoint_arch()


def stamp_pause(payload: dict[str, Any]) -> dict[str, Any]:
    """Mark a pause file with the stem it may load into."""
    payload["arch"] = checkpoint_arch()
    return payload


def _apply_checkpoint(net: Any, opt: Any, scaler: Any, fields: Mapping[str, Any], device: Any) -> None:
    """Load the paused weights and optimizer. Does not touch the strip folder.

    An older conv-relu file has no GroupNorm tag. It is refused before any
    weight is copied, so those tensors are not applied to the new stem.
    """
    if not pause_matches_stem(fields):
        raise ValueError(
            f"pause checkpoint arch {fields.get('arch')!r} does not match the GroupNorm stem {checkpoint_arch()}"
        )
    import torch

    current = net.state_dict()
    mapped = {}
    for key, value in fields["weights"].items():
        if key not in current:
            raise KeyError(f"pause weight {key} is not in the scene net")
        ref = current[key]
        mapped[key] = torch.tensor(np.array(value, copy=True), dtype=ref.dtype, device=ref.device)
    missing = [key for key in current if key not in mapped]
    if missing:
        raise KeyError(f"pause checkpoint is missing {missing[0]}")
    net.load_state_dict(mapped)
    optimizer = fields.get("optimizer") or {}
    if optimizer:
        tree = _from_numpy_tree(optimizer)
        state = tree.get("state")
        if isinstance(state, dict):
            tree["state"] = {int(key): value for key, value in state.items()}
        try:
            opt.load_state_dict(tree)
        except (RuntimeError, ValueError, KeyError):
            # Weights are already loaded. A mismatched optimizer (an older Adam
            # file, for example) keeps a fresh AdamW instead of dropping them.
            return
        for bucket in opt.state.values():
            for key, value in list(bucket.items()):
                if torch.is_tensor(value):
                    bucket[key] = value.to(device)
    scaler_state = fields.get("scaler")
    if scaler is not None and scaler_state:
        try:
            scaler.load_state_dict(_from_numpy_tree(scaler_state))
        except Exception:
            pass


def _save_pause(
    root: Path,
    net: Any,
    opt: Any,
    scaler: Any,
    view: TrainWindow,
    *,
    step: int,
    epoch: int,
    cursor: int,
    loss: float | None,
    lr: float,
    epochs: int,
    batch: int,
    device_name: str,
    recommended: bool,
) -> dict[str, Any]:
    """Save weights before the loop stops. Strip files are not rewritten."""
    payload = stamp_pause(
        {
            "step": int(step),
            "epoch": int(epoch),
            "cursor": int(cursor),
            "loss": loss,
            "lr": float(lr),
            "epochs": int(epochs),
            "batch": int(batch),
            "device": str(device_name),
            "recommended": bool(recommended),
            "losses": [float(value) for value in view.losses],
            "weights": {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()},
            "optimizer": _to_numpy_tree(opt.state_dict()),
            "scaler": None if scaler is None else _to_numpy_tree(scaler.state_dict()),
        }
    )
    save_checkpoint(root, payload)
    return payload


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
    own_view = view is None
    if view is None:
        view = TrainWindow()
    view.load_folder(root)
    # The CLI epoch count is not the stop. The holdout decides.
    requested_epochs = int(epochs)

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
            "requested_epochs": requested_epochs,
        }

    if not rows:
        return _empty()

    frame_bytes = bytes_per_strip(file_bytes, len(rows))
    chunk_len = choose_chunk_length(frame_bytes, chunk_vram_budget(int(free_vram_bytes)))
    stride = chunk_stride(chunk_len)
    plan = plan_fit(
        file_bytes=file_bytes,
        n_strips=len(rows),
        free_vram_bytes=int(free_vram_bytes),
        free_ram_bytes=int(free_ram_bytes),
        n_windows=1,
        window_strips=chunk_len,
    )
    torch.manual_seed(0)
    holder: dict[str, Any] = {}

    def _arm(device_name: str, learning_rate: float | None = None) -> None:
        device = torch.device(device_name)
        if "net" not in holder:
            holder["net"] = SceneNet().to(device)
        elif holder["device"].type != device_name:
            holder["net"].to(device)
            _release_cuda()
        cap = _capability(device, capability)
        amp = precision_for_capability(cap) == "amp" and device.type == "cuda"
        rate = float(holder.get("lr", lr) if learning_rate is None else learning_rate)
        holder["device"] = device
        holder["cap"] = cap
        holder["amp"] = amp
        holder["precision"] = "amp" if amp else "fp32"
        holder["lr"] = rate
        holder["opt"] = torch.optim.AdamW(holder["net"].parameters(), lr=rate, weight_decay=WEIGHT_DECAY)
        holder["scaler"] = _scaler(amp)

    def probe(device_name: str, batch: int) -> bool:
        if device_name != "cuda":
            return True
        if not torch.cuda.is_available():
            return False
        _arm("cuda")
        return _probe_cuda(holder["net"], int(batch), chunk_len, bool(holder["amp"]), holder["device"])

    if plan.device == "cuda":
        plan = fit_with_probe(plan, probe, n_windows=1)
    _arm(plan.device)

    device = holder["device"]
    net = holder["net"]
    batch = 1
    recommended = bool(plan.recommended) and device.type == "cuda"
    note = "recommended" if recommended else "not recommended"
    print(
        f"[GVD] scene fit file={file_bytes} free_vram={int(free_vram_bytes)} "
        f"free_ram={int(free_ram_bytes)} device={device.type} batch={batch} "
        f"chunk={chunk_len} {note}",
        flush=True,
    )
    view.open()
    supervised = 0
    paused_exit = False
    epoch = 0
    cursor = 0
    done = 0
    last_loss: float | None = None
    windows = 0
    started = time.perf_counter()

    def _clone_weights() -> dict[str, Any]:
        return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}

    def _load_weights(blob: Mapping[str, Any]) -> None:
        current = net.state_dict()
        mapped = {
            key: value.detach().to(device=current[key].device, dtype=current[key].dtype)
            for key, value in blob.items()
        }
        net.load_state_dict(mapped)

    def _pause_now() -> bool:
        nonlocal paused_exit, device, net
        payload = _save_pause(
            root,
            net,
            holder["opt"],
            holder["scaler"],
            view,
            step=done,
            epoch=epoch,
            cursor=cursor,
            loss=last_loss,
            lr=float(holder["lr"]),
            epochs=requested_epochs,
            batch=batch,
            device_name=device.type,
            recommended=recommended,
        )
        view.mark_paused(payload)
        if not _wait_for_start(view):
            paused_exit = True
            return True
        view.mark_running()
        return False

    def _state_for(incoming: str, carry: SceneState | None, end_state: SceneState | None, chunk_in: SceneState | None) -> SceneState:
        if incoming == "reset" or (incoming == "carry" and carry is None):
            return net.initial_state(device=device, batch=1)
        if incoming == "chunk":
            if chunk_in is None:
                return net.initial_state(device=device, batch=1)
            return chunk_in
        if incoming == "end":
            if end_state is None:
                return net.initial_state(device=device, batch=1)
            return end_state
        if carry is None:
            return net.initial_state(device=device, batch=1)
        return carry

    def _train_passes() -> float | None:
        nonlocal paused_exit, epoch, cursor, done, last_loss, supervised
        nonlocal device, net, batch, recommended, windows
        history: list[float] = []
        best_value: float | None = None
        best_blob: dict[str, Any] | None = None
        while not paused_exit:
            plan_steps = pass_plan(rows, chunk_len, stride)
            windows = sum(1 for item in plan_steps if item["kind"] == "train")
            carry: SceneState | None = None
            end_state: SceneState | None = None
            chunk_in: SceneState | None = None
            hold_sum = 0.0
            hold_n = 0
            index = 0
            while index < len(plan_steps) and not paused_exit:
                if view.consume_pause() and _pause_now():
                    break
                item = plan_steps[index]
                state_in = _state_for(str(item["incoming"]), carry, end_state, chunk_in)
                if item["kind"] == "holdout":
                    held_loss, held_count = _eval_sequence(
                        net,
                        root,
                        list(item["rows"]),
                        state_in,
                        device=device,
                        amp=bool(holder["amp"]),
                        view=view,
                    )
                    if held_loss is not None and held_count:
                        hold_sum += float(held_loss) * int(held_count)
                        hold_n += int(held_count)
                    index += 1
                    continue
                if item["incoming"] != "chunk":
                    chunk_in = state_in
                try:
                    step_loss, n_sup, carry, end_state = _optimizer_step(
                        net,
                        holder["opt"],
                        holder["scaler"],
                        root,
                        list(item["rows"]),
                        state_in,
                        device=device,
                        amp=bool(holder["amp"]),
                        view=view,
                        stride=int(item["stride"]),
                        clip=GRAD_CLIP,
                    )
                except RuntimeError as exc:
                    if not _is_oom(exc):
                        raise
                    net.zero_grad(set_to_none=True)
                    _release_cuda()
                    if device.type != "cuda" or not (int(free_ram_bytes) > int(free_vram_bytes)):
                        raise TrainFitError(
                            "the training step fits neither free VRAM nor a larger pool of CPU RAM"
                        ) from exc
                    cpu_batch = largest_batch(
                        usable_bytes(int(free_ram_bytes)),
                        file_bytes=file_bytes,
                        n_strips=len(rows),
                        window_strips=chunk_len,
                        limit=1,
                    )
                    if cpu_batch < 1:
                        raise TrainFitError(
                            "the training step fits neither free VRAM nor a larger pool of CPU RAM"
                        ) from exc
                    _arm("cpu")
                    device = holder["device"]
                    net = holder["net"]
                    batch = 1
                    recommended = False
                    index = 0
                    carry = None
                    end_state = None
                    chunk_in = None
                    continue
                if step_loss is not None:
                    last_loss = step_loss
                supervised += n_sup
                cursor += 1
                done += 1
                index += 1
                elapsed = time.perf_counter() - started
                remaining_steps = sum(1 for later in plan_steps[index:] if later["kind"] == "train")
                used, kind = _memory_reading(device.type)
                _publish(
                    view,
                    TrainStatus(
                        eta_s=predict_eta_s(done, elapsed, remaining_steps),
                        loss=last_loss,
                        lr=float(holder["lr"]),
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
            if paused_exit:
                break
            if hold_n <= 0:
                break
            mean = hold_sum / hold_n
            history.append(mean)
            if best_value is None or mean < best_value:
                best_value = mean
                best_blob = _clone_weights()
            if holdout_stopped(history):
                break
            epoch += 1
        if not paused_exit and best_blob is not None:
            _load_weights(best_blob)
        return best_value

    try:
        if not _wait_for_start(view):
            paused_exit = True
        else:
            existing = load_checkpoint(root)
            if existing is not None:
                fields = resume_fields(existing)
                _arm(device.type, float(fields["lr"]))
                device = holder["device"]
                net = holder["net"]
                _apply_checkpoint(net, holder["opt"], holder["scaler"], fields, device)
                # Weights, step, and optimizer resume. The pass starts again so the
                # GRU state is rebuilt; that activation is not stored in the pause file.
                epoch = int(fields["epoch"])
                cursor = int(fields["cursor"])
                done = int(fields["step"])
                last_loss = fields["loss"]
                view.restore_losses(fields["losses"])
                view.mark_running()
                used, kind = _memory_reading(device.type)
                _publish(
                    view,
                    TrainStatus(
                        eta_s=None,
                        loss=last_loss,
                        lr=float(holder["lr"]),
                        steps_per_sec=0.0,
                        memory_bytes=used,
                        memory_kind=kind,
                        device=device.type,
                        batch=batch,
                        recommended=recommended,
                        step=done,
                        steps=done,
                    ),
                    record_loss=False,
                )
                started = time.perf_counter()
                _train_passes()
            else:
                # No pause file: a new run. Strip JPEGs and state lines stay.
                init_blob = _clone_weights()
                scores: dict[float, float] = {}
                blobs: dict[float, dict[str, Any]] = {}
                series: dict[float, list[float]] = {}
                view.mark_running()
                for rate in ordered_learning_rates(lr):
                    _load_weights(init_blob)
                    _arm(device.type, rate)
                    started = time.perf_counter()
                    device = holder["device"]
                    net = holder["net"]
                    epoch = 0
                    cursor = 0
                    done = 0
                    last_loss = None
                    view.losses = []
                    score = _train_passes()
                    if paused_exit:
                        break
                    scores[rate] = float("inf") if score is None else float(score)
                    blobs[rate] = _clone_weights()
                    series[rate] = [float(value) for value in view.losses]
                if not paused_exit and scores:
                    winner = pick_learning_rate(scores)
                    _load_weights(blobs[winner])
                    holder["lr"] = float(winner)
                    view.restore_losses(series.get(winner, []))
        if not paused_exit and export_path is not None:
            export_scene_onnx(export_path, net)
        if not paused_exit:
            clear_checkpoint(root)
    finally:
        if own_view:
            view.close()
    return {
        "strips": len(rows),
        "windows": windows,
        "supervised": supervised,
        "precision": holder.get("precision", "fp32"),
        "capability": holder.get("cap", capability),
        "export": None if export_path is None else str(export_path),
        "device": device.type,
        "batch": batch,
        "recommended": recommended,
        "file_bytes": file_bytes,
        "free_vram_bytes": int(free_vram_bytes),
        "free_ram_bytes": int(free_ram_bytes),
        "requested_epochs": requested_epochs,
        "chunk": chunk_len,
        "lr": float(holder.get("lr", lr)),
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
