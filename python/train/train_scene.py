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
import faulthandler
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.strip_writer import CANVAS_H, CANVAS_W  # noqa: E402
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
    remaining_until_holdout_can_stop,
)
from python.train.fit import (  # noqa: E402
    TrainFitError,
    bytes_per_strip,
    fit_with_probe,
    largest_batch,
    plan_fit,
    step_bytes,
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
    labels_for_row,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the pause checkpoint when it exists. This is also the default.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start at step 0. A later checkpoint still replaces the pause file.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Overfit the first 8 strips. Exit 1 unless the lane loss falls below 0.05.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two steps and print tensor shapes.",
    )
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="Do not open the training window. The run starts on its own.",
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="Start without the Start button. The window still opens when a display is attached.",
    )
    parser.add_argument("--metrics-every", type=int, default=10, help="Write metrics.jsonl every N steps.")
    return parser


def headless_requested() -> bool:
    return os.environ.get("GVD_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}


def interactive_display() -> bool:
    """True when a desktop is attached. A service session and a headless Linux box are false."""
    if sys.platform == "win32":
        try:
            import ctypes

            # SM_CMONITORS
            return int(ctypes.windll.user32.GetSystemMetrics(80)) > 0
        except (AttributeError, OSError):
            return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_panel(*, no_panel: bool, autostart: bool) -> tuple[bool, bool]:
    """Whether to open the window, and whether to start without the Start button.

    No interactive display, ``GVD_HEADLESS=1``, and ``--no-panel`` all start
    on their own. ``--autostart`` skips the button and can still show the panel.
    """
    headless = headless_requested() or not interactive_display()
    open_panel = not bool(no_panel) and not headless
    start_now = bool(autostart) or bool(no_panel) or headless
    return open_panel, start_now


# Two optimizer steps on a handful of frames. Longer than a slow disk read,
# short enough that a stuck Start button or a decode wait does not sit for
# a quarter hour with no stack.
DRY_RUN_TIMEOUT_S = 300.0


def _load_bgr(root: Path, index: int) -> Any:
    """Training canvas. Raw cameras are stitched here, not in the recording.

    A PNG that does not decode is read from ``raw_repaired`` when that copy
    exists. The original file is left as it was written.
    """
    from python.train.scene_loader import load_canvas

    return load_canvas(root, index)


def arm_dry_run_watchdog(timeout_s: float = DRY_RUN_TIMEOUT_S) -> None:
    """Dump every thread and exit if the dry run is still inside ``timeout_s``."""
    faulthandler.enable(file=sys.stderr, all_threads=True)
    print(
        f"[GVD] dry-run watchdog {float(timeout_s):.0f}s; "
        "a hang dumps every thread stack and exits",
        flush=True,
    )
    faulthandler.dump_traceback_later(
        float(timeout_s),
        repeat=False,
        file=sys.stderr,
        exit=True,
    )


def disarm_dry_run_watchdog() -> None:
    faulthandler.cancel_dump_traceback_later()


def _confirm_start(
    view: TrainWindow,
    *,
    dry_run: bool,
    sanity: bool = False,
    autostart: bool = False,
) -> bool:
    """Dry-run, sanity, and autostart begin immediately. A panel run waits for Start."""
    if dry_run or sanity or autostart:
        if dry_run:
            why = "dry-run"
        elif sanity:
            why = "sanity"
        else:
            why = "autostart"
        print(f"[GVD] {why}: starting without the Start button", flush=True)
        return True
    return _wait_for_start(view)


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


def _frame_row(root: Path, row: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]] | None:
    """Stitch one tick. A missing frame is a gap, not a crash."""
    try:
        frame = np.ascontiguousarray(_load_bgr(root, int(row["i"])))
    except FileNotFoundError as exc:
        print(f"[GVD] scene load skip i={row.get('i')}: {exc}", flush=True)
        return None
    # The stitch is not a yaw roll, so this returns the frame and the row.
    frame, row = offcenter_view(frame, row)
    return frame, row


def _print_step_shapes(image: Any, pred: Any, step: Mapping[str, Any], state: SceneState) -> None:
    print(f"[GVD] dry image={tuple(image.shape)} dtype={getattr(image, 'dtype', None)}", flush=True)
    fields = getattr(pred, "__dataclass_fields__", {})
    names = list(fields) if fields else [name for name in dir(pred) if not name.startswith("_")]
    for name in names:
        value = getattr(pred, name, None)
        if value is None:
            print(f"[GVD] dry pred.{name}=None", flush=True)
        elif hasattr(value, "shape"):
            print(f"[GVD] dry pred.{name}={tuple(value.shape)} dtype={value.dtype}", flush=True)
    hidden = getattr(state, "h", None)
    if hidden is not None and hasattr(hidden, "shape"):
        print(f"[GVD] dry state.h={tuple(hidden.shape)}", flush=True)
    for key, value in step.items():
        if hasattr(value, "shape"):
            print(f"[GVD] dry loss.{key}={tuple(value.shape)}", flush=True)


def _grad_bad_counts(net: Any) -> tuple[int, int]:
    import torch

    nan_n = 0
    inf_n = 0
    for param in net.parameters():
        grad = param.grad
        if grad is None:
            continue
        detached = grad.detach()
        nan_n += int(torch.isnan(detached).sum().item())
        inf_n += int(torch.isinf(detached).sum().item())
    return nan_n, inf_n


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
    trace: dict[str, Any] | None = None,
) -> tuple[float | None, int, SceneState, SceneState]:
    """Backprop through this chunk only.

    ``state`` is already detached from the previous chunk. The returned carry
    is the state ``stride`` frames in, also detached, so the next backward
    pass starts before this chunk ends and does not grow the graph.
    """
    import torch

    from python.train.scene_net import LOSS_TERMS

    started = time.perf_counter()
    net.train()
    opt.zero_grad(set_to_none=True)
    state = _detach_state(state)
    total = None
    n_loss = 0
    head_acc = {key: 0.0 for key in LOSS_TERMS}
    nan_n = 0
    inf_n = 0
    stride_mark: SceneState | None = None
    step_at = max(1, int(stride))
    for tick, row in enumerate(rows):
        if trace is not None and trace.get("print_shapes"):
            print(
                f"[GVD] scene load {tick + 1}/{len(rows)} i={row.get('i')}",
                flush=True,
            )
        loaded = _frame_row(root, row)
        if loaded is None:
            continue
        frame, row = loaded
        image = torch.from_numpy(frame).to(device)
        dt = torch.tensor([step_dt(row.get("dt_s"))], dtype=torch.float32, device=device)
        with _autocast(amp):
            pred, state = net.forward_batch(image, dt, state)
        if loss_mask(row):
            one = _prediction_at(pred, 0)
            step = scene_loss(one, _as_target(labels_for_row(row), device))
            if trace is not None and trace.get("print_shapes") and not trace.get("shapes_printed"):
                _print_step_shapes(image, one, step, state)
                trace["shapes_printed"] = True
            value = step["total"]
            item = float(value.detach().item())
            if math.isnan(item):
                nan_n += 1
            elif not math.isfinite(item):
                inf_n += 1
            for key in LOSS_TERMS:
                head_acc[key] += float(step[key].detach().item())
            total = value if total is None else total + value
            n_loss += 1
        if tick + 1 == step_at:
            stride_mark = state
        if view is not None:
            view.pump()
    end_state = _detach_state(state)
    carry = end_state if stride_mark is None else _detach_state(stride_mark)
    if total is None or n_loss == 0:
        if trace is not None:
            trace.update(
                {
                    "loss": None,
                    "heads": {key: None for key in LOSS_TERMS},
                    "n_loss": 0,
                    "grad_norm": None,
                    "nan": nan_n,
                    "inf": inf_n,
                    "step_s": time.perf_counter() - started,
                    "lane_gt": None,
                }
            )
        return None, 0, carry, end_state
    loss = total / n_loss
    grad_norm = None
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip)))
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip)))
        opt.step()
    bad_nan, bad_inf = _grad_bad_counts(net)
    nan_n += bad_nan
    inf_n += bad_inf
    net.zero_grad(set_to_none=True)
    heads = {key: head_acc[key] / n_loss for key in LOSS_TERMS}
    if trace is not None:
        trace.update(
            {
                "loss": float(loss.detach().item()),
                "heads": heads,
                "n_loss": n_loss,
                "grad_norm": grad_norm,
                "nan": nan_n,
                "inf": inf_n,
                "step_s": time.perf_counter() - started,
                "lane_gt": heads.get("lane_gt"),
            }
        )
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
    pairs: list[tuple[float, float]] | None = None,
) -> tuple[float | None, int]:
    """Holdout forward. No optimizer step and no gradient through these frames."""
    import torch

    net.eval()
    total = None
    n_loss = 0
    try:
        with torch.no_grad():
            for row in rows:
                loaded = _frame_row(root, row)
                if loaded is None:
                    continue
                frame, row = loaded
                image = torch.from_numpy(frame).to(device)
                dt = torch.tensor([step_dt(row.get("dt_s"))], dtype=torch.float32, device=device)
                with _autocast(amp):
                    pred, state = net.forward_batch(image, dt, state)
                one = _prediction_at(pred, 0)
                if pairs is not None:
                    from python.debug.lane_targets import (
                        lane_center_is_valid,
                        lane_gt_from_record,
                        lane_x_from_vector,
                    )

                    lane = getattr(one, "lane_gt", None)
                    vec = None if lane is None else lane.detach().float().cpu().numpy().reshape(-1)
                    tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
                    gt_vec, gt_mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
                    if lane_center_is_valid(gt_mask):
                        pred_x = None if vec is None else lane_x_from_vector(vec)
                        gt_x = lane_x_from_vector(gt_vec)
                        pairs.append(
                            (
                                float("nan") if pred_x is None else float(pred_x),
                                float("nan") if gt_x is None else float(gt_x),
                            )
                        )
                if not loss_mask(row):
                    continue
                value = scene_loss(one, _as_target(labels_for_row(row), device))["total"]
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
    resume: bool = True,
    sanity: bool = False,
    dry_run: bool = False,
    metrics_every: int = 10,
    status_root: Path | None = None,
    autostart: bool = False,
    open_panel: bool = True,
) -> dict[str, Any]:
    import torch

    from python.train.scene_net import SceneNet

    root = Path(root)
    from python.data.raw_frames import load_scene_rows, training_frame_bytes
    from python.train.scene_loader import default_num_workers, select_training_rows

    print("[GVD] scene index: reading state.jsonl", flush=True)
    state_rows = load_scene_rows(root)
    print(f"[GVD] scene index: {len(state_rows)} state rows", flush=True)
    print("[GVD] scene index: measuring frames", flush=True)

    def _index_progress(n_frames: int, n_bytes: int) -> None:
        print(f"[GVD] scene index: {n_frames} frames {n_bytes} bytes", flush=True)

    file_bytes, _n_image_files = training_frame_bytes(root, progress=_index_progress)
    rows, missing_rows, excluded_rows = select_training_rows(root, state_rows)
    print(
        f"[GVD] scene index: {len(rows)} rows with frames, {missing_rows} missing, "
        f"{excluded_rows} stationary or tipped",
        flush=True,
    )
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
        stats = _empty()
        stats["sanity_ok"] = False if sanity or dry_run else None
        stats["dry_run"] = bool(dry_run)
        return stats

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
    # One recording is one GRU stream. A larger batch would be parallel
    # recordings, which this folder does not have. parallel_fit is how many
    # such streams the free VRAM could hold at this chunk length.
    batch = 1
    parallel_fit = 0
    if int(free_vram_bytes) > 0 and len(rows) > 0:
        parallel_fit = largest_batch(
            usable_bytes(int(free_vram_bytes)),
            file_bytes=file_bytes,
            n_strips=len(rows),
            window_strips=chunk_len,
            limit=32,
        )
    est_step = step_bytes(
        batch,
        file_bytes=file_bytes,
        n_strips=max(1, len(rows)),
        window_strips=chunk_len,
    )
    recommended = bool(plan.recommended) and device.type == "cuda"
    note = "recommended" if recommended else "not recommended"
    workers = default_num_workers()
    print(
        f"[GVD] scene fit file={file_bytes} free_vram={int(free_vram_bytes)} "
        f"free_ram={int(free_ram_bytes)} device={device.type} batch={batch} "
        f"chunk={chunk_len} step={est_step} parallel_fit={parallel_fit} "
        f"workers={workers} {note}",
        flush=True,
    )
    from python.debug.remote_status import (
        CHECKPOINT_S,
        DRY_RUN_STEPS,
        HEARTBEAT_S,
        SANITY_STEPS,
        append_metrics,
        gpu_util_percent,
        json_num,
        sanity_reached,
        status_dir,
        write_crash,
        write_heartbeat,
    )

    debug_dir = status_dir(status_root)
    run_started = time.time()
    write_heartbeat(
        debug_dir,
        {
            "kind": "train",
            "phase": "loading",
            "started": run_started,
            "updated": time.time(),
            "step": 0,
            "epoch": 0,
            "loss": None,
            "eta_s": None,
            "lr": float(holder.get("lr", lr)),
            "device": str(device.type),
            "strips": str(root),
            "free_ram_bytes": int(free_ram_bytes),
            "free_vram_bytes": int(free_vram_bytes),
            "workers": int(workers),
            "dry_run": bool(dry_run),
        },
    )
    print(f"[GVD] scene status: heartbeat {debug_dir}", flush=True)
    if dry_run:
        arm_dry_run_watchdog(DRY_RUN_TIMEOUT_S)
    supervised = 0
    paused_exit = False
    epoch = 0
    cursor = 0
    done = 0
    last_loss: float | None = None
    windows = 0
    started = time.perf_counter()
    last_batch: dict[str, Any] = {"step": 0, "rows": []}
    last_heart = time.monotonic()
    last_ckpt = time.monotonic()
    step_trace: dict[str, Any] = {}
    sanity_ok: bool | None = None
    every = max(1, int(metrics_every))
    steps_per_epoch = 0
    eta_pass: float | None = None
    rates_after = 0
    schedule_printed = False

    def _log_step(kind: str, eta: float | None, mem_bytes: int, *, save: bool) -> None:
        nonlocal last_heart, last_ckpt
        heads = {
            key: json_num(value)
            for key, value in (step_trace.get("heads") or {}).items()
        }
        record = {
            "kind": kind,
            "t": time.time(),
            "step": int(done),
            "epoch": int(epoch),
            "loss": json_num(step_trace.get("loss")),
            "heads": heads,
            "grad_norm": json_num(step_trace.get("grad_norm")),
            "lr": json_num(holder.get("lr")),
            "step_s": json_num(step_trace.get("step_s")),
            "gpu_mem_bytes": int(mem_bytes) if device.type == "cuda" else None,
            "gpu_util": gpu_util_percent() if device.type == "cuda" else None,
            "nan": int(step_trace.get("nan") or 0),
            "inf": int(step_trace.get("inf") or 0),
        }
        if int(done) == 1 or int(done) % every == 0:
            append_metrics(debug_dir, record)
        now = time.monotonic()
        if int(done) == 1 or now - last_heart >= HEARTBEAT_S:
            write_heartbeat(
                debug_dir,
                {
                    "kind": "train",
                    "started": run_started,
                    "updated": time.time(),
                    "step": int(done),
                    "epoch": int(epoch),
                    "epochs": int(requested_epochs),
                    "loss": json_num(last_loss),
                    "eta_s": json_num(eta),
                    "eta_pass_s": json_num(eta_pass),
                    "steps_per_epoch": int(steps_per_epoch),
                    "supervised": int(supervised),
                    "windows": int(done),
                    "lr": json_num(holder.get("lr")),
                    "device": str(device.type),
                    "strips": str(root),
                },
            )
            last_heart = now
        if save and now - last_ckpt >= CHECKPOINT_S:
            _save_pause(
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
            last_ckpt = now
            print(f"[GVD] checkpoint step={done}", flush=True)

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
        nonlocal device, net, batch, recommended, windows, last_batch
        nonlocal steps_per_epoch, eta_pass, schedule_printed
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
            val_pairs: list[tuple[float, float]] = []
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
                        pairs=val_pairs,
                    )
                    if held_loss is not None and held_count:
                        hold_sum += float(held_loss) * int(held_count)
                        hold_n += int(held_count)
                    index += 1
                    continue
                if item["incoming"] != "chunk":
                    chunk_in = state_in
                last_batch = {
                    "step": int(done),
                    "epoch": int(epoch),
                    "rows": [row.get("i") for row in item["rows"]],
                }
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
                        trace=step_trace,
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
                steps_per_epoch = sum(1 for item in plan_steps if item["kind"] == "train")
                has_holdout = any(item["kind"] == "holdout" for item in plan_steps)
                if not schedule_printed:
                    schedule_printed = True
                    print(
                        f"[GVD] scene schedule steps_per_epoch={steps_per_epoch} "
                        f"epoch={epoch} rates_after={rates_after} "
                        f"--epochs {requested_epochs} is recorded; the holdout stops the run",
                        flush=True,
                    )
                job_steps = remaining_until_holdout_can_stop(
                    remaining_steps,
                    steps_per_epoch,
                    epoch=epoch,
                    has_holdout=has_holdout,
                    rates_after=rates_after,
                )
                used, kind = _memory_reading(device.type)
                eta_pass = predict_eta_s(done, elapsed, remaining_steps)
                eta = predict_eta_s(done, elapsed, job_steps)
                _log_step("train", eta, used, save=True)
                _publish(
                    view,
                    TrainStatus(
                        eta_s=eta,
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
            scored: dict[str, Any] = {}
            if val_pairs:
                from python.debug.lane_targets import score_centers

                scored = score_centers(
                    [pair[0] for pair in val_pairs],
                    [pair[1] for pair in val_pairs],
                )
                append_metrics(
                    debug_dir,
                    {
                        "kind": "val",
                        "t": time.time(),
                        "step": int(done),
                        "epoch": int(epoch),
                        "loss": json_num(mean),
                        "n": scored["n"],
                        "r": json_num(scored["r"]),
                        "rms": json_num(scored["rms"]),
                        "rms_mean": json_num(scored["rms_mean"]),
                        "bias": json_num(scored["bias"]),
                    },
                )
                print(
                    f"[GVD] val step={done} n={scored['n']} r={scored['r']} "
                    f"rms={scored['rms']} rms_mean={scored['rms_mean']} "
                    f"bias={scored['bias']} loss={mean}",
                    flush=True,
                )
            # Stop on the 8 m center rms, not the composite loss. A tiny drop in
            # the other heads was keeping the run alive after the lane had plateaued.
            if scored.get("rms") is not None:
                history.append(float(scored["rms"]))
                improved = best_value is None or float(scored["rms"]) < best_value
            else:
                history.append(mean)
                improved = best_value is None or mean < best_value
            if improved:
                best_value = history[-1]
                best_blob = _clone_weights()
            plateau = holdout_stopped(history, tol=0.02, window=3) if scored.get("rms") is not None else holdout_stopped(history)
            if plateau:
                print(
                    f"[GVD] holdout stopped at step={done} "
                    f"rms={scored.get('rms')} rms_mean={scored.get('rms_mean')}",
                    flush=True,
                )
                break
            epoch += 1
        if not paused_exit and best_blob is not None:
            _load_weights(best_blob)
        return best_value

    def _limited(limit: int, subset: list[Mapping[str, Any]], shapes: bool) -> None:
        nonlocal last_loss, done, sanity_ok, last_batch, supervised, windows
        step_trace["print_shapes"] = shapes
        if not subset:
            sanity_ok = False
            print("[GVD] scene train: no strips for this mode", flush=True)
            return
        state = net.initial_state(device=device, batch=1)
        lane_loss = None
        for step_i in range(max(1, int(limit))):
            last_batch = {
                "step": step_i,
                "epoch": 0,
                "rows": [row.get("i") for row in subset],
            }
            loss, n_sup, _carry, state = _optimizer_step(
                net,
                holder["opt"],
                holder["scaler"],
                root,
                list(subset),
                state,
                device=device,
                amp=bool(holder["amp"]),
                view=view,
                stride=max(1, len(subset)),
                clip=GRAD_CLIP,
                trace=step_trace,
            )
            done = step_i + 1
            windows = done
            supervised += int(n_sup)
            if loss is not None:
                last_loss = loss
            lane_loss = step_trace.get("lane_gt")
            used, _kind = _memory_reading(device.type)
            _log_step("dry" if shapes else "sanity", None, used, save=False)
            if shapes:
                continue
            if sanity_reached(lane_loss):
                sanity_ok = True
                print(f"[GVD] sanity lane_gt={lane_loss} step={done}", flush=True)
                return
        if not shapes:
            sanity_ok = sanity_reached(lane_loss)
            print(f"[GVD] sanity lane_gt={lane_loss} step={done} ok={sanity_ok}", flush=True)

    try:
        if open_panel:
            view.open()
        if not _confirm_start(
            view,
            dry_run=dry_run,
            sanity=sanity,
            autostart=autostart or not open_panel,
        ):
            paused_exit = True
        elif dry_run or sanity:
            view.mark_running()
            subset = list(rows[:4] if dry_run else rows[:8])
            _limited(DRY_RUN_STEPS if dry_run else SANITY_STEPS, subset, bool(dry_run))
        else:
            existing = load_checkpoint(root) if resume else None
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
                rates = ordered_learning_rates(lr)
                for rate_index, rate in enumerate(rates):
                    rates_after = len(rates) - rate_index - 1
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
        if not paused_exit and not dry_run and not sanity and export_path is not None:
            export_scene_onnx(export_path, net)
        if not paused_exit and not dry_run and not sanity:
            clear_checkpoint(root)
    except Exception:
        write_crash(debug_dir, traceback.format_exc(), last_batch)
        raise
    finally:
        if dry_run:
            disarm_dry_run_watchdog()
        if own_view:
            view.close()
    return {
        "strips": len(rows),
        "windows": int(done),
        "supervised": int(supervised),
        "steps_per_epoch": int(steps_per_epoch),
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
        "sanity_ok": sanity_ok,
        "dry_run": bool(dry_run),
    }


def main(argv: list[str] | None = None) -> None:
    from python.data.data_root import DataDriveError, check_data_path, ensure_cache_dirs, scene_lanes_dir

    try:
        ensure_cache_dirs()
    except DataDriveError as exc:
        print(f"[GVD] scene train: {exc}")
        sys.exit(2)
    args = build_parser().parse_args(argv)
    export_path = None if args.export is None else Path(args.export)
    if export_path is not None and not args.strips:
        try:
            check_data_path(export_path)
        except DataDriveError as exc:
            print(f"[GVD] scene train: {exc}")
            sys.exit(2)
        if not torch_ready():
            print("[GVD] scene export needs PyTorch")
            sys.exit(2)
        path = export_scene_onnx(export_path)
        print(f"[GVD] exported {path}")
        return
    if not args.strips:
        args.strips = str(scene_lanes_dir())
    try:
        check_data_path(args.strips)
        if export_path is not None:
            check_data_path(export_path)
    except DataDriveError as exc:
        print(f"[GVD] scene train: {exc}")
        sys.exit(2)
    if not torch_ready():
        print("[GVD] scene train needs PyTorch")
        sys.exit(2)
    open_panel, start_now = resolve_panel(no_panel=bool(args.no_panel), autostart=bool(args.autostart))
    if not open_panel:
        print("[GVD] scene train: no panel", flush=True)
    try:
        stats = train_directory(
            Path(args.strips),
            epochs=args.epochs,
            lr=args.lr,
            capability=args.capability,
            export_path=export_path,
            resume=not bool(args.no_resume),
            sanity=bool(args.sanity) and not bool(args.dry_run),
            dry_run=bool(args.dry_run),
            metrics_every=int(args.metrics_every),
            autostart=start_now,
            open_panel=open_panel,
        )
    except TrainFitError as exc:
        print(f"[GVD] scene train: {exc}")
        sys.exit(2)
    if args.sanity and not args.dry_run and not stats.get("sanity_ok"):
        print("[GVD] sanity failed: lane_gt loss did not fall below 0.05", flush=True)
        sys.exit(1)
    note = "recommended" if stats["recommended"] else "not recommended"
    print(
        f"[GVD] scene train strips={stats['strips']} steps={stats['windows']} "
        f"supervised={stats['supervised']} steps_per_epoch={stats.get('steps_per_epoch', 0)} "
        f"precision={stats['precision']} "
        f"device={stats['device']} batch={stats['batch']} {note}"
    )
    if stats["export"]:
        print(f"[GVD] exported {stats['export']}")


if __name__ == "__main__":
    main()
