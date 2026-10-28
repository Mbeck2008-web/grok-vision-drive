"""Train the scene net from a strip directory.

Reads ``state.jsonl`` and the lossless ``raw/<cam>/NNNNNN.png`` camera frames
(``raw/`` is required: a folder with only lossy ``strips/*.jpg`` is refused).
Each step uses the
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

from python.data.raw_frames import LossyFramesError  # noqa: E402
from python.train.event_mask import MaskFileError  # noqa: E402
from python.data.strip_writer import CANVAS_H, CANVAS_W  # noqa: E402
from python.train.continual import (  # noqa: E402
    CHUNK_MAX,
    GRAD_CLIP,
    WEIGHT_DECAY,
    chunk_stride,
    chunk_vram_budget,
    choose_chunk_length,
    holdout_stopped,
    offcenter_view,
    pass_plan,
    remaining_until_holdout_can_stop,
    resume_learning_rate,
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
    LaneGtCoverage,
    ScenePrediction,
    SceneState,
    checkpoint_arch,
    default_onnx_path,
    export_scene_onnx,
    has_lane_gt,
    head_names,
    labels_for_row,
    loss_mask,
    precision_for_capability,
    scene_loss,
    step_dt,
    torch_ready,
)
from python.train.checkpoint import (  # noqa: E402
    PauseDirRefused,
    check_pause_dir,
    checkpoint_path,
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


def _pin_optimizer_lr(opt: Any, rate: float) -> None:
    """Write ``rate`` onto every param group.

    Loading a pause restores the learning rate that was saved with it. A
    1e-4 group must not replace the command-line rate. A group with an
    ``lr_mult`` (``--backbone-lr-mult``, default 1.0) gets ``rate * lr_mult``.
    """
    groups = getattr(opt, "param_groups", None)
    if not isinstance(groups, list):
        return
    pinned = float(rate)
    for group in groups:
        if isinstance(group, dict):
            group["lr"] = pinned * float(group.get("lr_mult", 1.0))


def resolve_scene_batch(requested: int, parallel_fit: int) -> int:
    """Windows in one optimizer step.

    ``requested`` 0 uses the batch that fits in free VRAM. A positive value
    is that many windows. The floor is one window.
    """
    if int(requested) > 0:
        return int(requested)
    if int(parallel_fit) > 0:
        return int(parallel_fit)
    return 1


def _maybe_jitter(frame: Any, augment: bool, rng: np.random.RandomState | None) -> Any:
    """Photometric jitter on a decoded frame. The row's metres are not an input."""
    if not augment or frame is None:
        return frame
    from python.train.cam_strip import photometric_jitter

    return photometric_jitter(frame, rng)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the GVD scene net from recorded strips")
    parser.add_argument("--strips", type=str, default="", help="Recording directory with state.jsonl and raw/<cam>/ frames. JPEG strips/ are refused.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Windows per optimizer step. 0 fits the largest batch in free VRAM.",
    )
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
        "--pause-dir",
        default="",
        help=(
            "Folder for this run's pause file (<strips name>.gvd-pause.npz): every pause write, "
            "the start-up read, resume, and the delete after a finished run use it. Refused when it "
            "is the folder holding the strip folder, under it, or above it, and (with --no-resume) "
            "when the pause file already exists. Default: the folder holding the strip folder."
        ),
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
    parser.add_argument(
        "--cams",
        default="",
        help="Comma list of cameras to train and export. Empty uses all eight.",
    )
    parser.add_argument(
        "--heads",
        default="",
        help="Comma list of heads that receive loss. Empty trains every head. The export still has every output.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Photometric jitter on training steps. Off for validation. --no-augment disables it.",
    )
    parser.add_argument(
        "--cache",
        default="",
        help="Directory of pre-decoded camera chunks. The loader uses them when the index is present.",
    )
    parser.add_argument(
        "--build-cache",
        default="",
        help="Decode --cams into this directory (chunks under 3.5 GB) and exit.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Max gradient norm. 1.0 clips the spikes from a curve after a straight.",
    )
    parser.add_argument(
        "--lane-loss",
        default="huber",
        choices=("huber", "l1", "mse"),
        help="lane_gt regression. huber is smooth-L1 and is the default.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Repeat windows with curves, weaves, ramps, or large steer.",
    )
    parser.add_argument(
        "--min-valid",
        type=float,
        default=0.3,
        help="Drop a training window when fewer than this fraction of rows have a lane center.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers. Default is 4 on Windows and 0 elsewhere.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Downscale the canvas before the cache and the net. 1 keeps the recorded size.",
    )
    parser.add_argument(
        "--holdout-blocks",
        type=int,
        default=None,
        help=(
            "Hold out the last 20%% of each of K equal time blocks in a long segment, "
            "with a 4 s gap on both sides of each slice. Default 1 (or GVD_HOLDOUT_BLOCKS) "
            "is the single tail cut."
        ),
    )
    parser.add_argument(
        "--holdout-cells",
        action="store_true",
        help=(
            "Spatial holdout: a seeded set of 250 m map cells with about 20%% of rows "
            "(fast and slow cells in proportion) is scored, and train drops rows within "
            "50 m of them. The list is saved and reused (see --holdout-cells-file)."
        ),
    )
    parser.add_argument(
        "--holdout-cells-file",
        type=str,
        default="",
        help="Cell list JSON. Loaded when it exists, else chosen and written. Default: <status dir>/holdout_cells.json.",
    )
    parser.add_argument("--holdout-seed", type=int, default=0, help="Seed for choosing --holdout-cells.")
    parser.add_argument(
        "--conf-weight",
        type=float,
        default=DEFAULT_CONF_WEIGHT,
        help=(
            "Scale on the lane_gt confidence BCE in the training loss (default 0.1). "
            "Model selection uses the 8 m center rms only."
        ),
    )
    parser.add_argument(
        "--init-head-bias",
        action="store_true",
        help=(
            "On a fresh start only, set lane_gt_head.bias to per-slot Huber constants "
            "of the train rows. Off by default; a diagnostic, not a fix."
        ),
    )
    parser.add_argument(
        "--eval-chunk",
        type=int,
        default=CHUNK_MAX,
        help=(
            "Holdout resets the GRU every N frames, like the 16-frame training windows "
            "(default 16). The old single-sequence score is reported as r_long/rms_long/bias_long. "
            "0 scores the single sequence only."
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=("scratch", "imagenet-resnet18", "mobilenetv3"),
        default="scratch",
        help=(
            "scratch is the GRU scene net. imagenet-resnet18 / mobilenetv3 is a single-frame "
            "baseline: a torchvision ImageNet backbone per frame, no GRU, the heads read "
            "per-frame features. BGR->RGB and ImageNet mean/std happen inside the net."
        ),
    )
    parser.add_argument(
        "--mask-events",
        action="store_true",
        help=(
            "Leave out rows inside an impacts.jsonl window [pre_t0, impact_end] (matched by time t) "
            "and rows the collector flagged settle or stall. Off by default."
        ),
    )
    parser.add_argument(
        "--mask-files",
        default="",
        help=(
            "Comma list of mask sidecars (JSON lines with a row id i), relative to the recording "
            "folder. Every listed row is left out. A missing file or an i the recording does not "
            "have stops the run. name@sha256prefix (8+ hex digits) also stops on a sha256 mismatch."
        ),
    )
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help=(
            "With --backbone imagenet-*, the backbone trains at --lr times this. Default 1.0 "
            "(one rate for every weight). Not used by the scratch net."
        ),
    )
    parser.add_argument(
        "--ckpt-dir",
        default="",
        help=(
            "Folder for the per-epoch checkpoints (<strips>.run<UTC start>.ckpt_ep<N>_step<S>.npz, "
            "the weights behind that epoch's val row). Default: the folder holding the strip folder."
        ),
    )
    parser.add_argument(
        "--ckpt-min-free-gb",
        type=float,
        default=None,
        help="Skip (and log) an epoch checkpoint that would leave less than this many GB free. Default 8.",
    )
    parser.add_argument(
        "--grad-ckpt",
        action="store_true",
        help=(
            "Gradient checkpointing on the per-tick encoder (the shared stem over every sector): its "
            "activations are recomputed in backward instead of stored. Less memory, more compute; the "
            "gradients, checkpoints and ONNX are the same. Scratch net only. Off by default."
        ),
    )
    parser.add_argument(
        "--cam-age",
        action="store_true",
        help=(
            "Hold-last side cameras: a camera that did not render this tick shows its most recent earlier "
            "frame (same segment, no respawn or train/holdout boundary, at most 1.5 s old) and the net gets "
            "8 per-camera ages (1/15 s ticks; -1 = black). The ONNX gets an 'ages' input and gvd_cam_age=1. "
            "Off by default: the live BeamNG backend cannot feed the ages yet, and the runtime refuses such a model."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="With --backbone imagenet-*, keep the ImageNet weights fixed and train only the projection and heads.",
    )
    return parser


def resolve_holdout_blocks(value: int | None) -> int:
    """CLI value, else ``GVD_HOLDOUT_BLOCKS``, else 1. Below 1 is an error."""
    from python.train.continual import holdout_blocks_from_env

    if value is None:
        return holdout_blocks_from_env()
    if int(value) < 1:
        raise SystemExit("--holdout-blocks must be at least 1")
    return int(value)


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


def _load_bgr(root: Path, index: int, *, allow_lossy: bool = False) -> Any:
    """Training canvas. Raw cameras are stitched here, not in the recording.

    A PNG that does not decode is read from ``raw_repaired`` when that copy
    exists. The original file is left as it was written. JPEG ``strips/``
    raise ``LossyFramesError``; only eval_scene_lanes passes ``allow_lossy``.
    """
    from python.train.scene_loader import load_canvas

    return load_canvas(root, index, allow_lossy=allow_lossy)


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


# Confidence BCE scale in the training loss. 1.0 let the 0/1 paint-coverage
# target dominate the lane loss; the val score never uses it.
DEFAULT_CONF_WEIGHT = 0.1
# A held-out 8 m center spread below this is a constant prediction.
COLLAPSE_STD_M = 0.05
LANE_SLOT_NAMES = (
    "c5", "l5", "r5", "c10", "l10", "r10", "c20", "l20", "r20", "c30", "l30", "r30", "heading", "conf_logit",
)
# Frames whose main area was all zero and were skipped in this process.
BLANK_MAIN_SKIPS = 0


# Near-black main sector: mean below 2% of full scale, or more than 98% of
# pixels with every channel below 8/255. Exact zeros missed nearly-black
# frames (a dark or failed capture with a few noisy pixels).
NEAR_BLACK_MEAN = 0.02
NEAR_BLACK_LEVEL = 8
NEAR_BLACK_FRAC = 0.98


def near_black(pixels: Any) -> bool:
    """True for an HxWx3 (or HxW) uint8 / 0-1 float area that is (nearly) black."""
    arr = np.asarray(pixels)
    if arr.size == 0:
        return True
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
    if float(arr.mean()) < NEAR_BLACK_MEAN * 255.0:
        return True
    peak = arr.max(axis=-1) if arr.ndim == 3 else arr
    return float((peak < NEAR_BLACK_LEVEL).mean()) > NEAR_BLACK_FRAC


def main_area_blank(frame: Any, cams: tuple[str, ...] | None, scale: float = 1.0) -> bool:
    """True when ``main`` is a trained camera and its area of ``frame`` is near-black.

    A missing front picture is stitched as a black sector, and that row still
    carries a real lane label. See ``near_black`` for the threshold.
    """
    from python.data.strip_writer import STRIP_ORDER
    from python.train.cam_strip import scaled_layout

    names = tuple(cams) if cams else tuple(STRIP_ORDER)
    if frame is None or "main" not in names:
        return False
    sectors, _height, width = scaled_layout(names, scale if float(scale) < 0.999 else 1.0)
    arr = np.asarray(frame)
    if arr.ndim < 2 or int(arr.shape[1]) != int(width):
        return False
    for sec in sectors:
        if sec.cam_id == "main":
            return near_black(arr[:, sec.x0 : sec.x1])
    return False


def count_near_black_main(
    rows: list[Mapping[str, Any]], cache: Any, cams: tuple[str, ...] | None, scale: float = 1.0
) -> tuple[int, int] | None:
    """(near-black, checked) over the cached main canvases of ``rows``. None without a cache."""
    if cache is None or (cams and "main" not in cams):
        return None
    from python.data.strip_writer import CANVAS_W, STRIP_ORDER
    from python.train.cam_strip import extract_cams, scale_canvas

    black = checked = 0
    for row in rows:
        index = int(row["i"])
        if not cache.has(index):
            continue
        frame = cache.canvas(index)
        if frame is None:
            continue
        if cams and tuple(cams) != tuple(STRIP_ORDER) and int(frame.shape[1]) == CANVAS_W:
            frame = extract_cams(frame, tuple(cams))
        if scale < 0.999:
            frame = scale_canvas(frame, scale, cams)
        checked += 1
        black += int(main_area_blank(frame, cams, scale))
    return black, checked


def huber_slot_constants(
    rows: list[Mapping[str, Any]],
    *,
    delta: float = 1.0,
    iters: int = 20,
) -> np.ndarray:
    """Per-slot Huber-optimal constant of the lane_gt targets over ``rows``.

    Slots 0..12 start at the median of the valid values and take ``iters``
    reweighted means (weight ``min(1, delta/|residual|)``), which is the
    smooth-L1 minimiser the loss uses. Slot 13 is the head's pre-sigmoid
    confidence: the logit of the mean target. An empty slot is 0.
    """
    from python.debug.lane_targets import LANE_GT_DIM, lane_gt_from_record

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else None
        if not isinstance(tech, Mapping) or not tech:
            continue
        vec, mask = lane_gt_from_record(tech)
        values.append(np.asarray(vec, dtype=np.float64))
        masks.append(np.asarray(mask, dtype=np.float64))
    out = np.zeros((LANE_GT_DIM,), dtype=np.float64)
    if not values:
        return out.astype(np.float32)
    V = np.stack(values)
    M = np.stack(masks)
    for j in range(LANE_GT_DIM - 1):
        col = V[M[:, j] > 0.5, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        c = float(np.median(col))
        for _ in range(int(iters)):
            res = np.abs(col - c)
            w = np.minimum(1.0, float(delta) / np.maximum(res, 1e-12))
            c = float(np.sum(w * col) / np.sum(w))
        out[j] = c
    conf = V[M[:, -1] > 0.5, -1]
    conf = conf[np.isfinite(conf)]
    if conf.size:
        p = float(np.clip(conf.mean(), 1e-3, 1.0 - 1e-3))
        out[-1] = math.log(p / (1.0 - p))
    return out.astype(np.float32)


def apply_head_bias(net: Any, constants: Any) -> None:
    """Copy ``constants`` into ``net.lane_gt_head.bias``."""
    import torch

    with torch.no_grad():
        bias = net.lane_gt_head.bias
        bias.copy_(torch.as_tensor(np.asarray(constants, dtype=np.float32), device=bias.device, dtype=bias.dtype))


def _stack_states(*states: SceneState) -> SceneState:
    import torch

    return SceneState(
        h=torch.cat([s.h for s in states], dim=1),
        tokens=torch.cat([s.tokens for s in states], dim=0),
        mask=torch.cat([s.mask for s in states], dim=0),
    )


def _reset_stream(net: Any, state: SceneState, b: int, device: Any) -> SceneState:
    """``state`` with stream ``b`` cleared to a fresh GRU state."""
    fresh = net.initial_state(device=device, batch=1)
    h = state.h.clone()
    tokens = state.tokens.clone()
    mask = state.mask.clone()
    h[:, b] = fresh.h[:, 0].to(dtype=h.dtype)
    tokens[b] = fresh.tokens[0].to(dtype=tokens.dtype)
    mask[b] = fresh.mask[0].to(dtype=mask.dtype)
    return SceneState(h=h, tokens=tokens, mask=mask)


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
            canvas_h = int(getattr(net, "canvas_h", CANVAS_H))
            canvas_w = int(getattr(net, "canvas_w", CANVAS_W))
            strip = torch.zeros(int(batch), 3, canvas_h, canvas_w, device=device)
            dt = torch.zeros(int(batch), device=device)
            with _autocast(amp):
                pred, state = _forward(net, strip, dt, state, [None] * int(batch))
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


def _state_first(state: SceneState) -> SceneState:
    """One stream, for the next window that is not part of this stacked step."""
    return SceneState(
        h=state.h[:, :1].detach().clone(),
        tokens=state.tokens[:1].detach().clone(),
        mask=state.mask[:1].detach().clone(),
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


def held_canvas(root: Path, row: Mapping[str, Any], cache: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Hold-last canvas (full 8-sector strip) and 8 ages for one planned row.

    Raw path: ``raw_frames.load_held_bgr``. Cache path: the same per-camera
    sources read from a format-2 cache (``frame_cache.FrameCache.sector``, a
    sector already passed through ``place_sector``), a camera the cache lacks
    read from raw, then the same ``compose_strip``.
    """
    from python.data.cam_hold import hold_sources
    from python.data.raw_frames import held_frames, load_held_bgr, load_raw_camera
    from python.data.strip_writer import compose_strip

    sources = hold_sources(row)
    if cache is None:
        return load_held_bgr(root, sources)
    if getattr(cache, "stale", False):
        cache.require_fresh()
    if float(getattr(cache, "scale", 1.0)) < 0.999:
        raise ValueError("--cam-age reads a cache at scale 1 only")

    def _read(cam: str, index: int) -> Any:
        got = cache.sector(cam, index)
        return got if got is not None else load_raw_camera(root, cam, index)

    frames, ages = held_frames(root, sources, reader=_read)
    if not frames:
        raise FileNotFoundError(f"no decodable frame for row {row.get('i')}'s hold plan")
    return compose_strip(frames), ages


def row_ages(row: Mapping[str, Any] | None) -> np.ndarray:
    """The 8 ages ``_frame_row`` put on a loaded row; all -1 for none (a padded stream)."""
    from python.data.cam_hold import ROW_AGES_KEY, ages_vector, no_ages

    got = None if row is None else row.get(ROW_AGES_KEY)
    if got is None:
        return no_ages()
    arr = np.asarray(got, dtype=np.float32).reshape(-1)
    return arr if arr.size == len(ages_vector({})) else no_ages()


def _forward(net: Any, image: Any, dt: Any, state: SceneState, rows: list[Mapping[str, Any] | None]) -> Any:
    """``net.forward_batch``, with the rows' ages when the net takes them."""
    if not bool(getattr(net, "cam_age", False)):
        return net.forward_batch(image, dt, state)
    import torch

    ages = torch.from_numpy(np.stack([row_ages(row) for row in rows])).to(image.device)
    return net.forward_batch(image, dt, state, ages)


def _frame_row(
    root: Path,
    row: Mapping[str, Any],
    *,
    cams: tuple[str, ...] | None = None,
    cache: Any = None,
    augment: bool = False,
    rng: np.random.RandomState | None = None,
    scale: float = 1.0,
) -> tuple[Any, Mapping[str, Any]] | None:
    """Stitch one tick. A missing frame is a gap, not a crash.

    A row with a hold plan (``cam_hold.ROW_HOLD_KEY``, ``--cam-age``) is
    stitched hold-last: each camera's sector is its plan's source frame, and
    the returned row is a copy that carries the 8 ages (``ROW_AGES_KEY``).
    """
    from python.data.cam_hold import ROW_AGES_KEY, ROW_HOLD_KEY
    from python.data.strip_writer import CANVAS_W, STRIP_ORDER
    from python.train.cam_strip import extract_cams, photometric_jitter

    index = int(row["i"])
    frame = None
    if isinstance(row.get(ROW_HOLD_KEY), Mapping):
        try:
            frame, ages = held_canvas(root, row, cache)
        except FileNotFoundError as exc:
            print(f"[GVD] scene load skip i={row.get('i')}: {exc}", flush=True)
            return None
        row = dict(row)
        row[ROW_AGES_KEY] = [float(value) for value in ages]
    elif cache is not None and cache.has(index):
        frame = cache.canvas(index)
    if frame is None:
        try:
            frame = np.ascontiguousarray(_load_bgr(root, index))
        except FileNotFoundError as exc:
            print(f"[GVD] scene load skip i={row.get('i')}: {exc}", flush=True)
            return None
    # An all-camera cache is the full strip. A main-only net needs that sector cut out.
    if (
        frame is not None
        and cams
        and tuple(cams) != tuple(STRIP_ORDER)
        and int(frame.shape[1]) == CANVAS_W
    ):
        frame = extract_cams(frame, tuple(cams))
    if scale < 0.999:
        from python.train.cam_strip import scale_canvas

        frame = scale_canvas(frame, scale, cams)
    if main_area_blank(frame, cams, scale):
        # No front picture: a black main sector with a real label. Skip it.
        global BLANK_MAIN_SKIPS
        BLANK_MAIN_SKIPS += 1
        return None
    if augment:
        frame = photometric_jitter(frame, rng)
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


def _with_ages(row: Mapping[str, Any], frame_ages: list[Any] | None, tick: int) -> Mapping[str, Any]:
    """A prefetched frame's row with the ages the worker loaded alongside it."""
    from python.data.cam_hold import ROW_AGES_KEY

    if frame_ages is None or tick >= len(frame_ages) or frame_ages[tick] is None:
        return row
    out = dict(row)
    out[ROW_AGES_KEY] = list(frame_ages[tick])
    return out


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
    cams: tuple[str, ...] | None = None,
    cache: Any = None,
    augment: bool = False,
    rng: np.random.RandomState | None = None,
    heads: tuple[str, ...] | None = None,
    lane_loss: str = "huber",
    frames: list[Any] | None = None,
    conf_weight: float = 1.0,
    scale: float = 1.0,
    frame_ages: list[Any] | None = None,
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
    lane_gt_rows = 0
    stride_mark: SceneState | None = None
    step_at = max(1, int(stride))
    for tick, row in enumerate(rows):
        if trace is not None and trace.get("print_shapes"):
            print(
                f"[GVD] scene load {tick + 1}/{len(rows)} i={row.get('i')}",
                flush=True,
            )
        if frames is not None:
            if tick >= len(frames) or frames[tick] is None:
                continue
            frame = _maybe_jitter(frames[tick], augment, rng)
            row = _with_ages(row, frame_ages, tick)
        else:
            loaded = _frame_row(
                root, row, cams=cams, cache=cache, augment=augment, rng=rng, scale=scale,
            )
            if loaded is None:
                continue
            frame, row = loaded
        image = torch.from_numpy(frame).to(device)
        dt = torch.tensor([step_dt(row.get("dt_s"))], dtype=torch.float32, device=device)
        with _autocast(amp):
            pred, state = _forward(net, image, dt, state, [row])
        if loss_mask(row):
            one = _prediction_at(pred, 0)
            labels = labels_for_row(row)
            lane_gt_rows += int(has_lane_gt(labels))
            step = scene_loss(
                one, _as_target(labels, device), heads, lane_loss=lane_loss, conf_weight=conf_weight
            )
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
                    "lane_gt_rows": lane_gt_rows,
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
                "lane_gt_rows": lane_gt_rows,
            }
        )
    return float(loss.detach().item()), n_loss, carry, end_state


def _optimizer_group(
    net: Any,
    opt: Any,
    scaler: Any,
    root: Path,
    sequences: list[list[Mapping[str, Any]]],
    *,
    device: Any,
    amp: bool,
    view: TrainWindow | None,
    clip: float,
    trace: dict[str, Any] | None = None,
    cams: tuple[str, ...] | None = None,
    cache: Any = None,
    augment: bool = False,
    rng: np.random.RandomState | None = None,
    heads: tuple[str, ...] | None = None,
    lane_loss: str = "huber",
    frame_sets: list[list[Any] | None] | None = None,
    conf_weight: float = 1.0,
    scale: float = 1.0,
    age_sets: list[list[Any] | None] | None = None,
) -> tuple[float | None, int, SceneState, SceneState]:
    """One step on several windows that each start from a fresh GRU state.

    Shorter windows are padded for the forward and left out of the loss.
    """
    import torch

    from python.train.scene_net import LOSS_TERMS

    started = time.perf_counter()
    net.train()
    opt.zero_grad(set_to_none=True)
    width = len(sequences)
    length = max((len(seq) for seq in sequences), default=0)
    state = net.initial_state(device=device, batch=max(1, width))
    total = None
    n_loss = 0
    head_acc = {key: 0.0 for key in LOSS_TERMS}
    nan_n = 0
    inf_n = 0
    lane_gt_rows = 0
    blank = None
    presets = list(frame_sets) if frame_sets is not None else [None] * width
    for tick in range(length):
        chosen: list[Any] = []
        alive: list[Mapping[str, Any] | None] = []
        dts: list[float] = []
        for b, seq in enumerate(sequences):
            frame = None
            row = None
            if tick < len(seq):
                preset = presets[b] if b < len(presets) else None
                if preset is not None:
                    if tick < len(preset) and preset[tick] is not None:
                        frame = _maybe_jitter(preset[tick], augment, rng)
                        sets = age_sets[b] if age_sets is not None and b < len(age_sets) else None
                        row = _with_ages(seq[tick], sets, tick)
                else:
                    loaded = _frame_row(
                        root, seq[tick], cams=cams, cache=cache, augment=augment, rng=rng, scale=scale,
                    )
                    if loaded is not None:
                        frame, row = loaded
            if frame is not None and blank is None:
                blank = np.zeros_like(frame)
            chosen.append(frame)
            alive.append(row if frame is not None else None)
            dts.append(step_dt(None if row is None else row.get("dt_s")))
        if blank is None or not any(frame is not None for frame in chosen):
            continue
        stacked = [blank if frame is None else frame for frame in chosen]
        image = torch.from_numpy(np.ascontiguousarray(np.stack(stacked))).to(device)
        dt = torch.tensor(dts, dtype=torch.float32, device=device)
        with _autocast(amp):
            pred, state = _forward(net, image, dt, state, alive)
        for b, row in enumerate(alive):
            if row is None or not loss_mask(row):
                continue
            one = _prediction_at(pred, b)
            labels = labels_for_row(row)
            lane_gt_rows += int(has_lane_gt(labels))
            step = scene_loss(
                one, _as_target(labels, device), heads, lane_loss=lane_loss, conf_weight=conf_weight
            )
            value = step["total"]
            item_loss = float(value.detach().item())
            if math.isnan(item_loss):
                nan_n += 1
            elif not math.isfinite(item_loss):
                inf_n += 1
            for key in LOSS_TERMS:
                head_acc[key] += float(step[key].detach().item())
            total = value if total is None else total + value
            n_loss += 1
        if view is not None:
            view.pump()
    end_state = _state_first(state)
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
                    "lane_gt_rows": lane_gt_rows,
                }
            )
        return None, 0, end_state, end_state
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
    head_mean = {key: head_acc[key] / n_loss for key in LOSS_TERMS}
    if trace is not None:
        trace.update(
            {
                "loss": float(loss.detach().item()),
                "heads": head_mean,
                "n_loss": n_loss,
                "grad_norm": grad_norm,
                "nan": nan_n,
                "inf": inf_n,
                "step_s": time.perf_counter() - started,
                "lane_gt": head_mean.get("lane_gt"),
                "lane_gt_rows": lane_gt_rows,
            }
        )
    return float(loss.detach().item()), n_loss, end_state, end_state


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
    cams: tuple[str, ...] | None = None,
    cache: Any = None,
    heads: tuple[str, ...] | None = None,
    scale: float = 1.0,
    reset_every: int = CHUNK_MAX,
    pairs_long: list[tuple[float, float]] | None = None,
    pair_rows: list[Mapping[str, Any]] | None = None,
    parts: dict[str, float] | None = None,
    conf_weight: float = 1.0,
    preds_out: list[Any] | None = None,
) -> tuple[float | None, int]:
    """Holdout forward. No optimizer step and no gradient through these frames.

    The primary stream clears the GRU every ``reset_every`` rows, the same
    fresh-state 16-frame windows training sees. ``pairs_long`` gets the old
    protocol: one stream through the whole slice from ``state``. Both run as
    one batch of two. ``reset_every`` <= 0 scores the single stream only.
    The loss, ``parts`` (``lane_reg``/``conf_bce`` sums and ``n``),
    ``pair_rows`` (the row behind each primary pair) and ``preds_out``
    (``(row, vector)`` per scored frame) follow the primary stream.
    """
    import torch

    # LATERAL FRAME: gt and prediction are both in the label file's frame (tech.gt_origin):
    # ref node for v102 and earlier, body centreline for v103. The metric is frame-agnostic.
    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record, lane_x_from_vector

    chunked = int(reset_every) > 0
    width = 2 if chunked and pairs_long is not None else 1
    if width == 2:
        state = _stack_states(state, state)
    net.eval()
    total = None
    n_loss = 0
    try:
        with torch.no_grad():
            for k, row in enumerate(rows):
                if chunked and k % int(reset_every) == 0:
                    state = _reset_stream(net, state, 0, device)
                loaded = _frame_row(root, row, cams=cams, cache=cache, augment=False, scale=scale)
                if loaded is None:
                    continue
                frame, row = loaded
                stacked = np.stack([frame] * width) if width > 1 else frame
                image = torch.from_numpy(np.ascontiguousarray(stacked)).to(device)
                dt_value = step_dt(row.get("dt_s"))
                dt = torch.tensor([dt_value] * width, dtype=torch.float32, device=device)
                with _autocast(amp):
                    pred, state = _forward(net, image, dt, state, [row] * width)
                one = _prediction_at(pred, 0)
                tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
                gt_vec, gt_mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
                if preds_out is not None and getattr(one, "lane_gt", None) is not None:
                    preds_out.append((row, one.lane_gt.detach().float().cpu().numpy().reshape(-1)))
                if (pairs is not None or pairs_long is not None) and lane_center_is_valid(gt_mask):
                    gt_x = lane_x_from_vector(gt_vec)
                    gt_f = float("nan") if gt_x is None else float(gt_x)
                    for b in range(width):
                        lane = getattr(_prediction_at(pred, b), "lane_gt", None)
                        vec = None if lane is None else lane.detach().float().cpu().numpy().reshape(-1)
                        pred_x = None if vec is None else lane_x_from_vector(vec)
                        pair = (float("nan") if pred_x is None else float(pred_x), gt_f)
                        if b == 0 and pairs is not None:
                            pairs.append(pair)
                            if pair_rows is not None:
                                pair_rows.append(row)
                        if pairs_long is not None and (b == 1 or width == 1):
                            pairs_long.append(pair)
                if not loss_mask(row):
                    continue
                step = scene_loss(one, _as_target(labels_for_row(row), device), heads, conf_weight=conf_weight)
                value = step["total"]
                total = value if total is None else total + value
                n_loss += 1
                if parts is not None:
                    parts["lane_reg"] = float(parts.get("lane_reg", 0.0)) + float(step["lane_reg"].detach().item())
                    parts["conf_bce"] = float(parts.get("conf_bce", 0.0)) + float(step["conf_bce"].detach().item())
                    parts["n"] = int(parts.get("n", 0)) + 1
                if view is not None:
                    view.pump()
    finally:
        net.train()
    if total is None or n_loss == 0:
        return None, 0
    return float((total / n_loss).detach().item()), n_loss


def val_summary(
    pairs: list[tuple[float, float]],
    pairs_long: list[tuple[float, float]] | None,
    parts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Primary r/rms/bias, the single-sequence ``*_long`` fields, the split
    lane/confidence losses, and the 8 m center prediction spread."""
    from python.debug.lane_targets import score_centers

    scored: dict[str, Any] = dict(score_centers([p[0] for p in pairs], [p[1] for p in pairs])) if pairs else {}
    if pairs_long:
        extra = score_centers([p[0] for p in pairs_long], [p[1] for p in pairs_long])
        scored["r_long"] = extra["r"]
        scored["rms_long"] = extra["rms"]
        scored["bias_long"] = extra["bias"]
        scored["n_long"] = extra["n"]
    preds = np.asarray([p[0] for p in pairs], dtype=np.float64) if pairs else np.zeros((0,))
    preds = preds[np.isfinite(preds)]
    scored["pred_mean"] = float(preds.mean()) if preds.size else None
    scored["pred_std"] = float(preds.std()) if preds.size else None
    n = int((parts or {}).get("n", 0) or 0)
    scored["lane_loss"] = float(parts["lane_reg"]) / n if n else None
    scored["conf_bce"] = float(parts["conf_bce"]) / n if n else None
    scored["collapse"] = bool(scored["pred_std"] is not None and scored["pred_std"] < COLLAPSE_STD_M)
    return scored


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


def pause_matches_stem(
    fields: Mapping[str, Any], cams: tuple[str, ...] | None = None, arch: str | None = None
) -> bool:
    """True when this pause file was written for the current GroupNorm stem.

    ``arch`` is a net's own ``pause_arch`` tag (the ``--backbone`` baseline);
    ``None`` is the GroupNorm stem.
    """
    return fields.get("arch") == (arch or checkpoint_arch(cams))


def stamp_pause(
    payload: dict[str, Any], cams: tuple[str, ...] | None = None, arch: str | None = None
) -> dict[str, Any]:
    """Mark a pause file with the stem it may load into."""
    payload["arch"] = arch or checkpoint_arch(cams)
    return payload


def _apply_checkpoint(net: Any, opt: Any, scaler: Any, fields: Mapping[str, Any], device: Any) -> None:
    """Load the paused weights and optimizer. Does not touch the strip folder.

    An older conv-relu file has no GroupNorm tag. It is refused before any
    weight is copied, so those tensors are not applied to the new stem.
    """
    used_cams = tuple(getattr(net, "cams", ()) or ()) or None
    net_arch = getattr(net, "pause_arch", None)
    if not pause_matches_stem(fields, used_cams, net_arch):
        raise ValueError(
            f"pause checkpoint arch {fields.get('arch')!r} does not match the "
            f"{net_arch or 'GroupNorm stem ' + checkpoint_arch(used_cams)}"
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
    pause_dir: Path | None = None,
) -> dict[str, Any]:
    """Save weights before the loop stops. Strip files are not rewritten."""
    used_cams = tuple(getattr(net, "cams", ()) or ()) or None
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
        },
        used_cams,
        getattr(net, "pause_arch", None),
    )
    save_checkpoint(root, payload, pause_dir=pause_dir)
    return payload


def _first_item(batch: list[Any]) -> Any:
    """Collate for a batch of one window. Must be a module function so spawn can pickle it."""
    return batch[0]


def _release_worker_handles(worker_id: int) -> None:
    """Spawn starts from a pickle. Handles are opened on the first cache read."""
    del worker_id


class _EpochWindows:
    """One training window per index. Workers decode; the step stays on the parent."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        root: Path,
        cams: tuple[str, ...] | None,
        cache_dir: str,
        scale: float,
    ) -> None:
        self.items = list(items)
        self.root = str(root)
        self.cams = None if cams is None else tuple(cams)
        self.cache_dir = str(cache_dir or "")
        self.scale = float(scale)
        self._cache: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache = None

    def __len__(self) -> int:
        return len(self.items)

    def close(self) -> None:
        """Close and drop this process's cache maps (a later read reopens a fresh cache)."""
        cache, self._cache = self._cache, None
        if cache is not None:
            cache.close()

    def _reader(self) -> Any:
        if self._cache is None and self.cache_dir:
            from python.train.frame_cache import FrameCache

            self._cache = FrameCache(self.cache_dir)
        return self._cache

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        frames: list[Any] = []
        ages: list[Any] = []
        from python.data.cam_hold import ROW_AGES_KEY

        for row in item.get("rows") or []:
            loaded = _frame_row(
                Path(self.root),
                row,
                cams=self.cams,
                cache=self._reader(),
                augment=False,
                scale=self.scale,
            )
            frames.append(None if loaded is None else loaded[0])
            ages.append(None if loaded is None else loaded[1].get(ROW_AGES_KEY))
        return {
            "frames": frames,
            "ages": ages,
            "rows": list(item.get("rows") or []),
            "stride": int(item.get("stride") or 1),
            "kind": item.get("kind") or "train",
            "incoming": "reset",
        }


def _prefetch_loader(items: list[dict[str, Any]], root: Path, cams, cache, scale: float, workers: int, cuda: bool):
    """A DataLoader over train windows, or None when this process stays in-line."""
    if int(workers) <= 0 or not items:
        return None
    import torch
    from torch.utils.data import DataLoader

    from python.train.scene_loader import loader_settings

    cache_dir = str(getattr(cache, "root", "") or "")
    dataset = _EpochWindows([item for item in items if item.get("kind") != "holdout"], root, cams, cache_dir, scale)
    if len(dataset) == 0:
        return None
    settings = loader_settings(int(workers), cuda=bool(cuda) and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_first_item,
        worker_init_fn=_release_worker_handles,
        **settings,
    )


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
    cams: tuple[str, ...] | None = None,
    heads: tuple[str, ...] | None = None,
    cache: Any = None,
    augment: bool = True,
    clip: float = GRAD_CLIP,
    lane_loss: str = "huber",
    balance: bool = False,
    min_valid: float = 0.3,
    workers: int = 0,
    scale: float = 1.0,
    batch_size: int = 0,
    holdout_blocks: int = 1,
    holdout_cells: bool = False,
    holdout_cells_file: Path | None = None,
    holdout_seed: int = 0,
    conf_weight: float = DEFAULT_CONF_WEIGHT,
    init_head_bias: bool = False,
    eval_chunk: int = CHUNK_MAX,
    backbone: str = "scratch",
    freeze_backbone: bool = False,
    backbone_lr_mult: float = 1.0,
    mask_events: bool = False,
    mask_files: str = "",
    ckpt_dir: Path | None = None,
    ckpt_min_free_gb: float | None = None,
    cam_age: bool = False,
    grad_ckpt: bool = False,
    pause_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    from python.train.scene_net import SceneNet

    root = Path(root)
    # STRIP_ORDER, as the CLI's parse_cams gives it. The net, its export
    # metadata and the offline eval all use that order; a caller's tuple in
    # another order used to stitch the sectors swapped (r sign flip).
    from python.train.cam_strip import canonical_cams

    train_cams = canonical_cams(cams) if cams else None
    train_heads = tuple(heads) if heads else None
    use_cam_age = bool(cam_age)
    if grad_ckpt and backbone not in ("", "scratch"):
        raise SystemExit("--grad-ckpt checkpoints the scratch net's per-tick encoder; it is not wired into --backbone imagenet-*")
    if use_cam_age and backbone not in ("", "scratch"):
        raise SystemExit("--cam-age appends the ages to the scratch net's token; it is not wired into --backbone imagenet-*")
    if cache is not None and getattr(cache, "stale", False):
        raise SystemExit(f"[GVD] cache refused: {cache.reason}")
    if use_cam_age and cache is not None and float(getattr(cache, "scale", 1.0)) < 0.999:
        raise SystemExit("--cam-age reads a cache at scale 1 only (sector parity with the live strip is proven there)")
    use_augment = bool(augment) and not bool(sanity) and not bool(dry_run)
    augment_rng = np.random.RandomState(0)
    window_rng = np.random.RandomState(1)
    grad_clip = float(clip)
    lane_kind = str(lane_loss or "huber")
    input_scale = float(scale) if float(scale) > 0.0 else 1.0
    from python.data.raw_frames import load_scene_rows, training_frame_bytes
    from python.train.scene_loader import default_num_workers, select_training_rows

    print("[GVD] scene index: reading state.jsonl", flush=True)
    from python.data.raw_frames import refuse_lossy

    refuse_lossy(root)
    state_rows = load_scene_rows(root)
    print(f"[GVD] scene index: {len(state_rows)} state rows", flush=True)
    from python.train.event_mask import drop_masked, mask_file_sources, selection_line

    # Mask files are read on the full recording rows, before any frame is
    # measured, so a bad file, an unknown id or a sha256 mismatch stops early.
    mask_sources = mask_file_sources(root, state_rows, mask_files)
    print("[GVD] scene index: measuring frames", flush=True)

    def _index_progress(n_frames: int, n_bytes: int) -> None:
        print(f"[GVD] scene index: {n_frames} frames {n_bytes} bytes", flush=True)

    file_bytes, _n_image_files = training_frame_bytes(root, progress=_index_progress)
    dropped: dict[str, int] = {}
    rows, missing_rows, excluded_rows = select_training_rows(root, state_rows, cams=train_cams, dropped=dropped)
    print(
        f"[GVD] scene index: {len(rows)} rows with frames, {missing_rows} missing, "
        f"{excluded_rows} stationary or tipped",
        flush=True,
    )
    print(
        f"[GVD] scene index: dropped {int(dropped.get('no_trained_cam', 0))} rows without a decodable "
        f"trained camera ({','.join(train_cams or ()) or 'all'}; main required when trained)",
        flush=True,
    )
    if mask_events:
        from python.train.event_mask import describe, mask_event_rows

        before = len(rows)
        rows, event_counts = mask_event_rows(root, rows)
        print(f"[GVD] mask-events: {describe(event_counts, before)}", flush=True)
    # Dropped before the holdout split, so a masked row is in neither the train
    # chunks nor the holdout. A masked run is one gap; past MEMORY_S it resets the GRU.
    selection_before = rows
    if mask_sources:
        rows = drop_masked(rows, set().union(*(source["ids"] for source in mask_sources)))
    selection_text, _selection_meta = selection_line(state_rows, selection_before, rows, mask_sources)
    print(selection_text, flush=True)
    blank_scan = count_near_black_main(rows, cache, train_cams, input_scale)
    near_rule = (
        f"mean<{NEAR_BLACK_MEAN} or >{NEAR_BLACK_FRAC:.0%} of px below {NEAR_BLACK_LEVEL}/255"
    )
    if blank_scan is None:
        print(f"[GVD] scene index: near-black main ({near_rule}) is skipped as rows load; no cache to count it up front", flush=True)
    else:
        print(
            f"[GVD] scene index: near-black main ({near_rule}): {blank_scan[0]} of {blank_scan[1]} cached rows; "
            "skipped when loaded (blank_main_skipped in the val line)",
            flush=True,
        )
    print(
        f"[GVD] conf_weight={float(conf_weight)} (confidence BCE scale in the training loss; "
        f"model selection uses the 8 m center rms only) eval_chunk={int(eval_chunk)}",
        flush=True,
    )
    cell_spec: dict[str, Any] | None = None
    if holdout_cells:
        from python.train.continual import (
            cell_holdout_labels,
            check_holdout_cells,
            choose_holdout_cells,
            load_holdout_cells,
            save_holdout_cells,
        )
        from python.debug.remote_status import status_dir as _cells_status_dir

        if int(holdout_blocks) > 1:
            raise SystemExit("--holdout-cells and --holdout-blocks > 1 are separate modes; pick one")
        cells_dir = _cells_status_dir(status_root) or root
        cells_path = Path(holdout_cells_file) if holdout_cells_file else Path(cells_dir) / "holdout_cells.json"
        if cells_path.is_file():
            cell_spec = load_holdout_cells(cells_path, rows)
            print(f"[GVD] holdout cells: loaded {len(cell_spec['cells'])} cells from {cells_path}", flush=True)
        else:
            cell_spec = choose_holdout_cells(rows, seed=int(holdout_seed))
            check_holdout_cells(cell_spec, rows, source="chosen cells")
            save_holdout_cells(cell_spec, cells_path)
            print(f"[GVD] holdout cells: chose {len(cell_spec['cells'])} cells (seed {int(holdout_seed)}), saved {cells_path}", flush=True)
        marks = cell_holdout_labels(rows, cell_spec)
        print(
            f"[GVD] holdout cells: train={marks.count('train')} holdout={marks.count('hold')} "
            f"dropped={marks.count('gap')}",
            flush=True,
        )
    elif int(holdout_blocks) > 1:
        print(f"[GVD] holdout blocks: {int(holdout_blocks)}", flush=True)
    from python.train.continual import HOLDOUT_FAST_MPS, holdout_mix

    startup_plan = pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2, blocks=int(holdout_blocks), cells=cell_spec)
    mix = holdout_mix(startup_plan)
    if use_cam_age:
        from python.data.cam_hold import HOLD_CAP_S, HOLD_TICK_HZ, annotate_hold

        planned = annotate_hold(state_rows, rows, startup_plan)
        print(
            f"[GVD] cam-age: hold-last plan on {planned} of {len(rows)} rows "
            f"(cap {HOLD_CAP_S:g} s, ages on the {HOLD_TICK_HZ} Hz tick; no look-ahead; "
            "reset at respawn, segment and train/holdout runs)",
            flush=True,
        )
    print(
        f"[GVD] holdout: rows={mix['rows']} slices={mix['slices']} fast={mix['fast']} slow={mix['slow']} "
        f"no_speed={mix['no_speed']} (fast is row speed >= {HOLDOUT_FAST_MPS} m/s)",
        flush=True,
    )
    if cell_spec is not None:
        held_cells = list(cell_spec.get("cells") or [])
        n_fast = sum(1 for c in held_cells if c.get("class") == "fast")
        print(
            f"[GVD] holdout cells: fast={n_fast} slow={len(held_cells) - n_fast} of {len(held_cells)} cells",
            flush=True,
        )
    if free_vram_bytes is None:
        free_vram_bytes = read_free_vram_bytes()
    if free_ram_bytes is None:
        free_ram_bytes = read_free_ram_bytes()
    own_view = view is None
    if view is None:
        view = TrainWindow()
    if pause_dir is not None:
        print(f"[GVD] pause file: {checkpoint_path(root, pause_dir)}", flush=True)
    view.load_folder(root, pause_dir=pause_dir)
    # The CLI epoch count is not the stop. The holdout decides.
    requested_epochs = int(epochs)

    def _empty(precision: str = "fp32") -> dict[str, Any]:
        if export_path is not None:
            export_scene_onnx(export_path, cams=train_cams, heads=train_heads, cam_age=use_cam_age)
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
            if backbone in ("", "scratch"):
                holder["net"] = SceneNet(cams=train_cams, scale=input_scale, cam_age=use_cam_age).to(device)
                if grad_ckpt:
                    holder["net"].grad_ckpt = True
                    print(
                        "[GVD] grad-ckpt: per-tick encoder (_token) recomputed in backward "
                        "(torch.utils.checkpoint, use_reentrant=False); eval and export unchanged",
                        flush=True,
                    )
            else:
                from python.train.pretrained_net import build_scene_net

                holder["net"] = build_scene_net(
                    train_cams, input_scale, backbone, freeze=bool(freeze_backbone)
                ).to(device)
                print(
                    f"[GVD] backbone={backbone} single-frame (no GRU, no skip) "
                    f"freeze={bool(freeze_backbone)} backbone_lr_mult={float(backbone_lr_mult)} "
                    f"token_dim={holder['net'].token_dim} "
                    f"params={sum(p.numel() for p in holder['net'].parameters())} "
                    "preprocess=BGR->RGB, /255, ImageNet mean/std inside the net",
                    flush=True,
                )
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
        if hasattr(holder["net"], "param_groups") and float(backbone_lr_mult) != 1.0:
            params: Any = holder["net"].param_groups(rate, float(backbone_lr_mult))
        else:
            params = holder["net"].parameters()
        holder["opt"] = torch.optim.AdamW(params, lr=rate, weight_decay=WEIGHT_DECAY)
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
    # Each shuffled window starts from a reset GRU, so several windows fit
    # in one step. parallel_fit is how many this free VRAM can hold.
    parallel_fit = 0
    if int(free_vram_bytes) > 0 and len(rows) > 0:
        parallel_fit = largest_batch(
            usable_bytes(int(free_vram_bytes)),
            file_bytes=file_bytes,
            n_strips=len(rows),
            window_strips=chunk_len,
            limit=32,
        )
    batch = resolve_scene_batch(int(batch_size), parallel_fit)
    est_step = step_bytes(
        batch,
        file_bytes=file_bytes,
        n_strips=max(1, len(rows)),
        window_strips=chunk_len,
    )
    recommended = bool(plan.recommended) and device.type == "cuda"
    note = "recommended" if recommended else "not recommended"
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
    from python.train.epoch_ckpt import DEFAULT_MIN_FREE_GB, GB, default_ckpt_dir, run_id

    epoch_run = run_id(run_started)
    epoch_ckpt_dir = Path(ckpt_dir) if ckpt_dir else default_ckpt_dir(root)
    epoch_ckpt_floor = int(float(DEFAULT_MIN_FREE_GB if ckpt_min_free_gb is None else ckpt_min_free_gb) * GB)
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
                pause_dir=pause_dir,
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

    def _write_epoch_ckpt(scored: Mapping[str, Any]) -> dict[str, Any] | None:
        """Save the weights behind this val row (create-only). A refusal is logged; training goes on."""
        from python.train.epoch_ckpt import EpochCkptRefused, epoch_ckpt_path, write_epoch_ckpt

        path = epoch_ckpt_path(epoch_ckpt_dir, root, epoch_run, int(epoch), int(done))
        meta = {
            "run": epoch_run,
            "epoch": int(epoch),
            "step": int(done),
            "strips": root.name,
            "val": {key: json_num(scored.get(key)) for key in ("n", "r", "rms", "rms_mean", "bias", "pred_mean", "pred_std")},
            "cams": list(train_cams or ()),
            "cam_age": bool(use_cam_age),
            "heads": list(train_heads or ()),
            "scale": float(input_scale),
            "backbone": str(backbone),
            "freeze_backbone": bool(freeze_backbone),
            "arch": getattr(net, "pause_arch", None) or checkpoint_arch(train_cams),
            "eval_chunk": int(eval_chunk),
            "gt_file": os.environ.get("GVD_GT_FILE") or "gt.jsonl",
            "mask_files": str(mask_files or ""),
            "mask_events": bool(mask_events),
            "holdout_blocks": int(holdout_blocks),
            "holdout_cells": bool(holdout_cells),
            "lr": float(holder.get("lr", lr)),
            "batch": int(batch),
        }
        weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
        try:
            info = write_epoch_ckpt(path, weights, meta, min_free_bytes=epoch_ckpt_floor, strip_root=root)
        except EpochCkptRefused as exc:
            print(f"[GVD] epoch checkpoint REFUSED epoch={epoch} step={done}: {exc} -- training continues", flush=True)
            return None
        print(
            f"[GVD] epoch checkpoint epoch={epoch} step={done} path={info['path']} "
            f"sha256={info['sha256']} bytes={info['bytes']} free_after_gb={info['free_after'] / GB:.2f}",
            flush=True,
        )
        return info

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
            pause_dir=pause_dir,
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
        lane_gt_seen = LaneGtCoverage("lane_gt" in head_names(train_heads))
        filter_logged = False
        while not paused_exit:
            refit = False
            lane_gt_seen.reset()
            from python.train.continual import epoch_windows, report_valid_filter

            filter_stats: dict[str, Any] = {}
            plan_steps = epoch_windows(
                pass_plan(rows, chunk_len, stride, blocks=int(holdout_blocks), cells=cell_spec),
                window_rng,
                min_valid=float(min_valid),
                balance=bool(balance),
                stats=filter_stats,
                recording=root.name or str(root),
            )
            # Log what --min-valid removed (first epoch, or whenever a whole
            # recording goes); a recording with no train window left raises.
            report_valid_filter(filter_stats, epoch=int(epoch), quiet=filter_logged)
            filter_logged = True
            windows = sum(1 for item in plan_steps if item["kind"] == "train")
            carry: SceneState | None = None
            end_state: SceneState | None = None
            chunk_in: SceneState | None = None
            hold_sum = 0.0
            hold_n = 0
            val_pairs: list[tuple[float, float]] = []
            val_pairs_long: list[tuple[float, float]] = []
            val_rows: list[Mapping[str, Any]] = []
            val_parts: dict[str, float] = {"lane_reg": 0.0, "conf_bce": 0.0, "n": 0}
            index = 0
            prefetch = _prefetch_loader(
                plan_steps,
                root,
                train_cams,
                cache,
                input_scale,
                int(workers),
                device.type == "cuda",
            )
            if prefetch is not None:
                train_n = sum(1 for item in plan_steps if item.get("kind") != "holdout")
                if len(prefetch.dataset) != train_n:
                    raise RuntimeError(
                        f"cache loader has {len(prefetch.dataset)} windows, the plan has {train_n}"
                    )
            prefetch_iter = iter(prefetch) if prefetch is not None else None
            while index < len(plan_steps) and not paused_exit:
                if view.consume_pause() and _pause_now():
                    break
                item = plan_steps[index]
                prefetched_frames = None
                prefetched_ages = None
                if prefetch_iter is not None and item.get("kind") != "holdout":
                    try:
                        packed = next(prefetch_iter)
                        prefetched_frames = packed.get("frames")
                        prefetched_ages = packed.get("ages")
                    except StopIteration:
                        prefetched_frames = None
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
                        cams=train_cams,
                        cache=cache,
                        heads=train_heads,
                        scale=input_scale,
                        reset_every=int(eval_chunk),
                        pairs_long=val_pairs_long,
                        pair_rows=val_rows,
                        parts=val_parts,
                        conf_weight=float(conf_weight),
                    )
                    if held_loss is not None and held_count:
                        hold_sum += float(held_loss) * int(held_count)
                        hold_n += int(held_count)
                    index += 1
                    continue
                group = [item]
                group_frames: list[Any] = [prefetched_frames]
                group_ages: list[Any] = [prefetched_ages]
                if int(batch) > 1 and str(item.get("incoming")) == "reset":
                    while len(group) < int(batch) and index + len(group) < len(plan_steps):
                        nxt = plan_steps[index + len(group)]
                        if nxt.get("kind") == "holdout" or str(nxt.get("incoming")) != "reset":
                            break
                        nxt_frames = None
                        nxt_ages = None
                        if prefetch_iter is not None:
                            try:
                                packed = next(prefetch_iter)
                                nxt_frames = packed.get("frames")
                                nxt_ages = packed.get("ages")
                            except StopIteration:
                                nxt_frames = None
                        group.append(nxt)
                        group_frames.append(nxt_frames)
                        group_ages.append(nxt_ages)
                if item["incoming"] != "chunk":
                    chunk_in = state_in
                last_batch = {
                    "step": int(done),
                    "epoch": int(epoch),
                    "rows": [row.get("i") for piece in group for row in piece["rows"]],
                }
                try:
                    if len(group) == 1:
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
                            clip=grad_clip,
                            trace=step_trace,
                            cams=train_cams,
                            cache=cache,
                            augment=use_augment,
                            rng=augment_rng,
                            heads=train_heads,
                            lane_loss=lane_kind,
                            conf_weight=float(conf_weight),
                            scale=input_scale,
                            frames=prefetched_frames,
                            frame_ages=prefetched_ages,
                        )
                    else:
                        step_loss, n_sup, carry, end_state = _optimizer_group(
                            net,
                            holder["opt"],
                            holder["scaler"],
                            root,
                            [list(piece["rows"]) for piece in group],
                            device=device,
                            amp=bool(holder["amp"]),
                            view=view,
                            clip=grad_clip,
                            trace=step_trace,
                            cams=train_cams,
                            cache=cache,
                            augment=use_augment,
                            rng=augment_rng,
                            heads=train_heads,
                            lane_loss=lane_kind,
                            conf_weight=float(conf_weight),
                            frame_sets=group_frames,
                            scale=input_scale,
                            age_sets=group_ages,
                        )
                except RuntimeError as exc:
                    if not _is_oom(exc):
                        raise
                    net.zero_grad(set_to_none=True)
                    _release_cuda()
                    if device.type == "cuda" and int(batch) > 1:
                        batch = max(1, int(batch) // 2)
                        refit = True
                        print(f"[GVD] scene fit oom, retry batch={batch}", flush=True)
                        break
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
                    prefetch_iter = None
                    carry = None
                    end_state = None
                    chunk_in = None
                    continue
                if step_loss is not None:
                    last_loss = step_loss
                if n_sup:
                    lane_gt_seen.note(int(step_trace.get("lane_gt_rows") or 0) > 0)
                supervised += n_sup
                cursor += len(group)
                done += len(group)
                index += len(group)
                elapsed = time.perf_counter() - started
                remaining_steps = sum(1 for later in plan_steps[index:] if later["kind"] == "train")
                steps_per_epoch = sum(1 for item in plan_steps if item["kind"] == "train")
                has_holdout = any(item["kind"] == "holdout" for item in plan_steps)
                if not schedule_printed:
                    schedule_printed = True
                    print(
                        f"[GVD] scene schedule steps_per_epoch={steps_per_epoch} "
                        f"epoch={epoch} lr={float(holder['lr'])} fixed "
                        f"--epochs {requested_epochs} is recorded; holdout stop ends the run",
                        flush=True,
                    )
                job_steps = remaining_until_holdout_can_stop(
                    remaining_steps,
                    steps_per_epoch,
                    epoch=epoch,
                    has_holdout=has_holdout,
                    rates_after=0,
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
            if refit:
                continue
            if paused_exit:
                break
            # Log the epoch's share of batches without a lane_gt target; all of
            # them missing with the lane_gt head on raises LaneGtMissingError.
            # Windows planned but zero lane_gt steps taken raises as well.
            lane_gt_seen.close_epoch(epoch, planned=windows)
            if hold_n <= 0:
                break
            mean = hold_sum / hold_n
            scored: dict[str, Any] = {}
            if val_pairs:
                scored = val_summary(val_pairs, val_pairs_long, val_parts)
                record = {
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
                    "eval_chunk": int(eval_chunk),
                    "r_long": json_num(scored.get("r_long")),
                    "rms_long": json_num(scored.get("rms_long")),
                    "bias_long": json_num(scored.get("bias_long")),
                    "lane_loss": json_num(scored.get("lane_loss")),
                    "conf_bce": json_num(scored.get("conf_bce")),
                    "conf_weight": float(conf_weight),
                    "pred_mean": json_num(scored.get("pred_mean")),
                    "pred_std": json_num(scored.get("pred_std")),
                    "collapse": bool(scored.get("collapse")),
                    "blank_main_skipped": int(BLANK_MAIN_SKIPS),
                }
                cell_lines: list[str] = []
                if cell_spec is not None:
                    from python.train.continual import bootstrap_cells, cell_scores

                    per_cell = cell_scores(val_pairs, val_rows, cell_spec)
                    boot = bootstrap_cells(per_cell, n_boot=1000, seed=0)
                    record["cells"] = {
                        key: {"n": c["n"], "r": json_num(c["r"]), "rms": json_num(c["rms"])}
                        for key, c in per_cell.items()
                    }
                    record["r_ci90"] = boot["r_ci"]
                    record["rms_ci90"] = boot["rms_ci"]
                    record["n_cells"] = boot["n_cells"]
                    for key, c in per_cell.items():
                        cell_lines.append(f"[GVD] val cell {key} n={c['n']} r={c['r']} rms={c['rms']}")
                    cell_lines.append(
                        f"[GVD] val cells n_cells={boot['n_cells']} r_ci90={boot['r_ci']} "
                        f"rms_ci90={boot['rms_ci']} (bootstrap over cells, 1000 resamples, seed 0)"
                    )
                # The weights have not moved since the holdout pass: these are the val row's weights.
                ckpt_info = _write_epoch_ckpt(scored)
                record["ckpt"] = None if ckpt_info is None else Path(ckpt_info["path"]).name
                record["ckpt_sha256"] = None if ckpt_info is None else ckpt_info["sha256"]
                append_metrics(debug_dir, record)
                print(
                    f"[GVD] val step={done} n={scored['n']} r={scored['r']} "
                    f"rms={scored['rms']} rms_mean={scored['rms_mean']} "
                    f"bias={scored['bias']} loss={mean} "
                    f"lane_loss={scored.get('lane_loss')} conf_bce={scored.get('conf_bce')} "
                    f"pred_mean={scored.get('pred_mean')} pred_std={scored.get('pred_std')} "
                    f"r_long={scored.get('r_long')} rms_long={scored.get('rms_long')} "
                    f"bias_long={scored.get('bias_long')} eval_chunk={int(eval_chunk)} "
                    f"blank_main_skipped={int(BLANK_MAIN_SKIPS)}",
                    flush=True,
                )
                for line in cell_lines:
                    print(line, flush=True)
                if scored.get("collapse"):
                    print(
                        f"[GVD] val COLLAPSE? pred_std={scored.get('pred_std')} m < {COLLAPSE_STD_M} m: "
                        "the 8 m center is a near-constant prediction; r is not meaningful",
                        flush=True,
                    )
            # Stop on the 8 m center rms, not the composite loss. A tiny drop in
            # the other heads was keeping the run alive after the lane had plateaued.
            # Without a scored center, the lane regression alone decides; the
            # confidence BCE never does.
            fallback = scored.get("lane_loss") if scored.get("lane_loss") is not None else None
            if fallback is None and int(val_parts.get("n", 0) or 0) > 0:
                fallback = float(val_parts["lane_reg"]) / int(val_parts["n"])
            if fallback is None:
                fallback = mean
            if scored.get("rms") is not None:
                history.append(float(scored["rms"]))
                improved = best_value is None or float(scored["rms"]) < best_value
            else:
                history.append(float(fallback))
                improved = best_value is None or float(fallback) < best_value
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
        """Repeat one fixed batch for the whole budget.

        Dry-run and sanity used to walk the shuffled plan once. Eight frames
        are a single window, so sanity exited at step 1. Augment stays off.
        """
        nonlocal last_loss, done, sanity_ok, last_batch, supervised, windows
        step_trace["print_shapes"] = shapes
        if not subset:
            sanity_ok = False
            print("[GVD] scene train: no strips for this mode", flush=True)
            return
        batch_rows = list(subset)
        steps = max(1, int(limit))
        print(
            f"[GVD] {'dry' if shapes else 'sanity'} batch rows={len(batch_rows)} "
            f"steps={steps} augment=off lr={float(holder.get('lr', lr))}",
            flush=True,
        )
        lane_loss = None
        for step_i in range(steps):
            state = net.initial_state(device=device, batch=1)
            last_batch = {
                "step": step_i,
                "epoch": 0,
                "rows": [row.get("i") for row in batch_rows],
            }
            loss, n_sup, _carry, _state = _optimizer_step(
                net,
                holder["opt"],
                holder["scaler"],
                root,
                batch_rows,
                state,
                device=device,
                amp=bool(holder["amp"]),
                view=view,
                stride=max(1, len(batch_rows)),
                clip=grad_clip,
                trace=step_trace,
                cams=train_cams,
                cache=cache,
                augment=False,
                heads=train_heads,
                lane_loss=lane_kind,
                conf_weight=float(conf_weight),
                scale=input_scale,
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
                print(f"[GVD] sanity lane_gt={lane_loss} step={done} ok=True", flush=True)
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
            existing = load_checkpoint(root, pause_dir=pause_dir) if resume else None
            if existing is not None:
                fields = resume_fields(existing)
                rate = resume_learning_rate(lr, fields.get("lr"))
                _arm(device.type, rate)
                device = holder["device"]
                net = holder["net"]
                _apply_checkpoint(net, holder["opt"], holder["scaler"], fields, device)
                _pin_optimizer_lr(holder["opt"], rate)
                if init_head_bias:
                    print("[GVD] --init-head-bias ignored: resuming from a checkpoint", flush=True)
                paused_lr = fields.get("lr")
                if paused_lr is None or not math.isclose(float(paused_lr), rate, rel_tol=0.0, abs_tol=1e-12):
                    print(
                        f"[GVD] scene lr={rate} fixed (pause had {paused_lr}; that rate does not continue)",
                        flush=True,
                    )
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
                # One rate for the whole run. Holdout stop returns. It does not
                # reload the initial weights and try 1e-4.
                view.mark_running()
                _arm(device.type, resume_learning_rate(lr))
                started = time.perf_counter()
                device = holder["device"]
                net = holder["net"]
                epoch = 0
                cursor = 0
                done = 0
                last_loss = None
                view.losses = []
                print(f"[GVD] scene lr={float(holder['lr'])} fixed", flush=True)
                if init_head_bias:
                    seen: dict[Any, Mapping[str, Any]] = {}
                    for item in startup_plan:
                        if item.get("kind") != "train":
                            continue
                        for row in item.get("rows") or []:
                            if loss_mask(row):
                                seen.setdefault(row.get("i"), row)
                    constants = huber_slot_constants(list(seen.values()))
                    apply_head_bias(net, constants)
                    print(
                        f"[GVD] init lane_gt_head.bias from {len(seen)} train rows (Huber, delta 1): "
                        + " ".join(f"{name}={float(v):+.3f}" for name, v in zip(LANE_SLOT_NAMES, constants)),
                        flush=True,
                    )
                _train_passes()
        if not paused_exit and not dry_run and not sanity and export_path is not None:
            export_scene_onnx(export_path, net, heads=train_heads)
        if not paused_exit and not dry_run and not sanity:
            clear_checkpoint(root, pause_dir=pause_dir)
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
    from python.train.cam_strip import parse_cams, parse_heads
    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.scene_loader import default_num_workers

    try:
        cam_names = parse_cams(args.cams)
        head_names = parse_heads(args.heads)
    except ValueError as exc:
        print(f"[GVD] scene train: {exc}")
        sys.exit(2)
    if args.build_cache:
        manifest = build_frame_cache(
            args.strips or scene_lanes_dir(),
            args.build_cache,
            cam_names,
            scale=float(args.scale),
        )
        print(
            f"[GVD] cache {args.build_cache} cams={','.join(cam_names)} "
            f"chunks={len(manifest.get('chunks') or [])} frames={manifest.get('frames')}",
            flush=True,
        )
        return
    frame_cache = FrameCache(args.cache) if args.cache else None
    try:
        if frame_cache is not None and frame_cache.stale:
            print(f"[GVD] scene train: cache refused: {frame_cache.reason}", flush=True)
            sys.exit(2)
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
            path = export_scene_onnx(export_path, cams=cam_names, heads=head_names, cam_age=bool(args.cam_age))
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
        pause_dir: Path | None = None
        if args.pause_dir:
            try:
                check_pause_dir(args.strips, args.pause_dir, no_resume=bool(args.no_resume))
            except PauseDirRefused as exc:
                print(f"[GVD] scene train: pause-dir refused: {exc}", flush=True)
                sys.exit(2)
            pause_dir = Path(args.pause_dir).resolve()
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
                cams=cam_names,
                heads=head_names,
                cache=frame_cache,
                augment=bool(args.augment),
                clip=float(args.clip),
                lane_loss=str(args.lane_loss),
                balance=bool(args.balance),
                min_valid=float(args.min_valid),
                workers=default_num_workers() if args.workers is None else int(args.workers),
                scale=float(args.scale),
                batch_size=int(args.batch),
                holdout_blocks=resolve_holdout_blocks(args.holdout_blocks),
                holdout_cells=bool(args.holdout_cells),
                holdout_cells_file=Path(args.holdout_cells_file) if args.holdout_cells_file else None,
                holdout_seed=int(args.holdout_seed),
                conf_weight=float(args.conf_weight),
                init_head_bias=bool(args.init_head_bias),
                eval_chunk=int(args.eval_chunk),
                backbone=str(args.backbone),
                freeze_backbone=bool(args.freeze_backbone),
                backbone_lr_mult=float(args.backbone_lr_mult),
                mask_events=bool(args.mask_events),
                mask_files=str(args.mask_files or ""),
                ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
                ckpt_min_free_gb=args.ckpt_min_free_gb,
                cam_age=bool(args.cam_age),
                grad_ckpt=bool(args.grad_ckpt),
                pause_dir=pause_dir,
            )
        except TrainFitError as exc:
            print(f"[GVD] scene train: {exc}")
            sys.exit(2)
        except MaskFileError as exc:
            print(f"[GVD] scene train: {exc}", flush=True)
            sys.exit(2)
        except LossyFramesError as exc:
            print(f"[GVD] scene train: {exc}", flush=True)
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
    finally:
        if frame_cache is not None:
            frame_cache.close()


if __name__ == "__main__":
    main()
