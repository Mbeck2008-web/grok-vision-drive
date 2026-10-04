"""Train the scene net from a strip directory.

Reads ``meta.json``, ``strips/*.jpg``, and ``state.jsonl``. Each step uses the
recorded seconds since the previous strip. Windows cover 15 seconds of real
time. This process does not start or close BeamNG.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.strip_writer import CANVAS_H, CANVAS_W, read_state_jsonl  # noqa: E402
from python.train.scene_net import (  # noqa: E402
    MEMORY_S,
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


def _resolve_device(capability: float | None) -> tuple[Any, Any, str]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if capability is not None:
        cap: Any = capability
    elif device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
    else:
        cap = None
    return device, cap, precision_for_capability(cap)


def _as_target(labels: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    out = {}
    long_keys = {"object_class", "sign_class", "sign_state"}
    for key, value in labels.items():
        dtype = torch.long if key in long_keys else torch.float32
        out[key] = torch.as_tensor(value, device=device, dtype=dtype)
    return out


def train_directory(
    root: Path,
    *,
    epochs: int,
    lr: float,
    capability: float | None,
    export_path: Path | None,
) -> dict[str, Any]:
    import torch

    from python.train.scene_net import SceneNet

    rows = read_state_jsonl(root / "state.jsonl")
    device, cap, precision = _resolve_device(capability)
    amp = precision == "amp" and device.type == "cuda"
    torch.manual_seed(0)
    net = SceneNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    scaler = _scaler(amp)
    windows = memory_windows(rows, MEMORY_S)
    supervised = 0
    net.train()
    for _epoch in range(max(1, int(epochs))):
        for window in windows:
            opt.zero_grad(set_to_none=True)
            state = net.initial_state(device=device)
            total = None
            n_loss = 0
            for row in window:
                bgr = _load_bgr(root, int(row["i"]))
                tensor = torch.from_numpy(np.ascontiguousarray(bgr)).to(device)
                dt = step_dt(row.get("dt_s"))
                with _autocast(amp):
                    pred, state = net(tensor, dt, state)
                if loss_mask(row):
                    step = scene_loss(pred, _as_target(labels_from_tech(row.get("tech")), device))
                    total = step["total"] if total is None else total + step["total"]
                    n_loss += 1
            if total is None or n_loss == 0:
                continue
            loss = total / n_loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            supervised += n_loss
    if export_path is not None:
        export_scene_onnx(export_path, net)
    return {
        "strips": len(rows),
        "windows": len(windows),
        "supervised": supervised,
        "precision": precision,
        "capability": cap,
        "export": None if export_path is None else str(export_path),
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
    stats = train_directory(
        Path(args.strips),
        epochs=args.epochs,
        lr=args.lr,
        capability=args.capability,
        export_path=export_path,
    )
    print(
        f"[GVD] scene train strips={stats['strips']} windows={stats['windows']} "
        f"supervised={stats['supervised']} precision={stats['precision']}"
    )
    if stats["export"]:
        print(f"[GVD] exported {stats['export']}")


if __name__ == "__main__":
    main()
