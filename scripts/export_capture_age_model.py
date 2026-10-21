#!/usr/bin/env python3
"""Export an UNTRAINED gvd_cam_age=1 eight-camera scratch scene model for the live capture.

The capture (scripts/live_age_capture.py) needs a model that makes run_vision
schedule the companion polls and feed the ages; its weights do not matter,
and engage is hard-disabled for the capture. This writes:

* SceneNet(cams = all eight, cam_age=True), seeded random init, lane_gt head;
* metadata gvd_cam_age=1 (+ order / cap / tick) and gvd_capture_untrained=1.

``--out`` is the path you pass. It is create-only and is refused inside this
repository, so the export never lands in the repo tree.

    py -3 scripts/export_capture_age_model.py --out <OUTSIDE_REPO>/capture_age_untrained.onnx
"""
# gvd-ci: requires torch
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def export(out: Path | str, *, seed: int = 0) -> Path:
    out = Path(out).expanduser().resolve()
    try:
        out.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(f"refused: {out} is inside the repository; pass a path outside it")
    if out.exists():
        raise SystemExit(f"refused: {out} already exists (create-only)")
    if out.suffix.lower() != ".onnx":
        raise SystemExit("refused: --out must end in .onnx")
    import torch

    from python.data.strip_writer import STRIP_ORDER
    from python.train.cam_strip import stamp_onnx_metadata
    from python.train.scene_net import SceneNet, export_scene_onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    net = SceneNet(cams=tuple(STRIP_ORDER), cam_age=True).eval()
    partial = out.with_name(f"{out.name}.partial-{os.getpid()}.onnx")
    try:
        export_scene_onnx(partial, net, cams=tuple(STRIP_ORDER), heads=("lane_gt",), cam_age=True)
        stamp_onnx_metadata(str(partial), {"gvd_capture_untrained": "1", "gvd_capture_seed": str(int(seed))})
        os.link(partial, out)
    finally:
        if partial.exists():
            partial.unlink()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Untrained gvd_cam_age=1 8-camera model for the live capture")
    parser.add_argument("--out", type=Path, required=True, help="Output .onnx path, outside the repository (create-only).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = export(args.out, seed=args.seed)
    print(f"[GVD] capture model (untrained, gvd_cam_age=1, 8 cams) -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
