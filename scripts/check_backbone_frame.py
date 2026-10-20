r"""One recorded main frame through the --backbone net: channel order, ImageNet
normalization stats, and top-5 ImageNet classes.

    py -3 scripts/check_backbone_frame.py --strips E:\GVD_data\scene_lanes [--i 3100]

CPU only. Reads frames; writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", default=r"E:\GVD_data\scene_lanes")
    ap.add_argument("--i", type=int, default=None, help="Row index (default: first row with a decodable main frame)")
    ap.add_argument("--backbone", default="imagenet-resnet18", choices=("imagenet-resnet18", "mobilenetv3"))
    ap.add_argument("--n", type=int, default=64, help="Also average the normalized stats over this many main frames")
    args = ap.parse_args()

    from python.data.raw_frames import load_scene_rows, row_has_decodable_frame
    from python.train.pretrained_net import BGR_TO_RGB, build_scene_net, imagenet_categories
    from python.train.scene_net import _strip_nchw
    from python.train.train_scene import _frame_row

    root = Path(args.strips)
    rows = load_scene_rows(root)
    if args.i is None:
        row = next(r for r in rows if row_has_decodable_frame(root, r, ("main",)))
    else:
        row = next(r for r in rows if int(r["i"]) == int(args.i))
    loaded = _frame_row(root, row, cams=("main",), cache=None, augment=False)
    if loaded is None:
        print(f"row i={row['i']}: main frame did not load")
        return 1
    frame, _ = loaded
    net = build_scene_net(("main",), 1.0, args.backbone).eval()
    image = _strip_nchw(frame, width=net.canvas_w, height=net.canvas_h)  # what the trainer feeds: 1x3xHxW, 0-1

    # Channel order, measured: the top quarter is sky. BGR puts blue in channel 0.
    h = int(image.shape[2])
    sky = image[0, :, : h // 4, :].reshape(3, -1).mean(dim=1).tolist()
    road = image[0, :, 3 * h // 4 :, :].reshape(3, -1).mean(dim=1).tolist()
    order = "BGR" if sky[0] > sky[2] else "RGB"
    print(f"row i={row['i']} frame {tuple(frame.shape)} {frame.dtype} -> net input {tuple(image.shape)} range [{float(image.min()):.3f}, {float(image.max()):.3f}]")
    print(f"sky (top 1/4) mean per input channel [c0, c1, c2] = [{sky[0]:.3f}, {sky[1]:.3f}, {sky[2]:.3f}]; road (bottom 1/4) = [{road[0]:.3f}, {road[1]:.3f}, {road[2]:.3f}]")
    print(f"input channel order: {order} (c0 {'>' if sky[0] > sky[2] else '<='} c2 on the sky); net swaps channels {BGR_TO_RGB} -> RGB")

    with torch.no_grad():
        x = net.preprocess(image)
        mean = x[0].reshape(3, -1).mean(dim=1).tolist()
        std = x[0].reshape(3, -1).std(dim=1).tolist()
        print(f"after BGR->RGB + ImageNet norm, per RGB channel: mean=[{mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:+.3f}] std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}] (expect about 0 and 1)")
        cats = imagenet_categories(args.backbone)
        for label, img in (("as the net sees it (RGB)", image), ("control: channels NOT swapped", image[:, list(BGR_TO_RGB)])):
            prob = torch.softmax(net.classify(img), dim=-1)[0]
            top = torch.topk(prob, 5)
            names = ", ".join(f"{cats[int(k)]} {float(v):.3f}" for v, k in zip(top.values, top.indices))
            print(f"top-5 {label}: {names}")
        if args.n > 1:
            good = [r for r in rows if row_has_decodable_frame(root, r, ("main",))]
            pick = good[:: max(1, len(good) // args.n)][: args.n]
            acc = []
            for r in pick:
                got = _frame_row(root, r, cams=("main",), cache=None, augment=False)
                if got is not None:
                    acc.append(net.preprocess(_strip_nchw(got[0], width=net.canvas_w, height=net.canvas_h))[0].reshape(3, -1))
            allx = torch.cat(acc, dim=1)
            m = allx.mean(dim=1).tolist()
            s = allx.std(dim=1).tolist()
            print(f"over {len(acc)} main frames after norm: mean=[{m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}] std=[{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}]")
        pred, _ = net.forward_batch(image, torch.tensor([0.1]), net.initial_state(batch=1))
        print(f"lane_gt out (untrained head) shape {tuple(pred.lane_gt.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
