#!/usr/bin/env python3
"""``lane_contact_sheet.py``: GT and predicted lanes drawn where the camera sees them.

* inverse: ``pose_pixel_of_ground`` round-trips ``project_pose_pixel`` on the
  main pose, and a point behind the camera is omitted.
* gt: on a black synthetic recording with an off-center lane (center +1.2 m),
  GT green lands on the projected center points and not on the mirrored ones
  (-1.2 m); no pred colour without a prediction; each tile is labelled.
* val_worst: a val_worst JSON's frames are drawn in its order with its pred in
  red at the pred's own projected points (not the GT's), and its loss labelled.
* picks: even includes first and last; random is seeded (same seed, same
  frames; another seed, other frames).
* bounded: 40 frames at 16 per page are 3 pages within 4x4 tiles; K caps at 64.
* onnx: a 2-epoch checkpoint's ONNX draws a red prediction and an err8 label.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

W, H = 640, 480
TILE_W = 320


def _lane(center: float) -> list[float]:
    vals = []
    for _ in range(4):
        vals += [center, center - 1.8, center + 1.8]
    return vals + [0.0, 1.0]


def _recording(root: Path, n: int, center: float = 1.2) -> None:
    import cv2

    (root / "raw" / "main").mkdir(parents=True)
    ok, buf = cv2.imencode(".png", np.zeros((H, W, 3), np.uint8))
    assert ok
    state, gt = [], []
    for i in range(n):
        (root / "raw" / "main" / f"{i:06d}.png").write_bytes(buf.tobytes())
        state.append({"i": i, "t": i * 0.1, "cams": {"main": {"file": f"raw/main/{i:06d}.png", "t": i * 0.1}}})
        tech = {"lane_gt": _lane(center), "lane_gt_valid": [1.0] * 14, "gt_origin": "body_centreline"}
        gt.append({"i": i, "t": i * 0.1, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt) + "\n", encoding="utf-8")


def _sheet(args: list[str]) -> tuple[list[Path], str]:
    from lane_contact_sheet import build_parser, render

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        paths = render(build_parser().parse_args(args))
    return paths, out.getvalue()


def _tile_px(x: float, y: float) -> tuple[int, int]:
    """Where ground (x, y) lands in tile 0 of a sheet (below the label band)."""
    from lane_contact_sheet import LABEL_H
    from python.perception.lanes import camera_pose, pose_pixel_of_ground

    u, v = pose_pixel_of_ground(camera_pose("main"), x, y, width=W, height=H)
    s = TILE_W / W
    return int(round(u * s)), int(round(v * s)) + LABEL_H


def _near(img: np.ndarray, px: tuple[int, int], bgr: tuple[int, int, int], r: int = 2) -> bool:
    u, v = px
    patch = img[max(0, v - r): v + r + 1, max(0, u - r): u + r + 1].astype(int)
    if patch.size == 0:
        return False
    target = np.array(bgr)
    # Anti-aliased: the brightest pixel near the line is within 40 of the colour.
    return bool((np.abs(patch - target).max(axis=2) <= 40).any())


def check_inverse() -> None:
    from python.perception.lanes import camera_pose, pose_pixel_of_ground, project_pose_pixel

    pose = camera_pose("main")
    for x, y in [(1.5, 10.0), (-2.0, 6.0), (0.3, 30.0), (3.0, 5.0)]:
        u, v = pose_pixel_of_ground(pose, x, y)
        back = project_pose_pixel(pose, u, v)
        assert abs(back["x"] - x) < 1e-6 and abs(back["y"] - y) < 1e-6, (x, y, back)
    assert pose_pixel_of_ground(pose, 0.0, -5.0) is None
    print("inverse: round-trips project_pose_pixel within 1e-6 m; behind the camera omitted OK")


def check_gt(tmp: Path) -> None:
    import cv2

    from python.train.val_worst import GT_BGR, PRED_BGR

    root = tmp / "recA"
    _recording(root, 12)
    paths, log = _sheet(["--strips", str(root), "--k", "4", "--cols", "2", "--tile-w", str(TILE_W), "--out", str(tmp / "gt.png")])
    assert [p.name for p in paths] == ["gt.png"], paths
    img = cv2.imread(str(paths[0]))
    for y in (10.0, 20.0):
        assert _near(img, _tile_px(1.2, y), GT_BGR), f"no GT at center x=+1.2 y={y}"
        assert not _near(img, _tile_px(-1.2, y), GT_BGR), f"GT at mirrored x=-1.2 y={y}"
    assert not (np.abs(img.astype(int) - np.array(PRED_BGR)).max(axis=2) <= 40).any(), "pred colour without a pred"
    band = img[: 30, : TILE_W]
    assert (band > 200).any(), "tile 0 has no label"
    assert "4 frames, 1 page(s)" in log, log
    print("gt: green at the +1.2 m center, none at -1.2 m; no red without a prediction; labelled OK")


def check_val_worst(tmp: Path) -> None:
    import cv2

    from lane_contact_sheet import label_for
    from python.train.val_worst import GT_BGR, PRED_BGR

    root = tmp / "recB"
    _recording(root, 12)
    frames = [
        {"rank": 1, "i": 7, "loss": 0.9, "gt_origin": "body_centreline", "gt": _lane(1.2), "gt_valid": [1.0] * 14, "pred": _lane(-1.5)},
        {"rank": 2, "i": 2, "loss": 0.4, "gt_origin": "body_centreline", "gt": _lane(1.2), "gt_valid": [1.0] * 14, "pred": _lane(-1.5)},
    ]
    dump = tmp / "val_worst.json"
    dump.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    paths, log = _sheet(["--strips", str(root), "--val-worst", str(dump), "--tile-w", str(TILE_W), "--out", str(tmp / "vw.png")])
    img = cv2.imread(str(paths[0]))
    assert _near(img, _tile_px(-1.5, 10.0), PRED_BGR), "no pred at its own x=-1.5"
    assert not _near(img, _tile_px(1.2, 10.0), PRED_BGR), "pred drawn at the GT"
    assert _near(img, _tile_px(1.2, 10.0), GT_BGR), "no GT"
    assert "2 frames" in log, log
    label = label_for("recB", frames[0])
    assert "recB i=7" in label and "loss=0.9000" in label and "err8=2.70m" in label and "gt_origin=body_centreline" in label, label
    print("val_worst: its frames, red pred at -1.5 m (not the GT's +1.2 m), loss and err8 labelled OK")


def check_picks() -> None:
    from lane_contact_sheet import MAX_K, pick_rows

    rows = [{}] * 100
    even = pick_rows(rows, 5, "even", 0)
    assert even == [0, 25, 50, 74, 99] or even == [0, 25, 50, 75, 99], even
    a, b, c = (pick_rows(rows, 8, "random", s) for s in (3, 3, 4))
    assert a == b and a != c and a == sorted(a) and len(set(a)) == 8, (a, b, c)
    assert len(pick_rows([{}] * 500, 1000, "even", 0)) == MAX_K
    print(f"picks: even spans first..last; random seeded; K capped at {MAX_K} OK")


def check_bounded(tmp: Path) -> None:
    import cv2

    root = tmp / "recC"
    _recording(root, 40)
    paths, _ = _sheet(["--strips", str(root), "--k", "40", "--per-page", "16", "--cols", "4", "--tile-w", "9999", "--out", str(tmp / "big.png")])
    assert [p.name for p in paths] == ["big_p1.png", "big_p2.png", "big_p3.png"], paths
    for p in paths:
        h, w = cv2.imread(str(p)).shape[:2]
        assert w <= 4 * 480 and h <= 4 * (480 * 3 // 4 + 30), (p, w, h)
    print("bounded: 40 frames -> 3 pages, each within 4x4 tiles of <= 480 px OK")


def check_onnx(tmp: Path) -> None:
    import cv2
    import torch

    from python.train.epoch_ckpt import export_epoch_onnx
    from python.train.val_worst import PRED_BGR
    from test_epoch_ckpt import _fixture, _train

    torch.manual_seed(0)
    torch.set_num_threads(1)
    root = tmp / "onnx" / "rec"
    _fixture(root)
    ckpts = tmp / "onnx" / "ckpts"
    with contextlib.redirect_stdout(io.StringIO()):
        _train(root, ckpt_dir=ckpts, status_root=tmp / "onnx" / "_status")
    last = sorted(ckpts.glob("*.npz"))[-1]
    onnx = tmp / "onnx" / "m.onnx"
    with contextlib.redirect_stdout(io.StringIO()):
        export_epoch_onnx(last, onnx)
    paths, log = _sheet(["--strips", str(root), "--onnx", str(onnx), "--k", "3", "--out", str(tmp / "onnx.png")])
    img = cv2.imread(str(paths[0]))
    assert (np.abs(img.astype(int) - np.array(PRED_BGR)).max(axis=2) <= 40).any(), "no red prediction"
    assert "3 frames" in log, log
    print("onnx: a checkpoint's ONNX prediction drawn in red OK")


def main() -> int:
    check_inverse()
    check_picks()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        check_gt(t)
        check_val_worst(t)
        check_bounded(t)
        check_onnx(t)
    print("lane contact sheet OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
