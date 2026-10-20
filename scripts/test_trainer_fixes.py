#!/usr/bin/env python3
"""Trainer fixes from the v5 collapse audit.

(a) rows without the trained main picture are dropped, and a black main
    area is skipped in ``_frame_row``;
(b) the holdout resets the GRU every 16 frames, with the old single
    sequence scored alongside;
(c) the val summary carries lane_loss, conf_bce, pred_mean, pred_std and
    the collapse flag; ``--conf-weight`` scales the confidence BCE;
(d) ``--init-head-bias`` constants are the per-slot Huber minimisers;
(e) a cell list for another map is refused, and the bootstrap over cells
    is seeded.

Synthetic checks always run, CPU only. ``--real DIR`` (or GVD_TEST_STRIPS)
also counts the rows fix (a) drops on a recorded index; on the 17,018-row
scene_lanes index that is about 1,816.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.debug.lane_targets import LANE_GT_DIM  # noqa: E402


def _png(path: Path, value: int = 128) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.full((48, 64, 3), value, dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def _tech(center: float = 0.3, width: float = 3.6, heading: float = 0.01) -> dict:
    values = []
    for k in range(4):
        c = center + 0.1 * k
        values += [c, c - width / 2, c + width / 2]
    values += [heading, 1.0]
    return {"lane_gt": values, "lane_gt_valid": [1.0] * LANE_GT_DIM}


class FakeCache:
    """Main-only canvases keyed by row index, like ``FrameCache``."""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self.frames = frames
        self.root = ""

    def has(self, index: int) -> bool:
        return int(index) in self.frames

    def canvas(self, index: int) -> np.ndarray:
        return self.frames[int(index)].copy()


def check_black_rows() -> None:
    from python.data.raw_frames import row_has_decodable_frame
    from python.train import train_scene
    from python.train.scene_loader import select_training_rows

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _png(root / "raw" / "main" / "000000.png")
        _png(root / "raw" / "pillarL" / "000000.png")
        _png(root / "raw" / "pillarL" / "000001.png")  # tick 1 has no front picture
        _png(root / "raw" / "rear" / "000002.png")
        rows = [
            {"i": 0, "t": 0.0, "cams": {"main": {}, "pillarL": {}}},
            {"i": 1, "t": 0.1, "cams": {"pillarL": {}}},
            {"i": 2, "t": 0.2, "cams": {"rear": {}}},
        ]
        assert row_has_decodable_frame(root, rows[1]), "old rule: any strip camera"
        assert not row_has_decodable_frame(root, rows[1], ("main",)), "main-only training needs main"
        assert not row_has_decodable_frame(root, rows[1], ("pillarL", "main")), "main is required when trained"
        assert row_has_decodable_frame(root, rows[1], ("pillarL",)), "a pillar-only net can use tick 1"
        old, old_missing, _ = select_training_rows(root, rows)
        dropped: dict = {}
        new, new_missing, _ = select_training_rows(root, rows, cams=("main",), dropped=dropped)
        assert [r["i"] for r in old] == [0, 1, 2] and old_missing == 0
        assert [r["i"] for r in new] == [0], new
        assert dropped == {"no_trained_cam": 2} and new_missing == 2, (dropped, new_missing)
    black = np.zeros((192, 256, 3), np.uint8)
    speck = black.copy()
    speck[100, 10] = 7  # one lit pixel: near-black, dropped too
    lit = np.random.RandomState(0).randint(20, 200, size=(192, 256, 3)).astype(np.uint8)
    assert train_scene.main_area_blank(black, ("main",))
    assert train_scene.main_area_blank(speck, ("main",))
    assert not train_scene.main_area_blank(lit, ("main",))
    assert not train_scene.main_area_blank(black, ("pillarL",)), "main not trained: not this check"
    from python.data.strip_writer import CANVAS_W, compose_strip

    full = compose_strip({"pillarL": np.full((48, 64, 3), 90, np.uint8)})
    assert full.shape[1] == CANVAS_W and train_scene.main_area_blank(full, None), "full strip, black main sector"
    cache = FakeCache({5: black, 6: lit})
    before = train_scene.BLANK_MAIN_SKIPS
    assert train_scene._frame_row(Path("."), {"i": 5}, cams=("main",), cache=cache) is None
    assert train_scene._frame_row(Path("."), {"i": 6}, cams=("main",), cache=cache) is not None
    assert train_scene.BLANK_MAIN_SKIPS == before + 1
    print("black rows: no-main rows dropped by select_training_rows(cams), black main area skipped by _frame_row OK")


def _synthetic_seq(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    frames = {}
    rows = []
    for i in range(n):
        frames[i] = rng.randint(0, 255, size=(192, 256, 3), dtype=np.uint8)
        rows.append(
            {
                "i": i,
                "t": 100.0 + 0.1 * i,
                "dt_s": 0.1,
                "wheel": {"steer": 0.0},
                "pedals": {"throttle": 0.1, "brake": 0.0},
                "tech": _tech(center=float(rng.uniform(-1, 1))),
            }
        )
    return rows, FakeCache(frames)


def check_chunked_eval() -> None:
    import torch

    from python.train.scene_net import SceneNet
    from python.train.train_scene import _eval_sequence, val_summary

    torch.manual_seed(0)
    net = SceneNet(cams=("main",))
    rows, cache = _synthetic_seq(40)
    dev = torch.device("cpu")
    common = dict(device=dev, amp=False, view=None, cams=("main",), cache=cache, heads=("lane_gt",))
    preds, pairs, pairs_long, prow = [], [], [], []
    parts: dict = {}
    loss, n = _eval_sequence(
        net, Path("."), rows, net.initial_state(batch=1), reset_every=16, pairs=pairs,
        pairs_long=pairs_long, pair_rows=prow, parts=parts, preds_out=preds, conf_weight=0.1, **common
    )
    assert n == 40 and loss is not None and len(pairs) == 40 and len(pairs_long) == 40 and len(prow) == 40
    chunked = []
    for j in range(0, 40, 16):
        out: list = []
        _eval_sequence(net, Path("."), rows[j : j + 16], net.initial_state(batch=1), reset_every=0, preds_out=out, **common)
        chunked += [vec for _row, vec in out]
    got = np.stack([vec for _row, vec in preds])
    want = np.stack(chunked)
    err = float(np.abs(got - want).max())
    assert err < 1e-4, f"chunked eval differs from per-chunk forward by {err}"
    whole: list = []
    long_pairs: list = []
    _eval_sequence(net, Path("."), rows, net.initial_state(batch=1), reset_every=0, preds_out=whole, pairs=long_pairs, **common)
    err_long = float(np.abs(np.array([p[0] for p in pairs_long]) - np.array([p[0] for p in long_pairs])).max())
    assert err_long < 1e-4, f"pairs_long differs from the single-sequence forward by {err_long}"
    # Frames 16.. differ between the two protocols (the GRU was reset), frames 0..15 match.
    head = float(np.abs(got[:16] - np.stack([v for _r, v in whole])[:16]).max())
    tail = float(np.abs(got[16:] - np.stack([v for _r, v in whole])[16:]).max())
    assert head < 1e-4 and tail > 1e-6, (head, tail)
    summary = val_summary(pairs, pairs_long, parts)
    print(
        f"chunked eval: matches per-16-frame forward (max err {err:.1e}), long stream matches the old protocol "
        f"(max err {err_long:.1e}) OK"
    )
    return summary


def check_val_fields(summary: dict) -> None:
    import torch

    from python.train import train_scene
    from python.train.scene_net import ScenePrediction, scene_loss
    from python.train.train_scene import val_summary

    for key in ("r", "rms", "bias", "r_long", "rms_long", "bias_long", "lane_loss", "conf_bce", "pred_mean", "pred_std", "collapse"):
        assert key in summary, key
    assert summary["lane_loss"] is not None and np.isfinite(summary["lane_loss"])
    assert summary["conf_bce"] is not None and np.isfinite(summary["conf_bce"])
    flat = val_summary([(-0.67 + 1e-3 * k, float(k % 5) - 2) for k in range(50)], None, {"lane_reg": 5.0, "conf_bce": 1.0, "n": 10})
    assert flat["collapse"] and flat["pred_std"] < train_scene.COLLAPSE_STD_M and abs(flat["lane_loss"] - 0.5) < 1e-9
    wide = val_summary([(float(k % 5) - 2, float(k % 5) - 2) for k in range(50)], None, None)
    assert not wide["collapse"] and wide["lane_loss"] is None
    src = Path(train_scene.__file__).read_text(encoding="utf-8")
    for token in ('"lane_loss": json_num', '"conf_bce": json_num', '"pred_mean": json_num', '"pred_std": json_num', "COLLAPSE?", '"r_long": json_num'):
        assert token in src, f"val record/line is missing {token}"
    # --conf-weight scales only the confidence BCE.
    rows, _cache = _synthetic_seq(1)
    from python.train.scene_net import labels_for_row
    from python.train.train_scene import _as_target

    torch.manual_seed(1)
    from python.train.scene_net import SceneNet

    net = SceneNet(cams=("main",))
    pred, _ = net.forward_batch(torch.zeros(1, 192, 256, 3, dtype=torch.uint8), torch.tensor([0.1]), net.initial_state(batch=1))
    from python.train.train_scene import _prediction_at

    one = _prediction_at(pred, 0)
    target = _as_target(labels_for_row(rows[0]), torch.device("cpu"))
    with torch.no_grad():
        a = {k: float(v) for k, v in scene_loss(one, target, ("lane_gt",), conf_weight=1.0).items()}
        b = {k: float(v) for k, v in scene_loss(one, target, ("lane_gt",), conf_weight=0.1).items()}
    assert abs(a["total"] - (a["conf_bce"] + a["lane_reg"])) < 1e-5
    assert abs(b["total"] - (0.1 * b["conf_bce"] + b["lane_reg"])) < 1e-5
    assert abs(a["conf_bce"] - b["conf_bce"]) < 1e-6 and a["conf_bce"] > 0
    from python.train.train_scene import build_parser

    args = build_parser().parse_args([])
    assert args.conf_weight == 0.1 and args.init_head_bias is False and args.eval_chunk == 16 and args.lr == 1e-3
    print("val fields: lane_loss conf_bce pred_mean pred_std r_long rms_long bias_long + COLLAPSE? flag; conf_weight scales the BCE only OK")


def check_bias_init() -> None:
    import torch
    from scipy.optimize import minimize_scalar

    from python.train.scene_net import SceneNet
    from python.train.train_scene import apply_head_bias, huber_slot_constants

    rng = np.random.RandomState(3)
    rows = []
    for k in range(400):
        tech = _tech(center=float(rng.standard_t(2) * 0.8 - 0.4), width=float(rng.uniform(3.2, 4.2)), heading=float(rng.normal(0, 0.05)))
        if k % 7 == 0:  # some invalid far slots, NaN like the painted labels
            tech["lane_gt"][9:12] = [float("nan")] * 3
            tech["lane_gt_valid"][9:12] = [0.0] * 3
        if k % 4 == 0:
            tech["lane_gt"][-1] = 0.0
        rows.append({"i": k, "tech": tech})
    consts = huber_slot_constants(rows)
    V = np.array([r["tech"]["lane_gt"] for r in rows], dtype=np.float64)
    M = np.array([r["tech"]["lane_gt_valid"] for r in rows], dtype=np.float64)

    def smooth_l1(c, a):
        d = np.abs(a - c)
        return np.where(d < 1.0, 0.5 * d * d, d - 0.5).mean()

    for j in range(LANE_GT_DIM - 1):
        col = V[M[:, j] > 0.5, j]
        best = minimize_scalar(lambda c: smooth_l1(c, col), bounds=(-20, 20), method="bounded", options={"xatol": 1e-7}).x
        assert abs(float(consts[j]) - best) < 2e-3, (j, float(consts[j]), best)
    p = V[:, -1].mean()
    assert abs(float(consts[-1]) - np.log(p / (1 - p))) < 1e-5
    net = SceneNet(cams=("main",))
    apply_head_bias(net, consts)
    assert torch.allclose(net.lane_gt_head.bias.detach(), torch.as_tensor(consts)), "bias must equal the constants"
    print(f"bias init: Huber constants match a direct smooth-L1 minimise (c5={consts[0]:+.3f}, conf_logit={consts[-1]:+.3f}); bias copied OK")


def check_map_mismatch() -> None:
    from python.train.continual import check_holdout_cells, load_holdout_cells, save_holdout_cells

    spec = {"version": 1, "map": "west_coast_usa", "cell_m": 250.0, "cells": [{"cx": 0, "cy": 0, "class": "fast", "rows": 5}]}
    same = [{"i": 0, "segment": "west_coast_usa#0"}]
    other = [{"i": 0, "segment": "italy#0"}]
    multi = same + [{"i": 1, "segment": "italy#0"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = save_holdout_cells(spec, Path(tmp) / "cells.json")
        assert load_holdout_cells(path) == spec
        assert load_holdout_cells(path, same) == spec
        assert load_holdout_cells(path, [{"i": 0}]) == spec, "rows without a map are not checked"
        for rows, word in ((other, "chosen on map 'west_coast_usa'"), (multi, "span 2 maps")):
            try:
                load_holdout_cells(path, rows)
            except ValueError as exc:
                assert word in str(exc), str(exc)
            else:
                raise AssertionError(f"no error for rows {rows}")
    per_map = dict(spec, cells=[dict(spec["cells"][0], map="west_coast_usa"), {"cx": 1, "cy": 1, "class": "slow", "rows": 3, "map": "italy"}])
    check_holdout_cells(per_map, multi)
    print("map mismatch: other-map rows and multi-map rows without per-map cells raise ValueError OK")


def check_bootstrap() -> None:
    from python.train.continual import bootstrap_cells, cell_scores

    rng = np.random.RandomState(5)
    pairs, rows = [], []
    for cell in range(12):
        off = rng.normal(0, 0.5)
        for _ in range(30):
            t = rng.normal(0, 1)
            pairs.append((0.6 * t + off + rng.normal(0, 0.4), t))
            rows.append({"ego": {"pos": [cell * 250.0 + 10.0, 5.0, 0.0]}})
    spec = {"cell_m": 250.0, "cells": []}
    cells = cell_scores(pairs, rows, spec)
    assert len(cells) == 12 and all(c["n"] == 30 for c in cells.values())
    a = bootstrap_cells(cells, n_boot=1000, seed=0)
    b = bootstrap_cells(cells, n_boot=1000, seed=0)
    c = bootstrap_cells(cells, n_boot=1000, seed=1)
    assert a == b, "same seed must give the same interval"
    assert a["r_ci"] != c["r_ci"], "another seed should move the interval"
    p = np.array(pairs)
    r_all = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
    assert a["r_ci"][0] < r_all < a["r_ci"][1], (a, r_all)
    assert a["n_cells"] == 12 and a["n_boot"] == 1000
    print(f"bootstrap: deterministic per seed, r={r_all:.3f} r_ci90={np.round(a['r_ci'], 3).tolist()} rms_ci90={np.round(a['rms_ci'], 3).tolist()} OK")


def check_backbone() -> None:
    """--backbone net: BGR->RGB + ImageNet norm inside, state passed through, own pause tag."""
    import torch

    try:
        import torchvision  # noqa: F401
    except ImportError:
        print("backbone: torchvision not installed, skipped")
        return
    from python.train.pretrained_net import IMAGENET_MEAN, IMAGENET_STD, ImageNetPreprocess, PretrainedSceneNet
    from python.train.train_scene import pause_matches_stem, stamp_pause

    pre = ImageNetPreprocess()
    bgr = torch.zeros(1, 3, 2, 2)
    bgr[:, 0] = 0.9  # blue channel of a BGR canvas
    out = pre(bgr)
    want_r = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    want_b = (0.9 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    assert abs(float(out[0, 0, 0, 0]) - want_r) < 1e-5 and abs(float(out[0, 2, 0, 0]) - want_b) < 1e-5, out[0, :, 0, 0]
    net = PretrainedSceneNet(("main",), 1.0, "mobilenetv3", pretrained=False)
    net.train()
    assert all(not m.training for m in net.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    state = net.initial_state(batch=2)
    pred, after = net.forward_batch(torch.rand(2, 3, net.canvas_h, net.canvas_w), torch.tensor([0.1, 0.1]), state)
    assert after is state and tuple(pred.lane_gt.shape) == (2, 14)
    # Frames are independent: frame 0's output does not depend on frame 1.
    net.eval()
    x = torch.rand(2, 3, net.canvas_h, net.canvas_w)
    with torch.no_grad():
        a = net.forward_batch(x, torch.tensor([0.1, 0.1]), net.initial_state(batch=2))[0].lane_gt[0]
        y = x.clone()
        y[1] = torch.rand_like(y[1])
        b = net.forward_batch(y, torch.tensor([0.1, 0.1]), net.initial_state(batch=2))[0].lane_gt[0]
    assert torch.allclose(a, b, atol=1e-5)
    payload = stamp_pause({}, ("main",), net.pause_arch)
    assert pause_matches_stem(payload, ("main",), net.pause_arch)
    assert not pause_matches_stem(payload, ("main",)), "a backbone pause must not load into SceneNet"
    assert not pause_matches_stem(stamp_pause({}, ("main",)), ("main",), net.pause_arch)
    frozen = PretrainedSceneNet(("main",), 1.0, "mobilenetv3", pretrained=False, freeze=True)
    assert not any(p.requires_grad for p in frozen.backbone.parameters())
    assert frozen.pause_arch != net.pause_arch
    print("backbone: BGR->RGB + ImageNet norm in the net, BN frozen, no state carried, frames independent, own pause tag OK")


def check_near_black() -> None:
    """Near-black main: mean < 2% or > 98% of pixels below 8/255. A dim real frame stays."""
    from python.train.train_scene import main_area_blank, near_black

    rng = np.random.RandomState(3)
    zero = np.zeros((192, 256, 3), np.uint8)
    noisy = zero.copy()
    noisy[rng.rand(192, 256) < 0.01] = 200  # 1% hot pixels, mean ~0.008
    dim = np.full((192, 256, 3), 3, np.uint8)
    dim[:2] = 255  # 1% bright rows: mean above 2%, but 99% of pixels below 8
    dusk = rng.randint(10, 40, size=(192, 256, 3)).astype(np.uint8)  # a dark but real picture
    assert near_black(zero) and near_black(noisy) and near_black(dim)
    assert not near_black(dusk), float(dusk.mean())
    assert near_black(zero.astype(np.float32)) and not near_black(dusk.astype(np.float32) / 255.0)
    assert main_area_blank(noisy, ("main",)) and not main_area_blank(dusk, ("main",))
    print(f"near-black: exact zeros, 1% hot pixels (mean {noisy.mean() / 255:.3f}) and 99%-dark frames dropped; a dim real frame (mean {dusk.mean() / 255:.3f}) kept OK")


def check_backbone_pool_and_lr() -> None:
    """4x8 pooling matches adaptive_avg_pool2d; --backbone-lr-mult groups survive LR pinning."""
    import torch
    import torch.nn.functional as functional

    try:
        import torchvision  # noqa: F401
    except ImportError:
        print("backbone pool/lr: torchvision not installed, skipped")
        return
    from python.train.pretrained_net import POOL_ROWS, PretrainedSceneNet, adaptive_pool
    from python.train.train_scene import _pin_optimizer_lr

    for shape in ((2, 8, 6, 8), (1, 4, 7, 13), (1, 2, 8, 16)):
        x = torch.randn(*shape)
        assert torch.allclose(adaptive_pool(x, 4, 8), functional.adaptive_avg_pool2d(x, (4, 8)), atol=1e-6), shape
    net = PretrainedSceneNet(("main",), 1.0, "mobilenetv3", pretrained=False)
    assert POOL_ROWS == 4 and net.token_dim == 64 * 4 * 8 and "pool4x8" in net.pause_arch
    # Rows matter now: two maps that differ only top vs bottom give different features.
    fm = torch.zeros(1, 576, 6, 8)
    top, bottom = fm.clone(), fm.clone()
    top[:, :, :3] = 1.0
    bottom[:, :, 3:] = 1.0
    with torch.no_grad():
        f_top = adaptive_pool(net.conv_head(top), 4, 8)
        f_bot = adaptive_pool(net.conv_head(bottom), 4, 8)
    assert float((f_top - f_bot).abs().max()) > 1e-4 and tuple(f_top.shape) == (1, 64, 4, 8)
    opt = torch.optim.AdamW(net.param_groups(1e-3, 1.0), lr=1e-3)
    _pin_optimizer_lr(opt, 1e-3)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 1e-3]
    opt = torch.optim.AdamW(net.param_groups(1e-3, 0.5), lr=1e-3)
    _pin_optimizer_lr(opt, 1e-3)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 5e-4]
    n_groups = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_groups == sum(p.numel() for p in net.parameters())
    print("backbone pool/lr: 4x8 pool == adaptive_avg_pool2d, vertical position survives, lr_mult 1.0 -> one rate, 0.5 kept through pinning OK")


def check_real(root: Path) -> None:
    from python.data.raw_frames import load_scene_rows
    from python.train.scene_loader import select_training_rows

    t0 = time.time()
    state = load_scene_rows(root)
    old, _m, _e = select_training_rows(root, state)
    dropped: dict = {}
    new, _m2, _e2 = select_training_rows(root, state, cams=("main",), dropped=dropped)
    lost = len(old) - len(new)
    print(f"real {root}: {len(old)} rows before, {len(new)} with main, dropped {dropped.get('no_trained_cam', 0)} ({time.time() - t0:.0f}s)")
    assert lost == int(dropped.get("no_trained_cam", 0))
    if len(old) == 17018:
        assert 1700 <= lost <= 1900, lost
        print(f"real: dropped {lost} vs audit estimate 1816 OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.environ.get("GVD_TEST_STRIPS", ""))
    args = ap.parse_args()
    check_black_rows()
    summary = check_chunked_eval()
    check_val_fields(summary)
    check_bias_init()
    check_map_mismatch()
    check_bootstrap()
    check_backbone()
    check_near_black()
    check_backbone_pool_and_lr()
    if args.real:
        check_real(Path(args.real))
    print("test_trainer_fixes: all OK")


if __name__ == "__main__":
    main()
