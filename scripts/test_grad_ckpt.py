#!/usr/bin/env python3
"""``--grad-ckpt``: the per-tick encoder under ``torch.utils.checkpoint``.

* same seed and batch, with vs without: same loss, max gradient difference
  <= 1e-5 over every parameter (CPU), and the checkpoint really ran.
* eval / no_grad does not checkpoint and gives the same outputs; the weights
  (state_dict) are the same keys and values.
* the ONNX of a grad-ckpt net equals the plain net's ONNX output.
* the CLI flag is off by default; a tiny training run with it on logs the
  flag, and every epoch checkpoint exports and scores within 1e-4 of its
  logged val row (checkpoint / export / eval unaffected).
* refused with ``--backbone imagenet-*``.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402


def _counting_checkpoint():
    import torch.utils.checkpoint as tuc

    calls = {"n": 0}
    real = tuc.checkpoint

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        assert kwargs.get("use_reentrant") is False, kwargs
        return real(*args, **kwargs)

    return tuc, real, wrapped, calls


def _loss(net, strips, dts, seed: int):
    import torch

    torch.manual_seed(seed)
    state = net.initial_state(batch=int(strips.shape[1]))
    total = None
    for tick in range(int(strips.shape[0])):
        pred, state = net.forward_batch(strips[tick], dts[tick], state)
        part = sum(value.float().pow(2).mean() for value in pred.as_dict().values() if value is not None)
        total = part if total is None else total + part
    return total


def check_gradients() -> None:
    import torch

    from python.train.scene_net import SceneNet

    torch.use_deterministic_algorithms(False)
    torch.manual_seed(0)
    plain = SceneNet(cams=("pillarL", "main", "wide"))
    ckpt = copy.deepcopy(plain)
    ckpt.grad_ckpt = True
    plain.train()
    ckpt.train()
    rng = np.random.RandomState(3)
    width = plain.canvas_w
    strips = torch.from_numpy(rng.rand(3, 2, 3, plain.canvas_h, width).astype(np.float32))
    dts = torch.full((3, 2), 1.0 / 15.0)
    tuc, real, wrapped, calls = _counting_checkpoint()
    tuc.checkpoint = wrapped
    try:
        loss_a = _loss(plain, strips, dts, seed=7)
        loss_a.backward()
        assert calls["n"] == 0
        loss_b = _loss(ckpt, strips, dts, seed=7)
        loss_b.backward()
        assert calls["n"] == 3, calls
    finally:
        tuc.checkpoint = real
    assert float(loss_a) == float(loss_b), (float(loss_a), float(loss_b))
    worst = 0.0
    named_b = dict(ckpt.named_parameters())
    for name, param in plain.named_parameters():
        other = named_b[name]
        if param.grad is None:
            assert other.grad is None, name
            continue
        worst = max(worst, float((param.grad - other.grad).abs().max()))
    assert worst <= 1e-5, worst
    print(f"gradients: same loss {float(loss_a):.6f}, max |grad diff| {worst:.2e} <= 1e-5 over all params; checkpoint ran 3x OK")


def check_eval_and_export() -> None:
    import onnxruntime as ort
    import torch

    from python.train.scene_net import SceneNet, export_scene_onnx

    torch.manual_seed(0)
    plain = SceneNet(cams=("main",))
    ckpt = copy.deepcopy(plain)
    ckpt.grad_ckpt = True
    sa, sb = plain.state_dict(), ckpt.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    plain.eval()
    ckpt.eval()
    strip = torch.rand(1, 3, plain.canvas_h, plain.canvas_w)
    dt = torch.tensor([1.0 / 15.0])
    tuc, real, wrapped, calls = _counting_checkpoint()
    tuc.checkpoint = wrapped
    try:
        with torch.no_grad():
            pa, _ = plain.forward_batch(strip, dt, plain.initial_state())
            pb, _ = ckpt.forward_batch(strip, dt, ckpt.initial_state())
        assert calls["n"] == 0
    finally:
        tuc.checkpoint = real
    assert torch.equal(pa.lane_gt, pb.lane_gt)
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        ckpt.train()  # the export puts it in eval itself
        for name, net in (("a", plain), ("b", ckpt)):
            path = Path(tmp) / f"{name}.onnx"
            export_scene_onnx(path, net, cams=("main",), heads=("lane_gt",))
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            feeds = {}
            for item in sess.get_inputs():
                shape = [int(d) for d in item.shape]
                feeds[item.name] = np.full(shape, 0.5, dtype=np.float32)
            outs.append(sess.run(None, feeds))
        for x, y in zip(*outs):
            assert np.array_equal(x, y)
    print("eval/export: no checkpoint under eval or no_grad; same weights, outputs and ONNX OK")


def check_cli_and_run() -> None:
    import torch

    import test_epoch_ckpt as fx
    from eval_onnx_holdout import evaluate
    from python.train.epoch_ckpt import export_epoch_onnx
    from python.train.train_scene import build_parser, train_directory

    assert build_parser().parse_args([]).grad_ckpt is False
    assert build_parser().parse_args(["--grad-ckpt"]).grad_ckpt is True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        fx._fixture(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fx._train(root, grad_ckpt=True, backbone="imagenet-resnet18", status_root=Path(tmp) / "_x")
        except SystemExit as exc:
            assert "--grad-ckpt" in str(exc), exc
        else:
            raise AssertionError("--grad-ckpt accepted with an ImageNet backbone")
        torch.manual_seed(0)
        ckpts = Path(tmp) / "ck"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            fx._train(root, ckpt_dir=ckpts, status_root=Path(tmp) / "_s", grad_ckpt=True)
        assert "[GVD] grad-ckpt:" in log.getvalue()
        vals = fx._val_records(Path(tmp) / "_s")
        assert vals
        for rec in vals[:2] + vals[-1:]:
            out = Path(tmp) / f"ep{rec['epoch']}.onnx"
            export_epoch_onnx(ckpts / rec["ckpt"], out)
            got = evaluate(root, out, holdout_blocks=1, eval_chunk=16)
            assert got["n"] == rec["n"]
            for key in ("rms", "pred_std", "r"):
                assert abs(float(got[key]) - float(rec[key])) < 1e-4, (key, got[key], rec[key])
    print("cli/run: off by default; a --grad-ckpt run's checkpoints export and score within 1e-4; imagenet refused OK")


def main() -> None:
    check_gradients()
    check_eval_and_export()
    check_cli_and_run()
    print("test_grad_ckpt: all OK")


if __name__ == "__main__":
    main()
