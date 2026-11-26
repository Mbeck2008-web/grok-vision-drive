#!/usr/bin/env python3
"""depth_hires (#265) CPU regression pins: shapes, dtypes, params, batch independence, ONNX parity.

Checks (python/train/depth_hires.py, tiny + small presets, CPU only):
  check_hires_shapes_dtypes_pinned   aux_depth / aux_depth_far are float32 (B,1,48,64) at scale 1 and 0.5, B in {1,3};
                                     tiny scale-1 FPN inputs are (32,48,64) / (64,24,32) / (64,12,16); metres >= MIN_DEPTH_M, finite
  check_hires_param_counts_pinned    decoder params: tiny 14594, small 19714 (exact)
  check_hires_batch_independent      a batch row equals the same image run alone (GroupNorm, no batch stats)
  check_hires_loss_edges_finite      all-invalid, all-far and 1 cm depth maps give finite loss + finite grads
  check_depth_term_is_masked_mean    the metric term is the masked mean only (huge error on masked-out cells is ignored)
  check_head_onnx_matches_torch      the decoder alone exports (opset 17) and onnxruntime matches torch (atol 1e-5)
  check_full_export_unchanged        export_scene_onnx on a depth_hires net: same output names as OUTPUT_FIELDS (head not
                                     exported in #265), every output matches torch (atol 1e-4)
Torch / onnx / onnxruntime checks skip visibly when missing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402

HEAD_PARAMS = {"tiny": 14594, "small": 19714}
TINY_FPN_IN = [(32, 48, 64), (64, 24, 32), (64, 12, 16)]


def _torch():
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(2)
    return torch


def _net(torch, preset: str = "tiny", scale: float = 1.0, hires: bool = True):
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    torch.manual_seed(0)
    return SceneNet(aux_sensors=True, depth_hires=hires, scale=scale, **scene_net_kwargs(preset)).eval()


def _fwd(torch, net, img):
    b = int(img.shape[0])
    pred, _state = net.forward_batch(img, torch.full((b,), 0.1), net.initial_state(batch=b))
    return pred


def _main_crop(net, img):
    s = next(s for s in net.sectors if str(s.cam_id) == "main")
    return img[:, :, :, s.x0 : s.x1]


def check_hires_shapes_dtypes_pinned() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_hires_shapes_dtypes_pinned", "torch not installed")
        return
    from python.train.depth_hires import DEPTH_HI_H, DEPTH_HI_W, MIN_DEPTH_M, stem_pyramid

    assert (DEPTH_HI_H, DEPTH_HI_W) == (48, 64)
    for scale in (1.0, 0.5):
        net = _net(torch, "tiny", scale)
        for b in (1, 3):
            img = torch.rand(b, 3, net.canvas_h, net.canvas_w)
            with torch.no_grad():
                pred = _fwd(torch, net, img)
            for name in ("aux_depth", "aux_depth_far"):
                t = getattr(pred, name)
                assert tuple(t.shape) == (b, 1, 48, 64), (scale, b, name, tuple(t.shape))
                assert t.dtype == torch.float32, (name, t.dtype)
                assert bool(torch.isfinite(t).all()), name
            assert float(pred.aux_depth.min()) >= MIN_DEPTH_M - 1e-6
        if scale == 1.0:
            with torch.no_grad():
                pyr = stem_pyramid(net.stem, _main_crop(net, torch.rand(1, 3, net.canvas_h, net.canvas_w)))
            got = [tuple(int(x) for x in m.shape[1:]) for m in pyr[-3:]]
            assert got == TINY_FPN_IN, got
            assert all(m.dtype == torch.float32 for m in pyr)
    print("shapes: float32 (B,1,48,64) depth + far at scale 1/0.5, B 1/3; tiny FPN inputs pinned OK")


def check_hires_param_counts_pinned() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_hires_param_counts_pinned", "torch not installed")
        return
    for preset, want in HEAD_PARAMS.items():
        net = _net(torch, preset)
        got = sum(int(p.numel()) for p in net.depth_hires_head.parameters())
        assert got == want, (preset, got, want)
    print(f"params: decoder {HEAD_PARAMS} OK")


def check_hires_batch_independent() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_hires_batch_independent", "torch not installed")
        return
    net = _net(torch, "tiny")
    img = torch.rand(3, 3, net.canvas_h, net.canvas_w)
    with torch.no_grad():
        pb = _fwd(torch, net, img)
        p1 = _fwd(torch, net, img[1:2])
    for name in ("aux_depth", "aux_depth_far"):
        d = float((getattr(pb, name)[1] - getattr(p1, name)[0]).abs().max())
        assert d < 1e-5, (name, d)
    print("batch: row 1 of B=3 == single run OK")


def check_hires_loss_edges_finite() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_hires_loss_edges_finite", "torch not installed")
        return
    from python.train.depth_hires import depth_hires_loss, depth_hires_targets

    net = _net(torch, "tiny").train()
    img = torch.rand(2, 3, net.canvas_h, net.canvas_w)
    keys = (("depth_m", "aux_depth"), ("depth_valid", "aux_depth_valid"), ("far", "aux_depth_far"), ("far_valid", "aux_depth_far_valid"))
    cases = {
        "all_invalid": (np.zeros((192, 256), np.uint16), (0.0, 0.0)),
        "all_far": (np.full((192, 256), 65535, np.uint16), (0.0, None)),
        "one_cm": (np.ones((192, 256), np.uint16), (None, None)),
    }
    for name, (cm, zero) in cases.items():
        t = depth_hires_targets(cm)
        tg = {k2: torch.tensor(t[k1])[None].repeat(2, 1, 1) for k1, k2 in keys}
        net.zero_grad()
        pred = _fwd(torch, net, img)
        d, f = depth_hires_loss(pred.aux_depth, pred.aux_depth_far, tg)
        (d + f).backward()
        assert bool(torch.isfinite(d)) and bool(torch.isfinite(f)), (name, float(d), float(f))
        for i, z in enumerate(zero):
            if z is not None:
                v = float((d, f)[i].detach())
                assert v == z, (name, i, v)
        assert all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None), name
    print("loss: all-invalid / all-far / 1 cm finite, masked terms exactly 0 OK")


def check_depth_term_is_masked_mean() -> None:
    """Invalid cells must not reach the metric term (a mutant with err.mean() survived every #265 check)."""
    torch = _torch()
    if torch is None:
        skip_check("check_depth_term_is_masked_mean", "torch not installed")
        return
    import torch.nn.functional as functional

    from python.train.depth_hires import depth_hires_loss

    pred = torch.full((1, 1, 48, 64), 10.0)
    goal = torch.full((1, 48, 64), 20.0)
    goal[:, :, 32:] = 1000.0  # huge error, but masked out
    mask = torch.zeros(1, 48, 64)
    mask[:, :, :32] = 1.0
    tg = {"aux_depth": goal, "aux_depth_valid": mask, "aux_depth_far": torch.zeros(1, 48, 64), "aux_depth_far_valid": torch.zeros(1, 48, 64)}
    d, f = depth_hires_loss(pred, torch.zeros(1, 1, 48, 64), tg)
    want = float(functional.smooth_l1_loss(torch.log(torch.tensor(10.0)), torch.log(torch.tensor(20.0))))
    assert abs(float(d) - want) < 1e-6, (float(d), want)
    assert float(f) == 0.0, float(f)
    print(f"loss: depth term == masked smooth-L1 on log m ({want:.6f}), invalid cells ignored OK")


def _ort():
    try:
        import onnx  # noqa: F401
        import onnxruntime
    except ImportError:
        return None
    return onnxruntime


def check_head_onnx_matches_torch() -> None:
    torch = _torch()
    ort = _ort()
    if torch is None or ort is None:
        skip_check("check_head_onnx_matches_torch", "torch / onnx / onnxruntime not installed")
        return
    from python.train.depth_hires import stem_pyramid

    net = _net(torch, "tiny")
    with torch.no_grad():
        pyr = stem_pyramid(net.stem, _main_crop(net, torch.rand(1, 3, net.canvas_h, net.canvas_w)))[-3:]

    class Head(torch.nn.Module):
        def __init__(self, head) -> None:
            super().__init__()
            self.head = head

        def forward(self, c2, c3, c4):
            return self.head([c2, c3, c4])

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "head.onnx")
        kw = dict(input_names=["c2", "c3", "c4"], output_names=["depth_m", "far_logit"], opset_version=17)
        try:
            torch.onnx.export(Head(net.depth_hires_head).eval(), tuple(pyr), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(Head(net.depth_hires_head).eval(), tuple(pyr), path, **kw)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"c2": pyr[0].numpy(), "c3": pyr[1].numpy(), "c4": pyr[2].numpy()})
    with torch.no_grad():
        ref = net.depth_hires_head(list(pyr))
    for name, a, b in zip(("depth_m", "far_logit"), got, ref):
        assert a.shape == (1, 1, 48, 64) and a.dtype == np.float32, (name, a.shape, a.dtype)
        d = float(np.abs(a - b.numpy()).max())
        assert d < 1e-5, (name, d)
    print("onnx head: ort == torch (atol 1e-5), float32 (1,1,48,64) OK")


def check_full_export_unchanged() -> None:
    torch = _torch()
    ort = _ort()
    if torch is None or ort is None:
        skip_check("check_full_export_unchanged", "torch / onnx / onnxruntime not installed")
        return
    import onnx

    from python.train import scene_net as sn

    net = _net(torch, "tiny")
    with tempfile.TemporaryDirectory() as td:
        path = sn.export_scene_onnx(os.path.join(td, "net.onnx"), net)
        names = [o.name for o in onnx.load(str(path)).graph.output]
        assert names == list(sn.OUTPUT_FIELDS) + ["h", "tokens", "mask"], names
        assert not any("depth" in n for n in names), names
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        model, args = sn.cpu_export_inputs(net, torch.zeros)
        torch.manual_seed(1)
        args = (torch.rand_like(args[0]),) + tuple(args[1:])
        got = sess.run(None, {i.name: a.numpy() for i, a in zip(sess.get_inputs(), args)})
    with torch.no_grad():
        ref = sn._OnnxScene(model).eval()(*args)
    worst = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(got, ref))
    assert len(got) == len(names) and worst < 1e-4, worst
    print(f"onnx net: depth_hires net exports {len(names)} outputs, no depth output (not exported in #265), max abs {worst:.1e} OK")


def main() -> int:
    checks = (
        check_hires_shapes_dtypes_pinned,
        check_hires_param_counts_pinned,
        check_hires_batch_independent,
        check_hires_loss_edges_finite,
        check_depth_term_is_masked_mean,
        check_head_onnx_matches_torch,
        check_full_export_unchanged,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
