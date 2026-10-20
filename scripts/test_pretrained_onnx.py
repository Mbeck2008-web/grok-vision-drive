"""ONNX export of the --backbone single-frame net (mobilenetv3 and resnet18).

Exports with the trainer's own exporter (``export_scene_onnx``, opset 17),
runs ONNX Runtime on 2 frames and checks every output against torch within
1e-4, and checks that the BGR->RGB swap and the ImageNet mean/std are inside
the exported graph (so the runtime feeds the same BGR 0-1 canvas the trainer
does). CPU only; writes only to a temp dir.

    py -3 scripts/test_pretrained_onnx.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

TOL = 1e-4


def _graph_constants(model) -> list[np.ndarray]:
    from onnx import numpy_helper

    out = [numpy_helper.to_array(t) for t in model.graph.initializer]
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    out.append(numpy_helper.to_array(attr.t))
    return out


def _has_const(consts: list[np.ndarray], want: tuple[float, ...]) -> bool:
    w = np.asarray(want, dtype=np.float32)
    return any(c.size == 3 and np.allclose(c.reshape(-1).astype(np.float32), w, atol=1e-6) for c in consts)


def check(backbone: str, tmp: Path) -> None:
    import onnx
    import onnxruntime as ort

    from python.train.pretrained_net import BGR_TO_RGB, IMAGENET_MEAN, IMAGENET_STD, build_scene_net
    from python.train.scene_net import OUTPUT_FIELDS, export_scene_onnx

    torch.manual_seed(0)
    net = build_scene_net(("main",), 1.0, backbone).eval()
    path = tmp / f"{backbone}.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_scene_onnx(path, net, cams=("main",), heads=("lane_gt",))
    model = onnx.load(str(path))
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    consts = _graph_constants(model)
    # BGR->RGB: a Gather on the channel axis with indices (2, 1, 0).
    swap = False
    gathers = [n for n in model.graph.node if n.op_type == "Gather"]
    for node in gathers:
        idx = next((c for c in consts if c.dtype.kind == "i" and c.reshape(-1).tolist() == list(BGR_TO_RGB)), None)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        if idx is not None and axis == 1:
            swap = True
            break
    has_mean = _has_const(consts, IMAGENET_MEAN)
    has_std = _has_const(consts, IMAGENET_STD)
    assert swap, "no channel Gather (2, 1, 0) in the graph"
    assert has_mean and has_std, (has_mean, has_std)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    rng = np.random.default_rng(1)
    worst = 0.0
    lane_outs = []
    for k in range(2):
        # A BGR canvas in 0-1 with a blue-ish top and grey bottom, plus noise.
        frame = rng.random((1, 3, net.canvas_h, net.canvas_w), dtype=np.float32) * 0.3
        frame[:, 0, : net.canvas_h // 2] += 0.6
        frame[:, :, net.canvas_h // 2 :] += 0.4
        frame = np.clip(frame, 0.0, 1.0).astype(np.float32)
        state = net.initial_state(batch=1)
        feeds = {
            "strip": frame,
            "dt": np.full((1,), 0.1, dtype=np.float32),
            "h": state.h.numpy(),
            "tokens": state.tokens.numpy(),
            "mask": state.mask.numpy(),
        }
        # Pass-through state inputs come out as h.1 / tokens.1 / mask.1; unused dt is dropped.
        feeds = {n: feeds[n.split(".")[0]] for n in names}
        got = sess.run(None, feeds)
        with torch.no_grad():
            pred, new_state = net(torch.from_numpy(frame), torch.tensor([0.1]), state)
        ref = [getattr(pred, f).numpy() for f in OUTPUT_FIELDS] + [new_state.h.numpy(), new_state.tokens.numpy(), new_state.mask.numpy()]
        for a, b in zip(got, ref):
            worst = max(worst, float(np.abs(np.asarray(a) - b).max()))
        lane_outs.append(got[OUTPUT_FIELDS.index("lane_gt")])
    # Swapping B and R at the input must move the output: the swap is real, not a no-op.
    flipped = dict(feeds)
    flipped["strip"] = np.ascontiguousarray(feeds["strip"][:, ::-1])
    moved = float(np.abs(sess.run(None, flipped)[OUTPUT_FIELDS.index("lane_gt")] - lane_outs[-1]).max())
    assert worst < TOL, f"{backbone}: ORT vs torch max abs diff {worst:.2e} >= {TOL}"
    assert moved > 1e-4, f"{backbone}: channel order does not change the output"
    size = path.stat().st_size / 1e6
    print(
        f"{backbone}: opset {opset}, {size:.1f} MB, {len(model.graph.node)} nodes; "
        f"BGR->RGB Gather(axis=1, [2,1,0]) in graph=yes, ImageNet mean/std constants in graph=yes; "
        f"ORT vs torch on 2 frames max abs diff {worst:.2e} (< {TOL}); B<->R swap moves lane_gt by {moved:.3f} OK"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for backbone in ("mobilenetv3", "imagenet-resnet18"):
            check(backbone, Path(tmp))
    print("test_pretrained_onnx: OK")


if __name__ == "__main__":
    main()