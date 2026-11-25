#!/usr/bin/env python3
"""SceneNet PyTorch vs ONNX Runtime FP32 harness (python/train/ort_bench.py), CPU EP only.

Export succeeds at opset 17 with the live input shapes; parity over a recurrent
rollout holds at the stated tolerance; the parity check FAILS on a perturbed
weight, a dropped output, a wrong strip layout and a state that is not fed back;
the op/EP report and a short bench run. No GPU, no BeamNG.
"""
# gvd-ci: requires torch
from __future__ import annotations

import ast
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402

_CACHE: dict = {}


def _deps() -> str | None:
    missing = []
    for mod in ("onnx", "onnxruntime"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return ", ".join(missing) or None


def _model(tmp: Path):
    """One seeded export shared by every check (exporting costs a few seconds)."""
    if "net" not in _CACHE:
        from python.train import ort_bench as ob

        net = ob.build_net(seed=7)
        path = ob.export(net, tmp / "scene.onnx")
        _CACHE.update(net=net, path=path)
    return _CACHE["net"], _CACHE["path"]


def test_tolerance_is_the_stated_one() -> None:
    from python.train import ort_bench as ob

    # Stated 2026-11-23 before any measurement. Changing these is widening the gate.
    assert ob.PARITY_ATOL == 1e-4 and ob.PARITY_RTOL == 1e-3, (ob.PARITY_ATOL, ob.PARITY_RTOL)
    assert ob.SCENE_OPSET == 17


def test_within_tolerance_boundaries() -> None:
    from python.train.ort_bench import PARITY_ATOL, PARITY_RTOL, within_tolerance

    ref = np.array([0.0, 1.0, -10.0], dtype=np.float32)
    bound = PARITY_ATOL + PARITY_RTOL * np.abs(ref.astype(np.float64))
    assert within_tolerance(ref, ref)
    assert within_tolerance(ref, (ref + 0.5 * bound).astype(np.float64))
    assert within_tolerance(ref, (ref - 0.99 * bound).astype(np.float64))
    for i in range(3):
        bad = ref.astype(np.float64).copy()
        bad[i] += 1.5 * bound[i]
        assert not within_tolerance(ref, bad), i
        bad[i] = ref[i] - 1.5 * bound[i]
        assert not within_tolerance(ref, bad), i
    assert not within_tolerance(ref, ref + 1.0)
    assert not within_tolerance(ref, ref[:2]), "shape mismatch must fail"
    assert not within_tolerance(ref, ref.reshape(1, 3)), "shape mismatch must fail"
    nan = ref.astype(np.float64).copy()
    nan[1] = math.nan
    assert not within_tolerance(ref, nan) and not within_tolerance(nan, ref)
    assert not within_tolerance(nan, nan), "NaN never equals NaN"
    inf = ref.astype(np.float64).copy()
    inf[2] = math.inf
    assert within_tolerance(inf, inf)
    neg = inf.copy()
    neg[2] = -math.inf
    assert not within_tolerance(inf, neg)


def test_export_shapes_and_opset(tmp: Path) -> None:
    import onnx

    from python.train import ort_bench as ob
    from python.train.scene_net import OUTPUT_FIELDS, count_module_parameters

    net, path = _model(tmp)
    # labels v2 (8bbd07d/29f6f0f) dropped one object class: 16 slots x (512+1) = 8208 fewer than 2203694.
    assert count_module_parameters(net) == 2195486, count_module_parameters(net)
    model = onnx.load(str(path))
    assert {i.domain or "ai.onnx": i.version for i in model.opset_import}.get("ai.onnx") == 17
    sess = ob.make_session(path, "cpu")
    shapes = {ob.base_name(i.name): list(i.shape) for i in sess.get_inputs()}
    assert shapes == {"strip": [1, 3, 192, 2272], "dt": [1], "h": [1, 1, 512], "tokens": [1, 90, 512],
                      "mask": [1, 90]}, shapes
    outs = [ob.base_name(o.name) for o in sess.get_outputs()]
    assert outs == list(OUTPUT_FIELDS) + ["h", "tokens", "mask"], outs
    assert any(n.op_type == "GRU" for n in model.graph.node)


def test_parity_cpu_ep(tmp: Path) -> None:
    from python.train import ort_bench as ob

    net, path = _model(tmp)
    res = ob.check_parity(path, net, ep="cpu", ticks=4, seed=11)
    assert res.ok, res.failures
    assert res.ticks == 4 and set(res.per_output) == set(ob.output_names(net))
    for name, st in res.per_output.items():
        assert st["max_abs"] <= ob.PARITY_ATOL * 10, (name, st)
    print("parity cpu max_abs: " + ", ".join(f"{k}={v['max_abs']:.2e}" for k, v in res.per_output.items()))


def _rollouts(tmp: Path):
    from python.train import ort_bench as ob

    net, path = _model(tmp)
    if "ref" not in _CACHE:
        inputs = ob.make_inputs(ob.net_strip_shape(net), ticks=3, seed=5)
        _CACHE["inputs"] = inputs
        _CACHE["ref"] = ob.torch_rollout(net, inputs)
    return net, path, _CACHE["inputs"], _CACHE["ref"]


def test_parity_fails_on_perturbed_weights(tmp: Path) -> None:
    import onnx
    from onnx import numpy_helper

    from python.train import ort_bench as ob

    net, path, inputs, ref = _rollouts(tmp)
    model = onnx.load(str(path))
    # The lane_gt head bias: one output nudged by 10x the tolerance.
    target = None
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim == 1 and arr.size == 14 and "lane_gt" in init.name:
            target = init
    assert target is not None, [i.name for i in model.graph.initializer if "lane_gt" in i.name]
    arr = numpy_helper.to_array(target).copy()
    arr[0] += 1e-3
    target.CopyFrom(numpy_helper.from_array(arr, target.name))
    bad = tmp / "perturbed.onnx"
    onnx.save(model, str(bad))
    got = ob.ort_rollout(ob.make_session(bad, "cpu"), inputs)
    res = ob.compare(ref, got, ob.output_names(net))
    assert not res.ok and any("lane_gt" in f for f in res.failures), res.failures
    assert not any(f.split(": ")[1].startswith(("h ", "tokens ", "mask ")) for f in res.failures), res.failures


def test_parity_fails_on_dropped_output(tmp: Path) -> None:
    import onnx

    from python.train import ort_bench as ob

    net, path, inputs, ref = _rollouts(tmp)
    model = onnx.load(str(path))
    keep = [o for o in model.graph.output if ob.base_name(o.name) != "lane_gt"]
    del model.graph.output[:]
    model.graph.output.extend(keep)
    bad = tmp / "dropped.onnx"
    onnx.save(model, str(bad))
    got = ob.ort_rollout(ob.make_session(bad, "cpu"), inputs)
    res = ob.compare(ref, got, ob.output_names(net))
    assert not res.ok and any("'lane_gt' missing" in f for f in res.failures), res.failures


def test_parity_fails_on_wrong_strip_layout(tmp: Path) -> None:
    from python.train import ort_bench as ob

    net, path, inputs, ref = _rollouts(tmp)
    sess = ob.make_session(path, "cpu")
    for label, fn in (("bgr<->rgb", lambda s: s[:, ::-1].copy()),
                      ("hw flipped", lambda s: s[:, :, ::-1, ::-1].copy()),
                      ("nhwc bytes read as nchw", lambda s: np.ascontiguousarray(s.transpose(0, 2, 3, 1)).reshape(s.shape))):
        wrong = [{"strip": fn(f["strip"]), "dt": f["dt"]} for f in inputs]
        res = ob.compare(ref, ob.ort_rollout(sess, wrong), ob.output_names(net))
        assert not res.ok, label


def test_parity_fails_when_state_not_fed_back(tmp: Path) -> None:
    from python.train import ort_bench as ob

    net, path, inputs, ref = _rollouts(tmp)
    sess = ob.make_session(path, "cpu")
    got = [ob.ort_rollout(sess, [f])[0] for f in inputs]  # cold state every tick
    res = ob.compare(ref, got, ob.output_names(net))
    assert not res.ok and all(not f.startswith("tick 0:") for f in res.failures), res.failures


def test_op_report_cpu(tmp: Path) -> None:
    from python.train import ort_bench as ob

    _net, path = _model(tmp)
    rep = ob.op_report(path, ep="cpu", workdir=tmp)
    assert rep["session_providers"] == ["CPUExecutionProvider"]
    assert set(rep["nodes_by_provider"]) == {"CPUExecutionProvider"}, rep["nodes_by_provider"]
    assert rep["profiled_nodes"] > 50 and rep["fallback_ops"] == {} and rep["memcpy_total"] == 0, rep
    assert "GRU" in rep["ops_by_provider"]["CPUExecutionProvider"]
    print(f"op report cpu: {rep['profiled_nodes']} nodes, memcpy={rep['memcpy_total']}")


def test_memcpy_counts_synthetic() -> None:
    """The counter finds partitioner copies (a CUDA graph has them; the CPU graph has none)."""
    import onnx
    from onnx import helper

    from python.train.ort_bench import memcpy_counts

    nodes = [helper.make_node("MemcpyFromHost", ["x"], ["a"]), helper.make_node("Relu", ["a"], ["b"]),
             helper.make_node("MemcpyToHost", ["b"], ["c"]), helper.make_node("MemcpyToHost", ["c"], ["y"])]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])])
    assert memcpy_counts(helper.make_model(graph)) == {"MemcpyToHost": 2, "MemcpyFromHost": 1}


def test_bench_smoke(tmp: Path) -> None:
    from python.train import ort_bench as ob

    net, path = _model(tmp)
    out = ob.bench(path, net, ep="cpu", warmup=1, iters=5, threads=1)
    for key in ("ort", "torch_cpu"):
        st = out[key]
        assert st["n"] == 5 and 0 < st["p50_ms"] <= st["p90_ms"] <= st["p99_ms"] <= st["max_ms"], st
    assert out["state_h_absmax_after"] > 0.0, "bench must carry the GRU state between ticks"
    assert out["peak_rss_mb_after_ort_runs"] and out["peak_rss_mb_after_ort_runs"] > 0
    assert out["h2d_bytes_per_tick"] == 4 * (3 * 192 * 2272 + 1 + 512 + 90 * 512 + 90), out["h2d_bytes_per_tick"]
    print(f"bench smoke cpu: ort p50={out['ort']['p50_ms']:.1f} ms torch p50={out['torch_cpu']['p50_ms']:.1f} ms "
          f"rss={out['peak_rss_mb_after_torch']:.0f} MiB")


def test_cli_parity_exit(tmp: Path) -> None:
    import json
    import subprocess

    out = tmp / "cli.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "ort_bench.py"), "--ep", "cpu", "--ticks", "2",
                          "--warmup", "1", "--iters", "2", "--threads", "1", "--out", str(out)],
                         capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, res.stderr[-2000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["parity"]["ok"] and rep["parity"]["atol"] == 1e-4 and rep["parity"]["rtol"] == 1e-3
    assert rep["model"]["opset"] == 17 and rep["ops"]["memcpy_total"] == 0


def test_offline_only_never_live() -> None:
    """Nothing live imports the harness, and the harness reads no live/driving signal."""
    src = (ROOT / "python" / "train" / "ort_bench.py").read_text(encoding="utf-8").lower()
    for word in ("wheel", "electrics", "beamngpy", "aeb", "heartbeat", "engage"):
        assert word not in src.replace("(heartbeat / aeb / engage)", ""), word
    for py in (ROOT / "python").rglob("*.py"):
        if py.name == "ort_bench.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""] + [a.name for a in node.names]
            assert not any("ort_bench" in m for m in mods), py


def main() -> None:
    missing = _deps()
    test_tolerance_is_the_stated_one()
    test_within_tolerance_boundaries()
    test_offline_only_never_live()
    if missing:
        skip_check("ort_bench onnx checks", f"{missing} not installed (pip install -r requirements-ort.txt)")
        print("test_ort_bench: OK (onnx checks skipped)")
        return
    with tempfile.TemporaryDirectory(prefix="gvd_ort_test_") as name:
        tmp = Path(name)
        test_export_shapes_and_opset(tmp)
        test_parity_cpu_ep(tmp)
        test_parity_fails_on_perturbed_weights(tmp)
        test_parity_fails_on_dropped_output(tmp)
        test_parity_fails_on_wrong_strip_layout(tmp)
        test_parity_fails_when_state_not_fed_back(tmp)
        test_op_report_cpu(tmp)
        test_memcpy_counts_synthetic()
        test_bench_smoke(tmp)
        test_cli_parity_exit(tmp)
    print("test_ort_bench: OK")


if __name__ == "__main__":
    main()
