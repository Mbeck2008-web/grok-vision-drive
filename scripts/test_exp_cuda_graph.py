"""CPU tests for --exp-cuda-graph (W1). Off by default; CUDA path falls back here."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ExpCudaGraphTests(unittest.TestCase):
    def test_flag_off_by_default(self) -> None:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(bool(getattr(args, "exp_cuda_graph", False)))

    def test_cpu_falls_back_without_raising(self) -> None:
        from python.train.exp_cuda_graph import CudaGraphRunner, graph_eligible

        self.assertFalse(graph_eligible(enabled=True, device="cpu", shape=(1, 16, 3, 192, 256)))
        runner = CudaGraphRunner(enabled=True, device="cpu")
        calls = {"n": 0}

        def eager(static_in, static_out):
            calls["n"] += 1
            static_out.copy_(static_in * 2)

        out = runner.run(
            shape=(2,),
            allocate=lambda: (
                __import__("torch").zeros(2),
                __import__("torch").zeros(2),
            ),
            fill=lambda static_in, static_out: static_in.copy_(__import__("torch").tensor([1.0, 2.0])),
            eager=eager,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(runner.mode, "eager_fallback")
        self.assertTrue(bool(out is not None))

    def test_shape_change_forces_recapture_or_eager(self) -> None:
        from python.train.exp_cuda_graph import CudaGraphRunner

        runner = CudaGraphRunner(enabled=True, device="cpu")
        torch = __import__("torch")

        def make_eager(mult):
            def eager(static_in, static_out):
                static_out.copy_(static_in * mult)

            return eager

        runner.run(
            shape=(2,),
            allocate=lambda: (torch.zeros(2), torch.zeros(2)),
            fill=lambda s_in, s_out: s_in.copy_(torch.tensor([1.0, 1.0])),
            eager=make_eager(2),
        )
        first_shape = runner.last_shape
        runner.run(
            shape=(3,),
            allocate=lambda: (torch.zeros(3), torch.zeros(3)),
            fill=lambda s_in, s_out: s_in.copy_(torch.tensor([1.0, 1.0, 1.0])),
            eager=make_eager(3),
        )
        self.assertNotEqual(first_shape, runner.last_shape)
        self.assertEqual(runner.recapture_count, 0)  # CPU never captures
        self.assertEqual(runner.fallback_count, 2)

    def test_default_train_path_ignores_helper(self) -> None:
        """Without the flag, graph_eligible is false even on a cuda device string."""
        from python.train.exp_cuda_graph import graph_eligible

        self.assertFalse(graph_eligible(enabled=False, device="cuda", shape=(1, 8, 3, 192, 256)))

    def test_heads_allow_graph_only_lane_gt(self) -> None:
        from python.train.cam_strip import parse_heads
        from python.train.exp_cuda_graph import heads_allow_graph

        self.assertFalse(heads_allow_graph(None))
        self.assertFalse(heads_allow_graph(()))
        self.assertFalse(heads_allow_graph(("lane_gt", "ctl")))
        self.assertFalse(heads_allow_graph(parse_heads("")))
        self.assertTrue(heads_allow_graph(("lane_gt",)))
        self.assertTrue(heads_allow_graph(["lane_gt"]))

    def _toy_window(self):
        import numpy as np
        import torch

        from python.train.scene_net import SceneNet

        torch.manual_seed(73)
        net = SceneNet(cams=("main",))
        net.eval()
        device = torch.device("cpu")
        net.to(device)
        rng = np.random.RandomState(73)
        frames = [rng.randint(0, 255, (192, 256, 3), dtype=np.uint8) for _ in range(4)]
        stacked = torch.from_numpy(np.ascontiguousarray(np.stack(frames)[None, ...]))
        dts = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
        state = net.initial_state(device=device, batch=1)
        return net, stacked, dts, state

    def test_forward_window_cpu_matches_eager(self) -> None:
        import torch

        from python.train.exp_cuda_graph import CudaGraphRunner

        net, strips, dts, state = self._toy_window()
        runner = CudaGraphRunner(enabled=True, device="cpu", heads=("lane_gt",))
        pred_r, end_r, _extra_r = runner.forward_window(net, strips, dts, state)
        pred_e, end_e, _extra_e = net.forward_window(strips, dts, net.initial_state(device=strips.device, batch=1))
        self.assertEqual(runner.mode, "eager_fallback")
        self.assertGreaterEqual(runner.fallback_count, 1)
        self.assertIsNotNone(pred_r.lane_gt)
        self.assertIsNotNone(pred_r.signs)
        self.assertEqual(float((pred_r.lane_gt - pred_e.lane_gt).detach().abs().max()), 0.0)
        self.assertEqual(float((pred_r.signs - pred_e.signs).detach().abs().max()), 0.0)
        self.assertEqual(float((end_r.h - end_e.h).detach().abs().max()), 0.0)

    def test_forward_window_refuses_default_heads(self) -> None:
        """Default / empty heads must not capture; eager keeps every head."""
        import python.train.exp_cuda_graph as mod
        from python.train.cam_strip import parse_heads
        from python.train.exp_cuda_graph import CudaGraphRunner

        net, strips, dts, state = self._toy_window()
        runner = CudaGraphRunner(enabled=True, device="cpu", heads=parse_heads(""))
        pred_r, _end_r, _ = runner.forward_window(net, strips, dts, state)
        self.assertEqual(runner.mode, "eager_fallback")
        self.assertIsNotNone(pred_r.lane_gt)
        self.assertIsNotNone(pred_r.signs)
        self.assertIsNotNone(pred_r.objects)

        # Even if CUDA eligibility is forced, default heads must not call capture.
        captured = {"n": 0}
        real_eligible = mod.graph_eligible

        def fake_eligible(**_kw):
            return True

        import torch

        real_make = torch.cuda.make_graphed_callables

        def fake_make(*_a, **_k):
            captured["n"] += 1
            raise AssertionError("must not capture unless heads == {lane_gt}")

        runner = CudaGraphRunner(enabled=True, device="cuda", heads=parse_heads(""))
        mod.graph_eligible = fake_eligible
        torch.cuda.make_graphed_callables = fake_make
        try:
            pred_r, _end_r, _ = runner.forward_window(net, strips, dts, state)
        finally:
            mod.graph_eligible = real_eligible
            torch.cuda.make_graphed_callables = real_make
        self.assertEqual(captured["n"], 0)
        self.assertEqual(runner.mode, "eager_fallback")
        self.assertIsNotNone(pred_r.signs)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(),
        "CUDA capture test skipped: no CUDA",
    )
    def test_forward_window_cuda_capture(self) -> None:
        import numpy as np
        import torch

        from python.train.exp_cuda_graph import CudaGraphRunner
        from python.train.scene_net import SceneNet

        torch.manual_seed(73)
        net = SceneNet(cams=("main",))
        net.eval()
        device = torch.device("cuda")
        net.to(device)
        rng = np.random.RandomState(73)
        frames = [rng.randint(0, 255, (192, 256, 3), dtype=np.uint8) for _ in range(4)]
        strips = torch.from_numpy(np.ascontiguousarray(np.stack(frames)[None, ...])).to(device)
        dts = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32, device=device)
        runner = CudaGraphRunner(enabled=True, device=device, heads=("lane_gt",))
        pred_r, end_r, _ = runner.forward_window(net, strips, dts, net.initial_state(device=device, batch=1))
        pred_e, end_e, _ = net.forward_window(strips, dts, net.initial_state(device=device, batch=1))
        self.assertEqual(runner.fallback_count, 0)
        self.assertIn(runner.mode, ("captured", "replay"))
        self.assertLessEqual(float((pred_r.lane_gt - pred_e.lane_gt).abs().max()), 1e-5)
        self.assertLessEqual(float((end_r.h - end_e.h).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
