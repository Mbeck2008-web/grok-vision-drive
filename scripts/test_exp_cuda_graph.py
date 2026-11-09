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


if __name__ == "__main__":
    unittest.main()
