"""CPU tests for --exp-ssl-pretrain / --init-stem (W4). Holdout never used."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ExpSslPretrainTests(unittest.TestCase):
    def test_flags_off_by_default(self) -> None:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(bool(getattr(args, "exp_ssl_pretrain", False)))
        self.assertEqual(str(getattr(args, "init_stem", "") or ""), "")

    def test_ssl_rows_skip_holdout(self) -> None:
        from python.train.exp_ssl import select_ssl_rows

        rows = [
            {"i": 0, "kind": "train"},
            {"i": 1, "kind": "holdout"},
            {"i": 2, "kind": "train"},
        ]
        kept = select_ssl_rows(rows)
        self.assertEqual([r["i"] for r in kept], [0, 2])

    def test_init_stem_loads_matching_keys(self) -> None:
        from python.train.exp_ssl import save_stem_weights, load_stem_into_net
        from python.train.scene_net import SceneNet

        import torch

        net = SceneNet(cams=("main",))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stem.npz"
            save_stem_weights(path, net)
            other = SceneNet(cams=("main",))
            # Scramble then reload
            with torch.no_grad():
                for p in other.stem.parameters():
                    p.mul_(0.0)
            load_stem_into_net(other, path)
            for a, b in zip(net.stem.parameters(), other.stem.parameters()):
                self.assertTrue(torch.allclose(a, b))

    def test_masked_pretext_step_finite(self) -> None:
        from python.train.exp_ssl import masked_token_pretext_loss
        from python.train.scene_net import SceneNet

        import torch

        net = SceneNet(cams=("main",))
        # main-only canvas
        image = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
        loss = masked_token_pretext_loss(net, image, mask_ratio=0.25, rng=np.random.RandomState(0))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
