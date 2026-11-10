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

    def test_ssl_rows_refuse_missing_kind(self) -> None:
        from python.train.exp_ssl import SslError, select_ssl_rows

        with self.assertRaises(SslError) as ctx:
            select_ssl_rows([{"i": 0}, {"i": 1, "kind": "train"}])
        self.assertIn("no kind", str(ctx.exception))

    def test_train_row_one_second_after_holdout_excluded(self) -> None:
        from python.train.exp_ssl import select_ssl_rows

        rows = [
            {"i": 0, "kind": "train", "t": 10.0},
            {"i": 1, "kind": "holdout", "t": 20.0},
            {"i": 2, "kind": "train", "t": 21.0},  # 1 s after holdout; #187 gap
        ]
        kept = select_ssl_rows(rows)
        self.assertEqual([r["i"] for r in kept], [0])

    def test_ssl_rows_from_split_uses_187_and_strips_meta(self) -> None:
        from python.train.exp_ssl import ssl_rows_from_split

        raw = [{"i": i, "t": float(i), "dt_s": 0.1, "session": "rec"} for i in range(50)]
        kept = ssl_rows_from_split(raw, {"multi": False}, holdout_blocks=1)
        ids = [int(r["i"]) for r in kept]
        self.assertTrue(ids)
        self.assertTrue(all(r["kind"] == "train" for r in kept))
        # Last-fraction holdout (and the 4 s gap) must not appear as train.
        self.assertNotIn(49, ids)
        self.assertLess(max(ids), 39)

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

    def test_run_ssl_pretrain_uses_train_frames_not_zeros(self) -> None:
        from python.train.exp_ssl import SslError, run_ssl_pretrain
        from python.train.scene_net import SceneNet

        import torch

        net = SceneNet(cams=("main",))
        seen: list[int] = []

        def image_fn(row):
            seen.append(int(row["i"]))
            image = torch.full((1, 3, net.canvas_h, net.canvas_w), (int(row["i"]) + 1) / 10.0)
            return image

        rows = [{"i": 4, "kind": "train"}, {"i": 7, "kind": "train"}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stem.npz"
            stats = run_ssl_pretrain(net, rows, image_fn=image_fn, out_path=out, lr=1e-3, seed=0, epochs=1)
            self.assertTrue(out.is_file())
            self.assertEqual(seen, [4, 7])
            self.assertEqual(int(stats["n"]), 2)

        with self.assertRaises(SslError):
            run_ssl_pretrain(net, [], image_fn=image_fn, out_path=out, lr=1e-3, seed=0)


if __name__ == "__main__":
    unittest.main()
