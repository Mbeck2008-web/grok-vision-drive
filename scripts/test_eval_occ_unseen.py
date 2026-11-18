#!/usr/bin/env python3
"""Smoke test for scripts/eval_occ_unseen.py: every reported field exists and is sane (tiny run)."""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "tests"))
    from gvd_test_env import skip_check

    skip_check("eval_occ_unseen", "torch missing")
    raise SystemExit(0)

import eval_occ_unseen  # noqa: E402


def main() -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eval_occ_unseen.main(["--steps", "2", "--train-seeds", "1", "--eval-seeds", "1", "--frames", "2", "--threads", "2"])
    rep = json.loads(buf.getvalue())
    assert rep["synthetic"] is True and rep["train_samples"] == 4
    for kind in ("intersection", "roundabout"):
        m = rep["metrics_last_frame_mean"][kind]
        assert set(m) == {"road_beyond_window_per_clip", "map_unmasked", "map_unmasked_no_memory"}, set(m)
        for key in ("map_unmasked", "map_unmasked_no_memory"):
            for f in ("iou_visible", "iou_occluded", "edge_chamfer_visible", "edge_chamfer_occluded"):
                assert f in m[key] and (math.isnan(m[key][f]) or 0 <= m[key][f] <= 5), (kind, key, f, m[key])
            assert m[key]["gt_cells_occluded"] > 0, m[key]
        assert all(0.0 <= v <= 1.0 for v in m["road_beyond_window_per_clip"].values())
    v = rep["vram_est_mb_cpu_1080ti"]
    assert v["batch1"] > 0 and abs(v["batch4"] - v["batch1"]) > 0 and rep["runner_batch"]["offline_eval_and_live"] == 1
    print(f"eval_occ_unseen smoke: both clip kinds, visible/occluded IoU + edge chamfer, memory ablation, "
          f"window share, VRAM est batch 1 {v['batch1']:.0f} / batch 4 {v['batch4']:.0f} MB (CPU estimate) OK")
    print("test_eval_occ_unseen: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
