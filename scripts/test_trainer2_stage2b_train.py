#!/usr/bin/env python3
"""Trainer 2.0 stage 2b: ``train_directory`` end to end on a recording with stop signs and red lights (CPU).

Own script (and suite timeout) because it trains: the held-out segment holds a stop sign and a red light, the train
segment one stop sign (the fit's braking rows). Fixture: ``scripts/stage2b_fixture.py``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _recipe():
    from python.train import trainer2_recipe

    return trainer2_recipe


from gvd_test_env import skip_check  # noqa: E402


def _train_2b(torch, train_scene, rec, strips, base, tag, *, stage="2b", epochs=1):
    import contextlib
    import io

    kw = rec.stage_train_kwargs(stage)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            stats = train_scene.train_directory(
                strips, epochs=1, lr=1e-3, capability=None, export_path=None, free_vram_bytes=0,
                free_ram_bytes=8_000_000_000, resume=False, open_panel=False, autostart=True, cams=("main",),
                heads=kw["heads"], aux_depth=False, aux_sem=False, augment=False, workers=0, min_valid=0.0,
                holdout_blocks=1, scale=1.0, status_root=base / f"st_{tag}", ckpt_dir=base / f"ck_{tag}",
                exact_epochs=epochs, recipe="trainer2", recipe_stage=stage)
    except Exception as exc:  # noqa: BLE001
        return None, exc, buf.getvalue()
    return stats, None, buf.getvalue()


def check_stage_2b_trains_end_to_end() -> None:
    """``train_directory`` stage 2b on a recording with stop signs and red lights in a held-out segment (CPU): the
    holdout pass reports missed_stop / missed_light / false_brake from the model's own ctl output, the motion map was
    fitted on train rows only, and a recording that cannot score 2b refuses with the reason (never a vacuous 0)."""
    torch = _torch()
    if torch is None:
        skip_check("check_stage_2b_trains_end_to_end", "torch not installed")
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        skip_check("check_stage_2b_trains_end_to_end", "cv2 not installed")
        return
    import re

    import stage2b_fixture as fx
    from python.train import train_scene
    from python.train.stop_pred import StopNotMeasured

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        # refusals first (they stop before the first epoch)
        bare = fx.build(base / "bare", fx.N_ROWS, fx.intersections(fx.N_ROWS), with_signs=False)
        _s, err, out = _train_2b(torch, train_scene, rec, bare, base, "bare")
        assert isinstance(err, StopNotMeasured) and "0 stop-sign/line" in str(err) and "vacuous" in str(err), (err, out[-800:])
        no_stops = fx.build(base / "nostops", fx.N_ROWS, [], with_signs=True)
        _s, err, out = _train_2b(torch, train_scene, rec, no_stops, base, "nostops")
        assert isinstance(err, StopNotMeasured), (err, out[-800:])  # never brakes: no braking rows to fit, no events
        # the real thing
        strips = fx.build(base / "stops", fx.N_ROWS, fx.intersections(fx.N_ROWS))
        stats, err, out = _train_2b(torch, train_scene, rec, strips, base, "ok")
        assert err is None, f"stage 2b crashed: {type(err).__name__}: {err}\n{out[-2500:]}"
        fit = re.search(r"stop motion fitted on (\d+) train row pairs \((\d+) braking\): k_thr=([\d.]+) k_brk=([\d.]+) "
                        r"k_drag=([\d.]+) \(holdout ids in fit set: 0\)", out)
        assert fit, out[-2500:]
        n_pairs, n_brake = int(fit.group(1)), int(fit.group(2))
        k_thr, k_brk, k_drag = (float(fit.group(g)) for g in (3, 4, 5))
        # the fixture's car has K_THR 2.0 / K_BRK 4.5 / K_DRAG 0.05 and the train segment is 150 rows: the fit saw only those
        assert 30 <= n_pairs < fx.SPLIT and n_brake >= 10, (n_pairs, n_brake)
        assert abs(k_thr - fx.K_THR) < 0.05 and abs(k_brk - fx.K_BRK) < 0.05 and abs(k_drag - fx.K_DRAG) < 0.01, fit.group(0)
        assert "stage 2b holdout events: stop-sign/line=1 red-yellow-light=1 rows=150" in out, out[-2500:]
        assert "[GVD] val stop:" in out, out[-2500:]
        evals = stats["holdout_evals"]
        assert len(evals) == 1, evals
        e = evals[0]
        for key in rec.STOP_METRICS:
            assert e[key] is not None and np.isfinite(e[key]) and e[key] >= 0.0, (key, e)
        assert e["stop_events"] == 1 and e["light_events"] == 1 and e["stop_km"] > 0.0, e
        assert e["missed_stop"] <= e["stop_events"] and e["missed_light"] <= e["light_events"], e
        # the recipe turns it into the stage result the runner gates on
        res = rec.stage_result_from_stats("2b", stats)
        assert all(res.metrics[k] == e[k] for k in rec.STOP_METRICS), (res.metrics, e)
        # stage 7 on the same recording measures them again (a 2b ran before it)
        stats7, err, out7 = _train_2b(torch, train_scene, rec, strips, base, "ok7", stage="7")
        assert err is None, f"stage 7 crashed: {type(err).__name__}: {err}\n{out7[-2500:]}"
        assert all(k in stats7["holdout_evals"][0] for k in rec.STOP_METRICS), stats7["holdout_evals"]
    print("stage 2b trains end to end: motion fitted on train rows only, missed_stop / missed_light / false_brake from the "
          "holdout pass, refused (not a vacuous 0) without sign ids or without braking OK")


def main() -> int:
    checks = (check_stage_2b_trains_end_to_end,)
    for fn in checks:
        fn()
    print(f"{len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
