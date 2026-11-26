#!/usr/bin/env python3
"""Trainer 2.0 PR 5: SSL stem pretraining default ON for every train_scene run.

docs/specs/trainer_2.md "Teacher/student" (decided by Michael 2026-11-25: "It
should be on every training run"); ``--no-ssl-pretrain`` opt-out, EMA target
stem (momentum 0.996) and a collapse guard (per-dim token std on a fixed train
probe < 0.01 aborts) are spec proposals.

Checks: check_ssl_default_on, check_ssl_ema_no_collapse, check_ssl_rows_train_only,
check_ssl_reads_numpy_frames, check_ssl_empty_split_defers_to_trainer,
check_recipe_off_is_bit_identical (argparse part: every pre-existing default
equals the main 8ce34d5 pin; only the new SSL flags are added).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

DEFAULTS_PIN = ROOT / "tests" / "fixtures" / "trainer2" / "train_scene_defaults_8ce34d5.json"
SCALE = 0.5  # half-size main canvas keeps the CPU steps short; the pretext code path is the same
NEW_FLAGS = {"no_ssl_pretrain": False, "ssl_epochs": 1, "ssl_max_rows": 4000}


def _torch():
    try:
        import torch
    except ImportError:
        return None
    # Small nets: extra intra-op threads only oversubscribe a shared CI / box CPU.
    torch.set_num_threads(2)
    return torch


def _args(**kw):
    base = dict(no_ssl_pretrain=False, init_stem="", dry_run=False, sanity=False, backbone="scratch",
                holdout_cells=False, exp_ssl_pretrain=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _fixture_frames(height: int, width: int, n: int):
    """Real recorded camera frames (tests/fixtures jpgs), resized to the canvas, BGR 0-1 NCHW."""
    import cv2
    import torch

    paths = sorted((ROOT / "tests" / "fixtures").rglob("*.jpg"))[: 5 * n : 5]
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (width, height)).astype(np.float32) / 255.0
        out.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None])
    assert len(out) == n, (len(out), n)
    return out


def check_ssl_default_on() -> None:
    from python.train.exp_ssl import ssl_plan

    assert ssl_plan(_args(), resuming=False) == (True, "on (default)")
    cases = {
        "--no-ssl-pretrain": _args(no_ssl_pretrain=True),
        "--init-stem": _args(init_stem="stem.npz"),
        "dry run": _args(dry_run=True),
        "sanity": _args(sanity=True),
        "backbone": _args(backbone="resnet18"),
    }
    for why, args in cases.items():
        on, reason = ssl_plan(args, resuming=False)
        assert not on and why in reason, (why, on, reason)
    on, reason = ssl_plan(_args(), resuming=True)
    assert not on and "resume" in reason, reason
    try:
        ssl_plan(_args(holdout_cells=True), resuming=False)
    except ValueError as exc:
        assert "--no-ssl-pretrain" in str(exc)
    else:
        raise AssertionError("--holdout-cells with default SSL not refused (SSL rows are not cell-aware)")
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_default_on:argparse", "torch not installed (train_scene imports torch)")
    else:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        assert args.no_ssl_pretrain is False and ssl_plan(args, resuming=False)[0], vars(args)
        off = build_parser().parse_args(["--no-ssl-pretrain"])
        assert off.no_ssl_pretrain is True and not ssl_plan(off, resuming=False)[0]
        src = (ROOT / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
        assert "ssl_plan(" in src and "default_ssl_stem(" in src, "train_scene main() does not wire default SSL"
    print("ssl default: bare parse -> on; --no-ssl-pretrain / --init-stem / dry run / sanity / pretrained backbone / resume -> off with reason; --holdout-cells refused OK")


def check_ssl_ema_no_collapse() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_ema_no_collapse", "torch not installed")
        return
    from python.train.exp_ssl import (
        COLLAPSE_STD,
        EMA_MOMENTUM,
        SslError,
        collapse_std,
        ema_update,
        make_ema_target,
        run_ssl_pretrain,
    )
    from python.train.scene_net import SceneNet

    assert EMA_MOMENTUM == 0.996 and COLLAPSE_STD == 0.01
    torch.manual_seed(0)
    net = SceneNet(cams=("main",), scale=SCALE)
    target = make_ema_target(net)
    with torch.no_grad():
        for p in net.stem.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in target.stem.parameters()]
    ema_update(target, net, 0.996)
    for b, t, o in zip(before, target.stem.parameters(), net.stem.parameters()):
        assert torch.allclose(t, 0.996 * b + 0.004 * o, atol=1e-6)
    assert not any(p.requires_grad for p in target.parameters()), "EMA target must not train"
    images = _fixture_frames(int(net.canvas_h), int(net.canvas_w), 10)
    std = collapse_std(SceneNet(cams=("main",), scale=SCALE), images)
    assert std > COLLAPSE_STD, std
    rows = [{"i": i, "kind": "train"} for i in range(len(images))]
    with tempfile.TemporaryDirectory() as tmp:
        stats = run_ssl_pretrain(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=lambda r: images[int(r["i"])],
                                 out_path=Path(tmp) / "stem.npz", lr=1e-3, seed=0, epochs=2)
        assert stats["ema"] == 0.996 and stats["probe_n"] == 8, stats  # PROBE_N distinct first-epoch frames
        assert stats["probe_std_start"] > COLLAPSE_STD, stats
        assert stats["probe_std_end"] > COLLAPSE_STD, stats
        # A stem that collapses (every token constant) trips the guard and writes nothing.
        bad = SceneNet(cams=("main",), scale=SCALE)

        def collapse(_net):
            with torch.no_grad():
                for p in _net.stem.parameters():
                    p.zero_()

        out = Path(tmp) / "collapsed.npz"
        try:
            run_ssl_pretrain(bad, rows, image_fn=lambda r: images[int(r["i"])], out_path=out, lr=1e-3, seed=0,
                             epochs=2, _after_step=collapse)
        except SslError as exc:
            assert "collapse" in str(exc), exc
        else:
            raise AssertionError("collapsed stem not refused")
        assert not out.exists() and not list(Path(tmp).glob("*.partial*")), sorted(Path(tmp).iterdir())
    print(f"ssl ema: target = 0.996 target + 0.004 online, frozen; probe std {std:.4f} > 0.01; collapsed stem aborts, nothing written OK")


def check_ssl_rows_train_only() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_rows_train_only", "torch not installed")
        return
    from python.train.exp_ssl import default_ssl_stem, ssl_rows_from_split
    from python.train.scene_net import SceneNet

    raw = [{"i": i, "t": float(i) * 0.1, "dt_s": 0.1, "session": "rec", "source": "rec"} for i in range(600)]
    rows = ssl_rows_from_split(raw, {"multi": False}, holdout_blocks=1)
    hold_from = 480  # last 20% of 60 s at 10 Hz
    assert rows and max(int(r["i"]) for r in rows) <= hold_from - 40, max(int(r["i"]) for r in rows)  # >= 4 s purge
    seen: list[int] = []
    probe_net = SceneNet(cams=("main",), scale=SCALE)
    real = _fixture_frames(int(probe_net.canvas_h), int(probe_net.canvas_w), 10)
    frames = {i: real[i % len(real)] for i in range(600)}

    def image_fn(row):
        seen.append(int(row["i"]))
        return frames[int(row["i"])]

    with tempfile.TemporaryDirectory() as tmp:
        net = SceneNet(cams=("main",), scale=SCALE)
        path, info = default_ssl_stem(net, rows, image_fn=image_fn, out_dir=Path(tmp), lr=1e-3, seed=0,
                                      epochs=1, max_rows=12, key_extra={"cams": ["main"]})
        assert path.is_file() and info["reused"] is False and info["rows"] == 12, info
        assert set(seen) <= {int(r["i"]) for r in rows}, "SSL read a non-train row"
        assert len(set(seen)) == 12  # 12 distinct train row ids read once each
        n_seen = len(seen)
        path2, info2 = default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=image_fn, out_dir=Path(tmp), lr=1e-3,
                                        seed=0, epochs=1, max_rows=12, key_extra={"cams": ["main"]})
        assert path2 == path and info2["reused"] is True and len(seen) == n_seen, "same key must reuse, not retrain"
        path3, info3 = default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=image_fn, out_dir=Path(tmp), lr=1e-3,
                                        seed=1, epochs=1, max_rows=12, key_extra={"cams": ["main"]})
        assert path3 != path and path.is_file(), "a different seed writes a new stem; the old one is kept"
        assert not list(Path(tmp).glob("*.partial*"))
        try:
            default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), [dict(rows[0], kind="holdout")], image_fn=image_fn,
                             out_dir=Path(tmp), lr=1e-3, seed=0, epochs=1, max_rows=0, key_extra={})
        except ValueError:
            pass
        else:
            raise AssertionError("holdout row accepted by default SSL")
    print(f"ssl rows: {len(rows)} train rows (holdout + 4 s purge excluded), 12 evenly spaced used, key reuse, old stems kept OK")


def check_ssl_reads_numpy_frames() -> None:
    """``_frame_row`` returns a uint8 HWC numpy canvas. On main 8ce34d5 ``run_ssl_pretrain`` passed it
    straight to ``_encode`` (it has ``.shape``), so SSL on real recordings crashed. It must be converted."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_reads_numpy_frames", "torch not installed")
        return
    from python.train.exp_ssl import run_ssl_pretrain
    from python.train.scene_net import SceneNet

    net = SceneNet(cams=("main",), scale=SCALE)
    frames = [(f[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
              for f in _fixture_frames(int(net.canvas_h), int(net.canvas_w), 3)]
    with tempfile.TemporaryDirectory() as tmp:
        stats = run_ssl_pretrain(net, [{"i": i, "kind": "train"} for i in range(3)], image_fn=lambda r: frames[int(r["i"])],
                                 out_path=Path(tmp) / "stem.npz", lr=1e-3, seed=0)
        assert int(stats["n"]) == 3 and (Path(tmp) / "stem.npz").is_file(), stats
    print("ssl numpy: uint8 HWC canvases from _frame_row are converted (crashed on main) OK")


def check_ssl_empty_split_defers_to_trainer() -> None:
    """0 train rows: SSL is skipped (None) so train_directory's own data errors still surface."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_empty_split_defers_to_trainer", "torch not installed (train_scene imports torch)")
        return
    import python.train.exp_ssl as exp_ssl
    import python.train.strip_sources as strip_sources
    from python.train import train_scene

    calls = []
    saved = (strip_sources.load_multi_source_rows, exp_ssl.ssl_rows_from_split, exp_ssl.default_ssl_stem)
    strip_sources.load_multi_source_rows = lambda roots, holdout_source=None: ([], {})
    exp_ssl.ssl_rows_from_split = lambda rows, meta, holdout_blocks=None: []
    exp_ssl.default_ssl_stem = lambda *a, **k: calls.append(1)
    try:
        args = SimpleNamespace(holdout_blocks=None, scale=SCALE, lr=1e-3, seed=0, ssl_epochs=1, ssl_max_rows=10)
        with tempfile.TemporaryDirectory() as tmp:
            got = train_scene._default_ssl_stage(args, [Path(tmp)], Path(tmp), ("main",), {})
    finally:
        strip_sources.load_multi_source_rows, exp_ssl.ssl_rows_from_split, exp_ssl.default_ssl_stem = saved
    assert got is None and not calls, (got, calls)
    print("empty split: SSL skipped (None, no stem written); trainer keeps its own refusal OK")


def check_recipe_off_is_bit_identical() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_recipe_off_is_bit_identical", "torch not installed (train_scene imports torch)")
        return
    from python.train.train_scene import build_parser

    pin = json.loads(DEFAULTS_PIN.read_text(encoding="utf-8"))["defaults"]
    got = json.loads(json.dumps(vars(build_parser().parse_args(["--no-ssl-pretrain"])), sort_keys=True, default=str))
    added = {k: got[k] for k in set(got) - set(pin)}
    assert added == dict(NEW_FLAGS, no_ssl_pretrain=True), added
    assert not set(pin) - set(got), set(pin) - set(got)
    changed = sorted(k for k in pin if got[k] != pin[k])
    assert not changed, f"pre-existing train_scene defaults changed: {changed}"
    print(f"recipe off: {len(pin)} pre-existing defaults == main 8ce34d5 pin; only {sorted(NEW_FLAGS)} added OK")


def main() -> int:
    checks = (check_ssl_default_on, check_ssl_ema_no_collapse, check_ssl_rows_train_only, check_ssl_reads_numpy_frames,
              check_ssl_empty_split_defers_to_trainer, check_recipe_off_is_bit_identical)
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
