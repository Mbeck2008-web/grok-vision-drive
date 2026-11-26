#!/usr/bin/env python3
"""Trainer 2.0 PR 5: SSL stem pretraining default ON for every train_scene run.

docs/specs/trainer_2.md "Teacher/student" (decided by Michael 2026-11-25: "It
should be on every training run"); ``--no-ssl-pretrain`` opt-out, EMA target
stem (momentum 0.996) and a collapse guard (per-dim token std on a fixed train
probe < 0.01 aborts) are spec proposals.

Checks: check_ssl_default_on, check_ssl_ema_no_collapse, check_ssl_rows_train_only,
check_ssl_reads_numpy_frames, check_ssl_empty_split_defers_to_trainer,
check_main_passes_ssl_stem, check_ssl_loss_target_is_ema_copy, check_ssl_empty_split_refuses_when_trainer_trains,
check_ssl_recorded_in_stats_and_ckpt, check_ssl_oom_exits_2_and_releases_cuda, check_ssl_stem_key_tracks_recipe_and_content,
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
    import python.train.exp_ssl as exp_ssl
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
        trained = {"n": 0}
        real_run = exp_ssl.run_ssl_pretrain

        def counting_run(*a, **k):
            trained["n"] += 1
            return real_run(*a, **k)

        exp_ssl.run_ssl_pretrain = counting_run
        stem_bytes = path.read_bytes()
        path2, info2 = default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=image_fn, out_dir=Path(tmp), lr=1e-3,
                                        seed=0, epochs=1, max_rows=12, key_extra={"cams": ["main"]})
        assert path2 == path and info2["reused"] is True and trained["n"] == 0 and path.read_bytes() == stem_bytes, "same key must reuse, not retrain"
        path3, info3 = default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=image_fn, out_dir=Path(tmp), lr=1e-3,
                                        seed=1, epochs=1, max_rows=12, key_extra={"cams": ["main"]})
        exp_ssl.run_ssl_pretrain = real_run
        assert trained["n"] == 1 and path3 != path and path.is_file(), "a different seed trains once and writes a new stem; the old one is kept"
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
            got, meta = train_scene._default_ssl_stage(args, [Path(tmp)], Path(tmp), ("main",), {})
    finally:
        strip_sources.load_multi_source_rows, exp_ssl.ssl_rows_from_split, exp_ssl.default_ssl_stem = saved
    assert got is None and not calls, (got, calls)
    assert meta["empty_split"] is True and meta["on"] is False and meta["key"] is None, meta
    print("empty split: SSL gives no stem and says empty_split; train_directory decides (see refuses_when_trainer_trains) OK")


class _MainHarness:
    """Run ``train_scene.main`` on a tiny recording with ``train_directory`` and the SSL pretext stubbed.

    Records what main hands to ``train_directory`` (``kw``) and whether SSL ran (``ssl_calls``).
    """

    def __init__(self, tmp: Path, *, ssl_error=None):
        self.tmp = tmp
        self.kw: dict = {}
        self.ssl_calls = 0
        self.ssl_error = ssl_error
        self.stem = tmp / "cache" / "ssl" / "ssl_stem_feedface00000001.npz"

    def run(self, extra: list[str]) -> tuple[int | None, str]:
        import contextlib
        import io
        import os

        import python.data.data_root as data_root
        import python.train.exp_ssl as exp_ssl
        from python.train import train_scene

        strips = self.tmp / "data" / "scene_x"
        if not strips.exists():
            strips.mkdir(parents=True)
            (strips / "state.jsonl").write_text("".join(json.dumps({"i": i, "t": 0.1 * i, "segment": "s", "cams": {}}) + "\n" for i in range(5)), encoding="utf-8")
            (strips / "raw" / "main").mkdir(parents=True)
        harness = self

        def fake_stem(net, rows, **kw):
            harness.ssl_calls += 1
            if harness.ssl_error is not None:
                raise harness.ssl_error
            return harness.stem, {"key": "feedface00000001", "rows": len(rows), "reused": False, "n": 1.0}

        def fake_train(root, **kw):
            harness.kw = kw
            return {"strips": 5, "windows": 0, "supervised": 0, "precision": "fp32", "device": "cpu", "batch": 1,
                    "recommended": False, "export": None}

        saved = (exp_ssl.default_ssl_stem, exp_ssl.ssl_rows_from_split, train_scene.train_directory, data_root.free_bytes,
                 os.environ.get("GVD_DATA_DIR"), sys.argv)
        exp_ssl.default_ssl_stem = fake_stem
        exp_ssl.ssl_rows_from_split = lambda rows, meta, holdout_blocks=None: [{"i": 0, "kind": "train", "source": ""}]
        train_scene.train_directory = fake_train
        data_root.free_bytes = lambda _path: 500 * 1024 ** 3
        os.environ["GVD_DATA_DIR"] = str(self.tmp)
        sys.argv = ["train_scene", "--strips", str(strips), "--no-panel", "--cams", "main", "--no-resume", *extra]
        out = io.StringIO()
        code = None
        try:
            with contextlib.redirect_stdout(out):
                try:
                    train_scene.main()
                except SystemExit as exc:
                    code = exc.code
        finally:
            (exp_ssl.default_ssl_stem, exp_ssl.ssl_rows_from_split, train_scene.train_directory, data_root.free_bytes,
             old_dir, sys.argv) = saved
            if old_dir is None:
                os.environ.pop("GVD_DATA_DIR", None)
            else:
                os.environ["GVD_DATA_DIR"] = old_dir
        return code, out.getvalue()


def check_main_passes_ssl_stem() -> None:
    """main() hands train_directory the SSL stem when SSL is on, and None / the user's --init-stem when off."""
    torch = _torch()
    if torch is None:
        skip_check("check_main_passes_ssl_stem", "torch not installed (train_scene.main needs it)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        h = _MainHarness(Path(tmp))
        code, text = h.run([])
        assert code is None, (code, text)
        assert h.ssl_calls == 1 and h.kw["init_stem"] == str(h.stem), (h.ssl_calls, h.kw.get("init_stem"), text)
        assert h.kw["ssl_meta"]["on"] is True and h.kw["ssl_meta"]["key"] == "feedface00000001", h.kw["ssl_meta"]
        h = _MainHarness(Path(tmp))
        code, text = h.run(["--no-ssl-pretrain"])
        assert code is None and h.ssl_calls == 0 and h.kw["init_stem"] is None, (code, h.ssl_calls, h.kw.get("init_stem"), text)
        assert h.kw["ssl_meta"]["on"] is False and "--no-ssl-pretrain" in h.kw["ssl_meta"]["why"], h.kw["ssl_meta"]
        for extra in (["--init-stem", "mine.npz"], ["--no-ssl-pretrain", "--init-stem", "mine.npz"]):
            h = _MainHarness(Path(tmp))
            code, text = h.run(extra)
            assert code is None and h.ssl_calls == 0 and h.kw["init_stem"] == "mine.npz", (extra, code, h.ssl_calls, h.kw.get("init_stem"), text)
            assert h.kw["ssl_meta"]["on"] is False and h.kw["ssl_meta"]["stem"] == "mine.npz", h.kw["ssl_meta"]
    print("main: SSL on -> train_directory init_stem is the SSL stem (+ssl_meta key); --no-ssl-pretrain -> None; user --init-stem kept, SSL not run OK")


def check_ssl_loss_target_is_ema_copy() -> None:
    """The pretext loss gets the EMA copy as target, and that copy follows target <- 0.996 target + 0.004 online."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_loss_target_is_ema_copy", "torch not installed")
        return
    import python.train.exp_ssl as exp_ssl
    from python.train.exp_ssl import masked_token_pretext_loss, run_ssl_pretrain
    from python.train.scene_net import SceneNet

    torch.manual_seed(0)
    net = SceneNet(cams=("main",), scale=SCALE)
    images = _fixture_frames(int(net.canvas_h), int(net.canvas_w), 6)
    rows = [{"i": i, "kind": "train"} for i in range(len(images))]
    initial = [p.detach().clone() for p in net.stem.parameters()]
    expected = [p.clone().double() for p in initial]
    seen: dict = {"targets": [], "calls": 0, "steps": 0, "moved": 0.0}
    real_loss = exp_ssl.masked_token_pretext_loss

    def spy(n, image, **kw):
        seen["calls"] += 1
        assert kw.get("target") is not None, "the loss was called without the EMA target"
        seen["targets"].append(kw["target"])
        assert kw["target"] is not n and kw["target"].stem is not n.stem, "target must be a copy, not the online net"
        assert not any(p.requires_grad for p in kw["target"].parameters()), "EMA target must not train"
        return real_loss(n, image, **kw)

    def after_step(online):
        # called after opt.step() and ema_update(): the target must now be the 0.996 recurrence on the post-step online weights
        seen["steps"] += 1
        target = seen["targets"][-1]
        for exp, o in zip(expected, online.stem.parameters()):
            exp.mul_(0.996).add_(o.detach().double(), alpha=0.004)
        worst = max(float((e - t.detach().double()).abs().max()) for e, t in zip(expected, target.stem.parameters()))
        assert worst < 1e-6, f"step {seen['steps']}: EMA target off the 0.996 recurrence by {worst:.3e}"
        seen["moved"] = max(float((t.detach() - i).abs().max()) for t, i in zip(target.stem.parameters(), initial))

    exp_ssl.masked_token_pretext_loss = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_ssl_pretrain(net, rows, image_fn=lambda r: images[int(r["i"])], out_path=Path(tmp) / "stem.npz",
                             lr=0.05, seed=0, epochs=2, _after_step=after_step)
    finally:
        exp_ssl.masked_token_pretext_loss = real_loss
    assert seen["calls"] == seen["steps"] == 12, seen["calls"]
    assert all(t is seen["targets"][0] for t in seen["targets"]), "the target is one EMA copy for the whole run"
    assert seen["moved"] > 1e-4, f"the EMA target never moved off the initial weights ({seen['moved']:.2e}): recurrence untested"
    # The loss itself reads the full-canvas token from the target, not from the online net.
    online, other = SceneNet(cams=("main",), scale=SCALE), SceneNet(cams=("main",), scale=SCALE)
    img = images[0]
    got = masked_token_pretext_loss(online, img, rng=np.random.RandomState(5), target=other)
    plain = masked_token_pretext_loss(online, img, rng=np.random.RandomState(5))
    masked = img.clone()
    rng = np.random.RandomState(5)
    h, w = masked.shape[2:]
    mh, mw = max(1, int(h * 0.25)), max(1, int(w * 0.25))
    y0, x0 = int(rng.randint(0, h - mh + 1)), int(rng.randint(0, w - mw + 1))
    masked[:, :, y0:y0 + mh, x0:x0 + mw] = 0
    with torch.no_grad():
        want = torch.mean(torch.abs(online._encode(masked)[0] - other._encode(img)[0]))
    assert torch.allclose(got, want, atol=1e-7) and not torch.allclose(got, plain, atol=1e-7), (float(got), float(want), float(plain))
    print(f"ssl loss target: 12 steps, one frozen EMA copy, recurrence 0.996/0.004 held to <1e-6 (moved {seen['moved']:.1e} from init); loss reads target token OK")


def _short_recording(base: Path, n_rows: int) -> Path:
    import test_pause_dir

    strips = base / "data" / "scene_x"
    strips.mkdir(parents=True)
    old = test_pause_dir.N_ROWS
    test_pause_dir.N_ROWS = n_rows
    try:
        test_pause_dir._fixture(strips)
    finally:
        test_pause_dir.N_ROWS = old
    return strips


def check_ssl_empty_split_refuses_when_trainer_trains() -> None:
    """SSL has 0 train rows but the trainer's own plan has some -> exit 2 naming --no-ssl-pretrain, no silent SSL-off.

    When the trainer's plan has 0 train rows too, SSL steps aside and the trainer reports.
    """
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_empty_split_refuses_when_trainer_trains", "torch not installed")
        return
    import contextlib
    import io
    import os

    import python.data.data_root as data_root
    import python.train.exp_ssl as exp_ssl
    from python.train import train_scene

    rec = {"empty_split": True}
    train_plan = [{"kind": "train", "rows": [{"i": 0}, {"i": 1}]}, {"kind": "holdout", "rows": [{"i": 2}]}]
    msg = train_scene.ssl_empty_split_refusal(rec, train_plan)
    assert msg and "--no-ssl-pretrain" in msg and "2 train rows" in msg, msg
    assert train_scene.ssl_empty_split_refusal(rec, [{"kind": "holdout", "rows": [{"i": 2}]}]) is None, "0 trainer rows: the trainer reports it"
    assert train_scene.ssl_empty_split_refusal(rec, []) is None
    assert train_scene.ssl_empty_split_refusal({"empty_split": False}, train_plan) is None

    def run_cli(strips: Path, base: Path, *, no_rows_for_ssl: bool) -> tuple[int | None, str]:
        old = (data_root.free_bytes, os.environ.get("GVD_DATA_DIR"), sys.argv, exp_ssl.ssl_rows_from_split)
        data_root.free_bytes = lambda _path: 500 * 1024 ** 3
        os.environ["GVD_DATA_DIR"] = str(base)
        if no_rows_for_ssl:
            exp_ssl.ssl_rows_from_split = lambda rows, meta, holdout_blocks=None: []
        sys.argv = ["train_scene", "--strips", str(strips), "--no-panel", "--cams", "main", "--no-resume", "--scale", str(SCALE),
                    "--workers", "0", "--epochs", "1", "--lr", "1e-5", "--ckpt-dir", str(base / "ck")]
        out, code = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out):
                try:
                    train_scene.main()
                except SystemExit as exc:
                    code = exc.code
        finally:
            data_root.free_bytes, prev_dir, sys.argv, exp_ssl.ssl_rows_from_split = old
            if prev_dir is None:
                os.environ.pop("GVD_DATA_DIR", None)
            else:
                os.environ["GVD_DATA_DIR"] = prev_dir
        return code, out.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _short_recording(base, 120)  # the trainer's plan has train rows here (56)
        code, text = run_cli(strips, base, no_rows_for_ssl=True)
        assert code == 2, (code, text)
        assert "--no-ssl-pretrain" in text and "default SSL pretraining has 0 train rows" in text and "56 train rows" in text, text
        assert "scene schedule" not in text and not list((base / "ck").glob("*.npz")), "training must not have started"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _short_recording(base, 30)  # 3 s: SSL 0 rows and the trainer plan 0 train rows too
        code, text = run_cli(strips, base, no_rows_for_ssl=False)
        assert code is None and "0 train rows; no stem" in text, (code, text)
        assert "default SSL pretraining has 0 train rows" not in text, "no refusal when the trainer has nothing to train either"
    print("empty SSL split + trainer plan with 56 train rows -> exit 2 naming --no-ssl-pretrain before training; both 0 -> SSL steps aside OK")


def check_ssl_recorded_in_stats_and_ckpt() -> None:
    """SSL on/off, why and the stem key are in the stats dict and in every epoch ckpt meta."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_recorded_in_stats_and_ckpt", "torch not installed")
        return
    import contextlib
    import io

    from python.train.epoch_ckpt import load_epoch_ckpt
    from python.train.train_scene import train_directory

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _short_recording(base, 60)
        runs = {
            "on": {"on": True, "why": "on (default)", "key": "feedface00000001", "rows": 7, "reused": True, "empty_split": False},
            "off": None,
        }
        for name, meta in runs.items():
            ck = base / f"ck_{name}"
            with contextlib.redirect_stdout(io.StringIO()):
                stats = train_directory(strips, epochs=1, lr=1e-5, capability=None, export_path=None, free_vram_bytes=0,
                                        free_ram_bytes=8_000_000_000, resume=False, open_panel=False, autostart=True,
                                        cams=("main",), heads=("lane_gt",), augment=False, workers=0, min_valid=0.0,
                                        holdout_blocks=1, scale=SCALE, status_root=base / f"st_{name}", ckpt_dir=ck,
                                        exact_epochs=1, ssl_meta=meta)
            files = sorted(ck.glob("*.npz"))
            assert files, f"{name}: no epoch checkpoint written"
            for rec in [stats["ssl"]] + [load_epoch_ckpt(f)["meta"]["ssl"] for f in files]:
                if meta is None:
                    assert rec["on"] is False and rec["key"] is None, rec
                else:
                    assert rec["on"] is True and rec["key"] == "feedface00000001" and rec["rows"] == 7 and rec["reused"] is True, rec
    print("ssl record: stats['ssl'] and every ckpt meta['ssl'] carry on/why/key/rows/reused (off when not requested) OK")


def check_ssl_oom_exits_2_and_releases_cuda() -> None:
    """GPU OOM in the SSL stage -> exit 2 naming --no-ssl-pretrain; the CUDA cache is emptied on success and on OOM."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_oom_exits_2_and_releases_cuda", "torch not installed")
        return
    import python.train.exp_ssl as exp_ssl
    from python.train import train_scene
    import python.train.scene_net as scene_net

    oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    # main(): OOM -> exit 2, message names the opt-out, train_directory never runs.
    with tempfile.TemporaryDirectory() as tmp:
        h = _MainHarness(Path(tmp), ssl_error=oom)
        code, text = h.run([])
        assert code == 2 and not h.kw and "--no-ssl-pretrain" in text and "out of GPU memory" in text, (code, h.kw, text)
        h = _MainHarness(Path(tmp), ssl_error=RuntimeError("boom: not memory"))
        try:
            h.run([])
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("a non-OOM RuntimeError must not be turned into a skip")
    # Stage: the cache is released after the stage and before train_directory could read free VRAM.
    class FakeNet:
        def __init__(self, *a, **k):
            pass

        def to(self, _device):
            return self

        def train(self):
            return self

    calls: list[str] = []
    saved = (torch.cuda.is_available, torch.cuda.empty_cache, scene_net.SceneNet, exp_ssl.default_ssl_stem, train_scene._ssl_train_rows_and_reader)
    torch.cuda.is_available = lambda: True
    torch.cuda.empty_cache = lambda: calls.append("empty_cache")
    scene_net.SceneNet = FakeNet
    train_scene._ssl_train_rows_and_reader = lambda *a, **k: ([{"i": 0, "kind": "train"}], lambda row: None)
    args = SimpleNamespace(holdout_blocks=None, scale=SCALE, lr=1e-3, seed=0, ssl_epochs=1, ssl_max_rows=10)
    try:
        for mode in ("ok", "oom"):
            calls.clear()

            def stem(net, rows, **kw):
                calls.append("ssl")
                if mode == "oom":
                    raise oom
                return Path("stem.npz"), {"key": "k", "rows": 1, "reused": True}

            exp_ssl.default_ssl_stem = stem
            with tempfile.TemporaryDirectory() as tmp:
                import os

                old = os.environ.get("GVD_DATA_DIR")
                os.environ["GVD_DATA_DIR"] = tmp
                try:
                    if mode == "ok":
                        path, meta = train_scene._default_ssl_stage(args, [Path(tmp)], Path(tmp), ("main",), {})
                        assert path == "stem.npz" and meta["key"] == "k" and meta["on"] is True, (path, meta)
                    else:
                        try:
                            train_scene._default_ssl_stage(args, [Path(tmp)], Path(tmp), ("main",), {})
                        except exp_ssl.SslError as exc:
                            assert "out of GPU memory" in str(exc), exc
                        else:
                            raise AssertionError("OOM not turned into SslError")
                finally:
                    if old is None:
                        os.environ.pop("GVD_DATA_DIR", None)
                    else:
                        os.environ["GVD_DATA_DIR"] = old
            assert calls == ["ssl", "empty_cache"], (mode, calls)
    finally:
        torch.cuda.is_available, torch.cuda.empty_cache, scene_net.SceneNet, exp_ssl.default_ssl_stem, train_scene._ssl_train_rows_and_reader = saved
    calls.clear()
    torch.cuda.empty_cache = lambda: calls.append("empty_cache")
    try:
        train_scene._release_ssl_cuda(torch.device("cpu"))
    finally:
        torch.cuda.empty_cache = saved[1]
    assert calls == [], "a CPU stage has no CUDA cache to release"
    print("ssl OOM: main exits 2 naming --no-ssl-pretrain (train_directory not run); empty_cache runs after the stage on ok and OOM; non-OOM errors propagate OK")


def check_ssl_stem_key_tracks_recipe_and_content() -> None:
    """The stem cache key changes with SSL_RECIPE and with frame pixels (same row ids), and per-source hashes are in it."""
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_stem_key_tracks_recipe_and_content", "torch not installed")
        return
    import python.train.exp_ssl as exp_ssl
    from python.train.exp_ssl import content_hashes, default_ssl_stem
    from python.train.scene_net import SceneNet

    probe = SceneNet(cams=("main",), scale=SCALE)
    frames = _fixture_frames(int(probe.canvas_h), int(probe.canvas_w), 6)
    rows = [{"i": i, "kind": "train", "source": "a" if i < 3 else "b"} for i in range(6)]
    reads = {"n": 0}

    def reader(frame_list):
        def image_fn(row):
            reads["n"] += 1
            return frame_list[int(row["i"])]
        return image_fn

    def run(out, image_fn):
        return default_ssl_stem(SceneNet(cams=("main",), scale=SCALE), rows, image_fn=image_fn, out_dir=out, lr=1e-3, seed=0,
                                epochs=1, max_rows=0, key_extra={})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        path, info = run(out, reader(frames))
        again, info2 = run(out, reader(frames))
        assert again == path and info2["reused"] is True, "same recipe + same pixels must reuse"
        # one pixel of one frame of source "b" changes (a re-export in place): new key, old stem kept
        edited = [f.clone() for f in frames]
        edited[4][0, 0, 0, 0] = 1.0 - float(edited[4][0, 0, 0, 0]) if float(edited[4][0, 0, 0, 0]) != 0.5 else 0.25
        path_px, info_px = run(out, reader(edited))
        assert path_px != path and info_px["reused"] is False and path.is_file(), "changed pixels must not reuse the old stem"
        old_recipe = exp_ssl.SSL_RECIPE
        exp_ssl.SSL_RECIPE = old_recipe + 1
        try:
            path_rec, info_rec = run(out, reader(frames))
        finally:
            exp_ssl.SSL_RECIPE = old_recipe
        assert path_rec not in (path, path_px) and info_rec["reused"] is False, "a new SSL_RECIPE must not reuse the old stem"
        assert path.is_file() and path_px.is_file(), "old stems are kept"
    base, moved = content_hashes(rows, reader(frames)), content_hashes(rows, reader(edited))
    assert sorted(base) == ["a", "b"] and base["a"] == moved["a"] and base["b"] != moved["b"], (base, moved)
    print("ssl key: SSL_RECIPE bump and a one-pixel change in one source each give a new stem (old kept); per-source hashes differ only for the edited source OK")


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
              check_ssl_empty_split_defers_to_trainer, check_main_passes_ssl_stem, check_ssl_loss_target_is_ema_copy,
              check_ssl_empty_split_refuses_when_trainer_trains, check_ssl_recorded_in_stats_and_ckpt,
              check_ssl_oom_exits_2_and_releases_cuda, check_ssl_stem_key_tracks_recipe_and_content,
              check_recipe_off_is_bit_identical)
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
