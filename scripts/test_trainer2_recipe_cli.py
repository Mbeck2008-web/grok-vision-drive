#!/usr/bin/env python3
"""Trainer 2.0 PR 6: one real ``train_scene --recipe trainer2`` CLI run through stages 0, 1, 2, 7 (CPU, fixture).

Separate from ``test_trainer2_recipe.py`` because it really trains four stages (minutes on a CI runner).
Nothing here is mocked: ``main`` parses the command line, pre-registers the split + run manifests in
``--ckpt-dir``, runs SSL (stage 0, pretext-loss plateau), stage 1 (depth AbsRel / sem mIoU), stage 2 (8 m lane),
stage 7 (joint, with ctl), gates every stage against the earlier metrics, then exports the camera-only deploy graph
of the last checkpoint and runs the preset-bound deploy check.

``check_cli_full_curriculum_stages_0_1_2_7`` runs ``--stage 0,1,2,7`` on the plain fixture (no stop events).
``check_cli_default_plan_trains_through_2b`` runs the DEFAULT plan (0, 1, 2, 2b, 7) on ``stage2b_fixture`` (stop signs and
red lights in the held-out segment) and ``check_cli_default_plan_refuses_without_stop_events`` shows the default plan
stopping before anything trains when the recording cannot score 2b.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

import test_trainer2_recipe as base_tests  # noqa: E402


def _ready(name) -> bool:
    if base_tests._torch() is None:
        skip_check(name, "torch not installed")
        return False
    try:
        import cv2  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        skip_check(name, "cv2 / onnx not installed (CI installs both)")
        return False
    import shutil

    if shutil.which("git") is None:
        skip_check(name, "git not installed (the run manifest records the commit)")
        return False
    return True


def _main(argv, base: Path):
    import python.data.data_root as data_root
    from python.train import train_scene

    saved = (data_root.free_bytes, os.environ.get("GVD_DATA_DIR"))
    data_root.free_bytes = lambda _p: 500 * 1024 ** 3
    os.environ["GVD_DATA_DIR"] = str(base)
    out = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            train_scene.main(argv)
    except SystemExit as exc:
        code = exc.code
    finally:
        data_root.free_bytes = saved[0]
        if saved[1] is None:
            os.environ.pop("GVD_DATA_DIR", None)
        else:
            os.environ["GVD_DATA_DIR"] = saved[1]
    return code, out.getvalue()


# A toy net on a 14-frame fixture moves its holdout metrics by noise (sem mIoU is ~1e-3 here, so a 5% relative move is
# a rounding-level change): the real, unchanged non-regression gate then rejects a stage on one run and not on the next
# (seen on CI and on the box, same code). The full-curriculum run therefore uses a negligible lr so the held-out metrics
# are the same from stage to stage and the gate decision cannot depend on thread / build noise. Nothing is mocked: every
# stage still trains, measures and exits through the real code (the training signal itself is covered by
# check_recipe_stages_train_on_fixture, which asserts real gradients at the default lr).
FULL_RUN_LR = "1e-9"


def _flags(strips, ck, stage, *extra):
    return ["--strips", str(strips), "--recipe", "trainer2", "--no-augment", "--cams", "main", "--no-panel", "--no-resume",
            "--ckpt-dir", str(ck), "--min-valid", "0", "--workers", "0", "--ssl-max-rows", "16", "--stage", stage, *extra]


def check_cli_full_curriculum_stages_0_1_2_7() -> None:
    if not _ready("check_cli_full_curriculum_stages_0_1_2_7"):
        return
    import onnx
    from onnx import numpy_helper

    from python.train import trainer2_recipe as rec
    from python.train.epoch_ckpt import load_epoch_ckpt
    from python.train.trainer2_split import require_preregistered

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = base_tests._fixture(base)
        ck = base / "ck"
        ck.mkdir()
        old = ck / "pre-existing.ckpt"
        old.write_bytes(b"an older run's checkpoint")
        export = base / "student.onnx"
        # the stage-exit rule actually used by the in-training loop, per stage (spy on the real function)
        seen: dict[str, list] = {}
        real_exit = rec.stage_exited_all

        def spy(stage_id, evals_by, *, epochs):
            seen.setdefault(stage_id, []).append(({k: list(v) for k, v in evals_by.items()}, int(epochs)))
            return real_exit(stage_id, evals_by, epochs=epochs)

        rec.stage_exited_all = spy
        try:
            code, text = _main(_flags(strips, ck, "0,1,2,7", "--export", str(export), "--lr", FULL_RUN_LR), base)
        finally:
            rec.stage_exited_all = real_exit
        assert code is None, (code, text[-2500:])
        assert "Traceback" not in text and "MetricNotMeasured" not in text, text[-2500:]
        # the manifests were written by this run before any stage trained (no hand pre-registration here)
        assert "run manifest pre-registered now" in text and text.index("pre-registered now") < text.index("stage 0 SSL"), text[:1500]
        run = require_preregistered(ck)
        assert run["ssl_pretrain"] is True and run["config"]["recipe"] == "trainer2" and (ck / "split_manifest.json").is_file()
        assert len(run["git_sha"]) >= 7
        # every stage exited on a measured metric and reported it
        exits = {m[0]: (int(m[1]), json.loads(m[2])) for m in re.findall(
            r"recipe trainer2: stage (\d+) \([^)]*\) exited after (\d+) epoch\(s\), ckpt=\S+ metrics=(\{.*\})", text)}
        assert sorted(exits) == ["0", "1", "2", "7"], (exits, text[-2000:])
        s0, s1, s2, s7 = (exits[k][1] for k in ("0", "1", "2", "7"))
        assert set(s0) == {"ssl_pretext"} and np.isfinite(s0["ssl_pretext"]) and s0["ssl_pretext"] > 0, s0
        assert set(s1) == {"depth_absrel", "sem_miou"}, s1
        assert 0.0 < s1["depth_absrel"] and 0.0 <= s1["sem_miou"] <= 1.0, s1
        for name, m in (("2", s2), ("7", s7)):
            assert set(m) == {"lane_rms", "lane_r", "depth_absrel", "sem_miou"}, (name, m)  # earlier aux metrics re-measured
            assert all(np.isfinite(v) for v in m.values()), (name, m)
        # the exit rule ran on real series (the in-training loop and the curriculum gate both call it): stage 1 on
        # both of its metrics, and each stage ended at a plateau or the cap
        assert sorted(seen) == ["0", "1", "2", "7"], sorted(seen)
        assert len(seen["2"]) >= 2 and len(seen["1"]) >= 2, {k: len(v) for k, v in seen.items()}  # per-epoch calls + the gate
        for sid, calls in seen.items():
            series, epochs = calls[-1]
            assert set(series) == set(rec.STAGE_PRIMARY_METRICS[sid]) and all(np.isfinite(v).all() for v in series.values()), (sid, series)
            assert epochs == exits[sid][0] and epochs <= rec.STAGE_MAX_EPOCHS, (sid, epochs, exits[sid])
            assert real_exit(sid, series, epochs=epochs), sid
        assert exits["0"][0] <= rec.STAGE_MAX_EPOCHS and len(seen["0"]) == 1  # stage 0 exits inside exp_ssl; the gate re-checks
        # SSL stage 0: the pretext loss history is on disk beside the stem and is what the exit used
        side = list((base / "cache" / "ssl").glob("ssl_stem_*.losses.json"))
        assert len(side) == 1, list((base / "cache").rglob("*"))
        hist = json.loads(side[0].read_text())["epoch_losses"]
        assert len(hist) == exits["0"][0] and abs(hist[-1] - s0["ssl_pretext"]) < 1e-5, (hist, s0)
        # the export: camera-only deploy graph of the LAST checkpoint, one file, deploy check passed
        assert export.is_file() and sorted(p.name for p in base.glob("student*")) == ["student.onnx"]
        assert "recipe trainer2: deploy check {'ok': True, 'preset': 'small'" in text, text[-1500:]
        model = onnx.load(str(export))
        props = {p.key: p.value for p in model.metadata_props}
        last = re.search(r"trained=\['0', '1', '2', '7'\] skipped=\[\] ckpt=(\S+)", text).group(1)
        assert props["gvd_ckpt_file"] == Path(last).name, (props.get("gvd_ckpt_file"), last)
        assert not any("ctl_head" in i.name for i in model.graph.initializer)
        assert not any("cond" in i.name.lower() for i in model.graph.input)
        meta7 = load_epoch_ckpt(last)["meta"]
        assert meta7["ctl"] is True and "ctl" in meta7["heads"], meta7  # stage 7 did train ctl; the export dropped it
        inits = {i.name.removeprefix("net."): numpy_helper.to_array(i) for i in model.graph.initializer}
        w = load_epoch_ckpt(last)["weights"]
        shared = [k for k in w if k in inits and inits[k].shape == np.shape(w[k])]
        assert len(shared) >= 10 and all(np.array_equal(inits[k], np.asarray(w[k])) for k in shared)
        # stage 2 and 7 keep the aux heads at weight 0: they get no gradient, so the heads are stage 1's own
        ck1 = re.search(r"stage 1 \([^)]*\) exited after \d+ epoch\(s\), ckpt=(\S+)", text).group(1)
        ck2 = re.search(r"stage 2 \([^)]*\) exited after \d+ epoch\(s\), ckpt=(\S+)", text).group(1)
        w1, w2 = load_epoch_ckpt(ck1)["weights"], load_epoch_ckpt(ck2)["weights"]
        heads = [k for k in w1 if k.startswith(("aux_depth_head.", "aux_sem_head."))]
        assert heads and all(np.array_equal(w1[k], w2[k]) for k in heads), "aux heads moved in a stage that gives them no loss"
        assert old.read_bytes() == b"an older run's checkpoint"
        # a second run into the same --export is refused up front and leaves the file alone
        before = export.read_bytes()
        code2, text2 = _main(_flags(strips, base / "ck_b", "0,1,2,7", "--export", str(export), "--lr", FULL_RUN_LR), base)
        assert code2 == 2 and "create-only" in text2 and "Traceback" not in text2, (code2, text2[-800:])
        assert export.read_bytes() == before and not (base / "ck_b" / "run_manifest.json").exists()
    print("CLI --recipe trainer2 --stage 0,1,2,7: manifests pre-registered, stages 0/1/2/7 exit on measured metrics, "
          "gates pass, camera-only export + preset deploy check OK")


def check_cli_default_plan_refuses_without_stop_events() -> None:
    if not _ready("check_cli_default_plan_refuses_without_stop_events"):
        return
    import stage2b_fixture as fx

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        for label, strips, words in (
            ("plain fixture", base_tests._fixture(base), ("stage 2b", "sim_t")),
            ("signs.jsonl missing", fx.build(base / "bare", fx.N_ROWS, fx.intersections(fx.N_ROWS), with_signs=False, with_aux=True),
             ("stage 2b", "0 stop-sign/line", "vacuous")),
        ):
            ck = base / f"ck_{label.split()[0]}"
            ck.mkdir()
            code, text = _main(["--strips", str(strips), "--recipe", "trainer2", "--no-augment", "--cams", "main", "--no-panel",
                                "--no-resume", "--ckpt-dir", str(ck)], base)
            assert code == 2 and "Traceback" not in text, (label, code, text[-1500:])
            for word in (*words, "Nothing was trained", "--stage"):
                assert word in text, (label, word, text[-1500:])
            assert "ssl-pretrain:" not in text and not (ck / "run_manifest.json").exists(), (label, "must stop before SSL and the manifest")
            assert not list((base / "cache").rglob("ssl_stem_*")) and not list(ck.glob("*.npz")), label
    print("CLI default plan: stops before SSL / manifest / any stage when the recording cannot score stage 2b OK")


def check_cli_default_plan_trains_through_2b() -> None:
    """The default plan (no --stage) trains 0, 1, 2, 2b, 7 end to end and gates every stage, 2b on its stop metrics."""
    if not _ready("check_cli_default_plan_trains_through_2b"):
        return
    import stage2b_fixture as fx

    from python.train import trainer2_recipe as rec

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = fx.build(base / "stops", fx.N_ROWS, fx.intersections(fx.N_ROWS), with_aux=True)
        ck = base / "ck"
        ck.mkdir()
        seen: dict[str, list] = {}
        real_exit = rec.stage_exited_all

        def spy(stage_id, evals_by, *, epochs):
            seen.setdefault(stage_id, []).append(({k: list(v) for k, v in evals_by.items()}, int(epochs)))
            return real_exit(stage_id, evals_by, epochs=epochs)

        rec.stage_exited_all = spy
        try:
            code, text = _main(_flags(strips, ck, "", "--lr", FULL_RUN_LR), base)
        finally:
            rec.stage_exited_all = real_exit
        assert code is None, (code, text[-3000:])
        assert "Traceback" not in text and "MetricNotMeasured" not in text and "StopNotMeasured" not in text, text[-3000:]
        # the 2b preflight ran before the manifest was written and before stage 0 trained
        assert "stage 2b preflight OK" in text and text.index("stage 2b preflight OK") < text.index("pre-registered now") < text.index("stage 0 SSL"), text[:2500]
        exits = {m[0]: (int(m[1]), json.loads(m[2])) for m in re.findall(
            r"recipe trainer2: stage (\w+) \([^)]*\) exited after (\d+) epoch\(s\), ckpt=\S+ metrics=(\{.*\})", text)}
        assert sorted(exits) == ["0", "1", "2", "2b", "7"], (sorted(exits), text[-2500:])
        assert "trained=['0', '1', '2', '2b', '7'] skipped=[]" in text, text[-1500:]
        s2b, s7 = exits["2b"][1], exits["7"][1]
        for name, m in (("2b", s2b), ("7", s7)):
            assert set(m) == {"lane_rms", "lane_r", "depth_absrel", "sem_miou", *rec.STOP_METRICS}, (name, m)
            assert all(np.isfinite(v) for v in m.values()), (name, m)
        assert all(s2b[k] >= 0.0 for k in rec.STOP_METRICS), s2b
        # 2b exited through the real rule on its three stop metrics
        series, epochs = seen["2b"][-1]
        assert set(series) == set(rec.STAGE_PRIMARY_METRICS["2b"]) == set(rec.STOP_METRICS), series
        assert all(np.isfinite(v).all() for v in series.values()) and epochs == exits["2b"][0] <= rec.STAGE_MAX_EPOCHS, (series, epochs)
        assert real_exit("2b", series, epochs=epochs)
        # the motion map was fitted on train rows only, once per stop stage, with no holdout id in the fit set
        fits = re.findall(r"stop motion fitted on (\d+) train row pairs.*\(holdout ids in fit set: 0\)", text)
        assert len(fits) == 2 and all(int(n) < fx.SPLIT for n in fits), fits  # stage 2b and stage 7
    print("CLI default plan: 0 / 1 / 2 / 2b / 7 train end to end, 2b exits on missed_stop / missed_light / false_brake, "
          "stage 7 re-measures them OK")


def check_cli_stage_errors_exit_2_without_traceback() -> None:
    """An SSL / data error raised inside the recipe path is a clean exit 2 (it used to be a traceback)."""
    if not _ready("check_cli_stage_errors_exit_2_without_traceback"):
        return
    from python.data.raw_frames import LossyFramesError
    from python.train import train_scene
    from python.train.exp_ssl import SslError
    from python.train.trainer2_split import SplitError

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = base_tests._fixture(base)
        saved = train_scene._run_trainer2_recipe
        try:
            for exc in (SslError("ssl boom"), LossyFramesError("lossy boom"), SplitError("split boom")):
                def boom(*a, _e=exc, **k):
                    raise _e

                train_scene._run_trainer2_recipe = boom
                code, text = _main(_flags(strips, base / "ck", "0,1,2,7"), base)
                assert code == 2 and "Traceback" not in text and str(exc) in text, (type(exc).__name__, code, text[-600:])
        finally:
            train_scene._run_trainer2_recipe = saved
    print("CLI: SslError / LossyFramesError / SplitError inside the recipe path exit 2 with the message, no traceback OK")


def main() -> int:
    checks = (check_cli_default_plan_refuses_without_stop_events, check_cli_stage_errors_exit_2_without_traceback,
              check_cli_full_curriculum_stages_0_1_2_7, check_cli_default_plan_trains_through_2b)
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
