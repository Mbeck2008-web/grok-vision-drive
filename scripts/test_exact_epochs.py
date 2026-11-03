#!/usr/bin/env python3
"""``--exact-epochs N`` (control_head spec sec. 14): a fixed epoch count, early stop off.

* flag fixture: lr 1e-5 keeps the holdout rms flat, so the default run stops at
  epoch 2 (``holdout stopped``). With ``exact_epochs=4`` the same fixture writes
  ``ckpt_ep0..ckpt_ep3`` and no ``ckpt_ep4``, logs the exact-epochs line once and
  never "holdout stopped"; the final weights are the ``ckpt_ep3`` weights (no
  ``best_blob`` reload). The schedule line names ``--exact-epochs 4``.
* records: ``exact_epochs`` is 4 in every ``ckpt_ep*`` meta, the pause file, the
  metrics val records and the stats dict; null without the flag.
* scheduler: ``steps_per_epoch`` and each ``ckpt_ep{E}_step{S}`` step match the
  default run for every epoch both reach. ``remaining_until_exact_end`` units.
* pause / resume: a flag run paused in epoch 2 resumes to ``ckpt_ep3`` (no
  ``ckpt_ep4``). Resuming it with ``--exact-epochs 5``, without the flag, or from
  a pause file with the key removed exits 2 and writes nothing.
* refusals: ``--exact-epochs 0``, ``--exact-epochs 4 --dry-run`` and
  ``--exact-epochs 4 --sanity`` exit 2 and write nothing.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

FLAT_LR = 1e-5
EXIT_LINE = "[GVD] exact epochs: stopped after epoch={last} (early stop off)"


def _fixture(base: Path) -> Path:
    from test_pause_dir import _fixture as lane_fixture

    strips = base / "data" / "scene_x"
    strips.mkdir(parents=True)
    lane_fixture(strips)
    return strips


def _listing(root: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            out[str(path.relative_to(root))] = (len(data), hashlib.sha256(data).hexdigest())
        else:
            out[str(path.relative_to(root)) + "/"] = (0, "")
    return out


def _ckpts(folder: Path) -> list[dict]:
    from python.train.epoch_ckpt import load_epoch_ckpt, parse_ckpt_name

    out = []
    for path in sorted(Path(folder).glob("*.npz")):
        parsed = parse_ckpt_name(path)
        if parsed is None:
            continue
        parsed["path"] = path
        parsed["meta"] = load_epoch_ckpt(path)["meta"]
        out.append(parsed)
    return sorted(out, key=lambda item: item["epoch"])


@contextlib.contextmanager
def _capture_export():
    """Record the net's weights at the final --export call (nothing is written)."""
    from python.train import train_scene

    seen: dict[str, dict] = {}
    old = train_scene.export_scene_onnx

    def _grab(path, net=None, **kw):
        if net is not None:
            seen["weights"] = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        return path

    train_scene.export_scene_onnx = _grab
    try:
        yield seen
    finally:
        train_scene.export_scene_onnx = old


def _run(strips: Path, ckpt_dir: Path, *, exact_epochs=None, resume=False, pause_dir=None, view=None, export=None):
    from python.train.train_scene import train_directory

    kw = {} if exact_epochs is None else {"exact_epochs": exact_epochs}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stats = train_directory(
            strips,
            epochs=1,
            lr=FLAT_LR,
            capability=None,
            export_path=export,
            free_vram_bytes=0,
            free_ram_bytes=8_000_000_000,
            resume=resume,
            open_panel=False,
            autostart=True,
            view=view,
            cams=("main",),
            heads=("lane_gt",),
            augment=False,
            workers=0,
            min_valid=0.0,
            holdout_blocks=1,
            status_root=ckpt_dir.parent / (ckpt_dir.name + "_status"),
            ckpt_dir=ckpt_dir,
            pause_dir=pause_dir,
            **kw,
        )
    return stats, out.getvalue()


def _metrics(ckpt_dir: Path) -> list[dict]:
    status = ckpt_dir.parent / (ckpt_dir.name + "_status")
    rows = []
    for path in sorted(status.rglob("metrics.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [row for row in rows if row.get("kind") == "val"]


def _schedule(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("[GVD] scene schedule")]


def _main_code(argv: list[str]) -> tuple[int | None, str]:
    from python.train import train_scene

    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            train_scene.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue()


def _cli(strips: Path, *extra: str) -> list[str]:
    return [
        "--strips", str(strips), "--cams", "main", "--heads", "lane_gt", "--lr", str(FLAT_LR),
        "--no-panel", "--autostart", "--workers", "0", "--min-valid", "0", "--holdout-blocks", "1",
        *extra,
    ]


def check_remaining_until_exact_end() -> None:
    from python.train.continual import remaining_until_exact_end

    assert remaining_until_exact_end(5, 13, epoch=0, exact_epochs=4) == 5 + 3 * 13
    assert remaining_until_exact_end(13, 13, epoch=0, exact_epochs=1) == 13
    assert remaining_until_exact_end(5, 13, epoch=3, exact_epochs=4) == 5
    assert remaining_until_exact_end(0, 13, epoch=3, exact_epochs=4) == 0
    assert remaining_until_exact_end(5, 13, epoch=7, exact_epochs=4) == 5
    print("remaining_until_exact_end: epoch 0 and epoch N-1 OK")


def check_parser() -> None:
    from python.train.train_scene import build_parser

    assert build_parser().parse_args([]).exact_epochs is None
    assert build_parser().parse_args(["--exact-epochs", "4"]).exact_epochs == 4
    print("cli: --exact-epochs parses, default off OK")


def check_refusals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _fixture(base)
        ck = base / "run" / "ckpts"
        cases = {
            "N=0": ["--exact-epochs", "0"],
            "N=-1": ["--exact-epochs", "-1"],
            "--dry-run": ["--exact-epochs", "4", "--dry-run"],
            "--sanity": ["--exact-epochs", "4", "--sanity"],
        }
        for label, extra in cases.items():
            before = _listing(base)
            code, text = _main_code(_cli(strips, "--ckpt-dir", str(ck), *extra))
            after = _listing(base)
            assert code == 2, (label, code, text[-2000:])
            assert "--exact-epochs" in text, (label, text[-2000:])
            assert before == after, (label, set(after) ^ set(before))
            print(f"refused ({label}): exit 2, nothing written OK")


def check_fixture_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _fixture(base)
        # Default: the flat rms stops the run at epoch 2, as today.
        ck_a = base / "a" / "ckpts"
        stats_a, text_a = _run(strips, ck_a)
        got_a = _ckpts(ck_a)
        assert [c["epoch"] for c in got_a] == [0, 1, 2], [c["path"].name for c in got_a]
        assert "[GVD] holdout stopped" in text_a
        assert "exact epochs:" not in text_a
        assert all(c["meta"].get("exact_epochs", "absent") is None for c in got_a), [c["meta"] for c in got_a]
        assert stats_a.get("exact_epochs", "absent") is None, stats_a
        vals_a = _metrics(ck_a)
        assert len(vals_a) == 3 and all(v.get("exact_epochs", "absent") is None for v in vals_a), vals_a
        sched_a = _schedule(text_a)
        assert len(sched_a) == 1 and "holdout stop ends the run" in sched_a[0], sched_a
        print(f"default: stops at epoch 2 ({[c['path'].name.split('.')[-2] for c in got_a]}), exact_epochs null OK")

        # Flag: the same fixture runs exactly 4 epochs.
        ck_b = base / "b" / "ckpts"
        with _capture_export() as seen:
            stats_b, text_b = _run(strips, ck_b, exact_epochs=4, export=base / "b" / "final.onnx")
        got_b = _ckpts(ck_b)
        assert [c["epoch"] for c in got_b] == [0, 1, 2, 3], [c["path"].name for c in got_b]
        assert "[GVD] holdout stopped" not in text_b, text_b[-3000:]
        assert text_b.count(EXIT_LINE.format(last=3)) == 1, text_b[-3000:]
        assert "exact epochs:" in text_b and text_b.count("exact epochs:") == 1
        sched_b = _schedule(text_b)
        assert len(sched_b) == 1 and "--exact-epochs 4 ends the run" in sched_b[0], sched_b
        assert "holdout stop ends the run" not in text_b
        assert all(c["meta"].get("exact_epochs") == 4 for c in got_b), [c["meta"].get("exact_epochs") for c in got_b]
        assert stats_b.get("exact_epochs") == 4, stats_b
        vals_b = _metrics(ck_b)
        assert len(vals_b) == 4 and all(v.get("exact_epochs") == 4 for v in vals_b), vals_b
        print("flag: ckpt_ep0..ckpt_ep3, no ckpt_ep4, exit line once, no holdout stop; exact_epochs=4 in meta/metrics/stats OK")

        # Final weights are ckpt_ep3 (no best_blob reload; the best rms here is epoch 0).
        from python.train.epoch_ckpt import load_epoch_ckpt

        assert "weights" in seen, "the final --export call did not see the net"
        ep3 = load_epoch_ckpt(got_b[-1]["path"])["weights"]
        ep0 = load_epoch_ckpt(got_b[0]["path"])["weights"]
        assert set(ep3) == set(seen["weights"])
        assert all(np.array_equal(ep3[k], seen["weights"][k]) for k in ep3), "final weights are not ckpt_ep3"
        assert any(not np.array_equal(ep0[k], ep3[k]) for k in ep3), "ep0 == ep3: the reload check cannot tell"
        print("flag: final weights == ckpt_ep3 (!= ckpt_ep0) OK")

        # Scheduler: same steps per epoch and the same ckpt steps for every epoch both reach.
        spe = lambda line: line.split("steps_per_epoch=")[1].split()[0]  # noqa: E731
        assert spe(sched_a[0]) == spe(sched_b[0]), (sched_a, sched_b)
        assert stats_a["steps_per_epoch"] == stats_b["steps_per_epoch"]
        for a, b in zip(got_a, got_b):
            assert (a["epoch"], a["step"]) == (b["epoch"], b["step"]), (a["path"].name, b["path"].name)
        print(f"scheduler: steps_per_epoch={spe(sched_a[0])} and ckpt steps {[c['step'] for c in got_a]} match OK")


@contextlib.contextmanager
def _pause_in_epoch(target: int, after_steps: int = 3):
    """A view that asks for a pause ``after_steps`` steps into ``target``; the pause then exits."""
    from python.train import epoch_ckpt as ec
    from python.train import train_scene
    from python.train.status_window import TrainWindow

    state = {"last": -1, "calls": 0, "fired": False}
    old_write, old_wait = ec.write_epoch_ckpt, train_scene._wait_for_start

    def _write(path, weights, meta, **kw):
        state["last"] = int(meta["epoch"])
        return old_write(path, weights, meta, **kw)

    view = TrainWindow()

    def _consume() -> bool:
        if state["fired"] or state["last"] != target - 1:
            return False
        state["calls"] += 1
        if state["calls"] > after_steps:
            state["fired"] = True
            return True
        return False

    view.consume_pause = _consume
    ec.write_epoch_ckpt = _write
    train_scene._wait_for_start = lambda _view: False
    try:
        yield view, state
    finally:
        ec.write_epoch_ckpt, train_scene._wait_for_start = old_write, old_wait


def check_pause_resume() -> None:
    from python.train.checkpoint import load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips = _fixture(base)
        ck = base / "run" / "ckpts"
        pause = base / "run" / "pause"
        with _pause_in_epoch(2) as (view, state):
            _, text = _run(strips, ck, exact_epochs=4, pause_dir=pause, view=view)
        assert state["fired"], text[-3000:]
        assert [c["epoch"] for c in _ckpts(ck)] == [0, 1], [c["path"].name for c in _ckpts(ck)]
        saved = load_checkpoint(strips, pause_dir=pause)
        assert saved is not None, "the pause did not leave a pause file"
        assert int(saved["epoch"]) == 2 and saved.get("exact_epochs") == 4, {k: saved.get(k) for k in ("epoch", "exact_epochs")}
        print(f"pause: paused in epoch 2 at step {int(saved['step'])}, pause file exact_epochs=4 OK")

        # Mismatched resumes: exit 2, nothing written.
        stripped = base / "run" / "pause_nokey"
        payload = dict(saved)
        payload.pop("exact_epochs")
        save_checkpoint(strips, payload, pause_dir=stripped)
        cases = {
            "--exact-epochs 5": (pause, ["--exact-epochs", "5"]),
            "no flag": (pause, []),
            "key removed, flag on": (stripped, ["--exact-epochs", "4"]),
        }
        for label, (folder, extra) in cases.items():
            before = _listing(base)
            code, text = _main_code(_cli(strips, "--ckpt-dir", str(ck), "--pause-dir", str(folder), *extra))
            after = _listing(base)
            assert code == 2, (label, code, text[-2000:])
            assert "does not match the paused run" in text, (label, text[-2000:])
            assert before == after, (label, set(after) ^ set(before))
            print(f"resume refused ({label}): exit 2, nothing written OK")

        # The matching resume ends at ckpt_ep3.
        _, text = _run(strips, ck, exact_epochs=4, resume=True, pause_dir=pause)
        got = _ckpts(ck)
        assert [c["epoch"] for c in got] == [0, 1, 2, 3], [c["path"].name for c in got]
        assert text.count(EXIT_LINE.format(last=3)) == 1, text[-3000:]
        assert "[GVD] holdout stopped" not in text
        assert all(c["meta"].get("exact_epochs") == 4 for c in got)
        assert load_checkpoint(strips, pause_dir=pause) is None, "finished run kept its pause file"
        print("resume: ends at ckpt_ep3, no ckpt_ep4, pause file cleared OK")


def check_resume_rule() -> None:
    from python.train.checkpoint import exact_epochs_resume_refusal

    assert exact_epochs_resume_refusal(None, 4) is None  # nothing paused: a fresh run
    assert exact_epochs_resume_refusal(None, None) is None
    assert exact_epochs_resume_refusal({"exact_epochs": 4}, 4) is None
    assert exact_epochs_resume_refusal({}, None) is None
    assert exact_epochs_resume_refusal({"exact_epochs": None}, None) is None
    for saved, flag in (({}, 4), ({"exact_epochs": None}, 4), ({"exact_epochs": 5}, 4), ({"exact_epochs": 4}, None)):
        why = exact_epochs_resume_refusal(saved, flag)
        assert why and "does not match the paused run" in why, (saved, flag, why)
    print("resume rule: missing/null/other refused with the flag; a number refused without it OK")


def main() -> None:
    check_remaining_until_exact_end()
    check_parser()
    check_resume_rule()
    check_refusals()
    check_fixture_runs()
    check_pause_resume()
    print("test_exact_epochs: all OK")


if __name__ == "__main__":
    main()
