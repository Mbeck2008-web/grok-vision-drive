#!/usr/bin/env python3
"""``--pause-dir``: where the pause file is written, loaded and deleted.

* default: with no flag the pause path is ``<strips parent>/<name>.gvd-pause.npz``,
  exactly as before; a run start reads it for the panel, and a finished run
  still overwrites then deletes that file.
* flag: save, load, the panel read and the delete-on-finish all use
  ``<pause dir>/<name>.gvd-pause.npz``. A fake original beside the strip folder
  keeps its bytes (sha256) through a real tiny CPU run that saves every step.
* refusal: a pause dir under the strips parent, equal to it, inside the strip
  folder, or an ancestor of it is refused; so is an existing target pause file
  with ``--no-resume``. The CLI exits non-zero before training starts.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from python.debug.lane_targets import LANE_GT_DIM  # noqa: E402

N_ROWS = 120
FAKE = b"fake original pause file: must never be read, rewritten or deleted\n" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lane(index: int) -> list[float]:
    center = 0.6 * math.sin(index / 9.0)
    values = []
    for _ in range(4):
        values += [center, center - 1.8, center + 1.8]
    return values + [0.0, 1.0]


def _fixture(root: Path) -> None:
    import cv2

    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    rng = np.random.default_rng(0)
    state_lines, gt_lines = [], []
    for index in range(N_ROWS):
        frame = (rng.random((48, 64, 3)) * 160 + 40).astype(np.uint8)
        col = int(32 + 20 * math.sin(index / 9.0))
        frame[:, max(0, col - 2): col + 2, :] = 250
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        (cam / f"{index:06d}.png").write_bytes(buf.tobytes())
        state_lines.append(
            {
                "i": index,
                "t": float(index) * 0.1,
                "dt_s": None if index == 0 else 0.1,
                "segment": "west_coast_usa#0",
                "cams": {"main": {"file": f"raw/main/{index:06d}.png", "t": float(index) * 0.1}},
                "wheel": {"steer": 0.1},
                "pedals": {"throttle": 0.2, "brake": 0.0},
            }
        )
        tech = {"lane_gt": _lane(index), "lane_gt_valid": [1.0] * LANE_GT_DIM, "speed_mps": 10.0}
        gt_lines.append({"i": index, "t": float(index) * 0.1, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")


def _layout(base: Path) -> tuple[Path, Path]:
    """``base/data/scene_x`` (strips) and the fake original beside it."""
    strips = base / "data" / "scene_x"
    strips.mkdir(parents=True)
    original = base / "data" / "scene_x.gvd-pause.npz"
    original.write_bytes(FAKE)
    return strips, original


def check_default_path() -> None:
    from python.train.checkpoint import checkpoint_path

    with tempfile.TemporaryDirectory() as tmp:
        strips = Path(tmp) / "data" / "scene_x"
        got = checkpoint_path(strips)
        want = strips.parent / "scene_x.gvd-pause.npz"
        print(f"default pause path: <strips parent>/{got.name} (parent is strips parent: {got.parent == strips.parent})")
        assert str(got) == str(want), (got, want)
    print("default: pause path unchanged OK")


def check_cli() -> None:
    from python.train.train_scene import build_parser

    assert build_parser().parse_args([]).pause_dir == ""
    assert build_parser().parse_args(["--pause-dir", "p"]).pause_dir == "p"
    print("cli: --pause-dir parses OK")


def check_scoped_io() -> None:
    from python.train.checkpoint import checkpoint_path, clear_checkpoint, load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        strips, original = _layout(Path(tmp))
        before = _sha(original)
        pause = Path(tmp) / "scratch" / "pause"
        payload = {"step": 7, "weights": {"w": np.arange(3, dtype=np.float32)}}
        saved = save_checkpoint(strips, payload, pause_dir=pause)
        assert saved == pause / "scene_x.gvd-pause.npz" == checkpoint_path(strips, pause)
        loaded = load_checkpoint(strips, pause_dir=pause)
        assert loaded is not None and int(loaded["step"]) == 7
        clear_checkpoint(strips, pause_dir=pause)
        assert not saved.exists()
        assert load_checkpoint(strips, pause_dir=pause) is None
        assert original.is_file() and _sha(original) == before
    print("io: save/load/clear stay in --pause-dir, original sha unchanged OK")


def check_window() -> None:
    from python.train.status_window import TrainWindow

    with tempfile.TemporaryDirectory() as tmp:
        strips, original = _layout(Path(tmp))
        before = _sha(original)
        window = TrainWindow()
        # The original is not an npz: reading it would raise.
        window.load_folder(strips, pause_dir=Path(tmp) / "scratch" / "pause")
        assert window.phase == "idle"
        assert _sha(original) == before
    print("panel: load_folder reads only --pause-dir OK")


def check_refusals() -> None:
    from python.train.checkpoint import PauseDirRefused, check_pause_dir

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        strips, original = _layout(base)
        bad = {
            "strips parent": base / "data",
            "under strips parent": base / "data" / "pause",
            "inside strip folder": strips / "pause",
            "ancestor of strips parent": base,
        }
        for label, folder in bad.items():
            try:
                check_pause_dir(strips, folder, no_resume=True)
            except PauseDirRefused as exc:
                print(f"refused ({label}): {exc}")
            else:
                raise AssertionError(f"{label} was not refused")
        ok = base / "scratch" / "pause"
        assert check_pause_dir(strips, ok, no_resume=True) == ok.resolve() / "scene_x.gvd-pause.npz"
        ok.mkdir(parents=True)
        (ok / "scene_x.gvd-pause.npz").write_bytes(b"x")
        try:
            check_pause_dir(strips, ok, no_resume=True)
        except PauseDirRefused as exc:
            print(f"refused (existing target with --no-resume): {exc}")
        else:
            raise AssertionError("existing target with --no-resume was not refused")
        assert check_pause_dir(strips, ok, no_resume=False) == ok.resolve() / "scene_x.gvd-pause.npz"
        assert _sha(original) == hashlib.sha256(FAKE).hexdigest()
    print("refusal: data-root and existing-target cases refused OK")


def check_cli_refuses() -> None:
    from python.train import train_scene

    with tempfile.TemporaryDirectory() as tmp:
        strips, original = _layout(Path(tmp))
        before = _sha(original)
        out = io.StringIO()
        code = None
        with contextlib.redirect_stdout(out):
            try:
                train_scene.main(
                    ["--strips", str(strips), "--pause-dir", str(strips.parent / "p"),
                     "--no-resume", "--no-panel", "--autostart", "--dry-run"]
                )
            except SystemExit as exc:
                code = exc.code
        text = out.getvalue()
        assert code not in (None, 0), (code, text)
        assert "pause-dir refused" in text, text
        assert not (strips.parent / "p").exists()
        assert _sha(original) == before
    print(f"cli: refused with exit {code} before training OK")


def _run(strips: Path, **extra) -> dict:
    from python.train.train_scene import train_directory

    return train_directory(
        strips,
        epochs=1,
        lr=1e-3,
        capability=None,
        export_path=None,
        free_vram_bytes=0,
        free_ram_bytes=8_000_000_000,
        resume=False,
        open_panel=False,
        autostart=True,
        cams=("main",),
        heads=("lane_gt",),
        augment=False,
        workers=0,
        min_valid=0.0,
        holdout_blocks=1,
        status_root=strips.parent.parent / "_status",
        **extra,
    )


@contextlib.contextmanager
def _save_every_step():
    """Pause-save on every step and record every save and clear target."""
    from python.debug import remote_status
    from python.train import checkpoint as ck
    from python.train import train_scene

    seen: dict[str, list[Path]] = {"save": [], "clear": [], "load": []}
    old = remote_status.CHECKPOINT_S, train_scene.save_checkpoint, train_scene.clear_checkpoint, train_scene.load_checkpoint

    def _save(root, payload, **kw):
        path = old[1](root, payload, **kw)
        seen["save"].append(Path(path))
        return path

    def _where(root, kw):
        return ck.checkpoint_path(root, kw["pause_dir"]) if kw.get("pause_dir") else ck.checkpoint_path(root)

    def _clear(root, **kw):
        seen["clear"].append(_where(root, kw))
        return old[2](root, **kw)

    def _load(root, **kw):
        seen["load"].append(_where(root, kw))
        return old[3](root, **kw)

    remote_status.CHECKPOINT_S = 0.0
    train_scene.save_checkpoint, train_scene.clear_checkpoint, train_scene.load_checkpoint = _save, _clear, _load
    try:
        yield seen
    finally:
        remote_status.CHECKPOINT_S = old[0]
        train_scene.save_checkpoint, train_scene.clear_checkpoint, train_scene.load_checkpoint = old[1], old[2], old[3]


def check_default_run_touches_original() -> None:
    """Today's behaviour, unchanged: the run saves beside the strips and deletes it at the end."""
    from python.train.checkpoint import save_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        strips, original = _layout(Path(tmp))
        _fixture(strips)
        # A valid (stale) pause beside the strips: the start reads it for the panel.
        save_checkpoint(strips, {"step": 1, "weights": {"w": np.zeros((2,), dtype=np.float32)}})
        stale = _sha(original)
        with _save_every_step() as seen, contextlib.redirect_stdout(io.StringIO()):
            _run(strips)
        assert seen["save"], "no pause save happened"
        assert all(p == original for p in seen["save"]), seen["save"]
        assert seen["clear"] == [original]
        assert not original.exists(), "finished run did not delete the default pause file"
    assert stale  # the stale file was overwritten by the first save, then deleted
    print(f"default run: {len(seen['save'])} saves to <strips parent>, then deleted it (unchanged) OK")


def check_flag_run_spares_original() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        strips, original = _layout(Path(tmp))
        _fixture(strips)
        before = _sha(original)
        pause = Path(tmp) / "scratch" / "pause"
        target = pause / "scene_x.gvd-pause.npz"
        with _save_every_step() as seen, contextlib.redirect_stdout(io.StringIO()) as out:
            _run(strips, pause_dir=pause)
        assert seen["save"], "no pause save happened"
        assert all(p == target for p in seen["save"]), seen["save"]
        assert seen["clear"] == [target], seen["clear"]
        assert not target.exists()
        assert original.is_file() and _sha(original) == before
        assert f"[GVD] pause file: {target}" in out.getvalue()
    print(f"flag run: {len(seen['save'])} saves + delete in --pause-dir, original sha {before[:12]} unchanged OK")


def main() -> None:
    check_default_path()
    check_cli()
    check_scoped_io()
    check_window()
    check_refusals()
    check_cli_refuses()
    check_default_run_touches_original()
    check_flag_run_spares_original()
    print("test_pause_dir: all OK")


if __name__ == "__main__":
    main()
