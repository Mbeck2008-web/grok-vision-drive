#!/usr/bin/env python3
"""GVD TRAIN panel (spec v3, PR-A).

* params: the panel's params row is the trainable sum of the live net it was
  given (``set_net``), for the default net, a ``cam_age=True`` net and a
  one-camera net.
* theme: imported from the nerd panel, not copied.
* replay: the fixture run renders all six graphs with a line; an empty status
  folder renders all six idle ("--", plot area untouched). No window opens.
* pause path: the default sibling and a ``--pause-dir`` file both show paused
  with the step and loss curve; with no path passed in, a sibling pause file
  is ignored and the panel is idle.
* val tail: only whole ``kind == "val"`` rows, only this run's (``since``),
  a half-written line waits, a truncated file starts over.
* labels: memory reads "system RAM used" and "trainer RSS", val reads
  "val since this launch"; a replay samples no host RAM or RSS.
* driving: two flat lines on separate scales stay visible (offset by the drawn line height).
* read-only: fixture hashes and mtimes are unchanged, and the module has no
  file writes outside the replay PNG and no training-control writes.
* replay PNG: create-only, reachable only from ``main`` (``--replay``), refuses
  an existing ``--out`` and any ``--out`` under the data root.
* headless: ``GVD_HEADLESS=1`` and no-``DISPLAY`` dry runs pass with
  ``cv2.namedWindow`` patched to raise.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "train_status"
STRIPS = FIXTURE / "scene_demo"
PAUSE = FIXTURE / "scene_demo.gvd-pause.npz"
GRAPH_NAMES = ("eta", "sps", "lr", "driving", "memory", "loss")


def _params_row(window) -> str:
    rows = [line for line in window.lines.splitlines() if line.startswith("params")]
    assert len(rows) == 1, window.lines
    return rows[0]


def check_params_live_net() -> None:
    import torch

    from python.train.scene_net import SceneNet
    from python.train.status_window import TrainWindow

    torch.manual_seed(0)
    configs = (
        ("cam_age=True", lambda: SceneNet(cam_age=True)),
        ("default", lambda: SceneNet()),
        ("cams=main", lambda: SceneNet(cams=("main",))),
    )
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "strips"
        empty.mkdir()
        for label, build in configs:
            net = build()
            want = sum(p.numel() for p in net.parameters() if p.requires_grad)
            window = TrainWindow()
            if hasattr(window, "set_net"):
                window.set_net(net)
            window.load_folder(empty)
            row = _params_row(window)
            assert f"({want})" in row, f"{label}: panel shows {row!r}, live net sum is {want}"
            assert "approximate" not in row, row
            print(f"params {label}: {row} == sum(p.numel()) {want} OK")


def _plot_pixels(panel) -> dict[str, int]:
    """Pixels in each graph's plot area that are not the graph fill."""
    from python.train.status_window import GRAPH_FILL, graph_boxes, loss_plot_box

    counts = {}
    for name, box in graph_boxes(panel.shape[0], panel.shape[1]).items():
        x0, y0, x1, y1 = loss_plot_box(box)
        crop = panel[y0 : y1 + 1, x0 : x1 + 1].reshape(-1, 3)
        counts[name] = int(np.any(crop != np.array(GRAPH_FILL, dtype=np.uint8), axis=1).sum())
    return counts


@contextlib.contextmanager
def _no_window():
    """``cv2.namedWindow`` / ``imshow`` raise; any call is recorded."""
    import cv2

    calls: list[str] = []
    real = cv2.namedWindow, cv2.imshow

    def _named(*_a, **_k):
        calls.append("namedWindow")
        raise RuntimeError("no window allowed here")

    def _show(*_a, **_k):
        calls.append("imshow")
        raise RuntimeError("no window allowed here")

    cv2.namedWindow, cv2.imshow = _named, _show
    try:
        yield calls
    finally:
        cv2.namedWindow, cv2.imshow = real


def check_theme_imported() -> None:
    from python.train import status_window as sw
    from python.viz import nerd

    for name in ("BG", "FG", "ICE", "DIM", "FONT", "FS_TITLE", "FS_BODY", "FS_DIM", "TH", "ROW_H", "PAD_X"):
        assert getattr(sw, name) == getattr(nerd, name), name
    assert sw.PANEL_W == nerd.NERD_WIDTH
    text = (ROOT / "python" / "train" / "status_window.py").read_text(encoding="utf-8")
    assert not re.search(r"^(BG|FG|ICE|DIM|FONT|FS_\w+|TH|ROW_H|PAD_X|PANEL_W)\s*=", text, re.M), "theme copied"
    assert text.count("def _paint_series(") == 1 and "_paint_loss_graph" not in text
    print("theme: imported from nerd, one _paint_series OK")


def check_replay() -> None:
    from python.train import status_window as sw

    assert [graph[0] for graph in sw.GRAPHS] == list(GRAPH_NAMES)
    with _no_window() as calls, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "panel.png"
        with contextlib.redirect_stdout(io.StringIO()):
            code = sw.main(["--replay", str(FIXTURE), "--strips", str(STRIPS), "--out", str(out)])
        assert code == 0 and out.is_file()
        import cv2

        live = cv2.imread(str(out))
        assert live is not None and live.shape[1] == sw.PANEL_W
        drawn = _plot_pixels(live)
        assert all(drawn[name] > 0 for name in GRAPH_NAMES), drawn

        empty = Path(tmp) / "empty_status"
        empty.mkdir()
        idle = sw.render_replay(empty)
        blank = _plot_pixels(idle)
        assert all(blank[name] == 0 for name in GRAPH_NAMES), blank
        assert calls == [], calls

    window = sw.TrainWindow()
    window.load_status_dir(FIXTURE)
    for status in sw.replay_statuses(FIXTURE):
        window.update(status)
    rows = [json.loads(line) for line in (FIXTURE / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    want_val = [row["loss"] for row in rows if row.get("kind") == "val"]
    assert len(want_val) == 2 and window.series["val"] == want_val, window.series["val"]
    assert len(window.losses) == 12 and len(window.series["vram"]) == 12
    assert len(window.series["ram"]) == 12 and all(v > 0 for v in window.series["ram"])
    assert len(window.series["rss"]) == 12 and all(v > 0 for v in window.series["rss"])
    assert "frames 0" in window.lines

    replayed = sw.TrainWindow(live_host=False)
    for status in sw.replay_statuses(FIXTURE):
        replayed.update(status)
    assert replayed.series["ram"] == [] and replayed.series["rss"] == [] and len(replayed.series["vram"]) == 12
    print(f"replay: six graphs drawn {drawn}; empty dir all idle; no window OK")


def check_labels_and_driving_offset() -> None:
    from python.train import status_window as sw

    labels = {key: label for _name, _title, keys, _shared in sw.GRAPHS for key, label, _color, _fmt in keys}
    assert labels["ram"] == "system RAM used" and labels["rss"] == "trainer RSS", labels
    assert labels["val"] == "val since this launch", labels
    status = sw.TrainStatus(
        eta_s=None, loss=None, lr=0.0, steps_per_sec=0.0, memory_bytes=2**30, memory_kind="ram",
        device="cpu", batch=1, recommended=False, step=0, steps=1,
    )
    assert "\nsystem RAM 1.00 GB used\n" in sw.format_status(status), sw.format_status(status)

    panel = sw.render_train_panel([], [], series={"hours": [2.0, 2.0, 2.0], "frames": [900.0, 900.0, 900.0]})
    x0, y0, x1, y1 = sw.loss_plot_box(sw.graph_boxes(panel.shape[0], panel.shape[1])["driving"])
    crop = panel[y0 - sw.LINE_W : y1 + 1, x0 : x1 + 1].reshape(-1, 3)  # a line at the top sits on y0
    seen = {name: int(np.all(crop == np.array(color, dtype=np.uint8), axis=1).sum()) for name, color in (("hours", sw.ICE), ("frames", sw.AMBER))}
    assert seen["hours"] > 0 and seen["frames"] > 0, seen
    print(f"labels: system RAM used / trainer RSS / val since this launch; flat driving lines both visible {seen} OK")


def check_replay_out() -> None:
    from python.train import status_window as sw

    text = (ROOT / "python" / "train" / "status_window.py").read_text(encoding="utf-8")
    assert "cv2.imwrite" not in text
    assert len(re.findall(r"open\([^)]*[\"']xb[\"']", text)) == 1
    calls = [m.start() for m in re.finditer(r"_write_png_create_only\(", text)]
    defs = [m.start() for m in re.finditer(r"def _write_png_create_only\(", text)]
    main_at = text.index("def main(")
    assert len(defs) == 1 and len(calls) == 2 and [c for c in calls if c not in [d + 4 for d in defs]][0] > main_at
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            sw.main(["--out", "x.png"])
        raise AssertionError("main without --replay must exit")
    except SystemExit as exc:
        assert exc.code == 2

    saved = os.environ.get("GVD_DATA_DIR")
    with _no_window(), tempfile.TemporaryDirectory() as tmp:
        try:
            data = Path(tmp) / "data"
            (data / "status").mkdir(parents=True)
            os.environ["GVD_DATA_DIR"] = str(data)
            existing = Path(tmp) / "keep.png"
            existing.write_bytes(b"not a png")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = sw.main(["--replay", str(FIXTURE), "--out", str(existing)])
            assert code == 2 and existing.read_bytes() == b"not a png", (code, out.getvalue())
            assert "already exists" in out.getvalue()

            for target in (data / "status" / "panel.png", data / "panel.png", data / ".." / "data" / "x.png"):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = sw.main(["--replay", str(FIXTURE), "--out", str(target)])
                assert code == 2 and not target.exists(), (target, code, out.getvalue())
                assert "under the data root" in out.getvalue(), out.getvalue()
            link = Path(tmp) / "into_data"
            link.symlink_to(data / "status")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = sw.main(["--replay", str(FIXTURE), "--out", str(link / "via_link.png")])
            assert code == 2 and not (data / "status" / "via_link.png").exists(), out.getvalue()

            fresh = Path(tmp) / "fresh.png"
            with contextlib.redirect_stdout(io.StringIO()):
                assert sw.main(["--replay", str(FIXTURE), "--out", str(fresh)]) == 0
            assert fresh.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert sorted(p.name for p in data.rglob("*")) == ["status"]
        finally:
            if saved is None:
                os.environ.pop("GVD_DATA_DIR", None)
            else:
                os.environ["GVD_DATA_DIR"] = saved
    print("replay --out: create-only; refuses an existing file and the data root (incl. via symlink) OK")


def check_pause_path() -> None:
    from python.train.checkpoint import checkpoint_path
    from python.train.status_window import TrainWindow, loss_graph_box, loss_plot_box, loss_polyline

    want = np.load(PAUSE)
    want.close()
    with tempfile.TemporaryDirectory() as tmp:
        strips = Path(tmp) / "data" / "scene_demo"
        shutil.copytree(STRIPS, strips)
        shutil.copy2(PAUSE, checkpoint_path(strips))
        pause = Path(tmp) / "scratch" / "pause"
        pause.mkdir(parents=True)
        shutil.copy2(PAUSE, checkpoint_path(strips, pause))

        for label, path in (("default", checkpoint_path(strips)), ("--pause-dir", checkpoint_path(strips, pause))):
            window = TrainWindow()
            window.load_folder(strips, path)
            assert window.phase == "paused", label
            assert "paused" in window.lines and "step 120" in window.lines and "loss 0.1616" in window.lines
            assert "recorded 1 hour 14 minutes" in window.lines and "frames 150" in window.lines
            assert len(window.losses) == 12
            plot = loss_plot_box(loss_graph_box(window.panel.shape[0], window.panel.shape[1]))
            assert len(loss_polyline(window.losses, plot)) == 12
            print(f"pause {label}: paused, step 120, 12-point loss curve OK")

        # A real pause file sits beside the folder, but none is passed in.
        ignored = TrainWindow()
        ignored.load_folder(strips)
        assert ignored.phase == "idle" and "idle" in ignored.lines and ignored.losses == []
        missing = TrainWindow()
        missing.load_folder(strips, Path(tmp) / "elsewhere" / "scene_demo.gvd-pause.npz")
        assert missing.phase == "idle"
    print("pause: sibling not passed in is ignored; missing path is idle OK")


def check_val_tail() -> None:
    from python.train.status_window import TrainWindow

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        metrics = folder / "metrics.jsonl"
        old = {"kind": "val", "t": 100.0, "loss": 0.9}
        train = {"kind": "train", "t": 200.0, "loss": 0.5}
        val = {"kind": "val", "t": 201.0, "loss": 0.4}
        metrics.write_text("".join(json.dumps(r) + "\n" for r in (old, train, val)), encoding="utf-8")
        window = TrainWindow()
        window.load_status_dir(folder, since=150.0)
        assert window.series["val"] == [0.4], window.series["val"]
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "val", "t": 300.0, "lo')
        window._tail_val()
        assert window.series["val"] == [0.4]
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write('ss": 0.3}\n')
        window._tail_val()
        assert window.series["val"] == [0.4, 0.3]
        metrics.write_text(json.dumps({"kind": "val", "t": 400.0, "loss": 0.2}) + "\n", encoding="utf-8")
        window._tail_val()
        assert window.series["val"] == [0.2], window.series["val"]
        everything = TrainWindow()
        everything.load_status_dir(folder)
        assert everything.series["val"] == [0.2]
        nothing = TrainWindow()
        nothing.load_status_dir(None)
        nothing.load_status_dir(folder / "missing")
        assert nothing.series["val"] == []
    print("val tail: since, half line, truncation OK")


def _tree_state(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def check_read_only() -> None:
    from python.train import status_window as sw
    from python.train.checkpoint import checkpoint_path

    before = _tree_state(FIXTURE)
    names_before = sorted(p.name for p in FIXTURE.rglob("*"))
    with tempfile.TemporaryDirectory() as tmp:
        sw.render_replay(FIXTURE, strips=STRIPS, pause=PAUSE)
        window = sw.TrainWindow()
        window.load_folder(STRIPS, checkpoint_path(STRIPS))
        window.load_status_dir(FIXTURE)
        window.refresh_recorded()
        for status in sw.replay_statuses(FIXTURE):
            window.update(status)
        window.mark_paused({"step": 1, "loss": 0.5, "losses": [0.5]})
        assert list(Path(tmp).iterdir()) == []
    assert _tree_state(FIXTURE) == before
    assert sorted(p.name for p in FIXTURE.rglob("*")) == names_before

    text = (ROOT / "python" / "train" / "status_window.py").read_text(encoding="utf-8")
    writes = re.findall(
        r"write_text|write_bytes|json\.dump\b|np\.save|savez|\.unlink\(|os\.remove|shutil\."
        r"|open\([^)]*[\"'][wa]b?[\"']",
        text,
    )
    assert writes == [], writes
    assert "cv2.imwrite" not in text  # the replay PNG goes through the create-only writer
    control = re.findall(
        r"(?:\blr|batch|seed|loss)\w*\s*\[[^]]*\]\s*=(?!=)|param_groups|manual_seed|\.step\(\)|backward\(",
        text,
    )
    assert control == [], control
    print(f"read-only: {len(before)} fixture files unchanged (sha256 + mtime); no writes in status_window OK")


def check_headless_dry_run() -> None:
    from test_pause_dir import _fixture

    from python.train import train_scene

    for label, env in (("GVD_HEADLESS=1", {"GVD_HEADLESS": "1"}), ("no DISPLAY", {"DISPLAY": None})):
        with tempfile.TemporaryDirectory() as tmp:
            strips = Path(tmp) / "data" / "scene_x"
            strips.mkdir(parents=True)
            _fixture(strips)
            saved = {key: os.environ.get(key) for key in ("GVD_HEADLESS", "DISPLAY", "WAYLAND_DISPLAY", "GVD_DATA_DIR")}
            os.environ["GVD_DATA_DIR"] = str(Path(tmp) / "data")
            os.environ.pop("WAYLAND_DISPLAY", None)
            for key, value in env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            out = io.StringIO()
            code = 0
            try:
                with _no_window() as calls, contextlib.redirect_stdout(out):
                    try:
                        train_scene.main(
                            ["--strips", str(strips), "--dry-run", "--cams", "main", "--heads", "lane_gt",
                             "--workers", "0", "--no-resume", "--holdout-blocks", "1", "--min-valid", "0"]
                        )
                    except SystemExit as exc:
                        code = exc.code or 0
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            text = out.getvalue()
            assert code == 0, (label, code, text[-2000:])
            assert calls == [], (label, calls)
            assert "[GVD] scene status: heartbeat" in text, text[-2000:]
            print(f"headless {label}: dry run exit 0, no window call OK")


def main() -> None:
    check_params_live_net()
    check_theme_imported()
    check_replay()
    check_labels_and_driving_offset()
    check_replay_out()
    check_pause_path()
    check_val_tail()
    check_read_only()
    check_headless_dry_run()
    print("test_train_panel: all OK")


if __name__ == "__main__":
    main()
