#!/usr/bin/env python3
"""Per-epoch checkpoints, their ONNX export, and the offline holdout eval.

* create-only: a second write to the same name is refused and the file is unchanged.
* the step and epoch are in the name and in the file, and a renamed file is refused.
* low space: a write that would leave less than the floor is refused, nothing is
  written, and a training run with an impossible floor logs the refusal and still
  finishes every val.
* round trip: a tiny CPU run writes one checkpoint per val row; each is exported
  to ONNX and scored offline on the same holdout. rms, pred_std and r must match
  the logged val row within 1e-4 and n must be equal.
* the same round trip with ``--cam-age`` (main + a pillar that renders every
  third tick): the export carries ``ages`` and ``gvd_cam_age=1``, the offline
  eval feeds the same hold-last sectors and ages, and the holdout n equals the
  run without ``--cam-age`` (row selection keys on main).
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
from python.train.epoch_ckpt import (  # noqa: E402
    EpochCkptRefused,
    epoch_ckpt_path,
    load_epoch_ckpt,
    parse_ckpt_name,
    sha256_file,
    write_epoch_ckpt,
)

N_ROWS = 200


def _weights() -> dict:
    return {"a.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "a.bias": np.ones(2, dtype=np.float32)}


def check_create_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = epoch_ckpt_path(tmp, Path(tmp) / "rec", "20261021T000000Z", 3, 12332)
        info = write_epoch_ckpt(path, _weights(), {"epoch": 3, "step": 12332}, min_free_bytes=0)
        before = sha256_file(path)
        assert info["sha256"] == before, info
        try:
            write_epoch_ckpt(path, {"a.weight": np.zeros((2, 3), np.float32)}, {"epoch": 3, "step": 12332}, min_free_bytes=0)
        except EpochCkptRefused as exc:
            assert "already exists" in str(exc), exc
        else:
            raise AssertionError("a second write to an existing checkpoint was not refused")
        assert sha256_file(path) == before, "the existing checkpoint was changed"
        # Never inside the strip folder.
        inner = Path(tmp) / "rec" / "x.run20261021T000000Z.ckpt_ep0_step1.npz"
        try:
            write_epoch_ckpt(inner, _weights(), {"epoch": 0, "step": 1}, min_free_bytes=0, strip_root=Path(tmp) / "rec")
        except EpochCkptRefused as exc:
            assert "inside the strip folder" in str(exc), exc
        else:
            raise AssertionError("a checkpoint inside the strip folder was written")
    print("create-only: second write refused, file unchanged; strip folder refused OK")


def check_step_embedded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = epoch_ckpt_path(tmp, Path(tmp) / "scene_lanes_rec5c", "20261021T040702Z", 3, 12332)
        assert path.name == "scene_lanes_rec5c.run20261021T040702Z.ckpt_ep3_step12332.npz", path.name
        write_epoch_ckpt(path, _weights(), {"epoch": 3, "step": 12332, "val": {"rms": 0.5}}, min_free_bytes=0)
        assert parse_ckpt_name(path) == {"strips": "scene_lanes_rec5c", "run": "20261021T040702Z", "epoch": 3, "step": 12332}
        loaded = load_epoch_ckpt(path)
        assert loaded["meta"]["epoch"] == 3 and loaded["meta"]["step"] == 12332, loaded["meta"]
        assert loaded["meta"]["val"] == {"rms": 0.5}
        for key, value in _weights().items():
            assert np.array_equal(loaded["weights"][key], value), key
        wrong = path.with_name(path.name.replace("step12332", "step99"))
        os.rename(path, wrong)
        try:
            load_epoch_ckpt(wrong)
        except ValueError as exc:
            assert "step 99" in str(exc) and "step 12332" in str(exc), exc
        else:
            raise AssertionError("a checkpoint whose name disagrees with its content loaded")
    print("step embedded: name and file carry epoch 3 step 12332, read back exactly; renamed file refused OK")


def check_low_space() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.run20261021T000000Z.ckpt_ep0_step10.npz"
        floor = 8_000_000_000
        try:
            write_epoch_ckpt(path, _weights(), {"epoch": 0, "step": 10}, min_free_bytes=floor, disk_free=lambda _p: floor + 100)
        except EpochCkptRefused as exc:
            assert "low space" in str(exc) and "floor 8.00 GB" in str(exc), exc
        else:
            raise AssertionError("a checkpoint that would cross the free-space floor was written")
        assert not path.exists(), "a refused checkpoint left a file"
        info = write_epoch_ckpt(path, _weights(), {"epoch": 0, "step": 10}, min_free_bytes=floor, disk_free=lambda _p: floor + 10_000_000)
        assert path.is_file() and info["free_after"] >= floor, info
    print("low space: refused below the 8 GB floor, nothing written; written with room OK")


def _lane(index: int) -> list[float]:
    center = 0.6 * math.sin(index / 9.0)
    values = []
    for _ in range(4):
        values += [center, center - 1.8, center + 1.8]
    return values + [0.0, 1.0]


def _fixture(root: Path, side: bool = False) -> None:
    import cv2

    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    pillar = root / "raw" / "pillarL"
    if side:
        pillar.mkdir(parents=True)
    rng = np.random.default_rng(0)
    state_lines, gt_lines = [], []
    for index in range(N_ROWS):
        frame = (rng.random((48, 64, 3)) * 160 + 40).astype(np.uint8)
        # A bright column that moves with the lane center: something to learn.
        col = int(32 + 20 * math.sin(index / 9.0))
        frame[:, max(0, col - 2): col + 2, :] = 250
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        (cam / f"{index:06d}.png").write_bytes(buf.tobytes())
        cams = {"main": {"file": f"raw/main/{index:06d}.png", "t": float(index) * 0.1}}
        if side and index % 3 == 0:
            # A pillar that sees the same column, mirrored: hold-last matters on 2 of 3 ticks.
            ok, buf = cv2.imencode(".png", np.ascontiguousarray(frame[:, ::-1]))
            assert ok
            (pillar / f"{index:06d}.png").write_bytes(buf.tobytes())
            cams["pillarL"] = {"file": f"raw/pillarL/{index:06d}.png", "t": float(index) * 0.1 + 0.004}
        state_lines.append(
            {
                "i": index,
                "t": float(index) * 0.1,
                "dt_s": None if index == 0 else 0.1,
                "segment": "west_coast_usa#0",
                "cams": cams,
                "wheel": {"steer": 0.1},
                "pedals": {"throttle": 0.2, "brake": 0.0},
            }
        )
        tech = {"lane_gt": _lane(index), "lane_gt_valid": [1.0] * LANE_GT_DIM, "speed_mps": 10.0}
        gt_lines.append({"i": index, "t": float(index) * 0.1, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")


def _train(root: Path, **extra):
    from python.train.train_scene import train_directory

    extra.setdefault("cams", ("main",))
    extra.setdefault("status_root", root.parent / "_status")
    return train_directory(
        root,
        epochs=2,
        lr=1e-3,
        capability=None,
        export_path=None,
        free_vram_bytes=0,
        free_ram_bytes=8_000_000_000,
        resume=False,
        open_panel=False,
        autostart=True,
        heads=("lane_gt",),
        augment=False,
        workers=0,
        min_valid=0.0,
        holdout_blocks=1,
        **extra,
    )


def _val_records(status: Path) -> list[dict]:
    path = status / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("kind") == "val"]


def check_round_trip() -> None:
    import torch

    from eval_onnx_holdout import evaluate, val_line
    from python.train.epoch_ckpt import export_epoch_onnx

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _fixture(root)
        ckpts = Path(tmp) / "ckpts"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _train(root, ckpt_dir=ckpts)
        text = log.getvalue()
        vals = _val_records(Path(tmp) / "_status")
        assert len(vals) >= 2, text[-3000:]
        written = sorted(ckpts.glob("*.npz"))
        assert len(written) == len(vals), (len(written), len(vals), text[-3000:])
        assert text.count("[GVD] epoch checkpoint epoch=") == len(vals), text[-3000:]
        # Create-only also across runs: a rerun writes new names, it never replaces these.
        for rec in vals:
            path = ckpts / rec["ckpt"]
            parsed = parse_ckpt_name(path)
            assert parsed["epoch"] == rec["epoch"] and parsed["step"] == rec["step"], (parsed, rec)
            assert sha256_file(path) == rec["ckpt_sha256"], rec
            out = Path(tmp) / f"ep{rec['epoch']}.onnx"
            info = export_epoch_onnx(path, out)
            try:
                export_epoch_onnx(path, out)
            except FileExistsError:
                pass
            else:
                raise AssertionError("export replaced an existing ONNX")
            result = evaluate(root, out, cams=("main",), holdout_blocks=1, eval_chunk=16)
            line = val_line(result)
            assert result["step"] == rec["step"] and result["epoch"] == rec["epoch"], (result, rec)
            assert result["ckpt_sha256"] == info["ckpt_sha256"] == rec["ckpt_sha256"]
            assert result["n"] == rec["n"], (result["n"], rec["n"])
            assert abs(float(result["rms"]) - float(rec["rms"])) < 1e-4, (result["rms"], rec["rms"])
            assert abs(float(result["pred_std"]) - float(rec["pred_std"])) < 1e-4, (result["pred_std"], rec["pred_std"])
            # Skeptic nit: r too, at the same 1e-4 as rms.
            assert rec.get("r") is not None and result.get("r") is not None, (rec.get("r"), result.get("r"))
            assert abs(float(result["r"]) - float(rec["r"])) < 1e-4, (result["r"], rec["r"])
            print(
                f"round trip epoch={rec['epoch']} step={rec['step']}: logged rms={rec['rms']:.6f} n={rec['n']} | "
                f"onnx rms={result['rms']:.6f} n={result['n']} (|d|={abs(result['rms'] - rec['rms']):.2e})"
            )
            assert line.startswith(f"[GVD] val (onnx) step={rec['step']} epoch={rec['epoch']} n={rec['n']} "), line
        # The command lines: export by --ckpt-dir/--export-epoch, then the offline eval's val line.
        from eval_onnx_holdout import main as eval_main
        from export_epoch_onnx import main as export_main

        last = vals[-1]
        cli_onnx = Path(tmp) / "cli.onnx"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert export_main(["--ckpt-dir", str(ckpts), "--export-epoch", str(last["epoch"]), "--out", str(cli_onnx)]) == 0
            assert export_main(["--ckpt-dir", str(ckpts), "--export-epoch", str(last["epoch"]), "--out", str(cli_onnx)]) == 2
            assert eval_main(["--strips", str(root), "--onnx", str(cli_onnx), "--holdout-blocks", "1", "--json", str(Path(tmp) / "r.json")]) == 0
        lines = [line for line in out.getvalue().splitlines() if line.startswith("[GVD] val (onnx)")]
        assert len(lines) == 1 and f"step={last['step']} epoch={last['epoch']} n={last['n']} " in lines[0], out.getvalue()
        assert abs(json.loads((Path(tmp) / "r.json").read_text())["rms"] - last["rms"]) < 1e-4
        print(f"cli: {lines[0]}")
    print("round trip: every val row's checkpoint exports and scores within 1e-4 rms / pred_std / r, same n OK")


def check_round_trip_cam_age() -> None:
    import onnxruntime as ort
    import torch

    from eval_onnx_holdout import evaluate
    from python.train.epoch_ckpt import export_epoch_onnx, load_epoch_ckpt

    # STRIP_ORDER order, as the CLI's parse_cams gives it (the export records that order).
    cams = ("pillarL", "main")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _fixture(root, side=True)
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            _train(root, cams=cams, ckpt_dir=Path(tmp) / "plain", status_root=Path(tmp) / "_plain")
        plain_n = [rec["n"] for rec in _val_records(Path(tmp) / "_plain")]
        torch.manual_seed(0)
        ckpts = Path(tmp) / "aged"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _train(root, cams=cams, ckpt_dir=ckpts, status_root=Path(tmp) / "_aged", cam_age=True)
        text = log.getvalue()
        import re

        planned = re.search(r"\[GVD\] cam-age: hold-last plan on (\d+) of (\d+) rows", text)
        assert planned is not None, text[:3000]
        # Every train and holdout row has a plan; only the rows in the holdout gap have none.
        assert 0 < int(planned.group(1)) < int(planned.group(2)) == N_ROWS, planned.group(0)
        print(planned.group(0))
        vals = _val_records(Path(tmp) / "_aged")
        # The holdout stop may end the two runs at different epochs; every
        # evaluation must still score the same rows (same n) as without --cam-age.
        assert vals and plain_n and {rec["n"] for rec in vals} == set(plain_n) and len(set(plain_n)) == 1, (
            [rec["n"] for rec in vals],
            plain_n,
        )
        for rec in vals:
            path = ckpts / rec["ckpt"]
            assert load_epoch_ckpt(path)["meta"]["cam_age"] is True
            out = Path(tmp) / f"aged_ep{rec['epoch']}.onnx"
            export_epoch_onnx(path, out)
            session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            assert "ages" in [item.name for item in session.get_inputs()]
            assert dict(session.get_modelmeta().custom_metadata_map)["gvd_cam_age"] == "1"
            result = evaluate(root, out, holdout_blocks=1, eval_chunk=16)
            assert result["cam_age"] is True and result["n"] == rec["n"], (result, rec)
            for key in ("rms", "pred_std", "r"):
                assert abs(float(result[key]) - float(rec[key])) < 1e-4, (key, result[key], rec[key])
            print(
                f"cam-age round trip epoch={rec['epoch']} step={rec['step']}: logged rms={rec['rms']:.6f} "
                f"r={rec['r']:.6f} n={rec['n']} | onnx rms={result['rms']:.6f} r={result['r']:.6f} n={result['n']}"
            )
    print("cam-age round trip: ages export, offline eval within 1e-4 rms/pred_std/r, same n as without --cam-age OK")


def check_low_space_run() -> None:
    import torch

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _fixture(root)
        ckpts = Path(tmp) / "ckpts"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = _train(root, ckpt_dir=ckpts, ckpt_min_free_gb=1e12)
        text = log.getvalue()
        vals = _val_records(Path(tmp) / "_status")
        refused = [line for line in text.splitlines() if "epoch checkpoint REFUSED" in line]
        assert vals and len(refused) == len(vals), text[-3000:]
        assert "low space" in refused[0] and "training continues" in refused[0], refused[0]
        assert not list(ckpts.glob("*.npz")), "a refused run wrote a checkpoint"
        assert all(rec["ckpt"] is None and rec["ckpt_sha256"] is None for rec in vals), vals
        assert stats["windows"] > 0
    print(f"low-space run: {len(refused)} val rows, every checkpoint refused and logged, training finished OK")


def check_cli_flags() -> None:
    from python.train.train_scene import build_parser

    args = build_parser().parse_args([])
    assert args.ckpt_dir == "" and args.ckpt_min_free_gb is None
    args = build_parser().parse_args(["--ckpt-dir", "x", "--ckpt-min-free-gb", "12"])
    assert args.ckpt_dir == "x" and args.ckpt_min_free_gb == 12.0
    from export_epoch_onnx import build_parser as export_parser

    args = export_parser().parse_args(["--ckpt-dir", "d", "--export-epoch", "3", "--out", "o.onnx"])
    assert args.epoch == 3
    print("cli: --ckpt-dir / --ckpt-min-free-gb / --export-epoch parse OK")


def main() -> None:
    check_create_only()
    check_step_embedded()
    check_low_space()
    check_cli_flags()
    check_round_trip()
    check_round_trip_cam_age()
    check_low_space_run()
    print("test_epoch_ckpt: all OK")


if __name__ == "__main__":
    main()
