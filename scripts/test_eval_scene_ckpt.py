#!/usr/bin/env python3
"""Eval-only scene checkpoint scoring on the trainer's own holdout.

* A tiny synthetic pair of recordings (holdout source + a train-only root) and
  one epoch of the trainer write a .npz. The holdout-source fixture plants a
  settle row and an overexposed (road-clip) frame in the scored tail so
  --mask-events and the default #211 filters drop rows. The eval script's
  n/r/rms/rms_mean match that val row within 1e-6 (same --holdout-blocks,
  train-only roots, --mask-events, and the default #211 filters), and n is
  below the unfiltered count.
* A missing path or an invalid file is refused (non-zero exit, no score).
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
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
from python.train.frame_quality import DEFAULT_MAX_ROAD_CLIP  # noqa: E402

N_HOLD = 80
N_TRAIN_ONLY = 24
# Last 20% of an 8 s / 0.1 s recording starts at t=6.32 (indices 64-79).
SETTLE_I = 70
CLIP_I = 75


def _lane(index: int) -> list[float]:
    center = 0.6 * math.sin(index / 9.0)
    values = []
    for _ in range(4):
        values += [center, center - 1.8, center + 1.8]
    return values + [0.0, 1.0]


def _write_recording(
    root: Path,
    n: int,
    *,
    segment: str,
    seed: int,
    settle_i: int | None = None,
    clip_i: int | None = None,
) -> None:
    import cv2

    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    state_lines, gt_lines = [], []
    for index in range(n):
        frame = (rng.random((48, 64, 3)) * 160 + 40).astype(np.uint8)
        col = int(32 + 20 * math.sin(index / 9.0))
        frame[:, max(0, col - 2): col + 2, :] = 250
        if clip_i is not None and index == int(clip_i):
            # Lower-half any-channel clip of 1.0, above DEFAULT_MAX_ROAD_CLIP 0.20.
            frame[frame.shape[0] // 2 :, :, :] = 255
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        (cam / f"{index:06d}.png").write_bytes(buf.tobytes())
        row = {
            "i": index,
            "t": float(index) * 0.1,
            "dt_s": None if index == 0 else 0.1,
            "segment": segment,
            "cams": {"main": {"file": f"raw/main/{index:06d}.png", "t": float(index) * 0.1}},
            "wheel": {"steer": 0.1},
            "pedals": {"throttle": 0.2, "brake": 0.0},
            "speed_mps": 10.0,
        }
        if settle_i is not None and index == int(settle_i):
            row["settle"] = True
        state_lines.append(row)
        tech = {"lane_gt": _lane(index), "lane_gt_valid": [1.0] * LANE_GT_DIM, "speed_mps": 10.0}
        gt_lines.append({"i": index, "t": float(index) * 0.1, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")


def _val_records(status: Path) -> list[dict]:
    path = status / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("kind") == "val"]


def _json_line(text: str) -> dict:
    for line in text.splitlines():
        raw = line.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
            except ValueError:
                continue
            if isinstance(payload, dict) and {"n", "r", "rms", "rms_mean"} <= set(payload):
                return payload
    raise AssertionError(f"no JSON score line in:\n{text[-3000:]}")


def _eval_argv(ckpt: Path, prior: Path, extra: Path, *, filtered: bool) -> list[str]:
    argv = [
        "--ckpt",
        str(ckpt),
        "--strips",
        str(prior),
        "--strips",
        str(extra),
        "--holdout-source",
        str(prior),
        "--holdout-blocks",
        "1",
        "--seed",
        "73",
    ]
    if filtered:
        argv.append("--mask-events")
    else:
        argv.extend(["--keep-wrong-way", "--max-road-clip", "-1"])
    return argv


def check_refuse_missing_and_invalid() -> None:
    from eval_scene_ckpt import main as eval_main

    with tempfile.TemporaryDirectory() as tmp:
        strips = Path(tmp) / "rec"
        _write_recording(strips, 8, segment="west_coast_usa#0", seed=1)
        missing = Path(tmp) / "no-such.npz"
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = eval_main(["--ckpt", str(missing), "--strips", str(strips), "--holdout-blocks", "1"])
        text = out.getvalue() + err.getvalue()
        assert code not in (0, None), (code, text[-2000:])
        assert "no JSON score line" not in text
        try:
            _json_line(text)
        except AssertionError:
            pass
        else:
            raise AssertionError("a missing checkpoint still printed a score")

        bogus = Path(tmp) / "bad.npz"
        np.savez(bogus, meta=np.array("not-a-ckpt"))
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = eval_main(["--ckpt", str(bogus), "--strips", str(strips), "--holdout-blocks", "1"])
        text = out.getvalue() + err.getvalue()
        assert code not in (0, None), (code, text[-2000:])
        try:
            _json_line(text)
        except AssertionError:
            pass
        else:
            raise AssertionError("an invalid checkpoint still printed a score")
    print("refuse: missing and invalid checkpoints exit non-zero, no score line OK")


def check_matches_trainer_val() -> None:
    import torch

    from eval_scene_ckpt import main as eval_main
    from python.train.model_size import scene_net_kwargs
    from python.train.train_scene import train_directory

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior" / "ctl_synth"
        extra = Path(tmp) / "extra" / "ctl_synth"
        _write_recording(
            prior,
            N_HOLD,
            segment="west_coast_usa#0",
            seed=0,
            settle_i=SETTLE_I,
            clip_i=CLIP_I,
        )
        _write_recording(extra, N_TRAIN_ONLY, segment="italy#0", seed=1)
        ckpts = Path(tmp) / "ckpts"
        status = Path(tmp) / "_status"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            train_directory(
                prior,
                roots=(prior, extra),
                holdout_source=prior,
                epochs=1,
                exact_epochs=1,
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
                mask_events=True,
                drop_wrong_way=True,
                max_road_clip=float(DEFAULT_MAX_ROAD_CLIP),
                seed=73,
                ckpt_dir=ckpts,
                status_root=status,
                model_net_kwargs=scene_net_kwargs("tiny"),
                model_preset="tiny",
            )
        text = log.getvalue()
        vals = _val_records(status)
        assert len(vals) == 1, text[-4000:]
        rec = vals[0]
        written = sorted(ckpts.glob("*.npz"))
        assert len(written) == 1, (written, text[-3000:])
        ckpt = written[0]
        assert rec.get("ckpt") == ckpt.name, (rec.get("ckpt"), ckpt.name)
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = eval_main(_eval_argv(ckpt, prior, extra, filtered=True))
        scored_text = out.getvalue() + err.getvalue()
        assert code in (0, None), scored_text[-4000:]
        scored = _json_line(scored_text)
        assert int(scored["n"]) == int(rec["n"]), (scored, rec)
        for key in ("r", "rms", "rms_mean"):
            left, right = scored.get(key), rec.get(key)
            assert left is not None and right is not None, (key, scored, rec)
            delta = abs(float(left) - float(right))
            assert delta < 1e-6, (key, left, right, delta)
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            raw_code = eval_main(_eval_argv(ckpt, prior, extra, filtered=False))
        raw_text = out.getvalue() + err.getvalue()
        assert raw_code in (0, None), raw_text[-4000:]
        raw = _json_line(raw_text)
        assert int(scored["n"]) < int(raw["n"]), (
            f"filtered n={scored['n']} was not below unfiltered n={raw['n']}"
        )
        print(
            f"match trainer val: n={rec['n']} r={rec['r']} rms={rec['rms']} "
            f"rms_mean={rec['rms_mean']} | eval n={scored['n']} r={scored['r']} "
            f"rms={scored['rms']} rms_mean={scored['rms_mean']} "
            f"unfiltered n={raw['n']} OK"
        )


def check_route_missing_matches_training() -> None:
    text = (ROOT / "scripts" / "eval_scene_ckpt.py").read_text(encoding="utf-8")
    assert 'attach_route_cmds(rows, missing="mask")' in text
    print("  eval route join uses missing=mask, same as training")


def main() -> None:
    check_route_missing_matches_training()
    check_refuse_missing_and_invalid()
    check_matches_trainer_val()
    print("test_eval_scene_ckpt: all OK")


if __name__ == "__main__":
    main()
