#!/usr/bin/env python3
"""Status files, the dataset check, and the eval report. No BeamNG and no torch."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.strip_writer import CANVAS_H, CANVAS_W, SECTORS  # noqa: E402
from python.debug.dataset_check import sector_is_blank, validate_dataset  # noqa: E402
from python.debug.eval_report import (  # noqa: E402
    lookahead_table,
    prediction_records,
    write_eval_bundle,
)
from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from python.debug.lane_targets import tech_from_hit  # noqa: E402
from python.debug.remote_status import (  # noqa: E402
    CollectMeter,
    sanity_reached,
    status_dir,
    write_collect_status,
    write_crash,
    write_heartbeat,
)
from python.train.train_scene import build_parser  # noqa: E402
from scripts.gvd_status import summarize  # noqa: E402
from scripts.validate_scene_lanes import main as validate_main  # noqa: E402


def _jpeg(path: Path, fill: int) -> None:
    import cv2

    image = np.full((CANVAS_H, CANVAS_W, 3), int(fill), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(path)


def _row(index: int, t: float, tech: dict, dt: float | None = 0.1) -> dict:
    return {"i": index, "t": t, "dt_s": dt, "tech": tech, "wheel": {}, "pedals": {}}


def check_status_helpers_and_parser() -> None:
    assert status_dir() is None or os.name == "nt"
    assert sanity_reached(0.01)
    assert not sanity_reached(0.2)
    assert not sanity_reached(None)
    assert not sanity_reached(float("nan"))
    meter = CollectMeter(("main", "wide"))
    meter.note({"main"}, True)
    for _ in range(5):
        meter.note({"main"}, False)
    snap = meter.snapshot(
        elapsed_s=6.0,
        disk_bytes=1000,
        free_bytes=30 * 1024 ** 3,
        ai_speed_mps=8.0,
        ai_mode="span",
        ai_waypoint=None,
        out="E:/GVD_data/scene_lanes",
        target_s=2700.0,
    )
    assert snap["frames"] == 6
    assert snap["dropped_cams"]["wide"] == 6
    assert "wide" in snap["stale_cams"]
    assert snap["gt_valid_frac"] < 0.2
    assert "main" not in snap["stale_cams"]
    args = build_parser().parse_args(["--dry-run", "--resume", "--strips", "x"])
    assert args.dry_run and args.resume and not args.sanity
    args = build_parser().parse_args(["--sanity", "--no-resume", "--metrics-every", "4"])
    assert args.sanity and args.no_resume and args.metrics_every == 4
    assert sector_is_blank(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not sector_is_blank(np.full((8, 8, 3), 40, dtype=np.uint8))


def check_validate_and_status_command() -> None:
    good = tech_from_hit(0.2, 0.05, 3.6, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        strips = root / "scene_lanes"
        (strips / "strips").mkdir(parents=True)
        rows = [_row(i, i * 0.1, good) for i in range(4)]
        (strips / "state.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        for i in range(4):
            _jpeg(strips / "strips" / f"{i:06d}.jpg", 40)
        clean = validate_dataset(strips, root / "validate_ok", montage_n=2)
        assert clean["issue_count"] == 0, clean["issues"]
        assert clean["rows"] == 4 and clean["jpegs"] == 4
        assert len(clean["montages"]) == 2
        assert (root / "validate_ok" / "report.json").is_file()

        bad_tech = dict(good)
        bad_tech["lane_gt"] = list(good["lane_gt"])
        bad_tech["lane_gt"][0] = float("nan")
        bad_rows = [
            _row(0, 0.0, good, None),
            _row(1, 5.0, bad_tech, 5.0),
        ]
        bad = root / "bad"
        (bad / "strips").mkdir(parents=True)
        (bad / "state.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in bad_rows),
            encoding="utf-8",
        )
        _jpeg(bad / "strips" / "000000.jpg", 0)
        _jpeg(bad / "strips" / "000001.jpg", 0)
        report = validate_dataset(bad, root / "validate_bad", montage_n=1)
        text = "\n".join(report["issues"])
        assert "gap" in text
        assert "NaN" in text
        assert any("blank on all" in item for item in report["issues"])
        code = validate_main(["--strips", str(bad), "--out", str(root / "validate_cli")])
        assert code == 1

        status = root / "status"
        write_collect_status(status, snap_from(meter_payload()))
        write_heartbeat(
            status,
            {"kind": "train", "step": 12, "epoch": 1, "loss": 0.4, "eta_s": 90, "lr": 0.001, "device": "cpu"},
        )
        from python.debug.remote_status import append_metrics

        append_metrics(status, {"kind": "train", "step": 12, "loss": 0.4, "nan": 0, "inf": 0})
        write_crash(status, "RuntimeError: boom\n", {"step": 12, "rows": [3, 4]})
        lines = summarize(status, root / "eval")
        blob = "\n".join(lines)
        assert "step=12" in blob
        assert "boom" in blob or "RuntimeError" in blob
        assert "frames=" in blob
        os.environ["GVD_DATA_DIR"] = str(root)
        try:
            from scripts.gvd_status import main as status_main

            assert status_main([]) == 0
        finally:
            os.environ.pop("GVD_DATA_DIR", None)


def meter_payload() -> dict:
    meter = CollectMeter(("main",))
    meter.note({"main"}, True)
    return meter.snapshot(
        elapsed_s=1.0,
        disk_bytes=10,
        free_bytes=21 * 1024 ** 3,
        ai_speed_mps=8.0,
        ai_mode="span",
        ai_waypoint=None,
        out="strips",
        target_s=60.0,
    )


def snap_from(payload: dict) -> dict:
    return payload


def check_eval_files_and_live_fields() -> None:
    pred = np.zeros(14, dtype=np.float32)
    gt = np.zeros(14, dtype=np.float32)
    pred[0] = 1.0
    pred[3] = 1.0
    gt[0] = 0.0
    gt[3] = 0.0
    pred[-1] = 0.8
    gt[-1] = 1.0
    rows = []
    image = np.full((CANVAS_H, CANVAS_W, 3), 30, dtype=np.uint8)
    for i in range(4):
        got = gt.copy()
        hat = pred.copy()
        got[0] = float(i)
        hat[0] = float(i) + (6.0 if i == 3 else 0.1)
        rows.append({"i": i, "t": float(i), "pred": hat, "gt": got, "conf": 0.8, "image": image})
    table = lookahead_table(rows)
    assert table[0]["y_m"] == 5.0
    assert table[0]["n"] == 4
    records = prediction_records(rows)
    assert records[-1]["abs_err_8m"] is not None and records[-1]["abs_err_8m"] > 1.0
    with tempfile.TemporaryDirectory() as tmp:
        summary = write_eval_bundle(Path(tmp), rows, k=2)
        assert (Path(tmp) / "predictions.csv").is_file()
        assert (Path(tmp) / "lookahead.png").is_file()
        assert summary["n"] == 4
        worst = list((Path(tmp) / "worst").glob("*.png"))
        assert len(worst) == 2
        log = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "frames")
        row = log.write(engaged=True, lane_x=0.2, model_lane_x=0.4, model_conf=0.7)
        assert row["model_lane_x"] == 0.4
        assert row["model_conf"] == 0.7
        saved = json.loads((Path(tmp) / "gt.jsonl").read_text(encoding="utf-8"))
        assert saved["model_lane_x"] == 0.4


def main() -> None:
    check_status_helpers_and_parser()
    check_validate_and_status_command()
    check_eval_files_and_live_fields()
    # A filled sector is not blank. The canvas constant is what the JPEGs use.
    assert SECTORS[0].x0 == 0
    print("remote debug ok")


if __name__ == "__main__":
    main()
