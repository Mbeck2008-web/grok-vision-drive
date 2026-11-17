#!/usr/bin/env python3
"""Offline checks for scripts/eval_detect.py (synthetic fixture, no weight download)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_detect as ev  # noqa: E402


def test_iou_and_threshold() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    assert abs(ev.iou_xyxy(a, a) - 1.0) < 1e-9
    # Intersection 5x10=50, union 100+100-50=150 -> 1/3
    assert abs(ev.iou_xyxy(a, (5.0, 0.0, 15.0, 10.0)) - (50.0 / 150.0)) < 1e-9
    assert ev.iou_xyxy(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    # Exactly 0.5: box (0,0,10,10) vs (5,0,15,10) is 1/3, so construct 0.5.
    # (0,0,10,10) vs (5,0,10,10): inter=50, union=100+50-50=100 -> 0.5
    assert abs(ev.iou_xyxy(a, (5.0, 0.0, 10.0, 10.0)) - 0.5) < 1e-9


def test_pr_perfect_miss_and_fp() -> None:
    gt = [{"image": "a.png", "xyxy": [10.0, 10.0, 50.0, 50.0]}]
    hit = ev.pr_for_class(
        [{"image": "a.png", "xyxy": [11.0, 11.0, 49.0, 49.0], "conf": 0.9}],
        gt,
        iou_thr=0.5,
    )
    assert hit["tp"] == 1 and hit["fp"] == 0 and hit["fn"] == 0
    assert hit["precision"] == 1.0 and hit["recall"] == 1.0

    miss = ev.pr_for_class([], gt, iou_thr=0.5)
    assert miss["tp"] == 0 and miss["fp"] == 0 and miss["fn"] == 1
    assert miss["precision"] is None and miss["recall"] == 0.0

    fp = ev.pr_for_class(
        [{"image": "a.png", "xyxy": [80.0, 80.0, 100.0, 100.0], "conf": 0.9}],
        gt,
        iou_thr=0.5,
    )
    assert fp["tp"] == 0 and fp["fp"] == 1 and fp["fn"] == 1
    assert fp["precision"] == 0.0 and fp["recall"] == 0.0

    # Exactly iou 0.5 is a hit; just under is a miss.
    edge_hit = ev.pr_for_class(
        [{"image": "a.png", "xyxy": [5.0, 0.0, 10.0, 10.0], "conf": 0.8}],
        [{"image": "a.png", "xyxy": [0.0, 0.0, 10.0, 10.0]}],
        iou_thr=0.5,
    )
    assert edge_hit["tp"] == 1 and edge_hit["fn"] == 0

    under = ev.pr_for_class(
        [{"image": "a.png", "xyxy": [5.01, 0.0, 10.0, 10.0], "conf": 0.8}],
        [{"image": "a.png", "xyxy": [0.0, 0.0, 10.0, 10.0]}],
        iou_thr=0.5,
    )
    assert under["tp"] == 0 and under["fp"] == 1 and under["fn"] == 1


def test_classes_are_separate() -> None:
    gts = {
        "a.png": [
            {"cls": "vehicle", "xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"cls": "pedestrian", "xyxy": [80.0, 80.0, 100.0, 120.0]},
        ]
    }
    preds = {
        "a.png": [
            {"cls": "vehicle", "xyxy": [11.0, 11.0, 49.0, 49.0], "conf": 0.9},
        ]
    }
    m = ev.per_class_metrics(preds, gts, iou_thr=0.5)
    assert m["vehicle"]["tp"] == 1 and m["vehicle"]["fn"] == 0
    assert m["pedestrian"]["tp"] == 0 and m["pedestrian"]["fn"] == 1
    # A vehicle pred must not close a pedestrian GT even if the box overlaps.
    overlap_pred = {
        "a.png": [
            {"cls": "vehicle", "xyxy": [80.0, 80.0, 100.0, 120.0], "conf": 0.99},
        ]
    }
    m2 = ev.per_class_metrics(overlap_pred, gts, iou_thr=0.5)
    assert m2["vehicle"]["fp"] == 1 and m2["vehicle"]["fn"] == 1
    assert m2["pedestrian"]["tp"] == 0 and m2["pedestrian"]["fn"] == 1


def _write_fixture(td: Path) -> tuple[Path, Path]:
    labels = td / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "image": "a.png",
                "boxes": [
                    {"cls": "vehicle", "xyxy": [10, 10, 50, 50]},
                    {"cls": "pedestrian", "xyxy": [80, 80, 100, 120]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    preds = {
        "yolox-nano": {
            "a.png": [
                {"cls": "vehicle", "xyxy": [11, 11, 49, 49], "conf": 0.9},
            ]
        },
        "yolox-tiny": {
            "a.png": [
                {"cls": "vehicle", "xyxy": [11, 11, 49, 49], "conf": 0.91},
                {"cls": "pedestrian", "xyxy": [80, 80, 100, 120], "conf": 0.8},
            ]
        },
        "yolov8n": {
            "a.png": [
                {"cls": "vehicle", "xyxy": [11, 11, 49, 49], "conf": 0.88},
                {"cls": "pedestrian", "xyxy": [200, 200, 220, 240], "conf": 0.7},
            ]
        },
    }
    preds_path = td / "preds.json"
    preds_path.write_text(json.dumps(preds), encoding="utf-8")
    return labels, preds_path


def test_cli_synthetic_fixture(tmp_path: Path) -> None:
    labels, preds = _write_fixture(tmp_path)
    out_json = tmp_path / "out.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ev.main(
            [
                "--labels",
                str(labels),
                "--preds-json",
                str(preds),
                "--json",
                str(out_json),
            ]
        )
    assert rc == 0
    text = buf.getvalue()
    assert "UNPROVEN" in text and "yolox-nano" in text and "yolox-tiny" in text
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["unproven"] is True
    nano = payload["results"]["yolox-nano"]["metrics"]
    tiny = payload["results"]["yolox-tiny"]["metrics"]
    v8 = payload["results"]["yolov8n"]["metrics"]
    assert nano["vehicle"]["precision"] == 1.0 and nano["vehicle"]["recall"] == 1.0
    assert nano["pedestrian"]["recall"] == 0.0 and nano["pedestrian"]["fn"] == 1
    assert tiny["vehicle"]["recall"] == 1.0 and tiny["pedestrian"]["recall"] == 1.0
    assert v8["vehicle"]["tp"] == 1 and v8["pedestrian"]["fp"] == 1 and v8["pedestrian"]["fn"] == 1


def test_cli_skips_missing_weights(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"image": str(tmp_path / "a.png"), "cls": "vehicle", "xyxy": [1, 1, 4, 4]}) + "\n",
        encoding="utf-8",
    )
    # Dummy image so resolve_images would work, but we still skip missing weights.
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # Point every weight path at a missing file; --preds-json is not set.
    # Without a readable image for OpenCV this would ERROR if we tried to run ONNX,
    # so use evaluate() directly on missing paths.
    results = ev.evaluate(
        ev.load_labels(labels),
        (
            ("yolox-nano", tmp_path / "missing_nano.onnx", "yolox"),
            ("yolox-tiny", tmp_path / "missing_tiny.onnx", "yolox"),
            ("yolov8n", tmp_path / "missing_v8.onnx", "yolov8"),
        ),
        images={str(tmp_path / "a.png"): tmp_path / "a.png"},
    )
    assert results["yolox-nano"]["status"] == "SKIPPED"
    assert results["yolox-tiny"]["status"] == "SKIPPED"
    assert results["yolov8n"]["status"] == "SKIPPED"
    text = ev.format_report(results, iou_thr=0.5, conf=0.35)
    assert "UNPROVEN" in text
    assert "25.8" in text and "37.3" in text and "32.8" in text
    assert "SKIPPED" in text


def test_cli_bad_labels(tmp_path: Path) -> None:
    def run(argv: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return ev.main(argv)

    missing = run(["--labels", str(tmp_path / "nope.jsonl")])
    assert missing == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("{}\n", encoding="utf-8")
    # {} is invalid JSONL (no image) -> 2
    assert run(["--labels", str(empty)]) == 2
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    assert run(["--labels", str(blank)]) == 2


def main() -> None:
    test_iou_and_threshold()
    test_pr_perfect_miss_and_fp()
    test_classes_are_separate()
    with tempfile.TemporaryDirectory() as td:
        test_cli_synthetic_fixture(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_cli_skips_missing_weights(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_cli_bad_labels(Path(td))
    print("test_eval_detect: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_eval_detect: FAIL - {e}")
        sys.exit(1)
