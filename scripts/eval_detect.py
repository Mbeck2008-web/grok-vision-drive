#!/usr/bin/env python3
"""Offline vehicle/pedestrian precision-recall for the shipped and optional detectors.

Compares YOLOX-Nano (shipped), optional YOLOX-Tiny, and an optional
user-supplied YOLOv8n once labeled images exist. Tiny and YOLOv8n weights
are never shipped; pass a local path or drop a file in models/.

This is a measurement command. It does not change the live detector.
Detection quality on GVD data and GPU VRAM use remain UNPROVEN until this
(or an equivalent) is run on a labeled holdout.

Published COCO val mAP 0.5:0.95 (not this dataset, and not this script's
single-threshold P/R): YOLOX-Nano 25.8 at 416, YOLOX-Tiny 32.8 at 416,
YOLOv8n 37.3 at 640. Input sizes differ, so those figures are not a
like-for-like comparison.

  PYTHONPATH=. python scripts/eval_detect.py --images DIR --labels labels.jsonl
  PYTHONPATH=. python scripts/eval_detect.py --images DIR --labels labels.jsonl \\
      --yolox-tiny models/yolox_tiny.onnx --yolov8n models/yolov8n.onnx

Labels JSONL, one image or one box per line:

  {"image": "frame.jpg", "boxes": [{"cls": "vehicle", "xyxy": [x1,y1,x2,y2]}]}
  {"image": "frame.jpg", "cls": "pedestrian", "xyxy": [x1,y1,x2,y2]}

Classes scored: vehicle, pedestrian. Other labels and predictions are ignored.
Default IoU threshold is 0.5; default score cutoff is the live detector's 0.35.
A missing weight is reported as SKIPPED, not as a score.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCORED_CLASSES = ("vehicle", "pedestrian")
DEFAULT_IOU = 0.5
DEFAULT_CONF = 0.35

# Published COCO val mAP 0.5:0.95. Not GVD measurements.
PUBLISHED_MAP = {
    "yolox-nano": "25.8 at 416 (YOLOX, COCO val 0.5:0.95)",
    "yolox-tiny": "32.8 at 416 (YOLOX, COCO val 0.5:0.95)",
    "yolov8n": "37.3 at 640 (Ultralytics YOLOv8n, COCO val 0.5:0.95)",
}

_CLS_ALIASES = {
    "vehicle": "vehicle",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "motorbike": "vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "ped": "pedestrian",
    "0": "pedestrian",
    "2": "vehicle",
    "3": "vehicle",
    "5": "vehicle",
    "7": "vehicle",
}


def normalize_cls(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return _CLS_ALIASES.get(key)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Axis-aligned IoU on xyxy boxes (no +1 pixel pad)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def pr_for_class(
    preds: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
    *,
    iou_thr: float = DEFAULT_IOU,
) -> dict[str, Any]:
    """Single-threshold precision/recall for one class.

    Predictions are matched greedily in descending confidence; a pred is a
    true positive when its IoU with an unmatched GT on the same image is
    at least ``iou_thr`` (exactly ``iou_thr`` counts).
    """
    gts_by_image: dict[str, list[Sequence[float]]] = defaultdict(list)
    for g in gts:
        gts_by_image[str(g["image"])].append(tuple(float(v) for v in g["xyxy"]))
    n_gt = sum(len(v) for v in gts_by_image.values())
    used: dict[str, set[int]] = {img: set() for img in gts_by_image}
    ordered = sorted(preds, key=lambda p: float(p.get("conf", 0.0)), reverse=True)
    tp = 0
    fp = 0
    for p in ordered:
        img = str(p["image"])
        boxes = gts_by_image.get(img, [])
        best_iou = -1.0
        best_j = -1
        already = used.setdefault(img, set())
        for j, gt in enumerate(boxes):
            if j in already:
                continue
            iou = iou_xyxy(p["xyxy"], gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            already.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = n_gt - tp
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / n_gt) if n_gt else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": tp + fp,
        "precision": precision,
        "recall": recall,
    }


def per_class_metrics(
    preds_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    gts_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iou_thr: float = DEFAULT_IOU,
    classes: Sequence[str] = SCORED_CLASSES,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cls in classes:
        preds: list[dict[str, Any]] = []
        gts: list[dict[str, Any]] = []
        for img, boxes in preds_by_image.items():
            for b in boxes:
                if normalize_cls(b.get("cls")) == cls:
                    preds.append({"image": img, "xyxy": b["xyxy"], "conf": b.get("conf", 0.0)})
        for img, boxes in gts_by_image.items():
            for b in boxes:
                if normalize_cls(b.get("cls")) == cls:
                    gts.append({"image": img, "xyxy": b["xyxy"]})
        out[cls] = pr_for_class(preds, gts, iou_thr=iou_thr)
    return out


def load_labels(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load JSONL labels. One image with ``boxes``, or one box per line."""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
        image = row.get("image") or row.get("file") or row.get("filename")
        if not image:
            raise ValueError(f"{path}:{lineno}: missing image")
        boxes = row.get("boxes")
        if boxes is None and "xyxy" in row:
            boxes = [{"cls": row.get("cls") or row.get("class"), "xyxy": row["xyxy"]}]
        if not isinstance(boxes, list):
            raise ValueError(f"{path}:{lineno}: expected boxes or xyxy")
        for box in boxes:
            cls = normalize_cls(box.get("cls") or box.get("class"))
            if cls is None:
                continue
            xyxy = box.get("xyxy")
            if not (isinstance(xyxy, (list, tuple)) and len(xyxy) == 4):
                raise ValueError(f"{path}:{lineno}: xyxy must have 4 numbers")
            by_image[str(image)].append({"cls": cls, "xyxy": [float(v) for v in xyxy]})
    return dict(by_image)


def load_preds_json(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """``{model: {image: [{cls, xyxy, conf}, ...]}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of model -> image -> boxes")
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model, images in data.items():
        if not isinstance(images, dict):
            raise ValueError(f"{path}: {model} must map image names to box lists")
        cleaned: dict[str, list[dict[str, Any]]] = {}
        for img, boxes in images.items():
            row: list[dict[str, Any]] = []
            for box in boxes or []:
                cls = normalize_cls(box.get("cls") or box.get("class"))
                if cls is None:
                    continue
                xyxy = box.get("xyxy")
                if not (isinstance(xyxy, (list, tuple)) and len(xyxy) == 4):
                    raise ValueError(f"{path}: {model}/{img} xyxy must have 4 numbers")
                row.append(
                    {
                        "cls": cls,
                        "xyxy": [float(v) for v in xyxy],
                        "conf": float(box.get("conf", 0.0)),
                    }
                )
            cleaned[str(img)] = row
        out[str(model)] = cleaned
    return out


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_report(
    results: Mapping[str, Mapping[str, Any]],
    *,
    iou_thr: float,
    conf: float,
) -> str:
    lines = [
        "GVD detector eval: vehicle and pedestrian precision/recall.",
        "Detection quality on GVD data and GPU VRAM use are UNPROVEN.",
        "Published COCO val mAP 0.5:0.95 (not this dataset): "
        "YOLOX-Nano 25.8 at 416, YOLOX-Tiny 32.8 at 416, YOLOv8n 37.3 at 640.",
        f"This script reports single-threshold P/R at iou={iou_thr} conf={conf}, not COCO mAP.",
        "YOLOX-Nano feeds tracks, CIPV and AEB in the live runtime.",
        "",
        f"{'model':<12} {'class':<12} {'P':>6} {'R':>6} {'tp':>4} {'fp':>4} {'fn':>4}  status",
    ]
    for name, block in results.items():
        status = str(block.get("status", "OK"))
        if status != "OK":
            reason = block.get("reason", "")
            lines.append(f"{name:<12} {'-':<12} {'-':>6} {'-':>6} {'-':>4} {'-':>4} {'-':>4}  {status} {reason}".rstrip())
            continue
        metrics = block["metrics"]
        for cls in SCORED_CLASSES:
            m = metrics[cls]
            lines.append(
                f"{name:<12} {cls:<12} {_fmt(m['precision']):>6} {_fmt(m['recall']):>6} "
                f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}  OK"
            )
        pub = PUBLISHED_MAP.get(name)
        if pub:
            lines.append(f"  published COCO mAP (not this run): {pub}")
    return "\n".join(lines) + "\n"


def _run_onnx(
    kind: str,
    path: Path,
    images: Mapping[str, Path],
    *,
    conf: float,
) -> dict[str, list[dict[str, Any]]]:
    from python.perception.detect import OnnxYoloDetector, OnnxYoloxDetector

    det: Any
    if kind == "yolox":
        det = OnnxYoloxDetector(path, conf=conf)
    elif kind == "yolov8":
        det = OnnxYoloDetector(path, conf=conf)
    else:
        raise ValueError(f"unknown detector kind: {kind}")
    import cv2

    out: dict[str, list[dict[str, Any]]] = {}
    for name, img_path in images.items():
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {img_path}")
        boxes = []
        for d in det.detect(bgr):
            cls = normalize_cls(d.cls)
            if cls is None:
                continue
            boxes.append({"cls": cls, "xyxy": [float(v) for v in d.xyxy], "conf": float(d.conf)})
        out[name] = boxes
    return out


def resolve_images(labels: Mapping[str, Any], images_dir: Path | None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in labels:
        p = Path(name)
        if p.is_file():
            found[name] = p
            continue
        if images_dir is not None:
            cand = images_dir / name
            if cand.is_file():
                found[name] = cand
                continue
        missing.append(name)
    if missing:
        raise FileNotFoundError(
            "missing image files for labeled frames: " + ", ".join(missing[:8])
            + ("…" if len(missing) > 8 else "")
        )
    return found


def evaluate(
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    models: Sequence[tuple[str, Path | None, str]],
    *,
    images: Mapping[str, Path] | None = None,
    preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    iou_thr: float = DEFAULT_IOU,
    conf: float = DEFAULT_CONF,
) -> dict[str, dict[str, Any]]:
    """Score each model. ``preds`` short-circuits ONNX loads (tests / saved output)."""
    results: dict[str, dict[str, Any]] = {}
    for name, path, kind in models:
        if preds is not None and name in preds:
            metrics = per_class_metrics(preds[name], labels, iou_thr=iou_thr)
            results[name] = {"status": "OK", "metrics": metrics}
            continue
        if path is None or not path.is_file():
            results[name] = {
                "status": "SKIPPED",
                "reason": "weights not present (not shipped)" if name != "yolox-nano" else "weights not present",
            }
            continue
        if images is None:
            results[name] = {"status": "SKIPPED", "reason": "no images to run"}
            continue
        try:
            pred_map = _run_onnx(kind, path, images, conf=conf)
        except Exception as e:
            results[name] = {"status": "ERROR", "reason": str(e)}
            continue
        metrics = per_class_metrics(pred_map, labels, iou_thr=iou_thr)
        results[name] = {"status": "OK", "metrics": metrics}
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=None, help="Directory of labeled images (or omit when --preds-json is set).")
    ap.add_argument("--labels", type=Path, required=True, help="JSONL labels (vehicle / pedestrian xyxy).")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold (default 0.5).")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Score cutoff (default 0.35, the live detector default).")
    ap.add_argument("--yolox-nano", type=Path, default=ROOT / "models" / "yolox_nano.onnx")
    ap.add_argument(
        "--yolox-tiny",
        type=Path,
        default=ROOT / "models" / "yolox_tiny.onnx",
        help="Optional local YOLOX-Tiny ONNX (Apache-2.0, not shipped).",
    )
    ap.add_argument(
        "--yolov8n",
        type=Path,
        default=ROOT / "models" / "yolov8n.onnx",
        help="Optional user-supplied YOLOv8n ONNX (AGPL, not shipped).",
    )
    ap.add_argument(
        "--preds-json",
        type=Path,
        default=None,
        help="Use precomputed boxes instead of running ONNX (tests / saved detections).",
    )
    ap.add_argument("--json", type=Path, default=None, help="Also write the result as JSON (create-only).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = load_labels(args.labels)
    except (OSError, ValueError) as e:
        print(f"[GVD] eval_detect: failed to read labels: {e}", file=sys.stderr)
        return 2
    if not labels:
        print("[GVD] eval_detect: labels file has no scored vehicle/pedestrian boxes", file=sys.stderr)
        return 2

    preds = None
    images = None
    if args.preds_json is not None:
        try:
            preds = load_preds_json(args.preds_json)
        except (OSError, ValueError) as e:
            print(f"[GVD] eval_detect: failed to read --preds-json: {e}", file=sys.stderr)
            return 2
    else:
        try:
            images = resolve_images(labels, args.images)
        except FileNotFoundError as e:
            print(f"[GVD] eval_detect: {e}", file=sys.stderr)
            return 2

    models = (
        ("yolox-nano", args.yolox_nano, "yolox"),
        ("yolox-tiny", args.yolox_tiny, "yolox"),
        ("yolov8n", args.yolov8n, "yolov8"),
    )
    results = evaluate(
        labels,
        models,
        images=images,
        preds=preds,
        iou_thr=args.iou,
        conf=args.conf,
    )
    report = format_report(results, iou_thr=args.iou, conf=args.conf)
    sys.stdout.write(report)
    if args.json is not None:
        if args.json.exists():
            print(f"[GVD] eval_detect: --json path already exists: {args.json}", file=sys.stderr)
            return 2
        payload = {
            "iou": args.iou,
            "conf": args.conf,
            "unproven": True,
            "published_coco_map": PUBLISHED_MAP,
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
