"""Object detect on cam_main — YOLOv8n ONNX if present, else synthetic/empty."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "models" / "yolov8n.onnx"
COCO_VEHICLE = {2, 3, 5, 7}  # car, motorcycle, bus, truck
COCO_PERSON = {0}
COCO_BIKE = {1}
COCO_TRAFFIC_LIGHT = {9}
COCO_STOP_SIGN = {11}
COCO_POLE = {10, 12}  # fire hydrant, parking meter — the pole-like street furniture COCO knows
# Road furniture: detected like anything else, but never tracked or offered to CIPV/AEB —
# a stop sign is not a lead vehicle. The pipeline routes these to state["signs"].
STATIC_CLASSES = ("traffic_light", "stop_sign", "pole")


def coco_class_name(cls_id: int) -> str | None:
    if cls_id in COCO_VEHICLE:
        return "vehicle"
    if cls_id in COCO_PERSON:
        return "pedestrian"
    if cls_id in COCO_BIKE:
        return "bike"
    if cls_id in COCO_TRAFFIC_LIGHT:
        return "traffic_light"
    if cls_id in COCO_STOP_SIGN:
        return "stop_sign"
    if cls_id in COCO_POLE:
        return "pole"
    return None


@dataclass
class Detection:
    cls: str
    conf: float
    # axis-aligned box in image px: x1,y1,x2,y2
    xyxy: tuple[float, float, float, float]
    # ego-frame guess filled by project_to_ego (x right, y forward)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 1.57


def project_box_to_ego(xyxy: tuple[float, float, float, float], img_w: int, img_h: int) -> tuple[float, float, float]:
    """Crude pinhole: image bottom-center ≈ near ego, up ≈ forward. Toy only."""
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = max(1.0, y2 - y1)
    # lateral: normalized from center
    x = ((cx / max(1, img_w)) - 0.5) * 8.0
    # distance from box height (bigger = closer)
    y = max(4.0, 40.0 * (1.0 - (bh / max(1, img_h))) + 6.0 * (1.0 - cy / max(1, img_h)))
    return x, y, 1.57


class Detector:
    name = "base"

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        raise NotImplementedError


class EmptyDetector(Detector):
    name = "empty"

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        return []


class SyntheticDetector(Detector):
    """Offline/smoke: stable fake leads so pipeline + CIPV can be tested without weights."""

    name = "synthetic"

    def __init__(self, t0: float | None = None) -> None:
        self.t0 = t0 if t0 is not None else time.time()

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        t = time.time() - self.t0
        # lead car ahead, slight weave
        lead_x = 0.4 * np.sin(t * 0.3)
        return [
            Detection(cls="vehicle", conf=0.9, xyxy=(280, 200, 360, 320), x=float(lead_x), y=18.0, yaw=1.57),
            Detection(cls="vehicle", conf=0.7, xyxy=(420, 180, 500, 280), x=3.2, y=26.0, yaw=1.55),
            Detection(cls="pedestrian", conf=0.6, xyxy=(500, 240, 540, 340), x=5.5, y=12.0, yaw=3.1),
            # Road furniture, same fake-but-stable contract as the cars above (detector=synthetic).
            Detection(cls="stop_sign", conf=0.5, xyxy=(120, 150, 160, 190), x=-5.6, y=22.0, yaw=1.57),
            Detection(cls="traffic_light", conf=0.5, xyxy=(300, 60, 330, 120), x=1.2, y=34.0, yaw=1.57),
        ]


class OnnxYoloDetector(Detector):
    name = "yolov8n-onnx"

    def __init__(self, onnx_path: Path = DEFAULT_ONNX, conf: float = 0.35) -> None:
        import onnxruntime as ort  # type: ignore

        self.path = Path(onnx_path)
        self.name = f"{self.path.stem}-onnx"
        self.conf = conf
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        avail = set(ort.get_available_providers())
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.in_h = int(shape[2]) if isinstance(shape[2], int) else 640
        self.in_w = int(shape[3]) if isinstance(shape[3], int) else 640

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        if bgr is None or bgr.size == 0:
            return []
        import cv2

        h0, w0 = bgr.shape[:2]
        img = cv2.resize(bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = self.session.run(None, {self.input_name: blob})[0]
        # YOLOv8 ONNX often [1, 84, N] or [1, N, 84]
        arr = np.array(out)
        if arr.ndim == 3 and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 1))
        dets: list[Detection] = []
        if arr.ndim != 3:
            return dets
        rows = arr[0]
        for row in rows:
            if row.shape[0] < 6:
                continue
            # xywh + class scores
            xywh = row[:4]
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])
            if score < self.conf:
                continue
            cls = coco_class_name(cls_id)
            if cls is None:
                continue
            cx, cy, bw, bh = map(float, xywh)
            # scale to original
            sx, sy = w0 / self.in_w, h0 / self.in_h
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy
            ex, ey, yaw = project_box_to_ego((x1, y1, x2, y2), w0, h0)
            dets.append(Detection(cls=cls, conf=score, xyxy=(x1, y1, x2, y2), x=ex, y=ey, yaw=yaw))
        return dets


class UltraYoloDetector(Detector):
    """Ultralytics .pt path (optional). Same COCO class map as the ONNX detector."""

    name = "yolov8n-ultra"

    def __init__(self, pt_path: Path, conf: float = 0.35) -> None:
        from ultralytics import YOLO  # type: ignore

        self.path = Path(pt_path)
        self.name = f"{self.path.stem}-ultra"
        self.conf = conf
        self.model = YOLO(str(self.path))

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        if bgr is None or (hasattr(bgr, "size") and bgr.size == 0):
            return []
        res = self.model.predict(bgr, imgsz=640, conf=self.conf, verbose=False)[0]
        out: list[Detection] = []
        h0, w0 = bgr.shape[:2]
        for box in res.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cls = coco_class_name(cls_id)
            if cls is None:
                continue
            ex, ey, yaw = project_box_to_ego((x1, y1, x2, y2), w0, h0)
            out.append(Detection(cls=cls, conf=score, xyxy=(x1, y1, x2, y2), x=ex, y=ey, yaw=yaw))
        return out


def make_detector(*, allow_synthetic: bool = True) -> tuple[Detector, list[str]]:
    """Return detector + missing keys. Prefer ONNX; Ultralytics pt optional; else synthetic/empty."""
    missing: list[str] = []
    if DEFAULT_ONNX.is_file():
        try:
            return OnnxYoloDetector(DEFAULT_ONNX), missing
        except Exception as e:
            missing.append(f"onnx_load:{e}")
    pt = ROOT / "models" / "yolov8n.pt"
    if pt.is_file():
        try:
            return UltraYoloDetector(pt), missing
        except Exception as e:
            missing.append(f"ultra_load:{e}")
    missing.append("yolo_weights")
    if allow_synthetic:
        return SyntheticDetector(), missing
    return EmptyDetector(), missing
