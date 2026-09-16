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


def _yolox_letterbox(bgr: np.ndarray, in_h: int, in_w: int) -> tuple[np.ndarray, float]:
    """Official YOLOX preproc: BGR, pad 114, no /255, CHW float32. Returns (blob, ratio)."""
    import cv2

    h0, w0 = bgr.shape[:2]
    ratio = min(in_h / max(1, h0), in_w / max(1, w0))
    new_w, new_h = int(w0 * ratio), int(h0 * ratio)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.ones((in_h, in_w, 3), dtype=np.uint8) * 114
    padded[:new_h, :new_w] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return chw[None, ...], float(ratio)


def _yolox_decode_raw(pred: np.ndarray, in_h: int, in_w: int, *, p6: bool = False) -> np.ndarray:
    """Official YOLOX grid/stride decode for Megvii ONNX with decode_in_inference=False.

    Expected tensor: ``[1, N, 85]`` (or ``[N, 85]``). At 640x640, N=8400
    (80x80 + 40x40 + 20x20). Channels: raw (cx, cy, log-w, log-h), objectness,
    80 COCO class scores. Not YOLOv8's ``[1, 84, N]`` xywh+cls (no objectness).
    """
    arr = np.array(pred, dtype=np.float32, copy=True)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return arr
    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]
    grids = []
    expanded = []
    for stride in strides:
        hsize, wsize = in_h // stride, in_w // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))
    grids_a = np.concatenate(grids, 1).astype(np.float32)
    strides_a = np.concatenate(expanded, 1)
    if arr.shape[1] != grids_a.shape[1]:
        return arr
    arr[..., :2] = (arr[..., :2] + grids_a) * strides_a
    arr[..., 2:4] = np.exp(arr[..., 2:4]) * strides_a
    return arr


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Single-class NMS (same maths as official YOLOX demo_utils.nms)."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


class OnnxYoloxDetector(Detector):
    """Optional Apache YOLOX-s ONNX. Same Detection / COCO map as OnnxYoloDetector.

    Official Megvii 0.1.1rc0 ``yolox_s.onnx`` (Apache-2.0): one output
    ``[1, 8400, 85]`` with decode_in_inference=False. Channels = raw
    (cx, cy, log-w, log-h) + objectness + 80 COCO. Input ``[1, 3, 640, 640]``
    BGR float32 0–255, letterbox pad 114. Score = objectness * class, then
    cxcywh→xyxy / letterbox ratio. If N does not match the 8/16/32 grid, treat
    the first four channels as already-decoded pixel cxcywh (later official
    ``--decode_in_inference`` exports).
    """

    name = "yolox-s-onnx"

    def __init__(self, onnx_path: Path, conf: float = 0.35, nms_thr: float = 0.45) -> None:
        import onnxruntime as ort  # type: ignore

        self.path = Path(onnx_path)
        self.name = "yolox-s-onnx"
        self.conf = conf
        self.nms_thr = nms_thr
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        avail = set(ort.get_available_providers())
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.in_h = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else 640
        self.in_w = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) else 640

    def detect(self, bgr: np.ndarray | None) -> list[Detection]:
        if bgr is None or bgr.size == 0:
            return []
        h0, w0 = bgr.shape[:2]
        blob, ratio = _yolox_letterbox(bgr, self.in_h, self.in_w)
        outs = self.session.run(None, {self.input_name: blob})
        boxes_xyxy, cls_ids, confs = self._decode_outputs(outs, ratio)
        dets: list[Detection] = []
        for xyxy, cls_id, score in zip(boxes_xyxy, cls_ids, confs):
            cls = coco_class_name(int(cls_id))
            if cls is None:
                continue
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            ex, ey, yaw = project_box_to_ego((x1, y1, x2, y2), w0, h0)
            dets.append(Detection(cls=cls, conf=float(score), xyxy=(x1, y1, x2, y2), x=ex, y=ey, yaw=yaw))
        return dets

    def _decode_outputs(self, outs: list[Any], ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))
        if not outs:
            return empty
        arr = np.array(outs[0])
        # Official: [1, 8400, 85]. Some graphs emit [1, 85, N].
        if arr.ndim == 3 and arr.shape[-1] != 85 and arr.shape[1] == 85:
            arr = np.transpose(arr, (0, 2, 1))
        if arr.ndim == 2 and arr.shape[-1] >= 6:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[-1] < 6:
            return empty
        n = int(arr.shape[1])
        expect = sum((self.in_h // s) * (self.in_w // s) for s in (8, 16, 32))
        expect_p6 = expect + (self.in_h // 64) * (self.in_w // 64)
        if n == expect:
            arr = _yolox_decode_raw(arr, self.in_h, self.in_w, p6=False)
        elif n == expect_p6:
            arr = _yolox_decode_raw(arr, self.in_h, self.in_w, p6=True)
        rows = arr[0]
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        xyxy = np.stack((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), axis=1)
        r = ratio if ratio > 1e-6 else 1.0
        xyxy = xyxy / r
        scores = rows[:, 4:5] * rows[:, 5:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]
        mask = confs >= self.conf
        xyxy, confs, cls_ids = xyxy[mask], confs[mask], cls_ids[mask]
        if xyxy.shape[0] == 0:
            return empty
        keep: list[int] = []
        for cid in np.unique(cls_ids):
            idx = np.where(cls_ids == cid)[0]
            for k in _nms_xyxy(xyxy[idx], confs[idx], self.nms_thr):
                keep.append(int(idx[k]))
        if not keep:
            return empty
        keep_a = np.array(keep, dtype=np.int32)
        return xyxy[keep_a], cls_ids[keep_a], confs[keep_a]


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
